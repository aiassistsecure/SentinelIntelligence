"""Gaia queue-drain: admission without execution, the
drainer, and batched perception (SI-50 §§4.2, 5.3, 7, 7.1,
7.2, 7.3, 8, 10, 12.2, 12.4, 13, 16 Q-1/Q-2, 17; slices
SI-51 and SI-52).

Q-1 builds the queue core of SI-50 on the journal SI already has.
The journal *is* the queue (§4): an intent is enqueued at the
moment its ADMIT verdict is journaled, claim and outcome are
journal entries, and queue state is a fold of those entries
(:mod:`si.gaia.queue`). This module is the drainer of §4.2 plus
the one admission helper the batched task surface needs:

- :func:`admit_tool_intent` gates and journals one tool intent
  exactly as :func:`si.gaia.executor.execute_tool_cycle` does —
  gate, classify, self-edit, provenance edges, result marking —
  except that on ADMIT it does **not** execute: the self-edit is
  left ``pending``, ``execution`` is None, and the intent sits
  ``queued`` in the fold for a drainer to claim. REJECT and
  ESCALATE journal their non-execution outcomes exactly as the
  serial cycle does (the B-1 shadow record when the report's
  execution label is the shadow label, otherwise the live
  non-execution record): a queue never claims them (§17.2).
- :class:`Drainer` folds, claims, and executes. It proposes
  nothing, gates nothing, and decides nothing about
  admissibility (§4.2): it reads verdicts from the fold and
  calls the existing executor through ``execute_journaled``,
  handing the executor the claim as its ticket (§5.3 — the
  intent sequence and claim sequence the claim names, so a
  claim/intent mismatch refuses at the executor and journals
  the refusal).

**Concurrency (Q-2, §8):** Q-1 fixed the drainer serial by
policy (:data:`Q1_CONCURRENCY` = 1, kept as the historical
record of that policy). Q-2 spends concurrency only where a
mistake is free: **unkeyed read-class intents** execute
concurrently in a bounded pool — policy
``queue_max_concurrent_reads``, default 4, hard cap 16,
malformed or sub-1 values resolving to 1 (serial, the most
restrictive reading). Everything else — every keyed intent
(writes key on their target, executions on the work root,
house tools on ``house``, reads with a proposer-declared key)
— still drains strictly serially: one claim, one execution,
one outcome. Claims are journaled serially in intent order
before a pool batch runs; outcome journaling stays serial
(the journal serializes its own appends — §8.3), so outcomes
append in completion order and the fold does not care. The
bound is a policy, **UNMEASURED** (Q-4): no throughput claim
is made for it anywhere in this slice.

``depends_on`` requirement semantics (Q-2, §7.3):
``outcome`` is satisfied when the dependency reaches
``executed``, ``refused-at-claim``, ``skipped``, ``rejected``,
or ``escalated`` — any recorded conclusion; the dependent
runs knowing the dependency *concluded*, not how. ``success``
is satisfied only by ``executed`` with outcome status
``completed`` (a ``timed-out`` execution's partial result
does not satisfy it — pinned here, where the spec leaves the
slice the choice). A queued intent whose ``success``
dependency concludes any other way is **skipped**: the
drainer journals a ``queue_skip`` naming the cause and the
dependency's outcome sequence, the intent never executes,
and the skip cascades to its own ``success`` dependents. A
dependency naming no folded intent still fails closed: the
dependent is blocked — never skipped on a guess, never run.

**Barrier is a sync point (Q-2, §10.2 point 2):** when an
intent declared ``barrier: true`` concludes during a drain —
executed, refused at claim, or skipped — the drain stops and
perception returns with sync reason ``barrier``; intents
after it stay queued for the next drain. Q-1's sync points
(queue empty, a stop rule, drain-budget exhaustion) are
unchanged.

**Claim-time residence refusals (Q-2, §12.2):** an intent
still ``queued`` longer than policy
``queue_max_residence_seconds`` (default 3600; malformed 0 —
the most-restrictive reading this module already uses for
its other bounds) after its journaled admission timestamp is
refused at claim with cause ``stale-admission`` when the
drainer reaches it, however it would re-validate; an
unparseable admission timestamp fails closed (never stale
on a guess), exactly like the claim-lease reading in
recovery.

**Pipelining (§9)** is real at the admission/drain boundary:
admission (gate + journal) shares no mutable state with a
drain in flight beyond the journal itself, whose appends
serialize, so batch N+1 can be admitted while batch N
executes — exercised under test with the journal's sequences
as the witness. The task loop's propose → admit → drain →
perceive cycle is unchanged; its next drain picks up
whatever was admitted meanwhile.

**Write/execute outcome-unknown recovery is NOT in Q-1**
(§13.3's write/execute arm is Q-3): :meth:`Drainer.recover`
re-claims only *read*-class orphaned claims; a write/execute
intent left ``claimed`` with no outcome is reported
``deferred`` and left claimed, untouched — never marked
``outcome-unknown`` here, never blindly re-executed.

Every performance quantity for the queue — drain throughput,
residence times, the benefit of any concurrency above 1 — is
**UNMEASURED** (Q-4, §16). Nothing in this module measures or
claims any of it.
"""
from __future__ import annotations

import inspect
import os
import secrets
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

from si.gaia.executor import Executor, record_non_execution_outcome
from si.gaia.queue import (
    TERMINAL_STATES,
    fold_queue,
    queue_summary,
    record_queue_claim,
    record_queue_recovery_reclaim,
    record_queue_refusal,
    record_queue_skip,
)
from si.gaia.registry import get_tool
from si.gaia.shadow import record_shadow_outcome
from si.gate import TOOL_SHADOW_EXECUTION

#: Q-1's concurrency, fixed at 1 by policy (SI-50 §16 Q-1): the
#: drainer was strictly serial. Kept as the historical record
#: of that policy; Q-2's bounded read pool (below) supersedes
#: it as the drainer's live concurrency rule.
Q1_CONCURRENCY = 1

#: Q-2's read-pool bound (SI-50 §8.2): the policy
#: ``queue_max_concurrent_reads`` defaults to 4 and is hard-
#: capped at 16 — a policy value above the cap resolves to the
#: cap; a malformed or sub-1 value resolves to 1 (serial, the
#: most restrictive reading). The number is UNMEASURED (Q-4).
Q2_DEFAULT_MAX_CONCURRENT_READS = 4
Q2_MAX_CONCURRENT_READS_CAP = 16

#: Q-2's claim-time residence bound (SI-50 §12.2): the policy
#: ``queue_max_residence_seconds`` defaults to 3600.
Q2_DEFAULT_MAX_RESIDENCE_SECONDS = 3600

#: Detail markers of an execution-integrity refusal (SI-50
#: §12.4 rule 4, extending SI-41 §9.2/§10.4.4's markers with the
#: §5.3 claim/intent mismatch): any of these in a refused
#: execution's detail freezes the drain immediately, task-wide.
_INTEGRITY_MARKERS = (
    "claim/intent mismatch",
    "TOCTOU",
    "changed between GATE and EXECUTE",
    "no longer validate against the registry schema",
)

#: Canonical journal-writer arguments -> the parameter names a
#: writer may carry them under. The queue writers
#: (:mod:`si.gaia.queue`) are called through :func:`_invoke_writer`
#: so this module binds to the *values* (intent sequence, edit id,
#: drainer identity, lease) and not to one spelling of a keyword.
_WRITER_ALIASES = {
    "intent_seq": ("intent_seq", "claimed_intent_seq",
                   "queue_intent_seq"),
    "edit_id": ("edit_id", "intent_id", "intent_edit_id"),
    "task_seq": ("task_seq",),
    "drainer_id": ("drainer_id", "claim_drainer", "claimer",
                   "claimed_by"),
    "lease_seconds": ("lease_seconds", "claim_lease_seconds",
                      "lease", "claim_lease"),
    "original_claim_seq": ("original_claim_seq", "claim_seq",
                           "prior_claim_seq", "previous_claim_seq",
                           "recovery_of", "original_seq",
                           "superseded_claim_seq"),
    "state": ("state", "intent_state"),
}


# -- small field / fold helpers ---------------------------------------------
# Queue states may be dataclasses or dicts (the fold's own shape
# is si.gaia.queue's); every read in this module goes through
# _field so the drainer never depends on one representation.

def _field(obj, *names, default=None):
    """The first non-None of ``names`` on a dict or an object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        for name in names:
            if name in obj and obj[name] is not None:
                return obj[name]
        return default
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _state_name(state) -> str | None:
    value = _field(state, "state", "status")
    return str(value) if value is not None else None


def _intent_seq(state):
    value = _field(state, "intent_seq", "seq")
    return value


def _edit_id(state):
    return _field(state, "edit_id", "intent_id")


def _task_seq_of(state):
    return _field(state, "task_seq")


def _fold_intents(fold) -> list:
    intents = _field(fold, "intents", default=None)
    if intents is None:
        return []
    if isinstance(intents, dict):
        return list(intents.values())
    return list(intents)


def _find_by_edit_id(fold, edit_id):
    for state in _fold_intents(fold):
        if _edit_id(state) == edit_id:
            return state
    return None


def _find_by_intent_seq(fold, intent_seq):
    for state in _fold_intents(fold):
        if _intent_seq(state) == intent_seq:
            return state
    return None


def _fold(journal, task_seq=None):
    """fold_queue(journal, task_seq=...), tolerant of a positional
    spelling of the fold's task argument."""
    try:
        return fold_queue(journal, task_seq=task_seq)
    except TypeError:
        return fold_queue(journal, task_seq)


def _is_terminal(state) -> bool:
    return _state_name(state) in TERMINAL_STATES


def _dep_seq(dep):
    """The journal sequence one ``depends_on`` entry names.

    Entries are an int (an ``outcome`` requirement, §7.3), or a
    dict/object carrying ``seq`` (the gate's normalised form).
    Anything else names nothing — None, and the caller fails
    closed."""
    if isinstance(dep, bool):
        return None
    if isinstance(dep, int):
        return dep
    return _field(dep, "seq", "intent_seq", "depends_on_seq")


def _dep_requirement(dep) -> str:
    """The requirement one ``depends_on`` entry declares (§7.3):
    ``success``, or ``outcome`` — the default, and the reading
    for anything else (malformed requirements are priced out at
    the gate, §6.2; the drainer never upgrades one to
    ``success`` on a guess)."""
    if isinstance(dep, dict):
        req = dep.get("requirement")
    else:
        req = _field(dep, "requirement")
    return "success" if req == "success" else "outcome"


#: Folded states that satisfy an ``outcome`` dependency (§7.3):
#: the dependency reached a recorded conclusion — it executed
#: (however the execution itself turned out), was refused at
#: claim, was skipped, or was never queued (rejected /
#: escalated at the gate, each with its journaled non-execution
#: outcome). ``outcome-unknown`` is deliberately absent: an
#: intent whose outcome is *unknown* has not concluded in any
#: sense a dependent may rely on; its dependents wait for the
#: Q-3 operator disposition.
_OUTCOME_DEP_STATES = frozenset({
    "executed", "refused-at-claim", "skipped", "rejected",
    "escalated",
})


def _dep_success(dep_state) -> bool:
    """True iff the dependency *succeeded* (§7.3): folded
    ``executed`` with outcome status ``completed``. A failed
    or timed-out execution concluded, but did not succeed."""
    return _state_name(dep_state) == "executed" \
        and _field(dep_state, "outcome_status") == "completed"


def _dep_satisfied(dep_state, requirement: str) -> bool:
    """True iff ``dep_state`` satisfies ``requirement`` (§7.3)."""
    if requirement == "success":
        return _dep_success(dep_state)
    return _state_name(dep_state) in _OUTCOME_DEP_STATES


def _parse_journal_ts(raw) -> datetime | None:
    """Parse a journal timestamp (ISO ``Z`` text, an epoch
    number, or a datetime) to an aware datetime, or None when
    unparseable — callers fail closed on None, never guessing
    an age (the §13.1 claim-lease discipline)."""
    if isinstance(raw, datetime):
        claimed_at = raw
    elif isinstance(raw, (int, float)) and not isinstance(raw, bool):
        try:
            claimed_at = datetime.fromtimestamp(
                float(raw), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    elif isinstance(raw, str):
        text = raw.strip()
        try:
            claimed_at = datetime.fromtimestamp(
                float(text), tz=timezone.utc)
        except ValueError:
            iso = text[:-1] + "+00:00" if text.endswith("Z") else text
            try:
                claimed_at = datetime.fromisoformat(iso)
            except ValueError:
                return None
    else:
        return None
    if claimed_at.tzinfo is None:
        claimed_at = claimed_at.replace(tzinfo=timezone.utc)
    return claimed_at


def _task_stop_in_force(fold, task_seq) -> bool:
    """True iff the fold says ``task_seq``'s leash stop is in force.

    Reads the fold's per-task record (``fold.tasks``) under the
    spellings such a record may carry; a task the fold does not
    know is not stopped."""
    tasks = _field(fold, "tasks", default=None)
    entry = None
    if isinstance(tasks, dict):
        entry = tasks.get(task_seq)
        if entry is None and task_seq is not None:
            entry = tasks.get(str(task_seq))
    elif isinstance(tasks, (list, tuple)):
        for item in tasks:
            if _field(item, "task_seq", "seq") == task_seq:
                entry = item
                break
    if entry is None:
        return False
    if isinstance(entry, bool):
        return entry
    for name in ("stop_in_force", "stopped", "is_stopped", "held"):
        value = _field(entry, name)
        if isinstance(value, bool):
            return value
    value = _field(entry, "state", "status", "stop_state")
    if isinstance(value, str):
        return value in ("held", "stopped", "stop-in-force",
                         "stop_in_force")
    stop_seq = _field(entry, "stop_seq", "stopped_seq",
                      "queue_stop_seq")
    if stop_seq is not None:
        resume_seq = _field(entry, "resume_seq", "resumed_seq",
                            "queue_resume_seq")
        return resume_seq is None or resume_seq < stop_seq
    stop_flag = _field(entry, "stop")
    return bool(stop_flag)


def _policy_nonneg_int(policy: dict, key: str, default: int) -> int:
    """One fail-closed policy integer (the si.gaia.policy
    precedent): absent -> the pinned default; a valid non-negative
    int -> itself; present but malformed -> 0, the most
    restrictive value."""
    if key not in (policy or {}):
        return default
    value = policy[key]
    if isinstance(value, bool) or not isinstance(value, int) \
            or value < 0:
        return 0
    return value


# -- journal writer invocation ------------------------------------------------

def _invoke_writer(func, journal, values: dict):
    """Call a queue writer with the arguments it declares.

    ``values`` is keyed by canonical name (see
    :data:`_WRITER_ALIASES`); the writer's own parameter names —
    read off its signature — select which values it receives, so
    the claim's journaled facts (intent seq, edit id, task seq,
    drainer, lease) are identical whatever the writer's keyword
    spellings are. ``journal`` is always the first positional
    argument, as in every house writer."""
    try:
        params = list(inspect.signature(func).parameters.values())
    except (TypeError, ValueError):
        params = []
    if params:
        params = params[1:]  # the journal argument
    kwargs = {}
    for param in params:
        if param.kind in (inspect.Parameter.VAR_KEYWORD,
                          inspect.Parameter.VAR_POSITIONAL):
            continue
        for canonical, aliases in _WRITER_ALIASES.items():
            if param.name in aliases \
                    and values.get(canonical) is not None:
                kwargs[param.name] = values[canonical]
                break
        else:
            if param.name in values \
                    and values[param.name] is not None:
                kwargs[param.name] = values[param.name]
    if kwargs or not params:
        return func(journal, **kwargs)
    # No declared parameter matched a canonical name: fall back to
    # the positional orders the house writers use, most specific
    # first. A TypeError from a wrong arity tries the next order;
    # the last order's error propagates.
    orders = []
    if "original_claim_seq" in values:
        orders = [
            (values.get("original_claim_seq"),
             values.get("intent_seq"), values.get("drainer_id"),
             values.get("lease_seconds")),
            (values.get("original_claim_seq"),
             values.get("edit_id"), values.get("drainer_id")),
        ]
    else:
        orders = [
            (values.get("intent_seq"), values.get("drainer_id"),
             values.get("lease_seconds")),
            (values.get("edit_id"), values.get("intent_seq"),
             values.get("drainer_id"), values.get("lease_seconds")),
        ]
    last_err = None
    for order in orders:
        try:
            return func(journal, *order)
        except TypeError as err:  # arity spelling; try the next
            last_err = err
    raise last_err


def _normalise_claim(raw, journal, state, drainer_id,
                     recovery_of=None) -> dict:
    """A claim dict with at least ``seq`` / ``edit_id`` /
    ``intent_seq`` / ``drainer_id``, whatever the writer returned
    (a dict, a bare sequence, an object, or None — in the last
    case the claim sequence is read back from a fresh fold, the
    journal being the only truth about it)."""
    claim: dict = {}
    if isinstance(raw, dict):
        claim = dict(raw)
    elif isinstance(raw, int) and not isinstance(raw, bool):
        claim = {"seq": raw}
    elif raw is not None:
        to_dict = getattr(raw, "to_dict", None)
        if callable(to_dict):
            try:
                claim = dict(to_dict())
            except Exception:
                claim = {}
        if not claim:
            seq = _field(raw, "seq", "claim_seq")
            if seq is not None:
                claim = {"seq": seq}
    if claim.get("seq") is None:
        for key in ("claim_seq", "recovery_seq", "new_claim_seq"):
            if claim.get(key) is not None:
                claim["seq"] = claim[key]
                break
    if claim.get("seq") is None and journal is not None:
        try:
            folded = _find_by_edit_id(
                _fold(journal, _task_seq_of(state)), _edit_id(state))
        except Exception:
            folded = None
        if folded is not None:
            claim["seq"] = _field(folded, "claim_seq")
            if recovery_of is None:
                recovery_of = _field(folded, "recovery_of")
    claim.setdefault("edit_id", _edit_id(state))
    claim.setdefault("intent_seq", _intent_seq(state))
    claim.setdefault("drainer_id", drainer_id)
    if recovery_of is not None:
        claim.setdefault("recovery_of", recovery_of)
    return claim


# -- admission without execution ------------------------------------------------

def admit_tool_intent(journal, gate, ctx, intent, batch_seq=None) -> dict:
    """Gate and journal one tool intent; on ADMIT, queue it —
    do not execute it (SI-50 §4.1, §6.1; SI-51 Q-1).

    Mirrors :func:`si.gaia.executor.execute_tool_cycle`'s
    journaling exactly — ``gate.evaluate``, ``classify``,
    ``record_self_edit`` with the full gate report, the provenance
    edges, ``set_edit_result`` — with the single Q-1 difference on
    the ADMIT branch: the result is ``pending``, no executor is
    constructed or called, and the returned ``execution`` is
    None. The intent is ``queued`` from this moment: admission
    *is* enqueueing (§4.1). On REJECT/ESCALATE the non-execution
    outcome is journaled exactly as the serial cycle journals it
    (shadow record under the shadow label, live non-execution
    record otherwise).

    When ``batch_seq`` is given it is set on the gate context as
    ``tool_queue_batch_seq`` before evaluation, so the gate
    prices it into the report's queue block; a context that
    cannot carry the attribute is tolerated (the attribute is
    bookkeeping for the gate, not a precondition of admission).

    Returns {"verdict", "edit_id", "gate_report",
    "effective_class", "execution"} — the execute_tool_cycle
    shape, with ``execution`` None exactly on ADMIT.
    """
    if batch_seq is not None:
        try:
            ctx.tool_queue_batch_seq = batch_seq
        except Exception:
            # Plain contexts (an object that forbids attribute
            # assignment, None) must not crash admission: the
            # batch seq is gate bookkeeping, priced when the
            # context can carry it.
            pass
    report = gate.evaluate(intent, None, ctx)
    info = gate.classify_tool_intent(intent, ctx) \
        if hasattr(gate, "classify_tool_intent") else {}
    effective = info.get("effective_class") \
        if isinstance(info, dict) else None
    edit_id = journal.record_self_edit(
        intent.kind, intent.to_dict(), intent.reason,
        report.verdict, report.to_dict())
    if intent.parents:
        journal.record_provenance(
            child=edit_id, parents=list(intent.parents),
            relation="learned_from", note=(intent.reason or "")[:200])
    if intent.source_seqs:
        journal.record_provenance(
            child=edit_id, parents=list(intent.source_seqs),
            relation="learned_from", note="source journal sequences")

    if report.verdict == "ADMIT":
        journal.set_edit_result(edit_id, "pending")
        return {"verdict": report.verdict, "edit_id": edit_id,
                "gate_report": report.to_dict(), "execution": None,
                "effective_class": effective}

    journal.set_edit_result(
        edit_id, "rejected" if report.verdict == "REJECT" else "escalated")
    if report.execution == TOOL_SHADOW_EXECUTION:
        execution = record_shadow_outcome(
            journal, intent, edit_id, report.verdict,
            effective_class=effective)
    else:
        execution = record_non_execution_outcome(
            journal, intent, edit_id, report.verdict,
            report.execution or "not-evaluated",
            effective_class=effective)
    return {"verdict": report.verdict, "edit_id": edit_id,
            "gate_report": report.to_dict(), "execution": execution,
            "effective_class": effective}


# -- batched perception -----------------------------------------------------------

def build_observation_set(journal, fold, task_seq, outcome_seqs,
                          sync_reason) -> dict:
    """One batched observation set for a task (SI-50 §10).

    One ``tool_result`` observation per outcome sequence in
    ``outcome_seqs``, in ascending outcome-sequence order — never
    proposal order, never clock order (§8.3) — each carrying its
    §4.5 provenance (task, intent, claim, and outcome sequences,
    edit id, tool, verdict) and the folded execution record, plus
    the §10.3 queue summary for the task. ``journal`` is accepted
    for the caller's symmetry with the journal-derived machinery;
    the set itself is built from the fold, which is the journal.
    """
    by_outcome = {}
    for state in _fold_intents(fold):
        if task_seq is not None \
                and _task_seq_of(state) is not None \
                and _task_seq_of(state) != task_seq:
            continue
        outcome_seq = _field(state, "outcome_seq")
        if outcome_seq is not None:
            by_outcome[outcome_seq] = state
    observations = []
    for outcome_seq in sorted(set(outcome_seqs or [])):
        state = by_outcome.get(outcome_seq)
        if state is None:
            continue
        tool = _field(state, "tool") or ""
        observations.append({
            "kind": "tool_result",
            "source": f"tool:{tool}",
            "task_seq": _task_seq_of(state)
            if _task_seq_of(state) is not None else task_seq,
            "intent_seq": _intent_seq(state),
            "claim_seq": _field(state, "claim_seq"),
            "outcome_seq": outcome_seq,
            "edit_id": _edit_id(state),
            "tool": tool,
            "verdict": _field(state, "verdict", "gate_verdict"),
            "execution": _field(state, "execution"),
        })
    return {
        "kind": "queue_perception",
        "task_seq": task_seq,
        "sync_point": sync_reason,
        "observations": observations,
        "queue_summary": queue_summary(fold, task_seq),
    }


# -- the drainer --------------------------------------------------------------------

class Drainer:
    """The drainer (SI-50 §4.2): fold, claim, execute — serially
    for everything keyed or state-changing, with Q-2's bounded
    concurrent pool for unkeyed read-class intents (§8).

    Constructed by the runtime with the bundle's journal, its
    verified manifest, the bundle policy (raw, as the gate takes
    it — the queue's own policy keys are resolved fail-closed
    from it here), and the task work root. ``drainer_id`` is the
    per-process identity journaled on every claim (§5.3); the
    default names the process and 8 random hex characters.
    ``executor`` defaults to the existing SI-43/SI-44
    :class:`~si.gaia.executor.Executor` over the same journal.
    """

    def __init__(self, journal, bundle_dir, manifest, policy,
                 work_root, drainer_id=None, executor=None):
        self.journal = journal
        self.bundle_dir = bundle_dir
        self.manifest = manifest
        self.policy = dict(policy or {})
        self.work_root = work_root
        self.drainer_id = drainer_id or \
            f"drainer-{os.getpid()}-{secrets.token_hex(4)}"
        self.executor = executor or Executor(
            bundle_dir, manifest, journal, self.policy, work_root)

    # -- policy-derived bounds -----------------------------------------
    def _claim_lease_seconds(self) -> int:
        return _policy_nonneg_int(
            self.policy, "queue_claim_lease_seconds", 300)

    def _drain_budget_seconds(self) -> int:
        return _policy_nonneg_int(
            self.policy, "queue_drain_budget_seconds", 1920)

    def max_concurrent_reads(self) -> int:
        """The read-pool bound (§8.2): policy
        ``queue_max_concurrent_reads`` — absent → the default
        (:data:`Q2_DEFAULT_MAX_CONCURRENT_READS`); a valid int
        ≥ 1 → itself, capped at
        (:data:`Q2_MAX_CONCURRENT_READS_CAP`); anything else
        (malformed, bool, sub-1) → 1, serial — the most
        restrictive reading. UNMEASURED (Q-4)."""
        if "queue_max_concurrent_reads" not in self.policy:
            return Q2_DEFAULT_MAX_CONCURRENT_READS
        value = self.policy.get("queue_max_concurrent_reads")
        if isinstance(value, bool) or not isinstance(value, int) \
                or value < 1:
            return 1
        return min(value, Q2_MAX_CONCURRENT_READS_CAP)

    def _max_residence_seconds(self) -> int:
        return _policy_nonneg_int(
            self.policy, "queue_max_residence_seconds",
            Q2_DEFAULT_MAX_RESIDENCE_SECONDS)

    def _residence_expired(self, state, now: datetime) -> bool:
        """True iff ``state`` has been queued past the residence
        bound (§12.2), measured from its journaled admission
        timestamp (the fold's ``intent_ts``). An unparseable
        timestamp is treated as *unexpired* — fail closed: an
        intent is never refused as stale on a guessed age."""
        admitted = _parse_journal_ts(
            _field(state, "intent_ts", "admitted_ts", "ts"))
        if admitted is None:
            return False
        return admitted + timedelta(
            seconds=float(self._max_residence_seconds())) <= now

    # -- claimability (§7.1) ---------------------------------------------
    def claimable(self, state, fold) -> bool:
        """True iff ``state`` may be claimed now (§7.1, Q-1 form).

        All of: the folded state is ``queued``; no operator stop
        is in force for its task (§11 — the fold's per-task
        record); ordering-key FIFO (§7.2) — no earlier-sequence
        intent in the same task shares its non-None ordering key
        in a non-terminal state; and every ``depends_on``
        dependency has concluded.

        ``depends_on`` satisfaction is §7.3's (Q-2): each
        dependency must satisfy its declared requirement —
        ``outcome`` by any recorded conclusion, ``success``
        only by a completed execution (see the module
        docstring). A ``success`` dependency that concluded
        any other way does not make the intent claimable: the
        drainer skips it instead (:meth:`_propagate_skips`).
        A dependency naming no folded intent fails closed:
        not claimable."""
        if _state_name(state) != "queued":
            return False
        task_seq = _task_seq_of(state)
        if task_seq is not None and _task_stop_in_force(fold, task_seq):
            return False
        ordering_key = _field(state, "ordering_key")
        if ordering_key is not None:
            my_seq = _intent_seq(state)
            for other in _fold_intents(fold):
                if other is state or _edit_id(other) == _edit_id(state):
                    continue
                if _task_seq_of(other) != task_seq:
                    continue
                if _field(other, "ordering_key") != ordering_key:
                    continue
                other_seq = _intent_seq(other)
                if my_seq is not None and other_seq is not None \
                        and other_seq < my_seq \
                        and not _is_terminal(other):
                    return False
        depends_on = _field(state, "depends_on", default=None) or []
        for dep in depends_on:
            dep_seq = _dep_seq(dep)
            if dep_seq is None:
                return False
            dep_state = _find_by_intent_seq(fold, dep_seq)
            if dep_state is None or not _dep_satisfied(
                    dep_state, _dep_requirement(dep)):
                return False
        return True

    # -- dependency skips (§7.3) -----------------------------------------
    def _propagate_skips(self, task_seq=None) -> list:
        """Journal ``queue_skip`` entries for every queued
        intent whose ``success`` dependency has concluded any
        other way (§7.3), cascading until no queued intent has
        an unmet-but-concluded ``success`` dependency.

        Each skip names its cause and the dependency's outcome
        sequence; the skipped intent is never claimed and never
        executed. A dependency naming no folded intent never
        causes a skip (fail closed — the dependent stays
        blocked, §7.3's construction makes the name a journaled
        fact or a gate error, not a drainer guess). Returns the
        skip entries journaled by this call, in the order
        journaled (dependent intent-sequence order per pass)."""
        skips: list = []
        while True:
            fold = _fold(self.journal, task_seq)
            target = None
            states = sorted(
                (s for s in _fold_intents(fold)
                 if task_seq is None or _task_seq_of(s) is None
                 or _task_seq_of(s) == task_seq),
                key=lambda s: (_intent_seq(s) is None,
                               _intent_seq(s) or 0))
            for state in states:
                if _state_name(state) != "queued":
                    continue
                for dep in _field(state, "depends_on",
                                  default=None) or []:
                    if _dep_requirement(dep) != "success":
                        continue
                    dep_seq = _dep_seq(dep)
                    if dep_seq is None:
                        continue
                    dep_state = _find_by_intent_seq(fold, dep_seq)
                    if dep_state is None \
                            or not _is_terminal(dep_state) \
                            or _dep_success(dep_state):
                        continue
                    target = (state, dep_seq, dep_state)
                    break
                if target is not None:
                    break
            if target is None:
                return skips
            state, dep_seq, dep_state = target
            cause = (
                f"depends_on success requirement unmet: dependency "
                f"intent seq {dep_seq} concluded as "
                f"{_state_name(dep_state)} (outcome status "
                f"{_field(dep_state, 'outcome_status')!r}, outcome "
                f"seq {_field(dep_state, 'outcome_seq')}) — "
                f"dependent skipped per SI-50 §7.3, never executed")
            raw = record_queue_skip(
                self.journal, _task_seq_of(state), _edit_id(state),
                _intent_seq(state), cause,
                dependency_seq=dep_seq,
                dependency_outcome_seq=_field(
                    dep_state, "outcome_seq"))
            skips.append(raw if isinstance(raw, dict) else {
                "edit_id": _edit_id(state),
                "intent_seq": _intent_seq(state), "cause": cause})

    # -- claim-time residence refusals (§12.2) ----------------------------
    def _refuse_stale(self, state) -> dict:
        """Journal the §12.2 stale-admission refusal for one
        queued intent the drainer has reached: refused at claim
        with cause ``stale-admission``, never executed under a
        stale price. Returns the refusal entry."""
        cause = (
            f"stale-admission: intent admitted at "
            f"{_field(state, 'intent_ts')!r} exceeded "
            f"queue_max_residence_seconds="
            f"{self._max_residence_seconds()} before claim "
            f"(SI-50 §12.2) — refused at claim, never executed "
            f"under a stale price; re-propose at fresh prices")
        raw = record_queue_refusal(
            self.journal, _task_seq_of(state), _edit_id(state),
            _intent_seq(state), cause)
        return raw if isinstance(raw, dict) else {
            "edit_id": _edit_id(state),
            "intent_seq": _intent_seq(state), "cause": cause}

    # -- read-pool eligibility (§8.1) ---------------------------------------
    @staticmethod
    def _pool_eligible(state) -> bool:
        """True iff ``state`` may execute in the read pool
        (§8.1): read-class (the folded, gate-priced effective
        class) with **no ordering key**. A read with a declared
        key is keyed work and drains serially like everything
        else (§7.2)."""
        return _field(state, "effective_class") == "read" \
            and _field(state, "ordering_key") is None

    # -- claiming (§5.3) ----------------------------------------------------
    def claim(self, state) -> dict:
        """Journal this drainer's claim on ``state`` (§5.3) and
        return the claim dict. The lease is the policy's
        ``queue_claim_lease_seconds`` (absent 300, malformed 0 —
        :func:`_policy_nonneg_int`)."""
        lease = self._claim_lease_seconds()
        raw = _invoke_writer(record_queue_claim, self.journal, {
            "intent_seq": _intent_seq(state),
            "edit_id": _edit_id(state),
            "task_seq": _task_seq_of(state),
            "drainer_id": self.drainer_id,
            "lease_seconds": lease,
            "effective_class": _field(state, "effective_class"),
            "ordering_key": _field(state, "ordering_key"),
            "state": state,
        })
        return _normalise_claim(raw, self.journal, state,
                                self.drainer_id)

    # -- recovery (§13, Q-1 arm) ----------------------------------------------
    @staticmethod
    def _lease_expired(state, now: datetime) -> bool:
        """True iff the state's claim lease has expired by ``now``.

        An unparseable claim timestamp is treated as *unexpired*
        — fail closed (§13.1): a claim whose age cannot be
        established is never treated as orphaned on a guess. A
        malformed lease likewise never expires."""
        raw_ts = _field(state, "claim_ts", "claimed_at", "claim_time",
                        "claim_timestamp")
        claimed_at = None
        if isinstance(raw_ts, datetime):
            claimed_at = raw_ts
        elif isinstance(raw_ts, (int, float)) \
                and not isinstance(raw_ts, bool):
            try:
                claimed_at = datetime.fromtimestamp(
                    float(raw_ts), tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                claimed_at = None
        elif isinstance(raw_ts, str):
            text = raw_ts.strip()
            try:
                claimed_at = datetime.fromtimestamp(
                    float(text), tz=timezone.utc)
            except ValueError:
                iso = text[:-1] + "+00:00" if text.endswith("Z") \
                    else text
                try:
                    claimed_at = datetime.fromisoformat(iso)
                except ValueError:
                    claimed_at = None
        if claimed_at is None:
            return False
        if claimed_at.tzinfo is None:
            claimed_at = claimed_at.replace(tzinfo=timezone.utc)
        lease = _field(state, "claim_lease_seconds", "lease_seconds",
                       "claim_lease")
        if isinstance(lease, bool) \
                or not isinstance(lease, (int, float)) or lease < 0:
            return False
        return claimed_at + timedelta(seconds=float(lease)) <= now

    def recover(self, task_seq=None) -> dict:
        """The Q-1 recovery fold (§13.1, §13.3 read arm).

        For every intent folded ``claimed`` with no outcome:

        - a claim by *another* drainer whose lease is unexpired
          (or unparseable — fail closed) is **foreign**: left
          alone, its claim sequence reported;
        - a claim whose lease has expired, or one of this
          drainer's own claims, is recovered by class: a
          **read**-class intent (the registry spec's
          ``consequence_class`` for the folded tool name) is
          recovery-re-claimed — a new claim journaled naming the
          original claim — and reported in ``reclaimed``; a
          write/execute-class intent is **deferred** (Q-3
          territory: Q-1 does not mark it outcome-unknown and
          does not re-execute it) and left claimed.

        Returns {"reclaimed": [claim, ...], "deferred":
        [intent_seq, ...], "foreign": [claim_seq, ...]}."""
        fold = _fold(self.journal, task_seq)
        now = datetime.now(timezone.utc)
        reclaimed: list = []
        deferred: list = []
        foreign: list = []
        for state in _fold_intents(fold):
            if task_seq is not None \
                    and _task_seq_of(state) is not None \
                    and _task_seq_of(state) != task_seq:
                continue
            if _state_name(state) != "claimed":
                continue
            if _field(state, "outcome_seq") is not None:
                continue
            claim_seq = _field(state, "claim_seq")
            claim_drainer = _field(state, "claim_drainer",
                                   "drainer_id")
            if claim_drainer != self.drainer_id \
                    and not self._lease_expired(state, now):
                foreign.append(claim_seq)
                continue
            spec = get_tool(_field(state, "tool") or "")
            if spec is not None and spec.consequence_class == "read":
                raw = _invoke_writer(
                    record_queue_recovery_reclaim, self.journal, {
                        "original_claim_seq": claim_seq,
                        "intent_seq": _intent_seq(state),
                        "edit_id": _edit_id(state),
                        "task_seq": _task_seq_of(state),
                        "drainer_id": self.drainer_id,
                        "lease_seconds": self._claim_lease_seconds(),
                        "effective_class": _field(
                            state, "effective_class"),
                        "ordering_key": _field(state, "ordering_key"),
                        "state": state,
                    })
                reclaimed.append(_normalise_claim(
                    raw, self.journal, state, self.drainer_id,
                    recovery_of=claim_seq))
            else:
                deferred.append(_intent_seq(state))
        return {"reclaimed": reclaimed, "deferred": deferred,
                "foreign": foreign}

    # -- execution -------------------------------------------------------------
    def execute_claim(self, state, claim) -> dict:
        """Execute one claimed intent through the executor (§5.3).

        The claim is the executor's ticket: the journaled intent
        sequence the claim names (``claim_intent_seq``) and the
        claim's own sequence (``claim_seq``) travel with the call,
        alongside the folded effective class for the executor's
        TOCTOU re-pricing. A claim/intent mismatch refuses at the
        executor and journals the refusal. An executor whose
        ``execute_journaled`` predates the claim parameters is
        called without them (signature-checked, never guessed)."""
        claim_seq = claim.get("seq") if isinstance(claim, dict) \
            else _field(claim, "seq", "claim_seq")
        kwargs = {}
        try:
            params = inspect.signature(
                self.executor.execute_journaled).parameters
            has_varkw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in params.values())
        except (TypeError, ValueError):
            params, has_varkw = {}, False
        if "claim_intent_seq" in params or has_varkw:
            kwargs["claim_intent_seq"] = _intent_seq(state)
        if "claim_seq" in params or has_varkw:
            kwargs["claim_seq"] = claim_seq
        return self.executor.execute_journaled(
            _edit_id(state),
            expected_effective_class=_field(state, "effective_class"),
            **kwargs)

    # -- perception (method form of the module function) -------------------------
    def build_observation_set(self, journal, fold, task_seq,
                              outcome_seqs, sync_reason) -> dict:
        return build_observation_set(
            journal, fold, task_seq, outcome_seqs, sync_reason)

    # -- the drain ----------------------------------------------------------------
    def drain_task(self, task_seq, budget_seconds=None,
                   max_intents=None) -> dict:
        """Drain one task's queue (§7.1, §8, §10, §12.4).

        Recovery runs first (§13). Then, re-folding after every
        step (the fold is the state; nothing is cached across an
        execution):

        - dependency skips propagate (§7.3): a queued intent
          whose ``success`` dependency concluded any other way
          is journaled ``skipped`` and never runs;
        - if the task's stop is in force, stop
          (``stop-in-force``);
        - a barrier intent that concluded during this drain
          flushes perception (``barrier``, §10.2 point 2);
        - an intent already claimed by *this* drainer with no
          outcome (e.g. just recovery-re-claimed) executes
          first, serially;
        - otherwise the lowest-sequence claimable intent is
          reached: stale ones (past the §12.2 residence bound)
          are refused at claim, and the claimable set is then
          either executed serially (keyed / write / execute
          work) or — when the lowest claimable intent is an
          unkeyed read and the §8 bound allows — claimed in
          intent order and executed concurrently in the read
          pool, up to ``queue_max_concurrent_reads`` at once.

        An integrity refusal (§12.4 rule 4 — claim/intent
        mismatch, TOCTOU, class change, schema re-validation)
        freezes the drain (``integrity-failure``). When nothing
        is claimable: ``stop-in-force`` if a held intent
        remains, ``blocked`` if queued/claimed work remains
        unconcluded, else ``queue-empty``. When the drain
        budget (the argument, or the policy's
        ``queue_drain_budget_seconds`` — absent 1920, malformed
        0) elapses with non-terminal intents remaining:
        ``drain-budget-exhausted`` (§10.2 point 4).

        Returns {"task_seq", "claims", "executions", "skips",
        "perception", "sync_reason", "integrity_failure",
        "concurrency_bound", "summary"}: the claims this drain
        journaled (recovery re-claims included), the skips it
        journaled (§7.3), each execution by edit id, ONE batched
        observation set (§10) over the outcomes concluded during
        this drain, and the final queue summary."""
        started = time.monotonic()
        budget = budget_seconds if budget_seconds is not None \
            else self._drain_budget_seconds()
        bound = self.max_concurrent_reads()

        baseline_fold = _fold(self.journal, task_seq)
        baseline_outcomes = {
            _field(s, "outcome_seq") for s in _fold_intents(baseline_fold)
            if _field(s, "outcome_seq") is not None}
        # Barrier intents still open when this drain began: if
        # one concludes during the drain, §10.2 point 2 fires.
        baseline_barrier = {
            _edit_id(s) for s in _fold_intents(baseline_fold)
            if _field(s, "barrier") and not _is_terminal(s)}

        recovery = self.recover(task_seq=task_seq)
        claims: list = list(recovery["reclaimed"])
        executions: dict = {}
        skips: list = []
        collected: set = set()
        integrity_failure = False
        sync_reason = "queue-empty"
        attempted: set = set()

        def _collect(fold):
            for state in _fold_intents(fold):
                outcome_seq = _field(state, "outcome_seq")
                if outcome_seq is not None \
                        and outcome_seq not in baseline_outcomes:
                    collected.add(outcome_seq)

        def _barrier_concluded(fold) -> bool:
            return any(
                _edit_id(s) in baseline_barrier and _is_terminal(s)
                for s in _fold_intents(fold))

        def _integrity_hit(execution) -> bool:
            return isinstance(execution, dict) \
                and execution.get("status") == "refused" \
                and any(marker in str(execution.get("detail") or "")
                        for marker in _INTEGRITY_MARKERS)

        while True:
            # §7.3: conclusions from the previous round may have
            # left success dependents unmet — skip them (and
            # cascade) before anything else is claimed.
            skips.extend(self._propagate_skips(task_seq))

            fold = _fold(self.journal, task_seq)
            _collect(fold)
            states = [s for s in _fold_intents(fold)
                      if _task_seq_of(s) is None
                      or _task_seq_of(s) == task_seq]
            non_terminal = [s for s in states
                            if not _is_terminal(s)]

            if _task_stop_in_force(fold, task_seq) or any(
                    _state_name(s) == "held" for s in states):
                sync_reason = "stop-in-force"
                break
            if _barrier_concluded(fold):
                sync_reason = "barrier"
                break
            if not non_terminal:
                sync_reason = "queue-empty"
                break
            if max_intents is not None \
                    and len(executions) >= max_intents:
                sync_reason = "drain-budget-exhausted"
                break
            if time.monotonic() - started >= budget:
                sync_reason = "drain-budget-exhausted"
                break

            own_claimed = sorted(
                (s for s in states
                 if _state_name(s) == "claimed"
                 and _field(s, "claim_drainer", "drainer_id")
                 == self.drainer_id
                 and _field(s, "outcome_seq") is None),
                key=lambda s: (_intent_seq(s) is None,
                               _intent_seq(s) or 0))
            if own_claimed:
                state = own_claimed[0]
                claim = next(
                    (c for c in claims
                     if isinstance(c, dict)
                     and c.get("seq") is not None
                     and c.get("seq") == _field(state, "claim_seq")),
                    {"seq": _field(state, "claim_seq"),
                     "edit_id": _edit_id(state),
                     "intent_seq": _intent_seq(state),
                     "drainer_id": self.drainer_id})
                attempt_key = (_edit_id(state),
                               claim.get("seq")
                               if isinstance(claim, dict) else None)
                if attempt_key in attempted:
                    # This claim was already executed in this drain
                    # and the fold still shows no outcome: executing
                    # it again would break at-most-once per claim
                    # (§14).
                    sync_reason = "blocked"
                    break
                attempted.add(attempt_key)
                execution = self.execute_claim(state, claim)
                executions[_edit_id(state)] = execution
                if _integrity_hit(execution):
                    integrity_failure = True
                    sync_reason = "integrity-failure"
                    break
                continue

            claimable = sorted(
                (s for s in states if self.claimable(s, fold)),
                key=lambda s: (_intent_seq(s) is None,
                               _intent_seq(s) or 0))
            if not claimable:
                if any(_state_name(s) == "held" for s in states):
                    sync_reason = "stop-in-force"
                elif any(_state_name(s) in (
                        "queued", "claimed") for s in states):
                    sync_reason = "blocked"
                else:
                    sync_reason = "queue-empty"
                break

            # §12.2: claim-time residence. A claimable intent the
            # drainer has reached that has been queued past the
            # residence bound is refused at claim, stale-priced,
            # however it would re-validate.
            now = datetime.now(timezone.utc)
            stale = [s for s in claimable
                     if self._residence_expired(s, now)]
            if stale:
                for state in stale:
                    self._refuse_stale(state)
                continue

            head = claimable[0]
            batch = []
            if bound > 1 and self._pool_eligible(head):
                batch = [s for s in claimable
                         if self._pool_eligible(s)][:bound]
            if len(batch) >= 2:
                # §8: claims journaled serially, in intent order,
                # BEFORE any of the batch executes; executions
                # then overlap in the pool; outcome journaling
                # serializes in the journal itself (§8.3).
                batch_claims = []
                for state in batch:
                    claim = self.claim(state)
                    claims.append(claim)
                    batch_claims.append((state, claim))
                    attempted.add((
                        _edit_id(state),
                        claim.get("seq")
                        if isinstance(claim, dict) else None))
                with ThreadPoolExecutor(
                        max_workers=len(batch_claims)) as pool:
                    futures = {
                        pool.submit(self.execute_claim, state, claim):
                        state
                        for state, claim in batch_claims}
                    for future in as_completed(futures):
                        state = futures[future]
                        executions[_edit_id(state)] = future.result()
                fold = _fold(self.journal, task_seq)
                _collect(fold)
                if any(_integrity_hit(e) for e in (
                        executions[_edit_id(s)] for s in batch)):
                    integrity_failure = True
                    sync_reason = "integrity-failure"
                    break
                continue

            # Serial step: keyed, write-class, and execute-class
            # work — and every intent when the bound resolves to
            # 1 — claims and executes one at a time (§8.1).
            state = head
            claim = self.claim(state)
            claims.append(claim)
            attempt_key = (_edit_id(state),
                           claim.get("seq")
                           if isinstance(claim, dict) else None)
            if attempt_key in attempted:
                sync_reason = "blocked"
                break
            attempted.add(attempt_key)
            execution = self.execute_claim(state, claim)
            executions[_edit_id(state)] = execution
            if _integrity_hit(execution):
                integrity_failure = True
                sync_reason = "integrity-failure"
                break

        final_fold = _fold(self.journal, task_seq)
        _collect(final_fold)
        perception = build_observation_set(
            self.journal, final_fold, task_seq, collected, sync_reason)
        return {
            "task_seq": task_seq,
            "claims": claims,
            "executions": executions,
            "skips": skips,
            "perception": perception,
            "sync_reason": sync_reason,
            "integrity_failure": integrity_failure,
            "concurrency_bound": bound,
            "summary": queue_summary(final_fold, task_seq),
        }
