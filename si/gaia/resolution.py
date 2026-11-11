"""Gaia escalation resolution: the operator's answer to a gate
ESCALATE (SI-41 §5.4; SI-50 §5.2, §12.4 rule 2; slice SI-92).

The gap this slice closes: a tool intent the gate ESCALATEd was
journaled and halted, and *nothing could answer the escalation*.
SI-50 §5.2's state table always said the escalated intent is
"the operator's to admit or refuse per SI-41 §5.4", and SI-41
§5.4 priced the shape of that answer — the operator's admission
"materialises *the journaled intent*, with the admitter and
basis recorded, never a fresh, ungated execution", following
the SI-10 operator-admission precedent — but no slice built it.
SI-74's live run escalated an intent that then simply sat;
the 2026-11-03 Resident death left a journaled restart
proposal equally unanswerable. This module is the answer path,
built exactly on the in-queue precedent of SI-53's operator
dispositions (§13.4): a journaled operator act, a fold that
reads it, and a verifying act that fails closed.

What is built here:

- **The journal entry** — :func:`record_escalation_resolution`
  journals an ``escalation_resolution`` experience (existing
  collections, no schema change): source ``operator``, the
  operator's name, the basis, the intent's task / edit id /
  journal sequence, the resolution, and a free-text detail.
  The only resolutions are :data:`RESOLUTIONS`:
  ``admit-after-review`` (the operator reviewed the journaled
  intent and its ESCALATE report and admits *that* intent) and
  ``reject`` (the intent dies). There is no third resolution —
  in particular nothing that rewrites the gate's verdict: the
  self-edit keeps ESCALATE forever; the resolution is a later,
  separate, journaled act, exactly as a §13.4 disposition
  annotates an outcome-unknown intent without rewriting it.

- **The verifying act** — :func:`resolve_escalation` folds the
  journal, verifies the named intent really is folded
  ``escalated`` and not already resolved, and only then
  journals the resolution. Fail closed: an unknown intent, a
  non-escalated intent, an already-resolved intent (a double
  resolution), an invalid resolution value, or an empty
  operator raises ValueError and journals NOTHING — a
  resolution is an explicit operator act on a verified state.

- **The fold** (in :mod:`si.gaia.queue`) — an
  ``escalation_resolution`` entry annotates its intent
  (``resolution`` / ``resolution_seq`` / ``resolution_operator``
  / ``resolution_detail``). ``reject`` changes no state: the
  intent stays ``escalated``, terminal, and the annotation is
  the record of its death. ``admit-after-review`` moves the
  intent from ``escalated`` to ``queued`` — the only transition
  out of ``escalated``, driven by the journaled operator act,
  in journal sequence order, so any prefix fold still yields
  the state as of that prefix (§5.1). An *unresolved*
  escalated intent is never claimable and never executes.

- **The one execution** — :func:`execute_resolved_escalation`
  is this slice's driver: it folds, refuses unless the intent
  stands ``queued`` *by* an ``admit-after-review`` resolution,
  journals a claim (§5.3 — claim before execute, as the
  drainer does), and calls the executor with the claim and
  the resolution's journal sequence as its tickets. The
  executor re-verifies the resolution from the journal itself
  (an ``escalation_resolution`` row at exactly that sequence,
  naming this intent, resolving ``admit-after-review``, with
  no genuine execution outcome already journaled for the
  intent) before it will run an ESCALATE-verdict intent at
  all; every other executor discipline (journaled args,
  live-set, miniature scope, schema re-validation, TOCTOU
  re-pricing, work-root containment) runs unchanged. The
  queue drainer reaches the same path on a later drain: the
  resolved intent folds ``queued``, is claimed under §7.1,
  and :meth:`Drainer.execute_claim` passes the folded
  ``resolution_seq`` with the claim. Either way the intent
  runs **once**: after its execution the fold reads
  ``executed`` (or ``refused-at-claim``), no second claim is
  possible, and :func:`execute_resolved_escalation` refuses
  to run it again. A crash between claim and outcome is
  SI-50 §13's business, unchanged: the claim folds
  outcome-unknown at recovery and is never blindly replayed
  — a resolution is not a replay licence.

Honest scope: this is operator-resolution machinery at the
existing miniature/embedded scope, exercised by this slice's
tests against the miniature bundle and the embedded journal.
Nothing here is a live deployment claim, no policy key makes
a resolution automatic, and the gate's ESCALATE verdicts are
untouched — the gate still escalates exactly what it
escalated; a human (or an operator process acting as one)
now can answer.
"""
from __future__ import annotations

from si.gaia.queue import fold_queue, record_queue_claim

#: The only operator resolutions of an escalated intent
#: (SI-41 §5.4: the operator's to admit or refuse). There is
#: no third resolution, and in particular no "mark-admitted"
#: that rewrites the verdict — admission is a new journaled
#: act standing next to the ESCALATE it answers.
RESOLUTIONS = ("admit-after-review", "reject")

#: The honest basis recorded on a resolution when the caller
#: supplies none: what the operator is recorded as having
#: reviewed. It claims a review of the record, never a
#: re-evaluation of the gate's checks — the gate's report
#: stands as journaled.
DEFAULT_BASIS = (
    "operator review of the journaled intent and its gate "
    "ESCALATE report (SI-41 §5.4); the verdict is not "
    "re-evaluated and not rewritten — this resolution is a "
    "separate journaled operator act admitting or refusing "
    "the journaled intent as it stands")


# ---------------------------------------------------------------------------
# Writer: the resolution entry kind (§5.4)
# ---------------------------------------------------------------------------

def _record_resolution_entry(journal, record: dict) -> dict:
    """Journal one resolution entry as an experience and read
    back its journal sequence, the way the queue writers do
    (:func:`si.gaia.queue._record_queue_entry`): after
    ``record_experience``, the newest matching row's ``seq``
    is the entry's own sequence (falling back to the journal
    tip). Returns ``{"seq", "row_id", ...record fields}``."""
    row_id = journal.record_experience(
        "escalation_resolution", record, "")
    rows = journal.backend.find(
        "experiences", filter={"kind": "escalation_resolution"},
        sort="-seq", limit=1)
    seq = rows[0]["seq"] if rows else journal.tip()[0]
    return {"seq": seq, "row_id": row_id, **record}


def record_escalation_resolution(journal, task_seq, edit_id,
                                 intent_seq, resolution,
                                 detail="", operator="operator",
                                 basis="") -> dict:
    """Journal a resolution entry (§5.4): the operator's
    explicit act on an escalated intent. Source ``operator``;
    not a gate verdict, and it does not alter the intent's
    self-edit — the ESCALATE stands in history beside it.
    ``resolution`` must be one of :data:`RESOLUTIONS` —
    anything else raises ValueError and journals nothing.
    This is the low-level writer; callers should prefer
    :func:`resolve_escalation`, which first verifies the
    folded intent really is escalated and unresolved."""
    if resolution not in RESOLUTIONS:
        raise ValueError(
            f"resolution {resolution!r} is not one of "
            f"{list(RESOLUTIONS)} (SI-41 §5.4: the operator's "
            f"answers are admit-after-review or reject; there "
            f"is no resolution that rewrites the verdict)")
    record = {
        "kind": "escalation_resolution",
        "source": "operator",
        "operator": operator,
        "basis": basis or DEFAULT_BASIS,
        "task_seq": task_seq,
        "edit_id": edit_id,
        "intent_seq": intent_seq,
        "resolution": resolution,
        "detail": detail,
    }
    return _record_resolution_entry(journal, record)


def resolve_escalation(journal, intent_seq, resolution,
                       detail="", operator="operator", basis="",
                       task_seq=None) -> dict:
    """The operator resolution act (§5.4, SI-92): fold the
    journal, verify the named intent is folded ``escalated``
    and not already resolved, then journal the resolution.
    Fail closed: an unknown intent, a non-``escalated``
    intent, an already-resolved intent (double resolution), an
    invalid resolution, or an empty operator raises ValueError
    and journals NOTHING — a resolution is an explicit
    operator act on a verified state, never a blind write and
    never a rewrite of the gate's record.

    Returns the journaled entry (``{"seq", ...}``): for
    ``admit-after-review`` that ``seq`` is the ticket the
    executor will demand before running the intent."""
    if resolution not in RESOLUTIONS:
        raise ValueError(
            f"resolution {resolution!r} is not one of "
            f"{list(RESOLUTIONS)} (SI-41 §5.4)")
    if not isinstance(operator, str) or not operator.strip():
        raise ValueError(
            "an escalation resolution requires an operator "
            "name (SI-41 §5.4: the admitter is recorded; an "
            "anonymous resolution is refused, nothing "
            "journaled)")
    fold = fold_queue(journal, task_seq=task_seq)
    state = fold.by_seq.get(intent_seq)
    if state is None:
        raise ValueError(
            f"no folded intent at seq {intent_seq}: nothing "
            f"to resolve (fail closed, nothing journaled)")
    if state.resolution is not None:
        raise ValueError(
            f"intent seq {intent_seq} is already resolved "
            f"({state.resolution!r} at seq "
            f"{state.resolution_seq}): a second resolution "
            f"would rewrite the operator's record (nothing "
            f"journaled)")
    if state.state != "escalated":
        raise ValueError(
            f"intent seq {intent_seq} folds to "
            f"{state.state!r}, not 'escalated': only an "
            f"escalated intent can be resolved (SI-41 §5.4; "
            f"nothing journaled)")
    return record_escalation_resolution(
        journal, state.task_seq, state.edit_id, state.intent_seq,
        resolution, detail=detail, operator=operator, basis=basis)


# ---------------------------------------------------------------------------
# The one execution (§5.3 claim, §5.4 ticket)
# ---------------------------------------------------------------------------

def execute_resolved_escalation(journal, bundle_dir, manifest,
                                policy, work_root, intent_seq,
                                drainer_id="resolution-drainer",
                                lease_seconds=300,
                                executor=None) -> dict:
    """Execute an escalation the operator admitted — once.

    Folds the journal and refuses (ValueError, nothing
    journaled) unless the named intent stands ``queued``
    *because* an ``admit-after-review`` resolution moved it
    there: an unresolved escalation never executes, a
    ``reject``-resolved intent is dead, and an intent already
    claimed or executed by anyone is not this call's to run.
    Then, in order: journal a claim (§5.3 — claim before
    execute), and run the executor on the journaled intent
    with the claim and the resolution sequence as tickets.
    The executor re-verifies both tickets against the journal
    and applies every ordinary discipline to the journaled
    args; a refusal there is journaled as the intent's
    outcome (``refused-at-claim`` in the fold) and is still
    the one shot — a resolved intent is never silently
    re-runnable.

    Returns {"intent_seq", "edit_id", "resolution_seq",
    "claim", "execution"}."""
    fold = fold_queue(journal)
    state = fold.by_seq.get(intent_seq)
    if state is None:
        raise ValueError(
            f"no folded intent at seq {intent_seq}: nothing "
            f"to execute (fail closed, nothing journaled)")
    if state.resolution is None:
        raise ValueError(
            f"intent seq {intent_seq} folds to "
            f"{state.state!r} with no operator resolution: an "
            f"unresolved escalation never executes (SI-41 "
            f"§5.4; nothing journaled)")
    if state.resolution != "admit-after-review":
        raise ValueError(
            f"intent seq {intent_seq} was resolved "
            f"{state.resolution!r} at seq "
            f"{state.resolution_seq}: the operator refused "
            f"it, and a refused intent never executes "
            f"(nothing journaled)")
    if state.state != "queued":
        raise ValueError(
            f"intent seq {intent_seq} folds to "
            f"{state.state!r}, not 'queued': a resolved "
            f"escalation runs exactly once, through its "
            f"claim — an intent already claimed, executed, "
            f"or otherwise concluded is not re-run (SI-50 "
            f"§13: never blindly replay; nothing journaled)")
    claim = record_queue_claim(
        journal, state.task_seq, state.edit_id, state.intent_seq,
        drainer_id, lease_seconds,
        effective_class=state.effective_class,
        ordering_key=state.ordering_key)
    if executor is None:
        from si.gaia.executor import Executor
        executor = Executor(bundle_dir, manifest, journal,
                            policy, work_root)
    execution = executor.execute_journaled(
        state.edit_id,
        expected_effective_class=state.effective_class,
        claim_intent_seq=state.intent_seq,
        claim_seq=claim["seq"],
        resolution_seq=state.resolution_seq)
    return {"intent_seq": state.intent_seq,
            "edit_id": state.edit_id,
            "resolution_seq": state.resolution_seq,
            "claim": claim,
            "execution": execution}
