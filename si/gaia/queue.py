"""Gaia Q-1 queue core: the queue's journal entry kinds and the
fold that derives queue state from them (SI-50 §4, §5; slice
SI-51).

SI-50's governing sentence: a gated, journaled tool intent *is*
an enqueued intent — the journal is the queue, and there is no
second queue anywhere. This module is the reading side of that
design plus the journaled writers for the queue's own entry
kinds:

- **Writers** — :func:`record_queue_batch`,
  :func:`record_queue_claim`, :func:`record_queue_recovery_reclaim`,
  :func:`record_queue_refusal`, :func:`record_queue_skip`,
  :func:`record_queue_stop`, and
  :func:`record_queue_resume` journal ``queue_*`` experiences via
  the journal's existing ``record_experience`` (the record dict
  rides in the experience's ``record`` field, exactly like
  :mod:`si.gaia.task`'s ``task`` / ``task_stop`` records). A
  writer returns ``{"seq", "row_id", ...record fields}`` where
  ``seq`` is the journal row's own sequence, read back the way
  ``record_task_entry`` reads it (claims additionally key the
  read-back on the intent's edit id, carried as the experience's
  ``self_edit_id``).

- **The fold** — :func:`fold_queue` reads the bundle journal's
  ``self_edits`` and ``experiences`` collections, merges them in
  journal sequence order, and derives every tool intent's queue
  state (§5.1: state is folded, never stored; any prefix of the
  journal yields the state as of that prefix). Nothing is read
  from the filesystem, from executor logs, or from memory of a
  previous process (§5.4).

Fold rules (§5.2, §5.4, §11, §13.1), as implemented here:

- a ``tool`` self-edit starts an intent: verdict ADMIT ->
  ``queued``, REJECT -> ``rejected``, ESCALATE -> ``escalated``,
  missing/other -> ``proposed``. The gate-priced queue block
  (``gate_report["queue"]`` — ordering key, normalised
  ``depends_on``, barrier, batch seq, effective class) is read
  from the verdict entry, never recomputed, falling back to the
  intent dict's own declared fields (§7.2).
- a ``queue_claim`` / ``queue_recovery_reclaim`` moves the
  intent to ``claimed`` and carries the claim fields (the
  recovery form also records ``recovery_of`` — the original
  claim it supersedes, §13.3); a ``queue_refusal`` moves it to
  ``refused-at-claim`` (unless already executed); a
  ``queue_skip`` to ``skipped`` and carries the skip fields
  (``skip_cause``, ``skip_dependency_seq`` and
  ``skip_dependency_outcome_seq`` — §7.3: a queued intent
  whose ``success`` dependency concluded any other way is
  skipped, journaled with the cause and the dependency's
  outcome sequence, and never executes); a ``tool_result``
  outcome to ``executed`` (or ``refused-at-claim`` when the
  execution status is ``refused``). An intent's ``intent_ts``
  is the ``ts`` of its own self-edit row — the admission
  timestamp recorded in the journal, from which Q-2 measures
  claim-time residence.
- ``queue_stop`` / ``queue_resume`` are per-task operator leash
  entries (§11): after folding, a task with a stop in force has
  its still-``queued`` intents derived to ``held``; a later
  resume un-holds them (they fold back to ``queued``). A restart
  never clears a stop — the fold is all there is (§11.3).
- integrity faults (§5.4), scoped to queue-managed intents
  (``batch_seq`` is not None, or a claim exists): an outcome
  entry with no prior claim, or a second outcome entry for the
  same intent, marks the intent ``outcome-unknown`` with the
  fault recorded on the intent and on its task. Non-queue-managed
  (legacy, pre-queue) outcomes fold normally — the serial task
  loop journaled outcomes without claims for its whole life, and
  that history is not a fault.
- compensation (SI-41 §11.2) is an annotation only: a later tool
  intent whose ``parents`` name an earlier intent sets that
  intent's ``compensated_by``; its state is never rewritten.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: Every queue state name of SI-50 §5.2, in the table's order.
#: ``compensated`` is listed for completeness of the vocabulary:
#: in the fold it is an annotation (``compensated_by``), never a
#: state an intent's ``state`` field takes (§5.2, honesty note 2).
ALL_STATES = (
    "proposed",
    "rejected",
    "escalated",
    "queued",
    "claimed",
    "executed",
    "refused-at-claim",
    "skipped",
    "held",
    "outcome-unknown",
    "compensated",
)

#: States after which an intent never runs and never changes
#: (``outcome-unknown`` is terminal pending operator disposition,
#: §13.4 — the fold itself never moves an intent out of it).
TERMINAL_STATES = {
    "rejected",
    "escalated",
    "executed",
    "refused-at-claim",
    "skipped",
    "outcome-unknown",
}

#: States counted toward a task summary's queue depth (the task
#: spec for :func:`queue_summary`: queued + claimed + held).
_DEPTH_STATES = ("queued", "claimed", "held")


# ---------------------------------------------------------------------------
# Writers: the queue's journal entry kinds (§5, §6.1, §11.1, §13.3)
# ---------------------------------------------------------------------------

def _record_queue_entry(journal, kind: str, record: dict,
                        self_edit_id: str = "") -> dict:
    """Journal one queue entry as an experience and read back its
    journal sequence, the way ``task.record_task_entry`` does:
    after ``record_experience``, the newest matching row's ``seq``
    is the entry's own sequence (falling back to the journal tip).
    Returns ``{"seq", "row_id", ...record fields}``."""
    row_id = journal.record_experience(kind, record, self_edit_id)
    filter_ = {"kind": kind}
    if self_edit_id:
        filter_["self_edit_id"] = self_edit_id
    rows = journal.backend.find("experiences", filter=filter_,
                                sort="-seq", limit=1)
    seq = rows[0]["seq"] if rows else journal.tip()[0]
    return {"seq": seq, "row_id": row_id, **record}


def record_queue_batch(journal, task_seq, task_ref, declared_size,
                       proposer="scripted") -> dict:
    """Journal a batch entry (§6.1): a PROPOSE step offered a
    batch for a task. The batch entry is provenance only — each
    member is then gated and journaled individually, in order;
    a batch is not atomic. ``declared_size`` is the proposer's
    declared member count (a mismatch with the members that
    follow is a §6.2 integrity matter for the folder to see,
    never a reason to drop members)."""
    record = {
        "kind": "queue_batch",
        "source": "proposer",
        "task_seq": task_seq,
        "task_ref": task_ref,
        "declared_size": declared_size,
        "proposer": proposer,
    }
    return _record_queue_entry(journal, "queue_batch", record)


def record_queue_claim(journal, task_seq, edit_id, intent_seq,
                       drainer_id, lease_seconds,
                       effective_class=None, ordering_key=None) -> dict:
    """Journal a claim entry (§5.3): a drainer claims an intent
    before executing it — the claim's own journal sequence (the
    returned ``seq``) is the attempt's identity, and the lease
    (``lease_seconds``) is what recovery (§13.1) later judges an
    outcome-less claim by."""
    record = {
        "kind": "queue_claim",
        "source": "drainer",
        "task_seq": task_seq,
        "edit_id": edit_id,
        "intent_seq": intent_seq,
        "drainer_id": drainer_id,
        "lease_seconds": lease_seconds,
        "effective_class": effective_class,
        "ordering_key": ordering_key,
    }
    # NOTE (SI-51 integration): queue entries do NOT take the
    # row's self_edit_id column — that column names the self-edit
    # an OUTCOME belongs to (the executor's tool_result rows);
    # existing readers (task outcome lookup, provenance walks,
    # the B-3 portability contract) find outcomes by it. The
    # claim names its intent by the record's edit_id instead.
    return _record_queue_entry(journal, "queue_claim", record)


def record_queue_recovery_reclaim(journal, task_seq, edit_id,
                                  intent_seq, drainer_id,
                                  lease_seconds, original_claim_seq,
                                  effective_class=None,
                                  ordering_key=None) -> dict:
    """Journal a recovery re-claim entry (§13.3, read class): a
    claim orphaned by a crash, its lease expired and no outcome
    journaled, is re-claimed for re-execution. The entry names
    the original claim (``original_claim_seq``); the original
    claim stays in the fold as superseded — history is
    continued, not rewritten."""
    record = {
        "kind": "queue_recovery_reclaim",
        "source": "drainer",
        "task_seq": task_seq,
        "edit_id": edit_id,
        "intent_seq": intent_seq,
        "drainer_id": drainer_id,
        "lease_seconds": lease_seconds,
        "effective_class": effective_class,
        "ordering_key": ordering_key,
        "original_claim_seq": original_claim_seq,
    }
    return _record_queue_entry(journal, "queue_recovery_reclaim",
                               record)


def record_queue_refusal(journal, task_seq, edit_id, intent_seq,
                         cause, claim_seq=None) -> dict:
    """Journal a refusal-at-claim entry (§5.2, §7.5): claim-time
    re-validation failed, or the claim/intent check of §5.3
    failed, and the intent was never executed. ``claim_seq`` is
    the claim the refusal belongs to, or None when the refusal
    happened before any claim could be journaled."""
    record = {
        "kind": "queue_refusal",
        "source": "drainer",
        "task_seq": task_seq,
        "edit_id": edit_id,
        "intent_seq": intent_seq,
        "claim_seq": claim_seq,
        "cause": cause,
    }
    return _record_queue_entry(journal, "queue_refusal", record)


def record_queue_skip(journal, task_seq, edit_id, intent_seq,
                      cause, dependency_seq=None,
                      dependency_outcome_seq=None) -> dict:
    """Journal a skip entry (§7.3): a queued intent whose
    ``success`` dependency concluded any other way is skipped
    — journaled with the cause and the dependency's outcome
    sequence — and never executes. ``dependency_seq`` is the
    intent sequence of the dependency and
    ``dependency_outcome_seq`` the journal sequence of its
    concluding outcome, or None when not applicable."""
    record = {
        "kind": "queue_skip",
        "source": "drainer",
        "task_seq": task_seq,
        "edit_id": edit_id,
        "intent_seq": intent_seq,
        "cause": cause,
        "dependency_seq": dependency_seq,
        "dependency_outcome_seq": dependency_outcome_seq,
    }
    return _record_queue_entry(journal, "queue_skip", record)


def record_queue_stop(journal, task_seq, cause, detail="",
                      operator="operator") -> dict:
    """Journal an operator stop entry (§11.1): the leash against
    a queue. Source ``operator``; it is not a gate verdict and
    the gate has no say in it. From its journal sequence the
    task's queued intents fold to ``held`` until a resume."""
    record = {
        "kind": "queue_stop",
        "source": "operator",
        "operator": operator,
        "task_seq": task_seq,
        "cause": cause,
        "detail": detail,
    }
    return _record_queue_entry(journal, "queue_stop", record)


def record_queue_resume(journal, task_seq, detail="",
                        operator="operator") -> dict:
    """Journal an operator resume entry (§11.1): the only way a
    held task's queue becomes claimable again — a new operator
    cause, journaled like the stop was."""
    record = {
        "kind": "queue_resume",
        "source": "operator",
        "operator": operator,
        "task_seq": task_seq,
        "detail": detail,
    }
    return _record_queue_entry(journal, "queue_resume", record)


# ---------------------------------------------------------------------------
# The fold (§5.1, §5.4)
# ---------------------------------------------------------------------------

@dataclass
class QueueIntentState:
    """One tool intent's folded queue state (§5.2). Every field
    is derived from journal entries naming the intent, in
    sequence order; ``source_seqs`` / ``parents`` are the
    intent's own journaled provenance chain (§4.1)."""

    edit_id: str
    intent_seq: int
    task_seq: int | None = None
    batch_seq: int | None = None
    tool: str = ""
    args: dict = field(default_factory=dict)
    reason: str = ""
    verdict: str | None = None
    effective_class: str | None = None
    ordering_key: str | None = None
    depends_on: list = field(default_factory=list)
    barrier: bool = False
    state: str = "proposed"
    claim_seq: int | None = None
    claim_drainer: str | None = None
    claim_lease_seconds: int | float | None = None
    claim_ts: str | None = None
    outcome_seq: int | None = None
    outcome_status: str | None = None
    execution: dict | None = None
    fault: str | None = None
    recovery_of: int | None = None
    compensated_by: str | None = None
    source_seqs: list = field(default_factory=list)
    parents: list = field(default_factory=list)
    #: The journaled cause of a ``queue_refusal`` naming this
    #: intent, when one exists (the state carries the outcome of
    #: the refusal; this carries its reason).
    refusal_cause: str | None = None
    #: The journaled fields of a ``queue_skip`` naming this
    #: intent, when one exists (§7.3): its cause, the intent
    #: sequence of the dependency that concluded another way,
    #: and the journal sequence of that dependency's outcome.
    skip_cause: str | None = None
    skip_dependency_seq: int | None = None
    skip_dependency_outcome_seq: int | None = None
    #: The ``ts`` of this intent's own self-edit row — the
    #: admission timestamp recorded in the journal, from which
    #: Q-2 measures claim-time residence.
    intent_ts: str | None = None


@dataclass
class QueueFold:
    """The folded queue: per-intent states, per-task leash /
    integrity bookkeeping, and the admission order (§5.4)."""

    intents: dict = field(default_factory=dict)   # edit_id -> state
    by_seq: dict = field(default_factory=dict)    # intent_seq -> state
    tasks: dict = field(default_factory=dict)     # task_seq -> slot
    order: list = field(default_factory=list)     # edit_ids by intent_seq


def _task_slot(fold: QueueFold, task_seq) -> dict:
    """The per-task bookkeeping slot, created on first sight of
    the task (a task entry, an intent chained to it, or any
    queue entry naming it)."""
    return fold.tasks.setdefault(task_seq, {
        "stop_in_force": False,
        "stop_seq": None,
        "resume_seq": None,
        "integrity_faults": [],
    })


def _normalise_depends_on(raw) -> list:
    """Normalise a declared ``depends_on`` (§7.3) to a list of
    ``{"seq": int, "requirement": str}``: a bare int is an
    ``outcome`` requirement; a dict carries its declared
    requirement (default ``outcome``). Unparseable entries are
    dropped here — pricing malformed dependencies is the gate's
    job (§6.2); the fold only reads the priced record."""
    if raw is None:
        return []
    if isinstance(raw, (int, dict)):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            out.append({"seq": item, "requirement": "outcome"})
        elif isinstance(item, dict):
            try:
                seq = int(item.get("seq"))
            except (TypeError, ValueError):
                continue
            out.append({"seq": seq,
                        "requirement": str(
                            item.get("requirement") or "outcome")})
        elif isinstance(item, str) and item.isdigit():
            out.append({"seq": int(item), "requirement": "outcome"})
    return out


def _derive_task_seq(intent: dict, task_seqs: set,
                     requested) -> int | None:
    """The task a tool intent belongs to (§4.4): the task entry
    whose seq appears in the intent's ``source_seqs``, or whose
    ``task@<seq>`` ref appears in its ``parents``. When the
    caller asked for one task, an intent chained to that task is
    recognised even if its task entry is outside the folded
    window. Otherwise None — the fold does not guess a task for
    an unchained intent."""
    source_seqs = intent.get("source_seqs") or []
    parents = intent.get("parents") or []
    for value in source_seqs:
        if isinstance(value, int) and not isinstance(value, bool) \
                and value in task_seqs:
            return value
    for parent in parents:
        if isinstance(parent, str) and parent.startswith("task@"):
            try:
                seq = int(parent[len("task@"):])
            except ValueError:
                continue
            if seq in task_seqs:
                return seq
    if requested is not None and (
            requested in source_seqs
            or f"task@{requested}" in parents):
        return requested
    return None


def _start_intent(row: dict, task_seqs: set,
                  requested) -> QueueIntentState | None:
    """Fold a ``tool`` self-edit row into a fresh intent state,
    or None when the caller scoped the fold to another task."""
    intent = row.get("intent") or {}
    derived = _derive_task_seq(intent, task_seqs, requested)
    if requested is not None and derived != requested:
        return None
    report = row.get("gate_report") or {}
    queue_block = report.get("queue")
    if not isinstance(queue_block, dict):
        queue_block = {}

    def priced(name, fallback=None):
        value = queue_block.get(name)
        return fallback if value is None else value

    verdict = row.get("gate_verdict")
    if verdict == "ADMIT":
        state = "queued"
    elif verdict == "REJECT":
        state = "rejected"
    elif verdict == "ESCALATE":
        state = "escalated"
    else:
        state = "proposed"
    return QueueIntentState(
        edit_id=row.get("edit_id"),
        intent_seq=row.get("seq"),
        task_seq=derived,
        batch_seq=priced("batch_seq", intent.get("batch_seq")),
        tool=intent.get("tool") or "",
        args=dict(intent.get("args") or {}),
        reason=row.get("reason") or intent.get("reason") or "",
        verdict=verdict,
        effective_class=priced("effective_class",
                               intent.get("effective_class")),
        ordering_key=priced("ordering_key",
                            intent.get("ordering_key")),
        depends_on=_normalise_depends_on(
            priced("depends_on", intent.get("depends_on"))),
        barrier=bool(priced("barrier", intent.get("barrier", False))),
        state=state,
        source_seqs=list(intent.get("source_seqs") or []),
        parents=list(intent.get("parents") or []),
        intent_ts=row.get("ts"),
    )


def _record_fault(fold: QueueFold, state: QueueIntentState,
                  description: str) -> None:
    """Mark an intent ``outcome-unknown`` with an integrity
    fault (§5.4), recorded both on the intent and on its task
    (the task's fault list is what the summary reports and what
    blocks the drain until disposition, §13.4)."""
    state.state = "outcome-unknown"
    state.fault = description
    if state.task_seq is not None:
        slot = _task_slot(fold, state.task_seq)
        if description not in slot["integrity_faults"]:
            slot["integrity_faults"].append(description)


def _apply_outcome(fold: QueueFold, state: QueueIntentState,
                   seq, record: dict) -> None:
    """Fold one ``tool_result`` outcome entry onto its intent."""
    execution = record.get("execution")
    if not isinstance(execution, dict):
        execution = {}
    status = execution.get("status")
    if state.state in ("rejected", "escalated"):
        # Verdict-terminal intents were never queued and never
        # executed (§5.2); the non-execution outcome the executor
        # journals for them (SI-43) is kept as information but
        # does not rewrite the verdict's terminal state.
        if state.outcome_seq is None:
            state.outcome_seq = seq
            state.outcome_status = status
            state.execution = execution
        return
    queue_managed = state.batch_seq is not None \
        or state.claim_seq is not None
    if state.outcome_seq is not None:
        if queue_managed:
            _record_fault(
                fold, state,
                f"second outcome entry at seq {seq} for intent "
                f"{state.edit_id} (intent_seq {state.intent_seq}); "
                f"first outcome at seq {state.outcome_seq}")
        else:
            # Legacy, non-queue-managed: fold normally — the
            # later outcome supersedes, no fault (§5.4 scope).
            state.outcome_seq = seq
            state.outcome_status = status
            state.execution = execution
            state.state = "refused-at-claim" if status == "refused" \
                else "executed"
        return
    if queue_managed and state.claim_seq is None:
        _record_fault(
            fold, state,
            f"outcome entry at seq {seq} for intent "
            f"{state.edit_id} (intent_seq {state.intent_seq}) has "
            f"no prior claim (queue-managed intent, batch_seq "
            f"{state.batch_seq})")
        state.outcome_seq = seq
        state.outcome_status = status
        state.execution = execution
        return
    state.outcome_seq = seq
    state.execution = execution
    if status == "refused":
        state.state = "refused-at-claim"
        state.outcome_status = "refused"
    else:
        state.state = "executed"
        state.outcome_status = status


def _find_state(fold: QueueFold, record: dict, row: dict):
    """Locate the intent a queue / outcome entry names, by edit
    id (the record's ``edit_id`` / ``intent_id``, or the row's
    ``self_edit_id``) or, failing that, by intent sequence."""
    edit_id = record.get("edit_id") or record.get("intent_id") \
        or row.get("self_edit_id")
    if edit_id and edit_id in fold.intents:
        return fold.intents[edit_id]
    intent_seq = record.get("intent_seq")
    if intent_seq is not None and intent_seq in fold.by_seq:
        return fold.by_seq[intent_seq]
    return None


def fold_queue(journal, task_seq=None) -> QueueFold:
    """Fold the journal into queue state (§5.4).

    Reads ALL rows of the ``self_edits`` and ``experiences``
    collections, merges them, and folds in journal sequence
    order. With ``task_seq``, only that task's intents are
    folded (per-task bookkeeping is still returned for every
    task the journal names, so a summary can be taken for any
    of them)."""
    entries = []  # (seq, collection, row)
    for row in journal.backend.find("self_edits") or []:
        entries.append((row.get("seq") or 0, "self_edits", row))
    for row in journal.backend.find("experiences") or []:
        entries.append((row.get("seq") or 0, "experiences", row))
    entries.sort(key=lambda item: item[0])

    # Task entries first (§4.4): an intent belongs to the task
    # whose entry its provenance chain names, wherever in the
    # journal that entry sits relative to the fold window.
    task_seqs = set()
    for _seq, collection, row in entries:
        if collection != "experiences":
            continue
        record = row.get("record") or {}
        if row.get("kind") == "task" or record.get("kind") == "task":
            if row.get("seq") is not None:
                task_seqs.add(row["seq"])

    fold = QueueFold()
    for seq in sorted(task_seqs):
        _task_slot(fold, seq)

    for seq, collection, row in entries:
        if collection == "self_edits":
            if row.get("kind") != "tool":
                continue
            state = _start_intent(row, task_seqs, task_seq)
            if state is None:
                continue
            fold.intents[state.edit_id] = state
            fold.by_seq[state.intent_seq] = state
            if state.task_seq is not None:
                _task_slot(fold, state.task_seq)
            continue

        record = row.get("record") or {}
        kind = row.get("kind") or record.get("kind")

        if kind == "task":
            _task_slot(fold, seq)
            continue
        if kind == "queue_batch":
            if record.get("task_seq") is not None:
                _task_slot(fold, record["task_seq"])
            continue
        if kind == "queue_stop":
            named = record.get("task_seq")
            if named is not None:
                slot = _task_slot(fold, named)
                slot["stop_in_force"] = True
                slot["stop_seq"] = seq
            continue
        if kind == "queue_resume":
            named = record.get("task_seq")
            if named is not None:
                slot = _task_slot(fold, named)
                slot["stop_in_force"] = False
                slot["resume_seq"] = seq
            continue

        if kind in ("queue_claim", "queue_recovery_reclaim"):
            state = _find_state(fold, record, row)
            if state is None or state.state in TERMINAL_STATES:
                continue
            state.state = "claimed"
            state.claim_seq = seq
            state.claim_drainer = record.get("drainer_id")
            state.claim_lease_seconds = record.get("lease_seconds")
            state.claim_ts = row.get("ts")
            if record.get("effective_class") is not None:
                state.effective_class = record["effective_class"]
            if record.get("ordering_key") is not None:
                state.ordering_key = record["ordering_key"]
            if kind == "queue_recovery_reclaim":
                state.recovery_of = record.get("original_claim_seq")
            continue

        if kind == "queue_refusal":
            state = _find_state(fold, record, row)
            if state is None:
                continue
            state.refusal_cause = record.get("cause")
            if state.state not in TERMINAL_STATES:
                state.state = "refused-at-claim"
                if state.outcome_seq is None:
                    state.outcome_status = "refused"
            continue

        if kind == "queue_skip":
            state = _find_state(fold, record, row)
            if state is None or state.state in TERMINAL_STATES:
                continue
            state.state = "skipped"
            state.skip_cause = record.get("cause")
            state.skip_dependency_seq = record.get("dependency_seq")
            state.skip_dependency_outcome_seq = \
                record.get("dependency_outcome_seq")
            continue

        if kind == "tool_result" and row.get("self_edit_id"):
            state = _find_state(fold, record, row)
            if state is None:
                continue
            _apply_outcome(fold, state, seq, record)
            continue
        # Every other experience kind is not a queue event; the
        # fold ignores it (it remains in the journal for its own
        # readers).

    # Compensation annotation (§5.2 / SI-41 §11.2): a tool intent
    # whose parents name an earlier intent annotates that intent.
    # Annotation only — the named intent's state is unchanged,
    # and the first compensating intent in journal order wins.
    tool_rows = [row for _seq, collection, row in entries
                 if collection == "self_edits"
                 and row.get("kind") == "tool"]
    for row in tool_rows:
        child_id = row.get("edit_id")
        for parent in (row.get("intent") or {}).get("parents") or []:
            target = fold.intents.get(parent)
            if target is not None and parent != child_id \
                    and target.compensated_by is None:
                target.compensated_by = child_id

    # The operator leash, derived (§11.1/§11.3): a task with a
    # stop in force (a stop with no later resume) holds every
    # intent of its that is still merely queued. Claimed and
    # terminal intents are untouched — in-flight work at a stop
    # is §11.2's business, and its outcome is journaled normally.
    for state in fold.intents.values():
        if state.task_seq is None or state.state != "queued":
            continue
        slot = fold.tasks.get(state.task_seq)
        if slot and slot["stop_in_force"]:
            state.state = "held"

    fold.order = [state.edit_id for state in sorted(
        fold.intents.values(), key=lambda s: s.intent_seq or 0)]
    return fold


def queue_summary(fold: QueueFold, task_seq) -> dict:
    """The §10.3 queue summary for one task: counts by folded
    state, the depth at the sync point, and the ``outcome-unknown``
    and ``held`` intents named by intent sequence, plus the
    task's integrity faults. This is perception of the queue
    itself — data the mind plans its next batch against."""
    intents = sorted(
        (state for state in fold.intents.values()
         if state.task_seq == task_seq),
        key=lambda s: s.intent_seq or 0)
    counts: dict = {}
    for state in intents:
        counts[state.state] = counts.get(state.state, 0) + 1
    slot = fold.tasks.get(task_seq) or {
        "stop_in_force": False, "stop_seq": None,
        "resume_seq": None, "integrity_faults": [],
    }
    return {
        "task_seq": task_seq,
        "counts": counts,
        "depth": sum(1 for state in intents
                     if state.state in _DEPTH_STATES),
        "outcome_unknown": [state.intent_seq for state in intents
                            if state.state == "outcome-unknown"],
        "held": [state.intent_seq for state in intents
                 if state.state == "held"],
        "integrity_faults": list(slot["integrity_faults"]),
    }
