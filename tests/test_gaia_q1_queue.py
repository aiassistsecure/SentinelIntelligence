"""SI-51 (SI-50 §5): Gaia Q-1 queue core — the queue's journal
entry kinds and the fold that derives queue state from them.

The journal is the queue (§4): these tests build a journal
directly on the embedded backend (no bundle, no NEDB), journal
task / intent / queue / outcome entries through the real
journal methods and the Q-1 writers, and fold with
:func:`si.gaia.queue.fold_queue`. Coverage:

- verdicts fold to queued / rejected / escalated / proposed,
  with the gate-priced queue block (ordering key, normalised
  depends_on, barrier, batch seq, effective class) read from
  the verdict entry;
- claim then outcome folds claimed then executed, claim fields
  carried (drainer, lease, claim seq / ts);
- an outcome with no prior claim for a batch-managed intent is
  a §5.4 integrity fault -> ``outcome-unknown``; a legacy
  (non-queue-managed) outcome without a claim folds normally,
  and a second outcome for a managed intent faults too;
- the operator leash (§11): a stop folds queued intents to
  held (claimed intents untouched), a resume un-holds them;
- recovery re-claim (§13.3) supersedes the original claim
  (``recovery_of`` set, ``claim_seq`` is the new claim);
- ``queue_summary`` counts / depth / outcome-unknown (§10.3);
- prefix stability: folding after each append matches the
  incremental expectation for a claim/outcome/refusal sequence;
- compensation is an annotation only (``compensated_by`` set,
  state unchanged).
"""
import pytest

from si.gaia.queue import (
    ALL_STATES,
    TERMINAL_STATES,
    fold_queue,
    queue_summary,
    record_queue_batch,
    record_queue_claim,
    record_queue_recovery_reclaim,
    record_queue_refusal,
    record_queue_resume,
    record_queue_stop,
)
from si.gaia.task import record_task_entry
from si.journal import EmbeddedBackend, Journal


@pytest.fixture()
def journal(tmp_path):
    journal = Journal(EmbeddedBackend(str(tmp_path / "journal")))
    yield journal
    journal.close()


def _task(journal):
    return record_task_entry(journal, "drain the queue",
                             "/tmp/q1-work")


def _add_intent(journal, task, verdict="ADMIT", tool="fs_read",
                queue=None, parents=None, source_seqs=None,
                args=None):
    """Journal one tool self-edit the way the executor's cycle
    does (intent dict + verdict + gate report). Returns
    (edit_id, intent_seq)."""
    intent = {
        "kind": "tool",
        "tool": tool,
        "args": args if args is not None else {"path": "a.txt"},
        "parents": parents if parents is not None else [task["ref"]],
        "source_seqs": (source_seqs if source_seqs is not None
                        else [task["seq"]]),
        "reason": "q1 test intent",
    }
    report = {"verdict": verdict, "checks": []}
    if queue is not None:
        report["queue"] = queue
    edit_id = journal.record_self_edit("tool", intent,
                                       "q1 test intent", verdict,
                                       report)
    rows = journal.backend.find("self_edits",
                                filter={"edit_id": edit_id}, limit=1)
    return edit_id, rows[0]["seq"]


def _add_outcome(journal, edit_id, status="completed",
                 tool="fs_read"):
    """Journal one tool_result outcome the way the executor does
    (record dict with the execution, self_edit_id naming the
    intent). Returns the outcome's journal seq."""
    execution = {"tool": tool, "intent_id": edit_id,
                 "class": "read", "status": status,
                 "result": {"ok": True}}
    journal.record_experience("tool_result", {
        "kind": "tool_result",
        "source": f"tool:{tool}",
        "intent_id": edit_id,
        "tool": tool,
        "gate_verdict": "ADMIT",
        "effective_class": "read",
        "execution": execution,
    }, edit_id)
    rows = journal.backend.find(
        "experiences", filter={"self_edit_id": edit_id},
        sort="-seq", limit=1)
    return rows[0]["seq"]


def _batch_queue_block(journal, task, **overrides):
    """A real batch entry plus the gate queue block that prices
    its members with that batch's seq."""
    batch = record_queue_batch(journal, task["seq"], task["ref"],
                               declared_size=2)
    block = {"ordering_key": None, "depends_on": [],
             "barrier": False, "batch_seq": batch["seq"],
             "effective_class": "read"}
    block.update(overrides)
    return batch, block


# -- (1) verdicts ---------------------------------------------------------

def test_fold_derives_states_from_verdicts(journal):
    task = _task(journal)
    _batch, block = _batch_queue_block(
        journal, task, ordering_key="fs:/tmp/q1-work/a.txt",
        depends_on=[5, {"seq": 6, "requirement": "success"}],
        barrier=True)
    edit_q, seq_q = _add_intent(journal, task, "ADMIT", queue=block)
    edit_r, seq_r = _add_intent(journal, task, "REJECT")
    edit_e, seq_e = _add_intent(journal, task, "ESCALATE")
    edit_p, seq_p = _add_intent(journal, task, "PROPOSE")

    fold = fold_queue(journal)

    assert fold.intents[edit_q].state == "queued"
    assert fold.intents[edit_r].state == "rejected"
    assert fold.intents[edit_e].state == "escalated"
    assert fold.intents[edit_p].state == "proposed"
    # Order and the by-seq index follow journal (intent) order.
    assert fold.order == [edit_q, edit_r, edit_e, edit_p]
    assert fold.by_seq[seq_q].edit_id == edit_q
    assert fold.by_seq[seq_p].state == "proposed"
    # The priced queue block is read from the verdict entry.
    priced = fold.intents[edit_q]
    assert priced.task_seq == task["seq"]
    assert priced.batch_seq == block["batch_seq"]
    assert priced.ordering_key == "fs:/tmp/q1-work/a.txt"
    assert priced.effective_class == "read"
    assert priced.barrier is True
    assert priced.depends_on == [
        {"seq": 5, "requirement": "outcome"},
        {"seq": 6, "requirement": "success"},
    ]
    assert priced.tool == "fs_read"
    assert priced.args == {"path": "a.txt"}
    # The state vocabulary is §5.2's, and the terminal set is
    # the one the slice spec pins.
    assert "held" in ALL_STATES and "outcome-unknown" in ALL_STATES
    assert TERMINAL_STATES == {
        "rejected", "escalated", "executed", "refused-at-claim",
        "skipped", "outcome-unknown"}


def test_writers_return_journal_seq_and_record_fields(journal):
    task = _task(journal)
    batch = record_queue_batch(journal, task["seq"], task["ref"],
                               declared_size=3, proposer="mind-x")
    assert batch["kind"] == "queue_batch"
    assert batch["source"] == "proposer"
    assert batch["task_seq"] == task["seq"]
    assert batch["task_ref"] == task["ref"]
    assert batch["declared_size"] == 3
    assert batch["proposer"] == "mind-x"
    assert batch["row_id"]
    rows = journal.backend.find(
        "experiences", filter={"kind": "queue_batch"},
        sort="-seq", limit=1)
    assert rows[0]["seq"] == batch["seq"]
    assert rows[0]["record"]["declared_size"] == 3


# -- (2) claim then outcome ------------------------------------------------

def test_claim_then_outcome_folds_claimed_then_executed(journal):
    task = _task(journal)
    _batch, block = _batch_queue_block(journal, task)
    edit_id, intent_seq = _add_intent(journal, task, "ADMIT",
                                      queue=block)

    claim = record_queue_claim(
        journal, task["seq"], edit_id, intent_seq,
        drainer_id="drainer-1", lease_seconds=300,
        effective_class="read", ordering_key="house")
    assert claim["kind"] == "queue_claim"
    assert claim["source"] == "drainer"
    assert claim["edit_id"] == edit_id

    fold = fold_queue(journal)
    state = fold.intents[edit_id]
    assert state.state == "claimed"
    assert state.claim_seq == claim["seq"]
    assert state.claim_drainer == "drainer-1"
    assert state.claim_lease_seconds == 300
    assert state.claim_ts is not None
    assert state.effective_class == "read"
    assert state.ordering_key == "house"

    outcome_seq = _add_outcome(journal, edit_id)
    fold = fold_queue(journal)
    state = fold.intents[edit_id]
    assert state.state == "executed"
    assert state.outcome_seq == outcome_seq
    assert state.outcome_status == "completed"
    assert state.execution["status"] == "completed"
    assert state.claim_seq == claim["seq"]
    assert state.fault is None


def test_refused_execution_outcome_folds_refused_at_claim(journal):
    task = _task(journal)
    _batch, block = _batch_queue_block(journal, task)
    edit_id, intent_seq = _add_intent(journal, task, "ADMIT",
                                      queue=block)
    record_queue_claim(journal, task["seq"], edit_id, intent_seq,
                       drainer_id="drainer-1", lease_seconds=300)
    _add_outcome(journal, edit_id, status="refused")
    state = fold_queue(journal).intents[edit_id]
    assert state.state == "refused-at-claim"
    assert state.outcome_status == "refused"


# -- (3) integrity faults ----------------------------------------------------

def test_outcome_without_claim_for_batch_intent_is_fault(journal):
    task = _task(journal)
    _batch, block = _batch_queue_block(journal, task)
    edit_id, intent_seq = _add_intent(journal, task, "ADMIT",
                                      queue=block)
    _add_outcome(journal, edit_id)  # no claim was ever journaled

    fold = fold_queue(journal)
    state = fold.intents[edit_id]
    assert state.state == "outcome-unknown"
    assert state.fault is not None
    assert "no prior claim" in state.fault
    assert state.fault in fold.tasks[task["seq"]]["integrity_faults"]

    summary = queue_summary(fold, task["seq"])
    assert summary["outcome_unknown"] == [intent_seq]
    assert summary["integrity_faults"] == [state.fault]


def test_legacy_outcome_without_claim_folds_normally(journal):
    task = _task(journal)
    # No batch, no gate queue block, no claim: the serial task
    # loop's historical shape — an outcome is just an outcome.
    edit_id, _seq = _add_intent(journal, task, "ADMIT")
    _add_outcome(journal, edit_id)
    fold = fold_queue(journal)
    state = fold.intents[edit_id]
    assert state.state == "executed"
    assert state.fault is None
    assert fold.tasks[task["seq"]]["integrity_faults"] == []


def test_second_outcome_for_managed_intent_is_fault(journal):
    task = _task(journal)
    _batch, block = _batch_queue_block(journal, task)
    edit_id, intent_seq = _add_intent(journal, task, "ADMIT",
                                      queue=block)
    record_queue_claim(journal, task["seq"], edit_id, intent_seq,
                       drainer_id="drainer-1", lease_seconds=300)
    _add_outcome(journal, edit_id)
    _add_outcome(journal, edit_id)
    state = fold_queue(journal).intents[edit_id]
    assert state.state == "outcome-unknown"
    assert state.fault is not None
    assert "second outcome" in state.fault


# -- (4) the operator leash ---------------------------------------------------

def test_stop_holds_queued_and_resume_unholds(journal):
    task = _task(journal)
    edit_a, seq_a = _add_intent(journal, task, "ADMIT")
    edit_b, seq_b = _add_intent(journal, task, "ADMIT")
    edit_c, _seq_c = _add_intent(journal, task, "ADMIT")
    record_queue_claim(journal, task["seq"], edit_c, _seq_c,
                       drainer_id="drainer-1", lease_seconds=300)

    stop = record_queue_stop(journal, task["seq"],
                             cause="operator-stop",
                             detail="leash pulled")
    assert stop["kind"] == "queue_stop"
    assert stop["source"] == "operator"

    fold = fold_queue(journal)
    assert fold.intents[edit_a].state == "held"
    assert fold.intents[edit_b].state == "held"
    # A claimed intent is in flight, not queued: the stop does
    # not rewrite it (§11.2 handles its outcome separately).
    assert fold.intents[edit_c].state == "claimed"
    slot = fold.tasks[task["seq"]]
    assert slot["stop_in_force"] is True
    assert slot["stop_seq"] == stop["seq"]

    summary = queue_summary(fold, task["seq"])
    assert summary["held"] == [seq_a, seq_b]
    assert summary["depth"] == 3  # held + held + claimed

    resume = record_queue_resume(journal, task["seq"],
                                 detail="operator resumed")
    fold = fold_queue(journal)
    assert fold.intents[edit_a].state == "queued"
    assert fold.intents[edit_b].state == "queued"
    slot = fold.tasks[task["seq"]]
    assert slot["stop_in_force"] is False
    assert slot["resume_seq"] == resume["seq"]
    assert queue_summary(fold, task["seq"])["held"] == []


# -- (5) recovery re-claim ------------------------------------------------------

def test_recovery_reclaim_supersedes_original_claim(journal):
    task = _task(journal)
    _batch, block = _batch_queue_block(journal, task)
    edit_id, intent_seq = _add_intent(journal, task, "ADMIT",
                                      queue=block)
    original = record_queue_claim(
        journal, task["seq"], edit_id, intent_seq,
        drainer_id="drainer-dead", lease_seconds=300)

    reclaim = record_queue_recovery_reclaim(
        journal, task["seq"], edit_id, intent_seq,
        drainer_id="drainer-new", lease_seconds=300,
        original_claim_seq=original["seq"])
    assert reclaim["kind"] == "queue_recovery_reclaim"
    assert reclaim["original_claim_seq"] == original["seq"]

    state = fold_queue(journal).intents[edit_id]
    assert state.state == "claimed"
    assert state.claim_seq == reclaim["seq"]
    assert state.claim_seq != original["seq"]
    assert state.claim_drainer == "drainer-new"
    assert state.recovery_of == original["seq"]


# -- (6) summary ------------------------------------------------------------------

def test_queue_summary_counts_and_depth(journal):
    task = _task(journal)
    edit_q, _s = _add_intent(journal, task, "ADMIT")
    edit_c, seq_c = _add_intent(journal, task, "ADMIT")
    edit_x, seq_x = _add_intent(journal, task, "ADMIT")
    _add_intent(journal, task, "REJECT")
    record_queue_claim(journal, task["seq"], edit_c, seq_c,
                       drainer_id="drainer-1", lease_seconds=300)
    record_queue_claim(journal, task["seq"], edit_x, seq_x,
                       drainer_id="drainer-1", lease_seconds=300)
    _add_outcome(journal, edit_x)

    fold = fold_queue(journal, task_seq=task["seq"])
    assert set(fold.intents) == {edit_q, edit_c, edit_x,
                                 fold.order[-1]}
    summary = queue_summary(fold, task["seq"])
    assert summary["task_seq"] == task["seq"]
    assert summary["counts"] == {
        "queued": 1, "claimed": 1, "executed": 1, "rejected": 1}
    assert summary["depth"] == 2  # queued + claimed (+ held: none)
    assert summary["outcome_unknown"] == []
    assert summary["held"] == []
    assert summary["integrity_faults"] == []


# -- (7) prefix stability -----------------------------------------------------------

def test_fold_of_prefix_is_stable(journal):
    task = _task(journal)
    _batch, block = _batch_queue_block(journal, task)
    edit_a, seq_a = _add_intent(journal, task, "ADMIT", queue=block)
    edit_b, seq_b = _add_intent(journal, task, "ADMIT", queue=block)

    fold = fold_queue(journal)
    assert fold.intents[edit_a].state == "queued"
    assert fold.intents[edit_b].state == "queued"

    record_queue_claim(journal, task["seq"], edit_a, seq_a,
                       drainer_id="drainer-1", lease_seconds=300)
    fold = fold_queue(journal)
    assert fold.intents[edit_a].state == "claimed"
    assert fold.intents[edit_b].state == "queued"

    _add_outcome(journal, edit_a)
    fold = fold_queue(journal)
    assert fold.intents[edit_a].state == "executed"
    assert fold.intents[edit_b].state == "queued"

    refusal = record_queue_refusal(
        journal, task["seq"], edit_b, seq_b,
        cause="stale-admission", claim_seq=None)
    assert refusal["kind"] == "queue_refusal"
    assert refusal["claim_seq"] is None
    fold = fold_queue(journal)
    state_b = fold.intents[edit_b]
    assert state_b.state == "refused-at-claim"
    assert state_b.refusal_cause == "stale-admission"
    assert state_b.outcome_status == "refused"
    assert state_b.fault is None
    # The earlier intent's folded state is untouched by the
    # later appends — a prefix fold is the state as of then.
    assert fold.intents[edit_a].state == "executed"


# -- compensation annotation ----------------------------------------------------------

def test_compensation_is_annotation_only(journal):
    task = _task(journal)
    edit_a, _seq_a = _add_intent(journal, task, "ADMIT")
    _add_outcome(journal, edit_a)
    edit_fix, _seq_fix = _add_intent(
        journal, task, "ADMIT", tool="fs_write",
        parents=[task["ref"], edit_a])
    state = fold_queue(journal).intents[edit_a]
    assert state.compensated_by == edit_fix
    assert state.state == "executed"
