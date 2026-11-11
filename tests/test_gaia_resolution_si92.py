"""SI-92 (SI-41 §5.4; SI-50 §5.2, §12.4 rule 2): Gaia
escalation resolution — the operator's answer to a gate
ESCALATE, on SI-53's §13.4 disposition pattern.

Coverage:

- an UNRESOLVED escalation never executes: the drainer does
  not claim it, :func:`execute_resolved_escalation` refuses,
  and the executor refuses even with a fabricated ticket;
- ``reject`` resolution is a journaled operator act (source,
  operator, basis recorded) after which the intent is dead —
  still folded ``escalated``, annotated, never executed;
- ``admit-after-review`` folds the intent to ``queued``; the
  one execution runs exactly the journaled args, under a
  journaled claim, with the resolution sequence on the
  execution record, journaled in the order resolution <
  claim < outcome; a second execution attempt is refused —
  the resolution buys exactly one run;
- the drainer reaches the same path on a later drain (fold
  ``queued`` -> claim -> executor ticket), and the queue
  summary / state view carry the resolution;
- double resolution, resolution of a non-escalated or
  unknown intent, an invalid resolution value, and an
  anonymous resolution are all refused and journal NOTHING;
- the executor's ticket is verified from the journal: a
  wrong sequence refuses (and that journaled refusal is the
  intent's outcome — the one shot is spent, never silently
  re-runnable).
"""
import os

import pytest

from si.bundle import init_bundle, load_bundle
from si.gaia.context import ToolContext
from si.gaia.corpus import corpus_policy
from si.gaia.drain import Drainer, admit_tool_intent
from si.gaia.executor import Executor
from si.gaia.queue import (
    fold_queue,
    queue_state_view,
    record_queue_batch,
)
from si.gaia.resolution import (
    RESOLUTIONS,
    execute_resolved_escalation,
    resolve_escalation,
)
from si.gaia.shadow import record_task_fixture
from si.gate import Gate, Intent
from si.journal import open_journal
from si.tiny_seed import seed_tiny_bundle


ESCALATE_WRITE = {"read": "admit", "write": "escalate",
                  "execute": "admit", "external": "escalate"}
DENY_WRITE = {"read": "admit", "write": "deny",
              "execute": "admit", "external": "escalate"}
CONTENT = "written after operator review\n"


@pytest.fixture()
def mini(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "mini.si"), name="gaia-si92",
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    work = tmp_path / "work"
    work.mkdir()
    (work / "notes").mkdir()
    (work / "notes" / "plan.md").write_text("# plan\nstep one\n")
    return bundle_dir, str(work)


def _open(bundle_dir):
    manifest = load_bundle(bundle_dir)
    return manifest, open_journal(bundle_dir, manifest)


def _setup(journal, work, class_policy=ESCALATE_WRITE):
    fixture = record_task_fixture(journal, "SI-92 scenario", work)
    policy = corpus_policy(work, {"tool_class_policy": class_policy})
    gate = Gate(policy)
    ctx = ToolContext(tool_work_root=work,
                      tool_task_seq=fixture["seq"],
                      tool_live_execution=True)
    return fixture, gate, ctx, policy


def _admit(journal, gate, ctx, fixture, tool, args, batch_seq=None):
    intent = Intent(kind="tool", tool=tool, args=args,
                    reason="Carry out the scripted SI-92 task step",
                    parents=[fixture["ref"]],
                    source_seqs=[fixture["seq"]])
    return admit_tool_intent(journal, gate, ctx, intent,
                             batch_seq=batch_seq)


def _intent_seq(journal, edit_id):
    rows = journal.backend.find("self_edits",
                                filter={"edit_id": edit_id}, limit=1)
    return rows[0]["seq"]


def _state(journal, edit_id, task_seq=None):
    return fold_queue(journal, task_seq=task_seq).intents[edit_id]


def _resolution_rows(journal):
    return journal.backend.find(
        "experiences", filter={"kind": "escalation_resolution"}) or []


def _escalated_write(mini):
    """A batch-admitted fs_write the gate ESCALATEs (write
    class priced escalate). Returns everything the tests need."""
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    fixture, gate, ctx, policy = _setup(journal, work)
    batch = record_queue_batch(journal, fixture["seq"], fixture["ref"],
                               declared_size=1)
    out = _admit(journal, gate, ctx, fixture, "fs_write",
                 {"path": "notes/new.md", "content": CONTENT},
                 batch_seq=batch["seq"])
    assert out["verdict"] == "ESCALATE"
    seq = _intent_seq(journal, out["edit_id"])
    target = os.path.join(work, "notes", "new.md")
    assert not os.path.exists(target)
    return (bundle_dir, work, manifest, journal, fixture, policy,
            out["edit_id"], seq, target)


# -- unresolved escalations never execute ------------------------------------

def test_unresolved_escalation_never_executes(mini):
    (bundle_dir, work, manifest, journal, fixture, policy,
     edit_id, seq, target) = _escalated_write(mini)
    state = _state(journal, edit_id, fixture["seq"])
    assert state.state == "escalated"
    assert state.resolution is None

    # The drainer does not claim an escalated intent.
    res = Drainer(journal, bundle_dir, manifest, policy,
                  work).drain_task(fixture["seq"])
    assert edit_id not in res["executions"]
    assert not os.path.exists(target)

    # The resolution driver refuses: no resolution, no run.
    with pytest.raises(ValueError):
        execute_resolved_escalation(
            journal, bundle_dir, manifest, policy, work, seq)
    assert not os.path.exists(target)

    # The executor refuses even with a fabricated ticket, and
    # journals the refusal (a caller cannot mistake it for a run).
    runner = Executor(bundle_dir, manifest, journal, policy, work)
    record = runner.execute_journaled(edit_id, resolution_seq=999999)
    assert record["status"] == "refused"
    assert not os.path.exists(target)
    assert _state(journal, edit_id, fixture["seq"]).state == "escalated"


# -- reject: the intent dies, journaled ---------------------------------------

def test_resolve_reject_kills_intent(mini):
    (bundle_dir, work, manifest, journal, fixture, policy,
     edit_id, seq, target) = _escalated_write(mini)
    entry = resolve_escalation(journal, seq, "reject",
                               operator="mark",
                               detail="the write is not wanted")
    rows = _resolution_rows(journal)
    assert len(rows) == 1
    record = rows[0]["record"]
    assert record["source"] == "operator"
    assert record["operator"] == "mark"
    assert record["resolution"] == "reject"
    assert record["edit_id"] == edit_id
    assert record["intent_seq"] == seq
    assert record["basis"]  # the §5.4 basis is recorded

    state = _state(journal, edit_id, fixture["seq"])
    assert state.state == "escalated"  # history not rewritten
    assert state.resolution == "reject"
    assert state.resolution_seq == entry["seq"]

    with pytest.raises(ValueError):
        execute_resolved_escalation(
            journal, bundle_dir, manifest, policy, work, seq)
    res = Drainer(journal, bundle_dir, manifest, policy,
                  work).drain_task(fixture["seq"])
    assert edit_id not in res["executions"]
    assert not os.path.exists(target)


# -- admit-after-review: exactly one run of the journaled args ----------------

def test_resolve_admit_executes_journaled_args_once(mini):
    (bundle_dir, work, manifest, journal, fixture, policy,
     edit_id, seq, target) = _escalated_write(mini)
    entry = resolve_escalation(journal, seq, "admit-after-review",
                               operator="mark",
                               detail="reviewed the journaled write")
    state = _state(journal, edit_id, fixture["seq"])
    assert state.state == "queued"  # the one transition out
    assert state.resolution == "admit-after-review"
    assert state.resolution_seq == entry["seq"]
    assert state.resolution_operator == "mark"

    result = execute_resolved_escalation(
        journal, bundle_dir, manifest, policy, work, seq)
    execution = result["execution"]
    assert execution["status"] == "completed"
    assert execution["resolution_seq"] == entry["seq"]
    with open(target) as f:
        assert f.read() == CONTENT  # exactly the journaled args

    # Journal order: resolution before claim before outcome.
    state = _state(journal, edit_id, fixture["seq"])
    assert state.state == "executed"
    assert state.outcome_status == "completed"
    assert entry["seq"] < result["claim"]["seq"] < state.outcome_seq

    # Exactly once: a second attempt is refused, file untouched.
    with pytest.raises(ValueError):
        execute_resolved_escalation(
            journal, bundle_dir, manifest, policy, work, seq)
    with open(target) as f:
        assert f.read() == CONTENT


def test_drain_executes_resolved_escalation(mini):
    (bundle_dir, work, manifest, journal, fixture, policy,
     edit_id, seq, target) = _escalated_write(mini)
    resolve_escalation(journal, seq, "admit-after-review",
                       operator="mark")
    res = Drainer(journal, bundle_dir, manifest, policy,
                  work).drain_task(fixture["seq"])
    assert edit_id in res["executions"]
    with open(target) as f:
        assert f.read() == CONTENT
    fold = fold_queue(journal, task_seq=fixture["seq"])
    assert fold.intents[edit_id].state == "executed"
    assert fold.intents[edit_id].resolution == "admit-after-review"
    summary = res["perception"]["queue_summary"]
    assert summary["resolutions"] == {seq: "admit-after-review"}
    view = queue_state_view(fold, fixture["seq"])
    intent_view = [i for i in view["intents"]
                   if i["edit_id"] == edit_id][0]
    assert intent_view["resolution"] == "admit-after-review"
    assert intent_view["resolution_seq"] is not None


def test_executor_runs_escalated_intent_under_verified_ticket(mini):
    (bundle_dir, work, manifest, journal, fixture, policy,
     edit_id, seq, target) = _escalated_write(mini)
    entry = resolve_escalation(journal, seq, "admit-after-review",
                               operator="mark")
    runner = Executor(bundle_dir, manifest, journal, policy, work)
    record = runner.execute_journaled(
        edit_id, expected_effective_class="write",
        resolution_seq=entry["seq"])
    assert record["status"] == "completed"
    assert record["resolution_seq"] == entry["seq"]
    with open(target) as f:
        assert f.read() == CONTENT


def test_executor_wrong_ticket_spends_the_one_shot(mini):
    (bundle_dir, work, manifest, journal, fixture, policy,
     edit_id, seq, target) = _escalated_write(mini)
    entry = resolve_escalation(journal, seq, "admit-after-review",
                               operator="mark")
    runner = Executor(bundle_dir, manifest, journal, policy, work)
    record = runner.execute_journaled(
        edit_id, expected_effective_class="write",
        resolution_seq=entry["seq"] + 999)
    assert record["status"] == "refused"
    assert not os.path.exists(target)
    # The journaled refusal is the intent's outcome: even the
    # right ticket now refuses — a resolved escalation is never
    # silently re-runnable.
    record = runner.execute_journaled(
        edit_id, expected_effective_class="write",
        resolution_seq=entry["seq"])
    assert record["status"] == "refused"
    assert not os.path.exists(target)


# -- fail-closed resolution acts ------------------------------------------------

def test_double_resolution_refused_journals_nothing(mini):
    (bundle_dir, work, manifest, journal, fixture, policy,
     edit_id, seq, target) = _escalated_write(mini)
    resolve_escalation(journal, seq, "admit-after-review",
                       operator="mark")
    tip_before = journal.tip()[0]
    with pytest.raises(ValueError):
        resolve_escalation(journal, seq, "reject", operator="mark")
    assert journal.tip()[0] == tip_before
    assert len(_resolution_rows(journal)) == 1
    state = _state(journal, edit_id, fixture["seq"])
    assert state.resolution == "admit-after-review"


def test_resolution_refusals_journal_nothing(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    fixture, gate, ctx, policy = _setup(journal, work)

    with pytest.raises(ValueError):
        resolve_escalation(journal, 999999, "admit-after-review",
                           operator="mark")

    admitted = _admit(journal, gate, ctx, fixture, "fs_list",
                      {"path": "notes"})
    assert admitted["verdict"] == "ADMIT"
    with pytest.raises(ValueError):
        resolve_escalation(
            journal, _intent_seq(journal, admitted["edit_id"]),
            "admit-after-review", operator="mark")

    _, deny_gate, _, _ = _setup(journal, work, DENY_WRITE)
    rejected = _admit(journal, deny_gate, ctx, fixture, "fs_write",
                      {"path": "notes/no.md", "content": "x\n"})
    assert rejected["verdict"] == "REJECT"
    with pytest.raises(ValueError):
        resolve_escalation(
            journal, _intent_seq(journal, rejected["edit_id"]),
            "admit-after-review", operator="mark")

    escalated = _admit(journal, gate, ctx, fixture, "fs_write",
                       {"path": "notes/new.md", "content": CONTENT})
    assert escalated["verdict"] == "ESCALATE"
    eseq = _intent_seq(journal, escalated["edit_id"])
    for bad in ("mark-executed", "admit", "", None):
        assert bad not in RESOLUTIONS
        with pytest.raises(ValueError):
            resolve_escalation(journal, eseq, bad, operator="mark")
    with pytest.raises(ValueError):
        resolve_escalation(journal, eseq, "admit-after-review",
                           operator="  ")
    assert _resolution_rows(journal) == []
