"""SI-51 (SI-50 §16 Q-1, §17): Gaia Q-1 queue drain — admission
without execution, the serial drainer, and batched perception.

The miniature bundle fixture is copied from
``tests/test_gaia_b3.py``'s mini fixture (``init_bundle`` embedded
+ ``seed_tiny_bundle`` + a work dir); the gate context mirrors
B-3's ``_live_ctx`` (a :class:`ToolContext` with
``tool_live_execution=True``) over the task fixture from
:func:`si.gaia.shadow.record_task_fixture`, and the policy is
``si.gaia.corpus.corpus_policy``. Coverage, against §17:

- a batch of three fs_writes is admitted *without* executing
  (no files, execution None, self-edits pending), then one drain
  executes all three, every outcome preceded in journal sequence
  by its claim naming the same intent (§17.3);
- a mixed batch: an unknown-tool REJECT and an external
  ESCALATE are never claimed and never executed; the ADMIT
  member drains (§17.2, §17.4);
- ordering keys (§7.2): two writes to the same path share a key
  and drain in journal sequence order (the second write's bytes
  are the final content); a different-key write drains too
  (§17.5);
- claim/intent mismatch (§5.3): a claim journaled naming a
  wrong intent sequence is the executor's refused ticket —
  refusal journaled, detail naming the mismatch, nothing written;
- recovery (§13): an expired read claim under a dead drainer is
  recovery-re-claimed (``recovery_of`` == the original claim
  seq) and then executed with exactly the journaled args; a
  queued (never claimed) intent is drained by a restarted
  drainer; an unexpired foreign claim is left alone, and an
  expired write claim is deferred, never marked
  outcome-unknown in Q-1;
- the leash fold (§11, §17.8): a queue stop journaled before
  any drain claims nothing (``stop-in-force``) until a resume
  is journaled, after which the drain proceeds;
- batched perception (§10, §17.6): a two-intent drain returns
  ONE observation set, two provenance-complete observations in
  outcome-sequence order, with the queue summary attached;
- claimable (§7.1) as a unit: per-key FIFO, Q-1 ``depends_on``
  satisfaction (any terminal state), the stop fold, and a
  missing dependency failing closed; Q-1 concurrency is 1.
"""
import os

import pytest

from si.bundle import init_bundle, load_bundle
from si.gaia.context import ToolContext
from si.gaia.corpus import corpus_policy
from si.gaia.drain import (
    Q1_CONCURRENCY,
    Drainer,
    admit_tool_intent,
    build_observation_set,
)
from si.gaia.executor import Executor
from si.gaia.queue import (
    QueueFold,
    QueueIntentState,
    fold_queue,
    record_queue_batch,
    record_queue_claim,
    record_queue_resume,
    record_queue_stop,
)
from si.gaia.shadow import record_task_fixture
from si.gate import Gate, Intent
from si.journal import open_journal
from si.tiny_seed import seed_tiny_bundle


# -- fixtures (copied from tests/test_gaia_b3.py's mini fixture) -------------

@pytest.fixture()
def mini(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "mini.si"), name="gaia-q1",
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


def _setup(journal, work):
    """The task fixture + a live gate context (B-3's _live_ctx
    shape, with the fixture's real seq) + the corpus gate."""
    fixture = record_task_fixture(journal, "Q-1 drain scenario", work)
    gate = Gate(corpus_policy(work))
    ctx = ToolContext(tool_work_root=work,
                      tool_task_seq=fixture["seq"],
                      tool_live_execution=True)
    return fixture, gate, ctx


def _admit(journal, gate, ctx, fixture, tool, args, batch_seq=None,
           **kw):
    intent = Intent(kind="tool", tool=tool, args=args,
                    reason="Carry out the scripted Q-1 task step",
                    parents=[fixture["ref"]],
                    source_seqs=[fixture["seq"]], **kw)
    return admit_tool_intent(journal, gate, ctx, intent,
                             batch_seq=batch_seq)


def _batch(journal, fixture, size):
    return record_queue_batch(journal, fixture["seq"], fixture["ref"],
                              declared_size=size)


def _drainer(journal, bundle_dir, manifest, work, **kw):
    return Drainer(journal, bundle_dir, manifest,
                   corpus_policy(work), work, **kw)


def _state(journal, edit_id, task_seq):
    return fold_queue(journal, task_seq=task_seq).intents[edit_id]


def _read(path):
    with open(path) as f:
        return f.read()


# -- (1) batch admission does not execute; one drain executes all -----------

def test_batch_of_three_writes_admitted_without_executing_then_drain(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture, gate, ctx = _setup(journal, work)
        batch = _batch(journal, fixture, 3)
        names = ["one.txt", "two.txt", "three.txt"]
        outs = [
            _admit(journal, gate, ctx, fixture, "fs_write",
                   {"path": name, "content": f"content of {name}\n"},
                   batch_seq=batch["seq"])
            for name in names
        ]
        # Admission is enqueueing (§4.1) — and nothing more: no
        # execution, no files, self-edits left pending.
        for out in outs:
            assert out["verdict"] == "ADMIT"
            assert out["execution"] is None
            assert out["gate_report"]["queue"]["batch_seq"] == \
                batch["seq"]
        for name in names:
            assert not os.path.exists(os.path.join(work, name))
        edits = journal.backend.find("self_edits")
        by_id = {row["edit_id"]: row for row in edits}
        for out in outs:
            assert by_id[out["edit_id"]]["result"] == "pending"
            assert _state(journal, out["edit_id"],
                          fixture["seq"]).state == "queued"

        drainer = _drainer(journal, bundle_dir, manifest, work)
        res = drainer.drain_task(fixture["seq"])
        assert res["sync_reason"] == "queue-empty"
        assert res["integrity_failure"] is False
        assert len(res["claims"]) == 3
        assert set(res["executions"]) == {o["edit_id"] for o in outs}
        for name in names:
            assert _read(os.path.join(work, name)) == \
                f"content of {name}\n"

        # §17.3: every outcome is preceded, in journal sequence,
        # by its claim naming the same intent.
        fold = fold_queue(journal, task_seq=fixture["seq"])
        claim_seqs_in_intent_order = []
        for out in outs:
            state = fold.intents[out["edit_id"]]
            assert state.state == "executed"
            assert state.execution["status"] == "completed"
            assert state.claim_drainer == drainer.drainer_id
            assert state.intent_seq < state.claim_seq \
                < state.outcome_seq
            claim_seqs_in_intent_order.append(state.claim_seq)
        assert res["summary"]["counts"] == {"executed": 3}
    finally:
        journal.close()


# -- (2) mixed batch: REJECT / ESCALATE members are never claimed -----------

def test_mixed_batch_reject_and_escalate_members_never_drain(mini, tmp_path):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture, gate, ctx = _setup(journal, work)
        batch = _batch(journal, fixture, 3)
        rejected = _admit(
            journal, gate, ctx, fixture, "fs_delete_everything",
            {"path": "."}, batch_seq=batch["seq"])
        outside = str(tmp_path / "outside-q1.txt")
        escalated = _admit(
            journal, gate, ctx, fixture, "fs_write",
            {"path": outside, "content": "never written\n"},
            batch_seq=batch["seq"])
        admitted = _admit(
            journal, gate, ctx, fixture, "fs_write",
            {"path": "inside.txt", "content": "the admitted write\n"},
            batch_seq=batch["seq"])

        # Each member journaled with its own verdict, in order
        # (§17.4); only the ADMIT member queued without executing.
        assert rejected["verdict"] == "REJECT"
        assert escalated["verdict"] == "ESCALATE"
        assert escalated["effective_class"] == "external"
        assert admitted["verdict"] == "ADMIT"
        assert admitted["execution"] is None
        assert rejected["execution"] is not None
        assert escalated["execution"] is not None
        assert not os.path.exists(outside)
        assert not os.path.exists(os.path.join(work, "inside.txt"))

        drainer = _drainer(journal, bundle_dir, manifest, work)
        res = drainer.drain_task(fixture["seq"])
        assert res["sync_reason"] == "queue-empty"
        # The drain claimed and executed exactly the ADMIT member.
        assert set(res["executions"]) == {admitted["edit_id"]}
        assert len(res["claims"]) == 1
        assert res["claims"][0]["edit_id"] == admitted["edit_id"]
        assert _read(os.path.join(work, "inside.txt")) == \
            "the admitted write\n"
        assert not os.path.exists(outside)

        fold = fold_queue(journal, task_seq=fixture["seq"])
        for out, expected in ((rejected, "rejected"),
                              (escalated, "escalated")):
            state = fold.intents[out["edit_id"]]
            assert state.state == expected
            assert state.claim_seq is None
        assert res["summary"]["counts"] == {
            "rejected": 1, "escalated": 1, "executed": 1}
    finally:
        journal.close()


# -- (3) ordering keys: per-key FIFO in journal sequence order ---------------

def test_ordering_keys_drain_same_path_writes_in_seq_order(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture, gate, ctx = _setup(journal, work)
        batch = _batch(journal, fixture, 3)
        first = _admit(journal, gate, ctx, fixture, "fs_write",
                       {"path": "same.txt", "content": "first\n"},
                       batch_seq=batch["seq"])
        second = _admit(journal, gate, ctx, fixture, "fs_write",
                        {"path": "same.txt", "content": "second\n"},
                        batch_seq=batch["seq"])
        other = _admit(journal, gate, ctx, fixture, "fs_write",
                       {"path": "other.txt", "content": "other\n"},
                       batch_seq=batch["seq"])

        fold = fold_queue(journal, task_seq=fixture["seq"])
        key_first = fold.intents[first["edit_id"]].ordering_key
        key_second = fold.intents[second["edit_id"]].ordering_key
        key_other = fold.intents[other["edit_id"]].ordering_key
        # The gate priced the key (§7.2): same resolved target,
        # same key; a different target, a different key.
        assert key_first is not None and key_first == key_second
        assert "same.txt" in key_first
        assert key_other is not None and key_other != key_first

        res = _drainer(journal, bundle_dir, manifest,
                       work).drain_task(fixture["seq"])
        assert res["sync_reason"] == "queue-empty"
        # FIFO per key: the second write lands last on its path.
        assert _read(os.path.join(work, "same.txt")) == "second\n"
        assert _read(os.path.join(work, "other.txt")) == "other\n"
        # And the journal shows the serial order: outcomes in
        # intent-sequence order across the whole task (§17.5).
        fold = fold_queue(journal, task_seq=fixture["seq"])
        states = [fold.intents[o["edit_id"]]
                  for o in (first, second, other)]
        assert [s.intent_seq for s in states] == \
            sorted(s.intent_seq for s in states)
        assert [s.outcome_seq for s in states] == \
            sorted(s.outcome_seq for s in states)
    finally:
        journal.close()


# -- (4) claim/intent mismatch refuses at the executor -----------------------

def test_claim_intent_mismatch_refuses_and_writes_nothing(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture, gate, ctx = _setup(journal, work)
        out = _admit(journal, gate, ctx, fixture, "fs_write",
                     {"path": "mismatch.txt", "content": "nope\n"})
        state = _state(journal, out["edit_id"], fixture["seq"])
        wrong_seq = state.intent_seq + 999
        # A claim journaled naming a wrong intent sequence for
        # this edit: the claim is the executor's ticket (§5.3),
        # and this ticket names another intent.
        claim = record_queue_claim(
            journal, fixture["seq"], out["edit_id"], wrong_seq,
            drainer_id="drainer-test", lease_seconds=300)
        executor = Executor(bundle_dir, manifest, journal,
                            corpus_policy(work), work)
        record = executor.execute_journaled(
            out["edit_id"], expected_effective_class="write",
            claim_intent_seq=wrong_seq, claim_seq=claim["seq"])
        assert record["status"] == "refused"
        assert "claim/intent mismatch" in record["detail"]
        assert not os.path.exists(os.path.join(work, "mismatch.txt"))
        edits = journal.backend.find(
            "self_edits", filter={"edit_id": out["edit_id"]})
        assert edits[0]["result"] == "refused"
    finally:
        journal.close()


# -- (5) recovery: expired read claim is re-claimed, then executed -----------

def test_recovery_reclaims_expired_read_claim_and_drain_executes(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture, gate, ctx = _setup(journal, work)
        out = _admit(journal, gate, ctx, fixture, "fs_read",
                     {"path": "notes/plan.md"})
        state = _state(journal, out["edit_id"], fixture["seq"])
        original = record_queue_claim(
            journal, fixture["seq"], out["edit_id"], state.intent_seq,
            drainer_id="drainer-dead", lease_seconds=0)

        drainer = _drainer(journal, bundle_dir, manifest, work,
                           drainer_id="drainer-new")
        rec = drainer.recover(task_seq=fixture["seq"])
        assert rec["foreign"] == [] and rec["deferred"] == []
        assert len(rec["reclaimed"]) == 1
        reclaim = rec["reclaimed"][0]
        assert reclaim["recovery_of"] == original["seq"]
        assert reclaim["original_claim_seq"] == original["seq"]
        folded = _state(journal, out["edit_id"], fixture["seq"])
        assert folded.state == "claimed"
        assert folded.recovery_of == original["seq"]
        assert folded.claim_seq == reclaim["seq"]
        assert folded.claim_drainer == "drainer-new"

        res = drainer.drain_task(fixture["seq"])
        assert res["sync_reason"] == "queue-empty"
        execution = res["executions"][out["edit_id"]]
        assert execution["status"] == "completed"
        # Executed with exactly the journaled args: the read of
        # notes/plan.md returns that file's content.
        assert "# plan" in execution["result"]["content"]
    finally:
        journal.close()


def test_recovery_foreign_unexpired_claim_is_left_alone(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture, gate, ctx = _setup(journal, work)
        out = _admit(journal, gate, ctx, fixture, "fs_read",
                     {"path": "notes/plan.md"})
        state = _state(journal, out["edit_id"], fixture["seq"])
        claim = record_queue_claim(
            journal, fixture["seq"], out["edit_id"], state.intent_seq,
            drainer_id="drainer-other", lease_seconds=300)
        drainer = _drainer(journal, bundle_dir, manifest, work,
                           drainer_id="drainer-new")
        rec = drainer.recover(task_seq=fixture["seq"])
        assert rec["foreign"] == [claim["seq"]]
        assert rec["reclaimed"] == [] and rec["deferred"] == []
        folded = _state(journal, out["edit_id"], fixture["seq"])
        assert folded.state == "claimed"
        assert folded.claim_drainer == "drainer-other"
        assert folded.claim_seq == claim["seq"]
    finally:
        journal.close()


def test_recovery_expired_write_claim_is_deferred_not_unknown(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture, gate, ctx = _setup(journal, work)
        out = _admit(journal, gate, ctx, fixture, "fs_write",
                     {"path": "deferred.txt", "content": "later\n"})
        state = _state(journal, out["edit_id"], fixture["seq"])
        record_queue_claim(
            journal, fixture["seq"], out["edit_id"], state.intent_seq,
            drainer_id="drainer-dead", lease_seconds=0)
        drainer = _drainer(journal, bundle_dir, manifest, work,
                           drainer_id="drainer-new")
        rec = drainer.recover(task_seq=fixture["seq"])
        # Write class is Q-3 territory (§13.3): deferred, left
        # claimed, NOT marked outcome-unknown, never re-executed.
        assert rec["deferred"] == [state.intent_seq]
        assert rec["reclaimed"] == [] and rec["foreign"] == []
        folded = _state(journal, out["edit_id"], fixture["seq"])
        assert folded.state == "claimed"
        assert folded.fault is None
        assert not os.path.exists(os.path.join(work, "deferred.txt"))
    finally:
        journal.close()


# -- (6) recovery of the queued: a restarted drainer drains it ---------------

def test_restarted_drainer_drains_a_queued_intent(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture, gate, ctx = _setup(journal, work)
        out = _admit(journal, gate, ctx, fixture, "fs_write",
                     {"path": "restart.txt", "content": "resumed\n"})
        assert _state(journal, out["edit_id"],
                      fixture["seq"]).state == "queued"
        # A new Drainer instance — the restarted process — folds
        # the journal alone and drains what was queued (§17.9).
        restarted = _drainer(journal, bundle_dir, manifest, work,
                             drainer_id="drainer-restarted")
        res = restarted.drain_task(fixture["seq"])
        assert res["sync_reason"] == "queue-empty"
        assert res["executions"][out["edit_id"]]["status"] == \
            "completed"
        assert _read(os.path.join(work, "restart.txt")) == "resumed\n"
    finally:
        journal.close()


# -- (7) the leash fold: stop holds, resume releases --------------------------

def test_queue_stop_holds_the_drain_until_resume(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture, gate, ctx = _setup(journal, work)
        out = _admit(journal, gate, ctx, fixture, "fs_write",
                     {"path": "held.txt", "content": "held\n"})
        record_queue_stop(journal, fixture["seq"],
                          cause="operator-stop",
                          detail="the leash, pulled before any drain")
        drainer = _drainer(journal, bundle_dir, manifest, work)
        res = drainer.drain_task(fixture["seq"])
        # From the fold alone (§11.3): nothing claimed, nothing
        # executed, the intent folds to held.
        assert res["sync_reason"] == "stop-in-force"
        assert res["claims"] == [] and res["executions"] == {}
        assert not os.path.exists(os.path.join(work, "held.txt"))
        folded = _state(journal, out["edit_id"], fixture["seq"])
        assert folded.state == "held"
        assert res["summary"]["held"] == [folded.intent_seq]

        record_queue_resume(journal, fixture["seq"],
                            detail="operator resumed the task")
        res = drainer.drain_task(fixture["seq"])
        assert res["sync_reason"] == "queue-empty"
        assert _read(os.path.join(work, "held.txt")) == "held\n"
        assert _state(journal, out["edit_id"],
                      fixture["seq"]).state == "executed"
    finally:
        journal.close()


# -- (8) batched perception: one set, provenance-complete, in order ----------

def test_drain_returns_one_batched_observation_set(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture, gate, ctx = _setup(journal, work)
        batch = _batch(journal, fixture, 2)
        outs = [
            _admit(journal, gate, ctx, fixture, "fs_write",
                   {"path": f"perc-{i}.txt", "content": f"{i}\n"},
                   batch_seq=batch["seq"])
            for i in range(2)
        ]
        drainer = _drainer(journal, bundle_dir, manifest, work)
        res = drainer.drain_task(fixture["seq"])

        perception = res["perception"]
        assert isinstance(perception, dict)  # ONE set, not a list
        assert perception["kind"] == "queue_perception"
        assert perception["task_seq"] == fixture["seq"]
        assert perception["sync_point"] == res["sync_reason"] \
            == "queue-empty"
        assert perception["queue_summary"] == res["summary"]
        assert res["summary"]["counts"] == {"executed": 2}

        observations = perception["observations"]
        assert len(observations) == 2
        outcome_seqs = [o["outcome_seq"] for o in observations]
        assert outcome_seqs == sorted(outcome_seqs)
        # Serial drain: outcome order is the intents' order.
        assert [o["edit_id"] for o in observations] == \
            [o["edit_id"] for o in outs]
        for obs in observations:
            # Provenance-complete (§10.1): every sequence named.
            assert obs["kind"] == "tool_result"
            assert obs["source"] == "tool:fs_write"
            assert obs["task_seq"] == fixture["seq"]
            assert obs["tool"] == "fs_write"
            assert obs["verdict"] == "ADMIT"
            for key in ("intent_seq", "claim_seq", "outcome_seq",
                        "edit_id", "execution"):
                assert obs[key] is not None, key
            assert obs["intent_seq"] < obs["claim_seq"] \
                < obs["outcome_seq"]
            assert obs["execution"]["status"] == "completed"

        # The module-level builder, handed the outcome seqs in
        # reverse, still emits them in outcome-sequence order.
        fold = fold_queue(journal, task_seq=fixture["seq"])
        direct = build_observation_set(
            journal, fold, fixture["seq"],
            list(reversed(outcome_seqs)), "queue-empty")
        assert [o["outcome_seq"] for o in direct["observations"]] \
            == outcome_seqs
        assert direct["queue_summary"] == res["summary"]
    finally:
        journal.close()


# -- claimable (§7.1) as a unit, and Q-1's fixed concurrency ------------------

def _qs(edit_id, seq, task_seq=99, state="queued", key=None,
        deps=None):
    return QueueIntentState(
        edit_id=edit_id, intent_seq=seq, task_seq=task_seq,
        tool="fs_write", verdict="ADMIT", effective_class="write",
        ordering_key=key, depends_on=deps or [], state=state)


def _fold_of(*states, stop=False):
    return QueueFold(
        intents={s.edit_id: s for s in states},
        by_seq={s.intent_seq: s for s in states},
        tasks={99: {"stop_in_force": stop, "stop_seq": None,
                    "resume_seq": None, "integrity_faults": []}},
        order=[s.edit_id for s in states])


def test_claimable_fifo_dependencies_and_stop():
    assert Q1_CONCURRENCY == 1
    drainer = object.__new__(Drainer)  # claimable reads no fields

    done = _qs("e-done", 10, state="executed")
    dependent = _qs("e-dep", 12,
                    deps=[{"seq": 10, "requirement": "outcome"}])
    fold = _fold_of(done, dependent)
    # Q-1 depends_on semantics: the dependency concluded in ANY
    # terminal state — satisfied (requirement semantics are Q-2).
    assert drainer.claimable(dependent, fold) is True
    rejected_dep = _qs("e-dep2", 13,
                       deps=[{"seq": 10, "requirement": "success"}])
    done.state = "rejected"  # also terminal, also satisfies in Q-1
    assert drainer.claimable(rejected_dep, _fold_of(done, rejected_dep))

    # A dependency naming no folded intent fails closed.
    orphan = _qs("e-orphan", 14, deps=[{"seq": 777,
                                        "requirement": "outcome"}])
    assert drainer.claimable(orphan, _fold_of(orphan)) is False
    # A non-terminal dependency blocks.
    pending = _qs("e-pending", 15, state="queued")
    blocked = _qs("e-blocked", 16, deps=[15])
    assert drainer.claimable(blocked, _fold_of(pending, blocked)) \
        is False

    # Per-key FIFO: the earlier queued intent on a key is
    # claimable; the later one on the same key is not, until the
    # earlier concludes. A different key is unaffected.
    first = _qs("e-first", 20, key="fs:/w/a.txt")
    second = _qs("e-second", 21, key="fs:/w/a.txt")
    different = _qs("e-diff", 22, key="fs:/w/b.txt")
    fold = _fold_of(first, second, different)
    assert drainer.claimable(first, fold) is True
    assert drainer.claimable(second, fold) is False
    assert drainer.claimable(different, fold) is True
    first.state = "executed"
    assert drainer.claimable(second, fold) is True

    # Not queued, not claimable; stop in force, not claimable.
    assert drainer.claimable(first, fold) is False
    queued = _qs("e-q", 30)
    assert drainer.claimable(queued, _fold_of(queued)) is True
    assert drainer.claimable(queued, _fold_of(queued, stop=True)) \
        is False


# -- §12.4.4 integrity freeze (coordinator integration test) -------


def test_integrity_failure_freezes_the_drain(mini, tmp_path):
    """An execution-integrity refusal mid-drain (here the B-3
    symlink TOCTOU: the priced directory is swapped for a symlink
    out of the work root between admission and execution) freezes
    the drain task-wide: the refusal is journaled, the drainer
    reports the freeze, and the next queued intent is NEVER
    claimed (§12.4.4 / §17.7)."""
    import shutil

    bundle_dir, work = mini
    target = os.path.join(work, "suite")
    os.makedirs(target)
    with open(os.path.join(target, "test_ok.py"), "w") as f:
        f.write("def test_ok():\n    assert True\n")
    outside = tmp_path / "outside-suite"
    outside.mkdir()
    manifest, journal = _open(bundle_dir)
    try:
        fixture, gate, ctx = _setup(journal, work)
        batch = _batch(journal, fixture, 2)
        first = _admit(journal, gate, ctx, fixture, "test_run",
                       {"path": "suite"}, batch_seq=batch["seq"])
        second = _admit(journal, gate, ctx, fixture, "fs_write",
                        {"path": "after.txt", "content": "A"},
                        batch_seq=batch["seq"])
        assert first["verdict"] == second["verdict"] == "ADMIT"
        # The swap: the priced directory becomes a symlink out.
        shutil.rmtree(target)
        os.symlink(str(outside), target)
        drainer = _drainer(journal, bundle_dir, manifest, work)
        res = drainer.drain_task(fixture["seq"])
        assert res["integrity_failure"] is True
        assert res["sync_reason"] == "integrity-failure"
        execution = res["executions"][first["edit_id"]]
        assert execution["status"] == "refused"
        assert "TOCTOU" in execution["detail"]
        # Frozen: the second intent was never claimed or executed.
        assert second["edit_id"] not in res["executions"]
        claims = journal.backend.find(
            "experiences", filter={"kind": "queue_claim"})
        assert [c["record"]["edit_id"] for c in claims] == \
            [first["edit_id"]]
        assert not os.path.exists(os.path.join(work, "after.txt"))
        state = _state(journal, second["edit_id"], fixture["seq"])
        assert state.state == "queued"
    finally:
        journal.close()
