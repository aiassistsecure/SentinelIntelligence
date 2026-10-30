"""SI-52 (SI-50 §16 Q-2, §§7.3, 8, 10.2, 12.2): Gaia Q-2 —
``depends_on`` requirement semantics, barrier sync points,
declared ordering keys under concurrency, the bounded read
pool, claim-time residence refusals, and the §9 pipeline
exercised at the admission/drain boundary.

Coverage:

- requirement semantics (§7.3): an ``outcome`` dependent runs
  after its dependency concludes *any* way (here: a failed
  execution); a ``success`` dependent whose dependency failed
  is ``skipped`` — a ``queue_skip`` entry journaled with the
  cause and the dependency's outcome sequence — and never
  executes; skips cascade to their own ``success`` dependents;
- claimable as a unit: ``success`` is satisfied only by an
  executed intent whose outcome status is ``completed``;
- barrier (§10.2 point 2): a drain stops at a concluded barrier
  intent — sync_reason ``barrier``, the observation set holds
  only the barrier outcome, later intents stay queued for the
  next drain;
- residence (§12.2): with ``queue_max_residence_seconds`` at 0,
  a queued intent is refused at claim with cause
  ``stale-admission`` and never executes;
- the read pool (§8): four unkeyed reads execute concurrently
  (a threading.Barrier(4) inside the stub executor can only be
  satisfied if all four are in flight at once), the policy
  bound 2 caps the peak at 2, keyed reads and writes never
  overlap anything, claims still precede outcomes in journal
  sequence, and the journal hash chain verifies afterwards;
- policy resolution (§8.2): default 4, hard cap 16, malformed
  or sub-1 values resolve to 1 (serial — most restrictive);
- the pipeline (§9): while a drain is blocked inside an
  execution, a second batch is admitted (gated + journaled)
  through the real admission path — its journal sequences land
  between the first batch's claim and outcome sequences — and
  the second batch then drains normally;
- journal thread-safety (the Q-2 enabler, SI-50 §8.3: outcome
  journaling is serial even when executions overlap): many
  threads appending to one shared embedded journal leave a
  verified chain with unique sequences.
"""
import os
import threading
import time
from types import SimpleNamespace

import pytest

from si.bundle import init_bundle, load_bundle
from si.gaia.context import ToolContext
from si.gaia.corpus import corpus_policy
from si.gaia.drain import (
    Q2_MAX_CONCURRENT_READS_CAP,
    Q2_DEFAULT_MAX_CONCURRENT_READS,
    Drainer,
    admit_tool_intent,
)
from si.gaia.executor import record_execution_outcome
from si.gaia.queue import fold_queue, record_queue_batch
from si.gaia.shadow import record_task_fixture
from si.gate import Gate, Intent
from si.journal import open_journal
from si.tiny_seed import seed_tiny_bundle


# -- fixtures (copied from tests/test_gaia_q1_drain.py) ----------------------

@pytest.fixture()
def mini(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "mini.si"), name="gaia-q2",
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
    fixture = record_task_fixture(journal, "Q-2 drain scenario", work)
    gate = Gate(corpus_policy(work))
    ctx = ToolContext(tool_work_root=work,
                      tool_task_seq=fixture["seq"],
                      tool_live_execution=True)
    return fixture, gate, ctx


def _admit(journal, gate, ctx, fixture, tool, args, batch_seq=None,
           **kw):
    intent = Intent(kind="tool", tool=tool, args=args,
                    reason="Carry out the scripted Q-2 task step",
                    parents=[fixture["ref"]],
                    source_seqs=[fixture["seq"]], **kw)
    return admit_tool_intent(journal, gate, ctx, intent,
                             batch_seq=batch_seq)


def _admit_dep(journal, gate, ctx, fixture, tool, args, dep_seqs,
               **kw):
    """Admit an intent declaring dependencies on already
    journaled intents of this task: the gate prices
    ``depends_on`` against the journal context (§6.2), which
    the test supplies exactly as the task loop does."""
    ctx.tool_journal_tip = journal.tip()[0]
    ctx.tool_task_intent_seqs = set(dep_seqs)
    return _admit(journal, gate, ctx, fixture, tool, args, **kw)


def _batch(journal, fixture, size):
    return record_queue_batch(journal, fixture["seq"], fixture["ref"],
                              declared_size=size)


def _drainer(journal, bundle_dir, manifest, work, policy=None, **kw):
    return Drainer(journal, bundle_dir, manifest,
                   policy if policy is not None
                   else corpus_policy(work), work, **kw)


def _state(journal, edit_id, task_seq):
    return fold_queue(journal, task_seq=task_seq).intents[edit_id]


def _read(path):
    with open(path) as f:
        return f.read()


# -- the stub executor: controllable concurrency, real journaling ------------

class _StubExecutor:
    """A drainer executor stand-in that journals real outcomes
    through the executor's own outcome writer (so the fold and
    the hash chain see exactly what a real execution produces)
    while the test controls timing: an optional per-call delay,
    an optional threading.Barrier every call must reach (only
    satisfiable under true concurrency), and start/stop
    intervals per edit id for overlap assertions."""

    def __init__(self, journal, delay=0.0, barrier=None):
        self.journal = journal
        self.delay = delay
        self.barrier = barrier
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.intervals = {}

    def execute_journaled(self, edit_id,
                          expected_effective_class=None,
                          claim_intent_seq=None, claim_seq=None):
        started = time.monotonic()
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if self.barrier is not None:
                try:
                    self.barrier.wait(timeout=15)
                except threading.BrokenBarrierError:
                    pass
            if self.delay:
                time.sleep(self.delay)
            rows = self.journal.backend.find(
                "self_edits", filter={"edit_id": edit_id}, limit=1)
            tool = (rows[0].get("intent") or {}).get("tool", "")
            execution = {
                "tool": tool, "intent_id": edit_id,
                "class": expected_effective_class,
                "status": "completed", "result": {},
                "detail": "stub execution (Q-2 concurrency test)",
            }
            record_execution_outcome(
                self.journal, SimpleNamespace(tool=tool), edit_id,
                execution)
            self.journal.set_edit_result(edit_id, "executed")
            return execution
        finally:
            with self._lock:
                self.active -= 1
                self.intervals[edit_id] = (
                    started, time.monotonic())


def _overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


# -- policy resolution (§8.2) --------------------------------------------------

def test_max_concurrent_reads_resolution():
    assert Q2_DEFAULT_MAX_CONCURRENT_READS == 4
    assert Q2_MAX_CONCURRENT_READS_CAP == 16
    drainer = object.__new__(Drainer)  # the resolver reads policy only
    cases = [
        ({}, 4), ({"queue_max_concurrent_reads": 8}, 8),
        ({"queue_max_concurrent_reads": 99}, 16),
        ({"queue_max_concurrent_reads": 16}, 16),
        ({"queue_max_concurrent_reads": 1}, 1),
        ({"queue_max_concurrent_reads": 0}, 1),
        ({"queue_max_concurrent_reads": -3}, 1),
        ({"queue_max_concurrent_reads": "4"}, 1),
        ({"queue_max_concurrent_reads": True}, 1),
        ({"queue_max_concurrent_reads": None}, 1),
    ]
    for policy, expected in cases:
        drainer.policy = policy
        assert drainer.max_concurrent_reads() == expected, policy


# -- requirement semantics (§7.3), real executor --------------------------------

def test_success_dependent_skipped_outcome_dependent_runs(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture, gate, ctx = _setup(journal, work)
        batch = _batch(journal, fixture, 4)
        failer = _admit(journal, gate, ctx, fixture, "fs_read",
                        {"path": "missing.txt"},
                        batch_seq=batch["seq"])
        failer_seq = _state(journal, failer["edit_id"],
                            fixture["seq"]).intent_seq
        success_dep = _admit_dep(
            journal, gate, ctx, fixture, "fs_write",
            {"path": "must-not-exist.txt", "content": "no\n"},
            [failer_seq], batch_seq=batch["seq"],
            depends_on=[{"seq": failer_seq, "requirement": "success"}])
        outcome_dep = _admit_dep(
            journal, gate, ctx, fixture, "fs_write",
            {"path": "outcome-ran.txt", "content": "ran\n"},
            [failer_seq], batch_seq=batch["seq"],
            depends_on=[{"seq": failer_seq, "requirement": "outcome"}])
        success_seq = _state(journal, success_dep["edit_id"],
                             fixture["seq"]).intent_seq
        cascade = _admit_dep(
            journal, gate, ctx, fixture, "fs_write",
            {"path": "cascade.txt", "content": "no\n"},
            [failer_seq, success_seq], batch_seq=batch["seq"],
            depends_on=[{"seq": success_seq,
                         "requirement": "success"}])
        for out in (failer, success_dep, outcome_dep, cascade):
            assert out["verdict"] == "ADMIT", out["gate_report"]

        res = _drainer(journal, bundle_dir, manifest,
                       work).drain_task(fixture["seq"])
        assert res["sync_reason"] == "queue-empty"

        failer_state = _state(journal, failer["edit_id"],
                              fixture["seq"])
        # The dependency really did conclude without success.
        assert failer_state.state == "executed"
        assert failer_state.outcome_status == "failed"

        skipped = _state(journal, success_dep["edit_id"],
                         fixture["seq"])
        assert skipped.state == "skipped"
        assert skipped.claim_seq is None
        assert skipped.skip_dependency_seq == failer_seq
        assert skipped.skip_dependency_outcome_seq == \
            failer_state.outcome_seq
        assert "success" in skipped.skip_cause
        assert not os.path.exists(
            os.path.join(work, "must-not-exist.txt"))
        # The skip itself is a journaled queue entry (§7.3).
        skips = journal.backend.find(
            "experiences", filter={"kind": "queue_skip"})
        skip_records = [r["record"] for r in skips]
        assert any(r["edit_id"] == success_dep["edit_id"]
                   for r in skip_records)

        # Cascade: a success dependent of a skipped intent is
        # itself skipped, naming the skipped dependency.
        cascaded = _state(journal, cascade["edit_id"], fixture["seq"])
        assert cascaded.state == "skipped"
        assert cascaded.skip_dependency_seq == success_seq
        assert not os.path.exists(os.path.join(work, "cascade.txt"))

        # The outcome dependent ran: any recorded conclusion
        # satisfies `outcome`.
        ran = _state(journal, outcome_dep["edit_id"], fixture["seq"])
        assert ran.state == "executed"
        assert ran.execution["status"] == "completed"
        assert _read(os.path.join(work, "outcome-ran.txt")) == "ran\n"
        assert res["summary"]["counts"] == {
            "executed": 2, "skipped": 2}
        assert journal.verify_chain()
    finally:
        journal.close()


def test_success_dependent_runs_after_successful_dependency(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture, gate, ctx = _setup(journal, work)
        batch = _batch(journal, fixture, 2)
        base = _admit(journal, gate, ctx, fixture, "fs_write",
                      {"path": "base.txt", "content": "base\n"},
                      batch_seq=batch["seq"])
        base_seq = _state(journal, base["edit_id"],
                          fixture["seq"]).intent_seq
        dependent = _admit_dep(
            journal, gate, ctx, fixture, "fs_write",
            {"path": "dependent.txt", "content": "dep\n"},
            [base_seq], batch_seq=batch["seq"],
            depends_on=[{"seq": base_seq, "requirement": "success"}])
        res = _drainer(journal, bundle_dir, manifest,
                       work).drain_task(fixture["seq"])
        assert res["sync_reason"] == "queue-empty"
        assert res["skips"] == []
        assert _read(os.path.join(work, "dependent.txt")) == "dep\n"
        assert res["summary"]["counts"] == {"executed": 2}
    finally:
        journal.close()


# -- barrier sync point (§10.2 point 2) -----------------------------------------

def test_barrier_flushes_perception_and_holds_the_rest(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture, gate, ctx = _setup(journal, work)
        batch = _batch(journal, fixture, 2)
        barrier_write = _admit(
            journal, gate, ctx, fixture, "fs_write",
            {"path": "barrier.txt", "content": "B\n"},
            batch_seq=batch["seq"], barrier=True)
        after = _admit(journal, gate, ctx, fixture, "fs_write",
                       {"path": "after-barrier.txt", "content": "A\n"},
                       batch_seq=batch["seq"])

        drainer = _drainer(journal, bundle_dir, manifest, work)
        res = drainer.drain_task(fixture["seq"])
        # The barrier concluded: perception returns NOW, with
        # only the barrier outcome in the set; the later intent
        # is still queued, unexecuted.
        assert res["sync_reason"] == "barrier"
        observations = res["perception"]["observations"]
        assert [o["edit_id"] for o in observations] == \
            [barrier_write["edit_id"]]
        assert res["perception"]["sync_point"] == "barrier"
        assert _read(os.path.join(work, "barrier.txt")) == "B\n"
        assert not os.path.exists(
            os.path.join(work, "after-barrier.txt"))
        assert _state(journal, after["edit_id"],
                      fixture["seq"]).state == "queued"

        # The next drain completes the held remainder.
        res = drainer.drain_task(fixture["seq"])
        assert res["sync_reason"] == "queue-empty"
        assert _read(os.path.join(work, "after-barrier.txt")) == "A\n"
    finally:
        journal.close()


# -- claim-time residence refusal (§12.2) ----------------------------------------

def test_stale_admission_refused_at_claim(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture, gate, ctx = _setup(journal, work)
        out = _admit(journal, gate, ctx, fixture, "fs_write",
                     {"path": "stale.txt", "content": "old\n"})
        policy = corpus_policy(work)
        policy["queue_max_residence_seconds"] = 0
        res = _drainer(journal, bundle_dir, manifest, work,
                       policy=policy).drain_task(fixture["seq"])
        assert res["sync_reason"] == "queue-empty"
        assert res["executions"] == {}
        assert not os.path.exists(os.path.join(work, "stale.txt"))
        state = _state(journal, out["edit_id"], fixture["seq"])
        assert state.state == "refused-at-claim"
        assert state.refusal_cause is not None
        assert "stale-admission" in state.refusal_cause
        assert res["summary"]["counts"] == {"refused-at-claim": 1}
    finally:
        journal.close()


# -- the read pool (§8) -----------------------------------------------------------

def test_four_unkeyed_reads_execute_concurrently(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture, gate, ctx = _setup(journal, work)
        batch = _batch(journal, fixture, 4)
        outs = [
            _admit(journal, gate, ctx, fixture, "fs_read",
                   {"path": "notes/plan.md"}, batch_seq=batch["seq"])
            for _ in range(4)
        ]
        stub = _StubExecutor(journal, barrier=threading.Barrier(4))
        drainer = _drainer(journal, bundle_dir, manifest, work,
                           executor=stub)
        assert drainer.max_concurrent_reads() == 4
        res = drainer.drain_task(fixture["seq"])
        assert res["sync_reason"] == "queue-empty"
        # The barrier could only be reached with all four reads
        # in flight at once: the pool really is concurrent.
        assert stub.peak == 4
        assert set(res["executions"]) == {o["edit_id"] for o in outs}
        fold = fold_queue(journal, task_seq=fixture["seq"])
        for out in outs:
            state = fold.intents[out["edit_id"]]
            assert state.state == "executed"
            # Claim before execution still holds per intent.
            assert state.intent_seq < state.claim_seq \
                < state.outcome_seq
        assert journal.verify_chain()
    finally:
        journal.close()


def test_read_pool_bound_two_caps_concurrency(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture, gate, ctx = _setup(journal, work)
        batch = _batch(journal, fixture, 4)
        for _ in range(4):
            _admit(journal, gate, ctx, fixture, "fs_read",
                   {"path": "notes/plan.md"}, batch_seq=batch["seq"])
        stub = _StubExecutor(journal, delay=0.3)
        policy = corpus_policy(work)
        policy["queue_max_concurrent_reads"] = 2
        drainer = _drainer(journal, bundle_dir, manifest, work,
                           policy=policy, executor=stub)
        res = drainer.drain_task(fixture["seq"])
        assert res["sync_reason"] == "queue-empty"
        assert len(res["executions"]) == 4
        assert stub.peak == 2
        assert journal.verify_chain()
    finally:
        journal.close()


def test_keyed_reads_and_writes_never_overlap(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture, gate, ctx = _setup(journal, work)
        batch = _batch(journal, fixture, 5)
        read1 = _admit(journal, gate, ctx, fixture, "fs_read",
                       {"path": "notes/plan.md"},
                       batch_seq=batch["seq"])
        read2 = _admit(journal, gate, ctx, fixture, "fs_read",
                       {"path": "notes/plan.md"},
                       batch_seq=batch["seq"])
        keyed1 = _admit(journal, gate, ctx, fixture, "fs_read",
                        {"path": "notes/plan.md"},
                        batch_seq=batch["seq"],
                        ordering_key="shared-key")
        keyed2 = _admit(journal, gate, ctx, fixture, "fs_read",
                        {"path": "notes/plan.md"},
                        batch_seq=batch["seq"],
                        ordering_key="shared-key")
        write = _admit(journal, gate, ctx, fixture, "fs_write",
                       {"path": "w.txt", "content": "w\n"},
                       batch_seq=batch["seq"])
        stub = _StubExecutor(journal, delay=0.25)
        res = _drainer(journal, bundle_dir, manifest, work,
                       executor=stub).drain_task(fixture["seq"])
        assert res["sync_reason"] == "queue-empty"
        assert len(res["executions"]) == 5
        # The two unkeyed reads overlapped each other (the pool);
        # the keyed reads and the write overlapped NOTHING.
        assert _overlaps(stub.intervals[read1["edit_id"]],
                         stub.intervals[read2["edit_id"]])
        serial_ids = [keyed1["edit_id"], keyed2["edit_id"],
                      write["edit_id"]]
        all_ids = [read1["edit_id"], read2["edit_id"]] + serial_ids
        for edit_id in serial_ids:
            for other in all_ids:
                if other != edit_id:
                    assert not _overlaps(stub.intervals[edit_id],
                                         stub.intervals[other]), \
                        (edit_id, other)
        assert journal.verify_chain()
    finally:
        journal.close()


# -- the pipeline (§9): admission overlaps drain ----------------------------------

def test_admission_overlaps_an_in_flight_drain(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture, gate, ctx = _setup(journal, work)
        batch1 = _batch(journal, fixture, 1)
        first = _admit(journal, gate, ctx, fixture, "fs_read",
                       {"path": "notes/plan.md"},
                       batch_seq=batch1["seq"])

        entered = threading.Event()
        release = threading.Event()

        class _BlockingStub(_StubExecutor):
            def execute_journaled(self, edit_id, **kw):
                entered.set()
                release.wait(timeout=15)
                return super().execute_journaled(edit_id, **kw)

        stub = _BlockingStub(journal)
        drainer = _drainer(journal, bundle_dir, manifest, work,
                           executor=stub)
        drain_thread = threading.Thread(
            target=drainer.drain_task, args=(fixture["seq"],))
        drain_thread.start()
        assert entered.wait(timeout=15)
        # Batch N+1 is gated and journaled WHILE batch N is
        # still executing: admission does not wait for the drain.
        batch2 = _batch(journal, fixture, 1)
        second = _admit(journal, gate, ctx, fixture, "fs_write",
                        {"path": "pipelined.txt", "content": "P\n"},
                        batch_seq=batch2["seq"])
        release.set()
        drain_thread.join(timeout=30)
        assert not drain_thread.is_alive()

        first_state = _state(journal, first["edit_id"],
                             fixture["seq"])
        assert first_state.state == "executed"
        # The journal is the witness: batch N+1's admission sits
        # strictly between batch N's claim and its outcome.
        second_state = _state(journal, second["edit_id"],
                              fixture["seq"])
        assert first_state.claim_seq < second_state.intent_seq \
            < first_state.outcome_seq

        res = drainer.drain_task(fixture["seq"])
        assert res["sync_reason"] == "queue-empty"
        assert _state(journal, second["edit_id"],
                      fixture["seq"]).state == "executed"
        assert journal.verify_chain()
    finally:
        journal.close()


# -- journal thread-safety (the Q-2 enabler) ---------------------------------------

def test_shared_journal_appends_from_many_threads_verify(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        def append_many():
            for _ in range(25):
                journal.record_experience(
                    "q2_thread_probe", {"kind": "q2_thread_probe"},
                    "")

        threads = [threading.Thread(target=append_many)
                   for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert all(not t.is_alive() for t in threads)
        rows = journal.backend.find(
            "experiences", filter={"kind": "q2_thread_probe"})
        assert len(rows) == 200
        assert len({r["seq"] for r in rows}) == 200
        assert journal.verify_chain()
    finally:
        journal.close()
