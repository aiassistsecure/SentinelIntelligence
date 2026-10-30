"""SI-51 (Gaia Q-1) task-loop integration tests: batch proposal,
batched perception, gate queue pricing, depends_on, and the
§12.4 stop rules — the §17 acceptance criteria at the task level.

The fold/claim/drain mechanics have their own suites
(test_gaia_q1_queue.py, test_gaia_q1_drain.py); this file proves
the B-3 task loop now runs propose-batch -> admit -> drain ->
perceive, that verdicts are per-member and journaled in proposal
order, that gate pricing (ordering key / depends_on / barrier)
lands in the verdict entries, and that §12.4's "stop proposing,
not draining" holds end to end.

Fixtures mirror tests/test_gaia_b3.py's miniature pattern.
"""

from __future__ import annotations

import os

import pytest

from si.bundle import init_bundle, load_bundle
from si.gaia.corpus import corpus_policy
from si.gaia.queue import fold_queue
from si.gaia.task import run_task
from si.gate import Intent
from si.journal import open_journal
from si.tiny_seed import seed_tiny_bundle


@pytest.fixture()
def mini(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "mini.si"), name="mini-q1",
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    work = str(tmp_path / "work")
    os.makedirs(work)
    return bundle_dir, work


def _write(path, content, **kw):
    return Intent(kind="tool", tool="fs_write",
                  args={"path": path, "content": content},
                  reason=f"write {path}", **kw)


def _journal(bundle_dir):
    manifest = load_bundle(bundle_dir)
    return open_journal(bundle_dir, manifest)


def _rows(journal, collection, **flt):
    return journal.backend.find(collection, filter=flt or None)


# -- §17.4 batch honesty ---------------------------------------------------


def test_batch_members_journaled_individually_in_proposal_order(mini):
    bundle_dir, work = mini
    script = [[
        _write("a.txt", "A"),
        Intent(kind="tool", tool="no_such_tool", args={},
               reason="unknown tool member"),
        _write("b.txt", "B"),
    ]]
    res = run_task(bundle_dir, script, "batch honesty",
                   work_root=work, policy=corpus_policy(work))
    assert [s["verdict"] for s in res["steps"]] == \
        ["ADMIT", "REJECT", "ADMIT"]
    assert res["cycles_used"] == 3  # §12.3: debited per EVALUATED
    # Both ADMITted members drained; the REJECTed one never ran.
    assert open(os.path.join(work, "a.txt")).read() == "A"
    assert open(os.path.join(work, "b.txt")).read() == "B"
    journal = _journal(bundle_dir)
    try:
        # One batch entry, journaled before its members, naming
        # the declared size; each member's verdict entry carries
        # its own queue pricing naming that batch.
        batches = _rows(journal, "experiences", kind="queue_batch")
        assert len(batches) == 1
        assert batches[0]["record"]["declared_size"] == 3
        assert res["batch_seqs"] == [batches[0]["seq"]]
        members = [r for r in _rows(journal, "self_edits")
                   if r.get("kind") == "tool"]
        members.sort(key=lambda r: r["seq"])
        assert len(members) == 3
        assert all(r["seq"] > batches[0]["seq"] for r in members)
        assert [r["gate_verdict"] for r in members] == \
            ["ADMIT", "REJECT", "ADMIT"]
        for r in members:
            queue = r["gate_report"]["queue"]
            assert queue["batch_seq"] == batches[0]["seq"]
            assert queue["depends_on"] == []
            assert queue["barrier"] is False
        # The fold, from the journal alone, agrees (§17.1 at the
        # task level): two executed, one rejected, none claimed.
        fold = fold_queue(journal, res["task_seq"])
        states = sorted(s.state for s in fold.intents.values())
        assert states == ["executed", "executed", "rejected"]
    finally:
        journal.close()


def test_no_verdict_references_another_member(mini):
    """§6.1/§17.4: a member's gate report speaks only about that
    member — the REJECTed member's report names its own problem,
    and no report mentions a sibling's tool or edit."""
    bundle_dir, work = mini
    script = [[
        _write("solo.txt", "S"),
        Intent(kind="tool", tool="no_such_tool", args={},
               reason="the bad one"),
    ]]
    run_task(bundle_dir, script, "no cross-reference",
             work_root=work, policy=corpus_policy(work))
    journal = _journal(bundle_dir)
    try:
        members = [r for r in _rows(journal, "self_edits")
                   if r.get("kind") == "tool"]
        members.sort(key=lambda r: r["seq"])
        good, bad = members
        import json
        assert "no_such_tool" not in json.dumps(good["gate_report"])
        assert "fs_write" not in json.dumps(bad["gate_report"])
    finally:
        journal.close()


# -- §17.6 batched perception ------------------------------------------------


def test_one_observation_set_per_batch_with_queue_summary(mini):
    bundle_dir, work = mini
    script = [[_write("p1.txt", "1"), _write("p2.txt", "2")],
              [_write("p3.txt", "3")]]
    res = run_task(bundle_dir, script, "batched perception",
                   work_root=work, policy=corpus_policy(work))
    assert len(res["perceptions"]) == 2  # one set per batch
    first, second = res["perceptions"]
    assert first["kind"] == "queue_perception"
    assert first["sync_point"] == "queue-empty"
    obs = first["observations"]
    assert len(obs) == 2
    # Outcome-seq order, provenance-complete (§10.1/§10.2).
    assert [o["outcome_seq"] for o in obs] == \
        sorted(o["outcome_seq"] for o in obs)
    for o in obs:
        assert o["task_seq"] == res["task_seq"]
        assert o["intent_seq"] and o["claim_seq"] and o["edit_id"]
        assert o["claim_seq"] < o["outcome_seq"]
    summary = first["queue_summary"]
    assert summary["counts"].get("executed") == 2
    assert summary["depth"] == 0
    assert len(second["observations"]) == 1
    assert second["queue_summary"]["counts"].get("executed") == 3


# -- §7.2 ordering keys at the gate ------------------------------------------


def test_gate_prices_default_ordering_keys(mini):
    bundle_dir, work = mini
    script = [[
        _write("k.txt", "first"),
        _write("k.txt", "second"),
        Intent(kind="tool", tool="fs_read",
               args={"path": "k.txt"}, reason="read it back"),
        Intent(kind="tool", tool="journal_query",
               args={"collection": "experiences", "limit": 1},
               reason="house tool"),
    ]]
    res = run_task(bundle_dir, script, "ordering keys",
                   work_root=work, policy=corpus_policy(work))
    # Same-key writes serialise in sequence order (§7.1 FIFO):
    # the second write wins.
    assert open(os.path.join(work, "k.txt")).read() == "second"
    journal = _journal(bundle_dir)
    try:
        members = [r for r in _rows(journal, "self_edits")
                   if r.get("kind") == "tool"]
        members.sort(key=lambda r: r["seq"])
        keys = [r["gate_report"]["queue"]["ordering_key"]
                for r in members]
        target = os.path.join(os.path.realpath(work), "k.txt")
        assert keys[0] == f"fs:{target}"
        assert keys[1] == f"fs:{target}"
        assert keys[2] is None       # read class: no key by default
        assert keys[3] == "house"    # house tools share one key
        # Claims were taken in intent-seq order (serial drain).
        claims = _rows(journal, "experiences", kind="queue_claim")
        claims.sort(key=lambda r: r["seq"])
        assert [c["record"]["intent_seq"] for c in claims] == \
            [r["seq"] for r in members]
    finally:
        journal.close()


def test_declared_ordering_key_and_barrier_are_priced(mini):
    bundle_dir, work = mini
    script = [[_write("d.txt", "D", ordering_key="release-train",
                      barrier=True)]]
    run_task(bundle_dir, script, "declared key",
             work_root=work, policy=corpus_policy(work))
    journal = _journal(bundle_dir)
    try:
        members = [r for r in _rows(journal, "self_edits")
                   if r.get("kind") == "tool"]
        queue = members[0]["gate_report"]["queue"]
        assert queue["ordering_key"] == "release-train"
        assert queue["barrier"] is True
    finally:
        journal.close()


# -- §6.2 / §7.3 depends_on pricing -------------------------------------------


class _JournalAwareMind:
    """A mind proposer that learns the journal's intent seqs
    through PERCEIVE (the journal itself), the way a real mind
    would: round 1 proposes one write; round 2 proposes a write
    depending on round 1's intent seq; round 3 completes."""

    def __init__(self):
        self.round = 0

    def perceive(self, journal):
        rows = journal.backend.find("self_edits") or []
        seqs = sorted(r["seq"] for r in rows
                      if r.get("kind") == "tool")
        return [{"kind": "intent_seqs", "seqs": seqs}]

    def propose(self, perceptions, ctx):
        self.round += 1
        task = perceptions[0]
        chain = {"parents": [f"task@{task['seq']}"],
                 "source_seqs": [task["seq"]]}
        if self.round == 1:
            return [_write("base.txt", "base", **chain)]
        if self.round == 2:
            info = [p for p in perceptions
                    if p.get("kind") == "intent_seqs"][0]
            dep = info["seqs"][0]
            return [_write("next.txt", "next",
                           depends_on=[dep], **chain)]
        intent = Intent(
            kind="memory",
            record={"kind": "task_complete",
                    "summary": "depends_on chain drained"},
            reason="done", **chain)
        return [intent]


def test_depends_on_earlier_intent_in_same_task_admits_and_drains(mini):
    bundle_dir, work = mini
    res = run_task(bundle_dir, _JournalAwareMind(), "depends chain",
                   work_root=work, policy=corpus_policy(work))
    assert res["stop_reason"] == "mind-reported-complete"
    assert open(os.path.join(work, "base.txt")).read() == "base"
    assert open(os.path.join(work, "next.txt")).read() == "next"
    journal = _journal(bundle_dir)
    try:
        members = [r for r in _rows(journal, "self_edits")
                   if r.get("kind") == "tool"]
        members.sort(key=lambda r: r["seq"])
        queue = members[1]["gate_report"]["queue"]
        assert queue["depends_on"] == [
            {"seq": members[0]["seq"], "requirement": "outcome"}]
    finally:
        journal.close()


def test_depends_on_future_sequence_rejects_that_member_only(mini):
    bundle_dir, work = mini
    script = [[
        _write("future.txt", "F", depends_on=[999999]),
        _write("fine.txt", "ok"),
    ]]
    res = run_task(bundle_dir, script, "future dependency",
                   work_root=work, policy=corpus_policy(work))
    assert [s["verdict"] for s in res["steps"]] == ["REJECT", "ADMIT"]
    assert not os.path.exists(os.path.join(work, "future.txt"))
    assert open(os.path.join(work, "fine.txt")).read() == "ok"
    journal = _journal(bundle_dir)
    try:
        members = [r for r in _rows(journal, "self_edits")
                   if r.get("kind") == "tool"]
        members.sort(key=lambda r: r["seq"])
        detail = str(members[0]["gate_report"])
        assert "not yet journaled" in detail
    finally:
        journal.close()


def test_depends_on_outside_task_rejects(mini):
    bundle_dir, work = mini
    # The task entry's own seq is journaled and earlier — but it
    # is not an intent OF this task, so §6.2 rejects the member.
    task_seq_holder = {}

    class _SneakyMind:
        def perceive(self, journal):
            return []

        def propose(self, perceptions, ctx):
            task_seq_holder["seq"] = perceptions[0]["seq"]
            return [_write("sneak.txt", "S",
                           depends_on=[perceptions[0]["seq"]])]

    res = run_task(bundle_dir, _SneakyMind(), "outside dependency",
                   work_root=work, policy=corpus_policy(work))
    assert res["steps"][0]["verdict"] == "REJECT"
    assert not os.path.exists(os.path.join(work, "sneak.txt"))
    journal = _journal(bundle_dir)
    try:
        members = [r for r in _rows(journal, "self_edits")
                   if r.get("kind") == "tool"]
        assert "outside this task" in str(members[0]["gate_report"])
    finally:
        journal.close()


# -- §12.4 stop-rule parity at the task level ---------------------------------


def test_cycle_cap_mid_batch_stops_proposing_but_drains(mini):
    bundle_dir, work = mini
    policy = corpus_policy(work, {"max_tool_cycles_per_task": 2})
    script = [[_write("c1.txt", "1"), _write("c2.txt", "2"),
               _write("c3.txt", "3")]]
    res = run_task(bundle_dir, script, "mid-batch cap",
                   work_root=work, policy=policy)
    assert res["stop_reason"] == "budget-exhausted"
    assert res["cycles_used"] == 2
    assert len(res["steps"]) == 2  # third member never evaluated
    # ...but the two admitted members DRAINED (§12.4.1).
    assert open(os.path.join(work, "c1.txt")).read() == "1"
    assert open(os.path.join(work, "c2.txt")).read() == "2"
    assert not os.path.exists(os.path.join(work, "c3.txt"))


def test_escalated_member_halts_proposing_queued_member_drains(mini):
    bundle_dir, work = mini
    outside = "/etc/si51-q1-never.txt"
    script = [[
        _write("before.txt", "B"),
        Intent(kind="tool", tool="fs_write",
               args={"path": outside, "content": "x"},
               reason="escapes the work root: external, escalates"),
        _write("after.txt", "A"),
    ]]
    res = run_task(bundle_dir, script, "escalation mid-batch",
                   work_root=work, policy=corpus_policy(work))
    assert res["stop_reason"] == "escalated"
    assert [s["verdict"] for s in res["steps"]] == ["ADMIT", "ESCALATE"]
    assert open(os.path.join(work, "before.txt")).read() == "B"
    assert not os.path.exists(os.path.join(work, "after.txt"))
    assert not os.path.exists(outside)


def test_completion_member_closes_only_after_batch_drains(mini):
    bundle_dir, work = mini
    script = [[
        _write("done-after.txt", "D"),
        Intent(kind="memory",
               record={"kind": "task_complete",
                       "summary": "wrote and done"},
               reason="report completion"),
    ]]
    res = run_task(bundle_dir, script, "completion in batch",
                   work_root=work, policy=corpus_policy(work))
    assert res["stop_reason"] == "mind-reported-complete"
    # The write DRAINED before the task closed (§12.4.5).
    assert open(os.path.join(work, "done-after.txt")).read() == "D"
    assert res["cycles_used"] == 1  # completion is not a tool cycle
    journal = _journal(bundle_dir)
    try:
        outcome = _rows(journal, "experiences", kind="tool_result")
        outcome.sort(key=lambda r: r["seq"])
        stop = _rows(journal, "experiences", kind="task_stop")[0]
        assert outcome[0]["seq"] < stop["seq"]
        fold = fold_queue(journal, res["task_seq"])
        assert all(s.state == "executed"
                   for s in fold.intents.values())
    finally:
        journal.close()
