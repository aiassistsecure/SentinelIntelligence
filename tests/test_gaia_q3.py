"""SI-53 (SI-50 §16 Q-3, §§11, 13.3, 13.4, 15.4, 15.5): Gaia
Q-3 — the operator leash against a LIVE drain, operator
dispositions for outcome-unknown write/execute claims, and
queue exposure through the service/SSE surface.

Coverage:

- live leash, execute class (§11.2): a ``shell_exec`` sleeping
  in flight when the operator stop is journaled is killed —
  the outcome is journaled ``timed-out`` with
  ``kill_cause == "operator-stop"`` and partial output
  labelled partial; the still-queued intent behind it is held,
  executes nothing, and drains only after a resume;
- live leash, write class (§11.2): a write in flight at the
  stop runs to its journaled conclusion; the queued intent
  behind it is held until resume;
- recovery, write class (§13.3): an expired write claim folds
  ``outcome-unknown`` (never re-executed), its target's
  post-state is journaled as a recovery observation labelled
  as evidence, and a drain leaves it untouched;
- recovery, execute class (§13.3): an expired execute claim
  folds ``outcome-unknown`` and is never replayed (the
  command's side-effect file is never created by recovery);
- dispositions (§13.4): ``accept-unknown`` is an explicit
  journaled operator act that satisfies an ``outcome``
  dependent (which then runs) and skips a ``success``
  dependent; before any disposition both dependents wait;
  ``mark-executed`` and dispositions of non-unknown or
  already-dispositioned intents are refused and journal
  nothing; ``compensate`` / ``re-propose`` annotate without
  rewriting the state and without satisfying dependents;
- service exposure (§15.4): the queue view names §5.2 states,
  a submitted task's response carries its queue, and the
  disposition endpoint journals the act (400 + nothing
  journaled for a non-unknown intent);
- SSE convention (§15.5): queue events carry ``queue_state``
  — the §5.2 state name the row transitions to — in both the
  event dicts and the serialised SSE payload.
"""
import json
import os
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from si.bundle import _anchor_checkpoint, _write_manifest, init_bundle, load_bundle
from si.gaia.context import ToolContext
from si.gaia.corpus import corpus_policy
from si.gaia.drain import Drainer, admit_tool_intent
from si.gaia.events import (
    format_sse,
    queue_state_for_row,
    read_journal_events,
)
from si.gaia.executor import record_execution_outcome
from si.gaia.queue import (
    DISPOSITIONS,
    disposition_outcome_unknown,
    fold_queue,
    queue_state_view,
    record_queue_batch,
    record_queue_claim,
    record_queue_resume,
    record_queue_stop,
)
from si.gaia.service import (
    make_service_server,
    queue_view_for_bundle,
    submit_disposition,
    submit_task,
)
from si.gaia.shadow import record_task_fixture
from si.gate import Gate, Intent
from si.journal import open_journal
from si.tiny_seed import seed_tiny_bundle


# -- fixtures (copied from tests/test_gaia_q2_drain.py) ----------------------

@pytest.fixture()
def mini(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "mini.si"), name="gaia-q3",
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
    fixture = record_task_fixture(journal, "Q-3 scenario", work)
    gate = Gate(corpus_policy(work))
    ctx = ToolContext(tool_work_root=work,
                      tool_task_seq=fixture["seq"],
                      tool_live_execution=True)
    return fixture, gate, ctx


def _admit(journal, gate, ctx, fixture, tool, args, batch_seq=None,
           **kw):
    intent = Intent(kind="tool", tool=tool, args=args,
                    reason="Carry out the scripted Q-3 task step",
                    parents=[fixture["ref"]],
                    source_seqs=[fixture["seq"]], **kw)
    return admit_tool_intent(journal, gate, ctx, intent,
                             batch_seq=batch_seq)


def _admit_dep(journal, gate, ctx, fixture, tool, args, dep_seqs,
               **kw):
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


def _anchor(journal, bundle_dir, manifest):
    """Re-anchor the manifest to the journal tip, as the real
    task machinery does after journaling — service helpers load
    the bundle fresh and fail closed on a stale anchor."""
    updated = _anchor_checkpoint(journal, manifest, manifest["head"],
                                 note="Q-3 service test scenario")
    _write_manifest(bundle_dir, updated)


def _read(path):
    with open(path) as f:
        return f.read()


def _orphan(journal, fixture, out, lease_seconds=0):
    """Journal a claim by a dead drainer over ``out``'s intent
    and return the folded state (the §13.2 setup: claimed, no
    outcome, lease expired)."""
    state = _state(journal, out["edit_id"], fixture["seq"])
    claim = record_queue_claim(
        journal, fixture["seq"], out["edit_id"], state.intent_seq,
        drainer_id="drainer-dead", lease_seconds=lease_seconds)
    return state, claim


# -- live leash: execute class in flight is killed (§11.2) ---------------------

def test_live_leash_kills_in_flight_execute_and_holds_queue(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture, gate, ctx = _setup(journal, work)
        batch = _batch(journal, fixture, 2)
        slow = _admit(
            journal, gate, ctx, fixture, "shell_exec",
            {"command": "echo started; sleep 30; echo finished"},
            batch_seq=batch["seq"])
        queued = _admit(journal, gate, ctx, fixture, "fs_write",
                        {"path": "after-stop.txt", "content": "no\n"},
                        batch_seq=batch["seq"])
        drainer = _drainer(journal, bundle_dir, manifest, work)
        result_box = {}
        drain_thread = threading.Thread(
            target=lambda: result_box.setdefault(
                "res", drainer.drain_task(fixture["seq"])))
        drain_thread.start()
        # Wait until the slow command is actually claimed and in
        # flight, then pull the leash from the operator side.
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            folded = _state(journal, slow["edit_id"], fixture["seq"])
            if folded.state == "claimed":
                break
            time.sleep(0.05)
        assert folded.state == "claimed"
        time.sleep(0.5)  # the process is running (echo started)
        record_queue_stop(journal, fixture["seq"],
                          cause="operator-stop",
                          detail="leash pulled mid-drain (Q-3 test)")
        drain_thread.join(timeout=30)
        assert not drain_thread.is_alive()
        res = result_box["res"]
        assert res["sync_reason"] == "stop-in-force"

        execution = res["executions"][slow["edit_id"]]
        # A leash kill is a timeout whose cause is the operator:
        # timed-out, partial, kill_cause journaled on the record.
        assert execution["status"] == "timed-out"
        assert execution["partial"] is True
        assert execution["kill_cause"] == "operator-stop"
        assert "operator stop" in execution["detail"]
        assert "started" in execution["result"]["stdout"]
        assert "finished" not in execution["result"]["stdout"]

        # The queued intent was held, never executed.
        assert _state(journal, queued["edit_id"],
                      fixture["seq"]).state == "held"
        assert not os.path.exists(os.path.join(work, "after-stop.txt"))

        # Resume is a new operator cause; the held intent drains.
        record_queue_resume(journal, fixture["seq"],
                            detail="operator resumed (Q-3 test)")
        res = drainer.drain_task(fixture["seq"])
        assert res["sync_reason"] == "queue-empty"
        assert _read(os.path.join(work, "after-stop.txt")) == "no\n"
        assert journal.verify_chain()
    finally:
        journal.close()


def test_live_leash_lets_in_flight_write_conclude(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture, gate, ctx = _setup(journal, work)
        batch = _batch(journal, fixture, 2)
        first = _admit(journal, gate, ctx, fixture, "fs_write",
                       {"path": "in-flight.txt", "content": "done\n"},
                       batch_seq=batch["seq"])
        second = _admit(journal, gate, ctx, fixture, "fs_write",
                        {"path": "held.txt", "content": "held\n"},
                        batch_seq=batch["seq"])

        entered = threading.Event()

        class _SlowWriteExecutor:
            """Journals a real outcome through the executor's
            own writer, slowly, so the stop lands mid-flight."""
            def execute_journaled(self, edit_id,
                                  expected_effective_class=None,
                                  claim_intent_seq=None,
                                  claim_seq=None):
                entered.set()
                time.sleep(1.0)
                rows = journal.backend.find(
                    "self_edits", filter={"edit_id": edit_id},
                    limit=1)
                tool = (rows[0].get("intent") or {}).get("tool", "")
                if tool == "fs_write":
                    args = (rows[0].get("intent") or {}).get(
                        "args") or {}
                    with open(os.path.join(work, args["path"]),
                              "w") as f:
                        f.write(args["content"])
                execution = {
                    "tool": tool, "intent_id": edit_id,
                    "class": expected_effective_class,
                    "status": "completed", "result": {},
                    "detail": "slow stub execution (Q-3 leash test)",
                }
                record_execution_outcome(
                    journal, SimpleNamespace(tool=tool), edit_id,
                    execution)
                journal.set_edit_result(edit_id, "executed")
                return execution

        drainer = _drainer(journal, bundle_dir, manifest, work,
                           executor=_SlowWriteExecutor())
        result_box = {}
        drain_thread = threading.Thread(
            target=lambda: result_box.setdefault(
                "res", drainer.drain_task(fixture["seq"])))
        drain_thread.start()
        assert entered.wait(timeout=15)
        record_queue_stop(journal, fixture["seq"],
                          cause="operator-stop",
                          detail="stop during a write (Q-3 test)")
        drain_thread.join(timeout=30)
        assert not drain_thread.is_alive()
        res = result_box["res"]
        assert res["sync_reason"] == "stop-in-force"
        # §11.2: the in-flight write ran to its conclusion and
        # its outcome is journaled normally...
        assert res["executions"][first["edit_id"]]["status"] == \
            "completed"
        assert _read(os.path.join(work, "in-flight.txt")) == "done\n"
        # ...while the next intent was held, not executed.
        assert second["edit_id"] not in res["executions"]
        assert _state(journal, second["edit_id"],
                      fixture["seq"]).state == "held"
        assert not os.path.exists(os.path.join(work, "held.txt"))
    finally:
        journal.close()


# -- recovery: write/execute outcome-unknown (§13.3) ----------------------------

def test_recovery_write_claim_folds_outcome_unknown_with_observation(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture, gate, ctx = _setup(journal, work)
        out = _admit(journal, gate, ctx, fixture, "fs_write",
                     {"path": "maybe.txt", "content": "landed?\n"})
        # The crashed drainer DID land the write before dying —
        # recovery cannot know that from the journal (§13.2).
        with open(os.path.join(work, "maybe.txt"), "w") as f:
            f.write("landed?\n")
        state, claim = _orphan(journal, fixture, out)
        drainer = _drainer(journal, bundle_dir, manifest, work,
                           drainer_id="drainer-new")
        rec = drainer.recover(task_seq=fixture["seq"])
        assert rec["outcome_unknown"] == [state.intent_seq]
        assert rec["deferred"] == [] and rec["reclaimed"] == []
        folded = _state(journal, out["edit_id"], fixture["seq"])
        assert folded.state == "outcome-unknown"
        assert folded.outcome_unknown_claim_seq == claim["seq"]
        assert "§13.3" in folded.outcome_unknown_cause
        # The post-state was gathered as a read and journaled as
        # an observation — evidence, never an outcome.
        assert len(folded.recovery_observations) == 1
        obs = folded.recovery_observations[0]["observation"]
        assert obs["exists"] is True and obs["kind"] == "file"
        assert obs["sha256"].startswith("sha256:")
        rows = journal.backend.find(
            "experiences",
            filter={"kind": "queue_recovery_observation"})
        assert len(rows) == 1
        assert "never an outcome" in rows[0]["record"]["label"]
        # A drain does not touch it: no claim, no execution, the
        # file is exactly as the crashed drainer left it.
        res = drainer.drain_task(fixture["seq"])
        assert res["executions"] == {}
        assert _state(journal, out["edit_id"],
                      fixture["seq"]).state == "outcome-unknown"
        assert _read(os.path.join(work, "maybe.txt")) == "landed?\n"
        assert journal.verify_chain()
    finally:
        journal.close()


def test_recovery_execute_claim_is_never_replayed(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture, gate, ctx = _setup(journal, work)
        marker = os.path.join(work, "replayed.txt")
        out = _admit(
            journal, gate, ctx, fixture, "shell_exec",
            {"command": f"echo ran >> {marker}"})
        state, _claim = _orphan(journal, fixture, out)
        drainer = _drainer(journal, bundle_dir, manifest, work,
                           drainer_id="drainer-new")
        rec = drainer.recover(task_seq=fixture["seq"])
        assert rec["outcome_unknown"] == [state.intent_seq]
        res = drainer.drain_task(fixture["seq"])
        assert res["executions"] == {}
        # Never replayed: the one-way command did not run again.
        assert not os.path.exists(marker)
        assert _state(journal, out["edit_id"],
                      fixture["seq"]).state == "outcome-unknown"
    finally:
        journal.close()


# -- dispositions (§13.4) ---------------------------------------------------------

def _unknown_with_dependents(mini_parts):
    bundle_dir, work = mini_parts
    manifest, journal = _open(bundle_dir)
    fixture, gate, ctx = _setup(journal, work)
    batch = _batch(journal, fixture, 3)
    unknown = _admit(journal, gate, ctx, fixture, "fs_write",
                     {"path": "unknown.txt", "content": "?\n"},
                     batch_seq=batch["seq"])
    state, _claim = _orphan(journal, fixture, unknown)
    drainer = _drainer(journal, bundle_dir, manifest, work,
                       drainer_id="drainer-new")
    rec = drainer.recover(task_seq=fixture["seq"])
    assert rec["outcome_unknown"] == [state.intent_seq]
    outcome_dep = _admit_dep(
        journal, gate, ctx, fixture, "fs_write",
        {"path": "outcome-dep.txt", "content": "ran\n"},
        [state.intent_seq], batch_seq=batch["seq"],
        depends_on=[{"seq": state.intent_seq,
                     "requirement": "outcome"}])
    success_dep = _admit_dep(
        journal, gate, ctx, fixture, "fs_write",
        {"path": "success-dep.txt", "content": "no\n"},
        [state.intent_seq], batch_seq=batch["seq"],
        depends_on=[{"seq": state.intent_seq,
                     "requirement": "success"}])
    return (bundle_dir, work, manifest, journal, fixture, drainer,
            state, outcome_dep, success_dep)


def test_dependents_wait_then_accept_unknown_releases_and_skips(mini):
    (bundle_dir, work, manifest, journal, fixture, drainer,
     state, outcome_dep, success_dep) = _unknown_with_dependents(mini)
    try:
        # Before any disposition: neither dependent runs and
        # neither is skipped — they WAIT for the operator (§13.4).
        res = drainer.drain_task(fixture["seq"])
        assert res["executions"] == {} and res["skips"] == []
        assert res["sync_reason"] == "blocked"
        for dep in (outcome_dep, success_dep):
            assert _state(journal, dep["edit_id"],
                          fixture["seq"]).state == "queued"

        entry = disposition_outcome_unknown(
            journal, state.intent_seq, "accept-unknown",
            detail="operator accepts the outcome is unknown")
        assert entry["kind"] == "queue_disposition"
        assert entry["source"] == "operator"
        folded = fold_queue(journal, task_seq=fixture["seq"])
        unknown_state = folded.by_seq[state.intent_seq]
        # History is not rewritten: still outcome-unknown, now
        # annotated by the disposition.
        assert unknown_state.state == "outcome-unknown"
        assert unknown_state.disposition == "accept-unknown"
        assert unknown_state.disposition_seq == entry["seq"]

        res = drainer.drain_task(fixture["seq"])
        # The outcome dependent proceeds on the honest basis;
        # the success dependent is skipped (§13.4's own text).
        assert _state(journal, outcome_dep["edit_id"],
                      fixture["seq"]).state == "executed"
        assert _read(os.path.join(work, "outcome-dep.txt")) == "ran\n"
        skipped = _state(journal, success_dep["edit_id"],
                         fixture["seq"])
        assert skipped.state == "skipped"
        assert not os.path.exists(
            os.path.join(work, "success-dep.txt"))
        assert journal.verify_chain()
    finally:
        journal.close()


def test_disposition_refusals_journal_nothing(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture, gate, ctx = _setup(journal, work)
        queued = _admit(journal, gate, ctx, fixture, "fs_write",
                        {"path": "q.txt", "content": "q\n"})
        qstate = _state(journal, queued["edit_id"], fixture["seq"])
        unknown = _admit(journal, gate, ctx, fixture, "fs_write",
                         {"path": "u.txt", "content": "u\n"})
        ustate, _c = _orphan(journal, fixture, unknown)
        _drainer(journal, bundle_dir, manifest, work,
                 drainer_id="drainer-new").recover(
                     task_seq=fixture["seq"])
        before = journal.tip()[0]
        # mark-executed does not exist (§13.4).
        with pytest.raises(ValueError):
            disposition_outcome_unknown(
                journal, ustate.intent_seq, "mark-executed")
        # A non-outcome-unknown intent cannot be dispositioned.
        with pytest.raises(ValueError):
            disposition_outcome_unknown(
                journal, qstate.intent_seq, "accept-unknown")
        # An unknown intent seq cannot be dispositioned.
        with pytest.raises(ValueError):
            disposition_outcome_unknown(
                journal, 999999, "accept-unknown")
        assert journal.tip()[0] == before  # nothing journaled
        # A real disposition lands; a second one is refused.
        disposition_outcome_unknown(
            journal, ustate.intent_seq, "re-propose",
            detail="operator will re-propose at fresh prices")
        mid = journal.tip()[0]
        with pytest.raises(ValueError):
            disposition_outcome_unknown(
                journal, ustate.intent_seq, "accept-unknown")
        assert journal.tip()[0] == mid
        assert DISPOSITIONS == (
            "accept-unknown", "compensate", "re-propose")
    finally:
        journal.close()


def test_compensate_disposition_annotates_without_satisfying(mini):
    (bundle_dir, work, manifest, journal, fixture, drainer,
     state, outcome_dep, success_dep) = _unknown_with_dependents(mini)
    try:
        disposition_outcome_unknown(
            journal, state.intent_seq, "compensate",
            detail="operator directs a compensation intent")
        folded = _state(journal, outcome_dep["edit_id"],
                        fixture["seq"])
        assert folded.state == "queued"  # still waiting
        res = drainer.drain_task(fixture["seq"])
        assert res["executions"] == {} and res["skips"] == []
        unknown_state = fold_queue(
            journal, task_seq=fixture["seq"]).by_seq[state.intent_seq]
        assert unknown_state.state == "outcome-unknown"
        assert unknown_state.disposition == "compensate"
    finally:
        journal.close()


# -- service exposure (§15.4) -----------------------------------------------------

def test_service_queue_view_and_task_response_carry_queue(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture, gate, ctx = _setup(journal, work)
        out = _admit(journal, gate, ctx, fixture, "fs_write",
                     {"path": "view.txt", "content": "v\n"})
        _state(journal, out["edit_id"], fixture["seq"])
        _orphan(journal, fixture, out)
        _drainer(journal, bundle_dir, manifest, work,
                 drainer_id="drainer-new").recover(
                     task_seq=fixture["seq"])
        _anchor(journal, bundle_dir, manifest)
    finally:
        journal.close()
    view = queue_view_for_bundle(bundle_dir, fixture["seq"])
    assert view["task_seq"] == fixture["seq"]
    assert "outcome-unknown" in view["states"]
    assert view["summary"]["counts"] == {"outcome-unknown": 1}
    assert view["summary"]["outcome_unknown"] == [
        view["intents"][0]["intent_seq"]]
    assert view["intents"][0]["state"] == "outcome-unknown"
    assert view["intents"][0]["recovery_observations"]
    with pytest.raises(KeyError):
        queue_view_for_bundle(bundle_dir, 424242)


def test_service_disposition_endpoint_journals_the_act(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture, gate, ctx = _setup(journal, work)
        out = _admit(journal, gate, ctx, fixture, "fs_write",
                     {"path": "svc.txt", "content": "s\n"})
        state, _c = _orphan(journal, fixture, out)
        _drainer(journal, bundle_dir, manifest, work,
                 drainer_id="drainer-new").recover(
                     task_seq=fixture["seq"])
        queued = _admit(journal, gate, ctx, fixture, "fs_write",
                        {"path": "svc-q.txt", "content": "q\n"})
        qstate = _state(journal, queued["edit_id"], fixture["seq"])
        _anchor(journal, bundle_dir, manifest)
    finally:
        journal.close()

    # A disposition for a non-unknown intent: refused, and the
    # journal does not move.
    manifest, journal = _open(bundle_dir)
    try:
        before = journal.tip()[0]
    finally:
        journal.close()
    with pytest.raises(ValueError):
        submit_disposition(bundle_dir, fixture["seq"], {
            "intent_seq": qstate.intent_seq,
            "disposition": "accept-unknown"})
    manifest, journal = _open(bundle_dir)
    try:
        assert journal.tip()[0] == before
    finally:
        journal.close()

    out = submit_disposition(bundle_dir, fixture["seq"], {
        "intent_seq": state.intent_seq,
        "disposition": "accept-unknown",
        "detail": "accepted over the service (Q-3 test)",
        "operator": "operator"})
    assert out["status"] == "dispositioned"
    assert out["disposition"]["disposition"] == "accept-unknown"
    assert out["queue"]["summary"]["dispositions"] == {
        state.intent_seq: "accept-unknown"}


def test_submit_task_response_carries_queue_vocabulary(mini):
    bundle_dir, work = mini
    payload = {
        "submitter": "q3-test",
        "instruction": "write one file",
        "script": [{"kind": "tool", "tool": "fs_write",
                    "args": {"path": "svc-task.txt",
                             "content": "via service\n"},
                    "reason": "Q-3 service queue test"}],
        "work_root": work,
    }
    out = submit_task(bundle_dir, payload,
                      policy=corpus_policy(work))
    assert out["status"] == "task-returned"
    queue = out["queue"]
    assert queue["summary"]["counts"] == {"executed": 1}
    assert queue["intents"][0]["state"] == "executed"
    # The provenance trail's queue section carries the claim.
    kinds = [r.get("kind") for r in
             out["provenance_trail"]["queue"]]
    assert "queue_claim" in kinds


def test_http_queue_and_disposition_routes(mini):
    bundle_dir, work = mini
    server = make_service_server(bundle_dir, host="127.0.0.1",
                                 port=0, policy=corpus_policy(work))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever,
                              daemon=True)
    thread.start()

    def call(method, path, payload=None):
        data = json.dumps(payload).encode() if payload is not None \
            else None
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}", data=data,
            headers={"Content-Type": "application/json"},
            method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode())

    try:
        code, body = call("POST", "/tasks", {
            "submitter": "q3-http", "instruction": "write one file",
            "script": [{"kind": "tool", "tool": "fs_write",
                        "args": {"path": "http.txt",
                                 "content": "http\n"},
                        "reason": "Q-3 HTTP queue test"}],
            "work_root": work})
        assert code == 200
        task_seq = body["result"]["task_seq"]
        intent_seq = body["queue"]["intents"][0]["intent_seq"]

        code, body = call("GET", f"/tasks/{task_seq}/queue")
        assert code == 200
        assert body["queue"]["summary"]["counts"] == {"executed": 1}
        assert "§15.4" in body["queue_exposure"]

        # The executed intent is not outcome-unknown: the
        # disposition route refuses it (400) and journals nothing.
        code, body = call("POST", f"/tasks/{task_seq}/dispositions",
                          {"intent_seq": intent_seq,
                           "disposition": "accept-unknown"})
        assert code == 400 and "outcome-unknown" in body["error"]
        code, body = call("GET", "/tasks/424242/queue")
        assert code == 404
        code, body = call("GET", "/health")
        assert code == 200 and "§15.4" in body["queue_exposure"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# -- SSE queue-event convention (§15.5) -------------------------------------------

def test_sse_queue_events_carry_queue_state(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture, gate, ctx = _setup(journal, work)
        batch = _batch(journal, fixture, 2)
        written = _admit(journal, gate, ctx, fixture, "fs_write",
                         {"path": "sse.txt", "content": "s\n"},
                         batch_seq=batch["seq"])
        unknown = _admit(journal, gate, ctx, fixture, "fs_write",
                         {"path": "sse-u.txt", "content": "u\n"},
                         batch_seq=batch["seq"])
        _orphan(journal, fixture, unknown)
        drainer = _drainer(journal, bundle_dir, manifest, work,
                           drainer_id="drainer-new")
        drainer.drain_task(fixture["seq"])
        disposition_outcome_unknown(
            journal, _state(journal, unknown["edit_id"],
                            fixture["seq"]).intent_seq,
            "accept-unknown", detail="SSE test disposition")
        record_queue_stop(journal, fixture["seq"],
                          cause="operator-stop", detail="SSE test")

        events = read_journal_events(journal)
        by_kind = {}
        for event in events:
            kind = event["row"].get("kind")
            by_kind.setdefault(kind, []).append(event)
        # The intent's own row transitions to queued (ADMIT).
        tool_rows = [e for e in events
                     if e["collection"] == "self_edits"
                     and e["row"].get("kind") == "tool"]
        assert tool_rows and all(
            e["queue_state"] == "queued" for e in tool_rows)
        assert by_kind["queue_claim"][0]["queue_state"] == "claimed"
        assert by_kind["tool_result"][0]["queue_state"] == "executed"
        assert by_kind["queue_outcome_unknown"][0][
            "queue_state"] == "outcome-unknown"
        assert by_kind["queue_disposition"][0][
            "queue_state"] == "outcome-unknown"
        assert by_kind["queue_stop"][0]["queue_state"] == "held"
        # Non-queue rows carry no queue_state at all.
        assert all("queue_state" not in e for e in events
                   if e["collection"] == "checkpoints")
        # The serialised payload carries the convention too.
        block = format_sse(by_kind["queue_claim"][0])
        data = json.loads(block.splitlines()[2][len("data: "):])
        assert data["queue_state"] == "claimed"
        # The mapping as a unit: verdicts and refusals.
        assert queue_state_for_row(
            "self_edits", {"kind": "tool",
                           "gate_verdict": "ESCALATE"}) == "escalated"
        assert queue_state_for_row(
            "experiences",
            {"kind": "tool_result",
             "record": {"execution": {"status": "refused"}}}) == \
            "refused-at-claim"
        assert queue_state_for_row(
            "experiences", {"kind": "task"}) is None
        assert journal.verify_chain()
    finally:
        journal.close()
