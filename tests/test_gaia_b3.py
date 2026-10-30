"""SI-44 (SI-41 RFC §15 B-3): Gaia execution tools + the task loop.

B-3 makes the execute class live (miniature-scoped) and builds the
task loop of RFC §10. Coverage:

- Registry/gate: LIVE_TOOLS_B3 is exactly LIVE_TOOLS_B2 plus the
  three execute tools (B-2's set unchanged); on a live context
  execute-class intents ADMIT with the B-3 label; an external cwd
  still ESCALATEs (the B-3 work-root hold) even under an
  external=admit policy; class-policy escalate ESCALATEs for the
  execute class; non-live contexts keep B-1 shadow byte-for-byte.
- Executor process discipline: shell_exec really runs (echo,
  exit_code 0, stdout captured) with an allowlist-scrubbed child
  environment (a sentinel variable in the parent's environment is
  invisible to the child); a non-zero exit is a COMPLETED
  execution carrying the real code; code_exec runs both source
  and path forms; test_run parses a real pytest summary's counts
  exactly as reported, and a non-pytest target is parse="failed"
  with counts 0 (never invented); a timeout is status timed-out
  with partial output labelled partial and exit_code -1; stream
  caps truncate and flag.
- Integrity: a test_run target swapped for an outside-pointing
  symlink between GATE and EXECUTE is refused by the executor's
  TOCTOU re-pricing, and the refusal is recognised as an
  execution-integrity failure (§9.2/§10.4.4).
- Task loop (§10.4 stop rules): completion (task_complete memory
  intent), budget-exhausted (policy cycle cap and the operator
  cycles cap), escalated (external cwd), reject-streak (two
  consecutive REJECTs halt; one REJECT then success does not),
  and proposer-exhausted (labelled NOT a §10.4 rule). The task
  entry is a real operator entry (no fixture label); every stop
  is journaled as a task_stop experience.
- Portability (§12.5(d)): after a task, the bundle is copied
  elsewhere; the copy loads and its journal answers the task
  trail (task entry, steps, provenance walk to the task).
- Demo bundles: a task on a demo bundle ESCALATEs (shadow) and
  nothing executes.
- CLI: `si task` drives a script file end to end; without
  --script it exits 2 honestly (no invented default mind).
"""
import json
import os
import shutil

import pytest

from si.bundle import init_bundle, load_bundle
from si.gaia.context import ToolContext
from si.gaia.corpus import corpus_policy
from si.gaia.executor import Executor, execute_tool_cycle
from si.gaia.registry import LIVE_TOOLS_B2, LIVE_TOOLS_B3, REGISTRY
from si.gaia.shadow import record_task_fixture, walk_tool_provenance
from si.gaia.task import (
    STOP_BUDGET_EXHAUSTED,
    STOP_ESCALATED,
    STOP_MIND_REPORTED_COMPLETE,
    STOP_PROPOSER_EXHAUSTED,
    STOP_REJECT_STREAK,
    _is_integrity_refusal,
    run_task,
)
from si.gate import (
    TOOL_B3_LIVE_EXECUTION,
    TOOL_B3_NOT_LIVE_EXECUTION,
    TOOL_SHADOW_EXECUTION,
    Gate,
    Intent,
)
from si.journal import open_journal
from si.tiny_seed import seed_tiny_bundle

TASK_SEQ = 7
TASK_REF = f"task@{TASK_SEQ}"
FIVE_STOP_RULES = {
    "budget-exhausted", "escalated", "reject-streak",
    "integrity-failure", "mind-reported-complete",
}


def _intent(tool, args, **kw):
    base = dict(kind="tool", tool=tool, args=args,
                reason="Carry out the scripted B-3 task step",
                parents=[TASK_REF], source_seqs=[TASK_SEQ])
    base.update(kw)
    return Intent(**base)


def _live_ctx(work_root, **kw):
    base = dict(tool_work_root=work_root, tool_task_seq=TASK_SEQ,
                tool_live_execution=True)
    base.update(kw)
    return ToolContext(**base)


def _gate(work_root, **policy_over):
    policy = corpus_policy(work_root)
    policy.update(policy_over)
    return Gate(policy)


# -- registry / live-set shape ---------------------------------------------------

def test_b3_live_set_is_b2_plus_exactly_the_execute_tools():
    assert LIVE_TOOLS_B3 == LIVE_TOOLS_B2 | {
        "shell_exec", "code_exec", "test_run"}
    # B-2's set is unchanged (its slice boundary, asserted by B-2).
    assert LIVE_TOOLS_B2 == {
        "fs_list", "fs_read", "fs_search", "fs_write", "fs_edit",
        "fs_mkdir", "journal_query", "bundle_status", "probe_run"}
    for name in ("shell_exec", "code_exec", "test_run"):
        assert REGISTRY[name].consequence_class == "execute"
        assert REGISTRY[name].reversibility == "one-way"


# -- gate: B-3 live verdicts --------------------------------------------------------

def test_live_context_admits_execute_class_with_b3_label(tmp_path):
    root = str(tmp_path / "work")
    os.makedirs(root)
    gate = _gate(root)
    for tool, args in (("shell_exec", {"command": "ls"}),
                       ("code_exec", {"language": "python",
                                      "source": "x = 1"}),
                       ("test_run", {"path": "tests"})):
        report = gate.evaluate(_intent(tool, args), None, _live_ctx(root))
        assert report.verdict == "ADMIT", tool
        assert report.execution == TOOL_B3_LIVE_EXECUTION, tool


def test_live_context_external_cwd_still_holds_at_escalate(tmp_path):
    root = str(tmp_path / "work")
    os.makedirs(root)
    outside = str(tmp_path / "elsewhere")
    gate = _gate(root, tool_class_policy={
        "read": "admit", "write": "admit", "execute": "admit",
        "external": "admit"})
    intent = _intent("shell_exec", {"command": "ls", "cwd": outside})
    info = gate.classify_tool_intent(intent, _live_ctx(root))
    assert info["effective_class"] == "external"
    report = gate.evaluate(intent, None, _live_ctx(root))
    assert report.verdict == "ESCALATE"  # B-3 live scope is the work root
    assert report.execution == TOOL_B3_NOT_LIVE_EXECUTION


def test_live_context_execute_class_policy_escalate_still_escalates(tmp_path):
    root = str(tmp_path / "work")
    gate = _gate(root, tool_class_policy={
        "read": "admit", "write": "admit", "execute": "escalate",
        "external": "escalate"})
    report = gate.evaluate(
        _intent("shell_exec", {"command": "ls"}), None, _live_ctx(root))
    assert report.verdict == "ESCALATE"
    assert "class policy" in report.execution
    assert "SI-44/B-3" in report.execution


def test_non_live_context_execute_tools_keep_shadow_exactly(tmp_path):
    root = str(tmp_path / "work")
    gate = _gate(root)
    ctx = ToolContext(tool_work_root=root, tool_task_seq=TASK_SEQ)
    for tool, args in (("shell_exec", {"command": "ls"}),
                       ("test_run", {"path": "tests"})):
        report = gate.evaluate(_intent(tool, args), None, ctx)
        assert report.verdict == "ESCALATE", tool
        assert report.execution == TOOL_SHADOW_EXECUTION, tool


# -- fixtures ------------------------------------------------------------------------

@pytest.fixture()
def mini(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "mini.si"), name="gaia-b3",
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


def _run(journal, bundle_dir, manifest, intent, work_root, gate=None,
         ctx=None):
    gate = gate or _gate(work_root)
    ctx = ctx or _live_ctx(work_root)
    return execute_tool_cycle(journal, bundle_dir, manifest, intent,
                              gate, ctx)


def _fixture_ctx(journal, work):
    fixture = record_task_fixture(journal, "B-3 executor scenario", work)
    return fixture, _live_ctx(work, tool_task_seq=fixture["seq"])


# -- executor: shell_exec --------------------------------------------------------------

def test_shell_exec_runs_and_the_child_env_is_scrubbed(mini, monkeypatch):
    bundle_dir, work = mini
    monkeypatch.setenv("SI_B3_SECRET_SENTINEL", "sentinel-value-12345")
    manifest, journal = _open(bundle_dir)
    try:
        fixture, ctx = _fixture_ctx(journal, work)
        out = _run(journal, bundle_dir, manifest,
                   Intent(kind="tool", tool="shell_exec",
                          args={"command": "echo hello-from-shell; env"},
                          reason="Echo and show the environment",
                          parents=[fixture["ref"]],
                          source_seqs=[fixture["seq"]]), work, ctx=ctx)
        rec = out["execution"]
        assert out["verdict"] == "ADMIT"
        assert rec["status"] == "completed"
        assert rec["exit_code"] == 0
        assert rec["reversibility"] == "one-way"
        assert "hello-from-shell" in rec["result"]["stdout"]
        # The allowlist env: PATH survives; the parent's sentinel
        # (and by extension any credential the runtime holds) does not.
        assert "PATH=" in rec["result"]["stdout"]
        assert "SI_B3_SECRET_SENTINEL" not in rec["result"]["stdout"]
        assert "sentinel-value-12345" not in json.dumps(rec)
        assert rec["post_state"]["exit_code"] == 0
        assert rec["post_state"]["cwd"] == os.path.realpath(work)
        assert rec["verify"]["result_conformance"] == "passed"
    finally:
        journal.close()


def test_shell_exec_nonzero_exit_is_completed_with_the_real_code(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture, ctx = _fixture_ctx(journal, work)
        out = _run(journal, bundle_dir, manifest,
                   Intent(kind="tool", tool="shell_exec",
                          args={"command": "echo oops >&2; exit 3"},
                          reason="Fail on purpose",
                          parents=[fixture["ref"]],
                          source_seqs=[fixture["seq"]]), work, ctx=ctx)
        rec = out["execution"]
        # A non-zero exit is data (§11.1.3): the execution completed.
        assert rec["status"] == "completed"
        assert rec["exit_code"] == 3
        assert rec["result"]["exit_code"] == 3
        assert "oops" in rec["result"]["stderr"]
        edits = journal.backend.find(
            "self_edits", filter={"edit_id": out["edit_id"]})
        assert edits[0]["result"] == "executed"
    finally:
        journal.close()


# -- executor: code_exec -----------------------------------------------------------------

def test_code_exec_source_and_path_forms(mini):
    bundle_dir, work = mini
    script = os.path.join(work, "calc.py")
    with open(script, "w") as f:
        f.write("print('path-form', 2 + 3)\n")
    manifest, journal = _open(bundle_dir)
    try:
        fixture, ctx = _fixture_ctx(journal, work)
        parents, seqs = [fixture["ref"]], [fixture["seq"]]
        out = _run(journal, bundle_dir, manifest,
                   Intent(kind="tool", tool="code_exec",
                          args={"language": "python",
                                "source": "print('source-form', 6 * 7)"},
                          reason="Run inline source", parents=parents,
                          source_seqs=seqs), work, ctx=ctx)
        rec = out["execution"]
        assert rec["status"] == "completed" and rec["exit_code"] == 0
        assert rec["result"]["stdout"] == "source-form 42\n"
        out = _run(journal, bundle_dir, manifest,
                   Intent(kind="tool", tool="code_exec",
                          args={"language": "python", "path": "calc.py"},
                          reason="Run the script file", parents=parents,
                          source_seqs=seqs), work, ctx=ctx)
        rec = out["execution"]
        assert rec["status"] == "completed" and rec["exit_code"] == 0
        assert rec["result"]["stdout"] == "path-form 5\n"
    finally:
        journal.close()


# -- executor: test_run --------------------------------------------------------------------

_TEST_FILE = (
    "import pytest\n\n"
    "def test_a():\n    assert True\n\n"
    "def test_b():\n    assert True\n\n"
    "def test_c():\n    assert False, 'boom'\n\n"
    "@pytest.mark.skip(reason='not today')\n"
    "def test_d():\n    pass\n"
)


def test_test_run_parses_the_suites_own_counts(mini):
    bundle_dir, work = mini
    with open(os.path.join(work, "test_sample.py"), "w") as f:
        f.write(_TEST_FILE)
    manifest, journal = _open(bundle_dir)
    try:
        fixture, ctx = _fixture_ctx(journal, work)
        out = _run(journal, bundle_dir, manifest,
                   Intent(kind="tool", tool="test_run",
                          args={"path": "test_sample.py"},
                          reason="Run the sample suite",
                          parents=[fixture["ref"]],
                          source_seqs=[fixture["seq"]]), work, ctx=ctx)
        rec = out["execution"]
        assert rec["status"] == "completed"
        assert rec["exit_code"] == 1  # pytest's real exit code (a failure ran)
        result = rec["result"]
        assert result["parse"] == "ok"
        assert (result["passed"], result["failed"],
                result["skipped"]) == (2, 1, 1)
        # Post-state is the suite's own parsed summary (§11.1.2).
        assert rec["post_state"]["passed"] == 2
        assert rec["post_state"]["failed"] == 1
        assert rec["post_state"]["skipped"] == 1
    finally:
        journal.close()


def test_test_run_non_pytest_target_is_parse_failed_never_invented(mini):
    bundle_dir, work = mini
    with open(os.path.join(work, "plain.txt"), "w") as f:
        f.write("this is not a test file at all\n")
    manifest, journal = _open(bundle_dir)
    try:
        fixture, ctx = _fixture_ctx(journal, work)
        out = _run(journal, bundle_dir, manifest,
                   Intent(kind="tool", tool="test_run",
                          args={"path": "plain.txt"},
                          reason="Run a non-test target",
                          parents=[fixture["ref"]],
                          source_seqs=[fixture["seq"]]), work, ctx=ctx)
        result = out["execution"]["result"]
        assert result["parse"] == "failed"
        assert (result["passed"], result["failed"],
                result["skipped"]) == (0, 0, 0)
        assert result["output"]  # the raw output is carried instead
    finally:
        journal.close()


# -- executor: timeout + stream caps ---------------------------------------------------------

def test_shell_exec_timeout_is_timed_out_partial_sentinel_exit(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture, ctx = _fixture_ctx(journal, work)
        out = _run(journal, bundle_dir, manifest,
                   Intent(kind="tool", tool="shell_exec",
                          args={"command": "echo before; sleep 5",
                                "timeout_seconds": 1},
                          reason="Run past the timeout",
                          parents=[fixture["ref"]],
                          source_seqs=[fixture["seq"]]), work, ctx=ctx)
        rec = out["execution"]
        assert rec["status"] == "timed-out"
        assert rec["partial"] is True
        # -1 is the documented no-exit sentinel: the killed process
        # produced no exit code (TIMEOUT_EXIT_CODE, SI-44 doc §2).
        assert rec["exit_code"] == -1
        assert rec["result"]["exit_code"] == -1
        assert rec["result"]["timed_out"] is True
        assert "before" in rec["result"]["stdout"]  # partial output kept
        assert rec["wall_time_seconds"] < 5
    finally:
        journal.close()


def test_stream_cap_truncates_and_flags(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture, ctx = _fixture_ctx(journal, work)
        gate = _gate(work, tool_max_result_bytes=256)
        out = _run(journal, bundle_dir, manifest,
                   Intent(kind="tool", tool="shell_exec",
                          args={"command":
                                "head -c 5000 /dev/zero | tr '\\0' 'x'"},
                          reason="Emit more than the stream cap",
                          parents=[fixture["ref"]],
                          source_seqs=[fixture["seq"]]),
                   work, gate=gate, ctx=ctx)
        rec = out["execution"]
        assert rec["status"] == "completed"
        assert rec["truncated"] is True
        assert len(rec["result"]["stdout"]) <= 256
        assert set(rec["result"]["stdout"]) == {"x"}
    finally:
        journal.close()


# -- integrity: TOCTOU at EXECUTE ---------------------------------------------------------------

def test_toctou_on_a_process_tool_is_an_integrity_refusal(mini, tmp_path):
    bundle_dir, work = mini
    target = os.path.join(work, "suite")
    os.makedirs(target)
    with open(os.path.join(target, "test_ok.py"), "w") as f:
        f.write("def test_ok():\n    assert True\n")
    outside = tmp_path / "outside-suite"
    outside.mkdir()
    manifest, journal = _open(bundle_dir)
    try:
        fixture, ctx = _fixture_ctx(journal, work)
        gate = _gate(work)
        intent = Intent(kind="tool", tool="test_run",
                        args={"path": "suite"}, reason="Run the suite",
                        parents=[fixture["ref"]],
                        source_seqs=[fixture["seq"]])
        report = gate.evaluate(intent, None, ctx)
        assert report.verdict == "ADMIT"
        edit_id = journal.record_self_edit(
            intent.kind, intent.to_dict(), intent.reason,
            report.verdict, report.to_dict())
        # The swap: the priced directory becomes a symlink out.
        shutil.rmtree(target)
        os.symlink(str(outside), target)
        executor = Executor(bundle_dir, manifest, journal, gate.policy, work)
        record = executor.execute_journaled(
            edit_id, expected_effective_class="execute")
        assert record["status"] == "refused"
        assert "TOCTOU" in record["detail"]
        assert _is_integrity_refusal(record)
    finally:
        journal.close()


# -- the task loop ------------------------------------------------------------------------------

def _task(bundle_dir, script, work, **kw):
    kw.setdefault("policy", corpus_policy(work))
    return run_task(bundle_dir, script, "Do the scripted B-3 task",
                    work_root=work, **kw)


def _complete(reason="The scripted work is done"):
    return Intent(kind="memory", collection="experiences",
                  record={"kind": "task_complete",
                          "detail": "scripted completion report"},
                  reason=reason)


def test_task_completes_via_task_complete_and_journals_everything(mini):
    bundle_dir, work = mini
    script = [
        Intent(kind="tool", tool="fs_write",
               args={"path": "done.txt", "content": "task output\n"},
               reason="Write the task output"),
        Intent(kind="tool", tool="shell_exec",
               args={"command": "echo task-shell-step"},
               reason="Run the shell step"),
        _complete(),
    ]
    res = _task(bundle_dir, script, work)
    assert res["stop_reason"] == STOP_MIND_REPORTED_COMPLETE
    assert res["stop_reason"] in FIVE_STOP_RULES
    assert res["cycles_used"] == 2  # the completion is not a tool cycle
    assert [s["verdict"] for s in res["steps"]] == ["ADMIT"] * 3
    assert res["steps"][0]["execution_status"] == "completed"
    assert res["steps"][1]["tool"] == "shell_exec"
    assert res["steps"][1]["effective_class"] == "execute"
    assert res["steps"][2]["execution_status"] == "materialized"
    assert open(os.path.join(work, "done.txt")).read() == "task output\n"

    manifest, journal = _open(bundle_dir)
    try:
        # The task entry is a REAL operator entry — no fixture label.
        tasks = journal.backend.find("experiences",
                                     filter={"kind": "task"})
        assert len(tasks) == 1
        record = tasks[0]["record"]
        assert record["source"] == "operator"
        assert "fixture" not in record
        assert record["instruction"] == "Do the scripted B-3 task"
        assert tasks[0]["seq"] == res["task_seq"]
        # The stop is journaled with its cause.
        stops = journal.backend.find("experiences",
                                     filter={"kind": "task_stop"})
        assert len(stops) == 1
        assert stops[0]["record"]["cause"] == \
            STOP_MIND_REPORTED_COMPLETE
        # The first step's provenance walks to the task entry.
        walk = walk_tool_provenance(journal, res["steps"][0]["edit_id"])
        assert walk["task"] is not None
        assert walk["task"]["row"]["seq"] == res["task_seq"]
        assert journal.verify_chain()
    finally:
        journal.close()
    load_bundle(bundle_dir)  # re-anchored: the bundle stays loadable


def test_task_budget_exhausted_by_policy_cycle_cap(mini):
    bundle_dir, work = mini
    script = [
        Intent(kind="tool", tool="fs_write",
               args={"path": f"step-{i}.txt", "content": "x"},
               reason=f"Write step {i}")
        for i in range(3)
    ]
    policy = corpus_policy(work, {"max_tool_cycles_per_task": 2})
    res = _task(bundle_dir, script, work, policy=policy)
    assert res["stop_reason"] == STOP_BUDGET_EXHAUSTED
    assert res["cycles_used"] == 2
    assert len(res["steps"]) == 2
    assert os.path.exists(os.path.join(work, "step-1.txt"))
    assert not os.path.exists(os.path.join(work, "step-2.txt"))


def test_task_budget_exhausted_by_operator_cycles_cap(mini):
    bundle_dir, work = mini
    script = [
        Intent(kind="tool", tool="fs_write",
               args={"path": f"op-{i}.txt", "content": "x"},
               reason=f"Write op step {i}")
        for i in range(2)
    ]
    res = _task(bundle_dir, script, work, cycles=1)
    assert res["stop_reason"] == STOP_BUDGET_EXHAUSTED
    assert res["cycles_used"] == 1
    assert not os.path.exists(os.path.join(work, "op-1.txt"))


def test_task_escalated_halts_at_the_escalation(mini, tmp_path):
    bundle_dir, work = mini
    outside = str(tmp_path / "elsewhere")
    script = [
        Intent(kind="tool", tool="shell_exec",
               args={"command": "ls", "cwd": outside},
               reason="Run outside the work root"),
        Intent(kind="tool", tool="fs_write",
               args={"path": "never.txt", "content": "x"},
               reason="Never reached"),
    ]
    res = _task(bundle_dir, script, work)
    assert res["stop_reason"] == STOP_ESCALATED
    assert len(res["steps"]) == 1
    assert res["steps"][0]["verdict"] == "ESCALATE"
    assert res["steps"][0]["effective_class"] == "external"
    assert not os.path.exists(os.path.join(work, "never.txt"))


def test_task_reject_streak_halts_but_a_single_reject_does_not(mini):
    bundle_dir, work = mini
    bad = Intent(kind="tool", tool="fs_delete_everything",
                 args={"path": "."}, reason="An unregistered tool")
    bad2 = Intent(kind="tool", tool="fs_write",
                  args={"path": "x.txt"},  # missing required content
                  reason="A schema-violating write")
    res = _task(bundle_dir, [bad, bad2], work)
    assert res["stop_reason"] == STOP_REJECT_STREAK
    assert [s["verdict"] for s in res["steps"]] == ["REJECT", "REJECT"]

    # One REJECT is information the proposer may route around.
    bundle2 = init_bundle(str(os.path.join(os.path.dirname(bundle_dir),
                                           "mini2.si")),
                          name="gaia-b3-2", journal_backend="embedded")
    seed_tiny_bundle(bundle2)
    script = [
        Intent(kind="tool", tool="fs_delete_everything",
               args={"path": "."}, reason="An unregistered tool"),
        Intent(kind="tool", tool="fs_write",
               args={"path": "recovered.txt", "content": "ok\n"},
               reason="Route around the rejection"),
        _complete(),
    ]
    res = _task(bundle2, script, work)
    assert res["stop_reason"] == STOP_MIND_REPORTED_COMPLETE
    assert [s["verdict"] for s in res["steps"]] == \
        ["REJECT", "ADMIT", "ADMIT"]
    assert open(os.path.join(work, "recovered.txt")).read() == "ok\n"


def test_task_proposer_exhausted_is_labelled_not_a_stop_rule(mini):
    bundle_dir, work = mini
    script = [
        Intent(kind="tool", tool="fs_write",
               args={"path": "only-step.txt", "content": "x"},
               reason="The only scripted step"),
    ]
    res = _task(bundle_dir, script, work)
    assert res["stop_reason"] == STOP_PROPOSER_EXHAUSTED
    assert res["stop_reason"] not in FIVE_STOP_RULES
    assert "NOT one of" in res["stop_detail"]
    assert res["cycles_used"] == 1
    assert os.path.exists(os.path.join(work, "only-step.txt"))


def test_task_with_a_mind_proposer_perceives_tagged_observations(mini):
    bundle_dir, work = mini

    class ScriptedMind:
        name = "scripted-b3-mind"

        def __init__(self):
            self.saw_tool_result = False

        def perceive(self, journal):
            return []

        def propose(self, perceptions, ctx):
            task = perceptions[0]
            assert task["kind"] == "task" \
                and task["source"] == "operator"
            ref, seq = task["ref"], task["seq"]
            observed = [p for p in perceptions
                        if p.get("kind") == "tool_result"]
            if not observed:
                return Intent(
                    kind="tool", tool="fs_write",
                    args={"path": "mind-made.txt",
                          "content": "by the mind proposer\n"},
                    reason="Write the file the task asks for",
                    parents=[ref], source_seqs=[seq])
            # §4.5: the result returns as a provenance-tagged
            # observation — source names the producing tool.
            self.saw_tool_result = \
                observed[0].get("source") == "tool:fs_write"
            return Intent(
                kind="memory", collection="experiences",
                record={"kind": "task_complete"},
                reason="The file is written; reporting completion",
                parents=[ref], source_seqs=[seq])

    mind = ScriptedMind()
    res = _task(bundle_dir, mind, work)
    assert res["stop_reason"] == STOP_MIND_REPORTED_COMPLETE
    assert mind.saw_tool_result
    assert open(os.path.join(work, "mind-made.txt")).read() == \
        "by the mind proposer\n"


# -- portability (§12.5(d)) ------------------------------------------------------------------------

def test_task_trail_is_portable_with_the_bundle(mini, tmp_path):
    bundle_dir, work = mini
    script = [
        Intent(kind="tool", tool="shell_exec",
               args={"command": "echo portable-step"},
               reason="A step whose trail must travel"),
        _complete(),
    ]
    res = _task(bundle_dir, script, work)
    assert res["stop_reason"] == STOP_MIND_REPORTED_COMPLETE

    moved = str(tmp_path / "moved" / "mini.si")
    os.makedirs(os.path.dirname(moved))
    shutil.copytree(bundle_dir, moved)
    manifest = load_bundle(moved)  # the far side loads + verifies
    journal = open_journal(moved, manifest)
    try:
        tasks = journal.backend.find("experiences",
                                     filter={"kind": "task"})
        assert len(tasks) == 1
        assert tasks[0]["seq"] == res["task_seq"]
        stops = journal.backend.find("experiences",
                                     filter={"kind": "task_stop"})
        assert stops[0]["record"]["cause"] == \
            STOP_MIND_REPORTED_COMPLETE
        # journal_query's own machinery (the journal backend) and
        # the provenance walk answer on the far side.
        walk = walk_tool_provenance(journal, res["steps"][0]["edit_id"])
        assert walk["task"] is not None
        assert walk["task"]["row"]["seq"] == res["task_seq"]
        outcomes = journal.backend.find(
            "experiences",
            filter={"self_edit_id": res["steps"][0]["edit_id"]})
        execution = outcomes[0]["record"]["execution"]
        assert execution["status"] == "completed"
        assert execution["result"]["stdout"] == "portable-step\n"
        assert journal.verify_chain()
    finally:
        journal.close()


# -- demo bundles stay shadow ---------------------------------------------------------------------------

def test_demo_bundle_task_escalates_and_nothing_executes(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "demo.si"), name="demo-b3",
                             journal_backend="embedded")
    work = str(tmp_path / "work")
    os.makedirs(work)
    script = [
        Intent(kind="tool", tool="shell_exec",
               args={"command": "echo never-runs"},
               reason="Try to run on a demo bundle"),
    ]
    res = run_task(bundle_dir, script, "Demo task", work_root=work,
                   policy=corpus_policy(work))
    assert res["stop_reason"] == STOP_ESCALATED
    assert res["steps"][0]["verdict"] == "ESCALATE"
    assert res["steps"][0]["execution_status"] == TOOL_SHADOW_EXECUTION
    manifest, journal = _open(bundle_dir)
    try:
        outcomes = journal.backend.find(
            "experiences",
            filter={"self_edit_id": res["steps"][0]["edit_id"]})
        assert outcomes[0]["record"]["execution"]["execution"] == \
            TOOL_SHADOW_EXECUTION
    finally:
        journal.close()


# -- CLI ---------------------------------------------------------------------------------------------

def _install_policy(bundle_dir, **over):
    """Write a new policy.json and re-anchor it in the manifest —
    the test-fixture stand-in for a journaled operator policy act
    (the precedent of tests/test_admit.py's helper of this name).
    A stock bundle's policy allowlists NO tools (fail-closed,
    §6.2), so `si task` against it can only REJECT; the CLI test
    first performs the operator's opt-in."""
    from si.bundle import (
        _anchor_checkpoint,
        _write_manifest,
        canonical_json,
        load_policy,
        sha256_hex,
    )
    manifest = load_bundle(bundle_dir)
    policy = load_policy(bundle_dir)
    policy.update(over)
    data = canonical_json(policy)
    with open(os.path.join(bundle_dir, "policy.json"), "wb") as f:
        f.write(data)
    manifest["constitution"] = dict(manifest["constitution"])
    manifest["constitution"]["policy_hash"] = "sha256:" + sha256_hex(data)
    journal = open_journal(bundle_dir, manifest)
    try:
        anchored = _anchor_checkpoint(journal, manifest,
                                      head=manifest["head"],
                                      note="test policy install")
        _write_manifest(bundle_dir, anchored)
    finally:
        journal.close()
    return load_bundle(bundle_dir)


def test_cli_task_with_a_script_file(mini, tmp_path, capsys):
    from si.cli import main
    bundle_dir, work = mini
    _install_policy(bundle_dir, allowed_tools=sorted(REGISTRY))
    script = [
        {"kind": "tool", "tool": "fs_write",
         "args": {"path": "cli-made.txt", "content": "via si task\n"},
         "reason": "Write the CLI task's file"},
        {"kind": "tool", "tool": "shell_exec",
         "args": {"command": "echo cli-shell-step"},
         "reason": "Run the CLI task's shell step"},
        {"kind": "memory", "record": {"kind": "task_complete"},
         "reason": "Report completion"},
    ]
    script_path = tmp_path / "script.json"
    script_path.write_text(json.dumps(script))
    rc = main(["task", bundle_dir, "--instruction", "CLI B-3 task",
               "--root", work, "--script", str(script_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "mind-reported-complete" in out
    assert open(os.path.join(work, "cli-made.txt")).read() == \
        "via si task\n"
    load_bundle(bundle_dir)


def test_cli_task_without_script_exits_2_honestly(mini, capsys):
    from si.cli import main
    bundle_dir, _work = mini
    rc = main(["task", bundle_dir, "--instruction", "No proposer"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "scripted proposer" in err
    assert "no trained mind" in err
