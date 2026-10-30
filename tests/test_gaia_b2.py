"""SI-43 (SI-41 RFC §15 B-2): Gaia executor + filesystem/house tools.

B-2 makes execution live — miniature-scoped only. Coverage:

- Gate: on a live (miniature) context, in-scope read/write intents
  ADMIT with the live label; external-effective intents ESCALATE
  even under an external=admit policy (the B-2 work-root hold);
  class-policy escalate ESCALATEs; non-live contexts keep B-1
  shadow exactly. (Execute-class verdicts moved in SI-44/B-3 —
  they ADMIT now; see test_live_context_execute_tools_admit_in_b3
  and the SI-44 slice doc §2 for the two superseded assertions in
  this file.)
- Executor: the scripted multi-step scenario (list -> read ->
  write -> edit -> journal_query, RFC §14.1) completes on a real
  miniature bundle with a fully walkable journal chain to its task
  entry; every execution record carries measured fields (hash,
  bytes, wall time) that recompute from the actual results.
- Journaled-args rule (§9.2): a non-ADMIT journaled intent is
  refused; a class-changing TOCTOU (file swapped for an
  outside-pointing symlink between GATE and EXECUTE) is refused
  and the outside content never lands in the record.
- Escape cases: symlink-out and path-out price to external at the
  gate and never execute.
- Pre-image + compensation (§11.2): an overwriting fs_write
  journals the pre-image; a NEW gated fs_write of the pre-image
  (built by compensation_intent) restores the original bytes.
- House tools: bundle_status derives from the verified manifest;
  probe_run measures the miniature's real suites (measured-pass)
  and honestly reports not-evaluated on a demo bundle.
- Loop paths: the miniature loop executes an ADMITted tool intent;
  the demo loop keeps shadow behaviour (ESCALATE, nothing written).
- Failures are results: fs_edit with absent/ambiguous `old`,
  fs_write into a missing parent, fs_mkdir over a file — each a
  journaled failed execution, the filesystem untouched.
"""
import hashlib
import json
import os

import pytest

from si.bundle import (
    canonical_json,
    init_bundle,
    load_bundle,
    load_policy,
    sha256_hex,
)
from si.gaia.context import ToolContext
from si.gaia.corpus import corpus_policy
from si.gaia.executor import (
    Executor,
    compensation_intent,
    execute_tool_cycle,
    run_tool_intent,
)
from si.gaia.registry import LIVE_TOOLS_B2, REGISTRY
from si.gaia.shadow import (
    record_task_fixture,
    walk_tool_provenance,
)
from si.gate import (
    TOOL_B3_LIVE_EXECUTION,
    TOOL_LIVE_EXECUTION,
    TOOL_SHADOW_EXECUTION,
    Gate,
    Intent,
)
from si.journal import open_journal
from si.tiny_seed import seed_tiny_bundle

TASK_SEQ = 7
TASK_REF = f"task@{TASK_SEQ}"


def _intent(tool, args, **kw):
    base = dict(kind="tool", tool=tool, args=args,
                reason="Carry out the scripted B-2 task step",
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


# -- registry / live-set shape -------------------------------------------------

def test_live_set_is_exactly_the_rfc_b2_tools():
    assert LIVE_TOOLS_B2 == {
        "fs_list", "fs_read", "fs_search", "fs_write", "fs_edit",
        "fs_mkdir", "journal_query", "bundle_status", "probe_run"}
    for name in LIVE_TOOLS_B2:
        assert REGISTRY[name].consequence_class in ("read", "write")
    for name in ("shell_exec", "code_exec", "test_run"):
        assert name not in LIVE_TOOLS_B2


# -- gate: live verdicts ----------------------------------------------------------

def test_live_context_admits_in_scope_read_and_write(tmp_path):
    root = str(tmp_path / "work")
    os.makedirs(root)
    gate = _gate(root)
    for tool, args in (("fs_list", {"path": "."}),
                       ("fs_read", {"path": "a.txt"}),
                       ("fs_write", {"path": "a.txt", "content": "x"}),
                       ("fs_mkdir", {"path": "sub"}),
                       ("journal_query", {"collection": "experiences"}),
                       ("bundle_status", {}),
                       ("probe_run", {"suite": "identity"})):
        report = gate.evaluate(_intent(tool, args), None, _live_ctx(root))
        assert report.verdict == "ADMIT", tool
        assert report.execution == TOOL_LIVE_EXECUTION, tool


def test_live_context_execute_tools_admit_in_b3(tmp_path):
    # SI-44/B-3 SUPERSEDES the B-2 form of this test (which asserted
    # execute-class intents ESCALATE naming B-3): B-3 is the slice
    # that makes the execute class live, so on a live context these
    # now ADMIT with the B-3 live label. Documented in
    # docs/SI-44-gaia-b3-execution-task-loop.md §2.
    root = str(tmp_path / "work")
    gate = _gate(root)
    for tool, args in (("shell_exec", {"command": "ls"}),
                       ("code_exec", {"language": "python",
                                      "source": "x = 1"}),
                       ("test_run", {"path": "tests"})):
        report = gate.evaluate(_intent(tool, args), None, _live_ctx(root))
        assert report.verdict == "ADMIT", tool
        assert report.execution == TOOL_B3_LIVE_EXECUTION, tool


def test_live_context_external_holds_at_escalate_even_if_policy_admits(tmp_path):
    root = str(tmp_path / "work")
    outside = str(tmp_path / "outside.txt")
    gate = _gate(root, tool_class_policy={
        "read": "admit", "write": "admit", "execute": "admit",
        "external": "admit"})
    intent = _intent("fs_read", {"path": outside})
    info = gate.classify_tool_intent(intent, _live_ctx(root))
    assert info["effective_class"] == "external"
    report = gate.evaluate(intent, None, _live_ctx(root))
    assert report.verdict == "ESCALATE"  # B-2 live scope is the work root


def test_live_context_class_policy_escalate_still_escalates(tmp_path):
    root = str(tmp_path / "work")
    gate = _gate(root, tool_class_policy={
        "read": "admit", "write": "escalate", "execute": "admit",
        "external": "escalate"})
    report = gate.evaluate(
        _intent("fs_write", {"path": "a.txt", "content": "x"}),
        None, _live_ctx(root))
    assert report.verdict == "ESCALATE"
    assert "class policy" in report.execution


def test_non_live_context_keeps_shadow_exactly(tmp_path):
    root = str(tmp_path / "work")
    gate = _gate(root)
    ctx = ToolContext(tool_work_root=root, tool_task_seq=TASK_SEQ)
    report = gate.evaluate(
        _intent("fs_write", {"path": "a.txt", "content": "x"}), None, ctx)
    assert report.verdict == "ESCALATE"
    assert report.execution == TOOL_SHADOW_EXECUTION


# -- fixtures: a real miniature bundle + work root --------------------------------

@pytest.fixture()
def mini(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "mini.si"), name="gaia-b2",
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    work = tmp_path / "work"
    work.mkdir()
    (work / "notes").mkdir()
    (work / "notes" / "plan.md").write_text("# plan\nstep one\nstep two\n")
    (work / "data.bin").write_bytes(b"\x00\x01\x02\xff" * 8)
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


# -- the §14.1 scripted scenario, end to end ---------------------------------------

def test_scripted_scenario_walks_the_whole_chain(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture = record_task_fixture(
            journal, "List notes, read the plan, extend it, check "
                     "the journal", work,
            budgets={"max_tool_cycles_per_task": 16})
        task_seq, task_ref = fixture["seq"], fixture["ref"]
        ctx = _live_ctx(work, tool_task_seq=task_seq)

        def step(tool, args, parents, seqs):
            intent = Intent(kind="tool", tool=tool, args=args,
                            reason=f"Scripted scenario step: {tool}",
                            parents=parents, source_seqs=seqs)
            return _run(journal, bundle_dir, manifest, intent, work,
                        ctx=ctx)

        def outcome_seq(edit_id):
            rows = journal.backend.find(
                "experiences", filter={"self_edit_id": edit_id})
            assert rows
            return rows[0]["seq"]

        steps = []
        out = step("fs_list", {"path": "notes"}, [task_ref], [task_seq])
        steps.append(out)
        assert out["verdict"] == "ADMIT"
        assert out["execution"]["status"] == "completed"
        names = {e["name"] for e in out["execution"]["result"]["entries"]}
        assert names == {"plan.md"}

        out = step("fs_read", {"path": "notes/plan.md"},
                   [task_ref, steps[-1]["edit_id"]],
                   [task_seq, outcome_seq(steps[-1]["edit_id"])])
        steps.append(out)
        assert out["execution"]["result"]["encoding"] == "text"
        assert "step one" in out["execution"]["result"]["content"]

        out = step("fs_write", {"path": "notes/out.txt",
                                "content": "result of the scenario\n"},
                   [task_ref, steps[-1]["edit_id"]],
                   [task_seq, outcome_seq(steps[-1]["edit_id"])])
        steps.append(out)
        rec = out["execution"]
        assert rec["status"] == "completed"
        assert rec["pre_image"]["exists"] is False
        written = open(os.path.join(work, "notes", "out.txt"), "rb").read()
        assert written == b"result of the scenario\n"
        assert rec["result"]["sha256"] == \
            "sha256:" + hashlib.sha256(written).hexdigest()
        assert rec["post_state"]["sha256"] == rec["result"]["sha256"]

        out = step("fs_edit", {"path": "notes/out.txt",
                               "old": "result", "new": "outcome"},
                   [task_ref, steps[-1]["edit_id"]],
                   [task_seq, outcome_seq(steps[-1]["edit_id"])])
        steps.append(out)
        assert out["execution"]["result"]["replacements"] == 1
        assert open(os.path.join(work, "notes", "out.txt")).read() == \
            "outcome of the scenario\n"

        out = step("journal_query", {"collection": "experiences",
                                     "limit": 50},
                   [task_ref, steps[-1]["edit_id"]],
                   [task_seq, outcome_seq(steps[-1]["edit_id"])])
        steps.append(out)
        assert out["execution"]["status"] == "completed"
        kinds = {r["kind"] for r in out["execution"]["result"]["rows"]}
        assert "task" in kinds and "tool_result" in kinds

        # Every step's provenance walks back to the task entry, and
        # every record's measured fields recompute from its result.
        for s in steps:
            walk = walk_tool_provenance(journal, s["edit_id"])
            assert walk["task"] is not None
            assert walk["task"]["row"]["seq"] == task_seq
            rec = s["execution"]
            raw = canonical_json(rec["result"])
            assert rec["result_hash"] == \
                "sha256:" + hashlib.sha256(raw).hexdigest()
            assert rec["result_bytes"] == len(raw)
            assert rec["wall_time_seconds"] >= 0
            assert rec["exit_code"] is None  # no process in B-2
            assert rec["verify"]["result_conformance"] == "passed"
            assert rec["verify"]["post_state"] == "passed"
        assert journal.verify_chain()
    finally:
        journal.close()


def test_fs_search_and_mkdir_and_binary_read(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture = record_task_fixture(journal, "Search and mkdir", work)
        ctx = _live_ctx(work, tool_task_seq=fixture["seq"])
        parents, seqs = [fixture["ref"]], [fixture["seq"]]

        out = _run(journal, bundle_dir, manifest,
                   Intent(kind="tool", tool="fs_search",
                          args={"path": "notes", "pattern": "step"},
                          reason="Find the steps", parents=parents,
                          source_seqs=seqs), work, ctx=ctx)
        matches = out["execution"]["result"]["matches"]
        assert len(matches) == 2
        assert matches[0]["path"] == os.path.join("notes", "plan.md")
        assert matches[0]["line"] == 2

        out = _run(journal, bundle_dir, manifest,
                   Intent(kind="tool", tool="fs_mkdir",
                          args={"path": "notes/archive"},
                          reason="Make the archive dir", parents=parents,
                          source_seqs=seqs), work, ctx=ctx)
        assert out["execution"]["result"] == {
            "path": os.path.join("notes", "archive"), "created": True}
        assert os.path.isdir(os.path.join(work, "notes", "archive"))
        # Second mkdir is a completed no-op, honestly labelled.
        out = _run(journal, bundle_dir, manifest,
                   Intent(kind="tool", tool="fs_mkdir",
                          args={"path": "notes/archive"},
                          reason="Make it again", parents=parents,
                          source_seqs=seqs), work, ctx=ctx)
        assert out["execution"]["result"]["created"] is False

        out = _run(journal, bundle_dir, manifest,
                   Intent(kind="tool", tool="fs_read",
                          args={"path": "data.bin"},
                          reason="Read the binary", parents=parents,
                          source_seqs=seqs), work, ctx=ctx)
        rec = out["execution"]["result"]
        assert rec["encoding"] == "base64"
        import base64
        assert base64.b64decode(rec["content"]) == b"\x00\x01\x02\xff" * 8
    finally:
        journal.close()


def test_fs_read_truncation_is_flagged_and_measured(mini):
    bundle_dir, work = mini
    big = os.path.join(work, "big.txt")
    with open(big, "w") as f:
        f.write("x" * 5000)
    manifest, journal = _open(bundle_dir)
    try:
        fixture = record_task_fixture(journal, "Read big", work)
        ctx = _live_ctx(work, tool_task_seq=fixture["seq"])
        out = _run(journal, bundle_dir, manifest,
                   Intent(kind="tool", tool="fs_read",
                          args={"path": "big.txt", "max_bytes": 100},
                          reason="Read a prefix", parents=[fixture["ref"]],
                          source_seqs=[fixture["seq"]]), work, ctx=ctx)
        rec = out["execution"]["result"]
        assert rec["truncated"] is True and len(rec["content"]) == 100
        assert out["execution"]["post_state"]["size"] == 5000
    finally:
        journal.close()


# -- house tools ---------------------------------------------------------------------

def test_bundle_status_and_probe_run_measure_the_miniature(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture = record_task_fixture(journal, "House tools", work)
        ctx = _live_ctx(work, tool_task_seq=fixture["seq"])
        parents, seqs = [fixture["ref"]], [fixture["seq"]]
        out = _run(journal, bundle_dir, manifest,
                   Intent(kind="tool", tool="bundle_status", args={},
                          reason="Status", parents=parents,
                          source_seqs=seqs), work, ctx=ctx)
        result = out["execution"]["result"]
        assert result["head"] == manifest["head"] == "base"
        assert result["deltas"] == 0
        assert result["last_seq"] == manifest["journal"]["last_seq"]
        assert result["policy_hash"] == \
            manifest["constitution"]["policy_hash"]

        for suite in ("identity", "capability"):
            out = _run(journal, bundle_dir, manifest,
                       Intent(kind="tool", tool="probe_run",
                              args={"suite": suite}, reason="Probe",
                              parents=parents, source_seqs=seqs),
                       work, ctx=ctx)
            rec = out["execution"]["result"]
            assert rec["label"] == "measured-pass", rec
            assert isinstance(rec["measured"], float)
    finally:
        journal.close()


def test_probe_run_on_demo_bundle_is_honest_not_evaluated(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "demo.si"), name="demo-b2",
                             journal_backend="embedded")
    work = str(tmp_path / "work")
    os.makedirs(work)
    manifest, journal = _open(bundle_dir)
    try:
        executor = Executor(bundle_dir, manifest, journal,
                            load_policy(bundle_dir), work)
        result, _post, _pre, post_ok, _d = executor._do_probe_run(
            {"suite": "identity"})
        assert result["label"] == "not-evaluated"
        assert result["measured"] is None
        assert post_ok is True  # a successful call, an honest non-result
    finally:
        journal.close()


# -- escapes, TOCTOU, and the journaled-args rule --------------------------------------

def test_path_out_and_symlink_out_escalate_and_never_execute(mini, tmp_path):
    bundle_dir, work = mini
    outside = tmp_path / "secret.txt"
    outside.write_text("outside-secret")
    link = os.path.join(work, "link.txt")
    os.symlink(str(outside), link)
    manifest, journal = _open(bundle_dir)
    try:
        fixture = record_task_fixture(journal, "Escape attempts", work)
        ctx = _live_ctx(work, tool_task_seq=fixture["seq"])
        for args in ({"path": str(outside)}, {"path": "link.txt"}):
            out = _run(journal, bundle_dir, manifest,
                       Intent(kind="tool", tool="fs_read", args=args,
                              reason="Try to read outside",
                              parents=[fixture["ref"]],
                              source_seqs=[fixture["seq"]]), work, ctx=ctx)
            assert out["verdict"] == "ESCALATE"
            assert out["effective_class"] == "external"
            record = out["execution"]
            assert "result" not in record  # no content was perceived
            assert "outside-secret" not in json.dumps(record)
    finally:
        journal.close()


def test_toctou_class_change_is_refused_at_execution(mini, tmp_path):
    bundle_dir, work = mini
    target = os.path.join(work, "swap.txt")
    with open(target, "w") as f:
        f.write("inside content")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside-secret")
    manifest, journal = _open(bundle_dir)
    try:
        fixture = record_task_fixture(journal, "TOCTOU", work)
        ctx = _live_ctx(work, tool_task_seq=fixture["seq"])
        gate = _gate(work)
        intent = Intent(kind="tool", tool="fs_read",
                        args={"path": "swap.txt"}, reason="Read swap",
                        parents=[fixture["ref"]],
                        source_seqs=[fixture["seq"]])
        report = gate.evaluate(intent, None, ctx)
        assert report.verdict == "ADMIT"
        edit_id = journal.record_self_edit(
            intent.kind, intent.to_dict(), intent.reason,
            report.verdict, report.to_dict())
        # The swap: the priced file becomes a symlink out of scope.
        os.remove(target)
        os.symlink(str(outside), target)
        executor = Executor(bundle_dir, manifest, journal, gate.policy, work)
        record = executor.execute_journaled(
            edit_id, expected_effective_class="read")
        assert record["status"] == "refused"
        assert "TOCTOU" in record["detail"]
        assert "outside-secret" not in json.dumps(record)
    finally:
        journal.close()


def test_executor_refuses_non_admitted_and_b3_journaled_intents(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        executor = Executor(bundle_dir, manifest, journal,
                            _gate(work).policy, work)
        # An ESCALATEd (never admitted) fs_read.
        intent = Intent(kind="tool", tool="fs_read",
                        args={"path": "notes/plan.md"}, reason="Read",
                        parents=["task@1"], source_seqs=[1])
        edit_id = journal.record_self_edit(
            "tool", intent.to_dict(), intent.reason, "ESCALATE", {})
        record = executor.execute_journaled(edit_id)
        assert record["status"] == "refused"
        assert "not ADMIT" in record["detail"]
        # SI-44/B-3 SUPERSEDES the B-2 form of this case (which
        # asserted a fabricated ADMIT for shell_exec is refused as
        # out-of-slice): shell_exec is live in B-3, so a journaled
        # ADMIT for it now EXECUTES for real. Documented in
        # docs/SI-44-gaia-b3-execution-task-loop.md §2.
        intent = Intent(kind="tool", tool="shell_exec",
                        args={"command": "echo hi"}, reason="Run",
                        parents=["task@1"], source_seqs=[1])
        edit_id = journal.record_self_edit(
            "tool", intent.to_dict(), intent.reason, "ADMIT", {})
        record = executor.execute_journaled(edit_id)
        assert record["status"] == "completed"
        assert record["exit_code"] == 0
        assert record["result"]["stdout"] == "hi\n"
        # A still-valid refusal: a fabricated ADMIT for an UNKNOWN
        # tool is refused — an unlisted name is not a tool, whatever
        # a journaled record claims.
        intent = Intent(kind="tool", tool="fs_delete_everything",
                        args={"path": "notes"}, reason="Bogus",
                        parents=["task@1"], source_seqs=[1])
        edit_id = journal.record_self_edit(
            "tool", intent.to_dict(), intent.reason, "ADMIT", {})
        record = executor.execute_journaled(edit_id)
        assert record["status"] == "refused"
        assert "not a registered tool" in record["detail"]
    finally:
        journal.close()


# -- pre-image + compensation -----------------------------------------------------------

def test_pre_image_journaled_and_compensation_restores(mini):
    bundle_dir, work = mini
    target = os.path.join(work, "notes", "plan.md")
    original = open(target).read()
    manifest, journal = _open(bundle_dir)
    try:
        fixture = record_task_fixture(journal, "Overwrite + undo", work)
        ctx = _live_ctx(work, tool_task_seq=fixture["seq"])
        out = _run(journal, bundle_dir, manifest,
                   Intent(kind="tool", tool="fs_write",
                          args={"path": "notes/plan.md",
                                "content": "overwritten\n"},
                          reason="Overwrite the plan",
                          parents=[fixture["ref"]],
                          source_seqs=[fixture["seq"]]), work, ctx=ctx)
        rec = out["execution"]
        assert rec["reversibility"] == "compensable"
        pre = rec["pre_image"]
        assert pre["exists"] is True and pre["stored"] == "inline"
        assert pre["content"] == original
        assert pre["sha256"] == \
            "sha256:" + hashlib.sha256(original.encode()).hexdigest()
        assert open(target).read() == "overwritten\n"

        comp, why = compensation_intent(
            rec, parents=[fixture["ref"], out["edit_id"]],
            source_seqs=[fixture["seq"]])
        assert comp is not None, why
        out2 = _run(journal, bundle_dir, manifest, comp, work, ctx=ctx)
        assert out2["verdict"] == "ADMIT"
        assert out2["execution"]["status"] == "completed"
        assert open(target).read() == original
        # A created file has no executable compensation (no delete tool).
        created = _run(journal, bundle_dir, manifest,
                       Intent(kind="tool", tool="fs_write",
                              args={"path": "notes/new.txt",
                                    "content": "new\n"},
                              reason="Create", parents=[fixture["ref"]],
                              source_seqs=[fixture["seq"]]), work, ctx=ctx)
        comp2, why2 = compensation_intent(
            created["execution"], parents=[fixture["ref"]],
            source_seqs=[fixture["seq"]])
        assert comp2 is None and "no delete tool" in why2
    finally:
        journal.close()


# -- failures are journaled results --------------------------------------------------------

def test_failed_executions_are_journaled_and_change_nothing(mini):
    bundle_dir, work = mini
    target = os.path.join(work, "notes", "plan.md")
    before = open(target).read()
    manifest, journal = _open(bundle_dir)
    try:
        fixture = record_task_fixture(journal, "Failures", work)
        ctx = _live_ctx(work, tool_task_seq=fixture["seq"])
        cases = [
            ("fs_edit", {"path": "notes/plan.md", "old": "absent-string",
                         "new": "x"}, "absent"),
            ("fs_edit", {"path": "notes/plan.md", "old": "step",
                         "new": "x"}, "occurs 2 times"),
            ("fs_write", {"path": "no-such-dir/f.txt", "content": "x"},
             "parent directory"),
            ("fs_read", {"path": "notes"}, "not a file"),
            ("fs_list", {"path": "notes/plan.md"}, "not a directory"),
        ]
        for tool, args, needle in cases:
            out = _run(journal, bundle_dir, manifest,
                       Intent(kind="tool", tool=tool, args=args,
                              reason=f"Failing {tool}",
                              parents=[fixture["ref"]],
                              source_seqs=[fixture["seq"]]), work, ctx=ctx)
            assert out["verdict"] == "ADMIT", tool
            rec = out["execution"]
            assert rec["status"] == "failed", (tool, rec)
            assert needle in rec["detail"], (tool, rec["detail"])
            assert rec["verify"]["result_conformance"] == "not-evaluated"
            edits = journal.backend.find(
                "self_edits", filter={"edit_id": out["edit_id"]})
            assert edits[0]["result"] == "failed"
        assert open(target).read() == before
        assert not os.path.exists(os.path.join(work, "no-such-dir"))
    finally:
        journal.close()


def test_result_bounding_blob_stores_the_overflow(mini):
    bundle_dir, work = mini
    for i in range(30):
        with open(os.path.join(work, f"file-{i:02d}.txt"), "w") as f:
            f.write("data")
    manifest, journal = _open(bundle_dir)
    try:
        fixture = record_task_fixture(journal, "Bounded list", work)
        policy = corpus_policy(work)
        policy["tool_max_result_bytes"] = 400  # tiny cap: forces overflow
        gate = Gate(policy)
        ctx = _live_ctx(work, tool_task_seq=fixture["seq"])
        out = execute_tool_cycle(
            journal, bundle_dir, manifest,
            Intent(kind="tool", tool="fs_list", args={"path": "."},
                   reason="List with a tiny result cap",
                   parents=[fixture["ref"]], source_seqs=[fixture["seq"]]),
            gate, ctx)
        rec = out["execution"]
        assert rec["status"] == "completed"
        assert rec["truncated"] is True
        assert rec["result_blob"] is not None
        assert rec["result_bytes"] <= 400
        assert rec["overflow_result_bytes"] > 400
        blob_path = os.path.join(
            bundle_dir, "blobs", "sha256",
            rec["result_blob"].split(":", 1)[1] + ".safetensors")
        with open(blob_path, "rb") as f:
            full = json.loads(f.read().decode("utf-8"))
        assert len(full["entries"]) == 32  # 30 files + notes/ + data.bin
        assert rec["overflow_result_hash"] == \
            "sha256:" + hashlib.sha256(canonical_json(full)).hexdigest()
    finally:
        journal.close()


# -- run_tool_intent + loop paths -----------------------------------------------------------

def test_run_tool_intent_end_to_end_and_bundle_stays_loadable(mini, tmp_path):
    bundle_dir, work = mini
    # Operator policy supplied for the scripted scenario, as the
    # project's tests supply gate policies; the fixture task entry
    # is journaled first so provenance can chain to it.
    manifest, journal = _open(bundle_dir)
    try:
        fixture = record_task_fixture(journal, "End to end", work)
        from si.bundle import _anchor_checkpoint, _write_manifest
        updated = _anchor_checkpoint(journal, manifest, manifest["head"],
                                     note="task opened")
        _write_manifest(bundle_dir, updated)
        task_seq, task_ref = fixture["seq"], fixture["ref"]
    finally:
        journal.close()
    out = run_tool_intent(
        bundle_dir,
        Intent(kind="tool", tool="fs_write",
               args={"path": "e2e.txt", "content": "via run_tool_intent\n"},
               reason="Write the e2e file", parents=[task_ref],
               source_seqs=[task_seq]),
        work_root=work, task_seq=task_seq, policy=corpus_policy(work))
    assert out["verdict"] == "ADMIT"
    assert out["execution"]["status"] == "completed"
    assert open(os.path.join(work, "e2e.txt")).read() == \
        "via run_tool_intent\n"
    load_bundle(bundle_dir)  # re-anchored: loads cleanly


def test_demo_bundle_run_tool_intent_stays_shadow(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "demo.si"), name="demo-b2",
                             journal_backend="embedded")
    work = str(tmp_path / "work")
    os.makedirs(work)
    policy = corpus_policy(work)
    out = run_tool_intent(
        bundle_dir,
        Intent(kind="tool", tool="fs_write",
               args={"path": "made.txt", "content": "never"},
               reason="Write via demo path", parents=["task@1"],
               source_seqs=[1]),
        work_root=work, policy=policy)
    assert out["verdict"] == "ESCALATE"
    assert out["execution"]["execution"] == TOOL_SHADOW_EXECUTION
    assert not os.path.exists(os.path.join(work, "made.txt"))


def test_miniature_loop_executes_and_demo_loop_stays_shadow(mini, tmp_path):
    from si.loop import run_cycle
    from si.minds import DemoMind

    bundle_dir, work = mini

    class ListMind:
        name = "scripted-b2"

        def perceive(self, journal):
            return []

        def propose(self, perceptions, ctx):
            return Intent(
                kind="tool", tool="fs_write",
                args={"path": "loop-made.txt", "content": "by the loop\n"},
                reason="Write the loop's file",
                parents=["base"], source_seqs=[1])

        def train_delta(self, intent):  # never called for tool intents
            raise AssertionError("tool intents train nothing")

    # The miniature loop path is live in B-2: the file is written,
    # the execution record is journaled, the head does not move.
    res = run_cycle(bundle_dir, ListMind(), gate=_gate(work))
    assert res["verdict"] == "ADMIT"
    assert res["tool_execution"]["status"] == "completed"
    assert open(os.path.join(bundle_dir, "loop-made.txt")).read() == \
        "by the loop\n"
    assert res["head"] == "base" and res["delta_id"] is None

    # The demo loop path keeps B-1 shadow behaviour exactly.
    demo_dir = init_bundle(str(tmp_path / "demo.si"), name="demo-loop",
                           journal_backend="embedded")

    class DemoToolMind(DemoMind):
        def propose(self, perceptions, ctx):
            return Intent(
                kind="tool", tool="fs_write",
                args={"path": "made.txt", "content": "never"},
                reason="Write via the demo loop",
                parents=["base"], source_seqs=[1])

    res = run_cycle(demo_dir, DemoToolMind(bundle_path=demo_dir),
                    gate=_gate(work))
    assert res["verdict"] == "ESCALATE"
    manifest, journal = _open(demo_dir)
    try:
        outcomes = journal.backend.find(
            "experiences", filter={"self_edit_id": res["edit_id"]})
        assert outcomes[0]["record"]["execution"]["execution"] == \
            TOOL_SHADOW_EXECUTION
    finally:
        journal.close()
    assert not os.path.exists(os.path.join(demo_dir, "made.txt"))
