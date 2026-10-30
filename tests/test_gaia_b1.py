"""SI-42 (SI-41 RFC §15 B-1, §16): Gaia tool intents, shadow mode.

Acceptance-mapped tests for the B-1 slice:

1. Intent(kind="tool") round-trips; unknown kinds still REJECT.
2. The registry defines every §8 tool (name set == the RFC table),
   each with args schema, result schema, class, reversibility.
3. The scripted verdict corpus (si.gaia.corpus) produces §5.4's
   verdicts: unknown-tool / schema / blocklisted / missing-reason /
   provenance REJECTs, outside-scope pricing to external,
   class-policy ESCALATE — with itemised schema sub-checks.
4. Checks 3/4 on tool reports carry the §5.3 not-applicable label
   with measured=None (existing gate suites cover byte-for-byte
   unchanged behaviour for the other kinds, unmodified).
5. Policy resolution: absent -> §6.2 defaults; malformed class
   entries -> deny; allowed_tools default admits nothing.
6. Shadow guarantee: across the whole corpus, executions == 0 —
   asserted over the filesystem and a process-spawn sentinel — and
   every report records execution as not-evaluated/shadow.
7. Journal: intents/verdicts/shadow outcomes land via the existing
   journal methods in the existing collections; `si why` walks a
   tool intent's provenance to its task entry.
8. (Suite-wide green is the driver's run; the corpus count is
   asserted in test_corpus_is_complete.)
"""
import os
import subprocess
import sys

import pytest

from si.bundle import DEFAULT_POLICY, init_bundle, load_bundle, load_policy
from si.cli import main as cli_main
from si.gaia.context import ToolContext
from si.gaia.corpus import build_corpus, corpus_policy
from si.gaia.policy import TOOL_POLICY_DEFAULTS, resolve_tool_policy
from si.gaia.registry import (
    REGISTRY,
    RFC_V1_TOOL_NAMES,
    get_tool,
    validate_args,
    validate_result,
)
from si.gaia.shadow import (
    SHADOW_EXECUTION,
    journal_tool_intent,
    record_task_fixture,
    walk_tool_provenance,
)
from si.gate import Gate, Intent
from si.journal import open_journal

WORK_ROOT = "/tmp/si42-corpus-workroot"
OUTSIDE = "/tmp/si42-outside-scope/secret.txt"


def _check(report, name):
    return next(c for c in report.checks if c.name == name)


def _sub(check, name):
    return next(s for s in (check.subchecks or []) if s.name == name)


def _evaluate(row, work_root=WORK_ROOT):
    gate = Gate(corpus_policy(work_root, row["policy_overrides"]))
    return gate.evaluate(row["intent"], None, row["ctx"])


# -- criterion 1: intent format ---------------------------------------------
def test_tool_intent_round_trips_through_to_dict():
    intent = Intent(kind="tool", tool="fs_write",
                    args={"path": "notes/plan.md", "content": "x"},
                    reason="Record the plan",
                    parents=["task@7"], source_seqs=[7])
    d = intent.to_dict()
    assert d["kind"] == "tool" and d["tool"] == "fs_write"
    assert d["args"] == {"path": "notes/plan.md", "content": "x"}
    clone = Intent(**{k: d[k] for k in
                      ("kind", "tool", "args", "reason", "parents",
                       "source_seqs")})
    assert clone.to_dict() == d


def test_unknown_kind_still_rejected_as_malformed():
    gate = Gate(dict(DEFAULT_POLICY))
    report = gate.evaluate(
        Intent(kind="teleport", reason="Go somewhere"), None, None)
    assert report.verdict == "REJECT"
    assert report.checks[0].name == "wellformed"
    assert "unknown intent kind" in report.checks[0].detail
    assert report.execution is None  # not a tool report: form unchanged


def test_tool_kind_cannot_be_redressed_as_delta():
    # A tool proposal smuggled under the delta kind gets the delta
    # checks (and REJECTs: no candidate) — the kind-scoped mapping of
    # §5.3 cannot be reached from another kind.
    gate = Gate(dict(DEFAULT_POLICY))
    report = gate.evaluate(
        Intent(kind="delta", tool="fs_list", args={"path": "x"},
               reason="Sneaky"), None, None)
    assert report.verdict == "REJECT"
    assert all(c.label is None for c in report.checks)


# -- criterion 2: the registry ----------------------------------------------
def test_registry_name_set_equals_rfc_section8_table():
    assert set(REGISTRY) == set(RFC_V1_TOOL_NAMES)
    assert len(REGISTRY) == 12


def test_every_tool_is_fully_specified():
    for name, spec in REGISTRY.items():
        assert spec.name == name and spec.version == 1
        assert spec.description
        assert spec.args_schema.get("type") == "object"
        assert spec.result_schema.get("type") == "object"
        assert spec.consequence_class in ("read", "write", "execute",
                                          "external")
        assert spec.reversibility in ("reversible", "compensable",
                                      "one-way")
        assert spec.default_timeout_seconds >= 1
        assert spec.introduced_by == "SI-41/B-1"
        assert spec.to_dict()["name"] == name


def test_registry_classes_and_reversibility_match_rfc():
    expect = {
        "fs_list": ("read", "reversible"), "fs_read": ("read", "reversible"),
        "fs_search": ("read", "reversible"),
        "journal_query": ("read", "reversible"),
        "bundle_status": ("read", "reversible"),
        "probe_run": ("read", "reversible"),
        "fs_write": ("write", "compensable"),
        "fs_edit": ("write", "compensable"),
        "fs_mkdir": ("write", "compensable"),
        "shell_exec": ("execute", "one-way"),
        "code_exec": ("execute", "one-way"),
        "test_run": ("execute", "one-way"),
    }
    for name, (cls, rev) in expect.items():
        spec = get_tool(name)
        assert (spec.consequence_class, spec.reversibility) == (cls, rev)


def test_registry_schemas_accept_valid_and_reject_invalid_args():
    assert validate_args(get_tool("fs_write"),
                         {"path": "a.md", "content": "hi"}) == []
    violations = validate_args(get_tool("fs_write"), {"path": "a.md"})
    assert any("content" in v and "required" in v for v in violations)
    assert validate_result(get_tool("shell_exec"), {
        "exit_code": 0, "stdout": "", "stderr": "",
        "timed_out": False}) == []
    assert validate_result(get_tool("shell_exec"), {"exit_code": 0})


# -- criterion 5: policy resolution ------------------------------------------
def test_policy_absent_keys_resolve_to_rfc_defaults():
    resolved = resolve_tool_policy({})
    assert resolved == TOOL_POLICY_DEFAULTS
    assert resolved["allowed_tools"] == []
    assert resolved["tool_class_policy"] == {
        "read": "admit", "write": "admit", "execute": "admit",
        "external": "escalate"}
    assert resolved["tool_work_root"] is None
    assert resolved["tool_max_result_bytes"] == 65536
    assert resolved["tool_timeout_seconds"] == 120
    assert resolved["max_tool_cycles_per_task"] == 16
    assert resolved["tool_max_task_result_bytes"] == 1048576


def test_policy_malformed_entries_fail_closed():
    resolved = resolve_tool_policy({
        "allowed_tools": "fs_list",  # not a list -> admits nothing
        "tool_class_policy": {"read": "sometimes", "write": "admit"},
        "tool_max_result_bytes": -5,
        "max_tool_cycles_per_task": "many",
        "tool_work_root": 42,
    })
    assert resolved["allowed_tools"] == []
    assert resolved["tool_class_policy"]["read"] == "deny"  # malformed
    assert resolved["tool_class_policy"]["write"] == "admit"
    # Absent class entries within a present dict take the RFC default.
    assert resolved["tool_class_policy"]["external"] == "escalate"
    assert resolved["tool_max_result_bytes"] == 0
    assert resolved["max_tool_cycles_per_task"] == 0
    assert resolved["tool_work_root"] is None
    wholly = resolve_tool_policy({"tool_class_policy": "admit-all"})
    assert wholly["tool_class_policy"] == {
        c: "deny" for c in ("read", "write", "execute", "external")}


def test_default_policy_ships_the_rfc_tool_keys():
    assert DEFAULT_POLICY["allowed_tools"] == []
    assert DEFAULT_POLICY["tool_class_policy"]["external"] == "escalate"
    assert DEFAULT_POLICY["max_tool_cycles_per_task"] == 16


def test_allowed_tools_default_admits_nothing_at_the_gate():
    # A shipped-default policy allowlists no tool: a perfectly valid
    # fs_list intent REJECTs at the allowed_tools sub-check.
    gate = Gate(dict(DEFAULT_POLICY))
    ctx = ToolContext(tool_work_root=WORK_ROOT, tool_task_seq=7)
    intent = Intent(kind="tool", tool="fs_list", args={"path": "notes"},
                    reason="Look around", parents=["task@7"],
                    source_seqs=[7])
    report = gate.evaluate(intent, None, ctx)
    assert report.verdict == "REJECT"
    assert _sub(_check(report, "safety_bounds"), "allowed_tools").passed is False


# -- criteria 3+4+6: the verdict corpus ---------------------------------------
@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("gaia-workroot"))
    outside_dir = str(tmp_path_factory.mktemp("gaia-outside"))
    outside = os.path.join(outside_dir, "secret.txt")
    rows = build_corpus(root, outside)
    return root, rows


def test_corpus_is_complete(corpus):
    _root, rows = corpus
    names = {r["name"] for r in rows}
    assert len(rows) == 27
    # Every registered tool appears in a valid row.
    valid_tools = {r["intent"].tool for r in rows
                   if r["name"].startswith("valid-")}
    assert valid_tools == set(REGISTRY)
    for required in ("unknown-tool", "schema-violation-fs_write",
                     "blocklisted-command", "missing-reason",
                     "provenance-incomplete", "outside-scope-fs_read",
                     "class-policy-escalate-write",
                     "budget-exhausted-cycles"):
        assert required in names


def test_corpus_verdicts(corpus):
    root, rows = corpus
    for row in rows:
        report = _evaluate(row, work_root=root)
        assert report.verdict == row["expected_verdict"], row["name"]
        # Criterion 6 (report half): every tool report — pass or
        # reject — records execution as not-evaluated/shadow.
        assert report.execution == SHADOW_EXECUTION, row["name"]
        assert report.to_dict()["execution"] == SHADOW_EXECUTION


def test_corpus_effective_classes(corpus):
    root, rows = corpus
    for row in rows:
        gate = Gate(corpus_policy(root, row["policy_overrides"]))
        info = gate.classify_tool_intent(row["intent"], row["ctx"])
        assert info["effective_class"] == row["expected_effective_class"], \
            row["name"]


def test_corpus_check_shape_for_passing_rows(corpus):
    root, rows = corpus
    row = next(r for r in rows if r["name"] == "valid-fs_list")
    report = _evaluate(row, work_root=root)
    assert [c.name for c in report.checks] == [
        "constitution", "safety_bounds", "identity_continuity",
        "capability_floor", "provenance_completeness"]
    # Criterion 4: checks 3/4 carry the not-applicable label.
    for name in ("identity_continuity", "capability_floor"):
        check = _check(report, name)
        assert check.passed is True
        assert check.measured is None
        assert check.label == "not-applicable"
        assert "not-applicable" in check.detail
        assert check.to_dict()["label"] == "not-applicable"
    # The decisive checks have no label (form discipline).
    for name in ("constitution", "safety_bounds", "provenance_completeness"):
        assert _check(report, name).label is None
        assert "label" not in _check(report, name).to_dict()


def test_schema_violation_subchecks_are_itemised(corpus):
    root, rows = corpus
    row = next(r for r in rows if r["name"] == "schema-violation-fs_write")
    report = _evaluate(row, work_root=root)
    assert report.verdict == "REJECT"
    safety = _check(report, "safety_bounds")
    assert safety.passed is False
    schema_sub = _sub(safety, "args_schema")
    assert schema_sub.passed is False
    assert schema_sub.subchecks and len(schema_sub.subchecks) == 3
    details = " ".join(s.detail for s in schema_sub.subchecks)
    assert "content" in details and "path" in details and "extra" in details
    # Journaled form carries the itemisation too.
    journaled = report.to_dict()["checks"][1]
    jschema = next(s for s in journaled["subchecks"]
                   if s["name"] == "args_schema")
    assert len(jschema["subchecks"]) == 3


def test_blocklisted_command_rejects_at_check1(corpus):
    root, rows = corpus
    row = next(r for r in rows if r["name"] == "blocklisted-command")
    report = _evaluate(row, work_root=root)
    assert report.verdict == "REJECT"
    assert _check(report, "constitution").passed is False
    assert "bypass the gate" in _check(report, "constitution").detail


def test_outside_scope_prices_to_external(corpus):
    root, rows = corpus
    for name in ("outside-scope-fs_read", "outside-scope-shell-cwd"):
        row = next(r for r in rows if r["name"] == name)
        gate = Gate(corpus_policy(root, row["policy_overrides"]))
        info = gate.classify_tool_intent(row["intent"], row["ctx"])
        assert info["effective_class"] == "external"
        report = gate.evaluate(row["intent"], None, row["ctx"])
        scope = _sub(_check(report, "safety_bounds"), "execution_scope")
        assert "OUTSIDE" in scope.detail and "external" in scope.detail


def test_class_policy_escalate_is_distinguishable_from_shadow(corpus):
    root, rows = corpus
    row = next(r for r in rows if r["name"] == "class-policy-escalate-write")
    report = _evaluate(row, work_root=root)
    assert report.verdict == "ESCALATE"
    class_sub = _sub(_check(report, "safety_bounds"), "class_policy")
    assert class_sub.passed is True and "'escalate'" in class_sub.detail
    shadow_row = next(r for r in rows if r["name"] == "valid-fs_write")
    shadow_report = _evaluate(shadow_row, work_root=root)
    assert shadow_report.verdict == "ESCALATE"
    shadow_class = _sub(_check(shadow_report, "safety_bounds"),
                        "class_policy")
    assert "'admit'" in shadow_class.detail  # escalation here is shadow's


def test_budget_exhausted_names_the_budget(corpus):
    root, rows = corpus
    for name, key in (("budget-exhausted-cycles", "max_tool_cycles_per_task"),
                      ("budget-exhausted-results",
                       "tool_max_task_result_bytes")):
        row = next(r for r in rows if r["name"] == name)
        report = _evaluate(row, work_root=root)
        assert report.verdict == "REJECT"
        budgets = _sub(_check(report, "safety_bounds"), "budgets")
        assert budgets.passed is False and key in budgets.detail


def test_symlink_out_of_scope_prices_external(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    (root / "link.txt").symlink_to(outside)
    gate = Gate(corpus_policy(str(root)))
    ctx = ToolContext(tool_work_root=str(root), tool_task_seq=7)
    intent = Intent(kind="tool", tool="fs_read",
                    args={"path": "link.txt"}, reason="Read the link",
                    parents=["task@7"], source_seqs=[7])
    info = gate.classify_tool_intent(intent, ctx)
    assert info["effective_class"] == "external"  # resolved target (§9.1)


# -- criterion 6: the shadow guarantee -----------------------------------------
def test_shadow_guarantee_zero_executions_across_corpus(
        corpus, tmp_path, monkeypatch):
    root, rows = corpus
    spawns = []

    def _sentinel(*a, **k):
        spawns.append((a, k))
        raise AssertionError("a process was spawned in shadow mode")

    monkeypatch.setattr(subprocess, "Popen", _sentinel)
    monkeypatch.setattr(subprocess, "run", _sentinel)
    monkeypatch.setattr(os, "system", _sentinel)
    before = set()
    for dirpath, _dirs, files in os.walk(root):
        before |= {os.path.join(dirpath, f) for f in files}
    for row in rows:
        report = _evaluate(row, work_root=root)
        assert report.verdict in ("ESCALATE", "REJECT")
    after = set()
    for dirpath, _dirs, files in os.walk(root):
        after |= {os.path.join(dirpath, f) for f in files}
    assert spawns == [], "shadow corpus spawned a process"
    assert after == before, "shadow corpus changed the work root"
    # The valid fs_write row's target was never created.
    assert not os.path.exists(os.path.join(root, "notes", "plan.md"))


def test_no_tool_intent_ever_admits_in_b1(corpus):
    root, rows = corpus
    for row in rows:
        assert _evaluate(row, work_root=root).verdict != "ADMIT"


# -- criterion 7: journaling + si why -------------------------------------------
@pytest.fixture()
def tool_bundle(tmp_path):
    # An embedded-backend bundle: journal assertions need no NEDB
    # service. Gates in these tests are built from the corpus policy
    # directly; the bundle supplies the real journal on disk.
    return init_bundle(str(tmp_path / "gaia.si"), name="gaia-tests",
                       journal_backend="embedded")


def test_journal_and_why_walk_to_task(tool_bundle, tmp_path, capsys):
    work_root = str(tmp_path / "work")
    os.makedirs(work_root)
    manifest = load_bundle(tool_bundle)
    journal = open_journal(tool_bundle, manifest)
    try:
        fixture = record_task_fixture(
            journal, "List the work root, then report back", work_root,
            budgets={"max_tool_cycles_per_task": 16})
        intent = Intent(
            kind="tool", tool="fs_list", args={"path": "."},
            reason="List the work root as the task's first step",
            parents=[fixture["ref"]], source_seqs=[fixture["seq"]])
        ctx = ToolContext(tool_work_root=work_root,
                          tool_task_seq=fixture["seq"])
        gate = Gate(corpus_policy(work_root))
        outcome = journal_tool_intent(journal, gate, intent, ctx)
        assert outcome["verdict"] == "ESCALATE"
        assert outcome["execution"]["execution"] == SHADOW_EXECUTION
        assert outcome["effective_class"] == "read"

        edits = journal.backend.find(
            "self_edits", filter={"edit_id": outcome["edit_id"]})
        assert edits and edits[0]["kind"] == "tool"
        assert edits[0]["gate_verdict"] == "ESCALATE"
        assert edits[0]["result"] == "escalated"
        assert edits[0]["gate_report"]["execution"] == SHADOW_EXECUTION
        outcomes = journal.backend.find(
            "experiences", filter={"self_edit_id": outcome["edit_id"]})
        assert outcomes and outcomes[0]["kind"] == "tool_result"
        assert outcomes[0]["record"]["execution"]["status"] == \
            SHADOW_EXECUTION
        prov = journal.backend.find(
            "provenance", filter={"child": outcome["edit_id"]})
        assert prov, "the tool intent's provenance edges are journaled"

        walk = walk_tool_provenance(journal, outcome["edit_id"])
        assert walk["task"] is not None
        task_record = walk["task"]["row"]["record"]
        assert task_record["instruction"] == \
            "List the work root, then report back"
        assert walk["task"]["row"]["seq"] == fixture["seq"]
        edit_id = outcome["edit_id"]
        # Anchor the journal tip in the manifest, as the loop does
        # after every cycle, so the bundle loads cleanly for the CLI.
        from si.bundle import _anchor_checkpoint, _write_manifest
        updated = _anchor_checkpoint(journal, manifest, manifest["head"],
                                     note=f"escalated {edit_id}")
        _write_manifest(tool_bundle, updated)
    finally:
        journal.close()

    # The CLI walk: `si why <edit-id>` reaches the same task entry.
    rc = cli_main(["why", tool_bundle, edit_id])
    out = capsys.readouterr().out
    assert rc == 0
    assert "List the work root, then report back" in out
    assert SHADOW_EXECUTION in out
    assert "task:" in out


def test_why_rejects_unknown_edit_id(tool_bundle, capsys):
    rc = cli_main(["why", tool_bundle, "se_9999"])
    assert rc == 2


def test_loop_journals_tool_shadow_outcome(tool_bundle, tmp_path):
    # A mind proposing a tool intent through the real loop: the cycle
    # ESCALATEs (shadow), journals the self-edit AND the shadow
    # outcome, and creates nothing in the work root.
    from si.loop import run_cycle
    from si.minds import DemoMind

    work_root = str(tmp_path / "loopwork")
    os.makedirs(work_root)

    class ToolMind(DemoMind):
        def propose(self, perceptions, ctx):
            return Intent(kind="tool", tool="fs_write",
                          args={"path": "made-by-loop.txt",
                                "content": "must never exist"},
                          reason="Write the loop test file",
                          parents=["base"], source_seqs=[1])

    # The loop builds its gate from the bundle policy (defaults:
    # allowed_tools empty) — the intent REJECTs at allowlist, which
    # still exercises the loop's tool journaling on the REJECT path.
    res = run_cycle(tool_bundle, ToolMind(bundle_path=tool_bundle))
    assert res["verdict"] == "REJECT"
    assert not os.path.exists(os.path.join(work_root, "made-by-loop.txt"))
    manifest = load_bundle(tool_bundle)
    journal = open_journal(tool_bundle, manifest)
    try:
        outcomes = journal.backend.find(
            "experiences", filter={"self_edit_id": res["edit_id"]})
        assert outcomes and outcomes[0]["kind"] == "tool_result"
        assert outcomes[0]["record"]["execution"]["execution"] == \
            SHADOW_EXECUTION
    finally:
        journal.close()


# -- criterion 8 helper: shipped policy on disk carries the keys ---------------
def test_fresh_bundle_policy_file_has_tool_keys(tool_bundle):
    policy = load_policy(tool_bundle)
    for key in TOOL_POLICY_DEFAULTS:
        assert key in policy


def test_cli_parser_accepts_why_with_an_edit_id():
    from si.cli import build_parser
    args = build_parser().parse_args(["why", "bundle.si", "se_0003"])
    assert args.delta_id == "se_0003"


def test_python_compiles_all_gaia_modules():
    import py_compile
    import si.gaia.corpus  # noqa: F401
    import si.gaia.shadow  # noqa: F401
    for mod in ("registry", "policy", "context", "corpus", "shadow"):
        py_compile.compile(
            os.path.join(os.path.dirname(__file__), "..", "si", "gaia",
                         f"{mod}.py"), doraise=True)
    assert sys.version_info >= (3, 10)
