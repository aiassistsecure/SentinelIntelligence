"""SI-55 / S-2b: the hearth.* tools are registered — seam-crossing,
external-class, NOT locally executable.

Pins:
- exactly the two in-repo-evidenced names are registered
  (hearth.restart from the S-2a watcher fixture; hearth.notify
  named in docs/S-2a-proposal-ingest.md §3), with schemas that
  accept the evidenced watcher args and reject undeclared ones;
- neither name is in any local LIVE set (no executor may run
  them — the B-3 executor refuses tools outside LIVE_TOOLS_B3);
- a watcher proposal now gates ON ITS MERITS: tool_registered
  passes; under external:admit it ADMITs and the self-edit rests
  pending with its gate report journaled, submitting/executing
  NOTHING locally; under the default external:escalate it
  ESCALATEs (still not a tool_registered REJECT);
- a still-unregistered hearth.* name REJECTs at tool_registered;
- the registry-gap label no longer claims the names are
  unregistered, and still states S-3 is UNBUILT.
"""
import copy

import pytest

from si.bundle import init_bundle, load_bundle
from si.gaia.context import ToolContext
from si.gaia.corpus import corpus_policy
from si.gaia.registry import (
    HEARTH_SEAM_TOOL_NAMES,
    LIVE_TOOLS_B2,
    LIVE_TOOLS_B3,
    REGISTRY,
    SEAM_CROSSING_TOOLS,
    get_tool,
    validate_args,
    validate_result,
)
from si.gaia.seam import REGISTRY_GAP_LABEL, REGISTRY_STATUS_LABEL
from si.gaia.service import submit_proposal
from si.gate import (
    TOOL_SEAM_PENDING_EXECUTION,
    Gate,
    Intent,
)
from si.journal import open_journal
from si.tiny_seed import seed_tiny_bundle

WATCHER = {
    "model": "muse",
    "tool": "hearth.restart",
    "args": {"model": "muse",
             "spine_event": {"hash": "hash-s2b-fixture", "seq": 41}},
    "reason": "observed loss at seq 41",
    "caused_by": {"hash": "hash-s2b-fixture", "seq": 41},
}

ADMIT_CLASS_POLICY = {"read": "admit", "write": "admit",
                      "execute": "admit", "external": "admit"}


@pytest.fixture()
def mini(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "mini.si"),
                             name="gaia-s2b",
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    work = tmp_path / "work"
    work.mkdir()
    return bundle_dir, str(work)


def _open(bundle_dir):
    return open_journal(bundle_dir, load_bundle(bundle_dir))


def _sub(report, check_name, sub_name):
    check = next(c for c in report["checks"] if c["name"] == check_name)
    return next(s for s in check["subchecks"] if s["name"] == sub_name)


def _gate_report(policy, tool, args, live=True):
    gate = Gate(policy)
    ctx = ToolContext(tool_work_root="/tmp", tool_task_seq=7,
                      tool_live_execution=live)
    intent = Intent(kind="tool", tool=tool, args=args,
                    reason="watcher observed a lost model",
                    parents=["task@7"], source_seqs=[7])
    return gate.evaluate(intent, None, ctx)


# -- registration -------------------------------------------------------

def test_exactly_the_evidenced_hearth_names_are_registered():
    assert HEARTH_SEAM_TOOL_NAMES == frozenset(
        {"hearth.restart", "hearth.notify"})
    assert SEAM_CROSSING_TOOLS == HEARTH_SEAM_TOOL_NAMES
    assert {n for n in REGISTRY if n.startswith("hearth.")} == \
        set(HEARTH_SEAM_TOOL_NAMES)
    for name in HEARTH_SEAM_TOOL_NAMES:
        spec = get_tool(name)
        assert spec is not None
        assert spec.consequence_class == "external"
        assert spec.introduced_by == "SI-55/S-2b"
        assert spec.args_schema["type"] == "object"
        assert spec.result_schema["type"] == "object"


def test_hearth_tools_are_not_locally_executable():
    for name in HEARTH_SEAM_TOOL_NAMES:
        assert name not in LIVE_TOOLS_B2
        assert name not in LIVE_TOOLS_B3


def test_restart_schema_accepts_the_watcher_fixture_args():
    spec = get_tool("hearth.restart")
    assert validate_args(spec, copy.deepcopy(WATCHER["args"])) == []
    assert validate_args(spec, {"model": "muse"}) == []
    assert validate_args(spec, {"spine_event": {"hash": "h", "seq": 1}})
    assert validate_args(spec, {"model": "muse", "surprise": 1})
    bad_seq = {"model": "muse",
               "spine_event": {"hash": "h", "seq": -1}}
    assert validate_args(spec, bad_seq)


def test_notify_schema_requires_only_model():
    spec = get_tool("hearth.notify")
    assert spec.reversibility == "one-way"
    assert validate_args(spec, {"model": "muse"}) == []
    assert validate_args(
        spec, {"model": "muse", "message": "muse was lost"}) == []
    assert validate_args(spec, {"message": "no model"})
    assert get_tool("hearth.restart").reversibility == "compensable"


def test_seam_pending_result_schema_is_the_pending_shape():
    for name in HEARTH_SEAM_TOOL_NAMES:
        spec = get_tool(name)
        assert validate_result(
            spec, {"status": "pending", "model": "muse"}) == []
        assert validate_result(
            spec, {"status": "restarted", "model": "muse"})


def test_gap_label_no_longer_claims_unregistered():
    assert REGISTRY_STATUS_LABEL == REGISTRY_GAP_LABEL
    assert "CLOSED" in REGISTRY_GAP_LABEL
    assert "hearth.restart" in REGISTRY_GAP_LABEL
    assert "NOT registered" not in REGISTRY_GAP_LABEL
    assert "UNBUILT" in REGISTRY_GAP_LABEL  # S-3 honesty retained


# -- gate merits ----------------------------------------------------------

def test_gate_admits_restart_under_external_admit_and_labels_pending(tmp_path):
    policy = corpus_policy(str(tmp_path),
                           {"tool_class_policy": ADMIT_CLASS_POLICY})
    report = _gate_report(policy, "hearth.restart",
                          copy.deepcopy(WATCHER["args"]))
    assert report.verdict == "ADMIT"
    assert report.execution == TOOL_SEAM_PENDING_EXECUTION
    assert "UNBUILT" in report.execution
    d = report.to_dict()
    assert _sub(d, "safety_bounds", "tool_registered")["passed"] is True
    # Per-model seam serialisation is priced at gate time.
    assert d["queue"]["ordering_key"] == "hearth:muse"
    assert d["queue"]["effective_class"] == "external"


def test_gate_escalates_restart_under_default_external_policy(tmp_path):
    policy = corpus_policy(str(tmp_path))  # external: escalate
    report = _gate_report(policy, "hearth.restart",
                          copy.deepcopy(WATCHER["args"]))
    assert report.verdict == "ESCALATE"
    d = report.to_dict()
    assert _sub(d, "safety_bounds", "tool_registered")["passed"] is True


def test_gate_nonlive_context_still_shadow_escalates(tmp_path):
    policy = corpus_policy(str(tmp_path),
                           {"tool_class_policy": ADMIT_CLASS_POLICY})
    report = _gate_report(policy, "hearth.notify", {"model": "muse"},
                          live=False)
    assert report.verdict == "ESCALATE"
    assert report.execution == "not-evaluated — shadow mode"


def test_gate_rejects_unregistered_hearth_name_at_tool_registered(tmp_path):
    policy = corpus_policy(str(tmp_path),
                           {"tool_class_policy": ADMIT_CLASS_POLICY})
    report = _gate_report(policy, "hearth.reap", {"model": "muse"})
    assert report.verdict == "REJECT"
    d = report.to_dict()
    assert _sub(d, "safety_bounds", "tool_registered")["passed"] is False


def test_gate_rejects_restart_with_undeclared_arg(tmp_path):
    policy = corpus_policy(str(tmp_path),
                           {"tool_class_policy": ADMIT_CLASS_POLICY})
    report = _gate_report(policy, "hearth.restart",
                          {"model": "muse", "force": True})
    assert report.verdict == "REJECT"
    d = report.to_dict()
    assert _sub(d, "safety_bounds", "tool_registered")["passed"] is True
    assert _sub(d, "safety_bounds", "args_schema")["passed"] is False


# -- seam ingest end-to-end ----------------------------------------------

def test_seam_watcher_admits_to_pending_and_executes_nothing(mini):
    bundle_dir, work = mini
    policy = corpus_policy(work,
                           {"tool_class_policy": ADMIT_CLASS_POLICY})
    out = submit_proposal(bundle_dir, copy.deepcopy(WATCHER),
                          policy=policy)
    assert out["status"] == "acked"
    assert out["verdict"] == "ADMIT"
    assert out["deduped"] is False
    journal = _open(bundle_dir)
    try:
        edits = [r for r in journal.backend.find("self_edits")
                 if (r.get("intent") or {}).get("caused_by")
                 == WATCHER["caused_by"]]
        assert len(edits) == 1
        row = edits[0]
        assert row["seq"] == out["ack"]["journal_seq"]
        assert row["gate_verdict"] == "ADMIT"
        # The ADMIT rests pending: journaled, not executed,
        # not submitted (no gaia_work kind exists — S-3 unbuilt).
        assert row["result"] == "pending"
        assert row["gate_report"]["execution"] == \
            TOOL_SEAM_PENDING_EXECUTION
        experiences = journal.backend.find("experiences")
        assert not [r for r in experiences
                    if r.get("kind") == "gaia_work"]
        assert not [r for r in experiences
                    if r.get("kind") == "tool_result"]
        assert journal.verify_chain() is True
    finally:
        journal.close()


def test_seam_watcher_escalates_under_corpus_policy(mini):
    bundle_dir, work = mini
    out = submit_proposal(bundle_dir, copy.deepcopy(WATCHER),
                          policy=corpus_policy(work))
    assert out["status"] == "acked"
    assert out["verdict"] == "ESCALATE"
    d = out["gate_report"]
    assert _sub(d, "safety_bounds", "tool_registered")["passed"] is True


def test_seam_notify_admits_to_pending(mini):
    bundle_dir, work = mini
    policy = corpus_policy(work,
                           {"tool_class_policy": ADMIT_CLASS_POLICY})
    payload = {
        "model": "muse", "tool": "hearth.notify",
        "args": {"model": "muse", "message": "muse was lost"},
        "reason": "observed loss",
        "caused_by": {"hash": "hash-notify", "seq": 9},
    }
    out = submit_proposal(bundle_dir, payload, policy=policy)
    assert out["verdict"] == "ADMIT"
    journal = _open(bundle_dir)
    try:
        edits = [r for r in journal.backend.find("self_edits")
                 if (r.get("intent") or {}).get("caused_by")
                 == {"hash": "hash-notify", "seq": 9}]
        assert len(edits) == 1 and edits[0]["result"] == "pending"
    finally:
        journal.close()


def test_seam_unregistered_name_still_rejects_at_tool_registered(mini):
    bundle_dir, work = mini
    payload = copy.deepcopy(WATCHER)
    payload["tool"] = "hearth.reap"
    out = submit_proposal(bundle_dir, payload,
                          policy=corpus_policy(work))
    assert out["status"] == "acked"
    assert out["verdict"] == "REJECT"
    assert _sub(out["gate_report"], "safety_bounds",
                "tool_registered")["passed"] is False
