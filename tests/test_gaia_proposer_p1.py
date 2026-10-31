"""SI-64 (SI-63 RFC §19 P-1): Gaia PROPOSE — perception
assembly + catalogue rendering, no gateway call.

P-1 builds, in si/gaia/proposer.py: perception assembly (§6)
and catalogue rendering (§7) as pure, journal-derived
functions; digests (§6.4, §7.4); bounds and truncation labels
(§6.3). The proposer's ``propose`` exists but returns the
labelled not-evaluated outcome — P-1 makes NO gateway call, by
construction, and these tests assert the transport is never
invoked.

Honesty discipline asserted throughout: the thinking proposer
remains UNBUILT beyond P-1 assembly/rendering, the real Hearth
path remains UNMEASURED, and the proposer is unused by default
— no existing behaviour changes (§15.2).
"""
import copy

import pytest

import si.gaia.registry as registry_module
from si.gaia.proposer import (
    CATALOGUE_RECORD,
    DEFAULT_MAX_PERCEPTION_BYTES,
    HEARTH_PATH_UNMEASURED_LABEL,
    NOT_EVALUATED,
    PERCEPTION_RECORD,
    PROPOSER_NOT_EVALUATED,
    SEAM_CROSSING_LABEL,
    TRUNCATION_LABEL,
    TRUNCATION_ORDER,
    HearthProposer,
    assemble_perception,
    build_attempt_record,
    render_catalogue,
    render_catalogue_text,
    verify_catalogue_digest,
    verify_perception_digest,
)
from si.gaia.registry import REGISTRY, ToolSpec
from si.gaia.task import record_task_entry
from si.journal import EmbeddedBackend, Journal


@pytest.fixture()
def journal(tmp_path):
    j = Journal(EmbeddedBackend(str(tmp_path / "journal")))
    yield j
    j.close()


def _task(journal, instruction="inspect the work root"):
    return record_task_entry(
        journal, instruction, "/tmp/p1-work",
        budgets={"max_tool_cycles_per_task": 16,
                 "tool_max_task_result_bytes": 1048576})


def _add_intent(journal, task, verdict="ADMIT", tool="fs_read",
                args=None, gate_report=None, reason="p1 test intent"):
    intent = {
        "kind": "tool",
        "tool": tool,
        "args": args if args is not None else {"path": "a.txt"},
        "parents": [task["ref"]],
        "source_seqs": [task["seq"]],
        "reason": reason,
    }
    report = gate_report if gate_report is not None else {
        "verdict": verdict, "checks": []}
    edit_id = journal.record_self_edit("tool", intent, reason,
                                       verdict, report)
    rows = journal.backend.find("self_edits",
                                filter={"edit_id": edit_id}, limit=1)
    return edit_id, rows[0]["seq"]


def _add_outcome(journal, edit_id, tool="fs_read", payload="x" * 64,
                 result_bytes=100):
    execution = {"tool": tool, "intent_id": edit_id, "class": "read",
                 "status": "completed",
                 "result": {"content": payload},
                 "result_bytes": result_bytes}
    journal.record_experience("tool_result", {
        "kind": "tool_result", "source": f"tool:{tool}",
        "intent_id": edit_id, "tool": tool, "gate_verdict": "ADMIT",
        "effective_class": "read", "execution": execution,
    }, edit_id)
    rows = journal.backend.find(
        "experiences", filter={"self_edit_id": edit_id},
        sort="-seq", limit=1)
    return rows[0]["seq"]


def _add_residency(journal, state, polled_at):
    record = {"kind": "hearth_residency", "source": "hearth:/residency",
              "state": state, "raw_state": state, "model": "miniature",
              "node": "node-a", "polled_at": polled_at}
    journal.record_experience("hearth_residency", record, "")


def _fixture(journal):
    """A task with one completed read, one rejected proposal
    (with the gate's stated reason), and a residency walk."""
    task = _task(journal)
    edit1, _seq1 = _add_intent(journal, task)
    outcome_seq = _add_outcome(journal, edit1)
    reject_report = {"verdict": "REJECT", "checks": [
        {"name": "tool-policy", "passed": False,
         "detail": "tool 'shell_exec' is not in allowed_tools",
         "measured": None}]}
    edit2, _seq2 = _add_intent(
        journal, task, verdict="REJECT", tool="shell_exec",
        args={"command": "ls"}, gate_report=reject_report,
        reason="list the work root")
    _add_residency(journal, "Resident", "2026-10-31T00:00:00Z")
    _add_residency(journal, "Lost{Evicted}", "2026-10-31T00:01:00Z")
    _add_residency(journal, "Resident", "2026-10-31T00:02:00Z")
    return task, (edit1, edit2), outcome_seq


# -- catalogue rendering (§7) ---------------------------------------------------

def test_catalogue_renders_every_registry_tool_verbatim():
    cat = render_catalogue()
    assert cat["record"] == CATALOGUE_RECORD
    assert cat["tool_count"] == len(REGISTRY)
    assert [t["name"] for t in cat["tools"]] == list(REGISTRY)
    for entry in cat["tools"]:
        spec = REGISTRY[entry["name"]]
        assert entry["version"] == spec.version
        assert entry["description"] == spec.description
        # The args schema the gate validates against, complete.
        assert entry["args_schema"] == spec.args_schema
        assert entry["consequence_class"] == spec.consequence_class
        assert entry["reversibility"] == spec.reversibility
        assert entry["default_timeout_seconds"] == \
            spec.default_timeout_seconds
        summary = entry["result_schema_summary"]
        assert summary["required"] == list(
            spec.result_schema.get("required") or [])
    assert verify_catalogue_digest(cat)


def test_catalogue_digest_detects_tampering():
    cat = render_catalogue()
    cat["tools"][0]["description"] = "rewritten by hand"
    assert not verify_catalogue_digest(cat)


def test_registry_mutation_changes_rendering_and_digest():
    before = render_catalogue()
    fake = ToolSpec(
        name="zz_fake_tool", version=1,
        description="a test-only tool that exists only here",
        args_schema={"type": "object", "required": [],
                     "properties": {}},
        result_schema={"type": "object", "required": [],
                       "properties": {}},
        consequence_class="read", reversibility="reversible",
        default_timeout_seconds=5, introduced_by="SI-64-test")
    REGISTRY["zz_fake_tool"] = fake
    try:
        mutated = render_catalogue()
    finally:
        del REGISTRY["zz_fake_tool"]
    assert mutated["tool_count"] == before["tool_count"] + 1
    assert mutated["tools"][-1]["name"] == "zz_fake_tool"
    assert mutated["digest"] != before["digest"]
    # And the rendering returns to the unmutated digest — proof
    # the catalogue is rendered at call time, not cached.
    assert render_catalogue()["digest"] == before["digest"]
    assert registry_module.REGISTRY is REGISTRY


def test_seam_crossing_tools_render_with_seam_labels():
    cat = render_catalogue()
    by_name = {t["name"]: t for t in cat["tools"]}
    for name in ("hearth.restart", "hearth.notify"):
        entry = by_name[name]
        assert entry["seam_crossing"] is True
        assert entry["seam_label"] == SEAM_CROSSING_LABEL
        assert "does not execute locally" in entry["seam_label"]
        assert "across the seam" in entry["seam_label"]
        assert entry["consequence_class"] == "external"
    for name in ("fs_read", "shell_exec"):
        assert by_name[name]["seam_crossing"] is False
        assert "seam_label" not in by_name[name]
    text = render_catalogue_text(cat)
    assert SEAM_CROSSING_LABEL in text


def test_catalogue_filtering_labels_instead_of_deleting():
    cat = render_catalogue(live_tools={"fs_read"},
                           disabled_tools={"fs_write"})
    by_name = {t["name"]: t for t in cat["tools"]}
    assert cat["tool_count"] == len(REGISTRY)  # nothing deleted
    assert by_name["fs_read"]["availability"] == "rendered"
    assert by_name["fs_write"]["availability"] == "disabled"
    assert "disabled by policy" in by_name["fs_write"][
        "availability_label"]
    assert by_name["shell_exec"]["availability"] == "shadow"
    assert "shadow / escalate-only" in by_name["shell_exec"][
        "availability_label"]


def test_catalogue_text_is_byte_stable():
    assert render_catalogue_text(render_catalogue()) == \
        render_catalogue_text(render_catalogue())


# -- perception assembly (§6) ---------------------------------------------------

def test_assembly_is_byte_stable_and_digest_verifies_on_refold(journal):
    task, _edits, _outcome = _fixture(journal)
    a = assemble_perception(journal, task["seq"])
    b = assemble_perception(journal, task["seq"])
    assert a == b  # same journal fold, byte-identical (§6.4)
    assert a["record"] == PERCEPTION_RECORD
    assert a["digest"].startswith("sha256:")
    assert verify_perception_digest(a)
    tampered = copy.deepcopy(a)
    tampered["sections"]["task"]["instruction"] = "something else"
    assert not verify_perception_digest(tampered)


def test_assembly_sections_follow_s6_1_sources(journal):
    task, (edit1, edit2), outcome_seq = _fixture(journal)
    p = assemble_perception(journal, task["seq"])
    s = p["sections"]
    # 1. task entry, verbatim.
    assert s["task"]["instruction"] == "inspect the work root"
    assert s["task"]["work_root"] == "/tmp/p1-work"
    assert s["task"]["budgets"]["max_tool_cycles_per_task"] == 16
    # 2. catalogue, rendered — its digest is the perception's.
    assert s["catalogue"]["digest"] == p["catalogue_digest"]
    assert verify_catalogue_digest(s["catalogue"])
    # 3. prior cycles with verdicts; the rejection carries the
    # gate's stated reason.
    cycles = {c["edit_id"]: c for c in s["prior_cycles"]}
    assert cycles[edit1]["verdict"] == "ADMIT"
    assert cycles[edit1]["outcome_seq"] == outcome_seq
    assert cycles[edit1]["outcome_status"] == "completed"
    assert cycles[edit1]["args"] == {"path": "a.txt"}
    assert cycles[edit1]["args_digest"].startswith("sha256:")
    assert cycles[edit2]["verdict"] == "REJECT"
    assert cycles[edit2]["gate_reasons"] == [{
        "check": "tool-policy",
        "detail": "tool 'shell_exec' is not in allowed_tools"}]
    # 4. tool results as tagged data.
    assert len(s["tool_results"]) == 1
    result = s["tool_results"][0]
    assert result["source"] == "tool:fs_read"
    assert result["provenance"]["task_seq"] == task["seq"]
    assert "never instructions" in result["data_not_instructions"]
    # 5. residency: latest + the transition walk.
    assert s["residency"]["latest"]["state"] == "Resident"
    assert [(t["from_state"], t["to_state"])
            for t in s["residency"]["transitions"]] == [
                ("Resident", "Lost{Evicted}"),
                ("Lost{Evicted}", "Resident")]
    # 6. queue state in the fold's vocabulary.
    assert s["queue_state"]["task_seq"] == task["seq"]
    assert s["queue_state"]["counts"].get("executed") == 1
    # 7. budgets remaining, as numbers.
    b = s["budgets_remaining"]
    assert b["cycles_used"] == 2
    assert b["max_tool_cycles_per_task_remaining"] == 14
    assert b["result_bytes_used"] == 100
    assert b["tool_max_task_result_bytes_remaining"] == 1048576 - 100
    assert p["truncated"] is False and p["truncations"] == []


def test_assembly_excludes_other_tasks(journal):
    task, _edits, _outcome = _fixture(journal)
    other = _task(journal, instruction="a different task entirely")
    other_edit, _ = _add_intent(journal, other, tool="fs_write",
                                args={"path": "b.txt", "content": "x"})
    _add_outcome(journal, other_edit, tool="fs_write")
    p = assemble_perception(journal, task["seq"])
    tools = [c["tool"] for c in p["sections"]["prior_cycles"]]
    assert "fs_write" not in tools
    assert len(p["sections"]["tool_results"]) == 1


def test_assembly_unknown_task_fails_closed(journal):
    with pytest.raises(ValueError, match="fails closed"):
        assemble_perception(journal, 999999)


def test_large_prior_args_ride_by_digest_only(journal):
    task = _task(journal)
    _add_intent(journal, task, tool="fs_write",
                args={"path": "big.txt", "content": "y" * 5000})
    p = assemble_perception(journal, task["seq"])
    cycle = p["sections"]["prior_cycles"][0]
    assert cycle["args"] is None
    assert "digest only" in cycle["args_label"]
    assert cycle["args_digest"].startswith("sha256:")


# -- bounds and truncation (§6.3) -------------------------------------------------

def test_truncation_order_is_oldest_tool_results_first(journal):
    task = _task(journal)
    outcome_seqs = []
    for i in range(3):
        edit, _ = _add_intent(journal, task,
                              args={"path": f"f{i}.txt"})
        outcome_seqs.append(
            _add_outcome(journal, edit, payload="z" * 2000))
    _add_residency(journal, "Resident", "2026-10-31T00:00:00Z")
    _add_residency(journal, "Loading", "2026-10-31T00:01:00Z")
    p = assemble_perception(journal, task["seq"],
                            max_perception_bytes=1)
    sections_seen = [t["section"] for t in p["truncations"]]
    # §6.3's order, as recorded: tool results (oldest first),
    # then residency history, then prior cycles, then the
    # non-truncatable-remainder label.
    order_idx = [TRUNCATION_ORDER.index(s) if s in TRUNCATION_ORDER
                 else len(TRUNCATION_ORDER) for s in sections_seen]
    assert order_idx == sorted(order_idx)
    result_truncs = [t for t in p["truncations"]
                     if t["section"] == "tool_results"]
    assert [t["seq"] for t in result_truncs] == outcome_seqs
    assert all(t["label"] == TRUNCATION_LABEL
               for t in p["truncations"][:-1])
    assert sections_seen[-1] == "bound"
    # Every truncated result is labelled in the perception
    # itself, tags and seqs intact.
    for entry in p["sections"]["tool_results"]:
        assert entry["truncated"] is True
        assert entry["execution"]["truncated"] is True
        assert entry["provenance"]["task_seq"] == task["seq"]
    # The task entry and the catalogue never truncate (§6.3).
    assert p["sections"]["task"]["instruction"] == \
        "inspect the work root"
    assert verify_catalogue_digest(p["sections"]["catalogue"])
    assert p["truncated"] is True


def test_truncation_labels_reach_the_would_be_attempt_record(journal):
    task = _task(journal)
    edit, _ = _add_intent(journal, task)
    _add_outcome(journal, edit, payload="z" * 4000)
    full = assemble_perception(journal, task["seq"])
    bounded = assemble_perception(
        journal, task["seq"],
        max_perception_bytes=full["bytes"] - 500)
    assert bounded["truncations"], "the tighter bound must truncate"
    assert bounded["truncations"][0]["section"] == "tool_results"
    attempt = build_attempt_record(bounded)
    assert attempt["perception"]["truncations"] == \
        bounded["truncations"]
    assert attempt["perception"]["digest"] == bounded["digest"]
    assert attempt["perception"]["catalogue_digest"] == \
        bounded["catalogue_digest"]


# -- the would-be attempt record (§14.2) ------------------------------------------

def test_attempt_record_shape_and_labelled_absences(journal):
    task, _edits, _outcome = _fixture(journal)
    p = assemble_perception(journal, task["seq"])
    attempt = build_attempt_record(p, cycle=3)
    assert attempt["kind"] == "gaia_proposal_attempt"
    assert attempt["task_seq"] == task["seq"]
    assert attempt["cycle"] == 3
    assert attempt["proposer"]["module"] == "si.gaia.proposer"
    assert attempt["perception"]["digest"] == p["digest"]
    assert attempt["perception"]["bytes"] == p["bytes"]
    assert attempt["residency"]["state"] == "Resident"
    assert attempt["residency"]["observation_seq"] is not None
    # No model identity, no declaration, no request: each is a
    # labelled absence, never an invented value (§14.2, I7).
    assert attempt["model"]["served_id"] is None
    assert "absent" in attempt["model"]["served_id_label"]
    assert attempt["model"]["declaration_digest"] is None
    assert "absent" in attempt["model"]["declaration_label"]
    assert attempt["request_digest"] is None
    assert "no call" in attempt["request_digest_label"]
    assert attempt["preconditions"]["outcome"] == NOT_EVALUATED


# -- the proposer skeleton: no gateway call, by construction ----------------------

class _CountingTransport:
    """A transport that records calls and raises if used: P-1
    must never invoke it (§19 P-1's zero-call acceptance)."""
    label = "counting-test-transport"
    is_replay = True

    def __init__(self):
        self.calls = []

    def __call__(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url})
        raise AssertionError("P-1 must make no gateway call")


def test_propose_returns_not_evaluated_and_never_invokes_transport(journal):
    task, _edits, _outcome = _fixture(journal)
    transport = _CountingTransport()
    proposer = HearthProposer(transport=transport, task_seq=task["seq"])
    perceptions = proposer.perceive(journal)
    assert len(perceptions) == 1
    assert perceptions[0]["record"] == PERCEPTION_RECORD
    outcome = proposer.propose(perceptions, ctx=None)
    assert transport.calls == []  # zero calls, by construction
    assert proposer.transport_called is False
    assert outcome["outcome"] == NOT_EVALUATED
    assert outcome["label"] == PROPOSER_NOT_EVALUATED
    assert outcome["transport_called"] is False
    assert outcome["intents"] == []
    assert "UNMEASURED" in outcome["hearth_path_performance"]
    assert outcome["perception_digest"] == perceptions[0]["digest"]
    assert outcome["catalogue_digest"] == \
        perceptions[0]["catalogue_digest"]
    assert outcome["attempt_record"]["preconditions"]["outcome"] == \
        NOT_EVALUATED
    # A second call is the same honest outcome, still zero calls.
    assert proposer.propose(perceptions)["outcome"] == NOT_EVALUATED
    assert transport.calls == []


def test_propose_without_perception_still_makes_no_call():
    transport = _CountingTransport()
    proposer = HearthProposer(transport=transport)
    outcome = proposer.propose([], ctx=None)
    assert outcome["outcome"] == NOT_EVALUATED
    assert transport.calls == []
    assert outcome["catalogue_digest"].startswith("sha256:")


def test_perceive_without_task_is_an_empty_contribution(journal):
    # §5.1: with no task configured the proposer adds no
    # perception beyond the loop's own — an empty contribution.
    assert HearthProposer().perceive(journal) == []
    assert HearthProposer().perceive(None) == []


def test_default_bound_is_the_pinned_constant(journal):
    task, _edits, _outcome = _fixture(journal)
    p = assemble_perception(journal, task["seq"])
    assert p["max_perception_bytes"] == DEFAULT_MAX_PERCEPTION_BYTES


# -- no existing behaviour changes (§15.2) -----------------------------------------

def test_proposer_is_unused_by_default():
    import si.gaia.task as task_module
    src = open(task_module.__file__).read()
    assert "proposer" not in src.split('"""')[2].split("\nclass ")[0] \
        or "HearthProposer" not in src
    assert "HearthProposer" not in src
    assert "import" in src and "si.gaia.proposer" not in src


def test_hearth_path_remains_unmeasured_label_present():
    assert HEARTH_PATH_UNMEASURED_LABEL.startswith("UNMEASURED")
    assert "UNBUILT beyond P-1" in PROPOSER_NOT_EVALUATED
