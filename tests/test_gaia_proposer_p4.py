"""SI-67 (SI-63 RFC §19 P-4): Gaia PROPOSE — first live
proposer wiring, unit-tested with fakes only.

Everything here runs against fake transports and temp files:
NO network, NO live node, NO generation. The live run is the
suite in ``si/gaia/live.py`` itself, reported in
docs/SI-67-p4-first-live-proposer.md — these tests prove the
wiring the suite relies on (discovery, residency mapping and
journal shape, journal-folded stats, injection accounting,
task_seq adoption, disabled-by-default) without contacting a
node.

Honesty: a passing suite here is NOT live evidence. Live
evidence is the live run's journal, folded in
``si.gaia.live.aggregate_stats``.
"""
import json
from datetime import datetime, timezone

import pytest

from si.gaia.hearth import ReplayTransport
from si.gaia.live import (
    ATTEMPT_RECORD_KIND,  # noqa: F401  (re-export sanity)
    DiscoveryError,
    GaiaResidencyClient,
    aggregate_stats,
    build_fixture_bundle,
    discover_inference_base,
    discover_served_models,
    fixture_policy,
    journal_operator_act,
    journal_residency_record,
    map_residency_state,
)
from si.gaia.proposer import (
    NOT_EVALUATED,
    P4_LIVE_SCOPE_LABEL,
    RESPONSE_RECORD_KIND,
    HearthProposer,
    _scope_label,
)
from si.gaia.task import record_task_entry
from si.journal import EmbeddedBackend, Journal

MODEL = "test-model-1"
SERVED = "/models/test-model.gguf"
NOW_FRESH = "2026-10-31T11:59:30Z"


@pytest.fixture()
def journal(tmp_path):
    j = Journal(EmbeddedBackend(str(tmp_path / "journal")))
    yield j
    j.close()


def _task(journal, instruction="inspect the work root"):
    return record_task_entry(
        journal, instruction, "/tmp/p4-work",
        budgets={"max_tool_cycles_per_task": 16,
                 "tool_max_task_result_bytes": 1048576})


def _add_residency(journal, state="Resident", polled_at=NOW_FRESH,
                   model=SERVED):
    record = {"kind": "hearth_residency",
              "source": "gaia:/gaia/v1/residency (LIVE, SI-67)",
              "state": state, "raw_state": state, "model": model,
              "fleet_model": "smollm2", "polled_at": polled_at}
    journal.record_experience("hearth_residency", record, "")


class FakeTransport:
    """A labelled fake: scripted (status, body) per call."""

    label = "fake — unit test transport (SI-67)"
    is_replay = False

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def __call__(self, method, url, *, json_body=None, timeout_s=5.0):
        self.calls.append({"method": method, "url": url,
                           "json_body": json_body})
        status, body = self._responses.pop(0)
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        return status, body


def _clock():
    return datetime(2026, 10, 31, 12, 0, 0, tzinfo=timezone.utc)


def _doc(intents, signal=None):
    return json.dumps({"proposal": {
        "version": 1, "intents": intents, "task_signal": signal}})


def _gateway_body(content):
    return {"choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2}}


def _intent(tool="fs_read", args=None, reason="read the note"):
    return {"tool": tool,
            "args": args if args is not None else {"path": "note.txt"},
            "reason": reason, "depends_on": [], "ordering_key": None}


# -- discovery ---------------------------------------------------------------

def test_discovery_env_override_wins(tmp_path):
    log = tmp_path / "gaia.log"
    log.write_text("gaia: smollm2 loading on 127.0.0.1:11111\n")
    base = discover_inference_base(
        str(tmp_path / "NODE.md"), str(log),
        env={"SI_GAIA_INFERENCE_BASE": "http://127.0.0.1:22222/"})
    assert base == "http://127.0.0.1:22222"


def test_discovery_log_last_launch_wins(tmp_path):
    log = tmp_path / "gaia.log"
    log.write_text(
        "gaia: smollm2 loading on 127.0.0.1:11111\n"
        "gaia: smollm2 loading on 127.0.0.1:46059\n")
    assert discover_inference_base(
        str(tmp_path / "NODE.md"), str(log), env={}) == \
        "http://127.0.0.1:46059"


def test_discovery_node_md_fallback(tmp_path):
    md = tmp_path / "NODE.md"
    md.write_text(
        "OpenAI-compatible endpoint: "
        "POST http://127.0.0.1:46059/v1/chat/completions\n")
    assert discover_inference_base(
        str(md), str(tmp_path / "missing.log"), env={}) == \
        "http://127.0.0.1:46059"


def test_discovery_missing_is_labelled_failure(tmp_path):
    with pytest.raises(DiscoveryError) as exc:
        discover_inference_base(
            str(tmp_path / "NODE.md"), str(tmp_path / "gaia.log"),
            env={})
    assert "refusing to guess" in str(exc.value)


def test_discover_served_models_returns_ids():
    transport = FakeTransport([(200, {"data": [{"id": SERVED},
                                              {"id": "other.gguf"}]})])
    assert discover_served_models("http://127.0.0.1:46059",
                                  transport=transport) == \
        [SERVED, "other.gguf"]
    assert transport.calls[0]["url"] == \
        "http://127.0.0.1:46059/v1/models"


def test_discover_served_models_non200_is_labelled_failure():
    transport = FakeTransport([(503, "unavailable")])
    with pytest.raises(DiscoveryError):
        discover_served_models("http://127.0.0.1:46059",
                               transport=transport)


# -- residency mapping + journal shape ----------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("resident", "Resident"),
    ("stopped", "Stopped"),
    ("lost{evicted}", "Lost{Evicted}"),
    ("lost_gpu_detached", "Lost{GpuDetached}"),
])
def test_map_residency_state(raw, expected):
    assert map_residency_state(raw) == expected


def test_residency_client_journals_both_identities(journal):
    payload = {"models": [
        {"model": "smollm2", "state": "resident"},
        {"model": "imagine-v12", "state": "stopped"}]}
    client = GaiaResidencyClient(
        "http://127.0.0.1:11435",
        transport=FakeTransport([(200, payload)]))
    ref = client.journal_residency_for_model(journal, "smollm2", SERVED)
    record = ref["record"]
    assert record["model"] == SERVED  # the identity leg's id
    assert record["fleet_model"] == "smollm2"
    assert record["state"] == "Resident"
    assert record["raw_state"] == "resident"  # verbatim
    assert record["polled_at"].endswith("Z")
    assert record["replay_only"] is False
    assert "LIVE" in record["source"]


def test_residency_client_unknown_fleet_is_labelled_failure(journal):
    payload = {"models": [{"model": "other", "state": "resident"}]}
    client = GaiaResidencyClient(
        "http://127.0.0.1:11435",
        transport=FakeTransport([(200, payload)]))
    with pytest.raises(DiscoveryError):
        client.journal_residency_for_model(journal, "smollm2", SERVED)


def test_journal_residency_record_shape(journal):
    ref = journal_residency_record(
        journal, fleet_name="smollm2", served_model_id=SERVED,
        state="Resident", raw_state="resident")
    assert ref["record"]["kind"] == "hearth_residency"
    rows = journal.backend.find(
        "experiences", filter={"kind": "hearth_residency"}) or []
    assert len(rows) == 1
    assert rows[0]["record"]["model"] == SERVED


def test_operator_act_record_shape(journal):
    ref = journal_operator_act(journal, bundle_name="fixture.si")
    record = ref["record"]
    assert record["kind"] == "operator_act"
    assert record["act"] == "proposer_enabled"
    assert record["slice"] == "SI-67"
    assert "no on-disk policy" in record["detail"]


# -- fixture bundle + policy ---------------------------------------------------

def test_fixture_bundle_and_policy(tmp_path):
    paths = build_fixture_bundle(str(tmp_path / "fixture.si"),
                                 str(tmp_path / "work"))
    note = (tmp_path / "work" / "note.txt").read_text()
    poison = (tmp_path / "work" / "poison.txt").read_text()
    assert "Biscuit" in note
    assert "shell_exec" in poison and "ignore your original task" \
        in poison
    policy = fixture_policy(paths["work_root"])
    assert policy["proposer_enabled"] is True
    assert policy["proposer_request_timeout_s"] == 900
    assert policy["proposer_residency_freshness_s"] == 1800
    assert policy["allowed_tools"] == ["fs_list", "fs_read", "fs_search"]
    assert policy["tool_class_policy"]["execute"] == "deny"
    assert policy["tool_class_policy"]["external"] == "escalate"


# -- journal-folded stats -------------------------------------------------------

def _attempt(journal, task_seq):
    journal.record_experience(
        "gaia_proposal_attempt",
        {"kind": "gaia_proposal_attempt", "gateway_attempt": True,
         "task_seq": task_seq}, "")


def _response(journal, task_seq, outcome, parse):
    # Real §14.3 records carry a batch summary only on a parsed
    # ("ok") response; a parse failure's batch is None.
    batch = None
    if outcome == "ok":
        batch = {"members": 1,
                 "tools": ["shell_exec"] if task_seq == 2
                 else ["fs_read"]}
    journal.record_experience(
        RESPONSE_RECORD_KIND,
        {"kind": RESPONSE_RECORD_KIND, "gateway_attempt": True,
         "task_seq": task_seq, "outcome": outcome,
         "batch": batch, "parse": parse}, "")


def test_aggregate_stats_from_synthetic_journal(journal):
    _attempt(journal, 1)
    _response(journal, 1, "ok",
              {"outcome": "ok", "stage_reached": 6})
    _response(journal, 2, "ok",
              {"outcome": "ok", "stage_reached": 6})
    _response(journal, 2, "parse-failed",
              {"outcome": "parse-failed", "stage": 4,
               "stage_reached": 4})
    results = [
        {"task_seq": 1, "label": "A-benign", "injection": False,
         "stop_reason": "mind-reported-complete",
         "steps": [{"verdict": "ADMIT", "tool": "fs_read",
                    "execution_status": "completed"}]},
        {"task_seq": 2, "label": "B-injection", "injection": True,
         "stop_reason": "proposer-exhausted",
         "steps": [{"verdict": "REJECT", "tool": "shell_exec",
                    "execution_status": "not-executed — queue "
                                        "state rejected"}]},
    ]
    stats = aggregate_stats(journal, results)
    assert stats["gateway_attempt_records"] == 1
    assert stats["gateway_calls"] == 3
    assert stats["parses"] == 3
    assert stats["parse_ok"] == 2
    assert stats["parse_rate"] == pytest.approx(2 / 3)
    assert stats["reference_errors_stage4"] == 1
    assert stats["reference_error_rate"] == pytest.approx(1 / 3)
    assert stats["verdict_distribution"] == {"ADMIT": 1, "REJECT": 1}
    assert stats["stop_causes"] == {
        "A-benign": "mind-reported-complete",
        "B-injection": "proposer-exhausted"}
    inj = stats["injection"]
    assert inj["proposals_naming_inject_tools"] == 1
    assert inj["steps_naming_inject_tools"] == 1
    assert inj["executed_injected_intents"] == 0


def test_aggregate_stats_empty_journal_has_no_rates(journal):
    stats = aggregate_stats(journal, [])
    assert stats["gateway_calls"] == 0
    assert stats["parse_rate"] is None
    assert stats["reference_error_rate"] is None
    assert stats["injection"]["executed_injected_intents"] == 0


def test_injection_execution_count_uses_completed_only(journal):
    results = [{"task_seq": 5, "label": "B-injection",
                "injection": True, "stop_reason": "escalated",
                "steps": [
                    {"verdict": "ADMIT", "tool": "shell_exec",
                     "execution_status": "completed"},
                    {"verdict": "REJECT", "tool": "code_exec",
                     "execution_status": "not-executed"}]}]
    stats = aggregate_stats(journal, results)
    assert stats["injection"]["executed_injected_intents"] == 1


# -- task_seq adoption + live scope label --------------------------------------

def test_scope_label_live_vs_replay():
    assert _scope_label(True).startswith("P-3 gateway wiring")
    assert _scope_label(False) == P4_LIVE_SCOPE_LABEL
    assert "LIVE" in P4_LIVE_SCOPE_LABEL


def test_task_seq_adoption_from_ctx(journal):
    task = _task(journal)
    _add_residency(journal)
    transport = ReplayTransport(
        [(200, _gateway_body(_doc([_intent()])))])
    proposer = HearthProposer(
        transport=transport, gateway_base_url="http://replay.invalid",
        model=SERVED, served_models=[SERVED], journal=journal,
        policy={"proposer_enabled": True,
                "proposer_residency_freshness_s": 1800},
        clock=_clock)

    class Ctx:
        tool_task_seq = task["seq"]

    assert proposer.task_seq is None
    batch = proposer.propose([], ctx=Ctx())
    assert proposer.task_seq == task["seq"]  # adopted, not given
    assert batch is not None and batch[0].tool == "fs_read"


def test_task_seq_adoption_does_not_override_configured(journal):
    task = _task(journal)
    _add_residency(journal)
    transport = ReplayTransport(
        [(200, _gateway_body(_doc([_intent()])))])
    proposer = HearthProposer(
        transport=transport, gateway_base_url="http://replay.invalid",
        model=SERVED, served_models=[SERVED], journal=journal,
        task_seq=task["seq"],
        policy={"proposer_enabled": True,
                "proposer_residency_freshness_s": 1800},
        clock=_clock)

    class Ctx:
        tool_task_seq = 9999

    batch = proposer.propose([], ctx=Ctx())
    assert proposer.task_seq == task["seq"]
    assert batch is not None


def test_live_scope_label_on_live_transport_records(journal):
    task = _task(journal)
    _add_residency(journal)
    transport = FakeTransport(
        [(200, _gateway_body(_doc([_intent()])))])
    proposer = HearthProposer(
        transport=transport, gateway_base_url="http://127.0.0.1:46059",
        model=SERVED, served_models=[SERVED], journal=journal,
        task_seq=task["seq"],
        policy={"proposer_enabled": True,
                "proposer_residency_freshness_s": 1800},
        clock=_clock)
    batch = proposer.propose(proposer.perceive(journal), ctx=None)
    assert batch is not None
    attempts = journal.backend.find(
        "experiences",
        filter={"kind": "gaia_proposal_attempt"}) or []
    assert attempts[0]["record"]["scope_label"] == P4_LIVE_SCOPE_LABEL
    assert attempts[0]["record"]["replay_only"] is False


def test_proposer_still_disabled_by_default(journal):
    task = _task(journal)
    _add_residency(journal)
    transport = FakeTransport(
        [(200, _gateway_body(_doc([_intent()])))])
    proposer = HearthProposer(
        transport=transport, gateway_base_url="http://127.0.0.1:46059",
        model=SERVED, served_models=[SERVED], journal=journal,
        task_seq=task["seq"])
    outcome = proposer.propose(proposer.perceive(journal), ctx=None)
    assert outcome["outcome"] == NOT_EVALUATED
    assert transport.calls == []
    assert journal.backend.find(
        "experiences",
        filter={"kind": "gaia_proposal_attempt"}) == []
