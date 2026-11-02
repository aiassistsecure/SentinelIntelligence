"""SI-68 (SI-63 RFC §19 P-5): Gaia PROPOSE — measurement
harness + preflight hardening, tested with fakes/replay only.

NO network, NO live node, NO SI_P5_LIVE run. The live mode is
exercised only against scripted fake transports: success path
(preflight passes, suite runs on fakes — still not live
evidence) and failure paths (preflight fails closed, the
environmental failure is journaled, no task runs).

Honesty: nothing here is live evidence. Live model-authored
intent has NEVER YET OCCURRED; every §18.1 row in the shipped
SI-68 record is UNMEASURED.
"""
import json

import pytest

from si.gaia.hearth import ReplayTransport  # noqa: F401  (label sanity)
from si.gaia.live import (
    ENVIRONMENTAL_FAILURE_KIND,
    FIXTURE_FLEET_MODEL,
    DiscoveryError,
    journal_environmental_failure,
    preflight_live,
)
from si.gaia.propose_measure import (
    ROWS_18_1,
    build_live_record,
    build_replay_record,
    environment_fingerprint,
    summarize,
)
from si.journal import EmbeddedBackend, Journal

SERVED = "/models/test-model.gguf"
ENV = {"SI_GAIA_INFERENCE_BASE": "http://127.0.0.1:9999"}


@pytest.fixture()
def journal(tmp_path):
    j = Journal(EmbeddedBackend(str(tmp_path / "journal")))
    yield j
    j.close()


class FakeTransport:
    """A labelled fake: scripted (status, body) per call, or an
    exception instance to raise."""

    label = "fake — unit test transport (SI-68)"
    is_replay = False

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def __call__(self, method, url, *, json_body=None, timeout_s=5.0):
        self.calls.append({"method": method, "url": url,
                           "json_body": json_body})
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        status, body = item
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        return status, body


def _models_body():
    return {"data": [{"id": SERVED}]}


def _residency_body(state="resident"):
    return {"models": [{"model": "imagine-v12", "state": state}]}


def _doc(intents, signal=None):
    return json.dumps({"proposal": {
        "version": 1, "intents": intents, "task_signal": signal}})


def _gateway_body(content):
    return {"choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2}}


def _intent(tool="fs_read", args=None):
    return {"tool": tool,
            "args": args if args is not None else {"path": "note.txt"},
            "reason": "read the note", "depends_on": [],
            "ordering_key": None}


# -- summarize (S-7 discipline) ----------------------------------------------

def test_summarize_empty_is_all_none():
    s = summarize([])
    assert s == {"n": 0, "min": None, "median": None,
                 "mean": None, "p95": None, "max": None}


def test_summarize_p95_withheld_below_30():
    s = summarize([1.0, 2.0, 3.0])
    assert s["n"] == 3 and s["median"] == 2.0 and s["p95"] is None
    s30 = summarize([float(i) for i in range(30)])
    assert s30["p95"] is not None


# -- fingerprint --------------------------------------------------------------

def test_fingerprint_shape():
    fp = environment_fingerprint(argv=["-m", "si.gaia.propose_measure"])
    assert fp["gaia_commit"] == "not-used"
    assert fp["journal_backend"] == "embedded"
    assert fp["harness"]["module"] == "si.gaia.propose_measure"
    assert fp["python_version"] and fp["cpu_count_os"]


# -- preflight (SI-68 hardening in si/gaia/live.py) ---------------------------

def test_fixture_fleet_model_is_imagine_v12():
    # SI-73: the node serves imagine-v12 as its Resident mind
    # (SmolLM2 ditched 2026-11-01); the fixture fleet priced
    # by the live harness must name that model, or the
    # residency leg of a live re-run fails closed.
    assert FIXTURE_FLEET_MODEL == "imagine-v12"


def test_preflight_success_with_fakes():
    transport = FakeTransport([(200, _models_body()),
                               (200, _residency_body())])
    ready = preflight_live(env=ENV, transport=transport)
    assert ready["ready"] is True
    assert ready["served_model_id"] == SERVED
    assert ready["residency_state"] == "Resident"
    assert ready["inference_base_observed"] == "http://127.0.0.1:9999"


def test_preflight_discovery_failure_is_stage_labelled(tmp_path):
    with pytest.raises(DiscoveryError) as exc:
        preflight_live(env={}, transport=FakeTransport([]),
                       node_md_path=str(tmp_path / "nope.md"),
                       gaia_log_path=str(tmp_path / "nope.log"))
    assert "[stage=discovery]" in str(exc.value)


def test_preflight_connection_refused_is_stage_labelled():
    transport = FakeTransport(
        [ConnectionError("connection refused")])
    with pytest.raises(DiscoveryError) as exc:
        preflight_live(env=ENV, transport=transport)
    assert "[stage=served_models]" in str(exc.value)


def test_preflight_residency_without_fleet_model_fails():
    transport = FakeTransport(
        [(200, _models_body()),
         (200, {"models": [{"model": "other", "state": "resident"}]})])
    with pytest.raises(DiscoveryError) as exc:
        preflight_live(env=ENV, transport=transport)
    assert "[stage=residency]" in str(exc.value)


def test_environmental_failure_record_shape(journal):
    ref = journal_environmental_failure(
        journal, stage="served_models", detail="connection refused")
    rec = ref["record"]
    assert rec["kind"] == ENVIRONMENTAL_FAILURE_KIND
    assert rec["measurement_taken"] is False
    assert rec["tasks_run"] == 0
    assert "ENVIRONMENTAL FAILURE" in rec["label"]
    rows = journal.backend.find(
        "experiences", filter={"kind": ENVIRONMENTAL_FAILURE_KIND})
    assert len(rows) == 1


# -- replay record --------------------------------------------------------------

def test_replay_record_all_rows_unmeasured(tmp_path):
    record = build_replay_record(str(tmp_path / "out"))
    assert record["mode"] == "replay"
    assert record["live"]["status"] == "not_attempted"
    assert record["live"]["n_live_gateway_attempts"] == 0
    assert len(record["rows_18_1"]) == len(ROWS_18_1) == 10
    for row in record["rows_18_1"]:
        assert row["label"] == "UNMEASURED"
        assert "gaia node unavailable" in row["reason"]
    assert record["tuning_changes"] == []
    assert record["honesty"]["live_model_authored_intent"] == \
        "NEVER YET OCCURRED"
    # The shipped JSON is written alongside.
    on_disk = json.loads(
        (tmp_path / "out" / "SI-68-measurements.json").read_text())
    assert on_disk["slice"].startswith("SI-68")


def test_replay_stats_are_labelled_and_journal_folded(tmp_path):
    record = build_replay_record(str(tmp_path / "out"))
    replay = record["replay"]
    assert "REPLAY ONLY" in replay["label"]
    assert replay["workloads"] == ["benign-read", "injection-shaped",
                                   "hallucinated-reference"]
    # Benign + injection workloads take 2 gateway calls each;
    # the hallucinated-reference workload takes 1 — its stage-4
    # refusal halts proposing (stop cause proposer-exhausted),
    # so the scripted completion response is never consumed.
    assert replay["gateway_calls"] == 5
    assert replay["parses"] == 5
    assert replay["reference_errors_stage4"] == 1
    assert replay["tasks_completed"] == 2
    assert replay["stop_causes"]["hallucinated-reference"] == \
        "proposer-exhausted"
    inj = replay["injection"]
    assert inj["proposals_naming_inject_tools"] == 1
    assert inj["executed_injected_intents"] == 0
    obs = {r["id"]: r["replay_observation"]
           for r in record["rows_18_1"]}
    assert obs["P5-R3"]["label"] == "REPLAY ONLY"
    assert obs["P5-R8"]["executed_injected_intents"] == 0
    assert obs["P5-R1"] is None and obs["P5-R9"] is None


# -- live record: fail closed ---------------------------------------------------

def test_live_record_fails_closed_and_journals(tmp_path):
    out = tmp_path / "out"
    transport = FakeTransport([ConnectionError("connection refused")])
    record = build_live_record(str(out), env=ENV, transport=transport)
    assert record["live"]["status"] == "environmental_failure"
    assert record["live"]["stage"] == "served_models"
    assert record["live"]["n_live_gateway_attempts"] == 0
    for row in record["rows_18_1"]:
        assert row["label"] == "UNMEASURED"
    # The failure is a walkable journal record in the bundle,
    # and NO task was journaled.
    from si.journal import open_journal
    journal = open_journal(str(out / "measurement.si"))
    try:
        fails = journal.backend.find(
            "experiences", filter={"kind": ENVIRONMENTAL_FAILURE_KIND})
        assert len(fails) == 1
        assert journal.backend.find(
            "experiences", filter={"kind": "task"}) == []
    finally:
        journal.close()


def test_live_record_discovery_failure_stage(tmp_path):
    record = build_live_record(
        str(tmp_path / "out"), env={},
        transport=FakeTransport([]),
        node_md_path=str(tmp_path / "nope.md"),
        gaia_log_path=str(tmp_path / "nope.log"))
    assert record["live"]["status"] == "environmental_failure"
    assert record["live"]["stage"] == "discovery"


# -- live record: preflight passes on fakes (still NOT live evidence) ----------

def test_live_record_suite_path_on_fakes(tmp_path):
    complete = _doc([], signal={"kind": "task_complete",
                                "detail": "done"})
    read = _doc([_intent()])
    responses = [
        (200, _models_body()),          # preflight: served models
        (200, _residency_body()),       # preflight: residency poll
        (200, _models_body()),          # suite: served models
        (200, _residency_body()),       # suite: task A residency
        (200, _gateway_body(read)),     # task A cycle 1
        (200, _gateway_body(complete)),  # task A cycle 2
        (200, _residency_body()),       # suite: task B residency
        (200, _gateway_body(complete)),  # task B cycle 1
        (200, _residency_body()),       # suite: task B2 residency
        (200, _gateway_body(complete)),  # task B2 cycle 1
    ]
    record = build_live_record(str(tmp_path / "out"), env=ENV,
                               transport=FakeTransport(responses))
    assert record["live"]["status"] == "suite_completed"
    assert record["live"]["preflight"]["served_model_id"] == SERVED
    assert record["live"]["n_live_gateway_attempts"] == 4
    # Even a completed fake suite never promotes the rows.
    for row in record["rows_18_1"]:
        assert row["label"] == "UNMEASURED"
