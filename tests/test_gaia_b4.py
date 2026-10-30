"""SI-45 (SI-41 RFC §15 B-4): Gaia's Hearth interface.

B-4 builds, SI-side only, in si/gaia/hearth.py: the §12.2.1
declaration record, the residency PERCEIVE adapter (§12.2.2), and
the gateway client for the conditional inference leg (§12.2.3).

Honesty discipline asserted throughout: no real Hearth node exists
in this environment, so every transport in this file is the
labelled replay/stub; residency observations are data with full
poll provenance; a failed poll is an observation, not an
exception (§12.4); the inference leg refuses — without calling
its transport — unless a Hearth-served real base is asserted
Resident; the Hearth path's performance is UNMEASURED and A-1
numbers are never cited as its predictions.
"""
import json
import os

import pytest

from si.bundle import init_bundle, load_bundle, manifest_id
from si.gaia.hearth import (
    HEARTH_PATH_UNMEASURED,
    INFERENCE_NOT_EVALUATED,
    REPLAY_TRANSPORT_LABEL,
    RESIDENCY_SOURCE,
    RESIDENCY_STATES,
    HearthGatewayClient,
    HearthResidencyAdapter,
    ReplayTransport,
    build_declaration,
    declaration_from_bundle,
    journal_residency_observation,
    parse_residency_state,
    residency_history,
    validate_declaration,
)
from si.journal import open_journal
from si.tiny_seed import seed_tiny_bundle

BASE_URL = "http://hearth.invalid:9999"


def _mini_bundle(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "mini.si"), name="gaia-b4",
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    return bundle_dir


def _residency_body(state, node="node-a"):
    return {"state": state, "node": node, "model": "miniature"}


# -- state vocabulary -----------------------------------------------------------

def test_residency_states_are_exactly_the_seven_named_states():
    assert RESIDENCY_STATES == (
        "Resident", "Loading", "Lost{Evicted}", "Lost{GpuDetached}",
        "Failed", "Stopped", "NotAdmitted")


@pytest.mark.parametrize("raw,expect", [
    ("Resident", "Resident"),
    ("Loading", "Loading"),
    ("Failed", "Failed"),
    ("Stopped", "Stopped"),
    ("NotAdmitted", "NotAdmitted"),
    ("not_admitted", "NotAdmitted"),
    ("Lost{Evicted}", "Lost{Evicted}"),
    ("LostEvicted", "Lost{Evicted}"),
    ("lost_evicted", "Lost{Evicted}"),
    ("Lost{GpuDetached}", "Lost{GpuDetached}"),
    ("lost_gpu_detached", "Lost{GpuDetached}"),
    ({"state": "Lost", "substate": "Evicted"}, "Lost{Evicted}"),
    ({"Lost": "GpuDetached"}, "Lost{GpuDetached}"),
    ({"Lost": {"Evicted": None}}, "Lost{Evicted}"),
])
def test_parse_residency_state_canonicalises(raw, expect):
    assert parse_residency_state(raw) == expect


@pytest.mark.parametrize("raw", ["Warm", "Unknown", "", None, 42,
                                   {"state": "Warm"}])
def test_parse_residency_state_unrecognised_is_none_never_coerced(raw):
    assert parse_residency_state(raw) is None


# -- declaration record (§12.2.1) -------------------------------------------------

def test_declaration_from_miniature_bundle_manifest(tmp_path):
    bundle_dir = _mini_bundle(tmp_path)
    manifest = load_bundle(bundle_dir)
    rec = build_declaration(manifest)
    assert validate_declaration(rec) == []
    assert rec["bundle"]["name"] == "gaia-b4"
    assert rec["bundle"]["bundle_id"] == manifest_id(manifest)
    model = rec["model"]
    # Identity and digests are the manifest's own, verbatim.
    assert model["identity"] == manifest["base"]["source"]
    assert "not a trained model" in model["identity"]
    assert model["digests"], "miniature base must record digests"
    for row in model["digests"]:
        assert row["sha256"].startswith("sha256:")
    assert model["digest"] == model["digests"][0]["sha256"]
    # No quant is recorded for the miniature; none is claimed.
    assert model["quant"]["value"] is None
    assert "no quantisation recorded" in model["quant"]["label"]
    # The fleet side is named as Mark's proposal, not built.
    assert "H-1" in rec["fleet_side"]
    assert rec["residency_requirement"]["state"] == "Resident"


def test_declaration_from_reference_style_manifest():
    # A by-reference base in the SmolLM2/Glimmer manifest shape:
    # digests come from parts, bytes stored 0, referenced counted.
    manifest = {
        "si_version": "1.0.0", "name": "ref-base", "head": "base",
        "base": {
            "source": "HuggingFaceTB/SmolLM2-360M-Instruct",
            "revision": "sha256:aaaa",
            "blob": "sha256:" + "b" * 64,
            "bytes": 0,
            "format": "safetensors",
            "seed_format": "si-ref/1",
            "deduplicated_bytes": 723666216,
            "checkpoint_config": {"architectures": ["LlamaForCausalLM"],
                                  "model_type": "llama",
                                  "torch_dtype": "bfloat16"},
            "parts": [{"file": "model.safetensors", "bytes": 723674912,
                       "sha256": "sha256:" + "e" * 64}],
        },
    }
    rec = build_declaration(manifest)
    assert validate_declaration(rec) == []
    model = rec["model"]
    assert model["by_reference"] is True
    assert model["bytes"]["stored_in_bundle"] == 0
    assert model["bytes"]["referenced"] == 723666216
    assert model["dtype"] == "bfloat16"
    assert model["architecture"]["model_type"] == "llama"
    digests = {r["sha256"] for r in model["digests"]}
    assert "sha256:" + "e" * 64 in digests


def test_declaration_digest_is_deterministic_and_tamper_evident():
    manifest = {"si_version": "1.0.0", "name": "x", "head": "base",
                "base": {"source": "demo:random-seed (not a real model)",
                         "blob": "sha256:" + "c" * 64, "bytes": 300}}
    a = build_declaration(manifest)
    b = build_declaration(manifest)
    assert a["declaration_digest"] == b["declaration_digest"]
    b["model"]["identity"] = "something else entirely"
    assert validate_declaration(b), "tampered record must fail validation"


def test_declaration_from_bundle_loads_and_verifies(tmp_path):
    bundle_dir = _mini_bundle(tmp_path)
    rec = declaration_from_bundle(bundle_dir)
    assert validate_declaration(rec) == []
    assert rec["bundle"]["name"] == "gaia-b4"


def test_declaration_requires_identity_and_digest():
    rec = build_declaration({"si_version": "1.0.0", "name": "x",
                             "head": "base", "base": {}})
    problems = validate_declaration(rec)
    assert any("identity" in p for p in problems)
    assert any("digests" in p for p in problems)


# -- residency adapter (§12.2.2) --------------------------------------------------

def test_poll_produces_provenance_tagged_observation():
    transport = ReplayTransport([_residency_body("Resident")])
    adapter = HearthResidencyAdapter(BASE_URL, transport=transport)
    obs = adapter.poll()
    assert obs["source"] == RESIDENCY_SOURCE == "hearth:/residency"
    assert obs["state"] == "Resident"
    assert obs["raw_state"] == "Resident"
    assert obs["node"] == "node-a"
    assert obs["polled_at"] and obs["poll_id"].startswith("hearth-poll:1:")
    assert obs["endpoint"] == BASE_URL + "/residency"
    assert obs["transport"] == REPLAY_TRANSPORT_LABEL
    assert obs["observation_status"] == "observed"
    assert "never a live guarantee" in obs["label"]
    assert "never instructions" in obs["data_not_instructions"]
    assert transport.calls[0]["method"] == "GET"


def test_poll_unrecognised_state_recorded_verbatim():
    adapter = HearthResidencyAdapter(
        BASE_URL, transport=ReplayTransport([_residency_body("Warm")]))
    obs = adapter.poll()
    assert obs["state"] is None
    assert obs["raw_state"] == "Warm"
    assert obs["state_parse"] == "unrecognised"
    assert "unrecognised" in obs["observation_status"]


def test_failed_poll_is_an_observation_not_an_exception():
    adapter = HearthResidencyAdapter(
        BASE_URL, transport=ReplayTransport(
            [ConnectionError("no route to host")]))
    obs = adapter.poll()  # must not raise (§12.4)
    assert obs["state"] is None
    assert obs["observation_status"].startswith("poll-failed")
    assert "ConnectionError" in obs["failure"]


def test_poll_http_error_status_labelled_failed():
    adapter = HearthResidencyAdapter(
        BASE_URL, transport=ReplayTransport(
            [(503, {"state": "Resident", "node": "node-a"})]))
    obs = adapter.poll()
    assert obs["http_status"] == 503
    assert obs["observation_status"].startswith("poll-failed")


def test_replay_exhaustion_is_a_labelled_failed_poll():
    adapter = HearthResidencyAdapter(
        BASE_URL, transport=ReplayTransport([]))
    obs = adapter.poll()
    assert obs["observation_status"].startswith("poll-failed")
    assert "replay/stub" in obs["failure"]


def test_residency_transitions_including_lost_evicted_journaled(tmp_path):
    bundle_dir = _mini_bundle(tmp_path)
    manifest = load_bundle(bundle_dir)
    journal = open_journal(bundle_dir, manifest)
    try:
        transport = ReplayTransport([
            _residency_body("Resident"),
            _residency_body("Lost{Evicted}"),
            _residency_body("Loading"),
            _residency_body("Resident"),
        ])
        adapter = HearthResidencyAdapter(BASE_URL, transport=transport)
        seqs = []
        for _ in range(4):
            out = adapter.poll_and_journal(journal)
            seqs.append(out["seq"])
        assert seqs == sorted(seqs) and len(set(seqs)) == 4
        history = residency_history(journal)
        assert [h["state"] for h in history] == [
            "Resident", "Lost{Evicted}", "Loading", "Resident"]
        for h in history:
            assert h["source"] == "hearth:/residency"
            assert h["transport"] == REPLAY_TRANSPORT_LABEL
            assert h["poll_id"] and h["polled_at"]
        # The eviction is in the journal, walkable like any history.
        evicted = history[1]
        assert evicted["raw_state"] == "Lost{Evicted}"
    finally:
        journal.close()


def test_journal_residency_observation_uses_existing_collection(tmp_path):
    bundle_dir = _mini_bundle(tmp_path)
    manifest = load_bundle(bundle_dir)
    journal = open_journal(bundle_dir, manifest)
    try:
        adapter = HearthResidencyAdapter(
            BASE_URL, transport=ReplayTransport(
                [_residency_body("Stopped", node="node-b")]))
        obs = adapter.poll()
        ref = journal_residency_observation(journal, obs)
        assert ref["seq"] > 0 and ref["row_id"]
        rows = journal.backend.find(
            "experiences", filter={"kind": "hearth_residency"})
        assert len(rows) == 1
        assert rows[0]["record"]["state"] == "Stopped"
        assert rows[0]["record"]["node"] == "node-b"
    finally:
        journal.close()


# -- gateway client (§12.2.3, conditional inference leg) --------------------------

def _completion_body(content="replayed completion"):
    return {"choices": [{"message": {"role": "assistant",
                                     "content": content}}],
            "usage": {"total_tokens": 7}}


def test_gateway_refuses_inference_without_served_real_base():
    transport = ReplayTransport([_completion_body()])
    client = HearthGatewayClient(BASE_URL, model="m", transport=transport)
    for kw in ({}, {"residency_state": "Resident"},
               {"served_real_base": True},
               {"served_real_base": True, "residency_state": "Loading"}):
        out = client.chat_completion([{"role": "user", "content": "hi"}],
                                     **kw)
        assert out["status"] == INFERENCE_NOT_EVALUATED
        assert out["content"] is None
        assert out["transport_called"] is False
        assert out["hearth_path_performance"] == HEARTH_PATH_UNMEASURED
    # The refusal never touched the transport — no fake inference.
    assert transport.calls == []


def test_gateway_completion_via_replay_is_labelled_replay():
    transport = ReplayTransport([_completion_body("stub answer")])
    client = HearthGatewayClient(BASE_URL, model="m", transport=transport)
    out = client.chat_completion(
        [{"role": "user", "content": "hi"}],
        residency_state="Resident", served_real_base=True)
    assert out["transport_called"] is True
    assert out["content"] == "stub answer"
    assert "replay/stub" in out["status"]
    assert "not a real base's output" in out["status"]
    assert out["hearth_path_performance"] == HEARTH_PATH_UNMEASURED
    sent = transport.calls[0]
    assert sent["method"] == "POST"
    assert sent["url"] == BASE_URL + "/v1/chat/completions"
    assert sent["json_body"]["model"] == "m"


def test_gateway_list_models_labelled_and_failure_as_data():
    ok = HearthGatewayClient(
        BASE_URL, transport=ReplayTransport(
            [{"data": [{"id": "model-a"}]}]))
    out = ok.list_models()
    assert out["status"] == "observed"
    assert out["models"] == [{"id": "model-a"}]
    assert out["transport"] == REPLAY_TRANSPORT_LABEL

    down = HearthGatewayClient(
        BASE_URL, transport=ReplayTransport([TimeoutError("slow")]))
    out = down.list_models()  # must not raise (§12.4)
    assert out["status"].startswith("failed")
    assert out["models"] is None
    assert "TimeoutError" in out["failure"]


def test_gateway_completion_failure_is_data_not_exception():
    client = HearthGatewayClient(
        BASE_URL, model="m",
        transport=ReplayTransport([ConnectionError("down")]))
    out = client.chat_completion(
        [{"role": "user", "content": "hi"}],
        residency_state="Resident", served_real_base=True)
    assert out["status"].startswith("failed")
    assert out["content"] is None


# -- honesty guards -----------------------------------------------------------------

def test_no_local_inference_engine_was_built():
    import si.gaia.hearth as hearth_mod
    src = open(hearth_mod.__file__).read()
    # B-4 must not grow an SI-local transformer inference path.
    assert "tiny_transformer" not in src
    assert "llama_measure" not in src
    assert "import torch" not in src
    assert INFERENCE_NOT_EVALUATED.startswith("not-evaluated")
    assert "UNMEASURED" in HEARTH_PATH_UNMEASURED
