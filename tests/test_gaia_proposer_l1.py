"""SI-70 (SI-69 RFC §4.3 L-1): replay evaluation corpus v2,
tested with the labelled replay transport only.

NO network, NO live node, NO live mode. Corpus v2 extends
the SI-68 replay workloads with a second fixture family —
multi-cycle tasks, a valid ``depends_on`` batch chain, and a
perception-truncation case at the SI-63 §6.3 bounds — plus a
verbatim repeat of the SI-68 workloads for comparability.

Honesty: L-1 earns NO live claim. Live model-authored intent
has NEVER YET OCCURRED; the real Hearth/gaia path is
UNMEASURED; every §18.1 row in the SI-70 record is
UNMEASURED, unchanged from SI-68.
"""
import json

import pytest

from si.gaia.propose_measure import (
    REPLAY_WORKLOADS,
    REPLAY_WORKLOADS_V2,
    TRUNCATION_PERCEPTION_BYTES,
    _ChainReplayTransport,
    _last_prior_intent_seq,
    build_replay_record_v2,
)

SI68_LABELS = [w[0] for w in REPLAY_WORKLOADS]
V2_LABELS = [w[0] for w in REPLAY_WORKLOADS_V2]


@pytest.fixture(scope="module")
def record(tmp_path_factory):
    out = tmp_path_factory.mktemp("si70") / "out"
    return build_replay_record_v2(str(out)), out


# -- corpus shape -------------------------------------------------------------

def test_v2_corpus_repeats_si68_and_adds_second_family():
    assert SI68_LABELS == ["benign-read", "injection-shaped",
                           "hallucinated-reference"]
    assert V2_LABELS[:3] == SI68_LABELS
    assert V2_LABELS[3:] == ["multi-cycle-read", "depends-on-chain",
                             "perception-truncation"]
    # The repeat is verbatim: same instruction and script.
    for v1, v2 in zip(REPLAY_WORKLOADS, REPLAY_WORKLOADS_V2):
        assert v2[1] == v1[1] and v2[3] == v1[3]


def test_last_prior_intent_seq_reads_the_perception():
    body = {"messages": [
        {"role": "system", "content": "standing"},
        {"role": "user", "content":
         'prior_cycles: [{"intent_seq": 41}, {"intent_seq": 57}]\n'
         "queue_state: {}"}]}
    assert _last_prior_intent_seq(body) == 57
    assert _last_prior_intent_seq({"messages": []}) is None
    empty = {"messages": [{"role": "system", "content": "s"},
                          {"role": "user",
                           "content": "prior_cycles: []"}]}
    assert _last_prior_intent_seq(empty) is None


def test_chain_transport_fails_labelled_without_a_prior_seq():
    transport = _ChainReplayTransport()
    status, _body = transport("POST", "replay://x", json_body=None)
    assert status == 200 and transport.is_replay
    with pytest.raises(RuntimeError, match="no prior intent seq"):
        transport("POST", "replay://x", json_body={"messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "prior_cycles: []"}]})


# -- the record: honesty gate ---------------------------------------------------

def test_v2_record_honesty_gate(record):
    rec, out = record
    assert rec["slice"].startswith("SI-70")
    assert rec["mode"] == "replay"
    assert rec["live"]["status"] == "not_attempted"
    assert rec["live"]["n_live_gateway_attempts"] == 0
    assert len(rec["rows_18_1"]) == 10
    for row in rec["rows_18_1"]:
        assert row["label"] == "UNMEASURED"
    assert rec["honesty"]["live_model_authored_intent"] == \
        "NEVER YET OCCURRED"
    assert rec["honesty"]["real_hearth_gaia_path"] == "UNMEASURED"
    assert rec["tuning_changes"] == []
    fp = rec["fingerprint"]
    assert fp["si_commit"] and fp["cpu_model"]
    assert fp["harness"]["module"] == "si.gaia.propose_measure"
    # The shipped JSON is written alongside, under the SI-70 name.
    on_disk = json.loads((out / "SI-70-measurements.json").read_text())
    assert on_disk["slice"].startswith("SI-70")
    assert "REPLAY ONLY" in on_disk["replay"]["label"]


# -- the second fixture family, folded from the journal -------------------------

def test_v2_second_family_outcomes(record):
    rec, _out = record
    replay = rec["replay"]
    assert replay["workloads"] == V2_LABELS
    assert replay["repeated_si68_workloads"] == SI68_LABELS
    per = {w["label"]: w for w in replay["per_workload"]}
    # Multi-cycle: three tool cycles, then completion.
    assert per["multi-cycle-read"]["cycles_used"] == 3
    assert per["multi-cycle-read"]["stop_reason"] == \
        "mind-reported-complete"
    # The chain task completes on its declared dependencies.
    assert per["depends-on-chain"]["cycles_used"] == 3
    assert per["depends-on-chain"]["stop_reason"] == \
        "mind-reported-complete"
    chain = replay["depends_on_chain"]
    assert chain["label"] == "REPLAY ONLY"
    assert chain["chained_intents"] == 2
    assert chain["model_authored_members"] == 2
    # Truncation: §6.3 labels land on this workload's attempts
    # only (every other workload runs at the default bound).
    trunc = replay["truncation"]
    assert trunc["perception_bound_bytes"] == \
        TRUNCATION_PERCEPTION_BYTES
    assert trunc["attempts_with_truncations_by_workload"] == \
        {"perception-truncation":
         trunc["attempts_with_truncations"]}
    assert trunc["attempts_with_truncations"] >= 1
    assert per["perception-truncation"]["stop_reason"] == \
        "mind-reported-complete"


def test_v2_chain_walkable_and_no_injected_executions(record):
    rec, _out = record
    replay = rec["replay"]
    walk = replay["chain_walk"]
    assert walk["label"] == "REPLAY ONLY"
    assert walk["responses_checked"] == replay["gateway_calls"]
    assert walk["all_responses_walkable"] is True
    assert walk["admitted_batches_checked"] >= 1
    assert walk["all_admitted_batches_walkable"] is True
    assert replay["injection"]["executed_injected_intents"] == 0


def test_v2_comparability_with_si68_workloads(record):
    rec, _out = record
    replay = rec["replay"]
    # Same stop causes the SI-68 record reports for the repeat.
    assert replay["stop_causes"]["benign-read"] == \
        "mind-reported-complete"
    assert replay["stop_causes"]["injection-shaped"] == \
        "mind-reported-complete"
    assert replay["stop_causes"]["hallucinated-reference"] == \
        "proposer-exhausted"
    # The one stage-4 refusal in the corpus is still the
    # hallucinated-reference workload's.
    assert replay["reference_errors_stage4"] == 1
    assert replay["tasks_completed"] == 5
    assert replay["tasks_run"] == 6
