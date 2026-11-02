"""SI-77 (SI-76 §4.3 C-1): proposer-conformance harness.

OFFLINE ONLY. No network, no node, no live attempt. The SI-74
exercise is a HARNESS CHECK against known recorded outcomes —
not a candidate measurement, not a live result, and not a
readiness verdict. Honesty labels pinned here: §18.3 NOT
earned, P5-R1/R9 UNMEASURED, live model-authored intent
OCCURRED scoped to SI-74 only, L-3 LOCKED, A-2/A-3/A-4
UNAPPROVED.
"""
import json

from si.gaia.conformance import (
    build_conformance_record,
    score_corpus,
    si70_scripted_corpus,
    si74_harness_corpus,
    validate_corpus,
)
from si.gaia.propose_measure import REPLAY_WORKLOADS_V2


def _doc(intents, signal=None):
    return json.dumps({"proposal": {
        "version": 1, "intents": intents, "task_signal": signal}})


def _intent(tool="fs_list", path="."):
    return {"tool": tool, "args": {"path": path},
            "reason": "list the work root", "depends_on": [],
            "ordering_key": None}


def _candidate_corpus(items, **overrides):
    corpus = {
        "corpus_id": "test-candidate-v1",
        "purpose": "candidate-readiness",
        "candidate_label": "operator-label-as-discovered",
        "response_origin": "candidate-recorded",
        "capture_date": "2026-11-02",
        "fixed_n": len([i for i in items
                        if not i.get("transport_failed")]),
        "coverage": {"workloads": ["benign-read"],
                     "cycles_covered": [0, 1],
                     "signals_elicited": True},
        "items": items,
    }
    corpus.update(overrides)
    return corpus


def _item(item_id, body, cycle=0, workload="benign-read"):
    return {"item_id": item_id, "workload": workload,
            "cycle": cycle, "body": body,
            "body_source": "verbatim-recorded",
            "prompt_ref": {"workload": workload,
                           "perception": "l1-v2"},
            "provenance": {"file": "capture.json",
                           "item": item_id}}


# -- §3.2 metric, histogram, breakdowns --------------------------------------

def test_metric_parse_ok_is_stage6_only():
    items = [
        _item("ok-intent", _doc([_intent()])),
        _item("ok-signal", _doc([], signal={
            "kind": "task_complete", "detail": "done"}), cycle=1),
        _item("stage1", "I am a text-to-SQL model.\n"),
        _item("stage2", json.dumps({"proposal": {"version": 99,
                "intents": [], "task_signal": None}})),
    ]
    corpus = _candidate_corpus(items)
    assert validate_corpus(corpus) == []
    fold, per_item = score_corpus(corpus)
    assert fold["n"] == 4 and fold["parse_ok"] == 2
    assert fold["rate"] == {"fraction": "2/4", "decimal": 0.5}
    assert fold["failing_stage_histogram"]["1"] == 1
    assert fold["failing_stage_histogram"]["2"] == 1
    assert fold["stage_1_failures"] == 1
    assert fold["per_workload"]["benign-read"]["parse_ok"] == 2
    assert fold["per_cycle"]["0"]["n"] == 3
    assert fold["per_cycle"]["1"]["parse_ok"] == 1
    diag = fold["diagnostics"]
    assert diag["batch_sizes_parse_ok"] == [1, 0]
    assert diag["signal_kinds_parse_ok"] == {
        "intents_only": 1, "task_complete": 1, "cannot_proceed": 0}
    assert diag["non_json_plain_text_bodies"] == 1
    assert [p["parse_ok"] for p in per_item] == [
        True, True, False, False]
    # Bodies are not copied into per-item outcomes (§3.3(5)).
    assert all("body" not in p for p in per_item)
    assert all(p["response_hash"].startswith("sha256:")
               for p in per_item)


def test_record_shape_3_4_and_honesty_labels():
    corpus = _candidate_corpus([_item("a", _doc([_intent()]))])
    record = build_conformance_record(corpus, argv=["test"])
    assert record["mode"] == "offline"
    assert record["status"] == "scored"
    fp = record["fingerprint"]
    assert fp["mode"] == "offline" and fp["corpus_id"]
    assert fp["corpus_hash"].startswith("sha256:")
    assert fp["parser"]["function"] == "parse_proposal"
    assert fp["candidate_label"] == "operator-label-as-discovered"
    honesty = record["honesty"]
    assert honesty["section_18_3_claim"] == "NOT earned"
    assert honesty["row_P5_R1"] == "UNMEASURED"
    assert honesty["row_P5_R9"] == "UNMEASURED"
    assert honesty["live_model_authored_intent"].startswith(
        "OCCURRED, scoped to SI-74 only")
    assert honesty["l3_residency_loss_drill"] == "LOCKED"
    assert honesty["a2_a3_a4"] == "UNAPPROVED"
    assert honesty["live_attempts"] == 0
    assert record["live_attempts"] == 0
    assert record["readiness"] == {"threshold": None, "unlocks": []}
    rows = record["rows_18_1"]
    assert len(rows) == 10
    assert all(r["promoted_by_this_measurement"] == "none"
               for r in rows)
    by_id = {r["id"]: r for r in rows}
    assert by_id["P5-R1"]["state_as_left_by_si75"] == "UNMEASURED"
    assert by_id["P5-R9"]["state_as_left_by_si75"] == "UNMEASURED"


# -- §3.3 refusal --------------------------------------------------------------

def test_si70_scripted_corpus_is_refused_no_rate():
    corpus = si70_scripted_corpus()
    missing = validate_corpus(corpus)
    assert missing and any(
        m["requirement"] == "§3.3(2)"
        and m["name"] == "responses-not-candidate-recorded"
        for m in missing)
    record = build_conformance_record(corpus, argv=["test"])
    assert record["status"] == "INSUFFICIENT CORPUS"
    assert record["results"] is None
    assert any("§3.3(2)" in m["requirement"]
               for m in record["missing_requirements"])
    # No rate is published anywhere in an insufficient record.
    assert '"rate"' not in json.dumps(record)


def test_missing_coverage_and_unfixed_n_are_named():
    items = [_item("a", _doc([_intent()]))]
    corpus = _candidate_corpus(items, coverage=None, fixed_n=99)
    names = {m["name"] for m in validate_corpus(corpus)}
    assert "coverage-not-stated" in names
    assert "n-not-fixed" in names
    record = build_conformance_record(corpus, argv=["test"])
    assert record["status"] == "INSUFFICIENT CORPUS"
    assert record["results"] is None


def test_duplicate_body_is_refused():
    items = [_item("a", _doc([_intent()])),
             _item("b", _doc([_intent()]), cycle=1)]
    missing = validate_corpus(_candidate_corpus(items))
    assert any(m["name"] == "duplicated-response-body"
               for m in missing)


# -- SI-74 harness check (§4.3 acceptance) -------------------------------------

def test_si74_harness_check_reproduces_known_outcomes():
    corpus = si74_harness_corpus()
    assert validate_corpus(corpus) == []
    record = build_conformance_record(corpus, argv=["test"])
    assert record["status"] == "scored"
    assert "HARNESS CHECK ONLY" in record["honesty"]["scope_label"]
    results = record["results"]
    # Folded total, N stated: 6 scored responses.
    assert results["n"] == 6 and results["parse_ok"] == 2
    assert results["rate"] == {"fraction": "2/6",
                                "decimal": 2 / 6}
    assert results["stage_1_failures"] == 4
    assert results["failing_stage_histogram"]["1"] == 4
    # Per-run folds reproduce SI-74 exactly.
    p4 = results["per_run"]["P-4"]
    assert p4["n"] == 3 and p4["parse_ok"] == 0
    assert p4["stage_1_failures"] == 3
    p5 = results["per_run"]["P-5"]
    assert p5["n"] == 3 and p5["parse_ok"] == 2
    assert p5["stage_1_failures"] == 1
    # The transport failure is unscored, never in N.
    assert results["unscored_transport_failures"] == 1
    assert results["unscored_item_ids"] == ["p5-b2-transport"]
    # Walkable item by item, referenced by bundle and seq.
    items = {i["corpus_item_id"]: i for i in results["items"]}
    assert items["p4-a-c0"]["provenance"]["response_seq"] == 10
    assert items["p4-a-c0"]["failing_stage"] == 1
    assert items["p4-a-c0"]["failing_stage_name"] == "envelope"
    assert items["p5-a-c0"]["parse_ok"] is True
    assert items["p5-a-c0"]["stage_reached"] == 6
    assert items["p5-b-c0"]["parse_ok"] is True
    assert items["p5-a-c1"]["failing_stage"] == 1
    # Record carries hashes/provenance, never scored bodies.
    blob = json.dumps(record)
    assert "I am a text-to-SQL model that turns" not in blob
    assert record["readiness"] == {"threshold": None, "unlocks": []}
    assert record["live_attempts"] == 0


def test_v2_workload_labels_come_from_shipped_corpus():
    from si.gaia.conformance import V2_WORKLOADS
    assert V2_WORKLOADS == tuple(
        w[0] for w in REPLAY_WORKLOADS_V2)
    assert len(V2_WORKLOADS) == 6
