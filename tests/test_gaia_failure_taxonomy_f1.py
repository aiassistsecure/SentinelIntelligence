"""SI-81 (SI-80 section 3.3 F-1): failure taxonomy over SI-79.

OFFLINE ONLY. Loads the stored SI-79 corpus + measurements and
the SI-81 taxonomy record from docs/, re-runs the stated
predicates, and checks the record reproduces exactly: stage-1
classes sum to 18, stage-4 to 2, control to 10, total 30; every
scored body appears exactly once, by hash; the 6 transport-failed
items are excluded and classified as nothing.

This is a classification check, not a re-measurement: SI-79
stage outcomes are inputs and no parse rate is computed here.
Honesty labels pinned: section 18.3 NOT earned, P5-R1/R9
UNMEASURED, live model-authored intent OCCURRED scoped to
SI-74 only, L-3 LOCKED, A-2/A-3/A-4 UNAPPROVED, unlocks [].
"""
import json
from pathlib import Path

from si.gaia.failure_taxonomy import build_taxonomy

DOCS = Path(__file__).resolve().parent.parent / "docs"


def _load(name):
    return json.loads((DOCS / name).read_text(encoding="utf-8"))


def _record():
    corpus = _load("SI-79-corpus.json")
    measurements = _load("SI-79-measurements.json")
    stored = _load("SI-81-taxonomy.json")
    return corpus, measurements, stored, build_taxonomy(
        corpus, measurements)


def test_sums_match_si79_totals():
    _, _, stored, rebuilt = _record()
    for key in ("partitions", "excluded_unscored", "totals"):
        assert stored[key] == rebuilt[key]
    t = stored["totals"]
    assert (t["stage_1"], t["stage_4"], t["control"]) == (18, 2, 10)
    assert t["scored"] == 30 and t["unclassified"] == 0
    assert stored["excluded_unscored"]["count"] == 6


def test_every_scored_body_exactly_once_by_hash():
    _, measurements, stored, _ = _record()
    cited = [i["response_hash"]
             for p in stored["partitions"].values()
             for c in p["classes"] for i in c["items"]]
    expected = [r["response_hash"]
                for r in measurements["results"]["items"]]
    assert sorted(cited) == sorted(expected)
    assert len(cited) == len(set(cited)) == 30


def test_class_definitions_counts_match_partitions():
    _, _, stored, _ = _record()
    defs = stored["method"]["class_definitions"]
    parts = stored["partitions"]
    for section, part in (("stage_1", parts["stage_1_failures"]),
                          ("stage_4", parts["stage_4_failures"]),
                          ("control", parts["control_parse_ok"])):
        actual = {c["class"]: c["count"] for c in part["classes"]}
        for cls, d in defs[section].items():
            assert d["observed_count"] == actual[cls], (section, cls)


def test_honesty_labels_unchanged():
    _, _, stored, _ = _record()
    h = stored["honesty"]
    assert h["section_18_3_claim"] == "NOT earned"
    assert h["row_P5_R1"] == "UNMEASURED"
    assert h["row_P5_R9"] == "UNMEASURED"
    assert h["l3_residency_loss_drill"] == "LOCKED"
    assert h["a2_a3_a4"] == "UNAPPROVED"
    assert h["unlocks"] == []
    assert stored["fingerprint"]["mode"] == "offline"
    assert stored["mode"] == "offline"
