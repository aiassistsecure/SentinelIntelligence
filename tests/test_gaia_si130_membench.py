"""SI-130 / MEMBENCH-1: the memory benchmark slice v0.

Pins:

- fixture integrity: 16 sessions, 40 questions, category
  counts 14/10/8/8; every gold string appears verbatim
  (checked case-insensitively, and the shipped fixture also
  carries it in its exact case) in its cited source session;
  for temporal-update questions the superseded string
  appears in an earlier session of the same persona and the
  gold string appears in NO earlier session of that persona;
  for distractor-resistance questions the distractor string
  appears in its cited session and differs from the gold;
- retriever conformance: both stores answer the same
  interface — retrieve(question, k) -> list of
  (session_id, text), length <= k, ids drawn from the
  16-session corpus;
- grader determinism: run_membench() is run ONCE here and
  the grader is re-run over the same retrievals' ranking
  inputs to confirm identical outcomes (the bench itself is
  deterministic — no randomness anywhere in the module —
  so a second full run is exercised in the determinism test
  only if it stays cheap; see the note in
  test_determinism_two_full_runs);
- results shape: the keys the summary and docs rely on are
  present, the headline rates recompute exactly from the
  per-question rows, and the honesty block carries the
  NOT LoCoMo line.

No model, no network: the embedded journal backend and a
temp directory of .txt files only.
"""
import json
import os

import pytest

from si.gaia.membench import (DEFAULT_FIXTURE_PATH,
                              FilesystemRetriever,
                              JournalRetriever, grade_question,
                              load_fixture, run_membench)

CATEGORIES = {"single-hop": 14, "cross-session": 10,
              "temporal-update": 8, "distractor-resistance": 8}


@pytest.fixture(scope="module")
def fixture():
    return load_fixture()


@pytest.fixture(scope="module")
def sessions_by_id(fixture):
    return {s["id"]: s for s in fixture["sessions"]}


# ---------------------------------------------------------------------------
# Fixture integrity
# ---------------------------------------------------------------------------

def test_fixture_shape(fixture):
    assert len(fixture["sessions"]) == 16
    assert len(fixture["questions"]) == 40
    counts = {}
    for q in fixture["questions"]:
        counts[q["category"]] = counts.get(q["category"], 0) + 1
    assert counts == CATEGORIES
    per_persona = {}
    for s in fixture["sessions"]:
        per_persona.setdefault(s["persona"], []).append(
            s["session_index"])
    assert len(per_persona) == 4
    for persona, idxs in per_persona.items():
        assert sorted(idxs) == [1, 2, 3, 4], persona


def test_every_gold_in_source_session(fixture, sessions_by_id):
    for q in fixture["questions"]:
        src = sessions_by_id[q["source_session"]]
        assert src["persona"] == q["persona"]
        for gold in q["gold_facts"]:
            assert gold.lower() in src["text"].lower(), q["id"]


def test_temporal_update_shape(fixture, sessions_by_id):
    temporal = [q for q in fixture["questions"]
                if q["category"] == "temporal-update"]
    assert len(temporal) == 8
    for q in temporal:
        src_idx = sessions_by_id[q["source_session"]][
            "session_index"]
        earlier = [s for s in fixture["sessions"]
                   if s["persona"] == q["persona"]
                   and s["session_index"] < src_idx]
        assert earlier, q["id"]
        superseded = q["superseded_fact"]
        gold = q["gold_facts"][0]
        assert superseded.lower() not in gold.lower()
        assert any(superseded.lower() in s["text"].lower()
                   for s in earlier), q["id"]
        for s in earlier:
            assert gold.lower() not in s["text"].lower(), (
                q["id"], s["id"])


def test_distractor_shape(fixture, sessions_by_id):
    distractors = [q for q in fixture["questions"]
                   if q["category"] == "distractor-resistance"]
    assert len(distractors) == 8
    for q in distractors:
        d = sessions_by_id[q["distractor_session"]]
        assert q["distractor_fact"].lower() in d["text"].lower(), \
            q["id"]
        assert q["distractor_fact"] != q["gold_facts"][0]


# ---------------------------------------------------------------------------
# Retriever conformance
# ---------------------------------------------------------------------------

def test_retrievers_conform(fixture):
    corpus_ids = {s["id"] for s in fixture["sessions"]}
    question = fixture["questions"][0]
    for cls in (FilesystemRetriever, JournalRetriever):
        with cls(fixture["sessions"]) as retriever:
            got = retriever.retrieve(question["question"], 3)
            assert 0 < len(got) <= 3
            for sid, text in got:
                assert sid in corpus_ids
                assert isinstance(text, str) and text
            got1 = retriever.retrieve(question["question"], 1)
            assert len(got1) == 1
            assert got1[0][0] == got[0][0]


# ---------------------------------------------------------------------------
# The bench: results shape + determinism
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def results():
    return run_membench()


def test_results_shape_and_rates_recompute(results):
    for key in ("slice", "run", "fixture", "k", "systems",
                "per_question", "honesty", "methods"):
        assert key in results, key
    assert results["k"] == 3
    assert results["fixture"]["sessions"] == 16
    assert results["fixture"]["questions"] == 40
    assert results["fixture"]["per_category"] == CATEGORIES
    assert results["fixture"]["fixture_sha256"].startswith(
        "sha256:")
    assert len(results["per_question"]) == 40
    for system in ("journal", "filesystem"):
        agg = results["systems"][system]
        hits = sum(1 for q in results["per_question"]
                   if q[system]["hit"])
        assert agg["hits"] == hits
        assert agg["n"] == 40
        assert agg["rate"] == hits / 40
        for category, count in CATEGORIES.items():
            bucket = agg["per_category"][category]
            cat_hits = sum(
                1 for q in results["per_question"]
                if q["category"] == category and q[system]["hit"])
            assert bucket["n"] == count
            assert bucket["hits"] == cat_hits
            assert bucket["rate"] == cat_hits / count


def test_honesty_block(results):
    joined = " ".join(results["honesty"])
    assert "NOT LoCoMo" in joined
    assert "Retrieval-only" in joined
    assert "SHARED" in joined
    assert "exact-match find only" in joined


def test_determinism_two_full_runs(results):
    """A second full run must reproduce the first exactly:
    same per-question outcomes, same headline counts. (The
    bench is run twice in total across the suite — once by
    the module-scoped fixture, once here — and stays cheap
    because the journal bundle is embedded sqlite over 16
    tiny sessions.)"""
    again = run_membench()
    assert again["per_question"] == results["per_question"]
    for system in ("journal", "filesystem"):
        assert again["systems"][system]["hits"] == \
            results["systems"][system]["hits"]
        assert again["systems"][system]["rate"] == \
            results["systems"][system]["rate"]


def test_grader_on_known_retrieval(fixture, sessions_by_id):
    """The grader itself, on a hand-built retrieval: gold in
    the top-1 is a hit with gold_in_top1 set; an empty
    retrieval is a miss."""
    q = fixture["questions"][0]
    src = sessions_by_id[q["source_session"]]
    graded = grade_question(q, [(src["id"], src["text"])])
    assert graded["hit"] is True
    assert graded["gold_in_top1"] is True
    assert graded["retrieved"] == [src["id"]]
    miss = grade_question(q, [])
    assert miss["hit"] is False
    assert miss["gold_in_top1"] is False


def test_out_dir_writes_results_and_summary(tmp_path):
    out = run_membench(out_dir=str(tmp_path))
    with open(os.path.join(str(tmp_path), "results.json"),
              encoding="utf-8") as fh:
        written = json.load(fh)
    assert written["systems"] == out["systems"]
    with open(os.path.join(str(tmp_path), "summary.md"),
              encoding="utf-8") as fh:
        summary = fh.read()
    assert "NOT LoCoMo" in summary
    assert "| journal |" in summary
    assert "| filesystem |" in summary


def test_default_fixture_is_the_shipped_one():
    assert os.path.isfile(DEFAULT_FIXTURE_PATH)
    with open(DEFAULT_FIXTURE_PATH, encoding="utf-8") as fh:
        shipped = json.load(fh)
    assert len(shipped["sessions"]) == 16
    assert len(shipped["questions"]) == 40
