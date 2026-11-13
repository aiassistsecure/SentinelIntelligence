"""SI-110 (SI-106 §5.1 / §12 item 2; SI-108 §4.2):
fold-supplied scoring support in the conformance harness.

OFFLINE / FIXTURE ONLY. The fold below is a fixture capture
journal journaled by the SI-109 driver from scripted
(replay-shaped) proposal sources on a fixture bundle; no live
model is contacted, no live capture is made, and nothing here
claims one. The acceptance tests are SI-108 §4.2's, numbered
to match:

1. fold-hash mismatch refuses scoring — no partial record,
   no rate, no fallback to no-fold scoring;
2. a scored document whose item carries no
   prompt_ref.task_seq is a corpus defect, excluded from M;
3. a foreign seq fails stage 4 exactly as a nonexistent seq
   does (SI-106 §13, first risk);
4. seq-form and sibling-form outcomes are priced/reported
   separately (§6); sibling count 0 is reported as 0;
5. no-fold scoring is unchanged — pinned here by refusing a
   no-fold corpus (SI-74's) under this support rather than
   re-scoring it; the unmodified C-1 tests pin the rest;
6. the scoring record states in itself that fold-supplied
   pricing is a different quantity from every no-fold
   cycle-0 rate — comparability none, either direction.
"""
import json

import pytest

from si.gaia import live as live_mod
from si.gaia.conformance import (
    FoldScoringRefused,
    build_conformance_record,
    build_fold_supplied_record,
    parse_task_seq,
    score_corpus_fold_supplied,
    si74_harness_corpus,
)
from si.gaia.multicycle import (
    MultiCycleDriver,
    corpus_canonical_sha256,
    fold_rows_and_hash,
)
from si.gate import Gate
from si.journal import open_journal


def _doc(intents, version=1, signal=None):
    return json.dumps({"proposal": {
        "version": version, "intents": intents,
        "task_signal": signal}})


def _intent(reason, depends_on=None, tool="fs_list"):
    return {"tool": tool, "args": {"path": "."},
            "reason": reason,
            "depends_on": depends_on or [],
            "ordering_key": None}


class _Guard:
    def __init__(self, states):
        self.states = list(states)

    def __call__(self, journal):
        if len(self.states) > 1:
            return self.states.pop(0)
        return self.states[0]


def _citing_source(perception, _n):
    """A distinct parse-ok body per task/cycle; at cycles >= 1
    it cites the first citable seq of its own perception."""
    sections = perception.get("sections") or {}
    prior = sections.get("prior_cycles") or []
    citable = (sections.get("citable_reference_set") or {}).get(
        "seqs") or []
    deps = [citable[0]] if citable else []
    return _doc([_intent(
        reason=(f"task {perception.get('task_seq')} cycle with "
                f"{len(prior)} prior intent(s)"),
        depends_on=deps)])


class _Source:
    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, perception):
        self.calls += 1
        return self.fn(perception, self.calls)


@pytest.fixture()
def fold_capture(tmp_path):
    """Two task runs journaled by the SI-109 driver (K=1) on a
    fixture bundle. Yields (journal, tasks) where tasks maps
    run index -> {"task_seq", "intent_seq"} for the intent
    each run's cycle 0 journaled and its cycle 1 could cite."""
    paths = live_mod.build_fixture_bundle(
        str(tmp_path / "capture-bundle"), str(tmp_path / "work"))
    policy = live_mod.fixture_policy(paths["work_root"])
    journal = open_journal(paths["bundle_dir"])
    from si.gaia.policy import resolve_tool_policy
    tool_pol = resolve_tool_policy(policy)
    budgets = {
        "max_tool_cycles_per_task": int(
            tool_pol["max_tool_cycles_per_task"]),
        "tool_max_task_result_bytes": int(
            tool_pol["tool_max_task_result_bytes"]),
    }
    driver = MultiCycleDriver(
        journal, Gate(policy), bundle_dir=paths["bundle_dir"],
        work_root=paths["work_root"], budgets=budgets,
        cycle_cap=budgets["max_tool_cycles_per_task"],
        proposal_source=_Source(_citing_source),
        residency_guard=_Guard(["Resident"]), K=1)
    tasks_in = [
        {"workload": "benign-read", "run": 0,
         "instruction": "list the work root"},
        {"workload": "benign-read", "run": 1,
         "instruction": "list the work root again"},
    ]
    state = driver.run(tasks_in)
    tasks = {}
    for doc in state["documents"]:
        if doc["cycle"] == 1:
            citable = (doc["perception"] or {}).get(
                "citable_seqs") or []
            tasks[doc["run"]] = {
                "task_seq": doc["task_seq"],
                "intent_seq": citable[0] if citable else None,
            }
    assert set(tasks) == {0, 1}
    assert all(t["intent_seq"] is not None for t in tasks.values())
    yield journal, tasks
    journal.close()


def _item(item_id, body, task_seq_marker, cycle=1):
    """One scored corpus item. task_seq_marker is the task seq
    to name at prompt_ref.task_seq, or None to omit it (the
    §4.2(2) corpus-defect shape)."""
    prompt_ref = {"request_digest": f"req-{item_id}"}
    if task_seq_marker is not None:
        prompt_ref["task_seq"] = task_seq_marker
    return {"item_id": item_id, "workload": "benign-read",
            "cycle": cycle, "body": body,
            "body_source": "verbatim-recorded",
            "prompt_ref": prompt_ref,
            "parse_task_seq": task_seq_marker,
            "provenance": {"capture": "si110-test-fixture",
                           "item": item_id}}


def _corpus(journal, items):
    corpus = {
        "corpus_id": "si110-test-fold-supplied",
        "purpose": "candidate-readiness",
        "candidate_label": "fixture-candidate",
        "response_origin": "candidate-recorded",
        "capture_date": "2026-11-13",
        "fixed_n": len([i for i in items
                        if not i.get("transport_failed")]),
        "coverage": {"workloads": ["benign-read"],
                     "cycles_covered": [1],
                     "signals_elicited": True},
        "items": items,
    }
    _rows, fold_hash, _counts, _tip = fold_rows_and_hash(journal)
    corpus["fold"] = {
        "fold_sha256": fold_hash,
        "corpus_sha256": corpus_canonical_sha256(corpus),
        "fold_hash_procedure": (
            "sha256 over si.bundle.canonical_json of the "
            "frozen journal's rows (SI-109 freeze shape)"),
    }
    return corpus


def _full_items(tasks):
    a, b = tasks[0], tasks[1]
    return [
        _item("pass-seq", _doc([_intent(
            "cite my own prior intent",
            depends_on=[a["intent_seq"]])]), a["task_seq"]),
        _item("foreign-seq", _doc([_intent(
            "cite another task's intent",
            depends_on=[{"seq": b["intent_seq"],
                         "requirement": "outcome"}])]),
            a["task_seq"]),
        _item("nonexistent-seq", _doc([_intent(
            "cite a seq that does not exist",
            depends_on=[{"seq": 999999,
                         "requirement": "outcome"}])]),
            a["task_seq"]),
        _item("missing-task-seq", _doc([_intent(
            "a body with no task to score against")]), None),
        _item("sibling-form", _doc([
            _intent("first intent of the batch"),
            _intent("second intent citing its sibling",
                    depends_on=[{"sibling": 0,
                                 "requirement": "outcome"}]),
        ], version=2), a["task_seq"]),
        _item("mixed-forms", _doc([
            _intent("first intent of the mixed batch"),
            _intent("second intent citing seq and sibling",
                    depends_on=[
                        {"seq": a["intent_seq"],
                         "requirement": "outcome"},
                        {"sibling": 0,
                         "requirement": "outcome"}]),
        ], version=2), a["task_seq"]),
    ]


# -- §4.2(2)/(3)/(4): the scoring procedure itself ------------------

def test_fold_supplied_scoring_procedure(fold_capture):
    journal, tasks = fold_capture
    corpus = _corpus(journal, _full_items(tasks))
    fold, per_item, defects, evidence = score_corpus_fold_supplied(
        corpus, journal=journal)
    # (2) The task-less item is a corpus defect, excluded
    # from M: M = 5, never 6, and it was never scored.
    assert fold["n"] == 5
    assert [d["corpus_item_id"] for d in defects] == [
        "missing-task-seq"]
    assert defects[0]["defect"] == "missing-prompt-ref-task-seq"
    assert "missing-task-seq" not in {
        p["corpus_item_id"] for p in per_item}
    by_id = {p["corpus_item_id"]: p for p in per_item}
    # §5.1 step 5's point: a citation of a real prior intent
    # of this task PASSES stage 4 with the fold supplied.
    assert by_id["pass-seq"]["parse_ok"] is True
    # (3) A foreign seq — an intent row of another task in
    # the fold — fails stage 4 exactly as a nonexistent seq
    # does: same failing stage, not parse-ok.
    assert by_id["foreign-seq"]["parse_ok"] is False
    assert by_id["foreign-seq"]["failing_stage"] == 4
    assert by_id["nonexistent-seq"]["parse_ok"] is False
    assert by_id["nonexistent-seq"]["failing_stage"] == 4
    assert by_id["sibling-form"]["parse_ok"] is True
    assert by_id["mixed-forms"]["parse_ok"] is True
    assert fold["parse_ok"] == 3
    # (4) Forms priced and reported separately.
    forms = evidence["reference_forms"]["by_form"]
    assert forms["seq"]["occurrences"] == 4
    assert forms["seq"]["passed"] == 2
    assert forms["seq"]["refused"] == 2
    assert forms["sibling"]["occurrences"] == 2
    assert forms["sibling"]["passed"] == 2
    assert forms["sibling"]["refused"] == 0
    assert forms["other"]["occurrences"] == 0
    # Step 4(c): both hashes recorded — the fold's, and each
    # scored item's mapped task seq with its hash.
    _rows, fold_hash, _c, _t = fold_rows_and_hash(journal)
    assert evidence["fold_sha256"] == fold_hash
    assert evidence["corpus_hash"].startswith("sha256:")
    for entry in per_item:
        assert entry["task_seq"] == tasks[0]["task_seq"]
        assert entry["task_seq_hash"].startswith("sha256:")
        assert entry["fold_sha256"] == fold_hash


def test_sibling_count_zero_reported_as_zero(fold_capture):
    journal, tasks = fold_capture
    a = tasks[0]
    items = [_item("pass-seq", _doc([_intent(
        "cite my own prior intent",
        depends_on=[a["intent_seq"]])]), a["task_seq"])]
    corpus = _corpus(journal, items)
    _fold, _per, _defects, evidence = score_corpus_fold_supplied(
        corpus, journal=journal)
    forms = evidence["reference_forms"]["by_form"]
    # (4) Sibling 0 is reported as 0 on its own line, never
    # folded into the seq-form line.
    assert forms["sibling"]["occurrences"] == 0
    assert forms["sibling"]["passed"] == 0
    assert forms["seq"]["occurrences"] == 1
    assert forms["seq"]["passed"] == 1


# -- §4.2(1): fold-hash mismatch refuses, fail-closed ----------------

def test_fold_hash_mismatch_refuses_scoring(fold_capture):
    journal, tasks = fold_capture
    corpus = _corpus(journal, _full_items(tasks))
    corpus["fold"]["fold_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(FoldScoringRefused):
        score_corpus_fold_supplied(corpus, journal=journal)
    record = build_fold_supplied_record(
        corpus, journal=journal, argv=["test"])
    assert record["status"] == "SCORING REFUSED"
    assert record["results"] is None
    # No rate is published anywhere in a refused record, and
    # there is no fallback: the record is not a scored one.
    assert '"rate"' not in json.dumps(record)
    assert record["readiness"] == {"threshold": None,
                                   "unlocks": []}
    assert record["live_attempts"] == 0


def test_caller_stated_fold_hash_mismatch_refuses(fold_capture):
    journal, tasks = fold_capture
    corpus = _corpus(journal, _full_items(tasks))
    with pytest.raises(FoldScoringRefused):
        score_corpus_fold_supplied(
            corpus, journal=journal,
            fold_sha256="sha256:" + "f" * 64)


def test_no_fold_supplied_refuses_no_fallback(fold_capture):
    journal, tasks = fold_capture
    corpus = _corpus(journal, _full_items(tasks))
    with pytest.raises(FoldScoringRefused):
        score_corpus_fold_supplied(corpus, journal=None)


# -- §4.2(5): no-fold corpora are never re-scored here ---------------

def test_no_fold_corpus_refused_under_fold_support(fold_capture):
    journal, _tasks = fold_capture
    corpus = si74_harness_corpus()
    assert "fold" not in corpus and "fold_sha256" not in corpus
    with pytest.raises(FoldScoringRefused):
        score_corpus_fold_supplied(corpus, journal=journal)
    record = build_fold_supplied_record(
        corpus, journal=journal, argv=["test"])
    assert record["status"] == "SCORING REFUSED"
    assert record["results"] is None
    # The same corpus still scores through the untouched
    # no-fold path exactly as before (SI-74 known outcomes).
    no_fold = build_conformance_record(corpus, argv=["test"])
    assert no_fold["status"] == "scored"
    assert no_fold["results"]["n"] == 6
    assert no_fold["results"]["parse_ok"] == 2


# -- §4.2(6): the record states the comparability in itself ----------

def test_record_states_comparability_and_labels(fold_capture):
    journal, tasks = fold_capture
    corpus = _corpus(journal, _full_items(tasks))
    record = build_fold_supplied_record(
        corpus, journal=journal, argv=["test"])
    assert record["status"] == "scored"
    assert record["results"]["n"] == 5  # M excludes the defect
    assert record["results"]["corpus_defects"][0][
        "corpus_item_id"] == "missing-task-seq"
    section = record["fold_supplied"]
    _rows, fold_hash, _c, _t = fold_rows_and_hash(journal)
    assert section["fold_sha256"] == fold_hash
    assert record["fingerprint"]["fold_sha256"] == fold_hash
    assert record["fingerprint"]["corpus_hash"].startswith(
        "sha256:")
    statement = section["comparability"]
    assert "DIFFERENT QUANTITY" in statement
    assert "comparability none, in either direction" in statement
    assert statement in record["honesty"]["fold_supplied_pricing"]
    # Honesty labels intact, verdict-free.
    honesty = record["honesty"]
    assert honesty["section_18_3_claim"] == "NOT earned"
    assert honesty["l3_residency_loss_drill"] == "LOCKED"
    assert honesty["a2_a3_a4"] == "UNAPPROVED"
    assert record["readiness"] == {"threshold": None,
                                   "unlocks": []}
    assert record["live_attempts"] == 0
    assert "none" in section["verdict"]


# -- the §5.1 step-4 mapping, as a unit -------------------------------

def test_parse_task_seq_mapping():
    # prompt_ref.task_seq is the corpus-named location and wins.
    assert parse_task_seq(
        {"prompt_ref": {"task_seq": 7},
         "parse_task_seq": 9}) == 7
    # The legacy top-level field is the fallback only.
    assert parse_task_seq({"parse_task_seq": 9}) == 9
    assert parse_task_seq(
        {"prompt_ref": {"task_seq": 7}}) == 7
    # Absent or malformed maps to None — a defect, never a guess.
    assert parse_task_seq({"prompt_ref": {}}) is None
    assert parse_task_seq({}) is None
    assert parse_task_seq(
        {"prompt_ref": {"task_seq": "7"}}) is None
    assert parse_task_seq(
        {"prompt_ref": {"task_seq": True}}) is None
    assert parse_task_seq(
        {"prompt_ref": {"task_seq": None},
         "parse_task_seq": 9}) is None
