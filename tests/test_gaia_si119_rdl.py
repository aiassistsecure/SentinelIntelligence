"""SI-119 / RDL-1 (SI-111 §2): the reference-discipline
linter over candidate corpora.

Pins:

- every §2.2 rule fires on its recorded shape, carrying
  the §9 parser / SI-95 / SI-101 vocabulary name the RFC
  assigns it — seq <= 0, a seq outside the citable set, a
  tool name in ``depends_on``, sibling self-reference,
  sibling forward reference (labelled unexercised),
  malformed entries, single-owner argument bleed,
  missing required arguments, and a prose prefix before
  the §8 document;
- the bleed scoping is the recorded one: an argument
  declared on zero or on several registry tools is NOT
  bleed, and the declared-on tool is read from the live
  registry (computed from ``REGISTRY`` here, never a
  constant copied from the linter);
- correct forms are counted only when priced: a seq
  citation counts correct only inside a supplied citable
  reference set (A1 shape, per-row or run-wide); without
  a set it is form-only and reported ungraded, never
  correct, never flagged;
- the eligibility report's absent-references line is its
  own line: rows declaring ``expects_dependency`` whose
  document carries no ``depends_on`` entry are counted
  there and nowhere else — the rows themselves stay
  clean, and a corpus declaring no expectation reports
  the line as 0 by construction;
- the §2.4 hard label travels verbatim in the report
  record and in both renderings, and bodies are never
  duplicated: verdicts cite row id, location, cited
  value, and the target's hash only;
- the accepted corpus shapes (items list, bare list,
  JSONL; target / body keys; dict targets) load, and an
  unlintable row fails the load closed;
- both CLI surfaces (``si gaia-lint`` and the module
  parser) run the lint; a dirty corpus is a report
  (exit 0), an unreadable corpus is a refusal (exit 1).

Fixture corpora only: offline work is never live
evidence, earns no §18.1 row and no §18.3 progress
(SI-111 §10). No trainer is called, no NEDB is touched.
"""
import json

import pytest

from si.cli import build_parser as build_si_parser
from si.gaia import ref_lint
from si.gaia.ref_lint import HARD_LABEL, lint_corpus, load_corpus
from si.gaia.registry import REGISTRY

HARD_LABEL_VERBATIM = (
    "RDL-1 grades data, never live conformance. "
    "No conformance claim may cite lint results.")


def _doc(intents, version=2):
    return json.dumps({"proposal": {
        "version": version, "intents": intents,
        "task_signal": None}})


def _intent(tool, args, depends_on=None):
    return {"tool": tool, "args": args,
            "reason": "fixture step",
            "depends_on": depends_on or [],
            "ordering_key": None}


READ = _intent("fs_read", {"path": "notes/a.md"})


def _corpus(rows):
    return {"corpus_id": "fixture-rdl-1", "corpus_sha256": None,
            "rows": [{"item": item, "target": item["target"]}
                     for item in rows]}


def _row(item_id, target, **extra):
    item = {"item_id": item_id, "target": target}
    item.update(extra)
    return item


def _rules(row):
    return [v["rule"] for v in row["verdicts"]]


# ---------------------------------------------------------------------------
# The nine rules, on their recorded shapes
# ---------------------------------------------------------------------------

def test_all_nine_rules_fire_on_their_recorded_shapes():
    rows = [
        _row("nonpositive", _doc([dict(
            READ, depends_on=[0, {"seq": -3}])])),
        _row("not-intent", _doc([dict(
            READ, depends_on=[{"seq": 5}])]),
            citable_reference_set={"seqs": [11], "empty": False}),
        _row("tool-name", _doc([dict(
            READ, depends_on=["fs_read"])])),
        _row("sibling-self", _doc([dict(
            READ, depends_on=[{"sibling": 0}])])),
        _row("sibling-forward", _doc([
            dict(READ, depends_on=[{"sibling": 1}]), READ])),
        _row("malformed", _doc([dict(READ, depends_on=[
            {"requirement": "outcome"}, 2.5, ["seq", 5]])])),
        _row("bleed-and-missing", _doc([
            _intent("fs_search",
                    {"path": "notes", "max_bytes": 10}),
            _intent("fs_edit", {"path": "notes/a.md",
                                "old": "a", "content": "x"})])),
        _row("prose-prefix",
             "Version 2\n" + _doc([READ])),
    ]
    report = lint_corpus(_corpus(rows))
    for rule in ref_lint.RULE_IDS:
        assert report["rule_row_counts"][rule] >= 1, rule
    by_id = {r["item_id"]: r for r in report["rows"]}
    assert _rules(by_id["nonpositive"]) == [
        "REF-SEQ-NONPOSITIVE", "REF-SEQ-NONPOSITIVE"]
    assert by_id["nonpositive"]["verdicts"][0][
        "taxonomy_name"] == "seq_nonpositive"
    assert by_id["nonpositive"]["verdicts"][0][
        "cited_value"] == 0
    assert _rules(by_id["not-intent"]) == ["REF-SEQ-NOT-INTENT"]
    assert by_id["not-intent"]["verdicts"][0][
        "taxonomy_name"] == "seq_not_intent_row"
    assert _rules(by_id["tool-name"]) == ["REF-TOOL-NAME"]
    assert by_id["tool-name"]["verdicts"][0][
        "taxonomy_name"] == \
        "depends_on_tool_name_not_journal_seq"
    assert _rules(by_id["sibling-self"]) == ["REF-SIBLING-SELF"]
    assert by_id["sibling-self"]["verdicts"][0][
        "taxonomy_name"] == "sibling_self_reference"
    fwd = by_id["sibling-forward"]["verdicts"][0]
    assert fwd["rule"] == "REF-SIBLING-FORWARD"
    assert fwd["taxonomy_name"] == "sibling_forward_reference"
    assert "unexercised" in fwd["detail"]
    assert _rules(by_id["malformed"]) == [
        "REF-ENTRY-MALFORMED"] * 3
    assert by_id["malformed"]["verdicts"][0][
        "taxonomy_name"] == "entry_malformed"
    bleed = by_id["bleed-and-missing"]
    assert _rules(bleed) == [
        "ARG-BLEED", "ARG-MISSING-REQUIRED",
        "ARG-BLEED", "ARG-MISSING-REQUIRED"]
    assert bleed["verdicts"][0]["arg"] == "max_bytes"
    assert bleed["verdicts"][0]["declared_on"] == "fs_read"
    assert bleed["verdicts"][1]["arg"] == "pattern"
    assert bleed["verdicts"][2]["arg"] == "content"
    assert bleed["verdicts"][2]["declared_on"] == "fs_write"
    assert bleed["verdicts"][3]["arg"] == "new"
    prose = by_id["prose-prefix"]
    assert _rules(prose) == ["SHAPE-PROSE-PREFIX"]
    assert prose["verdicts"][0]["taxonomy_name"] == \
        "version_narration_prefix"
    assert prose["document"] == "prose_prefixed"


def test_bleed_prices_against_the_live_registry():
    """The declared-on tool and the exactly-one-owner
    scoping are read from REGISTRY as data: recompute the
    expectation from the registry itself, and pin that an
    argument declared on several tools (or none) is not
    bleed under §2.2's definition."""
    owners = {}
    for name, spec in REGISTRY.items():
        for arg in (spec.args_schema.get("properties") or {}):
            owners.setdefault(arg, []).append(name)
    assert owners["max_bytes"] == ["fs_read"]
    assert len(owners["timeout_seconds"]) >= 2
    rows = [_row("scoped", _doc([_intent("fs_read", {
        "path": "notes/a.md",
        "timeout_seconds": 5,      # declared on >= 2 tools
        "frobnicate": 1,           # declared on no tool
    })]))]
    report = lint_corpus(_corpus(rows))
    assert report["rows"][0]["verdicts"] == []
    assert report["rows_clean"] == 1


# ---------------------------------------------------------------------------
# Correct forms, citability, and the absent-references line
# ---------------------------------------------------------------------------

def test_correct_forms_counted_only_when_priced():
    rows = [
        _row("seq-correct", _doc([dict(
            READ, depends_on=[{"seq": 11}])]),
            citable_reference_set={"seqs": [11, 12],
                                   "empty": False}),
        _row("sibling-correct", _doc([
            READ, dict(READ, depends_on=[{"sibling": 0}])])),
        _row("seq-ungraded", _doc([dict(
            READ, depends_on=[{"seq": 42}])])),
    ]
    report = lint_corpus(_corpus(rows))
    assert report["rows_clean"] == 3
    assert report["correct_form_row_counts"] == {
        "seq_citation_in_citable_set": 1,
        "sibling_citation_strictly_earlier_non_self": 1}
    assert report["correct_form_entry_counts"][
        "seq_citation_in_citable_set"] == 1
    assert report["seq_form_ungraded_entries"] == 1


def test_run_wide_reference_set_and_row_set_precedence():
    rows = [_row("both", _doc([dict(
        READ, depends_on=[{"seq": 7}])]),
        citable_reference_set={"seqs": [7], "empty": False})]
    corpus = _corpus(rows)
    # The row's own set prices seq 7 correct even though
    # the run-wide set does not contain it.
    report = lint_corpus(corpus, reference_set={"seqs": [9]})
    assert report["rows_clean"] == 1
    report = lint_corpus(_corpus([_row("wide", _doc([dict(
        READ, depends_on=[{"seq": 9}])]))]),
        reference_set={"seqs": [9], "empty": False})
    assert report["rows_clean"] == 1
    assert report["reference_set_supplied"] is True


def test_absent_references_line_is_its_own_line():
    rows = [
        _row("absent", _doc([READ]), expects_dependency=True),
        # 'present' declares the expectation and carries a
        # (correct) sibling citation: not absent.
        _row("present", _doc([
            READ, dict(READ, depends_on=[{"sibling": 0}])]),
            expects_dependency=True),
        _row("undeclared", _doc([READ])),
    ]
    report = lint_corpus(_corpus(rows))
    assert report["absent_references_rows"] == 1
    by_id = {r["item_id"]: r for r in report["rows"]}
    assert by_id["absent"][
        "references_absent_where_expected"] is True
    assert by_id["absent"]["verdicts"] == []  # stays clean
    assert "never folded into a pass figure" in \
        report["absent_references_note"]
    bare = lint_corpus(_corpus([_row("x", _doc([READ]))]))
    assert bare["absent_references_rows"] == 0
    assert "by construction" in bare["absent_references_note"]


# ---------------------------------------------------------------------------
# Labels, citation discipline, corpus shapes
# ---------------------------------------------------------------------------

def test_hard_label_verbatim_and_bodies_never_duplicated():
    assert HARD_LABEL == HARD_LABEL_VERBATIM
    secret_shape = "UNIQUE-BODY-MARKER-119"
    rows = [_row("dirty", "Version 2\n" + _doc([dict(
        READ, depends_on=[0])]).replace(
            "fixture step", secret_shape))]
    report = lint_corpus(_corpus(rows))
    assert report["label"] == HARD_LABEL_VERBATIM
    blob = json.dumps(report)
    assert secret_shape not in blob
    assert "Version 2" not in blob
    row = report["rows"][0]
    assert row["target_sha256"].startswith("sha256:")
    text = ref_lint.render_text(report, corpus_path="c.json")
    assert secret_shape not in text
    assert "absent references where the task shape" in text


def test_corpus_shapes_load_and_unlintable_row_fails_closed(
        tmp_path):
    items_file = tmp_path / "items.json"
    items_file.write_text(json.dumps({
        "corpus_id": "shape-items",
        "items": [{"item_id": "a", "body": _doc([READ])}]}))
    corpus = load_corpus(str(items_file))
    assert corpus["corpus_id"] == "shape-items"
    assert lint_corpus(corpus)["rows_clean"] == 1

    jsonl_file = tmp_path / "rows.jsonl"
    jsonl_file.write_text(
        json.dumps({"target": {"proposal": {
            "version": 2, "intents": [READ],
            "task_signal": None}}}) + "\n")
    corpus = load_corpus(str(jsonl_file))
    report = lint_corpus(corpus)
    assert report["rows_total"] == 1
    assert report["rows_clean"] == 1  # dict target, parsed

    bad_file = tmp_path / "bad.json"
    bad_file.write_text(json.dumps(
        {"items": [{"item_id": "no-target"}]}))
    with pytest.raises(ValueError):
        load_corpus(str(bad_file))


def test_unparseable_target_is_counted_not_cleaned():
    rows = [_row("fragment", "Version 2\n")]
    report = lint_corpus(_corpus(rows))
    assert report["rows_unparseable"] == 1
    assert report["rows_clean"] == 0
    assert _rules(report["rows"][0]) == ["SHAPE-PROSE-PREFIX"]


# ---------------------------------------------------------------------------
# CLI surfaces
# ---------------------------------------------------------------------------

def test_both_cli_surfaces(tmp_path, capsys):
    corpus_file = tmp_path / "corpus.json"
    corpus_file.write_text(json.dumps({"items": [
        {"item_id": "ok", "target": _doc([READ])},
        {"item_id": "bad", "target": _doc([dict(
            READ, depends_on=[0])])}]}))
    parser = build_si_parser()
    args = parser.parse_args(
        ["gaia-lint", str(corpus_file), "--json"])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert HARD_LABEL_VERBATIM in out
    report = json.loads(out[out.index("{"):])
    assert report["record"] == "si.gaia.ref_lint.report"
    assert report["rule_row_counts"][
        "REF-SEQ-NONPOSITIVE"] == 1

    assert ref_lint.main([str(corpus_file)]) == 0
    out = capsys.readouterr().out
    assert HARD_LABEL_VERBATIM in out
    assert "REF-SEQ-NONPOSITIVE" in out

    assert ref_lint.main(
        [str(tmp_path / "missing.json")]) == 1
    capsys.readouterr()
