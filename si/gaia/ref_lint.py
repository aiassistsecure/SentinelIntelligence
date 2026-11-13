"""Gaia RDL-1: the reference-discipline linter (SI-111 §2;
slice SI-119).

An **offline** linter over candidate corpora. It grades the
*data* a future candidate would be trained on — before any
training — against the recorded failure classes of SI-95,
SI-101, and SI-107: the reference forms and argument shapes
the graded measurements priced as missing. Its rules are
exactly SI-111 §2.2's nine; no rule is invented from
intuition, and the one class no recorded corpus exercised
(``REF-SIBLING-FORWARD``) is labelled unexercised wherever
it is named.

> **RDL-1 grades data, never live conformance.** No
> conformance claim may cite lint results. A lint-clean
> corpus is not evidence about any model; a lint-dirty
> corpus is not a verdict on one. The 24/30 bar falls only
> to a **trained candidate re-measured under D-C1** — fresh
> corpus, C-1 harness, D-C1 as decided — which is a
> training-side event in Mark's lane. RDL-1's entire claim
> is: *the data stopped teaching the recorded mistakes*,
> stated per rule, over a named corpus, by hash.
> (SI-111 §2.4, verbatim.)

Scope, stated in the design's own terms (§2.5): RDL-1 is
not a harness change (no live proposal is linted by it),
not trainer wiring (it reads corpus files and emits a
report — **no trainer is called, no training journal is
written, no live NEDB is touched**), and not a conformance
instrument (§2.4 binds its output). It lands in this repo
as offline corpus tooling under lead decision D-111-1
(SI-111 §7, resolved in the SI-119 slice doc); D-90-2 and
D-90-3 remain **DEFERRED** and are activated by nothing
here (D-111-6).

Inputs (§2.3):

- a candidate corpus in the C-2 corpus shape — prompt /
  target pairs, or target documents alone (see
  :func:`load_corpus` for the accepted file shapes);
- the tool registry **as data**: argument declarations are
  read from :data:`si.gaia.registry.REGISTRY` at lint time,
  never copied into this module, so the bleed rules price
  against the same surface the gate prices;
- optionally, a citable reference set in the A1
  ``citable_reference_set`` shape (SI-103 §4 / SI-104),
  against which seq-form citability — not merely seq
  form — is graded. Without a set, a positive seq citation
  is reported as *form-only, citability ungraded*: the
  linter never calls an unpriced citation correct.

Outputs (§2.3): per-row verdicts — for every flagged row
the rule id, the location (intent index, entry index), and
the cited value; rows are cited by corpus id and position
and by the target's hash, **bodies never duplicated**
(never duplicate bytes) — plus a corpus eligibility
report: the count of rows exhibiting each rule, the count
of rows clean, the count of rows exhibiting each *correct*
form, and — stated as its own line, never folded into a
pass figure — the count of rows whose references are
absent where the corpus's own task shape calls for a
dependency.

Vocabulary discipline (§2.3): verdicts reuse the §9
parser's stage-4 refusal names (``seq_nonpositive``,
``seq_not_intent_row``, ``sibling_self_reference``,
``entry_malformed``, and the parser's sibling detail names)
and SI-95/SI-101's taxonomy class names rather than
inventing synonyms. This module modifies neither the
parser, the gate, the registry, nor the contract text.

Usage (also wired into the main CLI as ``si gaia-lint``)::

    si gaia-lint <corpus.json> [--reference-set FILE] [--json]

CPU only; no network; no secrets. Machine learning
throughout; fixture corpora earn no §18.1 row and no
§18.3 progress (SI-111 §10).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

from si.gaia.registry import REGISTRY

#: The §2.4 hard label, verbatim. It travels with every
#: report this module emits, in both renderings.
HARD_LABEL: str = (
    "RDL-1 grades data, never live conformance. "
    "No conformance claim may cite lint results.")

#: Record name carried by every report.
REPORT_RECORD = "si.gaia.ref_lint.report"

#: The nine §2.2 rules, in the RFC's order. ``taxonomy``
#: is the recorded parser / taxonomy name the verdict
#: carries (None where the recorded classes are the
#: per-combination genus names of SI-101, which this
#: module does not reconstruct — the verdict's detail
#: fields carry the combination instead). ``evidence``
#: is the recorded source, cited not copied.
RULES: tuple = (
    {"rule": "REF-SEQ-NONPOSITIVE",
     "taxonomy": "seq_nonpositive",
     "evidence": "SI-95: seq 0 x1 (SI-94); SI-101: seq 0 x7 "
                 "across five workloads (SI-100); SI-107 §7: "
                 "3 x seq 0, priced-and-refused"},
    {"rule": "REF-SEQ-NOT-INTENT",
     "taxonomy": "seq_not_intent_row",
     "evidence": "SI-95: own task seq {5} x1; checkpoints-row "
                 "seq 1 x3 (SI-94); SI-107 §7: seq 181, the "
                 "run's own task row, x1"},
    {"rule": "REF-TOOL-NAME",
     "taxonomy": "depends_on_tool_name_not_journal_seq",
     "evidence": "SI-95: x1 item, a three-intent chain "
                 "(SI-94); the genus is SI-81's class from "
                 "SI-79"},
    {"rule": "REF-SIBLING-SELF",
     "taxonomy": "sibling_self_reference",
     "evidence": "SI-101 sibling audit: the corpus's only "
                 "two sibling entries, both {sibling: 0} "
                 "self-references at intents[0]; correct "
                 "sibling use zero (SI-100)"},
    {"rule": "REF-SIBLING-FORWARD",
     "taxonomy": "sibling_forward_reference",
     "evidence": "Defined by SI-97's positional rule "
                 "(strictly-earlier only) and carried in "
                 "SI-101's class vocabulary; exercised 0 "
                 "times in any recorded corpus — included "
                 "because the form's spec makes it "
                 "decidable, labelled unexercised"},
    {"rule": "REF-ENTRY-MALFORMED",
     "taxonomy": "entry_malformed",
     "evidence": "SI-107 §7: 1 occurrence, priced-and-"
                 "refused at stage 4"},
    {"rule": "ARG-BLEED",
     "taxonomy": None,
     "evidence": "SI-101: undeclared argument x9 — "
                 "max_bytes (declared only on fs_read) x7, "
                 "content (declared only on fs_write) x2"},
    {"rule": "ARG-MISSING-REQUIRED",
     "taxonomy": None,
     "evidence": "SI-101: fs_search missing required "
                 "pattern x3; fs_edit missing required "
                 "new x2"},
    {"rule": "SHAPE-PROSE-PREFIX",
     "taxonomy": "version_narration_prefix",
     "evidence": "SI-101: version_narration_prefix — bodies "
                 "opening with a 'Version 2' narration line "
                 "(SI-100)"},
)

RULE_IDS: tuple = tuple(r["rule"] for r in RULES)
_TAXONOMY = {r["rule"]: r["taxonomy"] for r in RULES}

#: Target-bearing keys of a corpus row, in read order: a
#: prompt/target pair carries its document under ``target``;
#: a capture-shaped row (docs/SI-94-corpus.json) under
#: ``body``; a replay-shaped row under ``response``.
_TARGET_KEYS = ("target", "body", "response", "completion",
                "document")


# ---------------------------------------------------------------------------
# Registry, read as data (§2.3 input ii)
# ---------------------------------------------------------------------------

def _registry_arg_owners() -> dict:
    """arg name -> sorted tool names declaring it, read
    from the live registry on every call — never copied
    into this module, so the bleed rules always price
    against the registry as it stands."""
    owners: dict = {}
    for name, spec in REGISTRY.items():
        props = (spec.args_schema or {}).get("properties") or {}
        for arg in props:
            owners.setdefault(arg, []).append(name)
    return {arg: sorted(names) for arg, names in owners.items()}


def _registry_required() -> dict:
    """tool name -> its declared required args, from the
    live registry."""
    return {
        name: list((spec.args_schema or {}).get("required") or [])
        for name, spec in REGISTRY.items()}


# ---------------------------------------------------------------------------
# Corpus loading (C-2 shape: prompt/target pairs or targets alone)
# ---------------------------------------------------------------------------

def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(
        text.encode("utf-8")).hexdigest()


def _target_hash(target) -> str:
    if isinstance(target, str):
        return _sha256_text(target)
    return _sha256_text(json.dumps(
        target, sort_keys=True, default=str))


def load_corpus(path: str) -> dict:
    """Read a candidate corpus file into lint rows.

    Accepted shapes: a JSON object with an ``items`` list
    (the C-2 capture shape), a bare JSON list of rows, or
    JSONL (one row object per line). Each row's target
    document is read from the first present of
    :data:`_TARGET_KEYS`; a row with no target key is a
    load error (fail closed — an unlintable row is never
    silently clean). The corpus is cited by id and by the
    file's hash; bodies are held in memory for this run
    only and never copied into the report.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    corpus_sha = "sha256:" + hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8")
    corpus_id = None
    items = None
    try:
        doc = json.loads(text)
    except ValueError:
        doc = None
    if isinstance(doc, dict) and isinstance(doc.get("items"), list):
        corpus_id = doc.get("corpus_id")
        items = doc["items"]
    elif isinstance(doc, list):
        items = doc
    elif isinstance(doc, dict) and any(
            key in doc for key in _TARGET_KEYS):
        items = [doc]
    else:
        items = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                items.append(json.loads(line))
    rows = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(
                f"corpus row {idx} is not an object — the "
                f"C-2 corpus shape is rows of prompt/target "
                f"pairs or target documents (fail closed)")
        target = None
        for key in _TARGET_KEYS:
            if key in item:
                target = item[key]
                break
        else:
            raise ValueError(
                f"corpus row {idx} carries no target under "
                f"any of {_TARGET_KEYS} — an unlintable row "
                f"is never silently clean (fail closed)")
        rows.append({"item": item, "target": target})
    return {"corpus_id": corpus_id, "corpus_sha256": corpus_sha,
            "rows": rows}


def _row_id(item: dict, idx: int):
    for key in ("item_id", "id", "name"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _row_expects_dependency(item: dict) -> bool:
    """Whether the corpus's own task shape for this row
    calls for a dependency (§2.3's absent-references line):
    declared by the row itself, as ``expects_dependency``
    at row level or inside its ``task_shape`` — the linter
    grades absence only where the corpus declares the
    expectation, and never infers one."""
    if item.get("expects_dependency") is True:
        return True
    shape = item.get("task_shape")
    return isinstance(shape, dict) \
        and shape.get("expects_dependency") is True


def _normalise_reference_set(ref) -> set | None:
    """An A1 ``citable_reference_set`` shape (SI-103 §4 /
    SI-104) — ``{"seqs": [...], "empty": bool, ...}``,
    optionally wrapped under a ``citable_reference_set``
    key, or a bare list of seqs — to the set of citable
    seqs. None stays None (no set supplied)."""
    if ref is None:
        return None
    if isinstance(ref, dict) and isinstance(
            ref.get("citable_reference_set"), dict):
        ref = ref["citable_reference_set"]
    if isinstance(ref, dict):
        seqs = ref.get("seqs") or []
    elif isinstance(ref, list):
        seqs = ref
    else:
        raise ValueError(
            "reference set is neither the A1 "
            "citable_reference_set shape nor a seq list")
    return {s for s in seqs
            if isinstance(s, int) and not isinstance(s, bool)}


def _row_reference_set(item: dict, global_set):
    """A row's own citable set (same A1 shape, carried on
    the row) wins over the run-wide set; either may be
    absent, in which case citability is ungraded."""
    for key in ("citable_reference_set", "citable_seqs"):
        if key in item:
            return _normalise_reference_set(item[key])
    return global_set


# ---------------------------------------------------------------------------
# Target documents
# ---------------------------------------------------------------------------

def _extract_document(target):
    """(document, prose_prefixed) for one target.

    A dict target is the §8 document itself. A text target
    is the document when its first non-whitespace character
    opens it; text opening with anything else — prose
    narration before the document — is prose-prefixed
    (``SHAPE-PROSE-PREFIX``), and the document after the
    prefix, when one is present and decodable, is still
    linted: stripping the line in the report would be a
    re-score, but grading the document's own forms beside
    the shape verdict is the linter's whole job. A target
    with no decodable document yields ``(None, prefixed)``.
    """
    if isinstance(target, dict):
        return target, False
    if not isinstance(target, str):
        return None, False
    stripped = target.strip()
    if not stripped:
        return None, False
    if stripped.startswith("{"):
        try:
            doc = json.loads(stripped)
        except ValueError:
            return None, False
        return (doc if isinstance(doc, dict) else None), False
    start = stripped.find("{")
    if start > 0:
        try:
            doc = json.loads(stripped[start:])
        except ValueError:
            doc = None
        return (doc if isinstance(doc, dict) else None), True
    return None, True


def _intents_of(doc) -> list | None:
    """The §8 document's intents list, or None when the
    document carries none in either the wrapped
    (``{"proposal": {...}}``) or bare-proposal shape."""
    if not isinstance(doc, dict):
        return None
    proposal = doc.get("proposal")
    if not isinstance(proposal, dict):
        proposal = doc if "intents" in doc else None
    if not isinstance(proposal, dict):
        return None
    intents = proposal.get("intents")
    return intents if isinstance(intents, list) else None


# ---------------------------------------------------------------------------
# The lint
# ---------------------------------------------------------------------------

def _cited_descriptor(value) -> object:
    """The cited value for a verdict (§2.3): scalars are
    cited as themselves; a dict entry is cited by its
    sorted key names and an array by its length — the
    entry's *shape*, never a copy of a body fragment."""
    if isinstance(value, dict):
        return {"keys": sorted(str(k) for k in value)}
    if isinstance(value, list):
        return {"type": "array", "length": len(value)}
    return value


def _verdict(rule, intent_index, *, entry_index=None, arg=None,
             cited_value=None, declared_on=None, detail=""):
    return {
        "rule": rule,
        "taxonomy_name": _TAXONOMY[rule],
        "intent_index": intent_index,
        "entry_index": entry_index,
        "arg": arg,
        "cited_value": cited_value,
        "declared_on": declared_on,
        "detail": detail,
    }


def _lint_depends_on(entry, i, j, n_intents, citable, row_out):
    """One ``depends_on`` entry, priced by the §2.2
    reference rules. Correct forms and ungraded citations
    are counted on ``row_out``; violations return a
    verdict."""
    if isinstance(entry, bool):
        return _verdict(
            "REF-ENTRY-MALFORMED", i, entry_index=j,
            cited_value=entry,
            detail="entry_malformed — a boolean is neither "
                   "seq-form nor sibling-form (§9.3)")
    if isinstance(entry, int):
        seq = entry
    elif isinstance(entry, str):
        if entry in REGISTRY:
            return _verdict(
                "REF-TOOL-NAME", i, entry_index=j,
                cited_value=entry,
                detail="depends_on_tool_name_not_journal_seq "
                       "— the entry names a tool instead of "
                       "a journal seq (§9.3)")
        return _verdict(
            "REF-ENTRY-MALFORMED", i, entry_index=j,
            cited_value=entry,
            detail="entry_malformed — a string entry is "
                   "neither seq-form nor sibling-form (§9.3)")
    elif isinstance(entry, dict):
        if "sibling" in entry and "seq" in entry:
            return _verdict(
                "REF-ENTRY-MALFORMED", i, entry_index=j,
                cited_value=_cited_descriptor(entry),
                detail="entry_names_both_seq_and_sibling — "
                       "exactly one designator is a form "
                       "(§9.3; parser name carried under "
                       "REF-ENTRY-MALFORMED)")
        if "sibling" in entry:
            k = entry.get("sibling")
            if isinstance(k, bool) or not isinstance(k, int):
                return _verdict(
                    "REF-ENTRY-MALFORMED", i, entry_index=j,
                    cited_value=_cited_descriptor(entry),
                    detail="sibling_not_an_index — the "
                           "sibling designator is not an "
                           "integer position (§9.3; parser "
                           "name carried under "
                           "REF-ENTRY-MALFORMED)")
            if k < 0:
                return _verdict(
                    "REF-ENTRY-MALFORMED", i, entry_index=j,
                    cited_value=k,
                    detail="sibling_negative_index — the "
                           "sibling designator is negative "
                           "(§9.3; parser name carried under "
                           "REF-ENTRY-MALFORMED)")
            if k == i:
                return _verdict(
                    "REF-SIBLING-SELF", i, entry_index=j,
                    cited_value=k,
                    detail="sibling_self_reference — the "
                           "entry cites the citing intent "
                           "itself (§9.3)")
            if k > i:
                name = ("sibling_forward_reference"
                        if k < n_intents
                        else "sibling_out_of_range")
                return _verdict(
                    "REF-SIBLING-FORWARD", i, entry_index=j,
                    cited_value=k,
                    detail=f"{name} — the entry cites an "
                           f"index at or beyond the citing "
                           f"intent (SI-97's strictly-earlier "
                           f"rule); rule labelled unexercised: "
                           f"no recorded corpus exercised "
                           f"this class")
            row_out["correct_sibling_entries"] += 1
            return None
        if "seq" in entry:
            seq = entry.get("seq")
            if isinstance(seq, bool) or not isinstance(seq, int):
                return _verdict(
                    "REF-ENTRY-MALFORMED", i, entry_index=j,
                    cited_value=_cited_descriptor(entry),
                    detail="entry_malformed — the seq "
                           "designator is not an integer "
                           "journal sequence (§9.3)")
        else:
            return _verdict(
                "REF-ENTRY-MALFORMED", i, entry_index=j,
                cited_value=_cited_descriptor(entry),
                detail="entry_malformed — the entry names "
                       "neither a journal seq nor a sibling "
                       "(§9.3)")
    else:
        return _verdict(
            "REF-ENTRY-MALFORMED", i, entry_index=j,
            cited_value=_cited_descriptor(entry),
            detail="entry_malformed — the entry is neither "
                   "seq-form nor sibling-form (§9.3)")
    # -- seq-form grading ----------------------------------
    if seq <= 0:
        return _verdict(
            "REF-SEQ-NONPOSITIVE", i, entry_index=j,
            cited_value=seq,
            detail="seq_nonpositive — the citation names "
                   "journal seq <= 0 (§9.3)")
    if citable is None:
        row_out["seq_form_ungraded_entries"] += 1
        return None
    if seq in citable:
        row_out["correct_seq_entries"] += 1
        return None
    return _verdict(
        "REF-SEQ-NOT-INTENT", i, entry_index=j,
        cited_value=seq,
        detail="seq_not_intent_row — the cited seq is not "
               "in the citable reference set, whose members "
               "are exactly the same task's tool-intent "
               "rows (SI-103 §4 / SI-104); the citation "
               "names a non-intent row, another task's row, "
               "or no row")


def lint_corpus(corpus: dict, reference_set=None) -> dict:
    """Lint one loaded corpus (from :func:`load_corpus`)
    and return the eligibility report of §2.3, carrying
    the §2.4 hard label. ``reference_set`` is the optional
    run-wide citable set (A1 shape); a row's own set wins
    over it. Pure and offline: no file is written, no
    journal is opened, no trainer is called."""
    global_set = _normalise_reference_set(reference_set)
    owners = _registry_arg_owners()
    required = _registry_required()

    report = {
        "record": REPORT_RECORD,
        "version": 1,
        "label": HARD_LABEL,
        "corpus_id": corpus.get("corpus_id"),
        "corpus_sha256": corpus.get("corpus_sha256"),
        "reference_set_supplied": global_set is not None,
        "rows_total": 0,
        "rows_clean": 0,
        "rows_unparseable": 0,
        "rule_row_counts": {rule: 0 for rule in RULE_IDS},
        "rule_verdict_counts": {rule: 0 for rule in RULE_IDS},
        "correct_form_row_counts": {
            "seq_citation_in_citable_set": 0,
            "sibling_citation_strictly_earlier_non_self": 0},
        "correct_form_entry_counts": {
            "seq_citation_in_citable_set": 0,
            "sibling_citation_strictly_earlier_non_self": 0},
        "seq_form_ungraded_entries": 0,
        "absent_references_rows": 0,
        "absent_references_note": "",
        "rows": [],
    }
    any_declared = False

    for idx, row in enumerate(corpus["rows"]):
        item, target = row["item"], row["target"]
        report["rows_total"] += 1
        row_out = {
            "row_index": idx,
            "item_id": _row_id(item, idx),
            "target_sha256": _target_hash(target),
            "document": "unparseable",
            "verdicts": [],
            "correct_seq_entries": 0,
            "correct_sibling_entries": 0,
            "seq_form_ungraded_entries": 0,
            "references_absent_where_expected": False,
        }
        doc, prefixed = _extract_document(target)
        if prefixed:
            row_out["verdicts"].append(_verdict(
                "SHAPE-PROSE-PREFIX", None,
                detail="version_narration_prefix — the "
                       "target opens with prose narration "
                       "before the §8 document (SI-101); "
                       "the recorded instance is a 'Version "
                       "N' narration line"))
        intents = _intents_of(doc) if doc is not None else None
        if intents is None:
            report["rows_unparseable"] += 1
        else:
            row_out["document"] = (
                "prose_prefixed" if prefixed else "parsed")
            citable = _row_reference_set(item, global_set)
            n_entries = 0
            for i, intent in enumerate(intents):
                if not isinstance(intent, dict):
                    continue
                tool = intent.get("tool")
                args = intent.get("args")
                spec = REGISTRY.get(tool) \
                    if isinstance(tool, str) else None
                if spec is not None and isinstance(args, dict):
                    declared = set(
                        (spec.args_schema or {})
                        .get("properties") or {})
                    for arg in sorted(args):
                        if arg in declared:
                            continue
                        arg_owners = owners.get(arg) or []
                        if len(arg_owners) == 1:
                            row_out["verdicts"].append(_verdict(
                                "ARG-BLEED", i, arg=arg,
                                cited_value=arg,
                                declared_on=arg_owners[0],
                                detail=f"the argument {arg!r} "
                                       f"is declared on exactly "
                                       f"one registry tool, "
                                       f"{arg_owners[0]}, and "
                                       f"is carried here on "
                                       f"{tool} (SI-101 bleed "
                                       f"genus)"))
                    for req in required.get(tool) or []:
                        if req not in args:
                            row_out["verdicts"].append(_verdict(
                                "ARG-MISSING-REQUIRED", i,
                                arg=req, cited_value=req,
                                detail=f"required argument "
                                       f"{req!r} of {tool} is "
                                       f"absent (SI-101)"))
                deps = intent.get("depends_on")
                if isinstance(deps, list):
                    for j, entry in enumerate(deps):
                        n_entries += 1
                        verdict = _lint_depends_on(
                            entry, i, j, len(intents),
                            citable, row_out)
                        if verdict is not None:
                            row_out["verdicts"].append(verdict)
            if _row_expects_dependency(item):
                any_declared = True
                if n_entries == 0:
                    row_out["references_absent_where_expected"] \
                        = True
                    report["absent_references_rows"] += 1
        # -- fold the row into the corpus counts -----------
        seen_rules = set()
        for verdict in row_out["verdicts"]:
            report["rule_verdict_counts"][verdict["rule"]] += 1
            seen_rules.add(verdict["rule"])
        for rule in seen_rules:
            report["rule_row_counts"][rule] += 1
        if not row_out["verdicts"]:
            report["rows_clean"] += 1
        if row_out["correct_seq_entries"]:
            report["correct_form_row_counts"][
                "seq_citation_in_citable_set"] += 1
            report["correct_form_entry_counts"][
                "seq_citation_in_citable_set"] += \
                row_out["correct_seq_entries"]
        if row_out["correct_sibling_entries"]:
            report["correct_form_row_counts"][
                "sibling_citation_strictly_earlier_non_self"] \
                += 1
            report["correct_form_entry_counts"][
                "sibling_citation_strictly_earlier_non_self"] \
                += row_out["correct_sibling_entries"]
        report["seq_form_ungraded_entries"] += \
            row_out["seq_form_ungraded_entries"]
        del row_out["correct_seq_entries"]
        del row_out["correct_sibling_entries"]
        del row_out["seq_form_ungraded_entries"]
        report["rows"].append(row_out)

    report["absent_references_note"] = (
        "counted over rows whose own task shape declares "
        "expects_dependency and whose document carries no "
        "depends_on entry; stated as its own line, never "
        "folded into a pass figure (§2.3)"
        if any_declared else
        "no row in this corpus declares expects_dependency, "
        "so the absent-references line is 0 by construction, "
        "not by measurement (§2.3)")
    return report


def lint_file(corpus_path: str, reference_set_path=None) -> dict:
    """Load a corpus file (and an optional reference-set
    file, A1 shape) and lint it. Read-only."""
    corpus = load_corpus(corpus_path)
    reference_set = None
    if reference_set_path is not None:
        with open(reference_set_path, "r",
                  encoding="utf-8") as fh:
            reference_set = json.load(fh)
    return lint_corpus(corpus, reference_set=reference_set)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_text(report: dict, corpus_path: str = "-") -> str:
    """The human-readable eligibility report: the hard
    label first, then the corpus cited by id and hash,
    the per-rule row counts, the clean count, the correct-
    form counts, the absent-references line on its own —
    and every verdict cited by row, location, and value.
    Bodies are never rendered."""
    lines = [
        f"corpus: {corpus_path}",
        f"corpus_id: {report['corpus_id']}",
        f"corpus_sha256: {report['corpus_sha256']}",
        f"rows: {report['rows_total']}  "
        f"clean: {report['rows_clean']}  "
        f"unparseable: {report['rows_unparseable']}",
        f"reference_set_supplied: "
        f"{report['reference_set_supplied']}",
        "rule row counts (rows exhibiting each rule):",
    ]
    for rule in RULE_IDS:
        lines.append(
            f"  {rule}: {report['rule_row_counts'][rule]} "
            f"row(s), {report['rule_verdict_counts'][rule]} "
            f"verdict(s)")
    correct = report["correct_form_row_counts"]
    correct_entries = report["correct_form_entry_counts"]
    lines.append("correct forms (rows exhibiting each):")
    lines.append(
        f"  seq citation inside the citable set: "
        f"{correct['seq_citation_in_citable_set']} row(s), "
        f"{correct_entries['seq_citation_in_citable_set']} "
        f"entr(ies)")
    lines.append(
        f"  sibling citation strictly-earlier and non-self: "
        f"{correct['sibling_citation_strictly_earlier_non_self']}"
        f" row(s), "
        f"{correct_entries['sibling_citation_strictly_earlier_non_self']} "
        f"entr(ies)")
    lines.append(
        f"seq-form citations with citability ungraded "
        f"(no reference set): "
        f"{report['seq_form_ungraded_entries']} entr(ies)")
    lines.append(
        f"absent references where the task shape calls for "
        f"a dependency: {report['absent_references_rows']} "
        f"row(s) — {report['absent_references_note']}")
    flagged = [r for r in report["rows"] if r["verdicts"]]
    lines.append(f"verdicts: "
                 f"{sum(len(r['verdicts']) for r in flagged)} "
                 f"over {len(flagged)} flagged row(s)")
    for row in flagged:
        where = row["item_id"] or f"row {row['row_index']}"
        for v in row["verdicts"]:
            loc = f"intents[{v['intent_index']}]" \
                if v["intent_index"] is not None else "target"
            if v["entry_index"] is not None:
                loc += f".depends_on[{v['entry_index']}]"
            extra = f" arg={v['arg']}" if v["arg"] else ""
            extra += (f" declared_on={v['declared_on']}"
                      if v["declared_on"] else "")
            lines.append(
                f"  {where}  {loc}  {v['rule']}  "
                f"cited={v['cited_value']!r}{extra}  "
                f"({v['detail']})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI front-ends (module main + the si.cli wiring share `execute`)
# ---------------------------------------------------------------------------

def execute(args) -> int:
    """Run one parsed lint command (the argument namespace
    built by :func:`build_parser` here or by ``si.cli``'s
    ``gaia-lint`` parser — the two are the same shape).
    Returns the process exit code: 0 when the lint ran (a
    dirty corpus is a report, not a process failure), 1
    when the corpus or reference set could not be read or
    graded (fail closed)."""
    print(f"# {HARD_LABEL}")
    try:
        report = lint_file(
            args.corpus,
            reference_set_path=args.reference_set)
    except (OSError, ValueError) as e:
        print(f"si: error: {e}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(render_text(report, corpus_path=args.corpus))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="si.gaia.ref_lint",
        description="SI-119/RDL-1 reference-discipline "
                    "linter: an offline grade of a "
                    "candidate corpus against the recorded "
                    "failure classes of SI-111 §2.2. "
                    + HARD_LABEL + " It calls no trainer, "
                    "touches no live NEDB, and modifies "
                    "neither the parser, the gate, the "
                    "registry, nor the contract text.")
    parser.add_argument("corpus",
                        help="candidate corpus file (C-2 "
                             "shape: prompt/target pairs or "
                             "target documents alone; JSON "
                             "with an items list, a bare JSON "
                             "list, or JSONL)")
    parser.add_argument("--reference-set", default=None,
                        help="optional citable reference set "
                             "file (the A1 "
                             "citable_reference_set shape, "
                             "SI-104); without it, seq-form "
                             "citability is ungraded")
    parser.add_argument("--json", action="store_true",
                        help="print the eligibility report "
                             "as JSON")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return execute(args)
    except Exception as e:  # unreadable input etc: fail closed
        print(f"si: FAILED: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
