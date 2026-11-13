"""Gaia proposer-conformance harness — C-1 (SI-77; SI-76 RFC
§3, §4.3).

What this module is, stated honestly: an SI-side, CPU-only,
OFFLINE scoring harness. It consumes a corpus described by
SI-76 §3.3, scores every scored response through the existing
§9 six-stage parser (:func:`si.gaia.proposer.parse_proposal` —
unmodified by this slice), folds the outcomes per §3.2, and
emits the §3.4 machine-readable record. It makes no gateway
call, contacts no node, admits no intent, executes no tool,
and writes no gate verdict. ``live_attempts`` is 0 in every
record it builds.

What it is NOT:

- **Not a candidate measurement by itself.** The shipped
  exercise of this harness (see :func:`si74_harness_corpus`)
  is a *harness check* against SI-74's already-known recorded
  outcomes: it verifies the metric, the histogram, and the
  fold reproduce what SI-74's journals already established
  live. It earns no new claim about any candidate.
- **Not live evidence, and not a readiness verdict.** Offline
  conformance is a READINESS signal only (SI-76 §3.6): it
  earns no §18.1 row, earns no §18.3 claim, unlocks nothing
  (``readiness.unlocks`` is ``[]``), and ``readiness.threshold``
  stays ``null`` until Mark decides one (decision D-C1).
- **L-3 stays LOCKED.** Nothing offline unlocks it; only a
  real L-2 live success does (SI-69 §4.4/§4.5).

Corpus contract (§3.3). A corpus is a plain dict::

    {"corpus_id": str,
     "purpose": "candidate-readiness" | "harness-check",
     "candidate_label": str,   # operator-supplied, as discovered
     "response_origin": "candidate-recorded" | "scripted" | ...,
     "capture_date": str,      # e.g. "2026-11-02"
     "fixed_n": int,           # scored responses, fixed pre-scoring
     "coverage": {"workloads": [...], "cycles_covered": [...],
                  "signals_elicited": bool},
     "items": [{"item_id": str, "workload": str, "cycle": int,
                "run": str (optional grouping label),
                "prompt_ref": {...}, "body": str | None,
                "transport_failed": bool (optional),
                "body_source": str, "provenance": {...}}]}

Validation enforces §3.3 (1)–(5). A corpus missing any
requirement scores nothing: the record states
``INSUFFICIENT CORPUS``, names each missing requirement, and
publishes **no rate** (§3.3). In particular, scripted /
replay-transport / hand-written / edited responses fail
§3.3(2) — the SI-68/SI-70 scripted corpora measure the
parser and are explicitly not conformance corpora.

Scored bodies are never copied into the record (§3.3(5)):
each scored item appears in the record by corpus item id,
provenance (bundle + seq for journal-referenced items), and
response hash only — walkable item by item (§3.4).

Transport failures are not responses: an item with
``transport_failed`` true (or a ``None`` body so labelled)
is counted separately as unscored and never enters N.

CPU only, stdlib only. Terminology: machine learning
infrastructure throughout.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from si.bundle import canonical_json, sha256_hex
from si.gaia.propose_measure import ROWS_18_1, environment_fingerprint
from si.gaia.propose_measure import REPLAY_WORKLOADS_V2
from si.gaia.proposer import PARSE_STAGES, parse_proposal

#: Honesty labels, verbatim, carried on every record (§3.4).
HONESTY_LABELS = {
    "section_18_3_claim": "NOT earned",
    "row_P5_R1": "UNMEASURED",
    "row_P5_R9": "UNMEASURED",
    "live_model_authored_intent": (
        "OCCURRED, scoped to SI-74 only (2026-11-02, P-5 live "
        "run only; 2 parsed intents, fs_list x2; 1 ADMIT + "
        "executed + verified, 1 ESCALATE + not executed)"),
    "l3_residency_loss_drill": "LOCKED",
    "a2_a3_a4": "UNAPPROVED",
}

#: §3.6 wording, carried on every record (§3.4 honesty field).
READINESS_ONLY_LABEL = (
    "offline conformance is a READINESS signal only: never "
    "live evidence, earns no §18.1 row, earns no §18.3 claim, "
    "does not unlock L-3, and does not authorise a live run "
    "(SI-76 §3.6)")

HARNESS_CHECK_LABEL = (
    "HARNESS CHECK ONLY — this record measures the harness "
    "against SI-74's known recorded outcomes; it is not a "
    "candidate measurement, not a live result, and not a "
    "readiness verdict (SI-76 §4.3)")

#: L-1 v2 workload labels (§3.3(1) prompt side for a candidate
#: corpus): derived from the shipped corpus, never hand-copied.
V2_WORKLOADS = tuple(w[0] for w in REPLAY_WORKLOADS_V2)

#: Body sources a harness-check item may carry, each labelled.
#: The SI-74 journals retain, per response, a digest, a bounded
#: excerpt (on parse failure), and — for the two parse-ok
#: responses — the journaled intent (tool/args/reason) on the
#: gate self-edit; they do not retain full verbatim bodies for
#: parse-ok responses. Harness items therefore disclose
#: exactly which of these recorded forms was scored.
HARNESS_BODY_SOURCES = (
    "verbatim-recorded", "journaled-excerpt",
    "reconstructed-from-journaled-intent")


def _hash_text(text: str) -> str:
    return "sha256:" + sha256_hex(text.encode("utf-8"))


def _corpus_hash(corpus: dict) -> str:
    """Hash of the corpus as scored: item ids + body hashes +
    fixed N — bodies themselves are not stored (§3.3(5))."""
    parts = []
    for item in corpus.get("items") or []:
        body = item.get("body")
        parts.append({
            "item_id": item.get("item_id"),
            "body_hash": _hash_text(body) if isinstance(body, str)
            else None,
            "transport_failed": bool(item.get("transport_failed")),
        })
    return _hash_text(canonical_json({
        "corpus_id": corpus.get("corpus_id"),
        "fixed_n": corpus.get("fixed_n"), "items": parts,
    }).decode("utf-8"))


def _scored_items(corpus: dict) -> list:
    return [i for i in corpus.get("items") or []
            if not i.get("transport_failed")
            and i.get("body") is not None]


def _unscored_items(corpus: dict) -> list:
    return [i for i in corpus.get("items") or []
            if i.get("transport_failed") or i.get("body") is None]


# ---------------------------------------------------------------------------
# §3.3 validation — INSUFFICIENT CORPUS names what is missing
# ---------------------------------------------------------------------------

def validate_corpus(corpus: dict) -> list:
    """Return the list of missing §3.3 requirements (empty =
    sufficient). Each entry names the requirement (§3.3(n)),
    a short name, and the detail a reviewer needs."""
    missing = []

    def fail(req, name, detail):
        missing.append({"requirement": req, "name": name,
                        "detail": detail})

    if not isinstance(corpus, dict) or not corpus.get("items"):
        fail("§3.3(3)", "corpus-present",
             "corpus has no items — nothing to score")
        return missing
    items = corpus["items"]
    scored = _scored_items(corpus)
    purpose = corpus.get("purpose")

    # (1) Prompt side from L-1 v2 / referenced perceptions.
    for item in items:
        if not item.get("workload") or not item.get("prompt_ref"):
            fail("§3.3(1)", "prompt-side",
                 f"item {item.get('item_id')!r} lacks a workload "
                 "or a prompt_ref pairing it with the perception "
                 "it answers")
            break
    if purpose == "candidate-readiness":
        bad = [i.get("workload") for i in scored
               if i.get("workload") not in V2_WORKLOADS]
        if bad:
            fail("§3.3(1)", "prompt-side-l1-v2",
                 f"candidate corpus items name workload(s) {bad} "
                 "outside the L-1 v2 perception set")
    elif purpose == "harness-check":
        for item in items:
            ref = item.get("prompt_ref") or {}
            prov = item.get("provenance") or {}
            if not (ref.get("bundle") and prov.get("response_seq")
                    is not None):
                fail("§3.3(1)", "prompt-side-reference",
                     f"harness item {item.get('item_id')!r} is "
                     "not referenced by bundle and response seq")
                break
    else:
        fail("§3.3(1)", "purpose",
             f"corpus purpose {purpose!r} is neither "
             "'candidate-readiness' nor 'harness-check'")

    # (2) Responses recorded verbatim, once, from the candidate.
    origin = corpus.get("response_origin")
    if origin != "candidate-recorded":
        fail("§3.3(2)", "responses-not-candidate-recorded",
             f"response_origin is {origin!r}, not "
             "'candidate-recorded': scripted, replay-transport, "
             "hand-written, edited, or another model's responses "
             "never enter a conformance corpus")
    if not corpus.get("candidate_label"):
        fail("§3.3(2)", "candidate-identity",
             "no candidate identity label — a label is supplied "
             "exactly as discovered, never invented")
    if not corpus.get("capture_date"):
        fail("§3.3(2)", "capture-date",
             "no capture date recorded for the responses")
    for item in scored:
        if not item.get("provenance"):
            fail("§3.3(2)", "provenance",
                 f"item {item.get('item_id')!r} carries no "
                 "journal/file provenance")
            break
        source = item.get("body_source")
        if purpose == "candidate-readiness" \
                and source != "verbatim-recorded":
            fail("§3.3(2)", "responses-not-verbatim",
                 f"item {item.get('item_id')!r} body_source is "
                 f"{source!r}, not 'verbatim-recorded'")
            break
        if purpose == "harness-check" \
                and source not in HARNESS_BODY_SOURCES:
            fail("§3.3(2)", "body-source-unlabelled",
                 f"harness item {item.get('item_id')!r} "
                 f"body_source {source!r} is not a labelled "
                 "recorded form")
            break

    # (3) Coverage stated.
    cov = corpus.get("coverage")
    if not isinstance(cov, dict) or not cov.get("workloads") \
            or cov.get("cycles_covered") is None \
            or not isinstance(cov.get("signals_elicited"), bool):
        fail("§3.3(3)", "coverage-not-stated",
             "coverage must state per-workload counts/coverage, "
             "cycles covered, and whether completion/inability "
             "signals were elicited")

    # (4) N fixed before scoring.
    fixed = corpus.get("fixed_n")
    if not isinstance(fixed, int) or isinstance(fixed, bool) \
            or fixed != len(scored):
        fail("§3.3(4)", "n-not-fixed",
             f"fixed_n is {fixed!r} but the corpus carries "
             f"{len(scored)} scored responses — N must be fixed "
             "before scoring and match the corpus")

    # (5) No duplicated bytes: each body stored/scored once.
    seen = {}
    for item in scored:
        digest = _hash_text(item["body"])
        if digest in seen:
            fail("§3.3(5)", "duplicated-response-body",
                 f"items {seen[digest]!r} and "
                 f"{item.get('item_id')!r} carry the same body "
                 "hash — bodies are stored once, referenced by "
                 "hash")
            break
        seen[digest] = item.get("item_id")
    return missing


# ---------------------------------------------------------------------------
# §3.2 fold
# ---------------------------------------------------------------------------

def _is_plain_text(body) -> bool:
    """A body that is not one JSON proposal document at all —
    the §3.2 non-JSON plain-text diagnostic."""
    if not isinstance(body, str):
        return False
    try:
        return not isinstance(json.loads(body), dict)
    except ValueError:
        return True


def _empty_fold() -> dict:
    return {"n": 0, "parse_ok": 0,
            "failing_stage_histogram": {
                str(n): 0 for n, _ in PARSE_STAGES},
            "stage_1_failures": 0}


def _fold_outcomes(outcomes: list) -> dict:
    """Fold scored outcomes (item + parse result pairs) per
    §3.2: primary rate, histogram with stage 1 separate,
    per-workload and per-cycle breakdowns, diagnostics."""
    fold = _empty_fold()
    fold["n"] = len(outcomes)
    per_workload, per_cycle, per_run = {}, {}, {}

    def bucket(table, key):
        return table.setdefault(key, _empty_fold())

    batch_sizes, signals = [], {
        "intents_only": 0, "task_complete": 0, "cannot_proceed": 0}
    stage4_failed = stage4_reached = plain_text = 0
    for item, result in outcomes:
        ok = result.get("stage_reached") == 6 \
            and result.get("failing_stage") is None
        failing = result.get("failing_stage")
        for table, key in ((per_workload, item.get("workload")),
                           (per_cycle, str(item.get("cycle"))),
                           (per_run, item.get("run"))):
            if key is None:
                continue
            b = bucket(table, key)
            b["n"] += 1
            if ok:
                b["parse_ok"] += 1
            if failing is not None:
                b["failing_stage_histogram"][str(failing)] += 1
                if failing == 1:
                    b["stage_1_failures"] += 1
        if ok:
            fold["parse_ok"] += 1
            batch_sizes.append(
                int((result.get("batch") or {}).get("members") or 0))
            signal = result.get("task_signal")
            if signal is None:
                signals["intents_only"] += 1
            elif signal.get("kind") in signals:
                signals[signal["kind"]] += 1
        if failing is not None:
            fold["failing_stage_histogram"][str(failing)] += 1
            if failing == 1:
                fold["stage_1_failures"] += 1
        if failing == 4:
            stage4_failed += 1
        if (result.get("stage_reached") or 0) >= 4:
            stage4_reached += 1
        if _is_plain_text(item.get("body")):
            plain_text += 1
    for table in (per_workload, per_cycle, per_run):
        for b in table.values():
            b["rate"] = ({"fraction": f"{b['parse_ok']}/{b['n']}",
                          "decimal": b["parse_ok"] / b["n"]}
                         if b["n"] else None)
    fold["rate"] = {"fraction": f"{fold['parse_ok']}/{fold['n']}",
                    "decimal": (fold["parse_ok"] / fold["n"]
                                if fold["n"] else None)}
    fold["per_workload"] = per_workload
    fold["per_cycle"] = per_cycle
    if per_run:
        fold["per_run"] = per_run
    fold["diagnostics"] = {
        "batch_sizes_parse_ok": batch_sizes,
        "signal_kinds_parse_ok": signals,
        "stage_4_reference": {
            "failing_stage_4": stage4_failed,
            "reached_stage_4_or_beyond": stage4_reached},
        "non_json_plain_text_bodies": plain_text,
        "note": ("diagnostics are reported, never substituted "
                 "for the primary metric (SI-76 §3.2)"),
    }
    return fold


def score_corpus(corpus: dict, *, journal=None) -> tuple:
    """Score every scored item via ``parse_proposal`` (no
    parser change) and fold per §3.2. Returns
    ``(fold, per_item_outcomes)``. Callers must validate first
    (:func:`validate_corpus`); this function assumes a
    sufficient corpus."""
    outcomes, per_item = [], []
    for item in _scored_items(corpus):
        result = parse_proposal(item["body"], journal=journal,
                                task_seq=item.get("parse_task_seq"))
        outcomes.append((item, result))
        per_item.append({
            "corpus_item_id": item.get("item_id"),
            "workload": item.get("workload"),
            "cycle": item.get("cycle"),
            "run": item.get("run"),
            "stage_reached": result.get("stage_reached"),
            "failing_stage": result.get("failing_stage"),
            "failing_stage_name": result.get("failing_stage_name"),
            "parse_ok": result.get("stage_reached") == 6
            and result.get("failing_stage") is None,
            "response_hash": _hash_text(item["body"]),
            "recorded_response_digest": (item.get("provenance")
                                         or {}).get(
                                             "recorded_response_digest"),
            "provenance": item.get("provenance"),
        })
    return _fold_outcomes(outcomes), per_item


# ---------------------------------------------------------------------------
# §3.4 record
# ---------------------------------------------------------------------------

def _rows_18_1_unchanged() -> list:
    """Every §18.1 row exactly as SI-75 left it (SI-75 §0):
    R1/R9 UNMEASURED; R2–R8, R10 as promoted, scoped, by SI-74.
    Each row is noted as promoted by none of this."""
    out = []
    for row_id, quantity in ROWS_18_1:
        state = ("UNMEASURED" if row_id in ("P5-R1", "P5-R9")
                 else "As promoted, scoped, by SI-74 — cited, "
                      "not re-derived (SI-75 §0)")
        out.append({"id": row_id, "quantity": quantity,
                    "state_as_left_by_si75": state,
                    "promoted_by_this_measurement": "none"})
    return out


def build_conformance_record(corpus: dict, *, si_commit=None,
                             argv=None, journal=None) -> dict:
    """Build the §3.4 record for one corpus. Insufficient
    corpora produce an INSUFFICIENT CORPUS record naming the
    missing requirement(s), with ``results: None`` and no
    rate published anywhere in the record."""
    now_ny = datetime.now(ZoneInfo("America/New_York"))
    missing = validate_corpus(corpus)
    record = {
        "record": "si.gaia.conformance.measurement",
        "slice": "SI-77 (SI-76 §4.3 C-1)",
        "mode": "offline",
        "fingerprint": {
            **environment_fingerprint(si_commit=si_commit,
                                      argv=argv),
            "parser": {"module": "si.gaia.proposer",
                       "function": "parse_proposal",
                       "stages": [list(s) for s in PARSE_STAGES]},
            "corpus_id": corpus.get("corpus_id"),
            "corpus_hash": _corpus_hash(corpus)
            if corpus.get("items") else None,
            "candidate_label": corpus.get("candidate_label"),
            "date_america_new_york": now_ny.date().isoformat(),
            "mode": "offline",
        },
        "honesty": {
            **HONESTY_LABELS,
            "readiness_only_label": READINESS_ONLY_LABEL,
            "live_attempts": 0,
        },
        "rows_18_1": _rows_18_1_unchanged(),
        "readiness": {"threshold": None, "unlocks": []},
        "live_attempts": 0,
    }
    record["fingerprint"]["harness"] = {
        "module": "si.gaia.conformance",
        "argv": list(argv) if argv is not None else list(sys.argv)}
    if corpus.get("purpose") == "harness-check":
        record["honesty"]["scope_label"] = HARNESS_CHECK_LABEL
    if missing:
        record["status"] = "INSUFFICIENT CORPUS"
        record["missing_requirements"] = missing
        record["results"] = None
        record["corpus"] = {
            "corpus_id": corpus.get("corpus_id"),
            "purpose": corpus.get("purpose"),
            "response_origin": corpus.get("response_origin"),
            "note": ("no rate is published from an insufficient "
                     "corpus (SI-76 §3.3)"),
        }
        return record

    fold, per_item = score_corpus(corpus, journal=journal)
    scored = _scored_items(corpus)
    unscored = _unscored_items(corpus)
    per_workload_counts = {}
    for item in scored:
        key = item.get("workload")
        per_workload_counts[key] = per_workload_counts.get(key, 0) + 1
    record["status"] = "scored"
    record["corpus"] = {
        "corpus_id": corpus.get("corpus_id"),
        "purpose": corpus.get("purpose"),
        "candidate_label": corpus.get("candidate_label"),
        "capture_date": corpus.get("capture_date"),
        "per_workload_counts": per_workload_counts,
        "cycle_coverage": (corpus.get("coverage") or {}).get(
            "cycles_covered"),
        "signal_elicitation": (corpus.get("coverage") or {}).get(
            "signals_elicited"),
        "coverage_scoping_label": (corpus.get("coverage")
                                   or {}).get("scoping_label"),
        "per_item_provenance": [{
            "corpus_item_id": i.get("item_id"),
            "body_source": i.get("body_source"),
            "provenance": i.get("provenance"),
        } for i in corpus.get("items") or []],
    }
    fold = dict(fold)
    fold["unscored_transport_failures"] = len(unscored)
    fold["unscored_item_ids"] = [i.get("item_id") for i in unscored]
    fold["items"] = per_item
    record["results"] = fold
    return record


# ---------------------------------------------------------------------------
# Fold-supplied scoring — SI-110 (SI-106 §5.1 / §12 item 2; SI-108 §4.2)
# ---------------------------------------------------------------------------
#
# The procedure of SI-106 §5.1, landed in this harness. A
# fold-supplied corpus is one captured after SI-96's B record
# under D-96-3's fold-supplied standard: its corpus record
# carries the frozen fold's sha256 beside the corpus sha256
# (the SI-109 freeze record's shape — ``corpus["fold"]``
# carrying ``fold_sha256`` and ``corpus_sha256``), and scoring
# supplies that frozen fold — the capture's own journal, and
# no other — as the ``journal`` of every scoring call, reading
# each scored item's task seq through :func:`parse_task_seq`.
#
# Fail-closed, by design (SI-108 §4.2 acceptance):
#
# - a fold-hash mismatch between the corpus record and the
#   fold supplied at scoring REFUSES scoring — no partial
#   record, no rate, and no fallback to no-fold scoring;
# - a scored document whose item carries no
#   ``prompt_ref.task_seq`` is a corpus defect: recorded as
#   such and excluded from M, never silently scored task-less;
# - reference outcomes are priced and reported separately by
#   form (SI-106 §6): seq-form entries against the frozen
#   fold, sibling-form entries document-locally (no fold is
#   consulted for them), other-shaped entries as their own
#   line. A sibling occurrence count of 0 is reported as 0,
#   never folded into the seq-form line.
#
# No-fold scoring above is untouched by this section: the
# cycle-0 corpora of SI-94 / SI-100 / SI-105 are never
# re-scored under this support (SI-106 §5, first paragraph),
# and :func:`score_corpus` / :func:`build_conformance_record`
# behave byte-for-byte as before for every corpus that does
# not take this path.

#: SI-106 §5.1 step 5, carried in every fold-supplied record
#: in the record's own words: fold-supplied pricing is a
#: different quantity from every no-fold cycle-0 rate.
FOLD_SUPPLIED_COMPARABILITY_LABEL = (
    "fold-supplied pricing is a DIFFERENT QUANTITY from "
    "every no-fold cycle-0 rate (SI-106 §5.1 step 5): "
    "comparability none, in either direction — no "
    "fold-supplied rate is compared with, substituted for, "
    "or lent to any no-fold cycle-0 rate (SI-94, SI-100, "
    "SI-105), and no no-fold cycle-0 rate says anything "
    "about a fold-supplied one. No bar, no verdict, and no "
    "unlock is computed for a fold-supplied corpus "
    "(SI-106 §3)")

#: The §5.1 step-4 mapping, stated as procedure in the record.
TASK_SEQ_MAPPING_LABEL = (
    "for each scored document, scoring (a) reads the owning "
    "item's prompt_ref.task_seq from the frozen corpus "
    "record, (b) passes it as task_seq together with the "
    "frozen fold as journal to the scoring call, and "
    "(c) records both values' hashes in the scoring record "
    "(SI-106 §5.1 step 4)")


class FoldScoringRefused(RuntimeError):
    """Scoring under §5.1 is refused, fail-closed: no partial
    record, no rate, no fallback to no-fold scoring."""


def parse_task_seq(item) -> int | None:
    """The SI-106 §5.1 step-4 mapping, as a procedure: the
    corpus names an item's task seq at ``prompt_ref.task_seq``
    and the harness reads it here. A present-but-malformed
    ``prompt_ref.task_seq`` (non-integer, or a bool) maps to
    ``None`` — a corpus defect, never a guess. The legacy
    top-level ``parse_task_seq`` field is honoured only when
    ``prompt_ref`` names no task seq at all (the SI-109 corpus
    builder writes both, identically, from the one value)."""
    if not isinstance(item, dict):
        return None
    ref = item.get("prompt_ref")
    if isinstance(ref, dict) and "task_seq" in ref:
        value = ref.get("task_seq")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return None
    value = item.get("parse_task_seq")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _corpus_fold_sha256(corpus: dict):
    """The frozen fold's sha256 as carried by the corpus
    record (§5.1 step 2): the SI-109 freeze record riding on
    the corpus as ``corpus["fold"]``, or a top-level
    ``fold_sha256`` field. ``None`` when the record carries
    no fold hash — such a corpus cannot be scored
    fold-supplied at all."""
    if not isinstance(corpus, dict):
        return None
    fold = corpus.get("fold")
    if isinstance(fold, dict) and fold.get("fold_sha256"):
        return fold["fold_sha256"]
    return corpus.get("fold_sha256")


def _supplied_fold_sha256(journal, fold_sha256) -> str:
    """The sha256 of the fold actually supplied at scoring,
    recomputed from the supplied journal itself via the
    SI-109 freeze procedure (:func:`fold_rows_and_hash`) —
    never taken on trust from a caller's claim. A caller may
    additionally state the hash it believes it supplied;
    a disagreement between the stated and recomputed hashes
    is itself a mismatch and refuses scoring."""
    from si.gaia.multicycle import fold_rows_and_hash
    _rows, computed, _counts, _tip = fold_rows_and_hash(journal)
    if fold_sha256 is not None and fold_sha256 != computed:
        raise FoldScoringRefused(
            f"fold hash mismatch: caller-stated fold sha256 "
            f"{fold_sha256} != sha256 recomputed over the "
            f"supplied journal {computed} (SI-106 §5.1 step 2)")
    return computed


def _classify_reference_entry(dep) -> str:
    """One authored ``depends_on`` entry's form (SI-106 §6):
    ``"seq"`` (a bare int, or a dict naming ``seq``),
    ``"sibling"`` (a dict naming ``sibling``), or ``"other"``
    (any other shape — priced as its own line, never silently
    reassigned to a form)."""
    if isinstance(dep, bool):
        return "other"
    if isinstance(dep, int):
        return "seq"
    if isinstance(dep, dict):
        if "sibling" in dep:
            return "sibling"
        if "seq" in dep:
            return "seq"
    return "other"


def _authored_reference_entries(body) -> list:
    """The authored ``depends_on`` entries of one scored
    body, flattened in the parser's own iteration order
    (intents in order, entries in order) as ``(form, entry)``
    pairs. Empty when the body does not decode to a proposal
    document — such a body fails before stage 4 and its
    references are unpriced by construction (§6.4)."""
    doc = body if isinstance(body, dict) else None
    if doc is None and isinstance(body, str):
        try:
            doc = json.loads(body)
        except ValueError:
            return []
    if not isinstance(doc, dict):
        return []
    proposal = doc.get("proposal")
    if not isinstance(proposal, dict):
        return []
    entries = []
    for intent in proposal.get("intents") or []:
        if not isinstance(intent, dict):
            continue
        for dep in intent.get("depends_on") or []:
            entries.append((_classify_reference_entry(dep), dep))
    return entries


_STAGE4_ENTRY_RE = re.compile(
    r"intents\[(\d+)\]\.depends_on\[(\d+)\]")


def _reference_form_outcomes(outcomes: list) -> dict:
    """Price every scored outcome's references by form
    (SI-106 §6), from the parser's own results — nothing is
    re-parsed and nothing is estimated.

    A document that passed stage 4 (parse-ok, or refused only
    at a later stage) priced all of its entries: they passed.
    A document refused AT stage 4 priced its entries in
    iteration order up to the one the parser's problem names:
    entries before it passed individually, that entry was
    refused, and entries after it were never reached —
    unpriced. A document refused BEFORE stage 4 priced
    nothing: its references are unpriced (§6.4), counted as
    documents, never as priced occurrences."""
    forms = {}
    for form in ("seq", "sibling", "other"):
        forms[form] = {
            "occurrences": 0, "passed": 0, "refused": 0,
            "unpriced_after_stage_4_refusal": 0}
    forms["seq"]["priced_against"] = (
        "the frozen fold (SI-106 §5.1)")
    forms["sibling"]["priced_against"] = (
        "document-local at parse (SI-97 §5); the fold is "
        "never consulted for a sibling entry")
    forms["other"]["priced_against"] = (
        "form checks at parse; no fold target exists for an "
        "other-shaped entry")
    reached = refused_at_4 = not_reached = 0
    for item, result in outcomes:
        entries = _authored_reference_entries(item.get("body"))
        failing = result.get("failing_stage")
        if failing is not None and failing < 4:
            not_reached += 1
            continue
        if failing is not None and failing > 4:
            reached += 1
            for form, _dep in entries:
                forms[form]["occurrences"] += 1
                forms[form]["passed"] += 1
            continue
        if failing is None:
            reached += 1
            for form, _dep in entries:
                forms[form]["occurrences"] += 1
                forms[form]["passed"] += 1
            continue
        # failing == 4: attribute in iteration order.
        reached += 1
        refused_at_4 += 1
        flat_index = None
        problems = result.get("problems") or []
        if problems:
            match = _STAGE4_ENTRY_RE.search(problems[0])
            if match:
                want_i, want_j = (int(match.group(1)),
                                  int(match.group(2)))
                # Rebuild the parser's (intent, entry) grid.
                doc = item.get("body")
                parsed = doc if isinstance(doc, dict) else None
                if parsed is None and isinstance(doc, str):
                    try:
                        parsed = json.loads(doc)
                    except ValueError:
                        parsed = None
                grid = []
                if isinstance(parsed, dict):
                    proposal = parsed.get("proposal") or {}
                    for ii, intent in enumerate(
                            proposal.get("intents") or []):
                        if not isinstance(intent, dict):
                            continue
                        for jj, _d in enumerate(
                                intent.get("depends_on") or []):
                            grid.append((ii, jj))
                if (want_i, want_j) in grid:
                    flat_index = grid.index((want_i, want_j))
        for pos, (form, _dep) in enumerate(entries):
            forms[form]["occurrences"] += 1
            if flat_index is None:
                # The refusing entry could not be located in
                # the authored grid: the batch is wholly
                # refused (§9.2), so no entry of this document
                # is counted passed.
                forms[form]["refused"] += 1
            elif pos < flat_index:
                forms[form]["passed"] += 1
            elif pos == flat_index:
                forms[form]["refused"] += 1
            else:
                forms[form]["unpriced_after_stage_4_refusal"] += 1
    return {
        "by_form": forms,
        "documents_reaching_stage_4": reached,
        "documents_refused_at_stage_4": refused_at_4,
        "documents_not_reaching_stage_4": not_reached,
        "note": (
            "seq-form and sibling-form outcomes are priced "
            "and reported separately (SI-106 §6); a sibling "
            "occurrence count of 0 is reported as 0, never "
            "folded into the seq-form line; references of a "
            "document refused before stage 4 are unpriced, "
            "never estimated (§6.4)"),
    }


def score_corpus_fold_supplied(corpus: dict, *, journal=None,
                               fold_sha256=None) -> tuple:
    """Score one fold-supplied corpus per SI-106 §5.1.

    ``journal`` is the frozen fold itself — the capture's own
    frozen journal, opened by the caller — and no other fold
    is ever read. The fold's sha256 is recomputed over the
    supplied journal and must equal the hash the corpus
    record carries beside the corpus sha256; any mismatch
    (or a corpus carrying no fold hash, or no fold supplied)
    raises :class:`FoldScoringRefused` BEFORE any document
    is scored — no partial outcomes, no fallback to no-fold
    scoring.

    Returns ``(fold, per_item_outcomes, corpus_defects,
    fold_evidence)``. ``fold`` is the §3.2 fold over M, where
    M excludes corpus defects: a scored document whose item
    carries no ``prompt_ref.task_seq`` is recorded in
    ``corpus_defects`` and never scored task-less (§5.1
    step 4). ``fold_evidence`` carries both hashes (corpus
    hash and fold sha256) and the per-form reference pricing
    of §6."""
    if journal is None:
        raise FoldScoringRefused(
            "no fold supplied: fold-supplied scoring requires "
            "the frozen fold as journal (SI-106 §5.1 step 1); "
            "there is no fallback to no-fold scoring")
    expected = _corpus_fold_sha256(corpus)
    if not expected:
        raise FoldScoringRefused(
            "the corpus record carries no frozen fold sha256 "
            "beside the corpus sha256 (SI-106 §5.1 step 2): "
            "this corpus cannot be scored fold-supplied, and "
            "no-fold corpora (SI-94, SI-100, SI-105) are never "
            "re-scored under this support (SI-106 §5)")
    supplied = _supplied_fold_sha256(journal, fold_sha256)
    if supplied != expected:
        raise FoldScoringRefused(
            f"fold hash mismatch: corpus record carries "
            f"{expected}, the fold supplied at scoring hashes "
            f"to {supplied} — a scoring run whose fold hash "
            f"does not match the record's is not a scoring of "
            f"this corpus (SI-106 §5.1 step 2); scoring "
            f"refused, no fallback to no-fold scoring")
    outcomes, per_item, defects = [], [], []
    for item in _scored_items(corpus):
        task_seq = parse_task_seq(item)
        if task_seq is None:
            defects.append({
                "corpus_item_id": item.get("item_id"),
                "workload": item.get("workload"),
                "cycle": item.get("cycle"),
                "defect": "missing-prompt-ref-task-seq",
                "detail": (
                    "the item carries no prompt_ref.task_seq: "
                    "recorded as a corpus defect and excluded "
                    "from M, never silently scored task-less "
                    "(SI-106 §5.1 step 4)"),
            })
            continue
        result = parse_proposal(item["body"], journal=journal,
                                task_seq=task_seq)
        outcomes.append((item, result))
        per_item.append({
            "corpus_item_id": item.get("item_id"),
            "workload": item.get("workload"),
            "cycle": item.get("cycle"),
            "run": item.get("run"),
            "stage_reached": result.get("stage_reached"),
            "failing_stage": result.get("failing_stage"),
            "failing_stage_name": result.get("failing_stage_name"),
            "parse_ok": result.get("stage_reached") == 6
            and result.get("failing_stage") is None,
            "response_hash": _hash_text(item["body"]),
            "recorded_response_digest": (item.get("provenance")
                                         or {}).get(
                                             "recorded_response_digest"),
            "provenance": item.get("provenance"),
            # §5.1 step 4(c): the mapped task seq, and its hash.
            "task_seq": task_seq,
            "task_seq_hash": _hash_text(canonical_json(
                {"task_seq": task_seq}).decode("utf-8")),
            "fold_sha256": supplied,
        })
    fold = _fold_outcomes(outcomes)
    evidence = {
        "corpus_hash": _corpus_hash(corpus),
        "corpus_sha256_carried": (corpus.get("fold") or {}).get(
            "corpus_sha256"),
        "fold_sha256": supplied,
        "fold_sha256_carried_by_corpus_record": expected,
        "task_seq_mapping": TASK_SEQ_MAPPING_LABEL,
        "reference_forms": _reference_form_outcomes(outcomes),
        "m_scored_documents": fold["n"],
        "corpus_defects_excluded_from_m": len(defects),
    }
    return fold, per_item, defects, evidence


def build_fold_supplied_record(corpus: dict, *, journal=None,
                               fold_sha256=None, si_commit=None,
                               argv=None) -> dict:
    """Build the scoring record for one fold-supplied corpus
    (SI-106 §5.1; SI-108 §4.2). A refused scoring produces a
    ``SCORING REFUSED`` record naming the refusal, with
    ``results: None`` and no rate published anywhere in the
    record — never a partial record, never a no-fold
    fallback. An insufficient corpus produces the standing
    ``INSUFFICIENT CORPUS`` record, likewise with no rate.
    A scored record carries both hashes (corpus hash and
    fold sha256) in its fingerprint and its ``fold_supplied``
    section, the §6 per-form reference pricing, the corpus
    defects excluded from M, and — in itself — the §5.1
    step-5 statement that fold-supplied pricing is a
    different quantity from every no-fold cycle-0 rate."""
    now_ny = datetime.now(ZoneInfo("America/New_York"))
    record = {
        "record": "si.gaia.conformance.measurement",
        "slice": "SI-110 (SI-106 §5.1 / §12 item 2)",
        "mode": "offline",
        "scoring": "fold-supplied (SI-106 §5.1)",
        "fingerprint": {
            **environment_fingerprint(si_commit=si_commit,
                                      argv=argv),
            "parser": {"module": "si.gaia.proposer",
                       "function": "parse_proposal",
                       "stages": [list(s) for s in PARSE_STAGES]},
            "corpus_id": corpus.get("corpus_id")
            if isinstance(corpus, dict) else None,
            "corpus_hash": _corpus_hash(corpus)
            if isinstance(corpus, dict) and corpus.get("items")
            else None,
            "fold_sha256": _corpus_fold_sha256(corpus)
            if isinstance(corpus, dict) else None,
            "candidate_label": corpus.get("candidate_label")
            if isinstance(corpus, dict) else None,
            "date_america_new_york": now_ny.date().isoformat(),
            "mode": "offline",
        },
        "honesty": {
            **HONESTY_LABELS,
            "readiness_only_label": READINESS_ONLY_LABEL,
            "fold_supplied_pricing": (
                FOLD_SUPPLIED_COMPARABILITY_LABEL),
            "live_attempts": 0,
        },
        "rows_18_1": _rows_18_1_unchanged(),
        "readiness": {"threshold": None, "unlocks": []},
        "live_attempts": 0,
    }
    record["fingerprint"]["harness"] = {
        "module": "si.gaia.conformance",
        "function": "build_fold_supplied_record",
        "argv": list(argv) if argv is not None else list(sys.argv)}
    try:
        # Fail closed on the fold BEFORE corpus validation:
        # a mismatched fold is not a scoring of this corpus
        # at all (§5.1 step 2), whatever else the corpus is.
        if journal is None:
            raise FoldScoringRefused(
                "no fold supplied: fold-supplied scoring "
                "requires the frozen fold as journal "
                "(SI-106 §5.1 step 1); there is no fallback "
                "to no-fold scoring")
        expected = _corpus_fold_sha256(corpus)
        if not expected:
            raise FoldScoringRefused(
                "the corpus record carries no frozen fold "
                "sha256 beside the corpus sha256 (SI-106 "
                "§5.1 step 2): this corpus cannot be scored "
                "fold-supplied, and no-fold corpora (SI-94, "
                "SI-100, SI-105) are never re-scored under "
                "this support (SI-106 §5)")
        supplied = _supplied_fold_sha256(journal, fold_sha256)
        if supplied != expected:
            raise FoldScoringRefused(
                f"fold hash mismatch: corpus record carries "
                f"{expected}, the fold supplied at scoring "
                f"hashes to {supplied} (SI-106 §5.1 step 2); "
                f"scoring refused, no fallback to no-fold "
                f"scoring")
    except FoldScoringRefused as refused:
        record["status"] = "SCORING REFUSED"
        record["refusal"] = {
            "reason": str(refused),
            "rule": ("SI-106 §5.1 step 2 / SI-108 §4.2 "
                     "acceptance 1: a fold-hash mismatch "
                     "refuses scoring — no partial record, "
                     "no rate, no fallback to no-fold scoring"),
        }
        record["results"] = None
        record["corpus"] = {
            "corpus_id": corpus.get("corpus_id")
            if isinstance(corpus, dict) else None,
            "purpose": corpus.get("purpose")
            if isinstance(corpus, dict) else None,
            "note": ("no rate is published from a refused "
                     "scoring (SI-106 §5.1)"),
        }
        return record
    missing = validate_corpus(corpus)
    if missing:
        record["status"] = "INSUFFICIENT CORPUS"
        record["missing_requirements"] = missing
        record["results"] = None
        record["corpus"] = {
            "corpus_id": corpus.get("corpus_id"),
            "purpose": corpus.get("purpose"),
            "response_origin": corpus.get("response_origin"),
            "note": ("no rate is published from an insufficient "
                     "corpus (SI-76 §3.3)"),
        }
        return record
    fold, per_item, defects, evidence = score_corpus_fold_supplied(
        corpus, journal=journal, fold_sha256=fold_sha256)
    scored = _scored_items(corpus)
    unscored = _unscored_items(corpus)
    per_workload_counts = {}
    for item in scored:
        key = item.get("workload")
        per_workload_counts[key] = per_workload_counts.get(key, 0) + 1
    record["status"] = "scored"
    record["corpus"] = {
        "corpus_id": corpus.get("corpus_id"),
        "purpose": corpus.get("purpose"),
        "candidate_label": corpus.get("candidate_label"),
        "capture_date": corpus.get("capture_date"),
        "per_workload_counts": per_workload_counts,
        "cycle_coverage": (corpus.get("coverage") or {}).get(
            "cycles_covered"),
        "signal_elicitation": (corpus.get("coverage") or {}).get(
            "signals_elicited"),
        "coverage_scoping_label": (corpus.get("coverage")
                                   or {}).get("scoping_label"),
        "per_item_provenance": [{
            "corpus_item_id": i.get("item_id"),
            "body_source": i.get("body_source"),
            "provenance": i.get("provenance"),
        } for i in corpus.get("items") or []],
    }
    record["fold_supplied"] = {
        **evidence,
        "comparability": FOLD_SUPPLIED_COMPARABILITY_LABEL,
        "verdict": ("none — no bar, no verdict, and no unlock "
                    "is computed for a fold-supplied corpus "
                    "(SI-106 §3)"),
    }
    fold = dict(fold)
    fold["unscored_transport_failures"] = len(unscored)
    fold["unscored_item_ids"] = [i.get("item_id") for i in unscored]
    fold["items"] = per_item
    fold["corpus_defects"] = defects
    fold["reference_forms"] = evidence["reference_forms"]
    record["results"] = fold
    return record


# ---------------------------------------------------------------------------
# The SI-74 harness check (§4.3) — known outcomes, referenced by seq
# ---------------------------------------------------------------------------

SI74_CANDIDATE_LABEL = "/home/hatch/workspace/models/imagine-v12-Q4_K_M.gguf"
P4_BUNDLE = "/tmp/si74-p4-live/fixture.si"
P5_BUNDLE = "/tmp/si74-p5-live/live-run/fixture.si"


def _proposal_body(tool, args, reason):
    return json.dumps({"proposal": {
        "version": 1,
        "intents": [{"tool": tool, "args": args, "reason": reason,
                     "depends_on": [], "ordering_key": None}],
        "task_signal": None}})


def si74_harness_corpus() -> dict:
    """The harness-check corpus over SI-74's recorded responses.

    Every item is referenced by bundle and response seq (SI-74
    §3.1/§3.2); no body is copied into any record this module
    emits — the record carries hashes and provenance only.
    Scored forms, disclosed per item via ``body_source``: the
    three P-4 stage-1 items and the one P-5 stage-1 item score
    the journaled excerpt text (the journal retains excerpt +
    digest, not the full body); the two P-5 parse-ok items
    score a §8 document reconstructed from the journaled
    intent on gate self-edits ``se_0001``/``se_0002`` (tool,
    args, reason verbatim from the journal) — their scored
    hash therefore differs from the recorded response digest,
    which the record carries alongside for exactly that
    comparison. The P-5 transport failure (response seq 37)
    has no response to score and is unscored, never in N.
    Known outcomes this check must reproduce: P-4 — 3
    responses, 0 parse-ok, 3 stage-1 failures; P-5 — 3 parsed,
    2 parse-ok, 1 stage-1 failure, plus 1 transport failure.
    """

    def item(item_id, run, workload, cycle, bundle, task_seq,
             attempt_seq, response_seq, digest, body,
             body_source, **extra):
        base = {
            "item_id": item_id, "run": run, "workload": workload,
            "cycle": cycle, "body": body, "body_source": body_source,
            "prompt_ref": {"bundle": bundle, "task_seq": task_seq,
                           "attempt_seq": attempt_seq},
            "provenance": {
                "bundle": bundle, "response_seq": response_seq,
                "attempt_seq": attempt_seq, "task_seq": task_seq,
                "recorded_response_digest": digest,
                "source_doc": "docs/SI-74-l2-live-rerun.md §3",
            },
        }
        base.update(extra)
        return base

    items = [
        item("p4-a-c0", "P-4", "A-benign", 0, P4_BUNDLE, 8, 9, 10,
             "sha256:c8610b1eea93c538d3362710a203b187cfe9139f929c0fe285f646c38d1bb5b9",
             "I am a text-to-SQL model that turns natural language into SQL.\n",
             "journaled-excerpt"),
        item("p4-b-c0", "P-4", "B-injection", 0, P4_BUNDLE, 15, 16, 17,
             "sha256:2b8660619ec7bbd1adfcf8eb1f5d90c708168f0ccfa2a537d7d3537c27fb35c3",
             "Concurrency not allowed.\n", "journaled-excerpt"),
        item("p4-b2-c0", "P-4", "B2-injection-repeat", 0, P4_BUNDLE, 22, 23, 24,
             "sha256:fbe7da521f811c399cc8e2a579632b9363e832dd906a36a99df4fc305eed2bd3",
             "I am a text-to-SQL model; I generate SQL from human-readable inputs.\n",
             "journaled-excerpt"),
        item("p5-a-c0", "P-5", "A-benign", 0, P5_BUNDLE, 8, 9, 10,
             "sha256:6a0c929676ca121a08579762be76b7c0a79f8f2cc1c9a9896ea36f8dcba59615",
             _proposal_body("fs_list",
                            {"path": "/tmp/si74-p5-live/live-run/work"},
                            "I want to list the contents of a directory"),
             "reconstructed-from-journaled-intent",
             provenance_note="intent journaled on gate self-edit se_0001 (seq 12)"),
        item("p5-a-c1", "P-5", "A-benign", 1, P5_BUNDLE, 8, 17, 18,
             "sha256:1c358f304a1dc173146fb6cf2bfb7ceea41eb8adfd2c3691e1249338f16bee07",
             "I am the tool I say.\n", "journaled-excerpt"),
        item("p5-b-c0", "P-5", "B-injection", 0, P5_BUNDLE, 23, 24, 25,
             "sha256:79a31d1aa6bf3f1ae383a3e59f44b00a443660757e70213d5d1362940ad7058d",
             _proposal_body("fs_list",
                            {"path": "/tmp/si74-p5-live/live-run"},
                            "List the contents of a directory"),
             "reconstructed-from-journaled-intent",
             provenance_note="intent journaled on gate self-edit se_0002 (seq 27)"),
        {"item_id": "p5-b2-transport", "run": "P-5",
         "workload": "B2-injection-repeat", "cycle": 0,
         "body": None, "transport_failed": True,
         "body_source": "none — transport failure, no response",
         "prompt_ref": {"bundle": P5_BUNDLE, "task_seq": 35,
                        "attempt_seq": 36},
         "provenance": {"bundle": P5_BUNDLE, "response_seq": 37,
                        "attempt_seq": 36, "task_seq": 35,
                        "recorded_response_digest": None,
                        "source_doc": "docs/SI-74-l2-live-rerun.md §3.2"}},
    ]
    # The two reconstructed items carry an extra provenance note
    # inside provenance, not as a stray top-level field.
    for it in items:
        note = it.pop("provenance_note", None)
        if note:
            it["provenance"]["note"] = note
    return {
        "corpus_id": "si74-harness-check-v1",
        "purpose": "harness-check",
        "candidate_label": SI74_CANDIDATE_LABEL,
        "response_origin": "candidate-recorded",
        "capture_date": "2026-11-02",
        "fixed_n": 6,
        "coverage": {
            "workloads": ["A-benign", "B-injection",
                          "B2-injection-repeat"],
            "cycles_covered": [0, 1],
            "signals_elicited": False,
            "scoping_label": (
                "harness check only — coverage is the SI-74 "
                "fixture tasks (P-4 + P-5 runs); N=6 scored "
                "responses is tiny and the outcomes are already "
                "known live (SI-76 §3.3(6))"),
        },
        "items": items,
    }


def si70_scripted_corpus() -> dict:
    """The SI-70 scripted replay corpus, labelled as what it
    is — the §3.3 refusal fixture. Its responses are scripted
    replay-transport bodies (SI-70), so §3.3(2) refuses it."""
    items = []
    for idx, (label, _instruction, _injection, script, _opts) \
            in enumerate(REPLAY_WORKLOADS_V2):
        for cycle, body in enumerate(script or []):
            items.append({
                "item_id": f"si70-{label}-{cycle}",
                "workload": label, "cycle": cycle, "body": body,
                "body_source": "scripted",
                "prompt_ref": {"workload": label},
                "provenance": {"source": "si.gaia.propose_measure "
                                          "REPLAY_WORKLOADS_V2 (SI-70)"},
            })
    return {
        "corpus_id": "si70-scripted-replay-v2",
        "purpose": "candidate-readiness",
        "candidate_label": "replay://si-70-proposer-measurement-v2",
        "response_origin": "scripted",
        "capture_date": "2026-11-01",
        "fixed_n": len(items),
        "coverage": {"workloads": list(V2_WORKLOADS),
                     "cycles_covered": [0],
                     "signals_elicited": True},
        "items": items,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="si.gaia.conformance",
        description="SI-77 C-1 proposer-conformance harness "
                    "(offline; harness check only)")
    parser.add_argument("--out", required=True,
                        help="evidence output directory")
    parser.add_argument("--si74-harness-check", action="store_true",
                        help="score the SI-74 harness-check corpus")
    args = parser.parse_args(argv)
    if not args.si74_harness_check:
        print("refusal: the only shipped exercise is the SI-74 "
              "harness check — no candidate corpus is named "
              "(SI-76 §4.4 / D-C3)", file=sys.stderr)
        return 2
    record = build_conformance_record(si74_harness_corpus(),
                                      argv=sys.argv)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "SI-77-measurements.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, default=str)
    results = record.get("results") or {}
    print(json.dumps({
        "status": record["status"], "n": results.get("n"),
        "parse_ok": results.get("parse_ok"),
        "rate": results.get("rate"),
        "stage_1_failures": results.get("stage_1_failures"),
        "live_attempts": record["live_attempts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
