"""SI-81 (SI-80 section 3.3 F-1): failure taxonomy over the SI-79 corpus.

OFFLINE ONLY. No network, no node, no live attempt, no capture.
This module classifies bodies already stored once in
``docs/SI-79-corpus.json`` under the stage outcomes SI-79 already
recorded in ``docs/SI-79-measurements.json``. Those outcomes are
inputs, never re-derived: this module never calls the proposer
parser, never re-scores, and publishes no new parse rate.

The taxonomy earns no claim and no unlock: section 18.3 NOT
earned, P5-R1/P5-R9 UNMEASURED, live model-authored intent
OCCURRED scoped to SI-74 only, L-3 LOCKED, A-2/A-3/A-4
UNAPPROVED, unlocks []. Bodies are cited by SI-79 response hash
and corpus item id only; body text is never copied into the
taxonomy record (never duplicate bytes).
"""
import hashlib
import json
import re

# Stage-1 precedence order (first matching predicate wins).
STAGE1_PRECEDENCE = (
    "empty_body",
    "fenced_json",
    "partial_json",
    "sql_shaped",
    "identity_text",
    "bare_fragment",
    "plain_prose",
)

_SQL_HEAD = re.compile(
    r"^(SELECT|INSERT\s+INTO|UPDATE|DELETE\s+FROM|WITH)\b",
    re.IGNORECASE)
_IDENTITY_HEAD = re.compile(r"^I am\b")
_SENTENCE_END = re.compile(r"[.!?]")


def response_hash(body):
    """The SI-79 response hash of a stored body."""
    return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


def classify_stage1_body(body):
    """Classify one stage-1 body by deterministic text predicates.

    Precedence is STAGE1_PRECEDENCE; ``unclassified`` is the
    fallback no predicate decides (never reached for plain text
    shapes, because plain_prose is the terminal text class; it
    remains the stated fallback for non-text inputs).
    """
    if not isinstance(body, str):
        return "unclassified"
    s = body.strip()
    if s == "":
        return "empty_body"
    if s.startswith("```"):
        return "fenced_json"
    if s.startswith("{") or s.startswith("["):
        return "partial_json"
    if _SQL_HEAD.match(s):
        return "sql_shaped"
    if _IDENTITY_HEAD.match(s):
        return "identity_text"
    if not _SENTENCE_END.search(s) and len(s.split()) <= 3:
        return "bare_fragment"
    return "plain_prose"


def classify_stage4_body(body):
    """Classify one stage-4 body by its recorded reference failure.

    The stage-4 outcome itself is SI-79's recorded parser outcome
    (an input). This predicate only names the observable failing
    reference: a ``depends_on`` entry that is a string naming a
    tool used by another intent of the same proposal, where the
    section 9.3 reference must name a journal sequence.
    Returns (class_id, failing_reference or None).
    """
    if not isinstance(body, str):
        return "unclassified", None
    try:
        doc = json.loads(body)
        intents = doc["proposal"]["intents"]
        tools = {it.get("tool") for it in intents}
        for it in intents:
            for dep in it.get("depends_on") or []:
                if isinstance(dep, str) and dep in tools:
                    return "depends_on_tool_name_not_journal_seq", dep
    except (ValueError, KeyError, TypeError, AttributeError):
        pass
    return "unclassified", None


def classify_control_body(body):
    """Control class: SI-79 parse-ok body in the bare-document shape."""
    if isinstance(body, str) and body.strip().startswith("{"):
        return "parse_ok_bare_json_proposal"
    return "unclassified"


def _entry(item, meas_row):
    return {"item_id": item["item_id"], "workload": item["workload"],
            "response_hash": meas_row["response_hash"]}


def build_taxonomy(corpus, measurements):
    """Fold the SI-79 corpus + measurements into the F-1 record.

    Partitions come from the SI-79 measurements (parse_ok /
    failing_stage); each stored body's hash is verified against
    the SI-79 response hash before it is classified.
    """
    meas = {r["corpus_item_id"]: r
            for r in measurements["results"]["items"]}
    s1 = {c: [] for c in STAGE1_PRECEDENCE}
    s1["unclassified"] = []
    s4 = {"depends_on_tool_name_not_journal_seq": [],
          "unclassified": []}
    s4_refs = {}
    ctl = {"parse_ok_bare_json_proposal": [], "unclassified": []}
    excluded = []
    for item in corpus["items"]:
        iid = item["item_id"]
        if item.get("transport_failed"):
            prov = item.get("provenance") or {}
            excluded.append({
                "item_id": iid, "workload": item["workload"],
                "reason": ("transport failure, no response "
                           "(SI-79): " + str(prov.get(
                               "detail", "transport failure")))})
            continue
        row = meas[iid]
        assert response_hash(item["body"]) == row["response_hash"], iid
        if row["parse_ok"]:
            ctl[classify_control_body(item["body"])].append(
                _entry(item, row))
        elif row["failing_stage"] == 1:
            s1[classify_stage1_body(item["body"])].append(
                _entry(item, row))
        elif row["failing_stage"] == 4:
            cls, ref = classify_stage4_body(item["body"])
            s4[cls].append(_entry(item, row))
            if ref is not None:
                s4_refs[iid] = ref
        else:  # a stage SI-79 did not record in this corpus
            raise ValueError(f"{iid}: unexpected failing_stage "
                             f"{row['failing_stage']}")

    def part(classes):
        return {
            "total": sum(len(v) for v in classes.values()),
            "classes": [{"class": c, "count": len(items),
                         "items": items}
                        for c, items in classes.items()],
            "unclassified_count": len(classes["unclassified"])}

    return {
        "fingerprint": {
            "corpus_id": corpus["corpus_id"],
            "corpus_hash": measurements["fingerprint"]["corpus_hash"],
            "mode": "offline"},
        "partitions": {
            "stage_1_failures": part(s1),
            "stage_4_failures": {**part(s4),
                                 "failing_references": s4_refs},
            "control_parse_ok": part(ctl)},
        "excluded_unscored": {
            "count": len(excluded), "items": excluded},
        "totals": {
            "scored": (sum(len(v) for v in s1.values())
                       + sum(len(v) for v in s4.values())
                       + sum(len(v) for v in ctl.values())),
            "stage_1": sum(len(v) for v in s1.values()),
            "stage_4": sum(len(v) for v in s4.values()),
            "control": sum(len(v) for v in ctl.values()),
            "unclassified": (len(s1["unclassified"])
                             + len(s4["unclassified"])
                             + len(ctl["unclassified"]))}}
