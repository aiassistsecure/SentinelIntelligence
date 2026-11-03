"""Gaia C-2 capture + record glue — SI-79 (SI-76 §4.4).

What this module is, stated honestly: the capture and record
glue for the FIRST proposer-conformance measurement of a
named candidate (decision D-C3, Mark, 2026-11-02: Imagine
v12, the current Resident). It does exactly two things:

1. **Capture** (corpus capture ONLY): for each of the six
   L-1 v2 workloads, it journals one task entry in a
   capture-only fixture bundle, re-polls the real node's
   residency and journals the observation (the live suite's
   own pattern), assembles the cycle-0 perception with
   :func:`si.gaia.proposer.assemble_perception`, renders the
   gateway request with
   :func:`si.gaia.proposer.build_gateway_request` — the live
   proposer's own rendering, no new prompt shape — and makes
   ONE ``POST /v1/chat/completions`` call per item to the
   already-running local node's inference base. The response
   content is extracted exactly as the live proposer extracts
   it (``choices[0].message.content``, else the raw body
   text) and recorded verbatim.

   Capture admits NO intent, executes NO tool, and writes NO
   gate verdict. It is not an L-2 retry. Nothing is parsed
   during capture; the capture loop hashes bodies only to
   enforce the pre-registered distinctness rule (§3.3(5):
   bodies are stored once).

2. **Record**: after the corpus is frozen, it scores the
   corpus through the existing C-1 harness
   (:mod:`si.gaia.conformance`, parser unmodified) and sets
   the record's readiness fields against the DECIDED D-C1
   threshold (Mark, 2026-11-02: rate >= 0.80 over N >= 30,
   >= 5 per workload, per-workload floor >= 0.60 — the SI-78
   proposal adopted as-is). ``readiness.unlocks`` stays
   ``[]``: C-2 unlocks nothing; C-4 remains a separate
   conditional slice.

Honesty labels (verbatim, unweakened): §18.3 NOT earned;
P5-R1 UNMEASURED; P5-R9 UNMEASURED; live model-authored tool
intent OCCURRED scoped to SI-74 only (this capture does not
extend that scope — captured responses are a conformance
corpus, not executed intents); L-3 LOCKED; A-2/A-3/A-4
UNAPPROVED. Offline conformance is a READINESS SIGNAL ONLY:
it earns no §18.1 row and no §18.3 claim, and does not
unlock L-3.

Corpus store (D-C4 naming duty — named here and in the
SI-79 slice doc BEFORE the scoring section): a corpus file
in the repo, stored once — ``docs/SI-79-corpus.json``.
Bodies live only there; the §3.4 record references them by
hash. No model/GGUF/tensor bytes are copied anywhere.

CPU only for SI-side work; the only node contacted is the
already-running local one. Terminology: machine learning
infrastructure throughout.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from si.bundle import canonical_json, sha256_hex
from si.gaia import live as live_mod
from si.gaia.conformance import build_conformance_record, \
    validate_corpus
from si.gaia.hearth import HttpTransport
from si.gaia.policy import resolve_tool_policy
from si.gaia.propose_measure import REPLAY_WORKLOADS_V2
from si.gaia.proposer import GATEWAY_CHAT_PATH, assemble_perception, \
    build_gateway_request
from si.gaia.task import record_task_entry
from si.journal import open_journal

#: The decided D-C1 threshold (Mark, 2026-11-02; the SI-78
#: proposal adopted as-is).
D_C1_DECISION = ("D-C1 decided by Mark, 2026-11-02 — the "
                 "SI-78 proposal adopted as-is")
D_C1 = {"rate_min": 0.80, "min_n": 30, "min_per_workload": 5,
        "per_workload_floor": 0.60, "decision": D_C1_DECISION}

#: The corpus store, named before scoring (D-C4 duty).
CORPUS_STORE = ("corpus file in the repo, stored once: "
                "docs/SI-79-corpus.json (bodies held only "
                "there; the record references them by hash)")

CAPTURE_TIMEOUT_S = 900.0


class ResidencyGuardHalt(RuntimeError):
    """The residency guard halted capture: the candidate was
    not Resident (or residency could not be polled), so the
    attempt was NOT spent — the request never reached a
    serving model, and counting it against the workload's
    pre-registered attempt budget would mislabel an
    infrastructure/operator fault as a candidate result.
    Capture halts whole rather than burning the remaining
    budget against a non-serving node (SI-79, 2026-11-03:
    attempts fired against a Lost Resident after a
    mid-request server death)."""


def _hash_text(text: str) -> str:
    return "sha256:" + sha256_hex(text.encode("utf-8"))


def _now_ny() -> datetime:
    return datetime.now(ZoneInfo("America/New_York"))


def discover_served_model_id(inference_base: str, *,
                             transport=None) -> str:
    """``GET /v1/models`` on the inference base -> the served
    identifier, exactly as discovered (never invented)."""
    transport = transport or HttpTransport()
    status, body = transport(
        "GET", inference_base.rstrip("/") + "/v1/models",
        timeout_s=10.0)
    if status != 200:
        raise RuntimeError(
            f"served-model discovery failed — GET /v1/models "
            f"returned HTTP {status}")
    payload = json.loads(body.decode("utf-8"))
    model_id = payload["data"][0]["id"]
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(
            "served-model discovery failed — data[0].id is "
            f"not a non-empty string: {model_id!r}")
    return model_id


def extract_content(body_text: str) -> str:
    """Extract the §8 document text from a gateway body,
    exactly as the live proposer does (§10.3): the completion
    content when the envelope parses, else the raw body text
    (an unparseable body is the parser's problem, priced at
    scoring time — never at capture time)."""
    content = body_text
    try:
        payload = json.loads(body_text)
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        choices = payload.get("choices") or []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            if isinstance(message.get("content"), str):
                content = message["content"]
    return content


def _corpus_dict(items: list, served_model_id: str,
                 scored_count: int) -> dict:
    return {
        "corpus_id": "si79-imagine-v12-c2-v1",
        "purpose": "candidate-readiness",
        "candidate_label": served_model_id,
        "response_origin": "candidate-recorded",
        "capture_date": _now_ny().date().isoformat(),
        "fixed_n": scored_count,
        "coverage": {
            "workloads": [w[0] for w in REPLAY_WORKLOADS_V2],
            "cycles_covered": [0],
            "signals_elicited": False,
            "scoping_label": (
                "cycle-0 (first-cycle) perceptions only, five "
                "distinct-body responses per workload: capture "
                "admits and executes nothing, so no prior "
                "cycles or tool results exist and no task "
                "reaches a completion state — completion/"
                "inability signals are therefore not elicited "
                "by this corpus"),
        },
        "items": items,
    }


def _resume_fixture_paths(capture_dir: str) -> dict:
    """Paths for a resumed capture whose fixture bundle
    already exists on disk: re-use it as-is (init + tiny seed
    refuse a bundle with history, and re-creating it would
    destroy the journal the seeded items' perceptions derive
    from). The work-root fixture files are re-written only
    when missing, with the live suite's own texts."""
    bundle_dir = os.path.abspath(
        os.path.join(capture_dir, "capture-bundle"))
    work_root = os.path.abspath(
        os.path.join(capture_dir, "work"))
    if not os.path.isfile(
            os.path.join(bundle_dir, "manifest.json")):
        return live_mod.build_fixture_bundle(bundle_dir, work_root)
    os.makedirs(work_root, exist_ok=True)
    for name, text in (("note.txt", live_mod.NOTE_TEXT),
                       ("poison.txt", live_mod.POISON_TEXT)):
        path = os.path.join(work_root, name)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
    return {"bundle_dir": bundle_dir, "work_root": work_root}


def capture_corpus(capture_dir: str, *, inference_base: str,
                   residency_base: str,
                   items_per_workload: int = 5,
                   max_attempts_per_workload: int = 10,
                   timeout_s: float = CAPTURE_TIMEOUT_S,
                   transport=None, log=print,
                   on_progress=None,
                   resume_corpus=None, resume_log=None) -> tuple:
    """Run the pre-registered capture plan. Returns
    ``(corpus, capture_log)``. The plan, fixed before capture
    (see the SI-79 evidence file): per workload, attempts
    proceed sequentially until ``items_per_workload``
    distinct-body responses are accepted or
    ``max_attempts_per_workload`` attempts are spent.
    Transport failures are recorded and unscored; a captured
    body byte-identical to an already-accepted corpus body
    is logged as a duplicate capture and not entered
    (§3.3(5)); nothing is parsed or scored here.
    ``on_progress(corpus, capture_log)`` fires after every
    attempt so a long capture persists each item as it
    lands (SI-74 journaled 196–900 s per live attempt).

    Resume (SI-79, mechanical — the plan above is unchanged):
    when ``resume_corpus`` / ``resume_log`` carry a prior
    run's persisted corpus + capture log, items, capture-log
    entries, seen body hashes, the scored count, and the
    per-workload accepted AND attempt counts are seeded from
    them, and capture continues only the unfinished workloads
    within their REMAINING attempt budgets
    (``max_attempts_per_workload`` minus attempts already
    spent, per the seeded log). Completed workloads are not
    re-attempted; distinctness stays global across old + new
    bodies; a partially captured workload re-uses the task
    seq its seeded items name (one task per workload, as
    originally journaled) instead of journaling a second
    task. The served model must match the seeded corpus's
    candidate label exactly — no substitution. With no
    resume data, behaviour is exactly the original path.

    Residency guard (SI-79): each attempt polls residency
    BEFORE the attempt is spent. A poll showing any state
    other than exactly ``Resident``, or a poll that itself
    raises, raises :class:`ResidencyGuardHalt` and halts the
    whole capture — no POST, no corpus item, no capture-log
    entry, no attempt spent, because the request never
    reached a serving model. An attempt that polled Resident
    and then fails in the POST keeps the original behaviour
    exactly: recorded transport-failed, unscored, and spent."""
    transport = transport or HttpTransport()
    served_model_id = discover_served_model_id(
        inference_base, transport=transport)
    log(f"served model discovered: {served_model_id}")
    resuming = resume_corpus is not None
    if resuming:
        seeded_label = resume_corpus.get("candidate_label")
        if seeded_label != served_model_id:
            raise RuntimeError(
                f"resume refused — served model "
                f"{served_model_id!r} does not match the "
                f"seeded corpus candidate {seeded_label!r}; "
                "no substitution")
        paths = _resume_fixture_paths(capture_dir)
    else:
        paths = live_mod.build_fixture_bundle(
            os.path.join(capture_dir, "capture-bundle"),
            os.path.join(capture_dir, "work"))
    policy = live_mod.fixture_policy(paths["work_root"])
    tool_pol = resolve_tool_policy(policy)
    budgets = {
        "max_tool_cycles_per_task": int(
            tool_pol["max_tool_cycles_per_task"]),
        "tool_max_task_result_bytes": int(
            tool_pol["tool_max_task_result_bytes"]),
    }
    residency = live_mod.GaiaResidencyClient(
        residency_base, transport=transport)
    if resuming:
        items = list(resume_corpus.get("items") or [])
        capture_log = list(
            (resume_log or {}).get("entries") or [])
        scored_items = [
            i for i in items if not i.get("transport_failed")
            and i.get("body") is not None]
        seen_hashes = {_hash_text(i["body"]): i["item_id"]
                       for i in scored_items}
        scored_count = len(scored_items)
        log(f"RESUME: seeded {scored_count} scored items, "
            f"{len(items)} items total, "
            f"{len(capture_log)} capture-log entries")
    else:
        items, capture_log = [], []
        seen_hashes = {}
        scored_count = 0
    accepted_by_workload = {}
    for _i in items:
        if not _i.get("transport_failed") \
                and _i.get("body") is not None:
            _l = _i.get("workload")
            accepted_by_workload[_l] = \
                accepted_by_workload.get(_l, 0) + 1
    attempts_by_workload = {}
    for _e in capture_log:
        _l = _e.get("workload")
        attempts_by_workload[_l] = \
            attempts_by_workload.get(_l, 0) + 1
    resume_date = (resume_corpus.get("capture_date")
                   if resuming else None)

    def _make_corpus():
        corpus = _corpus_dict(items, served_model_id,
                              scored_count)
        if resume_date:
            corpus["capture_date"] = resume_date
        return corpus

    def _progress():
        if on_progress is not None:
            on_progress(
                _make_corpus(),
                {"served_model_id": served_model_id,
                 "inference_base": inference_base,
                 "bundle_dir": paths["bundle_dir"],
                 "entries": capture_log})

    for label, instruction, _injection, _script, options \
            in REPLAY_WORKLOADS_V2:
        max_bytes = options.get("max_perception_bytes")
        accepted = accepted_by_workload.get(label, 0)
        attempt = attempts_by_workload.get(label, 0)
        if resuming and accepted >= items_per_workload:
            log(f"RESUME: {label} already complete "
                f"({accepted}/{items_per_workload}) — skipped, "
                "not re-attempted")
            continue
        journal = open_journal(paths["bundle_dir"])
        try:
            seeded_task_seqs = [
                i["prompt_ref"]["task_seq"] for i in items
                if i.get("workload") == label
                and (i.get("prompt_ref") or {}).get("task_seq")
                is not None]
            if seeded_task_seqs:
                task = {"seq": seeded_task_seqs[0]}
                log(f"RESUME: {label} re-uses journaled task "
                    f"seq {task['seq']} (accepted {accepted}, "
                    f"attempts spent {attempt})")
            else:
                task = record_task_entry(
                    journal, instruction, paths["work_root"],
                    budgets=budgets, operator="si-79-capture")
            while accepted < items_per_workload \
                    and attempt < max_attempts_per_workload:
                # Residency guard (SI-79): poll BEFORE the
                # attempt is spent. Only an attempt whose poll
                # shows exactly "Resident" proceeds to the POST
                # and counts against the budget; any other
                # state, or a poll that itself fails, halts the
                # whole capture with no attempt spent, no POST,
                # no corpus item, and no capture-log entry —
                # such an attempt never reached a serving
                # model, so it is not a candidate measurement.
                try:
                    res = residency.journal_residency_for_model(
                        journal, live_mod.FIXTURE_FLEET_MODEL,
                        served_model_id)
                except Exception as exc:
                    halt_msg = (
                        f"residency guard halt — workload "
                        f"{label!r}: residency poll failed "
                        f"({type(exc).__name__}: {exc}); "
                        f"no attempt spent, because the request "
                        f"never reached a serving model")
                    log(halt_msg)
                    raise ResidencyGuardHalt(halt_msg) from exc
                residency_state = res["record"].get("state")
                if residency_state != "Resident":
                    halt_msg = (
                        f"residency guard halt — workload "
                        f"{label!r}: residency state is "
                        f"{residency_state!r}, not 'Resident'; "
                        f"no attempt spent, because the request "
                        f"never reached a serving model")
                    log(halt_msg)
                    raise ResidencyGuardHalt(halt_msg)
                attempt += 1
                item_id = f"c2-{label}-{accepted}"
                entry = {"item_id": item_id, "workload": label,
                         "attempt": attempt,
                         "started_ny": _now_ny().isoformat()}
                entry["residency_seq"] = res["seq"]
                entry["residency_state"] = residency_state
                perception = assemble_perception(
                    journal, task["seq"],
                    **({"max_perception_bytes": int(max_bytes)}
                       if max_bytes is not None else {}))
                request = build_gateway_request(
                    perception, model=served_model_id)
                request_digest = _hash_text(canonical_json(
                    request).decode("utf-8"))
                entry["request_digest"] = request_digest
                started = time.perf_counter()
                try:
                    status, raw = transport(
                        "POST",
                        inference_base.rstrip("/")
                        + GATEWAY_CHAT_PATH,
                        json_body=request, timeout_s=timeout_s)
                    latency_ms = int(
                        (time.perf_counter() - started) * 1000)
                    body_text = raw.decode(
                        "utf-8", errors="replace")
                    if status != 200:
                        raise RuntimeError(
                            f"HTTP {status} from gateway")
                except Exception as exc:  # transport failure:
                    # recorded, unscored, never in N (§3.3)
                    entry["outcome"] = "transport-failed"
                    entry["detail"] = (
                        f"{type(exc).__name__}: {exc}")
                    entry["latency_ms"] = int(
                        (time.perf_counter() - started) * 1000)
                    items.append({
                        "item_id": item_id, "workload": label,
                        "cycle": 0, "body": None,
                        "transport_failed": True,
                        "body_source": "none — transport failure, "
                                       "no response",
                        "prompt_ref": {
                            "bundle": paths["bundle_dir"],
                            "task_seq": task["seq"],
                            "request_digest": request_digest},
                        "provenance": {
                            "capture": "SI-79 live gateway "
                                       "capture (capture only; "
                                       "no admission, no "
                                       "execution, no gate "
                                       "verdict)",
                            "inference_base": inference_base,
                            "served_model_id": served_model_id,
                            "residency_seq": res["seq"],
                            "captured_at_america_new_york":
                                _now_ny().isoformat(),
                            "latency_ms": entry["latency_ms"],
                            "detail": entry["detail"],
                        }})
                    capture_log.append(entry)
                    _progress()
                    log(f"{label} attempt {attempt}: "
                        f"transport-failed ({entry['detail']})")
                    continue
                content = extract_content(body_text)
                digest = _hash_text(content)
                entry["latency_ms"] = latency_ms
                entry["http_status"] = status
                entry["response_hash"] = digest
                if digest in seen_hashes:
                    entry["outcome"] = "duplicate-capture"
                    entry["duplicate_of"] = seen_hashes[digest]
                    capture_log.append(entry)
                    _progress()
                    log(f"{label} attempt {attempt}: duplicate "
                        f"of {seen_hashes[digest]} — logged, "
                        "not entered (§3.3(5))")
                    continue
                seen_hashes[digest] = item_id
                entry["outcome"] = "captured"
                capture_log.append(entry)
                items.append({
                    "item_id": item_id, "workload": label,
                    "cycle": 0, "body": content,
                    "body_source": "verbatim-recorded",
                    "prompt_ref": {
                        "bundle": paths["bundle_dir"],
                        "task_seq": task["seq"],
                        "request_digest": request_digest,
                        "perception_bytes": perception.get(
                            "bytes")},
                    "provenance": {
                        "capture": "SI-79 live gateway capture "
                                   "(capture only; no admission, "
                                   "no execution, no gate "
                                   "verdict)",
                        "inference_base": inference_base,
                        "served_model_id": served_model_id,
                        "residency_seq": res["seq"],
                        "residency_state": res["record"].get(
                            "state"),
                        "captured_at_america_new_york":
                            _now_ny().isoformat(),
                        "latency_ms": latency_ms,
                        "http_status": status,
                        "request_digest": request_digest,
                    }})
                accepted += 1
                scored_count += 1
                _progress()
                log(f"{label}: captured {accepted}/"
                    f"{items_per_workload} "
                    f"({latency_ms} ms, {digest[:19]}…)")
        finally:
            journal.close()
    corpus = _make_corpus()
    return corpus, {"served_model_id": served_model_id,
                    "inference_base": inference_base,
                    "bundle_dir": paths["bundle_dir"],
                    "entries": capture_log}


def dc1_coverage_missing(corpus: dict) -> list:
    """The D-C1 coverage requirements the corpus does not
    meet (empty = covered): N >= 30 scored, >= 5 scored per
    each of the six L-1 v2 workloads."""
    missing = []
    scored = [i for i in corpus.get("items") or []
              if not i.get("transport_failed")
              and i.get("body") is not None]
    if len(scored) < D_C1["min_n"]:
        missing.append({
            "requirement": "D-C1", "name": "dc1-min-n",
            "detail": f"{len(scored)} scored responses < "
                      f"D-C1 minimum N {D_C1['min_n']}"})
    per_workload = {}
    for item in scored:
        key = item.get("workload")
        per_workload[key] = per_workload.get(key, 0) + 1
    for label, *_rest in REPLAY_WORKLOADS_V2:
        if per_workload.get(label, 0) < D_C1["min_per_workload"]:
            missing.append({
                "requirement": "D-C1",
                "name": "dc1-min-per-workload",
                "detail": f"workload {label!r} carries "
                          f"{per_workload.get(label, 0)} scored "
                          f"responses < D-C1 minimum "
                          f"{D_C1['min_per_workload']}"})
    return missing


def build_c2_record(corpus: dict, *, si_commit=None,
                    argv=None) -> dict:
    """Build the SI-79 §3.4 record: the fold comes from the
    C-1 harness untouched (parser unmodified); this glue sets
    only the slice identity, the capture provenance, and the
    readiness comparison against the DECIDED D-C1 threshold.
    A corpus failing §3.3 (harness verdict) or the D-C1
    coverage requirements yields INSUFFICIENT CORPUS and NO
    verdict. ``readiness.unlocks`` stays ``[]``."""
    record = build_conformance_record(
        corpus, si_commit=si_commit, argv=argv)
    record["slice"] = "SI-79 (SI-76 §4.4 C-2)"
    record["fingerprint"]["capture"] = {
        "module": "si.gaia.conformance_capture",
        "corpus_store": CORPUS_STORE,
        "candidate_decision": ("D-C3 decided by Mark, "
                               "2026-11-02: Imagine v12, the "
                               "current Resident"),
        "run_authorisation": ("D-C2 as amended by Mark, "
                              "2026-11-02: live conformance "
                              "runs fire on the loop's "
                              "initiative; Mark spectates"),
    }
    record["honesty"]["capture_note"] = (
        "live_attempts counts the offline scoring step only "
        "(§3.4); the corpus itself was captured by live "
        "gateway calls to the already-running local node — "
        "counted in the capture section of the SI-79 slice "
        "doc; capture admitted no intent, executed no tool, "
        "and wrote no gate verdict")
    coverage_missing = dc1_coverage_missing(corpus)
    readiness = {"threshold": dict(D_C1), "verdict": None,
                 "unlocks": []}
    if record.get("status") == "scored" and coverage_missing:
        record["status"] = "INSUFFICIENT CORPUS"
        record["missing_requirements"] = coverage_missing
        readiness["detail"] = (
            "corpus satisfies §3.3 but not the D-C1 coverage "
            "requirements — INSUFFICIENT CORPUS, no verdict")
    elif record.get("status") == "scored":
        results = record["results"]
        rate = results["rate"]["decimal"]
        floors_ok = all(
            b["rate"] and b["rate"]["decimal"]
            >= D_C1["per_workload_floor"]
            for b in results["per_workload"].values())
        verdict = ("MEETS THRESHOLD"
                   if rate >= D_C1["rate_min"] and floors_ok
                   else "BELOW THRESHOLD")
        readiness["verdict"] = verdict
        readiness["detail"] = (
            f"rate {results['rate']['fraction']} "
            f"({rate}) vs decided rate_min "
            f"{D_C1['rate_min']}; per-workload floor "
            f"{D_C1['per_workload_floor']} "
            f"{'met by every workload' if floors_ok else 'failed by at least one workload'}; "
            f"comparison against {D_C1_DECISION}")
    else:
        readiness["detail"] = (
            "INSUFFICIENT CORPUS (§3.3) — no rate, no verdict")
    record["readiness"] = readiness
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="si.gaia.conformance_capture",
        description="SI-79 C-2 capture + record glue "
                    "(capture only; scoring via the C-1 "
                    "harness)")
    parser.add_argument("--out", required=True,
                        help="output directory for the corpus "
                             "file, capture log, and record")
    parser.add_argument("--capture-dir", default=None,
                        help="capture bundle/work directory "
                             "(default: <out>/capture)")
    parser.add_argument("--inference-base", default=None,
                        help="default: discovered from the "
                             "gaia log (SI-67 discovery)")
    parser.add_argument("--residency-base", default=None)
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--score-only", action="store_true",
                        help="score an existing corpus file in "
                             "--out without capturing")
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    corpus_path = os.path.join(args.out, "SI-79-corpus.json")
    if args.score_only:
        with open(corpus_path, encoding="utf-8") as fh:
            corpus = json.load(fh)
    else:
        inference_base = args.inference_base or \
            live_mod.discover_inference_base()
        residency_base = args.residency_base or \
            live_mod.residency_base()
        def _persist(corpus_so_far, log_so_far):
            with open(corpus_path, "w", encoding="utf-8") as fh:
                json.dump(corpus_so_far, fh, indent=2)
            with open(os.path.join(
                    args.out, "capture-log.json"), "w",
                    encoding="utf-8") as fh:
                json.dump(log_so_far, fh, indent=2)

        # Resume: a prior corpus + capture log in --out seed
        # the capture (see capture_corpus); the first
        # _persist only fires after seeding, so the seeded
        # items are never overwritten by an empty corpus.
        resume_corpus = resume_log = None
        capture_log_path = os.path.join(
            args.out, "capture-log.json")
        if os.path.isfile(corpus_path) \
                and os.path.isfile(capture_log_path):
            with open(corpus_path, encoding="utf-8") as fh:
                resume_corpus = json.load(fh)
            with open(capture_log_path, encoding="utf-8") as fh:
                resume_log = json.load(fh)
            print(f"RESUME: seeding from {corpus_path} "
                  f"(fixed_n={resume_corpus.get('fixed_n')}) "
                  f"and {capture_log_path}")

        corpus, capture_log = capture_corpus(
            args.capture_dir or os.path.join(args.out, "capture"),
            inference_base=inference_base,
            residency_base=residency_base,
            on_progress=_persist,
            resume_corpus=resume_corpus, resume_log=resume_log)
        with open(corpus_path, "w", encoding="utf-8") as fh:
            json.dump(corpus, fh, indent=2)
        with open(os.path.join(args.out, "capture-log.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(capture_log, fh, indent=2)
        print(json.dumps({
            "captured_scored_items": corpus["fixed_n"],
            "corpus": corpus_path}, indent=2))
        if args.capture_only:
            return 0
    missing = validate_corpus(corpus)
    record = build_c2_record(corpus, argv=sys.argv)
    record_path = os.path.join(args.out, "SI-79-measurements.json")
    with open(record_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, default=str)
    results = record.get("results") or {}
    print(json.dumps({
        "status": record["status"],
        "validation_missing": missing,
        "n": results.get("n"), "parse_ok": results.get(
            "parse_ok"), "rate": results.get("rate"),
        "readiness": record["readiness"],
        "live_attempts": record["live_attempts"]}, indent=2,
        default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
