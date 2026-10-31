"""Gaia seam Flow 3, SI side: outcome return + reconciliation
(G-9 §11, §12.5, §15 S-5; slice SI-59).

S-3 (:mod:`si.gaia.seam_submit`) submits an ADMITted seam edit
to the gaia fork and journals the submission (kind
``gaia_work``); until this slice the item then rested
``submitted/awaiting outcome`` forever, because nothing came
back. This module is the return leg, built exactly as G-9 §11
shapes it — **pull + reconciler together** (§11.3: "S-5 ships
pull + reconciler together or not at all"):

- **The poller (§11.2).** :func:`poll_seam_outcomes` takes
  every journaled submission with no journaled outcome and
  polls ``GET /gaia/v1/work/{id}`` (the fork's as-built status
  route, G-9 §12.2) for it. Only a terminal answer — HTTP 200,
  ``state`` ``executed``, and a ``result`` object carrying the
  fork's actuation result (``outcome`` one of
  ``executed`` / ``refused`` / ``failed``, per the fork's
  ``finish`` shape) — is an outcome. On a terminal answer the
  outcome is journaled **once, keyed by ``work_id``**, as an
  ``experiences`` row of kind ``tool_result`` whose record is
  the §12.5 SI-side envelope: ``source`` ``gaia-seam``, the
  seam pair ``(work_id, journal_seq)``, ``gate_verdict``
  ``ADMIT``, the fork result **verbatim** under
  ``fork_result``, and under ``execution`` the same fork
  result plus SI's one fold key ``status`` mirroring
  ``outcome`` (the vocabulary :func:`si.gaia.queue._apply_outcome`
  reads — so the queue fold learns a seam outcome through the
  same path SI's own executor outcomes take, and the row's
  ``self_edit_id`` is the intent's edit, the exact key
  ``build_provenance_trail`` collects outcomes by, unmodified).
  A repeat poll finding the ``work_id`` already journaled is
  a no-op (§7.2). Non-terminal states, 404s, other statuses,
  malformed bodies, and transport failures journal **no**
  outcome and are reported as data — a retry is free.
- **The reconciler (§11.3).** :func:`reconcile_seam_outcomes`
  tails the fork's ``outcomes.jsonl`` **read-only** (the file
  is only ever opened for reading; §7.3) from a caller-held
  byte-offset ``cursor``. Each record (the fork
  ``OutcomeRecord`` JSON: ``work_id, tool, model, outcome,
  refusal, detail, caused_by, spine_event_refs, result``) is
  matched by ``work_id`` against SI's journal: already
  journaled → the result values are cross-checked (agreement
  is a counted no-op; **disagreement is journaled as an
  integrity fault**, kind ``gaia_outcome``, never silently
  resolved); not journaled → journaled (this is the
  post-restart recovery path: the fork lost its in-memory
  work record, the outcome file survived). A *distinct second
  outcome* for one ``work_id`` is likewise journaled as an
  integrity fault (§11.2). The returned cursor advances only
  past records that were journaled, cross-checked, faulted,
  or counted unmatched (a ``work_id`` SI never submitted —
  counted, never journaled as an outcome: SI claims no intent
  for it). A torn or unparseable line stops the tail at that
  line's offset, is counted, and is retried on the next call —
  never fatal, never silent. The cursor itself is the
  caller's to persist (returned, never written by this
  module — no new store is invented for it).
- **Outcome-unknown (§11.4).** :func:`mark_seam_outcome_unknown`
  is the explicit operator-called act for the dangerous case
  (the fork may have actuated; SI has no outcome): it verifies
  a journaled submission exists for the ``work_id`` and that
  no outcome is journaled for it, then journals the intent
  ``outcome-unknown`` with the existing SI-50 §13.3 writer
  (:func:`si.gaia.queue.record_queue_outcome_unknown`, naming
  the explicit pair ``(work_id, journal_seq, caused_by)`` in
  the result and the evidence as a
  ``queue_recovery_observation`` — evidence, never an
  outcome). The intent is **never re-executed and never
  resubmitted** by this module; S-3 dedup already returns a
  journaled submission rather than submitting twice.
  :func:`seam_reproposal_blocked` reports, from the fold,
  whether a ``caused_by`` is cited by an undisposed seam
  outcome-unknown (the §11.4 block as a journal fact).
  **Disposition pin (§15 S-5, stated, no third option):**
  dispositions remain the existing SI-50 §13.4 operator acts —
  :func:`si.gaia.queue.disposition_outcome_unknown` (already
  exposed by the B-5 service) — applied to the seam intent
  this module marks; **S-5 adds no new disposition writer.**

**The fork-side contract relied on — and nothing beyond it.**
The status route and both record shapes are G-9's documented
as-built fork surface (§11.2/§12.2, §4.4). The work base URL
is **supplied by the caller/operator** (``work_url`` or a
transport): the per-work status URL is that URL plus
``/{work_id}`` — the §12.2 route applied to the operator's
own base, no host invented or defaulted. Transports are
:mod:`si.gaia.hearth`'s, honestly labelled (the B-4 pattern):
**no live fork endpoint is exercised by this slice** — live
interop and outcome latency (poll-based) are **UNMEASURED**;
an env-gated live test (``SI_S5_WORK_URL``) follows the
S-3/S-4 pattern and skips when unset.

**Ordering claims (§11.5).** Claimed: at most one journaled
outcome per ``work_id``; every fork outcome the reconciler
can read is eventually journaled or faulted; outcome
journaling never precedes the fork's own outcome record (it
is derived from it). NOT claimed: outcome latency, cross-work
ordering (outcomes journal in discovery order).

Performance is **UNMEASURED**. CPU only; no secrets; loopback
/ operator-configured endpoints only (G-9 §14, unchanged).
"""
from __future__ import annotations

import json
import os
import threading

from si.bundle import _anchor_checkpoint, _write_manifest, load_bundle
from si.gaia.hearth import HttpTransport
from si.gaia.queue import (
    fold_queue,
    record_queue_outcome_unknown,
    record_queue_recovery_observation,
)
from si.journal import open_journal

#: Journaled ``returned_via`` of every S-5 outcome row.
OUTCOME_RETURNED_VIA = "gaia-seam-outcome (G-9/S-5)"

#: The documented G-9 §12.2 work route (as-built in the gaia
#: fork's gaia-api); ``GET {work_url}/{work_id}`` is its status
#: form (§11.2). Only the path is named — a host is always
#: operator-supplied, never defaulted.
WORK_PATH = "/gaia/v1/work"

#: The fork actuation outcomes (its ``finish`` shape, §12.4).
FORK_OUTCOMES = ("executed", "refused", "failed")

#: Scope/honesty label carried on every S-5 result.
OUTCOME_RETURN_LABEL: str = (
    "seam outcome return is live SI-side (G-9 §11/§12.5, "
    "S-5/SI-59): the poller journals a terminal fork status "
    "as a tool_result seam outcome exactly once per work_id "
    "(§12.5 envelope, fork result verbatim, row keyed by the "
    "intent's edit so provenance trails collect it "
    "unmodified), and the read-only outcomes.jsonl reconciler "
    "journals outcomes the poller missed, cross-checks ones "
    "it has, and journals disagreements / distinct second "
    "outcomes as integrity faults; outcome-unknown is an "
    "explicit operator act in the SI-50 §13.3 shape, "
    "dispositioned only by the existing §13.4 acts (no new "
    "disposition writer in S-5); no live fork endpoint was "
    "exercised by this slice (transports are labelled "
    "replay/stub or loopback test double unless an operator "
    "supplies a real URL); live interop, outcome latency, and "
    "performance UNMEASURED"
    " Measured under S-7 (SI-62): median 1139 ms (range 1087–1648, N=10) for Q-D1 outcome poll pass (10 outstanding, all terminal) on the miniature fixture, embedded journal, Class L transport, host fingerprint in docs/S-7-seam-measurements.md; single-run; live fork interop performance remains UNMEASURED. Measured under S-7 (SI-62): median 0.76 s (range 0.75–0.77, N=5) for conditional outcome latency at the stated schedule (offset 0 s, period 1 s, terminal at +0.35 s) on the miniature fixture, embedded journal, Class L transport, host fingerprint in docs/S-7-seam-measurements.md; single-run; unconditional outcome latency and live fork interop remain UNMEASURED.")

#: Serialises scan → transport/file → journal per process.
_OUTCOME_LOCK = threading.RLock()


# -- reading the journal -------------------------------------------------------

def _submissions(journal) -> list[dict]:
    """Every journaled S-3 submission row, in journal order."""
    out = []
    for row in journal.backend.find(
            "experiences", filter={"kind": "gaia_work"}) or []:
        record = row.get("record") or {}
        if record.get("status") == "submitted" \
                and record.get("fork_work_id"):
            out.append(row)
    return sorted(out, key=lambda r: r.get("seq") or 0)


def _journaled_outcomes(journal) -> dict:
    """``{work_id: row}`` of journaled S-5 ``tool_result`` rows."""
    out = {}
    for row in journal.backend.find(
            "experiences", filter={"kind": "tool_result"}) or []:
        record = row.get("record") or {}
        if record.get("source") == "gaia-seam" \
                and record.get("work_id"):
            out.setdefault(record["work_id"], row)
    return out


def _journaled_faults(journal) -> set:
    """``{(work_id, line_offset)}`` of journaled integrity faults."""
    out = set()
    for row in journal.backend.find(
            "experiences", filter={"kind": "gaia_outcome"}) or []:
        record = row.get("record") or {}
        if record.get("status") == "integrity-fault":
            out.add((record.get("work_id"),
                     record.get("line_offset")))
    return out


def _status_url(work_url: str, work_id: str) -> str:
    return work_url.rstrip("/") + "/" + work_id


def _terminal_result(body) -> dict | None:
    """The fork result verbatim from a terminal status body.

    Terminal is exactly: HTTP 200's body is an object with
    ``state`` ``executed`` and a ``result`` object whose
    ``outcome`` is one of the fork's actuation outcomes and
    whose ``tool`` is a string. Anything else is not an
    outcome — never guessed into one (§13.3).
    """
    if not isinstance(body, dict):
        return None
    if body.get("state") != "executed":
        return None
    result = body.get("result")
    if not isinstance(result, dict):
        return None
    if result.get("outcome") not in FORK_OUTCOMES:
        return None
    if not isinstance(result.get("tool"), str):
        return None
    return result


# -- journaling one outcome ----------------------------------------------------

def _journal_outcome(bundle_dir, sub_row, fork_result, *, via,
                     transport_label) -> dict:
    """Journal one outcome for one submission (§12.5 envelope).

    Reloads the bundle and re-checks dedup first: if the
    ``work_id`` is already journaled, returns it deduped and
    journals nothing (§7.2 no-op). Re-anchors the manifest
    after journaling, as S-3/S-4 do.
    """
    record_in = sub_row.get("record") or {}
    work_id = record_in.get("fork_work_id")
    edit_id = record_in.get("edit_id")
    bundle_dir = os.path.abspath(bundle_dir)
    manifest = load_bundle(bundle_dir)
    journal = open_journal(bundle_dir, manifest)
    try:
        existing = _journaled_outcomes(journal).get(work_id)
        if existing is not None:
            return {"status": "journaled", "deduped": True,
                    "work_id": work_id, "edit_id": edit_id,
                    "outcome_seq": existing.get("seq"),
                    "outcome": (existing.get("record") or {})
                    .get("fork_result", {}).get("outcome"),
                    "label": OUTCOME_RETURN_LABEL}
        record = {
            "kind": "tool_result",
            "source": "gaia-seam",
            "intent_id": edit_id,
            "work_id": work_id,
            "journal_seq": record_in.get("self_edit_journal_seq"),
            "gate_verdict": "ADMIT",
            "tool": fork_result.get("tool"),
            "caused_by": record_in.get("caused_by"),
            "fork_result": fork_result,
            "execution": {**fork_result,
                          "status": fork_result.get("outcome")},
            "via": via,
            "transport": transport_label,
            "returned_via": OUTCOME_RETURNED_VIA,
            "label": OUTCOME_RETURN_LABEL,
        }
        journal.record_experience("tool_result", record, edit_id)
        row = _journaled_outcomes(journal).get(work_id)
        updated = _anchor_checkpoint(
            journal, manifest, manifest["head"],
            note=f"gaia seam outcome: work {work_id} for edit "
                 f"{edit_id} journaled via {via} (S-5; fork "
                 f"outcome {fork_result.get('outcome')!r})")
        _write_manifest(bundle_dir, updated)
        return {"status": "journaled", "deduped": False,
                "work_id": work_id, "edit_id": edit_id,
                "outcome_seq": row.get("seq") if row else None,
                "outcome": fork_result.get("outcome"),
                "label": OUTCOME_RETURN_LABEL}
    finally:
        journal.close()


def _journal_fault(bundle_dir, sub_row, fork_line, *, line_offset,
                   journaled_result, via) -> dict:
    """Journal one integrity fault (§11.2/§11.3): an outcome
    source disagrees with the journaled outcome for a
    ``work_id``, or a distinct second outcome exists for it.
    The journaled outcome is never rewritten."""
    record_in = sub_row.get("record") or {}
    work_id = record_in.get("fork_work_id")
    bundle_dir = os.path.abspath(bundle_dir)
    manifest = load_bundle(bundle_dir)
    journal = open_journal(bundle_dir, manifest)
    try:
        record = {
            "kind": "gaia_outcome",
            "status": "integrity-fault",
            "work_id": work_id,
            "edit_id": record_in.get("edit_id"),
            "journal_seq": record_in.get("self_edit_journal_seq"),
            "line_offset": line_offset,
            "journaled_result": journaled_result,
            "conflicting_line": fork_line,
            "via": via,
            "returned_via": OUTCOME_RETURNED_VIA,
            "label": OUTCOME_RETURN_LABEL,
        }
        journal.record_experience(
            "gaia_outcome", record, record_in.get("edit_id") or "")
        updated = _anchor_checkpoint(
            journal, manifest, manifest["head"],
            note=f"gaia seam outcome integrity fault: work "
                 f"{work_id} (S-5; journaled outcome stands)")
        _write_manifest(bundle_dir, updated)
        return {"status": "integrity-fault", "work_id": work_id,
                "edit_id": record_in.get("edit_id"),
                "line_offset": line_offset,
                "label": OUTCOME_RETURN_LABEL}
    finally:
        journal.close()


# -- the poller (§11.2) ----------------------------------------------------------

def poll_seam_outcomes(bundle_dir: str, *,
                       work_url: str | None = None,
                       transport=None,
                       timeout_s: float = 30.0) -> dict:
    """§11.2: poll every open seam submission once.

    For each journaled submission with no journaled outcome,
    GETs ``{work_url}/{work_id}`` through the injected
    transport (or the real ``HttpTransport`` against the
    operator-supplied ``work_url``). Terminal answers journal
    their outcome; everything else is reported per work item
    (``awaiting`` / ``not-found`` / ``poll-failed``) with
    nothing journaled. Already-journaled submissions are not
    polled at all (the §7.2 no-op is free).
    """
    if transport is None and not work_url:
        return {"status": "refused", "journaled": False,
                "problems": [
                    "no transport injected and no work_url "
                    "given — there is no default fork endpoint; "
                    "live interop is UNMEASURED"],
                "label": OUTCOME_RETURN_LABEL}
    if transport is None:
        transport = HttpTransport()
    transport_label = getattr(transport, "label", "http")
    with _OUTCOME_LOCK:
        bundle_dir = os.path.abspath(bundle_dir)
        manifest = load_bundle(bundle_dir)
        journal = open_journal(bundle_dir, manifest)
        try:
            done = _journaled_outcomes(journal)
            open_subs = [r for r in _submissions(journal)
                         if (r.get("record") or {})
                         .get("fork_work_id") not in done]
        finally:
            journal.close()
        journaled, awaiting, not_found, failed = [], [], [], []
        for sub in open_subs:
            record_in = sub.get("record") or {}
            work_id = record_in.get("fork_work_id")
            url = _status_url(work_url, work_id) if work_url else ""
            try:
                status, raw = transport(
                    "GET", url, timeout_s=timeout_s)
            except Exception as e:  # failure is data, not a claim
                failed.append({
                    "work_id": work_id, "status": "poll-failed",
                    "failure": f"{type(e).__name__}: {e}"})
                continue
            try:
                body = json.loads(raw.decode("utf-8")) \
                    if isinstance(raw, (bytes, bytearray)) else raw
            except (ValueError, UnicodeDecodeError):
                body = None
            if status == 404:
                not_found.append({"work_id": work_id,
                                  "status": "not-found"})
                continue
            if status != 200:
                failed.append({"work_id": work_id,
                               "status": "poll-failed",
                               "http_status": status,
                               "response": body})
                continue
            result = _terminal_result(body)
            if result is None:
                awaiting.append({
                    "work_id": work_id, "status": "awaiting",
                    "state": body.get("state")
                    if isinstance(body, dict) else None})
                continue
            journaled.append(_journal_outcome(
                bundle_dir, sub, result, via="poll",
                transport_label=transport_label))
        return {"status": "polled", "journaled": journaled,
                "awaiting": awaiting, "not_found": not_found,
                "failed": failed,
                "count": len(journaled),
                "transport": transport_label,
                "label": OUTCOME_RETURN_LABEL}


# -- the reconciler (§11.3) ------------------------------------------------------

def _valid_outcome_line(value) -> bool:
    """A recognisable fork ``OutcomeRecord`` JSON line: an
    object with a string ``work_id`` and a ``result`` object
    carrying a fork outcome. Anything else is not a record —
    torn and malformed lines share this verdict (§11.3)."""
    if not isinstance(value, dict):
        return False
    if not isinstance(value.get("work_id"), str) \
            or not value.get("work_id"):
        return False
    result = value.get("result")
    return isinstance(result, dict) \
        and result.get("outcome") in FORK_OUTCOMES


def reconcile_seam_outcomes(bundle_dir: str, outcomes_path: str,
                            *, cursor: int = 0) -> dict:
    """§11.3: tail the fork's ``outcomes.jsonl``, read-only.

    Reads from byte offset ``cursor`` (the caller's persisted
    cursor from the previous call; 0 tails from the start).
    Returns the new ``cursor`` alongside per-category counts
    and entries. The file is opened read-only and never
    written; a missing file is a refusal, not an empty tail.
    """
    if not os.path.isfile(outcomes_path):
        return {"status": "refused", "journaled": False,
                "cursor": cursor,
                "problems": [
                    f"no outcomes file at {outcomes_path!r} — "
                    "the reconciler tails the fork's durable "
                    "outcomes.jsonl (operator-supplied path); "
                    "nothing was read or journaled"],
                "label": OUTCOME_RETURN_LABEL}
    with _OUTCOME_LOCK:
        bundle_dir = os.path.abspath(bundle_dir)
        manifest = load_bundle(bundle_dir)
        journal = open_journal(bundle_dir, manifest)
        try:
            subs = {(r.get("record") or {}).get("fork_work_id"): r
                    for r in _submissions(journal)}
            done = _journaled_outcomes(journal)
            faults = _journaled_faults(journal)
        finally:
            journal.close()
        with open(outcomes_path, "rb") as fh:
            fh.seek(cursor)
            data = fh.read()
        journaled, checked, faulted, unmatched = [], [], [], []
        torn = 0
        pos = cursor
        offset = 0
        lines = data.split(b"\n")
        for idx, raw_line in enumerate(lines):
            is_last = idx == len(lines) - 1
            line_start = pos
            if is_last and raw_line == b"":
                break  # clean EOF after the final newline
            if is_last and not data.endswith(b"\n"):
                torn += 1  # torn final line: retried next tail
                break
            pos += len(raw_line) + 1
            offset = pos
            text = raw_line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except ValueError:
                value = None
            if not _valid_outcome_line(value):
                torn += 1  # unparseable: stop here, retry later
                pos = line_start
                offset = line_start
                break
            work_id = value["work_id"]
            result = value["result"]
            sub = subs.get(work_id)
            if sub is None:
                unmatched.append({"work_id": work_id,
                                  "line_offset": line_start})
                continue
            existing = done.get(work_id)
            if existing is not None:
                journaled_result = (existing.get("record") or {}) \
                    .get("fork_result")
                if journaled_result == result:
                    checked.append({"work_id": work_id,
                                    "line_offset": line_start})
                    continue
                if (work_id, line_start) in faults:
                    continue
                out = _journal_fault(
                    bundle_dir, sub, value, line_offset=line_start,
                    journaled_result=journaled_result,
                    via="reconciler")
                faults.add((work_id, line_start))
                faulted.append(out)
                continue
            out = _journal_outcome(
                bundle_dir, sub, result, via="reconciler",
                transport_label="file:outcomes.jsonl (read-only)")
            done[work_id] = {"record": {"fork_result": result}}
            journaled.append(out)
        return {"status": "reconciled", "cursor": offset,
                "journaled": journaled, "cross_checked": checked,
                "faults": faulted, "unmatched": unmatched,
                "torn": torn, "count": len(journaled),
                "label": OUTCOME_RETURN_LABEL}


# -- outcome-unknown (§11.4) -----------------------------------------------------

def mark_seam_outcome_unknown(bundle_dir: str, work_id: str, *,
                              cause: str,
                              evidence: list | None = None) -> dict:
    """§11.4: journal one submitted intent ``outcome-unknown``.

    An explicit operator act, fail-closed: an unknown
    ``work_id``, or one whose outcome is already journaled, is
    refused with nothing journaled; a repeat mark returns the
    existing entry deduped. The evidence consulted (poll /
    reconciler / spine, in §11.4's order) is journaled as a
    ``queue_recovery_observation`` — evidence, never an
    outcome. Nothing here re-executes or resubmits anything.
    Dispositions are the existing SI-50 §13.4 acts (the pin in
    this module's docstring), not a writer added here.
    """
    with _OUTCOME_LOCK:
        bundle_dir = os.path.abspath(bundle_dir)
        manifest = load_bundle(bundle_dir)
        journal = open_journal(bundle_dir, manifest)
        try:
            sub = next((r for r in _submissions(journal)
                        if (r.get("record") or {})
                        .get("fork_work_id") == work_id), None)
            if sub is None:
                return {"status": "refused", "journaled": False,
                        "problems": [
                            f"no journaled seam submission for "
                            f"work_id {work_id!r} — outcome-unknown "
                            "names a submitted intent, never an "
                            "invented one"],
                        "label": OUTCOME_RETURN_LABEL}
            record_in = sub.get("record") or {}
            edit_id = record_in.get("edit_id")
            if work_id in _journaled_outcomes(journal):
                return {"status": "refused", "journaled": False,
                        "problems": [
                            f"work_id {work_id!r} already has a "
                            "journaled outcome — it is not "
                            "unknown"],
                        "label": OUTCOME_RETURN_LABEL}
            fold = fold_queue(journal)
            state = fold.intents.get(edit_id)
            if state is not None \
                    and state.state == "outcome-unknown":
                return {"status": "outcome-unknown",
                        "deduped": True, "work_id": work_id,
                        "edit_id": edit_id,
                        "intent_seq": state.intent_seq,
                        "outcome_unknown_seq":
                            state.outcome_unknown_seq,
                        "label": OUTCOME_RETURN_LABEL}
            payload = record_in.get("payload") or {}
            task_seq = payload.get("task_seq")
            intent_seq = record_in.get("self_edit_journal_seq")
            entry = record_queue_outcome_unknown(
                journal, task_seq, edit_id, intent_seq, cause,
                claim_seq=None)
            if evidence:
                record_queue_recovery_observation(
                    journal, task_seq, edit_id, intent_seq,
                    "seam outcome-unknown evidence (G-9 §11.4)",
                    detail=json.dumps(evidence, default=str))
            updated = _anchor_checkpoint(
                journal, manifest, manifest["head"],
                note=f"gaia seam outcome-unknown: work {work_id} "
                     f"for edit {edit_id} (S-5, §11.4; never "
                     f"re-executed, never resubmitted)")
            _write_manifest(bundle_dir, updated)
            return {"status": "outcome-unknown", "deduped": False,
                    "work_id": work_id, "edit_id": edit_id,
                    "intent_seq": intent_seq,
                    "journal_seq": intent_seq,
                    "caused_by": record_in.get("caused_by"),
                    "outcome_unknown_seq": entry.get("seq"),
                    "evidence": evidence or [],
                    "disposition": (
                        "manual via the existing SI-50 §13.4 "
                        "operator acts "
                        "(si.gaia.queue.disposition_outcome_unknown)"
                        " — S-5 adds no disposition writer"),
                    "label": OUTCOME_RETURN_LABEL}
        finally:
            journal.close()


def seam_reproposal_blocked(bundle_dir: str, caused_by) -> bool:
    """§11.4, as a journal fact: True when a seam intent cited
    by ``caused_by`` folds ``outcome-unknown`` undisposed.

    Pure read (folds the journal, writes nothing). Ingest
    enforcement is not changed by S-5: the existing §8 dedup
    already returns the original ACK for a re-proposal of the
    same ``caused_by``; this reports the block the fold shows.
    """
    bundle_dir = os.path.abspath(bundle_dir)
    manifest = load_bundle(bundle_dir)
    journal = open_journal(bundle_dir, manifest)
    try:
        fold = fold_queue(journal)
        subs = {(r.get("record") or {}).get("edit_id")
                for r in _submissions(journal)
                if (r.get("record") or {}).get("caused_by")
                == caused_by}
        for edit_id in subs:
            state = fold.intents.get(edit_id)
            if state is not None \
                    and state.state == "outcome-unknown" \
                    and state.disposition is None:
                return True
        return False
    finally:
        journal.close()


__all__ = [
    "FORK_OUTCOMES",
    "OUTCOME_RETURNED_VIA",
    "OUTCOME_RETURN_LABEL",
    "WORK_PATH",
    "mark_seam_outcome_unknown",
    "poll_seam_outcomes",
    "reconcile_seam_outcomes",
    "seam_reproposal_blocked",
]
