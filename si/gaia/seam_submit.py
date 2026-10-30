"""Gaia seam Flow 2, SI side: work submission (G-9 §9, slice S-3 / SI-56).

S-2a (``si.gaia.seam``) gates and journals fork watcher proposals;
S-2b registered the ``hearth.*`` seam-crossing tools, so under an
``external: admit`` policy a proposal ADMITs and its self-edit rests
``pending`` with execution label
:data:`si.gate.TOOL_SEAM_PENDING_EXECUTION`. This module is what
spends such an ADMIT: it submits the journaled intent across the
seam to the gaia fork's work-submission surface and journals the
submission in the NEDB journal.

**The fork-side contract relied on — and nothing beyond it.** The
fork's gaia-api accepts work submissions and answers **202 with a
stable work ID** (the contract stated for this slice; the SI repo
documents no endpoint path and no response field name for it).
Consequences, by construction:

- The submission URL is **supplied by the caller/operator**
  (``submission_url`` or a transport) — this module invents no
  fork path.
- A response is a confirmed submission only when it is HTTP 202
  **and** its JSON body carries a work ID under one of the
  spellings checked by :func:`_extract_work_id` (``work_id``,
  ``id``, ``workId``); the body is journaled verbatim alongside
  the extracted ID and the field it came from. A 202 without a
  work ID is ``submission-unconfirmed`` and is journaled as no
  submission at all — never faked into one.
- Any other status, or a transport failure, is
  ``submission-failed``: nothing is journaled, the edit still
  rests ``pending``, and a retry is free because SI recorded no
  submission.

**Transports — honestly labelled (the B-4 pattern).**
:mod:`si.gaia.hearth`'s transports are reused verbatim: the real
``HttpTransport`` when a ``submission_url`` is given, or any
injected callable (tests use the labelled ``ReplayTransport``
and a loopback HTTP test double). **No live fork endpoint is
exercised by this slice**: live interop with a real gaia-api is
**UNMEASURED**, and every result carries its transport label so
a replay/loopback submission can never be read as a fork one.
An env-gated live test (``SI_S3_WORK_SUBMISSION_URL``) follows
the S-2a egress-probe pattern and skips cleanly when unset.

**Journaling.** A confirmed submission is journaled as an
``experiences`` row of kind ``gaia_work`` (the kind S-2a's tests
anticipated as S-3's) via ``Journal.record_experience`` — no new
collection, no schema change — carrying the edit id, self-edit
seq, tool, model, ``caused_by``, fork work ID, 202 status, and
transport label. The manifest is then re-anchored, as seam
ingest does after journaling. The self-edit's own ``result``
stays ``pending``: submission is not an outcome.

**State, plainly.** A submitted item's state is
:data:`SUBMITTED_STATE` — never ``completed``: **S-5 outcome
return is UNBUILT** and **S-4 leash crossing is UNBUILT** (no
leash record crosses; fork leash state is not consulted).
Ordering note: the transport call precedes journaling (the work
ID does not exist before the fork answers), so a crash between
a 202 and the journal write could orphan fork-side work SI has
no record of — a known window of this slice, named not hidden;
SI-side dedup covers every journaled submission.

**Eligibility and dedup.** Only a self-edit that is exactly a
spendable seam ADMIT is submitted: gate verdict ``ADMIT``,
``result`` ``pending``, tool in
:data:`~si.gaia.registry.SEAM_CROSSING_TOOLS`, and the journaled
gate report's execution label equal to
``TOOL_SEAM_PENDING_EXECUTION``. Anything else is refused before
any transport call, journaling nothing. A second submission
attempt for an edit with a journaled ``gaia_work`` submission
returns that submission (``deduped`` True) and calls no
transport — one edit, one submission, one work ID.

Performance is **UNMEASURED**. CPU only; no secrets; loopback /
operator-configured endpoints only (G-9 §14 posture, unchanged).
"""
from __future__ import annotations

import json
import os
import threading

from si.bundle import _anchor_checkpoint, _write_manifest, load_bundle
from si.gaia.hearth import HttpTransport
from si.gaia.registry import SEAM_CROSSING_TOOLS
from si.gate import TOOL_SEAM_PENDING_EXECUTION
from si.journal import open_journal

#: Journaled ``submitted_via`` of every S-3 submission row.
WORK_SUBMITTED_VIA = "gaia-seam-submit (G-9/S-3)"

#: The one state a submitted work item is ever in, in this
#: slice: submitted across the seam, awaiting an outcome that
#: S-5 (unbuilt) would return. Never "completed".
SUBMITTED_STATE = "submitted/awaiting outcome (S-5 unbuilt)"

#: Scope/honesty label carried on every submission result.
WORK_SUBMISSION_LABEL: str = (
    "seam work submission is live SI-side (G-9 §9, S-3/SI-56): "
    "an ADMITted seam-crossing self-edit resting pending is "
    "submitted across the seam and the submission (with the "
    "fork work ID from its 202 response) is journaled as a "
    "gaia_work row; the item's state is submitted/awaiting "
    "outcome — S-5 outcome return is UNBUILT and S-4 leash "
    "crossing is UNBUILT; no live fork endpoint was exercised "
    "by this slice (transports are labelled replay/stub or "
    "loopback test double unless an operator supplies a real "
    "URL); live interop and performance UNMEASURED")

#: Response keys checked, in order, for a 202's stable work ID.
#: No field name is documented in-repo, so the spellings are
#: named here and the one found is journaled with the ID.
_WORK_ID_KEYS = ("work_id", "id", "workId")

#: Serialises dedup-scan → transport → journal per process, so
#: two concurrent submissions of one edit cannot both submit.
_SUBMIT_LOCK = threading.RLock()


def _extract_work_id(body) -> tuple[str | None, str | None]:
    if not isinstance(body, dict):
        return None, None
    for key in _WORK_ID_KEYS:
        val = body.get(key)
        if isinstance(val, str) and val:
            return val, key
        if isinstance(val, int) and not isinstance(val, bool):
            return str(val), key
    return None, None


def build_work_payload(edit_row: dict) -> dict:
    """The SI-side submission body for one journaled seam edit.

    Built only from the journaled self-edit row (journaled-args
    discipline): tool, args, reason, ``caused_by``, edit id,
    journal seq, seam task seq. SI's submission record design,
    not claimed to be a fork-published schema (none documented).
    """
    intent = edit_row.get("intent") or {}
    source = intent.get("source_seqs") or []
    return {
        "record": "si.gaia.seam.work_submission",
        "version": 1,
        "edit_id": edit_row.get("edit_id"),
        "journal_seq": edit_row.get("seq"),
        "task_seq": source[0] if source else None,
        "tool": intent.get("tool"),
        "args": intent.get("args") or {},
        "reason": edit_row.get("reason") or intent.get("reason"),
        "caused_by": intent.get("caused_by"),
        "state_after_submit": SUBMITTED_STATE,
    }


def _eligibility_problems(edit_row: dict | None) -> list[str]:
    if edit_row is None:
        return ["no self-edit with that edit_id is journaled"]
    problems = []
    intent = edit_row.get("intent") or {}
    if edit_row.get("gate_verdict") != "ADMIT":
        problems.append(
            f"gate verdict is {edit_row.get('gate_verdict')!r}, "
            "not ADMIT — only an ADMIT is submitted (REJECT and "
            "ESCALATE never submit)")
    if edit_row.get("result") != "pending":
        problems.append(
            f"self-edit result is {edit_row.get('result')!r}, "
            "not 'pending'")
    if intent.get("tool") not in SEAM_CROSSING_TOOLS:
        problems.append(
            f"tool {intent.get('tool')!r} is not a registered "
            "seam-crossing tool")
    execution = (edit_row.get("gate_report") or {}).get("execution")
    if execution != TOOL_SEAM_PENDING_EXECUTION:
        problems.append(
            "the journaled gate report's execution label is not "
            "TOOL_SEAM_PENDING_EXECUTION")
    return problems


def _find_edit(journal, edit_id: str) -> dict | None:
    rows = journal.backend.find(
        "self_edits", filter={"edit_id": edit_id}, limit=1)
    return rows[0] if rows else None


def _find_submission(journal, edit_id: str) -> dict | None:
    """The journaled gaia_work submission row for ``edit_id``."""
    best = None
    for row in journal.backend.find(
            "experiences", filter={"kind": "gaia_work"}) or []:
        if row.get("self_edit_id") != edit_id:
            continue
        record = row.get("record") or {}
        if record.get("status") != "submitted":
            continue
        if best is None or (row.get("seq") or 0) < (best.get("seq") or 0):
            best = row
    return best


def _submission_result(row: dict, *, deduped: bool) -> dict:
    record = row.get("record") or {}
    return {
        "status": "submitted",
        "state": record.get("state"),
        "edit_id": record.get("edit_id"),
        "fork_work_id": record.get("fork_work_id"),
        "fork_work_id_field": record.get("fork_work_id_field"),
        "http_status": record.get("http_status"),
        "gaia_work_seq": row.get("seq"),
        "transport": record.get("transport"),
        "deduped": deduped,
        "label": WORK_SUBMISSION_LABEL,
    }


def submit_seam_edit(bundle_dir: str, edit_id: str, *,
                     submission_url: str | None = None,
                     transport=None,
                     timeout_s: float = 30.0) -> dict:
    """Submit one pending ADMITted seam edit across the seam.

    Refusals (unknown/ineligible edit, no transport and no URL)
    return refused with nothing journaled and no transport
    called. Non-202 / 202-without-ID / transport failure return
    submission-failed / -unconfirmed, journaling no submission.
    A confirmed 202 + work ID journals the ``gaia_work`` row,
    re-anchors the manifest, and returns the submitted result.
    A repeat call returns it deduped, calling no transport.
    """
    if transport is None and not submission_url:
        return {"status": "refused", "journaled": False,
                "problems": [
                    "no transport injected and no submission_url "
                    "given — there is no default fork endpoint "
                    "(the SI repo documents no gaia-api path); "
                    "live interop is UNMEASURED"],
                "label": WORK_SUBMISSION_LABEL}
    if transport is None:
        transport = HttpTransport()
    transport_label = getattr(transport, "label", "http")

    with _SUBMIT_LOCK:
        bundle_dir = os.path.abspath(bundle_dir)
        manifest = load_bundle(bundle_dir)
        journal = open_journal(bundle_dir, manifest)
        try:
            prior = _find_submission(journal, edit_id)
            if prior is not None:
                return _submission_result(prior, deduped=True)
            edit_row = _find_edit(journal, edit_id)
            problems = _eligibility_problems(edit_row)
            if problems:
                return {"status": "refused", "journaled": False,
                        "problems": problems,
                        "label": WORK_SUBMISSION_LABEL}
            payload = build_work_payload(edit_row)
            try:
                status, raw = transport(
                    "POST", submission_url or "",
                    json_body=payload, timeout_s=timeout_s)
            except Exception as e:  # failure is data, not a claim
                return {"status": "submission-failed",
                        "journaled": False,
                        "failure": f"{type(e).__name__}: {e}",
                        "transport": transport_label,
                        "label": WORK_SUBMISSION_LABEL}
            try:
                body = json.loads(raw.decode("utf-8")) \
                    if isinstance(raw, (bytes, bytearray)) else raw
            except (ValueError, UnicodeDecodeError):
                body = None
            if status != 202:
                return {"status": "submission-failed",
                        "journaled": False, "http_status": status,
                        "response": body,
                        "transport": transport_label,
                        "label": WORK_SUBMISSION_LABEL}
            work_id, id_field = _extract_work_id(body)
            if work_id is None:
                return {"status": "submission-unconfirmed",
                        "journaled": False, "http_status": 202,
                        "response": body,
                        "transport": transport_label,
                        "problems": [
                            "202 response carried no work ID under "
                            f"any of {list(_WORK_ID_KEYS)} — not "
                            "journaled as a submission"],
                        "label": WORK_SUBMISSION_LABEL}
            intent = edit_row.get("intent") or {}
            record = {
                "kind": "gaia_work",
                "status": "submitted",
                "state": SUBMITTED_STATE,
                "edit_id": edit_id,
                "self_edit_journal_seq": edit_row.get("seq"),
                "tool": intent.get("tool"),
                "model": (intent.get("args") or {}).get("model"),
                "caused_by": intent.get("caused_by"),
                "fork_work_id": work_id,
                "fork_work_id_field": id_field,
                "http_status": 202,
                "transport": transport_label,
                "is_replay": bool(getattr(transport, "is_replay", False)),
                "submission_url": submission_url,
                "submitted_via": WORK_SUBMITTED_VIA,
                "payload": payload,
                "response": body,
                "label": WORK_SUBMISSION_LABEL,
            }
            journal.record_experience("gaia_work", record, edit_id)
            row = _find_submission(journal, edit_id)
            updated = _anchor_checkpoint(
                journal, manifest, manifest["head"],
                note=f"gaia seam submit: edit {edit_id} tool "
                     f"{intent.get('tool')!r} submitted, fork "
                     f"work id {work_id} (S-3; awaiting outcome, "
                     f"S-5 unbuilt)")
            _write_manifest(bundle_dir, updated)
            return _submission_result(row, deduped=False)
        finally:
            journal.close()


def submit_pending_seam_work(bundle_dir: str, **kwargs) -> dict:
    """Submit every spendable pending seam ADMIT in the bundle.

    Scans the journaled self-edits for seam-crossing ADMITs
    resting ``pending`` and submits each via
    :func:`submit_seam_edit` (already-submitted edits return
    deduped). Returns ``{"submitted": [...], "count": n,
    "label": ...}`` — the per-edit results, in journal order.
    Non-ADMIT and non-seam edits are never candidates.
    """
    bundle_dir = os.path.abspath(bundle_dir)
    manifest = load_bundle(bundle_dir)
    journal = open_journal(bundle_dir, manifest)
    try:
        candidates = []
        for row in journal.backend.find("self_edits") or []:
            if not _eligibility_problems(row):
                candidates.append(row.get("edit_id"))
    finally:
        journal.close()
    seen: list = []
    for edit_id in candidates:
        if edit_id not in seen:
            seen.append(edit_id)
    results = [submit_seam_edit(bundle_dir, edit_id, **kwargs)
               for edit_id in seen]
    return {"submitted": results, "count": len(results),
            "label": WORK_SUBMISSION_LABEL}


__all__ = [
    "SUBMITTED_STATE",
    "WORK_SUBMISSION_LABEL",
    "WORK_SUBMITTED_VIA",
    "build_work_payload",
    "submit_pending_seam_work",
    "submit_seam_edit",
]
