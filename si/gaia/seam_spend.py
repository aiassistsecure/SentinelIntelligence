"""Gaia seam Flow 2, SI side: the resolution-spend bridge
(G-9 §9.2 as built; slice SI-116 / P-1 of SI-115).

SI-115's diagnosis: no path anywhere consumes a journaled
escalation resolution. The fork's Flow 2 intake is as-built
and running (submit with the verdict attached, S-1 dedup by
``journal_seq``, ticker claim, leash/stale checks, primitive,
outcome) — but a resolved escalation never becomes a
submission: S-3's caller refuses anything whose gate verdict
is not exactly ``ADMIT`` (a resolved intent's row still reads
``ESCALATE``), its payload is SI's own record design rather
than the fork's ``parse_intent`` shape, and the SI-side
Executor refuses the ``external`` class by construction.
This module is the one bridge SI-115 §4 prescribes, and
nothing else:

- **The spend act** — :func:`spend_resolved_escalation`:
  fold-verify that the named intent stands ``queued`` *by*
  an ``admit-after-review`` resolution (the
  ``resolve_escalation`` record is the ticket — its checks
  are reused, none invented), build the payload in the
  fork's as-built shape
  (``gateway.rs:170 parse_intent``), POST it to the fork's
  work route, and on the fork's 202 + ``work_id`` journal a
  ``gaia_work`` record tying intent seq + resolution seq +
  fork work id (the Flow 3 reconciliation anchor), exactly
  as S-3 journals its submissions. The spend is an explicit
  operator act (D-115-1): no watcher, no timer, no
  auto-spend — the operator's name and basis are recorded
  on the ``gaia_work`` row, and the act is invoked from the
  CLI (``si gaia-escalation ... spend``) or as a library
  call, never by a daemon.

- **The fork wire shape, as adopted (D-115-3).** The fork's
  as-built ``parse_intent`` is schema authority; SI adapts
  to it exactly. ``POST /gaia/v1/work`` with::

      {"type": "tool", "tool": <journaled tool>,
       "args": <journaled args verbatim>,
       "reason": <journaled reason>,
       "task_id": "task@<task_seq>",
       "journal_seq": <intent seq>,
       "verdict": "admit"}

  ``journal_seq`` is the intent's self-edit seq (G-9 §9.2),
  never the resolution seq. No ``class`` field is sent: the
  fork's parser defaults an absent class to ``execute``,
  which the running store carries — the priced path in
  SI-115 §2c (a submission today deterministically earns
  the fork's ``stale_state`` refusal, i.e. claim →
  actuation → outcome, not a 409). The 202 answer is
  ``{"work_id", "state": "accepted", "journal_seq"}``; a
  202 without a ``work_id`` is unconfirmed, never a spend.
  Dedup is twice over: this module returns a journaled
  ``gaia_work`` submission instead of submitting again, and
  the fork's S-1 dedup (``journal_seq`` → ``work_id``)
  returns the existing record for a resubmission that
  somehow reaches it — so a retry after a lost 202 can
  never actuate twice.

- **Refusal discipline.** A fold-verification refusal
  (unknown intent, unresolved escalation, a ``reject``
  resolution, an intent no longer ``queued``, a non-seam
  tool, an empty operator, a leash-held task) raises
  ValueError and journals NOTHING — the
  ``resolve_escalation`` discipline, verbatim. A second
  spend of an already-spent intent returns the original
  submission deduped and calls no transport. A fork
  refusal or a transport failure is the opposite case: it
  *is* journaled, as a ``gaia_work`` row of status
  ``spend-failed`` carrying the evidence (status / body /
  failure, the payload sent, operator, intent and
  resolution seqs) — a failed spend is evidence, and the
  act itself never retries silently. The failed row is not
  a submission, so a later operator re-invocation is free
  (and the fork's S-1 dedup still guards the resubmission).

- **The outcome leg (the (d)-leg caller SI-115 found
  missing).** :func:`poll_spent_outcome` and
  :func:`reconcile_spent_outcome` are intent-scoped
  wrappers over S-5's existing poller and reconciler
  (:mod:`si.gaia.seam_outcome`): they poll
  ``GET /gaia/v1/work/{work_id}`` for the one work id the
  spend journaled, and journal a terminal fork status
  through S-5's own ``_journal_outcome`` — once per
  ``work_id``, in S-5's §12.5 envelope, unmodified. No new
  outcome writer is added here; the residue is recorded in
  the module notes below rather than invented.

**Honest labels.** This bridge spends resolutions; it does
not itself execute tools, and a spend ACK (202 + work id)
is not an execution outcome — the fork's outcome returns
through the S-5 leg above. The spend's effect on the live
node is the fork's business (claim → leash/stale checks →
primitive → outcome); for ``se_0012`` SI-115 prices that
as the designed ``stale_state`` refusal, not a restart.
Live interop is UNMEASURED by this slice: all tests run
against injected transports and a loopback fork-route stub
that implements ``parse_intent`` + ACK + status from the
fork's as-built rules; the real spend is P-2, a lead-run
operational act, and no code here calls the real node.
"""
from __future__ import annotations

import json
import os
import threading

from si.bundle import _anchor_checkpoint, _write_manifest, load_bundle
from si.gaia.hearth import HttpTransport
from si.gaia.queue import fold_queue
from si.gaia.registry import SEAM_CROSSING_TOOLS
from si.gaia.seam_leash import seam_leash_block
from si.gaia.seam_outcome import (
    _journal_outcome,
    _journaled_outcomes,
    _terminal_result,
    reconcile_seam_outcomes,
)
from si.gaia.seam_submit import _extract_work_id
from si.journal import open_journal

#: The live fork's work route (SI-115 §4): the operator's
#: value for the CLI default. The library act takes the URL
#: as an argument (S-3's discipline — no default fork
#: endpoint is invented in the library); the CLI supplies
#: this one unless the operator overrides it.
DEFAULT_FORK_WORK_URL = "http://127.0.0.1:11435/gaia/v1/work"

#: Journaled ``submitted_via`` of every P-1 spend row.
SPEND_SUBMITTED_VIA = "gaia-seam-spend (G-9/SI-116 P-1)"

#: The one state a spent intent is ever in, in this slice:
#: submitted across the seam, awaiting an outcome that the
#: S-5 leg returns. Never "completed": a 202 is an ACK, not
#: an execution outcome.
SPEND_SUBMITTED_STATE = "submitted/awaiting outcome"

#: Honest basis recorded on a spend when the caller supplies
#: none: what the operator is recorded as having done.
DEFAULT_SPEND_BASIS = (
    "explicit operator spend of an admit-after-review "
    "resolution (SI-115 D-115-1): the operator reviewed the "
    "journaled intent and its resolution and spends that "
    "resolution across the seam; the spend is a second named "
    "operator act after the resolution, never automatic")

#: Scope/honesty label carried on every spend result.
SPEND_LABEL: str = (
    "resolution-spend bridge is live SI-side (G-9 §9.2, "
    "SI-116/P-1): a folded queued intent standing queued by "
    "an admit-after-review resolution is submitted to the "
    "fork's /gaia/v1/work in parse_intent's as-built shape "
    "(journal_seq = the intent's self-edit seq, verdict "
    "admit, no class field — the fork defaults it to "
    "execute) and the submission is journaled as a gaia_work "
    "row tying intent seq + resolution seq + fork work_id; "
    "a spend ACK is not an execution outcome (outcomes "
    "return through the S-5 leg, intent-scoped here); the "
    "bridge spends resolutions, it does not execute tools; "
    "live interop UNMEASURED (loopback stub + injected "
    "transports only); the real spend is P-2, lead-run")

#: Serialises verify → transport → journal per process, so
#: two concurrent spends of one intent cannot both submit.
_SPEND_LOCK = threading.RLock()


# -- reading the journal -------------------------------------------------------

def _find_edit_row(journal, edit_id: str) -> dict | None:
    rows = journal.backend.find(
        "self_edits", filter={"edit_id": edit_id}, limit=1)
    return rows[0] if rows else None


def _spend_rows(journal, intent_seq: int) -> list[dict]:
    """Every journaled ``gaia_work`` row naming this intent,
    in journal order — submissions and failed attempts."""
    out = []
    for row in journal.backend.find(
            "experiences", filter={"kind": "gaia_work"}) or []:
        record = row.get("record") or {}
        if record.get("self_edit_journal_seq") == intent_seq \
                or record.get("intent_seq") == intent_seq:
            out.append(row)
    return sorted(out, key=lambda r: r.get("seq") or 0)


def find_spend_row(journal, intent_seq: int) -> dict | None:
    """The journaled *submitted* spend row for an intent, or
    None. Failed attempts (``status`` ``spend-failed``) are
    evidence, not submissions, and never count here — the
    S-3 dedup shape, applied to the spend."""
    for row in _spend_rows(journal, intent_seq):
        if (row.get("record") or {}).get("status") == "submitted":
            return row
    return None


def _spend_result(row: dict, *, deduped: bool) -> dict:
    record = row.get("record") or {}
    return {
        "status": "submitted",
        "state": record.get("state"),
        "intent_seq": record.get("intent_seq"),
        "edit_id": record.get("edit_id"),
        "resolution": record.get("resolution"),
        "resolution_seq": record.get("resolution_seq"),
        "operator": record.get("operator"),
        "fork_work_id": record.get("fork_work_id"),
        "fork_work_id_field": record.get("fork_work_id_field"),
        "http_status": record.get("http_status"),
        "gaia_work_seq": row.get("seq"),
        "transport": record.get("transport"),
        "deduped": deduped,
        "label": SPEND_LABEL,
    }


# -- the fork-shaped payload (D-115-3) ------------------------------------------

def build_spend_payload(state, edit_row: dict) -> dict:
    """The fork's as-built submission body for one resolved
    intent (``gateway.rs:170 parse_intent``).

    Built only from the journaled self-edit row and the
    folded state — journaled-args discipline, verbatim:
    tool, args, reason, the task as ``task@<seq>``, the
    intent's own journal sequence, verdict ``admit``. No
    ``class`` field: the fork's parser defaults it to
    ``execute`` (SI-115 §2c's priced path). The resolution
    seq is deliberately NOT the ``journal_seq`` — G-9 §9.2
    names the intent's self-edit seq, and the resolution is
    carried in the journaled ``gaia_work`` row, not the wire
    payload."""
    intent = (edit_row or {}).get("intent") or {}
    task_seq = state.task_seq
    if task_seq is None:
        source = intent.get("source_seqs") or []
        task_seq = source[0] if source else None
    args = intent.get("args")
    if args is None:
        args = dict(state.args or {})
    return {
        "type": "tool",
        "tool": state.tool or intent.get("tool"),
        "args": args,
        "reason": state.reason,
        "task_id": f"task@{task_seq}",
        "journal_seq": state.intent_seq,
        "verdict": "admit",
    }


def verify_spendable(journal, intent_seq: int,
                     task_seq=None):
    """The fold verification (SI-115 §4; the ticket is the
    resolution record): the named intent must fold
    ``queued`` *because* an ``admit-after-review``
    resolution moved it there, with its resolution sequence
    present, and its tool must be a registered seam-crossing
    tool. Any failure raises ValueError and journals
    NOTHING — the ``resolve_escalation`` discipline,
    verbatim. Returns ``(state, edit_row)``."""
    fold = fold_queue(journal, task_seq=task_seq)
    state = fold.by_seq.get(intent_seq)
    if state is None:
        raise ValueError(
            f"no folded intent at seq {intent_seq}: nothing "
            f"to spend (fail closed, nothing journaled)")
    if state.resolution is None:
        raise ValueError(
            f"intent seq {intent_seq} folds to "
            f"{state.state!r} with no operator resolution: an "
            f"unresolved escalation is never spent (SI-41 "
            f"§5.4; nothing journaled)")
    if state.resolution != "admit-after-review":
        raise ValueError(
            f"intent seq {intent_seq} was resolved "
            f"{state.resolution!r} at seq "
            f"{state.resolution_seq}: the operator refused "
            f"it, and a refused intent is never spent "
            f"(nothing journaled)")
    if state.resolution_seq is None:
        raise ValueError(
            f"intent seq {intent_seq} is resolved "
            f"admit-after-review with no resolution sequence "
            f"on the fold: there is no ticket to spend "
            f"(nothing journaled)")
    if state.state != "queued":
        raise ValueError(
            f"intent seq {intent_seq} folds to "
            f"{state.state!r}, not 'queued': a resolved "
            f"escalation spends exactly once, through its "
            f"spend — an intent already claimed, executed, "
            f"or otherwise concluded is not re-spent (SI-50 "
            f"§13: never blindly replay; nothing journaled)")
    if state.tool not in SEAM_CROSSING_TOOLS:
        raise ValueError(
            f"tool {state.tool!r} is not a registered "
            f"seam-crossing tool: only a seam-crossing "
            f"intent is spent across the seam (nothing "
            f"journaled)")
    edit_row = _find_edit_row(journal, state.edit_id)
    if edit_row is None:
        raise ValueError(
            f"no journaled self-edit for intent seq "
            f"{intent_seq} (edit {state.edit_id!r}): the "
            f"payload is built from the journaled row and "
            f"there is none (nothing journaled)")
    return state, edit_row


# -- journaling ------------------------------------------------------------------

def _journal_spend(journal, manifest, bundle_dir, record,
                   edit_id, note) -> dict:
    """Journal one ``gaia_work`` row (submission or failed
    attempt) and re-anchor the manifest, as S-3 does — then
    return the journaled row."""
    journal.record_experience("gaia_work", record, edit_id)
    row = None
    rows = journal.backend.find(
        "experiences",
        filter={"kind": "gaia_work", "self_edit_id": edit_id},
        sort="-seq", limit=1)
    row = rows[0] if rows else None
    updated = _anchor_checkpoint(
        journal, manifest, manifest["head"], note=note)
    _write_manifest(bundle_dir, updated)
    return row


# -- the spend act ---------------------------------------------------------------

def spend_resolved_escalation(bundle_dir: str, intent_seq: int,
                              *, operator: str, basis: str = "",
                              work_url: str | None = None,
                              transport=None,
                              timeout_s: float = 30.0,
                              task_seq=None) -> dict:
    """Spend one admit-after-review resolution (SI-116/P-1).

    An explicit operator act (D-115-1): fold-verify the
    intent (see :func:`verify_spendable` — a verification
    refusal raises ValueError and journals NOTHING), then
    POST the fork-shaped payload to ``work_url`` and, on the
    fork's 202 + ``work_id``, journal the ``gaia_work``
    submission tying intent seq + resolution seq + fork work
    id, and re-anchor. A repeat spend returns the original
    submission deduped and calls no transport. A fork
    refusal or transport failure journals the failed attempt
    as evidence (status ``spend-failed``) and reports it —
    the act never retries silently, and the failed row is
    not a submission, so a later operator re-invocation is
    free (the fork's S-1 dedup guards the resubmission).

    ``work_url`` or an injected ``transport`` is required
    (S-3's discipline: the library invents no fork
    endpoint); with neither, returns refused with nothing
    journaled and no transport called. ``operator`` must be
    non-empty — an anonymous spend is refused before any
    verification, journaling nothing."""
    if not isinstance(operator, str) or not operator.strip():
        raise ValueError(
            "a resolution spend requires an operator name "
            "(D-115-1: the spender is recorded; an anonymous "
            "spend is refused, nothing journaled)")
    if transport is None and not work_url:
        return {"status": "refused", "journaled": False,
                "problems": [
                    "no transport injected and no work_url "
                    "given — there is no default fork endpoint "
                    "(the SI repo documents no gaia-api path); "
                    "live interop is UNMEASURED"],
                "label": SPEND_LABEL}
    if transport is None:
        transport = HttpTransport()
    transport_label = getattr(transport, "label", "http")

    with _SPEND_LOCK:
        bundle_dir = os.path.abspath(bundle_dir)
        manifest = load_bundle(bundle_dir)
        journal = open_journal(bundle_dir, manifest)
        try:
            prior = find_spend_row(journal, intent_seq)
            if prior is not None:
                return _spend_result(prior, deduped=True)
            state, edit_row = verify_spendable(
                journal, intent_seq, task_seq=task_seq)
            block = seam_leash_block(journal, edit_row)
            if block is not None:
                return {"status": "leash-held", "journaled": False,
                        "leash": block,
                        "problems": [
                            "an operator stop is in force for "
                            f"this intent's seam task "
                            f"{block['task_seq']} (stop journaled "
                            f"at seq {block['stop_seq']}) — the "
                            "leash is honoured SI-side (G-9 §10, "
                            "S-4): nothing is spent while "
                            "leashed; a journaled queue_resume "
                            "re-enables the spend"],
                        "label": SPEND_LABEL}
            intent = edit_row.get("intent") or {}
            payload = build_spend_payload(state, edit_row)
            try:
                status, raw = transport(
                    "POST", work_url or "",
                    json_body=payload, timeout_s=timeout_s)
            except Exception as e:  # failure is data, not a claim
                record = {
                    "kind": "gaia_work",
                    "status": "spend-failed",
                    "state": SPEND_SUBMITTED_STATE,
                    "edit_id": state.edit_id,
                    "intent_seq": state.intent_seq,
                    "self_edit_journal_seq": state.intent_seq,
                    "resolution": "admit-after-review",
                    "resolution_seq": state.resolution_seq,
                    "operator": operator,
                    "basis": basis or DEFAULT_SPEND_BASIS,
                    "tool": state.tool,
                    "model": (intent.get("args") or {}).get("model"),
                    "caused_by": intent.get("caused_by"),
                    "failure": f"{type(e).__name__}: {e}",
                    "transport": transport_label,
                    "is_replay": bool(
                        getattr(transport, "is_replay", False)),
                    "submission_url": work_url,
                    "submitted_via": SPEND_SUBMITTED_VIA,
                    "payload": payload,
                    "label": SPEND_LABEL,
                }
                row = _journal_spend(
                    journal, manifest, bundle_dir, record,
                    state.edit_id,
                    note=f"gaia resolution spend failed: intent "
                         f"seq {state.intent_seq} tool "
                         f"{state.tool!r} transport failure "
                         f"(SI-116 P-1; no submission)")
                return {"status": "spend-failed", "journaled": True,
                        "gaia_work_seq": row.get("seq") if row else None,
                        "intent_seq": state.intent_seq,
                        "edit_id": state.edit_id,
                        "resolution_seq": state.resolution_seq,
                        "failure": record["failure"],
                        "transport": transport_label,
                        "label": SPEND_LABEL}
            try:
                body = json.loads(raw.decode("utf-8")) \
                    if isinstance(raw, (bytes, bytearray)) else raw
            except (ValueError, UnicodeDecodeError):
                body = None
            if status != 202:
                record = {
                    "kind": "gaia_work",
                    "status": "spend-failed",
                    "state": SPEND_SUBMITTED_STATE,
                    "edit_id": state.edit_id,
                    "intent_seq": state.intent_seq,
                    "self_edit_journal_seq": state.intent_seq,
                    "resolution": "admit-after-review",
                    "resolution_seq": state.resolution_seq,
                    "operator": operator,
                    "basis": basis or DEFAULT_SPEND_BASIS,
                    "tool": state.tool,
                    "model": (intent.get("args") or {}).get("model"),
                    "caused_by": intent.get("caused_by"),
                    "http_status": status,
                    "response": body,
                    "transport": transport_label,
                    "is_replay": bool(
                        getattr(transport, "is_replay", False)),
                    "submission_url": work_url,
                    "submitted_via": SPEND_SUBMITTED_VIA,
                    "payload": payload,
                    "label": SPEND_LABEL,
                }
                row = _journal_spend(
                    journal, manifest, bundle_dir, record,
                    state.edit_id,
                    note=f"gaia resolution spend refused by the "
                         f"fork: intent seq {state.intent_seq} "
                         f"tool {state.tool!r} HTTP {status} "
                         f"(SI-116 P-1; no submission)")
                return {"status": "spend-failed", "journaled": True,
                        "gaia_work_seq": row.get("seq") if row else None,
                        "intent_seq": state.intent_seq,
                        "edit_id": state.edit_id,
                        "resolution_seq": state.resolution_seq,
                        "http_status": status, "response": body,
                        "transport": transport_label,
                        "label": SPEND_LABEL}
            work_id, id_field = _extract_work_id(body)
            if work_id is None:
                record = {
                    "kind": "gaia_work",
                    "status": "spend-failed",
                    "state": SPEND_SUBMITTED_STATE,
                    "edit_id": state.edit_id,
                    "intent_seq": state.intent_seq,
                    "self_edit_journal_seq": state.intent_seq,
                    "resolution": "admit-after-review",
                    "resolution_seq": state.resolution_seq,
                    "operator": operator,
                    "basis": basis or DEFAULT_SPEND_BASIS,
                    "tool": state.tool,
                    "model": (intent.get("args") or {}).get("model"),
                    "caused_by": intent.get("caused_by"),
                    "http_status": 202,
                    "response": body,
                    "transport": transport_label,
                    "is_replay": bool(
                        getattr(transport, "is_replay", False)),
                    "submission_url": work_url,
                    "submitted_via": SPEND_SUBMITTED_VIA,
                    "payload": payload,
                    "label": SPEND_LABEL,
                }
                row = _journal_spend(
                    journal, manifest, bundle_dir, record,
                    state.edit_id,
                    note=f"gaia resolution spend unconfirmed: "
                         f"intent seq {state.intent_seq} tool "
                         f"{state.tool!r} 202 without a work ID "
                         f"(SI-116 P-1; no submission)")
                return {"status": "spend-unconfirmed",
                        "journaled": True,
                        "gaia_work_seq": row.get("seq") if row else None,
                        "intent_seq": state.intent_seq,
                        "edit_id": state.edit_id,
                        "resolution_seq": state.resolution_seq,
                        "http_status": 202, "response": body,
                        "transport": transport_label,
                        "problems": [
                            "202 response carried no work ID under "
                            "any of the accepted spellings — not "
                            "journaled as a submission"],
                        "label": SPEND_LABEL}
            record = {
                "kind": "gaia_work",
                "status": "submitted",
                "state": SPEND_SUBMITTED_STATE,
                "edit_id": state.edit_id,
                "self_edit_journal_seq": state.intent_seq,
                "intent_seq": state.intent_seq,
                "task_seq": state.task_seq,
                "resolution": "admit-after-review",
                "resolution_seq": state.resolution_seq,
                "operator": operator,
                "basis": basis or DEFAULT_SPEND_BASIS,
                "tool": state.tool,
                "model": (intent.get("args") or {}).get("model"),
                "caused_by": intent.get("caused_by"),
                "fork_work_id": work_id,
                "fork_work_id_field": id_field,
                "http_status": 202,
                "transport": transport_label,
                "is_replay": bool(
                    getattr(transport, "is_replay", False)),
                "submission_url": work_url,
                "submitted_via": SPEND_SUBMITTED_VIA,
                "payload": payload,
                "response": body,
                "label": SPEND_LABEL,
            }
            row = _journal_spend(
                journal, manifest, bundle_dir, record,
                state.edit_id,
                note=f"gaia resolution spend: intent seq "
                     f"{state.intent_seq} tool {state.tool!r} "
                     f"spent, fork work id {work_id} "
                     f"(SI-116 P-1; awaiting outcome)")
            return _spend_result(row, deduped=False)
        finally:
            journal.close()


# -- the spent intent's state (read) ---------------------------------------------

def spent_view(journal, intent_seq: int) -> dict | None:
    """One spent intent as plain data: the folded state, the
    journaled spend row (if any), the failed attempts (if
    any), and the journaled S-5 outcome (if any). A read —
    folds and scans, journals nothing."""
    fold = fold_queue(journal)
    state = fold.by_seq.get(intent_seq)
    if state is None:
        return None
    spend = find_spend_row(journal, intent_seq)
    attempts = [r for r in _spend_rows(journal, intent_seq)
                if (r.get("record") or {}).get("status")
                == "spend-failed"]
    outcome = None
    if spend is not None:
        work_id = (spend.get("record") or {}).get("fork_work_id")
        for row in journal.backend.find(
                "experiences", filter={"kind": "tool_result"}) or []:
            record = row.get("record") or {}
            if record.get("source") == "gaia-seam" \
                    and record.get("work_id") == work_id:
                outcome = {"outcome_seq": row.get("seq"),
                           "outcome": (record.get("fork_result")
                                       or {}).get("outcome"),
                           "fork_result": record.get("fork_result")}
                break
    return {
        "intent_seq": state.intent_seq,
        "edit_id": state.edit_id,
        "task_seq": state.task_seq,
        "tool": state.tool,
        "state": state.state,
        "verdict": state.verdict,
        "resolution": state.resolution,
        "resolution_seq": state.resolution_seq,
        "resolution_operator": state.resolution_operator,
        "spend": None if spend is None else {
            "gaia_work_seq": spend.get("seq"),
            "fork_work_id": (spend.get("record") or {})
            .get("fork_work_id"),
            "operator": (spend.get("record") or {}).get("operator"),
            "submitted_via": (spend.get("record") or {})
            .get("submitted_via")},
        "failed_attempts": [
            {"gaia_work_seq": r.get("seq"),
             "failure": (r.get("record") or {}).get("failure"),
             "http_status": (r.get("record") or {})
             .get("http_status")}
            for r in attempts],
        "outcome": outcome,
    }


def read_spend_at(bundle_dir: str, intent_seq: int):
    """Open ``bundle_dir`` and read one spent intent's view:
    directly (no manifest-anchor demand — the SI-58 lesson),
    but the hash chain must verify before anything is folded
    from it — the same read discipline as the escalation
    trigger's reads (SI-113)."""
    journal = open_journal(bundle_dir)
    try:
        if not journal.verify_chain():
            raise ValueError(
                "journal hash chain does not verify: refusing "
                "to fold a state from an unverified journal "
                "(fail closed)")
        return spent_view(journal, intent_seq)
    finally:
        journal.close()


# -- the outcome leg (S-5, intent-scoped) -----------------------------------------

def poll_spent_outcome(bundle_dir: str, intent_seq: int, *,
                       work_url: str | None = None,
                       transport=None,
                       timeout_s: float = 30.0) -> dict:
    """§11.2, intent-scoped: poll the one work id the spend
    journaled for this intent.

    A thin wrapper over S-5's own poller machinery
    (:func:`si.gaia.seam_outcome._terminal_result` /
    ``_journal_outcome``): a terminal fork status is journaled
    as a ``tool_result`` seam outcome exactly once per
    ``work_id``, in S-5's §12.5 envelope — no new outcome
    writer is added here. An already-journaled outcome is
    returned deduped without polling; a non-terminal status,
    404, other status, or transport failure journals nothing
    and is reported as data. An intent with no journaled
    spend is refused with nothing journaled (polling names a
    spent intent, never an invented one)."""
    if transport is None and not work_url:
        return {"status": "refused", "journaled": False,
                "problems": [
                    "no transport injected and no work_url "
                    "given — there is no default fork endpoint; "
                    "live interop is UNMEASURED"],
                "label": SPEND_LABEL}
    if transport is None:
        transport = HttpTransport()
    transport_label = getattr(transport, "label", "http")
    with _SPEND_LOCK:
        bundle_dir = os.path.abspath(bundle_dir)
        manifest = load_bundle(bundle_dir)
        journal = open_journal(bundle_dir, manifest)
        try:
            sub = find_spend_row(journal, intent_seq)
            if sub is None:
                return {"status": "refused", "journaled": False,
                        "problems": [
                            f"no journaled spend for intent seq "
                            f"{intent_seq} — an outcome poll "
                            f"names a spent intent, never an "
                            f"invented one"],
                        "label": SPEND_LABEL}
            work_id = (sub.get("record") or {}).get("fork_work_id")
            done = _journaled_outcomes(journal)
            if work_id in done:
                existing = done[work_id]
                return {"status": "journaled", "deduped": True,
                        "intent_seq": intent_seq,
                        "work_id": work_id,
                        "edit_id": (sub.get("record") or {})
                        .get("edit_id"),
                        "outcome_seq": existing.get("seq"),
                        "outcome": (existing.get("record") or {})
                        .get("fork_result", {}).get("outcome"),
                        "label": SPEND_LABEL}
        finally:
            journal.close()
        url = work_url.rstrip("/") + "/" + work_id \
            if work_url else ""
        try:
            status, raw = transport("GET", url, timeout_s=timeout_s)
        except Exception as e:  # failure is data, not a claim
            return {"status": "poll-failed", "journaled": False,
                    "intent_seq": intent_seq, "work_id": work_id,
                    "failure": f"{type(e).__name__}: {e}",
                    "transport": transport_label,
                    "label": SPEND_LABEL}
        try:
            body = json.loads(raw.decode("utf-8")) \
                if isinstance(raw, (bytes, bytearray)) else raw
        except (ValueError, UnicodeDecodeError):
            body = None
        if status == 404:
            return {"status": "not-found", "journaled": False,
                    "intent_seq": intent_seq, "work_id": work_id,
                    "label": SPEND_LABEL}
        if status != 200:
            return {"status": "poll-failed", "journaled": False,
                    "intent_seq": intent_seq, "work_id": work_id,
                    "http_status": status, "response": body,
                    "transport": transport_label,
                    "label": SPEND_LABEL}
        result = _terminal_result(body)
        if result is None:
            return {"status": "awaiting", "journaled": False,
                    "intent_seq": intent_seq, "work_id": work_id,
                    "state": body.get("state")
                    if isinstance(body, dict) else None,
                    "label": SPEND_LABEL}
        out = _journal_outcome(
            bundle_dir, sub, result, via="poll (SI-116 spend outcome poll)",
            transport_label=transport_label)
        return {"status": out["status"], "deduped": out["deduped"],
                "intent_seq": intent_seq, "work_id": work_id,
                "edit_id": out["edit_id"],
                "outcome_seq": out["outcome_seq"],
                "outcome": out["outcome"],
                "label": SPEND_LABEL}


def reconcile_spent_outcome(bundle_dir: str, intent_seq: int,
                            outcomes_path: str, *,
                            cursor: int = 0) -> dict:
    """§11.3, intent-scoped: reconcile this spent intent
    against the fork's ``outcomes.jsonl``.

    A thin wrapper over S-5's read-only reconciler: the full
    tail runs (the cursor is shared across intents), and the
    returned view is filtered to this intent's work id — the
    journaled outcome, cross-check, fault, or unmatched
    entry for it, if any. The reconciler's own result is
    carried alongside, unfiltered, so the operator sees the
    whole tail's counts."""
    with _SPEND_LOCK:
        bundle_dir = os.path.abspath(bundle_dir)
        manifest = load_bundle(bundle_dir)
        journal = open_journal(bundle_dir, manifest)
        try:
            sub = find_spend_row(journal, intent_seq)
            if sub is None:
                return {"status": "refused", "journaled": False,
                        "cursor": cursor,
                        "problems": [
                            f"no journaled spend for intent seq "
                            f"{intent_seq} — reconciliation "
                            f"names a spent intent, never an "
                            f"invented one"],
                        "label": SPEND_LABEL}
            work_id = (sub.get("record") or {}).get("fork_work_id")
        finally:
            journal.close()
    res = reconcile_seam_outcomes(bundle_dir, outcomes_path,
                                  cursor=cursor)

    def _for(entry_list):
        return [e for e in (entry_list or [])
                if e.get("work_id") == work_id]

    return {"status": res.get("status"),
            "intent_seq": intent_seq, "work_id": work_id,
            "cursor": res.get("cursor"),
            "journaled": _for(res.get("journaled")),
            "cross_checked": _for(res.get("cross_checked")),
            "faults": _for(res.get("faults")),
            "unmatched": _for(res.get("unmatched")),
            "torn": res.get("torn"),
            "count": len(_for(res.get("journaled"))),
            "reconciler": res,
            "label": SPEND_LABEL}


__all__ = [
    "DEFAULT_FORK_WORK_URL",
    "DEFAULT_SPEND_BASIS",
    "SPEND_LABEL",
    "SPEND_SUBMITTED_STATE",
    "SPEND_SUBMITTED_VIA",
    "build_spend_payload",
    "find_spend_row",
    "poll_spent_outcome",
    "read_spend_at",
    "reconcile_spent_outcome",
    "spend_resolved_escalation",
    "spent_view",
    "verify_spendable",
]
