"""Gaia seam Flow 2b, SI side: the operator leash crossing
(G-9 §10, §12.3, §15 S-4; slice SI-57).

**Where the leash lives (G-9 §10.1).** In SI's journal. An
operator ``queue_stop`` / ``queue_resume`` on a model's seam
task *is* that model's stop/resume (the scope mapping of
§10.2: ingest maintains one seam task per model, so a stop on
that task names exactly one model). The fork's leash store is
a derived replica of that state; if the two disagree, SI's
journal wins. This module is the SI half of the crossing: it
reads the journaled leash state for seam tasks and carries it
across the seam. It mints no leash record — a record exists
here only because an operator journaled a stop/resume first
(§10.4: no fork-originated records, and none SI-originated
either — this client only carries journal acts).

**The fork-side contract relied on — and nothing beyond it.**
The fork half of S-4 (S-4a, ``POST /gaia/v1/leash`` and
``POST /gaia/v1/leash/heartbeat`` in the gaia fork's gaia-api)
is as-built and documented in G-9 §12.3:

- A leash record is exactly ``{"model", "kind": "stop"|"resume",
  "journal_seq", "reason"}`` — field-for-field the fork's
  ``LeashRecord``. ``journal_seq`` is the SI journal sequence
  of the stop/resume entry itself; ``reason`` is SI's own
  words: a stop's journaled ``cause`` (its ``detail`` when no
  cause is recorded), a resume's journaled ``detail``.
- Both routes answer **200** with ``{"applied": bool,
  "last_leash_seq": int}``. ``applied: false`` is the §10.3.2
  watermark no-op ACK (the record's ``journal_seq`` is at or
  below the fork's watermark — already applied), **not** an
  error: the delivery is confirmed either way.
- A heartbeat is exactly ``{"model", "leash_tip",
  "journal_tip"}`` (§12.3): ``leash_tip`` is the journal seq
  of the model's latest leash record, or its seam task's
  creation seq when none exists (§10.3.3's piggyback rule);
  ``journal_tip`` is the SI journal's current tip. A heartbeat
  carries no record and changes no leash state.

Consequences, by construction (the S-3 pattern):

- The fork base URL is **supplied by the caller/operator**
  (``leash_url`` / ``heartbeat_url``, or a transport called
  with those URLs); no host or port is invented or defaulted.
  The two paths above are G-9 §12.3's documented routes, and a
  heartbeat URL is derived from the leash URL only when that
  URL ends with the documented leash path.
- A crossing is journaled **only on a confirmed 200** whose
  body carries a boolean ``applied`` and an int
  ``last_leash_seq``. Any other status, a malformed 200, or a
  transport failure journals **no** crossing; the journaled
  leash record stays unsent and a retry is free.
- Journaled crossings are ``experiences`` rows of kind
  ``gaia_leash`` via ``Journal.record_experience`` (no new
  collection), carrying the record/heartbeat verbatim, the
  fork's ACK, the transport label, and whether the send was an
  event record, a heartbeat, or a §10.5 snapshot. Dedup is by
  ``journal_seq``: a record with a journaled confirmed crossing
  is never re-sent by the sync (the fork's watermark would
  no-op it anyway — SI simply does not spend the call).

**Sync, snapshot, heartbeat — explicit steps, no daemon.**
:func:`sync_seam_leash` sends, per seam model in journal
order, every leash record not yet confirmed across (§10.3.1's
event feed, operator-invoked — no thread, no timer).
:func:`send_leash_snapshot` is §10.5 re-sync: the model's
**current folded state** as a single record, sent even when it
already crossed as an event (the fork's watermark no-ops a
stale snapshot); a model with no leash record has nothing to
snapshot — nothing is sent and nothing is invented.
:func:`send_leash_heartbeat` sends one §12.3 heartbeat and
journals the confirmed delivery; the §10.3.4 interval/TTL
constants (10 s / 30 s) are UNMEASURED fork-side policy, and
this module claims no cadence.

**The leash honoured on this side too.** :mod:`si.gaia.seam_submit`
consults :func:`seam_leash_block` before submitting: while a
stop is in force for an edit's seam task, the submission is
refused (``leash-held``), nothing is submitted, and the edit
stays ``pending`` — journaling **nothing** in S-3's module (a
refusal is not a submission; the stop itself is already
journaled). A resume folds the stop away and submission
proceeds. §10.3.5's ``hearth.notify`` exemption is a
fork-actuation rule (notify actuates nothing fork-side);
SI-side submission is the seam crossing itself, so no tool is
exempted here — that asymmetry is named, not hidden.

**What is NOT built.** Fork-side staleness refusal is the
fork's S-4a half, not re-implemented or re-claimed here.
**S-5 outcome return remains UNBUILT**: no outcome returns, a
submitted item is still ``submitted/awaiting outcome``, and a
leash refusal fork-side would be an outcome — SI claims none.
**Live fork interop is UNMEASURED**: no live gaia-api endpoint
was exercised; evidence is the labelled replay/stub transport
and a loopback HTTP test double through the real
``HttpTransport``. An env-gated live test
(``SI_S4_LEASH_URL``) follows the S-2a/S-3 pattern and skips
when unset. Performance is **UNMEASURED**. CPU only; no
secrets; loopback / operator-configured endpoints only
(G-9 §14 posture, unchanged).
"""
from __future__ import annotations

import ast
import json
import os
import threading

from si.bundle import _anchor_checkpoint, _write_manifest, load_bundle
from si.gaia.hearth import HttpTransport
from si.gaia.queue import fold_queue
from si.gaia.seam import SEAM_OPERATOR, SEAM_SUBMITTED_VIA, seam_task_instruction
from si.journal import open_journal

#: Journaled ``crossed_via`` of every S-4 crossing row.
LEASH_CROSSED_VIA = "gaia-seam-leash (G-9/S-4)"

#: The documented G-9 §12.3 leash route (S-4a, as-built in the
#: gaia fork's gaia-api); the heartbeat route is this path +
#: "/heartbeat". Only the path is named — a host is always
#: operator-supplied, never defaulted.
LEASH_PATH = "/gaia/v1/leash"

#: Scope/honesty label carried on every S-4 result.
LEASH_CROSSING_LABEL: str = (
    "seam leash crossing is live SI-side (G-9 §10/§12.3, "
    "S-4/SI-57): journaled queue_stop/queue_resume records on "
    "a model's seam task cross to the fork's POST "
    "/gaia/v1/leash as {model, kind, journal_seq, reason}, "
    "heartbeats cross to /gaia/v1/leash/heartbeat, and a "
    "crossing is journaled (kind gaia_leash) only on a "
    "confirmed 200 ACK {applied, last_leash_seq} — applied "
    "false is the §10.3.2 watermark no-op, not an error; "
    "while a stop is in force, seam work submission for that "
    "model is refused SI-side (leash-held, nothing submitted, "
    "nothing journaled by the refusal); S-5 outcome return is "
    "UNBUILT and no outcome is claimed; no live fork endpoint "
    "was exercised by this slice (transports are labelled "
    "replay/stub or loopback test double unless an operator "
    "supplies a real URL); live interop and performance "
    "UNMEASURED")

#: Serialises scan → transport → journal per process, so two
#: concurrent syncs cannot both send one record.
_LEASH_LOCK = threading.RLock()


# -- reading the journaled leash (the source of truth) ------------------------

def seam_tasks(journal) -> list[dict]:
    """Every seam task in the journal: ``[{"task_seq", "model"}]``.

    A seam task is a ``task`` experience whose record carries
    the seam operator and submitted-via (the same identity
    :func:`si.gaia.seam.find_seam_task` matches) and whose
    instruction is the deterministic §10.2 construction for
    some model — the model is recovered by matching the
    instruction against that construction, never by parsing
    free text. Sorted by task seq; one task per model by
    construction (the earliest wins if history ever held two).
    """
    found: dict[str, dict] = {}
    rows = journal.backend.find("experiences", filter={"kind": "task"})
    for row in rows or []:
        record = row.get("record") or {}
        if not isinstance(record, dict):
            continue
        if record.get("operator") != SEAM_OPERATOR:
            continue
        if record.get("submitted_via") != SEAM_SUBMITTED_VIA:
            continue
        instruction = record.get("instruction")
        seq = row.get("seq")
        if not isinstance(instruction, str) \
                or not isinstance(seq, int) or isinstance(seq, bool):
            continue
        model = _model_from_instruction(instruction)
        if model is None:
            continue
        if model not in found or seq < found[model]["task_seq"]:
            found[model] = {"task_seq": seq, "model": model}
    return sorted(found.values(), key=lambda t: t["task_seq"])


def _model_from_instruction(instruction: str) -> str | None:
    """The model whose §10.2 instruction this is, or None.

    The instruction embeds ``repr(model)``; the candidate is
    recovered with :func:`ast.literal_eval` and accepted only
    when re-rendering the construction reproduces the
    instruction exactly — equality on the construction, the
    same discovery rule ``find_seam_task`` uses.
    """
    marker = "Gaia seam task for model "
    if not instruction.startswith(marker):
        return None
    rest = instruction[len(marker):]
    # repr of a str ends at its closing quote; scan for the
    # split point where the remainder is the fixed suffix.
    for end in range(1, len(rest) + 1):
        try:
            value = ast.literal_eval(rest[:end])
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, str) \
                and seam_task_instruction(value) == instruction:
            return value
    return None


def leash_records(journal, task_seq: int) -> list[dict]:
    """The journaled leash records for one task, in journal order.

    Each ``queue_stop`` / ``queue_resume`` experience naming
    ``task_seq`` becomes one §12.3 record (without ``model`` —
    the caller, who knows the task's model, adds it): ``kind``
    is ``stop`` / ``resume``, ``journal_seq`` is the entry's
    own journal sequence, and ``reason`` is the stop's
    journaled ``cause`` (its ``detail`` when no cause was
    recorded) or the resume's journaled ``detail``.
    """
    out = []
    rows = journal.backend.find("experiences") or []
    for row in sorted(rows, key=lambda r: r.get("seq") or 0):
        record = row.get("record") or {}
        if not isinstance(record, dict):
            continue
        kind = row.get("kind") or record.get("kind")
        if kind not in ("queue_stop", "queue_resume"):
            continue
        if record.get("task_seq") != task_seq:
            continue
        seq = row.get("seq")
        if not isinstance(seq, int) or isinstance(seq, bool):
            continue
        if kind == "queue_stop":
            reason = record.get("cause") or record.get("detail") or ""
            out.append({"kind": "stop", "journal_seq": seq,
                        "reason": str(reason)})
        else:
            reason = record.get("detail") or ""
            out.append({"kind": "resume", "journal_seq": seq,
                        "reason": str(reason)})
    return out


def build_leash_record(model: str, record: dict) -> dict:
    """One §12.3 wire record: ``{model, kind, journal_seq, reason}``."""
    return {"model": model, "kind": record["kind"],
            "journal_seq": record["journal_seq"],
            "reason": record["reason"]}


def leash_tip(journal, task: dict) -> int:
    """§10.3.3's tip for a seam task: its latest leash record's
    journal seq, or the seam task's creation seq when none."""
    records = leash_records(journal, task["task_seq"])
    if records:
        return records[-1]["journal_seq"]
    return task["task_seq"]


def seam_leash_block(journal, edit_row: dict) -> dict | None:
    """The leash block on one seam edit, or None when unheld.

    The edit's seam task is its intent's first ``source_seqs``
    entry (the §10.2 chaining ingest journaled). When the fold
    says that task has a stop in force, returns ``{"task_seq",
    "stop_seq", "model"}`` (``model`` when the task is a known
    seam task, else None); the caller must not submit. Pure
    read: folds the journal, writes nothing.
    """
    intent = (edit_row or {}).get("intent") or {}
    source = intent.get("source_seqs") or []
    task_seq = source[0] if source else None
    if not isinstance(task_seq, int) or isinstance(task_seq, bool):
        return None
    fold = fold_queue(journal)
    slot = fold.tasks.get(task_seq) or {}
    if not slot.get("stop_in_force"):
        return None
    model = None
    for task in seam_tasks(journal):
        if task["task_seq"] == task_seq:
            model = task["model"]
            break
    return {"task_seq": task_seq, "stop_seq": slot.get("stop_seq"),
            "model": model}


# -- confirmed crossings -------------------------------------------------------

def _find_crossing(journal, journal_seq: int, crossing: str) -> dict | None:
    """The journaled confirmed crossing of one leash seq, or None."""
    for row in journal.backend.find(
            "experiences", filter={"kind": "gaia_leash"}) or []:
        record = row.get("record") or {}
        if record.get("crossing") != crossing:
            continue
        if record.get("leash_journal_seq") != journal_seq:
            continue
        return row
    return None


def _parse_ack(body) -> dict | None:
    """A confirmed §12.3 ACK, or None: HTTP 200's body must be an
    object carrying a boolean ``applied`` and an int
    ``last_leash_seq`` — anything else confirms nothing."""
    if not isinstance(body, dict):
        return None
    applied = body.get("applied")
    last = body.get("last_leash_seq")
    if not isinstance(applied, bool):
        return None
    if not isinstance(last, int) or isinstance(last, bool):
        return None
    return {"applied": applied, "last_leash_seq": last}


def _deliver(transport, url, payload, timeout_s):
    """One transport call, S-3 style: ``(status, body, failure)``."""
    try:
        status, raw = transport("POST", url or "",
                                json_body=payload, timeout_s=timeout_s)
    except Exception as e:  # failure is data, not a claim
        return None, None, f"{type(e).__name__}: {e}"
    try:
        body = json.loads(raw.decode("utf-8")) \
            if isinstance(raw, (bytes, bytearray)) else raw
    except (ValueError, UnicodeDecodeError):
        body = None
    return status, body, None


def _journal_crossing(journal, manifest, bundle_dir, *, crossing,
                      payload, ack, transport_label, is_replay,
                      url, note):
    record = {
        "kind": "gaia_leash",
        "crossing": crossing,
        "status": "crossed",
        "model": payload.get("model"),
        "leash_kind": payload.get("kind"),
        "leash_journal_seq": payload.get("journal_seq",
                                         payload.get("leash_tip")),
        "payload": payload,
        "response": ack,
        "applied": ack["applied"],
        "last_leash_seq": ack["last_leash_seq"],
        "http_status": 200,
        "transport": transport_label,
        "is_replay": bool(is_replay),
        "url": url,
        "crossed_via": LEASH_CROSSED_VIA,
        "label": LEASH_CROSSING_LABEL,
    }
    journal.record_experience("gaia_leash", record, "")
    updated = _anchor_checkpoint(journal, manifest, manifest["head"],
                                 note=note)
    _write_manifest(bundle_dir, updated)
    return record


def _cross_one(bundle_dir, *, crossing, payload, url, transport,
               timeout_s, note) -> dict:
    """Send one payload and journal it only on a confirmed ACK."""
    transport_label = getattr(transport, "label", "http")
    status, body, failure = _deliver(transport, url, payload, timeout_s)
    if failure is not None:
        return {"status": "crossing-failed", "journaled": False,
                "crossing": crossing, "payload": payload,
                "failure": failure, "transport": transport_label,
                "label": LEASH_CROSSING_LABEL}
    ack = _parse_ack(body) if status == 200 else None
    if ack is None:
        return {"status": "crossing-unconfirmed"
                if status == 200 else "crossing-failed",
                "journaled": False, "crossing": crossing,
                "payload": payload, "http_status": status,
                "response": body, "transport": transport_label,
                "label": LEASH_CROSSING_LABEL}
    bundle_dir = os.path.abspath(bundle_dir)
    manifest = load_bundle(bundle_dir)
    journal = open_journal(bundle_dir, manifest)
    try:
        record = _journal_crossing(
            journal, manifest, bundle_dir, crossing=crossing,
            payload=payload, ack=ack, transport_label=transport_label,
            is_replay=getattr(transport, "is_replay", False),
            url=url, note=note)
    finally:
        journal.close()
    return {"status": "crossed", "journaled": True,
            "crossing": crossing, "payload": payload,
            "applied": ack["applied"],
            "last_leash_seq": ack["last_leash_seq"],
            "transport": transport_label, "record": record,
            "label": LEASH_CROSSING_LABEL}


def _resolve_transport(transport, url, what):
    if transport is None and not url:
        return None, {"status": "refused", "journaled": False,
                      "problems": [
                          f"no transport injected and no {what} "
                          "given — there is no default fork "
                          "endpoint; live interop is UNMEASURED"],
                      "label": LEASH_CROSSING_LABEL}
    return (transport if transport is not None else HttpTransport()), None


def _heartbeat_url(leash_url, heartbeat_url):
    if heartbeat_url:
        return heartbeat_url
    if leash_url and leash_url.rstrip("/").endswith(LEASH_PATH):
        return leash_url.rstrip("/") + "/heartbeat"
    return None


# -- the three sends -----------------------------------------------------------

def sync_seam_leash(bundle_dir: str, *, leash_url: str | None = None,
                    transport=None, timeout_s: float = 30.0) -> dict:
    """§10.3.1: send every uncrossed journaled leash record.

    Per seam model, in journal order, each stop/resume record
    with no journaled confirmed crossing is POSTed to
    ``leash_url`` and journaled on its confirmed ACK (a
    watermark no-op ACK, ``applied`` false, is confirmed too).
    The first unconfirmed/failed record stops that model's
    sync — later records are not sent past a gap — and is
    reported in the result.
    """
    transport, refusal = _resolve_transport(transport, leash_url,
                                            "leash_url")
    if refusal is not None:
        return refusal
    with _LEASH_LOCK:
        bundle_dir = os.path.abspath(bundle_dir)
        manifest = load_bundle(bundle_dir)
        journal = open_journal(bundle_dir, manifest)
        try:
            plan = []
            for task in seam_tasks(journal):
                pending = []
                for record in leash_records(journal, task["task_seq"]):
                    if _find_crossing(journal, record["journal_seq"],
                                      "record") is None:
                        pending.append(record)
                plan.append((task, pending))
        finally:
            journal.close()
        crossed, models = [], {}
        for task, pending in plan:
            sent = []
            blocked = None
            for record in pending:
                payload = build_leash_record(task["model"], record)
                out = _cross_one(
                    bundle_dir, crossing="record", payload=payload,
                    url=leash_url, transport=transport,
                    timeout_s=timeout_s,
                    note=f"gaia seam leash: {record['kind']} for "
                         f"model {task['model']!r} (journal seq "
                         f"{record['journal_seq']}) crossed (S-4)")
                if out["status"] != "crossed":
                    blocked = out
                    break
                sent.append(out)
                crossed.append(out)
            models[task["model"]] = {
                "task_seq": task["task_seq"], "crossed": sent,
                "blocked": blocked}
        return {"status": "synced" if all(
                    m["blocked"] is None for m in models.values())
                else "partial",
                "crossed": crossed, "count": len(crossed),
                "models": models, "label": LEASH_CROSSING_LABEL}


def send_leash_snapshot(bundle_dir: str, model: str, *,
                        leash_url: str | None = None,
                        transport=None,
                        timeout_s: float = 30.0) -> dict:
    """§10.5 re-sync: send ``model``'s current folded leash state.

    The single record sent is the model's latest journaled
    stop/resume (its folded state), even when it already
    crossed as an event — the fork's watermark no-ops a stale
    snapshot. A model with no seam task, or no leash record, is
    refused with nothing sent and nothing journaled: there is
    no folded stop/resume to snapshot and none is invented.
    """
    transport, refusal = _resolve_transport(transport, leash_url,
                                            "leash_url")
    if refusal is not None:
        return refusal
    with _LEASH_LOCK:
        bundle_dir = os.path.abspath(bundle_dir)
        manifest = load_bundle(bundle_dir)
        journal = open_journal(bundle_dir, manifest)
        try:
            task = next((t for t in seam_tasks(journal)
                         if t["model"] == model), None)
            records = leash_records(journal, task["task_seq"]) \
                if task else []
        finally:
            journal.close()
        if task is None:
            return {"status": "refused", "journaled": False,
                    "problems": [f"no seam task for model {model!r}"],
                    "label": LEASH_CROSSING_LABEL}
        if not records:
            return {"status": "refused", "journaled": False,
                    "problems": [
                        f"model {model!r} has no journaled leash "
                        "record — no folded stop/resume exists to "
                        "snapshot (§10.5 sends state, never invents "
                        "it)"],
                    "label": LEASH_CROSSING_LABEL}
        payload = build_leash_record(model, records[-1])
        return _cross_one(
            bundle_dir, crossing="snapshot", payload=payload,
            url=leash_url, transport=transport, timeout_s=timeout_s,
            note=f"gaia seam leash snapshot: model {model!r} folded "
                 f"state {records[-1]['kind']} at journal seq "
                 f"{records[-1]['journal_seq']} (S-4, §10.5)")


def send_leash_heartbeat(bundle_dir: str, model: str, *,
                         leash_url: str | None = None,
                         heartbeat_url: str | None = None,
                         transport=None,
                         timeout_s: float = 30.0) -> dict:
    """§12.3: send one leash heartbeat for ``model``.

    The payload is ``{"model", "leash_tip", "journal_tip"}``
    read from the journal at send time; the confirmed delivery
    is journaled like any crossing. No cadence is claimed — the
    §10.3.4 interval/TTL are UNMEASURED fork-side policy.
    """
    url = _heartbeat_url(leash_url, heartbeat_url)
    transport, refusal = _resolve_transport(transport, url,
                                            "heartbeat_url")
    if refusal is not None:
        if leash_url and heartbeat_url is None:
            refusal["problems"] = [
                "no heartbeat_url given and leash_url does not "
                f"end with the documented {LEASH_PATH} path, so "
                "no heartbeat URL is derived — no path is "
                "invented; live interop is UNMEASURED"]
        return refusal
    with _LEASH_LOCK:
        bundle_dir = os.path.abspath(bundle_dir)
        manifest = load_bundle(bundle_dir)
        journal = open_journal(bundle_dir, manifest)
        try:
            task = next((t for t in seam_tasks(journal)
                         if t["model"] == model), None)
            if task is None:
                return {"status": "refused", "journaled": False,
                        "problems": [
                            f"no seam task for model {model!r}"],
                        "label": LEASH_CROSSING_LABEL}
            payload = {"model": model,
                       "leash_tip": leash_tip(journal, task),
                       "journal_tip": journal.tip()[0]}
        finally:
            journal.close()
        return _cross_one(
            bundle_dir, crossing="heartbeat", payload=payload,
            url=url, transport=transport, timeout_s=timeout_s,
            note=f"gaia seam leash heartbeat: model {model!r} "
                 f"leash_tip {payload['leash_tip']} (S-4)")


__all__ = [
    "LEASH_CROSSED_VIA",
    "LEASH_CROSSING_LABEL",
    "LEASH_PATH",
    "build_leash_record",
    "leash_records",
    "leash_tip",
    "seam_leash_block",
    "seam_tasks",
    "send_leash_heartbeat",
    "send_leash_snapshot",
    "sync_seam_leash",
]
