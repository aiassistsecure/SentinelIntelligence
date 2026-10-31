"""Gaia seam S-6, SI side: the SI journal-event forwarder
(G-9 §4.5, §15 slice S-6; slice SI-60).

Until this slice the fork's SSE stream had an ``source=si``
source with no production producer: the fork's ``SiEventBuffer``
exists and requires strictly increasing SI seqs, but nothing
pushed SI journal events to it (G-9 §4.5). This module is the
**SI half only** of S-6: a forwarder that follows SI's own
journal and pushes each journal event, in order, to the fork's
documented events-ingest route. The fork half —
``POST /gaia/v1/events/ingest`` in ``aiassistsecure/gaia``,
feeding ``SiEventBuffer::push`` — is **OUT OF SCOPE / not
built by this slice**; nothing here claims it exists.

**The shape pushed (§4.5).** Each journal row becomes one
payload in the fork's ``StreamEvent::from_si`` shape, built by
:func:`build_stream_event`: ``source`` ``"si"``, ``seq`` the
**SI journal seq** (never presented as a spine seq — the two
sequence spaces are never merged, G-9 §2 invariant 2),
``kind`` the SI event name from
:func:`si.gaia.events.event_name_for_row` (carried on the
events :func:`si.gaia.events.read_journal_events` returns),
``collection``, the **full journal row verbatim** under
``row``, ``row_kind`` (the row's own ``kind`` or None), and
``facts`` = ``{"collection", "row"}`` plus ``queue_state``
only when the journal event carries one (the SI-53
convention). ``model`` / ``task`` keys are included **only**
when derivable from the row without invention (a model string
in the intent args or the record; a task seq / task ref the
row itself names) — otherwise the keys are omitted, never
guessed. ``hash`` is **never** included: it is hearth-only
per §4.5 (the fork does not hash what it merely carries).

**The SI-49 discipline, applied to pushing (§5.5 / §7
principle 7).** The caller holds the cursor; it is the last
journal seq the fork acknowledged. The forwarder:

- reads journal events via
  :func:`si.gaia.events.read_journal_events` in ascending
  seq order — exact seq fidelity, one journal row = one push;
- POSTs **one event per call** to the operator-supplied
  ``ingest_url`` (the full URL, supplied whole — see below);
- advances the cursor **only after HTTP 200** for that event,
  and returns it; the cursor is caller-held/returned, never
  stored — no cursor store is invented (the S-5 reconciler
  cursor pattern);
- treats a duplicate/refusal answer (HTTP 409, the fork
  buffer's loud refusal of a duplicate or out-of-order seq)
  as a **protocol failure, not a retry**: it raises
  :class:`SeamProtocolError`, pushes nothing after it, and
  does not advance the cursor past it. A duplicate means the
  two sides disagree about what was already pushed —
  silently deduping or retrying it away would hide exactly
  that disagreement (SI-49's rule for consumers, generalised
  by §7 principle 7);
- treats transport exceptions and other non-200 answers
  (5xx included) as ``forward-failed``: forwarding stops,
  the returned cursor is the last acked seq, and a retry
  from that cursor is free **because history is the
  journal** — events journaled while the fork is unreachable
  are simply pushed later, from the cursor.

**The forwarder journals NOTHING.** The journal is the
source; pushing is a read. No row is written, no experience
is recorded, and the manifest is never touched. For the same
reason the read path opens the journal directly and does not
demand manifest-anchor == journal-tip (the SI-58 lesson): a
live journal's tip legitimately advances past its anchor
between re-anchors, and those newest rows are exactly what a
forwarder exists to push.

**No host is invented or defaulted.** The documented fork
route is named as :data:`INGEST_PATH` only; the URL is the
operator's to supply whole (``ingest_url``), like S-5's
``work_url`` pattern — this module appends no path to any
base. With no transport and no URL the call is ``refused``.
Transports are :mod:`si.gaia.hearth`'s, honestly labelled
(the B-4 pattern): the real ``HttpTransport`` when a URL is
given, or any injected callable (tests use the labelled
``ReplayTransport`` and a loopback HTTP test double). **No
live fork endpoint is exercised by this slice**: live interop
is **UNMEASURED**, and every result carries its transport
label so a replay/loopback push can never be read as a fork
one. An env-gated live test (``SI_S6_INGEST_URL``) follows
the S-3/S-5 pattern and skips cleanly when unset.

Performance is **UNMEASURED**. CPU only; no secrets;
loopback / operator-configured endpoints only (G-9 §14
posture, unchanged).
"""
from __future__ import annotations

import json
import os
import threading

from si.gaia.events import read_journal_events
from si.gaia.hearth import HttpTransport
from si.journal import open_journal

#: The documented fork events-ingest route (G-9 §15 S-6:
#: ``POST /gaia/v1/events/ingest``, loopback, agency listener,
#: fork side). Only the path is named — a host is always
#: operator-supplied, never defaulted, and the operator
#: supplies the full ingest URL whole (this module appends
#: nothing to any base).
INGEST_PATH = "/gaia/v1/events/ingest"

#: Scope/honesty label carried on every S-6 result.
EVENT_FORWARD_LABEL: str = (
    "seam SI journal-event forwarding is live SI-side "
    "(G-9 §4.5/§15, S-6/SI-60): journal events are pushed "
    "one per POST, in ascending SI journal seq order, in "
    "the fork's StreamEvent::from_si shape (source=si, seq "
    "is the SI journal seq — never a spine seq, no hash "
    "key), to an operator-supplied ingest URL; the cursor "
    "advances only after the fork's HTTP 200, is "
    "caller-held/returned and never stored, and a 409 "
    "duplicate/refusal is a loud protocol failure "
    "(SeamProtocolError), never a silent retry; the "
    "forwarder journals nothing (pushing is a read of the "
    "journal, which remains the source); the fork half "
    "(POST /gaia/v1/events/ingest in aiassistsecure/gaia) "
    "is OUT OF SCOPE / not built by this slice, and SI's "
    "events_client consuming the fork stream is fork-side "
    "acceptance — not proven here; no live fork endpoint "
    "was exercised by this slice (transports are labelled "
    "replay/stub or loopback test double unless an operator "
    "supplies a real URL); live interop and performance "
    "UNMEASURED"
    " Measured under S-7 (SI-62): median 152 ms (range 138–158, N=10) for Q-E4 event forward pass (100 events) on the miniature fixture, embedded journal, Class L transport, double service median 0.3 ms, host fingerprint in docs/S-7-seam-measurements.md; single-run; live fork interop performance remains UNMEASURED. Measured under S-7 (SI-62): median 0.55 s (range 0.50–0.58, N=5) for Q-E5 event catch-up (backlog 500, single unlimited pass) on the miniature fixture, embedded journal, Class L transport, host fingerprint in docs/S-7-seam-measurements.md; single-run; fork ingest (UNBUILT) and SSE visibility remain UNMEASURED.")

#: Serialises one forwarding pass per process, so two
#: concurrent passes cannot push the same window twice or
#: interleave out of order in-process.
_FORWARD_LOCK = threading.RLock()


class SeamProtocolError(RuntimeError):
    """A seam protocol failure: the fork refused a push as a
    duplicate / out-of-order event (HTTP 409).

    Raised, never retried and never silently deduped (the
    SI-49 discipline, §7 principle 7). Carries the offending
    event's SI journal ``seq``, the ``http_status``, the
    decoded ``response``, the ``cursor`` as of the failure
    (the last acked seq — it does NOT include the refused
    event), and the seqs ``pushed`` before it in this pass.
    """

    def __init__(self, message: str, *, seq, http_status,
                 response, cursor, pushed=None):
        super().__init__(message)
        self.seq = seq
        self.http_status = http_status
        self.response = response
        self.cursor = cursor
        self.pushed = list(pushed or [])


# -- deriving model / task, only when the row itself names them ---------------

def _derive_model(row: dict) -> str | None:
    """The model a journal row concerns, only when the row
    itself carries it as a string: the record's ``model``,
    the intent args' ``model``, or the row's own ``model``.
    Anything else is not derivable — None, never a guess."""
    if not isinstance(row, dict):
        return None
    record = row.get("record")
    if isinstance(record, dict) \
            and isinstance(record.get("model"), str) \
            and record.get("model"):
        return record["model"]
    intent = row.get("intent")
    if isinstance(intent, dict):
        args = intent.get("args")
        if isinstance(args, dict) \
                and isinstance(args.get("model"), str) \
                and args.get("model"):
            return args["model"]
    if isinstance(row.get("model"), str) and row.get("model"):
        return row["model"]
    return None


def _derive_task(row: dict) -> str | None:
    """The task a journal row concerns, as SI's task ref
    (``task@<seq>``), only when the row itself names it: a
    ``task_seq`` on the row or its record, a ``task_ref`` /
    ``task_id`` string on the record or row, a ``task@<seq>``
    parent on the intent, or — for the task entry row itself
    (an ``experiences`` row of kind ``task``) — the row's own
    seq. Anything else is not derivable — None."""
    if not isinstance(row, dict):
        return None
    for holder in (row, row.get("record")):
        if isinstance(holder, dict):
            seq = holder.get("task_seq")
            if isinstance(seq, int) and not isinstance(seq, bool):
                return f"task@{seq}"
            for key in ("task_ref", "task_id"):
                val = holder.get(key)
                if isinstance(val, str) and val:
                    return val
    intent = row.get("intent")
    if isinstance(intent, dict):
        for parent in intent.get("parents") or []:
            if isinstance(parent, str) \
                    and parent.startswith("task@"):
                return parent
    if row.get("kind") == "task" \
            and isinstance(row.get("seq"), int):
        return f"task@{row['seq']}"
    return None


# -- the StreamEvent shape (§4.5) ----------------------------------------------

def build_stream_event(event: dict) -> dict:
    """One journal event as a ``StreamEvent::from_si`` payload.

    ``event`` is one dict from
    :func:`si.gaia.events.read_journal_events` (``id``,
    ``event``, ``collection``, ``row``, optional
    ``queue_state``). The result carries exactly: ``source``
    ``"si"``, ``seq`` (the SI journal seq — the event's
    ``id``), ``kind`` (the SI event name), ``collection``,
    ``row`` (the full journal row, verbatim), ``row_kind``
    (the row's own ``kind`` or None), and ``facts``
    (``{"collection", "row"}``, plus ``queue_state`` only
    when the journal event carries one). ``model`` / ``task``
    keys appear only when derivable from the row without
    invention (see :func:`_derive_model` /
    :func:`_derive_task`). There is **no** ``hash`` key:
    hash is hearth-only per §4.5.
    """
    row = event.get("row") or {}
    collection = event.get("collection")
    seq = event.get("id", event.get("seq"))
    facts = {"collection": collection, "row": row}
    if event.get("queue_state") is not None:
        facts["queue_state"] = event["queue_state"]
    payload = {
        "source": "si",
        "seq": seq,
        "kind": event.get("event", event.get("kind")),
        "collection": collection,
        "row": row,
        "row_kind": row.get("kind")
        if isinstance(row, dict) else None,
        "facts": facts,
    }
    model = _derive_model(row)
    if model is not None:
        payload["model"] = model
    task = _derive_task(row)
    if task is not None:
        payload["task"] = task
    return payload


# -- reading the journal (pure reads; the forwarder journals nothing) ---------

def _read_events(bundle_dir: str, cursor: int,
                 limit: int | None = None) -> list:
    journal = open_journal(os.path.abspath(bundle_dir))
    try:
        return read_journal_events(journal, after_seq=cursor,
                                   limit=limit)
    finally:
        journal.close()


def pending_event_seqs(bundle_dir: str, cursor: int = 0) -> list:
    """The SI journal seqs after ``cursor``, ascending.

    A pure read (journals nothing, stores nothing): the seqs
    a forward from ``cursor`` would push, in push order.
    """
    if isinstance(cursor, bool) or not isinstance(cursor, int) \
            or cursor < 0:
        raise ValueError(
            f"cursor must be a non-negative integer journal "
            f"seq, got {cursor!r}")
    return [e["id"] for e in _read_events(bundle_dir, cursor)]


# -- the forwarder ---------------------------------------------------------------

def forward_journal_events(bundle_dir: str, *,
                           ingest_url: str | None = None,
                           transport=None,
                           cursor: int = 0,
                           limit: int | None = None,
                           timeout_s: float = 30.0) -> dict:
    """Push journal events after ``cursor`` to the fork ingest.

    One POST per event, in ascending SI journal seq order, to
    ``ingest_url`` — the full URL supplied whole by the
    operator (the documented fork route is
    :data:`INGEST_PATH`; no host is invented or defaulted and
    no path is appended to any base). The returned cursor
    advances only after an event's HTTP 200.

    Returns ``{"status": "forwarded", "cursor": <last acked
    seq>, "pushed": [...], "count": n, "transport": label,
    "label": ...}`` on success (including the empty window);
    ``{"status": "refused", "journaled": False, ...}`` when
    no transport and no URL are given; and ``{"status":
    "forward-failed", "cursor": <last acked>, "pushed":
    [...], "failure": ..., ...}`` when a transport exception
    or a non-200, non-409 answer stops the pass — the failed
    event is not acked, events after it are not pushed, and
    a retry from the returned cursor is free because history
    is the journal. An HTTP 409 (duplicate/refusal) raises
    :class:`SeamProtocolError`: a protocol failure, NOT a
    retry — nothing after it is pushed and the cursor does
    not advance past it. This function journals NOTHING.
    """
    if isinstance(cursor, bool) or not isinstance(cursor, int) \
            or cursor < 0:
        raise ValueError(
            f"cursor must be a non-negative integer journal "
            f"seq, got {cursor!r}")
    if limit is not None and (isinstance(limit, bool)
                              or not isinstance(limit, int)
                              or limit < 1):
        raise ValueError(
            f"limit must be a positive integer or None, "
            f"got {limit!r}")
    if transport is None and not ingest_url:
        return {"status": "refused", "journaled": False,
                "cursor": cursor, "pushed": [], "count": 0,
                "problems": [
                    "no transport injected and no ingest_url "
                    "given — there is no default fork endpoint "
                    f"(the documented fork route is "
                    f"{INGEST_PATH} on an operator-supplied "
                    "host); live interop is UNMEASURED"],
                "label": EVENT_FORWARD_LABEL}
    if transport is None:
        transport = HttpTransport()
    transport_label = getattr(transport, "label", "http")
    with _FORWARD_LOCK:
        events = _read_events(bundle_dir, cursor, limit=limit)
        pushed: list = []
        acked = cursor
        for event in events:
            seq = event["id"]
            payload = build_stream_event(event)
            try:
                status, raw = transport(
                    "POST", ingest_url or "",
                    json_body=payload, timeout_s=timeout_s)
            except Exception as e:  # failure is data, not a claim
                return {"status": "forward-failed",
                        "journaled": False,
                        "cursor": acked, "pushed": pushed,
                        "count": len(pushed),
                        "failure": f"{type(e).__name__}: {e}",
                        "failed_seq": seq,
                        "transport": transport_label,
                        "label": EVENT_FORWARD_LABEL}
            try:
                body = json.loads(raw.decode("utf-8")) \
                    if isinstance(raw, (bytes, bytearray)) else raw
            except (ValueError, UnicodeDecodeError):
                body = None
            if status == 409:
                # Duplicate / refusal: a protocol failure,
                # surfaced loudly — never retried, never
                # silently deduped (§7 principle 7 / SI-49).
                raise SeamProtocolError(
                    f"fork ingest refused SI journal event "
                    f"seq {seq} as a duplicate/refusal (HTTP "
                    f"409) — a protocol failure, not a retry: "
                    f"the two sides disagree about what was "
                    f"already pushed; cursor held at {acked}",
                    seq=seq, http_status=409, response=body,
                    cursor=acked, pushed=pushed)
            if status != 200:
                return {"status": "forward-failed",
                        "journaled": False,
                        "cursor": acked, "pushed": pushed,
                        "count": len(pushed),
                        "http_status": status, "response": body,
                        "failed_seq": seq,
                        "transport": transport_label,
                        "label": EVENT_FORWARD_LABEL}
            acked = seq  # cursor advances ONLY after the 200
            pushed.append(seq)
        return {"status": "forwarded", "journaled": False,
                "cursor": acked, "pushed": pushed,
                "count": len(pushed),
                "transport": transport_label,
                "label": EVENT_FORWARD_LABEL}


__all__ = [
    "EVENT_FORWARD_LABEL",
    "INGEST_PATH",
    "SeamProtocolError",
    "build_stream_event",
    "forward_journal_events",
    "pending_event_seqs",
]
