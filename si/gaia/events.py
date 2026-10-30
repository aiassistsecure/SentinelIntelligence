"""Gaia SSE event spine: journal-derived server-sent events (SI-47).

Mark's 2026-10-29 direction for Gaia: the event spine is SSE —
loop transitions stream as server-sent events **derived from
the NEDB journal**, which stays the source of truth. Muse's
design addition, accepted the same evening: **SSE event ids
ARE journal sequence numbers** (``Last-Event-ID`` = journal
seq), so the live stream and the audit replay are one
mechanism — reconnect = resume from any point in the mind's
history.

This module is that spine, kept separable from HTTP so the
replay semantics are testable without a socket:

- **One journal row = one event.** The journal's ``seq`` is a
  single logical clock across all five collections
  (:data:`si.journal.COLLECTIONS` — ``verify_chain`` itself
  merges every collection and sorts by ``seq``), so the stream
  is the merge of all collections sorted by ``seq``. An event's
  ``id`` is exactly its row's ``seq``; its ``data`` is an
  envelope ``{"collection": ..., "row": {...}}`` carrying the
  full journal row, unabridged. No row is summarised, split, or
  merged, and nothing is read from anywhere but the journal:
  the stream is a serialisation of the journal, never a
  separate event store. **No new journal collection, no bytes
  duplicated** (Mark's design rule).
- **Event names** are derived deterministically from the row
  (see :func:`event_name_for_row` and the mapping table in
  ``docs/SI-47-gaia-sse-event-spine.md``). One honesty note the
  mapping makes explicit: the journal does **not** carry
  separate perceive / propose / gate rows — a ``self_edits``
  row *is* the propose+gate transition journaled as one row
  (the intent and its full gate report), and an ``experiences``
  row is a perception, an execution outcome, a task boundary,
  or a Hearth residency observation, per its ``kind``. The
  stream names what the journal actually contains; it does not
  invent finer-grained transitions than the journal records.
- **Resume.** :func:`parse_resume_seq` accepts the standard
  ``Last-Event-ID`` header value (and the ``after`` query
  equivalent the B-5 service surface supports). Replay from a
  seq delivers exactly the journal rows with ``seq`` greater
  than it, in ascending order — byte-for-byte the rows a fresh
  full replay would show for those seqs.
- **Live tail is polling, labelled as such.** The journal has
  no push/notification channel, so following the stream means
  re-reading the journal on an interval
  (:data:`DEFAULT_POLL_INTERVAL_SECONDS`). This module does
  not claim push or websocket semantics. While following with
  no new rows, the iterator emits SSE *comment* lines
  (``: heartbeat ...``) — comments carry no id and no event,
  so they can never be mistaken for journal history.

Stdlib only, CPU only. No Hearth code is imported and no
Hearth gateway exposure is built (that route is proposal H-2,
Mark's); the Hearth residency observations streamed here are
only the ones B-4 already journaled as ``experiences`` rows.
"""
from __future__ import annotations

import json
import time

from si.journal import COLLECTIONS

#: How often a following stream re-reads the journal. The tail
#: is polling — there is no push channel in the journal — and
#: this interval is the honest latency bound of the live edge:
#: a newly journaled row appears within roughly one interval.
DEFAULT_POLL_INTERVAL_SECONDS = 0.25

#: The live-tail honesty label, carried on the service surface.
SSE_TAIL_LABEL = (
    "live tail is poll-the-journal: the journal is re-read "
    f"every {DEFAULT_POLL_INTERVAL_SECONDS}s while a client is "
    "connected — there is no push channel and no websocket; "
    "the NEDB journal remains the source of truth and every "
    "event is a serialisation of one journal row (SI-47)")

#: The resume-semantics label, carried on the service surface.
SSE_RESUME_LABEL = (
    "event id == journal seq: resume with the standard "
    "Last-Event-ID header (or the ?after=<seq> query "
    "equivalent) and the stream replays exactly the journal "
    "rows after that seq, in order, then follows live "
    "(SI-47)")


# -- reading the journal as an event sequence ---------------------------------

def _backend_of(source):
    """Accept a Journal, or a bare backend, and return the backend."""
    return getattr(source, "backend", source)


def read_journal_events(source, after_seq: int = 0,
                        limit: int | None = None) -> list:
    """Every journal row with ``seq > after_seq``, as events, in order.

    ``source`` is a :class:`si.journal.Journal` or its backend.
    Rows are merged across all five journal collections and
    sorted by the journal's global ``seq`` — the same merge
    ``Journal.verify_chain`` performs — so the result is the
    journal's own ordering, not a per-collection concatenation.
    Each event is ``{"id": seq, "event": name, "collection":
    coll, "row": row}`` with ``row`` the full stored row.
    """
    backend = _backend_of(source)
    entries = []
    for coll in COLLECTIONS:
        for row in backend.find(coll):
            seq = row.get("seq") or 0
            if seq > after_seq:
                entries.append((seq, coll, row))
    entries.sort(key=lambda e: e[0])
    events = [{
        "id": seq,
        "event": event_name_for_row(coll, row),
        "collection": coll,
        "row": _jsonable_row(row),
    } for seq, coll, row in entries]
    if limit is not None:
        events = events[:limit]
    return events


def _jsonable_row(row: dict) -> dict:
    """A journal row as plain JSON data (defensive copy, the
    B-5 trail's rule: a serialisation never aliases live state)."""
    return json.loads(json.dumps(row, default=str))


def event_name_for_row(collection: str, row: dict) -> str:
    """The SSE event name for one journal row — deterministic.

    Mapping (full table in the SI-47 slice doc):

    - ``experiences`` rows name their transition directly:
      ``task`` -> ``task_open``, ``task_stop`` -> ``task_stop``,
      ``tool_result`` -> ``tool_result``, ``hearth_residency``
      -> ``hearth_residency``; any other experience kind ``k``
      -> ``experience.<k>``.
    - ``self_edits`` rows -> ``self_edit.<kind>`` (e.g.
      ``self_edit.tool``, ``self_edit.memory``) — the row is the
      propose+gate transition, carrying the intent and the full
      gate report in its data.
    - ``checkpoints`` -> ``checkpoint``; ``deltas`` ->
      ``delta.<kind>`` or ``delta.recorded``; ``provenance`` ->
      ``provenance.<relation>`` or ``provenance.edge``.
    - Anything unrecognised -> ``journal.<collection>`` — a
      name is always derivable and never fabricated finer than
      the row supports.
    """
    kind = row.get("kind")
    if collection == "experiences":
        if kind == "task":
            return "task_open"
        if kind == "task_stop":
            return "task_stop"
        if kind == "tool_result":
            return "tool_result"
        if kind == "hearth_residency":
            return "hearth_residency"
        return f"experience.{kind}" if kind else "experience"
    if collection == "self_edits":
        return f"self_edit.{kind}" if kind else "self_edit"
    if collection == "checkpoints":
        return "checkpoint"
    if collection == "deltas":
        return f"delta.{kind}" if kind else "delta.recorded"
    if collection == "provenance":
        relation = row.get("relation")
        return f"provenance.{relation}" if relation else "provenance.edge"
    return f"journal.{collection}"


# -- SSE serialisation ----------------------------------------------------------

def format_sse(event: dict) -> str:
    """Serialise one event dict as an SSE block.

    ``id: <journal seq>`` / ``event: <name>`` / ``data: <one
    JSON line>`` / blank line. The data line is the envelope
    ``{"collection": ..., "row": {...}}`` — the full journal
    row, so a consumer can verify any event against the journal
    by seq alone.
    """
    data = json.dumps({"collection": event["collection"],
                       "row": event["row"]},
                      sort_keys=True, default=str)
    return f"id: {event['id']}\nevent: {event['event']}\ndata: {data}\n\n"


def format_comment(text: str) -> str:
    """An SSE comment block (``: ...``) — carries no id and no
    event; used for the stream banner and heartbeats so neither
    can be mistaken for journal history."""
    lines = [ln for ln in str(text).splitlines() if ln] or [""]
    return "".join(f": {ln}\n" for ln in lines) + "\n"


# -- resume ---------------------------------------------------------------------

def parse_resume_seq(header_value=None, query_value=None) -> int:
    """Resolve the resume seq from Last-Event-ID / ?after= values.

    The header wins when both are present (it is the standard
    reconnect mechanism; the query value is the explicit
    equivalent for clients that cannot set headers). Absent
    both, the resume seq is 0 — replay from the journal's
    start. A present value must be a non-negative integer;
    anything else raises ValueError (fail closed: a malformed
    resume point is refused, never silently treated as 0,
    which would quietly replay — or skip — the wrong history).
    """
    raw = header_value if header_value not in (None, "") else query_value
    if raw is None or raw == "":
        return 0
    try:
        seq = int(str(raw).strip())
    except (TypeError, ValueError):
        raise ValueError(
            f"resume seq must be a non-negative integer journal "
            f"seq, got {raw!r}") from None
    if seq < 0:
        raise ValueError(
            f"resume seq must be a non-negative integer journal "
            f"seq, got {raw!r}")
    return seq


# -- the stream -------------------------------------------------------------------

def stream_events(source, after_seq: int = 0, follow: bool = False,
                  poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
                  should_stop=None, max_polls: int | None = None,
                  heartbeat: bool = True):
    """Yield SSE text blocks for the journal, from ``after_seq``.

    Replay first: every row with ``seq > after_seq``, in order,
    exactly as :func:`read_journal_events` returns them. With
    ``follow=False`` the generator then ends — a bounded,
    replay-only stream.

    With ``follow=True`` it tails the journal by polling:
    re-reads every ``poll_interval`` seconds and yields any new
    rows (a row journaled by any writer — the read goes to the
    backend each poll, never to a cached snapshot). On an idle
    poll it yields a heartbeat comment when ``heartbeat`` is
    true. Following ends when ``should_stop`` (a callable
    returning True) fires, or after ``max_polls`` idle-or-total
    polls when given (tests bound the tail with these; a real
    HTTP client bounds it by disconnecting, which the service
    handler turns into ``should_stop`` / a failed write).

    The generator owns no threads and no processes: stopping
    it — by exhausting it, closing it, or letting the consumer
    walk away — leaves nothing behind.
    """
    cursor = after_seq
    polls = 0
    while True:
        events = read_journal_events(source, after_seq=cursor)
        for event in events:
            cursor = event["id"]
            yield format_sse(event)
        if not follow:
            return
        polls += 1
        if max_polls is not None and polls >= max_polls:
            return
        if should_stop is not None and should_stop():
            return
        if not events and heartbeat:
            yield format_comment(
                f"heartbeat — following journal after seq {cursor} "
                f"(poll-the-journal every {poll_interval}s, SI-47)")
            # A heartbeat consumer (the HTTP handler) may have
            # walked away during the yield; re-check before sleeping.
            if should_stop is not None and should_stop():
                return
        time.sleep(poll_interval)


__all__ = [
    "DEFAULT_POLL_INTERVAL_SECONDS",
    "SSE_RESUME_LABEL",
    "SSE_TAIL_LABEL",
    "event_name_for_row",
    "format_comment",
    "format_sse",
    "parse_resume_seq",
    "read_journal_events",
    "stream_events",
]
