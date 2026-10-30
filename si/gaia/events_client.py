"""Gaia SI-48: the reference consumer for the SSE event spine.

A small **stdlib-only** client for the B-5 service's
``GET /events`` surface (SI-47 spine + SI-48 filtered
subscriptions). It exists so the filtered stream has one
worked, inspectable consumer: it connects, **resumes from a
``Last-Event-ID``**, asks the server for the SI-48 projection
(and re-checks every received event against the same
:class:`si.gaia.events.EventFilter` client-side, so a server
that ignored the filter is visible rather than trusted), and
can **verify** the events it received against a bundle's
journal — each event's id is a journal seq, so verification is
a row-for-row comparison, the same exactness the SI-47 tests
assert server-side.

**Honesty label.** This is a *reference consumer*, not a
production client library: no retries, no reconnect loop, no
TLS or auth story (the service itself has none — B-5's labels
stand), no buffering beyond one SSE block. It demonstrates and
verifies the wire contract; production consumers keep their
own transport discipline and resume from the last id this
module reports.

Stdlib only, CPU only. Reading through this client journals
nothing — it performs HTTP GETs of a read-only surface and,
for verification, read-only journal finds.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

from si.gaia.events import EventFilter, apply_event_filter

#: The reference-consumer label, printed by the CLI surface.
CONSUMER_LABEL = (
    "reference consumer (SI-48): a stdlib-only demonstration "
    "and verification client for the Gaia SSE spine — not a "
    "production client library; no retries, no auth, no TLS "
    "story of its own")


# -- SSE wire parsing -----------------------------------------------------------

def parse_sse_block(block: str) -> dict | None:
    """Parse one SSE block (text between blank lines).

    Returns ``{"id": int, "event": name, "data": envelope}``
    for an event block, ``{"comment": text}`` for a comment-only
    block (banner / heartbeat — carries no id, never history),
    or ``None`` for an empty block.
    """
    event: dict = {}
    comments = []
    for line in block.splitlines():
        if line.startswith(":"):
            comments.append(line[1:].strip())
        elif line.startswith("id:"):
            event["id"] = int(line[3:].strip())
        elif line.startswith("event:"):
            event["event"] = line[6:].strip()
        elif line.startswith("data:"):
            event["data"] = json.loads(line[5:].strip())
    if "id" in event:
        return event
    if comments:
        return {"comment": "\n".join(comments)}
    return None


def iter_sse(lines):
    """Yield parsed blocks from an iterable of text lines.

    Works over a live response (lines arrive as the server
    flushes them) or over a list of lines from a captured body.
    A trailing block without its closing blank line is still
    yielded at exhaustion.
    """
    buf: list = []
    for line in lines:
        line = line.rstrip("\n")
        if line == "":
            if buf:
                parsed = parse_sse_block("\n".join(buf))
                buf = []
                if parsed is not None:
                    yield parsed
        else:
            buf.append(line)
    if buf:
        parsed = parse_sse_block("\n".join(buf))
        if parsed is not None:
            yield parsed


def events_from_text(text: str) -> list:
    """Every event in a captured SSE body (comments dropped)."""
    return [b for b in iter_sse(text.splitlines()) if "id" in b]


# -- connecting -------------------------------------------------------------------

def build_events_url(base_url: str, after_seq: int = 0,
                     event_filter: EventFilter | None = None,
                     follow: bool = False) -> str:
    """The GET /events URL for a resume point + filter.

    The resume point rides the ``Last-Event-ID`` header on the
    request itself (see :func:`fetch_events`); ``after`` is
    also placed in the query so the URL alone is inspectable
    and copyable — the server gives the header precedence and
    both carry the same value here, so precedence never bites.
    Filter dimensions serialise as repeated query params in
    the exact shape :func:`si.gaia.events.parse_event_filter`
    reads.
    """
    query: dict = {"follow": "1" if follow else "0"}
    if after_seq:
        query["after"] = str(after_seq)
    if event_filter is not None and not event_filter.is_empty():
        if event_filter.collections is not None:
            query["collection"] = sorted(event_filter.collections)
        if event_filter.event_names is not None:
            query["event"] = sorted(event_filter.event_names)
        if event_filter.kinds is not None:
            query["kind"] = sorted(event_filter.kinds)
        if event_filter.task_seq is not None:
            query["task"] = str(event_filter.task_seq)
    qs = urllib.parse.urlencode(query, doseq=True)
    return base_url.rstrip("/") + "/events?" + qs


def fetch_events(base_url: str, after_seq: int = 0,
                 event_filter: EventFilter | None = None,
                 follow: bool = False, timeout: float = 30.0):
    """Yield event dicts from a running Gaia service.

    Each yielded dict is ``{"id", "event", "collection",
    "row"}`` — the wire envelope flattened to the
    :mod:`si.gaia.events` event shape, so consumers (and the
    verifier below) handle server events and journal events
    identically. Comment blocks (banner, heartbeats) are not
    yielded. With ``follow=False`` the generator ends when the
    replay-only response closes; with ``follow=True`` it yields
    live rows until the caller stops iterating (closing the
    generator closes the response).

    Resume discipline: pass the last ``id`` you durably
    recorded as ``after_seq`` on the next call — it is sent as
    the standard ``Last-Event-ID`` header, and the stream
    continues with exactly the journal rows after it (projected
    through the filter). Every received event is re-checked
    against ``event_filter``'s static dimensions client-side;
    a non-matching event raises ``ValueError`` rather than
    being silently accepted (the task dimension is a chain
    property the client cannot re-derive from a projected
    stream, so it is verified against the journal instead —
    see :func:`verify_events_against_journal`).
    """
    url = build_events_url(base_url, after_seq=after_seq,
                           event_filter=event_filter, follow=follow)
    headers = {}
    if after_seq:
        headers["Last-Event-ID"] = str(after_seq)
    req = urllib.request.Request(url, headers=headers)
    resp = urllib.request.urlopen(req, timeout=timeout)
    try:
        lines = (raw.decode("utf-8") for raw in resp)
        for block in iter_sse(lines):
            if "id" not in block:
                continue
            event = {"id": block["id"], "event": block["event"],
                     "collection": block["data"]["collection"],
                     "row": block["data"]["row"]}
            if event_filter is not None \
                    and not event_filter.matches_static(event):
                raise ValueError(
                    f"server sent event id {event['id']} "
                    f"({event['event']}) outside the requested "
                    f"filter's static dimensions — refusing to "
                    f"present a stream that does not match its "
                    f"subscription")
            yield event
    finally:
        resp.close()


def collect_events(base_url: str, after_seq: int = 0,
                   event_filter: EventFilter | None = None,
                   timeout: float = 30.0) -> list:
    """A bounded replay: every event after ``after_seq``."""
    return list(fetch_events(base_url, after_seq=after_seq,
                             event_filter=event_filter, follow=False,
                             timeout=timeout))


# -- verification -----------------------------------------------------------------

def verify_events_against_journal(events: list, source,
                                  event_filter: EventFilter | None = None
                                  ) -> dict:
    """Verify received events against the journal, row for row.

    ``source`` is a :class:`si.journal.Journal` or its backend
    for the same bundle the service serves. Checks, per event:
    the id is a real journal seq; the collection and the full
    row body equal the journal's row at that seq; ids are
    strictly ascending with no duplicates. When
    ``event_filter`` is given, additionally: the received id
    set equals exactly the journal's own projection through
    that filter (over the seq window the events span — from
    the journal's start through the last received id, so a
    filtered-out row *inside* the window is expected, and a
    missing matching row inside the window is a failure).
    Returns ``{"ok": bool, "checked": n, "problems": [...]}`` —
    problems are itemised, never summarised away.
    """
    from si.gaia.events import read_journal_events
    journal_events = read_journal_events(source, after_seq=0)
    by_id = {e["id"]: e for e in journal_events}
    problems: list = []
    seen: set = set()
    prev = None
    for event in events:
        eid = event["id"]
        if eid in seen:
            problems.append(f"duplicate event id {eid}")
        seen.add(eid)
        if prev is not None and eid <= prev:
            problems.append(
                f"event ids not strictly ascending: {eid} after {prev}")
        prev = eid
        truth = by_id.get(eid)
        if truth is None:
            problems.append(f"event id {eid} is not a journal seq")
            continue
        if event["collection"] != truth["collection"]:
            problems.append(
                f"event {eid}: collection {event['collection']!r} "
                f"!= journal {truth['collection']!r}")
        if event["event"] != truth["event"]:
            problems.append(
                f"event {eid}: name {event['event']!r} != journal "
                f"{truth['event']!r}")
        if json.loads(json.dumps(event["row"], default=str)) \
                != truth["row"]:
            problems.append(
                f"event {eid}: row body differs from the journal "
                f"row at that seq")
    if event_filter is not None and not event_filter.is_empty() and events:
        window = [e for e in journal_events if e["id"] <= events[-1]["id"]]
        expected = [e["id"] for e in
                    apply_event_filter(window, event_filter,
                                       universe=journal_events)]
        got = [e["id"] for e in events]
        if got != expected:
            problems.append(
                f"received ids {got} != the journal's own "
                f"projection {expected} over the same window")
    return {"ok": not problems, "checked": len(events),
            "problems": problems}


__all__ = [
    "CONSUMER_LABEL",
    "build_events_url",
    "collect_events",
    "events_from_text",
    "fetch_events",
    "iter_sse",
    "parse_sse_block",
    "verify_events_against_journal",
]
