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

**SI-49 — the resilient following consumer.** :func:`follow_events`
adds the reconnect/resume layer SI-48 §6 explicitly left unbuilt,
on top of this same consumer: it maintains one filtered or
unfiltered subscription across connection drops, reconnecting
with ``Last-Event-ID`` set to the last seq actually received,
with bounded exponential backoff (configurable base, cap, max
attempts and max duration). It never re-delivers an already-seen
seq: a duplicate or out-of-order seq from the server is a loud
:class:`FollowError`, not silent acceptance, and malformed
resume data fails closed the same way — protocol errors are
never retried, only transport errors are. The received window
remains verifiable against the journal through the SI-48
:func:`verify_events_against_journal` — there is no second
verification path. **It is still a reference follower, not a
production client**: no TLS or auth story is claimed (the
service has none), no Hearth gateway is involved (H-2 is not
built), and the server tail it follows is still the SI-47
poll-the-journal tail — no push, no websocket.

Stdlib only, CPU only. Reading through this client journals
nothing — it performs HTTP GETs of a read-only surface and,
for verification, read-only journal finds.
"""
from __future__ import annotations

import inspect
import json
import time
import urllib.error
import urllib.parse
import urllib.request

from si.gaia.events import EventFilter, apply_event_filter

#: The reference-consumer label, printed by the CLI surface.
CONSUMER_LABEL = (
    "reference consumer (SI-48): a stdlib-only demonstration "
    "and verification client for the Gaia SSE spine — not a "
    "production client library; no retries, no auth, no TLS "
    "story of its own")

#: The SI-49 follower label, printed by the CLI surface in
#: resilient-following mode.
FOLLOWER_LABEL = (
    "reference follower (SI-49): a stdlib-only resilient "
    "following consumer for the Gaia SSE spine — reconnects "
    "with Last-Event-ID resume and bounded exponential "
    "backoff, dedupes by journal seq, and fails closed on "
    "malformed resume; still a reference implementation, "
    "not a production client library; no auth, no TLS story "
    "of its own; the server tail remains poll-the-journal "
    "(SI-47), no push, no websocket")


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
                 follow: bool = False, timeout: float = 30.0,
                 should_stop=None):
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

    ``should_stop`` (SI-49, optional): a callable returning
    True ends the generator cleanly at the next SSE block —
    heartbeat comment blocks included, so an idle following
    stream (heartbeats only) still honours the stop within one
    heartbeat; without it the SI-48 behaviour is unchanged.

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
            if should_stop is not None and should_stop():
                return
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


# -- SI-49: the resilient following consumer -------------------------------------

class FollowError(RuntimeError):
    """A following subscription failed closed.

    Raised for protocol failures (a duplicate or out-of-order
    seq, a malformed event block, a server stream that violates
    the requested filter on reconnect) and for exhausted
    reconnect bounds. Protocol failures are never retried:
    resuming against a peer that misstates journal seqs would
    manufacture exactly the duplicated or gapped history the
    spine exists to prevent.
    """


#: Transport failures worth a reconnect: the connection dropped,
#: was refused, reset, timed out, or the HTTP layer reported a
#: transient error. Everything else fails closed (FollowError).
_TRANSPORT_ERRORS = (urllib.error.URLError, ConnectionError,
                    TimeoutError, OSError)

#: Default read-wake interval for a following subscription: the
#: per-connection socket timeout the follower installs when the
#: caller gives none, so an idle live stream wakes periodically
#: and the follower can re-check its stop discipline and bounds
#: instead of blocking inside one read forever. A wake is not a
#: failure: it reconnects from the last delivered seq without
#: consuming backoff or attempt budget (see follow_events).
DEFAULT_FOLLOW_WAKE_SECONDS = 2.0


def _is_wake_timeout(exc: BaseException) -> bool:
    """True when a transport error is a read/connect timeout —
    the follower's own wake interval firing, not a peer failure."""
    if isinstance(exc, TimeoutError):
        return True
    reason = getattr(exc, "reason", None)
    return isinstance(reason, TimeoutError)


class FollowConfig:
    """Bounds for one resilient following subscription (SI-49).

    ``backoff_base`` / ``backoff_cap``: the delay before the
    n-th consecutive reconnect is
    ``min(cap, base * 2 ** (n - 1))`` seconds — deterministic,
    no jitter, so the bound is exactly what the config states.
    ``max_attempts``: maximum consecutive failed reconnects
    before :class:`FollowError` (``None`` = bounded only by
    ``max_duration``). ``max_duration``: maximum total seconds
    spent following, measured from the first connection
    attempt (``None`` = no duration bound). ``max_events``:
    stop cleanly after delivering this many events (``None`` =
    follow until stopped). At least the caller's own stop
    discipline (``should_stop`` / closing the generator) or one
    of these bounds should be set for a live subscription;
    an unbounded follower runs until the caller stops it,
    exactly like the SI-48 ``follow=True`` stream.
    """

    __slots__ = ("backoff_base", "backoff_cap", "max_attempts",
                 "max_duration", "max_events")

    def __init__(self, backoff_base: float = 0.5,
                 backoff_cap: float = 30.0,
                 max_attempts: int | None = None,
                 max_duration: float | None = None,
                 max_events: int | None = None):
        if backoff_base <= 0:
            raise ValueError("backoff_base must be positive")
        if backoff_cap < backoff_base:
            raise ValueError(
                "backoff_cap must be >= backoff_base (the cap is "
                "a bound on the backoff, not a second schedule)")
        if max_attempts is not None and max_attempts < 1:
            raise ValueError("max_attempts must be >= 1 when given")
        if max_duration is not None and max_duration <= 0:
            raise ValueError("max_duration must be positive when given")
        if max_events is not None and max_events < 1:
            raise ValueError("max_events must be >= 1 when given")
        self.backoff_base = float(backoff_base)
        self.backoff_cap = float(backoff_cap)
        self.max_attempts = max_attempts
        self.max_duration = (None if max_duration is None
                             else float(max_duration))
        self.max_events = max_events

    def backoff_for(self, consecutive_failures: int) -> float:
        """The delay before reconnect attempt ``consecutive_failures``
        (1-based): exponential, capped, deterministic."""
        if consecutive_failures < 1:
            raise ValueError("consecutive_failures is 1-based")
        return min(self.backoff_cap,
                   self.backoff_base * (2 ** (consecutive_failures - 1)))

    def describe(self) -> str:
        return (f"backoff base={self.backoff_base}s cap={self.backoff_cap}s "
                f"max_attempts={self.max_attempts} "
                f"max_duration={self.max_duration} "
                f"max_events={self.max_events}")


def follow_events(base_url: str, after_seq: int = 0,
                  event_filter: EventFilter | None = None,
                  config: FollowConfig | None = None,
                  should_stop=None, sleep_fn=None, now_fn=None,
                  fetch_factory=None, timeout: float | None = None):
    """Yield events from a running Gaia service, resiliently (SI-49).

    One logical subscription, maintained across connection
    drops: each (re)connection is the SI-48 :func:`fetch_events`
    with ``follow=True`` and the resume point set to the last
    seq actually delivered by this generator — so the server
    replays exactly the journal rows after it, projected through
    ``event_filter``, and no already-delivered event is requested
    again. As a second, independent guard, every received id is
    checked against the last delivered id: a duplicate or
    out-of-order seq (``id <= last delivered``) raises
    :class:`FollowError` immediately — a server that replays
    from the wrong point, or renumbers, is a loud protocol
    failure, never silently deduped history.

    Only transport failures are retried (connection refused /
    dropped / reset / timed out, HTTP transport errors), with
    the bounded exponential backoff of ``config``: the
    consecutive-failure counter resets once a connection
    delivers at least one event (the subscription demonstrably
    made progress). Exhausting ``max_attempts`` consecutive
    failures, or exceeding ``max_duration``, raises
    :class:`FollowError` naming the bound that fired and the
    last delivered seq — the caller can resume a fresh follower
    from exactly that seq. Protocol failures — duplicate /
    out-of-order seqs, malformed SSE, a stream violating the
    requested filter (the SI-48 client-side re-check) — raise
    :class:`FollowError` at once and are never retried.

    ``should_stop`` (a callable returning True) ends the
    generator cleanly between events, before each reconnect, and
    at each read wake. ``sleep_fn`` / ``now_fn`` are injectable
    for tests (defaults: :func:`time.sleep` /
    :func:`time.monotonic`); ``fetch_factory`` is the transport
    seam for tests (defaults to :func:`fetch_events`, called
    with the :func:`fetch_events` signature). ``timeout`` is
    the per-connection socket timeout handed to the transport;
    when it is ``None`` the follower installs
    :data:`DEFAULT_FOLLOW_WAKE_SECONDS` as a *wake* interval:
    a read that times out on an otherwise healthy subscription
    is a wake, not a failure — the follower reconnects from the
    last delivered seq without consuming backoff or attempt
    budget, having first re-checked ``should_stop`` and
    ``max_duration``. (An event block interrupted mid-read by a
    wake was never delivered, so the resume replays it whole;
    the dedupe guard would catch any partial delivery loudly.)

    Yields the same ``{"id", "event", "collection", "row"}``
    dicts as :func:`fetch_events`. Journals nothing.
    """
    config = config if config is not None else FollowConfig()
    sleep = sleep_fn if sleep_fn is not None else time.sleep
    now = now_fn if now_fn is not None else time.monotonic
    factory = fetch_factory if fetch_factory is not None else fetch_events
    if timeout is None:
        timeout = DEFAULT_FOLLOW_WAKE_SECONDS
    # The stop discipline is handed to the transport when it
    # accepts it (the real fetch_events does, SI-49): an idle
    # following stream is kept warm by heartbeats, so a socket
    # timeout alone can never wake it — the per-block check in
    # fetch_events is what makes should_stop prompt there.
    try:
        _factory_accepts_stop = "should_stop" in \
            inspect.signature(factory).parameters
    except (TypeError, ValueError):
        _factory_accepts_stop = False
    started = now()
    last = after_seq
    delivered = 0
    failures = 0
    while True:
        if should_stop is not None and should_stop():
            return
        if config.max_duration is not None \
                and now() - started > config.max_duration:
            raise FollowError(
                f"following subscription exceeded max_duration "
                f"{config.max_duration}s after delivering "
                f"{delivered} event(s); last delivered seq is "
                f"{last} — resume a fresh follower from that seq")
        try:
            if _factory_accepts_stop:
                stream = factory(base_url, after_seq=last,
                                 event_filter=event_filter, follow=True,
                                 timeout=timeout,
                                 should_stop=should_stop)
            else:
                stream = factory(base_url, after_seq=last,
                                 event_filter=event_filter, follow=True,
                                 timeout=timeout)
            for event in stream:
                eid = event["id"]
                if eid <= last:
                    raise FollowError(
                        f"server delivered seq {eid} after seq "
                        f"{last} was already delivered — duplicate "
                        f"or out-of-order resume is a protocol "
                        f"failure, not history to dedupe silently; "
                        f"refusing the stream")
                last = eid
                delivered += 1
                failures = 0
                yield event
                if config.max_events is not None \
                        and delivered >= config.max_events:
                    return
                if should_stop is not None and should_stop():
                    return
        except FollowError:
            raise
        except ValueError as e:
            # The SI-48 layer's fail-closed signals: a stream
            # outside the requested filter, or a malformed SSE
            # block / resume value. Protocol, not transport.
            raise FollowError(
                f"following subscription failed closed: {e} "
                f"(last delivered seq is {last})") from e
        except _TRANSPORT_ERRORS as e:
            if _is_wake_timeout(e):
                # Our own wake interval fired on an idle stream:
                # not a peer failure — loop (the top re-checks
                # should_stop / max_duration) and reconnect from
                # the last delivered seq, budget untouched.
                continue
            if isinstance(e, urllib.error.HTTPError) and e.code < 500:
                # A 4xx on (re)connect — e.g. the server refusing
                # a malformed resume point with 400 — is a
                # protocol refusal, not a drop: fail closed.
                raise FollowError(
                    f"following subscription refused by the "
                    f"server with HTTP {e.code} ({e}); last "
                    f"delivered seq is {last} — refusing to "
                    f"retry a refused resume") from e
            failures += 1
            if config.max_attempts is not None \
                    and failures >= config.max_attempts:
                raise FollowError(
                    f"following subscription exhausted "
                    f"max_attempts {config.max_attempts} "
                    f"consecutive reconnects (last error: "
                    f"{type(e).__name__}: {e}); last delivered "
                    f"seq is {last} — resume a fresh follower "
                    f"from that seq") from e
            if should_stop is not None and should_stop():
                return
            delay = config.backoff_for(failures)
            if config.max_duration is not None \
                    and now() - started + delay > config.max_duration:
                raise FollowError(
                    f"following subscription exceeded max_duration "
                    f"{config.max_duration}s after {failures} "
                    f"consecutive failure(s); last delivered seq "
                    f"is {last} — resume a fresh follower from "
                    f"that seq") from e
            sleep(delay)
            continue
        # The transport generator ended without an error: for a
        # following stream that is a server-side close — a drop
        # by another name. Reconnect under the same bounds.
        failures += 1
        if config.max_attempts is not None \
                and failures >= config.max_attempts:
            raise FollowError(
                f"following subscription exhausted max_attempts "
                f"{config.max_attempts} consecutive reconnects "
                f"(the server closed the stream); last delivered "
                f"seq is {last} — resume a fresh follower from "
                f"that seq")
        if should_stop is not None and should_stop():
            return
        delay = config.backoff_for(failures)
        if config.max_duration is not None \
                and now() - started + delay > config.max_duration:
            raise FollowError(
                f"following subscription exceeded max_duration "
                f"{config.max_duration}s after {failures} "
                f"consecutive failure(s); last delivered seq is "
                f"{last} — resume a fresh follower from that seq")
        sleep(delay)


def collect_followed(base_url: str, after_seq: int = 0,
                     event_filter: EventFilter | None = None,
                     config: FollowConfig | None = None,
                     verify_source=None, **kwargs) -> dict:
    """A bounded resilient follow, collected, optionally verified.

    Runs :func:`follow_events` to its bound (``config`` should
    carry ``max_events`` and/or ``max_duration``, or the caller
    passes ``should_stop`` via ``kwargs``) and returns
    ``{"events": [...], "last_seq": n, "reconnects": None,
    "verify": report-or-None}``. When ``verify_source`` (a
    journal or its backend, as in
    :func:`verify_events_against_journal`) is given, the
    received window is verified with that same SI-48 function —
    the one verification path, reused over the followed window,
    reconnects included. A failed verification does not raise:
    it is reported in ``verify`` exactly as SI-48 reports it,
    so the caller sees the itemised problems.
    """
    events = list(follow_events(base_url, after_seq=after_seq,
                                event_filter=event_filter,
                                config=config, **kwargs))
    report = None
    if verify_source is not None:
        report = verify_events_against_journal(
            events, verify_source, event_filter=event_filter)
    return {"events": events,
            "last_seq": events[-1]["id"] if events else after_seq,
            "verify": report}


__all__ = [
    "CONSUMER_LABEL",
    "DEFAULT_FOLLOW_WAKE_SECONDS",
    "FOLLOWER_LABEL",
    "FollowConfig",
    "FollowError",
    "build_events_url",
    "collect_events",
    "collect_followed",
    "events_from_text",
    "fetch_events",
    "follow_events",
    "iter_sse",
    "parse_sse_block",
    "verify_events_against_journal",
]
