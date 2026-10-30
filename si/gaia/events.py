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

**SI-53 — the queue-event convention (SI-50 §15.5, pinned).**
A queue event's payload carries the §5.2 state name the row
transitions its intent *to*, as ``queue_state`` — on the event
dict from :func:`read_journal_events` and in the serialised
SSE envelope beside ``collection``/``row`` — so consumers do
not each re-implement the fold to answer "what is running."
Non-queue rows carry no ``queue_state`` key at all; event
names are unchanged from SI-47/48. The mapping is
:func:`queue_state_for_row`.

**SI-48 — filtered subscriptions.** :class:`EventFilter` adds
the server-side filter SI-47 §2.3 deferred, under exactly the
rule that section pinned: a filter is a **pure projection of
this same stream** — it selects events from the one journal
serialisation, creates no store, journals nothing, and never
renumbers: a filtered event's ``id`` is still its journal
``seq``, so gaps in a filtered stream are the filtered-out
seqs, labelled as such (:data:`SSE_FILTER_LABEL`), not missing
history. Dimensions: ``collection`` (one of the five journal
collections), ``event`` (the derived SSE event name),
``kind`` (the journal row's own ``kind`` field), and ``task``
(a task seq — membership derived from the journal's own
§10.2 chain, see :func:`task_member_seqs`). Values within one
dimension are alternatives (OR); dimensions combine
conjunctively (AND). Malformed filter values fail closed
with ``ValueError`` (the service maps it to HTTP 400), exactly
like malformed resume points. With no filter, the stream is
byte-identical to SI-47.
"""
from __future__ import annotations

import json
import re
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

#: The SI-48 filter label, carried on the service surface and
#: on every filtered connection's banner.
SSE_FILTER_LABEL = (
    "filtered streams (SI-48) are a projection of the same "
    "journal stream: event ids remain journal seqs and are "
    "never renumbered, so gaps in a filtered stream are the "
    "filtered-out seqs, not missing history; filters create "
    "no store and journal nothing")

#: Filter query parameters recognised on GET /events (SI-48).
FILTER_PARAMS = ("collection", "event", "kind", "task")

_FILTER_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


#: The resume-semantics label, carried on the service surface.
SSE_RESUME_LABEL = (
    "event id == journal seq: resume with the standard "
    "Last-Event-ID header (or the ?after=<seq> query "
    "equivalent) and the stream replays exactly the journal "
    "rows after that seq, in order, then follows live "
    "(SI-47)")


# -- SI-48: filtered subscriptions (a projection, never a store) -------------

class EventFilter:
    """A server-side filter over the journal event stream.

    Each dimension is ``None`` (unconstrained) or a frozenset
    of accepted values; ``task_seq`` is ``None`` or one task's
    journal seq. An event matches when every constrained
    dimension accepts it. ``collection`` / ``event`` / ``kind``
    are per-event checks (:meth:`matches_static`); ``task``
    membership is a property of the journal's §10.2 chain, not
    of one row alone, so it is resolved over an event universe
    by :func:`task_member_seqs` and applied by
    :func:`apply_event_filter` / the read/stream paths.
    """

    __slots__ = ("collections", "event_names", "kinds", "task_seq")

    def __init__(self, collections=None, event_names=None, kinds=None,
                 task_seq=None):
        self.collections = (frozenset(collections)
                            if collections is not None else None)
        self.event_names = (frozenset(event_names)
                            if event_names is not None else None)
        self.kinds = frozenset(kinds) if kinds is not None else None
        self.task_seq = task_seq

    def is_empty(self) -> bool:
        """True when no dimension is constrained (SI-47 stream)."""
        return (self.collections is None and self.event_names is None
                and self.kinds is None and self.task_seq is None)

    def matches_static(self, event: dict) -> bool:
        """The per-event dimensions: collection, event name, kind.

        A row with no ``kind`` field (checkpoints, provenance)
        never matches a kind filter — the filter names a field
        the row does not carry, it does not invent one.
        """
        if self.collections is not None \
                and event["collection"] not in self.collections:
            return False
        if self.event_names is not None \
                and event["event"] not in self.event_names:
            return False
        if self.kinds is not None:
            kind = event["row"].get("kind")
            if kind is None or kind not in self.kinds:
                return False
        return True

    def describe(self) -> str:
        """A one-line human description, for banners and logs."""
        if self.is_empty():
            return "no filter (full journal stream, SI-47)"
        parts = []
        if self.collections is not None:
            parts.append("collection in {"
                         + ",".join(sorted(self.collections)) + "}")
        if self.event_names is not None:
            parts.append("event in {"
                         + ",".join(sorted(self.event_names)) + "}")
        if self.kinds is not None:
            parts.append("kind in {"
                         + ",".join(sorted(self.kinds)) + "}")
        if self.task_seq is not None:
            parts.append(f"task == {self.task_seq} "
                         f"(journal §10.2 chain membership)")
        return "filter: " + " AND ".join(parts) \
            + " — ids remain journal seqs; gaps are filtered-out seqs"

    def __eq__(self, other):
        return isinstance(other, EventFilter) and all(
            getattr(self, f) == getattr(other, f) for f in self.__slots__)

    def __repr__(self):
        return f"EventFilter({self.describe()})"


def _filter_values(params: dict, name: str) -> list:
    """All values given for one filter param: repeated params
    and comma-separated values both count (``?kind=a&kind=b``
    and ``?kind=a,b`` are the same filter)."""
    raw = params.get(name) or []
    if isinstance(raw, str):
        raw = [raw]
    values = []
    for item in raw:
        values.extend(str(item).split(","))
    return values


def parse_event_filter(params: dict | None) -> EventFilter:
    """Build an :class:`EventFilter` from query params; fail closed.

    ``params`` maps names to value lists (``parse_qs`` shape) or
    single strings. Only :data:`FILTER_PARAMS` are read; other
    params (``after``, ``follow``…) are the caller's business.
    Every supplied token is validated: an unknown collection, a
    blank value, a token with characters no collection / event
    name / kind can carry, or a non-integer task seq each raise
    ``ValueError`` — a malformed filter is refused outright,
    never silently widened to the unfiltered stream (which
    would quietly deliver history the subscriber did not ask
    for — and, read the other way, never silently narrowed).
    """
    params = params or {}
    parsed: dict = {}
    for name in ("collection", "event", "kind"):
        values = _filter_values(params, name)
        if not values:
            continue
        for v in values:
            if not v or not _FILTER_TOKEN_RE.match(v):
                raise ValueError(
                    f"filter {name} value {v!r} is malformed: "
                    f"values are non-empty tokens of letters, "
                    f"digits, '_', '.', '-' (comma-separated or "
                    f"repeated)")
        if name == "collection":
            unknown = [v for v in values if v not in COLLECTIONS]
            if unknown:
                raise ValueError(
                    f"filter collection value(s) {unknown!r} are "
                    f"not journal collections; the journal's "
                    f"collections are {list(COLLECTIONS)!r}")
            parsed["collections"] = values
        elif name == "event":
            parsed["event_names"] = values
        else:
            parsed["kinds"] = values
    task_values = _filter_values(params, "task")
    if task_values:
        if len(task_values) != 1:
            raise ValueError(
                f"filter task takes exactly one task seq, got "
                f"{task_values!r}")
        raw = task_values[0]
        try:
            task_seq = int(raw.strip())
        except (TypeError, ValueError):
            raise ValueError(
                f"filter task must be a non-negative integer "
                f"journal seq (the seq of a task entry), got "
                f"{raw!r}") from None
        if task_seq < 0:
            raise ValueError(
                f"filter task must be a non-negative integer "
                f"journal seq (the seq of a task entry), got "
                f"{raw!r}")
        parsed["task_seq"] = task_seq
    return EventFilter(**parsed)


def task_member_seqs(events: list, task_seq: int) -> set:
    """The seqs belonging to one task, derived from the journal rows.

    Membership is the B-5 provenance trail's §10.2 chain,
    computed over ``events`` (the full journal serialisation —
    pass the unfiltered, untruncated list; membership derived
    from a truncated universe would silently drop chained
    rows): the task entry itself; its ``task_stop`` (whose
    record cites the task seq); every ``self_edits`` row whose
    intent cites a seq already in the set, or names
    ``task@<seq>`` among its parents, transitively; the
    ``provenance`` edges of those edits; the ``experiences``
    outcomes recorded against those edits; and the ``deltas``
    rows those edits materialised. Nothing else is a member —
    in particular checkpoints are shared journal bookkeeping,
    not task steps, and stay out. An unknown task seq yields
    just ``{task_seq}`` if such a row exists, else the empty
    set — membership is derived, never fabricated.
    """
    by_id = {e["id"]: e for e in events}
    task_event = by_id.get(task_seq)
    if task_event is None or not (
            task_event["collection"] == "experiences"
            and (task_event["row"].get("record") or {}).get("kind")
            == "task"):
        return set()
    task_ref = f"task@{task_seq}"
    members = {task_seq}
    edit_ids: set = set()
    for e in events:
        row = e["row"]
        if e["collection"] == "experiences" \
                and (row.get("record") or {}).get("task_seq") == task_seq:
            members.add(e["id"])
    changed = True
    while changed:
        changed = False
        for e in events:
            if e["id"] in members:
                continue
            row = e["row"]
            if e["collection"] == "self_edits":
                intent = row.get("intent") or {}
                cited = set(intent.get("source_seqs") or [])
                parents = set(intent.get("parents") or [])
                if cited & members or task_ref in parents:
                    members.add(e["id"])
                    if row.get("edit_id"):
                        edit_ids.add(row["edit_id"])
                    changed = True
            elif e["collection"] in ("provenance", "deltas"):
                if row.get("child") in edit_ids \
                        or row.get("self_edit_id") in edit_ids:
                    members.add(e["id"])
                    changed = True
            elif e["collection"] == "experiences":
                if row.get("self_edit_id") in edit_ids:
                    members.add(e["id"])
                    changed = True
    return members


def apply_event_filter(events: list, event_filter: EventFilter | None,
                       universe: list | None = None) -> list:
    """Project ``events`` through ``event_filter`` (SI-48).

    ``universe`` is the full journal serialisation the task
    dimension derives membership from; it defaults to
    ``events`` itself (correct whenever ``events`` is a full
    replay). Order and ids are untouched — projection only.
    """
    if event_filter is None or event_filter.is_empty():
        return list(events)
    members = None
    if event_filter.task_seq is not None:
        members = task_member_seqs(
            universe if universe is not None else events,
            event_filter.task_seq)
    return [e for e in events
            if event_filter.matches_static(e)
            and (members is None or e["id"] in members)]


# -- reading the journal as an event sequence ---------------------------------

def _backend_of(source):
    """Accept a Journal, or a bare backend, and return the backend."""
    return getattr(source, "backend", source)


def read_journal_events(source, after_seq: int = 0,
                        limit: int | None = None,
                        event_filter: EventFilter | None = None) -> list:
    """Every journal row with ``seq > after_seq``, as events, in order.

    ``source`` is a :class:`si.journal.Journal` or its backend.
    Rows are merged across all five journal collections and
    sorted by the journal's global ``seq`` — the same merge
    ``Journal.verify_chain`` performs — so the result is the
    journal's own ordering, not a per-collection concatenation.
    Each event is ``{"id": seq, "event": name, "collection":
    coll, "row": row}`` with ``row`` the full stored row.

    With ``event_filter`` (SI-48) the result is that same list
    projected through the filter: task membership is derived
    from the full journal (read from seq 0), never from the
    truncated ``after_seq`` window, and ``limit`` applies
    after filtering. Without a filter the result is exactly
    SI-47's.
    """
    backend = _backend_of(source)
    entries = []
    for coll in COLLECTIONS:
        for row in backend.find(coll):
            seq = row.get("seq") or 0
            if seq > after_seq:
                entries.append((seq, coll, row))
    entries.sort(key=lambda e: e[0])
    events = []
    for seq, coll, row in entries:
        event = {
            "id": seq,
            "event": event_name_for_row(coll, row),
            "collection": coll,
            "row": _jsonable_row(row),
        }
        queue_state = queue_state_for_row(coll, row)
        if queue_state is not None:
            event["queue_state"] = queue_state
        events.append(event)
    if event_filter is not None and not event_filter.is_empty():
        universe = events
        if event_filter.task_seq is not None and after_seq > 0:
            universe = read_journal_events(source, after_seq=0)
        events = apply_event_filter(events, event_filter,
                                    universe=universe)
    if limit is not None:
        events = events[:limit]
    return events


def _jsonable_row(row: dict) -> dict:
    """A journal row as plain JSON data (defensive copy, the
    B-5 trail's rule: a serialisation never aliases live state)."""
    return json.loads(json.dumps(row, default=str))


def queue_state_for_row(collection: str, row: dict) -> str | None:
    """The SI-50 §5.2 queue state one journal row transitions
    its intent TO (SI-53/Q-3, pinning §15.5's convention), or
    None when the row is not a queue transition.

    Queue event payloads carry this name as ``queue_state`` so
    a consumer can answer "what is running / held / unknown"
    without re-implementing the fold. The mapping names only
    what the row itself establishes: a ``tool_result`` row
    transitions to ``executed`` (or ``refused-at-claim`` when
    its execution status is ``refused``); a ``self_edits`` tool
    row transitions by its journaled gate verdict (ADMIT ->
    ``queued``, REJECT -> ``rejected``, ESCALATE ->
    ``escalated``, otherwise ``proposed``); the ``queue_*``
    experience kinds map to their §5.2 states (``queue_stop``
    and ``queue_resume`` transition the task's still-queued
    intents to ``held`` / back to ``queued`` respectively);
    ``queue_recovery_observation`` and ``queue_disposition``
    annotate an ``outcome-unknown`` intent without moving it,
    so they carry ``outcome-unknown``. Compensation is an
    annotation in the fold (§5.2), never a state, so no row
    carries ``compensated`` here.
    """
    kind = row.get("kind")
    if collection == "self_edits":
        if kind != "tool":
            return None
        verdict = row.get("gate_verdict")
        return {"ADMIT": "queued", "REJECT": "rejected",
                "ESCALATE": "escalated"}.get(verdict, "proposed")
    if collection != "experiences":
        return None
    if kind == "tool_result":
        record = row.get("record") or {}
        execution = record.get("execution") or {}
        return "refused-at-claim" \
            if execution.get("status") == "refused" else "executed"
    return {
        "queue_claim": "claimed",
        "queue_recovery_reclaim": "claimed",
        "queue_refusal": "refused-at-claim",
        "queue_skip": "skipped",
        "queue_outcome_unknown": "outcome-unknown",
        "queue_recovery_observation": "outcome-unknown",
        "queue_disposition": "outcome-unknown",
        "queue_stop": "held",
        "queue_resume": "queued",
    }.get(kind)


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
    envelope = {"collection": event["collection"],
                "row": event["row"]}
    if event.get("queue_state") is not None:
        # SI-53 (§15.5 convention): a queue event's payload
        # carries the §5.2 state name it transitions to.
        envelope["queue_state"] = event["queue_state"]
    data = json.dumps(envelope, sort_keys=True, default=str)
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
                  heartbeat: bool = True,
                  event_filter: EventFilter | None = None):
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

    With ``event_filter`` (SI-48) each poll reads the
    *unfiltered* window and projects it: the cursor advances
    over every scanned row, matching or not, so filtered-out
    rows are never re-scanned and the resume point is always a
    true journal seq — while only matching rows are yielded,
    with their original seqs as ids (gaps are the filtered-out
    seqs). Task membership is re-derived from the full journal
    each poll, so a chained row journaled mid-tail joins its
    task's stream. Without a filter the yielded blocks are
    byte-identical to SI-47.

    The generator owns no threads and no processes: stopping
    it — by exhausting it, closing it, or letting the consumer
    walk away — leaves nothing behind.
    """
    filtering = event_filter is not None and not event_filter.is_empty()
    cursor = after_seq
    polls = 0
    while True:
        scanned = read_journal_events(source, after_seq=cursor)
        if filtering:
            universe = scanned
            if event_filter.task_seq is not None:
                universe = read_journal_events(source, after_seq=0)
                # The universe read stops at the old cursor for
                # rows scanned before; rows in `scanned` extend it.
                seen = {e["id"] for e in universe}
                universe = universe + [e for e in scanned
                                       if e["id"] not in seen]
            events = apply_event_filter(scanned, event_filter,
                                        universe=universe)
        else:
            events = scanned
        for event in scanned:
            if event["id"] > cursor:
                cursor = event["id"]
        for event in events:
            yield format_sse(event)
        if not follow:
            return
        polls += 1
        if max_polls is not None and polls >= max_polls:
            return
        if should_stop is not None and should_stop():
            return
        if not events and heartbeat:
            label = "SI-48, filtered" if filtering else "SI-47"
            yield format_comment(
                f"heartbeat — following journal after seq {cursor} "
                f"(poll-the-journal every {poll_interval}s, {label})")
            # A heartbeat consumer (the HTTP handler) may have
            # walked away during the yield; re-check before sleeping.
            if should_stop is not None and should_stop():
                return
        time.sleep(poll_interval)


__all__ = [
    "DEFAULT_POLL_INTERVAL_SECONDS",
    "FILTER_PARAMS",
    "SSE_FILTER_LABEL",
    "SSE_RESUME_LABEL",
    "SSE_TAIL_LABEL",
    "EventFilter",
    "apply_event_filter",
    "event_name_for_row",
    "format_comment",
    "format_sse",
    "parse_event_filter",
    "parse_resume_seq",
    "queue_state_for_row",
    "read_journal_events",
    "stream_events",
    "task_member_seqs",
]
