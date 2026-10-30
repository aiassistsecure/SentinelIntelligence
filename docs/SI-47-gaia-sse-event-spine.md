# SI-47 — Gaia SSE Event Spine: Journal-Derived Server-Sent Events
(SI-41 RFC §12 aftermath; Mark's 2026-10-29 SSE + replay decisions)

> **Status: BUILT, in review (PR opened by the SI-47 builder;
> merging is the dev-loop driver's, on a later run).** This
> slice builds Gaia's event spine on the SI side, in a new
> pure-Python module `si/gaia/events.py`, exposed through the
> existing B-5 service adapter as `GET /events` in
> `si/gaia/service.py`: every journal row serialised as a
> server-sent event whose **id IS the journal seq**, resumable
> via the standard `Last-Event-ID` header, replaying exactly
> the journal rows after that seq and then following live.
>
> **Origin.** The B-line the RFC sequenced (§15 B-1..B-5)
> completed with SI-46. This slice is the next frontier from
> Mark's standing 2026-10-29 direction: *"Gaia's event spine
> is SSE — loop transitions stream as server-sent events
> derived from the NEDB journal, which stays the source of
> truth"* (recorded in `~/workspace/si-dev-loop/STATE.md`,
> 2026-10-29 19:05 EDT update), plus Muse's design addition
> Mark accepted the same evening: **SSE event ids ARE journal
> sequence numbers (Last-Event-ID = journal seq)**, so the
> live stream and the audit replay are one mechanism —
> reconnect = resume from any point in the mind's history
> (STATE.md 19:11 EDT update, item 2 "SSE REPLAY").
>
> **The live-scope boundary, stated exactly.** LIVE (real
> code, exercised this slice): the journal→SSE serialisation
> (one row = one event, full row in the data envelope, event
> names derived by a deterministic mapping, §1.2); replay from
> any seq with row-for-row exactness against the raw journal
> collections (asserted by test, §4); resume via
> `Last-Event-ID` and the `?after=` query equivalent, header
> taking precedence, malformed resume points refused; a live
> tail that follows the journal **by polling it** every
> 0.25 s while a client is connected (labelled polling
> everywhere it is described — there is no push channel and
> no websocket, and none is claimed); the HTTP surface
> (`GET /events` on the B-5 server, `text/event-stream`,
> `Cache-Control: no-cache`, banner + heartbeat as SSE
> *comments* that carry no id and are never history); clean
> disconnect handling (a failed write ends the stream, the
> journal handle closes, handler threads are daemons so a
> connected stream can never block shutdown). **Exposure
> through the Hearth gateway is NOT built** — proposal H-2
> (§12.6), Mark's; the stream rides the B-5 service beside a
> gateway, exactly as B-5's surface does, under the same
> no-authentication honesty labels.
>
> **Terminology:** machine learning infrastructure
> throughout. CPU only; stdlib only; no GPU, no cloud, no
> spend, no downloads. No bytes are duplicated and **no new
> journal collection exists**: the stream is a serialisation
> of the journal's five existing collections, read fresh on
> every poll (Mark's design rule; RFC §4.6).

---

## 1. What was built

### 1.1 The serialiser (`si/gaia/events.py`)

Deliberately separable from HTTP, so replay semantics are
verified without a socket:

- `read_journal_events(source, after_seq=0, limit=None)` —
  merges all five journal collections
  (`si.journal.COLLECTIONS`: `self_edits`, `deltas`,
  `checkpoints`, `provenance`, `experiences`) and sorts by
  `seq`. This merge is the journal's own ordering —
  `Journal.verify_chain` performs exactly it, because `seq`
  is a single logical clock across collections, assigned by
  the one `_append` path. Each event is
  `{"id": seq, "event": name, "collection": coll, "row": row}`
  with `row` the full stored row (defensive JSON copy — a
  serialisation never aliases live journal state, the B-5
  trail's rule).
- `format_sse(event)` — `id: <seq>` / `event: <name>` /
  `data: <one JSON line>` / blank line. The data line is the
  envelope `{"collection": ..., "row": {...}}`: the complete
  row, so any consumer can verify any event against the
  journal by seq alone. No row is summarised, split, merged,
  or enriched from any other source.
- `format_comment(text)` — SSE comment blocks (`: ...`),
  used for the stream banner and the live-tail heartbeat.
  Comments carry no id and no event; they can never be
  mistaken for journal history, and a test parses them back
  out as non-events.
- `parse_resume_seq(header_value, query_value)` — the header
  wins when both are present (it is the standard reconnect
  mechanism; the query value is the explicit equivalent for
  clients that cannot set headers). Absent both → 0 (replay
  from the journal's start). A present value must be a
  non-negative integer; anything else raises `ValueError` —
  fail closed: a malformed resume point is refused, never
  silently treated as 0, which would quietly replay (or
  skip) the wrong history.
- `stream_events(source, after_seq=0, follow=False,
  poll_interval=0.25, should_stop=None, max_polls=None,
  heartbeat=True)` — the generator behind the HTTP surface.
  Replay first (exactly `read_journal_events(after_seq)`);
  with `follow=False` it then ends — a bounded replay-only
  stream. With `follow=True` it polls: each poll re-reads the
  backend (never a cached snapshot), so a row journaled by
  any writer appears within roughly one interval; idle polls
  yield a heartbeat comment. Following ends when
  `should_stop` fires or `max_polls` is reached (tests bound
  the tail with these; an HTTP client bounds it by
  disconnecting). The generator owns **no threads and no
  processes** — exhausting it, closing it, or walking away
  leaves nothing behind.

### 1.2 Event format mapping (journal row → SSE event)

| Journal source | Condition | SSE `event:` |
|---|---|---|
| `experiences` | `kind == "task"` | `task_open` |
| `experiences` | `kind == "task_stop"` | `task_stop` |
| `experiences` | `kind == "tool_result"` | `tool_result` |
| `experiences` | `kind == "hearth_residency"` | `hearth_residency` |
| `experiences` | any other `kind` `k` | `experience.<k>` |
| `self_edits` | `kind` `k` (tool / memory / …) | `self_edit.<k>` |
| `checkpoints` | — | `checkpoint` |
| `deltas` | `kind` `k` | `delta.<k>` |
| `deltas` | no `kind` | `delta.recorded` |
| `provenance` | `relation` `r` | `provenance.<r>` |
| `provenance` | no `relation` | `provenance.edge` |

**An honesty note the mapping makes explicit.** Mark's SSE
decision names the loop transitions *perceive / propose /
gate / journal / execute / result* plus task open/stop. The
journal does not carry those as separate rows, and this slice
does not invent them: a `self_edits` row **is** the
propose+gate transition journaled as one row (the intent and
its full gate report ride in the event's data), an
`experiences` row is a perception, an execution outcome
(`tool_result`), a task boundary, or a Hearth residency
observation (B-4's journaled rows — streamed because they are
journal rows, not because any new event source exists), per
its `kind`. The stream names what the journal actually
contains; consumers needing the gate verdict read it from the
`self_edit` event's data, where the journal put it.

### 1.3 The HTTP surface (`GET /events`, in `si/gaia/service.py`)

Rides the B-5 server unchanged in pattern (stdlib
`ThreadingHTTPServer`, state on `server.state`, localhost by
default, `si gaia-serve` serves it with no CLI change):

- Query surface: `?after=<seq>` (resume equivalent),
  `?last_event_id=<seq>` (accepted alias), `?follow=0`
  (replay-only: the response ends and the connection closes
  at the end of the replay — with no `Content-Length`, the
  close is how the client learns the replay is complete;
  following streams are `Connection: keep-alive`).
- The handler opens its **own** journal handle per request
  (the B-5 trail's per-request pattern), writes a banner
  comment carrying the resume point and both honesty labels,
  then streams. A failed write *is* the disconnect signal:
  the stream stops, the journal closes, the handler ends.
- `make_service_server`'s server is now a
  `ThreadingHTTPServer` subclass with `daemon_threads = True`
  — additive, and necessary once a route can legitimately
  stay open for a client's whole session: a still-connected
  stream can never block server shutdown or process exit.
  All pre-existing B-5 routes behave exactly as before
  (their tests pass unmodified, §5); routing now parses the
  URL path, which is identical for the plain paths every
  existing client uses and additionally lets `/tasks/<seq>`
  carry a query string.
- `GET /` and `GET /health` now also carry the two SSE
  labels (`events_tail`, `events_resume`) and the `/`
  description lists `GET /events`. The labels are
  single-sourced constants in `si.gaia.events`
  (`SSE_TAIL_LABEL`, `SSE_RESUME_LABEL`).
- Reading the stream **journals nothing** (asserted by test
  at both the module and HTTP level: every collection's row
  count is unchanged after replay and bounded-follow reads).

### 1.4 Resume semantics (the one-mechanism property)

Because event id == journal seq and the journal is
append-only under a hash chain, the three ways of reading
history are the same read at different cursors:

- full replay: `after = 0`;
- audit replay from a point: `after = <seq>` delivers exactly
  the rows with greater seq, in ascending order, row-for-row
  identical to the corresponding slice of a full replay
  (asserted by test against the raw collections);
- live reconnect: a client that last saw seq *N* reconnects
  with `Last-Event-ID: N` and continues with *N+1…* — no
  event store to reconcile, no gap the stream itself can
  introduce: any gap would be a gap in the journal, which
  `verify_chain` already refuses.

## 2. Pinned deviations and open fields (stated, never silent)

1. **The live tail is polling, with a 0.25 s interval.**
   The journal has no push/notification channel; pretending
   otherwise would be fabrication. The honest latency bound
   of the live edge is ~one poll interval plus the write.
   `SSE_TAIL_LABEL` says this on the service surface, the
   banner repeats it on every connection, and every idle
   heartbeat comment repeats it again. If a future journal
   gains a real notification channel, the tail's transport
   can change without the event format or resume semantics
   moving — the spine is the journal, not the poll.
2. **One row per event, even when a transition is compound.**
   A task cycle journals several rows (task entry, self-edit,
   provenance edges, outcome, checkpoint…); the stream shows
   each as its own event rather than synthesising a compound
   "cycle" event the journal never recorded. Consumers that
   want cycle grouping already have the chain to group by —
   the provenance trail (B-5) is that grouping, serialised
   from the same rows.
3. **No filtering or subscription surface.** `GET /events`
   streams the whole journal of the served bundle. Per-kind
   or per-task filters are a consumer-side projection today
   (the data envelope carries collection + full row, so the
   projection is lossless); a server-side filter is a future
   slice's design, not a silent omission — and any future
   filter must stay a projection of this same stream, never
   a second store.
4. **No authentication, unchanged from B-5.** The stream is
   a read of the bundle's journal served under B-5's exact
   discipline: submitter/identity labels untouched, no
   credential checked, localhost by default, no security
   claim. The journal is the mind's history; exposing this
   route beyond localhost without the operator's own
   controls would expose that history — the no-auth label
   rides the banner for exactly that reason.
5. **Backpressure is the socket's.** A slow consumer slows
   its own handler thread (writes flush per event block);
   there is no server-side event buffer to fall behind in,
   because there is no buffer at all — a consumer that falls
   behind can always disconnect and resume by seq, which is
   the design's answer to backpressure rather than a queue.

## 3. Hearth-side assumptions

None. This slice imports no Hearth code, contacts no Hearth
node, and depends on no H-1/H-2/H-3 proposal. The
`hearth_residency` events it streams are B-4's already-
journaled observations — including their replay/stub
transport labels, carried verbatim in the event data; nothing
here re-labels a stub observation as a real node's. The
Hearth path remains **UNMEASURED** (B-4's label, unchanged).
The Hearth fork ("Hearth becomes Gaia") remains Mark's step;
no fork was created and the hearth repo was not touched.
When the fork exists, this same serialiser is the SSE spine
the RFC's §12 aftermath points at — it is transport-shaped
(stdlib HTTP here) but journal-defined, so the fork carries
the format, not a rewrite of it.

## 4. Evidence (the builder's own runs on the PR head)

- **Replay exactness (not prose):** with a scripted B-5 task
  journaled in a miniature bundle, replay from a mid-journal
  seq returns exactly the raw-collection rows after that
  seq — ids, collections, and full row bodies compared
  row-for-row against `backend.find` over all five
  collections. Resuming at the tip returns nothing; resuming
  at 0 returns everything; event ids are unique and equal
  the total row count across collections (one row = one
  event, nothing dropped, nothing duplicated).
- **Loop transitions in order:** the task's events arrive as
  `task_open` → `self_edit.tool` → `tool_result` →
  `task_stop`, with `checkpoint` present, ids ascending; the
  `task_open` event's row carries the submission's asserted
  submitter identity, labelled as in B-5.
- **Live tail, really live:** a row journaled by a *separate*
  journal handle after the replay was consumed arrives as
  the next event with the next seq — over the pure-Python
  generator and over real localhost HTTP. A heartbeat on an
  idle tail parses back as a comment with no id.
- **Resume over HTTP:** `Last-Event-ID` and `?after=` both
  resume exactly; with both present the header wins;
  `Last-Event-ID: banana` is a 400 naming the problem, and
  nothing streams.
- **Tests:** `tests/test_gaia_sse.py` — 22 tests (serialisation
  exactness, mapping table, resume rules, replay-only stream,
  live tail, heartbeat-as-comment, stop conditions,
  journals-nothing, honesty labels, HTTP surface incl.
  headers and disconnect-shaped reads). Full suite numbers
  in §5.

## 5. Test results

- Baseline on main `dc2a675` (driver-verified last run):
  **554 passed, 2 skipped**. (The builder did not re-run a
  baseline full suite this slice; the driver's solo run on
  the PR #51 head — the tree main `dc2a675` merged — is the
  recorded baseline, and §5's head number is the builder's
  own measured run.)
- New file alone (`tests/test_gaia_sse.py`): **22 passed**.
- Regression spot-check before the full run
  (`tests/test_gaia_b5.py`, the slice whose module this one
  extends): **24 passed** — no existing test modified.
- Full suite on the PR head (builder's own SOLO run,
  workspace venv, orphaned `nedbd-v2` processes checked and
  none present before the run): **576 passed, 2 skipped**
  in 1120.61 s (554 baseline + 22 new; no existing test
  modified). No failures this run — the known rotating
  stale-daemon flake in pre-existing real-NEDB-backend tests
  (disclosed in SI-46 §5) did not appear; had a single such
  failure appeared, the adjudication rule from SI-46 would
  apply (reproduce on pristine main in the same environment
  before attributing it to this diff — this slice's diff
  touches no journal, bundle, seed, or loop code).

## 6. Explicitly NOT built (do not read this slice as claiming them)

- **No exposure through the Hearth gateway** (proposal H-2),
  no gateway route, and no Hearth code of any kind. The
  Hearth fork is Mark's step; the hearth repo was not
  touched. The Hearth path remains UNMEASURED.
- **No push, no websocket, no notification channel** — the
  live tail is poll-the-journal at 0.25 s, labelled as
  polling in the module, the doc, the service surface, the
  connection banner, and every heartbeat.
- **No new event store, no new journal collection, no
  journal schema change** (§4.6): the stream is a
  serialisation of the existing five collections; reading
  it writes nothing.
- **No authentication or authorization added**: the stream
  inherits B-5's no-auth posture and labels verbatim; no
  submitter/identity semantics changed.
- **No server-side filtering, subscription, or per-task
  streams** (§2.3), no async task surface, and no change to
  B-5's synchronous submission.
- **No new execution scope and no new proposer**: live
  execution remains miniature-scoped (B-2/B-3); demo bundles
  remain shadow. Nothing in the stream executes anything.
- **A-2/A-3/A-4 and any real-base agency** — unapproved, not
  built. Real-base agency claims stay honestly scoped until
  measured.
