# SI-48 — Gaia SSE Filtered Subscriptions + Reference Consumer
(SI-47 §2.3's deferred slice: server-side filters as a projection)

> **Status: BUILT, in review (PR opened by the SI-48 builder;
> merging is the dev-loop driver's, on a later run).** This
> slice builds the server-side filtering SI-47 explicitly
> deferred, plus a reference consumer: `GET /events` on the
> B-5 service now accepts `collection` / `event` / `kind` /
> `task` filters that **project the same journal stream**,
> and `si/gaia/events_client.py` (CLI: `si gaia-events`) is a
> stdlib-only reference consumer that connects, resumes from
> `Last-Event-ID`, and verifies received events against the
> journal row for row.
>
> **Origin.** SI-47 §2.3, pinned when the spine was built:
> *"No filtering or subscription surface… a server-side
> filter is a future slice's design, not a silent omission —
> and any future filter must stay a projection of this same
> stream, never a second store."* SI-48 is that future slice,
> built under exactly that rule, per Mark's standing
> 2026-10-29 direction to develop the Gaia agentic loop slice
> by slice ("expand and expose and let's endure").
>
> **The live-scope boundary, stated exactly.** LIVE (real
> code, exercised this slice): server-side filtering on
> `GET /events` as a pure projection of the SI-47 stream —
> four dimensions (`collection`, `event` name, row `kind`,
> and `task` seq with membership derived from the journal's
> own §10.2 chain, §1.2); event ids remain journal seqs under
> filtering, **never renumbered**, so gaps in a filtered
> stream are the filtered-out seqs and are labelled as such
> on the service surface, the connection banner, and the
> filter's own description; resume semantics unchanged
> (`Last-Event-ID` / `?after=` compose with filters, header
> precedence intact, task membership derived from the full
> journal so resuming past a task entry loses none of its
> chained steps); malformed filter values fail closed with
> HTTP 400, like malformed resume points; the filtered live
> tail (still poll-the-journal at 0.25 s, labelled polling);
> and the reference consumer — connect, resume, server-side
> filter with a client-side static re-check, and journal
> verification. **Unfiltered behaviour is byte-identical to
> SI-47** (asserted by test at the generator level and over
> HTTP, banner included). **Exposure through the Hearth
> gateway is NOT built** — proposal H-2 (§12.6), Mark's; the
> stream rides the B-5 service beside a gateway, exactly as
> before, under the same no-authentication honesty labels.
>
> **Terminology:** machine learning infrastructure
> throughout. CPU only; stdlib only; no GPU, no cloud, no
> spend, no downloads. No bytes are duplicated and **no new
> journal collection exists**: a filter reads the journal's
> five existing collections and selects from their
> serialisation — it stores nothing, journals nothing
> (asserted by test), and writes no state anywhere.

---

## 1. What was built

### 1.1 The filter (`si/gaia/events.py`)

- `EventFilter` — one subscription: `collections`,
  `event_names`, `kinds` (each `None` = unconstrained, else a
  frozenset of accepted values) and `task_seq` (`None` or one
  task's journal seq). Values within a dimension are
  alternatives (OR); dimensions combine conjunctively (AND).
  `matches_static()` applies the per-event dimensions;
  `describe()` renders the filter for banners and logs,
  including the gaps-are-filtered-out-seqs label.
- `parse_event_filter(params)` — builds a filter from query
  params (`parse_qs` shape). Repeated params and
  comma-separated values are the same filter
  (`?kind=a&kind=b` ≡ `?kind=a,b`). Validation fails closed
  with `ValueError` (the service maps it to 400): an unknown
  collection, a blank value, a token carrying characters no
  collection / event name / kind can carry, a non-integer or
  negative task seq, or more than one task value are each
  refused outright — a malformed filter is never silently
  widened to the unfiltered stream (which would quietly
  deliver history the subscriber did not ask for) nor
  silently narrowed. Non-filter params (`after`, `follow`)
  are ignored by the parser; they remain the resume path's
  business.
- `read_journal_events(..., event_filter=None)` and
  `stream_events(..., event_filter=None)` — the SI-47 read
  and stream, projected. In the stream, each poll reads the
  *unfiltered* window and projects it: the cursor advances
  over every scanned row, matching or not, so filtered-out
  rows are never re-scanned and the resume point is always a
  true journal seq, while only matching rows are yielded —
  with their original seqs as ids. Task membership is
  re-derived from the full journal each poll, so a chained
  row journaled mid-tail joins its task's stream. With no
  filter (or an empty one) the yielded blocks are
  byte-identical to SI-47 — the unfiltered code path, banner,
  and heartbeat text are untouched.

### 1.2 The four dimensions, exactly

| Param | Matches | Derivation |
|---|---|---|
| `collection` | the journal collection the row lives in | one of the five `si.journal.COLLECTIONS`; anything else is a 400 |
| `event` | the derived SSE event name (SI-47 §1.2 mapping) | `event_name_for_row` on the row — the same name the unfiltered stream carries |
| `kind` | the journal row's own `kind` field | rows that carry no `kind` (checkpoints, provenance) never match a kind filter — the filter names a field, it does not invent one |
| `task` | the events of one task | §10.2 chain membership, §1.3 |

### 1.3 Task membership (`task_member_seqs`)

A task is not a field on most of its rows — it is the §10.2
chain the B-5 provenance trail already walks. Membership is
that chain, computed over the full journal serialisation:
the task entry itself; its `task_stop` (whose record cites
the task seq); every `self_edits` row whose intent cites a
seq already in the set, or names `task@<seq>` among its
parents, transitively; the `provenance` edges of those
edits; the `experiences` outcomes recorded against those
edits; and the `deltas` rows those edits materialised.
Checkpoints are shared journal bookkeeping, not task steps,
and stay out. An unknown or non-task seq yields the empty
set — membership is derived, never fabricated. The tests
cross-check this set against `build_provenance_trail` (B-5's
independently written chain walk): the filtered ids equal
the trail's seqs exactly, for two distinct tasks, with no
overlap between them.

One consequence worth stating plainly: because membership
needs the whole chain, a task-filtered read derives it from
the full journal (from seq 0) even when the caller resumes
mid-journal — resuming *past* a task entry therefore still
delivers its later chained steps (asserted by test). The
cost is a full-journal read per filtered request/poll; at
the journal scales this runtime serves, that is the honest
trade for never deriving membership from a truncated window.

### 1.4 The HTTP surface (`GET /events`, in `si/gaia/service.py`)

Unchanged in pattern; additive in surface:

- New query params `collection`, `event`, `kind`, `task`
  (repeatable / comma-separated), parsed with
  `keep_blank_values=True` so a blank filter value reaches
  validation and is refused rather than vanishing. Filter
  and resume validation share one fail-closed path: either
  failing is a 400 naming the problem, and nothing streams.
- The banner names the slice honestly: an unfiltered
  connection still opens with the SI-47 banner verbatim; a
  filtered connection opens `gaia-events (SI-48)` and
  carries the filter's description plus `SSE_FILTER_LABEL`
  ("ids remain journal seqs and are never renumbered, so
  gaps in a filtered stream are the filtered-out seqs, not
  missing history; filters create no store and journal
  nothing").
- `GET /` and `GET /health` now also carry `events_filter`
  (`SSE_FILTER_LABEL`), and the `/` description's
  `GET /events` entry documents the filter params. All
  pre-existing B-5 routes and labels behave exactly as
  before (their tests pass unmodified, §5).

### 1.5 The reference consumer (`si/gaia/events_client.py`, `si gaia-events`)

A small stdlib-only client, in `si/cli.py`'s established
pattern beside `gaia-serve`:

- `fetch_events(base_url, after_seq=0, event_filter=None,
  follow=False)` — yields `{"id", "event", "collection",
  "row"}` dicts (the wire envelope flattened to the
  `si.gaia.events` event shape). The resume point is sent as
  the standard `Last-Event-ID` header (and mirrored as
  `?after=` so the URL alone is inspectable; the server gives
  the header precedence and both carry the same value, so
  precedence never bites). Filters serialise as the exact
  query shape `parse_event_filter` reads. Every received
  event is re-checked against the filter's static dimensions
  client-side: a server that ignored the filter produces a
  loud `ValueError`, not a silently wrong stream. (The task
  dimension is a chain property a projected stream cannot
  re-derive — it is verified against the journal instead.)
- `collect_events(...)` — the bounded replay form.
- `verify_events_against_journal(events, source,
  event_filter=None)` — per event: the id is a real journal
  seq; collection, event name, and the full row body equal
  the journal's row at that seq; ids strictly ascending, no
  duplicates. With a filter, additionally: the received id
  set equals exactly the journal's own projection over the
  window the events span — a matching row dropped inside the
  window is a failure, a filtered-out row inside it is
  expected. Problems are itemised, never summarised away.
- `parse_sse_block` / `iter_sse` / `events_from_text` — the
  wire parser (comment blocks parse as comments: no id,
  never history), usable over a live response or a captured
  body.
- CLI: `si gaia-events [--url U] [--after SEQ]
  [--collection C] [--event E] [--kind K] [--task SEQ]
  [--follow] [--json] [--verify BUNDLE]` — prints the
  consumer label and the filter description, then one line
  per event (`id ⇥ event ⇥ collection`, or the full envelope
  with `--json`), then the resume hint (`last id N (resume:
  si gaia-events --after N)`). `--verify` runs the journal
  verification after a bounded replay and exits 1 on any
  mismatch; it refuses `--follow` (an unbounded stream has
  no completed window to verify).
- **Honesty label** (`CONSUMER_LABEL`, printed by the CLI):
  this is a *reference consumer*, not a production client
  library — no retries, no reconnect loop, no TLS or auth
  story of its own (the service has none; B-5's labels
  stand). It demonstrates and verifies the wire contract.

## 2. Pinned deviations and open fields (stated, never silent)

1. **Gaps are the semantics, not an error.** A filtered
   stream's ids jump over filtered-out seqs by design; the
   alternative — renumbering — would break the one property
   the spine exists for (id == journal seq, resume = replay).
   The gap label rides the service surface, every filtered
   banner, and the consumer's filter description, so no
   reader of any surface can mistake a gap for lost history.
2. **Filtering costs a full read; there is no index and no
   cache.** Each filtered request re-reads the journal's
   five collections and projects in memory; a following
   filtered stream re-derives task membership each poll.
   Nothing is materialised between requests — that is what
   "projection, never a second store" costs, stated plainly.
   At miniature journal scales this is negligible; a future
   slice that needs server-side scale must still derive from
   the journal, not cache a second event store beside it.
3. **The task dimension is chain membership, not a field
   match.** It is deliberately the B-5 trail's membership —
   one definition of "this task's events" across the whole
   system — rather than a cheaper `task_seq` field scan that
   would miss chained self-edits (they cite seqs, they do
   not carry the task seq as a field). Cross-checked against
   the trail by test (§4).
4. **Filter validation is syntactic, not existential.** An
   `event` or `kind` value that is well-formed but matches
   nothing is a valid subscription to an empty stream, not a
   400 — the filter language cannot enumerate the journal's
   kinds without reading it, and an empty result is honest
   data. Only `collection` is enumerated (the journal's five
   collections are a closed, schema-level set), so only it
   400s on an unknown value.
5. **No authentication, unchanged from B-5/SI-47.** Filters
   are a read projection, not an authorization boundary: a
   filtered stream under no-auth is still a read of the
   bundle's journal, served under B-5's exact discipline
   (localhost by default, no security claim). Nothing here
   narrows or widens who may read what.

## 3. Hearth-side assumptions

None. This slice imports no Hearth code, contacts no Hearth
node, and depends on no H-1/H-2/H-3 proposal. Filtered
streams carry `hearth_residency` events only as the journal
rows B-4 already recorded, labels verbatim. The Hearth path
remains **UNMEASURED** (B-4's label, unchanged). The Hearth
fork ("Hearth becomes Gaia") remains Mark's step; no fork
was created and the hearth repo was not touched. When the
fork exists, the filter is a projection of the same journal
serialisation the fork carries — the format moves, the
projection moves with it.

## 4. Evidence (the builder's own runs on the PR head)

- **Filter exactness (not prose):** for each collection, the
  filtered ids equal exactly the raw-collection rows of that
  collection (full row bodies compared); event-name and kind
  filters equal the unfiltered stream's matching events
  computed independently; dimensions combine conjunctively
  (collection=experiences ∧ kind=task → exactly the one
  `task_open` event; ∧ kind=tool → empty).
- **Task chain:** with two scripted B-5 tasks journaled in
  one bundle, each task filter's ids equal its B-5
  provenance trail's seq set exactly (task + stop + steps +
  provenance edges + outcomes), the two sets are disjoint,
  no checkpoint leaks in, and an unknown/non-task seq
  filters to empty.
- **Resume under filtering:** filtered replay after a
  mid-journal seq is exactly the filtered full replay after
  that seq — including for the task filter resumed past its
  own task entry (membership from the full journal). Over
  HTTP, `Last-Event-ID` composes with `?task=` the same way.
- **Gaps, not renumbering:** filtered ids are the original
  journal seqs (a kind=task stream starts above seq 1, where
  the filtered-out rows sit), strictly ascending.
- **Live tail, filtered:** with an `experience.observation`
  filter, a checkpoint journaled immediately before the
  observation is walked over (never yielded, cursor advances
  past it) and the observation arrives with its true seq.
- **Fail closed:** nine malformed-filter shapes raise at
  parse level; five HTTP shapes (`collection=bogus`,
  `task=banana`, blank `kind=`, blank `collection=`,
  `task=-3`) are 400s naming the filter problem.
- **Consumer:** collect + resume round-trips against a live
  server (from the last id: nothing; from the one before:
  exactly the tail); verification passes on an honest stream
  and itemises both a forged row body and a dropped middle
  event; the CLI runs against a live server with `--verify`
  (exit 0) and refuses a bogus filter (exit 2).

## 5. Test results

- Baseline on main `0c1e894` (driver-verified at the SI-47
  review): **576 passed, 2 skipped**.
- New file alone (`tests/test_gaia_sse_filter.py`):
  **26 passed** — filter exactness vs the raw collections
  (collection / event / kind / task-chain), OR-within /
  AND-across dimensions, parse fail-closed rules, filtered
  resume (read + task + HTTP header), filtered live tail,
  unfiltered byte-identity regression, journals-nothing
  under filtering (module + HTTP), HTTP surface (exact
  replay, task + header resume, five 400 shapes, SI-47
  banner regression, filter label), and the consumer
  (collect/resume, journal verification incl. forgery and
  hole detection, wire parsing, CLI with `--verify`).
  **No existing test file was modified.**
- Regression spot-check before the full run
  (`tests/test_gaia_sse.py` + `tests/test_gaia_b5.py`, the
  two suites whose modules this slice extends):
  **46 passed**.
- Full suite on the PR head: **602 passed, 2 skipped**
  in 1794.18 s (576 baseline + 26 new; no existing test
  modified) — the builder's own SOLO run, workspace venv,
  orphaned `nedbd-v2` processes checked with `pgrep nedbd`
  and none present before the run. No failures this run —
  the known rotating stale-daemon flake in pre-existing
  real-NEDB-backend tests (disclosed in SI-46 §5) did not
  appear; had a single such failure appeared, the SI-46
  adjudication rule would apply (reproduce on pristine main
  in the same environment before attributing it to this
  diff — this slice's diff touches no journal, bundle, seed,
  or loop code).

## 6. Explicitly NOT built (do not read this slice as claiming them)

- **No exposure through the Hearth gateway** (proposal H-2),
  no gateway route, and no Hearth code of any kind. The
  Hearth fork is Mark's step; the hearth repo was not
  touched. The Hearth path remains UNMEASURED.
- **No push, no websocket, no notification channel** — the
  filtered live tail is the SI-47 poll-the-journal tail at
  0.25 s with a projection applied, labelled as polling in
  the module, the doc, the service surface, and every
  heartbeat (filtered heartbeats name SI-48).
- **No new event store, no subscription registry, no new
  journal collection, no journal schema change** (§4.6): a
  subscription is a query string on a GET, evaluated against
  the journal at read time and forgotten when the connection
  closes. Nothing about a filter is persisted anywhere.
- **No authentication or authorization added**: filters are
  not access control (§2.5); the stream inherits B-5's
  no-auth posture and labels verbatim.
- **No production client library**: the consumer is a
  reference implementation of the wire contract — no
  retries, no reconnect loop, no TLS/auth story.
- **No new execution scope and no new proposer**: live
  execution remains miniature-scoped (B-2/B-3); demo bundles
  remain shadow. Filtering a stream executes nothing.
- **A-2/A-3/A-4 and any real-base agency** — unapproved, not
  built. Real-base agency claims stay honestly scoped until
  measured.
