# SI-49 — Gaia SSE Resilient Following Consumer
(SI-48 §6's deferred layer: reconnect + resume for the reference consumer)

> **Status: BUILT, in review (PR opened by the SI-49 builder;
> merging is the dev-loop driver's, on a later run).** This
> slice builds the reconnect/resume layer SI-48 explicitly
> left unbuilt — its §6: *"No production client library: the
> consumer is a reference implementation of the wire contract
> — no retries, no reconnect loop, no TLS/auth story."*
> SI-49 adds the reconnect loop and the resume discipline to
> that same reference consumer (`si/gaia/events_client.py`,
> CLI `si gaia-events --follow-resilient`): one logical
> subscription maintained across connection drops,
> reconnecting with `Last-Event-ID` set to the last seq
> actually received, under bounded exponential backoff,
> deduped by journal seq, fail-closed on malformed resume.
>
> **Origin.** SI-47 §2.5 pinned the semantics this slice
> implements client-side: *backpressure is the socket's;
> reconnect = resume from any point via Last-Event-ID /
> ?after=.* SI-48 built the filtered subscription and the
> reference consumer and named the missing piece. SI-49 is
> that piece, per Mark's standing 2026-10-29 direction to
> develop the Gaia agentic loop slice by slice ("expand and
> expose and let's endure").
>
> **The live-scope boundary, stated exactly.** LIVE (real
> code, exercised this slice): a following mode on the
> SI-48 reference consumer — module API
> (`follow_events` / `collect_followed` / `FollowConfig` /
> `FollowError`) and CLI flag (`si gaia-events
> --follow-resilient` with `--max-events`, `--max-attempts`,
> `--backoff-base`, `--backoff-cap`, `--max-duration`) — that
> keeps one filtered or unfiltered subscription alive across
> connection drops: transport failures reconnect with
> `Last-Event-ID` == the last seq actually delivered, under
> bounded exponential backoff (configurable base and cap,
> max consecutive attempts, max total duration); a duplicate
> or out-of-order seq from the server is a loud
> `FollowError`, never silent dedupe; protocol failures
> (malformed resume replay, filter violations caught by the
> SI-48 client-side re-check, malformed SSE, an HTTP 4xx
> refusal of a resume) fail closed and are never retried;
> and the followed window verifies against the journal
> through the SI-48 `verify_events_against_journal` — the
> one verification path, reused, not forked. **It remains a
> reference follower, honestly labelled**: no TLS/auth story
> is claimed, no production-client claims are made, no
> Hearth gateway exists here (H-2 is Mark's step, NOT
> built), the server tail it follows is still the SI-47
> poll-the-journal tail (no push, no websocket), and nothing
> is journaled, stored, or registered by following.
>
> **Terminology:** machine learning infrastructure
> throughout. CPU only; stdlib only; no GPU, no cloud, no
> spend, no downloads. **No new journal collection, no new
> event store, no subscription registry**: a followed
> subscription is a client-side loop over the same GET
> /events surface SI-47/SI-48 built; the server is untouched
> by this slice except that it answers the same requests
> more than once.

---

## 1. What was built

### 1.1 The follower (`si/gaia/events_client.py`)

- `follow_events(base_url, after_seq=0, event_filter=None,
  config=None, should_stop=None, sleep_fn=None, now_fn=None,
  fetch_factory=None, timeout=None)` — a generator yielding
  the same `{"id", "event", "collection", "row"}` dicts as
  SI-48's `fetch_events`. Each (re)connection is a
  `fetch_events(..., follow=True)` whose resume point is the
  last seq this generator actually delivered — the server
  replays exactly the rows after it (projected through the
  filter), so nothing already delivered is even requested
  again. As an independent second guard, every received id
  is checked against the last delivered id: `id <= last` is
  a `FollowError` on the spot (§2.1).
- `FollowConfig` — the bounds, validated fail-closed at
  construction (non-positive base/duration, a cap below the
  base, `max_attempts < 1`, `max_events < 1` are each
  `ValueError`): `backoff_base` (default 0.5 s),
  `backoff_cap` (default 30 s), `max_attempts` (max
  *consecutive* failed reconnects, default unbounded),
  `max_duration` (max total seconds from the first attempt,
  default unbounded), `max_events` (clean stop after N
  delivered events, default unbounded). The delay before
  the n-th consecutive reconnect is
  `min(cap, base * 2 ** (n-1))` — deterministic, no jitter,
  so the bound stated is the bound kept (asserted exactly
  by test). The consecutive-failure counter resets once a
  connection delivers at least one event: the subscription
  demonstrably made progress, so a later drop starts from
  the base delay again (asserted by test).
- `collect_followed(...)` — the bounded, collected form:
  runs `follow_events` to its bound and returns the events,
  the last seq, and — when a journal `verify_source` is
  given — the SI-48 verification report over the followed
  window, reconnects included. Verification failure is
  reported, not raised, exactly as SI-48 reports it.
- `FOLLOWER_LABEL` — the honesty label, printed by the CLI
  in resilient mode (§3).

### 1.2 Which failures reconnect, and which fail closed

| Failure | Class | Behaviour |
|---|---|---|
| Connection refused / reset / dropped, clean server close of a following stream, HTTP 5xx transport errors | transport | reconnect after the bounded backoff; consecutive-failure counter +1 |
| Read timeout on an idle stream (the follower's own wake interval, §1.3) | wake | reconnect immediately, budget untouched |
| Duplicate or out-of-order seq (`id <= last delivered`) | protocol | `FollowError` at once, never retried |
| Server replays the resume seq itself (malformed resume) | protocol | `FollowError` at once, never retried |
| Stream violates the requested filter (SI-48 client-side re-check `ValueError`), malformed SSE block | protocol | `FollowError` at once, never retried |
| HTTP 4xx on (re)connect (e.g. the server's 400 refusal of a malformed resume point) | protocol | `FollowError` at once, never retried |
| `max_attempts` consecutive transport failures, or `max_duration` exceeded | bound | `FollowError` naming the bound and the last delivered seq — the caller resumes a fresh follower from exactly that seq |

### 1.3 Stop discipline and the wake (a design point, stated)

A following stream is kept warm by server heartbeats every
poll interval, so a socket read timeout alone can never
wake an idle follower — the heartbeats keep resetting it.
Two mechanisms together make `should_stop` prompt:

1. `fetch_events` gains an optional `should_stop` parameter
   (SI-49, additive; SI-48 behaviour without it is
   unchanged): it is checked at every SSE block, heartbeat
   comment blocks included, so an idle following stream
   honours the stop within one heartbeat. `follow_events`
   hands its own `should_stop` to the transport when the
   transport accepts one (the real `fetch_events` does;
   injected test transports keep the SI-48 signature and are
   called without it).
2. When the caller gives no per-connection `timeout`, the
   follower installs `DEFAULT_FOLLOW_WAKE_SECONDS` (2.0 s)
   as a wake interval for transports/peers without
   heartbeats: a read timeout there is a *wake*, not a
   failure — reconnect from the last delivered seq, budget
   untouched, after re-checking `should_stop` /
   `max_duration`. An event block interrupted mid-read by a
   wake was never delivered, so the resume replays it whole
   (and the dedupe guard would catch any partial delivery
   loudly).

### 1.4 The CLI (`si gaia-events --follow-resilient`)

In the established `gaia-events` pattern, additive flags on
the same subcommand: `--follow-resilient` selects the
follower; `--max-events`, `--max-attempts`,
`--backoff-base`, `--backoff-cap`, `--max-duration` carry
the `FollowConfig` bounds. The mode prints both the SI-48
consumer label and the SI-49 follower label, then the
filter description and the bounds, then one line per event
as before, then the resume hint. Combinations are refused
fail-closed with exit 2: `--follow` with
`--follow-resilient` (two modes of one subscription), and
`--verify` with `--follow-resilient` *without*
`--max-events` (verification needs a completed window; with
`--max-events` the combination runs and verifies the
followed window through the SI-48 path). A follower that
fails closed exits 1 with the `FollowError` on stderr,
naming the last delivered seq.

### 1.5 What did not change

The server (`si/gaia/service.py`), the spine
(`si/gaia/events.py`), the journal, and every SI-48 surface
are untouched: `fetch_events` gains only the optional
`should_stop` parameter (§1.3), and the SI-48 tests pass
unmodified against it. No existing test file was modified
by this slice.

## 2. Pinned design decisions (stated, never silent)

1. **Dedupe is a guard, not a mechanism.** The primary
   no-duplicates mechanism is the resume point itself: the
   follower only ever asks for rows after the last seq it
   delivered. The `id <= last` check exists for the server
   that gets that wrong — and there the honest behaviour is
   to stop loudly, not to quietly swallow the duplicate and
   present a stream whose provenance no longer matches its
   claim. Silently deduping would hide exactly the class of
   bug (wrong resume, renumbering) the spine's
   id == journal seq property exists to expose.
2. **Backoff is deterministic.** No jitter: the schedule
   `min(cap, base * 2 ** (n-1))` is exactly what the config
   states and exactly what the tests assert. A reference
   follower's bound should be auditable arithmetic, not a
   distribution.
3. **Progress resets the failure counter.** "Consecutive"
   is the operative word in `max_attempts`: a subscription
   that reconnected and delivered is healthy, and counting
   an old, recovered drop against a later one would kill
   long-lived followers for history they already survived.
4. **A clean server close is a drop.** The SI-48 following
   transport ending without an error still means the
   subscription's connection is gone; the follower treats
   it under the same bounds rather than ending a
   subscription its caller asked it to maintain.
5. **Verification is reused, never forked.** The followed
   window is verified by SI-48's
   `verify_events_against_journal` — row-for-row against
   the journal, plus the exact-projection check under a
   filter — because a second, follower-specific verifier
   could drift from the consumer's and quietly weaken the
   claim. One path, one exactness.
6. **No server change, no registry.** Resilience is a
   client property here by design: the server's contract
   (resume from any seq, ids are journal seqs) already
   contains everything a follower needs, so this slice
   adds no server state, no session, and nothing to
   persist — a follower that dies loses nothing but its
   in-memory last seq, which its caller holds from the
   events it delivered.

## 3. Hearth-side assumptions

None. This slice imports no Hearth code, contacts no
Hearth node, and depends on no H-1/H-2/H-3 proposal. The
Hearth path remains **UNMEASURED** (B-4's label,
unchanged). The Hearth fork ("Hearth becomes Gaia")
remains Mark's step; no fork was created and the hearth
repo was not touched.

## 4. Evidence (the builder's own runs on the PR head)

- **Real drop-and-reconnect (the headline test).** The
  follower connects to the actual B-5 service through a
  killable raw-TCP proxy; after the initial replay is
  delivered, the proxy is killed mid-stream (every
  in-flight socket severed — a network drop, the service
  itself never restarts); a whole second B-5 task is then
  journaled through the service machinery while the
  follower is down; the proxy restarts on the same port;
  the follower reconnects, resumes from its last delivered
  seq, and delivers exactly the missed seqs: the full
  received id sequence equals the journal's event sequence
  end to end, strictly ascending, no duplicates.
- **Resume exactness over a fake transport:** the
  reconnect's `after_seq` is exactly the last delivered
  seq; a clean close reconnects the same way; the backoff
  counter resets after progress (two drops separated by a
  delivered event both sleep the base delay).
- **Backoff bounds, exactly:** schedule
  `[0.5, 1.0, 1.2, 1.2, …]` for base 0.5 / cap 1.2;
  `max_attempts=3` against a refused port makes exactly 3
  connection attempts and 2 sleeps, then a `FollowError`
  naming the bound and the last delivered seq;
  `max_duration` fires as a `FollowError` under an injected
  clock.
- **Fail closed, no retries:** duplicate seq, out-of-order
  seq, a replayed resume seq, a filter violation raised by
  the SI-48 re-check, and an HTTP 400 on resume each raise
  `FollowError` with the transport called exactly once.
- **Wake discipline:** a read timeout reconnects with no
  sleep and no attempt budget consumed, and `should_stop`
  is honoured at a wake with zero events delivered.
- **Verification reuse:** `collect_followed` over a full
  journal window returns the SI-48 report
  `{"ok": True, "checked": N, "problems": []}`, and the
  same path itemises a forged row body in a followed
  window.
- **Filtered live follow:** against the real service, an
  `experience.observation` filtered follower walks over a
  checkpoint journaled immediately before the observation
  and delivers the observation with its true seq.
- **CLI:** `--follow-resilient --max-events 3` against the
  real service exits 0 printing the follower label and the
  resume hint; `--verify` without `--max-events` exits 2;
  `--verify` with `--max-events` exits 0 with
  `verify: ok=True`; an unreachable service with
  `--max-attempts 2` exits 1 naming the bound.

## 5. Test results

- Baseline on main `787b9bf` (driver-verified at the SI-48
  merge): **602 passed, 2 skipped**.
- New file alone (`tests/test_gaia_sse_follower.py`):
  **19 passed** — backoff schedule + config validation,
  reconnect/resume exactness, clean-close reconnect,
  backoff reset after progress, max-attempts and
  max-duration bounds, wake discipline, the five
  fail-closed protocol shapes, verification reuse, the
  real-service drop-and-reconnect headline test, the
  real-service filtered live follow, and the CLI surface
  (bounded resilient follow, verify combinations,
  unreachable fail-closed). **No existing test file was
  modified.**
- Regression spot-check before the full run
  (`tests/test_gaia_sse_filter.py` + `tests/test_gaia_sse.py`
  + `tests/test_gaia_sse_follower.py`, the suites whose
  modules this slice extends): **67 passed**.
- Full suite on the PR head: **621 passed, 2 skipped**
  in 695.59 s (602 baseline + 19 new; no existing test
  modified) — the builder's own SOLO run, workspace venv,
  orphaned `nedbd-v2` processes checked with `pgrep nedbd`
  and none present before the run. No failures — the known
  stale-daemon flake in `test_nedb_journal_tamper_fails_closed`
  did not appear; had it, the SI-46 §5 adjudication rule
  would apply.

## 6. Explicitly NOT built (do not read this slice as claiming them)

- **No production client library**: the follower is a
  reference implementation of the reconnect/resume
  contract — bounded retries of exactly the stated shape,
  no TLS, no auth story, no connection pooling, no
  multi-subscription management. Production consumers keep
  their own transport discipline beyond this reference.
- **No exposure through the Hearth gateway** (proposal
  H-2), no gateway route, and no Hearth code of any kind.
  The Hearth fork is Mark's step; the hearth repo was not
  touched. The Hearth path remains UNMEASURED.
- **No push, no websocket, no notification channel** — the
  followed tail is the SI-47 poll-the-journal tail at
  0.25 s, reconnected to as needed; reconnect changes
  nothing about how new rows reach the server stream.
- **No new event store, no subscription registry, no new
  journal collection, no journal schema change, no server
  change**: resilience here is a client-side loop over the
  existing GET /events contract; the server neither knows
  nor records that a follower reconnected.
- **No authentication or authorization added**: the
  follower inherits B-5's no-auth posture and labels
  verbatim; reconnecting does not narrow or widen who may
  read what.
- **No new execution scope and no new proposer**: live
  execution remains miniature-scoped (B-2/B-3); demo
  bundles remain shadow. Following a stream executes
  nothing.
- **A-2/A-3/A-4 and any real-base agency** — unapproved,
  not built. Real-base agency claims stay honestly scoped
  until measured.
