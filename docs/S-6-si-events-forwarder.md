# S-6 — Gaia seam SI journal events into fork SSE `source=si`, SI side (SI-60)

**Status:** implemented on `s-6/si-events-forwarder` against main
`b64ffc9` (SI-59 / S-5 merged as PR #65). This is the SI half of
G-9 §4.5 / §15's **S-6**: SI journal events are forwarded, in
order, toward the fork's SSE `source=si` buffer. The fork half
(`POST /gaia/v1/events/ingest` feeding `SiEventBuffer::push` in
`aiassistsecure/gaia`) is **not built by this slice** — it is the
other PR of §15 S-6's pair, in the other repo. §0 says plainly
what is live and what is not.

| G-9 element | S-6 state |
| --- | --- |
| SI forwarder (§15 S-6) | **live, SI side, explicit step** — `si/gaia/seam_events.py:forward_journal_events` reads journal events after a caller-held cursor and POSTs one event per call to the operator-supplied ingest URL; no daemon, no timer, the push cadence is an UNMEASURED policy constant this slice does not set |
| Payload shape (§4.5) | **live** — `build_stream_event` produces the `StreamEvent::from_si` shape: `source` `si`, `seq` the SI journal seq, `kind` the SI event name (`event_name_for_row`, via `read_journal_events`), `collection`, the full journal row verbatim under `row`, `row_kind` (the row's own `kind` or None), and `facts` = `{"collection", "row"}` plus `queue_state` only when the journal event carries one (SI-53 convention); `model` / `task` keys appear only when derivable from the row without invention (a model string in intent args or the record; a task seq / task ref the row itself names); there is **no** `hash` key — hash is hearth-only per §4.5 |
| Two sequence spaces (§2 invariant 2) | **kept** — `seq` is the SI journal seq, never presented as a spine seq; no fused sequence is created anywhere in this slice |
| Cursor discipline (SI-49 / §7 principle 7) | **live** — the cursor advances **only after HTTP 200** for that event; it is caller-held and returned, never stored (no cursor store is invented, the S-5 reconciler pattern); a retry from the returned cursor is free because history is the journal — events journaled while the fork is unreachable are pushed later, from the cursor |
| Duplicate push (§15 S-6 acceptance 2) | **loud protocol failure** — an HTTP 409 (the fork buffer's duplicate / out-of-order refusal) raises `SeamProtocolError` carrying `(seq, http_status, response, cursor-so-far)`: it is NOT retried, NOT silently deduped, nothing after it is pushed, and the cursor does not advance past it |
| Other failures | **forward-failed, cursor unadvanced** — transport exceptions and non-200 / non-409 answers (5xx included) stop the pass and return `{"status": "forward-failed", "cursor": <last acked>, "pushed": [...], ...}`; the failed event is not acked |
| No endpoint | **refused** — with no transport and no `ingest_url` the call returns `refused` and pushes nothing; there is no default fork endpoint (the documented fork route is named as the constant `INGEST_PATH = "/gaia/v1/events/ingest"`, but the URL is operator-supplied whole — this module appends no path to any base, the S-5 `work_url` pattern) |
| Journaling | **none** — the forwarder journals NOTHING in SI's journal: the journal is the source and pushing is a read. The read path opens the journal directly and does not demand manifest-anchor == journal-tip (the SI-58 lesson: a live journal's tip advances past its anchor between re-anchors, and those rows are exactly what a forwarder exists to push) |
| Service surface | **wrapper only** — `si.gaia.service:forward_events` and a `seam_event_forward` description key carrying `EVENT_FORWARD_LABEL`; no new HTTP route (forwarding is outbound and operator-invoked, like S-3/S-5) |
| Fork ingest half (`POST /gaia/v1/events/ingest`) | **UNBUILT in this repo / slice** — it is fork-side work in `aiassistsecure/gaia`; nothing here claims the route exists |
| SI `events_client` consuming the fork stream (§15 S-6 acceptance 1) | **not proven here** — that acceptance runs against the fork's stream once the fork half exists; it is fork-side acceptance, not SI-side evidence |
| Live fork interop | **UNMEASURED** — no live fork endpoint was exercised; evidence is the labelled replay/stub transport and a loopback HTTP test double implementing the documented ingest contract (200 per event, 409 on duplicate seq) through the real `HttpTransport`, plus an env-gated live test (`SI_S6_INGEST_URL`) that skips when unset |
| Performance | **UNMEASURED** |

---

## 1. What was built

### 1.1 The forwarder — `si/gaia/seam_events.py` (new)

- **`build_stream_event(event)`** — as the table states. Its
  input is one dict from `read_journal_events`; its output is
  the §4.5 payload and nothing else (no envelope invented
  around it).
- **`forward_journal_events(bundle_dir, *, ingest_url=None,
  transport=None, cursor=0, limit=None, timeout_s=30.0)`** —
  reads events via `read_journal_events` (open journal,
  `after_seq=cursor`) in ascending seq order, POSTs one per
  call, and returns
  `{"status": "forwarded", "cursor", "pushed", "count",
  "transport", "label"}`. `limit` bounds one pass's window;
  the returned cursor is exactly where the next pass resumes.
- **`SeamProtocolError`** — the 409 path, with attributes
  `(seq, http_status, response, cursor, pushed)` pinned by
  tests.
- **`pending_event_seqs(bundle_dir, cursor=0)`** — a pure-read
  helper listing the seqs a forward from `cursor` would push.
- **`EVENT_FORWARD_LABEL` / `INGEST_PATH`** — the honesty label
  carried on every result, and the documented fork route named
  as a path constant only (no host invented or defaulted).

### 1.2 Service wrapper — `si/gaia/service.py`

`forward_events(bundle_dir, **kwargs)` is a thin wrapper over
`forward_journal_events`, and the service description gains a
`seam_event_forward` key carrying `EVENT_FORWARD_LABEL` — the
S-5 wrapper pattern exactly. No new HTTP route.

### 1.3 Tests — `tests/test_gaia_s6_event_forwarder.py` (new)

11 tests (10 passed, 1 skipped — the env-gated live test):
`build_stream_event` shape over every seeded journal row
(source / exact seq / kind / collection / verbatim row / no
`hash` / `row_kind` / `queue_state` in facts only when present)
plus synthetic derivable-vs-omitted `model`/`task`; in-order
push with exact seq fidelity through `ReplayTransport`
(payloads equal to `build_stream_event`, URL used whole, the
journal unchanged afterwards — the forwarder journals
nothing); `limit` windowing; cursor-advances-only-after-200
(a scripted transport exception mid-stream leaves the cursor
at the last acked seq, and a second forward from it — after
more rows are journaled via `submit_proposal` — pushes exactly
the remainder plus the new rows); HTTP 500 as
`forward-failed` with an unadvanced cursor and a free retry;
HTTP 409 raising `SeamProtocolError` with no further pushes
(transport call count pinned); the no-endpoint refusal; the
loopback HTTP ingest double through the real `HttpTransport`
(full fidelity, then a duplicate re-push raising the protocol
failure on the first event); the service wrapper and its
description key; and the env-gated live test.

## 2. Honest gaps and what remains unbuilt

- **The fork ingest half is UNBUILT in this repo/slice.**
  `POST /gaia/v1/events/ingest` in `aiassistsecure/gaia`,
  feeding `SiEventBuffer::push`, is the other half of §15 S-6.
  Until it lands, this forwarder's pushes have no real fork
  endpoint to land on — stated in `EVENT_FORWARD_LABEL`, in
  this doc, and in the PR body.
- **Fork-stream consumption is not proven here.** §15 S-6
  acceptance 1 (the fork replaying `source=si` with exact seq
  fidelity, and SI's own `events_client` follower consuming
  the fork stream through a drop-and-reconnect with zero
  missed / zero duplicated events) requires the fork half; it
  is fork-side acceptance, not evidence this slice produces.
- **No push automation.** Events are forwarded when invoked;
  there is no daemon, timer, or follow loop in this slice —
  a caller driving the forwarder in a loop supplies its own
  cadence (UNMEASURED).
- **UNBUILT after S-6** (per G-9 §15, cumulative): escalation
  resolution, auth, non-loopback anything, work-store
  persistence (G-4b), combined multi-source subscriptions
  (G-4b), and **seam measurements** — every latency and
  throughput number for the seam is UNMEASURED; the proposed
  S-7 measurement slice is proposed, not designed, in G-9.
