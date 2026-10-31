# SI-58 — SSE live-tail failure: diagnosis and fix

## Symptom

`tests/test_gaia_sse_follower.py::test_real_service_filtered_follow_live_tail`
(SI-49) failed identically on pristine main and on unrelated branches,
across at least three consecutive dev-loop driver runs:

```
FollowError: following subscription exceeded max_duration 30.0s after
10 consecutive failure(s); last delivered seq is 0
```
caused client-side by `http.client.RemoteDisconnected`, ~31 s per run.
The rest of the suite was green.

## Root cause (evidence, not conjecture)

The server's own traceback, printed by the HTTP handler thread on every
attempt, was:

```
File "si/gaia/service.py", in _handle_events
    manifest = load_bundle(state.bundle_dir)
si.bundle.TamperError: manifest journal anchor does not match journal
tip (manifest seq=17 tip seq=19)
```

Chain of causation:

1. The test journals two rows directly (`checkpoint` +
   `record_experience`, via a raw `Journal`) after starting the
   follower thread. Raw journal appends do not re-anchor the manifest —
   re-anchoring happens at task boundaries in the task machinery.
   The journal tip (19) is then legitimately past the manifest
   anchor (17).
2. `_handle_events` called the full `load_bundle()` on **every**
   connection. `load_bundle` verifies, among everything else, that the
   manifest anchor equals the journal tip — a demand that cannot hold
   for a live journal in exactly the state the SSE tail exists to
   follow (writers journal first, re-anchor later; the executor states
   the same principle for its mid-cycle rule: never `load_bundle`
   mid-cycle, the tip "has legitimately moved past the manifest
   anchor … and a re-load would fail closed for the wrong reason").
3. `TamperError` was unhandled in the handler, so the socket closed
   with **no HTTP response at all**. The client correctly classified
   `RemoteDisconnected` as a transport failure and retried — against a
   permanent condition — until `max_duration` expired.
4. Whether the first connection raced ahead of the writes decided the
   outcome, which is why the bug hid: the SI-47 HTTP live-tail test
   performs the identical raw writes but connects **first** (and
   consumes the replay before writing), so its established stream —
   which polls the backend directly and never re-loads the bundle —
   passed all along. The SI-49 filtered test has no connect-before-
   write handshake, so in practice its connections landed after the
   writes and failed deterministically.

This is a production defect in the events handler, not a test defect:
any real SSE client connecting (or reconnecting, as the SI-49
follower does by design) while a task is mid-flight — journal
appended, manifest not yet re-anchored — hit the same dropped
connection. The proxy/environment theories were ruled out by the
server-side traceback: the request reached the handler and died
inside it.

## Fix (`si/gaia/service.py`, `_handle_events` only)

- The handler no longer calls `load_bundle` per connection.
- The bundle remains fully verified **once at startup**
  (`ServiceState`: a tampered bundle is never served) — that
  guarantee is unchanged.
- Per connection, the handler opens the journal directly
  (`open_journal` reads only the backend name from `manifest.json`)
  and verifies the journal **hash chain** before streaming; a broken
  chain — or an unopenable journal — fails closed with a real
  HTTP 500 JSON error instead of a silently dropped socket.
- Replay / live-tail semantics are untouched: the stream still polls
  the backend per poll and delivers rows journaled by any writer,
  anchored or not — which is what SI-47 designed and what the
  established-stream tests already relied on.
- Other endpoints (`/tasks/<seq>`, queue view, `/health`) keep their
  existing `load_bundle` snapshot semantics; they are read after
  re-anchored task boundaries and were not part of this failure.

## Tests

- The SI-49 target test passes (previously ~31.7 s to failure;
  ~1 s to pass), on repeated consecutive runs.
- New regression test
  `test_real_service_events_connect_after_unanchored_write` pins the
  exact defective scenario: it journals without re-anchoring, asserts
  the precondition (`load_bundle` indeed raises `TamperError` for
  that state), then connects and requires the filtered replay to
  deliver the observation row. No existing assertion was weakened,
  skipped, or removed.
- Affected subsets (SSE follower + SI-47 SSE + SI-48 filter + B-5
  service, plus Q-3 / seam / S-3 / S-4 service users) pass.

## Honest labels

- Nothing about the stream's nature changed: still poll-the-journal,
  still no push/websocket, still no auth story (B-5 labels stand).
- Per-connection chain verification covers the journal as it stands
  at connect time — the same verification point the old code had;
  rows appended mid-stream by a writer still arrive via the
  `Journal._append` chain-maintaining path, as before.
- Live interop and performance remain UNMEASURED, as before.
