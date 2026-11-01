# SI-71 — Gaia L-5: Seam Label Reconciliation and Fork-Ingest Contract (SI-69 §4.7)

## The honest headline

**The fork half remains UNBUILT after L-5. L-5 builds no fork
code and claims no fork behaviour.** This slice is SI-side
only, CPU-only, and does no live work: it reconciles stale
S-5 wording with new label constants (§2) and writes the
SI-side contract spec for the fork ingest half of S-6 (§3)
as a design spec the fork lane can build against in
`aiassistsecure/gaia`. Nothing in this slice states or
implies that a fork ingest endpoint exists.

> live model-authored intent NEVER YET OCCURRED
> real Hearth/gaia path UNMEASURED

## §0 honesty table

**No §0 row changes.** SI-69 §0's table stands exactly as
merged, including its two seam rows: "S-5 outcome return
(seam label) — UNBUILT as labelled in seam code" remains
accurate as a statement about the historical label strings
(they are untouched, §2), and "Fork half of S-6 event
ingest — UNBUILT" remains accurate without qualification.
This slice promoted no row, weakened no label, and measured
nothing live.

## 1. Scope, exactly as SI-69 §4.7 defines it

Two SI-side parts:

- **(a) Label reconciliation.** New label constants stating
  the S-5 SI-side truth (§2), without rewriting any
  already-journaled record and without changing the meaning
  of any historical label string.
- **(b) Fork-ingest contract spec.** The SI-side contract
  for the fork half of S-6 (§3), derived only from merged
  SI-side evidence: the S-6 forwarder
  (`si/gaia/seam_events.py`, SI-60), its doc
  (`docs/S-6-si-events-forwarder.md`), and the S-7 doubles
  it was exercised against (SI-62).

Out of scope, by the RFC and by this slice: any code in
`aiassistsecure/gaia`, any live endpoint, any measurement
of F-E-INGEST or F-E-SSE, and A-2 / A-3 / A-4 (UNAPPROVED).

## 2. Part (a) — S-5 label reconciliation

### 2.1 The stale wording (SI-69 §3.1 evidence)

| Historical constant | Stale wording |
| --- | --- |
| `si.gate.TOOL_SEAM_PENDING_EXECUTION` | ends "and S-5 outcome return is UNBUILT" |
| `si.gaia.seam.REGISTRY_GAP_LABEL` | "S-5 outcome return UNBUILT" |
| `si.gaia.seam_submit.SUBMITTED_STATE` | `"submitted/awaiting outcome (S-5 unbuilt)"` |
| `si.gaia.seam_submit.WORK_SUBMISSION_LABEL` | "S-5 outcome return is UNBUILT" |
| `si.gaia.seam_leash.LEASH_CROSSING_LABEL` (+ module docstring) | "S-5 outcome return is UNBUILT" / "remains UNBUILT" |

Each was true when written. `SUBMITTED_STATE` and the gate
label are additionally load-bearing history: the state
string is journaled verbatim in `gaia_work` rows, and the
gate label's exact text is the eligibility key S-3
submission checks against the journaled gate report.

### 2.2 What S-5 actually built (the reconciled truth)

Per SI-69 §3.1's scoped reading and
`docs/S-5-outcome-return.md` §0/§2: the SI-side outcome
poller and reconciler are **BUILT** (S-5/SI-59,
`si/gaia/seam_outcome.py`), tested against labelled
replay/stub transports and a loopback double. Live fork
interop is **UNMEASURED**: no live fork endpoint was
exercised. And no outcome returns **to** the fork — S-5 is
the fork-to-SI return leg; nothing sends an outcome back
across the seam in the SI-to-fork direction. A seam-crossing
ADMIT's full live round trip has never been observed.

### 2.3 What this slice adds

`si/gaia/seam_reconcile.py` (new) carries:

- `HISTORICAL_S5_LABEL_SOURCES` — the five constants in
  §2.1, named, never edited.
- `S5_RECONCILIATION_LABEL` — the §2.2 truth in one label:
  poller and reconciler BUILT (S-5/SI-59), tested against
  labelled replay/stub transports and a loopback double;
  live fork interop UNMEASURED; no outcome returns to the
  fork; the historical strings are journal-time/slice-time
  truth, not rewritten, and no already-journaled record
  changes meaning.
- `FORK_INGEST_CONTRACT_LABEL` — the §3 label: the fork
  half remains UNBUILT, F-E-INGEST and F-E-SSE stay
  UNMEASURED until a live fork endpoint exists and is
  exercised.

The service description (`si/gaia/service.py`) gains one
key, `seam_label_reconciliation`, carrying
`S5_RECONCILIATION_LABEL` beside the existing seam keys —
so the reconciled reading is visible on the same surface
that already carries the historical labels. This is the
pattern S-5 itself set (`docs/S-5-outcome-return.md` §2:
the S-5 truth lives in its doc, in `OUTCOME_RETURN_LABEL`,
and in the service description; the historical strings are
untouched).

### 2.4 Test justification (SI-69 §4.7 acceptance)

No existing test is modified. Existing tests pinning the
historical labels (`tests/test_gaia_s2b_hearth_registry.py`,
`tests/test_gaia_s3_work_submission.py`, and the S-4/S-5/S-6
suites) pass unmodified — the historical constants are
byte-identical in this slice. The one new test file,
`tests/test_gaia_l5_seam_reconcile.py`, pins **only the new
constants** (`S5_RECONCILIATION_LABEL`,
`FORK_INGEST_CONTRACT_LABEL`,
`HISTORICAL_S5_LABEL_SOURCES`) and the new service
description key carrying a new constant. That is the only
test change, and it is justified exactly as §4.7 allows:
pinning a new constant, never editing history's meaning.

## 3. Part (b) — Fork-ingest contract spec (for the fork lane)

**Status of this section: design spec only.** It specifies
the receiving half the fork lane is to build in
`aiassistsecure/gaia`. It is written entirely from the
SI-side sender's merged behaviour, so a fork implementation
that satisfies it interoperates with the shipped forwarder
(`forward_journal_events`) without any SI change. It does
not describe an existing endpoint.

### 3.1 Endpoint and transport posture

- Route: `POST /gaia/v1/events/ingest` (the path is already
  named SI-side as `si.gaia.seam_events.INGEST_PATH`). The
  host is always operator-supplied; SI invents and defaults
  no host and appends no path to any base.
- One POST per event, in ascending SI journal seq order.
  There is no batch form in the exercised contract.
- Posture inherited from G-9 §14 and the S-6 slice: loopback
  / operator-configured endpoints; the SI sender sends no
  credential. Anything beyond that posture is fork-lane
  design, not specified here.

### 3.2 Request shape

The body of each POST is exactly one
`StreamEvent::from_si` payload, as produced by
`si.gaia.seam_events.build_stream_event`:

| Key | Value |
| --- | --- |
| `source` | `"si"` — always, exactly |
| `seq` | the SI journal seq (integer) — never a fork spine seq; the two sequence spaces are never fused |
| `kind` | the SI journal event name |
| `collection` | the journal collection the row came from |
| `row` | the full journal row, verbatim |
| `row_kind` | the row's own `kind`, or null |
| `facts` | `{"collection": ..., "row": ...}`, plus `queue_state` only when the journal event carries one |
| `model` | present only when derivable from the row without invention |
| `task` | present only when derivable from the row without invention |

There is **no `hash` key** — hash is hearth-only (G-9 §4.5).
A fork implementation must not require keys beyond this
shape, and must preserve `row` verbatim.

### 3.3 Response and dedup contract (200 / 409)

This contract is already exercised — against the S-7
double, not against a fork: `si.gaia.seam_measure.ingest_double`
and the loopback double in
`tests/test_gaia_s6_event_forwarder.py` both implement it
through the real `HttpTransport`.

- **HTTP 200 — accepted.** The event is accepted exactly
  once into the fork's `source=si` buffer (G-9 §4.5's
  `SiEventBuffer::push`). The exercised double's body is
  `{"status": "ok"}`; the SI sender advances its caller-held
  cursor only on a 200 and does not depend on the body —
  the fork lane should keep the body to that shape or
  simpler, and must not use any other 2xx as the ack, since
  the sender treats non-200/non-409 answers as failure.
- **HTTP 409 — duplicate / refusal.** A `seq` the buffer
  has already accepted is refused with 409 (the exercised
  doubles' body: `{"error": "duplicate seq"}`). SI treats
  any 409 as a loud protocol failure
  (`SeamProtocolError`): it is not retried, not silently
  deduped, nothing after it is pushed, and the cursor does
  not advance past it. The fork lane must therefore use 409
  only for a genuine duplicate/out-of-order disagreement —
  never as a transient error.
- **Any other status** (5xx included) or a transport
  failure: the SI pass stops as `forward-failed`, the
  cursor stays at the last acked seq, and a retry from that
  cursor is free because history is the journal.

### 3.4 `source=si` SSE visibility requirement

Acceptance (a 200) must make the event visible on the
fork's SSE stream as a `source=si` event, preserving
`source`, `seq` (the SI journal seq, unrenumbered),
`kind`, `collection`, and the verbatim `row`, in the order
accepted within the `source=si` stream. A subscriber
filtering on `source=si` must be able to observe every
accepted event; an event that is acked but never visible
violates this contract. (SI's own `events_client` consuming
that stream is S-6 acceptance on the fork side; it is not
proven by anything in this repo.)

### 3.5 Deferred quantities — still UNMEASURED

| ID | Quantity | State |
| --- | --- | --- |
| F-E-INGEST | fork ingest service time per event (`POST /gaia/v1/events/ingest`) | UNMEASURED |
| F-E-SSE | fork ingest → `source=si` SSE visibility lag; merged-stream timing | UNMEASURED |

Both stay **UNMEASURED until a live fork endpoint exists
and is exercised**. The S-7 numbers for the SI-side
forwarder pass (Q-E4/Q-E5, against the double) measure the
sender only; they are not ingest or visibility numbers and
must never be cited as such.

## 4. Honesty gate

Any wording that lets a reader believe the fork ingest
endpoint exists fails this slice. There is none: §3 is
labelled a design spec throughout, the fork half is stated
UNBUILT in the headline, in §0, in
`FORK_INGEST_CONTRACT_LABEL`, and in §3.5. Terminology:
machine learning infrastructure throughout; the verbatim
labels stand — live model-authored intent NEVER YET
OCCURRED, real Hearth/gaia path UNMEASURED.

## 5. Tests

New: `tests/test_gaia_l5_seam_reconcile.py` (4 tests,
constants + service-description key only, per §2.4).
Affected subsets run by the builder are recorded in the
build evidence; the full suite is the reviewer's run.
