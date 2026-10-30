# SI-52 — Gaia Q-2: `depends_on` requirement semantics, barrier sync points, claim-time residence refusals, and the bounded read pool

**Status:** implemented on `si-52/q2-concurrency-deps` against
the SI-50 queue-drain spec (PR #55, merged at `59ed1be`) and
the SI-51 Q-1 queue core (PR #56, merged at `214a564`). This
is §16's **Q-2** slice, built exactly to that scope — no Q-3
(leash/dispositions/exposure) or Q-4 (measurement) behaviour
is smuggled in, and §0 below says plainly what is live and
what is not. Q-2 is the slice where the queue stops being
strictly serial in the one place the spec allows it —
unkeyed reads — and where `depends_on` stops being treated
as sequence order and becomes the §7.3 semantics.

| SI-50 element | Q-2 state |
| --- | --- |
| §7.3 `depends_on` requirement semantics (`outcome` vs `success`) | **live** — `outcome` satisfied by any recorded conclusion; `success` only by a completed execution |
| §7.3 dependent skipping | **live** — a queued intent whose `success` dependency concluded any other way is journaled `skipped` (`queue_skip`, with cause + the dependency's outcome sequence), never executes, and the skip cascades |
| §10.2 sync point 2 (barrier) | **live** — a `barrier: true` intent concluding during a drain flushes perception with sync reason `barrier`; later intents stay queued |
| §7.2 declared ordering keys | **live (priced in Q-1, enforced under concurrency here)** — a proposer-declared key makes even a read keyed work: it drains serially and never overlaps a same-key sibling |
| §8 bounded read concurrency | **live** — unkeyed read-class intents execute concurrently up to `queue_max_concurrent_reads` (default 4, hard cap 16); **the bound and every performance quantity remain UNMEASURED (Q-4)** |
| §8.1 serial writes/executions | **unchanged, now load-bearing** — write-class and execute-class intents are always keyed (§7.2) and drain strictly serially; concurrency is confined to the class that cannot change the world |
| §12.2 claim-time residence refusals | **live** — `queue_max_residence_seconds` (default 3600); a queued intent past the bound is refused at claim with cause `stale-admission` |
| §9 pipelining (admission overlaps execution) | **live at the admission/drain boundary, exercised under test** — batch N+1 is gated and journaled while batch N executes; the journal's sequences are the witness (§5). The task loop's own propose→admit→drain→perceive cycle is not threaded (§4) |
| §5/§8.3 serial outcome journaling under concurrency | **live** — claims journal serially in intent order before a pool batch runs; the journal serializes its own appends, so concurrent outcomes append in completion order with the hash chain intact |
| §11 leash against a live drain, §13.3 write/execute dispositions, §15.4/§15.5 queue exposure | **not live — Q-3.** The fold-level stop/resume of Q-1 is unchanged; no operator disposition machinery is added here |
| §16 Q-4 measurement | **not live — Q-4.** No latency, throughput, or contention number appears in this slice; the default bound of 4 is the spec's starting policy, not a measured optimum |
| Miniature scope / `external` | **unchanged** — an external-effective intent still ESCALATEs and is never claimed or executed |

---

## 1. What was built

### 1.1 Requirement semantics and skips — `si/gaia/drain.py`, `si/gaia/queue.py`

`Drainer.claimable` now evaluates each `depends_on` entry
against its declared requirement (§7.3):

- **`outcome`** — satisfied when the dependency's folded
  state is `executed`, `refused-at-claim`, `skipped`,
  `rejected`, or `escalated`: any *recorded conclusion*. The
  dependent runs knowing the dependency concluded, not how.
  `outcome-unknown` is deliberately **not** in this set: an
  intent whose outcome is unknown has not concluded in any
  sense a dependent may rely on, so its dependents wait for
  the Q-3 operator disposition.
- **`success`** — satisfied only by `executed` with outcome
  status `completed`. A failed execution concluded but did
  not succeed; a `timed-out` execution's partial result does
  not satisfy either (pinned here — §2).

Before claiming anything, each drain round runs
`Drainer._propagate_skips`: a queued intent with a
`success` dependency that has reached a terminal state
without succeeding is skipped via the new queue writer
`record_queue_skip` (§7.3: *"journaled with the cause and
the dependency's outcome sequence — and never executes"*).
The folded intent carries `skip_cause`,
`skip_dependency_seq`, and `skip_dependency_outcome_seq`.
Propagation repeats to a fixpoint, so a skip cascades: a
skipped intent is itself a non-success conclusion for its
own `success` dependents. A dependency naming no folded
intent never causes a skip — the dependent stays blocked,
fail-closed, exactly as Q-1's claimable treated it.

The one Q-1 test that pinned Q-1's superseded reading
(`test_claimable_fifo_dependencies_and_stop`: "any terminal
state satisfies") now asserts the Q-2 reading, with the
supersession labelled in the test. No other prior test was
changed by this slice.

### 1.2 Barrier sync points — `si/gaia/drain.py`

`drain_task` records the barrier intents still open when a
drain begins; after every round (execution, skip, or
refusal), a barrier intent that has concluded stops the
drain and the observation set returns with sync reason
`barrier` (§10.2 point 2). Intents after the barrier remain
queued for the next drain — the barrier is the proposer's
"I need to see results before anything after this," and
the drain now honours it literally.

### 1.3 Claim-time residence refusals — `si/gaia/drain.py`, `si/gaia/queue.py`

The fold now carries each intent's admission timestamp
(`intent_ts`, the `ts` of its own self-edit row — the
journal timestamp recorded at admission, §12.2). When the
drainer reaches a claimable intent whose queued residence
exceeds `queue_max_residence_seconds`, it journals a
`queue_refusal` with cause `stale-admission` instead of a
claim: the intent folds to `refused-at-claim` (so its
`outcome` dependents see a recorded conclusion), is never
executed under a stale price, and the honest move — a
fresh proposal — is named in the cause. An unparseable
admission timestamp fails closed (never stale on a guess),
the §13.1 claim-lease discipline applied to residence.

### 1.4 The bounded read pool — `si/gaia/drain.py`, `si/journal.py`

The drain loop's claim step is now:

1. Compute the claimable set in §7.1 order.
2. Refuse any stale members (§1.3) and re-fold.
3. If the lowest-sequence claimable intent is an **unkeyed
   read** and the bound allows more than one, claim up to
   `bound` claimable unkeyed reads — claims journaled
   serially, in intent order, **before** any of them
   executes — and execute them concurrently in a
   `ThreadPoolExecutor`. Outcomes journal in completion
   order (§8.3).
4. Otherwise execute the lowest claimable intent serially,
   exactly as Q-1 did: keyed work (writes on their target,
   executions on the work root, house tools on `house`,
   reads with a declared key) and all write/execute work
   never overlap anything.

The bound resolves from policy `queue_max_concurrent_reads`:
absent → **4**; a valid int ≥ 1 → itself, capped at **16**;
malformed, bool, or sub-1 → **1** (serial, most
restrictive). `Q1_CONCURRENCY` remains in the module as the
historical record of Q-1's policy; the drain result now
reports `concurrency_bound` and `skips` alongside Q-1's
keys.

Concurrency needed one enabling change outside the drainer:
**the journal is now thread-safe on the embedded backend**
(`si/journal.py`). `Journal._append` runs under an RLock so
sequence allocation, hash-chain linking, and backend
insert are one atomic unit; `record_self_edit`'s counter,
`set_edit_result`, and `set_delta_status` take the same
lock; `EmbeddedBackend` opens sqlite with
`check_same_thread=False` and serializes its operations
through its own RLock. This is §8.3 implemented literally —
*outcome journaling is serial* — while the tool work
between appends overlaps. The NEDB backend is **not**
claimed thread-safe by this slice (§4).

### 1.5 Labels caught up with the code

The gate's check-2 `depends_on` sub-check text and the
`Intent.barrier` comment no longer say "semantics land in
Q-2"; `si/gaia/task.py`'s and `si/gaia/__init__.py`'s slice
summaries describe Q-2 as built. The B-5 provenance trail
now includes `queue_skip` entries in its queue section
(`si/gaia/service.py`), so a skipped intent's trail shows
why it never ran.

## 2. Pinned semantics (where the spec left the slice a choice)

- **`success` means outcome status `completed`.** The
  executor's other executed status, `timed-out`, carries a
  partial result by its own label; counting it as success
  would let a dependent build on data the journal itself
  calls partial.
- **`outcome` includes `rejected` and `escalated`.** Both
  are recorded conclusions with journaled non-execution
  outcomes (§5.2); a dependent that asked only "did it
  conclude" is answered truthfully by either. It excludes
  `outcome-unknown` (§1.1).
- **Malformed read-bound resolves to 1, not 0.** Q-1's other
  bounds resolve malformed → 0 because 0 is their most
  restrictive meaningful value (no lease, no budget). For a
  concurrency bound, 0 would deadlock the pool's class
  entirely; the most restrictive *working* value is 1
  (serial), which is also Q-1's proven behaviour.
- **Residence malformed → 0** follows the module's existing
  bounds pattern (an unparseable residence bound refuses
  everything at claim — most restrictive, and loud in the
  journal rather than silent).
- **Pool batch composition:** only the claimable set's
  unkeyed reads, and only when the *lowest* claimable
  intent is one of them — a keyed or state-changing head of
  the queue is never overtaken by a pool batch in the same
  round. Within a batch, claims are in intent order;
  outcomes may complete in any order (§8.3).

## 3. What is NOT in this slice

- **Q-3** — operator stop/resume against a *live* drain,
  write/execute `outcome-unknown` dispositions, the service
  adapter's queue vocabulary, and the SSE queue-event
  convention. Q-2's drainer still defers expired
  write/execute claims exactly as Q-1 did.
- **Q-4** — all measurement. The bound 4 is §8.2's starting
  policy on a 2 vCPU target, labelled UNMEASURED there and
  here; this slice's tests demonstrate *that* reads overlap
  and *that* the bound caps the overlap, never how fast
  anything is.
- **A threaded task loop.** §9's pipeline is exercised at
  the admission/drain boundary (§1.4/§5); `run_task` still
  proposes, admits, drains, and perceives in sequence.
- **NEDB-backend thread-safety.** The concurrent pool is
  built and tested on the embedded (sqlite) journal
  backend. The NEDB backend keeps its single pgwire
  connection discipline; concurrent drains on NEDB-backed
  bundles are not claimed by this slice.
- **A thinking proposer** — unchanged from B-3/B-4:
  proposals still come from scripted proposers or the
  miniature mind.

## 4. Evidence (acceptance, criterion by criterion)

Suite discipline (standing rule, 2026-10-30): the builder
ran the NEW tests plus the affected existing subsets; the
single full-suite run is the reviewer's. Subset results on
this branch (venv `si2-venv`):

- new: `tests/test_gaia_q2_drain.py` — **10 passed**;
- affected subsets, one combined run: all
  `tests/test_gaia_*.py` + `test_gate.py`,
  `test_probe_gate.py`, `test_admission_flip.py`,
  `test_journal_ordering.py`, `test_loop.py`, `test_serve.py`,
  `test_si1.py`, `test_run.py`, `test_admit.py`,
  `test_bundle.py` — **322 passed** (the Q-1 claimable test
  updated per §1.1 included; counts are the run's own).

| §16 Q-2 element | Demonstrated by |
| --- | --- |
| `outcome` vs `success` semantics | `test_gaia_q2_drain.py::test_success_dependent_skipped_outcome_dependent_runs` (a failed `fs_read`: its `outcome` dependent executes, its `success` dependent does not), `::test_success_dependent_runs_after_successful_dependency`; `test_gaia_q1_drain.py::test_claimable_fifo_dependencies_and_stop` (updated, §1.1) |
| Dependent skipping, journaled with cause + dependency outcome seq; cascades | `test_success_dependent_skipped_outcome_dependent_runs` (skip folds with `skip_cause`/`skip_dependency_seq`/`skip_dependency_outcome_seq`; the skipped intent's own `success` dependent is skipped in turn; the skipped file is never written) |
| Barrier sync point (§10.2.2) | `test_barrier_flushes_perception_and_holds_the_rest` (first drain returns sync_reason `barrier`, perception holds only the barrier outcome, the later intent stays queued; the next drain completes it) |
| Declared ordering keys under concurrency | `test_keyed_reads_and_writes_never_overlap` (two reads sharing a declared key, and a write, overlap *nothing*; the two unkeyed reads overlap each other), with Q-1's `test_declared_ordering_key_and_barrier_are_priced` (gate pricing) |
| Read pool at `queue_max_concurrent_reads` (§8) | `test_four_unkeyed_reads_execute_concurrently` (a `threading.Barrier(4)` inside the executor is reached — peak concurrency exactly 4 — with claim < outcome per intent and `verify_chain()` true), `test_read_pool_bound_two_caps_concurrency` (peak exactly 2 under policy 2), `test_max_concurrent_reads_resolution` (default 4, cap 16, malformed/sub-1 → 1) |
| Claim-time residence refusals (§12.2) | `test_stale_admission_refused_at_claim` (residence 0: refused at claim, cause names `stale-admission`, never executed, folds `refused-at-claim`) |
| Pipeline §9 exercised, journal as witness | `test_admission_overlaps_an_in_flight_drain` (batch N+1 admitted through the real gate path while batch N's execution is blocked; the journal shows claim(N) < intent(N+1) < outcome(N); batch N+1 then drains) |
| Serial outcome journaling / journal thread-safety (§8.3) | `test_shared_journal_appends_from_many_threads_verify` (8 threads × 25 appends on one shared embedded journal: 200 unique sequences, chain verifies) plus the pool tests' `verify_chain()` assertions |

## 5. Files

| File | Change |
| --- | --- |
| `si/gaia/drain.py` | §7.3 requirement semantics + skip propagation; barrier sync point; §12.2 residence refusals; §8 read pool (claims serial, executions concurrent for unkeyed reads); result gains `skips` + `concurrency_bound` |
| `si/gaia/queue.py` | `record_queue_skip` writer; fold carries skip fields + `intent_ts` (admission timestamp) |
| `si/journal.py` | thread-safety: `Journal` append/result locks; `EmbeddedBackend` `check_same_thread=False` + serialized operations |
| `si/gaia/service.py` | provenance trail's queue section includes `queue_skip` |
| `si/gate.py`, `si/gaia/task.py`, `si/gaia/__init__.py` | labels/summaries updated: Q-2 semantics are live, not future |
| `tests/test_gaia_q2_drain.py` | **new** — 10 tests (semantics, skips, barrier, residence, pool, pipeline, journal threads) |
| `tests/test_gaia_q1_drain.py` | the one claimable assertion block that pinned Q-1's superseded dep reading now asserts Q-2's, labelled (§1.1) |
| `docs/SI-52-q2-concurrency-deps.md` | this doc |
