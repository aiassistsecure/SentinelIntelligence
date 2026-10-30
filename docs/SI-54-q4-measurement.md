# SI-54 — Gaia Q-4: measurement — drain versus serial, on the record

**Status:** implemented on `si-54/q4-measurement` against the
SI-50 queue-drain spec (PR #55, merged), SI-51 Q-1 (PR #56),
SI-52 Q-2 (PR #57) and SI-53 Q-3 (PR #58). This is §16's
**Q-4** slice, built exactly to that scope: a measurement
harness (`si/gaia/measure.py`), scripted miniature tasks
(read-heavy, write-heavy, mixed, dependency-chained) run
under the serial cycle and under the queue at read bounds
1/2/4/8, and this document — wall times, queue residence
distributions, journal growth per task, the leash and
pipeline measurements SI-53 §2 and §9.2 left open, and a
recommendation, from those numbers, for §8.2's default and
§12's caps. §0 says plainly what is now measured and what
stays UNMEASURED.

| Quantity | Q-4 state |
| --- | --- |
| Drain vs serial wall time, 4 scripted miniature workloads (§16) | **measured** — §2; the serial cycle is faster at every bound for every workload at this scale |
| Read-bound sweep 1/2/4/8, read-heavy (§8.2, §16) | **measured** — §2 + the focused 15-rep sweep (§2.1): benefit plateaus at bound 4 |
| Queue residence distributions (§12.2, §16) | **measured** — §3, from journal timestamps (admission → claim) |
| Journal growth per task (§18.4, §16) | **measured** — §4: queue 81 entries/task vs serial 64 at size 16 (+26.6%) |
| Leash poll cost (SI-53 §2) | **measured** — §5: no penalty distinguishable from noise at this scale |
| Leash stop latency (SI-53 §2, §11.2) | **measured** — §5: median 53 ms from stop journaled to drain returned (n=3) |
| Two-deep pipeline overlap (§9.2) | **measured** — §6: median saving ≈ 0.24 s on a 1 s execution (n=3, ranges overlap) |
| §8.2 default / §12 caps recommendation (§16, §20.4) | **given, from the numbers** — §7: keep 4; keep §12's caps |
| Claim lease (300 s) adequacy | **UNMEASURED** — no crash/orphan was induced in this slice; the cap keeps §12.2's derivation, not Q-4 evidence |
| NEDB-backend queue performance | **UNMEASURED** — every run uses the embedded (sqlite) journal backend, as Q-1..Q-3's suites do |
| Real Hearth path (SI-45) | **UNMEASURED** — unchanged; nothing in this slice touches it |
| Non-miniature / real-base / other workloads | **UNMEASURED** — claims are scoped to the four scripted miniature workloads below |

---

## 1. Method and conditions

- **Harness:** `si/gaia/measure.py` (`python -m
  si.gaia.measure --size 16 --reps 5 --out
  docs/SI-54-measurements.json`). Every repetition builds a
  **fresh miniature bundle** (tiny seed, embedded journal)
  and a fresh work root, so journal growth and residence
  are attributable to exactly one task. Setup (bundle
  creation, seeding) is excluded from every wall time.
  The committed JSON records are the harness's records
  compacted to single-line JSON for commit (same data,
  no pretty-printing).
- **Workloads** (size 16 intents each, the §12.1 default
  task size; payloads a uniform 2,048 bytes):
  *read-heavy* — 16 `fs_read` of distinct pre-created
  files (unkeyed, pool-eligible); *write-heavy* — 16
  `fs_write` of distinct targets (keyed, always serial);
  *mixed* — 8 reads / 8 writes alternating;
  *dependency-chained* — 16 reads, each declaring
  `depends_on` `success` on its predecessor.
- **Modes:** *serial* — `execute_tool_cycle` per intent
  (gate + execute, one at a time: the pre-queue loop's
  shape); *queue* — one `queue_batch`, per-member
  `admit_tool_intent`, then one `Drainer.drain_task` at
  `queue_max_concurrent_reads` ∈ {1, 2, 4, 8}. Queue total
  wall = admission wall + drain wall (both reported).
- **Residence** is §12.2's own quantity: the fold's
  journaled `claim_ts` minus `intent_ts` per intent,
  aggregated over all intents and repetitions.
- **Conditions (measured, from the canonical run's
  record):** host `htch-runtime`, Linux
  7.0.0-39-generic x86_64, Python 3.12.3, **2 CPUs**
  (`os.cpu_count()`), 7.7 GiB RAM, CPU only. This VM also
  runs the agent runtime, so the host is contended;
  ranges are reported everywhere and no conclusion below
  rests on a single run or on a difference smaller than
  the range overlap that the text does not flag.
- **Completeness:** every repetition of every
  configuration executed all 16 intents (`executed_min`
  16 throughout); the raw records are
  `docs/SI-54-measurements.json` (canonical, 5 reps) and
  `docs/SI-54-read-sweep.json` (focused sweep, §2.1).

## 2. Wall times: drain versus serial

Median total wall per 16-intent task over 5 repetitions
(range in parentheses), seconds:

| Workload | Serial cycle | Queue b=1 | Queue b=2 | Queue b=4 | Queue b=8 |
| --- | --- | --- | --- | --- | --- |
| read-heavy | **0.186** (0.090–0.274) | 0.408 (0.210–0.488) | 0.390 (0.379–0.468) | 0.309 (0.145–0.389) | 0.270 (0.214–0.400) |
| write-heavy | **0.319** (0.198–0.424) | 0.536 (0.298–0.789) | 0.660 (0.387–0.690) | 0.425 (0.263–0.718) | 0.480 (0.303–0.724) |
| mixed | **0.229** (0.183–0.366) | 0.504 (0.377–1.098) | 0.561 (0.202–0.744) | 0.608 (0.424–0.908) | 0.722 (0.529–0.874) |
| dependency-chained | **0.317** (0.224–0.483) | 0.874 (0.747–1.089) | 0.460 (0.094–0.902) | 0.366 (0.094–0.631) | 0.676 (0.196–1.088) |

The headline, stated without varnish: **at miniature
scale the queue is slower than the serial cycle for every
workload at every bound.** Per-intent tool work here is
milliseconds; the queue's per-task machinery — batch
entry, claim entries, re-folds after every step — costs
more wall than the concurrency it unlocks saves. Derived
throughput (intents ÷ total wall, medians) makes the same
point: serial read-heavy ≈ 86 intents/s vs the queue's
best ≈ 59 intents/s at bound 8. What the queue buys at
this scale is what Q-1..Q-3 built it for — batched
admission, pipelining (§6), governance bookkeeping (§4) —
not raw speed on millisecond tools. Any future claim of a
queue speedup needs a workload whose per-intent execution
time dwarfs the bookkeeping (execute-class work is the
obvious candidate) and a measurement like this one; this
slice measured the workloads §16 named, and on those the
answer is no speedup.

Within the queue, the bound sweep reads as §8 predicted
for reads and as §8.1 requires for everything else:

- **Read-heavy:** medians improve monotonically with the
  bound in the canonical run (0.408 → 0.390 → 0.309 →
  0.270). The focused sweep (§2.1) refines where the
  plateau is.
- **Write-heavy:** no bound effect is expected (writes
  are keyed and serial at every bound) and none is
  claimed: medians bounce 0.425–0.660 with fully
  overlapping ranges — that spread is host noise, and the
  slice reports it as noise.
- **Mixed:** half the intents are writes; medians do not
  improve with the bound (0.504 → 0.722 across the sweep,
  ranges overlapping). No concurrency benefit is claimed.
- **Dependency-chained:** a `success` chain can only
  drain in sequence, so no bound effect is expected;
  medians (0.366–0.874) move with admission-wall noise
  (17–207 ms across repetitions), not with the bound. No
  effect is claimed.

### 2.1 Focused read-bound sweep (15 repetitions)

Because the canonical run's bound-8 median rested on 5
repetitions with overlapping ranges, the read-heavy sweep
was re-run at 15 repetitions per bound (same harness,
same conditions; raw record
`docs/SI-54-read-sweep.json`):

| Bound | Median total wall (s) | Range (s) | Median residence (s) |
| --- | --- | --- | --- |
| 1 | 0.372 | 0.146–0.911 | 0.153 |
| 2 | 0.400 | 0.148–0.909 | 0.151 |
| 4 | **0.242** | 0.090–0.524 | 0.116 |
| 8 | 0.258 | 0.118–0.487 | 0.086 |

Reading: bounds 4 and 8 sit on one plateau (medians
within 7% of each other, ranges overlapping almost
entirely) and both sit clearly below bounds 1/2 at the
median. **The measured contention point is 4:** past it,
extra readers buy no measurable wall time — §8.2's
prediction ("past a small bound, extra readers mostly
queue at the journal") is what the numbers show. The
§8.2 recommendation in §7 is derived from this sweep by
the harness's stated rule (smallest bound within 10% of
the best median), which returns **4**.

## 3. Queue residence distributions

Admission → claim, seconds, aggregated over all intents
and the canonical run's 5 repetitions (80 intents per
row):

| Configuration | Median | p95 | Max |
| --- | --- | --- | --- |
| read-heavy b=1 | 0.168 | 0.350 | 0.414 |
| read-heavy b=4 | 0.102 | 0.236 | 0.285 |
| read-heavy b=8 | 0.107 | 0.151 | 0.160 |
| write-heavy b=1 | 0.252 | 0.432 | 0.491 |
| write-heavy b=4 | 0.187 | 0.475 | 0.633 |
| mixed b=1 | 0.244 | 0.729 | 0.913 |
| mixed b=4 | 0.277 | 0.477 | 0.700 |
| dependency-chained b=1 | 0.476 | 0.718 | 0.765 |
| dependency-chained b=4 | 0.109 | 0.426 | 0.469 |

(Full distributions for every configuration are in the
raw JSON.) Two facts matter for §12.2: residence at this
scale is **sub-second everywhere** — the largest single
residence observed in any configuration is 0.913 s,
against a 3,600 s cap — and chained intents reside
longest at bound 1, as they must: each waits for its
predecessor's whole admission-plus-execution chain.

## 4. Journal growth per task (§18.4, priced)

Journal entries (tip-sequence delta) per 16-intent task,
identical across all 5 repetitions of every configuration
— this quantity is deterministic, not sampled:

| Mode | Entries / task | Entries / intent (derived) | Journal file growth (median) |
| --- | --- | --- | --- |
| serial cycle | 64 | 4.0 | ≈ 1.29–1.44 MB |
| queue (any bound, any workload) | 81 | 5.0625 | ≈ 1.52–1.68 MB |

The queue's premium is exactly **+17 entries per task
(+26.6%)**: 1 batch entry + 16 claim entries, over the
serial cycle's 4 entries per intent (self-edit,
provenance, outcome, and the task-fixture class of
bookkeeping the two modes share — the decomposition is
visible in the raw records' per-collection deltas:
`experiences` +9 vs +4 at size 4 in the test-scale runs,
scaling to the batch + claims at size 16). §18.4's cost
is therefore a number: **one extra journal entry per
admitted intent, plus one per batch.** The byte column is
reported with its caveat: sqlite file growth is
page-granular (the file grows in allocation steps and
carries WAL overhead), so bytes are an upper-bound
footprint, not a per-entry cost; entries are the precise
figure. §20.5's compaction question is untouched by this
slice — growth is now priced, not managed.

## 5. The leash: poll cost and stop latency (SI-53 §2)

SI-53 pinned the executor's poll slice at 50 ms as an
implementation constant and left its cost UNMEASURED.
Both halves are now measured (canonical run):

- **Poll overhead.** One `shell_exec` running
  `sleep 0.3`, executed with (a) no leash check, (b) a
  trivial installed check, (c) the drainer's real
  fold-based check — median execution wall over 5
  repetitions: **(a) 0.325 s, (b) 0.319 s, (c) 0.332 s**,
  with raw ranges (0.311–0.414 s across all variants)
  overlapping completely. Conclusion, stated at the
  strength the data supports: at this scale **an
  installed leash check — including the real fold check —
  costs nothing distinguishable from run-to-run noise**
  on a 300 ms process (the differences are ≈ ±2% against
  a ±15% noise band). No finer claim is made: a
  microsecond-level poll cost, if one exists, is below
  this host's resolving power, and the slice says so
  rather than inventing precision.
- **Stop latency.** An operator stop journaled 0.5 s
  into a live drain of a `shell_exec` sleeping 30 s:
  **median 0.053 s from the stop's journaling to the
  drain returning** (n=3; range 0.047–0.059 s), and in
  every repetition the in-flight execution's journaled
  outcome was `timed-out` with the operator-stop cause
  (Q-3's semantics, now timed). The latency is one poll
  slice (50 ms) plus kill and journaling, as §11.2's
  design implies — measured, this time.

## 6. The two-deep pipeline (§9.2)

Batch A is one `shell_exec` sleeping 1 s; batch B is 8
reads. Sequential arm: drain A, admit B, drain B.
Pipelined arm: admit B while A's drain runs, then drain
B. Median total wall over 3 repetitions: **sequential
1.449 s, pipelined 1.207 s** — a median saving of
≈ 0.24 s, the size of batch B's admission wall, which is
exactly the work §9 says overlaps. The ranges overlap
(sequential 1.213–1.459 s; pipelined 1.199–1.391 s), so
this is median-level evidence at n=3, reported as such:
the two-deep regime works and its saving is the
admission leg, neither more nor less. Anything deeper
remains §20's open question, unmeasured here.

## 7. Recommendation (§8.2's default, §12's caps)

From the numbers above, and only from them:

- **`queue_max_concurrent_reads` default: keep 4.** The
  focused sweep's plateau (§2.1) starts at 4; 8 measured
  no better (0.258 vs 0.242 median, overlapping ranges),
  and 1/2 measured worse at the median. §8.2's
  justification — a small multiple of the 2 vCPUs,
  reviewable failure surface — now has the measurement
  it was waiting for, and the measurement agrees with
  the starting point. No code change is made: the
  default already reads 4; this slice replaces its
  UNMEASURED label with the evidence.
- **`queue_max_depth_per_task` (16): keep.** Every
  workload ran at exactly the cap with all 16 admitted
  and executed; nothing in the data argues for a larger
  buffer for default-size tasks, and §12.1's reasoning
  (the queue should never bind before the cycle budget
  does) is unrefuted.
- **`queue_drain_budget_seconds` (1920): keep.** The
  slowest measured 16-intent drain wall is ≈ 1.1 s —
  three orders of magnitude under the budget, which is
  derived from the 120 s per-intent timeout worst case,
  a case these workloads never approach. No evidence to
  move it.
- **`queue_max_residence_seconds` (3600): keep.** The
  largest measured residence is 0.913 s (§3). The cap
  exists for stale-pricing protection, not for this
  scale's performance; nothing measured argues against
  it.
- **`queue_claim_lease_seconds` (300): keep, UNMEASURED.**
  Q-4 induced no crash and no orphaned claim, so this
  slice gathered no evidence about the lease; it stands
  on §12.2's derivation alone, and the doc says so
  rather than laundering it through the measured caps.

## 8. What is NOT in this slice

- **No default or policy change.** §7 recommends keeping
  every current value; the slice's code changes are the
  harness, its tests, and the labels in §9.
- **No NEDB-backend measurement** (embedded backend
  only), **no Hearth-path measurement** (SI-45's label
  unchanged), **no non-miniature measurement**, **no
  execute-class workload sweep** (execute class appears
  only in the leash and pipeline measurements, where
  §11.2/§9.2 put it), **no lease/crash measurement**,
  **no compaction** (§20.5 untouched).
- **No tuning from these numbers beyond §7.** A slice
  that moved a default on this evidence would be
  overclaiming: the one candidate move (bound 8, from
  the canonical run's 5-rep medians) was checked at
  15 reps and not sustained.

## 9. Label changes (UNMEASURED → measured, scoped)

Only where a real measurement above backs the words:

- `si/gaia/drain.py` — the module docstring, the
  `Q2_DEFAULT_MAX_CONCURRENT_READS` comment, and
  `Drainer.max_concurrent_reads`: the bound and the
  queue's miniature fs-workload performance are now
  **measured (SI-54)** with the §2/§7 reading (plateau
  at 4, serial faster at this scale); the blanket
  "every performance quantity is UNMEASURED" sentence is
  replaced by the scoped statement — Hearth-path,
  NEDB-backend, non-miniature and lease quantities stay
  UNMEASURED.
- `si/gaia/task.py`, `si/gaia/__init__.py` — the
  "all queue performance remains UNMEASURED (Q-4)"
  summaries updated to Q-4's measured scope.
- Nothing in `si/gaia/hearth.py` changes: the Hearth
  path remains UNMEASURED, deliberately.

## 10. Files

| File | Change |
| --- | --- |
| `si/gaia/measure.py` | **new** — the Q-4 harness: workloads, serial/queue runners, residence + journal-growth accounting, leash (overhead + stop latency) and two-deep pipeline measurements, aggregation, the §8.2 recommendation helper, CLI |
| `tests/test_gaia_q4.py` | **new** — 7 tests (helpers + recommendation purity; all four workloads execute serially and queued with growth/residence assertions; full-measurement shape at test scale incl. leash kill and pipeline arms) |
| `docs/SI-54-measurements.json` | **new** — canonical run's raw records (size 16, 5 reps, bounds 1/2/4/8, leash 5/3, pipeline 3) |
| `docs/SI-54-read-sweep.json` | **new** — focused read-heavy bound sweep raw records (15 reps/bound) |
| `docs/SI-54-q4-measurement.md` | this doc |
| `si/gaia/drain.py`, `si/gaia/task.py`, `si/gaia/__init__.py` | scoped label updates per §9 — no behaviour change |

## 11. Evidence

Suite discipline (standing rule, 2026-10-30): the builder
ran the NEW tests plus the affected existing subsets; the
single full-suite run is the reviewer's. Results on this
branch (venv `si2-venv`):

- new: `tests/test_gaia_q4.py` — **7 passed**;
- affected subsets, one combined run (all
  `tests/test_gaia_*.py` except Q-4, plus `test_gate.py`,
  `test_probe_gate.py`, `test_admission_flip.py`,
  `test_journal_ordering.py`, `test_loop.py`,
  `test_serve.py`, `test_si1.py`, `test_run.py`,
  `test_admit.py`, `test_bundle.py`) — **332 passed,
  2 failed**. Both failures —
  `test_gaia_sse_follower.py::test_real_service_filtered_follow_live_tail`
  and `test_bundle.py::test_journal_doc_tamper_fails_closed`
  — reproduce **identically on pristine main**
  (`9318381`, fresh anonymous clone, same venv, run by
  this builder as a control), so they are pre-existing
  failures on main, not Q-4 regressions; this slice's
  code changes to existing files are docstring/label
  edits only.
