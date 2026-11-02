# SI-78 — Gaia C-3: Threshold Proposal and Readiness
# Review (SI-76 §4.5)

## Status: PROPOSED, NOT DECIDED — and there is no C-2 record

**The threshold in §3 of this document is PROPOSED, NOT
DECIDED, everywhere it appears.** No record's `threshold`
changes: it stays `null` in `docs/SI-77-measurements.json`
and in every other record until Mark decides D-C1
(SI-76 §3.5, §5). This slice touches no existing record
and no code — it is one new document only, CPU-only,
with no live call, no node contact, and no spend.

**There is no C-2 record.** C-2 (SI-76 §4.4) is parked
because Mark has not named a candidate: decision D-C3
(SI-76 §5) is unanswered. SI-76 §5 / D-C3 states that
naming none parks C-2 without penalty and C-1/C-3 still
stand, and SI-76 §4.4 states that if none is named, C-2
does not run and the line reports that state rather than
substituting any model, corpus, or scripted stand-in.
This proposal therefore does **not** rest on a C-2
candidate measurement, because none exists. §1 states
the actual basis in full. No candidate is invented in
this document, no candidate corpus is invented, no
conformance rate is stated for any candidate, and no
candidate is claimed to meet or fail anything.

No live run is scheduled, attempted, or implied as
approved by this document. A live run remains Mark's
per-run decision under D-C2 (§5).

The work in scope is machine learning infrastructure
throughout; the served model in the SI-74 evidence is a
specialised machine learning model.

## 0. Honesty: no §18.1 row changes, no claim changes

**Explicit statement: no §18.1 row changes and no claim
changes in this slice.** This document promotes no row,
demotes no row, earns no claim, and edits no record.
Every row in `docs/SI-77-measurements.json` stays
exactly as SI-75 left it, each promoted by none of
this, and its `readiness` stays `{threshold: null,
unlocks: []}`.

| Element | State, unchanged by this slice |
|---|---|
| SI-63 §18.3 claim | **NOT earned** |
| SI-63 §18.1 row P5-R1 | **UNMEASURED** |
| SI-63 §18.1 row P5-R9 | **UNMEASURED** |
| SI-63 §18.1 rows P5-R2–R8, R10 | As promoted, scoped, by SI-74; cited here, promoted by none of this |
| Live model-authored tool intent | **OCCURRED, scoped to SI-74 only** (2026-11-02, P-5 live run only; 2 parsed intents, `fs_list` ×2; 1 ADMIT + executed + verified, 1 ESCALATE + not executed) |
| L-3 residency-loss drill | **LOCKED** — nothing offline unlocks it (SI-69 §4.5; SI-76 §3.6) |
| A-2 / A-3 / A-4 | **UNAPPROVED** |
| Readiness threshold (D-C1) | **Undecided** — `threshold: null` until Mark decides; §3 is PROPOSED, NOT DECIDED |
| C-2 candidate measurement | **Does not exist** — C-2 parked, D-C3 unanswered (§1) |
| C-4 live retry | **Not authorised, not scheduled** — D-C2 unanswered, and its other conditions are also unmet (§7) |

Labels carried verbatim, unweakened: §18.3 NOT earned;
P5-R1 UNMEASURED; P5-R9 UNMEASURED; live model-authored
intent OCCURRED scoped to SI-74 only; L-3 LOCKED;
A-2/A-3/A-4 UNAPPROVED.

## 1. Basis: what this proposal rests on — and what it does not

### 1.1 The basis that exists

This proposal rests on exactly two evidence sources,
and nothing else:

1. **C-1's harness evidence on corpus size**
   (`docs/SI-77-c1-conformance-harness.md`,
   `docs/SI-77-measurements.json`; SI-77, PR #83). C-1
   built the SI-76 §3 scoring harness and exercised it
   only as a harness check on SI-74's recorded
   responses: folded **N=6** scored responses,
   **parse-ok 2 (2/6)**, stage-1 failures 4, plus 1
   transport failure unscored and never in N. C-1's own
   record states its scope plainly: HARNESS CHECK ONLY —
   not a candidate measurement, not a live result, and
   not a readiness verdict. Its corpus section states
   that N=6 is tiny and the outcomes were already known
   live (SI-76 §3.3(6)). What C-1 contributes to C-3 is
   therefore the *harness*, the metric definition, and
   a worked demonstration that a corpus of this size
   cannot carry a readiness decision — not a rate for
   any candidate.
2. **SI-74's measured intermittence**
   (`docs/SI-74-l2-live-rerun.md` §3). In the two live
   runs of 2026-11-02, the same seat behaved
   inconsistently across runs: **P-4: 0/3 parse-ok,
   3 stage-1 failures; P-5: 2/3 parse-ok among parsed
   responses, 1 stage-1 failure, plus 1 transport
   failure** (900 s timeout, no response to score).
   SI-76 §2.3 draws the consequence this proposal
   follows: conformance on this evidence is
   *intermittent*, not absent — hence a rate over a
   corpus is the right readiness quantity, not one
   anecdotal response.

SI-75 §0 is cited only for the state of the §18.1 rows
as left (P5-R1 and P5-R9 UNMEASURED; the scoped SI-74
promotions standing as SI-74 left them). SI-75 proposed
no conformance number and contributes none here.

### 1.2 The basis that does not exist

- **No C-2 record exists.** SI-76 §4.5 describes C-3's
  scope as "from C-2's record (and C-1 evidence on
  corpus size)". The first half of that basis is absent
  through no defect: D-C3 is Mark's decision, it is
  unanswered, and SI-76 §5 provides that naming none
  parks C-2 without penalty while C-1/C-3 still stand.
  This document proceeds on the second half of the
  §4.5 basis only, and says so rather than silently
  presenting the proposal as C-2-based.
- **No candidate is named, measured, or implied.**
  Nothing here states or implies that any model exists,
  is available, was measured, or would meet or fail the
  proposed threshold. With no recorded corpus for a
  candidate, the honest C-2 result — when and if C-2
  ever runs — would be whatever that corpus folds to;
  for a corpus missing any §3.3 requirement it would be
  INSUFFICIENT CORPUS, not a rate (SI-76 §3.3, §3.6).
- **No new live result exists.** This slice ran nothing
  live. The SI-74 numbers in §2 are a re-statement of
  merged evidence, not a new measurement.

## 2. The evidence, as recorded

### 2.1 Conformance folds (SI-74, re-scored by C-1)

Metric (SI-76 §3.2, as implemented by C-1): a response
is parse-ok exactly when `stage_reached == 6` and
`failing_stage is None` through the existing §9 parser
(`parse_proposal` in `si/gaia/proposer.py`, unmodified
by C-1 and untouched by this slice). No partial credit.
Transport failures are unscored and never enter N.

| Scope | N (scored) | Parse-ok | Stage-1 failures |
|---|---|---|---|
| SI-74 P-4 run | 3 | 0 (0/3) | 3 |
| SI-74 P-5 run | 3 | 2 (2/3) | 1 |
| Folded (C-1 harness check) | 6 | 2 (2/6) | 4 |

Plus, in P-5, 1 transport failure with no response to
score. The folded 2/6 mixes two distinct live runs and
is a harness artefact only (SI-77 §2); the per-run
folds are the meaningful figures, and neither fold is
a new measurement of any candidate. The two per-run
rates — 0/3 and 2/3 — disagree with each other. That
disagreement is the intermittence the corpus-size
argument in §4 is built on.

### 2.2 Latencies (SI-74 §3)

Client-observed gateway latencies, journaled per call
in SI-74 §3, labelled in the journal as client-observed
and explicitly **not** a Hearth-path performance
measurement (that is why P5-R1 stays UNMEASURED):

- P-4 (§3.1, line 130): 325,676 ms / 260,907 ms /
  434,400 ms — range **260,907–434,400 ms** per attempt.
- P-5 (§3.2, lines 183–184; also §4 row P5-R2):
  830,963 ms / 196,293 ms / 741,441 ms / 900,101 ms —
  range **196,293–900,101 ms** per attempt, the last
  value being the 900 s timeout on the
  transport-failed attempt.

§6 derives the cost estimate from these values and
these values only, by arithmetic shown there.

## 3. The proposal: PROPOSED, NOT DECIDED

> **PROPOSED, NOT DECIDED.** The following is a proposal
> for Mark's decision D-C1 (SI-76 §5). It is binding on
> nothing. Until Mark decides, `threshold` stays `null`
> in every record, no MEETS THRESHOLD or BELOW
> THRESHOLD verdict exists for any corpus, and no slice
> may state one (SI-76 §3.5 steps 2–3).

**Proposed threshold — one rate, one N basis, one
per-workload floor:**

| Component | Proposed value |
|---|---|
| §8-conformance rate (SI-76 §3.2 metric) | **≥ 0.80** — at least 24 parse-ok of 30 scored responses |
| Corpus / N basis | **N ≥ 30 scored responses**, fixed before scoring (SI-76 §3.3(4)), from a corpus satisfying every §3.3 requirement; transport failures unscored, never in N |
| Coverage inside N | At least **5 scored responses per workload** across all six L-1 v2 workloads (SI-76 §3.3(1): the SI-68 three plus `multi-cycle-read`, `depends-on-chain`, `perception-truncation`), rendered as the live proposer renders them |
| Per-workload floor | **≥ 0.60 per workload** — at least 3 parse-ok of 5 in every workload; a corpus meeting the overall rate but failing any workload's floor does not meet the proposed threshold |

Scope notes, so the proposal cannot be read wider
than it is written:

- The rate is the §3.2 shape metric only: how often a
  candidate's recorded output is a §8-conformant
  proposal document. It is not a task-completion rate,
  a quality score, a verdict distribution, or a live-
  behaviour prediction (SI-76 §3.2) — a conformant
  proposal can still be escalated, name the wrong
  tool, or fail to complete a task.
- The proposal governs readiness to *propose* a live
  re-run only (SI-76 §3.5 step 4). It is not a §18.1
  row, not a property of any model beyond a measured
  corpus, and it does not authorise a live run: D-C2
  is separately required, per run (§5).
- The proposal applies to a future C-2 record, if one
  ever exists. It is not retroactively a verdict on
  SI-74 or on the C-1 harness check: those folds are
  below the proposed N, were not gathered as a §3.3
  candidate corpus for a threshold comparison, and are
  cited here as intermittence evidence only. Stating a
  MEETS/BELOW verdict against them would fail SI-76
  §4.5's honesty gate, and this document states none.
- If Mark decides a different rate, N, or floor under
  D-C1, the decided values replace this proposal in
  full; this document remains the record of what was
  proposed, not a second source of the threshold.

## 4. Why this corpus size: intermittence arithmetic

All figures below are arithmetic on the §2.1 folds —
no new measurement is claimed.

- **One response dominates a small corpus.** A single
  scored response is worth 33.3 percentage points at
  N=3 (one P-4 or P-5 run), 16.7 points at N=6 (the C-1
  fold), and 3.3 points at N=30. A threshold decided
  at run-scale N would be decided, in effect, by which
  single responses happened to arrive.
- **The observed runs already disagree at N=3.** P-4
  folded 0/3 and P-5 folded 2/3 on the same day, same
  seat. Any rate threshold placed between those values
  would be met by one run and failed by the other with
  no change in the underlying behaviour — the
  definition of a corpus too small to support the
  decision the threshold exists to gate (§6's cost).
- **N=30 is the smallest round basis that answers
  both defects while satisfying §3.3 coverage.** Six
  workloads × 5 responses each gives every workload
  its own fold (so the per-workload floor in §3 has a
  denominator), keeps any single response at 3.3
  points, and makes the proposed 0.80 an exact count
  (24/30) rather than a rounded fraction. Larger N
  would sharpen the rate further; §3 proposes the
  minimum basis this argument supports, and D-C1 may
  set a larger one.
- **Why a floor, not only an overall rate.** Without a
  floor, 24/30 could be earned while one workload
  folded 0/5 — the P-4 pattern reproduced inside a
  passing corpus. The floor (3/5 per workload) is set
  below the overall rate deliberately: it excludes a
  wholly non-conformant workload without demanding
  uniform performance across workloads of differing
  difficulty, a difference this evidence does not
  measure.

This reasoning proposes; it does not measure. It
claims no confidence interval, no significance test,
and no property of any candidate.

## 5. Decisions for Mark: D-C1 and D-C2

Both decisions are Mark's (SI-76 §5). Neither is
answered, pre-empted, or implied by this document.

### D-C1 — Conformance threshold

**Question:** does the §3 proposal (rate ≥ 0.80 over
N ≥ 30 fixed before scoring, ≥ 5 per workload, floor
≥ 0.60 per workload) become the decided threshold —
or a different rate/N/floor of Mark's choosing — or
does it stay undecided?

- **If Mark decides it (as proposed or amended):** the
  decided values, naming the decision and its date,
  become the only threshold a later record may compare
  against (SI-76 §3.5 step 3). A future C-2 record —
  which additionally requires a candidate named under
  D-C3 — could then state MEETS THRESHOLD or BELOW
  THRESHOLD. Nothing else changes: no live run is
  authorised (that is D-C2), L-3 stays LOCKED, and no
  §18.1 row or §18.3 claim moves.
- **If Mark does not decide it:** `threshold` stays
  `null`, §3 remains PROPOSED, NOT DECIDED, no
  MEETS/BELOW verdict can ever be stated, and the
  C-line's conditional slices stay inactive. C-1 and
  this proposal still stand as built/written; there is
  no penalty and no expiry stated in SI-76 for leaving
  D-C1 open.

### D-C2 — Spend another live run

**Question:** given a MEETS THRESHOLD record against
the D-C1-decided threshold, and the §6 cost estimate,
authorise one specific L-2 retry (SI-76 §4.6, C-4) —
yes or no, for that run only?

- **If Mark says yes for a specific run:** that run —
  and only that run — may proceed, and only if all
  three C-4 conditions hold at the time: a C-2 record
  exists for the candidate to be served, its rate
  MEETS the D-C1-decided threshold, and the yes names
  that run (SI-76 §4.6). The run's results would be
  whatever its journal folds to; a yes guarantees no
  outcome, earns no claim in advance, and is not a
  standing yes for any later run.
- **If Mark says no, or does not answer:** no live run
  is authorised, none is scheduled, and none may be
  implied from this proposal, from a decided threshold,
  or from any conformance record. The state stays
  exactly as §0 records it.

As of this document, D-C2 cannot be exercised at all:
two of C-4's three conditions are independently unmet
(no C-2 record exists; no decided threshold exists),
before any question of Mark's go.

## 6. Wall-clock cost estimate for one further live run

> **Estimate, with its basis — not a measurement, not
> a quote, not a guarantee.** Derived only from the
> SI-74 §3 latencies in §2.2 by the arithmetic below.
> Latencies are client-observed gateway time per call;
> they exclude operator setup, preflight, and any
> post-run folding, so total operator wall-clock for a
> run would be at least the figures here.

Observed per-attempt ranges (§2.2, verified in the
SI-74 doc this run): **P-5: 196,293–900,101 ms** per
attempt, including the 900 s timeout;
**P-4: 260,907–434,400 ms** per attempt. Attempts are
minutes-scale: roughly 3.3–15.0 minutes each in P-5
and 4.3–7.2 minutes each in P-4.

Observed totals, summed from the journaled values:

| Run (SI-74) | Attempts | Sum of latencies | Wall-clock |
|---|---|---|---|
| P-4 suite | 3 | 1,020,983 ms | ≈ 17.0 min |
| P-5 measurement | 4 | 2,668,798 ms | ≈ 44.5 min |
| Both suites in sequence | 7 | 3,689,781 ms | ≈ 61.5 min |

The combined sum is consistent with SI-74's own
reporting: P-5's stated wall was 2,676 s (§2.2 there,
≈ 44.6 min, against 2,668.8 s of gateway latency), and
the two suites ran ≈ 05:01–06:03 EST (§2.1/§2.2
there).

**Estimate for one further live run:**

- One P-5-style measurement run (4 attempts, as
  observed): **≈ 45 minutes** of gateway wall-clock,
  on the observed total. If a new run's four attempts
  each landed at the observed extremes, the same
  attempt count would span ≈ 13 minutes (4 × 196,293
  ms) to ≈ 60 minutes (4 × 900,101 ms).
- One full L-2 retry as C-4 defines it (both suites,
  7 attempts as observed): **≈ 60–62 minutes** of
  gateway wall-clock, on the observed combined total.

Attempt count is the swing factor and is not fixed by
this estimate: SI-75 §2.1 records at most 2 attempts
spent by any task in SI-74 (policy allows 8 per task),
so a run whose tasks spend more attempts costs more,
at ≈ 3.3–15.0 minutes per additional P-5-scale
attempt, and any attempt can consume the full 900 s
timeout. A run could therefore exceed the headline
estimate; it could also finish sooner. That variance
is precisely the cost SI-76 §2.2 says an unmeasured
retry spends blind — and the reason D-C2 is per-run.

## 7. Readiness review: what would and would not follow

Reviewed against SI-76 §4.6/§4.7 and §3.6, from the
state in §0:

- **C-4 (conditional L-2 retry): inactive.** All three
  conditions unmet or unexercisable today — no C-2
  record (D-C3 unanswered), no decided threshold
  (D-C1 unanswered, §3 only proposed), no per-run go
  (D-C2 unanswered). This document activates none of
  them.
- **C-5 / L-3: LOCKED, unchanged.** L-3 unlocks only
  on a real L-2 live success (≥ 1 live task end-to-end,
  §14 chain walkable; SI-69 §4.4/§4.5). No offline
  rate, at any threshold, substitutes (SI-76 §3.6).
- **Readiness signal, stated at its true strength:**
  the only readiness-relevant fact established to date
  is negative-space — the SI-74 seat's conformance was
  intermittent at run scale (0/3, 2/3), and no other
  candidate has been measured at all. That is a reason
  the gate exists, not a readiness verdict for or
  against anything.
- **What happens next is Mark's, not the loop's:**
  D-C3 (name a candidate / corpus capture) enables
  C-2; D-C1 (decide a threshold) enables comparison;
  D-C2 (per-run go) enables C-4. Until then the
  C-line has built C-1, parked C-2 without penalty, and
  written this C-3 proposal — and invents no
  replacement live work (SI-76 §4.2).

## 8. Tests

Docs-only slice; no code, test, policy, or record file
is added, modified, or deleted by this slice except
this document. Builder ran the slice-relevant sanity
suite in the build clone, on `~/workspace/si2-venv`
(Python 3.12.3):

- `tests/test_gaia_conformance_c1.py`: **7 passed** in
  2.00s. The full suite is the reviewer's job on the
  next run; this slice does not claim it.

## 9. Sources

All on `main` at `e1ceb1c3b7d186d93e93f861c2525460a8c03d08`
(PR #83 merge, SI-77) unless stated.

- `docs/SI-76-next-frontier-rfc.md` — §3.5 (how a
  threshold is proposed, not decided), §4.5 (C-3 scope
  and acceptance), §5 (D-C1/D-C2/D-C3), §3.2/§3.3
  (metric and corpus), §3.6 (honesty rules), §2.2/§2.3
  (cost fact; intermittence), §4.4 (C-2 condition),
  §4.6/§4.7 (C-4/C-5 conditions).
- `docs/SI-77-c1-conformance-harness.md` and
  `docs/SI-77-measurements.json` — C-1: §2 folds
  (P-4 0/3, P-5 2/3, folded N=6, parse-ok 2, stage-1
  4, 1 transport failure unscored), §0/`readiness`
  (`threshold: null`, `unlocks: []`), harness-check-
  only scope label.
- `docs/SI-74-l2-live-rerun.md` — §3.1 (P-4 latencies,
  line 130), §3.2 (P-5 latencies, lines 183–184),
  §2.1/§2.2 (run windows; P-5 wall 2,676 s), §4
  (P5-R1/P5-R9 UNMEASURED; scoped promotions).
- `docs/SI-75-l4-tuning-from-live-evidence.md` — §0
  (rows as left; L-3 LOCKED), §2.1 (attempts spent per
  task: maximum 2 of 8), §2.7 (model-output-content
  root cause).
- `si/gaia/proposer.py` — `parse_proposal`, the §9
  parser the §3.2 metric is defined over (unchanged).

End of SI-78. The proposal in §3 is PROPOSED, NOT
DECIDED; `threshold` stays `null` until Mark decides
D-C1, and no live run exists until Mark decides D-C2
for that specific run.
