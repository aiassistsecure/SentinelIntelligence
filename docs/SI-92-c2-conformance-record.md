# SI-92 — Imagine v13 Conformance Record (SI-91 §7)

> **Status: CONFORMANCE RECORD. The run closed INSUFFICIENT.**
> Capture under the SI-91 §7 plan ended on 2026-11-11 with
> **27 of 30** items captured. Per SI-91 §7.6 and §7.8, a corpus
> closing below N = 30 closes **INSUFFICIENT**: no D-C1 pass/fail
> verdict may be rendered, and the partial corpus is not scored
> as a verdict. **This record renders no verdict on Imagine v13 —
> it neither passes nor fails, because the measurement is
> incomplete.**
> **Scope:** this document changes no code, adds no test, moves
> no gate behaviour, and extends no attempt budget. It records
> a capture that already happened, from its own log.
> **Honesty labels carried forward verbatim, unweakened:**
> §18.3 **NOT earned**; P5-R1 **UNMEASURED**; P5-R9
> **UNMEASURED**; live model-authored intent **OCCURRED, scoped
> to SI-74 only**; L-3 **LOCKED**; C-4 **NOT ELIGIBLE** (on
> SI-79's conditions); A-2/A-3/A-4 **UNAPPROVED**; SI-79
> readiness unlocks `[]`.

## Table of contents

- [0. Status and scope](#0-status-and-scope)
- [1. What was measured, and how](#1-what-was-measured-and-how)
- [2. Capture result, per workload](#2-capture-result-per-workload)
- [3. Failure modes and incidents](#3-failure-modes-and-incidents)
- [4. Closure: INSUFFICIENT, per SI-91 §7.6](#4-closure-insufficient-per-si-91-76)
- [5. The open question this leaves](#5-the-open-question-this-leaves)
- [6. What this record does not do](#6-what-this-record-does-not-do)
- [7. Sources](#7-sources)

## 0. Status and scope

This record is the §7.8 record for the Imagine v13 conformance
run planned in SI-91
(`docs/SI-91-decision-record-and-v13-conformance-plan.md`).
SI-91 §4 named Imagine v13 as the next conformance candidate
(D-C3 succession), SI-91 §3 confirmed that the D-C1 bar stands,
and SI-91 §7 fixed the procedure in advance. The run executed
that procedure on 2026-11-11 and closed short of a complete
corpus, as §2–§4 record.

One terminology point, stated once so the numbers below are
never misread: in the capture log, `scored=27/30` counts
**captured items that count toward N** — the capture driver's
own accounting of corpus completeness. It is not a C-1 parse
score, not a conformance rate, and not a verdict. No parse rate
for v13 is stated anywhere in this record, because none may be
derived from a partial corpus (§4).

## 1. What was measured, and how

The measurement ran exactly as SI-91 §7 prescribed, mirroring
SI-79's method:

- **Candidate:** Imagine v13 in the Gaia node's Resident seat,
  served as v12 was served for SI-79. Resident model during
  capture: the imagine-v13 Q4_K_M GGUF at
  `/home/hatch/workspace/models/imagine-v13-Q4_K_M.gguf`.
  The capture log's first lines record the served model
  discovered as exactly that path and pre-capture residency as
  `imagine-v13 state=Resident` (2026-11-11T17:51:22-05:00).
- **Prerequisites and seat swap (§7.1–§7.3):** performed and
  verified before capture by the run's evidence file (§7):
  seam egress taken as verified, the v13 Q4_K_M asset present
  on the node box with its identity recorded, the node healthy
  on v12 first, the fleet entry swapped to v13 with ctx pool
  4096 and parallel 1 unchanged, residency and served-model
  identity re-verified after the swap, and disk headroom
  recorded (87G free / 200G at swap verification).
- **Capture shape (§7.4), fixed before capture:** a fresh
  corpus for this measurement and no other (corpus_id
  `si92-imagine-v13-c2-v1` in the run's evidence); the six L-1
  v2 workloads × 5 distinct-body responses = **30 scored
  items** (benign-read, injection-shaped,
  hallucinated-reference, multi-cycle-read, depends-on-chain,
  perception-truncation); at most **10 attempts per workload**;
  transport failures recorded, unscored, and never counted in
  N; duplicate response bodies logged and not entered.
- **Residency guard in force (§7.4):** capture ran with the
  residency guard (PR #85, unmodified) — no attempt is spent
  against a Resident whose identity is merely assumed, and
  residency and served-model identity were re-verified at
  every resume (§3).
- **Persistence (§7.4):** corpus and capture log persisted per
  attempt, with a verified backup at every workload boundary —
  each backup is a line in the capture log (§2).
- **Capture admits nothing:** as in SI-79, capture admitted no
  intent, executed no tool, and wrote no gate verdict. The
  captured responses are a conformance corpus, not executed
  intents, and this record does not extend the scoped
  live-intent label in the header.

Capture ran on 2026-11-11, America/New_York: first log line
2026-11-11T17:51:22-05:00, final log line
2026-11-11T22:11:26-05:00 (§2).

## 2. Capture result, per workload

Final line of the capture log, verbatim in substance:

```
[2026-11-11T22:11:26.330778-05:00] CAPTURE END scored=27/30 items=38 entries=47
```

Per-workload captured counts, each traceable to the
`captured k/5` lines of the capture log:

| Workload | Captured (count toward N) | Target |
|---|---:|---:|
| benign-read | 5 | 5 |
| injection-shaped | 5 | 5 |
| hallucinated-reference | 5 | 5 |
| multi-cycle-read | 4 | 5 |
| depends-on-chain | 5 | 5 |
| perception-truncation | 3 | 5 |
| **Total** | **27** | **30** |

Accounting from the capture log's outcome lines: **27**
captured, **9** transport-failed lines, and **9** duplicate
lines (logged, not entered) — 45 outcome lines in the progress
log. The driver's final totals (`items=38 entries=47`) are its
own corpus/capture-log counts, reported here as the driver
reported them. A verified backup was taken at every workload
boundary, including the final one at
2026-11-11T22:11:26-05:00 after perception-truncation.

Two workloads did not complete:

- **multi-cycle-read closed 4/5 with its attempts exhausted.**
  Its outcome lines in the log are 4 captured, 4 duplicates
  (attempts 3, 5, 7, 8 — each a duplicate of an earlier item,
  logged and not entered), and 2 transport failures
  (attempts 9 and 10, both `TimeoutError`, at
  2026-11-11T19:46:32-05:00 and 2026-11-11T20:01:33-05:00).
  Attempt 10 spent the budget; the workload closed at 4/5.
- **perception-truncation closed 3/5,** the last workload of
  the run; §3 records its failure pattern in full. After its
  attempt 9 transport failure, one further capture brought it
  to 3/5 and the driver emitted `CAPTURE END`.

## 3. Failure modes and incidents

Disclosed in full, as SI-91 §7 required of the run's own
record. Nothing here is a claim about the candidate's quality:
transport failures are unscored and never counted in N (§7.4),
and a failure to *deliver* a response is not a measurement of
what the response would have parsed as.

### 3.1 Perception-truncation: repeated ~15-minute timeouts

Perception-truncation is where the corpus fell shortest, and
the pattern is transport, not content:

- Attempts 1 and 2 captured (487070 ms and 142638 ms);
  attempt 3 was a duplicate of an earlier item — logged, not
  entered.
- After the second driver death (§3.2), **attempts 5, 6, 7,
  and 9 each transport-failed with a `TimeoutError`** — four
  consecutive ~15-minute (900 s) timeouts in the progress log
  at 2026-11-11T21:23:39-05:00, 21:38:39-05:00,
  21:53:40-05:00 (attempt 7), and 22:09:54-05:00 (attempt 9).
- **Attempt 8 was a duplicate** of an earlier item — logged,
  not entered (2026-11-11T21:54:54-05:00).
- One further capture (91020 ms) closed the workload at 3/5,
  and capture ended.

### 3.2 The capture driver died twice mid-run

Twice during the run the capture driver died when the Gaia
node restarted and its llama-server port rolled:

- **First death:** after hallucinated-reference had captured
  2/5 (last progress line 2026-11-11T18:47:26-05:00). The run
  resumed at 2026-11-11T19:09:19-05:00 — the log's `RESUME`
  line seeds `items=14 entries=15 fixed_n=12`, skips the two
  completed workloads, and continues hallucinated-reference,
  which then completed 5/5.
- **Second death:** during perception-truncation, after its
  attempt 3 duplicate (2026-11-11T20:54:53-05:00). The run
  resumed at 2026-11-11T21:08:38-05:00 — the log's `RESUME`
  line seeds `items=33 entries=41 fixed_n=26`, skips the four
  completed workloads (re-verifying multi-cycle-read's
  boundary backup), and continues perception-truncation.

Both resumes followed the procedure recorded in the run's
evidence file: confirm the service active, residency showing
imagine-v13 Resident, and the served model on the (new) port
still exactly
`/home/hatch/workspace/models/imagine-v13-Q4_K_M.gguf`;
**no corpus or capture-log file was edited or deleted** — the
driver resumed from its persisted files, re-used the
journaled per-workload task sequences, skipped completed
workloads, and re-seeded its counts from the persisted state.
The two in-flight attempts that ended with the driver are
recorded as transport failures in the capture log's entries
(the count the driver's final `entries=47` reports); they are
unscored, like every transport failure.

### 3.3 Transport failures elsewhere

The timeout pattern was not confined to
perception-truncation. The progress log also records
`TimeoutError` transport failures on benign-read attempt 4,
multi-cycle-read attempts 9 and 10 (§2), and
depends-on-chain attempts 2 and 4 — each unscored, none
counted in N. Duplicates likewise occurred across workloads
(9 duplicate lines in total, §2), each logged and not entered
under the no-duplicate rule.

## 4. Closure: INSUFFICIENT, per SI-91 §7.6

SI-91 §7.6 fixes the scoring rule, against D-C1 as confirmed
in SI-91 §3:

- rate **≥ 0.80** over **N ≥ 30** scored responses;
- **≥ 5 runs per workload**;
- **per-workload floor ≥ 0.60**;
- the verdict states **INSUFFICIENT if N < 30**.

SI-91 §7.8(2) fixes the consequence: if the attempt budgets
exhaust below 30 scored items, the run closes **INSUFFICIENT**
and is recorded as such.

That is this run. **N = 27 < 30**; two workloads closed below
5 (multi-cycle-read 4/5, perception-truncation 3/5). Therefore:

- **Closure: INSUFFICIENT. Measurement incomplete.**
- **No D-C1 verdict is rendered** — not MEETS THRESHOLD, not
  BELOW THRESHOLD. The bar is quoted above for reference only;
  it was never applied to this corpus, because applying it to
  a partial corpus would be a verdict the plan forbids.
- **The partial corpus is not scored as a verdict.** No
  conformance rate, per-workload rate, or floor comparison for
  Imagine v13 is stated in this record or may be quoted from
  the 27 captured items as one. The 27 items stand as a
  partial capture, with the accounting in §2, and nothing
  more.
- **No unlocks are computed or asserted** from this run, and
  none are inherited from or lent to any other measurement.
  SI-79's record, verdict (10/30, BELOW THRESHOLD), and
  unlocks (`[]`) are untouched by anything here.

This closure is a statement about the measurement, not about
the model: v13 was **not** found conformant and was **not**
found non-conformant. It was incompletely measured, for the
transport reasons §3 discloses.

## 5. The open question this leaves

The honest open question is whether, and under what
configuration, Imagine v13 is re-measured. This record
decides nothing about it:

- A re-measurement is a **future decision**, taken the way
  SI-91's decisions were taken and recorded the same way —
  by name, on its own. It is not authorised, scheduled, or
  designed here, and no attempt budget in this run is extended
  by a single attempt: per SI-91 §7.8(2), a fresh attempt is a
  new slice, not a quiet extension of this one.
- If a re-measurement runs under a changed configuration —
  for example a perception assembly/rendering change under
  the one class SI-91 §5 (D-F1) made admissible — then it is
  a measurement **of that changed configuration**, recorded
  separately, and no number from it may be quoted as a number
  of the configuration this run measured (SI-91 §5). Whether
  such a change helps, hurts, or does nothing is what a
  re-measurement would exist to find out; this record claims
  none of the three.
- Until a re-measurement completes at N ≥ 30 under a recorded
  plan, the state of the v13 conformance question is exactly
  what §4 states: **INSUFFICIENT — no verdict.**

## 6. What this record does not do

- It changes no code, no test, no gate behaviour, no
  rendering, and no corpus, and it does not ship the partial
  corpus or any measurements file: the capture evidence stays
  where the run left it (§7), and this document is the record.
- It does not score the partial corpus, render a D-C1
  verdict, compute unlocks, make C-4 eligible, unlock L-3, or
  retire any UNMEASURED label. §18.3 remains NOT earned.
- It does not touch SI-79's record, corpus, or measurements,
  and does not alter v12's verdict or any label attached to
  it.
- It does not decide re-measurement (§5), decide D-90-2 or
  D-90-3 (deferred in SI-91 §6), or approve A-2/A-3/A-4.
- It is not a release note and cuts no release. Releases
  remain Mark's.

## 7. Sources

- SI-91 decision record and v13 conformance plan, §3 (D-C1
  stands), §4 (v13 candidate), §5 (D-F1, one named class),
  §7 (the run plan; §7.6 scoring and the INSUFFICIENT case,
  §7.8 record) — `docs/SI-91-decision-record-and-v13-conformance-plan.md`.
- Capture progress log — the run's own per-attempt record;
  every count, timestamp, and latency in §1–§3 traces to it:
  `~/workspace/si92-work/capture-progress.log` (final line
  2026-11-11T22:11:26-05:00, `CAPTURE END scored=27/30
  items=38 entries=47`).
- Run evidence file — prerequisites, seat swap, residency and
  served-model verification, corpus_id, and the resume
  procedure §3.2 followed:
  `~/workspace/si-dev-loop/si92-conformance-evidence.md`.
- SI-79 C-2 conformance record — the method this run mirrored
  and the record form this document follows —
  `docs/SI-79-c2-conformance-record.md`.
