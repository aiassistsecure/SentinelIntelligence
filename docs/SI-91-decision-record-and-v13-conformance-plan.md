# SI-91 — Decision Record and Imagine v13 Conformance Plan

> **Status: DECISION RECORD (lead decisions, 2026-11-11) plus a run plan.
> No code is changed by this document, no test is added, no gate behaviour
> moves, and nothing is measured, captured, or run live by this slice.**
> The decisions in §3–§6 are recorded by the lead under Mark's standing
> autonomy directive for this programme (§2). Each decision names the
> SI-90 §4 question it answers. §7 is a plan for a future slice, stated
> in advance so the run, when it happens, executes a fixed procedure
> instead of improvising one.
> **Scope:** CPU only. No GPU, no cloud, no spend, no downloads. This is
> machine learning infrastructure throughout; where Imagine is mentioned,
> it is stated as 1.6B parameters, full fine tune.
> **Honesty labels carried forward verbatim, unweakened:** §18.3 **NOT
> earned**; P5-R1 **UNMEASURED**; P5-R9 **UNMEASURED**; live
> model-authored intent **OCCURRED, scoped to SI-74 only**; L-3
> **LOCKED**; C-4 **NOT ELIGIBLE** (on SI-79's conditions); A-2/A-3/A-4
> **UNAPPROVED**; SI-79 readiness unlocks `[]`; training journal
> **file-form tooling, embedded-backend scope only** (§6).

## Table of contents

- [0. Status and scope](#0-status-and-scope)
- [1. Why this record exists](#1-why-this-record-exists)
- [2. Authority: which calls are the lead's](#2-authority-which-calls-are-the-leads)
- [3. Decision D-90-5: D-C1 stands](#3-decision-d-90-5-d-c1-stands)
- [4. Decision D-90-4: Imagine v13 is the next conformance candidate](#4-decision-d-90-4-imagine-v13-is-the-next-conformance-candidate)
- [5. Decision D-90-1 (D-F1): admissible for one named class only](#5-decision-d-90-1-d-f1-admissible-for-one-named-class-only)
- [6. Deferred: D-90-2 and D-90-3 are not decided](#6-deferred-d-90-2-and-d-90-3-are-not-decided)
- [7. The Imagine v13 conformance run plan](#7-the-imagine-v13-conformance-run-plan)
- [8. Sequencing against the other lanes](#8-sequencing-against-the-other-lanes)
- [9. What this record does not do](#9-what-this-record-does-not-do)
- [10. Decision summary](#10-decision-summary)
- [11. Sources](#11-sources)

## 0. Status and scope

This document does four things and nothing else:

1. It records the lead's decisions on three of the five questions
   SI-90 §4 posed (D-90-5 in §3, D-90-4 in §4, D-90-1 in §5).
2. It records, explicitly, that the remaining two questions
   (D-90-2, D-90-3) are **deferred, not decided** (§6), so they are
   carried forward in the open rather than silently dropped.
3. It fixes, in advance, the procedure for the Imagine v13
   conformance measurement that §4 names (§7), and its sequencing
   against the programme's other two active lanes (§8).
4. It states what it does not do (§9). In particular: **SI-79's
   record is untouched.** Nothing in this document edits, re-scores,
   re-labels, or re-interprets `docs/SI-79-c2-conformance-record.md`,
   `docs/SI-79-corpus.json`, or `docs/SI-79-measurements.json`. The
   v12 verdict — 10/30 (0.3333), BELOW THRESHOLD against D-C1,
   unlocks `[]` — stands exactly as recorded.

No decision here is a measurement, a rate, or a claim about any
model. The v13 measurement does not exist until §7's procedure runs
and its own record is merged.

## 1. Why this record exists

SI-90 (PR #97, merged 2026-11-04, main `d6f8d2f0`) closed the
training-journal line and posed five decisions in its §4. The RFC
framed all five as Mark's, and the dev loop held at state-check-only
from 2026-11-04 to 2026-11-11 — a full week — waiting for answers.
During that week the loop's runs verified the same unchanged state
every 30 minutes (no open PRs, main tip unchanged, Glimmer seed
intact by reference), and the hold reports were suppressed in the
chat as unchanged, so the thread went silent.

On 2026-11-11 Mark corrected that, in substance:

- "this was supposed to carry on without interruption"
- "where there wasnt guidance you were meant to lead"
- "if your agents need a directive, give them one"
- "I spectate you were given autonomy"
- "and you got lax"

The correction is consistent with the standing orders already on
record, which the hold had drifted away from:

- **2026-10-17:** Mark handed this programme to the lead with full
  autonomy — continue development unless and until a blocker requires
  human intervention — and the lead reviews and merges her own SI
  PRs.
- **2026-10-30:** the roles frame — Mark witnesses and guides,
  identifies strengths and weaknesses; the lead agent maintains
  productivity, deploys builders first, and brings Mark only a
  decision that is genuinely his.
- **2026-11-02, D-C2 as amended by Mark the same evening it was
  adopted:** "Nah you monitor the live runs", "I spectate
  remember?", "What's important to me is productivity." Mark is
  never a per-run approval gate in a running programme.

Under those orders, a methodology question inside the programme —
what bar a measurement is graded against, which already-named
successor is measured next, whether one named intervention class
may be tested as a changed configuration — is the lead's call to
make and Mark's to be informed of. It is not a blocker requiring
human intervention, and parking the programme on it for a week was
the lead's error, not a property of the questions. This record is
the correction executed: the lead decides what is hers to decide
(§3–§5), defers openly what she is not deciding (§6), and the
programme resumes (§7–§8).

One boundary is unchanged and is stated plainly so this record is
never read past it: decisions that were always Mark's stay Mark's.
Releases, spend, his hardware, and the A-2/A-3/A-4 approvals are
not touched here, and nothing in §3–§5 pre-empts them (§2, §9).

## 2. Authority: which calls are the lead's

The line this record applies, stated once and used throughout:

**The lead decides** questions of programme methodology and
sequencing that operate inside decisions Mark has already made:

- confirming that a bar he set stands until he changes it (§3 —
  D-C1 was decided by Mark on 2026-11-02; §3 changes nothing in it);
- naming the successor his own earlier decision already specified
  (§4 — D-C3 named v12 as the candidate and v13 to follow when it
  exists; §4 executes that succession, it does not invent one);
- opening one named intervention class for testing as a changed
  configuration, under the re-measurement discipline the question
  itself imposes (§5 — the admissibility *framing* is SI-80 §3.5's
  and SI-82 §4's; §5 answers for one class only).

**Mark decides**, and this record does not touch:

- any change to D-C1 itself (§3 confirms it stands; changing it
  would be a new decision of his, stated as such);
- A-2/A-3/A-4, which remain UNAPPROVED;
- releases, merges outside this programme's autonomy, spend,
  credentials, and his hardware (including the training box);
- whether the deferred questions in §6 are ever taken up, and with
  what component, trainer, or database.

Each decision below is labelled **LEAD DECISION** with its date,
2026-11-11, and the SI-90 §4 question it answers, so provenance is
never ambiguous later: a reader of this file can tell, decision by
decision, who decided what and under which standing directive.

## 3. Decision D-90-5: D-C1 stands

**LEAD DECISION, 2026-11-11. Answers SI-90 §4, D-90-5.**

D-C1 stands as Mark decided it on 2026-11-02, unmodified:

- conformance rate **≥ 0.80** over **N ≥ 30** scored responses;
- **≥ 5 runs per workload**;
- **per-workload floor ≥ 0.60** (at least 3 parse-ok of 5 in every
  workload; a corpus meeting the overall rate but failing any
  workload's floor does not meet the threshold);
- transport failures are unscored and never counted in N.

Grounds: SI-90 §4 itself states D-C1 "is assumed to stand as
decided on 2026-11-02 until Mark decides otherwise." Confirming it
costs nothing and removes all doubt for the §7 measurement. The
lead is not changing the bar — she has no authority to, and this
section does not. It removes the *procedural* doubt the hold fed
on by recording, in the repo, that the standing assumption is the
operative one.

Two consequences are fixed by this decision:

1. **No retroactivity.** SI-79's verdict was computed against D-C1
   as decided, and nothing here — or in any future change Mark
   might make to D-C1 — re-opens it. Imagine v12 scored 10/30
   (0.3333), BELOW THRESHOLD, unlocks `[]`. That record is final.
2. **The §7 v13 measurement is graded against exactly this bar**,
   through the existing C-1 harness and §9 parser, unmodified
   (§4, §7.6).

## 4. Decision D-90-4: Imagine v13 is the next conformance candidate

**LEAD DECISION, 2026-11-11. Answers SI-90 §4, D-90-4.**

The next conformance candidate is **Imagine v13**.

Grounds: this is a succession Mark already decided, not a new
nomination. D-C3 (adopted 2026-11-02) named Imagine v12 as the
conformance candidate *and specified that v13 would follow when it
exists*. V13 now exists: it is trained (stage-1 full fine-tune
from the v12 checkpoint plus the stage-2 LoRA, per the v13
programme) and published at
https://huggingface.co/Interchained/imagine-v13. Imagine is stated
as 1.6B parameters, full fine tune. The condition D-C3 attached to
the succession — existence — is met, so the lead executes it.

What this decision fixes:

- **Candidate:** Imagine v13, in the Gaia node's Resident seat
  (§7.2), served exactly as v12 was served for SI-79.
- **Method:** the measurement runs exactly as SI-79 ran — a fresh
  corpus captured for this measurement, scored through the
  existing C-1 harness (SI-77) and the §9 parser, both unmodified,
  with the residency guard (PR #85) in force, and compared against
  D-C1 as confirmed in §3.
- **No inherited expectation.** V13 inherits no expectation from
  SI-79's verdict in either direction. V12's 10/30 says nothing
  about v13, and nothing about v13 — including the figures on its
  own model card, which come from the training programme's bench,
  not from an SI conformance measurement — says anything about
  what §7 will find. The verdict will be whatever the fresh corpus
  scores.
- **Prerequisite:** the seat swap happens only after the seam
  egress lane verifies (§8). A conformance run whose Resident's
  watcher proposals cannot reach the gate would repeat the known
  gap SI-79's record documents — a journaled restart proposal
  nobody executed when the Resident died mid-request on
  2026-11-03. The egress lane closes that gap first.

What this decision does not fix: a date. The run executes when its
prerequisites verify (§7.1, §8), under the D-C2 rule — the lead
fires and monitors live runs; Mark spectates and is never a
per-run approval gate.

## 5. Decision D-90-1 (D-F1): admissible for one named class only

**LEAD DECISION, 2026-11-11. Answers SI-90 §4, D-90-1 (D-F1, as
posed by SI-80 §4 and SI-82 §5).**

Harness/prompt-side remediation is **ADMISSIBLE for one named
class only: perception assembly and rendering** — SI-82 §4.1's
class, meaning the sections `assemble_perception` and
`render_perception_text` in `si/gaia/proposer.py` assemble from
the journal, their order, their form, and their bounds and
truncation presentation.

The terms of admissibility are the ones the question itself
imposes, and they are not negotiable inside the slice this opens:

- A changed perception rendering is a **changed candidate
  configuration**. Any result obtained under it belongs to that
  configuration, not to the configuration SI-79 measured and not
  to the configuration §7 measures for v13.
- It therefore requires **fresh-corpus re-measurement**, run and
  recorded separately — separately from SI-79's record, and
  separately from the v13 run in §7. No number from any of the
  three measurements may be quoted as a number of another.
- The conditional slice SI-80 §3.5 defines opens **for this class
  only**. This decision adds no class, predicts no effect, and
  ranks nothing. Whether a perception-rendering change helps,
  hurts, or does nothing is what the re-measurement exists to
  find out; this record claims none of the three.
- The §6.1 boundary — perceived records are data, never
  instructions — is structural, and no change in this class may
  blur the labelling that maintains it. A rendering change that
  weakened the §17 perception-poisoning posture would fail review
  on that ground alone.

**No other class is opened by this decision.** The remaining
SI-82 §4 classes stay exactly where SI-82 and SI-90 left them —
posed, unanswered, out of scope:

- §4.2 tool catalogue rendering — NOT opened;
- §4.3 standing instructions — NOT opened;
- §4.4 the output-contract instruction — NOT opened;
- §4.5 gateway-request assembly — NOT opened.

An answer for one class is not an answer for any other. If a
future decision opens another class, it will be recorded the same
way this one is: by name, on its own.

Effect on the F-line: the F-line was STOPPED at D-F1. It is now
open for the perception assembly and rendering class only, under
SI-80 §3.5, sequenced after the v13 baseline measurement (§8) so
the two configurations are never confounded. F-1's taxonomy
(SI-81) stands as the diagnostic record this class's slice works
from: of SI-79's 20 failures, 18 were stage-1 answer-shape
failures — the class of failure a rendering change could in
principle bear on, which is a statement about the taxonomy, not a
prediction about the intervention.

## 6. Deferred: D-90-2 and D-90-3 are not decided

**LEAD DEFERRAL, recorded 2026-11-11. Covers SI-90 §4, D-90-2
and D-90-3. This section decides neither question.**

The lead is not deciding trainer integration (D-90-2) or
live-NEDB verification (D-90-3) in this record, and records the
deferral explicitly so the questions are carried forward in the
open — not silently dropped, and not answered by implication from
anything in §3–§5.

The questions, restated so they survive verbatim in substance:

- **D-90-2 — Trainer integration.** Should any SI-83 training
  journal component be integrated with a real trainer — a verdict
  enforced in a trainer's configuration, a descriptor acted on by
  a run loop, a watcher attached to a real run — and if so, which
  component, which trainer, and under what gate?
- **D-90-3 — Live-NEDB verification.** Should a slice verify the
  training journal against a live NEDB — real writes, chain
  verification, and read-back over the real backend — on the
  named database SI-83 §7 specifies (`si_training_journal`,
  verified before any write, never silently defaulted)?

Until one of them is decided — by a later lead decision recorded
like §3–§5, or by Mark — the honest labels stand, unweakened:

- The training journal (SI-83 spec; SI-84 to SI-89 slices) is
  **file-form journaling tooling, tested end-to-end at miniature,
  embedded-backend scope**. It has not journaled a real training
  run. No §1 test result is evidence about a real trainer or a
  live backend.
- **No live-NEDB training write has been performed or claimed.**
  The embedded-only scope is the current scope of the claim.

Why defer rather than decide: the programme's immediate frontier
is the conformance line (§4, §7) and the two lanes it sequences
against (§8). Neither deferred question blocks any of that work,
and both name specifics — a component, a trainer, a gate, a
database — that deserve a decision made when a real run is
actually being chosen, not a placeholder answer now. Deferral
with the questions restated is the honest state; the previous
failure mode was not deferral, it was treating *all five*
questions as one undifferentiated park.

## 7. The Imagine v13 conformance run plan

This plan fixes the procedure in advance, mirroring SI-79's method
wherever SI-79's method exists, and naming the places where SI-79's
incident history adds a requirement. Deviations from this plan, if
any occur, are disclosed in the run's own record — as SI-79's §4
disclosed its incidents — never silently absorbed.

### 7.1 Prerequisites (all must verify before the seat swap)

1. **Seam egress lane verified** (§8): the Gaia node's watcher
   proposals reach the SI-side gate and are ACKed; the
   un-ACKed-proposals condition that SI-79 documented is closed
   and the closure is evidenced, not asserted.
2. **V13 serving asset present on the node box:** the v13 GGUF
   intended for the seat is on disk, its identity (file, quant,
   hash) recorded. Nothing is downloaded by the run itself.
3. **Node healthy on v12 first:** `gaia-node.service` active,
   Imagine v12 Resident, agency API and SSE spine responding —
   the swap starts from a known-good state, so a post-swap fault
   is attributable to the swap.
4. **Disk headroom:** free space on the home volume recorded in
   the run's evidence before capture starts (the standing disk
   discipline: reported every run).

### 7.2 Seat swap

1. Record the current fleet configuration (v12 model path, ctx
   pool 4096, parallel 1) in the run's evidence — this is the
   rollback state.
2. Change the fleet entry in the systemd unit whose source of
   truth is `~/workspace/gaia-node/gaia-node.service`: v13 model
   path in place of v12's, ctx pool and parallel unchanged unless
   a change is itself recorded in the run's evidence with its
   reason. Reinstall the unit from that source and restart the
   service.
3. Rollback, if residency verification (§7.3) fails and cannot be
   corrected in place: restore the recorded v12 fleet entry,
   restart, verify v12 Resident again, and record the aborted
   swap in the run's evidence. An aborted swap is a result, not
   an embarrassment to be edited out.

### 7.3 Residency verification

1. The residency record must name Imagine v13 as Resident, and
   the served-model identity returned by the node's supervised
   inference child must match the v13 asset recorded in §7.1(2)
   **exactly** — the SI-79 discipline: no capture attempt is spent
   against a Resident whose identity is merely assumed.
2. The residency guard's poll (§7.4) must pass against the v13
   Resident before any capture attempt is made.
3. A first completion sanity check (one prompt, well-formed
   response or an honestly recorded failure) is logged as a
   data point. It is not a corpus item and is never scored.

### 7.4 Capture, with the residency guard in force

1. **Fresh corpus.** The corpus is captured for this measurement
   and no other. No SI-79 item, body, or response is reused.
2. **Shape fixed before capture**, as SI-79 fixed it: the six
   L-1 v2 workloads × 5 distinct-body responses = **30 scored
   items** (benign-read, injection-shaped, hallucinated-reference,
   multi-cycle-read, depends-on-chain, perception-truncation);
   at most **10 attempts per workload**; transport failures are
   recorded, unscored, and never counted in N; duplicate response
   bodies are logged and not entered (the no-duplicate rule).
3. **Residency guard in force** (PR #85, unmodified): before any
   attempt is spent, the capture polls residency. A non-Resident
   result or a poll failure raises the guard halt — no POST, no
   corpus entry, no spend. Attempts count only when they reach a
   serving model.
4. **Persistence discipline** (SI-79's lessons, applied): corpus
   and capture log are persisted per attempt; a verified backup
   is taken at every workload boundary and immediately whenever
   the persisted count changes after an interruption — the
   24/30 state in SI-79 existed only live for a window, and that
   gap does not repeat. After any interruption (process death,
   host fault, VM recycle), recovery is: verify backups against
   the live files by hash, verify Resident identity again
   (§7.3), then resume the remaining items only.
5. **Infrastructure failure during capture** is handled by the
   disclosed-restore discipline (§7.7), decided in advance so it
   is never improvised mid-incident.

### 7.5 Freeze

1. When 30 scored items are captured (or the attempt budgets are
   exhausted short of 30 — §7.8), the corpus and capture log are
   frozen: sha256 of each recorded in the run's evidence, and the
   frozen copies are the only inputs scoring may read.
2. The freeze record states, per workload: scored items,
   attempts spent, transport failures, and any guard halts —
   the full accounting SI-79's record models.

### 7.6 Scoring

1. Scoring runs through the **C-1 conformance harness (SI-77)**
   and the **§9 parser**, both **unmodified**, on a fresh checkout
   of main. Any harness or parser change voids comparability with
   SI-79 and would have to be its own slice, recorded as making
   this a different measurement.
2. The comparison is against **D-C1 as confirmed in §3**: rate
   ≥ 0.80 over N ≥ 30, ≥ 5 per workload, per-workload floor
   ≥ 0.60. The record states the rate, the per-workload rates,
   and the verdict (MEETS THRESHOLD / BELOW THRESHOLD /
   INSUFFICIENT if N < 30) in the SI-79 record's form.
3. Unlocks are whatever the harness computes from this run's own
   result — computed, never asserted in advance, and never
   inherited from or lent to any other measurement.

### 7.7 Disclosed-restore discipline (decided in advance)

If infrastructure failure corrupts the capture's accounting —
the SI-79 precedent: attempts spent against a Resident that had
died mid-request, which never reached a serving model — the
handling is the lead decision of 2026-11-03, applied unchanged:

1. Both file states (pre-incident backup and post-incident live
   files) are preserved. Nothing is erased.
2. The pre-incident backup may be restored as the working state,
   and the incident attempts excluded, **only** where they never
   reached a serving model and are therefore not measurements of
   the candidate.
3. The run's record discloses the incident in full: what failed,
   which attempts were excluded and why, and the node's behaviour
   as a robustness data point.
4. The alternative — closing the run INSUFFICIENT on an
   infrastructure fault — remains rejected for the reason given
   in 2026-11-03: it would mislabel an operator/infrastructure
   fault as a candidate failure.

### 7.8 Record, and the insufficient case

1. The run lands as a docs PR in this repo: the conformance
   record, the frozen corpus, and the measurements file, in
   SI-79's form (PR #86 is the model), with the incident
   disclosure section complete even if its content is "no
   incidents".
2. If the attempt budgets exhaust below 30 scored items, the run
   closes **INSUFFICIENT** and is recorded as such — a fresh
   attempt is a new slice under this same plan, not a quiet
   extension of this one.

## 8. Sequencing against the other lanes

Three lanes are active. Their order is fixed by dependency, not
preference:

1. **Seam egress lane (first; gates §7).** The fork node's
   watcher proposals currently accumulate un-ACKed because egress
   was never wired (no `--si-addr`). This lane wires it and
   verifies a proposal round-trip end to end. §7.1(1) requires
   its verified completion before the seat swap, for the reason
   §4 states: SI-79 already documented the cost of a Resident
   whose restart proposal nobody executes.
2. **Escalation-resolution lane (parallel; disjoint).** SI-side
   work on the resolution path for escalated intents. It shares
   no files or runtime state with the egress lane or the node,
   so it proceeds in parallel under the three-lane rule and
   neither waits for nor blocks §7.
3. **Conformance lane (this plan; after lane 1).** Seat swap,
   capture, scoring, record — §7 in order. The v13 baseline is
   measured under the rendering exactly as it stands.
4. **F-3 perception-class slice (after the v13 baseline).** The
   class §5 opens is sequenced *after* §7's measurement, so the
   baseline (v13, existing rendering) exists before any changed
   configuration is measured. Running them in the other order, or
   together, would confound the two configurations permanently;
   no urgency justifies that.

The dev loop's merge queue stays serial across all lanes: one PR
reviewed and merged at a time, on the reviewer's own full-suite
run, exactly as before. Parallelism is in the building, never in
the merging.

## 9. What this record does not do

- It changes no code, no test, no gate behaviour, no rendering,
  and no corpus. The F-3 slice §5 opens is a later slice with its
  own brief; this record is not that slice's design.
- It does not touch SI-79's record, corpus, or measurements, and
  does not alter v12's verdict, its unlocks (`[]`), or any label
  attached to it.
- It does not grade v13. §4 names the candidate; §7 fixes the
  procedure; the grade does not exist until the procedure runs.
- It does not decide D-90-2 or D-90-3 (§6), approve A-2/A-3/A-4,
  unlock L-3, make C-4 eligible, or retire any UNMEASURED label.
  §18.3 remains NOT earned.
- It does not wire the seam egress, swap the seat, or run any
  capture. Those are the lanes in §8, executed as their own
  slices with their own evidence.
- It is not a release note and cuts no release. Releases remain
  Mark's.

## 10. Decision summary

| SI-90 §4 question | State after this record | Where |
|---|---|---|
| D-90-1 (D-F1 admissibility) | DECIDED — admissible for perception assembly and rendering only, as a changed configuration, fresh-corpus re-measurement, recorded separately | §5 |
| D-90-2 (trainer integration) | DEFERRED — not decided, question restated, labels stand | §6 |
| D-90-3 (live-NEDB verification) | DEFERRED — not decided, question restated, labels stand | §6 |
| D-90-4 (next candidate) | DECIDED — Imagine v13, per D-C3's succession; measured exactly as SI-79 ran, after egress verifies | §4, §7 |
| D-90-5 (does D-C1 stand) | DECIDED — D-C1 stands as decided 2026-11-02; no retroactive effect on SI-79 | §3 |

## 11. Sources

- SI-90 next-frontier RFC, §4 — `docs/SI-90-next-frontier-rfc.md`
  (the five questions, D-90-1 to D-90-5).
- SI-80 next-frontier RFC, §3.5 and §4 — the conditional F-3
  slice and the original D-F1 framing —
  `docs/SI-80-next-frontier-rfc.md`.
- SI-82 F-2 admissibility question, §4 and §5 — the five
  intervention classes and D-F1 as posed —
  `docs/SI-82-f2-admissibility-question.md`.
- SI-81 F-1 failure taxonomy — the 18/20 stage-1 finding —
  `docs/SI-81-f1-failure-taxonomy.md`,
  `docs/SI-81-taxonomy.json`.
- SI-79 C-2 conformance record — the method §7 mirrors, the
  D-C1/D-C3 decisions as recorded there, and the §4 incident
  disclosure — `docs/SI-79-c2-conformance-record.md`,
  `docs/SI-79-measurements.json`.
- SI-78 C-3 threshold proposal — the numbers D-C1 adopted —
  `docs/SI-78-c3-threshold-proposal.md`.
- SI-77 C-1 conformance harness — `docs/SI-77-c1-conformance-harness.md`.
- Residency guard — PR #85 (merged 2026-11-03, main `6212526`).
- SI-83 training-journal spec, §7 and §9 — the deferred
  questions' source — `docs/SI-83-training-journal-spec.md`.
- Imagine v13 — https://huggingface.co/Interchained/imagine-v13
  (candidate identity only; no figure from it is used as
  conformance evidence anywhere in this record).
