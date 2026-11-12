# SI-106 — C2 Multi-Cycle Diagnostic: Design

## The honest headline

This is a **design only** for SI-96 §6.2's Option C2 — multi-cycle capture —
commissioned by lead decision **D-96-4** as a diagnostic, verdict-free
measurement. It designs the one measurement under which journal-seq reference
formation, the skill §9.3 actually specifies, is priced at all. It changes no
code, runs no model, captures nothing, scores nothing, and claims no result.

The measurement designed here is **diagnostic only, by pre-stated absence of a
bar**. D-C1 (rate ≥ 0.80 over N ≥ 30, per-workload floor ≥ 0.60) belongs to the
cycle-0 conformance measurement and **does not transfer to this design and is
not applied by it**. There is no verdict language for a C2 record, and no
unlock of any kind can be earned by one (§3).

Three graded v13 configurations stand, unchanged, as the context this design is
sequenced after: SI-94, v1 teaching, plain rendering, **20/30, BELOW
THRESHOLD**; SI-100, v2 teaching, **9/30, BELOW THRESHOLD**, that teaching
**withdrawn by SI-102** while the SI-98 parser amendment stays in force;
SI-105, v1 teaching plus A1 rendering, **18/30, BELOW THRESHOLD**. Lead
synthesis, 2026-11-12: the F-line harness investigation for v13 is
**complete**; C2 is the last open measurement design; no new harness classes
are opened for v13 by this document.

## Table of contents

- [0. Honesty table — no row changes](#0-honesty-table--no-row-changes)
- [1. Fingerprint](#1-fingerprint)
- [2. Authority and sequencing](#2-authority-and-sequencing)
- [3. Verdict-free by construction](#3-verdict-free-by-construction)
- [4. The item definition — derived, not assumed](#4-the-item-definition--derived-not-assumed)
- [5. Fold-supplied scoring — the first fold-supplied corpus](#5-fold-supplied-scoring--the-first-fold-supplied-corpus)
- [6. What this measurement prices — both reference forms](#6-what-this-measurement-prices--both-reference-forms)
- [7. What this measurement still does not price](#7-what-this-measurement-still-does-not-price)
- [8. Candidate configuration — one, named exactly](#8-candidate-configuration--one-named-exactly)
- [9. Capture disciplines, re-stated in advance](#9-capture-disciplines-re-stated-in-advance)
- [10. Comparability — none, in either direction](#10-comparability--none-in-either-direction)
- [11. The record a C2 run would make](#11-the-record-a-c2-run-would-make)
- [12. Future slices — scope sketch](#12-future-slices--scope-sketch)
- [13. Risks and open points](#13-risks-and-open-points)
- [14. What this design does not do](#14-what-this-design-does-not-do)

## 0. Honesty table — no row changes

| Row | State, unchanged by this slice |
|---|---|
| SI-63 §18.3 claim | **NOT earned** |
| SI-63 §18.1 rows P5-R1, P5-R9 | **UNMEASURED** |
| Live model-authored tool intent | **OCCURRED, scoped to SI-74 only** |
| SI-94 verdict (v13, v1 teaching) | **BELOW THRESHOLD — 20/30 = 0.6667** |
| SI-100 verdict (v13, v2 teaching) | **BELOW THRESHOLD — 9/30 = 0.30** |
| SI-105 verdict (v13, v1 + A1) | **BELOW THRESHOLD — 18/30 = 0.60** |
| SI-99 v2 teaching | **WITHDRAWN by SI-102**; SI-98 parser amendment **stays** |
| C-4 for v13 | **Does not proceed** on the SI-94, SI-100, or SI-105 record |
| L-3 residency-loss drill | **LOCKED** |
| A-2 / A-3 / A-4 | **UNAPPROVED** |
| Readiness unlocks | `[]` |
| D-C1 | **Stands** for cycle-0 conformance (SI-91 §3); **not applied here** |

The candidate is a specialised machine learning model. This slice is machine
learning infrastructure throughout: one design document, written offline on CPU
only, that ran no parser, harness, capture, or live call in its production.

## 1. Fingerprint

| Field | Value |
|---|---|
| SI commit | `d6ddd3054fbc297d932b358d7e4188a1c58f0955` (main, SI-105 / PR #113 merged) |
| Inputs | `docs/SI-96-reference-formation-rfc.md` (§6.1, §6.2, §6.4, §7, §8, §9 D-96-4), `docs/SI-91-decision-record-and-v13-conformance-plan.md` (D-C1 §3, §7.6, INSUFFICIENT discipline §7.8), `docs/SI-97-intra-document-dependency-spec.md` (§8 v2 sibling form, §9.3, §12), `docs/SI-103-f3-perception-class-design.md` (structure and tone), `docs/SI-105-v13-a1-conformance-record.md`, `docs/SI-94-c2-conformance-record.md`, `docs/SI-100-v13-v2-conformance-record.md` — all on main, read only |
| Candidate under discussion | Imagine v13 — discussed, not measured |
| Date | 2026-11-12 (America/New_York) |
| Mode | offline / design only — one document, no code, no tests |

## 2. Authority and sequencing

- **D-96-4 (lead decision, 2026-11-12; SI-96 §9):** a C2
multi-cycle capture design is commissioned **as a diagnostic, verdict-free
measurement** — the first of the two admissible branches in SI-96 §6.2. The
other branch (the lead separately re-stating a bar for a multi-cycle design, in
advance) was **not** taken, and this design does not take it either (§3).
- **Sequencing after SI-97 (D-96-4's own order):** C2 was
sequenced **after** the §8 v2 intra-document spec precisely so that this design
prices **both** reference forms — the cross-cycle journal-seq form (§9.3) and
the intra-document sibling form (SI-97) — instead of designing a measurement
the sibling amendment would immediately obsolete (§6).
- **D-96-3 (scoring standard):** fold-supplied scoring is the
standard for future corpora, with one sequencing guard — the B-line re-
measurement (SI-100) kept no-fold scoring so the parser amendment was the only
thing that moved. That guard is now spent: the B record exists. This design is
therefore the **first fold-supplied corpus design** under D-96-3 (§5).
- **SI-91 disciplines carried, not re-decided:** the §7.6 INSUFFICIENT
discipline (a run that cannot reach its stated N closes as such, never quietly
extended) is carried into §9.7 in diagnostic form; the one-configuration-at-a-
time rule (SI-91 §5, SI-96 §7) is carried into §8.
- **No new harness class:** the F-line investigation for v13 is complete
(lead synthesis, 2026-11-12). This design opens no SI-82 §4 class, rewords no
instruction, and renders nothing new: the perception it exercises is the live
A1 rendering (SI-104), at the cycle positions SI-105 could not reach (§8).

## 3. Verdict-free by construction

Stated here, in writing, before any capture — as SI-96 §6.2 and §6.4 require of
a C2 design:

1. **There is no bar.** No rate, floor, or count in this design
is a threshold. Numbers a C2 run reports are descriptive statistics of a
diagnostic corpus, full stop.
2. **D-C1 does not transfer.** Its N, its per-workload floors,
and its denominator were derived for cycle-0, no-fold, single-response items
(SI-91 §3). §4 derives this measurement's item and denominator from its own
definition; nothing is borrowed from D-C1's arithmetic, and applying "≥ 0.80
over N ≥ 30" to a multi-cycle rate by assumption would be exactly the quiet
bar-moving SI-96 §6.4 forbids.
3. **No verdict language.** A C2 record states counts, rates,
and failure classes. It never states MEETS THRESHOLD, BELOW THRESHOLD, PASS, or
FAIL, and it computes **no unlocks** — unlocks stay `[]` whatever the numbers
are.
4. **No retroactive grading.** Nothing in a C2 record re-grades,
adjusts, or re-opens SI-79, SI-94, SI-100, or SI-105. Those verdicts stand as
recorded (§10).
5. **A future bar is a separate act.** If a multi-cycle bar is
ever wanted, the lead re-states it in advance, in writing, for a future corpus
— never for a corpus already captured.

## 4. The item definition — derived, not assumed

### 4.1 What one item is

One **item** is one **task run**: a single task, proposed over several proposal
cycles, under one task identity, in one capture fixture. Concretely, a task run
consists of:

- **Cycle 0:** the candidate proposes a batch for the task
against an empty `prior_cycles`. Cycle 0 is structurally the
SI-94/SI-100/SI-105 shape and is **not a scored cycle** of this measurement
(§4.3) — it exists to seed the journal.
- **Journaling between cycles:** the cycle's admitted intents
are journaled — parsed, gate-priced, admitted, and written to the capture
journal — **without execution** (§9.5). Journaling suffices for parse pricing
(SI-96 §6.2).
- **Cycles 1..K:** the candidate re-proposes for the same task
with perception rebuilt from the journal fold, so cycle *k*+1's perception
contains **real prior intents with real seqs** —
  `prior_cycles` **non-empty**, and, under the live A1 rendering,
a non-empty `citable_reference_set`. K is fixed per workload in the capture
plan, before capture (§9.3); this design's reference shape is K = 2, i.e. three
proposal cycles per task run, of which two are scored cycles.

A task run ends when its K continuation cycles are captured, when its attempt
budget is exhausted (§9.3), or when the residency guard halts the capture
(§9.2) — each ending recorded.

### 4.2 What N means, and the denominator

Because an item is a task run and not a response, the cycle-0 meaning of N does
not carry over. This design states both quantities a C2 record must report, and
which is which:

- **N — the item count:** the number of task runs completed to
the corpus freeze under §9. The capture plan states its target N in advance
(§9.7); the reference shape is six workload families × 5 task runs = **N = 30
task runs** — a shape chosen for workload balance, **not** inherited from D-C1,
and carrying no threshold meaning (§3).
- **M — the rate denominator:** the number of **scored cycle
documents** across all frozen items: the cycle-1..K proposal documents whose
`prior_cycles` is non-empty. Under the reference shape, M = N × K = 60 at full
capture. Every rate this measurement reports is a rate **over M**, stated with
M beside it, and broken down per workload and per cycle position. A rate over N
(task runs with all scored cycles parse-ok) may also be reported, labelled as
the strict per-item rate, with N beside it. The two are never conflated, and
neither is ever quoted without its denominator.
- **Unscored, never in any denominator:** cycle-0 documents,
transport failures, duplicate bodies (logged, not entered), and attempts spent
against a halted residency (§9.2) — the SI-91 §7 accounting, applied to the new
item shape.

### 4.3 Why cycle 0 is not scored here

Cycle 0 of a C2 task run is a cycle-0 measurement, which already has a
standing measurement, a standing bar, and three recorded v13 results. Scoring
it again inside C2 would lend a number between the measurements (§10). C2's
scored cycles are exactly those no prior measurement could price.

## 5. Fold-supplied scoring — the first fold-supplied corpus

This is the first corpus designed under D-96-3's fold-supplied standard (SI-96
§6.1), taking effect for corpora captured after the B record (SI-100). SI-94,
SI-100, and SI-105 were scored no-fold and are **not** re-scored under this
section, ever — SI-96 §6.4's rule, restated for this design.

### 5.1 The procedure, specified

1. **The fold is the capture's own journal.** At corpus freeze
(§9.6), the capture journal is frozen with the corpus. The fold supplied to
scoring is that frozen journal and no other: no bundle reuse across captures,
no rows from any other run, no live journal read at scoring time.
2. **The fold is hashed into the corpus record.** The corpus
record carries the frozen fold's sha256 beside the corpus sha256, computed over
the fold's canonical bytes exactly as frozen. A scoring run whose fold hash
does not match the record's is not a scoring of this corpus.
3. **Fold provenance is a review surface.** The corpus record
states, per item, the fold's provenance for that item: the task identity, the
journaled cycle-0..k−1 intent seqs the item's scored cycles could cite, and the
fact that every fold row originated in this capture's own journaling step
(§9.5). A reviewer must be able to see, from the record alone, that no
reference could pass against a target this capture did not itself journal
(SI-96 §6.1's stated cost, accepted and priced into the record's shape).
4. **The task_seq mapping (SI-96 §6.1).** The corpus names each
item's task seq at `prompt_ref.task_seq`; the harness reads it through its
`parse_task_seq` procedure. The mapping is therefore specified as procedure,
not assumed: for each scored cycle document, scoring (a) reads the owning
item's
   `prompt_ref.task_seq` from the frozen corpus record,
(b) passes it as `task_seq` together with the frozen fold as `journal` to the
scoring call, and (c) records both values' hashes in the scoring record. A
scored document whose item carries no `prompt_ref.task_seq` is a corpus defect:
it is recorded as such and excluded from M, never silently scored task-less.
5. **What the fold changes, honestly.** Fold-supplied pricing
matches how the live path prices references: a citation of a real prior intent
of this task can now **pass** stage 4 instead of failing closed. That is the
point of C2 — and it is also why §3 exists: a rate produced with targets
present and a fold supplied is a different quantity from every no-fold cycle-0
rate, and no bar written for the latter says anything about the former.

## 6. What this measurement prices — both reference forms

A scored cycle document's `depends_on` entries are priced by form, and the
record splits every reference outcome by form. (Sibling entries are priced
document-locally at parse, SI-97 §5; scoring them consults no fold — §5's fold
exists for the seq form, and the record states per reference which path priced
it.)

1. **Cross-cycle journal-seq references (§9.3).** Bare-int and
   `{seq, requirement}` entries naming a journal seq are priced
against the frozen fold (§5): the seq exists in the fold, is an intent **of
this task** (per §5.1 step 4's mapping), and carries a valid requirement. This
prices the formation skill §9.3 specifies: citing the right seq, of the right
task, with a valid requirement, **when targets exist** — including refraining
correctly is *not* what is priced here; scored cycles all have non-empty
citable sets by construction.
2. **Intra-document §8 v2 sibling references (SI-97).** A
   `{sibling: k, requirement}` entry inside one cycle's batch
is priced document-locally under the SI-98 parser amendment that stays in
force: dict-only, k a strictly earlier position in the same document,
requirement in {`outcome`, `success`}. Mixed forms in one intent are allowed
and each entry keeps its own form's pricing (SI-97 §9.3). Sequencing C2 after
SI-97 (§2) is what makes this second pricing line exist at all.
3. **The teaching boundary, stated plainly.** The candidate
configuration (§8) carries the **v1 teaching** (SI-102's state): the sibling
form is **not taught**, and this design does not revive its teaching — doing so
would move a second variable (§8) and re-open what SI-102 settled. Sibling
entries are therefore priced **when they appear**, and their occurrence count
is itself a reported diagnostic quantity. A C2 record in which sibling entries
never appear reports exactly that — occurrence 0, formation unobserved — and
never presents the seq-form rate as a sibling-form rate. Pricing taught sibling
formation is a different configuration and a different decision (SI-97 §12.3).
4. **All other stages unchanged.** Stages 1–3, 5, and 6 price
every scored document exactly as the unmodified parser prices them; a document
that fails before stage 4 is recorded with its stage, and its references are
unpriced, never estimated. The record's stage histogram is stated whole, as in
SI-94/SI-100/SI-105.

## 7. What this measurement still does not price

Stated in advance, so no later reading can stretch the record:

| Quantity | State under this design |
|---|---|
| Execution correctness of any intent | **NOT priced** — nothing executes (§9.5) |
| §18.1 rows P5-R1, P5-R9 | **UNMEASURED**, unchanged |
| Dependency-ordered execution (queue/drain behaviour) | **NOT priced** — journaling is not draining |
| Gate behaviour beyond admission pricing at journaling | **NOT priced** as a quantity |
| Resolution correctness in a live walk (SI-97 §6) | **NOT priced** — no live walk runs |
| Correct refraining at cycle 0 (empty citable set) | **NOT priced here** — that is SI-105's quantity |
| §18.3 claim | **NOT earned** by any C2 outcome, by §3 |

## 8. Candidate configuration — one, named exactly

Exactly one configuration, per the one-configuration-at-a-time discipline
(SI-91 §5; SI-96 §7). A C2 record names it in full:

- **Candidate:** Imagine v13, unmodified GGUF, served with its
**corrected trained template at serving** — the SI-94 §1 serving correction
(the template v13 was trained expecting, supplied at serving, per SI-93's
diagnosis).
- **Teaching:** the **v1 teaching**, exactly as restored by
SI-102. No v2 teaching, no contract-instruction change (§6.3).
- **Rendering:** **A1 is IN** — the `citable_reference_set`
rendering as implemented by SI-104. Reason, stated once: A1 is the live
configuration SI-105 measured, and C2 is the first design that exercises A1
where it has content — at cycles k ≥ 1 the citable set it states as data is
**non-empty**. Removing A1 for C2 would move a second variable and price a
rendering no live configuration runs. SI-105 priced A1's empty-set statement at
cycle 0; C2 prices its non-empty statement. Neither number is lent to the other
(§10).
- **Everything else held at SI-105's configuration:** same seat
discipline, same serving shape, residency guard in force (§9.2), no execution
(§9.5). The perception-digest difference at k ≥ 1 (real `prior_cycles`, non-
empty citable set) is part of this measurement's definition, recorded per item.

Any deviation — a different candidate, A1 out, a teaching change, execution
enabled — is a different measurement, requiring its own design. It is not a
variant of this one.

## 9. Capture disciplines, re-stated in advance

C2 capture stops being propose-only (SI-96 §6.2's stated cost): it journals
admissions between cycles and borrows the L-line's machinery. It must not
borrow its claims. The disciplines of a live-adjacent run are therefore re-
stated here, in advance; a capture slice does not improvise them.

### 9.1 Authorisation

This design authorises **nothing**. A C2 capture requires its own lead
authorisation naming this design, the candidate configuration of §8, and the
attempt budgets of §9.3, recorded before the smoke gate (§9.4) runs. No
capture, scoring, or live-adjacent step is in this slice or implied by it.

### 9.2 Residency guard

The residency guard (the SI-94/SI-105 discipline) polls residency before
**every** attempt — every cycle of every task run, not only cycle 0 — and halts
the whole capture, spending no attempt, if the candidate is not the resident. A
guard halt is a capture ending (§4.1), recorded with its position; it is never
retried around, and the guard is never disabled to finish a corpus.

### 9.3 Attempt budgets

Fixed in the capture plan before capture, and never extended mid-run: a per-
task-run attempt budget per cycle position (reference shape: at most 10
attempts per workload per cycle position, on the SI-94 pattern, restated for
the multi-cycle shape), a per-request timeout, and the target shape of §4.2
(six workload families × 5 task runs × K = 2 scored cycles). Transport failures
are recorded, unscored, and never counted in N or M; duplicate bodies are
logged, not entered — at every cycle, not only cycle 0.

### 9.4 Smoke gate

Before any attempt budget is spent, a smoke gate on the SI-94/SI-105 pattern:
one proposer-shaped request proving the served model is the exact candidate
under §8's configuration, **plus** one complete dummy task run in fixture scope
proving the multi-cycle machinery end to end — journaling between cycles,
perception rebuilt with non-empty `prior_cycles`, and the §5.1 fold assembled
from the fixture journal. A capture whose smoke gate has not passed spends no
budget.

### 9.5 Journaling between cycles — without execution

Between cycles, each cycle's batch is parsed, gate-priced, and its admitted
intents **journaled** to the capture journal — the journaling that makes the
next cycle's fold real. **No intent is executed**: the queue is not drained, no
tool runs, no fixture state changes by a tool's hand. Journaling suffices for
parse pricing (SI-96 §6.2); executing in fixture scope is a further step with
its own discipline, **not in this design** and not required to price
references. A capture that executes anything has left this design, and its
record must say so.

### 9.6 Driver and freeze

The capture driver runs as a systemd service on the SI-94/SI-100/SI-105 pattern
(a `si<nn>-capture.service` unit,
`Restart=on-failure` with the residency-guard exit status in
`RestartPreventExitStatus`, so a guard halt is never restarted
around). At completion — target reached or budgets exhausted — the corpus **and
the capture journal (the fold, §5.1)** are frozen together: sha256 of each
recorded in the run's evidence, and the frozen copies are the only inputs
scoring may read.

### 9.7 Closure below the stated N — the INSUFFICIENT analogue

If the attempt budgets exhaust below the capture plan's stated target N (§4.2),
the run closes **INCOMPLETE-DIAGNOSTIC** and is recorded as such, in SI-91
§7.6/§7.8's INSUFFICIENT spirit: the frozen partial corpus is preserved, its
counts are stated whole, no headline rate over M is quoted from it, and a fresh
attempt is a new slice under this same design — never a quiet extension of the
closed one. The analogue label is used because §3 forbids verdict language:
INCOMPLETE-DIAGNOSTIC describes the corpus, never the candidate.

## 10. Comparability — none, in either direction

A C2 record is a **separate record of a different measurement**. Its item is a
task run, its scored documents are post-initial cycles, its scoring is fold-
supplied, and it has no bar. Its rates are therefore **not comparable** with
the cycle-0 rates of SI-79 (v12), SI-94 (20/30), SI-100 (9/30), or SI-105
(18/30): no number is lent in either direction, and no C2 rate is quoted as a
successor to, or an improvement on, any of them. The three v13 records stand
exactly as graded under D-C1; this measurement stands beside them, ungraded.
## 11. The record a C2 run would make

A future capture/scoring slice under this design lands one record stating,
whole: the configuration (§8, in full); the freeze hashes (corpus **and** fold,
§5.1); N and M with their per-workload and per-cycle-position breakdowns; per-
form reference outcomes (seq-form priced against the fold, sibling-form priced
document-locally, occurrence counts for each, §6); the stage histogram over
scored documents; the strict per-item rate over N, labelled as such (§4.2);
guard halts, transport failures, duplicates, and budget accounting (§9); and
the §3 statement — diagnostic only, no bar applied, no verdict, unlocks `[]` —
printed in the record itself, not only here. Corpus and fold are referenced by
hash; no corpus body is copied into the record (never duplicate bytes).

## 12. Future slices — scope sketch

Not this slice. In order, each its own slice and review: (1) a multi-cycle
capture driver implementing §9, with replay/fixture evidence only, no live
capture; (2) fold-supplied scoring support implementing §5.1's procedure
(frozen fold, `prompt_ref.task_seq` / `parse_task_seq` mapping, fold hash in
the record), with tests pinning fail-closed behaviour on a fold-hash mismatch
and on a missing `task_seq`; (3) only then, under §9.1's separate
authorisation, a live capture and its §11 record. A diff in slice (1) or (2)
that touches gate pricing, parser stage rules, or teaching text fails review.

## 13. Risks and open points

- **Fold provenance is the new review surface** (SI-96 §6.1):
a fold containing rows this capture did not journal would silently manufacture
passing references. §5.1 steps 1–3 are the mitigation; slice (2)'s tests must
prove a foreign seq fails exactly as a nonexistent one does.
- **Continuation shape:** the cycle k+1 re-proposal is a fixed, harness-
generated continuation of the same task — not new model-facing instruction
text. If the driver (§12) cannot produce one without new instruction text,
that is a §4.4-class question for the lead, surfaced before capture.
- **Journaling without execution** leaves fold rows whose outcomes never
occur; `success` requirements against them are priced at parse as §9.3 prices
them — existence and task membership (§7 records the rest as unpriced).
- **Attrition across cycles:** later cycles exist only for task runs whose
earlier cycles journaled admissions; the per-cycle-position breakdown (§11)
exists so a thinning M at k = 2 is visible rather than averaged away.

## 14. What this design does not do

No code, no tests, no capture, no scoring run, no live model contact, no
execution design, and no conformance claim. It does not apply, re-state, move,
or borrow D-C1; does not re-grade SI-79, SI-94, SI-100, or SI-105; does not
revive the withdrawn v2 teaching; does not open any harness class; and does not
authorise the capture it designs. It fixes, in advance and in writing: the
item, the denominator, the absence of a bar, the fold-supplied scoring
procedure, both priced reference forms, the unpriced remainder, the one
candidate configuration, and the capture disciplines — so a C2 run, if
authorised, is the execution of decisions already taken, not a second design.
## Honesty labels carried forward

§18.3 **NOT earned**; P5-R1 / P5-R9 **UNMEASURED**; live model-authored intent
**OCCURRED, scoped to SI-74 only**; L-3 **LOCKED**; C-4 **does not proceed for
v13**; A-2 / A-3 / A-4 **UNAPPROVED**; unlocks `[]`; SI-94 verdict **BELOW
THRESHOLD 20/30**; SI-100 verdict **BELOW THRESHOLD 9/30**, v2 teaching
**withdrawn by SI-102**, parser amendment stays; SI-105 verdict **BELOW
THRESHOLD 18/30**; D-C1 **stands** for cycle-0 conformance and is **not
applied** by this design.
