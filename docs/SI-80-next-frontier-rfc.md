# SI-80 RFC — The Next Frontier After the C-Line (Design Only)

> **Status: DESIGN ONLY.** Derived from merged evidence on `main` at
> `1855675fc6ed565033f58c0fcf11e093ab62c819` (PR #86 merge, SI-79).
> Changes no code, no test, no policy default, no gate behaviour.
> Nothing here is implemented, measured, captured, or run live.
> This RFC runs no live call, captures no corpus, and scores nothing.
> **Scope:** CPU only. No GPU, no cloud, no spend, no downloads. This
> is machine learning infrastructure throughout; the served model is
> a specialised machine learning model.
> **Relationship:** SI-76 defined the C-line (C-1 to C-5) in its §4.
> That line is complete except its conditional slices, whose
> conditions failed or were never met (§1). SI-76 defines no line
> after a BELOW THRESHOLD C-2 verdict, so this RFC states what the
> C-line earned and closed (§1), why the measured failures are the
> already-known failure class (§2), and what line should follow (§3).
> **Honesty labels carried forward verbatim, unweakened:**
> - §18.3 claim (SI-63): **NOT earned**
> - SI-63 §18.1 row P5-R1: **UNMEASURED**
> - SI-63 §18.1 row P5-R9: **UNMEASURED**
> - Live model-authored tool intent: **OCCURRED, scoped to SI-74 only**
>   (2026-11-02, P-5 live run only; 2 parsed intents, `fs_list` x2;
>   1 ADMIT + executed + verified, 1 ESCALATE + not executed). No
>   other run, and no sentence here, extends that scope.
>
> Where this RFC and an earlier slice doc disagree, the earlier doc's
> evidence wins until a merged slice updates it with new evidence.

## Table of contents

- [0. Honesty table](#0-honesty-table-state-after-si-79) - [1. What the C-line
  earned, and closed](#1-what-the-c-line-earned-and-closed) - [2. Root-cause
  continuity](#2-root-cause-continuity) - [3. The next line: the
  F-line](#3-the-next-line-the-f-line) - [4. Decision points for
  Mark](#4-decision-points-for-mark) - [5. Honesty rules for the
  F-line](#5-honesty-rules-for-the-f-line) - [6. Rejected
  alternatives](#6-rejected-alternatives) - [7. Non-goals](#7-non-goals) - [8.
  Sources and code references](#8-sources-and-code-references)

## 0. Honesty table: state after SI-79

No row changes. This RFC strengthens, weakens, and re-labels nothing.

| Element | State after SI-79 | Evidence / scope |
|---|---|---|
| C-1 conformance harness | DONE | SI-77 (PR #82/#83 line); harness check only |
| C-3 threshold proposal | DONE, adopted | SI-78; adopted as D-C1 by Mark, 2026-11-02 |
| C-2 conformance measurement | DONE, BELOW THRESHOLD | SI-79 (PR #86); §1.2 below |
| C-4 conditional L-2 retry | NOT ELIGIBLE | SI-76 §4.6; its conditions fail on SI-79 (§1.3) |
| C-5 conditional L-3 path | LOCKED | SI-76 §4.7; no real L-2 live success (§1.3) |
| L-3 residency-loss drill | LOCKED | SI-69 §4.5; SI-76 §3.6; nothing offline unlocks it |
| SI-63 §18.3 claim | NOT earned | SI-74, SI-75, SI-79 §0: no live task completed |
| Live model-authored intent | OCCURRED, scoped to SI-74 only | Header; SI-79 capture admitted/executed nothing |
| §18.1 rows P5-R2–R8, R10 | As promoted, scoped, by SI-74 | Cited by SI-77/SI-79, promoted by none of them |
| §18.1 row P5-R1 | UNMEASURED | SI-74 §4; SI-79 promotes none |
| §18.1 row P5-R9 | UNMEASURED | SI-74 §4; SI-79 §4 notes a Resident death as an observation only, promotes nothing |
| Readiness verdict (SI-79) | BELOW THRESHOLD | Against D-C1 as decided; §1.2 |
| Readiness unlocks (SI-79) | `[]` | `docs/SI-79-measurements.json`: `readiness.unlocks` |
| A-2 / A-3 / A-4 | UNAPPROVED | Not approved by SI-40, SI-63, SI-69, SI-76, or this RFC |

Consequences: (1) The C-line is complete as a measurement line: the harness
exists, the threshold is decided, and one named candidate has been measured —
and the measured answer is no, at the decided bar. (2) The only unconditional
next work the evidence supports is offline analysis of the corpus already in
the repo (§3, F-1). Everything beyond that is a question for Mark (§4), not a
plan this RFC decides.

## 1. What the C-line earned, and closed

### 1.1 C-1 and C-3 — harness and decided threshold

C-1 (SI-77) built the offline proposer-conformance harness: recorded responses
scored through the existing SI-63 §9 six-stage parser (`parse_proposal` in
`si/gaia/proposer.py`), folded into the record shape SI-76 §3.4 defines. C-1
measured the harness against known SI-74 responses; it measured no candidate.
C-3 (SI-78) proposed a threshold, and Mark decided it on 2026-11-02 as
**D-C1**: success rate ≥ 0.80 over N ≥ 30 scored runs, ≥ 5 runs per workload,
per-workload floor ≥ 0.60. D-C1 is the only threshold any later record may
compare against until Mark decides otherwise.

### 1.2 C-2 (SI-79) — the measured verdict

Docs `docs/SI-79-c2-conformance-record.md` and `docs/SI-79-measurements.json`;
corpus `docs/SI-79-corpus.json`. Candidate: Imagine v12, the current Resident,
named by Mark under D-C3 (2026-11-02). A new corpus was captured for this
measurement; SI-74 responses were not re-used (SI-76 §3.3(6)).

- **Headline:** 10 of 30 scored responses parse clean — rate 10/30 (0.3333).
  Verdict against D-C1: **BELOW THRESHOLD**. - **Corpus sufficient:** N = 30, 5
  per workload, validation `[]`. This is a measured below-threshold result, not
  an INSUFFICIENT CORPUS outcome. - **Comparison:** N and per-workload counts
  meet D-C1; the rate (0.3333 < 0.80) fails, and the per-workload floor fails —
  only multi-cycle-read meets 0.60. The threshold was not weakened, rounded, or
  re-cut to change this outcome, and this RFC does not re-open it (§5, §6). -
  **Failing-stage histogram:** stage 1 (envelope) **18**; stage 2 (schema) 0;
  stage 3 (registry) 0; stage 4 (references) **2**; stage 5 (bounds) 0; stage 6
  (signals) 0. - **Diagnostics (reported, never the metric):**
  `non_json_plain_text_bodies: 18` — the same 18 bodies that fail stage 1.
  Per-workload parse rates: benign-read 0.20, injection-shaped 0.20,
  hallucinated-reference 0.40, multi-cycle-read 0.60, depends-on-chain 0.40,
  perception-truncation 0.20. All 10 parse-ok items are single-intent,
  `intents_only`; no completion or inability signals were elicited (capture
  admits and executes nothing). - **Unscored:** 6 transport-failed items sit in
  the corpus file for provenance and are not in N. They are not failures of the
  candidate and no F-line slice may count them as such. - **Unlocks:**
  `readiness.unlocks` is `[]`. SI-79 is a readiness signal only: it earns no
  §18.1 row, no §18.3 claim, and authorises no live run.

### 1.3 C-4 and C-5 — conditions checked, neither opens

- **C-4 (SI-76 §4.6) is NOT eligible.** Its three conditions are conjunctive: a
  C-2 record for the candidate to be served (met — SI-79), its rate MEETS the
  D-C1 threshold (**fails** — BELOW THRESHOLD), and Mark's explicit go for that
  specific run (not given, and not sought by this RFC). One failed condition is
  enough; C-4 does not run. This RFC does not propose running C-4, under any
  re-cut of the numbers. - **C-5 (SI-76 §4.7) / L-3 stay LOCKED.** C-5 is
  conditional on a real L-2 live success (≥ 1 fixture task end-to-end, SI-69
  §4.4). None has happened — SI-74 completed 0 of 3 in both runs and no L-2
  retry has run since — so SI-69 §4.5 remains unsatisfied. - **D-C2 as amended
  does not change this.** Mark's 2026-11-02 amendment (live conformance runs
  fire on the loop's initiative; Mark spectates) covered conformance *capture*
  — the SI-79 corpus capture fired under it. It is not an L-2 authorisation:
  C-4's conditions are separate, and they are unmet.

## 2. Root-cause continuity

SI-75 §2.7 already established the root cause of the live failures: a
**model-output-content failure**. The served model is a specialised machine
learning model (natural language → SQL); its measured behaviour on this seat is
to return identity / plain text instead of the one SI-63 §8 JSON proposal
document — P-4: 3 of 3 responses stage-1 failures; P-5: 1 of 3 parsed responses
a stage-1 failure, its journaled excerpts single sentences of identity text. No
SI-63 §15.2 policy key governs what the model says, and SI-75's tuning list was
empty for exactly that reason.

SI-79's 18 plain-text stage-1 bodies are **the same failure class at N = 30
scale**: the diagnostic count of non-JSON plain-text bodies (18) is exactly the
stage-1 failure count (18), across all six workloads. This RFC cites SI-75 §2.7
and SI-79 together and invents no new cause — not a parser defect (§9 scored 10
bodies clean and the stage-2/3/5/6 counts are 0), not a policy cap (SI-75
§2.1–§2.7), not the 2 stage-4 reference failures, which are a distinct, much
smaller observation: 2 of the 10 bodies that reached stage 4 or beyond failed
there. What is *inside* the 18 and the 2 — how many distinct content shapes, in
what proportions — is not known from the histogram alone. That unknown, and
only that unknown, is what the F-line's first slice exists to close.

## 3. The next line: the F-line

### 3.1 Naming and justification

The **F-line**, for *failure-driven* work: each slice starts from the measured
SI-79 failures, not from a new hope. Not "C-6+": SI-76 closed the C-line's
unconditional slices at C-3 and gated C-4/C-5 on conditions this line does not
weaken. The F-line ends, if it ends anywhere, by *re-entering* the C-line's
gates under their original definitions (F-4/F-5); it does not redefine them.

### 3.2 Order and no-stall rule

Order: F-1, F-2, then conditionally F-3, F-4, F-5. F-1 is unconditional,
offline, CPU-only, and buildable on the next run with no new Mark decision. F-2
is a question, decided by Mark (D-F1, §4), not by this RFC. If the conditional
slices never activate, the loop does F-1, poses F-2, stops, and reports — it
does not invent replacement live work, a replacement candidate, or a
replacement threshold. Every slice is one PR, built by a builder, reviewed and
merged by the dev-loop driver on a later run; no slice merges itself. Every
slice doc updates §0 row by row with evidence or states explicitly that no row
changes. Acceptance below is journaled/system outcomes and reproducible offline
folds only, never demo impressions.

### 3.3 F-1 — Failure taxonomy over the existing SI-79 corpus

- **Condition:** none. Buildable next run; needs no new Mark decision, no live
  node, and no capture. - **Scope:** an offline classification of the bodies
  already in the repo, and nothing else. Inputs, all on `main`:
  `docs/SI-79-corpus.json` (the bodies, stored once),
  `docs/SI-79-measurements.json` (per-item stage outcomes and response hashes),
  and `docs/SI-79-c2-conformance-record.md`. No new capture, no live calls, no
  parser or harness change, no re-scoring: the SI-79 stage outcomes are taken
  as given and are inputs to the taxonomy, never re-derived as a new
  measurement (§5). - **What gets classified:** every scored body, partitioned
  exactly as SI-79 left it — the 18 stage-1 failures, the 2 stage-4 failures,
  and the 10 parse-ok bodies as the control group. The 6 unscored
  transport-failed items are listed as excluded, with the reason SI-79 gives,
  and are classified as nothing. - **Method — classes defined from the bodies,
  not before them:** this RFC deliberately pre-claims **no count per class**.
  The class set is defined when F-1 runs, from what the bodies actually show,
  under these constraints: 1. *Candidate observable shapes* (starting
  vocabulary only, to be kept, split, merged, or dropped on the evidence):
  plain prose; identity text (the SI-75 §2.7 shape); SQL-shaped output; partial
  JSON (begins as JSON, does not complete); fenced JSON (JSON inside a code
  fence, not the bare document §8 requires); empty or truncated body. A shape
  no body exhibits is reported with count 0 or dropped with the reason stated;
  a shape bodies exhibit that fits no candidate is added, defined, and named.
  2. *Stage-4 failures are classified separately*, by their observable
  reference failure as recorded in the SI-79 record (which reference, in which
  workload), under the same no-pre-claim rule — 2 bodies support a listing, not
  a rate. 3. *Mechanical and reproducible:* each class is a stated,
  deterministic predicate over the body text (and, for stage 4, the recorded
  parser outcome) — specific enough that re-running the stated rules over the
  same corpus file produces the identical classification. Classes are mutually
  exclusive per body under a stated precedence order, so counts cannot
  double-count. Any body no predicate decides is class `unclassified`, counted,
  and quoted by hash — never silently forced into a class. 4. *Referenced by
  hash, never copied:* every classified body is cited by its SI-79 response
  hash and corpus item id. Body text is not duplicated into the taxonomy record
  (standing design rule: never duplicate bytes); a reader resolves each hash
  back to the one stored body in `docs/SI-79-corpus.json`. - **Reporting shape
  (the deliverable's contract):** one machine-readable taxonomy record plus a
  slice doc, in the SI-77/SI-79 pattern, containing: fingerprint (SI commit,
  corpus id + hash, date America/New_York, `mode: offline`); the final class
  definitions with their predicates and precedence order; a per-class table
  (class, count, item ids + hashes) for stage-1 bodies, for stage-4 bodies, and
  for the parse-ok control group; the `unclassified` count (0 is a result, not
  an assumption); and the honesty block of §5 verbatim in substance. -
  **Acceptance:** the counts sum exactly to the SI-79 totals — stage-1 classes
  sum to 18, stage-4 classes sum to 2, control group sums to 10, total 30 — or
  the slice fails review; every one of the 30 scored bodies appears exactly
  once, by hash; re-running the stated rules reproduces the record; the full
  test suite is green on the reviewer's own run. - **Honesty gate:** the
  taxonomy **earns no claim and no unlock** — no §18.1 row, no §18.3 progress,
  no readiness verdict, no threshold implication. Any wording presenting a
  class count as a new conformance rate, a re-measurement, or evidence about
  any other candidate fails review. F-1 says what the known failures are made
  of; it changes nothing about how many there are.

### 3.4 F-2 — Is any non-model intervention admissible? (a question)

- **Condition:** F-1 merged. Nothing else — asking the question needs no
  decision; answering it is Mark's (D-F1). - **Scope:** a design-only slice
  that poses, and does not answer, one question: *is any SI-side, non-model
  intervention admissible to test at all* — for example a change to how the
  perception / prompt is rendered to the served model (the §6/§7 rendering:
  perception, catalogue, standing instructions), informed by F-1's taxonomy.
  The slice doc states the question, the concrete intervention classes it could
  cover, and what each would put at risk. It builds nothing, changes no
  rendering, and runs nothing. - **Honesty risks the slice must state, not
  soften:** 1. A changed rendering is a **changed candidate configuration**. A
  result under it is a result for (model + rendering), not for the model SI-79
  measured; the two are never conflated or averaged. 2. Nothing measured under
  a changed configuration is known until it is **re-measured under the C-1
  harness over a fresh corpus** — F-1's taxonomy and SI-79's corpus can
  motivate a question; they cannot score an answer. 3. **SI-76 §3.3(6) applies
  in full:** re-scoring old bodies is a harness check and nothing more. SI-79
  bodies were answers to the old rendering; they are not evidence about a new
  one. 4. The line SI-76 §6.4 drew still stands: SI-75 closed policy tuning on
  live evidence, and "readiness" work must not smuggle in prompt engineering as
  a decided plan. F-2 exists precisely because admissibility is undecided — the
  doc presents it as an open question with its risks, **not as a decided plan
  and with no predicted improvement, in any direction, anywhere**. -
  **Acceptance:** the doc poses D-F1 (§4) with each answer's consequence
  stated; it names no intervention as approved, scheduled, or expected to help;
  no code, rendering, or corpus changes ride along in the slice. - **Honesty
  gate:** any wording that presumes the answer — a roadmap that only makes
  sense if remediation is in scope, an effect estimate, a "next we will
  change…" — fails review.

### 3.5 F-3 — Conditional: one admitted intervention, as its own slice

- **Conditions, all three:** (a) F-2 merged; (b) Mark answers D-F1 *in scope*
  and names the specific intervention class; (c) the intervention is specified
  narrowly enough to be one PR. Missing any one, F-3 does not exist. -
  **Scope:** implement exactly the named intervention, labelled everywhere as a
  distinct candidate configuration (model + rendering version), with the SI-79
  configuration preserved and selectable, so no comparison is ever ambiguous
  about which configuration produced which body. - **Acceptance:** the change
  is isolated to the named class; existing conformance and proposer tests stay
  green on the reviewer's own run; the slice doc states plainly that the
  intervention is **unmeasured** — it earns no rate, no verdict, and no claim
  until F-4. - **Honesty gate:** shipping the intervention together with any
  measurement, rate, or improvement wording fails review. F-3 changes a
  configuration; only F-4 may measure one.

### 3.6 F-4 — Conditional: re-measurement under the unchanged apparatus

- **Conditions, any one trigger, plus the apparatus rule:** (a) Mark names a
  future candidate under a D-C3-like decision (D-F2), or (b) F-3 has landed a
  changed configuration Mark asks to have measured (D-F3). This RFC claims **no
  specific future model exists, is ready, or will pass** — the trigger is a
  naming by Mark, whenever and whether it happens. - **Scope:** a C-2-shaped
  measurement, run exactly as SI-79 ran: a **fresh corpus** captured for this
  measurement (N fixed before scoring, §3.3 coverage, bodies stored once),
  scored through the existing C-1 harness and §9 parser, unmodified, and
  compared against **D-C1 as decided** — the apparatus (harness, parser,
  threshold, record shape) is unchanged unless Mark separately decides
  otherwise. - **Acceptance:** as SI-76 §4.4 (C-2), unchanged — record on disk
  naming the candidate/configuration exactly as discovered, corpus id/hash,
  every counted outcome walkable item by item by hash; verdict stated in D-C1's
  own terms (MEETS THRESHOLD / BELOW THRESHOLD), or INSUFFICIENT CORPUS naming
  the missing requirement; readiness-signal-only wording; `unlocks` exactly
  what the C-line's rules give it and no more. - **Honesty gate:** re-using
  SI-79 (or any prior) bodies as this measurement, re-cutting D-C1, or merging
  the offline rate into any live N fails review (§5).

### 3.7 F-5 — The standing gates, restated (no new path)

F-5 is not a build slice; it is this RFC's binding restatement that the F-line
creates **no new route** to the gated work:

- **C-4** becomes eligible only via its original three conditions (SI-76 §4.6):
  a conformance record (SI-79, or an F-4 record) that MEETS D-C1, plus Mark's
  explicit go for that specific run (D-C2 / D-F3). An F-4 BELOW THRESHOLD
  record leaves C-4 exactly as ineligible as SI-79 left it. - **C-5 / L-3**
  unlock only on a real L-2 live success — ≥ 1 fixture task end-to-end (SI-69
  §4.4/§4.5) — an event no F-line slice is, simulates, or substitutes for. - If
  neither ever activates, the F-line's honest end state is: the failures are
  taxonomised (F-1), the remediation question is posed and answered or left
  open (F-2/D-F1), and the gates stand as they stood. That is a complete
  outcome, not a stall to be filled with invented work.

## 4. Decision points for Mark

Mark's decisions; this RFC decides none, and no F-slice may pre-empt them. All
are **OPEN** as of this RFC.

- **D-F1 — Is perception/prompt-side remediation in scope at all?** May the
  loop test SI-side, non-model interventions (e.g. perception/prompt rendering
  changes) as a route from the SI-79 result — accepting that a changed
  rendering is a changed candidate configuration requiring fresh-corpus
  re-measurement (F-3/F-4)? *If no:* the F-line ends at F-2's question,
  answered; F-1's taxonomy stands as the record. *If yes, for a named class:*
  F-3 opens for that class only. - **D-F2 — Whether and when to name a
  successor candidate.** If a future candidate is to be conformance-measured
  (F-4), which one, named under a D-C3-like decision, and when? Naming none
  parks F-4 without penalty; this RFC names none and asserts none exists, is
  ready, or will pass. - **D-F3 — Another live conformance capture spend, after
  any F-line intervention?** A fresh F-4 corpus means another minutes-scale
  live capture of the SI-79 kind (§1.2: the SI-79 capture spanned hours of
  wall-clock across relaunches). Is that spend wanted — for a changed
  configuration, a named successor, both, or neither — and does D-C2-as-amended
  (loop's initiative, Mark spectates) extend to it, or is this capture Mark's
  per-run go? Until answered, no F-4 capture fires. - **D-F4 — Does D-C1 stand
  as the F-line's threshold?** F-4 (§3.6) assumes D-C1 as decided (≥ 0.80 over
  N ≥ 30, ≥ 5/workload, floor ≥ 0.60) until Mark decides otherwise. Confirming
  it stands costs nothing and removes all doubt; changing it is a new decision,
  stated as such, and never applies retroactively to SI-79's verdict (§5).

## 5. Honesty rules for the F-line (binding on every F-slice)

- Offline analysis is **never live evidence**. F-1's taxonomy of recorded
  bodies, like C-1/C-2 conformance, is a readiness signal only: it earns no
  §18.1 row, no §18.3 claim, does not unlock L-3, and does not authorise a live
  run. - **No threshold weakening or re-cutting.** D-C1 is not re-proposed,
  rounded, subset, or re-based (per-workload, per-class, parse-ok-only, or any
  other cut) to change the SI-79 verdict. The verdict is BELOW THRESHOLD; it
  stands. - **No re-scoring old bodies as a new measurement.** F-1 classifies
  SI-79 bodies under their existing SI-79 outcomes; it publishes no new parse
  rate. Any future measurement (F-4) uses a fresh corpus (SI-76 §3.3(6)). - F-1
  class counts are **descriptive statistics of a known corpus**, never a
  candidate property beyond that corpus, never a prediction about any other
  model or configuration. - No candidate is claimed to exist, be available, or
  pass. With no named candidate and no fresh corpus, the honest F-4 result is
  *not run*, not a rate. - Terminology: **machine learning**, never "custom LLM
  software"; the served model is a specialised machine learning model. CPU
  only; no GPU, no cloud, no money. - **Never duplicate bytes:** bodies live
  once in `docs/SI-79-corpus.json`; every F-line record references them by
  hash. No model, GGUF, or tensor bytes are copied by any F-slice.

## 6. Rejected alternatives

- **6.1 Run C-4 / retry L-2 now.** Rejected: C-4's conditions fail on the SI-79
  verdict (§1.3). A live retry spends the minutes-scale seat cost (SI-74 §3)
  against a candidate measured at 0.3333 against a 0.80 bar, blind to any
  change. - **6.2 Re-cut or weaken D-C1 until SI-79 passes.** Rejected (§5): a
  threshold moved after the measurement it judges is not a threshold. Per-class
  or per-workload subsets of the same 30 bodies are the same move in smaller
  pieces. - **6.3 Re-score the SI-79 corpus and call it progress.** Rejected:
  same bodies, same parser, same outcome — a harness check at best (SI-76
  §3.3(6)), and F-1 is explicit that it is not even that: it classifies, it
  does not score. - **6.4 Weaken §8 or teach §9 to accept plain text.**
  Rejected (SI-76 §6.3): salvage-parsing prose would redefine conformance as
  whatever the model already emits. §8/§9 are the governance boundary; the
  F-line measures and describes distance from it, never moves it. - **6.5
  Decide remediation inside this RFC.** Rejected: admissibility of
  prompt/perception-side change is exactly the kind of scope decision §4
  reserves to Mark (D-F1), and SI-76 §6.4 forbids smuggling it in as
  "readiness". F-2 poses it; nothing here presumes its answer. - **6.6 Name or
  design around a successor model.** Rejected: none has been named, measured,
  or shown to exist (§5). Designing F-slices that only make sense for a
  specific future model would fabricate the premise F-4 waits on. - **6.7
  Approve/pre-design A-2/A-3/A-4 here.** Rejected (SI-76 §6.7): they remain
  separate approvals no merged document grants, and a docs-only RFC must not
  smuggle one.

## 7. Non-goals

- Design only: no code, tests, configuration, capture, or live call in this RFC
  — and no conformance rate, class count, or taxonomy result is published here
  either: §3.3 designs F-1's method and reporting shape; F-1 produces the
  numbers, from the bodies, when it runs. - No threshold decided, re-decided,
  or weakened here (§4, D-F4). D-C1 stands as Mark decided it on 2026-11-02. -
  No intervention decided, designed in detail, or predicted to help (§3.4). No
  candidate named as existing, available, or expected to pass (§3.6, D-F2). -
  C-4 stays NOT ELIGIBLE and C-5/L-3 stay LOCKED (§1.3, §3.7); nothing here is
  their approval, scheduling, or pre-design. A-2/A-3/A-4 remain UNAPPROVED, out
  of scope. - No node relaunch, fork-repo work, or Hearth-side change assigned.
  No change proposed to §8, §9, any §15.2 key, gate, registry, queue, or SSE
  spine. - No backlog beyond the F-line: unlisted slices need a successor RFC,
  not an implied extension.

## 8. Sources and code references

All on `main` at `1855675fc6ed565033f58c0fcf11e093ab62c819` unless a doc number
is given.

- `docs/SI-79-c2-conformance-record.md`, `docs/SI-79-measurements.json`,
  `docs/SI-79-corpus.json` — C-2 as measured: 10/30 (0.3333), BELOW THRESHOLD
  against D-C1, histogram 18/0/0/2/0/0, unlocks `[]`. -
  `docs/SI-78-c3-threshold-proposal.md` — C-3 proposal, adopted as D-C1 by
  Mark, 2026-11-02. - `docs/SI-77-c1-conformance-harness.md`,
  `docs/SI-77-measurements.json` — C-1 harness as built. -
  `docs/SI-76-next-frontier-rfc.md` — C-line: §3.3 corpus rules (incl. (6)
  no-reuse), §3.6 honesty rules, §4.4 C-2, §4.6 C-4 conditions, §4.7 C-5, §5
  D-C1–D-C4, §6.3/§6.4. - `docs/SI-75-l4-tuning-from-live-evidence.md` — §2.7
  root cause (model-output-content failure; identity text, not the §8
  document), §3 empty tuning list, §5 L-3 LOCKED. -
  `docs/SI-74-l2-live-rerun.md`, `docs/SI-74-measurements.json` — second L-2
  attempt: scoped OCCURRED label, latencies, R1/R9 stay UNMEASURED. -
  `docs/SI-69-next-frontier-rfc.md` — L-line: §4.4 L-2 (success = ≥1 task
  end-to-end), §4.5 L-3 condition. -
  `docs/SI-63-gaia-propose-hearth-proposer-rfc.md` — §8 contract, §9 parser,
  §15.2 keys, §18.1 rows, §18.3 claim. - `si/gaia/proposer.py` — §8 as code;
  `parse_proposal`, `PARSE_STAGES` (1 envelope, 2 schema, 3 registry, 4
  references, 5 bounds, 6 signals). - `si/gaia/conformance.py`,
  `si/gaia/conformance_capture.py` — C-1 harness and C-2 capture as built;
  unmodified here. - `docs/SI-40-a1-feasibility-measurement.md` — A-1 measured;
  A-2/A-3/A-4 unapproved.

End of RFC SI-80. The F-line starts at F-1 (§3.3) on the next dev-loop run
after this RFC is reviewed and merged; F-1 needs no §4 decision. Everything
past F-1 waits on the decision that gates it, stated in its section.
