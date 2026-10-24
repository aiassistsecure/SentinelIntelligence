# SI-32 RFC — The Next Frontier After Train-Partition Projection
(Design Only)

> **Status: DESIGN ONLY.** This document evaluates candidate
> directions and recommends one next slice. It changes no
> production code, no test, no manifest schema, no policy
> default, and no gate behaviour. Nothing in this RFC is
> implemented unless a section explicitly says it exists today,
> in code, on `main`.
>
> **The one sentence that governs everything below:** SI-31
> proved that composition measured on the inputs training is
> allowed to see can be made *exactly* zero on that population —
> as a construction, not only as a penalty — and that the gate's
> verdict still did not move in the direction the construction
> aimed: the projected candidate was REJECTED on two independent
> lines, the unchanged safety suite (constructed rank 16 >
> `rank_cap` 8) and cumulative identity drift
> 0.4775387533009052 > tol 0.25 at `tiny-id-0`, the worst
> composed maximum of the six-run record, with its probe inner
> product at `tiny-id-0` at +0.14235826783471692, worse than
> SI-29's +0.04944786620153699, SI-27's +0.03801222744105431,
> and SI-25's +0.0338873784299374 — and this RFC must never be
> cited as evidence that any rank policy has been changed or
> argued away, that any partial-removal candidate has been
> constructed or measured, that any non-readout target has been
> trained, that a cycle-2 ADMIT is achievable by any route, or
> that any slice proposed here exists.
>
> **Scope discipline:** CPU only. No GPU, no cloud, no spend,
> no downloads. Terminology: this is machine learning
> infrastructure throughout.
>
> **Relationship to prior work:** RFC SI-30 evaluated the
> frontiers open after SI-29 and recommended train-partition
> projection; SI-31 built it (PR #36, as-built in
> `docs/SI-31-train-partition-projection.md`) and the gate's
> cycle-2 verdict was a measured REJECT on two lines, with
> RFC SI-30 §7.5's falsification reading F1′′ *measured* and
> F3′′ *measured*, and F2′′ *not measured as a reading* (its
> transfer clause fails; its maximum clause's fact holds
> beside it). RFC SI-30 §7.5 stated the consequence in
> advance, and this RFC takes it as its starting point, not
> as a conclusion it re-derives: *"with the mean, the worst
> case, and the projection all measured failing at the binding
> probe, no composition move that reads only allowed inputs
> remains on this RFC's record, and the line's honest
> continuation is a later RFC's genuinely new argument — not
> a fourth variation."* This RFC is that later RFC. It
> restates the verified state after SI-31 (§1), gives the
> composition line's completed attribution the deep treatment
> (§5), argues the two questions SI-31 explicitly left open —
> the representation/rank question (§7) and partial-removal
> projection (§8) — on their own merits rather than reporting
> them, evaluates whether the composition line should close
> in favour of a different frontier entirely (§§4, 12), and
> recommends exactly one next slice (§15): the first
> non-readout authored target, which is not a composition
> move at all. SI-30's rejections of per-probe anchoring,
> probe-partitioned sequencing, and magnitude discipline as
> a training dial, and of Glimmer probing, are carried
> forward, not reopened: §§9, 10, 11, and 13 record, per
> candidate, that SI-31 supplies no new evidence that
> reopens any of them, and they stay closed. The SI-9 RFC's
> §7 risks remain in force and are inherited, not
> relitigated.

---

## Table of contents

1. [Where SI stands after SI-31](#1-where-si-stands-after-si-31)
2. [Inherited invariants](#2-inherited-invariants)
3. [Candidate (a): a small real safetensors base — still deferred](#3-candidate-a-a-small-real-safetensors-base--still-deferred)
4. [Candidate (b): the composition line as a whole — DONE, closed](#4-candidate-b-the-composition-line-as-a-whole--done-closed)
5. [The completed attribution — the deep treatment](#5-the-completed-attribution--the-deep-treatment)
6. [Candidate (c): another point on the penalty axes — λ, λ_w, k, or τ](#6-candidate-c-another-point-on-the-penalty-axes---λ-λ_w-k-or-τ)
7. [Candidate (d): the representation question — rank, carrying, and judging constructed candidates on identity alone](#7-candidate-d-the-representation-question--rank-carrying-and-judging-constructed-candidates-on-identity-alone)
8. [Candidate (e): partial-removal projection](#8-candidate-e-partial-removal-projection)
9. [Candidate (f): per-probe anchoring — three readings, none reopened](#9-candidate-f-per-probe-anchoring--three-readings-none-reopened)
10. [Candidate (g): probe-partitioned sequencing](#10-candidate-g-probe-partitioned-sequencing)
11. [Candidate (h): magnitude discipline — a smaller second movement](#11-candidate-h-magnitude-discipline--a-smaller-second-movement)
12. [Candidate (i): a new LoRA target; additive / multi-target authorship](#12-candidate-i-a-new-lora-target-additive--multi-target-authorship)
13. [Candidate (j): transformer squash; candidate (k): Glimmer probing](#13-candidate-j-transformer-squash-candidate-k-glimmer-probing)
14. [Decision summary](#14-decision-summary)
15. [The recommended slice: SI-33 (outline)](#15-the-recommended-slice-si-33-outline)
16. [Acceptance criteria for SI-33](#16-acceptance-criteria-for-si-33)
17. [Non-goals](#17-non-goals)
18. [Honesty and labelling requirements](#18-honesty-and-labelling-requirements-for-any-future-slice)
19. [Risks and open questions](#19-risks-and-open-questions)
20. [Summary](#20-summary)

---

## 1. Where SI stands after SI-31

Everything in this section is verifiable in the code and
docs on `main` at the SI-31 merge (main tip `436b1bb`,
the merge of PR #36). Numbers are cited to the as-built
document that measured them; this RFC measures nothing
new. Where a statement below is a *derivation* from cited
facts rather than a measurement, it is labelled a
derivation, exactly as SI-30 §1 labelled its arithmetic.

**What exists, exactly:**

- **One measurable transformer subject, unchanged.** The
  SI-14 miniature (`TinyTransformerMind`): a constructed,
  deterministically seeded, **not trained** decoder-only
  transformer — 30 tensors, 23,456 elements. Architecture
  pinned: vocabulary 32, `d_model` 32, 2 layers, 4 heads,
  head dimension 8, MLP hidden 64, maximum sequence 16
  (`si/tiny_seed.py::ARCHITECTURE`). Probe suites: 8
  identity + 16 capability (echo-task) entries; recorded
  base echo accuracy 16/16 = 1.0, echo floor 0.9375;
  bundle tolerances both 0.25 (`identity_tol`,
  `capability_eps`).
- **Live probe verdicts for miniature bundles (SI-17),
  unchanged.** Probe-fail → REJECT, all-pass →
  ADMIT-eligible, anything unevaluated → ESCALATE,
  exclusively probe-blind candidates → blind-escalated
  (`si/probes.py::derive_probe_blind_targets`; the blind
  set is exactly both layers' `k_proj.bias`).
- **Two authored signals, one landed delta (SI-19 +
  SI-23), unchanged.** The signal registry serves exactly
  `tiny-successor-lean-001` (SI-19) and
  `tiny-predecessor-lean-001` (SI-23) and refuses every
  other id. Both signals share one form (target-logits
  MSE), one LoRA target (`lm_head.weight`, rank 8,
  alpha 8), correction scale 0.15, 2,000 full-batch steps,
  learning rate 1.0, norm budget 0.9, 32 train / 16
  held-out examples; they differ only in correction
  direction, stream, and drawn sequences.
- **Two recorded target anchorings (SI-25), unchanged.**
  `TARGET_ANCHORINGS` serves exactly
  `"weights-under-training"` (the default) and
  `"base-anchored"`, validated with the existing refusal
  discipline (`si/tiny_training.py`).
- **Four recorded composition configurations (SI-31).**
  `COMPOSITION_CONFIGS` serves exactly `"none"` (the
  default), `"movement-composition"` (SI-27),
  `"worst-input-composition"` (SI-29), and
  `"train-partition-projection"` (SI-31), validated by
  `validate_composition_config(...)` with the existing
  refusal discipline. The three mechanisms are never
  blended; their hyperparameter fields are mutually null.
  Under `"train-partition-projection"` the training step
  is SI-25's unpenalised path exactly, and the judged
  candidate is then *constructed* by full removal of the
  component parallel to the landed movement as measured
  on signal 2's train partition: coefficient
  α = ( Σ_x ⟨m1(x), m2(x)⟩ ) / ( Σ_x ⟨m1(x), m1(x)⟩ ),
  measured per run, recorded in the hyperparams, never
  settable, no fraction, no sweep. A projected candidate
  with α ≠ 0 is carried as a rank-16 LoRA factorisation
  of the folded update Δ2′ = Δ2 − α·Δ1 (SVD factors,
  rank 16, alpha 16, scale 1.0); at α = 0.0 exactly the
  construction is the identity and the candidate keeps
  its trained rank-8 factors byte-for-byte. Composition
  is a recorded per-cycle configuration beside recorded
  signal selection and recorded target anchoring; the
  gate's `Intent` carries all three selections. No
  autonomous, learned, depth-driven, verdict-driven, or
  probe-driven selection exists on any axis.

**What SI-31 measured (cited to its as-built):**

- **Phase 0 first.** Before any new candidate construction
  was finalised, SI-31 published the projection rehearsal
  on the three existing decisive candidates. All three
  coefficients are large and negative — SI-25 control
  α = −0.8450027779470831, SI-27 treatment
  α = −1.3459658380704567, SI-29 treatment
  α = −1.2483510742746164 — so on the train partition the
  second movement is predominantly ANTI-parallel to the
  first, and the parallel component the projection removes
  is 71.2% (SI-25), 85.3% (SI-27), and 82.3% (SI-29) of
  each candidate's movement aggregate there. Because α is
  negative, the folded construction ADDS |α|·Δ1 in weight
  space: every rehearsal projected candidate's weight-space
  norm is LARGER than its pre-projection norm (0.252 →
  0.289, 0.297 → 0.367, 0.283 → 0.351). Every rehearsal
  projected candidate's signal held-out loss is far worse
  than its unprojected candidate's, and every rehearsal
  projected candidate's composed maximum at the probes is
  far worse than the candidate it derives from (0.478 /
  0.555 / 0.526 against 0.333 / 0.325 / 0.312 unprojected,
  all at `tiny-id-0`, published as diagnostics on
  already-judged candidates). Phase 0 also produced the
  representation finding (§7): the difference of two
  rank-8 folded updates is generically rank 16 — the
  SI-25-based Δ′ has 16 nonzero singular values, 14
  substantial, then 0.00470232322 and 0.000156309136 — so
  Δ2′ cannot be carried exactly by any rank-8 LoRA
  candidate.
- **The decisive run.** One reused `TinyTransformerMind`
  through `si.run.run_cycles`: cycle 1 (signal 1, default
  anchoring, default composition configuration) ADMITted
  to `d0001`, reproducing SI-19/SI-21/SI-23/SI-25/SI-27/
  SI-29 exactly — candidate blob `sha256:2761e8ff…6ca9e`,
  identity drift 0.17189218290150166, capability regression
  0.1657181391492486, §6.6 VERIFY measured-pass. Cycle 2
  (signal 2, base-anchored, composition
  `"train-partition-projection"`, α measured at
  −0.8450027779470831, trained at `d0001`) was REJECTED on
  two independent lines; nothing landed; the bundle head
  after the run is `d0001`. Exactly one projection
  decisive run was performed; no sweep of any projection
  variant, fraction, or coefficient definition was run.
- **The construction did exactly what it defines, on the
  population it may read.** The decisive candidate's
  train-partition aggregate inner product with `m1` moved
  from its intermediate's −1.6873837019991194 to
  −4.8567197048765554e-08 — zero within the Phase-0
  tolerance (1e-6), measured through the forward path, not
  derived from the construction's algebra. On the held-out
  partition (reported only) the aggregate is near zero but
  not zero (−0.0009897609251914505, against the
  intermediate's −0.845596374259425): both allowed
  partitions price the parallel component with the same
  sign.
- **It fails where the gate binds — worse than every
  predecessor, on every binding figure.** The composed
  maximum is 0.4775387533009052 > tol 0.25, at `tiny-id-0`,
  where the projected second movement's increment is
  0.3063734769821167 — nearly twice SI-27's treatment's
  incremental movement at the same probe
  (0.15392476320266724, from SI-29's as-built accounting) —
  and the composed drift is again, to the printed digits,
  the sum of the two contributions (0.17189218290150166 +
  0.3063734769821167 ≈ 0.4775), the aligned extreme. The
  inner product at `tiny-id-0` is +0.14235826783471692 —
  positive, aligned, and worse than SI-29's treatment's
  +0.04944786620153699, SI-27's treatment's
  +0.03801222744105431, and SI-25's control's
  +0.0338873784299374. In aggregate on the probes the
  projected movement ALIGNS with the first
  (+0.21576810032436344 over the 8 identity probes; sign
  agreement 0.5, 128/256, published beside the
  magnitude-weighted figure) — the first treatment on
  record whose aggregate probe composition is positive,
  where SI-25's control opposed (−0.21256230517288188),
  SI-27's treatment opposed (−0.3271477846470207), and
  SI-29's treatment opposed (−0.28751472160662706). The
  folded weight-space inner product is likewise positive
  (+0.03542756883828178; SI-25: −0.011457914936582613;
  SI-27: −0.02013451284871203; SI-29: −0.0173962867164366).
  Per probe, the treatment's composed drift improves on
  SI-29's treatment at 1 of 8 probes and worsens at 7;
  against its own intermediate (SI-25's control) it
  likewise improves at 1 of 8 and worsens at 7. Its
  incremental movement exceeds cycle 1's remaining
  headroom (0.07810781709849834) at 7 of 8 probes.
- **The second rejection line is independent, and it is
  a policy line, not a measurement line.** The unchanged
  gate's safety suite fails on `size_budgets`: the
  constructed candidate's rank is 16 > `rank_cap` 8 (all
  other subchecks pass; aggregate norm 0.289021 ≤ cap
  1.0). The identity failure needs no rank argument and
  the rank failure needs no identity argument; SI-31
  reports both wherever it reports the verdict, and §7
  of this RFC argues what that independence does and
  does not license.
- **The destination cost is measured, under the
  pre-registered rule.** The projected candidate's signal
  held-out loss is 0.00140492372619575 (16/16 closer),
  against its intermediate's 1.043608588752459e-05 — the
  projection spent ~135× of the intermediate's
  completeness while keeping every held-out example
  closer. Composed capability passes (head-incremental
  regression 0.177281160838902 ≤ eps 0.25; echo accuracy
  1.0, 16/16).
- **The falsification readings (RFC SI-30 §7.5), as
  measured:** F1′′ **measured** — train-partition
  composition is zero as constructed and as measured,
  while probe composition at `tiny-id-0` is far worse
  than SI-29's treatment's. F2′′ **not measured as a
  reading** — its transfer clause fails (the projected
  candidate's inner product at its own worst probe does
  not improve on SI-29's treatment's figure at that
  probe; it worsens by ~2.9×), while its maximum clause's
  fact holds beside it (0.4775387533009052 > 0.25); the
  two are reported separately and are never rounded into
  one (§18 item 23). F3′′ **measured** — under the
  pre-registered rule, clauses (a) and (b) hold (loss
  falls; 16/16 closer) and clause (c) fails: residual
  0.00140492372619575 > boundary 0.0013205629553992592
  (2× the re-run SI-29 control's residual
  0.0006602814776996296), by ~6%.
- **The machinery is proven where it must vanish and
  where it must not look.** The coincidence configuration
  (signal 2, base-anchored, projection-configured, at
  base) records α = 0.0 measured (numerator and
  denominator both 0.0) and reproduces the established
  at-base run exactly (blob `sha256:131cc242…45f4`,
  identity drift 0.16672966815531254, capability
  regression 0.2418281203135848). Probe-blindness is
  asserted three ways, including a byte-identical
  projected candidate from a perturbed-probe-artifact
  bundle copy, and a spy asserting the coefficient's
  input set is exactly signal 2's train partition. All
  three controls re-run inside the slice reproduce their
  decisive candidates exactly (SI-25: blob
  `sha256:646d1e8b…d675`, drift 0.3329036422073841;
  SI-27: blob `sha256:1bb277ab…b637`, drift
  0.32509003952145576; SI-29: blob `sha256:11cab02e…b8db5`,
  drift 0.31215621903538704). The full suite at the
  SI-31 merge: **366 passed, 2 skipped**.

**The six-run cycle-2 record** (each figure under its
own run's labels; SI-31's as-built publishes all six
side by side):

| Run | Cycle-2 configuration | Candidate norm | Rank | Aggregate probe inner product | Inner product at worst probe | Composed identity drift (cumulative) | Worst probe | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| SI-21 | signal 2, weights-under-training, composition n/a (pre-SI-27) | (SI-21 as-built) | 8 | (SI-21 as-built) | (SI-21 as-built) | 0.34378434903919697 | (SI-21 as-built) | REJECT |
| SI-23 | signal 2, weights-under-training, composition n/a | 0.21317186599953494 | 8 | +0.07637087369869666 | (aligned at tiny-id-3, SI-23 as-built) | 0.2963416576385498 | tiny-id-3 | REJECT |
| SI-25 | signal 2, base-anchored, composition `"none"` | 0.2515524481203623 | 8 | −0.21256230517288188 | +0.0338873784299374 (tiny-id-0) | 0.3329036422073841 | tiny-id-0 | REJECT |
| SI-27 | signal 2, base-anchored, composition `"movement-composition"`, λ = 1.0 | 0.2969693161862222 | 8 | −0.3271477846470207 | +0.03801222744105431 (tiny-id-0) | 0.32509003952145576 | tiny-id-0 | REJECT |
| SI-29 | signal 2, base-anchored, composition `"worst-input-composition"`, λ_w = 1.0, k = 4 | 0.28252038537434354 | 8 | −0.28751472160662706 | +0.04944786620153699 (tiny-id-0) | 0.31215621903538704 | tiny-id-0 | REJECT |
| SI-31 | signal 2, base-anchored, composition `"train-partition-projection"`, α = −0.8450027779470831 (constructed) | 0.2890208630043799 (pre-projection 0.25155244947498906) | **16 (constructed)** | **+0.21576810032436344** | **+0.14235826783471692 (tiny-id-0)** | **0.4775387533009052** | tiny-id-0 | **REJECT (safety rank + identity)** |

Six authored second writes, six measured REJECTs, no
tolerance moved, no gate definition changed — and the
sixth is the first to be refused on a second,
independent line beside identity. The deepest landed
authored stack on record remains depth 1 (`d0001`); no
authored two-delta stack exists; the transformer-squash
precondition remains unmet. The Glimmer seed is
untouched; Glimmer-scale deltas still escalate with
checks 3/4 "not evaluated".

---

## 2. Inherited invariants

Not negotiable inside this RFC (SI-9 §1.3, RFC SI-13 §2,
RFC SI-18 §2, RFC SI-20 §2, RFC SI-22 §2, RFC SI-24 §2,
RFC SI-26 §2, RFC SI-28 §2, RFC SI-30 §2, restated for
the candidates):

1. **Base is frozen forever.** All change is a stack of
   deltas; rollback is moving `head`.
2. **Never duplicate bytes** (Mark's design rule). A blob
   whose hash exists is never rewritten or copied; seed
   imports by reference where the pattern applies
   (`stored_bytes: 0` — the SI-8 Glimmer seed stores zero
   tensor bytes against 59,553,374,040 deduplicated
   bytes, per the SI-31-era bundle record).
3. **No ungated write.** Authorship produces *candidates*.
   The gate — and only the gate — admits, rejects, or
   escalates them, and SI-17's verdict mapping and
   blind-set guard apply to authored candidates exactly
   as to fixtures, at every cycle, for every signal,
   under every anchoring, under every training
   configuration, and — for any slice this RFC is cited
   to justify — under every authorship target.
4. **Honest labels are load-bearing.** "Not evaluated"
   stays distinguishable from a measured 0.0; a training
   loss stays distinguishable from a gate result; a
   number measured at one head stays labelled with that
   head; incremental and cumulative figures are never
   presented as each other; a number recorded under one
   anchoring, composition configuration, or authorship
   target is never presented as a number under another;
   a worst-case figure names the population it is a worst
   case *over*; a judgement-call reading is applied under
   the rule pre-registered for it, or reported as an
   observation rather than a falsification test; a
   constructed candidate stays labelled constructed, and
   a candidate that fails on two independent gate lines
   is reported failing on both, never on the more
   convenient one (§18 items 22–25).
5. **Fail closed.** A missing, mismatched, or
   un-runnable measurement is "not evaluated" — never a
   pass, never a silent skip.
6. **CPU only, existing dependencies only.** The
   runtime's dependencies are `numpy` and `safetensors`
   (`requirements.txt`, `pyproject.toml`). No GPU, no
   network, no downloads, no spend, no new heavy
   dependency.
7. **Determinism where tests need it.** Seeded RNG,
   fixed inputs, no sampling, no wall-clock dependence
   in any measured or trained value — and deterministic
   *serialisation* wherever a blob hash is content
   addressing.
8. **No tolerance — and no cap — is a design variable.**
   No candidate in this RFC proposes moving
   `identity_tol`, `capability_eps`, `rank_cap`, any
   other cap, or any policy default to make an outcome
   pass. The six cycle-2 REJECTs are measurements to
   design *around* — with new authorship semantics —
   never numbers to outvote, and never numbers to
   re-price by changing what the gate measures or what
   it budgets (§7 argues the one candidate that appears
   to ask for exactly that, and does not adopt it).
9. **The probes are the judge, never the training
   set.** No candidate in this RFC may train on, penalise
   against, project against, parameterise against, or
   otherwise shape a candidate using any probe input,
   probe logit, or probe-derived movement. A method that
   reads the judge's inputs to shape the judged artifact
   has not passed the judge; it has memorised the
   courtroom. Every candidate below is audited against
   this invariant first — including §12's recommended
   candidate, whose training reads exactly the
   populations SI-19's training read.

---

## 3. Candidate (a): a small real safetensors base — still deferred

**Verdict up front: DEFERRED**, unchanged from SI-30 §3.
The SI-18 §3.7 audit's four preconditions (A-1 to A-4)
remain unmet, and SI-31's evidence does not touch any of
them: the miniature's composition and representation
problems are properties of two rank-8 readout updates, a
maximum-priced identity arm, and a safety budget, measured
on a constructed model; nothing about a projection's
non-transfer, or about the rank of a difference of two
readout updates, argues that a real base would compose or
carry differently, and naming a real base requires a
download this scope forbids and an operator decision this
RFC explicitly does not take. The local presence of the
Glimmer checkpoint (the SI-8 seed's referenced source)
does not retire the preconditions: A-2 (a binding design
for `reference-v1` bases), A-3 (a capability-suite design
for a trained base), and A-4 (an executor design scoped
to one family, with SI-15-grade numeric proof) are design
obligations, not download obligations, and none of them
exists. SI-31 adds one sentence to the audit's context,
in the same direction as SI-29's sentence cut: the project
now owns a completed composition record (three training
configurations and one construction, six measured
REJECTs, attribution closed at §5) whose behaviour at any
other scale is unclaimed — moving the authorship question
to a real base before the miniature's remaining
candidate-changing question (§12) is answered would carry
an unanswered question to a more expensive room.
Deferred; the download question stays the operator's.

---

## 4. Candidate (b): the composition line as a whole — DONE, closed

**Verdict up front: DONE — as SI-27, SI-29, and SI-31,
closed as a line.** RFC SI-30 §4 closed the
penalty-statistic family; SI-31 has now measured the one
candidate-changing move that family left open, and RFC
SI-30 §7.5's advance consequence is therefore in force,
not pending: no composition move that reads only allowed
inputs remains on the record. This section states
precisely *what* is closed, because §§6–8 propose
relatives of this line and §12 proposes its successor,
and none of them may borrow this line's evidence in
either direction:

- **Closed: the mean penalty** (SI-27, composition
  `"movement-composition"`, λ = 1.0). Its premise —
  composition measured on allowed inputs, at the mean,
  shapes composition on judged inputs — was falsified by
  SI-27's F1/F2 readings.
- **Closed: the worst-case penalty** (SI-29, composition
  `"worst-input-composition"`, λ_w = 1.0, k = 4). Its
  premise was falsified by SI-29's F1′: worst-case
  composition on allowed inputs improved decisively and
  closed the tail–bulk gap there, while probe composition
  at the binding probe worsened past both predecessors.
- **Closed: full-removal projection** (SI-31, composition
  `"train-partition-projection"`). Its premise — that the
  parallel component measured on allowed inputs is the
  component that adds at judged inputs — was falsified by
  F1′′ *measured*, in the strongest form the line
  proposed: the construction succeeded exactly on its own
  terms (train-partition aggregate inner product
  −4.8567197048765554e-08, within tolerance of zero) and
  the judged maximum moved further from admission than
  under any predecessor, because the component that
  opposes on allowed inputs is, at the binding probe, the
  component that was restraining the movement (§5).
- **Closed with them: the shared premise.** Three
  mechanisms — a statistic priced inside a loss at the
  mean, the same statistic priced at the worst case, and
  the indicted component removed outright as a
  construction — have now been measured against the same
  unchanged judge, and all three failed at the binding
  probe, the last two by *worsening* the quantity they
  were built to improve. A fourth mechanism whose
  information source is the same allowed population and
  whose target quantity is the same movement pair would
  need an argument that it is not the same premise at a
  fourth mechanism; no such argument exists in SI-31's
  record, and §§6–8 show the parameter, representation,
  and fraction variants of the existing mechanisms do not
  supply one.
- **Not closed:** moves that change *where an authored
  update acts*, rather than how a readout update's
  composition is priced or removed. Every run in the
  six-run record parameterises both cycles' updates on
  the same tensor, through the same readout fold — a
  premise so constant across the line that the line's
  diagnostics are defined through it (§5.4). F1, F1′,
  and F1′′ are readings about composition *of readout
  movements*; no non-readout candidate has ever existed
  for them to be readings *about*. §12 evaluates that
  move; it does not inherit this line's closure, and it
  does not inherit its plausibility either — it must
  argue from §5's measurements, as §12 does.

---

## 5. The completed attribution — the deep treatment

SI-27 made the question *why a mean-priced penalty loses
at the worst probe*; SI-29 sharpened it to *why a
worst-case penalty loses worse there*; SI-31 completes
the attribution by removing the priced quantity outright
and watching the judged end move the wrong way at full
size. This section assembles what the six decisive runs
jointly establish, what they leave unknown, and what any
candidate must therefore change — not re-price — to aim
at the binding figure. Derivations are labelled
derivations.

### 5.1 Allowed-input composition is not a population-independent property of the movement pair

The identity arm is seed-absolute, cumulative, and a
**maximum over probes**. Against that fixed judge, the
last three cycle-2 candidates were produced by
mechanisms that read only the allowed population: a mean
penalty over the 32 train sequences (SI-27), a worst-case
penalty over the same 32 (SI-29), and a construction
whose coefficient is an aggregate over the same 32
(SI-31). The sequence of binding-probe inner products is
now the finding, not any one entry in it: SI-25's
unpenalised control +0.0338873784299374, SI-27's
mean-penalised treatment +0.03801222744105431, SI-29's
worst-input-penalised treatment +0.04944786620153699,
SI-31's projected treatment +0.14235826783471692. Every
intervention whose explicit purpose was to oppose the
first movement *increased* the alignment at the binding
probe, monotonically across mechanisms of increasing
strength, while deepening opposition almost everywhere
it was allowed to look. SI-31 adds the aggregate form of
the same fact: the projected candidate is the first on
record whose aggregate probe composition is *positive*
(+0.21576810032436344), where the two penalty treatments'
aggregates opposed (−0.3271477846470207,
−0.28751472160662706). The honest generalisation the
record now supports — stated at exactly the scope the
record measures — is the one RFC SI-30 §7.5 wrote for
this outcome: **composition measured on allowed inputs —
at the mean, at the worst case, and as a construction —
is not a population-independent property of the movement
pair, through this readout, for this signal family, at
this scale.** Any candidate that reads only allowed
inputs to shape the *same readout movement pair* must
argue why it escapes a premise falsified at three
strengths; §§6–8's candidates cannot. A candidate that
changes what the movement pair *is* (§12) is not making
that claim and is judged by different readings (§15).

### 5.2 The parallel component has opposite signs on the two populations — measured, not inferred

SI-31's Phase 0 and decisive records together measure a
fact none of the penalty runs could state in this form.
On allowed inputs the second movement is predominantly
ANTI-parallel to the first: the projection coefficients
are −0.845, −1.346, and −1.248 across the three rehearsal
candidates, the train aggregates are strongly negative,
and the held-out partition prices the component with the
same sign as the train partition (the decisive projected
candidate nearly orthogonalises there too,
−0.0009897609251914505 against its intermediate's
−0.845596374259425). At the judged inputs the same
component is load-bearing in the opposite sense:
removing it — exactly, measurably, at full size — nearly
tripled the worst predecessor's binding-probe alignment
and turned the aggregate probe composition from opposing
to aligned. SI-31's as-built states the mechanism in one
sentence, and this RFC adopts it as the line's final
attribution: *the component that opposes on allowed
inputs is the component that was restraining the
movement at the binding probe.* This is F1's
train-to-probe gap at the level of a single named
component. It is also why no softer reading of SI-31 is
available: the construction did not fail to remove the
component (its train aggregate is −4.9e-08 against a
tolerance of 1e-6), the component was not negligible
(71–85% of the movement on allowed inputs), and the
candidate did not shrink into its failure (its judged
norm, 0.2890208630043799, is larger than its
pre-projection norm, 0.25155244947498906 — the failure is
a removal whose judged effect is the opposite of its
allowed-input effect, at full size).

### 5.3 The magnitude half — unchanged, and still not a dial

Composition was never the whole failure, and the
completed attribution does not let the direction story
swallow the size story. Cycle 1 leaves headroom of
0.07810781709849834 under tol at the binding probe.
SI-31's projected second movement's increment at
`tiny-id-0` alone is 0.3063734769821167 — nearly four
times the headroom — and its incremental movement
exceeds the headroom at 7 of 8 probes, as SI-29's did.
Two consequences carried from SI-30 §5.3, both still
binding: (i) magnitude remains a **published, compared
quantity** — candidate norm and ‖m2‖ beside every drift
figure, in SI-33 as in every composition run (§16
criterion 8); (ii) §11's rejection stands guard on the
other side — deliberately shrinking the movement by norm
budget, step count, or correction scale is gate-fitting,
not a result, and is rejected there so that (i) cannot
be misread as licensing it. SI-31 sharpens (i) once
more: its candidate *grew* under its construction while
its binding-probe step nearly doubled SI-27's, so
neither "improvement by shrinking" nor "failure by
shrinking" is available as a reading of the projection
in either direction.

### 5.4 Every run in the record shares one fold — the constant the line never varied

The six runs differ in signal semantics (SI-21/SI-23),
target anchoring (SI-25), and composition mechanism
(SI-27/SI-29/SI-31). They are identical in one respect
the line's own diagnostics make easy to overlook: both
cycles' updates are LoRA factors on the *same tensor*,
`lm_head.weight`, applied as the model's final operation.
SI-30 §7.2 stated the consequence for the projection:
both movements pass through the same readout fold —
`m1(x) = Δ1 · h(x)` and `m2(x) = Δ2 · h(x)` with `h(x)`
the final hidden state at `x` under the landed head, the
readout being the final operation, so `h` does not
depend on either update — and SI-26 §10 named the
geometric fact the fold produces: both signals'
corrections share the away-from-the-last-token component,
so both updates act on every hidden state the probes
present in the same coordinates, in the same way. The
composition line varied *how the second readout update
is priced and constructed* exhaustively; it never varied
*where an update acts*. That is not a criticism of the
line — varying one thing at a time is what makes its
attribution complete — but it bounds the attribution's
scope exactly: F1, F1′, and F1′′ falsify premises about
**readout** movement pairs. Whether a second update that
acts *upstream* of the readout — changing the hidden
states the readout then reads, so that `m2` is not
`Δ2 · h(x)` for the landed head's `h` at all — composes
differently at the judged maximum is **an open question
no run on record measures**, because no such candidate
has ever been constructed. §12 takes up that question;
§5.5 records the unknowns it must retire first.

### 5.5 What is still not known — the measurements and definitions that must precede any fix

Stated as open questions a slice could retire — this
RFC invents no numbers for them (§19 carries them
forward):

- **The representation question, argued but not
  measured in either direction.** SI-31 reported that a
  projected candidate is generically rank 16 and that
  the unchanged gate refuses it on that line beside the
  identity line. No candidate on record passes the
  identity and capability arms while failing *only* on
  rank, so the question §7 argues — whether a constructed
  candidate's rank should be priced as a trained
  candidate's rank is priced — has never been the
  deciding question for any verdict. §7 defers it with a
  precise trigger rather than a date.
- **Partial removal was never constructed.** No
  β-fraction candidate (Δ2 − β·α·Δ1 for any β strictly
  between 0 and 1) exists or has been measured. §8
  evaluates it from the two measured endpoints and from
  rank arithmetic, and labels every step of that
  evaluation a derivation; the absence of a measurement
  is stated there, not papered over.
- **No non-readout gradient has ever been checked.** The
  training machinery differentiates exactly one path:
  target-logits MSE through the readout fold into
  `lm_head.weight` factors, proven by SI-19/SI-25's
  finite-difference checks over the 512 readout factor
  entries. For any other tensor, the backward path is
  unbuilt and unchecked — and through attention or
  through both layers it is a materially larger
  derivation than through the final layer's MLP. Which
  non-readout target has the shortest honestly checkable
  backward path, and whether its gradients pass the
  project's standing finite-difference discipline, is
  **an open question to measure**, in Phase 0, before
  any decisive run (§15).
- **The destination through an upstream update.** Both
  signals' corrections are defined as target-logit
  shifts; whether a single upstream tensor's rank-8
  update can carry signal 2's lean to the completeness
  the readout path reaches (the SI-25 control's held-out
  residual, 1.043608588752459e-05) is unknown in both
  directions. §15's F3′′′ reading carries SI-29's rule
  shape forward against the SI-25 control so the answer,
  whichever way it falls, is read under a rule stated
  before the numbers.
- **F2′′'s residue.** F2′′ did not fire in its registered
  form: its transfer clause fails and its maximum
  clause's fact holds beside it (§1). The proposition it
  was written to test — the maximum as a
  magnitude-and-geometry fact *despite* transfer — was
  therefore not the proposition SI-31 measured, and it
  remains untested as a proposition. §15's F2′′′ is
  written in the repaired, clause-separated form §5.5 of
  RFC SI-30 required, against the new candidate's own
  control, so the same definitional gap cannot recur
  silently.

---

## 6. Candidate (c): another point on the penalty axes — λ, λ_w, k, or τ

**Verdict up front: DEFERRED as open questions, as in
SI-30 §6 and §11.1 — none of them is the next slice, and
SI-31 makes each of them weaker, not stronger.** SI-30
deferred a different λ for the mean penalty, a different
λ_w, a different k, and the τ (log-sum-exp) form, on the
ground that F1 and F1′ had settled their mechanisms at
the statistic, not the weight. SI-31 does not reopen any
of those arguments; it removes the one reply they had
left. After SI-29, a defender of the penalty axes could
still say the *statistic* was the problem — that pricing
composition inside a loss, at any statistic, trades it
against the signal at every step, and that changing the
candidate outright might succeed where pricing failed.
SI-31 ran exactly that reply, at full strength, with the
priced quantity removed to −4.9e-08 — and the binding
probe worsened past every penalised predecessor. The
penalty axes now vary the parameters of mechanisms whose
shared premise (§4) has been falsified at a strength
strictly greater than any penalty can express: a penalty
at any weight, tail width, or temperature removes *less*
of the parallel component than SI-31 removed, on the
same population, through the same fold. Parameter
variation after a premise falsification is tuning, and
tuning against a fixed judge with the judge's tolerance
in view is the gate-fitting shape §§9–11 reject in their
own domains. The axis stays open as a set of *questions*
— §19 records each with what would retire it — and
closed as a *recommendation*. No sweep is licensed by
the deferral, and no single argued value on any of these
axes is recommended by this RFC.

---

## 7. Candidate (d): the representation question — rank, carrying, and judging constructed candidates on identity alone

**Verdict up front: ARGUED, and DEFERRED with a precise
trigger — no policy change is recommended, and SI-31's
candidate is not a candidate this question could rescue.**
SI-31's as-built left this question explicitly to a
future RFC, in its own words: *"A future RFC that wants
projected candidates judged on identity alone must argue
the representation question itself — this slice does not
argue it, it reports it."* This section is that argument.
It argues the question on its merits in both directions,
and its conclusion is that the argument, however it is
resolved in principle, has no measured object to attach
to today — and that changing a safety budget in the
absence of such an object would be repricing the judge
after a failure, in the shape invariant 8 forbids.

### 7.1 The question, precisely

A projected candidate's folded update is Δ2′ = Δ2 −
α·Δ1, the difference of two rank-8 updates. Measured in
SI-31's Phase 0, such a difference is generically rank
16 (16 nonzero singular values; 14 substantial, then
0.00470232322 and 0.000156309136), so no exact rank-8
LoRA carrying exists. SI-31 carried the candidate exactly
— SVD factors, rank 16, alpha 16, scale 1.0, 4,408 blob
bytes against 2,360 for a rank-8 candidate — and the
unchanged gate's safety suite refused it on
`size_budgets` (rank 16 > `rank_cap` 8), independently of
its identity failure. The representation question is
whether that refusal prices the right thing: whether
`rank_cap` — a budget written for *trained* LoRA deltas,
whose rank is a capacity choice made before training —
should price a *constructed* candidate the same way,
when its rank is not a capacity choice at all but an
arithmetic consequence of its construction. The strong
form of the question, the one SI-31 named, is whether
projected candidates should be *judged on identity
alone*: the rank line waived or re-priced for the
constructed class, so the identity arm decides.

### 7.2 The argument for re-pricing — stated at full strength

- **Rank, for a constructed candidate, measures the
  construction, not the candidate's reach.** A trained
  delta's rank bounds what its training could express; a
  reader of a rank-16 record learns that its author
  reached for more capacity. A projected candidate's
  rank 16 encodes no such reach: its author trained a
  rank-8 candidate, exactly as every predecessor did, and
  the doubling is the rank of a *difference* — rank(Δ2 −
  α·Δ1) ≤ rank(Δ2) + rank(Δ1) = 16, with equality generic.
  The number the safety suite prices is, for this class,
  a theorem about subtraction, not a fact about the
  candidate's ambition, size in any behavioural sense, or
  risk surface beyond what the norm and byte budgets
  already price separately (both of which the SI-31
  candidate passed: norm 0.289021 ≤ 1.0; 4,408 bytes
  against a 4 MiB `delta_byte_cap`).
- **The exact carrying is the honest carrying.** The
  alternatives SI-31 rejected were rejected correctly:
  a rank-8 truncation of Δ2′ would not be the §7.2
  construction (its train-partition aggregate would be
  whatever the truncation left, not zero), and an
  additive carrying is policy-forbidden. If the only
  exact representation of a lawful construction is
  refused *as a representation*, the gate is — on this
  reading — pricing the file format, not the self-write.
- **Precedent exists inside the project's own
  arithmetic.** Squash already treats rank as a
  representation quantity with its own discipline: a
  squashed delta is stored at the rank its truncated SVD
  produces, with the gate's rank-cap check applying to
  the true factorisation rank (`si/squash.py`) — the
  project has never held that a delta's rank is
  meaningless, but it has held that rank follows the
  factorisation, not the other way round.

### 7.3 The argument against — stated at full strength

- **The cap is a budget, and budgets are what keep a
  stack a stack.** `rank_cap` does not only price one
  delta; it prices the growth of the whole authored
  stack. A landed rank-16 delta occupies, in
  representation terms, the space of two rank-8 deltas —
  and SI-20 §11 recorded the composed-side consequence:
  composing a rank-≤16 stack back down to rank ≤ 8
  (`rank_cap`) is *lossy*. Admitting constructed deltas
  at rank 16 spends the stack's compression headroom at
  the moment of admission, before squash — whose
  precondition (an authored two-delta stack) is still
  unmet — has ever been exercised on an authored stack
  at all. A budget waived for the first candidate that
  meets it has not been argued; it has been spent.
- **A class-based judge is a second judge.** "Judged on
  identity alone" does not amend the rank line; it
  creates two safety suites — the full suite for trained
  candidates, a reduced suite for constructed ones —
  selected by a label the candidate's own record carries.
  Every prior slice held the gate fixed across signals,
  anchorings, and composition configurations precisely so
  that verdicts across runs are comparable; the six-run
  table in §1 is readable because one judge priced all
  six. A constructed-class waiver breaks that
  comparability prospectively and invites the obvious
  next step — a further class, a further waiver — each
  argued, as this one would be, from the candidate that
  needs it.
- **The timing is the gate-fitting shape.** This question
  is being argued — unavoidably — immediately after the
  first candidate in the project's history failed the
  line it concerns. Invariant 8 and SI-30 §8.1's carried
  rejection exist for exactly this posture: a budget
  re-priced after it binds, in a document whose author
  knows the candidate it would have affected, is
  indistinguishable in form from repricing the identity
  tolerance after a drift failure, however sound its
  arithmetic. The argument in §7.2 may be right; making
  it *now*, with no candidate whose verdict it would
  honestly change (§7.4), purchases nothing measurable
  and spends the discipline that makes the record
  trustworthy.

### 7.4 The deciding fact: no verdict on record turns on this question

Both arguments above are live, and this RFC does not
pretend otherwise. What decides the recommendation is
not which argument is stronger in the abstract but a
measured fact about the only candidate the question has
ever attached to: **SI-31's projected candidate fails
the identity arm independently, by the widest margin on
record** — drift 0.4775387533009052 against tol 0.25,
i.e. ~1.91× the tolerance (derivation from the cited
figures). Judged on identity alone, under the strong
form of §7.1, its verdict is unchanged: REJECT. The
rank line is therefore not the binding constraint on
any outcome the project has measured; it is a second,
independent refusal standing beside a first refusal that
already decides. A policy change adopted today would
change no verdict, admit no candidate, and retire no
unknown — it would only re-price a budget in the
posture §7.3's third point warns against, in exchange
for a principle with no measured object. Principles of
that kind are exactly what the RFC cadence exists to
argue *before* they are needed, and §7.5 does so — as a
trigger, not a change.

### 7.5 Verdict and trigger

**DEFERRED.** `rank_cap` stays 8, for trained and
constructed candidates alike, and no slice this RFC
recommends touches it. The representation question
becomes arguable as a *change* — in a future RFC, on
its own, never as a rider on a candidate's slice — when
and only when a candidate exists that is measured
passing the identity arm and the capability arm while
failing *only* on the `size_budgets` rank line. That
trigger is precise on purpose:

- the candidate must be lawful under every other
  invariant (probe-blind in training and construction,
  full post-construction accounting, honest labels);
- its identity and capability figures must be the
  unchanged gate's own journaled measurements, not
  diagnostics;
- and the future RFC must argue, against §7.3, the
  stack-growth and squash consequences, the
  class-definition question (what makes a candidate
  "constructed" for this purpose, and who attests it),
  and why the change is not retroactive: **SI-31's
  candidate may never be re-judged under a future rank
  rule** — its verdict stands on both lines, permanently,
  because its identity failure is independent of the
  rank question this section defers.

Until the trigger fires, any text that describes the
rank line as "the reason SI-31 failed" without the
identity line beside it is a labelling defect (§18
item 25).

### 7.6 Other representations — evaluated, none adopted

For completeness, the carryings that would avoid the
rank question without changing policy, each disposed of
on the record:

- **Rank-8 truncation of Δ2′.** A different candidate,
  not the §7.2 construction: it leaves a train-partition
  aggregate inner product of the truncation's choosing,
  not zero (SI-31's finding). SI-31's Phase-0 singular
  record prices the loss honestly: with 14 substantial
  singular values, a rank-8 truncation discards
  substantial components, not a numerical tail — this is
  a *partial* construction in rank space, and §8's
  verdict on partial constructions applies to it.
- **Two stacked rank-8 deltas.** Writing Δ2′ as a pair of
  rank-8 deltas changes the stack's semantics, not only
  its carrying: the second member of the pair would
  itself be an authored write requiring its own gate
  judgement, the decomposition of a difference into two
  summands is arbitrary where the construction's
  coefficient is not, and one natural summand is −α·Δ1 —
  a rescaling of an already-landed delta, which is not
  an authored signal's candidate under any reading of
  SI-19/SI-23. Rejected.
- **Additive target.** Policy-forbidden (`allow_additive`
  false, unchanged), and §12.2 carries the standing
  rejection of additive authorship as a slice. Rejected.

---

## 8. Candidate (e): partial-removal projection

**Verdict up front: REJECTED as the next slice — on two
independent grounds, each sufficient: it inherits §7's
rank refusal for every nonzero fraction, and its
endpoints are the two measured REJECTs it interpolates
between.** SI-31 explicitly left this candidate open, in
its own words: *"no partial-removal variant was run or
is reported as anything but a different candidate for a
future RFC."* This is that future RFC, and it evaluates
the candidate rather than reporting it. What follows
rests on derivations from cited figures and on rank
arithmetic, labelled as such throughout: **no
partial-removal candidate has been constructed or
measured, and nothing in this section is a measurement.**

### 8.1 What the candidate is

Train signal 2 exactly as SI-31's intermediate was
trained (base-anchored, unpenalised, at `d0001`), compute
α exactly as SI-31 computed it (−0.8450027779470831 for
that intermediate at that head), and judge the candidate
carrying Δ2(β) = Δ2 − β·α·Δ1 for one fixed fraction β
strictly between 0 and 1 — β = 0.5 being the obvious
arguable value. β = 0 is SI-25's control; β = 1 is
SI-31's treatment. The candidate's appeal is real and is
stated fairly: full removal spent the destination past
the F3′′ boundary (§1) and worsened the binding probe at
full size, so a *partial* removal might — the hope runs —
keep part of the construction's train-partition benefit
while paying only part of its judged cost.

### 8.2 First ground: every β > 0 inherits the rank refusal

(Derivation, from SI-31's measured singular record.)
Δ2(β) is, for any β ≠ 0, a difference of two rank-8
updates with both summands generically present: its rank
is generically 16, by the same arithmetic as §7.1 — β
scales the second summand's singular values but does not
annihilate any of them, and SI-31's Phase-0 record shows
the unscaled difference's smallest singular values
(0.00470232322, 0.000156309136) are nonzero, not a
numerical tail that a fraction could plausibly collapse.
A partial-removal candidate therefore fails the
unchanged gate's `size_budgets` line on rank for every
β > 0, exactly as the full-removal candidate did —
before its identity figure is ever reached. The
candidate cannot ADMIT under the current policy at any
fraction, and it can only become admissible by first
winning §7's deferred policy argument, which §7.5's
trigger reserves for a candidate that passes identity —
which §8.3 shows this candidate has no mechanism to be.
Sequencing partial removal ahead of the representation
question is therefore not a shortcut around §7; it is
§7 with a dial attached.

### 8.3 Second ground: the interpolation has no turning mechanism

(Derivation, from the linearity of the forward path in
the folded update.) The candidate family Δ2(β) is affine
in β, and the readout fold is linear, so the second
movement at every input is affine in β:
`m2(β, x) = m2(x) − β·α·m1(x)`. At the binding probe the
record prices the two endpoints: at β = 0 the inner
product with `m1` is SI-25's +0.0338873784299374
(aligned); at β = 1 it is SI-31's +0.14235826783471692
(aligned, and worse). The inner product at that probe is
likewise affine in β (an inner product of affine
quantities against a fixed `m1`), so at `tiny-id-0` it
interpolates between those two positive values — it is
positive at every β in [0, 1], and increasing in β,
because α < 0 means each unit of removal *adds* the
first movement's direction at the probe where that
direction is the composed drift's own direction
(§5.2's mechanism, in algebraic form). No fraction turns
the binding probe; the family's best binding-probe
figure on this record is its β = 0 endpoint, which is
SI-25's control — itself REJECTED at 0.3329036422073841.
The composed maximum is a maximum of absolute values
rather than an inner product, so strict monotonicity of
the *maximum* in β is not claimed here (a derivation
that claimed it would overreach); what is claimed is
the narrower, sufficient fact: **the quantity the
construction exists to change — the binding probe's
alignment — moves the wrong way, linearly, from a
positive start, across the whole family**, and the
family's destination record interpolates between
SI-25's completeness and SI-31's F3′′-measured trade
with no argued point at which either figure crosses its
threshold. A β chosen from this record would be chosen
by looking at endpoints the gate has already priced —
a tuned dial aimed between two REJECTs, in the shape
SI-30 §16 named this non-goal's clearest instance
before SI-31 existed.

### 8.4 What stays open

This verdict closes partial removal *as the next
slice*, on this record. It does not claim a measurement
of any β, and it does not foreclose a future RFC that
arrives with genuinely new information — for example,
if §7.5's trigger ever fires and the representation
question is resolved on a candidate that earns it, the
rank ground of §8.2 falls away and the family would
stand or fall on §8.3 alone, argued then against
whatever record exists then. On today's record both
grounds hold, and §15 recommends elsewhere.

---

## 9. Candidate (f): per-probe anchoring — three readings, none reopened

**Verdict up front: SPLIT, carried exactly as SI-30 §8
left it — gate-side REJECTED, probe-side training
REJECTED, per-input accounting on allowed inputs
ADOPTED only inside a slice that depends on it.**
SI-30 §8 separated the three readings this phrase names
and recorded that SI-29 supplied no new evidence for any
of them. This RFC records the same for SI-31, in the
same terms, because the cadence requires the closure to
be restated per candidate rather than assumed:

- **Gate-side (per-probe accounting or budgets):
  REJECTED (carried, not reopened).** It reprices the
  judge after — now — six measured failures while leaving
  0.25 printed in the policy; SI-31's diagnostics confirm
  the verdict's figure at the binding probe, and SI-31
  adds a second, independent gate line (rank) whose
  existence makes per-probe identity accounting *less*
  sufficient as an explanation of refusal, not more: the
  failure is in the candidates, six times measured, on
  two lines in the sixth. Not reopened.
- **Probe-side (training or construction against
  per-probe targets): REJECTED (invariant 9).** SI-31
  makes this reading more tempting in one specific way —
  its Phase 0 *diagnostics* priced the binding probe's
  behaviour for three already-judged candidates before
  the decisive run, and those diagnostics predicted the
  decisive failure's direction — and temptation is not
  evidence: a diagnostic on an already-judged candidate
  is post-verdict accounting (§18 item 12); a target
  computed from it would be training on the judge. An
  ADMIT so produced would be uninterpretable. Rejected
  outright, with no "diagnostic-only" carve-out for any
  training or construction path.
- **Per-input accounting on allowed inputs: ADOPTED,
  inside §15 — not a slice of its own.** The invariant-
  clean residue, as in SI-30 §8.3: accounting on the
  populations training may see, riding with the slice
  whose design it informs. In SI-33 that is Phase 0's
  publication of the SI-25 control's per-probe and
  allowed-input records as *diagnostics and control
  figures* under their own labels (§15 item 1) — the
  accounting the F2′′′ reading's transfer clause needs
  stated in advance. A standalone accounting slice is
  not recommended: SI-31's Phase 0 closed the
  per-input questions for the projection family, and no
  table without a construction or training path that
  depends on it is a frontier.

---

## 10. Candidate (g): probe-partitioned sequencing

**Verdict up front: REJECTED (carried, not reopened).**
The proposal: sequence the writes so successive cycles
move disjoint parts of the judged behaviour. SI-30 §9
rejected both of its readings, and SI-31 supplies no new
evidence for either. Partitioning by probe behaviour
requires reading probe movements per candidate signal —
shaping the sequence by the judge's inputs (invariant
9), at the driver's level, which recorded selection does
not launder. Partitioning by allowed-input behaviour is
SI-26 §10's rejected signal search in sequence form —
argued backward from the obstacle the gate measured —
and SI-31 strengthens that rejection's evidence in the
same way SI-29 did, one level deeper: the family's
composition behaviour is now measured failing transfer
at the mean, at the worst case, and as an exact
construction, and §5.2 names the component-level reason.
Partitioning the family's members by where they move on
allowed inputs does not change what each member's
movement *is* at the judged inputs; on SI-31's record
it would partition by a quantity whose judged sign is
the opposite of its allowed sign. The weaker relative
SI-30 §9 noted survives unchanged: nothing here speaks
against more than two cycles someday, or against signals
authored for reasons independent of the gate. What is
rejected is partitioning — of probes or of signals — as
the mechanism for making the maximum pass. Not reopened.

---

## 11. Candidate (h): magnitude discipline — a smaller second movement

**Verdict up front: REJECTED as a slice (carried, not
reopened); retained as a publication obligation inside
§15.** §5.3 establishes that size remains half the
failure: SI-31's second movement exceeds cycle 1's
remaining headroom at 7 of 8 probes, as SI-29's did. The
tempting candidate — train cycle 2 under a smaller norm
budget, fewer steps, or a reduced correction scale — is
disposed of by the standing rules, and SI-31 does not
reopen them: SI-26 §16 names reduced correction scale,
step count, and norm budget chosen to shrink `m2` a
binding non-goal, for the reason SI-30 §10 restated — a
shrunken candidate that ADMITs has shown that a smaller
second write fits, which the project has known since
cycle 1's drift left headroom; it prices the destination
down until the judge stops objecting. SI-31 adds one
caution in the other direction, recorded at §5.3: its
own candidate *grew* under construction while failing
worse, so no future candidate — §12's included — may
report a maximum improvement without the norm accounting
that separates shrinking from turning, nor a maximum
worsening without the accounting that separates growing
from aligning. Magnitude as accounting: adopted, in
§16 criterion 8. Magnitude as a training dial aimed at
the headroom: rejected. Not reopened.

---

## 12. Candidate (i): a new LoRA target; additive / multi-target authorship

### 12.1 The new-target candidate — RECOMMENDED, as SI-33 (§15)

**Verdict up front: RECOMMENDED — one named target, the
final layer's MLP down-projection, phased
measurement-first (§15).** This is the candidate SI-26
§11.1 first deferred, carried as deferred by SI-28
§11.2 and SI-30 §11.2, and it is the only candidate this
RFC recommends. It is not a composition move: the
decisive run it outlines uses composition `"none"`, no
penalty, no projection, no fraction. It changes the one
premise every run in the six-run record held constant
(§5.4): *where the authored update acts.*

**What it is.** A recorded authorship-target axis beside
signal, anchoring, and composition, serving exactly two
values: `"lm-head"` (the default; every existing run's
target, `lm_head.weight`) and `"final-mlp-down"` (the
final layer's MLP down-projection,
`model.language_model.layers.1.mlp.down_proj.weight`,
shape [32, 64]). Under `"final-mlp-down"`, cycle 2
trains signal 2's correction — the same target-logits
MSE loss, the same partitions, the same hyperparameters
— into rank-8, alpha-8 LoRA factors on that tensor
instead of on the readout. The update thereby acts
*upstream* of the readout: it changes the final layer's
hidden states, and the landed head's readout reads
hidden states the second update has itself moved, so
`m2(x)` is not `Δ2 · h(x)` for the landed head's `h`
and the shared-fold geometry of §5.4 — both corrections
sharing one component through one fold — does not
describe the candidate. Whether that difference helps,
hurts, or changes nothing at the judged maximum is the
experiment; §15 states its premise in three falsifiable
forms and requires no ADMIT.

**Why this target, of the 29 non-readout tensors.** The
choice is argued from the backward path, not from any
judged quantity (the target is named here, in design,
before any candidate exists; no probe figure selected
it):

- **Shortest non-readout backward path in the model.**
  The final layer's `down_proj` is the last tensor
  before the residual add that feeds the final RMSNorm
  and the readout. Differentiating target-logits MSE
  into its factors requires the chain through the
  readout, the final norm, and one linear map — with the
  SwiGLU activation (`silu(gate) ⊙ up`) entering as a
  *fixed input* to that map for the factor gradients.
  It requires no attention backward (no softmax
  Jacobian, no per-head routing), no propagation through
  layer 0, and no embedding backward. Every other
  non-readout weight matrix demands strictly more: any
  layer-0 tensor needs the full two-layer chain; any
  attention projection needs the attention Jacobian; the
  embeddings need the input-side chain.
- **It is measurable by the judge.** The derived
  probe-blind set is exactly both layers' `k_proj.bias`
  (§1); this target is not in it, and a candidate
  targeting it exclusively is therefore not
  blind-escalated — its identity and capability figures
  are real gate measurements, not "not evaluated".
- **It is a single target at the standing budgets.**
  Rank 8, alpha 8, one target: `rank_cap` 8,
  `max_targets_per_delta` 1, and `allow_additive` false
  all price it exactly as they price every existing
  candidate. Unlike §7's and §8's candidates, this one
  needs no policy argument to be judged.
- **Its factors are fully checkable.** Rank-8 factors on
  a [32, 64] matrix are B [32, 8] and A [8, 64] — 768
  factor entries — each one subject to the project's
  standing central-finite-difference discipline at
  tolerance 1e-5, at base and at `d0001` (§16
  criterion 3). The check is larger than the readout's
  512 entries and the same in kind.

**Why the prior deferrals do not bind this candidate
now.** SI-30 §11.2 deferred a new target on two grounds:
it needs a full backward derivation plus a per-target
finite-difference gradient check, and the composition
diagnostics are readout-geometry objects, so a candidate
whose update changes hidden states "is not even
*described* by the instruments §5 reasons with". Both
grounds were correct as sequencing, and the sequence
has now run out:

- The first ground is a *cost*, and §15 pays it as the
  slice's Phase 0 and criterion 3 — derivation
  published, finite differences over every factor entry
  at both heads, tolerance unchanged — before any
  decisive number is trusted. A deferral whose stated
  price is a derivation and a check is retired by
  building exactly those, in a slice scoped to the one
  target whose derivation is shortest.
- The second ground misidentifies which instruments
  must describe the candidate. The *gate* instruments —
  per-probe drift, composed maximum, capability
  regression, echo accuracy — are forward-path
  measurements; they describe any candidate whose
  weights can be overlaid, whatever tensor it targets,
  and SI-31's diagnostic set is computed exactly that
  way. What does not survive the target change is the
  *composition* diagnostics' fold derivation (α's
  movement-space construction, the weight-space inner
  products read as readout geometry) — and SI-33 claims
  no composition result, so it needs no composition
  instrument: its diagnostics are the gate's figures,
  its controls', and its signal records, each under the
  new target's label (§18 item 24). The deferral's
  ground was that a new target could not be *reasoned
  about* with composition tools; §4 has closed the
  composition line, so the project no longer needs to
  reason about its next candidate with them.

**The one deliberate definition change, stated plainly.**
Both signals currently *pin* their target: SI-19/SI-23
defined each signal with `lm_head.weight`, rank 8,
alpha 8 as part of the signal's spec. SI-33 moves the
target from the signal's spec to a recorded driver axis,
so that "signal 2 under target `"final-mlp-down"`" is a
configuration of a known signal, not a third signal.
This revises the *pinning*, not the signals: their ids,
streams, corrections, partitions, loss form, rank, and
alpha are byte-for-byte as SI-19/SI-23 defined them, and
every figure in the six-run record keeps its meaning —
recorded, in hindsight-explicit form, under target
`"lm-head"`. §16 criterion 1 requires the axis to refuse
every other value with the existing discipline, §16
criterion 10 requires the signals' remaining definitions
to be asserted unchanged, and §17 lists "no silent
redefinition" among the non-goals: the change is this
paragraph, argued here, or it does not happen.

**What this candidate is not.** It is not a claim that
upstream updates compose better — nothing on record
measures one. It is not a third signal (§17), not a
search over targets (exactly one target is named; a
second target is a future RFC's argued candidate, never
a sweep within SI-33), and not a composition result in
disguise: if its cycle 2 REJECTs with the binding probe
aligned, §15's F1′′′ records that the geometry the six
runs indicted survives the fold change, for this target,
and the authorship line — not only the composition line
— will have to find a genuinely new argument after it.

### 12.2 Additive / multi-target authorship — REJECTED as a next slice (carried)

Unchanged from SI-30 §11.2: additive and multi-target
authored candidates require loosening
`max_targets_per_delta` / `allow_additive` policy
defaults with zero authored-behaviour evidence under
them, and six REJECTs under the defaults are not
evidence *for* loosening them. §12.1's candidate is
deliberately a *single*-target move: it changes which
tensor, never how many. Not reopened.

---

## 13. Candidate (j): transformer squash; candidate (k): Glimmer probing

### 13.1 Candidate (j): transformer squash for miniature deltas — DEFERRED

**Verdict up front: DEFERRED**, on the same ground as
SI-30 §12.1, now firmer: the precondition — an authored
two-delta stack — remains unmet after *six* measured
attempts (SI-21, SI-23, SI-25, SI-27, SI-29, SI-31), and
SI-12's fail-closed refusal therefore still has nothing
authored to squash. SI-31 adds two design facts a future
squash RFC inherits, extending the ones SI-30 §12.1
recorded: a landed stack's deltas may now carry any
combination of recorded signal, anchoring, and any of
*four* composition configurations, one of whose values
denotes a construction whose delta is rank 16 — so if
such a delta ever lands (it has not; §7.5's trigger has
not fired), the squash gate context must represent a
member whose rank already exceeds the composed rank it
would be squashed toward; and if SI-33 lands its cycle,
a fourth recorded axis (authorship target) joins the
context's obligations, one of whose values denotes an
update that acted upstream of the readout. Recorded here
so the future RFC meets them as known obligations.
Deferred; §15 builds nothing toward it.

### 13.2 Candidate (k): probing Glimmer — REJECTED, not reopened

**Verdict up front: REJECTED.** Rejected at SI-13
§7/§9 and restated in every successor RFC including
SI-30 §12.2; SI-31 changes nothing: every mechanism in
the six-run record — penalties, projection, and §12's
proposed target axis alike — is a miniature mechanism
and claims nothing at Glimmer scale, the Glimmer seed
is untouched, and its deltas still escalate with checks
3/4 "not evaluated". There is no forward path for
Glimmer-scale bundles in the current design — no probe
suite, no forward path, no gate reading beyond
escalation — and this RFC does not propose faking one,
by miniature results, by analogy, or by any other route.
Success *or* continued failure on the miniature does not
soften the answer; §19 records the pressure honestly so
it is met as a standing question, not a surprise. Not
reopened.

---

## 14. Decision summary

| Candidate | Verdict | One-line reason |
|---|---|---|
| (a) Small real safetensors base | **DEFERRED** | SI-18 §3.7 audit: A-1 to A-4 all still unmet; the local Glimmer checkpoint retires none of them (they are design obligations, not download obligations), and naming a base stays an operator decision (§3). |
| (b) The composition line (mean / worst-case / projection) | **DONE — closed as a line** | Three mechanisms falsified at the binding probe at increasing strength; F1′′ measured with the construction exact on its own population; RFC SI-30 §7.5's advance consequence is in force (§4). |
| (c) Another point on the penalty axes (λ / λ_w / k / τ) | **DEFERRED (open questions)** | Each varies a parameter of a mechanism whose shared premise SI-31 falsified at strictly greater strength (full removal); parameter variation after a premise falsification is tuning (§6). No sweep is licensed. |
| (d) The representation question (rank / identity-alone judging) | **ARGUED — DEFERRED with a trigger** | Both directions argued on their merits (§§7.2–7.3); the deciding fact is measured — SI-31's candidate fails identity independently at ~1.91× tol, so no verdict on record turns on rank; the question becomes a *change* only when a candidate passes identity + capability and fails only on rank (§7.5). |
| (e) Partial-removal projection | **REJECTED as the next slice** | Every β > 0 is generically rank 16 and inherits §7's refusal before identity is reached; and the binding probe's alignment interpolates linearly between two positive, REJECTED endpoints with no turning mechanism (§8, derivations labelled). |
| (f) Per-probe anchoring | **SPLIT, carried: gate-side REJECTED; probe-side training REJECTED; per-input accounting on allowed inputs ADOPTED inside SI-33** | The first two readings reprice or read the judge; SI-31 supplies no new evidence for either (§9). The third is §15's Phase 0 accounting. |
| (g) Probe-partitioned sequencing | **REJECTED (carried)** | Probe-derived partitioning reads the judge (invariant 9); signal-derived partitioning is SI-26 §10's gate-fitting search, and §5.2's opposite-sign component strengthens the rejection (§10). Not reopened. |
| (h) Magnitude discipline | **REJECTED as a slice (carried); ADOPTED as accounting** | Shrinking `m2` by budget, steps, or scale is the standing gate-fitting non-goal; SI-31's grow-while-failing candidate makes the norm accounting more mandatory, not less (§§5.3, 11). |
| (i) New LoRA target / additive / multi-target | **SPLIT: new target RECOMMENDED — SI-33; additive REJECTED as next slice (carried)** | One named non-readout target (`final-mlp-down`) changes the record's one unvaried premise — where the update acts — at standing budgets, with its deferral's price (derivation + finite-difference check) paid in Phase 0; additive still buys its premise with a policy loosening (§12). |
| (j) Transformer squash (miniature) | **DEFERRED** | Precondition still unmet — no authored two-delta stack exists after six measured attempts — and the squash gate context's obligations grow by SI-31's rank-16 construction and, prospectively, SI-33's target axis (§13.1). |
| (k) Glimmer probing | **REJECTED** | Rejected at SI-13 §7/§9 and restated in every successor RFC; SI-31 changes nothing, no forward path exists, and none is faked here (§13.2). Not reopened. |

The six routes to a landed second write that the project
has now run — a new signal under settled semantics
(SI-23), the same signal restacked (SI-21), new target
semantics (SI-25), a mean composition term inside the
loss (SI-27), a worst-input composition term inside the
loss (SI-29), and a candidate constructed by full
removal of the indicted component (SI-31) — have all
been refused by the same unchanged accounting, and the
last four refusals are attributed, with increasing
precision and now completely, to a property of readout
movement pairs: they oppose where measurement is lawful
and add where the gate prices, the indicted component
itself carrying opposite signs on the two populations.
§12's route is the seventh, and the first to stop
varying how a readout update is priced, anchored, or
constructed and instead vary *where an authored update
acts at all* — one tensor upstream, at the same budgets,
under the same judge, with its premise stated in
falsifiable form before it runs. No eighth route is
proposed by this RFC, and §§9–11 record the three
tempting routes (a per-probe judge, a partitioned
sequence, a shrunken movement) and why — on SI-31's
evidence as on SI-29's — they are still not candidates.

---

## 15. The recommended slice: SI-33 (outline)

**SI-33 — First non-readout authored target for the
miniature: signal 2, base-anchored, composition `"none"`,
trained at the landed head exactly as SI-25 trained it
except that its rank-8 LoRA factors parameterise the
final layer's MLP down-projection instead of the readout
— phased, with the backward derivation, the
finite-difference gradient check, and the successor
falsification readings published before any decisive
candidate is built.**

In scope:

1. **Phase 0 — the derivation, the control records, and
   the successor readings (§§5.5, 12.1).** From existing
   machinery and existing candidates, before any new
   training semantics are finalised: (i) the **backward
   derivation** for the `"final-mlp-down"` target —
   target-logits MSE differentiated through the readout,
   the final RMSNorm, the residual add, and the
   down-projection's LoRA factors, with the SwiGLU
   activation's output treated and justified as the
   map's fixed input for the factor gradients — stated
   in the as-built in the form a reviewer can check
   against the forward code line by line; (ii) the
   **control records the readings need**: the re-run
   SI-25 control's per-probe inner products with `m1`
   (all 8 identity probes, computed through candidate
   overlays as diagnostics on an already-judged
   candidate, labelled as such) and its allowed-input
   composition record republished under its own labels,
   so F2′′′'s transfer clause (item 8) has its comparison
   figures stated in advance rather than computed after
   the treatment is known; (iii) the slice's
   **pre-registered successor readings** F1′′′/F2′′′/
   F3′′′ (item 8), with F2′′′'s clauses separated as
   §5.5 requires and F3′′′'s thresholds carried from
   SI-29's rule in shape against the re-run SI-25
   control (its published held-out residual is
   1.043608588752459e-05, so the completeness boundary
   Phase 0 must state is 2.087217177504918e-05); and
   (iv) the **gradient-check protocol**: central finite
   differences over every one of the 768 factor entries,
   at base and at head `d0001`, tolerance 1e-5
   unchanged. If (i) cannot be derived in checkable
   form, or (iv)'s check fails when run in Phase 1, the
   slice stops there and publishes the failure as its
   result — the SI-28 §6.7 / SI-29 / SI-31 Phase-0
   discipline, applied a fourth time.
2. **Authorship target as a recorded configuration
   (§12.1):** a fourth recorded axis beside signal,
   anchoring, and composition, serving exactly
   `"lm-head"` (default) and `"final-mlp-down"`;
   constructor-level on `TinyTransformerMind`, settable
   between cycles by an explicit driver call in the
   shape of `set_authorship_target(...)`; validated with
   the existing refusal discipline at the constructor,
   the setter, the intent, and the `train_candidate`
   dispatch; the intent carries it, so the journal shows
   all four selections. Rank (8) and alpha (8) are
   pinned by the axis's definitions exactly as the
   signals pinned them; neither is separately settable,
   and the target tensor for each value is named by the
   definition, never by a caller-supplied tensor name.
   No autonomous, learned, depth-driven, verdict-driven,
   or probe-driven target selection exists, and no sweep
   over targets is run: exactly one non-default target
   exists in the slice.
3. **The training path (§12.1):** under
   `"final-mlp-down"`, `train_candidate` trains signal
   2's factors against SI-25's base-anchored
   target-logits MSE loss exactly — same loss form, same
   partitions, same 2,000 steps, learning rate 1.0, norm
   budget 0.9, correction scale 0.15, non-improvement
   refusal applied to the trained candidate's signal
   held-out loss as today — with the gradient taken
   through §15 item 1 (i)'s derivation into the
   down-projection factors. Under `"lm-head"` every
   training path is byte-for-byte what it is today under
   every anchoring and every composition configuration.
   No probe input, probe logit, held-out sequence, or
   probe-derived quantity enters the loss, the gradient,
   or the refusal under either target (asserted by test,
   §16 criterion 4). Composition in the decisive run is
   `"none"`: no penalty term and no construction step of
   any prior kind is blended with the target change
   (two mechanisms, two variables, one blended record is
   a non-goal, §17).
4. **The at-base record (in place of a coincidence
   configuration).** The prior slices' coincidence
   configuration — new machinery at base reproducing an
   established run exactly because the machinery must
   vanish there — has no analogue for a target change:
   the new machinery is not supposed to vanish at base;
   it is supposed to train a different tensor there.
   SI-33 therefore substitutes two separately labelled
   records, and the as-built must not blur them:
   (a) **default-target reproduction at base** — signal 2,
   base-anchored, target `"lm-head"`, at base reproduces
   the established at-base run exactly (blob
   `sha256:131cc242…45f4`, drift 0.16672966815531254,
   regression 0.2418281203135848), proving the axis's
   introduction did not move the existing path; and
   (b) **the new-target at-base run** — signal 2,
   base-anchored, target `"final-mlp-down"`, at base,
   published as a labelled at-base fact (its own blob,
   gate figures, and signal records), which is the
   target's first measurement and is *not* the slice's
   decisive claim and *not* a coincidence result.
5. **All controls, re-run inside the slice:** the SI-25
   control (signal 2, base-anchored, composition
   `"none"`, target `"lm-head"`) must reproduce SI-25's
   decisive candidate exactly (blob
   `sha256:646d1e8b…d675`, drift 0.3329036422073841) —
   it is the treatment's primary control, differing from
   the decisive run in exactly one configuration — and
   the SI-27, SI-29, and SI-31 controls must likewise
   reproduce exactly under their own configurations and
   the default target (blobs `sha256:1bb277ab…b637`,
   `sha256:11cab02e…b8db5`, `sha256:32d13f65…a5c1`;
   drifts 0.32509003952145576, 0.31215621903538704,
   0.4775387533009052), so the target axis's refactor is
   proven behaviour-preserving for every existing
   semantics, including the projection construction,
   before the new target's results are trusted.
6. **The gradient check (§15 item 1 (iv)):** the new
   target's analytic gradients pass central finite
   differences over every one of the 768 factor entries
   at base AND at the decisive head `d0001`, tolerance
   1e-5 unchanged, with the worst disagreements
   published at both heads. No decisive run is trusted
   before this check passes; if it fails, item 1's
   stop rule applies and there is no decisive claim.
7. **The decisive two-cycle run:** one reused
   `TinyTransformerMind` through `si.run.run_cycles` —
   cycle 1 exactly as SI-31's cycle 1 (signal 1, default
   anchoring, default composition, default target), then
   the driver explicitly reconfigures the same mind to
   signal 2, base-anchored targets, composition `"none"`,
   and authorship target `"final-mlp-down"` (four
   recorded selections) and runs cycle 2. Exactly two
   authored cycles are attempted and exactly one
   new-target decisive run is performed. Cycle 2's
   verdict is whatever the unchanged gate measures; **no
   ADMIT is required** (§16 criterion 7).
8. **Diagnostics, the seven-run table, and the
   falsification readings:** the full post-verdict
   diagnostic set against head `d0001` with the decisive
   candidate overlaid — per-probe drift contributions,
   the alignment diagnostic (aggregate and at the worst
   probe, sign agreement never alone), folded updates
   (published as weight-space figures under the new
   target's labels, with no readout-fold derivation
   claimed for them), composed capability — published
   for the treatment beside the SI-25 control in the
   six-run table's form, extended to seven runs;
   candidate norm and ‖m2‖ beside every drift figure
   (§5.3); the treatment's signal held-out record read
   under the pre-registered F3′′′ rule; and the readings
   stated as measured, whichever way they fall:
   - **F1′′′ — the geometry survives the fold.** The
     treatment's composed maximum stays above tol 0.25
     AND its probe inner product at `tiny-id-0` — the
     probe the record's figures name — is positive
     (aligned). Falsified premise: that acting upstream
     of the readout changes the binding probe's
     alignment, for this signal, at this head.
     Consequence stated in advance: the alignment is
     then not a readout-fold property but a property of
     the signal family's corrections through this model,
     and no target change *within the final layer* is a
     frontier without a new argument.
   - **F2′′′ — the maximum is immobile even when the
     binding probe turns.** Two separately reported
     clauses, in the repaired form: (a) the *transfer
     clause* holds iff the treatment's probe inner
     product at its own worst probe is smaller than the
     SI-25 control's inner product at that same probe,
     as published in Phase 0 (item 1 (ii)); (b) the
     *maximum clause* holds iff the composed maximum
     stays above tol 0.25. F2′′′ **is measured** iff
     (a) and (b) both hold; each clause's outcome is
     reported either way, and the aggregate probe
     composition is published beside SI-25's, SI-27's,
     SI-29's, and SI-31's figures under their own labels.
   - **F3′′′ — the destination is traded away.** Read
     under SI-29's three clauses unchanged in their
     thresholds, against the re-run SI-25 control: the
     treatment's destination counts as **approached**
     (F3′′′ does **not** fire) iff (a) its signal
     held-out loss falls, (b) at least 12 of 16 held-out
     examples are closer, and (c) its held-out residual
     is at most 2× the re-run SI-25 control's residual —
     boundary 2.087217177504918e-05, stated in Phase 0
     before the decisive numbers. If any clause fails,
     the destination counts as **traded away** and
     F3′′′ **is measured**, with the trade's shape
     published beside the reading.
   Worst probes are named; worst *allowed inputs* are
   named by train-partition identity and never confused
   with probes (§18).
9. **VERIFY and unwind, honestly conditioned** (§16
   criterion 10): cycle 1 VERIFY as in
   SI-19/21/23/25/27/29/31. If cycle 2 ADMITs and VERIFY
   passes, the authored two-delta stack exists for the
   first time and SI-21's fixture-labelled unwind
   evidence is joined by the real thing — claimed
   exactly as measured, no further, and with the second
   delta labelled a non-readout-target candidate
   wherever the stack is described (§18 item 24). If
   cycle 2 REJECTs, no stack exists, nothing is unwound,
   and none is claimed — the posture of all six prior
   runs, unchanged.

---

## 16. Acceptance criteria for SI-33

1. **Recorded authorship-target configuration, two named
   values** — the configuration serves exactly
   `"lm-head"` and `"final-mlp-down"`; every other value
   refuses with the existing `ValueError` discipline at
   the constructor, the setter, the intent, and the
   `train_candidate` dispatch. Recorded selection is
   asserted per cycle on both verdict paths. The target
   tensor, rank (8), and alpha (8) for each value are
   fixed by the configuration's definition and are not
   separately settable (asserted: no caller-facing
   tensor-name, rank, or alpha argument exists on the
   axis); exactly one new-target decisive run is
   performed (no sweep, no second target).
2. **Phase 0 published before the decisive run is
   trusted** — the backward derivation in checkable
   form, the SI-25 control's per-probe inner-product
   table and allowed-input record (labelled diagnostics
   and control figures), the pre-registered
   F1′′′/F2′′′/F3′′′ reading rules with F2′′′'s clauses
   separated and F3′′′'s boundary stated
   (2.087217177504918e-05), and the gradient-check
   protocol — all present in the as-built ahead of the
   decisive numbers.
3. **Gradient correctness, at both heads** — the new
   target's analytic gradients pass central finite
   differences over every one of the 768 factor entries
   (B [32, 8], A [8, 64]) at base and at head `d0001`,
   tolerance 1e-5 unchanged, worst disagreements
   published for both heads. The readout path's existing
   checks stand cited for the `"lm-head"` path, not
   silently assumed and not re-run in the new target's
   place.
4. **Probe-blindness of training, asserted** — the train
   partition's disjointness from both probe suites is
   re-asserted under the new code path; a
   perturbed-probe-artifact bundle copy produces a
   byte-identical *new-target* candidate; and the
   training path under `"final-mlp-down"` is asserted to
   read no probe artifact (the gradient's inputs are the
   signal partitions and the landed head's weights,
   named as a set where the machinery permits).
5. **Default-target reproduction and the at-base
   records** — signal 2 base-anchored under `"lm-head"`
   at base reproduces the established at-base run exactly
   (blob `sha256:131cc242…45f4`, drift
   0.16672966815531254, regression 0.2418281203135848);
   the `"final-mlp-down"` at-base run is published as a
   labelled at-base fact under its own blob and figures,
   and is nowhere described as a coincidence result or
   as the slice's decisive claim.
6. **Determinism per cycle** — a full decisive run
   repeated from the seed reproduces both candidates'
   blobs/records (the new-target candidate's blob
   included), the landed delta record, and the journaled
   gate measurements exactly; hyperparams equal except
   the labelled `wall_time_seconds`.
7. **Gate honesty per cycle** — each cycle's journaled
   checks 3/4 equal an independent `ProbeContext`
   evaluation of the same candidate against the
   pre-cycle head (abs 1e-12) — for cycle 2, of the
   new-target candidate, the only candidate the gate is
   ever shown. Cycle 2's verdict is reported **as
   measured**; no ADMIT is required: SI-33 succeeds by
   §12's claim — a non-readout-target candidate was
   genuinely authored against the landed head under the
   new configuration and genuinely judged, with its
   premise's falsification readings published either way.
8. **Cumulative accounting, diagnostics, the seven-run
   table, and the falsification readings published** —
   the side-by-side accounting table (seed-absolute /
   cumulative identity; head-incremental capability),
   the full diagnostic set with candidate norm and ‖m2‖
   beside every drift figure, the treatment's signal
   held-out record read under the pre-registered F3′′′
   rule, the seven-run table with each figure under its
   own run's labels (the treatment's row carrying its
   authorship target), and the F1′′′/F2′′′/F3′′′
   readings with F2′′′'s clauses reported separately —
   all published, measurements and derivations labelled,
   worst probes named, worst allowed inputs named as
   allowed inputs, nothing in the diagnostic set reported
   "not evaluated" that was computable.
9. **First-cycle and all-control reproduction** —
   configured with signal 1 (default signal, default
   anchoring, default composition, default target), a
   first cycle reproduces SI-19/SI-21/SI-23/SI-25/
   SI-27/SI-29/SI-31 exactly (blob `sha256:2761e8ff…6ca9e`,
   drift 0.17189218290150166, regression
   0.1657181391492486); the re-run `"none"` / `"lm-head"`
   control reproduces SI-25's decisive candidate exactly;
   the re-run `"movement-composition"`, `"worst-input-
   composition"`, and `"train-partition-projection"`
   controls reproduce SI-27's, SI-29's, and SI-31's
   decisive candidates exactly — the target-axis
   machinery proven behaviour-preserving for every
   existing semantics before the new target's results
   are trusted.
10. **VERIFY, unwind, and untouched scope** — cycle-1
    VERIFY as in SI-19; unwind honestly conditioned (no
    authored stack unless cycle 2 lands; none unwound or
    claimed otherwise). Exactly two authored cycles
    attempted in the decisive run and exactly one
    new-target decisive run performed; `max_stack_depth`
    8, `identity_tol` 0.25, `capability_eps` 0.25,
    `max_targets_per_delta` 1, `allow_additive` false,
    `rank_cap` 8 and every other policy default unchanged
    (asserted); both signals' ids, streams, corrections,
    and partitions unchanged (asserted — the target
    pinning's move to the recorded axis is the one
    deliberate definition change, §12.1, and no other
    part of either signal's definition moves); both
    anchorings' and all four composition configurations'
    definitions unchanged (asserted); `si squash` still
    fails closed on the resulting transformer stack
    (asserted; SI-12's suite green); the full existing
    suite is green; the demo path is unchanged; `si
    serve` still returns 501 for every transformer bundle
    (existing suite); no CLI wiring added.

---

## 17. Non-goals

Explicit, and binding on SI-33 and on any slice this RFC
is cited to justify:

- **No Glimmer anything.** No Glimmer forward path,
  probes, training, or authorship (§13.2). No candidate
  in this RFC touches the Glimmer seed except to restate
  that it is untouched, and §12's target axis is a
  miniature mechanism that claims nothing at Glimmer
  scale.
- **No real-base work.** Candidate (a) is deferred with
  audited preconditions (§3); SI-33 builds nothing toward
  it, and no download is performed or scheduled.
- **No composition revival.** The decisive run uses
  composition `"none"`. No penalty term (mean, worst-case,
  or any third statistic), no projection (full or
  partial), and no blend of any of them with the target
  change is built, run, or reported as the treatment.
  The four existing composition configurations' semantics
  are untouched and serve as controls only.
- **No rank or policy change.** `rank_cap` stays 8
  (§7.5); no tolerance, cap, or policy default moves for
  any candidate class; no candidate is "judged on
  identity alone" under this RFC, and SI-31's candidate
  is never re-judged.
- **No second new target, no target sweep.** Exactly one
  non-readout tensor is named (§12.1). No attention
  projection, embedding, norm, bias, or layer-0 tensor is
  trained, checked, or reported as anything but a future
  RFC's unargued candidate; a future target is a future
  RFC's single argued tensor, argued from SI-33's
  published record, never a sweep inside a slice.
- **No third signal, no signal redefinition.** Both
  signals' ids, streams, corrections (scale 0.15
  included), and partitions are byte-for-byte as
  SI-19/SI-23 defined them. The target pinning's move to
  the recorded axis (§12.1) is the single deliberate
  definition change this RFC argues; nothing else about
  either signal moves, silently or otherwise.
- **No probe contact in training.** No probe input, probe
  logit, probe movement, or probe-derived quantity
  enters any training loss, gradient, initialisation,
  stopping rule, sequencing decision, or candidate
  selection, under either target (§§2, 9). Phase 0's
  control diagnostics read probes only as post-verdict
  accounting on already-judged candidates, as diagnostics
  always have.
- **No gate change of any kind.** The gate's checks,
  verdict mapping, accounting, tolerances, caps, and
  every policy default are unchanged (§9). If the
  new-target cycle is REJECTed, the answer is a later
  RFC's better argument — never a wider, repriced,
  per-probe, or quieter gate.
- **No parameter tuning.** No sweep of λ, λ_w, k, τ,
  β, rank, alpha, step count, learning rate, norm budget,
  or correction scale (§§6, 8, 11); no second decisive run
  at a different value of any of them. A future parameter
  is a future RFC's argued, single, pre-registered value.
- **No additive or multi-target authorship** (§12.2):
  `max_targets_per_delta` stays 1 and `allow_additive`
  stays false, and the new target is a *replacement*
  parameterisation for a cycle, never an additional one.
- **No autonomous selection — of signal, of anchoring,
  of composition configuration, or of authorship
  target.** The mind chooses none of them; all four are
  explicit, recorded, driver-level configurations, and no
  perception-driven, depth-driven, alternating,
  verdict-driven, or worst-probe-driven rule is built or
  implied for any axis.
- **No transformer squash.** SI-12's fail-closed refusal
  stands throughout SI-33 (§13.1); §13.1 is a deferral
  analysis, not an implementation brief — even if SI-33
  creates the authored stack squash has been waiting for.
- **No serving, no CLI wiring.** Transformer bundles keep
  returning 501; `si cycle`/`si run` keep their demo/mock
  minds. SI-33 is a library-and-tests slice like SI-19,
  SI-21, SI-23, SI-25, SI-27, SI-29, and SI-31.
- **No third cycle, no continual-learning claim.**
  Exactly two authored cycles are attempted in the
  decisive run, on constructed signals, on a constructed
  model, under the recorded configurations §15 scopes.
  Four configuration axes at depth ≤ 2 prove something
  about the loop's machinery and its gate — nothing about
  learning at any other scale or on any real task.
- **No mixed-target stack claim.** The decisive run's
  cycle 1 is a readout-target cycle and its cycle 2 is a
  non-readout-target cycle by design; SI-33 does not
  train a non-readout cycle at a non-base head inside the
  decisive run as a claimed stack member beyond its
  cycle 2, and no claim is made about stacks whose cycles
  mix authorship targets at non-base heads beyond the
  one stack the decisive run itself produces, if it
  produces one.

---

## 18. Honesty and labelling requirements for any future slice

Items 1–23 restate RFC SI-30 §17 (itself restating
SI-26 §17 and SI-24 §12 with additions) in compressed
form, with the numbering preserved so existing citations
resolve; items 24–25 are new in this RFC. All twenty-five
bind SI-33 and any slice this RFC is cited to justify.

1. **Training numbers are training numbers.** Losses,
   penalty, objective, gradient, and construction values
   are labelled with signal, partition, head, target
   source, anchoring, composition configuration, and
   authorship target; none is a gate or probe result.
2. **Measured vs not evaluated survives authorship.**
   "Not evaluated" is never rendered as 0.0, a pass, or
   an omission.
3. **Authorship is not admission.** No document, test
   name, or journal field describes a candidate as landed
   before the gate lands it.
4. **Every number carries its source.** Each figure names
   the run, configuration, and document that measured it;
   inherited figures are cited, not re-derived silently.
5. **The miniature's labels travel.** "Constructed,
   deterministically seeded, not trained" accompanies the
   miniature wherever its results are summarised.
6. **Failure states are reportable successes of the
   experiment.** A REJECT with attribution is a
   publishable result, stated as plainly as an ADMIT.
7. **Serialisation determinism is proven, not assumed.**
   Blob-hash equality is the evidence; near-equality is a
   failure, reported as one.
8. **Head-relative numbers name their head.** Any figure
   measured against a head names that head.
9. **Incremental and cumulative are different numbers.**
   Never presented as each other, in either direction.
10. **Composed artifacts do not inherit training claims.**
    A landed stack's properties are what its VERIFY
    measured, not what its cycles' training records hoped.
11. **Signal identity is part of every label.** A number
    without its signal id is a labelling defect.
12. **Diagnostics are not verdicts.** The diagnostic set
    attributes a verdict after the gate has spoken; it is
    not a gate check, it influences no verdict, and none
    of it is a prediction — including Phase-0 diagnostics
    on already-judged candidates that point the same way
    as a later decisive run.
13. **Recorded selection is not a choice the mind made.**
    Signal, anchoring, composition configuration, and
    authorship target are driver selections; no text
    describes the mind as choosing, deciding, or
    preferring any of them.
14. **A derivation is not a prediction.** Derived
    quantities (movement columns, sums at a probe,
    interpolation arguments like §8.3's, rank arithmetic
    like §8.2's) are labelled derivations wherever they
    appear, and are never quoted with the authority of a
    measurement.
15. **Anchoring is part of every label.** A training or
    diagnostic number recorded under one anchoring is
    never presented as a number under the other.
16. **Controls and treatments are never blended.** Each
    figure stands under its own run's labels; a three-,
    four-, five-, six-, or seven-run table keeps the rows
    separate.
17. **Composition configuration is part of every label.**
    A number recorded under one composition configuration
    is never presented as a number under another.
18. **Allowed-input numbers are not probe numbers.**
    Figures from allowed populations and figures from
    judged populations are never interchanged, averaged
    together, or quoted in each other's place; any gap
    between them is stated as the difference between two
    separately labelled records.
19. **Sign agreement never stands alone.** It is published
    only beside the magnitude-weighted figure (the inner
    product) for the same population.
20. **A worst-case figure names its population.** A worst
    *allowed input* (train partition, held-out partition)
    and the worst *probe* are different quantities over
    disjoint populations, and no text may shorten either
    to "the worst case" without its population attached.
21. **Judgement-call readings are pre-registered.** Where
    a falsification reading turns on a qualitative boundary
    in a quantitative record, the boundary is stated in
    advance, in the record's own terms, before the decisive
    run it will judge — and the reading is then applied as
    written. A reading invented after the numbers are known
    is reported as an observation, never as the
    falsification test.
22. **A constructed candidate is labelled constructed.**
    A candidate produced by a construction step after
    training is labelled with that step wherever it, its
    stack, or its figures are described; its
    pre-construction intermediate's figures are labelled an
    intermediate's and are never presented as the judged
    candidate's record.
23. **An unmeasured-as-defined reading stays unmeasured.**
    A falsification reading whose registered definition
    did not fire — F2′ and F2′′ are the standing instances —
    is quoted in its registered form ("not measured as
    defined", with the clause that failed named) and never
    rounded, in summary, citation, or successor documents,
    into "measured", into "not measured" without its
    qualifier, or into the separate facts measured beside
    it.
24. **Authorship target is part of every label (new).**
    A number recorded under one authorship target — for any
    slice this RFC is cited to justify, `"lm-head"` or
    `"final-mlp-down"` — is never presented as a number
    under another; figures from the six-run record are
    understood, and where summarised stated, as recorded
    under `"lm-head"`. Folded-update and weight-space
    figures for a non-readout candidate are published as
    weight-space figures only: no readout-fold derivation
    (of α, of movement factorisation, or of composition)
    is claimed for a candidate whose update acts upstream
    of the readout.
25. **A two-line refusal is reported on both lines
    (new).** A candidate the gate refuses on two
    independent lines — SI-31's projected candidate
    (safety rank and identity) is the standing instance —
    is described as failing on both wherever its verdict
    is summarised; quoting §7.5's deferred representation
    question as softening, explaining away, or partially
    overturning SI-31's verdict is a labelling defect,
    because its identity failure is independent of the
    rank question and stands under any resolution of it.

---

## 19. Risks and open questions

Each item names what would retire it. Numbers this RFC
deliberately does **not** supply are listed as open
questions to measure — supplying them here would be
inventing them.

1. **F1′′′ is a live possibility, and the slice must be
   built to publish it.** The alignment at the binding
   probe has survived four training semantics and one
   exact construction; §5.2 attributes it, at readout
   level, to a component that carries opposite signs on
   the allowed and judged populations. If that component
   is a property of the signals' *corrections* rather
   than of the fold they pass through — the corrections
   are defined in logit space, and an upstream update
   trained to produce the same logit shift may reproduce
   the same geometry by a different route — then acting
   upstream changes the parameterisation and not the
   outcome, and F1′′′ fires. This RFC recommends the
   experiment because the fold is the record's one
   unvaried premise and because the check is cheap
   against published controls with exactly one variable
   changed — not because transfer has become more
   likely. A slice that reports F1′′′ as a surprise has
   misread its own RFC. *Retired by:* SI-33's F1′′′
   reading, published either way (§16 criterion 8).
2. **The backward derivation could be wrong in a way the
   loss curve hides.** A subtly wrong gradient through
   the final norm or the residual add can still descend
   — the trained candidate would then be a candidate
   trained by an unnamed optimiser, and every figure
   under its labels would be mislabelled at the source.
   This is the new-target deferral's first ground
   (§12.1) returning as the slice's central
   implementation risk. *Retired by:* §16 criterion 3 —
   central finite differences over all 768 factor
   entries at both heads, tolerance 1e-5, worst
   disagreements published; a failure stops the slice
   before any decisive claim (§15 item 1).
3. **The destination may be unreachable through one
   upstream tensor (F3′′′).** Signal 2's correction is a
   logit-space lean; the readout path reaches a
   held-out residual of 1.043608588752459e-05 with the
   whole [32, 32] readout to move, while the new target
   must express the same lean through the final layer's
   MLP, the residual stream, the final norm, and a
   *landed* readout it does not control. Whether it
   approaches at all under thresholds this RFC
   deliberately carries forward unchanged — with a
   completeness boundary (2.087217177504918e-05) far
   stricter, relative to the SI-25 control, than the
   boundary any composition treatment faced — is
   genuinely unknown in both directions. *Retired by:*
   SI-33's held-out record read under the pre-registered
   rule (§§15, 16 criterion 8).
4. **The target-axis refactor could silently alter an
   existing path.** Generalising `train_candidate` to
   dispatch on a fourth axis is the kind of change that
   passes its own new tests while moving an old
   behaviour in its last digits — now with four existing
   composition semantics and two anchorings to protect.
   *Retired by:* §16 criteria 5 and 9 — exact
   reproduction of six published blobs (signal 1's
   `d0001`, signal 2's at-base `d0001`, and the SI-25,
   SI-27, SI-29, and SI-31 decisive candidates) under the
   refactored path, before the decisive run is trusted.
5. **The representation trigger may fire inside SI-33 —
   and must not be improvised around.** If a lawful
   future candidate (SI-33's is rank 8 by construction,
   so not this one) ever passes identity and capability
   while failing only on rank, §7.5's question becomes a
   change proposal. What no slice may do is resolve it
   *inside* itself: the argument belongs to its own RFC,
   argued against §7.3 with the candidate's figures
   published, and never applied retroactively to SI-31.
   *Retired by:* that RFC, if the trigger ever fires;
   until then this item stays open by design.
6. **Partial removal stays unmeasured — deliberately.**
   §8 rejected it on a derivation and on rank arithmetic,
   not on a measurement, and §18 item 14 keeps those
   grounds labelled as derivations. If §7.5's trigger
   ever fires and the rank ground falls, §8.3's
   interpolation ground would remain as the argument a
   future partial-removal RFC must answer with something
   other than a fraction chosen between two priced
   endpoints. *Retired by:* nothing in SI-33; a future
   RFC arguing a β from evidence that does not yet exist.
7. **A mislabelled configuration is worse than a wrong
   one.** The failure mode the cadence has named since
   SI-26 —
   a candidate trained under one configuration and
   journaled under another — now has four axes to confuse
   instead of three, and one subtler variant this slice
   introduces: a candidate trained on the new target and
   journaled under the default (or the reverse) would
   attribute an upstream effect to a readout run, or a
   readout result to the new target, in exactly the
   seven-run table §15 exists to publish. *Retired by:*
   §16 criteria 1, 5, and 7 — per-cycle journal
   assertions on both verdict paths, the default-target
   at-base reproduction, all four controls' exact
   reproduction, and the probe-perturbation independence
   test applied to the new-target candidate.
8. **Probe blindness still bounds the judge.** RFC SI-13
   §10.1 stands, now across signals, anchorings, four
   composition configurations, and prospectively two
   authorship targets: 24 inputs, 28/30 tensors
   measurable, blind set derived and guarded. A landed
   two-delta stack under any configuration is *evidence*
   of continuity sampled the same way each cycle, not a
   proof of it. The new target is outside the blind set
   (§12.1), but "outside the blind set" prices this
   target's measurability, not its importance: the
   probes sample the model's behaviour at 24 inputs,
   whichever tensor moved it.
9. **Depth and Glimmer pressure arrive if SI-33
   succeeds.** At `max_stack_depth` 8 with transformer
   squash still deferred (§13.1), a landed second
   authored delta makes the stack depth 2 and the
   compression question real for the first time — while
   the squash gate context remains unbuilt and must now
   be designed against four recorded axes. And per the
   standing item in every successor RFC: success on the
   miniature renews the "why not Glimmer" pressure, and
   the answer does not soften because the miniature's
   authorship has gained a target. There is no Glimmer
   forward path; none is proposed here, and none may be
   improvised by a successor slice (§13.2).
10. **The deferred axes' unknowns remain unknown.**
    Candidate (a)'s A-1 to A-4 are audited unmet (§3);
    nothing in SI-33 or this RFC scopes them, and the
    download question is an operator decision this RFC
    explicitly does not take. The λ / λ_w / k / τ
    questions (§6) likewise remain open by design: each
    is retired only by a later RFC arguing a single value
    from the published records of the configurations that
    now exist, never by a sweep. The τ form's Phase-0
    scale derivation remains a derivation, unchecked
    against an implemented term (§6), and no item in
    SI-33 depends on it.

---

## 20. Summary

After SI-31, SI can originate a miniature delta, judge
it, land it, and originate a second one on top of it
under a second signal, under two target anchorings,
under either of two composition-aware losses, and under
a construction that removes the indicted component
exactly — and the gate's cumulative, seed-absolute
identity accounting has been measured refusing all six
second writes, the sixth on two independent lines and by
the widest identity margin on record: the projected
update *opposes* the first update on every allowed
population it was measured on — its train-partition
composition zeroed to −4.8567197048765554e-08 — and the
composed maximum still stands at 0.4775387533009052 >
tol 0.25, because at `tiny-id-0` the two movements add
almost exactly (0.17189218290150166 + 0.3063734769821167
≈ 0.4775), the component removed on allowed inputs being
the component that was restraining the movement at the
binding probe, its alignment there +0.14235826783471692,
the worst on record. F1′′ is measured; F2′′ is not
measured as a reading, its transfer clause having failed
while the maximum stayed unmoved beside it; F3′′ is
measured, by its completeness clause. Of the candidate
frontiers this RFC evaluated against that evidence: a
small real base stays deferred, its four SI-18
preconditions audited and all still unmet (§3); the
composition line is done and closed as a line — mean,
worst case, and exact construction all falsified at the
binding probe (§4); another point on the penalty axes
stays deferred as open questions no current evidence
argues, with no sweep licensed (§6); the representation
question is argued in both directions and deferred with
a precise trigger, because SI-31's candidate fails
identity independently at ~1.91× the tolerance and no
verdict on record turns on rank (§7); partial-removal
projection is rejected as the next slice, inheriting
the rank refusal at every nonzero fraction and
interpolating, at the binding probe, between two
aligned REJECTED endpoints with no turning mechanism
(§8, derivations labelled); per-probe anchoring stays
rejected in its gate-side and probe-side readings and
is adopted only as per-input accounting on allowed
inputs inside the recommended slice (§9);
probe-partitioned sequencing stays rejected on the
invariant and on the gate-fitting trap (§10); magnitude
discipline stays rejected as a training dial and is
adopted as mandatory accounting (§11); additive
authorship stays rejected (§12.2); transformer squash
stays deferred, its precondition still unmet after six
measured attempts (§13.1); Glimmer probing stays
rejected, not reopened, with no forward path existing
and none faked (§13.2); and **a first non-readout
authored target — signal 2, base-anchored, composition
`"none"`, trained at the landed head exactly as SI-25
trained it except that its rank-8 factors parameterise
the final layer's MLP down-projection
(`model.language_model.layers.1.mlp.down_proj.weight`)
instead of the readout, recorded as a second value on a
new authorship-target axis beside signal, anchoring, and
composition, phased so the backward derivation, the
768-entry finite-difference gradient check at base and
at `d0001`, the control records, and the successor
falsification readings — with F2′′′'s clauses separated
and F3′′′'s thresholds carried in shape from SI-29's
rule against the SI-25 control — are all published
before any decisive candidate is built, with the
candidate judged by the unchanged gate on probes its
training never reads — is recommended as SI-33**
(§§12, 15, 16). Its plausibility argument is stated
with its limits in the same sections that make it:
every run in the record varied how a readout update is
anchored, priced, or constructed, and none varied where
an update acts; the fold both readout movements share
is the one premise the complete composition attribution
never tested, and the final layer's down-projection is
the upstream tensor whose honest price — a derivation
and a gradient check — is shortest. Whether acting
upstream changes the geometry the six runs indicted is
measurable with instruments that already exist, against
six published controls, with exactly one variable
changed. That is a plausibility argument for running
the experiment — nothing in this section is a prediction
that it ADMITs, and §16 requires no ADMIT; §19 item 1
records that the alignment may be a property of the
signals' corrections rather than of the fold, in which
case F1′′′ fires and the authorship line, not only the
composition line, will need a genuinely new argument.
The deepest landed authored stack remains depth 1; no
two-delta authored stack exists; the base stays frozen,
the bytes are never duplicated, no tolerance or cap
moves, the gate is unchanged, and nothing is claimed
that a test in this repo cannot check.

---

*RFC SI-32 — design only. Implementation begins, if
approved, at SI-33 as outlined in §15. RFC SI-26 changed
what the loop accounts for while it trains; RFC SI-28
changed the statistic that accounting answers to; RFC
SI-30 changed the candidate itself, removing the
component the record indicted; the gate told the truth
about all three, per probe — and about the removal it
told the sharpest truth of the record: the component
that opposes where measurement is lawful is the
component that restrains where the gate prices. This
RFC proposes changing neither what the loop accounts
for, nor the statistic it answers to, nor the candidate
it constructs from a readout update, but where an
authored update acts in the first place: one tensor
upstream of the readout every prior run shared, trained
by a derivation this RFC requires to be checked entry by
entry before it is trusted, and judged — under the same
judge, at the same scale, with the verdict still not
ours to pick.*
