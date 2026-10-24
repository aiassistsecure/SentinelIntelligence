# SI-30 RFC — The Next Frontier After Worst-Input Composition-Aware Training
(Design Only)

> **Status: DESIGN ONLY.** This document evaluates candidate
> directions and recommends one next slice. It changes no
> production code, no test, no manifest schema, no policy
> default, and no gate behaviour. Nothing in this RFC is
> implemented unless a section explicitly says it exists today,
> in code, on `main`.
>
> **The one sentence that governs everything below:** SI-29
> proved that composition measured on the inputs training is
> allowed to see can be shaped — decisively, by a worst-input
> penalty term — at the *worst case* of those inputs as well as
> in aggregate, the tail–bulk gap on allowed inputs closed,
> and that the gate's verdict still did not move, because the
> gate prices the *worst* probe and at that probe,
> `tiny-id-0`, the worst-input candidate's movements align
> *more* than under either composition predecessor (inner
> product +0.04944786620153699, against SI-27's
> +0.03801222744105431 and SI-25's +0.0338873784299374), the
> composed maximum is 0.31215621903538704 > tol 0.25, RFC
> SI-28's falsification reading F1′ is *measured*, F2′ is
> *not measured as defined*, F3′ is *not measured under the
> pre-registered rule* — and this RFC must never be cited as
> evidence that any projection has been implemented or
> tested, that any λ_w, k, or τ other than λ_w = 1.0,
> k = 4 has been measured, that any authored two-delta stack
> exists, that a cycle-2 ADMIT is achievable by any route, or
> that any slice proposed here exists.
>
> **Scope discipline:** CPU only. No GPU, no cloud, no spend,
> no downloads. Terminology: this is machine learning
> infrastructure throughout.
>
> **Relationship to prior work:** RFC SI-28 evaluated the
> frontiers open after SI-27 and recommended worst-input
> composition-aware training; SI-29 built it (PR #34,
> as-built in
> `docs/SI-29-worst-input-composition-training.md`) and the
> gate's cycle-2 verdict was a measured REJECT under the
> worst-input penalty as it had been under the mean penalty
> and under both anchorings without one — now attributed
> more sharply than before: the worst-case statistic works
> where it is allowed to look, and fails at the binding
> probe by a wider margin than either predecessor. This RFC
> is SI-28's successor in the established cadence (SI-18 RFC
> → SI-19 build → SI-20 RFC → SI-21 build → SI-22 RFC →
> SI-23 build → SI-24 RFC → SI-25 build → SI-26 RFC →
> SI-27 build → SI-28 RFC → SI-29 build): it restates the
> verified state after SI-29, re-evaluates the standing
> candidates against SI-29's measured evidence, gives the
> question SI-29's diagnostics have made precise — why a
> penalty that closes the tail–bulk gap on allowed inputs
> still loses, and loses worse, at the one probe the gate
> prices, and why the composed maximum has now barely moved
> across five measured REJECTs — the deep treatment (§5),
> and recommends exactly one next slice (§14). SI-28's
> rejections of per-probe anchoring in its gate-side and
> probe-side readings (§8 there), of probe-partitioned
> sequencing (§9 there), of magnitude discipline as a
> training dial (§10 there), and of Glimmer probing (§12.2
> there) are carried forward, not reopened: §8, §9, §10,
> and §12.2 of this RFC record, per candidate, that SI-29
> supplies no new evidence that reopens any of them, and
> they stay closed. One candidate SI-28 explicitly left
> arguable *is* taken up — the projection form of
> train-partition orthogonalization, which SI-28 §7
> deferred on sequencing behind the worst-input slice and
> which RFC SI-28 §6.6 named, in advance, as the
> composition line's remaining honest move if the
> worst-case statistic failed to price the judged maximum.
> SI-29 measured that failure's shape (F1′ measured; the
> maximum immobile beside F2′ unmeasured as defined), and
> §7 evaluates the projection form on its merits and
> recommends it. The SI-9 RFC's §7 risks remain in force
> and are inherited, not relitigated.

---

## Table of contents

1. [Where SI stands after SI-29](#1-where-si-stands-after-si-29)
2. [Inherited invariants](#2-inherited-invariants)
3. [Candidate (a): a small real safetensors base — still deferred](#3-candidate-a-a-small-real-safetensors-base--still-deferred)
4. [Candidate (b): worst-input composition-aware training — DONE as SI-29, closed](#4-candidate-b-worst-input-composition-aware-training--done-as-si-29-closed)
5. [The non-transfer and immobility questions — the deep treatment](#5-the-non-transfer-and-immobility-questions--the-deep-treatment)
6. [Candidate (c): another point on the worst-input axis — a different λ_w, a different k, or the τ form](#6-candidate-c-another-point-on-the-worst-input-axis--a-different-λ_w-a-different-k-or-the-τ-form)
7. [Candidate (d): the projection form — train-partition orthogonalization](#7-candidate-d-the-projection-form--train-partition-orthogonalization)
8. [Candidate (e): per-probe anchoring — three readings, three verdicts, none reopened](#8-candidate-e-per-probe-anchoring--three-readings-three-verdicts-none-reopened)
9. [Candidate (f): probe-partitioned sequencing](#9-candidate-f-probe-partitioned-sequencing)
10. [Candidate (g): magnitude discipline — a smaller second movement](#10-candidate-g-magnitude-discipline--a-smaller-second-movement)
11. [Candidates (h) and (i): a different λ for the mean penalty; new target / additive](#11-candidates-h-and-i-a-different-λ-for-the-mean-penalty-new-target--additive)
12. [Candidate (j): transformer squash; candidate (k): Glimmer probing](#12-candidate-j-transformer-squash-candidate-k-glimmer-probing)
13. [Decision summary](#13-decision-summary)
14. [The recommended slice: SI-31 (outline)](#14-the-recommended-slice-si-31-outline)
15. [Acceptance criteria for SI-31](#15-acceptance-criteria-for-si-31)
16. [Non-goals](#16-non-goals)
17. [Honesty and labelling requirements](#17-honesty-and-labelling-requirements-for-any-future-slice)
18. [Risks and open questions](#18-risks-and-open-questions)
19. [Summary](#19-summary)

---

## 1. Where SI stands after SI-29

Everything in this section is verifiable in the code and
docs on `main` at the SI-29 merge (main tip `b0ba756`,
the merge of PR #34). Numbers are cited to the as-built
document that measured them; this RFC measures nothing
new. Where a statement below is a *derivation* from cited
facts rather than a measurement, it is labelled a
derivation, exactly as SI-28 §1 labelled its arithmetic.

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
  SI-23), unchanged.** The signal registry
  (`SIGNAL_REGISTRY`, `signal_spec` in
  `si/tiny_training.py`) serves exactly two named signals
  — `tiny-successor-lean-001` (SI-19) and
  `tiny-predecessor-lean-001` (SI-23) — and refuses every
  other id with the existing `ValueError` discipline.
  Selection is explicit and recorded; no autonomous
  selection exists. Both signals share one form
  (target-logits MSE), one LoRA target
  (`lm_head.weight`, rank 8, alpha 8), correction scale
  0.15, 2,000 full-batch steps, learning rate 1.0, norm
  budget 0.9, 32 train / 16 held-out examples; they differ
  only in correction direction, stream, and drawn
  sequences.
- **Two recorded target anchorings (SI-25), unchanged.**
  `TARGET_ANCHORINGS` serves exactly
  `"weights-under-training"` (the default, SI-21's
  semantics) and `"base-anchored"` (SI-25), validated by
  `validate_target_anchoring(...)` with the existing
  refusal discipline (`si/tiny_training.py`). Anchoring
  is a recorded per-cycle configuration beside recorded
  signal selection.
- **Three recorded composition configurations (SI-29).**
  `COMPOSITION_CONFIGS` serves exactly `"none"` (the
  default), `"movement-composition"` (SI-27), and
  `"worst-input-composition"` (SI-29), validated by
  `validate_composition_config(...)` with the existing
  refusal discipline (`si/tiny_training.py`). Under
  `"movement-composition"` the training loss gains
  exactly one term — λ · mean over the signal's train
  partition of the per-coordinate mean product `p(x)`
  between the landed head's existing movement `m1(x)`
  and the candidate movement `m2(x)` — with
  `COMPOSITION_LAMBDA = 1.0` fixed in exactly one place.
  Under `"worst-input-composition"` the training loss
  gains exactly one term instead — λ_w · `W`, where `W`
  is the mean of the k = 4 largest per-input `p(x)`
  values over the signal's train partition — with
  `WORST_INPUT_LAMBDA = 1.0` and `WORST_INPUT_K = 4`
  fixed in exactly one place (`WORST_INPUT_LAMBDA`,
  `WORST_INPUT_K` in `si/tiny_training.py`), part of the
  configuration's definition and never read from an
  intent or any other caller input, so no per-cycle
  λ_w or k dial exists. The two penalty terms are never
  blended; the mean-penalty hyperparams stay null under
  the worst-input configuration and the worst-input
  hyperparams stay null under both prior configurations.
  Composition is a recorded per-cycle configuration
  beside recorded signal selection and recorded target
  anchoring: a constructor-level `composition_config` on
  `TinyTransformerMind`, settable between cycles by the
  explicit driver call `set_composition_config(...)`;
  the gate's `Intent` carries the value in `to_dict()`
  so the journal shows all three selections. Under
  `"none"` the training path is byte-for-byte SI-25's;
  under `"movement-composition"` it is byte-for-byte
  SI-27's — both proven inside SI-29 by exact
  reproduction of the two prior decisive candidates
  (criterion 9 of the SI-29 as-built). No autonomous,
  learned, depth-driven, or verdict-driven composition
  selection exists, and the setter takes no λ_w or k
  argument.

**What SI-29 measured (cited to its as-built):**

- **Phase 0 first.** Before any new training semantics
  were finalised, SI-29 published the per-input
  composition record on both allowed partitions for the
  SI-25 control and the SI-27 treatment, chose the
  top-k mean form (k = 4) over the log-sum-exp form on
  that evidence — at `p`'s measured scale (~1e-3) a
  log-sum-exp term at any natural temperature τ would
  be numerically indistinguishable from the mean plus a
  constant, and a τ small enough to price the tail
  (τ ~ 1e-3) would be a fitted dial — and
  pre-registered its F3 reading rule. Phase 0's central
  readings: every allowed input already opposes under
  both candidates on both partitions (maximum `p`
  negative everywhere); the worst allowed input opposes
  only about half as strongly as the mean (SI-25 control
  train: worst `p` −0.0006878804284308964 against a mean
  of −0.0016478356464835148, 42%; SI-27 treatment train:
  worst `p` −0.001378012707059647 against a mean of
  −0.00262476117807562, 53%); the per-input distribution
  is clustered, not smooth (the weakest train cluster
  under the SI-27 treatment has eight members, so
  k = 4 sits inside it).
- **The decisive run.** One reused
  `TinyTransformerMind` through `si.run.run_cycles`:
  cycle 1 (signal 1, default anchoring, default
  composition configuration) ADMITted to `d0001`,
  reproducing SI-19/SI-21/SI-23/SI-25/SI-27 exactly —
  candidate blob `sha256:2761e8ff…6ca9e`, identity drift
  0.17189218290150166, capability regression
  0.1657181391492486, §6.6 VERIFY measured-pass. Cycle 2
  (signal 2, base-anchored, composition
  `"worst-input-composition"`, λ_w = 1.0, k = 4, trained
  at `d0001`) was REJECTED on cumulative identity drift;
  nothing landed; the bundle head after the run is
  `d0001`. Exactly one worst-input decisive run was
  performed; no sweep of λ_w, k, or τ of any kind was
  run.
- **The term works where it is allowed to look —
  decisively, at the worst case.** The worst-input term
  `W` (train partition, optimised) moved from
  +0.0004697766937776053 at initialisation to
  −0.0023225615446231264 at the end of training. The
  treatment's worst train input `p` is
  −0.0023129050472012187 (train index 27, sequence
  [5, 3, 20, 0, 8, 17, 27, 22]) — ~3.4× as opposing as
  the SI-25 control's worst train input
  (−0.0006878804284308964) and ~1.7× as opposing as the
  SI-27 control's (−0.001378012707059647); its top-4
  mean `W` on the record is −0.002322561516640352,
  likewise far below both controls'. The same holds on
  the held-out partition (reported only: worst held-out
  `p` −0.0022598436988714773). The treatment's tail is
  now nearly as opposing as its mean (max `p`
  −0.0023129050472012187 against mean
  −0.002434403120559355, 95% — against 53% for SI-27's
  treatment in Phase 0): pricing the tail directly
  closed the tail–bulk gap Phase 0 measured, on the
  population training may read.
- **The price shows in the bulk, and in the signal.**
  The treatment's mean and aggregate on allowed inputs
  are slightly *less* opposing than SI-27's treatment's
  (train mean −0.002434403120559355 against
  −0.00262476117807562; aggregate inner product
  −2.4928287954527795 against −2.687755446349435). The
  total loss ends **negative** (−0.001671632314028651),
  driven by the worst-input term, and the term *fights*
  the signal loss: the treatment's signal train residual
  (0.0006509292305944755) is ~69× the SI-25 control's
  (9.412649442227854e-06) and ~1.25× SI-27's treatment's
  (0.0005191352673009495).
- **It fails where the gate binds — worse than both
  composition predecessors.** The composed maximum is
  0.31215621903538704 > tol 0.25, at `tiny-id-0`, where
  the two movements' inner product is
  +0.04944786620153699 — positive, aligned, and worse
  than SI-27's treatment's +0.03801222744105431 and
  SI-25's control's +0.0338873784299374 at the same
  probe. At `tiny-id-0` the composed drift is again, to
  the printed digits, the sum of the two contributions
  (0.17189218290150166 + 0.1409909427165985 ≈ 0.3122) —
  the aligned extreme, for the fourth decisive run in a
  row at the probe where cycle 1 moved most. In
  aggregate on the probes the treatment still opposes
  (inner product −0.28751472160662706 over the 8
  identity probes; sign agreement 0.38671875, 99/256,
  published beside the magnitude-weighted figure) —
  more than SI-25's control's −0.21256230517288188,
  less than SI-27's treatment's −0.3271477846470207.
  The folded weight-space inner product is
  −0.0173962867164366, opposing (SI-25:
  −0.011457914936582613; SI-27: −0.02013451284871203).
- **The maximum's small improvement is partly
  shrinkage, and the as-built says so.** The treatment
  improves on both composition predecessors' composed
  maxima (0.3329036422073841 → 0.32509003952145576 →
  0.31215621903538704) and worsens on SI-23's
  (0.2963416576385498). Its movement is smaller than
  SI-27's penalised movement (candidate norm
  0.28252038537434354 against 0.2969693161862222;
  probe-space ‖m2‖ 0.9424814001349618 against
  0.9688074699782941) and larger than SI-25's control's
  (0.2515524481203623 / 0.7816722548913912). At the
  binding probe the improvement is therefore an
  improvement by a smaller step rather than by turning:
  the treatment's incremental movement at `tiny-id-0`
  (0.1409909427165985) is smaller than SI-27's
  (0.15392476320266724), so the composed sum there is
  smaller despite the movements aligning more (overall
  against SI-27: candidate norm −4.9%, probe ‖m2‖
  −2.7%, maximum −4.0% — derivations from the cited
  figures). The treatment's incremental movement exceeds
  cycle 1's remaining headroom (0.07810781709849834) at
  7 of 8 probes. Per probe, the treatment's composed
  drift improves on SI-27's treatment at 2 of 8 probes
  and worsens at 6; against SI-25's control it likewise
  improves at 2 and worsens at 6.
- **The destination cost is the least complete approach
  on record, at the pre-registered rule's boundary.**
  The treatment's signal held-out loss falls from
  0.002757805049077469 to 0.0006602814776996296 (~4.2×,
  12/16 examples closer) against the SI-25 control's
  ~264× fall and SI-27's treatment's ~5.3× fall (16/16
  closer); its held-out residual is ~63× the SI-25
  control's and ~1.26× SI-27's.
- **Selection oscillation is measured present.** The
  worst-input term's top-4 set on the train partition
  turned over completely between initialisation
  ([12, 28, 4, 20], single worst index 12, sequence
  [3, 19]) and the end of training ([27, 11, 19, 26],
  single worst index 27) — no input is in both sets —
  while the priced quantity moved monotonically in the
  published endpoints' terms (+0.0004697766937776053 →
  −0.0023225615446231264) and the run is deterministic.
  At the gradient-check points, by contrast, top-k
  membership is locally stable (0 of 1,024 single-entry
  ±eps perturbations change membership, at base and at
  `d0001`, on the check subset and on the full train
  partition).
- **The falsification readings (RFC SI-28 §6.6), as
  measured:** F1′ **measured** — the worst-case penalty
  does not transfer: the treatment's worst-input
  composition on allowed inputs improves decisively
  against both controls while its probe composition at
  the worst probe is worse than SI-27's treatment's.
  F2′ **not measured as defined** — its defining
  aggregate condition fails (the treatment's aggregate
  probe composition, −0.28751472160662706, is worse than
  SI-27's treatment's −0.3271477846470207), so the
  reading does not fire in its registered form; measured
  beside it, separately: the worst allowed input opposes
  decisively, and the composed maximum stays above tol
  0.25 (0.31215621903538704). F3′ **not measured under
  the pre-registered rule** — applying the Phase-0 rule
  as written, all three clauses hold (loss falls;
  12/16 closer, exactly the rule's stated 3/4 boundary;
  residual 0.0006602814776996296 ≤
  0.0010455625910972676, 2× the re-run SI-27 control's
  residual 0.0005227812955486338), so the destination
  counts as approached and F3′ does not fire — at the
  rule's boundary, with the least complete approach of
  any composition treatment on record measured beside
  the reading. §5.5 states what these two unmeasured
  readings leave open and what a successor slice must do
  about them.
- **The machinery is proven where it must vanish and
  where it must not look.** The coincidence
  configuration (signal 2, base-anchored, worst-input,
  at base) reproduces the established at-base run
  exactly (blob `sha256:131cc242…45f4`, identity drift
  0.16672966815531254, capability regression
  0.2418281203135848), with the new term identically
  zero on both partitions at initialisation and at the
  end of training. The worst-input loss's gradients pass
  central finite differences over every one of the 512
  factor entries at both base (worst
  5.5249277028738815e-08) and the decisive head `d0001`
  (worst 5.13570189768905e-08), tolerance 1e-5
  unchanged. Probe-blindness is asserted three ways,
  including a byte-identical candidate from a
  perturbed-probe-artifact bundle copy. Both controls
  re-run inside the slice reproduce their decisive
  candidates exactly (SI-25: blob `sha256:646d1e8b…d675`,
  drift 0.3329036422073841; SI-27: blob
  `sha256:1bb277ab…b637`, drift 0.32509003952145576).
  Composed capability passes (head-incremental
  regression 0.11724743689410388 ≤ eps 0.25; echo
  accuracy 1.0, 16/16). The full suite at the SI-29
  merge: **353 passed, 2 skipped**.

**The five-run cycle-2 record** (each figure under its
own run's labels; SI-29's as-built publishes the last
four side by side, and SI-21's figure is cited to its
own as-built via RFC SI-28 §1):

| Run | Cycle-2 configuration | Composed identity drift (cumulative) | Worst probe | Verdict |
|---|---|---|---|---|
| SI-21 | signal 2, weights-under-training, composition n/a (pre-SI-27) | 0.34378434903919697 | (SI-21 as-built) | REJECT |
| SI-23 | signal 2, weights-under-training, composition n/a | 0.2963416576385498 | tiny-id-3 | REJECT |
| SI-25 | signal 2, base-anchored, composition `"none"` | 0.3329036422073841 | tiny-id-0 | REJECT |
| SI-27 | signal 2, base-anchored, composition `"movement-composition"`, λ = 1.0 | 0.32509003952145576 | tiny-id-0 | REJECT |
| SI-29 | signal 2, base-anchored, composition `"worst-input-composition"`, λ_w = 1.0, k = 4 | 0.31215621903538704 | tiny-id-0 | REJECT |

Five authored second writes, five measured REJECTs, no
tolerance moved, no gate definition changed. The
deepest landed authored stack on record remains depth
1 (`d0001`); no authored two-delta stack exists; the
transformer-squash precondition remains unmet. The
Glimmer seed is untouched; Glimmer-scale deltas still
escalate with checks 3/4 "not evaluated".

---

## 2. Inherited invariants

Not negotiable inside this RFC (SI-9 §1.3, RFC SI-13 §2,
RFC SI-18 §2, RFC SI-20 §2, RFC SI-22 §2, RFC SI-24 §2,
RFC SI-26 §2, RFC SI-28 §2, restated for the candidates):

1. **Base is frozen forever.** All change is a stack of
   deltas; rollback is moving `head`.
2. **Never duplicate bytes** (Mark's design rule). A blob
   whose hash exists is never rewritten or copied; seed
   imports by reference where the pattern applies
   (`stored_bytes: 0` — the SI-8 Glimmer seed stores zero
   tensor bytes against 59,553,253,376 logical bytes,
   SI-9 §0).
3. **No ungated write.** Authorship produces *candidates*.
   The gate — and only the gate — admits, rejects, or
   escalates them, and SI-17's verdict mapping and
   blind-set guard apply to authored candidates exactly
   as to fixtures, at every cycle, for every signal,
   under every anchoring, under every training
   configuration.
4. **Honest labels are load-bearing.** "Not evaluated"
   stays distinguishable from a measured 0.0; a training
   loss stays distinguishable from a gate result; a
   number measured at one head stays labelled with that
   head (SI-20 §12.8); incremental and cumulative
   figures are never presented as each other
   (SI-20 §12.9); a number recorded under one anchoring
   is never presented as a number under the other
   (SI-24 §12.15); a number recorded under one training
   configuration is never presented as a number under
   another (SI-26 §17 item 17, extended by RFC SI-28
   §17 item 17 to the third composition value); a
   worst-case figure names the population it is a worst
   case *over* (RFC SI-28 §17 item 20); a
   judgement-call reading is applied under the rule
   pre-registered for it, or reported as an observation
   rather than a falsification test (RFC SI-28 §17
   item 21).
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
   addressing (SI-19's canonical-header lesson,
   SI-24 §12.7).
8. **No tolerance is a design variable.** No candidate
   in this RFC proposes moving `identity_tol`,
   `capability_eps`, any cap, or any policy default to
   make an outcome pass. The five cycle-2 REJECTs are
   measurements to design *around* — with new training
   semantics — never numbers to outvote, and never
   numbers to re-price by changing what the gate
   measures (SI-26 §8.2, carried at §8 of RFC SI-28 and
   at §8 of this RFC).
9. **The probes are the judge, never the training
   set** (in force in fact since SI-19, stated at
   SI-26 §2). No candidate in this RFC may train on,
   penalise against, project against, or otherwise
   shape a candidate using any probe input, probe
   logit, or probe-derived movement. A method that
   reads the judge's inputs to shape the judged
   artifact has not passed the judge; it has memorised
   the courtroom. SI-26 §9's rejection rests on this
   invariant, SI-27's and SI-29's designs were
   constrained by it at every step, and every candidate
   below is audited against it first.

---

## 3. Candidate (a): a small real safetensors base — still deferred

**Verdict up front: DEFERRED**, unchanged from SI-28 §3.
The SI-18 §3.7 audit's four preconditions (A-1 to A-4)
remain unmet, and SI-29's evidence does not touch any of
them: the miniature's composition problem is a property
of two rank-8 readout updates and a maximum-priced
identity arm, measured on a constructed model; nothing
about a worst-input penalty's non-transfer argues that a
real base would compose differently, and naming a real
base requires a download this scope forbids and an
operator decision this RFC explicitly does not take
(SI-28 §18 item 10, carried). SI-29 adds one sentence to
the audit's context, and it cuts *against* urgency, not
for it: the project now owns a training-side composition
instrument (three configurations, two measured negative
results at two statistics) whose behaviour at any other
scale is unclaimed — moving the question to a real base
before the miniature's remaining candidate-changing
question (§7) is answered would carry an unanswered
question to a more expensive room. Deferred; the
download question stays the operator's.

---

## 4. Candidate (b): worst-input composition-aware training — DONE as SI-29, closed

**Verdict up front: DONE — as SI-29, closed.** Built and
measured (PR #34, as-built
`docs/SI-29-worst-input-composition-training.md`):
signal 2, base-anchored, worst-input-penalised at
`d0001`, REJECTED on composed drift
0.31215621903538704, with the attribution published in
full (§1). This section exists so the candidate's
closure is on the record in the cadence's usual place,
and to state precisely *what* is closed, because §6
proposes relatives of this candidate and §7 proposes
its successor, and none of them may borrow this
candidate's evidence in either direction:

- **Closed:** the *worst-case* composition objective in
  the form SI-29 built it — one penalty term equal to
  λ_w times the top-k mean of `p(x)` over the train
  partition, at λ_w = 1.0, k = 4. Its premise is settled
  in both directions by SI-29's readings: it shapes
  worst-case allowed-input composition decisively and
  closes the tail–bulk gap there (F1′'s near end,
  measured), and it does not transfer to the binding
  probe, where alignment worsened past both composition
  predecessors (F1′'s far end, measured). No re-run of
  the worst-input objective at the same λ_w and k is a
  frontier; it is a control, and SI-31's design (§14)
  uses it as exactly that.
- **Closed with it:** composition objectives that price
  a *statistic of `p(x)` on allowed inputs* as their
  mechanism, as a family of *premises*. The mean's
  premise was falsified by SI-27's F2; the worst case's
  premise was falsified by SI-29's F1′. Two statistics
  of the same quantity, on the same population, through
  the same readout, have now failed to price the judged
  maximum — the second by design against the escape
  route the first left open. A third statistic of
  `p(x)` (a different quantile, a different reduction)
  would need an argument that it is not the same premise
  at a third statistic; no such argument exists in
  SI-29's record, and §6 shows the weight/form variants
  of the *same* statistic do not supply one either.
- **Not closed:** moves that change the *candidate*
  rather than the statistic it is priced by. F1 and F1′
  falsify premises about *penalty terms inside a loss*.
  A projection makes a different falsifiable claim —
  that removing the parallel component outright, as
  measured on allowed inputs, changes what the candidate
  *is* at the inputs where the movements currently add —
  and SI-29's published record neither confirms nor
  refutes it, because no projected candidate has ever
  been constructed or judged. §7 evaluates that claim;
  it does not inherit this candidate's closure, and it
  does not inherit its plausibility either — it must
  argue from §5's measurements, as §7 does.

---

## 5. The non-transfer and immobility questions — the deep treatment

SI-27 made the question *why a mean-priced penalty
loses at the worst probe*; SI-29 answered that question
for the worst case and made two sharper questions
unavoidable: **why does a training intervention that
closes the tail–bulk gap on allowed inputs, improves
the composed maximum on both composition predecessors,
and still opposes in aggregate on the probes, lose
*worse* at the single probe the gate prices — and why
has the composed maximum itself barely moved across
five differently-argued second writes?** This section
assembles what the five decisive runs jointly
establish, what they leave unknown, and what any
candidate must therefore measure before it may claim
to aim at the binding figure. Derivations are labelled
derivations.

### 5.1 The gate prices a maximum; every statistic of the allowed population so far optimised has failed to price it

The identity arm is seed-absolute, cumulative, and a
**maximum over probes** of per-probe drift (max
absolute logit coordinate per probe, maximised again
over the 8 identity probes). Nothing in SI-21, SI-23,
SI-25, SI-27, or SI-29 changed that definition, and §8
of this RFC carries SI-28 §8.1's rejection of changing
it. Against that fixed judge, the five cycle-2
candidates were produced by objectives that are, in the
relevant sense, statistics of the populations training
may read: target-logits MSE over 32 training sequences
(SI-21, SI-23, SI-25), in SI-27 an added term equal to
the *mean* of the per-input composition product over
the same 32 sequences, and in SI-29 an added term equal
to the *worst-case reduction* (top-4 mean) of the same
product over the same 32 sequences. SI-28 §5.1 argued a
mean objective purchases average-case behaviour, and
that the worst case was the one statistic of the right
quantity not yet measured. It has now been measured,
and the gap SI-28 priced for the mean is measured for
the worst case too — in the opposite direction from the
one the design needed: the treatment improved the
worst-case composition on every allowed population it
was measured on, improved the composed drift at 2 of 8
probes, and the maximum — the one figure the gate
reads — moved from SI-27's 0.32509003952145576 to
0.31215621903538704, a ~4.0% improvement against the
~25% fall the tolerance demands (derivation, as in
SI-28 §5.1: 0.25 / 0.3329036422073841 ≈ 0.751, so
admission required the maximum to fall by ~25% from
the SI-25 control's value; from SI-29's own value the
remaining fall is ~20%). F1′, in RFC SI-28 §6.6's
words, settles the premise at the worst case as F1
settled it at the mean: **composition measured on
allowed inputs — at either statistic — is not the
binding variable as a training-side penalty.** Any
candidate that optimises another statistic of `p(x)`
on allowed inputs must argue why its statistic binds
the maximum when the mean and the worst case did not;
§6's candidates cannot, and §7's candidate is designed
so that it does not price a statistic at all — it
changes the candidate.

### 5.2 At the binding probe, the movements do not merely fail to oppose — they add, by more under each composition penalty

SI-29's per-probe table (treatment, derivation column
labelled there as a derivation from two measured
forwards) shows the mechanism at `tiny-id-0` with no
room for a softer reading: cycle 1's drift there is
0.17189218290150166, the worst-input second movement's
increment there is 0.1409909427165985, and the composed
drift is 0.31215621903538704 — the sum, to the printed
digits. The inner product at that probe is *positive*
(+0.04944786620153699): in the coordinates where cycle
1 moved most, the second movement moves the same way.
And the sequence across training semantics is now the
finding, not any one entry in it: the SI-25 control's
inner product at the same probe was positive
(+0.0338873784299374), SI-27's mean-penalised treatment
raised it (+0.03801222744105431), SI-29's
worst-input-penalised treatment raised it again
(+0.04944786620153699), and SI-23's was positive at
*its* worst probe. Four different training semantics —
head-relative targets, base-anchored targets,
base-anchored targets with a mean composition penalty,
base-anchored targets with a worst-input composition
penalty — all produce second movements that align with
the first at whichever probe binds, and the two
penalties whose explicit purpose was opposition each
*increased* the alignment at the binding probe while
deepening opposition almost everywhere else. The
alignment at the binding probe is therefore a property
of the *signal family's geometry through this readout*
(both corrections share the away-from-the-last-token
component, SI-26 §10, and both updates are rank-8
readout matrices applied to every hidden state the
probes present), not of any one loss — and, after
SI-29, it is a property that *penalty pressure on
allowed inputs pushes the wrong way at the one judged
input where it matters*, for reasons the allowed
population cannot show: the worst allowed input and
the binding probe are different inputs from disjoint
populations, and improving the first has now twice
coincided with worsening the second. That is the
finding any successor candidate must confront: it is
not enough for a method to push movements apart
*somewhere*, nor even at the worst input it may read;
and training is forbidden by invariant 9 from knowing
which judged inputs the movements add at. The only
legal moves left are ones that change the candidate's
geometry itself (§7) — or a later RFC's genuinely new
argument.

### 5.3 The magnitude half of the problem — and why SI-29's improvement must not be misread

Composition is not the whole failure, and this RFC
refuses — as SI-28 §5.3 refused — to let the direction
story swallow the size story. Cycle 1 leaves headroom
of 0.07810781709849834 under tol at the binding probe.
SI-29's second movement's increment at `tiny-id-0`
alone is 0.1409909427165985 — nearly twice the headroom
— and its incremental movement exceeds the headroom at
7 of 8 probes. SI-29 adds a sharper form of the trap
SI-28 §5.3 named: the treatment's composed maximum
*improved* on both composition predecessors while its
binding-probe alignment *worsened* — the improvement
is accounted for, in the as-built's own figures, by a
smaller movement (candidate norm 0.28252038537434354
against SI-27's 0.2969693161862222; incremental
movement at the binding probe 0.1409909427165985
against 0.15392476320266724), not by turning. A reader
who quotes "0.33290 → 0.32509 → 0.31216" as a trend
toward admission is blending an improvement by
shrinking with an improvement by composing, in exactly
the way RFC SI-28 §17's labelling rules and SI-29's
§5.3-style accounting exist to prevent. Two
consequences for design, both binding on §7:
(i) magnitude remains a **published, compared
quantity** — candidate norm and ‖m2‖ beside every
drift figure — and a treatment whose maximum improves
only because its movement *shrank* must remain
distinguishable from one whose movement *turned*; for
a projection candidate this obligation sharpens,
because a projection removes a component and therefore
shrinks the candidate *by construction* — §7.2 and
§15 require the projected candidate's norm and its
pre-projection norm to be published together, with the
distinguishability reading stated in the as-built;
(ii) §10's rejection stands guard on the other side —
deliberately shrinking the movement by norm budget,
step count, or correction scale is gate-fitting, not a
composition result, and is rejected there so that (i)
cannot be misread as licensing it, and so that §7's
projection — which removes a *named component*, in
full, by a definition fixed in advance — cannot be
misread as a shrinkage dial either.

### 5.4 The binding probe is stable across the last three runs — and designing around that is still forbidden

SI-23's binding probe was `tiny-id-3`; SI-25's,
SI-27's, and SI-29's is `tiny-id-0` — in each case the
probe where cycle 1 itself moved most (cycle-1 drift
at `tiny-id-0` is 0.17189218290150166, the largest of
the eight). SI-28 §5.4 recorded two consecutive
confirmations and warned that designing a slice around
the assumption that this is a regularity would be
designing around the judge. Three consecutive
confirmations do not retire that warning; they sharpen
the temptation it guards against, and SI-29 sharpens
it further, because the binding probe is now also the
probe where two successive penalties *increased*
alignment (§5.2) — a pattern a designer could itch to
aim at. This RFC therefore records the regularity
again as a *measured observation about past runs* and
forbids its use as a design input: no candidate below
names `tiny-id-0`, weights it, or shapes anything by
its identity. The legal formulation — and the only one
§7 uses — is population-level: *the component of the
second movement parallel to the first, as measured
over whatever population the method may legally read.*
On allowed inputs that population is the train
partition; on judged inputs it is the probe suite,
which training never reads. If the parallel component
measured on allowed inputs is the same phenomenon as
the addition at the binding probe, removing it has a
chance to bind the maximum; if it is not, F1 repeats
for a candidate-changing move instead of a penalty,
and §7's falsification readings are written to say
exactly that (§7.5).

### 5.5 What is still not known — the measurements and definitions that must precede any fix

Stated as open questions a slice could retire — this
RFC invents no numbers for them (§18 carries them
forward):

- **F2′'s conjunction, and what its failure leaves
  standing.** RFC SI-28 §6.6 defined F2′ with an
  aggregate clause (the treatment's aggregate probe
  composition no worse than SI-27's treatment's) and a
  maximum clause (the composed maximum stays above
  tol). SI-29 measured the clauses separately and they
  split: the aggregate clause *fails*
  (−0.28751472160662706 against −0.3271477846470207)
  while the maximum clause's fact *holds*
  (0.31215621903538704 > 0.25). The registered reading
  therefore does not fire, and the project is left
  without a falsification verdict for the proposition
  F2′ was written to test — that worst-case opposition
  on allowed inputs leaves the judged maximum unmoved
  *even when transfer otherwise holds*. The honest
  residue is two separately-labelled facts, not a
  reading (§17 item 23 of this RFC makes quoting them
  as one a labelling defect). A successor slice must
  repair the definition *in advance*: state its F2-class
  reading so that the aggregate clause and the maximum
  clause are judged as the separate propositions they
  turned out to be, with each clause's boundary stated
  before the decisive run it will judge. §7.5 and §15
  therefore require SI-31 to pre-register its successor
  readings — F1′′, F2′′, F3′′ — in that repaired form
  as part of Phase 0, and §18 records the general
  lesson as an open methodological question for the
  cadence, not only for SI-31.
- **F3′'s boundary was met exactly, which is not the
  same as comfortably.** SI-29's pre-registered rule
  returned "approached" with its closer-fraction clause
  met *exactly* (12/16, the stated 3/4 boundary) and
  its completeness clause met with the least complete
  approach on record beside it. A boundary met exactly
  prices nothing about how much destination a *stronger*
  intervention — and a projection, which removes part
  of the trained candidate outright, is stronger in
  kind — would leave. The successor slice must carry
  the same rule shape forward unchanged in its
  thresholds (moving a threshold after it has been met
  exactly would be re-pricing a judgement call after
  the numbers are known, against RFC SI-28 §17
  item 21), apply it to its own treatment against its
  own re-run control, and publish the partial trade
  beside the reading whichever way it falls (§7.5).
- **Whether the parallel component on allowed inputs
  tracks the addition at judged inputs.** The
  composition-generalisation gap of SI-26 §6.5, one
  level sharper again: F1 measured the gap for the
  mean, F1′ for the worst case — both for *penalties*.
  For a *projection* there is no measurement at all:
  no projected candidate exists, so neither the size of
  the train-partition parallel component in any
  existing candidate, nor what removing it would leave
  (norm, signal arrival, per-probe behaviour), has ever
  been published. Whether the component a projection
  would remove is large enough to matter — or so large
  that removing it leaves no destination — is **an
  open question to measure**, on existing candidates,
  from existing machinery, before any projected
  candidate is judged. §7.4 makes that measurement
  Phase 0 of the recommended slice.
- **The τ form was never implemented, and Phase 0's
  scale argument is a derivation, not a measurement.**
  SI-29 chose the top-k form on a derivation from `p`'s
  measured scale (~1e-3): a log-sum-exp term at natural
  τ would approximate the mean; a tail-pricing τ would
  be fitted. The derivation is sound as far as it goes
  and it has never been checked against an implemented
  smooth-maximum term. §6 records this as part of why
  the τ question stays open rather than closed — and
  why it still does not become the next slice.
- **Selection oscillation's dynamics across the run.**
  SI-29 measured the top-k set's complete turnover
  between the run's endpoints and its local stability
  at the gradient-check points; the trajectory between
  those facts (how often membership changed across the
  2,000 steps, and whether the priced quantity's path
  was monotone throughout rather than only at its
  endpoints) was not published. It is recorded here so
  it is not mistaken for a measured property of the
  worst-input objective in either direction; no
  candidate below depends on it.

---

## 6. Candidate (c): another point on the worst-input axis — a different λ_w, a different k, or the τ form

**Verdict up front: DEFERRED as open questions — none
of them is the next slice, and no sweep is licensed by
the deferral.** SI-29 ran exactly one point on this
axis (λ_w = 1.0, k = 4, top-k form; τ nowhere) and its
as-built states, in its own "still NOT claimed" list,
that it draws no λ_w / k conclusion beyond that point
and that whether a different single λ_w or k would
behave differently — and any τ at all — is an open
question for a future RFC. This RFC takes up each
variant in turn, keeps all three open deliberately, and
shows why none of them answers §5.

### 6.1 A different λ_w

- **The worst-case premise is settled at the statistic,
  not the weight.** F1′ falsified "worst-case
  composition measured on allowed inputs shapes
  worst-case composition on judged inputs" — a claim
  about what the term *reduces*, not about how hard it
  pushes. SI-29's term values show λ_w = 1.0 was not
  negligibly scaled: the term moved the worst allowed
  input ~3.4× past the unpenalised control, closed the
  tail–bulk gap (95% of the mean), drove the total loss
  negative, and raised the signal train residual ~69×
  against the control. The "λ_w was simply too small to
  matter" reading is contradicted by the published
  record at every one of those figures, and the "λ_w
  was too large" reading is F3′'s boundary trade — an
  argument about destination cost, not about the
  maximum, which *worsened at the binding probe* at
  this λ_w already: a larger weight pushes further in
  a direction whose judged end moved the wrong way,
  and a smaller weight interpolates toward the SI-27
  and SI-25 controls, both of which also REJECTED with
  positive binding-probe alignment. No λ_w argument
  from SI-29's record reaches the binding figure.
- **Sequencing.** A future λ_w is a future RFC's
  argued, single, pre-registered value (SI-28 §16's
  discipline, carried at §16 of this RFC) — argued from
  evidence that does not yet exist. If SI-31's
  projection readings (§7) suggest the *weight* axis is
  where the remaining signal lives, that RFC can argue
  it from three composition configurations' published
  term values instead of two. Deferred to §18.

### 6.2 A different k

- **k = 4 was argued from Phase 0's distribution, and
  SI-29's record does not un-argue it.** Phase 0 showed
  the weakest train cluster has eight members and the
  top-4 values under the SI-27 treatment are extremely
  close (−0.001378012707059647, −0.0013805419923998206,
  −0.0013907916268110303, −0.0014092673012295295 — the
  SI-27 treatment's four largest train `p` values, from
  SI-29's Phase-0 record): k inside the cluster prices
  the tail as a group, not a single input. A smaller k
  (k = 1, the pure maximum) prices one input whose
  identity SI-29 measured turning over completely
  across a single run (§1) — a noisier target with no
  argued advantage; a larger k prices more of the bulk
  and interpolates toward the mean objective SI-27
  already falsified. Neither direction carries an
  argument from SI-29's measurements that it reaches
  the binding probe, and the binding-probe failure
  (F1′) is a transfer failure, not a tail-width
  failure: the tail *as priced* moved decisively and
  the judged end moved the other way.
- **What stays open.** Whether the tail–bulk structure
  Phase 0 measured is stable enough that *any* k is a
  well-defined target across a full training run is
  partly answered (endpoint turnover measured;
  trajectory unpublished, §5.5) and partly not. A
  future k, like a future λ_w, is a single argued value
  in a future RFC — never a sweep, and never a second
  decisive run inside a slice scoped for one (§16).

### 6.3 The τ form (log-sum-exp / smooth maximum)

- **What it is.** RFC SI-28 §6.2's second form:
  `W` = τ · log Σ_x exp(p(x)/τ), τ fixed in the
  configuration's definition — smooth everywhere, its
  gradient a softmax-weighted combination of the same
  per-input gradients. SI-29 did not implement it; τ
  appears nowhere in SI-29's code or record.
- **Why it is not the next slice.** Two independent
  reasons, either sufficient. First, SI-29's Phase-0
  derivation (§5.5): at `p`'s measured scale (~1e-3),
  any τ of natural scale makes the term numerically
  indistinguishable from the *mean* plus a constant —
  the statistic F2 falsified — and a τ small enough to
  price the tail (τ ~ 1e-3) is a scale fitted to `p`'s
  observed magnitude, exactly the fitted dial SI-28
  §6.2 and §16 forbid; the form's honest τ choices
  therefore collapse to a falsified statistic or a
  fitted parameter. Second, even taken on its own
  terms, the smooth maximum is *the same statistic at a
  softer reduction*: it prices worst-case composition
  on allowed inputs, the premise F1′ falsified. A
  smoother route to a falsified premise is not a new
  premise.
- **What stays open.** The Phase-0 scale derivation has
  never been checked against an implemented term
  (§5.5). If a future RFC wishes to close the τ
  question by measurement rather than derivation, the
  honest shape is a measurement of the term's behaviour
  at a single pre-registered τ argued in that RFC —
  not a τ sweep, and not a decisive cycle-2 run whose
  premise F1′ has already priced. Deferred to §18,
  behind §7, with no implementation licensed here.

### 6.4 The axis as a whole

The three variants share one property that decides
their verdict jointly: each keeps the mechanism —
*penalise a statistic of `p(x)` on allowed inputs
inside the loss* — and varies its parameters. SI-27
and SI-29 have now measured that mechanism at its two
most defensible points (the mean at parity weight; the
worst case at parity weight with a cluster-argued k)
and the mechanism's premise failed at the judged end
both times, the second time in the direction opposite
to its design. Parameter variation after a premise
falsification is tuning, and tuning against a fixed
judge with the judge's tolerance in view is the
gate-fitting shape §§9–10 reject in their own domains.
The axis stays open as a set of *questions* — §18
records each with what would retire it — and closed
as a *recommendation*.

---

## 7. Candidate (d): the projection form — train-partition orthogonalization

**Verdict up front: RECOMMENDED — as SI-31, phased
measurement-first (§14).** The candidate SI-26 §9 left
open, SI-28 §7 took up and deferred on sequencing
behind the worst-input slice, and RFC SI-28 §6.6 named
in advance as the composition line's remaining honest
move under the outcome SI-29 then measured. It is the
only candidate that changes the *candidate* rather
than the *statistic*, it satisfies invariant 9 exactly
as the penalty forms did, its premise is stated in
falsifiable form (§7.5), its decisive configuration
changes exactly one variable against SI-29's treatment
(which becomes its re-run control), and no ADMIT is
required.

### 7.1 Why the prior rejections, closures, and deferral do not bind this candidate

- **It is not §4's closed family.** The mean and
  worst-case penalties price statistics of `p(x)`
  inside a loss and let optimisation trade them against
  the signal at every step; their premises are
  falsified (§4). A projection prices nothing: it
  constructs the judged candidate by removing, in full,
  the one component the record indicts — the part of
  the second movement parallel to the first, as measured
  on allowed inputs — and the resulting candidate is
  then measured on *every* published record (signal
  arrival, composition, gate) as the candidate it is.
  F1/F1′ are readings about penalty transfer; no
  projected candidate has ever existed for them to be
  readings *about*.
- **It is not SI-26 §9's rejected orthogonalization.**
  That rejection named methods that read *probe* inputs
  or probe movements (rejected here again at §8.2), and
  a weight-space projection refuted on SI-25's table.
  This candidate reads no probe anything: its
  coefficient is computed from signal 2's train
  partition, the same 32 sequences SI-27's and SI-29's
  penalties read, with the same probe-blindness
  assertions required (§15 criterion 4). The
  *measurement* defining the removed component is the
  movement inner product on allowed inputs — not a
  Frobenius inner product in weight space, which is the
  variant SI-25's table refuted and which §7.2 excludes
  by definition.
- **It is not §8's rejected per-probe anchoring.** The
  gate is untouched (§8.1); no probe is anchored,
  weighted, or named (§8.2). "Parallel" in this
  candidate ranges over *allowed inputs during
  candidate construction*, never over probes; the
  probe-side composition appears only where it always
  has — in post-verdict diagnostics.
- **SI-28 §7's sequencing deferral is discharged, not
  overridden.** SI-28 deferred this candidate for one
  stated reason — it was "sequenced behind (c) because
  a projected candidate's every property is unmeasured
  during training, and SI-27's F3 record prices exactly
  that gap" — and RFC SI-28 §6.6 wrote, in advance,
  that if the worst-case reading failed, "the
  composition line's remaining honest moves are §7's
  projection form (which changes the candidate, not the
  statistic) or a later RFC's new argument". SI-29 has
  now run the slice it was sequenced behind, and
  measured the failure §6.6 described (F1′ measured;
  the maximum immobile beside F2′ unmeasured as
  defined). The deferral's condition is met on
  SI-29's evidence, and the objection it rested on —
  unmeasured-during-training properties — is answered
  by design in §7.2/§7.4, not waved away: Phase 0
  rehearses the projection on existing candidates, and
  the built slice re-measures *everything* about the
  projected candidate before the gate sees it.

### 7.2 What the projection means, precisely

Under the new composition configuration (a fourth named
value, §7.3), candidate construction for signal 2,
base-anchored, at head `d0001` proceeds in two steps,
both deterministic:

1. **Train as SI-25 did.** Signal 2's factors are
   trained against SI-25's base-anchored signal loss
   under composition `"none"` semantics — no penalty
   term of either prior kind enters the loss. (Training
   under a penalty and *then* projecting would change
   two variables against every control and blend two
   mechanisms' records; §16 forbids it.)
2. **Project, once, in full.** Let `m1(x)`, `m2(x)` be
   SI-27's movement definitions unchanged, evaluated on
   signal 2's 32 training sequences. The projection
   coefficient is the single scalar

     α = ( Σ_x ⟨m1(x), m2(x)⟩ ) / ( Σ_x ⟨m1(x), m1(x)⟩ ),

   sums over the train partition — the least-squares
   coefficient of `m2` on `m1` *as measured on allowed
   inputs*. The judged candidate's folded update is
   Δ2′ = Δ2 − α·Δ1, where Δ1, Δ2 are the two cycles'
   folded `lm_head.weight` updates: because both
   movements pass through the same readout fold, the
   candidate whose train-partition movements are
   `m2(x) − α·m1(x)` — aggregate inner product with
   `m1` exactly zero on that partition, by construction
   and then *by measurement* (§15 criterion 3) — is the
   candidate carrying Δ2′. The projection fraction is
   **full removal (α as defined, not scaled)**: α is a
   measured coefficient, not a dial; there is no
   projection-strength parameter, nothing is swept, and
   any partial-removal variant is a different candidate
   for a future RFC (§16). The slice must show, in its
   as-built, the derivation that the movement-space
   coefficient and the folded-update construction agree
   — and must verify it by direct measurement of the
   projected candidate's allowed-input record, not rest
   on the derivation (§15 criterion 3).

   Two definitional exclusions, stated so they cannot
   drift: the coefficient is **not** computed per input
   (a per-input removal would need a different update
   per input, which a single LoRA candidate cannot
   carry — the aggregate coefficient is the only form a
   single candidate admits), and it is **not** computed
   in weight space (the Frobenius variant is the one
   SI-25's table refuted; the movement coefficient is
   the quantity §§5.2–5.4 reason about).

   **Post-projection accounting is part of the
   construction, not an afterthought** — this is the
   obligation SI-28 §7 laid down and §7.1 carries: the
   projected candidate is re-measured on the full
   published record before it is judged — signal train
   and held-out losses, the non-improvement refusal
   applied to the *projected* candidate's signal
   held-out loss (a projected candidate that no longer
   improves on its signal fails closed at construction,
   exactly as a trained candidate would), candidate
   norm beside its pre-projection norm, and the
   allowed-input composition record. The gate then
   judges the projected candidate and no other; the
   pre-projection candidate is a construction
   intermediate, labelled as one wherever its figures
   appear (§17 item 22), and is never itself judged,
   landed, or reported as a run.

### 7.3 The representation: a fourth value on the composition axis

Composition remains **one recorded configuration axis**
beside signal and anchoring. `COMPOSITION_CONFIGS`
gains exactly one value — `"train-partition-projection"` —
validated by the existing discipline; any other value
refuses. The mind chooses neither its signal, its
anchoring, nor its composition configuration (SI-26
§17 item 13, carried); the driver configures all three
and the journal shows all three. The value's definition
carries no weight, no k, and no fraction: the training
step is SI-25's, and the projection coefficient α is
*measured from the run's own candidates on the train
partition* and recorded in the hyperparams beside the
signal loss's values — recorded, never settable. Under
`"none"` the training path remains byte-for-byte
SI-25's; under `"movement-composition"` byte-for-byte
SI-27's; under `"worst-input-composition"`
byte-for-byte SI-29's — proven, not asserted, by exact
reproduction of all three controls inside the slice
(§15 criterion 9). No autonomous, learned,
depth-driven, verdict-driven, or probe-driven selection
exists: in particular, nothing anywhere in the slice
reads a probe result to choose the configuration or to
compute α.

### 7.4 Measurement first: Phase 0 retires §5.5's unknowns before the projection is trusted

The slice is phased, and the phase order is part of
the recommendation, not a scheduling nicety:

- **Phase 0 (measurement and definitions, no new
  candidate construction).** Using existing machinery
  on existing candidates, publish:
  (i) the **projection rehearsal** — for the SI-25
  control, the SI-27 treatment, and the SI-29 treatment
  (each under its own labels): the train-partition
  projection coefficient α each would carry, the
  parallel component's share of each candidate's
  movement (aggregate inner products and norms, train
  and held-out partitions, the held-out figures
  reported only), and the *rehearsal* projected
  candidates' signal held-out records and norms —
  computed as **diagnostics on already-judged
  candidates**, labelled derivations/diagnostics, never
  gate verdicts, never training (§17 items 12, 22).
  This prices, before any new candidate is built,
  whether the removable component is large enough to
  matter and whether removing it leaves a destination
  at all; (ii) the slice's **pre-registered successor
  readings** F1′′/F2′′/F3′′ (§7.5), with F2′′'s clauses
  separated as §5.5 requires and F3′′'s thresholds
  carried from SI-29's rule unchanged in shape; and
  (iii) the **projection-correctness tolerance** — the
  numerical bound within which the built slice's
  measured post-projection aggregate inner product on
  the train partition must sit to count as zero — stated
  from Phase 0's rehearsal figures, before the decisive
  run. If (i) shows the parallel component is negligible
  on allowed inputs, or that removing it collapses
  every rehearsal candidate's destination, §7.2's
  construction must be re-argued in the as-built from
  that measurement before Phase 1's decisive run is
  trusted; the RFC's recommendation survives that
  outcome only as a measurement of the projection
  itself, and the as-built must say which of the two
  slices it turned out to be (the SI-28 §6.7 / SI-29
  Phase-0 discipline, applied a third time).
- **Phase 1 (the build).** The fourth composition
  configuration, its projection-correctness check, the
  coincidence configuration, all three re-run controls,
  the decisive run, the full diagnostic set, and the
  falsification readings — as specified in §14 and §15.

### 7.5 What would falsify the design's premise

A recommendation that cannot be wrong is not a design;
it is a hope. This slice's premise is falsifiable in
three named ways, in the repaired form §5.5 requires —
each reading's clauses stated separately, each boundary
stated in Phase 0 before the decisive run, and §15's
criteria require the measurements that would falsify
it to be published whatever they show:

- **F1′′ — the projection does not transfer.** The
  projected candidate's train-partition composition is
  zero as constructed and *as measured* (aggregate
  inner product within the Phase-0 tolerance of 0.0),
  while its probe composition at the worst probe is
  unchanged or worse than SI-29's treatment's
  (+0.04944786620153699 at `tiny-id-0`) — the
  candidate-changing move fails transfer as both
  penalty moves did. Falsified premise: that the
  parallel component measured on allowed inputs is the
  component that adds at judged inputs, through this
  readout, at this scale. Consequence stated in
  advance: with the mean, the worst case, and the
  projection all measured failing at the binding probe,
  no composition move that reads only allowed inputs
  remains on this RFC's record, and the line's honest
  continuation is a later RFC's genuinely new argument
  — not a fourth variation (§18 item 1).
- **F2′′ — the maximum is immobile even when the
  binding probe turns.** Stated as two separately
  judged clauses, repairing F2′'s definition (§5.5):
  (a) the *transfer clause* — the projected candidate's
  probe composition at its own worst probe improves on
  SI-29's treatment's figure at that probe, and its
  aggregate probe composition is published beside
  SI-27's and SI-29's treatments' figures under their
  own labels, with no no-worse conjunction required;
  (b) the *maximum clause* — the composed maximum
  nonetheless stays above tol 0.25 (another probe
  binds, or the remaining movement's magnitude alone
  exceeds the headroom). F2′′ **is measured** iff
  clause (a) holds and clause (b) holds. Falsified
  premise, if it fires: that the composed maximum is a
  composition property of the binding probe at all —
  from which it would follow that the maximum's
  immobility across six runs is a magnitude-and-geometry
  fact (§5.3), and composition work on this signal
  family, in any form, has reached its measured end.
- **F3′′ — the destination is traded away.** The
  projected candidate composes better at the worst case
  but its signal held-out record shows the destination
  no longer approached — under the **pre-registered
  reading rule carried from SI-29's**: before the
  decisive run, the slice restates SI-29's three
  clauses unchanged in their thresholds (the loss
  falls; at least 12 of 16 held-out examples closer;
  residual at most 2× the re-run control's residual —
  for SI-31 the completeness reference is the re-run
  SI-29 treatment, whose published residual is
  0.0006602814776996296, so the boundary the Phase-0
  registration must state is
  0.0013205629553992592), and the as-built applies that
  rule as written, in either direction. The projection
  is the first intervention that removes part of the
  trained candidate outright, so this reading is the
  one SI-28 §7's deferral priced: a candidate whose
  every property was re-measured after construction
  (§7.2) can still arrive less far, and the record must
  say so in those words. Such a candidate, even if the
  gate ADMITted it, would not be evidence that
  projected *authorship* works; it would be evidence
  that the gate can be approached by a candidate that
  does less of its task.

### 7.6 The decisive configuration: signal 2, base-anchored, projection-constructed, at `d0001`

The decisive run is SI-29's decisive run with **exactly
one configuration changed**: cycle 2's composition
configuration, from `"worst-input-composition"` to
`"train-partition-projection"`. Signal (signal 2),
anchoring (base-anchored), head (`d0001`), target
source (base), step count, learning rate, norm budget,
and candidate construction (apart from the projection
step that *is* the configuration) are SI-29's,
unchanged. Cycle 1 is SI-29's cycle 1 (default signal,
anchoring, and composition configuration) and must
reproduce `d0001` exactly. The treatment's primary
control is SI-29's treatment itself, **re-run inside
the slice** under `"worst-input-composition"` and
required to reproduce SI-29's decisive candidate
exactly (blob `sha256:11cab02e…b8db5`, drift
0.31215621903538704); the SI-25 and SI-27 controls are
likewise re-run and required to reproduce exactly
(§15 criterion 9), so the projection's difference is
attributable to the construction step and not to the
refactor that introduced it — the attribution
discipline SI-27 applied to SI-25's control and SI-29
applied to both of its controls. Exactly one
projection decisive run is performed; no sweep of any
projection variant, fraction, or coefficient definition
is run (§16).

---

## 8. Candidate (e): per-probe anchoring — three readings, three verdicts, none reopened

This candidate names three different proposals that
share a phrase. SI-28 §8 separated them and SI-29's
evidence treats none of them differently; they are
separated here so none can borrow another's verdict —
and so the record states explicitly, as this RFC's
brief requires: **SI-29 supplies no new evidence that
reopens any reading. All three verdicts stand exactly
as SI-28 left them.**

### 8.1 As a gate or accounting change: REJECTED (carried, not reopened)

Any reading under which the gate accounts identity per
probe — budgets per probe, headroom carried per probe
from cycle 1, a per-probe anchor the candidate must
match — is SI-26 §8.2's rejected candidate in new
vocabulary, carried by SI-28 §8.1, and its rejection is
carried forward in full: it reprices the judge after
(now) five measured failures while leaving 0.25 printed
in the policy; the maximum is not an accounting
accident; and SI-29 gives the change no evidence to
stand on — the gate's journaled checks again equal an
independent evaluation (SI-29 criterion 7), and the
diagnostics *confirm* the verdict's figure at the
binding probe. The failure is in the candidates, five
times measured. Not reopened.

### 8.2 As training against per-probe targets: REJECTED (invariant 9)

The reading under which cycle 2's targets, penalty, or
projection are anchored *per probe* — each identity
probe's desired movement computed and trained or
projected toward — reads probe inputs and probe-derived
movements to shape the candidate. That is training on
the judge (invariant 9), SI-26 §9's rejection, carried
by SI-28 §8.2, unchanged by SI-29: an ADMIT so produced
would demonstrate that per-probe targets can satisfy 8
known inputs, not that two authored deltas compose, and
would be uninterpretable — worse than a REJECT, because
a REJECT at least measures something. SI-29's
binding-probe record (§5.2) makes this reading *more*
tempting — the binding probe is stable, named, and its
alignment is the worst on record — and temptation is
not evidence: nothing in SI-29 measures a per-probe
method or argues one could be interpreted if it
passed. Rejected outright, and §16 lists it among the
non-goals with no exceptions and no "diagnostic-only"
carve-out for the training or construction path.

### 8.3 As per-input accounting on allowed inputs: ADOPTED, inside §7 — not a slice of its own

The one invariant-clean residue: accounting composition
*per input* on the populations training and
construction may see — which SI-28 §5.5 named as the
project's outstanding measurement, SI-29's Phase 0
published, and §7.4 of this RFC extends to the
projection rehearsal. Adopted in exactly that form,
inside SI-31, riding with the slice whose design it
informs (the discipline of SI-26 §8.1 and SI-28 §8.3,
applied again). A standalone per-input-accounting
slice — more tables without a construction that
depends on them — is **not recommended**: SI-29's
Phase 0 closed the per-input *distribution* question
for the penalty family, and the only per-input
accounting still outstanding is the rehearsal §7.4
scopes, whose figures are inputs to a decision SI-31
itself must take.

---

## 9. Candidate (f): probe-partitioned sequencing

**Verdict up front: REJECTED (carried, not reopened).**
The proposal, stated fairly: if the composed maximum
is the obstacle, sequence the writes so that successive
cycles move *disjoint parts* of the judged behaviour —
partition the probes (or the behaviour they sample) and
assign each cycle a partition, so no single probe
accumulates two aligned movements. SI-28 §9 rejected
it on both of its possible readings, and SI-29
supplies no new evidence for either:

- **Partitioned by probe behaviour.** Deciding which
  cycle may move which probe requires reading probe
  movements per candidate signal — shaping the sequence
  by the judge's inputs and their measured responses
  (invariant 9), at the level of the *driver's* choices,
  which §17 item 13's rule (recorded selection is not a
  choice the mind made) does not launder. Rejected on
  the invariant.
- **Partitioned by allowed-input behaviour.**
  Partition the *signals* instead — author or select
  second signals whose allowed-input movement profiles
  are disjoint from the first's, in sequence. This is
  invariant-clean and is SI-26 §10's rejected candidate
  in sequence form, as SI-28 §9 held: a search over
  signal variants for ones whose movements happen to
  fit, argued backward from the obstacle the gate
  measured — and SI-29 strengthens the rejection's
  evidence, because the family's composition behaviour
  is now measured failing transfer at two statistics
  and *worsening at the binding probe under both
  penalties* (§5.2); partitioning its members by where
  they move does not change what each member's movement
  *is* at the judged inputs. The gate-fitting trap of
  SI-24 §5.1 / SI-26 §13, in the same costume, worn by a
  sequence instead of a signal. Rejected on evidence
  and on the trap, either way.

A weaker relative survives and is noted so it is not
mistaken for this candidate, as SI-28 §9 noted it:
*nothing* in this section speaks against running more
than two cycles someday, or against signals authored
for reasons independent of the gate (SI-23's signal 2
was one). What is rejected is partitioning — of probes
or of signals — *as the mechanism* for making the
maximum pass. Not reopened.

---

## 10. Candidate (g): magnitude discipline — a smaller second movement

**Verdict up front: REJECTED as a slice (carried, not
reopened); retained as a publication obligation inside
§7.** §5.3 establishes that size is half the failure:
SI-29's second movement exceeds cycle 1's remaining
headroom at 7 of 8 probes, and SI-29's maximum
improved partly *by shrinking* while its binding-probe
alignment worsened. The tempting candidate — train
cycle 2 under a smaller norm budget, fewer steps, or a
reduced correction scale, so `m2` shrinks toward the
headroom — is disposed of by the project's standing
rules, and SI-29's evidence does not reopen them:
SI-26 §16 already names "no reduced correction scale,
no reduced step count or norm budget chosen to shrink
`m2`" a binding non-goal, and SI-24 rejected the
reduced-scale route before that, as SI-28 §10 restated.
The reason is not tidiness: a shrunken candidate that
ADMITs has shown that a *smaller* second write fits,
which the project has known since cycle 1's drift left
headroom — it prices the destination down until the
judge stops objecting, and its held-out record would
show the trade (F3′′'s shape) by construction. SI-29
does add one caution *against* a softer misreading in
the other direction: because its improvement was
partly shrinkage (§5.3), no future candidate — §7's
included — may report a maximum improvement without
the norm accounting that separates shrinking from
turning. What SI-29 licenses is exactly what SI-28
§10 licensed, sharpened by §5.3 (i): magnitude is a
**published, compared quantity** in every composition
run — candidate norm and ‖m2‖ beside every drift
figure, and for a projected candidate its
pre-projection norm beside its judged norm. Discipline
as accounting: adopted. Discipline as a training dial
aimed at the headroom: rejected. Not reopened.

---

## 11. Candidates (h) and (i): a different λ for the mean penalty; new target / additive

### 11.1 Candidate (h): a different λ for the mean penalty

**Verdict up front: DEFERRED as an open question
(carried) — not the next slice.** SI-28 §11.1 deferred
this candidate on the ground that F2 settled the mean
at the statistic, not the weight, and that SI-27's term
values contradict the under-scaled reading. SI-29
changes nothing in that argument and adds a parallel
one (§6.1 makes it for λ_w): two objectives have now
been run at parity weight, both dominated their total
losses, both raised their signal residuals by orders of
magnitude against the unpenalised control, and both
failed at the binding probe in the same direction. A
future λ — for the mean objective or for the
worst-input objective — is a future RFC's argued,
single, pre-registered value, argued from evidence
that does not yet exist; if SI-31's projection readings
suggest the *weight* axis is where the remaining signal
lives, that RFC can argue it from three composition
configurations' published records instead of two.
Deferred to §18, not recommended now, and no sweep is
licensed by the deferral.

### 11.2 Candidate (i): a new LoRA target; additive / multi-target authorship

**Verdicts carried, unchanged.** A new target
(non-readout tensor) stays **DEFERRED** (SI-26 §11.1,
carried by SI-28 §11.2): it needs a full backward
derivation plus a per-target finite-difference gradient
check, and SI-29 sharpens rather than softens the
obstacle — the composition diagnostics themselves are
readout-geometry objects (`m1`, `m2` are last-position
logit movements, and α in §7 is defined through them),
so a candidate whose update changes hidden states is
not even *described* by the instruments §5 reasons
with. Additive and multi-target authored candidates
stay **REJECTED as a next slice** (SI-26 §11.2, carried
by SI-28 §11.2): they require loosening
`max_targets_per_delta` / `allow_additive` policy
defaults with zero authored-behaviour evidence, and
five REJECTs under the defaults are not evidence *for*
loosening them.

---

## 12. Candidate (j): transformer squash; candidate (k): Glimmer probing

### 12.1 Candidate (j): transformer squash for miniature deltas

**Verdict up front: DEFERRED**, on the same ground as
SI-28 §12.1, now firmer: the precondition — an authored
two-delta stack — remains unmet after *five* measured
attempts (SI-21, SI-23, SI-25, SI-27, SI-29), and
SI-12's fail-closed refusal therefore still has nothing
authored to squash. SI-29 adds one design fact a future
squash RFC inherits, extending the one SI-28 §12.1
recorded: a landed stack's deltas may now carry any
combination of recorded signal, anchoring, *and* any of
three composition configurations — and if SI-31 lands
its cycle, a fourth, whose second delta was
*constructed* by a projection step the squash gate
context must be able to represent — so that context,
still demo-shaped, must be designed against three
recorded axes, one of whose values denotes a
construction, not only a loss. Recorded here so the
future RFC meets it as a known obligation. Deferred;
§14 builds nothing toward it.

### 12.2 Candidate (k): probing Glimmer — REJECTED, not reopened

**Verdict up front: REJECTED.** Rejected at SI-13
§7/§9, restated at SI-18 §5, SI-20 §7, SI-22 §7.2,
SI-24 §7.2, SI-26 §12.2, and SI-28 §12.2; SI-29 changes
nothing: the composition machinery — now including the
projection this RFC recommends — is a miniature
mechanism and claims nothing at Glimmer scale, the
Glimmer seed is untouched, and its deltas still
escalate with checks 3/4 "not evaluated". There is no
forward path for Glimmer-scale bundles in the current
design — no probe suite, no forward path, no gate
reading beyond escalation — and this RFC does not
propose faking one, by miniature results, by analogy,
or by any other route. Success *or* continued failure
on the miniature does not soften the answer; §18
item 9 records the pressure honestly so it is met as a
standing question, not a surprise. Not reopened.

---

## 13. Decision summary

| Candidate | Verdict | One-line reason |
|---|---|---|
| (a) Small real safetensors base | **DEFERRED** | SI-18 §3.7 audit: A-1 to A-4 all still unmet, unchanged by SI-29; naming a real base needs a download this scope forbids and an operator decision this RFC does not take (§3). |
| (b) Worst-input composition-aware training | **DONE — as SI-29, closed** | Built and measured (PR #34): REJECT at 0.31215621903538704 with F1′ measured, F2′ not measured as defined, F3′ not measured under the pre-registered rule; the *penalty-statistic* family is closed, candidate-changing moves are not (§4). |
| (c) Another point on the worst-input axis (λ_w / k / τ) | **DEFERRED (open questions)** | F1′ settled the worst case at the statistic, not the weight; k = 4's cluster argument stands un-unargued; the τ form collapses to the falsified mean at natural scale or to a fitted dial — and all three keep a falsified mechanism (§6). No sweep is licensed. |
| (d) Projection form (train-partition orthogonalization) | **RECOMMENDED — SI-31** | The only candidate that changes the candidate rather than the statistic — the move RFC SI-28 §6.6 named in advance for the outcome SI-29 measured — phased measurement-first, exactly one variable changed against SI-29's treatment as its re-run control (§7). |
| (e) Per-probe anchoring | **SPLIT, carried: gate-side REJECTED; probe-side training REJECTED; per-input accounting on allowed inputs ADOPTED inside SI-31** | The first two readings reprice or read the judge; SI-29 supplies no new evidence for either, and both stay closed (§8). The third is §7.4's Phase 0 (§8.3). |
| (f) Probe-partitioned sequencing | **REJECTED (carried)** | Probe-derived partitioning reads the judge (invariant 9); signal-derived partitioning is SI-26 §10's gate-fitting search worn by a sequence, and SI-29's transfer record strengthens the rejection (§9). Not reopened. |
| (g) Magnitude discipline | **REJECTED as a slice (carried); ADOPTED as accounting** | Shrinking `m2` by budget, steps, or scale is the standing gate-fitting non-goal (SI-26 §16); SI-29's partly-by-shrinking improvement makes the norm accounting — including pre-/post-projection norms — more mandatory, not less (§§5.3, 10). |
| (h) A different λ for the mean penalty | **DEFERRED (open question, carried)** | F2 settled the mean at the statistic, not the weight; two parity-weight objectives have now failed at the binding probe in the same direction; a future λ is a future RFC's single argued value (§11.1). |
| (i) New LoRA target / additive / multi-target | **DEFERRED / REJECTED as next slice (carried)** | New target still lacks its backward derivation and is not described by the readout-geometry diagnostics, α included; additive still buys its premise with a policy loosening (§11.2). |
| (j) Transformer squash (miniature) | **DEFERRED** | Precondition still unmet — no authored two-delta stack exists after five measured attempts — and the squash gate context must now be designed against three recorded axes, one value of which denotes a construction (§12.1). |
| (k) Glimmer probing | **REJECTED** | Rejected at SI-13 §7/§9 and restated in every successor RFC; SI-29 changes nothing, no forward path exists, and none is faked here (§12.2). Not reopened. |

The five routes to a landed second write that the
project has now run — a new signal under settled
semantics (SI-23), the same signal restacked (SI-21),
new target semantics (SI-25), a mean composition term
inside the loss (SI-27), and a worst-input composition
term inside the loss (SI-29) — have all been refused by
the same unchanged accounting, and the last three
refusals are attributed to the same probe-level
mechanism with increasing precision: movements that
oppose in aggregate — and now also at the worst allowed
input — add at the binding probe, by more under each
penalty designed to oppose them, and neither a
mean-priced nor a worst-case-priced penalty reaches a
maximum-priced failure. §7's route is the sixth, and
the first to stop pricing the movement and start
*changing* it: remove the indicted component outright,
as measured where measurement is lawful, re-measure
everything the removal touches, and let the unchanged
gate judge what remains. No seventh route is proposed
by this RFC, and §§8–10 record the three tempting
routes (a per-probe judge, a partitioned sequence, a
shrunken movement) and why — on SI-29's evidence as on
SI-27's — they are still not candidates.

---

## 14. The recommended slice: SI-31 (outline)

**SI-31 — Train-partition-projection composition for
the miniature: signal 2, base-anchored, trained at the
landed head as SI-25 trained it, then constructed by
full removal of the component parallel to the landed
movement as measured on its own train partition —
phased, with the projection rehearsed on existing
candidates and the successor falsification readings
pre-registered before any projected candidate is
built.**

In scope:

1. **Phase 0 — the rehearsal, the successor readings,
   and the correctness tolerance (§§5.5, 7.4).** From
   existing machinery and existing candidates, before
   any new candidate construction is finalised:
   (i) the projection rehearsal for the SI-25 control,
   the SI-27 treatment, and the SI-29 treatment — each
   candidate's train-partition projection coefficient
   α, the parallel component's share of its movement
   (aggregate inner products and norms on both allowed
   partitions, held-out reported only), and the
   rehearsal projected candidates' signal held-out
   records and norms — published as diagnostics on
   already-judged candidates, labelled
   derivations/diagnostics, beside the published
   aggregates; (ii) the slice's pre-registered
   successor readings F1′′/F2′′/F3′′ (§7.5) — F2′′ with
   its transfer and maximum clauses stated and judged
   separately, F3′′ restating SI-29's three clauses
   unchanged in threshold against the re-run SI-29
   control (completeness boundary
   0.0013205629553992592) — stated in the as-built
   before the decisive numbers; (iii) the
   projection-correctness tolerance, stated from (i)'s
   figures. If (i) undercuts the slice's premise, the
   as-built says so and says which slice it turned out
   to be (§7.4).
2. **Projection as a recorded configuration (§7.3):**
   `COMPOSITION_CONFIGS` gains exactly
   `"train-partition-projection"` beside `"none"`,
   `"movement-composition"`, and
   `"worst-input-composition"`; constructor-level on
   `TinyTransformerMind`, settable between cycles by an
   explicit driver call in the shape of
   `set_composition_config(...)`; validated with the
   existing refusal discipline; the intent carries it;
   `train_candidate` dispatches on it. The measured
   coefficient α is recorded in the hyperparams beside
   the signal loss's values; α, any projection fraction,
   and the training step's hyperparameters are not
   separately settable. No autonomous, learned,
   depth-driven, verdict-driven, or probe-driven
   configuration selection exists, and no sweep of any
   projection variant is run.
3. **The construction (§7.2):** under
   `"train-partition-projection"`, `train_candidate`
   trains signal 2's factors against SI-25's
   base-anchored signal loss exactly (no penalty term),
   then constructs the judged candidate by the single
   full-removal projection — coefficient α computed
   from the train partition's aggregate movement inner
   products, folded update Δ2′ = Δ2 − α·Δ1 — and
   re-measures the projected candidate in full (signal
   train and held-out losses, with the non-improvement
   refusal applied to the projected candidate's signal
   held-out loss; candidate norm beside its
   pre-projection norm; the allowed-input composition
   record on both partitions). Under `"none"` the path
   is byte-for-byte SI-25's; under
   `"movement-composition"` byte-for-byte SI-27's;
   under `"worst-input-composition"` byte-for-byte
   SI-29's. No probe input, probe logit, held-out
   sequence, or probe-derived quantity enters the loss,
   the coefficient, the construction, or the refusal
   under any configuration (asserted by test, §15
   criterion 4). The pre-projection candidate is a
   construction intermediate: it is never judged,
   landed, or reported as a run, and its figures are
   labelled as an intermediate's wherever they appear.
4. **The coincidence configuration (§7.2):** configured
   with signal 2, base-anchoring, and the projection
   configuration, a first cycle at base records the
   coefficient and the projected candidate's records
   (α identically zero, measured — `m1` is the zero
   vector at base, so both aggregate inner products in
   α's definition vanish and the construction is the
   identity) and must reproduce the established at-base
   run exactly — candidate blob `sha256:131cc242…45f4`,
   same gate numbers — proving the projection machinery
   correct where it must vanish, before the decisive
   cycle is trusted. Its outcomes are labelled at-base
   facts; they are not the slice's decisive claim.
5. **All three controls, re-run inside the slice
   (§7.6):** the SI-25 control (composition `"none"`)
   must reproduce SI-25's decisive cycle-2 candidate
   exactly (blob `sha256:646d1e8b…d675`, drift
   0.3329036422073841); the SI-27 control (composition
   `"movement-composition"`, λ = 1.0) must reproduce
   SI-27's decisive cycle-2 candidate exactly (blob
   `sha256:1bb277ab…b637`, drift
   0.32509003952145576); and the SI-29 control
   (composition `"worst-input-composition"`, λ_w = 1.0,
   k = 4) must reproduce SI-29's decisive cycle-2
   candidate exactly (blob `sha256:11cab02e…b8db5`,
   drift 0.31215621903538704) — so the projection's
   difference is attributable to the construction step
   and not to the refactor that introduced it.
6. **The projection-correctness check (§7.2):** the
   projected candidate's allowed-input composition
   record is measured directly (not derived) on the
   train partition, and its aggregate inner product
   with `m1` must sit within the Phase-0 tolerance of
   0.0; the same record on the held-out partition is
   published (reported only — the construction never
   reads it). The derivation connecting the
   movement-space coefficient to the folded-update
   construction is shown in the as-built, and the
   check — not the derivation's elegance — is the
   evidence. No new loss exists under this
   configuration, so SI-25's gradient check stands for
   the training step unchanged; if the construction's
   implementation introduces any new differentiable
   path, its finite-difference check is run over every
   one of the 512 factor entries at base AND at head
   `d0001`, tolerance 1e-5 unchanged, and the as-built
   says which case obtained.
7. **The decisive two-cycle run (§7.6):** one reused
   `TinyTransformerMind` through `si.run.run_cycles` —
   cycle 1 exactly as SI-29's cycle 1, then the driver
   explicitly reconfigures the same mind to signal 2,
   base-anchored targets, and the projection
   composition configuration (three recorded selections)
   and runs cycle 2. Exactly two authored cycles are
   attempted and exactly one projection decisive run is
   performed. Cycle 2's verdict is whatever the
   unchanged gate measures; **no ADMIT is required**
   (§15 criterion 7).
8. **Diagnostics, the five-run table, and the
   falsification readings (§§7.5, 15 criterion 8):**
   the full post-verdict diagnostic set against head
   `d0001` with the decisive candidate overlaid —
   per-probe drift contributions, the alignment
   diagnostic (aggregate and at the worst probe, sign
   agreement never alone), folded updates, composed
   capability — published for the treatment beside
   *four* controls (SI-23, SI-25, SI-27, SI-29) in the
   five-run table's form, extended to six runs;
   candidate norm and ‖m2‖ beside every drift figure,
   with the treatment's pre-projection norm beside its
   judged norm (§5.3); the allowed-input composition
   records for the controls and the treatment,
   including the treatment's measured post-projection
   aggregate; the treatment's signal held-out record
   read under the pre-registered F3′′ rule; and the
   F1′′/F2′′/F3′′ readings stated as measured —
   F2′′'s two clauses judged and reported separately —
   whichever way they fall. Worst probes are named;
   worst *allowed inputs* are named by train-partition
   identity and never confused with probes (§17).
9. **VERIFY and unwind, honestly conditioned** (§15
   criterion 10): cycle 1 VERIFY as in
   SI-19/21/23/25/27/29. If cycle 2 ADMITs and VERIFY
   passes, the authored two-delta stack exists for the
   first time and SI-21's fixture-labelled unwind
   evidence is joined by the real thing — claimed
   exactly as measured, no further, and with the second
   delta labelled a projection-constructed candidate
   wherever the stack is described (§17 item 22). If
   cycle 2 REJECTs, no stack exists, nothing is
   unwound, and none is claimed — the
   SI-21/SI-23/SI-25/SI-27/SI-29 posture, unchanged.

---

## 15. Acceptance criteria for SI-31

1. **Recorded composition configuration, four named
   values** — the configuration serves exactly
   `"none"`, `"movement-composition"`,
   `"worst-input-composition"`, and
   `"train-partition-projection"`; every other value
   refuses with the existing `ValueError` discipline at
   the constructor, the setter, the intent, and the
   `train_candidate` dispatch. Recorded selection is
   asserted per cycle on both verdict paths. The
   projection coefficient α is not independently
   settable (asserted: the constructor and the setter
   take no such argument, and no projection-fraction
   argument exists anywhere in the configuration);
   exactly one projection decisive run is performed
   (no sweep, no partial-removal variant).
2. **Phase 0 published before the decisive run is
   trusted** — the projection rehearsal for all three
   existing decisive candidates (coefficients, parallel
   shares, rehearsal held-out records and norms, all
   labelled diagnostics on already-judged candidates),
   the pre-registered F1′′/F2′′/F3′′ reading rules with
   F2′′'s clauses separated and F3′′'s thresholds
   carried from SI-29's rule, and the
   projection-correctness tolerance — all present in
   the as-built ahead of the decisive numbers.
3. **Projection correctness, at the decisive head** —
   the decisive candidate's train-partition aggregate
   inner product with `m1` is *measured* (through the
   forward-path record, not the construction's algebra)
   within the Phase-0 tolerance of 0.0, and published
   beside the same record's held-out figure; the
   construction's derivation is shown in the as-built.
   If any new differentiable path was introduced, its
   gradient check is published as §14 item 6 requires;
   if none was, the as-built states that and SI-25's
   check stands cited, not silently assumed.
4. **Probe-blindness of training and construction,
   asserted** — the coefficient's input set is asserted
   as a set (exactly the train partition, via a spy on
   the coefficient's computation during a real
   construction); the train partition's disjointness
   from both probe suites is re-asserted under the new
   code path; a perturbed-probe-artifact bundle copy
   produces a byte-identical *projected* candidate.
5. **Coincidence at base** — signal 2 base-anchored +
   projection-configured at base reproduces the
   established at-base run exactly (blob
   `sha256:131cc242…45f4`, drift 0.16672966815531254,
   regression 0.2418281203135848), with the coefficient
   measured at α = 0.0 and the construction measured as
   the identity on both partitions.
6. **Determinism per cycle** — a full decisive run
   repeated from the seed reproduces both candidates'
   blobs/records (the projected candidate's blob
   included), the landed delta record, and the
   journaled gate measurements exactly; hyperparams
   equal except the labelled `wall_time_seconds`. The
   projection construction alone, applied twice to the
   same pre-projection candidate, is likewise
   byte-identical across repeats.
7. **Gate honesty per cycle** — each cycle's journaled
   checks 3/4 equal an independent `ProbeContext`
   evaluation of the same candidate against the
   pre-cycle head (abs 1e-12) — for cycle 2, of the
   *projected* candidate, the only candidate the gate
   is ever shown. Cycle 2's verdict is reported **as
   measured**; no ADMIT is required: SI-31 succeeds by
   §7's claim — a projection-constructed candidate was
   genuinely authored against the landed head under the
   new configuration and genuinely judged, with its
   premise's falsification readings published either
   way.
8. **Cumulative accounting, diagnostics, the six-run
   table, and the falsification readings published** —
   the side-by-side accounting table (seed-absolute /
   cumulative identity; head-incremental capability),
   the full diagnostic set with candidate norm and
   ‖m2‖ beside every drift figure and the treatment's
   pre-projection norm beside its judged norm, the
   Phase-0 and decisive allowed-input composition
   records, the treatment's signal held-out record read
   under the pre-registered F3′′ rule, the six-run
   table with each figure under its own run's labels,
   and the F1′′/F2′′/F3′′ readings with F2′′'s clauses
   reported separately — all published, measurements
   and derivations labelled, worst probes named, worst
   allowed inputs named as allowed inputs, nothing in
   the diagnostic set reported "not evaluated" that was
   computable.
9. **First-cycle and all-control reproduction** —
   configured with signal 1 (default signal, default
   anchoring, default composition configuration), a
   first cycle reproduces SI-19/SI-21/SI-23/SI-25/
   SI-27/SI-29 exactly (blob `sha256:2761e8ff…6ca9e`,
   drift 0.17189218290150166, regression
   0.1657181391492486); the re-run `"none"` control
   reproduces SI-25's decisive candidate exactly; the
   re-run `"movement-composition"` control reproduces
   SI-27's decisive candidate exactly; the re-run
   `"worst-input-composition"` control reproduces
   SI-29's decisive candidate exactly — the
   configuration machinery proven behaviour-preserving
   for all three existing semantics before the new
   configuration's results are trusted.
10. **VERIFY, unwind, and untouched scope** — cycle-1
    VERIFY as in SI-19; unwind honestly conditioned
    (no authored stack unless cycle 2 lands; none
    unwound or claimed otherwise). Exactly two authored
    cycles attempted in the decisive run and exactly
    one projection decisive run performed;
    `max_stack_depth` 8, `identity_tol` 0.25,
    `capability_eps` 0.25, `max_targets_per_delta` 1,
    `allow_additive` false and every other policy
    default unchanged (asserted); both signals' ids,
    streams, corrections, and partitions unchanged
    (asserted); both anchorings' and all three prior
    composition configurations' definitions unchanged
    (asserted); `si squash` still fails closed on the
    resulting transformer stack (asserted; SI-12's
    suite green); the full existing suite is green;
    the demo path is unchanged; `si serve` still
    returns 501 for every transformer bundle (existing
    suite); no CLI wiring added.

---

## 16. Non-goals

Explicit, and binding on SI-31 and on any slice this
RFC is cited to justify:

- **No Glimmer anything.** No Glimmer forward path,
  probes, training, or authorship (§12.2). No candidate
  in this RFC touches the Glimmer seed except to
  restate that it is untouched, and §7's projection is a
  miniature mechanism that claims nothing at Glimmer
  scale.
- **No real-base work.** Candidate (a) is deferred with
  audited preconditions (§3); SI-31 builds nothing
  toward it, and no download is performed or scheduled.
- **No signal redefinition, no third signal.** Both
  signals' ids, streams, corrections (scale 0.15
  included), partitions, target, rank, and alpha are
  byte-for-byte as SI-19/SI-23 defined them. The
  projection changes how a known signal's trained
  candidate is *constructed* before judgement; it does
  not change what either signal *is*, and no third
  signal is introduced (§§8–9 rejected the per-probe
  and partition forms of one).
- **No probe contact in training or construction.** No
  probe input, probe logit, probe movement, or
  probe-derived quantity enters any training loss,
  penalty, objective, projection coefficient,
  construction step, initialisation, stopping rule,
  sequencing decision, or candidate selection
  (§§2, 8, 9). The binding probe is never named in
  training or construction code, and no per-probe
  target, weight, anchor, or coefficient exists in
  either path. This non-goal has no exceptions and no
  "diagnostic-only" carve-out for the training or
  construction paths: diagnostics read probes after the
  gate has spoken, as they always have; training and
  construction never do.
- **No gate change of any kind.** The gate's checks,
  verdict mapping, accounting (seed-absolute cumulative
  identity maximum; head-incremental capability),
  tolerances, caps, and every policy default are
  unchanged (§8.1). If the projection cycle is
  REJECTed, the answer is a later RFC's better
  argument — never a wider, repriced, per-probe, or
  quieter gate.
- **No parameter tuning of any composition axis.** No
  sweep of λ, λ_w, k, or τ (§§6, 11.1); no second
  decisive run at a different weight, tail width,
  temperature, projection fraction, or coefficient
  definition; no reporting of an un-run parameter's
  hypothetical outcome as anything but an open
  question. A future parameter — for any objective or
  for the projection — is a future RFC's argued,
  single, pre-registered value. The projection's
  removal is **full** by definition (§7.2): a partial
  removal is a tuned dial aimed at the headroom, and
  is this non-goal's clearest instance.
- **No gate-fitting variant of any other kind.** No
  reduced correction scale, no reduced step count or
  norm budget chosen to shrink `m2` (§10), no signal,
  anchoring, configuration, construction, or sequence
  chosen backward from the identity tolerance (§9),
  and no selection rule a gate verdict can influence.
  In particular: no blending of a penalty term with
  the projection in the decisive run (two mechanisms,
  two variables, one blended record), and no
  projecting a *penalised* candidate as the decisive
  treatment (§7.2).
- **No other widening.** No new LoRA target (§11.2),
  no additive or multi-target authored candidates
  (§11.2), and no policy default changes — in
  particular `max_targets_per_delta` stays 1 and
  `allow_additive` stays false.
- **No autonomous selection — of signal, of anchoring,
  or of composition configuration.** The mind chooses
  none of them; all three are explicit, recorded,
  driver-level configurations (§7.3), and no
  perception-driven, depth-driven, alternating,
  verdict-driven, or worst-probe-driven rule is built
  or implied for any axis.
- **No transformer squash.** SI-12's fail-closed refusal
  stands throughout SI-31 (§12.1); §12.1 is a deferral
  analysis, not an implementation brief — even if
  SI-31 creates the authored stack squash has been
  waiting for.
- **No serving, no CLI wiring.** Transformer bundles
  keep returning 501; `si cycle`/`si run` keep their
  demo/mock minds. SI-31 is a library-and-tests slice
  like SI-19, SI-21, SI-23, SI-25, SI-27, and SI-29.
- **No third cycle, no continual-learning claim.**
  Exactly two authored cycles are attempted in the
  decisive run, on constructed signals, on a constructed
  model, under the recorded configurations §14 scopes.
  Three configuration axes at depth ≤ 2 prove something
  about the loop's machinery and its gate — nothing
  about learning at any other scale or on any real task.
- **No mixed-configuration stack claim.** The decisive
  run's cycle 1 is penalty-free at base only in the
  degenerate sense that both composition terms vanish
  there and the projection is the identity there;
  SI-31 does not train or construct a projection cycle
  at a non-base head inside the decisive run as a
  claimed stack member beyond its cycle 2 (the §14
  item 5 controls and the §14 item 1 rehearsal are
  separate runs and diagnostics, labelled as such), and
  no claim is made about stacks whose cycles mix
  composition configurations at non-base heads.

---

## 17. Honesty and labelling requirements for any future slice

Items 1–21 restate RFC SI-28 §17 (itself restating
SI-26 §17 and SI-24 §12 with additions) in compressed
form, with the numbering preserved so existing
citations resolve; items 22–23 are new in this RFC.
All twenty-three bind SI-31 and any slice this RFC is
cited to justify.

1. **Training numbers are training numbers.** Losses,
   penalty, objective, and construction values are
   labelled with signal, partition, head, target
   source, anchoring, and composition configuration;
   none is a gate or probe result.
2. **Measured vs not evaluated survives authorship.**
   "Not evaluated" is never rendered as 0.0, a pass, or
   an omission.
3. **Authorship is not admission.** No document, test
   name, or journal field describes a candidate as
   landed before the gate lands it.
4. **Every number carries its source.** Each figure
   names the run, configuration, and document that
   measured it; inherited figures are cited, not
   re-derived silently.
5. **The miniature's labels travel.** "Constructed,
   deterministically seeded, not trained" accompanies
   the miniature wherever its results are summarised.
6. **Failure states are reportable successes of the
   experiment.** A REJECT with attribution is a
   publishable result, stated as plainly as an ADMIT.
7. **Serialisation determinism is proven, not
   assumed.** Blob-hash equality is the evidence;
   near-equality is a failure, reported as one.
8. **Head-relative numbers name their head.** Any
   figure measured against a head names that head.
9. **Incremental and cumulative are different
   numbers.** Never presented as each other, in either
   direction.
10. **Composed artifacts do not inherit training
    claims.** A landed stack's properties are what its
    VERIFY measured, not what its cycles' training
    records hoped.
11. **Signal identity is part of every label.** A
    number without its signal id is a labelling defect.
12. **Diagnostics are not verdicts.** The diagnostic
    set attributes a verdict after the gate has
    spoken; it is not a gate check, it influences no
    verdict, and none of it is a prediction.
13. **Recorded selection is not a choice the mind
    made.** Signal, anchoring, and composition
    configuration are driver selections; no text
    describes the mind as choosing, deciding, or
    preferring any of them.
14. **A derivation is not a prediction.** Derived
    quantities (movement columns, sums at a probe,
    rehearsal projections) are labelled derivations
    wherever they appear.
15. **Anchoring is part of every label.** A training
    or diagnostic number recorded under one anchoring
    is never presented as a number under the other.
16. **Controls and treatments are never blended.**
    Each figure stands under its own run's labels; a
    three-, four-, five-, or six-run table keeps the
    rows separate.
17. **Composition configuration is part of every
    label.** A number recorded under one composition
    configuration — now including
    `"worst-input-composition"` and, for any slice this
    RFC is cited to justify,
    `"train-partition-projection"` — is never presented
    as a number under another.
18. **Allowed-input numbers are not probe numbers.**
    Figures from allowed populations and figures from
    judged populations are never interchanged,
    averaged together, or quoted in each other's place;
    any gap between them is stated as the difference
    between two separately labelled records.
19. **Sign agreement never stands alone.** It is
    published only beside the magnitude-weighted figure
    (the inner product) for the same population.
20. **A worst-case figure names its population.** A
    worst *allowed input* (train partition, held-out
    partition) and the worst *probe* are different
    quantities over disjoint populations, and no text
    may shorten either to "the worst case" without its
    population attached.
21. **Judgement-call readings are pre-registered.**
    Where a falsification reading turns on a qualitative
    boundary in a quantitative record, the boundary is
    stated in advance, in the record's own terms, before
    the decisive run it will judge — and the reading is
    then applied as written. A reading invented after
    the numbers are known is reported as an observation,
    never as the falsification test.
22. **A constructed candidate is labelled constructed
    (new).** A candidate produced by a construction
    step after training — the projection of §7 is the
    standing instance — is labelled with that step
    wherever it, its stack, or its figures are
    described; its pre-construction intermediate's
    figures are labelled an intermediate's and are
    never presented as the judged candidate's record,
    and the judged candidate's figures are never
    presented as a purely *trained* candidate's. A
    maximum that improves across a construction step is
    published with both norms (§5.3), so improvement by
    removal is distinguishable from improvement by
    turning and from improvement by shrinking.
23. **An unmeasured-as-defined reading stays
    unmeasured (new).** A falsification reading whose
    registered definition did not fire — F2′ is the
    standing instance — is quoted in its registered
    form ("not measured as defined", with the clause
    that failed named) and never rounded, in summary,
    citation, or successor documents, into "measured",
    into "not measured" without its qualifier, or into
    the separate facts measured beside it. The facts
    beside a reading and the reading itself are two
    separately labelled records (§18's discipline,
    made a labelling rule).

---

## 18. Risks and open questions

Each item names what would retire it. Numbers this RFC
deliberately does **not** supply are listed as open
questions to measure — supplying them here would be
inventing them.

1. **F1′′ is the expected outcome, and the slice must
   be built to publish it.** The transfer premise — a
   quantity of `p(x)` on allowed inputs determines the
   corresponding quantity on judged inputs — has now
   been falsified at the aggregate (SI-25's destination
   argument), at the mean (SI-27's F1), and at the
   worst case (SI-29's F1′). The projection is the same
   premise in a stronger form: not pricing the parallel
   component but *removing* it, as measured on the same
   allowed population. This RFC recommends the
   experiment because the projection is the only
   remaining move that changes the candidate itself and
   because RFC SI-28 §6.6 named it in advance for
   exactly this outcome — not because transfer has
   become more likely. A slice that reports F1′′ as a
   surprise has misread its own RFC. *Retired by:*
   SI-31's F1′′ reading, published either way (§15
   criterion 8).
2. **The projected candidate's properties are
   unmeasured during its training — by construction.**
   This is SI-28 §7's standing objection, carried into
   the recommended slice as its central design risk:
   the training step optimises a candidate the gate
   never sees, and the construction then removes a
   component the training never priced. §7.2's answer
   is full post-construction re-measurement (including
   re-applying the non-improvement refusal to the
   projected candidate) and §7.4's answer is the
   Phase-0 rehearsal — but both are accountings *after*
   the fact of construction, and neither makes the
   projected candidate an *optimised* whole. If the
   rehearsal shows the removable component is small,
   the slice measures a near-control; if it is large,
   the slice measures a candidate whose destination
   the projection may have removed with the component.
   **Open question to measure.** *Retired by:* Phase 0
   itself (§14 item 1), and by SI-31's published
   pre-/post-projection records (§15 criterion 8).
3. **The destination trade may deepen past the rule's
   boundary (F3′′).** SI-29's treatment already met
   F3′'s closer-fraction clause *exactly* (12/16) with
   the least complete approach on record; a projection
   removes part of the trained candidate outright, and
   the removed component is, by the signal family's
   geometry, partly the destination's own direction
   (both corrections share a component, §5.2). Whether
   the projected candidate still approaches — under
   thresholds this RFC deliberately carries forward
   unchanged — is genuinely unknown in both directions:
   removing the parallel component could spare the
   destination (the component removed is defined
   against `m1`, not against the signal) or spend it.
   *Retired by:* SI-31's held-out record read under
   the pre-registered rule (§§7.5, 15 criterion 8) —
   the second F3-class reading in the project taken
   against a boundary stated in advance, and the first
   whose thresholds were carried across a slice
   unchanged after being met exactly.
4. **Phase 0 may undercut the recommendation.** If the
   rehearsal shows the train-partition parallel
   component is negligible in all three existing
   candidates, §7's premise narrows to "the component
   training may lawfully remove was never the binding
   quantity on allowed inputs either" — in which case
   SI-31's honest product is the Phase-0 measurement
   plus an F1′′ it can predict but must still measure
   (a projected candidate whose construction is
   near-identity is a measurement of exactly that).
   This RFC accepts that outcome in advance; §7.4 binds
   the as-built to say which slice it turned out to be.
   *Retired by:* Phase 0 itself (§14 item 1).
5. **F2′'s definitional lesson is methodological, not
   only local.** A falsification reading written as a
   conjunction of an aggregate clause and a maximum
   clause failed to fire when its clauses split — the
   project's second reading (after SI-27's F3) to land
   in an unpriced definitional gap, in the opposite
   direction (F3 was qualitative where a boundary was
   needed; F2′ was conjunctive where separability was
   needed). SI-31's F2′′ separates its clauses (§7.5),
   but the cadence-level question — how to write
   readings whose clauses cannot split silently —
   remains **an open methodological question** this RFC
   records rather than solves. *Retired by:* SI-31's
   F2′′ reading firing or not firing *as written*, and
   by successor RFCs stating, per reading, what each
   clause's failure separately means.
6. **A mislabelled configuration is worse than a wrong
   one.** The failure mode SI-26 §18 item 6 named — a
   candidate trained under one configuration and
   journaled under another — now has four values to
   confuse instead of three, and two subtler variants
   this slice introduces: a coefficient silently
   computed over the held-out partition (or over any
   probe-derived quantity) would attribute a projection
   effect to a run that never lawfully computed one,
   and a pre-projection intermediate silently judged or
   reported in the treatment's place would blend two
   candidates' records into one row. *Retired by:*
   §15 criteria 1, 4, and 7 — per-cycle journal
   assertions on both verdict paths, the coincidence
   configuration's exact reproduction, *all three*
   controls' exact reproduction, the coefficient
   spy assertion, and the probe-perturbation
   independence test applied to the projected candidate.
7. **The construction path could silently alter signal
   behaviour under the existing configurations.**
   Generalising `train_candidate` to dispatch a
   construction step is the kind of change that passes
   its own new tests while moving an old behaviour in
   its last digits. *Retired by:* §15 criteria 5 and
   9 — exact reproduction of *five* published blobs
   (signal 1's `d0001`, signal 2's at-base `d0001`,
   SI-25's decisive candidate, SI-27's decisive
   candidate, SI-29's decisive candidate) under the
   refactored path, before the decisive run is trusted.
8. **Probe blindness still bounds the judge.** RFC
   SI-13 §10.1 stands, now across signals, anchorings,
   and three — prospectively four — composition
   configurations: 24 inputs, 28/30 tensors measurable,
   blind set derived and guarded. A landed two-delta
   stack under any configuration is *evidence* of
   continuity sampled the same way each cycle, not a
   proof of it.
9. **Depth and Glimmer pressure arrive if SI-31
   succeeds.** At `max_stack_depth` 8 with transformer
   squash still deferred (§12.1), a landed second
   authored delta makes the stack depth 2 and the
   compression question real for the first time — while
   the squash gate context remains unbuilt and must now
   be designed against three recorded axes, one value
   of which denotes a construction (§12.1). And per
   SI-26 §18 item 10: success on the miniature renews
   the "why not Glimmer" pressure, and the answer does
   not soften because the miniature's authorship has
   gained a construction step. There is no Glimmer
   forward path; none is proposed here, and none may be
   improvised by a successor slice (§12.2).
10. **The deferred axes' unknowns remain unknown.**
    Candidate (a)'s A-1 to A-4 are audited unmet (§3);
    nothing in SI-31 or this RFC scopes them, and the
    download question is an operator decision this RFC
    explicitly does not take. The λ_w / k / τ questions
    (§6) and candidate (h)'s λ question (§11.1) likewise
    remain open by design: each is retired only by a
    later RFC arguing a single value from the published
    records of the configurations that now exist, never
    by a sweep. The worst-input selection trajectory
    (§5.5) remains unpublished and unretired; no item
    in SI-31 depends on it.

---

## 19. Summary

After SI-29, SI can originate a miniature delta, judge
it, land it, and originate a second one on top of it
under a second signal, under two target anchorings, and
under either of two composition-aware losses — and the
gate's cumulative, seed-absolute identity accounting
has been measured refusing all five second writes, the
fifth with its cause attributed more precisely than the
first four: the worst-input update *opposes* the first
update at the worst input it may see (tail–bulk gap
closed, worst train `p` −0.0023129050472012187 against
its mean −0.002434403120559355), in aggregate on the
probes, and in weight space, and the composed maximum
still stands at 0.31215621903538704 > tol 0.25, because
at `tiny-id-0` — the probe where cycle 1 had already
moved most — the two movements add almost exactly
(0.17189218290150166 + 0.1409909427165985 ≈ 0.3122),
the worst-input penalty having improved the *worst
case* of a quantity the gate prices at *its* worst
case, on a disjoint population — and having worsened
the alignment where the gate binds
(+0.04944786620153699, the worst on record). F1′ is
measured; F2′ is not measured as defined, its aggregate
clause having failed while the maximum stayed unmoved
beside it; F3′ is not measured under its pre-registered
rule, the destination approached at that rule's exact
boundary. Of the candidate frontiers this RFC evaluated
against that evidence: a small real base stays
deferred, its four SI-18 preconditions audited and all
still unmet, with a checkpoint download flagged as an
operator decision rather than a step (§3); the
worst-input penalty is done and closed, and with it the
penalty-statistic family (§4); another point on the
worst-input axis — a different λ_w, a different k, the
τ form — is deferred as open questions no current
evidence argues, with no sweep licensed (§6); per-probe
anchoring stays rejected in its gate-side and
probe-side readings — repricing the judge, and reading
the judge — with SI-29 supplying no new evidence for
either, and is adopted only as per-input accounting on
allowed inputs inside the recommended slice (§8);
probe-partitioned sequencing stays rejected on the
invariant and on the gate-fitting trap (§9); magnitude
discipline stays rejected as a training dial and is
adopted as mandatory accounting, sharpened by SI-29's
partly-by-shrinking improvement (§10); a different λ
for the settled mean objective stays deferred (§11.1);
a new target stays deferred and additive authorship
stays rejected (§11.2); transformer squash stays
deferred, its precondition still unmet after five
measured attempts (§12.1); Glimmer probing stays
rejected, not reopened, with no forward path existing
and none faked (§12.2); and **train-partition
projection — signal 2, base-anchored, trained at the
landed head exactly as SI-25 trained it, then
constructed by full removal of the component parallel
to the landed movement as measured on its own train
partition (coefficient α measured, never settable),
recorded as a fourth value on the composition axis
beside signal and anchoring, phased so the projection
is rehearsed on all three existing decisive candidates
and the successor falsification readings — with F2′′'s
clauses separated and F3′′'s thresholds carried
unchanged — are pre-registered before any projected
candidate is built, with the projected candidate
re-measured in full before the unchanged gate judges
it on probes that neither its training nor its
construction ever reads, with its premise stated in
three falsifiable forms — is recommended as SI-31**
(§§7, 14, 15). Its plausibility argument is stated with
its limits in the same sections that make it: SI-27
and SI-29 proved a second update's composition *can* be
shaped on allowed inputs, at the mean and at the worst
case, and proved just as plainly that shaping it by
penalty leaves the maximum to the signal family's
geometry — and pushes the binding probe the wrong way.
Whether *removing* the parallel component outright —
the gate's indicted quantity, on the only population
construction may read — moves the one figure the gate
prices is measurable with instruments that already
exist, against four published controls, with exactly
one variable changed. That is a plausibility argument
for running the experiment — nothing in this section is
a prediction that it ADMITs, and §15 requires no ADMIT;
§18 item 1 records that the transfer premise has been
falsified at three weaker forms already, and this RFC
recommends measuring the strongest form with its eyes
open. The deepest landed authored stack remains
depth 1; no two-delta authored stack exists; the base
stays frozen, the bytes are never duplicated, no
tolerance moves, the gate is unchanged, and nothing is
claimed that a test in this repo cannot check.

---

*RFC SI-30 — design only. Implementation begins, if
approved, at SI-31 as outlined in §14. RFC SI-26
changed what the loop accounts for while it trains —
the first movement it already made; RFC SI-28 changed
the statistic that accounting answers to — the worst
input it is allowed to see; the gate told the truth
about both, per probe: the accounting worked almost
everywhere and failed where it binds, by more where it
binds under each penalty built to fix it. This RFC
proposes changing neither what the loop accounts for,
nor the statistic it answers to, nor where it may look,
but the candidate itself: remove the component the
record indicts, measure everything the removal touches,
and let the same judge price what remains — under the
same judge, at the same scale, with the verdict still
not ours to pick.*
