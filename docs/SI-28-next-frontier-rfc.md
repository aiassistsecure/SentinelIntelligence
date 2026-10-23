# SI-28 RFC — The Next Frontier After Movement-Composition-Aware Training
(Design Only)

> **Status: DESIGN ONLY.** This document evaluates candidate
> directions and recommends one next slice. It changes no
> production code, no test, no manifest schema, no policy
> default, and no gate behaviour. Nothing in this RFC is
> implemented unless a section explicitly says it exists today,
> in code, on `main`.
>
> **The one sentence that governs everything below:** SI-27
> proved that composition measured on the inputs training is
> allowed to see can be shaped — decisively, by a penalty term —
> in aggregate on those inputs and in aggregate on the probes,
> and that the gate's verdict still did not move, because the
> gate prices the *worst* probe, not the average one: at
> `tiny-id-0` the penalised candidate's movements align slightly
> *more* than the control's, the composed maximum is
> 0.32509003952145576 > tol 0.25, and RFC SI-26's falsification
> readings F1 and F2 are both *measured* — and this RFC must
> never be cited as evidence that any worst-input objective has
> been implemented or tested, that any authored two-delta stack
> exists, that a cycle-2 ADMIT is achievable by any route, or
> that any slice proposed here exists.
>
> **Scope discipline:** CPU only. No GPU, no cloud, no spend,
> no downloads. Terminology: this is machine learning
> infrastructure throughout.
>
> **Relationship to prior work:** RFC SI-26 evaluated the
> frontiers open after SI-25 and recommended
> movement-composition-aware training; SI-27 built it (PR #32,
> as-built in `docs/SI-27-movement-composition-training.md`)
> and the gate's cycle-2 verdict was a measured REJECT under
> the composition penalty as it had been under both anchorings
> without one — now attributed three-sidedly: the penalty works
> where it is allowed to look, transfers to aggregate probe
> composition, and fails at the binding probe. This RFC is
> SI-26's successor in the established cadence (SI-18 RFC →
> SI-19 build → SI-20 RFC → SI-21 build → SI-22 RFC → SI-23
> build → SI-24 RFC → SI-25 build → SI-26 RFC → SI-27 build):
> it restates the verified state after SI-27, re-evaluates the
> standing candidates against SI-27's measured evidence, gives
> the question SI-27's diagnostics have made precise — why a
> penalty that improves composition almost everywhere still
> loses at the one probe the gate prices — the deep treatment
> (§5), and recommends exactly one next slice (§14). SI-26's
> rejections of a repriced gate (§8.2 there), of
> binding-probe orthogonalization (§9 there), of a third
> signal (§10 there), and of Glimmer probing are carried
> forward, not reopened, unless SI-27's evidence genuinely
> reopens them; §§7–11 record, per candidate, whether it
> does. One candidate SI-26 explicitly left arguable *is*
> taken up — the projection form of train-partition
> orthogonalization, which SI-26 §9 made conditional on
> exactly the F1/F2 outcomes SI-27 measured — and is
> evaluated on its merits in §7 and deferred on sequencing,
> not on principle. The SI-9 RFC's §7 risks remain in force
> and are inherited, not relitigated.

---

## Table of contents

1. [Where SI stands after SI-27](#1-where-si-stands-after-si-27)
2. [Inherited invariants](#2-inherited-invariants)
3. [Candidate (a): a small real safetensors base — still deferred](#3-candidate-a-a-small-real-safetensors-base--still-deferred)
4. [Candidate (b): movement-composition-aware training (mean penalty) — DONE as SI-27, closed](#4-candidate-b-movement-composition-aware-training-mean-penalty--done-as-si-27-closed)
5. [The worst-probe question — the deep treatment](#5-the-worst-probe-question--the-deep-treatment)
6. [Candidate (c): a worst-input composition objective on allowed inputs](#6-candidate-c-a-worst-input-composition-objective-on-allowed-inputs)
7. [Candidate (d): the projection form — train-partition orthogonalization](#7-candidate-d-the-projection-form--train-partition-orthogonalization)
8. [Candidate (e): per-probe anchoring — three readings, three verdicts](#8-candidate-e-per-probe-anchoring--three-readings-three-verdicts)
9. [Candidate (f): probe-partitioned sequencing](#9-candidate-f-probe-partitioned-sequencing)
10. [Candidate (g): magnitude discipline — a smaller second movement](#10-candidate-g-magnitude-discipline--a-smaller-second-movement)
11. [Candidates (h) and (i): a different λ; new target / additive](#11-candidates-h-and-i-a-different--new-target--additive)
12. [Candidate (j): transformer squash; candidate (k): Glimmer probing](#12-candidate-j-transformer-squash-candidate-k-glimmer-probing)
13. [Decision summary](#13-decision-summary)
14. [The recommended slice: SI-29 (outline)](#14-the-recommended-slice-si-29-outline)
15. [Acceptance criteria for SI-29](#15-acceptance-criteria-for-si-29)
16. [Non-goals](#16-non-goals)
17. [Honesty and labelling requirements](#17-honesty-and-labelling-requirements-for-any-future-slice)
18. [Risks and open questions](#18-risks-and-open-questions)
19. [Summary](#19-summary)

---

## 1. Where SI stands after SI-27

Everything in this section is verifiable in the code and
docs on `main` at the SI-27 merge (main tip `90ded5a`,
the merge of PR #32). Numbers are cited to the as-built
document that measured them; this RFC measures nothing
new. Where a statement below is a *derivation* from cited
facts rather than a measurement, it is labelled a
derivation, exactly as SI-26 §1 labelled its arithmetic.

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
- **Two recorded composition configurations (SI-27).**
  `COMPOSITION_CONFIGS` serves exactly `"none"` (the
  default) and `"movement-composition"`, validated by
  `validate_composition_config(...)` with the existing
  refusal discipline (`si/tiny_training.py`). Under
  `"movement-composition"` the training loss gains
  exactly one term — λ · mean over the signal's train
  partition of the per-coordinate mean product `p(x)`
  between the landed head's existing movement `m1(x)`
  and the candidate movement `m2(x)` — with
  `COMPOSITION_LAMBDA = 1.0` fixed in exactly one place,
  part of the configuration's definition and never read
  from an intent or any other caller input. Composition
  is a recorded per-cycle configuration beside recorded
  signal selection and recorded target anchoring: a
  constructor-level `composition_config` on
  `TinyTransformerMind`, settable between cycles by the
  explicit driver call `set_composition_config(...)`;
  the gate's `Intent` carries the value in `to_dict()`
  so the journal shows all three selections. Under
  `"none"` the training path is byte-for-byte SI-25's.
  No autonomous, learned, depth-driven, or
  verdict-driven composition selection exists.

**What SI-27 measured (cited to its as-built):**

- **The decisive run.** One reused
  `TinyTransformerMind` through `si.run.run_cycles`:
  cycle 1 (signal 1, default anchoring, default
  composition configuration) ADMITted to `d0001`,
  reproducing SI-19/SI-21/SI-23/SI-25 exactly — candidate
  blob `sha256:2761e8ff…6ca9e`, identity drift
  0.17189218290150166, capability regression
  0.1657181391492486, §6.6 VERIFY measured-pass. Cycle 2
  (signal 2, base-anchored, composition
  `"movement-composition"`, λ = 1.0, trained at `d0001`)
  was REJECTED on cumulative identity drift; nothing
  landed; the bundle head after the run is `d0001`.
- **The penalty works where it is allowed to look.**
  On signal 2's train partition the penalty term moved
  from +0.00017392083908663979 at initialisation to
  −0.0026247611759719963 at the end of training, and the
  allowed-input aggregate inner product deepened from
  the re-run control's −1.6873837019991191 to the
  treatment's −2.687755446349435 (held-out partition,
  reported only: −0.845596374259425 →
  −1.3460015097625617). The total loss ends **negative**
  (−0.0021056259086710467), driven by the penalty term —
  a measured property of the signed term, published in
  the as-built, not hidden.
- **It transfers in aggregate on the probes.** The
  per-probe aggregate inner product over the 8 identity
  probes is −0.3271477846470207, more opposing than
  SI-25's control's −0.21256230517288188 (SI-23's was
  +0.07637087369869666, aligned); sign agreement 0.390625
  (100/256), published beside the magnitude-weighted
  figure. The folded weight-space inner product is
  −0.02013451284871203, opposing more strongly than
  SI-25's −0.011457914936582613.
- **It fails where the gate binds.** The composed
  maximum is 0.32509003952145576 > tol 0.25, at
  `tiny-id-0`, where the two movements' inner product
  is +0.03801222744105431 — positive, aligned, and
  slightly *worse* than SI-25's control's
  +0.0338873784299374 at the same probe (sign agreement
  0.4375, 14/32). At `tiny-id-0` the composed drift is,
  to the printed digits, the sum of cycle 1's drift
  there and the second movement's increment there
  (0.17189218290150166 + 0.15392476320266724 ≈ 0.3251) —
  the aligned extreme, for the third decisive run in a
  row at the probe where cycle 1 moved most (SI-27's
  as-built states this; §5.4 examines what it does and
  does not license).
- **Per probe, the treatment is a trade, not a uniform
  improvement.** Composed drift improves on SI-25's
  control at 4 of 8 identity probes (tiny-id-0 only
  marginally, 0.3329036422073841 → 0.32509003952145576;
  tiny-id-2, tiny-id-3, tiny-id-4 substantially) and
  worsens at 4 (tiny-id-1 marginally; tiny-id-5,
  tiny-id-6, tiny-id-7 substantially). The penalised
  second movement is the largest on record: candidate
  norm 0.2969693161862222 (SI-25: 0.2515524481203623;
  SI-23: 0.21317186599953494), probe-space ‖m2‖
  0.9688074699782941 (SI-25: 0.7816722548913912), and its
  incremental movement exceeds cycle 1's remaining
  headroom (0.07810781709849834) at 7 of 8 probes.
- **The destination cost is real and partial.** The
  penalised candidate's signal held-out loss falls from
  0.002757805049077469 to 0.0005227812955486338 (~5.3×,
  16/16 examples closer) against the control's ~264×
  fall to 1.043608588752459e-05; its signal train
  residual (0.0005191352673009495) is ~55× the control's
  (9.412649442227854e-06). The destination is still
  approached; it is approached far less completely.
- **The falsification readings (RFC SI-26 §7.6), as
  measured:** F1 **measured** — at the worst probe, the
  penalty does not transfer. F2 **measured** — the
  penalty transfers in aggregate and the maximum does
  not move below tol. F3 **not measured as defined** —
  the destination is still approached, so F3's
  falsification does not fire; a partial destination
  trade is measured beside it (§5.5 states what
  completing F3's definition requires).
- **The machinery is proven where it must vanish and
  where it must not look.** The coincidence
  configuration (signal 2, base-anchored, penalised, at
  base) reproduces SI-25's at-base run exactly (blob
  `sha256:131cc242…45f4`, penalty identically zero on
  both partitions). The penalised loss's gradients pass
  central finite differences over every one of the 512
  factor entries at both base (worst
  5.5249277028738815e-08) and the decisive head `d0001`
  (worst 5.149878699880166e-08), tolerance 1e-5
  unchanged. Probe-blindness is asserted three ways,
  including a byte-identical candidate from a
  perturbed-probe-artifact bundle copy. The full suite
  at the SI-27 merge: **340 passed, 2 skipped**.

**The four-run cycle-2 record** (each figure under its
own run's labels; SI-27's as-built publishes the last
three side by side, and SI-21's figure is cited to its
own as-built):

| Run | Cycle-2 configuration | Composed identity drift (cumulative) | Worst probe | Verdict |
|---|---|---|---|---|
| SI-21 | signal 2, weights-under-training, composition n/a (pre-SI-27) | 0.34378434903919697 | (SI-21 as-built) | REJECT |
| SI-23 | signal 2, weights-under-training, composition n/a | 0.2963416576385498 | tiny-id-3 | REJECT |
| SI-25 | signal 2, base-anchored, composition `"none"` | 0.3329036422073841 | tiny-id-0 | REJECT |
| SI-27 | signal 2, base-anchored, composition `"movement-composition"`, λ = 1.0 | 0.32509003952145576 | tiny-id-0 | REJECT |

Four authored second writes, four measured REJECTs, no
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
RFC SI-26 §2, restated for the candidates):

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
   another (SI-26 §17 item 17); a worst-case figure
   names the population it is a worst case *over*
   (§17 of this RFC).
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
   make an outcome pass. The four cycle-2 REJECTs are
   measurements to design *around* — with new training
   semantics — never numbers to outvote, and never
   numbers to re-price by changing what the gate
   measures (SI-26 §8.2, carried at §8 of this RFC).
9. **The probes are the judge, never the training
   set** (in force in fact since SI-19, stated at
   SI-26 §2). No candidate in this RFC may train on,
   penalise against, project against, or otherwise
   shape a candidate using any probe input, probe
   logit, or probe-derived movement. A method that
   reads the judge's inputs to shape the judged
   artifact has not passed the judge; it has memorised
   the courtroom. SI-26 §9's rejection rests on this
   invariant, SI-27's design was constrained by it at
   every step, and every candidate below is audited
   against it first.

---

## 3. Candidate (a): a small real safetensors base — still deferred

**Verdict up front: DEFERRED**, unchanged from SI-26 §3.
The SI-18 §3.7 audit's four preconditions (A-1 to A-4)
remain unmet, and SI-27's evidence does not touch any of
them: the miniature's composition problem is a property
of two rank-8 readout updates and a maximum-priced
identity arm, measured on a constructed model; nothing
about it argues that a real base would compose
differently, and naming a real base requires a download
this scope forbids and an operator decision this RFC
explicitly does not take (SI-26 §18 item 10, carried).
SI-27 does add one sentence to the audit's context, and
it cuts *against* urgency, not for it: the project now
owns a training-side composition instrument (two
configurations, one measured negative result) whose
behaviour at any other scale is unclaimed — moving the
question to a real base before the miniature's
worst-case question is answered would carry an
unanswered question to a more expensive room. Deferred;
the download question stays the operator's.

---

## 4. Candidate (b): movement-composition-aware training (mean penalty) — DONE as SI-27, closed

**Verdict up front: DONE — as SI-27, closed.** Built and
measured (PR #32, as-built
`docs/SI-27-movement-composition-training.md`): signal 2,
base-anchored, composition-penalised at `d0001`, REJECTED
on composed drift 0.32509003952145576, with the
attribution published in full (§1). This section exists
so the candidate's closure is on the record in the
cadence's usual place, and to state precisely *what* is
closed, because §6 proposes a relative of this candidate
and the two must not be confused:

- **Closed:** the *mean* composition objective — one
  penalty term equal to λ times the mean of `p(x)` over
  the train partition, at λ = 1.0. Its premise is settled
  in both directions by SI-27's readings: it shapes
  allowed-input composition (F1's near end, measured),
  it transfers to aggregate probe composition (F2's
  first clause, measured), and neither moves the judged
  maximum below tol (F2's second clause, measured). No
  re-run of the mean objective at the same λ is a
  frontier; it is a control, and SI-29's design (§14)
  uses it as exactly that.
- **Not closed:** composition objectives as a family.
  F1 and F2 falsify premises about the *mean*. A
  worst-case objective makes a different falsifiable
  claim — that the binding variable is the worst
  input's composition, not the population's average —
  and SI-27's published record neither confirms nor
  refutes it, because the per-input distribution of
  `p(x)` on allowed inputs was asserted in tests but
  never published (§5.5). §6 evaluates that claim; it
  does not inherit this candidate's closure, and it
  does not inherit its plausibility either — it must
  argue from §5's measurements, as §6 does.

---

## 5. The worst-probe question — the deep treatment

SI-25 made the question *per-probe composition*; SI-27
answered that question for the mean and made the
sharper question unavoidable: **why does a training
intervention that improves composition on the train
partition, on the held-out partition, in aggregate on
the probes, and at four of eight individual probes,
still lose at the single probe the gate prices — and
slightly worsen there?** This section assembles what
the four decisive runs jointly establish, what they
leave unknown, and what any candidate must therefore
measure before it may claim to aim at the binding
figure. Derivations are labelled derivations.

### 5.1 The gate prices a maximum; every intervention so far optimised a mean

The identity arm is seed-absolute, cumulative, and a
**maximum over probes** of per-probe drift (max
absolute logit coordinate per probe, maximised again
over the 8 identity probes). Nothing in SI-21, SI-23,
SI-25, or SI-27 changed that definition, and §8 of
this RFC carries SI-26 §8.2's rejection of changing
it. Against that fixed judge, the four cycle-2
candidates were produced by objectives that are all,
in the relevant sense, *averages*: target-logits MSE
over 32 training sequences (SI-21, SI-23, SI-25), and
in SI-27 an added term equal to the *mean* of the
per-input composition product over the same 32
sequences. A mean objective purchases average-case
behaviour. SI-27's record is the cleanest possible
demonstration of the gap: the treatment improved the
mean composition on every population it was allowed to
be measured on, improved the composed drift at half
the probes, and the maximum — the one figure the gate
reads — moved from 0.3329036422073841 to
0.32509003952145576, a ~2.3% improvement against the
~25% fall the tolerance demands (derivation: 0.25 /
0.3329036422073841 ≈ 0.751, so admission required the
maximum to fall by ~25% from the control's value; it
fell by ~2.3%). F2, in RFC SI-26 §7.6's words, settles
the premise: **aggregate composition is not the binding
variable.** Any candidate that optimises another
average must argue why its average binds the maximum;
§6's candidate is designed so that its objective and
the gate's statistic are the same *kind* of quantity —
a worst case — differing only in the population it is
computed over (allowed inputs vs judged inputs), which
is the one difference invariant 9 makes mandatory.

### 5.2 At the binding probe, the movements do not merely fail to oppose — they add

SI-27's per-probe table (treatment, derivation column
labelled there as a derivation from two measured
forwards) shows the mechanism at `tiny-id-0` with no
room for a softer reading: cycle 1's drift there is
0.17189218290150166, the penalised second movement's
increment there is 0.15392476320266724, and the
composed drift is 0.32509003952145576 — the sum, to the
printed digits. The inner product at that probe is
*positive* (+0.03801222744105431): in the coordinates
where cycle 1 moved most, the second movement moves
the same way. And this is not the penalty's accident:
the SI-25 control's inner product at the same probe
was also positive (+0.0338873784299374), and SI-23's
was positive at *its* worst probe. Three different
training semantics — head-relative targets,
base-anchored targets, base-anchored targets with a
mean composition penalty — all produce second
movements that align with the first at whichever
probe binds. The alignment at the binding probe is
therefore a property of the *signal family's geometry
through this readout* (both corrections share the
away-from-the-last-token component, SI-26 §10, and
both updates are rank-8 readout matrices applied to
every hidden state the probes present), not of any
one loss. That is the finding any §6-class candidate
must confront: it is not enough for an objective to
push movements apart *somewhere*; it must push them
apart at the inputs where they currently add, and
training is forbidden by invariant 9 from knowing
which judged inputs those are. The only legal proxy
is the worst *allowed* input — and whether the legal
proxy tracks the judged one is an empirical question
SI-27 leaves open (§5.5), not a premise §6 may assume.

### 5.3 The magnitude half of the problem

Composition is not the whole failure, and this RFC
refuses to let the direction story swallow the size
story. Cycle 1 leaves headroom of
0.07810781709849834 under tol at the binding probe.
SI-27's second movement's increment at `tiny-id-0`
alone is 0.15392476320266724 — twice the headroom —
and its incremental movement exceeds the headroom at
7 of 8 probes. Even a second movement *orthogonal* to
the first at every probe (inner product exactly 0.0,
the strongest composition outcome a penalty could
name) composes, per probe, to roughly the larger of
the two drifts only under favourable coordinate
geometry; the measured record (composed ≈ sum at the
binding probe across three runs) shows the actual
geometry is the unfavourable one. Meanwhile the
penalty *increased* the movement's size: candidate
norm 0.2969693161862222, the largest second update on
record, against the control's 0.2515524481203623 —
opposition was bought partly with magnitude (SI-27's
as-built says so in those words). Two consequences
for design, both binding on §6: (i) an objective that
rewards worst-case opposition without watching size
can repeat SI-27's trade in a stronger form — the
worst-input term must be published beside the
candidate norm and ‖m2‖, and a treatment whose maximum
improves only because its movement *shrank* must be
distinguishable from one whose movement *turned*;
(ii) §10's rejection stands guard on the other side —
deliberately shrinking the movement by norm budget,
step count, or correction scale is gate-fitting, not
a composition result, and is rejected there so that
(i) cannot be misread as licensing it.

### 5.4 The binding probe is stable across the last two runs — and designing around that is forbidden

SI-23's binding probe was `tiny-id-3`; SI-25's and
SI-27's is `tiny-id-0` — in both cases the probe where
cycle 1 itself moved most (cycle-1 drift at
`tiny-id-0` is 0.17189218290150166, the largest of the
eight). SI-26 §6.5 named "whether the binding probe is
stable" an open question and warned that designing a
slice around the assumption that it is a regularity
would be designing around the judge. Two consecutive
confirmations do not retire that warning; they sharpen
the temptation it guards against. This RFC therefore
records the regularity as a *measured observation
about past runs* and forbids its use as a design
input: no candidate below names `tiny-id-0`, weights
it, or shapes anything by its identity. The legal
formulation — and the only one §6 uses — is
population-level: *the worst input in whatever
population the objective may legally read.* On allowed
inputs that population is the train partition; on
judged inputs it is the probe suite, which training
never reads. If the worst-allowed-input and
worst-probe behaviours are the same phenomenon, a
worst-case objective has a chance to bind the maximum;
if they are not, F1 repeats at the worst case instead
of the mean, and §6's falsification readings are
written to say exactly that (§6.6).

### 5.5 What is still not known — the measurements that must precede any fix

Stated as open questions a slice could retire — this
RFC invents no numbers for them (§18 carries them
forward):

- **The per-input distribution of composition on
  allowed inputs.** SI-27 asserted per-input rows in
  tests (32 train / 16 held-out per candidate) but
  published only aggregates. The *maximum* per-input
  inner product on each allowed partition — for the
  SI-25 control and for the SI-27 treatment — has
  never been published. This is the load-bearing
  unknown for §6: if the treatment's worst train input
  *also* improved while its mean improved, the
  worst-case objective is aiming at a phenomenon the
  mean already reaches, and its premise weakens before
  it is built; if the worst train input worsened (or
  barely moved) while the mean deepened by a full
  point, then F2's shape — average purchased, worst
  case untouched — exists *inside* the allowed
  population, the minimax objective has a measured
  target, and the train-side analogue of the gate's
  statistic can be priced before any new training is
  trusted. **This measurement must be published before
  §6's objective is finalised, and §14 makes it Phase 0
  of the recommended slice, computed for both existing
  candidates from existing machinery.**
- **F3's threshold.** RFC SI-26 §7.6 defined F3
  qualitatively — the destination "no longer
  approached" — and SI-27 measured a record that fits
  neither the definition's firing condition (16/16
  closer, loss falling ~5.3×: approached) nor its
  clean negation (residual ~50× the control's:
  approached *far less completely*). The reading came
  out "not measured as defined" because the definition
  carries no pre-registered boundary between
  *approached* and *traded away*. Until a successor
  RFC states that boundary in advance — in terms of
  the published held-out record, before the decisive
  run it will judge — every future composition
  candidate's destination cost will land in the same
  unpriced grey zone. §6.6 and §15 therefore require
  SI-29 to pre-register its F3 reading rule as part of
  the slice's design, stated in the as-built *before*
  the decisive numbers, and §18 records the general
  lesson as an open methodological question for the
  cadence, not only for SI-29.
- **Whether worst-case composition transfers.** The
  composition-generalisation gap of SI-26 §6.5, one
  level sharper: F1 measured the gap for the mean. For
  the worst case there is no measurement at all — the
  worst judged input is, by invariant 9, the one input
  class training can never inspect, and the worst
  allowed input is a different input drawn from a
  different (disjoint) population. Whether worst-case
  behaviour on 32 constructed training sequences says
  anything about worst-case behaviour on 8 constructed
  probes is **an open question to measure**, and §6 is
  written so that its decisive run answers it in
  either direction.

---

## 6. Candidate (c): a worst-input composition objective on allowed inputs

**Verdict up front: RECOMMENDED — as SI-29, phased
measurement-first (§14).** The only candidate aimed at
the statistic the gate actually prices — a worst case —
computed over the only population training may legally
read, with its premise stated in falsifiable form
(§6.6), its decisive configuration changing exactly one
variable against SI-27's treatment (which becomes its
re-run control), and no ADMIT required.

### 6.1 Why the prior rejections and closures do not bind this candidate

- **It is not §4's closed candidate.** The mean penalty's
  premise — aggregate composition is the binding
  variable — is falsified by F2. This candidate's premise
  is the negation F2 leaves standing: that the binding
  variable is *worst-case* composition, and that the mean
  objective failed *because* it priced the wrong
  statistic of the right quantity. The two objectives
  share machinery (movements `m1`, `m2`, the product
  `p(x)`) and differ in the reduction over inputs — mean
  vs worst case. That is exactly one variable, changed
  against a published control, in the cadence's
  established discipline (SI-26 §10's sequencing rule).
- **It is not SI-26 §9's rejected orthogonalization.**
  That rejection named methods that read *probe* inputs
  or probe movements, and a weight-space projection
  refuted on SI-25's table. This candidate reads no
  probe anything: its population is signal 2's train
  partition, the same 32 sequences SI-27's penalty read,
  with the same probe-blindness assertions required
  (§15 criterion 4). The projection *form* on allowed
  inputs is a separate candidate (§7), deferred on
  sequencing.
- **It is not §8's rejected per-probe anchoring.** The
  gate is untouched (§8.1); no probe is anchored,
  weighted, or named (§8.2). "Worst" in this candidate
  ranges over *allowed inputs during training*, never
  over probes; the probe-side worst case appears only
  where it always has — in post-verdict diagnostics.
- **It does not fit the headroom by construction.**
  Nothing in the objective references tol, headroom, or
  any probe quantity; its weight is fixed in advance by
  a parity argument (§6.2), not searched against the
  gate (§16 forbids the search).

### 6.2 What a worst-input objective means, precisely

Under the new composition configuration (a third named
value, §6.4), `train_candidate` trains signal 2's
factors against

  signal MSE + λ_w · W(m2),

where `m1(x)`, `m2(x)`, and `p(x)` are SI-27's
definitions unchanged (`m1`: landed head minus base
last-position logits through the existing forwards,
constant during optimisation; `m2`: candidate movement;
`p(x)`: per-coordinate mean product), and `W` is a
**worst-case reduction of `p(x)` over the signal's 32
training sequences**, in exactly one of two forms,
which §14 requires the slice to choose between *on
Phase 0's published evidence and derivation, before
any decisive training* — this RFC specifies both forms'
obligations and recommends the first:

- **Top-k mean (recommended).** `W` = mean of the k
  largest per-input `p(x)` values, with k fixed in the
  configuration's definition (k = 4 of 32 is this RFC's
  argued value: large enough that the gradient is a
  combination of several inputs' cached hidden states
  rather than a single input's, small enough that the
  term prices the tail, not the bulk; the choice and
  its argument must be restated in the as-built, and k
  is not a per-cycle dial). The top-k mean is piecewise
  smooth: on any region where the top-k set is fixed,
  it is a mean of fixed terms, so SI-27's gradient
  machinery — the penalty's fixed-matrix gradient
  through the same readout fold — applies term by term.
- **Smooth maximum (log-sum-exp).** `W` = τ · log Σ_x
  exp(p(x)/τ) with τ fixed in the configuration's
  definition. Smooth everywhere, and its gradient is a
  softmax-weighted combination of the same per-input
  gradients; but it introduces a temperature whose
  scale interacts with `p`'s magnitude in ways Phase 0
  must price, and its "worst case" is a soft claim the
  labelling rules (§17) would have to carry carefully.

λ_w is fixed by the same parity argument SI-26 §7.2
used for λ = 1.0 — the term enters with weight 1.0 in
the units it is measured in — and is part of the
configuration's definition, recorded in the
hyperparams, never read from an intent or caller
input, and never swept (§16). One further obligation
the mean objective did not carry, from §5.3: the
slice must publish, beside the worst-input term's
values, the candidate's norm and ‖m2‖ for control and
treatment, so that a maximum improved by shrinkage is
distinguishable from a maximum improved by turning.
The non-improvement refusal still judges the *signal*
held-out loss alone, exactly as under both existing
configurations.

### 6.3 Gradient obligations under the worst-input loss

SI-27's derivation bought the penalised gradient
cheaply because the penalty is linear in the folded
update. The worst-input term is *not* linear in that
sense, and this RFC prices the difference honestly:

- Under the top-k form, the gradient is the mean of
  the selected inputs' per-input penalty gradients —
  each of which *is* SI-27's fixed-matrix gradient —
  but the selected set is itself a function of the
  factors: at points where the top-k membership
  changes, the objective is continuous and its
  gradient has a jump. The training dynamics therefore
  carry a risk the mean objective did not (selection
  oscillation across the 2,000 steps), which is **an
  open question to measure**, answered by the slice's
  published term values at initialisation and at the
  end of training and by determinism (§15 criterion 6)
  — not argued away here.
- The finite-difference gradient check must be re-run
  for the new loss at the decisive head `d0001` and at
  base, over every one of the 512 factor entries,
  tolerance 1e-5 unchanged (§15 criterion 3). At base
  the coincidence configuration must make the new term
  identically zero exactly as SI-27's did (`m1` is the
  zero vector at base), so the at-base check prices the
  signal path under the new code and the at-head check
  prices the worst-input machinery itself. A check
  that passes only at points where the top-k set is
  locally stable must say so; the pinned initial
  factors are the project's established check point
  and membership stability there is itself reportable.
- No new backward machinery beyond the selection and
  weighting of existing per-input gradients is
  anticipated; if the derivation requires more, the
  slice's as-built must show it, and the check — not
  the derivation's elegance — is the evidence.

### 6.4 The representation: a third value on the composition axis

Composition remains **one recorded configuration axis**
beside signal and anchoring. `COMPOSITION_CONFIGS`
gains exactly one value — `"worst-input-composition"` —
validated by the existing discipline; any other value
refuses. The mind chooses neither its signal, its
anchoring, nor its composition configuration (SI-26
§17 item 13, carried); the driver configures all three
and the journal shows all three. λ_w (and k, under the
top-k form) live in exactly one place, in the new
value's definition. Under `"none"` and under
`"movement-composition"` the training paths remain
byte-for-byte SI-25's and SI-27's respectively —
proven, not asserted, by exact reproduction of both
controls inside the slice (§15 criterion 9). No
autonomous, learned, depth-driven, verdict-driven, or
worst-probe-driven selection exists: in particular,
nothing anywhere in the slice reads a probe result to
choose the configuration, the weight, or k.

### 6.5 The decisive configuration: signal 2, base-anchored, worst-input-penalised, at `d0001`

The decisive run is SI-27's decisive run with **exactly
one configuration changed**: cycle 2's composition
configuration, from `"movement-composition"` to
`"worst-input-composition"`. Signal (signal 2),
anchoring (base-anchored), head (`d0001`), target
source (base), step count, learning rate, norm budget,
and candidate construction are SI-27's, unchanged.
Cycle 1 is SI-27's cycle 1 (default signal, anchoring,
and composition configuration) and must reproduce
`d0001` exactly. The treatment's control is SI-27's
treatment itself, **re-run inside the slice** under
`"movement-composition"` and required to reproduce
SI-27's decisive candidate exactly (blob
`sha256:1bb277ab…b637`, drift 0.32509003952145576), so
the new objective's difference is attributable to the
reduction over inputs and not to the refactor that
introduced it — the same attribution discipline SI-27
applied to SI-25's control. Exactly one worst-input
decisive run is performed; no sweep of λ_w, k, or τ
of any kind is run (§16).

### 6.6 What would falsify the design's premise

A recommendation that cannot be wrong is not a design;
it is a hope. This slice's premise is falsifiable in
three named ways, mirroring RFC SI-26 §7.6's forms with
the worst-case statistic substituted for the mean, and
§15's criteria require the measurements that would
falsify it to be published whatever they show:

- **F1′ — the worst-case penalty does not transfer.**
  The treatment's worst-input composition on allowed
  inputs improves (the published Phase-0 statistic
  falls for the treatment against both controls) while
  its probe composition at the worst probe is unchanged
  or worse than SI-27's treatment's (+0.03801222744105431
  at `tiny-id-0`) — the train-to-probe gap swallows the
  worst-case term as it swallowed the destination
  argument in SI-25 and the mean term in SI-27.
  Falsified premise: that worst-case composition
  measured on allowed inputs shapes worst-case
  composition on judged inputs, through this readout,
  at this scale. (Two prior falsifications of the
  transfer premise at weaker statistics make this the
  *expected* outcome to design the publication around,
  not a remote one; §18 item 1 says so plainly.)
- **F2′ — the worst case transfers on allowed inputs
  and the judged maximum still does not move.** The
  treatment's worst allowed input opposes decisively,
  its aggregate probe composition is no worse than
  SI-27's treatment's, yet the composed maximum stays
  above tol 0.25 — worst-case opposition purchased on
  the population training may read, alignment surviving
  on the population it may not. Falsified premise:
  that the worst case is a population-independent
  property of the movement pair at all — from which it
  would follow that no training-side statistic of
  `p(x)` on allowed inputs prices the judged maximum,
  and the composition line's remaining honest moves
  are §7's projection form (which changes the
  candidate, not the statistic) or a later RFC's new
  argument.
- **F3′ — the destination is traded away.** The
  treatment composes better at the worst case but its
  signal held-out record shows the destination no
  longer approached — under the **pre-registered
  reading rule** §5.5 requires: before the decisive
  run, the slice states the boundary in the held-out
  record's own terms (the record must show the loss
  falling and a stated fraction of held-out examples
  closer, at a stated completeness relative to the
  re-run SI-27 control) that separates *approached*
  from *traded away*, and the as-built applies that
  rule as written, in either direction. Such a
  candidate, even if the gate ADMITted it, would not
  be evidence that worst-input *authorship* works; it
  would be evidence that the gate can be approached by
  a candidate that does less of its task, and the
  as-built must say so in those words.

### 6.7 Measurement first: Phase 0 retires §5.5's unknowns before the objective is trusted

The slice is phased, and the phase order is part of
the recommendation, not a scheduling nicety:

- **Phase 0 (measurement, no new training semantics).**
  Using existing machinery on existing candidates,
  publish (i) the per-input composition distribution
  on both allowed partitions for the SI-25 control and
  the SI-27 treatment — per-input inner products,
  their maximum per partition per candidate, beside
  the already-published aggregates; and (ii) the
  slice's pre-registered F3 reading rule (§6.6). If
  (i) shows the SI-27 treatment's worst allowed input
  already opposes as strongly as its mean, §6.2's
  choice between forms — and arguably the slice's
  premise — must be re-argued in the as-built from
  that measurement before Phase 1's decisive run is
  trusted; the RFC's recommendation survives that
  outcome only as a measurement of the worst-case
  statistic itself, and the as-built must say which of
  the two slices it turned out to be.
- **Phase 1 (the build).** The third composition
  configuration, its gradient check, the coincidence
  configuration, both re-run controls, the decisive
  run, the full diagnostic set, and the falsification
  readings — as specified in §14 and §15.

---

## 7. Candidate (d): the projection form — train-partition orthogonalization

**Verdict up front: DEFERRED — legitimate, argued, and
sequenced behind §6.** This is the candidate SI-26 §9
explicitly left open: orthogonalization *on the train
partition* — penalise or project the component of `m2`
parallel to `m1` on allowed inputs — was "not
rejected" there; the RFC recommended the penalty form
over the projection form for one stated reason, and
added: "If SI-27's F1/F2 outcomes falsify the penalty's
premise, the projection form is a legitimate subject
for a later RFC to argue from SI-27's measurements."
F1 and F2 are both measured. This RFC therefore takes
the candidate up in earnest — and defers it, on
sequencing grounds it states rather than on the
principle SI-26 already cleared:

- **What it is.** Train signal 2's candidate as SI-27
  did (or unpenalised), then project the folded Δ2 to
  remove the component parallel to Δ1 *as measured on
  the train partition* — or, in the harder variant,
  constrain the optimisation itself to the orthogonal
  complement on allowed inputs. No probe contact in
  either variant; invariant 9 is satisfied exactly as
  §6 satisfies it.
- **Why not now — the objection SI-26 stated still
  stands, and SI-27 strengthened it.** A projection
  applied after training produces a candidate whose
  factors were never optimised as a whole: its signal
  loss, norm, and destination arrival are whatever the
  projection leaves behind. SI-27's F3 record — a
  candidate that composes better and authors *measurably
  less* even when its every property was optimised and
  published — shows how much of the story lives in
  exactly the quantities a post-hoc projection leaves
  unmeasured during training. §6's objective keeps a
  single optimisation whose every published record
  describes the candidate actually judged; the
  projection does not, and repairing that (re-measuring
  everything post-projection, re-checking the
  non-improvement refusal against a candidate training
  never produced) is design work this RFC scopes but
  does not finish.
- **Why it stays on the record.** If §6's F2′ fires —
  no training-side statistic of `p(x)` prices the
  judged maximum — the projection form is the
  composition line's remaining move that changes the
  *candidate* rather than the statistic, and this
  section is its argued starting point: a later RFC
  takes it up from SI-29's measurements, with the
  post-projection accounting obligations above as its
  first design problem. Deferred, not rejected; and
  it inherits invariant 9 in full either way.

---

## 8. Candidate (e): per-probe anchoring — three readings, three verdicts

This candidate names three different proposals that
share a phrase, and SI-27's evidence treats them
differently. They are separated here so none can borrow
another's verdict — the same separation SI-26 §8
performed for per-probe *accounting*, extended to
per-probe *anchoring*.

### 8.1 As a gate or accounting change: REJECTED (carried, not reopened)

Any reading under which the gate accounts identity per
probe — budgets per probe, headroom carried per probe
from cycle 1, a per-probe anchor the candidate must
match — is SI-26 §8.2's rejected candidate in new
vocabulary, and its rejection is carried forward in
full: it reprices the judge after (now) four measured
failures while leaving 0.25 printed in the policy;
the maximum is not an accounting accident; and SI-27
gives the change no evidence to stand on — the gate's
journaled checks again equal an independent evaluation
(SI-27 criterion 7), and the diagnostics *confirm* the
verdict's figure at the binding probe. The failure is
in the candidates, four times measured. Not reopened.

### 8.2 As training against per-probe targets: REJECTED (invariant 9)

The reading under which cycle 2's targets or penalty
are anchored *per probe* — each identity probe's
desired movement computed and trained toward — reads
probe inputs and probe-derived movements to shape the
candidate. That is training on the judge (invariant 9),
SI-26 §9's rejection, unchanged by SI-27: an ADMIT so
produced would demonstrate that per-probe targets can
satisfy 8 known inputs, not that two authored deltas
compose, and would be uninterpretable — worse than a
REJECT, because a REJECT at least measures something.
Rejected outright, and §16 lists it among the
non-goals with no exceptions and no "diagnostic-only"
carve-out for the training path.

### 8.3 As per-input accounting on allowed inputs: ADOPTED, inside §6 — not a slice of its own

The one invariant-clean residue: accounting composition
*per input* on the populations training may see —
which §5.5 names as the project's outstanding
measurement and §6 builds its objective and its Phase 0
on. Adopted in exactly that form, inside SI-29, riding
with the slice whose design it informs (the discipline
of SI-26 §8.1, applied again). A standalone
per-input-accounting slice — the tables without the
objective — is **not recommended**: Phase 0 publishes
the tables *because* the objective's form depends on
them, and publishing them without the slice would spend
a cycle measuring a system whose next measurement
SI-29 will change anyway.

---

## 9. Candidate (f): probe-partitioned sequencing

**Verdict up front: REJECTED.** The proposal, stated
fairly: if the composed maximum is the obstacle,
sequence the writes so that successive cycles move
*disjoint parts* of the judged behaviour — partition
the probes (or the behaviour they sample) and assign
each cycle a partition, so no single probe accumulates
two aligned movements. It fails on both of its possible
readings:

- **Partitioned by probe behaviour.** Deciding which
  cycle may move which probe requires reading probe
  movements per candidate signal — shaping the
  sequence by the judge's inputs and their measured
  responses (invariant 9), and doing so at the level
  of the *driver's* choices, which §17 item 13's rule
  (recorded selection is not a choice the mind made)
  does not launder: a driver-level selection rule
  derived from probe behaviour is probe-derived
  shaping with an extra step. Rejected on the
  invariant.
- **Partitioned by allowed-input behaviour.**
  Partition the *signals* instead — author or select
  second signals whose allowed-input movement profiles
  are disjoint from the first's, in sequence. This is
  invariant-clean and is SI-26 §10's rejected candidate
  in sequence form: a search over signal variants for
  ones whose movements happen to fit, argued backward
  from the obstacle the gate measured. The family's
  composition behaviour is the measured obstacle
  (§5.2); partitioning its members by where they move
  does not change what each member's movement *is* at
  the judged inputs, and the search's selection
  criterion — composed behaviour — is observable only
  through the judge. The gate-fitting trap of SI-24
  §5.1 / SI-26 §13, in the same costume, worn by a
  sequence instead of a signal. Rejected on evidence
  and on the trap, either way.

A weaker relative survives and is noted so it is not
mistaken for this candidate: *nothing* in this section
speaks against running more than two cycles someday,
or against signals authored for reasons independent of
the gate (SI-23's signal 2 was one). What is rejected
is partitioning — of probes or of signals — *as the
mechanism* for making the maximum pass.

---

## 10. Candidate (g): magnitude discipline — a smaller second movement

**Verdict up front: REJECTED as a slice; retained as a
publication obligation inside §6.** §5.3 establishes
that size is half the failure: SI-27's second movement
exceeds cycle 1's remaining headroom at 7 of 8 probes,
and even a perfectly orthogonal movement of that size
is not argued to fit. The tempting candidate — train
cycle 2 under a smaller norm budget, fewer steps, or
a reduced correction scale, so `m2` shrinks toward the
headroom — is disposed of by the project's standing
rules, and SI-27's evidence does not reopen them:
SI-26 §16 already names "no reduced correction scale,
no reduced step count or norm budget chosen to shrink
`m2`" a binding non-goal, and SI-24 rejected the
reduced-scale route before that. The reason is not
tidiness: a shrunken candidate that ADMITs has shown
that a *smaller* second write fits, which the project
has known since cycle 1's drift left headroom — it
prices the destination down until the judge stops
objecting, and its held-out record would show the
trade (F3's shape) by construction. What SI-27 *does*
license is narrower, and §6.2 takes it: magnitude is
now a **published, compared quantity** in every
composition run — candidate norm and ‖m2‖ beside every
drift figure — so that improvements by turning and
improvements by shrinking are never blended (§17).
Discipline as accounting: adopted. Discipline as a
training dial aimed at the headroom: rejected.

---

## 11. Candidates (h) and (i): a different λ; new target / additive

### 11.1 Candidate (h): a different λ for the mean penalty

**Verdict up front: DEFERRED as an open question — not
the next slice.** SI-27 ran exactly one λ (1.0, the
configuration's pre-registered definition) and its
as-built states, in its own "still NOT claimed" list,
that whether a different single λ would behave
differently is an open question it does not measure.
This RFC keeps it open, deliberately, for two reasons:

- **The mean premise is settled at the statistic, not
  the weight.** F2 falsified "aggregate composition is
  the binding variable" — a claim about what the term
  *averages*, not about how hard it pushes. SI-27's
  term values show λ = 1.0 was not negligibly scaled
  (the penalty dominated the total loss, which ends
  negative; the signal residual rose ~55×), so the
  "λ was simply too small to matter" reading is
  contradicted by the published record, and the "λ was
  too large" reading is F3's partial trade — an
  argument about destination cost, not about the
  maximum, which worsened at the binding probe at this
  λ already. No λ argument from SI-27's record reaches
  the binding figure; re-running the mean at a new λ
  re-runs a settled question at a new weight.
- **Sequencing.** A future λ is a future RFC's argued,
  single, pre-registered value (SI-26 §16) — argued
  from evidence that does not yet exist. If SI-29's
  worst-input readings suggest the *weight* axis is
  where the remaining signal lives (for either
  objective), that RFC can argue it from two objectives'
  published term values instead of one. Deferred to
  §18, not recommended now, and no sweep is licensed
  by the deferral.

### 11.2 Candidate (i): a new LoRA target; additive / multi-target authorship

**Verdicts carried, unchanged.** A new target
(non-readout tensor) stays **DEFERRED** (SI-26 §11.1):
it needs a full backward derivation plus a per-target
finite-difference gradient check, and SI-27 sharpens
rather than softens the obstacle — the composition
diagnostics themselves are readout-geometry objects
(`m1`, `m2` are last-position logit movements), so a
candidate whose update changes hidden states is not
even *described* by the instruments §5 reasons with.
Additive and multi-target authored candidates stay
**REJECTED as a next slice** (SI-26 §11.2): they
require loosening `max_targets_per_delta` /
`allow_additive` policy defaults with zero
authored-behaviour evidence, and four REJECTs under
the defaults are not evidence *for* loosening them.

---

## 12. Candidate (j): transformer squash; candidate (k): Glimmer probing

### 12.1 Candidate (j): transformer squash for miniature deltas

**Verdict up front: DEFERRED**, on the same ground as
SI-26 §12.1, now firmer: the precondition — an authored
two-delta stack — remains unmet after *four* measured
attempts (SI-21, SI-23, SI-25, SI-27), and SI-12's
fail-closed refusal therefore still has nothing
authored to squash. SI-27 adds one design fact a future
squash RFC inherits: a landed stack's deltas may now
carry any combination of recorded signal, anchoring,
*and* composition configuration, so the squash gate
context — still demo-shaped — must be designed against
three recorded axes, not two. Recorded here so the
future RFC meets it as a known obligation. Deferred;
§14 builds nothing toward it.

### 12.2 Candidate (k): probing Glimmer — REJECTED, not reopened

**Verdict up front: REJECTED.** Rejected at SI-13
§7/§9, restated at SI-18 §5, SI-20 §7, SI-22 §7.2,
SI-24 §7.2, and SI-26 §12.2; SI-27 changes nothing: the
composition machinery is a miniature mechanism and
claims nothing at Glimmer scale, the Glimmer seed is
untouched, and its deltas still escalate with checks
3/4 "not evaluated". Success *or* continued failure on
the miniature does not soften the answer; §18 item 9
records the pressure honestly so it is met as a
standing question, not a surprise. Not reopened.

---

## 13. Decision summary

| Candidate | Verdict | One-line reason |
|---|---|---|
| (a) Small real safetensors base | **DEFERRED** | SI-18 §3.7 audit: A-1 to A-4 all still unmet, unchanged by SI-27; naming a real base needs a download this scope forbids and an operator decision this RFC does not take (§3). |
| (b) Movement-composition-aware training (mean penalty) | **DONE — as SI-27, closed** | Built and measured (PR #32): REJECT at 0.32509003952145576 with F1 and F2 both measured; the *mean* premise is settled, the family is not (§4). |
| (c) Worst-input composition objective | **RECOMMENDED — SI-29** | The only candidate aimed at the statistic the gate prices — a worst case — over the only population training may read, phased measurement-first, exactly one variable changed against SI-27's treatment as its re-run control (§6). |
| (d) Projection form (train-partition orthogonalization) | **DEFERRED** | Legitimate since SI-26 §9 and expressly arguable now that F1/F2 are measured; sequenced behind (c) because a projected candidate's every property is unmeasured during training, and SI-27's F3 record prices exactly that gap (§7). |
| (e) Per-probe anchoring | **SPLIT: gate-side REJECTED; probe-side training REJECTED; per-input accounting on allowed inputs ADOPTED inside SI-29** | The first two readings reprice or read the judge (§§8.1–8.2); the third is §5.5's outstanding measurement and §6's Phase 0 (§8.3). |
| (f) Probe-partitioned sequencing | **REJECTED** | Probe-derived partitioning reads the judge (invariant 9); signal-derived partitioning is SI-26 §10's gate-fitting search worn by a sequence (§9). |
| (g) Magnitude discipline | **REJECTED as a slice; ADOPTED as accounting** | Shrinking `m2` by budget, steps, or scale is the standing gate-fitting non-goal (SI-26 §16); publishing norm and ‖m2‖ beside every drift figure is now mandatory (§§5.3, 10). |
| (h) A different λ for the mean penalty | **DEFERRED (open question)** | F2 settled the mean at the statistic, not the weight; SI-27's term values contradict the under-scaled reading; a future λ is a future RFC's single argued value (§11.1). |
| (i) New LoRA target / additive / multi-target | **DEFERRED / REJECTED as next slice (carried)** | New target still lacks its backward derivation and is not described by the readout-geometry diagnostics; additive still buys its premise with a policy loosening (§11.2). |
| (j) Transformer squash (miniature) | **DEFERRED** | Precondition still unmet — no authored two-delta stack exists after four measured attempts — and the squash gate context must now be designed against three recorded axes (§12.1). |
| (k) Glimmer probing | **REJECTED** | Rejected at SI-13 §7/§9 and restated in every successor RFC; SI-27 changes nothing (§12.2). Not reopened. |

The four routes to a landed second write that the
project has now run — a new signal under settled
semantics (SI-23), the same signal restacked (SI-21),
new target semantics (SI-25), and a composition term
inside the loss (SI-27) — have all been refused by the
same unchanged accounting, and the last two refusals
are attributed to the same probe-level mechanism with
increasing precision: movements that oppose in
aggregate add at the binding probe, and a mean-priced
penalty cannot reach a maximum-priced failure. §6's
route is the fifth, and the first to price its
training objective in the gate's own statistic — a
worst case — on the only population it may lawfully
compute that statistic over. No sixth route is
proposed by this RFC, and §§8–10 record the three
tempting routes (a per-probe judge, a partitioned
sequence, a shrunken movement) and why they are not
candidates.

---

## 14. The recommended slice: SI-29 (outline)

**SI-29 — Worst-input composition-aware training for
the miniature: signal 2, base-anchored, trained at the
landed head with a worst-input composition term on its
own train partition — phased, with the outstanding
composition measurements published before the new
objective is finalised.**

In scope:

1. **Phase 0 — the outstanding measurements (§§5.5,
   6.7).** From existing machinery and existing
   candidates, before any new training semantics are
   finalised: (i) the per-input composition record on
   both allowed partitions for the SI-25 control and
   the SI-27 treatment — per-input inner products, the
   maximum per partition per candidate, sign agreements
   (never alone, §17 item 19), movement norms — beside
   the published aggregates; (ii) the slice's
   pre-registered F3 reading rule (§6.6), stated in the
   as-built before the decisive numbers. The §6.2 form
   choice (top-k mean recommended; log-sum-exp the
   alternative) is made on (i)'s evidence and
   derivation, and the as-built states the choice and
   its argument. If (i) undercuts the slice's premise,
   the as-built says so and says which slice it turned
   out to be (§6.7).
2. **Worst-input composition as a recorded
   configuration** (§6.4): `COMPOSITION_CONFIGS` gains
   exactly `"worst-input-composition"` beside `"none"`
   and `"movement-composition"`; constructor-level on
   `TinyTransformerMind`, settable between cycles by an
   explicit driver call in the shape of
   `set_composition_config(...)`; validated with the
   existing refusal discipline; the intent carries it;
   `train_candidate` dispatches on it. λ_w = 1.0 (and
   k = 4 under the top-k form; τ under the smooth form)
   are part of the new value's definition, recorded in
   the hyperparams, not separately settable. No
   autonomous, learned, depth-driven, verdict-driven,
   or probe-driven configuration selection exists, and
   no sweep of λ_w, k, or τ of any kind is run.
3. **The worst-input loss** (§6.2): under
   `"worst-input-composition"`, `train_candidate` adds
   exactly one term to SI-25's base-anchored signal
   loss — λ_w · W(m2), the worst-case reduction over
   signal 2's 32 training sequences of SI-27's `p(x)`,
   with `m1` prepared exactly as SI-27 prepares it
   (train partition only, constants during
   optimisation), and hidden states, target source,
   candidate construction, step count, learning rate,
   and the norm guard exactly as in SI-25/SI-27. Under
   `"none"` the path is byte-for-byte SI-25's; under
   `"movement-composition"` byte-for-byte SI-27's. No
   probe input, probe logit, held-out sequence, or
   probe-derived quantity enters the loss under any
   configuration (asserted by test, §15 criterion 4).
   The term's values at initialisation and at the end
   of training are recorded on both partitions (train:
   optimised; held-out: reported only), and the worst
   train input contributing to the term is recorded
   *by its train-partition identity* — an allowed
   input, named as one (§17).
4. **The coincidence configuration** (§6.3): configured
   with signal 2, base-anchoring, and the worst-input
   configuration, a first cycle at base records the new
   term's value at initialisation and at the end of
   training on both partitions (identically zero,
   measured — `m1` is the zero vector at base) and must
   reproduce SI-25's/SI-27's at-base run exactly —
   candidate blob `sha256:131cc242…45f4`, same gate
   numbers — proving the worst-input machinery correct
   where it must vanish, before the decisive cycle is
   trusted. Its outcomes are labelled at-base facts;
   they are not the slice's decisive claim.
5. **Both controls, re-run inside the slice** (§6.5):
   the SI-25 control (composition `"none"`) must
   reproduce SI-25's decisive cycle-2 candidate exactly
   (blob `sha256:646d1e8b…d675`, drift
   0.3329036422073841), and the SI-27 treatment
   (composition `"movement-composition"`, λ = 1.0) must
   reproduce SI-27's decisive cycle-2 candidate exactly
   (blob `sha256:1bb277ab…b637`, drift
   0.32509003952145576) — so the new objective's
   difference is attributable to the reduction over
   inputs and not to the refactor that introduced it.
6. **The gradient check for the worst-input loss**
   (§6.3): central finite differences of signal 2's
   base-anchored, worst-input-penalised loss through
   the float-side loss path, at the pinned initial
   factors, over EVERY one of the 512 factor entries,
   at base AND at head `d0001`, tolerance 1e-5
   unchanged; top-k membership stability at the check
   points is reported with the figures.
7. **The decisive two-cycle run** (§6.5): one reused
   `TinyTransformerMind` through `si.run.run_cycles` —
   cycle 1 exactly as SI-27's cycle 1, then the driver
   explicitly reconfigures the same mind to signal 2,
   base-anchored targets, and the worst-input
   composition configuration (three recorded selections)
   and runs cycle 2. Exactly two authored cycles are
   attempted and exactly one worst-input decisive run
   is performed. Cycle 2's verdict is whatever the
   unchanged gate measures; **no ADMIT is required**
   (§15 criterion 7).
8. **Diagnostics, the four-run table, and the
   falsification readings** (§§6.6, 15 criterion 8):
   the full post-verdict diagnostic set against head
   `d0001` with the decisive candidate overlaid —
   per-probe drift contributions, the alignment
   diagnostic (aggregate and at the worst probe, sign
   agreement never alone), folded updates, composed
   capability — published for the treatment beside
   *three* controls (SI-23, SI-25, SI-27) in the
   four-run table's form; candidate norm and ‖m2‖
   beside every drift figure (§5.3); the allowed-input
   worst-input record for both controls and the
   treatment; the treatment's signal held-out record
   read under the pre-registered F3 rule; and the
   F1′/F2′/F3′ readings stated as measured, whichever
   way they fall. Worst probes are named; worst
   *allowed inputs* are named by train-partition
   identity and never confused with probes (§17).
9. **VERIFY and unwind, honestly conditioned** (§15
   criterion 10): cycle 1 VERIFY as in SI-19/21/23/25/27.
   If cycle 2 ADMITs and VERIFY passes, the authored
   two-delta stack exists for the first time and
   SI-21's fixture-labelled unwind evidence is joined
   by the real thing — claimed exactly as measured, no
   further. If cycle 2 REJECTs, no stack exists,
   nothing is unwound, and none is claimed — the
   SI-21/SI-23/SI-25/SI-27 posture, unchanged.

---

## 15. Acceptance criteria for SI-29

1. **Recorded composition configuration, three named
   values** — the configuration serves exactly
   `"none"`, `"movement-composition"`, and
   `"worst-input-composition"`; every other value
   refuses with the existing `ValueError` discipline at
   the constructor, the setter, the intent, and the
   `train_candidate` dispatch. Recorded selection is
   asserted per cycle on both verdict paths. λ_w (and
   k or τ) are not independently settable (asserted:
   the constructor and the setter take no such
   argument); exactly one worst-input decisive run is
   performed (no sweep).
2. **Phase 0 published before the decisive run is
   trusted** — the per-input composition record for
   both existing candidates on both allowed partitions
   (per-input inner products, maximum per partition,
   beside the published aggregates), the §6.2 form
   choice stated with its argument, and the
   pre-registered F3 reading rule, all present in the
   as-built ahead of the decisive numbers.
3. **Gradient check for the worst-input loss, at the
   decisive head** — re-run at `d0001` AND at base
   under the new code path, over EVERY factor entry,
   tolerance 1e-5 unchanged, both figures published;
   top-k (or τ-softmax) behaviour at the check points
   reported with them.
4. **Probe-blindness of training, asserted** — the
   worst-input term's input set is asserted as a set
   (exactly the train partition, via a spy on the new
   gradient path during a real training run); the
   train partition's disjointness from both probe
   suites is re-asserted under the new code path; a
   perturbed-probe-artifact bundle copy produces a
   byte-identical candidate.
5. **Coincidence at base** — signal 2 base-anchored +
   worst-input-penalised at base reproduces the
   established at-base run exactly (blob
   `sha256:131cc242…45f4`, drift 0.16672966815531254,
   regression 0.2418281203135848), with the new term
   measured at initialisation and at the end of
   training on both partitions (identically zero).
6. **Determinism per cycle** — a full decisive run
   repeated from the seed reproduces both candidates'
   blobs/records, the landed delta record, and the
   journaled gate measurements exactly; hyperparams
   equal except the labelled `wall_time_seconds`. The
   worst-input `train_delta` alone is likewise
   byte-identical across repeats.
7. **Gate honesty per cycle** — each cycle's journaled
   checks 3/4 equal an independent `ProbeContext`
   evaluation of the same candidate against the
   pre-cycle head (abs 1e-12). Cycle 2's verdict is
   reported **as measured**; no ADMIT is required:
   SI-29 succeeds by §6's claim — a worst-input-penalised
   candidate was genuinely authored against the landed
   head under the new configuration and genuinely
   judged, with its premise's falsification readings
   published either way.
8. **Cumulative accounting, diagnostics, the four-run
   table, and the falsification readings published** —
   the side-by-side accounting table (seed-absolute /
   cumulative identity; head-incremental capability),
   the full diagnostic set with candidate norm and
   ‖m2‖ beside every drift figure, the Phase-0 and
   decisive allowed-input worst-input records, the
   treatment's signal held-out record read under the
   pre-registered rule, the four-run table with each
   figure under its own run's labels, and the
   F1′/F2′/F3′ readings — all published, measurements
   and derivations labelled, worst probes named, worst
   allowed inputs named as allowed inputs, nothing in
   the diagnostic set reported "not evaluated" that
   was computable.
9. **First-cycle and both-control reproduction** —
   configured with signal 1 (default signal, default
   anchoring, default composition configuration), a
   first cycle reproduces SI-19/SI-21/SI-23/SI-25/SI-27
   exactly (blob `sha256:2761e8ff…6ca9e`, drift
   0.17189218290150166, regression 0.1657181391492486);
   the re-run `"none"` control reproduces SI-25's
   decisive candidate exactly; the re-run
   `"movement-composition"` control reproduces SI-27's
   decisive candidate exactly — the configuration
   machinery proven behaviour-preserving for both
   existing semantics before the new configuration's
   results are trusted.
10. **VERIFY, unwind, and untouched scope** — cycle-1
    VERIFY as in SI-19; unwind honestly conditioned
    (no authored stack unless cycle 2 lands; none
    unwound or claimed otherwise). Exactly two authored
    cycles attempted in the decisive run and exactly
    one worst-input decisive run performed;
    `max_stack_depth` 8, `identity_tol` 0.25,
    `capability_eps` 0.25, `max_targets_per_delta` 1,
    `allow_additive` false and every other policy
    default unchanged (asserted); both signals' ids,
    streams, corrections, and partitions unchanged
    (asserted); both anchorings' and both prior
    composition configurations' definitions unchanged
    (asserted); `si squash` still fails closed on the
    resulting transformer stack (asserted; SI-12's
    suite green); the full existing suite is green;
    the demo path is unchanged; `si serve` still
    returns 501 for every transformer bundle (existing
    suite); no CLI wiring added.

---

## 16. Non-goals

Explicit, and binding on SI-29 and on any slice this
RFC is cited to justify:

- **No Glimmer anything.** No Glimmer forward path,
  probes, training, or authorship (§12.2). No candidate
  in this RFC touches the Glimmer seed except to
  restate that it is untouched, and §6's objective is a
  miniature mechanism that claims nothing at Glimmer
  scale.
- **No real-base work.** Candidate (a) is deferred with
  audited preconditions (§3); SI-29 builds nothing
  toward it, and no download is performed or scheduled.
- **No signal redefinition, no third signal.** Both
  signals' ids, streams, corrections (scale 0.15
  included), partitions, target, rank, and alpha are
  byte-for-byte as SI-19/SI-23 defined them. The
  worst-input term changes the reduction of the
  composition quantity a known signal's factors are
  optimised against; it does not change what either
  signal *is*, and no third signal is introduced
  (§§9–10 rejected the partition and search forms of
  one).
- **No probe contact in training.** No probe input,
  probe logit, probe movement, or probe-derived
  quantity enters any training loss, penalty, objective,
  projection, initialisation, stopping rule, sequencing
  decision, or candidate selection (§§2, 8, 9). The
  binding probe is never named in training code, and
  no per-probe target, weight, or anchor exists in the
  training path. This non-goal has no exceptions and no
  "diagnostic-only" carve-out for the training path:
  diagnostics read probes after the gate has spoken, as
  they always have; training never does.
- **No gate change of any kind.** The gate's checks,
  verdict mapping, accounting (seed-absolute cumulative
  identity maximum; head-incremental capability),
  tolerances, caps, and every policy default are
  unchanged (§8.1). If the worst-input cycle is
  REJECTed, the answer is a later RFC's better
  argument — never a wider, repriced, per-probe, or
  quieter gate.
- **No weight tuning.** λ_w = 1.0 (and k = 4, or τ, per
  the §6.2 form choice) is the configuration's
  definition for SI-29. No sweep, no second worst-input
  decisive run at a different weight, no reporting of
  an un-run weight's hypothetical outcome as anything
  but an open question. A future weight — for this
  objective or for the mean's λ (§11.1) — is a future
  RFC's argued, single, pre-registered value.
- **No gate-fitting variant of any other kind.** No
  reduced correction scale, no reduced step count or
  norm budget chosen to shrink `m2` (§10), no signal,
  anchoring, configuration, or sequence chosen backward
  from the identity tolerance (§9), and no selection
  rule a gate verdict can influence.
- **No other widening.** No new LoRA target (§11.2),
  no additive or multi-target authored candidates
  (§11.2), and no policy default changes — in
  particular `max_targets_per_delta` stays 1 and
  `allow_additive` stays false.
- **No autonomous selection — of signal, of anchoring,
  or of composition configuration.** The mind chooses
  none of them; all three are explicit, recorded,
  driver-level configurations (§6.4), and no
  perception-driven, depth-driven, alternating,
  verdict-driven, or worst-probe-driven rule is built
  or implied for any axis.
- **No transformer squash.** SI-12's fail-closed refusal
  stands throughout SI-29 (§12.1); §12.1 is a deferral
  analysis, not an implementation brief — even if
  SI-29 creates the authored stack squash has been
  waiting for.
- **No serving, no CLI wiring.** Transformer bundles
  keep returning 501; `si cycle`/`si run` keep their
  demo/mock minds. SI-29 is a library-and-tests slice
  like SI-19, SI-21, SI-23, SI-25, and SI-27.
- **No third cycle, no continual-learning claim.**
  Exactly two authored cycles are attempted in the
  decisive run, on constructed signals, on a constructed
  model, under the recorded configurations §14 scopes.
  Three configuration axes at depth ≤ 2 prove something
  about the loop's machinery and its gate — nothing
  about learning at any other scale or on any real task.
- **No mixed-configuration stack claim.** The decisive
  run's cycle 1 is unpenalised-at-base only in the
  degenerate sense that both composition terms vanish
  there; SI-29 does not train a worst-input cycle at a
  non-base head inside the decisive run as a claimed
  stack member beyond its cycle 2 (the §14 item 5
  controls are separate runs, labelled controls), and
  no claim is made about stacks whose cycles mix
  composition configurations at non-base heads.

---

## 17. Honesty and labelling requirements for any future slice

Items 1–19 restate RFC SI-26 §17 (itself restating
SI-24 §12 with additions) in compressed form, with the
numbering preserved so existing citations resolve;
items 20–21 are new in this RFC. All nineteen-plus-two
bind SI-29 and any slice this RFC is cited to justify.

1. **Training numbers are training numbers.** Losses,
   penalty and objective values are labelled with
   signal, partition, head, target source, anchoring,
   and composition configuration; none is a gate or
   probe result.
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
    quantities (movement columns, sums at a probe) are
    labelled derivations wherever they appear.
15. **Anchoring is part of every label.** A training
    or diagnostic number recorded under one anchoring
    is never presented as a number under the other.
16. **Controls and treatments are never blended.**
    Each figure stands under its own run's labels; a
    three- or four-run table keeps the rows separate.
17. **Composition configuration is part of every
    label.** A number recorded under one composition
    configuration — now including
    `"worst-input-composition"` — is never presented as
    a number under another.
18. **Allowed-input numbers are not probe numbers.**
    Figures from allowed populations and figures from
    judged populations are never interchanged,
    averaged together, or quoted in each other's place;
    any gap between them is stated as the difference
    between two separately labelled records.
19. **Sign agreement never stands alone.** It is
    published only beside the magnitude-weighted figure
    (the inner product) for the same population.
20. **A worst-case figure names its population (new).**
    "Worst" is meaningless without the set it ranges
    over: a worst *allowed input* (train partition,
    held-out partition) and the worst *probe* are
    different quantities over disjoint populations, and
    no text may shorten either to "the worst case"
    without its population attached. A treatment that
    improves the worst allowed input has measured
    exactly that — nothing about the worst probe —
    until the post-verdict diagnostics measure the
    latter separately.
21. **Judgement-call readings are pre-registered
    (new).** Where a falsification reading turns on a
    qualitative boundary in a quantitative record (F3's
    "destination no longer approached" is the standing
    example), the boundary is stated in advance, in the
    record's own terms, before the decisive run it will
    judge — and the reading is then applied as written.
    A reading invented after the numbers are known is
    reported as an observation, never as the
    falsification test.

---

## 18. Risks and open questions

Each item names what would retire it. Numbers this RFC
deliberately does **not** supply are listed as open
questions to measure — supplying them here would be
inventing them.

1. **F1′ is the expected outcome, and the slice must be
   built to publish it.** The transfer premise — a
   statistic of `p(x)` on allowed inputs shapes the
   corresponding statistic on judged inputs — has now
   been falsified at the aggregate (SI-25's destination
   argument) and at the mean (SI-27's F1). The worst
   case is the same premise at a third statistic. This
   RFC recommends the experiment because the worst case
   is the *gate's* statistic and the only one not yet
   measured — not because transfer has become more
   likely. A slice that reports F1′ as a surprise has
   misread its own RFC. *Retired by:* SI-29's F1′
   reading, published either way (§15 criterion 8).
2. **The worst allowed input may not be a stable
   training target.** Under the top-k form, the set
   the term prices can change membership as the factors
   move; over 2,000 full-batch steps at learning rate
   1.0 the objective may oscillate between inputs
   rather than converge on opposing any of them — a
   dynamics question the mean objective never posed.
   **Open question to measure.** *Retired by:* SI-29's
   published term values (initialisation, end of
   training, both partitions), its determinism
   criterion, and its gradient-check report (§15
   criteria 3, 6).
3. **The destination trade may deepen (F3′).**
   Worst-case pressure is, per input, stronger than
   mean pressure on the inputs it selects; SI-27
   already paid ~50× held-out residual for its mean
   term. A worst-input term could buy its opposition
   further at the destination's expense — or, by
   concentrating on few inputs, spare the bulk of the
   signal and pay less. Both stories are currently
   unfounded. *Retired by:* SI-29's held-out record
   read under the pre-registered rule (§§6.6, 15
   criterion 8) — the first F3-class reading in the
   project taken against a boundary stated in advance.
4. **Phase 0 may undercut the recommendation.** If the
   SI-27 treatment's worst allowed input already
   opposes as strongly as its mean, §6's premise
   narrows to "the statistic was already worst-case-
   healthy on allowed inputs and the failure is purely
   transfer" — in which case SI-29's honest product is
   the Phase-0 measurement plus an F1′ it can predict
   but must still measure. This RFC accepts that
   outcome in advance; §6.7 binds the as-built to say
   which slice it turned out to be. *Retired by:*
   Phase 0 itself (§14 item 1).
5. **λ_w = 1.0 is a parity argument, not a measured
   optimum — again.** As with SI-26 §18 item 5: the
   weight is fixed by units precisely so it cannot be
   accused of being fitted, which also means its scale
   relative to the signal term at initialisation is
   **an open question to measure** (SI-29 publishes
   both terms' values). If the worst-input term is
   negligible at parity, the decisive run measures
   something close to the SI-27 control and the finding
   is about scaling; if it dominates, F3′ is the risk.
   Neither licenses a second weight inside SI-29
   (§16). *Retired by:* SI-29's published term values;
   any successor weight belongs to a later RFC arguing
   from two objectives' records (§11.1).
6. **A mislabelled configuration is worse than a wrong
   one.** The failure mode SI-26 §18 item 6 named —
   a candidate trained under one configuration and
   journaled under another — now has three values to
   confuse instead of two, and a subtler variant: a
   worst-input term silently evaluated on the held-out
   partition, or over a top-k set drawn from the wrong
   population, would attribute a worst-case effect to a
   run that never computed one. *Retired by:* §15
   criteria 1 and 4 — per-cycle journal assertions on
   both verdict paths, the coincidence configuration's
   exact reproduction, *both* controls' exact
   reproduction, and the probe-perturbation independence
   test.
7. **The split-movement path could silently alter
   signal behaviour under the existing
   configurations.** Generalising `train_candidate` to
   a third loss shape is the kind of change that passes
   its own new tests while moving an old behaviour in
   its last digits. *Retired by:* §15 criteria 5 and
   9 — exact reproduction of *four* published blobs
   (signal 1's `d0001`, signal 2's at-base `d0001`,
   SI-25's decisive candidate, SI-27's decisive
   candidate) under the refactored path, before the
   decisive run is trusted.
8. **Probe blindness still bounds the judge.** RFC
   SI-13 §10.1 stands, now across signals, anchorings,
   and three composition configurations: 24 inputs,
   28/30 tensors measurable, blind set derived and
   guarded. A landed two-delta stack under any
   configuration is *evidence* of continuity sampled
   the same way each cycle, not a proof of it.
9. **Depth and Glimmer pressure arrive if SI-29
   succeeds.** At `max_stack_depth` 8 with transformer
   squash still deferred (§12.1), a landed second
   authored delta makes the stack depth 2 and the
   compression question real for the first time — while
   the squash gate context remains unbuilt and must
   now be designed against three recorded axes (§12.1).
   And per SI-26 §18 item 10: success on the miniature
   renews the "why not Glimmer" pressure, and the
   answer does not soften because the miniature's
   training has gained a worst-case objective.
10. **Candidate (a)'s unknowns remain unknown.** A-1 to
    A-4 are audited unmet (§3); nothing in SI-29 or
    this RFC scopes them, and the download question is
    an operator decision this RFC explicitly does not
    take. Candidate (h)'s λ question likewise remains
    open by design (§11.1): it is retired only by a
    later RFC arguing a single value from the two
    objectives' published records, never by a sweep.

---

## 19. Summary

After SI-27, SI can originate a miniature delta, judge
it, land it, and originate a second one on top of it
under a second signal, under two target anchorings, and
under a composition-aware loss — and the gate's
cumulative, seed-absolute identity accounting has been
measured refusing all four second writes, the fourth
with its cause attributed more precisely than the
first three: the penalised update *opposes* the first
update in aggregate on the inputs it may see, in
aggregate on the probes, and in weight space, and the
composed maximum still stands at 0.32509003952145576 >
tol 0.25, because at `tiny-id-0` — the probe where
cycle 1 had already moved most — the two movements add
almost exactly (0.17189218290150166 +
0.15392476320266724 ≈ 0.3251), the penalty having
improved the *mean* of a quantity the gate prices at
its *maximum*. Of the candidate frontiers this RFC
evaluated against that evidence: a small real base
stays deferred, its four SI-18 preconditions audited
and all still unmet, with a checkpoint download flagged
as an operator decision rather than a step (§3); the
mean composition penalty is done and closed, its
premise settled by F1 and F2 (§4); per-probe anchoring
is rejected in its gate-side and probe-side readings —
repricing the judge, and reading the judge — and
adopted only as per-input accounting on allowed inputs
inside the recommended slice (§8); probe-partitioned
sequencing is rejected on the invariant and on the
gate-fitting trap (§9); magnitude discipline is
rejected as a training dial and adopted as mandatory
accounting (§10); a different λ for the settled mean
objective is deferred as an open question no current
evidence argues (§11.1); a new target stays deferred
and additive authorship stays rejected (§11.2);
transformer squash stays deferred, its precondition
still unmet after four measured attempts (§12.1);
Glimmer probing stays rejected, not reopened (§12.2);
the projection form — legitimate since SI-26 §9 and
expressly arguable from SI-27's F1/F2 measurements —
is taken up, argued, and deferred behind the
recommended slice on sequencing (§7); and **a
worst-input composition objective — signal 2,
base-anchored, at the landed head, with one worst-case
term of pre-registered weight λ_w = 1.0 computed only
over the signal's own train partition, recorded as a
third value on the composition axis beside signal and
anchoring, phased so the outstanding per-input
measurements and a pre-registered F3 rule are published
before the new objective is trusted, judged by the
unchanged gate on probes the training never reads,
with its premise stated in three falsifiable forms —
is recommended as SI-29** (§§6, 14, 15). Its
plausibility argument is stated with its limits in the
same sections that make it: SI-27 proved a second
update's composition *can* be shaped on allowed inputs
and proved just as plainly that shaping the mean
leaves the maximum to the signal family's geometry.
Whether pricing the worst allowed input — the gate's
own statistic, on the only population training may
read — moves the one figure the gate prices is
measurable with instruments that already exist,
against three published controls, with exactly one
variable changed. That is a plausibility argument for
running the experiment — nothing in this section is a
prediction that it ADMITs, and §15 requires no ADMIT;
§18 item 1 records that the transfer premise has been
falsified at two weaker statistics already, and this
RFC recommends measuring the third with its eyes open.
The deepest landed authored stack remains depth 1; no
two-delta authored stack exists; the base stays
frozen, the bytes are never duplicated, no tolerance
moves, the gate is unchanged, and nothing is claimed
that a test in this repo cannot check.

---

*RFC SI-28 — design only. Implementation begins, if
approved, at SI-29 as outlined in §14. RFC SI-26
changed what the loop accounts for while it trains —
the first movement it already made — and the gate told
the truth about it, per probe: the accounting worked
almost everywhere and failed where it binds. This RFC
proposes changing not what the loop accounts for, nor
where it may look, but the statistic it answers to
while it accounts: the worst input it is allowed to
see, priced the way the judge prices the worst input
it is not — under the same judge, at the same scale,
with the verdict still not ours to pick.*
