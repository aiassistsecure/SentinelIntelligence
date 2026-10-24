# SI-31 — Train-partition-projection composition for the miniature (as-built)

> **Scope:** RFC SI-30 §14 (the SI-31 outline) and §15 (the ten
> acceptance criteria — each addressed below with its
> measurements), plus §7 (the projection design), §16
> non-goals, and §17 honesty/labelling. Composition is now a
> recorded per-cycle configuration with exactly four named
> values, and signal 2 was trained at the landed head
> `d0001`, base-anchored, exactly as SI-25 trained it — then
> the judged candidate was CONSTRUCTED by full removal of
> the component parallel to the landed movement as measured
> on its own train partition (coefficient
> α = −0.8450027779470831, measured from the train-partition
> aggregate, never settable, no sweep, no partial removal).
> Exactly one variable changed against SI-29's decisive run:
> cycle 2's composition configuration, whose treatment is
> re-run inside this slice as a control, as are SI-25's and
> SI-27's. **Cycle-2 ADMIT was explicitly NOT required (RFC
> §15 criterion 7).** The gate's verdict on the decisive
> cycle 2 is reported as measured in the decisive-run section
> below: **REJECT** — on the unchanged gate's safety suite
> (the constructed candidate's rank, see the representation
> finding in Phase 0) and on cumulative identity drift
> (0.4775387533009052 > tol 0.25), the worst composed maximum
> of the six-run record.
> **This document is ordered as RFC §14 item 1 requires:
> Phase 0 — the projection rehearsal on the three existing
> decisive candidates, the pre-registered successor readings
> F1′′/F2′′/F3′′, and the projection-correctness tolerance —
> comes first, before any decisive number.**

## What was built

- **`si/tiny_training.py`** — the SI-31 additions. (a) The
  fourth composition configuration: `COMPOSITION_CONFIGS`
  (exactly `"none"`, `"movement-composition"`,
  `"worst-input-composition"`, `"train-partition-projection"`),
  `TRAIN_PARTITION_PROJECTION`, and
  `validate_composition_config(...)` serving all four with
  the existing `ValueError` discipline. The configuration's
  definition carries no weight, no k, and no fraction: α is
  measured per run and recorded in the hyperparams, never
  read from an intent or any other caller input. (b) The
  projection machinery: `projection_coefficient(...)` (α and
  its numerator/denominator sums over a named allowed
  population; α defined as 0.0 when the denominator
  vanishes) and `projection_factors(...)` (the SVD
  factorisation carrying a folded update at rank 16 /
  alpha 16, scale 1.0 — see the representation finding in
  Phase 0). (c) `train_candidate` dispatches on the intent's
  composition configuration: under
  `"train-partition-projection"` it trains signal 2's
  factors against SI-25's base-anchored signal loss exactly
  (the unpenalised path — no penalty term of either prior
  kind), then constructs the judged candidate
  Δ2′ = Δ2 − α·Δ1 from the trained intermediate (α computed
  from the intermediate's movements on the train partition
  only), re-measures the projected candidate in full
  (signal train and held-out losses, with the
  non-improvement refusal applied to the PROJECTED
  candidate's signal held-out loss; candidate norm beside
  its pre-projection norm; the allowed-input composition
  record on both partitions), and returns the projected
  candidate with α, its sums, the pre-/post-projection
  norms, the projected train loss, the intermediate's
  held-out record (under `projection_pre_*` labels), and
  the measured post-projection aggregate inner products
  recorded in the hyperparams. At α = 0.0 exactly the
  construction is the identity and the candidate keeps its
  trained rank-8 factors byte-for-byte. Under `"none"` the
  path is byte-for-byte SI-25's, under
  `"movement-composition"` byte-for-byte SI-27's, under
  `"worst-input-composition"` byte-for-byte SI-29's —
  proven by exact reproduction of all three controls inside
  this slice. The mean-penalty, worst-input, and projection
  fields are mutually null: the mechanisms are never blended
  (§17 item 17). The pre-projection candidate is a
  construction intermediate — never judged, landed, or
  reported as a run.
- **`si/tiny_transformer.py`** — the recorded composition
  configuration on `TinyTransformerMind` now serves the
  fourth value through the same constructor-level
  `composition_config` (default `"none"`) and the same
  explicit driver call `set_composition_config(...)`; the
  setter takes no α or fraction argument (asserted);
  `propose` carries the configured value on the intent
  beside the configured signal and anchoring.
- **`si/gate.py`** — `Intent.composition_config`'s comment
  now names the four served values. No gate check, verdict
  mapping, tolerance, cap, or policy default changed; the
  field itself and its default (`"none"`) are unchanged
  from SI-27.
- **`tests/test_train_partition_projection.py`** (new,
  13 tests), the two `COMPOSITION_CONFIGS` assertions in
  `tests/test_movement_composition.py` and the two in
  `tests/test_worst_input_composition.py` updated to the
  new four-value tuple (the prior slices' exactness
  assertions, carried forward to the configuration's new
  definition — every other prior assertion is untouched),
  and this doc. No probe-suite change, no serve change
  (501 for every transformer bundle), no CLI wiring, the
  Glimmer seed untouched, the demo path unchanged, both
  signals' ids, streams, corrections, and partitions
  unchanged, both anchorings' definitions unchanged, all
  three prior composition configurations' definitions
  unchanged.

There is no autonomous, learned, depth-driven,
verdict-driven, or probe-driven composition selection
anywhere in the slice — the mind chooses neither its
signal, its anchoring, nor its composition configuration;
the driver configures all three, and the journal shows all
three. Exactly one projection decisive run was performed;
no sweep of any projection variant, fraction, or
coefficient definition was run.

## Phase 0 — the projection rehearsal, the successor readings, and the correctness tolerance (RFC §§5.5, 7.4, 14 item 1)

Computed from existing machinery on the three existing
decisive candidates, re-run inside this slice before any
new candidate construction was finalised — the SI-25
control (signal 2, base-anchored, composition `"none"`, at
`d0001`, blob `sha256:646d1e8b…d675`), the SI-27 treatment
(composition `"movement-composition"`, λ = 1.0, blob
`sha256:1bb277ab…b637`), and the SI-29 treatment
(composition `"worst-input-composition"`, λ_w = 1.0,
k = 4, blob `sha256:11cab02e…b8db5`). All three reproduced
exactly (blobs and gate figures in the reproduction
criterion below), so these records describe the published
candidates, not approximations of them. For each candidate
the rehearsal computes — as **diagnostics on already-judged
candidates**, labelled derivations/diagnostics, never gate
verdicts, never training (§17 items 12, 22) — the
train-partition projection coefficient α it would carry,
the parallel component's share of its movement, and the
rehearsal projected candidate (Δ′ = Δ − α·Δ1, constructed
through the same float32 fold the built slice uses) with
its signal held-out record and norms. Allowed inputs only:
signal 2's train partition (32 sequences) and held-out
partition (16 sequences, reported only). The denominator
Σ_x ⟨m1(x), m1(x)⟩ is a property of the landed head and is
the same for all three candidates: 1.9968972245257979
(‖m1‖ aggregate on the train partition 1.4131161397867473).

| Rehearsal (each under its own labels) | α (train partition) | Numerator Σ⟨m1, m2⟩ (train) | Parallel component ‖α·m1‖ (train aggregate) | Share of ‖m2‖ (train aggregate) | ‖m2‖ train, pre → post | Candidate norm, pre → post |
|---|---|---|---|---|---|---|
| SI-25 control (`"none"`) | −0.8450027779470831 | −1.6873837019991194 | 1.19408706368166 | 71.2% (of 1.6765305214306858) | 1.6765305214306858 → 1.1768221925416444 | 0.2515524481203623 → 0.2890208582779514 |
| SI-27 treatment (`"movement-composition"`) | −1.3459658380704567 | −2.6877554463494344 | 1.902006049378958 | 85.3% (of 2.2303160062528544) | 2.2303160062528544 → 1.1647670917806054 | 0.2969693161862222 → 0.366754808469668 |
| SI-29 treatment (`"worst-input-composition"`) | −1.2483510742746164 | −2.49282879545278 | 1.764065051177585 | 82.3% (of 2.1423496133427675) | 2.1423496133427675 → 1.2156217163204783 | 0.28252038537434354 → 0.3505026282753374 |

| Rehearsal projected candidate (diagnostic) | Train aggregate inner product with m1, pre → post (measured) | Held-out aggregate inner product, pre → post (reported only) | Signal held-out loss, candidate → rehearsal projected (closer) | Signal train loss, candidate → rehearsal projected |
|---|---|---|---|---|
| SI-25 control | −1.6873837019991194 → −4.5362175883634226e-08 | −0.845596374259425 → −0.0009897604352243888 | 1.043608588752459e-05 → 0.001404923739144991 (16/16 → 16/16) | 9.412652536610669e-06 → 0.0014032997448586238 |
| SI-27 treatment | −2.6877554463494344 → −1.1595250314078154e-07 | −1.3460015097625617 → −0.0006667292197597534 | 0.0005227812955486338 → 0.0014293604799881757 (16/16 → 16/16) | 0.0005191352438434636 → 0.0014244850313627471 |
| SI-29 treatment | −2.49282879545278 → −1.4108283225011675e-07 | −1.2470211966092322 → +0.0007445007456977869 | 0.0006602814776996296 → 0.0017458852255362923 (12/16 → 14/16) | 0.0006509292699374634 → 0.001728253214220491 |

Rehearsal probe diagnostics (diagnostics on already-judged
candidates, measured through candidate-weight overlays —
not gate verdicts): every rehearsal projected candidate is
far worse at the probes than the candidate it derives
from. Composed maxima: SI-25-based 0.47753881290555,
SI-27-based 0.5554727502167225, SI-29-based
0.5258305259048939 — all at `tiny-id-0`; inner products at
`tiny-id-0`: +0.14235826394688586, +0.21079050222308737,
+0.20969554919790298 (against the unprojected candidates'
+0.0338873784299374, +0.03801222744105431,
+0.04944786620153699); aggregate probe inner products:
+0.21576809097010846, +0.35511991736802795,
+0.34527215722378113 — **positive (aligned) in aggregate**,
where every unprojected candidate opposed in aggregate.
Head-incremental capability regressions of the rehearsal
projected candidates: 0.1772811310365796 (SI-25-based,
pass), 0.2588021755218506 (SI-27-based, would fail eps
0.25), 0.24198293685913086 (SI-29-based, pass).

### Phase 0 readings

- **The removable component is not negligible — it is the
  majority of each candidate's movement on allowed
  inputs.** All three coefficients are large and negative
  (α = −0.845, −1.346, −1.248): on the train partition the
  second movement is predominantly ANTI-parallel to the
  first (the aggregate inner products are strongly
  negative), so "removing the parallel component" means
  removing 71–85% of the movement's aggregate magnitude —
  and because α is negative, the folded construction
  Δ′ = Δ2 − α·Δ1 ADDS |α|·Δ1 in weight space: every
  rehearsal projected candidate's weight-space norm is
  LARGER than its pre-projection norm (0.252 → 0.289,
  0.297 → 0.367, 0.283 → 0.351), while its aggregate
  movement norm on the train partition falls (the
  projected movement is orthogonal to m1 there, by
  construction and by measurement). §5.3's
  distinguishability obligation is therefore discharged
  in a sharper form than the RFC anticipated: this
  construction cannot improve a maximum "by shrinking"
  the candidate — the candidate grows; whatever it does
  at the probes, it does as a larger, turned candidate.
- **Removing it spends most of the destination on every
  rehearsal candidate.** Each rehearsal projected
  candidate's signal held-out loss is far worse than its
  unprojected candidate's (SI-25-based: ~135× the
  control's residual; SI-27-based: ~2.7×; SI-29-based:
  ~2.6×) — the parallel component the projection removes
  is, on allowed inputs, largely the component that
  carries the second signal's lean (both corrections
  share the component SI-26 §10 named, measured here in
  the opposite direction from the probe record). Every
  rehearsal projected candidate still improves on the
  pre-training held-out loss (0.002757805049077469), so
  the non-improvement refusal does not fire on any of
  them — but the destination each approaches, it
  approaches far less completely.
- **The rehearsal already shows the judged end, as a
  diagnostic.** At the probes, removing the train-parallel
  component makes the composed maximum dramatically
  worse on all three candidates (0.478 / 0.555 / 0.526
  against 0.333 / 0.325 / 0.312 unprojected) and turns the
  aggregate probe composition from opposing to aligned.
  The component that is anti-parallel on allowed inputs
  is, at the judged inputs, load-bearing in the opposite
  sense: the train partition and the probe suite price
  the same component with opposite signs. This is F1's
  train-to-probe gap at the level of a single component,
  measured before the decisive run — and it is why the
  falsification readings below are registered expecting
  F1′′ to fire (RFC §18 item 1: the slice must be built
  to publish it).

### The representation finding (measured in Phase 0; not in the RFC)

RFC §7.2 defines the judged candidate by its folded
update, Δ2′ = Δ2 − α·Δ1, and §7.2's exclusion note reasons
about what "a single LoRA candidate can carry" only for
the per-input coefficient form. Phase 0 measured what the
RFC did not price: **the difference of two rank-8 folded
updates is generically rank 16** (the SI-25-based Δ′ has
16 nonzero singular values — 14 substantial, then
0.00470232322 and 0.000156309136; likewise for the other
rehearsals), so Δ2′ cannot be carried exactly by any
rank-8 LoRA candidate. The slice therefore carries a
projected candidate (α ≠ 0) as a rank-16 LoRA
factorisation of Δ2′ (SVD factors, rank 16, alpha 16,
scale 1.0 — the same scale as every prior candidate;
4408 blob bytes for the decisive candidate against 2360
for a rank-8 candidate). The alternatives were rejected
on the record: a rank-8 truncation of Δ2′ would NOT be
the §7.2 construction (it would leave a train-partition
aggregate inner product of the truncation's choosing, not
zero), and an additive target is policy-forbidden
(`allow_additive` false, unchanged). The consequence is
measured, not hidden: the unchanged gate's safety suite
prices rank against `rank_cap` 8, so a projected
candidate fails the `size_budgets` subcheck independently
of its identity drift — the decisive run's journaled
safety check fails on exactly that line, beside an
identity check that also fails on its own numbers. Both
failures are reported as measured wherever the verdict is
reported. At α = 0.0 the construction is the identity and
no rank question arises (the coincidence configuration
below).

### Which slice this turned out to be (§7.4)

Phase 0 does **not** show the parallel component is
negligible — it is the majority of the movement — and it
does not show that removing it collapses every rehearsal
candidate's destination outright (all three still improve
on the pre-training loss). §7.2's construction therefore
stands as designed and Phase 1 built it unchanged. The
honest narrowing Phase 0 *does* impose is recorded here:
the rehearsal's probe diagnostics and held-out records
together predict, as diagnostics, that the decisive
projected candidate will fail at the binding probe by a
wide margin and will trade destination completeness past
the F3′′ boundary — the slice that was built is a
measurement of the projection itself under exactly that
expectation, and its decisive readings are read against
exactly that premise (RFC §18 item 1: F1′′ is the expected
outcome, and a slice that reports it as a surprise has
misread its own RFC).

### The pre-registered successor readings (RFC §§5.5, 7.5 — stated before the decisive numbers)

> **F1′′ — the projection does not transfer** is
> **measured** iff BOTH hold for the decisive projected
> candidate (signal `tiny-predecessor-lean-001`, head
> `d0001`, target source `base`, anchoring
> `base-anchored`, composition
> `"train-partition-projection"`): (a) its
> train-partition composition is zero as constructed and
> as measured — aggregate inner product with `m1` within
> the projection-correctness tolerance (below) of 0.0;
> and (b) its probe composition at `tiny-id-0` — the
> probe SI-29's treatment's figure names — is unchanged
> or worse than SI-29's treatment's
> +0.04944786620153699 at that probe.
>
> **F2′′ — the maximum is immobile even when the binding
> probe turns** is judged as TWO SEPARATELY REPORTED
> CLAUSES (the repaired form §5.5 requires; no conjunction
> is smuggled back): (a) the *transfer clause* holds iff
> the projected candidate's probe inner product at its
> own worst probe improves on (is smaller than) SI-29's
> treatment's inner product at that same probe, with its
> aggregate probe composition published beside SI-27's
> (−0.3271477846470207) and SI-29's (−0.28751472160662706)
> treatments' figures under their own labels; (b) the
> *maximum clause* holds iff the composed maximum stays
> above tol 0.25. F2′′ **is measured** iff (a) and (b)
> both hold; each clause's outcome is reported either
> way.
>
> **F3′′ — the destination is traded away** is read under
> SI-29's three clauses unchanged in their thresholds:
> the decisive projected candidate's destination counts
> as **approached** (F3′′ does **not** fire) iff ALL
> THREE hold for its signal held-out record (same labels
> as F1′′): (a) the held-out signal loss **falls**
> (`held_out_loss_after < base_held_out_loss`);
> (b) at least **12 of 16** held-out examples are closer
> after construction; and (c) the held-out residual is at
> most **2× the re-run SI-29 control's residual** — the
> completeness reference is the `"worst-input-composition"`
> control re-run inside this slice (its published
> residual is 0.0006602814776996296, so the boundary is
> **0.0013205629553992592**). If any one of (a)–(c)
> fails, the destination counts as **traded away** and
> F3′′ **is measured**.

### The projection-correctness tolerance (RFC §7.4 (iii) — stated before the decisive numbers)

> The built slice's measured post-projection aggregate
> inner product on the train partition counts as zero iff
> its absolute value is at most **1e-6** — stated from the
> Phase-0 rehearsal figures above (the largest rehearsal
> |aggregate| under the float32 construction is
> 1.4108283225011675e-07, the SI-29-based rehearsal; the
> tolerance is ~7× that figure), before the decisive run.

## The construction, precisely (§7.2)

Under `"train-partition-projection"`, `train_candidate`
trains signal 2's factors against SI-25's base-anchored
signal loss exactly — the unpenalised training path, no
penalty term of either prior kind — producing the
construction intermediate (in the decisive run, exactly
SI-25's control candidate: its held-out record appears
below under `projection_pre_*` labels). It then computes,
on signal 2's 32 training sequences only, the movements
`m1(x)` (landed head minus frozen base) and `m2(x)`
(intermediate minus landed head) under SI-27's definitions,
and the coefficient

  α = ( Σ_x ⟨m1(x), m2(x)⟩ ) / ( Σ_x ⟨m1(x), m1(x)⟩ )
    = −1.6873837019991194 / 1.9968972245257979
    = **−0.8450027779470831**  (decisive run, measured).

The derivation the RFC requires (§7.2): both movements
pass through the same readout fold — `m1(x) = Δ1 · h(x)`
and `m2(x) = Δ2 · h(x)` with `h(x)` the final hidden state
at `x` under the landed head (the readout is the final
operation, so `h` does not depend on either update) — so
the candidate carrying the folded update
Δ2′ = Δ2 − α·Δ1 has train-partition movements
`m2(x) − α·m1(x)`, whose aggregate inner product with
`m1` is Σ⟨m1, m2⟩ − α·Σ⟨m1, m1⟩ = 0 by the definition of
α. The slice does not rest on the derivation: the
projected candidate's allowed-input composition record is
measured directly through the forward path (next
sections) and its train aggregate sits at
−4.8567197048765554e-08, within the Phase-0 tolerance of
0.0. The coefficient is not computed per input and not
computed in weight space (both exclusions of §7.2 hold by
construction: α is one scalar from train-partition
movement aggregates; the weight-space inner product of
the folded updates is a separate, published diagnostic —
for the decisive candidate it is **+0.03542756883828178**,
positive, beside the movement-space numerator
−1.6873837019991194: the two geometries disagree in sign
here, which is exactly why §7.2 defines α in movement
space and why the weight-space variant SI-25's table
refuted is not this construction). The removal is full
(α as defined, not scaled); there is no
projection-strength parameter anywhere in the slice.

**Post-projection accounting** is part of the
construction: the projected candidate's signal train loss
(0.0014032997323342922), signal held-out loss
(0.00140492372619575, 16/16 closer — the refusal judges
this figure), candidate norm (0.2890208630043799) beside
its pre-projection norm (0.25155244947498906, the
intermediate's applied-update norm; its recorded Gram
norm is 0.2515524481203623), and the allowed-input
composition record on both partitions are all measured
and recorded before the candidate is returned. The gate
then judges the projected candidate and no other.

**Coincidence configuration (§15 criterion 5).**
Configured with signal 2, base-anchoring, and the
projection configuration, a first cycle at base records
the coefficient and the construction's records:
**α = 0.0, measured** — `m1` is the zero vector at base,
so both aggregate inner products in α's definition
vanish (numerator 0.0, denominator 0.0, recorded) and α
is 0.0 by the definition's denominator rule; the
construction is the identity (the candidate keeps its
trained rank-8 factors; post-projection norms and losses
equal the intermediate's; the measured aggregate inner
products are 0.0 on both partitions) — and the run
reproduces the established at-base run **exactly**:
candidate blob
`sha256:131cc242728024b4f833bc78ea627825f1e8616972fc2670603673e1814645f4`,
identity drift 0.16672966815531254, capability regression
0.2418281203135848, ADMIT → `d0001`, §6.6 VERIFY pass.
The projection machinery is therefore proven correct
where it must vanish, before the decisive cycle is
trusted. These are at-base facts; they are not the
slice's decisive claim.

## Gradient check (§14 item 6, §15 criterion 3)

No new loss exists under this configuration — the
training step is SI-25's unpenalised loss, byte-for-byte —
and the construction introduces **no new differentiable
path**: α is a ratio of measured forward-path aggregates
computed after training, and the factorisation is a
post-training representation step, not a function any
gradient flows through. SI-25's finite-difference check
therefore stands for the training step unchanged, cited
here, not silently assumed and not re-run: signal 2's
base-anchored loss, central finite differences over every
one of the 512 factor entries, worst disagreement
5.5249277028738815e-08 at base (tolerance 1e-5, published
in `docs/SI-25-base-anchored-targets.md`). The
construction's own correctness evidence is the measured
projection-correctness check above (aggregate inner
product through the forward path, within the Phase-0
tolerance) and the folded-space verification in the test
suite (the judged candidate's folded update equals
Δ2 − α·Δ1 against the re-run SI-25 control to < 1e-5 per
entry).

## Probe-blindness of training and construction (§15 criterion 4)

Three assertions, all in
`tests/test_train_partition_projection.py`:

- A spy on `projection_coefficient` during a real
  projected `train_candidate` run captures every call's
  movement-map key sets and sequence set: each is
  **exactly signal 2's train partition** (as a set) — the
  coefficient's input set is the train partition, asserted
  as a set, on the code path that constructed the
  decisive candidate (which the spy run reproduces: blob
  `sha256:32d13f65…a5c1`).
- The train partition's disjointness from both probe
  suites (identity and capability) and from the signal's
  held-out partition is re-asserted under the new code
  path.
- A copy of the bundle whose `probe_suite.json` was
  perturbed on disk (probe inputs and expected logits
  replaced) produces a **byte-identical projected
  candidate** through `train_candidate` called directly
  with the pre-perturbation verified manifest — neither
  the training path nor the construction ever opens the
  probe artifact, so no probe input, probe logit, or
  probe-derived quantity can reach the loss, the
  coefficient, the construction, or the refusal.
  (`load_bundle` itself refuses the perturbed copy —
  fail-closed tamper evidence, a separate, pre-existing
  guarantee.)

## The allowed-input projection records (§15 criteria 3 and 8)

For the decisive **treatment** candidate (signal
`tiny-predecessor-lean-001`, head `d0001`, target source
`base`, anchoring `base-anchored`, composition
`"train-partition-projection"` — blob
`sha256:32d13f65…a5c1`): the composition record under
SI-27's movement definitions on signal 2's train and
held-out partitions, measured through the forward path.
The controls' train aggregates are their published
figures (SI-25: −1.6873837019991191; SI-27:
−2.687755446349435; SI-29: −2.4928287954527795 — the
treatment's α numerator, −1.6873837019991194, is the
SI-25 control's aggregate, as it must be: the
intermediate IS the SI-25 control candidate).

| Record (partition labelled) | Aggregate inner product | Sign agreement | Mean `p` | ‖m1‖ | ‖m2‖ |
|---|---|---|---|---|---|
| **SI-31 treatment (projected), train** | **−4.8567197048765554e-08** | 0.390625 (400/1024) | −4.7428903354382584e-11 | 1.4131161397867473 | 1.176822205010781 |
| **SI-31 treatment (projected), held-out** | **−0.0009897609251914505** | 0.3984375 (204/512) | −1.9331268070145517e-06 | 0.9997655382584564 | 0.8300941917309416 |

(Sign agreement here is the record's per-coordinate
aggregate over the partition's movements, published
beside the magnitude-weighted figure, never alone —
§17 item 19.)

Readings, stated plainly:

- **On the train partition, the construction did exactly
  what it defines.** The aggregate inner product moved
  from the intermediate's −1.6873837019991194 to
  −4.8567197048765554e-08 — zero within the Phase-0
  tolerance (1e-6), measured through the forward path,
  not derived from the construction's algebra. The
  projected movement's aggregate norm on the train
  partition is 1.176822205010781, down from the
  intermediate's 1.6765305214306858 — the orthogonalised
  remainder, exactly the component of `m2` perpendicular
  to `m1` on that population.
- **On the held-out partition (reported only — the
  construction never reads it), the aggregate is near
  zero but not zero** (−0.0009897609251914505, against the
  intermediate's −0.845596374259425): the train-partition
  coefficient nearly orthogonalises the held-out
  movements too, because both allowed partitions price
  the parallel component with the same sign. The gap
  between the allowed partitions and the probes — where
  the same construction's aggregate is **+0.2158,
  aligned** — is the transfer failure, stated in both
  places (§17 item 18). No figure above is a probe
  figure, and no probe figure below is an allowed-input
  figure.

## The decisive two-cycle run (measured on this VM, this slice's runs)

Driver shape: **one reused `TinyTransformerMind`** driven
through `si.run.run_cycles` — one `run_cycles` call for
cycle 1 (signal 1, default anchoring, default composition
configuration), then the driver explicitly reconfigures
the same mind to signal 2 (`set_training_signal`), to
base-anchored targets (`set_target_anchoring`), **and**
to the projection composition configuration
(`set_composition_config`) — three recorded selections —
then a second `run_cycles` call for cycle 2. Exactly two
authored cycles were attempted, and exactly one
projection decisive run was performed. This is SI-29's
decisive run with exactly one configuration changed:
cycle 2's composition configuration.

| | Cycle 1 | Cycle 2 (treatment, projected) |
|---|---|---|
| Signal (recorded selection) | `tiny-successor-lean-001` (default) | `tiny-predecessor-lean-001` (explicit) |
| Anchoring (recorded selection) | `weights-under-training` (default) | **`base-anchored` (explicit)** |
| Composition (recorded selection) | `none` (default), all composition fields null | **`train-partition-projection` (explicit), α −0.8450027779470831 measured** |
| Head trained against | `base` | `d0001` |
| Target-source head | `base` | **`base`** |
| Stack depth at training | 0 | 1 |
| Candidate blob | `sha256:2761e8ff…6ca9e` (2,360 B) | `sha256:32d13f65…a5c1` (4,408 B, rank 16) |
| Candidate norm (pre-projection → judged) | 0.23555380990005192 | 0.25155244947498906 → **0.2890208630043799** |
| Signal train loss, initial → final (training step) | 0.0028961365437885785 → 1.824004756500067e-06 | 0.0042998498036725735 → 9.412649442227854e-06 (intermediate's) |
| Projected candidate's signal train loss | — | **0.0014032997323342922** |
| Pre-training held-out signal loss | 0.00140625 | **0.002757805049077469** (genuine measurement, as in SI-25) |
| Held-out signal loss after (judged candidate) | 3.2118398935406443e-06 | **0.00140492372619575** (~2.0× lower; intermediate's: 1.043608588752459e-05, ~264×) |
| Held-out examples improved | 16/16 | **16/16** (intermediate's: 16/16) |
| Norm-guard activations | 0 | 0 |
| Gate verdict | **ADMIT** → `d0001`, §6.6 VERIFY pass | **REJECT** (safety: rank 16 > rank_cap 8; identity: drift 0.4775387533009052 > tol 0.25), nothing landed |
| Blob-store growth | +2,360 B | +0 B (candidate never written) |

Full candidate blob hashes: cycle 1
`sha256:2761e8ffc139ce3e107c11da035e8a73c20792945a494f4e493dccf418b6ca9e`;
cycle 2
`sha256:32d13f65d0de1de2901fcd1c1a5cdece12cdfe441454f5d75e8932941827a5c1`.
Cycle 1 reproduces SI-19/SI-21/SI-23/SI-25/SI-27/SI-29
**exactly** (criterion 9). All three controls re-run
inside this slice reproduce their decisive candidates
**exactly** (SI-25: blob `sha256:646d1e8b…d675`, drift
0.3329036422073841, regression 0.10096919978968799, under
`"none"`; SI-27: blob `sha256:1bb277ab…b637`, drift
0.32509003952145576, regression 0.14536726917140186, under
`"movement-composition"` with λ = 1.0; SI-29: blob
`sha256:11cab02e…b8db5`, drift 0.31215621903538704,
regression 0.11724743689410388, under
`"worst-input-composition"` with λ_w = 1.0, k = 4), so
the treatment's difference is attributable to the
construction step and not to the refactor that
introduced it.

## Cumulative vs incremental accounting (§15 criterion 8) — side by side

| Measurement | Cycle 1 | Cycle 2 (treatment, projected) | Accounting |
|---|---|---|---|
| Identity drift (gate, candidate overlay) | 0.17189218290150166 (pass, tol 0.25) | **0.4775387533009052 (FAIL, tol 0.25)** | **seed-absolute / cumulative** |
| Capability regression (gate) | 0.1657181391492486 (pass, eps 0.25) | 0.177281160838902 (pass, eps 0.25) | **head-incremental** |
| Echo accuracy | 1.0 (16/16, floor 0.9375) | 1.0 (16/16, floor 0.9375) | absolute, vs the recorded floor |
| Safety suite (gate, decisive check) | pass | **FAIL — `size_budgets`: rank 16 exceeds rank_cap 8** (all other subchecks pass; aggregate norm 0.289021 ≤ cap 1.0) | policy, unchanged |
| Landed head's cumulative identity drift after the cycle | 0.17189218290150166 (§6.6 VERIFY, measured-pass) | 0.17189218290150166 (unchanged — re-measured after the REJECT) | seed-absolute, of the head that actually landed |

No incremental figure above is presented as cumulative or
vice versa, no base-anchored figure as a head-relative
one, no figure under one composition configuration as a
figure under another, and no trained candidate's figure
as the projected candidate's or vice versa (§17 items 9,
15, 17, 22).

## The diagnostics (published whatever the verdict)

All diagnostics below are computed against head `d0001`
with the decisive cycle-2 (projected) candidate overlaid.
They are published to *attribute* the gate's verdict
after the gate has spoken; they are not gate checks and
influenced no verdict (§17 item 12). Per-probe drift
values are measurements (max absolute logit coordinate
per probe); the incremental movement column is a
**derivation** from two measured forwards (composed
forward minus the `d0001` forward), labelled as such.
Candidate norm and ‖m2‖ stand beside every drift figure
(§§5.3, 10 of the RFC): the treatment's judged candidate
norm is 0.2890208630043799 (pre-projection:
0.25155244947498906) and its probe-space ‖m2‖ is
0.7834031060820517.

**(i) Per-probe drift contributions (treatment,
projected):**

| Probe | `d0001` drift (measured) | Signal-2 projected incremental movement (derivation) | Composed drift (measured) |
|---|---|---|---|
| tiny-id-0 | 0.17189218290150166 | 0.3063734769821167 | **0.4775387533009052** |
| tiny-id-1 | 0.15088725090026855 | 0.06947946548461914 | 0.2203667163848877 |
| tiny-id-2 | 0.11858986946754158 | 0.14929723739624023 | 0.14972496032714844 |
| tiny-id-3 | 0.14782869815826416 | 0.14890391938388348 | 0.2727522850036621 |
| tiny-id-4 | 0.11720322235487401 | 0.13440751284360886 | 0.21748418174684048 |
| tiny-id-5 | 0.09891846403479576 | 0.09731395542621613 | 0.19604265317320824 |
| tiny-id-6 | 0.12363102938979864 | 0.14727314934134483 | 0.13208984676748514 |
| tiny-id-7 | 0.08795571327209473 | 0.13717103376984596 | 0.20603866130113602 |

Worst probes, named: `d0001`'s own drift —
**tiny-id-0** (0.17189218290150166); the composed drift —
**tiny-id-0** (0.4775387533009052, equal to the gate's
measured cycle-2 drift). The incremental movement's
largest probe is also tiny-id-0 (0.3063734769821167). At
tiny-id-0 the composed drift is again, to the printed
digits, the sum of the two contributions (0.17189218290150166 +
0.3063734769821167 ≈ 0.4775) — the aligned extreme, at
the same probe where cycle 1 moved most, as in SI-25,
SI-27, and SI-29, and by far the largest sum on record:
the projected second movement at the binding probe
(0.3063734769821167) is nearly twice SI-27's treatment's
incremental movement there (0.15392476320266724, from
SI-29's as-built accounting). There is no worst *allowed input*
in this slice's construction to confuse with the worst
probe: the projection has no per-input worst case — its
worst-case figures are the aggregates above (§17 item
20).

**(ii) Alignment diagnostic (derivation):** `m1` =
(`d0001` forward − seed expected) and `m2` = (composed
forward − `d0001` forward), per identity probe:

- **In aggregate the projected movement ALIGNS with the
  first — the first treatment on record whose aggregate
  probe composition is positive.** Over all 8 identity
  probes (256 coordinates): inner product
  **+0.21576810032436344** (aligned — SI-23: +0.07637087369869666;
  SI-25 control: −0.21256230517288188; SI-27 treatment:
  −0.3271477846470207; SI-29 treatment:
  −0.28751472160662706); sign agreement **0.5** (128/256)
  — published beside the magnitude-weighted figure, never
  alone (§17 item 19); movement-vector norms ‖m1‖ =
  0.7119678353441855 (unchanged — cycle 1's movement),
  ‖m2‖ = 0.7834031060820517 (SI-25: 0.7816722548913912;
  SI-27: 0.9688074699782941; SI-29: 0.9424814001349618).
- **At the binding probe it fails by the widest margin
  on record.** At `tiny-id-0`: inner product
  **+0.14235826783471692** (positive — aligned; SI-25
  control: +0.0338873784299374; SI-27 treatment:
  +0.03801222744105431; SI-29 treatment:
  +0.04944786620153699). The projection zeroed the
  composition on the train partition exactly — and at
  the one *probe* where the gate binds, the alignment
  nearly tripled the worst predecessor's.

**(iii) Folded updates (measured):** Frobenius inner
product of the two folded `lm_head.weight` updates
**+0.03542756883828178** (ALIGNED in weight space —
SI-25: −0.011457914936582613; SI-27: −0.02013451284871203;
SI-29: −0.0173962867164366, all opposing); norms
‖Δ1‖_F = 0.23555380990005192, ‖Δ2′‖_F =
0.2890208630043799 (Δ2′'s norm equals its candidate norm,
as it must for a single-target LoRA). The projected
candidate is LARGER than its intermediate (0.28902
against 0.25155) and larger than SI-25's control, smaller
than SI-27's treatment (0.29697) and larger than SI-29's
(0.28252): the composed maximum's dramatic worsening
(0.47754 against every predecessor) is therefore **not**
an improvement or a failure "by shrinking" in either
direction that §5.3 guards — the candidate GREW while
its movement at the binding probe nearly doubled
SI-27's treatment's step there (0.30637 against
0.15392476320266724), because the component removed on
allowed inputs is the component that was *restraining*
the movement at the binding probe. Improvement-by-removal is distinguishable
from improvement-by-turning and from
improvement-by-shrinking by these figures standing
together (§17 item 22): there is no improvement here to
distinguish — there is a removal whose judged effect is
the opposite of its allowed-input effect, at full size.

**(iv) Composed capability (measured):** head-incremental
regression **0.177281160838902** ≤ eps 0.25 (pass;
SI-25: 0.10096919978968799; SI-27: 0.14536726917140186;
SI-29: 0.11724743689410388); echo accuracy **1.0
(16/16)** vs the recorded floor 0.9375.

**(v) The six-run control/treatment table (§14 item 8,
§17 item 16).** Same signal, same head, same target, same
hyperparameters — differing only in anchoring (SI-23 vs
SI-25), then only in composition configuration (SI-25 vs
SI-27 vs SI-29), then only in construction (SI-31's
intermediate is SI-25's candidate; SI-31 differs from
SI-25's run by the projection step alone). SI-21's run
predates the per-probe table's form; its composed
maximum (0.34378434903919697, REJECT) is carried in the
summary table below. Each figure under its own run's
labels:

| Per-probe composed drift | tiny-id-0 | tiny-id-1 | tiny-id-2 | tiny-id-3 | tiny-id-4 | tiny-id-5 | tiny-id-6 | tiny-id-7 | Max (worst probe) |
|---|---|---|---|---|---|---|---|---|---|
| SI-23 — signal 2, weights-under-training, composition n/a (pre-SI-27) | 0.2092923242598772 | 0.15134692192077637 | 0.14906740188598633 | **0.2963416576385498** | 0.23394920490682125 | 0.14004023000597954 | 0.1332004750147462 | 0.2018739078193903 | 0.2963416576385498 (tiny-id-3) |
| SI-25 — signal 2, base-anchored, composition `"none"` | **0.3329036422073841** | 0.09286642074584961 | 0.14936351776123047 | 0.1489148810505867 | 0.14728415571153164 | 0.11245626583695412 | 0.14491976890712976 | 0.14189285784959793 | 0.3329036422073841 (tiny-id-0) |
| SI-27 — signal 2, base-anchored, composition `"movement-composition"`, λ 1.0 | **0.32509003952145576** | 0.09391975402832031 | 0.1488508302718401 | 0.14755446463823318 | 0.10617947578430176 | 0.1450248844921589 | 0.15063166618347168 | 0.18760260567069054 | 0.32509003952145576 (tiny-id-0) |
| SI-29 — signal 2, base-anchored, composition `"worst-input-composition"`, λ_w 1.0, k 4 | **0.31215621903538704** | 0.1174006462097168 | 0.14997844584286213 | 0.19014504551887512 | 0.11752748489379883 | 0.13741302862763405 | 0.15289807319641113 | 0.19354498013854027 | 0.31215621903538704 (tiny-id-0) |
| SI-31 — signal 2, base-anchored, composition `"train-partition-projection"`, α −0.8450027779470831 (constructed) | **0.4775387533009052** | 0.2203667163848877 | 0.14972496032714844 | 0.2727522850036621 | 0.21748418174684048 | 0.19604265317320824 | 0.13208984676748514 | 0.20603866130113602 | 0.4775387533009052 (tiny-id-0) |

| Gate / candidate figure | SI-21 (head-relative, pre-SI-23 signal set) | SI-23 (head-relative) | SI-25 (base-anchored, `"none"`) | SI-27 (base-anchored, mean-penalised) | SI-29 (base-anchored, worst-input) | SI-31 (base-anchored, projected) |
|---|---|---|---|---|---|---|
| Candidate blob | (SI-21 as-built) | `sha256:184cc7f0…6e6cf7` | `sha256:646d1e8b…d675` | `sha256:1bb277ab…b637` | `sha256:11cab02e…b8db5` | `sha256:32d13f65…a5c1` |
| Candidate rank | 8 | 8 | 8 | 8 | 8 | **16 (constructed; see Phase 0)** |
| Candidate norm | (SI-21 as-built) | 0.21317186599953494 | 0.2515524481203623 | 0.2969693161862222 | 0.28252038537434354 | 0.2890208630043799 (pre-projection 0.25155244947498906) |
| Identity drift (cumulative) | 0.34378434903919697 — FAIL | 0.2963416576385498 — FAIL | 0.3329036422073841 — FAIL | 0.32509003952145576 — FAIL | 0.31215621903538704 — FAIL | **0.4775387533009052 — FAIL** |
| Capability regression (head-incremental) | (SI-21 as-built) | 0.15023326850496233 — pass | 0.10096919978968799 — pass | 0.14536726917140186 — pass | 0.11724743689410388 — pass | 0.177281160838902 — pass |
| Echo accuracy | (SI-21 as-built) | 1.0 — pass | 1.0 — pass | 1.0 — pass | 1.0 — pass | 1.0 — pass |
| Aggregate probe inner product | (SI-21 as-built) | +0.07637087369869666 | −0.21256230517288188 | −0.3271477846470207 | −0.28751472160662706 | **+0.21576810032436344** |
| Inner product at own worst probe | (SI-21 as-built) | (SI-23 doc: aligned at tiny-id-3) | +0.0338873784299374 (tiny-id-0) | +0.03801222744105431 (tiny-id-0) | +0.04944786620153699 (tiny-id-0) | **+0.14235826783471692 (tiny-id-0)** |
| Verdict | REJECT | REJECT | REJECT | REJECT | REJECT | **REJECT (safety rank + identity)** |

Per probe, the treatment's composed drift improves on
SI-29's treatment at 1 of 8 probes (tiny-id-6) and
worsens at 7; against its own intermediate (SI-25's
control) it improves at 1 of 8 (tiny-id-6) and worsens
at 7. The maximum worsens on every predecessor
(0.34378 / 0.29634 / 0.33290 / 0.32509 / 0.31216 →
0.47754) and stays far above tol 0.25. The treatment's
incremental movement exceeds cycle 1's remaining
headroom (0.07810781709849834) at **7 of 8** probes (all
but tiny-id-1, at 0.06947946548461914 — under). No
diagnostic was uncomputable; nothing in this set is
reported "not evaluated".

**(vi) The falsification readings (RFC §7.5), each
stated as measured, under the rules registered in
Phase 0:**

- **F1′′ — the projection does not transfer: MEASURED.**
  The projected candidate's train-partition composition
  is zero as constructed and as measured (aggregate
  inner product −4.8567197048765554e-08, within the
  Phase-0 tolerance 1e-6 of 0.0), while its probe
  composition at `tiny-id-0` is far worse than SI-29's
  treatment's (+0.04944786620153699 →
  **+0.14235826783471692**). Falsified premise, in the
  RFC's words: that the parallel component measured on
  allowed inputs is the component that adds at judged
  inputs, through this readout, at this scale. The
  measured record is sharper than the premise's
  negation: the train-parallel component is
  ANTI-parallel on allowed inputs (α = −0.845), so its
  removal adds the first movement's direction back —
  and at the binding probe that is precisely the
  direction the composed drift was already worst in.
  Consequence stated in advance (§7.5): with the mean,
  the worst case, and the projection all measured
  failing at the binding probe, no composition move that
  reads only allowed inputs remains on this RFC's
  record, and the line's honest continuation is a later
  RFC's genuinely new argument — not a fourth variation.
  RFC §18 item 1 recorded this as the *expected*
  outcome; it is reported as a measurement, not as a
  surprise.
- **F2′′ — judged as two separately reported clauses
  (the repaired form), NOT measured as a reading: its
  transfer clause fails, its maximum clause's fact
  holds.** (a) *Transfer clause: FAILS* — the projected
  candidate's inner product at its own worst probe
  (tiny-id-0, +0.14235826783471692) does not improve on
  SI-29's treatment's figure at that probe
  (+0.04944786620153699); it worsens by ~2.9×. Its
  aggregate probe composition is **+0.21576810032436344
  (aligned)**, published beside SI-27's treatment's
  −0.3271477846470207 and SI-29's treatment's
  −0.28751472160662706 under their own labels, with no
  no-worse conjunction required or claimed. (b)
  *Maximum clause: the fact HOLDS* — the composed
  maximum stays above tol 0.25 (0.4775387533009052).
  Because clause (a) fails, F2′′ does not fire in its
  registered form, and this document does not round it
  into one: the maximum's immobility is a fact of this
  run measured beside the reading, and its cause is
  F1′′'s, not F2′′'s — the binding probe did not merely
  fail to turn, it turned the wrong way, so the
  proposition F2′′ was written to test (the maximum as a
  magnitude-and-geometry fact despite transfer) is not
  the proposition this run measured.
- **F3′′ — the destination is traded away: MEASURED,
  under the pre-registered rule, by its completeness
  clause.** Applying the Phase-0 rule as written to the
  treatment's signal held-out record (signal
  `tiny-predecessor-lean-001`, head `d0001`, target
  source `base`, anchoring `base-anchored`, composition
  `"train-partition-projection"`): (a) the loss falls —
  0.002757805049077469 → 0.00140492372619575 ✓; (b)
  closer fraction — **16/16**, comfortably past the
  rule's 3/4 boundary ✓ (the intermediate's: 16/16); (c)
  residual completeness — 0.00140492372619575 ≤
  0.0013205629553992592 (2× the re-run SI-29 control's
  residual 0.0006602814776996296) ✗ **FAILS** — the
  projected candidate's residual exceeds the boundary by
  ~6%. One clause failing suffices under the rule as
  registered: the destination counts as **traded away**
  and F3′′ is measured. The trade's shape is published
  beside the reading: the projection spent ~135× of the
  intermediate's completeness (residual
  1.043608588752459e-05 → 0.00140492372619575) while
  keeping every held-out example closer — a candidate
  that still leans the right way on all 16 examples,
  and arrives only ~half as far in loss as it started
  from. Any future reading of this run that quotes the
  train-partition orthogonalisation without this cost
  beside it is blending the record.

**Attribution:** all six cycle-2 REJECTs in the project's
record are now measured, and the composition line's
attribution is complete at every move the line proposed.
SI-27 proved the mean of `p(x)` on allowed inputs can be
shaped and does not bind the judged maximum. SI-29
proved the same for the worst case of `p(x)`. SI-31
proves it for the strongest form: the parallel component
itself, removed outright, exactly, as measured on the
only population construction may read — the construction
succeeded perfectly on its own terms (aggregate inner
product −4.9e-08 against a tolerance of 1e-6) and the
judged maximum moved further from admission than any
predecessor's, because the component that opposes on
allowed inputs is the component that was restraining
the movement at the binding probe. By F1′′ measured
outright, the remaining honest reading is the one RFC
SI-30 §7.5 wrote in advance for this outcome:
composition measured on allowed inputs — at the mean,
at the worst case, and now as a construction — is not a
population-independent property of the movement pair.
That is a successful slice with a negative experimental
result, in the SI-21/SI-23/SI-25/SI-27/SI-29 pattern.

## VERIFY and unwind (§15 criterion 10)

- **Cycle 1** ADMITted and passed the §6.6 probe VERIFY
  exactly as in SI-19/SI-21/SI-23/SI-25/SI-27/SI-29
  (journaled `transformer-probe-verify`, measured-pass on
  both arms, echo 1.0).
- **Cycle 2** was REJECTed at the gate and never
  materialised, so there is no cycle-2 VERIFY and no
  VERIFY rollback — nothing landed to verify or roll
  back.
- **Unwind, honestly conditioned:** because cycle 2
  REJECTed, **no authored two-delta stack exists** —
  nothing is unwound, and no unwind is claimed. The
  bundle head after a decisive SI-31 run is `d0001`.
  SI-21's fixture-proven unwind remains the only unwind
  evidence. The transformer-squash precondition (an
  authored two-delta stack) therefore remains **unmet**
  after SI-31 as well.

## Acceptance criteria (RFC SI-30 §15) — status

1. ✅ **Recorded composition configuration, four named
   values** — exactly `"none"`, `"movement-composition"`,
   `"worst-input-composition"`,
   `"train-partition-projection"`; every other value
   refuses with the existing `ValueError` discipline at
   the constructor, the setter, the intent, and the
   `train_candidate` dispatch (asserted). Recorded
   selection asserted per cycle on both verdict paths:
   cycle 1's journaled note records `"none"`; cycle 2's
   records the full triple — signal 2,
   `target_anchoring = "base-anchored"`,
   `composition_config = "train-partition-projection"`,
   `projection_alpha = −0.8450027779470831` — with the
   construction's accounting beside the signal loss's.
   α is not independently settable (asserted: neither
   the constructor nor the setter takes such an
   argument, and no projection-fraction argument exists
   anywhere in the configuration); exactly one
   projection decisive run was performed (no sweep, no
   partial-removal variant).
2. ✅ **Phase 0 published before the decisive run is
   trusted** — the projection rehearsal for all three
   existing decisive candidates (coefficients, parallel
   shares, rehearsal held-out records and norms, all
   labelled diagnostics on already-judged candidates),
   the pre-registered F1′′/F2′′/F3′′ reading rules with
   F2′′'s clauses separated and F3′′'s thresholds
   carried from SI-29's rule (completeness boundary
   0.0013205629553992592), and the
   projection-correctness tolerance (1e-6) — all in the
   Phase 0 section above, ahead of the decisive numbers.
   Phase 0 also produced the representation finding the
   RFC had not priced (rank 16), recorded there with its
   gate consequence.
3. ✅ **Projection correctness, at the decisive head** —
   the decisive candidate's train-partition aggregate
   inner product with `m1` is *measured* (through the
   forward-path record, not the construction's algebra)
   at −4.8567197048765554e-08, within the Phase-0
   tolerance of 0.0, published beside the same record's
   held-out figure (−0.0009897609251914505, reported
   only); the construction's derivation is shown in the
   construction section above, and the folded-space
   identity Δ2′ = Δ2 − α·Δ1 is verified against the
   re-run SI-25 control in the test suite. No new
   differentiable path was introduced, so no new
   gradient check was owed: the as-built states this
   (gradient-check section above) and SI-25's check
   stands cited for the training step, not silently
   assumed.
4. ✅ **Probe-blindness of training and construction,
   asserted** — the coefficient's input set asserted as
   a set (exactly the train partition, via a spy on
   `projection_coefficient` during a real construction);
   the train partition's disjointness from both probe
   suites re-asserted under the new code path; a
   perturbed-probe-artifact bundle copy produces a
   byte-identical *projected* candidate.
5. ✅ **Coincidence at base** — signal 2 base-anchored +
   projection-configured at base reproduces the
   established at-base run exactly (blob
   `sha256:131cc242…45f4`, drift 0.16672966815531254,
   regression 0.2418281203135848), with the coefficient
   measured at α = 0.0 (numerator and denominator both
   0.0, recorded) and the construction measured as the
   identity on both partitions (aggregate inner products
   0.0; rank 8 retained).
6. ✅ **Determinism per cycle** — a full decisive run
   repeated from the seed reproduces both candidates'
   blobs/records (the projected candidate's blob
   included), the landed delta record, and the journaled
   gate measurements exactly; hyperparams equal except
   the labelled `wall_time_seconds`. The projected
   `train_delta` alone is likewise byte-identical across
   repeats (asserted) — the SVD factorisation is
   deterministic on this runtime.
7. ✅ **Gate honesty per cycle** — cycle 2's journaled
   checks 3/4 equal an independent `ProbeContext`
   evaluation of the same (projected) candidate against
   `d0001` (abs 1e-12). Cycle 2's verdict is **REJECT,
   as measured** — the safety suite fails on the
   constructed candidate's rank (16 > rank_cap 8, the
   unchanged gate pricing the construction's necessary
   representation) and the identity check fails on drift
   (0.4775387533009052 > tol 0.25); both are journaled
   and reported. No ADMIT was required: SI-31 succeeds
   by §7's claim — a projection-constructed candidate
   was genuinely authored against the landed head under
   the new configuration and genuinely judged, with its
   premise's falsification readings published either
   way.
8. ✅ **Cumulative accounting, diagnostics, the six-run
   table, and the falsification readings published** —
   the side-by-side accounting table, the full
   diagnostic set with candidate norm and ‖m2‖ beside
   every drift figure and the treatment's
   pre-projection norm beside its judged norm, the
   Phase-0 and decisive allowed-input composition
   records, the treatment's signal held-out record read
   under the pre-registered F3′′ rule, the six-run table
   with each figure under its own run's labels, and the
   F1′′/F2′′/F3′′ readings with F2′′'s clauses reported
   separately (F1′′ measured; F2′′ not measured — its
   transfer clause fails, its maximum clause's fact
   holds beside it; F3′′ measured under the
   pre-registered rule, by its completeness clause) —
   all above, measurements and derivations labelled,
   worst probes named, nothing in the diagnostic set
   reported "not evaluated" that was computable.
9. ✅ **First-cycle and all-control reproduction** —
   cycle 1 reproduces SI-19/SI-21/SI-23/SI-25/SI-27/SI-29
   exactly (blob `sha256:2761e8ff…6ca9e`, drift
   0.17189218290150166, regression 0.1657181391492486);
   the re-run `"none"` control reproduces SI-25's
   decisive candidate exactly (blob
   `sha256:646d1e8b…d675`, drift 0.3329036422073841);
   the re-run `"movement-composition"` control
   reproduces SI-27's decisive candidate exactly (blob
   `sha256:1bb277ab…b637`, drift 0.32509003952145576);
   the re-run `"worst-input-composition"` control
   reproduces SI-29's decisive candidate exactly (blob
   `sha256:11cab02e…b8db5`, drift 0.31215621903538704) —
   the configuration machinery proven
   behaviour-preserving for all three existing semantics
   before the new configuration's results are trusted.
10. ✅ **VERIFY, unwind, and untouched scope** — cycle-1
    VERIFY as in SI-19; unwind honestly conditioned (no
    authored stack exists, none unwound, none claimed —
    see VERIFY and unwind). Exactly two authored cycles
    attempted in the decisive run and exactly one
    projection decisive run performed;
    `max_stack_depth` 8, `identity_tol` 0.25,
    `capability_eps` 0.25, `max_targets_per_delta` 1,
    `allow_additive` false, `rank_cap` 8 and every other
    policy default unchanged (asserted); both signals'
    ids, streams, corrections, and partitions unchanged
    (asserted); both anchorings' and all three prior
    composition configurations' definitions unchanged
    (asserted); `si squash` still fails closed on the
    resulting transformer stack (asserted; SI-12's suite
    green); the full existing suite is green (below);
    the demo path is unchanged; `si serve` still returns
    501 for every transformer bundle (existing suite); no
    CLI wiring added. Per-cycle `train_delta` wall times are recorded in the
    cycles' hyperparams; total decisive two-cycle wall time
    is on the order of seconds on this VM. All on this VM,
    CPU only.

Full suite on the author's own run: **366 passed,
2 skipped** in 399.94s — the SI-29 baseline (353 passed,
2 skipped) plus this slice's 13 new tests, zero
regressions. Focused run:
`tests/test_train_partition_projection.py`, 13 passed in
36.73s.

## What is still NOT claimed

- **No authored stack deeper than one.** Cycle 2
  REJECTed under projection construction as it did under
  both anchorings unpenalised and under both composition
  penalties; the deepest landed authored stack on record
  remains depth 1 (`d0001`). No authored two-delta stack
  exists, and the transformer-squash precondition remains
  unmet.
- **Train-partition projection is not shown to work — at
  the judged maximum, it is shown to fail worse than any
  predecessor.** On allowed inputs the construction is
  shown to work exactly (the train-partition aggregate
  inner product is zero as constructed and as measured).
  On the one figure the gate prices, the treatment
  (0.47754) worsens on every prior run (0.34378 /
  0.29634 / 0.33290 / 0.32509 / 0.31216), and at the
  binding probe its movements align more than any
  predecessor's (+0.14236). One decisive configuration
  was measured; no claim is made about projections at
  other heads, other depths, or for signal 1 as a
  decisive matter, and no partial-removal variant was
  run or is reported as anything but a different
  candidate for a future RFC.
- **No α conclusion beyond the measured coefficient.**
  α = −0.8450027779470831 is this run's measured
  coefficient for this intermediate at this head; it is
  not a tuned value, not a recommended value, and no
  other coefficient definition was run.
- **The rank-16 representation is a measured property of
  the §7.2 construction, not a widening of the
  project's policy.** `rank_cap` stays 8; the gate
  priced the constructed candidate against it and
  refused on that line beside the identity line. A
  future RFC that wants projected candidates judged on
  identity alone must argue the representation question
  itself — this slice does not argue it, it reports it.
- **No third cycle, no continual learning, no
  mixed-configuration stack claim.** Exactly two
  authored cycles were attempted in the decisive run, on
  constructed signals, on a constructed model. The
  controls are separate runs, labelled controls; the
  Phase-0 rehearsals are diagnostics on already-judged
  candidates, labelled as such, and the SI-25-based
  rehearsal's near-identity to the decisive construction
  (same intermediate, same α to the printed digits) is
  stated in Phase 0, not discovered after the fact.
- **No Glimmer anything.** No Glimmer forward path,
  probes, training, or authorship; §7's projection is a
  miniature mechanism and claims nothing at Glimmer
  scale; the Glimmer seed is untouched and its deltas
  still escalate with checks 3/4 "not evaluated".
- **No widening.** Two signals, two anchorings, four
  composition configurations, one target, one bundle
  kind; no additive or multi-target authorship; no
  policy default changed; no tolerance moved; both
  signals are byte-for-byte as SI-19/SI-23 defined them;
  the default anchoring and the default composition
  configuration are unchanged. The decisive candidate's
  rank (16) is a property of its construction, recorded
  on its record — not a change to any signal's pinned
  rank (8) or to the training step's factors.
- **No serving, no CLI wiring.** Transformer bundles
  keep returning 501.
- **Diagnostics are not verdicts.** The diagnostic set
  above attributes the gate's REJECT after the fact; it
  is not a gate check, it influenced no verdict, and
  none of it was a prediction (§17 items 12/14) — the
  Phase-0 rehearsal diagnostics that pointed the same
  way are labelled diagnostics on already-judged
  candidates, and the decisive figures stand on the
  decisive run's own measurements.
- **Training numbers are training numbers.** Every loss,
  coefficient, and construction value above is labelled
  with its signal id, partition, head, target source,
  anchoring, and composition configuration; none of them
  is a gate or probe result (§17 item 1). Allowed-input
  figures are never presented as probe figures or vice
  versa (§17 item 18), and a worst-case figure always
  names its population (§17 item 20). The
  pre-projection intermediate's figures are labelled an
  intermediate's wherever they appear (§17 item 22).
- **Recorded selection is not a choice the mind made**
  (§17 item 13). The driver configured each cycle's
  signal, anchoring, and composition configuration; the
  journal shows all three.

## Terminology

Machine learning infrastructure. This document contains
no "custom LLM software" claims; the miniature is a
constructed, deterministically seeded, not trained
machine-learning test instrument, its authored deltas are
trained on constructed signals, and its second signal's
projection-constructed second-cycle delta was judged —
and rejected — on the unchanged gate's safety suite and
on cumulative identity drift, with the projection
measured doing exactly what it was defined to do on the
train partition it was allowed to see (aggregate inner
product −4.9e-08, within tolerance of zero) and the
gate's binding probe measured moving the other way,
further than under any predecessor.
