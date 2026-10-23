# SI-27 — Movement-composition-aware training for the miniature (as-built)

> **Scope:** RFC SI-26 §14 (the SI-27 outline) and §15 (the ten
> acceptance criteria — each addressed below with its
> measurements). Composition-awareness is now a recorded per-cycle
> configuration beside recorded signal selection and recorded
> target anchoring, and signal 2 was trained at the landed head
> `d0001`, base-anchored, with exactly one added composition-penalty
> loss term (λ = 1.0, fixed in advance) computed only over signal
> 2's own train partition — exactly one variable changed against
> SI-25's decisive run. **Cycle-2 ADMIT was explicitly NOT required
> (RFC §15.7) — the gate's verdict on the decisive cycle 2 is
> REJECT, reported below as measured.** The diagnostics attribute
> it, and the attribution is now three-sided: the penalty *works
> where it is allowed to look* — on signal 2's train partition the
> candidate's movement composition improves decisively (aggregate
> inner product −1.687 → −2.688 against the re-run control) and the
> penalty term itself moves from +0.000174 at initialisation to
> −0.002625 at the end of training; in aggregate on the probes it
> also improves (−0.2126 → −0.3271 against SI-25's control); and it
> *still fails where the gate binds* — at `tiny-id-0` the two
> movements align slightly *worse* than the control's
> (+0.03801222744105431 against +0.0338873784299374), the composed
> maximum is 0.32509003952145576 > tol 0.25, and the falsification
> readings of RFC §7.6 come out F1 measured (at the worst probe),
> F2 measured, F3 not measured as defined — with a partial
> destination trade measured beside it (the penalised candidate's
> signal held-out residual is ~50× the control's, though the
> destination is still approached: held-out loss falls ~5.3×,
> 16/16 examples closer). One further §7.5 finding stands beside
> the verdict: the re-run control's movements *already oppose* on
> the train partition, so in aggregate the penalty was pushing on
> an already-pushed direction — and it pushed the binding probe
> the wrong way regardless.

## What was built

- **`si/tiny_training.py`** — the SI-27 additions. (a) The
  composition configuration: `COMPOSITION_CONFIGS` (exactly
  `"none"`, `"movement-composition"`), `DEFAULT_COMPOSITION_CONFIG`
  (`"none"`), `MOVEMENT_COMPOSITION`, and
  `validate_composition_config(...)` (any other value refuses with
  the existing `ValueError` discipline, mirroring
  `validate_target_anchoring`). λ lives in exactly one place:
  `COMPOSITION_LAMBDA = 1.0`, part of the
  `"movement-composition"` configuration's definition — it is
  never read from an intent or any other caller input, so no
  per-cycle λ dial exists. (b) The penalty machinery:
  `existing_movements(...)` (`m1` per input — landed-head minus
  base last-position logits through the existing forwards,
  constants during optimisation), `composition_penalty(...)`
  (the term's analytic value), `analytic_gradients_penalised(...)`
  (signal gradient plus the penalty's fixed-matrix gradient
  through the same readout fold), `penalised_factor_loss(...)`
  (the same loss through the float32 shared fold and the forwards
  — the finite-difference check's target), and
  `movement_composition_record(...)` (the allowed-input alignment
  diagnostic). (c) `train_candidate` dispatches on the intent's
  composition configuration: under `"movement-composition"` it
  prepares `m1` on the signal's train partition only (base
  assembly is SI-25's — under base-anchoring it *is* the target
  weights; under head-relative anchoring it is assembled for the
  movement role alone, fail-closed with no fallback), trains
  against `signal_MSE + λ · penalty`, and records the penalty's
  values; under `"none"` it runs the SI-25 path byte-for-byte —
  no base assembly for the penalty, no penalty quantity computed,
  penalty hyperparams recorded as null (not measured). The
  non-improvement refusal still judges the *signal* held-out loss
  alone under both configurations.
- **`si/tiny_transformer.py`** — recorded composition
  configuration on `TinyTransformerMind`: a constructor-level
  `composition_config` (default `"none"`), settable between
  cycles by the explicit driver call `set_composition_config(...)`,
  validated against exactly the two named configurations;
  `propose` carries the configured value on the intent beside the
  configured signal and anchoring.
- **`si/gate.py`** — `Intent` gains one field,
  `composition_config` (default `"none"`), carried in `to_dict()`
  so the journaled intent shows the configuration the driver
  chose. No gate check, verdict mapping, tolerance, cap, or
  policy default changed.
- **`tests/test_movement_composition.py`** (new, 12 tests) and
  this doc. No probe-suite change, no serve change (501 for every
  transformer bundle), no CLI wiring, the Glimmer seed untouched,
  the demo path unchanged, both signals' ids, streams,
  corrections, and partitions unchanged, both anchorings'
  definitions unchanged.

There is no autonomous, learned, depth-driven, or verdict-driven
composition selection anywhere in the slice — the mind chooses
neither its signal, its anchoring, nor its composition
configuration; the driver configures all three, and the journal
shows all three. Exactly one penalised decisive run was
performed, at exactly λ = 1.0; no sweep of any kind was run.

## The penalty's two terms, separately (§7.2(iii))

Decisive cycle 2 (signal `tiny-predecessor-lean-001`, head
`d0001`, target source `base`, anchoring `base-anchored`,
composition `movement-composition`, λ = 1.0). Training numbers —
labelled with signal, head, target source, anchoring, and
composition configuration wherever they appear; the penalty's
train values are optimised, its held-out values are reported
only and never entered the loss:

| Term (train partition, optimised) | At initialisation | End of training |
|---|---|---|
| Signal MSE | 0.0042998498036725735 | 0.0005191352673009495 |
| Penalty (mean `p(x)`) | +0.00017392083908663979 | **−0.0026247611759719963** |
| Total (signal + λ·penalty) | 0.004473770642759213 | **−0.0021056259086710467** |

| Term (held-out partition, reported only) | At initialisation | End of training |
|---|---|---|
| Penalty (mean `p(x)`) | +0.0001744748535695612 | −0.002628909254542555 |

Both §7.2(iii) properties are measured, not hidden: the penalty
is signed and the total loss ends **negative** (driven by the
penalty term), and the penalty *fights* the signal loss — the
penalised signal train residual (0.0005191352673009495) is ~55×
the control's (9.412649442227854e-06). The destination record is
in the decisive-run table below and in the F3 reading.

**Coincidence configuration (§15.2).** Configured with signal 2,
base-anchoring, and the movement-composition configuration, a
first cycle at base records the penalty at initialisation and at
the end of training, on both partitions: **identically zero**
(measured — `m1` is the zero vector at base), the totals equal
the signal losses, and the run reproduces SI-25's at-base run
**exactly** — candidate blob
`sha256:131cc242728024b4f833bc78ea627825f1e8616972fc2670603673e1814645f4`,
identity drift 0.16672966815531254, capability regression
0.2418281203135848, ADMIT → `d0001`, §6.6 VERIFY pass. The
penalty machinery is therefore proven correct where it must
vanish, before the decisive cycle is trusted. These are at-base
facts; they are not the slice's decisive claim.

## Gradient check for the penalised loss (§7.3, §15.3)

Central finite differences of signal 2's **base-anchored,
composition-penalised** loss — base-derived targets, head hidden
states and readout, landed `m1` on the train partition — through
`penalised_factor_loss` (the float32 shared fold plus the
existing forwards), at signal 2's pinned initial factors over an
8-example subset, `eps = 0.01`, EVERY one of the 512 factor
entries:

| Configuration | Worst absolute disagreement | Tolerance |
|---|---|---|
| Signal 2 base-anchored + penalised at base (new code path) | **5.5249277028738815e-08** | 1e-5 (SI-19's pin, unchanged) |
| Signal 2 base-anchored + penalised at head `d0001` (the decisive configuration) | **5.149878699880166e-08** | 1e-5 |

Both pass with ~200× headroom. The at-base figure is identical
to SI-25's at-base figure, as it must be: at base the penalised
loss coincides with the unpenalised loss — and the coincidence is
measured under the new code path (penalty exactly 0.0 at the
pinned factors; penalised gradients array-equal to the
unpenalised gradients), not asserted from the derivation. No new
backward machinery was needed (§7.3: the penalty is linear in
the folded update; its gradient is the `m1`-weighted combination
of the same cached hidden states), and no tolerance was loosened.

## Probe-blindness of training (§15.4)

Three assertions, all in `tests/test_movement_composition.py`:

- A spy on `analytic_gradients_penalised` during a real
  penalised `train_candidate` run captures every call's example
  set and `m1` key set: each is **exactly signal 2's train
  partition** (as a set) — the penalty's input set is the train
  partition, asserted as a set, on the code path that trained
  the decisive candidate (which the spy run reproduces:
  blob `sha256:1bb277ab…b637`).
- The train partition's disjointness from both probe suites
  (identity and capability) and from the signal's held-out
  partition is re-asserted under the new code path.
- A copy of the bundle whose `probe_suite.json` was perturbed on
  disk (probe inputs and expected logits replaced) produces a
  **byte-identical candidate** through `train_candidate` called
  directly with the pre-perturbation verified manifest — the
  training path never opens the probe artifact, so no probe
  input, probe logit, or probe-derived movement can reach the
  loss, the penalty, factor initialisation, or selection.
  (`load_bundle` itself refuses the perturbed copy — fail-closed
  tamper evidence, a separate, pre-existing guarantee.)

## The allowed-input composition record (§7.5, §15.5)

For the re-run **control** candidate (signal 2, base-anchored,
composition `"none"`, at `d0001` — blob `sha256:646d1e8b…d675`)
and for the **treatment** candidate (same labels, composition
`"movement-composition"` — blob `sha256:1bb277ab…b637`): the
alignment diagnostic under the §7.2 movement definitions,
computed on signal 2's train partition and on its held-out
partition. Per-input rows (32 train / 16 held-out per candidate)
are asserted in the test; the aggregates are:

| Record (partition labelled) | Aggregate inner product | Sign agreement | Mean `p` (= penalty value) | ‖m1‖ | ‖m2‖ |
|---|---|---|---|---|---|
| Control, train partition | −1.6873837019991191 | 0.267578125 (274/1024) | −0.0016478356464835148 | 1.4131161397867473 | 1.6765305214306858 |
| Control, held-out partition | −0.845596374259425 | 0.26953125 (138/512) | −0.0016515554184754395 | 0.9997655382584564 | 1.1850839719703719 |
| Treatment, train partition | **−2.687755446349435** | 0.2763671875 (283/1024) | **−0.00262476117807562** | 1.4131161397867473 | 2.2303160062528544 |
| Treatment, held-out partition | **−1.3460015097625617** | 0.28125 (144/512) | **−0.0026289091987550034** | 0.9997655382584564 | 1.57723836245857 |

Two readings, stated plainly:

- **The §7.5 check on the design premise:** the control's
  movements **already oppose** on the training inputs (negative
  aggregate inner product and negative mean `p` on both allowed
  partitions) while aligning at the binding probe (SI-25:
  +0.0338873784299374 at `tiny-id-0`). The penalty was therefore
  pushing on an already-pushed direction in aggregate on the
  inputs it may see — the slice says so, as §7.5 requires. What
  the penalty still changed there is magnitude: it drove the
  train aggregate from −1.687 to −2.688 and the mean penalty
  value from the control's −0.001648 to −0.002625, at the cost of
  a substantially larger second movement (‖m2‖ 1.677 → 2.230 on
  the train partition).
- **Cross-check:** the record's forward-path mean `p` for the
  treatment equals the training run's analytic penalty values
  (train −0.0026247611759719963, held-out −0.002628909254542555)
  to ~2e-12 — two independent paths to the same numbers. `m1`'s
  norms are identical under both candidates on both partitions,
  as they must be: `m1` is a property of the landed head, not of
  the candidate.

No figure above is a probe figure, and no probe figure below is
an allowed-input figure (§17.18).

## The decisive two-cycle run (measured on this VM, this slice's runs)

Driver shape: **one reused `TinyTransformerMind`** driven
through `si.run.run_cycles` — one `run_cycles` call for cycle 1
(signal 1, default anchoring, default composition
configuration), then the driver explicitly reconfigures the same
mind to signal 2 (`set_training_signal`), to base-anchored
targets (`set_target_anchoring`), **and** to the
movement-composition configuration (`set_composition_config`) —
three recorded selections — then a second `run_cycles` call for
cycle 2. Exactly two authored cycles were attempted, and exactly
one penalised decisive run was performed. This is SI-25's
decisive run with exactly one configuration changed: cycle 2's
composition configuration.

| | Cycle 1 | Cycle 2 (treatment) |
|---|---|---|
| Signal (recorded selection) | `tiny-successor-lean-001` (default) | `tiny-predecessor-lean-001` (explicit) |
| Anchoring (recorded selection) | `weights-under-training` (default) | **`base-anchored` (explicit)** |
| Composition (recorded selection) | `none` (default), λ 0.0, penalty fields null | **`movement-composition` (explicit), λ 1.0** |
| Head trained against | `base` | `d0001` |
| Target-source head | `base` | **`base`** |
| Stack depth at training | 0 | 1 |
| Candidate blob | `sha256:2761e8ff…6ca9e` (2,360 B) | `sha256:1bb277ab…b637` (2,360 B) |
| Candidate norm | 0.23555380990005192 | 0.2969693161862222 |
| Signal train loss, initial → final | 0.0028961365437885785 → 1.824004756500067e-06 | 0.0042998498036725735 → 0.0005191352673009495 |
| Penalty (train), initial → final | not measured (configuration `"none"`) | +0.00017392083908663979 → −0.0026247611759719963 |
| Pre-training held-out signal loss | 0.00140625 | **0.002757805049077469** (genuine measurement, as in SI-25) |
| Held-out signal loss after training | 3.2118398935406443e-06 | 0.0005227812955486338 (~5.3× lower) |
| Held-out examples improved | 16/16 | 16/16 |
| Norm-guard activations | 0 | 0 |
| `train_delta` wall time | 1.00 s | 2.01 s |
| Gate verdict | **ADMIT** → `d0001`, §6.6 VERIFY pass | **REJECT** (identity), nothing landed |
| Blob-store growth | +2,360 B | +0 B (candidate never written) |

Full candidate blob hashes: cycle 1
`sha256:2761e8ffc139ce3e107c11da035e8a73c20792945a494f4e493dccf418b6ca9e`;
cycle 2
`sha256:1bb277ab40b9aa53a6218d971e46f29b261ba2bb5f7932ffb98e3951f372b637`.
Total two-cycle wall time **3.39 s** (both `run_cycles` calls,
gate included, on this VM). Cycle 1 reproduces
SI-19/SI-21/SI-23/SI-25 **exactly** (criterion 9). The re-run
unpenalised control inside this slice reproduces SI-25's
decisive cycle-2 candidate **exactly** (blob
`sha256:646d1e8ba3978d858d4173d561f8799bce37350f88ff2de65f40c223d9e4d675`,
drift 0.3329036422073841, regression 0.10096919978968799, under
composition `"none"` with λ 0.0 and null penalty fields), so the
treatment's difference is attributable to the penalty term and
not to the refactor that introduced it. Cycle 2's candidate is
genuinely new (a different blob from the coincidence candidate,
SI-25's control candidate, and SI-23's), and as a
destination-relative improvement it still succeeds — but far
less completely than the control's (see the F3 reading). The
gate does not judge destination-relative improvement; it judges
the composed head against the seed.

## Cumulative vs incremental accounting (§15.8) — side by side

| Measurement | Cycle 1 | Cycle 2 (treatment) | Accounting |
|---|---|---|---|
| Identity drift (gate, candidate overlay) | 0.17189218290150166 (pass, tol 0.25) | **0.32509003952145576 (FAIL, tol 0.25)** | **seed-absolute / cumulative** — drift of the would-be head vs the seed's recorded logits |
| Capability regression (gate) | 0.1657181391492486 (pass, eps 0.25) | 0.14536726917140186 (pass, eps 0.25) | **head-incremental** — charged only since the pre-cycle head |
| Echo accuracy | 1.0 (16/16, floor 0.9375) | 1.0 (16/16, floor 0.9375) | absolute, vs the recorded floor |
| Landed head's cumulative identity drift after the cycle | 0.17189218290150166 (§6.6 VERIFY, measured-pass) | 0.17189218290150166 (unchanged — re-measured after the REJECT) | seed-absolute, of the head that actually landed |

No incremental figure above is presented as cumulative or vice
versa, no base-anchored figure as a head-relative one, and no
penalised figure as an unpenalised one.

## The diagnostics (published whatever the verdict)

All diagnostics below are computed against head `d0001` with the
decisive cycle-2 candidate overlaid. They are published to
*attribute* the gate's verdict after the gate has spoken; they
are not gate checks and influenced no verdict (RFC §17.12).
Per-probe drift values are measurements (max absolute logit
coordinate per probe); the incremental movement column is a
**derivation** from two measured forwards (composed forward
minus the `d0001` forward), labelled as such.

**(i) Per-probe drift contributions (treatment):**

| Probe | `d0001` drift (measured) | Signal-2 penalised incremental movement (derivation) | Composed drift (measured) |
|---|---|---|---|
| tiny-id-0 | 0.17189218290150166 | 0.15392476320266724 | **0.32509003952145576** |
| tiny-id-1 | 0.15088725090026855 | 0.0669945478439331 | 0.09391975402832031 |
| tiny-id-2 | 0.11858986946754158 | 0.1752924919128418 | 0.1488508302718401 |
| tiny-id-3 | 0.14782869815826416 | 0.22549264132976532 | 0.14755446463823318 |
| tiny-id-4 | 0.11720322235487401 | 0.1778034269809723 | 0.10617947578430176 |
| tiny-id-5 | 0.09891846403479576 | 0.12068814411759377 | 0.1450248844921589 |
| tiny-id-6 | 0.12363102938979864 | 0.18399151042103767 | 0.15063166618347168 |
| tiny-id-7 | 0.08795571327209473 | 0.15689591597765684 | 0.18760260567069054 |

Worst probes, named: `d0001`'s own drift — **tiny-id-0**
(0.17189218290150166); the composed drift — **tiny-id-0**
(0.32509003952145576, equal to the gate's measured cycle-2
drift). The incremental movement's largest probe is tiny-id-3
(0.22549264132976532) — the penalised update is the largest
second update on record and its movement is *least* restrained
exactly where SI-23's control composed worst. At tiny-id-0 the
composed drift is again, to the printed digits, the sum of the
two contributions (0.17189218290150166 + 0.15392476320266724 ≈
0.3251) — the aligned extreme, at that probe, for the third
decisive run in a row at the probe where cycle 1 moved most.

**(ii) Alignment diagnostic (derivation):** `m1` = (`d0001`
forward − seed expected) and `m2` = (composed forward −
`d0001` forward), per identity probe:

- **In aggregate the opposition deepens.** Over all 8 identity
  probes (256 coordinates): inner product
  **−0.3271477846470207** (more opposing than SI-25's
  −0.21256230517288188; SI-23's was +0.07637087369869666,
  aligned); sign agreement **0.390625** (100/256; SI-25:
  0.37109375; SI-23: 0.54296875) — published beside the
  magnitude-weighted figure, never alone (§17.19);
  movement-vector norms ‖m1‖ = 0.7119678353441855 (unchanged —
  cycle 1's movement), ‖m2‖ = 0.9688074699782941 (SI-25:
  0.7816722548913912 — the penalised movement is larger).
- **At the binding probe it fails, slightly worse.** At
  `tiny-id-0`: inner product **+0.03801222744105431** (positive
  — aligned; SI-25 control: +0.0338873784299374), sign agreement
  0.4375 (14/32). The penalty improved composition on the
  train partition, on the held-out partition, and in aggregate
  on the probes — and at the one probe where the gate binds,
  the alignment *increased*.

**(iii) Folded updates (measured):** Frobenius inner product of
the two folded `lm_head.weight` updates
**−0.02013451284871203** (opposing in weight space, more
strongly than SI-25's −0.011457914936582613; SI-23's was
+0.004356018453758104); norms ‖Δ1‖_F = 0.23555380990005192,
‖Δ2‖_F = 0.2969693161862222 (Δ2's norm equals its candidate
norm, as it must for a single-target LoRA). The penalised
update is the largest second update on record — opposition was
bought partly with magnitude.

**(iv) Composed capability (measured):** head-incremental
regression **0.14536726917140186** ≤ eps 0.25 (pass — worse
than SI-25's 0.10096919978968799, better than SI-23's
0.15023326850496233); echo accuracy **1.0 (16/16)** vs the
recorded floor 0.9375.

**(v) The three-run control/treatment table (§14 item 8,
§17.16).** Same signal, same head, same target, same
hyperparameters — differing only in anchoring (SI-23 vs SI-25)
and then only in composition configuration (SI-25 vs SI-27).
Each figure under its own run's labels:

| Per-probe composed drift | tiny-id-0 | tiny-id-1 | tiny-id-2 | tiny-id-3 | tiny-id-4 | tiny-id-5 | tiny-id-6 | tiny-id-7 | Max (worst probe) |
|---|---|---|---|---|---|---|---|---|---|
| SI-23 — signal 2, weights-under-training, composition n/a (pre-SI-27) | 0.2092923242598772 | 0.15134692192077637 | 0.14906740188598633 | **0.2963416576385498** | 0.23394920490682125 | 0.14004023000597954 | 0.1332004750147462 | 0.2018739078193903 | 0.2963416576385498 (tiny-id-3) |
| SI-25 — signal 2, base-anchored, composition `"none"` | **0.3329036422073841** | 0.09286642074584961 | 0.14936351776123047 | 0.1489148810505867 | 0.14728415571153164 | 0.11245626583695412 | 0.14491976890712976 | 0.14189285784959793 | 0.3329036422073841 (tiny-id-0) |
| SI-27 — signal 2, base-anchored, composition `"movement-composition"`, λ 1.0 | **0.32509003952145576** | 0.09391975402832031 | 0.1488508302718401 | 0.14755446463823318 | 0.10617947578430176 | 0.1450248844921589 | 0.15063166618347168 | 0.18760260567069054 | 0.32509003952145576 (tiny-id-0) |

| Gate / candidate figure | SI-23 (head-relative) | SI-25 (base-anchored, `"none"`) | SI-27 (base-anchored, penalised) |
|---|---|---|---|
| Candidate blob | `sha256:184cc7f0…6e6cf7` | `sha256:646d1e8b…d675` | `sha256:1bb277ab…b637` |
| Candidate norm | 0.21317186599953494 | 0.2515524481203623 | 0.2969693161862222 |
| Identity drift (cumulative) | 0.2963416576385498 — FAIL | 0.3329036422073841 — FAIL | 0.32509003952145576 — FAIL |
| Capability regression (head-incremental) | 0.15023326850496233 — pass | 0.10096919978968799 — pass | 0.14536726917140186 — pass |
| Echo accuracy | 1.0 — pass | 1.0 — pass | 1.0 — pass |
| Aggregate probe inner product | +0.07637087369869666 | −0.21256230517288188 | −0.3271477846470207 |
| Inner product at own worst probe | (SI-23 doc: aligned at tiny-id-3) | +0.0338873784299374 (tiny-id-0) | +0.03801222744105431 (tiny-id-0) |
| Verdict | REJECT | REJECT | REJECT |

Per probe, the treatment's composed drift improves on SI-25's
control at 4 of 8 probes (tiny-id-0 marginally, tiny-id-2,
tiny-id-3, tiny-id-4 substantially) and worsens at 4
(tiny-id-1 marginally, tiny-id-5, tiny-id-6, tiny-id-7
substantially). The maximum improves slightly (0.33290 →
0.32509) and stays far above tol 0.25. The treatment's
incremental movement exceeds cycle 1's remaining headroom
(0.07810781709849834) at **7 of 8** probes (SI-25: 7 of 8;
SI-23: 5 of 8). No diagnostic was uncomputable; nothing in
this set is reported "not evaluated".

**(vi) The falsification readings (RFC §7.6), each stated as
measured:**

- **F1 — the penalty does not transfer: MEASURED, at the
  worst probe.** The penalised candidate's training-input
  composition improves (train aggregate inner product
  −1.6873837019991191 → −2.687755446349435; penalty term
  +0.000174 → −0.002625) while its probe composition *at the
  worst probe* is worse than the control's (+0.0338873784299374
  → +0.03801222744105431 at `tiny-id-0`). The train-to-probe
  composition gap swallows the term where the gate binds, as
  it swallowed the destination argument at `tiny-id-0` in
  SI-25. (In aggregate on the probes the composition *did*
  improve — that is F2's measurement, not a rescue of F1's:
  F1 is stated at the worst probe, and there it is measured.)
  Falsified premise, in the RFC's words: that composition
  measured on allowed inputs shapes composition on judged
  inputs, through this readout, at this scale — at the judged
  input that matters most, it did not.
- **F2 — the penalty transfers but the maximum does not move:
  MEASURED.** Probe composition improves in aggregate (inner
  product −0.3271477846470207, below the control's
  −0.21256230517288188) yet the composed maximum stays above
  tol 0.25 (0.32509003952145576) — opposition purchased in
  aggregate and on allowed inputs, alignment surviving where
  the first movement was largest. This is the second measured
  instance of §6.2's arithmetic (SI-25 was the first), and by
  the RFC's own statement it settles the premise: aggregate
  composition is not the binding variable.
- **F3 — the destination is traded away: NOT measured as
  defined; a partial trade IS measured and reported.** The
  penalised candidate's signal held-out record (signal
  `tiny-predecessor-lean-001`, head `d0001`, target source
  `base`, anchoring `base-anchored`, composition
  `"movement-composition"`): pre-training 0.002757805049077469
  → after training 0.0005227812955486338, 16/16 examples
  closer. The destination is still approached — opposition was
  NOT achieved by abandoning the lean the cycle exists to
  author, so F3's falsification does not fire and this
  candidate is not an instance of the RFC's warning sentence.
  But the approach is far less complete than the control's
  (~5.3× against ~264×; residual ~50× the control's
  1.043608588752459e-05; signal train residual ~55× the
  control's): the penalty bought its opposition partly at the
  destination's expense, and any future reading of this run
  that quotes the composition improvement without the
  destination cost beside it is blending the record.

**Attribution:** all three REJECTs in the table are
movement-composition failures, now of three distinguishable
characters. SI-23's second update aligned in aggregate and at
its worst probe. SI-25's opposed in aggregate and failed at
the binding probe. SI-27's was *trained* to oppose on the
inputs training may see, opposed measurably harder there and
in aggregate on the probes, and failed at the same binding
probe with slightly *increased* alignment there. The variable
SI-27 changed — composition on allowed inputs — is now
measured to be shapeable on those inputs and not binding on
the judged maximum. That is a successful slice with a negative
experimental result, in the SI-21/SI-23/SI-25 pattern.

## VERIFY and unwind (§15.10)

- **Cycle 1** ADMITted and passed the §6.6 probe VERIFY exactly
  as in SI-19/SI-21/SI-23/SI-25 (journaled
  `transformer-probe-verify`, measured-pass on both arms, echo
  1.0, baseline labelled "pre-cycle head (caller-captured)").
- **Cycle 2** was REJECTed at the gate and never materialised,
  so there is no cycle-2 VERIFY and no VERIFY rollback —
  nothing landed to verify or roll back. The would-be VERIFY
  rollback target, had it landed and failed, is `d0001` by the
  miniature branch's existing construction; that path is
  unchanged and remains covered by SI-17's injected-failure
  test.
- **Unwind, honestly conditioned (§14 item 10):** because
  cycle 2 REJECTed, **no authored two-delta stack exists** —
  nothing is unwound, and no unwind is claimed. The bundle head
  after a decisive SI-27 run is `d0001`. SI-21's fixture-proven
  unwind (authored `d0001` + a benign hand-built fixture
  `d0002`, labelled a fixture) remains the only unwind
  evidence, exactly as SI-21 left it. The transformer-squash
  precondition (an authored two-delta stack) therefore remains
  **unmet** after SI-27 as well.

## Acceptance criteria (RFC SI-26 §15) — status

1. ✅ **Recorded composition configuration** — the
   configuration serves exactly the two named values; every
   other value refuses with the existing `ValueError`
   discipline at the constructor, the setter, the intent, and
   the `train_candidate` dispatch (asserted). Recorded
   selection is asserted per cycle on both verdict paths:
   cycle 1's journaled `training-hyperparams` (ADMIT, under
   `d0001`) records `composition_config = "none"`,
   `composition_lambda = 0.0`, signal 1,
   `target_anchoring = "weights-under-training"`; cycle 2's
   (REJECT, under its self-edit) records the full triple —
   signal 2, `target_anchoring = "base-anchored"`,
   `composition_config = "movement-composition"`,
   `composition_lambda = 1.0` — with the penalty's values
   beside the signal loss's. No autonomous selection exists;
   λ is not independently settable (asserted: the constructor
   and the setter take no λ argument); exactly one penalised
   decisive run was performed (no sweep).
2. ✅ **Coincidence at base** — signal 2 base-anchored +
   penalised at base reproduces SI-25's at-base run exactly
   (blob `sha256:131cc242…45f4`, drift 0.16672966815531254,
   regression 0.2418281203135848), with the penalty measured
   at initialisation and at the end of training on both
   partitions (identically zero); signal 1's first cycle
   under the default composition configuration reproduces
   `d0001` exactly (criterion 9, asserted standalone in the
   decisive run and in the control run).
3. ✅ **Gradient check for the penalised loss, at the
   decisive head** — re-run at `d0001` (worst
   5.149878699880166e-08) AND at base under the new code path
   (worst 5.5249277028738815e-08), over EVERY factor entry,
   tolerance 1e-5 unchanged, both published above.
4. ✅ **Probe-blindness of training, asserted** — the
   penalty's input set is asserted as a set (exactly the
   train partition, via a spy on the penalised gradient path
   during a real training run); the train partition's
   disjointness from both probe suites is re-asserted under
   the new code path; a perturbed-probe-artifact bundle copy
   produces a byte-identical candidate (see above).
5. ✅ **The allowed-input composition record, published** —
   for the re-run control and for the treatment: per-input
   and aggregate inner products, sign agreements (never
   alone — always beside the magnitude-weighted figure,
   §17.19), movement norms, each partition labelled, plus the
   penalty term's value at initialisation and at the end of
   training on each partition (train: optimised; held-out:
   reported only). Published whatever it shows — including
   that the control already opposes on the train partition.
6. ✅ **Determinism per cycle** — a full decisive run repeated
   from the seed reproduces both candidates' blobs/records,
   the landed delta record, and the journaled gate
   measurements exactly; hyperparams equal except the
   labelled `wall_time_seconds`. The penalised `train_delta`
   alone is likewise byte-identical across repeats
   (asserted).
7. ✅ **Gate honesty per cycle** — cycle 2's journaled checks
   3/4 equal an independent `ProbeContext` evaluation of the
   same candidate against `d0001` (abs 1e-12); cycle 1's
   equality is covered by the SI-19/SI-21/SI-23/SI-25 suites
   this run keeps green. Cycle 2's verdict is **REJECT, as
   measured**. No ADMIT was required: SI-27 succeeds by §7's
   claim — a composition-penalised candidate was genuinely
   authored against the landed head under the new
   configuration and genuinely judged.
8. ✅ **Cumulative accounting, diagnostics, the three-run
   table, and the falsification readings published** — the
   side-by-side accounting table, the full diagnostic set,
   the allowed-input records, the penalised run's signal
   held-out records (the F3 evidence), the three-run table
   with each figure under its own run's labels, and the
   F1/F2/F3 readings (F1 measured at the worst probe, F2
   measured, F3 not measured as defined with a partial trade
   measured beside it) — all above, measurements and
   derivations labelled, worst probes named, nothing
   uncomputable (nothing reported "not evaluated" in the
   diagnostic set).
9. ✅ **First-cycle and control reproduction** — configured
   with signal 1 (default signal, default anchoring, default
   composition configuration), a first cycle reproduces
   SI-19/SI-21/SI-23/SI-25 exactly — candidate blob
   `sha256:2761e8ffc139ce3e107c11da035e8a73c20792945a494f4e493dccf418b6ca9e`,
   drift 0.17189218290150166, regression 0.1657181391492486 —
   and the re-run unpenalised control at `d0001` reproduces
   SI-25's decisive cycle-2 candidate exactly (blob
   `sha256:646d1e8ba3978d858d4173d561f8799bce37350f88ff2de65f40c223d9e4d675`),
   so the configuration machinery is proven
   behaviour-preserving for both existing semantics and for
   the unpenalised control before the new configuration's
   results are trusted.
10. ✅ **VERIFY, unwind, and untouched scope** — cycle-1
    VERIFY as in SI-19; unwind honestly conditioned (no
    authored stack exists, none unwound, none claimed — see
    VERIFY and unwind). Exactly two authored cycles attempted
    in the decisive run and exactly one penalised decisive
    run performed; `max_stack_depth` 8, `identity_tol` 0.25,
    `capability_eps` 0.25, `max_targets_per_delta` 1,
    `allow_additive` false and every other policy default
    unchanged (asserted); both signals' ids, streams,
    corrections, and partitions unchanged (asserted); both
    anchorings' definitions unchanged (asserted); `si squash`
    still fails closed on the resulting transformer stack
    (asserted; SI-12's suite green); the full existing suite
    is green (below); the demo path is unchanged; `si serve`
    still returns 501 for every transformer bundle (existing
    suite); no CLI wiring added. Per-cycle `train_delta` wall
    times 1.00 s / 2.01 s; total two-cycle wall time 3.39 s;
    full suite runtime below. All on this VM, CPU only.

Full suite on the author's own run: **340 passed,
2 skipped** in 215.90s — the SI-25 baseline (328 passed,
2 skipped) plus this slice's 12 new tests, zero regressions.
Focused run: `tests/test_movement_composition.py`, 12 passed
in 62.90s.

## What is still NOT claimed

- **No authored stack deeper than one.** Cycle 2 REJECTed
  under movement-composition training as it did under both
  anchorings unpenalised; the deepest landed authored stack
  on record remains depth 1 (`d0001`). The penalised cycle-2
  candidate exists only as a rejected, never-written
  candidate (blob-store growth +0 B). No authored two-delta
  stack exists, and the transformer-squash precondition
  remains unmet.
- **Composition-aware training is not shown to work.** On the
  one figure the gate prices, the treatment (0.32509) improves
  only marginally on SI-25's control (0.33290) and worsens on
  SI-23's (0.29634); at the binding probe its movements align
  slightly *more* than the control's. What is shown is
  narrower: composition on allowed inputs is shapeable by a
  penalty term (train aggregate −1.687 → −2.688), the shaping
  transfers to aggregate probe composition (−0.2126 →
  −0.3271), and neither moves the judged maximum below tol.
  One decisive configuration was measured; no claim is made
  about composition penalties at other heads, other depths,
  other λ (no other λ was run, and none is reported as
  anything but an open question for a future RFC), or for
  signal 1 as a decisive matter.
- **No λ conclusion beyond λ = 1.0.** λ = 1.0 was the
  configuration's pre-registered definition. Whether a
  different single λ would behave differently is an open
  question this slice does not measure and does not answer;
  no sweep was run and no un-run λ's outcome is reported.
- **No third cycle, no continual learning, no
  mixed-configuration stack claim.** Exactly two authored
  cycles were attempted in the decisive run, on constructed
  signals, on a constructed model. The decisive run's cycle 1
  is unpenalised-at-base only in the degenerate sense that
  the penalty vanishes there; no stack whose cycles mix
  composition configurations at non-base heads was trained
  or is claimed (the §14 item 4 control is a separate run,
  labelled a control).
- **No Glimmer anything.** No Glimmer forward path, probes,
  training, or authorship; §7's penalty is a miniature
  mechanism and claims nothing at Glimmer scale; the Glimmer
  seed is untouched and its deltas still escalate with checks
  3/4 "not evaluated".
- **No widening.** Two signals, two anchorings, two
  composition configurations, one target, one bundle kind;
  no additive or multi-target authorship; no policy default
  changed; no tolerance moved; both signals are byte-for-byte
  as SI-19/SI-23 defined them; the default anchoring and the
  default composition configuration are unchanged.
- **No serving, no CLI wiring.** Transformer bundles keep
  returning 501.
- **Diagnostics are not verdicts.** The diagnostic set above
  attributes the gate's REJECT after the fact; it is not a
  gate check, it influenced no verdict, and none of it was a
  prediction (RFC §17.12/§17.14). The deepened aggregate
  opposition is a finding about movement composition, not
  evidence the verdict should have differed.
- **Training numbers are training numbers.** Every loss and
  penalty value above is labelled with its signal id,
  partition, head, target source, anchoring, and composition
  configuration; none of them is a gate or probe result
  (§17.1). Allowed-input figures are never presented as probe
  figures or vice versa (§17.18), and the train-to-probe
  composition gap is stated only as the difference between
  the two separately labelled records.
- **Recorded selection is not a choice the mind made**
  (§17.13). The driver configured each cycle's signal,
  anchoring, and composition configuration; the journal shows
  all three. Nothing here describes the mind as choosing,
  deciding, or preferring any of them.

## Terminology

Machine learning infrastructure. This document contains no
"custom LLM software" claims; the miniature is a constructed
machine-learning test instrument, its authored deltas are
trained on constructed signals, and its second signal's
composition-penalised second-cycle delta was judged — and
rejected — on cumulative identity drift, with the penalty
measured doing exactly what it was designed to do on the
inputs it was allowed to see and the gate's binding probe
measured moving the other way.
