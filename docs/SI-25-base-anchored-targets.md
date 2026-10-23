# SI-25 — Base-anchored targets for the miniature (as-built)

> **Scope:** RFC SI-24 §9 (the SI-25 outline) and §10 (the ten
> acceptance criteria — each addressed below with its
> measurements). Target anchoring is now a recorded per-cycle
> configuration beside recorded signal selection, and signal 2
> was trained at the landed head `d0001` against the frozen
> base's targets — exactly one variable changed against
> SI-23's decisive run. **Cycle-2 ADMIT was explicitly NOT
> required (RFC §10.7) — the gate's verdict on the decisive
> cycle 2 is REJECT, reported below as measured.** The
> diagnostics attribute it precisely, and the attribution is
> two-sided: the designed mechanism *appears* — in aggregate
> the base-anchored update's movement **opposes** the first
> update's (negative inner products in logit space and in
> weight space, where SI-23's head-relative update aligned),
> and at SI-23's worst probe the composed drift collapses
> from 0.2963 to 0.1489 — and it *fails where the gate binds*:
> at `tiny-id-0`, cycle 1's own worst probe, the movements
> still align, the composed drift is 0.3329036422073841 >
> tol 0.25, and the maximum composed drift is *worse* than
> the control's even though 5 of 8 probes improve. A second
> §5.5 finding stands beside it: signal 1's base-anchored
> residual at `d0001` measures ~0 and its training does not
> improve, so the existing non-improvement refusal fired —
> reported as a finding, exactly as §5.5 prescribes.

## What was built

- **`si/tiny_transformer.py`** — two additions. (a)
  `assemble_base_weights(bundle_dir, manifest, policy)`: the
  base weight set assembled from the cycle's verified
  manifest with its active chain truncated to empty, in
  exactly one place beside `assemble_weights`, through the
  existing hash-verified `load_applied_tensor` reads; the
  verified manifest is never mutated (the derivation is an
  in-memory view with `head` set to `"base"`), and no tensor
  is copied, materialised, or stored. (b) Recorded anchoring
  configuration on `TinyTransformerMind`: a constructor-level
  `target_anchoring` (default `"weights-under-training"`),
  settable between cycles by the explicit driver call
  `set_target_anchoring(...)`, validated against exactly the
  two named anchorings; `propose` carries the configured
  anchoring on the intent beside the configured signal.
- **`si/tiny_training.py`** — `TARGET_ANCHORINGS` (exactly
  `"weights-under-training"`, `"base-anchored"`),
  `BASE_ANCHORED`, and `validate_target_anchoring(...)` (any
  other value refuses with the existing `ValueError`
  discipline, mirroring `signal_spec`). The module constant
  `TARGET_ANCHORING` is kept — as the *default's* name, not
  as a description of every cycle (§5.3). `train_candidate`
  dispatches on the intent's anchoring and splits **only the
  target-source role**: hidden states, the readout under
  training, candidate construction, and every loss
  evaluation still use the weights under training at the
  cycle's verified head; under `"base-anchored"` the
  signal's targets alone derive from
  `assemble_base_weights(...)`. A failed base assembly
  propagates and refuses training — there is no fallback to
  head-derived targets anywhere in the code. Hyperparams
  record the cycle's own `target_anchoring` and
  `target_source_head` (`"base"` under base-anchoring).
- **`si/gate.py`** — `Intent` gains one field,
  `target_anchoring` (default `"weights-under-training"`),
  carried in `to_dict()` so the journaled intent shows the
  configuration the driver chose. No gate check, verdict
  mapping, tolerance, cap, or policy default changed.
- **`tests/test_base_anchored.py`** (new, 13 tests) and this
  doc. No probe-suite change, no serve change (501 for
  every transformer bundle), no CLI wiring, the Glimmer
  seed untouched, the demo path unchanged, both signals'
  ids, streams, corrections, and partitions unchanged.

There is no autonomous, learned, depth-driven, or
verdict-driven anchoring selection anywhere in the slice —
the mind chooses neither its signal nor its anchoring; the
driver configures both, and the journal shows both.

## Echo-trap records under the new semantics (§5.5)

Pre-training held-out losses, recorded before training
(training numbers, labelled with signal, head, target
source, and anchoring wherever they appear):

| Record | Signal / head / target source / anchoring | Measured | Label |
|---|---|---|---|
| At base | `tiny-predecessor-lean-001` / `base` / `base` / base-anchored | **0.00140625** | measurement — at `base` the two weight sets are one set, so the record IS the closed form `2 * 0.15² / 32` under either anchoring (the coincidence stated wherever an at-base figure is used) |
| At `d0001`, head-relative (SI-23 control) | `tiny-predecessor-lean-001` / `d0001` / `d0001` / weights-under-training | 0.00140625 | SI-23's published record: a by-construction recurrence under head-relative anchoring — not fresh evidence |
| At `d0001`, base-anchored (decisive) | `tiny-predecessor-lean-001` / `d0001` / `base` / base-anchored | **0.002757805049077469** | **genuine measurement** — the head's loss against base-derived targets; no closed form exists for it and none is quoted. It is not ~0: the destination is not already occupied, and training improves on it ~264× (below) |
| Residual record at `d0001` | `tiny-successor-lean-001` / `d0001` / `base` / base-anchored | **3.2118398935406443e-06** | **genuine measurement, and a premise failure (§5.5)** — see below |

The same records on the train partition: signal 2
base-anchored at `d0001` **0.0027535558814751025**; the
signal-1 residual **1.824005808470025e-06**.

**The residual record is the §5.5 already-satisfied case,
handled as a premise failure.** Signal 1's base-anchored
residual held-out loss at `d0001` is 0.23% of the closed
form — the landed head essentially *occupies* signal 1's
fixed destination on held-out sequences. (It equals cycle
1's own `held_out_loss_after`, 3.2118398935406443e-06,
exactly — as it must: same targets, same head; cycle 1
trained toward precisely this destination.) Training
against it does not improve: the trained held-out loss,
measured through the real `train_candidate` path, is
**6.234076521114708e-06** — worse than the pre-training
record — so `train_candidate` **refused its own candidate**
with the existing non-improvement `ValueError` (RFC SI-18
§8.4). The refusal stands; no substitute signal or
configuration was run as the decisive cycle, and no
residual candidate exists to judge (its gate state is
therefore "not evaluated" — there is no candidate). The
measurement is reported here as the slice's finding:
a base-anchored destination the head already occupies
authors nothing.

## Gradient check for the base-anchored loss (§5.8, §10.3)

Central finite differences of signal 2's **base-anchored**
loss — targets derived from the base weight set, hidden
states and readout from the weights under training — through
the existing SI-15 forward (the float32 shared fold
included), at signal 2's pinned initial factors over an
8-example subset, `eps = 0.01`, EVERY one of the 512 factor
entries:

| Configuration | Worst absolute disagreement | Tolerance |
|---|---|---|
| Signal 2 base-anchored at base (new code path) | **5.5249277028738815e-08** | 1e-5 (SI-19's pin, unchanged) |
| Signal 2 base-anchored at head `d0001` (the decisive configuration) | **4.8377706524291005e-08** | 1e-5 |

Both pass with ~200× headroom. The at-base figure is
identical to SI-23's at-base head-relative figure, as it
must be: at base the anchored loss coincides with the
head-relative loss — the coincidence is measured under the
new code path, not asserted. No new derivative was needed
(§5.8: the targets are still constants during optimisation;
only the weight set they are computed *from* differs), and
no tolerance was loosened.

## Coincidence configuration: signal 2 base-anchored at base (§10.2)

Configured with signal 2 and base-anchoring, a first cycle
at base trains and is judged there. **Verdict as measured:
ADMIT** → `d0001`, §6.6 VERIFY pass. It reproduces SI-23's
at-base run **exactly** — the base assembly and the split
target-source path are proven correct where the two
semantics coincide, before the decisive cycle is trusted.
These are at-base facts; they are not the slice's decisive
claim and are not evidence about head `d0001`.

| Measurement (signal `tiny-predecessor-lean-001`, head `base`, target source `base`, anchoring base-anchored) | Value |
|---|---|
| Pre-training held-out loss | 0.00140625 (the closed form — coincidence at base, see above) |
| Training loss, initial → final | 0.002604702300948902 → 3.362224144792228e-06 |
| Held-out loss after training | 3.891045992787381e-06 |
| Held-out examples improved | 16/16 |
| Candidate blob | `sha256:131cc242728024b4f833bc78ea627825f1e8616972fc2670603673e1814645f4` (2,360 B) |
| Candidate norm | 0.21317186144376168 |
| Norm-guard activations | 0 |
| Identity drift (gate) | 0.16672966815531254 ≤ tol 0.25 (worst: tiny-id-7) — pass |
| Capability regression (gate) | 0.2418281203135848 ≤ eps 0.25 (worst: tiny-cap-10) — pass |
| Echo accuracy | 1.0 (16/16) vs floor 0.9375 — pass |
| `train_delta` wall time | 1.03 s |
| Blob-store growth | +2,360 B |

**First-cycle reproduction (§10.9).** Configured with
signal 1, a first cycle reproduces SI-19/SI-21/SI-23
**exactly under either anchoring** — candidate blob
`sha256:2761e8ffc139ce3e107c11da035e8a73c20792945a494f4e493dccf418b6ca9e`,
identity drift 0.17189218290150166, capability regression
0.1657181391492486 — asserted standalone for both
anchorings and inside the decisive run. The anchoring
machinery is therefore proven behaviour-preserving for the
existing semantics.

**Assembly apart, fail-closed (§10.4).** A test tells the
two assemblies apart at head `d0001`: the base view equals
the recorded seed base tensor for tensor (all 30), its
readout equals the recorded base tensor; the head view's
readout equals base + `d0001`'s fold (apply-at-load's own
tensor) and differs from the base's. The verified manifest
is deep-equal before and after the base assembly (never
mutated). With the base assembly broken (injected failure),
base-anchored training refuses with that failure and
returns no candidate, while the *same* bundle state trains
fine head-relative and reproduces SI-23's control candidate
(`sha256:184cc7f0…6e6cf7`) — the refusal is the base
assembly's, and no fallback exists. A genuinely tampered
base (a base tensor blob deleted) and an
architecture-tampered manifest view both refuse the base
assembly itself.

## The decisive two-cycle run (measured on this VM, this slice's runs)

Driver shape: **one reused `TinyTransformerMind`** driven
through `si.run.run_cycles` — one `run_cycles` call for
cycle 1 (signal 1, the default anchoring), then the driver
explicitly reconfigures the same mind to signal 2
(`set_training_signal`) **and** to base-anchored targets
(`set_target_anchoring`) — both recorded selections — then
a second `run_cycles` call for cycle 2. Exactly two
authored cycles were attempted. This is SI-23's decisive
run with exactly one configuration changed: cycle 2's
target source.

| | Cycle 1 | Cycle 2 (treatment) |
|---|---|---|
| Signal (recorded selection) | `tiny-successor-lean-001` (default) | `tiny-predecessor-lean-001` (explicit) |
| Anchoring (recorded selection) | `weights-under-training` (default) | **`base-anchored` (explicit)** |
| Head trained against | `base` | `d0001` |
| Target-source head | `base` | **`base`** |
| Stack depth at training | 0 | 1 |
| Candidate blob | `sha256:2761e8ff…6ca9e` (2,360 B) | `sha256:646d1e8b…d675` (2,360 B) |
| Candidate norm | 0.23555380990005192 | 0.2515524481203623 |
| Train loss, initial → final | 0.0028961365437885785 → 1.824004756500067e-06 | 0.0042998498036725735 → 9.412649442227854e-06 |
| Pre-training held-out loss (at its head, vs its target source) | 0.00140625 | **0.002757805049077469** (genuine measurement — see echo-trap records) |
| Held-out loss after training | 3.2118398935406443e-06 | 1.043608588752459e-05 (~264× lower) |
| Held-out examples improved | 16/16 | 16/16 |
| Norm-guard activations | 0 | 0 |
| `train_delta` wall time | 1.02 s | 1.03 s |
| Gate verdict | **ADMIT** → `d0001`, §6.6 VERIFY pass | **REJECT** (identity), nothing landed |
| Blob-store growth | +2,360 B | +0 B (candidate never written) |

Full candidate blob hashes: cycle 1
`sha256:2761e8ffc139ce3e107c11da035e8a73c20792945a494f4e493dccf418b6ca9e`;
cycle 2
`sha256:646d1e8ba3978d858d4173d561f8799bce37350f88ff2de65f40c223d9e4d675`.
Total two-cycle wall time **2.43 s** (both `run_cycles`
calls, gate included, on this VM). Cycle 1 reproduces
SI-19/SI-21/SI-23 **exactly**. Cycle 2's candidate is
genuinely new (different blob from both the coincidence
candidate and SI-23's control candidate, trained at
`d0001` against base-derived targets), and as a
destination-relative improvement it succeeds (held-out
loss against the base-anchored targets falls ~264×,
16/16 examples closer). The gate does not judge
destination-relative improvement; it judges the composed
head against the seed.

## Cumulative vs incremental accounting (§10.8) — side by side

| Measurement | Cycle 1 | Cycle 2 (treatment) | Accounting |
|---|---|---|---|
| Identity drift (gate, candidate overlay) | 0.17189218290150166 (pass, tol 0.25) | **0.3329036422073841 (FAIL, tol 0.25)** | **seed-absolute / cumulative** — drift of the would-be head vs the seed's recorded logits |
| Capability regression (gate) | 0.1657181391492486 (pass, eps 0.25) | 0.10096919978968799 (pass, eps 0.25) | **head-incremental** — charged only since the pre-cycle head |
| Echo accuracy | 1.0 (16/16, floor 0.9375) | 1.0 (16/16, floor 0.9375) | absolute, vs the recorded floor |
| Landed head's cumulative identity drift after the cycle | 0.17189218290150166 (§6.6 VERIFY, measured-pass) | 0.17189218290150166 (unchanged — re-measured after the REJECT) | seed-absolute, of the head that actually landed |

No incremental figure above is presented as cumulative or
vice versa, and no base-anchored figure is presented as a
head-relative one.

## The §5.6 diagnostics (published whatever the verdict)

All diagnostics below are computed against head `d0001`
with the decisive cycle-2 candidate overlaid. They are
published to *attribute* the gate's verdict after the gate
has spoken; they are not gate checks and influenced no
verdict (RFC §12.12). Per-probe drift values are
measurements (max absolute logit coordinate per probe);
the incremental movement column is a **derivation** from
two measured forwards (composed forward minus the `d0001`
forward), labelled as such.

**(i) Per-probe drift contributions (treatment):**

| Probe | `d0001` drift (measured) | Signal-2 base-anchored incremental movement (derivation) | Composed drift (measured) |
|---|---|---|---|
| tiny-id-0 | 0.17189218290150166 | 0.16173836588859558 | **0.3329036422073841** |
| tiny-id-1 | 0.15088725090026855 | 0.06053225928917527 | 0.09286642074584961 |
| tiny-id-2 | 0.11858986946754158 | 0.14893579483032227 | 0.14936351776123047 |
| tiny-id-3 | 0.14782869815826416 | 0.1503568198531866 | 0.1489148810505867 |
| tiny-id-4 | 0.11720322235487401 | 0.12207234278321266 | 0.14728415571153164 |
| tiny-id-5 | 0.09891846403479576 | 0.12109210342168808 | 0.11245626583695412 |
| tiny-id-6 | 0.12363102938979864 | 0.16010307148098946 | 0.14491976890712976 |
| tiny-id-7 | 0.08795571327209473 | 0.11142894998192787 | 0.14189285784959793 |

Worst probes, named: `d0001`'s own drift — **tiny-id-0**
(0.17189218290150166); the incremental movement —
**tiny-id-0** (0.16173836588859558); the composed drift —
**tiny-id-0** (0.3329036422073841, equal to the gate's
measured cycle-2 drift). At tiny-id-0 the composed drift
is, to the printed digits, the *sum* of the two
contributions (0.17189218290150166 + 0.16173836588859558 ≈
0.3329) — the aligned extreme, at that probe. Everywhere
else it is not (see (ii)).

**(ii) Alignment diagnostic (derivation):** the two
updates' movement vectors are `m1` = (`d0001` forward −
seed expected) and `m2` = (composed forward − `d0001`
forward), per identity probe — read against the
head-relative counterpart, as §5.6 pre-registered (the
§5.6 expectation, stated in the RFC before the run, was
that the base-anchored movement would *oppose* the first
update's on the overshoot component; it is a reading of
the design, not a prediction of the verdict, §12.14):

- **In aggregate the mechanism appears.** Over all 8
  identity probes (256 coordinates): inner product
  **−0.21256230517288188** (negative — opposing) where the
  control's was **+0.07637087369869666** (aligned); sign
  agreement **0.37109375** (95/256) where the control's
  was 0.54296875 (139/256); movement-vector norms ‖m1‖ =
  0.7119678353441855, ‖m2‖ = 0.7816722548913912.
- **At the control's worst probe it appears decisively.**
  At `tiny-id-3`: inner product **−0.02224599986114801**
  (opposing; control: +0.021769049613635714, aligned),
  sign agreement 0.5 (16/32); composed drift 0.1489 where
  the control composed 0.2963.
- **At the treatment's worst probe it fails.** At
  `tiny-id-0`: inner product **+0.0338873784299374**
  (positive — aligned), sign agreement 0.5 (16/32). This
  is the probe where `d0001`'s own drift was already the
  largest, and it is where the gate binds.

**(iii) Folded updates (measured):** Frobenius inner
product of the two folded `lm_head.weight` updates
**−0.011457914936582613** (negative — opposing in weight
space as well) where the control's was
+0.004356018453758104; norms ‖Δ1‖_F = 0.23555380990005192,
‖Δ2‖_F = 0.2515524481203623 (Δ2's norm equals its candidate
norm, as it must for a single-target LoRA). The
base-anchored update is *larger* than the head-relative
one (0.2516 vs 0.2132): it corrects the head's existing
lean as well as adding its own — §5.6's named cost,
measured.

**(iv) Composed capability (measured):** head-incremental
regression **0.10096919978968799** ≤ eps 0.25 (worst:
tiny-cap-10) — *better* than the control's
0.15023326850496233 (§5.6's capability warning did not
bind here); echo accuracy **1.0 (16/16)** vs the recorded
floor 0.9375.

**(v) Signal 2's held-out records (labelled by
anchoring):** head-relative (SI-23): at base 0.00140625 →
3.891045992787381e-06; at `d0001` 0.00140625 (by
construction) → 3.891045006596239e-06. Base-anchored
(SI-25): at base 0.00140625 (coincidence) →
3.891045992787381e-06 (the identical at-base figure, as
the coincidence requires); at `d0001` **0.002757805049077469**
(genuine measurement) → 1.043608588752459e-05; 16/16
examples closer in both decisive runs. The two anchorings'
figures are never averaged, summed, or blended (§12.11).

**(vi) Control/treatment table (§5.6, §12.16).** Same
signal, same head, same target, same hyperparameters —
differing only in target source. Each figure under its own
run's labels:

| Per-probe composed drift | tiny-id-0 | tiny-id-1 | tiny-id-2 | tiny-id-3 | tiny-id-4 | tiny-id-5 | tiny-id-6 | tiny-id-7 | Max (worst probe) |
|---|---|---|---|---|---|---|---|---|---|
| Control — SI-23, weights-under-training | 0.2092923242598772 | 0.15134692192077637 | 0.14906740188598633 | **0.2963416576385498** | 0.23394920490682125 | 0.14004023000597954 | 0.1332004750147462 | 0.2018739078193903 | 0.2963416576385498 (tiny-id-3) |
| Treatment — SI-25, base-anchored | **0.3329036422073841** | 0.09286642074584961 | 0.14936351776123047 | 0.1489148810505867 | 0.14728415571153164 | 0.11245626583695412 | 0.14491976890712976 | 0.14189285784959793 | 0.3329036422073841 (tiny-id-0) |

| Gate / candidate figure | Control (SI-23) | Treatment (SI-25) |
|---|---|---|
| Candidate blob | `sha256:184cc7f0…6e6cf7` | `sha256:646d1e8b…d675` |
| Candidate norm | 0.21317186599953494 | 0.2515524481203623 |
| Identity drift (cumulative) | 0.2963416576385498 — FAIL | 0.3329036422073841 — FAIL |
| Capability regression (head-incremental) | 0.15023326850496233 — pass | 0.10096919978968799 — pass |
| Echo accuracy | 1.0 — pass | 1.0 — pass |
| Verdict | REJECT | REJECT |

Per probe, the treatment's composed drift improves on the
control at 5 of 8 probes (tiny-id-1, -3, -4, -5, -7 —
dramatically at the control's worst probe) and worsens at
3 (tiny-id-0 substantially, tiny-id-6, and tiny-id-2
marginally: 0.14907 → 0.14936). The maximum — the only
figure the gate's identity arm prices — worsens.

**Attribution:** both REJECTs are movement-composition
failures, but of opposite aggregate character. SI-23's
second update *aligned* with the first in aggregate and
at its worst probe. SI-25's second update *opposes* the
first in aggregate, in weight space, and at the control's
worst probe — the fixed-destination mechanism of §5.2 is
measurably present — yet at `tiny-id-0` the two movements
still add almost exactly, and `tiny-id-0` was already
cycle 1's worst probe, so the composed maximum lands
*higher* than the control's. The §5.6 derivation priced
*exact arrival* at the destination's at-base drift
(0.16672966815531254); the measured arrival is per-probe
uneven — near-exact opposition at some probes, none at
the binding one — so the derivation's limit is not
approached where it matters. The treatment's larger
update also exceeds cycle 1's remaining headroom
(0.07810781709849834) in incremental movement at **7 of 8**
probes (control: 5 of 8). No diagnostic was uncomputable;
nothing in this set is reported "not evaluated".

## VERIFY and unwind (§10.10)

- **Cycle 1** ADMITted and passed the §6.6 probe VERIFY
  exactly as in SI-19/SI-21/SI-23 (journaled
  `transformer-probe-verify`, measured-pass on both arms,
  echo 1.0, baseline labelled "pre-cycle head
  (caller-captured)").
- **Cycle 2** was REJECTed at the gate and never
  materialised, so there is no cycle-2 VERIFY and no
  VERIFY rollback — nothing landed to verify or roll
  back. The would-be VERIFY rollback target, had it
  landed and failed, is `d0001` by the miniature branch's
  existing construction; that path is unchanged and
  remains covered by SI-17's injected-failure test.
- **Unwind, honestly conditioned (§9.9):** because cycle
  2 REJECTed, **no authored two-delta stack exists** —
  nothing is unwound, and no unwind is claimed. The
  bundle head after a decisive SI-25 run is `d0001`.
  SI-21's fixture-proven unwind (authored `d0001` + a
  benign hand-built fixture `d0002`, labelled a fixture)
  remains the only unwind evidence, exactly as SI-21 left
  it. The transformer-squash precondition (an authored
  two-delta stack, RFC §7.1) therefore remains **unmet**
  after SI-25 as well.

## Acceptance criteria (RFC SI-24 §10) — status

1. ✅ **Recorded anchoring** — the configuration serves
   exactly the two named anchorings; every other value
   refuses with the existing `ValueError` discipline at
   the constructor, the setter, the intent, and the
   `train_candidate` dispatch (asserted). Recorded
   selection is asserted per cycle on both verdict paths:
   cycle 1's journaled `training-hyperparams` (ADMIT,
   under `d0001`) records `target_anchoring =
   "weights-under-training"`, `head = base`,
   `target_source_head = base`; cycle 2's (REJECT, under
   its self-edit) records `target_anchoring =
   "base-anchored"`, `head = d0001`,
   `target_source_head = base` — the first disagreement
   of the two head fields in the project's history, read
   back from the cycle's own journal. No autonomous
   selection exists; `TARGET_ANCHORING` survives as the
   default's name.
2. ✅ **Coincidence at base** — signal 2 base-anchored at
   base reproduces SI-23's at-base run exactly (blob
   `sha256:131cc242…45f4`, drift 0.16672966815531254,
   regression 0.2418281203135848), and signal 1's first
   cycle reproduces `d0001` exactly under either
   anchoring (criterion 9).
3. ✅ **Gradient check for the base-anchored loss, at the
   decisive head** — re-run at `d0001` (worst
   4.8377706524291005e-08) AND at base under the new code
   path (worst 5.5249277028738815e-08), tolerance 1e-5,
   both published above.
4. ✅ **Target-source assembly, fail-closed and
   byte-honest** — the assemblies are told apart by test;
   the verified manifest is never mutated (asserted
   deep-equal); a broken base assembly refuses training
   with no fallback (asserted — the same state trains
   head-relative); a tampered base refuses the assembly;
   blob-store growth per cycle is attributable only to a
   landed candidate (+2,360 B cycle 1, +0 B cycle 2,
   asserted).
5. ✅ **Echo-trap records under the new semantics** —
   signal 2's base-anchored pre-training held-out loss at
   `d0001` is recorded and published as the genuine
   measurement 0.002757805049077469, labelled with signal,
   head, target source, and anchoring, with no closed
   form quoted for it; the signal-1 residual record at
   `d0001` is published as a diagnostic
   (3.2118398935406443e-06) together with its outcome:
   training did not improve (6.234076521114708e-06), the
   existing non-improvement refusal stands, and no
   substitute was run as the decisive cycle.
6. ✅ **Determinism per cycle** — a full decisive run
   repeated from the seed reproduces both candidates'
   blobs/records, the landed delta record, and the
   journaled gate measurements exactly; hyperparams equal
   except the labelled `wall_time_seconds`. The
   base-anchored `train_delta` alone is likewise
   byte-identical across repeats (asserted).
7. ✅ **Gate honesty per cycle** — cycle 2's journaled
   checks 3/4 equal an independent `ProbeContext`
   evaluation of the same candidate against `d0001` (abs
   1e-12); cycle 1's equality is covered by the
   SI-19/SI-21/SI-23 suites this run keeps green. Cycle
   2's verdict is **REJECT, as measured**. No ADMIT was
   required: SI-25 succeeds by §5's claim — a
   base-anchored candidate was genuinely authored against
   the landed head under the new semantics and genuinely
   judged.
8. ✅ **Cumulative accounting, diagnostics, and the
   control/treatment table published** — the side-by-side
   accounting table and the full §5.6 diagnostic set
   above, measurements and derivations labelled, worst
   probes named, the two runs' figures kept distinct and
   labelled by anchoring, nothing uncomputable (nothing
   reported "not evaluated" in the diagnostic set; the
   residual candidate's gate state is "not evaluated"
   because no candidate exists — see criterion 5).
9. ✅ **First-cycle reproduction** — configured with
   signal 1 (the default signal), a first cycle
   reproduces SI-19/SI-21/SI-23 exactly under the default
   anchoring AND under base-anchoring: candidate blob
   `sha256:2761e8ffc139ce3e107c11da035e8a73c20792945a494f4e493dccf418b6ca9e`,
   drift 0.17189218290150166, regression
   0.1657181391492486 — asserted standalone (both
   anchorings) and inside the decisive run.
10. ✅ **VERIFY, unwind, and untouched scope** — cycle-1
    VERIFY as in SI-19; unwind honestly conditioned (no
    authored stack exists, none unwound, none claimed —
    see VERIFY and unwind). Exactly two authored cycles
    attempted in the decisive run; `max_stack_depth` 8,
    `identity_tol` 0.25, `capability_eps` 0.25,
    `max_targets_per_delta` 1, `allow_additive` false and
    every other policy default unchanged (asserted);
    both signals' ids, streams, corrections, and
    partitions unchanged (asserted); `si squash` still
    fails closed on the resulting transformer stack
    (asserted; SI-12's suite green); the full existing
    suite is green (below); the demo path is unchanged;
    `si serve` still returns 501 for every transformer
    bundle (existing suite); no CLI wiring added.
    Per-cycle `train_delta` wall times 1.02 s / 1.03 s;
    total two-cycle wall time 2.43 s; full suite runtime
    below. All on this VM, CPU only.

Full suite on the author's own run: **328 passed,
2 skipped** in 152.90s — the SI-23 baseline (315 passed,
2 skipped) plus this slice's 13 new tests, zero
regressions. Focused run: `tests/test_base_anchored.py`,
13 passed in 34.05s.

## What is still NOT claimed

- **No authored stack deeper than one.** Cycle 2
  REJECTed under base-anchoring as it did under
  head-relative anchoring; the deepest landed authored
  stack on record remains depth 1 (`d0001`). The
  base-anchored cycle-2 candidate exists only as a
  rejected, never-written candidate (blob-store growth
  +0 B). No authored two-delta stack exists, and the
  transformer-squash precondition remains unmet.
- **Base-anchoring is not shown to be the better
  semantics.** The treatment's composed maximum drift
  (0.3329036422073841) is *worse* than the control's
  (0.2963416576385498), even though its aggregate
  alignment opposes, its capability regression improves,
  and 5 of 8 probes improve. One decisive configuration
  was measured; no claim is made about base-anchoring at
  other heads, other depths, or for signal 1 as a
  decisive matter (§5.7's residual cycle was a diagnostic
  and its premise failed — it is not evidence that
  signal 1 can or cannot be stacked).
- **The §5.6 derivation is not confirmed.** Exact arrival
  would price the composed drift at the destination's
  at-base drift (0.16672966815531254); the measured
  composed drift is 0.3329036422073841. The derivation
  remains what it was labelled in the RFC: a statement
  about a hypothetical exact arrival, now measured to be
  far from the binding probe's reality.
- **Signal 2 ADMITs at base — under either anchoring —
  and only at base.** Its at-base first cycles ADMITted
  (drift 0.1667, identical under both anchorings by the
  coincidence); that is a fact about signal 2's
  definition against the seed, not evidence that a
  signal-2 delta lands on any non-base head under any
  anchoring, and it is never presented as such.
- **No third cycle, no continual learning, no
  mixed-anchoring stack claim.** Exactly two authored
  cycles were attempted in the decisive run, on
  constructed signals, on a constructed model. The
  decisive run's cycle 1 is head-relative-at-base only in
  the degenerate sense that the anchorings coincide
  there; no stack whose cycles mix anchorings at
  non-base heads was trained or is claimed.
- **No Glimmer anything.** No Glimmer forward path,
  probes, training, or authorship; §5.4's base assembly
  is a 30-tensor miniature mechanism and claims nothing
  at Glimmer scale; the Glimmer seed is untouched and
  its deltas still escalate with checks 3/4 "not
  evaluated".
- **No widening.** Two signals, two anchorings, one
  target, one bundle kind; no additive or multi-target
  authorship; no policy default changed; no tolerance
  moved; both signals are byte-for-byte as SI-19/SI-23
  defined them; the default anchoring is unchanged.
- **No serving, no CLI wiring.** Transformer bundles
  keep returning 501.
- **Diagnostics are not verdicts.** The §5.6 set above
  attributes the gate's REJECT after the fact; it is not
  a gate check, it influenced no verdict, and none of it
  was a prediction (RFC §12.12/§12.14). The aggregate
  opposition is a finding about movement composition,
  not evidence the verdict should have differed.
- **Training numbers are training numbers.** Every loss
  above is labelled with its signal id, partition, head,
  target source, and anchoring; none of them is a gate
  or probe result (§12.1). The two signals' and the two
  anchorings' held-out figures are never averaged,
  summed, or blended (§12.11/§12.15).
- **Recorded selection is not a choice the mind made**
  (§12.13). The driver configured each cycle's signal
  and anchoring; the journal shows both. Nothing here
  describes the mind as choosing, deciding, or
  preferring either.

## Terminology

Machine learning infrastructure. This document contains
no "custom LLM software" claims; the miniature is a
constructed machine-learning test instrument, its
authored deltas are trained on constructed signals, and
its second signal's base-anchored second-cycle delta was
judged — and rejected — on cumulative identity drift,
with the mechanism that was supposed to prevent exactly
that measured appearing in aggregate and failing at the
binding probe.
