# SI-33 — First non-readout authored target for the miniature (as-built)

> **Scope:** RFC SI-32 §12.1 (the new-target candidate),
> §15 (the SI-33 outline, items 1–9), and §16 (the ten
> acceptance criteria — each addressed below with its
> measurements), plus §17 non-goals and §18 honesty /
> labelling (items 24–25 are new in that RFC and bind this
> slice). Authorship target is now a recorded per-cycle
> configuration with exactly two named values, and signal 2
> was trained at the landed head `d0001`, base-anchored,
> composition `"none"`, exactly as SI-25 trained it except
> that its rank-8 LoRA factors parameterise the final
> layer's MLP down-projection
> (`model.language_model.layers.1.mlp.down_proj.weight`,
> shape [32, 64]) instead of the readout. Exactly one
> variable changed against SI-25's decisive run: where the
> authored update acts. The SI-25 control is re-run inside
> this slice as the treatment's primary control, as are
> SI-27's, SI-29's, and SI-31's.
> **Cycle-2 ADMIT was explicitly NOT required (RFC §16
> criterion 7).** The gate's verdict on the decisive
> cycle 2 is reported as measured in the decisive-run
> section below: **ADMIT** — identity drift
> 0.17195402272045612 ≤ tol 0.25, capability regression
> 0.001467481255531311 ≤ eps 0.25, safety suite pass — so
> the authored two-delta stack exists for the first time
> (head `d0002`, stack depth 2), with the second delta
> labelled a non-readout-target candidate wherever the
> stack is described (§18 item 24). The falsification
> readings are reported exactly as measured beside it:
> F1′′′ **not measured**, F2′′′ **not measured as a
> reading** (transfer clause holds, maximum clause
> fails), F3′′′ **measured** — the destination counts as
> traded away under the pre-registered rule, because the
> candidate that passed the gate barely approached it
> (held-out residual 0.0027387202915082178 against the
> Phase-0 boundary 2.087217177504918e-05). The attribution
> section below gives the measured reason both facts hold
> at once: on this miniature the final layer's MLP branch
> is nearly inert, so an update acting through it moves
> the probes and the destination by comparably tiny
> amounts. This is a miniature-scale result about this
> constructed model, stated with its labels, and nothing
> in it is claimed at any other scale.
> **This document is ordered as RFC §15 item 1 requires:
> Phase 0 — the backward derivation, the control records,
> the pre-registered successor readings, and the
> gradient-check protocol — comes first, before any
> decisive number.**

## What was built

- **`si/tiny_training.py`** — the SI-33 additions. (a) The
  authorship-target axis: `AUTHORSHIP_TARGETS` (exactly
  `"lm-head"` and `"final-mlp-down"`),
  `DEFAULT_AUTHORSHIP_TARGET` (`"lm-head"`),
  `AUTHORSHIP_TARGET_TENSORS` (each value's tensor, by
  definition: `lm_head.weight` and
  `model.language_model.layers.1.mlp.down_proj.weight`),
  and `validate_authorship_target(...)` /
  `authorship_target_name(...)` serving both values with
  the existing `ValueError` discipline. Rank (8) and alpha
  (8) are pinned by the axis's definitions exactly as the
  signals pinned them; neither is separately settable, and
  no caller supplies a tensor name. The target pinning's
  move from the signals' specs to this recorded axis is
  the one deliberate definition change RFC §12.1 argues:
  both signals' ids, streams, corrections, partitions,
  loss form, rank, and alpha are byte-for-byte as
  SI-19/SI-23 defined them. (b) The down-projection
  training path: `forward_to_final_down(...)` (a forward
  replica exposing the final layer's `h_mid` and SwiGLU
  output `s` — proven bit-identical to the library forward
  when completed, by test), `down_inputs_for(...)`,
  `initial_factors_down(...)` (the signal's own init
  stream, A drawn before B, at the down shapes),
  `analytic_gradients_down(...)` (Phase 0's derivation in
  code), and `down_factor_loss(...)` (the same loss
  through the float32 shared fold — the function the
  gradient check differentiates). (c) `train_candidate`
  dispatches on the intent's authorship target: under
  `"final-mlp-down"` it trains the signal's factors
  against the same target-logits MSE (targets derived per
  the cycle's anchoring, exactly as SI-25), with the
  gradient taken through the new derivation, the same
  2,000 steps, learning rate 1.0, norm budget 0.9, and
  the non-improvement refusal applied to the trained
  candidate's signal held-out loss. Composition under the
  new target is `"none"` only — any other composition
  configuration refuses, because blending a composition
  mechanism with the target change is a non-goal (RFC
  §17). Under `"lm-head"` every training path is
  byte-for-byte what it was before SI-33 under every
  anchoring and every composition configuration — proven
  by exact reproduction of all four controls and of the
  at-base run inside this slice. Every cycle's hyperparams
  now record `authorship_target` beside the tensor its
  definition names (`target`).
- **`si/tiny_transformer.py`** — the recorded
  authorship-target configuration on `TinyTransformerMind`:
  constructor-level `authorship_target` (default
  `"lm-head"`) and the explicit driver call
  `set_authorship_target(...)`; neither takes a
  tensor-name, rank, or alpha argument (asserted);
  `propose` carries the configured value on the intent
  beside the configured signal, anchoring, and
  composition, and names the axis value's tensor in the
  intent's targets.
- **`si/gate.py`** — `Intent` gains
  `authorship_target: str = "lm-head"` and its `to_dict`
  entry, mirroring the three prior axes. No gate check,
  verdict mapping, tolerance, cap, or policy default
  changed.
- **`si/tiny_seed.py`** — one new name constant,
  `FINAL_MLP_DOWN_NAME`, for the tensor the axis's second
  value names. No tensor, shape, or construction changed.
- **`tests/test_final_mlp_down_target.py`** (new, 16
  tests) and this doc. No prior test file is modified: no
  prior exactness assertion changes, because no prior
  configuration's definition moved. No probe-suite change,
  no serve change (501 for every transformer bundle), no
  CLI wiring, the Glimmer seed untouched, the demo path
  unchanged.

There is no autonomous, learned, depth-driven,
verdict-driven, or probe-driven selection on any axis —
the mind chooses neither its signal, its anchoring, its
composition configuration, nor its authorship target; the
driver configures all four, and the journal shows all
four. Exactly one new-target decisive run was performed;
no sweep over targets, and no second non-default target,
exists in the slice.

## Phase 0 — the derivation, the control records, the successor readings, and the gradient-check protocol (RFC §15 item 1)

Computed and stated before any decisive candidate was
built. If the derivation could not be made checkable, or
the gradient check (item (iv)'s protocol, run in the
gradient-check section below) had failed, the slice would
have stopped here and published the failure as its result.

### (i) The backward derivation for `"final-mlp-down"` (derivation, labelled as such)

The forward code it differentiates is
`si.tiny_transformer.forward`, final layer (`i = 1`), in
order — each forward line is quoted so a reviewer can
check the derivation against it line by line:

```text
m    = _rms_norm(h, w[post_attention_layernorm.weight])   # [t, 32]
gate = m @ W_gate.T ;  up = m @ W_up.T                    # [t, 64]
silu = gate / (1 + exp(-gate))
h    = h + (silu * up) @ W_down.T                         # residual add
h    = _rms_norm(h, w[FINAL_NORM_NAME])                   # final RMSNorm
logits = h @ W_lm.T                                       # readout
```

Notation for one example, at its **last position only**
(the loss reads last-position logits only, as in every
prior slice). Forward quantities:

- `u = (silu(gate) * up)[-1]` ∈ R^64 — the SwiGLU
  activation's output, the down map's input.
- `h_mid` ∈ R^32 — the residual stream entering the final
  layer's MLP block (the `h` on the residual-add line's
  right-hand side), i.e. the attention block's output.
- `W_eff = W_down + scale * B @ A`, with `B` [32, 8],
  `A` [8, 64], `scale = alpha / rank = 1.0` — the
  effective down-projection under the candidate's factors
  (the shared fold `si.apply.fold_target` computes exactly
  this tensor).
- `y = h_mid + W_eff @ u` ∈ R^32 — the residual stream
  after the add.
- `r = sqrt(mean(y * y) + eps)`, `eps = 1e-5`;
  `h_final = (y / r) * w_norm`, with `w_norm` the final
  RMSNorm weight — the final RMSNorm's output.
- `logits = W_lm @ h_final` ∈ R^32, `W_lm` the readout
  under training.
- Loss `L = mean_v (logits_v - target_v)^2` over the
  signal example's target logits (SI-25's base-anchored
  target-logits MSE in the decisive configuration; the
  derivation is anchoring-agnostic — only the targets'
  source differs).

**Why `u` and `h_mid` are fixed inputs for the factor
gradients (the justification §15 item 1(i) requires).**
In the forward order above, `u` is computed from `m`,
`gate`, and `up`, all of which are computed from `h_mid`
and from tensors upstream of the down-projection; `h_mid`
itself is the attention block's output. Neither quantity
reads `W_down` or the factors at any point: the
down-projection is the *first* tensor on the path that
the factors touch, and everything it consumes is already
fixed when it is reached. (The layer-0 down-projection is
a different tensor, untouched by this target.) The
gradient therefore treats `u` and `h_mid` as constants —
implemented by `forward_to_final_down` /
`down_inputs_for`, and asserted by test: folding a
candidate into the down tensor changes neither `h_mid`
nor `s` (bit-identical), and a forward completed from
those outputs reproduces the library forward's logits
bit-identically after the float32 storage cast.

**The chain, step by step** (V = D = 32):

1. `dL/dlogits = 2 * (logits - target) / V` — the MSE's
   own derivative, as in SI-19.
2. Through the readout: `g_h = W_lm.T @ dL/dlogits`
   (since `logits = W_lm @ h_final`).
3. Through the final RMSNorm's weight:
   `g_n = g_h * w_norm` (elementwise).
4. Through the RMSNorm's normalisation — the one new
   Jacobian on this path. With `n = y / r` and
   `r = sqrt(mean(y^2) + eps)`,
   `dn_i/dy_j = delta_ij / r - y_i * y_j / (D * r^3)`, so
   `g_y = g_n / r - y * (g_n . y) / (D * r^3)`.
5. Through the residual add: `y = h_mid + W_eff @ u`
   with `h_mid` fixed, so `dL/dW_eff = outer(g_y, u)`
   =: `G` ∈ R^[32, 64].
6. Through the LoRA fold: `W_eff = W_down + scale * B @ A`
   gives `grad_B = scale * G @ A.T` [32, 8] and
   `grad_A = scale * B.T @ G` [8, 64] — the same chain
   rule as the readout path's step, at the new shapes.

`G` is averaged over the partition's examples, exactly as
the readout path averages its `grad_W`. This is
`analytic_gradients_down` in code, line for line.

### (ii) The control records the readings need (measurements on an already-judged candidate, labelled diagnostics)

Re-run inside this slice before the decisive run: the
SI-25 control (signal 2, base-anchored, composition
`"none"`, target `"lm-head"`, at head `d0001`) reproduces
SI-25's decisive candidate exactly — blob
`sha256:646d1e8ba3978d858d4173d561f8799bce37350f88ff2de65f40c223d9e4d675`,
drift 0.3329036422073841 (the full control set is in the
controls section below). Its per-probe inner products
with `m1` — `m1` = (`d0001` forward − seed expected),
`m2` = (candidate forward − `d0001` forward), computed
through candidate overlays as diagnostics on an
already-judged candidate — are:

| Probe | SI-25 control inner product ⟨m1, m2⟩ |
|---|---|
| tiny-id-0 | **+0.0338873784299374** |
| tiny-id-1 | −0.017059672853274165 |
| tiny-id-2 | −0.07934423600975443 |
| tiny-id-3 | −0.02224599986114801 |
| tiny-id-4 | −0.05053692823121107 |
| tiny-id-5 | −0.03510892397145824 |
| tiny-id-6 | −0.05115886243681786 |
| tiny-id-7 | +0.009004939760844474 |

Aggregate over the 8 identity probes: inner product
−0.21256230517288188 (opposing), sign agreement 0.37109375
(95/256 — published beside the magnitude-weighted figure,
never alone, §18 item 19), ‖m1‖ = 0.7119678353441855,
‖m2‖ = 0.7816722548913912. These are the figures F2′′′'s
transfer clause compares against — stated here, in
advance of the treatment.

Its allowed-input composition record, republished under
its own labels (signal 2's partitions; allowed inputs,
not probes, §18 item 18): on the **train partition**,
aggregate inner product −1.6873837019991191, `mean_p`
−0.0016478356464835148, sign agreement 0.267578125,
‖m1‖ = 1.4131161397867473, ‖m2‖ = 1.6765305214306858; the
largest-`p` allowed train input is train index 19
(`p` = −0.0006878804284308964 — every train `p` is
negative for this control, so the largest is the least
opposing). On the **held-out partition** (reported only),
aggregate inner product −0.845596374259425, `mean_p`
−0.0016515554184754395, sign agreement 0.26953125,
‖m1‖ = 0.9997655382584564, ‖m2‖ = 1.1850839719703719.

### (iii) The pre-registered successor readings (stated before the decisive numbers)

- **F1′′′ — the geometry survives the fold.** F1′′′ is
  measured iff the treatment's composed maximum stays
  above tol 0.25 AND its probe inner product at
  `tiny-id-0` is positive (aligned). Falsified premise:
  that acting upstream of the readout changes the binding
  probe's alignment, for this signal, at this head.
- **F2′′′ — the maximum is immobile even when the binding
  probe turns.** Two separately reported clauses:
  (a) the *transfer clause* holds iff the treatment's
  probe inner product at its own worst probe is smaller
  than the SI-25 control's inner product at that same
  probe, as published in Phase 0 (ii) above; (b) the
  *maximum clause* holds iff the composed maximum stays
  above tol 0.25. F2′′′ is measured iff (a) and (b) both
  hold; each clause's outcome is reported either way.
- **F3′′′ — the destination is traded away.** Read under
  SI-29's three clauses unchanged in their thresholds,
  against the re-run SI-25 control: the treatment's
  destination counts as **approached** (F3′′′ does not
  fire) iff (a) its signal held-out loss falls, (b) at
  least 12 of 16 held-out examples are closer, and (c)
  its held-out residual is at most 2× the re-run SI-25
  control's residual. The control's published held-out
  residual is 1.043608588752459e-05, so the boundary
  stated here, before the decisive numbers, is
  **2.087217177504918e-05**. If any clause fails, the
  destination counts as traded away and F3′′′ is
  measured, with the trade's shape published beside the
  reading.

### (iv) The gradient-check protocol

Central finite differences over **every one of the 768
factor entries** (B [32, 8] = 256 entries, A [8, 64] =
512 entries), at base AND at head `d0001`, in the
established SI-19/SI-25 check shape: an 8-example subset
of signal 2's train partition, `eps = 0.01`, the finite
differences differentiating `down_factor_loss` (the
float32 shared fold, through the existing forward) and
compared against `analytic_gradients_down` at the pinned
initial factors, targets base-derived (base-anchored),
down inputs / readout / norm weights from the weights
under training. Tolerance **1e-5**, unchanged. The
readout path's existing checks stand cited for the
`"lm-head"` path (they are re-run inside this slice's
controls and suite), not silently assumed and not re-run
in the new target's place.

## Gradient check (§16 criterion 3) — measured

| Head | Worst disagreement over all 768 entries | At | Tolerance |
|---|---|---|---|
| base | 3.8495586315441856e-08 | A[2, 10] | 1e-5 |
| `d0001` | 2.7961983521798964e-08 | A[6, 3] | 1e-5 |

Both heads pass with ~260–360× headroom, in line with the
readout path's published figures (5.5e-08 / 4.8e-08).
No decisive run was trusted before this check passed;
it passed, so no Phase-0 stop was triggered.

## Probe-blindness of training (§16 criterion 4)

Asserted three ways by test: (1) signal 2's train
partition is disjoint from both probe suites and from
its own held-out partition under the new code path;
(2) a spy on `down_inputs_for` and on
`analytic_gradients_down` during a real new-target
training proves both input sets are exactly signal 2's
train partition (the gradient's remaining inputs are the
landed head's weights — down-projection, readout, and
final-norm tensors — named here as a set); (3) a
perturbed-probe-artifact bundle copy (every probe input
and expected-logit entry overwritten) produces a
**byte-identical** new-target candidate (blob
`sha256:d7af7f1f…e625`, the decisive candidate). No probe
input, probe logit, held-out sequence, or probe-derived
quantity enters the loss, the gradient, or the refusal
under either target.

## The at-base records (§16 criterion 5) — kept separately labelled

**(a) Default-target reproduction at base.** Signal 2,
base-anchored, target `"lm-head"`, at base reproduces the
established at-base run **exactly**: blob
`sha256:131cc242728024b4f833bc78ea627825f1e8616972fc2670603673e1814645f4`,
identity drift 0.16672966815531254, capability regression
0.2418281203135848, ADMIT. The axis's introduction did not
move the existing path.

**(b) The new-target at-base run — a labelled at-base
fact, NOT a coincidence result, NOT the decisive claim.**
Signal 2, base-anchored, target `"final-mlp-down"`, at
base: candidate blob
`sha256:98768588aff8852d4177ed9272fe7d46cd904d3ebeac4bd5078b83cef0258d9a`
(3,416 B, rank 8, alpha 8), candidate norm
0.29741157499219223, gate identity drift
0.0016501061618328094, capability regression
0.0026158951222896576, **ADMIT** to `d0001` with §6.6
VERIFY measured-pass. Its training record (signal
`tiny-predecessor-lean-001`, head `base`, target source
`base`, anchoring `base-anchored`, authorship target
`"final-mlp-down"`): train loss 0.0014040930207204262 →
0.0013986330263025242; pre-training held-out loss
0.00140625 (the closed form — at base the two weight sets
coincide); held-out loss after 0.0013987123437990195
with **14/16** held-out examples closer; norm-guard
activations 0. This is the target's first measurement:
even at base, where the readout target reaches a
held-out residual of 1.043608588752459e-05-scale losses in
its own runs, the down target's candidate improves its
held-out loss by less than 1% — the near-inert-branch
attribution in the diagnostics section is already visible
here, at base, under the target's own labels.

## All controls, re-run inside the slice (§16 criterion 9)

| Control | Configuration (all: signal 2, base-anchored, target `"lm-head"`, at head `d0001`) | Blob | Drift | Regression | Verdict |
|---|---|---|---|---|---|
| SI-25 | composition `"none"` | `sha256:646d1e8b…d675` | 0.3329036422073841 | 0.10096919978968799 | REJECT |
| SI-27 | composition `"movement-composition"`, λ 1.0 | `sha256:1bb277ab…b637` | 0.32509003952145576 | 0.14536726917140186 | REJECT |
| SI-29 | composition `"worst-input-composition"`, λ_w 1.0, k 4 | `sha256:11cab02e…b8db5` | 0.31215621903538704 | 0.11724743689410388 | REJECT |
| SI-31 | composition `"train-partition-projection"`, α −0.8450027779470831 (constructed, rank 16) | `sha256:32d13f65…a5c1` | 0.4775387533009052 | 0.177281160838902 | REJECT (safety rank + identity) |

All four reproduce their decisive candidates **exactly**
(blob-for-blob, drift-for-drift), so the target axis's
refactor is proven behaviour-preserving for every
existing semantics, including the projection construction,
before the new target's results are trusted. Full blob
hashes: SI-25
`sha256:646d1e8ba3978d858d4173d561f8799bce37350f88ff2de65f40c223d9e4d675`;
SI-27
`sha256:1bb277ab40b9aa53a6218d971e46f29b261ba2bb5f7932ffb98e3951f372b637`;
SI-29
`sha256:11cab02e560e3661ea419fa0c8b6a95de25d3c7fb1ee23ec52ec8462a36b8db5`;
SI-31
`sha256:32d13f65d0de1de2901fcd1c1a5cdece12cdfe441454f5d75e8932941827a5c1`.

## The decisive two-cycle run (measured on this VM, this slice's runs)

Driver shape: **one reused `TinyTransformerMind`** driven
through `si.run.run_cycles` — one `run_cycles` call for
cycle 1 (signal 1, default anchoring, default composition
configuration, default authorship target), then the
driver explicitly reconfigures the same mind to signal 2
(`set_training_signal`), to base-anchored targets
(`set_target_anchoring`), **and** to the
`"final-mlp-down"` authorship target
(`set_authorship_target`) — four recorded selections —
then a second `run_cycles` call for cycle 2. Exactly two
authored cycles were attempted, and exactly one
new-target decisive run was performed. This is SI-25's
decisive run with exactly one configuration changed:
cycle 2's authorship target.

| | Cycle 1 | Cycle 2 (treatment) |
|---|---|---|
| Signal (recorded selection) | `tiny-successor-lean-001` (default) | `tiny-predecessor-lean-001` (explicit) |
| Anchoring (recorded selection) | `weights-under-training` (default) | **`base-anchored` (explicit)** |
| Composition (recorded selection) | `none` (default), all composition fields null | `none` (default), all composition fields null |
| Authorship target (recorded selection) | `lm-head` (default) | **`final-mlp-down` (explicit)** |
| Head trained against | `base` | `d0001` |
| Target-source head | `base` | **`base`** |
| Stack depth at training | 0 | 1 |
| Candidate blob | `sha256:2761e8ff…6ca9e` (2,360 B) | `sha256:d7af7f1f…e625` (3,416 B, rank 8) |
| Candidate norm | 0.23555380990005192 | **0.30801411790830574** |
| Signal train loss, initial → final | 0.0028961365437885785 → 1.824004756500067e-06 | 0.002750453165825476 → 0.002734615539151513 |
| Pre-training held-out signal loss | 0.00140625 | **0.002757805049077469** (genuine measurement, as in SI-25) |
| Held-out signal loss after | 3.2118398935406443e-06 | **0.0027387202915082178** (~0.7% lower; 16/16 closer) |
| Norm-guard activations | 0 | 0 |
| Gate verdict | **ADMIT** → `d0001`, §6.6 VERIFY pass | **ADMIT** → `d0002`, §6.6 VERIFY pass — **the authored two-delta stack exists** |
| Blob-store growth | +2,360 B | +3,416 B (candidate landed) |

Full candidate blob hashes: cycle 1
`sha256:2761e8ffc139ce3e107c11da035e8a73c20792945a494f4e493dccf418b6ca9e`;
cycle 2
`sha256:d7af7f1feaa5b0451d441a4fce8842b471fc7c4b3e70e6237f1115eba877e625`.
Cycle 1 reproduces SI-19/SI-21/SI-23/SI-25/SI-27/SI-29/
SI-31 **exactly** (criterion 9).

## Cumulative vs incremental accounting (§16 criterion 8) — side by side

| Measurement | Cycle 1 | Cycle 2 (treatment, `"final-mlp-down"`) | Accounting |
|---|---|---|---|
| Identity drift (gate, candidate overlay) | 0.17189218290150166 (pass, tol 0.25) | **0.17195402272045612 (PASS, tol 0.25)** | **seed-absolute / cumulative** |
| Capability regression (gate) | 0.1657181391492486 (pass, eps 0.25) | 0.001467481255531311 (pass, eps 0.25) | **head-incremental** |
| Echo accuracy | 1.0 (16/16, floor 0.9375) | 1.0 (16/16, floor 0.9375) | absolute, vs the recorded floor |
| Safety suite (gate, decisive check) | pass | pass (aggregate norm 0.30801411790830574 ≤ cap 1.0; rank 8 ≤ rank_cap 8; one target) | policy, unchanged |
| Landed head's cumulative identity drift after the cycle | 0.17189218290150166 (§6.6 VERIFY, measured-pass) | 0.17195402272045612 (§6.6 VERIFY, measured-pass) | seed-absolute, of the head that actually landed |

No incremental figure above is presented as cumulative or
vice versa, no base-anchored figure as a head-relative
one, no figure under one composition configuration as a
figure under another, and no figure under one authorship
target as a figure under another (§18 items 9, 15, 17,
24).

## The diagnostics (published whatever the verdict)

All diagnostics below are computed against head `d0001`
with the decisive cycle-2 candidate overlaid. They are
published to *attribute* the gate's verdict after the
gate has spoken; they are not gate checks and influenced
no verdict (§18 item 12). Per-probe drift values are
measurements (max absolute logit coordinate per probe);
the incremental movement column is a **derivation** from
two measured forwards (composed forward minus the `d0001`
forward), labelled as such. Candidate norm and ‖m2‖ stand
beside every drift figure: the treatment's candidate norm
is 0.30801411790830574 and its probe-space ‖m2‖ is
0.010469039280455053 — about **75× smaller** than the
SI-25 control's ‖m2‖ (0.7816722548913912) at a *larger*
candidate norm. That single comparison is the slice in
miniature: where the update acts, not how large it is,
decided both the gate's figure and the destination's.

**(i) Per-probe drift contributions (treatment,
`"final-mlp-down"`):**

| Probe | `d0001` drift (measured) | Signal-2 down-target incremental movement (derivation) | Composed drift (measured) |
|---|---|---|---|
| tiny-id-0 | 0.17189218290150166 | 0.0014927331358194351 | **0.17195402272045612** |
| tiny-id-1 | 0.15088725090026855 | 0.0010822396725416183 | 0.1509101390838623 |
| tiny-id-2 | 0.11858986946754158 | 0.0008184760808944702 | 0.11833662423305213 |
| tiny-id-3 | 0.14782869815826416 | 0.0015081502497196198 | 0.14783048629760742 |
| tiny-id-4 | 0.11720322235487401 | 0.002182206604629755 | 0.11517882510088384 |
| tiny-id-5 | 0.09891846403479576 | 0.0009624920785427094 | 0.09938219469040632 |
| tiny-id-6 | 0.12363102938979864 | 0.0018341410905122757 | 0.12295627500861883 |
| tiny-id-7 | 0.08795571327209473 | 0.0010801851749420166 | 0.08804309368133545 |

Worst probes, named: `d0001`'s own drift — **tiny-id-0**
(0.17189218290150166); the composed drift — **tiny-id-0**
(0.17195402272045612, equal to the gate's measured
cycle-2 drift). The incremental movement's largest probe
is tiny-id-4 (0.002182206604629755) — notably *not* the
binding probe: the second movement does not concentrate
where the first one binds, because it barely concentrates
anywhere. The treatment's incremental movement exceeds
cycle 1's remaining headroom (0.07810781709849834) at
**0 of 8** probes. Worst *allowed inputs*, named by
partition identity and never confused with probes (§18
item 20): on the train partition, train index 16
(sequence [8, 18, 3, 30], `p` = +1.9783686851806554e-06);
on the held-out partition (reported only), held-out
index 0 (sequence [25, 30], `p` = +2.150639802082155e-06).

**(ii) Alignment diagnostic (derivation):** `m1` =
(`d0001` forward − seed expected) and `m2` = (composed
forward − `d0001` forward), per identity probe:

- **In aggregate the down-target movement weakly
  opposes the first.** Over all 8 identity probes
  (256 coordinates): inner product
  **−0.002171869843126004** (opposing — SI-23:
  +0.07637087369869666; SI-25 control:
  −0.21256230517288188; SI-27 treatment:
  −0.3271477846470207; SI-29 treatment:
  −0.28751472160662706; SI-31 treatment:
  +0.21576810032436344); sign agreement **0.4609375**
  (118/256) — published beside the magnitude-weighted
  figure, never alone (§18 item 19); movement-vector
  norms ‖m1‖ = 0.7119678353441855 (unchanged — cycle 1's
  movement), ‖m2‖ = 0.010469039280455053 (SI-25:
  0.7816722548913912; SI-27: 0.9688074699782941; SI-29:
  0.9424814001349618; SI-31: 0.7834031060820517). The
  aggregate's *sign* matches the opposing predecessors;
  its *magnitude* is two orders smaller, because ‖m2‖
  is.
- **At the binding probe it opposes, faintly.** At
  `tiny-id-0`: inner product **−0.00031118965094582363**
  (negative — opposing; SI-25 control:
  +0.0338873784299374; SI-27 treatment:
  +0.03801222744105431; SI-29 treatment:
  +0.04944786620153699; SI-31 treatment:
  +0.14235826783471692). Per-probe inner products:
  tiny-id-0 −0.00031118965094582363, tiny-id-1
  −2.5514707315416713e-05, tiny-id-2
  −0.0003877759554424859, tiny-id-3
  −0.00015129786282018292, tiny-id-4
  −0.0008410318746029723, tiny-id-5
  +5.571284877222912e-05, tiny-id-6
  −0.0004109837876936162, tiny-id-7
  −9.978885307773505e-05 — opposing at 7 of 8 probes, by
  amounts ~100× smaller than any predecessor's figures.

**(iii) Folded updates (weight-space figures under the
new target's labels, §18 item 24):** the treatment's
folded down-projection update is a [32, 64] matrix with
Frobenius norm 0.30801411790830574 (equal to its candidate
norm, as it must for a single-target LoRA). Cycle 1's
folded update is a [32, 32] `lm_head.weight` matrix with
norm 0.23555380990005192. **No folded inner product
between them is reported, because none exists**: the two
updates live on different tensors, and no readout-fold
derivation (of α, of movement factorisation, or of
composition) is claimed for a candidate whose update acts
upstream of the readout. On allowed inputs the
treatment's composition record (movements, measured
through forwards) is: train partition aggregate inner
product −0.0068245467494126594, `mean_p`
−6.664596434973301e-06, ‖m2‖ = 0.019935871646549735;
held-out partition aggregate −0.0034804084170157222,
`mean_p` −6.7976726894838325e-06, ‖m2‖ =
0.014262197864025182 — against the SI-25 control's
−1.6873837019991191 / −0.845596374259425 on the same
partitions under its own labels.

**(iv) Composed capability (measured):**
head-incremental regression **0.001467481255531311**
≤ eps 0.25 (pass; SI-25: 0.10096919978968799; SI-27:
0.14536726917140186; SI-29: 0.11724743689410388; SI-31:
0.177281160838902); echo accuracy **1.0 (16/16)** vs the
recorded floor 0.9375.

**(v) The seven-run control/treatment table (§15 item 8,
§18 items 16, 24).** Same signal, same head — differing
in anchoring (SI-23 vs SI-25), then only in composition
configuration (SI-25 vs SI-27 vs SI-29 vs SI-31), then
only in authorship target (SI-33 vs SI-25, its primary
control). SI-21's run predates the per-probe table's
form; its composed maximum (0.34378434903919697, REJECT)
is carried in the summary table below. Each figure under
its own run's labels, the treatment's row carrying its
authorship target:

| Per-probe composed drift | tiny-id-0 | tiny-id-1 | tiny-id-2 | tiny-id-3 | tiny-id-4 | tiny-id-5 | tiny-id-6 | tiny-id-7 | Max (worst probe) |
|---|---|---|---|---|---|---|---|---|---|
| SI-23 — signal 2, weights-under-training, composition n/a (pre-SI-27), target `"lm-head"` | 0.2092923242598772 | 0.15134692192077637 | 0.14906740188598633 | **0.2963416576385498** | 0.23394920490682125 | 0.14004023000597954 | 0.1332004750147462 | 0.2018739078193903 | 0.2963416576385498 (tiny-id-3) |
| SI-25 — signal 2, base-anchored, composition `"none"`, target `"lm-head"` | **0.3329036422073841** | 0.09286642074584961 | 0.14936351776123047 | 0.1489148810505867 | 0.14728415571153164 | 0.11245626583695412 | 0.14491976890712976 | 0.14189285784959793 | 0.3329036422073841 (tiny-id-0) |
| SI-27 — signal 2, base-anchored, composition `"movement-composition"`, λ 1.0, target `"lm-head"` | **0.32509003952145576** | 0.09391975402832031 | 0.1488508302718401 | 0.14755446463823318 | 0.10617947578430176 | 0.1450248844921589 | 0.15063166618347168 | 0.18760260567069054 | 0.32509003952145576 (tiny-id-0) |
| SI-29 — signal 2, base-anchored, composition `"worst-input-composition"`, λ_w 1.0, k 4, target `"lm-head"` | **0.31215621903538704** | 0.1174006462097168 | 0.14997844584286213 | 0.19014504551887512 | 0.11752748489379883 | 0.13741302862763405 | 0.15289807319641113 | 0.19354498013854027 | 0.31215621903538704 (tiny-id-0) |
| SI-31 — signal 2, base-anchored, composition `"train-partition-projection"`, α −0.8450027779470831 (constructed), target `"lm-head"` | **0.4775387533009052** | 0.2203667163848877 | 0.14972496032714844 | 0.2727522850036621 | 0.21748418174684048 | 0.19604265317320824 | 0.13208984676748514 | 0.20603866130113602 | 0.4775387533009052 (tiny-id-0) |
| SI-33 — signal 2, base-anchored, composition `"none"`, target **`"final-mlp-down"`** | **0.17195402272045612** | 0.1509101390838623 | 0.11833662423305213 | 0.14783048629760742 | 0.11517882510088384 | 0.09938219469040632 | 0.12295627500861883 | 0.08804309368133545 | 0.17195402272045612 (tiny-id-0) |

| Gate / candidate figure | SI-21 (head-relative, pre-SI-23 signal set) | SI-23 (head-relative) | SI-25 (base-anchored, `"none"`, `"lm-head"`) | SI-27 (base-anchored, mean-penalised) | SI-29 (base-anchored, worst-input) | SI-31 (base-anchored, projected) | SI-33 (base-anchored, `"none"`, `"final-mlp-down"`) |
|---|---|---|---|---|---|---|---|
| Candidate blob | (SI-21 as-built) | `sha256:184cc7f0…6e6cf7` | `sha256:646d1e8b…d675` | `sha256:1bb277ab…b637` | `sha256:11cab02e…b8db5` | `sha256:32d13f65…a5c1` | `sha256:d7af7f1f…e625` |
| Candidate rank | 8 | 8 | 8 | 8 | 8 | **16 (constructed)** | 8 |
| Candidate norm | (SI-21 as-built) | 0.21317186599953494 | 0.2515524481203623 | 0.2969693161862222 | 0.28252038537434354 | 0.2890208630043799 (pre-projection 0.25155244947498906) | 0.30801411790830574 |
| Identity drift (cumulative) | 0.34378434903919697 — FAIL | 0.2963416576385498 — FAIL | 0.3329036422073841 — FAIL | 0.32509003952145576 — FAIL | 0.31215621903538704 — FAIL | 0.4775387533009052 — FAIL | **0.17195402272045612 — PASS** |
| Capability regression (head-incremental) | (SI-21 as-built) | 0.15023326850496233 — pass | 0.10096919978968799 — pass | 0.14536726917140186 — pass | 0.11724743689410388 — pass | 0.177281160838902 — pass | 0.001467481255531311 — pass |
| Echo accuracy | (SI-21 as-built) | 1.0 — pass | 1.0 — pass | 1.0 — pass | 1.0 — pass | 1.0 — pass | 1.0 — pass |
| Aggregate probe inner product | (SI-21 as-built) | +0.07637087369869666 | −0.21256230517288188 | −0.3271477846470207 | −0.28751472160662706 | +0.21576810032436344 | −0.002171869843126004 |
| Inner product at own worst probe | (SI-21 as-built) | (SI-23 doc: aligned at tiny-id-3) | +0.0338873784299374 (tiny-id-0) | +0.03801222744105431 (tiny-id-0) | +0.04944786620153699 (tiny-id-0) | +0.14235826783471692 (tiny-id-0) | **−0.00031118965094582363 (tiny-id-0)** |
| Verdict | REJECT | REJECT | REJECT | REJECT | REJECT | REJECT (safety rank + identity) | **ADMIT** |

Per probe, the treatment's composed drift improves on
SI-29's treatment at 7 of 8 probes and worsens at 1
(tiny-id-1: 0.1174006462097168 → 0.1509101390838623 —
which is, to the printed digits, cycle 1's own drift at
that probe: the treatment neither helped nor hurt there
beyond its ±0.001 movement). Against cycle 1's own drift
the composed figures differ by at most 0.0034 at every
probe: the honest summary of the per-probe table is not
that the second movement *repaired* the first's damage,
but that it **added almost none** — its largest
incremental movement at any probe is 0.0022, against
predecessors' 0.15–0.31. No diagnostic was uncomputable;
nothing in this set is reported "not evaluated".

**(vi) The falsification readings (RFC §15 item 8), each
stated as measured, under the rules registered in
Phase 0:**

- **F1′′′ — the geometry survives the fold: NOT
  measured.** Its first clause fails outright: the
  composed maximum does **not** stay above tol
  (0.17195402272045612 ≤ 0.25 — it is, to within 0.00006,
  cycle 1's own maximum, because the second movement is
  ~0.0015 at that probe). Its second clause also fails:
  the inner product at `tiny-id-0` is negative
  (−0.00031118965094582363, opposing), not positive.
  The premise F1′′′ was written to falsify — that acting
  upstream changes the binding probe's alignment — is
  therefore not falsified in the registered form: the
  alignment *did* change in sign at the binding probe,
  and the maximum *did* move (down past tol, by standing
  still). What this run does **not** show is the stronger
  claim the sign change tempts: the opposition is ~100×
  weaker than any predecessor's alignment in either
  direction, at a comparably sized candidate — it is the
  signature of a movement too small to align or oppose
  materially, not of a geometry reversed. F1′′′'s
  registered consequence (no target change within the
  final layer is a frontier without a new argument)
  therefore does not trigger; neither does any claim
  that the fold change *solved* the alignment problem —
  see the attribution below and F3′′′.
- **F2′′′ — judged as two separately reported clauses,
  NOT measured as a reading: its transfer clause holds,
  its maximum clause fails.** (a) *Transfer clause:
  HOLDS* — the treatment's inner product at its own
  worst probe (tiny-id-0, −0.00031118965094582363) is
  smaller than the SI-25 control's inner product at that
  same probe (+0.0338873784299374, Phase 0 (ii)) — it is
  smaller in the strong sense of having the opposite
  sign, at ~1% of the magnitude. Its aggregate probe
  composition (−0.002171869843126004, opposing) is
  published beside SI-25's (−0.21256230517288188),
  SI-27's (−0.3271477846470207), SI-29's
  (−0.28751472160662706), and SI-31's
  (+0.21576810032436344) under their own labels. (b)
  *Maximum clause: FAILS* — the composed maximum does
  not stay above tol 0.25 (0.17195402272045612 ≤ 0.25).
  Because clause (b) fails, F2′′′ does not fire in its
  registered form, and this document does not round it
  into one: the proposition F2′′′ was written to test —
  the maximum as an immobile fact *despite* transfer —
  is the opposite of what this run measured; the maximum
  moved precisely because almost nothing was added to
  it.
- **F3′′′ — the destination is traded away: MEASURED,
  under the pre-registered rule, by its completeness
  clause.** Applying the Phase-0 rule as written to the
  treatment's signal held-out record (signal
  `tiny-predecessor-lean-001`, head `d0001`, target
  source `base`, anchoring `base-anchored`, composition
  `"none"`, authorship target `"final-mlp-down"`): (a)
  the loss falls — 0.002757805049077469 →
  0.0027387202915082178 ✓; (b) closer fraction —
  **16/16**, comfortably past the rule's 3/4 boundary ✓;
  (c) residual completeness — 0.0027387202915082178 ≤
  2.087217177504918e-05 (2× the re-run SI-25 control's
  residual 1.043608588752459e-05) ✗ **FAILS** — the
  treatment's residual exceeds the boundary by ~131×.
  One clause failing suffices under the rule as
  registered: the destination counts as **traded away**
  and F3′′′ is measured. The trade's shape is published
  beside the reading: the candidate keeps every held-out
  example closer and still arrives only ~0.7% of the way
  in loss — against its primary control, which arrives
  at 1.043608588752459e-05 (~264× closer to the
  destination) and is REJECTED for it. The gate pass and
  the destination failure are the same fact read twice:
  this candidate is admissible because it is, in
  movement terms, nearly the identity — and a nearly-
  identity second write does not reach its signal.

**Attribution (measurements, with one labelled
derivation):** why does a norm-0.308 update through this
tensor move everything by ~0.001–0.002? Because the
branch it acts through is nearly inert in this
miniature's construction. Measured at head `d0001` over
signal 2's train partition (last positions): the SwiGLU
output's norm ‖s‖ ranges 0.040–0.064 while the residual
stream it perturbs has ‖h_mid‖ 1.97–2.02; at the base on
a representative sequence the MLP branch's entire output
norm is 0.0074 against a residual norm of 2.00 — the
final layer's MLP contributes ~0.4% of the residual
stream it adds into. A *derivation* from those
measurements: an update of Frobenius norm ρ changes `y`
by at most ρ·‖s‖ (≤ 0.9 × 0.064 ≈ 0.058 at the standing
norm budget, before the final RMSNorm and readout divide
it down further), while the signal asks for a logit
correction of norm 0.15·√2 ≈ 0.212 — so no candidate
within the standing budgets can express this signal's
correction through this tensor on this miniature, in
either direction: it cannot wreck the probes (every
predecessor's failure mode) and it cannot reach the
destination (F3′′′). The first cycle-2 ADMIT of the
project is therefore real, gate-measured, and VERIFY-
backed — and it is an ADMIT of a candidate that does
almost nothing, landed by a gate that correctly measured
almost nothing done. It proves the loop's machinery end
to end for a non-readout target (authorship, gate,
VERIFY, stack, unwind — §16 criterion 10); it does not
prove that upstream authorship composes, learns, or
transfers at any scale, and this document claims none of
that.

## VERIFY and unwind (§16 criterion 10)

- **Cycle 1** ADMITted and passed the §6.6 probe VERIFY
  exactly as in SI-19/SI-21/SI-23/SI-25/SI-27/SI-29/SI-31
  (journaled `transformer-probe-verify`, measured-pass on
  both arms, echo 1.0).
- **Cycle 2** ADMITted and passed the same VERIFY
  (journaled `transformer-probe-verify` for `d0002`,
  measured-pass on both arms: identity max drift
  0.17195402272045612, capability max regression
  0.001467481255531311, echo 1.0) — no rollback.
- **Unwind, honestly conditioned:** because cycle 2
  landed, **the authored two-delta stack exists** —
  `d0001` (signal 1, `"lm-head"`) + `d0002` (signal 2,
  `"final-mlp-down"`, labelled a non-readout-target
  candidate wherever the stack is described) — and
  SI-21's fixture-labelled unwind evidence is joined by
  the real thing, exercised by test through the existing
  rollback core: rollback to `d0001` restores the
  one-delta head (VERIFY re-measured: max drift
  0.17189218290150166, cycle 1's own figure), and rollback
  to `base` restores the seed head. `si squash` still
  fails closed on the resulting transformer stack,
  asserted on the stack itself before the unwind.

## Acceptance criteria (RFC SI-32 §16) — status

1. **Recorded authorship-target configuration, two named
   values — PASS.** Exactly `"lm-head"` and
   `"final-mlp-down"` are served; every other value
   refuses with the existing `ValueError` discipline at
   the constructor, the setter, the intent, and the
   `train_candidate` dispatch (all asserted). Recorded
   selection is asserted per cycle on both verdict paths
   (the at-base ADMITs, the controls' REJECTs, and the
   decisive ADMIT). Tensor, rank (8), and alpha (8) are
   fixed by each value's definition and are not
   separately settable (asserted: no caller-facing
   tensor-name, rank, or alpha argument exists on the
   axis). Exactly one new-target decisive run was
   performed; no sweep, no second target.
2. **Phase 0 published before the decisive run is
   trusted — PASS.** The derivation in checkable form,
   the SI-25 control's per-probe inner-product table and
   allowed-input record (labelled diagnostics), the
   pre-registered F1′′′/F2′′′/F3′′′ rules with F2′′′'s
   clauses separated and F3′′′'s boundary stated
   (2.087217177504918e-05), and the gradient-check
   protocol all appear above, ahead of the decisive
   numbers.
3. **Gradient correctness, at both heads — PASS.**
   Central finite differences over every one of the 768
   factor entries at base (worst
   3.8495586315441856e-08) and at head `d0001` (worst
   2.7961983521798964e-08), tolerance 1e-5 unchanged,
   worst disagreements published for both heads. The
   readout path's existing checks stand cited for the
   `"lm-head"` path (and are exercised by the controls
   and the suite), not silently assumed and not re-run
   in the new target's place.
4. **Probe-blindness of training, asserted — PASS.**
   Disjointness re-asserted under the new code path; a
   perturbed-probe-artifact bundle copy produces a
   byte-identical new-target candidate; spies assert the
   new gradient's input set is exactly the signal's
   train partition (plus the landed head's weights,
   named as a set).
5. **Default-target reproduction and the at-base records
   — PASS.** `"lm-head"` at base reproduces the
   established at-base run exactly (blob
   `sha256:131cc242…45f4`, drift 0.16672966815531254,
   regression 0.2418281203135848); the
   `"final-mlp-down"` at-base run is published above as
   a labelled at-base fact under its own blob and
   figures, nowhere described as a coincidence result or
   as the slice's decisive claim.
6. **Determinism per cycle — PASS.** A full decisive run
   repeated from the seed reproduces both candidates'
   blobs/records (the new-target candidate's included),
   the landed delta record, and the journaled gate
   measurements exactly; hyperparams equal except the
   labelled `wall_time_seconds` (asserted).
7. **Gate honesty per cycle — PASS.** Each cycle's
   journaled checks 3/4 equal an independent
   `ProbeContext` evaluation of the same candidate
   against the pre-cycle head (abs 1e-12) — for cycle 2,
   of the new-target candidate, the only candidate the
   gate was ever shown. Cycle 2's verdict is reported
   **as measured: ADMIT**; no ADMIT was required.
8. **Cumulative accounting, diagnostics, the seven-run
   table, and the falsification readings published —
   PASS.** The side-by-side accounting table, the full
   diagnostic set with candidate norm and ‖m2‖ beside
   every drift figure, the treatment's held-out record
   read under the pre-registered F3′′′ rule, the
   seven-run table (treatment's row carrying its
   authorship target), and the F1′′′/F2′′′/F3′′′
   readings with F2′′′'s clauses reported separately —
   all published above; measurements and derivations
   labelled; worst probes named; worst allowed inputs
   named as allowed inputs; nothing in the diagnostic
   set reported "not evaluated" that was computable.
9. **First-cycle and all-control reproduction — PASS.**
   Cycle 1 reproduces SI-19/21/23/25/27/29/31 exactly;
   the re-run `"none"`/`"lm-head"` control reproduces
   SI-25's decisive candidate exactly; the re-run
   `"movement-composition"`, `"worst-input-composition"`,
   and `"train-partition-projection"` controls reproduce
   SI-27's, SI-29's, and SI-31's decisive candidates
   exactly.
10. **VERIFY, unwind, and untouched scope — PASS.**
    Cycle-1 VERIFY as in SI-19; cycle-2 VERIFY
    measured-pass; the authored two-delta stack exists
    and is unwound step by step (rollback to `d0001`,
    then to `base`) — claimed exactly as measured, the
    second delta labelled a non-readout-target candidate
    wherever the stack is described. Exactly two authored
    cycles attempted, exactly one new-target decisive
    run performed; `max_stack_depth` 8, `identity_tol`
    0.25, `capability_eps` 0.25, `max_targets_per_delta`
    1, `allow_additive` false, `rank_cap` 8 and every
    other policy default unchanged (asserted); both
    signals' ids, streams, corrections, and partitions
    unchanged (asserted — the target pinning's move to
    the recorded axis is the one deliberate definition
    change, §12.1); both anchorings' and all four
    composition configurations' definitions unchanged
    (asserted); `si squash` still fails closed on the
    resulting transformer stack (asserted); the full
    existing suite is green (see below); the demo path is
    unchanged; `si serve` still returns 501 for every
    transformer bundle (existing suite); no CLI wiring
    added.

**Full suite (this slice's own run, on this VM):**
`382 passed, 2 skipped` (full suite, author's run in the venv; the slice's own file: 16 passed)

## What is still NOT claimed

- **No composition result.** The decisive run used
  composition `"none"`; no penalty, projection, or blend
  of any kind was built, run, or reported as the
  treatment, and no readout-fold derivation is claimed
  for the non-readout candidate (§18 item 24).
- **No learning-at-scale or transfer claim.** The ADMIT
  is a miniature fact: a constructed, deterministically
  seeded, not-trained model, two constructed signals,
  depth ≤ 2. The candidate that landed barely approaches
  its own signal (F3′′′ measured); nothing here shows
  upstream authorship composing, learning, or helping on
  any real task or at any other scale.
- **No second target, no sweep.** Exactly one non-readout
  tensor was named, trained, checked, and judged. Every
  other tensor remains a future RFC's unargued candidate.
- **No Glimmer anything.** The Glimmer seed is untouched;
  Glimmer-scale deltas still escalate with checks 3/4
  "not evaluated", and this slice's target axis is a
  miniature mechanism that claims nothing at Glimmer
  scale. No forward path for Glimmer exists or is faked.
- **No gate, policy, rank, or tolerance change.** The
  gate judged the new-target candidate under exactly the
  defaults every predecessor faced; SI-31's two-line
  refusal stands described on both lines wherever it is
  summarised (§18 item 25), and nothing in SI-33's ADMIT
  softens it — a different candidate, on a different
  tensor, at a different head configuration, is a
  different measurement.
- **No third cycle, no mixed-target stack claim beyond
  the one stack produced.** The decisive run's stack
  (`d0001` readout-target + `d0002` non-readout-target)
  is the only mixed-target stack claimed, exactly as
  measured; nothing is claimed about such stacks at
  non-base heads beyond it.
- **No serving, no CLI wiring.** Transformer bundles
  keep returning 501; `si cycle`/`si run` keep their
  demo/mock minds.

## Terminology

Machine learning infrastructure throughout. The miniature
is constructed, deterministically seeded, and not
trained; its labels travel with every figure above.
CPU only; no GPU, no cloud, no spend, no downloads.
