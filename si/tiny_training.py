"""SI-19: delta authorship for the miniature — real gradient-based
training of ONE LoRA target by ``TinyTransformerMind`` (RFC SI-18
§4, §7, §8).

Module-location decision (following the SI-9a/SI-15 precedent of a
new module per slice): the training signal, the analytic gradient,
and the training loop live in this NEW module,
``si/tiny_training.py``; ``si.tiny_transformer.TinyTransformerMind``
keeps its SI-15 surface and delegates ``propose`` / ``train_delta``
here. The gate, the loop, the probe harness, and the delta format
are untouched: an authored candidate is an ordinary
``construct_transformer_delta`` pair and the gate judges it exactly
as it judges a fixture.

What is trained, exactly (RFC SI-18 §4.3/§4.5, settled here with
the measurements published in docs/SI-19-delta-authorship.md):

- **One named signal**, ``tiny-successor-lean-001``. The base is
  *measurably imperfect* on it by construction: the target form is
  **target logits** (MSE, DemoMind's form — RFC §4.3 leaves the
  choice to this slice). Each example's target is the base's own
  last-position logits plus one small recorded correction,
  ``v = 0.15 * (onehot(successor) - onehot(last_token))`` with
  ``successor = (last_token + 1) % 32`` — the readout is asked to
  *lean toward the successor token*. The echo task cannot serve
  as the signal (the base's echo accuracy is already 16/16 =
  1.0, SI-14 — the RFC's echo trap); the base's held-out loss on
  THIS signal is ``2 * 0.15**2 / 32 = 0.00140625`` (measured
  through the loss below, and equal to that closed form), not 0.
- **Deterministic construction** from one dedicated RNG stream,
  ``default_rng(MINI_SEED + 11)`` — distinct from the identity
  stream (``MINI_SEED + 2``) and the capability stream
  (``MINI_SEED + 3``) recorded in ``si/tiny_seed.py``. 48
  sequences: lengths 2..8, last tokens assigned round-robin from
  8 tokens drawn from the signal stream, duplicates and any
  sequence appearing in either probe suite rejected during the
  draw (disjointness is also asserted by test). The first 32 are
  the training partition; the last 16 are **held out** — never
  trained on; improvement is claimed only there (RFC §4.3).
- **One LoRA target**: ``lm_head.weight`` — chosen because it is
  probe-measurable (it is not in the derived blind set, which is
  exactly the two ``k_proj.bias`` tensors), because policy caps
  already fit it (rank 8 = ``rank_cap``), and because its
  gradient is *exactly* derivable: the readout is the final
  operation, so no gradient with respect to any other tensor is
  ever needed and the derivation has no downstream chain to get
  wrong (RFC §4.4 route (i)). Rank 8, alpha 8 (scale 1.0).
- **Analytic gradients, finite-difference checked** (route (i)).
  With ``W_eff = W + scale * B @ A`` and per-example loss
  ``mean_v (logits_v - target_v)**2`` at the last position,
  ``d loss / d logits = 2 * (logits - target) / 32``,
  ``G = d loss / d W_eff = outer(d loss / d logits, h_final)``
  averaged over examples, and ``d/dB = scale * G @ A.T``,
  ``d/dA = scale * B.T @ G``. ``tests/test_tiny_training.py``
  checks EVERY factor entry against central finite differences
  of the same loss computed through the existing SI-15 forward;
  the measured worst agreement and the pinned tolerance are
  published in the as-built doc (§8.3).
- **Training**: full-batch gradient descent, learning rate 1.0,
  2000 steps, factors initialised from a dedicated stream
  (``default_rng(MINI_SEED + 17)``, ``0.05 * standard_normal``,
  A drawn before B). A norm guard projects the scaled update
  back to ``||scale * B @ A||_F <= 0.9`` whenever a step would
  exceed it (the policy caps are 1.0); activations of the guard
  are counted and recorded, never silent. Arithmetic mirrors
  SI-15: float32 storage widened to float64 for compute.
  Training runs against the bundle's applied weights at its
  current head (base + active chain, via apply-at-load); every
  SI-19 measurement is at head = base.
- **Refusal is part of authorship** (RFC §8.4/§8.6): an unknown
  ``training_signal_id``, a declared target list other than the
  one trained target, or a rank/alpha other than the pinned
  pair raise ``ValueError`` — never a silently retargeted or
  resized candidate; and a trained candidate whose held-out
  loss does not improve on the base's is refused (raised), not
  returned.

Honest labels: a training loss is a training number (RFC SI-18
§10) — it is labelled with the signal id wherever it is recorded
and is never a gate or probe result. Authorship is not
admission: the gate's verdict on an authored candidate is
reported as measured, whatever it is. CPU only, existing
dependencies only (numpy), no autograd framework (RFC §4.4
route (iii) rejected). This is machine learning infrastructure.

SI-21 (RFC SI-20 §4.3.2/§9.2) — sequential authorship semantics,
named here before any second cycle trained:

- **Signal definition unchanged.** Same id
  (``tiny-successor-lean-001``), same stream
  (``MINI_SEED + 11``), same correction scale, same partitions.
  SI-21 changes *which head trains*, never what the signal is.
- **Target anchoring: weights-under-training (head-relative).**
  ``build_training_signal(weights)`` derives each example's
  target from the weight set it is given — the bundle's applied
  weights at the head under training — plus the recorded
  successor-lean correction. At head = base this is exactly
  SI-19's definition. At cycle >= 2 the readout is therefore
  asked to lean a further correction-step from where the head
  already leans; the target SOURCE moves with the head. The
  alternative — **base-anchored** targets (derived from the
  seed base's logits at every cycle, later cycles correcting
  the residual toward a fixed destination) — was considered
  and NOT chosen: changing the signal definition in the same
  slice that tests composition would confound the experiment
  (RFC SI-20 §4.3.2). The choice is recorded in code as
  :data:`TARGET_ANCHORING` and in every candidate's
  hyperparams (``target_anchoring``, ``head``,
  ``target_source_head``, ``stack_depth``), so a head-relative
  number can never be mistaken for a base fact (RFC SI-20
  §12.8): cycle 2's ``base_held_out_loss`` field — kept under
  its SI-19 name for continuity — is the loss of the *weights
  under training* (at its head) against targets derived from
  those same weights, and the as-built doc says so wherever
  the figure appears.

SI-23 (RFC SI-22 §§5, 9, 10) — a second authored signal:

- **Registry/dispatch.** ``train_candidate`` dispatches on
  ``intent.training_signal_id`` through :data:`SIGNAL_REGISTRY`,
  which serves exactly two signals and refuses every other id
  with the existing ``ValueError`` discipline. Foreign-target
  and foreign-rank/alpha refusals apply per signal, unchanged.
  Selection is explicit and recorded: ``TinyTransformerMind``
  is configured with the signal it will propose (constructor
  choice, default signal 1); there is no autonomous, learned,
  depth-driven, or alternating selection.
- **Signal 2: ``tiny-predecessor-lean-001``.** Same form
  (target-logits MSE), same target (``lm_head.weight``), rank
  8, alpha 8, 2,000 steps, lr 1.0, norm budget 0.9, correction
  scale 0.15 — differing only in correction direction, stream,
  and drawn sequences: ``v = 0.15 * (onehot(predecessor) -
  onehot(last_token))`` with ``predecessor = (last_token - 1)
  % 32``. Its sequences stream is ``MINI_SEED + 23``
  (:data:`SIGNAL2_SEED_OFFSET`) and its factor-init stream is
  ``MINI_SEED + 29`` (:data:`INIT2_SEED_OFFSET`) — neither
  reuses +2, +3, +11, or +17. Its rejection draws reject
  against the union of both probe suites, signal 1's full 48
  sequences, and its own draws so far (RFC §5.4).
- **Anchoring unchanged.** Signal 2 inherits
  :data:`TARGET_ANCHORING` unchanged; its pre-training
  held-out loss at any head is the closed form
  ``2 * 0.15**2 / 32 = 0.00140625`` by construction under
  weights-under-training anchoring (targets derive from the
  weights under training), labelled with its head wherever
  it appears — not fresh evidence about the base.

SI-25 (RFC SI-24 §§5, 9, 10) — base-anchored targets:

- **Anchoring as a recorded configuration (§5.3).** The mind
  is configured with the anchoring it will train under
  (constructor-level ``target_anchoring`` on
  ``TinyTransformerMind``, defaulting to
  :data:`TARGET_ANCHORING` = ``"weights-under-training"``),
  settable between cycles by the explicit driver call
  ``set_target_anchoring(...)``. The value is validated
  against exactly :data:`TARGET_ANCHORINGS` by
  :func:`validate_target_anchoring` (any other value refuses
  with the existing ``ValueError`` discipline), carried on
  the intent (``Intent.target_anchoring``), dispatched on by
  :func:`train_candidate`, and recorded in every cycle's
  hyperparams as ``target_anchoring`` / ``target_source_head``.
  :data:`TARGET_ANCHORING` survives as the *default's* name,
  not as a description of every cycle. No autonomous,
  learned, depth-driven, or verdict-driven anchoring
  selection exists.
- **Only the target-source role splits (§5.2/§5.4).**
  ``train_candidate`` still assembles the weights under
  training at the cycle's verified manifest head — hidden
  states, the readout under training, candidate construction,
  the gate, and VERIFY all use that set, unchanged. Under
  ``"base-anchored"`` the signal's *targets* alone derive
  from a second set: the frozen seed base, assembled by
  ``si.tiny_transformer.assemble_base_weights`` from the
  same verified manifest with its active chain truncated to
  empty, through hash-verified apply-at-load reads, with no
  bytes copied or stored and the manifest never mutated. A
  failed base assembly refuses training — there is NO
  fallback to head-derived targets. Under
  ``"weights-under-training"`` the target source IS the
  weights under training, exactly as before (byte-for-byte).
- **Echo-trap records change character (§5.5).** Under
  base-anchoring a non-base head's pre-training held-out
  loss is the head's loss against *base-derived* targets —
  a genuine measurement, never the closed form
  ``0.00140625`` (which holds only where the two weight sets
  coincide: at head = ``base``, under either anchoring).

SI-27 (RFC SI-26 §§7, 14, 15) — movement-composition-aware
training:

- **Composition as a recorded configuration (§7.4).** The
  mind is configured with the composition configuration it
  will train under (constructor-level ``composition_config``
  on ``TinyTransformerMind``, defaulting to
  :data:`DEFAULT_COMPOSITION_CONFIG` = ``"none"``), settable
  between cycles by the explicit driver call
  ``set_composition_config(...)``. The value is validated
  against exactly :data:`COMPOSITION_CONFIGS` by
  :func:`validate_composition_config` (any other value
  refuses with the existing ``ValueError`` discipline),
  carried on the intent (``Intent.composition_config``),
  dispatched on by :func:`train_candidate`, and recorded in
  every cycle's hyperparams as ``composition_config`` /
  ``composition_lambda`` beside the penalty's measured
  values. No autonomous, learned, depth-driven, or
  verdict-driven configuration selection exists, and the
  weight is NOT independently settable: ``λ = 1.0``
  (:data:`COMPOSITION_LAMBDA`) is part of the
  ``"movement-composition"`` configuration's definition
  (§7.2(ii) — fixed by parity of units, in advance, never
  swept).
- **Exactly one added term (§7.2).** Under
  ``"movement-composition"`` the loss is ``signal_MSE +
  λ · mean_x p(x)`` over the signal's 32 TRAINING sequences
  only, where ``p(x) = (1/32) Σ_c m1_c(x) · m2_c(x)``,
  ``m1(x)`` is the landed head's realised movement at ``x``
  (landed-head last-position logits minus the frozen base's,
  both through the existing forwards, computed once —
  constants during optimisation), and ``m2(x)`` is the
  candidate's movement at ``x`` (candidate-head minus
  landed-head last-position logits). No probe input, probe
  logit, probe-derived quantity, or held-out sequence
  enters the loss, the penalty, factor initialisation,
  early stopping, or candidate selection (invariant 9);
  the penalty's value on the held-out partition is
  *reported only* (initial and final), never optimised.
  Under ``"none"`` the training path is byte-for-byte
  SI-25's: no base assembly is performed for the penalty
  and no penalty quantity is computed.
- **Gradient shape (§7.3).** The penalty is linear in the
  folded update: ``m2(x) = ΔW · h_final(x)`` over the same
  cached hidden states the signal loss reads, so its
  gradient is the fixed matrix ``mean_x (1/32) ·
  outer(m1(x), h_final(x))`` carried through the same
  readout fold as the signal gradient — no new backward
  machinery. At head = ``base``, ``m1`` is identically zero,
  the penalty and its gradient vanish, and penalised
  training coincides with SI-25's unpenalised training
  byte-for-byte (measured by test, not asserted).

SI-29 (RFC SI-28 §§6, 14, 15) — worst-input composition-aware
training:

- **A third recorded composition configuration (§6.4).**
  :data:`COMPOSITION_CONFIGS` gains exactly
  ``"worst-input-composition"`` beside ``"none"`` and
  ``"movement-composition"``; the same constructor / setter /
  intent / ``train_candidate`` validation and refusal
  discipline serves all three. λ_w = 1.0
  (:data:`WORST_INPUT_LAMBDA`) and k = 4
  (:data:`WORST_INPUT_K`, top-k form chosen in Phase 0 — see
  ``docs/SI-29-worst-input-composition-training.md``) are part
  of the new value's DEFINITION, recorded in hyperparams,
  never read from an intent or caller input, and never swept.
  No autonomous, learned, depth-driven, verdict-driven, or
  probe-driven selection exists.
- **Exactly one added term (§6.2).** Under
  ``"worst-input-composition"`` the loss is ``signal_MSE +
  λ_w · W(m2)`` where ``W`` is the mean of the k largest
  per-input ``p(x)`` values over the signal's 32 TRAINING
  sequences only, ``p``/``m1``/``m2`` exactly as SI-27 defines
  them (``m1`` train partition only, constants during
  optimisation). Held-out worst-input values are REPORTED
  only. Under ``"none"`` the path is byte-for-byte SI-25's;
  under ``"movement-composition"`` byte-for-byte SI-27's. No
  probe input, probe logit, held-out sequence, or
  probe-derived quantity enters the loss under any
  configuration. The non-improvement refusal still judges
  the *signal* held-out loss alone.
- **Gradient shape (§6.3).** On any region where the top-k
  set is fixed, ``W`` is a mean of fixed per-input penalty
  terms, so its gradient is the mean of the selected inputs'
  SI-27 fixed-matrix gradients through the same readout
  fold. The selected set is a function of the factors; its
  membership at the finite-difference check points is
  reported with the check (as-built doc). At head =
  ``base``, ``m1`` is identically zero, ``W`` and its
  gradient vanish, and worst-input training coincides with
  SI-25's at-base run byte-for-byte (measured by test).

SI-31 (RFC SI-30 §§7, 14, 15) — train-partition-projection
composition:

- **A fourth recorded composition configuration (§7.3).**
  :data:`COMPOSITION_CONFIGS` gains exactly
  ``"train-partition-projection"`` beside ``"none"``,
  ``"movement-composition"``, and
  ``"worst-input-composition"``; the same constructor /
  setter / intent / ``train_candidate`` validation and
  refusal discipline serves all four. The configuration's
  definition carries no weight, no k, and no fraction: the
  projection coefficient α is MEASURED from the run's own
  candidates on the train partition and recorded in the
  hyperparams — never read from an intent or any other
  caller input, never settable, never swept. No autonomous,
  learned, depth-driven, verdict-driven, or probe-driven
  selection exists.
- **The construction (§7.2).** Under
  ``"train-partition-projection"`` the training step is
  SI-25's exactly (base-anchored signal loss under
  ``"none"`` semantics — no penalty term of either prior
  kind enters the loss). The judged candidate is then
  CONSTRUCTED, once, in full: with ``m1(x)`` / ``m2(x)``
  SI-27's movement definitions on the signal's 32 TRAINING
  sequences, α = (Σ_x ⟨m1(x), m2(x)⟩) / (Σ_x ⟨m1(x), m1(x)⟩)
  (α = 0.0 when the denominator vanishes — at head =
  ``base``, where ``m1`` is identically zero and the
  construction is the identity), and the judged candidate's
  folded update is Δ2′ = Δ2 − α·Δ1, where Δ1 / Δ2 are the
  landed head's and the trained candidate's folded
  ``lm_head.weight`` updates. The projected candidate is
  re-measured in full before it is returned (signal train
  and held-out losses, with the non-improvement refusal
  applied to the PROJECTED candidate's signal held-out
  loss; candidate norm beside its pre-projection norm; the
  allowed-input composition record on both partitions).
  The pre-projection candidate is a construction
  intermediate: it is never judged, landed, or reported as
  a run, and its figures are labelled an intermediate's
  wherever they are recorded.
- **Representation (measured, not in the RFC).** The
  difference of two rank-8 folded updates is generically
  rank 16 (measured on every Phase-0 rehearsal candidate),
  so a projected candidate with α ≠ 0 is carried by a
  rank-16 LoRA factorisation of Δ2′ (SVD factors, rank 16,
  alpha 16, scale 1.0 — the same scale as every prior
  candidate). At α = 0.0 exactly the construction is the
  identity and the candidate keeps its trained rank-8
  factors byte-for-byte. The gate is unchanged: its
  ``rank_cap`` of 8 therefore judges a rank-16 constructed
  candidate exactly as it judges any other — the as-built
  doc reports that interaction as measured, never hidden.
- **Probe-blindness (§7.2/§15 criterion 4).** No probe
  input, probe logit, held-out sequence, or probe-derived
  quantity enters the loss, the coefficient, the
  construction, or the refusal under any configuration:
  α's input set is exactly the signal's train partition
  (asserted by test through a spy on
  :func:`projection_coefficient`). No new loss exists
  under this configuration and the construction is a
  post-training step, not a differentiable path, so
  SI-25's gradient check stands for the training step
  unchanged (RFC SI-30 §14 item 6).
"""
from __future__ import annotations

import time

import numpy as np

from si.apply import fold_target
from si.bundle import _active_chain, sha256_hex
from si.tiny_seed import (
    EMBED_NAME,
    FINAL_NORM_NAME,
    LM_HEAD_NAME,
    MINI_SEED,
    POS_NAME,
    capability_inputs,
    identity_inputs,
)
from si.tiny_transformer import forward

__all__ = [
    "ALPHA",
    "CANDIDATE_CREATED_AT",
    "CORRECTION_SCALE",
    "INIT2_SEED_OFFSET",
    "INIT_SCALE",
    "INIT_SEED_OFFSET",
    "LEARNING_RATE",
    "N_HELD_OUT",
    "N_LAST_TOKENS",
    "N_TRAIN",
    "NORM_BUDGET",
    "PROJECTION_ALPHA",
    "PROJECTION_RANK",
    "RANK",
    "SIGNAL2_SEED_OFFSET",
    "SIGNAL_REGISTRY",
    "SIGNAL_SEED_OFFSET",
    "STEPS",
    "BASE_ANCHORED",
    "COMPOSITION_CONFIGS",
    "COMPOSITION_LAMBDA",
    "COMPOSITION_NONE",
    "DEFAULT_COMPOSITION_CONFIG",
    "MOVEMENT_COMPOSITION",
    "TRAIN_PARTITION_PROJECTION",
    "WORST_INPUT_COMPOSITION",
    "WORST_INPUT_K",
    "WORST_INPUT_LAMBDA",
    "TARGET_ANCHORING",
    "TARGET_ANCHORINGS",
    "TARGET_NAME",
    "TRAINING_SIGNAL_ID",
    "TRAINING_SIGNAL_ID_2",
    "analytic_gradients",
    "analytic_gradients_penalised",
    "analytic_gradients_worst_input",
    "build_training_signal",
    "composition_penalty",
    "existing_movements",
    "factor_loss",
    "final_hidden",
    "forward_with_final_hidden",
    "initial_factors",
    "movement_composition_record",
    "penalised_factor_loss",
    "projection_coefficient",
    "projection_factors",
    "worst_input_composition",
    "worst_input_composition_record",
    "worst_input_factor_loss",
    "worst_input_selection",
    "signal_loss",
    "signal_spec",
    "signal_sequences",
    "train_candidate",
    "validate_composition_config",
    "validate_target_anchoring",
]

#: The first training signal (SI-19; RFC SI-18 §4.3: a mind
#: serves exactly the signal(s) its design names).
TRAINING_SIGNAL_ID = "tiny-successor-lean-001"

#: The second training signal (SI-23; RFC SI-22 §5.6.2): the
#: directional counterpart of signal 1 — predecessor-lean.
TRAINING_SIGNAL_ID_2 = "tiny-predecessor-lean-001"

#: SI-21 target anchoring (RFC SI-20 §4.3.2), named in code:
#: targets are derived from the weights under training — the
#: bundle's applied weights at the head being trained against —
#: plus the recorded correction. The alternative (targets
#: anchored to the seed base's logits at every cycle) is NOT
#: the semantics of this signal id; see the module docstring.
TARGET_ANCHORING = "weights-under-training"

#: SI-25 (RFC SI-24 §5.3): the second named anchoring — targets
#: derived from the frozen seed base's logits at every cycle.
BASE_ANCHORED = "base-anchored"

#: SI-25: exactly the two named anchorings the configuration
#: serves. Any other value refuses (see
#: :func:`validate_target_anchoring`).
TARGET_ANCHORINGS = (TARGET_ANCHORING, BASE_ANCHORED)


def validate_target_anchoring(anchoring: str) -> str:
    """Return ``anchoring`` iff it is one of the two named
    anchorings in :data:`TARGET_ANCHORINGS`.

    Raises ``ValueError`` with the existing refusal discipline
    for any other value — the one refusal every anchoring
    dispatch path shares (constructor, setter, and
    ``train_candidate``), mirroring :func:`signal_spec`.
    """
    if anchoring not in TARGET_ANCHORINGS:
        raise ValueError(
            f"unknown target anchoring {anchoring!r}: "
            f"exactly {list(TARGET_ANCHORINGS)!r} are served "
            "(RFC SI-24 §5.3)")
    return anchoring

#: SI-27 (RFC SI-26 §7.4): the default composition
#: configuration — the unpenalised SI-25 training path.
COMPOSITION_NONE = "none"

#: SI-27: the composition-penalised configuration — the
#: signal loss gains exactly one composition-penalty term
#: (see the module docstring and :data:`COMPOSITION_LAMBDA`).
MOVEMENT_COMPOSITION = "movement-composition"

#: SI-29: the worst-input composition configuration — the
#: signal loss gains exactly one worst-input term (see the
#: module docstring and :data:`WORST_INPUT_LAMBDA` /
#: :data:`WORST_INPUT_K`).
WORST_INPUT_COMPOSITION = "worst-input-composition"

#: SI-31 (RFC SI-30 §7.3): the train-partition-projection
#: composition configuration — the training step is SI-25's
#: and the judged candidate is CONSTRUCTED by full removal
#: of the component parallel to the landed movement, as
#: measured on the signal's train partition (see the module
#: docstring). The configuration's definition carries no
#: weight, no k, and no fraction: the coefficient α is
#: measured per run, never settable.
TRAIN_PARTITION_PROJECTION = "train-partition-projection"

#: SI-27/SI-29/SI-31: exactly the named composition
#: configurations the configuration serves (two in SI-27,
#: three in SI-29, four in SI-31). Any other value refuses
#: (see :func:`validate_composition_config`).
COMPOSITION_CONFIGS = (COMPOSITION_NONE, MOVEMENT_COMPOSITION,
                       WORST_INPUT_COMPOSITION,
                       TRAIN_PARTITION_PROJECTION)

#: SI-27: the default composition configuration's name —
#: cycles train unpenalised unless the driver explicitly
#: configures otherwise (§7.4's consequence: with the
#: default ``"none"``, the unpenalised path remains
#: byte-for-byte the SI-25 path).
DEFAULT_COMPOSITION_CONFIG = COMPOSITION_NONE

#: SI-27 (RFC SI-26 §7.2(ii)): the composition penalty's
#: weight, fixed by parity of units in advance — one unit
#: of per-coordinate mean movement product weighs exactly
#: as one unit of per-coordinate mean squared target error.
#: Part of the ``"movement-composition"`` configuration's
#: DEFINITION: not a per-cycle dial, never swept in SI-27.
COMPOSITION_LAMBDA = 1.0

#: SI-29 (RFC SI-28 §6.2): the worst-input term's weight,
#: fixed by the same parity-of-units argument as
#: :data:`COMPOSITION_LAMBDA` — part of the
#: ``"worst-input-composition"`` configuration's DEFINITION:
#: not a per-cycle dial, never swept in SI-29.
WORST_INPUT_LAMBDA = 1.0

#: SI-29 (RFC SI-28 §6.2, Phase 0 form choice): the top-k
#: form's k — ``W`` is the mean of the k largest per-input
#: ``p(x)`` values over the train partition (4 of 32: large
#: enough that the gradient combines several inputs' cached
#: hidden states, small enough that the term prices the
#: tail, not the bulk). Part of the configuration's
#: DEFINITION, like :data:`WORST_INPUT_LAMBDA`.
WORST_INPUT_K = 4


def validate_composition_config(config: str) -> str:
    """Return ``config`` iff it is one of the named
    composition configurations in :data:`COMPOSITION_CONFIGS`.

    Raises ``ValueError`` with the existing refusal discipline
    for any other value — the one refusal every composition
    dispatch path shares (constructor, setter, and
    ``train_candidate``), mirroring
    :func:`validate_target_anchoring`.
    """
    if config not in COMPOSITION_CONFIGS:
        raise ValueError(
            f"unknown composition configuration {config!r}: "
            f"exactly {list(COMPOSITION_CONFIGS)!r} are served "
            "(RFC SI-30 §7.3)")
    return config

#: The one LoRA target trained per candidate (RFC §4.5).
TARGET_NAME = LM_HEAD_NAME

RANK = 8
ALPHA = 8
LEARNING_RATE = 1.0
STEPS = 2000
INIT_SCALE = 0.05

#: Target-correction scale (RFC §4.3: stated, recorded): each
#: example's target logits are the base's plus
#: ``CORRECTION_SCALE * (onehot(successor) - onehot(last))``,
#: a correction of L2 norm ``CORRECTION_SCALE * sqrt(2) ~= 0.212``.
CORRECTION_SCALE = 0.15

#: Norm guard: a training step that would push the scaled
#: update's Frobenius norm past this budget is projected back.
#: Below the policy caps (1.0) by construction (RFC §8.7).
NORM_BUDGET = 0.9

#: The signal's dedicated RNG stream offset from MINI_SEED —
#: distinct from identity (+2) and capability (+3) (RFC §4.3).
SIGNAL_SEED_OFFSET = 11

#: Factor-initialisation stream offset (DemoMind precedent: the
#: training RNG is its own stream, seed + 101 there, + 17 here).
INIT_SEED_OFFSET = 17

#: SI-23: signal 2's sequence stream offset — a NEW offset,
#: distinct from the probe offsets (+2, +3) and signal 1's pair
#: (+11, +17) (RFC SI-22 §5.3). Reusing +11 would make signal
#: 2's sequences signal 1's; reusing a probe stream would
#: collide with the suites the draws must stay disjoint from.
SIGNAL2_SEED_OFFSET = 23

#: SI-23: signal 2's factor-initialisation stream offset —
#: likewise new and distinct. Reusing +17 would start signal
#: 2's factors at signal 1's initial point.
INIT2_SEED_OFFSET = 29

#: SI-23 signal registry (RFC SI-22 §5.2): the per-signal record
#: ``train_candidate`` dispatches on. Exactly two signals are
#: served; any other id is refused. Each record carries the
#: stream offsets, the correction direction (+1 successor,
#: -1 predecessor, applied modulo the vocabulary), and the
#: pinned rank/alpha/target shared by both signals.
SIGNAL_REGISTRY = {
    TRAINING_SIGNAL_ID: {
        "signal_id": TRAINING_SIGNAL_ID,
        "seed_offset": SIGNAL_SEED_OFFSET,
        "init_offset": INIT_SEED_OFFSET,
        "direction": 1,
    },
    TRAINING_SIGNAL_ID_2: {
        "signal_id": TRAINING_SIGNAL_ID_2,
        "seed_offset": SIGNAL2_SEED_OFFSET,
        "init_offset": INIT2_SEED_OFFSET,
        "direction": -1,
    },
}


def signal_spec(signal_id: str) -> dict:
    """The registry record for ``signal_id``.

    Raises ``ValueError`` with the existing unknown-signal
    discipline for any id not in :data:`SIGNAL_REGISTRY` —
    the one refusal every dispatch path shares.
    """
    spec = SIGNAL_REGISTRY.get(signal_id)
    if spec is None:
        raise ValueError(f"unknown training signal {signal_id!r}")
    return spec

N_TRAIN = 32
N_HELD_OUT = 16
N_LAST_TOKENS = 8

#: Candidate records are proposals: the loop re-stamps id /
#: parent / self_edit_id when a candidate lands, and the journal
#: records real timestamps then. §8.5 determinism forbids a
#: wall-clock anywhere a measured value could depend on it, so
#: the proposal carries this fixed marker instead.
CANDIDATE_CREATED_AT = "2026-10-21T00:00:00Z"

_V = 32
_D = 32
_LAYERS = 2
_HEADS = 4
_HEAD_DIM = 8
_EPS = 1e-5


# -- the signal (RFC §4.3) ----------------------------------------------------

def signal_sequences(signal_id: str = TRAINING_SIGNAL_ID):
    """A signal's raw inputs, deterministically constructed.

    Returns ``(train_seqs, held_out_seqs, last_tokens)``: 32 + 16
    token-id sequences (lengths 2..8, vocabulary 32) whose last
    tokens cycle round-robin through the 8 ``last_tokens`` drawn
    first from the signal's own stream. Draws that duplicate an
    earlier sequence of the same signal or collide with ANY
    probe-suite input (identity or capability) are rejected and
    redrawn from the same stream, so the construction is
    disjoint by construction as well as by test.

    SI-23 (RFC SI-22 §5.4): signal 2's draws additionally reject
    against signal 1's full 48 sequences (32 train + 16
    held-out), so its reject set is the union of both probe
    suites, signal 1's full partition set, and its own draws
    so far — all five §5.4 obligations hold by construction.
    The default (no argument) is signal 1, byte-for-byte
    SI-19's construction.
    """
    spec = signal_spec(signal_id)
    rng = np.random.default_rng(MINI_SEED + spec["seed_offset"])
    probe_seqs = {tuple(s) for s in identity_inputs()} \
        | {tuple(s) for s in capability_inputs()}
    reject = set(probe_seqs)
    if signal_id == TRAINING_SIGNAL_ID_2:
        s1_train, s1_held, _ = signal_sequences(TRAINING_SIGNAL_ID)
        reject |= {tuple(s) for s in s1_train} \
            | {tuple(s) for s in s1_held}
    last_tokens = [int(x) for x in
                   rng.choice(_V, size=N_LAST_TOKENS, replace=False)]
    seqs: list = []
    seen: set = set()
    while len(seqs) < N_TRAIN + N_HELD_OUT:
        length = int(rng.integers(2, 9))
        seq = [int(x) for x in rng.integers(0, _V, size=length)]
        seq[-1] = last_tokens[len(seqs) % N_LAST_TOKENS]
        key = tuple(seq)
        if key in reject or key in seen:
            continue
        seen.add(key)
        seqs.append(seq)
    return seqs[:N_TRAIN], seqs[N_TRAIN:], last_tokens


def build_training_signal(weights: dict,
                          signal_id: str = TRAINING_SIGNAL_ID) -> dict:
    """The full signal against a base weight set: inputs + targets.

    ``weights`` is the TARGET SOURCE: the miniature's full
    applied weight set (the SI-15 forward's input) whose
    last-position logits the targets derive from. Under
    weights-under-training anchoring (SI-19/SI-21/SI-23) the
    caller passes the weights under training — at SI-19's only
    head, the base; at cycle >= 2, the landed head's applied
    weights. Under SI-25 base-anchoring the caller passes the
    frozen seed base's weight set instead (assembled by
    ``assemble_base_weights``); this function itself is
    anchoring-agnostic — it derives targets from whatever
    weight set it is given, and the caller records which set
    that was. Each example's target is the given weights' own
    last-position logits (through the existing forward) plus
    the signal's recorded correction — successor-lean for
    signal 1 (``(last + 1) % 32``), predecessor-lean for
    signal 2 (``(last - 1) % 32``).
    Returns ``{"train": [(seq, target_logits)], "held_out":
    [...], "last_tokens": [...]}`` with float64 target logits.
    """
    spec = signal_spec(signal_id)
    direction = int(spec["direction"])
    train_seqs, held_out_seqs, last_tokens = signal_sequences(signal_id)

    def with_targets(seqs):
        out = []
        for seq in seqs:
            base_logits = forward(seq, weights)[-1].astype(np.float64)
            correction = np.zeros(_V, dtype=np.float64)
            correction[(seq[-1] + direction) % _V] += CORRECTION_SCALE
            correction[seq[-1]] -= CORRECTION_SCALE
            out.append((list(seq), base_logits + correction))
        return out

    return {"train": with_targets(train_seqs),
            "held_out": with_targets(held_out_seqs),
            "last_tokens": list(last_tokens)}


# -- forward replica exposing the final hidden state --------------------------
# The analytic gradient needs h_final (the final RMSNorm output the
# readout multiplies). si.tiny_transformer.forward returns logits
# only, so this replica repeats its computation — same operation
# order, same float64-over-float32 discipline — and returns both.
# tests/test_tiny_training.py asserts its logits, cast to float32,
# are BIT-IDENTICAL to the library forward's on every probe and
# signal input, so the gradient is taken of the same function the
# gate measures (the SI-15 precedent: an independent path proven
# against the recorded one, never assumed equal).

def _rms_norm(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(ms + _EPS) * w


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _layer_prefix(i: int) -> str:
    return f"model.language_model.layers.{i}"


def forward_with_final_hidden(token_ids, weights: dict):
    """``(logits, h_final)`` in float64: the SI-15 forward's
    computation, additionally returning the final RMSNorm output
    at every position (the gradient needs its last row)."""
    w = {k: np.asarray(v, dtype=np.float64)
         for k, v in weights.items()}
    ids = [int(t) for t in token_ids]
    t = len(ids)
    h = w[EMBED_NAME][ids] + w[POS_NAME][:t]
    causal = np.triu(np.ones((t, t), dtype=bool), k=1)
    for i in range(_LAYERS):
        p = _layer_prefix(i)
        a = _rms_norm(h, w[f"{p}.input_layernorm.weight"])
        q = (a @ w[f"{p}.self_attn.q_proj.weight"].T
             + w[f"{p}.self_attn.q_proj.bias"]).reshape(t, _HEADS, _HEAD_DIM)
        k = (a @ w[f"{p}.self_attn.k_proj.weight"].T
             + w[f"{p}.self_attn.k_proj.bias"]).reshape(t, _HEADS, _HEAD_DIM)
        v = (a @ w[f"{p}.self_attn.v_proj.weight"].T
             + w[f"{p}.self_attn.v_proj.bias"]).reshape(t, _HEADS, _HEAD_DIM)
        q = np.transpose(q, (1, 0, 2))
        k = np.transpose(k, (1, 0, 2))
        v = np.transpose(v, (1, 0, 2))
        scores = q @ np.transpose(k, (0, 2, 1)) / np.sqrt(float(_HEAD_DIM))
        scores = np.where(causal, -np.inf, scores)
        attn = _softmax(scores) @ v                       # [H, t, dh]
        attn = np.transpose(attn, (1, 0, 2)).reshape(t, _D)
        h = h + attn @ w[f"{p}.self_attn.o_proj.weight"].T \
            + w[f"{p}.self_attn.o_proj.bias"]
        m = _rms_norm(h, w[f"{p}.post_attention_layernorm.weight"])
        gate = m @ w[f"{p}.mlp.gate_proj.weight"].T
        up = m @ w[f"{p}.mlp.up_proj.weight"].T
        silu = gate / (1.0 + np.exp(-gate))
        h = h + (silu * up) @ w[f"{p}.mlp.down_proj.weight"].T
    h_final = _rms_norm(h, w[FINAL_NORM_NAME])
    return h_final @ w[LM_HEAD_NAME].T, h_final


def final_hidden(weights: dict, seq) -> np.ndarray:
    """The last position's final hidden state for ``seq`` (float64)."""
    return forward_with_final_hidden(seq, weights)[1][-1]


# -- losses (training numbers; labelled with the signal wherever recorded) ----

def signal_loss(weights: dict, examples) -> float:
    """Mean over examples of the per-example MSE (mean over the 32
    last-position logit dims) of ``weights`` against the signal
    targets, computed through the existing SI-15 forward."""
    total = 0.0
    for seq, target in examples:
        logits = forward(seq, weights)[-1].astype(np.float64)
        diff = logits - np.asarray(target, dtype=np.float64)
        total += float(np.mean(diff * diff))
    return total / len(examples)


def _folded_weights(weights: dict, A: np.ndarray, B: np.ndarray) -> dict:
    """``weights`` with the LoRA update folded into ``lm_head`` —
    through ``si.apply.fold_target``, the one shared fold, so the
    loss a candidate is judged by in training is computed on
    exactly the tensor apply-at-load would produce (float32)."""
    target = {"name": TARGET_NAME, "kind": "lora",
              "factor_keys": ["t0.A", "t0.B"], "rank": RANK,
              "alpha": ALPHA, "base_shape": [_V, _D]}
    factors = {"t0.A": np.ascontiguousarray(A, dtype=np.float32),
               "t0.B": np.ascontiguousarray(B, dtype=np.float32)}
    out = dict(weights)
    out[TARGET_NAME] = fold_target(
        np.array(weights[TARGET_NAME], dtype=np.float32, copy=True),
        target, factors, allow_additive=False, delta_id="si-19-training")
    return out


def factor_loss(weights: dict, A: np.ndarray, B: np.ndarray,
                examples) -> float:
    """The signal loss of the candidate factors ``(A, B)``: the
    factors are cast to float32 and folded through the shared
    fold, then measured through the existing forward. This is the
    loss the finite-difference gradient check differentiates."""
    return signal_loss(_folded_weights(weights, A, B), examples)


# -- analytic gradients (RFC §4.4 route (i)) -----------------------------------

def initial_factors(signal_id: str = TRAINING_SIGNAL_ID):
    """The pinned initial factors ``(A, B)`` (float64) for a
    signal: its own dedicated init stream (signal 1: +17,
    signal 2: +29), A drawn before B, ``INIT_SCALE``-scaled.
    The default is signal 1, byte-for-byte SI-19's factors."""
    spec = signal_spec(signal_id)
    rng = np.random.default_rng(MINI_SEED + spec["init_offset"])
    A = rng.standard_normal((RANK, _D)) * INIT_SCALE
    B = rng.standard_normal((_D, RANK)) * INIT_SCALE
    return A, B


def analytic_gradients(A: np.ndarray, B: np.ndarray, examples,
                       hiddens: dict, lm_head: np.ndarray):
    """``(loss, grad_A, grad_B)`` for the signal loss, analytically.

    ``examples`` is ``[(seq, target_logits)]``; ``hiddens`` maps
    ``tuple(seq)`` to the last position's final hidden state under
    the BASE weights (the readout is the final operation, so the
    hidden state does not depend on the factors — it is computed
    once and cached by the caller); ``lm_head`` is the base
    readout (float64). The loss returned is the float64 training
    loss at these factors (the same quantity ``factor_loss``
    measures through the float32 fold, up to the fold's cast).
    """
    scale = float(ALPHA) / float(RANK)
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    W_eff = np.asarray(lm_head, dtype=np.float64) + scale * (B @ A)
    grad_W = np.zeros((_V, _D), dtype=np.float64)
    total = 0.0
    for seq, target in examples:
        h = hiddens[tuple(seq)]
        logits = h @ W_eff.T
        diff = logits - np.asarray(target, dtype=np.float64)
        total += float(np.mean(diff * diff))
        grad_W += np.outer((2.0 / _V) * diff, h)
    grad_W /= len(examples)
    grad_B = scale * (grad_W @ A.T)
    grad_A = scale * (B.T @ grad_W)
    return total / len(examples), grad_A, grad_B


# -- SI-27: the composition penalty (RFC SI-26 §7.2/§7.3) ----------------------
# m1 is a CONSTANT during a cycle's optimisation (two forwards
# of fixed weight sets, computed once — exactly as the targets
# are computed once). m2 is the folded update applied to the
# cached final hidden state, so the penalty is linear in the
# folded update and its gradient is a fixed matrix carried
# through the same readout fold as the signal gradient.

def existing_movements(head_weights: dict, base_weights: dict,
                       seqs) -> dict:
    """``m1`` per sequence (RFC SI-26 §7.2): the landed head's
    realised movement — landed-head last-position logits minus
    the frozen base's last-position logits, both through the
    existing SI-15 forward, in float64. Returns
    ``{tuple(seq): m1_vector}``. At head = ``base`` every
    vector is identically zero (the landed head IS the base)."""
    out = {}
    for seq in seqs:
        head_logits = forward(seq, head_weights)[-1].astype(np.float64)
        base_logits = forward(seq, base_weights)[-1].astype(np.float64)
        out[tuple(seq)] = head_logits - base_logits
    return out


def composition_penalty(A: np.ndarray, B: np.ndarray, seqs,
                        hiddens: dict, m1_map: dict) -> float:
    """The penalty term's value at factors ``(A, B)`` (float64,
    analytic path): ``mean_x (1/32) Σ_c m1_c(x) · m2_c(x)``
    with ``m2(x) = (scale · B @ A) @ h_final(x)`` over the
    cached hidden states. Signed by construction (§7.2(i)):
    a negative value IS aggregate opposition, in the
    diagnostic's own units (logit²)."""
    scale = float(ALPHA) / float(RANK)
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    delta_w = scale * (B @ A)
    total = 0.0
    for seq in seqs:
        m2 = delta_w @ hiddens[tuple(seq)]
        total += float(np.mean(m1_map[tuple(seq)] * m2))
    return total / len(seqs)


def analytic_gradients_penalised(A: np.ndarray, B: np.ndarray,
                                 examples, hiddens: dict,
                                 lm_head: np.ndarray, m1_map: dict,
                                 lam: float = COMPOSITION_LAMBDA):
    """``(total_loss, signal_loss, penalty, grad_A, grad_B)``
    for the composition-penalised loss (RFC SI-26 §7.2):
    ``total = signal_MSE + lam · penalty``, where the signal
    part is :func:`analytic_gradients`' loss exactly and the
    penalty part is :func:`composition_penalty`. The penalty's
    contribution to ``d loss / d W_eff`` is the fixed matrix
    ``lam · mean_x (1/32) · outer(m1(x), h_final(x))`` (§7.3);
    both parts then share the one readout-fold chain rule.
    With ``m1`` identically zero (head = ``base``) the penalty,
    its gradient, and the difference from the unpenalised
    gradients are all identically zero."""
    scale = float(ALPHA) / float(RANK)
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    W_eff = np.asarray(lm_head, dtype=np.float64) + scale * (B @ A)
    grad_W = np.zeros((_V, _D), dtype=np.float64)
    grad_pen = np.zeros((_V, _D), dtype=np.float64)
    total_signal = 0.0
    total_pen = 0.0
    for seq, target in examples:
        h = hiddens[tuple(seq)]
        logits = h @ W_eff.T
        diff = logits - np.asarray(target, dtype=np.float64)
        total_signal += float(np.mean(diff * diff))
        grad_W += np.outer((2.0 / _V) * diff, h)
        m1 = m1_map[tuple(seq)]
        m2 = (W_eff - np.asarray(lm_head, dtype=np.float64)) @ h
        total_pen += float(np.mean(m1 * m2))
        grad_pen += np.outer((1.0 / _V) * m1, h)
    n = len(examples)
    grad_W = grad_W / n + float(lam) * (grad_pen / n)
    grad_B = scale * (grad_W @ A.T)
    grad_A = scale * (B.T @ grad_W)
    signal_loss_value = total_signal / n
    penalty_value = total_pen / n
    return (signal_loss_value + float(lam) * penalty_value,
            signal_loss_value, penalty_value, grad_A, grad_B)


def penalised_factor_loss(head_weights: dict, base_weights: dict,
                          A: np.ndarray, B: np.ndarray, examples,
                          lam: float = COMPOSITION_LAMBDA) -> float:
    """The penalised loss of the candidate factors ``(A, B)``
    measured through the float32 shared fold and the existing
    forwards — the penalised counterpart of :func:`factor_loss`
    and the function the SI-27 finite-difference gradient check
    differentiates: the signal part is ``factor_loss`` of the
    weights under training against ``examples``' targets; the
    penalty part is ``lam · mean_x mean_c m1_c(x) · m2_c(x)``
    with ``m1`` = landed-head minus base forwards and ``m2`` =
    candidate minus landed-head forwards, all at last
    positions."""
    signal = factor_loss(head_weights, A, B, examples)
    cand_weights = _folded_weights(
        head_weights,
        np.ascontiguousarray(A, dtype=np.float32),
        np.ascontiguousarray(B, dtype=np.float32))
    total_pen = 0.0
    for seq, _target in examples:
        f_head = forward(seq, head_weights)[-1].astype(np.float64)
        f_base = forward(seq, base_weights)[-1].astype(np.float64)
        f_cand = forward(seq, cand_weights)[-1].astype(np.float64)
        total_pen += float(np.mean((f_head - f_base) * (f_cand - f_head)))
    return signal + float(lam) * (total_pen / len(examples))


def movement_composition_record(head_weights: dict,
                                base_weights: dict,
                                candidate_weights: dict,
                                seqs) -> dict:
    """The alignment diagnostic on ALLOWED inputs (RFC SI-26
    §7.5/§14 item 5): per-input and aggregate inner products,
    sign agreements, and movement norms for ``m1`` (landed
    head minus base) and ``m2`` (candidate minus landed head),
    under the §7.2 movement definitions — the same definitions
    the probe diagnostics use, computed here on a named,
    non-probe input set (a signal's train or held-out
    partition). Every figure is a derivation from measured
    forwards. The aggregate penalty mean (``mean_p``) is the
    penalty term's value for ``candidate_weights`` on this
    input set, in logit²."""
    per_input = []
    m1_all, m2_all = [], []
    for seq in seqs:
        f_head = forward(seq, head_weights)[-1].astype(np.float64)
        f_base = forward(seq, base_weights)[-1].astype(np.float64)
        f_cand = forward(seq, candidate_weights)[-1].astype(np.float64)
        m1, m2 = f_head - f_base, f_cand - f_head
        m1_all.append(m1)
        m2_all.append(m2)
        per_input.append({
            "seq": [int(x) for x in seq],
            "inner_product": float(np.dot(m1, m2)),
            "p": float(np.mean(m1 * m2)),
            "sign_agreement": float(np.mean(np.sign(m1) == np.sign(m2))),
            "m1_norm": float(np.linalg.norm(m1)),
            "m2_norm": float(np.linalg.norm(m2)),
            "m1_max_abs": float(np.max(np.abs(m1))),
            "m2_max_abs": float(np.max(np.abs(m2))),
        })
    a_all = np.concatenate(m1_all)
    b_all = np.concatenate(m2_all)
    return {
        "per_input": per_input,
        "aggregate_inner_product": float(np.dot(a_all, b_all)),
        "aggregate_sign_agreement": float(
            np.mean(np.sign(a_all) == np.sign(b_all))),
        "mean_p": float(np.mean([r["p"] for r in per_input])),
        "m1_norm": float(np.linalg.norm(a_all)),
        "m2_norm": float(np.linalg.norm(b_all)),
    }


# -- SI-29: the worst-input composition term (RFC SI-28 §6.2/§6.3) -------------
# W(m2) is a worst-case reduction of SI-27's p(x) over a named
# allowed population: the mean of the k largest per-input p
# values (top-k form, k = WORST_INPUT_K). On any region where
# the top-k set is fixed, W is a mean of fixed per-input terms
# and its gradient is the mean of the selected inputs' SI-27
# fixed-matrix gradients — no new backward machinery beyond
# the selection and weighting of existing per-input gradients.

def _per_input_p(A: np.ndarray, B: np.ndarray, seqs,
                 hiddens: dict, m1_map: dict) -> list:
    """Per-input ``p(x)`` values (analytic path, float64) in
    ``seqs`` order: ``p(x) = mean_c m1_c(x) · m2_c(x)`` with
    ``m2(x) = (scale · B @ A) @ h_final(x)``."""
    scale = float(ALPHA) / float(RANK)
    delta_w = scale * (np.asarray(B, dtype=np.float64)
                       @ np.asarray(A, dtype=np.float64))
    return [float(np.mean(m1_map[tuple(seq)]
                          * (delta_w @ hiddens[tuple(seq)])))
            for seq in seqs]


def worst_input_selection(A: np.ndarray, B: np.ndarray, seqs,
                          hiddens: dict, m1_map: dict,
                          k: int = WORST_INPUT_K) -> dict:
    """The worst-input selection at factors ``(A, B)`` over
    ``seqs``: per-input ``p`` values, the top-k index set
    (indices into ``seqs``, ordered by descending ``p`` with
    index tie-break — deterministic), the top-k mean ``W``,
    and the single worst input (largest ``p``) by its index
    in this population. The population is whatever the caller
    names (train partition in the loss; held-out partition
    for reported-only values) — never a probe population."""
    p_values = _per_input_p(A, B, seqs, hiddens, m1_map)
    order = sorted(range(len(seqs)), key=lambda i: (-p_values[i], i))
    topk = order[:min(int(k), len(seqs))]
    return {
        "p_values": p_values,
        "topk_indices": topk,
        "worst_index": int(order[0]) if order else None,
        "value": float(np.mean([p_values[i] for i in topk])) if topk
        else 0.0,
    }


def worst_input_composition(A: np.ndarray, B: np.ndarray, seqs,
                            hiddens: dict, m1_map: dict,
                            k: int = WORST_INPUT_K) -> float:
    """``W`` at factors ``(A, B)`` (float64, analytic path):
    the mean of the k largest per-input ``p(x)`` values over
    ``seqs`` — SI-29's worst-case reduction of SI-27's
    ``p(x)``. Signed: a negative value IS worst-case
    opposition on this population, in logit²."""
    return worst_input_selection(A, B, seqs, hiddens, m1_map, k)["value"]


def analytic_gradients_worst_input(A: np.ndarray, B: np.ndarray,
                                   examples, hiddens: dict,
                                   lm_head: np.ndarray,
                                   m1_map: dict,
                                   lam_w: float = WORST_INPUT_LAMBDA,
                                   k: int = WORST_INPUT_K):
    """``(total_loss, signal_loss, worst_value, grad_A, grad_B)``
    for the worst-input loss (RFC SI-28 §6.2):
    ``total = signal_MSE + lam_w · W``, where the signal part
    is :func:`analytic_gradients`' loss exactly and ``W`` is
    :func:`worst_input_composition` over ``examples`` (the
    train partition). The worst term's contribution to
    ``d loss / d W_eff`` is ``lam_w`` times the mean, over
    the selected top-k inputs, of SI-27's per-input fixed
    matrix ``(1/32) · outer(m1(x), h_final(x))`` (§6.3); both
    parts then share the one readout-fold chain rule. With
    ``m1`` identically zero (head = ``base``) ``W``, its
    gradient, and the difference from the unpenalised
    gradients are all identically zero."""
    scale = float(ALPHA) / float(RANK)
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    lm = np.asarray(lm_head, dtype=np.float64)
    W_eff = lm + scale * (B @ A)
    seqs = [seq for seq, _ in examples]
    selection = worst_input_selection(A, B, seqs, hiddens, m1_map, k)
    selected = set(selection["topk_indices"])
    n = len(examples)
    total_signal = 0.0
    grad_signal = np.zeros((_V, _D), dtype=np.float64)
    grad_worst = np.zeros((_V, _D), dtype=np.float64)
    for idx, (seq, target) in enumerate(examples):
        h = hiddens[tuple(seq)]
        logits = h @ W_eff.T
        diff = logits - np.asarray(target, dtype=np.float64)
        total_signal += float(np.mean(diff * diff))
        grad_signal += np.outer((2.0 / _V) * diff, h)
        if idx in selected:
            grad_worst += np.outer((1.0 / _V) * m1_map[tuple(seq)], h)
    # The signal gradient is the mean over ALL examples; the
    # worst-input gradient is lam_w times the mean over the
    # SELECTED top-k inputs only — the two means have
    # different denominators by the term's definition.
    grad_W = grad_signal / n \
        + float(lam_w) * (grad_worst / len(selected))
    grad_B = scale * (grad_W @ A.T)
    grad_A = scale * (B.T @ grad_W)
    signal_loss_value = total_signal / n
    worst_value = selection["value"]
    return (signal_loss_value + float(lam_w) * worst_value,
            signal_loss_value, worst_value, grad_A, grad_B)


def worst_input_factor_loss(head_weights: dict, base_weights: dict,
                            A: np.ndarray, B: np.ndarray, examples,
                            lam_w: float = WORST_INPUT_LAMBDA,
                            k: int = WORST_INPUT_K) -> float:
    """The worst-input loss of the candidate factors ``(A, B)``
    measured through the float32 shared fold and the existing
    forwards — the worst-input counterpart of
    :func:`factor_loss` / :func:`penalised_factor_loss` and
    the function the SI-29 finite-difference gradient check
    differentiates: the signal part is ``factor_loss`` of the
    weights under training against ``examples``' targets; the
    worst part is ``lam_w · W`` with per-input ``p`` from
    ``m1`` = landed-head minus base forwards and ``m2`` =
    candidate minus landed-head forwards, all at last
    positions, reduced by the top-k mean over ``examples``."""
    signal = factor_loss(head_weights, A, B, examples)
    cand_weights = _folded_weights(
        head_weights,
        np.ascontiguousarray(A, dtype=np.float32),
        np.ascontiguousarray(B, dtype=np.float32))
    p_values = []
    for seq, _target in examples:
        f_head = forward(seq, head_weights)[-1].astype(np.float64)
        f_base = forward(seq, base_weights)[-1].astype(np.float64)
        f_cand = forward(seq, cand_weights)[-1].astype(np.float64)
        p_values.append(float(np.mean((f_head - f_base)
                                      * (f_cand - f_head))))
    topk = sorted(p_values, reverse=True)[:min(int(k), len(p_values))]
    return signal + float(lam_w) * float(np.mean(topk))


def worst_input_composition_record(head_weights: dict,
                                   base_weights: dict,
                                   candidate_weights: dict,
                                   seqs,
                                   k: int = WORST_INPUT_K) -> dict:
    """The worst-input composition record on ALLOWED inputs:
    :func:`movement_composition_record` for the same weights
    and population, plus the worst-case reduction of its
    per-input ``p`` values — ``max_p`` (the worst allowed
    input's ``p``), its index and sequence in this population
    (an allowed input, named as one — never a probe),
    ``topk_indices`` / ``topk_mean_p`` (``W`` for this
    candidate on this population), beside the record's
    aggregates. Every figure is a derivation from measured
    forwards."""
    record = movement_composition_record(
        head_weights, base_weights, candidate_weights, seqs)
    p_values = [r["p"] for r in record["per_input"]]
    order = sorted(range(len(seqs)), key=lambda i: (-p_values[i], i))
    topk = order[:min(int(k), len(seqs))]
    record = dict(record)
    record.update({
        "k": int(k),
        "max_p": float(p_values[order[0]]),
        "max_inner_product": float(
            record["per_input"][order[0]]["inner_product"]),
        "worst_index": int(order[0]),
        "worst_seq": [int(x) for x in seqs[order[0]]],
        "topk_indices": [int(i) for i in topk],
        "topk_mean_p": float(np.mean([p_values[i] for i in topk])),
    })
    return record


# -- SI-31: the train-partition projection (RFC SI-30 §7.2) --------------------
# The projection prices nothing and adds no loss term: it
# CONSTRUCTS the judged candidate from the SI-25-trained
# candidate by removing, in full, the component of its
# folded update parallel to the landed head's folded
# update, with the parallelism measured by the movement
# inner products on the signal's TRAIN partition only.

#: SI-31: the rank a projected candidate's folded update is
#: carried at when α ≠ 0 — the difference of two rank-8
#: updates is generically rank 16 (measured; see the module
#: docstring). Its alpha is likewise 16, so the scale
#: (alpha / rank = 1.0) is every prior candidate's scale.
PROJECTION_RANK = 16
PROJECTION_ALPHA = 16


def projection_coefficient(m1_map: dict, m2_map: dict,
                           seqs) -> dict:
    """The projection coefficient α (RFC SI-30 §7.2) from the
    movement maps on a named allowed population (in the
    construction, exactly the signal's train partition):
    ``α = (Σ_x ⟨m1(x), m2(x)⟩) / (Σ_x ⟨m1(x), m1(x)⟩)`` with
    ``m1`` / ``m2`` SI-27's movement definitions. Returns
    ``{"alpha", "numerator", "denominator"}`` — the sums
    beside the coefficient, so the as-built can publish the
    derivation's inputs. When the denominator is exactly
    0.0 — at head = ``base``, where ``m1`` is identically
    zero and both sums vanish — α is DEFINED as 0.0 (the
    construction is the identity there; RFC §14 item 4).
    α is a measured coefficient, never a dial: nothing in
    this function reads a caller-supplied value for it."""
    numerator = 0.0
    denominator = 0.0
    for seq in seqs:
        m1 = np.asarray(m1_map[tuple(seq)], dtype=np.float64)
        m2 = np.asarray(m2_map[tuple(seq)], dtype=np.float64)
        numerator += float(np.dot(m1, m2))
        denominator += float(np.dot(m1, m1))
    alpha = numerator / denominator if denominator != 0.0 else 0.0
    return {"alpha": float(alpha), "numerator": float(numerator),
            "denominator": float(denominator)}


def projection_factors(delta_prime: np.ndarray):
    """``(A, B)`` float64 LoRA factors carrying the folded
    update ``delta_prime`` at :data:`PROJECTION_RANK` /
    :data:`PROJECTION_ALPHA` (scale 1.0): the SVD
    factorisation ``delta_prime = U S Vt`` (rank at most 16
    for a difference of two rank-8 updates, padded with zero
    singular values when the rank is smaller), split as
    ``B = U·sqrt(S)``, ``A = sqrt(S)·Vt``. Deterministic:
    the same input matrix yields the same factors on this
    runtime (asserted by the determinism tests)."""
    U, S, Vt = np.linalg.svd(
        np.asarray(delta_prime, dtype=np.float64))
    r = PROJECTION_RANK
    sq = np.sqrt(S[:r])
    A = sq[:, None] * Vt[:r]
    B = U[:, :r] * sq[None, :]
    return A, B


# -- deterministic blob serialisation (§8.5) -----------------------------------
# construct_transformer_delta serialises through safetensors'
# save(), whose header emits the __metadata__ mapping in hash-map
# order — observed on this runtime to FLIP between two calls in
# the same process ({"format","targets"} vs {"targets","format"})
# over bit-identical factor bytes. A candidate's blob hash is
# content addressing, so that flip alone would make two identical
# training runs produce different blobs and break §8.5. The
# authored candidate is therefore re-serialised here with a
# pinned header (metadata keys sorted, factor keys sorted, data
# laid out in that order): the tensor DATA bytes are the
# constructed blob's own, byte for byte; only header key order is
# canonicalised. The re-serialised pair is run back through
# validate_transformer_delta before it is returned, so the
# canonicalisation can never emit an invalid candidate.

def _canonical_blob_bytes(blob_bytes: bytes) -> bytes:
    import json
    import struct

    from si.transformer_delta import FACTOR_DTYPE_CODE, parse_blob

    parsed = parse_blob(blob_bytes)
    keys = sorted(parsed.tensors)
    header: dict = {"__metadata__": {
        k: parsed.metadata[k] for k in sorted(parsed.metadata)}}
    data = bytearray()
    for key in keys:
        arr = np.ascontiguousarray(parsed.tensors[key], dtype=np.float32)
        raw = arr.tobytes(order="C")
        header[key] = {"dtype": FACTOR_DTYPE_CODE,
                       "shape": [int(x) for x in arr.shape],
                       "data_offsets": [len(data), len(data) + len(raw)]}
        data += raw
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = -(8 + len(header_json)) % 8
    header_json += b" " * pad
    return struct.pack("<Q", len(header_json)) + header_json + bytes(data)


# -- the training run + candidate construction ---------------------------------

def train_candidate(bundle_dir: str, manifest: dict, intent,
                    policy: dict | None = None) -> dict:
    """Train the named signal's candidate for a miniature bundle.

    The ``TinyTransformerMind.train_delta`` implementation:
    validates the intent (unknown signal / foreign targets /
    foreign rank-alpha all refuse), assembles the bundle's
    applied weights, records the base's held-out loss BEFORE
    training, runs full-batch gradient descent, refuses the
    result if its held-out loss did not improve, and otherwise
    returns the SI-9b candidate contract plus ``hyperparams``:
    ``{"record", "blob_bytes", "norm", "hyperparams"}`` with the
    candidate built by ``construct_transformer_delta`` against
    the verified manifest (callers never supply the base
    binding; construction never emits an invalid delta).
    SI-21: ``manifest`` is whatever verified manifest the
    caller supplies — the freshness contract (exactly one
    manifest trains, the cycle's) is enforced by the adapter
    in ``si.tiny_transformer``; this function assembles at
    ``manifest``'s head and records that head in the
    hyperparams (see the module docstring's anchoring note).
    SI-25: the intent's ``target_anchoring`` is validated
    against exactly the two named anchorings and dispatched
    on here — under ``"base-anchored"`` only the signal's
    targets derive from the frozen seed base (see the module
    docstring); every other role stays with the weights under
    training. An intent carrying no anchoring field at all
    (a duck-typed pre-SI-25 intent) trains under the default,
    :data:`TARGET_ANCHORING`; an intent carrying any other
    value refuses.
    SI-27: the intent's ``composition_config`` is validated
    against exactly the two named composition configurations
    and dispatched on here — under ``"movement-composition"``
    exactly one penalty term is added to the loss (see the
    module docstring); under ``"none"`` nothing about the
    SI-25 path changes. An intent carrying no composition
    field at all trains under the default,
    :data:`DEFAULT_COMPOSITION_CONFIG`; an intent carrying
    any other value refuses. λ is never read from the intent:
    it is the configured value's definition
    (:data:`COMPOSITION_LAMBDA` under
    ``"movement-composition"``, 0.0 under ``"none"``).
    SI-29: the same validation now serves exactly THREE
    named configurations — under
    ``"worst-input-composition"`` exactly one worst-input
    term is added (λ_w = :data:`WORST_INPUT_LAMBDA`, k =
    :data:`WORST_INPUT_K`, both the configuration's
    definition, never read from the intent); under the two
    prior values the paths are byte-for-byte SI-25's /
    SI-27's respectively.
    SI-31: the validation now serves exactly FOUR named
    configurations — under ``"train-partition-projection"``
    the training step is SI-25's exactly and the judged
    candidate is then constructed by the full-removal
    projection of §7.2 (see the module docstring): the
    measured coefficient α, the pre-/post-projection norms,
    and the projected candidate's re-measured records are
    recorded in the hyperparams beside the signal loss's
    values; the pre-projection candidate is a construction
    intermediate and is never the returned candidate. α is
    never read from the intent or any other caller input.
    """
    from si.tiny_transformer import assemble_base_weights, assemble_weights
    from si.transformer_delta import construct_transformer_delta

    signal_id = getattr(intent, "training_signal_id", "")
    # SI-23 dispatch (RFC SI-22 §5.2): the registry serves
    # exactly the two named signals; signal_spec raises the
    # existing ValueError discipline for any other id.
    signal_spec(signal_id)
    # SI-25 dispatch (RFC SI-24 §5.3): exactly the two named
    # anchorings; validate_target_anchoring raises the same
    # ValueError discipline for any other value.
    _raw_anchoring = getattr(intent, "target_anchoring", None)
    anchoring = validate_target_anchoring(
        TARGET_ANCHORING if _raw_anchoring is None else _raw_anchoring)
    # SI-27 dispatch (RFC SI-26 §7.4), extended by SI-29
    # (RFC SI-28 §6.4) and SI-31 (RFC SI-30 §7.3): exactly
    # the four named composition configurations;
    # validate_composition_config raises the same ValueError
    # discipline for any other value. λ / λ_w / k follow from
    # the configuration's definition — they are never taken
    # from the intent or any other caller input, so no
    # per-cycle dial exists for any of them. Under SI-31's
    # value there is no dial at all: α is measured from the
    # run's own candidates on the train partition.
    _raw_comp = getattr(intent, "composition_config", None)
    comp_config = validate_composition_config(
        DEFAULT_COMPOSITION_CONFIG if _raw_comp is None else _raw_comp)
    comp_lambda = COMPOSITION_LAMBDA if comp_config == MOVEMENT_COMPOSITION \
        else 0.0
    penalised = comp_config == MOVEMENT_COMPOSITION
    worst_lambda = WORST_INPUT_LAMBDA \
        if comp_config == WORST_INPUT_COMPOSITION else 0.0
    worst_k = WORST_INPUT_K \
        if comp_config == WORST_INPUT_COMPOSITION else None
    worst_penalised = comp_config == WORST_INPUT_COMPOSITION
    projected = comp_config == TRAIN_PARTITION_PROJECTION
    composition_active = penalised or worst_penalised or projected
    declared = list(getattr(intent, "targets", None) or [])
    if declared and declared != [TARGET_NAME]:
        raise ValueError(
            f"intent declares targets {declared!r}: "
            f"{signal_id} trains exactly {[TARGET_NAME]!r} "
            "and never silently retargets (RFC SI-18 §8.6)")
    rank = int(getattr(intent, "rank", 0) or 0)
    alpha = int(getattr(intent, "alpha", 0) or 0)
    if (rank and rank != RANK) or (alpha and alpha != ALPHA):
        raise ValueError(
            f"intent requests rank={rank} alpha={alpha}: "
            f"{signal_id} is pinned at rank={RANK} "
            f"alpha={ALPHA} and refuses a resized candidate "
            "(RFC SI-18 §8.6)")

    started = time.perf_counter()
    weights = assemble_weights(bundle_dir, manifest, policy)
    # SI-25 (§5.2/§5.4): split ONLY the target-source role.
    # The weights under training (``weights``) still supply the
    # hidden states, the readout under training, candidate
    # construction, and every loss evaluation below. Under
    # base-anchoring the targets alone derive from the frozen
    # seed base, assembled from THIS verified manifest with its
    # active chain truncated to empty. Fail closed: a failed
    # base assembly raises here and refuses training — there
    # is deliberately NO try/except fallback to head-derived
    # targets (a silent fallback would train a head-relative
    # candidate wearing a base-anchored label).
    if anchoring == BASE_ANCHORED:
        target_weights = assemble_base_weights(
            bundle_dir, manifest, policy)
        target_source_head = "base"
    else:
        target_weights = weights
        target_source_head = manifest["head"]
    signal = build_training_signal(target_weights, signal_id)
    train_ex, held_ex = signal["train"], signal["held_out"]
    lm_head = np.asarray(weights[TARGET_NAME], dtype=np.float64)
    hiddens = {tuple(seq): final_hidden(weights, seq)
               for seq, _ in train_ex}

    # SI-27 (§7.2): under "movement-composition" ONLY, the
    # penalty's constants are prepared here — m1 on the
    # signal's TRAIN partition, from the landed head's and
    # the frozen base's forwards (the base assembly is the
    # SI-25 one: under base-anchoring it IS target_weights;
    # under head-relative anchoring it is assembled here for
    # the movement role alone, fail-closed with no fallback).
    # Held-out m1 / hidden states are prepared ONLY so the
    # penalty's held-out value can be REPORTED (initial and
    # final, §7.2/§14 item 5); they never enter the loss, the
    # gradients, stopping, or selection. Under "none" none of
    # this runs: no base assembly, no penalty quantity — the
    # SI-25 path, byte-for-byte.
    m1_train = None
    held_hiddens = None
    m1_held = None
    if composition_active:
        if anchoring == BASE_ANCHORED:
            base_weights = target_weights
        else:
            base_weights = assemble_base_weights(
                bundle_dir, manifest, policy)
        m1_train = existing_movements(
            weights, base_weights, [seq for seq, _ in train_ex])
        held_hiddens = {tuple(seq): final_hidden(weights, seq)
                        for seq, _ in held_ex}
        m1_held = existing_movements(
            weights, base_weights, [seq for seq, _ in held_ex])

    # The base's scores, recorded BEFORE training (RFC §8.1/§8.4):
    # the zero-delta losses of the frozen applied weights.
    base_train_loss = signal_loss(weights, train_ex)
    base_held_out_loss = signal_loss(weights, held_ex)

    A, B = initial_factors(signal_id)
    scale = float(ALPHA) / float(RANK)
    penalty_train_initial = None
    penalty_held_initial = None
    worst_train_initial = None
    worst_held_initial = None
    worst_train_selection_initial = None
    worst_held_selection_initial = None
    if penalised:
        _total_initial, train_loss_initial, penalty_train_initial, \
            _gA, _gB = analytic_gradients_penalised(
                A, B, train_ex, hiddens, lm_head, m1_train, comp_lambda)
        penalty_held_initial = composition_penalty(
            A, B, [seq for seq, _ in held_ex], held_hiddens, m1_held)
    elif worst_penalised:
        _total_initial, train_loss_initial, worst_train_initial, \
            _gA, _gB = analytic_gradients_worst_input(
                A, B, train_ex, hiddens, lm_head, m1_train,
                worst_lambda, WORST_INPUT_K)
        worst_train_selection_initial = worst_input_selection(
            A, B, [seq for seq, _ in train_ex], hiddens, m1_train,
            WORST_INPUT_K)
        worst_held_selection_initial = worst_input_selection(
            A, B, [seq for seq, _ in held_ex], held_hiddens, m1_held,
            WORST_INPUT_K)
        worst_held_initial = worst_held_selection_initial["value"]
    else:
        train_loss_initial, _, _ = analytic_gradients(
            A, B, train_ex, hiddens, lm_head)
    projection_activations = 0
    for _step in range(STEPS):
        if penalised:
            _loss, _sig, _pen, grad_A, grad_B = \
                analytic_gradients_penalised(
                    A, B, train_ex, hiddens, lm_head, m1_train,
                    comp_lambda)
        elif worst_penalised:
            _loss, _sig, _worst, grad_A, grad_B = \
                analytic_gradients_worst_input(
                    A, B, train_ex, hiddens, lm_head, m1_train,
                    worst_lambda, WORST_INPUT_K)
        else:
            _loss, grad_A, grad_B = analytic_gradients(
                A, B, train_ex, hiddens, lm_head)
        B = B - LEARNING_RATE * grad_B
        A = A - LEARNING_RATE * grad_A
        norm = float(np.linalg.norm(scale * (B @ A)))
        if norm > NORM_BUDGET:
            B = B * (NORM_BUDGET / norm)
            projection_activations += 1
    penalty_train_final = None
    penalty_held_final = None
    worst_train_final = None
    worst_held_final = None
    worst_train_selection_final = None
    worst_held_selection_final = None
    if penalised:
        _total_final, train_loss_final, penalty_train_final, \
            _gA, _gB = analytic_gradients_penalised(
                A, B, train_ex, hiddens, lm_head, m1_train, comp_lambda)
        penalty_held_final = composition_penalty(
            A, B, [seq for seq, _ in held_ex], held_hiddens, m1_held)
    elif worst_penalised:
        _total_final, train_loss_final, worst_train_final, \
            _gA, _gB = analytic_gradients_worst_input(
                A, B, train_ex, hiddens, lm_head, m1_train,
                worst_lambda, WORST_INPUT_K)
        worst_train_selection_final = worst_input_selection(
            A, B, [seq for seq, _ in train_ex], hiddens, m1_train,
            WORST_INPUT_K)
        worst_held_selection_final = worst_input_selection(
            A, B, [seq for seq, _ in held_ex], held_hiddens, m1_held,
            WORST_INPUT_K)
        worst_held_final = worst_held_selection_final["value"]
    else:
        train_loss_final, _, _ = analytic_gradients(
            A, B, train_ex, hiddens, lm_head)

    A32 = np.ascontiguousarray(A, dtype=np.float32)
    B32 = np.ascontiguousarray(B, dtype=np.float32)
    held_out_after = factor_loss(weights, A32, B32, held_ex)
    closer = 0
    trained_weights = _folded_weights(weights, A32, B32)
    for seq, target in held_ex:
        before = float(np.mean(
            (forward(seq, weights)[-1].astype(np.float64) - target) ** 2))
        after = float(np.mean(
            (forward(seq, trained_weights)[-1].astype(np.float64)
             - target) ** 2))
        if after < before:
            closer += 1

    # SI-31 (RFC SI-30 §7.2): under "train-partition-
    # projection" ONLY, the candidate above is a construction
    # INTERMEDIATE — trained exactly as SI-25 trained it —
    # and the judged candidate is constructed from it here,
    # once, in full. The intermediate's held-out figures are
    # kept under their own labels (projection_pre_*); the
    # non-improvement refusal below then judges the
    # PROJECTED candidate's held-out loss, exactly as it
    # judges a trained candidate's under every other
    # configuration (under those configurations the judged
    # candidate IS the trained candidate and this block does
    # not run). No probe input, probe logit, held-out
    # sequence, or probe-derived quantity enters α, the
    # construction, or the refusal: α's inputs are the
    # intermediate's movements on the train partition alone.
    cand_A32, cand_B32 = A32, B32
    cand_rank, cand_alpha = RANK, ALPHA
    proj_alpha = None
    proj_numerator = None
    proj_denominator = None
    proj_pre_norm = None
    proj_train_loss_final = None
    proj_pre_held_out_after = None
    proj_pre_closer = None
    proj_train_agg_inner = None
    proj_held_agg_inner = None
    if projected:
        proj_pre_held_out_after = float(held_out_after)
        proj_pre_closer = int(closer)
        head_lm = np.asarray(weights[TARGET_NAME], dtype=np.float64)
        base_lm = np.asarray(base_weights[TARGET_NAME],
                             dtype=np.float64)
        delta1 = head_lm - base_lm
        delta2 = np.asarray(trained_weights[TARGET_NAME],
                            dtype=np.float64) - head_lm
        proj_pre_norm = float(np.linalg.norm(delta2))
        m2_train = existing_movements(
            trained_weights, weights, [seq for seq, _ in train_ex])
        coeff = projection_coefficient(
            m1_train, m2_train, [seq for seq, _ in train_ex])
        proj_alpha = coeff["alpha"]
        proj_numerator = coeff["numerator"]
        proj_denominator = coeff["denominator"]
        delta_prime = delta2 - proj_alpha * delta1
        if proj_alpha == 0.0:
            # The construction is the identity (at base, or
            # wherever the measured coefficient vanishes):
            # the judged candidate keeps the intermediate's
            # factors byte-for-byte.
            proj_weights = trained_weights
        else:
            pA, pB = projection_factors(delta_prime)
            cand_A32 = np.ascontiguousarray(pA, dtype=np.float32)
            cand_B32 = np.ascontiguousarray(pB, dtype=np.float32)
            cand_rank, cand_alpha = PROJECTION_RANK, PROJECTION_ALPHA
            proj_target = {"name": TARGET_NAME, "kind": "lora",
                           "factor_keys": ["t0.A", "t0.B"],
                           "rank": PROJECTION_RANK,
                           "alpha": PROJECTION_ALPHA,
                           "base_shape": [_V, _D]}
            proj_weights = dict(weights)
            proj_weights[TARGET_NAME] = fold_target(
                np.array(weights[TARGET_NAME], dtype=np.float32,
                         copy=True),
                proj_target,
                {"t0.A": cand_A32, "t0.B": cand_B32},
                allow_additive=False, delta_id="si-31-projection")
        # Post-projection accounting (§7.2): the projected
        # candidate is re-measured in full before it is
        # returned — signal losses on both partitions and
        # the allowed-input composition record (train:
        # the construction's own check; held-out: reported
        # only, never an input to the construction).
        proj_train_loss_final = float(signal_loss(proj_weights,
                                                  train_ex))
        held_out_after = float(signal_loss(proj_weights, held_ex))
        closer = 0
        for seq, target in held_ex:
            before = float(np.mean(
                (forward(seq, weights)[-1].astype(np.float64)
                 - target) ** 2))
            after = float(np.mean(
                (forward(seq, proj_weights)[-1].astype(np.float64)
                 - target) ** 2))
            if after < before:
                closer += 1
        proj_train_agg_inner = float(movement_composition_record(
            weights, base_weights, proj_weights,
            [seq for seq, _ in train_ex])["aggregate_inner_product"])
        proj_held_agg_inner = float(movement_composition_record(
            weights, base_weights, proj_weights,
            [seq for seq, _ in held_ex])["aggregate_inner_product"])

    if not held_out_after < base_held_out_loss:
        raise ValueError(
            f"train_delta refuses its own candidate: held-out loss "
            f"on signal {signal_id} did not improve "
            f"(base {base_held_out_loss:.6f} -> trained "
            f"{held_out_after:.6f}); a non-improving candidate is a "
            "failed experiment, not authorship (RFC SI-18 §8.4)")

    delta_id = f"d{len(manifest.get('deltas', [])) + 1:04d}"
    record, blob_bytes = construct_transformer_delta(
        manifest,
        [{"name": TARGET_NAME, "kind": "lora",
          "A": cand_A32, "B": cand_B32, "alpha": cand_alpha,
          "rank": cand_rank}],
        delta_id=delta_id, parent=manifest["head"],
        created_at=CANDIDATE_CREATED_AT, self_edit_id="se_" + delta_id)
    # §8.5: canonicalise the blob header (see _canonical_blob_bytes)
    # and rebind the record to the canonical bytes, re-validated.
    blob_bytes = _canonical_blob_bytes(blob_bytes)
    record = dict(record)
    record["blob"] = "sha256:" + sha256_hex(blob_bytes)
    record["bytes"] = len(blob_bytes)
    from si.transformer_delta import validate_transformer_delta

    validate_transformer_delta(record, blob_bytes, manifest)
    wall_time = time.perf_counter() - started
    hyperparams = {
        "training_signal_id": signal_id,
        # SI-21 (RFC SI-20 §10.2): every cycle records the head
        # it trained against, the target-source head, and the
        # stack depth at training time, alongside the SI-19
        # fields. SI-25: the two head fields are read from the
        # cycle's own configuration — under
        # "weights-under-training" the target source IS the
        # head trained against; under "base-anchored" the
        # target source is "base" at every head, so the fields
        # disagree exactly when the new semantics ran. A
        # cycle's anchoring is read from THIS record, never
        # inferred from the TARGET_ANCHORING constant (which
        # now names only the default). "stack_depth" is the
        # active chain's length at that head (0 at base).
        # Note (§5.5): under base-anchoring at a non-base head,
        # "base_held_out_loss" below means the pre-training
        # held-out loss of the weights under training (at
        # "head") against targets derived from the recorded
        # target source — a genuine measurement, not the
        # closed form.
        "head": manifest["head"],
        "target_source_head": target_source_head,
        "target_anchoring": anchoring,
        # SI-27 (RFC SI-26 §7.4): every cycle records the
        # composition configuration it trained under and the
        # λ that configuration defines (1.0 under
        # "movement-composition", 0.0 under "none" — λ is the
        # configuration's definition, never a dial). Under
        # "movement-composition" the penalty's measured values
        # are recorded beside the signal loss's: on the train
        # partition (optimised) and on the held-out partition
        # (REPORTED ONLY — never an optimisation input), at
        # initialisation and at the end of training, plus the
        # total (signal + λ·penalty) train loss at both points
        # — the two terms separately, as §7.2(iii) requires
        # (the total can be negative; that is expected and is
        # not hidden). "train_loss_initial/final" above remain
        # the SIGNAL loss alone under both configurations, and
        # the non-improvement refusal still judges the signal
        # held-out loss alone. Under "none" the penalty fields
        # are null (not measured — the unpenalised path
        # computes no penalty quantity) and the totals equal
        # the signal losses.
        "composition_config": comp_config,
        "composition_lambda": float(comp_lambda),
        "penalty_train_initial": (
            None if penalty_train_initial is None
            else float(penalty_train_initial)),
        "penalty_train_final": (
            None if penalty_train_final is None
            else float(penalty_train_final)),
        "penalty_held_out_initial": (
            None if penalty_held_initial is None
            else float(penalty_held_initial)),
        "penalty_held_out_final": (
            None if penalty_held_final is None
            else float(penalty_held_final)),
        "composition_lambda_w": float(worst_lambda),
        "composition_worst_k": worst_k,
        "worst_input_train_initial": (
            None if worst_train_initial is None
            else float(worst_train_initial)),
        "worst_input_train_final": (
            None if worst_train_final is None
            else float(worst_train_final)),
        "worst_input_held_out_initial": (
            None if worst_held_initial is None
            else float(worst_held_initial)),
        "worst_input_held_out_final": (
            None if worst_held_final is None
            else float(worst_held_final)),
        "worst_input_train_topk_indices_initial": (
            None if worst_train_selection_initial is None else
            [int(i) for i in
             worst_train_selection_initial["topk_indices"]]),
        "worst_input_train_topk_indices_final": (
            None if worst_train_selection_final is None else
            [int(i) for i in
             worst_train_selection_final["topk_indices"]]),
        "worst_input_train_worst_index_initial": (
            None if worst_train_selection_initial is None else
            int(worst_train_selection_initial["worst_index"])),
        "worst_input_train_worst_index_final": (
            None if worst_train_selection_final is None else
            int(worst_train_selection_final["worst_index"])),
        "worst_input_train_worst_seq_initial": (
            None if worst_train_selection_initial is None else
            [int(x) for x in train_ex[
                worst_train_selection_initial["worst_index"]][0]]),
        "worst_input_train_worst_seq_final": (
            None if worst_train_selection_final is None else
            [int(x) for x in train_ex[
                worst_train_selection_final["worst_index"]][0]]),
        "worst_input_held_out_worst_index_initial": (
            None if worst_held_selection_initial is None else
            int(worst_held_selection_initial["worst_index"])),
        "worst_input_held_out_worst_index_final": (
            None if worst_held_selection_final is None else
            int(worst_held_selection_final["worst_index"])),
        # SI-31 (RFC SI-30 §7.2/§7.3): under
        # "train-partition-projection" the measured
        # coefficient α and the construction's accounting
        # are recorded beside the signal loss's values —
        # α recorded, never settable; the intermediate's
        # figures under projection_pre_* labels (an
        # intermediate's, never the judged candidate's);
        # the projected candidate's own held-out loss is
        # "held_out_loss_after" above, as under every
        # configuration. Under the three prior
        # configurations every projection field is null
        # (not measured — no construction ran) and the
        # penalty / worst-input fields above keep their
        # own meanings: the mechanisms are never blended.
        "projection_alpha": (
            None if proj_alpha is None else float(proj_alpha)),
        "projection_numerator": (
            None if proj_numerator is None
            else float(proj_numerator)),
        "projection_denominator": (
            None if proj_denominator is None
            else float(proj_denominator)),
        "projection_pre_norm": (
            None if proj_pre_norm is None
            else float(proj_pre_norm)),
        "projection_post_norm": (
            None if proj_alpha is None
            else float(record["norm"])),
        "projection_candidate_rank": (
            None if proj_alpha is None else int(cand_rank)),
        "projection_train_loss_final": (
            None if proj_train_loss_final is None
            else float(proj_train_loss_final)),
        "projection_pre_held_out_loss_after": (
            None if proj_pre_held_out_after is None
            else float(proj_pre_held_out_after)),
        "projection_pre_held_out_examples_closer": (
            None if proj_pre_closer is None
            else int(proj_pre_closer)),
        "projection_train_aggregate_inner_product": (
            None if proj_train_agg_inner is None
            else float(proj_train_agg_inner)),
        "projection_held_out_aggregate_inner_product": (
            None if proj_held_agg_inner is None
            else float(proj_held_agg_inner)),
        "total_train_loss_initial": float(
            train_loss_initial
            + (comp_lambda * penalty_train_initial if penalised else 0.0)
            + (worst_lambda * worst_train_initial
               if worst_penalised else 0.0)),
        "total_train_loss_final": float(
            train_loss_final
            + (comp_lambda * penalty_train_final if penalised else 0.0)
            + (worst_lambda * worst_train_final
               if worst_penalised else 0.0)),
        "stack_depth": len(_active_chain(manifest)),
        "target": TARGET_NAME,
        "rank": RANK,
        "alpha": ALPHA,
        "scale": scale,
        "steps": STEPS,
        "lr": LEARNING_RATE,
        "init_scale": INIT_SCALE,
        "norm_budget": NORM_BUDGET,
        "norm_projection_activations": projection_activations,
        "target_correction_scale": CORRECTION_SCALE,
        "train_loss_initial": float(train_loss_initial),
        "train_loss_final": float(train_loss_final),
        "base_train_loss": float(base_train_loss),
        "base_held_out_loss": float(base_held_out_loss),
        "held_out_loss_after": float(held_out_after),
        "held_out_examples_closer": closer,
        "held_out_examples": len(held_ex),
        "target_norm": float(record["norm"]),
        # The one non-deterministic field (§8.5): a measured wall
        # time, labelled as such; no measured value depends on it.
        "wall_time_seconds": wall_time,
    }
    return {"record": record, "blob_bytes": blob_bytes,
            "norm": float(record["norm"]), "hyperparams": hyperparams}
