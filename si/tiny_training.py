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
    "INIT_SCALE",
    "LEARNING_RATE",
    "N_HELD_OUT",
    "N_LAST_TOKENS",
    "N_TRAIN",
    "NORM_BUDGET",
    "RANK",
    "SIGNAL_SEED_OFFSET",
    "STEPS",
    "TARGET_ANCHORING",
    "TARGET_NAME",
    "TRAINING_SIGNAL_ID",
    "analytic_gradients",
    "build_training_signal",
    "factor_loss",
    "final_hidden",
    "forward_with_final_hidden",
    "initial_factors",
    "signal_loss",
    "signal_sequences",
    "train_candidate",
]

#: The one training signal this slice authors against (RFC §4.3:
#: a mind serves exactly the signal(s) its design names).
TRAINING_SIGNAL_ID = "tiny-successor-lean-001"

#: SI-21 target anchoring (RFC SI-20 §4.3.2), named in code:
#: targets are derived from the weights under training — the
#: bundle's applied weights at the head being trained against —
#: plus the recorded correction. The alternative (targets
#: anchored to the seed base's logits at every cycle) is NOT
#: the semantics of this signal id; see the module docstring.
TARGET_ANCHORING = "weights-under-training"

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

def signal_sequences():
    """The signal's raw inputs, deterministically constructed.

    Returns ``(train_seqs, held_out_seqs, last_tokens)``: 32 + 16
    token-id sequences (lengths 2..8, vocabulary 32) whose last
    tokens cycle round-robin through the 8 ``last_tokens`` drawn
    first from the signal stream. Draws that duplicate an earlier
    signal sequence or collide with ANY probe-suite input
    (identity or capability) are rejected and redrawn from the
    same stream, so the construction is disjoint by construction
    as well as by test.
    """
    rng = np.random.default_rng(MINI_SEED + SIGNAL_SEED_OFFSET)
    probe_seqs = {tuple(s) for s in identity_inputs()} \
        | {tuple(s) for s in capability_inputs()}
    last_tokens = [int(x) for x in
                   rng.choice(_V, size=N_LAST_TOKENS, replace=False)]
    seqs: list = []
    seen: set = set()
    while len(seqs) < N_TRAIN + N_HELD_OUT:
        length = int(rng.integers(2, 9))
        seq = [int(x) for x in rng.integers(0, _V, size=length)]
        seq[-1] = last_tokens[len(seqs) % N_LAST_TOKENS]
        key = tuple(seq)
        if key in probe_seqs or key in seen:
            continue
        seen.add(key)
        seqs.append(seq)
    return seqs[:N_TRAIN], seqs[N_TRAIN:], last_tokens


def build_training_signal(weights: dict) -> dict:
    """The full signal against a base weight set: inputs + targets.

    ``weights`` is the miniature's full applied weight set (the
    SI-15 forward's input) — at SI-19's only head, the base; at
    cycle >= 2 (SI-21), the landed head's applied weights.
    Anchoring (:data:`TARGET_ANCHORING`): each example's target
    is THOSE weights' own last-position logits (through the
    existing forward) plus the recorded successor-lean
    correction — the target source is the weights under
    training, never silently the seed base on a non-base head.
    Returns ``{"train": [(seq, target_logits)], "held_out":
    [...], "last_tokens": [...]}`` with float64 target logits.
    """
    train_seqs, held_out_seqs, last_tokens = signal_sequences()

    def with_targets(seqs):
        out = []
        for seq in seqs:
            base_logits = forward(seq, weights)[-1].astype(np.float64)
            correction = np.zeros(_V, dtype=np.float64)
            correction[(seq[-1] + 1) % _V] += CORRECTION_SCALE
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

def initial_factors():
    """The pinned initial factors ``(A, B)`` (float64): one
    dedicated stream, A drawn before B, ``INIT_SCALE``-scaled."""
    rng = np.random.default_rng(MINI_SEED + INIT_SEED_OFFSET)
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
    """
    from si.tiny_transformer import assemble_weights
    from si.transformer_delta import construct_transformer_delta

    signal_id = getattr(intent, "training_signal_id", "")
    if signal_id != TRAINING_SIGNAL_ID:
        raise ValueError(f"unknown training signal {signal_id!r}")
    declared = list(getattr(intent, "targets", None) or [])
    if declared and declared != [TARGET_NAME]:
        raise ValueError(
            f"intent declares targets {declared!r}: "
            f"{TRAINING_SIGNAL_ID} trains exactly {[TARGET_NAME]!r} "
            "and never silently retargets (RFC SI-18 §8.6)")
    rank = int(getattr(intent, "rank", 0) or 0)
    alpha = int(getattr(intent, "alpha", 0) or 0)
    if (rank and rank != RANK) or (alpha and alpha != ALPHA):
        raise ValueError(
            f"intent requests rank={rank} alpha={alpha}: "
            f"{TRAINING_SIGNAL_ID} is pinned at rank={RANK} "
            f"alpha={ALPHA} and refuses a resized candidate "
            "(RFC SI-18 §8.6)")

    started = time.perf_counter()
    weights = assemble_weights(bundle_dir, manifest, policy)
    signal = build_training_signal(weights)
    train_ex, held_ex = signal["train"], signal["held_out"]
    lm_head = np.asarray(weights[TARGET_NAME], dtype=np.float64)
    hiddens = {tuple(seq): final_hidden(weights, seq)
               for seq, _ in train_ex}

    # The base's scores, recorded BEFORE training (RFC §8.1/§8.4):
    # the zero-delta losses of the frozen applied weights.
    base_train_loss = signal_loss(weights, train_ex)
    base_held_out_loss = signal_loss(weights, held_ex)

    A, B = initial_factors()
    scale = float(ALPHA) / float(RANK)
    train_loss_initial, _, _ = analytic_gradients(
        A, B, train_ex, hiddens, lm_head)
    projection_activations = 0
    for _step in range(STEPS):
        _loss, grad_A, grad_B = analytic_gradients(
            A, B, train_ex, hiddens, lm_head)
        B = B - LEARNING_RATE * grad_B
        A = A - LEARNING_RATE * grad_A
        norm = float(np.linalg.norm(scale * (B @ A)))
        if norm > NORM_BUDGET:
            B = B * (NORM_BUDGET / norm)
            projection_activations += 1
    train_loss_final, _, _ = analytic_gradients(
        A, B, train_ex, hiddens, lm_head)

    A32 = np.ascontiguousarray(A, dtype=np.float32)
    B32 = np.ascontiguousarray(B, dtype=np.float32)
    held_out_after = factor_loss(weights, A32, B32, held_ex)
    if not held_out_after < base_held_out_loss:
        raise ValueError(
            f"train_delta refuses its own candidate: held-out loss "
            f"on signal {TRAINING_SIGNAL_ID} did not improve "
            f"(base {base_held_out_loss:.6f} -> trained "
            f"{held_out_after:.6f}); a non-improving candidate is a "
            "failed experiment, not authorship (RFC SI-18 §8.4)")
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

    delta_id = f"d{len(manifest.get('deltas', [])) + 1:04d}"
    record, blob_bytes = construct_transformer_delta(
        manifest,
        [{"name": TARGET_NAME, "kind": "lora",
          "A": A32, "B": B32, "alpha": ALPHA, "rank": RANK}],
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
        "training_signal_id": TRAINING_SIGNAL_ID,
        # SI-21 (RFC SI-20 §10.2): every cycle records the head
        # it trained against, the target-source head, and the
        # stack depth at training time, alongside the SI-19
        # fields. Under TARGET_ANCHORING the target source IS
        # the head trained against (targets derive from the
        # weights under training); both fields are recorded so
        # the identity is explicit, not assumed. "stack_depth"
        # is the active chain's length at that head (0 at base).
        "head": manifest["head"],
        "target_source_head": manifest["head"],
        "target_anchoring": TARGET_ANCHORING,
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
