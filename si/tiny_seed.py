"""``si seed-tiny``: seed the SI-13 miniature transformer as a bundle base.

SI-14 (RFC SI-13 §3, §5, §8): a purpose-built miniature decoder-only
transformer — ``TinyTransformerMind`` — constructed deterministically
from a fixed seed, **not trained**, seeded as an ordinary per-tensor
bundle (SI-7 copy-mode machinery) together with the §5 probe suites
and §5.4 baselines/margins/tolerances recorded as the §5.5 probe
artifact (``probe_suite.json`` in the bundle, content-hashed into the
manifest, journaled at seed time).

POSITIONAL TABLE DECISION (RFC §3.3 left this to SI-14): the
positional embedding table IS included as a base tensor
(``model.language_model.embed_positions.weight``, ``[16, 32]``).
The inventory is therefore **30 tensors / 23,456 elements** — the
RFC §3.1 budget (29 / 22,944) plus exactly the +1 tensor / +512
elements the RFC flagged for this variant.

Scope discipline (the external contract of this slice):

- The forward computation in this module is an INTERNAL seed-time
  helper. It exists so seed construction can measure expected probe
  outputs, margins, and the echo floor on the seed it is about to
  record. It is wired to NOTHING: no gate, no loop, no serve. SI-15
  owns the library forward path; SI-16 owns any gate consumption.
- Nothing consumes the probe artifact in this slice. Gate behaviour
  is byte-for-byte unchanged: transformer deltas — miniature ones
  included — still report checks 3/4 as "not evaluated" and still
  ESCALATE (SI-9b / SI-10 behaviour).
- The miniature is NOT Glimmer, at any scale claim. It shares a
  family shape and Glimmer-style tensor names so SI's existing
  tensor machinery applies unchanged; it shares no weights, no
  tokenizer, and no behaviour with the 30B seed. Probe results on
  the miniature say something about deltas against the miniature —
  nothing else.

Seed-time assertions, fail closed (RFC §3.3): before any baseline
is recorded and before anything is written, construction verifies
by running the internal forward helper on the §5 suites:

- determinism — construction twice produces byte-identical tensors,
  and two forward runs over the suites are bit-identical;
- finiteness — every tensor and every expected logit is finite;
- the echo floor — the constructed base's echo accuracy (argmax of
  last-position logits == last token, §5.3) is >= 15/16;
- tolerance sanity — the §5.4 derived tolerances
  (``min(0.25, 0.4 * min_margin)``) are positive; a seed whose
  margins collapse fails closed instead of recording a tolerance
  no honest delta could meet.

A seed that fails any assertion is REFUSED: ``seed_tiny_bundle``
raises ``ValueError`` before writing a single byte, and the bundle
keeps whatever base it had.

Honesty label: the manifest ``base.source`` for a miniature base
reads ``"tiny:constructed-miniature (SI-14; constructed miniature
for probe development, not a trained model)"`` — the demo seed's
honesty-label precedent, applied to this construction.

Never duplicate bytes: tensors are stored through SI-7's canonical
per-tensor blobs in the content-addressed store; identical tensors
(the all-ones RMSNorm weights, the all-zero attention biases)
address the same blob and are stored exactly once.
"""
from __future__ import annotations

import json
import os

import numpy as np

from si.bundle import (
    _anchor_checkpoint,
    _write_manifest,
    canonical_json,
    load_bundle,
    sha256_hex,
    utc_now,
)
from si.seed import _canonical_tensor_blob, _store_blob

#: Fixed construction seed for the miniature. Probe streams mirror
#: DemoMind's discipline: identity draws from ``MINI_SEED + 2`` and
#: capability draws from ``MINI_SEED + 3`` (RFC §5.2/§5.3).
MINI_SEED = 20261020

#: Kind marker recorded in the manifest base + probe artifact.
TINY_KIND = "tiny-transformer-miniature-v1"

#: Working name from RFC SI-13 §3, settled here as the mind's name.
TINY_MIND_NAME = "TinyTransformerMind"

#: Honest manifest source label (see module docstring).
TINY_SOURCE_LABEL = (
    "tiny:constructed-miniature (SI-14; constructed miniature for "
    "probe development, not a trained model)"
)

#: Architecture descriptor (RFC §3.1 + the positional-table variant).
#: This dict is recorded verbatim in the manifest and the probe
#: artifact; SI-16's binding (§6.2 rule 3) matches against it.
ARCHITECTURE = {
    "name": TINY_MIND_NAME,
    "kind": TINY_KIND,
    "family": "pre-norm decoder-only transformer (RMSNorm, causal "
              "multi-head self-attention, SwiGLU MLP)",
    "vocab_size": 32,
    "d_model": 32,
    "layers": 2,
    "heads": 4,
    "head_dim": 8,
    "d_ff": 64,
    "max_seq_len": 16,
    "rms_norm_eps": 1e-5,
    "positional_embedding": True,
    "attention_biases": True,
    "mlp_biases": False,
    "embeddings_tied": False,
}

V = ARCHITECTURE["vocab_size"]
D_MODEL = ARCHITECTURE["d_model"]
LAYERS = ARCHITECTURE["layers"]
HEADS = ARCHITECTURE["heads"]
HEAD_DIM = ARCHITECTURE["head_dim"]
D_FF = ARCHITECTURE["d_ff"]
T_MAX = ARCHITECTURE["max_seq_len"]
RMS_EPS = ARCHITECTURE["rms_norm_eps"]

EMBED_NAME = "model.language_model.embed_tokens.weight"
POS_NAME = "model.language_model.embed_positions.weight"
FINAL_NORM_NAME = "model.language_model.norm.weight"
LM_HEAD_NAME = "lm_head.weight"
#: SI-33 (RFC SI-32 §12.1): the final layer's MLP down-projection
#: — the one non-default authorship target's tensor, named here
#: beside the readout it sits upstream of.
FINAL_MLP_DOWN_NAME = "model.language_model.layers.1.mlp.down_proj.weight"


def _layer_prefix(i: int) -> str:
    return f"model.language_model.layers.{i}"


def tiny_tensor_shapes() -> dict:
    """The exact tensor inventory: {name: shape} (RFC §3.1 + pos table)."""
    shapes = {
        EMBED_NAME: [V, D_MODEL],
        POS_NAME: [T_MAX, D_MODEL],
        FINAL_NORM_NAME: [D_MODEL],
        LM_HEAD_NAME: [V, D_MODEL],
    }
    for i in range(LAYERS):
        p = _layer_prefix(i)
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            shapes[f"{p}.self_attn.{proj}.weight"] = [D_MODEL, D_MODEL]
            shapes[f"{p}.self_attn.{proj}.bias"] = [D_MODEL]
        shapes[f"{p}.mlp.gate_proj.weight"] = [D_FF, D_MODEL]
        shapes[f"{p}.mlp.up_proj.weight"] = [D_FF, D_MODEL]
        shapes[f"{p}.mlp.down_proj.weight"] = [D_MODEL, D_FF]
        shapes[f"{p}.input_layernorm.weight"] = [D_MODEL]
        shapes[f"{p}.post_attention_layernorm.weight"] = [D_MODEL]
    return shapes


def build_tiny_weights(construction_seed: int = MINI_SEED,
                       readout_shift: int = 0) -> dict:
    """Construct the miniature's tensors deterministically. NOT training.

    Structured construction (RFC §3.3):

    - ``embed_tokens`` is residual-dominant: token ``t`` maps
      predominantly to residual direction ``t`` (2.0 * I + 0.01-scale
      noise), and ``lm_head`` is its approximate inverse readout
      (0.5 * permutation + 0.002-scale noise), so the last consumed
      token dominates the next-token logits — the echo behaviour
      §5.3 measures. ``readout_shift`` permutes the readout by that
      many positions (default 0 = the honest construction); a
      non-zero shift constructs a seed that CANNOT echo, which is
      how the tests prove the echo floor refuses bad seeds.
    - The positional table is small (0.01-scale) seeded noise.
    - Attention/MLP projection weights are small (0.02-scale, the
      DemoMind adapter precedent) but NON-ZERO, so every branch is
      live: a delta on any projection measurably moves the logits.
    - RMSNorm weights are exactly 1.0; attention biases exactly 0.0
      (the tensors exist so ``additive`` deltas have a real home).

    All draws come from one ``default_rng(construction_seed)`` in a
    fixed order, so construction is byte-for-byte reproducible.
    """
    rng = np.random.default_rng(construction_seed)

    def small(*shape):
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    weights: dict[str, np.ndarray] = {}
    weights[EMBED_NAME] = (
        2.0 * np.eye(V, D_MODEL) + rng.standard_normal((V, D_MODEL)) * 0.01
    ).astype(np.float32)
    weights[POS_NAME] = (
        rng.standard_normal((T_MAX, D_MODEL)) * 0.01).astype(np.float32)
    readout = np.zeros((V, D_MODEL), dtype=np.float64)
    for t in range(V):
        readout[(t + readout_shift) % V, t] = 0.5
    weights[LM_HEAD_NAME] = (
        readout + rng.standard_normal((V, D_MODEL)) * 0.002
    ).astype(np.float32)
    weights[FINAL_NORM_NAME] = np.ones((D_MODEL,), dtype=np.float32)
    for i in range(LAYERS):
        p = _layer_prefix(i)
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            weights[f"{p}.self_attn.{proj}.weight"] = small(D_MODEL, D_MODEL)
            weights[f"{p}.self_attn.{proj}.bias"] = np.zeros(
                (D_MODEL,), dtype=np.float32)
        weights[f"{p}.mlp.gate_proj.weight"] = small(D_FF, D_MODEL)
        weights[f"{p}.mlp.up_proj.weight"] = small(D_FF, D_MODEL)
        weights[f"{p}.mlp.down_proj.weight"] = small(D_MODEL, D_FF)
        weights[f"{p}.input_layernorm.weight"] = np.ones(
            (D_MODEL,), dtype=np.float32)
        weights[f"{p}.post_attention_layernorm.weight"] = np.ones(
            (D_MODEL,), dtype=np.float32)
    shapes = tiny_tensor_shapes()
    assert set(weights) == set(shapes), "construction/inventory drift"
    for name, arr in weights.items():
        assert list(arr.shape) == shapes[name], f"shape drift on {name}"
        assert arr.dtype == np.float32, f"dtype drift on {name}"
    return weights


# -- internal seed-time forward helper (wired to NOTHING; see docstring) -----
def _rms_norm(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(ms + RMS_EPS) * w


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def tiny_forward(token_ids, weights: dict) -> np.ndarray:
    """Last-position logits are what the suites record; this returns
    the full ``[len, V]`` logits. Seed-time helper ONLY (RFC §8
    SI-14 note): no gate, loop, or serve path imports this.

    Arithmetic is float64 over the float32 weights for a stable,
    deterministic seed-time measurement; the recorded expected
    logits are cast back to float32 for storage. (SI-15 owns the
    library forward path and its own arithmetic discipline.)
    """
    ids = [int(t) for t in token_ids]
    if not (1 <= len(ids) <= T_MAX):
        raise ValueError(f"sequence length {len(ids)} outside [1, {T_MAX}]")
    if any(t < 0 or t >= V for t in ids):
        raise ValueError(f"token id outside [0, {V}): {ids!r}")
    w = {k: np.asarray(v, dtype=np.float64) for k, v in weights.items()}
    t = len(ids)
    h = w[EMBED_NAME][ids] + w[POS_NAME][:t]
    causal = np.triu(np.ones((t, t), dtype=bool), k=1)
    for i in range(LAYERS):
        p = _layer_prefix(i)
        a = _rms_norm(h, w[f"{p}.input_layernorm.weight"])
        q = (a @ w[f"{p}.self_attn.q_proj.weight"].T
             + w[f"{p}.self_attn.q_proj.bias"]).reshape(t, HEADS, HEAD_DIM)
        k = (a @ w[f"{p}.self_attn.k_proj.weight"].T
             + w[f"{p}.self_attn.k_proj.bias"]).reshape(t, HEADS, HEAD_DIM)
        v = (a @ w[f"{p}.self_attn.v_proj.weight"].T
             + w[f"{p}.self_attn.v_proj.bias"]).reshape(t, HEADS, HEAD_DIM)
        q = np.transpose(q, (1, 0, 2))
        k = np.transpose(k, (1, 0, 2))
        v = np.transpose(v, (1, 0, 2))
        scores = q @ np.transpose(k, (0, 2, 1)) / np.sqrt(float(HEAD_DIM))
        scores = np.where(causal, -np.inf, scores)
        attn = _softmax(scores) @ v                       # [H, t, dh]
        attn = np.transpose(attn, (1, 0, 2)).reshape(t, D_MODEL)
        h = h + attn @ w[f"{p}.self_attn.o_proj.weight"].T \
            + w[f"{p}.self_attn.o_proj.bias"]
        m = _rms_norm(h, w[f"{p}.post_attention_layernorm.weight"])
        gate = m @ w[f"{p}.mlp.gate_proj.weight"].T
        up = m @ w[f"{p}.mlp.up_proj.weight"].T
        silu = gate / (1.0 + np.exp(-gate))
        h = h + (silu * up) @ w[f"{p}.mlp.down_proj.weight"].T
    h = _rms_norm(h, w[FINAL_NORM_NAME])
    return h @ w[LM_HEAD_NAME].T


# -- probe suites (RFC §5.2 / §5.3: fixed, frozen at seed time) --------------
def identity_inputs() -> list:
    """The 8 identity inputs, classes pinned by RFC §5.2."""
    inputs = [
        [7],                                            # 1: single token
        [3] * T_MAX,                                    # 2: repeated, max len
        [(7 * i + 3) % V for i in range(T_MAX)],        # 3: distinct, max len
        list(range(10)),                                # 4: ascending run
    ]
    rng = np.random.default_rng(MINI_SEED + 2)          # 5-8: pseudo-random
    for length in (4, 8, 12, 16):
        inputs.append([int(x) for x in rng.integers(0, V, size=length)])
    return inputs


def capability_inputs() -> list:
    """The 16 echo-capability inputs (RFC §5.3).

    Drawn from the capability stream (``MINI_SEED + 3``), lengths
    spread over 2..16, with the last token of entry ``i`` then set
    to ``(2 * i) % V`` so the suite's last tokens provably cover
    exactly half the vocabulary (16 distinct tokens) — the coverage
    §5.3 requires, guaranteed by construction rather than by luck
    of the draw. The echo target of an entry is its last token.
    """
    rng = np.random.default_rng(MINI_SEED + 3)
    lengths = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16]
    inputs = []
    for i, length in enumerate(lengths):
        seq = [int(x) for x in rng.integers(0, V, size=length)]
        seq[-1] = (2 * i) % V
        inputs.append(seq)
    return inputs


def _margin(logits: np.ndarray) -> float:
    """Top-1 minus top-2 logit gap (RFC §5.4 margin recording)."""
    s = np.sort(np.asarray(logits, dtype=np.float64))
    return float(s[-1] - s[-2])


def _measure_suites(weights: dict):
    """Run the internal forward over both suites; return the records
    + summary measurements that become the probe artifact."""
    identity = []
    for i, seq in enumerate(identity_inputs()):
        logits = tiny_forward(seq, weights)[-1].astype(np.float32)
        identity.append({
            "id": f"tiny-id-{i}",
            "input": [int(x) for x in seq],
            "expected_logits": [float(x) for x in logits],
            "baseline_error": 0.0,   # expected IS the base output (§5.4)
            "margin": _margin(logits),
        })
    capability = []
    correct = 0
    for i, seq in enumerate(capability_inputs()):
        logits = tiny_forward(seq, weights)[-1].astype(np.float32)
        target = int(seq[-1])
        if int(np.argmax(logits)) == target:
            correct += 1
        capability.append({
            "id": f"tiny-cap-{i}",
            "input": [int(x) for x in seq],
            "target_token": target,
            "expected_logits": [float(x) for x in logits],
            "baseline_error": 0.0,
            "margin": _margin(logits),
        })
    return identity, capability, correct / len(capability)


def seed_tiny_bundle(bundle_path: str, importer: str = "operator",
                     construction_seed: int = MINI_SEED,
                     readout_shift: int = 0) -> dict:
    """Seed a fresh bundle with the constructed miniature + probes.

    The bundle must already exist (``si init``) and have no history,
    exactly as for ``si seed``. All seed-time assertions run BEFORE
    anything is written; a failing construction raises ``ValueError``
    and leaves the bundle untouched. Re-seeding the identical
    miniature is an idempotent no-op (``already_seeded=True``),
    mirroring ``si.seed``'s contract.

    Returns a dict with the manifest base, stored/deduplicated byte
    accounting, and the measured probe summary (echo accuracy,
    margins, derived tolerances, probe artifact hash).
    """
    from si.journal import open_journal  # local import: no cycle

    bundle_dir = os.path.abspath(bundle_path)
    manifest = load_bundle(bundle_dir)  # fail closed on any tamper first

    # 1. Construct + measure, in memory only.
    weights = build_tiny_weights(construction_seed, readout_shift)
    identity, capability, echo_accuracy = _measure_suites(weights)

    # 2. Seed-time assertions, fail closed (RFC §3.3). Nothing has
    #    been written yet; raising here refuses the seed.
    again = build_tiny_weights(construction_seed, readout_shift)
    for name in weights:
        if not np.array_equal(weights[name], again[name]):
            raise ValueError(
                f"tiny seed refused: construction is not deterministic "
                f"(tensor {name} differs across constructions)")
    for entry in identity + capability:
        logits = tiny_forward(entry["input"], weights)[-1].astype(np.float32)
        if not np.array_equal(logits,
                              np.asarray(entry["expected_logits"],
                                         dtype=np.float32)):
            raise ValueError(
                f"tiny seed refused: forward is not deterministic "
                f"(probe {entry['id']})")
        if not np.all(np.isfinite(entry["expected_logits"])):
            raise ValueError(
                f"tiny seed refused: non-finite expected logits on "
                f"probe {entry['id']}")
    for name, arr in weights.items():
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"tiny seed refused: non-finite tensor {name}")
    if echo_accuracy < 15 / 16:
        raise ValueError(
            f"tiny seed refused: constructed base echo accuracy "
            f"{echo_accuracy:.4f} ({int(round(echo_accuracy * 16))}/16) "
            f"is below the 15/16 floor it claims (RFC SI-13 §3.3/§5.3); "
            f"no baselines were recorded and nothing was written")
    id_min_margin = min(e["margin"] for e in identity)
    cap_min_margin = min(e["margin"] for e in capability)
    identity_tol = min(0.25, 0.4 * id_min_margin)
    capability_eps = min(0.25, 0.4 * cap_min_margin)
    if identity_tol <= 0.0 or capability_eps <= 0.0:
        raise ValueError(
            "tiny seed refused: probe margins collapse to zero "
            f"(identity min margin {id_min_margin}, capability min "
            f"margin {cap_min_margin}); the derived tolerance would "
            "be one no honest delta could meet (RFC SI-13 §5.4)")

    # 3. Plan the per-tensor blobs (canonical SI-7 form, in memory).
    tensor_blobs: dict = {}
    planned: dict[str, bytes] = {}
    tensors_meta: dict = {}
    for name in sorted(weights):
        arr = np.ascontiguousarray(weights[name], dtype=np.float32)
        raw = arr.tobytes(order="C")
        blob_bytes = _canonical_tensor_blob("F32", list(arr.shape), raw)
        planned[name] = blob_bytes
        tensor_blobs[name] = {
            "blob": "sha256:" + sha256_hex(blob_bytes),
            "bytes": len(blob_bytes),
            "shape": [int(x) for x in arr.shape],
            "dtype": "float32",
        }
        tensors_meta[name] = {"shape": [int(x) for x in arr.shape],
                              "dtype": "float32"}
    total_elements = int(sum(int(np.prod(v["shape"]))
                             for v in tensors_meta.values()))
    total_logical = sum(ref["bytes"] for ref in tensor_blobs.values())

    # 4. The probe artifact (RFC §5.5). Canonical JSON, no wall-clock
    #    anywhere in it: deterministic re-seeding must produce an
    #    identical artifact (acceptance criterion).
    tensor_map_hash = "sha256:" + sha256_hex(canonical_json(tensor_blobs))
    artifact = {
        "artifact": "si-probe-suite-v1",
        "kind": TINY_KIND,
        "mind": TINY_MIND_NAME,
        "honesty": "constructed miniature for probe development, "
                   "not a trained model; probe results on this "
                   "bundle say something about deltas against this "
                   "miniature only — nothing about any other base",
        "construction_seed": construction_seed,
        "suite_generation": (
            "identity: 4 pinned corner sequences + 4 draws from "
            "numpy.random.default_rng(construction_seed + 2), "
            "lengths 4/8/12/16; capability: 16 draws from "
            "numpy.random.default_rng(construction_seed + 3), "
            "lengths 2..16, last token of entry i set to (2*i) % 32 "
            "for provable half-vocabulary coverage; expected logits "
            "are the constructed base's last-position logits, "
            "measured at seed time by the internal seed-time "
            "forward helper in si/tiny_seed.py"),
        "architecture": dict(ARCHITECTURE),
        "base_binding": {
            "tensor_map_sha256": tensor_map_hash,
            "tensor_count": len(tensor_blobs),
            "total_elements": total_elements,
        },
        "identity_suite": identity,
        "capability_suite": capability,
        "baselines": {
            "identity": {e["id"]: 0.0 for e in identity},
            "capability": {e["id"]: 0.0 for e in capability},
            "note": "0.0 by construction at head=base: each entry's "
                    "expected output IS the base forward's output "
                    "(recorded explicitly, not assumed; RFC §5.4)",
        },
        "base_echo_accuracy": echo_accuracy,
        "echo_floor": echo_accuracy - 1 / 16,
        "margins": {
            "identity_min": id_min_margin,
            "capability_min": cap_min_margin,
            "identity_per_entry": {e["id"]: e["margin"] for e in identity},
            "capability_per_entry": {e["id"]: e["margin"]
                                     for e in capability},
        },
        "tolerances": {
            "identity_tol": identity_tol,
            "capability_eps": capability_eps,
            "derivation": "min(0.25, 0.4 * min_margin) over the "
                          "respective suite, computed and recorded "
                          "at seed time (RFC SI-13 §5.4)",
        },
    }
    artifact_bytes = canonical_json(artifact)
    artifact_hash = sha256_hex(artifact_bytes)

    # 5. Idempotent re-seed of the identical miniature: no writes.
    base = manifest["base"]
    if base.get("kind") == TINY_KIND \
            and base.get("tensor_blobs") == tensor_blobs \
            and base.get("probe_artifact", {}).get("sha256") \
            == "sha256:" + artifact_hash:
        return {"base": base, "tensors": len(tensor_blobs),
                "elements": total_elements, "bytes": base.get("bytes", 0),
                "stored_bytes": base.get("bytes", 0),
                "deduplicated_bytes": total_logical,
                "already_seeded": True, "head": manifest["head"],
                "echo_accuracy": echo_accuracy,
                "identity_tol": identity_tol,
                "capability_eps": capability_eps,
                "probe_artifact_sha256": "sha256:" + artifact_hash}

    # 6. Refuse bundles with history (the base is frozen).
    if manifest.get("deltas") or manifest.get("head") != "base":
        raise ValueError(
            "refusing to seed a bundle with existing history "
            f"(deltas={len(manifest.get('deltas', []))}, "
            f"head={manifest.get('head')!r})")

    # 7. Store the blobs, deduplicating globally (never duplicate
    #    bytes: an existing blob is never rewritten or copied).
    deduplicated_bytes = 0
    stored_refs: set = set()
    for name in sorted(planned):
        hexpart, existed = _store_blob(bundle_dir, planned[name])
        assert "sha256:" + hexpart == tensor_blobs[name]["blob"]
        if existed:
            deduplicated_bytes += len(planned[name])
        stored_refs.add(hexpart)
    stored_bytes = sum(
        os.path.getsize(os.path.join(bundle_dir, "blobs", "sha256",
                                     h + ".safetensors"))
        for h in stored_refs)

    # 8. Probe artifact file + per-bundle policy tolerances. The
    #    policy write updates policy.json and the manifest's
    #    policy_hash together, as a seed-time act journaled below.
    with open(os.path.join(bundle_dir, "probe_suite.json"), "wb") as f:
        f.write(artifact_bytes)
    from si.bundle import load_policy
    policy = load_policy(bundle_dir)
    policy["identity_tol"] = identity_tol
    policy["capability_eps"] = capability_eps
    policy_bytes = canonical_json(policy)
    with open(os.path.join(bundle_dir, "policy.json"), "wb") as f:
        f.write(policy_bytes)
    manifest["constitution"] = dict(manifest["constitution"])
    manifest["constitution"]["policy_hash"] = "sha256:" + sha256_hex(
        policy_bytes)

    # 9. Rewrite the base (ordinary per-tensor seed, SI-7 form).
    utc = utc_now()
    manifest["base"] = {
        "source": TINY_SOURCE_LABEL,
        "format": "safetensors",
        "seed_format": "per-tensor-v1",
        "kind": TINY_KIND,
        "mind": TINY_MIND_NAME,
        "architecture": dict(ARCHITECTURE),
        "construction_seed": construction_seed,
        "blob": tensor_blobs[EMBED_NAME]["blob"],
        "parts": [],
        "tensor_blobs": tensor_blobs,
        "tensors": tensors_meta,
        "bytes": stored_bytes,
        "imported_bytes": total_logical,
        "deduplicated_bytes": deduplicated_bytes,
        "tensor_count": len(tensor_blobs),
        "total_elements": total_elements,
        "probe_artifact": {
            "file": "probe_suite.json",
            "sha256": "sha256:" + artifact_hash,
            "bytes": len(artifact_bytes),
        },
        "importer": importer,
        "imported_at": utc,
    }

    # 10. Provenance + re-anchor (journal tip must match on load).
    journal = open_journal(bundle_dir, manifest)
    try:
        journal.record_provenance(
            child="base",
            parents=[f"construction-seed:{construction_seed}"],
            relation="seeded_from",
            note=(
                f"constructed {len(tensor_blobs)} tensors, "
                f"{total_elements} elements via si seed-tiny "
                f"(structured deterministic construction, NOT "
                f"trained; stored {stored_bytes} unique blob bytes, "
                f"deduplicated {deduplicated_bytes} bytes); "
                f"source={TINY_SOURCE_LABEL}"),
        )
        journal.record_provenance(
            child="probe_suite",
            parents=["base"],
            relation="probe_suite_recorded",
            note=(
                f"probe_suite.json sha256={artifact_hash}; "
                f"identity 8 entries, capability 16 entries; "
                f"base echo accuracy {echo_accuracy} "
                f"({int(round(echo_accuracy * 16))}/16, floor "
                f"{echo_accuracy - 1 / 16}); identity min margin "
                f"{id_min_margin}, capability min margin "
                f"{cap_min_margin}; identity_tol={identity_tol}, "
                f"capability_eps={capability_eps} "
                f"(min(0.25, 0.4*min_margin), RFC SI-13 §5.4); "
                f"base tensor map {tensor_map_hash}; probes are "
                f"recorded, NOT consumed by any gate/loop/serve "
                f"path in SI-14"),
        )
        manifest = _anchor_checkpoint(
            journal, manifest, head="base", note="seed tiny-miniature")
        _write_manifest(bundle_dir, manifest)
    finally:
        journal.close()

    return {"base": manifest["base"], "tensors": len(tensor_blobs),
            "elements": total_elements, "bytes": stored_bytes,
            "stored_bytes": stored_bytes,
            "deduplicated_bytes": deduplicated_bytes,
            "seed_format": manifest["base"]["seed_format"],
            "already_seeded": False, "head": "base",
            "echo_accuracy": echo_accuracy,
            "identity_tol": identity_tol,
            "capability_eps": capability_eps,
            "identity_min_margin": id_min_margin,
            "capability_min_margin": cap_min_margin,
            "probe_artifact_sha256": "sha256:" + artifact_hash}
