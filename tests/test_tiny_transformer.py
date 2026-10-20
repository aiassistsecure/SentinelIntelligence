"""SI-15: the miniature forward path (RFC SI-13 §4, §8 SI-15).

Covers the SI-15 acceptance criteria, in order:

1. a hand-computed single-layer reference case matches the
   implementation EXACTLY (weights crafted so both cases are
   verifiable by hand in the test: identity blocks, and one active
   attention layer whose contribution is a single scalar derived
   with ``math.sqrt``);
2. multi-layer outputs match an independent naive reimplementation
   written in this file from RFC §4.1 (explicit Python loops over
   positions/heads — not copied from the module);
3. repeat runs are bit-identical;
4. overlay-vs-materialised agreement: a fixture delta applied via
   the §4.2 candidate overlay and via actually materialising it on
   the chain agree EXACTLY (weights and logits);
5. rolled-back (inactive) deltas do not affect the forward;
6. the forward refuses any architecture descriptor other than
   §3.1's (as amended by SI-14);
7. gate behaviour is byte-for-byte unchanged: checks 3/4 still
   "not evaluated" for a transformer delta against the miniature,
   verdict ESCALATE, and the gate never touches a probe method.

Plus: the library forward reproduces SI-14's recorded probe-suite
expected logits for the seeded base BIT-IDENTICALLY (the module's
stated arithmetic discipline: float32 weights/factors, float64
compute mirroring the seed-time helper, float32 logits out); the
candidate overlay validates before trusting factors and never reads
the candidate for untargeted tensors; the adapter's train_delta
refuses explicitly (§4.4); nothing is wired to gate/loop/serve.
"""
import copy
import json
import math
import os

import numpy as np
import pytest

from si.apply import ApplyError, applied_tensor_with_candidate, load_applied_tensor
from si.bundle import (
    _anchor_checkpoint,
    _write_manifest,
    init_bundle,
    load_bundle,
)
from si.journal import open_journal
from si.loop import _write_blob
from si.tiny_seed import build_tiny_weights, seed_tiny_bundle, tiny_tensor_shapes
from si.tiny_transformer import (
    EXPECTED_ARCHITECTURE,
    TinyTransformerError,
    TinyTransformerMind,
    assemble_weights,
    forward,
    forward_from_bundle,
    validate_architecture,
)
from si.transformer_delta import (
    DeltaValidationError,
    construct_transformer_delta,
)

QPROJ = "model.language_model.layers.0.self_attn.q_proj.weight"
NORM0 = "model.language_model.layers.0.input_layernorm.weight"
EPS = 1e-5


def _init(tmp_path, name):
    return init_bundle(str(tmp_path / (name + ".si")), name=name,
                       journal_backend="embedded")


def _seeded(tmp_path, name="tiny"):
    bundle_dir = _init(tmp_path, name)
    seed_tiny_bundle(bundle_dir)
    return bundle_dir


def _artifact(bundle_dir):
    with open(os.path.join(bundle_dir, "probe_suite.json"), "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def _persist(bundle_dir, manifest, note):
    journal = open_journal(bundle_dir, manifest)
    try:
        anchored = _anchor_checkpoint(journal, manifest,
                                      head=manifest["head"], note=note)
        _write_manifest(bundle_dir, anchored)
    finally:
        journal.close()
    return load_bundle(bundle_dir)


def _stack_delta(bundle_dir, specs, delta_id):
    """Materialise a delta on the chain (the state a landed delta
    would be in). Returns (reloaded manifest, record, blob_bytes)."""
    manifest = load_bundle(bundle_dir)
    record, blob = construct_transformer_delta(
        manifest, specs, delta_id=delta_id, parent=manifest["head"],
        created_at="2026-10-20T00:00:00Z", self_edit_id="se_" + delta_id)
    _write_blob(bundle_dir, blob)
    manifest["deltas"].append(record)
    manifest["head"] = delta_id
    return _persist(bundle_dir, manifest, f"stack {delta_id}"), record, blob


def _move_head(bundle_dir, to):
    manifest = load_bundle(bundle_dir)
    manifest["head"] = to
    return _persist(bundle_dir, manifest, f"head -> {to}")


def _lora_spec(name, rank=2, alpha=4, seed=1, scale=0.05):
    rng = np.random.default_rng(seed)
    shape = tiny_tensor_shapes()[name]
    return {"name": name, "kind": "lora",
            "A": (rng.standard_normal((rank, shape[1])) * scale).astype(np.float32),
            "B": (rng.standard_normal((shape[0], rank)) * scale).astype(np.float32),
            "rank": rank, "alpha": alpha}


def _candidate(manifest, specs, delta_id="dcand"):
    record, blob = construct_transformer_delta(
        manifest, specs, delta_id=delta_id, parent=manifest["head"],
        created_at="2026-10-20T00:00:00Z", self_edit_id="se_" + delta_id)
    return {"record": record, "blob_bytes": blob}


# ---------------------------------------------------------------------------
# Criterion 1: hand-computed reference cases, exact
# ---------------------------------------------------------------------------

def _blank_weights():
    return {n: np.zeros(s, dtype=np.float32)
            for n, s in tiny_tensor_shapes().items()}


def _hand_weights():
    """embed = 2*I, positional = 0, lm_head = I, all norms = 1,
    every projection/MLP weight and bias = 0."""
    w = _blank_weights()
    w["model.language_model.embed_tokens.weight"] = \
        (2.0 * np.eye(32)).astype(np.float32)
    w["lm_head.weight"] = np.eye(32, dtype=np.float32)
    for i in range(2):
        p = f"model.language_model.layers.{i}"
        w[f"{p}.input_layernorm.weight"] = np.ones(32, dtype=np.float32)
        w[f"{p}.post_attention_layernorm.weight"] = \
            np.ones(32, dtype=np.float32)
    w["model.language_model.norm.weight"] = np.ones(32, dtype=np.float32)
    return w


def test_hand_computed_identity_blocks_exact():
    """With every branch zeroed, each transformer block is an exact
    identity (attention output is softmax @ 0 = 0 through a zero
    o_proj; the MLP is silu(0)*0 through a zero down_proj), so by
    hand: h = embed[5] = 2*e5; final RMSNorm divides by
    sqrt(mean(h^2)+eps) = sqrt(4/32 + 1e-5); lm_head = I passes the
    normalised vector through. Expected built with math.sqrt only."""
    out = forward([5], _hand_weights())
    expected = np.zeros(32, dtype=np.float32)
    expected[5] = np.float32(2.0 / math.sqrt(4.0 / 32.0 + EPS))
    assert out.shape == (1, 32) and out.dtype == np.float32
    np.testing.assert_array_equal(out[0], expected)


def test_hand_computed_single_active_layer_exact():
    """One active layer, verified by hand: layer 0 gets v_proj = o_proj
    = I (q/k stay zero). For the single token 5:
      a  = RMSNorm(2*e5)         -> s*e5,  s = 2/sqrt(4/32+eps)
      scores = [0] (q = k = 0)   -> softmax = [1] exactly
      attn = 1 * v = a           -> h = 2*e5 + s*e5 = (2+s)*e5
    Layer 1 is an identity block (zero branches, as above), the MLP
    of layer 0 is zero, so by hand the logits are the final RMSNorm
    of (2+s)*e5 read out by lm_head = I."""
    w = _hand_weights()
    p0 = "model.language_model.layers.0"
    w[f"{p0}.self_attn.v_proj.weight"] = np.eye(32, dtype=np.float32)
    w[f"{p0}.self_attn.o_proj.weight"] = np.eye(32, dtype=np.float32)
    out = forward([5], w)
    s = 2.0 / math.sqrt(4.0 / 32.0 + EPS)
    h1 = 2.0 + s
    expected = np.zeros(32, dtype=np.float32)
    expected[5] = np.float32(h1 / math.sqrt(h1 * h1 / 32.0 + EPS))
    np.testing.assert_array_equal(out[0], expected)


# ---------------------------------------------------------------------------
# Criterion 2: independent naive reimplementation (written from §4.1)
# ---------------------------------------------------------------------------

def _naive_forward(token_ids, weights):
    """A from-the-RFC reimplementation with explicit Python loops:
    per position, per head, per score — lists and math only, no
    numpy linear algebra. Deliberately NOT the module's code."""
    W = {k: [[float(x) for x in row] for row in np.asarray(v, dtype=np.float64)]
         if np.asarray(v).ndim == 2 else
         [float(x) for x in np.asarray(v, dtype=np.float64)]
         for k, v in weights.items()}
    V, D, H, DH, FF = 32, 32, 4, 8, 64
    t = len(token_ids)

    def rms(vec, wname):
        ms = sum(x * x for x in vec) / len(vec)
        inv = 1.0 / math.sqrt(ms + EPS)
        return [x * inv * g for x, g in zip(vec, W[wname])]

    def linear(vec, wname, bname=None):
        M = W[wname]
        out = [sum(vec[j] * M[i][j] for j in range(len(vec)))
               for i in range(len(M))]
        if bname is not None:
            out = [o + b for o, b in zip(out, W[bname])]
        return out

    h = [[W["model.language_model.embed_tokens.weight"][tok][j]
          + W["model.language_model.embed_positions.weight"][pos][j]
          for j in range(D)] for pos, tok in enumerate(token_ids)]
    for layer in range(2):
        p = f"model.language_model.layers.{layer}"
        a = [rms(h[pos], f"{p}.input_layernorm.weight") for pos in range(t)]
        qs = [linear(a[pos], f"{p}.self_attn.q_proj.weight",
                     f"{p}.self_attn.q_proj.bias") for pos in range(t)]
        ks = [linear(a[pos], f"{p}.self_attn.k_proj.weight",
                     f"{p}.self_attn.k_proj.bias") for pos in range(t)]
        vs = [linear(a[pos], f"{p}.self_attn.v_proj.weight",
                     f"{p}.self_attn.v_proj.bias") for pos in range(t)]
        attn = [[0.0] * D for _ in range(t)]
        for head in range(H):
            sl = slice(head * DH, (head + 1) * DH)
            for pos in range(t):
                scores = []
                for src in range(t):
                    if src > pos:
                        scores.append(None)  # causal mask
                    else:
                        scores.append(
                            sum(qs[pos][sl][d] * ks[src][sl][d]
                                for d in range(DH)) / math.sqrt(DH))
                live = [s for s in scores if s is not None]
                mx = max(live)
                exps = [0.0 if s is None else math.exp(s - mx)
                        for s in scores]
                denom = sum(exps)
                for d in range(DH):
                    attn[pos][head * DH + d] = sum(
                        exps[src] / denom * vs[src][sl][d]
                        for src in range(t))
        for pos in range(t):
            o = linear(attn[pos], f"{p}.self_attn.o_proj.weight",
                       f"{p}.self_attn.o_proj.bias")
            h[pos] = [x + y for x, y in zip(h[pos], o)]
        for pos in range(t):
            m = rms(h[pos], f"{p}.post_attention_layernorm.weight")
            gate = linear(m, f"{p}.mlp.gate_proj.weight")
            up = linear(m, f"{p}.mlp.up_proj.weight")
            act = [g / (1.0 + math.exp(-g)) * u for g, u in zip(gate, up)]
            down = linear(act, f"{p}.mlp.down_proj.weight")
            h[pos] = [x + y for x, y in zip(h[pos], down)]
    out = []
    for pos in range(t):
        n = rms(h[pos], "model.language_model.norm.weight")
        out.append(linear(n, "lm_head.weight"))
    return np.asarray(out, dtype=np.float64)


def test_multilayer_matches_independent_naive_reimplementation():
    weights = build_tiny_weights()
    for seq in ([7], [3, 1, 4, 1, 5], list(range(10)),
                [31, 0, 17, 17, 2, 9, 4, 4, 4, 8, 13, 21]):
        got = forward(seq, weights)
        want = _naive_forward(seq, weights)
        # Tolerance is the float32 OUTPUT rounding, stated plainly:
        # the module returns float32 logits (§4.1 contract) while
        # the naive reimplementation stays in float64, so agreement
        # is bounded by one float32 ulp at these magnitudes
        # (observed max diff ~1e-7), not by the float64 compute,
        # whose loop-vs-numpy summation-order noise is far smaller.
        np.testing.assert_allclose(
            got.astype(np.float64), want, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Criterion 3 + probe reproduction: bit-identical repeats and baselines
# ---------------------------------------------------------------------------

def test_repeat_runs_are_bit_identical(tmp_path):
    weights = build_tiny_weights()
    seq = [5, 3, 8, 13, 21]
    np.testing.assert_array_equal(forward(seq, weights),
                                  forward(seq, weights))
    bundle_dir = _seeded(tmp_path)
    manifest = load_bundle(bundle_dir)
    np.testing.assert_array_equal(
        forward_from_bundle(bundle_dir, manifest, seq),
        forward_from_bundle(bundle_dir, manifest, seq))
    w1 = assemble_weights(bundle_dir, manifest)
    w2 = assemble_weights(bundle_dir, manifest)
    for name in w1:
        np.testing.assert_array_equal(w1[name], w2[name])


def test_forward_reproduces_recorded_probe_logits_bit_identically(tmp_path):
    """The §4.3 discipline claim, proven: the library forward over
    the seeded base (assembled via load_applied_tensor) reproduces
    every expected logit SI-14 recorded — exactly, all 24 entries."""
    bundle_dir = _seeded(tmp_path)
    manifest = load_bundle(bundle_dir)
    art = _artifact(bundle_dir)
    checked = 0
    for suite in ("identity_suite", "capability_suite"):
        for entry in art[suite]:
            logits = forward_from_bundle(
                bundle_dir, manifest, entry["input"])[-1]
            np.testing.assert_array_equal(
                logits,
                np.asarray(entry["expected_logits"], dtype=np.float32))
            checked += 1
    assert checked == 24


# ---------------------------------------------------------------------------
# Criterion 4: overlay vs materialised agreement (exact)
# ---------------------------------------------------------------------------

def test_overlay_vs_materialised_agreement_exact(tmp_path):
    bundle_dir = _seeded(tmp_path)
    base_manifest = load_bundle(bundle_dir)
    spec = _lora_spec(QPROJ, seed=21)
    cand = _candidate(base_manifest, [spec])
    seqs = ((7,), (3, 1, 4, 1, 5), tuple(range(10)))

    # Overlay way: candidate never lands.
    overlaid_w = {name: applied_tensor_with_candidate(
        bundle_dir, base_manifest, name, cand)
        for name in tiny_tensor_shapes()}
    overlaid_logits = {s: forward_from_bundle(
        bundle_dir, base_manifest, list(s), candidate=cand) for s in seqs}
    assert load_bundle(bundle_dir)["head"] == "base"  # nothing landed

    # Materialised way: the same delta actually stacked on the chain.
    manifest, _record, _blob = _stack_delta(bundle_dir, [spec], "d0001")
    landed_w = assemble_weights(bundle_dir, manifest)
    for name in overlaid_w:
        np.testing.assert_array_equal(overlaid_w[name], landed_w[name])
    for s in seqs:
        np.testing.assert_array_equal(
            overlaid_logits[s],
            forward_from_bundle(bundle_dir, manifest, list(s)))
    # And the delta is real: the targeted weight differs from base,
    # and so do multi-token logits. (A single-token sequence is NOT
    # evidence either way for a q_proj delta: softmax over one
    # position is 1 whatever q/k are, so [7] cannot move.)
    base_w = assemble_weights(bundle_dir, base_manifest)
    assert not np.array_equal(overlaid_w[QPROJ], base_w[QPROJ])
    assert not np.array_equal(
        overlaid_logits[(3, 1, 4, 1, 5)],
        forward_from_bundle(bundle_dir, base_manifest, [3, 1, 4, 1, 5]))


def test_overlay_validation_and_untargeted_discipline(tmp_path):
    bundle_dir = _seeded(tmp_path)
    manifest = load_bundle(bundle_dir)
    spec = _lora_spec(QPROJ, seed=22)
    cand = _candidate(manifest, [spec])

    # Untargeted tensor: returned untouched — exactly the chain result.
    other = "model.language_model.layers.1.mlp.gate_proj.weight"
    np.testing.assert_array_equal(
        applied_tensor_with_candidate(bundle_dir, manifest, other, cand),
        load_applied_tensor(bundle_dir, manifest, other))

    # Tampered candidate: for the TARGETED tensor, SI-9a validation
    # runs first and refuses; for an untargeted tensor the candidate
    # is never read at all (RFC §4.2 / §5.1 discipline).
    bad = {"record": cand["record"],
           "blob_bytes": cand["blob_bytes"][:-1] + bytes(
               [cand["blob_bytes"][-1] ^ 0xFF])}
    with pytest.raises(DeltaValidationError):
        applied_tensor_with_candidate(bundle_dir, manifest, QPROJ, bad)
    np.testing.assert_array_equal(
        applied_tensor_with_candidate(bundle_dir, manifest, other, bad),
        load_applied_tensor(bundle_dir, manifest, other))

    # No candidate at all == the plain chain result.
    np.testing.assert_array_equal(
        applied_tensor_with_candidate(bundle_dir, manifest, QPROJ, None),
        load_applied_tensor(bundle_dir, manifest, QPROJ))


def test_additive_overlay_matches_hand_computation_and_policy(tmp_path):
    bundle_dir = _seeded(tmp_path)
    manifest = load_bundle(bundle_dir)
    rng = np.random.default_rng(23)
    d = (rng.standard_normal(32) * 0.05).astype(np.float32)
    cand = _candidate(manifest, [{"name": NORM0, "kind": "additive", "d": d}])

    with pytest.raises(ApplyError, match="allow_additive"):
        applied_tensor_with_candidate(bundle_dir, manifest, NORM0, cand)
    got = applied_tensor_with_candidate(
        bundle_dir, manifest, NORM0, cand, policy={"allow_additive": True})
    base = load_applied_tensor(bundle_dir, manifest, NORM0)
    np.testing.assert_array_equal(got, base + d)  # w + d, float32, exact


# ---------------------------------------------------------------------------
# Criterion 5: rolled-back deltas do not affect the forward
# ---------------------------------------------------------------------------

def test_rolled_back_deltas_do_not_affect_forward(tmp_path):
    bundle_dir = _seeded(tmp_path)
    base_manifest = load_bundle(bundle_dir)
    seq = [3, 1, 4, 1, 5]
    base_logits = forward_from_bundle(bundle_dir, base_manifest, seq)

    spec1 = _lora_spec(QPROJ, seed=31)
    spec2 = _lora_spec(QPROJ, seed=32)
    manifest, _, _ = _stack_delta(bundle_dir, [spec1], "d0001")
    one_logits = forward_from_bundle(bundle_dir, manifest, seq)
    manifest, _, _ = _stack_delta(bundle_dir, [spec2], "d0002")
    assert not np.array_equal(
        forward_from_bundle(bundle_dir, manifest, seq), base_logits)
    assert not np.array_equal(one_logits, base_logits)

    manifest = _move_head(bundle_dir, "d0001")
    np.testing.assert_array_equal(
        forward_from_bundle(bundle_dir, manifest, seq), one_logits)
    manifest = _move_head(bundle_dir, "base")
    np.testing.assert_array_equal(
        forward_from_bundle(bundle_dir, manifest, seq), base_logits)


# ---------------------------------------------------------------------------
# Criterion 6: refuse any other architecture descriptor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key,bad", [
    ("vocab_size", 64), ("d_model", 64), ("layers", 3), ("heads", 8),
    ("head_dim", 16), ("d_ff", 128), ("max_seq_len", 32),
    ("positional_embedding", False), ("rms_norm_eps", 1e-6),
    ("embeddings_tied", True),
])
def test_forward_refuses_other_architectures(tmp_path, key, bad):
    bundle_dir = _seeded(tmp_path)
    manifest = load_bundle(bundle_dir)
    mutated = copy.deepcopy(manifest)
    mutated["base"]["architecture"][key] = bad
    with pytest.raises(TinyTransformerError):
        forward_from_bundle(bundle_dir, mutated, [1, 2, 3])
    arch = dict(EXPECTED_ARCHITECTURE)
    arch[key] = bad
    with pytest.raises(TinyTransformerError):
        forward([1, 2, 3], build_tiny_weights(), architecture=arch)
    with pytest.raises(TinyTransformerError):
        validate_architecture(arch)


def test_forward_refuses_missing_or_extra_descriptor_keys():
    arch = dict(EXPECTED_ARCHITECTURE)
    del arch["heads"]
    with pytest.raises(TinyTransformerError):
        validate_architecture(arch)
    arch = dict(EXPECTED_ARCHITECTURE)
    arch["extra"] = 1
    with pytest.raises(TinyTransformerError):
        validate_architecture(arch)
    with pytest.raises(TinyTransformerError):
        validate_architecture(None)
    assert validate_architecture(dict(EXPECTED_ARCHITECTURE)) == \
        EXPECTED_ARCHITECTURE


def test_forward_refuses_bad_weights_and_sequences():
    weights = build_tiny_weights()
    missing = dict(weights)
    del missing[QPROJ]
    with pytest.raises(TinyTransformerError, match="inventory"):
        forward([1], missing)
    extra = dict(weights)
    extra["model.nope.weight"] = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(TinyTransformerError, match="inventory"):
        forward([1], extra)
    wrong = dict(weights)
    wrong[QPROJ] = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(TinyTransformerError, match="shape"):
        forward([1], wrong)
    with pytest.raises(TinyTransformerError, match="length"):
        forward([], weights)
    with pytest.raises(TinyTransformerError, match="length"):
        forward(list(range(17)), weights)
    with pytest.raises(TinyTransformerError, match="token id"):
        forward([32], weights)
    with pytest.raises(TinyTransformerError, match="token id"):
        forward([-1], weights)


# ---------------------------------------------------------------------------
# Criterion 7: gate behaviour byte-for-byte unchanged
# ---------------------------------------------------------------------------

class _FakeCtx:
    """The probe methods RAISE — a transformer gate evaluation that
    touches them in SI-15 is faking a check (same discipline as
    tests/test_tiny_seed.py / test_gate_transformer.py)."""

    def __init__(self, manifest):
        self._manifest = manifest

    def base_manifest(self):
        return self._manifest

    def stack_depth(self):
        return 0

    def identity_probes(self):
        raise AssertionError("SI-15 gate must not run identity probes")

    def capability_suite(self):
        raise AssertionError("SI-15 gate must not run capability probes")

    def capability_baseline(self):
        raise AssertionError("SI-15 gate must not read a baseline")

    def forward_candidate(self, x, candidate):
        raise AssertionError("SI-15 gate must not run a forward pass")


def test_gate_behaviour_unchanged_miniature_still_escalates(tmp_path):
    from si.bundle import DEFAULT_POLICY, load_policy
    from si.gate import Gate, Intent
    from si.transformer_delta import KIND

    bundle_dir = _seeded(tmp_path)
    manifest = load_bundle(bundle_dir)
    # Exercise the new forward path first: using it must not change
    # anything about how the gate treats the same bundle.
    forward_from_bundle(bundle_dir, manifest, [1, 2, 3])

    cand = _candidate(manifest, [_lora_spec(QPROJ, seed=41)], "d0001")
    candidate = {"record": cand["record"], "blob_bytes": cand["blob_bytes"]}
    policy = dict(DEFAULT_POLICY)
    policy["allowed_target_patterns"] = [QPROJ]
    policy["identity_tol"] = load_policy(bundle_dir)["identity_tol"]
    policy["capability_eps"] = load_policy(bundle_dir)["capability_eps"]
    intent = Intent(kind=KIND, reason="probe the SI-15 gate contract",
                    parents=["base"], source_seqs=[1], targets=[QPROJ])
    report = Gate(policy).evaluate(intent, candidate, _FakeCtx(manifest))

    assert report.verdict == "ESCALATE"  # never ADMIT in SI-15
    checks = {c.name: c for c in report.checks}
    for name in ("identity_continuity", "capability_floor"):
        assert checks[name].passed is False
        assert checks[name].measured is None
        assert "not evaluated" in checks[name].detail


# ---------------------------------------------------------------------------
# Adapter (§4.4): forward/probe surface works; training refuses explicitly
# ---------------------------------------------------------------------------

def test_adapter_surface_and_explicit_refusals(tmp_path):
    bundle_dir = _seeded(tmp_path)
    mind = TinyTransformerMind(bundle_dir)
    assert mind.name == "TinyTransformerMind"

    art = _artifact(bundle_dir)
    probes = mind.identity_probes()
    assert [p[0] for p in probes] == [e["id"] for e in art["identity_suite"]]
    qid, x, expected = probes[0]
    np.testing.assert_array_equal(mind.forward(x)[-1], expected)
    assert [c[0] for c in mind.capability_suite()] == \
        [e["id"] for e in art["capability_suite"]]
    assert mind.capability_baseline() == \
        {e["id"]: 0.0 for e in art["capability_suite"]}

    manifest = load_bundle(bundle_dir)
    cand = _candidate(manifest, [_lora_spec(QPROJ, seed=51)])
    np.testing.assert_array_equal(
        mind.forward_candidate([7], cand),
        forward_from_bundle(bundle_dir, manifest, [7], candidate=cand))

    from si.gate import Intent
    with pytest.raises(NotImplementedError, match="refuses"):
        mind.train_delta(Intent(kind="delta", reason="x"))
    with pytest.raises(NotImplementedError, match="does not propose"):
        mind.propose([], None)
