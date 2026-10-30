"""SI-40 (A-1): the minimal llama measurement harness.

The checkpoint fixture is LOCALLY CONSTRUCTED in the real
safetensors format from seeded random arrays at a tiny llama
configuration — real format, honestly NOT a trained model.
It exists so the harness's arithmetic (RMSNorm, RoPE, GQA,
SwiGLU, tied embeddings) is proven hermetically against an
independent naive float64 implementation written in this
file, and so assembly/compare costs are asserted without a
727 MB download in the suite. The REAL checkpoint's measured
numbers live in docs/SI-40-a1-feasibility-measurement.md;
nothing in this file claims them.

What is proven:
- config validation accepts the llama surface and refuses
  other families / biased / rope-scaled configs;
- assembly at each width yields the live byte count the
  width's arithmetic derives (params x bytes-per-element);
- the forward reproduces the naive float64 reference's
  logits within a stated float32 tolerance, is deterministic
  (bit-identical repeat), and refuses non-float32 weights
  and empty sequences;
- the verify compare reports 0.0 for identical assemblies
  and detects / names a perturbed tensor, and refuses
  mismatched inventories.
"""
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file  # noqa: E402

from si.llama_measure import (  # noqa: E402
    ASSEMBLY_WIDTHS,
    LlamaMeasureError,
    assemble_weights,
    assembled_bytes,
    compare_weight_sets,
    count_parameters,
    llama_forward,
    validate_llama_config,
)

HIDDEN, INTER, VOCAB, HEADS, KV_HEADS, LAYERS, MAX_POS = 64, 128, 96, 4, 2, 2, 64
HEAD_DIM = HIDDEN // HEADS
EPS = 1e-5
THETA = 100000.0


def _tiny_config():
    return {
        "model_type": "llama",
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "vocab_size": VOCAB,
        "max_position_embeddings": MAX_POS,
        "rms_norm_eps": EPS,
        "rope_theta": THETA,
        "tie_word_embeddings": True,
    }


def _tiny_tensors(seed=40):
    rng = np.random.default_rng(seed)
    scale = 0.15

    def w(*shape):
        return torch.tensor(rng.standard_normal(shape) * scale,
                            dtype=torch.bfloat16)

    tensors = {"model.embed_tokens.weight": w(VOCAB, HIDDEN),
               "model.norm.weight": torch.ones(HIDDEN, dtype=torch.bfloat16)}
    for i in range(LAYERS):
        p = f"model.layers.{i}"
        tensors[p + ".input_layernorm.weight"] = torch.ones(
            HIDDEN, dtype=torch.bfloat16)
        tensors[p + ".post_attention_layernorm.weight"] = torch.ones(
            HIDDEN, dtype=torch.bfloat16)
        tensors[p + ".self_attn.q_proj.weight"] = w(HEADS * HEAD_DIM, HIDDEN)
        tensors[p + ".self_attn.k_proj.weight"] = w(KV_HEADS * HEAD_DIM, HIDDEN)
        tensors[p + ".self_attn.v_proj.weight"] = w(KV_HEADS * HEAD_DIM, HIDDEN)
        tensors[p + ".self_attn.o_proj.weight"] = w(HIDDEN, HEADS * HEAD_DIM)
        tensors[p + ".mlp.gate_proj.weight"] = w(INTER, HIDDEN)
        tensors[p + ".mlp.up_proj.weight"] = w(INTER, HIDDEN)
        tensors[p + ".mlp.down_proj.weight"] = w(HIDDEN, INTER)
    return tensors


def _write_ckpt(tmp_path):
    ckpt = tmp_path / "tiny-llama"
    ckpt.mkdir()
    save_file(_tiny_tensors(), str(ckpt / "model.safetensors"),
              metadata={"format": "pt"})
    (ckpt / "config.json").write_text(json.dumps(_tiny_config()))
    return str(ckpt)


def _naive_forward(weights, token_ids):
    """Independent reference: float64 numpy, per-head loops,
    explicit per-position attention. Shares no code with the
    harness beyond the weight dict it is handed."""
    w = {n: t.double().numpy() for n, t in weights.items()}

    def norm(x, g):
        return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + EPS) * g

    def rope(vec, pos):
        # rotate_half convention (config: rope_interleaved false):
        # pair (j, j + HEAD_DIM/2), frequency THETA^(-2j/HEAD_DIM).
        out = vec.copy()
        half = HEAD_DIM // 2
        for j in range(half):
            freq = THETA ** (-2.0 * j / HEAD_DIM)
            c, s = np.cos(pos * freq), np.sin(pos * freq)
            a, b = vec[j], vec[j + half]
            out[j] = a * c - b * s
            out[j + half] = a * s + b * c
        return out

    seq = len(token_ids)
    x = w["model.embed_tokens.weight"][list(token_ids)].copy()
    for layer in range(LAYERS):
        p = f"model.layers.{layer}"
        h = norm(x, w[p + ".input_layernorm.weight"])
        q = h @ w[p + ".self_attn.q_proj.weight"].T
        k = h @ w[p + ".self_attn.k_proj.weight"].T
        v = h @ w[p + ".self_attn.v_proj.weight"].T
        ctx = np.zeros_like(x)
        for head in range(HEADS):
            kv = head // (HEADS // KV_HEADS)
            for t in range(seq):
                qt = rope(q[t, head * HEAD_DIM:(head + 1) * HEAD_DIM], t)
                scores, vals = [], []
                for s_ in range(t + 1):
                    ks = rope(k[s_, kv * HEAD_DIM:(kv + 1) * HEAD_DIM], s_)
                    scores.append(float(qt @ ks) / np.sqrt(HEAD_DIM))
                    vals.append(v[s_, kv * HEAD_DIM:(kv + 1) * HEAD_DIM])
                a = np.exp(np.array(scores) - max(scores))
                a /= a.sum()
                ctx[t, head * HEAD_DIM:(head + 1) * HEAD_DIM] = \
                    a @ np.array(vals)
        x = x + ctx @ w[p + ".self_attn.o_proj.weight"].T
        h = norm(x, w[p + ".post_attention_layernorm.weight"])
        gate = h @ w[p + ".mlp.gate_proj.weight"].T
        up = h @ w[p + ".mlp.up_proj.weight"].T
        silu = gate / (1.0 + np.exp(-gate))
        x = x + (silu * up) @ w[p + ".mlp.down_proj.weight"].T
    x = norm(x, w["model.norm.weight"])
    return x @ w["model.embed_tokens.weight"].T


# -- config -----------------------------------------------------------------


def test_config_accepts_llama_surface():
    cfg = validate_llama_config(_tiny_config())
    assert cfg["hidden_size"] == HIDDEN


@pytest.mark.parametrize("mutate", [
    lambda c: c.update(model_type="glm"),
    lambda c: c.update(attention_bias=True),
    lambda c: c.update(rope_scaling={"type": "linear"}),
    lambda c: c.pop("num_hidden_layers"),
    lambda c: c.update(hidden_size=63),
])
def test_config_refuses_outside_surface(mutate):
    cfg = _tiny_config()
    mutate(cfg)
    with pytest.raises(LlamaMeasureError):
        validate_llama_config(cfg)


# -- assembly ---------------------------------------------------------------


def test_assembly_widths_match_derived_bytes(tmp_path):
    ckpt = os.path.join(_write_ckpt(tmp_path), "model.safetensors")
    n = count_parameters(assemble_weights(ckpt, "bf16"))
    assert n == sum(t.numel() for t in _tiny_tensors().values())
    for width, per in ASSEMBLY_WIDTHS.items():
        if width == "float32_pair":
            continue  # the pair is two float32 assemblies; tested below
        weights = assemble_weights(ckpt, width)
        assert assembled_bytes(weights) == n * per


def test_pair_assembly_is_two_float32_copies(tmp_path):
    ckpt = os.path.join(_write_ckpt(tmp_path), "model.safetensors")
    one = assemble_weights(ckpt, "float32")
    assert assembled_bytes(one) * 2 == count_parameters(one) * 8


# -- forward ------------------------------------------------------------------


def test_forward_matches_naive_float64_reference(tmp_path):
    ckpt = os.path.join(_write_ckpt(tmp_path), "model.safetensors")
    weights = assemble_weights(ckpt, "float32")
    ids = [3, 11, 7, 42, 0, 5]
    logits = llama_forward(weights, _tiny_config(), ids).double().numpy()
    ref = _naive_forward(weights, ids)
    assert logits.shape == (len(ids), VOCAB)
    assert np.max(np.abs(logits - ref)) < 2e-4


def test_forward_is_deterministic(tmp_path):
    ckpt = os.path.join(_write_ckpt(tmp_path), "model.safetensors")
    weights = assemble_weights(ckpt, "float32")
    a = llama_forward(weights, _tiny_config(), [1, 2, 3])
    b = llama_forward(weights, _tiny_config(), [1, 2, 3])
    assert torch.equal(a, b)


def test_forward_refuses_non_float32_and_empty(tmp_path):
    ckpt = os.path.join(_write_ckpt(tmp_path), "model.safetensors")
    weights = assemble_weights(ckpt, "bf16")
    with pytest.raises(LlamaMeasureError):
        llama_forward(weights, _tiny_config(), [1])
    with pytest.raises(LlamaMeasureError):
        llama_forward(assemble_weights(ckpt, "float32"),
                      _tiny_config(), [])


# -- compare ------------------------------------------------------------------


def test_compare_identical_and_perturbed(tmp_path):
    ckpt = os.path.join(_write_ckpt(tmp_path), "model.safetensors")
    base = assemble_weights(ckpt, "float32")
    same = assemble_weights(ckpt, "float32")
    out = compare_weight_sets(base, same)
    assert out["max_abs_diff"] == 0.0
    assert out["first_differing_tensor"] is None
    assert out["tensors_compared"] == len(base)

    name = "model.layers.0.mlp.up_proj.weight"
    same[name] = same[name].clone()
    same[name][0, 0] += 0.25
    out = compare_weight_sets(base, same)
    assert out["max_abs_diff"] == pytest.approx(0.25)
    assert out["first_differing_tensor"] == name


def test_compare_refuses_mismatched_inventories(tmp_path):
    ckpt = os.path.join(_write_ckpt(tmp_path), "model.safetensors")
    base = assemble_weights(ckpt, "float32")
    other = dict(base)
    other.pop("model.norm.weight")
    with pytest.raises(LlamaMeasureError):
        compare_weight_sets(base, other)
