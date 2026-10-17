"""SI-4: arch-aware ``si export`` for real transformer checkpoints.

Covers: config.json captured at seed time -> real ``<arch>.*`` GGUF KV
entries matching the fixture config (read back with the gguf library);
the tensor-shape-inference fallback (labeled, conservative, omits what
it cannot know); fail-closed ValueError on ambiguous shapes; export
determinism for a seeded checkpoint base; and the demo base keeping
``general.architecture="unknown"`` with ``si.arch_source="demo-unknown"``.

The fixture checkpoint is synthesized on the fly (tiny llama-shaped
safetensors + config.json) — nothing is downloaded.
"""
import json
import os

import gguf
import numpy as np
import pytest
from safetensors.numpy import save

from si.bundle import init_bundle, load_bundle, sha256_hex
from si.export import export_bundle
from si.seed import seed_bundle

#: Tiny llama-shaped fixture dimensions.
HIDDEN = 16
INTERMEDIATE = 32
VOCAB = 32
LAYERS = 2
HEADS = 4
KV_HEADS = 2
MAX_POS = 64
RMS_EPS = 1e-5


def _llama_tensors():
    rng = np.random.default_rng(1234)
    tensors = {
        "model.embed_tokens.weight": (VOCAB, HIDDEN),
        "model.norm.weight": (HIDDEN,),
        "lm_head.weight": (VOCAB, HIDDEN),
    }
    for i in range(LAYERS):
        p = f"model.layers.{i}"
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            tensors[f"{p}.self_attn.{proj}.weight"] = (HIDDEN, HIDDEN)
        tensors[f"{p}.mlp.gate_proj.weight"] = (INTERMEDIATE, HIDDEN)
        tensors[f"{p}.mlp.up_proj.weight"] = (INTERMEDIATE, HIDDEN)
        tensors[f"{p}.mlp.down_proj.weight"] = (HIDDEN, INTERMEDIATE)
        tensors[f"{p}.input_layernorm.weight"] = (HIDDEN,)
        tensors[f"{p}.post_attention_layernorm.weight"] = (HIDDEN,)
    return {n: rng.standard_normal(s).astype(np.float32)
            for n, s in tensors.items()}


def _llama_config():
    return {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "vocab_size": VOCAB,
        "max_position_embeddings": MAX_POS,
        "rms_norm_eps": RMS_EPS,
    }


def _make_ckpt(tmp_path, name, with_config=True, tensors=None, config=None):
    """Write a synthetic checkpoint dir: model.safetensors + config.json."""
    ckpt = tmp_path / name
    ckpt.mkdir(exist_ok=True)
    (ckpt / "model.safetensors").write_bytes(
        save(tensors if tensors is not None else _llama_tensors()))
    if with_config:
        (ckpt / "config.json").write_text(
            json.dumps(config if config is not None else _llama_config()))
    return str(ckpt)


def _init(tmp_path, name):
    return init_bundle(str(tmp_path / (name + ".si")), name=name,
                       journal_backend="embedded")


def _gguf_field(path, key):
    reader = gguf.GGUFReader(path)
    assert key in reader.fields, f"{key} missing from GGUF metadata"
    return reader.fields[key].contents()


# -- config path ------------------------------------------------------------------
def test_export_arch_from_config(tmp_path):
    bundle_dir = _init(tmp_path, "arch-cfg")
    seed_bundle(bundle_dir, _make_ckpt(tmp_path, "ckpt-cfg"), importer="test")

    manifest = load_bundle(bundle_dir)
    base = manifest["base"]
    assert base["arch_source"] == "checkpoint-config", \
        "seed must store the arch_source marker"
    assert base["checkpoint_config"]["model_type"] == "llama"
    assert base["checkpoint_config_sha256"].startswith("sha256:")

    out = str(tmp_path / "llama.gguf")
    res = export_bundle(bundle_dir, out)
    assert res["head"] == "base"

    assert _gguf_field(out, "general.architecture") == "llama"
    assert _gguf_field(out, "llama.block_count") == LAYERS
    assert _gguf_field(out, "llama.embedding_length") == HIDDEN
    assert _gguf_field(out, "llama.attention.head_count") == HEADS
    assert _gguf_field(out, "llama.attention.head_count_kv") == KV_HEADS
    assert _gguf_field(out, "llama.vocab_size") == VOCAB
    assert _gguf_field(out, "llama.context_length") == MAX_POS
    assert _gguf_field(out, "llama.rope.dimension_count") == HIDDEN // HEADS
    eps = _gguf_field(out, "llama.attention.layer_norm_rms_epsilon")
    assert abs(eps - RMS_EPS) < 1e-9, f"rms eps {eps} != {RMS_EPS}"
    assert _gguf_field(out, "si.arch_source") == "checkpoint-config"


def test_export_arch_config_omits_unknown(tmp_path):
    """A config without kv-head info must omit the field, not invent it."""
    cfg = _llama_config()
    del cfg["num_key_value_heads"]
    bundle_dir = _init(tmp_path, "arch-omit")
    seed_bundle(bundle_dir,
                _make_ckpt(tmp_path, "ckpt-omit", config=cfg),
                importer="test")
    out = str(tmp_path / "omit.gguf")
    export_bundle(bundle_dir, out)
    reader = gguf.GGUFReader(out)
    assert "llama.attention.head_count_kv" not in reader.fields, \
        "unknown head_count_kv must be omitted, never invented"
    assert _gguf_field(out, "llama.attention.head_count") == HEADS


# -- inference path ---------------------------------------------------------------
def test_export_arch_tensor_inference(tmp_path):
    bundle_dir = _init(tmp_path, "arch-inf")
    seed_bundle(bundle_dir,
                _make_ckpt(tmp_path, "ckpt-inf", with_config=False),
                importer="test")

    manifest = load_bundle(bundle_dir)
    assert "checkpoint_config" not in manifest["base"], \
        "no config.json -> nothing stored"

    out = str(tmp_path / "inf.gguf")
    export_bundle(bundle_dir, out)

    assert _gguf_field(out, "general.architecture") == "llama"
    assert _gguf_field(out, "si.arch_source") == "tensor-shape-inference"
    assert _gguf_field(out, "llama.block_count") == LAYERS
    assert _gguf_field(out, "llama.embedding_length") == HIDDEN
    assert _gguf_field(out, "llama.vocab_size") == VOCAB
    reader = gguf.GGUFReader(out)
    for missing in ("llama.attention.head_count",
                    "llama.attention.head_count_kv",
                    "llama.context_length",
                    "llama.attention.layer_norm_rms_epsilon",
                    "llama.rope.dimension_count"):
        assert missing not in reader.fields, \
            f"{missing} is not inferable from shapes; must be omitted"


def test_export_arch_inference_ambiguous_names(tmp_path):
    """Unrecognized tensor names -> ValueError, never a guessed arch."""
    rng = np.random.default_rng(7)
    tensors = {"mystery.weight": rng.standard_normal((4, 4)).astype(np.float32),
               "other": rng.standard_normal((4,)).astype(np.float32)}
    bundle_dir = _init(tmp_path, "arch-amb")
    seed_bundle(bundle_dir,
                _make_ckpt(tmp_path, "ckpt-amb", with_config=False,
                           tensors=tensors),
                importer="test")
    with pytest.raises(ValueError, match="cannot infer architecture"):
        export_bundle(bundle_dir, str(tmp_path / "amb.gguf"))


def test_export_arch_inference_ambiguous_shapes(tmp_path):
    """Llama layers without embed_tokens, or a disagreeing lm_head -> fail."""
    # Case 1: layers present but no embed_tokens to size the model.
    tensors = {n: t for n, t in _llama_tensors().items()
               if n != "model.embed_tokens.weight"}
    bundle_dir = _init(tmp_path, "arch-amb2")
    seed_bundle(bundle_dir,
                _make_ckpt(tmp_path, "ckpt-amb2", with_config=False,
                           tensors=tensors),
                importer="test")
    with pytest.raises(ValueError, match="cannot infer architecture"):
        export_bundle(bundle_dir, str(tmp_path / "amb2.gguf"))

    # Case 2: lm_head disagrees with embed_tokens on vocab size.
    tensors = dict(_llama_tensors())
    rng = np.random.default_rng(9)
    tensors["lm_head.weight"] = rng.standard_normal(
        (VOCAB // 2, HIDDEN)).astype(np.float32)
    bundle_dir = _init(tmp_path, "arch-amb3")
    seed_bundle(bundle_dir,
                _make_ckpt(tmp_path, "ckpt-amb3", with_config=False,
                           tensors=tensors),
                importer="test")
    with pytest.raises(ValueError, match="ambiguous architecture"):
        export_bundle(bundle_dir, str(tmp_path / "amb3.gguf"))


# -- determinism ------------------------------------------------------------------
def test_export_arch_determinism(tmp_path):
    bundle_dir = _init(tmp_path, "arch-det")
    seed_bundle(bundle_dir, _make_ckpt(tmp_path, "ckpt-det"), importer="test")

    a = str(tmp_path / "a.gguf")
    b = str(tmp_path / "b.gguf")
    export_bundle(bundle_dir, a)
    export_bundle(bundle_dir, b)
    with open(a, "rb") as f:
        ha = sha256_hex(f.read())
    with open(b, "rb") as f:
        hb = sha256_hex(f.read())
    assert ha == hb, \
        "two exports of the same seeded head must be byte-identical"


# -- demo base keeps honest unknown -------------------------------------------------
def test_export_demo_arch_unknown(tmp_path):
    bundle_dir = _init(tmp_path, "arch-demo")
    out = str(tmp_path / "demo.gguf")
    export_bundle(bundle_dir, out)
    assert _gguf_field(out, "general.architecture") == "unknown"
    assert _gguf_field(out, "si.arch_source") == "demo-unknown"
    manifest = load_bundle(bundle_dir)
    assert manifest["base"]["arch_source"] == "demo-unknown"
