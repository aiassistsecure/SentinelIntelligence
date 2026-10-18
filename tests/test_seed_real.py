"""SI-7: real-checkpoint seed — per-tensor dedup store, sharded layout,
config-driven arch proof.

The llama-style checkpoints here are LOCALLY CONSTRUCTED in the real
safetensors format/layout (``model.layers.N.*`` tensor names, a real
``config.json``, a real ``model.safetensors.index.json`` + shards) from
seeded random arrays. They are real-layout checkpoints, honestly NOT
trained models — they exist so the seed/export machinery can be proven
hermetically (no network). A separate opt-in test at the bottom runs
the same path against a real downloaded checkpoint when the
``SI_REAL_CKPT`` env var points at one, and skips otherwise.
"""
import json
import os

import gguf
import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save

from si.bundle import (
    TamperError,
    _anchor_checkpoint,
    _write_manifest,
    init_bundle,
    load_bundle,
    sha256_hex,
)
from si.export import export_bundle, materialize_head
from si.journal import open_journal
from si.seed import seed_bundle

HIDDEN = 16
INTERMEDIATE = 32
VOCAB = 32
LAYERS = 2
HEADS = 4
KV_HEADS = 2
MAX_POS = 64
RMS_EPS = 1e-5


def _llama_tensors(with_lm_head=True, seed=1234):
    rng = np.random.default_rng(seed)
    tensors = {
        "model.embed_tokens.weight": (VOCAB, HIDDEN),
        "model.norm.weight": (HIDDEN,),
    }
    if with_lm_head:
        tensors["lm_head.weight"] = (VOCAB, HIDDEN)
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


def _llama_config(tie=False):
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
        "tie_word_embeddings": tie,
    }


def _write_single_ckpt(tmp_path, name, tensors, config=None):
    ckpt = tmp_path / name
    ckpt.mkdir(exist_ok=True)
    (ckpt / "model.safetensors").write_bytes(save(tensors))
    if config is not None:
        (ckpt / "config.json").write_text(json.dumps(config))
    return str(ckpt)


def _write_sharded_ckpt(tmp_path, name, tensors, config=None,
                        weight_map_override=None, shard_tensors=None):
    """Write a sharded checkpoint: 2 shards + index.json (+ config).

    Default split: names sorted, first half -> shard 1, rest -> shard 2.
    ``shard_tensors`` / ``weight_map_override`` let failure tests build
    deliberately inconsistent layouts.
    """
    ckpt = tmp_path / name
    ckpt.mkdir(exist_ok=True)
    names = sorted(tensors)
    if shard_tensors is None:
        half = len(names) // 2
        shard_tensors = {
            "model-00001-of-00002.safetensors": {n: tensors[n] for n in names[:half]},
            "model-00002-of-00002.safetensors": {n: tensors[n] for n in names[half:]},
        }
    weight_map = {}
    for shard, shard_ts in shard_tensors.items():
        (ckpt / shard).write_bytes(save(shard_ts))
        for n in shard_ts:
            weight_map[n] = shard
    if weight_map_override is not None:
        weight_map = weight_map_override
    index = {"metadata": {"total_size": 0}, "weight_map": weight_map}
    (ckpt / "model.safetensors.index.json").write_text(json.dumps(index))
    if config is not None:
        (ckpt / "config.json").write_text(json.dumps(config))
    return str(ckpt)


def _init(tmp_path, name):
    return init_bundle(str(tmp_path / (name + ".si")), name=name,
                       journal_backend="embedded")


def _store_stats(bundle_dir):
    blob_dir = os.path.join(bundle_dir, "blobs", "sha256")
    files = sorted(os.listdir(blob_dir))
    return len(files), sum(os.path.getsize(os.path.join(blob_dir, f))
                           for f in files)


def _gguf_field(path, key):
    reader = gguf.GGUFReader(path)
    assert key in reader.fields, f"{key} missing from GGUF metadata"
    return reader.fields[key].contents()


# -- per-tensor format ------------------------------------------------------------
def test_per_tensor_blobs_and_config_arch_end_to_end(tmp_path):
    """Seed a real-layout llama checkpoint (locally constructed, not a
    trained model): every tensor becomes a valid single-tensor blob,
    the data round-trips exactly, and seed -> export writes GGUF arch
    KVs matching the checkpoint's config.json."""
    tensors = _llama_tensors()
    cfg = _llama_config()
    src = _write_single_ckpt(tmp_path, "ckpt-llama", tensors, config=cfg)
    bundle_dir = _init(tmp_path, "si7-llama")
    res = seed_bundle(bundle_dir, src, importer="test")

    manifest = load_bundle(bundle_dir)
    base = manifest["base"]
    assert res["tensors"] == len(tensors)
    assert base["seed_format"] == "per-tensor-v1"
    assert base["tensor_count"] == len(tensors)
    assert set(base["tensor_blobs"]) == set(tensors)
    # Config captured with its sha, arch source labeled.
    assert base["checkpoint_config"] == cfg
    with open(os.path.join(src, "config.json"), "rb") as f:
        assert base["checkpoint_config_sha256"] == "sha256:" + sha256_hex(f.read())
    assert base["arch_source"] == "checkpoint-config"

    # Each per-tensor blob is a valid single-tensor safetensors file
    # under the canonical key, holding exactly the source tensor's data.
    for name, ref in base["tensor_blobs"].items():
        assert ref["shape"] == list(tensors[name].shape)
        assert ref["dtype"] == "float32"
        blob = os.path.join(bundle_dir, "blobs", "sha256",
                            ref["blob"].split(":", 1)[1] + ".safetensors")
        assert ref["bytes"] == os.path.getsize(blob)
        with safe_open(blob, framework="np") as f:
            assert list(f.keys()) == ["tensor"]
            assert np.array_equal(f.get_tensor("tensor"), tensors[name])

    # Materialization (the export read path) reproduces every tensor.
    materialized = materialize_head(bundle_dir, manifest)
    assert set(materialized) == set(tensors)
    for name in tensors:
        assert np.array_equal(materialized[name], tensors[name])

    out = str(tmp_path / "llama.gguf")
    export_bundle(bundle_dir, out)
    assert _gguf_field(out, "general.architecture") == "llama"
    assert _gguf_field(out, "llama.block_count") == LAYERS
    assert _gguf_field(out, "llama.embedding_length") == HIDDEN
    assert _gguf_field(out, "llama.attention.head_count") == HEADS
    assert _gguf_field(out, "llama.attention.head_count_kv") == KV_HEADS
    assert _gguf_field(out, "llama.vocab_size") == VOCAB
    assert _gguf_field(out, "llama.context_length") == MAX_POS
    assert _gguf_field(out, "si.arch_source") == "checkpoint-config"


def test_per_tensor_blob_tamper_fails_closed(tmp_path):
    src = _write_single_ckpt(tmp_path, "ckpt-tamper", _llama_tensors(),
                             config=_llama_config())
    bundle_dir = _init(tmp_path, "si7-tamper")
    seed_bundle(bundle_dir, src, importer="test")
    manifest = load_bundle(bundle_dir)
    ref = manifest["base"]["tensor_blobs"]["model.norm.weight"]["blob"]
    blob = os.path.join(bundle_dir, "blobs", "sha256",
                        ref.split(":", 1)[1] + ".safetensors")
    with open(blob, "r+b") as f:
        f.seek(-1, os.SEEK_END)
        last = f.read(1)
        f.seek(-1, os.SEEK_END)
        f.write(bytes([last[0] ^ 0xFF]))
    with pytest.raises(TamperError):
        load_bundle(bundle_dir)


# -- never duplicate bytes ----------------------------------------------------------
def test_identical_tensors_share_one_blob(tmp_path):
    """Two tensors with identical bytes under different names must be
    stored as ONE blob file (content addressing over tensor data)."""
    rng = np.random.default_rng(5)
    shared = rng.standard_normal((4, 4)).astype(np.float32)
    tensors = {"a.weight": shared, "b.weight": shared.copy(),
               "c.weight": rng.standard_normal((2,)).astype(np.float32)}
    src = _write_single_ckpt(tmp_path, "ckpt-dup", tensors)
    bundle_dir = _init(tmp_path, "si7-dup")
    before = _store_stats(bundle_dir)
    res = seed_bundle(bundle_dir, src, importer="test")

    manifest = load_bundle(bundle_dir)
    tb = manifest["base"]["tensor_blobs"]
    assert tb["a.weight"]["blob"] == tb["b.weight"]["blob"]
    assert tb["a.weight"]["blob"] != tb["c.weight"]["blob"]
    after = _store_stats(bundle_dir)
    # Exactly two NEW blob files for three tensors (a/b share one).
    assert after[0] - before[0] == 2
    assert res["deduplicated_bytes"] == tb["a.weight"]["bytes"]
    # Both names still materialize to the shared data.
    materialized = materialize_head(bundle_dir, manifest)
    assert np.array_equal(materialized["a.weight"], shared)
    assert np.array_equal(materialized["b.weight"], shared)


def test_reseed_same_checkpoint_is_idempotent_noop(tmp_path):
    """Re-seeding the exact checkpoint already serving as the base
    writes zero bytes: blob-store file count AND total bytes are
    unchanged, the manifest file is untouched, and the run reports
    already_seeded instead of journaling a duplicate import."""
    src = _write_single_ckpt(tmp_path, "ckpt-reseed", _llama_tensors(),
                             config=_llama_config())
    bundle_dir = _init(tmp_path, "si7-reseed")
    first = seed_bundle(bundle_dir, src, importer="test")
    assert first["already_seeded"] is False
    stats_before = _store_stats(bundle_dir)
    with open(os.path.join(bundle_dir, "manifest.json"), "rb") as f:
        manifest_bytes_before = f.read()
    tip_before = load_bundle(bundle_dir)["journal"]["last_seq"]

    second = seed_bundle(bundle_dir, src, importer="test")
    assert second["already_seeded"] is True
    assert second["tensors"] == first["tensors"]
    assert second["deduplicated_bytes"] == first["base"]["imported_bytes"]
    assert _store_stats(bundle_dir) == stats_before
    with open(os.path.join(bundle_dir, "manifest.json"), "rb") as f:
        assert f.read() == manifest_bytes_before
    manifest = load_bundle(bundle_dir)  # still verifies cleanly
    assert manifest["journal"]["last_seq"] == tip_before


def test_seed_then_export_adds_no_base_bytes(tmp_path):
    """Export reads the base; it must not copy base bytes into the
    blob store (blob-store stats identical before and after)."""
    src = _write_single_ckpt(tmp_path, "ckpt-exp", _llama_tensors(),
                             config=_llama_config())
    bundle_dir = _init(tmp_path, "si7-exp")
    seed_bundle(bundle_dir, src, importer="test")
    before = _store_stats(bundle_dir)
    export_bundle(bundle_dir, str(tmp_path / "out.gguf"))
    assert _store_stats(bundle_dir) == before


# -- sharded HF layout ----------------------------------------------------------------
def test_sharded_checkpoint_seed_and_export(tmp_path):
    """A locally constructed real-layout sharded checkpoint (index.json
    + 2 shards) seeds per-tensor and exports with config arch KVs."""
    tensors = _llama_tensors()
    src = _write_sharded_ckpt(tmp_path, "ckpt-sharded", tensors,
                              config=_llama_config())
    bundle_dir = _init(tmp_path, "si7-sharded")
    res = seed_bundle(bundle_dir, src, importer="test")

    manifest = load_bundle(bundle_dir)
    base = manifest["base"]
    assert res["parts"] == 2
    assert base["sharded"] is True
    assert set(base["weight_map"]) == set(tensors)
    assert set(base["weight_map"].values()) == {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors"}
    assert set(base["tensor_blobs"]) == set(tensors)
    materialized = materialize_head(bundle_dir, manifest)
    for name in tensors:
        assert np.array_equal(materialized[name], tensors[name])

    out = str(tmp_path / "sharded.gguf")
    export_bundle(bundle_dir, out)
    assert _gguf_field(out, "general.architecture") == "llama"
    assert _gguf_field(out, "llama.block_count") == LAYERS
    assert _gguf_field(out, "llama.vocab_size") == VOCAB


def test_sharded_missing_shard_fails_closed(tmp_path):
    tensors = _llama_tensors()
    src = _write_sharded_ckpt(tmp_path, "ckpt-missing", tensors,
                              config=_llama_config())
    os.remove(os.path.join(src, "model-00002-of-00002.safetensors"))
    bundle_dir = _init(tmp_path, "si7-missing")
    with pytest.raises(ValueError, match="missing shard"):
        seed_bundle(bundle_dir, src, importer="test")
    # The failed seed changed nothing.
    assert load_bundle(bundle_dir)["base"]["source"].startswith("demo:")


def test_sharded_mapped_tensor_missing_fails_closed(tmp_path):
    tensors = _llama_tensors()
    names = sorted(tensors)
    half = len(names) // 2
    shard_tensors = {
        "model-00001-of-00002.safetensors": {n: tensors[n] for n in names[:half]},
        "model-00002-of-00002.safetensors": {n: tensors[n] for n in names[half:]},
    }
    # Map claims a layer-0 tensor lives in shard 2; it actually lives in 1.
    weight_map = {}
    for shard, shard_ts in shard_tensors.items():
        for n in shard_ts:
            weight_map[n] = shard
    victim = "model.layers.0.mlp.gate_proj.weight"
    assert weight_map[victim] == "model-00001-of-00002.safetensors"
    weight_map[victim] = "model-00002-of-00002.safetensors"
    src = _write_sharded_ckpt(tmp_path, "ckpt-badmap", tensors,
                              config=_llama_config(),
                              weight_map_override=weight_map,
                              shard_tensors=shard_tensors)
    bundle_dir = _init(tmp_path, "si7-badmap")
    with pytest.raises(ValueError, match="not in that shard|claimed by two shards"):
        seed_bundle(bundle_dir, src, importer="test")
    assert load_bundle(bundle_dir)["base"]["source"].startswith("demo:")


def test_sharded_tensor_in_two_shards_fails_closed(tmp_path):
    tensors = _llama_tensors()
    names = sorted(tensors)
    half = len(names) // 2
    shared_name = names[0]
    shard_tensors = {
        "model-00001-of-00002.safetensors": {n: tensors[n] for n in names[:half]},
        # The same tensor ALSO physically present in shard 2.
        "model-00002-of-00002.safetensors": {
            **{n: tensors[n] for n in names[half:]},
            shared_name: tensors[shared_name]},
    }
    src = _write_sharded_ckpt(tmp_path, "ckpt-twoshard", tensors,
                              config=_llama_config(),
                              shard_tensors=shard_tensors)
    bundle_dir = _init(tmp_path, "si7-twoshard")
    with pytest.raises(ValueError, match="claimed by two shards"):
        seed_bundle(bundle_dir, src, importer="test")
    assert load_bundle(bundle_dir)["base"]["source"].startswith("demo:")


def test_sharded_index_rejects_path_traversal_shard(tmp_path):
    ckpt = tmp_path / "ckpt-evil"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(
        save({"W": np.zeros((2, 2), dtype=np.float32)}))
    (ckpt / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"W": "../evil.safetensors"}}))
    bundle_dir = _init(tmp_path, "si7-evil")
    with pytest.raises(ValueError, match="invalid shard name"):
        seed_bundle(bundle_dir, str(ckpt), importer="test")


# -- tied embeddings / config gaps ------------------------------------------------------
def test_tied_embeddings_checkpoint_exports(tmp_path):
    """A llama checkpoint with tied embeddings has NO lm_head.weight;
    seed -> export must not crash and must still label arch from config."""
    tensors = _llama_tensors(with_lm_head=False)
    src = _write_single_ckpt(tmp_path, "ckpt-tied", tensors,
                             config=_llama_config(tie=True))
    bundle_dir = _init(tmp_path, "si7-tied")
    seed_bundle(bundle_dir, src, importer="test")
    manifest = load_bundle(bundle_dir)
    assert "lm_head.weight" not in manifest["base"]["tensor_blobs"]
    out = str(tmp_path / "tied.gguf")
    export_bundle(bundle_dir, out)
    assert _gguf_field(out, "general.architecture") == "llama"
    assert _gguf_field(out, "llama.vocab_size") == VOCAB
    assert _gguf_field(out, "llama.embedding_length") == HIDDEN


# -- provenance accounting ---------------------------------------------------------------
def test_provenance_records_full_accounting(tmp_path):
    tensors = _llama_tensors()
    cfg = _llama_config()
    src = _write_single_ckpt(tmp_path, "ckpt-prov", tensors, config=cfg)
    bundle_dir = _init(tmp_path, "si7-prov")
    res = seed_bundle(bundle_dir, src, importer="test")
    manifest = load_bundle(bundle_dir)
    journal = open_journal(bundle_dir, manifest)
    try:
        rows = journal.backend.find(
            "provenance", filter={"child": "base", "relation": "seeded_from"})
    finally:
        journal.close()
    assert rows, "expected a seeded_from provenance record"
    note = rows[0]["note"]
    assert rows[0]["parents"] == [os.path.abspath(src)]
    assert f"{len(tensors)} tensors" in note
    assert f"{res['base']['imported_bytes']} bytes" in note
    assert f"deduplicated {res['deduplicated_bytes']} bytes" in note
    assert f"source={os.path.abspath(src)}" in note
    with open(os.path.join(src, "config.json"), "rb") as f:
        assert f"config_sha256=sha256:{sha256_hex(f.read())}" in note


def test_duplicate_tensor_across_plain_files_fails_closed(tmp_path):
    ckpt = tmp_path / "ckpt-dupnames"
    ckpt.mkdir()
    arr = np.zeros((2, 2), dtype=np.float32)
    (ckpt / "a.safetensors").write_bytes(save({"W": arr}))
    (ckpt / "b.safetensors").write_bytes(save({"W": arr}))
    bundle_dir = _init(tmp_path, "si7-dupnames")
    with pytest.raises(ValueError, match="duplicate tensor name"):
        seed_bundle(bundle_dir, str(ckpt), importer="test")


# -- backward compatibility: pre-SI-7 whole-file bundles ----------------------------------
def _make_legacy_bundle(tmp_path, name, tensors, config):
    """Build a bundle in the PRE-SI-7 layout exactly as the old seed
    wrote it: whole source files as blobs, ``base.parts[*].blob`` refs,
    no ``tensor_blobs``. Constructed with the same journal anchoring
    the old seed used, so the bundle verifies like a real legacy one."""
    bundle_dir = _init(tmp_path, name)
    src = _write_single_ckpt(tmp_path, name + "-src", tensors, config=config)
    src_file = os.path.join(src, "model.safetensors")
    with open(src_file, "rb") as f:
        data = f.read()
    hexpart = sha256_hex(data)
    dest = os.path.join(bundle_dir, "blobs", "sha256",
                        hexpart + ".safetensors")
    if not os.path.isfile(dest):
        with open(dest, "wb") as f:
            f.write(data)
    meta = {n: {"shape": list(t.shape), "dtype": str(t.dtype)}
            for n, t in tensors.items()}
    manifest = load_bundle(bundle_dir)
    manifest["base"] = {
        "source": f"seed:{src} (imported legacy-test)",
        "format": "safetensors",
        "blob": "sha256:" + hexpart,
        "parts": [{"file": "model.safetensors", "blob": "sha256:" + hexpart,
                   "bytes": len(data), "tensors": meta}],
        "tensors": meta,
        "bytes": len(data),
        "importer": "legacy-test",
        "imported_at": "2026-10-17T00:00:00Z",
        "checkpoint_config": config,
        "checkpoint_config_sha256": "sha256:" + sha256_hex(
            json.dumps(config).encode("utf-8")),
        "arch_source": "checkpoint-config",
    }
    journal = open_journal(bundle_dir, manifest)
    try:
        journal.record_provenance(
            child="base", parents=[src], relation="seeded_from",
            note=f"imported 1 tensor file(s), {len(data)} bytes via si seed")
        manifest = _anchor_checkpoint(journal, manifest, head="base",
                                      note="seed legacy")
        _write_manifest(bundle_dir, manifest)
    finally:
        journal.close()
    return bundle_dir


def test_legacy_whole_file_bundle_still_loads_and_exports(tmp_path):
    """Backward compat: a pre-SI-7 bundle (whole-file parts, no
    per-tensor refs) still verifies and exports with config arch KVs,
    and its tensors materialize exactly."""
    tensors = _llama_tensors()
    bundle_dir = _make_legacy_bundle(tmp_path, "si7-legacy", tensors,
                                     _llama_config())
    manifest = load_bundle(bundle_dir)  # verifies: parts blob included
    assert "tensor_blobs" not in manifest["base"]
    materialized = materialize_head(bundle_dir, manifest)
    assert set(materialized) == set(tensors)
    for name in tensors:
        assert np.array_equal(materialized[name], tensors[name])
    out = str(tmp_path / "legacy.gguf")
    export_bundle(bundle_dir, out)
    assert _gguf_field(out, "general.architecture") == "llama"
    assert _gguf_field(out, "llama.block_count") == LAYERS
    assert _gguf_field(out, "si.arch_source") == "checkpoint-config"


# -- opt-in: a real downloaded checkpoint ---------------------------------------------------
def test_real_downloaded_checkpoint_env(tmp_path):
    """End-to-end against a REAL checkpoint downloaded from HuggingFace.

    Runs only when SI_REAL_CKPT points at the checkpoint directory
    (e.g. a tiny ungated llama repo); skips otherwise so the committed
    suite stays hermetic.
    """
    ckpt = os.environ.get("SI_REAL_CKPT")
    if not ckpt:
        pytest.skip("SI_REAL_CKPT not set; hermetic suite only")
    bundle_dir = _init(tmp_path, "si7-real")
    res = seed_bundle(bundle_dir, ckpt, importer="test")
    assert res["already_seeded"] is False
    assert res["tensors"] > 0
    manifest = load_bundle(bundle_dir)
    assert set(manifest["base"]["tensor_blobs"]) == set(
        manifest["base"]["tensors"])
    out = str(tmp_path / "real.gguf")
    export_bundle(bundle_dir, out)
    cfg_path = os.path.join(ckpt, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        assert manifest["base"]["checkpoint_config"]["model_type"] == \
            cfg["model_type"]
        assert _gguf_field(out, "general.architecture") == cfg["model_type"]
