"""SI-8: by-reference seed — zero stored tensor bytes, streaming
hashes, fail-closed reference verification.

The checkpoints here are LOCALLY CONSTRUCTED in the real safetensors
format/layout from seeded random arrays (plus one torch-built bf16
fixture). They are real-format checkpoints, honestly NOT trained
models — they exist so the by-reference machinery can be proven
hermetically (no network, no 59 GB download).

What is proven:
- a by-reference seed stores ZERO new blob files and records complete
  per-tensor refs (absolute source path, absolute byte offset, byte
  length, dtype name + code, shape, canonical content hash);
- the recorded content hash is EXACTLY copy mode's canonical blob hash
  for the same checkpoint (same identity, different storage);
- load_bundle verifies referenced sources (existence, size, full
  streaming file hash, per-tensor streaming canonical hash): a flipped
  tensor byte, a truncated file, and a missing file all fail closed;
- the read path (load_base_tensor / export materialization) verifies
  the tensor it reads, works for sharded + single-file + bf16 bases,
  and a ref-seeded demo-shaped base still drives the self-write loop;
- re-seeding is an idempotent no-op in-mode AND across modes (same
  checkpoint, either storage): nothing written, no mode conversion;
- export keeps config-driven GGUF arch metadata; serve still returns
  an honest 501 for a ref-seeded transformer bundle;
- copy mode's new disk preflight refuses BEFORE writing when the new
  blob bytes cannot fit (free space mocked via si.seed._disk_free).
"""
import json
import os
import threading
import urllib.error
import urllib.request

os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

import gguf
import numpy as np
import pytest
from safetensors.numpy import save

from si.bundle import TamperError, init_bundle, load_base_tensor, load_bundle
from si.cli import main as cli_main
from si.export import export_bundle, materialize_head
from si.journal import open_journal
from si.seed import _read_header, seed_bundle


def _init(tmp_path, name):
    return init_bundle(str(tmp_path / (name + ".si")), name=name,
                       journal_backend="embedded")


def _store_stats(bundle_dir):
    blob_dir = os.path.join(bundle_dir, "blobs", "sha256")
    files = sorted(os.listdir(blob_dir))
    return len(files), sum(os.path.getsize(os.path.join(blob_dir, f))
                           for f in files)


def _write_ckpt(tmp_path, name, tensors, config=None):
    ckpt = tmp_path / name
    ckpt.mkdir(exist_ok=True)
    (ckpt / "model.safetensors").write_bytes(save(tensors))
    if config is not None:
        (ckpt / "config.json").write_text(json.dumps(config))
    return str(ckpt)


def _write_sharded(tmp_path, name, tensors, config=None):
    ckpt = tmp_path / name
    ckpt.mkdir(exist_ok=True)
    names = sorted(tensors)
    half = len(names) // 2
    shards = {
        "model-00001-of-00002.safetensors": {n: tensors[n] for n in names[:half]},
        "model-00002-of-00002.safetensors": {n: tensors[n] for n in names[half:]},
    }
    weight_map = {}
    for shard, shard_ts in shards.items():
        (ckpt / shard).write_bytes(save(shard_ts))
        for n in shard_ts:
            weight_map[n] = shard
    (ckpt / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": 0}, "weight_map": weight_map}))
    if config is not None:
        (ckpt / "config.json").write_text(json.dumps(config))
    return str(ckpt)


def _simple_tensors(seed=42):
    rng = np.random.default_rng(seed)
    return {"W": rng.standard_normal((8, 8)).astype(np.float32),
            "b": rng.standard_normal((8,)).astype(np.float32)}


HIDDEN, INTER, VOCAB, HEADS, KV_HEADS, MAX_POS = 16, 32, 32, 4, 2, 64


def _llama_tensors(seed=7):
    rng = np.random.default_rng(seed)
    tensors = {
        "model.embed_tokens.weight": (VOCAB, HIDDEN),
        "model.norm.weight": (HIDDEN,),
        "lm_head.weight": (VOCAB, HIDDEN),
        "model.layers.0.self_attn.q_proj.weight": (HIDDEN, HIDDEN),
        "model.layers.0.self_attn.k_proj.weight": (HIDDEN, HIDDEN),
        "model.layers.0.self_attn.v_proj.weight": (HIDDEN, HIDDEN),
        "model.layers.0.self_attn.o_proj.weight": (HIDDEN, HIDDEN),
        "model.layers.0.mlp.gate_proj.weight": (INTER, HIDDEN),
        "model.layers.0.mlp.up_proj.weight": (INTER, HIDDEN),
        "model.layers.0.mlp.down_proj.weight": (HIDDEN, INTER),
        "model.layers.0.input_layernorm.weight": (HIDDEN,),
        "model.layers.0.post_attention_layernorm.weight": (HIDDEN,),
    }
    return {n: rng.standard_normal(s).astype(np.float32)
            for n, s in tensors.items()}


def _llama_config():
    return {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "num_hidden_layers": 1,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "vocab_size": VOCAB,
        "max_position_embeddings": MAX_POS,
        "rms_norm_eps": 1e-5,
    }


def _gguf_field(path, key):
    reader = gguf.GGUFReader(path)
    assert key in reader.fields, f"{key} missing from GGUF metadata"
    return reader.fields[key].contents()


# -- seed: zero stored bytes, complete refs, copy-mode identity -------------------

def test_ref_seed_stores_zero_tensor_bytes_and_records_refs(tmp_path):
    tensors = _simple_tensors()
    src = _write_ckpt(tmp_path, "ckpt-ref", tensors)
    bundle_dir = _init(tmp_path, "si8-ref")
    before = _store_stats(bundle_dir)

    res = seed_bundle(bundle_dir, src, importer="test", by_reference=True)

    # Zero new blob files, zero stored tensor bytes — the real numbers.
    assert _store_stats(bundle_dir) == before
    assert res["already_seeded"] is False
    assert res["bytes"] == 0 and res["stored_bytes"] == 0
    assert res["seed_format"] == "reference-v1"
    assert res["tensors"] == len(tensors)
    assert res["deduplicated_bytes"] == res["base"]["imported_bytes"] > 0

    manifest = load_bundle(bundle_dir)  # verifies against the source
    base = manifest["base"]
    assert base["seed_format"] == "reference-v1"
    assert base["storage"] == "by-reference"
    assert base["bytes"] == 0 and base["stored_bytes"] == 0
    assert "tensor_blobs" not in base  # nothing claims store residency
    assert set(base["tensor_refs"]) == set(tensors)
    assert base["tensor_count"] == len(tensors)
    assert base["referenced_bytes"] == sum(
        int(np.prod(t.shape)) * 4 for t in tensors.values())

    # Refs are complete and exact: absolute source path, ABSOLUTE byte
    # offset (header data_start + the header's relative offset), byte
    # length, dtype name + code, shape.
    src_file = os.path.join(src, "model.safetensors")
    header, data_start = _read_header(src_file)
    for name, ref in base["tensor_refs"].items():
        start, end = header[name]["data_offsets"]
        assert ref["source_file"] == os.path.abspath(src_file)
        assert ref["file_offset"] == data_start + start
        assert ref["byte_length"] == end - start
        assert ref["shape"] == list(tensors[name].shape)
        assert ref["dtype"] == "float32" and ref["dtype_code"] == "F32"
        assert ref["blob"].startswith("sha256:")
        assert ref["bytes"] > ref["byte_length"]  # canonical incl. header

    # Parts keep per-file provenance (size + streaming sha) for load
    # verification, and the compat blob is the "W" tensor's ref.
    assert base["parts"][0]["file"] == "model.safetensors"
    assert base["parts"][0]["bytes"] == os.path.getsize(src_file)
    assert base["blob"] == base["tensor_refs"]["W"]["blob"]

    # Reads round-trip exactly.
    for name in tensors:
        arr = load_base_tensor(bundle_dir, manifest, name)
        assert np.array_equal(arr, tensors[name])


def test_ref_seed_identity_matches_copy_mode(tmp_path):
    """Same checkpoint, same canonical blob refs in either mode — only
    the storage differs, and each manifest says which it is."""
    tensors = _llama_tensors()
    src = _write_ckpt(tmp_path, "ckpt-parity", tensors,
                      config=_llama_config())
    copy_dir = _init(tmp_path, "si8-copy")
    seed_bundle(copy_dir, src, importer="test")
    ref_dir = _init(tmp_path, "si8-refparity")
    seed_bundle(ref_dir, src, importer="test", by_reference=True)

    copy_base = load_bundle(copy_dir)["base"]
    ref_base = load_bundle(ref_dir)["base"]
    assert ref_base["seed_format"] == "reference-v1"
    assert copy_base["seed_format"] == "per-tensor-v1"
    for name in tensors:
        assert ref_base["tensor_refs"][name]["blob"] == \
            copy_base["tensor_blobs"][name]["blob"]
        assert ref_base["tensor_refs"][name]["bytes"] == \
            copy_base["tensor_blobs"][name]["bytes"]
    assert ref_base["blob"] == copy_base["blob"]
    assert ref_base["imported_bytes"] == copy_base["imported_bytes"]
    assert ref_base["bytes"] == 0 < copy_base["bytes"]


def test_ref_seed_sharded_checkpoint(tmp_path):
    tensors = _llama_tensors()
    src = _write_sharded(tmp_path, "ckpt-ref-sharded", tensors,
                         config=_llama_config())
    bundle_dir = _init(tmp_path, "si8-sharded")
    before = _store_stats(bundle_dir)
    res = seed_bundle(bundle_dir, src, importer="test", by_reference=True)

    assert _store_stats(bundle_dir) == before
    assert res["parts"] == 2
    manifest = load_bundle(bundle_dir)
    base = manifest["base"]
    assert base["sharded"] is True
    assert set(base["weight_map"]) == set(tensors)
    assert set(base["tensor_refs"]) == set(tensors)
    # Each ref points into the shard the weight_map claims.
    for name, ref in base["tensor_refs"].items():
        assert os.path.basename(ref["source_file"]) == \
            base["weight_map"][name]
    materialized = materialize_head(bundle_dir, manifest)
    for name in tensors:
        assert np.array_equal(materialized[name], tensors[name])


def test_ref_seed_provenance_accounting(tmp_path):
    tensors = _simple_tensors()
    src = _write_ckpt(tmp_path, "ckpt-ref-prov", tensors,
                      config={"model_type": "demo-x"})
    bundle_dir = _init(tmp_path, "si8-prov")
    res = seed_bundle(bundle_dir, src, importer="test", by_reference=True)
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
    assert "by-reference" in note
    assert f"{len(tensors)} tensors" in note
    assert f"{res['base']['imported_bytes']} bytes" in note
    assert "stored 0 tensor bytes" in note
    assert f"source={os.path.abspath(src)}" in note
    assert "config_sha256=sha256:" in note
    # Config captured exactly as in copy mode (export arch source).
    assert manifest["base"]["checkpoint_config"] == {"model_type": "demo-x"}
    assert manifest["base"]["arch_source"] == "checkpoint-config"


# -- verification fails closed ------------------------------------------------------

def test_ref_load_detects_tampered_tensor_byte(tmp_path):
    tensors = _simple_tensors()
    src = _write_ckpt(tmp_path, "ckpt-ref-tamper", tensors)
    bundle_dir = _init(tmp_path, "si8-tamper")
    seed_bundle(bundle_dir, src, importer="test", by_reference=True)
    manifest = load_bundle(bundle_dir)  # clean load first
    ref = manifest["base"]["tensor_refs"]["W"]

    # Flip one byte inside W's recorded range in the SOURCE file.
    with open(ref["source_file"], "r+b") as f:
        f.seek(ref["file_offset"])
        orig = f.read(1)
        f.seek(ref["file_offset"])
        f.write(bytes([orig[0] ^ 0xFF]))

    with pytest.raises(TamperError):
        load_bundle(bundle_dir)
    # The read path verifies the tensor it reads too — even handed a
    # manifest loaded before the tamper.
    with pytest.raises(TamperError):
        load_base_tensor(bundle_dir, manifest, "W")


def test_ref_load_detects_missing_source(tmp_path):
    src = _write_ckpt(tmp_path, "ckpt-ref-missing", _simple_tensors())
    bundle_dir = _init(tmp_path, "si8-missing")
    seed_bundle(bundle_dir, src, importer="test", by_reference=True)
    manifest = load_bundle(bundle_dir)
    os.remove(os.path.join(src, "model.safetensors"))
    with pytest.raises(TamperError, match="missing referenced source"):
        load_bundle(bundle_dir)
    with pytest.raises(TamperError):
        load_base_tensor(bundle_dir, manifest, "W")


def test_ref_load_detects_truncated_source(tmp_path):
    src_file = os.path.join(
        _write_ckpt(tmp_path, "ckpt-ref-trunc", _simple_tensors()),
        "model.safetensors")
    bundle_dir = _init(tmp_path, "si8-trunc")
    seed_bundle(bundle_dir, os.path.dirname(src_file), importer="test",
                by_reference=True)
    assert load_bundle(bundle_dir)
    with open(src_file, "r+b") as f:
        f.truncate(os.path.getsize(src_file) - 16)
    with pytest.raises(TamperError):
        load_bundle(bundle_dir)


def test_ref_seed_rejects_corrupt_source(tmp_path):
    bundle_dir = _init(tmp_path, "si8-corrupt")
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"not a safetensors file at all, just bytes")
    with pytest.raises(ValueError, match="invalid safetensors"):
        seed_bundle(bundle_dir, str(bad), by_reference=True)
    assert load_bundle(bundle_dir)["base"]["source"].startswith("demo:")


# -- idempotence / mixed modes / history ----------------------------------------------

def test_ref_reseed_is_idempotent_noop(tmp_path):
    src = _write_ckpt(tmp_path, "ckpt-ref-reseed", _llama_tensors(),
                      config=_llama_config())
    bundle_dir = _init(tmp_path, "si8-reseed")
    first = seed_bundle(bundle_dir, src, importer="test", by_reference=True)
    assert first["already_seeded"] is False
    stats = _store_stats(bundle_dir)
    with open(os.path.join(bundle_dir, "manifest.json"), "rb") as f:
        manifest_bytes = f.read()
    tip = load_bundle(bundle_dir)["journal"]["last_seq"]

    second = seed_bundle(bundle_dir, src, importer="test", by_reference=True)
    assert second["already_seeded"] is True
    assert second["tensors"] == first["tensors"]
    assert _store_stats(bundle_dir) == stats
    with open(os.path.join(bundle_dir, "manifest.json"), "rb") as f:
        assert f.read() == manifest_bytes
    assert load_bundle(bundle_dir)["journal"]["last_seq"] == tip


def test_mixed_mode_reseed_same_checkpoint_is_noop_not_conversion(tmp_path):
    tensors = _simple_tensors()
    src = _write_ckpt(tmp_path, "ckpt-mixed", tensors)

    # copy -> reference request: recognized as the same checkpoint,
    # nothing written, base stays in copy format (no silent conversion).
    copy_dir = _init(tmp_path, "si8-mix-copy")
    seed_bundle(copy_dir, src, importer="test")
    stats = _store_stats(copy_dir)
    res = seed_bundle(copy_dir, src, importer="test", by_reference=True)
    assert res["already_seeded"] is True
    assert _store_stats(copy_dir) == stats
    assert load_bundle(copy_dir)["base"]["seed_format"] == "per-tensor-v1"

    # reference -> copy request: same — no blobs materialize.
    ref_dir = _init(tmp_path, "si8-mix-ref")
    seed_bundle(ref_dir, src, importer="test", by_reference=True)
    stats = _store_stats(ref_dir)
    res = seed_bundle(ref_dir, src, importer="test", by_reference=False)
    assert res["already_seeded"] is True
    assert _store_stats(ref_dir) == stats
    assert load_bundle(ref_dir)["base"]["seed_format"] == "reference-v1"


def test_ref_seed_refuses_history_and_ref_bundle_runs_cycle(tmp_path):
    """A ref-seeded demo-shaped base drives the real loop (the read
    path serves the W tensor), and once the bundle has history any
    further seed — either mode — is refused."""
    from si.loop import run_cycle
    from si.minds import DemoMind

    src = _write_ckpt(tmp_path, "ckpt-ref-cycle",
                      {"W": _simple_tensors()["W"]})
    bundle_dir = _init(tmp_path, "si8-cycle")
    seed_bundle(bundle_dir, src, importer="test", by_reference=True)

    res = run_cycle(bundle_dir, DemoMind(bundle_path=bundle_dir))
    assert res["verdict"] == "ADMIT"
    assert load_bundle(bundle_dir)["head"] != "base"

    other = _write_ckpt(tmp_path, "ckpt-ref-other", _simple_tensors(seed=9))
    with pytest.raises(ValueError, match="existing history"):
        seed_bundle(bundle_dir, other, by_reference=True)
    with pytest.raises(ValueError, match="existing history"):
        seed_bundle(bundle_dir, other, by_reference=False)


# -- export / serve ----------------------------------------------------------------------

def test_ref_export_keeps_config_arch(tmp_path):
    tensors = _llama_tensors()
    src = _write_ckpt(tmp_path, "ckpt-ref-export", tensors,
                      config=_llama_config())
    bundle_dir = _init(tmp_path, "si8-export")
    seed_bundle(bundle_dir, src, importer="test", by_reference=True)
    before = _store_stats(bundle_dir)
    out = str(tmp_path / "ref.gguf")
    export_bundle(bundle_dir, out)
    assert _store_stats(bundle_dir) == before  # export copies nothing in
    assert _gguf_field(out, "general.architecture") == "llama"
    assert _gguf_field(out, "llama.block_count") == 1
    assert _gguf_field(out, "llama.embedding_length") == HIDDEN
    assert _gguf_field(out, "llama.vocab_size") == VOCAB
    assert _gguf_field(out, "si.arch_source") == "checkpoint-config"


def test_ref_serve_transformer_501(tmp_path):
    from si.serve import make_server

    src = _write_ckpt(tmp_path, "ckpt-ref-serve", _llama_tensors(),
                      config=_llama_config())
    bundle_dir = _init(tmp_path, "si8-serve")
    seed_bundle(bundle_dir, src, importer="test", by_reference=True)

    server = make_server(bundle_dir, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=10) as r:
            health = json.loads(r.read().decode("utf-8"))
        assert health["mind_kind"] == "transformer"
        assert health["inference_supported"] is False
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/infer",
            data=json.dumps({"input": [0.0] * 8}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(req, timeout=10)
        assert excinfo.value.code == 501
        body = json.loads(excinfo.value.read().decode("utf-8"))
        assert "not supported" in body["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_ref_bf16_tensor_reads_exact_values(tmp_path):
    """Glimmer's dtype: bf16 has no numpy dtype here, so ref reads widen
    losslessly to float32 (documented in si.bundle.read_tensor_ref);
    hashes/verification are over the raw bf16 bytes regardless."""
    import torch
    from safetensors.torch import save as torch_save

    ckpt = tmp_path / "ckpt-ref-bf16"
    ckpt.mkdir()
    w = torch.tensor([[1.5, -2.25], [3.0, 4.5]], dtype=torch.bfloat16)
    (ckpt / "model.safetensors").write_bytes(torch_save({"W": w}))
    bundle_dir = _init(tmp_path, "si8-bf16")
    seed_bundle(bundle_dir, str(ckpt), importer="test", by_reference=True)
    manifest = load_bundle(bundle_dir)
    ref = manifest["base"]["tensor_refs"]["W"]
    assert ref["dtype"] == "bfloat16" and ref["dtype_code"] == "BF16"
    arr = load_base_tensor(bundle_dir, manifest, "W")
    assert arr.dtype == np.float32
    assert np.array_equal(arr, w.float().numpy())


# -- copy-mode preflight -------------------------------------------------------------------

def test_copy_preflight_refuses_when_space_insufficient(tmp_path, monkeypatch):
    import si.seed as seed_mod

    src = _write_ckpt(tmp_path, "ckpt-preflight", _simple_tensors())
    bundle_dir = _init(tmp_path, "si8-preflight")
    before = _store_stats(bundle_dir)
    monkeypatch.setattr(seed_mod, "_disk_free", lambda path: 0)
    with pytest.raises(ValueError, match="insufficient disk space"):
        seed_bundle(bundle_dir, src, importer="test")
    # Refused BEFORE writing: no new blobs, base untouched.
    assert _store_stats(bundle_dir) == before
    assert load_bundle(bundle_dir)["base"]["source"].startswith("demo:")

    # The same mocked-out space does NOT block a by-reference seed:
    # it stores nothing, so the copy preflight does not apply.
    res = seed_bundle(bundle_dir, src, importer="test", by_reference=True)
    assert res["already_seeded"] is False
    assert _store_stats(bundle_dir) == before
    assert load_bundle(bundle_dir)["base"]["seed_format"] == "reference-v1"


def test_copy_seed_still_works_with_real_space(tmp_path):
    src = _write_ckpt(tmp_path, "ckpt-preflight-ok", _simple_tensors())
    bundle_dir = _init(tmp_path, "si8-preflight-ok")
    res = seed_bundle(bundle_dir, src, importer="test")
    assert res["already_seeded"] is False
    assert res["bytes"] > 0
    assert load_bundle(bundle_dir)["base"]["seed_format"] == "per-tensor-v1"


# -- CLI --------------------------------------------------------------------------------------

def test_cli_seed_by_reference(tmp_path, capsys):
    src = _write_ckpt(tmp_path, "ckpt-ref-cli", _simple_tensors())
    bundle_dir = _init(tmp_path, "si8-cli")
    rc = cli_main(["seed", bundle_dir, src, "--by-reference"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "reference-v1" in out
    assert "DEPENDS on the source checkpoint" in out
    assert load_bundle(bundle_dir)["base"]["seed_format"] == "reference-v1"
