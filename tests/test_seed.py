"""``si seed``: import a real safetensors checkpoint as the frozen base.

Covers: happy path (manifest rewritten, per-tensor blobs content-
addressed, provenance recorded, bundle still verifies), multi-file
imports, corrupt / truncated / non-safetensors rejection, refusal on
bundles with history, and a seeded bundle running a real self-write
cycle. SI-7 per-tensor / dedup / sharded-layout specifics live in
tests/test_seed_real.py.
"""
import os
import struct

import numpy as np
import pytest
from safetensors.numpy import save

from si.bundle import init_bundle, load_bundle, sha256_hex
from si.journal import open_journal
from si.seed import seed_bundle


def _make_checkpoint(tmp_path, files=None):
    """Write real .safetensors files under tmp_path/ckpt.

    files: {filename: {tensor_name: shape, ...}}; tensors are float32.
    Returns the checkpoint directory path.
    """
    if files is None:
        files = {"model.safetensors": {"W": (8, 8), "b": (8,)}}
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    for name, spec in files.items():
        tensors = {t: rng.standard_normal(shape).astype(np.float32)
                   for t, shape in spec.items()}
        (ckpt_dir / name).write_bytes(save(tensors))
    return str(ckpt_dir)


def _init(tmp_path, name):
    return init_bundle(str(tmp_path / (name + ".si")), name=name)


def _blob_path(bundle_dir, blob_ref):
    scheme, hexpart = blob_ref.split(":", 1)
    assert scheme == "sha256"
    return os.path.join(bundle_dir, "blobs", "sha256", hexpart + ".safetensors")


def test_seed_happy_path(tmp_path):
    bundle_dir = _init(tmp_path, "seed-happy")
    src = _make_checkpoint(tmp_path)
    res = seed_bundle(bundle_dir, src, importer="operator")

    manifest = load_bundle(bundle_dir)  # still verifies cleanly
    base = manifest["base"]
    assert base["source"].startswith("seed:")
    assert "imported" in base["source"]
    assert base["format"] == "safetensors"
    assert base["importer"] == "operator"
    assert base["imported_at"]
    assert len(base["parts"]) == 1

    # SI-7: every TENSOR has its own content-addressed blob, and every
    # referenced blob exists with a matching content hash.
    assert base["seed_format"] == "per-tensor-v1"
    assert set(base["tensor_blobs"]) == {"W", "b"}
    unique_sizes = {}
    for name, ref in base["tensor_blobs"].items():
        p = _blob_path(bundle_dir, ref["blob"])
        assert os.path.isfile(p), f"missing blob {p}"
        with open(p, "rb") as f:
            assert sha256_hex(f.read()) == ref["blob"].split(":", 1)[1]
        assert ref["bytes"] == os.path.getsize(p)
        unique_sizes[ref["blob"]] = ref["bytes"]
    # base["bytes"] counts each unique blob once (never duplicated).
    assert base["bytes"] == sum(unique_sizes.values())

    # Parts are now source-file provenance (name, source bytes + hash,
    # the file's tensor metadata) — the source file itself is not a blob.
    part = base["parts"][0]
    assert part["file"] == "model.safetensors"
    assert part["sha256"].startswith("sha256:")
    assert "blob" not in part

    # Tensor metadata recorded from the headers.
    assert base["tensors"]["W"] == {"shape": [8, 8], "dtype": "float32"}
    assert base["tensors"]["b"] == {"shape": [8], "dtype": "float32"}
    assert base["parts"][0]["tensors"]["W"]["shape"] == [8, 8]

    # "blob" points at the "W" tensor's blob: single-blob loaders
    # (DemoMind, the loop) keep working on seeded demo-shaped bases.
    assert base["blob"] == base["tensor_blobs"]["W"]["blob"]

    # Provenance: seeded_from edge on child "base".
    journal = open_journal(bundle_dir, manifest)
    try:
        rows = journal.backend.find(
            "provenance", filter={"child": "base", "relation": "seeded_from"})
    finally:
        journal.close()
    assert rows, "expected seeded_from provenance for child 'base'"
    assert rows[0]["parents"] == [os.path.abspath(src)]
    assert "1 tensor file" in rows[0]["note"]

    # Return contract.
    assert res["parts"] == 1
    assert res["bytes"] == base["bytes"]
    assert res["head"] == "base"
    assert manifest["head"] == "base"
    assert res["base"] == base


def test_seed_single_file_source(tmp_path):
    """A single .safetensors file (not a directory) is a valid source."""
    bundle_dir = _init(tmp_path, "seed-file")
    src_file = os.path.join(
        _make_checkpoint(tmp_path), "model.safetensors")
    res = seed_bundle(bundle_dir, src_file)
    manifest = load_bundle(bundle_dir)
    assert manifest["base"]["parts"][0]["file"] == "model.safetensors"
    assert manifest["base"]["source"].startswith(f"seed:{os.path.abspath(src_file)}")
    assert res["parts"] == 1


def test_seed_multifile(tmp_path):
    bundle_dir = _init(tmp_path, "seed-multi")
    src = _make_checkpoint(tmp_path, files={
        "a_part.safetensors": {"W": (8, 8)},
        "b_part.safetensors": {"V": (4, 4)},
    })
    res = seed_bundle(bundle_dir, src)

    manifest = load_bundle(bundle_dir)
    parts = manifest["base"]["parts"]
    assert len(parts) == 2
    assert [p["file"] for p in parts] == ["a_part.safetensors", "b_part.safetensors"]
    unique_sizes = {ref["blob"]: ref["bytes"]
                    for ref in manifest["base"]["tensor_blobs"].values()}
    assert manifest["base"]["bytes"] == sum(unique_sizes.values())
    assert res["bytes"] == manifest["base"]["bytes"]
    assert res["parts"] == 2
    # Tensors merged across parts; "blob" points at the "W" tensor blob.
    assert set(manifest["base"]["tensors"]) == {"W", "V"}
    assert manifest["base"]["tensors"]["V"] == {"shape": [4, 4], "dtype": "float32"}
    assert manifest["base"]["blob"] == manifest["base"]["tensor_blobs"]["W"]["blob"]
    for ref in manifest["base"]["tensor_blobs"].values():
        assert os.path.isfile(_blob_path(bundle_dir, ref["blob"]))


def test_seed_rejects_corrupt_header(tmp_path):
    bundle_dir = _init(tmp_path, "seed-corrupt")
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"this is not a safetensors file, just garbage bytes")
    with pytest.raises(ValueError, match="invalid safetensors"):
        seed_bundle(bundle_dir, str(bad))
    # The failed seed changed nothing: bundle still loads, base untouched.
    manifest = load_bundle(bundle_dir)
    assert manifest["base"]["source"].startswith("demo:")


def test_seed_rejects_truncated(tmp_path):
    bundle_dir = _init(tmp_path, "seed-trunc")
    bad = tmp_path / "trunc.safetensors"
    bad.write_bytes(struct.pack("<Q", 2 ** 40) + b"\x00" * 16)
    with pytest.raises(ValueError, match="invalid safetensors"):
        seed_bundle(bundle_dir, str(bad))
    manifest = load_bundle(bundle_dir)
    assert manifest["base"]["source"].startswith("demo:")


def test_seed_rejects_non_safetensors_extension(tmp_path):
    bundle_dir = _init(tmp_path, "seed-ext")
    bad = tmp_path / "weights.bin"
    bad.write_bytes(b"\x00" * 64)
    with pytest.raises(ValueError, match=r"\.safetensors"):
        seed_bundle(bundle_dir, str(bad))


def test_seed_rejects_missing_source(tmp_path):
    bundle_dir = _init(tmp_path, "seed-missing")
    with pytest.raises(FileNotFoundError):
        seed_bundle(bundle_dir, str(tmp_path / "nope"))


def test_seed_refuses_history(tmp_path):
    from si.loop import run_cycle
    from si.minds import DemoMind

    bundle_dir = _init(tmp_path, "seed-hist")
    res = run_cycle(bundle_dir, DemoMind(bundle_path=bundle_dir))
    assert res["verdict"] == "ADMIT"  # the bundle now has a delta
    manifest = load_bundle(bundle_dir)
    assert manifest["deltas"] and manifest["head"] != "base"

    src = _make_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="existing history"):
        seed_bundle(bundle_dir, src)
    # Refusal changed nothing.
    manifest = load_bundle(bundle_dir)
    assert manifest["base"]["source"].startswith("demo:")


def test_seeded_bundle_runs_cycle(tmp_path):
    """A seeded bundle drives the real self-write loop end to end."""
    from si.loop import run_cycle
    from si.minds import DemoMind

    bundle_dir = _init(tmp_path, "seed-run")
    _make_checkpoint(tmp_path, files={"model.safetensors": {"W": (8, 8)}})
    seed_bundle(bundle_dir, os.path.join(str(tmp_path), "ckpt"))

    res = run_cycle(bundle_dir, DemoMind(bundle_path=bundle_dir))
    assert res["verdict"] == "ADMIT"

    manifest = load_bundle(bundle_dir)
    assert manifest["head"] == res["head"] != "base"
