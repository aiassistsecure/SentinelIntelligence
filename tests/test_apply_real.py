"""SI-9c: apply-at-load — ONE real-checkpoint proof (RFC §6.3, §7.1).

Read-only against the existing seeded Glimmer bundle
(``~/workspace/sentinel-glimmer.si``, SI-8 by-reference seed of
``~/workspace/glimmer-safetensors``): a delta fixture authored HERE
in the test — not proposed by any mind, not learned from data —
targets one small real tensor, a vision-tower projection
``model.vision_tower.layers.0.attn.q_proj.weight`` ([1536, 1536],
bfloat16; chosen per RFC §7.4 Q4 because it is small — the language
matrices are >= [6656, ...] and the 1-D alternatives would only
exercise the policy-refused additive path). The applied result must
have the right shape, be finite, and equal the hand-computed
``base + (alpha/rank)·B@A`` over the real bf16 bytes (widened
losslessly to float32); the delta-free read must be unchanged.

Cost discipline (RFC §7.1): this test NEVER calls ``load_bundle`` —
a full verification of the by-reference Glimmer base is a ~29-minute
streaming pass over 59.5 GB, and triggering one per assertion would
make the suite unusable. The manifest is read ONCE, directly as
JSON; every tensor read goes through ``load_base_tensor`` /
``load_applied_tensor``, which re-verify only the single tensor's
byte range (4.7 MB here). Marked ``slow`` like the existing dogfood
tests, and skipped when the seeded bundle is not present (other
machines keep a hermetic suite).

Read-only discipline: the seeded bundle and the checkpoint are
never written. The bundle directory is copied to a tmp dir (2.9 MB
of manifest/journal/policy — the base tensors are by-reference and
are NOT copied; the copy still points at the original checkpoint),
the delta blob is written only into the COPY's store, and the test
asserts the original bundle's manifest bytes and blob store are
unchanged afterwards.
"""
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from si.apply import load_applied_tensor
from si.bundle import load_base_tensor
from si.transformer_delta import construct_transformer_delta

TARGET = "model.vision_tower.layers.0.attn.q_proj.weight"
SEEDED_BUNDLE = Path.home() / "workspace" / "sentinel-glimmer.si"


@pytest.mark.slow
def test_apply_real_glimmer_vision_projection(tmp_path):
    if not (SEEDED_BUNDLE / "manifest.json").is_file():
        pytest.skip("seeded Glimmer bundle not present on this machine")
    original_manifest_bytes = (SEEDED_BUNDLE / "manifest.json").read_bytes()
    original_store = sorted(os.listdir(SEEDED_BUNDLE / "blobs" / "sha256"))

    # Work on a copy of the bundle DIR only (see module docstring).
    bundle_dir = str(tmp_path / "glimmer-copy.si")
    shutil.copytree(SEEDED_BUNDLE, bundle_dir)

    # ONE direct manifest read — no load_bundle, no full verification.
    with open(os.path.join(bundle_dir, "manifest.json"), "rb") as f:
        manifest = json.loads(f.read().decode("utf-8"))
    assert manifest["base"]["seed_format"] == "reference-v1"
    assert manifest["head"] == "base" and manifest["deltas"] == []
    ref = manifest["base"]["tensor_refs"][TARGET]
    assert ref["shape"] == [1536, 1536] and ref["dtype"] == "bfloat16"

    # Delta-free baseline: one per-tensor verified read of real bf16.
    base = load_base_tensor(bundle_dir, manifest, TARGET)
    assert base.dtype == np.float32 and base.shape == (1536, 1536)
    assert np.all(np.isfinite(base))
    # With no deltas, apply-at-load IS the base read.
    np.testing.assert_array_equal(
        load_applied_tensor(bundle_dir, manifest, TARGET), base)

    # The fixture delta, authored here: rank 2, seeded RNG, small.
    rng = np.random.default_rng(20261019)
    rank, alpha = 2, 4.0
    A = (rng.standard_normal((rank, 1536)) * 0.001).astype(np.float32)
    B = (rng.standard_normal((1536, rank)) * 0.001).astype(np.float32)
    record, blob = construct_transformer_delta(
        manifest,
        [{"name": TARGET, "kind": "lora", "A": A, "B": B,
          "rank": rank, "alpha": alpha}],
        delta_id="d0001", parent="base",
        created_at="2026-10-19T00:00:00Z", self_edit_id="se_fixture")
    # Written ONLY into the copy's content-addressed store.
    from si.loop import _write_blob
    _write_blob(bundle_dir, blob)
    manifest["deltas"] = [record]
    manifest["head"] = "d0001"

    applied = load_applied_tensor(bundle_dir, manifest, TARGET)
    assert applied.shape == (1536, 1536)
    assert applied.dtype == np.float32
    assert np.all(np.isfinite(applied))
    expect = base + ((alpha / rank) * (B @ A)).astype(np.float32)
    np.testing.assert_allclose(applied, expect, rtol=1e-6, atol=1e-6)
    assert not np.array_equal(applied, base)  # the delta really moved it

    # The delta-free read is unchanged by the apply (base is frozen).
    np.testing.assert_array_equal(
        load_base_tensor(bundle_dir, {**manifest, "head": "base",
                                      "deltas": []}, TARGET), base)

    # The originals were never touched.
    assert (SEEDED_BUNDLE / "manifest.json").read_bytes() == \
        original_manifest_bytes
    assert sorted(os.listdir(SEEDED_BUNDLE / "blobs" / "sha256")) == \
        original_store
