"""SI-9c: apply-at-load for transformer deltas (RFC SI-9 §5, §6.3).

Numeric tests on a synthetic miniature transformer-shaped checkpoint:
Glimmer-style tensor names at toy dimensions, seeded in-test from a
seeded RNG (a locally constructed real-format checkpoint, honestly
NOT a trained model). Proven here:

- an applied tensor equals a hand-computed ``W + (alpha/rank)·B@A``
  reference (computed in this test from the raw factors, never via
  the apply path) to float32 tolerance;
- two stacked deltas accumulate, applied oldest-first through
  ``_active_chain``;
- stack ORDER matters and is distinguishable: an engineered float32
  rounding case gives different results in the two orders, and each
  order's result equals its own sequential hand computation;
- a rolled-back delta (head moved back) no longer applies;
- untargeted tensors return base-identical values;
- ``additive`` targets are refused under the default policy
  (``allow_additive`` false, RFC §4.5 / SI-9b) and applied exactly
  when a policy explicitly allows them;
- apply materialises nothing: the bundle on disk is byte-for-byte
  untouched by a read, and the returned array is a fresh copy;
- a tampered targeting delta fails closed at apply time;
- the loop's VERIFY smoke step (``_verify_transformer_apply``)
  passes a sound delta and fails a tampered one — as an apply smoke
  test only, never as identity/capability probes.
"""
import os

import numpy as np
import pytest
from safetensors.numpy import save

from si.apply import ApplyError, load_applied_tensor
from si.bundle import (
    _anchor_checkpoint,
    _write_manifest,
    init_bundle,
    load_base_tensor,
    load_bundle,
)
from si.journal import open_journal
from si.loop import _verify_transformer_apply, _write_blob
from si.seed import seed_bundle
from si.transformer_delta import DeltaValidationError, construct_transformer_delta

QPROJ = "model.language_model.layers.0.self_attn.q_proj.weight"   # [4, 3]
VPROJ = "model.vision_tower.layers.0.attn.q_proj.weight"          # [2, 2]
GATE = "model.language_model.layers.0.mlp.gate_proj.weight"       # [3, 2]
NORM = "model.language_model.layers.0.input_layernorm.weight"     # [5]

SHAPES = {QPROJ: (4, 3), VPROJ: (2, 2), GATE: (3, 2), NORM: (5,)}


def _tensors(seed=11, overrides=None):
    rng = np.random.default_rng(seed)
    tensors = {n: rng.standard_normal(s).astype(np.float32)
               for n, s in SHAPES.items()}
    if overrides:
        tensors.update(overrides)
    return tensors


def _make_bundle(tmp_path, name, tensors=None):
    tensors = _tensors() if tensors is None else tensors
    ckpt = tmp_path / (name + "-ckpt")
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(save(tensors))
    (ckpt / "config.json").write_text(
        '{"model_type": "muse_glimmer", "architectures": '
        '["MuseGlimmerForConditionalGeneration"]}')
    bundle_dir = init_bundle(str(tmp_path / (name + ".si")), name=name,
                             journal_backend="embedded")
    seed_bundle(bundle_dir, str(ckpt), importer="test")
    return bundle_dir, tensors


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
    """Construct a delta against the current head, write its blob
    (content-addressed, skip-if-exists), append the record, move the
    head, and re-anchor — the state a materialised delta would be in.
    Returns (reloaded manifest, record)."""
    manifest = load_bundle(bundle_dir)
    record, blob = construct_transformer_delta(
        manifest, specs, delta_id=delta_id, parent=manifest["head"],
        created_at="2026-10-19T00:00:00Z",
        self_edit_id="se_" + delta_id)
    _write_blob(bundle_dir, blob)
    manifest["deltas"].append(record)
    manifest["head"] = delta_id
    return _persist(bundle_dir, manifest, f"stack {delta_id}"), record


def _move_head(bundle_dir, to):
    manifest = load_bundle(bundle_dir)
    manifest["head"] = to
    return _persist(bundle_dir, manifest, f"head -> {to}")


def _lora_spec(name, rank=2, alpha=4, seed=1, scale=0.05):
    rng = np.random.default_rng(seed)
    out_dim, in_dim = SHAPES[name]
    return {"name": name, "kind": "lora",
            "A": (rng.standard_normal((rank, in_dim)) * scale).astype(np.float32),
            "B": (rng.standard_normal((out_dim, rank)) * scale).astype(np.float32),
            "rank": rank, "alpha": alpha}


def _hand_applied(base, spec):
    """The reference, computed HERE from raw factors: float32,
    ``W + (alpha/rank)·B@A`` for lora, ``w + d`` for additive."""
    if spec["kind"] == "lora":
        update = (float(spec["alpha"]) / float(spec["rank"])
                  * (spec["B"] @ spec["A"])).astype(np.float32)
    else:
        update = spec["d"]
    return np.asarray(base, dtype=np.float32) + update


def _bundle_stats(bundle_dir):
    stats = {}
    for root, _dirs, files in os.walk(bundle_dir):
        for f in files:
            p = os.path.join(root, f)
            stats[os.path.relpath(p, bundle_dir)] = os.path.getsize(p)
    return stats


# -- numeric correctness ---------------------------------------------------

def test_lora_applied_matches_hand_computed(tmp_path):
    bundle_dir, tensors = _make_bundle(tmp_path, "si9c-lora")
    spec = _lora_spec(QPROJ, seed=3)
    manifest, _record = _stack_delta(bundle_dir, [spec], "d0001")

    applied = load_applied_tensor(bundle_dir, manifest, QPROJ)
    assert applied.dtype == np.float32
    assert applied.shape == SHAPES[QPROJ]
    np.testing.assert_allclose(
        applied, _hand_applied(tensors[QPROJ], spec),
        rtol=1e-6, atol=1e-7)


def test_two_deltas_stack_cumulatively_in_chain_order(tmp_path):
    bundle_dir, tensors = _make_bundle(tmp_path, "si9c-stack")
    spec1 = _lora_spec(QPROJ, seed=5)
    spec2 = _lora_spec(QPROJ, seed=6)
    manifest, _ = _stack_delta(bundle_dir, [spec1], "d0001")
    manifest, _ = _stack_delta(bundle_dir, [spec2], "d0002")

    applied = load_applied_tensor(bundle_dir, manifest, QPROJ)
    expect = _hand_applied(_hand_applied(tensors[QPROJ], spec1), spec2)
    np.testing.assert_allclose(applied, expect, rtol=1e-6, atol=1e-7)


def test_stack_order_matters_and_is_distinguishable(tmp_path):
    """Addition commutes in exact arithmetic but NOT in float32:
    with base entry 1.0 and updates of +2**24 then -2**24, applying
    in chain order rounds the 1.0 away (result 0.0); the reverse
    order preserves it (result 1.0). Both orders are built as real
    stacks and each must equal its own sequential hand computation —
    proving the chain is applied oldest-first, in order."""
    base_v = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    big = float(2 ** 24)
    spec_pos = {"name": VPROJ, "kind": "lora", "rank": 1, "alpha": 1,
                "A": np.array([[1.0, 0.0]], dtype=np.float32),
                "B": np.array([[big], [0.0]], dtype=np.float32)}
    spec_neg = {"name": VPROJ, "kind": "lora", "rank": 1, "alpha": 1,
                "A": np.array([[1.0, 0.0]], dtype=np.float32),
                "B": np.array([[-big], [0.0]], dtype=np.float32)}

    results = {}
    for label, order in (("pos_first", (spec_pos, spec_neg)),
                         ("neg_first", (spec_neg, spec_pos))):
        bundle_dir, _ = _make_bundle(
            tmp_path, f"si9c-order-{label}",
            tensors=_tensors(overrides={VPROJ: base_v}))
        manifest = None
        for i, spec in enumerate(order):
            manifest, _ = _stack_delta(bundle_dir, [spec], f"d000{i + 1}")
        applied = load_applied_tensor(bundle_dir, manifest, VPROJ)
        expect = base_v.copy()
        for spec in order:  # sequential float32 hand computation
            expect = _hand_applied(expect, spec)
        np.testing.assert_array_equal(applied, expect)
        results[label] = float(applied[0, 0])

    assert results["pos_first"] == 0.0   # 1.0 rounded away, then cancelled
    assert results["neg_first"] == 1.0   # 1.0 preserved exactly
    assert results["pos_first"] != results["neg_first"]


def test_rolled_back_delta_no_longer_applies(tmp_path):
    bundle_dir, tensors = _make_bundle(tmp_path, "si9c-rollback")
    spec1 = _lora_spec(QPROJ, seed=7)
    spec2 = _lora_spec(QPROJ, seed=8)
    _stack_delta(bundle_dir, [spec1], "d0001")
    manifest, _ = _stack_delta(bundle_dir, [spec2], "d0002")
    both = load_applied_tensor(bundle_dir, manifest, QPROJ)

    # Rollback = head moves back (blobs/records stay as evidence);
    # the popped delta simply stops being applied (RFC §4.4/§5.1).
    manifest = _move_head(bundle_dir, "d0001")
    np.testing.assert_allclose(
        load_applied_tensor(bundle_dir, manifest, QPROJ),
        _hand_applied(tensors[QPROJ], spec1), rtol=1e-6, atol=1e-7)

    manifest = _move_head(bundle_dir, "base")
    np.testing.assert_array_equal(
        load_applied_tensor(bundle_dir, manifest, QPROJ), tensors[QPROJ])
    assert not np.array_equal(both, tensors[QPROJ])  # the stack was real


def test_untargeted_tensor_returns_base_identical(tmp_path):
    bundle_dir, tensors = _make_bundle(tmp_path, "si9c-untargeted")
    manifest, _ = _stack_delta(bundle_dir, [_lora_spec(QPROJ, seed=9)],
                               "d0001")
    applied = load_applied_tensor(bundle_dir, manifest, GATE)
    assert applied.dtype == np.float32
    np.testing.assert_array_equal(applied, tensors[GATE])
    np.testing.assert_array_equal(
        applied, load_base_tensor(bundle_dir, manifest, GATE))


# -- additive: policy-gated (RFC §4.5, SI-9b allow_additive default false) --

def test_additive_refused_by_default_applies_when_policy_allows(tmp_path):
    bundle_dir, tensors = _make_bundle(tmp_path, "si9c-additive")
    rng = np.random.default_rng(4)
    spec = {"name": NORM, "kind": "additive",
            "d": (rng.standard_normal(SHAPES[NORM]) * 0.05).astype(np.float32)}
    manifest, _ = _stack_delta(bundle_dir, [spec], "d0001")

    # The bundle's own policy.json (DEFAULT_POLICY) has
    # allow_additive false — and an explicit empty policy is the same.
    with pytest.raises(ApplyError, match="allow_additive"):
        load_applied_tensor(bundle_dir, manifest, NORM)
    with pytest.raises(ApplyError, match="allow_additive"):
        load_applied_tensor(bundle_dir, manifest, NORM, policy={})

    applied = load_applied_tensor(bundle_dir, manifest, NORM,
                                  policy={"allow_additive": True})
    np.testing.assert_allclose(
        applied, _hand_applied(tensors[NORM], spec), rtol=1e-6, atol=1e-7)


# -- read-only / fail-closed -------------------------------------------------

def test_apply_materialises_nothing_and_returns_a_copy(tmp_path):
    bundle_dir, tensors = _make_bundle(tmp_path, "si9c-readonly")
    spec = _lora_spec(QPROJ, seed=10)
    manifest, _ = _stack_delta(bundle_dir, [spec], "d0001")
    before = _bundle_stats(bundle_dir)

    applied = load_applied_tensor(bundle_dir, manifest, QPROJ)
    applied[:] = 0.0  # mutating the result must not touch anything
    assert _bundle_stats(bundle_dir) == before
    again = load_applied_tensor(bundle_dir, manifest, QPROJ)
    np.testing.assert_allclose(
        again, _hand_applied(tensors[QPROJ], spec), rtol=1e-6, atol=1e-7)
    assert _bundle_stats(bundle_dir) == before


def test_tampered_targeting_delta_fails_closed(tmp_path):
    bundle_dir, _tensors_used = _make_bundle(tmp_path, "si9c-tamper")
    manifest, record = _stack_delta(
        bundle_dir, [_lora_spec(QPROJ, seed=12)], "d0001")
    blob = os.path.join(
        bundle_dir, "blobs", "sha256",
        record["blob"].split(":", 1)[1] + ".safetensors")
    with open(blob, "r+b") as f:
        f.seek(-1, os.SEEK_END)
        last = f.read(1)
        f.seek(-1, os.SEEK_END)
        f.write(bytes([last[0] ^ 0xFF]))
    # The in-memory manifest is pre-tamper; apply re-validates the
    # record/blob pair (content hash) and must refuse, not apply.
    with pytest.raises(DeltaValidationError):
        load_applied_tensor(bundle_dir, manifest, QPROJ)


def test_unknown_tensor_raises_keyerror(tmp_path):
    bundle_dir, _ = _make_bundle(tmp_path, "si9c-unknown")
    manifest = load_bundle(bundle_dir)
    with pytest.raises(KeyError):
        load_applied_tensor(bundle_dir, manifest, "model.nope.weight")


# -- loop VERIFY smoke step (RFC §4.4/§5.5) -----------------------------------

def test_verify_transformer_apply_smoke_passes_and_labels_itself(tmp_path):
    bundle_dir, _ = _make_bundle(tmp_path, "si9c-smoke")
    manifest, record = _stack_delta(
        bundle_dir, [_lora_spec(QPROJ, seed=13)], "d0001")
    ok, report = _verify_transformer_apply(bundle_dir, manifest, record)
    assert ok is True
    assert report["kind"] == "transformer-apply-smoke"
    assert report["targets"][0]["name"] == QPROJ
    assert report["targets"][0]["passed"] is True
    # Honestly labeled: NOT a probe result.
    assert "NOT identity or capability" in report["note"]


def test_verify_transformer_apply_smoke_fails_on_tamper(tmp_path):
    bundle_dir, _ = _make_bundle(tmp_path, "si9c-smoke-bad")
    manifest, record = _stack_delta(
        bundle_dir, [_lora_spec(QPROJ, seed=14)], "d0001")
    blob = os.path.join(
        bundle_dir, "blobs", "sha256",
        record["blob"].split(":", 1)[1] + ".safetensors")
    with open(blob, "r+b") as f:
        f.seek(-1, os.SEEK_END)
        last = f.read(1)
        f.seek(-1, os.SEEK_END)
        f.write(bytes([last[0] ^ 0xFF]))
    ok, report = _verify_transformer_apply(bundle_dir, manifest, record)
    assert ok is False
    assert report["targets"][0]["passed"] is False
    assert "error" in report["targets"][0]
