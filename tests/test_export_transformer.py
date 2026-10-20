"""SI-11: export reconciliation for transformer deltas (RFC §5.5).

Before SI-11, ``materialize_head`` called the demo-era ``_load_delta``
for every record, so an operator-admitted transformer delta crashed
export with a raw ``SafetensorError: File does not contain tensor
delta.B`` (reproduced by the driver and by the SI-11 author on a
synthetic miniature bundle + admitted LoRA delta to q_proj).

Proven here, on synthetic miniature fixtures (test_admit/test_apply
style):
(a) an admitted lora delta materializes/exports to the hand-computed
    value and differs from base, and matches load_applied_tensor;
(b) untargeted tensors are base-identical;
(c) two stacked transformer deltas apply oldest-first matching
    load_applied_tensor;
(d) a rolled-back delta is NOT applied;
(e) additive refused when allow_additive is false, applied when true;
(f) malformed/unknown-target transformer delta fails closed with
    ExportError, writes no output file, journals no export;
(g) demo export unchanged — covered by tests/test_export.py (run in
    the full suite); a mixed-stack test proves the documented SI-11
    choice: both kinds apply in chain order, each by its own rules.
"""
import os

import numpy as np
import pytest
from safetensors.numpy import save

from si.admit import admit_transformer_delta
from si.apply import load_applied_tensor
from si.bundle import (
    DEFAULT_POLICY,
    _active_chain,
    _anchor_checkpoint,
    _write_manifest,
    init_bundle,
    load_bundle,
)
from si.export import ExportError, export_bundle, materialize_head
from si.gate import Gate, Intent
from si.journal import open_journal
from si.loop import _write_blob
from si.minds import _adapter_blob
from si.seed import seed_bundle
from si.transformer_delta import KIND, construct_transformer_delta

QPROJ = "model.language_model.layers.0.self_attn.q_proj.weight"   # [4, 3]
VPROJ = "model.vision_tower.layers.0.attn.q_proj.weight"          # [2, 2]
NORM = "model.language_model.layers.0.input_layernorm.weight"     # [5]
SHAPES = {QPROJ: (4, 3), VPROJ: (2, 2), NORM: (5,)}
OPERATOR = "test-operator"
REASON = "reviewed the structural report; admitting as the operator check"


def _make_bundle(tmp_path, name, tensors=None):
    rng = np.random.default_rng(11)
    tensors = ({n: rng.standard_normal(s).astype(np.float32)
                for n, s in SHAPES.items()}
               if tensors is None else tensors)
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


def _lora_spec(name=QPROJ, rank=2, alpha=4, seed=1, scale=0.05):
    rng = np.random.default_rng(seed)
    out_dim, in_dim = SHAPES[name]
    return {"name": name, "kind": "lora",
            "A": (rng.standard_normal((rank, in_dim)) * scale).astype(np.float32),
            "B": (rng.standard_normal((out_dim, rank)) * scale).astype(np.float32),
            "rank": rank, "alpha": alpha}


def _hand(base, spec):
    if spec["kind"] == "lora":
        update = (float(spec["alpha"]) / float(spec["rank"])
                  * (spec["B"] @ spec["A"])).astype(np.float32)
    else:
        update = np.asarray(spec["d"], dtype=np.float32)
    return np.asarray(base, dtype=np.float32) + update


def _policy(**over):
    policy = dict(DEFAULT_POLICY)
    policy["allowed_target_patterns"] = [QPROJ, VPROJ, NORM]
    policy["max_targets_per_delta"] = 4
    policy.update(over)
    return policy


def _admit(bundle_dir, specs, candidate_id="d0007"):
    manifest = load_bundle(bundle_dir)
    record, blob = construct_transformer_delta(
        manifest, specs, delta_id=candidate_id,
        parent=manifest["head"], created_at="2026-10-19T00:00:00Z",
        self_edit_id="se_" + candidate_id)
    intent = Intent(kind=KIND, reason="Adapt one named tensor.",
                    parents=["base"], source_seqs=[1],
                    targets=[t["name"] for t in record["targets"]])
    # Additive admission needs the gate to allow additive; lora does not.
    allow_add = any(s.get("kind") == "additive" for s in specs)
    gate = Gate(_policy(allow_additive=True) if allow_add else _policy())
    return admit_transformer_delta(
        bundle_dir, {"record": record, "blob_bytes": blob}, intent,
        operator=OPERATOR, reason=REASON, gate=gate)


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
    manifest = load_bundle(bundle_dir)
    record, blob = construct_transformer_delta(
        manifest, specs, delta_id=delta_id, parent=manifest["head"],
        created_at="2026-10-19T00:00:00Z", self_edit_id="se_" + delta_id)
    _write_blob(bundle_dir, blob)
    manifest["deltas"].append(record)
    manifest["head"] = delta_id
    return _persist(bundle_dir, manifest, f"stack {delta_id}"), record


def _move_head(bundle_dir, to):
    manifest = load_bundle(bundle_dir)
    manifest["head"] = to
    return _persist(bundle_dir, manifest, f"head -> {to}")


# (a) + (b)
def test_admitted_lora_materializes_to_hand_computed_and_exports(tmp_path):
    bundle_dir, tensors = _make_bundle(tmp_path, "si11-a")
    spec = _lora_spec(seed=3)
    res = _admit(bundle_dir, [spec])
    assert res["admitted"] is True and res["head"] == "d0001"

    out = materialize_head(bundle_dir)
    expected = _hand(tensors[QPROJ], spec)
    np.testing.assert_allclose(out[QPROJ], expected, rtol=1e-6, atol=1e-7)
    assert not np.array_equal(out[QPROJ], tensors[QPROJ])
    # Consistency with apply-at-load for the same tensor.
    manifest = load_bundle(bundle_dir)
    np.testing.assert_allclose(
        out[QPROJ], load_applied_tensor(bundle_dir, manifest, QPROJ),
        rtol=1e-6, atol=1e-7)
    # (b) untargeted tensors base-identical
    np.testing.assert_array_equal(out[VPROJ], tensors[VPROJ])
    np.testing.assert_array_equal(out[NORM], tensors[NORM])

    # Full export writes a GGUF and journals export provenance.
    gguf_path = str(tmp_path / "head.gguf")
    eres = export_bundle(bundle_dir, gguf_path)
    assert eres["head"] == "d0001"
    assert os.path.isfile(gguf_path) and eres["bytes"] > 0


# (c) two stacked, oldest-first, matching load_applied_tensor
def test_two_stacked_transformer_deltas_oldest_first(tmp_path):
    bundle_dir, tensors = _make_bundle(tmp_path, "si11-c")
    spec1 = _lora_spec(seed=5)
    spec2 = _lora_spec(seed=6)
    manifest, _ = _stack_delta(bundle_dir, [spec1], "d0001")
    manifest, _ = _stack_delta(bundle_dir, [spec2], "d0002")
    out = materialize_head(bundle_dir, manifest)
    expected = _hand(_hand(tensors[QPROJ], spec1), spec2)
    np.testing.assert_allclose(out[QPROJ], expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        out[QPROJ], load_applied_tensor(bundle_dir, manifest, QPROJ),
        rtol=1e-6, atol=1e-7)


# (d) rolled-back delta not applied
def test_rolled_back_delta_not_applied(tmp_path):
    bundle_dir, tensors = _make_bundle(tmp_path, "si11-d")
    spec1 = _lora_spec(seed=7)
    spec2 = _lora_spec(seed=8)
    _stack_delta(bundle_dir, [spec1], "d0001")
    manifest, _ = _stack_delta(bundle_dir, [spec2], "d0002")
    assert [d["id"] for d in _active_chain(manifest)] == ["d0001", "d0002"]
    manifest = _move_head(bundle_dir, "d0001")
    assert [d["id"] for d in _active_chain(manifest)] == ["d0001"]
    out = materialize_head(bundle_dir, manifest)
    np.testing.assert_allclose(out[QPROJ], _hand(tensors[QPROJ], spec1),
                               rtol=1e-6, atol=1e-7)
    manifest = _move_head(bundle_dir, "base")
    out = materialize_head(bundle_dir, manifest)
    np.testing.assert_array_equal(out[QPROJ], tensors[QPROJ])


# (e) additive refused / allowed
def test_additive_refused_when_not_allowed_applied_when_true(tmp_path):
    bundle_dir, tensors = _make_bundle(tmp_path, "si11-e")
    rng = np.random.default_rng(4)
    spec = {"name": NORM, "kind": "additive",
            "d": (rng.standard_normal(SHAPES[NORM]) * 0.05).astype(np.float32)}
    manifest, _ = _stack_delta(bundle_dir, [spec], "d0001")
    # Default bundle policy has allow_additive False; explicit {} too.
    with pytest.raises(ExportError, match="allow_additive"):
        materialize_head(bundle_dir, manifest)
    with pytest.raises(ExportError, match="allow_additive"):
        materialize_head(bundle_dir, manifest, policy={})
    with pytest.raises(ExportError, match="allow_additive"):
        export_bundle(bundle_dir, str(tmp_path / "add.gguf"))
    assert not os.path.exists(str(tmp_path / "add.gguf"))
    out = materialize_head(bundle_dir, manifest,
                           policy={"allow_additive": True})
    np.testing.assert_allclose(out[NORM], _hand(tensors[NORM], spec),
                               rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        out[NORM],
        load_applied_tensor(bundle_dir, manifest, NORM,
                            policy={"allow_additive": True}),
        rtol=1e-6, atol=1e-7)


# (f) malformed / unknown target fails closed, no file, no export journal
def test_unknown_target_fails_closed_no_file_no_journal(tmp_path):
    bundle_dir, tensors = _make_bundle(tmp_path, "si11-f")
    manifest, record = _stack_delta(bundle_dir, [_lora_spec(seed=9)], "d0001")
    # Tamper the record in the manifest: unknown exact target name.
    # Blob hash still matches; journal re-anchored via _persist, so
    # load_bundle passes and export's own validation must fail closed.
    manifest = load_bundle(bundle_dir)
    manifest["deltas"][0] = dict(manifest["deltas"][0])
    manifest["deltas"][0]["targets"] = [
        dict(manifest["deltas"][0]["targets"][0], name="model.nope.weight")]
    manifest = _persist(bundle_dir, manifest, "tamper target name")

    with pytest.raises(ExportError) as ei:
        materialize_head(bundle_dir, manifest)
    # Clear named error, not a raw SafetensorError/KeyError leak.
    assert "SafetensorError" not in str(type(ei.value))
    assert not isinstance(ei.value, KeyError)

    manifest_before = load_bundle(bundle_dir)
    journal = open_journal(bundle_dir, manifest_before)
    try:
        n_prov_before = journal.backend.count("provenance")
    finally:
        journal.close()
    out_path = str(tmp_path / "should-not-exist.gguf")
    with pytest.raises(ExportError):
        export_bundle(bundle_dir, out_path)
    assert not os.path.exists(out_path), "failed export must write no file"
    journal = open_journal(bundle_dir, load_bundle(bundle_dir))
    try:
        assert journal.backend.count("provenance") == n_prov_before
        rows = journal.backend.find("provenance",
                                    filter={"relation": "export"})
        assert rows == []
    finally:
        journal.close()


def test_tampered_factor_blob_fails_closed_with_export_error(tmp_path):
    bundle_dir, tensors = _make_bundle(tmp_path, "si11-f2")
    manifest, record = _stack_delta(bundle_dir, [_lora_spec(seed=10)], "d0001")
    # Corrupt a norm in the record (validation recomputes norms).
    manifest = load_bundle(bundle_dir)
    manifest["deltas"][0] = dict(manifest["deltas"][0])
    manifest["deltas"][0]["norm"] = 999.0
    manifest = _persist(bundle_dir, manifest, "tamper norm")
    with pytest.raises(ExportError, match="invalid transformer delta"):
        materialize_head(bundle_dir, manifest)


# Mixed stacks: documented choice — apply both in chain order
def test_mixed_demo_and_transformer_apply_in_chain_order(tmp_path):
    bundle_dir, tensors = _make_bundle(tmp_path, "si11-mixed")
    # Transformer delta first (stacked directly).
    spec_t = _lora_spec(name=VPROJ, seed=12)
    manifest, _ = _stack_delta(bundle_dir, [spec_t], "d0001")
    # Demo delta second, targeting VPROJ by substring "vision_tower".
    # Its B@A shape must match VPROJ [2,2]: rank 1, A [1,2], B [2,1].
    B = np.array([[0.5], [-0.25]], dtype=np.float32)
    A = np.array([[1.0, 2.0]], dtype=np.float32)
    blob = _adapter_blob("test", B, A, rank=1, alpha=2,
                         target_modules=["vision_tower"],
                         training_signal_id="mixed-test")
    blob_hex, byte_len = _write_blob(bundle_dir, blob)
    manifest = load_bundle(bundle_dir)
    manifest["deltas"].append({
        "id": "d0002", "parent": "d0001", "blob": "sha256:" + blob_hex,
        "bytes": byte_len, "rank": 1, "alpha": 2,
        "target_modules": ["vision_tower"], "norm": 1.0,
        "created_at": "2026-10-19T00:00:00Z", "self_edit_id": "se_demo"})
    manifest["head"] = "d0002"
    manifest = _persist(bundle_dir, manifest, "stack mixed demo")

    out = materialize_head(bundle_dir, manifest)
    # VPROJ: transformer first, then demo update (alpha/rank=2)*B@A.
    after_tf = _hand(tensors[VPROJ], spec_t)
    demo_update = (2.0 * (B @ A)).astype(np.float32)
    expected = np.asarray(after_tf + demo_update, dtype=np.float32)
    np.testing.assert_allclose(out[VPROJ], expected, rtol=1e-6, atol=1e-7)
    # QPROJ untargeted by either in this stack.
    np.testing.assert_array_equal(out[QPROJ], tensors[QPROJ])
