"""SI-9a: transformer-delta format — round-trip + fail-closed validation.

Synthetic fixtures only: small tensors with Glimmer-style names, fake
content hashes, no Glimmer reads, no bundle on disk. Every malformed
case named in RFC §6.1 / SI-9a scope must fail closed with
DeltaValidationError: unknown target, wrong base_blob, shape lies
(record vs blob header vs base), dtype mismatch, non-1D/2D targets,
lora-on-1D / additive-on-2D, rank < 1, missing/extra factor keys,
metadata-vs-manifest mismatch, and norm lies.
"""
import copy
import hashlib
import json

import numpy as np
import pytest
from safetensors.numpy import save as st_save

from si.transformer_delta import (
    KIND,
    DeltaValidationError,
    aggregate_norm,
    construct_transformer_delta,
    parse_blob,
    parse_record,
    serialize_blob,
    serialize_record,
    validate_transformer_delta,
)

QPROJ = "model.language_model.layers.0.self_attn.q_proj.weight"   # [4, 3]
VPROJ = "model.vision_tower.layers.0.attn.q_proj.weight"          # [2, 2]
NORM = "model.language_model.layers.0.input_layernorm.weight"     # [5]
CONV = "model.vision_tower.patch_embedding.weight"                # [2, 2, 2] 3-D


def _fake_blob(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode()).hexdigest()


def _base(mode="tensor_refs"):
    entries = {
        QPROJ: {"blob": _fake_blob("q"), "shape": [4, 3], "dtype": "bfloat16"},
        VPROJ: {"blob": _fake_blob("v"), "shape": [2, 2], "dtype": "bfloat16"},
        NORM: {"blob": _fake_blob("n"), "shape": [5], "dtype": "bfloat16"},
        CONV: {"blob": _fake_blob("c"), "shape": [2, 2, 2], "dtype": "bfloat16"},
    }
    return {"base": {mode: entries,
                     "seed_format": "reference-v1" if mode == "tensor_refs"
                     else "per-tensor-v1"}}


def _lora_spec(name=QPROJ, rank=2, alpha=4, seed=1):
    rng = np.random.default_rng(seed)
    base_shape = {QPROJ: (4, 3), VPROJ: (2, 2)}.get(name, (4, 3))
    return {"name": name, "kind": "lora",
            "A": rng.standard_normal((rank, base_shape[1])).astype(np.float32),
            "B": rng.standard_normal((base_shape[0], rank)).astype(np.float32),
            "rank": rank, "alpha": alpha}


def _additive_spec(name=NORM, seed=2):
    rng = np.random.default_rng(seed)
    return {"name": name, "kind": "additive",
            "d": rng.standard_normal((5,)).astype(np.float32)}


def _valid_pair(base=None, specs=None):
    base = base if base is not None else _base()
    specs = specs if specs is not None else [_lora_spec(), _additive_spec()]
    return construct_transformer_delta(
        base, specs, delta_id="d0001", parent="base",
        created_at="2026-10-19T00:00:00Z", self_edit_id="se_0001")


def _rebuild_blob(record, factors):
    """Re-serialise a blob for a (possibly mutated) record's targets."""
    return serialize_blob(factors, record["targets"])


def _valid_factors():
    rec, _ = _valid_pair()
    parsed = parse_blob(construct_transformer_delta(
        _base(), [_lora_spec(), _additive_spec()], delta_id="d0001",
        parent="base", created_at="2026-10-19T00:00:00Z",
        self_edit_id="se_0001")[1])
    return rec, dict(parsed.tensors)


# ---------------------------------------------------------------------------
# Round-trip + construction
# ---------------------------------------------------------------------------

def test_round_trip_lora_and_additive():
    record, blob = _valid_pair()
    validate_transformer_delta(record, blob, _base())  # must not raise
    assert record["kind"] == KIND
    assert record["blob"] == "sha256:" + hashlib.sha256(blob).hexdigest()
    assert record["bytes"] == len(blob)
    assert [t["factor_keys"] for t in record["targets"]] == [
        ["t0.A", "t0.B"], ["t1.d"]]
    parsed = parse_blob(blob)
    assert parsed.metadata["format"] == "transformer-delta-v1"
    assert set(parsed.tensors) == {"t0.A", "t0.B", "t1.d"}
    spec = _lora_spec()
    np.testing.assert_array_equal(parsed.tensors["t0.A"], spec["A"])
    np.testing.assert_array_equal(parsed.tensors["t0.B"], spec["B"])
    np.testing.assert_array_equal(
        parsed.tensors["t1.d"], _additive_spec()["d"])
    assert all(a.dtype == np.float32 for a in parsed.tensors.values())


def test_record_serialise_parse_round_trip():
    record, blob = _valid_pair()
    again = parse_record(serialize_record(record))
    assert again == record
    validate_transformer_delta(again, blob, _base())


def test_aggregate_norm_is_root_sum_square():
    # Norms 3 and 4 aggregate to exactly 5 under root-sum-square.
    base = _base()
    spec_a = {"name": NORM, "kind": "additive",
              "d": np.array([3.0, 0, 0, 0, 0], dtype=np.float32)}
    other = "model.language_model.layers.1.input_layernorm.weight"
    base["base"]["tensor_refs"][other] = {
        "blob": _fake_blob("n2"), "shape": [5], "dtype": "bfloat16"}
    spec_b = {"name": other, "kind": "additive",
              "d": np.array([4.0, 0, 0, 0, 0], dtype=np.float32)}
    record, blob = construct_transformer_delta(
        base, [spec_a, spec_b], delta_id="d0001", parent="base",
        created_at="2026-10-19T00:00:00Z", self_edit_id="se_0001")
    assert record["targets"][0]["target_norm"] == pytest.approx(3.0)
    assert record["targets"][1]["target_norm"] == pytest.approx(4.0)
    assert record["norm"] == pytest.approx(5.0)
    assert aggregate_norm([3.0, 4.0]) == pytest.approx(5.0)
    validate_transformer_delta(record, blob, base)


def test_target_norm_is_scaled_lora_norm():
    spec = _lora_spec(rank=2, alpha=4)
    record, _ = _valid_pair(specs=[spec])
    expect = float(np.linalg.norm((4 / 2) * (spec["B"] @ spec["A"])))
    assert record["targets"][0]["target_norm"] == pytest.approx(expect)
    assert record["norm"] == pytest.approx(expect)  # single target


def test_mode_independent_base_binding():
    # Same record validates against a copy-mode base carrying the same
    # per-tensor hashes (SI-8 mode independence, RFC §2.4).
    record, blob = _valid_pair()
    validate_transformer_delta(record, blob, _base(mode="tensor_blobs"))


def test_base_section_accepted_directly():
    record, blob = _valid_pair()
    validate_transformer_delta(record, blob, _base()["base"])


def test_metadata_targets_duplicated():
    record, blob = _valid_pair()
    parsed = parse_blob(blob)
    meta = parsed.metadata_targets
    assert [m["name"] for m in meta] == [t["name"] for t in record["targets"]]
    assert meta[0]["rank"] == 2 and meta[0]["alpha"] == 4
    assert "rank" not in meta[1]  # additive carries no rank/alpha


# ---------------------------------------------------------------------------
# Fail-closed: base binding
# ---------------------------------------------------------------------------

def test_unknown_target_fails():
    with pytest.raises(DeltaValidationError):
        construct_transformer_delta(
            _base(), [_lora_spec(name="model.nope.weight")],
            delta_id="d0001", parent="base",
            created_at="2026-10-19T00:00:00Z", self_edit_id="se_0001")


def test_substring_name_is_not_a_match():
    # Exact names only: a substring of a real name must NOT resolve.
    record, blob = _valid_pair()
    record = copy.deepcopy(record)
    record["targets"][0]["name"] = "layers.0.self_attn.q_proj.weight"
    with pytest.raises(DeltaValidationError):
        validate_transformer_delta(record, blob, _base())


def test_base_blob_mismatch_fails():
    record, blob = _valid_pair()
    record = copy.deepcopy(record)
    record["targets"][0]["base_blob"] = _fake_blob("attacker")
    with pytest.raises(DeltaValidationError):
        validate_transformer_delta(record, blob, _base())


def test_base_shape_lie_in_record_fails():
    record, blob = _valid_pair()
    record = copy.deepcopy(record)
    record["targets"][0]["base_shape"] = [3, 4]
    with pytest.raises(DeltaValidationError):
        validate_transformer_delta(record, blob, _base())


def test_base_dtype_mismatch_fails():
    record, blob = _valid_pair()
    record = copy.deepcopy(record)
    record["targets"][0]["base_dtype"] = "float32"
    with pytest.raises(DeltaValidationError):
        validate_transformer_delta(record, blob, _base())


# ---------------------------------------------------------------------------
# Fail-closed: shapes, dtypes, dimensionality, kind eligibility
# ---------------------------------------------------------------------------

def test_blob_factor_shape_lie_fails():
    # Blob header claims A is [2, 99] while base is [4, 3] / rank 2.
    record, factors = _valid_factors()
    factors["t0.A"] = np.zeros((2, 99), dtype=np.float32)
    blob = _rebuild_blob(record, factors)
    record = copy.deepcopy(record)
    # Keep hash/bytes honest so ONLY the shape lie is at fault.
    record["blob"] = "sha256:" + hashlib.sha256(blob).hexdigest()
    record["bytes"] = len(blob)
    with pytest.raises(DeltaValidationError):
        validate_transformer_delta(record, blob, _base())


def test_factor_dtype_not_float32_fails():
    record, factors = _valid_factors()
    factors = {k: v.astype(np.float64) for k, v in factors.items()}
    blob = st_save(factors, metadata={
        "format": "transformer-delta-v1",
        "targets": json.dumps(json.loads(
            parse_blob(_valid_pair()[1]).metadata["targets"]))})
    record = copy.deepcopy(record)
    record["blob"] = "sha256:" + hashlib.sha256(blob).hexdigest()
    record["bytes"] = len(blob)
    with pytest.raises(DeltaValidationError):
        validate_transformer_delta(record, blob, _base())


def test_lora_on_1d_fails():
    with pytest.raises(DeltaValidationError):
        construct_transformer_delta(
            _base(), [{"name": NORM, "kind": "lora",
                       "A": np.ones((1, 5), dtype=np.float32),
                       "B": np.ones((5, 1), dtype=np.float32),
                       "rank": 1, "alpha": 1}],
            delta_id="d0001", parent="base",
            created_at="2026-10-19T00:00:00Z", self_edit_id="se_0001")


def test_additive_on_2d_fails():
    with pytest.raises(DeltaValidationError):
        construct_transformer_delta(
            _base(), [{"name": QPROJ, "kind": "additive",
                       "d": np.ones((12,), dtype=np.float32)}],
            delta_id="d0001", parent="base",
            created_at="2026-10-19T00:00:00Z", self_edit_id="se_0001")


def test_non_1d_2d_base_target_fails():
    with pytest.raises(DeltaValidationError):
        construct_transformer_delta(
            _base(), [{"name": CONV, "kind": "additive",
                       "d": np.ones((8,), dtype=np.float32)}],
            delta_id="d0001", parent="base",
            created_at="2026-10-19T00:00:00Z", self_edit_id="se_0001")


def test_rank_below_one_fails():
    record, blob = _valid_pair()
    record = copy.deepcopy(record)
    record["targets"][0]["rank"] = 0
    with pytest.raises(DeltaValidationError):
        validate_transformer_delta(record, blob, _base())


def test_rank_disagreeing_with_blob_fails():
    record, blob = _valid_pair()
    record = copy.deepcopy(record)
    record["targets"][0]["rank"] = 3  # blob A is [2, 3]
    with pytest.raises(DeltaValidationError):
        validate_transformer_delta(record, blob, _base())


# ---------------------------------------------------------------------------
# Fail-closed: factor keys, metadata, norms, blob identity
# ---------------------------------------------------------------------------

def test_missing_factor_key_in_record_fails():
    record, blob = _valid_pair()
    record = copy.deepcopy(record)
    record["targets"][0]["factor_keys"] = ["t0.A"]
    with pytest.raises(DeltaValidationError):
        validate_transformer_delta(record, blob, _base())


def test_missing_factor_key_in_blob_fails():
    record, factors = _valid_factors()
    del factors["t0.B"]
    blob = _rebuild_blob(record, factors)
    record = copy.deepcopy(record)
    record["blob"] = "sha256:" + hashlib.sha256(blob).hexdigest()
    record["bytes"] = len(blob)
    with pytest.raises(DeltaValidationError):
        validate_transformer_delta(record, blob, _base())


def test_extra_factor_key_in_blob_fails():
    record, factors = _valid_factors()
    factors["t9.X"] = np.zeros((1,), dtype=np.float32)
    blob = _rebuild_blob(record, factors)
    record = copy.deepcopy(record)
    record["blob"] = "sha256:" + hashlib.sha256(blob).hexdigest()
    record["bytes"] = len(blob)
    with pytest.raises(DeltaValidationError):
        validate_transformer_delta(record, blob, _base())


def test_metadata_manifest_mismatch_is_tamper():
    record, factors = _valid_factors()
    tampered_targets = copy.deepcopy(record["targets"])
    tampered_targets[0]["name"] = VPROJ  # metadata will claim VPROJ
    blob = serialize_blob(factors, tampered_targets)
    record = copy.deepcopy(record)
    record["blob"] = "sha256:" + hashlib.sha256(blob).hexdigest()
    record["bytes"] = len(blob)
    with pytest.raises(DeltaValidationError):
        validate_transformer_delta(record, blob, _base())


def test_metadata_format_missing_fails():
    record, factors = _valid_factors()
    blob = st_save(factors, metadata={
        "targets": parse_blob(_valid_pair()[1]).metadata["targets"]})
    record = copy.deepcopy(record)
    record["blob"] = "sha256:" + hashlib.sha256(blob).hexdigest()
    record["bytes"] = len(blob)
    with pytest.raises(DeltaValidationError):
        validate_transformer_delta(record, blob, _base())


def test_target_norm_lie_fails():
    record, blob = _valid_pair()
    record = copy.deepcopy(record)
    record["targets"][0]["target_norm"] *= 2
    record["norm"] = aggregate_norm(
        [t["target_norm"] for t in record["targets"]])  # keep RSS honest
    with pytest.raises(DeltaValidationError):
        validate_transformer_delta(record, blob, _base())


def test_aggregate_norm_lie_fails():
    record, blob = _valid_pair()
    record = copy.deepcopy(record)
    record["norm"] = record["norm"] + 0.5  # not the root-sum-square
    with pytest.raises(DeltaValidationError):
        validate_transformer_delta(record, blob, _base())


def test_blob_hash_mismatch_fails():
    record, blob = _valid_pair()
    record = copy.deepcopy(record)
    record["blob"] = _fake_blob("other")
    with pytest.raises(DeltaValidationError):
        validate_transformer_delta(record, blob, _base())


def test_non_finite_factor_fails():
    record, factors = _valid_factors()
    factors["t1.d"] = factors["t1.d"].copy()
    factors["t1.d"][0] = np.inf
    blob = _rebuild_blob(record, factors)
    record = copy.deepcopy(record)
    record["blob"] = "sha256:" + hashlib.sha256(blob).hexdigest()
    record["bytes"] = len(blob)
    with pytest.raises(DeltaValidationError):
        validate_transformer_delta(record, blob, _base())


def test_duplicate_target_names_fail():
    with pytest.raises(DeltaValidationError):
        construct_transformer_delta(
            _base(), [_lora_spec(seed=1), _lora_spec(seed=3)],
            delta_id="d0001", parent="base",
            created_at="2026-10-19T00:00:00Z", self_edit_id="se_0001")


def test_empty_targets_fail():
    record, blob = _valid_pair()
    record = copy.deepcopy(record)
    record["targets"] = []
    with pytest.raises(DeltaValidationError):
        validate_transformer_delta(record, blob, _base())


def test_wrong_kind_fails():
    record, blob = _valid_pair()
    record = copy.deepcopy(record)
    record["kind"] = "delta"
    with pytest.raises(DeltaValidationError):
        validate_transformer_delta(record, blob, _base())


def test_additive_with_rank_fails():
    record, blob = _valid_pair()
    record = copy.deepcopy(record)
    record["targets"][1]["rank"] = 1
    with pytest.raises(DeltaValidationError):
        validate_transformer_delta(record, blob, _base())


def test_garbage_blob_fails():
    record, _ = _valid_pair()
    with pytest.raises(DeltaValidationError):
        validate_transformer_delta(record, b"not safetensors at all", _base())
