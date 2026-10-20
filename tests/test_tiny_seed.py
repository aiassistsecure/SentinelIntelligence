"""SI-14: miniature seed + probe artifacts (RFC SI-13 §3, §5, §8).

Covers the SI-14 acceptance criteria:

- tensor/parameter inventory matches §3.1 + the explicitly stated
  positional-table variant (30 tensors, 23,456 elements);
- deterministic re-seeding produces byte-identical base blobs and
  an identical probe artifact;
- seed-time assertions run and fail closed — including a
  constructed seed that fails the echo floor and is refused with
  the bundle left untouched;
- stored-bytes accounting: the base is stored once (content-
  addressed, identical tensors deduplicated), probes add only the
  §5.5 artifact file;
- the probe artifact is content-hashed into the manifest and
  verified on load (tamper fails closed);
- honesty: the manifest source label says what the miniature is;
- gate behaviour is unchanged: a valid transformer delta against
  the miniature bundle still gets checks 3/4 "not evaluated" and
  ESCALATES — nothing consumes the probes in SI-14.
"""
import json
import os

import numpy as np
import pytest

from si.bundle import (
    TamperError,
    init_bundle,
    load_bundle,
    load_base_tensor,
    load_policy,
    sha256_hex,
)
from si.journal import open_journal
from si.tiny_seed import (
    TINY_SOURCE_LABEL,
    build_tiny_weights,
    capability_inputs,
    identity_inputs,
    seed_tiny_bundle,
    tiny_forward,
    tiny_tensor_shapes,
)

QPROJ = "model.language_model.layers.0.self_attn.q_proj.weight"


def _init(tmp_path, name):
    return init_bundle(str(tmp_path / (name + ".si")), name=name,
                       journal_backend="embedded")


def _seeded(tmp_path, name="tiny"):
    bundle_dir = _init(tmp_path, name)
    res = seed_tiny_bundle(bundle_dir)
    return bundle_dir, res


def _blob_path(bundle_dir, blob_ref):
    return os.path.join(bundle_dir, "blobs", "sha256",
                        blob_ref.split(":", 1)[1] + ".safetensors")


def _artifact(bundle_dir):
    with open(os.path.join(bundle_dir, "probe_suite.json"), "rb") as f:
        return json.loads(f.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Inventory (RFC §3.1 + positional-table variant, stated in si/tiny_seed.py)
# ---------------------------------------------------------------------------

def test_inventory_matches_rfc_3_1_plus_positional_table(tmp_path):
    bundle_dir, res = _seeded(tmp_path)
    manifest = load_bundle(bundle_dir)
    base = manifest["base"]

    shapes = tiny_tensor_shapes()
    # §3.1's 29 tensors / 22,944 elements + positional table (+1 / +512).
    assert len(shapes) == 30
    assert sum(int(np.prod(s)) for s in shapes.values()) == 23_456
    assert shapes["model.language_model.embed_positions.weight"] == [16, 32]

    assert base["tensor_count"] == 30
    assert base["total_elements"] == 23_456
    assert set(base["tensor_blobs"]) == set(shapes)
    for name, ref in base["tensor_blobs"].items():
        assert ref["shape"] == shapes[name]
        assert ref["dtype"] == "float32"
        arr = load_base_tensor(bundle_dir, manifest, name)
        assert list(arr.shape) == shapes[name]
        assert arr.dtype == np.float32
    assert res["tensors"] == 30 and res["elements"] == 23_456

    arch = base["architecture"]
    assert arch["vocab_size"] == 32 and arch["d_model"] == 32
    assert arch["layers"] == 2 and arch["heads"] == 4
    assert arch["head_dim"] == 8 and arch["d_ff"] == 64
    assert arch["max_seq_len"] == 16
    assert arch["positional_embedding"] is True


def test_honesty_label_and_kind(tmp_path):
    bundle_dir, _ = _seeded(tmp_path)
    base = load_bundle(bundle_dir)["base"]
    assert base["source"] == TINY_SOURCE_LABEL
    assert "not a trained model" in base["source"]
    assert "constructed miniature" in base["source"]
    assert base["kind"] == "tiny-transformer-miniature-v1"
    art = _artifact(bundle_dir)
    assert "not a trained model" in art["honesty"]


# ---------------------------------------------------------------------------
# Determinism: byte-identical blobs + identical probe artifact on re-seed
# ---------------------------------------------------------------------------

def test_deterministic_reseed_byte_identical(tmp_path):
    b1, _ = _seeded(tmp_path, "tiny-a")
    b2, _ = _seeded(tmp_path, "tiny-b")
    m1, m2 = load_bundle(b1), load_bundle(b2)

    assert m1["base"]["tensor_blobs"] == m2["base"]["tensor_blobs"]
    for ref in m1["base"]["tensor_blobs"].values():
        with open(_blob_path(b1, ref["blob"]), "rb") as f:
            bytes1 = f.read()
        with open(_blob_path(b2, ref["blob"]), "rb") as f:
            bytes2 = f.read()
        assert bytes1 == bytes2

    with open(os.path.join(b1, "probe_suite.json"), "rb") as f:
        art1 = f.read()
    with open(os.path.join(b2, "probe_suite.json"), "rb") as f:
        art2 = f.read()
    assert art1 == art2
    assert m1["base"]["probe_artifact"] == m2["base"]["probe_artifact"]


def test_reseed_same_bundle_is_idempotent_noop(tmp_path):
    bundle_dir, _ = _seeded(tmp_path)
    res = seed_tiny_bundle(bundle_dir)
    assert res["already_seeded"] is True
    assert load_bundle(bundle_dir)["head"] == "base"


# ---------------------------------------------------------------------------
# Seed-time assertions, fail closed (RFC §3.3)
# ---------------------------------------------------------------------------

def test_seed_failing_echo_floor_is_refused(tmp_path):
    """readout_shift=1 constructs a readout that answers last+1:
    echo accuracy is 0/16 by construction, so seeding must refuse
    BEFORE writing anything — the bundle keeps its demo base."""
    bundle_dir = _init(tmp_path, "tiny-bad")
    with pytest.raises(ValueError, match="echo accuracy"):
        seed_tiny_bundle(bundle_dir, readout_shift=1)
    manifest = load_bundle(bundle_dir)  # still verifies, untouched
    assert manifest["base"]["source"].startswith("demo:")
    assert not os.path.exists(os.path.join(bundle_dir, "probe_suite.json"))


def test_seed_refuses_bundle_with_history(tmp_path):
    from si.loop import run_cycle
    from si.minds import DemoMind

    bundle_dir = _init(tmp_path, "tiny-hist")
    res = run_cycle(bundle_dir, DemoMind(bundle_path=bundle_dir))
    assert res["verdict"] == "ADMIT"
    with pytest.raises(ValueError, match="existing history"):
        seed_tiny_bundle(bundle_dir)


def test_construction_and_forward_are_deterministic():
    w1 = build_tiny_weights()
    w2 = build_tiny_weights()
    for name in w1:
        assert np.array_equal(w1[name], w2[name])
        assert np.all(np.isfinite(w1[name]))
    for seq in identity_inputs() + capability_inputs():
        assert np.array_equal(tiny_forward(seq, w1), tiny_forward(seq, w2))


# ---------------------------------------------------------------------------
# Probe artifact contents (RFC §5.2/§5.3/§5.4/§5.5)
# ---------------------------------------------------------------------------

def test_probe_artifact_contents(tmp_path):
    bundle_dir, res = _seeded(tmp_path)
    art = _artifact(bundle_dir)

    assert [e["id"] for e in art["identity_suite"]] == \
        [f"tiny-id-{i}" for i in range(8)]
    assert [e["id"] for e in art["capability_suite"]] == \
        [f"tiny-cap-{i}" for i in range(16)]
    for e in art["identity_suite"] + art["capability_suite"]:
        assert len(e["expected_logits"]) == 32
        assert e["baseline_error"] == 0.0
        assert all(np.isfinite(e["expected_logits"]))
    # Identity inputs are the §5.2 pinned classes.
    id_inputs = [e["input"] for e in art["identity_suite"]]
    assert len(id_inputs[0]) == 1
    assert len(id_inputs[1]) == 16 and len(set(id_inputs[1])) == 1
    assert len(id_inputs[2]) == 16 and len(set(id_inputs[2])) == 16
    assert id_inputs[3] == list(range(10))
    assert [len(x) for x in id_inputs[4:]] == [4, 8, 12, 16]
    # Capability: echo targets cover at least half the vocabulary.
    targets = [e["target_token"] for e in art["capability_suite"]]
    assert len(set(targets)) >= 16
    for e in art["capability_suite"]:
        assert e["target_token"] == e["input"][-1]

    # Measured on THIS seed (never invented): the base passes its
    # claimed floor, and the recorded tolerances are the §5.4
    # derivation from the recorded margins.
    assert art["base_echo_accuracy"] >= 15 / 16
    assert art["base_echo_accuracy"] == res["echo_accuracy"]
    assert art["echo_floor"] == pytest.approx(
        art["base_echo_accuracy"] - 1 / 16)
    id_min = min(e["margin"] for e in art["identity_suite"])
    cap_min = min(e["margin"] for e in art["capability_suite"])
    assert art["margins"]["identity_min"] == id_min
    assert art["margins"]["capability_min"] == cap_min
    assert art["tolerances"]["identity_tol"] == min(0.25, 0.4 * id_min)
    assert art["tolerances"]["capability_eps"] == min(0.25, 0.4 * cap_min)

    # Binding (§6.2 fields recorded now; consumption is SI-16's):
    # the artifact's tensor-map hash matches the manifest's map.
    from si.bundle import canonical_json
    manifest = load_bundle(bundle_dir)
    expect = "sha256:" + sha256_hex(
        canonical_json(manifest["base"]["tensor_blobs"]))
    assert art["base_binding"]["tensor_map_sha256"] == expect
    assert art["base_binding"]["tensor_count"] == 30
    assert art["base_binding"]["total_elements"] == 23_456
    assert art["architecture"] == manifest["base"]["architecture"]

    # Per-bundle policy carries the recorded tolerances.
    policy = load_policy(bundle_dir)
    assert policy["identity_tol"] == art["tolerances"]["identity_tol"]
    assert policy["capability_eps"] == art["tolerances"]["capability_eps"]


def test_probe_artifact_hash_verified_on_load(tmp_path):
    bundle_dir, _ = _seeded(tmp_path)
    manifest = load_bundle(bundle_dir)
    rec = manifest["base"]["probe_artifact"]
    assert rec["file"] == "probe_suite.json"
    with open(os.path.join(bundle_dir, "probe_suite.json"), "rb") as f:
        raw = f.read()
    assert rec["sha256"] == "sha256:" + sha256_hex(raw)
    assert rec["bytes"] == len(raw)

    # Tamper with the suite: load must fail closed.
    with open(os.path.join(bundle_dir, "probe_suite.json"), "wb") as f:
        f.write(raw + b" ")
    with pytest.raises(TamperError, match="probe_suite.json"):
        load_bundle(bundle_dir)


def test_probe_provenance_journaled(tmp_path):
    bundle_dir, _ = _seeded(tmp_path)
    manifest = load_bundle(bundle_dir)
    journal = open_journal(bundle_dir, manifest)
    try:
        seeded = journal.backend.find(
            "provenance", filter={"child": "base",
                                  "relation": "seeded_from"})
        probes = journal.backend.find(
            "provenance", filter={"child": "probe_suite",
                                  "relation": "probe_suite_recorded"})
    finally:
        journal.close()
    assert seeded and "NOT trained" in seeded[0]["note"]
    assert probes and "NOT consumed" in probes[0]["note"]
    assert "echo accuracy" in probes[0]["note"]


# ---------------------------------------------------------------------------
# Stored-bytes accounting: base stored once, probes add only the artifact
# ---------------------------------------------------------------------------

def test_stored_bytes_accounting(tmp_path):
    bundle_dir, res = _seeded(tmp_path)
    manifest = load_bundle(bundle_dir)
    base = manifest["base"]

    unique = {ref["blob"]: ref["bytes"]
              for ref in base["tensor_blobs"].values()}
    # The base is stored exactly once: manifest bytes == sum of the
    # UNIQUE blob sizes, and every unique blob exists exactly once.
    assert base["bytes"] == sum(unique.values()) == res["bytes"]
    for blob_ref in unique:
        assert os.path.isfile(_blob_path(bundle_dir, blob_ref))
    # Identical tensors (5 all-ones norm weights, 8 zero biases)
    # deduplicate onto shared blobs — never duplicated bytes.
    assert len(unique) < base["tensor_count"]
    assert res["deduplicated_bytes"] > 0
    # Probes add only the §5.5 artifact at the bundle root.
    assert os.path.isfile(os.path.join(bundle_dir, "probe_suite.json"))
    assert base["probe_artifact"]["bytes"] == os.path.getsize(
        os.path.join(bundle_dir, "probe_suite.json"))


# ---------------------------------------------------------------------------
# Gate behaviour unchanged: checks 3/4 still "not evaluated" (SI-14 contract)
# ---------------------------------------------------------------------------

class _FakeCtx:
    """Same discipline as tests/test_gate_transformer.py: the probe
    methods RAISE — a transformer gate evaluation that touches them
    in SI-14 is faking a check."""

    def __init__(self, manifest):
        self._manifest = manifest

    def base_manifest(self):
        return self._manifest

    def stack_depth(self):
        return 0

    def identity_probes(self):
        raise AssertionError("SI-14 gate must not run identity probes")

    def capability_suite(self):
        raise AssertionError("SI-14 gate must not run capability probes")

    def capability_baseline(self):
        raise AssertionError("SI-14 gate must not read a baseline")

    def forward_candidate(self, x, candidate):
        raise AssertionError("SI-14 gate must not run a forward pass")


def test_miniature_transformer_delta_still_escalates(tmp_path):
    from si.bundle import DEFAULT_POLICY
    from si.gate import Gate, Intent
    from si.transformer_delta import KIND, construct_transformer_delta

    bundle_dir, _ = _seeded(tmp_path)
    manifest = load_bundle(bundle_dir)

    rng = np.random.default_rng(1)
    spec = {"name": QPROJ, "kind": "lora",
            "A": (rng.standard_normal((2, 32)) * 0.01).astype(np.float32),
            "B": (rng.standard_normal((32, 2)) * 0.01).astype(np.float32),
            "rank": 2, "alpha": 4}
    record, blob = construct_transformer_delta(
        manifest, [spec], delta_id="d0001", parent="base",
        created_at="2026-10-20T00:00:00Z", self_edit_id="se_0001")
    candidate = {"record": record, "blob_bytes": blob}

    policy = dict(DEFAULT_POLICY)
    policy["allowed_target_patterns"] = [QPROJ]
    policy["identity_tol"] = load_policy(bundle_dir)["identity_tol"]
    policy["capability_eps"] = load_policy(bundle_dir)["capability_eps"]
    intent = Intent(kind=KIND, reason="probe the miniature gate contract",
                    parents=["base"], source_seqs=[1], targets=[QPROJ])
    report = Gate(policy).evaluate(intent, candidate, _FakeCtx(manifest))

    assert report.verdict == "ESCALATE"  # never ADMIT in SI-14
    checks = {c.name: c for c in report.checks}
    for name in ("identity_continuity", "capability_floor"):
        assert checks[name].passed is False
        assert checks[name].measured is None
        assert "not evaluated" in checks[name].detail


def test_cli_seed_tiny(tmp_path):
    from si.cli import main
    bundle_dir = _init(tmp_path, "tiny-cli")
    assert main(["seed-tiny", bundle_dir]) == 0
    base = load_bundle(bundle_dir)["base"]
    assert base["kind"] == "tiny-transformer-miniature-v1"
