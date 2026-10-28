"""SI-35: branch-contribution and expressibility audit of
the miniature (RFC SI-34 §§6, 15, 16 — measurement only).

Covers the §16 criteria, in order:

1. coverage, complete and inventoried — all 30 tensors
   appear in the record: 17 weight matrices tabled with
   shapes (14 in-stream carried through every figure),
   the readout and the two embedding matrices tabled
   under their own labels, the 13 vector tensors
   inventoried with blind-set status (the derived blind
   set is exactly both layers' ``k_proj.bias``); a figure
   a tensor cannot carry is "not applicable" with the
   reason, never 0.0, never an omission;
2. populations and probe exclusion, asserted — the
   audit's input sets are exactly the named partitions
   (both signals' train partitions primary, held-out
   reported only), and a perturbed-probe-artifact bundle
   copy produces a byte-identical audit record;
3. reproduction of SI-33's attribution — the audit
   reproduces SI-33's published ranges on signal 2's
   train partition at ``d0001`` (‖s‖ 0.040–0.064;
   ‖h_mid‖ 1.97–2.02), and reconciles the base
   representative figures (0.0074 / 2.00) in the record,
   with the method difference named;
4. measurements and derivations, labelled — branch
   ratios and input norms carry population, head, and
   position-statistic on every figure; bounds recompute
   as ``0.9 * max input norm`` from the record's own
   measured inputs; downstream transfer is "unknown" for
   every tensor except the readout and the final
   down-projection;
5. readings applied as registered — exactly one of
   R1/R2 is measured; on the measured record R1 is NOT
   measured and R2 IS measured (no in-stream matrix
   satisfies R1's pair on the primary cell: the
   branch-ratio clause fails for all 14, the bound clause
   fails for exactly the two down-projections), and R3
   is NOT measured (the qualifying set is empty at both
   ``base`` and ``d0001``; the ``d0002`` cell is
   published beside them). The shortlist is empty;
6. magnitude accounting — SI-33's candidate norm and
   ‖m2‖ and the SI-25 control's stand in the record
   beside the bounds;
7. no side effects, asserted — a bundle snapshot is
   byte-identical before/after an audit run; spies prove
   the audit invokes no training entry point and opens
   no gate / probe evaluation; the authorship-target
   axis still serves exactly two values and every policy
   default is untouched;
8. determinism — a repeated audit reproduces the record
   byte-identically;
9. untouched scope — ``si squash`` still fails closed on
   the landed stack; the audit fails closed on a bundle
   without the landed heads rather than fabricate them.

The audit itself trains nothing and authors nothing: the
landed bundles below are built with the existing SI-33
decisive-run driver shape purely as *fixtures* — the
stack the audit measures — exactly as SI-33's own tests
build theirs.
"""
import hashlib
import json
import os
import shutil

import numpy as np
import pytest

import si.tiny_audit as au
import si.tiny_training as tt
from si.bundle import DEFAULT_POLICY, init_bundle, load_bundle, load_policy
from si.gate import Gate
from si.run import run_cycles
from si.squash import SquashError, squash_bundle
from si.tiny_seed import (
    build_tiny_weights,
    capability_inputs,
    identity_inputs,
    seed_tiny_bundle,
    tiny_tensor_shapes,
)
from si.tiny_transformer import TinyTransformerMind, forward

DOWN_NAME = "model.language_model.layers.1.mlp.down_proj.weight"


def _seeded(tmp_path, name="tiny35"):
    bundle_dir = init_bundle(str(tmp_path / (name + ".si")), name=name,
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    return bundle_dir


def _policy(bundle_dir):
    policy = load_policy(bundle_dir)
    policy["allowed_target_patterns"] = ["*"]
    return policy


def _landed(tmp_path, name="tiny35"):
    """The SI-33 decisive stack as a fixture: cycle 1
    (signal 1, defaults) then cycle 2 (signal 2,
    base-anchored, composition "none", authorship target
    "final-mlp-down") — head d0002, depth 2."""
    bundle_dir = _seeded(tmp_path, name)
    policy = _policy(bundle_dir)
    mind = TinyTransformerMind(bundle_dir, load_bundle(bundle_dir))
    s1 = run_cycles(bundle_dir, mind, 1, gate=Gate(policy))
    assert s1["admitted"] == 1 and s1["head"] == "d0001"
    mind.set_training_signal(tt.TRAINING_SIGNAL_ID_2)
    mind.set_target_anchoring(tt.BASE_ANCHORED)
    mind.set_authorship_target("final-mlp-down")
    s2 = run_cycles(bundle_dir, mind, 1, gate=Gate(policy))
    assert s2["admitted"] == 1 and s2["head"] == "d0002"
    return bundle_dir, policy


def _snapshot(bundle_dir):
    out = {}
    for root, _dirs, files in os.walk(bundle_dir):
        for fn in sorted(files):
            path = os.path.join(root, fn)
            with open(path, "rb") as f:
                out[os.path.relpath(path, bundle_dir)] = \
                    hashlib.sha256(f.read()).hexdigest()
    return out


@pytest.fixture(scope="module")
def landed_record(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("si35landed")
    bundle_dir, policy = _landed(tmp, "tiny35mod")
    manifest = load_bundle(bundle_dir)
    return bundle_dir, policy, manifest, au.run_audit(
        bundle_dir, manifest, policy)


# -- §16.1: coverage, complete and inventoried --------------------------------

def test_inventory_is_complete(landed_record):
    _bd, _pol, _man, rec = landed_record
    inv = rec["inventory"]
    shapes = tiny_tensor_shapes()
    assert len(shapes) == 30
    assert inv["tensor_count"] == 30
    assert inv["weight_matrix_count"] == 17
    assert inv["in_stream_count"] == 14
    assert inv["vector_tensor_count"] == 13
    mat_names = {m["name"] for m in inv["weight_matrices"]}
    vec_names = {v["name"] for v in inv["vector_tensors"]}
    assert mat_names | vec_names == set(shapes)
    assert not (mat_names & vec_names)
    for m in inv["weight_matrices"]:
        assert m["shape"] == shapes[m["name"]]
    by_name = {m["name"]: m for m in inv["weight_matrices"]}
    assert by_name["lm_head.weight"]["category"].startswith("readout")
    for emb in ("model.language_model.embed_tokens.weight",
                "model.language_model.embed_positions.weight"):
        assert by_name[emb]["category"].startswith("embedding")
    in_stream = {n for n, m in by_name.items()
                 if m["category"] == "in-stream"}
    assert in_stream == set(au.IN_STREAM_MATRICES)
    assert len(in_stream) == 14
    # The 13 vector tensors, with blind-set status: the
    # derived blind set is exactly both layers' k_proj.bias.
    assert inv["derived_blind_set"] == [
        "model.language_model.layers.0.self_attn.k_proj.bias",
        "model.language_model.layers.1.self_attn.k_proj.bias"]
    blind_vecs = {v["name"] for v in inv["vector_tensors"] if v["blind"]}
    assert blind_vecs == set(inv["derived_blind_set"])
    # No weight matrix is in the derived blind set (the
    # blind set is biases only — a derived fact, §15 item 6).
    assert all(not (n in blind_vecs) for n in mat_names)
    # Every in-stream matrix is carried through every
    # figure: branch measurements exist via its branch,
    # input records, bounds, and the backward inventory.
    assert {r["matrix"] for r in rec["backward_inventory"]} == in_stream
    for r in rec["backward_inventory"]:
        assert r["blind"] is False
        rows, cols = r["shape"]
        assert r["factor_entries_rank8"] == 8 * (rows + cols)
        assert r["chain_count"] == len(r["chain_elements"]) > 0
    bound_mats = {b["matrix"] for b in rec["expressibility_bounds"]
                  if b.get("population_role") == "primary"}
    assert bound_mats == in_stream
    input_mats = {r["matrix"] for r in rec["input_activations"]}
    assert input_mats == mat_names
    branch_rows = {r["branch"] for r in rec["branch_measurements"]}
    assert branch_rows == set(au.BRANCHES) and len(branch_rows) == 5
    # Figures a tensor cannot carry are "not applicable"
    # with the reason — never 0.0, never an omission.
    emb_rows = [r for r in rec["input_activations"]
                if r["matrix"].endswith("embed_tokens.weight")]
    assert emb_rows and all(
        r["status"] == "not applicable" and "index" in r["reason"]
        for r in emb_rows)
    lm_row = [b for b in rec["expressibility_bounds"]
              if b["matrix"] == "lm_head.weight"]
    assert len(lm_row) == 1 and lm_row[0]["kind"] == "not applicable"


# -- the audit forward is the library forward ----------------------------------

def test_forward_audit_matches_library_forward():
    weights = build_tiny_weights()
    seqs = []
    for sid in (tt.TRAINING_SIGNAL_ID, tt.TRAINING_SIGNAL_ID_2):
        train, held, _ = tt.signal_sequences(sid)
        seqs.extend(train[:4] + held[:2])
    for seq in seqs:
        rec = au.forward_audit(seq, weights)
        assert np.array_equal(rec["logits"], forward(seq, weights))
    # The final layer's exposed quantities agree with the
    # SI-33 training replica's fixed inputs.
    seq = seqs[0]
    rec = au.forward_audit(seq, weights)
    h_mid, s = tt.forward_to_final_down(seq, weights)
    assert np.array_equal(
        rec["branches"]["layer-1-mlp"]["residual"], h_mid)
    assert np.array_equal(rec["inputs"][DOWN_NAME], s)


# -- §16.2: populations and probe exclusion -------------------------------------

def test_populations_are_exactly_the_named_partitions(landed_record):
    _bd, _pol, _man, rec = landed_record
    assert [(p["signal_id"], p["partition"], p["role"], p["size"])
            for p in rec["populations"]] == [
        (tt.TRAINING_SIGNAL_ID, "train", "primary", 32),
        (tt.TRAINING_SIGNAL_ID_2, "train", "primary", 32),
        (tt.TRAINING_SIGNAL_ID, "held_out", "reported-only", 16),
        (tt.TRAINING_SIGNAL_ID_2, "held_out", "reported-only", 16)]
    probe_seqs = {tuple(s) for s in identity_inputs()} \
        | {tuple(s) for s in capability_inputs()}
    for sid in (tt.TRAINING_SIGNAL_ID, tt.TRAINING_SIGNAL_ID_2):
        train, held, _ = tt.signal_sequences(sid)
        assert {tuple(s) for s in train}.isdisjoint(probe_seqs)
        assert {tuple(s) for s in held}.isdisjoint(probe_seqs)
        assert {tuple(s) for s in train}.isdisjoint(
            {tuple(s) for s in held})
    # Bounds are derived from primary populations only;
    # held-out figures are reported only and enter none.
    assert {b["population"] for b in rec["expressibility_bounds"]
            if "population" in b} == {
        f"{tt.TRAINING_SIGNAL_ID}:train",
        f"{tt.TRAINING_SIGNAL_ID_2}:train"}
    # Readings rest on the primary cell only.
    assert rec["readings"]["primary_cell"] == {
        "population": f"{tt.TRAINING_SIGNAL_ID_2}:train",
        "head": "d0001"}


def test_perturbed_probe_artifact_byte_identical(tmp_path):
    bundle_dir, policy = _landed(tmp_path, "tiny35pert")
    manifest = load_bundle(bundle_dir)
    before = au.audit_record_bytes(au.run_audit(bundle_dir, manifest,
                                                policy))
    copy_dir = str(tmp_path / "perturbed.si")
    shutil.copytree(bundle_dir, copy_dir)
    probe_path = os.path.join(copy_dir, "probe_suite.json")
    with open(probe_path, "rb") as f:
        artifact = json.loads(f.read().decode("utf-8"))
    for entry in artifact["identity_suite"] + artifact["capability_suite"]:
        entry["input"] = [0, 1, 2]
        entry["expected_logits"] = [0.0] * 32
    with open(probe_path, "wb") as f:
        f.write(json.dumps(artifact).encode("utf-8"))
    after = au.audit_record_bytes(au.run_audit(copy_dir, manifest,
                                               policy))
    assert after == before


# -- §16.3: reproduction of SI-33's attribution ----------------------------------

def test_reproduction_of_si33_attribution(landed_record):
    _bd, _pol, _man, rec = landed_record
    rep = rec["reproduction"]
    assert rep["ranges_reproduce"] is True
    s_min, s_max = rep["s_norm_range_measured"]
    h_min, h_max = rep["h_mid_norm_range_measured"]
    assert 0.039 <= s_min <= 0.041 and 0.063 <= s_max <= 0.065
    assert 1.96 <= h_min <= 1.98 and 2.01 <= h_max <= 2.03
    yard = rep["base_yardstick_row"]
    assert yard["branch"] == "layer-1-mlp" and yard["head"] == "base"
    # The published residual (2.00) matches the population
    # mean to SI-33's printed precision; the published
    # branch output (0.0074) is reconciled in print with
    # the method difference named (single unnamed sequence
    # vs the audit's named population).
    assert yard["residual_stream_norm_mean"] == pytest.approx(
        2.00, abs=0.01)
    assert "representative sequence it does not name" in \
        yard["reconciliation"]
    assert "method difference" in yard["reconciliation"]


# -- §16.4/§16.5: labelled figures and the readings, as registered ---------------

def test_bounds_recompute_from_measured_inputs(landed_record):
    _bd, _pol, _man, rec = landed_record
    inputs = {(r["matrix"], r["population"], r["head"]): r
              for r in rec["input_activations"]}
    for b in rec["expressibility_bounds"]:
        if b.get("kind") != "derivation" or "bound" not in b \
                or "population" not in b:
            continue
        src = inputs[(b["matrix"], b["population"], b["head"])]
        assert b["max_input_norm"] == \
            src["last_position"]["max"]
        assert b["bound"] == pytest.approx(
            0.9 * src["last_position"]["max"], rel=1e-12)
        assert b["norm_budget"] == 0.9
    # Downstream transfer: unknown everywhere except the
    # readout and the final down-projection (§15 item 5).
    inv = {r["matrix"]: r["downstream_transfer"]
           for r in rec["backward_inventory"]}
    for name, transfer in inv.items():
        if name == DOWN_NAME:
            assert transfer.startswith("measured")
        else:
            assert transfer == "unknown"
    # Every branch / input figure carries population,
    # head, and its position statistic on the figure.
    for row in rec["branch_measurements"]:
        assert row["kind"] == "measurement"
        assert row["population"] and row["head"] in au.AUDIT_HEADS
        assert "last_position" in row and "all_positions" in row
        assert "no reading rests" in row["all_positions"]["label"]


def test_readings_applied_as_registered(landed_record):
    _bd, _pol, _man, rec = landed_record
    readings = rec["readings"]
    assert readings["exactly_one_of_R1_R2"] is True
    # As measured on the landed stack: R1 NOT measured,
    # R2 measured, R3 NOT measured; the shortlist is empty.
    assert readings["R1"]["measured"] is False
    assert readings["R2"]["measured"] is True
    assert readings["R3"]["measured"] is False
    assert readings["R3"]["qualifying_at_base"] == []
    assert readings["R3"]["qualifying_at_d0001"] == []
    assert readings["R3"]["qualifying_at_d0002"] == []
    assert rec["shortlist"] == []
    table = readings["qualification_table_primary"]
    assert len(table) == 14
    assert all(q["qualifies"] is False for q in table)
    # The deciding facts, published in full: the
    # branch-ratio clause fails for every matrix (the
    # largest branch ratio on the primary cell is the
    # layer-1 attention branch's, below the registered
    # 0.04), and the bound clause fails for exactly the
    # two down-projections.
    assert all(q["branch_ratio_clause"] is False for q in table)
    assert max(q["branch_ratio_max_per_example"] for q in table) == \
        pytest.approx(0.0341, abs=0.001)
    bound_fails = {q["matrix"] for q in table
                   if q["bound_clause"] is False}
    assert bound_fails == {
        "model.language_model.layers.0.mlp.down_proj.weight",
        DOWN_NAME}
    for q in table:
        assert q["branch_ratio_threshold"] == 0.04
        assert q["bound_threshold"] == pytest.approx(
            0.15 * np.sqrt(2.0))
    # The yardstick row: the final down-projection's
    # audit bound sits beside SI-33's cited ≈ 0.058.
    down_q = [q for q in table if q["matrix"] == DOWN_NAME][0]
    assert down_q["bound"] == pytest.approx(0.058, abs=0.002)


def test_magnitude_accounting_present(landed_record):
    _bd, _pol, _man, rec = landed_record
    acct = rec["magnitude_accounting"]
    assert acct["si33_candidate"] == {
        "norm": 0.30801411790830574, "m2": 0.010469039280455053}
    assert acct["si25_control"] == {
        "norm": 0.2515524481203623, "m2": 0.7816722548913912}
    assert rec["thresholds"]["norm_budget"] == 0.9
    assert rec["thresholds"]["r1_branch_ratio_min"] == 0.04
    assert rec["thresholds"]["r1_bound_min"] == pytest.approx(
        0.15 * np.sqrt(2.0))


# -- §16.7/§16.8: no side effects; determinism -------------------------------------

def test_no_side_effects_and_no_training_or_gate_contact(
        tmp_path, monkeypatch):
    bundle_dir, policy = _landed(tmp_path, "tiny35fx")
    manifest = load_bundle(bundle_dir)
    before = _snapshot(bundle_dir)

    def _boom(*a, **k):
        raise AssertionError("the audit must not call this")

    monkeypatch.setattr(tt, "train_candidate", _boom)
    import si.gate as gate_mod
    monkeypatch.setattr(gate_mod.Gate, "evaluate", _boom)
    import si.probes as probes_mod
    monkeypatch.setattr(probes_mod, "ProbeContext", _boom)
    rec = au.run_audit(bundle_dir, manifest, policy)
    assert rec["shortlist"] == []
    assert _snapshot(bundle_dir) == before
    assert load_bundle(bundle_dir) == manifest


def test_audit_determinism(landed_record):
    bundle_dir, policy, manifest, rec = landed_record
    again = au.run_audit(bundle_dir, manifest, policy)
    assert au.audit_record_bytes(again) == au.audit_record_bytes(rec)


# -- §16.9: fail-closed heads and untouched scope -----------------------------------

def test_audit_fails_closed_without_landed_heads(tmp_path):
    bundle_dir = _seeded(tmp_path, "tiny35base")
    manifest = load_bundle(bundle_dir)
    with pytest.raises(ValueError, match="d0001"):
        au.run_audit(bundle_dir, manifest, _policy(bundle_dir))
    weights = build_tiny_weights()
    with pytest.raises(ValueError, match="missing"):
        au.run_audit_on_weight_sets({"base": weights})


def test_untouched_scope_on_landed_stack(landed_record):
    bundle_dir, _pol, _man, _rec = landed_record
    assert DEFAULT_POLICY["max_stack_depth"] == 8
    assert DEFAULT_POLICY["identity_tol"] == 0.25
    assert DEFAULT_POLICY["capability_eps"] == 0.25
    assert DEFAULT_POLICY["rank_cap"] == 8
    assert DEFAULT_POLICY["max_targets_per_delta"] == 1
    assert DEFAULT_POLICY["allow_additive"] is False
    assert tt.AUTHORSHIP_TARGETS == ("lm-head", "final-mlp-down")
    assert tt.RANK == 8 and tt.ALPHA == 8
    assert tt.CORRECTION_SCALE == 0.15
    # SI-38 note (this assertion superseded): SI-12's blanket
    # SquashError refusal was replaced by RFC SI-37's designed
    # pipeline. On the landed stack the pipeline runs and,
    # under the bundle's OWN policy (empty allowlist), the
    # gate REJECTS the composite — nothing lands.
    res = squash_bundle(bundle_dir, keep=0)
    assert res["squashed"] is False and res["verdict"] == "REJECT"
    manifest = load_bundle(bundle_dir)
    assert len(manifest["deltas"]) == 2
    assert all("superseded_by" not in d for d in manifest["deltas"])
