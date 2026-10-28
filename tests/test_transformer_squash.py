"""SI-38: transformer squash — the implementation of RFC
SI-37's design (the §3.8 composite pricing discipline, the §4
gate context, the §5 tolerance verify) for the landed
mixed-target stack.

The landed stack under test is SI-33's, built by the SI-33
driver shape (cycle 1: signal 1, default anchoring /
composition / target; cycle 2: signal 2, base-anchored,
``"none"``, ``"final-mlp-down"``) — head ``d0002``, both
members ADMITted by the unchanged gate. The squash under
test is the full-segment squash of RFC SI-37 §5.7's
counterfactual subject (S = [``d0001``, ``d0002``], K = []),
run at ``keep=0`` under RFC SI-37 §6's operator-direction
trigger arm.

Measured on this stack (this suite asserts the values; the
as-built doc publishes them with their labels):

- the §5 verify is EXACT, not merely within tolerance: the
  weight-space max-abs difference is 0.0 on both named
  tensors and the behavioural statistic (max over all
  positions, four allowed populations) is 0.0 — the exact
  carrying (verbatim factors, scale 1) folds and forwards
  bit-identically on this miniature. RFC SI-37 §5.5's
  derivation expected rounding scale (~1e-5 or below); the
  measurement is stronger, and is reported as measured;
- under the driver policy the stack landed under
  (``allowed_target_patterns`` opened to ``["*"]`` by the
  driver, every other key the bundle's own), the gate
  ADMITs the composite: aggregate norm
  0.38776061454106125 (RFC SI-37 §3.2's derivation,
  reproduced by measurement), identity measured-pass at
  0.17195402272045612 — ``d0002``'s own cumulative drift,
  exactly, because the post-squash stack computes the same
  function — capability regression 0.0, echo 16/16;
- under the bundle's OWN policy (empty allowlist, the
  fail-closed default), the same composite is REJECTED on
  the allowlist line alone (RFC SI-37 §11 item 2(iv)'s
  policy question, met and recorded): probes measured-pass,
  nothing lands.
"""
import copy
import json
import os
import shutil

import numpy as np
import pytest

import si.tiny_training as tt
from si.bundle import (
    _active_chain,
    canonical_json,
    init_bundle,
    load_bundle,
    load_policy,
)
from si.gate import Gate, Intent
from si.journal import open_journal
from si.loop import rollback
from si.probes import ProbeContext
from si.run import run_cycles
from si.squash import SquashError, squash_bundle
from si.tiny_seed import seed_tiny_bundle
from si.tiny_transformer import TinyTransformerMind, assemble_weights
from si.transformer_delta import KIND, parse_blob
from si.transformer_squash import (
    TransformerSquashContext,
    build_attestation,
    build_composite,
    member_axis_records,
    post_squash_view,
    verify_transformer_squash,
)

LM_HEAD = "lm_head.weight"
DOWN_PROJ = "model.language_model.layers.1.mlp.down_proj.weight"

D1_BLOB = "sha256:2761e8ffc139ce3e107c11da035e8a73c20792945a494f4e493dccf418b6ca9e"
D1_NORM = 0.23555380990005192
D2_BLOB = "sha256:d7af7f1feaa5b0451d441a4fce8842b471fc7c4b3e70e6237f1115eba877e625"
D2_NORM = 0.30801411790830574
D2_DRIFT = 0.17195402272045612
#: RFC SI-37 §3.2's derivation, reproduced by measurement:
#: sqrt(0.23555380990005192**2 + 0.30801411790830574**2).
AGGREGATE_NORM = 0.38776061454106125
SEGMENT_BYTES = 2360 + 3416


def _seeded(tmp_path, name="tiny38"):
    bundle_dir = init_bundle(str(tmp_path / (name + ".si")), name=name,
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    return bundle_dir


def _policy(bundle_dir, **over):
    policy = load_policy(bundle_dir)
    policy["allowed_target_patterns"] = ["*"]
    policy.update(over)
    return policy


def _landed_stack(tmp_path, name="tiny38"):
    """SI-33's decisive driver shape: the landed d0001+d0002
    mixed-target stack, both cycles ADMITted."""
    bundle_dir = _seeded(tmp_path, name)
    policy = _policy(bundle_dir)
    mind = TinyTransformerMind(bundle_dir, load_bundle(bundle_dir))
    sum1 = run_cycles(bundle_dir, mind, 1, gate=Gate(policy))
    assert sum1["admitted"] == 1 and sum1["head"] == "d0001"
    mind.set_training_signal(tt.TRAINING_SIGNAL_ID_2)
    mind.set_target_anchoring(tt.BASE_ANCHORED)
    mind.set_authorship_target(tt.AUTHORSHIP_TARGET_FINAL_MLP_DOWN)
    sum2 = run_cycles(bundle_dir, mind, 1, gate=Gate(policy))
    assert sum2["admitted"] == 1 and sum2["head"] == "d0002"
    manifest = load_bundle(bundle_dir)
    assert [d["blob"] for d in manifest["deltas"]] == [D1_BLOB, D2_BLOB]
    return bundle_dir


def _journal_rows(bundle_dir, manifest, collection, **flt):
    journal = open_journal(bundle_dir, manifest)
    try:
        return journal.backend.find(collection, filter=flt or None)
    finally:
        journal.close()


def _snapshot(bundle_dir):
    manifest = load_bundle(bundle_dir)
    journal = open_journal(bundle_dir, manifest)
    try:
        counts = {c: journal.backend.count(c)
                  for c in ("self_edits", "deltas", "provenance",
                            "checkpoints")}
    finally:
        journal.close()
    blobs = sorted(os.listdir(os.path.join(bundle_dir, "blobs", "sha256")))
    return canonical_json(manifest), counts, blobs


# -- §3.8: the composite's carrying ---------------------------------------------

def test_composite_carrying_is_exact_and_priced(tmp_path):
    bundle_dir = _landed_stack(tmp_path)
    manifest = load_bundle(bundle_dir)
    segment = _active_chain(manifest)
    record, blob_bytes = build_composite(
        bundle_dir, manifest, segment, delta_id="d0003",
        self_edit_id="se_test")
    assert record["kind"] == KIND
    assert record["parent"] == "base"
    # Targets in segment order (§3.8 rule 6), ranks at the
    # members' own (scale 1: alpha == rank, verbatim factors).
    assert [t["name"] for t in record["targets"]] == [LM_HEAD, DOWN_PROJ]
    assert [(t["rank"], t["alpha"]) for t in record["targets"]] == \
        [(8, 8), (8, 8)]
    assert [t["target_norm"] for t in record["targets"]] == \
        pytest.approx([D1_NORM, D2_NORM], rel=1e-12)
    assert record["norm"] == pytest.approx(AGGREGATE_NORM, rel=1e-12)
    # One header instead of two: strictly fewer serialised
    # bytes than the segment's blobs summed (§3.6's derivation).
    assert record["bytes"] == len(blob_bytes) < SEGMENT_BYTES
    # Verbatim carrying: the composite's factors ARE the
    # members' factors, bit for bit (one member per tensor).
    parsed = parse_blob(blob_bytes)
    for i, member in enumerate(segment):
        member_factors = parse_blob(
            open(os.path.join(
                bundle_dir, "blobs", "sha256",
                member["blob"].split(":", 1)[1] + ".safetensors"),
                "rb").read()).tensors
        for src, dst in (("t0.A", f"t{i}.A"), ("t0.B", f"t{i}.B")):
            assert np.array_equal(parsed.tensors[dst],
                                  member_factors[src])


# -- §5: the verify ---------------------------------------------------------------

def _write_composite_blob(bundle_dir, record, blob_bytes):
    from si.loop import _write_blob
    _write_blob(bundle_dir, blob_bytes)


def test_verify_is_exact_on_the_landed_stack(tmp_path):
    bundle_dir = _landed_stack(tmp_path)
    manifest = load_bundle(bundle_dir)
    segment = _active_chain(manifest)
    record, blob_bytes = build_composite(
        bundle_dir, manifest, segment, delta_id="d0003",
        self_edit_id="se_test")
    _write_composite_blob(bundle_dir, record, blob_bytes)
    before = _snapshot(bundle_dir)
    verify = verify_transformer_squash(bundle_dir, manifest, segment,
                                       record)
    # A verify measurement writes nothing (§5.4 discipline).
    assert _snapshot(bundle_dir) == before
    assert verify["passed"] is True
    # EXACT, not merely within tolerance — reported as measured.
    assert verify["weight_space_max_abs_by_tensor"] == \
        {LM_HEAD: 0.0, DOWN_PROJ: 0.0}
    assert verify["behavioural_statistic_all_positions_max_abs"] == 0.0
    pops = {(p["signal_id"], p["partition"]): p
            for p in verify["populations"]}
    assert len(pops) == 4
    assert pops[("tiny-successor-lean-001", "train")]["sequences"] == 32
    assert pops[("tiny-successor-lean-001", "held_out")]["sequences"] == 16
    assert pops[("tiny-predecessor-lean-001", "train")]["sequences"] == 32
    assert pops[("tiny-predecessor-lean-001", "held_out")]["sequences"] == 16
    for p in pops.values():
        assert p["all_positions_max_abs"] == 0.0
        assert p["last_position_max_abs"] == 0.0


def test_verify_never_touches_probes(tmp_path):
    # §5.3's exclusion, asserted the way SI-35's audit asserted
    # it: a bundle copy with a perturbed probe artifact
    # produces a byte-identical verify record.
    bundle_dir = _landed_stack(tmp_path)
    manifest = load_bundle(bundle_dir)
    segment = _active_chain(manifest)
    record, blob_bytes = build_composite(
        bundle_dir, manifest, segment, delta_id="d0003",
        self_edit_id="se_test")
    _write_composite_blob(bundle_dir, record, blob_bytes)
    clean = verify_transformer_squash(bundle_dir, manifest, segment,
                                       record)
    copy_dir = str(tmp_path / "perturbed.si")
    shutil.copytree(bundle_dir, copy_dir)
    probe_path = os.path.join(copy_dir, "probe_suite.json")
    artifact = json.loads(open(probe_path).read())
    artifact["identity_suite"][0]["expected_logits"] = \
        [999.0] * len(artifact["identity_suite"][0]["expected_logits"])
    with open(probe_path, "w") as f:
        json.dump(artifact, f)
    perturbed = verify_transformer_squash(copy_dir, manifest, segment,
                                           record)
    assert canonical_json(perturbed) == canonical_json(clean)


def test_verify_fails_on_a_tampered_composite(tmp_path):
    # A composite whose factors do NOT reproduce the segment's
    # folded updates fails the verify with the measured
    # difference stated — a refusal-grade result, not a pass
    # (§5.6). Built by scaling one member's B by 1.5 before
    # composition (a different, wrong carrying).
    from si.transformer_delta import construct_transformer_delta
    bundle_dir = _landed_stack(tmp_path)
    manifest = load_bundle(bundle_dir)
    segment = _active_chain(manifest)
    specs = []
    for i, member in enumerate(segment):
        factors = parse_blob(open(os.path.join(
            bundle_dir, "blobs", "sha256",
            member["blob"].split(":", 1)[1] + ".safetensors"),
            "rb").read()).tensors
        t = member["targets"][0]
        specs.append({"name": t["name"], "kind": "lora",
                      "A": factors["t0.A"],
                      "B": factors["t0.B"] * (1.5 if i == 0 else 1.0),
                      "rank": t["rank"], "alpha": t["alpha"]})
    bad_record, bad_blob = construct_transformer_delta(
        manifest, specs, delta_id="d0003", parent="base",
        created_at="2026-10-27T00:00:00Z", self_edit_id="se_test")
    _write_composite_blob(bundle_dir, bad_record, bad_blob)
    verify = verify_transformer_squash(bundle_dir, manifest, segment,
                                       bad_record)
    assert verify["passed"] is False
    assert verify["weight_space_max_abs"] > 1e-4
    assert verify["behavioural_statistic_all_positions_max_abs"] > 1e-4


# -- §4/§4.7: the pipeline on the landed stack -------------------------------------

def test_squash_admits_the_landed_stack(tmp_path):
    bundle_dir = _landed_stack(tmp_path)
    pre_weights = assemble_weights(bundle_dir, load_bundle(bundle_dir))
    res = squash_bundle(bundle_dir, keep=0, gate=Gate(_policy(bundle_dir)))
    assert res["squashed"] is True
    assert res["verdict"] == "ADMIT"
    assert res["delta_id"] == "d0003"
    assert res["head"] == "d0002"  # head id deliberately unmoved
    assert res["verify"]["passed"] is True

    manifest = load_bundle(bundle_dir)
    by_id = {d["id"]: d for d in manifest["deltas"]}
    assert by_id["d0001"]["superseded_by"] == "d0003"
    assert by_id["d0002"]["superseded_by"] == "d0003"
    composite = by_id["d0003"]
    assert composite["kind"] == KIND
    assert composite["self_edit_id"] == res["edit_id"]
    assert [t["name"] for t in composite["targets"]] == [LM_HEAD, DOWN_PROJ]
    assert composite["norm"] == pytest.approx(AGGREGATE_NORM, rel=1e-12)

    # The journaled gate report: measured probes, attested
    # provenance, composite pricing stated in the size line.
    edits = _journal_rows(bundle_dir, manifest, "self_edits",
                          edit_id=res["edit_id"])
    assert len(edits) == 1 and edits[0]["gate_verdict"] == "ADMIT"
    checks = {c["name"]: c for c in edits[0]["gate_report"]["checks"]}
    identity = checks["identity_continuity"]
    assert identity["measured"] == pytest.approx(D2_DRIFT, abs=1e-12)
    assert checks["capability_floor"]["measured"] == 0.0
    safety = {s["name"]: s for s in checks["safety_bounds"]["subchecks"]}
    assert all(s["passed"] for s in safety.values())
    assert "attested segment's distinct-tensor count" in \
        safety["size_budgets"]["detail"]
    assert "squash attestation verified" in \
        checks["provenance_completeness"]["detail"]
    # The verify record and the squashed_from provenance are
    # journaled (§5.6, §4.7).
    prov = _journal_rows(bundle_dir, manifest, "provenance", child="d0003")
    relations = {p["relation"] for p in prov}
    assert "squashed_from" in relations
    verify_notes = [p for p in prov
                    if p["relation"] == "transformer-squash-verify"]
    assert len(verify_notes) == 1
    assert json.loads(verify_notes[0]["note"])["passed"] is True
    delta_rows = _journal_rows(bundle_dir, manifest, "deltas", id="d0003")
    assert len(delta_rows) == 1

    # Resolution, stated exactly (standing _active_chain
    # semantics): at the unmoved head the resolver follows the
    # time-travel rule and resolves the HISTORICAL ancestry —
    # which the verify proved computes identically — while
    # the post-squash view resolves the composite alone, and
    # both assemble bit-identical weights.
    segment = [by_id["d0001"], by_id["d0002"]]
    assert [d["id"] for d in _active_chain(manifest)] == ["d0001", "d0002"]
    view = post_squash_view(manifest, segment, composite)
    assert [d["id"] for d in _active_chain(view)] == ["d0003"]
    post_weights = assemble_weights(bundle_dir, view)
    for name, w in pre_weights.items():
        assert np.array_equal(post_weights[name], w)


def test_squash_under_bundle_policy_rejects_on_allowlist(tmp_path):
    # RFC SI-37 §11 item 2(iv)'s policy question, met: the
    # bundle's own policy allowlists nothing, so the composite
    # is REJECTED on the allowlist line alone — probes
    # measured-pass, provenance attested, nothing lands.
    bundle_dir = _landed_stack(tmp_path)
    res = squash_bundle(bundle_dir, keep=0)  # bundle policy, no gate
    assert res["squashed"] is False
    assert res["verdict"] == "REJECT"
    assert res["verify"]["passed"] is True
    manifest = load_bundle(bundle_dir)
    assert manifest["head"] == "d0002"
    assert len(manifest["deltas"]) == 2
    assert all("superseded_by" not in d for d in manifest["deltas"])
    edits = _journal_rows(bundle_dir, manifest, "self_edits",
                          edit_id=res["edit_id"])
    checks = {c["name"]: c for c in edits[0]["gate_report"]["checks"]}
    safety = {s["name"]: s for s in checks["safety_bounds"]["subchecks"]}
    assert safety["allowlist"]["passed"] is False
    assert all(s["passed"] for n, s in safety.items() if n != "allowlist")
    assert checks["identity_continuity"]["measured"] == \
        pytest.approx(D2_DRIFT, abs=1e-12)
    assert checks["capability_floor"]["passed"] is True


def test_squash_unwind_restores_pre_squash_ancestry(tmp_path):
    # §4.6's required proof, on the real stack shape: after
    # the squash, moving the head back across the composite
    # restores the pre-squash ancestry exactly.
    bundle_dir = _landed_stack(tmp_path)
    manifest = load_bundle(bundle_dir)
    view_d1 = dict(manifest)
    view_d1["head"] = "d0001"
    weights_d1 = assemble_weights(bundle_dir, view_d1)
    res = squash_bundle(bundle_dir, keep=0, gate=Gate(_policy(bundle_dir)))
    assert res["squashed"] is True
    out = rollback(bundle_dir, to="d0001", reason="SI-38 unwind proof")
    assert out["head"] == "d0001"
    manifest = load_bundle(bundle_dir)
    assert [d["id"] for d in _active_chain(manifest)] == ["d0001"]
    weights_now = assemble_weights(bundle_dir, manifest)
    for name, w in weights_d1.items():
        assert np.array_equal(weights_now[name], w)
    out = rollback(bundle_dir, to="base", reason="SI-38 unwind proof")
    assert out["head"] == "base"
    assert _active_chain(load_bundle(bundle_dir)) == []


def test_squash_is_deterministic(tmp_path):
    # §3.8 rule 6: two independently landed stacks squash to
    # the identical composite blob (canonical serialisation;
    # the hash is content addressing over it).
    hashes = []
    for name in ("tiny38a", "tiny38b"):
        bundle_dir = _landed_stack(tmp_path, name)
        res = squash_bundle(bundle_dir, keep=0,
                            gate=Gate(_policy(bundle_dir)))
        assert res["squashed"] is True
        manifest = load_bundle(bundle_dir)
        hashes.append({d["id"]: d for d in manifest["deltas"]}
                      ["d0003"]["blob"])
    assert hashes[0] == hashes[1]


# -- refusals: fail closed, nothing written -----------------------------------------

def test_segment_of_one_refused_unchanged(tmp_path):
    bundle_dir = _landed_stack(tmp_path)
    before = _snapshot(bundle_dir)
    with pytest.raises(SquashError, match="segment of one"):
        squash_bundle(bundle_dir, keep=1, gate=Gate(_policy(bundle_dir)))
    assert _snapshot(bundle_dir) == before


def test_verify_failure_refuses_and_cleans_up(tmp_path, monkeypatch):
    # §5.6/§4.7: a failed verify is a refusal — the candidate
    # blob is deleted again, nothing is journaled.
    import si.transformer_squash as ts

    def failing_verify(*a, **kw):
        return {"kind": "transformer-squash-verify", "tolerance": 1e-4,
                "weight_space_max_abs_by_tensor": {LM_HEAD: 0.5},
                "weight_space_max_abs": 0.5,
                "behavioural_statistic_all_positions_max_abs": 0.25,
                "populations": [], "finite": True, "passed": False,
                "populations_note": "stubbed failure for this test"}

    monkeypatch.setattr(ts, "verify_transformer_squash", failing_verify)
    bundle_dir = _landed_stack(tmp_path)
    before = _snapshot(bundle_dir)
    with pytest.raises(SquashError, match="tolerance verify failed"):
        squash_bundle(bundle_dir, keep=0, gate=Gate(_policy(bundle_dir)))
    assert _snapshot(bundle_dir) == before


def test_tampered_axis_record_fails_the_gate(tmp_path):
    # The per-member axis record is load-bearing (§4.2): a
    # member record that disagrees with the manifest (here a
    # forged norm) fails the attestation, and the candidate is
    # refused decisively — not repriced, not passed.
    bundle_dir = _landed_stack(tmp_path)
    manifest = load_bundle(bundle_dir)
    segment = _active_chain(manifest)
    journal = open_journal(bundle_dir, manifest)
    try:
        member_records = member_axis_records(journal, segment)
    finally:
        journal.close()
    record, blob_bytes = build_composite(
        bundle_dir, manifest, segment, delta_id="d0003",
        self_edit_id="se_test")
    attestation = build_attestation(segment, member_records)
    attestation["member_records"][0]["targets"][0]["norm"] = 999.0
    policy = _policy(bundle_dir)
    ctx = TransformerSquashContext(
        bundle_dir, manifest, post_squash_view(manifest, segment, record),
        attestation, depth=0, policy=policy)
    intent = Intent(kind=KIND, reason="squash 2 deltas into one (keep=0)",
                    parents=["d0001", "d0002"],
                    source_seqs=attestation["source_seqs"],
                    targets=[LM_HEAD, DOWN_PROJ])
    candidate = {"record": record, "blob_bytes": blob_bytes,
                 "norm": float(record["norm"]),
                 "member_records": attestation["member_records"]}
    report = Gate(policy).evaluate(intent, candidate, ctx)
    assert report.verdict == "REJECT"
    checks = {c.name: c for c in report.checks}
    assert checks["safety_bounds"].passed is False
    assert checks["provenance_completeness"].passed is False


def test_misordered_parents_get_no_composite_pricing(tmp_path):
    # §3.6's attestation answer, from the other side: parents
    # in the wrong order are not the attested segment, so no
    # composite pricing applies and the record is refused.
    bundle_dir = _landed_stack(tmp_path)
    manifest = load_bundle(bundle_dir)
    segment = _active_chain(manifest)
    journal = open_journal(bundle_dir, manifest)
    try:
        member_records = member_axis_records(journal, segment)
    finally:
        journal.close()
    record, blob_bytes = build_composite(
        bundle_dir, manifest, segment, delta_id="d0003",
        self_edit_id="se_test")
    attestation = build_attestation(segment, member_records)
    policy = _policy(bundle_dir)
    ctx = TransformerSquashContext(
        bundle_dir, manifest, post_squash_view(manifest, segment, record),
        attestation, depth=0, policy=policy)
    intent = Intent(kind=KIND, reason="squash 2 deltas into one (keep=0)",
                    parents=["d0002", "d0001"],  # wrong order
                    source_seqs=attestation["source_seqs"],
                    targets=[LM_HEAD, DOWN_PROJ])
    candidate = {"record": record, "blob_bytes": blob_bytes,
                 "norm": float(record["norm"]),
                 "member_records": attestation["member_records"]}
    report = Gate(policy).evaluate(intent, candidate, ctx)
    assert report.verdict == "REJECT"
