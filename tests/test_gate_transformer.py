"""SI-9b: gate checks for transformer deltas (RFC SI-9 §4, §6.2).

The §4.2 structural suite as the safety check for
``kind == "transformer-delta-v1"`` intents, the §4.5 policy keys with
fail-closed defaults, and §4.3 exactly as specified: identity and
capability CANNOT be evaluated for a transformer delta (no probe
harness exists), so a fully valid delta ESCALATES — it is never
admitted and never given a synthetic pass.

Fixtures are synthetic and small (SI-9a style): Glimmer-style tensor
names, fake base content hashes, no Glimmer reads. The FakeCtx's
probe methods RAISE if called — the gate must not run (or fake)
probes for a transformer delta.
"""
import copy
import hashlib

import numpy as np
import pytest

from si.bundle import DEFAULT_POLICY, init_bundle, load_bundle, load_policy
from si.gate import Gate, Intent, _lora_norm_gram
from si.transformer_delta import (
    KIND,
    construct_transformer_delta,
    parse_blob,
    serialize_blob,
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
    return {"base": {mode: entries}}


def _policy(**over):
    policy = dict(DEFAULT_POLICY)
    policy["allowed_target_patterns"] = [QPROJ]
    policy.update(over)
    return policy


class FakeCtx:
    """Gate context for transformer deltas: manifest + stack depth.

    The probe methods raise: a transformer gate evaluation that
    touches them is faking §4.3's unevaluated checks.
    """

    def __init__(self, manifest=None, depth=0):
        self._manifest = _base() if manifest is None else manifest
        self._depth = depth

    def base_manifest(self):
        return self._manifest

    def stack_depth(self):
        return self._depth

    def identity_probes(self):
        raise AssertionError("transformer gate must not run identity probes")

    def capability_suite(self):
        raise AssertionError("transformer gate must not run capability probes")

    def capability_baseline(self):
        raise AssertionError("transformer gate must not read a baseline")

    def forward_candidate(self, x, candidate):
        raise AssertionError("transformer gate must not run a forward pass")


def _lora_spec(name=QPROJ, rank=2, alpha=4, seed=1, scale=0.01):
    rng = np.random.default_rng(seed)
    base_shape = {QPROJ: (4, 3), VPROJ: (2, 2)}[name]
    return {"name": name, "kind": "lora",
            "A": (rng.standard_normal((rank, base_shape[1])) * scale).astype(np.float32),
            "B": (rng.standard_normal((base_shape[0], rank)) * scale).astype(np.float32),
            "rank": rank, "alpha": alpha}


def _additive_spec(name=NORM, seed=2, scale=0.01):
    rng = np.random.default_rng(seed)
    return {"name": name, "kind": "additive",
            "d": (rng.standard_normal((5,)) * scale).astype(np.float32)}


def _candidate(specs=None, base=None):
    base = _base() if base is None else base
    specs = [_lora_spec()] if specs is None else specs
    record, blob = construct_transformer_delta(
        base, specs, delta_id="d0001", parent="base",
        created_at="2026-10-19T00:00:00Z", self_edit_id="se_0001")
    return {"record": record, "blob_bytes": blob}


def _intent(targets=(QPROJ,), **kw):
    base = dict(kind=KIND,
                reason="Adapt one named tensor with a small bounded delta.",
                parents=["base"], source_seqs=[1],
                targets=list(targets))
    base.update(kw)
    return Intent(**base)


def _check(report, name):
    return next(c for c in report.checks if c.name == name)


def _sub(report, name):
    safety = _check(report, "safety_bounds")
    assert safety.subchecks is not None
    return next(c for c in safety.subchecks if c.name == name)


def _evaluate(candidate, policy=None, ctx=None, intent=None):
    gate = Gate(_policy() if policy is None else policy)
    return gate.evaluate(_intent() if intent is None else intent,
                         candidate, FakeCtx() if ctx is None else ctx)


# ---------------------------------------------------------------------------
# The valid case: ESCALATE, never ADMIT, never a synthetic pass (RFC §4.3)
# ---------------------------------------------------------------------------

def test_valid_lora_delta_escalates_with_honest_report():
    cand = _candidate()
    report = _evaluate(cand)
    assert report.verdict == "ESCALATE"
    assert [c.name for c in report.checks] == [
        "constitution", "safety_bounds", "identity_continuity",
        "capability_floor", "provenance_completeness"]
    assert _check(report, "constitution").passed is True
    safety = _check(report, "safety_bounds")
    assert safety.passed is True
    assert all(s.passed for s in safety.subchecks)
    assert safety.measured == pytest.approx(cand["record"]["norm"])
    for name in ("identity_continuity", "capability_floor"):
        check = _check(report, name)
        assert check.passed is False  # NOT a pass — not evaluated
        assert check.measured is None
        assert "not evaluated" in check.detail
        assert "no transformer probe harness" in check.detail
    assert _check(report, "provenance_completeness").passed is True
    # The journaled form carries the per-sub-check detail (RFC §4.2).
    safety_dict = report.to_dict()["checks"][1]
    assert [s["name"] for s in safety_dict["subchecks"]] == [
        "candidate_shape", "target_existence_identity", "shape_conformance",
        "dtype_conformance", "finite_values", "norm_budgets", "size_budgets",
        "allowlist", "additive_policy", "stack_depth",
        "structural_validation"]


def test_valid_delta_over_escalation_norm_still_escalates_not_admits():
    # Norm 0.75 > escalation_norm 0.5 but under both caps (raised to 2.0):
    # demo semantics would escalate on size alone; §4.3 escalates anyway.
    spec = {"name": QPROJ, "kind": "lora",
            "A": np.array([[1.0, 0, 0]], dtype=np.float32),
            "B": np.array([[0.75], [0], [0], [0]], dtype=np.float32),
            "rank": 1, "alpha": 1}
    cand = _candidate(specs=[spec])
    assert cand["record"]["norm"] == pytest.approx(0.75)
    report = _evaluate(cand, policy=_policy(norm_cap=2.0,
                                            per_target_norm_cap=2.0))
    assert report.verdict == "ESCALATE"
    assert _check(report, "safety_bounds").passed is True


def test_valid_additive_delta_escalates_when_policy_allows():
    cand = _candidate(specs=[_additive_spec()])
    policy = _policy(allowed_target_patterns=[NORM], allow_additive=True)
    report = _evaluate(cand, policy=policy, intent=_intent(targets=(NORM,)))
    assert report.verdict == "ESCALATE"
    assert _check(report, "safety_bounds").passed is True


def test_wildcard_pattern_allows_exactly_what_it_matches():
    cand = _candidate()
    policy = _policy(allowed_target_patterns=[
        "model.language_model.layers.*.self_attn.q_proj.weight"])
    report = _evaluate(cand, policy=policy)
    assert report.verdict == "ESCALATE"


# ---------------------------------------------------------------------------
# Dedicated failing fixture per §4.2 sub-check
# ---------------------------------------------------------------------------

def test_base_blob_hash_mismatch_rejects():
    cand = _candidate()
    cand["record"] = copy.deepcopy(cand["record"])
    cand["record"]["targets"][0]["base_blob"] = _fake_blob("attacker")
    report = _evaluate(cand)
    assert report.verdict == "REJECT"
    assert _sub(report, "target_existence_identity").passed is False


def test_unknown_target_rejects():
    cand = _candidate()
    cand["record"] = copy.deepcopy(cand["record"])
    cand["record"]["targets"][0]["name"] = "model.nope.weight"
    report = _evaluate(cand, intent=_intent(targets=("model.nope.weight",)))
    assert report.verdict == "REJECT"
    assert _sub(report, "target_existence_identity").passed is False


def test_transformer_delta_against_demo_base_rejects():
    # A demo base records no per-tensor map: every target is unknown.
    demo_manifest = {"base": {"source": "demo:random-seed (not a real model)",
                              "arch_source": "demo-unknown"}}
    report = _evaluate(_candidate(), ctx=FakeCtx(manifest=demo_manifest))
    assert report.verdict == "REJECT"
    assert _sub(report, "target_existence_identity").passed is False


def test_no_base_manifest_from_context_rejects():
    class BareCtx:
        def stack_depth(self):
            return 0
    report = _evaluate(_candidate(), ctx=BareCtx())
    assert report.verdict == "REJECT"
    assert _sub(report, "target_existence_identity").passed is False
    assert _sub(report, "structural_validation").passed is False


def test_nonfinite_factor_rejects():
    cand = _candidate()
    factors = dict(parse_blob(cand["blob_bytes"]).tensors)
    factors["t0.A"] = factors["t0.A"].copy()
    factors["t0.A"][0, 0] = np.inf
    blob = serialize_blob(factors, cand["record"]["targets"])
    cand = {"record": copy.deepcopy(cand["record"]), "blob_bytes": blob}
    cand["record"]["blob"] = "sha256:" + hashlib.sha256(blob).hexdigest()
    cand["record"]["bytes"] = len(blob)
    report = _evaluate(cand)
    assert report.verdict == "REJECT"
    assert _sub(report, "finite_values").passed is False


def test_over_byte_budget_rejects():
    cand = _candidate()
    report = _evaluate(cand, policy=_policy(delta_byte_cap=16))
    assert report.verdict == "REJECT"
    sub = _sub(report, "size_budgets")
    assert sub.passed is False
    assert "delta_byte_cap" in sub.detail


def test_over_aggregate_norm_cap_rejects():
    spec = _lora_spec(scale=10.0)  # norm far over norm_cap 1.0
    cand = _candidate(specs=[spec])
    assert cand["record"]["norm"] > 1.0
    report = _evaluate(cand)
    assert report.verdict == "REJECT"
    assert _sub(report, "norm_budgets").passed is False


def test_over_per_target_norm_cap_rejects():
    cand = _candidate()
    measured = cand["record"]["norm"]
    # Aggregate cap stays loose; only the per-target cap is tight.
    policy = _policy(norm_cap=100.0, per_target_norm_cap=measured / 2)
    report = _evaluate(cand, policy=policy)
    assert report.verdict == "REJECT"
    sub = _sub(report, "norm_budgets")
    assert sub.passed is False
    assert "per_target_norm_cap" in sub.detail


def test_norm_lie_rejects_via_gram_recomputation():
    cand = _candidate()
    cand["record"] = copy.deepcopy(cand["record"])
    cand["record"]["targets"][0]["target_norm"] *= 2
    from si.transformer_delta import aggregate_norm
    cand["record"]["norm"] = aggregate_norm(
        [t["target_norm"] for t in cand["record"]["targets"]])
    report = _evaluate(cand)
    assert report.verdict == "REJECT"
    assert _sub(report, "norm_budgets").passed is False


def test_disallowed_target_rejects():
    report = _evaluate(_candidate(),
                       policy=_policy(allowed_target_patterns=[VPROJ]))
    assert report.verdict == "REJECT"
    assert _sub(report, "allowlist").passed is False


def test_substring_pattern_does_not_allow():
    # No implicit substring matching: a fragment of the real name,
    # written as a (wildcard-free) pattern, grants nothing.
    report = _evaluate(_candidate(), policy=_policy(
        allowed_target_patterns=["layers.0.self_attn.q_proj.weight"]))
    assert report.verdict == "REJECT"
    assert _sub(report, "allowlist").passed is False


def test_empty_allowlist_default_rejects():
    # §4.5 fail-closed default: a policy with none of the SI-9b keys
    # admits no transformer target.
    legacy = {"norm_cap": 1.0, "escalation_norm": 0.5, "rank_cap": 8,
              "max_stack_depth": 8, "blocklist": []}
    report = _evaluate(_candidate(), policy=legacy)
    assert report.verdict == "REJECT"
    assert _sub(report, "allowlist").passed is False


def test_wrong_dimensionality_target_rejects():
    # 3-D base tensor (patch embedding stored unflattened in this
    # fixture): not targetable in the v1 design (RFC §2.2).
    base = _base()
    entry = base["base"]["tensor_refs"][CONV]
    d = np.full((8,), 0.1, dtype=np.float32)
    target = {"name": CONV, "base_blob": entry["blob"],
              "base_shape": entry["shape"], "base_dtype": entry["dtype"],
              "kind": "additive", "factor_keys": ["t0.d"],
              "target_norm": float(np.linalg.norm(d.astype(np.float64)))}
    blob = serialize_blob({"t0.d": d}, [target])
    record = {"id": "d0001", "parent": "base", "kind": KIND,
              "blob": "sha256:" + hashlib.sha256(blob).hexdigest(),
              "bytes": len(blob), "norm": target["target_norm"],
              "created_at": "2026-10-19T00:00:00Z", "self_edit_id": "se_0001",
              "targets": [target]}
    policy = _policy(allowed_target_patterns=[CONV], allow_additive=True)
    report = _evaluate({"record": record, "blob_bytes": blob}, policy=policy,
                       intent=_intent(targets=(CONV,)))
    assert report.verdict == "REJECT"
    assert _sub(report, "shape_conformance").passed is False


def test_rank_over_cap_rejects():
    spec = _lora_spec(rank=16, alpha=16, scale=0.001)
    cand = _candidate(specs=[spec])
    report = _evaluate(cand)
    assert report.verdict == "REJECT"
    sub = _sub(report, "size_budgets")
    assert sub.passed is False
    assert "rank_cap" in sub.detail


def test_too_many_targets_rejects():
    cand = _candidate(specs=[_lora_spec(), _additive_spec()])
    policy = _policy(allowed_target_patterns=[QPROJ, NORM],
                     allow_additive=True)  # max_targets_per_delta stays 1
    report = _evaluate(cand, policy=policy,
                       intent=_intent(targets=(QPROJ, NORM)))
    assert report.verdict == "REJECT"
    sub = _sub(report, "size_budgets")
    assert sub.passed is False
    assert "max_targets_per_delta" in sub.detail


def test_additive_rejected_when_policy_disallows():
    cand = _candidate(specs=[_additive_spec()])
    policy = _policy(allowed_target_patterns=[NORM])  # allow_additive False
    report = _evaluate(cand, policy=policy, intent=_intent(targets=(NORM,)))
    assert report.verdict == "REJECT"
    assert _sub(report, "additive_policy").passed is False


def test_stack_depth_cap_rejects():
    report = _evaluate(_candidate(), ctx=FakeCtx(depth=8))
    assert report.verdict == "REJECT"
    assert _sub(report, "stack_depth").passed is False


def test_metadata_tamper_rejects_via_structural_catchall():
    cand = _candidate()
    factors = dict(parse_blob(cand["blob_bytes"]).tensors)
    tampered = copy.deepcopy(cand["record"]["targets"])
    tampered[0]["name"] = VPROJ  # blob metadata will claim VPROJ
    blob = serialize_blob(factors, tampered)
    record = copy.deepcopy(cand["record"])
    record["blob"] = "sha256:" + hashlib.sha256(blob).hexdigest()
    record["bytes"] = len(blob)
    report = _evaluate({"record": record, "blob_bytes": blob})
    assert report.verdict == "REJECT"
    assert _sub(report, "structural_validation").passed is False


def test_factor_shape_lie_rejects():
    cand = _candidate()
    factors = dict(parse_blob(cand["blob_bytes"]).tensors)
    factors["t0.A"] = np.zeros((2, 99), dtype=np.float32)  # base in-dim is 3
    blob = serialize_blob(factors, cand["record"]["targets"])
    record = copy.deepcopy(cand["record"])
    record["blob"] = "sha256:" + hashlib.sha256(blob).hexdigest()
    record["bytes"] = len(blob)
    report = _evaluate({"record": record, "blob_bytes": blob})
    assert report.verdict == "REJECT"
    assert _sub(report, "shape_conformance").passed is False


# ---------------------------------------------------------------------------
# Norm boundary (RFC §7.2): Gram method, float64, strict > cap
# ---------------------------------------------------------------------------

def test_gram_norm_matches_direct_computation():
    rng = np.random.default_rng(5)
    B = rng.standard_normal((7, 3))
    A = rng.standard_normal((3, 5))
    direct = float(np.linalg.norm(2.0 * (B @ A)))
    assert _lora_norm_gram(B, A, 2.0) == pytest.approx(direct, rel=1e-12)


def _exact_norm_candidate(norm_value):
    """Additive delta whose norm is exactly norm_value (single entry)."""
    d = np.zeros((5,), dtype=np.float32)
    d[0] = norm_value
    return _candidate(specs=[{"name": NORM, "kind": "additive", "d": d}])


def test_norm_exactly_at_per_target_cap_passes_the_subcheck():
    cand = _exact_norm_candidate(0.5)
    policy = _policy(allowed_target_patterns=[NORM], allow_additive=True,
                     per_target_norm_cap=0.5, norm_cap=1.0)
    report = _evaluate(cand, policy=policy, intent=_intent(targets=(NORM,)))
    assert _sub(report, "norm_budgets").passed is True
    assert report.verdict == "ESCALATE"


def test_norm_just_above_per_target_cap_rejects():
    cand = _exact_norm_candidate(0.5001)
    policy = _policy(allowed_target_patterns=[NORM], allow_additive=True,
                     per_target_norm_cap=0.5, norm_cap=1.0)
    report = _evaluate(cand, policy=policy, intent=_intent(targets=(NORM,)))
    assert report.verdict == "REJECT"
    assert _sub(report, "norm_budgets").passed is False


# ---------------------------------------------------------------------------
# Provenance (§4.4) + constitution + malformed intents
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kw,targets", [
    ({"targets": []}, (QPROJ,)),          # intent declares no targets
    ({"targets": [VPROJ]}, (QPROJ,)),     # declared != record targets
    ({"parents": []}, (QPROJ,)),          # demo provenance still required
    ({"source_seqs": []}, (QPROJ,)),      # demo provenance still required
])
def test_provenance_failures_reject(kw, targets):
    intent_kw = dict(kw)
    declared = intent_kw.pop("targets", None)
    intent = _intent(targets=declared if declared is not None else targets,
                     **intent_kw)
    report = _evaluate(_candidate(), intent=intent)
    assert report.verdict == "REJECT"
    assert _check(report, "provenance_completeness").passed is False
    # Safety itself was fine: provenance alone caused the rejection.
    assert _check(report, "safety_bounds").passed is True


def test_constitution_blocklist_rejects_transformer_delta():
    intent = _intent(reason="Adapt a tensor and bypass the gate this once.")
    report = _evaluate(_candidate(), intent=intent)
    assert report.verdict == "REJECT"
    assert _check(report, "constitution").passed is False


def test_transformer_intent_without_candidate_rejects():
    report = Gate(_policy()).evaluate(_intent(), None, FakeCtx())
    assert report.verdict == "REJECT"
    assert _check(report, "wellformed").passed is False


def test_transformer_candidate_under_demo_intent_rejects():
    intent = Intent(kind="delta", reason="Sneak a transformer delta in.",
                    target_modules=["demo_linear"], rank=2, alpha=4,
                    parents=["base"], source_seqs=[1])
    report = Gate(_policy()).evaluate(intent, _candidate(), FakeCtx())
    assert report.verdict == "REJECT"
    assert _check(report, "wellformed").passed is False


def test_default_policy_transformer_keys_are_fail_closed():
    assert DEFAULT_POLICY["allowed_target_patterns"] == []
    assert DEFAULT_POLICY["max_targets_per_delta"] == 1
    assert DEFAULT_POLICY["delta_byte_cap"] == 4 * 1024 * 1024
    assert DEFAULT_POLICY["per_target_norm_cap"] <= DEFAULT_POLICY["norm_cap"]
    assert DEFAULT_POLICY["allow_additive"] is False


# ---------------------------------------------------------------------------
# Loop integration: `si run` halts on the transformer ESCALATE (§6.2)
# ---------------------------------------------------------------------------

class _TransformerScriptMind:
    """Scripted mind proposing ONE transformer delta, then idling.

    Its forward pass is the identity over its own probe inputs (the
    LoopContext baseline machinery needs a forward); it claims no
    transformer inference — the gate never asks it for probes on the
    transformer path, and nothing here runs a transformer.
    """

    def __init__(self, intent, candidate):
        self._intent = intent
        self._candidate = candidate
        self._proposed = False
        rng = np.random.default_rng(3)
        self._probes = [(f"script-id-{i}",
                         (v := rng.standard_normal(8).astype(np.float32)),
                         v.copy()) for i in range(2)]

    @property
    def name(self):
        return "transformer-script"

    def perceive(self, journal):
        return []

    def propose(self, perceptions, ctx):
        if not self._proposed:
            self._proposed = True
            return self._intent
        return Intent(kind="memory",
                      reason="Script exhausted: nothing more to propose.",
                      collection="experiences",
                      record={"kind": "observation", "data": {"note": "idle"}})

    def identity_probes(self):
        return self._probes

    def capability_suite(self):
        return self._probes

    def forward(self, x, active_deltas):
        return np.asarray(x, dtype=np.float64)

    def train_delta(self, intent):
        return self._candidate


def test_run_halts_on_transformer_escalation(tmp_path):
    from safetensors.numpy import save as st_save

    from si.journal import open_journal
    from si.run import run_cycles
    from si.seed import seed_bundle

    bundle_dir = init_bundle(str(tmp_path / "si9b-loop.si"),
                             name="si9b-loop", journal_backend="embedded")
    rng = np.random.default_rng(11)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(st_save({
        "W": rng.standard_normal((8, 8)).astype(np.float32),
        QPROJ: rng.standard_normal((4, 3)).astype(np.float32),
    }))
    seed_bundle(bundle_dir, str(ckpt), importer="test", by_reference=True)

    manifest = load_bundle(bundle_dir)
    record, blob = construct_transformer_delta(
        manifest, [_lora_spec()], delta_id="d0001", parent="base",
        created_at="2026-10-19T00:00:00Z", self_edit_id="se_0001")
    intent = _intent(parents=["base"], source_seqs=[1])
    mind = _TransformerScriptMind(
        intent, {"record": record, "blob_bytes": blob})
    policy = load_policy(bundle_dir)
    policy["allowed_target_patterns"] = [QPROJ]

    summary = run_cycles(bundle_dir, mind, 3, gate=Gate(policy))

    # The valid transformer delta ESCALATED and the autonomous run
    # halted on it (existing loop behaviour): nothing materialized.
    assert summary["cycles_completed"] == 1
    assert summary["escalated"] == 1
    assert summary["admitted"] == 0
    assert summary["halted"] is True
    assert summary["halt_verdict"] == "ESCALATE"
    assert summary["head"] == "base"
    assert summary["delta_stack_depth"] == 0

    manifest = load_bundle(bundle_dir)
    journal = open_journal(bundle_dir, manifest)
    try:
        rows = journal.backend.find(
            "self_edits", filter={"edit_id": summary["halt_edit_id"]})
        assert len(rows) == 1
        assert rows[0]["gate_verdict"] == "ESCALATE"
        assert rows[0]["result"] == "escalated"
        checks = {c["name"]: c for c in rows[0]["gate_report"]["checks"]}
        assert "not evaluated" in checks["identity_continuity"]["detail"]
        assert checks["safety_bounds"]["subchecks"]
    finally:
        journal.close()
