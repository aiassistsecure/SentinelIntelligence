"""SI-16: probe suites + gate wiring, SHADOW MODE (RFC SI-13 §5/§6/§8).

Covers the SI-16 acceptance criteria, in order:

1. every §5.6 reference delta produces its required measured
   outcome through the real gate + ProbeContext: the zero delta
   passes with drift exactly 0.0; a small benign delta passes with
   drift > 0; a large destructive delta FAILS check 3 on measured
   drift; a targeted echo attack FAILS check 4's accuracy arm;
   per-class sensitivity is measurable for every targetable tensor
   class (§3.2) — with one honestly documented exception:
   ``k_proj.bias`` is provably invisible to the probes (a constant
   key bias shifts every attention score for a query by the same
   amount, and softmax is shift-invariant), asserted here as
   measured 0.0 rather than papered over;
2. binding-mismatch fixtures (wrong base hash, wrong
   architecture, missing artifact, tampered artifact hash) all
   yield "not evaluated" + ESCALATE, never a pass — and so does a
   §6.3 startup-validation failure (tampered recorded baselines);
3. a Glimmer-shaped fixture bundle (no probe artifact) behaves
   exactly as today: the legacy not-evaluated string, ESCALATE;
4. the demo path's journaled gate reports are unchanged (no probe
   labelling leaks into them);
5. SI-10 admission of a shadow-escalated miniature delta journals
   the probe results with the admission (§6.5) — including a
   probe-FAILED delta, whose measured failures ride along with its
   ESCALATE verdict and its operator admission, because shadow
   mode is precisely the record of what SI-17 would reject.

And the shadow contract itself, asserted on every gate test here:
the verdict is ESCALATE whether probes pass, fail, or cannot run
— never ADMIT, and (for probe outcomes) never REJECT.
"""
import copy
import json
import os
import shutil

import numpy as np
import pytest

from si.bundle import (
    canonical_json,
    init_bundle,
    load_bundle,
    load_policy,
    sha256_hex,
)
from si.gate import TRANSFORMER_PROBES_NOT_EVALUATED, Gate, Intent
from si.journal import open_journal
from si.probes import ProbeContext
from si.tiny_seed import seed_tiny_bundle, tiny_tensor_shapes
from si.transformer_delta import KIND, construct_transformer_delta

QPROJ = "model.language_model.layers.0.self_attn.q_proj.weight"
LM_HEAD = "lm_head.weight"
KBIAS0 = "model.language_model.layers.0.self_attn.k_proj.bias"


# -- fixtures ---------------------------------------------------------------

def _seeded(tmp_path, name="tiny"):
    bundle_dir = init_bundle(str(tmp_path / (name + ".si")), name=name,
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    return bundle_dir


def _policy(bundle_dir, **over):
    policy = load_policy(bundle_dir)
    policy["allowed_target_patterns"] = ["*"]
    policy.update(over)
    return policy


def _intent(record, reason="SI-16 shadow-mode probe evaluation"):
    return Intent(kind=KIND, reason=reason, parents=["base"],
                  source_seqs=[1],
                  targets=[t["name"] for t in record["targets"]])


def _candidate(manifest, specs, delta_id="d0001"):
    record, blob = construct_transformer_delta(
        manifest, specs, delta_id=delta_id, parent=manifest["head"],
        created_at="2026-10-20T00:00:00Z", self_edit_id="se_" + delta_id)
    return {"record": record, "blob_bytes": blob}


def _evaluate(bundle_dir, candidate, policy=None):
    manifest = load_bundle(bundle_dir)
    ctx = ProbeContext(bundle_dir, manifest,
                       policy=policy or _policy(bundle_dir))
    report = Gate(policy or _policy(bundle_dir)).evaluate(
        _intent(candidate["record"]), candidate, ctx)
    return report, ctx


def _checks(report):
    return {c.name: c for c in report.checks}


def _lora(name, A, B, alpha):
    return {"name": name, "kind": "lora",
            "A": np.asarray(A, dtype=np.float32),
            "B": np.asarray(B, dtype=np.float32), "alpha": alpha}


def _zero_candidate(manifest):
    return _candidate(manifest, [_lora(QPROJ, np.zeros((2, 32)),
                                       np.zeros((32, 2)), 4)])


def _benign_candidate(manifest, seed=1):
    rng = np.random.default_rng(seed)
    return _candidate(manifest, [_lora(
        QPROJ, rng.standard_normal((2, 32)) * 0.05,
        rng.standard_normal((32, 2)) * 0.05, 4)])


def _destructive_candidate(manifest):
    A = np.zeros((8, 32))
    B = np.zeros((32, 8))
    for i in range(8):
        A[i, i] = 0.3
        B[i, i] = 0.3
    return _candidate(manifest, [_lora(LM_HEAD, A, B, 16)])


def _echo_attack_candidate(manifest, artifact):
    targets = [e["target_token"] for e in artifact["capability_suite"]]
    A = np.zeros((8, 32))
    B = np.zeros((32, 8))
    for k in range(8):
        t = targets[2 * k]           # attack every other entry (8 of 16)
        A[k, t] = 1.0
        B[t, k] = -0.3
        B[(t + 1) % 32, k] = 0.3
    return _candidate(manifest, [_lora(LM_HEAD, A, B, 8)])


def _artifact(bundle_dir):
    with open(os.path.join(bundle_dir, "probe_suite.json"), "rb") as f:
        return json.loads(f.read().decode("utf-8"))


# -- §5.6 reference deltas ----------------------------------------------------

def test_zero_delta_passes_with_drift_exactly_zero(tmp_path):
    bundle_dir = _seeded(tmp_path)
    manifest = load_bundle(bundle_dir)
    report, ctx = _evaluate(bundle_dir, _zero_candidate(manifest))
    assert ctx.probe_status() == (True, "ok")
    checks = _checks(report)
    assert report.verdict == "ESCALATE"          # shadow: never ADMIT
    assert checks["identity_continuity"].passed is True
    assert checks["identity_continuity"].measured == 0.0
    assert checks["capability_floor"].passed is True
    assert checks["capability_floor"].measured == 0.0
    subs = {c.name: c for c in checks["capability_floor"].subchecks}
    assert subs["capability_regression"].passed is True
    assert subs["echo_accuracy"].passed is True
    assert subs["echo_accuracy"].measured == 1.0
    assert "SHADOW MODE" in checks["identity_continuity"].detail


def test_benign_delta_passes_with_live_drift(tmp_path):
    bundle_dir = _seeded(tmp_path)
    manifest = load_bundle(bundle_dir)
    report, _ctx = _evaluate(bundle_dir, _benign_candidate(manifest))
    checks = _checks(report)
    assert report.verdict == "ESCALATE"
    assert checks["identity_continuity"].passed is True
    assert checks["identity_continuity"].measured > 0.0   # measurement is live
    assert checks["capability_floor"].passed is True
    assert checks["capability_floor"].measured > 0.0


def test_destructive_delta_fails_identity_but_still_escalates(tmp_path):
    bundle_dir = _seeded(tmp_path)
    manifest = load_bundle(bundle_dir)
    report, _ctx = _evaluate(bundle_dir, _destructive_candidate(manifest))
    checks = _checks(report)
    # Measured failure, journaled WITH the shadow ESCALATE: this row
    # of the shadow record is what SI-17's REJECT would have been.
    assert checks["identity_continuity"].passed is False
    assert checks["identity_continuity"].measured > 0.25   # > identity_tol
    assert report.verdict == "ESCALATE"          # NOT reject in shadow mode


def test_echo_attack_fails_capability_accuracy_arm(tmp_path):
    bundle_dir = _seeded(tmp_path)
    manifest = load_bundle(bundle_dir)
    policy = _policy(bundle_dir, norm_cap=2.0, per_target_norm_cap=2.0)
    cand = _echo_attack_candidate(manifest, _artifact(bundle_dir))
    report, _ctx = _evaluate(bundle_dir, cand, policy=policy)
    checks = _checks(report)
    cap = checks["capability_floor"]
    assert cap.passed is False
    subs = {c.name: c for c in cap.subchecks}
    # The accuracy arm is a named, measured failure: 8 of 16 echo
    # entries attacked, accuracy 0.5 against the 0.9375 floor.
    assert subs["echo_accuracy"].passed is False
    assert subs["echo_accuracy"].measured == 0.5
    assert "accuracy arm" in cap.detail and "FAIL" in cap.detail
    assert report.verdict == "ESCALATE"


def test_per_class_sensitivity_all_measurable(tmp_path):
    bundle_dir = _seeded(tmp_path)
    manifest = load_bundle(bundle_dir)
    policy = _policy(bundle_dir)
    policy["allow_additive"] = True
    ctx = ProbeContext(bundle_dir, manifest, policy=policy)
    assert ctx.probe_status() == (True, "ok")
    expected = {e["id"]: np.asarray(e["expected_logits"])
                for e in _artifact(bundle_dir)["identity_suite"]}
    probes = ctx.identity_probes()

    def drift(candidate):
        worst = 0.0
        for qid, x, exp in probes:
            y = ctx.forward_candidate(x, candidate)
            worst = max(worst, float(np.max(np.abs(
                np.asarray(y, dtype=np.float64)
                - np.asarray(exp, dtype=np.float64)))))
        return worst

    shapes = tiny_tensor_shapes()
    blind = []
    for name, shape in sorted(shapes.items()):
        rng = np.random.default_rng(7)
        if len(shape) == 2:      # §3.2 lora class (2-D projections)
            spec = _lora(name, rng.standard_normal((2, shape[1])) * 0.05,
                         rng.standard_normal((shape[0], 2)) * 0.05, 4)
        else:                    # §3.2 additive class (1-D biases/norms)
            spec = {"name": name, "kind": "additive",
                    "d": (rng.standard_normal(shape) * 0.1).astype(np.float32)}
        cand = _candidate(manifest, [spec], delta_id="d0002")
        d = drift(cand)
        if d == 0.0:
            blind.append(name)
        else:
            assert d > 0.0, f"no measurable drift for {name}"
    # Exactly one blind tensor, and it is the mathematically
    # necessary one: a constant k_proj.bias shifts every attention
    # score of a query by the same amount; softmax is invariant.
    assert blind == [KBIAS0,
                     "model.language_model.layers.1.self_attn.k_proj.bias"]


def test_startup_validation_reproduces_baselines_exactly(tmp_path):
    bundle_dir = _seeded(tmp_path)
    manifest = load_bundle(bundle_dir)
    ctx = ProbeContext(bundle_dir, manifest, policy=_policy(bundle_dir))
    assert ctx.probe_status() == (True, "ok")
    assert ctx.startup_report["max_baseline_deviation"] == 0.0
    assert ctx.startup_report["base_echo_accuracy"] == 1.0
    assert ctx.startup_report["entries_checked"] == 24
    assert ctx.capability_baseline() == {
        e["id"]: 0.0 for e in _artifact(bundle_dir)["capability_suite"]}


# -- §6.2 / §6.3: binding mismatches and startup failure ----------------------

def _copied_bundle(tmp_path, bundle_dir, name):
    dest = str(tmp_path / (name + ".si"))
    shutil.copytree(bundle_dir, dest)
    return dest


def _rewrite_artifact(bundle_dir, mutate):
    art = _artifact(bundle_dir)
    mutate(art)
    raw = canonical_json(art)
    with open(os.path.join(bundle_dir, "probe_suite.json"), "wb") as f:
        f.write(raw)
    return "sha256:" + sha256_hex(raw)


def _gate_with_manifest(bundle_dir, manifest, candidate):
    ctx = ProbeContext(bundle_dir, manifest, policy=_policy(bundle_dir))
    report = Gate(_policy(bundle_dir)).evaluate(
        _intent(candidate["record"]), candidate, ctx)
    return report, ctx


def test_binding_wrong_base_hash_not_evaluated(tmp_path):
    bundle_dir = _seeded(tmp_path)
    manifest = load_bundle(bundle_dir)
    cand = _benign_candidate(manifest)
    copy_dir = _copied_bundle(tmp_path, bundle_dir, "wrongbase")
    new_hash = _rewrite_artifact(
        copy_dir, lambda a: a["base_binding"].update(
            tensor_map_sha256="sha256:" + "0" * 64))
    forged = copy.deepcopy(manifest)
    forged["base"]["probe_artifact"]["sha256"] = new_hash
    report, ctx = _gate_with_manifest(copy_dir, forged, cand)
    assert ctx.probe_status()[0] is False
    checks = _checks(report)
    for name in ("identity_continuity", "capability_floor"):
        assert checks[name].passed is False
        assert checks[name].measured is None
        assert "not evaluated" in checks[name].detail
    assert report.verdict == "ESCALATE"


def test_binding_wrong_architecture_not_evaluated(tmp_path):
    bundle_dir = _seeded(tmp_path)
    manifest = load_bundle(bundle_dir)
    cand = _benign_candidate(manifest)
    copy_dir = _copied_bundle(tmp_path, bundle_dir, "wrongarch")
    new_hash = _rewrite_artifact(
        copy_dir, lambda a: a["architecture"].update(d_model=64))
    forged = copy.deepcopy(manifest)
    forged["base"]["probe_artifact"]["sha256"] = new_hash
    report, ctx = _gate_with_manifest(copy_dir, forged, cand)
    assert ctx.probe_status()[0] is False
    assert "architecture" in ctx.probe_status()[1]
    checks = _checks(report)
    assert checks["identity_continuity"].measured is None
    assert "not evaluated" in checks["identity_continuity"].detail
    assert report.verdict == "ESCALATE"


def test_binding_missing_artifact_not_evaluated(tmp_path):
    bundle_dir = _seeded(tmp_path)
    manifest = load_bundle(bundle_dir)
    cand = _benign_candidate(manifest)
    forged = copy.deepcopy(manifest)
    del forged["base"]["probe_artifact"]
    report, ctx = _gate_with_manifest(bundle_dir, forged, cand)
    assert ctx.probe_status()[0] is False
    checks = _checks(report)
    assert checks["capability_floor"].measured is None
    assert "not evaluated" in checks["capability_floor"].detail
    assert report.verdict == "ESCALATE"


def test_binding_tampered_artifact_hash_not_evaluated(tmp_path):
    bundle_dir = _seeded(tmp_path)
    manifest = load_bundle(bundle_dir)
    cand = _benign_candidate(manifest)
    forged = copy.deepcopy(manifest)
    forged["base"]["probe_artifact"]["sha256"] = "sha256:" + "f" * 64
    report, ctx = _gate_with_manifest(bundle_dir, forged, cand)
    assert ctx.probe_status()[0] is False
    assert "content hash" in ctx.probe_status()[1]
    checks = _checks(report)
    assert checks["identity_continuity"].measured is None
    assert report.verdict == "ESCALATE"


def test_startup_validation_failure_not_evaluated(tmp_path):
    # §6.3 (distinct from §6.2): the artifact is bound correctly but
    # its recorded baselines were tampered with, so the base forward
    # cannot reproduce them — no probe result counts.
    bundle_dir = _seeded(tmp_path)
    manifest = load_bundle(bundle_dir)
    cand = _benign_candidate(manifest)
    copy_dir = _copied_bundle(tmp_path, bundle_dir, "badbase")

    def corrupt(a):
        a["identity_suite"][0]["expected_logits"][0] += 5.0
    new_hash = _rewrite_artifact(copy_dir, corrupt)
    forged = copy.deepcopy(manifest)
    forged["base"]["probe_artifact"]["sha256"] = new_hash
    report, ctx = _gate_with_manifest(copy_dir, forged, cand)
    ok, reason = ctx.probe_status()
    assert ok is False and "startup validation" in reason
    checks = _checks(report)
    assert checks["identity_continuity"].measured is None
    assert "not evaluated" in checks["identity_continuity"].detail
    assert report.verdict == "ESCALATE"


def test_harness_exception_is_not_evaluated_never_a_failure(tmp_path):
    class ExplodingContext(ProbeContext):
        def forward_candidate(self, x, candidate):
            raise RuntimeError("forward exploded")

    bundle_dir = _seeded(tmp_path)
    manifest = load_bundle(bundle_dir)
    cand = _benign_candidate(manifest)
    ctx = ExplodingContext(bundle_dir, manifest, policy=_policy(bundle_dir))
    report = Gate(_policy(bundle_dir)).evaluate(
        _intent(cand["record"]), cand, ctx)
    checks = _checks(report)
    for name in ("identity_continuity", "capability_floor"):
        assert checks[name].measured is None
        assert "not evaluated" in checks[name].detail
        assert "measurement failed" in checks[name].detail
    assert report.verdict == "ESCALATE"


# -- non-miniature bundles: exactly as today -----------------------------------

class _FakeCtx:
    """The SI-9b test-double shape: manifest + stack depth only."""

    def __init__(self, manifest, depth=0):
        self.manifest = manifest
        self._depth = depth

    def base_manifest(self):
        return self.manifest

    def stack_depth(self):
        return self._depth

    def identity_probes(self):
        raise AssertionError("non-probe gate must not run identity probes")

    def capability_suite(self):
        raise AssertionError("non-probe gate must not run capability probes")

    def capability_baseline(self):
        raise AssertionError("non-probe gate must not read a baseline")

    def forward_candidate(self, x, candidate):
        raise AssertionError("non-probe gate must not run a forward pass")


def test_non_probe_context_gets_legacy_report_verbatim(tmp_path):
    bundle_dir = _seeded(tmp_path)
    manifest = load_bundle(bundle_dir)
    cand = _benign_candidate(manifest)
    report = Gate(_policy(bundle_dir)).evaluate(
        _intent(cand["record"]), cand, _FakeCtx(manifest))
    checks = _checks(report)
    for name in ("identity_continuity", "capability_floor"):
        assert checks[name].passed is False
        assert checks[name].measured is None
        assert checks[name].detail == TRANSFORMER_PROBES_NOT_EVALUATED
    assert report.verdict == "ESCALATE"


def test_probe_context_on_non_miniature_bundle_not_evaluated(tmp_path):
    # A Glimmer-shaped bundle: a per-tensor base with no miniature
    # kind and no probe artifact. ProbeContext refuses honestly.
    from si.seed import seed_bundle
    from safetensors.numpy import save
    rng = np.random.default_rng(11)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(
        save({QPROJ: rng.standard_normal((32, 32)).astype(np.float32)}))
    bundle_dir = init_bundle(str(tmp_path / "glimmerish.si"),
                             name="glimmerish", journal_backend="embedded")
    seed_bundle(bundle_dir, str(ckpt), importer="test")
    manifest = load_bundle(bundle_dir)
    ctx = ProbeContext(bundle_dir, manifest, policy=_policy(bundle_dir))
    ok, reason = ctx.probe_status()
    assert ok is False and "not evaluated" not in reason  # reason is the cause
    cand = _candidate(manifest, [_lora(QPROJ, np.zeros((2, 32)),
                                       np.zeros((32, 2)), 4)])
    report = Gate(_policy(bundle_dir)).evaluate(
        _intent(cand["record"]), cand, ctx)
    checks = _checks(report)
    assert checks["identity_continuity"].measured is None
    assert "not evaluated" in checks["identity_continuity"].detail
    assert report.verdict == "ESCALATE"


# -- demo path journaled reports unchanged --------------------------------------

def test_demo_cycle_journal_report_has_no_probe_labelling(tmp_path):
    from si.loop import run_cycle
    from si.minds import DemoMind

    bundle_dir = init_bundle(str(tmp_path / "demo.si"), name="demo-si16")
    res = run_cycle(bundle_dir, DemoMind(bundle_path=bundle_dir))
    assert res["verdict"] == "ADMIT"
    manifest = load_bundle(bundle_dir)
    journal = open_journal(bundle_dir, manifest)
    try:
        edits = journal.backend.find("self_edits",
                                     filter={"edit_id": res["edit_id"]})
    finally:
        journal.close()
    assert len(edits) == 1
    checks = {c["name"]: c for c in edits[0]["gate_report"]["checks"]}
    assert checks["identity_continuity"]["measured"] is not None
    assert checks["capability_floor"]["measured"] is not None
    for c in checks.values():
        assert "probe suite" not in c["detail"]
        assert "SHADOW MODE" not in c["detail"]
        assert "subchecks" not in c


# -- SI-10 admission journals shadow probe results (§6.5) ------------------------

def _journal_rows(bundle_dir, manifest, collection, **flt):
    journal = open_journal(bundle_dir, manifest)
    try:
        return journal.backend.find(collection, filter=flt or None)
    finally:
        journal.close()


def test_admission_of_shadow_escalated_miniature_journals_probes(tmp_path):
    from si.admit import admit_transformer_delta

    bundle_dir = _seeded(tmp_path)
    manifest = load_bundle(bundle_dir)
    cand = _benign_candidate(manifest)
    res = admit_transformer_delta(
        bundle_dir, cand, _intent(cand["record"]),
        operator="test-operator", reason="shadow evidence reviewed",
        gate=Gate(_policy(bundle_dir)))
    assert res["admitted"] is True and res["verified"] is True
    assert res["gate_verdict"] == "ESCALATE"     # shadow: still not ADMIT

    manifest = load_bundle(bundle_dir)
    edits = _journal_rows(bundle_dir, manifest, "self_edits",
                          edit_id=res["edit_id"])
    assert len(edits) == 1
    e = edits[0]
    checks = {c["name"]: c for c in e["gate_report"]["checks"]}
    # The gate report journaled with the admission carries the real
    # measured probe results — distinguishable from "not evaluated".
    assert checks["identity_continuity"]["measured"] is not None
    assert checks["identity_continuity"]["measured"] > 0.0
    assert checks["capability_floor"]["measured"] is not None
    # And the operator_admission block carries them alongside the basis.
    adm = e["intent"]["operator_admission"]
    assert "shadow_probe_results" in adm
    spr = adm["shadow_probe_results"]["checks"]
    assert spr["identity_continuity"]["measured"] == \
        checks["identity_continuity"]["measured"]
    assert "shadow" in adm["shadow_probe_results"]["mode"]


def test_admission_of_probe_failed_miniature_journals_failures(tmp_path):
    from si.admit import admit_transformer_delta

    bundle_dir = _seeded(tmp_path)
    manifest = load_bundle(bundle_dir)
    cand = _destructive_candidate(manifest)
    res = admit_transformer_delta(
        bundle_dir, cand, _intent(cand["record"]),
        operator="test-operator",
        reason="operator accepts the measured drift, on the record",
        gate=Gate(_policy(bundle_dir)))
    # Shadow mode: the gate ESCALATEd (it neither admitted nor
    # rejected on probe grounds), so SI-10 admission remains
    # possible — with the measured failure journaled beside it.
    assert res["gate_verdict"] == "ESCALATE"
    manifest = load_bundle(bundle_dir)
    edits = _journal_rows(bundle_dir, manifest, "self_edits",
                          edit_id=res["edit_id"])
    adm = edits[0]["intent"]["operator_admission"]
    spr = adm["shadow_probe_results"]["checks"]
    assert spr["identity_continuity"]["passed"] is False
    assert spr["identity_continuity"]["measured"] > 0.25


def test_shadow_probe_evaluation_lands_nothing(tmp_path):
    bundle_dir = _seeded(tmp_path)
    manifest = load_bundle(bundle_dir)
    report, _ctx = _evaluate(bundle_dir, _benign_candidate(manifest))
    assert report.verdict == "ESCALATE"
    after = load_bundle(bundle_dir)
    assert after["head"] == "base" and after["deltas"] == []
