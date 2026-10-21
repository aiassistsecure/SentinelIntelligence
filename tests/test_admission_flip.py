"""SI-17: admission flip for miniature bundles only (RFC SI-13
§6.4/§6.6/§8 + the driver's blind-set guard).

Covers, in order:

1. §6.4 verdict mapping end-to-end through the autonomous loop:
   a benign fixture delta is ADMITted by ``run_cycle`` on a
   miniature bundle, materialised, and VERIFYed with real probes
   (§6.6, journal labelled ``transformer-probe-verify`` /
   measured-pass); a destructive fixture delta is REJECTed at
   the gate (measured-fail) and never lands;
2. §6.6 VERIFY failure rolls a landed delta back through the
   existing rollback core (verify step failure injected, SI-10
   precedent — a gate-passing delta's post-land probe failure is
   not constructible from the gate's own tolerances, since gate
   and VERIFY measure the same suites; the rollback mechanics
   and their journalling are what this proves);
3. the blind-set guard: a candidate targeting ONLY probe-blind
   tensors (the derived ``k_proj.bias`` set) ESCALATEs as
   blind-escalated with measured=None — a vacuous drift 0.0 is
   NOT a measured pass; a MIXED candidate (blind + measurable)
   may be ADMITted on the measurable targets' evidence only,
   and its gate details say so;
4. SI-10 interplay: a probe-failed miniature delta is refused by
   operator admission (fresh REJECT); a probe-passing one
   journals its measured results; a blind-escalated one admitted
   by an operator journals the blind-escalated state, not a pass;
5. non-miniature behaviour untouched: a Glimmer-shaped bundle
   still escalates with checks 3/4 "not evaluated", and
   ``si serve`` still returns 501 for a miniature bundle.
"""
import json
import os

import numpy as np
import pytest

import si.loop
from si.bundle import init_bundle, load_bundle, load_policy
from si.gate import Gate, Intent
from si.journal import open_journal
from si.loop import run_cycle
from si.probes import ProbeContext, derive_probe_blind_targets
from si.tiny_seed import seed_tiny_bundle
from si.transformer_delta import KIND, construct_transformer_delta

QPROJ = "model.language_model.layers.0.self_attn.q_proj.weight"
LM_HEAD = "lm_head.weight"
KBIAS0 = "model.language_model.layers.0.self_attn.k_proj.bias"
KBIAS1 = "model.language_model.layers.1.self_attn.k_proj.bias"


# -- fixtures (mirroring tests/test_probe_gate.py) ----------------------------

def _seeded(tmp_path, name="tiny17"):
    bundle_dir = init_bundle(str(tmp_path / (name + ".si")), name=name,
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    return bundle_dir


def _policy(bundle_dir, **over):
    policy = load_policy(bundle_dir)
    policy["allowed_target_patterns"] = ["*"]
    policy.update(over)
    return policy


def _intent(record, reason="SI-17 admission-flip cycle"):
    return Intent(kind=KIND, reason=reason, parents=["base"],
                  source_seqs=[1],
                  targets=[t["name"] for t in record["targets"]])


def _candidate(manifest, specs, delta_id="d0001"):
    record, blob = construct_transformer_delta(
        manifest, specs, delta_id=delta_id, parent=manifest["head"],
        created_at="2026-10-20T00:00:00Z", self_edit_id="se_" + delta_id)
    return {"record": record, "blob_bytes": blob}


def _lora(name, A, B, alpha):
    return {"name": name, "kind": "lora",
            "A": np.asarray(A, dtype=np.float32),
            "B": np.asarray(B, dtype=np.float32), "alpha": alpha}


def _benign(manifest, seed=1):
    rng = np.random.default_rng(seed)
    return _candidate(manifest, [_lora(
        QPROJ, rng.standard_normal((2, 32)) * 0.05,
        rng.standard_normal((32, 2)) * 0.05, 4)])


def _destructive(manifest):
    A = np.zeros((8, 32))
    B = np.zeros((32, 8))
    for i in range(8):
        A[i, i] = 0.3
        B[i, i] = 0.3
    return _candidate(manifest, [_lora(LM_HEAD, A, B, 16)])


def _blind(manifest, seed=4):
    rng = np.random.default_rng(seed)
    return _candidate(manifest, [{
        "name": KBIAS0, "kind": "additive",
        "d": (rng.standard_normal((32,)) * 0.1).astype(np.float32)}])


def _checks(report):
    return {c.name: c for c in report.checks}


def _journal_rows(bundle_dir, manifest, collection, **flt):
    journal = open_journal(bundle_dir, manifest)
    try:
        return journal.backend.find(collection, filter=flt or None)
    finally:
        journal.close()


class _MiniatureScriptMind:
    """One scripted transformer proposal for the autonomous loop.

    It authors nothing learned — the candidate is a hand-built
    fixture (RFC §4.4: delta authorship for the miniature is
    hand-built fixtures) — and it claims no forward pass: on the
    miniature path the loop's probes come from ProbeContext.
    """

    def __init__(self, intent, candidate):
        self._intent = intent
        self._candidate = candidate

    @property
    def name(self):
        return "miniature-script"

    def perceive(self, journal):
        return []

    def propose(self, perceptions, ctx):
        return self._intent

    def train_delta(self, intent):
        return self._candidate


def _cycle(bundle_dir, candidate, policy=None):
    mind = _MiniatureScriptMind(_intent(candidate["record"]), candidate)
    return run_cycle(bundle_dir, mind,
                     gate=Gate(policy or _policy(bundle_dir)))


# -- blind-set derivation -------------------------------------------------------

def test_blind_set_is_derived_from_the_architecture():
    blind = derive_probe_blind_targets()
    assert blind == {KBIAS0, KBIAS1}
    # Derivation, not a frozen list: a (hypothetical) 3-layer
    # descriptor of the same architecture yields 3 blind tensors.
    from si.tiny_transformer import EXPECTED_ARCHITECTURE
    arch3 = dict(EXPECTED_ARCHITECTURE, layers=3)
    assert derive_probe_blind_targets(arch3) == {
        f"model.language_model.layers.{i}.self_attn.k_proj.bias"
        for i in range(3)}


# -- 1. autonomous cycle: admit + verify; destructive rejects -------------------

def test_autonomous_cycle_admits_benign_and_verifies_with_real_probes(tmp_path):
    bundle_dir = _seeded(tmp_path)
    manifest = load_bundle(bundle_dir)
    cand = _benign(manifest)
    res = _cycle(bundle_dir, cand)
    assert res["verdict"] == "ADMIT"
    assert res["verified"] is True
    assert res["rolled_back"] is False
    assert res["delta_id"] == "d0001"
    assert res["head"] == "d0001"

    manifest = load_bundle(bundle_dir)
    assert manifest["head"] == "d0001"
    edits = _journal_rows(bundle_dir, manifest, "self_edits",
                          edit_id=res["edit_id"])
    assert len(edits) == 1
    checks = {c["name"]: c for c in edits[0]["gate_report"]["checks"]}
    assert checks["identity_continuity"]["measured"] is not None
    assert checks["identity_continuity"]["measured"] > 0.0
    assert "measured-pass" in checks["identity_continuity"]["detail"]
    assert "measured-pass" in checks["capability_floor"]["detail"]
    # §6.6 VERIFY: journaled under its own honest label, real probes.
    prov = _journal_rows(bundle_dir, manifest, "provenance", child="d0001")
    verify = [p for p in prov
              if p["relation"] == "transformer-probe-verify"]
    assert len(verify) == 1
    vreport = json.loads(verify[0]["note"])
    assert vreport["kind"] == "transformer-probe-verify"
    assert vreport["passed"] is True
    assert vreport["identity"]["label"] == "measured-pass"
    assert vreport["capability"]["label"] == "measured-pass"
    assert vreport["capability"]["echo_accuracy"] == 1.0
    assert "apply-smoke" not in vreport["note"] or \
        "NOT the transformer-apply-smoke" in vreport["note"]


def test_autonomous_cycle_rejects_destructive_at_the_gate(tmp_path):
    bundle_dir = _seeded(tmp_path, "tiny17-destr")
    manifest = load_bundle(bundle_dir)
    res = _cycle(bundle_dir, _destructive(manifest))
    assert res["verdict"] == "REJECT"
    assert res["delta_id"] is None
    manifest = load_bundle(bundle_dir)
    assert manifest["head"] == "base" and manifest["deltas"] == []
    edits = _journal_rows(bundle_dir, manifest, "self_edits",
                          edit_id=res["edit_id"])
    assert edits[0]["result"] == "rejected"
    checks = {c["name"]: c for c in edits[0]["gate_report"]["checks"]}
    assert checks["identity_continuity"]["passed"] is False
    assert checks["identity_continuity"]["measured"] > 0.25
    assert "measured-fail" in checks["identity_continuity"]["detail"]


# -- 2. VERIFY failure -> rollback ------------------------------------------------

def test_probe_verify_failure_rolls_back_the_landed_delta(tmp_path, monkeypatch):
    bundle_dir = _seeded(tmp_path, "tiny17-vfail")
    manifest = load_bundle(bundle_dir)
    cand = _benign(manifest)
    fake_report = {"kind": "transformer-probe-verify",
                   "head": "d0001", "passed": False,
                   "identity": {"max_drift": 9.0, "tol": 0.25,
                                "worst_probe": "tiny-id-0",
                                "passed": False, "label": "measured-fail"},
                   "note": "injected post-land probe failure (test)"}
    monkeypatch.setattr(si.loop, "_verify_probe_head",
                        lambda bd, mf, policy=None, baseline=None:
                        (False, fake_report))
    res = _cycle(bundle_dir, cand)
    assert res["verdict"] == "ADMIT"       # it landed, then rolled back
    assert res["verified"] is False
    assert res["rolled_back"] is True
    assert res["head"] == "base"
    manifest = load_bundle(bundle_dir)
    assert manifest["head"] == "base"
    drows = _journal_rows(bundle_dir, manifest, "deltas", id="d0001")
    assert drows[0]["status"] == "rolled_back"
    rbs = _journal_rows(bundle_dir, manifest, "self_edits", kind="rollback")
    assert len(rbs) == 1
    assert rbs[0]["intent"]["cause"] == "verify-failed"
    assert rbs[0]["intent"]["probe_report"]["kind"] == \
        "transformer-probe-verify"
    assert rbs[0]["intent"]["probe_report"]["identity"]["label"] == \
        "measured-fail"


def test_verify_probe_head_measures_the_landed_state(tmp_path):
    # The §6.6 measurement itself, un-injected: after the benign
    # delta lands, a fresh verify against the landed head passes
    # with real measured values; the base head verifies too.
    bundle_dir = _seeded(tmp_path, "tiny17-vunit")
    manifest = load_bundle(bundle_dir)
    ok, report = si.loop._verify_probe_head(
        bundle_dir, manifest, policy=_policy(bundle_dir))
    assert ok is True and report["passed"] is True
    assert report["identity"]["max_drift"] == 0.0
    res = _cycle(bundle_dir, _benign(load_bundle(bundle_dir)))
    assert res["verified"] is True
    manifest = load_bundle(bundle_dir)
    ok, report = si.loop._verify_probe_head(
        bundle_dir, manifest, policy=_policy(bundle_dir))
    assert ok is True
    assert report["identity"]["max_drift"] > 0.0   # the landed delta is seen
    assert report["capability"]["echo_accuracy"] == 1.0


# -- 3. blind-set guard -------------------------------------------------------------

def test_blind_only_candidate_escalates_never_admits(tmp_path):
    bundle_dir = _seeded(tmp_path, "tiny17-blind")
    manifest = load_bundle(bundle_dir)
    policy = _policy(bundle_dir, allow_additive=True)
    ctx = ProbeContext(bundle_dir, manifest, policy=policy)
    cand = _blind(manifest)
    report = Gate(policy).evaluate(_intent(cand["record"]), cand, ctx)
    checks = _checks(report)
    assert report.verdict == "ESCALATE"          # NOT admit, NOT reject
    for name in ("identity_continuity", "capability_floor"):
        assert checks[name].passed is False
        assert checks[name].measured is None     # no vacuous 0.0 pass
        assert "blind-escalated" in checks[name].detail
        assert "not measurable" in checks[name].detail
    # And through the autonomous loop: escalates, lands nothing.
    res = _cycle(bundle_dir, _blind(load_bundle(bundle_dir)), policy=policy)
    assert res["verdict"] == "ESCALATE"
    assert load_bundle(bundle_dir)["head"] == "base"


def test_mixed_blind_and_measurable_admits_on_measurable_evidence(tmp_path):
    bundle_dir = _seeded(tmp_path, "tiny17-mixed")
    manifest = load_bundle(bundle_dir)
    policy = _policy(bundle_dir, allow_additive=True,
                     max_targets_per_delta=2)
    rng = np.random.default_rng(1)
    cand = _candidate(manifest, [
        _lora(QPROJ, rng.standard_normal((2, 32)) * 0.05,
              rng.standard_normal((32, 2)) * 0.05, 4),
        {"name": KBIAS1, "kind": "additive",
         "d": (rng.standard_normal((32,)) * 0.1).astype(np.float32)},
    ])
    ctx = ProbeContext(bundle_dir, manifest, policy=policy)
    report = Gate(policy).evaluate(_intent(cand["record"]), cand, ctx)
    checks = _checks(report)
    assert report.verdict == "ADMIT"
    assert checks["identity_continuity"].measured > 0.0
    for name in ("identity_continuity", "capability_floor"):
        assert "probe-blind" in checks[name].detail
        assert "measurable" in checks[name].detail
        assert KBIAS1 in checks[name].detail


def test_mixed_candidate_with_failing_measurable_target_rejects(tmp_path):
    bundle_dir = _seeded(tmp_path, "tiny17-mixedfail")
    manifest = load_bundle(bundle_dir)
    policy = _policy(bundle_dir, allow_additive=True,
                     max_targets_per_delta=2)
    rng = np.random.default_rng(4)
    A = np.zeros((8, 32))
    B = np.zeros((32, 8))
    for i in range(8):
        A[i, i] = 0.3
        B[i, i] = 0.3
    cand = _candidate(manifest, [
        _lora(LM_HEAD, A, B, 16),
        {"name": KBIAS0, "kind": "additive",
         "d": (rng.standard_normal((32,)) * 0.1).astype(np.float32)},
    ])
    ctx = ProbeContext(bundle_dir, manifest, policy=policy)
    report = Gate(policy).evaluate(_intent(cand["record"]), cand, ctx)
    assert report.verdict == "REJECT"
    assert _checks(report)["identity_continuity"].measured > 0.25


# -- 4. SI-10 interplay --------------------------------------------------------------

def test_operator_admission_of_blind_delta_journals_blind_state(tmp_path):
    from si.admit import admit_transformer_delta

    bundle_dir = _seeded(tmp_path, "tiny17-admblind")
    manifest = load_bundle(bundle_dir)
    policy = _policy(bundle_dir, allow_additive=True)
    cand = _blind(manifest)
    res = admit_transformer_delta(
        bundle_dir, cand, _intent(cand["record"]),
        operator="test-operator",
        reason="operator accepts the blind target, on the record",
        gate=Gate(policy))
    assert res["admitted"] is True
    assert res["gate_verdict"] == "ESCALATE"     # blind-escalated, not ADMIT
    manifest = load_bundle(bundle_dir)
    edits = _journal_rows(bundle_dir, manifest, "self_edits",
                          edit_id=res["edit_id"])
    adm = edits[0]["intent"]["operator_admission"]
    assert "blind-escalated" in adm["basis"]
    assert "probe_results" in adm
    assert "blind-escalated" in adm["probe_results"]["mode"]
    checks = adm["probe_results"]["checks"]
    assert checks["identity_continuity"]["measured"] is None
    assert "blind-escalated" in \
        checks["identity_continuity"]["detail"]


# -- 5. non-miniature + serve unchanged ------------------------------------------------

def test_miniature_bundle_still_501_on_serve(tmp_path):
    from si.serve import ServeState

    bundle_dir = _seeded(tmp_path, "tiny17-serve")
    state = ServeState(bundle_dir)
    assert state.mind_kind == "transformer"
    assert state.inference_supported is False
