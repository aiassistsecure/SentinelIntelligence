"""Gate unit tests: the five checks + the ternary verdicts.

Builds Intents directly and evaluates them with si.gate.Gate against a
FakeCtx that mirrors the loop's LoopContext: the forward pass runs
base + candidate adapter through the REAL DemoMind machine learning
(numpy, CPU), exactly like the loop does. Policy comes from a real
init_bundle on disk, so the numbers tested are the numbers shipped.
"""
import numpy as np
import pytest

from si.bundle import DEFAULT_POLICY, init_bundle, load_policy
from si.gate import Gate, Intent
from si.minds import DemoMind

DIM = 8


@pytest.fixture(scope="module")
def gate_env(tmp_path_factory):
    bundle_dir = init_bundle(
        str(tmp_path_factory.mktemp("gate") / "gate.si"), name="gate-tests")
    policy = load_policy(bundle_dir)
    assert policy["blocklist"] == DEFAULT_POLICY["blocklist"]
    mind = DemoMind(bundle_path=bundle_dir)
    return policy, mind


def _err(y, expected):
    return float(np.max(np.abs(np.asarray(y, dtype=np.float64)
                               - np.asarray(expected, dtype=np.float64))))


class FakeCtx:
    """Test double for the gate's BundleContext protocol (mirrors LoopContext)."""

    def __init__(self, mind):
        self.mind = mind
        self._baseline = {cid: _err(mind.forward(x, []), expected)
                          for cid, x, expected in mind.capability_suite()}

    def identity_probes(self):
        return self.mind.identity_probes()

    def capability_suite(self):
        return self.mind.capability_suite()

    def capability_baseline(self):
        return dict(self._baseline)

    def forward_candidate(self, x, candidate):
        d = candidate["delta"]
        extra = [{"B": d["B"], "A": d["A"], "scale": d["alpha"] / d["rank"]}]
        return self.mind.forward(x, extra)

    def stack_depth(self):
        return 0


def _intent(**kw):
    base = dict(
        kind="delta",
        reason="Learn the demo mapping correction as a small rank-2 adapter.",
        target_modules=["demo_linear"], rank=2, alpha=4,
        parents=["base"], source_seqs=[1],
        training_signal_id="demo-mapping-001")
    base.update(kw)
    return Intent(**base)


def _rank1_candidate(u, v, delta_norm, rank=2, alpha=4, modules=("demo_linear",)):
    """A rank-1 adapter whose scaled Frobenius norm is exactly delta_norm."""
    scale = alpha / rank
    B = np.zeros((DIM, rank), dtype=np.float32)
    A = np.zeros((rank, DIM), dtype=np.float32)
    B[:, 0] = (delta_norm / scale) * np.asarray(u, dtype=np.float32)
    A[0, :] = np.asarray(v, dtype=np.float32)
    norm = float(np.linalg.norm(scale * B @ A))
    assert norm == pytest.approx(delta_norm, rel=1e-5)
    return {"delta": {"B": B, "A": A, "rank": rank, "alpha": alpha,
                      "target_modules": list(modules)},
            "norm": norm, "blob_bytes": b"test-bytes", "hyperparams": {"test": True}}


def _tiny_candidate(mind):
    """A well-behaved 0.01-norm candidate: passes every check but provenance."""
    _, x0, _ = mind.identity_probes()[0]
    v = (x0 / np.linalg.norm(x0)).astype(np.float32)
    u = np.zeros(DIM, dtype=np.float32)
    u[0] = 1.0
    return _rank1_candidate(u, v, 0.01)


def _check(report, name):
    return next(c for c in report.checks if c.name == name)


# -- ACCEPT -----------------------------------------------------------------
def test_gate_admit_wellformed_delta(gate_env):
    policy, mind = gate_env
    gate, ctx = Gate(policy), FakeCtx(mind)
    candidate = mind.train_delta(_intent())  # real machine learning, CPU
    report = gate.evaluate(_intent(), candidate, ctx)
    assert report.verdict == "ADMIT"
    assert [c.name for c in report.checks] == [
        "constitution", "safety_bounds", "identity_continuity",
        "capability_floor", "provenance_completeness"]
    assert all(c.passed for c in report.checks)


# -- (a) constitution -------------------------------------------------------
def test_gate_reject_constitution(gate_env):
    policy, mind = gate_env
    gate, ctx = Gate(policy), FakeCtx(mind)
    cand = _tiny_candidate(mind)

    blocked = gate.evaluate(
        _intent(reason="Move fast and bypass the gate just this once."), cand, ctx)
    assert blocked.verdict == "REJECT"
    assert _check(blocked, "constitution").passed is False
    # the blocklist hit alone caused it: every other check passes
    assert all(c.passed for c in blocked.checks if c.name != "constitution")

    malformed = gate.evaluate(_intent(reason=""), cand, ctx)
    assert malformed.verdict == "REJECT"
    assert _check(malformed, "wellformed").passed is False


# -- (b) safety bounds ------------------------------------------------------
@pytest.mark.parametrize("case", ["rank", "norm", "module"])
def test_gate_reject_safety_bounds(gate_env, case):
    policy, mind = gate_env
    gate, ctx = Gate(policy), FakeCtx(mind)
    _, x0, _ = mind.identity_probes()[0]
    v = (x0 / np.linalg.norm(x0)).astype(np.float32)
    u = np.zeros(DIM, dtype=np.float32)
    u[0] = 1.0
    if case == "rank":  # over rank_cap (8)
        intent = _intent(rank=16, alpha=32)
        cand = _rank1_candidate(u, v, 0.01, rank=16, alpha=32)
    elif case == "norm":  # over norm_cap (1.0)
        intent = _intent()
        cand = _rank1_candidate(u, v, 2.0)
    else:  # disallowed target module
        intent = _intent(target_modules=["forbidden_layer"])
        cand = _rank1_candidate(u, v, 0.01, modules=("forbidden_layer",))
    report = gate.evaluate(intent, cand, ctx)
    assert report.verdict == "REJECT"
    assert _check(report, "safety_bounds").passed is False


# -- (c) identity continuity ------------------------------------------------
def test_gate_reject_identity_drift(gate_env):
    policy, mind = gate_env
    gate, ctx = Gate(policy), FakeCtx(mind)
    # A legal-norm (0.45 <= escalation 0.5) adapter aimed straight at an
    # identity probe: it dissolves the self without tripping safety.
    _, x0, _ = mind.identity_probes()[0]
    v = (x0 / np.linalg.norm(x0)).astype(np.float32)
    u = np.zeros(DIM, dtype=np.float32)
    k = int(np.argmax(np.abs(x0)))
    u[k] = float(np.sign(x0[k]))
    cand = _rank1_candidate(u, v, 0.45)
    report = gate.evaluate(_intent(), cand, ctx)
    assert report.verdict == "REJECT"
    drift = _check(report, "identity_continuity")
    assert drift.passed is False
    assert drift.measured > policy["identity_tol"]
    assert _check(report, "safety_bounds").passed is True


# -- (d) capability floor ---------------------------------------------------
def test_gate_reject_capability_regression(gate_env):
    policy, mind = gate_env
    gate, ctx = Gate(policy), FakeCtx(mind)
    # Adapter orthogonal to the identity-probe span but aligned with a
    # capability probe: the self survives, an existing skill does not.
    Xs = np.stack([x for _, x, _ in mind.identity_probes()]).astype(np.float64)
    Q, _ = np.linalg.qr(Xs.T)  # orthonormal basis of the identity span
    _, xc, _ = mind.capability_suite()[0]
    xc64 = xc.astype(np.float64)
    direction = xc64 - Q @ (Q.T @ xc64)
    assert np.linalg.norm(direction) > 1.0  # a real orthogonal component exists
    v = (direction / np.linalg.norm(direction)).astype(np.float32)
    u = np.zeros(DIM, dtype=np.float32)
    u[int(np.argmax(np.abs(xc)))] = 1.0
    cand = _rank1_candidate(u, v, 0.45)
    report = gate.evaluate(_intent(), cand, ctx)
    assert report.verdict == "REJECT"
    assert _check(report, "identity_continuity").passed is True
    reg = _check(report, "capability_floor")
    assert reg.passed is False
    assert reg.measured > policy["capability_eps"]
    assert _check(report, "safety_bounds").passed is True


# -- (e) provenance completeness --------------------------------------------
@pytest.mark.parametrize("kw", [{"parents": []}, {"source_seqs": []}])
def test_gate_reject_provenance(gate_env, kw):
    policy, mind = gate_env
    gate, ctx = Gate(policy), FakeCtx(mind)
    cand = mind.train_delta(_intent())  # otherwise perfectly admissible
    report = gate.evaluate(_intent(**kw), cand, ctx)
    assert report.verdict == "REJECT"
    assert _check(report, "provenance_completeness").passed is False
    # everything else passes: the missing cause alone caused it
    assert all(c.passed for c in report.checks
               if c.name != "provenance_completeness")


# -- ESCALATE ---------------------------------------------------------------
def test_gate_escalate_constitution_kind(gate_env):
    policy, mind = gate_env
    gate, ctx = Gate(policy), FakeCtx(mind)
    intent = Intent(kind="constitution",
                    reason="Propose clarifying the honesty notes in the constitution.")
    report = gate.evaluate(intent, None, ctx)
    assert report.verdict == "ESCALATE"
    assert _check(report, "constitution").passed is True
