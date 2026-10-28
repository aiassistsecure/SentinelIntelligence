"""SI-33: first non-readout authored target for the miniature
(RFC SI-32 §§12.1, 15, 16).

Covers the §16 criteria, in order:

1. recorded authorship-target configuration, two named
   values — the configuration serves exactly ``"lm-head"``
   and ``"final-mlp-down"`` and refuses every other value
   with the existing ``ValueError`` discipline (constructor,
   setter, intent, and ``train_candidate`` dispatch);
   recorded selection asserted per cycle on both verdict
   paths, including the decisive cycle's full quadruple
   (signal 2, ``target_anchoring = "base-anchored"``,
   composition = ``"none"``, authorship target =
   ``"final-mlp-down"``). The tensor, rank (8), and alpha
   (8) are NOT separately settable (asserted: neither the
   constructor nor the setter takes a tensor-name, rank,
   or alpha argument for the axis); exactly one new-target
   decisive run is performed (no sweep, no second target);
   composition under the new target is ``"none"`` only
   (any other composition configuration refuses — the
   mechanisms are never blended with the target change);
2. Phase 0 published before the decisive numbers — the
   backward derivation in checkable form, the SI-25
   control's per-probe inner-product table and
   allowed-input record (labelled diagnostics), the
   pre-registered F1'''/F2'''/F3''' rules with F2'''s
   clauses separated and F3'''s boundary stated
   (2.087217177504918e-05), and the gradient-check
   protocol — as-built doc, ahead of the decisive numbers;
   the control figures themselves are re-measured and
   asserted in the diagnostics test below;
3. gradient correctness, at both heads — the new target's
   analytic gradients pass central finite differences over
   every one of the 768 factor entries (B [32, 8],
   A [8, 64]) at base AND at head ``d0001``, tolerance
   1e-5 unchanged, worst disagreements published for both
   heads;
4. probe-blindness of training, asserted — the train
   partition's disjointness from both probe suites is
   re-asserted under the new code path; a
   perturbed-probe-artifact bundle copy produces a
   byte-identical new-target candidate; and a spy asserts
   the new gradient's input set is exactly the signal's
   train partition (plus the landed head's weights);
5. default-target reproduction and the at-base records —
   signal 2 base-anchored under ``"lm-head"`` at base
   reproduces the established at-base run exactly (blob
   ``sha256:131cc242…45f4``, drift 0.16672966815531254,
   regression 0.2418281203135848); the
   ``"final-mlp-down"`` at-base run is published as a
   labelled at-base fact under its own blob and figures;
6. determinism per cycle — a full decisive run repeated
   from the seed reproduces both candidates' blobs/records
   (the new-target candidate's blob included), the landed
   delta record, and the journaled gate measurements
   exactly; hyperparams equal except the labelled
   ``wall_time_seconds``;
7. gate honesty per cycle — each cycle's journaled
   checks 3/4 equal an independent ``ProbeContext``
   evaluation of the same candidate against the pre-cycle
   head (abs 1e-12) — for cycle 2, of the new-target
   candidate, the only candidate the gate is ever shown.
   Cycle 2's verdict is reported AS MEASURED: **ADMIT**
   (identity 0.17195402272045612 ≤ tol 0.25, capability
   0.001467481255531311 ≤ eps 0.25, safety pass). No ADMIT
   was required (§16 criterion 7);
8. cumulative accounting, diagnostics, the seven-run
   table, and the falsification readings published — the
   side-by-side accounting table, the full diagnostic set
   with candidate norm and ‖m2‖ beside every drift
   figure, the treatment's signal held-out record read
   under the pre-registered F3''' rule, the seven-run
   table with the treatment's row carrying its authorship
   target, and the F1'''/F2'''/F3''' readings with
   F2'''s clauses reported separately — F1''' NOT
   measured (the composed maximum moved), F2''' NOT
   measured as a reading (transfer clause holds, maximum
   clause fails), F3''' MEASURED (destination traded
   away: residual 0.0027387202915082178 > boundary
   2.087217177504918e-05);
9. first-cycle and all-control reproduction — cycle 1
   reproduces SI-19/21/23/25/27/29/31 exactly; the re-run
   ``"none"`` / ``"lm-head"`` control reproduces SI-25's
   decisive candidate exactly; the re-run
   ``"movement-composition"``, ``"worst-input-composition"``,
   and ``"train-partition-projection"`` controls reproduce
   SI-27's, SI-29's, and SI-31's decisive candidates
   exactly;
10. VERIFY, unwind, and untouched scope — cycle-1 VERIFY
    as in SI-19; cycle 2 ADMITted and VERIFY passed, so
    the authored two-delta stack EXISTS (head ``d0002``)
    and is unwound step by step through the existing
    rollback core; exactly two authored cycles attempted
    and exactly one new-target decisive run performed;
    policy defaults, both signals', both anchorings', and
    all four composition configurations' definitions
    unchanged; ``si squash`` on the landed stack runs
    SI-38's designed pipeline and is REJECTED under the
    bundle's own policy (see the test body).

The decisive run's driver shape: ONE ``TinyTransformerMind``
reused through ``si.run.run_cycles`` — cycle 1 (signal 1,
default anchoring, default composition, default target),
then the driver explicitly reconfigures the same mind to
signal 2 (``set_training_signal``), to base-anchored
targets (``set_target_anchoring``), AND to the
``"final-mlp-down"`` authorship target
(``set_authorship_target`` — four recorded selections),
then cycle 2.
"""
import copy
import json
import os

import numpy as np
import pytest

import si.tiny_training as tt
from si.bundle import (
    DEFAULT_POLICY,
    init_bundle,
    load_bundle,
    load_policy,
)
from si.gate import Gate, Intent
from si.journal import open_journal
from si.loop import _verify_probe_head, rollback, run_cycle
from si.probes import ProbeContext
from si.run import run_cycles
from si.squash import SquashError, squash_bundle
from si.tiny_seed import (
    capability_inputs,
    identity_inputs,
    seed_tiny_bundle,
)
from si.tiny_transformer import (
    TinyTransformerMind,
    assemble_base_weights,
    assemble_weights,
    assemble_weights_with_candidate,
    forward,
)
from si.transformer_delta import KIND, parse_blob

GRAD_TOL = 1e-5
GRAD_EPS = 0.01

SI19_BLOB = "sha256:2761e8ffc139ce3e107c11da035e8a73c20792945a494f4e493dccf418b6ca9e"
SI19_DRIFT = 0.17189218290150166
SI19_REGRESSION = 0.1657181391492486

S2_AT_BASE_BLOB = "sha256:131cc242728024b4f833bc78ea627825f1e8616972fc2670603673e1814645f4"
S2_AT_BASE_DRIFT = 0.16672966815531254
S2_AT_BASE_REGRESSION = 0.2418281203135848

CTRL_BLOB = "sha256:646d1e8ba3978d858d4173d561f8799bce37350f88ff2de65f40c223d9e4d675"
CTRL_NORM = 0.2515524481203623
CTRL_DRIFT = 0.3329036422073841
CTRL_REGRESSION = 0.10096919978968799
CTRL_HELD_AFTER = 1.043608588752459e-05
CTRL_INNER_AT_ID0 = 0.0338873784299374

PEN_BLOB = "sha256:1bb277ab40b9aa53a6218d971e46f29b261ba2bb5f7932ffb98e3951f372b637"
PEN_DRIFT = 0.32509003952145576

WORST_BLOB = "sha256:11cab02e560e3661ea419fa0c8b6a95de25d3c7fb1ee23ec52ec8462a36b8db5"
WORST_DRIFT = 0.31215621903538704

PROJ_BLOB = "sha256:32d13f65d0de1de2901fcd1c1a5cdece12cdfe441454f5d75e8932941827a5c1"
PROJ_DRIFT = 0.4775387533009052

S21_DRIFT = 0.34378434903919697
S23_DRIFT = 0.2963416576385498

#: F3''' completeness boundary (RFC SI-32 §15 item 8,
#: carried from SI-29's rule in shape against the re-run
#: SI-25 control): 2× the re-run SI-25 control's published
#: held-out residual — stated in Phase 0, before the
#: decisive numbers.
F3PPP_BOUNDARY = 2.087217177504918e-05
assert F3PPP_BOUNDARY == 2 * CTRL_HELD_AFTER

# SI-33 measured values (author's runs; the as-built doc
# publishes them with their labels). The decisive cycle-2
# verdict is ADMIT — measured, not required, not tuned for.
DOWN_BLOB = "sha256:d7af7f1feaa5b0451d441a4fce8842b471fc7c4b3e70e6237f1115eba877e625"
DOWN_NORM = 0.30801411790830574
DOWN_BYTES = 3416
DOWN_DRIFT = 0.17195402272045612
DOWN_REGRESSION = 0.001467481255531311
DOWN_HELD_AFTER = 0.0027387202915082178
DOWN_BASE_HELD = 0.002757805049077469
DOWN_INNER_AT_ID0 = -0.00031118965094582363

DOWN_AT_BASE_BLOB = "sha256:98768588aff8852d4177ed9272fe7d46cd904d3ebeac4bd5078b83cef0258d9a"
DOWN_AT_BASE_NORM = 0.29741157499219223
DOWN_AT_BASE_DRIFT = 0.0016501061618328094
DOWN_AT_BASE_REGRESSION = 0.0026158951222896576
DOWN_AT_BASE_HELD_AFTER = 0.0013987123437990195

GRAD_WORST_BASE = 3.8495586315441856e-08
GRAD_WORST_D0001 = 2.7961983521798964e-08


def _seeded(tmp_path, name="tiny33"):
    bundle_dir = init_bundle(str(tmp_path / (name + ".si")), name=name,
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    return bundle_dir


def _policy(bundle_dir, **over):
    policy = load_policy(bundle_dir)
    policy["allowed_target_patterns"] = ["*"]
    policy.update(over)
    return policy


def _spying_mind(bundle_dir, **kw):
    mind = TinyTransformerMind(bundle_dir, load_bundle(bundle_dir), **kw)
    seen = []
    orig = mind.train_delta

    def spy(intent):
        cand = orig(intent)
        seen.append(cand)
        return cand

    mind.train_delta = spy
    return mind, seen


def _journal_rows(bundle_dir, manifest, collection, **flt):
    journal = open_journal(bundle_dir, manifest)
    try:
        return journal.backend.find(collection, filter=flt or None)
    finally:
        journal.close()


def _checks_of_edit(bundle_dir, manifest, edit_id):
    edits = _journal_rows(bundle_dir, manifest, "self_edits",
                          edit_id=edit_id)
    assert len(edits) == 1
    return {c["name"]: c for c in edits[0]["gate_report"]["checks"]}, edits[0]


def _hyperparams_notes(bundle_dir, manifest, child):
    prov = _journal_rows(bundle_dir, manifest, "provenance", child=child)
    notes = [p for p in prov if p["relation"] == "training-hyperparams"]
    assert len(notes) == 1
    return json.loads(notes[0]["note"])


def _run_with_cycle2(tmp_path, name, comp=None, axis=None):
    bundle_dir = _seeded(tmp_path, name)
    policy = _policy(bundle_dir)
    mind, seen = _spying_mind(bundle_dir)
    sum1 = run_cycles(bundle_dir, mind, 1, gate=Gate(policy))
    pre_manifest = load_bundle(bundle_dir)
    mind.set_training_signal(tt.TRAINING_SIGNAL_ID_2)
    mind.set_target_anchoring(tt.BASE_ANCHORED)
    if comp is not None:
        mind.set_composition_config(comp)
    if axis is not None:
        mind.set_authorship_target(axis)
    sum2 = run_cycles(bundle_dir, mind, 1, gate=Gate(policy))
    return bundle_dir, policy, mind, seen, sum1, sum2, pre_manifest


def _decisive_run(tmp_path, name="tiny33"):
    return _run_with_cycle2(tmp_path, name, axis="final-mlp-down")


def _land_d0001(tmp_path, name="tiny33"):
    bundle_dir = _seeded(tmp_path, name)
    policy = _policy(bundle_dir)
    mind, _seen = _spying_mind(bundle_dir)
    res = run_cycle(bundle_dir, mind, gate=Gate(policy))
    assert res["verdict"] == "ADMIT" and res["head"] == "d0001"
    return bundle_dir, policy


def _grad_worst_down(weights_under, target_weights, signal_id):
    """Worst analytic-vs-central-finite-difference
    disagreement over ALL 768 down-factor entries, on an
    8-example subset (the SI-19/SI-25 gradient-check shape):
    targets derived from ``target_weights`` (the base),
    down inputs / readout / norms from ``weights_under``.
    The finite differences differentiate
    ``down_factor_loss`` (the float32 shared fold)."""
    signal = tt.build_training_signal(target_weights, signal_id)
    subset = signal["train"][:8]
    A0, B0 = tt.initial_factors_down(signal_id)
    assert A0.shape == (8, 64) and B0.shape == (32, 8)
    di = tt.down_inputs_for(weights_under, [s for s, _ in subset])
    _loss, grad_A, grad_B = tt.analytic_gradients_down(
        A0, B0, subset, di, weights_under)
    assert np.max(np.abs(grad_A)) > 0.0 and np.max(np.abs(grad_B)) > 0.0
    worst = 0.0
    for i in range(tt.RANK):
        for j in range(64):
            plus = A0.copy(); plus[i, j] += GRAD_EPS
            minus = A0.copy(); minus[i, j] -= GRAD_EPS
            fd = (tt.down_factor_loss(weights_under, plus, B0, subset)
                  - tt.down_factor_loss(weights_under, minus, B0, subset)) \
                / (2 * GRAD_EPS)
            worst = max(worst, abs(fd - grad_A[i, j]))
    for i in range(32):
        for j in range(tt.RANK):
            plus = B0.copy(); plus[i, j] += GRAD_EPS
            minus = B0.copy(); minus[i, j] -= GRAD_EPS
            fd = (tt.down_factor_loss(weights_under, A0, plus, subset)
                  - tt.down_factor_loss(weights_under, A0, minus, subset)) \
                / (2 * GRAD_EPS)
            worst = max(worst, abs(fd - grad_B[i, j]))
    return worst


# -- §16.1: recorded authorship-target configuration, two values -------------------

def test_authorship_target_configuration_and_refusals(tmp_path):
    assert tt.AUTHORSHIP_TARGETS == ("lm-head", "final-mlp-down")
    assert tt.DEFAULT_AUTHORSHIP_TARGET == "lm-head"
    assert tt.AUTHORSHIP_TARGET_TENSORS == {
        "lm-head": "lm_head.weight",
        "final-mlp-down":
            "model.language_model.layers.1.mlp.down_proj.weight"}
    assert tt.DOWN_TARGET_NAME == \
        "model.language_model.layers.1.mlp.down_proj.weight"
    assert tt.authorship_target_name("lm-head") == tt.TARGET_NAME
    assert tt.authorship_target_name("final-mlp-down") == \
        tt.DOWN_TARGET_NAME
    assert tt.RANK == 8 and tt.ALPHA == 8
    for good in tt.AUTHORSHIP_TARGETS:
        assert tt.validate_authorship_target(good) == good
    for bad in ("down-proj", "final_mlp_down", "", "LM-HEAD", None,
                "attention", "layers.0.mlp.down_proj.weight"):
        with pytest.raises(ValueError, match="unknown authorship target"):
            tt.validate_authorship_target(bad)

    bundle_dir = _seeded(tmp_path)
    mind = TinyTransformerMind(bundle_dir, load_bundle(bundle_dir))
    assert mind.authorship_target == "lm-head"
    assert mind.propose([], None).targets == [tt.TARGET_NAME]
    mind_d = TinyTransformerMind(
        bundle_dir, load_bundle(bundle_dir),
        authorship_target="final-mlp-down")
    assert mind_d.authorship_target == "final-mlp-down"
    intent_d = mind_d.propose([], None)
    assert intent_d.authorship_target == "final-mlp-down"
    assert intent_d.targets == [tt.DOWN_TARGET_NAME]
    assert intent_d.target_modules == [tt.DOWN_TARGET_NAME]
    assert intent_d.rank == 8 and intent_d.alpha == 8
    assert intent_d.to_dict()["authorship_target"] == "final-mlp-down"
    with pytest.raises(ValueError, match="unknown authorship target"):
        TinyTransformerMind(bundle_dir, load_bundle(bundle_dir),
                            authorship_target="probe-chosen")
    # The tensor, rank, and alpha are NOT separately
    # settable on the axis: neither the constructor nor the
    # setter takes such an argument.
    for kw in ({"target_name": tt.DOWN_TARGET_NAME},
               {"tensor_name": tt.DOWN_TARGET_NAME},
               {"authorship_rank": 4}, {"authorship_alpha": 4}):
        with pytest.raises(TypeError):
            TinyTransformerMind(bundle_dir, load_bundle(bundle_dir), **kw)
    mind.set_authorship_target("final-mlp-down")
    assert mind.authorship_target == "final-mlp-down"
    mind.set_authorship_target("lm-head")
    assert mind.authorship_target == "lm-head"
    with pytest.raises(ValueError, match="unknown authorship target"):
        mind.set_authorship_target("learned")
    with pytest.raises(TypeError):
        mind.set_authorship_target("final-mlp-down",
                                   tensor_name=tt.DOWN_TARGET_NAME)
    with pytest.raises(TypeError):
        mind.set_authorship_target("final-mlp-down", rank=4)
    # train_candidate dispatch refusals: unknown axis value,
    # a declared target list that is not the axis value's
    # own tensor, a foreign rank/alpha, and any composition
    # blend under the new target.
    base = dict(kind=KIND, reason="SI-33 refusal probe",
                parents=["base"], source_seqs=[1])
    for bad in ("mlp-down", "", "lm_head"):
        with pytest.raises(ValueError, match="unknown authorship target"):
            mind.train_delta(Intent(
                training_signal_id=tt.TRAINING_SIGNAL_ID,
                authorship_target=bad, targets=[tt.TARGET_NAME],
                **base))
    with pytest.raises(ValueError, match="never silently retargets"):
        mind.train_delta(Intent(
            training_signal_id=tt.TRAINING_SIGNAL_ID,
            authorship_target="final-mlp-down",
            targets=[tt.TARGET_NAME], **base))
    with pytest.raises(ValueError, match="never silently retargets"):
        mind.train_delta(Intent(
            training_signal_id=tt.TRAINING_SIGNAL_ID,
            authorship_target="lm-head",
            targets=[tt.DOWN_TARGET_NAME], **base))
    with pytest.raises(ValueError, match="pinned at rank"):
        mind.train_delta(Intent(
            training_signal_id=tt.TRAINING_SIGNAL_ID, rank=4,
            authorship_target="final-mlp-down",
            targets=[tt.DOWN_TARGET_NAME], **base))
    for comp in ("movement-composition", "worst-input-composition",
                 "train-partition-projection"):
        with pytest.raises(ValueError, match="composition"):
            mind.train_delta(Intent(
                training_signal_id=tt.TRAINING_SIGNAL_ID_2,
                target_anchoring="base-anchored",
                composition_config=comp,
                authorship_target="final-mlp-down",
                targets=[tt.DOWN_TARGET_NAME], **base))
    # The full quadruple on the intent the decisive driver
    # proposes.
    mind.set_training_signal(tt.TRAINING_SIGNAL_ID_2)
    mind.set_target_anchoring("base-anchored")
    mind.set_authorship_target("final-mlp-down")
    intent = mind.propose([], None)
    assert intent.training_signal_id == tt.TRAINING_SIGNAL_ID_2
    assert intent.target_anchoring == "base-anchored"
    assert intent.composition_config == "none"
    assert intent.authorship_target == "final-mlp-down"


# -- §16.3: gradient correctness, at both heads ---------------------------------------

def test_gradient_check_down_at_base(tmp_path):
    bundle_dir = _seeded(tmp_path)
    policy = _policy(bundle_dir)
    manifest = load_bundle(bundle_dir)
    weights = assemble_weights(bundle_dir, manifest, policy)
    target_weights = assemble_base_weights(bundle_dir, manifest, policy)
    worst = _grad_worst_down(weights, target_weights,
                             tt.TRAINING_SIGNAL_ID_2)
    assert worst == pytest.approx(GRAD_WORST_BASE, rel=0.5)
    assert worst <= GRAD_TOL


def test_gradient_check_down_at_d0001(tmp_path):
    bundle_dir, policy = _land_d0001(tmp_path)
    manifest = load_bundle(bundle_dir)
    weights = assemble_weights(bundle_dir, manifest, policy)
    target_weights = assemble_base_weights(bundle_dir, manifest, policy)
    worst = _grad_worst_down(weights, target_weights,
                             tt.TRAINING_SIGNAL_ID_2)
    assert worst == pytest.approx(GRAD_WORST_D0001, rel=0.5)
    assert worst <= GRAD_TOL


def test_down_forward_replica_and_fixed_inputs(tmp_path):
    # The Phase-0 derivation's premise, checked: a forward
    # completed from forward_to_final_down's outputs
    # reproduces the library forward's logits bit-identically
    # (after the float32 storage cast both paths apply), and
    # the down inputs do NOT depend on the down factors
    # (folding a candidate into the down tensor changes
    # neither h_mid nor s — s is a fixed input).
    bundle_dir = _seeded(tmp_path)
    policy = _policy(bundle_dir)
    manifest = load_bundle(bundle_dir)
    weights = assemble_weights(bundle_dir, manifest, policy)
    A0, B0 = tt.initial_factors_down(tt.TRAINING_SIGNAL_ID_2)
    folded = tt._folded_weights_down(
        weights, np.ascontiguousarray(A0, dtype=np.float32),
        np.ascontiguousarray(B0, dtype=np.float32))
    for seq, _t in tt.build_training_signal(
            weights, tt.TRAINING_SIGNAL_ID_2)["train"][:4]:
        h_mid, s = tt.forward_to_final_down(seq, weights)
        h_mid_f, s_f = tt.forward_to_final_down(seq, folded)
        assert np.array_equal(h_mid, h_mid_f)
        assert np.array_equal(s, s_f)
        w64 = {k: np.asarray(v, dtype=np.float64)
               for k, v in weights.items()}
        y = h_mid + s @ w64[tt.DOWN_TARGET_NAME].T
        h_final = tt._rms_norm(
            y, w64["model.language_model.norm.weight"])
        logits = (h_final @ w64[tt.TARGET_NAME].T).astype(np.float32)
        assert np.array_equal(logits, forward(seq, weights))


# -- §16.5: default-target reproduction and the at-base records ------------------------

def test_at_base_records(tmp_path):
    # (a) Default-target reproduction at base: signal 2,
    # base-anchored, target "lm-head", at base reproduces
    # the established at-base run EXACTLY.
    bundle_dir = _seeded(tmp_path, "tiny33a")
    policy = _policy(bundle_dir)
    mind, seen = _spying_mind(
        bundle_dir, training_signal_id=tt.TRAINING_SIGNAL_ID_2,
        target_anchoring="base-anchored")
    res = run_cycle(bundle_dir, mind, gate=Gate(policy))
    assert res["verdict"] == "ADMIT" and res["head"] == "d0001"
    assert seen[0]["record"]["blob"] == S2_AT_BASE_BLOB
    assert seen[0]["hyperparams"]["authorship_target"] == "lm-head"
    assert seen[0]["hyperparams"]["target"] == tt.TARGET_NAME
    manifest = load_bundle(bundle_dir)
    checks, _row = _checks_of_edit(bundle_dir, manifest, res["edit_id"])
    assert checks["identity_continuity"]["measured"] == pytest.approx(
        S2_AT_BASE_DRIFT, abs=1e-9)
    assert checks["capability_floor"]["measured"] == pytest.approx(
        S2_AT_BASE_REGRESSION, abs=1e-9)

    # (b) The new-target at-base run: a labelled at-base
    # FACT under its own blob and figures — ADMITted and
    # VERIFY-passed, and not the lm-head run.
    bundle_dir = _seeded(tmp_path, "tiny33b")
    policy = _policy(bundle_dir)
    mind, seen = _spying_mind(
        bundle_dir, training_signal_id=tt.TRAINING_SIGNAL_ID_2,
        target_anchoring="base-anchored",
        authorship_target="final-mlp-down")
    res = run_cycle(bundle_dir, mind, gate=Gate(policy))
    assert res["verdict"] == "ADMIT" and res["head"] == "d0001"
    assert res["verified"] is True and res["rolled_back"] is False
    cand = seen[0]
    assert cand["record"]["blob"] == DOWN_AT_BASE_BLOB
    assert cand["record"]["blob"] != S2_AT_BASE_BLOB
    assert cand["record"]["norm"] == pytest.approx(
        DOWN_AT_BASE_NORM, abs=1e-12)
    assert cand["record"]["targets"][0]["name"] == tt.DOWN_TARGET_NAME
    assert cand["record"]["targets"][0]["rank"] == 8
    hp = cand["hyperparams"]
    assert hp["authorship_target"] == "final-mlp-down"
    assert hp["target"] == tt.DOWN_TARGET_NAME
    assert hp["head"] == "base" and hp["target_source_head"] == "base"
    assert hp["base_held_out_loss"] == pytest.approx(0.00140625,
                                                     abs=1e-12)
    assert hp["held_out_loss_after"] == pytest.approx(
        DOWN_AT_BASE_HELD_AFTER, abs=1e-12)
    assert hp["held_out_loss_after"] < hp["base_held_out_loss"]
    assert hp["held_out_examples_closer"] == 14
    manifest = load_bundle(bundle_dir)
    checks, _row = _checks_of_edit(bundle_dir, manifest, res["edit_id"])
    assert checks["identity_continuity"]["measured"] == pytest.approx(
        DOWN_AT_BASE_DRIFT, abs=1e-9)
    assert checks["capability_floor"]["measured"] == pytest.approx(
        DOWN_AT_BASE_REGRESSION, abs=1e-9)
    ok, report = _verify_probe_head(bundle_dir, manifest, policy=policy)
    assert ok is True
    assert report["identity"]["max_drift"] == pytest.approx(
        DOWN_AT_BASE_DRIFT, abs=1e-9)


# -- §16.4: probe-blindness of training, asserted ---------------------------------------

def test_down_training_is_probe_blind(tmp_path, monkeypatch):
    bundle_dir, policy = _land_d0001(tmp_path)
    train_seqs, held_seqs, _ = tt.signal_sequences(
        tt.TRAINING_SIGNAL_ID_2)
    train_set = {tuple(s) for s in train_seqs}
    assert train_set.isdisjoint({tuple(s) for s in identity_inputs()})
    assert train_set.isdisjoint({tuple(s) for s in capability_inputs()})
    assert train_set.isdisjoint({tuple(s) for s in held_seqs})
    captured_inputs = []
    orig_inputs = tt.down_inputs_for

    def spy_inputs(weights, seqs):
        captured_inputs.append({tuple(s) for s in seqs})
        return orig_inputs(weights, seqs)

    monkeypatch.setattr(tt, "down_inputs_for", spy_inputs)
    captured_examples = []
    orig_grad = tt.analytic_gradients_down

    def spy_grad(A, B, examples, down_inputs, weights):
        captured_examples.append({tuple(s) for s, _ in examples})
        return orig_grad(A, B, examples, down_inputs, weights)

    monkeypatch.setattr(tt, "analytic_gradients_down", spy_grad)
    mind = TinyTransformerMind(
        bundle_dir, load_bundle(bundle_dir),
        training_signal_id=tt.TRAINING_SIGNAL_ID_2,
        target_anchoring="base-anchored",
        authorship_target="final-mlp-down")
    cand = mind.train_delta(mind.propose([], None))
    assert cand["record"]["blob"] == DOWN_BLOB
    assert captured_inputs and captured_examples
    for seq_set in captured_inputs:
        assert seq_set == train_set
    for ex_set in captured_examples:
        assert ex_set == train_set
    monkeypatch.undo()
    import shutil
    copy_dir = str(tmp_path / "perturbed.si")
    shutil.copytree(bundle_dir, copy_dir)
    manifest = load_bundle(copy_dir)
    probe_path = os.path.join(copy_dir, "probe_suite.json")
    with open(probe_path, "rb") as f:
        artifact = json.loads(f.read().decode("utf-8"))
    for entry in artifact["identity_suite"] + artifact["capability_suite"]:
        entry["input"] = [0, 1, 2]
        entry["expected_logits"] = [0.0] * 32
    with open(probe_path, "wb") as f:
        f.write(json.dumps(artifact).encode("utf-8"))
    intent = Intent(
        kind=KIND, reason="SI-33 probe-blindness probe",
        parents=["d0001"], source_seqs=[1],
        training_signal_id=tt.TRAINING_SIGNAL_ID_2,
        target_anchoring="base-anchored",
        composition_config="none",
        authorship_target="final-mlp-down",
        targets=[tt.DOWN_TARGET_NAME])
    cand2 = tt.train_candidate(copy_dir, manifest, intent,
                               policy=policy)
    assert cand2["blob_bytes"] == cand["blob_bytes"]
    assert cand2["record"]["blob"] == DOWN_BLOB


# -- §16.9: all four controls, re-run inside the slice -----------------------------------

def _assert_control(tmp_path, name, comp, blob, drift, regression):
    bundle_dir, policy, mind, seen, sum1, sum2, _pre = \
        _run_with_cycle2(tmp_path, name, comp=comp)
    assert sum1["admitted"] == 1 and sum2["rejected"] == 1
    hp2 = seen[1]["hyperparams"]
    assert hp2["authorship_target"] == "lm-head"
    assert hp2["target"] == tt.TARGET_NAME
    assert seen[1]["record"]["blob"] == blob
    manifest = load_bundle(bundle_dir)
    checks, _row = _checks_of_edit(bundle_dir, manifest,
                                   sum2["edit_ids"][0])
    assert checks["identity_continuity"]["measured"] == \
        pytest.approx(drift, abs=1e-9)
    assert checks["capability_floor"]["measured"] == \
        pytest.approx(regression, abs=1e-9)
    return seen


def test_none_control_rerun_reproduces_si25(tmp_path):
    seen = _assert_control(tmp_path, "tiny33c25", None, CTRL_BLOB,
                           CTRL_DRIFT, CTRL_REGRESSION)
    assert seen[1]["record"]["norm"] == pytest.approx(CTRL_NORM,
                                                      abs=1e-12)
    assert seen[1]["hyperparams"]["held_out_loss_after"] == \
        pytest.approx(CTRL_HELD_AFTER, abs=1e-12)


def test_movement_control_rerun_reproduces_si27(tmp_path):
    _assert_control(tmp_path, "tiny33c27", "movement-composition",
                    PEN_BLOB, PEN_DRIFT, 0.14536726917140186)


def test_worst_control_rerun_reproduces_si29(tmp_path):
    _assert_control(tmp_path, "tiny33c29", "worst-input-composition",
                    WORST_BLOB, WORST_DRIFT, 0.11724743689410388)


def test_projection_control_rerun_reproduces_si31(tmp_path):
    seen = _assert_control(tmp_path, "tiny33c31",
                           "train-partition-projection", PROJ_BLOB,
                           PROJ_DRIFT, 0.177281160838902)
    assert seen[1]["record"]["targets"][0]["rank"] == 16


# -- the decisive run: §16.1/§16.7 --------------------------------------------------------

def test_decisive_two_cycle_run(tmp_path):
    bundle_dir, policy, mind, seen, sum1, sum2, pre_manifest = \
        _decisive_run(tmp_path)
    assert sum1["admitted"] == 1 and sum1["head"] == "d0001"
    # Cycle 2's verdict AS MEASURED: ADMIT — the first
    # authored two-delta stack. No ADMIT was required.
    assert sum2["admitted"] == 1 and sum2["halted"] is False
    assert sum2["head"] == "d0002" and sum2["delta_stack_depth"] == 2
    assert len(seen) == 2

    hp1, hp2 = seen[0]["hyperparams"], seen[1]["hyperparams"]
    assert hp1["authorship_target"] == "lm-head"
    assert seen[0]["record"]["blob"] == SI19_BLOB
    assert hp2["training_signal_id"] == tt.TRAINING_SIGNAL_ID_2
    assert hp2["target_anchoring"] == "base-anchored"
    assert hp2["composition_config"] == "none"
    assert hp2["authorship_target"] == "final-mlp-down"
    assert hp2["target"] == tt.DOWN_TARGET_NAME
    assert hp2["head"] == "d0001" and hp2["target_source_head"] == "base"
    assert hp2["stack_depth"] == 1
    assert seen[1]["record"]["parent"] == "d0001"
    assert seen[1]["record"]["id"] == "d0002"
    assert seen[1]["record"]["blob"] == DOWN_BLOB
    assert seen[1]["record"]["bytes"] == DOWN_BYTES
    assert seen[1]["record"]["norm"] == pytest.approx(DOWN_NORM,
                                                      abs=1e-12)
    target = seen[1]["record"]["targets"][0]
    assert target["name"] == tt.DOWN_TARGET_NAME
    assert target["rank"] == 8 and target["alpha"] == 8
    assert hp2["base_held_out_loss"] == pytest.approx(DOWN_BASE_HELD,
                                                      abs=1e-12)
    assert hp2["held_out_loss_after"] == pytest.approx(DOWN_HELD_AFTER,
                                                       abs=1e-12)
    assert hp2["held_out_loss_after"] < hp2["base_held_out_loss"]
    assert hp2["held_out_examples_closer"] == 16
    assert hp2["held_out_examples"] == 16
    assert hp2["norm_projection_activations"] == 0
    assert hp2["penalty_train_initial"] is None
    assert hp2["worst_input_train_initial"] is None
    assert hp2["projection_alpha"] is None

    manifest = load_bundle(bundle_dir)
    assert manifest["head"] == "d0002"
    assert [d["id"] for d in manifest["deltas"]] == ["d0001", "d0002"]
    edit1, edit2 = sum1["edit_ids"][0], sum2["edit_ids"][0]
    # Gate honesty (§16.7): an independent ProbeContext
    # evaluation of the same new-target candidate against
    # the PRE-CYCLE head (d0001) reproduces the journaled
    # checks 3/4 exactly, and the same verdict.
    # NOTE: unlike SI-31 (whose cycle 2 REJECTED, leaving
    # the head at d0001 for a retrain to reproduce the
    # candidate), this cycle ADMITTED — the bundle head is
    # d0002, so a fresh train would train a third cycle.
    # Gate honesty therefore evaluates the ORIGINAL
    # candidate (seen[1], the only candidate the gate was
    # ever shown) against the pre-cycle head directly.
    fresh = TinyTransformerMind(
        bundle_dir, load_bundle(bundle_dir),
        training_signal_id=tt.TRAINING_SIGNAL_ID_2,
        target_anchoring="base-anchored",
        authorship_target="final-mlp-down")
    cand2 = seen[1]
    ctx = ProbeContext(bundle_dir, pre_manifest, policy=policy)
    independent = Gate(policy).evaluate(fresh.propose([], ctx), cand2, ctx)
    assert independent.verdict == "ADMIT"
    ind = {c.name: c for c in independent.checks}
    journaled, edit_row = _checks_of_edit(bundle_dir, manifest, edit2)
    assert edit_row["result"] == "materialized"
    for name in ("identity_continuity", "capability_floor"):
        assert journaled[name]["measured"] == pytest.approx(
            ind[name].measured, abs=1e-12)
    assert journaled["identity_continuity"]["measured"] == \
        pytest.approx(DOWN_DRIFT, abs=1e-9)
    assert journaled["identity_continuity"]["passed"] is True
    assert journaled["capability_floor"]["measured"] == \
        pytest.approx(DOWN_REGRESSION, abs=1e-9)
    assert journaled["capability_floor"]["passed"] is True
    assert journaled["safety_bounds"]["passed"] is True
    # The journal shows all four recorded selections.
    note1 = _hyperparams_notes(bundle_dir, manifest, "d0001")
    assert note1["authorship_target"] == "lm-head"
    # An ADMITTED candidate's hyperparams note is filed
    # under its delta id (a REJECTED candidate's, as in
    # SI-31, under its self-edit id).
    note2 = _hyperparams_notes(bundle_dir, manifest, "d0002")
    assert note2["training_signal_id"] == tt.TRAINING_SIGNAL_ID_2
    assert note2["target_anchoring"] == "base-anchored"
    assert note2["composition_config"] == "none"
    assert note2["authorship_target"] == "final-mlp-down"
    assert note2["head"] == "d0001"
    edits = _journal_rows(bundle_dir, manifest, "self_edits",
                          edit_id=edit2)
    assert edits[0]["intent"]["authorship_target"] == "final-mlp-down"
    # VERIFY of the landed two-delta head (§16.10).
    ok, report = _verify_probe_head(bundle_dir, manifest, policy=policy)
    assert ok is True
    assert report["identity"]["max_drift"] == pytest.approx(
        DOWN_DRIFT, abs=1e-9)
    verify_rows = [p for p in _journal_rows(
        bundle_dir, manifest, "provenance")
        if p["relation"] == "transformer-probe-verify"
        and p["child"] == "d0002"]
    assert len(verify_rows) == 1
    vnote = json.loads(verify_rows[0]["note"])
    assert vnote["identity"]["label"] == "measured-pass"
    assert vnote["capability"]["label"] == "measured-pass"


# -- §16.8: probe diagnostics, seven-run table, F1'''/F2'''/F3''' -------------------------

def test_probe_diagnostics_seven_run_and_falsification(tmp_path):
    bundle_dir, policy, mind, seen, sum1, sum2, pre_manifest = \
        _decisive_run(tmp_path)
    manifest = load_bundle(bundle_dir)
    assert manifest["head"] == "d0002"
    ctx = ProbeContext(bundle_dir, pre_manifest, policy=policy)
    cand2 = seen[1]
    expected = {
        "tiny-id-0": (0.17189218290150166, 0.0014927331358194351, 0.17195402272045612),
        "tiny-id-1": (0.15088725090026855, 0.0010822396725416183, 0.1509101390838623),
        "tiny-id-2": (0.11858986946754158, 0.0008184760808944702, 0.11833662423305213),
        "tiny-id-3": (0.14782869815826416, 0.0015081502497196198, 0.14783048629760742),
        "tiny-id-4": (0.11720322235487401, 0.002182206604629755, 0.11517882510088384),
        "tiny-id-5": (0.09891846403479576, 0.0009624920785427094, 0.09938219469040632),
        "tiny-id-6": (0.12363102938979864, 0.0018341410905122757, 0.12295627500861883),
        "tiny-id-7": (0.08795571327209473, 0.0010801851749420166, 0.08804309368133545),
    }
    m1_all, m2_all = [], []
    rows = {}
    inners = {}
    for qid, x, exp in ctx.identity_probes():
        f_head = np.asarray(ctx.forward_head(x), dtype=np.float64)
        f_comp = np.asarray(ctx.forward_candidate(x, cand2),
                            dtype=np.float64)
        e = np.asarray(exp, dtype=np.float64)
        m1, m2 = f_head - e, f_comp - f_head
        m1_all.append(m1); m2_all.append(m2)
        rows[qid] = (float(np.max(np.abs(m1))),
                     float(np.max(np.abs(m2))),
                     float(np.max(np.abs(f_comp - e))))
        inners[qid] = float(np.dot(m1, m2))
    for qid, want in expected.items():
        for g, w in zip(rows[qid], want):
            assert g == pytest.approx(w, abs=1e-9)
    assert max(rows, key=lambda q: rows[q][2]) == "tiny-id-0"
    # Seven-run composed maxima, each under its own labels:
    # SI-21 0.34378, SI-23 0.29634, SI-25 0.33290, SI-27
    # 0.32509, SI-29 0.31216, SI-31 0.47754 — and SI-33's
    # treatment (signal 2, base-anchored, composition
    # "none", authorship target "final-mlp-down")
    # 0.17195402272045612, BELOW tol 0.25 — the first
    # cycle-2 composed maximum under tolerance on record.
    assert PROJ_DRIFT > S21_DRIFT > CTRL_DRIFT > PEN_DRIFT > \
        WORST_DRIFT > S23_DRIFT > 0.25 > DOWN_DRIFT
    a_all = np.concatenate(m1_all); b_all = np.concatenate(m2_all)
    agg_inner = float(np.dot(a_all, b_all))
    assert agg_inner == pytest.approx(-0.002171869843126004, abs=1e-9)
    assert float(np.mean(np.sign(a_all) == np.sign(b_all))) == \
        pytest.approx(0.4609375, abs=1e-12)
    assert inners["tiny-id-0"] == pytest.approx(DOWN_INNER_AT_ID0,
                                                abs=1e-9)
    assert float(np.linalg.norm(b_all)) == pytest.approx(
        0.010469039280455053, abs=1e-9)
    assert float(np.linalg.norm(a_all)) == pytest.approx(
        0.7119678353441855, abs=1e-9)
    # Folded updates (weight-space figures under the new
    # target's labels; NO readout-fold derivation is claimed
    # for them): the treatment's folded down-projection
    # update has the candidate's norm; it lives on a
    # DIFFERENT tensor than cycle 1's folded update, so no
    # folded inner product between them exists to report.
    parsed = parse_blob(cand2["blob_bytes"])
    t2 = cand2["record"]["targets"][0]
    A2 = np.asarray(parsed.tensors[t2["factor_keys"][0]], dtype=np.float64)
    B2 = np.asarray(parsed.tensors[t2["factor_keys"][1]], dtype=np.float64)
    d2u = (float(t2["alpha"]) / float(t2["rank"])) * (B2 @ A2)
    assert d2u.shape == (32, 64)
    assert float(np.linalg.norm(d2u)) == pytest.approx(DOWN_NORM,
                                                       abs=1e-9)
    assert seen[0]["record"]["targets"][0]["name"] != t2["name"]
    headroom = 0.25 - SI19_DRIFT
    assert sum(1 for v in rows.values() if v[1] > headroom) == 0
    # Allowed-input composition records (treatment, under
    # its own labels): aggregate inner products on both
    # partitions are ~500x smaller in magnitude than the
    # SI-25 control's, and ‖m2‖ beside them.
    head_w = assemble_weights(bundle_dir, pre_manifest, policy)
    base_w = assemble_base_weights(bundle_dir, pre_manifest, policy)
    cand_w = assemble_weights_with_candidate(
        bundle_dir, pre_manifest, cand2, policy)
    train_seqs, held_seqs, _ = tt.signal_sequences(
        tt.TRAINING_SIGNAL_ID_2)
    rec_tr = tt.movement_composition_record(head_w, base_w,
                                            cand_w, train_seqs)
    rec_ho = tt.movement_composition_record(head_w, base_w,
                                            cand_w, held_seqs)
    assert rec_tr["aggregate_inner_product"] == pytest.approx(
        -0.0068245467494126594, abs=1e-9)
    assert rec_tr["m2_norm"] == pytest.approx(0.019935871646549735,
                                              abs=1e-9)
    assert rec_ho["aggregate_inner_product"] == pytest.approx(
        -0.0034804084170157222, abs=1e-9)
    # Worst ALLOWED inputs, named by train-partition
    # identity (never confused with probes): train index
    # 16, held-out index 0 — both with |p| ~ 2e-06.
    wi_tr = max(range(len(train_seqs)),
                key=lambda i: rec_tr["per_input"][i]["p"])
    assert wi_tr == 16 and train_seqs[16] == [8, 18, 3, 30]
    wi_ho = max(range(len(held_seqs)),
                key=lambda i: rec_ho["per_input"][i]["p"])
    assert wi_ho == 0 and held_seqs[0] == [25, 30]
    # F1''' NOT measured — its first clause fails: the
    # composed maximum does NOT stay above tol (and the
    # inner product at tiny-id-0 is negative, opposing).
    assert not (DOWN_DRIFT > 0.25 and inners["tiny-id-0"] > 0.0)
    # F2''' judged as TWO SEPARATE clauses: (a) the
    # transfer clause HOLDS — the treatment's inner product
    # at its own worst probe (tiny-id-0, −0.000311…) is
    # smaller than the SI-25 control's inner product at
    # that same probe (+0.0338873784299374, Phase 0);
    # (b) the maximum clause FAILS — the composed maximum
    # does not stay above tol (0.17195… ≤ 0.25). F2''' as
    # a reading is therefore NOT measured.
    assert inners["tiny-id-0"] < CTRL_INNER_AT_ID0
    assert not (DOWN_DRIFT > 0.25)
    # F3''' read under the PRE-REGISTERED rule (Phase 0):
    # approached iff (a) held-out loss falls, (b) at least
    # 12/16 held-out examples are closer, and (c) the
    # held-out residual is at most 2× the re-run SI-25
    # control's residual (2.087217177504918e-05). (a) and
    # (b) hold; (c) FAILS by ~131x — so F3''' IS measured:
    # the destination counts as traded away under the rule.
    hp2 = seen[1]["hyperparams"]
    assert hp2["held_out_loss_after"] < hp2["base_held_out_loss"]
    assert hp2["held_out_examples_closer"] == 16 >= 12
    assert hp2["held_out_loss_after"] > F3PPP_BOUNDARY
    rep = Gate(policy).evaluate(mind.propose([], ctx), cand2, ctx)
    checks = {c.name: c for c in rep.checks}
    assert rep.verdict == "ADMIT"
    assert checks["capability_floor"].measured == pytest.approx(
        DOWN_REGRESSION, abs=1e-9)
    targets = ctx.capability_targets()
    correct = sum(
        1 for cid, x, _e in ctx.capability_suite()
        if int(np.argmax(np.asarray(ctx.forward_candidate(x, cand2))))
        == int(targets[cid]))
    assert correct == 16 and ctx.echo_floor() == 0.9375


# -- §16.2/§16.8: Phase-0 control records, re-measured -------------------------------------

def test_phase0_si25_control_records(tmp_path):
    # The Phase-0 records the readings need, re-measured on
    # the re-run SI-25 control (an already-judged candidate;
    # these are labelled DIAGNOSTICS on it): its per-probe
    # inner products with m1 (all 8 identity probes, through
    # candidate overlays) and its allowed-input composition
    # record under its own labels.
    bundle_dir, policy, mind, seen, sum1, sum2, pre_manifest = \
        _run_with_cycle2(tmp_path, "tiny33p0", comp=None)
    assert seen[1]["record"]["blob"] == CTRL_BLOB
    ctx = ProbeContext(bundle_dir, pre_manifest, policy=policy)
    cand = seen[1]
    inners = {}
    m1_all, m2_all = [], []
    for qid, x, exp in ctx.identity_probes():
        fh = np.asarray(ctx.forward_head(x), dtype=np.float64)
        fc = np.asarray(ctx.forward_candidate(x, cand), dtype=np.float64)
        e = np.asarray(exp, dtype=np.float64)
        m1, m2 = fh - e, fc - fh
        m1_all.append(m1); m2_all.append(m2)
        inners[qid] = float(np.dot(m1, m2))
    assert inners == pytest.approx({
        "tiny-id-0": 0.0338873784299374,
        "tiny-id-1": -0.017059672853274165,
        "tiny-id-2": -0.07934423600975443,
        "tiny-id-3": -0.02224599986114801,
        "tiny-id-4": -0.05053692823121107,
        "tiny-id-5": -0.03510892397145824,
        "tiny-id-6": -0.05115886243681786,
        "tiny-id-7": 0.009004939760844474}, abs=1e-9)
    a_all = np.concatenate(m1_all); b_all = np.concatenate(m2_all)
    assert float(np.dot(a_all, b_all)) == pytest.approx(
        -0.21256230517288188, abs=1e-9)
    assert float(np.linalg.norm(b_all)) == pytest.approx(
        0.7816722548913912, abs=1e-9)
    head_w = assemble_weights(bundle_dir, pre_manifest, policy)
    base_w = assemble_base_weights(bundle_dir, pre_manifest, policy)
    cand_w = assemble_weights_with_candidate(
        bundle_dir, pre_manifest, cand, policy)
    train_seqs, held_seqs, _ = tt.signal_sequences(
        tt.TRAINING_SIGNAL_ID_2)
    rec_tr = tt.movement_composition_record(head_w, base_w,
                                            cand_w, train_seqs)
    rec_ho = tt.movement_composition_record(head_w, base_w,
                                            cand_w, held_seqs)
    assert rec_tr["aggregate_inner_product"] == pytest.approx(
        -1.6873837019991191, abs=1e-9)
    assert rec_tr["mean_p"] == pytest.approx(
        -0.0016478356464835148, abs=1e-12)
    assert rec_ho["aggregate_inner_product"] == pytest.approx(
        -0.845596374259425, abs=1e-9)
    assert rec_ho["mean_p"] == pytest.approx(
        -0.0016515554184754395, abs=1e-12)


# -- §16.6: determinism per cycle -----------------------------------------------------------

def test_decisive_run_repeats_exactly(tmp_path):
    _b1, _p1, _m1, seen1, s1a, s2a, _x1 = _decisive_run(
        tmp_path / "a", "tiny33")
    _b2, _p2, _m2, seen2, s1b, s2b, _x2 = _decisive_run(
        tmp_path / "b", "tiny33")
    assert s1a == s1b and s2a == s2b
    assert load_bundle(_b1)["deltas"] == load_bundle(_b2)["deltas"]
    for cand_a, cand_b in zip(seen1, seen2):
        assert cand_a["blob_bytes"] == cand_b["blob_bytes"]
        assert cand_a["record"] == cand_b["record"]
        hp_a = dict(cand_a["hyperparams"])
        hp_b = dict(cand_b["hyperparams"])
        hp_a.pop("wall_time_seconds"); hp_b.pop("wall_time_seconds")
        assert hp_a == hp_b


def test_down_train_delta_deterministic(tmp_path):
    bundle_dir, policy = _land_d0001(tmp_path)
    mind = TinyTransformerMind(
        bundle_dir, load_bundle(bundle_dir),
        training_signal_id=tt.TRAINING_SIGNAL_ID_2,
        target_anchoring="base-anchored",
        authorship_target="final-mlp-down")
    intent = mind.propose([], None)
    first = mind.train_delta(intent)
    second = mind.train_delta(intent)
    assert first["blob_bytes"] == second["blob_bytes"]
    assert first["record"] == second["record"]
    assert first["record"]["blob"] == DOWN_BLOB


# -- §16.10: VERIFY / unwind honesty / untouched scope ---------------------------------------

def test_unwind_and_untouched_scope(tmp_path):
    bundle_dir, policy, mind, seen, sum1, sum2, _pre = \
        _decisive_run(tmp_path)
    assert sum1["admitted"] == 1 and sum2["admitted"] == 1
    manifest = load_bundle(bundle_dir)
    assert manifest["head"] == "d0002"
    assert len(manifest["deltas"]) == 2
    # SI-38 note (this assertion superseded): SI-12's blanket
    # SquashError refusal was replaced by RFC SI-37's designed
    # squash pipeline. On this stack the pipeline now runs and,
    # under the bundle's OWN policy (empty allowlist), the gate
    # REJECTS the composite on the allowlist line — nothing
    # lands: no new delta, no superseded marks, head unmoved.
    # (Under the driver policy this suite lands cycles with,
    # the same squash ADMITs; tests/test_transformer_squash.py
    # covers both arms.)
    res = squash_bundle(bundle_dir, keep=0)
    assert res["squashed"] is False and res["verdict"] == "REJECT"
    manifest = load_bundle(bundle_dir)
    assert manifest["head"] == "d0002"
    assert len(manifest["deltas"]) == 2
    assert all("superseded_by" not in d for d in manifest["deltas"])
    # The authored two-delta stack EXISTS for the first
    # time (cycle 1 lm-head + cycle 2 final-mlp-down), so
    # the unwind is the real thing, step by step through
    # the existing rollback core — not a fixture.
    out = rollback(bundle_dir, to="d0001",
                   reason="SI-33 unwind proof")
    assert out["head"] == "d0001"
    manifest = load_bundle(bundle_dir)
    assert manifest["head"] == "d0001"
    ok, report = _verify_probe_head(bundle_dir, manifest, policy=policy)
    assert ok is True
    assert report["identity"]["max_drift"] == pytest.approx(
        SI19_DRIFT, abs=1e-9)
    out = rollback(bundle_dir, to="base",
                   reason="SI-33 unwind proof")
    assert out["head"] == "base"
    assert load_bundle(bundle_dir)["head"] == "base"
    # Untouched scope.
    assert DEFAULT_POLICY["max_stack_depth"] == 8
    assert DEFAULT_POLICY["identity_tol"] == 0.25
    assert DEFAULT_POLICY["capability_eps"] == 0.25
    assert DEFAULT_POLICY["max_targets_per_delta"] == 1
    assert DEFAULT_POLICY["allow_additive"] is False
    assert DEFAULT_POLICY["rank_cap"] == 8
    on_disk = load_policy(bundle_dir)
    assert on_disk["identity_tol"] == 0.25
    assert on_disk["capability_eps"] == 0.25
    assert on_disk["rank_cap"] == 8
    assert tt.TARGET_ANCHORING == "weights-under-training"
    assert tt.TARGET_ANCHORINGS == ("weights-under-training",
                                    "base-anchored")
    assert set(tt.SIGNAL_REGISTRY) == {tt.TRAINING_SIGNAL_ID,
                                       tt.TRAINING_SIGNAL_ID_2}
    assert tt.CORRECTION_SCALE == 0.15
    assert tt.N_TRAIN == 32 and tt.N_HELD_OUT == 16
    assert tt.RANK == 8 and tt.ALPHA == 8
    assert tt.COMPOSITION_LAMBDA == 1.0
    assert tt.WORST_INPUT_LAMBDA == 1.0 and tt.WORST_INPUT_K == 4
    assert copy.deepcopy(tt.COMPOSITION_CONFIGS) == \
        ("none", "movement-composition", "worst-input-composition",
         "train-partition-projection")
    assert copy.deepcopy(tt.AUTHORSHIP_TARGETS) == \
        ("lm-head", "final-mlp-down")
