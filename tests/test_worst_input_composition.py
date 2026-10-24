"""SI-29: worst-input composition-aware training for the
miniature (RFC SI-28 §14 outline, §15 acceptance criteria).

Covers the §15 criteria, in order:

1. recorded composition configuration, three named values —
   the configuration serves exactly ``"none"``,
   ``"movement-composition"``, and
   ``"worst-input-composition"`` and refuses every other
   value with the existing ``ValueError`` discipline
   (constructor, setter, intent, and ``train_candidate``
   dispatch); recorded selection asserted per cycle on both
   verdict paths, including the decisive cycle's full triple
   (signal 2, ``target_anchoring = "base-anchored"``,
   composition = ``"worst-input-composition"``). λ_w = 1.0
   and k = 4 are the configuration's definition, recorded in
   hyperparams, NOT independently settable (asserted: the
   constructor and the setter take no such argument);
   exactly one worst-input decisive run is performed (no
   sweep of λ_w, k, or τ);
2. Phase 0 — published the per-input composition record
   for the SI-25 control and the SI-27 treatment on both
   allowed partitions before the decisive run, stated the
   top-k form choice with its argument, and pre-registered
   the F3 reading rule (as-built doc, ahead of the decisive
   numbers; the record values themselves are re-measured
   and asserted in the allowed-input record test below);
3. gradient check for the worst-input loss — central finite
   differences at head ``d0001`` AND at base under the new
   code path, over EVERY one of the 512 factor entries,
   tolerance 1e-5 unchanged, with top-k membership
   stability at the check points reported (asserted on the
   8-example check subset: the top-k set is recomputed
   under every single-entry ±eps perturbation);
4. probe-blindness of training, asserted — a spy on the
   worst-input gradient path proves the term's input set
   is exactly signal 2's train partition; disjointness from
   both probe suites re-asserted; a perturbed-probe-
   artifact bundle copy produces a byte-identical
   candidate;
5. coincidence at base — signal 2, base-anchored,
   worst-input at base reproduces the established at-base
   run exactly (blob ``sha256:131cc242…45f4``, drift
   0.16672966815531254, regression 0.2418281203135848),
   with the new term measured identically zero at
   initialisation and at the end of training on both
   partitions;
6. determinism per cycle — a full decisive run repeated
   from the seed reproduces both candidates exactly
   (hyperparams equal except the labelled wall-time
   field); the worst-input ``train_delta`` alone is
   likewise byte-identical across repeats;
7. gate honesty per cycle — the decisive cycle 2's
   journaled checks 3/4 equal an independent
   ``ProbeContext`` evaluation of the same candidate
   against ``d0001`` (abs 1e-12). The verdict is REJECT
   (cumulative identity drift 0.31215621903538704 > tol
   0.25), reported AS MEASURED. No ADMIT is required;
8. cumulative accounting, diagnostics, the four-run
   table, and the F1′/F2′/F3′ readings — per-probe drift
   for the treatment beside SI-23, SI-25, and SI-27,
   alignment and folded-update diagnostics, candidate
   norm and ‖m2‖ beside the drift figures, the
   allowed-input worst-input records, and the treatment's
   signal held-out record read under the PRE-REGISTERED
   F3 rule (stated in the as-built doc's Phase 0 section:
   the destination counts as approached iff the held-out
   loss falls, at least 12/16 held-out examples are
   closer, and the held-out residual is at most 2× the
   re-run SI-27 control's residual);
9. first-cycle and both-control reproduction — cycle 1
   reproduces SI-19/21/23/25/27 exactly; the re-run
   ``"none"`` control reproduces SI-25's decisive
   candidate exactly; the re-run
   ``"movement-composition"`` control reproduces SI-27's
   decisive candidate exactly;
10. VERIFY, unwind, and untouched scope — cycle-1 VERIFY
    as in SI-19; cycle 2 never landed (REJECT), so no
    authored two-delta stack exists and none is claimed;
    exactly two authored cycles attempted; policy
    defaults, both signals', both anchorings', and both
    prior composition configurations' definitions
    unchanged; ``si squash`` still fails closed.

The decisive run's driver shape: ONE ``TinyTransformerMind``
reused through ``si.run.run_cycles`` — cycle 1 (signal 1,
default anchoring, default composition), then the driver
explicitly reconfigures the same mind to signal 2
(``set_training_signal``), to base-anchored targets
(``set_target_anchoring``), AND to the worst-input
composition configuration (``set_composition_config`` —
three recorded selections), then cycle 2.
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
from si.loop import _verify_probe_head, run_cycle
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

PEN_BLOB = "sha256:1bb277ab40b9aa53a6218d971e46f29b261ba2bb5f7932ffb98e3951f372b637"
PEN_NORM = 0.2969693161862222
PEN_DRIFT = 0.32509003952145576
PEN_REGRESSION = 0.14536726917140186
PEN_HELD_AFTER = 0.0005227812955486338
PEN_WORST_PROBE_INNER = 0.03801222744105431  # at tiny-id-0
PEN_AGG_INNER = -0.3271477846470207

S23_DRIFT = 0.2963416576385498
S23_COMPOSED = {
    "tiny-id-0": 0.2092923242598772,
    "tiny-id-1": 0.15134692192077637,
    "tiny-id-2": 0.14906740188598633,
    "tiny-id-3": 0.2963416576385498,
    "tiny-id-4": 0.23394920490682125,
    "tiny-id-5": 0.14004023000597954,
    "tiny-id-6": 0.1332004750147462,
    "tiny-id-7": 0.2018739078193903,
}
S25_COMPOSED = {
    "tiny-id-0": 0.3329036422073841,
    "tiny-id-1": 0.09286642074584961,
    "tiny-id-2": 0.14936351776123047,
    "tiny-id-3": 0.1489148810505867,
    "tiny-id-4": 0.14728415571153164,
    "tiny-id-5": 0.11245626583695412,
    "tiny-id-6": 0.14491976890712976,
    "tiny-id-7": 0.14189285784959793,
}
S27_COMPOSED = {
    "tiny-id-0": 0.32509003952145576,
    "tiny-id-1": 0.09391975402832031,
    "tiny-id-2": 0.1488508302718401,
    "tiny-id-3": 0.14755446463823318,
    "tiny-id-4": 0.10617947578430176,
    "tiny-id-5": 0.1450248844921589,
    "tiny-id-6": 0.15063166618347168,
    "tiny-id-7": 0.18760260567069054,
}

# SI-29 measured values (author's runs; the as-built doc
# publishes them with their labels). The decisive cycle-2
# verdict is REJECT — measured, not required, not tuned for.
WORST_BLOB = "sha256:11cab02e560e3661ea419fa0c8b6a95de25d3c7fb1ee23ec52ec8462a36b8db5"
WORST_NORM = 0.28252038537434354
WORST_DRIFT = 0.31215621903538704
WORST_REGRESSION = 0.11724743689410388
WORST_TRAIN_INITIAL = 0.0004697766937776053
WORST_TRAIN_FINAL = -0.0023225615446231264
WORST_HELD_INITIAL = 0.00037773328307693754
WORST_HELD_FINAL = -0.0023250720244677755
WORST_TOTAL_INITIAL = 0.004769626497450179
WORST_TOTAL_FINAL = -0.001671632314028651
WORST_SIG_TRAIN_FINAL = 0.0006509292305944755
WORST_HELD_AFTER = 0.0006602814776996296
BA_PRE_HELD = 0.002757805049077469
WORST_TOPK_INITIAL = [12, 28, 4, 20]
WORST_TOPK_FINAL = [27, 11, 19, 26]
GRAD_WORST_AT_BASE = 5.5249277028738815e-08
GRAD_WORST_AT_D0001 = 5.13570189768905e-08
GRAD_TOPK_UNSTABLE_AT_BASE = 0
GRAD_TOPK_UNSTABLE_AT_D0001 = 0


def _seeded(tmp_path, name="tiny29"):
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


def _blobs_total(bundle_dir):
    root = os.path.join(bundle_dir, "blobs", "sha256")
    return sum(os.path.getsize(os.path.join(root, f))
               for f in os.listdir(root))


def _run_with_cycle2_config(tmp_path, name, comp_config):
    bundle_dir = _seeded(tmp_path, name)
    policy = _policy(bundle_dir)
    mind, seen = _spying_mind(bundle_dir)
    sum1 = run_cycles(bundle_dir, mind, 1, gate=Gate(policy))
    mind.set_training_signal(tt.TRAINING_SIGNAL_ID_2)
    mind.set_target_anchoring(tt.BASE_ANCHORED)
    if comp_config is not None:
        mind.set_composition_config(comp_config)
    sum2 = run_cycles(bundle_dir, mind, 1, gate=Gate(policy))
    return bundle_dir, policy, mind, seen, sum1, sum2


def _decisive_run(tmp_path, name="tiny29"):
    return _run_with_cycle2_config(
        tmp_path, name, tt.WORST_INPUT_COMPOSITION)


def _control_run(tmp_path, name="tiny29ctrl"):
    return _run_with_cycle2_config(tmp_path, name, None)


def _movement_control_run(tmp_path, name="tiny29pen"):
    return _run_with_cycle2_config(
        tmp_path, name, tt.MOVEMENT_COMPOSITION)


def _land_d0001(tmp_path, name="tiny29"):
    bundle_dir = _seeded(tmp_path, name)
    policy = _policy(bundle_dir)
    mind, _seen = _spying_mind(bundle_dir)
    res = run_cycle(bundle_dir, mind, gate=Gate(policy))
    assert res["verdict"] == "ADMIT" and res["head"] == "d0001"
    return bundle_dir, policy


# -- §15.1: recorded composition configuration, three values ---------------------

def test_worst_configuration_and_refusals(tmp_path):
    assert tt.COMPOSITION_CONFIGS == ("none", "movement-composition",
                                      "worst-input-composition",
                                      "train-partition-projection")  # SI-31: fourth value added
    assert tt.DEFAULT_COMPOSITION_CONFIG == "none"
    assert tt.WORST_INPUT_COMPOSITION == "worst-input-composition"
    assert tt.WORST_INPUT_LAMBDA == 1.0
    assert tt.WORST_INPUT_K == 4
    assert tt.COMPOSITION_LAMBDA == 1.0
    for good in tt.COMPOSITION_CONFIGS:
        assert tt.validate_composition_config(good) == good
    for bad in ("worst", "worst-input", "", "NONE", None, "top-k"):
        with pytest.raises(ValueError,
                           match="unknown composition configuration"):
            tt.validate_composition_config(bad)

    bundle_dir = _seeded(tmp_path)
    mind = TinyTransformerMind(bundle_dir, load_bundle(bundle_dir))
    assert mind.composition_config == "none"
    mind_w = TinyTransformerMind(
        bundle_dir, load_bundle(bundle_dir),
        composition_config="worst-input-composition")
    assert mind_w.composition_config == "worst-input-composition"
    assert mind_w.propose([], None).composition_config == \
        "worst-input-composition"
    with pytest.raises(ValueError,
                       match="unknown composition configuration"):
        TinyTransformerMind(bundle_dir, load_bundle(bundle_dir),
                            composition_config="probe-driven")
    # λ_w and k are NOT independently settable: neither the
    # constructor nor the setter takes such an argument.
    with pytest.raises(TypeError):
        TinyTransformerMind(bundle_dir, load_bundle(bundle_dir),
                            composition_lambda_w=0.5)
    with pytest.raises(TypeError):
        TinyTransformerMind(bundle_dir, load_bundle(bundle_dir),
                            worst_input_k=2)
    with pytest.raises(TypeError):
        TinyTransformerMind(bundle_dir, load_bundle(bundle_dir),
                            composition_lambda=0.5)
    mind.set_composition_config("worst-input-composition")
    assert mind.composition_config == "worst-input-composition"
    mind.set_composition_config("none")
    assert mind.composition_config == "none"
    with pytest.raises(ValueError,
                       match="unknown composition configuration"):
        mind.set_composition_config("learned")
    with pytest.raises(TypeError):
        mind.set_composition_config("worst-input-composition", lam_w=0.5)
    with pytest.raises(TypeError):
        mind.set_composition_config("worst-input-composition", k=2)
    base = dict(kind=KIND, reason="SI-29 refusal probe",
                parents=["base"], source_seqs=[1])
    for bad in ("worst_input_composition", "", "worst-case"):
        with pytest.raises(ValueError,
                           match="unknown composition configuration"):
            mind.train_delta(Intent(
                training_signal_id=tt.TRAINING_SIGNAL_ID,
                composition_config=bad, targets=[tt.TARGET_NAME],
                **base))
    mind.set_training_signal(tt.TRAINING_SIGNAL_ID_2)
    mind.set_target_anchoring("base-anchored")
    mind.set_composition_config("worst-input-composition")
    intent = mind.propose([], None)
    assert intent.training_signal_id == tt.TRAINING_SIGNAL_ID_2
    assert intent.target_anchoring == "base-anchored"
    assert intent.composition_config == "worst-input-composition"
    assert intent.to_dict()["composition_config"] == \
        "worst-input-composition"


# -- §15.5: coincidence at base -----------------------------------------------------

def test_worst_signal2_at_base_coincides(tmp_path):
    bundle_dir = _seeded(tmp_path)
    policy = _policy(bundle_dir)
    before = _blobs_total(bundle_dir)
    mind, seen = _spying_mind(
        bundle_dir, training_signal_id=tt.TRAINING_SIGNAL_ID_2,
        target_anchoring="base-anchored",
        composition_config="worst-input-composition")
    res = run_cycle(bundle_dir, mind, gate=Gate(policy))
    assert res["verdict"] == "ADMIT" and res["head"] == "d0001"
    assert res["verified"] is True and res["rolled_back"] is False
    hp = seen[0]["hyperparams"]
    assert hp["composition_config"] == "worst-input-composition"
    assert hp["composition_lambda_w"] == 1.0
    assert hp["composition_worst_k"] == 4
    assert hp["composition_lambda"] == 0.0
    assert hp["penalty_train_initial"] is None
    # The new term, MEASURED at initialisation and at the end
    # of training, on both partitions: identically zero —
    # m1 is the zero vector at base, so every p is 0.0 and
    # the top-k mean of zeros is 0.0.
    assert hp["worst_input_train_initial"] == 0.0
    assert hp["worst_input_train_final"] == 0.0
    assert hp["worst_input_held_out_initial"] == 0.0
    assert hp["worst_input_held_out_final"] == 0.0
    assert hp["total_train_loss_initial"] == hp["train_loss_initial"]
    assert hp["total_train_loss_final"] == hp["train_loss_final"]
    assert seen[0]["record"]["blob"] == S2_AT_BASE_BLOB
    manifest = load_bundle(bundle_dir)
    assert _blobs_total(bundle_dir) == before + 2360
    checks, _row = _checks_of_edit(bundle_dir, manifest, res["edit_id"])
    assert checks["identity_continuity"]["measured"] == pytest.approx(
        S2_AT_BASE_DRIFT, abs=1e-9)
    assert checks["capability_floor"]["measured"] == pytest.approx(
        S2_AT_BASE_REGRESSION, abs=1e-9)
    note = _hyperparams_notes(bundle_dir, manifest, "d0001")
    assert note["composition_config"] == "worst-input-composition"
    assert note["composition_lambda_w"] == 1.0
    assert note["composition_worst_k"] == 4
    ok, report = _verify_probe_head(bundle_dir, manifest, policy=policy)
    assert ok is True
    assert report["identity"]["max_drift"] == pytest.approx(
        S2_AT_BASE_DRIFT, abs=1e-9)


# -- §15.3: gradient checks for the worst-input loss ---------------------------------

def _grad_check_worst(head_weights, base_weights, signal_id):
    """(worst FD disagreement, top-k set, unstable count) for
    signal 2's BASE-ANCHORED, WORST-INPUT loss on an
    8-example subset at the pinned initial factors: central
    finite differences of ``worst_input_factor_loss`` over
    EVERY factor entry, plus top-k membership stability —
    the analytic top-k set recomputed under every
    single-entry ±eps perturbation (1024 perturbations)."""
    signal = tt.build_training_signal(base_weights, signal_id)
    subset = signal["train"][:8]
    seqs = [s for s, _ in subset]
    A0, B0 = tt.initial_factors(signal_id)
    lm_head = np.asarray(head_weights[tt.TARGET_NAME], dtype=np.float64)
    hiddens = {tuple(s): tt.final_hidden(head_weights, s)
               for s in seqs}
    m1 = tt.existing_movements(head_weights, base_weights, seqs)
    _tot, _sig, _w, grad_A, grad_B = tt.analytic_gradients_worst_input(
        A0, B0, subset, hiddens, lm_head, m1,
        tt.WORST_INPUT_LAMBDA, tt.WORST_INPUT_K)
    assert np.max(np.abs(grad_A)) > 0.0 or np.max(np.abs(grad_B)) > 0.0
    sel = tt.worst_input_selection(A0, B0, seqs, hiddens, m1,
                                   tt.WORST_INPUT_K)
    base_topk = set(sel["topk_indices"])
    unstable = 0
    for is_a, arr in ((True, A0), (False, B0)):
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for sgn in (1, -1):
                    X, Y = A0.copy(), B0.copy()
                    if is_a:
                        X[i, j] += sgn * GRAD_EPS
                    else:
                        Y[i, j] += sgn * GRAD_EPS
                    s2 = tt.worst_input_selection(
                        X, Y, seqs, hiddens, m1, tt.WORST_INPUT_K)
                    if set(s2["topk_indices"]) != base_topk:
                        unstable += 1
    worst = 0.0
    for i in range(tt.RANK):
        for j in range(32):
            plus = A0.copy(); plus[i, j] += GRAD_EPS
            minus = A0.copy(); minus[i, j] -= GRAD_EPS
            fd = (tt.worst_input_factor_loss(head_weights, base_weights,
                                             plus, B0, subset)
                  - tt.worst_input_factor_loss(head_weights, base_weights,
                                               minus, B0, subset)) \
                / (2 * GRAD_EPS)
            worst = max(worst, abs(fd - grad_A[i, j]))
    for i in range(32):
        for j in range(tt.RANK):
            plus = B0.copy(); plus[i, j] += GRAD_EPS
            minus = B0.copy(); minus[i, j] -= GRAD_EPS
            fd = (tt.worst_input_factor_loss(head_weights, base_weights,
                                             A0, plus, subset)
                  - tt.worst_input_factor_loss(head_weights, base_weights,
                                               A0, minus, subset)) \
                / (2 * GRAD_EPS)
            worst = max(worst, abs(fd - grad_B[i, j]))
    return worst, sel, unstable


def test_gradient_check_worst_at_base(tmp_path):
    bundle_dir = _seeded(tmp_path)
    policy = _policy(bundle_dir)
    manifest = load_bundle(bundle_dir)
    head_w = assemble_weights(bundle_dir, manifest, policy)
    base_w = assemble_base_weights(bundle_dir, manifest, policy)
    seqs = [s for s, _ in tt.build_training_signal(
        base_w, tt.TRAINING_SIGNAL_ID_2)["train"][:8]]
    m1 = tt.existing_movements(head_w, base_w, seqs)
    assert all(np.array_equal(v, np.zeros(32)) for v in m1.values())
    A0, B0 = tt.initial_factors(tt.TRAINING_SIGNAL_ID_2)
    hiddens = {tuple(s): tt.final_hidden(head_w, s) for s in seqs}
    subset = tt.build_training_signal(
        base_w, tt.TRAINING_SIGNAL_ID_2)["train"][:8]
    lm_head = np.asarray(head_w[tt.TARGET_NAME], dtype=np.float64)
    tot, sig, wval, gA_w, gB_w = tt.analytic_gradients_worst_input(
        A0, B0, subset, hiddens, lm_head, m1,
        tt.WORST_INPUT_LAMBDA, tt.WORST_INPUT_K)
    _loss, gA, gB = tt.analytic_gradients(A0, B0, subset, hiddens,
                                          lm_head)
    assert wval == 0.0 and tot == sig
    assert np.array_equal(gA_w, gA) and np.array_equal(gB_w, gB)
    worst, sel, unstable = _grad_check_worst(
        head_w, base_w, tt.TRAINING_SIGNAL_ID_2)
    assert sel["value"] == 0.0
    assert sel["topk_indices"] == [0, 1, 2, 3]
    assert unstable == GRAD_TOPK_UNSTABLE_AT_BASE
    assert worst == pytest.approx(GRAD_WORST_AT_BASE, rel=0.5)
    assert worst <= GRAD_TOL


def test_gradient_check_worst_at_d0001(tmp_path):
    bundle_dir, policy = _land_d0001(tmp_path)
    manifest = load_bundle(bundle_dir)
    head_w = assemble_weights(bundle_dir, manifest, policy)
    base_w = assemble_base_weights(bundle_dir, manifest, policy)
    worst, sel, unstable = _grad_check_worst(
        head_w, base_w, tt.TRAINING_SIGNAL_ID_2)
    assert sel["topk_indices"] == [4, 5, 2, 3]
    assert sel["value"] == pytest.approx(0.0003049183564004984,
                                         abs=1e-12)
    assert unstable == GRAD_TOPK_UNSTABLE_AT_D0001
    assert worst == pytest.approx(GRAD_WORST_AT_D0001, rel=0.5)
    assert worst <= GRAD_TOL


# -- §15.4: probe-blindness of training, asserted --------------------------------------

def test_worst_training_is_probe_blind(tmp_path, monkeypatch):
    bundle_dir, policy = _land_d0001(tmp_path)
    train_seqs, held_seqs, _ = tt.signal_sequences(
        tt.TRAINING_SIGNAL_ID_2)
    train_set = {tuple(s) for s in train_seqs}
    assert train_set.isdisjoint({tuple(s) for s in identity_inputs()})
    assert train_set.isdisjoint({tuple(s) for s in capability_inputs()})
    assert train_set.isdisjoint({tuple(s) for s in held_seqs})
    captured = []
    orig = tt.analytic_gradients_worst_input

    def spy(A, B, examples, hiddens, lm_head, m1_map, lam_w, k):
        captured.append(({tuple(s) for s, _ in examples},
                         set(m1_map.keys())))
        return orig(A, B, examples, hiddens, lm_head, m1_map, lam_w, k)

    monkeypatch.setattr(tt, "analytic_gradients_worst_input", spy)
    mind = TinyTransformerMind(
        bundle_dir, load_bundle(bundle_dir),
        training_signal_id=tt.TRAINING_SIGNAL_ID_2,
        target_anchoring="base-anchored",
        composition_config="worst-input-composition")
    cand = mind.train_delta(mind.propose([], None))
    assert cand["record"]["blob"] == WORST_BLOB
    assert captured, "the worst-input path never ran"
    for ex_set, m1_set in captured:
        assert ex_set == train_set
        assert m1_set == train_set
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
        kind=KIND, reason="SI-29 probe-blindness probe",
        parents=["d0001"], source_seqs=[1],
        training_signal_id=tt.TRAINING_SIGNAL_ID_2,
        target_anchoring="base-anchored",
        composition_config="worst-input-composition",
        targets=[tt.TARGET_NAME])
    cand2 = tt.train_candidate(copy_dir, manifest, intent,
                               policy=policy)
    assert cand2["blob_bytes"] == cand["blob_bytes"]
    assert cand2["record"]["blob"] == WORST_BLOB


# -- §15.9: both controls, re-run inside the slice ---------------------------------------

def test_none_control_rerun_reproduces_si25(tmp_path):
    bundle_dir, policy, mind, seen, sum1, sum2 = _control_run(tmp_path)
    assert sum1["admitted"] == 1 and sum2["rejected"] == 1
    hp2 = seen[1]["hyperparams"]
    assert hp2["composition_config"] == "none"
    assert hp2["composition_lambda_w"] == 0.0
    assert hp2["composition_worst_k"] is None
    assert hp2["worst_input_train_initial"] is None
    assert hp2["worst_input_train_final"] is None
    assert seen[1]["record"]["blob"] == CTRL_BLOB
    assert seen[1]["record"]["norm"] == pytest.approx(CTRL_NORM,
                                                      abs=1e-12)
    manifest = load_bundle(bundle_dir)
    checks, _row = _checks_of_edit(bundle_dir, manifest,
                                   sum2["edit_ids"][0])
    assert checks["identity_continuity"]["measured"] == \
        pytest.approx(CTRL_DRIFT, abs=1e-9)
    assert checks["capability_floor"]["measured"] == \
        pytest.approx(CTRL_REGRESSION, abs=1e-9)


def test_movement_control_rerun_reproduces_si27(tmp_path):
    bundle_dir, policy, mind, seen, sum1, sum2 = \
        _movement_control_run(tmp_path)
    assert sum1["admitted"] == 1 and sum2["rejected"] == 1
    hp2 = seen[1]["hyperparams"]
    assert hp2["composition_config"] == "movement-composition"
    assert hp2["composition_lambda"] == 1.0
    assert hp2["composition_lambda_w"] == 0.0
    assert hp2["composition_worst_k"] is None
    assert hp2["worst_input_train_final"] is None
    assert seen[1]["record"]["blob"] == PEN_BLOB
    assert seen[1]["record"]["norm"] == pytest.approx(PEN_NORM,
                                                      abs=1e-12)
    manifest = load_bundle(bundle_dir)
    checks, _row = _checks_of_edit(bundle_dir, manifest,
                                   sum2["edit_ids"][0])
    assert checks["identity_continuity"]["measured"] == \
        pytest.approx(PEN_DRIFT, abs=1e-9)
    assert checks["capability_floor"]["measured"] == \
        pytest.approx(PEN_REGRESSION, abs=1e-9)
    note2 = _hyperparams_notes(bundle_dir, manifest, sum2["edit_ids"][0])
    assert note2["composition_config"] == "movement-composition"


# -- the decisive run: §15.1/§15.7 --------------------------------------------------------

def test_decisive_two_cycle_run(tmp_path):
    bundle_dir, policy, mind, seen, sum1, sum2 = _decisive_run(tmp_path)
    assert sum1["admitted"] == 1 and sum1["head"] == "d0001"
    assert sum2["rejected"] == 1 and sum2["halted"] is True
    assert sum2["halt_verdict"] == "REJECT" and sum2["head"] == "d0001"
    assert sum2["delta_stack_depth"] == 1
    assert len(seen) == 2

    hp1, hp2 = seen[0]["hyperparams"], seen[1]["hyperparams"]
    assert hp1["composition_config"] == "none"
    assert hp1["worst_input_train_initial"] is None
    assert seen[0]["record"]["blob"] == SI19_BLOB
    assert hp2["training_signal_id"] == tt.TRAINING_SIGNAL_ID_2
    assert hp2["target_anchoring"] == "base-anchored"
    assert hp2["composition_config"] == "worst-input-composition"
    assert hp2["composition_lambda_w"] == 1.0
    assert hp2["composition_worst_k"] == 4
    assert hp2["composition_lambda"] == 0.0
    assert hp2["penalty_train_initial"] is None
    assert hp2["penalty_train_final"] is None
    assert hp2["head"] == "d0001" and hp2["target_source_head"] == "base"
    assert hp2["stack_depth"] == 1
    assert seen[1]["record"]["parent"] == "d0001"
    assert seen[1]["record"]["id"] == "d0002"
    assert seen[1]["record"]["blob"] == WORST_BLOB
    assert seen[1]["record"]["norm"] == pytest.approx(WORST_NORM,
                                                      abs=1e-12)
    # The two loss terms, separately: the worst-input term
    # starts positive (the initial factors' worst inputs
    # align) and ends negative (worst-case opposition on the
    # train partition); the total ends negative.
    assert hp2["worst_input_train_initial"] == pytest.approx(
        WORST_TRAIN_INITIAL, abs=1e-12)
    assert hp2["worst_input_train_final"] == pytest.approx(
        WORST_TRAIN_FINAL, abs=1e-12)
    assert hp2["worst_input_held_out_initial"] == pytest.approx(
        WORST_HELD_INITIAL, abs=1e-12)
    assert hp2["worst_input_held_out_final"] == pytest.approx(
        WORST_HELD_FINAL, abs=1e-12)
    assert hp2["total_train_loss_initial"] == pytest.approx(
        WORST_TOTAL_INITIAL, abs=1e-12)
    assert hp2["total_train_loss_final"] == pytest.approx(
        WORST_TOTAL_FINAL, abs=1e-12)
    assert hp2["total_train_loss_final"] < 0.0 < \
        hp2["total_train_loss_initial"]
    assert hp2["train_loss_final"] == pytest.approx(
        WORST_SIG_TRAIN_FINAL, abs=1e-12)
    assert hp2["base_held_out_loss"] == pytest.approx(BA_PRE_HELD,
                                                       abs=1e-12)
    assert hp2["held_out_loss_after"] == pytest.approx(WORST_HELD_AFTER,
                                                       abs=1e-12)
    assert hp2["held_out_loss_after"] < hp2["base_held_out_loss"]
    assert hp2["held_out_examples_closer"] == 12
    assert hp2["held_out_examples"] == 16
    assert hp2["norm_projection_activations"] == 0
    # The worst TRAIN input, recorded by train-partition
    # identity (an allowed input, named as one).
    assert hp2["worst_input_train_topk_indices_initial"] == \
        WORST_TOPK_INITIAL
    assert hp2["worst_input_train_topk_indices_final"] == WORST_TOPK_FINAL
    assert hp2["worst_input_train_worst_index_initial"] == 12
    assert hp2["worst_input_train_worst_seq_initial"] == [3, 19]
    assert hp2["worst_input_train_worst_index_final"] == 27
    assert hp2["worst_input_train_worst_seq_final"] == \
        [5, 3, 20, 0, 8, 17, 27, 22]
    train_seqs, _held, _lt = tt.signal_sequences(tt.TRAINING_SIGNAL_ID_2)
    assert train_seqs[27] == hp2["worst_input_train_worst_seq_final"]
    assert train_seqs[12] == hp2["worst_input_train_worst_seq_initial"]

    manifest = load_bundle(bundle_dir)
    edit1, edit2 = sum1["edit_ids"][0], sum2["edit_ids"][0]
    fresh = TinyTransformerMind(
        bundle_dir, load_bundle(bundle_dir),
        training_signal_id=tt.TRAINING_SIGNAL_ID_2,
        target_anchoring="base-anchored",
        composition_config="worst-input-composition")
    cand2 = fresh.train_delta(fresh.propose([], None))
    assert cand2["blob_bytes"] == seen[1]["blob_bytes"]
    ctx = ProbeContext(bundle_dir, manifest, policy=policy)
    independent = Gate(policy).evaluate(fresh.propose([], ctx), cand2, ctx)
    assert independent.verdict == "REJECT"
    ind = {c.name: c for c in independent.checks}
    journaled, edit_row = _checks_of_edit(bundle_dir, manifest, edit2)
    assert edit_row["result"] == "rejected"
    for name in ("identity_continuity", "capability_floor"):
        assert journaled[name]["measured"] == pytest.approx(
            ind[name].measured, abs=1e-12)
    assert journaled["identity_continuity"]["measured"] == \
        pytest.approx(WORST_DRIFT, abs=1e-9)
    assert journaled["identity_continuity"]["passed"] is False
    assert journaled["capability_floor"]["measured"] == \
        pytest.approx(WORST_REGRESSION, abs=1e-9)
    assert journaled["capability_floor"]["passed"] is True
    note1 = _hyperparams_notes(bundle_dir, manifest, "d0001")
    assert note1["composition_config"] == "none"
    note2 = _hyperparams_notes(bundle_dir, manifest, edit2)
    assert note2["training_signal_id"] == tt.TRAINING_SIGNAL_ID_2
    assert note2["target_anchoring"] == "base-anchored"
    assert note2["composition_config"] == "worst-input-composition"
    assert note2["composition_lambda_w"] == 1.0
    assert note2["composition_worst_k"] == 4
    assert note2["worst_input_train_final"] == pytest.approx(
        WORST_TRAIN_FINAL, abs=1e-12)
    assert note2["head"] == "d0001"
    ok, report = _verify_probe_head(bundle_dir, manifest, policy=policy)
    assert ok is True
    assert report["identity"]["max_drift"] == pytest.approx(
        SI19_DRIFT, abs=1e-9)


# -- §15.2/§15.8: the allowed-input worst-input records --------------------------------------

def _allowed_worst_record(bundle_dir, manifest, policy, candidate):
    head_w = assemble_weights(bundle_dir, manifest, policy)
    base_w = assemble_base_weights(bundle_dir, manifest, policy)
    cand_w = assemble_weights_with_candidate(bundle_dir, manifest,
                                             candidate, policy)
    train_seqs, held_seqs, _ = tt.signal_sequences(
        tt.TRAINING_SIGNAL_ID_2)
    return {
        "train": tt.worst_input_composition_record(
            head_w, base_w, cand_w, train_seqs),
        "held_out": tt.worst_input_composition_record(
            head_w, base_w, cand_w, held_seqs),
    }


def test_allowed_input_worst_records(tmp_path):
    c_dir, c_pol, _cm, c_seen, _s1, _s2 = _control_run(
        tmp_path / "c", "tiny29ctrl")
    c_rec = _allowed_worst_record(c_dir, load_bundle(c_dir), c_pol,
                                  c_seen[1])
    p_dir, p_pol, _pm, p_seen, _q1, _q2 = _movement_control_run(
        tmp_path / "p", "tiny29pen")
    p_rec = _allowed_worst_record(p_dir, load_bundle(p_dir), p_pol,
                                  p_seen[1])
    t_dir, t_pol, _tm, t_seen, _u1, _u2 = _decisive_run(
        tmp_path / "t", "tiny29")
    t_rec = _allowed_worst_record(t_dir, load_bundle(t_dir), t_pol,
                                  t_seen[1])

    # Phase 0 values, re-measured here for both existing
    # candidates (the as-built doc publishes the full
    # per-input tables from these records).
    assert c_rec["train"]["max_p"] == pytest.approx(
        -0.0006878804284308964, abs=1e-12)
    assert c_rec["train"]["worst_index"] == 19
    assert c_rec["train"]["topk_mean_p"] == pytest.approx(
        -0.0006952966321423392, abs=1e-12)
    assert c_rec["train"]["topk_indices"] == [19, 31, 27, 11]
    assert c_rec["held_out"]["max_p"] == pytest.approx(
        -0.0006654083712165222, abs=1e-12)
    assert c_rec["held_out"]["worst_index"] == 11
    assert p_rec["train"]["max_p"] == pytest.approx(
        -0.001378012707059647, abs=1e-12)
    assert p_rec["train"]["worst_index"] == 31
    assert p_rec["train"]["topk_mean_p"] == pytest.approx(
        -0.001389653406875007, abs=1e-12)
    assert p_rec["train"]["topk_indices"] == [31, 15, 23, 7]
    assert p_rec["train"]["mean_p"] == pytest.approx(
        -0.00262476117807562, abs=1e-12)
    assert p_rec["held_out"]["max_p"] == pytest.approx(
        -0.0013635384980049186, abs=1e-12)
    assert p_rec["held_out"]["worst_index"] == 15
    # The TREATMENT: the worst allowed input opposes
    # decisively MORE than under either control, on both
    # allowed partitions — the term did on the worst allowed
    # inputs what it was designed to do there.
    assert t_rec["train"]["max_p"] == pytest.approx(
        -0.0023129050472012187, abs=1e-12)
    assert t_rec["train"]["max_p"] < p_rec["train"]["max_p"] < \
        c_rec["train"]["max_p"]
    assert t_rec["train"]["worst_index"] == 27
    assert t_rec["train"]["worst_seq"] == [5, 3, 20, 0, 8, 17, 27, 22]
    assert t_rec["train"]["topk_indices"] == WORST_TOPK_FINAL
    assert t_rec["train"]["topk_mean_p"] == pytest.approx(
        -0.002322561516640352, abs=1e-12)
    assert t_rec["train"]["aggregate_inner_product"] == pytest.approx(
        -2.4928287954527795, abs=1e-9)
    assert t_rec["held_out"]["max_p"] == pytest.approx(
        -0.0022598436988714773, abs=1e-12)
    assert t_rec["held_out"]["worst_index"] == 3
    assert t_rec["held_out"]["topk_mean_p"] == pytest.approx(
        -0.002325071873910509, abs=1e-12)
    # Cross-checks: the record's forward-path top-k mean on
    # the train partition equals the training run's analytic
    # worst-input value at the end of training (and likewise
    # held-out); m1 norms are candidate-independent.
    hp2 = t_seen[1]["hyperparams"]
    assert t_rec["train"]["topk_mean_p"] == pytest.approx(
        hp2["worst_input_train_final"], abs=1e-9)
    assert t_rec["held_out"]["topk_mean_p"] == pytest.approx(
        hp2["worst_input_held_out_final"], abs=1e-9)
    for part in ("train", "held_out"):
        assert t_rec[part]["m1_norm"] == pytest.approx(
            c_rec[part]["m1_norm"], abs=1e-12)
    assert t_rec["train"]["m2_norm"] == pytest.approx(
        2.1423496133427675, abs=1e-9)


# -- §15.8: probe diagnostics, four-run table, F1′/F2′/F3′ -----------------------------------

def _folded_update(candidate):
    parsed = parse_blob(candidate["blob_bytes"])
    target = candidate["record"]["targets"][0]
    A = np.asarray(parsed.tensors[target["factor_keys"][0]],
                   dtype=np.float64)
    B = np.asarray(parsed.tensors[target["factor_keys"][1]],
                   dtype=np.float64)
    return (float(target["alpha"]) / float(target["rank"])) * (B @ A)


def test_probe_diagnostics_four_run_and_falsification(tmp_path):
    bundle_dir, policy, mind, seen, sum1, sum2 = _decisive_run(tmp_path)
    manifest = load_bundle(bundle_dir)
    assert manifest["head"] == "d0001"
    ctx = ProbeContext(bundle_dir, manifest, policy=policy)
    cand2 = seen[1]
    expected = {
        "tiny-id-0": (0.17189218290150166, 0.1409909427165985, 0.31215621903538704),
        "tiny-id-1": (0.15088725090026855, 0.07746927067637444, 0.1174006462097168),
        "tiny-id-2": (0.11858986946754158, 0.14978677220642567, 0.14997844584286213),
        "tiny-id-3": (0.14782869815826416, 0.3379737436771393, 0.19014504551887512),
        "tiny-id-4": (0.11720322235487401, 0.16256514564156532, 0.11752748489379883),
        "tiny-id-5": (0.09891846403479576, 0.10969258844852448, 0.13741302862763405),
        "tiny-id-6": (0.12363102938979864, 0.1797015145421028, 0.15289807319641113),
        "tiny-id-7": (0.08795571327209473, 0.16283829044550657, 0.19354498013854027),
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
    # Four-run composed maxima, each under its own labels:
    # SI-23 0.29634, SI-25 0.33290, SI-27 0.32509, SI-29
    # (treatment) 0.31216 — improves on SI-25 and SI-27,
    # worsens on SI-23, stays above tol 0.25.
    assert S23_DRIFT < WORST_DRIFT < PEN_DRIFT < CTRL_DRIFT
    assert WORST_DRIFT > 0.25
    # Alignment (derivation): aggregate probe inner product
    # opposes vs the SI-25 control but LESS than SI-27's
    # treatment; at tiny-id-0 the movements align MORE than
    # under SI-27's treatment.
    a_all = np.concatenate(m1_all); b_all = np.concatenate(m2_all)
    agg_inner = float(np.dot(a_all, b_all))
    assert agg_inner == pytest.approx(-0.28751472160662706, abs=1e-9)
    assert PEN_AGG_INNER < agg_inner < -0.21256230517288188
    assert float(np.mean(np.sign(a_all) == np.sign(b_all))) == \
        pytest.approx(0.38671875, abs=1e-12)
    assert inners["tiny-id-0"] == pytest.approx(0.04944786620153699,
                                                abs=1e-9)
    assert inners["tiny-id-0"] > PEN_WORST_PROBE_INNER > 0.0
    assert float(np.linalg.norm(b_all)) == pytest.approx(
        0.9424814001349618, abs=1e-9)
    d1u, d2u = _folded_update(seen[0]), _folded_update(cand2)
    assert float(np.sum(d1u * d2u)) == pytest.approx(
        -0.0173962867164366, abs=1e-9)
    assert float(np.linalg.norm(d2u)) == pytest.approx(WORST_NORM,
                                                       abs=1e-9)
    headroom = 0.25 - SI19_DRIFT
    assert sum(1 for v in rows.values() if v[1] > headroom) == 7
    # F1′ MEASURED — the worst-input term's worst allowed
    # input improved against both controls (asserted in the
    # record test: train max p −0.000688 → −0.001378 →
    # −0.002313) while probe composition at the worst probe
    # worsened against SI-27's treatment (+0.03801 →
    # +0.04945 at tiny-id-0).
    assert inners["tiny-id-0"] > PEN_WORST_PROBE_INNER
    # F2′ NOT measured as defined — its aggregate condition
    # fails (the treatment's aggregate probe inner product
    # −0.2875 is LESS opposing than SI-27's −0.3271), even
    # though the composed maximum stays above tol. Stated
    # as measured, not rounded into F2′.
    assert agg_inner > PEN_AGG_INNER and WORST_DRIFT > 0.25
    # F3′ read under the PRE-REGISTERED rule (Phase 0,
    # as-built doc): approached iff (a) held-out loss falls,
    # (b) at least 12/16 held-out examples are closer, and
    # (c) the held-out residual is at most 2× the re-run
    # SI-27 control's residual. All three hold — (b) exactly
    # at its boundary (12/16) — so F3′ does NOT fire: the
    # destination is approached under the rule, less
    # completely than SI-27's treatment (16/16, residual
    # 0.000523).
    hp2 = seen[1]["hyperparams"]
    assert hp2["held_out_loss_after"] < hp2["base_held_out_loss"]
    assert hp2["held_out_examples_closer"] == 12 >= 12
    assert hp2["held_out_loss_after"] <= 2 * PEN_HELD_AFTER
    assert hp2["held_out_loss_after"] > PEN_HELD_AFTER
    rep = Gate(policy).evaluate(mind.propose([], ctx), cand2, ctx)
    checks = {c.name: c for c in rep.checks}
    assert checks["capability_floor"].measured == pytest.approx(
        WORST_REGRESSION, abs=1e-9)
    targets = ctx.capability_targets()
    correct = sum(
        1 for cid, x, _e in ctx.capability_suite()
        if int(np.argmax(np.asarray(ctx.forward_candidate(x, cand2))))
        == int(targets[cid]))
    assert correct == 16 and ctx.echo_floor() == 0.9375


# -- §15.6: determinism per cycle --------------------------------------------------------------

def test_decisive_run_repeats_exactly(tmp_path):
    _b1, _p1, _m1, seen1, s1a, s2a = _decisive_run(tmp_path / "a",
                                                  "tiny29")
    _b2, _p2, _m2, seen2, s1b, s2b = _decisive_run(tmp_path / "b",
                                                  "tiny29")
    assert s1a == s1b and s2a == s2b
    assert load_bundle(_b1)["deltas"] == load_bundle(_b2)["deltas"]
    for cand_a, cand_b in zip(seen1, seen2):
        assert cand_a["blob_bytes"] == cand_b["blob_bytes"]
        assert cand_a["record"] == cand_b["record"]
        hp_a = dict(cand_a["hyperparams"])
        hp_b = dict(cand_b["hyperparams"])
        hp_a.pop("wall_time_seconds"); hp_b.pop("wall_time_seconds")
        assert hp_a == hp_b


def test_worst_train_delta_deterministic(tmp_path):
    bundle_dir, policy = _land_d0001(tmp_path)
    mind = TinyTransformerMind(
        bundle_dir, load_bundle(bundle_dir),
        training_signal_id=tt.TRAINING_SIGNAL_ID_2,
        target_anchoring="base-anchored",
        composition_config="worst-input-composition")
    intent = mind.propose([], None)
    first = mind.train_delta(intent)
    second = mind.train_delta(intent)
    assert first["blob_bytes"] == second["blob_bytes"]
    assert first["record"] == second["record"]
    assert first["record"]["blob"] == WORST_BLOB


# -- §15.10: VERIFY / unwind honesty / untouched scope -------------------------------------------

def test_unwind_honesty_and_untouched_scope(tmp_path):
    bundle_dir, policy, mind, seen, sum1, sum2 = _decisive_run(tmp_path)
    assert sum1["admitted"] == 1 and sum2["rejected"] == 1
    manifest = load_bundle(bundle_dir)
    assert manifest["head"] == "d0001"
    assert len(manifest["deltas"]) == 1
    cand2_hex = seen[1]["record"]["blob"].split(":", 1)[1]
    assert not os.path.isfile(os.path.join(
        bundle_dir, "blobs", "sha256", cand2_hex + ".safetensors"))
    assert DEFAULT_POLICY["max_stack_depth"] == 8
    assert DEFAULT_POLICY["identity_tol"] == 0.25
    assert DEFAULT_POLICY["capability_eps"] == 0.25
    assert DEFAULT_POLICY["max_targets_per_delta"] == 1
    assert DEFAULT_POLICY["allow_additive"] is False
    on_disk = load_policy(bundle_dir)
    assert on_disk["identity_tol"] == 0.25
    assert on_disk["capability_eps"] == 0.25
    assert tt.TARGET_ANCHORING == "weights-under-training"
    assert tt.TARGET_ANCHORINGS == ("weights-under-training",
                                    "base-anchored")
    assert set(tt.SIGNAL_REGISTRY) == {tt.TRAINING_SIGNAL_ID,
                                       tt.TRAINING_SIGNAL_ID_2}
    assert tt.CORRECTION_SCALE == 0.15
    assert tt.N_TRAIN == 32 and tt.N_HELD_OUT == 16
    assert copy.deepcopy(tt.COMPOSITION_CONFIGS) == \
        ("none", "movement-composition", "worst-input-composition",
         "train-partition-projection")
    with pytest.raises(SquashError):
        squash_bundle(bundle_dir, keep=0)
