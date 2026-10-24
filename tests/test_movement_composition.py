"""SI-27: movement-composition-aware training for the miniature
(RFC SI-26 §14 outline, §15 acceptance criteria).

Covers the §15 criteria, in order:

1. recorded composition configuration — the configuration
   serves exactly the two named values (``"none"``,
   ``"movement-composition"``) and refuses every other value
   with the existing ``ValueError`` discipline (constructor,
   setter, intent, and ``train_candidate`` dispatch); a test
   asserts, per cycle on both verdict paths, that the
   journaled ``training-hyperparams`` note's composition
   configuration, its λ (= 1.0 under
   ``"movement-composition"``), its signal id, and its
   anchoring equal the configuration the driver set —
   including the decisive cycle's full triple (signal 2,
   ``target_anchoring = "base-anchored"``, composition =
   ``"movement-composition"``). λ is not independently
   settable (no constructor/setter/intent dial exists);
2. coincidence at base — configured with signal 2,
   base-anchoring, and the movement-composition
   configuration, a first cycle at base reproduces SI-25's
   at-base run EXACTLY (blob, drift, regression), with the
   penalty term measured at initialisation and at the end
   of training (identically zero on both partitions);
   configured with signal 1 under the default composition
   configuration, a first cycle reproduces ``d0001``
   exactly (criterion 9, asserted in the decisive run);
3. gradient check for the penalised loss — finite
   differences of signal 2's base-anchored,
   composition-penalised loss at head ``d0001`` AND at base
   under the new code path, over EVERY factor entry,
   tolerance 1e-5 unchanged;
4. probe-blindness of training, asserted — a spy on the
   penalised gradient path proves the penalty's input set
   equals signal 2's train partition exactly; the train
   partition's disjointness from both probe suites is
   re-asserted under the new code path; and a training run
   against a bundle copy whose ``probe_suite.json`` was
   perturbed produces a byte-identical candidate;
5. the allowed-input composition record — for the re-run
   control candidate and for the treatment candidate, the
   alignment diagnostic on signal 2's train and held-out
   partitions (aggregates asserted here, published in the
   as-built doc), with each partition labelled and the
   penalty's forward-path value cross-checked against the
   hyperparams' analytic value;
6. determinism per cycle — a full decisive run repeated
   from the seed reproduces both cycles' records and blob
   hashes exactly, except the labelled wall-time field;
   the penalised ``train_delta`` alone is likewise
   byte-identical across repeats;
7. gate honesty per cycle — the decisive cycle 2's
   journaled checks 3/4 equal an independent
   ``ProbeContext`` evaluation of the same candidate
   against ``d0001``. The verdict is REJECT (cumulative
   identity drift 0.32509003952145576 > tol 0.25), reported
   AS MEASURED. No criterion requires an ADMIT;
8. cumulative accounting, diagnostics, the three-run
   table, and the falsification readings — per-probe
   composed drift for the treatment against both published
   controls (SI-23 head-relative, SI-25 base-anchored),
   alignment and folded-update diagnostics, the penalised
   run's signal held-out records (the F3 evidence), and
   the F1/F2/F3 readings of RFC §7.6, each stated as
   measured / not measured from the measured values;
9. first-cycle and control reproduction — cycle 1
   reproduces SI-19/SI-21/SI-23/SI-25 exactly, and the
   re-run unpenalised control at ``d0001`` reproduces
   SI-25's decisive cycle-2 candidate exactly (blob
   ``sha256:646d1e8b…d675``, drift 0.3329036422073841);
10. VERIFY, unwind, and untouched scope — cycle 1's §6.6
    VERIFY passes as in SI-19; cycle 2 never landed
    (REJECT), so NO authored two-delta stack exists and no
    unwind is claimed (asserted: head stays ``d0001``, one
    delta only); exactly two cycles attempted in the
    decisive run; policy defaults, tolerances, caps, both
    signals' definitions, and both anchorings' definitions
    unchanged; ``si squash`` still fails closed.

The decisive run's driver shape: ONE ``TinyTransformerMind``
reused through ``si.run.run_cycles`` — one ``run_cycles``
call for cycle 1 (signal 1, default anchoring, default
composition configuration), then the driver explicitly
reconfigures the same mind to signal 2
(``set_training_signal``), to base-anchored targets
(``set_target_anchoring``), AND to the movement-composition
configuration (``set_composition_config`` — three recorded
selections), then a second ``run_cycles`` call for cycle 2.
The mind never chooses — neither its signal, its anchoring,
nor its composition configuration.
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
CLOSED_FORM = 0.00140625

SI19_BLOB = "sha256:2761e8ffc139ce3e107c11da035e8a73c20792945a494f4e493dccf418b6ca9e"
SI19_DRIFT = 0.17189218290150166
SI19_REGRESSION = 0.1657181391492486

S2_AT_BASE_BLOB = "sha256:131cc242728024b4f833bc78ea627825f1e8616972fc2670603673e1814645f4"
S2_AT_BASE_DRIFT = 0.16672966815531254
S2_AT_BASE_REGRESSION = 0.2418281203135848

# SI-25's decisive cycle 2 — SI-27's CONTROL run, re-run
# inside this slice under composition "none" (docs/
# SI-25-base-anchored-targets.md, published measurements).
CTRL_BLOB = "sha256:646d1e8ba3978d858d4173d561f8799bce37350f88ff2de65f40c223d9e4d675"
CTRL_NORM = 0.2515524481203623
CTRL_DRIFT = 0.3329036422073841
CTRL_REGRESSION = 0.10096919978968799
CTRL_HELD_AFTER = 1.043608588752459e-05
CTRL_ALIGN_AGG_INNER = -0.21256230517288188
CTRL_WORST_INNER = 0.0338873784299374  # at tiny-id-0

# SI-23's head-relative cycle 2 (docs/SI-23-second-signal.md)
# — the three-run table's first column.
S23_DRIFT = 0.2963416576385498
S23_REGRESSION = 0.15023326850496233
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
# SI-25's composed drift per probe (docs/
# SI-25-base-anchored-targets.md §5.6(i)).
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

# SI-27 measured values (author's runs; the as-built doc
# publishes them with their labels). The decisive cycle-2
# verdict is REJECT — measured, not required, not tuned for.
PEN_BLOB = "sha256:1bb277ab40b9aa53a6218d971e46f29b261ba2bb5f7932ffb98e3951f372b637"
PEN_NORM = 0.2969693161862222
PEN_DRIFT = 0.32509003952145576
PEN_REGRESSION = 0.14536726917140186
PEN_TRAIN_INITIAL = 0.00017392083908663979
PEN_TRAIN_FINAL = -0.0026247611759719963
PEN_HELD_INITIAL = 0.0001744748535695612
PEN_HELD_FINAL = -0.002628909254542555
TOTAL_INITIAL = 0.004473770642759213
TOTAL_FINAL = -0.0021056259086710467
SIG_TRAIN_FINAL = 0.0005191352673009495
SIG_HELD_AFTER = 0.0005227812955486338
BA_PRE_HELD = 0.002757805049077469


def _seeded(tmp_path, name="tiny27"):
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


def _decisive_run(tmp_path, name="tiny27"):
    """The decisive two-cycle run (see module docstring for
    the driver shape): one reused mind, cycle 1 signal 1
    under the default anchoring and default composition
    configuration via ``run_cycles``, explicit reconfiguration
    to signal 2 AND base-anchored targets AND the
    movement-composition configuration, cycle 2 via
    ``run_cycles``."""
    bundle_dir = _seeded(tmp_path, name)
    policy = _policy(bundle_dir)
    mind, seen = _spying_mind(bundle_dir)
    sum1 = run_cycles(bundle_dir, mind, 1, gate=Gate(policy))
    mind.set_training_signal(tt.TRAINING_SIGNAL_ID_2)
    mind.set_target_anchoring(tt.BASE_ANCHORED)
    mind.set_composition_config(tt.MOVEMENT_COMPOSITION)
    sum2 = run_cycles(bundle_dir, mind, 1, gate=Gate(policy))
    return bundle_dir, policy, mind, seen, sum1, sum2


def _control_run(tmp_path, name="tiny27ctrl"):
    """SI-25's decisive run re-run inside this slice:
    identical driver shape, composition left at ``"none"``."""
    bundle_dir = _seeded(tmp_path, name)
    policy = _policy(bundle_dir)
    mind, seen = _spying_mind(bundle_dir)
    sum1 = run_cycles(bundle_dir, mind, 1, gate=Gate(policy))
    mind.set_training_signal(tt.TRAINING_SIGNAL_ID_2)
    mind.set_target_anchoring(tt.BASE_ANCHORED)
    sum2 = run_cycles(bundle_dir, mind, 1, gate=Gate(policy))
    return bundle_dir, policy, mind, seen, sum1, sum2


def _land_d0001(tmp_path, name="tiny27"):
    bundle_dir = _seeded(tmp_path, name)
    policy = _policy(bundle_dir)
    mind, _seen = _spying_mind(bundle_dir)
    res = run_cycle(bundle_dir, mind, gate=Gate(policy))
    assert res["verdict"] == "ADMIT" and res["head"] == "d0001"
    return bundle_dir, policy


# -- §15.1: recorded composition configuration ----------------------------------

def test_composition_configuration_and_refusals(tmp_path):
    # Exactly the two named configurations are served; the
    # default's name and λ's definition are pinned constants.
    assert tt.COMPOSITION_CONFIGS == ("none", "movement-composition",
                                      "worst-input-composition")  # SI-29: third value added
    assert tt.DEFAULT_COMPOSITION_CONFIG == "none"
    assert tt.COMPOSITION_NONE == "none"
    assert tt.MOVEMENT_COMPOSITION == "movement-composition"
    assert tt.COMPOSITION_LAMBDA == 1.0
    assert tt.validate_composition_config("none") == "none"
    assert tt.validate_composition_config("movement-composition") == \
        "movement-composition"
    for bad in ("penalised", "composition", "", "NONE", None):
        with pytest.raises(ValueError,
                           match="unknown composition configuration"):
            tt.validate_composition_config(bad)

    bundle_dir = _seeded(tmp_path)
    # Constructor: default is "none"; the explicit choice is
    # served; any other value refuses.
    mind = TinyTransformerMind(bundle_dir, load_bundle(bundle_dir))
    assert mind.composition_config == "none"
    assert mind.propose([], None).composition_config == "none"
    mind_pen = TinyTransformerMind(
        bundle_dir, load_bundle(bundle_dir),
        composition_config="movement-composition")
    assert mind_pen.composition_config == "movement-composition"
    assert mind_pen.propose([], None).composition_config == \
        "movement-composition"
    with pytest.raises(ValueError,
                       match="unknown composition configuration"):
        TinyTransformerMind(bundle_dir, load_bundle(bundle_dir),
                            composition_config="learned")
    # λ is NOT independently settable: the constructor takes
    # no λ argument at all.
    with pytest.raises(TypeError):
        TinyTransformerMind(bundle_dir, load_bundle(bundle_dir),
                            composition_lambda=0.5)
    # Setter: explicit driver reconfiguration, same refusals,
    # and no λ dial on it either.
    mind.set_composition_config("movement-composition")
    assert mind.composition_config == "movement-composition"
    assert mind.propose([], None).composition_config == \
        "movement-composition"
    mind.set_composition_config("none")
    assert mind.propose([], None).composition_config == "none"
    with pytest.raises(ValueError,
                       match="unknown composition configuration"):
        mind.set_composition_config("verdict-driven")
    with pytest.raises(TypeError):
        mind.set_composition_config("movement-composition", lam=0.5)
    # Dispatch: an intent carrying any other configuration
    # refuses through train_delta, per the existing discipline.
    base = dict(kind=KIND, reason="SI-27 refusal probe",
                parents=["base"], source_seqs=[1])
    for bad in ("movement_composition", "", "penalty"):
        with pytest.raises(ValueError,
                           match="unknown composition configuration"):
            mind.train_delta(Intent(
                training_signal_id=tt.TRAINING_SIGNAL_ID,
                composition_config=bad, targets=[tt.TARGET_NAME],
                **base))
    # The composition axis is independent of the signal and
    # anchoring axes: propose carries ALL THREE recorded
    # selections, and the intent's dict form records them.
    mind.set_training_signal(tt.TRAINING_SIGNAL_ID_2)
    mind.set_target_anchoring("base-anchored")
    mind.set_composition_config("movement-composition")
    intent = mind.propose([], None)
    assert intent.training_signal_id == tt.TRAINING_SIGNAL_ID_2
    assert intent.target_anchoring == "base-anchored"
    assert intent.composition_config == "movement-composition"
    assert intent.to_dict()["composition_config"] == \
        "movement-composition"


# -- §15.2: coincidence at base ----------------------------------------------------

def test_penalised_signal2_at_base_coincides_with_si25(tmp_path):
    bundle_dir = _seeded(tmp_path)
    policy = _policy(bundle_dir)
    before = _blobs_total(bundle_dir)
    mind, seen = _spying_mind(
        bundle_dir, training_signal_id=tt.TRAINING_SIGNAL_ID_2,
        target_anchoring="base-anchored",
        composition_config="movement-composition")
    res = run_cycle(bundle_dir, mind, gate=Gate(policy))
    # Reported AS MEASURED: at base m1 is identically zero,
    # the penalty vanishes, and the run reproduces SI-25's
    # at-base run EXACTLY — same blob, same gate numbers.
    assert res["verdict"] == "ADMIT" and res["head"] == "d0001"
    assert res["verified"] is True and res["rolled_back"] is False
    hp = seen[0]["hyperparams"]
    assert hp["composition_config"] == "movement-composition"
    assert hp["composition_lambda"] == 1.0
    assert hp["target_anchoring"] == "base-anchored"
    assert hp["head"] == "base" and hp["target_source_head"] == "base"
    # The penalty term, MEASURED at initialisation and at the
    # end of training, on both partitions: identically zero.
    assert hp["penalty_train_initial"] == 0.0
    assert hp["penalty_train_final"] == 0.0
    assert hp["penalty_held_out_initial"] == 0.0
    assert hp["penalty_held_out_final"] == 0.0
    # With the penalty zero the totals ARE the signal losses.
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
    assert note["composition_config"] == "movement-composition"
    assert note["composition_lambda"] == 1.0
    ok, report = _verify_probe_head(bundle_dir, manifest, policy=policy)
    assert ok is True
    assert report["identity"]["max_drift"] == pytest.approx(
        S2_AT_BASE_DRIFT, abs=1e-9)


# -- §15.3: gradient checks for the penalised loss ---------------------------------

def _grad_worst_penalised(head_weights, base_weights, signal_id):
    """Worst analytic-vs-central-finite-difference disagreement
    over ALL 512 factor entries, on an 8-example subset, for
    the BASE-ANCHORED, COMPOSITION-PENALISED loss: targets
    derived from ``base_weights``, hidden states and readout
    from ``head_weights``, landed ``m1`` on the train
    partition. The finite differences differentiate
    ``penalised_factor_loss`` (the float32 shared fold plus
    the existing forwards)."""
    signal = tt.build_training_signal(base_weights, signal_id)
    subset = signal["train"][:8]
    A0, B0 = tt.initial_factors(signal_id)
    lm_head = np.asarray(head_weights[tt.TARGET_NAME], dtype=np.float64)
    hiddens = {tuple(s): tt.final_hidden(head_weights, s)
               for s, _ in subset}
    m1 = tt.existing_movements(head_weights, base_weights,
                               [s for s, _ in subset])
    _tot, _sig, _pen, grad_A, grad_B = tt.analytic_gradients_penalised(
        A0, B0, subset, hiddens, lm_head, m1, tt.COMPOSITION_LAMBDA)
    assert np.max(np.abs(grad_A)) > 0.0 and np.max(np.abs(grad_B)) > 0.0
    worst = 0.0
    for i in range(tt.RANK):
        for j in range(32):
            plus = A0.copy(); plus[i, j] += GRAD_EPS
            minus = A0.copy(); minus[i, j] -= GRAD_EPS
            fd = (tt.penalised_factor_loss(head_weights, base_weights,
                                           plus, B0, subset)
                  - tt.penalised_factor_loss(head_weights, base_weights,
                                             minus, B0, subset)) \
                / (2 * GRAD_EPS)
            worst = max(worst, abs(fd - grad_A[i, j]))
    for i in range(32):
        for j in range(tt.RANK):
            plus = B0.copy(); plus[i, j] += GRAD_EPS
            minus = B0.copy(); minus[i, j] -= GRAD_EPS
            fd = (tt.penalised_factor_loss(head_weights, base_weights,
                                           A0, plus, subset)
                  - tt.penalised_factor_loss(head_weights, base_weights,
                                             A0, minus, subset)) \
                / (2 * GRAD_EPS)
            worst = max(worst, abs(fd - grad_B[i, j]))
    return worst


def test_gradient_check_penalised_at_base(tmp_path):
    bundle_dir = _seeded(tmp_path)
    policy = _policy(bundle_dir)
    manifest = load_bundle(bundle_dir)
    head_w = assemble_weights(bundle_dir, manifest, policy)
    base_w = assemble_base_weights(bundle_dir, manifest, policy)
    # At base m1 is identically zero: every existing movement
    # is the zero vector, the penalty at the pinned initial
    # factors is exactly 0.0, and the penalised gradients
    # coincide with the unpenalised ones — measured under the
    # new code path, not asserted from the derivation.
    seqs = [s for s, _ in tt.build_training_signal(
        base_w, tt.TRAINING_SIGNAL_ID_2)["train"][:8]]
    m1 = tt.existing_movements(head_w, base_w, seqs)
    assert all(np.array_equal(v, np.zeros(32)) for v in m1.values())
    A0, B0 = tt.initial_factors(tt.TRAINING_SIGNAL_ID_2)
    hiddens = {tuple(s): tt.final_hidden(head_w, s) for s in seqs}
    subset = tt.build_training_signal(
        base_w, tt.TRAINING_SIGNAL_ID_2)["train"][:8]
    lm_head = np.asarray(head_w[tt.TARGET_NAME], dtype=np.float64)
    tot, sig, pen, gA_pen, gB_pen = tt.analytic_gradients_penalised(
        A0, B0, subset, hiddens, lm_head, m1, tt.COMPOSITION_LAMBDA)
    _loss, gA, gB = tt.analytic_gradients(A0, B0, subset, hiddens,
                                          lm_head)
    assert pen == 0.0 and tot == sig
    assert np.array_equal(gA_pen, gA) and np.array_equal(gB_pen, gB)
    worst = _grad_worst_penalised(head_w, base_w,
                                  tt.TRAINING_SIGNAL_ID_2)
    # Measured 5.5249277028738815e-08 under the new code path —
    # identical to SI-25's at-base figure, as the coincidence
    # requires. The pin is SI-19's 1e-5, unchanged.
    assert worst == pytest.approx(5.5249277028738815e-08, rel=0.5)
    assert worst <= GRAD_TOL


def test_gradient_check_penalised_at_d0001(tmp_path):
    bundle_dir, policy = _land_d0001(tmp_path)
    manifest = load_bundle(bundle_dir)
    head_w = assemble_weights(bundle_dir, manifest, policy)
    base_w = assemble_base_weights(bundle_dir, manifest, policy)
    worst = _grad_worst_penalised(head_w, base_w,
                                  tt.TRAINING_SIGNAL_ID_2)
    # Measured 5.149878699880166e-08 at head d0001 (as-built
    # doc) — the decisive configuration: base-derived
    # targets, head hidden states, head readout, landed m1
    # on the train partition. A check run only at base does
    # not satisfy §15.3; this is the decisive one.
    assert worst == pytest.approx(5.149878699880166e-08, rel=0.5)
    assert worst <= GRAD_TOL


# -- §15.4: probe-blindness of training, asserted -----------------------------------

def test_penalised_training_is_probe_blind(tmp_path, monkeypatch):
    bundle_dir, policy = _land_d0001(tmp_path)
    train_seqs, held_seqs, _ = tt.signal_sequences(
        tt.TRAINING_SIGNAL_ID_2)
    train_set = {tuple(s) for s in train_seqs}
    # Disjointness re-asserted under the new code path: the
    # penalty's partition is disjoint from both probe suites
    # and from the signal's own held-out partition.
    assert train_set.isdisjoint({tuple(s) for s in identity_inputs()})
    assert train_set.isdisjoint({tuple(s) for s in capability_inputs()})
    assert train_set.isdisjoint({tuple(s) for s in held_seqs})
    # Spy on the penalised gradient path during a REAL
    # train_candidate run: every examples set and every m1
    # key set it ever receives is exactly the train partition.
    captured = []
    orig = tt.analytic_gradients_penalised

    def spy(A, B, examples, hiddens, lm_head, m1_map, lam):
        captured.append(({tuple(s) for s, _ in examples},
                         set(m1_map.keys())))
        return orig(A, B, examples, hiddens, lm_head, m1_map, lam)

    monkeypatch.setattr(tt, "analytic_gradients_penalised", spy)
    mind = TinyTransformerMind(
        bundle_dir, load_bundle(bundle_dir),
        training_signal_id=tt.TRAINING_SIGNAL_ID_2,
        target_anchoring="base-anchored",
        composition_config="movement-composition")
    cand = mind.train_delta(mind.propose([], None))
    assert cand["record"]["blob"] == PEN_BLOB
    assert captured, "the penalised path never ran"
    for ex_set, m1_set in captured:
        assert ex_set == train_set
        assert m1_set == train_set
    monkeypatch.undo()
    # Perturbation: a copy of the bundle whose probe artifact
    # on disk is rewritten (probe inputs replaced) produces a
    # BYTE-IDENTICAL candidate through train_candidate called
    # directly with the pre-perturbation verified manifest —
    # the training path never opens probe_suite.json, so no
    # probe input, logit, or derived movement can reach the
    # loss. (load_bundle itself refuses the perturbed copy —
    # fail-closed tamper evidence, a separate guarantee.)
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
        kind=KIND, reason="SI-27 probe-blindness probe",
        parents=["d0001"], source_seqs=[1],
        training_signal_id=tt.TRAINING_SIGNAL_ID_2,
        target_anchoring="base-anchored",
        composition_config="movement-composition",
        targets=[tt.TARGET_NAME])
    cand2 = tt.train_candidate(copy_dir, manifest, intent,
                               policy=policy)
    assert cand2["blob_bytes"] == cand["blob_bytes"]
    assert cand2["record"]["blob"] == PEN_BLOB


# -- §15.9: the control, re-run inside the slice -------------------------------------

def test_control_rerun_reproduces_si25_decisive(tmp_path):
    bundle_dir, policy, mind, seen, sum1, sum2 = _control_run(tmp_path)
    assert sum1["admitted"] == 1 and sum2["rejected"] == 1
    hp2 = seen[1]["hyperparams"]
    # The unpenalised control under the refactored path: the
    # recorded configuration is "none", λ 0.0, penalty fields
    # null (not measured — the path computes no penalty), and
    # the candidate reproduces SI-25's decisive cycle 2
    # EXACTLY, so the treatment's difference is attributable
    # to the penalty term and not to the refactor.
    assert hp2["composition_config"] == "none"
    assert hp2["composition_lambda"] == 0.0
    assert hp2["penalty_train_initial"] is None
    assert hp2["penalty_train_final"] is None
    assert hp2["penalty_held_out_initial"] is None
    assert hp2["penalty_held_out_final"] is None
    assert hp2["total_train_loss_final"] == hp2["train_loss_final"]
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
    note2 = _hyperparams_notes(bundle_dir, manifest, sum2["edit_ids"][0])
    assert note2["composition_config"] == "none"
    assert note2["target_anchoring"] == "base-anchored"


# -- the decisive run: §15.1/§15.7/§15.8 ----------------------------------------------

def test_decisive_two_cycle_run(tmp_path):
    bundle_dir, policy, mind, seen, sum1, sum2 = _decisive_run(tmp_path)
    assert sum1["admitted"] == 1 and sum1["head"] == "d0001"
    # Cycle 2's verdict, AS MEASURED: REJECT on cumulative
    # identity drift; the run halts; head stays d0001.
    assert sum2["rejected"] == 1 and sum2["halted"] is True
    assert sum2["halt_verdict"] == "REJECT" and sum2["head"] == "d0001"
    assert sum2["delta_stack_depth"] == 1
    assert len(seen) == 2

    hp1, hp2 = seen[0]["hyperparams"], seen[1]["hyperparams"]
    # Cycle 1 reproduces SI-19 exactly (criterion 9, in-run),
    # under the default anchoring AND the default composition
    # configuration, its penalty fields null.
    assert hp1["training_signal_id"] == tt.TRAINING_SIGNAL_ID
    assert hp1["target_anchoring"] == "weights-under-training"
    assert hp1["composition_config"] == "none"
    assert hp1["composition_lambda"] == 0.0
    assert hp1["penalty_train_initial"] is None
    assert hp1["head"] == "base" and hp1["target_source_head"] == "base"
    assert seen[0]["record"]["blob"] == SI19_BLOB
    # Cycle 2: the full recorded triple — signal 2,
    # base-anchored, movement-composition — trained at d0001
    # under the unchanged freshness contract.
    assert hp2["training_signal_id"] == tt.TRAINING_SIGNAL_ID_2
    assert hp2["target_anchoring"] == "base-anchored"
    assert hp2["composition_config"] == "movement-composition"
    assert hp2["composition_lambda"] == 1.0
    assert hp2["head"] == "d0001" and hp2["target_source_head"] == "base"
    assert hp2["stack_depth"] == 1
    assert seen[1]["record"]["parent"] == "d0001"
    assert seen[1]["record"]["id"] == "d0002"
    assert seen[1]["record"]["blob"] == PEN_BLOB
    assert seen[1]["record"]["norm"] == pytest.approx(PEN_NORM,
                                                      abs=1e-12)
    # The two loss terms, separately (§7.2(iii)): the penalty
    # starts positive (the initial factors' movement aligns,
    # on the train partition) and ends negative (aggregate
    # opposition there); the total ends negative, as the RFC
    # said it can; the SIGNAL loss still improves on its
    # pre-training record on both partitions.
    assert hp2["penalty_train_initial"] == pytest.approx(
        PEN_TRAIN_INITIAL, abs=1e-12)
    assert hp2["penalty_train_final"] == pytest.approx(
        PEN_TRAIN_FINAL, abs=1e-12)
    assert hp2["penalty_held_out_initial"] == pytest.approx(
        PEN_HELD_INITIAL, abs=1e-12)
    assert hp2["penalty_held_out_final"] == pytest.approx(
        PEN_HELD_FINAL, abs=1e-12)
    assert hp2["total_train_loss_initial"] == pytest.approx(
        TOTAL_INITIAL, abs=1e-12)
    assert hp2["total_train_loss_final"] == pytest.approx(
        TOTAL_FINAL, abs=1e-12)
    assert hp2["total_train_loss_final"] < 0.0 < \
        hp2["total_train_loss_initial"]
    assert hp2["train_loss_initial"] == pytest.approx(
        0.0042998498036725735, abs=1e-12)
    assert hp2["train_loss_final"] == pytest.approx(SIG_TRAIN_FINAL,
                                                    abs=1e-12)
    assert hp2["base_held_out_loss"] == pytest.approx(BA_PRE_HELD,
                                                       abs=1e-12)
    assert hp2["held_out_loss_after"] == pytest.approx(SIG_HELD_AFTER,
                                                       abs=1e-12)
    assert hp2["held_out_loss_after"] < hp2["base_held_out_loss"]
    assert hp2["held_out_examples_closer"] == \
        hp2["held_out_examples"] == 16
    assert hp2["norm_projection_activations"] == 0
    assert hp1["wall_time_seconds"] > 0.0
    assert hp2["wall_time_seconds"] > 0.0

    manifest = load_bundle(bundle_dir)
    edit1, edit2 = sum1["edit_ids"][0], sum2["edit_ids"][0]
    # Gate honesty (§15.7): cycle 2's journaled checks 3/4
    # equal an independent ProbeContext evaluation of the
    # same candidate against the same pre-cycle head (d0001).
    fresh = TinyTransformerMind(
        bundle_dir, load_bundle(bundle_dir),
        training_signal_id=tt.TRAINING_SIGNAL_ID_2,
        target_anchoring="base-anchored",
        composition_config="movement-composition")
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
    # Cumulative vs incremental, labelled (§15.8): identity is
    # seed-absolute and FAILS; capability is head-incremental
    # and passes; echo is absolute vs the floor and passes.
    assert journaled["identity_continuity"]["measured"] == \
        pytest.approx(PEN_DRIFT, abs=1e-9)
    assert journaled["identity_continuity"]["passed"] is False
    assert "measured-fail" in journaled["identity_continuity"]["detail"]
    assert journaled["capability_floor"]["measured"] == \
        pytest.approx(PEN_REGRESSION, abs=1e-9)
    assert journaled["capability_floor"]["passed"] is True
    # Recorded configuration on BOTH verdict paths (§15.1):
    # cycle 1's note (ADMIT, under d0001) records the defaults;
    # cycle 2's note (REJECT, under its self-edit) records the
    # full triple with λ = 1.0 and the penalty's values.
    note1 = _hyperparams_notes(bundle_dir, manifest, "d0001")
    assert note1["training_signal_id"] == tt.TRAINING_SIGNAL_ID
    assert note1["target_anchoring"] == "weights-under-training"
    assert note1["composition_config"] == "none"
    note2 = _hyperparams_notes(bundle_dir, manifest, edit2)
    assert note2["training_signal_id"] == tt.TRAINING_SIGNAL_ID_2
    assert note2["target_anchoring"] == "base-anchored"
    assert note2["composition_config"] == "movement-composition"
    assert note2["composition_lambda"] == 1.0
    assert note2["penalty_train_final"] == pytest.approx(
        PEN_TRAIN_FINAL, abs=1e-12)
    assert note2["head"] == "d0001"
    assert note2["target_source_head"] == "base"
    # The landed head's cumulative drift after the REJECT is
    # unchanged — re-measured (§15.8), and cycle 1's VERIFY
    # stands as in SI-19.
    ok, report = _verify_probe_head(bundle_dir, manifest, policy=policy)
    assert ok is True
    assert report["identity"]["max_drift"] == pytest.approx(
        SI19_DRIFT, abs=1e-9)


# -- §15.5: the allowed-input composition record ---------------------------------------

def _allowed_record(bundle_dir, manifest, policy, candidate):
    head_w = assemble_weights(bundle_dir, manifest, policy)
    base_w = assemble_base_weights(bundle_dir, manifest, policy)
    cand_w = assemble_weights_with_candidate(bundle_dir, manifest,
                                             candidate, policy)
    train_seqs, held_seqs, _ = tt.signal_sequences(
        tt.TRAINING_SIGNAL_ID_2)
    return {
        "train": tt.movement_composition_record(head_w, base_w,
                                                cand_w, train_seqs),
        "held_out": tt.movement_composition_record(head_w, base_w,
                                                   cand_w, held_seqs),
    }


def test_allowed_input_composition_record(tmp_path):
    # Control candidate (re-run inside this slice).
    c_dir, c_pol, _cm, c_seen, _s1, _s2 = _control_run(
        tmp_path / "c", "tiny27ctrl")
    c_rec = _allowed_record(c_dir, load_bundle(c_dir), c_pol, c_seen[1])
    # Treatment candidate (the decisive run).
    t_dir, t_pol, _tm, t_seen, _u1, _u2 = _decisive_run(
        tmp_path / "t", "tiny27")
    t_rec = _allowed_record(t_dir, load_bundle(t_dir), t_pol, t_seen[1])

    # The §7.5 finding, measured for the CONTROL: on signal
    # 2's own train partition the control's movements ALREADY
    # OPPOSE (negative aggregate inner product, negative mean
    # penalty value) — as they do on its held-out partition.
    # The penalty therefore pushes on an already-pushed
    # direction in aggregate; the slice says so (§7.5).
    assert c_rec["train"]["aggregate_inner_product"] == pytest.approx(
        -1.6873837019991191, abs=1e-9)
    assert c_rec["train"]["mean_p"] == pytest.approx(
        -0.0016478356464835148, abs=1e-12)
    assert c_rec["train"]["aggregate_sign_agreement"] == pytest.approx(
        0.267578125, abs=1e-12)
    assert c_rec["held_out"]["aggregate_inner_product"] == \
        pytest.approx(-0.845596374259425, abs=1e-9)
    assert c_rec["held_out"]["mean_p"] == pytest.approx(
        -0.0016515554184754395, abs=1e-12)
    assert len(c_rec["train"]["per_input"]) == 32
    assert len(c_rec["held_out"]["per_input"]) == 16
    # The TREATMENT: training-input composition improves
    # (aggregate inner product falls, mean penalty falls) on
    # BOTH allowed partitions — the penalty did on allowed
    # inputs what it was designed to do there.
    assert t_rec["train"]["aggregate_inner_product"] == pytest.approx(
        -2.687755446349435, abs=1e-9)
    assert t_rec["train"]["aggregate_inner_product"] < \
        c_rec["train"]["aggregate_inner_product"]
    assert t_rec["train"]["mean_p"] == pytest.approx(
        -0.00262476117807562, abs=1e-12)
    assert t_rec["held_out"]["aggregate_inner_product"] == \
        pytest.approx(-1.3460015097625617, abs=1e-9)
    assert t_rec["held_out"]["mean_p"] == pytest.approx(
        -0.0026289091987550034, abs=1e-12)
    # Cross-check (two independent paths to the same number):
    # the record's forward-path mean penalty on the train
    # partition equals the training run's analytic penalty at
    # the end of training, and likewise on held-out.
    hp2 = t_seen[1]["hyperparams"]
    assert t_rec["train"]["mean_p"] == pytest.approx(
        hp2["penalty_train_final"], abs=1e-9)
    assert t_rec["held_out"]["mean_p"] == pytest.approx(
        hp2["penalty_held_out_final"], abs=1e-9)
    # m1 is a property of the landed head, not the candidate:
    # identical norms under both candidates, both partitions.
    for part in ("train", "held_out"):
        assert t_rec[part]["m1_norm"] == pytest.approx(
            c_rec[part]["m1_norm"], abs=1e-12)
    assert t_rec["train"]["m2_norm"] == pytest.approx(
        2.2303160062528544, abs=1e-9)
    assert c_rec["train"]["m2_norm"] == pytest.approx(
        1.6765305214306858, abs=1e-9)


# -- §15.8: probe diagnostics, three-run table, F1/F2/F3 --------------------------------

def _folded_update(candidate):
    parsed = parse_blob(candidate["blob_bytes"])
    target = candidate["record"]["targets"][0]
    A = np.asarray(parsed.tensors[target["factor_keys"][0]],
                   dtype=np.float64)
    B = np.asarray(parsed.tensors[target["factor_keys"][1]],
                   dtype=np.float64)
    return (float(target["alpha"]) / float(target["rank"])) * (B @ A)


def test_probe_diagnostics_three_run_and_falsification(tmp_path):
    bundle_dir, policy, mind, seen, sum1, sum2 = _decisive_run(tmp_path)
    manifest = load_bundle(bundle_dir)
    assert manifest["head"] == "d0001"  # candidate 2 never landed
    ctx = ProbeContext(bundle_dir, manifest, policy=policy)
    cand2 = seen[1]
    # (i) Per-probe contributions for the TREATMENT
    # (measurements; the incremental movement is a DERIVATION
    # from two measured forwards — composed minus d0001).
    expected = {
        "tiny-id-0": (0.17189218290150166, 0.15392476320266724, 0.32509003952145576),
        "tiny-id-1": (0.15088725090026855, 0.0669945478439331, 0.09391975402832031),
        "tiny-id-2": (0.11858986946754158, 0.1752924919128418, 0.1488508302718401),
        "tiny-id-3": (0.14782869815826416, 0.22549264132976532, 0.14755446463823318),
        "tiny-id-4": (0.11720322235487401, 0.1778034269809723, 0.10617947578430176),
        "tiny-id-5": (0.09891846403479576, 0.12068814411759377, 0.1450248844921589),
        "tiny-id-6": (0.12363102938979864, 0.18399151042103767, 0.15063166618347168),
        "tiny-id-7": (0.08795571327209473, 0.15689591597765684, 0.18760260567069054),
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
        got = rows[qid]
        for g, w in zip(got, want):
            assert g == pytest.approx(w, abs=1e-9)
    assert max(rows, key=lambda q: rows[q][2]) == "tiny-id-0"
    assert rows["tiny-id-0"][2] == pytest.approx(PEN_DRIFT, abs=1e-9)
    # Three-run composed-drift reading (each figure under its
    # own run's labels; never blended): the treatment's
    # maximum (0.32509) improves on SI-25's control maximum
    # (0.33290) and worsens on SI-23's (0.29634); per probe it
    # improves on SI-25 at 4 probes and worsens at 4.
    improved25 = [q for q in rows if rows[q][2] < S25_COMPOSED[q] - 1e-9]
    worsened25 = [q for q in rows if rows[q][2] > S25_COMPOSED[q] + 1e-9]
    assert sorted(improved25) == ["tiny-id-0", "tiny-id-2",
                                  "tiny-id-3", "tiny-id-4"]
    assert sorted(worsened25) == ["tiny-id-1", "tiny-id-5",
                                  "tiny-id-6", "tiny-id-7"]
    assert PEN_DRIFT < CTRL_DRIFT and S23_DRIFT < PEN_DRIFT
    # (ii) Alignment (derivation): in aggregate the penalised
    # movement opposes MORE than SI-25's control did
    # (-0.3271 < -0.2126); at the binding probe (tiny-id-0)
    # the movements still ALIGN, slightly worse than the
    # control's alignment there.
    a_all = np.concatenate(m1_all); b_all = np.concatenate(m2_all)
    agg_inner = float(np.dot(a_all, b_all))
    assert agg_inner == pytest.approx(-0.3271477846470207, abs=1e-9)
    assert agg_inner < CTRL_ALIGN_AGG_INNER
    assert float(np.mean(np.sign(a_all) == np.sign(b_all))) == \
        pytest.approx(0.390625, abs=1e-12)
    assert inners["tiny-id-0"] == pytest.approx(0.03801222744105431,
                                                abs=1e-9)
    assert inners["tiny-id-0"] > CTRL_WORST_INNER > 0.0
    # (iii) Folded updates (measurements): opposing in weight
    # space, more strongly than the control's; the penalised
    # update is the largest second update on record.
    d1u, d2u = _folded_update(seen[0]), _folded_update(cand2)
    assert float(np.sum(d1u * d2u)) == pytest.approx(
        -0.02013451284871203, abs=1e-9)
    assert float(np.linalg.norm(d2u)) == pytest.approx(PEN_NORM,
                                                       abs=1e-9)
    # Headroom arithmetic (derivation, labelled): the
    # treatment's incremental movement exceeds cycle 1's
    # remaining headroom at 7 of 8 probes (SI-25: 7 of 8).
    headroom = 0.25 - SI19_DRIFT
    assert sum(1 for v in rows.values() if v[1] > headroom) == 7
    # (iv) Falsification readings (RFC §7.6), stated from the
    # measured values asserted above and in the record test:
    # F1 MEASURED at the worst probe — training-input
    # composition improved (record test: -1.687 -> -2.688)
    # while probe composition at the worst probe worsened
    # (+0.03389 -> +0.03801).
    assert inners["tiny-id-0"] > CTRL_WORST_INNER
    # F2 MEASURED — aggregate probe composition improved
    # (inner fell below -0.2126) yet the composed maximum
    # stays above tol 0.25.
    assert agg_inner < CTRL_ALIGN_AGG_INNER and PEN_DRIFT > 0.25
    # F3 NOT measured as defined — the destination is still
    # approached (signal held-out loss falls ~5.3x, 16/16
    # examples closer), so opposition was NOT achieved by
    # abandoning the lean; but a PARTIAL trade is measured:
    # the penalised held-out residual is ~50x the control's.
    hp2 = seen[1]["hyperparams"]
    assert hp2["held_out_loss_after"] < hp2["base_held_out_loss"] / 5
    assert hp2["held_out_examples_closer"] == 16
    assert hp2["held_out_loss_after"] > 10 * CTRL_HELD_AFTER
    # (v) Composed capability (measurements via the gate's own
    # evaluation): head-incremental regression passes; echo
    # 1.0 vs floor 0.9375.
    rep = Gate(policy).evaluate(mind.propose([], ctx), cand2, ctx)
    checks = {c.name: c for c in rep.checks}
    assert checks["capability_floor"].measured == pytest.approx(
        PEN_REGRESSION, abs=1e-9)
    targets = ctx.capability_targets()
    correct = sum(
        1 for cid, x, _e in ctx.capability_suite()
        if int(np.argmax(np.asarray(ctx.forward_candidate(x, cand2))))
        == int(targets[cid]))
    assert correct == 16 and ctx.echo_floor() == 0.9375


# -- §15.6: determinism per cycle ---------------------------------------------------------

def test_decisive_run_repeats_exactly(tmp_path):
    _b1, _p1, _m1, seen1, s1a, s2a = _decisive_run(tmp_path / "a",
                                                  "tiny27")
    _b2, _p2, _m2, seen2, s1b, s2b = _decisive_run(tmp_path / "b",
                                                  "tiny27")
    assert s1a == s1b and s2a == s2b
    assert load_bundle(_b1)["deltas"] == load_bundle(_b2)["deltas"]
    for cand_a, cand_b in zip(seen1, seen2):
        assert cand_a["blob_bytes"] == cand_b["blob_bytes"]
        assert cand_a["record"] == cand_b["record"]
        hp_a = dict(cand_a["hyperparams"])
        hp_b = dict(cand_b["hyperparams"])
        hp_a.pop("wall_time_seconds"); hp_b.pop("wall_time_seconds")
        assert hp_a == hp_b


def test_penalised_train_delta_deterministic(tmp_path):
    bundle_dir, policy = _land_d0001(tmp_path)
    mind = TinyTransformerMind(
        bundle_dir, load_bundle(bundle_dir),
        training_signal_id=tt.TRAINING_SIGNAL_ID_2,
        target_anchoring="base-anchored",
        composition_config="movement-composition")
    intent = mind.propose([], None)
    first = mind.train_delta(intent)
    second = mind.train_delta(intent)
    assert first["blob_bytes"] == second["blob_bytes"]
    assert first["record"] == second["record"]
    assert first["record"]["blob"] == PEN_BLOB


# -- §15.10: VERIFY / unwind honesty / untouched scope --------------------------------------

def test_unwind_honesty_and_untouched_scope(tmp_path):
    bundle_dir, policy, mind, seen, sum1, sum2 = _decisive_run(tmp_path)
    assert sum1["admitted"] == 1 and sum2["rejected"] == 1
    manifest = load_bundle(bundle_dir)
    before_note = manifest["head"]
    # No authored two-delta stack exists: exactly one delta
    # landed, cycle 2's candidate was never written, and NO
    # unwind is claimed or exercised on an authored stack —
    # SI-21's fixture-proven unwind remains the only unwind
    # evidence (stated in the as-built doc).
    assert before_note == "d0001"
    assert len(manifest["deltas"]) == 1
    cand2_hex = seen[1]["record"]["blob"].split(":", 1)[1]
    assert not os.path.isfile(os.path.join(
        bundle_dir, "blobs", "sha256", cand2_hex + ".safetensors"))
    # Policy defaults and tolerances untouched.
    assert DEFAULT_POLICY["max_stack_depth"] == 8
    assert DEFAULT_POLICY["identity_tol"] == 0.25
    assert DEFAULT_POLICY["capability_eps"] == 0.25
    assert DEFAULT_POLICY["max_targets_per_delta"] == 1
    assert DEFAULT_POLICY["allow_additive"] is False
    on_disk = load_policy(bundle_dir)
    assert on_disk["identity_tol"] == 0.25
    assert on_disk["capability_eps"] == 0.25
    # Both signals' and both anchorings' definitions are
    # byte-for-byte SI-19/SI-23/SI-25's; the composition axis
    # adds exactly its two named values and λ's definition.
    assert tt.TARGET_ANCHORING == "weights-under-training"
    assert tt.TARGET_ANCHORINGS == ("weights-under-training",
                                    "base-anchored")
    assert set(tt.SIGNAL_REGISTRY) == {tt.TRAINING_SIGNAL_ID,
                                       tt.TRAINING_SIGNAL_ID_2}
    assert tt.CORRECTION_SCALE == 0.15
    assert tt.N_TRAIN == 32 and tt.N_HELD_OUT == 16
    assert copy.deepcopy(tt.COMPOSITION_CONFIGS) == \
        ("none", "movement-composition", "worst-input-composition")
    # Squash still fails closed on the transformer stack
    # (SI-12).
    with pytest.raises(SquashError):
        squash_bundle(bundle_dir, keep=0)
