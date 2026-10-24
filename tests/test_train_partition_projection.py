"""SI-31: train-partition-projection composition for the
miniature (RFC SI-30 §14 outline, §15 acceptance criteria).

Covers the §15 criteria, in order:

1. recorded composition configuration, four named values —
   the configuration serves exactly ``"none"``,
   ``"movement-composition"``, ``"worst-input-composition"``,
   and ``"train-partition-projection"`` and refuses every
   other value with the existing ``ValueError`` discipline
   (constructor, setter, intent, and ``train_candidate``
   dispatch); recorded selection asserted per cycle on both
   verdict paths, including the decisive cycle's full triple
   (signal 2, ``target_anchoring = "base-anchored"``,
   composition = ``"train-partition-projection"``). α is
   NOT independently settable (asserted: the constructor
   and the setter take no such argument, and no
   projection-fraction argument exists); exactly one
   projection decisive run is performed (no sweep, no
   partial-removal variant);
2. Phase 0 — published the projection rehearsal for all
   three existing decisive candidates (coefficients,
   parallel shares, rehearsal held-out records and norms,
   labelled diagnostics on already-judged candidates), the
   pre-registered F1′′/F2′′/F3′′ reading rules with F2′′'s
   clauses separated and F3′′'s thresholds carried from
   SI-29's rule, and the projection-correctness tolerance
   (1e-6 on the train-partition aggregate inner product,
   stated from the rehearsal figures) — as-built doc,
   ahead of the decisive numbers; the rehearsal values
   themselves are re-measured and asserted in the
   construction test below;
3. projection correctness, at the decisive head — the
   decisive candidate's train-partition aggregate inner
   product with ``m1`` is MEASURED (through the
   forward-path record, not the construction's algebra)
   within the Phase-0 tolerance of 0.0, published beside
   the held-out figure; the folded-update construction is
   verified against the re-run SI-25 control
   (Δ2′ = Δ2 − α·Δ1 in folded space). No new differentiable
   path was introduced (the construction is a post-training
   step), so SI-25's gradient check stands cited, not
   re-run — the as-built doc states this;
4. probe-blindness of training and construction, asserted
   — a spy on ``projection_coefficient`` during a real
   construction proves the coefficient's input set is
   exactly signal 2's train partition; disjointness from
   both probe suites re-asserted; a perturbed-probe-
   artifact bundle copy produces a byte-identical
   PROJECTED candidate;
5. coincidence at base — signal 2, base-anchored,
   projection-configured at base reproduces the
   established at-base run exactly (blob
   ``sha256:131cc242…45f4``, drift 0.16672966815531254,
   regression 0.2418281203135848), with the coefficient
   measured at α = 0.0 and the construction measured as
   the identity on both partitions;
6. determinism per cycle — a full decisive run repeated
   from the seed reproduces both candidates exactly
   (the projected candidate's blob included); the
   projection construction alone, applied twice to the
   same pre-projection candidate, is byte-identical
   across repeats (the deterministic ``train_delta``);
7. gate honesty per cycle — each cycle's journaled
   checks 3/4 equal an independent ``ProbeContext``
   evaluation of the same candidate against the pre-cycle
   head (abs 1e-12) — for cycle 2, of the PROJECTED
   candidate, the only candidate the gate is ever shown.
   The verdict is REJECT (the safety suite's size_budgets
   subcheck fails on rank 16 > rank_cap 8 AND the identity
   check fails at 0.4775387533009052 > tol 0.25), reported
   AS MEASURED. No ADMIT is required;
8. cumulative accounting, diagnostics, the six-run table,
   and the falsification readings published — per-probe
   drift for the treatment beside SI-23, SI-25, SI-27,
   and SI-29, alignment and folded-update diagnostics,
   candidate norm and ‖m2‖ beside the drift figures with
   the treatment's pre-projection norm beside its judged
   norm, the allowed-input composition records, the
   treatment's signal held-out record read under the
   PRE-REGISTERED F3′′ rule, and the F1′′/F2′′/F3′′
   readings with F2′′'s clauses reported separately;
9. first-cycle and all-control reproduction — cycle 1
   reproduces SI-19/21/23/25/27/29 exactly; the re-run
   ``"none"`` control reproduces SI-25's decisive
   candidate exactly; the re-run ``"movement-composition"``
   control reproduces SI-27's decisive candidate exactly;
   the re-run ``"worst-input-composition"`` control
   reproduces SI-29's decisive candidate exactly;
10. VERIFY, unwind, and untouched scope — cycle-1 VERIFY
    as in SI-19; cycle 2 never landed (REJECT), so no
    authored two-delta stack exists and none is claimed;
    exactly two authored cycles attempted; policy
    defaults, both signals', both anchorings', and all
    three prior composition configurations' definitions
    unchanged; ``si squash`` still fails closed.

The decisive run's driver shape: ONE ``TinyTransformerMind``
reused through ``si.run.run_cycles`` — cycle 1 (signal 1,
default anchoring, default composition), then the driver
explicitly reconfigures the same mind to signal 2
(``set_training_signal``), to base-anchored targets
(``set_target_anchoring``), AND to the projection
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

#: Phase-0 projection-correctness tolerance (RFC SI-30
#: §7.4/§15 criterion 3): the bound within which the built
#: slice's measured post-projection train-partition
#: aggregate inner product must sit to count as zero —
#: stated from the Phase-0 rehearsal figures (largest
#: rehearsal |aggregate| 1.41e-07 across the three existing
#: candidates under the float32 construction), before the
#: decisive run.
PROJECTION_TOLERANCE = 1e-6

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

WORST_BLOB = "sha256:11cab02e560e3661ea419fa0c8b6a95de25d3c7fb1ee23ec52ec8462a36b8db5"
WORST_NORM = 0.28252038537434354
WORST_DRIFT = 0.31215621903538704
WORST_REGRESSION = 0.11724743689410388
WORST_HELD_AFTER = 0.0006602814776996296
WORST_PROBE_INNER = 0.04944786620153699  # at tiny-id-0

S21_DRIFT = 0.34378434903919697
S23_DRIFT = 0.2963416576385498

#: F3′′ completeness boundary (RFC SI-30 §7.5, carried
#: from SI-29's rule unchanged in shape): 2× the re-run
#: SI-29 control's published held-out residual.
F3PP_BOUNDARY = 0.0013205629553992592
assert F3PP_BOUNDARY == 2 * WORST_HELD_AFTER

# SI-31 measured values (author's runs; the as-built doc
# publishes them with their labels). The decisive cycle-2
# verdict is REJECT — measured, not required, not tuned
# for: the safety suite fails on rank AND the identity
# check fails on drift.
PROJ_BLOB = "sha256:32d13f65d0de1de2901fcd1c1a5cdece12cdfe441454f5d75e8932941827a5c1"
PROJ_NORM = 0.2890208630043799
PROJ_BYTES = 4408
PROJ_DRIFT = 0.4775387533009052
PROJ_REGRESSION = 0.177281160838902
PROJ_ALPHA = -0.8450027779470831
PROJ_NUMERATOR = -1.6873837019991194
PROJ_DENOMINATOR = 1.9968972245257979
PROJ_PRE_NORM = 0.25155244947498906
PROJ_TRAIN_AGG_INNER = -4.8567197048765554e-08
PROJ_HELD_AGG_INNER = -0.0009897609251914505
PROJ_HELD_AFTER = 0.00140492372619575
PROJ_TRAIN_LOSS_FINAL = 0.0014032997323342922
BA_PRE_HELD = 0.002757805049077469


def _seeded(tmp_path, name="tiny31"):
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


def _decisive_run(tmp_path, name="tiny31"):
    return _run_with_cycle2_config(
        tmp_path, name, tt.TRAIN_PARTITION_PROJECTION)


def _control_run(tmp_path, name="tiny31ctrl"):
    return _run_with_cycle2_config(tmp_path, name, None)


def _movement_control_run(tmp_path, name="tiny31pen"):
    return _run_with_cycle2_config(
        tmp_path, name, tt.MOVEMENT_COMPOSITION)


def _worst_control_run(tmp_path, name="tiny31worst"):
    return _run_with_cycle2_config(
        tmp_path, name, tt.WORST_INPUT_COMPOSITION)


def _land_d0001(tmp_path, name="tiny31"):
    bundle_dir = _seeded(tmp_path, name)
    policy = _policy(bundle_dir)
    mind, _seen = _spying_mind(bundle_dir)
    res = run_cycle(bundle_dir, mind, gate=Gate(policy))
    assert res["verdict"] == "ADMIT" and res["head"] == "d0001"
    return bundle_dir, policy


def _folded_update(candidate):
    parsed = parse_blob(candidate["blob_bytes"])
    target = candidate["record"]["targets"][0]
    A = np.asarray(parsed.tensors[target["factor_keys"][0]],
                   dtype=np.float64)
    B = np.asarray(parsed.tensors[target["factor_keys"][1]],
                   dtype=np.float64)
    return (float(target["alpha"]) / float(target["rank"])) * (B @ A)


# -- §15.1: recorded composition configuration, four values -----------------------

def test_projection_configuration_and_refusals(tmp_path):
    assert tt.COMPOSITION_CONFIGS == ("none", "movement-composition",
                                      "worst-input-composition",
                                      "train-partition-projection")
    assert tt.DEFAULT_COMPOSITION_CONFIG == "none"
    assert tt.TRAIN_PARTITION_PROJECTION == "train-partition-projection"
    assert tt.COMPOSITION_LAMBDA == 1.0
    assert tt.WORST_INPUT_LAMBDA == 1.0 and tt.WORST_INPUT_K == 4
    for good in tt.COMPOSITION_CONFIGS:
        assert tt.validate_composition_config(good) == good
    for bad in ("projection", "train-projection", "", "NONE", None,
                "orthogonalization"):
        with pytest.raises(ValueError,
                           match="unknown composition configuration"):
            tt.validate_composition_config(bad)

    bundle_dir = _seeded(tmp_path)
    mind = TinyTransformerMind(bundle_dir, load_bundle(bundle_dir))
    assert mind.composition_config == "none"
    mind_p = TinyTransformerMind(
        bundle_dir, load_bundle(bundle_dir),
        composition_config="train-partition-projection")
    assert mind_p.composition_config == "train-partition-projection"
    assert mind_p.propose([], None).composition_config == \
        "train-partition-projection"
    with pytest.raises(ValueError,
                       match="unknown composition configuration"):
        TinyTransformerMind(bundle_dir, load_bundle(bundle_dir),
                            composition_config="probe-driven")
    # α is NOT independently settable: neither the
    # constructor nor the setter takes such an argument,
    # and no projection-fraction argument exists anywhere.
    with pytest.raises(TypeError):
        TinyTransformerMind(bundle_dir, load_bundle(bundle_dir),
                            projection_alpha=-0.5)
    with pytest.raises(TypeError):
        TinyTransformerMind(bundle_dir, load_bundle(bundle_dir),
                            projection_fraction=0.5)
    with pytest.raises(TypeError):
        TinyTransformerMind(bundle_dir, load_bundle(bundle_dir),
                            composition_lambda=0.5)
    mind.set_composition_config("train-partition-projection")
    assert mind.composition_config == "train-partition-projection"
    mind.set_composition_config("none")
    assert mind.composition_config == "none"
    with pytest.raises(ValueError,
                       match="unknown composition configuration"):
        mind.set_composition_config("learned")
    with pytest.raises(TypeError):
        mind.set_composition_config("train-partition-projection",
                                    alpha=0.5)
    with pytest.raises(TypeError):
        mind.set_composition_config("train-partition-projection",
                                    fraction=0.5)
    base = dict(kind=KIND, reason="SI-31 refusal probe",
                parents=["base"], source_seqs=[1])
    for bad in ("train_partition_projection", "", "projection"):
        with pytest.raises(ValueError,
                           match="unknown composition configuration"):
            mind.train_delta(Intent(
                training_signal_id=tt.TRAINING_SIGNAL_ID,
                composition_config=bad, targets=[tt.TARGET_NAME],
                **base))
    mind.set_training_signal(tt.TRAINING_SIGNAL_ID_2)
    mind.set_target_anchoring("base-anchored")
    mind.set_composition_config("train-partition-projection")
    intent = mind.propose([], None)
    assert intent.training_signal_id == tt.TRAINING_SIGNAL_ID_2
    assert intent.target_anchoring == "base-anchored"
    assert intent.composition_config == "train-partition-projection"
    assert intent.to_dict()["composition_config"] == \
        "train-partition-projection"


# -- §15.5: coincidence at base ------------------------------------------------------

def test_projection_signal2_at_base_coincides(tmp_path):
    bundle_dir = _seeded(tmp_path)
    policy = _policy(bundle_dir)
    before = _blobs_total(bundle_dir)
    mind, seen = _spying_mind(
        bundle_dir, training_signal_id=tt.TRAINING_SIGNAL_ID_2,
        target_anchoring="base-anchored",
        composition_config="train-partition-projection")
    res = run_cycle(bundle_dir, mind, gate=Gate(policy))
    assert res["verdict"] == "ADMIT" and res["head"] == "d0001"
    assert res["verified"] is True and res["rolled_back"] is False
    hp = seen[0]["hyperparams"]
    assert hp["composition_config"] == "train-partition-projection"
    # The coefficient, MEASURED at base: both aggregate
    # inner products in α's definition vanish (m1 is the
    # zero vector at base), α is 0.0 by the definition's
    # denominator rule, and the construction is the
    # identity — the candidate keeps its trained rank-8
    # factors and reproduces the at-base run EXACTLY.
    assert hp["projection_alpha"] == 0.0
    assert hp["projection_numerator"] == 0.0
    assert hp["projection_denominator"] == 0.0
    assert hp["projection_candidate_rank"] == 8
    assert hp["projection_train_aggregate_inner_product"] == 0.0
    assert hp["projection_held_out_aggregate_inner_product"] == 0.0
    assert hp["projection_post_norm"] == pytest.approx(
        hp["projection_pre_norm"], abs=1e-6)
    assert hp["held_out_loss_after"] == \
        hp["projection_pre_held_out_loss_after"]
    assert hp["penalty_train_initial"] is None
    assert hp["worst_input_train_initial"] is None
    assert seen[0]["record"]["blob"] == S2_AT_BASE_BLOB
    assert seen[0]["record"]["targets"][0]["rank"] == 8
    manifest = load_bundle(bundle_dir)
    assert _blobs_total(bundle_dir) == before + 2360
    checks, _row = _checks_of_edit(bundle_dir, manifest, res["edit_id"])
    assert checks["identity_continuity"]["measured"] == pytest.approx(
        S2_AT_BASE_DRIFT, abs=1e-9)
    assert checks["capability_floor"]["measured"] == pytest.approx(
        S2_AT_BASE_REGRESSION, abs=1e-9)
    note = _hyperparams_notes(bundle_dir, manifest, "d0001")
    assert note["composition_config"] == "train-partition-projection"
    assert note["projection_alpha"] == 0.0
    ok, report = _verify_probe_head(bundle_dir, manifest, policy=policy)
    assert ok is True
    assert report["identity"]["max_drift"] == pytest.approx(
        S2_AT_BASE_DRIFT, abs=1e-9)


# -- §15.3: projection correctness — the coefficient and the construction -----------

def test_projection_coefficient_and_factors_unit():
    seqs = [[1, 2], [3, 4]]
    m1 = {(1, 2): np.array([1.0, 0.0]), (3, 4): np.array([0.0, 2.0])}
    m2 = {(1, 2): np.array([3.0, 1.0]), (3, 4): np.array([1.0, 1.0])}
    coeff = tt.projection_coefficient(m1, m2, seqs)
    # numerator = 3 + 2 = 5; denominator = 1 + 4 = 5.
    assert coeff["numerator"] == pytest.approx(5.0, abs=1e-12)
    assert coeff["denominator"] == pytest.approx(5.0, abs=1e-12)
    assert coeff["alpha"] == pytest.approx(1.0, abs=1e-12)
    zero = {(1, 2): np.zeros(2), (3, 4): np.zeros(2)}
    coeff0 = tt.projection_coefficient(zero, m2, seqs)
    assert coeff0["alpha"] == 0.0 and coeff0["denominator"] == 0.0
    # The SVD factorisation carries the folded update at
    # rank 16 / alpha 16 (scale 1.0), exactly to float64
    # factorisation precision.
    rng = np.random.default_rng(7)
    d1 = rng.standard_normal((32, 8)) @ rng.standard_normal((8, 32))
    d2 = rng.standard_normal((32, 8)) @ rng.standard_normal((8, 32))
    dp = d2 - (-0.845) * d1
    A, B = tt.projection_factors(dp)
    assert A.shape == (16, 32) and B.shape == (32, 16)
    assert np.max(np.abs(B @ A - dp)) < 1e-10
    assert tt.PROJECTION_RANK == 16 and tt.PROJECTION_ALPHA == 16


def test_projection_construction_matches_control(tmp_path):
    # The decisive candidate's construction, verified in
    # folded space against the re-run SI-25 control: the
    # intermediate is the control's candidate, and the
    # judged candidate's folded update is Δ2 − α·Δ1.
    _b1, _p1, _m1, seen_ctrl, _s1, _s2 = _control_run(
        tmp_path / "c", "tiny31ctrl")
    b2, p2, _m2, seen_proj, _t1, _t2 = _decisive_run(
        tmp_path / "t", "tiny31")
    assert seen_ctrl[1]["record"]["blob"] == CTRL_BLOB
    assert seen_proj[1]["record"]["blob"] == PROJ_BLOB
    hp = seen_proj[1]["hyperparams"]
    alpha = hp["projection_alpha"]
    assert alpha == pytest.approx(PROJ_ALPHA, abs=1e-12)
    assert hp["projection_numerator"] == pytest.approx(
        PROJ_NUMERATOR, abs=1e-9)
    assert hp["projection_denominator"] == pytest.approx(
        PROJ_DENOMINATOR, abs=1e-9)
    d1 = _folded_update(seen_proj[0])
    d2 = _folded_update(seen_ctrl[1])
    dp = _folded_update(seen_proj[1])
    assert np.max(np.abs(dp - (d2 - alpha * d1))) < 1e-5
    # Projection correctness MEASURED through the forward
    # path (movement_composition_record), not the algebra:
    # the projected candidate's train-partition aggregate
    # inner product sits within the Phase-0 tolerance of
    # 0.0, while the control's is the published −1.687….
    manifest = load_bundle(b2)
    head_w = assemble_weights(b2, manifest, p2)
    base_w = assemble_base_weights(b2, manifest, p2)
    cand_w = assemble_weights_with_candidate(b2, manifest,
                                             seen_proj[1], p2)
    train_seqs, held_seqs, _ = tt.signal_sequences(
        tt.TRAINING_SIGNAL_ID_2)
    rec_tr = tt.movement_composition_record(head_w, base_w,
                                            cand_w, train_seqs)
    rec_ho = tt.movement_composition_record(head_w, base_w,
                                            cand_w, held_seqs)
    assert rec_tr["aggregate_inner_product"] == pytest.approx(
        PROJ_TRAIN_AGG_INNER, abs=1e-12)
    assert abs(rec_tr["aggregate_inner_product"]) <= \
        PROJECTION_TOLERANCE
    assert rec_tr["aggregate_inner_product"] == pytest.approx(
        hp["projection_train_aggregate_inner_product"], abs=1e-12)
    assert rec_ho["aggregate_inner_product"] == pytest.approx(
        PROJ_HELD_AGG_INNER, abs=1e-9)
    assert rec_ho["aggregate_inner_product"] == pytest.approx(
        hp["projection_held_out_aggregate_inner_product"],
        abs=1e-12)
    assert rec_tr["m2_norm"] == pytest.approx(1.176822205010781,
                                              abs=1e-9)


# -- §15.4: probe-blindness of training and construction, asserted -------------------

def test_projection_training_is_probe_blind(tmp_path, monkeypatch):
    bundle_dir, policy = _land_d0001(tmp_path)
    train_seqs, held_seqs, _ = tt.signal_sequences(
        tt.TRAINING_SIGNAL_ID_2)
    train_set = {tuple(s) for s in train_seqs}
    assert train_set.isdisjoint({tuple(s) for s in identity_inputs()})
    assert train_set.isdisjoint({tuple(s) for s in capability_inputs()})
    assert train_set.isdisjoint({tuple(s) for s in held_seqs})
    captured = []
    orig = tt.projection_coefficient

    def spy(m1_map, m2_map, seqs):
        captured.append((set(m1_map.keys()), set(m2_map.keys()),
                         {tuple(s) for s in seqs}))
        return orig(m1_map, m2_map, seqs)

    monkeypatch.setattr(tt, "projection_coefficient", spy)
    mind = TinyTransformerMind(
        bundle_dir, load_bundle(bundle_dir),
        training_signal_id=tt.TRAINING_SIGNAL_ID_2,
        target_anchoring="base-anchored",
        composition_config="train-partition-projection")
    cand = mind.train_delta(mind.propose([], None))
    assert cand["record"]["blob"] == PROJ_BLOB
    assert captured, "the projection construction never ran"
    for m1_set, m2_set, seq_set in captured:
        assert m1_set == train_set
        assert m2_set == train_set
        assert seq_set == train_set
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
        kind=KIND, reason="SI-31 probe-blindness probe",
        parents=["d0001"], source_seqs=[1],
        training_signal_id=tt.TRAINING_SIGNAL_ID_2,
        target_anchoring="base-anchored",
        composition_config="train-partition-projection",
        targets=[tt.TARGET_NAME])
    cand2 = tt.train_candidate(copy_dir, manifest, intent,
                               policy=policy)
    assert cand2["blob_bytes"] == cand["blob_bytes"]
    assert cand2["record"]["blob"] == PROJ_BLOB


# -- §15.9: all three controls, re-run inside the slice --------------------------------

def test_none_control_rerun_reproduces_si25(tmp_path):
    bundle_dir, policy, mind, seen, sum1, sum2 = _control_run(tmp_path)
    assert sum1["admitted"] == 1 and sum2["rejected"] == 1
    hp2 = seen[1]["hyperparams"]
    assert hp2["composition_config"] == "none"
    assert hp2["projection_alpha"] is None
    assert hp2["projection_train_aggregate_inner_product"] is None
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
    assert hp2["projection_alpha"] is None
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


def test_worst_control_rerun_reproduces_si29(tmp_path):
    bundle_dir, policy, mind, seen, sum1, sum2 = \
        _worst_control_run(tmp_path)
    assert sum1["admitted"] == 1 and sum2["rejected"] == 1
    hp2 = seen[1]["hyperparams"]
    assert hp2["composition_config"] == "worst-input-composition"
    assert hp2["composition_lambda_w"] == 1.0
    assert hp2["composition_worst_k"] == 4
    assert hp2["projection_alpha"] is None
    assert seen[1]["record"]["blob"] == WORST_BLOB
    assert seen[1]["record"]["norm"] == pytest.approx(WORST_NORM,
                                                      abs=1e-12)
    manifest = load_bundle(bundle_dir)
    checks, _row = _checks_of_edit(bundle_dir, manifest,
                                   sum2["edit_ids"][0])
    assert checks["identity_continuity"]["measured"] == \
        pytest.approx(WORST_DRIFT, abs=1e-9)
    assert checks["capability_floor"]["measured"] == \
        pytest.approx(WORST_REGRESSION, abs=1e-9)
    note2 = _hyperparams_notes(bundle_dir, manifest, sum2["edit_ids"][0])
    assert note2["composition_config"] == "worst-input-composition"


# -- the decisive run: §15.1/§15.7 ------------------------------------------------------

def test_decisive_two_cycle_run(tmp_path):
    bundle_dir, policy, mind, seen, sum1, sum2 = _decisive_run(tmp_path)
    assert sum1["admitted"] == 1 and sum1["head"] == "d0001"
    assert sum2["rejected"] == 1 and sum2["halted"] is True
    assert sum2["halt_verdict"] == "REJECT" and sum2["head"] == "d0001"
    assert sum2["delta_stack_depth"] == 1
    assert len(seen) == 2

    hp1, hp2 = seen[0]["hyperparams"], seen[1]["hyperparams"]
    assert hp1["composition_config"] == "none"
    assert hp1["projection_alpha"] is None
    assert seen[0]["record"]["blob"] == SI19_BLOB
    assert hp2["training_signal_id"] == tt.TRAINING_SIGNAL_ID_2
    assert hp2["target_anchoring"] == "base-anchored"
    assert hp2["composition_config"] == "train-partition-projection"
    assert hp2["composition_lambda"] == 0.0
    assert hp2["composition_lambda_w"] == 0.0
    assert hp2["composition_worst_k"] is None
    assert hp2["penalty_train_initial"] is None
    assert hp2["worst_input_train_initial"] is None
    assert hp2["head"] == "d0001" and hp2["target_source_head"] == "base"
    assert hp2["stack_depth"] == 1
    assert seen[1]["record"]["parent"] == "d0001"
    assert seen[1]["record"]["id"] == "d0002"
    assert seen[1]["record"]["blob"] == PROJ_BLOB
    assert seen[1]["record"]["bytes"] == PROJ_BYTES
    assert seen[1]["record"]["norm"] == pytest.approx(PROJ_NORM,
                                                      abs=1e-12)
    # The constructed candidate: rank 16 / alpha 16 (the
    # difference of two rank-8 updates), scale 1.0 as ever.
    target = seen[1]["record"]["targets"][0]
    assert target["rank"] == 16 and target["alpha"] == 16
    # The construction's accounting: α measured from the
    # train partition; the intermediate is SI-25's
    # candidate (its held-out record under its own label);
    # the judged candidate's records under the cycle's.
    assert hp2["projection_alpha"] == pytest.approx(PROJ_ALPHA,
                                                    abs=1e-12)
    assert hp2["projection_candidate_rank"] == 16
    assert hp2["projection_pre_norm"] == pytest.approx(PROJ_PRE_NORM,
                                                       abs=1e-12)
    assert hp2["projection_post_norm"] == pytest.approx(PROJ_NORM,
                                                        abs=1e-12)
    assert hp2["projection_pre_held_out_loss_after"] == pytest.approx(
        CTRL_HELD_AFTER, abs=1e-12)
    assert hp2["projection_pre_held_out_examples_closer"] == 16
    assert hp2["train_loss_final"] == pytest.approx(
        9.412649442227854e-06, abs=1e-12)
    assert hp2["projection_train_loss_final"] == pytest.approx(
        PROJ_TRAIN_LOSS_FINAL, abs=1e-12)
    assert hp2["base_held_out_loss"] == pytest.approx(BA_PRE_HELD,
                                                       abs=1e-12)
    assert hp2["held_out_loss_after"] == pytest.approx(PROJ_HELD_AFTER,
                                                       abs=1e-12)
    assert hp2["held_out_loss_after"] < hp2["base_held_out_loss"]
    assert hp2["held_out_examples_closer"] == 16
    assert hp2["held_out_examples"] == 16
    assert hp2["norm_projection_activations"] == 0
    assert abs(hp2["projection_train_aggregate_inner_product"]) <= \
        PROJECTION_TOLERANCE

    manifest = load_bundle(bundle_dir)
    edit1, edit2 = sum1["edit_ids"][0], sum2["edit_ids"][0]
    fresh = TinyTransformerMind(
        bundle_dir, load_bundle(bundle_dir),
        training_signal_id=tt.TRAINING_SIGNAL_ID_2,
        target_anchoring="base-anchored",
        composition_config="train-partition-projection")
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
        pytest.approx(PROJ_DRIFT, abs=1e-9)
    assert journaled["identity_continuity"]["passed"] is False
    assert journaled["capability_floor"]["measured"] == \
        pytest.approx(PROJ_REGRESSION, abs=1e-9)
    assert journaled["capability_floor"]["passed"] is True
    # The safety suite ALSO fails, on the rank the
    # construction necessarily carries — measured and
    # reported, never hidden (see the as-built doc).
    assert journaled["safety_bounds"]["passed"] is False
    safety = {c["name"]: c for c in
              journaled["safety_bounds"]["subchecks"]} \
        if "subchecks" in journaled["safety_bounds"] else {}
    if safety:
        assert safety["size_budgets"]["passed"] is False
        assert "rank 16 exceeds rank_cap 8" in \
            safety["size_budgets"]["detail"]
    else:  # journaled form without subcheck detail
        assert "rank 16 exceeds rank_cap 8" in \
            journaled["safety_bounds"]["detail"]
    note1 = _hyperparams_notes(bundle_dir, manifest, "d0001")
    assert note1["composition_config"] == "none"
    note2 = _hyperparams_notes(bundle_dir, manifest, edit2)
    assert note2["training_signal_id"] == tt.TRAINING_SIGNAL_ID_2
    assert note2["target_anchoring"] == "base-anchored"
    assert note2["composition_config"] == "train-partition-projection"
    assert note2["projection_alpha"] == pytest.approx(PROJ_ALPHA,
                                                      abs=1e-12)
    assert note2["head"] == "d0001"
    ok, report = _verify_probe_head(bundle_dir, manifest, policy=policy)
    assert ok is True
    assert report["identity"]["max_drift"] == pytest.approx(
        SI19_DRIFT, abs=1e-9)


# -- §15.8: probe diagnostics, six-run table, F1′′/F2′′/F3′′ ------------------------------

def test_probe_diagnostics_six_run_and_falsification(tmp_path):
    bundle_dir, policy, mind, seen, sum1, sum2 = _decisive_run(tmp_path)
    manifest = load_bundle(bundle_dir)
    assert manifest["head"] == "d0001"
    ctx = ProbeContext(bundle_dir, manifest, policy=policy)
    cand2 = seen[1]
    expected = {
        "tiny-id-0": (0.17189218290150166, 0.3063734769821167, 0.4775387533009052),
        "tiny-id-1": (0.15088725090026855, 0.06947946548461914, 0.2203667163848877),
        "tiny-id-2": (0.11858986946754158, 0.14929723739624023, 0.14972496032714844),
        "tiny-id-3": (0.14782869815826416, 0.14890391938388348, 0.2727522850036621),
        "tiny-id-4": (0.11720322235487401, 0.13440751284360886, 0.21748418174684048),
        "tiny-id-5": (0.09891846403479576, 0.09731395542621613, 0.19604265317320824),
        "tiny-id-6": (0.12363102938979864, 0.14727314934134483, 0.13208984676748514),
        "tiny-id-7": (0.08795571327209473, 0.13717103376984596, 0.20603866130113602),
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
    # Six-run composed maxima, each under its own labels:
    # SI-21 0.34378, SI-23 0.29634, SI-25 0.33290, SI-27
    # 0.32509, SI-29 0.31216, SI-31 (treatment, projected)
    # 0.47754 — worse than every predecessor, above tol.
    assert PROJ_DRIFT > S21_DRIFT > CTRL_DRIFT > PEN_DRIFT > \
        WORST_DRIFT > S23_DRIFT > 0.25
    # Alignment (derivation): in aggregate on the probes
    # the projected movement ALIGNS with the first
    # (+0.2158, positive — every predecessor's treatment
    # opposed in aggregate); at tiny-id-0 it aligns far
    # more than SI-29's treatment did (+0.0494).
    a_all = np.concatenate(m1_all); b_all = np.concatenate(m2_all)
    agg_inner = float(np.dot(a_all, b_all))
    assert agg_inner == pytest.approx(0.21576810032436344, abs=1e-9)
    assert agg_inner > 0.0
    assert float(np.mean(np.sign(a_all) == np.sign(b_all))) == \
        pytest.approx(0.5, abs=1e-12)
    assert inners["tiny-id-0"] == pytest.approx(0.14235826783471692,
                                                abs=1e-9)
    assert inners["tiny-id-0"] > WORST_PROBE_INNER > 0.0
    assert float(np.linalg.norm(b_all)) == pytest.approx(
        0.7834031060820517, abs=1e-9)
    d1u, d2u = _folded_update(seen[0]), _folded_update(cand2)
    assert float(np.sum(d1u * d2u)) == pytest.approx(
        0.03542756883828178, abs=1e-9)
    assert float(np.linalg.norm(d2u)) == pytest.approx(PROJ_NORM,
                                                       abs=1e-9)
    headroom = 0.25 - SI19_DRIFT
    assert sum(1 for v in rows.values() if v[1] > headroom) == 7
    # F1′′ MEASURED — the projected candidate's
    # train-partition composition is zero as constructed
    # and as measured (|aggregate| ≤ Phase-0 tolerance),
    # while its probe composition at tiny-id-0 is worse
    # than SI-29's treatment's (+0.0494 → +0.1424).
    hp2 = seen[1]["hyperparams"]
    assert abs(hp2["projection_train_aggregate_inner_product"]) <= \
        PROJECTION_TOLERANCE
    assert inners["tiny-id-0"] > WORST_PROBE_INNER
    # F2′′ judged as TWO SEPARATE clauses (the repaired
    # form): (a) the transfer clause FAILS — the projected
    # candidate's inner product at its own worst probe
    # (tiny-id-0, +0.1424) does not improve on SI-29's
    # treatment's figure at that probe (+0.0494), and its
    # aggregate probe composition (+0.2158, aligned) is
    # published beside SI-27's (−0.3271) and SI-29's
    # (−0.2875) with no no-worse conjunction required;
    # (b) the maximum clause's fact HOLDS — the composed
    # maximum stays above tol (0.4775 > 0.25). F2′′ as a
    # reading is therefore NOT measured: its transfer
    # clause failed.
    assert not (inners["tiny-id-0"] < WORST_PROBE_INNER)
    assert PROJ_DRIFT > 0.25
    # F3′′ read under the PRE-REGISTERED rule (Phase 0,
    # as-built doc): approached iff (a) held-out loss
    # falls, (b) at least 12/16 held-out examples are
    # closer, and (c) the held-out residual is at most
    # 2× the re-run SI-29 control's residual
    # (0.0013205629553992592). (a) and (b) hold; (c)
    # FAILS (0.0014049… > 0.0013205…) — so F3′′ IS
    # measured: the destination counts as traded away
    # under the rule, by its completeness clause.
    assert hp2["held_out_loss_after"] < hp2["base_held_out_loss"]
    assert hp2["held_out_examples_closer"] == 16 >= 12
    assert hp2["held_out_loss_after"] > F3PP_BOUNDARY
    rep = Gate(policy).evaluate(mind.propose([], ctx), cand2, ctx)
    checks = {c.name: c for c in rep.checks}
    assert checks["capability_floor"].measured == pytest.approx(
        PROJ_REGRESSION, abs=1e-9)
    targets = ctx.capability_targets()
    correct = sum(
        1 for cid, x, _e in ctx.capability_suite()
        if int(np.argmax(np.asarray(ctx.forward_candidate(x, cand2))))
        == int(targets[cid]))
    assert correct == 16 and ctx.echo_floor() == 0.9375


# -- §15.6: determinism per cycle ----------------------------------------------------------

def test_decisive_run_repeats_exactly(tmp_path):
    _b1, _p1, _m1, seen1, s1a, s2a = _decisive_run(tmp_path / "a",
                                                  "tiny31")
    _b2, _p2, _m2, seen2, s1b, s2b = _decisive_run(tmp_path / "b",
                                                  "tiny31")
    assert s1a == s1b and s2a == s2b
    assert load_bundle(_b1)["deltas"] == load_bundle(_b2)["deltas"]
    for cand_a, cand_b in zip(seen1, seen2):
        assert cand_a["blob_bytes"] == cand_b["blob_bytes"]
        assert cand_a["record"] == cand_b["record"]
        hp_a = dict(cand_a["hyperparams"])
        hp_b = dict(cand_b["hyperparams"])
        hp_a.pop("wall_time_seconds"); hp_b.pop("wall_time_seconds")
        assert hp_a == hp_b


def test_projection_train_delta_deterministic(tmp_path):
    bundle_dir, policy = _land_d0001(tmp_path)
    mind = TinyTransformerMind(
        bundle_dir, load_bundle(bundle_dir),
        training_signal_id=tt.TRAINING_SIGNAL_ID_2,
        target_anchoring="base-anchored",
        composition_config="train-partition-projection")
    intent = mind.propose([], None)
    first = mind.train_delta(intent)
    second = mind.train_delta(intent)
    assert first["blob_bytes"] == second["blob_bytes"]
    assert first["record"] == second["record"]
    assert first["record"]["blob"] == PROJ_BLOB


# -- §15.10: VERIFY / unwind honesty / untouched scope ---------------------------------------

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
    assert tt.COMPOSITION_LAMBDA == 1.0
    assert tt.WORST_INPUT_LAMBDA == 1.0 and tt.WORST_INPUT_K == 4
    assert copy.deepcopy(tt.COMPOSITION_CONFIGS) == \
        ("none", "movement-composition", "worst-input-composition",
         "train-partition-projection")
    with pytest.raises(SquashError):
        squash_bundle(bundle_dir, keep=0)
