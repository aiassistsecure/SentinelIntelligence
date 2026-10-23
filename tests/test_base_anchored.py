"""SI-25: base-anchored targets for the miniature
(RFC SI-24 §9 outline, §10 acceptance criteria).

Covers the §10 criteria, in order:

1. recorded anchoring — the configuration serves exactly the
   two named anchorings and refuses every other value with
   the existing ``ValueError`` discipline (constructor,
   setter, intent, and ``train_candidate`` dispatch); a test
   asserts, per cycle on both verdict paths, that the
   journaled ``training-hyperparams`` note's
   ``target_anchoring`` and ``target_source_head`` equal the
   anchoring and target source the driver configured —
   including the decisive cycle's ``head = d0001`` /
   ``target_source_head = base`` disagreement. No autonomous
   selection exists: ``propose`` emits exactly the configured
   anchoring, and the module constant still names only the
   default;
2. coincidence at base — configured with signal 2 and
   base-anchoring, a first cycle at base reproduces SI-23's
   at-base run exactly (blob, drift, regression); configured
   with signal 1 under EITHER anchoring, a first cycle
   reproduces SI-19/SI-21/SI-23's ``d0001`` exactly
   (criterion 9, asserted here and in the decisive run);
3. gradient check for the base-anchored loss — finite
   differences of signal 2's base-anchored loss (targets
   derived from the base, hidden states and readout from the
   weights under training) at head ``d0001`` AND at base
   under the new code path, tolerance 1e-5 unchanged;
4. target-source assembly, fail-closed and byte-honest — a
   test tells the two assemblies apart (base view's readout
   equals the recorded base tensor; head view's readout
   equals base + ``d0001``'s fold); the verified manifest is
   never mutated by the base view; a broken base assembly
   refuses base-anchored training with NO fallback to
   head-derived targets (the same bundle trains fine
   head-relative); blob-store growth per cycle is
   attributable only to a landed candidate;
5. echo-trap records under the new semantics — signal 2's
   pre-training held-out loss at ``d0001`` under
   base-anchoring is a genuine measurement
   (0.002757805049077469 — NOT the closed form, which is
   never quoted for a non-base head) and training improves
   on it; the signal-1 base-anchored residual record at
   ``d0001`` measures 3.2118398935406443e-06 (the landed head
   essentially occupies signal 1's fixed destination) and
   its training does NOT improve, so the existing
   non-improvement refusal stands and is reported as the
   finding — no substitute is run as the decisive cycle;
6. determinism per cycle — a full decisive run repeated
   from the seed reproduces both cycles' records and blob
   hashes exactly, except the labelled wall-time field;
7. gate honesty per cycle — the decisive cycle 2's
   journaled checks 3/4 equal an independent ``ProbeContext``
   evaluation of the same candidate against ``d0001``. The
   verdict is REJECT (cumulative identity drift
   0.3329036422073841 > tol 0.25), reported AS MEASURED. No
   criterion requires an ADMIT;
8. cumulative accounting, diagnostics, and the
   control/treatment table — per-probe drift contributions
   with worst probes named, the alignment diagnostic read
   against its head-relative (SI-23) counterpart, and the
   folded updates' inner product and norms, all asserted
   here and published in the as-built doc;
9. first-cycle reproduction — see criterion 2;
10. VERIFY, unwind, and untouched scope — cycle 1's §6.6
    VERIFY passes as in SI-19; cycle 2 never landed
    (REJECT), so NO authored two-delta stack exists and no
    unwind is claimed (asserted: head stays ``d0001``, one
    delta only); exactly two cycles attempted in the
    decisive run; policy defaults, tolerances, caps, both
    signals' definitions unchanged; ``si squash`` still
    fails closed.

The decisive run's driver shape: ONE ``TinyTransformerMind``
reused through ``si.run.run_cycles`` — one ``run_cycles``
call for cycle 1 (signal 1, the default anchoring), then the
driver explicitly reconfigures the same mind to signal 2
(``set_training_signal``) AND to base-anchored targets
(``set_target_anchoring`` — both recorded selections), then
a second ``run_cycles`` call for cycle 2. The mind never
chooses — neither its signal nor its anchoring.
"""
import copy
import json
import os

import numpy as np
import pytest

import si.tiny_training as tt
import si.tiny_transformer as ttf
from si.bundle import (
    DEFAULT_POLICY,
    blob_path,
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
from si.tiny_seed import build_tiny_weights, seed_tiny_bundle
from si.tiny_transformer import (
    TinyTransformerMind,
    assemble_base_weights,
    assemble_weights,
)
from si.transformer_delta import KIND, parse_blob

GRAD_TOL = 1e-5
GRAD_EPS = 0.01
CLOSED_FORM = 0.00140625

SI19_BLOB = "sha256:2761e8ffc139ce3e107c11da035e8a73c20792945a494f4e493dccf418b6ca9e"
SI19_DRIFT = 0.17189218290150166
SI19_REGRESSION = 0.1657181391492486
SI19_NORM = 0.23555380990005192

S2_AT_BASE_BLOB = "sha256:131cc242728024b4f833bc78ea627825f1e8616972fc2670603673e1814645f4"
S2_AT_BASE_NORM = 0.21317186144376168
S2_AT_BASE_DRIFT = 0.16672966815531254
S2_AT_BASE_REGRESSION = 0.2418281203135848

# SI-23's head-relative cycle 2 — SI-25's CONTROL run
# (docs/SI-23-second-signal.md, published measurements).
CTRL_BLOB = "sha256:184cc7f06cd9cb3bd83fbed466b6a939c7e2e18cf7b8a72be70c51043a6e6cf7"
CTRL_NORM = 0.21317186599953494
CTRL_DRIFT = 0.2963416576385498
CTRL_REGRESSION = 0.15023326850496233
CTRL_FOLDED_INNER = 0.004356018453758104
CTRL_ALIGN_AGG_INNER = 0.07637087369869666

# SI-25 measured values (author's runs; the as-built doc
# publishes them with their labels). The decisive cycle-2
# verdict is REJECT — measured, not required, not tuned for.
BA_CYCLE2_BLOB = "sha256:646d1e8ba3978d858d4173d561f8799bce37350f88ff2de65f40c223d9e4d675"
BA_CYCLE2_NORM = 0.2515524481203623
BA_CYCLE2_DRIFT = 0.3329036422073841
BA_CYCLE2_REGRESSION = 0.10096919978968799
BA_PRE_HELD = 0.002757805049077469
BA_PRE_TRAIN = 0.0027535558814751025
BA_HELD_AFTER = 1.043608588752459e-05
RESIDUAL_PRE_HELD = 3.2118398935406443e-06
RESIDUAL_PRE_TRAIN = 1.824005808470025e-06
RESIDUAL_TRAINED_HELD = 6.234076521114708e-06

KBIAS0 = "model.language_model.layers.0.self_attn.k_proj.bias"


def _seeded(tmp_path, name="tiny25"):
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


def _decisive_run(tmp_path, name="tiny25"):
    """The decisive two-cycle run (see module docstring for
    the driver shape): one reused mind, cycle 1 signal 1
    under the default anchoring via ``run_cycles``, explicit
    reconfiguration to signal 2 AND base-anchored targets,
    cycle 2 via ``run_cycles``."""
    bundle_dir = _seeded(tmp_path, name)
    policy = _policy(bundle_dir)
    mind, seen = _spying_mind(bundle_dir)
    sum1 = run_cycles(bundle_dir, mind, 1, gate=Gate(policy))
    mind.set_training_signal(tt.TRAINING_SIGNAL_ID_2)
    mind.set_target_anchoring(tt.BASE_ANCHORED)
    sum2 = run_cycles(bundle_dir, mind, 1, gate=Gate(policy))
    return bundle_dir, policy, mind, seen, sum1, sum2


def _land_d0001(tmp_path, name="tiny25"):
    """A bundle with signal 1's ``d0001`` landed (one cycle)."""
    bundle_dir = _seeded(tmp_path, name)
    policy = _policy(bundle_dir)
    mind, _seen = _spying_mind(bundle_dir)
    res = run_cycle(bundle_dir, mind, gate=Gate(policy))
    assert res["verdict"] == "ADMIT" and res["head"] == "d0001"
    return bundle_dir, policy


def _grad_worst_base_anchored(weights_under, target_weights, signal_id):
    """Worst analytic-vs-central-finite-difference disagreement
    over ALL 512 factor entries, on an 8-example subset, for
    the BASE-ANCHORED loss: targets derived from
    ``target_weights`` (the base), hidden states and readout
    from ``weights_under`` (the weights under training) — the
    SI-19 gradient-check shape with the target source split
    (§5.8). The finite differences differentiate
    ``factor_loss`` (the float32 shared fold) of the weights
    under training against the base-derived targets."""
    signal = tt.build_training_signal(target_weights, signal_id)
    subset = signal["train"][:8]
    A0, B0 = tt.initial_factors(signal_id)
    lm_head = np.asarray(weights_under[tt.TARGET_NAME], dtype=np.float64)
    hiddens = {tuple(s): tt.final_hidden(weights_under, s)
               for s, _ in subset}
    _loss, grad_A, grad_B = tt.analytic_gradients(
        A0, B0, subset, hiddens, lm_head)
    assert np.max(np.abs(grad_A)) > 0.0 and np.max(np.abs(grad_B)) > 0.0
    worst = 0.0
    for i in range(tt.RANK):
        for j in range(32):
            plus = A0.copy(); plus[i, j] += GRAD_EPS
            minus = A0.copy(); minus[i, j] -= GRAD_EPS
            fd = (tt.factor_loss(weights_under, plus, B0, subset)
                  - tt.factor_loss(weights_under, minus, B0, subset)) \
                / (2 * GRAD_EPS)
            worst = max(worst, abs(fd - grad_A[i, j]))
    for i in range(32):
        for j in range(tt.RANK):
            plus = B0.copy(); plus[i, j] += GRAD_EPS
            minus = B0.copy(); minus[i, j] -= GRAD_EPS
            fd = (tt.factor_loss(weights_under, A0, plus, subset)
                  - tt.factor_loss(weights_under, A0, minus, subset)) \
                / (2 * GRAD_EPS)
            worst = max(worst, abs(fd - grad_B[i, j]))
    return worst


# -- §10.1: recorded anchoring ---------------------------------------------------

def test_anchoring_configuration_and_refusals(tmp_path):
    # Exactly the two named anchorings are served; the module
    # constant survives as the DEFAULT's name only.
    assert tt.TARGET_ANCHORINGS == ("weights-under-training",
                                    "base-anchored")
    assert tt.TARGET_ANCHORING == "weights-under-training"
    assert tt.BASE_ANCHORED == "base-anchored"
    assert tt.validate_target_anchoring("weights-under-training") == \
        "weights-under-training"
    assert tt.validate_target_anchoring("base-anchored") == "base-anchored"
    for bad in ("base", "head-relative", "", "BASE-ANCHORED", None):
        with pytest.raises(ValueError, match="unknown target anchoring"):
            tt.validate_target_anchoring(bad)

    bundle_dir = _seeded(tmp_path)
    # Constructor: default is weights-under-training; the
    # explicit choice is served; any other value refuses.
    mind = TinyTransformerMind(bundle_dir, load_bundle(bundle_dir))
    assert mind.target_anchoring == "weights-under-training"
    assert mind.propose([], None).target_anchoring == \
        "weights-under-training"
    mind_ba = TinyTransformerMind(
        bundle_dir, load_bundle(bundle_dir),
        target_anchoring="base-anchored")
    assert mind_ba.target_anchoring == "base-anchored"
    assert mind_ba.propose([], None).target_anchoring == "base-anchored"
    with pytest.raises(ValueError, match="unknown target anchoring"):
        TinyTransformerMind(bundle_dir, load_bundle(bundle_dir),
                            target_anchoring="residual")
    # Setter: explicit driver reconfiguration, same refusals.
    mind.set_target_anchoring("base-anchored")
    assert mind.target_anchoring == "base-anchored"
    assert mind.propose([], None).target_anchoring == "base-anchored"
    mind.set_target_anchoring("weights-under-training")
    assert mind.propose([], None).target_anchoring == \
        "weights-under-training"
    with pytest.raises(ValueError, match="unknown target anchoring"):
        mind.set_target_anchoring("learned")
    # Dispatch: an intent carrying any other anchoring refuses
    # through train_delta, per the existing discipline.
    base = dict(kind=KIND, reason="SI-25 refusal probe",
                parents=["base"], source_seqs=[1])
    for bad in ("residual-anchored", "", "weights_under_training"):
        with pytest.raises(ValueError, match="unknown target anchoring"):
            mind.train_delta(Intent(
                training_signal_id=tt.TRAINING_SIGNAL_ID,
                target_anchoring=bad, targets=[tt.TARGET_NAME], **base))
    # The anchoring axis is independent of the signal axis:
    # either signal can be configured under either anchoring,
    # and propose carries BOTH recorded selections.
    mind.set_training_signal(tt.TRAINING_SIGNAL_ID_2)
    mind.set_target_anchoring("base-anchored")
    intent = mind.propose([], None)
    assert intent.training_signal_id == tt.TRAINING_SIGNAL_ID_2
    assert intent.target_anchoring == "base-anchored"
    assert intent.to_dict()["target_anchoring"] == "base-anchored"


# -- §10.4: target-source assembly -------------------------------------------------

def test_base_assembly_is_told_apart_and_never_mutates(tmp_path):
    bundle_dir, policy = _land_d0001(tmp_path)
    manifest = load_bundle(bundle_dir)
    before = copy.deepcopy(manifest)
    head_w = assemble_weights(bundle_dir, manifest, policy)
    base_w = assemble_base_weights(bundle_dir, manifest, policy)
    # The verified manifest is never mutated by the base view.
    assert manifest == before
    assert manifest["head"] == "d0001"
    seed_w = build_tiny_weights()
    # The base view IS the frozen seed base, tensor for tensor:
    # its readout equals the recorded base tensor.
    for name, arr in seed_w.items():
        assert np.array_equal(base_w[name], arr)
    assert np.array_equal(base_w[tt.TARGET_NAME], seed_w[tt.TARGET_NAME])
    # The head view's readout equals base + d0001's fold, and
    # differs from the base's readout (the assemblies are told
    # apart exactly where the landed delta acts).
    from si.apply import load_applied_tensor
    assert np.array_equal(
        head_w[tt.TARGET_NAME],
        load_applied_tensor(bundle_dir, manifest, tt.TARGET_NAME, policy))
    assert not np.array_equal(head_w[tt.TARGET_NAME],
                              base_w[tt.TARGET_NAME])
    # Untargeted tensors agree between the two views.
    assert np.array_equal(head_w[KBIAS0], base_w[KBIAS0])
    # At head = base the two assemblies coincide exactly.
    fresh_dir = _seeded(tmp_path, "tiny25base")
    fresh_man = load_bundle(fresh_dir)
    fresh_pol = _policy(fresh_dir)
    for name in seed_w:
        assert np.array_equal(
            assemble_base_weights(fresh_dir, fresh_man, fresh_pol)[name],
            assemble_weights(fresh_dir, fresh_man, fresh_pol)[name])


def test_base_assembly_fails_closed_with_no_fallback(tmp_path, monkeypatch):
    bundle_dir, policy = _land_d0001(tmp_path)
    manifest = load_bundle(bundle_dir)
    # A broken base assembly refuses base-anchored training:
    # train_candidate propagates the failure and returns NO
    # candidate — it never falls back to head-derived targets
    # (a fallback would have produced SI-23's head-relative
    # candidate here).
    def _broken(*a, **k):
        raise RuntimeError("base assembly broken (injected)")

    monkeypatch.setattr(ttf, "assemble_base_weights", _broken)
    mind = TinyTransformerMind(
        bundle_dir, manifest,
        training_signal_id=tt.TRAINING_SIGNAL_ID_2,
        target_anchoring="base-anchored")
    with pytest.raises(RuntimeError, match="base assembly broken"):
        mind.train_delta(mind.propose([], None))
    monkeypatch.undo()
    # The SAME bundle state trains fine head-relative — the
    # refusal above is the base assembly's, not the bundle's.
    mind_hr = TinyTransformerMind(
        bundle_dir, load_bundle(bundle_dir),
        training_signal_id=tt.TRAINING_SIGNAL_ID_2)
    cand = mind_hr.train_delta(mind_hr.propose([], None))
    assert cand["record"]["blob"] == CTRL_BLOB
    assert cand["hyperparams"]["target_anchoring"] == \
        "weights-under-training"
    # A genuinely tampered base (a base tensor blob deleted
    # from a copy of the bundle) refuses the base assembly
    # itself — fail closed, never a fabricated base.
    import shutil
    copy_dir = str(tmp_path / "tampered.si")
    shutil.copytree(bundle_dir, copy_dir)
    tampered_manifest = load_bundle(copy_dir)
    victim = tampered_manifest["base"]["tensor_blobs"][tt.TARGET_NAME]
    os.remove(blob_path(copy_dir, victim["blob"]))
    with pytest.raises(Exception):
        assemble_base_weights(copy_dir, tampered_manifest, policy)
    # An architecture-tampered manifest view refuses too.
    bad = copy.deepcopy(manifest)
    bad["base"]["architecture"] = dict(
        bad["base"]["architecture"], vocab_size=33)
    with pytest.raises(ValueError):
        assemble_base_weights(bundle_dir, bad, policy)


# -- §10.2/§10.9: coincidence at base + first-cycle reproduction -------------------

def test_signal2_base_anchored_at_base_coincides_with_si23(tmp_path):
    bundle_dir = _seeded(tmp_path)
    policy = _policy(bundle_dir)
    before = _blobs_total(bundle_dir)
    mind, seen = _spying_mind(
        bundle_dir, training_signal_id=tt.TRAINING_SIGNAL_ID_2,
        target_anchoring="base-anchored")
    res = run_cycle(bundle_dir, mind, gate=Gate(policy))
    # Reported AS MEASURED: at base the two anchorings
    # coincide, and the run reproduces SI-23's at-base run
    # EXACTLY — same blob, same gate numbers (§10.2).
    assert res["verdict"] == "ADMIT" and res["head"] == "d0001"
    assert res["verified"] is True and res["rolled_back"] is False
    hp = seen[0]["hyperparams"]
    assert hp["target_anchoring"] == "base-anchored"
    assert hp["head"] == "base" and hp["target_source_head"] == "base"
    assert hp["stack_depth"] == 0
    # At base the pre-training record IS the closed form under
    # either anchoring (the two weight sets are one set there).
    assert hp["base_held_out_loss"] == pytest.approx(CLOSED_FORM,
                                                     abs=1e-12)
    assert seen[0]["record"]["blob"] == S2_AT_BASE_BLOB
    assert seen[0]["record"]["norm"] == pytest.approx(S2_AT_BASE_NORM,
                                                      abs=1e-12)
    manifest = load_bundle(bundle_dir)
    assert _blobs_total(bundle_dir) == before + 2360
    checks, _row = _checks_of_edit(bundle_dir, manifest, res["edit_id"])
    assert checks["identity_continuity"]["measured"] == pytest.approx(
        S2_AT_BASE_DRIFT, abs=1e-9)
    assert checks["capability_floor"]["measured"] == pytest.approx(
        S2_AT_BASE_REGRESSION, abs=1e-9)
    note = _hyperparams_notes(bundle_dir, manifest, "d0001")
    assert note["target_anchoring"] == "base-anchored"
    assert note["target_source_head"] == "base"
    ok, report = _verify_probe_head(bundle_dir, manifest, policy=policy)
    assert ok is True
    assert report["identity"]["max_drift"] == pytest.approx(
        S2_AT_BASE_DRIFT, abs=1e-9)
    assert report["capability"]["echo_accuracy"] == 1.0


def test_signal1_first_cycle_reproduces_under_either_anchoring(tmp_path):
    # §10.9 + §10.2's second half: signal 1's first cycle
    # reproduces SI-19/SI-21/SI-23's d0001 EXACTLY under the
    # default anchoring AND under base-anchoring (where the
    # anchorings coincide, the new path is proven
    # behaviour-preserving before the decisive cycle is
    # trusted).
    for anchoring in ("weights-under-training", "base-anchored"):
        bundle_dir = _seeded(tmp_path, f"s1-{anchoring[:4]}")
        policy = _policy(bundle_dir)
        mind, seen = _spying_mind(bundle_dir,
                                  target_anchoring=anchoring)
        res = run_cycle(bundle_dir, mind, gate=Gate(policy))
        assert res["verdict"] == "ADMIT" and res["head"] == "d0001"
        assert seen[0]["record"]["blob"] == SI19_BLOB
        assert seen[0]["record"]["norm"] == pytest.approx(SI19_NORM,
                                                          abs=1e-12)
        hp = seen[0]["hyperparams"]
        assert hp["target_anchoring"] == anchoring
        assert hp["head"] == "base"
        assert hp["target_source_head"] == "base"
        manifest = load_bundle(bundle_dir)
        checks, _row = _checks_of_edit(bundle_dir, manifest,
                                       res["edit_id"])
        assert checks["identity_continuity"]["measured"] == \
            pytest.approx(SI19_DRIFT, abs=1e-12)
        assert checks["capability_floor"]["measured"] == \
            pytest.approx(SI19_REGRESSION, abs=1e-12)


# -- §10.3: gradient checks for the base-anchored loss ------------------------------

def test_gradient_check_signal2_base_anchored_at_base(tmp_path):
    bundle_dir = _seeded(tmp_path)
    policy = _policy(bundle_dir)
    manifest = load_bundle(bundle_dir)
    weights = assemble_weights(bundle_dir, manifest, policy)
    target_weights = assemble_base_weights(bundle_dir, manifest, policy)
    worst = _grad_worst_base_anchored(weights, target_weights,
                                      tt.TRAINING_SIGNAL_ID_2)
    # Measured 5.5249277028738815e-08 under the new code path
    # (as-built doc) — identical to SI-23's at-base figure, as
    # it must be: at base the anchored loss coincides with the
    # head-relative loss. The pin is SI-19's 1e-5, unchanged.
    assert worst == pytest.approx(5.5249277028738815e-08, rel=0.5)
    assert worst <= GRAD_TOL


def test_gradient_check_signal2_base_anchored_at_d0001(tmp_path):
    bundle_dir, policy = _land_d0001(tmp_path)
    manifest = load_bundle(bundle_dir)
    weights = assemble_weights(bundle_dir, manifest, policy)
    target_weights = assemble_base_weights(bundle_dir, manifest, policy)
    worst = _grad_worst_base_anchored(weights, target_weights,
                                      tt.TRAINING_SIGNAL_ID_2)
    # Measured 4.8377706524291005e-08 at head d0001 (as-built
    # doc) — the decisive configuration: base-derived targets,
    # head hidden states, head readout. A check run only at
    # base does not satisfy §10.3; this is the decisive one.
    assert worst == pytest.approx(4.8377706524291005e-08, rel=0.5)
    assert worst <= GRAD_TOL


# -- §10.5: echo-trap records under the new semantics ---------------------------------

def test_residual_record_is_a_premise_failure_and_refusal_stands(
        tmp_path, monkeypatch):
    bundle_dir, policy = _land_d0001(tmp_path)
    manifest = load_bundle(bundle_dir)
    head_w = assemble_weights(bundle_dir, manifest, policy)
    base_w = assemble_base_weights(bundle_dir, manifest, policy)
    # The residual record, measured directly: signal 1's
    # targets derived from the BASE, scored by the HEAD's
    # weights — genuinely new information the head-relative
    # semantics could never produce (there the same record is
    # the closed form by construction, asserted in SI-21/23).
    signal = tt.build_training_signal(base_w, tt.TRAINING_SIGNAL_ID)
    pre_held = tt.signal_loss(head_w, signal["held_out"])
    pre_train = tt.signal_loss(head_w, signal["train"])
    assert pre_held == pytest.approx(RESIDUAL_PRE_HELD, abs=1e-15)
    assert pre_train == pytest.approx(RESIDUAL_PRE_TRAIN, abs=1e-15)
    # The record is ~0 in §5.5's sense: the landed head
    # essentially occupies signal 1's fixed destination on
    # held-out sequences (0.23% of the closed form; exactly
    # cycle 1's own held_out_loss_after, as it must be — same
    # targets, same head).
    assert pre_held < CLOSED_FORM / 100
    assert pre_held == pytest.approx(3.2118398935406443e-06, abs=1e-15)
    # Training against it does NOT improve (measured through
    # the real train path: the factor_loss the refusal
    # consumed is recorded), so the existing non-improvement
    # refusal stands — reported as the slice's finding, with
    # no substitute run as the decisive cycle.
    recorded = []
    orig_factor_loss = tt.factor_loss

    def recording(weights, A, B, examples):
        value = orig_factor_loss(weights, A, B, examples)
        recorded.append(value)
        return value

    monkeypatch.setattr(tt, "factor_loss", recording)
    mind = TinyTransformerMind(
        bundle_dir, load_bundle(bundle_dir),
        training_signal_id=tt.TRAINING_SIGNAL_ID,
        target_anchoring="base-anchored")
    with pytest.raises(ValueError, match="did not improve"):
        mind.train_delta(mind.propose([], None))
    assert recorded == [pytest.approx(RESIDUAL_TRAINED_HELD, abs=1e-15)]
    assert recorded[0] > pre_held
    # Nothing landed and nothing was substituted: the bundle
    # still holds exactly cycle 1's delta.
    manifest = load_bundle(bundle_dir)
    assert manifest["head"] == "d0001"
    assert len(manifest["deltas"]) == 1


# -- the decisive run: §10.1/§10.5/§10.7 -------------------------------------------

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
    # under the default anchoring.
    assert hp1["training_signal_id"] == tt.TRAINING_SIGNAL_ID
    assert hp1["target_anchoring"] == "weights-under-training"
    assert hp1["head"] == "base" and hp1["target_source_head"] == "base"
    assert hp1["stack_depth"] == 0
    assert seen[0]["record"]["blob"] == SI19_BLOB
    # Cycle 2: signal 2 AND base-anchored, both explicitly
    # selected, trained at d0001 under the unchanged freshness
    # contract — the first record where the two head fields
    # disagree (§10.1).
    assert hp2["training_signal_id"] == tt.TRAINING_SIGNAL_ID_2
    assert hp2["target_anchoring"] == "base-anchored"
    assert hp2["head"] == "d0001" and hp2["target_source_head"] == "base"
    assert hp2["stack_depth"] == 1
    assert seen[1]["record"]["parent"] == "d0001"
    assert seen[1]["record"]["id"] == "d0002"
    assert seen[1]["record"]["blob"] == BA_CYCLE2_BLOB
    assert seen[1]["record"]["norm"] == pytest.approx(BA_CYCLE2_NORM,
                                                      abs=1e-12)
    # Echo trap at d0001 (§10.5): the pre-training held-out
    # loss under base-anchoring is a GENUINE measurement —
    # labelled with signal/head/target source/anchoring — and
    # is NOT the closed form (never quoted for a non-base
    # head). It is not ~0, and training improves on it ~264x.
    assert hp2["base_held_out_loss"] == pytest.approx(BA_PRE_HELD,
                                                      abs=1e-12)
    assert hp2["base_train_loss"] == pytest.approx(BA_PRE_TRAIN,
                                                   abs=1e-12)
    assert abs(hp2["base_held_out_loss"] - CLOSED_FORM) > 1e-3
    assert hp2["held_out_loss_after"] == pytest.approx(BA_HELD_AFTER,
                                                       abs=1e-12)
    assert hp2["held_out_loss_after"] < hp2["base_held_out_loss"] / 100
    assert hp2["held_out_examples_closer"] == \
        hp2["held_out_examples"] == 16
    assert hp2["train_loss_initial"] == pytest.approx(
        0.0042998498036725735, abs=1e-12)
    assert hp2["train_loss_final"] == pytest.approx(
        9.412649442227854e-06, abs=1e-12)
    assert hp2["norm_projection_activations"] == 0
    assert hp1["wall_time_seconds"] > 0.0
    assert hp2["wall_time_seconds"] > 0.0

    manifest = load_bundle(bundle_dir)
    edit1, edit2 = sum1["edit_ids"][0], sum2["edit_ids"][0]
    # Gate honesty (§10.7): cycle 2's journaled checks 3/4
    # equal an independent ProbeContext evaluation of the
    # same candidate against the same pre-cycle head (d0001).
    fresh = TinyTransformerMind(
        bundle_dir, load_bundle(bundle_dir),
        training_signal_id=tt.TRAINING_SIGNAL_ID_2,
        target_anchoring="base-anchored")
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
    # Cumulative vs incremental, labelled (§10.8): identity is
    # seed-absolute and FAILS; capability is head-incremental
    # and passes; echo is absolute vs the floor and passes.
    assert journaled["identity_continuity"]["measured"] == \
        pytest.approx(BA_CYCLE2_DRIFT, abs=1e-9)
    assert journaled["identity_continuity"]["passed"] is False
    assert "measured-fail" in journaled["identity_continuity"]["detail"]
    assert journaled["capability_floor"]["measured"] == \
        pytest.approx(BA_CYCLE2_REGRESSION, abs=1e-9)
    assert journaled["capability_floor"]["passed"] is True
    # Recorded anchoring on BOTH verdict paths (§10.1): cycle
    # 1's note (ADMIT, under d0001) records the default;
    # cycle 2's note (REJECT, under its self-edit) records
    # the disagreement head=d0001 / target_source_head=base.
    note1 = _hyperparams_notes(bundle_dir, manifest, "d0001")
    assert note1["target_anchoring"] == "weights-under-training"
    assert note1["head"] == "base" and note1["target_source_head"] == "base"
    note2 = _hyperparams_notes(bundle_dir, manifest, edit2)
    assert note2["target_anchoring"] == "base-anchored"
    assert note2["head"] == "d0001"
    assert note2["target_source_head"] == "base"
    assert note2["training_signal_id"] == tt.TRAINING_SIGNAL_ID_2
    # The landed head's cumulative drift after the REJECT is
    # unchanged — re-measured (§10.8), and cycle 1's VERIFY
    # stands as in SI-19.
    ok, report = _verify_probe_head(bundle_dir, manifest, policy=policy)
    assert ok is True
    assert report["identity"]["max_drift"] == pytest.approx(
        SI19_DRIFT, abs=1e-9)


# -- §10.8: the §5.6 diagnostics + control/treatment --------------------------------

def _folded_update(candidate):
    parsed = parse_blob(candidate["blob_bytes"])
    target = candidate["record"]["targets"][0]
    A = np.asarray(parsed.tensors[target["factor_keys"][0]],
                   dtype=np.float64)
    B = np.asarray(parsed.tensors[target["factor_keys"][1]],
                   dtype=np.float64)
    return (float(target["alpha"]) / float(target["rank"])) * (B @ A)


def test_composition_diagnostics_and_control_treatment(tmp_path):
    bundle_dir, policy, mind, seen, sum1, sum2 = _decisive_run(tmp_path)
    manifest = load_bundle(bundle_dir)
    assert manifest["head"] == "d0001"  # candidate 2 never landed
    ctx = ProbeContext(bundle_dir, manifest, policy=policy)
    cand2 = seen[1]
    # (i) Per-probe contributions for the TREATMENT
    # (base-anchored) candidate (measurements; the incremental
    # movement is a DERIVATION from two measured forwards —
    # composed minus d0001 — labelled as such in the doc).
    expected = {
        "tiny-id-0": (0.17189218290150166, 0.16173836588859558, 0.3329036422073841),
        "tiny-id-1": (0.15088725090026855, 0.06053225928917527, 0.09286642074584961),
        "tiny-id-2": (0.11858986946754158, 0.14893579483032227, 0.14936351776123047),
        "tiny-id-3": (0.14782869815826416, 0.1503568198531866, 0.1489148810505867),
        "tiny-id-4": (0.11720322235487401, 0.12207234278321266, 0.14728415571153164),
        "tiny-id-5": (0.09891846403479576, 0.12109210342168808, 0.11245626583695412),
        "tiny-id-6": (0.12363102938979864, 0.16010307148098946, 0.14491976890712976),
        "tiny-id-7": (0.08795571327209473, 0.11142894998192787, 0.14189285784959793),
    }
    # SI-23 CONTROL composed drift per probe (published,
    # docs/SI-23-second-signal.md) — kept distinct, never
    # blended with the treatment's figures (§12.16).
    control_composed = {
        "tiny-id-0": 0.2092923242598772,
        "tiny-id-1": 0.15134692192077637,
        "tiny-id-2": 0.14906740188598633,
        "tiny-id-3": 0.2963416576385498,
        "tiny-id-4": 0.23394920490682125,
        "tiny-id-5": 0.14004023000597954,
        "tiny-id-6": 0.1332004750147462,
        "tiny-id-7": 0.2018739078193903,
    }
    m1_all, m2_all = [], []
    rows = {}
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
    for qid, (d1, inc, comp) in expected.items():
        got = rows[qid]
        assert got[0] == pytest.approx(d1, abs=1e-9)
        assert got[1] == pytest.approx(inc, abs=1e-9)
        assert got[2] == pytest.approx(comp, abs=1e-9)
    # Worst probes, named: d0001's own drift, the incremental
    # movement, AND the composed drift are all tiny-id-0 for
    # the treatment — a DIFFERENT worst probe than the
    # control's (tiny-id-3). The composed worst equals the
    # gate's measured cycle-2 drift.
    assert max(rows, key=lambda q: rows[q][0]) == "tiny-id-0"
    assert max(rows, key=lambda q: rows[q][1]) == "tiny-id-0"
    assert max(rows, key=lambda q: rows[q][2]) == "tiny-id-0"
    assert rows["tiny-id-0"][2] == pytest.approx(BA_CYCLE2_DRIFT,
                                                 abs=1e-9)
    # Control/treatment, per probe: the treatment's composed
    # drift IMPROVES on the control at tiny-id-3 (the
    # control's worst: 0.2963 -> 0.1489) and at 4 further
    # probes, and WORSENS at tiny-id-0 (0.2093 -> 0.3329,
    # substantially), tiny-id-6 (0.1332 -> 0.1449), and
    # tiny-id-2 (0.14907 -> 0.14936, marginally) — the maximum
    # worsens even though most probes improve. Both
    # directions are asserted; neither run's figures are
    # blended.
    improved = [q for q in rows
                if rows[q][2] < control_composed[q] - 1e-9]
    worsened = [q for q in rows
                if rows[q][2] > control_composed[q] + 1e-9]
    assert sorted(improved) == ["tiny-id-1", "tiny-id-3",
                                "tiny-id-4", "tiny-id-5", "tiny-id-7"]
    assert sorted(worsened) == ["tiny-id-0", "tiny-id-2", "tiny-id-6"]
    # (ii) Alignment diagnostic (derivation), read against its
    # head-relative counterpart (§5.6): IN AGGREGATE the
    # base-anchored movement OPPOSES the first update's
    # (negative inner product, minority sign agreement) where
    # the control's ALIGNED (positive) — the designed
    # mechanism appears in aggregate. At the treatment's worst
    # probe (tiny-id-0) the movements still ALIGN (positive
    # inner product) — the mechanism fails exactly where the
    # gate binds. At the control's worst probe (tiny-id-3)
    # the treatment's movements oppose.
    ids = [q for q, _x, _e in ctx.identity_probes()]
    i0 = ids.index("tiny-id-0")
    i3 = ids.index("tiny-id-3")
    inner_worst = float(np.dot(m1_all[i0], m2_all[i0]))
    sign_worst = float(np.mean(np.sign(m1_all[i0])
                               == np.sign(m2_all[i0])))
    assert inner_worst == pytest.approx(0.0338873784299374, abs=1e-9)
    assert sign_worst == pytest.approx(0.5, abs=1e-12)
    inner_id3 = float(np.dot(m1_all[i3], m2_all[i3]))
    assert inner_id3 == pytest.approx(-0.02224599986114801, abs=1e-9)
    a_all = np.concatenate(m1_all); b_all = np.concatenate(m2_all)
    agg_inner = float(np.dot(a_all, b_all))
    assert agg_inner == pytest.approx(-0.21256230517288188, abs=1e-9)
    assert agg_inner < 0.0 < CTRL_ALIGN_AGG_INNER
    assert float(np.mean(np.sign(a_all) == np.sign(b_all))) == \
        pytest.approx(0.37109375, abs=1e-12)
    assert float(np.linalg.norm(a_all)) == pytest.approx(
        0.7119678353441855, abs=1e-9)
    assert float(np.linalg.norm(b_all)) == pytest.approx(
        0.7816722548913912, abs=1e-9)
    # (iii) Folded updates (measurements): the treatment's
    # folded inner product with d0001's update is NEGATIVE
    # (opposing in weight space) where the control's was
    # positive; the treatment's update is LARGER (it corrects
    # the head's existing lean as well as adding its own).
    d1u, d2u = _folded_update(seen[0]), _folded_update(cand2)
    folded_inner = float(np.sum(d1u * d2u))
    assert folded_inner == pytest.approx(-0.011457914936582613,
                                         abs=1e-9)
    assert folded_inner < 0.0 < CTRL_FOLDED_INNER
    assert float(np.linalg.norm(d1u)) == pytest.approx(SI19_NORM,
                                                       abs=1e-9)
    assert float(np.linalg.norm(d2u)) == pytest.approx(BA_CYCLE2_NORM,
                                                       abs=1e-9)
    assert BA_CYCLE2_NORM > CTRL_NORM
    # (iv) Composed capability (measurements via the gate's
    # own evaluation): head-incremental regression passes and
    # IMPROVES on the control's; echo 1.0 vs floor 0.9375.
    rep = Gate(policy).evaluate(mind.propose([], ctx), cand2, ctx)
    checks = {c.name: c for c in rep.checks}
    assert checks["capability_floor"].measured == pytest.approx(
        BA_CYCLE2_REGRESSION, abs=1e-9)
    assert BA_CYCLE2_REGRESSION < CTRL_REGRESSION
    targets = ctx.capability_targets()
    correct = sum(
        1 for cid, x, _e in ctx.capability_suite()
        if int(np.argmax(np.asarray(ctx.forward_candidate(x, cand2))))
        == int(targets[cid]))
    assert correct == 16 and ctx.echo_floor() == 0.9375
    # Headroom arithmetic (derivation, labelled): the
    # treatment's incremental movement exceeds cycle 1's
    # remaining headroom at 7 of 8 probes (control: 5 of 8).
    headroom = 0.25 - SI19_DRIFT
    assert sum(1 for v in rows.values() if v[1] > headroom) == 7


# -- §10.6: determinism per cycle ------------------------------------------------------

def test_decisive_run_repeats_exactly(tmp_path):
    _b1, _p1, _m1, seen1, s1a, s2a = _decisive_run(tmp_path / "a",
                                                  "tiny25")
    _b2, _p2, _m2, seen2, s1b, s2b = _decisive_run(tmp_path / "b",
                                                  "tiny25")
    assert s1a == s1b and s2a == s2b
    assert load_bundle(_b1)["deltas"] == load_bundle(_b2)["deltas"]
    for cand_a, cand_b in zip(seen1, seen2):
        assert cand_a["blob_bytes"] == cand_b["blob_bytes"]
        assert cand_a["record"] == cand_b["record"]
        hp_a = dict(cand_a["hyperparams"])
        hp_b = dict(cand_b["hyperparams"])
        hp_a.pop("wall_time_seconds"); hp_b.pop("wall_time_seconds")
        assert hp_a == hp_b
    man1, man2 = load_bundle(_b1), load_bundle(_b2)
    for edit_a, edit_b in zip(s1a["edit_ids"] + s2a["edit_ids"],
                              s1b["edit_ids"] + s2b["edit_ids"]):
        checks_a, _ = _checks_of_edit(_b1, man1, edit_a)
        checks_b, _ = _checks_of_edit(_b2, man2, edit_b)
        for name in ("identity_continuity", "capability_floor"):
            assert checks_a[name]["measured"] == \
                checks_b[name]["measured"]


def test_base_anchored_train_delta_deterministic(tmp_path):
    bundle_dir, policy = _land_d0001(tmp_path)
    mind = TinyTransformerMind(
        bundle_dir, load_bundle(bundle_dir),
        training_signal_id=tt.TRAINING_SIGNAL_ID_2,
        target_anchoring="base-anchored")
    intent = mind.propose([], None)
    first = mind.train_delta(intent)
    second = mind.train_delta(intent)
    assert first["blob_bytes"] == second["blob_bytes"]
    assert first["record"] == second["record"]
    assert first["record"]["blob"] == BA_CYCLE2_BLOB
    hp_a = dict(first["hyperparams"]); hp_b = dict(second["hyperparams"])
    hp_a.pop("wall_time_seconds"); hp_b.pop("wall_time_seconds")
    assert hp_a == hp_b


# -- §10.10: VERIFY / unwind honesty / untouched scope -----------------------------------

def test_unwind_honesty_and_untouched_scope(tmp_path):
    bundle_dir = _seeded(tmp_path)
    policy = _policy(bundle_dir)
    before = _blobs_total(bundle_dir)
    mind, seen = _spying_mind(bundle_dir)
    sum1 = run_cycles(bundle_dir, mind, 1, gate=Gate(policy))
    mind.set_training_signal(tt.TRAINING_SIGNAL_ID_2)
    mind.set_target_anchoring(tt.BASE_ANCHORED)
    sum2 = run_cycles(bundle_dir, mind, 1, gate=Gate(policy))
    assert sum1["admitted"] == 1 and sum2["rejected"] == 1
    manifest = load_bundle(bundle_dir)
    # No authored two-delta stack exists: exactly one delta
    # landed, cycle 2's candidate was never written (+0 B),
    # and NO unwind is claimed or exercised on an authored
    # stack — SI-21's fixture-proven unwind remains the only
    # unwind evidence (stated in the as-built doc).
    assert manifest["head"] == "d0001"
    assert len(manifest["deltas"]) == 1
    assert _blobs_total(bundle_dir) == before + 2360
    cand2_hex = seen[1]["record"]["blob"].split(":", 1)[1]
    assert not os.path.isfile(os.path.join(
        bundle_dir, "blobs", "sha256", cand2_hex + ".safetensors"))
    # Policy defaults and tolerances untouched (§11).
    assert DEFAULT_POLICY["max_stack_depth"] == 8
    assert DEFAULT_POLICY["identity_tol"] == 0.25
    assert DEFAULT_POLICY["capability_eps"] == 0.25
    assert DEFAULT_POLICY["max_targets_per_delta"] == 1
    assert DEFAULT_POLICY["allow_additive"] is False
    assert DEFAULT_POLICY["allowed_target_patterns"] == []
    on_disk = load_policy(bundle_dir)
    assert on_disk["identity_tol"] == 0.25
    assert on_disk["capability_eps"] == 0.25
    # The default's name is unchanged; both signals'
    # definitions are byte-for-byte SI-19/SI-23's (§11: no
    # redefinition, no third signal).
    assert tt.TARGET_ANCHORING == "weights-under-training"
    assert set(tt.SIGNAL_REGISTRY) == {tt.TRAINING_SIGNAL_ID,
                                       tt.TRAINING_SIGNAL_ID_2}
    assert tt.SIGNAL_SEED_OFFSET == 11 and tt.INIT_SEED_OFFSET == 17
    assert tt.SIGNAL2_SEED_OFFSET == 23 and tt.INIT2_SEED_OFFSET == 29
    assert tt.CORRECTION_SCALE == 0.15
    assert tt.N_TRAIN == 32 and tt.N_HELD_OUT == 16
    # Squash still fails closed on the transformer stack
    # (SI-12).
    with pytest.raises(SquashError):
        squash_bundle(bundle_dir, keep=0)
