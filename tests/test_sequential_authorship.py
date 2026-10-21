"""SI-21: sequential authored self-writes for the miniature
(RFC SI-20 §9 outline, §10 acceptance criteria).

Covers the §10 criteria, in order:

1. freshness proven — the SAME mind object reused across two
   cycles (the ``run_cycles`` driver shape that silently
   retrained from a stale construction-time snapshot before
   SI-21) trains cycle 2 at head = ``d0001``; a control trained
   through the stale snapshot directly reproduces cycle 1's
   byte-identical candidate, so the test can tell stale from
   fresh apart;
2. semantics recorded — cycle hyperparams journal ``head``,
   ``target_source_head``, ``target_anchoring`` and
   ``stack_depth`` (in the returned hyperparams AND in the
   journal's ``training-hyperparams`` provenance, on both the
   ADMIT and REJECT paths); the signal definition is SI-19's;
3. determinism per cycle — a full two-cycle run repeated from
   the seed reproduces both cycles' candidates, the landed
   record, and the journaled gate measurements exactly (except
   the labelled wall-time field);
4. gate honesty per cycle — cycle 2's journaled checks 3/4
   equal an independent ``ProbeContext`` evaluation of the
   same candidate against the same pre-cycle head. Cycle 2's
   verdict is reported AS MEASURED: it is REJECT (identity
   drift over tolerance). No criterion requires ADMIT;
5. cumulative accounting published — identity drift is
   seed-absolute (cumulative), capability regression is
   head-incremental; the §4.4 identical-stack derivation
   (predicted 0.3438) is confirmed by measurement, side by
   side, and the genuinely retrained cycle-2 candidate's
   cumulative drift is measured and published in the as-built
   doc;
6. VERIFY and unwind — cycle 1's §6.6 VERIFY is journaled as
   in SI-19; cycle 2 never materialises (REJECT), so there is
   no cycle-2 VERIFY — instead the landed stack is unwound
   through the existing rollback core: a two-delta stack
   (authored ``d0001`` + a benign FIXTURE ``d0002``, labelled
   a fixture — it is not authored) rolls back to ``d0001``
   and then to base, each restored head's probe state
   re-measured, never assumed;
7. held-out honesty in cycle 2 — cycle 2's held-out numbers
   are asserted against cycle 2's own recorded pre-training
   loss at its head, with the target source labelled; the
   recurring 0.00140625 is asserted to recur BY CONSTRUCTION
   (targets re-derived from the weights under training);
8. depth and policy untouched — exactly two authored cycles;
   ``max_stack_depth``, both tolerances and every policy
   default unchanged; ``si squash`` still fails closed;
9. no regressions — cycle 1, run as the first cycle of the
   two-cycle run, reproduces SI-19's published candidate and
   gate numbers exactly;
10. CPU budget — per-cycle ``train_delta`` wall time is
    recorded in the hyperparams of both cycles (measured
    values are published in the as-built doc).
"""
import json
import os

import numpy as np
import pytest

import si.tiny_training as tt
from si.apply import load_applied_tensor
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
from si.tiny_seed import seed_tiny_bundle
from si.tiny_transformer import TinyTransformerMind, assemble_weights
from si.transformer_delta import KIND, construct_transformer_delta

# SI-19's published measurements (docs/SI-19-delta-authorship.md):
# the reference every SI-21 cycle-1 assertion is checked against.
SI19_DRIFT = 0.17189218290150166
SI19_REGRESSION = 0.1657181391492486
SI19_NORM = 0.23555380990005192
SI19_BLOB_BYTES = 2360
# RFC SI-20 §4.4's derivation, labelled a derivation there:
# stacking cycle 1's update identically doubles the drift.
DERIVED_STACK_DRIFT = 2 * SI19_DRIFT  # 0.3438 (rounded in the RFC)

QPROJ = "model.language_model.layers.0.self_attn.q_proj.weight"


def _seeded(tmp_path, name="tiny21"):
    bundle_dir = init_bundle(str(tmp_path / (name + ".si")), name=name,
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    return bundle_dir


def _policy(bundle_dir, **over):
    policy = load_policy(bundle_dir)
    policy["allowed_target_patterns"] = ["*"]
    policy.update(over)
    return policy


def _spying_mind(bundle_dir):
    """A real TinyTransformerMind whose train_delta calls are
    recorded (the candidates it returns, in order)."""
    mind = TinyTransformerMind(bundle_dir, load_bundle(bundle_dir))
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


def _two_cycle_run(tmp_path, name="tiny21"):
    """One full SI-21 run: seed, then run_cycles(2) with ONE
    mind object reused across both cycles (Driver 1's shape)."""
    bundle_dir = _seeded(tmp_path, name)
    policy = _policy(bundle_dir)
    mind, seen = _spying_mind(bundle_dir)
    summary = run_cycles(bundle_dir, mind, 2, gate=Gate(policy))
    return bundle_dir, policy, mind, seen, summary


# -- §10.1/§10.9: freshness, and cycle 1 reproducing SI-19 ---------------------

def test_same_mind_object_trains_cycle2_at_the_landed_head(tmp_path):
    bundle_dir, policy, mind, seen, summary = _two_cycle_run(tmp_path)
    # The run's shape, as measured: cycle 1 ADMITs, cycle 2 is
    # REJECTed at the gate and run_cycles halts; head stays d0001.
    assert summary["cycles_completed"] == 2
    assert summary["admitted"] == 1 and summary["rejected"] == 1
    assert summary["halted"] is True
    assert summary["halt_verdict"] == "REJECT"
    assert summary["head"] == "d0001"
    assert summary["delta_stack_depth"] == 1
    assert len(seen) == 2

    hp1, hp2 = seen[0]["hyperparams"], seen[1]["hyperparams"]
    # Cycle 1 trained at the base, exactly as SI-19 published.
    assert hp1["head"] == "base" and hp1["target_source_head"] == "base"
    assert hp1["stack_depth"] == 0
    assert seen[0]["record"]["norm"] == pytest.approx(SI19_NORM, abs=1e-12)
    manifest = load_bundle(bundle_dir)
    assert manifest["deltas"][0]["blob"] == seen[0]["record"]["blob"]
    assert manifest["deltas"][0]["bytes"] == SI19_BLOB_BYTES

    # Cycle 2 trained against the LANDED head — the freshness
    # contract. Under the pre-SI-21 stale snapshot this candidate
    # was byte-identical to cycle 1's; it is not.
    assert hp2["head"] == "d0001"
    assert hp2["target_source_head"] == "d0001"
    assert hp2["stack_depth"] == 1
    assert seen[1]["blob_bytes"] != seen[0]["blob_bytes"]
    assert seen[1]["record"]["parent"] == "d0001"
    assert seen[1]["record"]["id"] == "d0002"


def test_train_delta_refreshes_a_stale_snapshot_directly(tmp_path):
    bundle_dir = _seeded(tmp_path)
    policy = _policy(bundle_dir)
    base_manifest = load_bundle(bundle_dir)
    # A mind constructed at the base, BEFORE anything lands:
    # its construction-time snapshot says head=base forever.
    stale_mind = TinyTransformerMind(bundle_dir, base_manifest)
    res = run_cycle(bundle_dir, TinyTransformerMind(
        bundle_dir, load_bundle(bundle_dir)), gate=Gate(policy))
    assert res["verdict"] == "ADMIT" and res["head"] == "d0001"
    manifest = load_bundle(bundle_dir)

    # The stale mind's train_delta refreshes: it trains at d0001.
    cand = stale_mind.train_delta(stale_mind.propose([], None))
    assert cand["hyperparams"]["head"] == "d0001"
    assert cand["hyperparams"]["stack_depth"] == 1
    assert cand["record"]["blob"] != manifest["deltas"][0]["blob"]
    # And its snapshot now agrees with the bundle's head.
    assert stale_mind._manifest["head"] == "d0001"

    # The training in cycle 2 assembled weights AT d0001: the
    # assembled readout equals apply-at-load's tensor at that
    # head (base + d0001 folded), and differs from the base's.
    assembled = assemble_weights(bundle_dir, manifest, policy)
    applied = load_applied_tensor(bundle_dir, manifest,
                                   tt.TARGET_NAME, policy)
    assert np.array_equal(assembled[tt.TARGET_NAME], applied)

    # Control — the stale path itself, exercised deliberately:
    # train_candidate handed the BASE manifest (Driver 1's
    # accident, reproduced on purpose) re-derives SI-19's
    # targets and produces cycle 1's byte-identical candidate.
    # This is what tells stale from fresh apart.
    intent = stale_mind.propose([], None)
    stale_cand = tt.train_candidate(bundle_dir, base_manifest, intent)
    assert stale_cand["blob_bytes"] != cand["blob_bytes"]
    assert stale_cand["record"]["blob"] == manifest["deltas"][0]["blob"]
    assert stale_cand["hyperparams"]["head"] == "base"


# -- §10.5: the §4.4 identical-stack derivation, measured -----------------------

def test_identical_stack_derivation_is_confirmed_by_measurement(tmp_path):
    bundle_dir = _seeded(tmp_path)
    policy = _policy(bundle_dir)
    base_manifest = load_bundle(bundle_dir)
    mind, _seen = _spying_mind(bundle_dir)
    res = run_cycle(bundle_dir, mind, gate=Gate(policy))
    assert res["verdict"] == "ADMIT"
    manifest = load_bundle(bundle_dir)

    # The identically stacked candidate: cycle 1's candidate
    # itself (trained from the base manifest — the derivation's
    # premise), evaluated against the landed head d0001, so the
    # gate composes base + d0001 + the same update again.
    intent = mind.propose([], None)
    stacked = tt.train_candidate(bundle_dir, base_manifest, intent)
    assert stacked["record"]["blob"] == manifest["deltas"][0]["blob"]
    ctx = ProbeContext(bundle_dir, manifest, policy=policy)
    intent2 = mind.propose([], ctx)
    report = Gate(policy).evaluate(intent2, stacked, ctx)
    checks = {c.name: c for c in report.checks}
    measured = checks["identity_continuity"].measured
    # Published side by side (as-built doc §"Cumulative
    # accounting"): predicted 0.3438 (derivation) vs measured.
    assert report.verdict == "REJECT"
    assert measured > 0.25
    assert abs(measured - DERIVED_STACK_DRIFT) < 1e-6
    assert "measured-fail" in checks["identity_continuity"].detail
    # Identity is the binding constraint: the head-incremental
    # capability arm still passes for the same candidate.
    assert checks["capability_floor"].passed is True


# -- §10.2/§10.4: semantics recorded; gate honesty per cycle --------------------

def test_cycle2_journal_equals_independent_evaluation(tmp_path):
    bundle_dir = _seeded(tmp_path)
    policy = _policy(bundle_dir)
    mind, seen = _spying_mind(bundle_dir)
    summary = run_cycles(bundle_dir, mind, 2, gate=Gate(policy))
    assert summary["rejected"] == 1
    manifest = load_bundle(bundle_dir)
    edit1, edit2 = summary["edit_ids"]

    # Independent evaluation of cycle 2's (deterministic)
    # candidate against the same pre-cycle head, d0001.
    fresh = TinyTransformerMind(bundle_dir, load_bundle(bundle_dir))
    cand2 = fresh.train_delta(fresh.propose([], None))
    assert cand2["blob_bytes"] == seen[1]["blob_bytes"]
    ctx = ProbeContext(bundle_dir, manifest, policy=policy)
    independent = Gate(policy).evaluate(
        fresh.propose([], ctx), cand2, ctx)
    assert independent.verdict == "REJECT"
    ind = {c.name: c for c in independent.checks}

    journaled, edit_row = _checks_of_edit(bundle_dir, manifest, edit2)
    assert edit_row["result"] == "rejected"
    for name in ("identity_continuity", "capability_floor"):
        assert journaled[name]["measured"] is not None
        assert journaled[name]["measured"] == pytest.approx(
            ind[name].measured, abs=1e-12)
    assert journaled["identity_continuity"]["passed"] is False
    assert "measured-fail" in journaled["identity_continuity"]["detail"]
    assert journaled["capability_floor"]["passed"] is True

    # §10.2: hyperparams are journaled per cycle — cycle 1 under
    # the landed delta, cycle 2 under its (rejected) self-edit.
    hp1 = _hyperparams_notes(bundle_dir, manifest, "d0001")
    assert hp1["head"] == "base" and hp1["stack_depth"] == 0
    hp2 = _hyperparams_notes(bundle_dir, manifest, edit2)
    assert hp2["head"] == "d0001"
    assert hp2["target_source_head"] == "d0001"
    assert hp2["target_anchoring"] == tt.TARGET_ANCHORING
    assert hp2["stack_depth"] == 1
    assert hp2["training_signal_id"] == tt.TRAINING_SIGNAL_ID


def test_signal_definition_is_byte_for_byte_si19s():
    # §10.2: same id, stream, correction, partitions (the SI-19
    # tests pin the construction itself; this pins the fields
    # SI-21's semantics record stands on).
    assert tt.TRAINING_SIGNAL_ID == "tiny-successor-lean-001"
    assert tt.SIGNAL_SEED_OFFSET == 11
    assert tt.CORRECTION_SCALE == 0.15
    assert tt.N_TRAIN == 32 and tt.N_HELD_OUT == 16
    assert tt.TARGET_ANCHORING == "weights-under-training"


# -- §10.3: determinism per cycle -------------------------------------------------

def test_two_cycle_run_repeats_exactly(tmp_path):
    _bd1, _p1, _m1, seen1, sum1 = _two_cycle_run(tmp_path / "a", "tiny21")
    _bd2, _p2, _m2, seen2, sum2 = _two_cycle_run(tmp_path / "b", "tiny21")
    assert sum1 == sum2
    man1 = load_bundle(_bd1)
    man2 = load_bundle(_bd2)
    assert man1["deltas"] == man2["deltas"]
    for cand_a, cand_b in zip(seen1, seen2):
        assert cand_a["blob_bytes"] == cand_b["blob_bytes"]
        assert cand_a["record"] == cand_b["record"]
        hp_a = dict(cand_a["hyperparams"])
        hp_b = dict(cand_b["hyperparams"])
        hp_a.pop("wall_time_seconds")
        hp_b.pop("wall_time_seconds")
        assert hp_a == hp_b
    # Journaled gate measurements repeat exactly too.
    for edit_a, edit_b in zip(sum1["edit_ids"], sum2["edit_ids"]):
        checks_a, _ = _checks_of_edit(_bd1, man1, edit_a)
        checks_b, _ = _checks_of_edit(_bd2, man2, edit_b)
        for name in ("identity_continuity", "capability_floor"):
            assert checks_a[name]["measured"] == \
                checks_b[name]["measured"]


# -- §10.5/§10.7: cumulative vs incremental; held-out honesty ---------------------

def test_cumulative_vs_incremental_accounting(tmp_path):
    bundle_dir, policy, mind, seen, summary = _two_cycle_run(tmp_path)
    manifest = load_bundle(bundle_dir)

    # Cycle 1's §6.6 VERIFY (journaled): identity drift of the
    # landed head is SEED-ABSOLUTE — the cumulative drift of
    # d0001 — and equals SI-19's published gate drift.
    prov = _journal_rows(bundle_dir, manifest, "provenance", child="d0001")
    verify = [p for p in prov
              if p["relation"] == "transformer-probe-verify"]
    assert len(verify) == 1
    vrep = json.loads(verify[0]["note"])
    assert vrep["passed"] is True
    assert vrep["identity"]["max_drift"] == pytest.approx(
        SI19_DRIFT, abs=1e-9)
    assert vrep["baseline_source"] == "pre-cycle head (caller-captured)"

    # Cycle 2's gate numbers, split as §4.4 describes: identity
    # is the WOULD-BE cumulative drift of head d0002 (seed-
    # absolute — it fails), capability regression is charged
    # only since d0001 (head-incremental — it passes).
    _checks, edit2_row = _checks_of_edit(
        bundle_dir, manifest, summary["edit_ids"][1])
    checks = {c["name"]: c
              for c in edit2_row["gate_report"]["checks"]}
    assert checks["identity_continuity"]["measured"] == pytest.approx(
        0.34378434903919697, abs=1e-9)
    assert checks["identity_continuity"]["passed"] is False
    assert checks["capability_floor"]["measured"] == pytest.approx(
        0.16571810841560364, abs=1e-9)
    assert checks["capability_floor"]["passed"] is True

    # After the REJECT the landed head is still d0001 and its
    # cumulative drift is unchanged — re-measured, not assumed.
    ok, report = _verify_probe_head(bundle_dir, manifest, policy=policy)
    assert ok is True
    assert report["identity"]["max_drift"] == pytest.approx(
        SI19_DRIFT, abs=1e-9)

    # §10.7 held-out honesty: cycle 2 improves on ITS OWN
    # pre-training loss at ITS head, and the recurring
    # 0.00140625 is the construction artifact the RFC predicts
    # (targets re-derived from the weights under training),
    # asserted as such — not fresh evidence about the base.
    hp2 = seen[1]["hyperparams"]
    assert hp2["base_held_out_loss"] == pytest.approx(0.00140625, abs=1e-12)
    assert hp2["held_out_loss_after"] < hp2["base_held_out_loss"] / 10
    assert hp2["held_out_examples_closer"] == hp2["held_out_examples"] == 16


# -- §10.6: unwind ------------------------------------------------------------------

class _FixtureMind:
    """One scripted fixture proposal (mirrors SI-17's script
    mind): authors nothing — the candidate is a hand-built
    benign fixture, used ONLY to give the rollback core a
    two-delta stack whose first delta is authored."""

    def __init__(self, intent, candidate):
        self._intent = intent
        self._candidate = candidate

    @property
    def name(self):
        return "si21-fixture"

    def perceive(self, journal):
        return []

    def propose(self, perceptions, ctx):
        return self._intent

    def train_delta(self, intent):
        return self._candidate


def _benign_fixture(manifest, seed=1):
    rng = np.random.default_rng(seed)
    spec = {"name": QPROJ, "kind": "lora",
            "A": (rng.standard_normal((2, 32)) * 0.05).astype(np.float32),
            "B": (rng.standard_normal((32, 2)) * 0.05).astype(np.float32),
            "alpha": 4}
    record, blob = construct_transformer_delta(
        manifest, [spec], delta_id="d0002", parent=manifest["head"],
        created_at="2026-10-21T00:00:00Z", self_edit_id="se_d0002")
    intent = Intent(kind=KIND, reason="SI-21 unwind fixture (not authored)",
                    parents=[manifest["head"]], source_seqs=[1],
                    targets=[QPROJ])
    return intent, {"record": record, "blob_bytes": blob}


def test_unwind_authored_stack_step_by_step(tmp_path):
    bundle_dir = _seeded(tmp_path)
    policy = _policy(bundle_dir)
    mind, _seen = _spying_mind(bundle_dir)
    res1 = run_cycle(bundle_dir, mind, gate=Gate(policy))
    assert res1["verdict"] == "ADMIT" and res1["head"] == "d0001"

    # A second delta lands on top (benign fixture, NOT authored
    # — the authored cycle 2 REJECTed, so no authored two-delta
    # stack exists to unwind; the granularity is what this
    # proves, and the doc labels it exactly this way).
    manifest = load_bundle(bundle_dir)
    intent, cand = _benign_fixture(manifest)
    res2 = run_cycle(bundle_dir, _FixtureMind(intent, cand),
                     gate=Gate(policy))
    assert res2["verdict"] == "ADMIT" and res2["head"] == "d0002"
    manifest = load_bundle(bundle_dir)
    ok, report = _verify_probe_head(bundle_dir, manifest, policy=policy)
    assert ok is True
    # Measured, and recorded honestly: the two-delta head's max
    # identity drift EQUALS the authored head's — the max is set
    # by tiny-id-0, where the fixture's contribution does not
    # exceed the authored delta's (the composed stack is seen
    # by the QPROJ applied tensor differing, asserted via the
    # rollback states below, not by inflating this number).
    assert report["identity"]["max_drift"] == pytest.approx(
        SI19_DRIFT, abs=1e-9)
    applied_stacked = load_applied_tensor(bundle_dir, manifest,
                                          QPROJ, policy)

    # Unwind one step: back to the authored head. Its probe
    # state is re-measured — exactly SI-19's cumulative drift.
    out = rollback(bundle_dir, to="d0001", reason="SI-21 unwind proof")
    assert out["head"] == "d0001"
    manifest = load_bundle(bundle_dir)
    assert manifest["head"] == "d0001"
    ok, report = _verify_probe_head(bundle_dir, manifest, policy=policy)
    assert ok is True
    assert report["identity"]["max_drift"] == pytest.approx(
        SI19_DRIFT, abs=1e-9)
    assert report["capability"]["echo_accuracy"] == 1.0
    # The fixture's fold is gone from the applied tensor.
    assert not np.array_equal(
        load_applied_tensor(bundle_dir, manifest, QPROJ, policy),
        applied_stacked)
    deltas = {d["id"]: d for d in _journal_rows(
        bundle_dir, manifest, "deltas")}
    assert deltas["d0002"]["status"] == "rolled_back"

    # Unwind to base: the seed state, re-measured (drift 0.0).
    out = rollback(bundle_dir, to="base", reason="SI-21 unwind proof")
    assert out["head"] == "base"
    manifest = load_bundle(bundle_dir)
    ok, report = _verify_probe_head(bundle_dir, manifest, policy=policy)
    assert ok is True
    assert report["identity"]["max_drift"] == 0.0
    assert report["capability"]["max_regression"] == 0.0
    deltas = {d["id"]: d for d in _journal_rows(
        bundle_dir, manifest, "deltas")}
    assert deltas["d0001"]["status"] == "rolled_back"


# -- §10.8: depth, policy, squash, stored bytes --------------------------------------

def test_depth_policy_squash_and_stored_bytes(tmp_path):
    bundle_dir = _seeded(tmp_path)
    policy = _policy(bundle_dir)
    total_before = _blobs_total(bundle_dir)
    mind, seen = _spying_mind(bundle_dir)
    summary = run_cycles(bundle_dir, mind, 2, gate=Gate(policy))
    assert summary["delta_stack_depth"] == 1  # exactly 2 attempted

    # Stored bytes, per cycle (RFC §4.5): cycle 1 grew the store
    # by exactly its delta's bytes; the REJECTed cycle 2 wrote
    # zero new bytes — its (genuinely new) candidate blob is not
    # in the store.
    manifest = load_bundle(bundle_dir)
    assert _blobs_total(bundle_dir) == \
        total_before + manifest["deltas"][0]["bytes"]
    cand2_hex = seen[1]["record"]["blob"].split(":", 1)[1]
    assert not os.path.isfile(os.path.join(
        bundle_dir, "blobs", "sha256", cand2_hex + ".safetensors"))
    assert len(seen[1]["blob_bytes"]) == SI19_BLOB_BYTES

    # Policy defaults untouched (the test harness opens the
    # allowlist on ITS copy only, exactly as SI-19's tests did).
    assert DEFAULT_POLICY["max_stack_depth"] == 8
    assert DEFAULT_POLICY["identity_tol"] == 0.25
    assert DEFAULT_POLICY["capability_eps"] == 0.25
    assert DEFAULT_POLICY["allowed_target_patterns"] == []
    on_disk = load_policy(bundle_dir)
    assert on_disk["identity_tol"] == 0.25
    assert on_disk["capability_eps"] == 0.25

    # Squash still fails closed on the transformer stack (SI-12).
    with pytest.raises(SquashError):
        squash_bundle(bundle_dir, keep=0)


# -- §10.10: CPU budget recorded --------------------------------------------------------

def test_wall_time_recorded_per_cycle(tmp_path):
    _bd, _p, _m, seen, _s = _two_cycle_run(tmp_path)
    for cand in seen:
        assert cand["hyperparams"]["wall_time_seconds"] > 0.0
