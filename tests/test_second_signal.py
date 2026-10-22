"""SI-23: a second authored signal for the miniature
(RFC SI-22 §9 outline, §10 acceptance criteria).

Covers the §10 criteria, in order:

1. dispatch and recorded selection — the registry serves
   exactly the two named signals and refuses every other id
   with the existing ``ValueError`` discipline; foreign-target
   and foreign-rank/alpha refusals apply per signal; a test
   asserts, per cycle on both verdict paths, that the
   journaled ``training-hyperparams`` signal id equals the
   signal the driver configured for that cycle. No autonomous
   selection exists: ``propose`` emits exactly the configured
   signal;
2. gradient check for signal 2 at head ``d0001`` AND at base,
   published in docs/SI-23-second-signal.md (tolerance pinned
   at 1e-5, the SI-19 pin);
3. streams audited — signal 2's two new named constants
   (+23 sequences, +29 factor init) and the guard asserting
   all four miniature training offsets and both probe offsets
   are pairwise distinct;
4. disjointness proven — all five §5.4 obligations, including
   signal 2 against signal 1's full 48 sequences, and the
   measured last-token overlap ({9, 22}, size 2) published;
5. echo-trap records — signal 2's pre-training held-out loss
   at base and at ``d0001`` is 0.00140625 at both heads: at
   base it CONFIRMS the §5.6.3 closed-form derivation by
   measurement; at ``d0001`` it recurs by construction under
   weights-under-training anchoring (asserted as such);
6. determinism per cycle — a full decisive run repeated from
   the seed reproduces both cycles' records and blob hashes
   exactly, except the labelled wall-time field;
7. gate honesty per cycle — each cycle's journaled checks 3/4
   equal an independent ``ProbeContext`` evaluation; cycle
   2's verdict is REJECT (cumulative identity drift
   0.2963416576385498 > tol 0.25), reported AS MEASURED. No
   criterion requires an ADMIT;
8. cumulative accounting and §5.6.4 diagnostics published —
   per-probe drift contributions / incremental movements /
   composed drift with worst probes named, the alignment
   diagnostic, and the folded updates' inner product and
   norms, all asserted here and published in the as-built doc;
9. first-cycle reproduction — the default (signal 1) first
   cycle reproduces SI-19/SI-21 exactly: candidate blob
   ``sha256:2761e8ff...6ca9e`` and the same gate numbers;
10. VERIFY, unwind, untouched scope — cycle 1's §6.6 VERIFY
    passes as in SI-19; cycle 2 never landed (REJECT), so NO
    authored two-delta stack exists and no unwind is claimed
    (asserted: head stays ``d0001``, one delta only); exactly
    two cycles attempted; policy defaults, tolerances and
    caps unchanged; ``si squash`` still fails closed.

The decisive run's driver shape: ONE ``TinyTransformerMind``
reused through ``si.run.run_cycles`` — one ``run_cycles``
call for cycle 1 (signal 1, the default), then the driver
explicitly reconfigures the same mind to signal 2
(``set_training_signal``, the recorded selection) and calls
``run_cycles`` again for cycle 2. The mind never chooses.
"""
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
    build_tiny_weights,
    capability_inputs,
    identity_inputs,
    seed_tiny_bundle,
)
from si.tiny_transformer import TinyTransformerMind, assemble_weights
from si.transformer_delta import KIND, parse_blob, validate_transformer_delta

GRAD_TOL = 1e-5
GRAD_EPS = 0.01

SI19_BLOB = "sha256:2761e8ffc139ce3e107c11da035e8a73c20792945a494f4e493dccf418b6ca9e"
SI19_DRIFT = 0.17189218290150166
SI19_REGRESSION = 0.1657181391492486
SI19_NORM = 0.23555380990005192

# SI-23 measured values (author's runs; as-built doc publishes
# them with their labels). The decisive cycle-2 verdict is
# REJECT — measured, not required and not tuned for.
S2_AT_BASE_BLOB = "sha256:131cc242728024b4f833bc78ea627825f1e8616972fc2670603673e1814645f4"
S2_AT_BASE_NORM = 0.21317186144376168
S2_AT_BASE_DRIFT = 0.16672966815531254
S2_AT_BASE_REGRESSION = 0.2418281203135848
S2_CYCLE2_BLOB = "sha256:184cc7f06cd9cb3bd83fbed466b6a939c7e2e18cf7b8a72be70c51043a6e6cf7"
S2_CYCLE2_NORM = 0.21317186599953494
S2_CYCLE2_DRIFT = 0.2963416576385498
S2_CYCLE2_REGRESSION = 0.15023326850496233
CLOSED_FORM = 0.00140625

KBIAS0 = "model.language_model.layers.0.self_attn.k_proj.bias"
QPROJ = "model.language_model.layers.0.self_attn.q_proj.weight"


def _seeded(tmp_path, name="tiny23"):
    bundle_dir = init_bundle(str(tmp_path / (name + ".si")), name=name,
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    return bundle_dir


def _policy(bundle_dir, **over):
    policy = load_policy(bundle_dir)
    policy["allowed_target_patterns"] = ["*"]
    policy.update(over)
    return policy


def _spying_mind(bundle_dir, signal_id=None):
    kw = {} if signal_id is None else {"training_signal_id": signal_id}
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


def _decisive_run(tmp_path, name="tiny23"):
    """The decisive two-cycle run (see module docstring for the
    driver shape): one reused mind, cycle 1 signal 1 via
    ``run_cycles``, explicit reconfiguration, cycle 2 signal 2
    via ``run_cycles``."""
    bundle_dir = _seeded(tmp_path, name)
    policy = _policy(bundle_dir)
    mind, seen = _spying_mind(bundle_dir)
    sum1 = run_cycles(bundle_dir, mind, 1, gate=Gate(policy))
    mind.set_training_signal(tt.TRAINING_SIGNAL_ID_2)
    sum2 = run_cycles(bundle_dir, mind, 1, gate=Gate(policy))
    return bundle_dir, policy, mind, seen, sum1, sum2


def _grad_worst(weights, signal_id):
    """Worst analytic-vs-central-finite-difference disagreement
    over ALL 512 factor entries, on an 8-example subset — the
    SI-19 gradient-check shape, for ``signal_id``'s loss at
    ``weights``."""
    signal = tt.build_training_signal(weights, signal_id)
    subset = signal["train"][:8]
    A0, B0 = tt.initial_factors(signal_id)
    lm_head = np.asarray(weights[tt.TARGET_NAME], dtype=np.float64)
    hiddens = {tuple(s): tt.final_hidden(weights, s) for s, _ in subset}
    _loss, grad_A, grad_B = tt.analytic_gradients(
        A0, B0, subset, hiddens, lm_head)
    assert np.max(np.abs(grad_A)) > 0.0 and np.max(np.abs(grad_B)) > 0.0
    worst = 0.0
    for i in range(tt.RANK):
        for j in range(32):
            plus = A0.copy(); plus[i, j] += GRAD_EPS
            minus = A0.copy(); minus[i, j] -= GRAD_EPS
            fd = (tt.factor_loss(weights, plus, B0, subset)
                  - tt.factor_loss(weights, minus, B0, subset)) / (2 * GRAD_EPS)
            worst = max(worst, abs(fd - grad_A[i, j]))
    for i in range(32):
        for j in range(tt.RANK):
            plus = B0.copy(); plus[i, j] += GRAD_EPS
            minus = B0.copy(); minus[i, j] -= GRAD_EPS
            fd = (tt.factor_loss(weights, A0, plus, subset)
                  - tt.factor_loss(weights, A0, minus, subset)) / (2 * GRAD_EPS)
            worst = max(worst, abs(fd - grad_B[i, j]))
    return worst


# -- §10.3: streams audited ------------------------------------------------------

def test_offset_table_is_pairwise_distinct():
    # The full miniature offset table (RFC §5.3 + SI-23 §9.2):
    # probe streams +2/+3 (si/tiny_seed.py), signal 1 +11/+17,
    # signal 2 +23/+29 — all six pairwise distinct, and the
    # registry records exactly the named constants.
    offsets = [2, 3, tt.SIGNAL_SEED_OFFSET, tt.INIT_SEED_OFFSET,
               tt.SIGNAL2_SEED_OFFSET, tt.INIT2_SEED_OFFSET]
    assert offsets == [2, 3, 11, 17, 23, 29]
    assert len(set(offsets)) == 6
    assert tt.SIGNAL2_SEED_OFFSET not in (2, 3, 11, 17)
    assert tt.INIT2_SEED_OFFSET not in (2, 3, 11, 17)
    spec1 = tt.SIGNAL_REGISTRY[tt.TRAINING_SIGNAL_ID]
    spec2 = tt.SIGNAL_REGISTRY[tt.TRAINING_SIGNAL_ID_2]
    assert (spec1["seed_offset"], spec1["init_offset"]) == (11, 17)
    assert (spec2["seed_offset"], spec2["init_offset"]) == (23, 29)
    # The two signals' initial factors start from different
    # streams — signal 2 does not start at signal 1's point.
    A1, B1 = tt.initial_factors(tt.TRAINING_SIGNAL_ID)
    A2, B2 = tt.initial_factors(tt.TRAINING_SIGNAL_ID_2)
    assert not np.array_equal(A1, A2) and not np.array_equal(B1, B2)
    # Signal 1's defaults are byte-for-byte SI-19's.
    A0, B0 = tt.initial_factors()
    assert np.array_equal(A0, A1) and np.array_equal(B0, B1)


# -- §10.4: disjointness proven ----------------------------------------------------

def test_signal2_disjointness_all_five_obligations():
    s1_train, s1_held, toks1 = tt.signal_sequences(tt.TRAINING_SIGNAL_ID)
    s2_train, s2_held, toks2 = tt.signal_sequences(tt.TRAINING_SIGNAL_ID_2)
    assert len(s2_train) == tt.N_TRAIN == 32
    assert len(s2_held) == tt.N_HELD_OUT == 16
    assert len(set(toks2)) == tt.N_LAST_TOKENS == 8
    s2 = {tuple(s) for s in s2_train} | {tuple(s) for s in s2_held}
    assert len(s2) == 48
    # Obligation 1 + 2: vs each probe suite, separately.
    assert s2.isdisjoint({tuple(s) for s in identity_inputs()})
    assert s2.isdisjoint({tuple(s) for s in capability_inputs()})
    # Obligation 3: vs signal 1's FULL 48 (32 train + 16 held-out).
    s1 = {tuple(s) for s in s1_train} | {tuple(s) for s in s1_held}
    assert len(s1) == 48 and s2.isdisjoint(s1)
    # Obligation 4: signal 2's own partitions are disjoint.
    assert {tuple(s) for s in s2_train}.isdisjoint(
        {tuple(s) for s in s2_held})
    # Obligation 5: last-token overlap is a MEASURED fact,
    # published — signal 2's set is [30, 12, 2, 22, 19, 0, 18, 9],
    # signal 1's is [28, 25, 31, 22, 3, 15, 1, 9]; the overlap is
    # exactly {9, 22} (2 of 8). Sequences, not tokens, are the
    # unit of disjointness.
    assert toks2 == [30, 12, 2, 22, 19, 0, 18, 9]
    assert sorted(set(toks1) & set(toks2)) == [9, 22]
    # Construction is deterministic, and signal 1's construction
    # is unchanged (its last tokens are SI-19's published set).
    assert tt.signal_sequences(tt.TRAINING_SIGNAL_ID_2) == \
        (s2_train, s2_held, toks2)
    assert toks1 == [28, 25, 31, 22, 3, 15, 1, 9]
    for seq in s2_train + s2_held:
        assert 2 <= len(seq) <= 8
        assert all(0 <= t < 32 for t in seq)
        assert seq[-1] in toks2


# -- §10.5: echo-trap records --------------------------------------------------------

def test_signal2_echo_trap_at_base_confirms_closed_form():
    weights = build_tiny_weights()
    signal = tt.build_training_signal(weights, tt.TRAINING_SIGNAL_ID_2)
    # At base the derivation is CONFIRMED by measurement: the
    # zero-delta loss is the correction's closed form.
    closed = 2 * tt.CORRECTION_SCALE ** 2 / 32
    assert closed == pytest.approx(CLOSED_FORM, abs=1e-15)
    assert tt.signal_loss(weights, signal["held_out"]) == \
        pytest.approx(CLOSED_FORM, abs=1e-12)
    assert tt.signal_loss(weights, signal["train"]) == \
        pytest.approx(CLOSED_FORM, abs=1e-12)
    assert tt.signal_loss(weights, signal["held_out"]) > 0.0
    # Signal 2's targets really are predecessor targets: each
    # target differs from the base logits by exactly the
    # predecessor correction, and by a DIFFERENT correction
    # than signal 1's successor correction on the same input.
    from si.tiny_transformer import forward
    seq, target = signal["train"][0]
    base = forward(seq, weights)[-1].astype(np.float64)
    corr = np.asarray(target) - base
    want = np.zeros(32); want[(seq[-1] - 1) % 32] += 0.15
    want[seq[-1]] -= 0.15
    assert np.allclose(corr, want, atol=1e-12)


# -- §10.2: gradient checks ------------------------------------------------------------

def test_gradient_check_signal2_at_base():
    worst = _grad_worst(build_tiny_weights(), tt.TRAINING_SIGNAL_ID_2)
    # Measured 5.5249277028738815e-08 (as-built doc); the pin is
    # SI-19's 1e-5, unchanged — no tolerance loosening (§11).
    assert worst <= GRAD_TOL


def test_gradient_check_signal2_at_d0001(tmp_path):
    bundle_dir = _seeded(tmp_path)
    policy = _policy(bundle_dir)
    mind, _seen = _spying_mind(bundle_dir)
    summary = run_cycles(bundle_dir, mind, 1, gate=Gate(policy))
    assert summary["head"] == "d0001"
    manifest = load_bundle(bundle_dir)
    weights = assemble_weights(bundle_dir, manifest, policy)
    worst = _grad_worst(weights, tt.TRAINING_SIGNAL_ID_2)
    # Measured 5.062663472655505e-08 at head d0001 (as-built
    # doc) — the configuration the decisive cycle trains in.
    assert worst <= GRAD_TOL
    # Echo trap at d0001, in the same configuration: signal 2's
    # pre-training held-out loss at the landed head recurs at
    # the closed form BY CONSTRUCTION (targets derive from the
    # weights under training) — asserted as a construction
    # artifact, not as fresh evidence about the base (§10.5).
    signal = tt.build_training_signal(weights, tt.TRAINING_SIGNAL_ID_2)
    assert tt.signal_loss(weights, signal["held_out"]) == \
        pytest.approx(CLOSED_FORM, abs=1e-12)


# -- §10.1: dispatch and recorded selection ----------------------------------------------

def test_registry_dispatch_and_refusals(tmp_path):
    # Exactly two signals are served.
    assert set(tt.SIGNAL_REGISTRY) == \
        {tt.TRAINING_SIGNAL_ID, tt.TRAINING_SIGNAL_ID_2}
    assert tt.TRAINING_SIGNAL_ID == "tiny-successor-lean-001"
    assert tt.TRAINING_SIGNAL_ID_2 == "tiny-predecessor-lean-001"
    with pytest.raises(ValueError, match="unknown training signal"):
        tt.signal_spec("tiny-successor-lean-002")

    bundle_dir = _seeded(tmp_path)
    mind = TinyTransformerMind(bundle_dir, load_bundle(bundle_dir))
    base = dict(kind=KIND, reason="SI-23 refusal probe", parents=["base"],
                source_seqs=[1])
    # Unknown ids refuse through train_delta, per the existing
    # discipline — including a near-miss id.
    for bad in ("no-such-signal", "", "tiny-predecessor-lean-002"):
        with pytest.raises(ValueError, match="unknown training signal"):
            mind.train_delta(Intent(training_signal_id=bad,
                                    targets=[tt.TARGET_NAME], **base))
    # Foreign-target and resized refusals apply PER SIGNAL:
    # signal 2 refuses exactly as signal 1 does.
    for sid in (tt.TRAINING_SIGNAL_ID, tt.TRAINING_SIGNAL_ID_2):
        with pytest.raises(ValueError, match="never silently retargets"):
            mind.train_delta(Intent(
                training_signal_id=sid, targets=[KBIAS0], **base))
        with pytest.raises(ValueError, match="never silently retargets"):
            mind.train_delta(Intent(
                training_signal_id=sid, targets=[QPROJ], **base))
        with pytest.raises(ValueError, match="refuses a resized candidate"):
            mind.train_delta(Intent(
                training_signal_id=sid, targets=[tt.TARGET_NAME],
                rank=2, alpha=2, **base))
    # Selection is explicit: the constructor choice and the
    # setter both refuse unknown ids, and propose emits exactly
    # the configured signal — defaulting to signal 1.
    with pytest.raises(ValueError, match="unknown training signal"):
        TinyTransformerMind(bundle_dir, load_bundle(bundle_dir),
                            training_signal_id="no-such-signal")
    assert mind.training_signal_id == tt.TRAINING_SIGNAL_ID
    assert mind.propose([], None).training_signal_id == tt.TRAINING_SIGNAL_ID
    mind.set_training_signal(tt.TRAINING_SIGNAL_ID_2)
    assert mind.training_signal_id == tt.TRAINING_SIGNAL_ID_2
    intent = mind.propose([], None)
    assert intent.training_signal_id == tt.TRAINING_SIGNAL_ID_2
    assert intent.targets == [tt.TARGET_NAME]
    assert intent.rank == tt.RANK and intent.alpha == tt.ALPHA
    with pytest.raises(ValueError, match="unknown training signal"):
        mind.set_training_signal("no-such-signal")
    # Both signals share the pinned shape (§5.6.2 minimality).
    for sid in (tt.TRAINING_SIGNAL_ID, tt.TRAINING_SIGNAL_ID_2):
        spec = tt.signal_spec(sid)
        assert spec["signal_id"] == sid
    assert tt.RANK == 8 and tt.ALPHA == 8 and tt.CORRECTION_SCALE == 0.15
    assert tt.TARGET_ANCHORING == "weights-under-training"


def test_signal2_train_delta_deterministic_and_refuses_nonimproving(
        tmp_path, monkeypatch):
    bundle_dir = _seeded(tmp_path)
    mind = TinyTransformerMind(
        bundle_dir, load_bundle(bundle_dir),
        training_signal_id=tt.TRAINING_SIGNAL_ID_2)
    intent = mind.propose([], None)
    first = mind.train_delta(intent)
    second = mind.train_delta(intent)
    assert first["blob_bytes"] == second["blob_bytes"]
    assert first["record"] == second["record"]
    assert first["record"]["blob"] == S2_AT_BASE_BLOB
    hp_a = dict(first["hyperparams"]); hp_b = dict(second["hyperparams"])
    hp_a.pop("wall_time_seconds"); hp_b.pop("wall_time_seconds")
    assert hp_a == hp_b
    assert hp_a["training_signal_id"] == tt.TRAINING_SIGNAL_ID_2
    validate_transformer_delta(first["record"], first["blob_bytes"],
                               load_bundle(bundle_dir))
    monkeypatch.setattr(tt, "factor_loss", lambda *a, **k: 999.0)
    with pytest.raises(ValueError, match="did not improve"):
        mind.train_delta(intent)


# -- §9.3 / §10.7: signal 2's at-base first-cycle sanity --------------------------------

def test_signal2_at_base_first_cycle_sanity(tmp_path):
    bundle_dir = _seeded(tmp_path)
    policy = _policy(bundle_dir)
    before = _blobs_total(bundle_dir)
    mind, seen = _spying_mind(bundle_dir, tt.TRAINING_SIGNAL_ID_2)
    res = run_cycle(bundle_dir, mind, gate=Gate(policy))
    # Reported AS MEASURED: at base, signal 2's candidate
    # ADMITs and VERIFYs. These are at-base facts about signal
    # 2's definition — NOT evidence about head d0001 (§9.3).
    assert res["verdict"] == "ADMIT" and res["head"] == "d0001"
    assert res["verified"] is True and res["rolled_back"] is False
    hp = seen[0]["hyperparams"]
    assert hp["training_signal_id"] == tt.TRAINING_SIGNAL_ID_2
    assert hp["head"] == "base" and hp["target_source_head"] == "base"
    assert hp["stack_depth"] == 0
    assert hp["base_held_out_loss"] == pytest.approx(CLOSED_FORM, abs=1e-12)
    assert hp["held_out_loss_after"] == pytest.approx(
        3.891045992787381e-06, abs=1e-12)
    assert hp["held_out_examples_closer"] == hp["held_out_examples"] == 16
    assert hp["norm_projection_activations"] == 0
    assert hp["wall_time_seconds"] > 0.0
    assert seen[0]["record"]["blob"] == S2_AT_BASE_BLOB
    assert seen[0]["record"]["norm"] == pytest.approx(
        S2_AT_BASE_NORM, abs=1e-12)
    manifest = load_bundle(bundle_dir)
    assert _blobs_total(bundle_dir) == before + 2360
    checks, _row = _checks_of_edit(bundle_dir, manifest, res["edit_id"])
    assert checks["identity_continuity"]["measured"] == pytest.approx(
        S2_AT_BASE_DRIFT, abs=1e-9)
    assert checks["identity_continuity"]["passed"] is True
    assert checks["capability_floor"]["measured"] == pytest.approx(
        S2_AT_BASE_REGRESSION, abs=1e-9)
    assert checks["capability_floor"]["passed"] is True
    # Recorded selection on the ADMIT path: the journaled
    # hyperparams name signal 2 (§10.1).
    note = _hyperparams_notes(bundle_dir, manifest, "d0001")
    assert note["training_signal_id"] == tt.TRAINING_SIGNAL_ID_2
    assert note["head"] == "base"
    # §6.6 VERIFY of the landed at-base head, re-measured.
    ok, report = _verify_probe_head(bundle_dir, manifest, policy=policy)
    assert ok is True
    assert report["identity"]["max_drift"] == pytest.approx(
        S2_AT_BASE_DRIFT, abs=1e-9)
    assert report["capability"]["echo_accuracy"] == 1.0


# -- §10.9: first-cycle reproduction ------------------------------------------------------

def test_first_cycle_reproduces_si19_exactly(tmp_path):
    bundle_dir = _seeded(tmp_path)
    policy = _policy(bundle_dir)
    mind, seen = _spying_mind(bundle_dir)  # default selection
    assert mind.training_signal_id == tt.TRAINING_SIGNAL_ID
    res = run_cycle(bundle_dir, mind, gate=Gate(policy))
    assert res["verdict"] == "ADMIT" and res["head"] == "d0001"
    assert seen[0]["record"]["blob"] == SI19_BLOB
    assert seen[0]["record"]["norm"] == pytest.approx(SI19_NORM, abs=1e-12)
    manifest = load_bundle(bundle_dir)
    assert manifest["deltas"][0]["blob"] == SI19_BLOB
    checks, _row = _checks_of_edit(bundle_dir, manifest, res["edit_id"])
    assert checks["identity_continuity"]["measured"] == pytest.approx(
        SI19_DRIFT, abs=1e-12)
    assert checks["capability_floor"]["measured"] == pytest.approx(
        SI19_REGRESSION, abs=1e-12)


# -- the decisive run: §10.1/§10.5/§10.7/§10.8 ---------------------------------------------

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
    # Cycle 1 reproduces SI-19 exactly (criterion 9, in-run).
    assert hp1["training_signal_id"] == tt.TRAINING_SIGNAL_ID
    assert hp1["head"] == "base" and hp1["stack_depth"] == 0
    assert seen[0]["record"]["blob"] == SI19_BLOB
    # Cycle 2: signal 2, explicitly selected, trained at d0001
    # under the unchanged freshness contract and anchoring.
    assert hp2["training_signal_id"] == tt.TRAINING_SIGNAL_ID_2
    assert hp2["head"] == "d0001" and hp2["target_source_head"] == "d0001"
    assert hp2["target_anchoring"] == "weights-under-training"
    assert hp2["stack_depth"] == 1
    assert seen[1]["record"]["parent"] == "d0001"
    assert seen[1]["record"]["id"] == "d0002"
    assert seen[1]["record"]["blob"] == S2_CYCLE2_BLOB
    assert seen[1]["record"]["norm"] == pytest.approx(
        S2_CYCLE2_NORM, abs=1e-12)
    # Echo trap at d0001 (§10.5): the pre-training held-out
    # loss recurs at the closed form by construction.
    assert hp2["base_held_out_loss"] == pytest.approx(CLOSED_FORM, abs=1e-12)
    assert hp2["held_out_loss_after"] == pytest.approx(
        3.891045006596239e-06, abs=1e-12)
    assert hp2["held_out_examples_closer"] == hp2["held_out_examples"] == 16
    assert hp2["train_loss_initial"] == pytest.approx(
        0.0026047023050908037, abs=1e-12)
    assert hp2["train_loss_final"] == pytest.approx(
        3.3622214870436204e-06, abs=1e-12)
    assert hp1["wall_time_seconds"] > 0.0 and hp2["wall_time_seconds"] > 0.0

    manifest = load_bundle(bundle_dir)
    edit1, edit2 = sum1["edit_ids"][0], sum2["edit_ids"][0]
    # Gate honesty (§10.7): cycle 2's journaled checks 3/4 equal
    # an independent ProbeContext evaluation of the same
    # candidate against the same pre-cycle head (d0001).
    fresh = TinyTransformerMind(
        bundle_dir, load_bundle(bundle_dir),
        training_signal_id=tt.TRAINING_SIGNAL_ID_2)
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
    assert journaled["identity_continuity"]["measured"] == pytest.approx(
        S2_CYCLE2_DRIFT, abs=1e-9)
    assert journaled["identity_continuity"]["passed"] is False
    assert "measured-fail" in journaled["identity_continuity"]["detail"]
    assert journaled["capability_floor"]["measured"] == pytest.approx(
        S2_CYCLE2_REGRESSION, abs=1e-9)
    assert journaled["capability_floor"]["passed"] is True
    # Recorded selection on BOTH verdict paths (§10.1): cycle
    # 1's note (ADMIT, under d0001) names signal 1; cycle 2's
    # note (REJECT, under its self-edit) names signal 2.
    note1 = _hyperparams_notes(bundle_dir, manifest, "d0001")
    assert note1["training_signal_id"] == tt.TRAINING_SIGNAL_ID
    assert note1["head"] == "base"
    note2 = _hyperparams_notes(bundle_dir, manifest, edit2)
    assert note2["training_signal_id"] == tt.TRAINING_SIGNAL_ID_2
    assert note2["head"] == "d0001" and note2["stack_depth"] == 1
    # The landed head's cumulative drift after the REJECT is
    # unchanged — re-measured (§10.8), and cycle 1's VERIFY
    # stands as in SI-19.
    ok, report = _verify_probe_head(bundle_dir, manifest, policy=policy)
    assert ok is True
    assert report["identity"]["max_drift"] == pytest.approx(
        SI19_DRIFT, abs=1e-9)


# -- §10.8: the §5.6.4 diagnostics -----------------------------------------------------------

def _folded_update(candidate):
    parsed = parse_blob(candidate["blob_bytes"])
    target = candidate["record"]["targets"][0]
    A = np.asarray(parsed.tensors[target["factor_keys"][0]],
                   dtype=np.float64)
    B = np.asarray(parsed.tensors[target["factor_keys"][1]],
                   dtype=np.float64)
    return (float(target["alpha"]) / float(target["rank"])) * (B @ A)


def test_composition_diagnostics(tmp_path):
    bundle_dir, policy, mind, seen, sum1, sum2 = _decisive_run(tmp_path)
    manifest = load_bundle(bundle_dir)
    assert manifest["head"] == "d0001"  # candidate 2 never landed
    ctx = ProbeContext(bundle_dir, manifest, policy=policy)
    cand2 = seen[1]
    # (i) Per-probe contributions (measurements; the incremental
    # movement is a DERIVATION from two measured forwards —
    # composed minus d0001 — labelled as such in the doc).
    expected = {
        "tiny-id-0": (0.17189218290150166, 0.100458356551826, 0.2092923242598772),
        "tiny-id-1": (0.15088725090026855, 0.07999316975474358, 0.15134692192077637),
        "tiny-id-2": (0.11858986946754158, 0.14863967895507812, 0.14906740188598633),
        "tiny-id-3": (0.14782869815826416, 0.14852380752563477, 0.2963416576385498),
        "tiny-id-4": (0.11720322235487401, 0.15136146545410156, 0.23394920490682125),
        "tiny-id-5": (0.09891846403479576, 0.11731460690498352, 0.14004023000597954),
        "tiny-id-6": (0.12363102938979864, 0.14838377758860588, 0.1332004750147462),
        "tiny-id-7": (0.08795571327209473, 0.16672981530427933, 0.2018739078193903),
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
    # Worst probes, named: d0001's is tiny-id-0, the incremental
    # movement's is tiny-id-7, the composed drift's is tiny-id-3
    # — and the composed worst equals the gate's measured drift.
    assert max(rows, key=lambda q: rows[q][0]) == "tiny-id-0"
    assert max(rows, key=lambda q: rows[q][1]) == "tiny-id-7"
    assert max(rows, key=lambda q: rows[q][2]) == "tiny-id-3"
    assert rows["tiny-id-3"][2] == pytest.approx(S2_CYCLE2_DRIFT, abs=1e-9)
    # (ii) Alignment (derivation): at the worst composed probe
    # the two movements ALIGN (positive inner product, majority
    # sign agreement) — the REJECT is attributable to alignment
    # at tiny-id-3, not to a large second movement alone.
    i3 = [q for q, _x, _e in ctx.identity_probes()].index("tiny-id-3")
    inner_worst = float(np.dot(m1_all[i3], m2_all[i3]))
    sign_worst = float(np.mean(np.sign(m1_all[i3]) == np.sign(m2_all[i3])))
    assert inner_worst == pytest.approx(0.021769049613635714, abs=1e-9)
    assert sign_worst == pytest.approx(0.625, abs=1e-12)
    a_all = np.concatenate(m1_all); b_all = np.concatenate(m2_all)
    assert float(np.dot(a_all, b_all)) == pytest.approx(
        0.07637087369869666, abs=1e-9)
    assert float(np.mean(np.sign(a_all) == np.sign(b_all))) == \
        pytest.approx(0.54296875, abs=1e-12)
    # (iii) Folded updates (measurements): inner product and
    # each update's Frobenius norm.
    d1u, d2u = _folded_update(seen[0]), _folded_update(cand2)
    assert float(np.sum(d1u * d2u)) == pytest.approx(
        0.004356018453758104, abs=1e-9)
    assert float(np.linalg.norm(d1u)) == pytest.approx(SI19_NORM, abs=1e-9)
    assert float(np.linalg.norm(d2u)) == pytest.approx(
        S2_CYCLE2_NORM, abs=1e-9)
    # (iv) Composed capability (measurements via the gate's own
    # evaluation): head-incremental regression passes, echo 1.0
    # vs the recorded floor 0.9375.
    rep = Gate(policy).evaluate(mind.propose([], ctx), cand2, ctx)
    checks = {c.name: c for c in rep.checks}
    assert checks["capability_floor"].measured == pytest.approx(
        S2_CYCLE2_REGRESSION, abs=1e-9)
    targets = ctx.capability_targets()
    correct = sum(
        1 for cid, x, _e in ctx.capability_suite()
        if int(np.argmax(np.asarray(ctx.forward_candidate(x, cand2))))
        == int(targets[cid]))
    assert correct == 16 and ctx.echo_floor() == 0.9375


# -- §10.6: determinism per cycle --------------------------------------------------------------

def test_decisive_run_repeats_exactly(tmp_path):
    _b1, _p1, _m1, seen1, s1a, s2a = _decisive_run(tmp_path / "a", "tiny23")
    _b2, _p2, _m2, seen2, s1b, s2b = _decisive_run(tmp_path / "b", "tiny23")
    assert s1a == s1b and s2a == s2b
    assert load_bundle(_b1)["deltas"] == load_bundle(_b2)["deltas"]
    for cand_a, cand_b in zip(seen1, seen2):
        assert cand_a["blob_bytes"] == cand_b["blob_bytes"]
        assert cand_a["record"] == cand_b["record"]
        hp_a = dict(cand_a["hyperparams"]); hp_b = dict(cand_b["hyperparams"])
        hp_a.pop("wall_time_seconds"); hp_b.pop("wall_time_seconds")
        assert hp_a == hp_b
    man1, man2 = load_bundle(_b1), load_bundle(_b2)
    for edit_a, edit_b in zip(s1a["edit_ids"] + s2a["edit_ids"],
                              s1b["edit_ids"] + s2b["edit_ids"]):
        checks_a, _ = _checks_of_edit(_b1, man1, edit_a)
        checks_b, _ = _checks_of_edit(_b2, man2, edit_b)
        for name in ("identity_continuity", "capability_floor"):
            assert checks_a[name]["measured"] == checks_b[name]["measured"]


# -- §10.10: VERIFY / unwind honesty / untouched scope ---------------------------------------------

def test_unwind_honesty_and_untouched_scope(tmp_path):
    bundle_dir = _seeded(tmp_path)
    policy = _policy(bundle_dir)
    before = _blobs_total(bundle_dir)
    _bd, _pol, _mind, seen, sum1, sum2 = (
        bundle_dir, policy, None, None, None, None)
    # Run the decisive shape inline so the totals line up with
    # this bundle (the helper above seeds its own).
    mind, seen = _spying_mind(bundle_dir)
    sum1 = run_cycles(bundle_dir, mind, 1, gate=Gate(policy))
    mind.set_training_signal(tt.TRAINING_SIGNAL_ID_2)
    sum2 = run_cycles(bundle_dir, mind, 1, gate=Gate(policy))
    assert sum1["admitted"] == 1 and sum2["rejected"] == 1
    manifest = load_bundle(bundle_dir)
    # No authored two-delta stack exists: exactly one delta
    # landed, cycle 2's candidate was never written (+0 B),
    # and NO unwind is claimed or exercised on an authored
    # stack — SI-21's fixture unwind remains the only unwind
    # evidence (stated in the as-built doc).
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
    assert tt.TARGET_ANCHORING == "weights-under-training"
    # Squash still fails closed on the transformer stack (SI-12).
    with pytest.raises(SquashError):
        squash_bundle(bundle_dir, keep=0)
    # The signal-1 definition is byte-for-byte SI-19's (§11: no
    # redefinition of signal 1).
    assert tt.TRAINING_SIGNAL_ID == "tiny-successor-lean-001"
    assert tt.SIGNAL_SEED_OFFSET == 11 and tt.INIT_SEED_OFFSET == 17
    assert tt.CORRECTION_SCALE == 0.15
    assert tt.N_TRAIN == 32 and tt.N_HELD_OUT == 16
