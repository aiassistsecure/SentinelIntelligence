"""SI-19: delta authorship for the miniature (RFC SI-18 §4/§7/§8).

Covers the §8 acceptance criteria, in order:

1. signal validity — deterministic construction; the base's
   held-out loss is recorded before training and is measurably
   imperfect (0.00140625, the successor-lean correction's closed
   form — NOT the echo task, which the base already passes
   16/16);
2. disjointness — no signal sequence (train or held-out) appears
   in either probe suite; the signal's RNG stream is its own;
3. gradient correctness — analytic gradients agree with central
   finite differences through the existing SI-15 forward on
   EVERY factor entry, to the pinned tolerance (measured worst
   agreement is published in docs/SI-19-delta-authorship.md);
4. real training, real improvement — train_delta returns a
   candidate whose held-out loss is far below the base's, whose
   training loss decreased, and refuses a non-improving result;
5. determinism — two train_delta runs produce byte-identical
   candidate blobs and identical hyperparameters except the
   labelled wall-time field;
6. refusals intact — unknown signal, foreign/blind targets,
   foreign rank/alpha, and non-bundle-backed minds all refuse;
7. format and policy — the candidate validates through
   construct/validate, one allowlistable target, within caps;
8. gate honesty end to end — a miniature cycle driven by the
   REAL TinyTransformerMind journals checks 3/4 genuinely
   measured, and the journaled verdict equals an independent
   ProbeContext evaluation of the same candidate;
9. no widenings — base stored-bytes accounting is unchanged by
   an authored delta landing (deltas are new bytes only);
10. CPU budget — the training wall time is recorded in the
    hyperparameters (measured and published in the as-built
    doc); the suite itself is the practicality proof.
"""
import json
import os

import numpy as np
import pytest

import si.tiny_training as tt
from si.bundle import init_bundle, load_bundle, load_policy
from si.gate import Gate, Intent
from si.journal import open_journal
from si.loop import run_cycle
from si.probes import ProbeContext, derive_probe_blind_targets
from si.tiny_seed import build_tiny_weights, capability_inputs, identity_inputs, seed_tiny_bundle
from si.tiny_transformer import TinyTransformerError, TinyTransformerMind, forward
from si.transformer_delta import KIND, validate_transformer_delta

#: Gradient-check tolerance, pinned from measurement (RFC §8.3):
#: the author's full run measured a worst absolute agreement of
#: 3.807e-08 over all 512 factor entries at eps=0.01 (published in
#: docs/SI-19-delta-authorship.md); 1e-5 leaves ~260x headroom
#: for platform float variation without loosening the check into
#: meaninglessness.
GRAD_TOL = 1e-5
GRAD_EPS = 0.01

KBIAS0 = "model.language_model.layers.0.self_attn.k_proj.bias"
QPROJ = "model.language_model.layers.0.self_attn.q_proj.weight"


def _seeded(tmp_path, name="tiny19"):
    bundle_dir = init_bundle(str(tmp_path / (name + ".si")), name=name,
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    return bundle_dir


def _policy(bundle_dir, **over):
    policy = load_policy(bundle_dir)
    policy["allowed_target_patterns"] = ["*"]
    policy.update(over)
    return policy


def _mind(bundle_dir):
    return TinyTransformerMind(bundle_dir, load_bundle(bundle_dir))


# -- §8.1/§8.2: the signal ------------------------------------------------------

def test_signal_construction_is_deterministic():
    train_a, held_a, toks_a = tt.signal_sequences()
    train_b, held_b, toks_b = tt.signal_sequences()
    assert (train_a, held_a, toks_a) == (train_b, held_b, toks_b)
    assert len(train_a) == tt.N_TRAIN == 32
    assert len(held_a) == tt.N_HELD_OUT == 16
    assert len(set(toks_a)) == tt.N_LAST_TOKENS == 8
    assert not {tuple(s) for s in train_a} & {tuple(s) for s in held_a}
    for seq in train_a + held_a:
        assert 2 <= len(seq) <= 8
        assert all(0 <= t < 32 for t in seq)
        assert seq[-1] in toks_a
    weights = build_tiny_weights()
    sig_a = tt.build_training_signal(weights)
    sig_b = tt.build_training_signal(weights)
    for part in ("train", "held_out"):
        for (seq_a, tgt_a), (seq_b, tgt_b) in zip(sig_a[part], sig_b[part]):
            assert seq_a == seq_b
            assert np.array_equal(tgt_a, tgt_b)


def test_signal_is_disjoint_from_both_probe_suites():
    train_seqs, held_seqs, _ = tt.signal_sequences()
    probe = {tuple(s) for s in identity_inputs()} \
        | {tuple(s) for s in capability_inputs()}
    signal = {tuple(s) for s in train_seqs} | {tuple(s) for s in held_seqs}
    assert signal.isdisjoint(probe)
    # The signal's RNG stream is its own — not the identity (+2)
    # or capability (+3) streams recorded in si/tiny_seed.py.
    assert tt.SIGNAL_SEED_OFFSET not in (2, 3)


def test_base_is_measurably_imperfect_on_the_signal():
    weights = build_tiny_weights()
    signal = tt.build_training_signal(weights)
    base_held = tt.signal_loss(weights, signal["held_out"])
    base_train = tt.signal_loss(weights, signal["train"])
    # The recorded-before-training score (RFC §8.1): the base's
    # loss is exactly the correction's closed form 2*C^2/32 —
    # measurably imperfect (a perfect score would be 0.0), and
    # NOT the echo task the base already passes 16/16.
    closed_form = 2 * tt.CORRECTION_SCALE ** 2 / 32
    assert base_held == pytest.approx(closed_form, abs=1e-12)
    assert base_train == pytest.approx(closed_form, abs=1e-12)
    assert base_held > 0.0


# -- §8.3: gradient correctness ---------------------------------------------------

def test_forward_replica_is_bit_identical_to_library_forward():
    weights = build_tiny_weights()
    signal = tt.build_training_signal(weights)
    seqs = identity_inputs() + capability_inputs() \
        + [s for s, _ in signal["train"]] + [s for s, _ in signal["held_out"]]
    for seq in seqs:
        logits, _h = tt.forward_with_final_hidden(seq, weights)
        assert np.array_equal(logits.astype(np.float32),
                              forward(seq, weights))


def test_analytic_gradients_match_finite_differences():
    weights = build_tiny_weights()
    signal = tt.build_training_signal(weights)
    subset = signal["train"][:8]
    A0, B0 = tt.initial_factors()
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
    assert worst <= GRAD_TOL, \
        f"analytic/FD worst disagreement {worst:.3e} > pinned {GRAD_TOL:g}"


# -- §8.4/§8.7: real training, real improvement, format ----------------------------

def test_train_delta_trains_and_improves_held_out(tmp_path):
    bundle_dir = _seeded(tmp_path)
    manifest = load_bundle(bundle_dir)
    mind = _mind(bundle_dir)
    cand = mind.train_delta(mind.propose([], None))
    hp = cand["hyperparams"]
    assert hp["training_signal_id"] == tt.TRAINING_SIGNAL_ID
    assert hp["target"] == tt.TARGET_NAME == "lm_head.weight"
    assert hp["base_held_out_loss"] == pytest.approx(0.00140625, abs=1e-12)
    assert hp["held_out_loss_after"] < hp["base_held_out_loss"] / 10
    assert hp["train_loss_final"] < hp["train_loss_initial"]
    assert hp["held_out_examples_closer"] == hp["held_out_examples"] == 16
    assert hp["norm_projection_activations"] == 0
    assert hp["wall_time_seconds"] > 0.0
    # Format + policy (§8.7): validates against the verified
    # manifest, exactly one target, within the SI-9b caps, and no
    # policy default was touched to get there.
    record = cand["record"]
    validate_transformer_delta(record, cand["blob_bytes"], manifest)
    assert [t["name"] for t in record["targets"]] == [tt.TARGET_NAME]
    assert record["targets"][0]["kind"] == "lora"
    assert record["targets"][0]["rank"] == tt.RANK <= 8
    assert record["norm"] == pytest.approx(hp["target_norm"])
    assert record["norm"] <= 1.0
    assert len(cand["blob_bytes"]) == record["bytes"] <= 4 * 1024 * 1024
    assert cand["norm"] == record["norm"]


def test_train_delta_refuses_a_non_improving_candidate(tmp_path, monkeypatch):
    bundle_dir = _seeded(tmp_path)
    mind = _mind(bundle_dir)
    monkeypatch.setattr(tt, "factor_loss", lambda *a, **k: 999.0)
    with pytest.raises(ValueError, match="did not improve"):
        mind.train_delta(mind.propose([], None))


# -- §8.5: determinism --------------------------------------------------------------

def test_train_delta_is_deterministic(tmp_path):
    bundle_dir = _seeded(tmp_path)
    mind = _mind(bundle_dir)
    intent = mind.propose([], None)
    first = mind.train_delta(intent)
    second = mind.train_delta(intent)
    assert first["blob_bytes"] == second["blob_bytes"]
    assert first["record"] == second["record"]
    hp_first = dict(first["hyperparams"])
    hp_second = dict(second["hyperparams"])
    hp_first.pop("wall_time_seconds")
    hp_second.pop("wall_time_seconds")
    assert hp_first == hp_second


# -- §8.6: refusals -------------------------------------------------------------------

def test_refusals_are_intact(tmp_path):
    bundle_dir = _seeded(tmp_path)
    mind = _mind(bundle_dir)
    base = dict(kind=KIND, reason="SI-19 refusal probe", parents=["base"],
                source_seqs=[1])
    with pytest.raises(ValueError, match="unknown training signal"):
        mind.train_delta(Intent(training_signal_id="no-such-signal",
                                targets=[tt.TARGET_NAME], **base))
    with pytest.raises(ValueError, match="unknown training signal"):
        mind.train_delta(Intent(training_signal_id="", **base))
    # A blind-set target is refused, never silently retargeted.
    with pytest.raises(ValueError, match="never silently retargets"):
        mind.train_delta(Intent(
            training_signal_id=tt.TRAINING_SIGNAL_ID,
            targets=[KBIAS0], **base))
    # So is any other single foreign target.
    with pytest.raises(ValueError, match="never silently retargets"):
        mind.train_delta(Intent(
            training_signal_id=tt.TRAINING_SIGNAL_ID,
            targets=[QPROJ], **base))
    # A resized candidate is refused, not quietly reshaped.
    with pytest.raises(ValueError, match="refuses a resized candidate"):
        mind.train_delta(Intent(
            training_signal_id=tt.TRAINING_SIGNAL_ID,
            targets=[tt.TARGET_NAME], rank=2, alpha=2, **base))
    # The trained target is provably outside the blind set.
    assert tt.TARGET_NAME not in derive_probe_blind_targets()
    # A mind with no bundle cannot train at all (existing refusal).
    with pytest.raises(TinyTransformerError, match="not bundle-backed"):
        TinyTransformerMind().train_delta(Intent(
            training_signal_id=tt.TRAINING_SIGNAL_ID,
            targets=[tt.TARGET_NAME], **base))
    # A non-miniature bundle is refused at adapter construction,
    # exactly as before SI-19 (existing refusal, unchanged).
    demo_dir = init_bundle(str(tmp_path / "demo.si"), name="demo19",
                           journal_backend="embedded")
    with pytest.raises(TinyTransformerError):
        TinyTransformerMind(demo_dir, load_bundle(demo_dir))


def test_propose_emits_exactly_the_named_signal_intent(tmp_path):
    bundle_dir = _seeded(tmp_path)
    mind = _mind(bundle_dir)
    first = mind.propose([], None)
    second = mind.propose([], None)
    assert first.to_dict() == second.to_dict()
    assert first.kind == KIND
    assert first.targets == [tt.TARGET_NAME]
    assert first.training_signal_id == tt.TRAINING_SIGNAL_ID
    assert first.rank == tt.RANK and first.alpha == tt.ALPHA
    assert first.parents == ["base"]
    assert first.source_seqs  # provenance needs a real journal seq


# -- §8.8: gate honesty end to end ------------------------------------------------------

def test_end_to_end_cycle_with_the_real_mind(tmp_path):
    bundle_dir = _seeded(tmp_path)
    manifest = load_bundle(bundle_dir)
    mind = _mind(bundle_dir)
    policy = _policy(bundle_dir)
    intent = mind.propose([], None)

    # Independent evaluation of the same (deterministic) candidate
    # through a fresh ProbeContext, BEFORE the cycle runs.
    candidate = mind.train_delta(intent)
    ctx = ProbeContext(bundle_dir, manifest, policy=policy)
    independent = Gate(policy).evaluate(intent, candidate, ctx)
    assert independent.verdict == "ADMIT"

    res = run_cycle(bundle_dir, mind, gate=Gate(policy))
    assert res["verdict"] == independent.verdict == "ADMIT"
    assert res["verified"] is True and res["rolled_back"] is False
    assert res["delta_id"] == "d0001" and res["head"] == "d0001"

    manifest = load_bundle(bundle_dir)
    # The landed delta IS the independently trained candidate.
    assert manifest["deltas"][0]["blob"] == candidate["record"]["blob"]
    journal = open_journal(bundle_dir, manifest)
    try:
        edits = journal.backend.find("self_edits",
                                     filter={"edit_id": res["edit_id"]})
        assert len(edits) == 1
        checks = {c["name"]: c
                  for c in edits[0]["gate_report"]["checks"]}
        independent_checks = {c.name: c for c in independent.checks}
        for name in ("identity_continuity", "capability_floor"):
            journaled = checks[name]
            assert journaled["measured"] is not None
            assert journaled["passed"] is True
            assert "measured-pass" in journaled["detail"]
            assert journaled["measured"] == pytest.approx(
                independent_checks[name].measured, abs=1e-12)
        prov = journal.backend.find("provenance",
                                    filter={"child": "d0001"})
        verify = [p for p in prov
                  if p["relation"] == "transformer-probe-verify"]
        assert len(verify) == 1
        vreport = json.loads(verify[0]["note"])
        assert vreport["passed"] is True
        assert vreport["capability"]["echo_accuracy"] == 1.0
    finally:
        journal.close()


# -- §8.9: stored-bytes accounting ---------------------------------------------------------

def _blobs_total(bundle_dir):
    root = os.path.join(bundle_dir, "blobs", "sha256")
    return sum(os.path.getsize(os.path.join(root, f))
               for f in os.listdir(root))


def test_authored_delta_adds_only_its_own_bytes(tmp_path):
    bundle_dir = _seeded(tmp_path)
    before = load_bundle(bundle_dir)
    base_before = dict(before["base"])
    total_before = _blobs_total(bundle_dir)
    mind = _mind(bundle_dir)
    res = run_cycle(bundle_dir, mind, gate=Gate(_policy(bundle_dir)))
    assert res["verdict"] == "ADMIT"
    after = load_bundle(bundle_dir)
    # The base section — stored bytes included — is untouched.
    assert after["base"] == base_before
    delta = after["deltas"][0]
    assert _blobs_total(bundle_dir) == total_before + delta["bytes"]
