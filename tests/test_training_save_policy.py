"""SI-89 (SI-83 section 6.4 point 3): SAVE_EVERY policy gate.

Miniature only: proposals journaled through the real SI-85
writer on the embedded backend; no trainer, no config, no
files beyond the journal's own storage.

Covers the brief's gate semantics:
- >= 500 on a full fine-tune ADMITs;
- < 500 on a full fine-tune ESCALATEs (never ADMITs), with
  the disk arithmetic present in the result AND the journal
  row, so an operator decides on numbers;
- no reason is refused outright and journals nothing;
- invalid inputs (save_every <= 0, total_steps <= 0,
  unknown layout) REJECT;
- a non-full-fine-tune ADMITs with the scoping reason;
- the section 6.3 arithmetic anchors at 500 and 250;
- the journal row lands in training_proposals with the
  SI-89 save_every_decision kind and its verdict BEFORE
  the result returns (hook-proven).
"""
from si.training_journal import (
    EVENT_KINDS, TRAINING_DB_NAME, TrainingJournalWriter,
)
from si.training_save_policy import (
    MIN_SAVE_EVERY_FULL_FINE_TUNE, VERDICT_ADMIT,
    VERDICT_ESCALATE, VERDICT_REJECT, gate_save_every,
    propose_save_every, save_every_figures,
)

STEPS = 7311  # the section 6.3 v13-shaped run


def make_writer(tmp_path):
    return TrainingJournalWriter.for_embedded(
        str(tmp_path / "journal"), database_name=TRAINING_DB_NAME)


def decisions(w):
    return [r for r in w.rows("training_proposals")
            if r.get("kind") == "save_every_decision"]


# -- arithmetic anchors (section 6.3) -------------------------------------------


def test_figures_anchor_at_500():
    fig = save_every_figures(500, STEPS, layout="A")
    assert fig["label"] == "ESTIMATE"
    assert fig["checkpoint_count"] == 15
    per = fig["per_checkpoint_gb"]
    assert per["weights_bf16"] == 3.2
    assert per["weights_fp32"] == 6.4
    assert per["adamw_moments"] == 12.8
    assert per["layout_a_full"] == 19.2
    assert per["layout_b_full"] == 25.6
    assert per["weights_only_bf16"] == 3.2
    assert per["optimizer_addon"] == 16.0
    full = fig["full_retention_gb"]
    assert full["layout_a_total"] == 288.0  # section 6.3
    assert full["layout_b_total"] == 384.0
    proposed = fig["proposed_retention_gb"]
    assert proposed["weights_only_total"] == 48.0
    assert proposed["total"] == 80.0  # section 6.3


def test_figures_anchor_at_250():
    fig = save_every_figures(250, STEPS)
    assert fig["label"] == "ESTIMATE"
    assert fig["checkpoint_count"] == 30
    assert fig["full_retention_gb"]["layout_a_total"] == 576.0
    assert fig["full_retention_gb"]["layout_b_total"] == 768.0
    assert fig["proposed_retention_gb"]["total"] == 128.0


def test_figures_checkpoint_count_is_ceiling():
    assert save_every_figures(500, 500)["checkpoint_count"] == 1
    assert save_every_figures(500, 501)["checkpoint_count"] == 2
    assert save_every_figures(100, STEPS)["checkpoint_count"] == 74


# -- gate: ADMIT paths ------------------------------------------------------------


def test_ge_500_full_fine_tune_admits(tmp_path):
    w = make_writer(tmp_path)
    for se in (500, 501, 1000):
        d = gate_save_every(save_every=se, total_steps=STEPS,
                            full_fine_tune=True)
        assert d.verdict == VERDICT_ADMIT
        assert d.figures["checkpoint_count"] >= 1
    r = propose_save_every(
        w, run_id="run-1", save_every=500, full_fine_tune=True,
        total_steps=STEPS, layout="A", reason="standing practice",
        operator="op")
    assert r.verdict == VERDICT_ADMIT
    assert r.decision_seq == decisions(w)[-1]["seq"]
    assert w.verify_chain() is True


def test_non_full_fine_tune_admits_with_scoping_reason(tmp_path):
    w = make_writer(tmp_path)
    r = propose_save_every(
        w, run_id="run-1", save_every=100, full_fine_tune=False,
        total_steps=STEPS, reason="adapter run, small checkpoints")
    assert r.verdict == VERDICT_ADMIT
    assert any("only to full fine-tunes" in x
               for x in r.gate_reasons)
    row = decisions(w)[-1]
    assert row["gate_verdict"] == "ADMIT"
    assert row["full_fine_tune"] is False


# -- gate: ESCALATE, never ADMIT ----------------------------------------------------


def test_below_500_full_fine_tune_escalates_never_admits(tmp_path):
    w = make_writer(tmp_path)
    for se in (1, 100, 250, 499):
        d = gate_save_every(save_every=se, total_steps=STEPS,
                            full_fine_tune=True)
        assert d.verdict == VERDICT_ESCALATE
        assert d.verdict != VERDICT_ADMIT
    assert MIN_SAVE_EVERY_FULL_FINE_TUNE == 500
    r = propose_save_every(
        w, run_id="run-1", save_every=250, full_fine_tune=True,
        total_steps=STEPS, layout="B", reason="want finer rollback",
        operator="op")
    assert r.verdict == VERDICT_ESCALATE
    # Arithmetic is present in the returned result...
    assert r.figures["checkpoint_count"] == 30
    assert r.figures["full_retention_gb"]["layout_a_total"] == 576.0
    assert r.figures["proposed_retention_gb"]["total"] == 128.0
    assert any("576" in x for x in r.gate_reasons)
    # ...and in the journaled row, so the operator decides
    # on numbers from the record alone.
    row = decisions(w)[-1]
    assert row["gate_verdict"] == "ESCALATE"
    assert row["save_every"] == 250
    assert row["proposed_save_every"] == 250
    assert row["full_fine_tune"] is True
    assert row["total_steps"] == STEPS
    assert row["layout"] == "B"
    assert row["operator"] == "op"
    assert row["reason"] == "want finer rollback"
    assert row["figures"]["checkpoint_count"] == 30
    assert row["checkpoint_count"] == 30
    assert row["layout_a_total_gb"] == 576.0
    assert row["proposed_retention_total_gb"] == 128.0
    assert w.verify_chain() is True


# -- no reason: refused outright, nothing journaled ---------------------------------


def test_no_reason_refused_outright_nothing_journaled(tmp_path):
    w = make_writer(tmp_path)
    before = w.total_rows()
    for bad in (None, "", "   "):
        r = propose_save_every(
            w, run_id="run-1", save_every=250, full_fine_tune=True,
            total_steps=STEPS, reason=bad)
        assert r.verdict == VERDICT_REJECT
        assert r.decision_seq is None
        assert r.figures is None
    assert w.total_rows() == before
    assert decisions(w) == []


# -- invalid inputs REJECT ------------------------------------------------------------


def test_invalid_inputs_reject(tmp_path):
    w = make_writer(tmp_path)
    cases = [
        dict(save_every=0, total_steps=STEPS, full_fine_tune=True),
        dict(save_every=-250, total_steps=STEPS, full_fine_tune=True),
        dict(save_every=250, total_steps=0, full_fine_tune=True),
        dict(save_every=250, total_steps=-1, full_fine_tune=True),
        dict(save_every=250, total_steps=STEPS, full_fine_tune=True,
             layout="Z"),
        dict(save_every="abc", total_steps=STEPS, full_fine_tune=True),
    ]
    for kw in cases:
        d = gate_save_every(**kw)
        assert d.verdict == VERDICT_REJECT
        assert d.figures is None
        r = propose_save_every(w, run_id="run-1", reason="try", **kw)
        assert r.verdict == VERDICT_REJECT
    # Invalid (but reasoned) proposals are journaled as REJECT.
    rows = decisions(w)
    assert len(rows) == len(cases)
    assert all(x["gate_verdict"] == "REJECT" for x in rows)
    assert w.verify_chain() is True


# -- journal-before-return, kind registration ------------------------------------------


def test_journal_row_lands_before_result_returns(tmp_path):
    w = make_writer(tmp_path)
    seen = {}

    def hook(edge):
        rows = decisions(w)
        seen["rows_at_hook"] = len(rows)
        seen["verdict_at_hook"] = rows[-1]["gate_verdict"]
        seen["seq_matches"] = rows[-1]["seq"] == edge["decision_seq"]

    r = propose_save_every(
        w, run_id="run-1", save_every=600, full_fine_tune=True,
        total_steps=STEPS, reason="coarser saves", operator="op",
        on_before_return=hook)
    assert r.verdict == VERDICT_ADMIT
    assert seen == {"rows_at_hook": 1, "verdict_at_hook": "ADMIT",
                    "seq_matches": True}
    assert r.decision_seq == decisions(w)[-1]["seq"]


def test_kind_is_registered_as_si89_extension():
    assert "save_every_decision" in EVENT_KINDS
    # The pre-existing section 2.5 kinds are all still present.
    for kind in ("run_started", "checkpoint_manifest", "eval_moment",
                 "window_consumed", "prune_proposed", "prune_verdict",
                 "rollback", "fork", "retention_decision",
                 "watcher_gap"):
        assert kind in EVENT_KINDS
