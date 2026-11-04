"""SI-88 (SI-83 section 9 slice 5): retention enforcement.

Miniature only: tiny checkpoint files in tmp dirs, journaled
through the real SI-85 writer on the embedded backend.

Covers the SI-83 section 9.5 gate criteria and section 6.4:
- no retention action without a journaled verdict: REJECT
  and ESCALATE paths delete nothing; the ADMIT path's
  retention_decision row exists before the optimizer file
  is gone (hook-proven);
- disk figures recomputed from the section 6.2 assumptions
  and labelled ESTIMATE unless measured bytes are available
  from the manifests (labelled MEASURED);
- protections: newest-2 REJECT; fork-parent REJECT;
  rollback-target REJECT; no reason refused outright;
  already-dropped / no-optimizer REJECT; unknown /
  incomplete REJECT;
- hash-mismatch / missing-file fail-closed with the slice-3
  typed errors, nothing deleted, no ADMIT journaled;
- weights file byte-identical after an optimizer drop;
  history never edited (the drop appends a NEW manifest
  revision with optimizer_blob null);
- delete_weights ESCALATEs (operator-only) and deletes
  nothing;
- post-drop resume kind is weights-exact/optimizer-fresh,
  never exact (the section 4.2 consequence of section 6.4).
"""
import hashlib
import json
import os

import pytest

from si.training_journal import TRAINING_DB_NAME, TrainingJournalWriter
from si.training_retention import (
    ACTION_DELETE_WEIGHTS, ACTION_DROP_OPTIMIZER,
    VERDICT_ADMIT, VERDICT_ESCALATE, VERDICT_REJECT,
    execute_retention, gate_retention, retention_figures,
)
from si.training_rollback import (
    OptimizerFileMissing, OptimizerHashMismatch,
    WeightsHashMismatch, find_manifest, resume, verify_checkpoint,
)
from si.training_fork_prune import fork_run


def sha256_tag(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def make_writer(tmp_path):
    return TrainingJournalWriter.for_embedded(
        str(tmp_path / "journal"), database_name=TRAINING_DB_NAME)


def write_file(path, data: bytes) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path


def journal_checkpoint(w, tmp_path, step=500, weights=None,
                       optimizer=None, complete=True,
                       run_id="run-1"):
    """Journal one miniature checkpoint; returns (manifest, dir).

    Defaults derive the file bytes from the step so multiple
    checkpoints never share contents."""
    weights = weights if weights is not None else f"w-{step}".encode()
    if optimizer is None:
        optimizer = f"o-{step}".encode()
    ckpt_dir = str(tmp_path / "ckpts" / f"step{step}")
    write_file(os.path.join(ckpt_dir, "weights.bin"), weights)
    optimizer_blob = None
    if optimizer is not None:
        write_file(os.path.join(ckpt_dir, "optimizer.bin"), optimizer)
        optimizer_blob = sha256_tag(optimizer)
    manifest = w.record_checkpoint_manifest(
        run_id, step, weights_blob=sha256_tag(weights),
        weights_path=ckpt_dir, weights_file="weights.bin",
        weights_bytes=len(weights), optimizer_blob=optimizer_blob,
        optimizer_bytes=len(optimizer) if optimizer else None,
        checkpoint_complete=complete)
    return manifest, ckpt_dir


def journal_run(w, tmp_path, steps=(500, 1000, 1500)):
    return {s: journal_checkpoint(w, tmp_path, step=s)
            for s in steps}


def decisions(w):
    return [r for r in w.rows("training_proposals")
            if r.get("kind") == "retention_decision"]


def opt_path(ckpt_dir):
    return os.path.join(ckpt_dir, "optimizer.bin")


# -- no reason: refused outright ---------------------------------------------------


def test_no_reason_refused_outright_nothing_journaled(tmp_path):
    w = make_writer(tmp_path)
    _m, ckpt_dir = journal_checkpoint(w, tmp_path)
    before = w.total_rows()
    for bad in (None, "", "   "):
        r = execute_retention(
            w, run_id="run-1", step=500,
            action=ACTION_DROP_OPTIMIZER, reason=bad)
        assert r.verdict == VERDICT_REJECT
        assert r.retention_seq is None
    assert w.total_rows() == before  # nothing journaled at all
    assert decisions(w) == []
    assert os.path.isfile(opt_path(ckpt_dir))  # nothing deleted


# -- ADMIT: journal -> execute -----------------------------------------------------


def test_admitted_drop_journals_before_delete_and_appends_revision(
        tmp_path):
    w = make_writer(tmp_path)
    made = journal_run(w, tmp_path)  # 500, 1000, 1500
    _manifest, ckpt_dir = made[500]
    weights_file = os.path.join(ckpt_dir, "weights.bin")
    with open(weights_file, "rb") as f:
        weights_bytes_before = f.read()
    seen = {}

    def hook(edge):
        # Between journal and deletion: the ADMIT row must
        # already exist and the optimizer file must still be
        # on disk.
        rows = decisions(w)
        seen["decision_rows"] = len(rows)
        seen["decision_verdict"] = rows[-1]["gate_verdict"]
        seen["file_exists"] = os.path.exists(opt_path(ckpt_dir))

    r = execute_retention(
        w, run_id="run-1", step=500, action=ACTION_DROP_OPTIMIZER,
        reason="old checkpoint, disk pressure", operator="op",
        on_before_execute=hook)
    assert r.verdict == VERDICT_ADMIT
    assert seen == {"decision_rows": 1,
                    "decision_verdict": "ADMIT",
                    "file_exists": True}
    # The result carries seqs, path, and MEASURED bytes freed.
    assert r.retention_seq == decisions(w)[-1]["seq"]
    assert r.deleted_path == opt_path(ckpt_dir)
    assert r.bytes_freed == len(b"o-500")
    assert not os.path.exists(opt_path(ckpt_dir))
    row = decisions(w)[-1]
    assert row["keep"] is False
    assert row["step"] == 500
    assert row["reason"] == "old checkpoint, disk pressure"
    assert row["operator"] == "op"
    # The weights file is byte-identical after the drop.
    with open(weights_file, "rb") as f:
        assert f.read() == weights_bytes_before
    # A NEW manifest revision says the optimizer is gone.
    latest = find_manifest(w, "run-1", 500)
    assert latest["seq"] == r.new_manifest_seq
    assert latest["optimizer_blob"] is None
    assert latest["optimizer_bytes"] is None  # documented choice
    assert latest["weights_blob"] == sha256_tag(b"w-500")
    assert latest["weights_bytes"] == len(b"w-500")
    assert latest["checkpoint_complete"] is True
    assert w.verify_chain() is True


def test_history_never_edited_drop_appends_a_revision(tmp_path):
    from si.training_journal import TRAINING_COLLECTIONS
    w = make_writer(tmp_path)
    journal_run(w, tmp_path)

    def snapshot(writer):
        rows = []
        for coll in TRAINING_COLLECTIONS:
            rows.extend(writer.rows(coll))
        rows.sort(key=lambda r: r["seq"])
        return [json.dumps(r, sort_keys=True, default=str)
                for r in rows]

    before = snapshot(w)
    execute_retention(
        w, run_id="run-1", step=500, action=ACTION_DROP_OPTIMIZER,
        reason="retention")
    after = snapshot(w)
    assert after[:len(before)] == before  # append-only
    # Old revision (optimizer present) AND new revision stand.
    revisions = [x for x in w.rows("training_checkpoints")
                 if x.get("kind") == "checkpoint_manifest"
                 and x.get("step") == 500]
    assert len(revisions) == 2
    assert revisions[0]["optimizer_blob"] == sha256_tag(b"o-500")
    assert revisions[1]["optimizer_blob"] is None


def test_post_drop_resume_kind_is_weights_exact_optimizer_fresh(
        tmp_path):
    w = make_writer(tmp_path)
    journal_run(w, tmp_path)
    assert verify_checkpoint(w, "run-1", 500).resume_kind == "exact"
    execute_retention(
        w, run_id="run-1", step=500, action=ACTION_DROP_OPTIMIZER,
        reason="retention")
    v = verify_checkpoint(w, "run-1", 500)
    assert v.resume_kind == "weights-exact/optimizer-fresh"
    assert v.optimizer_blob is None and v.optimizer_path is None
    d = resume(w, "run-1", 500, reason="rewind", verdict="ADMIT")
    assert d.resume_kind == "weights-exact/optimizer-fresh"


def test_already_dropped_checkpoint_rejects_on_second_proposal(
        tmp_path):
    w = make_writer(tmp_path)
    made = journal_run(w, tmp_path)
    execute_retention(
        w, run_id="run-1", step=500, action=ACTION_DROP_OPTIMIZER,
        reason="first drop")
    r = execute_retention(
        w, run_id="run-1", step=500, action=ACTION_DROP_OPTIMIZER,
        reason="drop it again")
    assert r.verdict == VERDICT_REJECT
    assert any("already dropped" in x or "no optimizer" in x
               for x in r.gate_reasons)
    # The REJECT is journaled and the weights still stand.
    assert decisions(w)[-1]["gate_verdict"] == "REJECT"
    assert os.path.isfile(
        os.path.join(made[500][1], "weights.bin"))


# -- gate (a): not a retention target ------------------------------------------------


def test_weights_only_checkpoint_rejected(tmp_path):
    w = make_writer(tmp_path)
    ckpt_dir = str(tmp_path / "ckpts" / "step500")
    write_file(os.path.join(ckpt_dir, "weights.bin"), b"w-only")
    w.record_checkpoint_manifest(
        "run-1", 500, weights_blob=sha256_tag(b"w-only"),
        weights_path=ckpt_dir, weights_file="weights.bin",
        weights_bytes=6, optimizer_blob=None,
        optimizer_bytes=None, checkpoint_complete=True)
    journal_checkpoint(w, tmp_path, step=1000)
    journal_checkpoint(w, tmp_path, step=1500)
    r = execute_retention(
        w, run_id="run-1", step=500, action=ACTION_DROP_OPTIMIZER,
        reason="drop")
    assert r.verdict == VERDICT_REJECT
    assert decisions(w)[-1]["gate_verdict"] == "REJECT"


def test_incomplete_checkpoint_rejected(tmp_path):
    w = make_writer(tmp_path)
    journal_checkpoint(w, tmp_path, step=500, complete=False)
    journal_checkpoint(w, tmp_path, step=1000)
    journal_checkpoint(w, tmp_path, step=1500)
    r = execute_retention(
        w, run_id="run-1", step=500, action=ACTION_DROP_OPTIMIZER,
        reason="drop")
    assert r.verdict == VERDICT_REJECT


def test_unknown_checkpoint_rejected_and_journaled(tmp_path):
    w = make_writer(tmp_path)
    journal_checkpoint(w, tmp_path)
    r = execute_retention(
        w, run_id="run-1", step=999, action=ACTION_DROP_OPTIMIZER,
        reason="drop")
    assert r.verdict == VERDICT_REJECT
    assert decisions(w)[-1]["gate_verdict"] == "REJECT"
    assert decisions(w)[-1]["keep"] is True


def test_unknown_action_rejected_and_journaled(tmp_path):
    w = make_writer(tmp_path)
    _m, ckpt_dir = journal_checkpoint(w, tmp_path)
    r = execute_retention(
        w, run_id="run-1", step=500, action="delete_everything",
        reason="drop")
    assert r.verdict == VERDICT_REJECT
    assert os.path.isfile(opt_path(ckpt_dir))
    assert os.path.isfile(os.path.join(ckpt_dir, "weights.bin"))


# -- protections (b) and (c) ----------------------------------------------------------


def test_newest_two_checkpoints_protected(tmp_path):
    w = make_writer(tmp_path)
    made = journal_run(w, tmp_path)  # 500, 1000, 1500
    for step in (1500, 1000):
        r = execute_retention(
            w, run_id="run-1", step=step,
            action=ACTION_DROP_OPTIMIZER, reason="drop")
        assert r.verdict == VERDICT_REJECT
        assert any("newest" in x for x in r.gate_reasons)
        assert os.path.isfile(opt_path(made[step][1]))
    # The oldest is not protected and drops.
    r = execute_retention(
        w, run_id="run-1", step=500, action=ACTION_DROP_OPTIMIZER,
        reason="drop the oldest")
    assert r.verdict == VERDICT_ADMIT
    assert not os.path.exists(opt_path(made[500][1]))


def test_fork_parent_protected(tmp_path):
    w = make_writer(tmp_path)
    journal_run(w, tmp_path, steps=(500, 1000, 1500, 2000))
    fork_run(w, "run-1", "run-2", 500, reason="branch from 500")
    r = execute_retention(
        w, run_id="run-1", step=500, action=ACTION_DROP_OPTIMIZER,
        reason="drop")
    assert r.verdict == VERDICT_REJECT
    assert any("fork parent" in x for x in r.gate_reasons)
    # A non-parent, non-newest checkpoint is not protected.
    r2 = execute_retention(
        w, run_id="run-1", step=1000, action=ACTION_DROP_OPTIMIZER,
        reason="drop")
    assert r2.verdict == VERDICT_ADMIT


def test_rollback_target_protected(tmp_path):
    w = make_writer(tmp_path)
    journal_run(w, tmp_path, steps=(500, 1000, 1500, 2000))
    resume(w, "run-1", 500, reason="rewind", verdict="ADMIT",
           operator="op")
    r = execute_retention(
        w, run_id="run-1", step=500, action=ACTION_DROP_OPTIMIZER,
        reason="drop")
    assert r.verdict == VERDICT_REJECT
    assert any("rollback target" in x for x in r.gate_reasons)


# -- fail closed: verification (flow step (d)) -----------------------------------------


def test_optimizer_hash_mismatch_fails_closed(tmp_path):
    w = make_writer(tmp_path)
    made = journal_run(w, tmp_path)
    write_file(opt_path(made[500][1]), b"tampered")
    with pytest.raises(OptimizerHashMismatch):
        execute_retention(
            w, run_id="run-1", step=500,
            action=ACTION_DROP_OPTIMIZER, reason="drop")
    # Nothing deleted, no retention verdict journaled.
    assert os.path.isfile(opt_path(made[500][1]))
    assert decisions(w) == []


def test_missing_optimizer_file_fails_closed(tmp_path):
    w = make_writer(tmp_path)
    made = journal_run(w, tmp_path)
    os.remove(opt_path(made[500][1]))
    with pytest.raises(OptimizerFileMissing):
        execute_retention(
            w, run_id="run-1", step=500,
            action=ACTION_DROP_OPTIMIZER, reason="drop")
    assert decisions(w) == []
    assert os.path.isfile(os.path.join(made[500][1], "weights.bin"))


def test_weights_hash_mismatch_fails_closed_optimizer_kept(tmp_path):
    w = make_writer(tmp_path)
    made = journal_run(w, tmp_path)
    write_file(os.path.join(made[500][1], "weights.bin"),
               b"tampered")
    with pytest.raises(WeightsHashMismatch):
        execute_retention(
            w, run_id="run-1", step=500,
            action=ACTION_DROP_OPTIMIZER, reason="drop")
    assert os.path.isfile(opt_path(made[500][1]))
    assert decisions(w) == []


# -- weights deletion: operator-only, ESCALATE, never executed --------------------------


def test_delete_weights_escalates_and_deletes_nothing(tmp_path):
    w = make_writer(tmp_path)
    _m, ckpt_dir = journal_checkpoint(w, tmp_path)
    decision = gate_retention(
        w, run_id="run-1", step=500, action=ACTION_DELETE_WEIGHTS)
    assert decision.verdict == VERDICT_ESCALATE
    assert any("operator-only" in x for x in decision.reasons)
    r = execute_retention(
        w, run_id="run-1", step=500, action=ACTION_DELETE_WEIGHTS,
        reason="free the disk", operator="op")
    assert r.verdict == VERDICT_ESCALATE
    assert r.deleted_path is None and r.bytes_freed is None
    assert os.path.isfile(os.path.join(ckpt_dir, "weights.bin"))
    assert os.path.isfile(opt_path(ckpt_dir))
    row = decisions(w)[-1]
    assert row["gate_verdict"] == "ESCALATE"
    assert row["keep"] is True
    assert w.verify_chain() is True


# -- disk figures ------------------------------------------------------------------------


def test_figures_labels_and_section_6_3_arithmetic(tmp_path):
    w = make_writer(tmp_path)
    # Journal 15 manifest-only checkpoints (figures read the
    # journal, never the files): the section 6.3 v13 shape.
    for i in range(15):
        step = 500 * (i + 1)
        w.record_checkpoint_manifest(
            "run-1", step, weights_blob=sha256_tag(b"w"),
            weights_bytes=100, optimizer_blob=sha256_tag(b"o"),
            optimizer_bytes=400, checkpoint_complete=True)
    fig = retention_figures(w, "run-1")
    assert fig["checkpoint_count"] == 15
    assert fig["checkpoints_retaining_optimizer"] == 15
    assert fig["checkpoints_weights_only"] == 0
    # MEASURED: summed from manifest byte fields, labelled.
    assert fig["measured"]["label"] == "MEASURED"
    assert fig["measured"]["weights_bytes"] == 15 * 100
    assert fig["measured"]["optimizer_bytes"] == 15 * 400
    assert fig["measured"]["total_bytes"] == 15 * 500
    # ESTIMATE: recomputed from the section 6.2 assumptions.
    assert fig["estimate"]["label"] == "ESTIMATE"
    per = fig["estimate"]["per_checkpoint_gb"]
    assert per["weights_bf16"] == 3.2
    assert per["weights_fp32"] == 6.4
    assert per["adamw_moments"] == 12.8
    assert per["layout_a_full"] == 19.2
    assert per["layout_b_full"] == 25.6
    assert per["weights_only_bf16"] == 3.2
    assert per["optimizer_addon"] == 16.0
    full = fig["estimate"]["full_run_gb"]
    assert full["layout_a_total"] == 288.0  # section 6.3
    assert full["layout_b_total"] == 384.0
    assert full["weights_only_total"] == 48.0
    assert full["proposed_retention_total"] == 80.0


def test_figures_reflect_a_drop_in_measured_bytes(tmp_path):
    w = make_writer(tmp_path)
    journal_run(w, tmp_path)  # 3 checkpoints, tiny real bytes
    before = retention_figures(w, "run-1")
    assert before["checkpoints_retaining_optimizer"] == 3
    assert before["measured"]["optimizer_bytes"] == sum(
        len(f"o-{s}".encode()) for s in (500, 1000, 1500))
    execute_retention(
        w, run_id="run-1", step=500, action=ACTION_DROP_OPTIMIZER,
        reason="retention")
    after = retention_figures(w, "run-1")
    assert after["checkpoint_count"] == 3  # revision, not removal
    assert after["checkpoints_retaining_optimizer"] == 2
    assert after["checkpoints_weights_only"] == 1
    assert after["measured"]["optimizer_bytes"] == sum(
        len(f"o-{s}".encode()) for s in (1000, 1500))
    assert after["estimate"]["label"] == "ESTIMATE"
    assert after["measured"]["label"] == "MEASURED"
