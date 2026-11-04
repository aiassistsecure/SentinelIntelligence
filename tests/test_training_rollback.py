"""SI-86 (SI-83 section 9 slice 3): training rollback / replay.

Miniature only: tiny checkpoint files in tmp dirs, journaled
through the real SI-85 writer on the embedded backend.

Covers the SI-83 section 9.3 gate criteria and section 4.2/4.3:
- exact resume ONLY when optimizer state hash-verifies;
- weights-only resume labelled weights-exact/optimizer-fresh
  (descriptor AND its journal row);
- fail closed: weights/optimizer hash mismatch, missing file,
  incomplete checkpoint, unknown run/step;
- the rollback row is journaled (parents + verdict + operator)
  before the descriptor returns; original rows untouched and
  later checkpoints still present after a rollback;
- replay writes a NEW eval_moment linked to the original by
  replay_of_seq / provenance_relation="replayed_from"; a
  disagreeing replay leaves BOTH rows standing;
- replay on a hash-mismatched checkpoint runs no eval and
  writes no replay row; replay with no original moment fails
  closed; a rollback without a reason is refused;
- forward training replay is a guard only: it refuses where
  data_order_seed is null (SI-83 4.3) and where one is set.
"""
import hashlib
import os

import pytest

from si.training_journal import TRAINING_DB_NAME, TrainingJournalWriter
from si.training_rollback import (
    RESUME_EXACT, RESUME_WEIGHTS_EXACT, CheckpointIncomplete,
    CheckpointNotFound, ForwardReplayRefused, MissingReason,
    NoOriginalEvalMoment, OptimizerFileMissing,
    OptimizerHashMismatch, WeightsFileMissing,
    WeightsHashMismatch, forward_replay, replay_eval, resume,
    verify_checkpoint,
)


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


def journal_checkpoint(w, tmp_path, step=500, weights=b"weights-v1",
                       optimizer=b"opt-v1", complete=True,
                       run_id="run-1"):
    """Journal one miniature checkpoint; returns (manifest, dir).

    weights_path names the step directory and weights_file the
    file inside it -- the watcher's manifest convention."""
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


def proposals(w):
    return w.rows("training_proposals")


def moments(w):
    return [r for r in w.rows("training_eval_moments")
            if r.get("kind") == "eval_moment"]


# -- resume classification ---------------------------------------------------


def test_exact_resume_only_when_optimizer_hash_verifies(tmp_path):
    w = make_writer(tmp_path)
    journal_checkpoint(w, tmp_path, optimizer=b"opt-v1")
    v = verify_checkpoint(w, "run-1", 500)
    assert v.resume_kind == RESUME_EXACT == "exact"
    assert v.optimizer_path.endswith("optimizer.bin")
    d = resume(w, "run-1", 500, reason="regressed",
               verdict="ADMIT", operator="op")
    assert d.resume_kind == "exact"
    assert d.weights_blob == sha256_tag(b"weights-v1")


def test_weights_only_resume_labelled_weights_exact_optimizer_fresh(
        tmp_path):
    w = make_writer(tmp_path)
    journal_checkpoint(w, tmp_path, optimizer=None)
    d = resume(w, "run-1", 500, reason="rewind", verdict="ADMIT",
               operator="op")
    assert d.resume_kind == RESUME_WEIGHTS_EXACT
    assert d.resume_kind == "weights-exact/optimizer-fresh"
    assert d.optimizer_path is None and d.optimizer_blob is None
    # The label is in the journal row too (SI-83 9.3 gate).
    row = proposals(w)[-1]
    assert row["resume_kind"] == "weights-exact/optimizer-fresh"


# -- fail closed ---------------------------------------------------------------


def test_weights_hash_mismatch_fails_closed(tmp_path):
    w = make_writer(tmp_path)
    _m, ckpt_dir = journal_checkpoint(w, tmp_path)
    write_file(os.path.join(ckpt_dir, "weights.bin"), b"tampered")
    before = w.total_rows()
    with pytest.raises(WeightsHashMismatch):
        verify_checkpoint(w, "run-1", 500)
    with pytest.raises(WeightsHashMismatch):
        resume(w, "run-1", 500, reason="x", verdict="ADMIT")
    assert w.total_rows() == before  # nothing journaled as success
    assert proposals(w) == []


def test_optimizer_hash_mismatch_fails_closed(tmp_path):
    w = make_writer(tmp_path)
    _m, ckpt_dir = journal_checkpoint(w, tmp_path)
    write_file(os.path.join(ckpt_dir, "optimizer.bin"), b"tampered")
    with pytest.raises(OptimizerHashMismatch):
        verify_checkpoint(w, "run-1", 500)
    assert proposals(w) == []


def test_missing_weights_file_fails_closed(tmp_path):
    w = make_writer(tmp_path)
    _m, ckpt_dir = journal_checkpoint(w, tmp_path)
    os.remove(os.path.join(ckpt_dir, "weights.bin"))
    with pytest.raises(WeightsFileMissing):
        verify_checkpoint(w, "run-1", 500)


def test_missing_optimizer_file_fails_closed(tmp_path):
    w = make_writer(tmp_path)
    _m, ckpt_dir = journal_checkpoint(w, tmp_path)
    os.remove(os.path.join(ckpt_dir, "optimizer.bin"))
    with pytest.raises(OptimizerFileMissing):
        verify_checkpoint(w, "run-1", 500)


def test_incomplete_checkpoint_refused(tmp_path):
    w = make_writer(tmp_path)
    journal_checkpoint(w, tmp_path, complete=False)
    with pytest.raises(CheckpointIncomplete):
        verify_checkpoint(w, "run-1", 500)
    with pytest.raises(CheckpointIncomplete):
        resume(w, "run-1", 500, reason="x")


def test_unknown_run_or_step_refused(tmp_path):
    w = make_writer(tmp_path)
    journal_checkpoint(w, tmp_path)
    with pytest.raises(CheckpointNotFound):
        verify_checkpoint(w, "run-1", 999)
    with pytest.raises(CheckpointNotFound):
        verify_checkpoint(w, "no-such-run", 500)


# -- journal before landing ----------------------------------------------------


def test_rollback_row_journaled_before_descriptor_returns(tmp_path):
    w = make_writer(tmp_path)
    manifest, _d = journal_checkpoint(w, tmp_path, step=500)
    journal_checkpoint(w, tmp_path, step=1000, weights=b"w2",
                       optimizer=b"o2")
    total_before = w.total_rows()
    d = resume(w, "run-1", 500, reason="step-1000 regressed",
               verdict="ADMIT", operator="mark")
    rows = proposals(w)
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "rollback"
    assert row["relation"] == "rolled_back_to"
    assert row["parents"] == [manifest["seq"]]
    assert row["target_step"] == 500
    assert row["reason"] == "step-1000 regressed"
    assert row["gate_verdict"] == "ADMIT"
    assert row["operator"] == "mark"
    assert d.rollback_seq == row["seq"]
    # Original rows untouched; the later checkpoint still stands.
    assert w.total_rows() == total_before + 1
    steps = sorted(r["step"] for r in w.rows("training_checkpoints")
                   if r.get("kind") == "checkpoint_manifest")
    assert steps == [500, 1000]
    assert w.verify_chain() is True


def test_missing_reason_refused(tmp_path):
    w = make_writer(tmp_path)
    journal_checkpoint(w, tmp_path)
    for bad in (None, "", "   "):
        with pytest.raises(MissingReason):
            resume(w, "run-1", 500, reason=bad)
    assert proposals(w) == []


# -- eval replay -----------------------------------------------------------------


def test_replay_writes_new_moment_linked_to_original(tmp_path):
    w = make_writer(tmp_path)
    journal_checkpoint(w, tmp_path)
    original = w.record_eval_moment(
        "run-1", 500, eval_suite="suite-v1", tier0_score=0.50,
        discrimination_score=0.25, loss=1.5)
    seen = []

    def eval_fn(path):
        seen.append(path)
        return {"tier0_score": 0.75, "discrimination_score": 0.25,
                "loss": 1.5}

    replay = replay_eval(w, "run-1", 500, eval_fn, operator="op")
    assert len(seen) == 1 and seen[0].endswith("weights.bin")
    assert replay["kind"] == "eval_moment"
    assert replay["replay_of_seq"] == original["seq"]
    assert replay["replay_of_entry_hash"] == original["entry_hash"]
    assert replay["provenance_relation"] == "replayed_from"
    assert replay["eval_suite"] == "suite-v1"  # original's suite
    assert replay["tier0_score"] == 0.75
    rows = moments(w)
    assert len(rows) == 2
    assert w.verify_chain() is True


def test_disagreeing_replay_leaves_both_rows_standing(tmp_path):
    w = make_writer(tmp_path)
    journal_checkpoint(w, tmp_path)
    w.record_eval_moment("run-1", 500, eval_suite="suite-v1",
                         tier0_score=0.50)
    replay_eval(w, "run-1", 500,
                lambda path: {"tier0_score": 0.10})
    rows = sorted(moments(w), key=lambda r: r["seq"])
    assert len(rows) == 2
    assert rows[0]["tier0_score"] == 0.50  # original unedited
    assert rows[0].get("replay_of_seq") is None
    assert rows[1]["tier0_score"] == 0.10
    assert rows[1]["replay_of_seq"] == rows[0]["seq"]


def test_replay_on_hash_mismatch_runs_no_eval_and_writes_no_row(
        tmp_path):
    w = make_writer(tmp_path)
    _m, ckpt_dir = journal_checkpoint(w, tmp_path)
    w.record_eval_moment("run-1", 500, eval_suite="suite-v1",
                         tier0_score=0.5)
    write_file(os.path.join(ckpt_dir, "weights.bin"), b"tampered")
    calls = []

    def eval_fn(path):
        calls.append(path)
        return {"tier0_score": 0.9}

    with pytest.raises(WeightsHashMismatch):
        replay_eval(w, "run-1", 500, eval_fn)
    assert calls == []                      # eval never ran
    assert len(moments(w)) == 1             # only the original


def test_replay_without_original_moment_fails_closed(tmp_path):
    w = make_writer(tmp_path)
    journal_checkpoint(w, tmp_path)
    calls = []
    with pytest.raises(NoOriginalEvalMoment):
        replay_eval(w, "run-1", 500,
                    lambda p: calls.append(p) or {"tier0_score": 1})
    assert calls == []
    assert moments(w) == []


def test_agreeing_replay_records_same_scores_as_new_row(tmp_path):
    w = make_writer(tmp_path)
    journal_checkpoint(w, tmp_path)
    original = w.record_eval_moment(
        "run-1", 500, eval_suite="suite-v1", tier0_score=0.50,
        discrimination_score=0.25)
    replay = replay_eval(
        w, "run-1", 500,
        lambda path: {"tier0_score": 0.50,
                      "discrimination_score": 0.25})
    assert replay["tier0_score"] == original["tier0_score"] == 0.50
    assert replay["discrimination_score"] == 0.25
    assert replay["seq"] != original["seq"]  # a NEW row
    assert replay["replay_of_seq"] == original["seq"]
    assert len(moments(w)) == 2


def test_original_rows_byte_identical_after_rollback(tmp_path):
    import json
    from si.training_journal import TRAINING_COLLECTIONS
    w = make_writer(tmp_path)
    journal_checkpoint(w, tmp_path, step=500)
    journal_checkpoint(w, tmp_path, step=1000, weights=b"w2",
                       optimizer=b"o2")
    w.record_eval_moment("run-1", 500, eval_suite="s",
                         tier0_score=0.5)

    def snapshot(writer):
        rows = []
        for coll in TRAINING_COLLECTIONS:
            rows.extend(writer.rows(coll))
        rows.sort(key=lambda r: r["seq"])
        return [json.dumps(r, sort_keys=True, default=str)
                for r in rows]

    before = snapshot(w)
    resume(w, "run-1", 500, reason="rewind", verdict="ADMIT",
           operator="op")
    after = snapshot(w)
    # Every pre-existing row is byte-identical; exactly one new
    # row (the rollback) was appended after them.
    assert after[:len(before)] == before
    assert len(after) == len(before) + 1


# -- forward training replay: guard only (SI-83 4.3) --------------------------


def test_forward_replay_refused_on_null_seed(tmp_path):
    w = make_writer(tmp_path)
    journal_checkpoint(w, tmp_path)  # data_order_seed stays null
    with pytest.raises(ForwardReplayRefused) as exc:
        forward_replay(w, "run-1", 500)
    assert "data_order_seed" in str(exc.value)
    assert proposals(w) == []  # the guard journals nothing


def test_forward_replay_refused_even_with_seed_recorded(tmp_path):
    w = make_writer(tmp_path)
    _m, ckpt_dir = journal_checkpoint(w, tmp_path)
    w.record_checkpoint_manifest(
        "run-1", 900, weights_blob=sha256_tag(b"weights-v1"),
        weights_path=ckpt_dir, weights_file="weights.bin",
        data_order_seed=1234, checkpoint_complete=True)
    with pytest.raises(ForwardReplayRefused):
        forward_replay(w, "run-1", 900)


def test_watcher_style_directory_manifest_resolves(tmp_path):
    # weights_path = bare step directory, no weights_file extra:
    # resolution falls back to the watcher's naming convention.
    w = make_writer(tmp_path)
    ckpt_dir = str(tmp_path / "ckpts" / "step700")
    write_file(os.path.join(ckpt_dir, "model.safetensors"), b"w700")
    write_file(os.path.join(ckpt_dir, "optimizer.pt"), b"o700")
    w.record_checkpoint_manifest(
        "run-1", 700, weights_blob=sha256_tag(b"w700"),
        weights_path=ckpt_dir, optimizer_blob=sha256_tag(b"o700"),
        checkpoint_complete=True)
    v = verify_checkpoint(w, "run-1", 700)
    assert v.resume_kind == "exact"
    assert v.weights_path.endswith("model.safetensors")
    assert v.optimizer_path.endswith("optimizer.pt")


def test_bare_file_weights_path_resolves(tmp_path):
    # Writer-style manifest: weights_path is the file itself.
    w = make_writer(tmp_path)
    path = write_file(str(tmp_path / "ck.bin"), b"solo-weights")
    w.record_checkpoint_manifest(
        "run-1", 42, weights_blob=sha256_tag(b"solo-weights"),
        weights_path=path, checkpoint_complete=True)
    d = resume(w, "run-1", 42, reason="bare file form")
    assert d.weights_path == path
    assert d.resume_kind == "weights-exact/optimizer-fresh"
