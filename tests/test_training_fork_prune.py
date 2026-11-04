"""SI-87 (SI-83 section 9 slice 4): fork + gated prune.

Miniature only: tiny checkpoint files and tiny JSONL corpora
in tmp dirs, journaled through the real SI-85 writer on the
embedded backend.

Covers the SI-83 section 9.4 gate criteria and sections
4.4 / 5.2 / 5.3:
- fork: reason required; incomplete or hash-mismatched
  checkpoint refused with nothing journaled; the fork
  references the SAME weights blob hash as the source
  manifest and copies no bytes; fork rows journaled before
  the descriptor returns; source rows untouched; a fork of
  a fork is lineage-walkable (descendants + ancestors);
- prune gate: no traced window -> REJECT; no reason ->
  REJECT; no journaled bad moment -> REJECT; example
  outside the traced windows -> REJECT; fraction above the
  PROPOSED maximum -> REJECT; configured-order windows
  only -> ESCALATE (a hypothesis, never an admission);
- prune land: an admitted prune produces a NEW corpus hash
  and a pruned_from edge; the verdict row (carrying the new
  hash) is journaled BEFORE the new corpus file exists;
  the original corpus file is byte-identical after the
  flow (hash-checked); REJECT / ESCALATE create no file.
"""
import hashlib
import json
import os

import pytest

from si.training_journal import TRAINING_DB_NAME, TrainingJournalWriter
from si.training_rollback import (
    CheckpointIncomplete, WeightsHashMismatch,
)
from si.training_fork_prune import (
    MAX_PRUNE_FRACTION, VERDICT_ADMIT, VERDICT_ESCALATE,
    VERDICT_REJECT, ForkTargetExists, LandRefused,
    MissingReason, ancestors, corpus_hash_of, descendants,
    fork_run, gate_prune, prune_corpus, read_corpus_ids,
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
    ckpt_dir = str(tmp_path / "ckpts" / f"step{step}")
    write_file(os.path.join(ckpt_dir, "weights.bin"), weights)
    optimizer_blob = None
    if optimizer is not None:
        write_file(os.path.join(ckpt_dir, "optimizer.bin"), optimizer)
        optimizer_blob = sha256_tag(optimizer)
    return w.record_checkpoint_manifest(
        run_id, step, weights_blob=sha256_tag(weights),
        weights_path=ckpt_dir, weights_file="weights.bin",
        weights_bytes=len(weights), optimizer_blob=optimizer_blob,
        optimizer_bytes=len(optimizer) if optimizer else None,
        checkpoint_complete=complete)


def make_corpus(tmp_path, n=20, name="corpus.jsonl"):
    path = str(tmp_path / name)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps(
                {"id": f"ex-{i}", "text": f"example {i}"}) + "\n")
    return path


def journal_trace(w, corpus_path, *, order_source="observed",
                  run_id="run-1", bad=True):
    """Journal a window + a bad moment tracing ex-8/ex-9."""
    window = w.record_window_consumed(
        run_id, 400, 500, corpus_hash=corpus_hash_of(corpus_path),
        example_start=8, example_end=9, order_source=order_source)
    moment = w.record_eval_moment(
        run_id, 500, eval_suite="suite-v1", tier0_score=0.10,
        bad=bad)
    return window, moment


def proposals(w):
    return w.rows("training_proposals")


# -- fork ----------------------------------------------------------------------


def test_fork_reason_required(tmp_path):
    w = make_writer(tmp_path)
    journal_checkpoint(w, tmp_path)
    before = w.total_rows()
    for bad_reason in (None, "", "   "):
        with pytest.raises(MissingReason):
            fork_run(w, "run-1", "run-2", 500, reason=bad_reason)
    assert w.total_rows() == before  # nothing journaled
    assert proposals(w) == []


def test_fork_incomplete_checkpoint_refused(tmp_path):
    w = make_writer(tmp_path)
    journal_checkpoint(w, tmp_path, complete=False)
    with pytest.raises(CheckpointIncomplete):
        fork_run(w, "run-1", "run-2", 500, reason="reimagine")
    assert proposals(w) == []
    assert w.children_of("run-1") == []


def test_fork_hash_mismatch_refused(tmp_path):
    w = make_writer(tmp_path)
    _m = journal_checkpoint(w, tmp_path)
    ckpt_dir = str(tmp_path / "ckpts" / "step500")
    write_file(os.path.join(ckpt_dir, "weights.bin"), b"tampered")
    with pytest.raises(WeightsHashMismatch):
        fork_run(w, "run-1", "run-2", 500, reason="reimagine")
    assert proposals(w) == []


def test_fork_references_same_weights_blob_no_copy(tmp_path):
    w = make_writer(tmp_path)
    manifest = journal_checkpoint(w, tmp_path)
    ckpt_dir = str(tmp_path / "ckpts" / "step500")
    files_before = sorted(os.listdir(ckpt_dir))
    d = fork_run(w, "run-1", "run-2", 500,
                 reason="try the pruned corpus", operator="op")
    # Same blob hash the source manifest references.
    assert d.weights_blob == manifest["weights_blob"]
    assert d.weights_blob == sha256_tag(b"weights-v1")
    # No bytes copied: the checkpoint tree is unchanged.
    assert sorted(os.listdir(ckpt_dir)) == files_before
    # The new run row carries the parent edge by hash.
    runs = [r for r in w.rows("training_runs")
            if r.get("run_id") == "run-2"]
    assert len(runs) == 1
    assert runs[0]["parent_run_id"] == "run-1"
    assert runs[0]["parent_step"] == 500
    assert runs[0]["parent_manifest_hash"] == manifest["weights_blob"]
    # The fork proposal row was journaled before return.
    rows = [r for r in proposals(w) if r.get("kind") == "fork"]
    assert len(rows) == 1
    assert rows[0]["gate_verdict"] == "ADMIT"
    assert rows[0]["parents"] == [manifest["seq"]]
    assert rows[0]["reason"] == "try the pruned corpus"
    assert d.fork_proposal_seq == rows[0]["seq"]
    assert w.verify_chain() is True


def test_fork_source_untouched(tmp_path):
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
    fork_run(w, "run-1", "run-2", 500, reason="branch")
    after = snapshot(w)
    assert after[:len(before)] == before  # append-only
    assert len(after) == len(before) + 2  # proposal + run row


def test_fork_of_fork_lineage_walkable(tmp_path):
    w = make_writer(tmp_path)
    w.record_run_started("run-1", model="mini")
    journal_checkpoint(w, tmp_path, run_id="run-1")
    fork_run(w, "run-1", "run-2", 500, reason="first branch")
    # The fork's own checkpoint, referencing the same blob.
    journal_checkpoint(w, tmp_path, run_id="run-2")
    fork_run(w, "run-2", "run-3", 500, reason="branch the branch")
    fork_run(w, "run-1", "run-4", 500, reason="second child")
    desc = {r["run_id"] for r in descendants(w, "run-1")}
    assert desc == {"run-2", "run-3", "run-4"}
    anc = [r["run_id"] for r in ancestors(w, "run-3")]
    assert anc == ["run-1", "run-2", "run-3"]
    assert {r["run_id"] for r in w.children_of("run-1")} == \
        {"run-2", "run-4"}


def test_fork_existing_run_id_refused(tmp_path):
    w = make_writer(tmp_path)
    journal_checkpoint(w, tmp_path)
    fork_run(w, "run-1", "run-2", 500, reason="first")
    with pytest.raises(ForkTargetExists):
        fork_run(w, "run-1", "run-2", 500, reason="again")


# -- prune gate ------------------------------------------------------------------


def test_prune_no_traced_window_rejected(tmp_path):
    w = make_writer(tmp_path)
    corpus = make_corpus(tmp_path)
    _win, moment = journal_trace(w, corpus)
    out = str(tmp_path / "pruned.jsonl")
    r = prune_corpus(
        w, run_id="run-1", corpus_path=corpus,
        new_corpus_path=out, example_ids=["ex-8"],
        window_seqs=[], bad_moment_seqs=[moment["seq"]],
        reason="bad examples")
    assert r.verdict == VERDICT_REJECT
    assert not os.path.exists(out)


def test_prune_no_reason_rejected(tmp_path):
    w = make_writer(tmp_path)
    corpus = make_corpus(tmp_path)
    window, moment = journal_trace(w, corpus)
    out = str(tmp_path / "pruned.jsonl")
    before = w.total_rows()
    r = prune_corpus(
        w, run_id="run-1", corpus_path=corpus,
        new_corpus_path=out, example_ids=["ex-8"],
        window_seqs=[window["seq"]],
        bad_moment_seqs=[moment["seq"]], reason=None)
    assert r.verdict == VERDICT_REJECT
    assert w.total_rows() == before  # malformed: nothing journaled
    assert not os.path.exists(out)


def test_prune_no_bad_moment_rejected(tmp_path):
    w = make_writer(tmp_path)
    corpus = make_corpus(tmp_path)
    window, _m = journal_trace(w, corpus)
    decision = gate_prune(
        w, run_id="run-1", corpus_path=corpus,
        example_ids=["ex-8"], window_seqs=[window["seq"]],
        bad_moment_seqs=[])
    assert decision.verdict == VERDICT_REJECT


def test_prune_unmarked_moment_rejected(tmp_path):
    w = make_writer(tmp_path)
    corpus = make_corpus(tmp_path)
    window, moment = journal_trace(w, corpus, bad=False)
    decision = gate_prune(
        w, run_id="run-1", corpus_path=corpus,
        example_ids=["ex-8"], window_seqs=[window["seq"]],
        bad_moment_seqs=[moment["seq"]])
    assert decision.verdict == VERDICT_REJECT


def test_prune_outside_traced_window_rejected(tmp_path):
    w = make_writer(tmp_path)
    corpus = make_corpus(tmp_path)
    window, moment = journal_trace(w, corpus)
    out = str(tmp_path / "pruned.jsonl")
    r = prune_corpus(
        w, run_id="run-1", corpus_path=corpus,
        new_corpus_path=out, example_ids=["ex-8", "ex-3"],
        window_seqs=[window["seq"]],
        bad_moment_seqs=[moment["seq"]], reason="traced bad")
    assert r.verdict == VERDICT_REJECT
    assert not os.path.exists(out)


def test_prune_over_max_fraction_rejected(tmp_path):
    w = make_writer(tmp_path)
    corpus = make_corpus(tmp_path, n=20)
    window = w.record_window_consumed(
        "run-1", 0, 500, corpus_hash=corpus_hash_of(corpus),
        example_start=0, example_end=9, order_source="observed")
    moment = w.record_eval_moment("run-1", 500, bad=True)
    ids = [f"ex-{i}" for i in range(6)]  # 6/20 = 0.30 > 0.25
    assert 6 / 20 > MAX_PRUNE_FRACTION
    decision = gate_prune(
        w, run_id="run-1", corpus_path=corpus, example_ids=ids,
        window_seqs=[window["seq"]],
        bad_moment_seqs=[moment["seq"]])
    assert decision.verdict == VERDICT_REJECT
    assert any("PROPOSED" in r for r in decision.reasons)


def test_prune_configured_order_only_escalates(tmp_path):
    w = make_writer(tmp_path)
    corpus = make_corpus(tmp_path)
    window, moment = journal_trace(w, corpus,
                                   order_source="configured")
    out = str(tmp_path / "pruned.jsonl")
    r = prune_corpus(
        w, run_id="run-1", corpus_path=corpus,
        new_corpus_path=out, example_ids=["ex-8", "ex-9"],
        window_seqs=[window["seq"]],
        bad_moment_seqs=[moment["seq"]],
        reason="hypothesis from configured order")
    assert r.verdict == VERDICT_ESCALATE
    assert not os.path.exists(out)  # escalate lands nothing
    verdicts = [x for x in proposals(w)
                if x.get("kind") == "prune_verdict"]
    assert len(verdicts) == 1
    assert verdicts[0]["gate_verdict"] == "ESCALATE"


# -- prune land --------------------------------------------------------------------


def test_admitted_prune_new_hash_edge_and_original_identical(
        tmp_path):
    w = make_writer(tmp_path)
    corpus = make_corpus(tmp_path)
    window, moment = journal_trace(w, corpus)
    out = str(tmp_path / "pruned.jsonl")
    hash_before = corpus_hash_of(corpus)
    with open(corpus, "rb") as f:
        bytes_before = f.read()
    seen = {}

    def hook(edge):
        # At this moment the verdict must already be
        # journaled and the new corpus must NOT yet exist.
        rows = [x for x in proposals(w)
                if x.get("kind") == "prune_verdict"]
        seen["verdict_rows"] = len(rows)
        seen["verdict_hash"] = rows[-1].get("new_corpus_hash")
        seen["file_exists"] = os.path.exists(out)

    r = prune_corpus(
        w, run_id="run-1", corpus_path=corpus,
        new_corpus_path=out, example_ids=["ex-8", "ex-9"],
        window_seqs=[window["seq"]],
        bad_moment_seqs=[moment["seq"]],
        reason="traced to the bad moment at step 500",
        operator="op", on_before_land=hook)
    assert r.verdict == VERDICT_ADMIT
    # Journal-before-landing, proven by the hook.
    assert seen == {"verdict_rows": 1,
                    "verdict_hash": r.new_corpus_hash,
                    "file_exists": False}
    # A NEW corpus hash, different from the original's.
    assert r.new_corpus_hash != hash_before
    assert r.new_corpus_hash == corpus_hash_of(out)
    assert r.original_corpus_hash == hash_before
    # The pruned_from edge walks the whole chain.
    edge = r.pruned_from
    assert edge["relation"] == "pruned_from"
    assert edge["new_corpus_hash"] == r.new_corpus_hash
    assert edge["original_corpus_hash"] == hash_before
    assert edge["window_seqs"] == [window["seq"]]
    assert edge["bad_moment_seqs"] == [moment["seq"]]
    assert edge["prune_verdict_seq"] == r.verdict_seq
    # The new corpus holds exactly the kept examples.
    assert read_corpus_ids(out) == [
        f"ex-{i}" for i in range(20) if i not in (8, 9)]
    # The original corpus file is byte-identical after.
    with open(corpus, "rb") as f:
        assert f.read() == bytes_before
    assert corpus_hash_of(corpus) == hash_before
    assert w.verify_chain() is True


def test_admitted_prune_journals_proposal_before_verdict(tmp_path):
    w = make_writer(tmp_path)
    corpus = make_corpus(tmp_path)
    window, moment = journal_trace(w, corpus)
    r = prune_corpus(
        w, run_id="run-1", corpus_path=corpus,
        new_corpus_path=str(tmp_path / "pruned.jsonl"),
        example_ids=["ex-8"], window_seqs=[window["seq"]],
        bad_moment_seqs=[moment["seq"]], reason="bad window")
    rows = proposals(w)
    kinds = [x["kind"] for x in rows]
    assert kinds == ["prune_proposed", "prune_verdict"]
    assert rows[0]["seq"] == r.proposed_seq
    assert rows[0]["example_ids"] == ["ex-8"]
    assert rows[0]["reason"] == "bad window"
    assert rows[1]["parents"] == [r.proposed_seq]
    assert rows[1]["relation"] == "pruned_from"
    assert rows[1]["gate_verdict"] == "ADMIT"
    assert rows[1]["new_corpus_hash"] == r.new_corpus_hash


def test_prune_land_refuses_existing_or_original_path(tmp_path):
    w = make_writer(tmp_path)
    corpus = make_corpus(tmp_path)
    window, moment = journal_trace(w, corpus)
    with pytest.raises(LandRefused):
        prune_corpus(
            w, run_id="run-1", corpus_path=corpus,
            new_corpus_path=corpus, example_ids=["ex-8"],
            window_seqs=[window["seq"]],
            bad_moment_seqs=[moment["seq"]], reason="bad")
    existing = write_file(str(tmp_path / "taken.jsonl"), b"{}\n")
    with pytest.raises(LandRefused):
        prune_corpus(
            w, run_id="run-1", corpus_path=corpus,
            new_corpus_path=existing, example_ids=["ex-8"],
            window_seqs=[window["seq"]],
            bad_moment_seqs=[moment["seq"]], reason="bad")


def test_index_based_ids_are_the_minimum_viable_form(tmp_path):
    w = make_writer(tmp_path)
    path = str(tmp_path / "noids.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for i in range(10):
            f.write(json.dumps({"text": f"row {i}"}) + "\n")
    assert read_corpus_ids(path) == [str(i) for i in range(10)]
    window = w.record_window_consumed(
        "run-1", 0, 100, corpus_hash=corpus_hash_of(path),
        example_start=3, example_end=3, order_source="observed")
    moment = w.record_eval_moment("run-1", 100, bad=True)
    r = prune_corpus(
        w, run_id="run-1", corpus_path=path,
        new_corpus_path=str(tmp_path / "noids-pruned.jsonl"),
        example_ids=["3"], window_seqs=[window["seq"]],
        bad_moment_seqs=[moment["seq"]], reason="row 3 is bad")
    assert r.verdict == VERDICT_ADMIT
    assert len(read_corpus_ids(r.new_corpus_path)) == 9
