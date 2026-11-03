"""SI-85 (SI-83 section 9 slice 2): training journal writer.

Miniature/synthetic only. Covers SI-83 section 9.2 (chain over
every section 2.5 kind; wrong-name writers refuse; broken chain
refuses open) plus section 7.3/7.4 (absent/derived names,
unreportable backend, mismatch stating both names, genesis,
fork lineage both ways, honest unknowns, truncation via hash).
"""
import hashlib

import pytest

from si.journal import EmbeddedBackend
from si.training_journal import (
    TRAINING_COLLECTIONS, TRAINING_DB_NAME, ChainVerificationFailed,
    DatabaseNameRefused, EmbeddedTrainingBackend, HandshakeFailed,
    TrainingJournalWriter, ports_for,
)

HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64


def make_writer(tmp_path, name=TRAINING_DB_NAME, sub="journal"):
    return TrainingJournalWriter.for_embedded(
        str(tmp_path / sub), database_name=name)


def total_rows(writer):
    return sum(writer.backend.count(c) for c in TRAINING_COLLECTIONS)


def test_chain_over_every_event_kind(tmp_path):
    w = make_writer(tmp_path)
    w.record_run_started("run-1", corpus_hash=HASH_A,
                         hyperparams={"lr": 1e-5})
    ck = w.record_checkpoint_manifest(
        "run-1", 500, epoch=0.5, weights_blob=HASH_B,
        weights_path="/ckpts/step500", weights_bytes=123,
        corpus_hash=HASH_A)
    w.record_eval_moment("run-1", 500, eval_suite="suite-v1",
                         tier0_score=0.5, loss=0.25)
    w.record_window_consumed("run-1", 1, 500, corpus_hash=HASH_A,
                             example_start=0, example_end=99,
                             example_ids_hash=HASH_A,
                             order_source="observed")
    w.record_prune_proposed("run-1", reason="bad window",
                            step_from=1, step_to=500,
                            example_ids=[3, 4])
    w.record_prune_verdict("run-1", verdict="ADMIT",
                           parents=[ck["seq"]], new_corpus_hash=HASH_B)
    w.record_rollback("run-1", target_step=500, reason="regressed",
                      verdict="ADMIT", parents=[ck["seq"]])
    w.record_fork("run-1", "run-2", parent_step=500,
                  parent_manifest_hash=HASH_B, parents=[ck["seq"]],
                  reason="reimagine")
    w.record_retention_decision("run-1", step=500, keep=True,
                                verdict="ADMIT", reason="fork parent")
    w.record_watcher_gap("run-1", step_from=501, step_to=600,
                         covered_steps=[550], reason="watcher died")
    assert w.verify_chain() is True
    kinds = set()
    for coll in TRAINING_COLLECTIONS:
        kinds |= {r["kind"] for r in w.rows(coll)}
    assert {"run_started", "checkpoint_manifest", "eval_moment",
            "window_consumed", "prune_proposed", "prune_verdict",
            "rollback", "fork", "retention_decision",
            "watcher_gap"} <= kinds
    # No training row landed in the SI-0 collections.
    for coll in ("self_edits", "deltas", "checkpoints", "provenance",
                 "experiences"):
        assert w.backend.count(coll) == 0
    all_rows = sorted(
        (r for c in TRAINING_COLLECTIONS for r in w.rows(c)),
        key=lambda r: r["seq"])
    assert [r["seq"] for r in all_rows] == list(
        range(1, len(all_rows) + 1))
    for prev, cur in zip(all_rows, all_rows[1:]):
        assert cur["prev_hash"] == prev["entry_hash"]


@pytest.mark.parametrize("bad_name", [
    "shop", "stealth_shop", "si_glimmer", "si_run-1", ""])
def test_wrong_or_empty_name_refuses_before_any_write(tmp_path, bad_name):
    with pytest.raises(DatabaseNameRefused):
        make_writer(tmp_path, name=bad_name)
    # Zero rows written (fresh backend over the same directory).
    backend = EmbeddedTrainingBackend(
        str(tmp_path / "journal"), TRAINING_DB_NAME)
    assert sum(backend.count(c) for c in TRAINING_COLLECTIONS) == 0


def test_absent_config_refuses(tmp_path):
    backend = EmbeddedTrainingBackend(
        str(tmp_path / "j"), TRAINING_DB_NAME)
    with pytest.raises(DatabaseNameRefused):
        TrainingJournalWriter(backend, None)
    with pytest.raises(DatabaseNameRefused):
        TrainingJournalWriter(backend, "")
    assert sum(backend.count(c) for c in TRAINING_COLLECTIONS) == 0


def test_backend_that_cannot_report_name_fails_closed(tmp_path):
    raw = EmbeddedBackend(str(tmp_path / "raw"))
    with pytest.raises(HandshakeFailed) as exc:
        TrainingJournalWriter(raw, TRAINING_DB_NAME)
    assert TRAINING_DB_NAME in str(exc.value)
    assert sum(raw.count(c) for c in TRAINING_COLLECTIONS) == 0


def test_readback_mismatch_states_both_names(tmp_path):
    # Storage directory created under a different (seed) name.
    backend = EmbeddedTrainingBackend(
        str(tmp_path / "j"), "stealth_shop")
    with pytest.raises(HandshakeFailed) as exc:
        TrainingJournalWriter(backend, TRAINING_DB_NAME)
    msg = str(exc.value)
    assert TRAINING_DB_NAME in msg and "stealth_shop" in msg
    assert sum(backend.count(c) for c in TRAINING_COLLECTIONS) == 0


def test_nedb_style_dbname_handshake(tmp_path):
    class FakeNedbBackend(EmbeddedTrainingBackend):
        pass

    backend = FakeNedbBackend(str(tmp_path / "j"), TRAINING_DB_NAME)
    backend.dbname = TRAINING_DB_NAME  # as NedbJournalBackend exposes
    w = TrainingJournalWriter(backend, TRAINING_DB_NAME)
    assert w.verified_database_name == TRAINING_DB_NAME
    backend2 = EmbeddedTrainingBackend(
        str(tmp_path / "j2"), TRAINING_DB_NAME)
    backend2.dbname = "stealth_shop"
    with pytest.raises(HandshakeFailed):
        TrainingJournalWriter(backend2, TRAINING_DB_NAME)


def test_genesis_row_records_name_ports_and_timestamp(tmp_path):
    w = make_writer(tmp_path)
    rows = sorted(
        (r for c in TRAINING_COLLECTIONS for r in w.rows(c)),
        key=lambda r: r["seq"])
    genesis = rows[0]
    pg_port, http_port = ports_for(TRAINING_DB_NAME)
    assert genesis["kind"] == "genesis"
    assert genesis["database_name"] == TRAINING_DB_NAME
    assert genesis["pg_port"] == pg_port
    assert genesis["http_port"] == http_port
    assert genesis["verified_at"]
    assert genesis["seq"] == 1 and genesis["prev_hash"] == "genesis"
    w2 = TrainingJournalWriter.for_embedded(
        str(tmp_path / "journal"), database_name=TRAINING_DB_NAME)
    assert total_rows(w2) == total_rows(w)
    assert w2.verify_chain() is True


def test_tampered_chain_refuses_open(tmp_path):
    w = make_writer(tmp_path)
    w.record_run_started("run-1", corpus_hash=HASH_A)
    w.record_eval_moment("run-1", 10, eval_suite="s", loss=1.0)
    # Tamper with a chain-protected field, hash left stale.
    rows = w.backend.find("training_eval_moments")
    row = dict(rows[0])
    row_id = row.pop("_id")
    row["loss"] = 0.0
    w.backend.update("training_eval_moments", row_id, row)
    assert w.verify_chain() is False
    with pytest.raises(ChainVerificationFailed):
        TrainingJournalWriter.for_embedded(
            str(tmp_path / "journal"), database_name=TRAINING_DB_NAME)


def test_fork_lineage_walkable_both_ways(tmp_path):
    w = make_writer(tmp_path)
    w.record_run_started("run-1", corpus_hash=HASH_A)
    ck = w.record_checkpoint_manifest(
        "run-1", 500, weights_blob=HASH_B, weights_bytes=10,
        corpus_hash=HASH_A)
    out = w.record_fork("run-1", "run-2", parent_step=500,
                        parent_manifest_hash=HASH_B,
                        parents=[ck["seq"]], reason="pruned corpus")
    prop = out["proposal"]
    assert prop["kind"] == "fork" and prop["relation"] == "forked_from"
    assert prop["parents"] == [ck["seq"]]
    new_run = out["run"]
    assert new_run["parent_run_id"] == "run-1"
    assert new_run["parent_step"] == 500
    assert new_run["parent_manifest_hash"] == HASH_B
    lineage = w.lineage_to_root("run-2")
    assert [r["run_id"] for r in lineage] == ["run-2", "run-1"]
    assert [r["run_id"] for r in w.children_of("run-1")] == ["run-2"]
    assert w.verify_chain() is True


def test_honest_unknowns_and_null_scores_preserved(tmp_path):
    w = make_writer(tmp_path)
    run = w.record_run_started("run-1")
    assert run["trainer_commit"] == "unknown"
    assert run["data_order_seed"] is None
    ck = w.record_checkpoint_manifest("run-1", 1, weights_blob=HASH_B)
    assert ck["trainer_commit"] == "unknown"
    assert ck["data_order_seed"] is None
    ev = w.record_eval_moment("run-1", 1, eval_suite="s")
    assert ev["eval_suite_commit"] == "unknown"
    assert ev["tier0_score"] is None
    assert ev["discrimination_score"] is None
    assert ev["temporal_score"] is None and ev["loss"] is None
    ev2 = w.record_eval_moment("run-1", 2, eval_suite="s",
                               tier0_score=0.3333)
    assert ev2["tier0_score"] == 0.3333
    assert ev2["discrimination_score"] is None


def test_eval_sample_truncation_detectable_via_hash(tmp_path):
    w = make_writer(tmp_path)
    full = "x" * 5000
    ev = w.record_eval_moment(
        "run-1", 1, eval_suite="s",
        sample_generations=[{"prompt": "p", "generation": full},
                            {"prompt": "q", "generation": "short"}])
    gen, short = ev["sample_generations"]
    assert len(gen["generation"]) == 4096
    assert gen["generation_truncated"] is True
    assert gen["generation_hash"] == \
        "sha256:" + hashlib.sha256(full.encode()).hexdigest()
    assert gen["generation"] != full  # truncation is visible, not silent
    assert short["generation"] == "short"
    assert short["generation_truncated"] is False
    assert short["generation_hash"] == \
        "sha256:" + hashlib.sha256(b"short").hexdigest()


def test_weights_bytes_and_reasonless_prune_refused(tmp_path):
    w = make_writer(tmp_path)
    with pytest.raises(TypeError):
        w.record_checkpoint_manifest("run-1", 1, weights_blob=b"raw-bytes")
    with pytest.raises(ValueError):
        w.record_prune_proposed("run-1", reason=None)
    with pytest.raises(ValueError):
        w.record_rollback("run-1", target_step=1, parents=[])
