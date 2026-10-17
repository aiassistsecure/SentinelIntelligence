"""Real dogfood: the journal on the actual NEDB engine over pgwire.

Each test spawns the real nedbd-v2 daemon via autostart (a few seconds per
test). Bundle names are distinct per test so the derived dbnames/ports
never collide.
"""
import os

import pytest

from si.backends.nedb import NedbJournalBackend
from si.bundle import TamperError, init_bundle, load_bundle
from si.journal import open_journal
from si.loop import rollback, run_cycle
from si.minds import DemoMind


def _init(tmp_path, name):
    return init_bundle(str(tmp_path / (name + ".si")), name=name,
                       journal_backend="nedb")


def test_nedb_init_records_backend(tmp_path):
    manifest = load_bundle(_init(tmp_path, "nedb-init"))
    assert manifest["journal"]["backend"] == "nedb"
    assert manifest["head"] == "base"


def test_nedb_full_demo_cycle(tmp_path):
    bundle_dir = _init(tmp_path, "nedb-cycle")
    res = run_cycle(bundle_dir, DemoMind(bundle_path=bundle_dir))
    assert res["verdict"] == "ADMIT"
    assert res["delta_id"] == "d0001"
    assert res["head"] == "d0001"
    assert res["verified"] is True

    manifest = load_bundle(bundle_dir)
    journal = open_journal(bundle_dir, manifest)
    try:
        assert journal.verify_chain() is True
        tip_seq, tip_hash = journal.tip()
        assert manifest["journal"]["last_seq"] == tip_seq
        assert manifest["journal"]["last_seq_hash"] == tip_hash
    finally:
        journal.close()


def test_nedb_rollback_restores_head(tmp_path):
    bundle_dir = _init(tmp_path, "nedb-rollback")
    assert run_cycle(bundle_dir, DemoMind(bundle_path=bundle_dir))["delta_id"] == "d0001"
    assert run_cycle(bundle_dir, DemoMind(bundle_path=bundle_dir))["delta_id"] == "d0002"
    out = rollback(bundle_dir, to="base", reason="nedb rollback test",
                   mind=DemoMind(bundle_path=bundle_dir))
    assert out["head"] == "base"
    assert out["probes_ok"] is True

    manifest = load_bundle(bundle_dir)
    assert manifest["head"] == "base"
    journal = open_journal(bundle_dir, manifest)
    try:
        statuses = {d["id"]: d["status"] for d in journal.backend.find("deltas")}
        assert statuses == {"d0001": "rolled_back", "d0002": "rolled_back"}
        assert journal.verify_chain() is True
    finally:
        journal.close()


def test_nedb_journal_tamper_fails_closed(tmp_path):
    bundle_dir = _init(tmp_path, "nedb-tamper")
    load_bundle(bundle_dir)  # clean before the tamper
    backend = NedbJournalBackend(os.path.join(bundle_dir, "journal"))
    try:
        rows = backend.find("checkpoints")
        assert rows, "expected the genesis checkpoint"
        row = dict(rows[0])
        row["note"] = "genesis (edited by an attacker)"
        backend.update("checkpoints", row["_id"], row)
    finally:
        backend.close()
    with pytest.raises(TamperError):
        load_bundle(bundle_dir)
