"""Rollback: the head moves back, history and blobs stay as evidence.

Two admitted demo-mind cycles (head d0002), then rollback --to base:
head returns to base, popped deltas carry "rolled_back" status in the
journal, their blobs still exist on disk, the rollback is its own
journaled self-edit with cause, and probes re-run ok.
"""
import numpy as np
import os

from si.bundle import init_bundle, load_bundle, sha256_hex
from si.journal import open_journal
from si.loop import rollback, run_cycle
from si.minds import DemoMind


def test_rollback_restores_prior_head(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "rb.si"), name="rollback-tests")
    mind = DemoMind(bundle_path=bundle_dir)
    r1 = run_cycle(bundle_dir, mind)
    r2 = run_cycle(bundle_dir, mind)
    assert (r1["delta_id"], r2["delta_id"]) == ("d0001", "d0002")
    assert load_bundle(bundle_dir)["head"] == "d0002"
    blob_hexes = [d["blob"].split(":", 1)[1]
                  for d in load_bundle(bundle_dir)["deltas"]]
    assert len(blob_hexes) == 2

    out = rollback(bundle_dir, to="base",
                   reason="operator rollback test: unwind two demo cycles",
                   mind=DemoMind(bundle_path=bundle_dir))
    assert out["head"] == "base"
    assert out["probes_ok"] is True

    manifest = load_bundle(bundle_dir)
    assert manifest["head"] == "base"
    # history is append-only: the delta records are still there
    assert [d["id"] for d in manifest["deltas"]] == ["d0001", "d0002"]

    journal = open_journal(bundle_dir, manifest)
    try:
        deltas = {d["id"]: d for d in journal.backend.find("deltas")}
        assert deltas["d0001"]["status"] == "rolled_back"
        assert deltas["d0002"]["status"] == "rolled_back"

        rbs = journal.backend.find("self_edits", filter={"kind": "rollback"})
        assert len(rbs) == 1
        rb = rbs[0]
        assert rb["edit_id"] == out["edit_id"]
        assert rb["intent"]["cause"] == "operator"
        assert rb["intent"]["from"] == "d0002"
        assert rb["intent"]["target"] == "base"
        assert rb["result"] == "materialized"

        assert journal.verify_chain() is True
    finally:
        journal.close()

    # evidence retained: both blobs still on disk, content-addressed
    for hexpart in blob_hexes:
        p = os.path.join(bundle_dir, "blobs", "sha256", hexpart + ".safetensors")
        assert os.path.isfile(p)
        with open(p, "rb") as f:
            assert sha256_hex(f.read()) == hexpart

    # the mind is whole at the restored head: identity probes still pass
    whole = DemoMind(bundle_path=bundle_dir)
    for _, x, expected in whole.identity_probes():
        drift = float(np.max(np.abs(whole.forward(x, []) - expected)))
        assert drift <= 0.25
