"""End-to-end self-write cycle on the REAL DemoMind (numpy, CPU).

One admitted cycle: verdict ADMIT, delta d0001, head advanced, VERIFIED,
blob content-addressed under blobs/sha256/, manifest<->journal tip anchor
consistent after load, and the `si why d0001` data present in the journal
(self_edit + provenance edges).
"""
import os

from si.bundle import init_bundle, load_bundle, sha256_hex
from si.journal import open_journal
from si.loop import run_cycle
from si.minds import DemoMind


def test_full_self_write_cycle(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "loop.si"), name="loop-e2e")
    res = run_cycle(bundle_dir, DemoMind(bundle_path=bundle_dir))

    assert res["verdict"] == "ADMIT"
    assert res["delta_id"] == "d0001"
    assert res["edit_id"] == "se_0001"
    assert res["head"] == "d0001"
    assert res["verified"] is True
    assert res["rolled_back"] is False

    manifest = load_bundle(bundle_dir)
    assert manifest["head"] == "d0001"
    assert len(manifest["deltas"]) == 1
    d = manifest["deltas"][0]
    assert d["id"] == "d0001" and d["parent"] == "base"
    assert d["self_edit_id"] == "se_0001"

    # The blob exists and its content hash matches the manifest reference.
    scheme, hexpart = d["blob"].split(":", 1)
    assert scheme == "sha256"
    blob_path = os.path.join(bundle_dir, "blobs", "sha256",
                             hexpart + ".safetensors")
    assert os.path.isfile(blob_path)
    with open(blob_path, "rb") as f:
        assert sha256_hex(f.read()) == hexpart

    journal = open_journal(bundle_dir, manifest)
    try:
        # manifest <-> journal tip anchor consistent after load
        tip_seq, tip_hash = journal.tip()
        assert manifest["journal"]["last_seq"] == tip_seq
        assert manifest["journal"]["last_seq_hash"] == tip_hash
        assert journal.verify_chain() is True

        # `si why d0001` reads exactly these: the self-edit + provenance edges
        edits = journal.backend.find("self_edits", filter={"edit_id": "se_0001"})
        assert len(edits) == 1
        e = edits[0]
        assert e["kind"] == "delta"
        assert e["gate_verdict"] == "ADMIT"
        assert e["result"] == "materialized"
        assert e["delta_id"] == "d0001"
        assert e["reason"], "the journal must record why the delta exists"
        checks = (e.get("gate_report") or {}).get("checks", [])
        assert checks and all(c["passed"] for c in checks)

        prov = journal.backend.find("provenance", filter={"child": "d0001"})
        assert prov, "expected a provenance edge for d0001"
        assert prov[0]["relation"] == "learned_from"
        assert "base" in prov[0]["parents"]
    finally:
        journal.close()
