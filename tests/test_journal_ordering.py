"""Journal-before-materialize ordering + rejected intents journaled cleanly.

The loop contract: on ADMIT the intent and gate verdict are journaled
BEFORE the blob lands; a rejected intent is journaled as a first-class
event and materializes nothing.
"""
import os

from si.bundle import init_bundle, load_bundle
from si.gate import Intent
from si.journal import open_journal
from si.loop import run_cycle
from si.minds import DemoMind, MockMind


def test_journal_before_materialize(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "ordering.si"), name="ordering")
    res = run_cycle(bundle_dir, DemoMind(bundle_path=bundle_dir))
    assert res["verdict"] == "ADMIT" and res["delta_id"] == "d0001"

    manifest = load_bundle(bundle_dir)
    journal = open_journal(bundle_dir, manifest)
    try:
        edits = journal.backend.find("self_edits", filter={"edit_id": res["edit_id"]})
        assert len(edits) == 1
        edit_seq = edits[0]["seq"]
        assert edits[0]["gate_verdict"] == "ADMIT"

        cps = journal.backend.find(
            "checkpoints", filter={"note": "post-materialize d0001"})
        assert len(cps) == 1
        mat_seq = cps[0]["seq"]

        # The journal knew about the admitted self-edit before the blob landed.
        assert edit_seq < mat_seq

        # ...and the verify checkpoint closed the cycle after materialize.
        vcp = journal.backend.find(
            "checkpoints", filter={"note": "post-verify d0001 VERIFIED"})
        assert len(vcp) == 1 and vcp[0]["seq"] > mat_seq
    finally:
        journal.close()


def test_rejected_intent_journaled_no_materialize(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "rejected.si"), name="rejected")
    blobs = os.path.join(bundle_dir, "blobs", "sha256")
    n_blobs_before = len(os.listdir(blobs))

    # Malformed intent: a mind that can't say why isn't ready to change.
    script = [{"intent": Intent(
        kind="delta", reason="",
        target_modules=["demo_linear"], rank=2, alpha=4,
        parents=["base"], source_seqs=[1],
        training_signal_id="mock-signal"),
        "candidate_norm": 0.01}]
    res = run_cycle(bundle_dir, MockMind(script=script))
    assert res["verdict"] == "REJECT"
    assert res["delta_id"] is None

    # No materialize: head unmoved, no delta record, no new blob.
    manifest = load_bundle(bundle_dir)
    assert manifest["head"] == "base"
    assert manifest["deltas"] == []
    assert len(os.listdir(blobs)) == n_blobs_before

    # But the rejection itself is journaled, chain intact.
    journal = open_journal(bundle_dir, manifest)
    try:
        edits = journal.backend.find("self_edits")
        assert len(edits) == 1
        e = edits[0]
        assert e["gate_verdict"] == "REJECT"
        assert e["result"] == "rejected"
        assert journal.backend.find("deltas") == []
        assert journal.verify_chain() is True
    finally:
        journal.close()
