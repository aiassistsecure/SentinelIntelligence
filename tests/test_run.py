"""Tests for the autonomous multi-cycle runner (si/run.py).

Serial runs through the shared gate: all-admit completion, halt on the
first REJECT or ESCALATE (with the journal evidence written by run_cycle),
summary bookkeeping, and one real dogfood run on the nedb backend.
"""
import pytest

from si.bundle import init_bundle, load_bundle
from si.gate import Intent
from si.journal import open_journal
from si.minds import MockMind, DemoMind
from si.run import run_cycles


def _good_delta(candidate_norm=0.01):
    return {
        "intent": Intent(
            kind="delta",
            reason="test delta: tiny scripted adapter for the multi-cycle runner.",
            target_modules=["demo_linear"],
            rank=2,
            alpha=4,
            parents=["base"],
            source_seqs=[1],
            training_signal_id="mock-signal"),
        "candidate_norm": candidate_norm,
    }


def test_run_completes_n_admitted_cycles(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "run.si"), name="run-n",
                             journal_backend="embedded")
    mind = MockMind(script=[_good_delta(), _good_delta(), _good_delta()])
    summary = run_cycles(bundle_dir, mind, 3)

    assert summary["cycles_requested"] == 3
    assert summary["cycles_completed"] == 3
    assert summary["admitted"] == 3
    assert summary["rejected"] == 0
    assert summary["escalated"] == 0
    assert summary["halted"] is False
    assert summary["halt_verdict"] is None
    assert summary["halt_edit_id"] is None
    assert summary["head"] == "d0003"
    assert summary["delta_stack_depth"] == 3
    assert len(summary["edit_ids"]) == 3
    assert summary["edit_ids"] == ["se_0001", "se_0002", "se_0003"]


def test_run_halts_on_reject(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "run-reject.si"), name="run-reject",
                             journal_backend="embedded")
    # A delta with an empty reason is malformed: the gate rejects it.
    script = [_good_delta(),
              {"intent": Intent(kind="delta", reason="",
                                target_modules=["demo_linear"], rank=2, alpha=4,
                                parents=["base"], source_seqs=[1],
                                training_signal_id="mock-signal"),
               "candidate_norm": 0.01}]
    mind = MockMind(script=script)
    summary = run_cycles(bundle_dir, mind, 3)

    assert summary["cycles_requested"] == 3
    assert summary["cycles_completed"] == 2
    assert summary["admitted"] == 1
    assert summary["rejected"] == 1
    assert summary["escalated"] == 0
    assert summary["halted"] is True
    assert summary["halt_verdict"] == "REJECT"
    assert summary["halt_edit_id"] == summary["edit_ids"][1] == "se_0002"
    # The rejected cycle materialized nothing: head is still the first delta.
    assert summary["head"] == "d0001"

    # The rejection was journaled by run_cycle (no double-journal from the runner).
    manifest = load_bundle(bundle_dir)
    journal = open_journal(bundle_dir, manifest)
    try:
        rows = journal.backend.find("self_edits",
                                    filter={"edit_id": summary["halt_edit_id"]})
        assert len(rows) == 1
        assert rows[0]["gate_verdict"] == "REJECT"
        assert rows[0]["result"] == "rejected"
    finally:
        journal.close()


def test_run_halts_on_escalate(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "run-escalate.si"), name="run-escalate",
                             journal_backend="embedded")
    # constitution intents always escalate in SI-0.
    script = [_good_delta(),
              {"intent": Intent(kind="constitution",
                                reason="propose amending directive 4"),
               "candidate_norm": 0.01}]
    mind = MockMind(script=script)
    summary = run_cycles(bundle_dir, mind, 3)

    assert summary["cycles_completed"] == 2
    assert summary["admitted"] == 1
    assert summary["rejected"] == 0
    assert summary["escalated"] == 1
    assert summary["halted"] is True
    assert summary["halt_verdict"] == "ESCALATE"
    assert summary["halt_edit_id"] == summary["edit_ids"][1] == "se_0002"
    # Nothing materialized for the escalated intent.
    assert summary["head"] == "d0001"
    assert summary["delta_stack_depth"] == 1

    manifest = load_bundle(bundle_dir)
    journal = open_journal(bundle_dir, manifest)
    try:
        rows = journal.backend.find("self_edits",
                                    filter={"edit_id": summary["halt_edit_id"]})
        assert len(rows) == 1
        assert rows[0]["gate_verdict"] == "ESCALATE"
        assert rows[0]["result"] == "escalated"
    finally:
        journal.close()


def test_run_summary_correctness(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "run-summary.si"), name="run-summary",
                             journal_backend="embedded")
    mind = MockMind(script=[_good_delta(), _good_delta()])
    summary = run_cycles(bundle_dir, mind, 2)

    assert summary["cycles_completed"] == 2
    manifest = load_bundle(bundle_dir)
    deltas_up_to_head = []
    for d in manifest["deltas"]:
        deltas_up_to_head.append(d["id"])
        if d["id"] == manifest["head"]:
            break
    assert summary["delta_stack_depth"] == len(deltas_up_to_head) == 2


@pytest.mark.slow
def test_run_nedb_multicycle(tmp_path):
    """Dogfood: two real DemoMind cycles against the real NEDB journal."""
    bundle_dir = init_bundle(str(tmp_path / "run-nedb.si"), name="run-nedb",
                             journal_backend="nedb")
    summary = run_cycles(bundle_dir, DemoMind(bundle_path=bundle_dir), 2)

    assert summary["cycles_completed"] == 2
    assert summary["admitted"] == 2
    assert summary["halted"] is False
    assert summary["head"] == "d0002"

    manifest = load_bundle(bundle_dir)
    journal = open_journal(bundle_dir, manifest)
    try:
        assert journal.verify_chain() is True
        edits = journal.backend.find("self_edits")
        assert len([e for e in edits if e.get("result") == "materialized"]) == 2
    finally:
        journal.close()
