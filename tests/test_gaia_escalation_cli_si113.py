"""SI-113 (ESC-LIVE-1; SI-111 §4.2): the CLI approval
trigger — transport, never decision.

Coverage:

- ``list`` reads exactly the pending escalations from the
  fold (folded ``escalated``, unresolved), with the
  journaled facts an operator decides on;
- ``show`` reads one folded intent by journal seq, and an
  unknown seq is a clean refusal (exit 1, nothing read
  into existence);
- ``resolve`` records the operator's act through the
  EXISTING :func:`si.gaia.resolution.resolve_escalation` —
  ``admit-after-review`` journals one
  ``escalation_resolution`` row (operator + basis recorded)
  and folds the intent to ``queued``; the intent leaves the
  pending list;
- ``reject`` annotates the intent (it stays folded
  ``escalated``) and also leaves the pending list;
- a second resolution, an invalid resolution value, and an
  anonymous operator are refused by the act and journal
  NOTHING — the trigger adds no verification of its own and
  no write path of its own;
- the ``si`` CLI wiring parses and runs the same commands.
"""
import json

import pytest

from si.bundle import (
    _anchor_checkpoint, _write_manifest, init_bundle, load_bundle)
from si.cli import build_parser, cmd_gaia_escalation
from si.cli import main as cli_main
from si.gaia import escalation_cli
from si.gaia.context import ToolContext
from si.gaia.corpus import corpus_policy
from si.gaia.drain import admit_tool_intent
from si.gaia.queue import fold_queue, record_queue_batch
from si.gaia.shadow import record_task_fixture
from si.gate import Gate, Intent
from si.journal import open_journal
from si.tiny_seed import seed_tiny_bundle


ESCALATE_WRITE = {"read": "admit", "write": "escalate",
                  "execute": "admit", "external": "escalate"}
CONTENT = "written after operator review\n"


@pytest.fixture()
def mini(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "mini.si"),
                             name="gaia-si113",
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    work = tmp_path / "work"
    work.mkdir()
    (work / "notes").mkdir()
    (work / "notes" / "plan.md").write_text("# plan\nstep one\n")
    return bundle_dir, str(work)


def _open(bundle_dir):
    manifest = load_bundle(bundle_dir)
    return manifest, open_journal(bundle_dir, manifest)


def _escalated(mini):
    """One ESCALATEd fs_write in a fresh miniature bundle.
    Returns (bundle_dir, edit_id, intent_seq). The fixture
    re-anchors the manifest checkpoint after journaling —
    mirroring the tail of every production journaling path
    (seam ingest, seam submission, run_task) — so the bundle
    stands the way a live bundle stands when the trigger
    opens it: fully verified, anchor at the tip."""
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture = record_task_fixture(journal, "SI-113 scenario",
                                      work)
        policy = corpus_policy(
            work, {"tool_class_policy": ESCALATE_WRITE})
        gate = Gate(policy)
        ctx = ToolContext(tool_work_root=work,
                          tool_task_seq=fixture["seq"],
                          tool_live_execution=True)
        batch = record_queue_batch(journal, fixture["seq"],
                                   fixture["ref"], declared_size=1)
        intent = Intent(
            kind="tool", tool="fs_write",
            args={"path": "notes/new.md", "content": CONTENT},
            reason="Carry out the scripted SI-113 task step",
            parents=[fixture["ref"]],
            source_seqs=[fixture["seq"]])
        out = admit_tool_intent(journal, gate, ctx, intent,
                                batch_seq=batch["seq"])
        assert out["verdict"] == "ESCALATE"
        rows = journal.backend.find(
            "self_edits", filter={"edit_id": out["edit_id"]},
            limit=1)
        updated = _anchor_checkpoint(
            journal, manifest, manifest["head"],
            note="SI-113 test fixture: anchor after journaling "
                 "the escalation (mirrors the ingest tail)")
        _write_manifest(bundle_dir, updated)
        return bundle_dir, out["edit_id"], rows[0]["seq"]
    finally:
        journal.close()


def _resolution_rows(bundle_dir):
    journal = open_journal(bundle_dir)
    try:
        return journal.backend.find(
            "experiences",
            filter={"kind": "escalation_resolution"}) or []
    finally:
        journal.close()


def _tip(bundle_dir):
    journal = open_journal(bundle_dir)
    try:
        return journal.tip()
    finally:
        journal.close()


# -- list / show: the fold, read ------------------------------------------------

def test_list_shows_exactly_the_pending_escalation(mini):
    bundle_dir, edit_id, seq = _escalated(mini)
    pending = escalation_cli.list_escalations(bundle_dir)
    assert len(pending) == 1
    view = pending[0]
    assert view["intent_seq"] == seq
    assert view["edit_id"] == edit_id
    assert view["tool"] == "fs_write"
    assert view["args"] == {"path": "notes/new.md",
                            "content": CONTENT}
    assert view["verdict"] == "ESCALATE"
    assert view["state"] == "escalated"
    assert view["resolution"] is None


def test_show_reads_one_intent_and_unknown_seq_refuses(mini, capsys):
    bundle_dir, edit_id, seq = _escalated(mini)
    view = escalation_cli.read_escalation_at(bundle_dir, seq)
    assert view["edit_id"] == edit_id
    assert view["state"] == "escalated"
    assert escalation_cli.read_escalation_at(
        bundle_dir, seq + 9999) is None
    rc = escalation_cli.main(
        ["show", bundle_dir, str(seq + 9999)])
    assert rc == 1
    assert "no folded intent" in capsys.readouterr().err


# -- resolve: the act, through the trigger -------------------------------------

def test_resolve_admit_after_review_journals_and_queues(mini):
    bundle_dir, edit_id, seq = _escalated(mini)
    rc = escalation_cli.main(
        ["resolve", bundle_dir, str(seq),
         "--resolution", "admit-after-review",
         "--operator", "si-113-test-operator",
         "--basis", "test basis: reviewed the journaled intent",
         "--json"])
    assert rc == 0
    rows = _resolution_rows(bundle_dir)
    assert len(rows) == 1
    record = rows[0]["record"]
    assert record["resolution"] == "admit-after-review"
    assert record["operator"] == "si-113-test-operator"
    assert record["basis"] == \
        "test basis: reviewed the journaled intent"
    assert record["intent_seq"] == seq
    assert record["edit_id"] == edit_id
    # The fold moved the intent escalated -> queued, and it
    # is no longer pending.
    _, journal = _open(bundle_dir)
    try:
        state = fold_queue(journal).intents[edit_id]
    finally:
        journal.close()
    assert state.state == "queued"
    assert state.resolution == "admit-after-review"
    assert state.resolution_seq == rows[0]["seq"]
    assert escalation_cli.list_escalations(bundle_dir) == []


def test_resolve_reject_annotates_and_unpends(mini):
    bundle_dir, edit_id, seq = _escalated(mini)
    entry = escalation_cli.resolve(
        bundle_dir, seq, "reject", operator="si-113-test-operator")
    assert entry["resolution"] == "reject"
    _, journal = _open(bundle_dir)
    try:
        state = fold_queue(journal).intents[edit_id]
    finally:
        journal.close()
    # reject changes no state — the annotation is the record
    # of its death — but the intent is not pending any more.
    assert state.state == "escalated"
    assert state.resolution == "reject"
    assert escalation_cli.list_escalations(bundle_dir) == []


def test_refused_resolutions_journal_nothing(mini):
    bundle_dir, edit_id, seq = _escalated(mini)
    # Invalid resolution value: refused by the act.
    tip_before = _tip(bundle_dir)
    assert escalation_cli.main(
        ["resolve", bundle_dir, str(seq),
         "--resolution", "mark-executed",
         "--operator", "si-113-test-operator"]) == 1
    assert _tip(bundle_dir) == tip_before
    assert _resolution_rows(bundle_dir) == []
    # Anonymous operator: refused by the act.
    assert escalation_cli.main(
        ["resolve", bundle_dir, str(seq),
         "--resolution", "admit-after-review",
         "--operator", "  "]) == 1
    assert _tip(bundle_dir) == tip_before
    # A real resolution, then a second one: refused, and the
    # journal gains nothing from the attempt.
    assert escalation_cli.main(
        ["resolve", bundle_dir, str(seq),
         "--resolution", "admit-after-review",
         "--operator", "si-113-test-operator"]) == 0
    tip_after_first = _tip(bundle_dir)
    assert escalation_cli.main(
        ["resolve", bundle_dir, str(seq),
         "--resolution", "reject",
         "--operator", "si-113-test-operator"]) == 1
    assert _tip(bundle_dir) == tip_after_first
    assert len(_resolution_rows(bundle_dir)) == 1


# -- the si CLI wiring -----------------------------------------------------------

def test_si_cli_wiring_parses_and_runs(mini, capsys):
    bundle_dir, edit_id, seq = _escalated(mini)
    args = build_parser().parse_args(
        ["gaia-escalation", bundle_dir, "list", "--json"])
    assert args.func is cmd_gaia_escalation
    rc = cli_main(["gaia-escalation", bundle_dir, "list",
                   "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out[out.index("["):])
    assert [row["intent_seq"] for row in payload] == [seq]
    rc = cli_main(["gaia-escalation", bundle_dir, "resolve",
                   str(seq), "--resolution", "admit-after-review",
                   "--operator", "si-113-test-operator"])
    assert rc == 0
    assert len(_resolution_rows(bundle_dir)) == 1
