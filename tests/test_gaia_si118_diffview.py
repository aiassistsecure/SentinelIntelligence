"""SI-118 / DIFFVIEW-1 (SI-111 §5): checkpoint diff views
over the journal.

Pins:

- the acceptance case: a fixture bundle replaying the
  se_0012 lifecycle shape (proposal ESCALATE -> resolution
  admit-after-review -> ``gaia_work`` submission -> outcome
  refused), built entirely from the programme's own
  writers (seam submission, the SI-113 resolution act,
  the SI-116 spend, the S-5 outcome leg — replay
  transports, labelled test doubles). The diff across the
  full window renders every leg with its seq; a sub-window
  renders only its legs; an empty window renders the
  explicit no-change result;
- prefix-fold diffing at the queue level: appearances,
  claim and outcome transitions with their causing seqs,
  verdict counts, materialisations, policy acts
  unabridged, and fold-derived integrity faults;
- chain refusal: a tampered journal is refused, nothing
  folded (fail closed);
- the read-only proof: every byte of the bundle is
  identical after diff runs, in both output formats,
  through both the library and the CLI surface;
- window validation: a reversed window is a usage error,
  a zero-width window is the explicit empty case.

No live node is touched: fixture bundles only, replay
transports only. A diff is a projection of the journal,
never new evidence (SI-111 §5.2/§10).
"""
import copy
import hashlib
import json
import os
import sqlite3

import pytest

from si.bundle import init_bundle, load_bundle
from si.cli import build_parser as build_si_parser
from si.gaia import diffview, escalation_cli
from si.gaia.corpus import corpus_policy
from si.gaia.diffview import compute_diff, diff_bundle
from si.gaia.hearth import ReplayTransport
from si.gaia.queue import record_queue_batch, record_queue_claim
from si.gaia.seam_spend import (
    poll_spent_outcome, spend_resolved_escalation)
from si.gaia.service import submit_proposal
from si.gaia.task import record_task_entry
from si.journal import EmbeddedBackend, Journal, open_journal
from si.tiny_seed import seed_tiny_bundle

WATCHER = {
    "model": "muse",
    "tool": "hearth.restart",
    "args": {"model": "muse",
             "spine_event": {"hash": "hash-s118-fixture", "seq": 41}},
    "reason": "observed loss at seq 41",
    "caused_by": {"hash": "hash-s118-fixture", "seq": 41},
}
ESCALATE = {"tool_class_policy": {
    "read": "admit", "write": "admit",
    "execute": "admit", "external": "escalate"}}
URL = "replay://fork/gaia/v1/work"
OPERATOR = "si118-test-operator"


# ---------------------------------------------------------------------------
# The acceptance fixture: the se_0012 lifecycle shape, from the real writers
# ---------------------------------------------------------------------------

@pytest.fixture()
def mini(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "mini.si"),
                             name="gaia-si118",
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    work = tmp_path / "work"
    work.mkdir()
    return bundle_dir, str(work)


def _lifecycle(mini):
    """Proposal ESCALATE -> resolution admit-after-review
    -> gaia_work submission -> outcome refused (the fork
    double answers stale_state, as P-2's live run earned).
    Returns (bundle_dir, intent_seq, resolution_seq,
    gaia_work_seq, outcome_seq)."""
    bundle_dir, work = mini
    out = submit_proposal(bundle_dir, copy.deepcopy(WATCHER),
                          policy=corpus_policy(work, ESCALATE))
    assert out["status"] == "acked" and out["verdict"] == "ESCALATE"
    intent_seq = out["journal_seq"]

    entry = escalation_cli.resolve(
        bundle_dir, intent_seq, "admit-after-review",
        operator=OPERATOR)
    resolution_seq = entry["seq"]

    spent = spend_resolved_escalation(
        bundle_dir, intent_seq, operator=OPERATOR,
        work_url=URL,
        transport=ReplayTransport([(202, {
            "work_id": "work-000001", "state": "accepted",
            "journal_seq": intent_seq})]))
    assert spent["status"] == "submitted"

    polled = poll_spent_outcome(
        bundle_dir, intent_seq, work_url=URL,
        transport=ReplayTransport([(200, {
            "work_id": "work-000001", "state": "executed",
            "provenance": {"si_journal_seqs": [intent_seq],
                           "spine_event_refs": []},
            "result": {
                "outcome": "refused", "tool": "hearth.restart",
                "model": "muse", "refusal": "stale_state",
                "detail": "the answered loss already healed "
                          "(SI-118 fixture double)",
                "caused_by": WATCHER["caused_by"],
                "spine_event_refs": []}})]))
    assert polled["status"] == "journaled"
    assert polled["outcome"] == "refused"
    return (bundle_dir, intent_seq, resolution_seq,
            spent["gaia_work_seq"], polled["outcome_seq"])


def _tip_seq(bundle_dir):
    journal = open_journal(bundle_dir)
    try:
        return journal.tip()[0]
    finally:
        journal.close()


def _state_legs(diff, intent_seq):
    """The folded state transitions of one intent, as
    (from_state, to_state, at_seq) triples."""
    return [(t["from_state"], t["to_state"], t["at_seq"])
            for t in diff["transitions"]
            if t["intent_seq"] == intent_seq
            and t["change"] in ("appeared", "state")]


# -- the acceptance case --------------------------------------------------------

def test_full_window_renders_every_leg_of_the_lifecycle(mini):
    bundle_dir, intent_seq, resolution_seq, work_seq, \
        outcome_seq = _lifecycle(mini)
    tip = _tip_seq(bundle_dir)
    diff = diff_bundle(bundle_dir, intent_seq - 1, tip)

    assert diff["changed"] is True
    assert diff["window"]["tip"] == tip

    proposals = [p for p in diff["proposals"]
                 if p["seq"] == intent_seq]
    assert len(proposals) == 1
    assert proposals[0]["kind"] == "tool"
    assert proposals[0]["tool"] == "hearth.restart"
    assert proposals[0]["gate_verdict"] == "ESCALATE"
    assert diff["verdict_counts"] == {"ESCALATE": 1}

    assert [r["seq"] for r in diff["resolutions"]] \
        == [resolution_seq]
    resolution = diff["resolutions"][0]
    assert resolution["intent_seq"] == intent_seq
    assert resolution["resolution"] == "admit-after-review"
    assert resolution["operator"] == OPERATOR

    assert [g["seq"] for g in diff["gaia_work"]] == [work_seq]
    submission = diff["gaia_work"][0]
    assert submission["intent_seq"] == intent_seq
    assert submission["resolution_seq"] == resolution_seq
    assert submission["fork_work_id"] == "work-000001"

    # Two outcome rows stand in the window: the
    # escalation's journaled non-execution answer (it
    # precedes the resolution — SI-92's record keeps it;
    # history is not rewritten) and the spend's refused
    # seam outcome, which is the lifecycle's terminal leg.
    outcome_seqs = [o["seq"] for o in diff["outcomes"]]
    assert len(outcome_seqs) == 2
    assert outcome_seqs[-1] == outcome_seq
    assert outcome_seqs[0] < resolution_seq
    assert diff["outcomes"][0]["status"].startswith(
        "not executed")
    assert diff["outcomes"][-1]["status"] == "refused"
    assert diff["outcomes"][-1]["work_id"] == "work-000001"

    # escalated -> queued -> refused-at-claim, each leg
    # cited by the seq that caused it — the P-2 record's
    # shape, rendered by one query.
    assert _state_legs(diff, intent_seq) == [
        (None, "escalated", intent_seq),
        ("escalated", "queued", resolution_seq),
        ("queued", "refused-at-claim", outcome_seq),
    ]
    annotations = [t for t in diff["transitions"]
                   if t["change"] == "resolution"]
    assert len(annotations) == 1
    assert annotations[0]["at_seq"] == resolution_seq
    assert annotations[0]["resolution"] == "admit-after-review"
    assert annotations[0]["resolution_operator"] == OPERATOR

    # The re-anchor checkpoints of the lifecycle's acts
    # are journaled state in the window too.
    assert diff["checkpoints"]


def test_subwindow_renders_only_its_own_legs(mini):
    bundle_dir, intent_seq, resolution_seq, work_seq, \
        outcome_seq = _lifecycle(mini)
    tip = _tip_seq(bundle_dir)

    # resolution -> outcome: only the spend/outcome legs.
    diff = diff_bundle(bundle_dir, resolution_seq, tip)
    assert diff["proposals"] == []
    assert diff["resolutions"] == []
    assert diff["verdict_counts"] == {}
    assert [g["seq"] for g in diff["gaia_work"]] == [work_seq]
    assert [o["seq"] for o in diff["outcomes"]] == [outcome_seq]
    assert _state_legs(diff, intent_seq) == [
        ("queued", "refused-at-claim", outcome_seq)]

    # proposal -> resolution: only the resolution leg
    # moves state; the window also holds the escalation's
    # non-execution outcome row (journaled before the
    # resolution, changing no folded state).
    diff = diff_bundle(bundle_dir, intent_seq, resolution_seq)
    assert [r["seq"] for r in diff["resolutions"]] \
        == [resolution_seq]
    assert diff["gaia_work"] == []
    assert len(diff["outcomes"]) == 1
    assert diff["outcomes"][0]["status"].startswith(
        "not executed")
    assert _state_legs(diff, intent_seq) == [
        ("escalated", "queued", resolution_seq)]


def test_empty_window_is_explicit_never_silent(mini):
    bundle_dir, intent_seq, *_ = _lifecycle(mini)
    tip = _tip_seq(bundle_dir)

    diff = diff_bundle(bundle_dir, tip, tip)
    assert diff["changed"] is False
    assert "no journaled change" in diff["message"]
    text = diffview.render_text(diff, bundle_dir=bundle_dir)
    assert "no journaled change" in text

    # The CLI agrees: exit 0, the explicit line printed.
    args = diffview.build_parser().parse_args(
        [bundle_dir, str(tip), str(tip)])
    assert diffview.execute(args) == 0


# -- the read-only proof ---------------------------------------------------------

def _bundle_hashes(root):
    hashes = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            path = os.path.join(dirpath, name)
            with open(path, "rb") as f:
                hashes[os.path.relpath(path, root)] = \
                    hashlib.sha256(f.read()).hexdigest()
    return hashes


def test_diff_is_strictly_read_only_bundle_bytes_identical(mini):
    bundle_dir, intent_seq, *_ = _lifecycle(mini)
    tip = _tip_seq(bundle_dir)
    before = _bundle_hashes(bundle_dir)

    diff_bundle(bundle_dir, intent_seq - 1, tip)
    diff_bundle(bundle_dir, tip, tip)
    args = diffview.build_parser().parse_args(
        [bundle_dir, str(intent_seq - 1), str(tip), "--json"])
    assert diffview.execute(args) == 0
    args = diffview.build_parser().parse_args(
        [bundle_dir, str(intent_seq - 1), str(tip)])
    assert diffview.execute(args) == 0

    assert _bundle_hashes(bundle_dir) == before


def test_broken_chain_is_refused_nothing_folded(mini):
    bundle_dir, intent_seq, *_ = _lifecycle(mini)
    tip = _tip_seq(bundle_dir)
    db = os.path.join(bundle_dir, "journal", "journal.sqlite3")
    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE self_edits SET doc = replace(doc, "
        "'ESCALATE', 'ADMIT') WHERE seq = ?", (intent_seq,))
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="does not verify"):
        diff_bundle(bundle_dir, intent_seq - 1, tip)
    args = diffview.build_parser().parse_args(
        [bundle_dir, str(intent_seq - 1), str(tip)])
    assert diffview.execute(args) == 1


# -- queue-level prefix diffing (writers of SI-50/Q-1) ----------------------------

@pytest.fixture()
def journal(tmp_path):
    journal = Journal(EmbeddedBackend(str(tmp_path / "journal")))
    yield journal
    journal.close()


def _add_intent(journal, task, verdict="ADMIT", tool="fs_read",
                queue=None):
    intent = {
        "kind": "tool", "tool": tool,
        "args": {"path": "a.txt"},
        "parents": [task["ref"]], "source_seqs": [task["seq"]],
        "reason": "si118 test intent",
    }
    report = {"verdict": verdict, "checks": []}
    if queue is not None:
        report["queue"] = queue
    edit_id = journal.record_self_edit(
        "tool", intent, "si118 test intent", verdict, report)
    rows = journal.backend.find(
        "self_edits", filter={"edit_id": edit_id}, limit=1)
    return edit_id, rows[0]["seq"]


def _add_outcome(journal, edit_id, status="completed"):
    execution = {"tool": "fs_read", "intent_id": edit_id,
                 "class": "read", "status": status,
                 "result": {"ok": True}}
    journal.record_experience("tool_result", {
        "kind": "tool_result", "source": "tool:fs_read",
        "intent_id": edit_id, "tool": "fs_read",
        "gate_verdict": "ADMIT", "effective_class": "read",
        "execution": execution}, edit_id)
    rows = journal.backend.find(
        "experiences", filter={"self_edit_id": edit_id},
        sort="-seq", limit=1)
    return rows[0]["seq"]


def test_claim_and_outcome_legs_with_causing_seqs(journal):
    task = record_task_entry(journal, "drain the queue",
                             "/tmp/si118-work")
    batch = record_queue_batch(journal, task["seq"],
                               task["ref"], declared_size=1)
    block = {"ordering_key": None, "depends_on": [],
             "barrier": False, "batch_seq": batch["seq"],
             "effective_class": "read"}
    edit_id, intent_seq = _add_intent(journal, task,
                                      queue=block)
    claim = record_queue_claim(
        journal, task["seq"], edit_id, intent_seq,
        "drainer-1", 30, effective_class="read")
    outcome_seq = _add_outcome(journal, edit_id)
    journal.record_delta("d-si118", "parent", "blob-ref", 12,
                         1, 1, ["mod"], 0.5, edit_id,
                         [intent_seq])
    tip = journal.tip()[0]

    diff = compute_diff(journal, 0, tip)
    assert [c["seq"] for c in diff["claims"]] == [claim["seq"]]
    assert diff["claims"][0]["drainer_id"] == "drainer-1"
    assert diff["claims"][0]["intent_seq"] == intent_seq
    assert [o["seq"] for o in diff["outcomes"]] == [outcome_seq]
    assert diff["outcomes"][0]["status"] == "completed"
    assert [m["delta_id"] for m in diff["materialisations"]] \
        == ["d-si118"]
    assert diff["verdict_counts"] == {"ADMIT": 1}
    assert _state_legs(diff, intent_seq) == [
        (None, "queued", intent_seq),
        ("queued", "claimed", claim["seq"]),
        ("claimed", "executed", outcome_seq),
    ]

    # A window opening after the claim sees only the
    # outcome leg.
    diff = compute_diff(journal, claim["seq"], tip)
    assert diff["claims"] == []
    assert _state_legs(diff, intent_seq) == [
        ("claimed", "executed", outcome_seq)]


def test_second_outcome_fault_surfaces_in_the_window(journal):
    task = record_task_entry(journal, "drain the queue",
                             "/tmp/si118-work")
    batch = record_queue_batch(journal, task["seq"],
                               task["ref"], declared_size=1)
    block = {"ordering_key": None, "depends_on": [],
             "barrier": False, "batch_seq": batch["seq"],
             "effective_class": "read"}
    edit_id, intent_seq = _add_intent(journal, task,
                                      queue=block)
    record_queue_claim(journal, task["seq"], edit_id,
                       intent_seq, "drainer-1", 30)
    _add_outcome(journal, edit_id)
    _add_outcome(journal, edit_id)

    diff = compute_diff(journal, 0, journal.tip()[0])
    faults = [f for f in diff["integrity_faults"]
              if f["kind"] == "fold"]
    assert len(faults) == 1
    assert faults[0]["fault"].startswith("second outcome entry")
    assert faults[0]["task_seq"] == task["seq"]


def test_policy_act_renders_unabridged(journal):
    record = {"kind": "operator_act",
              "act": "enable-live-proposer",
              "operator": "si118-test-operator",
              "detail": "the whole record, nothing trimmed"}
    journal.record_experience("operator_act", record, "")
    diff = compute_diff(journal, 0, journal.tip()[0])
    assert len(diff["policy_acts"]) == 1
    assert diff["policy_acts"][0]["record"] == record


# -- window validation and both CLI surfaces --------------------------------------

def test_reversed_window_is_a_usage_error(journal):
    with pytest.raises(ValueError, match="seq_b must be"):
        compute_diff(journal, 5, 3)
    with pytest.raises(ValueError, match=">= 0"):
        compute_diff(journal, -1, 3)


def test_both_output_formats_and_both_parsers(mini, capsys):
    bundle_dir, intent_seq, *_rest, outcome_seq = \
        _lifecycle(mini)
    tip = _tip_seq(bundle_dir)

    args = diffview.build_parser().parse_args(
        [bundle_dir, str(intent_seq - 1), str(tip), "--json"])
    assert diffview.execute(args) == 0
    parsed = json.loads(capsys.readouterr().out.split("\n", 1)[1])
    assert parsed["changed"] is True
    assert parsed["window"]["seq_b"] == tip
    assert parsed["outcomes"][-1]["seq"] == outcome_seq

    args = diffview.build_parser().parse_args(
        [bundle_dir, str(intent_seq - 1), str(tip)])
    assert diffview.execute(args) == 0
    text = capsys.readouterr().out
    assert "refused-at-claim" in text
    assert "verdicts: ESCALATE=1" in text
    assert "no journaled change" not in text

    # The si CLI wiring parses to the same command.
    si_args = build_si_parser().parse_args(
        ["gaia-diff", bundle_dir, "0", "5", "--json"])
    assert si_args.func.__name__ == "cmd_gaia_diff"
    assert si_args.seq_a == 0 and si_args.seq_b == 5
    assert si_args.json is True
