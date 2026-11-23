"""SI-122 / EXPORT-1 (SI-120 lane): verifiable run
bundle export + offline verifier.

Pins:

- the acceptance case: a fixture bundle replaying the
  se_0012 lifecycle shape (tool intent ESCALATE ->
  resolution admit-after-review -> ``gaia_work``
  submission -> outcome refused), built entirely from
  the programme's own writers (seam submission, the
  SI-113 resolution act, the SI-116 spend, the S-5
  outcome leg — replay transports, labelled test
  doubles). The full-journal export verifies offline;
  a sub-window export verifies offline; the export
  carries the manifest's constitution record, head,
  and manifest id, and the source tip seq + hash;
- the verifier's failure modes, each exit-non-zero /
  problems-non-empty: a tampered entry payload, a
  truncated entry list, two reordered entries, an
  altered recorded tip hash;
- the read-only proof: every byte of the bundle is
  identical after exports, through the library and
  both CLI surfaces;
- fail closed: a bundle whose journal chain is broken
  is refused at export (nothing exported);
- window validation: a reversed window and a window
  past the tip are usage errors.

No live node is touched: fixture bundles only, replay
transports only. The export proves internal consistency
and tamper-evidence against the recorded tip — not
that the outside world happened as recorded, and it is
not a signature (docs/SI-122-verifiable-export.md).
"""
import copy
import hashlib
import json
import os
import sqlite3

import pytest

from si.bundle import init_bundle, manifest_id
from si.cli import build_parser as build_si_parser
from si.gaia import run_export
from si.gaia.corpus import corpus_policy
from si.gaia.hearth import ReplayTransport
from si.gaia.run_export import (
    export_bundle, verify_export, verify_export_file)
from si.gaia.seam_spend import (
    poll_spent_outcome, spend_resolved_escalation)
from si.gaia.service import submit_proposal
from si.gaia import escalation_cli
from si.journal import open_journal
from si.tiny_seed import seed_tiny_bundle

WATCHER = {
    "model": "muse",
    "tool": "hearth.restart",
    "args": {"model": "muse",
             "spine_event": {"hash": "hash-s122-fixture", "seq": 41}},
    "reason": "observed loss at seq 41",
    "caused_by": {"hash": "hash-s122-fixture", "seq": 41},
}
ESCALATE = {"tool_class_policy": {
    "read": "admit", "write": "admit",
    "execute": "admit", "external": "escalate"}}
URL = "replay://fork/gaia/v1/work"
OPERATOR = "si122-test-operator"


# ---------------------------------------------------------------------------
# The acceptance fixture: the se_0012 lifecycle shape, from the real writers
# ---------------------------------------------------------------------------

@pytest.fixture()
def mini(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "mini.si"),
                             name="gaia-si122",
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
                          "(SI-122 fixture double)",
                "caused_by": WATCHER["caused_by"],
                "spine_event_refs": []}})]))
    assert polled["status"] == "journaled"
    assert polled["outcome"] == "refused"
    return (bundle_dir, intent_seq, resolution_seq,
            spent["gaia_work_seq"], polled["outcome_seq"])


def _tip(bundle_dir):
    journal = open_journal(bundle_dir)
    try:
        return journal.tip()
    finally:
        journal.close()


def _manifest(bundle_dir):
    with open(os.path.join(bundle_dir, "manifest.json"),
              "rb") as f:
        return json.loads(f.read().decode("utf-8"))


# -- the acceptance case --------------------------------------------------------

def test_full_export_verifies_offline(mini):
    bundle_dir, intent_seq, resolution_seq, work_seq, \
        outcome_seq = _lifecycle(mini)
    tip_seq, tip_hash = _tip(bundle_dir)
    manifest = _manifest(bundle_dir)

    doc = export_bundle(bundle_dir)
    assert verify_export(doc) == []

    meta = doc["export"]
    assert meta["source_tip"] == {"seq": tip_seq,
                                  "entry_hash": tip_hash}
    assert meta["window"] == {"from_seq": 0, "to_seq": tip_seq}
    assert meta["window_start_prev_hash"] == "genesis"
    assert meta["window_tip"] == {"seq": tip_seq,
                                  "entry_hash": tip_hash}
    assert meta["entry_count"] == len(doc["entries"])
    assert meta["exported_at"]

    # The bundle facts the export was taken at, exactly
    # as the manifest records them (no invented scheme).
    assert doc["bundle"]["constitution"] \
        == manifest["constitution"]
    assert doc["bundle"]["constitution"]["hash"].startswith(
        "sha256:")
    assert doc["bundle"]["head"] == manifest["head"]
    assert doc["bundle"]["manifest_id"] == manifest_id(manifest)

    # Every lifecycle leg is in the export, in order.
    seqs = [e["entry"]["seq"] for e in doc["entries"]]
    assert seqs == sorted(seqs)
    assert seqs[0] == 1 and seqs[-1] == tip_seq
    for leg in (intent_seq, resolution_seq, work_seq,
                outcome_seq):
        assert leg in seqs
    by_seq = {e["entry"]["seq"]: e for e in doc["entries"]}
    intent = by_seq[intent_seq]["entry"]
    assert intent["kind"] == "tool"
    assert intent["gate_verdict"] == "ESCALATE"
    assert by_seq[resolution_seq]["entry"]["kind"] \
        == "escalation_resolution" or \
        by_seq[resolution_seq]["entry"].get("record", {}) \
        .get("kind") == "escalation_resolution"


def test_window_export_verifies_offline(mini):
    bundle_dir, intent_seq, resolution_seq, work_seq, \
        outcome_seq = _lifecycle(mini)
    tip_seq, _tip_hash = _tip(bundle_dir)

    doc = export_bundle(bundle_dir,
                        from_seq=intent_seq - 1,
                        to_seq=resolution_seq)
    assert verify_export(doc) == []
    seqs = [e["entry"]["seq"] for e in doc["entries"]]
    assert intent_seq in seqs and resolution_seq in seqs
    assert work_seq not in seqs and outcome_seq not in seqs
    # A partial window's tip is its own last entry, not
    # the source tip — and the recorded source tip is
    # still the journal's.
    assert doc["export"]["window_tip"]["seq"] == resolution_seq
    assert doc["export"]["source_tip"]["seq"] == tip_seq


def test_export_file_round_trip_and_cli_verdicts(mini, tmp_path,
                                                 capsys):
    bundle_dir, *_ = _lifecycle(mini)
    out = str(tmp_path / "run-export.json")

    args = run_export.build_export_parser().parse_args(
        [bundle_dir, "--out", out])
    assert run_export.execute_export(args) == 0
    result = verify_export_file(out)
    assert result["ok"] is True
    assert result["problems"] == []

    args = run_export.build_verify_parser().parse_args([out])
    assert run_export.execute_verify(args) == 0
    assert "VERIFIED" in capsys.readouterr().out


# -- verifier failure modes -------------------------------------------------------

def test_tampered_entry_payload_fails(mini):
    bundle_dir, intent_seq, *_ = _lifecycle(mini)
    doc = export_bundle(bundle_dir)
    for item in doc["entries"]:
        if item["entry"]["seq"] == intent_seq:
            item["entry"]["gate_verdict"] = "ADMIT"
    problems = verify_export(doc)
    assert problems and "tampered" in problems[0]


def test_truncated_chain_fails(mini):
    bundle_dir, *_ = _lifecycle(mini)
    doc = export_bundle(bundle_dir)
    doc["entries"] = doc["entries"][:-1]
    problems = verify_export(doc)
    assert problems


def test_reordered_entries_fail(mini):
    bundle_dir, *_ = _lifecycle(mini)
    doc = export_bundle(bundle_dir)
    entries = doc["entries"]
    assert len(entries) >= 3
    entries[1], entries[2] = entries[2], entries[1]
    problems = verify_export(doc)
    assert problems


def test_altered_tip_hash_fails(mini, tmp_path, capsys):
    bundle_dir, *_ = _lifecycle(mini)
    doc = export_bundle(bundle_dir)
    doc["export"]["source_tip"]["entry_hash"] = "sha256:" + "0" * 64
    assert verify_export(doc)

    out = tmp_path / "tampered.json"
    out.write_text(json.dumps(doc), encoding="utf-8")
    args = run_export.build_verify_parser().parse_args(
        [str(out)])
    assert run_export.execute_verify(args) == 1
    assert "FAILED" in capsys.readouterr().out


def test_altered_entry_hash_fails(mini):
    bundle_dir, *_ = _lifecycle(mini)
    doc = export_bundle(bundle_dir)
    doc["entries"][0]["entry"]["entry_hash"] = \
        "sha256:" + "f" * 64
    assert verify_export(doc)


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


def test_export_is_strictly_read_only(mini, tmp_path):
    bundle_dir, intent_seq, *_ = _lifecycle(mini)
    before = _bundle_hashes(bundle_dir)

    export_bundle(bundle_dir)
    export_bundle(bundle_dir, from_seq=intent_seq - 1)
    out = str(tmp_path / "e.json")
    args = run_export.build_export_parser().parse_args(
        [bundle_dir, "--out", out])
    assert run_export.execute_export(args) == 0
    args = run_export.build_verify_parser().parse_args([out])
    assert run_export.execute_verify(args) == 0

    assert _bundle_hashes(bundle_dir) == before


# -- fail closed / window validation ----------------------------------------------

def test_broken_chain_bundle_is_refused_at_export(mini):
    bundle_dir, intent_seq, *_ = _lifecycle(mini)
    db = os.path.join(bundle_dir, "journal", "journal.sqlite3")
    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE self_edits SET doc = replace(doc, "
        "'ESCALATE', 'ADMIT') WHERE seq = ?", (intent_seq,))
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="does not verify"):
        export_bundle(bundle_dir)
    args = run_export.build_export_parser().parse_args(
        [bundle_dir])
    assert run_export.execute_export(args) == 1


def test_window_validation(mini):
    bundle_dir, *_ = _lifecycle(mini)
    tip_seq, _ = _tip(bundle_dir)
    with pytest.raises(ValueError, match="to_seq must be"):
        export_bundle(bundle_dir, from_seq=5, to_seq=3)
    with pytest.raises(ValueError, match="past the journal tip"):
        export_bundle(bundle_dir, to_seq=tip_seq + 1)
    with pytest.raises(ValueError, match=">= 0"):
        export_bundle(bundle_dir, from_seq=-1)


def test_zero_width_window_is_the_explicit_empty_case(mini):
    bundle_dir, *_ = _lifecycle(mini)
    tip_seq, _ = _tip(bundle_dir)
    doc = export_bundle(bundle_dir, from_seq=tip_seq,
                        to_seq=tip_seq)
    assert doc["entries"] == []
    assert doc["export"]["entry_count"] == 0
    assert doc["export"]["window_tip"] is None
    assert verify_export(doc) == []


def test_si_cli_wiring_parses_to_the_export_commands():
    args = build_si_parser().parse_args(
        ["gaia-export", "some.si", "--from", "3", "--to",
         "9", "--out", "e.json"])
    assert args.func.__name__ == "cmd_gaia_export"
    assert args.from_seq == 3 and args.to_seq == 9
    assert args.out == "e.json"

    args = build_si_parser().parse_args(
        ["gaia-verify-export", "e.json"])
    assert args.func.__name__ == "cmd_gaia_verify_export"
    assert args.file == "e.json"
