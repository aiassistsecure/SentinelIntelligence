"""SI-124 / OBS-1 (SI-120 lane): OpenTelemetry span
mapping from the journal + OTLP export.

Pins:

- the acceptance case: a fixture bundle replaying the
  se_0012 lifecycle shape (tool intent ESCALATE ->
  resolution admit-after-review -> ``gaia_work``
  submission -> outcome refused), built entirely from
  the programme's own writers (the SI-122 fixture
  pattern). The export maps every journal entry to
  exactly one span; the task's entries form ONE trace
  rooted at the task entry, with the intent parented
  to the task and the resolution / work submission /
  outcomes parented to the intent; seed entries the
  journal ties to no task land in the journal-scope
  trace;
- the refused outcome sets the span status to Error;
  the ESCALATE gate verdict is an ATTRIBUTE, not an
  error;
- deterministic ids: trace/span ids recompute from
  the documented derivations, and two exports of the
  same journal are byte-identical (the payload is a
  pure function of journal + manifest);
- timestamps are the journaled append times, and no
  duration is invented (end == start on every span);
- the read-only proof: every byte of the bundle is
  identical after exports, library and CLI;
- no network without the flag: with urllib's urlopen
  rigged to raise, an export with no --otlp-endpoint
  still succeeds; with the flag, the SAME payload
  bytes are POSTed to the given endpoint;
- fail closed: a bundle whose journal chain is broken
  is refused at export (nothing exported, exit 1).

No live node is touched: fixture bundles only, replay
transports only, urlopen stubbed in-process (no real
network anywhere in this file).
"""
import copy
import hashlib
import json
import os
import sqlite3
import urllib.request

import pytest

from si.bundle import init_bundle, manifest_id
from si.cli import build_parser as build_si_parser
from si.gaia import trace_export
from si.gaia.corpus import corpus_policy
from si.gaia.hearth import ReplayTransport
from si.gaia.seam_spend import (
    poll_spent_outcome, spend_resolved_escalation)
from si.gaia.service import submit_proposal
from si.gaia import escalation_cli
from si.gaia.trace_export import (
    export_traces_bundle, span_id_for, trace_id_for)
from si.journal import open_journal
from si.tiny_seed import seed_tiny_bundle

WATCHER = {
    "model": "muse",
    "tool": "hearth.restart",
    "args": {"model": "muse",
             "spine_event": {"hash": "hash-s124-fixture", "seq": 41}},
    "reason": "observed loss at seq 41",
    "caused_by": {"hash": "hash-s124-fixture", "seq": 41},
}
ESCALATE = {"tool_class_policy": {
    "read": "admit", "write": "admit",
    "execute": "admit", "external": "escalate"}}
URL = "replay://fork/gaia/v1/work"
OPERATOR = "si124-test-operator"


# ---------------------------------------------------------------------------
# The acceptance fixture: the se_0012 lifecycle shape, from the real writers
# ---------------------------------------------------------------------------

@pytest.fixture()
def mini(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "mini.si"),
                             name="gaia-si124",
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    work = tmp_path / "work"
    work.mkdir()
    return bundle_dir, str(work)


def _lifecycle(mini):
    """Proposal ESCALATE -> resolution admit-after-review
    -> gaia_work submission -> outcome refused (the
    SI-122 fixture replay). Returns (bundle_dir,
    intent_seq, resolution_seq, gaia_work_seq,
    outcome_seq)."""
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
                          "(SI-124 fixture double)",
                "caused_by": WATCHER["caused_by"],
                "spine_event_refs": []}})]))
    assert polled["status"] == "journaled"
    assert polled["outcome"] == "refused"
    return (bundle_dir, intent_seq, resolution_seq,
            spent["gaia_work_seq"], polled["outcome_seq"])


def _spans(payload):
    return payload["resourceSpans"][0]["scopeSpans"][0][
        "spans"]


def _attrs(span):
    return {a["key"]: a["value"] for a in span["attributes"]}


def _by_seq(payload):
    out = {}
    for span in _spans(payload):
        attrs = _attrs(span)
        out[int(attrs["si.journal.seq"]["intValue"])] = span
    return out


def _manifest(bundle_dir):
    with open(os.path.join(bundle_dir, "manifest.json"),
              "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def _task_seq(bundle_dir):
    journal = open_journal(bundle_dir)
    try:
        for row in journal.backend.find("experiences") or []:
            if row.get("kind") == "task":
                return row["seq"]
    finally:
        journal.close()
    raise AssertionError("fixture journal has no task entry")


# -- the acceptance case --------------------------------------------------------

def test_lifecycle_maps_to_one_task_trace(mini):
    bundle_dir, intent_seq, resolution_seq, work_seq, \
        outcome_seq = _lifecycle(mini)
    manifest = _manifest(bundle_dir)
    mid = manifest_id(manifest)
    task_seq = _task_seq(bundle_dir)

    payload = export_traces_bundle(bundle_dir)
    spans = _spans(payload)
    by_seq = _by_seq(payload)

    # One span per journal entry, seqs 1..tip.
    journal = open_journal(bundle_dir)
    try:
        tip_seq, _ = journal.tip()
    finally:
        journal.close()
    assert sorted(by_seq) == list(range(1, tip_seq + 1))
    assert len(spans) == tip_seq

    # Exactly two traces: the task's and journal scope.
    task_trace = trace_id_for(mid, f"task@{task_seq}")
    journal_trace = trace_id_for(mid, "journal")
    assert {s["traceId"] for s in spans} == {
        task_trace, journal_trace}

    # The task entry is the trace root.
    task_span = by_seq[task_seq]
    assert task_span["name"] == "gaia.task"
    assert task_span["traceId"] == task_trace
    assert "parentSpanId" not in task_span

    # The intent hangs from the task, carrying the tool,
    # the verdict as an attribute, and the external
    # caused_by as attributes (never parentage).
    intent = by_seq[intent_seq]
    assert intent["name"] == "gaia.tool_intent"
    assert intent["traceId"] == task_trace
    assert intent["parentSpanId"] == task_span["spanId"]
    attrs = _attrs(intent)
    assert attrs["tool.name"] == {"stringValue": "hearth.restart"}
    assert attrs["si.gate.verdict"] == {"stringValue": "ESCALATE"}
    assert attrs["si.caused_by.seq"] == {"intValue": "41"}
    assert attrs["si.caused_by.hash"] == {
        "stringValue": "hash-s124-fixture"}
    assert intent["status"] == {"code": 0}

    # Resolution, work submission, and the refused
    # outcome all hang from the intent.
    for seq, name in ((resolution_seq,
                       "gaia.escalation_resolution"),
                      (work_seq, "gaia.work_submission"),
                      (outcome_seq, "gaia.tool_result")):
        span = by_seq[seq]
        assert span["name"] == name
        assert span["traceId"] == task_trace
        assert span["parentSpanId"] == intent["spanId"]

    # The refused outcome is the one Error span; its
    # outcome is also carried as an attribute.
    outcome = by_seq[outcome_seq]
    assert outcome["status"]["code"] == 2
    assert _attrs(outcome)["si.execution.outcome"] == {
        "stringValue": "refused"}

    # Seed entries (before the task) are journal-scope.
    for seq in range(1, task_seq):
        assert by_seq[seq]["traceId"] == journal_trace


def test_ids_recompute_from_the_documented_derivations(mini):
    bundle_dir, intent_seq, *_ = _lifecycle(mini)
    payload = export_traces_bundle(bundle_dir)
    by_seq = _by_seq(payload)

    journal = open_journal(bundle_dir)
    try:
        rows = journal.backend.find("self_edits")
    finally:
        journal.close()
    intent_row = next(r for r in rows
                      if r["seq"] == intent_seq)
    assert by_seq[intent_seq]["spanId"] == span_id_for(
        intent_row["entry_hash"])
    assert by_seq[intent_seq]["attributes"]


def test_timestamps_are_append_times_with_no_invented_duration(mini):
    bundle_dir, *_ = _lifecycle(mini)
    payload = export_traces_bundle(bundle_dir)
    for span in _spans(payload):
        assert span["endTimeUnixNano"] \
            == span["startTimeUnixNano"]
        assert int(span["startTimeUnixNano"]) > 0


def test_two_exports_are_byte_identical(mini):
    bundle_dir, *_ = _lifecycle(mini)
    first = export_traces_bundle(bundle_dir)
    second = export_traces_bundle(bundle_dir)
    assert json.dumps(first, indent=2, sort_keys=True) \
        == json.dumps(second, indent=2, sort_keys=True)


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
    bundle_dir, *_ = _lifecycle(mini)
    before = _bundle_hashes(bundle_dir)

    export_traces_bundle(bundle_dir)
    out = str(tmp_path / "traces.json")
    args = trace_export.build_parser().parse_args(
        [bundle_dir, "--out", out])
    assert trace_export.execute_export_traces(args) == 0
    with open(out, "rb") as f:
        written = json.loads(f.read().decode("utf-8"))
    assert written == export_traces_bundle(bundle_dir)

    assert _bundle_hashes(bundle_dir) == before


# -- network gating ---------------------------------------------------------------

class _FakeResponse:
    status = 202

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_no_network_without_the_endpoint_flag(mini, tmp_path,
                                              monkeypatch):
    bundle_dir, *_ = _lifecycle(mini)

    def _boom(request, timeout=None):
        raise AssertionError(
            "urlopen called without --otlp-endpoint")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    out = str(tmp_path / "traces.json")
    args = trace_export.build_parser().parse_args(
        [bundle_dir, "--out", out])
    assert trace_export.execute_export_traces(args) == 0


def test_endpoint_posts_the_same_payload_bytes(mini, tmp_path,
                                               monkeypatch):
    bundle_dir, *_ = _lifecycle(mini)
    captured = {}

    def _capture(request, timeout=None):
        captured["url"] = request.full_url
        captured["data"] = request.data
        captured["content_type"] = request.headers.get(
            "Content-type")
        return _FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", _capture)
    out = str(tmp_path / "traces.json")
    endpoint = "http://127.0.0.1:4318/v1/traces"
    args = trace_export.build_parser().parse_args(
        [bundle_dir, "--out", out,
         "--otlp-endpoint", endpoint])
    assert trace_export.execute_export_traces(args) == 0

    with open(out, "rb") as f:
        file_bytes = f.read()
    assert captured["url"] == endpoint
    assert captured["data"] == file_bytes
    assert captured["content_type"] == "application/json"
    assert json.loads(captured["data"].decode("utf-8")) \
        == export_traces_bundle(bundle_dir)


def test_failed_post_is_a_loud_failure(mini, monkeypatch):
    bundle_dir, *_ = _lifecycle(mini)

    def _down(request, timeout=None):
        raise OSError("connection refused (test double)")

    monkeypatch.setattr(urllib.request, "urlopen", _down)
    args = trace_export.build_parser().parse_args(
        [bundle_dir, "--otlp-endpoint",
         "http://127.0.0.1:4318/v1/traces"])
    assert trace_export.execute_export_traces(args) == 1


# -- fail closed -------------------------------------------------------------------

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
        export_traces_bundle(bundle_dir)
    args = trace_export.build_parser().parse_args(
        [bundle_dir])
    assert trace_export.execute_export_traces(args) == 1


def test_si_cli_wiring_parses_to_the_traces_command():
    args = build_si_parser().parse_args(
        ["gaia-export-traces", "some.si", "--out", "t.json",
         "--otlp-endpoint", "http://127.0.0.1:4318/v1/traces"])
    assert args.func.__name__ == "cmd_gaia_export_traces"
    assert args.out == "t.json"
    assert args.otlp_endpoint \
        == "http://127.0.0.1:4318/v1/traces"
