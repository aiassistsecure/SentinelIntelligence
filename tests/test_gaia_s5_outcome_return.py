"""SI-59 / S-5 (G-9 §11/§12.5): Gaia seam outcome return, SI side.

Pins: the §11.2 poller journals a terminal fork status as a
``tool_result`` seam outcome exactly once per work_id — the
§12.5 envelope (source gaia-seam, seam pair, fork result
verbatim, row's self_edit_id = the intent's edit) — so the
intent's provenance trail (build_provenance_trail, unmodified)
collects it and the queue fold learns it; non-terminal / 404 /
failure answers journal nothing and retry free; a refusal
result folds refused-at-claim; the §11.3 reconciler tails
outcomes.jsonl read-only from a byte cursor, journals outcomes
the poller never saw (post-restart recovery), cross-checks
agreements, journals disagreements and distinct second
outcomes as integrity faults (never rewriting the journaled
outcome), counts unmatched work_ids without journaling them,
and stops at a torn line (counted, cursor held, recovered on
the next tail); §11.4 outcome-unknown is an explicit act in
the SI-50 §13.3 shape — the same caused_by is blocked while
undisposed and the existing §13.4 disposition act unblocks
it; the real HttpTransport runs against a loopback test
double (not a fork); live interop is env-gated on
SI_S5_WORK_URL and UNMEASURED.
"""
import copy
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from si.bundle import (
    _anchor_checkpoint,
    _write_manifest,
    init_bundle,
    load_bundle,
)
from si.gaia.corpus import corpus_policy
from si.gaia.hearth import ReplayTransport
from si.gaia.queue import disposition_outcome_unknown, fold_queue
from si.gaia.seam_outcome import (
    OUTCOME_RETURN_LABEL,
    mark_seam_outcome_unknown,
    poll_seam_outcomes,
    reconcile_seam_outcomes,
    seam_reproposal_blocked,
)
from si.gaia.seam_submit import submit_seam_edit
from si.gaia.service import (
    build_provenance_trail,
    poll_outcomes,
    submit_proposal,
)
from si.journal import open_journal
from si.tiny_seed import seed_tiny_bundle

WATCHER = {
    "model": "muse",
    "tool": "hearth.restart",
    "args": {"model": "muse",
             "spine_event": {"hash": "hash-s5-fixture", "seq": 41}},
    "reason": "observed loss at seq 41",
    "caused_by": {"hash": "hash-s5-fixture", "seq": 41},
}
ADMIT = {"tool_class_policy": {
    "read": "admit", "write": "admit",
    "execute": "admit", "external": "admit"}}
WORK_URL = "replay://fork/gaia/v1/work"
RESULT = {
    "outcome": "executed",
    "tool": "hearth.restart",
    "model": "muse",
    "detail": "restarted (S-5 test double)",
    "caused_by": {"hash": "hash-s5-fixture", "seq": 41},
    "spine_event_refs": [{"hash": "spine-ref-1", "seq": 99}],
}
STATUS_BODY = {
    "work_id": "work-1",
    "state": "executed",
    "provenance": {"si_journal_seqs": [0],
                   "spine_event_refs": RESULT["spine_event_refs"]},
    "result": RESULT,
}


@pytest.fixture()
def mini(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "mini.si"),
                             name="gaia-s5",
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    work = tmp_path / "work"
    work.mkdir()
    return bundle_dir, str(work)


def _open(bundle_dir):
    return open_journal(bundle_dir, load_bundle(bundle_dir))


def _submitted(mini):
    bundle_dir, work = mini
    out = submit_proposal(bundle_dir, copy.deepcopy(WATCHER),
                          policy=corpus_policy(work, ADMIT))
    assert out["status"] == "acked" and out["verdict"] == "ADMIT"
    sub = submit_seam_edit(
        bundle_dir, out["edit_id"], submission_url=WORK_URL,
        transport=ReplayTransport([(202, {"work_id": "work-1"})]))
    assert sub["status"] == "submitted"
    return bundle_dir, work, out


def _rows(bundle_dir, kind):
    journal = _open(bundle_dir)
    try:
        return journal.backend.find(
            "experiences", filter={"kind": kind})
    finally:
        journal.close()


def _outcome_line(result=RESULT, work_id="work-1"):
    return json.dumps({
        "work_id": work_id, "tool": result["tool"],
        "model": result["model"], "outcome": result["outcome"],
        "refusal": result.get("refusal"), "detail": result["detail"],
        "caused_by": result["caused_by"],
        "spine_event_refs": result["spine_event_refs"],
        "result": result})


# -- the poller (§11.2) --------------------------------------------------------

def test_poll_journals_terminal_outcome_once(mini):
    bundle_dir, work, admitted = _submitted(mini)
    transport = ReplayTransport([(200, STATUS_BODY)])
    out = poll_seam_outcomes(bundle_dir, work_url=WORK_URL,
                             transport=transport)
    assert out["status"] == "polled" and out["count"] == 1
    call = transport.calls[0]
    assert call["method"] == "GET"
    assert call["url"] == WORK_URL + "/work-1"
    entry = out["journaled"][0]
    assert entry["deduped"] is False
    assert entry["outcome"] == "executed"

    rows = [r for r in _rows(bundle_dir, "tool_result")
            if (r.get("record") or {}).get("source") == "gaia-seam"]
    assert len(rows) == 1
    row = rows[0]
    assert row["self_edit_id"] == admitted["edit_id"]
    rec = row["record"]
    assert rec["kind"] == "tool_result"
    assert rec["work_id"] == "work-1"
    assert rec["journal_seq"] == admitted["journal_seq"]
    assert rec["gate_verdict"] == "ADMIT"
    # The fork result is journaled verbatim, spine refs included.
    assert rec["fork_result"] == RESULT
    assert rec["execution"]["spine_event_refs"] == \
        RESULT["spine_event_refs"]
    assert rec["execution"]["status"] == "executed"
    assert rec["via"] == "poll"

    journal = _open(bundle_dir)
    try:
        # The provenance trail collects it, unmodified machinery.
        trail = build_provenance_trail(journal, admitted["task_seq"])
        assert "work-1" in json.dumps(trail)
        # The queue fold learns it through _apply_outcome.
        fold = fold_queue(journal)
        state = fold.intents[admitted["edit_id"]]
        assert state.state == "executed"
        assert state.outcome_seq == entry["outcome_seq"]
        assert journal.verify_chain() is True
    finally:
        journal.close()

    # A second poll is the §7.2 no-op: no transport, no new row.
    transport2 = ReplayTransport([])
    out2 = poll_outcomes(bundle_dir, work_url=WORK_URL,
                         transport=transport2)
    assert out2["count"] == 0 and transport2.calls == []
    assert len([r for r in _rows(bundle_dir, "tool_result")
                if (r.get("record") or {}).get("source")
                == "gaia-seam"]) == 1


def test_poll_non_terminal_and_failures_journal_nothing(mini):
    bundle_dir, work, admitted = _submitted(mini)
    out = poll_seam_outcomes(
        bundle_dir, work_url=WORK_URL,
        transport=ReplayTransport([(200, {"work_id": "work-1",
                                           "state": "admitted"})]))
    assert out["awaiting"][0]["state"] == "admitted"
    assert out["count"] == 0
    out = poll_seam_outcomes(
        bundle_dir, work_url=WORK_URL,
        transport=ReplayTransport([(404, {"error": "gone"})]))
    assert out["not_found"][0]["work_id"] == "work-1"
    out = poll_seam_outcomes(
        bundle_dir, work_url=WORK_URL,
        transport=ReplayTransport([(500, {"error": "no"})]))
    assert out["failed"][0]["http_status"] == 500
    out = poll_seam_outcomes(
        bundle_dir, work_url=WORK_URL,
        transport=ReplayTransport([RuntimeError("down")]))
    assert out["failed"][0]["status"] == "poll-failed"
    assert [r for r in _rows(bundle_dir, "tool_result")
            if (r.get("record") or {}).get("source")
            == "gaia-seam"] == []
    # A retry after failures is free and journals the outcome.
    out = poll_seam_outcomes(
        bundle_dir, work_url=WORK_URL,
        transport=ReplayTransport([(200, STATUS_BODY)]))
    assert out["count"] == 1


def test_poll_refusal_result_folds_refused(mini):
    bundle_dir, work, admitted = _submitted(mini)
    refused = {**RESULT, "outcome": "refused", "refusal": "leash",
               "detail": "leash state stale (S-5 test double)",
               "spine_event_refs": []}
    body = {**STATUS_BODY, "result": refused}
    out = poll_seam_outcomes(
        bundle_dir, work_url=WORK_URL,
        transport=ReplayTransport([(200, body)]))
    assert out["count"] == 1
    assert out["journaled"][0]["outcome"] == "refused"
    journal = _open(bundle_dir)
    try:
        fold = fold_queue(journal)
        assert fold.intents[admitted["edit_id"]].state == \
            "refused-at-claim"
    finally:
        journal.close()


def test_poll_without_endpoint_is_refused(mini):
    bundle_dir, work, admitted = _submitted(mini)
    out = poll_seam_outcomes(bundle_dir)
    assert out["status"] == "refused" and out["journaled"] is False
    assert "UNMEASURED" in out["problems"][0]
    assert "UNMEASURED" in OUTCOME_RETURN_LABEL


# -- the reconciler (§11.3) ------------------------------------------------------

def test_reconciler_recovers_outcome_after_restart(mini, tmp_path):
    bundle_dir, work, admitted = _submitted(mini)
    path = tmp_path / "outcomes.jsonl"
    path.write_text(_outcome_line() + "\n")
    out = reconcile_seam_outcomes(bundle_dir, str(path))
    assert out["status"] == "reconciled" and out["count"] == 1
    assert out["cursor"] == path.stat().st_size
    rows = [r for r in _rows(bundle_dir, "tool_result")
            if (r.get("record") or {}).get("source") == "gaia-seam"]
    assert len(rows) == 1
    assert rows[0]["record"]["fork_result"] == RESULT
    # Re-tailing from 0 cross-checks the agreement, journals none.
    out2 = reconcile_seam_outcomes(bundle_dir, str(path), cursor=0)
    assert out2["count"] == 0 and len(out2["cross_checked"]) == 1
    assert len([r for r in _rows(bundle_dir, "tool_result")
                if (r.get("record") or {}).get("source")
                == "gaia-seam"]) == 1
    # The file was only ever read.
    assert path.read_text() == _outcome_line() + "\n"


def test_reconciler_disagreement_and_second_outcome_fault(mini,
                                                          tmp_path):
    bundle_dir, work, admitted = _submitted(mini)
    poll_seam_outcomes(bundle_dir, work_url=WORK_URL,
                       transport=ReplayTransport([(200, STATUS_BODY)]))
    other = {**RESULT, "detail": "a different story"}
    path = tmp_path / "outcomes.jsonl"
    path.write_text(_outcome_line(result=other) + "\n")
    out = reconcile_seam_outcomes(bundle_dir, str(path))
    assert out["count"] == 0 and len(out["faults"]) == 1
    faults = _rows(bundle_dir, "gaia_outcome")
    assert len(faults) == 1
    assert faults[0]["record"]["status"] == "integrity-fault"
    # The journaled outcome stands, unrewritten.
    rows = [r for r in _rows(bundle_dir, "tool_result")
            if (r.get("record") or {}).get("source") == "gaia-seam"]
    assert len(rows) == 1
    assert rows[0]["record"]["fork_result"] == RESULT
    # The same fault line re-tailed is deduped by its offset.
    out2 = reconcile_seam_outcomes(bundle_dir, str(path), cursor=0)
    assert out2["faults"] == []
    assert len(_rows(bundle_dir, "gaia_outcome")) == 1


def test_reconciler_unmatched_and_torn_line(mini, tmp_path):
    bundle_dir, work, admitted = _submitted(mini)
    path = tmp_path / "outcomes.jsonl"
    good = _outcome_line()
    torn = '{"work_id": "work-1", "resu'
    path.write_bytes(
        (_outcome_line(work_id="work-ghost") + "\n" + good + "\n"
         ).encode() + torn.encode())
    out = reconcile_seam_outcomes(bundle_dir, str(path))
    assert [u["work_id"] for u in out["unmatched"]] == ["work-ghost"]
    assert out["count"] == 1 and out["torn"] == 1
    expected_cursor = path.stat().st_size - len(torn.encode())
    assert out["cursor"] == expected_cursor
    # The torn line completed later is recovered on the next tail
    # (here it duplicates the journaled outcome: a cross-check).
    with open(path, "ab") as fh:
        fh.write(b'lt": {}}' + b"\n")
    # (the completed bytes are not a valid record — still torn-class)
    out2 = reconcile_seam_outcomes(bundle_dir, str(path),
                                   cursor=out["cursor"])
    assert out2["torn"] == 1 and out2["cursor"] == expected_cursor


def test_reconciler_missing_file_is_refused(mini, tmp_path):
    bundle_dir, work, admitted = _submitted(mini)
    out = reconcile_seam_outcomes(
        bundle_dir, str(tmp_path / "nope.jsonl"))
    assert out["status"] == "refused" and out["journaled"] is False


# -- outcome-unknown (§11.4) -----------------------------------------------------

def test_outcome_unknown_blocks_and_disposition_unblocks(mini):
    bundle_dir, work, admitted = _submitted(mini)
    out = mark_seam_outcome_unknown(
        bundle_dir, "work-1", cause="fork restarted; no outcome "
        "in outcomes.jsonl (S-5 test)",
        evidence=["GET /gaia/v1/work/work-1 -> 404",
                  "outcomes.jsonl tail: no work-1 record"])
    assert out["status"] == "outcome-unknown"
    assert out["deduped"] is False
    assert out["caused_by"] == WATCHER["caused_by"]
    assert "§13.4" in out["disposition"]
    journal = _open(bundle_dir)
    try:
        fold = fold_queue(journal)
        state = fold.intents[admitted["edit_id"]]
        assert state.state == "outcome-unknown"
        assert state.recovery_observations  # evidence journaled
    finally:
        journal.close()
    assert seam_reproposal_blocked(
        bundle_dir, copy.deepcopy(WATCHER["caused_by"])) is True
    # A repeat mark is deduped, journaling no second entry.
    again = mark_seam_outcome_unknown(
        bundle_dir, "work-1", cause="again")
    assert again["deduped"] is True
    assert len(_rows(bundle_dir, "queue_outcome_unknown")) == 1
    # The existing §13.4 act dispositions it; the block lifts.
    manifest = load_bundle(bundle_dir)
    journal = open_journal(bundle_dir, manifest)
    try:
        entry = disposition_outcome_unknown(
            journal, admitted["journal_seq"], "accept-unknown",
            detail="operator accepts the unknown (S-5 test)")
        assert entry["disposition"] == "accept-unknown"
        updated = _anchor_checkpoint(
            journal, manifest, manifest["head"],
            note="s5 test disposition act")
        _write_manifest(bundle_dir, updated)
    finally:
        journal.close()
    assert seam_reproposal_blocked(
        bundle_dir, copy.deepcopy(WATCHER["caused_by"])) is False


def test_outcome_unknown_refusals(mini):
    bundle_dir, work, admitted = _submitted(mini)
    out = mark_seam_outcome_unknown(bundle_dir, "work-nope",
                                    cause="x")
    assert out["status"] == "refused"
    poll_seam_outcomes(bundle_dir, work_url=WORK_URL,
                       transport=ReplayTransport([(200, STATUS_BODY)]))
    out = mark_seam_outcome_unknown(bundle_dir, "work-1",
                                    cause="x")
    assert out["status"] == "refused"
    assert _rows(bundle_dir, "queue_outcome_unknown") == []


# -- loopback HTTP test double (not a fork) --------------------------------------

def test_loopback_http_double_via_real_http_transport(mini):
    bundle_dir, work, admitted = _submitted(mini)
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):  # noqa: N802 - stdlib name
            seen["path"] = self.path
            body = json.dumps(STATUS_BODY).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = (f"http://127.0.0.1:{server.server_address[1]}"
               "/gaia/v1/work")
        out = poll_seam_outcomes(bundle_dir, work_url=url)
    finally:
        server.shutdown()
        server.server_close()
    assert out["count"] == 1
    assert out["journaled"][0]["outcome"] == "executed"
    assert seen["path"] == "/gaia/v1/work/work-1"
    rows = [r for r in _rows(bundle_dir, "tool_result")
            if (r.get("record") or {}).get("source") == "gaia-seam"]
    assert rows[0]["record"]["transport"] == "http"


# -- live fork interop (env-gated; UNMEASURED in this slice) ---------------------

def test_live_fork_outcome_poll_env_gated(mini):
    url = os.environ.get("SI_S5_WORK_URL")
    if not url:
        pytest.skip(
            "live fork outcome poll: SI_S5_WORK_URL is unset — "
            "no live gaia-api endpoint was exercised by this "
            "slice; live interop UNMEASURED (the loopback test "
            "double above is the always-on proof)")
    bundle_dir, work, admitted = _submitted(mini)
    out = poll_seam_outcomes(bundle_dir, work_url=url)
    assert out["status"] == "polled"
