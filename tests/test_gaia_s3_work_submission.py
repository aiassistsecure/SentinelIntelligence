"""SI-56 / S-3 (G-9 §9): Gaia seam work submission, SI side.

Pins: a pending ADMITted seam edit is submitted via an injected
transport and journaled as a ``gaia_work`` row carrying the fork
work ID from its 202 response, in state ``submitted/awaiting
outcome (S-5 unbuilt)`` — never completed, the self-edit staying
``pending``; the payload is built from the journaled edit; dedup
returns the journaled submission with no second transport call;
REJECT/ESCALATE/unknown edits are refused before any transport
call; non-202 / 202-without-ID / transport failures are NOT
journaled as submissions; the real HttpTransport is exercised
against a loopback test double (not a fork), and live fork
interop is env-gated on SI_S3_WORK_SUBMISSION_URL.
"""
import copy
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from si.bundle import init_bundle, load_bundle
from si.gaia.corpus import corpus_policy
from si.gaia.hearth import ReplayTransport
from si.gaia.seam_submit import (
    SUBMITTED_STATE,
    WORK_SUBMISSION_LABEL,
    build_work_payload,
    submit_pending_seam_work,
    submit_seam_edit,
)
from si.gaia.service import submit_proposal, submit_work
from si.journal import open_journal
from si.tiny_seed import seed_tiny_bundle

WATCHER = {
    "model": "muse",
    "tool": "hearth.restart",
    "args": {"model": "muse",
             "spine_event": {"hash": "hash-s3-fixture", "seq": 41}},
    "reason": "observed loss at seq 41",
    "caused_by": {"hash": "hash-s3-fixture", "seq": 41},
}
ADMIT = {"tool_class_policy": {
    "read": "admit", "write": "admit",
    "execute": "admit", "external": "admit"}}
URL = "replay://fork-work-submission"


@pytest.fixture()
def mini(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "mini.si"),
                             name="gaia-s3",
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    work = tmp_path / "work"
    work.mkdir()
    return bundle_dir, str(work)


def _open(bundle_dir):
    return open_journal(bundle_dir, load_bundle(bundle_dir))


def _admit(bundle_dir, work, payload=None):
    out = submit_proposal(
        bundle_dir, copy.deepcopy(payload or WATCHER),
        policy=corpus_policy(work, ADMIT))
    assert out["status"] == "acked" and out["verdict"] == "ADMIT"
    return out


def _gaia_work(bundle_dir):
    journal = _open(bundle_dir)
    try:
        return journal.backend.find(
            "experiences", filter={"kind": "gaia_work"})
    finally:
        journal.close()


def _ok_transport(work_id="work-0001"):
    return ReplayTransport([(202, {"work_id": work_id})])


# -- submission + journaling ------------------------------------------------

def test_admitted_edit_submits_and_journals_gaia_work(mini):
    bundle_dir, work = mini
    admitted = _admit(bundle_dir, work)
    transport = _ok_transport()
    out = submit_seam_edit(bundle_dir, admitted["edit_id"],
                           submission_url=URL, transport=transport)
    assert out["status"] == "submitted"
    assert out["state"] == SUBMITTED_STATE
    assert "S-5 unbuilt" in out["state"]
    assert out["fork_work_id"] == "work-0001"
    assert out["http_status"] == 202 and out["deduped"] is False
    assert "replay/stub" in out["transport"]
    # The transport was called once, with the journaled payload.
    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call["method"] == "POST" and call["url"] == URL
    sent = call["json_body"]
    assert sent["tool"] == "hearth.restart"
    assert sent["edit_id"] == admitted["edit_id"]
    assert sent["journal_seq"] == admitted["journal_seq"]
    assert sent["caused_by"] == WATCHER["caused_by"]
    assert sent["args"] == WATCHER["args"]

    journal = _open(bundle_dir)
    try:
        rows = journal.backend.find(
            "experiences", filter={"kind": "gaia_work"})
        assert len(rows) == 1
        row = rows[0]
        assert row["seq"] == out["gaia_work_seq"]
        assert row["self_edit_id"] == admitted["edit_id"]
        rec = row["record"]
        assert rec["status"] == "submitted"
        assert rec["state"] == SUBMITTED_STATE
        assert rec["fork_work_id"] == "work-0001"
        assert rec["response"] == {"work_id": "work-0001"}
        # Submission is not an outcome: the edit stays pending.
        edit = journal.backend.find(
            "self_edits",
            filter={"edit_id": admitted["edit_id"]}, limit=1)[0]
        assert edit["result"] == "pending"
        assert journal.verify_chain() is True
    finally:
        journal.close()


def test_build_work_payload_reads_the_journaled_row(mini):
    bundle_dir, work = mini
    admitted = _admit(bundle_dir, work)
    journal = _open(bundle_dir)
    try:
        row = journal.backend.find(
            "self_edits",
            filter={"edit_id": admitted["edit_id"]}, limit=1)[0]
    finally:
        journal.close()
    payload = build_work_payload(row)
    assert payload["record"] == "si.gaia.seam.work_submission"
    assert payload["task_seq"] == admitted["task_seq"]
    assert payload["state_after_submit"] == SUBMITTED_STATE


def test_submission_dedup_calls_no_transport_twice(mini):
    bundle_dir, work = mini
    admitted = _admit(bundle_dir, work)
    first = submit_seam_edit(bundle_dir, admitted["edit_id"],
                             submission_url=URL,
                             transport=_ok_transport())
    assert first["deduped"] is False
    transport = _ok_transport("work-other")
    second = submit_seam_edit(bundle_dir, admitted["edit_id"],
                              submission_url=URL, transport=transport)
    assert second["deduped"] is True
    assert second["fork_work_id"] == first["fork_work_id"]
    assert transport.calls == []
    assert len(_gaia_work(bundle_dir)) == 1


def test_submit_pending_scans_only_spendable_admits(mini):
    bundle_dir, work = mini
    admitted = _admit(bundle_dir, work)
    # An ESCALATEd proposal for another model is no candidate.
    other = copy.deepcopy(WATCHER)
    other["model"] = other["args"]["model"] = "other"
    other["args"]["spine_event"] = {"hash": "hash-other", "seq": 3}
    other["caused_by"] = {"hash": "hash-other", "seq": 3}
    esc = submit_proposal(bundle_dir, other,
                          policy=corpus_policy(work))
    assert esc["verdict"] == "ESCALATE"
    out = submit_pending_seam_work(
        bundle_dir, submission_url=URL, transport=_ok_transport())
    assert out["count"] == 1
    assert out["submitted"][0]["edit_id"] == admitted["edit_id"]
    assert len(_gaia_work(bundle_dir)) == 1


def test_submit_work_service_wrapper(mini):
    bundle_dir, work = mini
    admitted = _admit(bundle_dir, work)
    out = submit_work(bundle_dir, admitted["edit_id"],
                      submission_url=URL, transport=_ok_transport())
    assert out["status"] == "submitted"
    assert "S-4" in WORK_SUBMISSION_LABEL
    assert "UNMEASURED" in WORK_SUBMISSION_LABEL


# -- refusals: nothing submitted, nothing journaled --------------------------

@pytest.mark.parametrize("verdict_policy,payload_tool", [
    (None, None), ("reject", None)])
def test_non_admit_edits_are_refused(mini, verdict_policy, payload_tool):
    bundle_dir, work = mini
    if verdict_policy == "reject":
        payload = copy.deepcopy(WATCHER)
        payload["tool"] = "hearth.reap"  # unregistered -> REJECT
        out0 = submit_proposal(bundle_dir, payload,
                               policy=corpus_policy(work, ADMIT))
        assert out0["verdict"] == "REJECT"
    else:
        out0 = submit_proposal(bundle_dir, copy.deepcopy(WATCHER),
                               policy=corpus_policy(work))
        assert out0["verdict"] == "ESCALATE"
    transport = _ok_transport()
    out = submit_seam_edit(bundle_dir, out0["edit_id"],
                           submission_url=URL, transport=transport)
    assert out["status"] == "refused" and out["journaled"] is False
    assert transport.calls == []
    assert _gaia_work(bundle_dir) == []


def test_unknown_edit_and_missing_endpoint_are_refused(mini):
    bundle_dir, work = mini
    admitted = _admit(bundle_dir, work)
    transport = _ok_transport()
    out = submit_seam_edit(bundle_dir, "se_nope",
                           submission_url=URL, transport=transport)
    assert out["status"] == "refused" and transport.calls == []
    out = submit_seam_edit(bundle_dir, admitted["edit_id"])
    assert out["status"] == "refused"
    assert "UNMEASURED" in out["problems"][0]
    assert _gaia_work(bundle_dir) == []


# -- fork answers that are not submissions -----------------------------------

def test_non_202_is_failed_not_journaled(mini):
    bundle_dir, work = mini
    admitted = _admit(bundle_dir, work)
    transport = ReplayTransport([(500, {"error": "no"})])
    out = submit_seam_edit(bundle_dir, admitted["edit_id"],
                           submission_url=URL, transport=transport)
    assert out["status"] == "submission-failed"
    assert out["http_status"] == 500 and out["journaled"] is False
    assert _gaia_work(bundle_dir) == []


def test_202_without_work_id_is_unconfirmed_not_submitted(mini):
    bundle_dir, work = mini
    admitted = _admit(bundle_dir, work)
    transport = ReplayTransport([(202, {"accepted": True})])
    out = submit_seam_edit(bundle_dir, admitted["edit_id"],
                           submission_url=URL, transport=transport)
    assert out["status"] == "submission-unconfirmed"
    assert out["journaled"] is False
    assert _gaia_work(bundle_dir) == []


def test_transport_failure_is_failed_not_submitted(mini):
    bundle_dir, work = mini
    admitted = _admit(bundle_dir, work)
    transport = ReplayTransport([RuntimeError("down")])
    out = submit_seam_edit(bundle_dir, admitted["edit_id"],
                           submission_url=URL, transport=transport)
    assert out["status"] == "submission-failed"
    assert "RuntimeError" in out["failure"]
    assert _gaia_work(bundle_dir) == []
    # A retry after failure is free (nothing was journaled).
    retry = submit_seam_edit(bundle_dir, admitted["edit_id"],
                             submission_url=URL,
                             transport=_ok_transport())
    assert retry["status"] == "submitted"


def test_alternate_work_id_spelling_is_recorded_with_its_field(mini):
    bundle_dir, work = mini
    admitted = _admit(bundle_dir, work)
    transport = ReplayTransport([(202, {"id": "w-9"})])
    out = submit_seam_edit(bundle_dir, admitted["edit_id"],
                           submission_url=URL, transport=transport)
    assert out["status"] == "submitted"
    assert out["fork_work_id"] == "w-9"
    assert out["fork_work_id_field"] == "id"


# -- loopback HTTP test double (not a fork) -----------------------------------

def test_loopback_http_double_via_real_http_transport(mini):
    bundle_dir, work = mini
    admitted = _admit(bundle_dir, work)
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):  # noqa: N802 - stdlib name
            length = int(self.headers.get("Content-Length") or 0)
            seen["body"] = json.loads(self.rfile.read(length))
            body = json.dumps({"work_id": "work-loopback-1"}).encode()
            self.send_response(202)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/work"
        out = submit_seam_edit(bundle_dir, admitted["edit_id"],
                               submission_url=url)
    finally:
        server.shutdown()
        server.server_close()
    assert out["status"] == "submitted"
    assert out["transport"] == "http"  # real transport, loopback double
    assert out["fork_work_id"] == "work-loopback-1"
    assert seen["body"]["tool"] == "hearth.restart"


# -- live fork interop (env-gated; UNMEASURED in this slice) -------------------

def test_live_fork_work_submission_env_gated(mini):
    url = os.environ.get("SI_S3_WORK_SUBMISSION_URL")
    if not url:
        pytest.skip(
            "live fork work submission: SI_S3_WORK_SUBMISSION_URL "
            "is unset — no live gaia-api endpoint was exercised "
            "by this slice; live interop UNMEASURED (the "
            "loopback test double above is the always-on proof)")
    bundle_dir, work = mini
    admitted = _admit(bundle_dir, work)
    out = submit_seam_edit(bundle_dir, admitted["edit_id"],
                           submission_url=url)
    assert out["status"] == "submitted"
    assert out["fork_work_id"]
