"""SI-57 / S-4 (G-9 §10/§12.3): Gaia seam leash crossing, SI side.

Pins: a journaled queue_stop on a model's seam task blocks seam
work submission SI-side (leash-held: no transport call, no
gaia_work row, the edit stays pending) and a queue_resume
re-enables it; the payload carries the §10.3.3 leash_tip
piggyback; journaled stop/resume records cross as {model, kind,
journal_seq, reason} and are journaled (kind gaia_leash) only on
a confirmed 200 ACK — applied:false (the §10.3.2 watermark
no-op) is confirmed, not an error; non-200 / malformed-200 /
transport failures journal no crossing and a retry is free; sync
dedups by journal_seq and stops at the first gap; the §10.5
snapshot sends the current folded state as one record; a §12.3
heartbeat carries leash_tip/journal_tip and journals its
delivery; the real HttpTransport runs against a loopback test
double (not a fork); live interop is env-gated on
SI_S4_LEASH_URL and UNMEASURED.
"""
import copy
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from si.bundle import _anchor_checkpoint, _write_manifest, init_bundle, load_bundle
from si.gaia.corpus import corpus_policy
from si.gaia.hearth import ReplayTransport
from si.gaia.queue import record_queue_resume, record_queue_stop
from si.gaia.seam_leash import (
    LEASH_CROSSING_LABEL,
    build_leash_record,
    leash_records,
    seam_tasks,
    send_leash_heartbeat,
    send_leash_snapshot,
    sync_seam_leash,
)
from si.gaia.seam_submit import submit_seam_edit
from si.gaia.service import submit_proposal
from si.journal import open_journal
from si.tiny_seed import seed_tiny_bundle

WATCHER = {
    "model": "muse",
    "tool": "hearth.restart",
    "args": {"model": "muse",
             "spine_event": {"hash": "hash-s4-fixture", "seq": 41}},
    "reason": "observed loss at seq 41",
    "caused_by": {"hash": "hash-s4-fixture", "seq": 41},
}
ADMIT = {"tool_class_policy": {
    "read": "admit", "write": "admit",
    "execute": "admit", "external": "admit"}}
LEASH_URL = "replay://fork/gaia/v1/leash"
WORK_URL = "replay://fork-work-submission"


@pytest.fixture()
def mini(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "mini.si"),
                             name="gaia-s4",
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    work = tmp_path / "work"
    work.mkdir()
    return bundle_dir, str(work)


def _open(bundle_dir):
    return open_journal(bundle_dir, load_bundle(bundle_dir))


def _admit(bundle_dir, work):
    out = submit_proposal(
        bundle_dir, copy.deepcopy(WATCHER),
        policy=corpus_policy(work, ADMIT))
    assert out["status"] == "acked" and out["verdict"] == "ADMIT"
    return out


def _leash_act(bundle_dir, fn, task_seq, **kwargs):
    """Journal one operator leash act + re-anchor, as the
    bundle machinery does after journaling."""
    manifest = load_bundle(bundle_dir)
    journal = open_journal(bundle_dir, manifest)
    try:
        entry = fn(journal, task_seq, **kwargs)
        updated = _anchor_checkpoint(
            journal, manifest, manifest["head"],
            note=f"s4 test leash act {fn.__name__}")
        _write_manifest(bundle_dir, updated)
        return entry
    finally:
        journal.close()


def _rows(bundle_dir, kind):
    journal = _open(bundle_dir)
    try:
        return journal.backend.find(
            "experiences", filter={"kind": kind})
    finally:
        journal.close()


def _ack_transport(applied=True, last=0):
    return ReplayTransport([(200, {"applied": applied,
                                   "last_leash_seq": last})])


def _stop_and_admit(mini):
    bundle_dir, work = mini
    admitted = _admit(bundle_dir, work)
    stop = _leash_act(bundle_dir, record_queue_stop,
                      admitted["task_seq"], cause="operator-stop",
                      detail="leash pulled (S-4 test)")
    return bundle_dir, work, admitted, stop


# -- reading the journaled leash ---------------------------------------------

def test_seam_tasks_and_records_read_from_journal(mini):
    bundle_dir, work, admitted, stop = _stop_and_admit(mini)
    journal = _open(bundle_dir)
    try:
        tasks = seam_tasks(journal)
        assert tasks == [{"task_seq": admitted["task_seq"],
                          "model": "muse"}]
        records = leash_records(journal, admitted["task_seq"])
        assert records == [{"kind": "stop",
                            "journal_seq": stop["seq"],
                            "reason": "operator-stop"}]
        wire = build_leash_record("muse", records[0])
        assert wire == {"model": "muse", "kind": "stop",
                        "journal_seq": stop["seq"],
                        "reason": "operator-stop"}
        assert journal.verify_chain() is True
    finally:
        journal.close()


# -- the leash honoured SI-side: no submission while leashed ------------------

def test_stop_blocks_submission_and_resume_reenables(mini):
    bundle_dir, work, admitted, stop = _stop_and_admit(mini)
    transport = ReplayTransport([(202, {"work_id": "work-1"})])
    out = submit_seam_edit(bundle_dir, admitted["edit_id"],
                           submission_url=WORK_URL, transport=transport)
    assert out["status"] == "leash-held" and out["journaled"] is False
    assert out["leash"]["stop_seq"] == stop["seq"]
    assert out["leash"]["model"] == "muse"
    assert transport.calls == []
    assert _rows(bundle_dir, "gaia_work") == []
    journal = _open(bundle_dir)
    try:
        edit = journal.backend.find(
            "self_edits",
            filter={"edit_id": admitted["edit_id"]}, limit=1)[0]
        assert edit["result"] == "pending"
    finally:
        journal.close()

    _leash_act(bundle_dir, record_queue_resume, admitted["task_seq"],
               detail="operator resumed (S-4 test)")
    transport = ReplayTransport([(202, {"work_id": "work-1"})])
    out = submit_seam_edit(bundle_dir, admitted["edit_id"],
                           submission_url=WORK_URL, transport=transport)
    assert out["status"] == "submitted"
    # The payload carries the §10.3.3 piggyback: the resume's seq.
    sent = transport.calls[0]["json_body"]
    resume_seq = _rows(bundle_dir, "queue_resume")[0]["seq"]
    assert sent["leash_tip"] == resume_seq


def test_payload_leash_tip_defaults_to_task_creation_seq(mini):
    bundle_dir, work = mini
    admitted = _admit(bundle_dir, work)
    transport = ReplayTransport([(202, {"work_id": "work-1"})])
    out = submit_seam_edit(bundle_dir, admitted["edit_id"],
                           submission_url=WORK_URL, transport=transport)
    assert out["status"] == "submitted"
    sent = transport.calls[0]["json_body"]
    assert sent["leash_tip"] == admitted["task_seq"]


# -- crossing: records --------------------------------------------------------

def test_sync_crosses_stop_and_journals_confirmed_only(mini):
    bundle_dir, work, admitted, stop = _stop_and_admit(mini)
    transport = _ack_transport(applied=True, last=stop["seq"])
    out = sync_seam_leash(bundle_dir, leash_url=LEASH_URL,
                          transport=transport)
    assert out["status"] == "synced" and out["count"] == 1
    call = transport.calls[0]
    assert call["method"] == "POST" and call["url"] == LEASH_URL
    assert call["json_body"] == {"model": "muse", "kind": "stop",
                                 "journal_seq": stop["seq"],
                                 "reason": "operator-stop"}
    rows = _rows(bundle_dir, "gaia_leash")
    assert len(rows) == 1
    rec = rows[0]["record"]
    assert rec["crossing"] == "record" and rec["applied"] is True
    assert rec["last_leash_seq"] == stop["seq"]
    assert rec["leash_journal_seq"] == stop["seq"]
    assert "replay/stub" in rec["transport"]
    # Dedup: a second sync spends no transport call.
    transport2 = _ack_transport()
    out2 = sync_seam_leash(bundle_dir, leash_url=LEASH_URL,
                           transport=transport2)
    assert out2["count"] == 0 and transport2.calls == []
    assert len(_rows(bundle_dir, "gaia_leash")) == 1
    # A watermark no-op ACK (applied:false) is a confirmed
    # crossing too, not an error (§10.3.2).
    resume = _leash_act(bundle_dir, record_queue_resume,
                        admitted["task_seq"],
                        detail="operator resumed (S-4 test)")
    out3 = sync_seam_leash(
        bundle_dir, leash_url=LEASH_URL,
        transport=_ack_transport(applied=False, last=resume["seq"]))
    assert out3["status"] == "synced" and out3["count"] == 1
    assert out3["crossed"][0]["applied"] is False
    assert len(_rows(bundle_dir, "gaia_leash")) == 2


def test_unconfirmed_crossing_journals_nothing_and_retries(mini):
    bundle_dir, work, admitted, stop = _stop_and_admit(mini)
    out = sync_seam_leash(
        bundle_dir, leash_url=LEASH_URL,
        transport=ReplayTransport([(500, {"error": "no"})]))
    assert out["status"] == "partial" and out["count"] == 0
    assert _rows(bundle_dir, "gaia_leash") == []
    out = sync_seam_leash(
        bundle_dir, leash_url=LEASH_URL,
        transport=ReplayTransport([(200, {"nope": True})]))
    assert out["count"] == 0 and _rows(bundle_dir, "gaia_leash") == []
    out = sync_seam_leash(
        bundle_dir, leash_url=LEASH_URL,
        transport=ReplayTransport([RuntimeError("down")]))
    assert out["count"] == 0 and _rows(bundle_dir, "gaia_leash") == []
    retry = sync_seam_leash(bundle_dir, leash_url=LEASH_URL,
                            transport=_ack_transport(last=stop["seq"]))
    assert retry["status"] == "synced" and retry["count"] == 1


def test_sync_sends_in_journal_order_and_stops_at_gap(mini):
    bundle_dir, work, admitted, stop = _stop_and_admit(mini)
    resume = _leash_act(bundle_dir, record_queue_resume,
                        admitted["task_seq"],
                        detail="operator resumed (S-4 test)")
    transport = ReplayTransport([
        (500, {"error": "no"}),
        (200, {"applied": True, "last_leash_seq": resume["seq"]})])
    out = sync_seam_leash(bundle_dir, leash_url=LEASH_URL,
                          transport=transport)
    # The failed stop blocks the model; the resume is not sent past it.
    assert out["status"] == "partial" and out["count"] == 0
    assert len(transport.calls) == 1
    transport = ReplayTransport([
        (200, {"applied": True, "last_leash_seq": stop["seq"]}),
        (200, {"applied": True, "last_leash_seq": resume["seq"]})])
    out = sync_seam_leash(bundle_dir, leash_url=LEASH_URL,
                          transport=transport)
    assert out["status"] == "synced" and out["count"] == 2
    kinds = [c["json_body"]["kind"] for c in transport.calls]
    assert kinds == ["stop", "resume"]


def test_missing_endpoint_is_refused(mini):
    bundle_dir, work = mini
    _admit(bundle_dir, work)
    out = sync_seam_leash(bundle_dir)
    assert out["status"] == "refused" and out["journaled"] is False
    assert "UNMEASURED" in out["problems"][0]


# -- snapshot (§10.5) -----------------------------------------------------------

def test_snapshot_sends_current_folded_state_once(mini):
    bundle_dir, work, admitted, stop = _stop_and_admit(mini)
    _leash_act(bundle_dir, record_queue_resume, admitted["task_seq"],
               detail="operator resumed (S-4 test)")
    resume_seq = _rows(bundle_dir, "queue_resume")[0]["seq"]
    transport = _ack_transport(applied=True, last=resume_seq)
    out = send_leash_snapshot(bundle_dir, "muse", leash_url=LEASH_URL,
                              transport=transport)
    assert out["status"] == "crossed" and out["crossing"] == "snapshot"
    assert len(transport.calls) == 1
    assert transport.calls[0]["json_body"] == {
        "model": "muse", "kind": "resume", "journal_seq": resume_seq,
        "reason": "operator resumed (S-4 test)"}
    rows = _rows(bundle_dir, "gaia_leash")
    assert len(rows) == 1
    assert rows[0]["record"]["crossing"] == "snapshot"


def test_snapshot_without_leash_record_refuses_and_invents_nothing(mini):
    bundle_dir, work = mini
    _admit(bundle_dir, work)
    transport = _ack_transport()
    out = send_leash_snapshot(bundle_dir, "muse", leash_url=LEASH_URL,
                              transport=transport)
    assert out["status"] == "refused" and transport.calls == []
    out = send_leash_snapshot(bundle_dir, "ghost", leash_url=LEASH_URL,
                              transport=transport)
    assert out["status"] == "refused" and transport.calls == []
    assert _rows(bundle_dir, "gaia_leash") == []


# -- heartbeat (§12.3) ----------------------------------------------------------

def test_heartbeat_carries_tips_and_journals(mini):
    bundle_dir, work, admitted, stop = _stop_and_admit(mini)
    transport = _ack_transport(applied=False, last=stop["seq"])
    out = send_leash_heartbeat(bundle_dir, "muse",
                               leash_url=LEASH_URL, transport=transport)
    assert out["status"] == "crossed" and out["crossing"] == "heartbeat"
    call = transport.calls[0]
    assert call["url"] == LEASH_URL + "/heartbeat"
    body = call["json_body"]
    assert body["model"] == "muse" and body["leash_tip"] == stop["seq"]
    assert isinstance(body["journal_tip"], int)
    assert body["journal_tip"] >= stop["seq"]
    rows = _rows(bundle_dir, "gaia_leash")
    assert len(rows) == 1
    assert rows[0]["record"]["crossing"] == "heartbeat"
    assert "S-5" in LEASH_CROSSING_LABEL
    assert "UNMEASURED" in LEASH_CROSSING_LABEL


# -- loopback HTTP test double (not a fork) -------------------------------------

def test_loopback_http_double_via_real_http_transport(mini):
    bundle_dir, work, admitted, stop = _stop_and_admit(mini)
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):  # noqa: N802 - stdlib name
            length = int(self.headers.get("Content-Length") or 0)
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(length))
            body = json.dumps({"applied": True,
                               "last_leash_seq": stop["seq"]}).encode()
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
               "/gaia/v1/leash")
        out = sync_seam_leash(bundle_dir, leash_url=url)
    finally:
        server.shutdown()
        server.server_close()
    assert out["status"] == "synced" and out["count"] == 1
    assert out["crossed"][0]["transport"] == "http"
    assert seen["path"] == "/gaia/v1/leash"
    assert seen["body"]["kind"] == "stop"


# -- live fork interop (env-gated; UNMEASURED in this slice) ---------------------

def test_live_fork_leash_crossing_env_gated(mini):
    url = os.environ.get("SI_S4_LEASH_URL")
    if not url:
        pytest.skip(
            "live fork leash crossing: SI_S4_LEASH_URL is unset "
            "— no live gaia-api endpoint was exercised by this "
            "slice; live interop UNMEASURED (the loopback test "
            "double above is the always-on proof)")
    bundle_dir, work, admitted, stop = _stop_and_admit(mini)
    out = sync_seam_leash(bundle_dir, leash_url=url)
    assert out["status"] == "synced"
