"""SI-60 / S-6 (G-9 §4.5/§15): Gaia seam SI journal-event
forwarder, SI side.

Pins: build_stream_event produces the StreamEvent::from_si
shape (source=si, seq = the SI journal seq, kind = the SI
event name, collection, the full row verbatim, row_kind, facts
with queue_state only when present, model/task only when
derivable, and NEVER a hash key); the forwarder pushes one
event per POST in ascending seq order with exact seq fidelity
to the operator-supplied ingest URL (whole, no path appended);
the caller-held cursor advances ONLY after an HTTP 200, so a
mid-stream transport failure leaves the cursor at the last
acked seq and a later forward from it pushes exactly the
remainder (including events journaled while the fork was
unreachable); an HTTP 409 duplicate/refusal raises
SeamProtocolError — a protocol failure, not a retry — with no
further pushes and no cursor advance past it; other non-200
answers are forward-failed with the cursor unadvanced and a
free retry; no endpoint is a refusal that pushes nothing; the
forwarder journals nothing; the real HttpTransport runs
against a loopback HTTP test double implementing the
documented ingest contract (200, 409 on duplicate seq) — not
a fork; live interop is env-gated on SI_S6_INGEST_URL and
UNMEASURED.
"""
import copy
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from si.bundle import init_bundle, load_bundle
from si.gaia.corpus import corpus_policy
from si.gaia.events import read_journal_events
from si.gaia.hearth import ReplayTransport
from si.gaia.seam_events import (
    EVENT_FORWARD_LABEL,
    INGEST_PATH,
    SeamProtocolError,
    build_stream_event,
    forward_journal_events,
    pending_event_seqs,
)
from si.gaia.service import ServiceState, forward_events, submit_proposal
from si.journal import open_journal
from si.tiny_seed import seed_tiny_bundle

INGEST_URL = "replay://fork/gaia/v1/events/ingest"
WATCHER = {
    "model": "muse",
    "tool": "hearth.restart",
    "args": {"model": "muse",
             "spine_event": {"hash": "hash-s6-fixture", "seq": 51}},
    "reason": "observed loss at seq 51",
    "caused_by": {"hash": "hash-s6-fixture", "seq": 51},
}
WATCHER2 = {
    **WATCHER,
    "args": {"model": "muse",
             "spine_event": {"hash": "hash-s6-second", "seq": 52}},
    "reason": "observed loss at seq 52",
    "caused_by": {"hash": "hash-s6-second", "seq": 52},
}
ADMIT = {"tool_class_policy": {
    "read": "admit", "write": "admit",
    "execute": "admit", "external": "admit"}}


@pytest.fixture()
def mini(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "mini.si"),
                             name="gaia-s6",
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    work = tmp_path / "work"
    work.mkdir()
    return bundle_dir, str(work)


def _events(bundle_dir, cursor=0):
    journal = open_journal(bundle_dir)
    try:
        return read_journal_events(journal, after_seq=cursor)
    finally:
        journal.close()


def _ok_transport(bundle_dir, cursor=0):
    return ReplayTransport(
        [(200, {"status": "ok"})] * len(_events(bundle_dir, cursor)))


# -- build_stream_event (§4.5) ---------------------------------------------------

def test_build_stream_event_shape_from_journal(mini):
    bundle_dir, work = mini
    events = _events(bundle_dir)
    assert events  # the seeded miniature journals rows
    for event in events:
        payload = build_stream_event(event)
        assert payload["source"] == "si"
        # seq is the SI journal seq (the event id), verbatim.
        assert payload["seq"] == event["id"]
        assert payload["kind"] == event["event"]
        assert payload["collection"] == event["collection"]
        assert payload["row"] == event["row"]  # full row, verbatim
        assert payload["row_kind"] == event["row"].get("kind")
        assert "hash" not in payload  # hearth-only per §4.5
        facts = payload["facts"]
        assert facts["collection"] == event["collection"]
        assert facts["row"] == event["row"]
        if "queue_state" in event:
            assert facts["queue_state"] == event["queue_state"]
        else:
            assert "queue_state" not in facts


def test_build_stream_event_derived_and_omitted_keys():
    row = {"seq": 42, "kind": "tool",
           "intent": {"tool": "hearth.restart",
                      "args": {"model": "muse"},
                      "parents": ["task@7"], "source_seqs": [7]},
           "gate_verdict": "ADMIT"}
    event = {"id": 42, "event": "self_edit.tool",
             "collection": "self_edits", "row": row,
             "queue_state": "queued"}
    payload = build_stream_event(event)
    assert payload["model"] == "muse"  # intent args, derivable
    assert payload["task"] == "task@7"  # intent parent, derivable
    assert payload["facts"]["queue_state"] == "queued"
    assert payload["row_kind"] == "tool"
    # Nothing derivable -> both keys omitted, never invented.
    bare = build_stream_event({
        "id": 3, "event": "checkpoint", "collection": "checkpoints",
        "row": {"seq": 3}})
    assert "model" not in bare and "task" not in bare
    assert bare["row_kind"] is None
    assert "queue_state" not in bare["facts"]


# -- in-order push with exact seq fidelity ----------------------------------------

def test_forward_in_order_exact_fidelity(mini):
    bundle_dir, work = mini
    events = _events(bundle_dir)
    seqs = [e["id"] for e in events]
    assert pending_event_seqs(bundle_dir) == seqs
    transport = _ok_transport(bundle_dir)
    out = forward_journal_events(bundle_dir, ingest_url=INGEST_URL,
                                 transport=transport)
    assert out["status"] == "forwarded"
    assert out["pushed"] == seqs
    assert out["cursor"] == seqs[-1]
    assert out["count"] == len(seqs)
    assert out["label"] == EVENT_FORWARD_LABEL
    assert "UNMEASURED" in EVENT_FORWARD_LABEL
    assert INGEST_PATH == "/gaia/v1/events/ingest"
    # One POST per event, to the operator URL whole (no path
    # appended), payloads exactly build_stream_event's.
    assert len(transport.calls) == len(events)
    for call, event in zip(transport.calls, events):
        assert call["method"] == "POST"
        assert call["url"] == INGEST_URL
        assert call["json_body"] == build_stream_event(event)
    # The forwarder journals NOTHING: the journal is unchanged.
    assert _events(bundle_dir) == events


def test_forward_limit_pushes_window_only(mini):
    bundle_dir, work = mini
    seqs = pending_event_seqs(bundle_dir)
    assert len(seqs) >= 2
    transport = ReplayTransport([(200, {})] * 2)
    out = forward_journal_events(bundle_dir, ingest_url=INGEST_URL,
                                 transport=transport, limit=2)
    assert out["pushed"] == seqs[:2]
    assert out["cursor"] == seqs[1]
    assert pending_event_seqs(bundle_dir, cursor=out["cursor"]) == \
        seqs[2:]


# -- cursor advances only after 200; resume pushes exactly the remainder ---------

def test_cursor_advances_only_after_200_and_resume(mini):
    bundle_dir, work = mini
    seqs = pending_event_seqs(bundle_dir)
    assert len(seqs) >= 3
    transport = ReplayTransport(
        [(200, {}), (200, {}), RuntimeError("fork unreachable")])
    out = forward_journal_events(bundle_dir, ingest_url=INGEST_URL,
                                 transport=transport)
    assert out["status"] == "forward-failed"
    assert out["pushed"] == seqs[:2]
    assert out["cursor"] == seqs[1]  # last ACKED seq only
    assert "fork unreachable" in out["failure"]
    assert len(transport.calls) == 3  # stopped at the failure

    # Events journaled while the fork was unreachable (a new
    # seam proposal) are pushed later, from the cursor.
    ack = submit_proposal(bundle_dir, copy.deepcopy(WATCHER),
                          policy=corpus_policy(work, ADMIT))
    assert ack["status"] == "acked"
    remaining = pending_event_seqs(bundle_dir, cursor=out["cursor"])
    assert remaining[:len(seqs) - 2] == seqs[2:]
    assert len(remaining) > len(seqs) - 2  # the new rows joined
    transport2 = ReplayTransport([(200, {})] * len(remaining))
    out2 = forward_journal_events(
        bundle_dir, ingest_url=INGEST_URL, transport=transport2,
        cursor=out["cursor"])
    assert out2["status"] == "forwarded"
    assert out2["pushed"] == remaining
    assert out2["cursor"] == remaining[-1]
    assert pending_event_seqs(bundle_dir, cursor=out2["cursor"]) == []


def test_non_200_500_forward_failed_cursor_unadvanced(mini):
    bundle_dir, work = mini
    seqs = pending_event_seqs(bundle_dir)
    out = forward_journal_events(
        bundle_dir, ingest_url=INGEST_URL,
        transport=ReplayTransport([(500, {"error": "no"})]))
    assert out["status"] == "forward-failed"
    assert out["http_status"] == 500
    assert out["cursor"] == 0 and out["pushed"] == []
    # A retry from the returned cursor is free (history is the
    # journal) and pushes the whole window.
    out2 = forward_journal_events(
        bundle_dir, ingest_url=INGEST_URL,
        transport=_ok_transport(bundle_dir), cursor=out["cursor"])
    assert out2["pushed"] == seqs and out2["cursor"] == seqs[-1]


# -- duplicate / refusal: a protocol failure, not a retry -------------------------

def test_duplicate_409_raises_protocol_error(mini):
    bundle_dir, work = mini
    seqs = pending_event_seqs(bundle_dir)
    assert len(seqs) >= 3
    transport = ReplayTransport(
        [(200, {}), (409, {"error": "duplicate seq"}), (200, {})])
    with pytest.raises(SeamProtocolError) as excinfo:
        forward_journal_events(bundle_dir, ingest_url=INGEST_URL,
                               transport=transport)
    err = excinfo.value
    assert err.seq == seqs[1]
    assert err.http_status == 409
    assert err.response == {"error": "duplicate seq"}
    assert err.cursor == seqs[0]  # NOT advanced past the dup
    assert err.pushed == seqs[:1]
    # Nothing after the duplicate was pushed.
    assert len(transport.calls) == 2


def test_no_endpoint_is_refused(mini):
    bundle_dir, work = mini
    out = forward_journal_events(bundle_dir)
    assert out["status"] == "refused" and out["journaled"] is False
    assert out["pushed"] == [] and out["count"] == 0
    assert "UNMEASURED" in out["problems"][0]
    assert "no default fork endpoint" in out["problems"][0]


# -- loopback HTTP test double (not a fork) ----------------------------------------

def test_loopback_ingest_double_via_real_http_transport(mini):
    bundle_dir, work = mini
    events = _events(bundle_dir)
    seqs = [e["id"] for e in events]
    received = []
    seen = set()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):  # noqa: N802 - stdlib name
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length))
            received.append(payload)
            if payload["seq"] in seen:
                code, body = 409, {"error": "duplicate seq"}
            else:
                seen.add(payload["seq"])
                code, body = 200, {"status": "ok"}
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = (f"http://127.0.0.1:{server.server_address[1]}"
               + INGEST_PATH)
        # Real HttpTransport (no transport injected).
        out = forward_journal_events(bundle_dir, ingest_url=url)
        assert out["status"] == "forwarded"
        assert out["transport"] == "http"
        assert out["pushed"] == seqs and out["cursor"] == seqs[-1]
        assert received == [build_stream_event(e) for e in events]
        # A re-push from 0 hits the double's duplicate refusal
        # on the FIRST event: protocol failure, one request.
        received.clear()
        with pytest.raises(SeamProtocolError) as excinfo:
            forward_journal_events(bundle_dir, ingest_url=url)
        assert excinfo.value.seq == seqs[0]
        assert excinfo.value.cursor == 0
        assert len(received) == 1
    finally:
        server.shutdown()
        server.server_close()


# -- the service wrapper ------------------------------------------------------------

def test_service_wrapper_forward_events(mini):
    bundle_dir, work = mini
    seqs = pending_event_seqs(bundle_dir)
    transport = _ok_transport(bundle_dir)
    out = forward_events(bundle_dir, ingest_url=INGEST_URL,
                         transport=transport)
    assert out["status"] == "forwarded" and out["pushed"] == seqs
    description = ServiceState(bundle_dir).description()
    assert description["seam_event_forward"] == EVENT_FORWARD_LABEL


# -- live fork interop (env-gated; UNMEASURED in this slice) ------------------------

def test_live_fork_ingest_env_gated(mini):
    url = os.environ.get("SI_S6_INGEST_URL")
    if not url:
        pytest.skip(
            "live fork ingest: SI_S6_INGEST_URL is unset — no "
            "live gaia-api endpoint was exercised by this slice; "
            "live interop UNMEASURED (the loopback test double "
            "above is the always-on proof)")
    bundle_dir, work = mini
    out = forward_journal_events(bundle_dir, ingest_url=url, limit=1)
    assert out["status"] == "forwarded"
