"""SI-47: Gaia SSE event spine — journal-derived server-sent events.

The spine (si/gaia/events.py, exposed as GET /events on the
B-5 service) serialises the NEDB journal itself: one journal
row = one SSE event, event id == journal seq, resume via
Last-Event-ID = journal seq. The live tail is poll-the-journal,
labelled as polling — no push, no websocket, no separate event
store, no new journal collection.

Honesty discipline asserted throughout: replay from a seq is
EXACTLY the journal rows after that seq (compared row-for-row
against the raw collections); reading the stream journals
nothing; comments (banner/heartbeat) carry no id and are never
mistaken for history; a malformed resume point is refused, not
silently treated as 0.
"""
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from si.bundle import init_bundle, load_bundle
from si.gaia.corpus import corpus_policy
from si.gaia.events import (
    SSE_RESUME_LABEL,
    SSE_TAIL_LABEL,
    event_name_for_row,
    format_sse,
    parse_resume_seq,
    read_journal_events,
    stream_events,
)
from si.gaia.hearth import journal_residency_observation
from si.gaia.service import make_service_server, submit_task
from si.journal import COLLECTIONS, open_journal
from si.tiny_seed import seed_tiny_bundle


@pytest.fixture()
def mini(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "mini.si"), name="gaia-sse",
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    work = tmp_path / "work"
    work.mkdir()
    return bundle_dir, str(work)


def _journal(bundle_dir):
    return open_journal(bundle_dir, load_bundle(bundle_dir))


def _raw_rows(bundle_dir):
    """{seq: (collection, row)} merged from the raw collections."""
    journal = _journal(bundle_dir)
    try:
        out = {}
        for coll in COLLECTIONS:
            for row in journal.backend.find(coll):
                out[row["seq"]] = (coll, row)
        return out
    finally:
        journal.close()


def _counts(bundle_dir):
    journal = _journal(bundle_dir)
    try:
        return {c: journal.backend.count(c) for c in COLLECTIONS}
    finally:
        journal.close()


def _run_task(bundle_dir, work):
    return submit_task(bundle_dir, {
        "submitter": "sse-test (test submitter)",
        "instruction": "Do the SSE test task",
        "work_root": work,
        "script": [
            {"kind": "tool", "tool": "fs_write",
             "args": {"path": "sse-made.txt", "content": "via sse test\n"},
             "reason": "Write the SSE test task's output"},
            {"kind": "memory",
             "record": {"kind": "task_complete",
                        "detail": "scripted completion report"},
             "reason": "Report completion"},
        ]}, policy=corpus_policy(work))


def _parse_blocks(text):
    events = []
    for block in text.split("\n\n"):
        block = block.strip("\n")
        if not block:
            continue
        ev = {}
        for line in block.splitlines():
            if line.startswith(":"):
                ev.setdefault("comments", []).append(line)
            elif line.startswith("id:"):
                ev["id"] = int(line[3:].strip())
            elif line.startswith("event:"):
                ev["event"] = line[6:].strip()
            elif line.startswith("data:"):
                ev["data"] = json.loads(line[5:].strip())
        events.append(ev)
    return events


# -- the serialisation: one row = one event, id == seq -------------------------

def test_event_ids_are_journal_seqs_in_order(mini):
    bundle_dir, _ = mini
    raw = _raw_rows(bundle_dir)
    journal = _journal(bundle_dir)
    try:
        events = read_journal_events(journal)
    finally:
        journal.close()
    assert [e["id"] for e in events] == sorted(raw)
    for e in events:
        assert e["id"] == e["row"]["seq"]
        coll, row = raw[e["id"]]
        assert e["collection"] == coll
        assert e["row"] == json.loads(json.dumps(row, default=str))


def test_one_row_one_event_no_duplicates_no_new_collection(mini):
    bundle_dir, _ = mini
    assert COLLECTIONS == ("self_edits", "deltas", "checkpoints",
                           "provenance", "experiences")
    counts = _counts(bundle_dir)
    journal = _journal(bundle_dir)
    try:
        events = read_journal_events(journal)
    finally:
        journal.close()
    assert len(events) == sum(counts.values())
    assert len({e["id"] for e in events}) == len(events)


def test_replay_from_seq_is_exactly_the_rows_after_it(mini):
    bundle_dir, work = mini
    _run_task(bundle_dir, work)
    raw = _raw_rows(bundle_dir)
    seqs = sorted(raw)
    mid = seqs[len(seqs) // 2]
    journal = _journal(bundle_dir)
    try:
        events = read_journal_events(journal, after_seq=mid)
    finally:
        journal.close()
    assert [e["id"] for e in events] == [s for s in seqs if s > mid]
    for e in events:
        coll, row = raw[e["id"]]
        assert e["collection"] == coll
        assert e["row"] == json.loads(json.dumps(row, default=str))
    # Resuming at the tip replays nothing; resuming at 0, everything.
    journal = _journal(bundle_dir)
    try:
        assert read_journal_events(journal, after_seq=seqs[-1]) == []
        assert len(read_journal_events(journal, after_seq=0)) == len(seqs)
    finally:
        journal.close()


def test_task_transitions_stream_in_journal_order(mini):
    bundle_dir, work = mini
    _run_task(bundle_dir, work)
    journal = _journal(bundle_dir)
    try:
        events = read_journal_events(journal)
    finally:
        journal.close()
    names = [e["event"] for e in events]
    for expected in ("task_open", "self_edit.tool", "tool_result",
                     "task_stop", "checkpoint"):
        assert expected in names
    i_task = names.index("task_open")
    i_edit = names.index("self_edit.tool")
    i_result = names.index("tool_result")
    i_stop = names.index("task_stop")
    assert i_task < i_edit < i_result < i_stop
    task_event = events[i_task]
    assert task_event["row"]["record"]["submitter"] == \
        "sse-test (test submitter)"


def test_hearth_residency_rows_stream(mini):
    """B-4's journaled residency observations are journal rows,
    so they stream like everything else — no new event source."""
    bundle_dir, _ = mini
    journal = _journal(bundle_dir)
    try:
        ref = journal_residency_observation(journal, {
            "state": "Resident", "previous_state": "Loading",
            "source": "hearth:/residency",
            "transport": "replay/stub — not a real Hearth node"})
        events = read_journal_events(journal, after_seq=ref["seq"] - 1)
    finally:
        journal.close()
    assert [e["id"] for e in events] == [ref["seq"]]
    assert events[0]["event"] == "hearth_residency"
    assert events[0]["row"]["record"]["state"] == "Resident"
    assert events[0]["row"]["record"]["transport"].startswith("replay/stub")


def test_event_name_mapping_is_deterministic():
    cases = [
        ("experiences", {"kind": "task"}, "task_open"),
        ("experiences", {"kind": "task_stop"}, "task_stop"),
        ("experiences", {"kind": "tool_result"}, "tool_result"),
        ("experiences", {"kind": "hearth_residency"}, "hearth_residency"),
        ("experiences", {"kind": "observation"}, "experience.observation"),
        ("self_edits", {"kind": "tool"}, "self_edit.tool"),
        ("self_edits", {"kind": "memory"}, "self_edit.memory"),
        ("checkpoints", {"head": "base"}, "checkpoint"),
        ("deltas", {"kind": "transformer-delta-v1"},
         "delta.transformer-delta-v1"),
        ("deltas", {"id": "d1"}, "delta.recorded"),
        ("provenance", {"relation": "derived-from"},
         "provenance.derived-from"),
        ("provenance", {"child": "x"}, "provenance.edge"),
    ]
    for coll, row, expected in cases:
        assert event_name_for_row(coll, row) == expected


def test_format_sse_shape():
    event = {"id": 42, "event": "task_open", "collection": "experiences",
             "row": {"seq": 42, "kind": "task", "record": {"x": 1}}}
    block = format_sse(event)
    lines = block.splitlines()
    assert lines[0] == "id: 42"
    assert lines[1] == "event: task_open"
    assert lines[2].startswith("data: ")
    assert json.loads(lines[2][5:]) == {
        "collection": "experiences", "row": event["row"]}
    assert block.endswith("\n\n")


# -- resume ----------------------------------------------------------------------

def test_parse_resume_seq_rules():
    assert parse_resume_seq() == 0
    assert parse_resume_seq("17") == 17
    assert parse_resume_seq(None, "9") == 9
    assert parse_resume_seq("17", "9") == 17  # header wins
    assert parse_resume_seq("") == 0
    for bad in ("banana", "-3", "1.5"):
        with pytest.raises(ValueError):
            parse_resume_seq(bad)


def test_stream_replay_only_matches_read_and_terminates(mini):
    bundle_dir, work = mini
    _run_task(bundle_dir, work)
    journal = _journal(bundle_dir)
    try:
        expected = read_journal_events(journal)
        blocks = list(stream_events(journal, follow=False))
    finally:
        journal.close()
    parsed = _parse_blocks("".join(blocks))
    assert [p["id"] for p in parsed] == [e["id"] for e in expected]
    assert [p["event"] for p in parsed] == [e["event"] for e in expected]


def test_stream_resume_after_seq(mini):
    bundle_dir, work = mini
    _run_task(bundle_dir, work)
    raw = _raw_rows(bundle_dir)
    mid = sorted(raw)[3]
    journal = _journal(bundle_dir)
    try:
        blocks = list(stream_events(journal, after_seq=mid, follow=False))
    finally:
        journal.close()
    parsed = _parse_blocks("".join(blocks))
    assert [p["id"] for p in parsed] == [s for s in sorted(raw) if s > mid]


# -- live tail (poll-the-journal, labelled) ---------------------------------------

def test_live_tail_yields_a_row_journaled_after_replay(mini):
    bundle_dir, _ = mini
    journal = _journal(bundle_dir)
    writer = _journal(bundle_dir)
    try:
        baseline = read_journal_events(journal)
        gen = stream_events(journal, follow=True, poll_interval=0.02,
                            heartbeat=False)
        replayed = [next(gen) for _ in baseline]
        assert [ _parse_blocks(b)[0]["id"] for b in replayed] == \
            [e["id"] for e in baseline]
        writer.record_experience(
            "observation", {"kind": "observation", "note": "tail probe"},
            "")
        block = next(gen)
        gen.close()
        parsed = _parse_blocks(block)
        assert len(parsed) == 1
        assert parsed[0]["id"] == baseline[-1]["id"] + 1
        assert parsed[0]["event"] == "experience.observation"
        assert parsed[0]["data"]["row"]["record"]["note"] == "tail probe"
    finally:
        journal.close()
        writer.close()


def test_live_tail_heartbeat_is_a_comment_not_an_event(mini):
    bundle_dir, _ = mini
    journal = _journal(bundle_dir)
    try:
        baseline = read_journal_events(journal)
        blocks = list(stream_events(journal, follow=True,
                                    poll_interval=0.01, max_polls=3))
    finally:
        journal.close()
    parsed = _parse_blocks("".join(blocks))
    events = [p for p in parsed if "id" in p]
    comments = [p for p in parsed if "id" not in p]
    assert [p["id"] for p in events] == [e["id"] for e in baseline]
    assert len(comments) == 1
    assert comments[0]["comments"][0].startswith(": heartbeat")
    assert "poll-the-journal" in comments[0]["comments"][0]


def test_live_tail_should_stop_ends_cleanly(mini):
    bundle_dir, _ = mini
    journal = _journal(bundle_dir)
    try:
        baseline = read_journal_events(journal)
        blocks = list(stream_events(journal, follow=True,
                                    poll_interval=0.01,
                                    should_stop=lambda: True,
                                    heartbeat=False))
    finally:
        journal.close()
    assert len(blocks) == len(baseline)


def test_reading_the_stream_journals_nothing(mini):
    bundle_dir, work = mini
    _run_task(bundle_dir, work)
    before = _counts(bundle_dir)
    journal = _journal(bundle_dir)
    try:
        list(stream_events(journal, follow=False))
        list(stream_events(journal, follow=True, poll_interval=0.01,
                           max_polls=2))
    finally:
        journal.close()
    assert _counts(bundle_dir) == before


def test_honesty_labels_say_polling_and_resume():
    assert "poll-the-journal" in SSE_TAIL_LABEL
    assert "no websocket" in SSE_TAIL_LABEL
    assert "Last-Event-ID" in SSE_RESUME_LABEL
    assert "journal seq" in SSE_RESUME_LABEL


# -- HTTP surface -------------------------------------------------------------------

class _Http:
    def __init__(self, bundle_dir, work):
        self.server = make_service_server(
            bundle_dir, host="127.0.0.1", port=0,
            policy=corpus_policy(work))
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=10)

    def get(self, path, headers=None):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            headers=headers or {})
        return urllib.request.urlopen(req, timeout=15)


@pytest.fixture()
def http(mini):
    bundle_dir, work = mini
    _run_task(bundle_dir, work)
    srv = _Http(bundle_dir, work)
    yield srv, bundle_dir
    srv.close()


def test_http_events_replay_only(http):
    srv, bundle_dir = http
    raw = _raw_rows(bundle_dir)
    with srv.get("/events?follow=0") as resp:
        assert resp.status == 200
        assert resp.headers.get_content_type() == "text/event-stream"
        assert resp.headers.get("Cache-Control") == "no-cache"
        body = resp.read().decode("utf-8")
    assert body.startswith(": gaia-events (SI-47)")
    parsed = _parse_blocks(body)
    events = [p for p in parsed if "id" in p]
    assert [p["id"] for p in events] == sorted(raw)


def test_http_resume_via_last_event_id_header(http):
    srv, bundle_dir = http
    raw = _raw_rows(bundle_dir)
    mid = sorted(raw)[2]
    with srv.get("/events?follow=0",
                 headers={"Last-Event-ID": str(mid)}) as resp:
        parsed = _parse_blocks(resp.read().decode("utf-8"))
    events = [p for p in parsed if "id" in p]
    assert [p["id"] for p in events] == [s for s in sorted(raw) if s > mid]


def test_http_resume_via_after_query_and_header_precedence(http):
    srv, bundle_dir = http
    raw = _raw_rows(bundle_dir)
    seqs = sorted(raw)
    with srv.get(f"/events?follow=0&after={seqs[1]}") as resp:
        parsed = _parse_blocks(resp.read().decode("utf-8"))
    assert [p["id"] for p in parsed if "id" in p] == seqs[2:]
    with srv.get(f"/events?follow=0&after={seqs[0]}",
                 headers={"Last-Event-ID": str(seqs[2])}) as resp:
        parsed = _parse_blocks(resp.read().decode("utf-8"))
    assert [p["id"] for p in parsed if "id" in p] == seqs[3:]


def test_http_invalid_resume_point_is_refused(http):
    srv, _ = http
    with pytest.raises(urllib.error.HTTPError) as ei:
        srv.get("/events?follow=0", headers={"Last-Event-ID": "banana"})
    assert ei.value.code == 400
    assert "resume seq" in ei.value.read().decode("utf-8")


def test_http_description_lists_events_with_labels(http):
    srv, _ = http
    with srv.get("/") as resp:
        desc = json.loads(resp.read().decode("utf-8"))
    assert "GET /events" in desc["endpoints"]
    assert "poll-the-journal" in desc["events_tail"]
    assert "Last-Event-ID" in desc["events_resume"]
    assert "no authentication" in desc["auth"]


def test_http_live_tail_streams_a_newly_journaled_row(http):
    srv, bundle_dir = http
    raw = _raw_rows(bundle_dir)
    tip = max(raw)
    resp = srv.get("/events")
    try:
        assert resp.headers.get_content_type() == "text/event-stream"
        seen = []
        deadline = time.time() + 20
        # Consume banner + the full replay first.
        while time.time() < deadline and (not seen or seen[-1] < tip):
            line = resp.readline().decode("utf-8")
            assert line, "stream ended during replay"
            if line.startswith("id:"):
                seen.append(int(line[3:].strip()))
        assert seen == sorted(raw)
        writer = _journal(bundle_dir)
        try:
            writer.record_experience(
                "observation",
                {"kind": "observation", "note": "live tail probe"}, "")
        finally:
            writer.close()
        new_id = None
        while time.time() < deadline and new_id is None:
            line = resp.readline().decode("utf-8")
            assert line, "stream ended before the live row arrived"
            if line.startswith("id:"):
                new_id = int(line[3:].strip())
        assert new_id == tip + 1
    finally:
        resp.close()


def test_http_replay_journals_nothing(http):
    srv, bundle_dir = http
    before = _counts(bundle_dir)
    with srv.get("/events?follow=0") as resp:
        resp.read()
    assert _counts(bundle_dir) == before
