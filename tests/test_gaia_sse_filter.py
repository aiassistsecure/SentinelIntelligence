"""SI-48: Gaia SSE filtered subscriptions + reference consumer.

SI-47 §2.3 deferred server-side filtering to a future slice
under one rule: any future filter must stay a projection of
the same stream, never a second store. SI-48 is that slice
(si/gaia/events.py EventFilter, GET /events query filters in
si/gaia/service.py, and the reference consumer in
si/gaia/events_client.py).

Honesty discipline asserted throughout: a filtered stream is
EXACTLY the unfiltered stream's events that match (compared
against the raw journal collections, row for row); ids remain
journal seqs — never renumbered, so gaps are the filtered-out
seqs; resume semantics are unchanged under filtering;
malformed filters fail closed (ValueError / HTTP 400) like
malformed resume points; filtering journals nothing; and the
unfiltered stream is byte-identical to SI-47.
"""
import json
import threading
import urllib.error
import urllib.request

import pytest

from si.bundle import init_bundle, load_bundle
from si.cli import main as cli_main
from si.gaia.corpus import corpus_policy
from si.gaia.events import (
    EventFilter,
    apply_event_filter,
    parse_event_filter,
    read_journal_events,
    stream_events,
    task_member_seqs,
)
from si.gaia.events_client import (
    build_events_url,
    collect_events,
    events_from_text,
    fetch_events,
    parse_sse_block,
    verify_events_against_journal,
)
from si.gaia.service import build_provenance_trail, make_service_server, submit_task
from si.journal import COLLECTIONS, open_journal
from si.tiny_seed import seed_tiny_bundle


@pytest.fixture()
def mini(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "mini.si"), name="gaia-sse-filter",
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    work = tmp_path / "work"
    work.mkdir()
    return bundle_dir, str(work)


def _journal(bundle_dir):
    return open_journal(bundle_dir, load_bundle(bundle_dir))


def _raw_rows(bundle_dir):
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


def _run_task(bundle_dir, work, tag):
    return submit_task(bundle_dir, {
        "submitter": f"filter-test {tag}",
        "instruction": f"Do the filter test task {tag}",
        "work_root": work,
        "script": [
            {"kind": "tool", "tool": "fs_write",
             "args": {"path": f"filter-{tag}.txt",
                      "content": f"via filter test {tag}\n"},
             "reason": f"Write the filter test task {tag} output"},
            {"kind": "memory",
             "record": {"kind": "task_complete",
                        "detail": f"scripted completion {tag}"},
             "reason": "Report completion"},
        ]}, policy=corpus_policy(work))


def _all_events(bundle_dir):
    journal = _journal(bundle_dir)
    try:
        return read_journal_events(journal)
    finally:
        journal.close()


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


# -- filter exactness vs the raw journal ---------------------------------------

def test_collection_filter_is_exact_vs_raw_journal(mini):
    bundle_dir, work = mini
    _run_task(bundle_dir, work, "a")
    raw = _raw_rows(bundle_dir)
    journal = _journal(bundle_dir)
    try:
        for coll in COLLECTIONS:
            got = read_journal_events(
                journal, event_filter=EventFilter(collections={coll}))
            assert [e["id"] for e in got] == \
                [s for s in sorted(raw) if raw[s][0] == coll]
            for e in got:
                assert e["row"] == json.loads(
                    json.dumps(raw[e["id"]][1], default=str))
    finally:
        journal.close()


def test_event_name_filter_is_exact(mini):
    bundle_dir, work = mini
    _run_task(bundle_dir, work, "a")
    all_events = _all_events(bundle_dir)
    for name in ("task_open", "tool_result", "checkpoint"):
        got = apply_event_filter(
            all_events, EventFilter(event_names={name}))
        assert [e["id"] for e in got] == \
            [e["id"] for e in all_events if e["event"] == name]
        assert got  # each of these names really occurs in this journal


def test_kind_filter_is_exact_and_kindless_rows_never_match(mini):
    bundle_dir, work = mini
    _run_task(bundle_dir, work, "a")
    all_events = _all_events(bundle_dir)
    got = apply_event_filter(all_events, EventFilter(kinds={"tool"}))
    assert [e["id"] for e in got] == \
        [e["id"] for e in all_events if e["row"].get("kind") == "tool"]
    # checkpoints / provenance rows carry no kind: a kind filter
    # never invents one for them.
    assert all(e["collection"] not in ("checkpoints", "provenance")
               for e in got)


def test_values_within_a_dimension_are_alternatives(mini):
    bundle_dir, work = mini
    _run_task(bundle_dir, work, "a")
    all_events = _all_events(bundle_dir)
    got = apply_event_filter(
        all_events, EventFilter(collections={"experiences", "deltas"}))
    assert [e["id"] for e in got] == \
        [e["id"] for e in all_events
         if e["collection"] in ("experiences", "deltas")]
    # Repeated params and comma-separated values parse identically.
    a = parse_event_filter({"collection": ["experiences", "deltas"]})
    b = parse_event_filter({"collection": ["experiences,deltas"]})
    assert a == b == EventFilter(collections={"experiences", "deltas"})


def test_dimensions_combine_conjunctively(mini):
    bundle_dir, work = mini
    _run_task(bundle_dir, work, "a")
    all_events = _all_events(bundle_dir)
    got = apply_event_filter(all_events, EventFilter(
        collections={"experiences"}, kinds={"task"}))
    assert [e["event"] for e in got] == ["task_open"]
    got = apply_event_filter(all_events, EventFilter(
        collections={"experiences"}, kinds={"tool"}))
    assert got == []  # no experiences row has kind "tool"


def test_task_filter_matches_the_b5_trail_chain(mini):
    bundle_dir, work = mini
    res_a = _run_task(bundle_dir, work, "a")
    res_b = _run_task(bundle_dir, work, "b")
    task_a = res_a["result"]["task_seq"]
    task_b = res_b["result"]["task_seq"]
    assert task_a != task_b
    journal = _journal(bundle_dir)
    try:
        all_events = read_journal_events(journal)
        for task_seq in (task_a, task_b):
            trail = build_provenance_trail(journal, task_seq)
            expected = {task_seq}
            if trail["task_stop"] is not None:
                expected.add(trail["task_stop"]["seq"])
            for step in trail["steps"]:
                expected.add(step["seq"])
                expected.update(p["seq"] for p in step["provenance"])
                expected.update(o["seq"] for o in step["outcomes"])
            got = read_journal_events(
                journal, event_filter=EventFilter(task_seq=task_seq))
            assert [e["id"] for e in got] == sorted(expected)
            # Checkpoints are shared bookkeeping, not task steps.
            assert all(e["collection"] != "checkpoints" for e in got)
        members_a = task_member_seqs(all_events, task_a)
        members_b = task_member_seqs(all_events, task_b)
        assert not (members_a & members_b)
    finally:
        journal.close()


def test_task_filter_unknown_or_nontask_seq_is_empty(mini):
    bundle_dir, work = mini
    _run_task(bundle_dir, work, "a")
    all_events = _all_events(bundle_dir)
    nontask = next(e["id"] for e in all_events
                   if e["event"] == "checkpoint")
    journal = _journal(bundle_dir)
    try:
        assert read_journal_events(
            journal, event_filter=EventFilter(task_seq=999999)) == []
        assert read_journal_events(
            journal, event_filter=EventFilter(task_seq=nontask)) == []
    finally:
        journal.close()


def test_filtered_ids_are_journal_seqs_never_renumbered(mini):
    bundle_dir, work = mini
    _run_task(bundle_dir, work, "a")
    all_events = _all_events(bundle_dir)
    got = apply_event_filter(all_events, EventFilter(kinds={"task"}))
    assert [e["id"] for e in got] == \
        [e["id"] for e in all_events if e["row"].get("kind") == "task"]
    # The surviving ids keep their original seqs: there is a gap
    # where filtered-out rows sit, and the filter says so.
    assert got[0]["id"] > 1
    assert "gaps are filtered-out seqs" in \
        EventFilter(kinds={"task"}).describe()


# -- parsing: malformed filters fail closed -------------------------------------

def test_parse_event_filter_rules():
    assert parse_event_filter({}).is_empty()
    assert parse_event_filter(
        {"after": ["5"], "follow": ["0"]}).is_empty()  # not filter params
    f = parse_event_filter({"collection": ["experiences"],
                            "event": ["task_open"],
                            "kind": ["task"], "task": ["12"]})
    assert f.collections == frozenset({"experiences"})
    assert f.event_names == frozenset({"task_open"})
    assert f.kinds == frozenset({"task"})
    assert f.task_seq == 12
    for bad in ({"collection": ["bogus"]},
                {"collection": ["Experiences"]},
                {"collection": [""]},
                {"event": ["task open"]},
                {"event": ["a/b"]},
                {"kind": [""]},
                {"task": ["banana"]},
                {"task": ["-3"]},
                {"task": ["1.5"]},
                {"task": ["3", "4"]}):
        with pytest.raises(ValueError):
            parse_event_filter(bad)


# -- resume under filtering -------------------------------------------------------

def test_filtered_resume_from_a_seq(mini):
    bundle_dir, work = mini
    _run_task(bundle_dir, work, "a")
    _run_task(bundle_dir, work, "b")
    all_events = _all_events(bundle_dir)
    filt = EventFilter(collections={"experiences"})
    full = apply_event_filter(all_events, filt)
    mid = all_events[len(all_events) // 2]["id"]
    journal = _journal(bundle_dir)
    try:
        got = read_journal_events(journal, after_seq=mid,
                                  event_filter=filt)
    finally:
        journal.close()
    assert [e["id"] for e in got] == \
        [e["id"] for e in full if e["id"] > mid]


def test_task_filter_resume_derives_membership_from_full_journal(mini):
    """Resuming past the task entry itself must not lose the
    chained steps: membership comes from the full journal, not
    from the post-resume window."""
    bundle_dir, work = mini
    res = _run_task(bundle_dir, work, "a")
    task_seq = res["result"]["task_seq"]
    filt = EventFilter(task_seq=task_seq)
    journal = _journal(bundle_dir)
    try:
        full = read_journal_events(journal, event_filter=filt)
        resumed = read_journal_events(journal, after_seq=task_seq,
                                      event_filter=filt)
    finally:
        journal.close()
    assert [e["id"] for e in resumed] == \
        [e["id"] for e in full if e["id"] > task_seq]
    assert resumed  # the steps after the task entry still arrive


def test_stream_filtered_replay_matches_read(mini):
    bundle_dir, work = mini
    _run_task(bundle_dir, work, "a")
    filt = EventFilter(event_names={"task_open", "task_stop"})
    journal = _journal(bundle_dir)
    try:
        expected = read_journal_events(journal, event_filter=filt)
        blocks = list(stream_events(journal, follow=False,
                                    event_filter=filt))
    finally:
        journal.close()
    parsed = _parse_blocks("".join(blocks))
    assert [p["id"] for p in parsed] == [e["id"] for e in expected]
    assert [p["event"] for p in parsed] == ["task_open", "task_stop"]


def test_unfiltered_stream_is_byte_identical_to_si47(mini):
    bundle_dir, work = mini
    _run_task(bundle_dir, work, "a")
    journal = _journal(bundle_dir)
    try:
        plain = list(stream_events(journal, follow=False))
        via_none = list(stream_events(journal, follow=False,
                                      event_filter=None))
        via_empty = list(stream_events(journal, follow=False,
                                       event_filter=EventFilter()))
    finally:
        journal.close()
    assert plain == via_none == via_empty


def test_filtered_live_tail_yields_only_matching_rows(mini):
    bundle_dir, _ = mini
    journal = _journal(bundle_dir)
    writer = _journal(bundle_dir)
    try:
        filt = EventFilter(event_names={"experience.observation"})
        gen = stream_events(journal, follow=True, poll_interval=0.02,
                            heartbeat=False, event_filter=filt)
        # Replay under this filter is empty (no observations yet):
        # the first yielded block must be the live observation,
        # even though a checkpoint (filtered out) is journaled
        # immediately before it — the cursor walks over it.
        writer.checkpoint("base", "m", "filtered-out probe")
        writer.record_experience(
            "observation", {"kind": "observation", "note": "tail"}, "")
        expected_seq = writer.tip()[0]
        block = next(gen)
        gen.close()
        parsed = _parse_blocks(block)
        assert len(parsed) == 1
        assert parsed[0]["id"] == expected_seq
        assert parsed[0]["event"] == "experience.observation"
    finally:
        journal.close()
        writer.close()


def test_filtering_journals_nothing(mini):
    bundle_dir, work = mini
    _run_task(bundle_dir, work, "a")
    before = _counts(bundle_dir)
    filt = EventFilter(collections={"experiences"}, kinds={"task"})
    journal = _journal(bundle_dir)
    try:
        read_journal_events(journal, event_filter=filt)
        list(stream_events(journal, follow=False, event_filter=filt))
        list(stream_events(journal, follow=True, poll_interval=0.01,
                           max_polls=2, event_filter=filt))
    finally:
        journal.close()
    assert _counts(bundle_dir) == before


# -- HTTP surface -----------------------------------------------------------------

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
    res = _run_task(bundle_dir, work, "a")
    srv = _Http(bundle_dir, work)
    yield srv, bundle_dir, res["result"]["task_seq"]
    srv.close()


def test_http_filtered_replay_is_exact(http):
    srv, bundle_dir, _ = http
    raw = _raw_rows(bundle_dir)
    with srv.get("/events?follow=0&collection=experiences") as resp:
        body = resp.read().decode("utf-8")
    assert body.startswith(": gaia-events (SI-48)")
    assert "filtered-out seqs" in body
    events = [p for p in _parse_blocks(body) if "id" in p]
    assert [p["id"] for p in events] == \
        [s for s in sorted(raw) if raw[s][0] == "experiences"]
    with srv.get("/events?follow=0&collection=experiences&kind=task") as resp:
        events = [p for p in _parse_blocks(resp.read().decode("utf-8"))
                  if "id" in p]
    assert [p["event"] for p in events] == ["task_open"]


def test_http_task_filter_and_header_resume(http):
    srv, bundle_dir, task_seq = http
    with srv.get(f"/events?follow=0&task={task_seq}") as resp:
        events = [p for p in _parse_blocks(resp.read().decode("utf-8"))
                  if "id" in p]
    ids = [p["id"] for p in events]
    assert ids[0] == task_seq
    assert "task_open" in [p["event"] for p in events]
    assert "task_stop" in [p["event"] for p in events]
    mid = ids[len(ids) // 2]
    with srv.get(f"/events?follow=0&task={task_seq}",
                 headers={"Last-Event-ID": str(mid)}) as resp:
        events = [p for p in _parse_blocks(resp.read().decode("utf-8"))
                  if "id" in p]
    assert [p["id"] for p in events] == [i for i in ids if i > mid]


def test_http_malformed_filters_are_400(http):
    srv, _, _ = http
    for path in ("/events?follow=0&collection=bogus",
                 "/events?follow=0&task=banana",
                 "/events?follow=0&kind=",
                 "/events?follow=0&collection=",
                 "/events?follow=0&task=-3"):
        with pytest.raises(urllib.error.HTTPError) as ei:
            srv.get(path)
        assert ei.value.code == 400, path
        assert "filter" in ei.value.read().decode("utf-8")


def test_http_unfiltered_is_the_si47_stream(http):
    srv, bundle_dir, _ = http
    raw = _raw_rows(bundle_dir)
    with srv.get("/events?follow=0") as resp:
        body = resp.read().decode("utf-8")
    assert body.startswith(": gaia-events (SI-47)")
    events = [p for p in _parse_blocks(body) if "id" in p]
    assert [p["id"] for p in events] == sorted(raw)


def test_http_filtered_read_journals_nothing(http):
    srv, bundle_dir, task_seq = http
    before = _counts(bundle_dir)
    with srv.get(f"/events?follow=0&task={task_seq}") as resp:
        resp.read()
    with srv.get("/events?follow=0&kind=tool") as resp:
        resp.read()
    assert _counts(bundle_dir) == before


def test_http_description_carries_the_filter_label(http):
    srv, _, _ = http
    with srv.get("/") as resp:
        desc = json.loads(resp.read().decode("utf-8"))
    assert "never renumbered" in desc["events_filter"]
    assert "task=" in desc["endpoints"]["GET /events"]


# -- the reference consumer --------------------------------------------------------

def test_consumer_collect_filter_and_resume(http):
    srv, bundle_dir, _ = http
    base = f"http://127.0.0.1:{srv.port}"
    filt = EventFilter(collections={"experiences"})
    got = collect_events(base, event_filter=filt)
    expected = apply_event_filter(_all_events(bundle_dir), filt)
    assert [e["id"] for e in got] == [e["id"] for e in expected]
    assert all(e["collection"] == "experiences" for e in got)
    # Resume: from the last received id, nothing remains; from
    # the one before it, exactly the tail arrives.
    assert collect_events(base, after_seq=got[-1]["id"],
                          event_filter=filt) == []
    tail = collect_events(base, after_seq=got[-2]["id"],
                          event_filter=filt)
    assert [e["id"] for e in tail] == [got[-1]["id"]]


def test_consumer_verify_against_journal(http):
    srv, bundle_dir, task_seq = http
    base = f"http://127.0.0.1:{srv.port}"
    filt = EventFilter(task_seq=task_seq)
    got = collect_events(base, event_filter=filt)
    journal = _journal(bundle_dir)
    try:
        report = verify_events_against_journal(got, journal,
                                               event_filter=filt)
        assert report == {"ok": True, "checked": len(got),
                          "problems": []}
        # A tampered row body is caught and itemised.
        forged = [dict(e) for e in got]
        forged[0] = dict(forged[0])
        forged[0]["row"] = dict(forged[0]["row"])
        forged[0]["row"]["entry_hash"] = "sha256:forged"
        report = verify_events_against_journal(forged, journal)
        assert not report["ok"]
        assert any("row body differs" in p for p in report["problems"])
        # A dropped middle event breaks the projection check.
        if len(got) > 2:
            holed = got[:1] + got[2:]
            report = verify_events_against_journal(
                holed, journal, event_filter=filt)
            assert not report["ok"]
    finally:
        journal.close()


def test_consumer_wire_parsing_and_url():
    assert parse_sse_block(": banner\n") == {"comment": "banner"}
    ev = parse_sse_block(
        'id: 7\nevent: task_open\n'
        'data: {"collection": "experiences", "row": {"seq": 7}}\n')
    assert ev["id"] == 7 and ev["event"] == "task_open"
    text = (": banner\n\nid: 1\nevent: checkpoint\n"
            'data: {"collection": "checkpoints", "row": {"seq": 1}}\n\n'
            ": heartbeat\n\n")
    events = events_from_text(text)
    assert [e["id"] for e in events] == [1]
    url = build_events_url("http://127.0.0.1:9/", after_seq=4,
                           event_filter=EventFilter(
                               collections={"experiences"}, task_seq=3))
    assert "collection=experiences" in url and "task=3" in url
    assert "after=4" in url and "follow=0" in url


def test_consumer_rejects_events_outside_the_filter(http):
    srv, bundle_dir, _ = http
    base = f"http://127.0.0.1:{srv.port}"
    # The honest server never does this; simulate a stream that
    # ignores the filter by fetching unfiltered events and
    # re-checking them the way fetch_events does internally:
    got = collect_events(base)
    filt = EventFilter(collections={"experiences"})
    assert any(not filt.matches_static(e) for e in got)
    assert all(filt.matches_static(e)
               for e in collect_events(base, event_filter=filt))


def test_cli_gaia_events_with_verify(http, tmp_path, capsys):
    srv, bundle_dir, _ = http
    base = f"http://127.0.0.1:{srv.port}"
    rc = cli_main(["gaia-events", "--url", base,
                   "--collection", "experiences",
                   "--verify", bundle_dir])
    out = capsys.readouterr().out
    assert rc == 0
    assert "reference consumer (SI-48)" in out
    assert "task_open" in out
    assert "verify: ok=True" in out
    rc = cli_main(["gaia-events", "--url", base,
                   "--collection", "bogus"])
    assert rc == 2
