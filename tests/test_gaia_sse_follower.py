"""SI-49: Gaia SSE resilient following consumer.

SI-48 §6 left the reconnect layer explicitly unbuilt ("no
retries, no reconnect loop"). SI-49 builds it as a reference
follower on top of the SI-48 consumer
(si/gaia/events_client.py: follow_events / collect_followed /
FollowConfig, CLI `si gaia-events --follow-resilient`):

- one logical subscription maintained across connection drops,
  reconnecting with Last-Event-ID == the last seq actually
  delivered;
- bounded exponential backoff (configurable base/cap, max
  consecutive attempts, max duration);
- never re-delivers an already-seen seq — a duplicate or
  out-of-order seq is a loud FollowError, not silent dedupe;
- protocol failures (malformed resume, filter violations,
  malformed SSE) fail closed and are never retried; only
  transport failures reconnect;
- the followed window verifies through the SI-48
  verify_events_against_journal — the one verification path.

The headline test is a REAL drop-and-reconnect against the
actual B-5 service: the follower connects through a killable
TCP proxy, the proxy is killed mid-stream, more rows are
journaled while the connection is down, the proxy restarts,
and the follower must deliver exactly the missed seqs — no
duplicates, no gaps.
"""
import itertools
import json
import socket
import threading
import urllib.error

import pytest

from si.bundle import init_bundle, load_bundle
from si.cli import main as cli_main
from si.gaia.corpus import corpus_policy
from si.gaia.events import EventFilter, read_journal_events
from si.gaia.events_client import (
    FollowConfig,
    FollowError,
    collect_events,
    collect_followed,
    follow_events,
    verify_events_against_journal,
)
from si.gaia.service import make_service_server, submit_task
from si.journal import open_journal
from si.tiny_seed import seed_tiny_bundle


@pytest.fixture()
def mini(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "mini.si"),
                             name="gaia-sse-follower",
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    work = tmp_path / "work"
    work.mkdir()
    return bundle_dir, str(work)


def _journal(bundle_dir):
    return open_journal(bundle_dir, load_bundle(bundle_dir))


def _run_task(bundle_dir, work, tag):
    return submit_task(bundle_dir, {
        "submitter": f"follower-test {tag}",
        "instruction": f"Do the follower test task {tag}",
        "work_root": work,
        "script": [
            {"kind": "tool", "tool": "fs_write",
             "args": {"path": f"follower-{tag}.txt",
                      "content": f"via follower test {tag}\n"},
             "reason": f"Write the follower test task {tag} output"},
            {"kind": "memory",
             "record": {"kind": "task_complete",
                        "detail": f"scripted completion {tag}"},
             "reason": "Report completion"},
        ]}, policy=corpus_policy(work))


def _ev(seq):
    return {"id": seq, "event": "checkpoint",
            "collection": "checkpoints", "row": {"seq": seq}}


class _SleepLog:
    def __init__(self):
        self.delays = []

    def __call__(self, delay):
        self.delays.append(delay)


# -- backoff config ---------------------------------------------------------------

def test_backoff_schedule_is_exponential_and_capped():
    cfg = FollowConfig(backoff_base=0.5, backoff_cap=1.2)
    assert [cfg.backoff_for(n) for n in (1, 2, 3, 4, 10)] == \
        [0.5, 1.0, 1.2, 1.2, 1.2]
    cfg = FollowConfig(backoff_base=0.25, backoff_cap=30.0)
    assert [cfg.backoff_for(n) for n in (1, 2, 3)] == [0.25, 0.5, 1.0]


def test_config_validation_fails_closed():
    for kwargs in ({"backoff_base": 0},
                   {"backoff_base": -1},
                   {"backoff_base": 2.0, "backoff_cap": 1.0},
                   {"max_attempts": 0},
                   {"max_duration": 0},
                   {"max_events": 0}):
        with pytest.raises(ValueError):
            FollowConfig(**kwargs)


# -- reconnect / resume over a fake transport --------------------------------------

def test_reconnect_resumes_from_last_delivered_seq():
    calls = []

    def factory(url, after_seq=0, event_filter=None, follow=False,
                timeout=None):
        calls.append(after_seq)
        if len(calls) == 1:
            def first():
                yield _ev(1)
                yield _ev(2)
                raise ConnectionResetError("simulated drop")
            return first()
        return iter([_ev(3), _ev(4)])

    sleep = _SleepLog()
    got = list(follow_events(
        "http://x", config=FollowConfig(backoff_base=0.01,
                                        backoff_cap=0.02,
                                        max_events=4),
        sleep_fn=sleep, fetch_factory=factory))
    assert [e["id"] for e in got] == [1, 2, 3, 4]
    # The reconnect asked for exactly the rows after the last
    # seq actually delivered — nothing already seen is requested.
    assert calls == [0, 2]
    assert sleep.delays == [0.01]


def test_server_close_without_error_is_also_a_reconnect():
    calls = []

    def factory(url, after_seq=0, event_filter=None, follow=False,
                timeout=None):
        calls.append(after_seq)
        if len(calls) == 1:
            return iter([_ev(1)])  # clean close: a drop by another name
        return iter([_ev(2)])

    got = list(follow_events(
        "http://x", config=FollowConfig(backoff_base=0.01,
                                        max_events=2),
        sleep_fn=_SleepLog(), fetch_factory=factory))
    assert [e["id"] for e in got] == [1, 2]
    assert calls == [0, 1]


def test_backoff_resets_after_progress():
    calls = []

    def factory(url, after_seq=0, event_filter=None, follow=False,
                timeout=None):
        calls.append(after_seq)
        n = len(calls)
        if n == 1:
            raise urllib.error.URLError("refused")

        def gen():
            yield _ev(after_seq + 1)
            if n == 2:
                raise ConnectionResetError("drop after progress")
        return gen()

    sleep = _SleepLog()
    got = list(follow_events(
        "http://x", config=FollowConfig(backoff_base=0.01,
                                        backoff_cap=1.0,
                                        max_events=2),
        sleep_fn=sleep, fetch_factory=factory))
    assert [e["id"] for e in got] == [1, 2]
    # Both delays are the BASE delay: the failure counter reset
    # once the second connection delivered an event.
    assert sleep.delays == [0.01, 0.01]


def test_max_attempts_bounds_consecutive_failures():
    calls = []

    def factory(url, after_seq=0, event_filter=None, follow=False,
                timeout=None):
        calls.append(after_seq)
        raise urllib.error.URLError("refused")

    sleep = _SleepLog()
    with pytest.raises(FollowError) as ei:
        list(follow_events(
            "http://x", after_seq=7,
            config=FollowConfig(backoff_base=0.01, backoff_cap=1.0,
                                max_attempts=3),
            sleep_fn=sleep, fetch_factory=factory))
    assert len(calls) == 3
    assert sleep.delays == [0.01, 0.02]  # no sleep before giving up
    assert "max_attempts 3" in str(ei.value)
    assert "last delivered seq is 7" in str(ei.value)


def test_max_duration_bounds_the_subscription():
    ticks = itertools.count(0, 10.0)

    def factory(url, after_seq=0, event_filter=None, follow=False,
                timeout=None):
        raise urllib.error.URLError("refused")

    with pytest.raises(FollowError) as ei:
        list(follow_events(
            "http://x",
            config=FollowConfig(backoff_base=0.01, backoff_cap=0.02,
                                max_duration=15.0),
            sleep_fn=_SleepLog(), now_fn=lambda: next(ticks),
            fetch_factory=factory))
    assert "max_duration" in str(ei.value)


def test_read_wake_timeout_is_not_a_failure():
    """The follower's own wake interval (a read timeout on an
    idle stream) reconnects without consuming backoff/attempt
    budget and honours should_stop at the wake."""
    calls = []

    def factory(url, after_seq=0, event_filter=None, follow=False,
                timeout=None):
        calls.append(after_seq)
        if len(calls) == 1:
            raise TimeoutError("read wake")
        return iter([_ev(1)])

    sleep = _SleepLog()
    got = list(follow_events(
        "http://x", config=FollowConfig(backoff_base=0.01,
                                        max_attempts=1,
                                        max_events=1),
        sleep_fn=sleep, fetch_factory=factory))
    assert [e["id"] for e in got] == [1]
    assert sleep.delays == []  # no backoff for a wake
    # should_stop is honoured at a wake even with no events:
    stop_calls = []

    def factory2(url, after_seq=0, event_filter=None, follow=False,
                 timeout=None):
        raise TimeoutError("read wake")

    def stop():
        stop_calls.append(1)
        return len(stop_calls) > 1

    assert list(follow_events("http://x", should_stop=stop,
                              sleep_fn=sleep,
                              fetch_factory=factory2)) == []


# -- fail closed: protocol failures are loud and never retried ----------------------

def test_duplicate_seq_is_a_loud_error_not_silent_dedupe():
    calls = []

    def factory(url, after_seq=0, event_filter=None, follow=False,
                timeout=None):
        calls.append(after_seq)
        return iter([_ev(1), _ev(2), _ev(2), _ev(3)])

    with pytest.raises(FollowError) as ei:
        list(follow_events("http://x", sleep_fn=_SleepLog(),
                           fetch_factory=factory))
    assert "duplicate or out-of-order" in str(ei.value)
    assert len(calls) == 1  # never retried


def test_out_of_order_seq_is_a_loud_error():
    def factory(url, after_seq=0, event_filter=None, follow=False,
                timeout=None):
        return iter([_ev(3), _ev(2)])

    with pytest.raises(FollowError):
        list(follow_events("http://x", sleep_fn=_SleepLog(),
                           fetch_factory=factory))


def test_malformed_resume_replayed_seq_fails_closed():
    """A server that resumes from the wrong point — delivering
    the resume seq itself again — is a protocol failure."""
    calls = []

    def factory(url, after_seq=0, event_filter=None, follow=False,
                timeout=None):
        calls.append(after_seq)
        return iter([_ev(5), _ev(6)])

    with pytest.raises(FollowError):
        list(follow_events("http://x", after_seq=5,
                           sleep_fn=_SleepLog(),
                           fetch_factory=factory))
    assert len(calls) == 1


def test_filter_violation_from_client_recheck_fails_closed():
    """The SI-48 layer raises ValueError when a server stream
    violates the requested filter; the follower must surface
    that as a fail-closed FollowError, never retry it."""
    calls = []

    def factory(url, after_seq=0, event_filter=None, follow=False,
                timeout=None):
        calls.append(after_seq)
        raise ValueError("server sent event id 9 outside the "
                         "requested filter's static dimensions")

    with pytest.raises(FollowError) as ei:
        list(follow_events(
            "http://x",
            event_filter=EventFilter(collections={"experiences"}),
            sleep_fn=_SleepLog(), fetch_factory=factory))
    assert "failed closed" in str(ei.value)
    assert len(calls) == 1


def test_http_400_on_resume_fails_closed_without_retry():
    calls = []

    def factory(url, after_seq=0, event_filter=None, follow=False,
                timeout=None):
        calls.append(after_seq)
        raise urllib.error.HTTPError(url, 400, "bad resume", {}, None)

    with pytest.raises(FollowError) as ei:
        list(follow_events("http://x", after_seq=3,
                           sleep_fn=_SleepLog(),
                           fetch_factory=factory))
    assert "HTTP 400" in str(ei.value)
    assert len(calls) == 1


# -- verification reuses the SI-48 path ----------------------------------------------

def test_collect_followed_verifies_against_journal(mini):
    bundle_dir, work = mini
    _run_task(bundle_dir, work, "a")
    journal = _journal(bundle_dir)
    try:
        truth = read_journal_events(journal)

        def factory(url, after_seq=0, event_filter=None, follow=False,
                    timeout=None):
            return iter([e for e in truth if e["id"] > after_seq])

        out = collect_followed(
            "http://x", config=FollowConfig(max_events=len(truth)),
            verify_source=journal, fetch_factory=factory,
            sleep_fn=_SleepLog())
        assert [e["id"] for e in out["events"]] == \
            [e["id"] for e in truth]
        assert out["last_seq"] == truth[-1]["id"]
        assert out["verify"] == {"ok": True, "checked": len(truth),
                                  "problems": []}
        # And the same path catches a forged followed window.
        forged = [dict(e) for e in truth]
        forged[0] = dict(forged[0])
        forged[0]["row"] = dict(forged[0]["row"])
        forged[0]["row"]["entry_hash"] = "sha256:forged"
        report = verify_events_against_journal(forged, journal)
        assert not report["ok"]
    finally:
        journal.close()


# -- the real service, over HTTP ------------------------------------------------------

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


class _TcpProxy:
    """A killable raw-TCP forwarder in front of the real service.

    Killing it severs every in-flight connection (the drop);
    starting it again on the same port restores the path (the
    reconnect). The service itself never restarts — exactly a
    network drop, not a server restart.
    """

    def __init__(self, target_port):
        self.target_port = target_port
        self.port = None
        self._listener = None
        self._sockets = []
        self._lock = threading.Lock()
        self._alive = False

    def start(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # After a kill, the old listener's sockets drain briefly;
        # rebinding the same port may need a few retries.
        import time as _time
        deadline = _time.monotonic() + 10.0
        while True:
            try:
                listener.bind(("127.0.0.1", self.port or 0))
                break
            except OSError:
                if _time.monotonic() > deadline:
                    raise
                _time.sleep(0.05)
        listener.listen(8)
        self.port = listener.getsockname()[1]
        self._listener = listener
        self._alive = True
        threading.Thread(target=self._accept, daemon=True).start()

    def _track(self, sock):
        with self._lock:
            self._sockets.append(sock)

    def _accept(self):
        while self._alive:
            try:
                client, _ = self._listener.accept()
            except OSError:
                return
            self._track(client)
            upstream = socket.create_connection(
                ("127.0.0.1", self.target_port))
            self._track(upstream)
            threading.Thread(target=self._pump, args=(client, upstream),
                             daemon=True).start()
            threading.Thread(target=self._pump, args=(upstream, client),
                             daemon=True).start()

    @staticmethod
    def _pump(src, dst):
        try:
            while True:
                data = src.recv(65536)
                if not data:
                    break
                dst.sendall(data)
        except OSError:
            pass
        finally:
            for s in (src, dst):
                try:
                    s.close()
                except OSError:
                    pass

    def kill(self):
        self._alive = False
        try:
            self._listener.close()
        except OSError:
            pass
        with self._lock:
            sockets, self._sockets = self._sockets, []
        for s in sockets:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                s.close()
            except OSError:
                pass


@pytest.fixture()
def http(mini):
    bundle_dir, work = mini
    _run_task(bundle_dir, work, "a")
    srv = _Http(bundle_dir, work)
    yield srv, bundle_dir, work
    srv.close()


def test_real_service_drop_and_reconnect_delivers_exactly_the_missed_seqs(http):
    srv, bundle_dir, work = http
    journal = _journal(bundle_dir)
    try:
        initial = read_journal_events(journal)
    finally:
        journal.close()
    proxy = _TcpProxy(srv.port)
    proxy.start()
    base = f"http://127.0.0.1:{proxy.port}"
    # A second task, run while the follower is down (below),
    # determines how many events the reconnect must deliver.
    journal = _journal(bundle_dir)
    try:
        pre_tip = journal.tip()[0]
    finally:
        journal.close()
    received = []
    errors = []
    done = threading.Event()
    stop = threading.Event()

    def run():
        try:
            for event in follow_events(
                    base, config=FollowConfig(
                        backoff_base=0.05, backoff_cap=0.2,
                        max_duration=60.0),
                    should_stop=stop.is_set,
                    timeout=None):
                received.append(event)
        except Exception as e:  # recorded, asserted below
            errors.append(e)
        finally:
            done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    # Wait until the follower has the whole initial replay.
    deadline = threading.Event()
    for _ in range(600):
        if len(received) >= len(initial):
            break
        deadline.wait(0.05)
    assert [e["id"] for e in received] == [e["id"] for e in initial]

    # THE DROP: sever the connection mid-stream.
    proxy.kill()

    # More history is journaled while the follower is down: a
    # whole second task, run through the B-5 machinery directly
    # (it journals and re-anchors the manifest, as a live
    # submission would — the follower's path, not the service,
    # is what is down).
    _run_task(bundle_dir, work, "b")
    journal = _journal(bundle_dir)
    try:
        full = read_journal_events(journal)
    finally:
        journal.close()
    missed = [e for e in full if e["id"] > pre_tip]
    assert missed  # the second task really journaled new rows

    # The path is restored; the follower must reconnect, resume
    # from its last delivered seq, and deliver exactly the
    # missed seqs — no duplicates, no gaps.
    proxy.start()
    for _ in range(600):
        if len(received) >= len(full):
            break
        deadline.wait(0.05)
    stop.set()
    assert done.wait(timeout=30), \
        f"follower did not stop; errors={errors!r} got={received!r}"
    thread.join(timeout=10)
    proxy.kill()
    assert errors == []
    ids = [e["id"] for e in received]
    assert ids == [e["id"] for e in full]
    assert [e["id"] for e in received[len(initial):]] == \
        [e["id"] for e in missed]
    assert len(ids) == len(set(ids))  # no duplicates
    assert ids == sorted(ids)         # in order, no gaps in delivery


def test_real_service_filtered_follow_live_tail(http):
    srv, bundle_dir, _ = http
    base = f"http://127.0.0.1:{srv.port}"
    received = []
    done = threading.Event()

    def run():
        for event in follow_events(
                base,
                event_filter=EventFilter(
                    event_names={"experience.observation"}),
                config=FollowConfig(backoff_base=0.05,
                                    max_events=1, max_duration=30.0),
                timeout=None):
            received.append(event)
        done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    writer = _journal(bundle_dir)
    try:
        writer.checkpoint("base", "m", "filtered out, walked over")
        writer.record_experience(
            "observation", {"kind": "observation", "note": "live"}, "")
        expected_seq = writer.tip()[0]
    finally:
        writer.close()
    assert done.wait(timeout=30)
    thread.join(timeout=10)
    assert [e["id"] for e in received] == [expected_seq]
    assert received[0]["event"] == "experience.observation"


def test_real_service_events_connect_after_unanchored_write(http):
    """SI-58 regression: a NEW /events connection made after a
    writer journaled rows without re-anchoring the manifest
    must still stream. The pre-fix handler re-ran load_bundle
    per connection; its anchor == tip demand failed for exactly
    this legitimate live-journal state, the unhandled
    TamperError dropped the socket with no HTTP response, and
    the follower above died in a RemoteDisconnected retry loop
    whenever its first (re)connection landed after the writes
    (a thread-scheduling race — the established-stream SI-47
    test, which connects before writing, passed all along).
    The fix gates the stream on the startup bundle verification
    plus a per-connection journal hash-chain check instead."""
    from si.bundle import TamperError
    srv, bundle_dir, _ = http
    base = f"http://127.0.0.1:{srv.port}"
    writer = _journal(bundle_dir)
    try:
        writer.checkpoint("base", "m", "filtered out, walked over")
        writer.record_experience(
            "observation", {"kind": "observation", "note": "live"}, "")
        expected_seq = writer.tip()[0]
    finally:
        writer.close()
    # The scenario's precondition, stated honestly: the journal
    # tip really is past the manifest anchor at connect time.
    with pytest.raises(TamperError):
        load_bundle(bundle_dir)
    got = collect_events(
        base,
        event_filter=EventFilter(
            event_names={"experience.observation"}),
        timeout=15)
    assert [e["id"] for e in got] == [expected_seq]
    assert got[0]["event"] == "experience.observation"


# -- the CLI surface ------------------------------------------------------------------

def test_cli_follow_resilient_bounded(http, capsys):
    srv, bundle_dir, _ = http
    base = f"http://127.0.0.1:{srv.port}"
    rc = cli_main(["gaia-events", "--url", base,
                   "--follow-resilient", "--max-events", "3",
                   "--backoff-base", "0.01"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "reference follower (SI-49)" in out
    assert "3 event(s); last id 3" in out


def test_cli_follow_resilient_verify_needs_a_bound(http, capsys):
    srv, bundle_dir, _ = http
    base = f"http://127.0.0.1:{srv.port}"
    rc = cli_main(["gaia-events", "--url", base,
                   "--follow-resilient", "--verify", bundle_dir])
    assert rc == 2
    rc = cli_main(["gaia-events", "--url", base,
                   "--follow-resilient", "--max-events", "2",
                   "--verify", bundle_dir])
    out = capsys.readouterr().out
    assert rc == 0
    assert "verify: ok=True" in out


def test_cli_follow_resilient_unreachable_fails_closed(capsys):
    # A port nothing listens on: bounded attempts, loud failure.
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    dead_port = probe.getsockname()[1]
    probe.close()
    rc = cli_main(["gaia-events", "--url",
                   f"http://127.0.0.1:{dead_port}",
                   "--follow-resilient", "--max-attempts", "2",
                   "--backoff-base", "0.01", "--backoff-cap", "0.02"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "max_attempts 2" in err
