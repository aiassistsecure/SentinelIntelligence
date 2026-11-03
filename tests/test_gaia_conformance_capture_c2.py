"""SI-79 (SI-76 §4.4 C-2): capture + record glue tests.

OFFLINE ONLY. No network, no node, no live attempt: capture
is exercised through a fake transport, and the record tests
score synthetic candidate corpora through the C-1 harness.
The live capture itself is evidence in the SI-79 slice doc,
not in these tests. Honesty labels pinned here: §18.3 NOT
earned, P5-R1/R9 UNMEASURED, live model-authored intent
OCCURRED scoped to SI-74 only, L-3 LOCKED, A-2/A-3/A-4
UNAPPROVED, readiness.unlocks [].
"""
import json

import pytest

from si.gaia.conformance import V2_WORKLOADS, validate_corpus
from si.gaia.conformance_capture import (
    D_C1, ResidencyGuardHalt, build_c2_record, capture_corpus,
    dc1_coverage_missing, discover_served_model_id,
    extract_content)


def _doc(intents, signal=None):
    return json.dumps({"proposal": {
        "version": 1, "intents": intents, "task_signal": signal}})


def _intent(tool="fs_list", path=".", reason="list the work root"):
    return {"tool": tool, "args": {"path": path},
            "reason": reason, "depends_on": [],
            "ordering_key": None}


def _item(item_id, body, workload="benign-read"):
    return {"item_id": item_id, "workload": workload,
            "cycle": 0, "body": body,
            "body_source": "verbatim-recorded",
            "prompt_ref": {"workload": workload},
            "provenance": {"capture": "test"}}


def _full_corpus(ok_per_workload):
    """A §3.3-sufficient candidate corpus: 5 distinct items
    per each of the six L-1 v2 workloads; ok_per_workload
    parse-ok docs, the rest distinct stage-1 text."""
    items = []
    for label in V2_WORKLOADS:
        for k in range(5):
            if k < ok_per_workload:
                body = _doc([_intent(
                    reason=f"{label} item {k} lists the root")])
            else:
                body = f"plain identity text {label} {k}\n"
            items.append(_item(f"{label}-{k}", body,
                               workload=label))
    return {
        "corpus_id": "test-c2-v1", "purpose": "candidate-readiness",
        "candidate_label": "operator-label-as-discovered",
        "response_origin": "candidate-recorded",
        "capture_date": "2026-11-02", "fixed_n": len(items),
        "coverage": {"workloads": list(V2_WORKLOADS),
                     "cycles_covered": [0],
                     "signals_elicited": False},
        "items": items,
    }


# -- extraction / discovery ---------------------------------

def test_extract_content_matches_live_proposer_rule():
    envelope = json.dumps({"choices": [{"message": {
        "content": "the §8 text"}}]})
    assert extract_content(envelope) == "the §8 text"
    assert extract_content("not json at all") == "not json at all"
    assert extract_content('{"usage": {}}') == '{"usage": {}}'


def test_discover_served_model_id_reads_data0_id():
    def fake(method, url, *, json_body=None, timeout_s=5.0):
        assert method == "GET" and url.endswith("/v1/models")
        return 200, json.dumps(
            {"data": [{"id": "served-id-as-discovered"}]}).encode()
    assert discover_served_model_id(
        "http://127.0.0.1:1", transport=fake) == \
        "served-id-as-discovered"


# -- D-C1 coverage + readiness verdicts ----------------------

def test_dc1_coverage_missing_names_shortfalls():
    corpus = _full_corpus(5)
    assert dc1_coverage_missing(corpus) == []
    corpus["items"] = corpus["items"][:-1]
    missing = dc1_coverage_missing(corpus)
    assert {m["name"] for m in missing} == {
        "dc1-min-n", "dc1-min-per-workload"}


def test_record_meets_threshold_when_rate_and_floors_hold():
    record = build_c2_record(_full_corpus(4), argv=["test"])
    assert record["status"] == "scored"
    assert record["results"]["rate"] == {
        "fraction": "24/30", "decimal": 0.8}
    readiness = record["readiness"]
    assert readiness["threshold"]["rate_min"] == D_C1["rate_min"]
    assert "2026-11-02" in readiness["threshold"]["decision"]
    assert readiness["verdict"] == "MEETS THRESHOLD"
    assert readiness["unlocks"] == []


def test_record_below_threshold_on_floor_failure():
    corpus = _full_corpus(5)
    # One workload drops to 2/5 parse-ok: overall 27/30 still
    # clears 0.80, but the 0.60 per-workload floor fails.
    items = corpus["items"]
    idx = [i for i, it in enumerate(items)
           if it["workload"] == "benign-read"][2:]
    for i in idx:
        items[i] = _item(items[i]["item_id"],
                         f"floor-breaking text {i}\n",
                         workload="benign-read")
    record = build_c2_record(corpus, argv=["test"])
    assert record["results"]["rate"]["decimal"] == 27 / 30
    assert record["readiness"]["verdict"] == "BELOW THRESHOLD"
    assert record["readiness"]["unlocks"] == []


def test_record_insufficient_corpus_publishes_no_verdict():
    corpus = _full_corpus(5)
    corpus["items"] = corpus["items"][:10]
    corpus["fixed_n"] = 10
    assert validate_corpus(corpus) == []  # §3.3-sufficient…
    record = build_c2_record(corpus, argv=["test"])
    # …but D-C1 coverage fails: INSUFFICIENT CORPUS, no verdict.
    assert record["status"] == "INSUFFICIENT CORPUS"
    assert record["readiness"]["verdict"] is None
    assert record["readiness"]["unlocks"] == []
    assert record["missing_requirements"]


def test_record_honesty_labels_verbatim():
    record = build_c2_record(_full_corpus(5), argv=["test"])
    honesty = record["honesty"]
    assert honesty["section_18_3_claim"] == "NOT earned"
    assert honesty["row_P5_R1"] == "UNMEASURED"
    assert honesty["row_P5_R9"] == "UNMEASURED"
    assert honesty["live_model_authored_intent"].startswith(
        "OCCURRED, scoped to SI-74 only")
    assert honesty["l3_residency_loss_drill"] == "LOCKED"
    assert honesty["a2_a3_a4"] == "UNAPPROVED"
    assert all(r["promoted_by_this_measurement"] == "none"
               for r in record["rows_18_1"])
    assert record["slice"].startswith("SI-79")


# -- Capture through a fake transport ------------------------

class _FakeTransport:
    """Answers residency polls, /v1/models discovery, and
    chat completions with rotating distinct bodies."""

    label = "fake-transport (tests)"
    is_replay = False

    def __init__(self):
        self.chat_calls = 0

    def __call__(self, method, url, *, json_body=None,
                 timeout_s=5.0):
        if url.endswith("/v1/models"):
            return 200, json.dumps(
                {"data": [{"id": "fake-served-id"}]}).encode()
        if url.endswith("/gaia/v1/residency"):
            return 200, json.dumps({"models": [{
                "model": "imagine-v12",
                "state": "resident"}]}).encode()
        if url.endswith("/v1/chat/completions"):
            self.chat_calls += 1
            body = _doc([_intent(
                reason=f"call {self.chat_calls} lists the root")])
            return 200, json.dumps({"choices": [{"message": {
                "content": body}}]}).encode()
        raise AssertionError(f"unexpected URL {url}")


def test_capture_corpus_shape_with_fake_transport(tmp_path):
    fake = _FakeTransport()
    corpus, log = capture_corpus(
        str(tmp_path), inference_base="http://fake-inference",
        residency_base="http://fake-residency",
        items_per_workload=2, max_attempts_per_workload=3,
        transport=fake, log=lambda *a: None)
    assert log["served_model_id"] == "fake-served-id"
    assert corpus["candidate_label"] == "fake-served-id"
    assert corpus["fixed_n"] == 2 * len(V2_WORKLOADS)
    assert validate_corpus(corpus) == []
    assert all(i["body_source"] == "verbatim-recorded"
               for i in corpus["items"])
    assert all(i["cycle"] == 0 for i in corpus["items"])
    assert corpus["coverage"]["cycles_covered"] == [0]


def test_capture_fires_progress_after_every_attempt(tmp_path):
    fake = _FakeTransport()
    snapshots = []
    capture_corpus(
        str(tmp_path), inference_base="http://fake-inference",
        residency_base="http://fake-residency",
        items_per_workload=1, max_attempts_per_workload=2,
        transport=fake, log=lambda *a: None,
        on_progress=lambda c, lg: snapshots.append(
            (c["fixed_n"], len(lg["entries"]))))
    # One snapshot per accepted item, counts growing in step.
    assert snapshots == [(n, n) for n in range(
        1, len(V2_WORKLOADS) + 1)]


# -- Resume (SI-79 interrupted-capture continuation) ---------

class _QueueTransport(_FakeTransport):
    """Chat bodies from a queue first, then the rotating
    distinct bodies of _FakeTransport."""

    def __init__(self, bodies):
        super().__init__()
        self.bodies = list(bodies)

    def __call__(self, method, url, *, json_body=None,
                 timeout_s=5.0):
        if url.endswith("/v1/chat/completions") and self.bodies:
            self.chat_calls += 1
            body = self.bodies.pop(0)
            return 200, json.dumps({"choices": [{"message": {
                "content": body}}]}).encode()
        return super().__call__(
            method, url, json_body=json_body, timeout_s=timeout_s)


def _initial_capture(tmp_path, **kw):
    fake = _FakeTransport()
    corpus, log = capture_corpus(
        str(tmp_path), inference_base="http://fake-inference",
        residency_base="http://fake-residency",
        transport=fake, log=lambda *a: None, **kw)
    return corpus, log


def test_resume_seeds_counts_and_skips_completed(tmp_path):
    corpus, log = _initial_capture(
        tmp_path, items_per_workload=1, max_attempts_per_workload=2)
    assert corpus["fixed_n"] == len(V2_WORKLOADS)
    fake = _FakeTransport()
    corpus2, log2 = capture_corpus(
        str(tmp_path), inference_base="http://fake-inference",
        residency_base="http://fake-residency",
        items_per_workload=1, max_attempts_per_workload=2,
        transport=fake, log=lambda *a: None,
        resume_corpus=corpus, resume_log=log)
    # Every workload was already complete: nothing re-attempted,
    # seeded items and log entries carried through intact.
    assert fake.chat_calls == 0
    assert corpus2["fixed_n"] == corpus["fixed_n"]
    assert [(i["item_id"], i["body"]) for i in corpus2["items"]] == \
        [(i["item_id"], i["body"]) for i in corpus["items"]]
    assert log2["entries"] == log["entries"]


def test_resume_global_duplicate_and_remaining_budget(tmp_path):
    corpus, log = _initial_capture(
        tmp_path, items_per_workload=1, max_attempts_per_workload=2)
    seeded_body = corpus["items"][0]["body"]
    assert corpus["items"][0]["item_id"] == "c2-benign-read-0"
    fake = _QueueTransport([seeded_body])
    fake.chat_calls = 100  # fallback bodies must not replay seeded ones
    corpus2, log2 = capture_corpus(
        str(tmp_path), inference_base="http://fake-inference",
        residency_base="http://fake-residency",
        items_per_workload=2, max_attempts_per_workload=2,
        transport=fake, log=lambda *a: None,
        resume_corpus=corpus, resume_log=log)
    # Seeded attempts: 1 per workload. Cap 2 => exactly ONE
    # remaining attempt per workload; no workload exceeds it.
    attempts = {}
    for e in log2["entries"]:
        attempts[e["workload"]] = attempts.get(e["workload"], 0) + 1
    assert attempts == {label: 2 for label in V2_WORKLOADS}
    assert fake.chat_calls - 100 == len(V2_WORKLOADS)
    # The queued body duplicates a SEEDED body globally: logged
    # as a duplicate of the seeded item, never entered, and it
    # spent benign-read's only remaining attempt.
    dup = [e for e in log2["entries"]
           if e["outcome"] == "duplicate-capture"]
    assert len(dup) == 1 and dup[0]["workload"] == "benign-read"
    assert dup[0]["duplicate_of"] == "c2-benign-read-0"
    assert dup[0]["attempt"] == 2
    per_workload = {}
    for i in corpus2["items"]:
        per_workload[i["workload"]] = \
            per_workload.get(i["workload"], 0) + 1
    assert per_workload["benign-read"] == 1
    assert all(per_workload[l] == 2 for l in V2_WORKLOADS
               if l != "benign-read")
    assert corpus2["fixed_n"] == 11


def test_resume_skips_one_completed_workload_only(tmp_path):
    corpus, log = _initial_capture(
        tmp_path, items_per_workload=1, max_attempts_per_workload=2)
    # Extend the seed: benign-read is now complete at target 2,
    # with its second attempt already spent in the log.
    corpus = dict(corpus, items=list(corpus["items"]) + [
        _item("c2-benign-read-1", "synthetic second benign body\n",
              workload="benign-read")], fixed_n=7)
    log = dict(log, entries=list(log["entries"]) + [{
        "item_id": "c2-benign-read-1", "workload": "benign-read",
        "attempt": 2, "outcome": "captured"}])
    fake = _FakeTransport()
    fake.chat_calls = 100  # bodies must not replay seeded ones
    corpus2, log2 = capture_corpus(
        str(tmp_path), inference_base="http://fake-inference",
        residency_base="http://fake-residency",
        items_per_workload=2, max_attempts_per_workload=3,
        transport=fake, log=lambda *a: None,
        resume_corpus=corpus, resume_log=log)
    # Only the five unfinished workloads made a chat call.
    assert fake.chat_calls - 100 == len(V2_WORKLOADS) - 1
    benign = [i for i in corpus2["items"]
              if i["workload"] == "benign-read"]
    assert [i["item_id"] for i in benign] == [
        "c2-benign-read-0", "c2-benign-read-1"]
    assert len([e for e in log2["entries"]
                if e["workload"] == "benign-read"]) == 2
    assert corpus2["fixed_n"] == 12


def test_resume_refuses_served_model_mismatch(tmp_path):
    corpus, log = _initial_capture(
        tmp_path, items_per_workload=1, max_attempts_per_workload=2)
    corpus = dict(corpus, candidate_label="some-other-model")
    with pytest.raises(RuntimeError, match="no substitution"):
        capture_corpus(
            str(tmp_path), inference_base="http://fake-inference",
            residency_base="http://fake-residency",
            items_per_workload=1, max_attempts_per_workload=2,
            transport=_FakeTransport(), log=lambda *a: None,
            resume_corpus=corpus, resume_log=log)


# -- Residency guard (SI-79: halt on non-Resident, no spend) --

class _ResidencyStateTransport(_FakeTransport):
    """Residency polls report a fixed raw state."""

    def __init__(self, raw_state):
        super().__init__()
        self.raw_state = raw_state

    def __call__(self, method, url, *, json_body=None,
                 timeout_s=5.0):
        if url.endswith("/gaia/v1/residency"):
            return 200, json.dumps({"models": [{
                "model": "imagine-v12",
                "state": self.raw_state}]}).encode()
        return super().__call__(
            method, url, json_body=json_body, timeout_s=timeout_s)


class _ResidencyRaisingTransport(_FakeTransport):
    """Residency polls themselves fail in transport."""

    def __call__(self, method, url, *, json_body=None,
                 timeout_s=5.0):
        if url.endswith("/gaia/v1/residency"):
            raise ConnectionError("connection refused (poll)")
        return super().__call__(
            method, url, json_body=json_body, timeout_s=timeout_s)


class _ChatFailingTransport(_FakeTransport):
    """Resident, but every chat POST fails in transport."""

    def __call__(self, method, url, *, json_body=None,
                 timeout_s=5.0):
        if url.endswith("/v1/chat/completions"):
            self.chat_calls += 1
            raise ConnectionError("server died mid-request")
        return super().__call__(
            method, url, json_body=json_body, timeout_s=timeout_s)


def _guard_run(tmp_path, fake):
    """Run capture collecting on_progress snapshots and log
    lines; returns (exception, snapshots, messages)."""
    snapshots, messages = [], []
    with pytest.raises(ResidencyGuardHalt) as excinfo:
        capture_corpus(
            str(tmp_path), inference_base="http://fake-inference",
            residency_base="http://fake-residency",
            items_per_workload=1, max_attempts_per_workload=3,
            transport=fake, log=messages.append,
            on_progress=lambda c, lg: snapshots.append(
                (list(c["items"]), list(lg["entries"]))))
    return excinfo.value, snapshots, messages


def test_guard_halts_on_lost_evicted_no_attempt_spent(tmp_path):
    fake = _ResidencyStateTransport("Lost{Evicted}")
    exc, snapshots, messages = _guard_run(tmp_path, fake)
    # No POST ever fired, and no progress snapshot ever fired:
    # on_progress only fires after a spent attempt appends a
    # corpus item / capture-log entry, so corpus items and
    # capture-log counts are unchanged and no attempt was spent.
    assert fake.chat_calls == 0
    assert snapshots == []
    assert "benign-read" in str(exc)
    assert "Lost{Evicted}" in str(exc)
    assert "no attempt spent" in str(exc)
    assert any("residency guard halt" in m for m in messages)


def test_guard_halts_on_stopped_no_attempt_spent(tmp_path):
    fake = _ResidencyStateTransport("stopped")
    exc, snapshots, messages = _guard_run(tmp_path, fake)
    assert fake.chat_calls == 0
    assert snapshots == []
    assert "Stopped" in str(exc)
    assert "no attempt spent" in str(exc)
    assert any("residency guard halt" in m for m in messages)


def test_guard_halts_when_residency_poll_raises(tmp_path):
    fake = _ResidencyRaisingTransport()
    exc, snapshots, messages = _guard_run(tmp_path, fake)
    assert fake.chat_calls == 0
    assert snapshots == []
    assert "benign-read" in str(exc)
    assert "no attempt spent" in str(exc)
    assert isinstance(exc.__cause__, ConnectionError)
    assert any("residency guard halt" in m for m in messages)


def test_guard_resident_state_captures_as_before(tmp_path):
    fake = _ResidencyStateTransport("resident")
    corpus, log = capture_corpus(
        str(tmp_path), inference_base="http://fake-inference",
        residency_base="http://fake-residency",
        items_per_workload=1, max_attempts_per_workload=2,
        transport=fake, log=lambda *a: None)
    assert corpus["fixed_n"] == len(V2_WORKLOADS)
    assert len(corpus["items"]) == len(V2_WORKLOADS)
    assert len(log["entries"]) == len(V2_WORKLOADS)
    assert all(e["outcome"] == "captured" for e in log["entries"])
    assert all(e["residency_state"] == "Resident"
               for e in log["entries"])


def test_guard_post_failure_after_resident_still_spends(tmp_path):
    fake = _ChatFailingTransport()
    corpus, log = capture_corpus(
        str(tmp_path), inference_base="http://fake-inference",
        residency_base="http://fake-residency",
        items_per_workload=1, max_attempts_per_workload=1,
        transport=fake, log=lambda *a: None)
    # The POST was reached (Resident poll passed), so the
    # failure keeps the original behaviour: recorded
    # transport-failed, unscored, and the attempt IS spent —
    # exactly one per workload at the cap of 1.
    assert fake.chat_calls == len(V2_WORKLOADS)
    assert corpus["fixed_n"] == 0
    assert len(corpus["items"]) == len(V2_WORKLOADS)
    assert all(i["transport_failed"] for i in corpus["items"])
    assert len(log["entries"]) == len(V2_WORKLOADS)
    assert all(e["outcome"] == "transport-failed"
               for e in log["entries"])
    assert all(e["attempt"] == 1 for e in log["entries"])
