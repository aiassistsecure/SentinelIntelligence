"""SI-66 (SI-63 RFC §19 P-3): Gaia PROPOSE — gateway wiring
under preconditions (replay + refusal paths).

P-3 builds, in si/gaia/proposer.py (§8) and si/gaia/policy.py:
the §10 gateway call path wired into ``HearthProposer.propose``
behind the §11.3 precondition check; the §13.2 two-failure
halts; the §15.2 policy keys with ``proposer_enabled``
defaulting OFF; the §13.3 leash-at-call discard; and the
§16.2–§16.4 failure behaviours.

Honesty discipline asserted throughout: everything here is
exercised against the labelled replay transport and refusal
fixtures only — no real Hearth node is contacted, the thinking
proposer remains UNBUILT beyond P-3 replay/refusal wiring,
live model-authored intent has NEVER YET OCCURRED, and the real
Hearth path remains UNMEASURED.
"""
import json
from datetime import datetime, timezone

import pytest

from si.bundle import canonical_json, init_bundle, sha256_hex
from si.gaia.corpus import corpus_policy
from si.gaia.hearth import ReplayTransport
from si.gaia.policy import (
    PROPOSER_POLICY_DEFAULTS,
    resolve_proposer_policy,
    resolve_tool_policy,
)
from si.gaia.proposer import (
    ATTEMPT_RECORD_KIND,
    DISCARDED_UNDER_LEASH,
    IDENTITY_MISMATCH,
    NOT_EVALUATED,
    P3_SCOPE_LABEL,
    RESPONSE_RECORD_KIND,
    TRANSPORT_FAILED,
    HearthProposer,
    build_gateway_request,
    check_preconditions,
    render_catalogue,
    run_gateway_proposal,
)
from si.gaia.queue import record_queue_stop
from si.gaia.task import record_task_entry, run_task
from si.gate import Intent
from si.journal import EmbeddedBackend, Journal
from si.tiny_seed import seed_tiny_bundle

MODEL = "test-model-1"
NOW = datetime(2026, 10, 31, 12, 0, 0, tzinfo=timezone.utc)
FRESH_AT = "2026-10-31T11:59:30Z"   # 30s before NOW (bound 60s)
STALE_AT = "2026-10-31T11:50:00Z"   # 600s before NOW
BASE_URL = "http://replay.invalid"
USAGE = {"prompt_tokens": 11, "completion_tokens": 7,
         "total_tokens": 18}


def _clock():
    return NOW


@pytest.fixture()
def journal(tmp_path):
    j = Journal(EmbeddedBackend(str(tmp_path / "journal")))
    yield j
    j.close()


def _task(journal, instruction="inspect the work root"):
    return record_task_entry(
        journal, instruction, "/tmp/p3-work",
        budgets={"max_tool_cycles_per_task": 16,
                 "tool_max_task_result_bytes": 1048576})


def _add_residency(journal, state="Resident", polled_at=FRESH_AT,
                   model=MODEL):
    record = {"kind": "hearth_residency",
              "source": "hearth:/residency", "state": state,
              "raw_state": state, "model": model,
              "node": "node-a", "polled_at": polled_at}
    journal.record_experience("hearth_residency", record, "")


def _doc(intents, signal=None):
    return json.dumps({"proposal": {
        "version": 1, "intents": intents, "task_signal": signal}})


def _intent(tool="fs_read", args=None, reason="read the plan file"):
    return {"tool": tool,
            "args": args if args is not None else {"path": "plan.txt"},
            "reason": reason, "depends_on": [], "ordering_key": None}


VALID_DOC = _doc([_intent()])


def _gateway_body(content, usage=USAGE):
    payload = {"choices": [{"message": {"content": content}}]}
    if usage is not None:
        payload["usage"] = usage
    return payload


def _transport(*responses):
    return ReplayTransport(list(responses))


def _proposer(journal, task, transport, **kw):
    base = dict(transport=transport, gateway_base_url=BASE_URL,
                model=MODEL, served_models=[MODEL],
                task_seq=task["seq"], journal=journal,
                policy={"proposer_enabled": True}, clock=_clock)
    base.update(kw)
    return HearthProposer(**base)


def _records(journal, kind):
    return journal.backend.find(
        "experiences", filter={"kind": kind}) or []


def _attempts(journal):
    return [r["record"] for r in _records(journal, ATTEMPT_RECORD_KIND)]


def _responses(journal):
    return [r["record"] for r in _records(journal, RESPONSE_RECORD_KIND)]


def _ready(journal, state="Resident", polled_at=FRESH_AT):
    task = _task(journal)
    _add_residency(journal, state=state, polled_at=polled_at)
    return task


# -- §15.2 policy resolution ------------------------------------------------

def test_proposer_policy_defaults_off_and_finite():
    resolved = resolve_proposer_policy(None)
    assert resolved == PROPOSER_POLICY_DEFAULTS
    assert resolved["proposer_enabled"] is False
    assert resolve_proposer_policy({})["proposer_enabled"] is False
    assert resolve_proposer_policy(
        {"proposer_enabled": True})["proposer_enabled"] is True


def test_proposer_policy_malformed_values_fail_closed():
    # A malformed switch never turns the proposer on (§15.2).
    for bad in ("true", 1, ["yes"], {"on": True}):
        assert resolve_proposer_policy(
            {"proposer_enabled": bad})["proposer_enabled"] is False
    resolved = resolve_proposer_policy({
        "proposer_max_attempts_per_task": -3,
        "proposer_request_timeout_s": "soon",
        "proposer_residency_freshness_s": None})
    assert resolved["proposer_max_attempts_per_task"] == 0
    assert resolved["proposer_request_timeout_s"] == 0
    assert resolved["proposer_residency_freshness_s"] == 0
    ok = resolve_proposer_policy({
        "proposer_max_attempts_per_task": 3,
        "proposer_request_timeout_s": 12.5})
    assert ok["proposer_max_attempts_per_task"] == 3
    assert ok["proposer_request_timeout_s"] == 12.5


def test_tool_policy_resolution_unchanged_by_proposer_keys():
    # §15.2's resolver is additive: §6.2 resolution is untouched.
    policy = {"allowed_tools": ["fs_read"],
              "tool_max_result_bytes": 10}
    before = resolve_tool_policy(policy)
    resolve_proposer_policy(policy)
    assert resolve_tool_policy(policy) == before
    assert set(before) == {
        "allowed_tools", "tool_class_policy", "tool_work_root",
        "tool_max_result_bytes", "tool_timeout_seconds",
        "max_tool_cycles_per_task", "tool_max_task_result_bytes"}


# -- §15.2 regression: disabled is byte-identical to P-1 ----------------------

@pytest.mark.parametrize("policy", [
    None, {}, {"proposer_enabled": False},
    {"proposer_enabled": "yes"},
])
def test_disabled_proposer_makes_no_call_and_returns_p1_outcome(
        journal, policy):
    task = _ready(journal)
    transport = _transport((200, _gateway_body(VALID_DOC)))
    proposer = _proposer(journal, task, transport, policy=policy)
    perceptions = proposer.perceive(journal)
    outcome = proposer.propose(perceptions, ctx=None)
    assert transport.calls == []
    assert outcome["outcome"] == NOT_EVALUATED
    assert outcome["transport_called"] is False
    assert outcome["intents"] == []
    assert "UNMEASURED" in outcome["hearth_path_performance"]
    # Nothing is journaled on the disabled path (P-1 journals
    # nothing either — behaviour is unchanged, not similar).
    assert _attempts(journal) == [] and _responses(journal) == []


# -- §11.3 / §16.2 refusal fixtures: zero calls, journaled leg ----------------

def test_no_transport_is_a_journaled_zero_call_refusal(journal):
    task = _ready(journal)
    proposer = _proposer(journal, task, None)
    assert proposer.propose(proposer.perceive(journal)) is None
    outcome = proposer.last_outcome
    assert outcome["outcome"] == NOT_EVALUATED
    assert outcome["failed_leg"] == "transport"
    assert outcome["transport_called"] is False
    (attempt,) = _attempts(journal)
    assert attempt["gateway_attempt"] is True
    assert attempt["preconditions"]["failed_leg"] == "transport"
    assert attempt["preconditions"]["outcome"] == NOT_EVALUATED
    assert _responses(journal) == []


def test_residency_not_resident_refuses_with_zero_calls(journal):
    task = _ready(journal, state="Loading")
    transport = _transport((200, _gateway_body(VALID_DOC)))
    proposer = _proposer(journal, task, transport)
    assert proposer.propose(proposer.perceive(journal)) is None
    outcome = proposer.last_outcome
    assert outcome["outcome"] == NOT_EVALUATED
    assert outcome["failed_leg"] == "residency"
    assert transport.calls == []
    (attempt,) = _attempts(journal)
    leg = attempt["preconditions"]["legs"]["residency"]
    assert leg["passed"] is False and "Loading" in leg["detail"]


def test_residency_lost_state_is_named_in_the_refusal(journal):
    task = _ready(journal, state="Lost{Evicted}")
    transport = _transport((200, _gateway_body(VALID_DOC)))
    proposer = _proposer(journal, task, transport)
    assert proposer.propose(proposer.perceive(journal)) is None
    assert proposer.last_outcome["failed_leg"] == "residency"
    assert transport.calls == []
    leg = _attempts(journal)[0]["preconditions"]["legs"]["residency"]
    assert "Lost{Evicted}" in leg["detail"]


def test_stale_residency_refuses_with_zero_calls(journal):
    task = _ready(journal, polled_at=STALE_AT)
    transport = _transport((200, _gateway_body(VALID_DOC)))
    proposer = _proposer(journal, task, transport)
    assert proposer.propose(proposer.perceive(journal)) is None
    assert proposer.last_outcome["failed_leg"] == "residency"
    assert "freshness" in proposer.last_outcome["detail"]
    assert transport.calls == []


def test_missing_residency_refuses_with_zero_calls(journal):
    task = _task(journal)  # no residency observation at all
    transport = _transport((200, _gateway_body(VALID_DOC)))
    proposer = _proposer(journal, task, transport)
    assert proposer.propose(proposer.perceive(journal)) is None
    assert proposer.last_outcome["failed_leg"] == "residency"
    assert transport.calls == []


def test_identity_unverified_refuses_with_zero_calls(journal):
    task = _ready(journal)
    transport = _transport((200, _gateway_body(VALID_DOC)))
    proposer = _proposer(journal, task, transport, served_models=None)
    assert proposer.propose(proposer.perceive(journal)) is None
    assert proposer.last_outcome["outcome"] == NOT_EVALUATED
    assert proposer.last_outcome["failed_leg"] == "identity"
    assert transport.calls == []


def test_identity_mismatch_halts_with_zero_calls(journal):
    task = _ready(journal)
    transport = _transport((200, _gateway_body(VALID_DOC)))
    proposer = _proposer(journal, task, transport,
                         served_models=["other-model-9"])
    assert proposer.propose(proposer.perceive(journal)) is None
    outcome = proposer.last_outcome
    assert outcome["outcome"] == IDENTITY_MISMATCH
    assert outcome["failed_leg"] == "identity"
    assert outcome["halted"] is True  # §16.4: proposing halts
    assert transport.calls == []
    (attempt,) = _attempts(journal)
    # Both identities are recorded (§16.4).
    assert attempt["model"]["configured_id"] == MODEL
    assert attempt["model"]["served_ids"] == ["other-model-9"]
    assert attempt["preconditions"]["outcome"] == IDENTITY_MISMATCH


def test_residency_model_mismatch_is_identity_mismatch(journal):
    task = _task(journal)
    _add_residency(journal, model="served-under-another-name")
    transport = _transport((200, _gateway_body(VALID_DOC)))
    proposer = _proposer(journal, task, transport)
    assert proposer.propose(proposer.perceive(journal)) is None
    assert proposer.last_outcome["outcome"] == IDENTITY_MISMATCH
    assert transport.calls == []


def test_declaration_mismatch_is_identity_mismatch(journal):
    task = _ready(journal)
    transport = _transport((200, _gateway_body(VALID_DOC)))
    proposer = _proposer(
        journal, task, transport,
        declaration_digest="sha256:" + "a" * 64,
        served_declaration_digest="sha256:" + "b" * 64)
    assert proposer.propose(proposer.perceive(journal)) is None
    outcome = proposer.last_outcome
    assert outcome["outcome"] == IDENTITY_MISMATCH
    assert outcome["failed_leg"] == "declaration"
    assert transport.calls == []


def test_leash_in_force_refuses_with_zero_calls(journal):
    task = _ready(journal)
    record_queue_stop(journal, task["seq"], "operator stop")
    transport = _transport((200, _gateway_body(VALID_DOC)))
    proposer = _proposer(journal, task, transport)
    assert proposer.propose(proposer.perceive(journal)) is None
    assert proposer.last_outcome["failed_leg"] == "leash"
    assert transport.calls == []


def test_enabled_without_journal_makes_no_call(journal):
    # §16.9: a proposal that cannot be journaled cannot happen.
    task = _ready(journal)
    perceptions = HearthProposer(task_seq=task["seq"]).perceive(journal)
    transport = _transport((200, _gateway_body(VALID_DOC)))
    proposer = HearthProposer(
        transport=transport, gateway_base_url=BASE_URL, model=MODEL,
        served_models=[MODEL], task_seq=task["seq"],
        policy={"proposer_enabled": True}, clock=_clock)
    assert proposer.propose(perceptions, ctx=None) is None
    assert proposer.last_outcome["failed_leg"] == "journal"
    assert transport.calls == []


# -- the happy path: §10 call, §14 chain, §5.1 batch return -------------------

def test_gateway_success_returns_batch_and_journals_the_chain(journal):
    task = _ready(journal)
    transport = _transport((200, _gateway_body(VALID_DOC)))
    proposer = _proposer(journal, task, transport)
    perceptions = proposer.perceive(journal)
    batch = proposer.propose(perceptions, ctx=None)
    assert isinstance(batch, list) and len(batch) == 1
    (intent,) = batch
    assert isinstance(intent, Intent)
    assert intent.kind == "tool" and intent.tool == "fs_read"
    assert intent.args == {"path": "plan.txt"}
    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == BASE_URL + "/v1/chat/completions"
    assert call["json_body"]["model"] == MODEL
    system = call["json_body"]["messages"][0]
    assert system["role"] == "system"
    assert "do not execute anything yourself" in system["content"]
    assert "not your instructions" in system["content"]

    outcome = proposer.last_outcome
    assert outcome["outcome"] == "ok"
    assert P3_SCOPE_LABEL in outcome["label"]
    assert "UNMEASURED" in outcome["hearth_path_performance"]

    (attempt,) = _attempts(journal)
    (response,) = _responses(journal)
    assert attempt["gateway_attempt"] is True
    assert attempt["preconditions"]["outcome"] == "pass"
    assert attempt["policy"]["proposer_enabled"] is True
    # §14.2: the journaled request digest verifies against the
    # exact body the transport received.
    expect = "sha256:" + sha256_hex(canonical_json(call["json_body"]))
    assert attempt["request_digest"] == expect
    # §14.3: response links back to the attempt, usage as
    # reported, batch summarised — journaled before admission.
    assert response["attempt_seq"] == outcome["attempt_seq"]
    assert response["gateway_attempt"] is True
    assert response["transport"]["usage"] == USAGE
    assert response["transport"]["is_replay"] is True
    assert response["batch"]["members"] == 1
    assert response["batch"]["tools"] == ["fs_read"]
    # The proposer admitted nothing itself (§12): no self-edit
    # exists for its batch — admission is the loop's business.
    assert journal.backend.find("self_edits") == []


def test_gateway_completion_signal_returns_completion_intent(journal):
    task = _ready(journal)
    doc = _doc([], signal={"kind": "task_complete",
                           "detail": "plan read"})
    transport = _transport((200, _gateway_body(doc)))
    proposer = _proposer(journal, task, transport)
    batch = proposer.propose(proposer.perceive(journal), ctx=None)
    assert batch is not None and len(batch) == 1
    assert batch[0].kind == "memory"
    assert batch[0].record["kind"] == "task_complete"
    assert proposer.last_outcome["task_signal"]["kind"] == \
        "task_complete"


def test_gateway_inability_signal_returns_none_with_sublabel(journal):
    task = _ready(journal)
    doc = _doc([], signal={"kind": "cannot_proceed",
                           "detail": "no readable plan"})
    transport = _transport((200, _gateway_body(doc)))
    proposer = _proposer(journal, task, transport)
    assert proposer.propose(proposer.perceive(journal)) is None
    outcome = proposer.last_outcome
    assert outcome["outcome"] == "ok"
    translation = outcome["parse"]["signal_translation"]
    assert translation["stop_family"] == "proposer-exhausted"
    assert translation["sub_label"] == "model-reported-inability"


def test_catalogue_rerender_rides_the_request(journal):
    import copy as _copy
    import si.gaia.registry as registry_module
    task = _ready(journal)
    registry = dict(registry_module.REGISTRY)
    proposer = _proposer(journal, task, _transport(
        (200, _gateway_body(_doc([_intent(tool="fs_search",
                                          args={"path": ".",
                                                "pattern": "plan"})])))),
        registry=registry)
    perceptions = proposer.perceive(journal)
    old_digest = perceptions[0]["catalogue_digest"]
    del registry["fs_read"]  # registry mutates after perception
    batch = proposer.propose(perceptions, ctx=None)
    assert batch is not None and batch[0].tool == "fs_search"
    (attempt,) = _attempts(journal)
    fresh = render_catalogue(registry)["digest"]
    assert fresh != old_digest
    assert attempt["catalogue_rerendered"] is True
    assert attempt["perception"]["catalogue_digest"] == fresh
    leg = attempt["preconditions"]["legs"]["catalogue"]
    assert leg["passed"] is True and "re-render" in leg["detail"]


def test_check_preconditions_is_a_pure_pricer(journal):
    task = _ready(journal)
    perception = HearthProposer(task_seq=task["seq"]).perceive(journal)[0]
    check = check_preconditions(
        perception, journal=journal, task_seq=task["seq"],
        transport=_transport(), gateway_base_url=BASE_URL,
        model=MODEL, served_models=[MODEL],
        policy={"proposer_enabled": True}, clock=_clock)
    assert check["outcome"] == "pass"
    assert check["failed_leg"] is None
    assert set(check["legs"]) == {
        "transport", "leash", "budgets", "identity", "residency",
        "declaration", "catalogue"}


def test_build_gateway_request_shape(journal):
    task = _ready(journal)
    perception = HearthProposer(task_seq=task["seq"]).perceive(journal)[0]
    request = build_gateway_request(perception, model=MODEL)
    assert request["model"] == MODEL
    assert [m["role"] for m in request["messages"]] == \
        ["system", "user"]
    assert "task_complete" in request["messages"][1]["content"]
    assert "inspect the work root" in request["messages"][1]["content"]


# -- §13.2 two-failure halts ----------------------------------------------------

def test_two_parse_failures_halt_proposing(journal):
    task = _ready(journal)
    bad = _gateway_body("this is not a proposal document")
    transport = _transport((200, bad), (200, bad),
                           (200, _gateway_body(VALID_DOC)))
    proposer = _proposer(journal, task, transport)
    perceptions = proposer.perceive(journal)
    assert proposer.propose(perceptions) is None
    assert proposer.last_outcome["outcome"] == "parse-failed"
    assert proposer.propose(perceptions) is None
    assert proposer.last_outcome["outcome"] == "parse-failed"
    assert proposer.propose(perceptions) is None
    outcome = proposer.last_outcome
    assert outcome["outcome"] == "halted"
    assert outcome["failure_kind"] == "parse-failed"
    assert "two consecutive parse failures" in outcome["halt_cause"]
    assert len(transport.calls) == 2  # the halt made no call
    stages = [r["parse"]["stage"] for r in _responses(journal)]
    assert stages == [1, 1]  # envelope failures, journaled (§9.5)


def test_two_transport_failures_halt_proposing(journal):
    task = _ready(journal)
    transport = _transport(RuntimeError("gateway down"),
                           (503, "unavailable"),
                           (200, _gateway_body(VALID_DOC)))
    proposer = _proposer(journal, task, transport)
    perceptions = proposer.perceive(journal)
    assert proposer.propose(perceptions) is None
    assert proposer.last_outcome["outcome"] == TRANSPORT_FAILED
    assert "RuntimeError" in proposer.last_outcome["detail"]
    assert proposer.propose(perceptions) is None
    assert proposer.last_outcome["outcome"] == TRANSPORT_FAILED
    assert "HTTP 503" in proposer.last_outcome["detail"]
    assert proposer.propose(perceptions) is None
    outcome = proposer.last_outcome
    assert outcome["outcome"] == "halted"
    assert outcome["failure_kind"] == TRANSPORT_FAILED
    assert len(transport.calls) == 2
    kinds = [r["failure_kind"] for r in _responses(journal)]
    assert kinds == [TRANSPORT_FAILED, TRANSPORT_FAILED]


def test_second_precondition_failure_halts_proposing(journal):
    task = _ready(journal, state="Loading")
    transport = _transport((200, _gateway_body(VALID_DOC)))
    proposer = _proposer(journal, task, transport)
    perceptions = proposer.perceive(journal)
    assert proposer.propose(perceptions) is None
    assert proposer.last_outcome["outcome"] == NOT_EVALUATED
    assert proposer.propose(perceptions) is None
    outcome = proposer.last_outcome
    assert outcome["outcome"] == "halted"
    assert outcome["failure_kind"] == "residency-lost"
    assert "refresh allowance" in outcome["halt_cause"]
    assert "Loading" in outcome["halt_cause"]
    assert transport.calls == []  # neither cycle called


def test_attempt_budget_exhaustion_halts_without_a_call(journal):
    task = _ready(journal)
    transport = _transport((200, _gateway_body(VALID_DOC)),
                           (200, _gateway_body(VALID_DOC)))
    proposer = _proposer(
        journal, task, transport,
        policy={"proposer_enabled": True,
                "proposer_max_attempts_per_task": 1})
    perceptions = proposer.perceive(journal)
    assert proposer.propose(perceptions) is not None
    assert proposer.propose(perceptions) is None
    outcome = proposer.last_outcome
    assert outcome["outcome"] == "halted"
    assert outcome["failure_kind"] == "attempt-budget-exhausted"
    assert len(transport.calls) == 1


def test_oversize_response_is_a_transport_class_failure(journal):
    task = _ready(journal)
    transport = _transport((200, _gateway_body(VALID_DOC)))
    proposer = _proposer(
        journal, task, transport,
        policy={"proposer_enabled": True,
                "proposer_max_response_bytes": 32})
    assert proposer.propose(proposer.perceive(journal)) is None
    outcome = proposer.last_outcome
    assert outcome["outcome"] == TRANSPORT_FAILED
    assert "proposer_max_response_bytes" in outcome["detail"]
    (response,) = _responses(journal)
    assert response["failure_kind"] == TRANSPORT_FAILED
    assert response["parse"] is None  # never reached the parser


# -- §13.3 leash engaged at call -------------------------------------------------

class _LeashAtCallTransport:
    """A replay transport that engages the operator leash
    (journals a queue stop) while the gateway call is in
    flight, then returns a valid proposal (§13.3's fixture)."""

    label = "replay/stub — leash-at-call fixture (SI-66)"
    is_replay = True

    def __init__(self, journal, task_seq):
        self.journal = journal
        self.task_seq = task_seq
        self.calls = []

    def __call__(self, method, url, *, json_body=None, timeout_s=5.0):
        self.calls.append({"method": method, "url": url,
                           "json_body": json_body})
        record_queue_stop(self.journal, self.task_seq,
                          "operator stop during the gateway call")
        return 200, json.dumps(_gateway_body(VALID_DOC)).encode()


def test_response_arriving_under_leash_is_discarded_whole(journal):
    task = _ready(journal)
    transport = _LeashAtCallTransport(journal, task["seq"])
    proposer = _proposer(journal, task, transport)
    assert proposer.propose(proposer.perceive(journal)) is None
    outcome = proposer.last_outcome
    assert outcome["outcome"] == DISCARDED_UNDER_LEASH
    assert len(transport.calls) == 1  # the call did happen
    (response,) = _responses(journal)
    assert response["outcome"] == DISCARDED_UNDER_LEASH
    assert response["failure_kind"] == DISCARDED_UNDER_LEASH
    assert response["response_digest"] is not None  # journaled
    assert response["batch"] is None  # never parsed for admission
    # Nothing was admitted: no intent was journaled or gated.
    assert journal.backend.find("self_edits") == []


def test_leash_callable_engaged_at_call_discards(journal):
    task = _ready(journal)
    state = {"engaged": False}

    class FlipTransport:
        label = "replay/stub — leash callable fixture (SI-66)"
        is_replay = True

        def __init__(self):
            self.calls = []

        def __call__(self, method, url, *, json_body=None,
                     timeout_s=5.0):
            self.calls.append({"method": method, "url": url})
            state["engaged"] = True
            return 200, json.dumps(_gateway_body(VALID_DOC)).encode()

    transport = FlipTransport()
    proposer = _proposer(journal, task, transport,
                         leash_check=lambda: state["engaged"])
    assert proposer.propose(proposer.perceive(journal)) is None
    assert proposer.last_outcome["outcome"] == DISCARDED_UNDER_LEASH
    assert journal.backend.find("self_edits") == []


# -- run_gateway_proposal as a direct entry point -------------------------------

def test_run_gateway_proposal_direct_refusal(journal):
    task = _ready(journal, state="Stopped")
    perception = HearthProposer(task_seq=task["seq"]).perceive(journal)[0]
    transport = _transport((200, _gateway_body(VALID_DOC)))
    outcome = run_gateway_proposal(
        journal, perception, task_seq=task["seq"],
        transport=transport, gateway_base_url=BASE_URL,
        model=MODEL, served_models=[MODEL],
        policy={"proposer_enabled": True}, clock=_clock)
    assert outcome["outcome"] == NOT_EVALUATED
    assert outcome["failed_leg"] == "residency"
    assert transport.calls == []


# -- regression: scripted + miniature proposers untouched -----------------------

def test_run_task_scripted_proposer_unchanged(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "mini.si"),
                             name="gaia-p3-regression",
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    work = tmp_path / "work"
    work.mkdir()
    (work / "note.txt").write_text("hello\n")
    policy = corpus_policy(str(work))
    assert "proposer_enabled" not in policy  # absent = off (§15.2)
    script = [
        Intent(kind="tool", tool="fs_read",
               args={"path": "note.txt"}, reason="read the note"),
        Intent(kind="memory", collection="experiences",
               record={"kind": "task_complete",
                       "detail": "read it"}, reason="done"),
    ]
    res = run_task(bundle_dir, script, "read note.txt",
                   work_root=str(work), policy=policy)
    assert res["stop_reason"] == "mind-reported-complete"
    assert [s["verdict"] for s in res["steps"]] == ["ADMIT", "ADMIT"]


def test_run_task_mind_proposer_unchanged(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "mini.si"),
                             name="gaia-p3-mind-regression",
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    work = tmp_path / "work"
    work.mkdir()

    class ScriptedMind:
        def perceive(self, journal):
            return []

        def propose(self, perceptions, ctx):
            task = perceptions[0]
            return Intent(
                kind="memory", collection="experiences",
                record={"kind": "task_complete"},
                reason="reporting completion",
                parents=[task["ref"]], source_seqs=[task["seq"]])

    res = run_task(bundle_dir, ScriptedMind(), "finish at once",
                   work_root=str(work),
                   policy=corpus_policy(str(work)))
    assert res["stop_reason"] == "mind-reported-complete"
