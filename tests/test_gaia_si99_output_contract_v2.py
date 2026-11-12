"""SI-99 (D-F1 §4.4, opened by lead decision 2026-11-12 for
exactly one change): Gaia PROPOSE — the output-contract
instruction now teaches §8 version 2.

The §4.4 class is the statement of the §8 document shape to
the model (``OUTPUT_CONTRACT_INSTRUCTION``, appended to the
§6.5 gateway request's user message); the parser (§9) is the
contract's enforcement. SI-98 implemented the v2 sibling
form; this slice changes only the statement, so the model is
told the form it may now use:

- the taught template is version 2;
- the journal-seq depends_on documentation is carried
  verbatim from the v1 statement, with its use named (an
  intent journaled in a prior cycle);
- the sibling form is documented beside it: the entry
  shape, the 0-based position in this same document, its
  use (an earlier step of this proposal), the
  strictly-earlier rule, and that the two forms may be
  mixed in one depends_on list;
- every other sentence of the statement is unchanged.

Coverage:

- the instruction constant: v2 template, seq documentation
  verbatim, sibling documentation complete, untouched
  sentences verbatim;
- the assembled gateway request carries the amended
  instruction as the tail of its user message, over a real
  assembled perception, with the standing instructions
  (the closed §4.3 class) as the system message;
- replay, end to end through the current contract path: a
  scripted reply written exactly as the amended contract
  teaches — a v2 document whose later intents depend on an
  earlier step via the sibling form, one of them mixing a
  prior cycle's journal seq — parses and resolves through
  the replay walk: resolved ``depends_on`` journaled,
  ``authored_depends_on`` verbatim, chain verified;
- a v1 document still parses under the amended statement
  (the parser's known set is {1, 2}, SI-97 §9.2).

Honesty discipline: every exercise here is the replay path
with scripted responses and stub admission, labelled as
such. No live model run, no conformance claim — the
re-measurement is a separate slice (SI-100).
"""
import json

import pytest

from si.gaia.proposer import (
    OUTPUT_CONTRACT_INSTRUCTION,
    STANDING_INSTRUCTIONS,
    HearthProposer,
    build_gateway_request,
    parse_proposal,
    run_replay_proposal,
)
from si.gaia.task import record_task_entry
from si.journal import EmbeddedBackend, Journal

MODEL = "test-model-1"
FRESH_AT = "2026-10-31T11:59:30Z"

#: The v1 statement's depends_on sentence, verbatim — the
#: seq documentation SI-99 must carry unchanged in substance.
SEQ_SENTENCE = (
    "depends_on entries name journal sequences of this "
    "task's prior intents, as bare integers or "
    '{"seq": <int>, "requirement": "outcome"|"success"}.')


# -- shared fixtures -------------------------------------------------------

@pytest.fixture()
def journal(tmp_path):
    j = Journal(EmbeddedBackend(str(tmp_path / "journal")))
    yield j
    j.close()


def _task(journal, instruction="inspect the work root"):
    return record_task_entry(
        journal, instruction, "/tmp/si99-work",
        budgets={"max_tool_cycles_per_task": 16,
                 "tool_max_task_result_bytes": 1048576})


def _ready(journal):
    task = _task(journal)
    journal.record_experience("hearth_residency", {
        "kind": "hearth_residency",
        "source": "hearth:/residency", "state": "Resident",
        "raw_state": "Resident", "model": MODEL,
        "node": "node-a", "polled_at": FRESH_AT}, "")
    return task


def _perception(journal, task):
    return HearthProposer(task_seq=task["seq"]).perceive(journal)[0]


def _intent(tool="fs_read", args=None, reason="read the plan file",
            depends_on=None):
    return {"tool": tool,
            "args": args if args is not None else {"path": "plan.txt"},
            "reason": reason,
            "depends_on": depends_on if depends_on is not None else [],
            "ordering_key": None}


def _prior_intent(journal, task):
    """An intent of this task journaled in a prior cycle —
    the seq form's legitimate target."""
    intent = {
        "kind": "tool", "tool": "fs_read", "args": {"path": "a.txt"},
        "parents": [task["ref"]], "source_seqs": [task["seq"]],
        "reason": "prior intent (fixture)",
    }
    edit_id = journal.record_self_edit(
        "tool", intent, intent["reason"], "ADMIT",
        {"verdict": "ADMIT", "checks": []})
    rows = journal.backend.find(
        "self_edits", filter={"edit_id": edit_id}, limit=1)
    return rows[0]["seq"]


def _stub_admit(journal):
    seen = []

    def admit(intent):
        seen.append(intent)
        edit_id = journal.record_self_edit(
            intent.kind, intent.to_dict(), intent.reason,
            "ADMIT", {"verdict": "ADMIT", "checks": []})
        rows = journal.backend.find(
            "self_edits", filter={"edit_id": edit_id}, limit=1)
        return rows[0]["seq"]

    admit.seen = seen
    return admit


# -- the statement itself ----------------------------------------------------

def test_instruction_teaches_the_v2_template():
    assert '{"proposal": {"version": 2, "intents"' in \
        OUTPUT_CONTRACT_INSTRUCTION
    assert '"version": 1' not in OUTPUT_CONTRACT_INSTRUCTION


def test_instruction_keeps_the_seq_documentation():
    # The v1 statement's seq sentence, verbatim, and its use
    # named: an intent journaled in a prior cycle.
    assert SEQ_SENTENCE in OUTPUT_CONTRACT_INSTRUCTION
    assert "journaled in a prior cycle" in \
        OUTPUT_CONTRACT_INSTRUCTION


def test_instruction_teaches_the_sibling_form():
    text = OUTPUT_CONTRACT_INSTRUCTION
    assert '{"sibling": <int>, "requirement": ' \
        '"outcome"|"success"}' in text
    assert "earlier intent in this same document" in text
    assert "0-based position" in text
    assert "earlier step of this proposal" in text
    # The strictly-earlier rule (SI-97 §4.3).
    assert "strictly less than the position of the intent " \
        "naming it" in text
    # Mixing the two forms in one list is allowed (§9.3).
    assert "mixed in one depends_on list" in text


def test_instruction_untouched_sentences_verbatim():
    text = OUTPUT_CONTRACT_INSTRUCTION
    assert text.startswith(
        "Respond with exactly one bare JSON document of "
        "the form ")
    assert '"intents" may be empty only when "task_signal" is ' \
        '{"kind": "task_complete", "detail": "<what was done>"} ' \
        'or {"kind": "cannot_proceed", "detail": "<why not>"}.' \
        in text


# -- the assembled request carries it ----------------------------------------

def test_gateway_request_carries_the_amended_contract(journal):
    task = _ready(journal)
    request = build_gateway_request(
        _perception(journal, task), model=MODEL)
    assert [m["role"] for m in request["messages"]] == \
        ["system", "user"]
    # The standing instructions (§4.3, a closed class) are
    # the system message, untouched by this slice.
    assert request["messages"][0]["content"] == \
        STANDING_INSTRUCTIONS
    user = request["messages"][1]["content"]
    assert user.endswith(OUTPUT_CONTRACT_INSTRUCTION)
    assert '"version": 2' in user
    assert '{"sibling": <int>' in user
    assert SEQ_SENTENCE in user


# -- replay through the current contract path --------------------------------

def test_replay_reply_as_taught_parses_and_resolves(journal):
    task = _ready(journal)
    # The request a proposer is actually sent carries the
    # amended contract (above); the scripted reply below is
    # written exactly as that contract teaches.
    request = build_gateway_request(
        _perception(journal, task), model=MODEL)
    assert request["messages"][1]["content"].endswith(
        OUTPUT_CONTRACT_INSTRUCTION)

    prior_seq = _prior_intent(journal, task)
    raw = json.dumps({"proposal": {
        "version": 2,
        "intents": [
            _intent(reason="first read", args={"path": "a.txt"}),
            _intent(reason="second read, after the first",
                    args={"path": "b.txt"},
                    depends_on=[{"sibling": 0,
                                 "requirement": "success"}]),
            _intent(reason="third read, after the prior "
                           "cycle's read and the second step",
                    args={"path": "c.txt"},
                    depends_on=[{"seq": prior_seq},
                                {"sibling": 1,
                                 "requirement": "success"}]),
        ],
        "task_signal": None}})
    admit = _stub_admit(journal)
    out = run_replay_proposal(journal, raw, task_seq=task["seq"],
                              admit=admit)
    assert out["outcome"] == "ok", out["parse"]["detail"]
    assert len(out["intent_seqs"]) == 3
    first_seq, second_seq, _third_seq = out["intent_seqs"]
    rows = journal.backend.find("self_edits")
    by_seq = {r["seq"]: r for r in rows}
    # The sibling form resolved to the walk's own seqs; the
    # authored form is journaled verbatim beside it (§8.1).
    assert by_seq[second_seq]["intent"]["depends_on"] == [
        {"seq": first_seq, "requirement": "success"}]
    assert by_seq[second_seq]["intent"]["authored_depends_on"] == [
        {"sibling": 0, "requirement": "success"}]
    third = admit.seen[2]
    assert third.depends_on == [
        {"seq": prior_seq, "requirement": "outcome"},
        {"seq": second_seq, "requirement": "success"}]
    assert third.authored_depends_on == [
        {"seq": prior_seq, "requirement": "outcome"},
        {"sibling": 1, "requirement": "success"}]
    assert journal.verify_chain()


def test_v1_document_still_parses(journal):
    task = _task(journal)
    raw = json.dumps({"proposal": {
        "version": 1, "intents": [_intent()],
        "task_signal": None}})
    res = parse_proposal(raw, journal=journal,
                         task_seq=task["seq"])
    assert res["outcome"] == "ok", res["detail"]
