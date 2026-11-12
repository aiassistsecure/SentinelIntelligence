"""SI-98 (SI-97, D-96-1): Gaia PROPOSE — the §9 intra-document
dependency amendment, implemented.

Coverage, mapped to SI-97 §11's test obligations:

- v1 regression: the SI-94 frozen corpus, re-scored through
  the amended parser as a parser regression check, reproduces
  its recorded stage outcomes item-for-item (20/30, same
  failing stages) — a test result, not a record; it does not
  re-grade SI-94;
- form: each §5.1 detail class refuses at stage 4 with its
  pinned problem string; a both-keys entry refuses; a
  ``sibling`` key in a v1 document fails stage 2; a v2
  document under v1 rules (known versions narrowed to {1})
  fails stage 2; valid fan-in, a maximal chain (n − 1 edges
  at the §15.2 batch bound), and mixed seq+sibling documents
  parse; a bare int never means a sibling;
- resolution (replay walk, stub admission): sibling entries
  resolve in place with authored order preserved and
  ``authored_depends_on`` verbatim (§8.1); walk order ==
  document order (§4.2); a REJECTED sibling and an ESCALATED
  sibling each still resolve (§6.3, verdict-blind); admission
  returning no seq produces exactly one refusal record per
  affected member and the cascade reaches the next dependent
  (§6.5); with no admit callback at all, dependents refuse
  with the no-member-admitted sub-case;
- gate invariant (§7): a spy gate asserts no ``sibling`` key
  in any intent it evaluates, across the walk tests; and the
  gate itself refuses an unresolved sibling entry closed
  (the §7 backstop — check 2 fails on it);
- journal shape (§8): the §14.6 chain walk surfaces refusal
  records; chaining counts split three ways (§8.4); the
  refusal record carries no verdict and no gate report;
- live walk (run_task): a scripted batch's sibling entry is
  resolved before the gate and journaled resolved +
  authored; a member whose batch position names a non-tool
  member is refused at resolution — never gated, never
  queued — and the task still completes;
- queue fold (§6.3/§7.3): a resolved dependent is priced
  exactly as a seq-authored one — the ``success`` dependent
  of a gate-REJECTED sibling is skipped at drain (the skip
  journaled), the ``outcome`` dependent executes.

Honesty discipline: every end-to-end exercise here is the
REPLAY path or a scripted task — stub admission and scripted
proposers, labelled as such. No live model run, no
conformance claim: SI-97 §12.3 authorises replay evidence
only, and this module is that evidence's test half.
"""
import json
import os

import pytest

from si.bundle import init_bundle, load_bundle
from si.gaia.conformance import score_corpus
from si.gaia.context import ToolContext
from si.gaia.corpus import corpus_policy
from si.gaia.drain import Drainer, admit_tool_intent
from si.gaia.proposer import (
    REPLAY_ONLY_LABEL,
    parse_proposal,
    run_replay_proposal,
    walk_attempt_chain,
)
from si.gaia.queue import fold_queue, record_queue_batch
from si.gaia.shadow import record_task_fixture
from si.gaia.sibling_resolution import (
    RESOLUTION_REFUSAL_CAUSE,
    RESOLUTION_REFUSAL_KIND,
    SUBCASE_ADMISSION_NO_SEQ,
    SUBCASE_NO_MEMBER_ADMITTED,
    SUBCASE_SIBLING_REFUSED,
    resolve_sibling_dependencies,
)
from si.gaia.task import record_task_entry, run_task
from si.gate import Gate, Intent
from si.journal import EmbeddedBackend, Journal, open_journal
from si.tiny_seed import seed_tiny_bundle


# -- shared fixtures -------------------------------------------------------

@pytest.fixture()
def journal(tmp_path):
    j = Journal(EmbeddedBackend(str(tmp_path / "journal")))
    yield j
    j.close()


@pytest.fixture()
def mini(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "mini.si"),
                             name="gaia-si98",
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    work = tmp_path / "work"
    work.mkdir()
    (work / "notes").mkdir()
    (work / "notes" / "plan.md").write_text("# plan\nstep one\n")
    (work / "note.txt").write_text("hello\n")
    return bundle_dir, str(work)


def _task(journal, instruction="inspect the work root"):
    return record_task_entry(
        journal, instruction, "/tmp/si98-work",
        budgets={"max_tool_cycles_per_task": 16,
                 "tool_max_task_result_bytes": 1048576})


def _intent(tool="fs_read", args=None, reason="read the plan file",
            depends_on=None, ordering_key=None):
    return {"tool": tool,
            "args": args if args is not None else {"path": "plan.txt"},
            "reason": reason,
            "depends_on": depends_on if depends_on is not None else [],
            "ordering_key": ordering_key}


def _doc(intents, signal=None, version=2):
    return json.dumps({"proposal": {
        "version": version, "intents": intents,
        "task_signal": signal}})


def _prior_intent(journal, task, tool="fs_read", args=None):
    intent = {
        "kind": "tool", "tool": tool,
        "args": args if args is not None else {"path": "a.txt"},
        "parents": [task["ref"]], "source_seqs": [task["seq"]],
        "reason": "prior intent (fixture)",
    }
    edit_id = journal.record_self_edit(
        "tool", intent, intent["reason"], "ADMIT",
        {"verdict": "ADMIT", "checks": []})
    rows = journal.backend.find(
        "self_edits", filter={"edit_id": edit_id}, limit=1)
    return rows[0]["seq"]


def _refusals(journal):
    return journal.backend.find(
        "experiences", filter={"kind": RESOLUTION_REFUSAL_KIND}) or []


def _stub_admit(journal, verdicts=None, none_for=()):
    """A stub admission in the replay walk's shape: journals
    each intent it is handed via record_self_edit under a
    scripted verdict and returns the journal seq — or None
    for positions in ``none_for`` (the caller's positions are
    counted in call order). Records every intent it saw, in
    order, on ``.seen``."""
    seen = []

    def admit(intent):
        position = len(seen)
        seen.append(intent)
        if position in none_for:
            return None
        verdict = (verdicts or {}).get(position, "ADMIT")
        edit_id = journal.record_self_edit(
            intent.kind, intent.to_dict(), intent.reason,
            verdict, {"verdict": verdict, "checks": []})
        rows = journal.backend.find(
            "self_edits", filter={"edit_id": edit_id}, limit=1)
        return rows[0]["seq"]

    admit.seen = seen
    return admit


# -- form: acceptance ------------------------------------------------------

def test_v2_sibling_document_parses(journal):
    task = _task(journal)
    raw = _doc([
        _intent(reason="first read"),
        _intent(reason="second read",
                depends_on=[{"sibling": 0,
                             "requirement": "success"}]),
        _intent(reason="third read",
                depends_on=[{"sibling": 0}]),
    ])
    res = parse_proposal(raw, journal=journal, task_seq=task["seq"])
    assert res["outcome"] == "ok", res["detail"]
    # §8.2: constructed intents carry the sibling entries in
    # normalised authored form — visibly unresolved.
    assert res["intents"][1].depends_on == [
        {"sibling": 0, "requirement": "success"}]
    assert res["intents"][2].depends_on == [
        {"sibling": 0, "requirement": "outcome"}]


def test_valid_fan_in_parses(journal):
    task = _task(journal)
    raw = _doc([
        _intent(reason="read a", args={"path": "a.txt"}),
        _intent(reason="read b", args={"path": "b.txt"}),
        _intent(tool="fs_write", reason="write the merge",
                args={"path": "out.txt", "content": "x\n"},
                depends_on=[{"sibling": 0},
                            {"sibling": 1,
                             "requirement": "success"}]),
    ])
    res = parse_proposal(raw, journal=journal, task_seq=task["seq"])
    assert res["outcome"] == "ok", res["detail"]
    assert res["intents"][2].depends_on == [
        {"sibling": 0, "requirement": "outcome"},
        {"sibling": 1, "requirement": "success"}]


def test_maximal_chain_at_batch_bound_parses(journal):
    task = _task(journal)
    intents = [_intent(reason="step 0", args={"path": "s0.txt"})]
    for i in range(1, 16):
        intents.append(_intent(
            reason=f"step {i}", args={"path": f"s{i}.txt"},
            depends_on=[{"sibling": i - 1}]))
    res = parse_proposal(_doc(intents), journal=journal,
                         task_seq=task["seq"])
    assert res["outcome"] == "ok", res["detail"]
    assert len(res["intents"]) == 16  # the §15.2 bound itself


def test_mixed_seq_and_sibling_parses(journal):
    task = _task(journal)
    prior_seq = _prior_intent(journal, task)
    raw = _doc([
        _intent(reason="this document's read"),
        _intent(reason="depends on both worlds",
                depends_on=[{"seq": prior_seq,
                             "requirement": "success"},
                            {"sibling": 0}]),
    ])
    res = parse_proposal(raw, journal=journal, task_seq=task["seq"])
    assert res["outcome"] == "ok", res["detail"]
    assert res["intents"][1].depends_on == [
        {"seq": prior_seq, "requirement": "success"},
        {"sibling": 0, "requirement": "outcome"}]


def test_bare_int_never_means_sibling(journal):
    task = _task(journal)
    raw = _doc([
        _intent(reason="first"),
        _intent(reason="cites bare 1", depends_on=[1]),
    ])
    res = parse_proposal(raw, journal=journal, task_seq=task["seq"])
    # Priced as a journal seq (the fold path), never as a
    # position: seq 1 is not a tool intent of this task, so
    # the batch fails closed at stage 4 on the seq rules.
    assert res["outcome"] == "parse-failed"
    assert res["failing_stage"] == 4
    assert "sibling" not in res["detail"]


# -- form: the §5.1 detail classes -----------------------------------------

@pytest.mark.parametrize("dep,klass", [
    ({"sibling": "0"}, "sibling_not_an_index"),
    ({"sibling": True}, "sibling_not_an_index"),
    ({"sibling": 0.5}, "sibling_not_an_index"),
    ({"sibling": -1}, "sibling_negative_index"),
    ({"sibling": 0, "requirement": "whenever"},
     "sibling_requirement_invalid"),
    ({"seq": 5, "sibling": 0}, "entry_names_both_seq_and_sibling"),
    ({"sibling": 1}, "sibling_self_reference"),
    ({"sibling": 7}, "sibling_out_of_range"),
])
def test_stage4_sibling_detail_classes(journal, dep, klass):
    task = _task(journal)
    raw = _doc([
        _intent(reason="first"),
        _intent(reason="citing member", depends_on=[dep]),
    ])
    res = parse_proposal(raw, journal=journal, task_seq=task["seq"])
    assert res["outcome"] == "parse-failed"
    assert res["failing_stage"] == 4
    assert klass in res["detail"]
    assert res["intents"] == []  # whole-batch refusal (§9.5)


def test_stage4_sibling_forward_reference(journal):
    task = _task(journal)
    raw = _doc([
        _intent(reason="cites forward", depends_on=[{"sibling": 1}]),
        _intent(reason="second"),
        _intent(reason="third"),
    ])
    res = parse_proposal(raw, journal=journal, task_seq=task["seq"])
    assert res["failing_stage"] == 4
    assert "sibling_forward_reference" in res["detail"]


def test_stage4_sibling_declared_twice(journal):
    task = _task(journal)
    raw = _doc([
        _intent(reason="first"),
        _intent(reason="second"),
        _intent(reason="cites 0 twice, differing requirement",
                depends_on=[{"sibling": 0},
                            {"sibling": 0,
                             "requirement": "success"}]),
    ])
    res = parse_proposal(raw, journal=journal, task_seq=task["seq"])
    assert res["failing_stage"] == 4
    assert "sibling_declared_twice" in res["detail"]


def test_sibling_key_in_v1_document_fails_stage2(journal):
    task = _task(journal)
    raw = _doc([
        _intent(reason="first"),
        _intent(reason="cites a sibling",
                depends_on=[{"sibling": 0}]),
    ], version=1)
    res = parse_proposal(raw, journal=journal, task_seq=task["seq"])
    assert res["outcome"] == "parse-failed"
    assert res["failing_stage"] == 2
    assert "unknown field(s) ['sibling']" in res["detail"]


def test_v2_document_under_v1_rules_fails_stage2(journal, monkeypatch):
    import si.gaia.proposer as proposer
    task = _task(journal)
    raw = _doc([_intent(reason="only member")], version=2)
    monkeypatch.setattr(proposer, "PROPOSAL_VERSIONS", (1,))
    res = parse_proposal(raw, journal=journal, task_seq=task["seq"])
    assert res["outcome"] == "parse-failed"
    assert res["failing_stage"] == 2
    assert "not a known version" in res["detail"]


def test_v1_document_parses_as_before(journal):
    task = _task(journal)
    prior_seq = _prior_intent(journal, task)
    raw = _doc([
        _intent(reason="v1 member",
                depends_on=[{"seq": prior_seq,
                             "requirement": "success"}]),
        _intent(reason="v1 bare int member", depends_on=[]),
    ], version=1)
    res = parse_proposal(raw, journal=journal, task_seq=task["seq"])
    assert res["outcome"] == "ok", res["detail"]
    assert res["intents"][0].depends_on == [
        {"seq": prior_seq, "requirement": "success"}]
    assert res["intents"][0].authored_depends_on == []


# -- v1 regression: the SI-94 frozen corpus --------------------------------

def test_si94_corpus_reparse_reproduces_recorded_stages():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    with open(os.path.join(repo, "docs", "SI-94-corpus.json")) as f:
        corpus = json.load(f)
    with open(os.path.join(repo, "docs",
                           "SI-94-measurements.json")) as f:
        recorded = json.load(f)
    expected = {item["corpus_item_id"]: item
                for item in recorded["results"]["items"]}
    _fold, per_item = score_corpus(corpus)
    assert len(per_item) == 30
    assert sum(1 for i in per_item if i["parse_ok"]) == 20
    for item in per_item:
        exp = expected[item["corpus_item_id"]]
        assert item["stage_reached"] == exp["stage_reached"], \
            item["corpus_item_id"]
        assert item["failing_stage"] == exp["failing_stage"], \
            item["corpus_item_id"]
        assert item["parse_ok"] == exp["parse_ok"], \
            item["corpus_item_id"]


# -- resolution: the replay walk -------------------------------------------

def test_replay_walk_resolves_sibling_in_place(journal):
    task = _task(journal)
    admit = _stub_admit(journal)
    raw = _doc([
        _intent(reason="first read", args={"path": "a.txt"}),
        _intent(reason="second read", args={"path": "b.txt"},
                depends_on=[{"sibling": 0,
                             "requirement": "success"}]),
    ])
    out = run_replay_proposal(journal, raw, task_seq=task["seq"],
                              admit=admit)
    assert out["outcome"] == "ok"
    assert len(out["intent_seqs"]) == 2
    first_seq, second_seq = out["intent_seqs"]
    rows = journal.backend.find("self_edits")
    by_seq = {r["seq"]: r for r in rows}
    # §8.1: the effective list is resolved; the authored list
    # is verbatim beside it; a member with no sibling entry
    # gains no authored field.
    assert by_seq[first_seq]["intent"].get(
        "authored_depends_on") is None
    assert by_seq[second_seq]["intent"]["depends_on"] == [
        {"seq": first_seq, "requirement": "success"}]
    assert by_seq[second_seq]["intent"]["authored_depends_on"] == [
        {"sibling": 0, "requirement": "success"}]
    # §7 invariant, journal side: no journaled effective
    # depends_on contains a sibling key.
    for row in rows:
        for entry in row["intent"]["depends_on"]:
            assert not (isinstance(entry, dict)
                        and "sibling" in entry)
    assert _refusals(journal) == []
    assert journal.verify_chain()


def test_replay_walk_authored_order_preserved_across_mixture(journal):
    task = _task(journal)
    prior_seq = _prior_intent(journal, task)
    admit = _stub_admit(journal)
    raw = _doc([
        _intent(reason="member zero", args={"path": "a.txt"}),
        _intent(reason="member one", args={"path": "b.txt"}),
        _intent(reason="member two", args={"path": "c.txt"},
                depends_on=[{"seq": prior_seq,
                             "requirement": "success"},
                            {"sibling": 1},
                            {"sibling": 0,
                             "requirement": "success"}]),
    ])
    out = run_replay_proposal(journal, raw, task_seq=task["seq"],
                              admit=admit)
    assert out["outcome"] == "ok", out["parse"]["detail"]
    seq0, seq1, _seq2 = out["intent_seqs"]
    third = admit.seen[2]
    assert third.depends_on == [
        {"seq": prior_seq, "requirement": "success"},
        {"seq": seq1, "requirement": "outcome"},
        {"seq": seq0, "requirement": "success"}]
    assert third.authored_depends_on == [
        {"seq": prior_seq, "requirement": "success"},
        {"sibling": 1, "requirement": "outcome"},
        {"sibling": 0, "requirement": "success"}]


def test_walk_order_is_document_order(journal):
    task = _task(journal)
    admit = _stub_admit(journal)
    raw = _doc([
        _intent(reason=f"member {i}", args={"path": f"m{i}.txt"},
                depends_on=([{"sibling": i - 1}] if i else []))
        for i in range(5)
    ])
    out = run_replay_proposal(journal, raw, task_seq=task["seq"],
                              admit=admit)
    assert out["outcome"] == "ok"
    assert [it.args["path"] for it in admit.seen] == [
        f"m{i}.txt" for i in range(5)]
    assert out["intent_seqs"] == sorted(out["intent_seqs"])


def test_rejected_and_escalated_siblings_still_resolve(journal):
    task = _task(journal)
    admit = _stub_admit(journal,
                        verdicts={0: "REJECT", 1: "ESCALATE"})
    raw = _doc([
        _intent(reason="will be rejected", args={"path": "a.txt"}),
        _intent(reason="will be escalated", args={"path": "b.txt"}),
        _intent(reason="depends on both", args={"path": "c.txt"},
                depends_on=[{"sibling": 0},
                            {"sibling": 1,
                             "requirement": "success"}]),
    ])
    out = run_replay_proposal(journal, raw, task_seq=task["seq"],
                              admit=admit)
    assert out["outcome"] == "ok"
    assert len(out["intent_seqs"]) == 3  # verdict-blind (§6.3)
    seq0, seq1, _seq2 = out["intent_seqs"]
    assert admit.seen[2].depends_on == [
        {"seq": seq0, "requirement": "outcome"},
        {"seq": seq1, "requirement": "success"}]
    assert _refusals(journal) == []


def test_admission_no_seq_refusal_and_cascade(journal):
    task = _task(journal)
    admit = _stub_admit(journal, none_for={0})
    raw = _doc([
        _intent(reason="admission returns no seq",
                args={"path": "a.txt"}),
        _intent(reason="depends on it", args={"path": "b.txt"},
                depends_on=[{"sibling": 0}]),
        _intent(reason="depends on the refused one",
                args={"path": "c.txt"},
                depends_on=[{"sibling": 1}]),
        _intent(reason="independent", args={"path": "d.txt"}),
    ])
    out = run_replay_proposal(journal, raw, task_seq=task["seq"],
                              admit=admit)
    assert out["outcome"] == "ok"
    refusals = _refusals(journal)
    assert len(refusals) == 2  # exactly one per refused member
    recs = [r["record"] for r in
            sorted(refusals, key=lambda r: r["record"]["position"])]
    assert [r["position"] for r in recs] == [1, 2]
    assert all(r["cause"] == RESOLUTION_REFUSAL_CAUSE
               for r in recs)
    assert recs[0]["sub_case"] == SUBCASE_ADMISSION_NO_SEQ
    assert recs[1]["sub_case"] == SUBCASE_SIBLING_REFUSED
    assert recs[0]["unresolved_entries"][0]["entry"] == {
        "sibling": 0, "requirement": "outcome"}
    # The refused members were never admitted; the walk
    # continued and admitted the independent member.
    assert [it.args["path"] for it in admit.seen] == [
        "a.txt", "d.txt"]
    assert len(out["intent_seqs"]) == 1
    # The response record carries the refused positions (§8.4).
    resp = journal.backend.find(
        "experiences", filter={"kind": "gaia_proposal_response"})
    assert resp[0]["record"]["batch"]["resolution_refused"] == [1, 2]
    assert journal.verify_chain()


def test_no_admit_callback_refuses_dependents(journal):
    task = _task(journal)
    raw = _doc([
        _intent(reason="never admitted", args={"path": "a.txt"}),
        _intent(reason="depends on it", args={"path": "b.txt"},
                depends_on=[{"sibling": 0}]),
    ])
    out = run_replay_proposal(journal, raw, task_seq=task["seq"])
    assert out["outcome"] == "ok"
    assert out["intent_seqs"] == []
    refusals = _refusals(journal)
    assert len(refusals) == 1
    rec = refusals[0]["record"]
    assert rec["position"] == 1
    assert rec["sub_case"] == SUBCASE_NO_MEMBER_ADMITTED
    assert journal.backend.find("self_edits") == []


def test_refusal_record_carries_no_verdict(journal):
    task = _task(journal)
    raw = _doc([
        _intent(reason="never admitted", args={"path": "a.txt"}),
        _intent(reason="depends on it", args={"path": "b.txt"},
                depends_on=[{"sibling": 0,
                             "requirement": "success"}]),
    ])
    run_replay_proposal(journal, raw, task_seq=task["seq"])
    rec = _refusals(journal)[0]["record"]
    assert rec["kind"] == RESOLUTION_REFUSAL_KIND
    assert rec["task_seq"] == task["seq"]
    assert "verdict" not in rec
    assert "gate_report" not in rec
    assert rec["replay_only"] is True
    assert rec["replay_label"] == REPLAY_ONLY_LABEL


def test_chain_walk_surfaces_refusals_and_chaining_split(journal):
    task = _task(journal)
    # Walk 1: a resolved batch — chaining splits three ways.
    admit = _stub_admit(journal)
    raw = _doc([
        _intent(reason="root read", args={"path": "a.txt"}),
        _intent(reason="chained read", args={"path": "b.txt"},
                depends_on=[{"sibling": 0}]),
    ])
    out = run_replay_proposal(journal, raw, task_seq=task["seq"],
                              admit=admit)
    chain = walk_attempt_chain(journal, out["response_seq"])
    assert chain["all_intents_found"]
    assert chain["resolution_refusals"] == []
    resp = journal.backend.find(
        "experiences", filter={"kind": "gaia_proposal_response"})
    chaining = resp[0]["record"]["batch"]["chaining"]
    assert chaining["model_authored"] == 1
    assert chaining["runner_filled"] == 1
    assert chaining["sibling_resolved"] == 1
    assert resp[0]["record"]["batch"]["resolution_refused"] == []
    # Walk 2: a refused member — the chain walk surfaces it.
    raw2 = _doc([
        _intent(reason="never admitted", args={"path": "x.txt"}),
        _intent(reason="depends on it", args={"path": "y.txt"},
                depends_on=[{"sibling": 0}]),
    ])
    out2 = run_replay_proposal(journal, raw2, task_seq=task["seq"])
    chain2 = walk_attempt_chain(journal, out2["response_seq"])
    assert chain2["resolution_refused_positions"] == [1]
    assert len(chain2["resolution_refusals"]) == 1
    assert chain2["resolution_refusals"][0]["record"]["cause"] == \
        RESOLUTION_REFUSAL_CAUSE


# -- the gate invariant (§7) ------------------------------------------------

def test_spy_gate_never_sees_a_sibling_key(journal, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "plan.txt").write_text("plan\n")
    task = _task(journal)
    gate = Gate(corpus_policy(str(work)))
    ctx = ToolContext(tool_work_root=str(work),
                      tool_task_seq=task["seq"],
                      tool_live_execution=True)
    evaluated = []
    admitted_seqs = {}

    def spy_admit(intent):
        evaluated.append(intent)
        for entry in intent.depends_on:
            assert not (isinstance(entry, dict)
                        and "sibling" in entry), entry
        ctx.tool_journal_tip = journal.tip()[0]
        ctx.tool_task_intent_seqs = set(admitted_seqs.values())
        report = gate.evaluate(intent, None, ctx)
        assert report.verdict == "ADMIT", report.to_dict()
        edit_id = journal.record_self_edit(
            intent.kind, intent.to_dict(), intent.reason,
            report.verdict, report.to_dict())
        rows = journal.backend.find(
            "self_edits", filter={"edit_id": edit_id}, limit=1)
        admitted_seqs[len(evaluated) - 1] = rows[0]["seq"]
        return rows[0]["seq"]

    raw = _doc([
        _intent(reason="read one", args={"path": "plan.txt"}),
        _intent(reason="read two", args={"path": "plan.txt"},
                depends_on=[{"sibling": 0,
                             "requirement": "success"}]),
        _intent(reason="read three", args={"path": "plan.txt"},
                depends_on=[{"sibling": 1}, {"sibling": 0}]),
    ])
    out = run_replay_proposal(journal, raw, task_seq=task["seq"],
                              admit=spy_admit)
    assert out["outcome"] == "ok"
    assert len(evaluated) == 3
    assert len(out["intent_seqs"]) == 3


def test_gate_refuses_unresolved_sibling_entry_closed(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "plan.txt").write_text("plan\n")
    gate = Gate(corpus_policy(str(work)))
    ctx = ToolContext(tool_work_root=str(work), tool_task_seq=1,
                      tool_live_execution=True)
    intent = Intent(kind="tool", tool="fs_read",
                    args={"path": "plan.txt"},
                    reason="an unresolved sibling entry",
                    depends_on=[{"sibling": 0,
                                 "requirement": "outcome"}])
    report = gate.evaluate(intent, None, ctx)
    # §7's backstop: a sibling entry reaching the gate anyway
    # fails §6.2 closed — refusal, never silent admission.
    assert report.verdict == "REJECT"
    safety = next(c for c in report.checks
                  if c.name == "safety_bounds")
    assert not safety.passed
    dep_sub = next(c for c in (safety.subchecks or [])
                   if c.name == "depends_on")
    assert not dep_sub.passed
    assert dep_sub.detail


def test_intent_record_shape_v1_unchanged():
    plain = Intent(kind="tool", tool="fs_read",
                   args={"path": "x"}, reason="plain")
    assert "authored_depends_on" not in plain.to_dict()
    authored = Intent(
        kind="tool", tool="fs_read", args={"path": "x"},
        reason="resolved member",
        depends_on=[{"seq": 7, "requirement": "success"}],
        authored_depends_on=[{"sibling": 0,
                              "requirement": "success"}])
    rec = authored.to_dict()
    assert rec["depends_on"] == [
        {"seq": 7, "requirement": "success"}]
    assert rec["authored_depends_on"] == [
        {"sibling": 0, "requirement": "success"}]


def test_resolve_helper_is_pure_and_verbatim():
    intent = Intent(
        kind="tool", tool="fs_read", args={"path": "x"},
        reason="member",
        depends_on=[{"sibling": 0, "requirement": "success"},
                    {"seq": 9, "requirement": "outcome"}])
    out = resolve_sibling_dependencies(
        intent, resolution_map={0: 42})
    assert out["resolved"] and out["had_sibling"]
    assert out["intent"].depends_on == [
        {"seq": 42, "requirement": "success"},
        {"seq": 9, "requirement": "outcome"}]
    assert out["intent"].authored_depends_on == [
        {"sibling": 0, "requirement": "success"},
        {"seq": 9, "requirement": "outcome"}]
    # The parsed intent itself is untouched (no aliasing).
    assert intent.depends_on[0] == {
        "sibling": 0, "requirement": "success"}
    miss = resolve_sibling_dependencies(
        intent, resolution_map={},
        visited_positions=frozenset({0}))
    assert not miss["resolved"]
    assert miss["failed"][0]["sub_case"] == SUBCASE_ADMISSION_NO_SEQ


# -- the live walk (run_task) ----------------------------------------------

def _complete(reason="The scripted work is done"):
    return Intent(kind="memory", collection="experiences",
                  record={"kind": "task_complete",
                          "detail": "scripted completion report"},
                  reason=reason)


def test_run_task_resolves_sibling_before_gate(mini):
    bundle_dir, work = mini
    batch = [
        Intent(kind="tool", tool="fs_read",
               args={"path": "note.txt"}, reason="read the note"),
        Intent(kind="tool", tool="fs_read",
               args={"path": "notes/plan.md"},
               reason="read the plan after the note",
               depends_on=[{"sibling": 0,
                            "requirement": "success"}]),
    ]
    res = run_task(bundle_dir, [batch, _complete()],
                   "Read both files in order",
                   work_root=work, policy=corpus_policy(work))
    assert res["stop_reason"] == "mind-reported-complete"
    assert [s["verdict"] for s in res["steps"]] == [
        "ADMIT", "ADMIT", "ADMIT"]
    manifest = load_bundle(bundle_dir)
    journal = open_journal(bundle_dir, manifest)
    try:
        rows = journal.backend.find("self_edits")
        tool_rows = [r for r in rows
                     if (r.get("intent") or {}).get("kind") == "tool"]
        assert len(tool_rows) == 2
        first, second = tool_rows
        assert second["intent"]["depends_on"] == [
            {"seq": first["seq"], "requirement": "success"}]
        assert second["intent"]["authored_depends_on"] == [
            {"sibling": 0, "requirement": "success"}]
        assert _refusals(journal) == []
        assert journal.verify_chain()
    finally:
        journal.close()


def test_run_task_refuses_member_whose_position_is_not_a_tool(mini):
    bundle_dir, work = mini
    batch = [
        Intent(kind="memory", collection="experiences",
               record={"kind": "operator_note", "detail": "hi"},
               reason="a non-tool member at position 0"),
        Intent(kind="tool", tool="fs_read",
               args={"path": "note.txt"},
               reason="depends on position 0",
               depends_on=[{"sibling": 0}]),
    ]
    res = run_task(bundle_dir, [batch, _complete()],
                   "Read the note after the note-taker",
                   work_root=work, policy=corpus_policy(work))
    assert res["stop_reason"] == "mind-reported-complete"
    manifest = load_bundle(bundle_dir)
    journal = open_journal(bundle_dir, manifest)
    try:
        refusals = _refusals(journal)
        assert len(refusals) == 1
        rec = refusals[0]["record"]
        assert rec["position"] == 1
        assert rec["cause"] == RESOLUTION_REFUSAL_CAUSE
        assert rec["sub_case"] == SUBCASE_NO_MEMBER_ADMITTED
        # The refused member was never gated: no self-edit
        # carries its tool intent.
        tool_rows = [r for r in journal.backend.find("self_edits")
                     if (r.get("intent") or {}).get("kind")
                     == "tool"]
        assert tool_rows == []
        refused_steps = [s for s in res["steps"]
                         if "refused at resolution"
                         in (s.get("execution_status") or "")]
        assert len(refused_steps) == 1
        assert journal.verify_chain()
    finally:
        journal.close()


# -- the queue fold prices the resolved form (§6.3 / §7.3) ------------------

DENY_WRITE = {"read": "admit", "write": "deny",
              "execute": "admit", "external": "escalate"}


def _open(bundle_dir):
    manifest = load_bundle(bundle_dir)
    return manifest, open_journal(bundle_dir, manifest)


def _state(journal, edit_id, task_seq):
    return fold_queue(journal, task_seq=task_seq).intents[edit_id]


def test_queue_fold_prices_resolved_dependent_like_seq_authored(mini):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture = record_task_fixture(journal, "SI-98 fold", work)
        policy = corpus_policy(
            work, {"tool_class_policy": DENY_WRITE})
        gate = Gate(policy)
        ctx = ToolContext(tool_work_root=work,
                          tool_task_seq=fixture["seq"],
                          tool_live_execution=True)
        batch = record_queue_batch(journal, fixture["seq"],
                                   fixture["ref"], declared_size=5)
        # The sibling: a write the gate REJECTs (deny class) —
        # journaled under its verdict, holding its seq (§6.3).
        sibling = Intent(kind="tool", tool="fs_write",
                         args={"path": "denied.txt",
                               "content": "no\n"},
                         reason="the rejected sibling write",
                         parents=[fixture["ref"]],
                         source_seqs=[fixture["seq"]])
        out_a = admit_tool_intent(
            journal, gate, ctx, sibling, batch_seq=batch["seq"])
        assert out_a["verdict"] == "REJECT"
        a_seq = _state(journal, out_a["edit_id"],
                       fixture["seq"]).intent_seq

        def admit_dependent(authored, requirement):
            intent = Intent(
                kind="tool", tool="fs_read",
                args={"path": "notes/plan.md"},
                reason="a dependent read",
                parents=[fixture["ref"]],
                source_seqs=[fixture["seq"]],
                depends_on=authored)
            ctx.tool_journal_tip = journal.tip()[0]
            ctx.tool_task_intent_seqs = {a_seq}
            return admit_tool_intent(
                journal, gate, ctx, intent, batch_seq=batch["seq"])

        # Resolved-from-sibling forms, built by the walk's own
        # resolution step (map: position 0 -> the sibling's seq).
        def resolved_form(requirement):
            probe = Intent(
                kind="tool", tool="fs_read",
                args={"path": "notes/plan.md"}, reason="probe",
                depends_on=[{"sibling": 0,
                             "requirement": requirement}])
            out = resolve_sibling_dependencies(
                probe, resolution_map={0: a_seq})
            assert out["resolved"]
            return out["intent"].depends_on

        succ_resolved = admit_dependent(
            resolved_form("success"), "success")
        succ_seq = admit_dependent(
            [{"seq": a_seq, "requirement": "success"}], "success")
        out_resolved = admit_dependent(
            resolved_form("outcome"), "outcome")
        out_seq = admit_dependent(
            [{"seq": a_seq, "requirement": "outcome"}], "outcome")
        for out in (succ_resolved, succ_seq, out_resolved, out_seq):
            assert out["verdict"] == "ADMIT", out["gate_report"]

        res = Drainer(journal, bundle_dir, manifest, policy,
                      work).drain_task(fixture["seq"])
        assert res["sync_reason"] == "queue-empty"
        # success on a REJECTED sibling: skipped at drain, in
        # both authored forms alike; the skip is journaled.
        for out in (succ_resolved, succ_seq):
            state = _state(journal, out["edit_id"], fixture["seq"])
            assert state.state == "skipped", state
            assert state.skip_dependency_seq == a_seq
        skips = journal.backend.find(
            "experiences", filter={"kind": "queue_skip"})
        skip_edit_ids = {r["record"]["edit_id"] for r in skips}
        assert {succ_resolved["edit_id"],
                succ_seq["edit_id"]} <= skip_edit_ids
        # outcome on a REJECTED sibling: any recorded
        # conclusion satisfies it — executed, both forms alike.
        for out in (out_resolved, out_seq):
            state = _state(journal, out["edit_id"], fixture["seq"])
            assert state.state == "executed", state
            assert state.outcome_status == "completed"
        assert journal.verify_chain()
    finally:
        journal.close()
