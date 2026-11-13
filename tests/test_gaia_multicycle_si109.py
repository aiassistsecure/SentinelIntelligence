"""SI-109 (SI-106 §12 item 1; SI-108 §4.1): the multi-cycle
capture driver, in-repo.

OFFLINE / FIXTURE ONLY. Every run below drives the driver with
a scripted (replay-shaped) proposal source on a fixture bundle;
no live model is contacted, no live capture is made, and
nothing here claims one. The acceptance tests are SI-108
§4.1's, numbered to match:

1. fixture K=2 accounting (cycle 0 unscored; cycles 1..2 only
   with non-empty prior_cycles; duplicates logged, not
   entered; transport failures unscored, in no denominator);
2. journaling-without-execution asserted (§9.5-faithful form);
3. residency-guard halt mid-run spends no attempt, persists
   state, never restarted around;
4. budget exhaustion ends the run; no path extends a budget;
5. freeze hashes reproducible by independent recomputation.

(§4.1 item 6 is this slice's review rule — a diff touching
gate pricing, parser stage rules, or teaching text fails
review — not a unit test; the slice diff itself is the
evidence.)
"""
import hashlib
import json

import pytest

from si.bundle import canonical_json, sha256_hex
from si.gaia import live as live_mod
from si.gaia.multicycle import (
    MultiCycleDriver,
    ResidencyGuardHalt,
    SmokeGateError,
    build_multicycle_corpus,
    collect_no_execution_proof,
    corpus_canonical_sha256,
    fold_rows_and_hash,
    freeze_multicycle,
    run_fixture_smoke_gate,
    scored_documents,
    verify_freeze,
)
from si.gate import Gate
from si.journal import open_journal


def _doc(intents, signal=None):
    return json.dumps({"proposal": {
        "version": 1, "intents": intents, "task_signal": signal}})


def _intent(tool="fs_list", args=None, reason="list the work root",
            depends_on=None):
    return {"tool": tool,
            "args": args if args is not None else {"path": "."},
            "reason": reason,
            "depends_on": depends_on or [],
            "ordering_key": None}


class _Guard:
    """§9.2 guard fake: states in order, then the last repeats."""

    def __init__(self, states):
        self.states = list(states)
        self.polls = 0

    def __call__(self, journal):
        self.polls += 1
        if len(self.states) > 1:
            return self.states.pop(0)
        return self.states[0]


class _Source:
    """Scripted proposal source: delegates to fn(perception,
    call_number) so a test can vary bodies by cycle / fold."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, perception):
        self.calls += 1
        return self.fn(perception, self.calls)


def _citing_source(perception, _n):
    """A distinct parse-ok body per call; at cycles >= 1 it
    cites the first citable seq of its own perception."""
    sections = perception.get("sections") or {}
    prior = sections.get("prior_cycles") or []
    citable = (sections.get("citable_reference_set") or {}).get(
        "seqs") or []
    deps = [citable[0]] if citable else []
    return _doc([_intent(
        reason=f"cycle with {len(prior)} prior intent(s)",
        depends_on=deps)])


@pytest.fixture()
def fixture(tmp_path):
    paths = live_mod.build_fixture_bundle(
        str(tmp_path / "capture-bundle"), str(tmp_path / "work"))
    policy = live_mod.fixture_policy(paths["work_root"])
    journal = open_journal(paths["bundle_dir"])
    yield paths, policy, journal
    journal.close()


def _driver(journal, policy, paths, source, guard=None, **kw):
    from si.gaia.policy import resolve_tool_policy
    tool_pol = resolve_tool_policy(policy)
    budgets = {
        "max_tool_cycles_per_task": int(
            tool_pol["max_tool_cycles_per_task"]),
        "tool_max_task_result_bytes": int(
            tool_pol["tool_max_task_result_bytes"]),
    }
    return MultiCycleDriver(
        journal, Gate(policy), bundle_dir=paths["bundle_dir"],
        work_root=paths["work_root"], budgets=budgets,
        cycle_cap=budgets["max_tool_cycles_per_task"],
        proposal_source=source,
        residency_guard=guard or _Guard(["Resident"]), **kw)


_TASK = {"workload": "benign-read", "run": 0,
         "instruction": "list the work root"}


# -- §4.1(1) fixture K=2 accounting ---------------------------------

def test_fixture_k2_cycle_accounting(fixture):
    paths, policy, journal = fixture
    source = _Source(_citing_source)
    driver = _driver(journal, policy, paths, source, K=2)
    state = driver.run([_TASK])
    docs = state["documents"]
    assert [d["cycle"] for d in docs] == [0, 1, 2]
    assert all(d["journaled"] for d in docs)
    # Cycle 0 present and unscored; cycles 1..2 scored, each
    # entered only with a non-empty prior fold.
    scored = scored_documents(docs)
    assert [d["cycle"] for d in scored] == [1, 2]
    assert all(d["perception"]["prior_cycles_count"] > 0
               for d in scored)
    assert docs[0]["perception"]["prior_cycles_count"] == 0
    corpus = build_multicycle_corpus(
        docs, state["entries"], corpus_id="si109-test-k2",
        candidate_label="fixture-candidate",
        capture_date="2026-11-13")
    assert corpus["fixed_n"] == 2  # M: scored documents only
    assert [i["cycle"] for i in corpus["items"]] == [1, 2]
    assert all(i["prompt_ref"]["task_seq"] == docs[0]["task_seq"]
               for i in corpus["items"])
    assert state["runs"]["benign-read#0"]["ended"] == "completed"


def test_duplicates_logged_not_entered(fixture):
    paths, policy, journal = fixture
    first = _doc([_intent(reason="the one entered body")])

    def fn(perception, n):
        if n <= 2:
            return first  # cycle 0 enters it; cycle 1 attempt 1 dups
        return _citing_source(perception, n)

    source = _Source(fn)
    driver = _driver(journal, policy, paths, source, K=1)
    state = driver.run([_TASK])
    dups = [e for e in state["entries"]
            if e["outcome"] == "duplicate-capture"]
    assert len(dups) == 1 and dups[0]["cycle"] == 1
    assert dups[0]["duplicate_of"] == state["documents"][0]["doc_id"]
    bodies = [d["body"] for d in state["documents"]]
    assert bodies.count(first) == 1  # stored once, never duplicated
    assert [d["cycle"] for d in state["documents"]] == [0, 1]


def test_transport_failures_unscored_in_no_denominator(fixture):
    paths, policy, journal = fixture

    def fn(perception, n):
        sections = perception.get("sections") or {}
        if (sections.get("prior_cycles") or []):
            raise ConnectionError("gateway died mid-request")
        return _citing_source(perception, n)

    source = _Source(fn)
    driver = _driver(journal, policy, paths, source, K=1,
                     max_attempts_per_workload_cycle=2)
    state = driver.run([_TASK])
    # Cycle 0 entered; both cycle-1 attempts transport-failed
    # and the run ended on the cycle-1 budget.
    assert [d["cycle"] for d in state["documents"]] == [0]
    run = state["runs"]["benign-read#0"]
    assert "attempt budget exhausted at cycle 1" in run["ended"]
    failures = [e for e in state["entries"]
                if e["outcome"] == "transport-failed"]
    assert len(failures) == 2
    corpus = build_multicycle_corpus(
        state["documents"], state["entries"],
        corpus_id="si109-test-tf", candidate_label="fixture",
        capture_date="2026-11-13")
    # M = 0: the failures are carried as unscored placeholders
    # at a scored position and enter no denominator.
    assert corpus["fixed_n"] == 0
    placeholders = [i for i in corpus["items"]
                    if i.get("transport_failed")]
    assert len(placeholders) == 2
    assert all(i["body"] is None for i in placeholders)


# -- §4.1(2) journaling without execution ---------------------------

def test_journaling_without_execution_asserted(fixture, tmp_path):
    paths, policy, journal = fixture
    write_body = _doc([
        _intent(reason="admitted read for the fold"),
        _intent(tool="fs_write",
                args={"path": "should-not-exist.txt",
                      "content": "never written"},
                reason="a write the fixture policy denies"),
    ])

    def fn(perception, n):
        sections = perception.get("sections") or {}
        if not (sections.get("prior_cycles") or []):
            return write_body
        return _citing_source(perception, n)

    source = _Source(fn)
    driver = _driver(journal, policy, paths, source, K=1)
    driver.run([_TASK])
    proof = collect_no_execution_proof(journal)
    assert proof["queue_claim_rows"] == 0
    assert proof["executed_outcome_rows"] == []
    assert proof["no_execution_proven"] is True
    # Non-vacuous: the denied write DID journal a tool_result
    # row, and it is a non-execution disposition (REJECT, no
    # measured fields) — the §9.5-faithful form.
    assert proof["tool_result_rows"] >= 1
    assert all(d["non_execution"]
               for d in proof["tool_result_dispositions"])
    verdicts = {d["gate_verdict"]
                for d in proof["tool_result_dispositions"]}
    assert "REJECT" in verdicts  # the denied write's disposition
    assert verdicts <= {"REJECT", "ESCALATE"}
    import os
    assert not os.path.exists(
        os.path.join(paths["work_root"], "should-not-exist.txt"))


# -- §4.1(3) residency-guard halt ------------------------------------

def test_guard_halt_mid_run_spends_no_attempt(fixture):
    paths, policy, journal = fixture
    guard = _Guard(["Resident", "Resident", "Lost"])
    source = _Source(_citing_source)
    snapshots = []
    driver = _driver(journal, policy, paths, source, K=2,
                     guard=guard,
                     on_progress=snapshots.append)
    with pytest.raises(ResidencyGuardHalt, match="no attempt spent"):
        driver.run([_TASK])
    # Two attempts spent (cycles 0 and 1); the halted cycle-2
    # poll spent none, and the source was never called again —
    # the driver does not restart around a halt.
    assert source.calls == 2
    assert driver.state["attempts"] == {
        "benign-read@0": 1, "benign-read@1": 1}
    assert driver.state["halted"]["cycle"] == 2
    assert driver.state["halted"]["residency_state"] == "Lost"
    # State persisted through on_progress at the halt.
    assert snapshots and snapshots[-1]["halted"] is not None
    assert snapshots[-1]["attempts"] == driver.state["attempts"]
    assert driver.state["runs"]["benign-read#0"]["ended"] is None


def test_guard_poll_failure_also_halts_without_spend(fixture):
    paths, policy, journal = fixture

    def bad_guard(_journal):
        raise ConnectionError("residency endpoint unreachable")

    source = _Source(_citing_source)
    driver = _driver(journal, policy, paths, source, K=2,
                     guard=bad_guard)
    with pytest.raises(ResidencyGuardHalt, match="no attempt spent"):
        driver.run([_TASK])
    assert source.calls == 0
    assert driver.state["attempts"] == {}


# -- §4.1(4) budget exhaustion, never extended ------------------------

def test_budget_exhaustion_ends_run_and_never_extends(fixture):
    paths, policy, journal = fixture
    same = _doc([_intent(reason="always the same body")])
    source = _Source(lambda _p, _n: same)
    driver = _driver(journal, policy, paths, source, K=2,
                     max_attempts_per_workload_cycle=2)
    state = driver.run([_TASK])
    run = state["runs"]["benign-read#0"]
    # Cycle 0 entered on attempt 1; cycle 1 burned its whole
    # budget on duplicates and the run ended there — cycle 2
    # was never attempted.
    assert "attempt budget exhausted at cycle 1" in run["ended"]
    assert state["attempts"] == {
        "benign-read@0": 1, "benign-read@1": 2}
    calls_after_first_run = source.calls
    # A second run() over the same plan spends nothing more:
    # the ended run is skipped and no budget is refunded or
    # extended by any path.
    driver.run([_TASK])
    assert source.calls == calls_after_first_run
    assert driver.state["attempts"]["benign-read@1"] == 2
    assert driver.max_attempts_per_workload_cycle == 2


# -- §4.1(5) freeze hashes reproduce independently ---------------------

def test_freeze_hashes_reproduce_independently(fixture, tmp_path):
    paths, policy, journal = fixture
    source = _Source(_citing_source)
    driver = _driver(journal, policy, paths, source, K=2)
    state = driver.run([_TASK])
    corpus = build_multicycle_corpus(
        state["documents"], state["entries"],
        corpus_id="si109-test-freeze", candidate_label="fixture",
        capture_date="2026-11-13")
    out = str(tmp_path / "freeze")
    record = freeze_multicycle(paths["bundle_dir"], corpus, out)
    # Corpus file hash: raw bytes, recomputed with hashlib alone.
    with open(record["corpus_file"], "rb") as fh:
        raw = fh.read()
    assert hashlib.sha256(raw).hexdigest() == \
        record["corpus_file_sha256"]
    # Corpus canonical hash: recomputed from the value alone.
    assert "sha256:" + sha256_hex(canonical_json(corpus)) == \
        record["corpus_sha256"]
    assert corpus_canonical_sha256(corpus) == record["corpus_sha256"]
    # Fold hash: recomputed from the FROZEN bundle alone.
    frozen = open_journal(record["fold_bundle"])
    try:
        _rows, digest, _counts, tip = fold_rows_and_hash(frozen)
    finally:
        frozen.close()
    assert digest == record["fold_sha256"]
    assert tip == record["fold_tip_seq"]
    assert record["fold_rows_total"] > 0
    assert verify_freeze(record, corpus, record["fold_bundle"])
    tampered = dict(corpus, candidate_label="someone-else")
    assert not verify_freeze(record, tampered,
                             record["fold_bundle"])
    with pytest.raises(RuntimeError, match="one-time act"):
        freeze_multicycle(paths["bundle_dir"], corpus, out)


# -- §9.4 smoke gate (fixture scope) -----------------------------------

def test_fixture_smoke_gate_passes_and_proves_no_execution(tmp_path):
    paths = live_mod.build_fixture_bundle(
        str(tmp_path / "smoke-bundle"), str(tmp_path / "work"))
    bodies = [
        _doc([_intent(reason="smoke cycle 0 read")]),
        _doc([_intent(reason="smoke cycle 1 read")]),
    ]
    summary = run_fixture_smoke_gate(
        paths["bundle_dir"], paths["work_root"], bodies=bodies,
        residency_guard=_Guard(["Resident"]))
    assert summary["passed"] is True
    assert all(summary["checks"].values())
    assert summary["fold"]["rows"] > 0
    assert summary["no_execution_proof"]["queue_claim_rows"] == 0
    assert summary["no_execution_proof"]["no_execution_proven"]


def test_fixture_smoke_gate_fails_when_fold_stays_empty(tmp_path):
    paths = live_mod.build_fixture_bundle(
        str(tmp_path / "smoke-bundle"), str(tmp_path / "work"))
    bodies = ["plain text, not a proposal document\n",
              _doc([_intent(reason="too late to matter")])]
    with pytest.raises(SmokeGateError, match="smoke gate failed"):
        run_fixture_smoke_gate(
            paths["bundle_dir"], paths["work_root"], bodies=bodies,
            residency_guard=_Guard(["Resident"]))
