"""Gaia B-3 task loop: operator tasks, chaining, budgets, and the
§10.4 stop rules (SI-41 RFC §10, §15 B-3; slice SI-44).

B-1 journaled *fixture* task entries because the real task surface
was B-3. This module is that surface:

- :func:`record_task_entry` journals a REAL operator task entry —
  the RFC §10.1 shape (kind ``task``, in the existing
  ``experiences`` collection, ``source: "operator"``, the
  instruction text, the work root, and the budgets in force) via
  the journal's existing ``record_experience``. Unlike B-1's
  :func:`si.gaia.shadow.record_task_fixture` it is **not** labelled
  a fixture: an operator (via ``si task``, or a caller of
  :func:`run_task`) actually opened this task.
- :func:`run_task` runs the serial agency loop of §10.2 against a
  bundle on disk: the task is journaled BEFORE any cycle runs;
  each cycle sets the gate context's task bookkeeping
  (``tool_task_seq`` / ``tool_cycles_used`` /
  ``tool_result_bytes_used``) and chains one intent through
  :func:`si.gaia.executor.execute_tool_cycle` (gate → journal →
  execute → outcome journaled); budgets are drawn from the
  bundle policy's §6.2 keys (``max_tool_cycles_per_task``,
  ``tool_max_task_result_bytes``), with the operator's ``cycles``
  argument capping total steps at the minimum of the two.

The proposer is the mind-shaped slot of §10.2, with two honest
forms (RFC §14.1 — the miniature's evidence vehicle is scripted):

- a **mind-like object** with ``perceive(journal)`` /
  ``propose(perceptions, ctx)``: PERCEIVE is the task entry plus
  the tool_result observations so far (provenance-tagged per
  §4.5 — observations are data, never instructions), plus
  whatever the mind's own ``perceive`` returns from the journal;
- an **iterable of Intent objects** (the scripted proposer):
  intents are consumed in order; a scripted intent whose
  ``parents`` / ``source_seqs`` are empty has the task chain
  filled by the runner (``parents = [task_ref] + [prior edit]``,
  ``source_seqs = [task_seq] + [prior outcome seq]``) — the
  chaining §10.2 prescribes, applied mechanically so a script
  need not predict journal seqs. Non-empty chains are left
  exactly as authored and priced by the gate as authored.

Stop rules (§10.4) — each stop is journaled as a ``task_stop``
experience with its cause before the task returns:

1. ``budget-exhausted`` — the cycle or result budget is reached
   (checked before proposing, and also mapped from a gate REJECT
   whose budgets sub-check failed);
2. ``escalated`` — any cycle's verdict is ESCALATE: the task
   halts *at* the escalation;
3. ``reject-streak`` — two consecutive REJECTs (a non-REJECT
   resets the streak);
4. ``integrity-failure`` — an executor refusal whose detail is a
   journaled-args re-validation failure or a class-changing
   TOCTOU (§9.2): halt immediately, no streak arithmetic;
5. ``mind-reported-complete`` — a memory intent whose record is
   ``{"kind": "task_complete", ...}``, gated as an ordinary
   memory intent (gate + ``record_self_edit`` + the memory
   experience, mirroring si.loop's memory handling); ADMIT
   closes the task with this label — the *mind reported*
   completion; the operator judges whether it is true (§10.4.5).

There is no rule 6 (§10.4). A scripted proposer running out of
intents, or a mind proposer returning None, stops the task with
``stop_reason == "proposer-exhausted"`` — explicitly **not** one
of §10.4's five rules: an operator-visible incomplete task,
never dressed as a §10.4 stop.

Contexts: miniature bundles gate through the live
``ProbeContext``; demo (non-miniature) bundles gate through a
non-live ``ToolContext`` — their intents therefore ESCALATE and
the task halts ``escalated`` with the B-1 shadow label, honestly:
no demo-bundle execution exists in B-3 either.

Portability (§12.5(d)): a task writes only through the existing
journal methods in the existing collections, so a moved bundle
carries the whole trail — ``journal_query`` and the provenance
walk work on the far side exactly as on the near side (proven by
the B-3 test that copies a bundle after a task and queries the
copy). Nothing in this module needs the bundle's original path.

B-5 (SI-46) extension, additive only: :func:`run_task` and
:func:`record_task_entry` accept an optional ``submitter`` /
``submitted_via`` pair. When a task arrives through the B-5
service adapter (:mod:`si.gaia.service`), the submitter identity
is journaled on the task entry as a mandatory provenance field
(RFC §12.5(a) — no identity, no task), labelled
:data:`SUBMITTER_IDENTITY_LABEL`: asserted, never authenticated.
Operator-opened tasks pass neither field and are unchanged.

SI-51 (SI-50 §16 Q-1, the queue-drain core) re-shapes the cycle
this module runs, without changing its contract above: a
proposal is now a BATCH (§6 — a list of intents; a lone intent is
the degenerate batch of one, so every B-3 behaviour above is
preserved for singleton proposals). Tool members are admitted —
gated and journaled, per-member verdicts — into the task's
queue, and a serial drainer (si/gaia/drain.py, concurrency fixed
at 1 by policy in Q-1) claims and executes them under
claim-before-execute (§5.3) before ONE batched perception
returns to the proposer (§10). The stop rules are §12.4's: budget
and reject-streak refusals halt proposing, not draining; an
escalation halts proposing and the queued members drain; an
integrity failure freezes the drain. Q-1's stated limits:
depends_on is parsed, priced, and journaled, and its
satisfaction is treated as sequence order (full barrier /
requirement semantics are Q-2); concurrency is 1 (Q-2);
write/execute outcome-unknown recovery is Q-3; all queue
performance is UNMEASURED (Q-4). See docs/SI-51-q1-queue-core.md.
"""
from __future__ import annotations

import os

from si.gaia.drain import Drainer, admit_tool_intent
from si.gaia.executor import execute_tool_cycle
from si.gaia.queue import fold_queue, record_queue_batch
from si.gaia.policy import resolve_tool_policy
from si.gate import Gate, Intent

#: §10.4's five stop causes (journaled verbatim as the task_stop
#: record's ``cause`` and returned as ``stop_reason``).
STOP_BUDGET_EXHAUSTED = "budget-exhausted"
STOP_ESCALATED = "escalated"
STOP_REJECT_STREAK = "reject-streak"
STOP_INTEGRITY_FAILURE = "integrity-failure"
STOP_MIND_REPORTED_COMPLETE = "mind-reported-complete"

#: NOT a §10.4 rule (the RFC has no rule 6): the proposer ran out
#: of intents / returned None. An operator-visible incomplete
#: task, labelled as itself.
STOP_PROPOSER_EXHAUSTED = "proposer-exhausted"
#: NOT one of §10.4's five rules either: an operator queue stop
#: (SI-50 §11, the leash) held the drain, so the driver returned
#: control with the task incomplete and its queued intents
#: journaled for a later resume — labelled as itself (SI-51).
STOP_OPERATOR_STOP = "operator-stop"

#: Detail markers of an execution-integrity refusal (§9.2): the
#: journaled args failed re-validation at EXECUTE time, or the
#: intent's effective class changed between GATE and EXECUTE.
_INTEGRITY_MARKERS = (
    "TOCTOU",
    "changed between GATE and EXECUTE",
    "no longer validate against the registry schema",
)


#: Label carried on every service-submitted task entry (SI-46/B-5):
#: the submitter identity is a journaled provenance field — who the
#: submission CLAIMS to be from — and is not authentication. No
#: credential is checked by B-3's machinery or by B-5's adapter;
#: the precedent is ``si admit``'s --operator (an attribution /
#: authorization record in the journal, NOT cryptographic
#: authentication). Single source so the journal record, the
#: service response, and the provenance trail can never drift.
SUBMITTER_IDENTITY_LABEL = (
    "submitter identity is an asserted provenance field, journaled "
    "verbatim — it is NOT authentication: no credential was "
    "checked, and the identity is not verified (SI-41 §12.5(a); "
    "the si admit --operator precedent)")


def record_task_entry(journal, instruction: str, work_root: str,
                      budgets: dict | None = None,
                      operator: str = "operator",
                      submitter: str | None = None,
                      submitted_via: str | None = None) -> dict:
    """Journal a real operator task entry (RFC §10.1).

    Same shape as B-1's scripted fixture — kind ``task`` in
    ``experiences``, ``source: "operator"``, instruction, work
    root, budgets — but NOT labelled a fixture: this is the
    entry a real task opens with. Returns ``{"seq", "ref",
    "row_id"}`` where ``ref`` is ``task@<seq>``, the parent ref
    every step in the task chains to (gate check 5, §5.2).

    B-5 (SI-46) adds two optional provenance fields, present only
    when a service submitted the task: ``submitter`` (mandatory
    on the B-5 surface — no identity, no task) and
    ``submitted_via``, plus :data:`SUBMITTER_IDENTITY_LABEL`
    stating what the identity is and is not. Operator-opened
    tasks (``si task``) pass neither and their entries are
    unchanged.
    """
    record = {
        "kind": "task",
        "source": "operator",
        "operator": operator,
        "instruction": instruction,
        "work_root": work_root,
        "budgets": dict(budgets or {}),
    }
    if submitter is not None:
        record["submitter"] = submitter
        record["submitted_via"] = submitted_via
        record["submitter_identity_label"] = SUBMITTER_IDENTITY_LABEL
    row_id = journal.record_experience("task", record, "")
    rows = journal.backend.find("experiences", filter={"kind": "task"},
                                sort="-seq", limit=1)
    seq = rows[0]["seq"] if rows else journal.tip()[0]
    return {"seq": seq, "ref": f"task@{seq}", "row_id": row_id}


def record_task_stop(journal, task: dict, cause: str, detail: str,
                     cycles_used: int, result_bytes_used: int) -> dict:
    """Journal a task's stop as a ``task_stop`` experience (§10.4:
    a task "halts, journals its stop with cause, and returns
    control to the operator"). Returns {"seq", "row_id"}."""
    record = {
        "kind": "task_stop",
        "source": "operator",
        "task_seq": task["seq"],
        "task_ref": task["ref"],
        "cause": cause,
        "stop_reason": cause,
        "detail": detail,
        "cycles_used": cycles_used,
        "result_bytes_used": result_bytes_used,
    }
    row_id = journal.record_experience("task_stop", record, "")
    rows = journal.backend.find(
        "experiences", filter={"kind": "task_stop"}, sort="-seq",
        limit=1)
    seq = rows[0]["seq"] if rows else journal.tip()[0]
    return {"seq": seq, "row_id": row_id}


def _is_budget_reject(gate_report: dict) -> bool:
    """True iff a REJECT's safety check failed on its budgets
    sub-check (§10.4.1: budget exhaustion surfaces as a check-2
    REJECT, and the task maps it to budget-exhausted).

    The detail must say so: for an unknown tool the gate journals
    every later sub-check (budgets included) as a failed
    "not evaluated" placeholder, and that placeholder is not a
    budget exhaustion — the priced budget failures are the ones
    whose detail names the exhaustion (or a malformed spend,
    which fails closed the same way)."""
    for check in gate_report.get("checks") or []:
        for sub in check.get("subchecks") or []:
            if sub.get("name") == "budgets" and not sub.get("passed"):
                detail = str(sub.get("detail") or "")
                if "exhausted" in detail or "malformed" in detail:
                    return True
    return False


def _is_integrity_refusal(execution: dict) -> bool:
    detail = str(execution.get("detail") or "")
    return any(marker in detail for marker in _INTEGRITY_MARKERS)


def _outcome_seq(journal, edit_id: str):
    """The journal seq of an edit's tool_result outcome row."""
    rows = journal.backend.find(
        "experiences", filter={"self_edit_id": edit_id}, sort="-seq",
        limit=1)
    return rows[0]["seq"] if rows else None


def _run_completion(journal, gate, ctx, intent: Intent) -> dict:
    """Gate a ``task_complete`` memory intent as an ordinary
    memory intent and, on ADMIT, record the memory experience —
    mirroring si.loop's memory handling (record the experience,
    mark the self-edit materialized). Returns the outcome dict."""
    if not intent.collection:
        # A task_complete report is an experience by construction;
        # the gate's memory check prices collection == "experiences"
        # (the only writable collection), so the runner fills the
        # default an authored mind would set (the loop's minds set
        # it explicitly) rather than REJECTing a completion report
        # for an undeclared default.
        intent.collection = "experiences"
    report = gate.evaluate(intent, None, ctx)
    edit_id = journal.record_self_edit(
        intent.kind, intent.to_dict(), intent.reason,
        report.verdict, report.to_dict())
    if intent.parents:
        journal.record_provenance(
            child=edit_id, parents=list(intent.parents),
            relation="learned_from", note=(intent.reason or "")[:200])
    if intent.source_seqs:
        journal.record_provenance(
            child=edit_id, parents=list(intent.source_seqs),
            relation="learned_from", note="source journal sequences")
    if report.verdict == "ADMIT":
        rec = intent.record or {}
        journal.record_experience(
            rec.get("kind", "observation"), rec, edit_id)
        journal.set_edit_result(edit_id, "materialized")
        status = "materialized"
    else:
        journal.set_edit_result(
            edit_id,
            "rejected" if report.verdict == "REJECT" else "escalated")
        status = "not-executed"
    return {"verdict": report.verdict, "edit_id": edit_id,
            "gate_report": report.to_dict(),
            "execution_status": status}


def run_task(bundle_dir: str, proposer, instruction: str,
             work_root: str | None = None, cycles: int | None = None,
             operator: str = "operator",
             policy: dict | None = None,
             submitter: str | None = None,
             submitted_via: str | None = None) -> dict:
    """Run one operator task against a bundle on disk (§10).

    Opens (journals) the task before any proposal, then runs the
    SI-51 queue flow (SI-50 §16 Q-1) per proposal: the proposer
    returns ONE intent (the B-3 degenerate batch) or a BATCH — a
    list of intents (§6). Each batch is journaled as a
    ``queue_batch`` entry, then its tool members are gated and
    journaled INDIVIDUALLY in proposal order via
    :func:`si.gaia.drain.admit_tool_intent` (per-member verdicts,
    no member's verdict references another's, §6.1); ADMITted
    members are QUEUED, not executed. The drainer
    (:class:`si.gaia.drain.Drainer`) then drains the task's queue
    serially — claim journaled before execution, §7.1 order,
    ordering keys enforced — and the round returns ONE batched
    perception (an observation set with the §10.3 queue summary,
    sync points §10.2 1/3/4) before the next proposal. A
    ``task_complete`` memory member is processed after its batch
    drains (§12.4.5: completion closes the task only once the
    queue has drained, or the operator has stopped it).

    Stop rules are §12.4's re-expressions of §10.4: a budget
    REJECT or two consecutive REJECTs halt PROPOSING but the
    queued members still drain; an ESCALATE halts proposing (the
    current batch's remaining members are not evaluated) and the
    queued members drain; an execution integrity failure freezes
    the drain and halts the task; budgets are debited per intent
    EVALUATED (§12.3). An operator queue stop in force (§11)
    returns control with stop_reason ``operator-stop`` — NOT one
    of §10.4's five rules, labelled as itself like
    ``proposer-exhausted``. Re-anchors the manifest checkpoint and
    rewrites the manifest before returning.

    Returns {"task_seq", "task_ref", "stop_reason", "stop_detail",
    "cycles_used", "result_bytes_used", "steps", "head",
    "perceptions", "batch_seqs"} — the B-3 keys unchanged, plus
    the batched perception sets and the batch entry seqs. Each
    step is {"edit_id", "verdict", "tool", "effective_class",
    "execution_status"}.
    """
    from si.bundle import (
        _anchor_checkpoint,
        _write_manifest,
        load_bundle,
        load_policy,
    )
    from si.journal import open_journal
    from si.probes import is_miniature_probe_bundle

    bundle_dir = os.path.abspath(bundle_dir)
    manifest = load_bundle(bundle_dir)
    pol = policy if policy is not None else load_policy(bundle_dir)
    tool_pol = resolve_tool_policy(pol)
    gate = Gate(pol)
    journal = open_journal(bundle_dir, manifest)
    try:
        if work_root:
            root = os.path.realpath(work_root)
        elif tool_pol.get("tool_work_root"):
            root = os.path.realpath(tool_pol["tool_work_root"])
        else:
            root = bundle_dir
        cycle_cap = int(tool_pol["max_tool_cycles_per_task"])
        if cycles is not None:
            cycle_cap = min(cycle_cap, int(cycles))
        byte_cap = int(tool_pol["tool_max_task_result_bytes"])
        budgets = {
            "max_tool_cycles_per_task": cycle_cap,
            "tool_max_task_result_bytes": byte_cap,
        }
        if cycles is not None:
            budgets["operator_cycles_cap"] = int(cycles)

        miniature = is_miniature_probe_bundle(manifest)
        if miniature:
            from si.probes import ProbeContext
            ctx = ProbeContext(bundle_dir, manifest, policy=pol)
        else:
            from si.gaia.context import ToolContext
            ctx = ToolContext(bundle_dir=bundle_dir)
        ctx.tool_work_root = root

        # The task is journaled BEFORE any cycle runs (§10.1).
        task = record_task_entry(journal, instruction, root,
                                 budgets=budgets, operator=operator,
                                 submitter=submitter,
                                 submitted_via=submitted_via)
        ctx.tool_task_seq = task["seq"]

        is_mind = hasattr(proposer, "perceive") \
            and hasattr(proposer, "propose")
        scripted = None if is_mind else iter(list(proposer))
        drainer = Drainer(journal, bundle_dir, manifest, pol, root)

        task_observation = {
            "kind": "task", "source": "operator", "seq": task["seq"],
            "ref": task["ref"], "instruction": instruction,
            "work_root": root, "budgets": dict(budgets),
        }
        observations: list = []
        perception_sets: list = []
        batch_seqs: list = []
        steps: list = []
        cycles_used = 0
        bytes_used = 0
        reject_streak = 0
        stop_reason = None
        stop_detail = ""
        last_edit_id = None
        last_outcome_seq = None
        task_intent_seqs: set = set()

        def _note_outcome(edit_id):
            """Track a step's journaled outcome for chaining and
            for the next PERCEIVE (the §4.5 observation is the
            journaled tool_result record, provenance tags intact,
            as data)."""
            nonlocal last_outcome_seq
            seq = _outcome_seq(journal, edit_id)
            if seq is not None:
                last_outcome_seq = seq
                rows = journal.backend.find(
                    "experiences",
                    filter={"self_edit_id": edit_id},
                    sort="-seq", limit=1)
                if rows:
                    observations.append(
                        dict(rows[0].get("record") or {},
                             seq=rows[0]["seq"]))

        def _chain(intent):
            """§10.2 chaining, filled mechanically for scripted
            intents that did not author their own chain."""
            if not intent.parents:
                intent.parents = [task["ref"]] + (
                    [last_edit_id] if last_edit_id else [])
            if not intent.source_seqs:
                intent.source_seqs = [task["seq"]] + (
                    [last_outcome_seq]
                    if last_outcome_seq is not None else [])

        while stop_reason is None:
            # §12.4.1, checked before proposing: either budget
            # reached means the next intent would fail check 2 —
            # the task halts as budget-exhausted, a priced outcome.
            if cycles_used >= cycle_cap:
                stop_reason = STOP_BUDGET_EXHAUSTED
                stop_detail = (
                    f"cycle budget exhausted: {cycles_used} tool "
                    f"cycle(s) used of max_tool_cycles_per_task "
                    f"{cycle_cap} (RFC §10.3; SI-50 §12.3 debits "
                    "per intent evaluated)")
                break
            if bytes_used >= byte_cap:
                stop_reason = STOP_BUDGET_EXHAUSTED
                stop_detail = (
                    f"result budget exhausted: {bytes_used} inline "
                    f"result byte(s) used of "
                    f"tool_max_task_result_bytes {byte_cap} "
                    "(RFC §10.3)")
                break

            if is_mind:
                perceptions = [task_observation]
                extra = proposer.perceive(journal)
                if isinstance(extra, list):
                    perceptions += extra
                perceptions += observations
                proposal = proposer.propose(perceptions, ctx)
            else:
                try:
                    proposal = next(scripted)
                except StopIteration:
                    proposal = None
            if proposal is None:
                stop_reason = STOP_PROPOSER_EXHAUSTED
                stop_detail = (
                    "the proposer produced no further intent: the "
                    "task is incomplete (NOT one of RFC §10.4's "
                    "five stop rules — there is no rule 6)")
                break
            if isinstance(proposal, Intent):
                batch = [proposal]
            elif isinstance(proposal, (list, tuple)):
                batch = list(proposal)
            else:
                raise TypeError(
                    "a task proposer must yield si.gate.Intent "
                    "objects or a batch (list) of them, got "
                    f"{type(proposal).__name__}")
            if not batch:
                stop_reason = STOP_PROPOSER_EXHAUSTED
                stop_detail = (
                    "the proposer produced an empty batch: the "
                    "task is incomplete (NOT one of RFC §10.4's "
                    "five stop rules — there is no rule 6)")
                break
            for member in batch:
                if not isinstance(member, Intent):
                    raise TypeError(
                        "a batch member must be a si.gate.Intent, "
                        f"got {type(member).__name__}")

            completions = [m for m in batch
                           if m.kind == "memory"
                           and (m.record or {}).get("kind")
                           == "task_complete"]
            completion_ids = {id(m) for m in completions}
            tool_members = [m for m in batch if m.kind == "tool"]
            other_members = [m for m in batch
                             if m.kind != "tool"
                             and id(m) not in completion_ids]

            # Unsupported kinds: a proposer contract violation,
            # counted as a REJECT step for streak purposes —
            # journaled in the steps list and the stop detail,
            # never executed and never given a fabricated gate
            # report (B-3 behaviour, unchanged).
            for _m in other_members:
                steps.append({
                    "edit_id": None, "verdict": "REJECT",
                    "tool": None, "effective_class": None,
                    "execution_status":
                        "not-executed — the task loop drives tool "
                        "intents (and task_complete reports) only",
                })
                reject_streak += 1
                if reject_streak >= 2:
                    stop_reason = STOP_REJECT_STREAK
                    stop_detail = (
                        "two consecutive REJECTs (unsupported "
                        "intent kinds proposed to the task loop): "
                        "the task halts (RFC §10.4.3)")
                    break
            if stop_reason is not None:
                break
            if other_members and not tool_members and not completions:
                continue

            if tool_members:
                if scripted is not None:
                    for member in tool_members:
                        _chain(member)
                batch_entry = record_queue_batch(
                    journal, task["seq"], task["ref"],
                    len(tool_members),
                    proposer="mind" if is_mind else "scripted")
                batch_seqs.append(batch_entry["seq"])
                proposing_halt = None
                for member in tool_members:
                    if cycles_used >= cycle_cap:
                        proposing_halt = (
                            STOP_BUDGET_EXHAUSTED,
                            "cycle budget exhausted mid-batch: "
                            f"{cycles_used} intent(s) evaluated of "
                            f"max_tool_cycles_per_task {cycle_cap}; "
                            "the batch's remaining members were "
                            "not evaluated, and the queued members "
                            "still drain (SI-50 §12.4.1)")
                        break
                    ctx.tool_cycles_used = cycles_used
                    ctx.tool_result_bytes_used = bytes_used
                    ctx.tool_journal_tip = journal.tip()[0]
                    ctx.tool_task_intent_seqs = set(task_intent_seqs)
                    outcome = admit_tool_intent(
                        journal, gate, ctx, member,
                        batch_seq=batch_entry["seq"])
                    cycles_used += 1
                    edit_id = outcome["edit_id"]
                    rows = journal.backend.find(
                        "self_edits", filter={"edit_id": edit_id},
                        limit=1)
                    if rows:
                        task_intent_seqs.add(rows[0]["seq"])
                    execution = outcome["execution"] or {}
                    step = {
                        "edit_id": edit_id,
                        "verdict": outcome["verdict"],
                        "tool": member.tool,
                        "effective_class": outcome["effective_class"],
                        "execution_status":
                            execution.get("status")
                            if outcome["execution"] is not None
                            else "queued — admitted, awaiting drain "
                                 "(SI-50 §5.2)",
                    }
                    steps.append(step)
                    last_edit_id = edit_id
                    if outcome["verdict"] == "ADMIT":
                        reject_streak = 0
                        continue
                    _note_outcome(edit_id)
                    if outcome["verdict"] == "ESCALATE":
                        proposing_halt = (
                            STOP_ESCALATED,
                            f"batch member ({member.tool}) "
                            "ESCALATEd; proposing halts at the "
                            "escalation, the batch's remaining "
                            "members are not evaluated, and the "
                            "queued members drain (SI-50 §12.4.2; "
                            "RFC §10.4.2)")
                        break
                    # REJECT
                    if _is_budget_reject(outcome["gate_report"]):
                        proposing_halt = (
                            STOP_BUDGET_EXHAUSTED,
                            "the gate REJECTed a batch member on "
                            "its budgets sub-check: proposing "
                            "stops, and the queued members still "
                            "drain (SI-50 §12.4.1)")
                        break
                    reject_streak += 1
                    if reject_streak >= 2:
                        proposing_halt = (
                            STOP_REJECT_STREAK,
                            "two consecutive REJECTs in admission "
                            "order: proposing halts, and the "
                            "queued members still drain "
                            "(SI-50 §12.4.3; RFC §10.4.3)")
                        break

                # Drain the task's queue (this batch's admissions
                # plus any earlier held/queued remainder), then
                # return ONE batched perception for the round.
                drain_res = drainer.drain_task(task["seq"])
                perception_sets.append(drain_res["perception"])
                folded = fold_queue(journal, task["seq"])
                by_edit = folded.intents
                for step in steps:
                    state = by_edit.get(step["edit_id"] or "")
                    if state is None:
                        continue
                    if state.outcome_seq is not None \
                            and state.execution is not None:
                        step["execution_status"] = \
                            state.execution.get("status")
                    elif state.state != "proposed":
                        step["execution_status"] = (
                            "not-executed — queue state "
                            f"{state.state} (SI-50 §5.2)")
                for edit_id, execution in \
                        drain_res["executions"].items():
                    bytes_used += int(
                        execution.get("result_bytes") or 0)
                    _note_outcome(edit_id)
                if drain_res["integrity_failure"]:
                    stop_reason = STOP_INTEGRITY_FAILURE
                    stop_detail = (
                        "execution integrity failure during the "
                        "drain: the drain froze task-wide "
                        "(SI-50 §12.4.4; RFC §9.2/§10.4.4 — halt "
                        "immediately, no streak arithmetic)")
                    break
                if proposing_halt is not None:
                    stop_reason, stop_detail = proposing_halt
                    break
                if drain_res["sync_reason"] == "stop-in-force":
                    stop_reason = STOP_OPERATOR_STOP
                    stop_detail = (
                        "an operator queue stop is in force for "
                        "this task (SI-50 §11): the drain is held, "
                        "queued intents stay journaled and "
                        "resumable only by an operator resume "
                        "entry; the task returns control incomplete "
                        "(NOT one of RFC §10.4's five stop rules)")
                    break

            if completions and stop_reason is None:
                member = completions[0]
                if scripted is not None:
                    _chain(member)
                outcome = _run_completion(journal, gate, ctx, member)
                steps.append({
                    "edit_id": outcome["edit_id"],
                    "verdict": outcome["verdict"],
                    "tool": None,
                    "effective_class": None,
                    "execution_status": outcome["execution_status"],
                })
                last_edit_id = outcome["edit_id"]
                _note_outcome(outcome["edit_id"])
                if outcome["verdict"] == "ADMIT":
                    folded = fold_queue(journal, task["seq"])
                    remaining = [
                        s for s in folded.intents.values()
                        if s.state in ("queued", "claimed", "held")]
                    stop_reason = STOP_MIND_REPORTED_COMPLETE
                    stop_detail = (
                        "the mind reported the task complete via a "
                        "task_complete memory intent, ADMITted as "
                        "an ordinary memory intent, after its "
                        "batch drained (SI-50 §12.4.5 — the label "
                        "records the report; whether the task is "
                        "actually done is the operator's judgement)")
                    if remaining:
                        stop_detail += (
                            f"; {len(remaining)} admitted intent(s) "
                            "remain undrained (held by an operator "
                            "stop) — the journal shows what was "
                            "actually done")
                elif outcome["verdict"] == "ESCALATE":
                    stop_reason = STOP_ESCALATED
                    stop_detail = (
                        "the task_complete memory intent ESCALATEd; "
                        "the task halts at the escalation "
                        "(RFC §10.4.2)")
                else:
                    reject_streak += 1
                    if reject_streak >= 2:
                        stop_reason = STOP_REJECT_STREAK
                        stop_detail = (
                            "two consecutive REJECTs (the second a "
                            "rejected task_complete report): the "
                            "task halts (RFC §10.4.3)")

        stop = record_task_stop(journal, task, stop_reason,
                                stop_detail, cycles_used, bytes_used)
        updated = _anchor_checkpoint(
            journal, manifest, manifest["head"],
            note=f"task {task['ref']} stopped: {stop_reason} "
                 f"({stop_detail})")
        _write_manifest(bundle_dir, updated)
        return {
            "task_seq": task["seq"],
            "task_ref": task["ref"],
            "stop_reason": stop_reason,
            "stop_detail": stop_detail,
            "stop_seq": stop["seq"],
            "cycles_used": cycles_used,
            "result_bytes_used": bytes_used,
            "steps": steps,
            "head": updated["head"],
            "perceptions": perception_sets,
            "batch_seqs": batch_seqs,
        }
    finally:
        journal.close()
