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
"""
from __future__ import annotations

import os

from si.gaia.executor import execute_tool_cycle
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

    Opens (journals) the task before any cycle, then runs serial
    gated cycles until a §10.4 stop rule fires (or the proposer
    is exhausted — labelled, not a §10.4 rule). Re-anchors the
    manifest checkpoint and rewrites the manifest before
    returning, so the bundle stays loadable with the whole task
    trail in its journal.

    Returns {"task_seq", "task_ref", "stop_reason", "stop_detail",
    "cycles_used", "result_bytes_used", "steps", "head"} where
    each step is {"edit_id", "verdict", "tool", "effective_class",
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

        task_observation = {
            "kind": "task", "source": "operator", "seq": task["seq"],
            "ref": task["ref"], "instruction": instruction,
            "work_root": root, "budgets": dict(budgets),
        }
        observations: list = []
        steps: list = []
        cycles_used = 0
        bytes_used = 0
        reject_streak = 0
        stop_reason = None
        stop_detail = ""
        last_edit_id = None
        last_outcome_seq = None

        while stop_reason is None:
            # §10.4.1, checked before proposing: either budget
            # reached means the next intent would fail check 2 —
            # the task halts as budget-exhausted, a priced outcome.
            if cycles_used >= cycle_cap:
                stop_reason = STOP_BUDGET_EXHAUSTED
                stop_detail = (
                    f"cycle budget exhausted: {cycles_used} tool "
                    f"cycle(s) used of max_tool_cycles_per_task "
                    f"{cycle_cap} (RFC §10.3)")
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
                intent = proposer.propose(perceptions, ctx)
            else:
                try:
                    intent = next(scripted)
                except StopIteration:
                    intent = None
            if intent is None:
                stop_reason = STOP_PROPOSER_EXHAUSTED
                stop_detail = (
                    "the proposer produced no further intent: the "
                    "task is incomplete (NOT one of RFC §10.4's "
                    "five stop rules — there is no rule 6)")
                break
            if not isinstance(intent, Intent):
                raise TypeError(
                    "a task proposer must yield si.gate.Intent "
                    f"objects, got {type(intent).__name__}")

            if scripted is not None:
                # §10.2 chaining, filled mechanically for scripted
                # intents that did not author their own chain.
                if not intent.parents:
                    intent.parents = [task["ref"]] + (
                        [last_edit_id] if last_edit_id else [])
                if not intent.source_seqs:
                    intent.source_seqs = [task["seq"]] + (
                        [last_outcome_seq]
                        if last_outcome_seq is not None else [])

            ctx.tool_cycles_used = cycles_used
            ctx.tool_result_bytes_used = bytes_used

            # §10.4.5: the mind reports completion via a memory
            # intent — an ordinary gated memory cycle, not a tool
            # cycle (it spends no tool-cycle budget).
            if intent.kind == "memory" and \
                    (intent.record or {}).get("kind") == "task_complete":
                outcome = _run_completion(journal, gate, ctx, intent)
                steps.append({
                    "edit_id": outcome["edit_id"],
                    "verdict": outcome["verdict"],
                    "tool": None,
                    "effective_class": None,
                    "execution_status": outcome["execution_status"],
                })
                last_edit_id = outcome["edit_id"]
                seq = _outcome_seq(journal, outcome["edit_id"])
                if seq is not None:
                    last_outcome_seq = seq
                if outcome["verdict"] == "ADMIT":
                    stop_reason = STOP_MIND_REPORTED_COMPLETE
                    stop_detail = (
                        "the mind reported the task complete via a "
                        "task_complete memory intent, ADMITted as "
                        "an ordinary memory intent (RFC §10.4.5 — "
                        "the label records the report; whether the "
                        "task is actually done is the operator's "
                        "judgement)")
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
                continue

            if intent.kind != "tool":
                # The B-3 task loop drives tool intents (and the
                # task_complete memory intent above). Any other
                # kind is a proposer contract violation, counted
                # as a REJECT step for streak purposes — journaled
                # in the steps list and the stop detail, never
                # executed and never given a fabricated gate report.
                steps.append({
                    "edit_id": None, "verdict": "REJECT",
                    "tool": None, "effective_class": None,
                    "execution_status":
                        "not-executed — the B-3 task loop drives "
                        "tool intents only",
                })
                reject_streak += 1
                if reject_streak >= 2:
                    stop_reason = STOP_REJECT_STREAK
                    stop_detail = (
                        "two consecutive REJECTs (unsupported "
                        "intent kinds proposed to the task loop): "
                        "the task halts (RFC §10.4.3)")
                continue

            outcome = execute_tool_cycle(
                journal, bundle_dir, manifest, intent, gate, ctx)
            execution = outcome["execution"] or {}
            cycles_used += 1
            bytes_used += int(execution.get("result_bytes") or 0)
            steps.append({
                "edit_id": outcome["edit_id"],
                "verdict": outcome["verdict"],
                "tool": intent.tool,
                "effective_class": outcome["effective_class"],
                "execution_status": execution.get("status"),
            })
            last_edit_id = outcome["edit_id"]
            seq = _outcome_seq(journal, outcome["edit_id"])
            if seq is not None:
                last_outcome_seq = seq
                rows = journal.backend.find(
                    "experiences",
                    filter={"self_edit_id": outcome["edit_id"]},
                    sort="-seq", limit=1)
                if rows:
                    # §4.5: the observation the next PERCEIVE hands
                    # the mind — the journaled tool_result record,
                    # provenance tags intact, as data.
                    observations.append(
                        dict(rows[0].get("record") or {},
                             seq=rows[0]["seq"]))

            # §10.4.4: an integrity refusal halts immediately.
            if outcome["verdict"] == "ADMIT" \
                    and execution.get("status") == "refused" \
                    and _is_integrity_refusal(execution):
                stop_reason = STOP_INTEGRITY_FAILURE
                stop_detail = (
                    "execution integrity failure: "
                    + str(execution.get("detail") or "")
                    + " (RFC §9.2/§10.4.4 — halt immediately, no "
                      "streak arithmetic)")
                break
            # §10.4.2: any ESCALATE halts the task at the escalation.
            if outcome["verdict"] == "ESCALATE":
                stop_reason = STOP_ESCALATED
                stop_detail = (
                    f"cycle {cycles_used} ({intent.tool}) "
                    "ESCALATEd; the task halts at the escalation "
                    "and the journaled intent is the operator's to "
                    "admit or refuse (RFC §10.4.2)")
                break
            if outcome["verdict"] == "REJECT":
                # §10.4.1: a budget REJECT maps to budget-exhausted.
                if _is_budget_reject(outcome["gate_report"]):
                    stop_reason = STOP_BUDGET_EXHAUSTED
                    stop_detail = (
                        "the gate REJECTed the intent on its "
                        "budgets sub-check: a budget is exhausted "
                        "(RFC §10.4.1)")
                    break
                reject_streak += 1
                # §10.4.3: two consecutive REJECTs halt the task.
                if reject_streak >= 2:
                    stop_reason = STOP_REJECT_STREAK
                    stop_detail = (
                        "two consecutive REJECTs: the mind's model "
                        "of its situation is priced as wrong, and "
                        "continuing would spend budget against a "
                        "broken premise (RFC §10.4.3)")
                    break
                continue
            reject_streak = 0

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
        }
    finally:
        journal.close()
