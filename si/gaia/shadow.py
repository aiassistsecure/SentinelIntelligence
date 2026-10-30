"""Gaia B-1 shadow journaling (SI-41 RFC §15 B-1, §16 criteria 6-7).

Shadow discipline, per the SI-16 precedent: in B-1 every tool intent
is gated and journaled, and **nothing executes**. This module is the
journaling half of that guarantee:

- :func:`record_task_fixture` journals a scripted operator task
  entry (kind ``task``, in the existing ``experiences`` collection)
  — a *fixture*, because the real task-opening surface, ``si task``,
  is B-3 and is deliberately not built here (RFC §15, §16.7).
- :func:`journal_tool_intent` gates one tool intent and journals the
  intent, the verdict, its provenance edges, and its shadow outcome
  through the journal's *existing* methods, in the *existing*
  collections (``self_edits``, ``provenance``, ``experiences``) —
  no new collection, no NEDB schema change (RFC §4.6).
- :func:`walk_tool_provenance` answers ``si why`` for a tool edit:
  the walk from the tool intent's self-edit, over its provenance
  edges and cited source seqs, to the task entry that caused it.

The execution record written for every tool intent is exactly::

    {"status": "not-evaluated — shadow mode",
     "execution": "not-evaluated — shadow mode", ...}

— the SI-16/SI-9b honest label: not a pass, not a failure, a plain
statement that execution was not evaluated because no executor
exists in B-1. The self-edit's result is ``escalated`` / ``rejected``
per the verdict — never ``materialized``: nothing materialised.
"""
from __future__ import annotations

from si.gate import TOOL_SHADOW_EXECUTION

#: The shadow execution label (single source: si.gate, so the gate
#: report and the journal record can never drift apart).
SHADOW_EXECUTION = TOOL_SHADOW_EXECUTION


def record_task_fixture(journal, instruction: str, work_root: str,
                        budgets: dict | None = None,
                        operator: str = "operator") -> dict:
    """Journal a scripted operator task entry; return its fixture ref.

    The entry lands in ``experiences`` via the existing
    :meth:`Journal.record_experience` (kind ``task``,
    ``source: "operator"``, the instruction text, the work root, and
    the budgets in force — RFC §10.1's task-entry shape). Its
    self-edit slot is empty: a task is opened by the operator, not
    by a self-edit. Returns ``{"seq", "ref", "row_id"}`` where
    ``ref`` is ``task@<seq>`` — the parent ref a tool intent cites
    to chain to this task (gate check 5, RFC §5.2).
    """
    record = {
        "kind": "task",
        "source": "operator",
        "operator": operator,
        "instruction": instruction,
        "work_root": work_root,
        "budgets": dict(budgets or {}),
        "fixture": "scripted task entry (SI-42/B-1 fixture; the "
                   "operator task surface `si task` is B-3)",
    }
    row_id = journal.record_experience("task", record, "")
    rows = journal.backend.find("experiences", filter={"kind": "task"},
                                sort="-seq", limit=1)
    seq = rows[0]["seq"] if rows else journal.tip()[0]
    return {"seq": seq, "ref": f"task@{seq}", "row_id": row_id}


def shadow_execution_record(intent, edit_id: str, verdict: str,
                            effective_class: str | None = None) -> dict:
    """The §4.4-shaped execution record, honestly shadow-labelled.

    No exit code, no result hash, no wall time: none was measured,
    because nothing ran. Fields that would carry measured values in
    B-2/B-3 are absent here rather than filled with placeholders —
    the record says what happened (nothing) and why (shadow mode).
    """
    spec_name = getattr(intent, "tool", "")
    return {
        "tool": spec_name,
        "intent_id": edit_id,
        "class": effective_class,
        "status": SHADOW_EXECUTION,
        "execution": SHADOW_EXECUTION,
        "verdict": verdict,
    }


def record_shadow_outcome(journal, intent, edit_id: str, verdict: str,
                          effective_class: str | None = None) -> dict:
    """Journal one tool intent's shadow outcome as an experience.

    The single writer of the ``tool_result`` experience shape, used
    by both :func:`journal_tool_intent` and the self-write loop's
    tool branch (si.loop) so the two paths can never drift.
    Returns the execution record journaled.
    """
    execution = shadow_execution_record(intent, edit_id, verdict,
                                        effective_class=effective_class)
    journal.record_experience("tool_result", {
        "kind": "tool_result",
        "source": f"tool:{getattr(intent, 'tool', '')}",
        "intent_id": edit_id,
        "tool": getattr(intent, "tool", ""),
        "gate_verdict": verdict,
        "effective_class": effective_class,
        "execution": execution,
    }, edit_id)
    return execution


def journal_tool_intent(journal, gate, intent, ctx=None) -> dict:
    """Gate one tool intent and journal it, shadow-style.

    Journals (all via existing methods, existing collections): the
    self-edit (``record_self_edit``, kind ``tool``, with the full
    gate report), the provenance edges (``record_provenance``:
    parents as ``learned_from``, source seqs as the loop records
    them), and the shadow outcome (``record_experience``, kind
    ``tool_result``, carrying the execution record and a
    ``source: tool:<name>`` tag in the record, per RFC §4.5's
    provenance-tagged observation shape — in shadow mode the
    "result" is the labelled non-execution itself).

    Returns {"verdict", "edit_id", "gate_report", "execution",
    "effective_class"}. Never executes anything: this function has
    no executor to call, by design.
    """
    report = gate.evaluate(intent, None, ctx)
    info = gate.classify_tool_intent(intent, ctx) \
        if hasattr(gate, "classify_tool_intent") else {}
    effective = info.get("effective_class")
    edit_id = journal.record_self_edit(
        intent.kind, intent.to_dict(), intent.reason,
        report.verdict, report.to_dict())
    journal.set_edit_result(
        edit_id, "rejected" if report.verdict == "REJECT" else "escalated")
    if intent.parents:
        journal.record_provenance(
            child=edit_id, parents=list(intent.parents),
            relation="learned_from", note=(intent.reason or "")[:200])
    if intent.source_seqs:
        journal.record_provenance(
            child=edit_id, parents=list(intent.source_seqs),
            relation="learned_from", note="source journal sequences")
    execution = record_shadow_outcome(journal, intent, edit_id,
                                      report.verdict,
                                      effective_class=effective)
    return {"verdict": report.verdict, "edit_id": edit_id,
            "gate_report": report.to_dict(), "execution": execution,
            "effective_class": effective}


def _find_by_seq(journal, seq) -> dict | None:
    """The journal row (any collection) carrying ``seq``, or None."""
    from si.journal import COLLECTIONS
    for coll in COLLECTIONS:
        rows = journal.backend.find(coll, filter={"seq": seq}, limit=1)
        if rows:
            return {"collection": coll, "row": rows[0]}
    return None


def walk_tool_provenance(journal, edit_id: str) -> dict:
    """Walk a tool self-edit's provenance to its task entry.

    Returns {"edit", "provenance", "sources", "task"}: the self-edit
    row; its provenance edges; the journal rows its ``source_seqs``
    cite (resolved across all five collections); and the task entry
    among them (an ``experiences`` row of kind ``task``), or None
    when the cited chain reaches no task — reported as None, never
    invented. This is the data ``si why <edit-id>`` prints.
    """
    edits = journal.backend.find("self_edits",
                                 filter={"edit_id": edit_id}, limit=1)
    if not edits:
        raise KeyError(f"unknown edit_id {edit_id}")
    edit = edits[0]
    prov = journal.backend.find("provenance", filter={"child": edit_id})
    sources = []
    task = None
    for seq in (edit.get("intent") or {}).get("source_seqs") or []:
        found = _find_by_seq(journal, seq)
        if found is None:
            sources.append({"seq": seq, "collection": None, "row": None})
            continue
        sources.append(found | {"seq": seq})
        row = found["row"]
        record = row.get("record") or {}
        if found["collection"] == "experiences" and (
                row.get("kind") == "task" or record.get("kind") == "task"):
            task = found
    return {"edit": edit, "provenance": prov, "sources": sources,
            "task": task}
