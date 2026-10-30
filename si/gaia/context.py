"""The gate context contract for tool intents (SI-42 / B-1).

The gate's tool branch (``si.gate``) reads *only* optional attributes
from its context — never the demo BundleContext methods — so any
object carrying these attributes (or none of them) can gate a tool
intent:

- ``tool_work_root`` (str | None): the task's declared execution
  scope root (RFC §9.1). Fallback order in the gate: this attribute,
  then a ``bundle_dir`` attribute, then the policy's
  ``tool_work_root``, then the process working directory.
- ``tool_task_seq`` (int | None): the journal seq of the operator
  task instruction that opened the current task (RFC §10.1). When
  set, gate check 5 requires the intent's ``source_seqs`` to include
  it (the chain must reach the task, RFC §5.2). When unset, the
  intent is priced as an out-of-task proposal, exactly as a delta's
  cited causes are priced.
- ``tool_cycles_used`` (int): tool cycles already spent in the task;
  priced against ``max_tool_cycles_per_task`` at check 2 (§10.3).
- ``tool_result_bytes_used`` (int): inline result bytes already
  drawn in the task; priced against ``tool_max_task_result_bytes``.
- ``tool_live_execution`` (bool, SI-43 / B-2): the context's loop
  path is priced LIVE for the B-2 set (read/write tools on a
  miniature bundle — RFC §15 B-2, by the SI-17 precedent of pricing
  paths separately). The miniature loop's ``ProbeContext`` carries
  True; every other context (the demo ``LoopContext``, scripted
  contexts that do not set it) defaults to False and keeps B-1
  shadow behaviour exactly.

RFC status, stated honestly: §6.2 pins the policy caps and §10.3
pins the budgets, but the *runtime bookkeeping* that tracks spend
inside a task belongs to the B-3 task loop, which does not exist
yet. B-1 pins this attribute contract as the gate-facing shape of
that bookkeeping (an open field the RFC leaves to the slice; see
the slice doc). :class:`ToolContext` is the reference carrier, used
by the scripted corpus and the tests; the loop's own contexts can
grow the same attributes without a gate change.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ToolContext:
    """Reference carrier for the tool gate-context attributes."""

    tool_work_root: str | None = None
    bundle_dir: str | None = None
    tool_task_seq: int | None = None
    tool_cycles_used: int = 0
    tool_result_bytes_used: int = 0
    tool_live_execution: bool = False
    # SI-51 (SI-50 §6/§7): queue-pricing context, set by the task
    # loop per batch — ``tool_queue_batch_seq`` (the batch entry's
    # journal seq, priced into the report's queue block),
    # ``tool_journal_tip`` (the journal tip when the batch was
    # proposed; a depends_on seq beyond it is not yet journaled),
    # and ``tool_task_intent_seqs`` (the journal seqs of the task's
    # already-journaled intents; a depends_on seq outside it is
    # outside the task). All None outside a queue batch, in which
    # case a declared depends_on fails closed at check 2 (§6.2).
    tool_queue_batch_seq: int | None = None
    tool_journal_tip: int | None = None
    tool_task_intent_seqs: object = None
