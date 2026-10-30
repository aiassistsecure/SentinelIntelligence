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
