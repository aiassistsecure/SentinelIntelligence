"""Gaia — SI's agentic layer (SI-41 line; Mark's codename, 2026-10-29).

Gaia is the layer that acts on computer systems via tool use; SI is
the model/mind. Gaia code lives in this subpackage of the
SentinelIntelligence repo (Mark, 2026-10-29): the agentic loop's
home — B-1 (SI-42) creates it with the tool registry, the gate's
tool policy resolution, the scripted verdict corpus, and the shadow
journaling helpers; successor slices grow it (executor, task loop,
SSE stream, Hearth bridge).

B-1 was SHADOW only: tool intents were gated and journaled, every
outcome ESCALATEd or REJECTed, and execution was recorded as
``not-evaluated — shadow mode``. B-2 (SI-43) adds
:mod:`si.gaia.executor`: live execution of the read/write tool set
on miniature bundles only, under the journaled-args rule. B-3
(SI-44) grows the executor to the execute-class tools
(``shell_exec`` / ``code_exec`` / ``test_run``) and adds the task
loop (:mod:`si.gaia.task`: operator tasks, chaining, budgets,
§10.4 stop rules, the ``si task`` CLI). B-4 (SI-45) adds
:mod:`si.gaia.hearth`: the §12.2.1 declaration record, the
residency PERCEIVE adapter (observations tagged
``hearth:/residency``, journaled via existing methods), and the
gateway client whose inference leg is conditional on a
Hearth-served real base (§12.3) — exercised against a labelled
replay/stub, with the Hearth path UNMEASURED. B-5 (SI-46) adds
:mod:`si.gaia.service`: the §12.5(a) service adapter — a
stdlib-HTTP service in ``si serve``'s pattern that accepts a
task with a mandatory journaled submitter identity (asserted,
not authentication), runs it through the B-3 machinery, and
returns the result with its provenance trail serialised from
the journal. Exposure through the Hearth gateway remains
proposal H-2, not built. Demo bundles keep shadow
behaviour throughout — claiming otherwise would be fabrication
(RFC §15). SI-47 adds :mod:`si.gaia.events`: the SSE event
spine — every journal row serialised as a server-sent event
whose id IS the journal seq (Last-Event-ID = journal seq, so
replay and live stream are one mechanism), exposed as
``GET /events`` on the B-5 service; the live tail is
poll-the-journal, labelled as polling. SI-48 adds filtered
subscriptions on that surface — ``collection`` / ``event`` /
``kind`` / ``task`` filters as a pure projection of the same
stream (ids stay journal seqs; gaps are filtered-out seqs) —
plus :mod:`si.gaia.events_client`, the stdlib-only reference
consumer (``si gaia-events``).

This ``__init__`` deliberately imports only the registry and policy
submodules (stdlib-only, no gate import) so ``si.gate`` can import
them without a cycle; :mod:`si.gaia.shadow`, :mod:`si.gaia.corpus`,
and :mod:`si.gaia.executor` are imported by their full module
paths.
"""
from si.gaia.policy import TOOL_POLICY_DEFAULTS, resolve_tool_policy
from si.gaia.registry import (
    LIVE_TOOLS_B2,
    LIVE_TOOLS_B3,
    REGISTRY,
    RFC_V1_TOOL_NAMES,
    TOOL_CLASSES,
    ToolSpec,
    get_tool,
    validate_args,
    validate_result,
)

__all__ = [
    "LIVE_TOOLS_B2",
    "LIVE_TOOLS_B3",
    "REGISTRY",
    "RFC_V1_TOOL_NAMES",
    "TOOL_CLASSES",
    "TOOL_POLICY_DEFAULTS",
    "ToolSpec",
    "get_tool",
    "resolve_tool_policy",
    "validate_args",
    "validate_result",
]
