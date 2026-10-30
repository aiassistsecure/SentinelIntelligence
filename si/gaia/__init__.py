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
§10.4 stop rules, the ``si task`` CLI). Demo bundles keep shadow
behaviour throughout — claiming otherwise would be fabrication
(RFC §15).

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
