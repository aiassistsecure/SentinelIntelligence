"""Gaia — SI's agentic layer (SI-41 line; Mark's codename, 2026-10-29).

Gaia is the layer that acts on computer systems via tool use; SI is
the model/mind. Gaia code lives in this subpackage of the
SentinelIntelligence repo (Mark, 2026-10-29): the agentic loop's
home — B-1 (SI-42) creates it with the tool registry, the gate's
tool policy resolution, the scripted verdict corpus, and the shadow
journaling helpers; successor slices grow it (executor, task loop,
SSE stream, Hearth bridge).

B-1 is SHADOW only: tool intents are gated and journaled, every
outcome ESCALATEs or REJECTs, and execution is recorded as
``not-evaluated — shadow mode``. No executor exists in this package
yet and nothing here runs a tool — claiming otherwise would be
fabrication (RFC §15 B-1).

This ``__init__`` deliberately imports only the registry and policy
submodules (stdlib-only, no gate import) so ``si.gate`` can import
them without a cycle; :mod:`si.gaia.shadow` and
:mod:`si.gaia.corpus` are imported by their full module paths.
"""
from si.gaia.policy import TOOL_POLICY_DEFAULTS, resolve_tool_policy
from si.gaia.registry import (
    REGISTRY,
    RFC_V1_TOOL_NAMES,
    TOOL_CLASSES,
    ToolSpec,
    get_tool,
    validate_args,
    validate_result,
)

__all__ = [
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
