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

SI-51 (the SI-50 spec's Q-1, the queue-drain core) turns the
journal into the work queue: :mod:`si.gaia.queue` journals the
queue's entry kinds (batch / claim / recovery re-claim / refusal
/ operator stop / resume) and folds queue state from the journal
alone, and :mod:`si.gaia.drain` admits batched proposals
(per-member verdicts, gate-priced ordering keys and depends_on)
and drains them under claim-before-execute through the
B-2/B-3 executor, returning one batched perception per sync
point. SI-52 (Q-2) adds depends_on requirement semantics with
journaled skips, barrier sync points, claim-time residence
refusals, and the bounded read pool (unkeyed reads concurrent,
default bound 4; keyed and state-changing work serial). See
docs/SI-51-q1-queue-core.md and
docs/SI-52-q2-concurrency-deps.md for exactly what is live and
what is not (Q-3 outcome-unknown recovery, Q-4 measurement).

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
