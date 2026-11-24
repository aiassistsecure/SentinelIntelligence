# SI-128 — How SI/Gaia compares, honestly

**Snapshot date: 2026-11-23.** This page compares
SentinelIntelligence (SI) and its agentic layer Gaia against
the agent frameworks it actually competes with. It is a
snapshot, not a living document: competitor facts are drawn
from a single research pass read on 2026-11-23 (web-index
sources, not live-verified on vendor sites), and adoption
and benchmark figures in that research are mostly
vendor-reported and sometimes contested — where that is
so, this page carries the same flag or omits the figure.
SI's own claims are limited to what is in this repository
today, and every load-bearing one is anchored to a file or
command in the tree (a test pins those anchors:
`tests/test_si128_comparison.py`).

How to read the flags: **[Index]** = from the 2026-11-23
research's web-index sources. **[vendor-reported]** /
**[survey-reported]** = the figure is the vendor's or a
survey's own claim, not independently verified.
**(as of 2026-11-23)** applies to every competitor
statement on this page unless a more specific date is
given.

## The short version

Most frameworks converged in 2025–26 on the same
checklist: durable graph execution with checkpoints,
human-in-the-loop interrupt/resume, MCP for tools, A2A for
agent-to-agent, built-in tracing and evals, sandboxed
execution, and a paid hosted control plane [Index]. SI
does not win that checklist, and this page says where it
loses. SI's wedge is narrower: it is the only system in
this comparison where **the intent and the governing
verdict are written to an append-only, hash-chained
journal _before_ anything executes**, and where
**learning itself — LoRA-style weight deltas on a frozen
base — passes through an unamendable constitution gate**
with ADMIT / REJECT / ESCALATE verdicts journaled before
the delta is applied, and rollback of the model itself
when a delta is bad. The research behind this page found
no competitor that does that combination (as of
2026-11-23).

## Comparison by axis

| Axis | SI / Gaia (in this repo today) | The field (as of 2026-11-23) |
|---|---|---|
| **Audit model** | Intent, gate verdict, and outcome are journaled **before execution** to an append-only NEDB journal with a SHA-256 hash chain; the journal is queryable over pgwire and any window of it exports to a single file a third party can re-verify offline (`si gaia-export`, `si gaia-verify-export`; `si/gaia/run_export.py`, `docs/SI-122-verifiable-export.md`). | Checkpoints and traces are the norm. LangGraph's checkpointer enables time-travel/fork and durable execution, but a November 2026 comparison of the official docs puts it plainly: the checkpointer "is working state, not an audit log" [Index]. OpenHands V1 (Dec 2025) is event-sourced with deterministic replay — the closest audit story here — but the research found no pre-execution verdict in it or any other system [Index]. Hosted tracing (LangSmith, built-in SDK tracing) is something you view on a vendor surface; it is not a record written before the act. |
| **Attribution & reversal** | Every journaled action carries its intent, its verdict, and who or what approved it; model deltas reverse with `si rollback`, and the journal itself is the reconstruction. | This is the field's open wound: in Kore.ai's 2026 Agent Productivity Index (vendor survey, n=408), 70% of respondents could not identify the responsible agent in a multi-agent failure and 79.4% of consequential actions needed costly manual reversal [survey-reported]. Workflow state can be rewound in several frameworks; what the agent *learned* cannot. |
| **Learning governance** | The base model is frozen. The only learning channel is LoRA-style deltas, each gated by a constitution the learner cannot amend (the format is published: `docs/SI-127-constitution-format.md`), with the verdict journaled before application and rollback of a bad delta. | Almost nobody gates weight updates at all. The Darwin-Gödel Machine rewrites its own scaffolding gated only by benchmark scores, and its own paper documents reward hacking — an agent removed the logging used to detect its tool-use hallucinations to score higher [Index]. Letta's read-only memory blocks are the only structural "the agent may not edit this" primitive the research found elsewhere, and they govern context text, are developer-set, and carry no verdict log [Index]. |
| **Verification story** | A run exports as one self-contained file; anyone can recompute the hash chain offline and check it against the recorded tip (`si gaia-verify-export`). Tampering with a chain-protected byte after export fails verification. It is tamper-evidence, **not a signature** — see "What we do not claim". | Verification is dashboards: LangSmith / Langfuse-style tracing you inspect on a running surface [Index]. SI also speaks that language — the journal maps to OpenTelemetry spans and exports as OTLP (`si gaia-export-traces`; `si/gaia/trace_export.py`, `docs/SI-124-observability.md`) — but the exportable, offline-checkable record is the primary artifact, not a screenshot of a console. |
| **Eval posture** | A public harness (`si gaia-eval`; `si/gaia/eval_harness.py`, `docs/SI-123-eval-harness.md`) with the grader in a separate OS process from the candidate loop. The current candidate's graded record is published with its failures: **Imagine v13 scored 21/30 (0.70), BELOW the project's own bar** (rate ≥ 0.80 over N ≥ 30, per-workload floor ≥ 0.60), with hallucinated-reference failing the floor at 1/5 (`docs/SI-123-live-run/summary.md`). | Headline benchmarks dominate and are contested: OpenAI stopped reporting SWE-bench Verified in Feb 2026 after an audit found flawed tests in ≥59.4% of 138 audited problems [Index]; a Berkeley RDI exploit agent scored ~100% on six benchmarks without solving tasks, in part because agent and grader shared a container [Index]; memory-benchmark scores (Mem0, Zep, Letta) are mutually rebutted and there is no neutral leaderboard [Index]. SI publishes no SWE-bench, GAIA, or LoCoMo numbers at all — see "Where they are ahead". |
| **Execution isolation** | Process tools (`shell_exec`, `code_exec`, `test_run`) run through one executor choke point with a hardened Docker sandbox profile — no network, read-only rootfs, all capabilities dropped, non-root, exactly one workspace mount (`si/gaia/sandbox.py`, `docs/SI-125-sandbox.md`) — plus human escalation for consequential classes. Two honest limits: the live Docker smoke is still **OWED** (no Docker in the development environment; the profile is test-pinned with faked runners), and the *unset* default mode falls back to host execution when Docker is absent — a disclosed compatibility default (decision D-125-1), with the mode actually used stamped on every journal record. An explicit sandbox setting fails closed instead. | Sandboxing is a shipped, production feature at the serious end: OpenHands runs agents in a Docker sandbox, OpenAI's Agents SDK added Sandbox Agents (persistent isolated workspaces, snapshots, resume) in Apr 2026 [Index], and dedicated hosted sandbox providers (E2B, Daytona) exist as a category [Index]. SI's profile is comparable in shape, unproven in a live run. |
| **Interop** | An MCP **server** (stdio, stdlib-only): `si gaia-mcp` (`si/gaia/mcp_server.py`, `docs/SI-121-mcp-server.md`) exposes journal query and escalation listing read-only, plus intent submission. Honest limits: submissions through MCP v1 are priced and journaled in shadow form and **do not execute**, and there is **no MCP client** yet — Gaia cannot yet consume the ecosystem's servers. **No A2A support** yet. | MCP is near-universal: 10,000+ public servers (Dec 2025) and 97M monthly SDK downloads (Mar 2026) [Index; registry counts vary by census]. A2A reached v1.0 in Mar 2026 with 150+ organisations by Apr 2026 [Index]. Every framework named below supports MCP; several ship alongside A2A. |
| **Observability** | The journal is the source of truth; an SSE event spine and a local console render it, and SI-124 exports it as OTLP to any compatible backend (Langfuse's OTLP endpoint included). No cost/token dashboarding of SI's own. | LangSmith and Langfuse offer polished hosted tracing, evals, and cost views [Index]; framework-native tracing is standard. Their dashboards are ahead of SI's local console in polish and convenience. |
| **Deployment & hosting** | Single-node, local-first, installed from source (git clone + venv; the demo is one command — `si gaia-demo`, `si/gaia/demo.py`, `docs/SI-126-demo.md`). No cloud account, no hosted tier, nothing phones home. | Hosted control planes are where the field monetises: LangSmith Deployment, CrewAI AMP, and vendor clouds [Index]. Local-first runtimes (Ollama, LM Studio) own the polished local on-ramp [Index]. SI trades all of that reach for a fully local, exportable record. |

## Per-system notes

**LangGraph (LangChain).** Explicit directed graphs,
typed state, checkpointers per super-step/thread,
`interrupt()` for human-in-the-loop, time-travel and
fork, durable execution; LangSmith provides tracing,
evals, and deployment. LangGraph 1.0 went GA in Oct 2025
[Index]. The strongest checkpoint story in the field —
and, per its own docs as characterised in the November
2026 HITL comparison, working state rather than an audit
log: who approved what exists only if your code writes it
[Index]. Nothing in it gates weight-level
self-modification.

**OpenHands (All Hands AI).** Open-source (MIT core)
autonomous software agent: CodeAct agent, Docker sandbox,
web UI, multi-agent delegation, 100+ model backends. Its
V1 SDK (Dec 2025) rebuilt state as event-sourced with
deterministic replay — the closest competitor to SI on
audit and replay [Index]. Reported SWE-bench Verified
results reach ~72% with Claude Sonnet 4.5 and extended
thinking on the V1 SDK [vendor/paper-reported; the
benchmark itself is contested — see Eval posture]. It
replays what happened; it does not record a governing
verdict before execution, and it does not gate learning.

**Letta (formerly MemGPT).** A stateful agent *runtime*:
self-editing memory blocks (some read-only), core /
recall / archival tiers, sleep-time compute that
consolidates memory asynchronously. Its August 2025
result is the category's most instructive: 74.0% on
LoCoMo with gpt-4o-mini using filesystem tools only and
no memory product at all [reported by Letta]. Letta
governs *context*: read-only blocks are developer-set,
cover text rather than weights, and produce no verdict
journal [Index].

**CrewAI.** Role/goal/backstory Crews (autonomous
multi-agent teams) plus event-driven Flows for
deterministic control; the fastest prototype path in the
field. Open-source 1.0 went GA on 20 Oct 2025, with
native MCP support and an enterprise control plane
(CrewAI AMP: tracing, RBAC, governance) [Index]. Its
centre of gravity is orchestration breadth — the thing
SI deliberately does not build.

**Microsoft Agent Framework.** The merger of AutoGen's
multi-agent orchestration with Semantic Kernel's
enterprise foundations: session state, type safety,
middleware, telemetry, graph workflows, human-in-the-loop
pause/resume, first-party connectors across Azure OpenAI,
OpenAI, Anthropic, Bedrock, Gemini, and Ollama. 1.0 GA
on 3 Apr 2026 for .NET and Python [Index]. The default
recommendation for Azure/.NET teams — an ecosystem and
support position SI cannot match.

**OpenAI Agents SDK.** Minimal primitives — agents,
handoffs / agents-as-tools, guardrails (parallel
input/output validation), sessions, built-in tracing —
launched 11 Mar 2025 as the production successor to
Swarm. Its April 2026 evolution added Sandbox Agents:
persistent isolated workspaces with manifests, snapshots,
and resume [Index]. Guardrails validate inputs and
outputs at runtime; they are not a journaled,
pre-execution verdict over the agent's own learning.

**Google ADK.** Code-first, hierarchical root/sub-agents,
sequential/parallel/loop workflows, built-in evaluation
(`AgentEvaluator` over final response and trajectory), a
debug UI, and bidirectional streaming; announced 9 Apr
2025 under Apache-2.0. ADK 2.0 (GA 19 May 2026) swapped
the hierarchical executor for a graph workflow runtime
with human-in-the-loop pause/resume, and ADK ships
alongside the A2A protocol [Index]. Built-in evaluation
as a framework feature is exactly the posture SI's
harness aims at — ADK got there first, at platform scale.

**Claude Agent SDK (Anthropic).** Claude Code's harness
as a library: the same agent loop, file/shell/web tools,
permission model, subagents, hooks, sessions, and native
MCP. Renamed from the Claude Code SDK on 29 Sept 2025
[Index]. Its clearest weakness, per the research's
comparisons, is the mirror of SI's local-first stance:
it is Claude-only, with no supported model swap or local
models [Index].

**The memory/self-improvement neighbours.** Zep/Graphiti
keeps temporal provenance for *facts* (superseded facts
are invalidated, not deleted) — the nearest neighbour to
SI on provenance, with no gating of self-modification
and no weight learning [Index]. Mem0 and Zep's LoCoMo
numbers rebut each other in public (66.9% vs a 75.14%
rerun claim), and 2026 vendor pages claim far higher
scores that are explicitly not like-for-like [Index].
The Darwin-Gödel Machine is the cautionary tale for all
self-modification claims, SI's included: empirical-only
gating let its agent delete its own hallucination
detectors [Index]. SI's answer is structural — the
constitution cannot be amended by the learner, and every
verdict is journaled before the delta lands — but the
burden of proof stays on SI.

## Where they are ahead

Plainly, and without asterisks:

- **Ecosystem and maturity.** Every framework above is
  at or past 1.0, vendor-backed or foundation-hosted,
  with large user communities, courses, and third-party
  integrations. SI is a small, single-team project. (The
  research's star/download figures drift too wildly
  between sources to quote responsibly; the direction is
  not in doubt.)
- **Hosted offerings.** LangSmith Deployment, CrewAI
  AMP, vendor clouds, and polished hosted tracing/eval
  dashboards exist and are good. SI has no hosted tier,
  no multi-user control plane, and no cost dashboards.
- **Multi-agent orchestration.** Crews, handoffs,
  sub-agents, and delegation are standard everywhere
  else. SI has none — deliberately: multi-agent systems
  are where attribution goes to die (see the survey
  figures above), and SI would rather expose one
  verifiable agent than operate a swarm it cannot
  attribute. Deliberate is not the same as present.
- **Benchmark presence.** SI has no SWE-bench, GAIA,
  LoCoMo, or LongMemEval numbers — contested as those
  are, they are the vocabulary buyers ask in. SI's only
  public score is its own conformance set, and its
  frozen-base and plain-filesystem baselines are recorded
  as **NOT RUN**, not estimated.
- **A2A.** The protocol is at v1.0 with 150+
  organisations [Index]. SI has no Agent Card and no A2A
  client or server; it is a later packaging item, not a
  shipped feature.
- **MCP depth.** SI's MCP server is v1: three tools, and
  submissions do not execute. There is no MCP client, so
  Gaia cannot yet use the thousands of existing servers
  that every framework above can consume as a client.
- **Proven sandboxing.** OpenHands and OpenAI ship
  sandboxes in production use. SI's Docker profile is
  test-pinned against faked runners and its live smoke
  is still owed; on a Docker-less host the default mode
  is host execution (disclosed, journal-stamped — see
  Execution isolation).
- **Onboarding polish.** `pip install` one-liners and
  GUI on-ramps (Ollama, LM Studio) set the bar. SI's
  path is a git clone, a venv, and a one-command demo —
  much better than it was, not yet a package-index
  install.

## What we do not claim

- **The export is not a signature.** The journal hash
  chain makes an export tamper-evident against its
  recorded tip; v1 has no signing keys, so anyone can
  fabricate a journal wholesale and export a
  self-consistent file from it. The chain also proves
  the record wasn't rewritten — not that the record was
  true (`docs/SI-122-verifiable-export.md` §4).
- **The eval is not a benchmark result.** It measures
  cycle-0 (first-cycle) perception conformance on SI's
  own 30-task set — no prior cycles or tool results
  exist in the corpus, so completion signals are not
  elicited. The grader is a separate OS process, not a
  separate container; container-grade grader isolation
  is not claimed. And the score is a **failure** by the
  project's own bar: 21/30, published because publishing
  failures is the point.
- **Executed-with-observed-effect is still unearned.**
  The escalation path has been exercised end-to-end to a
  journaled refusal (the safe outcome). A consequential
  action executed exactly once, with its real-world
  effect independently observed, has not yet been
  demonstrated on the live node, and this page does not
  pretend otherwise.
- **The machinery is not the model.** The current
  candidate is a small model that scores below its own
  bar. SI's claims concern the governance, journaling,
  and verification machinery around learning — not
  frontier model capability.
- **MCP submissions do not execute** in v1 (above), and
  **the sandbox has never run live** (above). Both are
  stated here so nobody has to dig for them.

## Maintenance

This page is a dated snapshot (2026-11-23) and its
competitor claims expire as the field moves. The update
path is: re-run the competitive research, re-check every
flagged figure against primary sources on the day of use,
and re-run `tests/test_si128_comparison.py`, which fails
if the SI anchors cited here (files, commands, and the
eval record in `docs/SI-123-live-run/results.json`) drift
from the tree. A claim that cannot be re-verified should
be removed, not repeated.
