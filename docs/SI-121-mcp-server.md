# SI-121 — MCP-1: Gaia MCP server v1 (stdio, stdlib-only)

**Status: BUILT.** A minimal Model Context Protocol (MCP)
server for Gaia bundles: JSON-RPC 2.0 over stdio, Python
standard library only, no new dependencies. Implementation:
`si/gaia/mcp_server.py`, wired into the main CLI as
`si gaia-mcp <bundle>`. Tests:
`tests/test_gaia_mcp_si121.py`.

| Element | SI-121 state |
| --- | --- |
| MCP server (stdio, stdlib-only) | **built** — `si/gaia/mcp_server.py`; methods `initialize`, `ping`, `tools/list`, `tools/call`; notifications answered with silence; unknown methods return JSON-RPC error `-32601` |
| Read-only tools | **built** — `gaia_journal_query`, `gaia_escalations_list`; both journal nothing |
| Submission tool | **built, ESCALATE-ONLY / shadow form** — `gaia_submit_intent` journals the real gate verdict through the existing B-1 shadow act (`si.gaia.shadow.journal_tool_intent`) and **never executes**; execution is NOT wired through MCP in v1 |
| CLI wiring | **built** — `si gaia-mcp <bundle>` runs the stdio server in the foreground |

---

## 1. What it exposes

Exactly three tools in v1. Every tool takes a `bundle`
argument (the bundle directory); when a call omits it, the
bundle the server was started with is served.

### `gaia_journal_query` — read-only

Arguments: `{bundle, limit?}`. Returns the most recent
journal entries (`seq`, `collection`, `kind`, and a record
summary) as JSON text content, most recent first, plus the
journal tip. The journal is opened read-only the way
`si.gaia.escalation_cli._open_read` opens it: directly, with
the hash chain verified before anything is read — fail
closed on an unverified chain. **This tool journals
nothing.**

### `gaia_escalations_list` — read-only

Arguments: `{bundle, task?}`. Returns the bundle's pending
escalations (folded `escalated`, unresolved), delegated to
`si.gaia.escalation_cli.list_escalations`. **This tool
journals nothing.**

### `gaia_submit_intent` — the one state-changing tool, shadow form

Arguments: `{bundle, tool, args, reason, operator?}`.

**Honest scope label: submission through MCP v1 is the
gate+journal shadow form. Verdicts are real; execution is
NOT wired through MCP in v1.**

- The tool name is validated against `si.gaia.registry`
  (`get_tool`) and the args against the tool's registered
  schema (`validate_args`). An unknown tool or invalid args
  are **refused before journaling** — the call returns an
  MCP tool error result and the journal tip does not move.
  An empty `reason` is refused the same way.
- A valid submission composes exactly like
  `si.gaia.executor.run_tool_intent`'s opening: `load_bundle`
  (fail closed), `load_policy`, `Gate(policy)`,
  `open_journal`, and a non-live `si.gaia.context.ToolContext`.
  The intent is a `kind="tool"` `si.gate.Intent`, an
  out-of-task proposal citing its real causes: the journal
  tip it was priced against as its source seq, and the MCP
  submission surface (`mcp:gaia_submit_intent`) as its
  parent.
- The act called is the existing B-1 shadow act,
  `si.gaia.shadow.journal_tool_intent`: the real gate
  verdict is journaled (`record_self_edit` with the full
  gate report) *before* any outcome, and the act **never
  executes** — it has no executor, by design. The manifest
  checkpoint is then re-anchored exactly as
  `run_tool_intent`'s tail does, so the bundle is left fully
  verified for its next opener.
- The result is `{verdict, edit_id, effective_class,
  execution}` plus the scope label. **An ADMIT verdict in
  this shadow form means "priced ADMIT, journaled, NOT
  executed — execution remains with the existing
  task/service paths."** Nothing here fakes an execution or
  an admission; the journaled execution record carries the
  standing shadow label (`not-evaluated — shadow mode`).

## 2. Protocol and transport

- Transport: newline-delimited JSON on stdin/stdout, one
  JSON-RPC 2.0 message per line, served until stdin EOF.
- `initialize` returns `serverInfo` (`gaia-mcp`) and
  `protocolVersion` `2024-11-05`; `ping` returns `{}`.
- Notifications (messages with no `id`) receive no response.
- Unknown methods return JSON-RPC error `-32601` (method
  not found); malformed lines return `-32700` (parse error)
  and the server keeps serving.
- Tool results use the standard MCP shape:
  `{content: [{type: "text", text: <json string>}], isError}`.
  Protocol-level failures are JSON-RPC errors; tool-level
  failures (refusals, unverified bundles) are tool results
  with `isError: true`.

## 3. Running it

```sh
si gaia-mcp <bundle-path>
```

(`python -m si.gaia.mcp_server <bundle-path>` is the same
server through the module's own `main()`.)

### MCP host configuration

Claude Code — add to `.mcp.json` in the project root (or
`~/.claude.json`):

```json
{
  "mcpServers": {
    "gaia": {
      "command": "si",
      "args": ["gaia-mcp", "/path/to/bundle.si"]
    }
  }
}
```

Cursor — add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "gaia": {
      "command": "si",
      "args": ["gaia-mcp", "/path/to/bundle.si"]
    }
  }
}
```

Codex — add to `~/.codex/config.toml`:

```toml
[mcp_servers.gaia]
command = "si"
args = ["gaia-mcp", "/path/to/bundle.si"]
```

## 4. Scope, stated plainly

- Two of the three tools are read-only and journal nothing;
  the third journals a gate verdict and nothing else.
- No execution path is reachable through MCP v1: there is
  no executor in the shadow act, no handler in this module,
  and no tool in the registry gains one here.
- CPU only; no network (stdio only); no secrets; no new
  dependencies — the server is Python standard library plus
  the repo's existing Gaia machinery.
