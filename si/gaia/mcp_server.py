"""Gaia MCP-1: a minimal MCP server for Gaia bundles (SI-121).

A Model Context Protocol (MCP) JSON-RPC 2.0 server over stdio,
Python standard library only — no new dependencies. It exposes
exactly three tools over one bundle:

- ``gaia_journal_query`` — **read-only**: the most recent
  journal entries of the bundle. Opens the journal the way
  :func:`si.gaia.escalation_cli._open_read` does (directly,
  hash chain verified, fail closed) and journals nothing.
- ``gaia_escalations_list`` — **read-only**: the pending
  escalations, delegated to
  :func:`si.gaia.escalation_cli.list_escalations`. Journals
  nothing.
- ``gaia_submit_intent`` — the ONE state-changing tool, in
  **ESCALATE-ONLY / shadow form**. It validates the tool name
  and args against :mod:`si.gaia.registry` (an unknown tool or
  invalid args are refused *before* anything is journaled),
  builds a ``kind="tool"`` :class:`si.gate.Intent`, and calls
  the existing B-1 shadow act,
  :func:`si.gaia.shadow.journal_tool_intent`: the real gate
  verdict is journaled (``record_self_edit`` with the full
  gate report) *before* any outcome, and the act **never
  executes** — it has no executor, by design. The manifest
  checkpoint is then re-anchored exactly as
  :func:`si.gaia.executor.run_tool_intent`'s tail does, so
  the bundle is left fully verified for its next opener.

Honest scope label, stated everywhere this tool is named: an
ADMIT verdict returned by ``gaia_submit_intent`` means
"priced ADMIT, journaled, **NOT executed** — execution
remains with the existing task/service paths". MCP v1 wires
the gate and the journal, not execution. Nothing here fakes
an execution or an admission.

Transport: newline-delimited JSON on stdin/stdout, one
JSON-RPC 2.0 message per line. Supported methods:
``initialize``, ``ping``, ``tools/list``, ``tools/call``.
Notifications (requests with no ``id``) receive no response.
Unknown methods fail with the JSON-RPC error ``-32601``
(method not found). Tool results use the standard MCP shape
``{content: [{type: "text", text: <json string>}], isError}``;
protocol-level failures are JSON-RPC errors.

Usage (also wired into the main CLI as ``si gaia-mcp``)::

    si gaia-mcp <bundle>

which runs the stdio server in the foreground, serving the
given bundle by default (each tool call may name another
bundle explicitly). CPU only; no network; no secrets.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

#: MCP protocol version this server speaks.
PROTOCOL_VERSION: str = "2024-11-05"

#: Server identity reported by the ``initialize`` handshake.
SERVER_INFO: dict = {"name": "gaia-mcp", "version": "1.0.0"}

#: The honest scope label for the one state-changing tool.
SUBMIT_SCOPE_LABEL: str = (
    "ESCALATE-ONLY / shadow form (SI-121 MCP-1): the gate "
    "verdict is real and journaled before any outcome, and "
    "nothing is executed — an ADMIT verdict here means "
    "priced ADMIT, journaled, NOT executed; execution "
    "remains with the existing task/service paths")

#: JSON-RPC 2.0 error codes used by this server.
ERR_PARSE = -32700
ERR_INVALID_REQUEST = -32600
ERR_METHOD_NOT_FOUND = -32601
ERR_INVALID_PARAMS = -32602

_DEFAULT_JOURNAL_LIMIT = 20
_MAX_JOURNAL_LIMIT = 200


# ---------------------------------------------------------------------------
# Tool definitions (exactly three in v1)
# ---------------------------------------------------------------------------

TOOLS: list = [
    {
        "name": "gaia_journal_query",
        "description": (
            "Read-only: return the most recent journal entries "
            "of a Gaia bundle (seq, collection, kind, record "
            "summary). The journal's hash chain is verified "
            "before anything is read (fail closed). This tool "
            "journals nothing."),
        "inputSchema": {
            "type": "object",
            "required": ["bundle"],
            "properties": {
                "bundle": {"type": "string",
                           "description": "bundle directory"},
                "limit": {"type": "integer", "minimum": 1,
                          "description": "maximum entries to "
                                         "return, most recent "
                                         "first (default 20)"},
            },
        },
    },
    {
        "name": "gaia_escalations_list",
        "description": (
            "Read-only: list a Gaia bundle's pending "
            "escalations (folded escalated, unresolved), via "
            "si.gaia.escalation_cli.list_escalations. This "
            "tool journals nothing."),
        "inputSchema": {
            "type": "object",
            "required": ["bundle"],
            "properties": {
                "bundle": {"type": "string",
                           "description": "bundle directory"},
                "task": {"type": "integer",
                         "description": "scope the fold to "
                                        "one task seq"},
            },
        },
    },
    {
        "name": "gaia_submit_intent",
        "description": (
            "Submit one Gaia tool intent to the gate. "
            + SUBMIT_SCOPE_LABEL + ". The tool name is "
            "validated against si.gaia.registry and the args "
            "against the tool's registered schema; an unknown "
            "tool or invalid args are refused before anything "
            "is journaled. Returns {verdict, edit_id, "
            "effective_class, execution}."),
        "inputSchema": {
            "type": "object",
            "required": ["bundle", "tool", "args", "reason"],
            "properties": {
                "bundle": {"type": "string",
                           "description": "bundle directory"},
                "tool": {"type": "string",
                         "description": "registered Gaia tool "
                                        "name (see "
                                        "si.gaia.registry)"},
                "args": {"type": "object",
                         "description": "the tool's declared "
                                        "args object"},
                "reason": {"type": "string",
                           "description": "why this intent is "
                                          "proposed (required, "
                                          "non-empty)"},
                "operator": {"type": "string",
                             "description": "operator name "
                                            "recorded on the "
                                            "result (optional)"},
            },
        },
    },
]

_TOOL_NAMES = {tool["name"] for tool in TOOLS}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _tool_result(payload, is_error: bool = False) -> dict:
    """One MCP tool result: JSON text content + the error flag."""
    return {
        "content": [{"type": "text",
                     "text": json.dumps(payload, default=str)}],
        "isError": is_error,
    }


def _tool_error(message: str, **extra) -> dict:
    return _tool_result({"error": message, **extra}, is_error=True)


def _record_summary(collection: str, row: dict) -> dict:
    """A compact, JSON-safe summary of one journal row: the
    fields a reader decides on, never an invented value."""
    from si.gaia.diffview import _kind, _record
    record = _record(row)
    summary = {"seq": row.get("seq"), "collection": collection,
               "kind": _kind(row)}
    for key in ("edit_id", "gate_verdict", "result"):
        if row.get(key) is not None:
            summary[key] = row[key]
    intent = row.get("intent") or {}
    if isinstance(intent, dict) and intent.get("tool"):
        summary["tool"] = intent["tool"]
    for key in ("kind", "tool", "resolution", "operator",
                "source"):
        if record.get(key) is not None and key not in summary:
            summary[key] = record[key]
    return summary


def journal_query(bundle_dir: str, limit=None) -> dict:
    """The read-only journal read behind ``gaia_journal_query``:
    open the bundle's journal read-only the escalation CLI's
    way (chain verified, fail closed), return the most recent
    entries. Journals nothing."""
    from si.gaia.diffview import _window_rows
    from si.gaia.escalation_cli import _open_read
    if limit is None:
        limit = _DEFAULT_JOURNAL_LIMIT
    if not isinstance(limit, int) or isinstance(limit, bool) \
            or limit < 1:
        raise ValueError("limit must be a positive integer")
    limit = min(limit, _MAX_JOURNAL_LIMIT)
    journal = _open_read(bundle_dir)
    try:
        tip_seq, tip_hash = journal.tip()
        rows = _window_rows(journal, 0, tip_seq)
        entries = [_record_summary(coll, row)
                   for _seq, coll, row in rows[-limit:]]
        entries.reverse()
        return {"bundle": bundle_dir, "tip": tip_seq,
                "tip_hash": tip_hash, "count": len(entries),
                "entries": entries}
    finally:
        journal.close()


def submit_intent(bundle_dir: str, tool: str, args, reason: str,
                  operator: str | None = None) -> dict:
    """The one state-changing act behind ``gaia_submit_intent``
    — ESCALATE-ONLY / shadow form (see the module docstring).

    Validation order is the guarantee: the tool name and args
    are checked against the registry *before* the bundle is
    opened, so a refused submission journals nothing. A valid
    submission is gated and journaled by the existing B-1
    shadow act (:func:`si.gaia.shadow.journal_tool_intent`),
    which never executes, and the manifest checkpoint is
    re-anchored exactly as ``run_tool_intent``'s tail does.

    Raises ValueError for a refused submission (unknown tool,
    invalid args, empty reason); bundle verification failures
    propagate from ``load_bundle`` (fail closed).
    """
    from si.gaia.registry import get_tool, validate_args
    spec = get_tool(tool)
    if spec is None:
        raise ValueError(
            f"unknown tool {tool!r}: not in si.gaia.registry — "
            f"refused before journaling")
    violations = validate_args(spec, args)
    if violations:
        raise ValueError(
            "invalid args for tool "
            f"{tool!r}: " + "; ".join(violations)
            + " — refused before journaling")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(
            "reason must be a non-empty string — refused "
            "before journaling")

    from si.bundle import (
        _anchor_checkpoint, _write_manifest, load_bundle,
        load_policy)
    from si.gaia.context import ToolContext
    from si.gaia.shadow import journal_tool_intent
    from si.gate import Gate, Intent
    from si.journal import open_journal

    bundle_dir = os.path.abspath(bundle_dir)
    manifest = load_bundle(bundle_dir)
    gate = Gate(load_policy(bundle_dir))
    journal = open_journal(bundle_dir, manifest)
    try:
        # Non-live context, exactly as run_tool_intent builds
        # for a non-miniature bundle: shadow pricing, the
        # bundle directory as the scope root.
        ctx = ToolContext(bundle_dir=bundle_dir)
        # Provenance (gate check 5, RFC §5.2): an MCP
        # submission is an out-of-task proposal, so it cites
        # its real causes — the journal tip it was priced
        # against as its source seq, and this submission
        # surface as its parent. Nothing is invented: with an
        # empty journal there is no tip to cite, the intent
        # carries no source seq, and the gate's REJECT for an
        # uncaused action is the honest, journaled outcome.
        tip_seq, _tip_hash = journal.tip()
        intent = Intent(
            kind="tool", tool=tool, args=dict(args),
            reason=reason, parents=["mcp:gaia_submit_intent"],
            source_seqs=[tip_seq] if tip_seq >= 1 else [])
        outcome = journal_tool_intent(journal, gate, intent, ctx)
        updated = _anchor_checkpoint(
            journal, manifest, manifest["head"],
            note=f"gaia MCP submit {outcome['edit_id']} "
                 f"{outcome['verdict'].lower()} (SI-121/MCP-1 "
                 f"shadow form: journaled, not executed)")
        _write_manifest(bundle_dir, updated)
    finally:
        journal.close()
    return {
        "verdict": outcome["verdict"],
        "edit_id": outcome["edit_id"],
        "effective_class": outcome["effective_class"],
        "execution": outcome["execution"],
        "operator": operator,
        "scope": SUBMIT_SCOPE_LABEL,
        "note": "journaled, NOT executed: an ADMIT verdict "
                "here means priced ADMIT and journaled only — "
                "execution remains with the existing "
                "task/service paths",
    }


# ---------------------------------------------------------------------------
# The JSON-RPC server
# ---------------------------------------------------------------------------

class GaiaMcpServer:
    """The MCP-1 server state: one default bundle (the CLI's
    positional path); each tool call may name another bundle
    explicitly in its arguments."""

    def __init__(self, bundle: str | None = None):
        self.bundle = bundle

    # -- tool dispatch --------------------------------------------------

    def call_tool(self, name: str, arguments) -> dict:
        """Dispatch one ``tools/call`` to its implementation.
        Every failure is an MCP tool error result (isError),
        never a raised exception and never a fake success."""
        if name not in _TOOL_NAMES:
            return _tool_error(f"unknown tool {name!r}")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return _tool_error("arguments must be an object")
        bundle = arguments.get("bundle") or self.bundle
        if not bundle:
            return _tool_error(
                "no bundle: pass the 'bundle' argument or "
                "start the server with a bundle path")
        try:
            if name == "gaia_journal_query":
                return _tool_result(journal_query(
                    bundle, limit=arguments.get("limit")))
            if name == "gaia_escalations_list":
                from si.gaia.escalation_cli import (
                    list_escalations)
                return _tool_result({
                    "bundle": bundle,
                    "pending": list_escalations(
                        bundle, task_seq=arguments.get("task")),
                })
            if name == "gaia_submit_intent":
                return _tool_result(submit_intent(
                    bundle,
                    arguments.get("tool"),
                    arguments.get("args"),
                    arguments.get("reason"),
                    operator=arguments.get("operator")))
        except Exception as e:  # refused act / unverified bundle
            return _tool_error(f"{type(e).__name__}: {e}")
        return _tool_error(f"unhandled tool {name!r}")

    # -- JSON-RPC --------------------------------------------------------

    def handle(self, message) -> dict | None:
        """Handle one decoded JSON-RPC message. Returns the
        response object, or None for a notification (a message
        with no ``id`` never receives a response)."""
        if not isinstance(message, dict):
            return _rpc_error(None, ERR_INVALID_REQUEST,
                              "request must be a JSON object")
        is_notification = "id" not in message
        request_id = message.get("id")
        method = message.get("method")
        if not isinstance(method, str) or not method:
            if is_notification:
                return None
            return _rpc_error(request_id, ERR_INVALID_REQUEST,
                              "missing or invalid 'method'")
        if method.startswith("notifications/"):
            return None
        if method == "initialize":
            result = {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": dict(SERVER_INFO),
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            params = message.get("params") or {}
            if not isinstance(params, dict) \
                    or not isinstance(params.get("name"), str):
                if is_notification:
                    return None
                return _rpc_error(
                    request_id, ERR_INVALID_PARAMS,
                    "tools/call requires params.name (string)")
            result = self.call_tool(params["name"],
                                    params.get("arguments"))
        else:
            if is_notification:
                return None
            return _rpc_error(request_id, ERR_METHOD_NOT_FOUND,
                              f"method not found: {method}")
        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": request_id,
                "result": result}


def _rpc_error(request_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id,
            "error": {"code": code, "message": message}}


def run_stdio(server: GaiaMcpServer, instream=None,
              outstream=None) -> int:
    """Serve newline-delimited JSON-RPC over stdio until EOF.
    Malformed lines answer with a parse error (id null) and
    the server keeps serving. Returns 0 at EOF."""
    instream = instream if instream is not None else sys.stdin
    outstream = outstream if outstream is not None else sys.stdout
    for line in instream:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as e:
            response = _rpc_error(None, ERR_PARSE,
                                  f"parse error: {e}")
            outstream.write(json.dumps(response) + "\n")
            outstream.flush()
            continue
        response = server.handle(message)
        if response is not None:
            outstream.write(json.dumps(response) + "\n")
            outstream.flush()
    return 0


# ---------------------------------------------------------------------------
# CLI front-ends (module main + the si.cli wiring share `execute`)
# ---------------------------------------------------------------------------

def execute(args) -> int:
    """Run the stdio server in the foreground over the parsed
    arguments' bundle. Returns the process exit code (0 at
    stdin EOF)."""
    server = GaiaMcpServer(bundle=args.path)
    return run_stdio(server)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="si.gaia.mcp_server",
        description="SI-121/MCP-1 Gaia MCP server: a minimal "
                    "MCP JSON-RPC 2.0 server over stdio "
                    "(stdlib only) exposing gaia_journal_query "
                    "and gaia_escalations_list (read-only, "
                    "journal nothing) and gaia_submit_intent "
                    "(ESCALATE-ONLY shadow form: real gate "
                    "verdict journaled, never executed).")
    parser.add_argument("path", help="bundle directory (the "
                                     "default bundle served)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return execute(args)
    except Exception as e:  # fail closed, loud
        print(f"si: FAILED: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
