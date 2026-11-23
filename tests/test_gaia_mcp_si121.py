"""SI-121 (MCP-1): the Gaia MCP server v1 — stdio, stdlib-only.

Coverage:

- ``initialize`` answers the handshake (serverInfo +
  protocolVersion) and ``ping`` answers empty;
- ``tools/list`` returns exactly the three v1 tools, each
  with an ``inputSchema`` object;
- ``gaia_escalations_list`` reads a fixture bundle's one
  journaled ESCALATEd intent as pending (and journals
  nothing doing it);
- ``gaia_submit_intent`` with a write-class tool under an
  escalate-write policy journals a self-edit whose verdict
  is ESCALATE, advances the journal tip, does NOT create
  the target file (shadow form: journaled, never executed),
  and the intent then shows as a pending escalation;
- an unknown tool name is refused before journaling (the
  journal tip is unchanged);
- an unknown JSON-RPC method returns error -32601, and a
  notification receives no response.

The server is driven in-process: ``GaiaMcpServer.handle``
directly, and ``run_stdio`` over ``io.StringIO``. No network,
no subprocess server.
"""
import io
import json
import os

import pytest

from si.bundle import (
    _anchor_checkpoint, _write_manifest, canonical_json,
    init_bundle, load_bundle, load_policy, sha256_hex)
from si.cli import build_parser, cmd_gaia_mcp
from si.gaia import mcp_server
from si.gaia.context import ToolContext
from si.gaia.corpus import corpus_policy
from si.gaia.drain import admit_tool_intent
from si.gaia.queue import record_queue_batch
from si.gaia.shadow import record_task_fixture
from si.gate import TOOL_SHADOW_EXECUTION, Gate, Intent
from si.journal import open_journal
from si.tiny_seed import seed_tiny_bundle


ESCALATE_WRITE = {"read": "admit", "write": "escalate",
                  "execute": "admit", "external": "escalate"}
CONTENT = "written after operator review\n"


@pytest.fixture()
def mini(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "mini.si"),
                             name="gaia-si121",
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    work = tmp_path / "work"
    work.mkdir()
    (work / "notes").mkdir()
    (work / "notes" / "plan.md").write_text("# plan\nstep one\n")
    return bundle_dir, str(work)


def _open(bundle_dir):
    manifest = load_bundle(bundle_dir)
    return manifest, open_journal(bundle_dir, manifest)


def _tip(bundle_dir):
    journal = open_journal(bundle_dir)
    try:
        return journal.tip()
    finally:
        journal.close()


def _call(server, name, arguments, req_id=1):
    """One tools/call through the JSON-RPC handler; returns
    the decoded tool payload."""
    response = server.handle({
        "jsonrpc": "2.0", "id": req_id, "method": "tools/call",
        "params": {"name": name, "arguments": arguments}})
    assert response is not None and "result" in response
    result = response["result"]
    assert result["content"][0]["type"] == "text"
    return result, json.loads(result["content"][0]["text"])


def _escalated(mini):
    """One ESCALATEd fs_write journaled the SI-113 way
    (queue batch + admit_tool_intent), manifest re-anchored.
    Returns (bundle_dir, edit_id, intent_seq)."""
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture = record_task_fixture(journal, "SI-121 scenario",
                                      work)
        policy = corpus_policy(
            work, {"tool_class_policy": ESCALATE_WRITE})
        gate = Gate(policy)
        ctx = ToolContext(tool_work_root=work,
                          tool_task_seq=fixture["seq"],
                          tool_live_execution=True)
        batch = record_queue_batch(journal, fixture["seq"],
                                   fixture["ref"], declared_size=1)
        intent = Intent(
            kind="tool", tool="fs_write",
            args={"path": "notes/new.md", "content": CONTENT},
            reason="Carry out the scripted SI-121 task step",
            parents=[fixture["ref"]],
            source_seqs=[fixture["seq"]])
        out = admit_tool_intent(journal, gate, ctx, intent,
                                batch_seq=batch["seq"])
        assert out["verdict"] == "ESCALATE"
        rows = journal.backend.find(
            "self_edits", filter={"edit_id": out["edit_id"]},
            limit=1)
        updated = _anchor_checkpoint(
            journal, manifest, manifest["head"],
            note="SI-121 test fixture: anchor after journaling "
                 "the escalation")
        _write_manifest(bundle_dir, updated)
        return bundle_dir, out["edit_id"], rows[0]["seq"]
    finally:
        journal.close()


def _escalate_write_bundle(mini):
    """Re-point the fixture bundle's own policy at
    escalate-write with fs_write opted in, the seed-time way
    (tiny_seed step 8): policy.json and the manifest's
    policy_hash are rewritten together, so load_bundle still
    verifies. Returns bundle_dir."""
    bundle_dir, _work = mini
    policy = load_policy(bundle_dir)
    policy["allowed_tools"] = ["fs_write"]
    policy["tool_class_policy"] = dict(ESCALATE_WRITE)
    policy_bytes = canonical_json(policy)
    with open(os.path.join(bundle_dir, "policy.json"),
              "wb") as f:
        f.write(policy_bytes)
    # load_bundle verifies the OLD hash, so read the manifest
    # raw here (the hash is exactly what is being updated).
    with open(os.path.join(bundle_dir, "manifest.json"),
              "rb") as f:
        manifest = json.loads(f.read().decode("utf-8"))
    manifest["constitution"] = dict(manifest["constitution"])
    manifest["constitution"]["policy_hash"] = \
        "sha256:" + sha256_hex(policy_bytes)
    _write_manifest(bundle_dir, manifest)
    # The tip checkpoint hashes the manifest, so re-anchor
    # after the rewrite (tiny_seed does its policy write
    # before its own re-anchor, step 8 before step 10).
    journal = open_journal(bundle_dir)
    try:
        updated = _anchor_checkpoint(
            journal, manifest, manifest["head"],
            note="SI-121 test fixture: re-anchor after the "
                 "seed-time policy rewrite")
    finally:
        journal.close()
    _write_manifest(bundle_dir, updated)
    load_bundle(bundle_dir)  # the rewritten bundle verifies
    return bundle_dir


# -- handshake / protocol ------------------------------------------------------

def test_initialize_handshake(mini):
    server = mcp_server.GaiaMcpServer(bundle=mini[0])
    response = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2024-11-05",
                   "capabilities": {},
                   "clientInfo": {"name": "test", "version": "0"}}})
    result = response["result"]
    assert result["protocolVersion"] == "2024-11-05"
    assert result["serverInfo"]["name"] == "gaia-mcp"
    assert "tools" in result["capabilities"]


def test_ping_and_notification_silence(mini):
    server = mcp_server.GaiaMcpServer(bundle=mini[0])
    response = server.handle({"jsonrpc": "2.0", "id": 2,
                              "method": "ping"})
    assert response["result"] == {}
    # A notification (no id) never receives a response.
    assert server.handle({"jsonrpc": "2.0",
                          "method": "notifications/initialized"}) \
        is None
    assert server.handle({"jsonrpc": "2.0", "method": "ping"}) \
        is None


def test_tools_list_exactly_three_with_schemas(mini):
    server = mcp_server.GaiaMcpServer(bundle=mini[0])
    response = server.handle({"jsonrpc": "2.0", "id": 3,
                              "method": "tools/list"})
    tools = response["result"]["tools"]
    assert [t["name"] for t in tools] == [
        "gaia_journal_query", "gaia_escalations_list",
        "gaia_submit_intent"]
    for tool in tools:
        assert isinstance(tool["inputSchema"], dict)
        assert tool["inputSchema"]["type"] == "object"
        assert tool["description"]


def test_unknown_method_is_minus_32601(mini):
    server = mcp_server.GaiaMcpServer(bundle=mini[0])
    response = server.handle({"jsonrpc": "2.0", "id": 4,
                              "method": "resources/list"})
    assert response["error"]["code"] == -32601


def test_run_stdio_round_trip(mini):
    bundle_dir, _work = mini
    lines = "\n".join([
        json.dumps({"jsonrpc": "2.0", "id": 1,
                    "method": "initialize", "params": {}}),
        json.dumps({"jsonrpc": "2.0",
                    "method": "notifications/initialized"}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"}),
        json.dumps({"jsonrpc": "2.0", "id": 3,
                    "method": "nope/nope"}),
        "this is not json",
    ]) + "\n"
    out = io.StringIO()
    rc = mcp_server.run_stdio(
        mcp_server.GaiaMcpServer(bundle=bundle_dir),
        instream=io.StringIO(lines), outstream=out)
    assert rc == 0
    responses = [json.loads(line)
                 for line in out.getvalue().splitlines()]
    # initialize, ping, unknown method, parse error — the
    # notification produced no line.
    assert [r["id"] for r in responses] == [1, 2, 3, None]
    assert responses[0]["result"]["serverInfo"]["name"] == \
        "gaia-mcp"
    assert responses[2]["error"]["code"] == -32601
    assert responses[3]["error"]["code"] == -32700


# -- read-only tools -------------------------------------------------------------

def test_escalations_list_reads_pending_and_journals_nothing(mini):
    bundle_dir, edit_id, seq = _escalated(mini)
    tip_before = _tip(bundle_dir)
    server = mcp_server.GaiaMcpServer(bundle=bundle_dir)
    result, payload = _call(server, "gaia_escalations_list",
                            {"bundle": bundle_dir})
    assert result["isError"] is False
    assert len(payload["pending"]) == 1
    view = payload["pending"][0]
    assert view["intent_seq"] == seq
    assert view["edit_id"] == edit_id
    assert view["tool"] == "fs_write"
    assert view["verdict"] == "ESCALATE"
    assert _tip(bundle_dir) == tip_before


def test_journal_query_reads_recent_entries(mini):
    bundle_dir, edit_id, seq = _escalated(mini)
    tip_before = _tip(bundle_dir)
    server = mcp_server.GaiaMcpServer(bundle=bundle_dir)
    result, payload = _call(server, "gaia_journal_query",
                            {"bundle": bundle_dir, "limit": 5})
    assert result["isError"] is False
    assert payload["tip"] == tip_before[0]
    assert payload["count"] == 5
    seqs = [e["seq"] for e in payload["entries"]]
    assert seqs == sorted(seqs, reverse=True)
    edit_entries = [e for e in payload["entries"]
                    if e.get("edit_id") == edit_id]
    assert edit_entries and \
        edit_entries[0]["collection"] == "self_edits"
    assert _tip(bundle_dir) == tip_before


# -- gaia_submit_intent: the shadow act ------------------------------------------

def test_submit_intent_escalates_journals_and_never_executes(mini):
    bundle_dir = _escalate_write_bundle(mini)
    tip_before = _tip(bundle_dir)
    server = mcp_server.GaiaMcpServer(bundle=bundle_dir)
    result, payload = _call(server, "gaia_submit_intent", {
        "bundle": bundle_dir, "tool": "fs_write",
        "args": {"path": "notes/new.md", "content": CONTENT},
        "reason": "SI-121 test: propose the scripted write",
        "operator": "si-121-test-operator"})
    assert result["isError"] is False
    assert payload["verdict"] == "ESCALATE"
    assert payload["effective_class"] == "write"
    assert payload["execution"]["execution"] == \
        TOOL_SHADOW_EXECUTION
    assert "NOT executed" in payload["note"]
    # The journal tip advanced and the self-edit is journaled
    # with the ESCALATE verdict.
    tip_after = _tip(bundle_dir)
    assert tip_after[0] > tip_before[0]
    journal = open_journal(bundle_dir)
    try:
        rows = journal.backend.find(
            "self_edits",
            filter={"edit_id": payload["edit_id"]}, limit=1)
    finally:
        journal.close()
    assert rows and rows[0]["gate_verdict"] == "ESCALATE"
    assert rows[0]["result"] == "escalated"
    # Shadow form: the target file was NOT created — nothing
    # executed, anywhere the intent could have written.
    assert not os.path.exists(
        os.path.join(bundle_dir, "notes", "new.md"))
    # And the intent now stands as a pending escalation.
    _result2, payload2 = _call(server, "gaia_escalations_list",
                               {"bundle": bundle_dir})
    assert [p["edit_id"] for p in payload2["pending"]] == \
        [payload["edit_id"]]


def test_submit_unknown_tool_refused_journals_nothing(mini):
    bundle_dir = _escalate_write_bundle(mini)
    tip_before = _tip(bundle_dir)
    server = mcp_server.GaiaMcpServer(bundle=bundle_dir)
    result, payload = _call(server, "gaia_submit_intent", {
        "bundle": bundle_dir, "tool": "fs_teleport",
        "args": {"path": "notes/new.md"},
        "reason": "SI-121 test: propose an unregistered tool"})
    assert result["isError"] is True
    assert "unknown tool" in payload["error"]
    assert _tip(bundle_dir) == tip_before


def test_submit_invalid_args_refused_journals_nothing(mini):
    bundle_dir = _escalate_write_bundle(mini)
    tip_before = _tip(bundle_dir)
    server = mcp_server.GaiaMcpServer(bundle=bundle_dir)
    result, payload = _call(server, "gaia_submit_intent", {
        "bundle": bundle_dir, "tool": "fs_write",
        "args": {"path": "notes/new.md"},  # content required
        "reason": "SI-121 test: propose a malformed write"})
    assert result["isError"] is True
    assert "invalid args" in payload["error"]
    assert _tip(bundle_dir) == tip_before


# -- the si CLI wiring -----------------------------------------------------------

def test_si_cli_wiring_parses_gaia_mcp(mini):
    bundle_dir, _work = mini
    args = build_parser().parse_args(["gaia-mcp", bundle_dir])
    assert args.func is cmd_gaia_mcp
    assert args.path == bundle_dir
    module_args = mcp_server.build_parser().parse_args(
        [bundle_dir])
    assert module_args.path == bundle_dir
