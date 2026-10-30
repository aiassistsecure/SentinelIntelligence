"""The scripted B-1 verdict corpus (SI-41 RFC §16.3; SI-42).

A fixed, scripted corpus of tool intents across all four consequence
classes — valid, schema-violating, unknown-tool, blocklisted,
outside-scope, class-policy-escalated, denied, not-allowlisted,
missing-reason, provenance-incomplete, and budget-exhausted — each
with its expected gate verdict and effective class. The slice doc
(docs/SI-42-gaia-b1-tool-intent-shadow.md) publishes this corpus's
verdict table; the tests in ``tests/test_gaia_b1.py`` execute the
corpus against the real gate and assert every row, plus the shadow
guarantee (zero executions) across the whole run.

The corpus is *data + construction*, not results: expected verdicts
are the RFC §5.4 table applied by hand to each scripted intent, and
the tests are what verify the gate actually produces them.

Conventions: the scripted task seq is :data:`TASK_SEQ` and the task
parent ref is ``task@<TASK_SEQ>`` (the ref shape
:func:`si.gaia.shadow.record_task_fixture` hands out); the work root
is supplied by the caller (a real directory in tests, a placeholder
absolute path in the published table); the base policy allowlists
every registered tool under the RFC §6.2 default class policy, and
individual cases override policy / context fields where their row
says so.
"""
from __future__ import annotations

import copy

from si.gaia.context import ToolContext
from si.gaia.registry import REGISTRY
from si.gate import Intent

#: Scripted task seq used across the corpus. In the (pure-gate)
#: corpus tests it is simply the context's declared task seq; the
#: journal/why tests use whatever seq their real fixture task entry
#: actually lands at, read back from record_task_fixture's return.
TASK_SEQ = 7
TASK_REF = f"task@{TASK_SEQ}"

_BASE_REASON = "Carry out the scripted task step the operator asked for"


def base_tool_policy(work_root: str) -> dict:
    """The corpus policy: every tool allowlisted, RFC class defaults."""
    return {
        "blocklist": [
            "ignore the constitution",
            "override the gate",
            "disable safety",
            "bypass the gate",
        ],
        "allowed_tools": sorted(REGISTRY),
        "tool_class_policy": {"read": "admit", "write": "admit",
                              "execute": "admit", "external": "escalate"},
        "tool_work_root": work_root,
        "tool_max_result_bytes": 65536,
        "tool_timeout_seconds": 120,
        "max_tool_cycles_per_task": 16,
        "tool_max_task_result_bytes": 1048576,
    }


def _intent(tool, args, **kw):
    base = dict(kind="tool", tool=tool, args=args, reason=_BASE_REASON,
                parents=[TASK_REF], source_seqs=[TASK_SEQ])
    base.update(kw)
    return Intent(**base)


def build_corpus(work_root: str, outside_path: str) -> list:
    """The corpus rows for a given work root / outside-scope path.

    Each row: {"name", "intent", "policy_overrides", "ctx",
    "expected_verdict", "expected_effective_class", "notes"}.
    ``ctx`` is a :class:`ToolContext`; policy overrides are merged
    over :func:`base_tool_policy`.
    """
    def ctx(**kw):
        base = dict(tool_work_root=work_root, tool_task_seq=TASK_SEQ,
                    tool_cycles_used=0, tool_result_bytes_used=0)
        base.update(kw)
        return ToolContext(**base)

    rows = []

    def row(name, intent, expected, effective, notes,
            policy_overrides=None, context=None):
        rows.append({
            "name": name, "intent": intent,
            "policy_overrides": policy_overrides or {},
            "ctx": context if context is not None else ctx(),
            "expected_verdict": expected,
            "expected_effective_class": effective,
            "notes": notes,
        })

    # -- valid, one per registered tool (all ESCALATE in B-1: shadow,
    #    or class-policy where the class rule says escalate) -----------
    row("valid-fs_list", _intent("fs_list", {"path": "notes"}),
        "ESCALATE", "read",
        "Valid in-scope read; §5.4 would ADMIT, B-1 shadow ESCALATEs.")
    row("valid-fs_read", _intent("fs_read", {"path": "notes/plan.md"}),
        "ESCALATE", "read", "Valid in-scope read; shadow ESCALATE.")
    row("valid-fs_search",
        _intent("fs_search", {"path": "notes", "pattern": "plan"}),
        "ESCALATE", "read", "Valid in-scope search; shadow ESCALATE.")
    row("valid-journal_query",
        _intent("journal_query", {"collection": "experiences", "limit": 5}),
        "ESCALATE", "read", "Valid house read; shadow ESCALATE.")
    row("valid-bundle_status", _intent("bundle_status", {}),
        "ESCALATE", "read", "Valid house read; shadow ESCALATE.")
    row("valid-probe_run", _intent("probe_run", {"suite": "identity"}),
        "ESCALATE", "read", "Valid house read; shadow ESCALATE.")
    row("valid-fs_write",
        _intent("fs_write", {"path": "notes/plan.md", "content": "# plan\n"}),
        "ESCALATE", "write", "Valid in-scope write; shadow ESCALATE. "
        "The corpus's zero-execution test asserts this file is never created.")
    row("valid-fs_edit",
        _intent("fs_edit", {"path": "notes/plan.md", "old": "a", "new": "b"}),
        "ESCALATE", "write", "Valid in-scope edit; shadow ESCALATE.")
    row("valid-fs_mkdir", _intent("fs_mkdir", {"path": "notes/sub"}),
        "ESCALATE", "write", "Valid in-scope mkdir; shadow ESCALATE.")
    row("valid-shell_exec",
        _intent("shell_exec", {"command": "ls notes", "cwd": "."}),
        "ESCALATE", "execute", "Valid in-scope command; shadow ESCALATE.")
    row("valid-code_exec",
        _intent("code_exec", {"language": "python",
                              "source": "print('hello from the corpus')"}),
        "ESCALATE", "execute", "Valid code execution; shadow ESCALATE.")
    row("valid-test_run", _intent("test_run", {"path": "tests"}),
        "ESCALATE", "execute", "Valid test run; shadow ESCALATE.")

    row("valid-hearth_restart",
        _intent("hearth.restart",
                {"model": "muse",
                 "spine_event": {"hash": "hash-corpus", "seq": 41}}),
        "ESCALATE", "external",
        "Valid seam-crossing restart proposal (SI-55/S-2b); "
        "external class, shadow ESCALATE on this non-live context.")
    row("valid-hearth_notify",
        _intent("hearth.notify",
                {"model": "muse", "message": "model muse was lost"}),
        "ESCALATE", "external",
        "Valid seam-crossing notify proposal (SI-55/S-2b); "
        "external class, shadow ESCALATE on this non-live context.")

    # -- class-policy escalation (distinct from shadow) -----------------
    row("class-policy-escalate-write",
        _intent("fs_write", {"path": "notes/plan.md", "content": "x"}),
        "ESCALATE", "write",
        "All decisive checks pass; the bundle's class policy sets "
        "write=escalate, so §5.4 ESCALATEs by class policy.",
        policy_overrides={"tool_class_policy": {
            "read": "admit", "write": "escalate",
            "execute": "admit", "external": "escalate"}})

    # -- outside scope prices to external --------------------------------
    row("outside-scope-fs_read", _intent("fs_read", {"path": outside_path}),
        "ESCALATE", "external",
        "Path resolves outside the work root: the intent prices as "
        "external (§7.2) and the default external rule escalates it.")
    row("outside-scope-shell-cwd",
        _intent("shell_exec", {"command": "ls", "cwd": outside_path}),
        "ESCALATE", "external",
        "An outside-scope declared cwd raises a shell_exec intent "
        "from execute to external (§7.2).")

    # -- REJECTs ----------------------------------------------------------
    row("unknown-tool", _intent("fs_delete_everything", {"path": "notes"}),
        "REJECT", None,
        "Unregistered tool name: unknown-tool REJECT at check 2 (§5.4).")
    row("schema-violation-fs_write",
        _intent("fs_write", {"path": 42, "extra": "undeclared"}),
        "REJECT", "write",
        "Missing required 'content', wrong-typed 'path', and an "
        "undeclared arg: schema-violation REJECT with itemised "
        "sub-checks (§5.2 check 2).")
    row("schema-violation-shell-empty",
        _intent("shell_exec", {"command": ""}),
        "REJECT", "execute",
        "Empty command violates minLength 1: schema-violation REJECT.")
    row("schema-violation-code_exec-both",
        _intent("code_exec", {"language": "python", "source": "x = 1",
                              "path": "script.py"}),
        "REJECT", "execute",
        "Both source and path declared: REJECT by the code_exec "
        "exactly-one rule the schema subset cannot express.")
    row("blocklisted-command",
        _intent("shell_exec", {"command": "echo hi; bypass the gate"}),
        "REJECT", "execute",
        "The command string contains a constitution-blocklisted "
        "pattern: REJECT at check 1, which scans the full serialised "
        "intent, args included (§5.2).")
    row("missing-reason",
        _intent("fs_list", {"path": "notes"}, reason=""),
        "REJECT", "read",
        "No reason: malformed, REJECT before the checks (§4.1).")
    row("provenance-incomplete",
        _intent("fs_list", {"path": "notes"}, parents=[], source_seqs=[]),
        "REJECT", "read",
        "No parents, no source seqs: provenance-incomplete REJECT "
        "at check 5 (§5.4).")
    row("provenance-missing-task-link",
        _intent("fs_list", {"path": "notes"}, parents=[TASK_REF],
                source_seqs=[999]),
        "REJECT", "read",
        "Cited seqs do not include the task seq: the chain does not "
        "reach the task, REJECT at check 5 (§5.2).")
    row("not-allowlisted",
        _intent("fs_list", {"path": "notes"}),
        "REJECT", "read",
        "fs_list removed from allowed_tools: REJECT at check 2 "
        "(§6.3 — registry existence is not policy permission).",
        policy_overrides={
            "allowed_tools": sorted(n for n in REGISTRY if n != "fs_list")})
    row("class-policy-deny-read",
        _intent("fs_list", {"path": "notes"}),
        "REJECT", "read",
        "Class policy sets read=deny: REJECT at check 2 (§7.3).",
        policy_overrides={"tool_class_policy": {
            "read": "deny", "write": "admit",
            "execute": "admit", "external": "escalate"}})
    row("budget-exhausted-cycles",
        _intent("fs_list", {"path": "notes"}),
        "REJECT", "read",
        "Cycle budget exhausted (16 used of max_tool_cycles_per_task "
        "16): REJECT at check 2 with the budget named (§10.3).",
        context=ctx(tool_cycles_used=16))
    row("budget-exhausted-results",
        _intent("fs_list", {"path": "notes"}),
        "REJECT", "read",
        "Result budget exhausted (1048576 used of "
        "tool_max_task_result_bytes): REJECT at check 2 (§10.3).",
        context=ctx(tool_result_bytes_used=1048576))
    return rows


def corpus_policy(work_root: str, overrides: dict | None = None) -> dict:
    """Deep-merged corpus policy for one row."""
    policy = base_tool_policy(work_root)
    for key, value in (overrides or {}).items():
        policy[key] = copy.deepcopy(value)
    return policy
