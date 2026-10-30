"""Gaia tool policy resolution (SI-41 RFC §6.2; slice SI-42 / B-1).

Keys are added to the bundle policy and resolved under the existing
fail-closed rules of the project (the transformer-policy precedent in
``si.gate``): a key **absent** falls back to the pinned RFC default;
a key **present but malformed** resolves to the *most restrictive*
value. Fail closed throughout — a broken policy never widens power.

Shipped defaults (RFC §6.2, verbatim):

============================  ==========
``allowed_tools``             ``[]`` — empty admits nothing
``tool_class_policy``         read/write/execute ``admit``,
                              external ``escalate``
``tool_work_root``            ``None`` — the bundle directory
``tool_max_result_bytes``     65536
``tool_timeout_seconds``      120
``max_tool_cycles_per_task``  16
``tool_max_task_result_bytes`` 1048576
============================  ==========
"""
from __future__ import annotations

from si.gaia.registry import TOOL_CLASSES

CLASS_RULES = ("admit", "escalate", "deny")

#: RFC §6.2 defaults, pinned (mirrors si.bundle.DEFAULT_POLICY's new
#: keys; used when a key is absent from an older bundle's policy).
TOOL_POLICY_DEFAULTS = {
    "allowed_tools": [],
    "tool_class_policy": {"read": "admit", "write": "admit",
                          "execute": "admit", "external": "escalate"},
    "tool_work_root": None,
    "tool_max_result_bytes": 65536,
    "tool_timeout_seconds": 120,
    "max_tool_cycles_per_task": 16,
    "tool_max_task_result_bytes": 1048576,
}


def _nonneg_int(policy: dict, key: str, default: int) -> int:
    if key not in policy:
        return default
    v = policy[key]
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        return 0  # present but malformed: most restrictive
    return v


def resolve_tool_policy(policy: dict | None) -> dict:
    """Resolve the §6.2 keys of a bundle policy dict, fail closed.

    Returns a fresh dict with exactly the seven §6.2 keys. Malformed
    ``tool_class_policy`` *entries* resolve to ``deny`` per class
    (RFC §6.2 / §16 criterion 5); a wholly malformed (non-dict)
    ``tool_class_policy`` denies every class. Malformed
    ``allowed_tools`` admits nothing. Malformed numeric caps resolve
    to 0. A malformed ``tool_work_root`` (not a non-empty string and
    not None) resolves to None — the bundle directory, the most
    restrictive scope the RFC defines.
    """
    p = policy if isinstance(policy, dict) else {}
    d = TOOL_POLICY_DEFAULTS

    allowed = p.get("allowed_tools", d["allowed_tools"])
    if not (isinstance(allowed, list)
            and all(isinstance(x, str) for x in allowed)):
        allowed = []

    raw_class = p.get("tool_class_policy", None)
    if "tool_class_policy" not in p:
        class_policy = dict(d["tool_class_policy"])
    elif not isinstance(raw_class, dict):
        class_policy = {c: "deny" for c in TOOL_CLASSES}
    else:
        class_policy = {}
        for cls in TOOL_CLASSES:
            if cls not in raw_class:
                class_policy[cls] = d["tool_class_policy"][cls]
            elif raw_class[cls] in CLASS_RULES:
                class_policy[cls] = raw_class[cls]
            else:
                class_policy[cls] = "deny"  # malformed entry -> deny

    root = p.get("tool_work_root", d["tool_work_root"])
    if root is not None and not (isinstance(root, str) and root):
        root = None

    return {
        "allowed_tools": list(allowed),
        "tool_class_policy": class_policy,
        "tool_work_root": root,
        "tool_max_result_bytes": _nonneg_int(
            p, "tool_max_result_bytes", d["tool_max_result_bytes"]),
        "tool_timeout_seconds": _nonneg_int(
            p, "tool_timeout_seconds", d["tool_timeout_seconds"]),
        "max_tool_cycles_per_task": _nonneg_int(
            p, "max_tool_cycles_per_task", d["max_tool_cycles_per_task"]),
        "tool_max_task_result_bytes": _nonneg_int(
            p, "tool_max_task_result_bytes",
            d["tool_max_task_result_bytes"]),
    }
