"""Autonomous multi-cycle self-write loop: `si run`.

Strictly serial: one cycle at a time, through the same gate, halted by
the first REJECT or ESCALATE. Rejections and escalations are already
journaled by ``run_cycle`` (see si/loop.py) — this module never
double-journals; it just counts and stops.

This is the SI-1 autonomy primitive: ``si run --cycles N`` hands the lead
to the mind for N self-write cycles and reports what happened.
"""
from __future__ import annotations

import os

from si.bundle import load_bundle, load_policy
from si.gate import Gate
from si.loop import run_cycle


def run_cycles(bundle_path: str, mind, cycles: int, gate=None) -> dict:
    """Autonomous serial multi-cycle self-write run.

    Parameters
    ----------
    bundle_path: the .si bundle directory.
    mind: a Mind (perceive/propose/train_delta/forward/probes).
    cycles: number of serial cycles requested.
    gate: an optional shared Gate; one is built from the bundle's policy
        and reused across all cycles if not given.

    The same Gate instance is reused for every cycle so gate-side state
    (none today, but the protocol is shared-instance) is consistent.

    Halts immediately on the first REJECT or ESCALATE — the rejection /
    escalation is already journaled by ``run_cycle``, so no journaling
    happens here, just counting and halting.

    Returns a summary dict with cycles requested/completed, per-verdict
    counts, halt info, the final head, the delta stack depth, and every
    cycle's self-edit id in order.
    """
    bundle_dir = os.path.abspath(bundle_path)
    gate = gate or Gate(load_policy(bundle_dir))

    admitted = 0
    rejected = 0
    escalated = 0
    edit_ids: list[str] = []
    halted = False
    halt_verdict = None
    halt_edit_id = None
    cycles_completed = 0

    for _ in range(cycles):
        res = run_cycle(bundle_dir, mind, gate=gate)
        cycles_completed += 1
        verdict = res["verdict"]
        edit_ids.append(res["edit_id"])
        if verdict == "ADMIT":
            admitted += 1
        elif verdict == "REJECT":
            rejected += 1
            halted = True
            halt_verdict = verdict
            halt_edit_id = res["edit_id"]
            break
        elif verdict == "ESCALATE":
            escalated += 1
            halted = True
            halt_verdict = verdict
            halt_edit_id = res["edit_id"]
            break
        else:  # pragma: no cover - run_cycle only produces the three verdicts
            raise ValueError(f"unknown cycle verdict {verdict!r}")

    manifest = load_bundle(bundle_dir)
    head = manifest["head"]
    if head == "base":
        delta_stack_depth = 0
    else:
        # Deltas are appended in stack order; count up to and including head.
        delta_stack_depth = 0
        for d in manifest["deltas"]:
            delta_stack_depth += 1
            if d["id"] == head:
                break

    return {
        "cycles_requested": cycles,
        "cycles_completed": cycles_completed,
        "admitted": admitted,
        "rejected": rejected,
        "escalated": escalated,
        "halted": halted,
        "halt_verdict": halt_verdict,
        "halt_edit_id": halt_edit_id,
        "head": head,
        "delta_stack_depth": delta_stack_depth,
        "edit_ids": edit_ids,
    }
