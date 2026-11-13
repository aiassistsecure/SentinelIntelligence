"""Gaia ESC-LIVE-1: the CLI approval trigger for escalated
intents (SI-111 §4.2; slice SI-113).

The trigger is **transport, never decision**. The SSE spine
surfaces escalations as journal events; this command is the
return path — an operator-facing CLI that

- **lists / reads** the pending escalations folded from a
  bundle's journal (:func:`si.gaia.queue.fold_queue` — state
  is folded, never stored, so what the operator sees is the
  journal's own account of itself), and
- **records a resolution** by calling the existing verifying
  act, :func:`si.gaia.resolution.resolve_escalation`, and
  nothing else. There is no second write path here: this
  module never journals a resolution entry itself, never
  rewrites a verdict, and adds no verification of its own —
  the closed vocabulary, the folded-state check, the
  single-resolution rule, and the non-empty-operator rule all
  stay inside ``resolve_escalation``, exactly as merged
  (PR #99). A trigger call that fails verification journals
  nothing, because the act it calls journals nothing.

No policy key makes a resolution automatic through this
trigger: no timer, no allowlist shortcut, no auto-admit, and
no webhook (D-111-4: CLI only — an unauthenticated webhook
that can journal an operator act is an approval channel
anyone on the network could ring, and no auth story exists
yet, SI-41 §19). A resolution is an explicit operator act
typed at a terminal, or it does not exist.

Usage (also wired into the main CLI as ``si
gaia-escalation``)::

    si gaia-escalation <bundle> list [--task SEQ] [--json]
    si gaia-escalation <bundle> show <intent_seq> [--json]
    si gaia-escalation <bundle> resolve <intent_seq> \
        --resolution admit-after-review --operator NAME \
        [--basis TEXT] [--detail TEXT] [--task SEQ]

Honest scope: the trigger operates on a bundle's journal
directly (the live node's seam journal is the bundle served
by ``si-seam-service``). Reads open the journal directly and
verify its hash chain before folding (the SI-58 lesson: a
live journal's tip legitimately advances past the manifest
anchor between re-anchors, so a read must not demand
anchor == tip — but it must not fold an unverified chain
either). The resolution write goes the verified way:
``load_bundle`` first (full bundle verification, fail
closed), the act, then the manifest checkpoint is
re-anchored exactly as seam ingest and seam submission
re-anchor after journaling — so the bundle is left fully
verified for its next opener. CPU only; no network; no
secrets.
"""
from __future__ import annotations

import argparse
import json
import sys

from si.gaia.queue import fold_queue
from si.gaia.resolution import resolve_escalation

#: Scope label printed with every command's output: what the
#: trigger is, and what it is not.
TRIGGER_LABEL: str = (
    "CLI approval trigger (SI-111 §4.2, SI-113/ESC-LIVE-1): "
    "transport for the operator's decision, never the "
    "decision itself — listing reads the folded journal; "
    "resolving calls si.gaia.resolution.resolve_escalation "
    "and no other write path; no automatic resolution exists")


# ---------------------------------------------------------------------------
# Core: fold reads and the one resolution act
# ---------------------------------------------------------------------------

def _state_view(state) -> dict:
    """One folded intent as plain data for the operator: the
    journaled facts a resolution decision is made on."""
    return {
        "intent_seq": state.intent_seq,
        "edit_id": state.edit_id,
        "task_seq": state.task_seq,
        "tool": state.tool,
        "args": state.args,
        "reason": state.reason,
        "verdict": state.verdict,
        "effective_class": state.effective_class,
        "state": state.state,
        "resolution": state.resolution,
        "resolution_seq": state.resolution_seq,
        "resolution_operator": state.resolution_operator,
        "intent_ts": state.intent_ts,
    }


def list_pending_escalations(journal, task_seq=None) -> list:
    """The pending escalations: folded intents standing
    ``escalated`` with no resolution journaled, in journal
    sequence order. A resolved intent (either resolution) is
    not pending — ``reject`` leaves the fold state
    ``escalated`` but annotated, and the annotation is
    exactly what removes it from this list."""
    fold = fold_queue(journal, task_seq=task_seq)
    pending = [
        _state_view(fold.by_seq[seq])
        for seq in sorted(fold.by_seq)
        if fold.by_seq[seq].state == "escalated"
        and fold.by_seq[seq].resolution is None
    ]
    return pending


def read_escalation(journal, intent_seq, task_seq=None):
    """One folded intent by journal sequence, or None when
    the journal folds no intent at that sequence."""
    fold = fold_queue(journal, task_seq=task_seq)
    state = fold.by_seq.get(intent_seq)
    return None if state is None else _state_view(state)


def _open_read(bundle_dir: str):
    """Open a bundle's journal for reading: directly (no
    manifest-anchor demand — the SI-58 lesson), but the hash
    chain must verify before anything is folded from it."""
    from si.journal import open_journal
    journal = open_journal(bundle_dir)
    if not journal.verify_chain():
        journal.close()
        raise ValueError(
            "journal hash chain does not verify: refusing to "
            "fold a state from an unverified journal (fail "
            "closed)")
    return journal


def list_escalations(bundle_dir: str, task_seq=None) -> list:
    """Open ``bundle_dir`` and list its pending escalations."""
    journal = _open_read(bundle_dir)
    try:
        return list_pending_escalations(journal, task_seq=task_seq)
    finally:
        journal.close()


def read_escalation_at(bundle_dir: str, intent_seq: int,
                       task_seq=None):
    """Open ``bundle_dir`` and read one folded intent."""
    journal = _open_read(bundle_dir)
    try:
        return read_escalation(journal, intent_seq,
                               task_seq=task_seq)
    finally:
        journal.close()


def resolve(bundle_dir: str, intent_seq: int, resolution: str,
            operator: str, basis: str = "", detail: str = "",
            task_seq=None) -> dict:
    """Record the operator's resolution on one escalated
    intent: verify the bundle (``load_bundle``, fail closed),
    call :func:`si.gaia.resolution.resolve_escalation` — the
    existing verifying act, unmodified — then re-anchor the
    manifest checkpoint, exactly as seam ingest and seam
    submission do after journaling, so the bundle is left
    fully verified for its next opener. Returns the journaled
    resolution entry. Any refusal raises (ValueError from the
    act; bundle verification failures from the open) and
    journals nothing."""
    from si.bundle import (
        _anchor_checkpoint, _write_manifest, load_bundle)
    from si.journal import open_journal
    manifest = load_bundle(bundle_dir)
    journal = open_journal(bundle_dir, manifest)
    try:
        entry = resolve_escalation(
            journal, intent_seq, resolution, detail=detail,
            operator=operator, basis=basis, task_seq=task_seq)
        updated = _anchor_checkpoint(
            journal, manifest, manifest["head"],
            note=f"gaia escalation resolution: intent seq "
                 f"{intent_seq} resolved {resolution!r} by "
                 f"{operator!r} (SI-113/ESC-LIVE-1 CLI trigger)")
        _write_manifest(bundle_dir, updated)
        return entry
    finally:
        journal.close()


# ---------------------------------------------------------------------------
# CLI front-ends (module main + the si.cli wiring share `execute`)
# ---------------------------------------------------------------------------

def _print_pending(pending: list, as_json: bool) -> None:
    if as_json:
        print(json.dumps(pending, indent=2, default=str))
        return
    if not pending:
        print("no pending escalations")
        return
    for view in pending:
        print(f"seq={view['intent_seq']} edit={view['edit_id']} "
              f"task={view['task_seq']} tool={view['tool']} "
              f"class={view['effective_class']} "
              f"verdict={view['verdict']} state={view['state']}")
        print(f"  args: {json.dumps(view['args'], default=str)}")
        print(f"  reason: {view['reason']}")


def execute(args) -> int:
    """Run one parsed trigger command (the argument namespace
    built by :func:`build_parser` here or by ``si.cli``'s
    ``gaia-escalation`` parser — the two are the same shape).
    Returns the process exit code: 0 on success, 1 on a
    refused act or an unknown intent, 2 on a usage error the
    caller built by hand."""
    print(f"# {TRIGGER_LABEL}")
    try:
        if args.esc_cmd == "list":
            pending = list_escalations(args.path,
                                       task_seq=args.task)
            _print_pending(pending, args.json)
            return 0
        if args.esc_cmd == "show":
            view = read_escalation_at(args.path, args.intent_seq,
                                      task_seq=args.task)
            if view is None:
                print(f"si: error: no folded intent at seq "
                      f"{args.intent_seq}", file=sys.stderr)
                return 1
            if args.json:
                print(json.dumps(view, indent=2, default=str))
            else:
                for key, value in view.items():
                    print(f"{key}: {value}")
            return 0
        if args.esc_cmd == "resolve":
            entry = resolve(
                args.path, args.intent_seq, args.resolution,
                operator=args.operator, basis=args.basis or "",
                detail=args.detail or "", task_seq=args.task)
            if args.json:
                print(json.dumps(entry, indent=2, default=str))
            else:
                print(f"resolution journaled at seq {entry['seq']}")
                print(f"  intent_seq: {entry['intent_seq']} "
                      f"edit: {entry['edit_id']} "
                      f"task: {entry['task_seq']}")
                print(f"  resolution: {entry['resolution']} "
                      f"operator: {entry['operator']}")
                print(f"  basis: {entry['basis']}")
            return 0
    except ValueError as e:
        print(f"si: error: {e}", file=sys.stderr)
        return 1
    print(f"si: error: unknown escalation command "
          f"{args.esc_cmd!r}", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="si.gaia.escalation_cli",
        description="SI-113/ESC-LIVE-1 CLI approval trigger: "
                    "list and read pending escalations in a "
                    "bundle's journal, and record an operator "
                    "resolution through the existing "
                    "resolve_escalation act (SI-111 §4.2). "
                    "Transport, never decision.")
    sub = parser.add_subparsers(dest="esc_cmd", required=True)

    p_list = sub.add_parser(
        "list", help="list pending escalations (folded "
                     "escalated, unresolved)")
    p_list.add_argument("path", help="bundle directory")
    p_list.add_argument("--task", type=int, default=None,
                        help="scope the fold to one task seq")
    p_list.add_argument("--json", action="store_true",
                        help="print the pending list as JSON")

    p_show = sub.add_parser(
        "show", help="read one folded intent by journal seq")
    p_show.add_argument("path", help="bundle directory")
    p_show.add_argument("intent_seq", type=int,
                        help="the intent's journal sequence")
    p_show.add_argument("--task", type=int, default=None,
                        help="scope the fold to one task seq")
    p_show.add_argument("--json", action="store_true",
                        help="print the folded view as JSON")

    p_resolve = sub.add_parser(
        "resolve", help="record the operator's resolution of "
                        "one escalated intent (the journaled "
                        "act is resolve_escalation's; a refused "
                        "act journals nothing)")
    p_resolve.add_argument("path", help="bundle directory")
    p_resolve.add_argument("intent_seq", type=int,
                           help="the intent's journal sequence")
    # Deliberately no choices= here: the closed vocabulary is
    # verified inside resolve_escalation (SI-111 §4.2 — every
    # verification stays in the act, the trigger adds none).
    p_resolve.add_argument(
        "--resolution", required=True,
        help="the operator's resolution "
             "(admit-after-review | reject — verified by "
             "resolve_escalation, not by this parser)")
    p_resolve.add_argument(
        "--operator", required=True,
        help="operator name recorded on the resolution "
             "(required by the act; an anonymous resolution "
             "is refused)")
    p_resolve.add_argument("--basis", default=None,
                           help="basis recorded on the "
                                "resolution (default: the "
                                "act's pinned review basis)")
    p_resolve.add_argument("--detail", default=None,
                           help="free-text detail recorded on "
                                "the resolution")
    p_resolve.add_argument("--task", type=int, default=None,
                           help="scope the fold to one task seq")
    p_resolve.add_argument("--json", action="store_true",
                           help="print the journaled entry as "
                                "JSON")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return execute(args)
    except Exception as e:  # bundle verification etc: fail closed, loud
        print(f"si: FAILED: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
