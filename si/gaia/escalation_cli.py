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
    si gaia-escalation <bundle> spend <intent_seq> \
        --operator NAME [--basis TEXT] \
        [--work-url URL] [--task SEQ] [--json]
    si gaia-escalation <bundle> poll-outcome <intent_seq> \
        [--work-url URL] [--json]
    si gaia-escalation <bundle> reconcile-outcome <intent_seq> \
        --outcomes PATH [--cursor N] [--json]

**SI-116 (P-1) extends this family** — the design choice,
recorded: the spend and its outcome leg live on the same
operator surface as the resolution they spend, rather than
a new command, because the escalation lifecycle is one
operator story (list → resolve → spend → poll) and D-115-1
keeps every step an explicit, named operator act. ``spend``
calls :func:`si.gaia.seam_spend.spend_resolved_escalation`
and nothing else — the fold verification (queued *by* an
admit-after-review resolution) and the fork-shaped payload
live in that act, exactly as the resolution's verification
lives in ``resolve_escalation``; this trigger adds no
verification of its own, and a verification refusal
journals nothing (a fork/transport failure is the act's own
business: journaled as evidence inside it). ``poll-outcome``
and ``reconcile-outcome`` are the S-5 outcome leg scoped to
the one spent intent (SI-115's missing (d)-leg caller):
reads and journaled-outcome writes through S-5's existing
machinery, invoked explicitly, never on a timer.

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
# The spend and its outcome leg (SI-116 / P-1)
# ---------------------------------------------------------------------------

def spend(bundle_dir: str, intent_seq: int, operator: str,
          basis: str = "", work_url: str | None = None,
          task_seq=None) -> dict:
    """Spend one admit-after-review resolution: call
    :func:`si.gaia.seam_spend.spend_resolved_escalation` —
    the verifying act, unmodified — and return its result.
    The trigger supplies the operator's name and the fork
    work URL (the CLI's default is the live node's route,
    SI-115 §4); every verification stays inside the act."""
    from si.gaia.seam_spend import spend_resolved_escalation
    return spend_resolved_escalation(
        bundle_dir, intent_seq, operator=operator, basis=basis,
        work_url=work_url, task_seq=task_seq)


def poll_outcome(bundle_dir: str, intent_seq: int,
                 work_url: str | None = None) -> dict:
    """Poll the fork for one spent intent's outcome (S-5
    §11.2, intent-scoped): a thin call into
    :mod:`si.gaia.seam_spend`'s poll helper, which journals
    a terminal status through S-5's own outcome writer."""
    from si.gaia.seam_spend import poll_spent_outcome
    return poll_spent_outcome(bundle_dir, intent_seq,
                              work_url=work_url)


def reconcile_outcome(bundle_dir: str, intent_seq: int,
                      outcomes_path: str, cursor: int = 0) -> dict:
    """Reconcile one spent intent against the fork's
    outcomes.jsonl (S-5 §11.3, intent-scoped): a thin call
    into :mod:`si.gaia.seam_spend`'s reconcile helper. The
    cursor is the operator's to persist between calls."""
    from si.gaia.seam_spend import reconcile_spent_outcome
    return reconcile_spent_outcome(
        bundle_dir, intent_seq, outcomes_path, cursor=cursor)


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
        if args.esc_cmd == "spend":
            from si.gaia.seam_spend import DEFAULT_FORK_WORK_URL
            result = spend(
                args.path, args.intent_seq,
                operator=args.operator, basis=args.basis or "",
                work_url=args.work_url or DEFAULT_FORK_WORK_URL,
                task_seq=args.task)
            if args.json:
                print(json.dumps(result, indent=2, default=str))
            else:
                print(f"spend: {result['status']}")
                for key in ("intent_seq", "edit_id",
                            "resolution_seq", "fork_work_id",
                            "gaia_work_seq", "http_status",
                            "deduped", "failure"):
                    if result.get(key) is not None:
                        print(f"  {key}: {result[key]}")
                for problem in result.get("problems") or []:
                    print(f"  problem: {problem}")
            return 0 if result["status"] == "submitted" else 1
        if args.esc_cmd == "poll-outcome":
            from si.gaia.seam_spend import DEFAULT_FORK_WORK_URL
            result = poll_outcome(
                args.path, args.intent_seq,
                work_url=args.work_url or DEFAULT_FORK_WORK_URL)
            if args.json:
                print(json.dumps(result, indent=2, default=str))
            else:
                print(f"poll-outcome: {result['status']}")
                for key in ("intent_seq", "work_id", "outcome",
                            "outcome_seq", "state", "failure"):
                    if result.get(key) is not None:
                        print(f"  {key}: {result[key]}")
                for problem in result.get("problems") or []:
                    print(f"  problem: {problem}")
            return 0 if result["status"] in (
                "journaled", "awaiting") else 1
        if args.esc_cmd == "reconcile-outcome":
            result = reconcile_outcome(
                args.path, args.intent_seq, args.outcomes,
                cursor=args.cursor or 0)
            if args.json:
                print(json.dumps(result, indent=2, default=str))
            else:
                print(f"reconcile-outcome: {result['status']}")
                for key in ("intent_seq", "work_id", "cursor",
                            "count", "torn"):
                    if result.get(key) is not None:
                        print(f"  {key}: {result[key]}")
                for problem in result.get("problems") or []:
                    print(f"  problem: {problem}")
            return 0 if result["status"] in (
                "reconciled",) else 1
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

    p_spend = sub.add_parser(
        "spend", help="spend one admit-after-review resolution "
                      "across the seam (SI-116/P-1: the act is "
                      "spend_resolved_escalation's; a refused "
                      "verification journals nothing, a fork / "
                      "transport failure is journaled as "
                      "evidence inside the act)")
    p_spend.add_argument("path", help="bundle directory")
    p_spend.add_argument("intent_seq", type=int,
                         help="the resolved intent's journal "
                              "sequence")
    p_spend.add_argument(
        "--operator", required=True,
        help="operator name recorded on the spend's "
             "gaia_work row (required by the act; an "
             "anonymous spend is refused)")
    p_spend.add_argument("--basis", default=None,
                         help="basis recorded on the spend "
                              "(default: the act's pinned "
                              "explicit-spend basis)")
    p_spend.add_argument("--work-url", default=None,
                         help="the fork's work route URL "
                              "(default: the live node's "
                              "http://127.0.0.1:11435/gaia/v1/work, "
                              "SI-115 §4)")
    p_spend.add_argument("--task", type=int, default=None,
                         help="scope the fold to one task seq")
    p_spend.add_argument("--json", action="store_true",
                         help="print the spend result as JSON")

    p_poll = sub.add_parser(
        "poll-outcome", help="poll the fork once for one spent "
                             "intent's outcome (S-5 §11.2, "
                             "intent-scoped; a terminal status "
                             "is journaled through S-5's "
                             "outcome writer, once per work id)")
    p_poll.add_argument("path", help="bundle directory")
    p_poll.add_argument("intent_seq", type=int,
                        help="the spent intent's journal "
                             "sequence")
    p_poll.add_argument("--work-url", default=None,
                        help="the fork's work route URL "
                             "(default: the live node's "
                             "http://127.0.0.1:11435/gaia/v1/work)")
    p_poll.add_argument("--json", action="store_true",
                        help="print the poll result as JSON")

    p_reconcile = sub.add_parser(
        "reconcile-outcome", help="reconcile one spent intent "
                                  "against the fork's "
                                  "outcomes.jsonl (S-5 §11.3, "
                                  "intent-scoped, read-only tail)")
    p_reconcile.add_argument("path", help="bundle directory")
    p_reconcile.add_argument("intent_seq", type=int,
                             help="the spent intent's journal "
                                  "sequence")
    p_reconcile.add_argument("--outcomes", required=True,
                             help="path to the fork's "
                                  "outcomes.jsonl (read-only)")
    p_reconcile.add_argument("--cursor", type=int, default=None,
                             help="byte offset to tail from "
                                  "(default: 0; persist the "
                                  "returned cursor between calls)")
    p_reconcile.add_argument("--json", action="store_true",
                             help="print the reconcile result "
                                  "as JSON")
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
