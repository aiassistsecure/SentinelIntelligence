"""Gaia DIFFVIEW-1: checkpoint diff views over the journal
(SI-111 §5; slice SI-118).

A **read-only** diff of journaled state between two journal
seqs. Given ``(seq_a, seq_b)`` with ``seq_a <= seq_b``, the
view folds both prefixes — prefix folds are already the
journal's semantics (SI-92 §1.2: any prefix yields the state
as of that prefix) — and reports the delta, every row cited
by its journal seq:

- proposals journaled in ``(seq_a, seq_b]`` (self-edits, by
  kind, with their tools), and gate verdicts in the window
  counted by verdict (ADMIT / REJECT / ESCALATE);
- resolutions and dispositions journaled in the window,
  with their operators;
- claims, ``gaia_work`` submissions, execution outcomes,
  and materialisations (delta rows) in the window, each
  tied to its intent seq and, where present, its
  ``resolution_seq``;
- policy acts and integrity faults in the window,
  unabridged;
- per-intent state transitions between the two folds —
  including the legs the fold itself derives (an
  admit-after-review resolution moving an intent
  ``escalated`` -> ``queued``; an outcome moving it to
  ``executed`` / ``refused-at-claim``) — each with the
  journal seq that caused it.

The view is a projection over the one journal — the same
source the SSE spine streams (SI-47) — and opens the
journal read-only. **It has no mutation surface**: it
cannot journal, resolve, claim, execute, or amend. There
is no write path to audit because there is no write path.
The chain is verified before anything is folded; a broken
chain refuses the view (fail closed), exactly as the
escalation trigger's reads refuse (SI-113).

A zero-width window (``seq_a == seq_b``) is the explicit
empty case, not an error: it renders "no journaled change"
in text and ``"changed": false`` in JSON — never silence
that could be mistaken for a successful non-empty diff.

Honest scope (SI-111 §5.2, §10): a diff is a projection of
the journal, never new evidence. It earns no §18.1 row and
no §18.3 progress; its claim is only that the journal —
already the source of truth — becomes cheap to interrogate
between two points in its own history (the SQL-rewind
property, given a diff's shape).

Usage (also wired into the main CLI as ``si gaia-diff``)::

    si gaia-diff <bundle> <seq_a> <seq_b> [--task SEQ] [--json]

CPU only; no network; no secrets.
"""
from __future__ import annotations

import argparse
import json
import sys
from types import SimpleNamespace

from si.gaia.queue import fold_queue
from si.journal import COLLECTIONS

#: Scope label printed with every command's output: what
#: the view is, and what it is not.
DIFFVIEW_LABEL: str = (
    "checkpoint diff view (SI-111 §5, SI-118/DIFFVIEW-1): "
    "a read-only projection of the journal between two "
    "prefixes — it reports what the journal records, "
    "cited by journal seq; it is not new evidence and "
    "earns no §18.1 row and no §18.3 progress")

#: Experience kinds the queue fold consumes — the journal
#: seqs at which a prefix fold can change. Folding at
#: exactly these seqs (plus the window's endpoints) is what
#: makes per-intent transitions attributable to the seq
#: that caused them; any other row leaves the fold
#: untouched by construction.
_FOLD_KINDS = {
    "task", "queue_batch", "queue_stop", "queue_resume",
    "queue_claim", "queue_recovery_reclaim", "queue_refusal",
    "queue_skip", "queue_outcome_unknown",
    "queue_recovery_observation", "queue_disposition",
    "escalation_resolution", "tool_result",
}


# ---------------------------------------------------------------------------
# Prefix folds: the journal's own semantics, read through a seq filter
# ---------------------------------------------------------------------------

class _PrefixBackend:
    """A read-only seq-filtering view over a journal
    backend: ``find`` returns only rows at or below
    ``upto``. This is how a prefix fold is taken without
    touching :func:`si.gaia.queue.fold_queue` — the fold
    reads the backend, so filtering the backend's rows to
    a prefix folds the prefix. Writes raise: the view has
    no write path."""

    def __init__(self, backend, upto: int):
        self._backend = backend
        self._upto = upto

    def find(self, collection, filter=None, sort=None,
             limit=None):
        rows = self._backend.find(
            collection, filter=filter, sort=sort) or []
        rows = [row for row in rows
                if (row.get("seq") or 0) <= self._upto]
        if limit is not None:
            rows = rows[:limit]
        return rows

    def count(self, collection) -> int:
        return len(self.find(collection))

    def insert(self, collection, doc):  # pragma: no cover
        raise RuntimeError("prefix view is read-only")

    def update(self, collection, row_id, doc):  # pragma: no cover
        raise RuntimeError("prefix view is read-only")

    def close(self) -> None:
        # The wrapped backend is owned (and closed) by the
        # caller; a prefix view closes nothing.
        pass


def _fold_at(journal, upto: int, task_seq=None):
    """Fold the queue as of journal prefix ``upto`` — the
    state the journal itself derives for that point in its
    history (SI-92 §1.2)."""
    shim = SimpleNamespace(
        backend=_PrefixBackend(journal.backend, upto))
    return fold_queue(shim, task_seq=task_seq)


def _snapshot(fold) -> dict:
    """The per-intent facts a transition is read from: the
    folded state plus the resolution / disposition
    annotations (a ``reject`` resolution changes no state,
    and the annotation is still a journaled change the
    diff must render)."""
    return {
        seq: {
            "state": state.state,
            "resolution": state.resolution,
            "resolution_operator": state.resolution_operator,
            "disposition": state.disposition,
            "edit_id": state.edit_id,
            "tool": state.tool,
        }
        for seq, state in fold.by_seq.items()
    }


# ---------------------------------------------------------------------------
# The diff
# ---------------------------------------------------------------------------

def _window_rows(journal, seq_a: int, seq_b: int) -> list:
    """Every journal row in ``(seq_a, seq_b]``, all
    collections, in journal sequence order."""
    rows = []
    for coll in COLLECTIONS:
        for row in journal.backend.find(coll) or []:
            seq = row.get("seq") or 0
            if seq_a < seq <= seq_b:
                rows.append((seq, coll, row))
    rows.sort(key=lambda item: item[0])
    return rows


def _record(row: dict) -> dict:
    return row.get("record") or {}


def _kind(row: dict) -> str:
    return row.get("kind") or _record(row).get("kind") or ""


def _intent_seq_of(row: dict, record: dict):
    """The intent a window row names, as best the row
    itself states it: an explicit intent/journal seq on
    the record, else the outcome's ``journal_seq``, else
    None (the row's own ``self_edit_id`` column still ties
    it to its edit, carried on the rendered item)."""
    for key in ("intent_seq", "journal_seq"):
        value = record.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def compute_diff(journal, seq_a: int, seq_b: int,
                 task_seq=None) -> dict:
    """The diff of journaled state between two prefixes of
    an already-open journal. The caller owns chain
    verification (the bundle-level entry points verify
    before calling this — see :func:`diff_bundle`); this
    function only reads.

    ``seq_a <= seq_b`` is required; ``seq_b < seq_a``
    raises ValueError. Seqs past the tip fold as the tip
    (a prefix beyond the end is the whole journal)."""
    if not isinstance(seq_a, int) or isinstance(seq_a, bool) \
            or not isinstance(seq_b, int) \
            or isinstance(seq_b, bool):
        raise ValueError("journal seqs must be integers")
    if seq_a < 0 or seq_b < 0:
        raise ValueError("journal seqs must be >= 0")
    if seq_b < seq_a:
        raise ValueError(
            f"empty-by-reversal window ({seq_a}, {seq_b}]: "
            f"seq_b must be >= seq_a (a zero-width window "
            f"is the explicit empty case)")

    tip_seq, tip_hash = journal.tip()
    rows = _window_rows(journal, seq_a, seq_b)

    diff = {
        "label": DIFFVIEW_LABEL,
        "window": {"seq_a": seq_a, "seq_b": seq_b,
                   "tip": tip_seq, "tip_hash": tip_hash},
        "proposals": [],
        "verdict_counts": {},
        "resolutions": [],
        "dispositions": [],
        "claims": [],
        "gaia_work": [],
        "outcomes": [],
        "materialisations": [],
        "checkpoints": [],
        "policy_acts": [],
        "integrity_faults": [],
        "transitions": [],
    }

    for seq, coll, row in rows:
        if coll == "self_edits":
            intent = row.get("intent") or {}
            diff["proposals"].append({
                "seq": seq,
                "edit_id": row.get("edit_id"),
                "kind": row.get("kind"),
                "gate_verdict": row.get("gate_verdict"),
                "tool": intent.get("tool"),
                "operator": row.get("operator"),
            })
            verdict = row.get("gate_verdict")
            if verdict is not None:
                diff["verdict_counts"][verdict] = \
                    diff["verdict_counts"].get(verdict, 0) + 1
            continue
        if coll == "deltas":
            diff["materialisations"].append({
                "seq": seq,
                "delta_id": row.get("id"),
                "self_edit_id": row.get("self_edit_id"),
                "status": row.get("status"),
                "bytes": row.get("bytes"),
            })
            continue
        if coll == "checkpoints":
            diff["checkpoints"].append({
                "seq": seq,
                "head": row.get("head"),
                "note": row.get("note"),
            })
            continue
        if coll != "experiences":
            continue
        record = _record(row)
        kind = _kind(row)
        if kind == "escalation_resolution":
            diff["resolutions"].append({
                "seq": seq,
                "intent_seq": record.get("intent_seq"),
                "edit_id": record.get("edit_id"),
                "task_seq": record.get("task_seq"),
                "resolution": record.get("resolution"),
                "operator": record.get("operator"),
                "detail": record.get("detail"),
            })
        elif kind == "queue_disposition":
            diff["dispositions"].append({
                "seq": seq,
                "intent_seq": _intent_seq_of(row, record),
                "edit_id": record.get("edit_id")
                or row.get("self_edit_id"),
                "disposition": record.get("disposition"),
                "operator": record.get("operator"),
                "detail": record.get("detail"),
            })
        elif kind in ("queue_claim", "queue_recovery_reclaim"):
            diff["claims"].append({
                "seq": seq,
                "kind": kind,
                "intent_seq": _intent_seq_of(row, record),
                "edit_id": record.get("edit_id")
                or row.get("self_edit_id"),
                "drainer_id": record.get("drainer_id"),
                "lease_seconds": record.get("lease_seconds"),
                "recovery_of": record.get("recovery_of")
                or record.get("original_claim_seq"),
            })
        elif kind == "gaia_work":
            diff["gaia_work"].append({
                "seq": seq,
                "intent_seq": _intent_seq_of(row, record),
                "edit_id": record.get("edit_id")
                or row.get("self_edit_id"),
                "resolution_seq": record.get("resolution_seq"),
                "fork_work_id": record.get("fork_work_id"),
                "status": record.get("status"),
                "operator": record.get("operator"),
            })
        elif kind == "tool_result":
            execution = record.get("execution")
            if not isinstance(execution, dict):
                execution = {}
            diff["outcomes"].append({
                "seq": seq,
                "intent_seq": _intent_seq_of(row, record),
                "edit_id": record.get("intent_id")
                or row.get("self_edit_id"),
                "status": execution.get("status"),
                "work_id": record.get("work_id"),
                "source": record.get("source"),
            })
        elif kind == "operator_act" or "policy" in kind:
            # §5.1: policy acts in the window, unabridged.
            diff["policy_acts"].append({
                "seq": seq, "kind": kind, "record": record})
        elif kind == "gaia_outcome" \
                and record.get("status") == "integrity-fault":
            # §5.1: integrity faults in the window,
            # unabridged.
            diff["integrity_faults"].append({
                "seq": seq, "kind": kind, "record": record})

    # -- per-intent transitions: successive prefix folds ---
    fold_a = _fold_at(journal, seq_a, task_seq=task_seq)
    fold_b = _fold_at(journal, seq_b, task_seq=task_seq)
    fold_points = sorted({
        seq for seq, coll, row in rows
        if coll == "self_edits" and row.get("kind") == "tool"
        or coll == "experiences" and _kind(row) in _FOLD_KINDS
    })
    prev = _snapshot(fold_a)
    for point in fold_points:
        snap = _snapshot(_fold_at(journal, point,
                                   task_seq=task_seq))
        for intent_seq in sorted(set(prev) | set(snap)):
            before = prev.get(intent_seq)
            after = snap.get(intent_seq)
            if before is None and after is None:
                continue
            if before is None:
                diff["transitions"].append({
                    "intent_seq": intent_seq,
                    "edit_id": after["edit_id"],
                    "tool": after["tool"],
                    "at_seq": intent_seq,
                    "change": "appeared",
                    "from_state": None,
                    "to_state": after["state"],
                })
            elif after is None:
                # A prefix fold never un-sees an intent;
                # reaching this would be a fold defect, and
                # the diff reports it rather than hiding it.
                diff["transitions"].append({
                    "intent_seq": intent_seq,
                    "edit_id": before["edit_id"],
                    "tool": before["tool"],
                    "at_seq": point,
                    "change": "vanished",
                    "from_state": before["state"],
                    "to_state": None,
                })
            else:
                if after["state"] != before["state"]:
                    diff["transitions"].append({
                        "intent_seq": intent_seq,
                        "edit_id": after["edit_id"],
                        "tool": after["tool"],
                        "at_seq": point,
                        "change": "state",
                        "from_state": before["state"],
                        "to_state": after["state"],
                    })
                if after["resolution"] != before["resolution"]:
                    diff["transitions"].append({
                        "intent_seq": intent_seq,
                        "edit_id": after["edit_id"],
                        "tool": after["tool"],
                        "at_seq": point,
                        "change": "resolution",
                        "from_state": before["state"],
                        "to_state": after["state"],
                        "resolution": after["resolution"],
                        "resolution_operator":
                            after["resolution_operator"],
                    })
                if after["disposition"] != before["disposition"]:
                    diff["transitions"].append({
                        "intent_seq": intent_seq,
                        "edit_id": after["edit_id"],
                        "tool": after["tool"],
                        "at_seq": point,
                        "change": "disposition",
                        "from_state": before["state"],
                        "to_state": after["state"],
                        "disposition": after["disposition"],
                    })
        prev = snap

    # Fold-level integrity faults are derived state (they
    # carry no row of their own): report the ones the fold
    # at seq_b holds that the fold at seq_a did not, beside
    # the journaled fault rows collected above (§5.1:
    # unabridged either way).
    for task, slot in sorted(fold_b.tasks.items(),
                             key=lambda kv: str(kv[0])):
        before_faults = (fold_a.tasks.get(task) or {}).get(
            "integrity_faults") or []
        for fault in slot.get("integrity_faults") or []:
            if fault not in before_faults:
                diff["integrity_faults"].append({
                    "seq": None, "kind": "fold",
                    "task_seq": task, "fault": fault})

    diff["changed"] = any(
        diff[key] for key in (
            "proposals", "resolutions", "dispositions",
            "claims", "gaia_work", "outcomes",
            "materialisations", "checkpoints", "policy_acts",
            "integrity_faults", "transitions"))
    if not diff["changed"]:
        diff["message"] = (
            f"no journaled change between seq {seq_a} and "
            f"seq {seq_b}")
    return diff


# ---------------------------------------------------------------------------
# Bundle-level read: verify the chain, then diff (read-only)
# ---------------------------------------------------------------------------

def _open_read(bundle_dir: str):
    """Open a bundle's journal for reading: directly (no
    manifest-anchor demand — the SI-58 lesson, as on the
    escalation trigger's reads), but the hash chain must
    verify before anything is folded from it."""
    from si.journal import open_journal
    journal = open_journal(bundle_dir)
    if not journal.verify_chain():
        journal.close()
        raise ValueError(
            "journal hash chain does not verify: refusing "
            "to diff an unverified journal (fail closed)")
    return journal


def diff_bundle(bundle_dir: str, seq_a: int, seq_b: int,
                task_seq=None) -> dict:
    """Open ``bundle_dir`` read-only, verify its chain,
    and diff journaled state between the two prefixes.
    Writes nothing: no journaling, no re-anchor, no
    manifest touch."""
    journal = _open_read(bundle_dir)
    try:
        return compute_diff(journal, seq_a, seq_b,
                            task_seq=task_seq)
    finally:
        journal.close()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_text(diff: dict, bundle_dir: str = "-") -> str:
    """The human-readable rendering: the label, the
    window, then every section with its rows cited by
    journal seq. An unchanged window renders the explicit
    no-change line and nothing else."""
    window = diff["window"]
    lines = [
        f"bundle: {bundle_dir}",
        f"window: ({window['seq_a']}, {window['seq_b']}] "
        f"tip={window['tip']} chain=verified",
    ]
    if not diff["changed"]:
        lines.append(diff["message"])
        return "\n".join(lines)

    def section(title, items, fmt):
        lines.append(f"{title}: {len(items)}")
        for item in items:
            lines.append("  " + fmt(item))

    section("proposals", diff["proposals"], lambda p:
            f"seq {p['seq']}  {p['edit_id']}  kind={p['kind']}"
            + (f"  tool={p['tool']}" if p["tool"] else "")
            + f"  verdict={p['gate_verdict']}")
    if diff["verdict_counts"]:
        counts = ", ".join(
            f"{verdict}={count}" for verdict, count
            in sorted(diff["verdict_counts"].items()))
        lines.append(f"verdicts: {counts}")
    section("resolutions", diff["resolutions"], lambda r:
            f"seq {r['seq']}  intent={r['intent_seq']} "
            f"edit={r['edit_id']}  "
            f"resolution={r['resolution']}  "
            f"operator={r['operator']}")
    section("dispositions", diff["dispositions"], lambda d:
            f"seq {d['seq']}  intent={d['intent_seq']} "
            f"edit={d['edit_id']}  "
            f"disposition={d['disposition']}  "
            f"operator={d['operator']}")
    section("claims", diff["claims"], lambda c:
            f"seq {c['seq']}  {c['kind']}  "
            f"intent={c['intent_seq']} edit={c['edit_id']}  "
            f"drainer={c['drainer_id']}")
    section("gaia_work", diff["gaia_work"], lambda g:
            f"seq {g['seq']}  intent={g['intent_seq']} "
            f"edit={g['edit_id']}  "
            f"resolution_seq={g['resolution_seq']}  "
            f"work={g['fork_work_id']}  status={g['status']}")
    section("outcomes", diff["outcomes"], lambda o:
            f"seq {o['seq']}  intent={o['intent_seq']} "
            f"edit={o['edit_id']}  status={o['status']}"
            + (f"  work={o['work_id']}" if o["work_id"] else ""))
    section("materialisations", diff["materialisations"],
            lambda m: f"seq {m['seq']}  delta={m['delta_id']} "
                      f"self_edit={m['self_edit_id']}  "
                      f"status={m['status']}")
    section("checkpoints", diff["checkpoints"], lambda c:
            f"seq {c['seq']}  head={c['head']}  "
            f"note={c['note']}")
    section("policy acts", diff["policy_acts"], lambda p:
            f"seq {p['seq']}  kind={p['kind']}  "
            f"record={json.dumps(p['record'], default=str)}")
    section("integrity faults", diff["integrity_faults"],
            lambda f: f"seq {f['seq']}  kind={f['kind']}  "
                      + (f"fault={f['fault']}"
                         if f["kind"] == "fold" else
                         f"record={json.dumps(f['record'], default=str)}"))
    section("transitions", diff["transitions"], lambda t:
            f"seq {t['at_seq']}  intent={t['intent_seq']} "
            f"edit={t['edit_id']} tool={t['tool']}: "
            f"{t['from_state'] or '—'} -> "
            f"{t['to_state'] or '—'}"
            + (f"  [{t['change']}]"
               if t["change"] != "state" else "")
            + (f"  resolution={t['resolution']} "
               f"by {t['resolution_operator']}"
               if t["change"] == "resolution" else "")
            + (f"  disposition={t['disposition']}"
               if t["change"] == "disposition" else ""))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI front-ends (module main + the si.cli wiring share `execute`)
# ---------------------------------------------------------------------------

def execute(args) -> int:
    """Run one parsed diff command (the argument namespace
    built by :func:`build_parser` here or by ``si.cli``'s
    ``gaia-diff`` parser — the two are the same shape).
    Returns the process exit code: 0 on success (an empty
    window is a success with an explicit no-change
    rendering), 1 on a refused view (broken chain, bad
    window), 2 on a usage error the caller built by hand."""
    print(f"# {DIFFVIEW_LABEL}")
    try:
        diff = diff_bundle(args.path, args.seq_a, args.seq_b,
                           task_seq=args.task)
    except ValueError as e:
        print(f"si: error: {e}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(diff, indent=2, default=str))
    else:
        print(render_text(diff, bundle_dir=args.path))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="si.gaia.diffview",
        description="SI-118/DIFFVIEW-1 checkpoint diff "
                    "view: a read-only diff of journaled "
                    "state between two journal seqs, folded "
                    "from the journal's own prefixes "
                    "(SI-111 §5). It journals nothing, "
                    "resolves nothing, executes nothing.")
    parser.add_argument("path", help="bundle directory")
    parser.add_argument("seq_a", type=int,
                        help="the earlier checkpoint: the "
                             "diff covers (seq_a, seq_b]")
    parser.add_argument("seq_b", type=int,
                        help="the later checkpoint (a "
                             "zero-width window, seq_b == "
                             "seq_a, renders the explicit "
                             "no-change result)")
    parser.add_argument("--task", type=int, default=None,
                        help="scope the folded transitions "
                             "to one task seq")
    parser.add_argument("--json", action="store_true",
                        help="print the diff as JSON")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return execute(args)
    except Exception as e:  # bundle open etc: fail closed, loud
        print(f"si: FAILED: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
