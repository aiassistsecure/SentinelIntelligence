"""Gaia EXPORT-1: verifiable run bundle export + offline
verifier (SI-120 lane; slice SI-122; the roadmap's FREE
TURN hero item).

One command exports a window of a bundle's journal as ONE
self-contained file::

    si gaia-export <bundle> [--from SEQ] [--to SEQ]
                            [--out FILE]

and anyone — with no access to the bundle, no node, no
network — can check it::

    si gaia-verify-export <file>

The export carries the journal entries in sequence order
(each with its collection), the chain data the journal's
own verification needs (every entry's ``seq`` /
``prev_hash`` / ``entry_hash``, plus the hash the window
hangs from), the bundle's constitution record exactly as
the manifest records it (``hash`` / ``gate_version`` /
``policy_hash`` — no new versioning scheme is invented
here), the manifest head and manifest id the export was
taken at, and export metadata (``exported_at``, the source
tip seq + tip hash, the recorded ``si_version``).

The verifier recomputes the journal's EXISTING hash chain
over the exported entries — the same ``_entry_hash`` and
the same chain-payload exclusions ``Journal.verify_chain``
uses; no second hash scheme is invented — and checks the
export's internal consistency (window anchors, entry
count, the window tip against the recorded source tip).
It prints a clear verdict and exits non-zero on ANY
mismatch: a tampered entry, reordered entries, a
truncated chain, a tip hash mismatch.

The export is strictly READ-ONLY on the source bundle: it
opens the journal the ``_open_read`` way (chain verified
before anything is read, fail closed) and writes nothing
to the bundle.

Honest scope (also in docs/SI-122-verifiable-export.md):
the chain proves the export is internally consistent and
tamper-evident against the recorded tip — change any
chain-protected byte after export and verification fails.
It does NOT prove the outside world happened as recorded,
and it is NOT a signature: v1 has no signing keys, so
anyone can produce a self-consistent export of a journal
they fabricated wholesale; what they cannot do is alter
THIS export's chain-protected record undetectably. As in
the live journal (si/journal.py), the operational-state
keys ``result`` / ``delta_id`` / ``status`` and the
wall-clock ``ts`` are carried in each entry but are not
chain-protected — that is the journal's own documented
semantics, unchanged here.

CPU only; stdlib only; no network; no secrets.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

from si.gaia.escalation_cli import _open_read
from si.journal import (
    COLLECTIONS, GENESIS_PREV, MUTABLE_KEYS, _entry_hash)

#: The export format tag + version. This versions the
#: EXPORT FILE format only; the constitution record inside
#: is the manifest's own, verbatim.
EXPORT_FORMAT = "si-gaia-run-export"
EXPORT_FORMAT_VERSION = 1

#: Scope label printed with every command's output.
EXPORT_LABEL: str = (
    "verifiable run export (SI-122/EXPORT-1): a "
    "self-contained window of the bundle's journal, "
    "re-verifiable offline against the journal's own "
    "hash chain. The chain proves internal consistency "
    "and tamper-evidence against the recorded tip — not "
    "that the outside world happened as recorded, and "
    "not a signature (no signing keys in v1)")

_CHAIN_KEYS = frozenset({"_id", "seq", "prev_hash", "entry_hash"})
_NATIVE_KEYS = frozenset({"_seq", "_hash"})
#: Row keys that are backend storage detail (the chain
#: itself excludes them — see Journal.verify_chain): the
#: export carries the chain-protected record, not one
#: backend's row ids.
_STORAGE_KEYS = _CHAIN_KEYS | _NATIVE_KEYS


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z")


def _chain_payload(row: dict) -> dict:
    """The payload the journal's chain hashes for a row —
    exactly ``Journal._chain_payload``: everything except
    the chain keys, the native backend keys, the mutable
    operational-state keys, and ``ts``."""
    return {k: v for k, v in row.items()
            if k not in _CHAIN_KEYS and k not in _NATIVE_KEYS
            and k not in MUTABLE_KEYS and k != "ts"}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _read_manifest(bundle_dir: str) -> dict:
    """The bundle's manifest as recorded on disk. Read
    directly (the SI-58 lesson, as on the escalation
    trigger's reads): the export anchors to what the
    manifest records; it does not re-anchor or rewrite
    it."""
    path = os.path.join(os.path.abspath(bundle_dir),
                        "manifest.json")
    with open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def export_journal(journal, bundle_dir: str, manifest: dict,
                   from_seq: int = 0, to_seq=None) -> dict:
    """Build the export document for the window
    ``(from_seq, to_seq]`` of an already-open,
    already-chain-verified journal (the caller owns the
    open — see :func:`export_bundle`). Pure read: builds
    the document from backend finds only."""
    if not isinstance(from_seq, int) \
            or isinstance(from_seq, bool) or from_seq < 0:
        raise ValueError("from_seq must be an integer >= 0")
    tip_seq, tip_hash = journal.tip()
    if to_seq is None:
        to_seq = tip_seq
    if not isinstance(to_seq, int) \
            or isinstance(to_seq, bool) or to_seq < 0:
        raise ValueError("to_seq must be an integer >= 0")
    if to_seq < from_seq:
        raise ValueError(
            f"empty-by-reversal window ({from_seq}, "
            f"{to_seq}]: to_seq must be >= from_seq (a "
            f"zero-width window is the explicit empty case)")
    if to_seq > tip_seq:
        raise ValueError(
            f"to_seq {to_seq} is past the journal tip "
            f"{tip_seq}: an export cannot claim entries "
            f"the journal does not hold")

    rows = []
    start_prev = GENESIS_PREV
    for coll in COLLECTIONS:
        for row in journal.backend.find(coll) or []:
            seq = row.get("seq") or 0
            if seq == from_seq and from_seq > 0:
                start_prev = row.get("entry_hash")
            if from_seq < seq <= to_seq:
                rows.append((seq, coll, row))
    rows.sort(key=lambda item: item[0])

    entries = []
    for seq, coll, row in rows:
        carried = {k: v for k, v in row.items()
                   if k not in _STORAGE_KEYS}
        carried["seq"] = row.get("seq")
        carried["prev_hash"] = row.get("prev_hash")
        carried["entry_hash"] = row.get("entry_hash")
        entries.append({"collection": coll, "entry": carried})
    if entries:
        # The window hangs from the first entry's own
        # prev_hash (which is the hash at from_seq, or the
        # genesis prev for a full export).
        start_prev = entries[0]["entry"]["prev_hash"]
        window_tip = {
            "seq": entries[-1]["entry"]["seq"],
            "entry_hash": entries[-1]["entry"]["entry_hash"],
        }
    else:
        window_tip = None

    from si.bundle import manifest_id
    return {
        "format": EXPORT_FORMAT,
        "format_version": EXPORT_FORMAT_VERSION,
        "label": EXPORT_LABEL,
        "export": {
            "exported_at": _utcnow(),
            "tool": "si gaia-export (SI-122/EXPORT-1)",
            "si_version": manifest.get("si_version"),
            "source_bundle": os.path.abspath(bundle_dir),
            "window": {"from_seq": from_seq, "to_seq": to_seq},
            "window_start_prev_hash": start_prev,
            "window_tip": window_tip,
            "entry_count": len(entries),
            "source_tip": {"seq": tip_seq,
                           "entry_hash": tip_hash},
        },
        "bundle": {
            "name": manifest.get("name"),
            "manifest_id": manifest_id(manifest),
            "head": manifest.get("head"),
            "constitution": manifest.get("constitution"),
            "journal_anchor": manifest.get("journal"),
        },
        "entries": entries,
    }


def export_bundle(bundle_dir: str, from_seq: int = 0,
                  to_seq=None) -> dict:
    """Open ``bundle_dir`` read-only (chain verified
    first, fail closed — the ``_open_read`` way) and
    build the export document for the window. Writes
    nothing to the bundle."""
    manifest = _read_manifest(bundle_dir)
    journal = _open_read(bundle_dir)
    try:
        return export_journal(journal, bundle_dir, manifest,
                              from_seq=from_seq, to_seq=to_seq)
    finally:
        journal.close()


# ---------------------------------------------------------------------------
# Offline verification: no bundle, no node — the file alone
# ---------------------------------------------------------------------------

def verify_export(doc) -> list:
    """Check an export document's hash chain and internal
    consistency. Returns a list of problems — empty iff
    the export verifies. Pure: reads only ``doc``."""
    problems = []
    if not isinstance(doc, dict):
        return ["export is not a JSON object"]
    if doc.get("format") != EXPORT_FORMAT:
        problems.append(
            f"format is {doc.get('format')!r}, expected "
            f"{EXPORT_FORMAT!r}")
    if doc.get("format_version") != EXPORT_FORMAT_VERSION:
        problems.append(
            f"format_version is {doc.get('format_version')!r}, "
            f"expected {EXPORT_FORMAT_VERSION}")
    meta = doc.get("export")
    if not isinstance(meta, dict):
        return problems + ["export metadata block missing"]
    window = meta.get("window") or {}
    from_seq = window.get("from_seq")
    to_seq = window.get("to_seq")
    if not isinstance(from_seq, int) \
            or not isinstance(to_seq, int) \
            or from_seq < 0 or to_seq < from_seq:
        problems.append(
            f"window ({from_seq!r}, {to_seq!r}] is not a "
            f"valid (from_seq, to_seq] window")
        return problems
    entries = doc.get("entries")
    if not isinstance(entries, list):
        return problems + ["entries missing or not a list"]
    if meta.get("entry_count") != len(entries):
        problems.append(
            f"entry_count is {meta.get('entry_count')!r} but "
            f"{len(entries)} entries are present (truncated "
            f"or padded export)")

    start_prev = meta.get("window_start_prev_hash")
    if from_seq == 0 and start_prev != GENESIS_PREV:
        problems.append(
            "a full export (from_seq 0) must hang from the "
            f"genesis prev {GENESIS_PREV!r}, got "
            f"{start_prev!r}")

    prev = start_prev
    prev_seq = from_seq
    for i, item in enumerate(entries):
        where = f"entries[{i}]"
        if not isinstance(item, dict) \
                or item.get("collection") not in COLLECTIONS \
                or not isinstance(item.get("entry"), dict):
            problems.append(
                f"{where}: not a journal entry in a known "
                f"collection")
            return problems
        row = item["entry"]
        seq = row.get("seq")
        if not isinstance(seq, int) \
                or isinstance(seq, bool):
            problems.append(f"{where}: seq is not an integer")
            return problems
        if seq <= prev_seq:
            problems.append(
                f"{where}: seq {seq} does not strictly "
                f"follow {prev_seq} (reordered or "
                f"duplicated entries)")
            return problems
        if not (from_seq < seq <= to_seq):
            problems.append(
                f"{where}: seq {seq} lies outside the "
                f"declared window ({from_seq}, {to_seq}]")
        if row.get("prev_hash") != prev:
            problems.append(
                f"{where} (seq {seq}): prev_hash does not "
                f"match the preceding entry's hash (chain "
                f"broken or entries reordered)")
            return problems
        recomputed = _entry_hash(prev, seq, _chain_payload(row))
        if recomputed != row.get("entry_hash"):
            problems.append(
                f"{where} (seq {seq}): entry_hash does not "
                f"recompute from its payload (entry "
                f"tampered)")
            return problems
        prev = row.get("entry_hash")
        prev_seq = seq

    window_tip = meta.get("window_tip")
    if entries:
        expected_tip = {"seq": prev_seq, "entry_hash": prev}
        if window_tip != expected_tip:
            problems.append(
                f"window_tip is {window_tip!r} but the "
                f"chain ends at {expected_tip!r} (truncated "
                f"chain or altered tip)")
        if prev_seq != to_seq:
            problems.append(
                f"the chain ends at seq {prev_seq} but the "
                f"window declares to_seq {to_seq} "
                f"(truncated chain)")
    elif window_tip is not None:
        problems.append(
            "window_tip is recorded but the export holds "
            "no entries")

    source_tip = meta.get("source_tip") or {}
    if not isinstance(source_tip.get("seq"), int) \
            or not isinstance(source_tip.get("entry_hash"), str):
        problems.append(
            "source_tip (seq + entry_hash) missing or "
            "malformed")
    else:
        if to_seq > source_tip["seq"]:
            problems.append(
                f"window to_seq {to_seq} is past the "
                f"recorded source tip {source_tip['seq']}")
        if entries and to_seq == source_tip["seq"] \
                and window_tip is not None \
                and window_tip.get("entry_hash") \
                != source_tip["entry_hash"]:
            problems.append(
                "tip hash mismatch: the window reaches the "
                "recorded source tip seq but its final "
                "entry hash differs from the recorded "
                "source tip hash")
    return problems


def verify_export_file(path: str) -> dict:
    """Load an export file and verify it — the file
    alone, no bundle opened. Returns a result dict with
    ``ok``, the ``problems`` list, and the export's own
    summary facts for the verdict line."""
    with open(path, "rb") as f:
        doc = json.loads(f.read().decode("utf-8"))
    problems = verify_export(doc)
    meta = doc.get("export") or {}
    window = meta.get("window") or {}
    return {
        "ok": not problems,
        "problems": problems,
        "format": doc.get("format"),
        "format_version": doc.get("format_version"),
        "window": window,
        "entry_count": meta.get("entry_count"),
        "source_tip": meta.get("source_tip"),
        "window_tip": meta.get("window_tip"),
        "bundle": doc.get("bundle"),
        "exported_at": meta.get("exported_at"),
    }


# ---------------------------------------------------------------------------
# CLI front-ends (module main + the si.cli wiring share these)
# ---------------------------------------------------------------------------

def execute_export(args) -> int:
    """Run one parsed ``gaia-export`` command. Exit 0 on
    success (the export written, or printed to stdout
    when no ``--out``), 1 on a refused export (broken
    chain, bad window)."""
    print(f"# {EXPORT_LABEL}")
    try:
        doc = export_bundle(args.path,
                            from_seq=args.from_seq,
                            to_seq=args.to_seq)
    except (ValueError, FileNotFoundError) as e:
        print(f"si: error: {e}", file=sys.stderr)
        return 1
    text = json.dumps(doc, indent=2, sort_keys=True)
    meta = doc["export"]
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"exported {meta['entry_count']} entries, "
              f"window ({meta['window']['from_seq']}, "
              f"{meta['window']['to_seq']}], source tip "
              f"seq {meta['source_tip']['seq']} -> "
              f"{args.out}")
    else:
        print(text)
    return 0


def execute_verify(args) -> int:
    """Run one parsed ``gaia-verify-export`` command:
    verify the export file ALONE (no bundle opened).
    Exit 0 iff the chain recomputes and the export is
    internally consistent; 1 on any mismatch, with the
    problem printed."""
    print(f"# {EXPORT_LABEL}")
    try:
        result = verify_export_file(args.file)
    except (ValueError, FileNotFoundError) as e:
        print(f"si: error: {e}", file=sys.stderr)
        return 1
    bundle = result.get("bundle") or {}
    if result["ok"]:
        print(f"VERIFIED: {result['entry_count']} entries, "
              f"window ({result['window'].get('from_seq')}, "
              f"{result['window'].get('to_seq')}], chain "
              f"recomputes to the recorded tip "
              f"(seq {result['source_tip'].get('seq')}, "
              f"{result['source_tip'].get('entry_hash')})")
        print(f"bundle: {bundle.get('name')}  head="
              f"{bundle.get('head')}  manifest="
              f"{bundle.get('manifest_id')}")
        return 0
    for problem in result["problems"]:
        print(f"FAILED: {problem}")
    return 1


def build_export_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="si.gaia.run_export",
        description="SI-122/EXPORT-1 verifiable run "
                    "export: a self-contained window of "
                    "a bundle's journal, re-verifiable "
                    "offline against the journal's own "
                    "hash chain. Read-only on the bundle.")
    parser.add_argument("path", help="bundle directory")
    parser.add_argument("--from", dest="from_seq", type=int,
                        default=0,
                        help="export entries with seq > "
                             "from_seq (default 0: the "
                             "whole journal)")
    parser.add_argument("--to", dest="to_seq", type=int,
                        default=None,
                        help="export entries with seq <= "
                             "to_seq (default: the journal "
                             "tip)")
    parser.add_argument("--out", default=None,
                        help="write the export to FILE "
                             "(default: stdout)")
    return parser


def build_verify_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="si.gaia.run_export verify",
        description="SI-122/EXPORT-1 offline verifier: "
                    "recompute the journal hash chain "
                    "over an export file's entries and "
                    "check its internal consistency. "
                    "Opens no bundle.")
    parser.add_argument("file", help="the export file")
    return parser


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "verify":
        args = build_verify_parser().parse_args(argv[1:])
        try:
            return execute_verify(args)
        except Exception as e:  # fail closed, loud
            print(f"si: FAILED: {type(e).__name__}: {e}",
                  file=sys.stderr)
            return 1
    args = build_export_parser().parse_args(argv)
    try:
        return execute_export(args)
    except Exception as e:  # fail closed, loud
        print(f"si: FAILED: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
