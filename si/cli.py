"""Command-line interface for the SI-0 reference runtime.

    si init <path> [--name NAME]        create a new .si bundle
    si cycle <path> [--mind demo|mock]  run one self-write cycle
    si rollback <path> --to <id|base> [--reason R] [--mind demo|mock]
    si log <path> [--rollbacks] [--limit N]
    si diff <path> <a>..<b>
    si why <path> <delta-id>

<path> is the bundle directory (conventionally <name>.si).
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _bundle(path: str) -> str:
    return os.path.abspath(path)


def cmd_init(args) -> int:
    from si.bundle import init_bundle, load_bundle, manifest_id
    bundle_dir = init_bundle(args.path, name=args.name,
                             journal_backend=args.journal_backend)
    manifest = load_bundle(bundle_dir)
    print(f"initialized {manifest_id(manifest)}")
    print(f"bundle: {bundle_dir}")
    print(f"journal backend: {args.journal_backend}")
    print("note: demo seed is RANDOM DATA, not a trained model "
          "(see base.source in manifest.json)")
    if args.journal_backend == "embedded":
        print("note: journal backend is EMBEDDED SQLITE; use "
              "--journal-backend nedb for the real NEDB engine")
    return 0


def _make_mind(kind: str, bundle_dir: str):
    from si.minds import DemoMind, MockMind
    if kind == "mock":
        return MockMind()
    return DemoMind(bundle_path=bundle_dir)


def cmd_cycle(args) -> int:
    from si.loop import run_cycle
    bundle_dir = _bundle(args.path)
    mind = _make_mind(args.mind, bundle_dir)
    res = run_cycle(bundle_dir, mind)
    print(f"verdict:   {res['verdict']}")
    print(f"edit_id:   {res['edit_id']}")
    print(f"delta_id:  {res['delta_id']}")
    print(f"head:      {res['head']}")
    print(f"verified:  {res['verified']}")
    print(f"rolled_back: {res['rolled_back']}")
    return 0


def cmd_rollback(args) -> int:
    from si.loop import rollback
    bundle_dir = _bundle(args.path)
    mind = _make_mind(args.mind, bundle_dir)
    res = rollback(bundle_dir, to=args.to, reason=args.reason, mind=mind)
    print(f"head:      {res['head']}")
    print(f"edit_id:   {res['edit_id']}")
    print(f"probes_ok: {res['probes_ok']}")
    return 0


def cmd_log(args) -> int:
    from si.bundle import load_bundle
    from si.journal import open_journal
    bundle_dir = _bundle(args.path)
    manifest = load_bundle(bundle_dir)  # fail closed before reading
    journal = open_journal(bundle_dir, manifest)
    try:
        if args.rollbacks:
            rows = journal.rollbacks(limit=args.limit)
            if not rows:
                print("no rollbacks recorded")
                return 0
            for r in rows:
                print(f"seq={r['seq']} {r['edit_id']} kind=rollback "
                      f"verdict={r['gate_verdict']} result={r['result']}")
                print(f"  reason: {r['reason']}")
                intent = r.get("intent") or {}
                print(f"  from={intent.get('from')} to={intent.get('target')} "
                      f"cause={intent.get('cause')}")
        else:
            rows = journal.log(limit=args.limit)
            for r in rows:
                print(f"seq={r['seq']} head={r['head']} manifest={r['manifest']}")
                print(f"  {r['note']}")
    finally:
        journal.close()
    return 0


def _resolve_manifest(bundle_dir: str, point: str) -> dict:
    """Resolve base|<delta-id>|head to a manifest snapshot dict."""
    from si.bundle import load_bundle
    from si.journal import open_journal
    manifest = load_bundle(bundle_dir)
    if point == "head":
        return manifest
    journal = open_journal(bundle_dir, manifest)
    try:
        cps = journal.backend.find("checkpoints", filter={"head": point},
                                   sort="-seq", limit=1)
        if not cps or "manifest_json" not in cps[0]:
            raise ValueError(f"no checkpoint found for {point!r}")
        return json.loads(cps[0]["manifest_json"])
    finally:
        journal.close()


def cmd_diff(args) -> int:
    bundle_dir = _bundle(args.path)
    if ".." not in args.range:
        print("usage: si diff <path> <a>..<b>  (e.g. base..d0001)", file=sys.stderr)
        return 2
    a, b = args.range.split("..", 1)
    ma = _resolve_manifest(bundle_dir, a or "base")
    mb = _resolve_manifest(bundle_dir, b or "head")
    ids_a = {d["id"] for d in ma["deltas"]}
    print(f"--- {a}  (head={ma['head']}, deltas={len(ma['deltas'])})")
    print(f"+++ {b}  (head={mb['head']}, deltas={len(mb['deltas'])})")
    for d in mb["deltas"]:
        if d["id"] not in ids_a:
            print(f"+ {d['id']}: parent={d['parent']} rank={d['rank']} "
                  f"alpha={d['alpha']} norm={d['norm']:.6f} "
                  f"modules={d['target_modules']} blob={d['blob'][:19]}... "
                  f"self_edit={d['self_edit_id']}")
    if ma["head"] != mb["head"]:
        print(f"head: {ma['head']} -> {mb['head']}")
    return 0


def cmd_why(args) -> int:
    from si.bundle import load_bundle
    from si.journal import open_journal
    bundle_dir = _bundle(args.path)
    manifest = load_bundle(bundle_dir)
    deltas = [d for d in manifest["deltas"] if d["id"] == args.delta_id]
    if not deltas:
        print(f"unknown delta {args.delta_id!r}", file=sys.stderr)
        return 2
    d = deltas[0]
    print(f"delta {d['id']}: parent={d['parent']} blob={d['blob']}")
    print(f"  rank={d['rank']} alpha={d['alpha']} norm={d['norm']:.6f} "
          f"modules={d['target_modules']} bytes={d['bytes']}")
    print(f"  created_at={d['created_at']} self_edit={d['self_edit_id']}")
    journal = open_journal(bundle_dir, manifest)
    try:
        edits = journal.backend.find("self_edits",
                                     filter={"edit_id": d["self_edit_id"]}, limit=1)
        if edits:
            e = edits[0]
            print(f"  reason: {e['reason']}")
            print(f"  gate_verdict={e['gate_verdict']} result={e['result']}")
            for c in (e.get("gate_report") or {}).get("checks", []):
                mark = "pass" if c["passed"] else "FAIL"
                print(f"    [{mark}] {c['name']}: {c['detail']}")
        prov = journal.backend.find("provenance",
                                    filter={"child": args.delta_id})
        for p in prov:
            print(f"  provenance: {p['relation']} parents={p['parents']} "
                  f"note={p['note']}")
    finally:
        journal.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="si", description="SI-0 reference runtime: the self-writing loop.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init", help="create a new .si bundle")
    pi.add_argument("path", help="bundle directory to create (e.g. ./sentinel-001.si)")
    pi.add_argument("--name", default="sentinel-001")
    pi.add_argument("--journal-backend", choices=["embedded", "nedb"],
                    default="embedded",
                    help="journal store: embedded sqlite (default, zero "
                         "services) or the real NEDB engine over pgwire "
                         "(requires: pip install -r requirements-nedb.txt)")
    pi.set_defaults(func=cmd_init)

    pc = sub.add_parser("cycle", help="run one self-write cycle")
    pc.add_argument("path", help="bundle directory")
    pc.add_argument("--mind", choices=["demo", "mock"], default="demo")
    pc.set_defaults(func=cmd_cycle)

    pr = sub.add_parser("rollback", help="move head back (blobs retained as evidence)")
    pr.add_argument("path", help="bundle directory")
    pr.add_argument("--to", required=True, help="delta id or 'base'")
    pr.add_argument("--reason", default=None)
    pr.add_argument("--mind", choices=["demo", "mock"], default="demo")
    pr.set_defaults(func=cmd_rollback)

    pl = sub.add_parser("log", help="show checkpoints (or rollbacks)")
    pl.add_argument("path", help="bundle directory")
    pl.add_argument("--rollbacks", action="store_true")
    pl.add_argument("--limit", type=int, default=20)
    pl.set_defaults(func=cmd_log)

    pd = sub.add_parser("diff", help="diff two points, e.g. base..d0001")
    pd.add_argument("path", help="bundle directory")
    pd.add_argument("range", help="<a>..<b> where a/b are base, a delta id, or head")
    pd.set_defaults(func=cmd_diff)

    pw = sub.add_parser("why", help="explain why a delta exists")
    pw.add_argument("path", help="bundle directory")
    pw.add_argument("delta_id")
    pw.set_defaults(func=cmd_why)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"si: error: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # TamperError etc: fail closed, loud
        print(f"si: FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
