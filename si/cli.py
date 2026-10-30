"""Command-line interface for the SI-0 reference runtime.

    si init <path> [--name NAME]        create a new .si bundle
    si seed <path> <safetensors> [--by-reference]  import a real checkpoint as the frozen base
    si seed-tiny <path>                 seed the SI-14 constructed miniature + probe artifact
    si cycle <path> [--mind demo|mock]  run one self-write cycle
    si run <path> --cycles N [--mind demo|mock]  autonomous multi-cycle run
    si rollback <path> --to <id|base> [--reason R] [--mind demo|mock]
    si log <path> [--rollbacks] [--limit N]
    si diff <path> <a>..<b>
    si why <path> <delta-id>
    si export <path> --out FILE [--quant Q4_K_M]
    si squash <path> [--keep N] [--rank R] [--tol T]
    si serve <path> [--host H] [--port P]
    si task <path> --instruction "..." [--root DIR] [--cycles N] [--script FILE] [--operator NAME]
    si amend-constitution <path> --from-file F --reason R --operator NAME
    si admit-transformer-delta <path> --record F --blob F --operator NAME --reason R [--parents a,b] [--source-seqs 1,2] [--intent-reason R]

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
          "(see base.source in manifest.json); use `si seed` to import a "
          "real safetensors checkpoint as the frozen base")
    if args.journal_backend == "embedded":
        print("note: journal backend is EMBEDDED SQLITE (fallback); the "
              "default is the real NEDB engine")
    return 0


def cmd_seed(args) -> int:
    from si.seed import seed_bundle
    res = seed_bundle(args.path, args.safetensors, importer="cli",
                      by_reference=getattr(args, "by_reference", False))
    base = res["base"]
    if res.get("already_seeded"):
        print("already seeded: this checkpoint is already the bundle's "
              "base (idempotent no-op; zero bytes written)")
    print(f"seeded base: {base['blob']}")
    print(f"parts: {res['parts']}  bytes: {res['bytes']}  head: {res['head']}")
    if "tensors" in res:
        print(f"tensors: {res['tensors']}  "
              f"deduplicated_bytes: {res['deduplicated_bytes']}")
    if base.get("seed_format") == "reference-v1":
        print(f"seed_format: reference-v1 (by-reference: "
              f"{res.get('stored_bytes', 0)} tensor bytes stored in the "
              f"bundle; tensors are referenced in place)")
        print("warning: this base DEPENDS on the source checkpoint "
              "remaining at its recorded path — moving or deleting it "
              "breaks the base (by design: SI never duplicates bytes)")
    print(f"source: {base['source']}")
    if base.get("checkpoint_config") is not None:
        print(f"arch: config.json captured "
              f"(arch_source=checkpoint-config, "
              f"sha256={base.get('checkpoint_config_sha256', '')[:19]}...)")
    else:
        print("arch: no config.json found; export will use "
              "tensor-shape inference (arch_source=tensor-shape-inference)")
    for name, info in sorted(base["tensors"].items()):
        print(f"  tensor {name}: shape={info['shape']} dtype={info['dtype']}")
    return 0


def cmd_seed_tiny(args) -> int:
    from si.tiny_seed import seed_tiny_bundle
    res = seed_tiny_bundle(args.path, importer="cli")
    base = res["base"]
    if res.get("already_seeded"):
        print("already seeded: this miniature is already the bundle's "
              "base (idempotent no-op; zero bytes written)")
    print(f"seeded base: {base['blob']}")
    print(f"tensors: {res['tensors']}  elements: {res['elements']}  "
          f"bytes: {res['bytes']}  "
          f"deduplicated_bytes: {res['deduplicated_bytes']}")
    print(f"probe artifact: probe_suite.json "
          f"sha256={res['probe_artifact_sha256']}")
    print(f"base echo accuracy: {res['echo_accuracy']} "
          f"(seed-time measured; floor is 15/16)")
    print(f"identity_tol: {res['identity_tol']}  "
          f"capability_eps: {res['capability_eps']} "
          f"(min(0.25, 0.4*min_margin), recorded in policy.json)")
    print(f"source: {base['source']}")
    print("note: probes are RECORDED, not consumed — no gate, loop, "
          "or serve path reads them in SI-14; transformer deltas "
          "still report checks 3/4 as not evaluated and escalate")
    return 0


def cmd_run(args) -> int:
    from si.run import run_cycles
    bundle_dir = _bundle(args.path)
    mind = _make_mind(args.mind, bundle_dir)
    s = run_cycles(bundle_dir, mind, args.cycles)
    print(f"cycles:    {s['cycles_completed']}/{s['cycles_requested']}")
    print(f"admitted:  {s['admitted']}")
    print(f"rejected:  {s['rejected']}")
    print(f"escalated: {s['escalated']}")
    print(f"halted:    {s['halted']}"
          + (f" ({s['halt_verdict']}, edit {s['halt_edit_id']})"
             if s["halted"] else ""))
    print(f"head:      {s['head']}")
    print(f"delta stack depth: {s['delta_stack_depth']}")
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
        # SI-42: a tool intent has no delta — `si why <edit-id>`
        # walks the tool self-edit's provenance to its task entry
        # (RFC SI-41 §5.2 check 5, §16 criterion 7).
        if str(args.delta_id).startswith("se_"):
            return _cmd_why_tool(bundle_dir, manifest, args.delta_id)
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


def _cmd_why_tool(bundle_dir: str, manifest: dict, edit_id: str) -> int:
    """`si why` for a tool self-edit (SI-42): intent, gate checks,
    provenance edges, and the walk to the task entry it chains to."""
    from si.gaia.shadow import walk_tool_provenance
    from si.journal import open_journal
    journal = open_journal(bundle_dir, manifest)
    try:
        try:
            walk = walk_tool_provenance(journal, edit_id)
        except KeyError:
            print(f"unknown edit {edit_id!r}", file=sys.stderr)
            return 2
        e = walk["edit"]
        intent = e.get("intent") or {}
        if e.get("kind") != "tool":
            print(f"edit {edit_id!r} is a {e.get('kind')!r} self-edit, "
                  "not a tool intent; `si why` walks deltas and tool "
                  "intents", file=sys.stderr)
            return 2
        print(f"tool intent {edit_id}: tool={intent.get('tool')} "
              f"args={intent.get('args')}")
        print(f"  reason: {e['reason']}")
        print(f"  gate_verdict={e['gate_verdict']} result={e['result']}")
        report = e.get("gate_report") or {}
        if report.get("execution"):
            print(f"  execution: {report['execution']}")
        for c in report.get("checks", []):
            mark = "pass" if c["passed"] else "FAIL"
            label = f" [{c['label']}]" if c.get("label") else ""
            print(f"    [{mark}] {c['name']}{label}: {c['detail']}")
        for p in walk["provenance"]:
            print(f"  provenance: {p['relation']} parents={p['parents']} "
                  f"note={p['note']}")
        task = walk["task"]
        if task is not None:
            row = task["row"]
            record = row.get("record") or {}
            print(f"  task: seq={row['seq']} collection={task['collection']} "
                  f"source={record.get('source')}")
            print(f"    instruction: {record.get('instruction')}")
            print(f"    work_root: {record.get('work_root')}")
        else:
            print("  task: none — the cited source seqs reach no task "
                  "entry (reported as none, not invented)")
        outcomes = journal.backend.find(
            "experiences", filter={"self_edit_id": edit_id})
        for o in outcomes:
            rec = o.get("record") or {}
            execution = rec.get("execution") or {}
            if execution:
                print(f"  outcome: seq={o['seq']} kind={o.get('kind')} "
                      f"execution={execution.get('execution')}")
    finally:
        journal.close()
    return 0


def cmd_export(args) -> int:
    from si.export import export_bundle
    res = export_bundle(args.path, args.out, quant=args.quant)
    print(f"head:   {res['head']}")
    print(f"out:    {res['out_path']}")
    print(f"bytes:  {res['bytes']}")
    print(f"sha256: {res['sha256']}")
    print(f"quant:  {res['quant']}")
    return 0


def cmd_squash(args) -> int:
    from si.squash import squash_bundle
    res = squash_bundle(args.path, keep=args.keep, rank=args.rank,
                        tol=args.tol)
    if not res["squashed"]:
        if res.get("verdict"):
            print(f"verdict:   {res['verdict']}")
            print(f"edit_id:   {res['edit_id']}")
            print("squash not committed (rejection journaled; see si log)")
        return 0
    print(f"squashed into: {res['delta_id']}")
    print(f"verdict:        {res['verdict']}")
    print(f"edit_id:        {res['edit_id']}")
    print(f"head:           {res['head']} (unchanged by design)")
    return 0


def cmd_amend_constitution(args) -> int:
    from si.amend import amend_constitution
    res = amend_constitution(args.path, args.from_file, args.reason,
                             args.operator)
    print(f"amended:   {res['amended']}")
    print(f"old_hash:  sha256:{res['old_hash']}")
    print(f"new_hash:  sha256:{res['new_hash']}")
    print(f"operator:  {res['operator']}")
    print(f"edit_id:   {res['edit_id']}")
    print(f"head:      {res['head']} (unchanged by design)")
    print("note: --operator is an attribution/authorization record in "
          "the journal, NOT cryptographic authentication (see "
          "si/amend.py)")
    return 0


def cmd_admit_transformer_delta(args) -> int:
    import json as _json

    from si.admit import admit_transformer_delta
    from si.bundle import load_bundle
    from si.gate import Intent
    from si.transformer_delta import KIND as TF_KIND

    bundle_dir = _bundle(args.path)
    try:
        with open(args.record, "rb") as f:
            record = _json.loads(f.read().decode("utf-8"))
    except (OSError, ValueError) as e:
        print(f"si: error: cannot read delta record {args.record!r}: {e}",
              file=sys.stderr)
        return 1
    try:
        with open(args.blob, "rb") as f:
            blob_bytes = f.read()
    except OSError as e:
        print(f"si: error: cannot read delta blob {args.blob!r}: {e}",
              file=sys.stderr)
        return 1
    if not isinstance(record, dict):
        print("si: error: delta record file must contain a JSON object",
              file=sys.stderr)
        return 1

    # The intent the operator puts in front of the gate: its declared
    # target list is taken from the record itself (they must match —
    # the gate's provenance check refuses any disagreement, and a CLI
    # that let the operator type a second, divergent list would only
    # manufacture such disagreements). Parents default to the current
    # head, which is the delta's true parent at admission time.
    targets = [t.get("name") for t in record.get("targets") or []
               if isinstance(t, dict)]
    if args.parents:
        parents = [p.strip() for p in args.parents.split(",") if p.strip()]
    else:
        parents = [load_bundle(bundle_dir)["head"]]
    source_seqs = []
    if args.source_seqs:
        try:
            source_seqs = [int(s) for s in args.source_seqs.split(",")
                           if s.strip()]
        except ValueError:
            print("si: error: --source-seqs must be comma-separated "
                  "journal sequence numbers", file=sys.stderr)
            return 1
    intent = Intent(
        kind=TF_KIND,
        reason=args.intent_reason or args.reason or "",
        parents=parents, source_seqs=source_seqs, targets=targets)
    res = admit_transformer_delta(
        bundle_dir, {"record": record, "blob_bytes": blob_bytes},
        intent, operator=args.operator, reason=args.reason)
    print(f"admitted:  {res['admitted']}")
    print(f"delta_id:  {res['delta_id']} "
          f"(candidate: {res['candidate_delta_id']})")
    print(f"edit_id:   {res['edit_id']}")
    print(f"operator:  {res['operator']}")
    print(f"head:      {res['head']}")
    print(f"verified:  {res['verified']} "
          "(transformer-apply-smoke: shape + finiteness only)")
    print(f"rolled_back: {res['rolled_back']}")
    _probe_checks = {c["name"]: c for c in
                     res["gate_report"]["checks"]}
    if any(_probe_checks.get(n, {}).get("measured") is not None
           for n in ("identity_continuity", "capability_floor")):
        # SI-17: a miniature bundle's probes WERE evaluated for
        # real and the §6.4 verdict mapping is live — say so.
        print("note: for this miniature bundle the probe harness "
              "DID evaluate checks 3/4 for real (measured results, "
              "gate verdict "
              f"{res['gate_verdict']}); the measured results are "
              "journaled with this admission. This admission is "
              "the operator's recorded decision via the operator "
              "path. --operator is an attribution/authorization "
              "record in the journal, NOT cryptographic "
              "authentication (see si/admit.py)")
    elif any("blind-escalated" in _probe_checks.get(n, {}).get("detail", "")
             for n in ("identity_continuity", "capability_floor")):
        # SI-17 blind guard: targets not measurable — the operator
        # acted as the check for them; no measured pass exists.
        print("note: admission is the operator acting as the "
              "identity/capability check for target(s) the probe "
              "suites cannot measure (blind-escalated: drift would "
              "be 0.0 vacuously, which is not a measured pass). "
              "--operator is an attribution/authorization record "
              "in the journal, NOT cryptographic authentication "
              "(see si/admit.py)")
    else:
        print("note: admission is the operator acting as the "
              "identity/capability check — identity/capability probes "
              "were NOT evaluated (no transformer probe harness exists) "
              "and this is not a probe pass. --operator is an "
              "attribution/authorization record in the journal, NOT "
              "cryptographic authentication (see si/admit.py)")
    return 0


def cmd_task(args) -> int:
    """`si task` (SI-44/B-3, RFC §10.1): open an operator task on a
    bundle and run the B-3 task loop against it.

    B-3's proposer is scripted (§14.1): --script names a JSON file
    holding an array of intent dicts — {"kind": "tool", "tool": ...,
    "args": {...}, "reason": ...} tool intents, or {"kind":
    "memory", "record": {"kind": "task_complete", ...}} completion
    reports. parents/source_seqs may be omitted; the task runner
    fills the task chain for scripted intents that omit them.
    Without --script there is no honest proposer to run: no trained
    mind proposes tool intents yet (§14), so the CLI refuses rather
    than inventing a default mind.
    """
    if not args.script:
        print("si: error: `si task` in B-3 drives a scripted proposer "
              "(--script FILE): no trained mind proposes tool intents "
              "yet (RFC §14), and B-3 does not invent a default mind",
              file=sys.stderr)
        return 2
    from si.gaia.task import run_task
    from si.gate import Intent
    bundle_dir = _bundle(args.path)
    try:
        with open(args.script, "rb") as f:
            script = json.loads(f.read().decode("utf-8"))
    except (OSError, ValueError) as e:
        print(f"si: error: cannot read script {args.script!r}: {e}",
              file=sys.stderr)
        return 1
    if not isinstance(script, list) \
            or not all(isinstance(d, dict) for d in script):
        print("si: error: script file must contain a JSON array of "
              "intent objects", file=sys.stderr)
        return 1
    intents = []
    for d in script:
        kind = d.get("kind", "tool")
        intents.append(Intent(
            kind=kind,
            reason=d.get("reason", ""),
            parents=list(d.get("parents") or []),
            source_seqs=list(d.get("source_seqs") or []),
            tool=d.get("tool", "") if kind == "tool" else "",
            args=dict(d.get("args") or {}) if kind == "tool" else {},
            record=dict(d.get("record") or {}) if kind == "memory"
            else {}))
    res = run_task(bundle_dir, intents, args.instruction,
                   work_root=args.root, cycles=args.cycles,
                   operator=args.operator or "operator")
    print(f"task:      {res['task_ref']} (seq {res['task_seq']})")
    for i, step in enumerate(res["steps"], start=1):
        print(f"  step {i}: {step['edit_id']} tool={step['tool']} "
              f"verdict={step['verdict']} "
              f"class={step['effective_class']} "
              f"execution={step['execution_status']}")
    print(f"stop:      {res['stop_reason']} — {res['stop_detail']}")
    print(f"cycles:    {res['cycles_used']}   "
          f"result bytes: {res['result_bytes_used']}")
    print(f"head:      {res['head']}")
    return 0


def cmd_serve(args) -> int:
    from si.serve import serve_bundle
    serve_bundle(args.path, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="si", description="SI-0 reference runtime: the self-writing loop.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init", help="create a new .si bundle")
    pi.add_argument("path", help="bundle directory to create (e.g. ./sentinel-001.si)")
    pi.add_argument("--name", default="sentinel-001")
    pi.add_argument("--journal-backend", choices=["nedb", "embedded"],
                    default="nedb",
                    help="journal store: the real NEDB engine over pgwire "
                         "(default; requires: pip install -r "
                         "requirements-nedb.txt) or embedded sqlite "
                         "(zero-service fallback)")
    pi.set_defaults(func=cmd_init)

    ps = sub.add_parser("seed", help="import a real safetensors checkpoint "
                                     "as the bundle's frozen base")
    ps.add_argument("path", help="bundle directory")
    ps.add_argument("safetensors", help=".safetensors file or directory of "
                                        ".safetensors files")
    ps.add_argument("--by-reference", action="store_true",
                    help="SI-8: store ZERO tensor bytes — record "
                         "per-tensor references into the source "
                         "checkpoint instead of copying blobs. The "
                         "base then depends on the source remaining "
                         "at its recorded path (see README)")
    ps.set_defaults(func=cmd_seed)

    pst = sub.add_parser("seed-tiny", help="seed the SI-14 constructed "
                                           "miniature transformer "
                                           "(TinyTransformerMind) plus "
                                           "its probe artifact into a "
                                           "fresh bundle; constructed, "
                                           "NOT trained; probes are "
                                           "recorded, not consumed")
    pst.add_argument("path", help="bundle directory")
    pst.set_defaults(func=cmd_seed_tiny)

    prun = sub.add_parser("run", help="autonomous serial multi-cycle "
                                      "self-write run")
    prun.add_argument("path", help="bundle directory")
    prun.add_argument("--cycles", type=int, default=3,
                      help="self-write cycles to attempt (halts early on "
                           "REJECT or ESCALATE)")
    prun.add_argument("--mind", choices=["demo", "mock"], default="demo")
    prun.set_defaults(func=cmd_run)

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

    pw = sub.add_parser("why", help="explain why a delta exists, or "
                                    "walk a tool intent's provenance to "
                                    "its task (pass the tool self-edit id, "
                                    "e.g. se_0003)")
    pw.add_argument("path", help="bundle directory")
    pw.add_argument("delta_id")
    pw.set_defaults(func=cmd_why)

    pe = sub.add_parser("export", help="materialize the head and write it "
                                       "as a GGUF file")
    pe.add_argument("path", help="bundle directory")
    pe.add_argument("--out", required=True, help="output GGUF file path")
    pe.add_argument("--quant", default=None,
                    help="quantize via a real llama-quantize binary "
                         "(e.g. Q4_K_M); requires hearth runtime llama")
    pe.set_defaults(func=cmd_export)

    psq = sub.add_parser("squash", help="compress the oldest deltas on the "
                                        "active stack into one gated delta "
                                        "(head id unchanged)")
    psq.add_argument("path", help="bundle directory")
    psq.add_argument("--keep", type=int, default=3,
                     help="newest deltas to leave untouched (default: 3)")
    psq.add_argument("--rank", type=int, default=None,
                     help="truncated-SVD rank for the squashed delta "
                          "(default: max rank of the squashed deltas)")
    psq.add_argument("--tol", type=float, default=1e-4,
                     help="max abs diff vs the pre-squash head allowed "
                          "before anything is committed (default: 1e-4)")
    psq.set_defaults(func=cmd_squash)

    psv = sub.add_parser("serve", help="serve the bundle head for inference "
                                       "over local HTTP (read-only; demo "
                                       "mind only — seeded transformer "
                                       "bundles return 501 on /infer)")
    psv.add_argument("path", help="bundle directory")
    psv.add_argument("--host", default="127.0.0.1",
                     help="bind host (default: 127.0.0.1, localhost only; "
                          "no auth — do not expose beyond localhost)")
    psv.add_argument("--port", type=int, default=8000,
                     help="bind port (default: 8000; 0 = ephemeral)")
    psv.set_defaults(func=cmd_serve)

    ptk = sub.add_parser("task", help="SI-44/B-3: open an operator "
                                      "task on a bundle and run the "
                                      "Gaia task loop (RFC §10) with "
                                      "a scripted proposer (--script "
                                      "FILE, a JSON array of intent "
                                      "dicts); journals the task, "
                                      "every step, and the §10.4 "
                                      "stop. Live execution is "
                                      "miniature-scoped; demo "
                                      "bundles stay shadow.")
    ptk.add_argument("path", help="bundle directory")
    ptk.add_argument("--instruction", required=True,
                     help="the operator instruction that opens the "
                          "task (journaled before any cycle runs)")
    ptk.add_argument("--root", default=None,
                     help="work root for the task (default: the "
                          "bundle policy's tool_work_root, else the "
                          "bundle directory)")
    ptk.add_argument("--cycles", type=int, default=None,
                     help="cap on total tool cycles for this task "
                          "(the effective cap is the minimum of "
                          "this and the policy's "
                          "max_tool_cycles_per_task)")
    ptk.add_argument("--script", default=None,
                     help="JSON file: an array of intent dicts "
                          "({\"kind\": \"tool\", \"tool\": ..., "
                          "\"args\": {...}, \"reason\": ...} or "
                          "{\"kind\": \"memory\", \"record\": "
                          "{\"kind\": \"task_complete\", ...}}); "
                          "parents/source_seqs may be omitted — "
                          "the runner fills the task chain")
    ptk.add_argument("--operator", default=None,
                     help="operator name recorded on the task "
                          "entry (attribution record, not "
                          "cryptographic authentication)")
    ptk.set_defaults(func=cmd_task)

    pam = sub.add_parser("amend-constitution",
                         help="OPERATOR-ONLY: replace the gate constitution "
                              "with a new hash-versioned text (journaled; "
                              "head/deltas/base never move; autonomous "
                              "amendment still escalates at the gate). The "
                              "--operator flag is an attribution record, "
                              "not cryptographic authentication.")
    pam.add_argument("path", help="bundle directory")
    pam.add_argument("--from-file", required=True,
                     help="UTF-8 file containing the new constitution text")
    pam.add_argument("--reason", default=None,
                     help="why the constitution is being amended "
                          "(required, non-empty; journaled)")
    pam.add_argument("--operator", default=None,
                     help="operator name, recorded in the journal as the "
                          "authorization record (required, non-empty; an "
                          "attribution record, not cryptographic auth)")
    pam.set_defaults(func=cmd_amend_constitution)

    pad = sub.add_parser("admit-transformer-delta",
                         help="OPERATOR-ONLY: admit one escalated "
                              "transformer delta to the active chain "
                              "(re-validated at admission; REJECT-class "
                              "deltas are never admitted; the autonomous "
                              "loop has no path here and still halts on "
                              "ESCALATE). The --operator flag is an "
                              "attribution record, not cryptographic "
                              "authentication.")
    pad.add_argument("path", help="bundle directory")
    pad.add_argument("--record", required=True,
                     help="JSON file with the transformer-delta-v1 "
                          "manifest record (the escalated candidate)")
    pad.add_argument("--blob", required=True,
                     help="the candidate's .safetensors delta blob file")
    pad.add_argument("--operator", default=None,
                     help="operator name, recorded in the journal as the "
                          "authorization record (required, non-empty; an "
                          "attribution record, not cryptographic auth)")
    pad.add_argument("--reason", default=None,
                     help="why the operator is admitting this delta "
                          "(required, non-empty; journaled)")
    pad.add_argument("--parents", default=None,
                     help="comma-separated intent parents "
                          "(default: the current head)")
    pad.add_argument("--source-seqs", default=None,
                     help="comma-separated journal seqs the proposal is "
                          "grounded in (default: none)")
    pad.add_argument("--intent-reason", default=None,
                     help="reason recorded on the gate intent "
                          "(default: the admission --reason)")
    pad.set_defaults(func=cmd_admit_transformer_delta)
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
