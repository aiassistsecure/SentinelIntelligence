"""Gaia Q-4 measurement harness (SI-50 §16 Q-4; slice SI-54).

The SI-40 tradition, applied to the queue: scripted miniature
tasks — read-heavy, write-heavy, mixed, dependency-chained —
run under the serial cycle (``execute_tool_cycle``, one intent
gated and executed at a time: the pre-queue loop's shape) and
under the queue (batch admission via
:func:`si.gaia.drain.admit_tool_intent`, then
:meth:`si.gaia.drain.Drainer.drain_task`) at read bounds
1/2/4/8, on this host, CPU only, against a fresh miniature
bundle per repetition so journal growth is attributable to
exactly one task.

What is measured, and only this:

- **wall times** — admission wall, drain wall, and total wall
  per task (queue mode); per-cycle and total wall (serial
  mode). Throughput is derived arithmetic (intents / total
  wall), labelled derived wherever it is reported.
- **queue residence** — per intent, claim timestamp minus
  admission timestamp, both read back from the journal fold
  (``intent_ts`` / ``claim_ts``, SI-50 §12.2's own definition
  of residence). Distributions are reported as min / median /
  mean / p95 / max over the task's intents, then aggregated
  over repetitions.
- **journal growth per task** — journal tip sequence delta
  and per-collection row deltas over the task, plus the
  journal sqlite file's byte growth. The queue's extra
  bookkeeping (§18.4) is therefore a number, not a worry.
- **leash cost (SI-53 §2's open measurement)** — (a) the
  overhead of an installed leash check on a short
  execute-class run, in two forms: a trivial ``False`` check
  (the 50 ms poll slice alone) and the drainer's real
  fold-based check; (b) end-to-end stop latency: the wall
  time from an operator stop being journaled against a live
  drain (a ``shell_exec`` sleeping 30 s in flight) to the
  drain returning with that execution killed.
- **the two-deep pipeline (§9.2)** — total wall when batch
  N+1's admission overlaps batch N's drain (one
  ``shell_exec`` sleeping 1 s) versus the same work run
  strictly sequentially (drain, then admit, then drain).

What is NOT measured here stays UNMEASURED: the real Hearth
path (SI-45's label, unchanged), NEDB-backend queue
performance (every run below uses the embedded journal
backend, like Q-1..Q-3's suites), any non-miniature bundle,
and any workload other than the four scripted ones. Numbers
produced on a contended host move; the harness reports
ranges and medians over repetitions and never a single
run's figure as the result.

The recommendation helper (:func:`recommend_read_bound`) is
pure arithmetic over measured medians: the smallest read
bound whose read-heavy median total wall is within a stated
tolerance of the best bound's median — parsimony, because
§8.2's justification for a bound is reviewability as much
as speed, and a bound that buys nothing measurable does
not earn its concurrency.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import statistics
import sys
import threading
import time
from datetime import datetime, timezone

#: The workloads of SI-50 §16 Q-4, in the spec's order.
WORKLOADS = ("read-heavy", "write-heavy", "mixed",
             "dependency-chained")

#: The read bounds §16 Q-4 sweeps.
READ_BOUNDS = (1, 2, 4, 8)

#: Deterministic payload sizes for the scripted workloads:
#: read sources and write contents are both 2,048 bytes, so
#: a workload's per-intent tool work is uniform by
#: construction and wall differences are the machinery's,
#: not the payloads'.
PAYLOAD_BYTES = 2048


# -- small statistics helpers (reported, never invented) ---------------------

def _dist(values):
    """A distribution summary of measured ``values`` (seconds
    or counts): n / min / median / mean / p95 / max. p95 is
    nearest-rank. An empty input yields all-None fields —
    the harness never fabricates a number for a run that
    produced no samples."""
    vals = sorted(float(v) for v in values)
    if not vals:
        return {"n": 0, "min": None, "median": None, "mean": None,
                "p95": None, "max": None}
    rank = max(1, math.ceil(0.95 * len(vals)))
    return {
        "n": len(vals),
        "min": vals[0],
        "median": statistics.median(vals),
        "mean": statistics.fmean(vals),
        "p95": vals[rank - 1],
        "max": vals[-1],
    }


def _parse_ts(raw):
    """Parse a journal timestamp (``...Z``, microsecond
    precision as written by the journal) to an aware datetime,
    or None when unparseable — residence is then unmeasured
    for that intent, never guessed."""
    if not raw or not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def recommend_read_bound(medians, tolerance=0.10):
    """Recommend a §8.2 default from measured read-heavy
    medians: the smallest bound whose median total wall is
    within ``tolerance`` (relative) of the best median.
    ``medians`` maps bound -> median seconds. Pure function
    of its inputs; ties and missing bounds are handled by
    ignoring non-positive / non-finite entries, and an empty
    map recommends nothing (None) rather than guessing."""
    clean = {int(b): float(v) for b, v in (medians or {}).items()
             if v is not None and float(v) > 0
             and math.isfinite(float(v))}
    if not clean:
        return None
    best = min(clean.values())
    for bound in sorted(clean):
        if clean[bound] <= best * (1.0 + tolerance):
            return bound
    return None  # unreachable: the best bound qualifies itself


# -- one measured task ---------------------------------------------------------

def _intent(workload, index, fixture, prev_seq=None):
    from si.gate import Intent
    if workload == "read-heavy":
        tool, args = "fs_read", {"path": f"src-{index:02d}.txt"}
    elif workload == "write-heavy":
        tool, args = "fs_write", {
            "path": f"out-{index:02d}.txt",
            "content": (f"q4-write-{index:02d}-" * 128)[:PAYLOAD_BYTES]}
    elif workload == "mixed":
        if index % 2 == 0:
            tool, args = "fs_read", {"path": f"src-{index:02d}.txt"}
        else:
            tool, args = "fs_write", {
                "path": f"out-{index:02d}.txt",
                "content": (f"q4-mixed-{index:02d}-" * 128)[:PAYLOAD_BYTES]}
    elif workload == "dependency-chained":
        tool, args = "fs_read", {"path": f"src-{index:02d}.txt"}
    else:  # pragma: no cover - guarded by WORKLOADS at call sites
        raise ValueError(f"unknown workload {workload!r}")
    kw = {}
    if prev_seq is not None:
        kw["depends_on"] = [{"seq": prev_seq, "requirement": "success"}]
    return Intent(kind="tool", tool=tool, args=args,
                  reason=f"Q-4 scripted {workload} step {index}",
                  parents=[fixture["ref"]],
                  source_seqs=[fixture["seq"]], **kw)


def _journal_bytes(bundle_dir):
    total = 0
    jdir = os.path.join(bundle_dir, "journal")
    for name in os.listdir(jdir):
        path = os.path.join(jdir, name)
        if os.path.isfile(path):
            total += os.path.getsize(path)
    return total


def _setup_bundle(root, label):
    """A fresh miniature bundle + work root under ``root``.
    Returns (bundle_dir, work, manifest, journal, fixture,
    gate, ctx). The caller closes the journal."""
    from si.bundle import init_bundle, load_bundle
    from si.gaia.context import ToolContext
    from si.gaia.corpus import corpus_policy
    from si.gaia.shadow import record_task_fixture
    from si.gate import Gate
    from si.journal import open_journal
    from si.tiny_seed import seed_tiny_bundle

    bundle_dir = init_bundle(os.path.join(root, f"{label}.si"),
                             name=f"gaia-q4-{label}",
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    work = os.path.join(root, f"{label}-work")
    os.makedirs(work)
    payload = ("q4 payload line\n" * 128)[:PAYLOAD_BYTES]
    for i in range(64):
        with open(os.path.join(work, f"src-{i:02d}.txt"), "w") as f:
            f.write(payload)
    manifest = load_bundle(bundle_dir)
    journal = open_journal(bundle_dir, manifest)
    fixture = record_task_fixture(journal, f"Q-4 {label}", work)
    gate = Gate(corpus_policy(work))
    ctx = ToolContext(tool_work_root=work,
                      tool_task_seq=fixture["seq"],
                      tool_live_execution=True)
    return bundle_dir, work, manifest, journal, fixture, gate, ctx


def _growth(journal, bundle_dir, before_tip, before_bytes,
            before_counts):
    counts = {}
    for coll in ("self_edits", "experiences"):
        counts[coll] = len(journal.backend.find(coll)) \
            - before_counts[coll]
    return {
        "journal_entries": journal.tip()[0] - before_tip,
        "rows": counts,
        "journal_bytes": _journal_bytes(bundle_dir) - before_bytes,
    }


def _snapshot(journal, bundle_dir):
    counts = {coll: len(journal.backend.find(coll))
              for coll in ("self_edits", "experiences")}
    return journal.tip()[0], _journal_bytes(bundle_dir), counts


def run_task_once(root, label, workload, mode, bound=1, size=16):
    """Run one scripted task once and return its raw record.

    ``mode`` is ``"serial"`` (the serial cycle) or ``"queue"``
    (batch admission + one drain at read bound ``bound``).
    Every intent must reach ``executed``; the record carries
    the executed count so a shortfall is visible in the data,
    never hidden by the aggregation."""
    from si.gaia.corpus import corpus_policy
    from si.gaia.drain import Drainer, admit_tool_intent
    from si.gaia.executor import execute_tool_cycle
    from si.gaia.queue import fold_queue, record_queue_batch

    bundle_dir, work, manifest, journal, fixture, gate, ctx = \
        _setup_bundle(root, label)
    try:
        before = _snapshot(journal, bundle_dir)
        record = {"workload": workload, "mode": mode,
                  "bound": bound if mode == "queue" else None,
                  "size": size}
        if mode == "serial":
            t0 = time.perf_counter()
            executed = 0
            for i in range(size):
                out = execute_tool_cycle(
                    journal, bundle_dir, manifest,
                    _intent(workload, i, fixture), gate, ctx)
                if (out.get("execution") or {}).get("status") \
                        == "completed":
                    executed += 1
            record["total_wall_s"] = time.perf_counter() - t0
            record["admit_wall_s"] = None
            record["drain_wall_s"] = None
            record["executed"] = executed
            record["residence_s"] = _dist([])
        elif mode == "queue":
            policy = corpus_policy(work)
            policy["queue_max_concurrent_reads"] = bound
            batch = record_queue_batch(
                journal, fixture["seq"], fixture["ref"],
                declared_size=size)
            t_admit = time.perf_counter()
            prev_seq = None
            admitted = []
            for i in range(size):
                if workload == "dependency-chained" \
                        and prev_seq is not None:
                    ctx.tool_journal_tip = journal.tip()[0]
                    ctx.tool_task_intent_seqs = {prev_seq}
                out = admit_tool_intent(
                    journal, gate, ctx, _intent(workload, i, fixture, prev_seq),
                    batch_seq=batch["seq"])
                admitted.append(out)
                if workload == "dependency-chained":
                    rows = journal.backend.find(
                        "self_edits",
                        filter={"edit_id": out["edit_id"]}, limit=1)
                    prev_seq = rows[0]["seq"]
            admit_wall = time.perf_counter() - t_admit
            drainer = Drainer(journal, bundle_dir, manifest,
                              policy, work)
            t_drain = time.perf_counter()
            res = drainer.drain_task(fixture["seq"])
            drain_wall = time.perf_counter() - t_drain
            fold = fold_queue(journal, task_seq=fixture["seq"])
            residences = []
            executed = 0
            for state in fold.intents.values():
                if state.state == "executed":
                    executed += 1
                start, end = _parse_ts(state.intent_ts), \
                    _parse_ts(state.claim_ts)
                if start is not None and end is not None:
                    residences.append((end - start).total_seconds())
            record["total_wall_s"] = admit_wall + drain_wall
            record["admit_wall_s"] = admit_wall
            record["drain_wall_s"] = drain_wall
            record["executed"] = executed
            record["sync_reason"] = res["sync_reason"]
            record["residence_s"] = _dist(residences)
            record["_residences"] = residences
        else:
            raise ValueError(f"unknown mode {mode!r}")
        record["growth"] = _growth(journal, bundle_dir, *before)
        return record
    finally:
        journal.close()


# -- the leash measurements (SI-53 §2) -----------------------------------------

def measure_leash(root, reps=5, stop_reps=3):
    """Leash poll overhead and end-to-end stop latency.

    Overhead: one admitted ``shell_exec`` (``sleep 0.3``)
    executed via the executor with (a) no leash check,
    (b) a trivial installed check, (c) the drainer's real
    fold-based check — per-repetition execution wall.
    Stop latency: a live drain of a ``shell_exec`` sleeping
    30 s; an operator stop is journaled 0.5 s in; the latency
    is stop-journaled -> drain-returned wall, and the killed
    execution's journaled status is carried in the record."""
    from si.gaia.corpus import corpus_policy
    from si.gaia.drain import Drainer, admit_tool_intent
    from si.gaia.executor import Executor
    from si.gaia.queue import (
        fold_queue, record_queue_batch, record_queue_stop)

    overhead = {"none": [], "trivial": [], "fold": []}
    for rep in range(reps):
        for variant in ("none", "trivial", "fold"):
            label = f"leash-{variant}-{rep}"
            bundle_dir, work, manifest, journal, fixture, gate, \
                ctx = _setup_bundle(root, label)
            try:
                batch = record_queue_batch(
                    journal, fixture["seq"], fixture["ref"],
                    declared_size=1)
                out = admit_tool_intent(
                    journal, gate, ctx,
                    _intent_shell("sleep 0.3", fixture), batch_seq=batch["seq"])
                executor = Executor(bundle_dir, manifest, journal,
                                    corpus_policy(work), work)
                if variant == "trivial":
                    executor.leash_check = lambda: False
                elif variant == "fold":
                    task_seq = fixture["seq"]

                    def _fold_check(journal=journal, task_seq=task_seq):
                        fold = fold_queue(journal, task_seq=task_seq)
                        slot = fold.tasks.get(task_seq) or {}
                        return bool(slot.get("stop_in_force"))
                    executor.leash_check = _fold_check
                t0 = time.perf_counter()
                execution = executor.execute_journaled(out["edit_id"])
                overhead[variant].append(time.perf_counter() - t0)
                assert execution.get("status") == "completed", execution
            finally:
                journal.close()

    stop_latency = []
    stop_status = []
    for rep in range(stop_reps):
        label = f"leash-stop-{rep}"
        bundle_dir, work, manifest, journal, fixture, gate, ctx = \
            _setup_bundle(root, label)
        try:
            batch = record_queue_batch(
                journal, fixture["seq"], fixture["ref"],
                declared_size=1)
            admit_tool_intent(
                journal, gate, ctx,
                _intent_shell("echo started; sleep 30; echo finished", fixture),
                batch_seq=batch["seq"])
            drainer = Drainer(journal, bundle_dir, manifest,
                              corpus_policy(work), work)
            box = {}
            thread = threading.Thread(
                target=lambda: box.setdefault(
                    "res", drainer.drain_task(fixture["seq"])))
            thread.start()
            time.sleep(0.5)
            t_stop = time.perf_counter()
            record_queue_stop(journal, fixture["seq"],
                              cause="q4-measurement")
            thread.join(timeout=30)
            stop_latency.append(time.perf_counter() - t_stop)
            executions = (box.get("res") or {}).get("executions") or {}
            stop_status.append(
                [e.get("status") for e in executions.values()])
        finally:
            journal.close()
    return {
        "overhead_wall_s": {k: _dist(v) for k, v in overhead.items()},
        "overhead_raw_s": overhead,
        "stop_latency_s": _dist(stop_latency),
        "stop_latency_raw_s": stop_latency,
        "stop_execution_statuses": stop_status,
        "sleep_seconds": 0.3,
        "poll_slice_seconds": 0.05,
    }


def _intent_shell(command, fixture):
    from si.gate import Intent
    return Intent(kind="tool", tool="shell_exec",
                  args={"command": command},
                  reason="Q-4 leash measurement command",
                  parents=[fixture["ref"]],
                  source_seqs=[fixture["seq"]])


# -- the two-deep pipeline (§9.2) ----------------------------------------------

def measure_pipeline(root, reps=3):
    """Two-deep admission overlap (§9): batch A is one
    ``shell_exec`` sleeping 1 s; batch B is 8 reads. Sequential
    total = drain A, admit B, drain B. Pipelined total = admit
    B while A's drain runs in a thread, then drain B. Both
    totals are measured walls over the same admitted work."""
    from si.gaia.corpus import corpus_policy
    from si.gaia.drain import Drainer, admit_tool_intent
    from si.gaia.queue import record_queue_batch

    def _admit_batch(journal, gate, ctx, fixture, size, first=None):
        batch = record_queue_batch(
            journal, fixture["seq"], fixture["ref"],
            declared_size=size + (1 if first else 0))
        if first is not None:
            admit_tool_intent(journal, gate, ctx, first,
                              batch_seq=batch["seq"])
        edits = []
        for i in range(size):
            out = admit_tool_intent(
                journal, gate, ctx, _intent("read-heavy", i, fixture),
                batch_seq=batch["seq"])
            edits.append(out["edit_id"])
        return edits

    sequential, pipelined = [], []
    for rep in range(reps):
        for mode, sink in (("seq", sequential), ("pipe", pipelined)):
            label = f"pipe-{mode}-{rep}"
            bundle_dir, work, manifest, journal, fixture, gate, \
                ctx = _setup_bundle(root, label)
            try:
                drainer = Drainer(journal, bundle_dir, manifest,
                                  corpus_policy(work), work)
                _admit_batch(journal, gate, ctx, fixture, 0,
                             first=_intent_shell("sleep 1", fixture))
                if mode == "seq":
                    t0 = time.perf_counter()
                    drainer.drain_task(fixture["seq"])
                    _admit_batch(journal, gate, ctx, fixture, 8)
                    drainer.drain_task(fixture["seq"])
                    sequential.append(time.perf_counter() - t0)
                else:
                    t0 = time.perf_counter()
                    box = {}
                    thread = threading.Thread(
                        target=lambda: box.setdefault(
                            "res", drainer.drain_task(fixture["seq"])))
                    thread.start()
                    _admit_batch(journal, gate, ctx, fixture, 8)
                    thread.join(timeout=30)
                    drainer.drain_task(fixture["seq"])
                    pipelined.append(time.perf_counter() - t0)
            finally:
                journal.close()
    return {"sequential_wall_s": _dist(sequential),
            "pipelined_wall_s": _dist(pipelined),
            "sequential_raw_s": sequential,
            "pipelined_raw_s": pipelined}


# -- aggregation + CLI -----------------------------------------------------------

def aggregate(records):
    """Aggregate raw per-repetition records into the reported
    distributions, grouped by (workload, mode, bound). All
    figures are arithmetic over the raw records."""
    groups = {}
    for rec in records:
        key = (rec["workload"], rec["mode"], rec["bound"])
        groups.setdefault(key, []).append(rec)
    out = []
    for (workload, mode, bound), recs in sorted(
            groups.items(), key=lambda kv: (kv[0][0], kv[0][1],
                                            kv[0][2] or 0)):
        totals = [r["total_wall_s"] for r in recs]
        residences = []
        for r in recs:
            residences.extend(r.get("_residences") or [])
        growth_entries = [r["growth"]["journal_entries"] for r in recs]
        growth_bytes = [r["growth"]["journal_bytes"] for r in recs]
        size = recs[0]["size"]
        out.append({
            "workload": workload, "mode": mode, "bound": bound,
            "size": size, "reps": len(recs),
            "total_wall_s": _dist(totals),
            "admit_wall_s": _dist(
                [r["admit_wall_s"] for r in recs
                 if r["admit_wall_s"] is not None]),
            "drain_wall_s": _dist(
                [r["drain_wall_s"] for r in recs
                 if r["drain_wall_s"] is not None]),
            "throughput_intents_per_s_derived":
                _dist([size / t for t in totals if t > 0]),
            "residence_s": _dist(residences),
            "executed_min": min(r["executed"] for r in recs),
            "journal_entries_per_task": _dist(growth_entries),
            "journal_bytes_per_task": _dist(growth_bytes),
            "journal_entries_per_intent_derived":
                _dist([g / size for g in growth_entries]),
        })
    return out


def run_measurement(root, size=16, reps=5, bounds=READ_BOUNDS,
                    workloads=WORKLOADS, leash_reps=5,
                    stop_reps=3, pipeline_reps=3):
    """The full Q-4 measurement: every workload under the
    serial cycle and under the queue at each bound, plus the
    leash and pipeline measurements. Returns the complete
    record (conditions, raw records, aggregates, and the
    §8.2 recommendation derived from the read-heavy medians)."""
    os.makedirs(root, exist_ok=True)
    raw = []
    counter = 0
    for workload in workloads:
        rec = run_task_once(root, f"m{counter:03d}-serial",
                            workload, "serial", size=size)
        counter += 1
        raw.append(rec)
        for _ in range(reps - 1):
            raw.append(run_task_once(
                root, f"m{counter:03d}-serial", workload,
                "serial", size=size))
            counter += 1
        for bound in bounds:
            for _ in range(reps):
                raw.append(run_task_once(
                    root, f"m{counter:03d}-q{bound}", workload,
                    "queue", bound=bound, size=size))
                counter += 1
    leash = measure_leash(root, reps=leash_reps, stop_reps=stop_reps)
    pipeline = measure_pipeline(root, reps=pipeline_reps)
    aggregates = aggregate(raw)
    read_medians = {
        row["bound"]: row["total_wall_s"]["median"]
        for row in aggregates
        if row["workload"] == "read-heavy" and row["mode"] == "queue"}
    return {
        "conditions": {
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "cpu_count_os": os.cpu_count(),
            "journal_backend": "embedded (sqlite)",
            "bundle": "miniature (tiny seed, by construction)",
            "size_per_task": size,
            "repetitions": reps,
            "read_bounds": list(bounds),
            "payload_bytes": PAYLOAD_BYTES,
            "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "aggregates": aggregates,
        "raw": [{k: v for k, v in r.items() if k != "_residences"}
                | {"residences_s": r.get("_residences")}
                for r in raw],
        "leash": leash,
        "pipeline": pipeline,
        "recommendation": {
            "read_bound": recommend_read_bound(read_medians),
            "read_heavy_queue_medians_s": {
                str(k): v for k, v in sorted(read_medians.items())},
            "tolerance": 0.10,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="si.gaia.measure",
        description="Gaia Q-4 measurement harness (SI-50 §16).")
    parser.add_argument("--root", default=None,
                        help="scratch root for miniature bundles "
                             "(default: a fresh temp dir, removed "
                             "on exit unless --keep)")
    parser.add_argument("--out", default=None,
                        help="write the measurement record JSON here")
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args(argv)
    root = args.root
    cleanup = False
    if root is None:
        import tempfile
        root = tempfile.mkdtemp(prefix="si-q4-measure-")
        cleanup = not args.keep
    try:
        result = run_measurement(root, size=args.size, reps=args.reps)
    finally:
        if cleanup:
            shutil.rmtree(root, ignore_errors=True)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
