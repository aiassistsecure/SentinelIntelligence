"""Gaia S-7 seam measurement harness (SI-62; design SI-61).

Implements docs/S-7-seam-measurement-design.md: it measures the
five merged SI-side seam hops (S-2 ingest, S-3 work submission,
S-4 leash, S-5 outcome return, S-6 event forwarding) on
miniature fixture bundles only, in evidence classes S (SI-side
only / replay transport) and L (real HttpTransport against a
loopback test double whose own service time is measured and
reported beside every result). Class F (live fork) is NOT
attempted: D-8's default is to complete in Class L, and the
fork half of S-6 (POST /gaia/v1/events/ingest) is UNBUILT, so
every fork-dependent quantity stays UNMEASURED by name in the
record's deferred list — never estimated, never faked.

Discipline (design §2/§3/§10/§11): durations are monotonic
(perf_counter_ns) SI-side durations; the two sequence spaces
(SI journal seq, fork spine seq) are never merged and no
cross-clock subtraction is presented as a latency (the two
labelled exceptions are same-host, same-clock-domain by
construction: the §7.5 conditional outcome latency against the
harness's own double, and the §8.4 journal-derived lag, whose
clock domain is named on the quantity). Correctness riders
ride every plan: a run whose riders fail is a failed run and
its timings are not publishable. The harness works only on
throwaway miniature bundles under a temp/scratch root, binds
loopback only, refuses non-loopback URLs in code (§10.5),
journals nothing of its own on any operator bundle, and never
touches the Glimmer bundle. D-1: harness-side timing only —
phases inside one merged function are reported as named
residuals, never by instrumenting the seam modules.

Run as: python -m si.gaia.seam_measure --out <dir>
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_SEED = 20261031
ADMIT_ALL = {"tool_class_policy": {
    "read": "admit", "write": "admit",
    "execute": "admit", "external": "admit"}}
REPLAY_URL = "replay://s7-measurement"

DEFERRED = [
    {"id": "F-A-EGRESS", "quantity": "fork egress send -> SI ACK received (Class F)",
     "reason": "deferred fork half (design §4.5); no live fork run (D-8)"},
    {"id": "F-B-ACCEPT", "quantity": "live fork work-acceptance 202 latency + fork queueing (Class F)",
     "reason": "deferred fork half (design §5.5); no live fork run (D-8)"},
    {"id": "F-C-ACTUATOR", "quantity": "stop journaled -> fork actuator refuses; fork staleness timing",
     "reason": "deferred fork half (design §6.5); SI-side components measured (Q-C1/Q-C3)"},
    {"id": "F-C-CADENCE", "quantity": "heartbeat interval/TTL (10 s / 30 s) adequacy",
     "reason": "fork-side policy constants (design §6.4, D-5): cost measured (Q-C4), adequacy not judged"},
    {"id": "F-D-LATENCY", "quantity": "unconditional outcome latency; Class F outcome latency",
     "reason": "undefined without a poll cadence (design §2.4/§7.5); only the conditional form is measured"},
    {"id": "F-E-INGEST", "quantity": "fork ingest service time per event (POST /gaia/v1/events/ingest)",
     "reason": "fork half UNBUILT (design §2.5/§8.5)"},
    {"id": "F-E-SSE", "quantity": "fork ingest -> source=si SSE visibility lag; merged-stream timing",
     "reason": "fork half UNBUILT (design §8.5)"},
    {"id": "F-A-CONC", "quantity": "Q-A7 concurrent-ingest contention observation",
     "reason": "deferred by D-6 default (out): the lock serialises by design"},
]


# -- statistics (design §3.4) -------------------------------------------------

def summarize(values):
    """Median/min/max/p95/N over measured values (native unit).
    p95 is nearest-rank, reported only when N >= 30 (§3.4);
    an empty input yields all-None — never a fabricated number."""
    vals = sorted(float(v) for v in values)
    if not vals:
        return {"n": 0, "min": None, "median": None, "mean": None,
                "p95": None, "max": None}
    rank = max(1, math.ceil(0.95 * len(vals)))
    return {"n": len(vals), "min": vals[0],
            "median": statistics.median(vals),
            "mean": statistics.fmean(vals),
            "p95": vals[rank - 1] if len(vals) >= 30 else None,
            "max": vals[-1]}


def _ms(seconds_list):
    return [s * 1000.0 for s in seconds_list]


def _timed(fn):
    t0 = time.perf_counter_ns()
    out = fn()
    return out, (time.perf_counter_ns() - t0) / 1e9


def _quantity(qid, kind, cls, fixture, unit, variant, raw,
              double_ms=None, notes="", derived=False):
    rec = {"id": qid, "kind": kind, "evidence_class": cls,
           "fixture": fixture, "unit": unit, "variant": variant,
           "stats": summarize(raw), "raw": [float(v) for v in raw],
           "exclusions": []}
    if double_ms is not None:
        rec["double_service_ms"] = summarize(_ms(double_ms))
        med = rec["stats"]["median"]
        dmed = rec["double_service_ms"]["median"]
        # D-11: double-bound when double median > 25% of round-trip.
        rec["double_bound"] = bool(
            med and dmed is not None and med > 0 and dmed > 0.25 * med)
    if derived:
        rec["derived"] = True
    if notes:
        rec["notes"] = notes
    return rec


class Riders:
    def __init__(self):
        self.items = []

    def check(self, name, passed, detail=""):
        self.items.append({"name": name, "passed": bool(passed),
                           "detail": str(detail)})
        return passed

    def all_passed(self):
        return all(r["passed"] for r in self.items)


# -- loopback discipline (design §10.5) ---------------------------------------

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}


def assert_loopback_url(url):
    """Refuse any http(s) URL whose host is not loopback, in
    code. replay:// labels name no host and pass (no network)."""
    if url is None:
        return
    if "://" not in url:
        raise ValueError(f"URL without scheme refused: {url!r}")
    scheme, rest = url.split("://", 1)
    if scheme in ("http", "https"):
        host = rest.split("/")[0].split(":")[0]
        if host not in _LOOPBACK_HOSTS:
            raise ValueError(
                f"non-loopback URL refused by the S-7 harness: {url!r}")
    elif scheme != "replay":
        raise ValueError(f"unknown URL scheme refused: {url!r}")


# -- environment fingerprint (design §12.1) -----------------------------------

def _si_commit(explicit=None):
    if explicit:
        return explicit
    root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", ".."))
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root,
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _ram_bytes():
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def environment_fingerprint(si_commit=None, seed=DEFAULT_SEED,
                            argv=None):
    return {
        "si_commit": _si_commit(si_commit),
        # Mandatory honesty (§12.1): Class S/L runs record the
        # literal string, never a fork SHA by association.
        "gaia_commit": "not-used",
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "venv_path": os.environ.get("VIRTUAL_ENV") or os.path.dirname(
            os.path.dirname(sys.executable)),
        "os": f"{platform.system()} {platform.release()} "
              f"({platform.platform()})",
        "cpu_model": _cpu_model(),
        "cpu_count_os": os.cpu_count(),
        "ram_bytes_total": _ram_bytes(),
        "journal_backend": "embedded",
        "host_note": f"{platform.node()}; dev-loop VM, shared with "
                     f"the agent runtime (contention note, §11.5)",
        "harness": {"module": "si.gaia.seam_measure",
                    "argv": list(argv) if argv is not None
                    else list(sys.argv),
                    "seed": seed},
    }


# -- loopback doubles (design §10.3) ------------------------------------------

class LoopbackDouble:
    """A loopback HTTP test double. ``handler(method, path, body,
    state) -> (status, obj)``. Service time (request parsed ->
    response written) is timed inside the double and logged per
    request, beside every Class L result (§10.3)."""

    def __init__(self, handler, delay_s=0.0):
        self.handler = handler
        self.delay_s = delay_s
        self.state = {}
        self.log = []          # {method, path, body, service_s, perf, wall}
        self.service_s = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _go(self, method):
                t0 = time.perf_counter_ns()
                wall0 = time.time()
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw) if raw else None
                except ValueError:
                    body = None
                if outer.delay_s:
                    time.sleep(outer.delay_s)
                status, obj = outer.handler(
                    method, self.path, body, outer.state)
                payload = json.dumps(obj).encode()
                # Log BEFORE the response is written, so a client
                # that has received the response can rely on the
                # log entry existing (no handler-thread race);
                # service_s is filled in after the write.
                entry = {
                    "method": method, "path": self.path, "body": body,
                    "service_s": None, "perf_s": t0 / 1e9,
                    "wall": wall0, "status": status}
                outer.log.append(entry)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                svc = (time.perf_counter_ns() - t0) / 1e9
                entry["service_s"] = svc
                outer.service_s.append(svc)

            def do_POST(self):  # noqa: N802 - stdlib name
                self._go("POST")

            def do_GET(self):  # noqa: N802 - stdlib name
                self._go("GET")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    @property
    def base(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def url(self, path):
        u = self.base + path
        assert_loopback_url(u)
        return u


def work_double(spellings=("work_id",), mode="success"):
    """S-3 contract double: 202 + work ID (scriptable failures)."""
    counter = {"n": 0}

    def handler(method, path, body, state):
        counter["n"] += 1
        if mode == "non202":
            return 500, {"error": "double scripted non-202"}
        if mode == "noid":
            return 202, {"accepted": True}
        key = spellings[(counter["n"] - 1) % len(spellings)]
        return 202, {key: f"work-{counter['n']:04d}"}
    return handler


def leash_double(applied=True):
    def handler(method, path, body, state):
        last = (body or {}).get("journal_seq",
                                (body or {}).get("leash_tip", 0))
        return 200, {"applied": applied, "last_leash_seq": last}
    return handler


def status_double(answers):
    """answers: work_id -> (status, body). Default: awaiting."""
    def handler(method, path, body, state):
        wid = path.rstrip("/").rsplit("/", 1)[-1]
        if wid in answers:
            return answers[wid]
        return 200, {"work_id": wid, "state": "admitted"}
    return handler


def ingest_double(fail_at=None):
    """S-6 contract double: 200 per event, 409 on duplicate seq."""
    seen = set()

    def handler(method, path, body, state):
        seq = (body or {}).get("seq")
        if fail_at is not None and len(state.get("ok", [])) == fail_at:
            return 500, {"error": "double scripted 500"}
        if seq in seen:
            return 409, {"error": "duplicate seq"}
        seen.add(seq)
        state.setdefault("ok", []).append(seq)
        return 200, {"status": "ok"}
    handler.seen = seen
    return handler


# -- fixtures (design §10.2) ----------------------------------------------------

def _new_bundle(root, label):
    from si.bundle import init_bundle
    from si.tiny_seed import seed_tiny_bundle
    bundle_dir = init_bundle(os.path.join(root, f"{label}.si"),
                             name=f"gaia-s7-{label}",
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    work = os.path.join(root, f"{label}-work")
    os.makedirs(work, exist_ok=True)
    return bundle_dir, work


def _watcher(seq, tool="hearth.restart", model="muse", hash_=None):
    h = hash_ or f"hash-s7-{seq}"
    return {"model": model, "tool": tool,
            "args": {"model": model,
                     "spine_event": {"hash": h, "seq": seq}},
            "reason": f"s7 fixture proposal {seq}",
            "caused_by": {"hash": h, "seq": seq}}


def _counts(bundle_dir):
    from si.bundle import load_bundle
    from si.journal import COLLECTIONS, open_journal
    journal = open_journal(bundle_dir, load_bundle(bundle_dir))
    try:
        return {c: journal.backend.count(c) for c in COLLECTIONS}
    finally:
        journal.close()


def _tip(bundle_dir):
    from si.bundle import load_bundle
    from si.journal import open_journal
    journal = open_journal(bundle_dir, load_bundle(bundle_dir))
    try:
        return journal.tip()[0]
    finally:
        journal.close()


def _open(bundle_dir):
    from si.bundle import load_bundle
    from si.journal import open_journal
    return open_journal(bundle_dir, load_bundle(bundle_dir))


def _leash_act(bundle_dir, fn, task_seq, **kwargs):
    from si.bundle import _anchor_checkpoint, _write_manifest, load_bundle
    manifest = load_bundle(bundle_dir)
    journal = _open(bundle_dir)
    try:
        entry = fn(journal, task_seq, **kwargs)
        updated = _anchor_checkpoint(
            journal, manifest, manifest["head"], note="s7 leash act")
        _write_manifest(bundle_dir, updated)
        return entry
    finally:
        journal.close()


def _admit(bundle_dir, work, seq, **kw):
    from si.gaia.corpus import corpus_policy
    from si.gaia.seam import ingest_proposal
    out = ingest_proposal(bundle_dir, _watcher(seq, **kw),
                          policy=corpus_policy(work, ADMIT_ALL))
    assert out["status"] == "acked" and out["verdict"] == "ADMIT", out
    return out


def _submit(bundle_dir, edit_id, work_id):
    from si.gaia.hearth import ReplayTransport
    from si.gaia.seam_submit import submit_seam_edit
    out = submit_seam_edit(
        bundle_dir, edit_id, submission_url=REPLAY_URL,
        transport=ReplayTransport([(202, {"work_id": work_id})]))
    assert out["status"] == "submitted", out
    return out


def _result_for(wid, detail="executed (s7 double)"):
    return {"outcome": "executed", "tool": "hearth.restart",
            "model": "muse", "detail": detail,
            "caused_by": {"hash": "hash-s7-result", "seq": 1},
            "spine_event_refs": []}


def _outcome_line(wid, result):
    return json.dumps({
        "work_id": wid, "tool": result["tool"],
        "model": result["model"], "outcome": result["outcome"],
        "refusal": None, "detail": result["detail"],
        "caused_by": result["caused_by"],
        "spine_event_refs": result["spine_event_refs"],
        "result": result})


class Cfg:
    def __init__(self, reps=30, pass_reps=10, catchup_reps=5,
                 warmup=3, seed=DEFAULT_SEED, ev_big=500, ev_mid=100,
                 pending_big=50, poll_big=50, reconcile_big=10000):
        self.reps = reps
        self.pass_reps = pass_reps
        self.catchup_reps = catchup_reps
        self.warmup = warmup
        self.seed = seed
        self.ev_big = ev_big
        self.ev_mid = ev_mid
        self.pending_big = pending_big
        self.poll_big = poll_big
        self.reconcile_big = reconcile_big


def _warm(fn, n=3):
    for _ in range(n):
        fn()


# -- Hop A: S-2 ingest (design §4) ---------------------------------------------

def plan_a(root, cfg, riders):
    from si.gaia.corpus import corpus_policy
    from si.gaia.hearth import HttpTransport
    from si.gaia.seam import (
        find_prior_proposal_ack, ingest_proposal, validate_proposal)
    from si.gaia.service import make_service_server
    qs = []
    seq_counter = {"n": 1000}

    def nxt():
        seq_counter["n"] += 1
        return seq_counter["n"]

    # Q-A1 validation (valid + refused), interleaved.
    valid = _watcher(41)
    bad = _watcher(41)
    bad["caused_by"] = {"hash": "hash-s7-41", "seq": "41"}
    raw = {"valid": [], "refused": []}
    _warm(lambda: validate_proposal(copy.deepcopy(valid)), cfg.warmup)
    for _ in range(cfg.reps):
        for name, payload in (("valid", valid), ("refused", bad)):
            _, dt = _timed(lambda p=payload: validate_proposal(
                copy.deepcopy(p)))
            raw[name].append(dt)
    ok, problems = validate_proposal(copy.deepcopy(valid))
    riders.check("A1-valid-fixture-validates", ok is not None)
    ok2, probs2 = validate_proposal(copy.deepcopy(bad))
    riders.check("A1-refused-fixture-refused",
                 ok2 is None and bool(probs2))
    for name in ("valid", "refused"):
        qs.append(_quantity("Q-A1", "phase", "S", "watcher-fixture",
                            "ms", name, _ms(raw[name])))

    # Pre-grown bundles for the size axis + Q-A2 dedup scan.
    grown = {}
    for size in (0, 10, 50):
        bd, work = _new_bundle(root, f"a-grow{size}")
        pol = corpus_policy(work)
        for i in range(size):
            ingest_proposal(bd, _watcher(2000 + size * 100 + i),
                            policy=pol)
        grown[size] = (bd, work, pol)
    raw_scan = {}
    for size, (bd, work, pol) in grown.items():
        journal = _open(bd)
        try:
            first_seq = 2000 + size * 100
            target = {"hash": f"hash-s7-{first_seq}", "seq": first_seq} \
                if size else {"hash": "hash-none", "seq": 999999}
            _warm(lambda: find_prior_proposal_ack(journal, target),
                  cfg.warmup)
            vals = []
            for _ in range(cfg.reps):
                _, dt = _timed(lambda: find_prior_proposal_ack(
                    journal, target))
                vals.append(dt)
            raw_scan[size] = vals
            if size:
                riders.check(f"A2-scan-finds-prior@{size}",
                             find_prior_proposal_ack(journal, target)
                             is not None)
        finally:
            journal.close()
        qs.append(_quantity("Q-A2", "phase", "S",
                            f"miniature-pre{size}", "ms",
                            f"prior-edits-{size}", _ms(vals)))

    # Q-A3 gate evaluation at the exposed boundary (Gate.evaluate
    # with the ingest's own context construction; D-1 harness-side).
    from si.bundle import load_bundle
    from si.gaia.context import ToolContext
    from si.gaia.policy import resolve_tool_policy
    from si.gaia.seam import find_seam_task
    from si.gate import Gate, Intent
    from si.probes import ProbeContext, is_miniature_probe_bundle
    gate_variants = [
        ("ADMIT-fs_list", "fs_list", {"path": "notes"}, None, "ADMIT"),
        ("ESCALATE-hearth.restart", "hearth.restart",
         {"model": "muse", "spine_event": {"hash": "h", "seq": 1}},
         None, "ESCALATE"),
        ("REJECT-hearth.reap", "hearth.reap",
         {"model": "muse", "spine_event": {"hash": "h", "seq": 1}},
         None, "REJECT"),
        ("ADMIT-hearth.restart", "hearth.restart",
         {"model": "muse", "spine_event": {"hash": "h", "seq": 1}},
         ADMIT_ALL, "ADMIT"),
    ]
    gate_medians = {}
    for vi, (name, tool, args, over, expect) in enumerate(gate_variants):
        bd, work = _new_bundle(root, f"a-gate{vi}")
        ingest_proposal(bd, _watcher(3000 + vi),
                        policy=corpus_policy(work))
        manifest = load_bundle(bd)
        pol = corpus_policy(work, over)
        gate = Gate(pol)
        tool_pol = resolve_tool_policy(pol)
        root_dir = os.path.realpath(
            tool_pol.get("tool_work_root") or bd)
        journal = _open(bd)
        try:
            task = find_seam_task(journal, "muse")
            if is_miniature_probe_bundle(manifest):
                ctx = ProbeContext(bd, manifest, policy=pol)
            else:
                ctx = ToolContext(bundle_dir=bd)
            ctx.tool_work_root = root_dir
            ctx.tool_task_seq = task["seq"]

            def evaluate():
                intent = Intent(
                    kind="tool", tool=tool, args=copy.deepcopy(args),
                    reason="s7 gate timing",
                    parents=[task["ref"]], source_seqs=[task["seq"]])
                return gate.evaluate(intent, None, ctx)
            _warm(evaluate, cfg.warmup)
            vals, verdicts = [], set()
            for _ in range(cfg.reps):
                rep, dt = _timed(evaluate)
                vals.append(dt)
                verdicts.add(rep.verdict)
            riders.check(f"A3-verdict-{name}",
                         verdicts == {expect}, str(verdicts))
            gate_medians[name] = statistics.median(_ms(vals))
            qs.append(_quantity("Q-A3", "phase", "S", "watcher-fixture",
                                "ms", name, _ms(vals)))
        finally:
            journal.close()

    # Q-A5 invocation by verdict class (interleaved bundles) +
    # the ESCALATE journal-size axis at pre-10 / pre-50.
    a5_variants = [
        ("ESCALATE", _watcher(0), None),
        ("ADMIT-external", _watcher(0), ADMIT_ALL),
        ("ADMIT-fs_list", None, None),
        ("REJECT", None, None),
    ]
    bundles = {}
    for vi, (name, _, _) in enumerate(a5_variants):
        bundles[name] = _new_bundle(root, f"a5-{vi}")
    raw_a5 = {name: [] for name, _, _ in a5_variants}
    expect_a5 = {"ESCALATE": "ESCALATE", "ADMIT-external": "ADMIT",
                 "ADMIT-fs_list": "ADMIT", "REJECT": "REJECT"}
    for rep in range(cfg.reps):
        for name, _, over in a5_variants:
            bd, work = bundles[name]
            s = nxt()
            if name == "ADMIT-fs_list":
                payload = {"model": "muse", "tool": "fs_list",
                           "args": {"path": "notes"},
                           "reason": "s7 control",
                           "caused_by": {"hash": f"hash-s7-{s}",
                                         "seq": s}}
            elif name == "REJECT":
                payload = _watcher(s, tool="hearth.reap")
            else:
                payload = _watcher(s)
            before = _counts(bd)["self_edits"]
            out, dt = _timed(lambda: ingest_proposal(
                bd, payload, policy=corpus_policy(work, over)))
            raw_a5[name].append(dt)
            riders.check(f"A5-growth-{name}-rep{rep}",
                         out["verdict"] == expect_a5[name]
                         and _counts(bd)["self_edits"] == before + 1,
                         out.get("verdict"))
    for name, vals in raw_a5.items():
        qs.append(_quantity("Q-A5", "invocation", "S",
                            "miniature-fresh-caused_by", "ms", name,
                            _ms(vals)))
    for size in (10, 50):
        bd, work, pol = grown[size]
        vals = []
        for _ in range(cfg.reps):
            s = nxt()
            out, dt = _timed(lambda: ingest_proposal(
                bd, _watcher(s), policy=pol))
            vals.append(dt)
            assert out["verdict"] == "ESCALATE"
        qs.append(_quantity("Q-A5", "invocation", "S",
                            f"miniature-pre{size}", "ms",
                            f"ESCALATE-pre{size}", _ms(vals)))
    # Q-A4: journal + re-anchor as the named residual (§3.3, D-1).
    resid = (statistics.median(_ms(raw_a5["ESCALATE"]))
             - summarize(_ms(raw["valid"]))["median"]
             - summarize(_ms(raw_scan[0]))["median"]
             - gate_medians["ESCALATE-hearth.restart"])
    qs.append(_quantity(
        "Q-A4", "phase", "S", "miniature-fresh-caused_by", "ms",
        "journal+reanchor-residual", [resid],
        notes="derived residual: median Q-A5(ESCALATE) minus the "
              "medians of Q-A1(valid), Q-A2(0 prior) and "
              "Q-A3(ESCALATE) measured at exposed boundaries on "
              "separate fixtures (D-1 harness-side timing); a "
              "single derived value, not a distribution",
        derived=True))

    # Q-A6: HTTP ingest through the real B-5 service handler.
    bd, work = _new_bundle(root, "a6-http")
    server = make_service_server(bd, host="127.0.0.1", port=0,
                                 policy=corpus_policy(work))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{port}/gaia/proposals"
        assert_loopback_url(url)
        transport = HttpTransport()
        _warm(lambda: transport("POST", url,
                                json_body=_watcher(nxt())), cfg.warmup)
        vals, acks = [], 0
        for _ in range(cfg.reps):
            s = nxt()
            (status, rawb), dt = _timed(lambda: transport(
                "POST", url, json_body=_watcher(s)))
            vals.append(dt)
            if status == 200 and json.loads(rawb)["verdict"] == "ESCALATE":
                acks += 1
        riders.check("A6-http-acks", acks == cfg.reps, f"{acks}")
        qs.append(_quantity(
            "Q-A6", "invocation", "L", "miniature-service-handler",
            "ms", "POST /gaia/proposals", _ms(vals),
            notes="double is the real B-5 service handler itself; "
                  "its in-process cost distribution is Q-A5 "
                  "(ESCALATE) — the HTTP+JSON share is the "
                  "difference of those distributions (§4.2)"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    # Q-A7: serial throughput, windows of `reps` sequential ingests.
    thr = []
    for w in range(cfg.catchup_reps):
        bd, work = _new_bundle(root, f"a7-{w}")
        pol = corpus_policy(work)
        _warm(lambda: ingest_proposal(bd, _watcher(nxt()), policy=pol),
              cfg.warmup)

        def window():
            for _ in range(cfg.reps):
                ingest_proposal(bd, _watcher(nxt()), policy=pol)
        _, dt = _timed(window)
        thr.append(cfg.reps / dt)
    qs.append(_quantity("Q-A7", "pass-throughput", "S",
                        "miniature-serial-caller", "items/s",
                        f"window-{cfg.reps}", thr))

    # Q-A8: dedup retry — zero growth rider.
    bd, work = _new_bundle(root, "a8-dedup")
    pol = corpus_policy(work)
    first = ingest_proposal(bd, _watcher(777), policy=pol)
    before = _counts(bd)
    _warm(lambda: ingest_proposal(bd, _watcher(777), policy=pol),
          cfg.warmup)
    vals, same = [], True
    for _ in range(cfg.reps):
        out, dt = _timed(lambda: ingest_proposal(
            bd, _watcher(777), policy=pol))
        vals.append(dt)
        same = same and out["deduped"] and out["ack"] == first["ack"]
    riders.check("A8-dedup-zero-growth",
                 same and _counts(bd) == before)
    qs.append(_quantity("Q-A8", "invocation", "S",
                        "miniature-dedup-retry", "ms", "retry", _ms(vals)))
    return qs


# -- Hop B: S-3 work submission (design §5) -------------------------------------

def plan_b(root, cfg, riders):
    from si.gaia.corpus import corpus_policy
    from si.gaia.hearth import HttpTransport, ReplayTransport
    from si.gaia.seam import ingest_proposal
    from si.gaia.seam_leash import seam_leash_block
    from si.gaia.seam_submit import (
        _eligibility_problems, build_work_payload, submit_pending_seam_work,
        submit_seam_edit)
    from si.gaia.queue import record_queue_stop
    qs = []
    seq = {"n": 5000}

    def nxt():
        seq["n"] += 1
        return seq["n"]

    # Fixture bundle with one admitted edit.
    bd, work = _new_bundle(root, "b-core")
    admitted = _admit(bd, work, nxt())
    journal = _open(bd)
    try:
        row = journal.backend.find(
            "self_edits", filter={"edit_id": admitted["edit_id"]},
            limit=1)[0]
        # Q-B1 eligible.
        _warm(lambda: (_eligibility_problems(row),
                       seam_leash_block(journal, row)), cfg.warmup)
        vals = []
        for _ in range(cfg.reps):
            _, dt = _timed(lambda: (_eligibility_problems(row),
                                    seam_leash_block(journal, row)))
            vals.append(dt)
        riders.check("B1-eligible-no-block",
                     seam_leash_block(journal, row) is None
                     and _eligibility_problems(row) == [])
        qs.append(_quantity("Q-B1", "phase", "S", "miniature-admit",
                            "ms", "eligible", _ms(vals)))
        # Q-B2 payload construction.
        _warm(lambda: build_work_payload(row, leash_tip=1), cfg.warmup)
        vals = []
        for _ in range(cfg.reps):
            _, dt = _timed(lambda: build_work_payload(row, leash_tip=1))
            vals.append(dt)
        qs.append(_quantity("Q-B2", "phase", "S", "miniature-admit",
                            "ms", "payload", _ms(vals)))
    finally:
        journal.close()
    # Q-B1 leash-held variant.
    _leash_act(bd, record_queue_stop, admitted["task_seq"],
               cause="s7-stop", detail="s7 leash-held fixture")
    journal = _open(bd)
    try:
        row = journal.backend.find(
            "self_edits", filter={"edit_id": admitted["edit_id"]},
            limit=1)[0]
        vals = []
        for _ in range(cfg.reps):
            _, dt = _timed(lambda: (_eligibility_problems(row),
                                    seam_leash_block(journal, row)))
            vals.append(dt)
        riders.check("B1-leash-held-blocks",
                     seam_leash_block(journal, row) is not None)
        qs.append(_quantity("Q-B1", "phase", "S", "miniature-leash-held",
                            "ms", "leash-held", _ms(vals)))
    finally:
        journal.close()

    # Q-B3 transport leg: loopback double, 3 spellings interleaved,
    # replay floor, and the D-4 5 ms-delay sensitivity variant.
    payload = build_work_payload(row, leash_tip=1)
    spellings = ("work_id", "id", "workId")
    with LoopbackDouble(work_double(spellings)) as dbl:
        url = dbl.url("/work")
        transport = HttpTransport()
        _warm(lambda: transport("POST", url, json_body=payload),
              cfg.warmup)
        per = {s: [] for s in spellings}
        for rep in range(cfg.reps * len(spellings)):
            _, dt = _timed(lambda: transport(
                "POST", url, json_body=payload))
            per[spellings[rep % len(spellings)]].append(dt)
        for s in spellings:
            qs.append(_quantity("Q-B3", "invocation", "L",
                                "loopback-work-double", "ms",
                                f"spelling-{s}", _ms(per[s]),
                                double_ms=dbl.service_s))
    replay = ReplayTransport([(202, {"work_id": "work-r"})] * (
        cfg.reps + cfg.warmup))
    _warm(lambda: replay("POST", REPLAY_URL, json_body=payload),
          cfg.warmup)
    vals = []
    for _ in range(cfg.reps):
        _, dt = _timed(lambda: replay("POST", REPLAY_URL,
                                      json_body=payload))
        vals.append(dt)
    qs.append(_quantity("Q-B3", "invocation", "S",
                        "replay-transport-floor", "ms", "replay",
                        _ms(vals)))
    replay_median = statistics.median(_ms(vals))
    with LoopbackDouble(work_double(spellings), delay_s=0.005) as dbl:
        url = dbl.url("/work")
        transport = HttpTransport()
        _warm(lambda: transport("POST", url, json_body=payload),
              cfg.warmup)
        vals = []
        for _ in range(cfg.reps):
            _, dt = _timed(lambda: transport(
                "POST", url, json_body=payload))
            vals.append(dt)
        qs.append(_quantity("Q-B3", "invocation", "L",
                            "loopback-work-double-delay5ms", "ms",
                            "delay-5ms-sensitivity", _ms(vals),
                            double_ms=dbl.service_s,
                            notes="D-4 sensitivity variant: double "
                                  "adds a fixed 5 ms service delay"))

    # Q-B5 success, Class L and Class S companion.
    b5_medians = {}
    for cls, label in (("L", "loopback"), ("S", "replay")):
        bd2, work2 = _new_bundle(root, f"b5-{label}")
        edits = [_admit(bd2, work2, nxt())["edit_id"]
                 for _ in range(cfg.reps + cfg.warmup)]
        if cls == "L":
            dbl = LoopbackDouble(work_double())
            dbl.__enter__()
            url = dbl.url("/work")
            kw = {"submission_url": url}
        else:
            dbl = None
            kw = {"submission_url": REPLAY_URL,
                  "transport": ReplayTransport(
                      [(202, {"work_id": f"work-s{i}"})
                       for i in range(cfg.reps + cfg.warmup)])}
        try:
            for e in edits[:cfg.warmup]:
                submit_seam_edit(bd2, e, **kw)
            vals = []
            for e in edits[cfg.warmup:]:
                out, dt = _timed(lambda e=e: submit_seam_edit(
                    bd2, e, **kw))
                vals.append(dt)
                assert out["status"] == "submitted", out
        finally:
            if dbl is not None:
                dbl.__exit__()
        riders.check(f"B5-{label}-all-submitted",
                     _counts(bd2)["experiences"] >= cfg.reps)
        b5_medians[cls] = statistics.median(_ms(vals))
        qs.append(_quantity("Q-B5", "invocation", cls,
                            "miniature-admit", "ms", f"success-{label}",
                            _ms(vals),
                            double_ms=dbl.service_s if dbl else None))
    # Q-B5 failure paths (one edit, retries free, nothing journaled).
    bd3, work3 = _new_bundle(root, "b5-fail")
    fail_edit = _admit(bd3, work3, nxt())["edit_id"]
    failures = [
        ("non-202", ReplayTransport(
            [(500, {"error": "no"})] * (cfg.reps + cfg.warmup)),
         "submission-failed"),
        ("202-no-work-id", ReplayTransport(
            [(202, {"accepted": True})] * (cfg.reps + cfg.warmup)),
         "submission-unconfirmed"),
        ("transport-exception", ReplayTransport(
            [RuntimeError("down")] * (cfg.reps + cfg.warmup)),
         "submission-failed"),
    ]
    for name, transport, expect in failures:
        _warm(lambda: submit_seam_edit(
            bd3, fail_edit, submission_url=REPLAY_URL,
            transport=transport), cfg.warmup)
        vals, statuses = [], set()
        for _ in range(cfg.reps):
            out, dt = _timed(lambda: submit_seam_edit(
                bd3, fail_edit, submission_url=REPLAY_URL,
                transport=transport))
            vals.append(dt)
            statuses.add(out["status"])
        riders.check(f"B5-failure-{name}-journals-nothing",
                     statuses == {expect}
                     and not _open_rows(bd3, "gaia_work"))
        qs.append(_quantity("Q-B5", "invocation", "S",
                            "miniature-admit", "ms", name, _ms(vals)))
    # Q-B4 residual (Class S companion chain).
    resid = (b5_medians["S"] - summarize(qs[0]["raw"])["median"]
             - summarize(qs[1]["raw"])["median"] - replay_median)
    qs.append(_quantity(
        "Q-B4", "phase", "S", "miniature-admit", "ms",
        "journal-submission-residual", [resid],
        notes="derived residual: median Q-B5(replay) minus medians "
              "of Q-B1(eligible), Q-B2 and Q-B3(replay) (D-1)",
        derived=True))
    # Q-B7 dedup short-circuit: zero transport calls.
    counting = ReplayTransport([])
    vals = []
    for _ in range(cfg.reps):
        # bd2's first edit was submitted in the B5 warmup.
        out, dt = _timed(lambda: submit_seam_edit(
            bd2, edits[0], submission_url=REPLAY_URL,
            transport=counting))
        vals.append(dt)
        assert out["deduped"] is True
    riders.check("B7-dedup-zero-transport", counting.calls == [])
    qs.append(_quantity("Q-B7", "invocation", "S", "miniature-admit",
                        "ms", "dedup", _ms(vals)))
    # Q-B6 pending scan at 1 / 10 / big spendable (+ non-spendable).
    for k in dict.fromkeys((1, 10, cfg.pending_big)):
        durs, thr = [], []
        for p in range(cfg.pass_reps):
            bdp, workp = _new_bundle(root, f"b6-{k}-{p}")
            for _ in range(k):
                _admit(bdp, workp, nxt())
            esc = ingest_proposal(
                bdp, _watcher(nxt(), model="other"),
                policy=corpus_policy(workp))
            assert esc["verdict"] == "ESCALATE"
            transport = ReplayTransport(
                [(202, {"work_id": f"work-p{i}"}) for i in range(k)])
            out, dt = _timed(lambda: submit_pending_seam_work(
                bdp, submission_url=REPLAY_URL, transport=transport))
            riders.check(f"B6-count-{k}-pass{p}", out["count"] == k,
                         str(out["count"]))
            durs.append(dt)
            thr.append(k / dt if dt else 0.0)
        qs.append(_quantity("Q-B6", "invocation", "S",
                            f"miniature-pending-{k}", "ms",
                            f"pass-{k}-spendable", _ms(durs)))
        qs.append(_quantity("Q-B6", "pass-throughput", "S",
                            f"miniature-pending-{k}", "items/s",
                            f"pass-{k}-spendable", thr))
    return qs


def _open_rows(bundle_dir, kind):
    journal = _open(bundle_dir)
    try:
        return journal.backend.find("experiences",
                                    filter={"kind": kind})
    finally:
        journal.close()


# -- Hop C: S-4 leash (design §6) -----------------------------------------------

def plan_c(root, cfg, riders):
    from si.gaia.hearth import HttpTransport
    from si.gaia.queue import record_queue_resume, record_queue_stop
    from si.gaia.seam_leash import (
        leash_records, leash_tip, seam_leash_block, seam_tasks,
        send_leash_heartbeat, send_leash_snapshot, sync_seam_leash)
    qs = []
    seq = {"n": 9000}

    def nxt():
        seq["n"] += 1
        return seq["n"]

    # Q-C1 fold in three journal states on one bundle/journal.
    bd, work = _new_bundle(root, "c-core")
    admitted = _admit(bd, work, nxt())
    journal = _open(bd)
    try:
        row = journal.backend.find(
            "self_edits", filter={"edit_id": admitted["edit_id"]},
            limit=1)[0]
        vals = []
        for _ in range(cfg.reps):
            _, dt = _timed(lambda: seam_leash_block(journal, row))
            vals.append(dt)
        qs.append(_quantity("Q-C1", "phase", "S", "miniature-seam-task",
                            "ms", "no-leash-record", _ms(vals)))
    finally:
        journal.close()
    _leash_act(bd, record_queue_stop, admitted["task_seq"],
               cause="s7-stop", detail="s7 c1 stop")
    journal = _open(bd)
    try:
        vals = []
        for _ in range(cfg.reps):
            _, dt = _timed(lambda: seam_leash_block(journal, row))
            vals.append(dt)
        riders.check("C1-stop-in-force",
                     seam_leash_block(journal, row) is not None)
        qs.append(_quantity("Q-C1", "phase", "S", "miniature-seam-task",
                            "ms", "stop-in-force", _ms(vals)))
    finally:
        journal.close()
    _leash_act(bd, record_queue_resume, admitted["task_seq"],
               detail="s7 c1 resume")
    journal = _open(bd)
    try:
        vals = []
        for _ in range(cfg.reps):
            _, dt = _timed(lambda: seam_leash_block(journal, row))
            vals.append(dt)
        riders.check("C1-resumed-unheld",
                     seam_leash_block(journal, row) is None)
        qs.append(_quantity("Q-C1", "phase", "S", "miniature-seam-task",
                            "ms", "stop-then-resume", _ms(vals)))
    finally:
        journal.close()

    # Q-C2 derivation scaling: leash entries 0 / 10 / 100; models 1 vs 10.
    for n_entries in (0, 10, 100):
        bd2, work2 = _new_bundle(root, f"c2-{n_entries}")
        adm = _admit(bd2, work2, nxt())
        for i in range(n_entries):
            if i % 2 == 0:
                _leash_act(bd2, record_queue_stop, adm["task_seq"],
                           cause=f"s7-{i}", detail="s7 c2")
            else:
                _leash_act(bd2, record_queue_resume, adm["task_seq"],
                           detail="s7 c2")
        journal = _open(bd2)
        try:
            task = {"task_seq": adm["task_seq"]}
            for fn, qid_name in ((leash_records, "leash_records"),
                                 (leash_tip, "leash_tip")):
                _warm(lambda: fn(journal, task["task_seq"])
                      if fn is leash_records else fn(journal, task),
                      cfg.warmup)
                vals = []
                for _ in range(cfg.reps):
                    if fn is leash_records:
                        _, dt = _timed(lambda: fn(
                            journal, task["task_seq"]))
                    else:
                        _, dt = _timed(lambda: fn(journal, task))
                    vals.append(dt)
                riders.check(f"C2-count-{qid_name}-{n_entries}",
                             len(leash_records(journal, task["task_seq"]))
                             == n_entries if fn is leash_records else True)
                qs.append(_quantity("Q-C2", "phase", "S",
                                    f"miniature-leash-{n_entries}",
                                    "ms", qid_name, _ms(vals)))
        finally:
            journal.close()
    for n_models in (1, 10):
        bd3, work3 = _new_bundle(root, f"c2-models{n_models}")
        for m in range(n_models):
            from si.gaia.corpus import corpus_policy
            from si.gaia.seam import ingest_proposal
            ingest_proposal(bd3, _watcher(nxt(), model=f"m{m}"),
                            policy=corpus_policy(work3))
        journal = _open(bd3)
        try:
            vals = []
            for _ in range(cfg.reps):
                _, dt = _timed(lambda: seam_tasks(journal))
                vals.append(dt)
            riders.check(f"C2-seam-tasks-{n_models}",
                         len(seam_tasks(journal)) == n_models)
            qs.append(_quantity("Q-C2", "phase", "S",
                                f"miniature-models-{n_models}", "ms",
                                "seam_tasks-scan", _ms(vals)))
        finally:
            journal.close()

    # Q-C3 per-record crossing (incremental sync), watermark variant,
    # transport leg, pass, and gap pass.
    with LoopbackDouble(leash_double(applied=True)) as dbl:
        url = dbl.url("/gaia/v1/leash")
        bd4, work4 = _new_bundle(root, "c3-single")
        adm4 = _admit(bd4, work4, nxt())
        vals = []
        for rep in range(cfg.reps):
            if rep % 2 == 0:
                _leash_act(bd4, record_queue_stop, adm4["task_seq"],
                           cause="s7-c3", detail="s7 c3")
            else:
                _leash_act(bd4, record_queue_resume, adm4["task_seq"],
                           detail="s7 c3")
            out, dt = _timed(lambda: sync_seam_leash(
                bd4, leash_url=url))
            vals.append(dt)
            assert out["count"] == 1, out
        riders.check("C3-single-crossings-journaled",
                     len(_open_rows(bd4, "gaia_leash")) == cfg.reps)
        qs.append(_quantity("Q-C3", "invocation", "L",
                            "loopback-leash-double", "ms",
                            "per-record-applied-true", _ms(vals),
                            double_ms=dbl.service_s))
        transport_leg = []
        transport = HttpTransport()
        rec = {"model": "muse", "kind": "stop", "journal_seq": 1,
               "reason": "s7 transport leg"}
        for _ in range(cfg.reps):
            _, dt = _timed(lambda: transport(
                "POST", url, json_body=rec))
            transport_leg.append(dt)
        qs.append(_quantity("Q-C3", "invocation", "L",
                            "loopback-leash-double", "ms",
                            "transport-leg-only", _ms(transport_leg),
                            double_ms=dbl.service_s,
                            notes="journaling share of per-record "
                                  "crossing is the difference of the "
                                  "per-record and transport-leg "
                                  "distributions (§3.3)"))
    with LoopbackDouble(leash_double(applied=False)) as dbl:
        url = dbl.url("/gaia/v1/leash")
        bd5, work5 = _new_bundle(root, "c3-watermark")
        adm5 = _admit(bd5, work5, nxt())
        vals = []
        for rep in range(cfg.reps):
            if rep % 2 == 0:
                _leash_act(bd5, record_queue_stop, adm5["task_seq"],
                           cause="s7-c3w", detail="s7 c3 watermark")
            else:
                _leash_act(bd5, record_queue_resume, adm5["task_seq"],
                           detail="s7 c3 watermark")
            out, dt = _timed(lambda: sync_seam_leash(
                bd5, leash_url=url))
            vals.append(dt)
            assert out["count"] == 1 and \
                out["crossed"][0]["applied"] is False, out
        riders.check("C3-watermark-confirmed-crossings",
                     len(_open_rows(bd5, "gaia_leash")) == cfg.reps)
        qs.append(_quantity("Q-C3", "invocation", "L",
                            "loopback-leash-double", "ms",
                            "watermark-applied-false", _ms(vals),
                            double_ms=dbl.service_s))
    # Pass over 10 records, and the gap pass.
    with LoopbackDouble(leash_double(applied=True)) as dbl:
        url = dbl.url("/gaia/v1/leash")
        durs, thr = [], []
        for p in range(cfg.pass_reps):
            bd6, work6 = _new_bundle(root, f"c3-pass{p}")
            adm6 = _admit(bd6, work6, nxt())
            for i in range(10):
                if i % 2 == 0:
                    _leash_act(bd6, record_queue_stop, adm6["task_seq"],
                               cause="s7-pass", detail="s7 c3 pass")
                else:
                    _leash_act(bd6, record_queue_resume,
                               adm6["task_seq"], detail="s7 c3 pass")
            out, dt = _timed(lambda: sync_seam_leash(
                bd6, leash_url=url))
            riders.check(f"C3-pass-{p}-count", out["count"] == 10)
            durs.append(dt)
            thr.append(10 / dt if dt else 0.0)
        qs.append(_quantity("Q-C3", "invocation", "L",
                            "loopback-leash-double", "ms",
                            "pass-10-records", _ms(durs),
                            double_ms=dbl.service_s))
        qs.append(_quantity("Q-C3", "pass-throughput", "L",
                            "loopback-leash-double", "items/s",
                            "pass-10-records", thr,
                            double_ms=dbl.service_s))
    from si.gaia.hearth import ReplayTransport
    bd7, work7 = _new_bundle(root, "c3-gap")
    adm7 = _admit(bd7, work7, nxt())
    _leash_act(bd7, record_queue_stop, adm7["task_seq"],
               cause="s7-gap", detail="s7 gap")
    _leash_act(bd7, record_queue_resume, adm7["task_seq"],
               detail="s7 gap")
    vals = []
    for _ in range(cfg.pass_reps):
        out, dt = _timed(lambda: sync_seam_leash(
            bd7, leash_url=REPLAY_URL,
            transport=ReplayTransport([(500, {"error": "no"})])))
        vals.append(dt)
        assert out["count"] == 0
    riders.check("C3-gap-journals-nothing",
                 _open_rows(bd7, "gaia_leash") == [])
    qs.append(_quantity("Q-C3", "invocation", "S", "replay-leash",
                        "ms", "gap-pass-first-record-500", _ms(vals)))

    # Q-C4 heartbeat, Q-C5 snapshot.
    with LoopbackDouble(leash_double(applied=False)) as dbl:
        url = dbl.url("/gaia/v1/leash")
        bd8, work8 = _new_bundle(root, "c4-heartbeat")
        adm8 = _admit(bd8, work8, nxt())
        _leash_act(bd8, record_queue_stop, adm8["task_seq"],
                   cause="s7-hb", detail="s7 heartbeat")
        _warm(lambda: send_leash_heartbeat(bd8, "muse", leash_url=url),
              cfg.warmup)
        vals = []
        for _ in range(cfg.reps):
            out, dt = _timed(lambda: send_leash_heartbeat(
                bd8, "muse", leash_url=url))
            vals.append(dt)
            assert out["status"] == "crossed", out
        qs.append(_quantity("Q-C4", "invocation", "L",
                            "loopback-leash-double", "ms", "heartbeat",
                            _ms(vals), double_ms=dbl.service_s))
        _leash_act(bd8, record_queue_resume, adm8["task_seq"],
                   detail="s7 snapshot state")
        _warm(lambda: send_leash_snapshot(bd8, "muse", leash_url=url),
              cfg.warmup)
        vals = []
        for _ in range(cfg.reps):
            out, dt = _timed(lambda: send_leash_snapshot(
                bd8, "muse", leash_url=url))
            vals.append(dt)
            assert out["status"] == "crossed", out
        qs.append(_quantity("Q-C5", "invocation", "L",
                            "loopback-leash-double", "ms", "snapshot",
                            _ms(vals), double_ms=dbl.service_s))
        riders.check("C4C5-crossings-journaled",
                     len(_open_rows(bd8, "gaia_leash"))
                     >= 2 * cfg.reps)
    return qs


# -- Hop D: S-5 outcome return (design §7) ---------------------------------------

def _submitted_bundle(root, label, k, seq):
    bd, work = _new_bundle(root, label)
    wids = []
    for i in range(k):
        seq["n"] += 1
        adm = _admit(bd, work, seq["n"])
        wid = f"work-{label}-{i}"
        _submit(bd, adm["edit_id"], wid)
        wids.append(wid)
    return bd, work, wids


def plan_d(root, cfg, riders):
    from si.gaia.hearth import HttpTransport, ReplayTransport
    from si.gaia.seam_outcome import (
        mark_seam_outcome_unknown, poll_seam_outcomes,
        reconcile_seam_outcomes, seam_reproposal_blocked)
    qs = []
    seq = {"n": 20000}

    # Q-D2 per-work GET leg by answer class.
    answers = {}
    bd0, work0, wids0 = _submitted_bundle(root, "d2", 3, seq)
    for wid, kind in zip(wids0, ("terminal", "awaiting", "notfound")):
        if kind == "terminal":
            answers[wid] = (200, {"work_id": wid, "state": "executed",
                                  "result": _result_for(wid)})
        elif kind == "awaiting":
            answers[wid] = (200, {"work_id": wid, "state": "admitted"})
        else:
            answers[wid] = (404, {"error": "gone"})
    with LoopbackDouble(status_double(answers)) as dbl:
        base = dbl.url("/gaia/v1/work")
        transport = HttpTransport()
        for wid, kind in zip(wids0, ("terminal", "awaiting", "notfound")):
            _warm(lambda: transport("GET", f"{base}/{wid}"), cfg.warmup)
            vals = []
            for _ in range(cfg.reps):
                _, dt = _timed(lambda: transport(
                    "GET", f"{base}/{wid}"))
                vals.append(dt)
            qs.append(_quantity("Q-D2", "invocation", "L",
                                "loopback-status-double", "ms", kind,
                                _ms(vals), double_ms=dbl.service_s))

    # Q-D1 poll passes: sizes x mixes.
    mixes = ("terminal", "awaiting", "mixed")
    for k in dict.fromkeys((1, 10, cfg.poll_big)):
        for mix in mixes:
            durs, thr = [], []
            for p in range(cfg.pass_reps):
                label = f"d1-{k}-{mix}-{p}"
                bd, work, wids = _submitted_bundle(root, label, k, seq)
                ans = {}
                n_term = 0
                for i, wid in enumerate(wids):
                    if mix == "terminal" or (mix == "mixed" and i % 2 == 0):
                        ans[wid] = (200, {
                            "work_id": wid, "state": "executed",
                            "result": _result_for(wid)})
                        n_term += 1
                    elif mix == "mixed" and i % 4 == 1:
                        ans[wid] = (404, {"error": "gone"})
                    else:
                        ans[wid] = (200, {"work_id": wid,
                                          "state": "admitted"})
                with LoopbackDouble(status_double(ans)) as dbl:
                    out, dt = _timed(lambda: poll_seam_outcomes(
                        bd, work_url=dbl.url("/gaia/v1/work")))
                riders.check(f"D1-{label}-terminal-count",
                             out["count"] == n_term,
                             f"{out['count']}!={n_term}")
                durs.append(dt)
                thr.append(k / dt if dt else 0.0)
            qs.append(_quantity("Q-D1", "invocation", "L",
                                f"loopback-poll-{k}-{mix}", "ms",
                                f"pass-{k}-{mix}", _ms(durs)))
            qs.append(_quantity("Q-D1", "pass-throughput", "L",
                                f"loopback-poll-{k}-{mix}", "items/s",
                                f"pass-{k}-{mix}", thr))

    # Q-D3 outcome journaling phase (replay transport floor, 1 outcome).
    vals = []
    for rep in range(cfg.reps):
        bd, work, wids = _submitted_bundle(root, f"d3-{rep}", 1, seq)
        body = {"work_id": wids[0], "state": "executed",
                "result": _result_for(wids[0])}
        out, dt = _timed(lambda: poll_seam_outcomes(
            bd, work_url=REPLAY_URL,
            transport=ReplayTransport([(200, body)])))
        vals.append(dt)
        assert out["count"] == 1
    qs.append(_quantity(
        "Q-D3", "phase", "S", "miniature-single-outcome", "ms",
        "journal-outcome-replay", _ms(vals),
        notes="poll of exactly one outstanding submission over the "
              "replay floor: scan + journaling + re-anchor, with "
              "the transport leg ~zero by construction (§3.2)"))

    # Q-D4 reconciler unit cost by line class (derived per-1,000).
    ghost_lines = [_outcome_line(f"ghost-{i}", _result_for(f"ghost-{i}"))
                   for i in range(1000)]
    bd_g, _, _ = _submitted_bundle(root, "d4-ghost", 1, seq)
    durs, size = [], 0
    for p in range(cfg.pass_reps):
        path = os.path.join(root, f"d4-ghost-{p}.jsonl")
        with open(path, "w") as fh:
            fh.write("".join(l + "\n" for l in ghost_lines))
        size = os.path.getsize(path)
        out, dt = _timed(lambda: reconcile_seam_outcomes(bd_g, path))
        durs.append(dt)
        assert out["status"] == "reconciled", out
    qs.append(_quantity("Q-D4", "invocation", "S",
                        "outcomes-file-1000-unmatched", "ms",
                        "unmatched-per-1000-lines", _ms(durs),
                        notes=f"file bytes {size}; per-KiB derived in "
                              f"the results document"))
    for cls_name, mode in (("agreement", "agree"),
                           ("disagreement", "disagree"),
                           ("distinct-second-outcome", "second")):
        k = 20
        durs = []
        for p in range(cfg.pass_reps):
            bd, work, wids = _submitted_bundle(
                root, f"d4-{mode}-{p}", k, seq)
            if mode in ("agree", "disagree"):
                bodies = [(200, {"work_id": wid, "state": "executed",
                                 "result": _result_for(wid)})
                          for wid in wids]
                poll_seam_outcomes(
                    bd, work_url=REPLAY_URL,
                    transport=ReplayTransport(bodies))
            lines = []
            for wid in wids:
                if mode == "agree":
                    lines.append(_outcome_line(wid, _result_for(wid)))
                elif mode == "disagree":
                    lines.append(_outcome_line(
                        wid, _result_for(wid, detail="different")))
                else:
                    lines.append(_outcome_line(wid, _result_for(wid)))
                    lines.append(_outcome_line(
                        wid, _result_for(wid, detail="second story")))
            path = os.path.join(root, f"d4-{mode}-{p}.jsonl")
            with open(path, "w") as fh:
                fh.write("".join(l + "\n" for l in lines))
            out, dt = _timed(lambda: reconcile_seam_outcomes(bd, path))
            durs.append(dt)
            assert out["status"] == "reconciled", out
            if mode == "agree":
                riders.check(f"D4-agree-{p}",
                             len(out["cross_checked"]) == k)
            else:
                riders.check(f"D4-faults-{mode}-{p}",
                             len(out["faults"]) >= 1)
        n_lines = k if mode != "second" else 2 * k
        qs.append(_quantity(
            "Q-D4", "invocation", "S", f"outcomes-file-{cls_name}",
            "ms", f"{cls_name}-per-{n_lines}-lines", _ms(durs),
            notes="per-1,000-line unit cost is derived arithmetic "
                  "(pass median / lines x 1000), reported in the "
                  "results document (§7.2)"))
    # Torn-line partial pass.
    bd_t, work_t, wids_t = _submitted_bundle(root, "d4-torn", 1, seq)
    path_t = os.path.join(root, "d4-torn.jsonl")
    good = _outcome_line("ghost-torn", _result_for("ghost-torn"))
    with open(path_t, "wb") as fh:
        fh.write((good + "\n").encode() + b'{"work_id": "x", "resu')
    out, dt = _timed(lambda: reconcile_seam_outcomes(bd_t, path_t))
    riders.check("D4-torn-cursor-held",
                 out["torn"] == 1
                 and out["cursor"] == len(good) + 1)
    qs.append(_quantity("Q-D4", "invocation", "S", "outcomes-file-torn",
                        "ms", "torn-partial-pass", [dt * 1000.0]))

    # Q-D5 scaling: file size (unmatched) x cursor position.
    for n_lines in (100, 1000, cfg.reconcile_big):
        lines = [_outcome_line(f"g5-{i}", _result_for(f"g5-{i}"))
                 for i in range(n_lines)]
        path = os.path.join(root, f"d5-{n_lines}.jsonl")
        with open(path, "w") as fh:
            fh.write("".join(l + "\n" for l in lines))
        durs = []
        for _ in range(cfg.pass_reps):
            out, dt = _timed(lambda: reconcile_seam_outcomes(
                bd_g, path, cursor=0))
            durs.append(dt)
            assert out["cursor"] == os.path.getsize(path)
        qs.append(_quantity("Q-D5", "invocation", "S",
                            f"outcomes-file-{n_lines}", "ms",
                            f"from-0-{n_lines}-lines", _ms(durs)))
    last = _outcome_line("g5-last", _result_for("g5-last"))
    path_big = os.path.join(root, f"d5-{cfg.reconcile_big}.jsonl")
    # The near-EOF cursor is the byte offset where the appended
    # known last line begins, so the tail reads exactly one line.
    near_eof = os.path.getsize(path_big)
    with open(path_big, "a") as fh:
        fh.write(last + "\n")
    vals = []
    for _ in range(cfg.reps):
        out, dt = _timed(lambda: reconcile_seam_outcomes(
            bd_g, path_big, cursor=near_eof))
        vals.append(dt)
        assert len(out["unmatched"]) == 1
    qs.append(_quantity("Q-D5", "invocation", "S",
                        f"outcomes-file-{cfg.reconcile_big}", "ms",
                        "near-eof-1-line", _ms(vals)))

    # Q-D6 outcome-unknown marking + reproposal-block read.
    marked = []
    vals = []
    for rep in range(cfg.reps):
        bd, work, wids = _submitted_bundle(root, f"d6-{rep}", 1, seq)
        out, dt = _timed(lambda: mark_seam_outcome_unknown(
            bd, wids[0], cause="s7 measurement fixture",
            evidence=["s7: double reported no outcome"]))
        vals.append(dt)
        assert out["status"] == "outcome-unknown", out
        marked.append((bd, out["caused_by"]))
    qs.append(_quantity("Q-D6", "invocation", "S",
                        "miniature-single-submission", "ms",
                        "mark-outcome-unknown", _ms(vals)))
    vals = []
    for bd, caused_by in marked:
        _, dt = _timed(lambda: seam_reproposal_blocked(bd, caused_by))
        vals.append(dt)
    riders.check("D6-reproposal-blocked",
                 all(seam_reproposal_blocked(bd, cb)
                     for bd, cb in marked[:3]))
    qs.append(_quantity("Q-D6", "phase", "S",
                        "miniature-single-submission", "ms",
                        "reproposal-block-read", _ms(vals)))

    # §7.5 conditional outcome latency (D-3 schedule: offset 0, period 1 s).
    lat = []
    for run in range(cfg.catchup_reps):
        bd, work, wids = _submitted_bundle(root, f"dcond-{run}", 1, seq)
        flip = {"at": None}

        def handler(method, path, body, state, flip=flip, wid=wids[0]):
            if flip["at"] is not None \
                    and time.perf_counter() >= flip["at"]:
                return 200, {"work_id": wid, "state": "executed",
                             "result": _result_for(wid)}
            return 200, {"work_id": wid, "state": "admitted"}
        with LoopbackDouble(handler) as dbl:
            url = dbl.url("/gaia/v1/work")
            t_start = time.perf_counter()
            flip["at"] = t_start + 0.35
            out = poll_seam_outcomes(bd, work_url=url)  # offset 0 poll
            assert out["count"] == 0
            time.sleep(1.0)  # the stated period, a harness parameter
            out = poll_seam_outcomes(bd, work_url=url)
            t_journaled = time.perf_counter()
            riders.check(f"D-cond-{run}-journaled", out["count"] == 1)
            lat.append(t_journaled - flip["at"])
    qs.append(_quantity(
        "Q-D-COND", "journal-derived-lag", "L",
        "loopback-status-double-schedule", "s",
        "conditional-offset0-period1s", lat,
        notes="conditional outcome latency (§7.5): outcome "
              "journaled time minus the double's terminal time, "
              "both on this host's monotonic clock, under the "
              "stated harness schedule (first poll at offset 0 s, "
              "then every 1 s; terminal at +0.35 s). Not an "
              "unconditional outcome latency."))
    return qs


# -- Hop E: S-6 event forwarding (design §8) --------------------------------------

def _grow_events(root, label, target, seq):
    from si.gaia.corpus import corpus_policy
    from si.gaia.seam import ingest_proposal
    from si.gaia.seam_events import pending_event_seqs
    bd, work = _new_bundle(root, label)
    pol = corpus_policy(work)
    while len(pending_event_seqs(bd)) < target:
        for _ in range(10):
            seq["n"] += 1
            ingest_proposal(bd, _watcher(seq["n"]), policy=pol)
    return bd, work


def plan_e(root, cfg, riders):
    from si.gaia.events import read_journal_events
    from si.gaia.seam_events import (
        SeamProtocolError, build_stream_event, forward_journal_events,
        pending_event_seqs)
    qs = []
    seq = {"n": 40000}
    bd, work = _grow_events(root, "e-grow", cfg.ev_big + 10, seq)
    all_seqs = pending_event_seqs(bd)
    riders.check("E-grow-reached-target",
                 len(all_seqs) >= cfg.ev_big, str(len(all_seqs)))
    before_counts = _counts(bd)

    journal = _open(bd)
    try:
        events = read_journal_events(journal, after_seq=0)
        # Q-E1 payload construction per event.
        sample = events[:cfg.reps]
        _warm(lambda: build_stream_event(sample[0]), cfg.warmup)
        vals = []
        for ev in sample:
            _, dt = _timed(lambda ev=ev: build_stream_event(ev))
            vals.append(dt)
        qs.append(_quantity("Q-E1", "phase", "S",
                            "miniature-grown-journal", "ms",
                            "build_stream_event-per-event", _ms(vals)))
        riders.check("E1-shape-no-hash",
                     all("hash" not in build_stream_event(ev)
                         and build_stream_event(ev)["source"] == "si"
                         for ev in sample[:5]))
        # Q-E2 journal read windows x cursor positions.
        tip = events[-1]["id"]
        for window in (1, 10, 100, cfg.ev_big):
            for cname, cursor in (("start", 0),
                                  ("near-tip", max(0, tip - cfg.ev_mid))):
                _warm(lambda: read_journal_events(
                    journal, after_seq=cursor, limit=window), cfg.warmup)
                vals = []
                for _ in range(cfg.reps):
                    got, dt = _timed(lambda: read_journal_events(
                        journal, after_seq=cursor, limit=window))
                    vals.append(dt)
                expect = min(window, len(
                    [e for e in events if e["id"] > cursor]))
                riders.check(f"E2-window-{window}-{cname}",
                             len(got) == expect,
                             f"{len(got)}!={expect}")
                qs.append(_quantity("Q-E2", "phase", "S",
                                    "miniature-grown-journal", "ms",
                                    f"window-{window}-@{cname}",
                                    _ms(vals)))
        # D-10: journal timestamp grain (adjacent-row ts deltas).
        deltas = []
        parsed = []
        for r in [ev["row"] for ev in events]:
            try:
                parsed.append(datetime.fromisoformat(
                    r["ts"].replace("Z", "+00:00")))
            except Exception:
                parsed.append(None)
        for a, b in zip(parsed, parsed[1:]):
            if a is not None and b is not None:
                deltas.append((b - a).total_seconds())
        qs.append(_quantity(
            "Q-E0", "journal-derived-lag", "S",
            "miniature-grown-journal", "ms",
            "journal-ts-adjacent-delta (D-10 grain)", _ms(deltas),
            notes="timestamp grain measured first per D-10: the "
                  "median adjacent-row delta bounds what the "
                  "journal-derived lag (Q-E-LAGJ) can resolve"))
    finally:
        journal.close()

    # Q-E3 per-event forward (limit=1), + D-4 delay variant.
    for delay, vname in ((0.0, "per-event"), (0.005, "per-event-delay5ms")):
        with LoopbackDouble(ingest_double(), delay_s=delay) as dbl:
            url = dbl.url("/gaia/v1/events/ingest")
            # Warmup forwards advance the caller-held cursor, so
            # no event is ever pushed twice to this double.
            cursor = 0
            for _ in range(cfg.warmup):
                out_w = forward_journal_events(
                    bd, ingest_url=url, cursor=cursor, limit=1)
                cursor = out_w["cursor"]
            vals = []
            for _ in range(cfg.reps):
                out, dt = _timed(lambda: forward_journal_events(
                    bd, ingest_url=url, cursor=cursor, limit=1))
                vals.append(dt)
                assert out["count"] == 1, out
                cursor = out["cursor"]
            qs.append(_quantity("Q-E3", "invocation", "L",
                                "loopback-ingest-double", "ms", vname,
                                _ms(vals), double_ms=dbl.service_s,
                                notes="D-4 sensitivity variant"
                                if delay else ""))

    # Q-E4 pass throughput 10 / mid / big + interrupted pass.
    for size in (10, cfg.ev_mid, cfg.ev_big):
        durs, thr = [], []
        svc = []
        for p in range(cfg.pass_reps):
            with LoopbackDouble(ingest_double()) as dbl:
                url = dbl.url("/gaia/v1/events/ingest")
                out, dt = _timed(lambda: forward_journal_events(
                    bd, ingest_url=url, limit=size))
                riders.check(f"E4-pass-{size}-{p}",
                             out["count"] == size
                             and out["pushed"] == all_seqs[:size])
                svc.extend(dbl.service_s)
            durs.append(dt)
            thr.append(size / dt if dt else 0.0)
        qs.append(_quantity("Q-E4", "invocation", "L",
                            "loopback-ingest-double", "ms",
                            f"pass-{size}", _ms(durs), double_ms=svc))
        qs.append(_quantity("Q-E4", "pass-throughput", "L",
                            "loopback-ingest-double", "items/s",
                            f"pass-{size}", thr, double_ms=svc))
    k_fail = cfg.ev_mid // 2
    durs = []
    for p in range(cfg.pass_reps):
        with LoopbackDouble(ingest_double(fail_at=k_fail)) as dbl:
            url = dbl.url("/gaia/v1/events/ingest")
            out, dt = _timed(lambda: forward_journal_events(
                bd, ingest_url=url, limit=cfg.ev_mid))
            riders.check(f"E4-interrupted-{p}",
                         out["status"] == "forward-failed"
                         and out["cursor"] == all_seqs[k_fail - 1]
                         and out["pushed"] == all_seqs[:k_fail])
        durs.append(dt)
    qs.append(_quantity("Q-E4", "invocation", "L",
                        "loopback-ingest-double", "ms",
                        f"interrupted-pass-{cfg.ev_mid}-at-{k_fail}",
                        _ms(durs)))

    # Q-E5 catch-up: repeated limited passes + single unlimited pass.
    for backlog in (cfg.ev_mid, cfg.ev_big):
        for mode in ("repeated-limit50", "single-unlimited"):
            walls, rates, passes = [], [], []
            for run in range(cfg.catchup_reps):
                with LoopbackDouble(ingest_double()) as dbl:
                    url = dbl.url("/gaia/v1/events/ingest")
                    cursor, n_pass = 0, 0
                    t0 = time.perf_counter()
                    if mode == "single-unlimited":
                        out = forward_journal_events(
                            bd, ingest_url=url, limit=backlog)
                        cursor, n_pass = out["cursor"], 1
                    else:
                        pushed = 0
                        while cursor < all_seqs[backlog - 1]:
                            out = forward_journal_events(
                                bd, ingest_url=url, cursor=cursor,
                                limit=min(50, backlog - pushed))
                            cursor = out["cursor"]
                            pushed += out["count"]
                            n_pass += 1
                    wall = time.perf_counter() - t0
                riders.check(f"E5-{backlog}-{mode}-{run}",
                             cursor == all_seqs[backlog - 1])
                walls.append(wall)
                rates.append(backlog / wall if wall else 0.0)
                passes.append(float(n_pass))
            qs.append(_quantity("Q-E5", "invocation", "L",
                                "loopback-ingest-double", "s",
                                f"catchup-{backlog}-{mode}",
                                walls))
            qs.append(_quantity("Q-E5", "pass-throughput", "L",
                                "loopback-ingest-double", "items/s",
                                f"catchup-{backlog}-{mode}", rates))
            qs.append(_quantity("Q-E5", "pass-throughput", "L",
                                "loopback-ingest-double", "passes",
                                f"catchup-{backlog}-{mode}-passes-to-tip",
                                passes))

    # Q-E6 duplicate push: protocol failure behaviour.
    with LoopbackDouble(ingest_double()) as dbl:
        url = dbl.url("/gaia/v1/events/ingest")
        forward_journal_events(bd, ingest_url=url, limit=cfg.ev_mid)
        calls_before = len(dbl.log)
        vals = []
        for _ in range(cfg.reps):
            def attempt():
                try:
                    forward_journal_events(bd, ingest_url=url)
                except SeamProtocolError as e:
                    return e
                return None
            err, dt = _timed(attempt)
            vals.append(dt)
            assert err is not None and err.seq == all_seqs[0] \
                and err.cursor == 0 and err.pushed == []
        riders.check("E6-duplicate-raises-first-event",
                     len(dbl.log) == calls_before + cfg.reps)
        qs.append(_quantity("Q-E6", "invocation", "L",
                            "loopback-ingest-double", "ms",
                            "duplicate-push-time-to-raise", _ms(vals),
                            double_ms=dbl.service_s,
                            notes="quantity of record is correctness "
                                  "(cursor/pushed/call count, pinned "
                                  "by the rider); duration is "
                                  "supporting data (§8.2)"))

    # §8.4(b) in-pass lag by position decile (Class L, monotonic).
    lag_by_decile = {d: [] for d in range(10)}
    for p in range(3):
        with LoopbackDouble(ingest_double()) as dbl:
            url = dbl.url("/gaia/v1/events/ingest")
            t0 = time.perf_counter()
            out = forward_journal_events(bd, ingest_url=url,
                                         limit=cfg.ev_big)
            assert out["count"] == cfg.ev_big
            arrivals = [e["perf_s"] for e in dbl.log]
            for i, arr in enumerate(arrivals):
                decile = min(9, int(10 * i / cfg.ev_big))
                lag_by_decile[decile].append((arr - t0) * 1000.0)
    for d in range(10):
        qs.append(_quantity("Q-E-LAG", "invocation", "L",
                            "loopback-ingest-double", "ms",
                            f"in-pass-lag-decile-{d + 1}",
                            lag_by_decile[d],
                            notes="§8.4(b): pass start -> that "
                                  "event's 200, by position decile "
                                  "in a serial one-POST-per-event "
                                  "pass"))

    # §8.4 journal-derived lag under the harness cadence (back-to-
    # back passes immediately after growth), D-10 grain permitting.
    with LoopbackDouble(ingest_double()) as dbl:
        url = dbl.url("/gaia/v1/events/ingest")
        wall0 = time.time()
        perf0 = time.perf_counter()
        forward_journal_events(bd, ingest_url=url,
                               limit=cfg.ev_mid)
        ts_by_seq = {}
        journal = _open(bd)
        try:
            for ev in read_journal_events(journal, after_seq=0):
                try:
                    ts_by_seq[ev["id"]] = datetime.fromisoformat(
                        ev["row"]["ts"].replace(
                            "Z", "+00:00")).timestamp()
                except Exception:
                    pass
        finally:
            journal.close()
        lags = []
        for entry in dbl.log:
            seqv = (entry["body"] or {}).get("seq")
            if seqv in ts_by_seq:
                ack_wall = wall0 + (entry["perf_s"] - perf0)
                lags.append((ack_wall - ts_by_seq[seqv]) * 1000.0)
        qs.append(_quantity(
            "Q-E-LAGJ", "journal-derived-lag", "L",
            "miniature-grown-journal", "ms",
            "journal-derived-queueing-plus-inpass", lags,
            notes="§8.4 journal-derived lag under the harness "
                  "cadence (one pass driven immediately after "
                  "journal growth): queueing lag is a property "
                  "of that cadence, not of the system; never "
                  "merged with the monotonic Q-E-LAG statistics"))
    riders.check("E-forwarder-journals-nothing",
                 _counts(bd) == before_counts)
    return qs


# -- runner + CLI -----------------------------------------------------------------

PLANS = (("A", plan_a), ("B", plan_b), ("C", plan_c),
         ("D", plan_d), ("E", plan_e))


def run_measurement(root, cfg=None, si_commit=None, argv=None,
                    hops=("A", "B", "C", "D", "E")):
    cfg = cfg or Cfg()
    os.makedirs(root, exist_ok=True)
    riders = Riders()
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    quantities, setup = [], {}
    for name, plan in PLANS:
        if name not in hops:
            continue
        t_plan = time.perf_counter()
        quantities.extend(plan(root, cfg, riders))
        setup[f"hop_{name}_wall_s"] = time.perf_counter() - t_plan
    finished = datetime.now(timezone.utc)
    fingerprint = environment_fingerprint(
        si_commit=si_commit, seed=cfg.seed, argv=argv)
    fingerprint["started_at"] = started.isoformat()
    fingerprint["finished_at"] = finished.isoformat()
    fingerprint["transports"] = {
        "A": "Class S in-process + Class L real B-5 service handler "
             "on loopback (Q-A6)",
        "B": "Class S ReplayTransport + Class L loopback work "
             "double via HttpTransport",
        "C": "Class S journal reads + Class L loopback leash double",
        "D": "Class S file/replay + Class L loopback status double",
        "E": "Class S journal reads + Class L loopback ingest "
             "double (200/409 contract; fork ingest UNBUILT)",
    }
    valid = riders.all_passed()
    return {
        "record": "S-7 seam measurements (SI-62)",
        "design": "docs/S-7-seam-measurement-design.md (SI-61)",
        "fingerprint": fingerprint,
        "valid": valid,
        "quantities": quantities,
        "riders": riders.items,
        "deferred": [{"id": d["id"], "quantity": d["quantity"],
                      "status": "UNMEASURED", "reason": d["reason"]}
                     for d in DEFERRED],
        "decisions": {
            "D-1": "harness-side timing only; internal phases as "
                   "named residuals (Q-A4, Q-B4, Q-C3)",
            "D-2": "§11.3 minimums adopted (30 / 10 passes / 5 "
                   "catch-up runs)",
            "D-3": "one schedule each: poll offset 0 s, period 1 s; "
                   "forward passes back-to-back",
            "D-4": "5 ms-delay sensitivity variant run for Q-B3 "
                   "and Q-E3",
            "D-5": "Q-C4 cost reported only; cadence adequacy not "
                   "judged",
            "D-6": "concurrent-ingest observation OUT (deferred "
                   "F-A-CONC)",
            "D-7": "chain worksheet published without a total row",
            "D-8": "Class L only; no Class F attempt",
            "D-9": "record committed beside the results document "
                   "(docs/S-7-measurements.json)",
            "D-10": "journal timestamp grain measured first "
                    "(Q-E0); journal-derived lag reported at that "
                    "grain (Q-E-LAGJ)",
            "D-11": "double-bound threshold 25% of round-trip "
                    "median, computed per Class L quantity",
            "D-12": "p95 published for all N>=30 quantities",
        },
        "setup": setup,
        "wall_s": time.perf_counter() - t0,
        "exclusions": [],
        "invalidations": [] if valid else [
            "one or more correctness riders failed — this run is "
            "INVALID and its timings are not publishable (§10.1)"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="si.gaia.seam_measure",
        description="Gaia S-7 seam measurement harness (SI-62).")
    parser.add_argument("--out", required=True,
                        help="output directory for the record JSON")
    parser.add_argument("--root", default=None,
                        help="scratch root (default: <out>/scratch)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--pass-reps", type=int, default=10)
    parser.add_argument("--catchup-reps", type=int, default=5)
    parser.add_argument("--si-commit", default=None)
    parser.add_argument("--quick", action="store_true",
                        help="small-N smoke configuration (tests)")
    parser.add_argument("--keep", action="store_true",
                        help="keep the scratch root")
    args = parser.parse_args(argv)
    if args.quick:
        cfg = Cfg(reps=3, pass_reps=1, catchup_reps=1, warmup=1,
                  seed=args.seed, ev_big=60, ev_mid=30,
                  pending_big=10, poll_big=10, reconcile_big=200)
    else:
        cfg = Cfg(reps=args.reps, pass_reps=args.pass_reps,
                  catchup_reps=args.catchup_reps, seed=args.seed)
    root = args.root or os.path.join(args.out, "scratch")
    record = run_measurement(root, cfg=cfg, si_commit=args.si_commit,
                             argv=[sys.argv[0]] + (argv or []))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "seam-measurements.json"),
              "w") as fh:
        json.dump(record, fh, indent=1, sort_keys=True)
        fh.write("\n")
    if not args.keep and args.root is None:
        import shutil
        shutil.rmtree(root, ignore_errors=True)
    print(json.dumps({
        "valid": record["valid"],
        "quantities": len(record["quantities"]),
        "riders_failed": [r["name"] for r in record["riders"]
                          if not r["passed"]],
        "wall_s": round(record["wall_s"], 1)}, indent=1))
    return 0 if record["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
