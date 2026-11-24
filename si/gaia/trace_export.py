"""Gaia OBS-1: OpenTelemetry span mapping from the
journal + OTLP export (SI-120 lane; slice SI-124).

One command maps a bundle's journal onto OpenTelemetry
spans and emits them as ONE OTLP/JSON payload::

    si gaia-export-traces <bundle> [--out FILE]
                                   [--otlp-endpoint URL]

The payload is the standard OTLP/JSON ``resourceSpans``
structure — the wire format Langfuse's OTLP endpoint
(and any OTLP collector) ingests natively. With no
``--otlp-endpoint`` nothing is sent anywhere: the
payload goes to ``--out`` or stdout only. Passing the
endpoint flag is the explicit operator act that turns
network export on; the same payload bytes are POSTed.

This is a MAPPING + EXPORTER, not an observability
stack: no SDK is vendored or required (stdlib only —
the mapping targets the OTLP wire format, which is the
standard), no metrics pipeline, no dashboard. The
journal stays the source of truth; the spans are a
projection of it, derived deterministically:

- one trace per task the journal records (an
  ``experiences`` entry of kind ``task``), plus ONE
  journal-scope trace for the entries the journal does
  not tie to any task (seed checkpoints, head
  provenance). The journal's task structure is the
  seam/provenance linkage — ``record.task_seq`` on
  queue records, ``self_edit_id`` / ``record.edit_id``
  / ``record.intent_seq`` references, and provenance
  rows chaining a self-edit to ``task@<seq>``. Where
  none of those reach, an entry is journal-scope; the
  mapping says so rather than guessing a task.
- span parentage follows those same references (an
  outcome hangs from the intent it answers, an intent
  from its task). A seam intent's ``caused_by`` points
  OUTSIDE the journal (the fork's residency spine), so
  it is carried as span attributes, never as parentage
  — there is no journal span to parent to.
- trace ids are ``sha256("si-trace:" + manifest_id +
  ":" + scope)`` truncated to 32 hex chars, span ids
  ``sha256("si-span:" + entry_hash)`` truncated to 16
  hex chars: re-exporting the same journal yields the
  same ids, byte for byte. The payload carries no
  export timestamp — it is a pure function of the
  journal and the manifest.
- timestamps come from each entry's journaled ``ts``.
  The journal records append times, not durations, so
  every span is a zero-duration point at its append
  instant (``endTimeUnixNano == startTimeUnixNano``).
  No duration is invented.

The export is strictly READ-ONLY on the source bundle:
it opens the journal the ``_open_read`` way (chain
verified before anything is read, fail closed) and
writes nothing to the bundle.

Honest limits (also in docs/SI-124-observability.md):
journal-granularity tracing — one span per journaled
entry, nothing finer exists to map; append-time points,
not measured durations; verdicts and outcomes are span
ATTRIBUTES (a gate REJECT is a designed outcome, not a
tracing error — only refused/failed tool outcomes set
the span status to Error).

CPU only; stdlib only; network only when the operator
passes --otlp-endpoint; no secrets (Langfuse keys, when
the endpoint needs them, come from the operator's
environment — see :func:`_auth_headers`).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

from si.gaia.escalation_cli import _open_read
from si.gaia.run_export import _read_manifest
from si.journal import COLLECTIONS

#: Scope label printed with every command's output.
EXPORT_LABEL: str = (
    "observability export (SI-124/OBS-1): the bundle's "
    "journal mapped onto OpenTelemetry spans, emitted "
    "as OTLP/JSON. A deterministic projection of the "
    "journal — one span per journaled entry, append-time "
    "points, no invented durations. Nothing is sent "
    "anywhere unless --otlp-endpoint is passed")

_SCOPE_NAME = "si.gaia.trace_export"
_SERVICE_NAME = "sentinel-intelligence-gaia"

#: Span status codes (OTLP): 0 unset, 1 Ok, 2 Error.
_STATUS_UNSET, _STATUS_OK, _STATUS_ERROR = 0, 1, 2

#: Tool-outcome values that mark a span as an error.
_ERROR_OUTCOMES = frozenset(
    {"refused", "failed", "error", "failure", "timeout"})

#: Scalar record keys copied onto experience spans as
#: ``si.record.<key>`` attributes when present.
_RECORD_ATTRS = ("outcome", "status", "resolution",
                 "decision", "verdict", "fork_work_id",
                 "work_id", "refusal", "operator", "tool")


# ---------------------------------------------------------------------------
# Deterministic ids + OTLP value/timestamp plumbing
# ---------------------------------------------------------------------------

def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def trace_id_for(manifest_id: str, scope: str) -> str:
    """The trace id for one scope (``task@<seq>`` or
    ``journal``) of one bundle: the first 32 hex chars
    of ``sha256("si-trace:" + manifest_id + ":" +
    scope)``."""
    return _sha(f"si-trace:{manifest_id}:{scope}")[:32]


def span_id_for(entry_hash: str) -> str:
    """The span id for one journal entry: the first 16
    hex chars of ``sha256("si-span:" + entry_hash)``.
    Entry hashes are unique per entry, so span ids are
    too."""
    return _sha(f"si-span:{entry_hash}")[:16]


def _attr_value(value):
    """One journal value as an OTLP AnyValue, or None
    when the value has no honest scalar/array form
    (dicts and None are skipped by the caller, never
    stringified into fake precision — except refs,
    which callers stringify explicitly)."""
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, (list, tuple)):
        items = [_attr_value(v) for v in value]
        if all(i is not None for i in items):
            return {"arrayValue": {"values": items}}
    return None


def _attrs(pairs) -> list:
    """(key, value) pairs -> OTLP attribute list,
    skipping values with no honest OTLP form."""
    out = []
    for key, value in pairs:
        if value is None:
            continue
        wrapped = _attr_value(value)
        if wrapped is not None:
            out.append({"key": key, "value": wrapped})
    return out


def _ts_ns(ts) -> str:
    """A journal ``ts`` (ISO-8601, Z-suffixed) as OTLP
    Unix-nanoseconds string. A missing/unparseable ts
    maps to "0" — the epoch, visibly wrong rather than
    plausibly invented (every journal append stamps ts,
    so this is a defensive path, not an expected one).
    """
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(
                ts.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return str(int(dt.timestamp() * 1_000_000_000))
        except ValueError:
            pass
    return "0"


# ---------------------------------------------------------------------------
# The mapping: journal rows -> spans
# ---------------------------------------------------------------------------

def span_name(collection: str, row: dict) -> str:
    """The span name for one journal entry. Gaia
    entries (the acting layer) are ``gaia.*``; the
    mind's own journal machinery is ``si.*``."""
    kind = row.get("kind")
    if collection == "self_edits":
        return "gaia.tool_intent" if kind == "tool" \
            else "si.self_edit"
    if collection == "deltas":
        return "si.delta"
    if collection == "checkpoints":
        return "si.checkpoint"
    if collection == "provenance":
        return "si.provenance"
    if collection == "experiences":
        if kind == "task":
            return "gaia.task"
        if kind == "tool_result":
            return "gaia.tool_result"
        if kind == "escalation_resolution":
            return "gaia.escalation_resolution"
        if kind == "gaia_work":
            return "gaia.work_submission"
        if isinstance(kind, str) and \
                kind.startswith("queue_"):
            return f"gaia.queue.{kind[len('queue_'):]}"
        return "si.experience"
    return "si.journal_entry"


def _record(row: dict) -> dict:
    rec = row.get("record")
    return rec if isinstance(rec, dict) else {}


def _edit_task_map(rows) -> dict:
    """edit_id -> task seq, from provenance evidence
    only: a provenance row whose child is the edit and
    whose parents cite the task (``task@<seq>`` or the
    task entry's seq). No evidence, no mapping."""
    task_seqs = {seq for seq, coll, row in rows
                 if coll == "experiences"
                 and row.get("kind") == "task"}
    mapping = {}
    for _seq, coll, row in rows:
        if coll != "provenance":
            continue
        child = row.get("child")
        if not isinstance(child, str):
            continue
        for parent in row.get("parents") or []:
            seq = None
            if isinstance(parent, str) \
                    and parent.startswith("task@"):
                try:
                    seq = int(parent[len("task@"):])
                except ValueError:
                    seq = None
            elif isinstance(parent, int):
                seq = parent
            if seq in task_seqs:
                mapping[child] = seq
    return mapping


def _entry_task(row, coll, edit_task, by_seq) -> int | None:
    """The task seq one entry belongs to, per the
    journal's own references — or None (journal scope).
    """
    if coll == "experiences" and row.get("kind") == "task":
        return row["seq"]
    if coll == "self_edits":
        return edit_task.get(row.get("edit_id"))
    if coll == "deltas":
        return edit_task.get(row.get("self_edit_id"))
    if coll == "provenance":
        return edit_task.get(row.get("child"))
    if coll == "experiences":
        rec = _record(row)
        task_seq = rec.get("task_seq")
        if isinstance(task_seq, int) and not \
                isinstance(task_seq, bool):
            return task_seq
        for key in ("edit_id",):
            if rec.get(key) in edit_task:
                return edit_task[rec[key]]
        if row.get("self_edit_id") in edit_task:
            return edit_task[row["self_edit_id"]]
        intent_seq = rec.get("intent_seq")
        if isinstance(intent_seq, int):
            target = by_seq.get(intent_seq)
            if target is not None:
                return _entry_task(target[1], target[0],
                                   edit_task, by_seq)
        execution = rec.get("execution")
        if isinstance(execution, dict):
            intent_id = execution.get("intent_id")
            if intent_id in edit_task:
                return edit_task[intent_id]
    return None


def _entry_parent(row, coll, by_seq) -> int | None:
    """The seq of the entry this one hangs from, per
    the journal's references — an outcome from its
    intent, an intent's provenance from the intent,
    a delta from its self-edit. None = trace root."""
    if coll == "provenance":
        child = row.get("child")
        for seq, (_c, other) in by_seq.items():
            if _c == "self_edits" \
                    and other.get("edit_id") == child:
                return seq
        return None
    if coll == "deltas":
        edit_id = row.get("self_edit_id")
        for seq, (_c, other) in by_seq.items():
            if _c == "self_edits" \
                    and other.get("edit_id") == edit_id:
                return seq
        return None
    if coll == "experiences":
        rec = _record(row)
        intent_seq = rec.get("intent_seq")
        if isinstance(intent_seq, int) \
                and intent_seq in by_seq:
            return intent_seq
        edit_id = rec.get("edit_id") \
            or row.get("self_edit_id")
        execution = rec.get("execution")
        if not edit_id and isinstance(execution, dict):
            edit_id = execution.get("intent_id")
        if edit_id:
            for seq, (_c, other) in by_seq.items():
                if _c == "self_edits" \
                        and other.get("edit_id") == edit_id:
                    return seq
        task_seq = rec.get("task_seq")
        if isinstance(task_seq, int) \
                and task_seq in by_seq:
            return task_seq
    if coll == "self_edits":
        return None  # parented to the task root by caller
    return None


def _span_attributes(coll, row) -> list:
    pairs = [
        ("si.journal.collection", coll),
        ("si.journal.seq", row.get("seq")),
        ("si.journal.entry_hash", row.get("entry_hash")),
        ("si.journal.prev_hash", row.get("prev_hash")),
        ("si.journal.kind", row.get("kind")),
    ]
    if coll == "self_edits":
        intent = row.get("intent")
        intent = intent if isinstance(intent, dict) else {}
        pairs += [
            ("si.edit_id", row.get("edit_id")),
            ("si.gate.verdict", row.get("gate_verdict")),
            ("si.self_edit.result", row.get("result")),
            ("si.delta_id", row.get("delta_id")),
            ("si.operator", row.get("operator")),
            ("tool.name", intent.get("tool")),
        ]
        caused_by = intent.get("caused_by")
        if isinstance(caused_by, dict):
            pairs += [
                ("si.caused_by.seq", caused_by.get("seq")),
                ("si.caused_by.hash", caused_by.get("hash")),
            ]
        report = row.get("gate_report")
        if isinstance(report, dict):
            checks = report.get("checks")
            if isinstance(checks, list):
                pairs.append((
                    "si.gate.checks",
                    [c.get("name") for c in checks
                     if isinstance(c, dict)
                     and isinstance(c.get("name"), str)]))
    elif coll == "deltas":
        pairs += [
            ("si.delta.id", row.get("id")),
            ("si.delta.parent", row.get("parent")),
            ("si.delta.status", row.get("status")),
            ("si.delta.bytes", row.get("bytes")),
            ("si.edit_id", row.get("self_edit_id")),
        ]
    elif coll == "checkpoints":
        pairs += [
            ("si.checkpoint.head", row.get("head")),
            ("si.checkpoint.manifest", row.get("manifest")),
            ("si.checkpoint.note", row.get("note")),
        ]
    elif coll == "provenance":
        parents = row.get("parents")
        pairs += [
            ("si.provenance.child", row.get("child")),
            ("si.provenance.relation", row.get("relation")),
            ("si.provenance.parents",
             [str(p) for p in parents]
             if isinstance(parents, list) else None),
        ]
    elif coll == "experiences":
        rec = _record(row)
        pairs += [
            ("si.experience.kind", row.get("kind")),
            ("si.edit_id", rec.get("edit_id")
             or row.get("self_edit_id")),
            ("si.intent.seq", rec.get("intent_seq")),
            ("si.task.seq", rec.get("task_seq")),
        ]
        for key in _RECORD_ATTRS:
            if key in rec:
                pairs.append((f"si.record.{key}", rec[key]))
        execution = rec.get("execution")
        if isinstance(execution, dict):
            for key in ("outcome", "status", "tool",
                        "refusal"):
                if key in execution:
                    pairs.append(
                        (f"si.execution.{key}",
                         execution[key]))
            if "tool" in execution:
                pairs.append(
                    ("tool.name", execution["tool"]))
    return _attrs(pairs)


def _span_status(coll, row) -> dict:
    """Error only for refused/failed tool outcomes —
    the one place a journaled result is an execution
    failure. Gate verdicts (ADMIT/REJECT/ESCALATE) are
    attributes: a REJECT is the gate working, not an
    error."""
    if coll == "experiences" \
            and row.get("kind") == "tool_result":
        rec = _record(row)
        execution = rec.get("execution")
        execution = execution if isinstance(
            execution, dict) else {}
        outcome = execution.get("outcome") \
            or execution.get("status") \
            or rec.get("outcome") or rec.get("status")
        if isinstance(outcome, str) \
                and outcome.lower() in _ERROR_OUTCOMES:
            return {"code": _STATUS_ERROR,
                    "message": outcome}
    return {"code": _STATUS_UNSET}


def build_trace_payload(journal, manifest: dict) -> dict:
    """Map an open, chain-verified journal onto the
    OTLP/JSON payload. Pure function of the journal +
    manifest: no wall clock, no randomness — the same
    journal always yields the same payload bytes."""
    from si.bundle import manifest_id
    mid = manifest_id(manifest)

    rows = []
    for coll in COLLECTIONS:
        for row in journal.backend.find(coll) or []:
            rows.append((row.get("seq") or 0, coll, row))
    rows.sort(key=lambda item: item[0])
    by_seq = {seq: (coll, row) for seq, coll, row in rows}
    edit_task = _edit_task_map(rows)

    task_of = {}
    for seq, coll, row in rows:
        task_of[seq] = _entry_task(row, coll, edit_task,
                                   by_seq)

    spans = []
    for seq, coll, row in rows:
        task_seq = task_of[seq]
        scope = f"task@{task_seq}" if task_seq is not None \
            else "journal"
        parent_seq = _entry_parent(row, coll, by_seq)
        if parent_seq is None and coll == "self_edits" \
                and task_seq is not None:
            parent_seq = task_seq
        if parent_seq is not None \
                and task_of.get(parent_seq) != task_seq:
            parent_seq = None  # never parent across traces
        span = {
            "traceId": trace_id_for(mid, scope),
            "spanId": span_id_for(row.get("entry_hash")),
            "name": span_name(coll, row),
            "kind": 1,  # SPAN_KIND_INTERNAL
            "startTimeUnixNano": _ts_ns(row.get("ts")),
            "endTimeUnixNano": _ts_ns(row.get("ts")),
            "attributes": _span_attributes(coll, row),
            "status": _span_status(coll, row),
        }
        if parent_seq is not None:
            parent_row = by_seq[parent_seq][1]
            span["parentSpanId"] = span_id_for(
                parent_row.get("entry_hash"))
        spans.append(span)

    return {
        "resourceSpans": [{
            "resource": {
                "attributes": _attrs([
                    ("service.name", _SERVICE_NAME),
                    ("si.bundle.name", manifest.get("name")),
                    ("si.bundle.manifest_id", mid),
                    ("si.bundle.head", manifest.get("head")),
                    ("telemetry.sdk.name", _SCOPE_NAME),
                    ("telemetry.sdk.language", "python"),
                ]),
            },
            "scopeSpans": [{
                "scope": {"name": _SCOPE_NAME, "version": "1"},
                "spans": spans,
            }],
        }],
    }


def export_traces_bundle(bundle_dir: str) -> dict:
    """Open ``bundle_dir`` read-only (chain verified
    first, fail closed — the ``_open_read`` way) and
    build the OTLP payload for its whole journal.
    Writes nothing to the bundle; sends nothing
    anywhere."""
    manifest = _read_manifest(bundle_dir)
    journal = _open_read(bundle_dir)
    try:
        return build_trace_payload(journal, manifest)
    finally:
        journal.close()


# ---------------------------------------------------------------------------
# The one network path: an explicit --otlp-endpoint POST
# ---------------------------------------------------------------------------

def _auth_headers() -> dict:
    """Auth headers from the OPERATOR's environment —
    never hardcoded, never required for file export.
    Langfuse's OTLP endpoint takes HTTP Basic over its
    public/secret key pair; a verbatim Authorization
    value (SI_OTLP_AUTHORIZATION) wins when set."""
    verbatim = os.environ.get("SI_OTLP_AUTHORIZATION")
    if verbatim:
        return {"Authorization": verbatim}
    public = os.environ.get("SI_LANGFUSE_PUBLIC_KEY")
    secret = os.environ.get("SI_LANGFUSE_SECRET_KEY")
    if public and secret:
        token = base64.b64encode(
            f"{public}:{secret}".encode("utf-8")
        ).decode("ascii")
        return {"Authorization": f"Basic {token}"}
    return {}


def post_otlp(payload_bytes: bytes, endpoint: str,
              timeout: float = 30.0) -> int:
    """POST payload bytes to an OTLP endpoint; returns
    the HTTP status. The ONLY network call in this
    module, reached only via the explicit endpoint
    flag."""
    request = urllib.request.Request(
        endpoint, data=payload_bytes, method="POST",
        headers={"Content-Type": "application/json",
                 **_auth_headers()})
    with urllib.request.urlopen(request,
                                timeout=timeout) as resp:
        return resp.status


# ---------------------------------------------------------------------------
# CLI front-ends (module main + the si.cli wiring share these)
# ---------------------------------------------------------------------------

def execute_export_traces(args) -> int:
    """Run one parsed ``gaia-export-traces`` command.
    Exit 0 on success (payload written/printed, and
    POSTed when --otlp-endpoint was passed), 1 on a
    refused export or a failed POST."""
    print(f"# {EXPORT_LABEL}", file=sys.stderr)
    try:
        payload = export_traces_bundle(args.path)
    except (ValueError, FileNotFoundError) as e:
        print(f"si: error: {e}", file=sys.stderr)
        return 1
    text = json.dumps(payload, indent=2, sort_keys=True)
    data = (text + "\n").encode("utf-8")
    spans = payload["resourceSpans"][0]["scopeSpans"][0][
        "spans"]
    traces = {s["traceId"] for s in spans}
    if args.out:
        with open(args.out, "wb") as f:
            f.write(data)
        print(f"exported {len(spans)} spans across "
              f"{len(traces)} traces -> {args.out}",
              file=sys.stderr)
    else:
        sys.stdout.write(text + "\n")
    if args.otlp_endpoint:
        try:
            status = post_otlp(data, args.otlp_endpoint)
        except Exception as e:  # network failure is loud
            print(f"si: error: OTLP POST to "
                  f"{args.otlp_endpoint} failed: "
                  f"{type(e).__name__}: {e}",
                  file=sys.stderr)
            return 1
        print(f"POSTed {len(data)} bytes to "
              f"{args.otlp_endpoint} -> HTTP {status}",
              file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="si.gaia.trace_export",
        description="SI-124/OBS-1 observability export: "
                    "map a bundle's journal onto "
                    "OpenTelemetry spans and emit OTLP/"
                    "JSON. Read-only on the bundle; "
                    "nothing is sent anywhere unless "
                    "--otlp-endpoint is passed.")
    parser.add_argument("path", help="bundle directory")
    parser.add_argument("--out", default=None,
                        help="write the payload to FILE "
                             "(default: stdout)")
    parser.add_argument("--otlp-endpoint", default=None,
                        help="POST the payload to this "
                             "OTLP endpoint URL (e.g. a "
                             "Langfuse OTLP traces "
                             "endpoint); default: no "
                             "network, export only")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return execute_export_traces(args)
    except Exception as e:  # fail closed, loud
        print(f"si: FAILED: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
