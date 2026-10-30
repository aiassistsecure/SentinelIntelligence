"""Gaia B-5: agentic work as a service — the SI-side adapter
(SI-41 RFC §12.5(a), §15 B-5; slice SI-46).

B-1…B-3 built the agency loop (registry, gate, executor, task
loop); B-4 built the Hearth interface. This module is the
§12.5(a) surface, **SI side only**: a stdlib-HTTP service in
``si serve``'s pattern that accepts a task, runs it through the
B-3 machinery (:func:`si.gaia.task.run_task`), and returns the
result **with its provenance trail** serialised from the
journal — the task instruction, the submitter identity, every
intent, verdict, gate report, execution record, and the stop,
walkable end to end. The trail is the deliverable (§12.5(e)):
an external agent's work product is its output; an SI mind's
work product is its output **and** the gated, journaled,
replayable account of every action that produced it.

Surface (localhost by default; see the honesty labels below):

    GET  /              -> service description + honesty labels
    GET  /health        -> bundle id, head, service labels
    POST /tasks         -> submit one task; 200 returns the task
                           result + provenance trail; 400 returns
                           the validation problems and NOTHING
                           was journaled or run
    GET  /tasks/<seq>   -> the provenance trail for the task
                           journaled at <seq>, rebuilt from the
                           journal (404 when no such task)
    GET  /events        -> (SI-47) the SSE event spine: every
                           journal row serialised as a
                           server-sent event whose id IS the
                           journal seq; resume via the
                           Last-Event-ID header (or ?after=);
                           replays, then follows live by
                           poll-the-journal (see si.gaia.events)

A submission is a JSON object::

    {"submitter": "<identity>",            # mandatory
     "instruction": "<task instruction>",  # mandatory
     "script": [<intent dict>, ...],       # mandatory, non-empty
     "work_root": "<path>",                # optional
     "cycles": <positive int>}             # optional

``script`` is the scripted proposer of RFC §14.1 — the same
intent-dict shape ``si task --script`` accepts, converted by the
same rules (:func:`intents_from_script` mirrors ``si.cli``'s
conversion). There is no other proposer: **no trained mind
proposes tool intents yet** (§14), and this adapter does not
invent a default mind — a submission without a script is
refused, exactly as ``si task`` without ``--script`` refuses.

**Submitter identity (§12.5(a)'s priced risk).** A task from
the network is an operator instruction only if the service's
auth story says so — and B-5 has **no auth story**: the
submitter identity is a mandatory *journaled provenance field*
(no identity, no task: a submission without one is refused
before anything is journaled), carried on the task entry, the
response, and the trail, labelled everywhere with
:data:`si.gaia.task.SUBMITTER_IDENTITY_LABEL` — asserted,
**not authentication**: no credential is checked and the
identity is not verified (the ``si admit`` --operator
precedent). Multi-tenancy is out of scope until a real auth
story exists (RFC §19), and nothing here claims one. The
service binds localhost by default and carries **no security
claim**: do not expose it beyond localhost without putting
your own controls in front of it (``si serve``'s discipline,
kept verbatim in spirit).

**Exposure through the Hearth gateway is NOT built.** That
route is proposal **H-2** (§12.6) — Mark's to build in the
Hearth repo. B-5 completes without it and says so
(:data:`GATEWAY_EXPOSURE_LABEL` rides on every service
response): this service is exposed *beside* a gateway on the
same node, never *through* one, and it imports no Hearth code.

**Execution scope is B-3's, unchanged.** A submitted task runs
exactly what ``si task`` runs: live execution is
miniature-scoped; demo bundles keep shadow behaviour (their
intents ESCALATE, the task halts ``escalated``, nothing
executes). Submission is **synchronous**: POST /tasks blocks
until the task stops (a §10.4 stop rule or proposer
exhaustion) and then returns — there is no job queue, no
async handle, no streaming in B-5. The NEDB journal stays the
source of truth; every response field is serialised from
journal rows or from the task runner's return, which itself
summarises journaled steps.

**SI-47 adds ``GET /events``** — the Gaia SSE event spine
(:mod:`si.gaia.events`): every journal row serialised as a
server-sent event whose id IS the journal seq, resumable via
``Last-Event-ID``, replaying then following live by
poll-the-journal (labelled polling — no push channel exists).
It rides this same server and the same honesty labels: no
authentication, localhost by default, and no gateway exposure
(H-2, unchanged). Reading the stream journals nothing.

**SI-48 adds filtered subscriptions on ``GET /events``** —
``?collection=``, ``?event=``, ``?kind=`` and ``?task=<seq>``
project the same stream server-side (a projection, never a
second store: ids remain journal seqs, gaps are the
filtered-out seqs, malformed filter values are refused with
400 like malformed resume points, and unfiltered requests are
byte-identical to SI-47). The reference consumer is
:mod:`si.gaia.events_client` (``si gaia-events``).
"""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from si.gaia.events import (
    SSE_FILTER_LABEL,
    SSE_RESUME_LABEL,
    SSE_TAIL_LABEL,
    format_comment,
    parse_event_filter,
    parse_resume_seq,
    stream_events,
)
from si.gaia.task import SUBMITTER_IDENTITY_LABEL, run_task
from si.gate import Intent

#: The submitted-via value journaled on every B-5 task entry.
SUBMITTED_VIA = "gaia-service (SI-46/B-5)"

#: The H-2 honesty label: exposure through the Hearth gateway
#: is proposal H-2 (§12.6), Mark's — not built in B-5.
GATEWAY_EXPOSURE_LABEL = (
    "not built — exposure through the Hearth gateway is "
    "proposal H-2 (SI-41 §12.6), Mark's to build in the Hearth "
    "repo; this service is exposed beside a gateway on the "
    "same node, never through one (SI-41 §12.5(a))")

#: The no-auth honesty label, carried on the service surface.
NO_AUTH_LABEL = (
    "no authentication: submitter identity is asserted and "
    "journaled as provenance, never verified; localhost by "
    "default; no security claim — do not expose beyond "
    "localhost without your own controls in front of it")

#: POST body cap: a submission is an instruction plus a script
#: of intents, not a data channel. Labelled, enforced before
#: parsing, and a refusal journals nothing.
MAX_BODY_BYTES = 1_048_576


# -- submission validation ---------------------------------------------------

def validate_submission(payload) -> tuple[dict | None, list]:
    """Validate one task submission; fail closed.

    Returns ``(normalised, [])`` on success — ``normalised`` has
    ``submitter``, ``instruction``, ``script`` (the raw intent
    dicts), and the optional ``work_root`` / ``cycles`` — or
    ``(None, problems)`` with every problem itemised. The
    submitter identity is mandatory (§12.5(a)): missing, blank,
    or non-string is a refusal, and a refused submission is
    journaled nowhere and runs nothing (the caller enforces
    this by validating before any journal is opened).
    """
    if not isinstance(payload, dict):
        return None, ["submission body must be a JSON object"]
    problems: list = []

    submitter = payload.get("submitter")
    if not isinstance(submitter, str) or not submitter.strip():
        problems.append(
            "submitter is mandatory: a non-empty string naming "
            "the submitting party, journaled as a provenance "
            "field — no identity, no task (SI-41 §12.5(a))")
    instruction = payload.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        problems.append(
            "instruction is mandatory: a non-empty string — the "
            "task instruction journaled on the task entry "
            "(RFC §10.1)")
    script = payload.get("script")
    if not isinstance(script, list) or not script:
        problems.append(
            "script is mandatory: a non-empty JSON array of "
            "intent objects (the scripted proposer of RFC "
            "§14.1 — no trained mind proposes tool intents "
            "yet, and B-5 does not invent a default mind)")
    elif not all(
            isinstance(d, dict)
            or (isinstance(d, list)
                and all(isinstance(m, dict) for m in d))
            for d in script):
        problems.append(
            "script entries must all be JSON objects (a nested "
            "array of objects is one SI-51 batch)")
    else:
        flat = []
        for i, d in enumerate(script):
            if isinstance(d, list):
                flat.extend((f"{i}.{j}", m) for j, m in enumerate(d))
            else:
                flat.append((str(i), d))
        for i, d in flat:
            kind = d.get("kind", "tool")
            if kind not in ("tool", "memory"):
                problems.append(
                    f"script[{i}]: kind must be \"tool\" or "
                    f"\"memory\" (the B-3 task loop's proposer "
                    f"contract), got {kind!r}")
            elif kind == "tool" and not d.get("tool"):
                problems.append(
                    f"script[{i}]: a tool intent must name its "
                    "tool")

    work_root = payload.get("work_root")
    if work_root is not None and not isinstance(work_root, str):
        problems.append("work_root must be a string path when given")
    cycles = payload.get("cycles")
    if cycles is not None and (
            isinstance(cycles, bool)
            or not isinstance(cycles, int) or cycles < 1):
        problems.append("cycles must be a positive integer when given")

    if problems:
        return None, problems
    return {
        "submitter": submitter,
        "instruction": instruction,
        "script": script,
        "work_root": work_root,
        "cycles": cycles,
    }, []


def intents_from_script(script: list) -> list:
    """Convert validated script dicts to Intent objects.

    Mirrors ``si task``'s conversion in :func:`si.cli.cmd_task`
    exactly (kind defaults to ``"tool"``; tool intents carry
    tool/args, memory intents carry record; parents/source_seqs
    pass through and the task runner fills the chain for
    scripted intents that omit them, §10.2).
    """
    def _intent_from(d):
        kind = d.get("kind", "tool")
        return Intent(
            kind=kind,
            reason=d.get("reason", ""),
            parents=list(d.get("parents") or []),
            source_seqs=list(d.get("source_seqs") or []),
            tool=d.get("tool", "") if kind == "tool" else "",
            args=dict(d.get("args") or {}) if kind == "tool" else {},
            record=dict(d.get("record") or {}) if kind == "memory"
            else {},
            ordering_key=d.get("ordering_key")
            if kind == "tool" else None,
            depends_on=list(d.get("depends_on") or [])
            if kind == "tool" else [],
            barrier=bool(d.get("barrier") or False)
            if kind == "tool" else False)

    intents = []
    for d in script:
        # SI-51 (SI-50 §6): a nested array is one batch proposal.
        if isinstance(d, list):
            intents.append([_intent_from(m) for m in d])
        else:
            intents.append(_intent_from(d))
    return intents


# -- the provenance trail ------------------------------------------------------

def _jsonable_row(row: dict) -> dict:
    """A journal row as plain JSON data (rows already are, via
    the journal's _jsonable append path; this copies defensively
    so a trail never aliases live journal state)."""
    return json.loads(json.dumps(row, default=str))


def build_provenance_trail(journal, task_seq: int) -> dict:
    """Serialise one task's provenance trail from the journal.

    The trail is the §12.5(a) deliverable: the task entry (with
    its submitter identity), every chained step's self-edit —
    the intent **and** its full gate report (verdict + itemised
    checks) — the provenance edges those edits carry, every
    execution outcome recorded for them, and the task_stop,
    ordered by journal seq, walkable end to end. Steps are found
    by the §10.2 chain itself: a self-edit belongs to the task
    when its intent cites the task seq, or cites the outcome
    seq of a step already in the trail (the runner's chaining
    fills exactly those citations), transitively. Rows are read
    from the existing collections only — no new collection, no
    NEDB schema change (§4.6) — and ``journal_chain_verified``
    is the journal's own hash-chain verdict over the whole
    journal at read time.

    Raises KeyError when no task entry exists at ``task_seq`` —
    an unknown task is reported as unknown, never fabricated.
    """
    task_rows = journal.backend.find(
        "experiences", filter={"kind": "task", "seq": task_seq},
        limit=1)
    if not task_rows:
        raise KeyError(f"no task entry journaled at seq {task_seq}")
    task_row = task_rows[0]
    task_record = task_row.get("record") or {}
    task_ref = f"task@{task_seq}"

    stops = [r for r in journal.backend.find(
        "experiences", filter={"kind": "task_stop"})
        if (r.get("record") or {}).get("task_seq") == task_seq]
    stop_row = stops[0] if stops else None

    edits = journal.backend.find("self_edits")
    edits.sort(key=lambda r: r.get("seq") or 0)
    outcomes = journal.backend.find("experiences")
    outcome_seq_by_edit: dict = {}
    for row in outcomes:
        edit_id = row.get("self_edit_id")
        if edit_id:
            outcome_seq_by_edit.setdefault(edit_id, []).append(
                row.get("seq"))

    # The §10.2 chain, followed transitively from the task seq.
    known_seqs = {task_seq}
    chained: list = []
    chained_seqs: set = set()
    changed = True
    while changed:
        changed = False
        for row in edits:
            if row.get("seq") in chained_seqs:
                continue
            intent = row.get("intent") or {}
            cited = set(intent.get("source_seqs") or [])
            parents = set(intent.get("parents") or [])
            if cited & known_seqs or task_ref in parents:
                chained.append(row)
                chained_seqs.add(row.get("seq"))
                known_seqs.add(row.get("seq"))
                for s in outcome_seq_by_edit.get(
                        row.get("edit_id"), []):
                    known_seqs.add(s)
                changed = True

    # SI-51 (SI-50 §5): the task's queue entries — batch entries,
    # claims, recovery re-claims, claim refusals, operator stops
    # and resumes — are part of the task's provenance: they are
    # how its intents were claimed before they ran. Collected by
    # the record's own task_seq, in journal order (the same rule
    # the SSE spine's task filter uses, si.gaia.events).
    queue_kinds = ("queue_batch", "queue_claim",
                   "queue_recovery_reclaim", "queue_refusal",
                   "queue_stop", "queue_resume")
    queue_rows = [r for r in outcomes
                  if r.get("kind") in queue_kinds
                  and (r.get("record") or {}).get("task_seq")
                  == task_seq]
    queue_rows.sort(key=lambda r: r.get("seq") or 0)

    steps = []
    for row in chained:
        edit_id = row.get("edit_id")
        edges = journal.backend.find(
            "provenance", filter={"child": edit_id})
        edges.sort(key=lambda r: r.get("seq") or 0)
        step_outcomes = [r for r in outcomes
                         if r.get("self_edit_id") == edit_id]
        step_outcomes.sort(key=lambda r: r.get("seq") or 0)
        steps.append({
            "seq": row.get("seq"),
            "edit_id": edit_id,
            "kind": row.get("kind"),
            "intent": row.get("intent"),
            "reason": row.get("reason"),
            "gate_verdict": row.get("gate_verdict"),
            "gate_report": row.get("gate_report"),
            "result": row.get("result"),
            "provenance": [_jsonable_row(e) for e in edges],
            "outcomes": [_jsonable_row(o) for o in step_outcomes],
        })

    return {
        "task_seq": task_seq,
        "task_ref": task_ref,
        "submitter": task_record.get("submitter"),
        "submitted_via": task_record.get("submitted_via"),
        "submitter_identity_label": SUBMITTER_IDENTITY_LABEL,
        "task_entry": _jsonable_row(task_row),
        "steps": steps,
        "queue": [_jsonable_row(r) for r in queue_rows],
        "task_stop": _jsonable_row(stop_row) if stop_row else None,
        "journal_chain_verified": journal.verify_chain(),
        "trail_label": (
            "serialised from the bundle's NEDB journal, the "
            "source of truth — the task entry, every chained "
            "intent with its gate report, its provenance edges "
            "and execution outcomes, and the stop, in journal "
            "seq order (SI-41 §12.5(a))"),
    }


def provenance_trail_for_bundle(bundle_dir: str, task_seq: int) -> dict:
    """Open a bundle read-only-ish and build one task's trail.

    (The journal open is the runtime's existing path; building a
    trail writes nothing.) Raises KeyError for an unknown task.
    """
    from si.bundle import load_bundle
    from si.journal import open_journal
    bundle_dir = os.path.abspath(bundle_dir)
    manifest = load_bundle(bundle_dir)
    journal = open_journal(bundle_dir, manifest)
    try:
        return build_provenance_trail(journal, task_seq)
    finally:
        journal.close()


# -- submission -----------------------------------------------------------------

def submit_task(bundle_dir: str, payload,
                policy: dict | None = None) -> dict:
    """Validate and run one submitted task; return result + trail.

    A refused submission returns ``{"status": "refused",
    "problems": [...]}`` and — the §12.5(a) discipline — nothing
    is journaled and nothing runs: validation completes before
    the bundle is even loaded. An accepted submission runs
    through B-3's :func:`run_task` with the submitter journaled
    on the task entry, and returns the runner's summary plus
    the provenance trail serialised from the journal after the
    task stops.
    """
    normalised, problems = validate_submission(payload)
    if normalised is None:
        return {
            "status": "refused",
            "problems": problems,
            "journaled": False,
            "label": (
                "a refused submission journals nothing and runs "
                "nothing — validation precedes any journal write "
                "(SI-41 §12.5(a): no identity, no task)"),
        }
    bundle_dir = os.path.abspath(bundle_dir)
    result = run_task(
        bundle_dir, intents_from_script(normalised["script"]),
        normalised["instruction"],
        work_root=normalised["work_root"],
        cycles=normalised["cycles"],
        submitter=normalised["submitter"],
        submitted_via=SUBMITTED_VIA,
        policy=policy)
    trail = provenance_trail_for_bundle(bundle_dir, result["task_seq"])
    return {
        "status": "task-returned",
        "submitter": normalised["submitter"],
        "submitted_via": SUBMITTED_VIA,
        "submitter_identity_label": SUBMITTER_IDENTITY_LABEL,
        "gateway_exposure": GATEWAY_EXPOSURE_LABEL,
        "result": result,
        "provenance_trail": trail,
    }


# -- the HTTP service -------------------------------------------------------------

class ServiceState:
    """The served bundle, snapshotted at startup like si serve's.

    The bundle is loaded and verified once (``load_bundle``
    fails closed on a tampered bundle — a tampered bundle is
    never served). Unlike ``si serve`` this service is NOT
    read-only: accepting a task runs the B-3 machinery, which
    journals the task and re-anchors the manifest — that is
    the service's purpose. The snapshot therefore carries
    identity and the startup head only; each trail is read
    fresh from the journal, the source of truth.
    """

    def __init__(self, bundle_dir: str, policy: dict | None = None):
        from si.bundle import load_bundle, manifest_id
        self.bundle_dir = os.path.abspath(bundle_dir)
        self.manifest = load_bundle(self.bundle_dir)
        self.bundle_id = manifest_id(self.manifest)
        self.startup_head = self.manifest["head"]
        self.policy = policy

    def health(self) -> dict:
        from si.bundle import load_bundle
        head = load_bundle(self.bundle_dir)["head"]
        return {
            "status": "ok",
            "service": "gaia-service (SI-46/B-5)",
            "bundle_id": self.bundle_id,
            "bundle_name": self.manifest.get("name"),
            "head": head,
            "startup_head": self.startup_head,
            "submitter_identity": SUBMITTER_IDENTITY_LABEL,
            "auth": NO_AUTH_LABEL,
            "gateway_exposure": GATEWAY_EXPOSURE_LABEL,
            "events_tail": SSE_TAIL_LABEL,
            "events_resume": SSE_RESUME_LABEL,
            "events_filter": SSE_FILTER_LABEL,
        }

    def description(self) -> dict:
        return {
            "service": "gaia-service (SI-46/B-5)",
            "bundle_id": self.bundle_id,
            "head": self.startup_head,
            "endpoints": {
                "GET /health": "bundle id, current head, labels",
                "POST /tasks": (
                    'submit one task: {"submitter": ..., '
                    '"instruction": ..., "script": [<intent '
                    'dict>, ...]} -> {"result": ..., '
                    '"provenance_trail": ...}; a submission '
                    "without a submitter identity is refused "
                    "(400) and journals nothing"),
                "GET /tasks/<seq>": (
                    "the provenance trail for the task journaled "
                    "at <seq>, rebuilt from the journal"),
                "GET /events": (
                    "the SSE event spine (SI-47): every journal "
                    "row as a server-sent event, id == journal "
                    "seq; Last-Event-ID (or ?after=<seq>) resumes "
                    "the replay; then follows live by "
                    "poll-the-journal (?follow=0 for replay only); "
                    "SI-48 filters project the same stream: "
                    "?collection=<coll>[,<coll>...], "
                    "?event=<name>, ?kind=<kind>, ?task=<seq> "
                    "(repeatable; malformed values are 400; ids "
                    "stay journal seqs — gaps are filtered-out "
                    "seqs)"),
            },
            "events_tail": SSE_TAIL_LABEL,
            "events_resume": SSE_RESUME_LABEL,
            "events_filter": SSE_FILTER_LABEL,
            "submitter_identity": SUBMITTER_IDENTITY_LABEL,
            "auth": NO_AUTH_LABEL,
            "gateway_exposure": GATEWAY_EXPOSURE_LABEL,
            "execution_scope": (
                "B-3's scope, unchanged: live execution is "
                "miniature-scoped; demo bundles stay shadow "
                "(intents ESCALATE, nothing executes)"),
        }


def make_service_server(bundle_dir: str, host: str = "127.0.0.1",
                        port: int = 8000,
                        policy: dict | None = None):
    """Build (but do not start) the B-5 service HTTP server.

    In ``si serve``'s pattern: stdlib ``ThreadingHTTPServer``,
    ``port=0`` picks an ephemeral port (used by tests). The
    returned server carries its :class:`ServiceState` as
    ``server.state``. ``policy`` is the operator's bundle-policy
    override handed to the task runner (tests use it to install
    the corpus policy; production passes None and the bundle's
    own policy governs — fail-closed by default, §6.2).
    """
    state = ServiceState(bundle_dir, policy=policy)

    class Handler(BaseHTTPRequestHandler):
        server_version = "gaia-service/0.1"

        def log_message(self, fmt, *args):  # noqa: A003 - stdlib name
            pass

        def _send_json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _handle_events(self) -> None:
            """GET /events — the SI-47 SSE spine over real HTTP.

            Opens its own journal handle (the B-5 trail's
            per-request pattern — a following stream polls for
            as long as the client stays connected, so it never
            shares the request path's handles), streams the
            replay from the resume seq, then follows live by
            poll-the-journal until the client disconnects. A
            failed write IS the disconnect signal: the stream
            stops, the journal closes, and the handler thread —
            a daemon thread, see make_service_server — ends.
            Nothing is journaled by reading the stream.
            """
            from si.bundle import load_bundle
            from si.journal import open_journal
            query = parse_qs(urlparse(self.path).query,
                             keep_blank_values=True)
            after_q = (query.get("after", [None])[0]
                       or query.get("last_event_id", [None])[0])
            try:
                after = parse_resume_seq(
                    self.headers.get("Last-Event-ID"), after_q)
                event_filter = parse_event_filter(query)
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return
            follow = query.get("follow", ["1"])[0] \
                not in ("0", "false", "no")
            manifest = load_bundle(state.bundle_dir)
            journal = open_journal(state.bundle_dir, manifest)
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                # A following stream is keep-alive by nature; a
                # replay-only stream must CLOSE at its end — with
                # no Content-Length, the close is how the client
                # learns the replay is complete.
                self.send_header("Connection", "keep-alive" if follow
                                 else "close")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                if event_filter.is_empty():
                    banner_text = (
                        f"gaia-events (SI-47): replaying journal "
                        f"rows after seq {after}; {SSE_TAIL_LABEL}; "
                        f"{SSE_RESUME_LABEL}; "
                        f"auth: {NO_AUTH_LABEL}")
                else:
                    banner_text = (
                        f"gaia-events (SI-48): replaying journal "
                        f"rows after seq {after}; "
                        f"{event_filter.describe()}; "
                        f"{SSE_FILTER_LABEL}; {SSE_TAIL_LABEL}; "
                        f"{SSE_RESUME_LABEL}; "
                        f"auth: {NO_AUTH_LABEL}")
                banner = format_comment(banner_text)
                self.wfile.write(banner.encode("utf-8"))
                self.wfile.flush()
                for block in stream_events(
                        journal, after_seq=after, follow=follow,
                        event_filter=event_filter):
                    try:
                        self.wfile.write(block.encode("utf-8"))
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError,
                            OSError):
                        break
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                if not follow:
                    self.close_connection = True
                journal.close()

        def do_GET(self) -> None:  # noqa: N802 - stdlib name
            path = urlparse(self.path).path
            if path in ("/health", "/health/"):
                self._send_json(200, state.health())
            elif path == "/":
                self._send_json(200, state.description())
            elif path in ("/events", "/events/"):
                self._handle_events()
            elif path.startswith("/tasks/"):
                raw = path[len("/tasks/"):].strip("/")
                try:
                    task_seq = int(raw)
                except ValueError:
                    self._send_json(400, {
                        "error": f"task seq must be an integer, "
                                 f"got {raw!r}"})
                    return
                try:
                    trail = provenance_trail_for_bundle(
                        state.bundle_dir, task_seq)
                except KeyError as e:
                    self._send_json(404, {"error": str(e)})
                    return
                self._send_json(200, {
                    "provenance_trail": trail,
                    "gateway_exposure": GATEWAY_EXPOSURE_LABEL})
            else:
                self._send_json(404, {
                    "error": f"unknown path {self.path!r}"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib name
            if self.path not in ("/tasks", "/tasks/"):
                self._send_json(404, {
                    "error": f"unknown path {self.path!r}"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if length > MAX_BODY_BYTES:
                self._send_json(400, {
                    "error": f"submission body exceeds the "
                             f"{MAX_BODY_BYTES}-byte cap — a "
                             "submission is an instruction plus a "
                             "script of intents, not a data "
                             "channel; nothing was journaled"})
                return
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                payload = json.loads(raw.decode("utf-8")) if raw \
                    else {}
            except (ValueError, UnicodeDecodeError) as e:
                self._send_json(400, {
                    "error": f"invalid JSON body: {e}"})
                return
            out = submit_task(state.bundle_dir, payload,
                              policy=state.policy)
            if out["status"] == "refused":
                self._send_json(400, out)
                return
            self._send_json(200, out)

    class _Server(ThreadingHTTPServer):
        # SI-47: an /events stream follows for as long as its
        # client stays connected — handler threads are daemons
        # so a still-connected stream can never block server
        # shutdown or process exit.
        daemon_threads = True

    server = _Server((host, int(port)), Handler)
    server.state = state  # type: ignore[attr-defined]
    return server


def serve_service(bundle_dir: str, host: str = "127.0.0.1",
                  port: int = 8000,
                  policy: dict | None = None) -> dict:
    """Serve the B-5 adapter for ``bundle_dir`` until interrupted.

    Returns a small summary dict after shutdown. NOT read-only:
    every accepted task journals and may move the bundle head
    (the B-3 task machinery re-anchors the manifest), by design.
    """
    server = make_service_server(bundle_dir, host=host, port=port,
                                 policy=policy)
    state = server.state  # type: ignore[attr-defined]
    actual_host, actual_port = server.server_address[:2]
    print(f"serving gaia-service for bundle {state.bundle_id}")
    print(f"head at startup: {state.startup_head}")
    print(f"listening on http://{actual_host}:{actual_port} "
          f"(GET /health, POST /tasks, GET /tasks/<seq>, "
          f"GET /events)")
    print(f"note: {NO_AUTH_LABEL}")
    print(f"note: {GATEWAY_EXPOSURE_LABEL}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return {"bundle_id": state.bundle_id, "host": actual_host,
            "port": actual_port}


__all__ = [
    "GATEWAY_EXPOSURE_LABEL",
    "MAX_BODY_BYTES",
    "NO_AUTH_LABEL",
    "SUBMITTED_VIA",
    "ServiceState",
    "build_provenance_trail",
    "intents_from_script",
    "make_service_server",
    "provenance_trail_for_bundle",
    "serve_service",
    "submit_task",
    "validate_submission",
]
