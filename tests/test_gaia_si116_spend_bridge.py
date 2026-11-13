"""SI-116 / P-1 (SI-115 §4; D-115-1/2/3): the resolution-
spend bridge.

Pins:

- the eligibility table: an unresolved escalation, a
  ``reject``-resolved intent, an unknown seq, a resolved
  intent no longer folded ``queued``, a non-seam-crossing
  tool, and an anonymous operator are each refused and
  journal NOTHING (the journal tip is byte-identical);
- the golden payload: field-for-field the fork's as-built
  ``parse_intent`` shape (gateway.rs:170) — ``type`` tool,
  journaled args verbatim, ``task_id`` ``task@<seq>``,
  ``journal_seq`` the intent's self-edit seq (never the
  resolution seq), verdict ``admit``, no ``class`` field;
- a confirmed spend journals one ``gaia_work`` row tying
  intent seq + resolution seq + fork work id, with the
  operator and basis recorded, and re-anchors;
- dedup twice over: a second spend returns the original
  submission and calls no transport; the fork double's S-1
  dedup (``journal_seq`` → work id) returns the same record
  for a resubmission;
- a fork refusal / a transport failure / a 202 without a
  work id journals the failed attempt as evidence
  (``spend-failed``), never as a submission — and a later
  operator re-invocation then spends cleanly;
- the CLI wiring: ``spend`` / ``poll-outcome`` /
  ``reconcile-outcome`` parse on both the module parser and
  the ``si`` CLI parser;
- a loopback end-to-end over real HTTP: an in-process stub
  implementing ``parse_intent``'s rules + the 202 ACK + S-1
  dedup + the status route (claim → ``stale_state``
  refusal → executed) — spend → poll journals the refusal
  outcome once (folded ``refused-at-claim``), a second poll
  dedupes, and the intent-scoped reconciler cross-checks
  the same outcome from an ``outcomes.jsonl``.

No live fork endpoint is exercised: transports are replay
or the loopback stub (labelled test double), and live
interop is UNMEASURED.
"""
import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from si.bundle import (
    _anchor_checkpoint, _write_manifest, init_bundle, load_bundle)
from si.cli import build_parser
from si.gaia import escalation_cli
from si.gaia.context import ToolContext
from si.gaia.corpus import corpus_policy
from si.gaia.drain import admit_tool_intent
from si.gaia.hearth import HttpTransport, ReplayTransport
from si.gaia.queue import fold_queue, record_queue_claim
from si.gaia.resolution import resolve_escalation
from si.gaia.seam_spend import (
    DEFAULT_SPEND_BASIS,
    SPEND_SUBMITTED_VIA,
    build_spend_payload,
    find_spend_row,
    poll_spent_outcome,
    read_spend_at,
    reconcile_spent_outcome,
    spend_resolved_escalation,
    verify_spendable,
)
from si.gaia.service import submit_proposal
from si.gaia.shadow import record_task_fixture
from si.gate import Gate, Intent
from si.journal import open_journal
from si.tiny_seed import seed_tiny_bundle

WATCHER = {
    "model": "muse",
    "tool": "hearth.restart",
    "args": {"model": "muse",
             "spine_event": {"hash": "hash-s116-fixture", "seq": 41}},
    "reason": "observed loss at seq 41",
    "caused_by": {"hash": "hash-s116-fixture", "seq": 41},
}
ESCALATE = {"tool_class_policy": {
    "read": "admit", "write": "admit",
    "execute": "admit", "external": "escalate"}}
URL = "replay://fork/gaia/v1/work"
OPERATOR = "si116-test-operator"


@pytest.fixture()
def mini(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "mini.si"),
                             name="gaia-si116",
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    work = tmp_path / "work"
    work.mkdir()
    return bundle_dir, str(work)


def _open(bundle_dir):
    manifest = load_bundle(bundle_dir)
    return manifest, open_journal(bundle_dir, manifest)


def _escalated(mini):
    """One ESCALATEd hearth.restart seam intent.
    Returns (bundle_dir, ack)."""
    bundle_dir, work = mini
    out = submit_proposal(bundle_dir, copy.deepcopy(WATCHER),
                          policy=corpus_policy(work, ESCALATE))
    assert out["status"] == "acked" and out["verdict"] == "ESCALATE"
    return bundle_dir, out


def _resolved(mini, resolution="admit-after-review"):
    bundle_dir, out = _escalated(mini)
    entry = escalation_cli.resolve(
        bundle_dir, out["journal_seq"], resolution,
        operator=OPERATOR)
    return bundle_dir, out, entry


def _tip(bundle_dir):
    manifest, journal = _open(bundle_dir)
    try:
        return journal.tip()
    finally:
        journal.close()


def _gaia_work_rows(bundle_dir):
    manifest, journal = _open(bundle_dir)
    try:
        return journal.backend.find(
            "experiences", filter={"kind": "gaia_work"}) or []
    finally:
        journal.close()


def _golden(out):
    return {
        "type": "tool",
        "tool": "hearth.restart",
        "args": WATCHER["args"],
        "reason": WATCHER["reason"],
        "task_id": f"task@{out['task_seq']}",
        "journal_seq": out["journal_seq"],
        "verdict": "admit",
    }


def _ok(work_id="work-000777"):
    return ReplayTransport([(202, {
        "work_id": work_id, "state": "accepted",
        "journal_seq": 0})])


# -- eligibility: refusals journal NOTHING -------------------------------------

def test_unresolved_escalation_refused_journals_nothing(mini):
    bundle_dir, out = _escalated(mini)
    tip = _tip(bundle_dir)
    with pytest.raises(ValueError, match="no operator resolution"):
        spend_resolved_escalation(
            bundle_dir, out["journal_seq"], operator=OPERATOR,
            work_url=URL, transport=_ok())
    assert _tip(bundle_dir) == tip
    assert _gaia_work_rows(bundle_dir) == []


def test_reject_resolved_refused_journals_nothing(mini):
    bundle_dir, out, _entry = _resolved(mini, resolution="reject")
    tip = _tip(bundle_dir)
    with pytest.raises(ValueError, match="operator refused"):
        spend_resolved_escalation(
            bundle_dir, out["journal_seq"], operator=OPERATOR,
            work_url=URL, transport=_ok())
    assert _tip(bundle_dir) == tip
    assert _gaia_work_rows(bundle_dir) == []


def test_unknown_intent_refused_journals_nothing(mini):
    bundle_dir, _out = _escalated(mini)
    tip = _tip(bundle_dir)
    with pytest.raises(ValueError, match="no folded intent"):
        spend_resolved_escalation(
            bundle_dir, 999999, operator=OPERATOR,
            work_url=URL, transport=_ok())
    assert _tip(bundle_dir) == tip
    assert _gaia_work_rows(bundle_dir) == []


def test_anonymous_spend_refused_journals_nothing(mini):
    bundle_dir, out, _entry = _resolved(mini)
    tip = _tip(bundle_dir)
    with pytest.raises(ValueError, match="operator name"):
        spend_resolved_escalation(
            bundle_dir, out["journal_seq"], operator="  ",
            work_url=URL, transport=_ok())
    assert _tip(bundle_dir) == tip
    assert _gaia_work_rows(bundle_dir) == []


def test_no_transport_no_url_refused_journals_nothing(mini):
    bundle_dir, out, _entry = _resolved(mini)
    tip = _tip(bundle_dir)
    result = spend_resolved_escalation(
        bundle_dir, out["journal_seq"], operator=OPERATOR)
    assert result["status"] == "refused"
    assert result["journaled"] is False
    assert _tip(bundle_dir) == tip


def test_claimed_resolved_intent_refused_journals_nothing(mini):
    """A resolved intent a claim already took is no longer
    folded queued: the spend refuses (never blindly replay)."""
    bundle_dir, out, _entry = _resolved(mini)
    manifest, journal = _open(bundle_dir)
    try:
        state = fold_queue(journal).by_seq[out["journal_seq"]]
        record_queue_claim(journal, state.task_seq, state.edit_id,
                           state.intent_seq, "si116-test-drainer",
                           300, effective_class=state.effective_class,
                           ordering_key=state.ordering_key)
        updated = _anchor_checkpoint(
            journal, manifest, manifest["head"],
            note="si116 test: claim the resolved intent")
        _write_manifest(bundle_dir, updated)
    finally:
        journal.close()
    tip = _tip(bundle_dir)
    with pytest.raises(ValueError, match="not 'queued'"):
        spend_resolved_escalation(
            bundle_dir, out["journal_seq"], operator=OPERATOR,
            work_url=URL, transport=_ok())
    assert _tip(bundle_dir) == tip
    assert _gaia_work_rows(bundle_dir) == []


def test_non_seam_tool_refused_journals_nothing(mini):
    """An fs_write the gate escalated on class, resolved
    admit-after-review, folds queued — but it is not a
    seam-crossing tool and is never spent across the seam."""
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        fixture = record_task_fixture(journal, "SI-116 scenario",
                                      work)
        policy = corpus_policy(work, {"tool_class_policy": {
            "read": "admit", "write": "escalate",
            "execute": "admit", "external": "escalate"}})
        gate = Gate(policy)
        ctx = ToolContext(tool_work_root=work,
                          tool_task_seq=fixture["seq"],
                          tool_live_execution=True)
        intent = Intent(kind="tool", tool="fs_write",
                        args={"path": "notes/new.md",
                              "content": "x\n"},
                        reason="Carry out the scripted step",
                        parents=[fixture["ref"]],
                        source_seqs=[fixture["seq"]])
        out = admit_tool_intent(journal, gate, ctx, intent)
        assert out["verdict"] == "ESCALATE"
        rows = journal.backend.find(
            "self_edits", filter={"edit_id": out["edit_id"]},
            limit=1)
        seq = rows[0]["seq"]
        entry = resolve_escalation(journal, seq,
                                   "admit-after-review",
                                   operator=OPERATOR)
        assert entry["resolution"] == "admit-after-review"
        updated = _anchor_checkpoint(
            journal, manifest, manifest["head"],
            note="si116 test: resolve the fs_write escalation")
        _write_manifest(bundle_dir, updated)
    finally:
        journal.close()
    tip = _tip(bundle_dir)
    with pytest.raises(ValueError, match="seam-crossing"):
        spend_resolved_escalation(
            bundle_dir, seq, operator=OPERATOR,
            work_url=URL, transport=_ok())
    assert _tip(bundle_dir) == tip
    assert _gaia_work_rows(bundle_dir) == []


# -- the golden payload ----------------------------------------------------------

def test_payload_is_the_fork_shape_field_for_field(mini):
    bundle_dir, out, _entry = _resolved(mini)
    manifest, journal = _open(bundle_dir)
    try:
        state, edit_row = verify_spendable(journal,
                                           out["journal_seq"])
        payload = build_spend_payload(state, edit_row)
    finally:
        journal.close()
    assert payload == _golden(out)
    # parse_intent's required set, exactly — and no class
    # field (the fork defaults it to "execute", D-115-3).
    assert set(payload) == {"type", "tool", "args", "reason",
                            "task_id", "journal_seq", "verdict"}
    assert "class" not in payload
    # journal_seq is the intent's seq, never the resolution's.
    assert payload["journal_seq"] == out["journal_seq"]
    assert payload["journal_seq"] != _entry["seq"]


# -- a confirmed spend -----------------------------------------------------------

def test_spend_journals_gaia_work_tying_all_three_seqs(mini):
    bundle_dir, out, entry = _resolved(mini)
    transport = _ok()
    result = spend_resolved_escalation(
        bundle_dir, out["journal_seq"], operator=OPERATOR,
        work_url=URL, transport=transport)
    assert result["status"] == "submitted"
    assert result["deduped"] is False
    assert result["fork_work_id"] == "work-000777"
    assert result["resolution_seq"] == entry["seq"]
    assert result["http_status"] == 202
    # The wire call: one POST, the golden payload, to the URL.
    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call["method"] == "POST" and call["url"] == URL
    assert call["json_body"] == _golden(out)

    rows = _gaia_work_rows(bundle_dir)
    assert len(rows) == 1
    row = rows[0]
    assert row["seq"] == result["gaia_work_seq"]
    assert row["self_edit_id"] == out["edit_id"]
    record = row["record"]
    assert record["status"] == "submitted"
    assert record["intent_seq"] == out["journal_seq"]
    assert record["self_edit_journal_seq"] == out["journal_seq"]
    assert record["resolution"] == "admit-after-review"
    assert record["resolution_seq"] == entry["seq"]
    assert record["fork_work_id"] == "work-000777"
    assert record["operator"] == OPERATOR
    assert record["basis"] == DEFAULT_SPEND_BASIS
    assert record["submitted_via"] == SPEND_SUBMITTED_VIA
    assert record["payload"] == _golden(out)
    assert record["caused_by"] == WATCHER["caused_by"]
    # The bundle still opens fully verified (re-anchored).
    manifest, journal = _open(bundle_dir)
    try:
        assert find_spend_row(journal, out["journal_seq"])["seq"] \
            == row["seq"]
    finally:
        journal.close()


def test_second_spend_dedupes_and_calls_no_transport(mini):
    bundle_dir, out, _entry = _resolved(mini)
    first = spend_resolved_escalation(
        bundle_dir, out["journal_seq"], operator=OPERATOR,
        work_url=URL, transport=_ok())
    assert first["status"] == "submitted"
    second_transport = ReplayTransport([])
    second = spend_resolved_escalation(
        bundle_dir, out["journal_seq"], operator=OPERATOR,
        work_url=URL, transport=second_transport)
    assert second["status"] == "submitted"
    assert second["deduped"] is True
    assert second["fork_work_id"] == first["fork_work_id"]
    assert second["gaia_work_seq"] == first["gaia_work_seq"]
    assert second_transport.calls == []
    assert len(_gaia_work_rows(bundle_dir)) == 1


# -- failures are journaled as evidence, never as submissions --------------------

def test_fork_refusal_journals_failed_attempt_then_retry_spends(mini):
    bundle_dir, out, _entry = _resolved(mini)
    refused = spend_resolved_escalation(
        bundle_dir, out["journal_seq"], operator=OPERATOR,
        work_url=URL,
        transport=ReplayTransport([(409, {
            "error": {"message": "class not carried",
                      "type": "gaia_class_not_carried"}})]))
    assert refused["status"] == "spend-failed"
    assert refused["journaled"] is True
    assert refused["http_status"] == 409
    rows = _gaia_work_rows(bundle_dir)
    assert len(rows) == 1
    record = rows[0]["record"]
    assert record["status"] == "spend-failed"
    assert record["http_status"] == 409
    assert record["resolution_seq"] == _entry["seq"]
    assert record["operator"] == OPERATOR
    # Not a submission: the operator re-invocation spends.
    spent = spend_resolved_escalation(
        bundle_dir, out["journal_seq"], operator=OPERATOR,
        work_url=URL, transport=_ok("work-000888"))
    assert spent["status"] == "submitted"
    assert spent["fork_work_id"] == "work-000888"
    assert len(_gaia_work_rows(bundle_dir)) == 2
    view = read_spend_at(bundle_dir, out["journal_seq"])
    assert view["spend"]["fork_work_id"] == "work-000888"
    assert len(view["failed_attempts"]) == 1
    assert view["failed_attempts"][0]["http_status"] == 409


def test_transport_failure_journals_failed_attempt(mini):
    bundle_dir, out, _entry = _resolved(mini)
    failed = spend_resolved_escalation(
        bundle_dir, out["journal_seq"], operator=OPERATOR,
        work_url=URL,
        transport=ReplayTransport([ConnectionError("fork down")]))
    assert failed["status"] == "spend-failed"
    assert "ConnectionError" in failed["failure"]
    rows = _gaia_work_rows(bundle_dir)
    assert len(rows) == 1
    assert rows[0]["record"]["status"] == "spend-failed"
    assert "fork down" in rows[0]["record"]["failure"]


def test_202_without_work_id_is_unconfirmed_not_a_submission(mini):
    bundle_dir, out, _entry = _resolved(mini)
    result = spend_resolved_escalation(
        bundle_dir, out["journal_seq"], operator=OPERATOR,
        work_url=URL,
        transport=ReplayTransport([(202, {"state": "accepted"})]))
    assert result["status"] == "spend-unconfirmed"
    rows = _gaia_work_rows(bundle_dir)
    assert len(rows) == 1
    assert rows[0]["record"]["status"] == "spend-failed"
    manifest, journal = _open(bundle_dir)
    try:
        assert find_spend_row(journal, out["journal_seq"]) is None
    finally:
        journal.close()


def test_poll_of_unspent_intent_refused(mini):
    bundle_dir, out, _entry = _resolved(mini)
    result = poll_spent_outcome(
        bundle_dir, out["journal_seq"], work_url=URL,
        transport=ReplayTransport([]))
    assert result["status"] == "refused"
    assert result["journaled"] is False


# -- CLI wiring --------------------------------------------------------------------

def test_cli_parsers_carry_the_spend_family(mini):
    bundle_dir, out, _entry = _resolved(mini)
    module_args = escalation_cli.build_parser().parse_args(
        ["spend", bundle_dir, str(out["journal_seq"]),
         "--operator", OPERATOR, "--json"])
    assert module_args.esc_cmd == "spend"
    assert module_args.operator == OPERATOR
    si_args = build_parser().parse_args(
        ["gaia-escalation", bundle_dir, "poll-outcome",
         str(out["journal_seq"]), "--json"])
    assert si_args.esc_cmd == "poll-outcome"
    assert si_args.intent_seq == out["journal_seq"]
    si_args = build_parser().parse_args(
        ["gaia-escalation", bundle_dir, "reconcile-outcome",
         str(out["journal_seq"]), "--outcomes", "o.jsonl"])
    assert si_args.esc_cmd == "reconcile-outcome"
    assert si_args.outcomes == "o.jsonl"


# -- loopback end-to-end: fork double over real HTTP -------------------------------

class _ForkDouble:
    """An in-process fork-route stub implementing the
    as-built rules read from gateway.rs/work.rs:
    parse_intent's required set, the 202 ACK shape, S-1
    dedup by journal_seq, carried classes {read, write,
    execute} (absent class defaults to execute), and the
    status route — an admitted record actuates on read:
    claim gates pass, the answered loss is stale (SI-115
    §2c), so it completes executed with a refused /
    stale_state result. A test double, not a fork."""

    CARRIED = {"read", "write", "execute"}

    def __init__(self):
        self.records = {}
        self.by_seq = {}
        self.next = 1
        self.posts = []
        self.gets = []

    def submit(self, body):
        self.posts.append(body)
        if not isinstance(body, dict) or body.get("type") != "tool":
            return 400, {"error": {"type": "gaia_malformed_request",
                                   "message": "type must be tool"}}
        for field in ("tool", "reason", "task_id"):
            if not isinstance(body.get(field), str) \
                    or not body.get(field):
                return 400, {"error": {
                    "type": "gaia_malformed_request",
                    "message": f"{field} required"}}
        seq = body.get("journal_seq")
        if not isinstance(seq, int) or isinstance(seq, bool) \
                or seq < 0:
            return 400, {"error": {"type": "gaia_malformed_request",
                                   "message": "journal_seq required"}}
        verdict = body.get("verdict")
        if verdict is not None and verdict not in ("admit", "reject"):
            return 400, {"error": {"type": "gaia_malformed_request",
                                   "message": "unknown verdict"}}
        if body.get("class", "execute") not in self.CARRIED:
            return 409, {"error": {"type": "gaia_class_not_carried",
                                   "message": "class not carried"}}
        if seq in self.by_seq:  # S-1 dedup
            work_id = self.by_seq[seq]
            return 202, {"work_id": work_id, "state": "accepted",
                         "journal_seq": seq}
        work_id = f"work-{self.next:06d}"
        self.next += 1
        self.by_seq[seq] = work_id
        self.records[work_id] = {
            "intent": body, "state": "admitted", "result": None}
        return 202, {"work_id": work_id, "state": "accepted",
                     "journal_seq": seq}

    def status(self, work_id):
        self.gets.append(work_id)
        record = self.records.get(work_id)
        if record is None:
            return 404, {"error": {"type": "gaia_work_unknown",
                                   "message": "unknown work id"}}
        if record["state"] == "admitted":
            intent = record["intent"]
            record["result"] = {
                "outcome": "refused",
                "tool": intent["tool"],
                "model": (intent.get("args") or {}).get("model"),
                "refusal": "stale_state",
                "detail": "the answered loss already healed "
                          "(SI-115 §2c, fork double)",
                "caused_by": WATCHER["caused_by"],
                "spine_event_refs": [],
            }
            record["state"] = "executed"
        body = {"work_id": work_id, "state": record["state"],
                "provenance": {
                    "si_journal_seqs": [
                        record["intent"]["journal_seq"]],
                    "spine_event_refs": []}}
        if record["result"] is not None:
            body["result"] = record["result"]
        return 200, body


@pytest.fixture()
def fork_double():
    double = _ForkDouble()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the test log clean
            pass

        def _send(self, status, payload):
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            if self.path.rstrip("/") == "/gaia/v1/work":
                try:
                    body = json.loads(raw.decode("utf-8"))
                except ValueError:
                    body = None
                status, payload = double.submit(body)
                self._send(status, payload)
            else:
                self._send(404, {"error": {
                    "type": "gaia_route_not_found"}})

        def do_GET(self):
            prefix = "/gaia/v1/work/"
            if self.path.startswith(prefix):
                work_id = self.path[len(prefix):].strip("/")
                status, payload = double.status(work_id)
                self._send(status, payload)
            else:
                self._send(404, {"error": {
                    "type": "gaia_route_not_found"}})

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever,
                              daemon=True)
    thread.start()
    try:
        yield double, f"http://127.0.0.1:{server.server_port}" \
            "/gaia/v1/work"
    finally:
        server.shutdown()
        server.server_close()


def test_loopback_end_to_end_spend_poll_reconcile(mini, fork_double,
                                                 tmp_path):
    double, work_url = fork_double
    bundle_dir, out, entry = _resolved(mini)

    spent = spend_resolved_escalation(
        bundle_dir, out["journal_seq"], operator=OPERATOR,
        work_url=work_url, transport=HttpTransport())
    assert spent["status"] == "submitted"
    assert spent["fork_work_id"] == "work-000001"
    assert spent["transport"] == "http"
    assert double.posts == [_golden(out)]

    # A resubmission straight at the double dedups fork-side
    # (S-1): same work id, no second record.
    status, raw = HttpTransport()(
        "POST", work_url, json_body=_golden(out))
    assert status == 202
    assert json.loads(raw.decode("utf-8"))["work_id"] == "work-000001"
    assert len(double.records) == 1

    polled = poll_spent_outcome(
        bundle_dir, out["journal_seq"], work_url=work_url,
        transport=HttpTransport())
    assert polled["status"] == "journaled"
    assert polled["deduped"] is False
    assert polled["outcome"] == "refused"

    manifest, journal = _open(bundle_dir)
    try:
        state = fold_queue(journal).by_seq[out["journal_seq"]]
        assert state.state == "refused-at-claim"
        assert state.outcome_status == "refused"
        assert state.resolution_seq == entry["seq"]
        outcomes = [r for r in journal.backend.find(
            "experiences", filter={"kind": "tool_result"}) or []
            if (r.get("record") or {}).get("source") == "gaia-seam"]
        assert len(outcomes) == 1
        record = outcomes[0]["record"]
        assert record["work_id"] == "work-000001"
        assert record["journal_seq"] == out["journal_seq"]
        assert record["fork_result"]["refusal"] == "stale_state"
    finally:
        journal.close()

    again = poll_spent_outcome(
        bundle_dir, out["journal_seq"], work_url=work_url,
        transport=HttpTransport())
    assert again["status"] == "journaled"
    assert again["deduped"] is True
    assert again["outcome_seq"] == polled["outcome_seq"]

    respent = spend_resolved_escalation(
        bundle_dir, out["journal_seq"], operator=OPERATOR,
        work_url=work_url, transport=HttpTransport())
    assert respent["deduped"] is True
    assert len(double.posts) == 2  # only the manual S-1 probe added one

    outcomes_path = tmp_path / "outcomes.jsonl"
    result = double.records["work-000001"]["result"]
    outcomes_path.write_text(json.dumps({
        "work_id": "work-000001", "tool": result["tool"],
        "model": result["model"], "outcome": result["outcome"],
        "refusal": result["refusal"], "detail": result["detail"],
        "caused_by": result["caused_by"],
        "spine_event_refs": result["spine_event_refs"],
        "result": result}) + "\n")
    reconciled = reconcile_spent_outcome(
        bundle_dir, out["journal_seq"], str(outcomes_path))
    assert reconciled["status"] == "reconciled"
    assert reconciled["count"] == 0
    assert [e["work_id"] for e in reconciled["cross_checked"]] \
        == ["work-000001"]
    assert reconciled["cursor"] > 0
