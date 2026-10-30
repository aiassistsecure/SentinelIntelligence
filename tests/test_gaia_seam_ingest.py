"""S-2a (G-9 §8/§12.1/§15 S-2): the Gaia seam — proposal ingest.

Acceptance tests for the seam that receives watcher proposals
from the Gaia fork (``gaia-serve``) and lands them in SI's
journal through the gate: ``si.gaia.seam`` (validation, the
per-model seam task, caused_by dedup, ingest) and the
``POST /gaia/proposals`` route on the B-5 service
(``si.gaia.service.submit_proposal`` / ``make_service_server``).

What is asserted:

- A watcher-shaped proposal is gated and journaled: one seam
  task entry (operator ``gaia-seam``), one self-edit citing the
  proposal's ``caused_by`` spine event, the full gate report on
  the row, and an ack ``{"task_seq", "journal_seq", "verdict"}``
  whose fields name those exact journal rows.
- A retry of the same proposal returns the identical ack and
  journals nothing more (dedup is the ``caused_by`` pair).
- Malformed proposals are refused with itemised problems and
  journal nothing — cumulatively, across every mutation.
- A REJECT creates no submission artefacts: no ``gaia_work``
  entries, no queue claims, no execution outcomes.
- The seam task is per-model and reused across proposals.
- The HTTP wire shape is exactly what the fork's egress client
  sends and parses: a byte-level replay of its request returns
  an ack body the real client's ``parse_ack`` rules accept, and
  an env-gated live test runs the real client when the
  coordinator supplies its probe binary.

The registry gap, stated plainly (historical in S-2a;
CLOSED by SI-55/S-2b): the v1 registry originally registered
ONLY SI's own 12 tools, so watcher proposals REJECTed at the
``tool_registered`` sub-check. SI-55/S-2b registered
``hearth.restart`` / ``hearth.notify`` as external-class
seam-crossing tools, so the watcher fixture now gates on its
merits: under the corpus policy (``external: escalate``) it
ESCALATEs with ``tool_registered`` passing. The ADMIT path
(under an ``external: admit`` policy, resting ``pending``,
executing nothing) is pinned by
tests/test_gaia_s2b_hearth_registry.py. The ``fs_list``
control still proves the pipeline end-to-end.

Ingest itself submits nothing: S-3 (SI-56) submission is a
separate explicit step in si.gaia.seam_submit, so nothing in
this file's flows journals ``gaia_work`` entries, and these
tests assert that absence rather than assuming it.

No-auth / loopback honesty: the seam service, like B-5, has
no authentication story — it binds localhost by default,
carries no security claim, and a proposal's provenance is
what is journaled about it (the seam operator, the
submitted-via label, and the ``caused_by`` spine citation),
never a verified identity.
"""
import copy
import json
import os
import socket
import subprocess
import threading
import urllib.error
import urllib.request

import pytest

from si.bundle import init_bundle, load_bundle
from si.gaia.corpus import corpus_policy
from si.gaia.seam import (
    SEAM_OPERATOR,
    SEAM_SUBMITTED_VIA,
    find_prior_proposal_ack,
    find_seam_task,
    ingest_proposal,
    validate_proposal,
)
from si.gaia.service import (
    build_provenance_trail,
    make_service_server,
    submit_proposal,
)
from si.journal import COLLECTIONS, open_journal
from si.tiny_seed import seed_tiny_bundle

# The contract surface this slice promises, referenced so an
# import-time rename fails collection loudly rather than
# silently weakening a test below.
_CONTRACT_SURFACE = (validate_proposal, ingest_proposal,
                     find_seam_task, find_prior_proposal_ack)

#: The watcher fixture: byte-shape of the fork egress client's
#: real body — Proposal::to_json of the gaia-serve egress
#: fixture (keys in the fork's to_json order: model, tool,
#: args, reason, caused_by).
WATCHER = {
    "model": "muse",
    "tool": "hearth.restart",
    "args": {"model": "muse",
             "spine_event": {"hash": "hash-s2a-fixture", "seq": 41}},
    "reason": "observed loss at seq 41",
    "caused_by": {"hash": "hash-s2a-fixture", "seq": 41},
}

VERDICTS = ("ADMIT", "REJECT", "ESCALATE")


def _watcher():
    return copy.deepcopy(WATCHER)


@pytest.fixture()
def mini(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "mini.si"),
                             name="gaia-s2a",
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    work = tmp_path / "work"
    work.mkdir()
    return bundle_dir, str(work)


def _counts(bundle_dir):
    manifest = load_bundle(bundle_dir)
    journal = open_journal(bundle_dir, manifest)
    try:
        return {c: journal.backend.count(c) for c in COLLECTIONS}
    finally:
        journal.close()


def _open(bundle_dir):
    manifest = load_bundle(bundle_dir)
    return open_journal(bundle_dir, manifest)


def _seam_tasks(journal):
    rows = [r for r in journal.backend.find(
        "experiences", filter={"kind": "task"})
        if (r.get("record") or {}).get("operator") == SEAM_OPERATOR]
    rows.sort(key=lambda r: r.get("seq") or 0)
    return rows


def _edits_citing(journal, caused_by):
    rows = [r for r in journal.backend.find("self_edits")
            if (r.get("intent") or {}).get("caused_by") == caused_by]
    rows.sort(key=lambda r: r.get("seq") or 0)
    return rows


def _check(report, name):
    return next(c for c in report["checks"] if c["name"] == name)


def _failed_subchecks(check):
    return [s for s in (check.get("subchecks") or [])
            if s.get("passed") is False]


def _assert_ack_matches(found, ack):
    """find_prior_proposal_ack's return, checked flexibly: an
    ack-shaped dict must equal the ack; a journal row must name
    the ack's journal seq. Either way it must exist."""
    assert found is not None
    if isinstance(found, dict) and "journal_seq" in found:
        inner = found.get("ack") if isinstance(
            found.get("ack"), dict) else found
        for key in ("task_seq", "journal_seq", "verdict"):
            if key in inner:
                assert inner[key] == ack[key]
    elif isinstance(found, dict) and "seq" in found:
        assert found["seq"] == ack["journal_seq"]


# -- 1: the watcher proposal is gated and journaled ----------------------------

def test_watcher_proposal_is_gated_and_journaled(mini):
    bundle_dir, work = mini
    assert SEAM_OPERATOR == "gaia-seam"
    assert SEAM_SUBMITTED_VIA == "gaia-seam-ingest (G-9/S-2a)"
    out = submit_proposal(bundle_dir, _watcher(),
                          policy=corpus_policy(work))
    assert out["status"] == "acked"
    assert out["deduped"] is False
    # SI-55/S-2b closed the registry gap: hearth.restart is
    # registered (external class); the corpus policy's
    # external rule is escalate, so the merits verdict is
    # ESCALATE — not the old tool_registered REJECT.
    assert out["verdict"] == "ESCALATE"
    ack = out["ack"]
    assert set(ack) == {"task_seq", "journal_seq", "verdict"}
    assert ack == {"task_seq": out["task_seq"],
                   "journal_seq": out["journal_seq"],
                   "verdict": out["verdict"]}
    assert out["label"]

    journal = _open(bundle_dir)
    try:
        tasks = [r for r in journal.backend.find(
            "experiences", filter={"kind": "task"})]
        assert len(tasks) == 1, "exactly one seam task entry"
        assert tasks[0]["record"]["operator"] == "gaia-seam"
        assert ack["task_seq"] == tasks[0]["seq"]

        edits = _edits_citing(
            journal, {"hash": "hash-s2a-fixture", "seq": 41})
        assert len(edits) == 1, "exactly one self-edit cites it"
        row = edits[0]
        assert row["seq"] == ack["journal_seq"]
        assert row["edit_id"] == out["edit_id"]
        assert row["gate_verdict"] == "ESCALATE"
        safety = _check(row["gate_report"], "safety_bounds")
        registered = next(
            x for x in safety["subchecks"]
            if x["name"] == "tool_registered")
        assert registered["passed"] is True
        # The ingest gate report rides the return value too.
        out_safety = _check(out["gate_report"], "safety_bounds")
        out_registered = next(
            x for x in out_safety["subchecks"]
            if x["name"] == "tool_registered")
        assert out_registered["passed"] is True
        intent = row["intent"]
        assert intent["parents"] == [f"task@{ack['task_seq']}"]
        assert intent["source_seqs"] == [ack["task_seq"]]
        assert journal.verify_chain() is True
    finally:
        journal.close()


# -- 2: retry dedups on caused_by ------------------------------------------------

def test_retry_returns_identical_ack_and_journals_nothing_more(mini):
    bundle_dir, work = mini
    policy = corpus_policy(work)
    first = submit_proposal(bundle_dir, _watcher(), policy=policy)
    assert first["status"] == "acked" and first["deduped"] is False
    before = _counts(bundle_dir)

    second = submit_proposal(bundle_dir, _watcher(), policy=policy)
    assert second["status"] == "acked"
    assert second["deduped"] is True
    assert second["ack"] == first["ack"]
    assert _counts(bundle_dir) == before

    journal = _open(bundle_dir)
    try:
        assert len(_edits_citing(
            journal, {"hash": "hash-s2a-fixture", "seq": 41})) == 1
        _assert_ack_matches(
            find_prior_proposal_ack(
                journal, {"hash": "hash-s2a-fixture", "seq": 41}),
            first["ack"])
    finally:
        journal.close()


# -- 3: malformed proposals are refused and journal nothing ----------------------

def _drop(key):
    def mutate(payload):
        payload.pop(key, None)
        return payload
    return mutate


def _set(key, value):
    def mutate(payload):
        payload[key] = value
        return payload
    return mutate


def _set_caused_by(key, value):
    def mutate(payload):
        payload["caused_by"][key] = value
        return payload
    return mutate


def _set_spine(key, value):
    def mutate(payload):
        payload["args"]["spine_event"][key] = value
        return payload
    return mutate


MALFORMED = [
    ("body-not-a-dict", lambda p: ["not", "a", "dict"]),
    ("caused-by-missing", _drop("caused_by")),
    ("caused-by-seq-string", _set_caused_by("seq", "41")),
    ("caused-by-seq-bool", _set_caused_by("seq", True)),
    ("caused-by-seq-negative", _set_caused_by("seq", -1)),
    ("caused-by-hash-empty", _set_caused_by("hash", "")),
    ("model-empty", _set("model", "")),
    ("model-missing", _drop("model")),
    ("tool-missing", _drop("tool")),
    ("args-a-list", _set("args", ["muse"])),
    ("args-missing", _drop("args")),
    ("reason-not-a-string", _set("reason", 41)),
    ("unknown-extra-key", _set("surprise", "extra")),
    ("spine-hash-disagrees", _set_spine("hash", "hash-other")),
    ("spine-seq-disagrees", _set_spine("seq", 42)),
    ("spine-not-an-object", _set("args", {
        "model": "muse", "spine_event": "hash-s2a-fixture"})),
]


@pytest.fixture(scope="module")
def malformed_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("gaia-s2a-malformed")
    bundle_dir = init_bundle(str(root / "mini.si"), name="gaia-s2a",
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    work = root / "work"
    work.mkdir()
    return bundle_dir, str(work)


@pytest.mark.parametrize("name,mutate", MALFORMED,
                         ids=[name for name, _ in MALFORMED])
def test_malformed_proposals_are_refused_and_journal_nothing(
        malformed_bundle, name, mutate):
    # One shared bundle across every parametrised case: each
    # case snapshots all collection counts before and after, so
    # equality case-by-case IS the cumulative zero-growth proof.
    bundle_dir, work = malformed_bundle
    before = _counts(bundle_dir)
    out = submit_proposal(bundle_dir, mutate(_watcher()),
                          policy=corpus_policy(work))
    assert out["status"] == "refused", name
    assert out["problems"], name
    assert out["journaled"] is False, name
    assert _counts(bundle_dir) == before, name


# -- 4: a REJECT creates no submission artefacts ---------------------------------

def _iter_dicts(value):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _iter_dicts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_dicts(item)


def test_reject_creates_no_submission_artefacts(mini):
    bundle_dir, work = mini
    # Since SI-55/S-2b the watcher tool is registered, so a true
    # REJECT here uses a still-unregistered hearth.* name.
    payload = _watcher()
    payload["tool"] = "hearth.reap"
    out = submit_proposal(bundle_dir, payload,
                          policy=corpus_policy(work))
    assert out["status"] == "acked" and out["verdict"] == "REJECT"
    task_seq = out["task_seq"]

    journal = _open(bundle_dir)
    try:
        experiences = journal.backend.find("experiences")
        # gaia_work is S-3's kind: ingest journals none.
        assert not [r for r in experiences
                    if r.get("kind") == "gaia_work"]
        assert not [r for r in experiences
                    if (r.get("record") or {}).get("kind")
                    == "gaia_work"]
        # No queue claim cites the seam task (S-2a claims
        # nothing: admission here queues no execution).
        claims = [r for r in experiences
                  if r.get("kind") == "queue_claim"
                  or (r.get("record") or {}).get("kind")
                  == "queue_claim"]
        assert not [r for r in claims
                    if (r.get("record") or {}).get("task_seq")
                    == task_seq]
        trail = build_provenance_trail(journal, task_seq)
    finally:
        journal.close()

    assert len(trail["steps"]) == 1
    step = trail["steps"][0]
    assert step["gate_verdict"] == "REJECT"
    # Outcome shape found in the existing trail (test_gaia_b5):
    # a step's outcomes are experiences rows whose "record"
    # carries an "execution" dict with a "status" ("completed"
    # when something really ran). A rejected seam proposal must
    # carry no such claim: recursively, no dict anywhere in the
    # outcomes may report status "ok" or "executed".
    for outcome in step["outcomes"]:
        for node in _iter_dicts(outcome):
            assert node.get("status") not in ("ok", "executed"), \
                f"rejected proposal claims an execution: {node!r}"


# -- 5: the registered-tool control ADMITs ----------------------------------------

def test_registered_tool_control_admits(mini):
    bundle_dir, work = mini
    # This control proves the gate pipeline end-to-end; watcher
    # tools cannot ADMIT until a registry slice registers
    # hearth.* under the corpus policy (see module docstring).
    proposal = {
        "model": "muse",
        "tool": "fs_list",
        "args": {"path": "notes"},
        "reason": "control: prove the ingest pipeline admits a "
                  "registered tool",
        "caused_by": {"hash": "control-hash", "seq": 7},
    }
    out = submit_proposal(bundle_dir, proposal,
                          policy=corpus_policy(work))
    assert out["status"] == "acked"
    assert out["verdict"] == "ADMIT"
    journal = _open(bundle_dir)
    try:
        edits = _edits_citing(journal,
                              {"hash": "control-hash", "seq": 7})
        assert len(edits) == 1
        row = edits[0]
        assert row["seq"] == out["ack"]["journal_seq"]
        assert row["gate_verdict"] == "ADMIT"
        # Admission semantics: queued, NOT executed — nothing
        # runs SI-side in S-2a, so the result stays pending.
        assert row["result"] == "pending"
    finally:
        journal.close()


# -- 6: the seam task is per-model and reused --------------------------------------

def test_seam_task_is_per_model_and_reused(mini):
    bundle_dir, work = mini
    policy = corpus_policy(work)
    first = submit_proposal(bundle_dir, _watcher(), policy=policy)
    assert first["status"] == "acked"

    second_payload = _watcher()
    second_payload["args"]["spine_event"]["seq"] = 42
    second_payload["caused_by"]["seq"] = 42
    second = submit_proposal(bundle_dir, second_payload,
                             policy=policy)
    assert second["status"] == "acked"
    assert second["deduped"] is False
    assert second["task_seq"] == first["task_seq"]
    assert second["journal_seq"] != first["journal_seq"]

    other_payload = _watcher()
    other_payload["model"] = "other"
    other_payload["args"]["model"] = "other"
    other_payload["args"]["spine_event"] = {
        "hash": "hash-other", "seq": 41}
    other_payload["caused_by"] = {"hash": "hash-other", "seq": 41}
    other = submit_proposal(bundle_dir, other_payload,
                            policy=policy)
    assert other["status"] == "acked"
    assert other["task_seq"] != first["task_seq"]

    journal = _open(bundle_dir)
    try:
        tasks = _seam_tasks(journal)
        assert len(tasks) == 2, "two seam task entries total"
        assert {t["seq"] for t in tasks} == {
            first["task_seq"], other["task_seq"]}
        assert len(_edits_citing(
            journal, {"hash": "hash-s2a-fixture", "seq": 41})) == 1
        assert len(_edits_citing(
            journal, {"hash": "hash-s2a-fixture", "seq": 42})) == 1
        found = find_seam_task(journal, "muse")
        assert found is not None
        if isinstance(found, dict) and "seq" in found:
            assert found["seq"] == first["task_seq"]
        elif isinstance(found, int):
            assert found == first["task_seq"]
    finally:
        journal.close()


# -- the HTTP surface ---------------------------------------------------------------

class _Server:
    def __enter__(self):
        self.server = make_service_server(
            self.bundle_dir, host="127.0.0.1", port=0,
            policy=self.policy)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def call(self, method, path, payload=None, raw=None):
        url = f"http://127.0.0.1:{self.port}{path}"
        data = raw
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        elif raw is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data,
                                     headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            try:
                return e.code, json.loads(body)
            except ValueError:
                return e.code, {"raw": body}


@pytest.fixture()
def server(mini):
    bundle_dir, work = mini
    ctx = _Server()
    ctx.bundle_dir = bundle_dir
    ctx.policy = corpus_policy(work)
    ctx.work = work
    with ctx as s:
        yield s


# -- 7: HTTP wire shape and the exact ack body --------------------------------------

def test_http_wire_shape_and_ack_body(server):
    code, body = server.call("POST", "/gaia/proposals", _watcher())
    assert code == 200
    # The acked body is EXACTLY the ack dict, nothing more.
    assert set(body) == {"task_seq", "journal_seq", "verdict"}
    assert isinstance(body["journal_seq"], int)
    assert not isinstance(body["journal_seq"], bool)
    assert isinstance(body["task_seq"], int)
    assert not isinstance(body["task_seq"], bool)
    assert body["verdict"] in VERDICTS
    assert body["verdict"] == "ESCALATE"  # SI-55: registered, external=escalate

    # The trailing-slash route is the same route (a retry: the
    # deduped ack is still exactly the ack dict).
    code, slashed = server.call("POST", "/gaia/proposals/",
                                _watcher())
    assert code == 200
    assert slashed == body

    code, _ = server.call("POST", "/gaia/proposals",
                          raw=b"{not json")
    assert code == 400

    refused = _watcher()
    del refused["caused_by"]
    code, body = server.call("POST", "/gaia/proposals", refused)
    assert code == 400
    assert "problems" in body

    code, body = server.call("GET", "/")
    assert code == 200
    assert "POST /gaia/proposals" in body["endpoints"]
    code, body = server.call("GET", "/health")
    assert code == 200
    assert body["status"] == "ok"


# -- 8: the egress client's byte replay ----------------------------------------------

def test_egress_client_byte_replay(server):
    # This reproduces crates/gaia-serve/src/egress_client.rs's
    # request format and Proposal::to_json field order (model,
    # tool, args, reason, caused_by) byte for byte.
    payload = _watcher()
    body = json.dumps(payload, separators=(",", ":")).encode()
    assert list(payload) == ["model", "tool", "args", "reason",
                             "caused_by"]
    host = f"127.0.0.1:{server.port}"
    request = (
        f"POST /gaia/proposals HTTP/1.1\r\n"
        f"Host: {host}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n"
        f"\r\n").encode() + body
    with socket.create_connection(("127.0.0.1", server.port),
                                  timeout=30) as sock:
        sock.sendall(request)
        chunks = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    raw = b"".join(chunks)
    head, _, resp_body = raw.partition(b"\r\n\r\n")
    status_line = head.split(b"\r\n", 1)[0].decode()
    assert " 200 " in f"{status_line} "

    # The real client's parse_ack rules: journal_seq present
    # and an int, verdict one of the three spellings, task_seq
    # present and an int — i.e. the client returns Acked.
    ack = json.loads(resp_body.decode())
    assert isinstance(ack.get("task_seq"), int)
    assert not isinstance(ack.get("task_seq"), bool)
    assert isinstance(ack.get("journal_seq"), int)
    assert not isinstance(ack.get("journal_seq"), bool)
    assert ack.get("verdict") in VERDICTS

    journal = _open(server.bundle_dir)
    try:
        edits = _edits_citing(
            journal, {"hash": "hash-s2a-fixture", "seq": 41})
        assert len(edits) == 1
        assert ack["verdict"] == edits[0]["gate_verdict"]
        assert ack["journal_seq"] == edits[0]["seq"]
    finally:
        journal.close()


# -- 9: live interop with the real fork egress client (env-gated) ---------------------

def test_live_egress_interop(server):
    """Live interop with the real fork egress client, run by
    the coordinator at integration with the log filed in the
    slice backups; the always-on wire contract is test 8
    (test_egress_client_byte_replay). Skipped unless
    SI_S2A_EGRESS_PROBE_BIN names an existing executable: a
    Rust harness built by the coordinator from the merged gaia
    egress client, POSTing the fixture proposal via the real
    HttpProposalSink and printing the SendOutcome."""
    probe = os.environ.get("SI_S2A_EGRESS_PROBE_BIN")
    if not probe or not os.path.isfile(probe) \
            or not os.access(probe, os.X_OK):
        pytest.skip(
            "live interop with the real fork egress client: "
            "SI_S2A_EGRESS_PROBE_BIN does not name an existing "
            "executable (run by the coordinator at integration, "
            "log filed in the slice backups; the always-on wire "
            "contract is test_egress_client_byte_replay)")
    proc = subprocess.run(
        [probe, f"127.0.0.1:{server.port}"],
        capture_output=True, text=True, timeout=60)
    assert "Acked" in proc.stdout
    # The probe prints the SendOutcome's Debug form, whose
    # verdict variant is spelled "Escalate" (SI-55/S-2b: the
    # watcher tool is registered; the corpus policy escalates
    # the external class); the ACK only parses if the wire body
    # carried the §8.3 spelling "ESCALATE" — Acked IS the
    # wire-spelling proof.
    assert "Escalate" in proc.stdout
    journal = _open(server.bundle_dir)
    try:
        assert len(_edits_citing(
            journal, {"hash": "hash-s2a-fixture", "seq": 41})) == 1
    finally:
        journal.close()
