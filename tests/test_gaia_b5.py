"""SI-46 (SI-41 RFC §15 B-5, §12.5(a)): agentic work as a service.

B-5 builds, SI-side only, in si/gaia/service.py: a stdlib-HTTP
service in `si serve`'s pattern that accepts a task — with
submitter identity as a mandatory journaled provenance field
(no identity, no task) — runs it through the B-3 machinery, and
returns the result with the provenance trail serialised from
the journal.

Honesty discipline asserted throughout: the submitter identity
is asserted, never authenticated (no credential exists to
check); a refused submission journals nothing and runs nothing;
exposure through the Hearth gateway is proposal H-2, NOT
built; execution scope is B-3's unchanged (miniature live,
demo shadow); no trained mind proposes intents — the proposer
is the scripted script of RFC §14.1, as with `si task`.
"""
import json
import os
import threading
import urllib.error
import urllib.request

import pytest

from si.bundle import init_bundle, load_bundle
from si.gaia.corpus import corpus_policy
from si.gaia.service import (
    GATEWAY_EXPOSURE_LABEL,
    SUBMITTED_VIA,
    build_provenance_trail,
    intents_from_script,
    make_service_server,
    provenance_trail_for_bundle,
    submit_task,
    validate_submission,
)
from si.gaia.shadow import walk_tool_provenance
from si.gaia.task import SUBMITTER_IDENTITY_LABEL
from si.gate import TOOL_SHADOW_EXECUTION
from si.journal import COLLECTIONS, open_journal
from si.tiny_seed import seed_tiny_bundle

SUBMITTER = "pin-node-7 (test submitter)"


@pytest.fixture()
def mini(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "mini.si"), name="gaia-b5",
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


def _submission(work, **over):
    payload = {
        "submitter": SUBMITTER,
        "instruction": "Do the submitted B-5 task",
        "work_root": work,
        "script": [
            {"kind": "tool", "tool": "fs_write",
             "args": {"path": "service-made.txt",
                      "content": "via the B-5 service\n"},
             "reason": "Write the submitted task's output"},
            {"kind": "tool", "tool": "shell_exec",
             "args": {"command": "echo service-shell-step"},
             "reason": "Run the submitted task's shell step"},
            {"kind": "memory",
             "record": {"kind": "task_complete",
                        "detail": "scripted completion report"},
             "reason": "Report completion"},
        ],
    }
    payload.update(over)
    return payload


# -- submission validation: no identity, no task (§12.5(a)) --------------------

@pytest.mark.parametrize("mutate", [
    lambda p: p.pop("submitter"),
    lambda p: p.update(submitter=""),
    lambda p: p.update(submitter="   "),
    lambda p: p.update(submitter=42),
    lambda p: p.update(submitter=None),
])
def test_submitter_identity_is_mandatory(mutate):
    payload = {"submitter": SUBMITTER, "instruction": "x",
               "script": [{"kind": "tool", "tool": "fs_list",
                           "args": {}}]}
    mutate(payload)
    normalised, problems = validate_submission(payload)
    assert normalised is None
    assert any("submitter is mandatory" in p for p in problems)
    assert any("no identity, no task" in p for p in problems)


@pytest.mark.parametrize("payload,needle", [
    ({"submitter": "s", "script": [{"tool": "fs_list"}]},
     "instruction is mandatory"),
    ({"submitter": "s", "instruction": "x"}, "script is mandatory"),
    ({"submitter": "s", "instruction": "x", "script": []},
     "script is mandatory"),
    ({"submitter": "s", "instruction": "x", "script": ["nope"]},
     "must all be JSON objects"),
    ({"submitter": "s", "instruction": "x",
      "script": [{"kind": "delta"}]}, 'kind must be "tool" or "memory"'),
    ({"submitter": "s", "instruction": "x",
      "script": [{"kind": "tool"}]}, "must name its tool"),
    ({"submitter": "s", "instruction": "x",
      "script": [{"tool": "fs_list"}], "cycles": 0},
     "cycles must be a positive integer"),
    ("not-an-object", "must be a JSON object"),
])
def test_submission_validation_itemises_problems(payload, needle):
    normalised, problems = validate_submission(payload)
    assert normalised is None
    assert any(needle in p for p in problems)


def test_valid_submission_normalises():
    normalised, problems = validate_submission(_submission("/w"))
    assert problems == []
    assert normalised["submitter"] == SUBMITTER
    assert normalised["work_root"] == "/w"
    intents = intents_from_script(normalised["script"])
    assert [i.kind for i in intents] == ["tool", "tool", "memory"]
    assert intents[0].tool == "fs_write"
    assert intents[2].record["kind"] == "task_complete"


def test_refused_submission_journals_nothing_and_runs_nothing(mini):
    bundle_dir, work = mini
    before = _counts(bundle_dir)
    out = submit_task(bundle_dir, {"instruction": "no identity here",
                                   "script": [{"tool": "fs_write",
                                               "args": {
                                                   "path": "nope.txt",
                                                   "content": "x"}}]},
                      policy=corpus_policy(work))
    assert out["status"] == "refused"
    assert out["journaled"] is False
    assert _counts(bundle_dir) == before
    assert not os.path.exists(os.path.join(work, "nope.txt"))


# -- end-to-end: submission -> B-3 machinery -> result + trail ------------------

def test_submitted_task_runs_and_returns_walkable_trail(mini):
    bundle_dir, work = mini
    out = submit_task(bundle_dir, _submission(work),
                      policy=corpus_policy(work))
    assert out["status"] == "task-returned"
    assert out["submitter"] == SUBMITTER
    assert out["submitted_via"] == SUBMITTED_VIA
    assert "NOT authentication" in out["submitter_identity_label"]
    assert "H-2" in out["gateway_exposure"]

    result = out["result"]
    assert result["stop_reason"] == "mind-reported-complete"
    assert result["cycles_used"] == 2
    assert [s["verdict"] for s in result["steps"]] == ["ADMIT"] * 3
    # The work really happened (B-3 live scope, miniature).
    assert open(os.path.join(work, "service-made.txt")).read() == \
        "via the B-5 service\n"

    trail = out["provenance_trail"]
    assert trail["task_seq"] == result["task_seq"]
    assert trail["task_ref"] == result["task_ref"]
    assert trail["submitter"] == SUBMITTER
    assert trail["journal_chain_verified"] is True

    # The task entry carries the identity as a journaled field.
    entry = trail["task_entry"]["record"]
    assert entry["kind"] == "task"
    assert entry["instruction"] == "Do the submitted B-5 task"
    assert entry["submitter"] == SUBMITTER
    assert entry["submitted_via"] == SUBMITTED_VIA
    assert entry["submitter_identity_label"] == \
        SUBMITTER_IDENTITY_LABEL
    assert "fixture" not in entry

    # Every step: intent + full gate report + provenance + outcome,
    # in journal seq order, chained back to the task.
    steps = trail["steps"]
    assert len(steps) == 3
    assert [s["seq"] for s in steps] == sorted(s["seq"] for s in steps)
    assert steps[0]["intent"]["tool"] == "fs_write"
    assert steps[0]["gate_verdict"] == "ADMIT"
    assert steps[0]["gate_report"]["checks"], \
        "the gate report rides the trail, itemised"
    assert result["task_ref"] in steps[0]["intent"]["parents"]
    for step in steps:
        assert step["provenance"], "each step carries its edges"
        assert step["outcomes"], "each step carries its outcome"
    outcomes0 = steps[0]["outcomes"][0]["record"]
    assert outcomes0["kind"] == "tool_result"
    assert outcomes0["execution"]["status"] == "completed"
    # The stop closes the trail.
    stop = trail["task_stop"]["record"]
    assert stop["cause"] == "mind-reported-complete"
    assert stop["task_seq"] == result["task_seq"]

    # The trail is walkable by the existing tooling, too: `si why`'s
    # walk reaches the same task entry, submitter included.
    manifest = load_bundle(bundle_dir)
    journal = open_journal(bundle_dir, manifest)
    try:
        walk = walk_tool_provenance(journal, steps[0]["edit_id"])
        assert walk["task"]["row"]["seq"] == result["task_seq"]
        assert walk["task"]["row"]["record"]["submitter"] == SUBMITTER
    finally:
        journal.close()
    load_bundle(bundle_dir)  # re-anchored: bundle stays loadable


def test_trail_rebuilt_from_journal_matches_and_unknown_task_404s(mini):
    bundle_dir, work = mini
    out = submit_task(bundle_dir, _submission(work),
                      policy=corpus_policy(work))
    rebuilt = provenance_trail_for_bundle(
        bundle_dir, out["result"]["task_seq"])
    assert rebuilt["task_seq"] == out["provenance_trail"]["task_seq"]
    assert [s["edit_id"] for s in rebuilt["steps"]] == \
        [s["edit_id"] for s in out["provenance_trail"]["steps"]]
    assert rebuilt["submitter"] == SUBMITTER
    with pytest.raises(KeyError):
        provenance_trail_for_bundle(bundle_dir, 999999)


def test_operator_task_entries_are_unchanged_without_submitter(mini):
    # B-5's task.py change is additive: an operator-opened task
    # (the `si task` path) journals no submitter fields.
    from si.gaia.task import run_task
    bundle_dir, work = mini
    res = run_task(bundle_dir, [], "Operator task, no steps",
                   work_root=work, policy=corpus_policy(work),
                   operator="operator")
    assert res["stop_reason"] == "proposer-exhausted"
    manifest = load_bundle(bundle_dir)
    journal = open_journal(bundle_dir, manifest)
    try:
        rows = journal.backend.find("experiences",
                                    filter={"kind": "task"})
        assert len(rows) == 1
        assert "submitter" not in rows[0]["record"]
        assert "submitted_via" not in rows[0]["record"]
        # And its trail reports the absence as None, never invented.
        trail = build_provenance_trail(journal, res["task_seq"])
        assert trail["submitter"] is None
    finally:
        journal.close()


def test_demo_bundle_submission_stays_shadow(mini, tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "demo.si"), name="demo-b5",
                             journal_backend="embedded")
    work = str(tmp_path / "demo-work")
    os.makedirs(work)
    out = submit_task(
        bundle_dir,
        _submission(work, script=[
            {"kind": "tool", "tool": "fs_write",
             "args": {"path": "never.txt", "content": "x"},
             "reason": "Try to write via a demo bundle"}]),
        policy=corpus_policy(work))
    assert out["status"] == "task-returned"
    assert out["result"]["stop_reason"] == "escalated"
    assert out["result"]["steps"][0]["execution_status"] == \
        TOOL_SHADOW_EXECUTION
    assert not os.path.exists(os.path.join(work, "never.txt"))
    # The identity is journaled even when nothing executes.
    assert out["provenance_trail"]["submitter"] == SUBMITTER


# -- the HTTP surface -------------------------------------------------------------

class _Server:
    def __enter__(self):
        self.server = make_service_server(
            self.bundle_dir, host="127.0.0.1", port=0,
            policy=self.policy)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def call(self, method, path, payload=None):
        url = f"http://127.0.0.1:{self.port}{path}"
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers,
                                     method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode())


@pytest.fixture()
def server(mini):
    bundle_dir, work = mini
    ctx = _Server()
    ctx.bundle_dir = bundle_dir
    ctx.policy = corpus_policy(work)
    ctx.work = work
    with ctx as s:
        yield s


def test_http_health_and_description_carry_the_labels(server):
    code, body = server.call("GET", "/health")
    assert code == 200
    assert body["status"] == "ok"
    assert "NOT authentication" in body["submitter_identity"]
    assert "no authentication" in body["auth"]
    assert "H-2" in body["gateway_exposure"]
    code, body = server.call("GET", "/")
    assert code == 200
    assert "POST /tasks" in body["endpoints"]
    assert "demo bundles stay shadow" in body["execution_scope"]


def test_http_post_without_submitter_is_400_and_journals_nothing(server):
    before = _counts(server.bundle_dir)
    payload = _submission(server.work)
    del payload["submitter"]
    code, body = server.call("POST", "/tasks", payload)
    assert code == 400
    assert body["status"] == "refused"
    assert body["journaled"] is False
    assert _counts(server.bundle_dir) == before
    assert not os.path.exists(
        os.path.join(server.work, "service-made.txt"))


def test_http_submit_then_fetch_trail(server):
    code, body = server.call("POST", "/tasks",
                             _submission(server.work))
    assert code == 200
    assert body["status"] == "task-returned"
    task_seq = body["result"]["task_seq"]
    assert body["provenance_trail"]["submitter"] == SUBMITTER
    assert open(os.path.join(server.work, "service-made.txt")).read() \
        == "via the B-5 service\n"

    code, fetched = server.call("GET", f"/tasks/{task_seq}")
    assert code == 200
    trail = fetched["provenance_trail"]
    assert trail["task_seq"] == task_seq
    assert [s["edit_id"] for s in trail["steps"]] == \
        [s["edit_id"] for s in body["provenance_trail"]["steps"]]
    assert trail["task_stop"]["record"]["cause"] == \
        "mind-reported-complete"

    code, body = server.call("GET", "/tasks/999999")
    assert code == 404
    code, body = server.call("GET", "/nope")
    assert code == 404
    code, body = server.call("GET", "/tasks/abc")
    assert code == 400


# -- CLI + honesty guards ---------------------------------------------------------

def test_cli_gaia_serve_parser_is_wired():
    from si.cli import build_parser, cmd_gaia_serve
    args = build_parser().parse_args(["gaia-serve", "some.si",
                                      "--port", "0"])
    assert args.func is cmd_gaia_serve
    assert args.host == "127.0.0.1"  # localhost by default


def test_no_gateway_or_hearth_code_in_the_service_module():
    import si.gaia.service as service_mod
    src = open(service_mod.__file__).read()
    # Exposure through the Hearth gateway is H-2, not built: the
    # adapter imports no Hearth client and no gateway code.
    assert "gaia.hearth" not in src
    assert "HearthGatewayClient" not in src
    assert "H-2" in GATEWAY_EXPOSURE_LABEL
    assert "not built" in GATEWAY_EXPOSURE_LABEL
    # Identity honesty is a single sourced label, never "verified".
    assert "NOT authentication" in SUBMITTER_IDENTITY_LABEL
    assert "verified identity" not in src
