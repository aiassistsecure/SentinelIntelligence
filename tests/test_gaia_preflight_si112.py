"""SI-112 (SI-111 §3.3; D-111-2): the pre-flight template
lock — PFL-1.

OFFLINE / FIXTURE ONLY. The GGUF probe is exercised against
synthetic GGUF byte strings built in this file and, where
the real candidate files exist on this box, against the
recorded SI-93 ground truth (Imagine v13's GGUF carries no
``tokenizer.chat_template``; v12's embedded template hashes
to 993c0af7…, byte-identical to the external file the node
serves v13 with). Capture wiring is exercised through the
SI-79 fake transport and the SI-109 fixture driver pattern;
no live model is contacted and no live capture is made.

What is proven here:

- the probe is a real GGUF reader (typed values, arrays,
  strings skipped correctly; garbage refused loudly);
- registration prices embedded / external / unresolved
  exactly as §3.3(a) states;
- the pre-flight refuses fail-closed on mismatch,
  TEMPLATE-UNRESOLVED, and an unresolvable serving
  configuration (the SI-93 ChatML-fallback shape), and
  passes on a hash match;
- both capture entry points spend no attempt on a refusal,
  carry the check record as evidence, and halt with no
  attempt spent when the served template changes
  mid-capture (§3.3(c));
- the operator CLI refuses a fresh capture with no
  registration (D-111-2).
"""
import hashlib
import json
import os
import struct

import pytest

from si.gaia import conformance_capture as capture_mod
from si.gaia.conformance import V2_WORKLOADS
from si.gaia.conformance_capture import (
    ResidencyGuardHalt, capture_corpus)
from si.gaia.preflight import (
    GGUFProbeError, PreflightRefused, check_preflight,
    probe_gguf_template, register_candidate, require_preflight)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# -- Synthetic GGUF fixtures --------------------------------

def _gs(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv_str(key, value):
    return _gs(key) + struct.pack("<I", 8) + _gs(value)


def _kv_u32(key, value):
    return _gs(key) + struct.pack("<I", 4) + struct.pack("<I", value)


def _kv_f32(key, value):
    return _gs(key) + struct.pack("<I", 6) + struct.pack("<f", value)


def _kv_bool(key, value):
    return _gs(key) + struct.pack("<I", 7) + bytes([1 if value else 0])


def _kv_u64(key, value):
    return _gs(key) + struct.pack("<I", 10) + struct.pack("<Q", value)


def _kv_str_array(key, values):
    out = _gs(key) + struct.pack("<I", 9) + struct.pack("<I", 8)
    out += struct.pack("<Q", len(values))
    for v in values:
        out += _gs(v)
    return out


TEMPLATE_TEXT = "{% for message in messages %}TEMPLATE-A{% endfor %}"


def _gguf(kvs: list, version: int = 3) -> bytes:
    header = b"GGUF" + struct.pack("<I", version)
    header += struct.pack("<Q", 0) + struct.pack("<Q", len(kvs))
    return header + b"".join(kvs)


def _gguf_with_template(path, template=TEMPLATE_TEXT):
    kvs = [
        _kv_str("general.name", "synthetic-candidate"),
        _kv_u32("synthetic.context_length", 4096),
        _kv_str_array("tokenizer.ggml.tokens", ["a", "b", "c"]),
        _kv_str("tokenizer.chat_template", template),
        _kv_f32("synthetic.rope_freq", 10000.0),
        _kv_bool("synthetic.flag", True),
        _kv_u64("synthetic.big", 2**40),
    ]
    path.write_bytes(_gguf(kvs))
    return str(path)


def _gguf_without_template(path):
    kvs = [
        _kv_str("general.name", "synthetic-candidate"),
        _kv_u32("synthetic.context_length", 4096),
        _kv_str_array("tokenizer.ggml.tokens", ["a", "b"]),
    ]
    path.write_bytes(_gguf(kvs))
    return str(path)


# -- The probe ----------------------------------------------

def test_probe_reads_embedded_template_and_skips_typed_values(tmp_path):
    gguf = _gguf_with_template(tmp_path / "c.gguf")
    probe = probe_gguf_template(gguf)
    assert probe["present"] is True
    assert probe["template"] == TEMPLATE_TEXT
    assert probe["template_sha256"] == _sha(TEMPLATE_TEXT)
    assert probe["template_bytes"] == len(TEMPLATE_TEXT.encode())


def test_probe_reports_absent_template(tmp_path):
    gguf = _gguf_without_template(tmp_path / "c.gguf")
    probe = probe_gguf_template(gguf)
    assert probe["present"] is False
    assert probe["template_sha256"] is None


def test_probe_rejects_non_gguf_and_truncated(tmp_path):
    junk = tmp_path / "junk.gguf"
    junk.write_bytes(b"NOPE" + b"\x00" * 64)
    with pytest.raises(GGUFProbeError):
        probe_gguf_template(str(junk))
    trunc = tmp_path / "trunc.gguf"
    trunc.write_bytes(_gguf([_kv_str("a", "b")])[:10])
    with pytest.raises(GGUFProbeError):
        probe_gguf_template(str(trunc))
    with pytest.raises(GGUFProbeError):
        probe_gguf_template(str(tmp_path / "missing.gguf"))


# -- Registration (§3.3(a)) ----------------------------------

def test_registration_embedded_template(tmp_path):
    gguf = _gguf_with_template(tmp_path / "c.gguf")
    reg = register_candidate("cand", gguf_path=gguf)
    assert reg["status"] == "TEMPLATE-RESOLVED"
    assert reg["registered_template_source"] == "embedded"
    assert reg["registered_template_sha256"] == _sha(TEMPLATE_TEXT)
    assert reg["probe"]["tokenizer_chat_template"] == "present"


def test_registration_external_template_when_gguf_absent(tmp_path):
    gguf = _gguf_without_template(tmp_path / "c.gguf")
    ext = tmp_path / "template.jinja"
    ext.write_text(TEMPLATE_TEXT, encoding="utf-8")
    reg = register_candidate(
        "cand", gguf_path=gguf, external_template_path=str(ext))
    assert reg["status"] == "TEMPLATE-RESOLVED"
    assert reg["registered_template_source"] == "external"
    assert reg["registered_template_sha256"] == _sha(TEMPLATE_TEXT)
    assert reg["probe"]["tokenizer_chat_template"] == "absent"


def test_registration_unresolved_is_ineligible(tmp_path):
    gguf = _gguf_without_template(tmp_path / "c.gguf")
    reg = register_candidate("cand", gguf_path=gguf)
    assert reg["status"] == "TEMPLATE-UNRESOLVED"
    assert reg["registered_template_sha256"] is None
    assert "INELIGIBLE" in reg["eligibility"]


# -- The pre-flight check (§3.3(b)) ---------------------------

def test_preflight_passes_on_hash_match(tmp_path):
    gguf = _gguf_with_template(tmp_path / "c.gguf")
    reg = register_candidate("cand", gguf_path=gguf)
    record = check_preflight(reg, served_gguf_path=gguf)
    assert record["decision"] == "PASS"
    assert record["served_template_source"] == "served-gguf-embedded"
    assert record["served_template_sha256"] == \
        record["registered_template_sha256"]


def test_preflight_spawn_file_wins_over_embedded(tmp_path):
    gguf = _gguf_with_template(tmp_path / "c.gguf")
    reg = register_candidate("cand", gguf_path=gguf)
    other = tmp_path / "other.jinja"
    other.write_text("a different template", encoding="utf-8")
    record = check_preflight(reg, spawn_template_file=str(other),
                             served_gguf_path=gguf)
    assert record["served_template_source"] == "spawn-template-file"
    assert record["decision"] == "REFUSED"
    same = tmp_path / "same.jinja"
    same.write_text(TEMPLATE_TEXT, encoding="utf-8")
    record = check_preflight(reg, spawn_template_file=str(same),
                             served_gguf_path=gguf)
    assert record["decision"] == "PASS"


def test_preflight_refuses_unresolved_and_unresolvable(tmp_path):
    gguf = _gguf_without_template(tmp_path / "c.gguf")
    unresolved = register_candidate("cand", gguf_path=gguf)
    record = check_preflight(
        unresolved, served_template_sha256=_sha(TEMPLATE_TEXT))
    assert record["decision"] == "REFUSED"
    assert "TEMPLATE-UNRESOLVED" in record["reason"]
    # The SI-93 shape: a resolved registration, but the served
    # GGUF carries no template and no spawn file is in play —
    # the server would fall back to a built-in template.
    ext = tmp_path / "t.jinja"
    ext.write_text(TEMPLATE_TEXT, encoding="utf-8")
    resolved = register_candidate(
        "cand", gguf_path=gguf, external_template_path=str(ext))
    record = check_preflight(resolved, served_gguf_path=gguf)
    assert record["decision"] == "REFUSED"
    assert "fall back" in record["served_detail"]
    record = check_preflight(resolved)
    assert record["decision"] == "REFUSED"
    assert record["served_template_source"] == "unresolvable"


def test_require_preflight_raises_with_record(tmp_path):
    gguf = _gguf_with_template(tmp_path / "c.gguf")
    reg = register_candidate("cand", gguf_path=gguf)
    with pytest.raises(PreflightRefused) as caught:
        require_preflight(reg, served_template_sha256=_sha("nope"))
    assert caught.value.record["decision"] == "REFUSED"
    assert caught.value.record["registered_template_sha256"] == \
        _sha(TEMPLATE_TEXT)


# -- Recorded ground truth (SI-93), when the files exist -----

V13_GGUF = "/home/hatch/workspace/models/imagine-v13-Q4_K_M.gguf"
V12_GGUF = "/home/hatch/workspace/models/imagine-v12-Q4_K_M.gguf"
V13_TEMPLATE_FILE = "/home/hatch/workspace/gaia-node/chat-template-v13.jinja"
V12_EMBEDDED_SHA256 = \
    "993c0af704bad76aeba4263a5318f7873a06420a8eb7d40b72a49c39599c0a66"


@pytest.mark.skipif(not os.path.isfile(V13_GGUF),
                    reason="v13 GGUF not on this box")
def test_si93_ground_truth_v13_has_no_embedded_template():
    probe = probe_gguf_template(V13_GGUF)
    assert probe["present"] is False


@pytest.mark.skipif(not os.path.isfile(V12_GGUF),
                    reason="v12 GGUF not on this box")
def test_si93_ground_truth_v12_embedded_template_hash():
    probe = probe_gguf_template(V12_GGUF)
    assert probe["present"] is True
    assert probe["template_sha256"] == V12_EMBEDDED_SHA256


@pytest.mark.skipif(
    not (os.path.isfile(V13_GGUF) and os.path.isfile(V13_TEMPLATE_FILE)),
    reason="v13 GGUF / node template file not on this box")
def test_v13_registration_via_external_template_file():
    reg = register_candidate(
        "imagine-v13", gguf_path=V13_GGUF,
        external_template_path=V13_TEMPLATE_FILE)
    assert reg["status"] == "TEMPLATE-RESOLVED"
    assert reg["registered_template_source"] == "external"
    assert reg["registered_template_sha256"] == V12_EMBEDDED_SHA256
    record = check_preflight(
        reg, spawn_template_file=V13_TEMPLATE_FILE)
    assert record["decision"] == "PASS"


# -- Single-cycle capture wiring (SI-79 entry point) ---------

def _doc(intents):
    return json.dumps({"proposal": {
        "version": 1, "intents": intents, "task_signal": None}})


def _intent(reason="list the root"):
    return {"tool": "fs_list", "args": {"path": "."},
            "reason": reason, "depends_on": [],
            "ordering_key": None}


class _FakeTransport:
    """The SI-79 fake transport pattern: residency polls,
    /v1/models discovery, rotating distinct chat bodies."""

    label = "fake-transport (tests)"
    is_replay = False

    def __init__(self):
        self.chat_calls = 0

    def __call__(self, method, url, *, json_body=None,
                 timeout_s=5.0):
        if url.endswith("/v1/models"):
            return 200, json.dumps(
                {"data": [{"id": "fake-served-id"}]}).encode()
        if url.endswith("/gaia/v1/residency"):
            return 200, json.dumps({"models": [{
                "model": "imagine-v12",
                "state": "resident"}]}).encode()
        if url.endswith("/v1/chat/completions"):
            self.chat_calls += 1
            body = _doc([_intent(
                reason=f"call {self.chat_calls} lists the root")])
            return 200, json.dumps({"choices": [{"message": {
                "content": body}}]}).encode()
        raise AssertionError(f"unexpected URL {url}")


def _external_registration(tmp_path, text=TEMPLATE_TEXT):
    ext = tmp_path / "served.jinja"
    ext.write_text(text, encoding="utf-8")
    reg = register_candidate(
        "fake-served-id", external_template_path=str(ext))
    return reg, str(ext)


def test_capture_preflight_pass_records_and_captures(tmp_path):
    reg, ext = _external_registration(tmp_path)
    fake = _FakeTransport()
    corpus, log = capture_corpus(
        str(tmp_path / "cap"), inference_base="http://fake-inference",
        residency_base="http://fake-residency",
        items_per_workload=1, max_attempts_per_workload=2,
        transport=fake, log=lambda *a: None,
        candidate_registration=reg, spawn_template_file=ext)
    assert corpus["fixed_n"] == len(V2_WORKLOADS)
    assert log["preflight"]["decision"] == "PASS"
    assert log["preflight"]["registered_template_sha256"] == \
        _sha(TEMPLATE_TEXT)
    assert log["preflight"]["rechecks"]
    assert all(r["decision"] == "match"
               for r in log["preflight"]["rechecks"])


def test_capture_preflight_refusal_spends_no_attempt(tmp_path):
    reg, _ext = _external_registration(tmp_path)
    fake = _FakeTransport()
    with pytest.raises(PreflightRefused) as caught:
        capture_corpus(
            str(tmp_path / "cap"),
            inference_base="http://fake-inference",
            residency_base="http://fake-residency",
            items_per_workload=1, max_attempts_per_workload=2,
            transport=fake, log=lambda *a: None,
            candidate_registration=reg,
            served_template_sha256=_sha("some other template"))
    assert fake.chat_calls == 0
    assert caught.value.record["decision"] == "REFUSED"


def test_capture_preflight_unresolved_refuses(tmp_path):
    reg = register_candidate("fake-served-id")
    assert reg["status"] == "TEMPLATE-UNRESOLVED"
    fake = _FakeTransport()
    with pytest.raises(PreflightRefused):
        capture_corpus(
            str(tmp_path / "cap"),
            inference_base="http://fake-inference",
            residency_base="http://fake-residency",
            items_per_workload=1, max_attempts_per_workload=2,
            transport=fake, log=lambda *a: None,
            candidate_registration=reg,
            served_template_sha256=_sha(TEMPLATE_TEXT))
    assert fake.chat_calls == 0


class _TemplateSwappingTransport(_FakeTransport):
    """Rewrites the spawn template file after the first chat
    call — a mid-capture serving change (§3.3(c))."""

    def __init__(self, template_path):
        super().__init__()
        self.template_path = template_path

    def __call__(self, method, url, *, json_body=None,
                 timeout_s=5.0):
        result = super().__call__(
            method, url, json_body=json_body, timeout_s=timeout_s)
        if url.endswith("/v1/chat/completions") \
                and self.chat_calls == 1:
            with open(self.template_path, "w",
                      encoding="utf-8") as fh:
                fh.write("swapped template — not the "
                         "registered one")
        return result


def test_capture_template_change_halts_at_guard(tmp_path):
    reg, ext = _external_registration(tmp_path)
    fake = _TemplateSwappingTransport(ext)
    snapshots = []
    with pytest.raises(ResidencyGuardHalt):
        capture_corpus(
            str(tmp_path / "cap"),
            inference_base="http://fake-inference",
            residency_base="http://fake-residency",
            items_per_workload=2, max_attempts_per_workload=4,
            transport=fake, log=lambda *a: None,
            on_progress=lambda c, lg: snapshots.append(lg),
            candidate_registration=reg, spawn_template_file=ext)
    # Exactly one attempt was spent; the halt came at the
    # next attempt's guard poll, before its POST.
    assert fake.chat_calls == 1
    rechecks = snapshots[-1]["preflight"]["rechecks"]
    assert rechecks[-1]["decision"] == "MISMATCH"


def test_cli_refuses_fresh_capture_without_registration(tmp_path):
    rc = capture_mod.main([
        "--out", str(tmp_path / "out"),
        "--inference-base", "http://fake-inference",
        "--residency-base", "http://fake-residency"])
    assert rc == 2


# -- Multicycle driver wiring (SI-109 entry point) -----------

from si.gaia import live as live_mod  # noqa: E402
from si.gaia.multicycle import MultiCycleDriver  # noqa: E402
from si.gate import Gate  # noqa: E402
from si.journal import open_journal  # noqa: E402


class _Guard:
    def __init__(self, states):
        self.states = list(states)

    def __call__(self, journal):
        if len(self.states) > 1:
            return self.states.pop(0)
        return self.states[0]


def _citing_source(perception, _n):
    sections = perception.get("sections") or {}
    prior = sections.get("prior_cycles") or []
    citable = (sections.get("citable_reference_set") or {}).get(
        "seqs") or []
    deps = [citable[0]] if citable else []
    return _doc([{"tool": "fs_list", "args": {"path": "."},
                  "reason": f"cycle with {len(prior)} prior",
                  "depends_on": deps, "ordering_key": None}])


class _Source:
    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, perception):
        self.calls += 1
        return self.fn(perception, self.calls)


@pytest.fixture()
def mc_fixture(tmp_path):
    paths = live_mod.build_fixture_bundle(
        str(tmp_path / "capture-bundle"), str(tmp_path / "work"))
    policy = live_mod.fixture_policy(paths["work_root"])
    journal = open_journal(paths["bundle_dir"])
    yield paths, policy, journal
    journal.close()


def _driver(journal, policy, paths, source, **kw):
    from si.gaia.policy import resolve_tool_policy
    tool_pol = resolve_tool_policy(policy)
    budgets = {
        "max_tool_cycles_per_task": int(
            tool_pol["max_tool_cycles_per_task"]),
        "tool_max_task_result_bytes": int(
            tool_pol["tool_max_task_result_bytes"]),
    }
    return MultiCycleDriver(
        journal, Gate(policy), bundle_dir=paths["bundle_dir"],
        work_root=paths["work_root"], budgets=budgets,
        cycle_cap=budgets["max_tool_cycles_per_task"],
        proposal_source=source,
        residency_guard=_Guard(["Resident"]), **kw)


_MC_TASK = {"workload": "benign-read", "run": 0,
            "instruction": "list the work root"}
_GOOD = _sha(TEMPLATE_TEXT)


def _mc_registration(tmp_path):
    ext = tmp_path / "served.jinja"
    ext.write_text(TEMPLATE_TEXT, encoding="utf-8")
    return register_candidate(
        "cand", external_template_path=str(ext))


def test_driver_preflight_pass_and_rechecks(mc_fixture, tmp_path):
    paths, policy, journal = mc_fixture
    driver = _driver(
        journal, policy, paths, _Source(_citing_source), K=1,
        candidate_registration=_mc_registration(tmp_path),
        served_template_probe=lambda: _GOOD)
    state = driver.run([_MC_TASK])
    assert state["preflight"]["decision"] == "PASS"
    assert state["preflight"]["rechecks"]
    assert all(r["decision"] == "match"
               for r in state["preflight"]["rechecks"])
    assert state["documents"]


def test_driver_preflight_refusal_spends_nothing(mc_fixture, tmp_path):
    paths, policy, journal = mc_fixture
    driver = _driver(
        journal, policy, paths, _Source(_citing_source), K=1,
        candidate_registration=_mc_registration(tmp_path),
        served_template_probe=lambda: _sha("wrong template"))
    with pytest.raises(PreflightRefused) as caught:
        driver.run([_MC_TASK])
    assert caught.value.record["decision"] == "REFUSED"
    assert driver.state["attempts"] == {}
    assert driver.state["documents"] == []
    assert driver.state["runs"] == {}


def test_driver_template_flip_halts_at_guard(mc_fixture, tmp_path):
    paths, policy, journal = mc_fixture
    calls = {"n": 0}

    def probe():
        calls["n"] += 1
        # Pre-flight + first guard poll see the registered
        # template; the served template then changes, as a
        # mid-capture restart without the wrapper would do.
        return _GOOD if calls["n"] <= 2 else _sha("swapped")

    driver = _driver(
        journal, policy, paths, _Source(_citing_source), K=1,
        candidate_registration=_mc_registration(tmp_path),
        served_template_probe=probe)
    with pytest.raises(ResidencyGuardHalt):
        driver.run([_MC_TASK])
    assert driver.state["halted"]["reason"] == \
        "served template changed mid-capture"
