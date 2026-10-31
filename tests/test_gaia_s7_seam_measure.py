"""SI-62 / S-7: the seam measurement harness itself.

Pins (design §13.1 item 2): the harness *behaves* — fingerprint
honesty (gaia_commit is the mandatory "not-used" for Class S/L
runs), statistics discipline (§3.4: median/min/max, p95 only at
N >= 30, no fabricated number for an empty input), the
non-loopback refusal in code (§10.5), the deferred list carried
as explicit UNMEASURED entries (§12.2), correctness riders on a
small-N smoke run of every hop plan (§10.1: a run whose riders
fail is invalid), the forwarder-journals-nothing side-effect
rider, and residual quantities labelled derived (D-1). Tests
never assert that a timing meets a target — there are no
targets (§16).
"""
import pytest

from si.gaia.hearth import HttpTransport
from si.gaia.seam_measure import (
    Cfg, DEFERRED, LoopbackDouble, _quantity, assert_loopback_url,
    environment_fingerprint, run_measurement, summarize,
    work_double,
)


# -- statistics (§3.4) -----------------------------------------------------------

def test_summarize_reports_median_range_and_gates_p95():
    stats = summarize([3.0, 1.0, 2.0])
    assert stats["n"] == 3 and stats["min"] == 1.0
    assert stats["median"] == 2.0 and stats["max"] == 3.0
    assert stats["p95"] is None  # N < 30: no p95 claim (§3.4)
    stats = summarize([float(i) for i in range(30)])
    assert stats["p95"] == 28.0  # nearest-rank: ceil(.95*30)=29th
    empty = summarize([])
    assert empty["n"] == 0 and empty["median"] is None


def test_quantity_double_bound_flag_d11():
    fast_double = _quantity("Q-X", "invocation", "L", "fx", "ms",
                            "v", [10.0] * 30, double_ms=[0.001] * 30)
    assert fast_double["double_bound"] is False
    slow_double = _quantity("Q-X", "invocation", "L", "fx", "ms",
                            "v", [10.0] * 30, double_ms=[0.005] * 30)
    assert slow_double["double_bound"] is True  # 5 ms > 25% of 10 ms
    assert slow_double["double_service_ms"]["median"] == 5.0


# -- fingerprint (§12.1) -----------------------------------------------------------

def test_fingerprint_mandatory_not_used_and_fields():
    fp = environment_fingerprint(si_commit="abc123", seed=7, argv=[])
    assert fp["gaia_commit"] == "not-used"  # never a fork SHA here
    assert fp["si_commit"] == "abc123"
    assert fp["journal_backend"] == "embedded"
    for field in ("python_version", "python_executable", "venv_path",
                  "os", "cpu_model", "cpu_count_os",
                  "ram_bytes_total", "host_note", "harness"):
        assert field in fp, field
    assert fp["harness"]["seed"] == 7


# -- loopback discipline (§10.5) -----------------------------------------------------

def test_non_loopback_urls_refused_in_code():
    assert_loopback_url("http://127.0.0.1:9/work")
    assert_loopback_url("http://localhost:9/work")
    assert_loopback_url("replay://s7-measurement")
    for bad in ("http://example.com/work", "https://10.0.0.5/x",
                "ftp://127.0.0.1/x", "no-scheme"):
        with pytest.raises(ValueError):
            assert_loopback_url(bad)


def test_double_records_service_time_and_logs_before_response():
    with LoopbackDouble(work_double(("work_id",))) as dbl:
        transport = HttpTransport()
        for _ in range(3):
            status, _ = transport("POST", dbl.url("/work"),
                                  json_body={"tool": "hearth.restart"})
            assert status == 202
            # The log entry exists the moment the response is in
            # hand — no handler-thread race for the harness.
            assert len(dbl.log) >= 1
        assert len(dbl.log) == 3
        assert all(s is not None and s >= 0 for s in dbl.service_s)
        assert all(e["body"] == {"tool": "hearth.restart"}
                   for e in dbl.log)


# -- the smoke run: every hop plan, small N ---------------------------------------

@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("s7-smoke") / "root")
    cfg = Cfg(reps=3, pass_reps=1, catchup_reps=1, warmup=1,
              ev_big=60, ev_mid=30, pending_big=5, poll_big=5,
              reconcile_big=200)
    return run_measurement(root, cfg=cfg, si_commit="smoke-sha",
                           argv=["--quick"])


def test_smoke_run_is_valid_and_riders_all_pass(smoke):
    assert smoke["valid"] is True, [
        r for r in smoke["riders"] if not r["passed"]]
    assert smoke["invalidations"] == []


def test_smoke_record_schema_and_coverage(smoke):
    ids = {q["id"] for q in smoke["quantities"]}
    for prefix in ("Q-A", "Q-B", "Q-C", "Q-D", "Q-E"):
        assert any(i.startswith(prefix) for i in ids), prefix
    for q in smoke["quantities"]:
        assert q["evidence_class"] in ("S", "L")
        assert q["stats"]["n"] == len(q["raw"]) >= 1
        assert q["unit"] and q["fixture"] and q["variant"]
        if q["evidence_class"] == "L" and "double_service_ms" in q:
            assert "double_bound" in q  # D-11 computed, not buried
    assert smoke["fingerprint"]["gaia_commit"] == "not-used"
    assert smoke["fingerprint"]["si_commit"] == "smoke-sha"


def test_smoke_deferred_quantities_stay_unmeasured(smoke):
    assert {d["id"] for d in smoke["deferred"]} == \
        {d["id"] for d in DEFERRED}
    assert all(d["status"] == "UNMEASURED" for d in smoke["deferred"])
    # The UNBUILT fork ingest is named, not estimated (§8.5).
    ingest = next(d for d in smoke["deferred"] if d["id"] == "F-E-INGEST")
    assert "UNBUILT" in ingest["reason"]


def test_smoke_residuals_are_labelled_derived(smoke):
    for qid in ("Q-A4", "Q-B4"):
        rec = next(q for q in smoke["quantities"] if q["id"] == qid)
        assert rec.get("derived") is True
        assert "residual" in rec["variant"]


def test_smoke_forwarder_journals_nothing_rider(smoke):
    rider = next(r for r in smoke["riders"]
                 if r["name"] == "E-forwarder-journals-nothing")
    assert rider["passed"] is True
    # And the seq-fidelity / cursor riders rode the E passes.
    names = {r["name"] for r in smoke["riders"]}
    assert any(n.startswith("E4-pass-") for n in names)
    assert any(n.startswith("E5-") for n in names)
    assert "E6-duplicate-raises-first-event" in names
    assert "A8-dedup-zero-growth" in names
    assert "B7-dedup-zero-transport" in names
