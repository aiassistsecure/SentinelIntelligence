"""SI-54 (SI-50 §16 Q-4): Gaia Q-4 — the measurement slice.

The harness's own correctness, at miniature test scale
(size 4, 1 repetition, bounds 1/4 — the canonical numbers
live in docs/SI-54-q4-measurement.md and were produced by
the full run, not by these tests):

- every scripted workload executes completely under the
  serial cycle and under the queue at each swept bound;
- queue residence samples exist and are non-negative
  (journal timestamps, §12.2's definition);
- journal growth is a positive number per task, and the
  queue's growth exceeds the serial cycle's for the same
  workload (§18.4: claims + the batch entry are real
  bookkeeping, priced here as entries per intent);
- the §8.2 recommendation helper is parsimonious pure
  arithmetic: the smallest bound within tolerance of the
  best median, and None on no data;
- the leash measurements run: an installed check does not
  change a short execution's outcome, and a stop journaled
  against a live drain kills the in-flight sleep with a
  ``timed-out`` outcome;
- the pipeline measurement runs both arms over the same
  admitted work and returns measured walls for each.
"""
import pytest

from si.gaia.measure import (
    WORKLOADS,
    _dist,
    _parse_ts,
    recommend_read_bound,
    run_measurement,
    run_task_once,
)


def test_dist_and_ts_helpers():
    assert _dist([])["median"] is None
    d = _dist([3.0, 1.0, 2.0])
    assert d["n"] == 3 and d["min"] == 1.0 and d["median"] == 2.0
    assert d["max"] == 3.0
    ts = _parse_ts("2026-10-30T20:40:38.409261Z")
    assert ts is not None and ts.tzinfo is not None
    assert _parse_ts("not-a-time") is None
    assert _parse_ts(None) is None


def test_recommend_read_bound_is_parsimonious():
    # Bound 2 within 10% of the best (bound 8): recommend 2.
    assert recommend_read_bound({1: 1.00, 2: 0.55, 4: 0.52, 8: 0.50}) == 2
    # Serial-fastest data recommends 1.
    assert recommend_read_bound({1: 0.40, 2: 0.50, 4: 0.60, 8: 0.70}) == 1
    # A real gain beyond tolerance earns the larger bound.
    assert recommend_read_bound({1: 1.00, 2: 0.80, 4: 0.79, 8: 0.78}) == 2
    assert recommend_read_bound({}) is None
    assert recommend_read_bound({4: None}) is None


@pytest.mark.parametrize("workload", WORKLOADS)
def test_workload_executes_serially_and_queued(tmp_path, workload):
    root = str(tmp_path)
    serial = run_task_once(root, f"t-s-{workload}", workload,
                           "serial", size=4)
    assert serial["executed"] == 4
    assert serial["total_wall_s"] > 0
    assert serial["growth"]["journal_entries"] > 0
    queued = run_task_once(root, f"t-q-{workload}", workload,
                           "queue", bound=4, size=4)
    assert queued["executed"] == 4
    assert queued["sync_reason"] == "queue-empty"
    # §18.4 priced: the queue journals strictly more per task
    # (one batch entry + one claim per intent).
    assert queued["growth"]["journal_entries"] \
        > serial["growth"]["journal_entries"]
    # Residence (§12.2) is sampled from journal timestamps.
    assert queued["residence_s"]["n"] == 4
    assert queued["residence_s"]["min"] >= 0


def test_full_measurement_shape(tmp_path):
    result = run_measurement(
        str(tmp_path / "m"), size=4, reps=1, bounds=(1, 4),
        leash_reps=1, stop_reps=1, pipeline_reps=1)
    assert result["conditions"]["journal_backend"] \
        == "embedded (sqlite)"
    assert result["conditions"]["size_per_task"] == 4
    rows = {(r["workload"], r["mode"], r["bound"])
            for r in result["aggregates"]}
    for workload in WORKLOADS:
        assert (workload, "serial", None) in rows
        assert (workload, "queue", 1) in rows
        assert (workload, "queue", 4) in rows
    for row in result["aggregates"]:
        assert row["executed_min"] == 4
    # Leash: outcomes unchanged by an installed check; the
    # live stop killed the sleeping execution.
    leash = result["leash"]
    assert leash["overhead_wall_s"]["none"]["n"] == 1
    assert leash["overhead_wall_s"]["fold"]["n"] == 1
    assert leash["stop_latency_s"]["n"] == 1
    assert leash["stop_execution_statuses"] == [["timed-out"]]
    # Pipeline: both arms measured the same admitted work.
    pipe = result["pipeline"]
    assert pipe["sequential_wall_s"]["n"] == 1
    assert pipe["pipelined_wall_s"]["n"] == 1
    # The recommendation is one of the swept bounds.
    assert result["recommendation"]["read_bound"] in (1, 4)
