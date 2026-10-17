"""Tests for ``si squash`` (SI-3): compressing the oldest deltas.

Covers: squash correctness (head id unchanged, resolved ancestry routed
through the squashed delta, behavior preserved within tolerance),
provenance journaling (``squashed_from`` + append-only history),
rollback time travel to a superseded delta (exact historical behavior
restored), fail-closed tolerance violations (nothing written anywhere,
CLI errors), keep semantics, export determinism across squash
(byte-identical GGUF), and the gate REJECT path (rejection journaled
loop-style, nothing committed).
"""
import os

import numpy as np
import pytest

from si.bundle import _active_chain, canonical_json, init_bundle, load_bundle, sha256_hex
from si.export import export_bundle, materialize_head
from si.journal import open_journal
from si.loop import rollback, run_cycle
from si.minds import DemoMind
from si.squash import SquashError, squash_bundle


def _init(tmp_path, name):
    return init_bundle(str(tmp_path / (name + ".si")), name=name,
                       journal_backend="embedded")


def _five_cycles(bundle_dir):
    """Build the standard fixture: 5 admitted demo cycles, head d0005."""
    for _ in range(5):
        res = run_cycle(bundle_dir, DemoMind(bundle_path=bundle_dir))
        assert res["verdict"] == "ADMIT", f"setup cycle not admitted: {res}"
    manifest = load_bundle(bundle_dir)
    assert manifest["head"] == "d0005"
    return manifest


def _chain_ids(manifest):
    return [d["id"] for d in _active_chain(manifest)]


def _max_abs_diff(a, b):
    assert set(a) == set(b), "tensor name sets differ"
    return max(float(np.max(np.abs(a[k].astype(np.float64)
                                   - b[k].astype(np.float64))))
               for k in a)


def _journal_counts(bundle_dir, manifest):
    journal = open_journal(bundle_dir, manifest)
    try:
        return {c: journal.backend.count(c)
                for c in ("self_edits", "deltas", "provenance", "checkpoints")}
    finally:
        journal.close()


def _blob_files(bundle_dir):
    return sorted(os.listdir(os.path.join(bundle_dir, "blobs", "sha256")))


# -- correctness ---------------------------------------------------------------

def test_squash_correctness(tmp_path):
    bundle_dir = _init(tmp_path, "squash-correct")
    _five_cycles(bundle_dir)
    pre = materialize_head(bundle_dir)

    res = squash_bundle(bundle_dir, keep=3)
    assert res["squashed"] is True
    assert res["verdict"] == "ADMIT"
    assert res["delta_id"] == "d0006"

    manifest = load_bundle(bundle_dir)
    # The head id deliberately does NOT move; only its resolved ancestry
    # is rewritten through the squashed delta.
    assert manifest["head"] == "d0005"
    assert res["head"] == "d0005"
    assert _chain_ids(manifest) == ["d0006", "d0003", "d0004", "d0005"]

    post = materialize_head(bundle_dir)
    worst = _max_abs_diff(pre, post)
    assert worst < 1e-4, \
        f"squash changed the head by max abs {worst:.3e} >= tol 1e-4"


# -- provenance ------------------------------------------------------------------

def test_squash_provenance(tmp_path):
    bundle_dir = _init(tmp_path, "squash-prov")
    _five_cycles(bundle_dir)
    res = squash_bundle(bundle_dir, keep=3)
    new_id = res["delta_id"]

    # History is append-only: the squashed deltas stay in the manifest,
    # marked superseded_by; the kept tail is untouched.
    manifest = load_bundle(bundle_dir)
    assert len(manifest["deltas"]) == 6  # 5 original + 1 squashed
    by_id = {d["id"]: d for d in manifest["deltas"]}
    for did in ("d0001", "d0002"):
        assert did in by_id, f"squashed delta {did} must stay in history"
        assert by_id[did].get("superseded_by") == new_id
    for did in ("d0003", "d0004", "d0005"):
        assert "superseded_by" not in by_id[did]
    assert by_id[new_id]["parent"] == "base"

    journal = open_journal(bundle_dir, manifest)
    try:
        rows = journal.backend.find("provenance",
                                    filter={"relation": "squashed_from"})
        assert len(rows) == 1, "exactly one squashed_from provenance row"
        row = rows[0]
        assert row["child"] == new_id
        assert row["parents"] == ["d0001", "d0002"]
        # The new delta's own journal record carries the causal seqs.
        drows = journal.backend.find("deltas", filter={"id": new_id}, limit=1)
        assert drows and drows[0]["source_seqs"], \
            "squashed delta must record non-empty source_seqs"
    finally:
        journal.close()


# -- rollback time travel ----------------------------------------------------------

def test_squash_rollback_time_travel(tmp_path):
    bundle_dir = _init(tmp_path, "squash-tt")
    manifest = _five_cycles(bundle_dir)
    # Capture the exact pre-squash behavior at d0002 (a delta the squash
    # is about to subsume).
    sub = dict(manifest)
    sub["head"] = "d0002"
    pre_d0002 = materialize_head(bundle_dir, sub)

    squash_bundle(bundle_dir, keep=3)

    out = rollback(bundle_dir, to="d0002",
                   reason="test: time travel to a superseded delta",
                   mind=DemoMind(bundle_path=bundle_dir))
    assert out["head"] == "d0002"

    manifest = load_bundle(bundle_dir)
    assert manifest["head"] == "d0002"
    # The true historical ancestry is resolved, not the squash
    # replacement: d0002's own update must not be applied twice.
    assert _chain_ids(manifest) == ["d0001", "d0002"]

    post = materialize_head(bundle_dir)
    assert set(post) == set(pre_d0002)
    for name in pre_d0002:
        assert np.array_equal(post[name], pre_d0002[name]), \
            f"{name}: post-rollback weights differ from pre-squash d0002"


# -- tolerance violation: fail closed -----------------------------------------------

def test_squash_tolerance_violation_writes_nothing(tmp_path, capsys):
    bundle_dir = _init(tmp_path, "squash-tol")
    _five_cycles(bundle_dir)

    def _snapshot():
        manifest = load_bundle(bundle_dir)
        return (canonical_json(manifest),
                _journal_counts(bundle_dir, manifest),
                _blob_files(bundle_dir),
                manifest["head"])

    before = _snapshot()
    # rank=1 cannot represent the 2-delta combined update to 1e-15.
    with pytest.raises(SquashError):
        squash_bundle(bundle_dir, keep=3, rank=1, tol=1e-15)
    assert _snapshot() == before, \
        "tolerance violation must leave manifest, journal, blobs, head untouched"

    # The CLI surfaces the same failure as a nonzero exit with "error".
    from si.cli import main
    rc = main(["squash", bundle_dir, "--rank", "1", "--tol", "1e-15"])
    assert rc != 0, "si squash must exit nonzero on tolerance violation"
    assert "error" in capsys.readouterr().err.lower()
    assert _snapshot() == before


# -- keep semantics ------------------------------------------------------------------

def test_squash_keep_respected(tmp_path):
    bundle_dir = _init(tmp_path, "squash-keep")
    manifest = _five_cycles(bundle_dir)
    before = canonical_json(manifest)

    # keep == depth: nothing to squash, manifest byte-identical.
    res = squash_bundle(bundle_dir, keep=5)
    assert res == {"squashed": False}
    assert canonical_json(load_bundle(bundle_dir)) == before

    # keep=2 on 5-deep: oldest 3 -> one delta, newest 2 untouched.
    res = squash_bundle(bundle_dir, keep=2)
    assert res["squashed"] is True
    manifest = load_bundle(bundle_dir)
    assert _chain_ids(manifest) == ["d0006", "d0004", "d0005"]
    assert len(manifest["deltas"]) == 6, "exactly 1 new delta appended"
    assert manifest["head"] == "d0005"


# -- export determinism across squash (the money test) ----------------------------------

def test_export_deterministic_after_squash(tmp_path):
    bundle_dir = _init(tmp_path, "squash-expdet")
    _five_cycles(bundle_dir)

    f1 = str(tmp_path / "pre.gguf")
    r1 = export_bundle(bundle_dir, f1)
    with open(f1, "rb") as f:
        sha1 = sha256_hex(f.read())

    res = squash_bundle(bundle_dir, keep=3)
    assert res["squashed"] is True

    f2 = str(tmp_path / "post.gguf")
    r2 = export_bundle(bundle_dir, f2)
    with open(f2, "rb") as f:
        sha2 = sha256_hex(f.read())

    # Squash is behavior-preserving, and the export is a pure function of
    # the head: the artifact must be byte-identical.
    assert sha1 == sha2, \
        "pre/post-squash exports of the same head must be byte-identical"
    assert r1["sha256"] == r2["sha256"] == sha1


# -- gate REJECT path --------------------------------------------------------------------

def test_squash_gate_reject_path(tmp_path, monkeypatch):
    from si import squash as squash_mod
    from si.gate import CheckResult, GateReport

    bundle_dir = _init(tmp_path, "squash-reject")
    _five_cycles(bundle_dir)
    blobs_before = _blob_files(bundle_dir)

    class _RejectGate:
        """Force the gate to REJECT without touching the bundle policy."""
        def __init__(self, policy):
            self.policy = dict(policy)

        def evaluate(self, intent, candidate, ctx):
            return GateReport("REJECT", [CheckResult(
                "safety_bounds", False,
                "test-forced REJECT (not a real policy verdict)", 0.0)])

    monkeypatch.setattr(squash_mod, "Gate", _RejectGate)
    res = squash_bundle(bundle_dir, keep=3)

    assert res["squashed"] is False
    assert res["verdict"] == "REJECT"

    # Nothing committed: no new delta, head unmoved, no ancestry rewrite.
    manifest = load_bundle(bundle_dir)
    assert manifest["head"] == "d0005"
    assert [d["id"] for d in manifest["deltas"]] == \
        ["d0001", "d0002", "d0003", "d0004", "d0005"]
    assert all("superseded_by" not in d for d in manifest["deltas"])

    # ...but the rejection IS journaled, exactly like loop.run_cycle
    # journals rejections (self-edit + verdict + report, result rejected).
    journal = open_journal(bundle_dir, manifest)
    try:
        rows = journal.backend.find("self_edits",
                                    filter={"edit_id": res["edit_id"]})
        assert len(rows) == 1
        assert rows[0]["gate_verdict"] == "REJECT"
        assert rows[0]["result"] == "rejected"
        assert not journal.backend.find("deltas", filter={"id": "d0006"}), \
            "rejected squash must not journal a delta row"
    finally:
        journal.close()

    # The candidate blob stays on disk as evidence (mirrors the loop's
    # "rejected blobs stay on disk" guarantee).
    assert len(_blob_files(bundle_dir)) == len(blobs_before) + 1
