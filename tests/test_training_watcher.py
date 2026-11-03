"""SI-84 (SI-83 section 9 slice 1): the observe-only training watcher.

A synthetic trainer saves miniature checkpoints as plain files.
Verifications required by SI-83 section 9.1:
(a) watcher manifests match independently computed hashes for
    every complete checkpoint;
(b) a deliberately truncated save is checkpoint_complete=false;
(c) kill-and-restart mid-sequence journals a watcher_gap row
    (not silence) and recovers the missing manifests on re-scan.
Plus: the watcher writes nothing into the checkpoint tree, and
checkpoints discovered in the same poll are journaled in step
order (SI-83 section 3.3).
"""
import hashlib
import json
import os

import pytest

from si.training_watcher import (
    TrainingWatcher,
    WatcherAlreadyRunning,
    load_journal_records,
    rollback_targets,
)


def sha256_tag(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


class SyntheticTrainer:
    """Miniature trainer: writes step<N>/ dirs, marker last."""

    def __init__(self, root):
        self.root = str(root)
        os.makedirs(self.root, exist_ok=True)

    def save(self, step, weights: bytes, optimizer: bytes = b"opt",
             epoch=None, complete=True):
        d = os.path.join(self.root, f"step{step}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "weights.bin"), "wb") as f:
            f.write(weights)
        if optimizer is not None:
            with open(os.path.join(d, "optimizer.bin"), "wb") as f:
                f.write(optimizer)
        if epoch is not None:
            with open(os.path.join(d, "meta.json"), "w") as f:
                json.dump({"epoch": epoch}, f)
        if complete:
            with open(os.path.join(d, "SAVE_COMPLETE"), "w") as f:
                f.write("done\n")
        return d

    def save_truncated(self, step, partial: bytes):
        d = os.path.join(self.root, f"step{step}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "weights.bin"), "wb") as f:
            f.write(partial)  # partial weights, no marker, ever
        return d


def tree_state(root):
    state = {}
    for base, _dirs, files in os.walk(root):
        for name in files:
            p = os.path.join(base, name)
            with open(p, "rb") as f:
                state[os.path.relpath(p, root)] = hashlib.sha256(
                    f.read()).hexdigest()
    return state


def make_watcher(tmp_path, trainer_root):
    return TrainingWatcher(
        str(trainer_root), str(tmp_path / "journal" / "train.jsonl"),
        run_id="run-test", poll_interval=0.0)


def observe(watcher, times=3):
    out = []
    for _ in range(times):
        out.extend(watcher.observe_once())
    return out


def manifests(records):
    return [r for r in records if r["kind"] == "checkpoint_manifest"]


def test_a_manifests_match_independent_hashes(tmp_path):
    trainer = SyntheticTrainer(tmp_path / "ckpts")
    expected = {}
    for step, weights, opt in [(1, b"weights-one", b"opt-one"),
                               (2, b"weights-two!!", b"opt-two"),
                               (3, b"w3" * 100, b"o3")]:
        trainer.save(step, weights, opt, epoch=step / 2)
        expected[step] = (weights, opt)
    w = make_watcher(tmp_path, trainer.root)
    observe(w)
    recs = manifests(load_journal_records(w.journal_path))
    assert [r["step"] for r in recs] == [1, 2, 3]
    for r in recs:
        weights, opt = expected[r["step"]]
        assert r["checkpoint_complete"] is True
        assert r["weights_blob"] == sha256_tag(weights)
        assert r["weights_bytes"] == len(weights)
        assert r["optimizer_blob"] == sha256_tag(opt)
        assert r["optimizer_bytes"] == len(opt)
        assert r["epoch"] == r["step"] / 2
        # Honest unknowns: the trainer recorded neither.
        assert r["trainer_commit"] == "unknown"
        assert r["data_order_seed"] is None
    # Chain: seq is monotonic and prev_hash links hold.
    all_recs = load_journal_records(w.journal_path)
    assert [r["seq"] for r in all_recs] == list(
        range(1, len(all_recs) + 1))
    for prev, cur in zip(all_recs, all_recs[1:]):
        assert cur["prev_hash"] == prev["entry_hash"]
    assert [r["step"] for r in rollback_targets(all_recs)] == [1, 2, 3]


def test_b_truncated_save_is_incomplete(tmp_path):
    trainer = SyntheticTrainer(tmp_path / "ckpts")
    trainer.save(1, b"good-weights", b"opt")
    trainer.save_truncated(2, b"partial-wei")
    w = make_watcher(tmp_path, trainer.root)
    observe(w)
    recs = manifests(load_journal_records(w.journal_path))
    by_step = {r["step"]: r for r in recs}
    assert by_step[1]["checkpoint_complete"] is True
    assert by_step[2]["checkpoint_complete"] is False
    # A false row is never presented as a rollback target.
    assert [r["step"] for r in rollback_targets(
        load_journal_records(w.journal_path))] == [1]


def test_c_kill_restart_journals_gap_and_recovers(tmp_path):
    trainer = SyntheticTrainer(tmp_path / "ckpts")
    trainer.save(1, b"w1", b"o1")
    w1 = make_watcher(tmp_path, trainer.root)
    observe(w1)
    journal = w1.journal_path
    del w1  # killed: no shutdown record of any kind
    trainer.save(2, b"w2-longer", b"o2")
    trainer.save(3, b"w3-longer", b"o3")
    w2 = make_watcher(tmp_path, trainer.root)
    # Same journal path as w1 (make_watcher uses a fixed name).
    assert w2.journal_path == journal
    observe(w2)
    recs = load_journal_records(journal)
    gaps = [r for r in recs if r["kind"] == "watcher_gap"]
    assert len(gaps) == 1
    assert gaps[0]["step_from"] == 2 and gaps[0]["step_to"] == 3
    assert gaps[0]["covered_steps"] == [2, 3]
    # The gap precedes the recovered manifests in seq order.
    gap_seq = gaps[0]["seq"]
    recovered = [r for r in manifests(recs) if r["step"] in (2, 3)]
    assert recovered and all(r["seq"] > gap_seq for r in recovered)
    assert {r["step"] for r in manifests(recs)} == {1, 2, 3}
    for r in recovered:
        data = {2: b"w2-longer", 3: b"w3-longer"}[r["step"]]
        assert r["weights_blob"] == sha256_tag(data)
        assert r["checkpoint_complete"] is True


def test_watcher_writes_nothing_into_checkpoint_tree(tmp_path):
    trainer = SyntheticTrainer(tmp_path / "ckpts")
    trainer.save(1, b"w1", b"o1")
    trainer.save(2, b"w2", b"o2")
    before = tree_state(trainer.root)
    w = make_watcher(tmp_path, trainer.root)
    with w:  # lock lifecycle included
        observe(w)
    after = tree_state(trainer.root)
    assert before == after
    assert not os.path.exists(w.lock_path)  # released on exit
    assert os.path.isfile(w.journal_path)
    assert not w.journal_path.startswith(trainer.root)


def test_step_order_for_same_poll_discovery(tmp_path):
    trainer = SyntheticTrainer(tmp_path / "ckpts")
    # Created out of order, all discovered in the same first poll.
    for step in (3, 1, 2):
        trainer.save(step, f"w{step}".encode(), b"o")
    w = make_watcher(tmp_path, trainer.root)
    observe(w)
    recs = manifests(load_journal_records(w.journal_path))
    assert [r["step"] for r in recs] == [1, 2, 3]
    assert [r["seq"] for r in recs] == sorted(r["seq"] for r in recs)


def test_one_watcher_lock_refuses_second(tmp_path):
    trainer = SyntheticTrainer(tmp_path / "ckpts")
    trainer.save(1, b"w1", b"o1")
    w = make_watcher(tmp_path, trainer.root)
    w2 = TrainingWatcher(
        trainer.root, str(tmp_path / "other" / "j.jsonl"),
        run_id="run-test", lock_path=w.lock_path)
    with w:
        with pytest.raises(WatcherAlreadyRunning):
            w2.acquire_lock()
    # After release the lock can be taken again.
    w2.acquire_lock()
    w2.release_lock()


def test_weights_only_checkpoint_has_null_optimizer(tmp_path):
    trainer = SyntheticTrainer(tmp_path / "ckpts")
    trainer.save(5, b"only-weights", optimizer=None)
    w = make_watcher(tmp_path, trainer.root)
    observe(w)
    (r,) = manifests(load_journal_records(w.journal_path))
    assert r["checkpoint_complete"] is True
    assert r["optimizer_blob"] is None and r["optimizer_bytes"] is None
    assert r["weights_blob"] == sha256_tag(b"only-weights")
