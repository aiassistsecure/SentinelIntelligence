"""SI-84: training watcher, observe-only (SI-83 section 9, slice 1).

This module implements the observation half of the SI-83 training
journal design, and nothing else:

- It polls a trainer's checkpoint root for ``step<N>`` directories.
- A checkpoint is treated as complete only after quiescence: its
  file set and file sizes are unchanged across two consecutive
  observations, a weights file is present and non-empty, and the
  trainer's completion marker (``SAVE_COMPLETE`` by default, the
  file the synthetic trainers in the tests write last) is present.
- It stream-hashes the weights file, and the optimizer file when
  one is present, with SHA-256. Hashing is read-only and happens
  after quiescence, so a hash names bytes the trainer finished
  writing. The journal stores hashes and paths, never weight
  bytes: SI never duplicates bytes.
- It appends ``checkpoint_manifest`` records, in the SI-83
  section 2.3 field shape, to a PLAIN FILE record (JSON lines).
  No NEDB write path exists in this slice, and none is used.
  Records carry ``seq`` / ``prev_hash`` / ``entry_hash`` using
  the SI-0 journal's hash-chain computation
  (``si.journal._entry_hash`` / ``GENESIS_PREV``), continued
  across restarts from the file's existing tip.
- A checkpoint observed but not complete -- for example a
  deliberately truncated save that stabilises without its
  completion marker -- is journaled with
  ``checkpoint_complete = false``. Such a row is never a
  rollback target (see :func:`rollback_targets`).
- On a restart that finds checkpoints on disk which were never
  journaled, the watcher first appends a ``watcher_gap`` record
  naming the covered step range, then journals the missing
  manifests. A gap is a row, never silence (SI-83 section 3.4).
- Unknown fields stay honest: ``trainer_commit`` is the literal
  string ``"unknown"`` and ``data_order_seed`` is ``None`` when
  the trainer did not record them. No value is invented.
- Within one poll, checkpoints are journaled in step order
  (SI-83 section 3.3).

What this watcher never does:

- It never writes into, renames, moves, or deletes anything in
  the trainer's checkpoint tree. Its only writes are the journal
  file and a one-watcher lock file, both OUTSIDE the trainer
  tree (SI-83 section 3.4). It does not import, wrap, patch, or
  otherwise modify trainer code; deleting this watcher leaves a
  trainer bit-for-bit unaffected.
- It does not run evaluations, roll anything back, replay
  anything, prune a corpus, or enforce retention. Those are
  later SI-83 section 9 slices and do not exist here.
- It does not journal any real training run. The Imagine model
  referenced by the wider mission is a specialised machine
  learning model, 1.6B parameters, full fine tune; no run of it
  is observed by this code as built and tested here.

This is machine learning infrastructure. CPU only, standard
library only.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone

from si.journal import GENESIS_PREV, _entry_hash

__all__ = [
    "COMPLETION_MARKER",
    "TrainingWatcher",
    "WatcherAlreadyRunning",
    "load_journal_records",
    "rollback_targets",
    "sha256_file_streaming",
]

_STEP_RE = re.compile(r"^step(\d+)$")

#: The default completion signal: the file a trainer writes last
#: for a save. Configurable per watcher (SI-83 section 3.1 step 2:
#: the concrete signal is a per-trainer decision, never guessed
#: silently -- here it is an explicit constructor argument with
#: this stated default, recorded on every manifest row).
COMPLETION_MARKER = "SAVE_COMPLETE"

_WEIGHTS_NAMES = (
    "weights.bin", "model.safetensors", "weights.safetensors",
    "pytorch_model.bin", "model.bin", "weights.pt", "model.pt",
)
_OPTIMIZER_HINTS = ("optimizer", "optim")
_META_NAMES = ("meta.json", "checkpoint.json", "manifest.json")
_CHUNK = 1024 * 1024


class WatcherAlreadyRunning(RuntimeError):
    """The one-watcher lock file already exists (SI-83 3.4)."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file_streaming(path: str) -> str:
    """``sha256:<hex>`` over a file, read in bounded chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def load_journal_records(journal_path: str) -> list:
    """Every record in a file-form training journal, in order."""
    if not os.path.exists(journal_path):
        return []
    out = []
    with open(journal_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def rollback_targets(records: list) -> list:
    """Manifest records that may be named as rollback targets.

    Only ``checkpoint_complete`` manifests qualify; a
    ``checkpoint_complete = false`` row is never returned.
    (This is a filter over records, not a rollback tool:
    no rollback exists in this slice.)
    """
    return [r for r in records
            if r.get("kind") == "checkpoint_manifest"
            and r.get("checkpoint_complete") is True]


def _snapshot(step_dir: str) -> tuple:
    """Sorted ``(relative_path, size)`` for every file below a
    checkpoint directory. Read-only: stats only, no writes."""
    items = []
    for root, _dirs, files in os.walk(step_dir):
        for name in files:
            path = os.path.join(root, name)
            try:
                items.append((os.path.relpath(path, step_dir),
                              os.path.getsize(path)))
            except OSError:
                items.append((os.path.relpath(path, step_dir), -1))
    return tuple(sorted(items))


def _pick(snapshot: tuple, step_dir: str, marker: str):
    """Choose (weights_rel, optimizer_rel) from a snapshot."""
    names = [rel for rel, _size in snapshot]
    weights = next((n for n in _WEIGHTS_NAMES if n in names), None)
    optimizer = next(
        (n for n in names
         if any(h in os.path.basename(n).lower() for h in _OPTIMIZER_HINTS)),
        None)
    if weights is None:
        candidates = [n for n in names
                      if n != marker and n not in _META_NAMES
                      and n != optimizer]
        if candidates:
            sizes = dict(snapshot)
            weights = max(candidates, key=lambda n: sizes[n])
    return weights, optimizer


class TrainingWatcher:
    """Observe-only watcher over one trainer checkpoint root.

    Parameters are explicit configuration, never derived:
    ``checkpoint_root`` (trainer tree, read-only to us),
    ``journal_path`` (JSONL file, outside the trainer tree),
    and ``run_id``. ``lock_path`` defaults to the journal path
    plus ``.lock``, also outside the trainer tree.
    """

    def __init__(self, checkpoint_root: str, journal_path: str,
                 run_id: str, *, corpus_hash: str | None = None,
                 hyperparams: dict | None = None,
                 trainer_commit: str = "unknown",
                 data_order_seed: int | None = None,
                 epoch: float | None = None,
                 completion_marker: str | None = COMPLETION_MARKER,
                 poll_interval: float = 30.0,
                 lock_path: str | None = None):
        self.checkpoint_root = os.path.abspath(checkpoint_root)
        self.journal_path = os.path.abspath(journal_path)
        if self.journal_path == self.checkpoint_root or \
                self.journal_path.startswith(self.checkpoint_root + os.sep):
            raise ValueError("journal_path must be outside the "
                             "trainer checkpoint tree")
        self.run_id = str(run_id)
        self.corpus_hash = corpus_hash
        self.hyperparams = dict(hyperparams) if hyperparams else {}
        self.trainer_commit = trainer_commit
        self.data_order_seed = data_order_seed
        self.epoch = epoch
        self.completion_marker = completion_marker
        self.poll_interval = float(poll_interval)
        self.lock_path = os.path.abspath(lock_path) if lock_path else \
            self.journal_path + ".lock"
        if self.lock_path.startswith(self.checkpoint_root + os.sep):
            raise ValueError("lock_path must be outside the "
                             "trainer checkpoint tree")
        # Resume the file journal's chain / clock from its tip.
        self._records = load_journal_records(self.journal_path)
        self._seq = 0
        self._last_hash = GENESIS_PREV
        for r in self._records:
            if int(r.get("seq") or 0) > self._seq:
                self._seq = int(r["seq"])
                self._last_hash = r.get("entry_hash") or self._last_hash
        self._journaled = {}  # step -> (snapshot, complete)
        for r in self._records:
            if r.get("kind") == "checkpoint_manifest":
                self._journaled[int(r["step"])] = (
                    tuple(tuple(x) for x in r.get("_snapshot", [])),
                    bool(r.get("checkpoint_complete")))
        self._pending = {}  # step -> (snapshot, consecutive_count)
        self._gap_checked = False
        self._lock_fd = None

    # -- lock (outside the trainer tree, SI-83 section 3.4) -----------------
    def acquire_lock(self) -> None:
        if self._lock_fd is not None:
            return
        os.makedirs(os.path.dirname(self.lock_path) or ".", exist_ok=True)
        try:
            self._lock_fd = os.open(
                self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise WatcherAlreadyRunning(
                f"watcher lock already exists: {self.lock_path}") from exc
        os.write(self._lock_fd, f"{os.getpid()}\n".encode())

    def release_lock(self) -> None:
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None
            try:
                os.unlink(self.lock_path)
            except FileNotFoundError:
                pass

    def __enter__(self):
        self.acquire_lock()
        return self

    def __exit__(self, *exc):
        self.release_lock()
        return False

    close = release_lock

    # -- journal append (file form only; no NEDB write path) ----------------
    def _append(self, payload: dict) -> dict:
        self._seq += 1
        entry_hash = _entry_hash(self._last_hash, self._seq, payload)
        record = {"ts": _utcnow(), **payload, "seq": self._seq,
                  "prev_hash": self._last_hash, "entry_hash": entry_hash}
        os.makedirs(os.path.dirname(self.journal_path) or ".",
                    exist_ok=True)
        with open(self.journal_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        self._last_hash = entry_hash
        self._records.append(record)
        return record

    # -- discovery ------------------------------------------------------------
    def discover(self) -> list:
        """``[(step, dir_path)]`` for step<N> dirs, in step order."""
        found = []
        if not os.path.isdir(self.checkpoint_root):
            return found
        for name in os.listdir(self.checkpoint_root):
            m = _STEP_RE.match(name)
            path = os.path.join(self.checkpoint_root, name)
            if m and os.path.isdir(path):
                found.append((int(m.group(1)), path))
        return sorted(found)

    def _meta(self, step_dir: str) -> dict:
        for name in _META_NAMES:
            path = os.path.join(step_dir, name)
            if os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        return data
                except (OSError, ValueError):
                    return {}
        return {}

    def _manifest(self, step: int, step_dir: str, snapshot: tuple,
                  complete: bool) -> dict:
        meta = self._meta(step_dir)
        weights_rel, optimizer_rel = _pick(
            snapshot, step_dir, self.completion_marker or "")
        sizes = dict(snapshot)

        def blob(rel):
            if rel is None:
                return None, None
            return (sha256_file_streaming(os.path.join(step_dir, rel)),
                    int(sizes[rel]))

        weights_blob, weights_bytes = blob(weights_rel)
        optimizer_blob, optimizer_bytes = blob(optimizer_rel)
        hyper = dict(self.hyperparams)
        if isinstance(meta.get("hyperparams"), dict):
            hyper.update(meta["hyperparams"])
        return {
            "kind": "checkpoint_manifest",
            "event_kind": "checkpoint_manifest",
            "run_id": self.run_id,
            "step": int(step),
            "epoch": meta.get("epoch", self.epoch),
            "weights_blob": weights_blob,
            "weights_path": step_dir if weights_rel else None,
            "weights_file": weights_rel,
            "weights_bytes": weights_bytes,
            "optimizer_blob": optimizer_blob,
            "optimizer_bytes": optimizer_bytes,
            "corpus_hash": meta.get("corpus_hash", self.corpus_hash),
            "data_order_seed": meta.get(
                "data_order_seed", self.data_order_seed),
            "hyperparams": hyper,
            "trainer_commit": meta.get(
                "trainer_commit", self.trainer_commit),
            "checkpoint_complete": bool(complete),
            "completion_marker": self.completion_marker,
            "_snapshot": [list(x) for x in snapshot],
        }

    # -- one poll ---------------------------------------------------------------
    def observe_once(self) -> list:
        """One poll. Returns the records appended by this poll.

        A checkpoint is journaled only after its snapshot is
        unchanged across two consecutive observations. On the
        first poll after a restart, missing manifests are
        preceded by a ``watcher_gap`` record (see module doc).
        Checkpoints discovered in the same poll are journaled
        in step order (SI-83 section 3.3).
        """
        discovered = self.discover()
        appended = []
        # Restart gap check, once per watcher instance: a prior
        # journal exists and disk holds steps it never recorded.
        if not self._gap_checked:
            self._gap_checked = True
            prior_steps = {int(r["step"]) for r in self._records
                           if r.get("kind") == "checkpoint_manifest"}
            if self._records and prior_steps:
                missing_now = [s for s, _ in discovered
                               if s not in prior_steps]
                if missing_now:
                    appended.append(self._append({
                        "kind": "watcher_gap",
                        "event_kind": "watcher_gap",
                        "run_id": self.run_id,
                        "step_from": min(missing_now),
                        "step_to": max(missing_now),
                        "covered_steps": sorted(missing_now),
                        "last_journaled_step_before_gap":
                            max(prior_steps),
                        "reason": "watcher restarted; re-scan found "
                                  "checkpoints not present in the "
                                  "journal -- gap recorded, then "
                                  "missing manifests journaled",
                    }))
        ready = []
        for step, step_dir in discovered:
            snapshot = _snapshot(step_dir)
            prev = self._journaled.get(step)
            if prev is not None and prev[0] == snapshot:
                continue  # this exact state already journaled
            seen_snapshot, count = self._pending.get(step, (None, 0))
            count = count + 1 if seen_snapshot == snapshot else 1
            self._pending[step] = (snapshot, count)
            if count >= 2:
                ready.append((step, step_dir, snapshot))
        for step, step_dir, snapshot in sorted(ready):
            names = {rel for rel, _ in snapshot}
            weights_rel, _opt = _pick(
                snapshot, step_dir, self.completion_marker or "")
            sizes = dict(snapshot)
            marker_ok = (self.completion_marker is None
                         or self.completion_marker in names)
            complete = bool(weights_rel and sizes.get(weights_rel, 0) > 0
                            and marker_ok)
            record = self._append(
                self._manifest(step, step_dir, snapshot, complete))
            self._journaled[step] = (snapshot, complete)
            self._pending.pop(step, None)
            appended.append(record)
        return appended

    # Alias matching the spec's "poll" language.
    poll_once = observe_once

    def run(self, max_polls: int | None = None) -> list:
        """Poll loop (blocking). ``max_polls`` bounds it for tests
        and bounded operator runs; ``None`` polls until stopped
        by KeyboardInterrupt. Acquires the one-watcher lock."""
        self.acquire_lock()
        appended = []
        try:
            polls = 0
            while max_polls is None or polls < max_polls:
                appended.extend(self.observe_once())
                polls += 1
                if max_polls is None or polls < max_polls:
                    time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            pass
        finally:
            self.release_lock()
        return appended
