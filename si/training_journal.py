"""SI-85: training journal writer + database (SI-83 section 9, slice 2).

Implements the five SI-83 section 2.2 training collections as
SEPARATE collections -- ``training_runs``, ``training_checkpoints``,
``training_eval_moments``, ``training_windows``,
``training_proposals`` -- never merged into the SI-0 collections
(``self_edits`` / ``deltas`` / ``checkpoints`` / ``provenance`` /
``experiences``). Rows are written through the SI-0 journal's
machinery: the same ``seq`` logical clock, ``prev_hash`` /
``entry_hash`` chain computed by ``si.journal._entry_hash`` with
``GENESIS_PREV`` for the first entry, the same mutable-key
exclusion, and the same ``JournalBackend`` protocol
(``si.journal``). No parallel hash scheme is invented here.

Database naming (SI-83 section 7, fail closed):

- The ONLY accepted database name is the literal
  ``si_training_journal``, taken from explicit configuration.
  Absent/empty configuration is a hard failure, and a writer
  configured with ANY other name -- including ``shop`` and
  ``stealth_shop``, and any derived name such as ``si_<bundle>``
  -- refuses to write. There is NO fallback derivation (not
  ``_dbname_for``, not an environment default).
- Before the first write, every open performs the section 7.3
  read-back handshake: the connected database's actual name is
  read back from the backend and compared byte-for-byte with the
  configured name. A mismatch -- or an inability to read the
  name back -- fails closed: no write, and the error states BOTH
  names (configured vs actual/unavailable).
- The writer never creates a differently-named database to make
  progress; it creates no database at all.
- If ``verify_chain`` fails on an existing journal, opening for
  write refuses.
- The genesis (first) row records the verified database name,
  the pgwire/HTTP ports derived per
  ``si/backends/nedb.py::_ports_for``, and the verification
  timestamp.

Which backends the handshake is REAL for, stated plainly:

- Real NEDB backend (``si.backends.nedb.NedbJournalBackend``):
  the read-back is the backend's ``dbname`` attribute -- the name
  that backend actually opened over pgwire (its DSN database).
  It is NOT an independent server query (no
  ``SELECT current_database()`` is issued). No live-NEDB write is
  performed or claimed by this slice's tests (embedded only).
- Embedded (sqlite): a raw ``si.journal.EmbeddedBackend`` cannot
  report a database name, so the handshake fails closed for it.
  :class:`EmbeddedTrainingBackend` persists the name given when
  its storage directory is first created (``database_name``
  file); the handshake reads THAT file back from disk -- not the
  caller's in-memory configuration -- so a directory created
  under another name fails, testable without a live daemon.
- Other backends: ``dbname`` (str, checked first), then
  ``database_name`` (str), then a callable
  ``reported_database_name()`` / ``database_name()``; a backend
  exposing none of them fails the handshake closed.

Honesty (SI-83 section 2.6): no weights bytes are ever stored --
checkpoint rows carry content hashes and external paths only,
and byte values passed where a hash belongs are refused.
``trainer_commit`` / ``eval_suite_commit`` default to the
literal ``"unknown"`` and ``data_order_seed`` to ``None`` when
unrecorded; no value is invented. Eval scores are recorded
verbatim; ``None`` stays ``None``. Sample generations are
stored truncated to 4,096 characters per generation with the
full text's sha256 always recorded, so truncation is detectable
and never mistaken for completeness.

Scope: this slice is the writer only. It builds no rollback /
replay tooling (slice 3), no fork/prune EXECUTION flow beyond
journaling the proposal/verdict/lineage rows (slice 4), and no
retention enforcement (slice 5). This is machine learning
infrastructure. CPU only, standard library only.

Honesty labels, unchanged: section 18.3 NOT earned; P5-R1 /
P5-R9 UNMEASURED; live model-authored intent OCCURRED scoped
to SI-74 only; L-3 LOCKED; A-2 / A-3 / A-4 UNAPPROVED;
unlocks []; D-F1 OPEN; F-line STOPPED.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone

from si.journal import (
    EmbeddedBackend, GENESIS_PREV, MUTABLE_KEYS, _entry_hash, _jsonable,
)

__all__ = [
    "TRAINING_DB_NAME", "TRAINING_COLLECTIONS", "EVENT_KINDS",
    "GENESIS_KIND", "TrainingJournalError", "DatabaseNameRefused",
    "HandshakeFailed", "ChainVerificationFailed",
    "EmbeddedTrainingBackend", "TrainingJournalWriter", "ports_for",
]

#: SI-83 section 7.2: the ONLY accepted training-journal database name.
TRAINING_DB_NAME = "si_training_journal"

#: SI-83 section 2.2: five NEW collections, separate from SI-0's.
TRAINING_COLLECTIONS = (
    "training_runs", "training_checkpoints", "training_eval_moments",
    "training_windows", "training_proposals",
)

#: SI-83 section 2.5: every event kind a row's ``kind`` may carry.
EVENT_KINDS = (
    "run_started", "checkpoint_manifest", "eval_moment",
    "window_consumed", "prune_proposed", "prune_verdict",
    "rollback", "fork", "retention_decision", "watcher_gap",
)

#: Kind of the section 7.4 genesis record (not a section 2.5 event).
GENESIS_KIND = "genesis"

_RELATIONS = ("rolled_back_to", "forked_from", "pruned_from", "supersedes")
_GENESIS_PREV = GENESIS_PREV
_CHAIN_KEYS = frozenset({"_id", "seq", "prev_hash", "entry_hash"})
_NATIVE_KEYS = frozenset({"_seq", "_hash"})
_MAX_GENERATION_CHARS = 4096


def ports_for(dbname: str) -> tuple:
    """(pgwire, http) ports, derived exactly as
    ``si.backends.nedb._ports_for`` derives them (imported when
    importable; formula restated so the embedded path needs no
    psycopg2 import)."""
    try:
        from si.backends.nedb import _ports_for
        return _ports_for(dbname)
    except Exception:
        h = int(hashlib.sha256(dbname.encode("utf-8")).hexdigest(), 16)
        return 55433 + (h % 1000), 17070 + (h % 1000)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


class TrainingJournalError(RuntimeError):
    """Base for all training-journal refusals (fail closed)."""


class DatabaseNameRefused(TrainingJournalError):
    """Configured name is absent/empty or not the literal."""


class HandshakeFailed(TrainingJournalError):
    """Read-back handshake failed: mismatch or name unavailable."""


class ChainVerificationFailed(TrainingJournalError):
    """Existing journal's chain does not verify; open refused."""


class EmbeddedTrainingBackend:
    """Embedded (sqlite) backend that can report its database
    name: wraps ``si.journal.EmbeddedBackend`` (delegated
    protocol), persists the name in the journal directory at
    first creation, and reads that file back on every call."""

    NAME_FILE = "database_name"

    def __init__(self, journal_dir: str, database_name: str):
        self.journal_dir = os.path.abspath(journal_dir)
        os.makedirs(self.journal_dir, exist_ok=True)
        self._name_path = os.path.join(self.journal_dir, self.NAME_FILE)
        if not os.path.exists(self._name_path):
            with open(self._name_path, "w", encoding="utf-8") as f:
                f.write(str(database_name))
        self._inner = EmbeddedBackend(self.journal_dir)

    def reported_database_name(self):
        try:
            with open(self._name_path, "r", encoding="utf-8") as f:
                return f.read()
        except OSError:
            return None

    def insert(self, collection, doc):
        return self._inner.insert(collection, doc)

    def update(self, collection, row_id, doc):
        return self._inner.update(collection, row_id, doc)

    def find(self, collection, filter=None, sort=None, limit=None):
        return self._inner.find(collection, filter=filter, sort=sort,
                                limit=limit)

    def count(self, collection):
        return self._inner.count(collection)

    def close(self):
        return self._inner.close()


def _reported_name(backend):
    """The backend's connected database name, or None if the
    backend cannot report one (handshake then fails closed)."""
    val = getattr(backend, "dbname", None)
    if isinstance(val, str):
        return val
    for attr in ("reported_database_name", "database_name"):
        val = getattr(backend, attr, None)
        if callable(val):
            try:
                val = val()
            except Exception:
                return None
        if isinstance(val, str):
            return val
    return None


def _honest_commit(value):
    return value if value else "unknown"  # SI-83 section 2.6


class TrainingJournalWriter:
    """Append-only writer over the five training collections.

    Construction validates the configured name; :meth:`open`
    (also run by the constructor by default) performs the
    section 7.3 handshake and chain verification BEFORE any
    write, then appends the genesis row if the journal is empty.
    """

    def __init__(self, backend, database_name=None, *, open_now=True):
        if database_name is None or str(database_name) == "":
            raise DatabaseNameRefused(
                "training journal database name is absent/empty in "
                "explicit configuration; the only accepted name is "
                f"{TRAINING_DB_NAME!r} (SI-83 section 7.3: no fallback "
                "derivation)")
        if database_name != TRAINING_DB_NAME:
            raise DatabaseNameRefused(
                f"training journal database name {database_name!r} "
                f"refused: the only accepted name is "
                f"{TRAINING_DB_NAME!r} (SI-83 section 7.2)")
        self.database_name = database_name
        self.backend = backend
        self._lock = threading.RLock()
        self._seq = 0
        self._last_hash = GENESIS_PREV
        self._opened = False
        self.verified_database_name = None
        self.verified_at = None
        if open_now:
            self.open()

    @classmethod
    def for_embedded(cls, journal_dir: str, database_name=None):
        """Writer over :class:`EmbeddedTrainingBackend` storage."""
        backend = EmbeddedTrainingBackend(
            journal_dir, database_name if database_name else "")
        return cls(backend, database_name)

    # -- open / handshake ---------------------------------------------------

    def open(self):
        with self._lock:
            if self._opened:
                return self
            reported = _reported_name(self.backend)
            if reported is None:
                raise HandshakeFailed(
                    "read-back handshake failed: configured database "
                    f"name {self.database_name!r}, actual database "
                    "name unavailable (backend cannot report its "
                    "database name); refusing to write (SI-83 "
                    "section 7.3)")
            if reported != self.database_name:
                raise HandshakeFailed(
                    "read-back handshake failed: configured database "
                    f"name {self.database_name!r} != actual database "
                    f"name {reported!r}; refusing to write (SI-83 "
                    "section 7.3)")
            # Catch up the logical clock from existing rows.
            rows = self._all_rows()
            for row in rows:
                seq = row.get("seq") or 0
                if seq > self._seq:
                    self._seq = seq
                    self._last_hash = row.get("entry_hash") or self._last_hash
            if rows and not self.verify_chain():
                raise ChainVerificationFailed(
                    "existing training journal failed hash-chain "
                    "verification; opening for write refused")
            self.verified_database_name = reported
            self.verified_at = _utcnow()
            self._opened = True
            if not rows:
                pg_port, http_port = ports_for(self.database_name)
                self._append("training_runs", {
                    "kind": GENESIS_KIND, "event_kind": GENESIS_KIND,
                    "database_name": reported,
                    "pg_port": pg_port, "http_port": http_port,
                    "verified_at": self.verified_at,
                    "note": "genesis: database name verified by "
                            "read-back handshake (SI-83 section 7.4)",
                })
            return self

    def close(self):
        self.backend.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
        return False

    # -- chain machinery (SI-0 patterns, si.journal._entry_hash) -------------

    def _all_rows(self):
        rows = []
        for coll in TRAINING_COLLECTIONS:
            rows.extend(self.backend.find(coll))
        rows.sort(key=lambda r: (r.get("seq") or 0, str(r["_id"])))
        return rows

    def _append(self, collection, payload):
        if collection not in TRAINING_COLLECTIONS:
            raise ValueError(f"not a training collection: {collection!r}")
        if not self._opened:
            raise TrainingJournalError("writer is not open")
        with self._lock:
            payload = _jsonable(payload)
            hash_payload = {k: v for k, v in payload.items()
                            if k not in MUTABLE_KEYS}
            self._seq += 1
            seq = self._seq
            entry_hash = _entry_hash(self._last_hash, seq, hash_payload)
            doc = {"ts": _utcnow(), **payload, "seq": seq,
                   "prev_hash": self._last_hash, "entry_hash": entry_hash}
            row_id = self.backend.insert(collection, doc)
            self._last_hash = entry_hash
            return {"_id": row_id, "seq": seq, "entry_hash": entry_hash,
                    **payload}

    @staticmethod
    def _chain_payload(row):
        return {k: v for k, v in row.items()
                if k not in _CHAIN_KEYS and k not in _NATIVE_KEYS
                and k not in MUTABLE_KEYS and k != "ts"}

    def verify_chain(self) -> bool:
        """True iff every link across all five collections verifies."""
        prev, prev_seq = GENESIS_PREV, 0
        for row in self._all_rows():
            seq = row.get("seq") or 0
            if seq <= prev_seq or row.get("prev_hash") != prev:
                return False
            if _entry_hash(prev, seq, self._chain_payload(row)) \
                    != row.get("entry_hash"):
                return False
            prev, prev_seq = row.get("entry_hash"), seq
        return True

    def rows(self, collection, filter=None):
        return self.backend.find(collection, filter=filter, sort="seq")

    def total_rows(self) -> int:
        return sum(self.backend.count(c) for c in TRAINING_COLLECTIONS)

    # -- record methods: one per section 2.5 event kind -----------------------

    def _record(self, collection, kind, fields):
        return self._append(collection, {
            "kind": kind, "event_kind": kind, **fields})

    def record_run_started(self, run_id, *, model=None, corpus_hash=None,
                           hyperparams=None, trainer_commit=None,
                           data_order_seed=None, parent_run_id=None,
                           parent_step=None, parent_manifest_hash=None,
                           **extra):
        return self._record("training_runs", "run_started", {
            "run_id": str(run_id), "model": model,
            "corpus_hash": corpus_hash,
            "hyperparams": dict(hyperparams) if hyperparams else {},
            "trainer_commit": _honest_commit(trainer_commit),
            "data_order_seed": data_order_seed,
            "parent_run_id": parent_run_id, "parent_step": parent_step,
            "parent_manifest_hash": parent_manifest_hash, **extra})

    def record_checkpoint_manifest(self, run_id, step, *, epoch=None,
                                   weights_blob=None, weights_path=None,
                                   weights_bytes=None, optimizer_blob=None,
                                   optimizer_bytes=None, corpus_hash=None,
                                   data_order_seed=None, hyperparams=None,
                                   trainer_commit=None,
                                   checkpoint_complete=True, **extra):
        for name, val in (("weights_blob", weights_blob),
                          ("optimizer_blob", optimizer_blob)):
            if isinstance(val, (bytes, bytearray)):
                raise TypeError(
                    f"{name} must be a sha256 content hash, never "
                    "weights bytes (SI-83 section 2.6)")
        return self._record("training_checkpoints", "checkpoint_manifest", {
            "run_id": str(run_id), "step": int(step), "epoch": epoch,
            "weights_blob": weights_blob, "weights_path": weights_path,
            "weights_bytes": weights_bytes,
            "optimizer_blob": optimizer_blob,
            "optimizer_bytes": optimizer_bytes,
            "corpus_hash": corpus_hash,
            "data_order_seed": data_order_seed,
            "hyperparams": dict(hyperparams) if hyperparams else {},
            "trainer_commit": _honest_commit(trainer_commit),
            "checkpoint_complete": bool(checkpoint_complete), **extra})

    def record_eval_moment(self, run_id, step, *, eval_suite=None,
                           eval_suite_commit=None, tier0_score=None,
                           discrimination_score=None, temporal_score=None,
                           loss=None, grad_norm_mean=None,
                           grad_norm_max=None, sample_generations=None,
                           eval_wall_ts=None, **extra):
        gens = []
        for item in (sample_generations or []):
            text = item.get("generation", "")
            if not isinstance(text, str):
                text = str(text)
            gens.append({
                "prompt": item.get("prompt"),
                "generation": text[:_MAX_GENERATION_CHARS],
                "generation_hash": _sha256_text(text),
                "generation_truncated": len(text) > _MAX_GENERATION_CHARS,
            })
        return self._record("training_eval_moments", "eval_moment", {
            "run_id": str(run_id), "step": int(step),
            "eval_suite": eval_suite,
            "eval_suite_commit": _honest_commit(eval_suite_commit),
            "tier0_score": tier0_score,
            "discrimination_score": discrimination_score,
            "temporal_score": temporal_score, "loss": loss,
            "grad_norm_mean": grad_norm_mean,
            "grad_norm_max": grad_norm_max,
            "sample_generations": gens,
            "eval_wall_ts": eval_wall_ts or _utcnow(), **extra})

    def record_window_consumed(self, run_id, step_from, step_to, *,
                               corpus_hash=None, data_order_seed=None,
                               example_start=None, example_end=None,
                               example_ids_hash=None, order_source=None,
                               **extra):
        return self._record("training_windows", "window_consumed", {
            "run_id": str(run_id), "step_from": int(step_from),
            "step_to": int(step_to), "corpus_hash": corpus_hash,
            "data_order_seed": data_order_seed,
            "example_start": example_start, "example_end": example_end,
            "example_ids_hash": example_ids_hash,
            "order_source": order_source, **extra})

    def record_prune_proposed(self, run_id, *, reason=None, step_from=None,
                              step_to=None, window_seqs=None,
                              example_ids=None, **extra):
        if not reason:
            raise ValueError("a prune proposal without a reason is "
                             "malformed (SI-83 section 5.3)")
        return self._record("training_proposals", "prune_proposed", {
            "run_id": str(run_id), "reason": reason,
            "step_from": step_from, "step_to": step_to,
            "window_seqs": list(window_seqs) if window_seqs else [],
            "example_ids": list(example_ids) if example_ids else [],
            **extra})

    def _proposal_with_parents(self, kind, run_id, parents, relation,
                               fields):
        if relation not in _RELATIONS:
            raise ValueError(f"relation must be one of {_RELATIONS}, "
                             f"got {relation!r}")
        if not parents:
            raise ValueError(f"{kind} rows must carry parents "
                             "(SI-83 section 2.5)")
        return self._record("training_proposals", kind, {
            "run_id": str(run_id), "parents": list(parents),
            "relation": relation, **fields})

    def record_prune_verdict(self, run_id, *, verdict=None, parents=None,
                             relation="pruned_from", operator=None,
                             new_corpus_hash=None, **extra):
        return self._proposal_with_parents(
            "prune_verdict", run_id, parents, relation, {
                "gate_verdict": verdict, "operator": operator,
                "new_corpus_hash": new_corpus_hash, **extra})

    def record_rollback(self, run_id, *, target_step=None, reason=None,
                        verdict=None, parents=None,
                        relation="rolled_back_to", operator=None, **extra):
        return self._proposal_with_parents(
            "rollback", run_id, parents, relation, {
                "target_step": target_step, "reason": reason,
                "gate_verdict": verdict, "operator": operator, **extra})

    def record_fork(self, source_run_id, new_run_id, *, parent_step=None,
                    parent_manifest_hash=None, parents=None,
                    relation="forked_from", reason=None, verdict=None,
                    operator=None, hyperparams=None, corpus_hash=None,
                    trainer_commit=None, **extra):
        """Journal a fork: a ``fork`` row in training_proposals AND
        a new ``training_runs`` row carrying parent_run_id /
        parent_step / parent_manifest_hash (lineage walkable both
        ways, SI-83 section 2.5). Rows only; no run is started or
        copied (slice 4 execution is not built)."""
        if parents is None:
            parents = []
        proposal = self._proposal_with_parents(
            "fork", source_run_id, parents, relation, {
                "reason": reason, "gate_verdict": verdict,
                "operator": operator, "new_run_id": str(new_run_id),
                "parent_step": parent_step,
                "parent_manifest_hash": parent_manifest_hash, **extra})
        new_run = self._record("training_runs", "fork", {
            "run_id": str(new_run_id),
            "parent_run_id": str(source_run_id),
            "parent_step": parent_step,
            "parent_manifest_hash": parent_manifest_hash,
            "corpus_hash": corpus_hash,
            "hyperparams": dict(hyperparams) if hyperparams else {},
            "trainer_commit": _honest_commit(trainer_commit),
            "data_order_seed": None, "reason": reason})
        return {"proposal": proposal, "run": new_run}

    def record_retention_decision(self, run_id, *, step=None, keep=None,
                                  verdict=None, operator=None,
                                  reason=None, **extra):
        return self._record("training_proposals", "retention_decision", {
            "run_id": str(run_id), "step": step, "keep": keep,
            "gate_verdict": verdict, "operator": operator,
            "reason": reason, **extra})

    def record_watcher_gap(self, run_id, *, step_from=None, step_to=None,
                           covered_steps=None, reason=None, **extra):
        return self._record("training_runs", "watcher_gap", {
            "run_id": str(run_id), "step_from": step_from,
            "step_to": step_to,
            "covered_steps": (list(covered_steps)
                              if covered_steps else []),
            "reason": reason, **extra})

    # -- lineage reads ---------------------------------------------------------

    def lineage_to_root(self, run_id):
        """Run rows from ``run_id`` back to its lineage root."""
        runs = {r.get("run_id"): r for r in self.rows("training_runs")
                if r.get("run_id")}
        out, seen, cur = [], set(), str(run_id)
        while cur and cur in runs and cur not in seen:
            seen.add(cur)
            row = runs[cur]
            out.append(row)
            cur = row.get("parent_run_id")
        return out

    def children_of(self, run_id):
        """Run rows whose parent_run_id is ``run_id`` (fork children)."""
        return [r for r in self.rows("training_runs")
                if r.get("parent_run_id") == str(run_id)]
