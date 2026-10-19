"""The journal: hash-chained, logical-clocked record of every self-edit.

Implements docs/nedb-journal-schema.md for the SI-0 reference runtime.

Every entry carries ``seq`` (monotonic int, runtime-assigned — the mind's
own time), ``prev_hash``, and ``entry_hash = sha256(canonical(prev_hash |
seq | payload))``. The runtime verifies the chain at load and refuses to
run on a broken chain (fail closed).

The journal never lets a write bypass seq/chain: all writes go through
``Journal._append``, which assigns seq, links the hash, and only then
touches the backend.

``result`` / ``delta_id`` / ``status`` are *operational state* (they change
as a self-edit moves pending -> materialized, or a delta is rolled back).
They are excluded from the hash chain; the causal record — intent, reason,
gate verdict/report — is chain-protected and immutable. This is documented
here and not hidden: mutating a result does not rewrite history, it only
updates bookkeeping.

HONEST: the default backend is the REAL NEDB engine over pgwire
(``si.backends.nedb``) -- the dogfood directive: the mind's journal lives
in the actual NEDB engine. ``si init --journal-backend embedded`` selects
the zero-service sqlite fallback. The backend name is recorded in the
manifest and honored on every load.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from typing import Protocol

COLLECTIONS = ("self_edits", "deltas", "checkpoints", "provenance", "experiences")

#: prev_hash of the first journal entry ever.
GENESIS_PREV = "genesis"

#: Keys excluded from the hash chain (operational state, see module doc).
MUTABLE_KEYS = frozenset({"result", "delta_id", "status"})

_CHAIN_KEYS = frozenset({"_id", "seq", "prev_hash", "entry_hash"})
#: Native keys a real-NEDB backend merges into find() results. They were not
#: present when entry_hash was computed, so they are excluded from the
#: chain payload (same reason as _CHAIN_KEYS). See si/backends/nedb.py.
_NATIVE_KEYS = frozenset({"_seq", "_hash"})
_TABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def make_backend(bundle_dir: str, backend_name: str = "nedb") -> JournalBackend:
    """Instantiate the journal backend recorded in the manifest.

    ``nedb`` (default): the real NEDB engine over pgwire (si.backends.nedb).
    ``embedded``: sqlite file under the bundle's journal dir (fallback).
    Unknown names raise ValueError -- the caller (load_bundle) fails closed.
    """
    journal_dir = os.path.join(os.path.abspath(bundle_dir), "journal")
    if backend_name == "embedded":
        return EmbeddedBackend(journal_dir)
    if backend_name == "nedb":
        from si.backends.nedb import NedbJournalBackend
        return NedbJournalBackend(journal_dir)
    raise ValueError(f"unknown journal backend {backend_name!r}")


def open_journal(bundle_dir: str, manifest: dict | None = None) -> "Journal":
    """Open the Journal for a bundle, honoring the manifest's backend.

    Pass an already-loaded (verified) manifest when you have one; otherwise
    the backend name is read from manifest.json on disk.
    """
    if manifest is None:
        manifest_path = os.path.join(os.path.abspath(bundle_dir), "manifest.json")
        with open(manifest_path, "rb") as f:
            manifest = json.loads(f.read().decode("utf-8"))
    backend_name = manifest.get("journal", {}).get("backend", "embedded")
    return Journal(make_backend(bundle_dir, backend_name))


class JournalBackend(Protocol):
    """Storage protocol for the journal. The real NEDB backend
    (si.backends.nedb) is the default; the embedded sqlite backend is the
    zero-service fallback."""

    def insert(self, collection: str, doc: dict) -> str:
        """Store doc (a JSON-able dict), return the row id as str."""
        ...

    def update(self, collection: str, row_id: str, doc: dict) -> None:
        """Replace the stored doc for row_id (used for operational-state
        transitions like pending -> materialized)."""
        ...

    def find(self, collection: str, filter: dict | None = None,
             sort: list | str | None = None, limit: int | None = None) -> list:
        """Return stored docs (each merged with its ``_id``). Filter is
        exact-match on top-level keys. Sort: e.g. ``"-seq"`` or
        ``[("seq", True)]`` (True = descending)."""
        ...

    def count(self, collection: str) -> int:
        ...

    def close(self) -> None:
        ...


def _entry_hash(prev_hash: str, seq: int, payload: dict) -> str:
    body = {"prev_hash": prev_hash, "seq": seq, "payload": payload}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _jsonable(obj):
    """Convert numpy scalars/arrays (and friends) to plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, (str, int, float)):
        return obj
    if hasattr(obj, "tolist"):  # numpy arrays
        return _jsonable(obj.tolist())
    if hasattr(obj, "item"):  # numpy scalars
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


class EmbeddedBackend(JournalBackend):
    """Sqlite-backed journal. One table per collection; docs stored as JSON
    text plus indexed seq/_id columns. File: ``journal/journal.sqlite3``."""

    def __init__(self, journal_dir: str):
        os.makedirs(journal_dir, exist_ok=True)
        self.path = os.path.join(journal_dir, "journal.sqlite3")
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA journal_mode=WAL")

    def _table(self, collection: str) -> str:
        if not _TABLE_RE.match(collection):
            raise ValueError(f"bad collection name: {collection!r}")
        self._conn.execute(
            f'CREATE TABLE IF NOT EXISTS "{collection}" ('
            ' "_id" INTEGER PRIMARY KEY AUTOINCREMENT,'
            ' "seq" INTEGER,'
            ' "prev_hash" TEXT,'
            ' "entry_hash" TEXT,'
            ' "doc" TEXT NOT NULL)'
        )
        self._conn.execute(
            f'CREATE INDEX IF NOT EXISTS "{collection}_seq_idx" ON "{collection}" ("seq")'
        )
        return collection

    def insert(self, collection: str, doc: dict) -> str:
        table = self._table(collection)
        doc = _jsonable(doc)
        cur = self._conn.execute(
            f'INSERT INTO "{table}" ("seq","prev_hash","entry_hash","doc") VALUES (?,?,?,?)',
            (doc.get("seq"), doc.get("prev_hash"), doc.get("entry_hash"),
             json.dumps(doc, sort_keys=True, separators=(",", ":"))),
        )
        self._conn.commit()
        return str(cur.lastrowid)

    def update(self, collection: str, row_id: str, doc: dict) -> None:
        table = self._table(collection)
        doc = _jsonable(doc)
        self._conn.execute(
            f'UPDATE "{table}" SET "seq"=?, "prev_hash"=?, "entry_hash"=?, "doc"=? WHERE "_id"=?',
            (doc.get("seq"), doc.get("prev_hash"), doc.get("entry_hash"),
             json.dumps(doc, sort_keys=True, separators=(",", ":")),
             int(row_id)),
        )
        self._conn.commit()

    def find(self, collection: str, filter: dict | None = None,
             sort: list | str | None = None, limit: int | None = None) -> list:
        table = self._table(collection)
        where, params = [], []
        for key, value in (filter or {}).items():
            if key in ("_id", "seq", "prev_hash", "entry_hash"):
                where.append(f'"{key}" = ?')
                params.append(int(value) if key in ("_id", "seq") else value)
            else:
                where.append("json_extract(\"doc\", ?) = ?")
                params.append("$." + key)
                params.append(value)
        order = ""
        norm_sort = self._normalize_sort(sort)
        if norm_sort:
            parts = []
            for field, desc in norm_sort:
                if field in ("_id", "seq", "prev_hash", "entry_hash"):
                    parts.append(f'"{field}" {"DESC" if desc else "ASC"}')
                else:
                    parts.append(
                        'json_extract("doc", \'$.' + field + f"') {'DESC' if desc else 'ASC'}"
                    )
            order = " ORDER BY " + ", ".join(parts)
        sql = f'SELECT "_id","doc" FROM "{table}"'
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += order
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        rows = []
        for row_id, doc_text in self._conn.execute(sql, params):
            doc = json.loads(doc_text)
            rows.append({"_id": str(row_id), **doc})
        return rows

    @staticmethod
    def _normalize_sort(sort):
        if sort is None:
            return []
        if isinstance(sort, str):
            sort = [sort]
        out = []
        for item in sort:
            if isinstance(item, str):
                desc = item.startswith("-")
                out.append((item[1:] if desc else item, desc))
            else:
                field, desc = item
                out.append((field, bool(desc)))
        return out

    def count(self, collection: str) -> int:
        table = self._table(collection)
        return self._conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]

    def close(self) -> None:
        self._conn.close()


class Journal:
    """The mind's memory: every self-edit recorded, hash-chained, sequenced."""

    def __init__(self, backend: JournalBackend):
        self.backend = backend
        self._seq = 0
        self._last_hash = GENESIS_PREV
        self._edit_n = 0
        # Catch up the logical clock and edit counter from existing rows.
        for coll in COLLECTIONS:
            for row in backend.find(coll):
                seq = row.get("seq") or 0
                if seq > self._seq:
                    self._seq = seq
                    self._last_hash = row.get("entry_hash") or self._last_hash
        for row in backend.find("self_edits"):
            eid = row.get("edit_id", "")
            if eid.startswith("se_"):
                try:
                    self._edit_n = max(self._edit_n, int(eid[3:]))
                except ValueError:
                    pass

    def close(self) -> None:
        self.backend.close()

    # -- logical clock ----------------------------------------------------
    def next_seq(self) -> int:
        """Allocate the next logical-clock tick (monotonic ints)."""
        self._seq += 1
        return self._seq

    def peek_seq(self) -> int:
        """Next seq without allocating (lets callers pre-compute hashes)."""
        return self._seq + 1

    def tip(self) -> tuple[int, str]:
        """(seq, entry_hash) of the journal tip."""
        return self._seq, self._last_hash

    # -- core append path: nothing bypasses seq/chain ----------------------
    def _append(self, collection: str, payload: dict) -> dict:
        payload = _jsonable(payload)
        hash_payload = {k: v for k, v in payload.items() if k not in MUTABLE_KEYS}
        seq = self.next_seq()
        entry_hash = _entry_hash(self._last_hash, seq, hash_payload)
        doc = {"ts": _utcnow(), **payload,
               "seq": seq, "prev_hash": self._last_hash, "entry_hash": entry_hash}
        row_id = self.backend.insert(collection, doc)
        self._last_hash = entry_hash
        return {"_id": row_id, "seq": seq, "entry_hash": entry_hash}

    def _chain_payload(self, row: dict) -> dict:
        return {k: v for k, v in row.items()
                if k not in _CHAIN_KEYS and k not in _NATIVE_KEYS
                and k not in MUTABLE_KEYS and k != "ts"}

    def verify_chain(self) -> bool:
        """Recompute every link. True iff the whole journal is intact."""
        rows = []
        for coll in COLLECTIONS:
            rows.extend(self.backend.find(coll))
        rows.sort(key=lambda r: (r.get("seq") or 0, str(r["_id"])))
        prev = GENESIS_PREV
        prev_seq = 0
        for row in rows:
            seq = row.get("seq") or 0
            if seq <= prev_seq:
                return False
            if row.get("prev_hash") != prev:
                return False
            if _entry_hash(prev, seq, self._chain_payload(row)) != row.get("entry_hash"):
                return False
            prev = row.get("entry_hash")
            prev_seq = seq
        return True

    # -- self_edits --------------------------------------------------------
    def record_self_edit(self, kind: str, intent: dict, reason: str,
                         gate_verdict: str, gate_report: dict,
                         operator: str | None = None) -> str:
        """Journal a proposed self-write. Returns the stable edit id."""
        self._edit_n += 1
        edit_id = f"se_{self._edit_n:04d}"
        self._append("self_edits", {
            "edit_id": edit_id,
            "kind": kind,
            "intent": intent,
            "reason": reason,
            "gate_verdict": gate_verdict,
            "gate_report": gate_report,
            "operator": operator,
            "result": "pending",
            "delta_id": None,
        })
        return edit_id

    def set_edit_result(self, edit_id: str, result: str, delta_id: str | None = None) -> None:
        rows = self.backend.find("self_edits", filter={"edit_id": edit_id}, limit=1)
        if not rows:
            raise KeyError(f"unknown edit_id {edit_id}")
        row = rows[0]
        row_id = row.pop("_id")
        row["result"] = result
        if delta_id is not None:
            row["delta_id"] = delta_id
        self.backend.update("self_edits", row_id, row)

    # -- deltas ------------------------------------------------------------
    def record_delta(self, delta_id: str, parent: str, blob: str, byte_len: int,
                     rank: int, alpha: int, target_modules: list, norm: float,
                     self_edit_id: str, source_seqs: list,
                     status: str = "active") -> str:
        res = self._append("deltas", {
            "id": delta_id,
            "parent": parent,
            "blob": blob,
            "bytes": byte_len,
            "rank": rank,
            "alpha": alpha,
            "target_modules": list(target_modules),
            "norm": norm,
            "self_edit_id": self_edit_id,
            "source_seqs": list(source_seqs),
            "status": status,
        })
        return res["_id"]

    def record_transformer_delta(self, delta_id: str, parent: str, blob: str,
                                 byte_len: int, norm: float,
                                 self_edit_id: str, source_seqs: list,
                                 targets: list,
                                 status: str = "active") -> str:
        """Journal a materialised transformer delta (SI-10).

        The demo-shaped :meth:`record_delta` cannot describe this
        kind honestly: a transformer delta has no single rank/alpha —
        rank/alpha live per target — so its journal row carries the
        SI-9a ``targets`` array verbatim (exact names, base hashes,
        per-target rank/alpha/norm) plus ``kind``. ``target_modules``
        is the target *names*, recorded so name-based journal reads
        see the same list the manifest record carries. No single
        rank/alpha is fabricated.
        """
        res = self._append("deltas", {
            "id": delta_id,
            "parent": parent,
            "blob": blob,
            "bytes": byte_len,
            "kind": "transformer-delta-v1",
            "targets": list(targets),
            "target_modules": [t.get("name") for t in targets
                               if isinstance(t, dict)],
            "norm": norm,
            "self_edit_id": self_edit_id,
            "source_seqs": list(source_seqs),
            "status": status,
        })
        return res["_id"]

    def set_delta_status(self, delta_id: str, status: str) -> None:
        rows = self.backend.find("deltas", filter={"id": delta_id}, limit=1)
        if not rows:
            raise KeyError(f"unknown delta_id {delta_id}")
        row = rows[0]
        row_id = row.pop("_id")
        row["status"] = status
        self.backend.update("deltas", row_id, row)

    # -- checkpoints -------------------------------------------------------
    def checkpoint(self, head: str, manifest_hash: str, note: str, **extra) -> dict:
        """Record where the head was. Returns {'_id','seq','entry_hash'}."""
        return self._append("checkpoints", {
            "head": head, "manifest": manifest_hash, "note": note, **extra,
        })

    # -- provenance / experiences ------------------------------------------
    def record_provenance(self, child: str, parents: list, relation: str,
                          note: str = "") -> str:
        res = self._append("provenance", {
            "child": child, "parents": list(parents),
            "relation": relation, "note": note,
        })
        return res["_id"]

    def record_experience(self, kind: str, record: dict, self_edit_id: str) -> str:
        res = self._append("experiences", {
            "kind": kind, "record": record, "self_edit_id": self_edit_id,
        })
        return res["_id"]

    # -- reads --------------------------------------------------------------
    def log(self, limit: int = 20) -> list:
        """Recent checkpoints, newest first. (`si log` pretty-prints these.)"""
        return self.backend.find("checkpoints", sort="-seq", limit=limit)

    def rollbacks(self, limit: int = 20) -> list:
        """Every rollback self-edit and why, newest first."""
        return self.backend.find("self_edits", filter={"kind": "rollback"},
                                 sort="-seq", limit=limit)
