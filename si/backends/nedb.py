"""Real-NEDB journal backend for the SI-0 runtime.

Implements the ``JournalBackend`` protocol (``insert`` / ``update`` /
``find`` / ``count`` / ``close``) against a REAL NEDB store over the pgwire
endpoint served by the ``nedbd-v2`` daemon (shipped inside the
``nedb-engine`` PyPI wheel -- no Rust toolchain needed; the daemon is a
prebuilt native binary).

This is the dogfood backend: the mind's journal lives in the actual NEDB
engine — schemaless collections, native ``_seq`` logical clock, ``_hash``
chain entries, real causal edges, time travel available over raw SQL.

Storage mapping (collection == NEDB collection == SQL table, schemaless):
  * each top-level field of ``doc`` is written as its own column, so scalar
    fields stay filterable/sortable with ordinary SQL WHERE / ORDER BY;
  * the ENTIRE original document is additionally stored verbatim as JSON in
    the reserved ``_journal_doc`` column, so ``find`` reconstructs the exact
    doc the runtime wrote (nested types preserved exactly);
  * ``_id``  -- document id (server-assigned unless ``doc`` carries ``_id``);
  * ``_seq``  -- NEDB's native global logical clock, one per write;
  * ``_hash`` -- NEDB's native hash-chain entry for the write.
  * NOTE: the SI journal's OWN logical clock (``seq``) and hash chain
    (``prev_hash`` / ``entry_hash``) are stored as ordinary top-level fields
    and are what ``Journal.verify_chain`` checks. NEDB's native ``_seq`` /
    ``_hash`` are transport metadata; the journal layer ignores them for
    integrity (see ``si.journal._chain_payload``).

Process lifecycle:
  * default (``autostart=True``): the backend first tries to connect to an
    already-running endpoint on the bundle-derived ports; if the port
    refuses, it spawns ``nedbd-v2`` itself (binary located via
    NEDB_NEDBD_BIN env, PATH, or the installed ``nedb`` package), waits for
    the pgwire port, creates the database via the daemon's HTTP API if
    needed, and terminates the child on ``close()``.
  * ``autostart=False``: connect to an externally managed server, and fail
    fast with a clear error if it is unreachable. ``close()`` never kills a
    server it did not start.

Ports: derived deterministically from the database name
(``55433 + h % 1000`` / ``17070 + h % 1000``) so two bundles never collide.
Data dir defaults to ``<bundle>/journal/nedb-data`` -- the bundle stays
self-contained; backup = copy the bundle directory with the daemon stopped.

Operational caveats (proven during bring-up):
  1. Simple-query protocol only -- NEDB's pgwire implements PQexec, not
     Parse/Bind/Execute. ``psycopg2`` works; ``psycopg3`` / prepared
     statements will fail. Pin ``psycopg2-binary``.
  2. Database must be created first (``POST /v1/databases``); autostart mode
     does this automatically.
  3. NEDB's native ``_seq`` is global per database and may have gaps --
     monotonic, not dense. The SI journal does not rely on it.
  4. Filters are scalar equality only; nested-field filtering is not
     translated. Doc keys must match ``[A-Za-z_][A-Za-z0-9_]*``.
  5. pgwire is cleartext -- keep on loopback.
  6. Pinned against wheel ``nedb-engine==12.15.16``.

Requires: ``pip install -r requirements-nedb.txt``.
"""

import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import time
import urllib.request

try:
    import psycopg2
    from psycopg2.extras import Json as _PgJson
    from psycopg2 import OperationalError as _PgOperationalError
    from psycopg2 import errors as _pg_errors
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "si.backends.nedb requires psycopg2-binary and nedb-engine: "
        "pip install -r requirements-nedb.txt"
    ) from _e

#: Keys reserved by NEDB itself; a doc may not use these as top-level keys
#: (except _id, which is honored as the document id).
_ENVELOPE_COL = "_journal_doc"
_RESERVED = {"_seq", "_hash", "_coll", "_valid_from", "_valid_to",
             "_journal_doc"}

#: Native keys merged into find() results; the SI journal layer excludes
#: these from its own hash-chain payload (see si.journal._NATIVE_KEYS).
_NATIVE_KEYS = frozenset({"_id", "_seq", "_hash"})

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote(ident):
    if not isinstance(ident, str) or not _IDENT.match(ident):
        raise ValueError(
            "not a safe SQL identifier: %r (use [A-Za-z_][A-Za-z0-9_]*)" % (ident,))
    return '"%s"' % ident


def _adapt(value):
    """Adapt a Python value to a psycopg2 parameter NEDB's pgwire parses."""
    if isinstance(value, (dict, list)):
        return _PgJson(value)  # sent as a JSON literal; stored as JSON text
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int, float)) or value is None:
        return value
    raise TypeError(
        "unsupported doc value type %s (use str/int/float/bool/None/dict/list)"
        % type(value).__name__)


def _dbname_for(journal_dir: str) -> str:
    bundle_name = os.path.basename(os.path.dirname(os.path.abspath(journal_dir)))
    safe = re.sub(r"[^A-Za-z0-9_]", "_", bundle_name) or "journal"
    return "si_" + safe


def _ports_for(dbname: str) -> tuple[int, int]:
    h = int(hashlib.sha256(dbname.encode("utf-8")).hexdigest(), 16)
    return 55433 + (h % 1000), 17070 + (h % 1000)


class NedbJournalBackend:
    """JournalBackend over a real NEDB store via pgwire."""

    def __init__(self, journal_dir: str, dbname: str | None = None,
                 data_dir: str | None = None, host: str = "127.0.0.1",
                 http_port: int | None = None, pg_port: int | None = None,
                 nedbd_bin: str | None = None, autostart: bool = True,
                 startup_timeout: float = 25.0):
        self.journal_dir = os.path.abspath(journal_dir)
        self.dbname = dbname or _dbname_for(self.journal_dir)
        default_pg, default_http = _ports_for(self.dbname)
        self.host = host
        self.http_port = default_http if http_port is None else http_port
        self.pg_port = default_pg if pg_port is None else pg_port
        self.data_dir = data_dir or os.path.join(self.journal_dir, "nedb-data")
        self._nedbd_bin = nedbd_bin
        self._autostart = autostart
        self._startup_timeout = startup_timeout
        self._child = None
        self._conn = None
        self._connect()

    # ------------------------------------------------------------------ setup

    def _dsn(self):
        return dict(host=self.host, port=self.pg_port, dbname=self.dbname,
                    user="nedb", connect_timeout=5)

    def _connect(self):
        try:
            self._conn = psycopg2.connect(**self._dsn())
            self._conn.autocommit = True
            return
        except _pg_errors.InvalidCatalogName as e:
            # Server is up, but this database is not open on it.
            if not self._autostart:
                raise RuntimeError(
                    "database %r is not open on the NEDB server at %s:%d "
                    "(%s). Create it first (POST /v1/databases on the "
                    "daemon's HTTP port), or use autostart=True."
                    % (self.dbname, self.host, self.pg_port, e)) from e
            self._ensure_database()
        except _PgOperationalError as e:
            if not self._autostart:
                raise RuntimeError(
                    "NEDB pgwire endpoint not reachable at %s:%d (%s). "
                    "Start it yourself, e.g. `nedbd-v2 --data <dir> "
                    "--pg-port %d`, or construct NedbJournalBackend with "
                    "autostart=True." % (self.host, self.pg_port, e,
                                         self.pg_port)) from e
            # autostart path: spawn the daemon, create the db, connect
            self._spawn_daemon()
            self._ensure_database()
        try:
            self._conn = psycopg2.connect(**self._dsn())
            self._conn.autocommit = True
        except _PgOperationalError as e:
            raise RuntimeError(
                "NEDB server prepared but connect to %s:%d still fails: %s"
                % (self.host, self.pg_port, e)) from e

    def _find_binary(self):
        for cand in (self._nedbd_bin, os.environ.get("NEDB_NEDBD_BIN"),
                     shutil.which("nedbd-v2")):
            if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand
        try:
            import nedb  # the nedb-engine wheel ships the daemon binary
            cand = os.path.join(os.path.dirname(nedb.__file__), "nedbd-v2")
            if os.path.isfile(cand):
                return cand
        except ImportError:
            pass
        raise RuntimeError(
            "cannot autostart NEDB: no nedbd-v2 binary found. Install the "
            "real engine (`pip install -r requirements-nedb.txt`), put "
            "nedbd-v2 on PATH, or pass nedbd_bin=/ set NEDB_NEDBD_BIN.")

    def _wait_port(self):
        deadline = time.time() + self._startup_timeout
        while time.time() < deadline:
            try:
                with socket.create_connection((self.host, self.pg_port),
                                              timeout=1):
                    return
            except OSError:
                if self._child.poll() is not None:
                    err = (self._child.stderr.read() or b"").decode(
                        "utf-8", "replace")[-2000:]
                    raise RuntimeError(
                        "nedbd-v2 exited during startup. stderr tail:\n%s"
                        % err)
                time.sleep(0.25)
        raise RuntimeError(
            "nedbd-v2 did not open pgwire port %d within %.0fs"
            % (self.pg_port, self._startup_timeout))

    def _ensure_database(self):
        base = "http://%s:%d/v1/databases" % (self.host, self.http_port)
        try:
            with urllib.request.urlopen(base, timeout=5) as r:
                existing = {d["name"] for d in
                            json.load(r).get("databases", [])}
        except Exception as e:
            raise RuntimeError(
                "spawned nedbd-v2 but its HTTP API is not answering on %d "
                "(needed to create database %r): %s"
                % (self.http_port, self.dbname, e)) from e
        if self.dbname not in existing:
            req = urllib.request.Request(
                base, data=json.dumps({"name": self.dbname}).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=10) as r:
                if r.status not in (200, 201):
                    raise RuntimeError(
                        "failed to create NEDB database %r: HTTP %d"
                        % (self.dbname, r.status))

    def _spawn_daemon(self):
        binary = self._find_binary()
        os.makedirs(self.data_dir, exist_ok=True)
        cmd = [binary, self.data_dir,
               "-p", str(self.http_port),
               "--pg-port", str(self.pg_port),
               "-H", self.host]
        try:
            self._child = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except OSError as e:
            raise RuntimeError(
                "failed to launch %s: %s" % (binary, e)) from e
        self._wait_port()

    # ------------------------------------------------------------------ cursor

    def _cur(self):
        if self._conn is None or self._conn.closed:
            self._connect()
        return self._conn.cursor()

    # ---------------------------------------------------------------- protocol

    def _clean(self, doc: dict) -> dict:
        """Strip NEDB-native keys; validate the rest."""
        if not isinstance(doc, dict):
            raise TypeError("doc must be a dict, got %s" % type(doc).__name__)
        clean = {k: v for k, v in doc.items() if k not in _NATIVE_KEYS}
        for key in clean:
            if key in _RESERVED:
                raise ValueError(
                    "doc key %r is reserved by the NEDB backend" % (key,))
        return clean

    def insert(self, collection, doc):
        """Insert a schemaless doc; return its document id (str)."""
        clean = self._clean(doc)
        table = _quote(collection)
        cols, params = [], []
        doc_id = None
        for key, value in clean.items():
            if key == "_id":
                doc_id = str(value)
                continue
            cols.append(_quote(key))
            params.append(_adapt(value))
        # verbatim envelope: exact doc reconstruction on find()
        cols.append(_quote(_ENVELOPE_COL))
        params.append(json.dumps(clean, sort_keys=True, default=str))
        if doc_id is not None:
            cols.insert(0, _quote("_id"))
            params.insert(0, doc_id)
        placeholders = ", ".join(["%s"] * len(cols))
        sql = "INSERT INTO %s (%s) VALUES (%s) RETURNING _id" % (
            table, ", ".join(cols), placeholders)
        cur = self._cur()
        try:
            cur.execute(sql, params)
            row = cur.fetchone()
        except Exception as e:
            self._conn.rollback()
            raise RuntimeError(
                "NEDB insert into %r failed: %s" % (collection, e)) from e
        if not row:
            raise RuntimeError("insert reported success but returned no _id")
        return str(row[0])

    def update(self, collection, row_id, doc):
        """Replace the stored doc for row_id (operational-state transitions).

        Implemented as a real SQL UPDATE (proven against the endpoint);
        the verbatim envelope is regenerated so find() keeps reconstructing
        the exact doc.
        """
        clean = self._clean(doc)
        table = _quote(collection)
        sets, params = [], []
        for key, value in clean.items():
            if key == "_id":
                continue
            sets.append("%s = %%s" % _quote(key))
            params.append(_adapt(value))
        sets.append("%s = %%s" % _quote(_ENVELOPE_COL))
        params.append(json.dumps(clean, sort_keys=True, default=str))
        params.append(str(row_id))
        sql = "UPDATE %s SET %s WHERE _id = %%s" % (table, ", ".join(sets))
        cur = self._cur()
        try:
            cur.execute(sql, params)
        except Exception as e:
            self._conn.rollback()
            raise RuntimeError(
                "NEDB update on %r failed: %s" % (collection, e)) from e
        if cur.rowcount == 0:
            raise KeyError("no row %r in collection %r" % (row_id, collection))

    def find(self, collection, filter=None, sort=None, limit=None):
        """Return matching docs, oldest write first by default.

        filter: dict of top-level scalar equality matches (e.g.
                {"kind": "rollback"}). Non-scalar filter values raise
                ValueError -- NEDB's pgwire compares them as JSON text.
        sort:   None (chronological by native _seq), a field name, or
                "-field" for descending. The SI journal passes "-seq"
                (its own logical clock, stored as a regular column).
        limit:  max rows.
        Each returned dict is the exact doc passed to insert()/update(),
        plus the native ``_id``, ``_seq`` (logical clock) and ``_hash``
        (chain entry). The SI journal layer ignores the native keys for
        integrity (see si.journal._NATIVE_KEYS).
        """
        table = _quote(collection)
        where, params = [], []
        for key, value in (filter or {}).items():
            if isinstance(value, (dict, list)):
                raise ValueError(
                    "filter value for %r must be scalar, got %s"
                    % (key, type(value).__name__))
            where.append("%s = %%s" % _quote(key))
            params.append(_adapt(value))
        order = "ORDER BY _seq ASC"
        if sort is not None:
            desc = sort.startswith("-")
            field = sort[1:] if desc else sort
            _quote(field)  # validates
            order = "ORDER BY %s %s" % (_quote(field),
                                       "DESC" if desc else "ASC")
        sql = "SELECT _id, _seq, _hash, %s FROM %s" % (
            _quote(_ENVELOPE_COL), table)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " " + order
        if limit is not None:
            sql += " LIMIT %d" % int(limit)
        cur = self._cur()
        try:
            cur.execute(sql, params)
            rows = cur.fetchall()
        except Exception as e:
            self._conn.rollback()
            raise RuntimeError(
                "NEDB find on %r failed: %s" % (collection, e)) from e
        out = []
        for _id, _seq, _hash, envelope in rows:
            if envelope is None:
                # row written outside this backend: best-effort passthrough
                doc = {}
            else:
                doc = json.loads(envelope) if isinstance(envelope, str) \
                    else envelope
            doc["_id"] = str(_id)
            doc["_seq"] = int(_seq)
            doc["_hash"] = str(_hash)
            out.append(doc)
        return out

    def count(self, collection):
        cur = self._cur()
        try:
            cur.execute("SELECT COUNT(*) FROM %s" % _quote(collection))
            return int(cur.fetchone()[0])
        except Exception as e:
            self._conn.rollback()
            raise RuntimeError(
                "NEDB count on %r failed: %s" % (collection, e)) from e

    def close(self):
        """Close the connection; stop the daemon only if we started it."""
        try:
            if self._conn is not None and not self._conn.closed:
                self._conn.close()
        finally:
            self._conn = None
        child, self._child = self._child, None
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
