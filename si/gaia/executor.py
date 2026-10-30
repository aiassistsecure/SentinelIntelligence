"""Gaia B-2 executor: live execution of gate-ADMITted tool intents
(SI-41 RFC §15 B-2, §4.3, §4.4, §9.2, §9.3, §11; slice SI-43).

B-1 (SI-42) built the tool intent, the registry, and the gate's
tool branch in shadow mode: nothing admitted, nothing executed.
B-2 is where execution becomes real — **live but miniature-scoped**,
exactly as RFC §15 B-2 bounds it:

- **Live scope.** Only the ``read`` / ``write`` class tools of
  RFC §8.1/§8.2 — ``fs_list``, ``fs_read``, ``fs_search``,
  ``fs_write``, ``fs_edit``, ``fs_mkdir``, ``journal_query``,
  ``bundle_status``, ``probe_run`` (the registry's
  :data:`~si.gaia.registry.LIVE_TOOLS_B2` set) — and only on
  **miniature** bundles (the SI-13 probe line's bundles, the only
  bundles whose loop path B-2 prices live, by the SI-17 precedent
  of pricing paths separately). Demo bundles keep B-1 shadow
  behaviour; their gate context is never live, so their tool
  intents ESCALATE/REJECT with the shadow label exactly as in B-1.
- **Execution tools stay shadow.** ``shell_exec`` / ``code_exec`` /
  ``test_run`` are B-3 (RFC §15): on a live context they ESCALATE
  with a not-evaluated label naming B-3, and this executor refuses
  them even if handed a journaled ADMIT for one (defence in depth —
  the gate is the authority, the executor re-checks).
- **External stays held.** An intent whose effective class prices
  to ``external`` (a path or cwd resolving outside the work root,
  symlinks resolved first, RFC §7.2/§9.1) ESCALATEs in B-2 even
  when the bundle's class policy would admit ``external``: B-2's
  live scope is work-root-contained execution. That is a
  slice-scoped hold, stated here and in the slice doc — not a
  permanent cage (RFC §7.3's dial remains policy's; a later slice
  may lift the hold on its own evidence).

The executor's discipline (RFC §9.2, the journaled-args rule): it
executes **the journaled intent** — the self-edit row the JOURNAL
step wrote — never an in-memory copy from the mind. Before running
anything it re-reads that row and refuses (journaling the refusal)
when: the row's gate verdict is not ADMIT; the tool is unknown or
outside the B-2 live set; the bundle is not miniature; the args no
longer validate against the registry schema; or the intent's
effective class has **changed** since admission (the TOCTOU rule —
a path whose classification moved between GATE and EXECUTE is
refused and must be re-proposed, never executed under a stale
price).

Every execution produces a §4.4 execution record with **measured**
fields only — status, result hash/bytes over the exact result the
mind will perceive, wall time from a real clock, post-state
observed after the act — journaled as a ``tool_result`` experience
via the journal's existing methods in the existing collections
(RFC §4.6: no new collection, no schema change). VERIFY for a tool
cycle is RFC §11.1: result conformance against the tool's result
schema plus post-state observation. There is **no rollback of
world effects** (§11.2): compensable writes journal their
pre-image so a *new, ordinary, gated* tool intent can compensate;
:func:`compensation_intent` builds that intent where the v1 tool
surface can honestly carry it, and says so where it cannot.

Location note: RFC §4.3 named ``si/tools.py`` as the B-slices'
plan; Mark's 2026-10-29 directive placed Gaia's code in the
``si/gaia/`` subpackage (registry, gate policy, and shadow
journaling already live here from B-1), so the executor lives
here too. The deviation is recorded in the SI-43 slice doc.
"""
from __future__ import annotations

import base64
import hashlib
import os
import time

from si.bundle import canonical_json, sha256_hex
from si.gaia.policy import resolve_tool_policy
from si.gaia.registry import (
    LIVE_TOOLS_B2,
    get_tool,
    validate_args,
    validate_result,
)
from si.gate import Gate, Intent

#: The honest execution label carried by a live B-2 execution
#: record: a real execution happened, by this executor, in scope.
LIVE_EXECUTION = "live — executed by the SI-43/B-2 executor (miniature-scoped)"

#: fs_list's RFC §8.1 entry cap.
_FS_LIST_CAP = 512
#: fs_search's default match cap when the intent declares none.
_FS_SEARCH_DEFAULT_CAP = 50
#: Excerpt length cap for fs_search matches.
_EXCERPT_CAP = 200


class ToolExecutionError(Exception):
    """A tool ran (or tried to) and failed honestly: the failure is
    journaled as a failed execution record, never raised past the
    executor as an unrecorded exception."""


def _sha256_ref(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _is_within(path: str, root: str) -> bool:
    return os.path.commonpath([path, root]) == root


def _resolve(work_root: str, raw: str) -> str:
    joined = raw if os.path.isabs(raw) else os.path.join(work_root, raw)
    return os.path.realpath(joined)


# -- journaling writers ------------------------------------------------------

def record_execution_outcome(journal, intent, edit_id: str,
                             execution: dict) -> dict:
    """Journal one live execution record as a ``tool_result``
    experience (the §4.5 provenance-tagged observation shape, via
    the journal's existing ``record_experience`` — RFC §4.6)."""
    journal.record_experience("tool_result", {
        "kind": "tool_result",
        "source": f"tool:{getattr(intent, 'tool', '')}",
        "intent_id": edit_id,
        "tool": getattr(intent, "tool", ""),
        "gate_verdict": "ADMIT",
        "effective_class": execution.get("class"),
        "execution": execution,
    }, edit_id)
    return execution


def record_non_execution_outcome(journal, intent, edit_id: str,
                                 verdict: str, label: str,
                                 effective_class=None) -> dict:
    """Journal a live-scope non-execution (REJECT / ESCALATE on a
    live context): the §4.4-shaped record with the gate report's
    honest label and **no measured fields** — nothing ran, so no
    exit code, hash, or wall time exists to record (the B-1 shadow
    record's discipline, kept)."""
    execution = {
        "tool": getattr(intent, "tool", ""),
        "intent_id": edit_id,
        "class": effective_class,
        "status": label,
        "execution": label,
        "verdict": verdict,
    }
    journal.record_experience("tool_result", {
        "kind": "tool_result",
        "source": f"tool:{getattr(intent, 'tool', '')}",
        "intent_id": edit_id,
        "tool": getattr(intent, "tool", ""),
        "gate_verdict": verdict,
        "effective_class": effective_class,
        "execution": execution,
    }, edit_id)
    return execution


# -- the executor --------------------------------------------------------------

class Executor:
    """Runs gate-ADMITted, journaled tool intents (RFC §4.3).

    Constructed by the runtime (the loop's miniature branch, or
    :func:`execute_tool_cycle` / :func:`run_tool_intent`) — never
    by a mind. ``manifest`` is the bundle's verified manifest (the
    miniature-scope check and ``bundle_status`` read it; handlers
    never call ``load_bundle`` mid-cycle, because the journal tip
    has legitimately moved past the manifest anchor by EXECUTE
    time and a re-load would fail closed for the wrong reason).
    """

    def __init__(self, bundle_dir, manifest, journal, policy: dict,
                 work_root: str):
        self.bundle_dir = os.path.abspath(bundle_dir) if bundle_dir else None
        self.manifest = manifest if isinstance(manifest, dict) else {}
        self.journal = journal
        self.policy = dict(policy or {})
        self.tool_policy = resolve_tool_policy(self.policy)
        self.work_root = os.path.realpath(work_root)
        self._gate = Gate(self.policy)

    # -- public: execute one journaled intent ------------------------------
    def execute_journaled(self, edit_id: str,
                          expected_effective_class: str | None = None) -> dict:
        """Execute the journaled intent ``edit_id``; journal and
        return its §4.4 execution record.

        Refusals are executions that did not happen, journaled as
        such (status ``refused``) — a caller can never mistake a
        refusal for a run, and the journal always answers why.
        """
        started = time.perf_counter()
        rows = self.journal.backend.find(
            "self_edits", filter={"edit_id": edit_id}, limit=1)
        if not rows:
            # Nothing journaled: there is nothing honest to attach
            # an outcome to, and inventing one would be fabrication.
            raise KeyError(f"unknown edit_id {edit_id}: no journaled "
                           "intent exists to execute (RFC §9.2)")
        row = rows[0]
        intent_dict = row.get("intent") or {}
        intent = Intent(
            kind=intent_dict.get("kind", ""),
            reason=intent_dict.get("reason", ""),
            parents=list(intent_dict.get("parents") or []),
            source_seqs=list(intent_dict.get("source_seqs") or []),
            tool=intent_dict.get("tool", ""),
            args=dict(intent_dict.get("args") or {}))
        spec = get_tool(intent.tool)

        def refuse(detail: str, effective=None) -> dict:
            record = {
                "tool": intent.tool,
                "intent_id": edit_id,
                "class": effective,
                "reversibility": spec.reversibility if spec else None,
                "status": "refused",
                "exit_code": None,
                "result": None,
                "result_hash": None,
                "result_bytes": 0,
                "truncated": False,
                "result_blob": None,
                "wall_time_seconds": time.perf_counter() - started,
                "post_state": None,
                "pre_image": None,
                "verify": {"result_conformance": "not-evaluated",
                           "post_state": "not-evaluated",
                           "detail": "execution refused before the "
                                     "tool ran; nothing was measured"},
                "execution": LIVE_EXECUTION,
                "detail": detail,
            }
            record_execution_outcome(self.journal, intent, edit_id, record)
            self.journal.set_edit_result(edit_id, "refused")
            return record

        # The journaled-args rule (RFC §9.2), step by step.
        if row.get("gate_verdict") != "ADMIT":
            return refuse(
                f"gate verdict on the journaled intent is "
                f"{row.get('gate_verdict')!r}, not ADMIT: the executor "
                "runs only gate-ADMITted intents")
        if intent.kind != "tool" or spec is None:
            return refuse(
                f"journaled intent is not a registered tool intent "
                f"(kind={intent.kind!r}, tool={intent.tool!r})")
        if spec.name not in LIVE_TOOLS_B2:
            return refuse(
                f"tool {spec.name!r} is outside the SI-43/B-2 live "
                "set: execution tools (shell_exec / code_exec / "
                "test_run) land in B-3 (RFC §15) and do not execute "
                "in this slice, whatever a journaled record claims")
        from si.probes import is_miniature_probe_bundle
        if not is_miniature_probe_bundle(self.manifest):
            return refuse(
                "live execution in B-2 is miniature-scoped (RFC §15 "
                "B-2): this bundle's base is not the miniature kind, "
                "so its tool intents keep shadow behaviour")
        violations = validate_args(spec, intent.args)
        if violations:
            return refuse(
                "journaled args no longer validate against the "
                "registry schema (RFC §9.2 re-validation): "
                + "; ".join(violations))
        from si.gaia.context import ToolContext
        info = self._gate.classify_tool_intent(
            intent, ToolContext(tool_work_root=self.work_root,
                                tool_live_execution=True))
        effective = info.get("effective_class")
        if expected_effective_class is not None \
                and effective != expected_effective_class:
            return refuse(
                f"effective class changed between GATE and EXECUTE: "
                f"admitted as {expected_effective_class!r}, re-priced "
                f"as {effective!r} — the TOCTOU rule (RFC §9.2) "
                "refuses execution under a stale price; re-propose "
                "the intent", effective=effective)
        if effective not in ("read", "write"):
            return refuse(
                f"effective class {effective!r} is outside the B-2 "
                "live scope (read/write, work-root-contained): "
                "refused at execution", effective=effective)
        for key, target in (info.get("resolved_paths") or {}).items():
            if not _is_within(target, self.work_root):
                return refuse(
                    f"arg {key!r} resolves to {target}, outside the "
                    f"work root {self.work_root}: refused at "
                    "execution", effective=effective)

        # Run the tool. A handler failure is a failed execution —
        # a result, not an exception (RFC §11.1.3).
        pre_image = None
        try:
            handler = getattr(self, f"_do_{spec.name}")
            result, post_state, pre_image, post_ok, post_detail = \
                handler(intent.args)
            status = "completed"
            detail = post_detail
        except ToolExecutionError as err:
            result, post_state, post_ok = None, None, False
            status = "failed"
            detail = f"{type(err).__name__}: {err}"
        except Exception as err:  # a handler bug is still a journaled failure
            result, post_state, post_ok = None, None, False
            status = "failed"
            detail = f"unexpected {type(err).__name__}: {err}"

        # VERIFY (RFC §11.1): result conformance + post-state.
        if status == "completed":
            result_violations = validate_result(spec, result)
            conformance = "failed" if result_violations else "passed"
            if result_violations:
                status = "failed"
                detail = ("result does not conform to the tool's "
                          "result schema (RFC §11.1): "
                          + "; ".join(result_violations))
            post_label = "passed" if post_ok else "failed"
            if not post_ok:
                status = "failed"
                detail = ("post-state observation does not match "
                          "the execution's claimed result "
                          "(RFC §11.1): " + detail)
        else:
            conformance, post_label = "not-evaluated", "not-evaluated"

        inline, truncated, blob_ref, overflow = (result, False, None, None)
        if isinstance(result, dict):
            inline, truncated, blob_ref, overflow = self._bound_result(result)
        raw_inline = canonical_json(inline) if inline is not None else b""
        target_path = None
        if isinstance(intent.args.get("path"), str):
            target_path = self._rel(
                _resolve(self.work_root, intent.args["path"]))
        record = {
            "tool": spec.name,
            "intent_id": edit_id,
            "class": effective,
            "target_path": target_path,
            "reversibility": spec.reversibility,
            "status": status,
            "exit_code": None,  # no process is spawned by any B-2 tool
            "result": inline,
            "result_hash": _sha256_ref(raw_inline) if inline is not None else None,
            "result_bytes": len(raw_inline),
            "truncated": truncated,
            "result_blob": blob_ref,
            "wall_time_seconds": time.perf_counter() - started,
            "post_state": post_state,
            "pre_image": pre_image,
            "verify": {"result_conformance": conformance,
                       "post_state": post_label,
                       "detail": detail},
            "execution": LIVE_EXECUTION,
            "detail": detail,
        }
        if overflow is not None:
            record["overflow_result_hash"] = overflow[0]
            record["overflow_result_bytes"] = overflow[1]
        record_execution_outcome(self.journal, intent, edit_id, record)
        self.journal.set_edit_result(
            edit_id, "executed" if status == "completed" else "failed")
        return record

    # -- result bounding + blob overflow (RFC §9.3) --------------------------
    def _store_blob(self, data: bytes) -> str:
        """Content-address ``data`` in the bundle's blob store.

        The store is the never-duplicate-bytes store the deltas use
        (``blobs/sha256/<hex>.safetensors`` — the store's existing
        naming for every blob, whatever its bytes): an identical
        payload already present is referenced, never rewritten.
        Returns the ``sha256:<hex>`` ref.
        """
        hexpart = sha256_hex(data)
        dest = os.path.join(self.bundle_dir, "blobs", "sha256",
                            hexpart + ".safetensors")
        if not os.path.isfile(dest):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            tmp = dest + ".tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dest)
        return "sha256:" + hexpart

    def _bound_result(self, result: dict):
        """Bound a result to the policy's inline cap (RFC §9.3).

        Returns (inline_result, truncated, blob_ref, overflow) where
        overflow is (full_hash, full_bytes) when the full result was
        blob-stored. Shrinking is per-shape and honest: list fields
        (``entries`` / ``matches`` / ``rows``) drop tail items and
        set the result's own ``truncated`` flag; ``fs_read``'s
        ``content`` is prefix-truncated the same way. The result
        hash/bytes on the execution record describe the *inline*
        result — the exact result PERCEIVE hands the mind (§4.4);
        the overflow fields bind the full one.
        """
        cap = int(self.tool_policy.get("tool_max_result_bytes") or 0)
        raw = canonical_json(result)
        if cap <= 0 or len(raw) <= cap:
            return result, False, None, None
        blob_ref = self._store_blob(raw) if self.bundle_dir else None
        overflow = (_sha256_ref(raw), len(raw))
        inline = dict(result)
        inline["truncated"] = True
        for key in ("entries", "matches", "rows"):
            if isinstance(inline.get(key), list):
                while inline[key] and len(canonical_json(inline)) > cap:
                    inline[key] = inline[key][:-1]
        if isinstance(inline.get("content"), str):
            content = inline["content"]
            while content and len(canonical_json(inline)) > cap:
                content = content[: max(0, len(content) // 2)]
                inline["content"] = content
        return inline, True, blob_ref, overflow

    # -- shared handler helpers ----------------------------------------------
    def _scoped(self, raw: str) -> str:
        target = _resolve(self.work_root, raw)
        if not _is_within(target, self.work_root):
            raise ToolExecutionError(
                f"path {raw!r} resolves to {target}, outside the "
                f"work root {self.work_root}")
        return target

    def _rel(self, target: str) -> str:
        return os.path.relpath(target, self.work_root)

    def _capture_pre_image(self, target: str) -> dict:
        """Journal-grade pre-image of a file about to be written
        (RFC §11.2): existence, hash, size — and the bytes
        themselves, inline when they fit the result cap (text when
        UTF-8-decodable, base64 otherwise), blob-stored when they
        do not, hash-only when no bundle store is available."""
        if not os.path.exists(target):
            return {"exists": False, "sha256": None, "bytes": 0,
                    "stored": "none — the path did not exist before "
                              "the write"}
        if os.path.isdir(target):
            raise ToolExecutionError(
                f"path {target} is a directory, not a file")
        from si.bundle import sha256_file
        digest = "sha256:" + sha256_file(target)
        size = os.path.getsize(target)
        cap = int(self.tool_policy.get("tool_max_result_bytes") or 0)
        if size <= cap:
            with open(target, "rb") as f:
                data = f.read()
            try:
                return {"exists": True, "sha256": digest, "bytes": size,
                        "stored": "inline", "encoding": "text",
                        "content": data.decode("utf-8")}
            except UnicodeDecodeError:
                return {"exists": True, "sha256": digest, "bytes": size,
                        "stored": "inline", "encoding": "base64",
                        "content": base64.b64encode(data).decode("ascii")}
        if self.bundle_dir:
            with open(target, "rb") as f:
                data = f.read()
            return {"exists": True, "sha256": digest, "bytes": size,
                    "stored": "blob", "blob": self._store_blob(data)}
        return {"exists": True, "sha256": digest, "bytes": size,
                "stored": "hash-only — no bundle blob store available "
                          "to hold the pre-image bytes"}

    def _verify_written(self, target: str, expect_sha256: str):
        from si.bundle import sha256_file
        if not os.path.isfile(target):
            return False, {"path": target, "exists": False}
        actual = "sha256:" + sha256_file(target)
        post = {"path": target, "exists": True,
                "size": os.path.getsize(target), "sha256": actual}
        return actual == expect_sha256, post

    # -- §8.1 filesystem tools -------------------------------------------------
    def _do_fs_list(self, args):
        target = self._scoped(args["path"])
        if not os.path.isdir(target):
            raise ToolExecutionError(
                f"path {args['path']!r} is not a directory in scope")
        names = sorted(os.listdir(target))
        entries = []
        for name in names[:_FS_LIST_CAP]:
            full = os.path.join(target, name)
            if os.path.isdir(full):
                kind, size = "dir", 0
            elif os.path.isfile(full):
                kind, size = "file", os.path.getsize(full)
            else:
                kind, size = "other", 0
            entries.append({"name": name, "kind": kind, "size": size})
        result = {"entries": entries, "truncated": len(names) > _FS_LIST_CAP}
        post = {"path": target, "exists": True, "kind": "dir",
                "entries_observed": len(names)}
        return result, post, None, True, \
            f"listed {len(entries)} of {len(names)} entries"

    def _do_fs_read(self, args):
        target = self._scoped(args["path"])
        if not os.path.isfile(target):
            raise ToolExecutionError(
                f"path {args['path']!r} is not a file in scope")
        from si.bundle import sha256_file
        size = os.path.getsize(target)
        cap = int(self.tool_policy.get("tool_max_result_bytes") or 0)
        limit = args.get("max_bytes") or size
        read_n = min(size, int(limit), cap if cap > 0 else size)
        with open(target, "rb") as f:
            data = f.read(read_n)
        truncated = size > len(data)
        try:
            content, encoding = data.decode("utf-8"), "text"
        except UnicodeDecodeError:
            content = base64.b64encode(data).decode("ascii")
            encoding = "base64"
        result = {"content": content, "encoding": encoding,
                  "truncated": truncated}
        post = {"path": target, "exists": True, "kind": "file",
                "size": size, "sha256": "sha256:" + sha256_file(target)}
        return result, post, None, True, \
            f"read {len(data)} of {size} bytes ({encoding})"

    def _do_fs_search(self, args):
        target = self._scoped(args["path"])
        pattern = args["pattern"]
        cap_matches = int(args.get("max_matches") or _FS_SEARCH_DEFAULT_CAP)
        files = []
        if os.path.isfile(target):
            files = [target]
        elif os.path.isdir(target):
            for dirpath, dirnames, filenames in os.walk(target):
                dirnames.sort()
                for name in sorted(filenames):
                    files.append(os.path.join(dirpath, name))
        else:
            raise ToolExecutionError(
                f"path {args['path']!r} does not exist in scope")
        matches, searched, truncated = [], 0, False
        for path in files:
            try:
                with open(path, "rb") as f:
                    text = f.read().decode("utf-8")
            except (UnicodeDecodeError, OSError):
                continue  # binary/unreadable files are skipped, honestly
            searched += 1
            for lineno, line in enumerate(text.splitlines(), start=1):
                if pattern in line:
                    if len(matches) >= cap_matches:
                        truncated = True
                        break
                    matches.append({
                        "path": self._rel(path), "line": lineno,
                        "excerpt": line.strip()[:_EXCERPT_CAP]})
            if truncated:
                break
        result = {"matches": matches, "truncated": truncated}
        post = {"path": target, "exists": True,
                "files_searched": searched,
                "matches_observed": len(matches)}
        return result, post, None, True, \
            f"{len(matches)} match(es) over {searched} text file(s)"

    def _do_fs_write(self, args):
        target = self._scoped(args["path"])
        pre_image = self._capture_pre_image(target)
        parent = os.path.dirname(target)
        if parent and not os.path.isdir(parent):
            raise ToolExecutionError(
                f"parent directory {parent} does not exist; "
                "fs_mkdir is the tool that creates directories")
        data = args["content"].encode("utf-8")
        tmp = target + ".si-tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, target)
        digest = _sha256_ref(data)
        result = {"path": self._rel(target), "bytes_written": len(data),
                  "sha256": digest}
        post_ok, post = self._verify_written(target, digest)
        return result, post, pre_image, post_ok, \
            f"wrote {len(data)} bytes to {self._rel(target)}"

    def _do_fs_edit(self, args):
        target = self._scoped(args["path"])
        if not os.path.isfile(target):
            raise ToolExecutionError(
                f"path {args['path']!r} is not a file in scope")
        pre_image = self._capture_pre_image(target)
        with open(target, "rb") as f:
            raw = f.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise ToolExecutionError(
                "fs_edit edits text files; the target is not "
                "UTF-8-decodable")
        count = text.count(args["old"])
        if count == 0:
            raise ToolExecutionError(
                "the `old` string is absent from the file; "
                "fs_edit fails rather than guessing (RFC §8.1)")
        if count > 1:
            raise ToolExecutionError(
                f"the `old` string occurs {count} times; fs_edit "
                "requires an unambiguous match (RFC §8.1)")
        data = text.replace(args["old"], args["new"], 1).encode("utf-8")
        tmp = target + ".si-tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, target)
        digest = _sha256_ref(data)
        result = {"path": self._rel(target), "replacements": 1,
                  "sha256": digest}
        post_ok, post = self._verify_written(target, digest)
        return result, post, pre_image, post_ok, \
            f"replaced 1 occurrence in {self._rel(target)}"

    def _do_fs_mkdir(self, args):
        target = self._scoped(args["path"])
        if os.path.isdir(target):
            pre_image = {"exists": True, "sha256": None, "bytes": 0,
                         "stored": "none — directory already existed"}
            result = {"path": self._rel(target), "created": False}
            post = {"path": target, "exists": True, "kind": "dir"}
            return result, post, pre_image, True, \
                "directory already existed; nothing created"
        if os.path.exists(target):
            raise ToolExecutionError(
                f"path {args['path']!r} exists and is not a directory")
        parent = os.path.dirname(target)
        if parent and not os.path.isdir(parent):
            raise ToolExecutionError(
                f"parent directory {parent} does not exist; fs_mkdir "
                "creates one directory level")
        pre_image = {"exists": False, "sha256": None, "bytes": 0,
                     "stored": "none — the path did not exist before"}
        os.mkdir(target)
        result = {"path": self._rel(target), "created": True}
        post = {"path": target, "exists": os.path.isdir(target),
                "kind": "dir"}
        return result, post, pre_image, bool(post["exists"]), \
            f"created directory {self._rel(target)}"

    # -- §8.2 house tools -------------------------------------------------------
    def _do_journal_query(self, args):
        collection = args["collection"]
        raw_filter = args.get("filter") or {}
        for key, value in raw_filter.items():
            if not isinstance(value, (str, int, float, bool)) \
                    and value is not None:
                raise ToolExecutionError(
                    f"journal_query filter value for {key!r} is not a "
                    "scalar; the journal backend's exact-match filter "
                    "cannot price it")
        limit = int(args.get("limit") or 50)
        rows = self.journal.backend.find(
            collection, filter=raw_filter or None, limit=limit + 1)
        truncated = len(rows) > limit
        rows = rows[:limit]
        result = {"rows": rows, "count": len(rows), "truncated": truncated}
        post = {"collection": collection, "rows_observed": len(rows)}
        return result, post, None, True, \
            f"{len(rows)} row(s) from the mind's own {collection}"

    def _do_bundle_status(self, args):
        manifest = self.manifest
        result = {
            "head": str(manifest.get("head", "")),
            "deltas": len(manifest.get("deltas") or []),
            "last_seq": int((manifest.get("journal") or {}).get(
                "last_seq", 0)),
            "policy_hash": str((manifest.get("constitution") or {}).get(
                "policy_hash", "")),
        }
        post = {"head": result["head"],
                "note": "derived from the verified manifest loaded "
                        "at cycle start; last_seq is the "
                        "manifest-anchored seq (RFC §8.2)"}
        return result, post, None, True, \
            "bundle status from the verified manifest; nothing invented"

    def _do_probe_run(self, args):
        suite = args["suite"]
        from si.probes import ProbeContext, is_miniature_probe_bundle
        if not is_miniature_probe_bundle(self.manifest):
            result = {"suite": suite, "label": "not-evaluated",
                      "measured": None,
                      "detail": "no probe harness exists for this "
                                "bundle's base (not the miniature "
                                "kind): the honest result is "
                                "not-evaluated (RFC §8.2)"}
            return result, {"suite": suite, "harness": "absent"}, \
                None, True, result["detail"]
        try:
            ctx = ProbeContext(self.bundle_dir, self.manifest,
                               policy=self.policy)
            ok, reason = ctx.probe_status()
            if not ok:
                result = {"suite": suite, "label": "not-evaluated",
                          "measured": None,
                          "detail": f"probe harness not evaluated: "
                                    f"{reason} (RFC §8.2)"}
                return result, {"suite": suite, "harness": "not-evaluated"}, \
                    None, True, result["detail"]
            import numpy as np
            if suite == "identity":
                tol = float(self.policy.get("identity_tol", 0.0))
                worst, worst_q = 0.0, ""
                probes = ctx.identity_probes()
                for qid, x, expected in probes:
                    err = float(np.max(np.abs(
                        np.asarray(ctx.forward_head(x), dtype=np.float64)
                        - np.asarray(expected, dtype=np.float64))))
                    if err > worst:
                        worst, worst_q = err, qid
                passed = worst <= tol
                result = {"suite": suite,
                          "label": "measured-pass" if passed
                                   else "measured-fail",
                          "measured": worst,
                          "detail": f"identity suite at the current "
                                    f"head: max drift {worst:.6g} vs "
                                    f"tol {tol:.6g} over {len(probes)} "
                                    f"probe(s); worst {worst_q}"}
            else:
                eps = float(self.policy.get("capability_eps", 0.0))
                # Regression is priced against the RECORDED seed
                # baselines (0.0 by construction — the suites'
                # expected logits are the base's own outputs),
                # exactly as _verify_probe_head's no-baseline arm;
                # the echo-accuracy arm below is measured live.
                baseline = {}
                targets = ctx.capability_targets()
                floor = float(ctx.echo_floor())
                worst_reg, correct = -1e18, 0
                suite_rows = ctx.capability_suite()
                for cid, x, expected in suite_rows:
                    logits = ctx.forward_head(x)
                    err = float(np.max(np.abs(
                        np.asarray(logits, dtype=np.float64)
                        - np.asarray(expected, dtype=np.float64))))
                    worst_reg = max(worst_reg,
                                    err - float(baseline.get(cid, 0.0)))
                    if int(np.argmax(np.asarray(logits))) == int(targets[cid]):
                        correct += 1
                accuracy = correct / len(suite_rows) if suite_rows else 0.0
                passed = worst_reg <= eps and accuracy >= floor
                result = {"suite": suite,
                          "label": "measured-pass" if passed
                                   else "measured-fail",
                          "measured": worst_reg,
                          "detail": f"capability suite at the current "
                                    f"head: max regression "
                                    f"{worst_reg:.6g} vs eps {eps:.6g}; "
                                    f"echo accuracy {accuracy:.4g} vs "
                                    f"floor {floor:.4g} over "
                                    f"{len(suite_rows)} eval(s)"}
            return result, {"suite": suite, "harness": "evaluated",
                            "label": result["label"]}, None, True, \
                result["detail"]
        except Exception as err:
            result = {"suite": suite, "label": "not-evaluated",
                      "measured": None,
                      "detail": f"probe harness raised "
                                f"{type(err).__name__}: {err}; reported "
                                "as not-evaluated, never as a pass "
                                "(RFC §8.2)"}
            return result, {"suite": suite, "harness": "not-evaluated"}, \
                None, True, result["detail"]


# -- compensation (RFC §11.2) ---------------------------------------------------

def compensation_intent(execution: dict, *, parents: list,
                        source_seqs: list, bundle_dir: str | None = None,
                        reason: str | None = None):
    """Build the compensating intent for a compensable execution,
    or explain why the v1 surface cannot carry one.

    Returns (intent_or_None, explanation). Compensation is a NEW
    action — proposed, gated, and journaled like any other tool
    intent — never a rewind (RFC §11.2). It exists exactly where
    it is honest:

    - ``fs_write`` / ``fs_edit`` that overwrote a UTF-8 text file
      whose pre-image bytes are recoverable (inline in the record,
      or blob-stored and readable): an ``fs_write`` of the
      pre-image back to the same path.
    - Everything else returns None with the reason: a created
      file's prior *absence* cannot be restored (the v1 registry
      has no delete tool, RFC §8.4), a binary pre-image cannot be
      carried by ``fs_write``'s text ``content``, a directory's
      creation has no inverse tool, and one-way/read actions have
      nothing to compensate.
    """
    tool = execution.get("tool")
    pre = execution.get("pre_image") or {}
    post = execution.get("post_state") or {}
    if tool not in ("fs_write", "fs_edit"):
        return None, (f"tool {tool!r} carries no compensable file "
                      "effect in B-2 (reads change nothing; mkdir's "
                      "inverse has no tool in the v1 registry)")
    if not pre.get("exists"):
        return None, ("the write created the file: its pre-image is "
                      "absence, and the v1 registry has no delete "
                      "tool (RFC §8.4) — compensation is not "
                      "executable through the tool surface")
    content = None
    if pre.get("stored") == "inline" and pre.get("encoding") == "text":
        content = pre.get("content")
    elif pre.get("stored") == "blob" and bundle_dir:
        hexpart = str(pre.get("blob", "")).split(":", 1)[-1]
        path = os.path.join(bundle_dir, "blobs", "sha256",
                            hexpart + ".safetensors")
        if os.path.isfile(path):
            with open(path, "rb") as f:
                data = f.read()
            if _sha256_ref(data) == pre.get("blob"):
                try:
                    content = data.decode("utf-8")
                except UnicodeDecodeError:
                    content = None
    if content is None:
        return None, ("the pre-image bytes are not recoverable as "
                      "UTF-8 text (stored "
                      f"{pre.get('stored')!r}, encoding "
                      f"{pre.get('encoding')!r}): fs_write's content "
                      "is text, so the surface cannot honestly "
                      "carry this compensation")
    # The compensating write targets the same work-root-relative
    # path the original write landed on; the record's post-state
    # holds the resolved path — the caller re-roots it through
    # their work root by passing the relative form recorded here.
    rel = execution.get("target_path")
    if not rel:
        return None, "the execution record carries no target path"
    intent = Intent(
        kind="tool", tool="fs_write",
        args={"path": rel, "content": content},
        reason=reason or (
            f"Compensate {execution.get('intent_id')}: restore the "
            f"journaled pre-image of {rel} (a new gated action, "
            "RFC §11.2 — not a rollback)"),
        parents=list(parents), source_seqs=list(source_seqs))
    return intent, (f"fs_write of the journaled pre-image back to "
                    f"{rel}")


# -- the gated cycle (gate -> journal -> execute) -------------------------------

def execute_tool_cycle(journal, bundle_dir, manifest, intent, gate,
                       ctx, executor: Executor | None = None) -> dict:
    """Gate one tool intent and, on ADMIT, execute it — the B-2
    form of :func:`si.gaia.shadow.journal_tool_intent`.

    Journals through the existing methods/collections, in the
    loop's order: gate first; the self-edit (with the full gate
    report) and provenance edges BEFORE any execution (RFC §4.2's
    journal-before-execute rule); then the executor runs the
    journaled intent (§9.2); REJECT/ESCALATE journal their outcome
    and halt. On a non-live context the outcome record is B-1's
    shadow record, byte-for-byte (the demo path is unchanged); on
    a live context it carries the gate report's live label with no
    fabricated measured fields.

    Returns {"verdict", "edit_id", "gate_report", "execution",
    "effective_class"} — ``execution`` is the §4.4 record on ADMIT,
    the shadow/non-execution record otherwise.
    """
    from si.gaia.shadow import record_shadow_outcome

    report = gate.evaluate(intent, None, ctx)
    info = gate.classify_tool_intent(intent, ctx) \
        if hasattr(gate, "classify_tool_intent") else {}
    effective = info.get("effective_class")
    edit_id = journal.record_self_edit(
        intent.kind, intent.to_dict(), intent.reason,
        report.verdict, report.to_dict())
    if intent.parents:
        journal.record_provenance(
            child=edit_id, parents=list(intent.parents),
            relation="learned_from", note=(intent.reason or "")[:200])
    if intent.source_seqs:
        journal.record_provenance(
            child=edit_id, parents=list(intent.source_seqs),
            relation="learned_from", note="source journal sequences")

    if report.verdict == "ADMIT":
        journal.set_edit_result(edit_id, "pending")
        runner = executor or Executor(
            bundle_dir, manifest, journal, gate.policy,
            info.get("work_root") or bundle_dir)
        execution = runner.execute_journaled(
            edit_id, expected_effective_class=effective)
        return {"verdict": report.verdict, "edit_id": edit_id,
                "gate_report": report.to_dict(), "execution": execution,
                "effective_class": effective}

    journal.set_edit_result(
        edit_id, "rejected" if report.verdict == "REJECT" else "escalated")
    from si.gate import TOOL_SHADOW_EXECUTION
    if report.execution == TOOL_SHADOW_EXECUTION:
        execution = record_shadow_outcome(
            journal, intent, edit_id, report.verdict,
            effective_class=effective)
    else:
        execution = record_non_execution_outcome(
            journal, intent, edit_id, report.verdict,
            report.execution or "not-evaluated", effective_class=effective)
    return {"verdict": report.verdict, "edit_id": edit_id,
            "gate_report": report.to_dict(), "execution": execution,
            "effective_class": effective}


def run_tool_intent(bundle_dir: str, intent, work_root: str | None = None,
                    task_seq: int | None = None, cycles_used: int = 0,
                    result_bytes_used: int = 0,
                    policy: dict | None = None) -> dict:
    """One complete gated tool cycle against a bundle on disk.

    The scripted-scenario driver for B-2 (the real task surface,
    ``si task``, is B-3): loads and verifies the bundle, gates the
    intent under the bundle's policy (or an operator-supplied
    policy dict, as the project's tests supply gate policies),
    executes on ADMIT when the bundle is miniature, journals
    everything, and re-anchors the manifest checkpoint so the
    bundle loads cleanly afterwards. Demo (non-miniature) bundles
    keep shadow behaviour — their context is never live.
    """
    from si.bundle import (
        _anchor_checkpoint,
        _write_manifest,
        load_bundle,
        load_policy,
    )
    from si.journal import open_journal
    from si.probes import is_miniature_probe_bundle

    bundle_dir = os.path.abspath(bundle_dir)
    manifest = load_bundle(bundle_dir)
    pol = policy if policy is not None else load_policy(bundle_dir)
    gate = Gate(pol)
    journal = open_journal(bundle_dir, manifest)
    try:
        if is_miniature_probe_bundle(manifest):
            from si.probes import ProbeContext
            ctx = ProbeContext(bundle_dir, manifest, policy=pol)
            ctx.tool_work_root = work_root
            ctx.tool_task_seq = task_seq
            ctx.tool_cycles_used = cycles_used
            ctx.tool_result_bytes_used = result_bytes_used
        else:
            from si.gaia.context import ToolContext
            ctx = ToolContext(tool_work_root=work_root,
                              bundle_dir=bundle_dir,
                              tool_task_seq=task_seq,
                              tool_cycles_used=cycles_used,
                              tool_result_bytes_used=result_bytes_used)
        outcome = execute_tool_cycle(journal, bundle_dir, manifest,
                                     intent, gate, ctx)
        updated = _anchor_checkpoint(
            journal, manifest, manifest["head"],
            note=f"tool cycle {outcome['edit_id']} "
                 f"{outcome['verdict'].lower()}")
        _write_manifest(bundle_dir, updated)
        outcome["head"] = updated["head"]
        return outcome
    finally:
        journal.close()
