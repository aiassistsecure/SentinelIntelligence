"""``si squash``: compress the oldest deltas on the active stack into one.

A long-lived mind accumulates one delta per cycle; the active stack grows
without bound and every forward pass pays for every delta. Squash rewrites
history's *representation* without changing what the head computes: the
oldest ``L - keep`` deltas (S) are replaced by a single delta whose update
is the truncated-SVD recompression of their combined update, while the
newest ``keep`` deltas (K) are left untouched.

Pipeline (fail closed at every step):

1. Resolve the active chain (``bundle._active_chain``) and split it into
   S (to squash) and K (to keep). ``L <= keep`` prints "nothing to squash"
   and returns ``{"squashed": False}`` — a no-op that involves no delta
   of any kind, so it returns before every refusal below.
2. SI-12: TRANSFORMER DELTAS FAIL CLOSED. If the resolved active chain
   contains any record with ``kind == "transformer-delta-v1"`` — in S,
   in K, or both (a mixed stack) — raise SquashError BEFORE any blob is
   written, any gate/journal activity, or any manifest change.
   Transformer squash (composition over named tensors) is NOT
   implemented; it is an explicit non-goal (RFC SI-9 §6.4), and a
   transformer delta must never be silently converted to a demo delta.
   The whole resolved chain is checked, not only S, because a squash
   also *involves* K: the kept tail is materialized for the tolerance
   verify and loaded as demo factors by the gate stack machinery
   (``loop._load_stack`` expects ``delta.B``/``delta.A`` and a
   record-level ``rank``/``alpha`` a transformer record does not carry)
   — reproduced pre-SI-12 behaviour (raw ``KeyError: 'rank'`` /
   ``SafetensorError``, plus an orphaned demo-format candidate blob
   written before the crash) is documented in
   ``docs/SI-12-squash-transformer-fail-closed.md``. Records that are
   NOT on the resolved active chain (rolled back, superseded history,
   beyond the head) are not involved in a squash and do not block one:
   ``_active_chain`` never loads, applies, or gates them.
3. SI-3 scope is single-tensor bases (the demo seed's one ``W``). A
   multi-tensor base raises SquashError: squashing across named tensors
   would need a per-target layout this format does not have yet, and we
   refuse to guess it.
4. The combined update of S is ``materialize_head(sub-head at S[-1]) -
   base``. The exporter's delta math is reused, never reimplemented.
5. Truncated SVD at rank r (``--rank``, else the max rank in S):
   B = U_r·√Σ_r, A = √Σ_r·V_rᵀ, stored with ``alpha = rank = r``
   (scale = 1 — see "The scale=1 choice" below). Blob layout keeps the
   legacy ``delta.B``/``delta.A`` convention so export/loop need no
   format changes.
6. TOLERANCE VERIFY, before anything is journaled or committed: the head
   is materialized with d_new substituted for S (an in-memory manifest
   copy — S marked ``superseded_by``, d_new appended, head unchanged)
   and the max abs diff against the pre-squash materialized head must be
   < tol. A violation raises SquashError and NOTHING is left behind: no
   blob, no journal entries, no manifest change. (The candidate blob is
   written to its content-addressed path before this step so the
   exporter can load it through the substituted manifest; on violation
   it is deleted again unless the identical blob already existed.)
7. GATE: a squash is a delta intent and takes the full five checks like
   any self-write. The gate's BundleContext sees the post-squash stack
   (squashed delta + kept tail); the capability baseline is measured on
   the pre-squash stack; ``stack_depth()`` reports the pre-candidate
   depth so the gate's ``+1`` is the true post-squash depth. Provenance
   ``source_seqs`` are the journal seqs of the squashed deltas' own
   records — the causal events this squash exists because of (the gate
   requires non-empty ``source_seqs``; inventing them would be lying).
8. REJECT/ESCALATE: journaled exactly the way the loop journals
   rejections (self-edit + verdict + report, result rejected/escalated,
   checkpoint + manifest re-anchor). Nothing else is written. The
   candidate blob stays on disk as evidence, mirroring the loop's
   "rejected blobs stay on disk as evidence" guarantee.
9. ADMIT: the squash is journaled (self-edit ADMIT, ``squashed_from``
   provenance, delta record, result materialized), every squashed delta
   is marked ``superseded_by`` in the manifest, the new record is
   appended, and the manifest is checkpointed + re-anchored.

The head id deliberately does NOT move. Rationale: the head is the
newest delta id, and rollback/diff/why semantics are all keyed on delta
ids — moving the head onto the squashed delta would rewrite what "the
newest self-write" means and orphan the kept tail above it. What changes
is the head's *resolved ancestry*: ``_active_chain`` now routes through
the squashed delta via ``superseded_by``, so export, forward passes, and
future squashes see the compressed stack, while the journal still
records every original delta that ever existed.

The scale=1 choice: the squashed delta stores the raw truncated-SVD
factors, so its effective update is exactly ``B @ A``. Any
``alpha``/``rank`` pair with ``alpha/rank == 1`` would compute the same
forward pass; we record ``alpha = rank = r`` so the record is
self-describing and the gate's rank-cap check applies to the true
factorization rank rather than a decorative number.

CPU only. This is machine learning infrastructure, not "custom LLM
software."
"""
from __future__ import annotations

import os

import numpy as np
from safetensors.numpy import save as _safetensors_save

from si.bundle import (
    _active_chain,
    load_bundle,
    load_policy,
    sha256_hex,
    utc_now,
)
from si.export import _load_base_tensors, materialize_head
from si.gate import Gate, Intent
from si.journal import open_journal
from si.loop import (
    _load_stack,
    _next_delta_id,
    _probe_err,
    _save_and_anchor,
    _write_blob,
)
from si.transformer_delta import KIND as TRANSFORMER_DELTA_KIND


class SquashError(ValueError):
    """A squash that refuses to proceed. Fail closed: nothing half-written."""


def _make_squash_mind(kind: str, bundle_dir: str):
    """Mind supplying probes + forward for the squash gate context."""
    if kind == "demo":
        from si.minds import DemoMind
        return DemoMind(bundle_path=bundle_dir)
    if kind == "mock":
        from si.minds import MockMind
        return MockMind()
    raise SquashError(f"unknown mind_kind {kind!r} (expected 'demo' or 'mock')")


class SquashContext:
    """The gate's BundleContext for a squash candidate.

    Probes come from the mind; the capability baseline is measured on the
    PRE-squash stack; ``forward_candidate`` runs the POST-squash stack
    (squashed delta, then the kept tail); ``stack_depth`` reports the
    pre-candidate depth so the gate's ``+1`` yields the true post-squash
    depth.
    """

    def __init__(self, mind, post_stack: list, baseline: dict, depth: int):
        self._mind = mind
        self._post_stack = post_stack
        self._baseline = dict(baseline)
        self._depth = depth

    # BundleContext protocol
    def identity_probes(self): return self._mind.identity_probes()
    def capability_suite(self): return self._mind.capability_suite()
    def capability_baseline(self): return dict(self._baseline)
    def stack_depth(self): return self._depth

    def forward_candidate(self, x, candidate):
        # The candidate IS the squashed delta, already first in post_stack.
        return self._mind.forward(x, self._post_stack)


def _remove_blob(bundle_dir: str, hexpart: str) -> None:
    path = os.path.join(bundle_dir, "blobs", "sha256", hexpart + ".safetensors")
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def squash_bundle(bundle_dir: str, keep: int = 3, rank: int | None = None,
                  tol: float = 1e-4, mind_kind: str = "demo") -> dict:
    """Compress the oldest deltas on the active stack into one gated delta.

    Returns ``{"squashed": False}`` when there is nothing to squash (or the
    gate refuses), else ``{"squashed": True, "delta_id", "edit_id",
    "verdict": "ADMIT", "head"}``. Raises SquashError (fail closed) on a
    multi-tensor base, a bad rank, or a tolerance violation — in which
    case nothing is written anywhere. Also raises SquashError (SI-12)
    when the resolved active chain contains any transformer-delta-v1
    record (in the squash segment, the kept tail, or both): transformer
    squash is not implemented, and that refusal happens before any
    blob, journal, or manifest write.
    """
    if keep < 0:
        raise SquashError(f"keep must be >= 0, got {keep!r}")
    if rank is not None and int(rank) < 1:
        raise SquashError(f"rank must be >= 1, got {rank!r}")

    bundle_dir = os.path.abspath(bundle_dir)
    manifest = load_bundle(bundle_dir)  # fail closed before touching anything

    # -- 1. split the active chain --------------------------------------
    chain = _active_chain(manifest)
    L = len(chain)
    if L <= keep:
        print(f"nothing to squash (active stack depth {L} <= keep {keep})")
        return {"squashed": False}
    S = chain[:L - keep]   # oldest: recompress these
    K = chain[L - keep:]   # newest: leave untouched

    # -- 2. SI-12: transformer deltas on the active chain fail closed ---
    # The check is over the WHOLE resolved chain (S + K), not only S:
    # K is involved in a squash — it is materialized for the tolerance
    # verify and loaded as demo factors by _load_stack for the gate —
    # and a transformer record has no demo rank/alpha or delta.B/delta.A
    # keys for that machinery. Refusing here, before _load_base_tensors,
    # any SVD, any blob write, any journal open, or any manifest change,
    # is what makes the refusal fail closed: nothing is left behind,
    # and no transformer delta is ever converted to a demo delta.
    # Records not on the resolved chain (rolled back / superseded /
    # beyond the head) are not checked: _active_chain never involves
    # them in a squash.
    transformer_ids = [d["id"] for d in chain
                       if d.get("kind") == TRANSFORMER_DELTA_KIND]
    if transformer_ids:
        raise SquashError(
            "si squash does not support transformer deltas: the active "
            "chain contains transformer-delta-v1 delta(s) "
            f"{', '.join(transformer_ids)} (in the squash segment and/or "
            "the kept tail); transformer squash is not implemented — "
            "it is deferred (RFC SI-9 §6.4; see "
            "docs/SI-12-squash-transformer-fail-closed.md). Refusing "
            "before any blob, journal, or manifest write, and refusing "
            "to convert a transformer delta to a demo delta"
        )

    # -- 3. SI-3 scope: single-tensor bases only ------------------------
    base_tensors = _load_base_tensors(bundle_dir, manifest)
    if len(base_tensors) != 1:
        raise SquashError(
            "si squash is single-tensor scope in SI-3: the base has "
            f"{len(base_tensors)} tensors; refusing to guess a squash layout"
        )
    (tname,) = base_tensors.keys()

    # -- 4. combined update of S: sub-head at S[-1] minus the base ------
    sub_manifest = dict(manifest)
    sub_manifest["head"] = S[-1]["id"]
    sub_tensors = materialize_head(bundle_dir, sub_manifest)
    update = (sub_tensors[tname].astype(np.float64)
              - base_tensors[tname].astype(np.float64))

    # -- 5. truncated SVD: B = U_r·√Σ_r, A = √Σ_r·V_rᵀ, alpha = rank = r --
    U, svals, Vt = np.linalg.svd(update, full_matrices=False)
    r_want = int(rank) if rank is not None else max(int(d["rank"]) for d in S)
    r = max(1, min(r_want, svals.shape[0]))
    root = np.sqrt(svals[:r])
    B = np.ascontiguousarray((U[:, :r] * root).astype(np.float32))
    A = np.ascontiguousarray((root[:, None] * Vt[:r, :]).astype(np.float32))
    # scale = alpha/rank = 1: the effective update is exactly B@A.
    frob = float(np.linalg.norm(B.astype(np.float64) @ A.astype(np.float64)))

    blob_bytes = _safetensors_save({"delta.B": B, "delta.A": A})
    blob_hexpart = sha256_hex(blob_bytes)
    delta_id = _next_delta_id(manifest)
    target_modules = sorted({m for d in S for m in (d.get("target_modules") or [])})
    new_rec = {
        "id": delta_id,
        "parent": S[0]["parent"],
        "blob": "sha256:" + blob_hexpart,
        "bytes": len(blob_bytes),
        "rank": r,
        "alpha": r,
        "target_modules": target_modules,
        "norm": frob,
        "created_at": utc_now(),
        "self_edit_id": None,  # filled in at commit
    }

    # The blob must exist on disk for the tolerance verify below, which
    # materializes through the substituted manifest via the exporter.
    blob_path = os.path.join(bundle_dir, "blobs", "sha256",
                             blob_hexpart + ".safetensors")
    blob_preexisted = os.path.isfile(blob_path)
    _write_blob(bundle_dir, blob_bytes)  # content-addressed; idempotent

    # -- 6. tolerance verify BEFORE gate/commit -------------------------
    pre_tensors = materialize_head(bundle_dir, manifest)
    squashed_ids = {d["id"] for d in S}
    verify_deltas = []
    for d in manifest["deltas"]:
        c = dict(d)
        if c["id"] in squashed_ids:
            c["superseded_by"] = delta_id  # substitution, on copies only
        verify_deltas.append(c)
    verify_deltas.append(dict(new_rec))
    verify_manifest = dict(manifest)
    verify_manifest["deltas"] = verify_deltas
    # head deliberately unchanged: the substitution is the ancestry rewrite.
    try:
        post_tensors = materialize_head(bundle_dir, verify_manifest)
    except Exception:
        if not blob_preexisted:
            _remove_blob(bundle_dir, blob_hexpart)
        raise
    worst = 0.0
    for name, pre in pre_tensors.items():
        diff = float(np.max(np.abs(post_tensors[name].astype(np.float64)
                                   - pre.astype(np.float64))))
        worst = max(worst, diff)
    if not worst <= tol:  # `not <=` so NaN fails closed too
        if not blob_preexisted:
            _remove_blob(bundle_dir, blob_hexpart)
        raise SquashError(
            f"tolerance violated: squashed head differs from the pre-squash "
            f"head by max abs {worst:.3e} >= tol {tol:.3e} "
            f"({len(S)} deltas -> {delta_id} at rank {r}); nothing written"
        )

    # -- 7+8. gate (full five checks, like any delta self-write) --------
    journal = open_journal(bundle_dir, manifest)
    try:
        # Provenance sources: the journal seqs of the squashed deltas' own
        # records — the causal events this squash exists because of.
        source_seqs = []
        for d in S:
            rows = journal.backend.find("deltas", filter={"id": d["id"]},
                                        limit=1)
            if rows and rows[0].get("seq"):
                source_seqs.append(rows[0]["seq"])

        intent = Intent(
            kind="delta",
            reason=f"squash {len(S)} deltas into one (keep={keep})",
            target_modules=target_modules,
            rank=r,
            alpha=r,
            parents=[d["id"] for d in S],
            source_seqs=source_seqs,
        )
        candidate = {
            "delta": {"B": B, "A": A, "alpha": r, "rank": r},
            "norm": frob,
            "blob_bytes": blob_bytes,
        }

        mind = _make_squash_mind(mind_kind, bundle_dir)
        full_stack = _load_stack(bundle_dir, manifest)  # pre-squash, in order
        pre_stack = full_stack
        k_entries = [{"B": e["B"], "A": e["A"],
                      "scale": float(e["scale"]), "id": e["id"]}
                     for e in full_stack[len(S):]]
        baseline = {cid: _probe_err(mind.forward(x, pre_stack), expected)
                    for cid, x, expected in mind.capability_suite()}
        post_stack = ([{"B": B, "A": A, "scale": 1.0, "id": delta_id}]
                      + k_entries)
        ctx = SquashContext(mind, post_stack, baseline, depth=L - len(S))

        gate = Gate(load_policy(bundle_dir))
        report = gate.evaluate(intent, candidate, ctx)

        if report.verdict in ("REJECT", "ESCALATE"):
            # Journaled exactly like the loop journals rejections. Nothing
            # else is written; the candidate blob stays as evidence.
            edit_id = journal.record_self_edit(
                intent.kind, intent.to_dict(), intent.reason,
                report.verdict, report.to_dict())
            journal.set_edit_result(
                edit_id,
                "rejected" if report.verdict == "REJECT" else "escalated")
            _save_and_anchor(
                bundle_dir, journal, manifest,
                note=f"{report.verdict.lower()} {edit_id}")
            return {"squashed": False, "verdict": report.verdict,
                    "edit_id": edit_id}

        # -- 9. commit (ADMIT only) --------------------------------------
        edit_id = journal.record_self_edit(
            "delta", intent.to_dict(), intent.reason,
            "ADMIT", report.to_dict())
        journal.record_provenance(
            child=delta_id,
            parents=[d["id"] for d in S],
            relation="squashed_from",
            note=(f"squash of {len(S)} deltas into {delta_id} "
                  f"(keep={keep}, rank={r})")[:200])
        journal.record_delta(
            delta_id, new_rec["parent"], "sha256:" + blob_hexpart,
            len(blob_bytes), r, r, target_modules, frob, edit_id,
            list(source_seqs))
        journal.set_edit_result(edit_id, "materialized", delta_id=delta_id)

        new_rec["self_edit_id"] = edit_id
        for d in manifest["deltas"]:
            if d["id"] in squashed_ids:
                d["superseded_by"] = delta_id
        manifest["deltas"].append(new_rec)
        # Head id deliberately UNCHANGED (see module docstring): the
        # resolved ancestry now routes through the squashed delta.
        manifest = _save_and_anchor(
            bundle_dir, journal, manifest,
            note=f"squash {len(S)}->{delta_id} (keep={keep})")
        return {"squashed": True, "delta_id": delta_id, "edit_id": edit_id,
                "verdict": "ADMIT", "head": manifest["head"]}
    finally:
        journal.close()


__all__ = ["SquashError", "SquashContext", "squash_bundle"]
