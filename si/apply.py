"""SI-9c: apply-at-load for transformer deltas (RFC SI-9 §5, §6.3).

A library-level, per-tensor, read-only apply path:

    applied(name) = base(name)  +  sum over the active chain, oldest
    first, of each transformer delta that targets ``name`` exactly

where a ``lora`` target contributes ``(alpha/rank) * B @ A`` and an
``additive`` target contributes its stored vector ``d`` (RFC §5.1).
Resolution goes through :func:`si.bundle._active_chain` (the same
resolution squash and export use, so a rolled-back delta — one no
longer on the chain to ``head`` — simply stops being applied) and the
base tensor itself comes from :func:`si.bundle.load_base_tensor`,
which hash-verifies the tensor it reads in every base layout
(copy-mode blobs, SI-8 by-reference byte ranges). Deltas that do not
target ``name`` cost one manifest lookup, not a read.

Module-location decision (RFC §7.4 Q3 left this open; the RFC leans
``si/apply.py`` and SI-9c follows the lean): the apply path lives in
this NEW module, not in ``si/bundle.py``. ``bundle.py``'s contract is
container storage + integrity verification; arithmetic is neither,
and keeping it separate means the gate / format layers never grow a
dependency on apply internals.

Honest labels — what this is and is not:

- This is a **library capability**, used by tests and by the loop's
  post-write VERIFY smoke step for transformer deltas
  (``si.loop._verify_transformer_apply``). It is NOT inference:
  nothing here runs a transformer, and ``si serve`` still returns
  501 for transformer bundles. Nothing here trains or authors a
  delta; authorship is out of scope (RFC §2.5).
- **Nothing is materialised.** The applied tensor exists only as the
  returned in-memory array. No file is written, no "applied base" is
  ever stored, and the base bytes are never modified or copied —
  the content-addressed store rule (never duplicate bytes) holds.
- **Dtype (RFC §5.3):** base tensors widen to float32 on read (bf16
  widening is lossless; numpy has no bf16 dtype in this runtime),
  factors are float32, the arithmetic is float32, and the result is
  float32. Nothing casts back to bf16 here because nothing persists
  the result; a future writer must make that cast explicitly.

Fail-closed behaviour:

- Before a targeting delta is applied, its record/blob pair is run
  through SI-9a's :func:`si.transformer_delta.validate_transformer_delta`
  against the caller's manifest — blob content hash, metadata-vs-
  manifest agreement, exact-name base binding (``base_blob`` /
  ``base_shape`` / ``base_dtype``), factor shapes/dtypes, finiteness,
  and recomputed norms are all re-checked at apply time, so a
  tampered delta raises :class:`DeltaValidationError` (a ValueError)
  instead of being silently applied. Non-targeting deltas are not
  read or validated by a call that does not need them.
- ``additive`` targets are applied only when the policy allows them
  (RFC §4.5 / SI-9b: ``allow_additive``, default False). The policy
  is taken from the ``policy`` argument when given, else from the
  bundle's ``policy.json`` when it exists, else the fail-closed
  default. Only the literal ``True`` permits additive application —
  the same resolution the gate uses; anything else raises
  :class:`ApplyError`. ``lora`` targets need no such policy key:
  gating (allowlist, caps) already happened — or honestly did not,
  for a hand-authored fixture — before a delta ever lands in a
  manifest, and this path does not re-litigate gate policy beyond
  the additive rule the RFC ties to apply.
- Records of any other kind in the active chain (demo deltas) carry
  no ``targets`` and contribute nothing here. This path never
  applies a demo delta; the demo loop's forward path is separate
  and unchanged. Mixed stacks remain a non-goal (RFC §3).

SI-15 additions (RFC SI-13 §4.2): the per-target fold arithmetic now
lives in ONE shared function, :func:`fold_target`, used by both
:func:`load_applied_tensor` (active chain) and
:func:`applied_tensor_with_candidate` (the candidate overlay the
probe harness needs: the gate evaluates a candidate *before* it
lands, so the candidate is not on the active chain and
``load_applied_tensor`` alone cannot see it). The overlay is NOT a
third hand-copied fold — RFC SI-13 §4.2 forbids that (SI-11's export
materialise path already mirrors this arithmetic once, as its own
docstring records; the overlay deliberately does not add another).
``si/tiny_transformer.py`` re-exports the overlay for the forward
path's weight assembly.
"""
from __future__ import annotations

import os

import numpy as np

from si.bundle import _active_chain, blob_path, load_base_tensor, load_policy
from si.transformer_delta import (
    KIND as TRANSFORMER_DELTA_KIND,
    base_tensor_map,
    parse_blob,
    validate_transformer_delta,
)


class ApplyError(ValueError):
    """A transformer delta cannot be applied as requested.

    Subclasses ValueError so existing call sites that treat malformed
    or impermissible input as ValueError keep working.
    """


def _resolve_allow_additive(bundle_dir: str, policy: dict | None) -> bool:
    """``allow_additive`` resolution, fail closed (see module docs)."""
    if policy is None:
        if os.path.isfile(os.path.join(bundle_dir, "policy.json")):
            policy = load_policy(bundle_dir)
        else:
            policy = {}
    if not isinstance(policy, dict):
        return False
    return policy.get("allow_additive", False) is True


def _read_blob_bytes(bundle_dir: str, blob_ref: str) -> bytes:
    path = blob_path(bundle_dir, blob_ref)
    if not os.path.isfile(path):
        raise ApplyError(f"delta blob missing from the bundle store: {path}")
    with open(path, "rb") as f:
        return f.read()


def fold_target(result: np.ndarray, target: dict, factor_tensors: dict,
                *, allow_additive: bool,
                delta_id: str | None = None) -> np.ndarray:
    """Fold ONE validated target's update into ``result`` (shared fold).

    This is the single implementation of the RFC §5.1 per-target
    arithmetic — ``result + (alpha/rank)·B@A`` for ``lora`` (float32,
    the update cast to float32 before the add, exactly as the
    pre-SI-15 inline apply loop computed it) and ``result + d`` for
    ``additive`` (policy-gated here, fail closed). Extracted verbatim
    from ``load_applied_tensor`` in SI-15 so the active-chain path
    and the candidate overlay (RFC SI-13 §4.2) share arithmetic by
    construction, not by copy. Callers must have validated the delta
    (SI-9a) before trusting its factors — this function trusts
    ``target``/``factor_tensors`` and only enforces the additive
    policy rule and the post-fold shape invariant. Returns a new
    array; ``result`` is not mutated.
    """
    kind = target["kind"]
    name = target["name"]
    if kind == "lora":
        A = np.asarray(factor_tensors[target["factor_keys"][0]],
                       dtype=np.float32)
        B = np.asarray(factor_tensors[target["factor_keys"][1]],
                       dtype=np.float32)
        scale = float(target["alpha"]) / float(target["rank"])
        result = result + (scale * (B @ A)).astype(np.float32)
    elif kind == "additive":
        if not allow_additive:
            raise ApplyError(
                f"delta {delta_id!r} targets {name!r} "
                "with an additive update, but policy "
                "allow_additive is not true (RFC §4.5 default "
                "is false): refusing to apply it")
        d = np.asarray(factor_tensors[target["factor_keys"][0]],
                       dtype=np.float32)
        result = result + d
    else:  # unreachable for validated deltas: validation rejects kinds
        raise ApplyError(
            f"delta {delta_id!r} target {name!r} has "
            f"unknown kind {kind!r}")
    if tuple(result.shape) != tuple(target["base_shape"]):
        raise ApplyError(
            f"applied tensor {name!r} has shape "
            f"{tuple(result.shape)} != base shape "
            f"{tuple(target['base_shape'])}")
    return result


def load_applied_tensor(bundle_dir: str, manifest: dict, name: str,
                        policy: dict | None = None) -> np.ndarray:
    """Return base tensor ``name`` with the active transformer deltas
    for that exact name folded in, oldest first (RFC §5.1).

    ``manifest`` is the caller's (verified) manifest dict — this
    function deliberately does NOT call ``load_bundle``: for a
    by-reference base a full verification is a streaming pass over
    the whole checkpoint (RFC §7.1), and per-tensor reads through
    ``load_base_tensor`` already re-verify the tensor actually read.
    The result is a new float32 ndarray; the base array is never
    mutated. Raises KeyError if the base carries no such tensor,
    DeltaValidationError if a targeting delta fails SI-9a
    validation, and ApplyError if an additive target is not
    policy-permitted or a delta blob is missing.
    """
    # Exact-name existence, fail closed: on a multi-tensor base (one
    # with a per-tensor map) an unknown name is a KeyError here, NOT
    # a fall-through to load_base_tensor's single-blob fallback,
    # which would silently return some other tensor's data.
    tensor_map = base_tensor_map(manifest)
    if tensor_map and name not in tensor_map:
        raise KeyError(f"base carries no tensor named {name!r}")
    base = load_base_tensor(bundle_dir, manifest, name)
    result = np.array(base, dtype=np.float32, copy=True)
    allow_additive = _resolve_allow_additive(bundle_dir, policy)

    for record in _active_chain(manifest):
        if record.get("kind") != TRANSFORMER_DELTA_KIND:
            continue  # no targets; never a demo delta (module docs)
        targets = [t for t in record.get("targets") or []
                   if isinstance(t, dict) and t.get("name") == name]
        if not targets:
            continue  # one manifest lookup, not a read (RFC §5.1)
        blob_bytes = _read_blob_bytes(bundle_dir, record["blob"])
        # Fail closed: re-validate the pair against the manifest
        # before any factor is trusted (module docs).
        validate_transformer_delta(record, blob_bytes, manifest)
        parsed = parse_blob(blob_bytes)
        for target in targets:
            result = fold_target(result, target, parsed.tensors,
                                 allow_additive=allow_additive,
                                 delta_id=record.get("id"))
    return result


def applied_tensor_with_candidate(bundle_dir: str, manifest: dict,
                                  name: str, candidate,
                                  policy: dict | None = None) -> np.ndarray:
    """Base + active chain + ONE not-yet-landed candidate, for ``name``.

    The RFC SI-13 §4.2 candidate overlay. ``candidate`` is the SI-9b
    candidate contract — ``{"record": <transformer-delta record>,
    "blob_bytes": <safetensors bytes>}`` — or ``None`` (no candidate:
    the result is exactly :func:`load_applied_tensor`'s).

    - For a name the candidate does NOT target, the
      ``load_applied_tensor`` result is returned untouched: the
      candidate blob is not read and not validated by this call
      (one manifest lookup — the RFC §5.1 apply principle).
    - For a targeted name, the candidate pair is FIRST run through
      SI-9a's :func:`validate_transformer_delta` against the
      caller's verified ``manifest`` (validation first; factors are
      trusted only after it passes, exactly as the active-chain
      path does), then the candidate's factor(s) for that exact
      name are folded in with :func:`fold_target` — the shared
      arithmetic, including the ``allow_additive`` policy rule
      resolved the same way as the chain path.

    Read-only and in-memory, like apply-at-load itself: nothing is
    materialised, no bundle state is written, and the candidate does
    not land anywhere by being overlaid.
    """
    result = load_applied_tensor(bundle_dir, manifest, name, policy)
    if candidate is None:
        return result
    record = candidate.get("record") if isinstance(candidate, dict) else None
    if not isinstance(record, dict):
        raise ApplyError(
            "candidate overlay needs the SI-9b candidate contract "
            "{'record': <transformer-delta record>, 'blob_bytes': "
            "<bytes>}: candidate carries no record dict")
    targets = [t for t in record.get("targets") or []
               if isinstance(t, dict) and t.get("name") == name]
    if not targets:
        return result  # untargeted: candidate never read (see docs)
    blob_bytes = candidate.get("blob_bytes")
    if not isinstance(blob_bytes, (bytes, bytearray)):
        raise ApplyError(
            "candidate overlay: candidate targets "
            f"{name!r} but carries no blob bytes to validate")
    blob_bytes = bytes(blob_bytes)
    # Fail closed, validation first (same rule as the chain path).
    validate_transformer_delta(record, blob_bytes, manifest)
    parsed = parse_blob(blob_bytes)
    allow_additive = _resolve_allow_additive(bundle_dir, policy)
    for target in targets:
        result = fold_target(result, target, parsed.tensors,
                             allow_additive=allow_additive,
                             delta_id=record.get("id"))
    return result
