"""``si export``: materialize the head and write it out as a GGUF file.

The mind's learned weights are exportable, not just inspectable: the
frozen base plus every active LoRA-style delta (W = W + scale*(B@A))
materialized into full float32 tensors, then written as F16 GGUF with
``si.*`` metadata (bundle id, head, base source, arch source) plus real
``<arch>.*`` architecture KV entries when the base came from a genuine
transformer checkpoint (see ``si.arch``). An optional quant step
delegates to a REAL ``llama-quantize`` binary — never faked.

Determinism is a hard requirement: the same head produces a
byte-identical GGUF. That means no wall-clock timestamps anywhere in the
file, tensors written sorted by name, and ``si.bundle_id`` anchored to
head content rather than export-time journal state or manifest lineage:
the export itself journals (which moves the manifest's journal anchor),
and ``si squash`` rewrites the delta list — anchoring the id to either
would make behavior-preserving operations change the file.

Delta -> tensor resolution (honest, documented):
- Demo deltas (no ``kind`` or a non-transformer kind): unchanged from
  the original export. Single-tensor base (the demo seed's ``W``): the
  delta applies to that tensor, whatever the module string says.
  Multi-tensor base: a delta's module string (e.g. ``"demo_linear"``)
  resolves to every base tensor whose NAME contains the module string
  as a substring. No match, or a (B@A) result whose shape does not
  equal the target tensor's shape, is a ValueError.
- Transformer deltas (``kind == "transformer-delta-v1"``, SI-11 —
  RFC SI-9 §5.5 reconciliation): resolved by EXACT tensor name only,
  never substring (§2.4/§5.5 explicitly rejected substring for this
  kind). Each target is re-validated with
  ``si.transformer_delta.validate_transformer_delta`` against the
  manifest before any factor is trusted, factors are read from the
  indexed keys in the target's ``factor_keys`` (``t{i}.A``/``t{i}.B``
  / ``t{i}.d``), and the arithmetic mirrors
  ``si.apply.load_applied_tensor`` exactly: ``W + (alpha/rank)*B@A``
  for ``lora``, ``w + d`` for ``additive``. ``additive`` targets are
  applied only when ``allow_additive`` resolves to the literal
  ``True`` (the ``policy`` argument, else the bundle's
  ``policy.json``, else the fail-closed default False); otherwise
  export REFUSES with :class:`ExportError` — never silently skips or
  applies. Any transformer delta that cannot be validated/applied
  (unknown target, shape mismatch, metadata/record disagreement,
  missing factor keys, missing blob) fails closed with
  :class:`ExportError` — no raw ``SafetensorError``/``KeyError``
  leaks, no partial GGUF file, and no journal entry claiming a
  successful export (materialisation happens before any file write
  or journaling in ``export_bundle``).
- Mixed stacks: if a bundle somehow contains both demo and
  transformer deltas, export applies BOTH in ``_active_chain`` order
  (oldest first), each by its own kind's rules above. This is the
  documented SI-11 choice — not a silent guess: each kind's
  resolution is deterministic and fail-closed, so chain-order
  application is well defined. Rolled-back/superseded deltas are
  not on ``_active_chain`` and are therefore never applied.

The export is NOT gated and does NOT move the head: it reads the
current head, writes a file, and journals ``relation="export"``
provenance for the produced artifact. No delta, no self-edit.

Honest limits: materialisation still loads ALL base tensors into
memory as float32, so exporting a full Glimmer head (~119 GB as
float32) is out of scope on this machine (RFC §5.4) — SI-11 changes
correctness for transformer deltas, not that cost. ``si serve`` is
unchanged (still 501 for transformer bundles).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess

from si.bundle import (
    _active_chain,
    _anchor_checkpoint,
    _write_manifest,
    blob_path,
    load_bundle,
    sha256_hex,
    utc_now,
)


class ExportError(ValueError):
    """A head cannot be materialised/exported honestly. Fail closed.

    Raised for transformer-delta problems in export (SI-11):
    validation failure, unknown exact target, shape mismatch,
    missing factor keys/blob, or an additive target the policy does
    not allow. Subclasses ValueError so existing call sites that
    treat malformed input as ValueError keep working. No raw
    SafetensorError/KeyError from the transformer path leaks past
    this error.
    """


# -- delta resolution -------------------------------------------------------
def _load_base_tensors(bundle_dir: str, manifest: dict) -> dict:
    """Load ALL base tensors as float32 numpy arrays, keyed by name.

    Handles all four base layouts: SI-8 by-reference bases
    (``base["tensor_refs"]`` — each name resolves to a byte range in
    the source checkpoint, read and hash-verified per tensor via
    ``si.bundle.load_base_tensor``), SI-7 per-tensor bases
    (``base["tensor_blobs"]`` — each name maps to its own blob, read
    via ``si.bundle.read_tensor_blob`` so identical tensors sharing one
    deduplicated blob both resolve), pre-SI-7 seeded multi-part bases
    (``base["parts"]`` whole-file blobs), and the demo single-blob base
    (``base["blob"]``). Duplicate tensor names across parts fail closed
    with ValueError rather than silently shadowing.
    Pure read: no journal, no manifest writes.
    """
    import numpy as np
    from safetensors import safe_open

    from si.bundle import load_base_tensor, read_tensor_blob

    base = manifest["base"]
    tensor_refs = base.get("tensor_refs")
    if isinstance(tensor_refs, dict) and tensor_refs:
        tensors = {}
        for name in sorted(tensor_refs):
            arr = load_base_tensor(bundle_dir, manifest, name)
            tensors[name] = np.asarray(arr, dtype=np.float32)
        return tensors

    tensor_blobs = base.get("tensor_blobs")
    if isinstance(tensor_blobs, dict) and tensor_blobs:
        tensors = {}
        for name in sorted(tensor_blobs):
            arr = read_tensor_blob(bundle_dir, tensor_blobs[name]["blob"], name)
            tensors[name] = np.asarray(arr, dtype=np.float32)
        return tensors

    part_refs = [p["blob"] for p in base.get("parts") or []
                 if isinstance(p, dict) and p.get("blob")]
    blob_refs = part_refs if part_refs else [base["blob"]]

    tensors = {}
    for ref in blob_refs:
        blob_hex = ref.split(":", 1)[1]
        path = os.path.join(bundle_dir, "blobs", "sha256", blob_hex + ".safetensors")
        with safe_open(path, framework="np") as f:
            for name in f.keys():
                if name in tensors:
                    raise ValueError(
                        f"duplicate tensor name {name!r} across base parts "
                        "(refusing to shadow)"
                    )
                tensors[name] = np.asarray(f.get_tensor(name), dtype=np.float32)
    return tensors


def _load_delta(bundle_dir: str, delta: dict) -> tuple:
    """Return (B, A, scale) for a delta record, float32 numpy."""
    import numpy as np
    from safetensors import safe_open

    blob_hex = delta["blob"].split(":", 1)[1]
    path = os.path.join(bundle_dir, "blobs", "sha256", blob_hex + ".safetensors")
    with safe_open(path, framework="np") as f:
        B = np.asarray(f.get_tensor("delta.B"), dtype=np.float32)
        A = np.asarray(f.get_tensor("delta.A"), dtype=np.float32)
    return B, A, float(delta["alpha"]) / float(delta["rank"])


def _resolve_targets(delta_id: str, modules: list, base_names: list) -> list:
    """Resolve a delta's target modules to base tensor names.

    Single-tensor base: that tensor, whatever the module strings say.
    Multi-tensor base: tensors whose name contains the module substring.
    No match -> ValueError (honest; never guess).
    """
    if len(base_names) == 1:
        return [base_names[0]]
    targets = []
    for module in modules:
        hits = [n for n in base_names if module in n]
        if not hits:
            raise ValueError(
                f"delta {delta_id}: target module {module!r} matches no "
                f"base tensor (base has {len(base_names)} tensors)"
            )
        targets.extend(hits)
    return targets


# -- transformer deltas in export (SI-11) ------------------------------------

def _resolve_export_allow_additive(bundle_dir: str,
                                   policy: dict | None) -> bool:
    """``allow_additive`` resolution for export, fail closed.

    Mirrors ``si.apply._resolve_allow_additive`` exactly: the
    ``policy`` argument when given, else the bundle's ``policy.json``
    when it exists, else False. Only the literal ``True`` permits
    additive application.
    """
    from si.apply import _resolve_allow_additive
    return _resolve_allow_additive(bundle_dir, policy)


def _apply_transformer_delta(out: dict, bundle_dir: str, manifest: dict,
                             record: dict, allow_additive: bool) -> None:
    """Fold one transformer-delta-v1 record into ``out`` in place.

    Exact names only. The record/blob pair is re-validated with
    SI-9a validation before any factor is trusted; factor math
    mirrors ``si.apply.load_applied_tensor`` (float32, per target).
    Every failure is an ExportError — no raw SafetensorError/KeyError.
    """
    import numpy as np

    from si.transformer_delta import (
        DeltaValidationError,
        parse_blob,
        validate_transformer_delta,
    )

    delta_id = record.get("id", "?")
    blob_ref = record.get("blob")
    if not isinstance(blob_ref, str) or not blob_ref.startswith("sha256:"):
        raise ExportError(
            f"delta {delta_id}: transformer delta has no valid blob "
            f"ref {blob_ref!r}; refusing to export")
    try:
        path = blob_path(bundle_dir, blob_ref)
    except Exception as err:
        raise ExportError(
            f"delta {delta_id}: malformed blob ref {blob_ref!r}: {err}"
        ) from err
    if not os.path.isfile(path):
        raise ExportError(
            f"delta {delta_id}: transformer delta blob missing from "
            f"the bundle store: {path}; refusing to export")
    with open(path, "rb") as f:
        blob_bytes = f.read()
    try:
        validate_transformer_delta(record, blob_bytes, manifest)
    except DeltaValidationError as err:
        raise ExportError(
            f"delta {delta_id}: invalid transformer delta: {err}"
        ) from err
    except Exception as err:
        raise ExportError(
            f"delta {delta_id}: transformer delta validation failed: "
            f"{type(err).__name__}: {err}"
        ) from err
    try:
        parsed = parse_blob(blob_bytes)
    except Exception as err:
        raise ExportError(
            f"delta {delta_id}: transformer delta blob does not "
            f"parse: {err}"
        ) from err

    targets = record.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ExportError(
            f"delta {delta_id}: transformer delta carries no targets; "
            "refusing to export")
    for target in targets:
        if not isinstance(target, dict):
            raise ExportError(
                f"delta {delta_id}: transformer target is not an "
                "object; refusing to export")
        name = target.get("name")
        kind = target.get("kind")
        if not isinstance(name, str) or name not in out:
            raise ExportError(
                f"delta {delta_id}: transformer target {name!r} is "
                "not a base tensor (exact name match required); "
                "refusing to export")
        factor_keys = target.get("factor_keys") or []
        if kind == "lora":
            if len(factor_keys) != 2:
                raise ExportError(
                    f"delta {delta_id}: lora target {name!r} must "
                    f"carry exactly 2 factor keys, got {factor_keys!r}")
            for key in factor_keys:
                if key not in parsed.tensors:
                    raise ExportError(
                        f"delta {delta_id}: lora target {name!r} "
                        f"factor {key!r} missing from blob; refusing "
                        "to export")
            A = np.asarray(parsed.tensors[factor_keys[0]],
                           dtype=np.float32)
            B = np.asarray(parsed.tensors[factor_keys[1]],
                           dtype=np.float32)
            try:
                scale = float(target["alpha"]) / float(target["rank"])
            except Exception as err:
                raise ExportError(
                    f"delta {delta_id}: lora target {name!r} has "
                    f"invalid rank/alpha: {err}"
                ) from err
            update = (scale * (B @ A)).astype(np.float32)
            if tuple(update.shape) != tuple(out[name].shape):
                raise ExportError(
                    f"delta {delta_id}: (B@A) shape "
                    f"{tuple(update.shape)} does not match target "
                    f"tensor {name!r} shape {tuple(out[name].shape)}")
            out[name] = np.asarray(out[name] + update, dtype=np.float32)
        elif kind == "additive":
            if not allow_additive:
                raise ExportError(
                    f"delta {delta_id}: additive target {name!r} "
                    "cannot be exported because policy allow_additive "
                    "is not true (default false, RFC §4.5): refusing "
                    "to export rather than silently skipping or "
                    "applying it")
            if len(factor_keys) != 1:
                raise ExportError(
                    f"delta {delta_id}: additive target {name!r} "
                    f"must carry exactly 1 factor key, got "
                    f"{factor_keys!r}")
            key = factor_keys[0]
            if key not in parsed.tensors:
                raise ExportError(
                    f"delta {delta_id}: additive target {name!r} "
                    f"factor {key!r} missing from blob; refusing to "
                    "export")
            d = np.asarray(parsed.tensors[key], dtype=np.float32)
            if tuple(d.shape) != tuple(out[name].shape):
                raise ExportError(
                    f"delta {delta_id}: additive shape "
                    f"{tuple(d.shape)} does not match target tensor "
                    f"{name!r} shape {tuple(out[name].shape)}")
            out[name] = np.asarray(out[name] + d, dtype=np.float32)
        else:
            raise ExportError(
                f"delta {delta_id}: target {name!r} has unknown "
                f"kind {kind!r}; refusing to export")


# -- materialization ---------------------------------------------------------
def materialize_head(bundle_dir: str, manifest: dict | None = None,
                     policy: dict | None = None) -> dict:
    """Materialize the current head into full float32 weight tensors.

    Pure read. Loads every base tensor plus the active delta stack (the
    ``_active_chain`` resolution of head: parent links with squash
    ``superseded_by`` replacements followed) and applies deltas in
    chain order, oldest first: demo deltas as
    ``W = W + scale*(B@A)`` on torch CPU in float32 (unchanged), and
    transformer deltas per-target by exact name with the
    ``si.apply.load_applied_tensor`` semantics (SI-11; see module
    docs). ``manifest`` may be supplied; otherwise it is loaded AND
    verified via ``load_bundle`` (fail closed). ``policy`` optionally
    overrides the bundle's ``policy.json`` for the additive rule
    (only the literal ``True`` for ``allow_additive`` permits
    additive transformer targets). No gate, no journal, no manifest
    writes.

    Returns {tensor_name: float32 np.ndarray}.
    """
    import numpy as np
    import torch

    from si.transformer_delta import KIND as TRANSFORMER_KIND

    if manifest is None:
        manifest = load_bundle(bundle_dir)

    base = _load_base_tensors(bundle_dir, manifest)
    base_names = list(base.keys())
    head = manifest["head"]

    if head != "base" and head not in [d["id"] for d in manifest["deltas"]]:
        raise ValueError(f"unknown head {head!r}")

    out = {name: base[name].copy() for name in base_names}
    allow_additive = None
    for d in _active_chain(manifest):
        if d.get("kind") == TRANSFORMER_KIND:
            if allow_additive is None:
                allow_additive = _resolve_export_allow_additive(
                    bundle_dir, policy)
            _apply_transformer_delta(out, bundle_dir, manifest, d,
                                     allow_additive)
            continue
        B, A, scale = _load_delta(bundle_dir, d)
        modules = list(d.get("target_modules") or [])
        targets = _resolve_targets(d["id"], modules, base_names)
        with torch.no_grad():
            update = (torch.from_numpy(scale * (B @ A))).to(torch.float32)
            for name in targets:
                W = torch.from_numpy(out[name])
                if update.shape != tuple(W.shape):
                    raise ValueError(
                        f"delta {d['id']}: (B@A) shape {tuple(update.shape)} "
                        f"does not match target tensor {name!r} shape "
                        f"{tuple(W.shape)}"
                    )
                out[name] = (W + update).numpy().astype(np.float32)
    return out


def head_hash(tensors: dict) -> str:
    """sha256 over sorted (name, shape, dtype, raw bytes) of a tensor dict."""
    import numpy as np

    h = hashlib.sha256()
    for name in sorted(tensors):
        arr = np.ascontiguousarray(tensors[name], dtype=np.float32)
        h.update(name.encode("utf-8"))
        h.update(str(tuple(arr.shape)).encode("utf-8"))
        h.update(b"float32")
        h.update(arr.tobytes())
    return h.hexdigest()


# -- GGUF writing ------------------------------------------------------------
def _bundle_id_for_head(manifest: dict, tensors: dict) -> str:
    """Bundle id anchored to head content, not export-time journal state.

    The export itself journals (which moves the manifest's journal
    anchor), and representation rewrites like ``si squash`` change the
    manifest's delta list without changing what the head computes.
    Deriving the id from the manifest would let either move the id, so
    the id is a pure function of the head instead: the head id plus the
    exported (F16) tensor bytes. The same head content always yields the
    same id, and a behavior-preserving rewrite keeps the exported file
    byte-identical.
    """
    import numpy as np

    h = hashlib.sha256()
    h.update(b"si-bundle-content-v1")
    h.update(str(manifest["head"]).encode("utf-8"))
    for name in sorted(tensors):
        arr = np.ascontiguousarray(tensors[name], dtype=np.float16)
        h.update(name.encode("utf-8"))
        h.update(str(tuple(arr.shape)).encode("utf-8"))
        h.update(b"float16")
        h.update(arr.tobytes())
    return "si1:" + h.hexdigest()


def _write_f16_gguf(out_path: str, tensors: dict, manifest: dict) -> None:
    """Write materialized head tensors as F16 GGUF. Deterministic by design.

    Tensors sorted by name; fixed KV insertion order; no timestamps. The
    same head must produce a byte-identical file.

    Architecture metadata comes from ``si.arch.derive_arch``: a real
    checkpoint base yields real ``<arch>.*`` KV entries (arch name, block
    count, head counts, embedding length, vocab size, ...); a demo base
    keeps ``general.architecture = "unknown"`` with
    ``si.arch_source = "demo-unknown"``. Unknown values are omitted,
    never invented.
    """
    import numpy as np
    import gguf

    from si.arch import ARCH_FIELD_ORDER, derive_arch

    shapes = {name: tuple(tensors[name].shape) for name in tensors}
    arch = derive_arch(manifest, shapes)

    writer = gguf.GGUFWriter(out_path, arch["arch"])
    writer.add_name(str(manifest.get("name", "sentinel")))
    # Arch KVs in fixed order: derive_arch is a pure function of the
    # manifest + shapes, so this stays deterministic.
    _add_arch_kvs(writer, arch)
    writer.add_key_value("si.arch_source", str(arch["arch_source"]),
                         gguf.GGUFValueType.STRING)
    writer.add_key_value("si.bundle_id",
                         _bundle_id_for_head(manifest, tensors),
                         gguf.GGUFValueType.STRING)
    writer.add_key_value("si.head", str(manifest["head"]),
                         gguf.GGUFValueType.STRING)
    writer.add_key_value("si.base_source",
                         str(manifest["base"].get("source", "")),
                         gguf.GGUFValueType.STRING)
    for name in sorted(tensors):
        writer.add_tensor(name,
                          np.ascontiguousarray(tensors[name], dtype=np.float16))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _add_arch_kvs(writer, arch: dict) -> None:
    """Write derived arch fields as real llama.cpp KV entries.

    Each value was read from the checkpoint's own config.json or
    conservatively inferred from tensor shapes; None means unknown and
    is omitted rather than fabricated.
    """
    from si.arch import ARCH_FIELD_ORDER

    for field in ARCH_FIELD_ORDER:
        value = arch.get(field)
        if value is None:
            continue
        if field == "block_count":
            writer.add_block_count(value)
        elif field == "embedding_length":
            writer.add_embedding_length(value)
        elif field == "context_length":
            writer.add_context_length(value)
        elif field == "head_count":
            writer.add_head_count(value)
        elif field == "head_count_kv":
            writer.add_head_count_kv(value)
        elif field == "layer_norm_rms_eps":
            writer.add_layer_norm_rms_eps(value)
        elif field == "rope_dimension_count":
            writer.add_rope_dimension_count(value)
        elif field == "vocab_size":
            writer.add_vocab_size(value)


# -- quant -------------------------------------------------------------------
def _llama_quantize_bin() -> str:
    """Path to a real llama-quantize binary, or raise an honest error."""
    bin_path = os.path.expanduser("~/.hearth/runtime/bin/llama-quantize")
    if not (os.path.isfile(bin_path) and os.access(bin_path, os.X_OK)):
        raise RuntimeError(
            "no llama-quantize found at ~/.hearth/runtime/bin/llama-quantize; "
            "fetch a real one with `hearth runtime llama` (pip install "
            "hearth-engine), then re-run `si export --quant`. Refusing to "
            "fake a quantized file."
        )
    return bin_path


def _run_quantize(f16_path: str, out_path: str, quant: str) -> None:
    """Quantize via the real llama-quantize. Real errors propagate."""
    bin_path = _llama_quantize_bin()
    subprocess.run([bin_path, f16_path, out_path, quant], check=True)


# -- the export ---------------------------------------------------------------
def export_bundle(bundle_dir: str, out_path: str,
                  quant: str | None = None,
                  policy: dict | None = None) -> dict:
    """Export the current head as a GGUF file.

    1. ``load_bundle`` first: fail closed on any tamper before touching
       anything.
    2. ``materialize_head``: base + active deltas -> float32 tensors.
       Transformer deltas are validated/applied there (SI-11); a
       failure raises ExportError BEFORE any file is written or any
       journal entry is made — no partial GGUF, no export provenance
       claiming success.
    3. Write F16 GGUF (deterministic; see ``_write_f16_gguf``).
    4. If ``quant``: delegate to a REAL llama-quantize binary. Missing
       binary -> RuntimeError (honest; never faked). The quantized file
       replaces ``out_path``.
    5. Journal (NOT gated, head NOT moved, no delta): provenance
       ``relation="export"`` for the artifact, then re-anchor the
       manifest to the new journal tip (same pattern as si/seed.py).
    6. Return {"head","head_hash","out_path","bytes","sha256","quant",
       "journal_seq"} — journal_seq is the export provenance entry's seq.
    ``policy`` is forwarded to ``materialize_head`` for the additive
    rule; when None the bundle's ``policy.json`` is used.
    """
    from si.journal import open_journal  # local import: no cycle

    # Fail fast on a requested-but-impossible quant BEFORE writing anything.
    if quant is not None:
        _llama_quantize_bin()

    # 1. Load + verify first: fail closed on any tamper.
    bundle_dir = os.path.abspath(bundle_dir)
    manifest = load_bundle(bundle_dir)

    # 2. Materialize the head (pure read).
    tensors = materialize_head(bundle_dir, manifest, policy=policy)
    hhash = head_hash(tensors)
    head = manifest["head"]

    # 3/4. Write F16 GGUF; quantize in place if requested.
    # out_path is absolute in the return contract and the journal note.
    out_path = os.path.abspath(str(out_path))
    if quant is None:
        _write_f16_gguf(out_path, tensors, manifest)
    else:
        f16_path = out_path + ".f16.tmp"
        try:
            _write_f16_gguf(f16_path, tensors, manifest)
            _run_quantize(f16_path, out_path, quant)
        finally:
            if os.path.exists(f16_path):
                os.remove(f16_path)

    with open(out_path, "rb") as f:
        file_bytes = f.read()
    file_hex = sha256_hex(file_bytes)

    # 5. Journal the artifact (not gated, head untouched), then re-anchor
    #    the manifest to the new tip so load_bundle verifies afterwards.
    journal = open_journal(bundle_dir, manifest)
    try:
        child = "gguf:sha256:" + file_hex
        note = json.dumps({
            "out_path": out_path,
            "bytes": len(file_bytes),
            "sha256": file_hex,
            "quant": quant,
            "exported_at": utc_now(),
            "head": head,
        })
        journal.record_provenance(child=child, parents=[head],
                                  relation="export", note=note)
        journal_seq = journal.tip()[0]
        manifest = _anchor_checkpoint(journal, manifest, head,
                                      note=f"export {head} -> {out_path}")
        _write_manifest(bundle_dir, manifest)
    finally:
        journal.close()

    # 6. Report.
    return {"head": head, "head_hash": hhash, "out_path": out_path,
            "bytes": len(file_bytes), "sha256": file_hex, "quant": quant,
            "journal_seq": journal_seq}


__all__ = ["ExportError", "materialize_head", "head_hash", "export_bundle"]
