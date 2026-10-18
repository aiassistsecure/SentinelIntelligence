"""``si seed``: import a REAL safetensors checkpoint as a bundle's frozen base.

SI-7: the import is PER-TENSOR and content-addressed. Each tensor of the
checkpoint becomes its own blob in ``blobs/sha256/``; the manifest records
per-tensor refs (``base.tensor_blobs``: name -> {blob, bytes, shape,
dtype}) alongside the legacy summary fields. Checkpoints in the sharded
HuggingFace layout (``model.safetensors.index.json`` + shard files) are
supported and validated fail-closed.

Safety story (this is the whole point of the format check):
- safetensors is inherently pickle-free: there is no deserialization
  gadgetry to exploit. A file that ``safetensors.safe_open`` opens and
  reads keys from IS a valid safetensors container, and that successful
  open is the safety proof. Corrupt headers, truncated files, and
  non-safetensors bytes all raise and are wrapped as ``ValueError``.
- The base is content-addressed: every tensor blob is SHA-256'd and
  stored under ``blobs/sha256/<hex>.safetensors``. ``load_bundle``
  re-verifies the hash of every referenced blob on every load, so a
  tampered blob fails closed.
- ``si seed`` refuses any bundle that already has history (non-empty
  deltas or head != "base"): the base is frozen, and seeding over a lived
  life would silently rewrite the mind's origin. Fail closed instead.
  One exception, by design: re-seeding the EXACT checkpoint that is
  already the bundle's base is an idempotent no-op — nothing is written,
  nothing is journaled, and the result reports ``already_seeded``.
- The import is journaled (provenance ``seeded_from``: tensor count,
  total bytes, deduplicated bytes, source path, config sha256 when
  present) and the manifest is re-anchored to the new journal tip, so
  ``load_bundle`` verifies cleanly afterwards.
- When a ``config.json`` sits beside the weights, it is stored in the
  manifest (``base.checkpoint_config`` + ``arch_source="checkpoint-config"``)
  so ``si export`` can write real GGUF architecture metadata. Without it,
  export falls back to conservative tensor-shape inference.

Per-tensor blob format (the dedup design):
- A tensor blob is a valid single-tensor safetensors file whose sole
  tensor is stored under the canonical key ``"tensor"`` — NOT under the
  tensor's original name. The name lives in the manifest
  (``base.tensor_blobs``), exactly like a git tree names blobs. This is
  what makes content addressing address the tensor's DATA (dtype +
  shape + raw bytes): two tensors with identical data under different
  names produce byte-identical blobs, hash to the same address, and are
  stored exactly once.
- Tensor data is copied RAW (byte slices out of the source container,
  located via its header's data_offsets): no dtype round-trip, so
  bf16/fp16/int tensors survive byte-exact and no precision is ever
  silently changed by the import.
- NEVER DUPLICATE BYTES (project design rule): a blob whose hash
  already exists in the bundle's store is never rewritten or copied —
  storing skips the write entirely. Dedup therefore applies at blob
  level (skip-if-exists). Honest limit: per-tensor import
  re-serializes each tensor into a new canonical container, so source
  FILES are never hardlinked into the store (their bytes are not the
  blob's bytes); no hardlink is claimed where none happens. The demo
  base blob left behind by ``si init`` is not deleted either — blobs
  are evidence and the store is append-only; an unreferenced blob is
  inert, not a duplicate of anything.
- Backward compatibility: ``base.blob`` still points at a single blob
  containing a usable tensor (the ``"W"`` tensor's blob when the
  checkpoint has one, else the first tensor's in sorted-name order),
  and ``base.parts`` still records one entry per source file (file
  name, source-file bytes + sha256, and that file's tensor metadata).
  Pre-SI-7 bundles (whole-file ``parts`` with ``blob`` refs, demo
  bundles) still load, verify, and export: ``load_bundle`` verifies
  whichever refs a base actually carries, and ``si.export`` reads
  ``tensor_blobs`` first, then legacy ``parts``, then ``blob``.
"""
from __future__ import annotations

import json
import os
import struct

# Underscore helpers below live in si.bundle -- the same package as this
# module. Reusing them (rather than duplicating the manifest/journal
# anchoring logic) keeps the checkpoint invariant in one place.
from si.bundle import (
    _anchor_checkpoint,
    _write_manifest,
    load_bundle,
    sha256_hex,
    utc_now,
)

#: Canonical key under which every per-tensor blob stores its tensor.
TENSOR_BLOB_KEY = "tensor"

#: Marker recorded in ``base.seed_format`` for per-tensor seeded bases.
SEED_FORMAT = "per-tensor-v1"

#: safetensors dtype code -> manifest dtype name (numpy-style, matching
#: what pre-SI-7 seeds recorded via ``str(np.dtype)``).
_DTYPE_NAMES = {
    "F64": "float64", "F32": "float32", "F16": "float16",
    "BF16": "bfloat16", "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2", "I64": "int64", "I32": "int32",
    "I16": "int16", "I8": "int8", "U64": "uint64", "U32": "uint32",
    "U16": "uint16", "U8": "uint8", "BOOL": "bool",
}


def _dtype_name(code: str) -> str:
    return _DTYPE_NAMES.get(code, str(code).lower())


def _read_header(path: str) -> tuple[dict, int]:
    """Parse a safetensors header directly: ({name: info}, data_start).

    ``info`` carries dtype/shape/data_offsets exactly as written.
    Anything malformed (short file, absurd header length, non-JSON or
    non-object header, malformed tensor entries, offsets outside the
    file) is a ValueError — the same fail-closed contract as the
    safe_open validation that runs before this is trusted.
    """
    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) < 8:
            raise ValueError(f"invalid safetensors file {path}: too short")
        (header_len,) = struct.unpack("<Q", raw_len)
        file_size = os.path.getsize(path)
        if header_len > file_size - 8:
            raise ValueError(
                f"invalid safetensors file {path}: header length "
                f"{header_len} exceeds file size {file_size}")
        header_raw = f.read(header_len)
    try:
        header = json.loads(header_raw.decode("utf-8"))
    except Exception as err:
        raise ValueError(f"invalid safetensors file {path}: {err}") from err
    if not isinstance(header, dict):
        raise ValueError(f"invalid safetensors file {path}: header is not an object")
    data_start = 8 + header_len
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(info, dict) or not isinstance(info.get("data_offsets"), list) \
                or len(info["data_offsets"]) != 2:
            raise ValueError(
                f"invalid safetensors file {path}: malformed entry for {name!r}")
        start, end = info["data_offsets"]
        if not (isinstance(start, int) and isinstance(end, int)
                and 0 <= start <= end and data_start + end <= file_size):
            raise ValueError(
                f"invalid safetensors file {path}: data_offsets for "
                f"{name!r} lie outside the file")
        tensors[name] = info
    return tensors, data_start


def _validate_tensor_file(path: str) -> dict:
    """Validate ``path`` as a real safetensors file and read tensor metadata.

    Opens with ``safetensors.safe_open(framework="np")`` and reads ``keys()``.
    The format is pickle-free, so a successful open IS the safety proof;
    anything else (corrupt header, truncated body, non-safetensors bytes)
    raises ``SafetensorError`` and is wrapped as ``ValueError``.

    Returns {tensor_name: {"shape": [...], "dtype": str}} with dtypes in
    the manifest's numpy-style naming.
    """
    from safetensors import safe_open

    try:
        with safe_open(path, framework="np") as f:
            keys = list(f.keys())
    except Exception as err:
        raise ValueError(f"invalid safetensors file {path}: {err}") from err
    header, _ = _read_header(path)
    if set(keys) != set(header):
        raise ValueError(
            f"invalid safetensors file {path}: header tensors disagree "
            "with safe_open keys")
    return {
        name: {"shape": [int(x) for x in header[name]["shape"]],
               "dtype": _dtype_name(str(header[name]["dtype"]))}
        for name in keys
    }


def _canonical_tensor_blob(dtype_code: str, shape: list, raw: bytes) -> bytes:
    """Bytes of the canonical single-tensor safetensors blob.

    Deterministic in (dtype, shape, raw data): the JSON header has one
    fixed key in a fixed field order, padded with spaces so tensor data
    starts 8-byte aligned (the layout every safetensors writer emits).
    Identical tensors therefore produce byte-identical blobs and share
    one content address, whatever their original names were.
    """
    header = json.dumps(
        {TENSOR_BLOB_KEY: {"dtype": dtype_code, "shape": list(shape),
                           "data_offsets": [0, len(raw)]}},
        separators=(",", ":")).encode("utf-8")
    header += b" " * (-(8 + len(header)) % 8)
    return struct.pack("<Q", len(header)) + header + raw


def _read_tensor_raw(path: str, data_start: int, start: int, end: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(data_start + start)
        raw = f.read(end - start)
    if len(raw) != end - start:
        raise ValueError(f"invalid safetensors file {path}: truncated tensor data")
    return raw


def _store_blob(bundle_dir: str, data: bytes) -> tuple[str, bool]:
    """Store immutable content-addressed bytes; return (hex, existed).

    NEVER DUPLICATE BYTES: when the addressed file already exists the
    write is skipped entirely — the blob is not rewritten or copied.
    """
    hexpart = sha256_hex(data)
    dest = os.path.join(bundle_dir, "blobs", "sha256", hexpart + ".safetensors")
    if os.path.isfile(dest):
        return hexpart, True
    tmp = dest + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dest)
    return hexpart, False


# -- source resolution: single file, plain directory, sharded HF layout ------
def _load_index(src_dir: str) -> dict | None:
    """Parse ``model.safetensors.index.json`` in ``src_dir``, or None.

    A present-but-malformed index fails closed (ValueError): it claims
    to describe the checkpoint's layout, and seeding from a layout we
    cannot trust would mislabel the base.
    """
    index_path = os.path.join(src_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return None
    try:
        with open(index_path, "rb") as f:
            index = json.loads(f.read().decode("utf-8"))
    except Exception as err:
        raise ValueError(f"invalid model.safetensors.index.json: {err}") from err
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict) \
            or not index["weight_map"]:
        raise ValueError(
            "invalid model.safetensors.index.json: missing or empty weight_map")
    weight_map = index["weight_map"]
    for tensor_name, shard in weight_map.items():
        if not isinstance(shard, str) or not shard.endswith(".safetensors") \
                or os.path.basename(shard) != shard:
            raise ValueError(
                f"invalid model.safetensors.index.json: tensor {tensor_name!r} "
                f"maps to invalid shard name {shard!r}")
    return weight_map


def _resolve_source_files(source: str) -> tuple[str, list[str], dict | None]:
    """Return (abspath_of_source, sorted .safetensors paths, weight_map).

    ``source`` is one .safetensors file, or a directory. A directory
    carrying ``model.safetensors.index.json`` is a sharded checkpoint:
    exactly the shards named by its weight_map are the source files.
    Any other directory contributes all of its .safetensors files,
    sorted by name. ``weight_map`` is None for unsharded sources.
    """
    src_abs = os.path.abspath(source)
    if os.path.isdir(src_abs):
        weight_map = _load_index(src_abs)
        if weight_map is not None:
            files = []
            for shard in sorted(set(weight_map.values())):
                shard_path = os.path.join(src_abs, shard)
                if not os.path.isfile(shard_path):
                    raise ValueError(
                        f"sharded checkpoint is missing shard {shard!r} "
                        f"named by model.safetensors.index.json in {src_abs}")
                files.append(shard_path)
            return src_abs, files, weight_map
        files = sorted(
            os.path.join(src_abs, name)
            for name in os.listdir(src_abs)
            if name.endswith(".safetensors")
            and os.path.isfile(os.path.join(src_abs, name))
        )
        if not files:
            raise ValueError(
                f"no .safetensors files found in directory {src_abs}"
            )
        return src_abs, files, None
    if os.path.isfile(src_abs):
        if not src_abs.endswith(".safetensors"):
            raise ValueError(
                f"seed source must be a .safetensors file or a directory, "
                f"got {src_abs}"
            )
        return src_abs, [src_abs], None
    raise FileNotFoundError(f"seed source not found: {src_abs}")


def _validate_shard_layout(weight_map: dict, per_file: dict) -> None:
    """Fail closed unless the shards exactly match the weight_map.

    Every mapped tensor must exist in its shard; every tensor found in
    a shard must be claimed by the map for THAT shard — so a tensor
    present in two shards, or present but unmapped, is a ValueError.
    """
    for tensor_name, shard in weight_map.items():
        shard_tensors = per_file.get(shard, {})
        if tensor_name not in shard_tensors:
            raise ValueError(
                f"sharded checkpoint: tensor {tensor_name!r} is mapped to "
                f"{shard!r} by model.safetensors.index.json but is not in "
                "that shard")
    for shard, tensors in per_file.items():
        for tensor_name in tensors:
            owner = weight_map.get(tensor_name)
            if owner is None:
                raise ValueError(
                    f"sharded checkpoint: tensor {tensor_name!r} is present "
                    f"in shard {shard!r} but missing from "
                    "model.safetensors.index.json weight_map")
            if owner != shard:
                raise ValueError(
                    f"sharded checkpoint: tensor {tensor_name!r} is claimed "
                    f"by two shards: present in {shard!r}, mapped to "
                    f"{owner!r}")


def _capture_checkpoint_config(src_abs: str, files: list[str]) -> dict | None:
    """Capture the checkpoint's config.json when one sits beside the weights.

    Looks for ``config.json`` in the source directory (or next to a
    single-file source). Returns {"config": dict, "sha256": hex} or
    None when absent. A present-but-unparseable config is a ValueError:
    ``si export`` derives GGUF architecture metadata from this file, and
    a corrupt config must fail closed at seed time rather than mislabel
    an export later.
    """
    if os.path.isdir(src_abs):
        cfg_path = os.path.join(src_abs, "config.json")
    else:
        cfg_path = os.path.join(os.path.dirname(src_abs), "config.json")
    if not os.path.isfile(cfg_path):
        return None
    with open(cfg_path, "rb") as f:
        raw = f.read()
    try:
        cfg = json.loads(raw.decode("utf-8"))
    except Exception as err:
        raise ValueError(f"invalid config.json at {cfg_path}: {err}") from err
    if not isinstance(cfg, dict):
        raise ValueError(
            f"config.json at {cfg_path} is not a JSON object; refusing to seed"
        )
    return {"config": cfg, "sha256": sha256_hex(raw)}


def _is_same_checkpoint(base: dict, tensor_blobs: dict,
                        ckpt_cfg: dict | None) -> bool:
    """True when the bundle's base already IS exactly this checkpoint.

    Identity is content identity: the same tensor names, each with the
    same shape/dtype and the same content-addressed blob ref, plus the
    same config.json sha256 (or both sides having none). Only
    per-tensor-format bases can match — a legacy whole-file base is
    re-seeded into the new format rather than compared approximately.
    """
    if base.get("seed_format") != SEED_FORMAT:
        return False
    existing = base.get("tensor_blobs")
    if not isinstance(existing, dict) or set(existing) != set(tensor_blobs):
        return False
    for name, ref in tensor_blobs.items():
        cur = existing[name]
        if cur.get("blob") != ref["blob"] or cur.get("bytes") != ref["bytes"] \
                or cur.get("shape") != ref["shape"] \
                or cur.get("dtype") != ref["dtype"]:
            return False
    existing_cfg_sha = base.get("checkpoint_config_sha256")
    planned_cfg_sha = ("sha256:" + ckpt_cfg["sha256"]) if ckpt_cfg else None
    return existing_cfg_sha == planned_cfg_sha


def seed_bundle(bundle_path: str, source: str, importer: str = "operator") -> dict:
    """Import a REAL safetensors checkpoint as the bundle's frozen base.

    ``source`` is one .safetensors file, a directory of them (all used,
    sorted by name), or a sharded checkpoint directory carrying
    ``model.safetensors.index.json``. Every tensor is imported as its
    own content-addressed blob (see the module docstring for the format
    and the never-duplicate-bytes rule); ``manifest["base"]`` records
    the per-tensor refs, ``seeded_from`` provenance is journaled, and
    the return is ``{"base", "parts", "tensors", "bytes",
    "deduplicated_bytes", "already_seeded", "head"}`` where ``parts``
    is the number of source files, ``bytes`` is the total bytes of the
    UNIQUE blobs representing the base in the store, and
    ``deduplicated_bytes`` counts bytes not written because their blob
    already existed.

    Fail-closed: ``load_bundle`` runs first (any tamper raises);
    bundles with existing history are refused with ``ValueError`` —
    except the idempotent re-seed of the exact checkpoint already
    serving as the base, which writes nothing and reports
    ``already_seeded=True``.
    """
    from si.journal import open_journal  # local import: no cycle

    # 1. Load + verify first: fail closed on any tamper before touching
    #    anything.
    bundle_dir = os.path.abspath(bundle_path)
    manifest = load_bundle(bundle_dir)

    # 2. Resolve + validate the checkpoint fully and plan the import
    #    BEFORE the history refusal, so an idempotent re-seed (step 3)
    #    can be recognized without writing anything.
    src_abs, files, weight_map = _resolve_source_files(source)
    per_file: dict[str, dict] = {}   # basename -> {tensor: meta}
    headers: dict[str, tuple[dict, int]] = {}
    merged_tensors: dict = {}
    for path in files:
        tensors = _validate_tensor_file(path)
        basename = os.path.basename(path)
        if weight_map is None:
            for name in tensors:
                if name in merged_tensors:
                    raise ValueError(
                        f"duplicate tensor name {name!r} across checkpoint "
                        f"files ({basename}); refusing to shadow a tensor")
        # Sharded sources defer cross-file duplicates to
        # _validate_shard_layout below, which names the actual fault
        # (a tensor claimed by / present in two shards).
        per_file[basename] = tensors
        headers[path] = _read_header(path)
        merged_tensors.update(tensors)
    if weight_map is not None:
        _validate_shard_layout(weight_map, per_file)

    ckpt_cfg = _capture_checkpoint_config(src_abs, files)

    # Plan every tensor's canonical blob from its raw bytes. Hashing is
    # done here; the store writes happen only after the refusals below.
    tensor_blobs: dict = {}
    planned: dict[str, bytes] = {}   # tensor name -> canonical blob bytes
    for path in files:
        header, data_start = headers[path]
        for name in sorted(per_file[os.path.basename(path)]):
            info = header[name]
            start, end = info["data_offsets"]
            raw = _read_tensor_raw(path, data_start, start, end)
            blob_bytes = _canonical_tensor_blob(
                str(info["dtype"]), [int(x) for x in info["shape"]], raw)
            planned[name] = blob_bytes
            tensor_blobs[name] = {
                "blob": "sha256:" + sha256_hex(blob_bytes),
                "bytes": len(blob_bytes),
                "shape": [int(x) for x in info["shape"]],
                "dtype": _dtype_name(str(info["dtype"])),
            }

    # 3. Idempotent re-seed: the base already IS this checkpoint.
    #    Nothing to write, nothing to journal — report it honestly.
    if _is_same_checkpoint(manifest["base"], tensor_blobs, ckpt_cfg):
        total_logical = sum(len(b) for b in planned.values())
        return {"base": manifest["base"], "parts": len(files),
                "tensors": len(tensor_blobs),
                "bytes": manifest["base"].get("bytes", 0),
                "deduplicated_bytes": total_logical,
                "already_seeded": True, "head": manifest["head"]}

    # 4. Refuse bundles that already have history: seeding would rewrite
    #    the mind's frozen origin over a lived life.
    if manifest.get("deltas") or manifest.get("head") != "base":
        raise ValueError(
            "refusing to seed a bundle with existing history "
            f"(deltas={len(manifest.get('deltas', []))}, "
            f"head={manifest.get('head')!r})"
        )

    # 5. Store the blobs, deduplicating globally within the store: a
    #    blob that already exists (an identical tensor in this same
    #    checkpoint, or one stored by an earlier seed) is never
    #    rewritten or copied; its bytes count as deduplicated.
    deduplicated_bytes = 0
    stored_refs: set = set()
    for name in sorted(planned):
        blob_bytes = planned[name]
        hexpart, existed = _store_blob(bundle_dir, blob_bytes)
        assert "sha256:" + hexpart == tensor_blobs[name]["blob"]
        if existed:
            deduplicated_bytes += len(blob_bytes)
        stored_refs.add(hexpart)
    stored_bytes = 0
    for hexpart in stored_refs:
        stored_bytes += os.path.getsize(
            os.path.join(bundle_dir, "blobs", "sha256",
                         hexpart + ".safetensors"))
    total_logical = sum(len(b) for b in planned.values())

    # 6. Rewrite the base. "blob" keeps pointing at a single usable
    #    blob (the "W" tensor's when present — the DemoMind/loop compat
    #    path — else the first tensor in sorted order) and "parts"
    #    keeps one provenance entry per source file, so legacy readers
    #    keep working; "tensor_blobs" is the canonical per-tensor map.
    parts = []
    for path in files:
        basename = os.path.basename(path)
        with open(path, "rb") as f:
            source_sha = sha256_hex(f.read())
        parts.append({
            "file": basename,
            "bytes": os.path.getsize(path),
            "sha256": "sha256:" + source_sha,
            "tensors": per_file[basename],
        })
    compat_name = "W" if "W" in tensor_blobs else sorted(tensor_blobs)[0]
    utc = utc_now()
    manifest["base"] = {
        "source": f"seed:{src_abs} (imported {utc})",
        "format": "safetensors",
        "seed_format": SEED_FORMAT,
        "blob": tensor_blobs[compat_name]["blob"],
        "parts": parts,
        "tensor_blobs": tensor_blobs,
        "tensors": merged_tensors,
        "bytes": stored_bytes,
        "imported_bytes": total_logical,
        "deduplicated_bytes": deduplicated_bytes,
        "tensor_count": len(tensor_blobs),
        "importer": importer,
        "imported_at": utc,
    }
    if weight_map is not None:
        manifest["base"]["sharded"] = True
        manifest["base"]["weight_map"] = dict(weight_map)
    if ckpt_cfg is not None:
        manifest["base"]["checkpoint_config"] = ckpt_cfg["config"]
        manifest["base"]["checkpoint_config_sha256"] = (
            "sha256:" + ckpt_cfg["sha256"]
        )
        manifest["base"]["arch_source"] = "checkpoint-config"

    # 7. Provenance + re-anchor: the checkpoint must be the journal tip
    #    so the manifest anchor matches on the next load. The note
    #    carries the full accounting the project requires: tensor
    #    count, total bytes, deduplicated bytes, source path, and the
    #    config sha256 when the checkpoint shipped one.
    journal = open_journal(bundle_dir, manifest)
    try:
        note = (
            f"imported {len(parts)} tensor file(s), {len(tensor_blobs)} "
            f"tensors, {total_logical} bytes via si seed "
            f"(stored {stored_bytes} unique blob bytes, deduplicated "
            f"{deduplicated_bytes} bytes); source={src_abs}"
        )
        if ckpt_cfg is not None:
            note += f"; config_sha256=sha256:{ckpt_cfg['sha256']}"
        journal.record_provenance(
            child="base",
            parents=[src_abs],
            relation="seeded_from",
            note=note,
        )
        src_label = os.path.basename(src_abs)
        manifest = _anchor_checkpoint(journal, manifest, head="base",
                                      note=f"seed {src_label}")
        _write_manifest(bundle_dir, manifest)
    finally:
        journal.close()

    return {"base": manifest["base"], "parts": len(parts),
            "tensors": len(tensor_blobs), "bytes": stored_bytes,
            "deduplicated_bytes": deduplicated_bytes,
            "already_seeded": False, "head": "base"}
