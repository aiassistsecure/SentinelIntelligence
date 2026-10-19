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

SI-8 adds a BY-REFERENCE seed mode (``seed_bundle(...,
by_reference=True)`` / ``si seed --by-reference``) for checkpoints too
large to copy into the store (the 55.46 GiB Muse Glimmer 30B seed:
copy mode would need ~59.5 GB of store on a volume without it, and
its planning pass held every tensor's blob bytes in RAM). In this
mode the bundle stores ZERO new tensor bytes. The manifest records
``seed_format="reference-v1"`` and ``base.tensor_refs``: per tensor,
the source file's absolute path, the tensor's absolute byte offset
and length within that file, dtype (name + safetensors code), shape,
and a content hash. The hash reuses COPY MODE's identity exactly: it
is the SHA-256 of the tensor's canonical blob bytes — but computed
STREAMING (canonical prefix, then the raw byte range in bounded
chunks), so neither a whole shard nor all tensors are ever held in
RAM. Same checkpoint, same ``blob`` refs in either mode; only the
storage differs, and the manifest says so honestly (``bytes`` /
``stored_bytes`` are 0; ``base.blob`` is a virtual ref in this mode —
no such file exists in the store, and nothing claims otherwise).
Verification fails closed at the same strength as copy mode:
``load_bundle`` re-checks every referenced source file (existence,
exact size, full streaming SHA-256) and every tensor's streaming
canonical hash on every load; reads through ``load_base_tensor`` /
``si export`` re-verify the tensor they read. The dependency is the
design's price, stated plainly: a by-reference base DEPENDS on the
source checkpoint remaining at its recorded absolute path — moving
or deleting it breaks the base (``load_bundle`` raises TamperError),
by design of never duplicating bytes. Copy mode remains the choice
for portable bundles. Copy mode also gains two hardenings here: its
source-file hashes are computed streaming, and a fail-closed disk
preflight refuses the import BEFORE writing anything when the unique
new blob bytes clearly cannot fit in the free space at the bundle.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct

# Underscore helpers below live in si.bundle -- the same package as this
# module. Reusing them (rather than duplicating the manifest/journal
# anchoring logic) keeps the checkpoint invariant in one place.
from si.bundle import (
    REF_SEED_FORMAT,
    STREAM_CHUNK,
    _anchor_checkpoint,
    _write_manifest,
    canonical_tensor_prefix,
    load_bundle,
    sha256_file,
    sha256_hex,
    utc_now,
)

#: Canonical key under which every per-tensor blob stores its tensor.
TENSOR_BLOB_KEY = "tensor"

#: Marker recorded in ``base.seed_format`` for per-tensor seeded bases.
SEED_FORMAT = "per-tensor-v1"

#: Marker recorded in ``base.seed_format`` for by-reference seeded
#: bases (SI-8). The constant itself lives in si.bundle (see
#: REF_SEED_FORMAT there) because load_bundle branches on it.
REFERENCE_SEED_FORMAT = REF_SEED_FORMAT

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
    # Built through si.bundle.canonical_tensor_prefix so the in-RAM
    # blob and the streaming by-reference hash share ONE construction.
    return canonical_tensor_prefix(dtype_code, list(shape), len(raw)) + raw


def _hash_canonical_tensor_streaming(path: str, data_start: int,
                                     start: int, end: int,
                                     dtype_code: str,
                                     shape: list) -> tuple[str, int]:
    """(canonical blob sha256 hex, canonical blob byte length) for one
    tensor, computed by streaming its raw byte range in bounded chunks.

    Identity is EXACTLY copy mode's: the hash covers the canonical
    prefix + raw bytes, i.e. the very bytes copy mode would store.
    Memory stays bounded no matter how large the tensor (or the shard
    containing it) is.
    """
    raw_len = end - start
    h = hashlib.sha256()
    prefix = canonical_tensor_prefix(dtype_code, list(shape), raw_len)
    h.update(prefix)
    remaining = raw_len
    with open(path, "rb") as f:
        f.seek(data_start + start)
        while remaining > 0:
            chunk = f.read(min(STREAM_CHUNK, remaining))
            if not chunk:
                raise ValueError(
                    f"invalid safetensors file {path}: truncated tensor data")
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest(), len(prefix) + raw_len


def _disk_free(path: str) -> int:
    """Free bytes at ``path``'s filesystem. A named seam so the copy-mode
    preflight (and its tests) have exactly one place to read/override."""
    return shutil.disk_usage(path).free


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


def _is_same_checkpoint_identity(base: dict, identity: dict,
                                 ckpt_cfg: dict | None) -> bool:
    """True when the bundle's base already IS exactly this checkpoint.

    Identity is content identity: the same tensor names, each with the
    same shape/dtype and the same canonical content-addressed blob ref,
    plus the same config.json sha256 (or both sides having none).
    Mode-independent since SI-8 (``identity`` carries canonical refs,
    identical in copy and by-reference mode): an existing base matches
    via ``tensor_blobs`` (copy) or ``tensor_refs`` (reference), which
    is what makes a mixed-mode re-seed of the SAME checkpoint an
    honest idempotent no-op rather than a silent storage-mode
    conversion — converting modes is not supported implicitly;
    re-seeding identical content writes nothing whichever mode is
    requested, and a mode change needs a fresh bundle. A legacy
    whole-file base never matches (no per-tensor refs to compare
    exactly) — it is re-seeded into a current format instead.
    """
    fmt = base.get("seed_format")
    if fmt == SEED_FORMAT:
        existing = base.get("tensor_blobs")
    elif fmt == REFERENCE_SEED_FORMAT:
        existing = base.get("tensor_refs")
    else:
        return False
    if not isinstance(existing, dict) or set(existing) != set(identity):
        return False
    for name, ref in identity.items():
        cur = existing[name]
        if cur.get("blob") != ref["blob"] or cur.get("bytes") != ref["bytes"] \
                or cur.get("shape") != ref["shape"] \
                or cur.get("dtype") != ref["dtype"]:
            return False
    existing_cfg_sha = base.get("checkpoint_config_sha256")
    planned_cfg_sha = ("sha256:" + ckpt_cfg["sha256"]) if ckpt_cfg else None
    return existing_cfg_sha == planned_cfg_sha


def seed_bundle(bundle_path: str, source: str, importer: str = "operator",
                by_reference: bool = False) -> dict:
    """Import a REAL safetensors checkpoint as the bundle's frozen base.

    ``source`` is one .safetensors file, a directory of them (all used,
    sorted by name), or a sharded checkpoint directory carrying
    ``model.safetensors.index.json``. In copy mode (the default) every
    tensor is imported as its own content-addressed blob (see the
    module docstring for the format and the never-duplicate-bytes
    rule); ``manifest["base"]`` records the per-tensor refs,
    ``seeded_from`` provenance is journaled, and the return is
    ``{"base", "parts", "tensors", "bytes", "deduplicated_bytes",
    "already_seeded", "head", ...}`` where ``parts`` is the number of
    source files, ``bytes`` is the total bytes of the UNIQUE blobs
    representing the base in the store, and ``deduplicated_bytes``
    counts bytes not written because their blob already existed.

    With ``by_reference=True`` (SI-8) NO tensor bytes are stored: the
    manifest records per-tensor external references
    (``base.tensor_refs``, ``seed_format="reference-v1"``) with the
    same canonical content hashes copy mode would have stored, all
    computed streaming. The return then reports ``bytes`` /
    ``stored_bytes`` as 0 (the real number), ``deduplicated_bytes`` as
    the full logical size (every byte not copied), and
    ``seed_format="reference-v1"``. The base depends on the source
    checkpoint remaining at its recorded path — see the module
    docstring.

    Checkpoint identity is mode-independent (canonical hashes), so
    re-seeding the exact checkpoint already serving as the base — in
    EITHER mode, requested in either mode — is an idempotent no-op
    that writes nothing and reports ``already_seeded=True``; it never
    silently converts the base's storage mode.

    Fail-closed: ``load_bundle`` runs first (any tamper raises);
    bundles with existing history are refused with ``ValueError`` —
    except the idempotent re-seed above. Copy mode additionally
    refuses up front when the unique new blob bytes cannot fit in the
    free space at the bundle location.
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

    # Per-file provenance is computed ONCE, streaming (never a whole
    # source file in RAM — a shard can be ~50 GB), and shared by both
    # modes: copy mode records it as before; by-reference mode's
    # load-time verification checks source files against exactly these
    # sizes + hashes.
    parts = []
    for path in files:
        basename = os.path.basename(path)
        parts.append({
            "file": basename,
            "bytes": os.path.getsize(path),
            "sha256": "sha256:" + sha256_file(path),
            "tensors": per_file[basename],
        })

    # Plan the checkpoint's content identity. In by-reference mode the
    # plan is pure metadata + streaming hashes (bounded memory at any
    # checkpoint size). In copy mode the canonical blob bytes are also
    # materialized, as before SI-8 (copy mode's checkpoints are the
    # ones that fit the store; SI-8 does not change its storage path).
    tensor_blobs: dict = {}   # copy mode: name -> blob ref record
    tensor_refs: dict = {}    # reference mode: name -> external ref
    identity: dict = {}       # both modes: name -> {blob,bytes,shape,dtype}
    planned: dict[str, bytes] = {}   # copy mode only: name -> blob bytes
    for path in files:
        header, data_start = headers[path]
        for name in sorted(per_file[os.path.basename(path)]):
            info = header[name]
            start, end = info["data_offsets"]
            shape = [int(x) for x in info["shape"]]
            dtype = _dtype_name(str(info["dtype"]))
            if by_reference:
                hexpart, canonical_len = _hash_canonical_tensor_streaming(
                    path, data_start, start, end, str(info["dtype"]), shape)
                identity[name] = {
                    "blob": "sha256:" + hexpart, "bytes": canonical_len,
                    "shape": shape, "dtype": dtype,
                }
                tensor_refs[name] = {
                    **identity[name],
                    "dtype_code": str(info["dtype"]),
                    "source_file": path,
                    "file_offset": data_start + start,
                    "byte_length": end - start,
                }
            else:
                raw = _read_tensor_raw(path, data_start, start, end)
                blob_bytes = _canonical_tensor_blob(
                    str(info["dtype"]), shape, raw)
                planned[name] = blob_bytes
                tensor_blobs[name] = {
                    "blob": "sha256:" + sha256_hex(blob_bytes),
                    "bytes": len(blob_bytes),
                    "shape": shape,
                    "dtype": dtype,
                }
                identity[name] = tensor_blobs[name]
    total_logical = sum(ref["bytes"] for ref in identity.values())

    # 3. Idempotent re-seed: the base already IS this checkpoint, in
    #    either storage mode (identity is mode-independent). Nothing
    #    to write, nothing to journal — report it honestly, and never
    #    silently convert the base's storage mode.
    if _is_same_checkpoint_identity(manifest["base"], identity, ckpt_cfg):
        return {"base": manifest["base"], "parts": len(files),
                "tensors": len(identity),
                "bytes": manifest["base"].get("bytes", 0),
                "stored_bytes": manifest["base"].get("bytes", 0),
                "deduplicated_bytes": total_logical,
                "seed_format": manifest["base"].get("seed_format"),
                "already_seeded": True, "head": manifest["head"]}

    # 4. Refuse bundles that already have history: seeding would rewrite
    #    the mind's frozen origin over a lived life.
    if manifest.get("deltas") or manifest.get("head") != "base":
        raise ValueError(
            "refusing to seed a bundle with existing history "
            f"(deltas={len(manifest.get('deltas', []))}, "
            f"head={manifest.get('head')!r})"
        )

    compat_name = "W" if "W" in identity else sorted(identity)[0]
    utc = utc_now()

    if by_reference:
        # 5R. Store NOTHING: the manifest records references only.
        #     stored_bytes is really 0; every logical byte counts as
        #     not-copied (deduplicated) — the design rule's end state.
        stored_bytes = 0
        deduplicated_bytes = total_logical
        manifest["base"] = {
            "source": f"seed:{src_abs} (imported {utc})",
            "format": "safetensors",
            "seed_format": REFERENCE_SEED_FORMAT,
            "storage": "by-reference",
            "blob": identity[compat_name]["blob"],
            "parts": parts,
            "tensor_refs": tensor_refs,
            "tensors": merged_tensors,
            "bytes": 0,
            "stored_bytes": 0,
            "imported_bytes": total_logical,
            "referenced_bytes": sum(
                ref["byte_length"] for ref in tensor_refs.values()),
            "deduplicated_bytes": deduplicated_bytes,
            "tensor_count": len(tensor_refs),
            "importer": importer,
            "imported_at": utc,
        }
    else:
        # 5C. Disk preflight (fail closed, BEFORE writing anything):
        #     the unique blob bytes not already in the store must fit
        #     in the free space at the bundle location.
        unique_planned: dict[str, int] = {}
        for name in sorted(planned):
            unique_planned.setdefault(
                tensor_blobs[name]["blob"], len(planned[name]))
        required_bytes = 0
        for blob_ref, size in unique_planned.items():
            hexpart = blob_ref.split(":", 1)[1]
            if not os.path.isfile(os.path.join(
                    bundle_dir, "blobs", "sha256",
                    hexpart + ".safetensors")):
                required_bytes += size
        free_bytes = _disk_free(bundle_dir)
        if required_bytes > free_bytes:
            raise ValueError(
                f"insufficient disk space for copy-mode seed: the "
                f"checkpoint needs {required_bytes} new blob bytes in "
                f"the bundle store but only {free_bytes} bytes are "
                f"free at {bundle_dir}; refusing before writing "
                f"anything (use --by-reference to seed without "
                f"copying tensor bytes)")

        # 6C. Store the blobs, deduplicating globally within the
        #     store: a blob that already exists (an identical tensor
        #     in this same checkpoint, or one stored by an earlier
        #     seed) is never rewritten or copied; its bytes count as
        #     deduplicated.
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

        # 7C. Rewrite the base. "blob" keeps pointing at a single
        #     usable blob (the "W" tensor's when present — the
        #     DemoMind/loop compat path — else the first tensor in
        #     sorted order) and "parts" keeps one provenance entry per
        #     source file, so legacy readers keep working;
        #     "tensor_blobs" is the canonical per-tensor map.
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

    # 8. Provenance + re-anchor: the checkpoint must be the journal tip
    #    so the manifest anchor matches on the next load. The note
    #    carries the full accounting the project requires: tensor
    #    count, total bytes, stored bytes, deduplicated/not-copied
    #    bytes, source path, and the config sha256 when present.
    journal = open_journal(bundle_dir, manifest)
    try:
        if by_reference:
            note = (
                f"imported {len(parts)} tensor file(s), "
                f"{len(tensor_refs)} tensors, {total_logical} bytes "
                f"via si seed --by-reference (stored {stored_bytes} "
                f"tensor bytes in the bundle store, deduplicated "
                f"{deduplicated_bytes} bytes not copied; tensors are "
                f"referenced in place and the base depends on the "
                f"source remaining at its recorded path); "
                f"source={src_abs}"
            )
        else:
            note = (
                f"imported {len(parts)} tensor file(s), "
                f"{len(identity)} tensors, {total_logical} bytes via "
                f"si seed (stored {stored_bytes} unique blob bytes, "
                f"deduplicated {deduplicated_bytes} bytes); "
                f"source={src_abs}"
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
            "tensors": len(identity), "bytes": stored_bytes,
            "stored_bytes": stored_bytes,
            "deduplicated_bytes": deduplicated_bytes,
            "seed_format": manifest["base"]["seed_format"],
            "already_seeded": False, "head": "base"}
