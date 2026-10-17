"""``si seed``: import a REAL safetensors checkpoint as a bundle's frozen base.

Replaces the honest-but-random demo seed written by ``si init`` with real
weights, so the self-write loop can run against a genuine checkpoint.

Safety story (this is the whole point of the format check):
- safetensors is inherently pickle-free: there is no deserialization
  gadgetry to exploit. A file that ``safetensors.safe_open`` opens and
  reads keys from IS a valid safetensors container, and that successful
  open is the safety proof. Corrupt headers, truncated files, and
  non-safetensors bytes all raise and are wrapped as ``ValueError``.
- The base is content-addressed: every part is SHA-256'd and stored under
  ``blobs/sha256/<hex>.safetensors``. ``load_bundle`` re-verifies the hash
  of every referenced blob on every load, so a tampered part fails closed.
- ``si seed`` refuses any bundle that already has history (non-empty
  deltas or head != "base"): the base is frozen, and seeding over a lived
  life would silently rewrite the mind's origin. Fail closed instead.
- The import is journaled (provenance ``seeded_from``) and the manifest is
  re-anchored to the new journal tip, so ``load_bundle`` verifies cleanly
  afterwards.
- When a ``config.json`` sits beside the weights, it is stored in the
  manifest (``base.checkpoint_config`` + ``arch_source="checkpoint-config"``)
  so ``si export`` can write real GGUF architecture metadata. Without it,
  export falls back to conservative tensor-shape inference.
"""
from __future__ import annotations

import os

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


def _validate_tensor_file(path: str) -> dict:
    """Validate ``path`` as a real safetensors file and read tensor metadata.

    Opens with ``safetensors.safe_open(framework="np")`` and reads ``keys()``.
    The format is pickle-free, so a successful open IS the safety proof;
    anything else (corrupt header, truncated body, non-safetensors bytes)
    raises ``SafetensorError`` and is wrapped as ``ValueError``.

    Returns {tensor_name: {"shape": [...], "dtype": str}}. Shapes/dtypes
    come from the header-driven numpy views -- no full tensor data is
    copied or loaded.
    """
    from safetensors import safe_open

    try:
        with safe_open(path, framework="np") as f:
            info = {}
            for name in f.keys():
                t = f.get_tensor(name)
                info[name] = {"shape": list(t.shape), "dtype": str(t.dtype)}
    except Exception as err:
        raise ValueError(f"invalid safetensors file {path}: {err}") from err
    return info


def _resolve_source_files(source: str) -> tuple[str, list[str]]:
    """Return (abspath_of_source, sorted list of .safetensors file paths).

    ``source`` is either one .safetensors file or a directory containing
    .safetensors files (all of them, sorted by name).
    """
    src_abs = os.path.abspath(source)
    if os.path.isdir(src_abs):
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
        return src_abs, files
    if os.path.isfile(src_abs):
        if not src_abs.endswith(".safetensors"):
            raise ValueError(
                f"seed source must be a .safetensors file or a directory, "
                f"got {src_abs}"
            )
        return src_abs, [src_abs]
    raise FileNotFoundError(f"seed source not found: {src_abs}")


def _capture_checkpoint_config(src_abs: str, files: list[str]) -> dict | None:
    """Capture the checkpoint's config.json when one sits beside the weights.

    Looks for ``config.json`` in the source directory (or next to a
    single-file source). Returns the parsed dict, or None when absent.
    A present-but-unparseable config is a ValueError: ``si export``
    derives GGUF architecture metadata from this file, and a corrupt
    config must fail closed at seed time rather than mislabel an export
    later.
    """
    import json

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


def _store_blob(bundle_dir: str, data: bytes) -> str:
    """Store immutable content-addressed bytes; return the hex digest.

    Skips the copy when the addressed file already exists (content
    addressing: identical bytes already have an identical address).
    """
    hexpart = sha256_hex(data)
    dest = os.path.join(bundle_dir, "blobs", "sha256", hexpart + ".safetensors")
    if not os.path.isfile(dest):
        tmp = dest + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, dest)
    return hexpart


def seed_bundle(bundle_path: str, source: str, importer: str = "operator") -> dict:
    """Import a REAL safetensors checkpoint as the bundle's frozen base.

    ``source`` is one .safetensors file or a directory of them (all used,
    sorted by name). Rewrites ``manifest["base"]`` to point at the imported
    parts, records ``seeded_from`` provenance, re-anchors the journal
    checkpoint, and returns
    ``{"base": manifest["base"], "parts": n, "bytes": total, "head": "base"}``.

    Fail-closed: ``load_bundle`` runs first (any tamper raises); bundles
    with existing history are refused with ``ValueError``.
    """
    from si.journal import open_journal  # local import: no cycle

    # 1. Load + verify first: fail closed on any tamper before touching
    #    anything.
    bundle_dir = os.path.abspath(bundle_path)
    manifest = load_bundle(bundle_dir)

    # 2. Refuse bundles that already have history: seeding would rewrite the
    #    mind's frozen origin over a lived life.
    if manifest.get("deltas") or manifest.get("head") != "base":
        raise ValueError(
            "refusing to seed a bundle with existing history "
            f"(deltas={len(manifest.get('deltas', []))}, "
            f"head={manifest.get('head')!r})"
        )

    # 3. Resolve + validate every tensor file (safe_open is the safety
    #    proof; failures become ValueError).
    src_abs, files = _resolve_source_files(source)
    parts = []
    total = 0
    merged_tensors: dict = {}
    for path in files:
        tensors = _validate_tensor_file(path)
        with open(path, "rb") as f:
            data = f.read()
        hexpart = _store_blob(bundle_dir, data)
        total += len(data)
        parts.append({
            "file": os.path.basename(path),
            "blob": "sha256:" + hexpart,
            "bytes": len(data),
            "tensors": tensors,
        })
        merged_tensors.update(tensors)

    # 4. Rewrite the base. "blob" keeps pointing at the FIRST part so
    #    existing single-blob loaders (DemoMind, LoopContext) keep working.
    #    When the checkpoint ships a config.json beside the weights, store
    #    it: `si export` derives real GGUF architecture metadata from it
    #    (arch_source="checkpoint-config"). No config -> export falls back
    #    to conservative tensor-shape inference.
    utc = utc_now()
    manifest["base"] = {
        "source": f"seed:{src_abs} (imported {utc})",
        "format": "safetensors",
        "blob": parts[0]["blob"],
        "parts": parts,
        "tensors": merged_tensors,
        "bytes": total,
        "importer": importer,
        "imported_at": utc,
    }
    ckpt_cfg = _capture_checkpoint_config(src_abs, files)
    if ckpt_cfg is not None:
        manifest["base"]["checkpoint_config"] = ckpt_cfg["config"]
        manifest["base"]["checkpoint_config_sha256"] = (
            "sha256:" + ckpt_cfg["sha256"]
        )
        manifest["base"]["arch_source"] = "checkpoint-config"

    # 5. Provenance + re-anchor: the checkpoint must be the journal tip so
    #    the manifest anchor matches on the next load.
    journal = open_journal(bundle_dir, manifest)
    try:
        journal.record_provenance(
            child="base",
            parents=[src_abs],
            relation="seeded_from",
            note=f"imported {len(parts)} tensor file(s), {total} bytes via si seed",
        )
        src_label = os.path.basename(src_abs)
        manifest = _anchor_checkpoint(journal, manifest, head="base",
                                      note=f"seed {src_label}")
        _write_manifest(bundle_dir, manifest)
    finally:
        journal.close()

    return {"base": manifest["base"], "parts": len(parts),
            "bytes": total, "head": "base"}
