"""``si export``: materialize the head and write it out as a GGUF file.

The mind's learned weights are exportable, not just inspectable: the
frozen base plus every active LoRA-style delta (W = W + scale*(B@A))
materialized into full float32 tensors, then written as F16 GGUF with
``si.*`` metadata (bundle id, head, base source). An optional quant step
delegates to a REAL ``llama-quantize`` binary — never faked.

Determinism is a hard requirement: the same head produces a
byte-identical GGUF. That means no wall-clock timestamps anywhere in the
file, tensors written sorted by name, and ``si.bundle_id`` anchored to
head content rather than export-time journal state (the export itself
journals, which moves the manifest's journal anchor — anchoring the id
to that would make consecutive exports differ).

Delta -> tensor resolution (honest, documented):
- Single-tensor base (the demo seed's ``W``): the delta applies to that
  tensor, whatever the module string says.
- Multi-tensor base (a seeded real checkpoint): a delta's module string
  (e.g. ``"demo_linear"``) resolves to every base tensor whose NAME
  contains the module string as a substring.
- No match, or a (B@A) result whose shape does not equal the target
  tensor's shape, is a ValueError. Export never silently misapplies a
  delta to the wrong tensor.

The export is NOT gated and does NOT move the head: it reads the
current head, writes a file, and journals ``relation="export"``
provenance for the produced artifact. No delta, no self-edit.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess

from si.bundle import (
    _anchor_checkpoint,
    _write_manifest,
    load_bundle,
    manifest_id,
    sha256_hex,
    utc_now,
)


# -- delta resolution -------------------------------------------------------
def _load_base_tensors(bundle_dir: str, manifest: dict) -> dict:
    """Load ALL base tensors as float32 numpy arrays, keyed by name.

    Handles both the demo single-blob base (``base["blob"]``) and seeded
    multi-part bases (``base["parts"]``). Duplicate tensor names across
    parts fail closed with ValueError rather than silently shadowing.
    Pure read: no journal, no manifest writes.
    """
    import numpy as np
    from safetensors import safe_open

    base = manifest["base"]
    if "parts" in base and base["parts"]:
        blob_refs = [p["blob"] for p in base["parts"]]
    else:
        blob_refs = [base["blob"]]

    tensors: dict = {}
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


# -- materialization ---------------------------------------------------------
def materialize_head(bundle_dir: str, manifest: dict | None = None) -> dict:
    """Materialize the current head into full float32 weight tensors.

    Pure read. Loads every base tensor plus the active delta stack (manifest
    deltas up to and including ``head``, in manifest order) and applies
    ``W = W + scale*(B@A)`` per delta on torch CPU in float32.
    ``manifest`` may be supplied; otherwise it is loaded AND verified via
    ``load_bundle`` (fail closed). No gate, no journal, no manifest writes.

    Returns {tensor_name: float32 np.ndarray}.
    """
    import numpy as np
    import torch

    if manifest is None:
        manifest = load_bundle(bundle_dir)

    base = _load_base_tensors(bundle_dir, manifest)
    base_names = list(base.keys())
    head = manifest["head"]

    if head != "base" and head not in [d["id"] for d in manifest["deltas"]]:
        raise ValueError(f"unknown head {head!r}")

    out = {name: base[name].copy() for name in base_names}
    for d in manifest["deltas"]:
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
        if d["id"] == head:
            break
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
def _bundle_id_for_head(manifest: dict) -> str:
    """Bundle id anchored to head content, not export-time journal state.

    The export journals (which advances the manifest's journal anchor), so
    manifest_id() of the live manifest would differ between two exports of
    the same head. Blank the anchor fields the way _anchor_checkpoint does
    with its placeholder, and the id is a pure function of the head.
    """
    probe = dict(manifest)
    probe["journal"] = dict(probe["journal"])
    probe["journal"]["last_seq"] = 0
    probe["journal"]["last_seq_hash"] = ""
    return manifest_id(probe)


def _write_f16_gguf(out_path: str, tensors: dict, manifest: dict) -> None:
    """Write materialized head tensors as F16 GGUF. Deterministic by design.

    Tensors sorted by name; fixed KV insertion order; no timestamps. The
    same head must produce a byte-identical file.
    """
    import numpy as np
    import gguf

    writer = gguf.GGUFWriter(out_path, "unknown")
    writer.add_name(str(manifest.get("name", "sentinel")))
    writer.add_key_value("si.bundle_id",
                         _bundle_id_for_head(manifest),
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
                  quant: str | None = None) -> dict:
    """Export the current head as a GGUF file.

    1. ``load_bundle`` first: fail closed on any tamper before touching
       anything.
    2. ``materialize_head``: base + active deltas -> float32 tensors.
    3. Write F16 GGUF (deterministic; see ``_write_f16_gguf``).
    4. If ``quant``: delegate to a REAL llama-quantize binary. Missing
       binary -> RuntimeError (honest; never faked). The quantized file
       replaces ``out_path``.
    5. Journal (NOT gated, head NOT moved, no delta): provenance
       ``relation="export"`` for the artifact, then re-anchor the
       manifest to the new journal tip (same pattern as si/seed.py).
    6. Return {"head","head_hash","out_path","bytes","sha256","quant",
       "journal_seq"} — journal_seq is the export provenance entry's seq.
    """
    from si.journal import open_journal  # local import: no cycle

    # Fail fast on a requested-but-impossible quant BEFORE writing anything.
    if quant is not None:
        _llama_quantize_bin()

    # 1. Load + verify first: fail closed on any tamper.
    bundle_dir = os.path.abspath(bundle_dir)
    manifest = load_bundle(bundle_dir)

    # 2. Materialize the head (pure read).
    tensors = materialize_head(bundle_dir, manifest)
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


__all__ = ["materialize_head", "head_hash", "export_bundle"]
