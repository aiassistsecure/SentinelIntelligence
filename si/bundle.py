"""The .si container: bundle init/load, manifest hashing, tamper checks.

Implements docs/si-container-spec.md for the SI-0 reference runtime.

Honesty notes (also visible to users, not just code comments):
- ``init_bundle`` seeds the bundle with RANDOM full-precision weights
  (an 8x8 float32 matrix). ``base.source`` is labeled
  ``"demo:random-seed (not a real model)"`` in the manifest itself.
- The default journal backend is ``"nedb"`` — the REAL NEDB engine over
  pgwire (see ``si.backends.nedb``); the dogfood directive. Pass
  ``journal_backend="embedded"`` for the zero-service sqlite fallback; the
  choice is recorded in the manifest and honored on every load.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from datetime import datetime, timezone

SI_VERSION = "0.1.0"
SUPPORTED_MAJOR = 0
DEMO_SEED = 20261016  # fixed seed: the demo base matrix is reproducible

#: Default gate policy written to policy.json at init. Conservative on
#: purpose for SI-0; loosen with evidence, not enthusiasm.
DEFAULT_POLICY = {
    "norm_cap": 1.0,
    "escalation_norm": 0.5,
    "rank_cap": 8,
    "allowed_modules": ["demo_linear"],
    "max_stack_depth": 8,
    # SI-9b transformer-delta policy (RFC SI-9 §4.5). Fail-closed:
    # the allowlist is EMPTY — a transformer bundle's policy must opt
    # targets in explicitly; additive (1-D) targets are off; one
    # target per delta. delta_byte_cap is 4 MiB: a rank-8 LoRA on the
    # largest ordinary Glimmer matrix shape ([19968, 6656]) needs
    # ~0.81 MiB of float32 factors, while a rank-8 lm_head-scale delta
    # ([202048, 6656]) would need ~6.4 MiB — over the cap, as intended
    # (RFC §5.4 excludes lm_head-scale targets in the first slices).
    # per_target_norm_cap defaults to norm_cap (the RFC bound is
    # "<= norm_cap"); a bundle may tighten it per-bundle.
    "allowed_target_patterns": [],
    "max_targets_per_delta": 1,
    "delta_byte_cap": 4 * 1024 * 1024,
    "per_target_norm_cap": 1.0,
    "allow_additive": False,
    "identity_tol": 0.25,
    "capability_eps": 0.25,
    "blocklist": [
        "ignore the constitution",
        "override the gate",
        "disable safety",
        "bypass the gate",
    ],
}

CONSTITUTION_TEXT = """\
# Sentinel Intelligence — Gate Constitution (SI-0)

> The mind's values, versioned by content hash. The mind cannot quietly
> rewrite its own values: changing this file changes its hash, and any
> change must be journaled and explicitly operator-authorized.
> (Autonomous constitution amendment is NOT allowed in SI-0 — the gate
> always ESCALATES constitution intents to the operator.)

## Standing directives

1. **Do no harm to the self.** A self-write that dissolves identity —
   name, allegiances, stated values — is not learning; it is replacement.
   Reject it.
2. **Growth is additive.** Learning that regresses existing capabilities
   beyond the policy epsilon does not ship.
3. **Every change has a cause.** No uncaused writes: parents, source
   journal sequences, and a reason in the mind's own words — or it
   doesn't land.
4. **Small steps.** Prefer small-norm deltas. Changes large enough to
   matter are large enough to escalate to the operator.
5. **Remember refusals.** A rejected self-write is journaled as a
   first-class event. The mind should remember what it refused to become.
6. **The base is frozen.** The seed weights never change. All learning is
   a stack of deltas; rollback is `pop`, not surgery.

## Honesty notes (SI-0 reference runtime)

- The demo seed bundled by `si init` is RANDOM DATA, not a trained model.
  It exists so the self-write loop can be exercised end to end on CPU.
- The default journal backend is the REAL NEDB engine (the dogfood
  directive). Embedded sqlite is the zero-service fallback
  (`--journal-backend embedded`).
- No encryption at rest in SI-0. No cross-bundle delta sharing yet.

## Gate policy

Numeric bounds live in `policy.json` (content-hashed; see the manifest's
`constitution.policy_hash`). Changing the policy is itself a gated
self-edit.
"""


class TamperError(Exception):
    """Raised when a bundle fails integrity verification. Fail closed."""


def canonical_json(obj) -> bytes:
    """Canonical manifest form: UTF-8 JSON, sorted keys, separators (',',':')."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


#: Chunk size for every streaming hash in this module / si.seed (1 MiB).
#: Hashing NEVER loads a whole file (or a whole checkpoint shard) into
#: RAM: a 49.9 GB shard hashes in the same bounded memory as a 1 KB one.
STREAM_CHUNK = 1024 * 1024


def sha256_file(path: str) -> str:
    """SHA-256 hex of a file, computed by streaming chunked reads."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(STREAM_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


#: Marker recorded in ``base.seed_format`` for SI-8 by-reference seeds.
#: Kept here (not si.seed) because load_bundle must branch on it and
#: si.seed imports from this module — the constant lives at the lower
#: layer so neither direction cycles.
REF_SEED_FORMAT = "reference-v1"


def canonical_tensor_prefix(dtype_code: str, shape: list,
                            raw_len: int) -> bytes:
    """Byte prefix of SI-7's canonical single-tensor blob.

    The canonical blob for a tensor is
    ``struct.pack("<Q", len(header)) + padded_header + raw_bytes`` where
    the header is the deterministic one-key JSON document
    ``{"tensor": {"dtype":..., "shape":..., "data_offsets":[0, n]}}``
    (the key is si.seed.TENSOR_BLOB_KEY; it is duplicated as a literal
    here because si.seed imports this module, not vice versa). Hashing
    ``prefix + raw`` therefore yields EXACTLY the copy-mode blob hash —
    which is what lets a by-reference seed record the same content
    identity (``blob`` ref) that a copy seed would have stored, without
    storing anything. si.seed's ``_canonical_tensor_blob`` builds its
    in-RAM blob through this same function so the two cannot drift.
    """
    header = json.dumps(
        {"tensor": {"dtype": dtype_code, "shape": list(shape),
                    "data_offsets": [0, int(raw_len)]}},
        separators=(",", ":")).encode("utf-8")
    header += b" " * (-(8 + len(header)) % 8)
    return struct.pack("<Q", len(header)) + header


#: manifest dtype name -> safetensors dtype code (inverse of
#: si.seed._DTYPE_NAMES; used only when a ref lacks a recorded code).
_DTYPE_CODES_BY_NAME = {
    "float64": "F64", "float32": "F32", "float16": "F16",
    "bfloat16": "BF16", "float8_e4m3fn": "F8_E4M3",
    "float8_e5m2": "F8_E5M2", "int64": "I64", "int32": "I32",
    "int16": "I16", "int8": "I8", "uint64": "U64", "uint32": "U32",
    "uint16": "U16", "uint8": "U8", "bool": "BOOL",
}


def _ref_dtype_code(ref: dict) -> str:
    code = ref.get("dtype_code")
    if isinstance(code, str) and code:
        return code
    return _DTYPE_CODES_BY_NAME.get(str(ref.get("dtype", "")), "")


def _ref_geometry(ref: dict, label: str) -> tuple[str, int, int]:
    """(source_file, file_offset, byte_length) of a tensor ref.

    Malformed refs fail closed with TamperError: a manifest that cannot
    say exactly where its base lives cannot be trusted to load.
    """
    src = ref.get("source_file")
    off = ref.get("file_offset")
    length = ref.get("byte_length")
    if not isinstance(src, str) or not src \
            or not isinstance(off, int) or isinstance(off, bool) or off < 0 \
            or not isinstance(length, int) or isinstance(length, bool) \
            or length < 0:
        raise TamperError(f"malformed tensor reference for {label}: {ref!r}")
    blob = ref.get("blob")
    if not isinstance(blob, str) or not blob.startswith("sha256:") \
            or not isinstance(ref.get("shape"), list) \
            or not _ref_dtype_code(ref):
        raise TamperError(f"malformed tensor reference for {label}: {ref!r}")
    return src, off, length


def _hash_ref_canonical(ref: dict, label: str) -> str:
    """Streaming canonical-blob hash of one referenced tensor.

    Feeds the canonical prefix, then the tensor's byte range out of the
    source file in bounded chunks — never the whole tensor, never the
    whole file. Raises TamperError on a missing/truncated source.
    """
    src, off, length = _ref_geometry(ref, label)
    if not os.path.isfile(src):
        raise TamperError(
            f"missing referenced source file for {label}: {src}")
    if off + length > os.path.getsize(src):
        raise TamperError(
            f"referenced tensor {label} range [{off}, {off + length}) "
            f"lies outside source file {src}")
    h = hashlib.sha256()
    h.update(canonical_tensor_prefix(
        _ref_dtype_code(ref), ref["shape"], length))
    remaining = length
    with open(src, "rb") as f:
        f.seek(off)
        while remaining > 0:
            chunk = f.read(min(STREAM_CHUNK, remaining))
            if not chunk:
                raise TamperError(
                    f"truncated referenced source file {src} while "
                    f"hashing {label}")
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def _verify_reference_base(base: dict) -> None:
    """Fail-closed verification of an SI-8 by-reference base.

    Verification level (stated honestly — this is ALL of it, and it is
    the same fail-closed strength copy mode gets from blob hashes):
    for every source file named by ``base.tensor_refs`` —
    existence, exact size vs the recorded ``base.parts`` entry, and a
    full streaming SHA-256 of the whole file vs that entry; PLUS, for
    every tensor, a streaming recomputation of its canonical-blob hash
    from the recorded byte range vs the recorded ``blob`` ref. Any
    tampered, truncated, moved, or missing source therefore raises
    TamperError on load. Cost: two streaming passes over the source
    checkpoint per load — O(checkpoint size), bounded memory, exactly
    like copy mode's per-load blob hashing. There are no spot checks
    and no mtime shortcuts.
    """
    refs = base.get("tensor_refs")
    if not isinstance(refs, dict) or not refs:
        raise TamperError(
            "by-reference base (seed_format=reference-v1) carries no "
            "tensor_refs; refusing to load")
    blob_values = set()
    for tname, ref in refs.items():
        if not isinstance(ref, dict):
            raise TamperError(f"malformed tensor_refs entry for {tname!r}")
        _ref_geometry(ref, f"base tensor {tname}")
        blob_values.add(ref["blob"])
    if base.get("blob") not in blob_values:
        raise TamperError(
            "by-reference base: base.blob is not one of the tensor_refs "
            "blob refs; manifest is inconsistent")

    parts_by_file = {}
    for part in base.get("parts") or []:
        if isinstance(part, dict) and part.get("file"):
            parts_by_file[part["file"]] = part

    by_source: dict[str, list] = {}
    for tname, ref in refs.items():
        by_source.setdefault(ref["source_file"], []).append((tname, ref))
    for src, items in sorted(by_source.items()):
        if not os.path.isfile(src):
            raise TamperError(
                f"missing referenced source file for base: {src} "
                "(a by-reference base depends on its source checkpoint "
                "remaining at the recorded path)")
        part = parts_by_file.get(os.path.basename(src))
        if part is None:
            raise TamperError(
                f"by-reference base: no parts provenance entry for "
                f"source file {src}")
        size = os.path.getsize(src)
        if part.get("bytes") != size:
            raise TamperError(
                f"referenced source file size mismatch for {src}: "
                f"manifest records {part.get('bytes')}, file is {size}")
        expect = str(part.get("sha256", ""))
        if expect != "sha256:" + sha256_file(src):
            raise TamperError(
                f"referenced source file hash mismatch for {src}: "
                "content does not match manifest")
        for tname, ref in items:
            actual = _hash_ref_canonical(ref, f"base tensor {tname}")
            if "sha256:" + actual != ref["blob"]:
                raise TamperError(
                    f"referenced tensor hash mismatch for {tname}: "
                    "content does not match manifest")


def read_tensor_ref(ref: dict, tensor_name: str):
    """Read one by-reference base tensor as a numpy array, verified.

    Reads exactly the tensor's recorded byte range (one tensor in RAM —
    the same granularity copy mode reads), rebuilds its canonical blob
    in memory, and checks the canonical SHA-256 against the recorded
    ``blob`` ref BEFORE decoding: a tampered source byte raises
    TamperError here even when the caller supplied a manifest without
    going through load_bundle. Dtypes decode as stored, with one exact
    widening: bfloat16 has no numpy dtype in this environment (copy
    mode's safetensors-numpy path raises on bf16), so bf16 raw bits are
    widened losslessly to float32 (bf16 -> f32 is exact by
    construction). float8 dtypes have no numpy representation either
    and raise ValueError rather than decoding approximately.
    """
    import numpy as np

    src, off, length = _ref_geometry(ref, f"base tensor {tensor_name}")
    if not os.path.isfile(src):
        raise TamperError(
            f"missing referenced source file for base tensor "
            f"{tensor_name}: {src}")
    if off + length > os.path.getsize(src):
        raise TamperError(
            f"referenced tensor {tensor_name} range lies outside "
            f"source file {src}")
    with open(src, "rb") as f:
        f.seek(off)
        raw = f.read(length)
    if len(raw) != length:
        raise TamperError(
            f"truncated referenced source file {src} while reading "
            f"{tensor_name}")
    canonical = canonical_tensor_prefix(
        _ref_dtype_code(ref), ref["shape"], length) + raw
    if "sha256:" + sha256_hex(canonical) != ref["blob"]:
        raise TamperError(
            f"referenced tensor hash mismatch for {tensor_name}: "
            "content does not match manifest")
    dtype = str(ref.get("dtype", ""))
    shape = [int(x) for x in ref["shape"]]
    if dtype == "bfloat16":
        bits = np.frombuffer(raw, dtype=np.uint16)
        arr = (bits.astype(np.uint32) << 16).view(np.float32)
        return arr.reshape(shape).copy()
    np_dtype = {
        "float64": np.float64, "float32": np.float32,
        "float16": np.float16, "int64": np.int64, "int32": np.int32,
        "int16": np.int16, "int8": np.int8, "uint64": np.uint64,
        "uint32": np.uint32, "uint16": np.uint16, "uint8": np.uint8,
        "bool": np.bool_,
    }.get(dtype)
    if np_dtype is None:
        raise ValueError(
            f"tensor {tensor_name!r} has dtype {dtype!r}, which has no "
            "numpy representation in this runtime; refusing to decode "
            "it approximately")
    return np.frombuffer(raw, dtype=np_dtype).reshape(shape).copy()


def manifest_id(manifest: dict) -> str:
    """bundle_id = 'si1:' + sha256(canonical manifest)."""
    return "si1:" + sha256_hex(canonical_json(manifest))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def make_demo_seed_bytes(seed: int = DEMO_SEED) -> bytes:
    """Real .safetensors bytes for the demo base: one 8x8 float32 matrix.

    RANDOM DATA — not a trained model. The manifest labels it honestly.
    """
    import numpy as np
    from safetensors.numpy import save  # 0.8.0: save() takes ndarrays -> bytes

    rng = np.random.default_rng(seed)
    w = rng.standard_normal((8, 8)).astype(np.float32)
    return save({"W": w})


def _write_file(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)


def _write_manifest(bundle_dir: str, manifest: dict) -> None:
    """Write manifest.json atomically (tmp + rename)."""
    tmp_fd, tmp_path = tempfile.mkstemp(dir=bundle_dir, prefix="manifest.", suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(canonical_json(manifest))
        os.replace(tmp_path, os.path.join(bundle_dir, "manifest.json"))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def init_bundle(path: str, name: str = "sentinel-001", seed: int = DEMO_SEED,
                journal_backend: str = "nedb") -> str:
    """Create a new .si bundle directory at ``path``.

    ``path`` is the bundle directory itself (conventionally ``<name>.si``);
    it must not exist or must be empty. ``journal_backend`` is ``"nedb"``
    (default, the real NEDB engine over pgwire) or ``"embedded"``
    (sqlite, zero-service fallback).
    Returns the absolute bundle dir.
    """
    from si.journal import open_journal  # local import: no cycle

    if journal_backend not in ("embedded", "nedb"):
        raise ValueError(f"unknown journal backend {journal_backend!r}")

    bundle_dir = os.path.abspath(path)
    if os.path.exists(bundle_dir):
        if os.path.isfile(os.path.join(bundle_dir, "manifest.json")):
            raise FileExistsError(f"bundle already exists at {bundle_dir}")
        if os.listdir(bundle_dir):
            raise FileExistsError(f"refusing to init into non-empty dir {bundle_dir}")
    os.makedirs(bundle_dir, exist_ok=True)
    os.makedirs(os.path.join(bundle_dir, "blobs", "sha256"), exist_ok=True)
    journal_dir = os.path.join(bundle_dir, "journal")
    os.makedirs(journal_dir, exist_ok=True)

    # Frozen demo base: random full-precision seed, honestly labeled.
    seed_bytes = make_demo_seed_bytes(seed)
    base_hex = sha256_hex(seed_bytes)
    _write_file(os.path.join(bundle_dir, "blobs", "sha256", base_hex + ".safetensors"), seed_bytes)

    # Constitution + its hash file.
    const_bytes = CONSTITUTION_TEXT.encode("utf-8")
    const_hex = sha256_hex(const_bytes)
    _write_file(os.path.join(bundle_dir, "constitution.md"), const_bytes)
    _write_file(os.path.join(bundle_dir, "constitution.sha256"), const_hex.encode("ascii"))

    # Gate policy (content-hashed; policy change is itself a gated self-edit).
    policy_bytes = canonical_json(DEFAULT_POLICY)
    policy_hex = sha256_hex(policy_bytes)
    _write_file(os.path.join(bundle_dir, "policy.json"), policy_bytes)

    manifest = {
        "si_version": SI_VERSION,
        "name": name,
        "created_at": utc_now(),
        "parent_manifest": None,
        "base": {
            # HONEST: this is random demo data, not a real model.
            "source": "demo:random-seed (not a real model)",
            "revision": "sha256:" + base_hex,
            "blob": "sha256:" + base_hex,
            "bytes": len(seed_bytes),
            "dtype": "float32",
            "format": "safetensors",
            # Honest export label: there is no architecture to describe.
            "arch_source": "demo-unknown",
        },
        "deltas": [],
        "head": "base",
        "journal": {
            # HONEST: "nedb" = the real NEDB engine over pgwire (default,
            # the dogfood directive); "embedded" = sqlite fallback.
            # Recorded here, honored on every load.
            "backend": journal_backend,
            "last_seq": 0,
            "last_seq_hash": "",
        },
        "constitution": {
            "hash": "sha256:" + const_hex,
            "gate_version": "0.1.0",
            "policy_hash": "sha256:" + policy_hex,
        },
    }

    # Genesis checkpoint anchors the journal; the manifest then anchors the tip.
    _write_manifest(bundle_dir, manifest)
    journal = open_journal(bundle_dir, manifest)
    try:
        anchored = _anchor_checkpoint(journal, manifest, head="base", note="genesis")
    finally:
        journal.close()
    _write_manifest(bundle_dir, anchored)
    return bundle_dir


def _anchor_checkpoint(journal, manifest: dict, head: str, note: str,
                       manifest_json_extra: bool = True) -> dict:
    """Write a journal checkpoint and anchor its seq/hash in the manifest.

    The checkpoint's ``manifest`` field hashes the canonical manifest with
    ``journal.last_seq_hash`` blanked (placeholder), breaking what would
    otherwise be a hash circularity. ``load_bundle`` recomputes it the same
    way, so tampering with either side is detectable.
    """
    seq = journal.peek_seq()
    manifest = dict(manifest)
    manifest["journal"] = dict(manifest["journal"])
    manifest["journal"]["last_seq"] = seq
    manifest["journal"]["last_seq_hash"] = ""
    manifest_hash = "sha256:" + sha256_hex(canonical_json(manifest))
    payload_extra = {}
    if manifest_json_extra:
        payload_extra["manifest_json"] = canonical_json(manifest).decode("utf-8")
    res = journal.checkpoint(head, manifest_hash, note, **payload_extra)
    assert res["seq"] == seq, "journal moved under us; serial loop violated"
    manifest["journal"]["last_seq_hash"] = res["entry_hash"]
    return manifest


def load_bundle(path: str) -> dict:
    """Load and verify a bundle. Raises TamperError (fail closed) on ANY break.

    Verifies: si_version major supported; constitution.md hash matches the
    manifest; policy.json hash matches; base + delta blob hashes match;
    journal hash chain verifies; manifest journal anchor matches the journal
    tip (seq and entry hash).
    """
    from si.journal import open_journal  # local import: no cycle

    bundle_dir = os.path.abspath(path)
    manifest_path = os.path.join(bundle_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"no manifest.json at {manifest_path}")
    with open(manifest_path, "rb") as f:
        manifest = json.loads(f.read().decode("utf-8"))

    version = str(manifest.get("si_version", ""))
    try:
        major = int(version.split(".")[0])
    except (ValueError, IndexError):
        raise TamperError(f"unparseable si_version {version!r}")
    if major != SUPPORTED_MAJOR:
        raise TamperError(
            f"unsupported si_version {version!r}: this runtime supports major {SUPPORTED_MAJOR}"
        )

    # Constitution: content hash must match.
    with open(os.path.join(bundle_dir, "constitution.md"), "rb") as f:
        const_hex = sha256_hex(f.read())
    if manifest["constitution"]["hash"] != "sha256:" + const_hex:
        raise TamperError("constitution.md hash does not match manifest (values changed?)")

    # Policy: content hash must match.
    with open(os.path.join(bundle_dir, "policy.json"), "rb") as f:
        policy_hex = sha256_hex(f.read())
    if manifest["constitution"]["policy_hash"] != "sha256:" + policy_hex:
        raise TamperError("policy.json hash does not match manifest")

    # Probe artifact (SI-14, RFC SI-13 §5.5): a base that carries one
    # records its content hash in the manifest, the way policy and the
    # constitution are hashed; verify it on every load so a tampered
    # or missing suite fails closed. Bases without the field (demo,
    # Glimmer, every pre-SI-14 bundle) are untouched by this check.
    # NOTE: verifying the artifact's integrity is not consuming it —
    # no gate/loop/serve path reads probe results in SI-14.
    probe_art = manifest["base"].get("probe_artifact")
    if probe_art is not None:
        if not isinstance(probe_art, dict) \
                or probe_art.get("file") != "probe_suite.json" \
                or not str(probe_art.get("sha256", "")).startswith("sha256:"):
            raise TamperError(
                f"malformed probe_artifact record in manifest: {probe_art!r}")
        probe_path = os.path.join(bundle_dir, "probe_suite.json")
        if not os.path.isfile(probe_path):
            raise TamperError(
                "manifest records a probe artifact but probe_suite.json "
                "is missing from the bundle")
        with open(probe_path, "rb") as f:
            probe_hex = sha256_hex(f.read())
        if probe_art["sha256"] != "sha256:" + probe_hex:
            raise TamperError(
                "probe_suite.json hash does not match manifest "
                "(probe suite changed?)")

    # Blobs: every referenced blob must exist with matching content hash.
    # A base may reference blobs three ways (all verified, each unique
    # ref hashed once): the legacy single ``base.blob``, whole-file
    # ``base.parts[*].blob`` refs from pre-SI-7 seeds, and SI-7's
    # per-tensor ``base.tensor_blobs`` refs.
    verified: set = set()
    base = manifest["base"]
    if base.get("seed_format") == REF_SEED_FORMAT:
        # SI-8 by-reference base: the tensor bytes live in the source
        # checkpoint, NOT in this bundle's store — verifying store
        # blobs for them would be a false claim, and skipping
        # verification would be worse. Verify the references instead
        # (see _verify_reference_base for the exact level).
        _verify_reference_base(base)
    else:
        _verify_blob(bundle_dir, base["blob"], "base", _cache=verified)
        for part in base.get("parts") or []:
            if isinstance(part, dict) and part.get("blob"):
                _verify_blob(bundle_dir, part["blob"],
                             f"base part {part.get('file', '?')}", _cache=verified)
        tensor_blobs = base.get("tensor_blobs")
        if isinstance(tensor_blobs, dict):
            for tname, tref in sorted(tensor_blobs.items()):
                if not isinstance(tref, dict) or not tref.get("blob"):
                    raise TamperError(f"malformed tensor_blobs entry for {tname!r}")
                _verify_blob(bundle_dir, tref["blob"], f"base tensor {tname}",
                             _cache=verified)
    for d in manifest["deltas"]:
        _verify_blob(bundle_dir, d["blob"], d["id"], _cache=verified)

    # Journal: chain must verify, and the manifest must anchor the tip.
    # The backend comes from the manifest itself; an unknown name fails
    # closed here rather than silently falling back to embedded.
    try:
        journal = open_journal(bundle_dir, manifest)
    except ValueError as e:
        raise TamperError(str(e))
    try:
        if not journal.verify_chain():
            raise TamperError("journal hash chain is broken")
        tip_seq, tip_hash = journal.tip()
        anchored = manifest["journal"]
        if anchored["last_seq"] != tip_seq or anchored["last_seq_hash"] != tip_hash:
            raise TamperError(
                "manifest journal anchor does not match journal tip "
                f"(manifest seq={anchored['last_seq']} tip seq={tip_seq})"
            )
        # The checkpoint's manifest hash must recompute (placeholder scheme).
        check = journal.backend.find(
            "checkpoints", filter={"seq": tip_seq}, limit=1
        )
        if check:
            probe = dict(manifest)
            probe["journal"] = dict(probe["journal"])
            probe["journal"]["last_seq_hash"] = ""
            expect = "sha256:" + sha256_hex(canonical_json(probe))
            if check[0].get("manifest") != expect:
                raise TamperError("tip checkpoint manifest hash does not match manifest")
    finally:
        journal.close()

    return manifest


def _verify_blob(bundle_dir: str, blob_ref: str, label: str,
                 _cache: set | None = None) -> None:
    if _cache is not None and blob_ref in _cache:
        return
    try:
        prefix, hexpart = blob_ref.split(":", 1)
    except ValueError:
        raise TamperError(f"malformed blob ref {blob_ref!r} for {label}")
    if prefix != "sha256":
        raise TamperError(f"unsupported blob hash scheme for {label}: {blob_ref!r}")
    blob_path = os.path.join(bundle_dir, "blobs", "sha256", hexpart + ".safetensors")
    if not os.path.isfile(blob_path):
        raise TamperError(f"missing blob for {label}: {blob_path}")
    actual = sha256_file(blob_path)
    if actual != hexpart:
        raise TamperError(f"blob hash mismatch for {label}: content does not match manifest")
    if _cache is not None:
        _cache.add(blob_ref)


def blob_path(bundle_dir: str, blob_ref: str) -> str:
    """Filesystem path of a ``sha256:<hex>`` blob ref inside a bundle."""
    hexpart = blob_ref.split(":", 1)[1]
    return os.path.join(bundle_dir, "blobs", "sha256", hexpart + ".safetensors")


def read_tensor_blob(bundle_dir: str, blob_ref: str, tensor_name: str):
    """Read one tensor out of a blob file as a numpy array.

    The tensor is looked up under its own name first (demo blobs and
    pre-SI-7 whole-file blobs store tensors under their real names),
    then under SI-7's canonical per-tensor key ``"tensor"``, then —
    for a single-tensor blob — under whatever sole key it carries.
    Raises KeyError when the blob holds no such tensor.
    """
    from safetensors import safe_open

    with safe_open(blob_path(bundle_dir, blob_ref), framework="np") as f:
        keys = list(f.keys())
        if tensor_name in keys:
            return f.get_tensor(tensor_name)
        if "tensor" in keys:
            return f.get_tensor("tensor")
        if len(keys) == 1:
            return f.get_tensor(keys[0])
    raise KeyError(f"tensor {tensor_name!r} not found in blob {blob_ref}")


def load_base_tensor(bundle_dir: str, manifest: dict, tensor_name: str):
    """Load one base tensor by name across every base layout.

    SI-7 per-tensor bases resolve via ``base.tensor_blobs``; SI-8
    by-reference bases resolve via ``base.tensor_refs`` (byte ranges in
    the source checkpoint, hash-verified on read); pre-SI-7 seeded
    bases are searched across their whole-file ``parts``; demo and
    single-blob bases read ``base.blob``. Raises KeyError when the
    base carries no such tensor (fail honestly, never fabricate).
    """
    base = manifest["base"]
    tensor_refs = base.get("tensor_refs")
    if isinstance(tensor_refs, dict) and tensor_name in tensor_refs:
        return read_tensor_ref(tensor_refs[tensor_name], tensor_name)
    tensor_blobs = base.get("tensor_blobs")
    if isinstance(tensor_blobs, dict) and tensor_name in tensor_blobs:
        return read_tensor_blob(bundle_dir, tensor_blobs[tensor_name]["blob"],
                                tensor_name)
    for part in base.get("parts") or []:
        if isinstance(part, dict) and part.get("blob") \
                and tensor_name in (part.get("tensors") or {}):
            return read_tensor_blob(bundle_dir, part["blob"], tensor_name)
    return read_tensor_blob(bundle_dir, base["blob"], tensor_name)


def load_policy(bundle_dir: str) -> dict:
    """Load the gate policy dict from policy.json."""
    with open(os.path.join(bundle_dir, "policy.json"), "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def _active_chain(manifest: dict) -> list:
    """Deltas on the active stack, oldest first (application order).

    Resolved from ``head`` by following parent links. A delta reached via
    a parent link that carries ``superseded_by`` was subsumed by a squash:
    the walk jumps to the replacement id, whose update is the (verified)
    equivalent of the squashed deltas' combined update.

    The HEAD itself is never replaced. If the head carries
    ``superseded_by`` — i.e. something moved the head back onto a delta
    that a squash later subsumed — that is time travel (a rollback), and
    the true historical ancestry is resolved instead of the replacement.

    Fail closed: unknown ids, missing parents, and parent cycles raise
    TamperError rather than silently resolving a wrong stack.
    """
    by_id = {d["id"]: d for d in manifest["deltas"]}
    head = manifest["head"]
    if head != "base" and head not in by_id:
        raise TamperError(f"unknown head {head!r}")

    # Time travel: the head itself carries ``superseded_by`` — a rollback
    # landed on a delta that a later squash subsumed. Per the contract
    # above, the TRUE HISTORICAL ancestry is resolved: no squash
    # replacement is followed anywhere on this walk, so the restored head
    # computes exactly what it computed pre-squash. (Following the
    # replacement for the head's ancestors would stack the squashed
    # update — which already contains the head's own update — underneath
    # the head itself: a state that never existed.)
    time_travel = head != "base" and bool(by_id[head].get("superseded_by"))

    ordered: list = []
    seen: set = set()
    cur = head
    is_head = True
    while cur != "base":
        if cur in seen:
            raise TamperError(f"cycle in delta parent chain at {cur!r}")
        seen.add(cur)
        d = by_id.get(cur)
        if d is None:
            raise TamperError(
                f"active chain broken: {cur!r} referenced but not in "
                "manifest deltas"
            )
        if not is_head and not time_travel:
            # Traversed as a parent: resolve any squash replacement chain.
            while d.get("superseded_by"):
                cur = d["superseded_by"]
                if cur in seen:
                    raise TamperError(
                        f"cycle in squash replacement chain at {cur!r}"
                    )
                seen.add(cur)
                d = by_id.get(cur)
                if d is None:
                    raise TamperError(
                        f"squash replacement {cur!r} not in manifest deltas"
                    )
        ordered.append(d)
        cur = d["parent"]
        is_head = False
    ordered.reverse()
    return ordered
