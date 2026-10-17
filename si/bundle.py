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

    # Blobs: every referenced blob must exist with matching content hash.
    _verify_blob(bundle_dir, manifest["base"]["blob"], "base")
    for d in manifest["deltas"]:
        _verify_blob(bundle_dir, d["blob"], d["id"])

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


def _verify_blob(bundle_dir: str, blob_ref: str, label: str) -> None:
    try:
        prefix, hexpart = blob_ref.split(":", 1)
    except ValueError:
        raise TamperError(f"malformed blob ref {blob_ref!r} for {label}")
    if prefix != "sha256":
        raise TamperError(f"unsupported blob hash scheme for {label}: {blob_ref!r}")
    blob_path = os.path.join(bundle_dir, "blobs", "sha256", hexpart + ".safetensors")
    if not os.path.isfile(blob_path):
        raise TamperError(f"missing blob for {label}: {blob_path}")
    with open(blob_path, "rb") as f:
        actual = sha256_hex(f.read())
    if actual != hexpart:
        raise TamperError(f"blob hash mismatch for {label}: content does not match manifest")


def load_policy(bundle_dir: str) -> dict:
    """Load the gate policy dict from policy.json."""
    with open(os.path.join(bundle_dir, "policy.json"), "rb") as f:
        return json.loads(f.read().decode("utf-8"))
