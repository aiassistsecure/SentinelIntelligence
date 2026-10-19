"""SI-9a: the transformer-delta format — data definitions + pure validation.

Implements §2 (blob layout) and §3 (manifest record) of
``docs/SI-9-transformer-self-write-rfc.md`` as *code*, with NO gate
integration, NO loop integration, NO serve changes, and NO
apply-at-load (those are SI-9b / SI-9c). This module can construct,
serialise, parse, and structurally validate a transformer-delta
record/blob pair against a base manifest — and nothing else. It never
reads base tensor bytes, never opens a bundle, and never writes to a
bundle's blob store.

Module-location decision (RFC §6.1 leaves the choice to SI-9a):
this lives in a NEW module, ``si/transformer_delta.py``, rather than
inside ``si/bundle.py``. Rationale: ``bundle.py`` is already ~745
lines and its contract is container storage + integrity verification;
the delta format is neither storage nor arithmetic (apply arithmetic
is SI-9c, whose ``load_applied_tensor`` the RFC leans toward placing
in a future ``si/apply.py``). A separate pure module keeps the format
importable by the gate (SI-9b) and the apply path (SI-9c) without
either depending on bundle internals, and it has no import cycle:
it needs only a base *manifest dict* (data), never bundle I/O.

Format decisions pinned down by SI-9a (the RFC left these to 9a):

1. **Aggregate norm (RFC §3).** The record-level ``norm`` is the
   ROOT-SUM-SQUARE over the per-target norms:
   ``norm = sqrt(sum(target_norm_i ** 2))``. Per-target norm is the
   Frobenius norm of the *scaled* LoRA update,
   ``||(alpha/rank) * B @ A||_F``, for ``lora`` targets, and the L2
   norm ``||d||_2`` for ``additive`` targets. Norms are recomputed
   from the blob's factors (in float64) during validation; a record
   whose ``target_norm`` or aggregate ``norm`` disagrees beyond a
   small float tolerance is invalid.
2. **Float32 factor storage (RFC §2.3) — CONFIRMED.** Delta factors
   are stored in float32, exactly as the RFC proposed: factors are
   small (rank << dimensions), the demo adapters already store
   float32, and the gate's norm accounting is float arithmetic.
   Storing bf16 factors to "match the base" would save little and
   complicate exact numeric tests. Validation therefore requires
   every factor's safetensors dtype to be ``F32``.
3. **Metadata duplication (RFC §2.3).** The blob's safetensors
   metadata carries ``format: "transformer-delta-v1"`` and a
   ``targets`` key holding a JSON array with, per target, exactly
   ``name``, ``kind``, ``factor_keys``, and (``lora`` only) ``rank``
   and ``alpha``. Validation compares that list against the manifest
   record's targets on exactly those fields; ANY disagreement
   between metadata and manifest is invalid — treated as tamper,
   consistent with how ``load_bundle`` treats manifest inconsistency.

Blob layout (RFC §2.3): one blob per delta; keys are indexed, never
name-derived — ``t{i}.A`` / ``t{i}.B`` for the ``lora`` target at
index ``i``, ``t{i}.d`` for the ``additive`` target at index ``i``.

Base binding (RFC §2.4): each target records ``base_blob`` copied
from the base manifest — ``base.tensor_refs[name].blob`` for a
by-reference base or ``base.tensor_blobs[name].blob`` for a copy-mode
base. The hash is mode-independent (SI-8), so the same record
validates against either storage mode of the same checkpoint.
Matching is by EXACT tensor name only — never substring (the
``si export`` substring resolution is explicitly not carried over).

Validation fails closed (raises :class:`DeltaValidationError`) on,
at minimum: unknown target name; ``base_blob`` / ``base_shape`` /
``base_dtype`` mismatch vs the base manifest; shape lies (record vs
blob header vs base shape); factor dtype other than float32;
non-1-D/2-D base targets; ``lora`` on a 1-D target or ``additive``
on a 2-D target; ``rank < 1``; missing or extra factor keys (in the
record or in the blob); metadata-vs-manifest target mismatch;
record/blob hash or byte-count mismatch; non-finite factors; and
norm lies (per-target or aggregate). This is machine learning
infrastructure; nothing here trains, infers, or claims to.
"""
from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass

import numpy as np

from si.bundle import canonical_json, sha256_hex

#: Manifest ``kind`` value for transformer delta records (RFC §3).
KIND = "transformer-delta-v1"

#: Safetensors metadata ``format`` value for transformer delta blobs.
FORMAT = "transformer-delta-v1"

#: The one storage dtype for delta factors (decision 2 above).
FACTOR_DTYPE_NAME = "float32"
FACTOR_DTYPE_CODE = "F32"

#: Relative tolerance for recomputed-norm comparisons. Factors are
#: float32 and norms are recomputed in float64 from the exact stored
#: bytes, so agreement should be near-exact; this only absorbs JSON
#: float round-tripping and summation-order noise.
_NORM_RTOL = 1e-6


class DeltaValidationError(ValueError):
    """A transformer-delta record/blob pair is structurally invalid.

    Raised by every fail-closed path in this module (construction
    self-checks, parsing, and validation). Subclasses ValueError so
    existing call sites that treat malformed input as ValueError keep
    working.
    """


# ---------------------------------------------------------------------------
# Base manifest access (mode-independent, exact names only)
# ---------------------------------------------------------------------------

def base_tensor_map(base_or_manifest: dict) -> dict:
    """Return ``{tensor_name: {"blob", "shape", "dtype"}}`` for a base.

    Accepts either a full manifest dict (its ``base`` section is used)
    or the base section itself. Reads ``tensor_refs`` (by-reference,
    SI-8) and/or ``tensor_blobs`` (copy mode, SI-7); when both carry a
    name they must agree on blob/shape/dtype, else the manifest is
    inconsistent and this raises DeltaValidationError. A base with
    neither map (e.g. a demo base) yields an empty map — every target
    is then unknown, which is the correct fail-closed outcome for a
    transformer delta against a non-transformer base.
    """
    if not isinstance(base_or_manifest, dict):
        raise DeltaValidationError("base manifest is not an object")
    base = base_or_manifest.get("base")
    if not isinstance(base, dict):
        base = base_or_manifest
    out: dict = {}
    for key in ("tensor_refs", "tensor_blobs"):
        entries = base.get(key)
        if entries is None:
            continue
        if not isinstance(entries, dict):
            raise DeltaValidationError(f"base.{key} is not an object")
        for name, ref in entries.items():
            if not isinstance(ref, dict) or not isinstance(ref.get("blob"), str) \
                    or not isinstance(ref.get("shape"), list) \
                    or not isinstance(ref.get("dtype"), str):
                raise DeltaValidationError(
                    f"malformed base entry for tensor {name!r} in base.{key}")
            entry = {"blob": ref["blob"],
                     "shape": [int(x) for x in ref["shape"]],
                     "dtype": ref["dtype"]}
            if name in out and out[name] != entry:
                raise DeltaValidationError(
                    f"base manifest disagrees with itself about tensor "
                    f"{name!r} (tensor_refs vs tensor_blobs)")
            out[name] = entry
    return out


# ---------------------------------------------------------------------------
# Norms (decision 1: aggregate = root-sum-square)
# ---------------------------------------------------------------------------

def compute_target_norm(kind: str, factors: dict, rank=None, alpha=None) -> float:
    """Per-target norm from factor arrays, computed in float64.

    ``lora``: ``||(alpha/rank) * B @ A||_F``. ``additive``: ``||d||_2``.
    ``factors`` maps "A"/"B" (lora) or "d" (additive) to arrays.
    """
    if kind == "lora":
        B = np.asarray(factors["B"], dtype=np.float64)
        A = np.asarray(factors["A"], dtype=np.float64)
        return float(np.linalg.norm((float(alpha) / float(rank)) * (B @ A)))
    if kind == "additive":
        d = np.asarray(factors["d"], dtype=np.float64)
        return float(np.linalg.norm(d))
    raise DeltaValidationError(f"unknown target kind {kind!r}")


def aggregate_norm(target_norms) -> float:
    """Record-level norm: ROOT-SUM-SQUARE over per-target norms.

    ``sqrt(sum(n_i ** 2))`` — the definition SI-9a pins down for
    RFC §3. An empty stack of targets aggregates to 0.0 (a record
    with no targets is separately invalid; see validation).
    """
    return float(math.sqrt(sum(float(n) ** 2 for n in target_norms)))


# ---------------------------------------------------------------------------
# Blob serialisation / parsing
# ---------------------------------------------------------------------------

def _metadata_targets(targets: list) -> list:
    """The metadata copy of the target list (decision 3's field set)."""
    out = []
    for t in targets:
        entry = {"name": t["name"], "kind": t["kind"],
                 "factor_keys": list(t["factor_keys"])}
        if t["kind"] == "lora":
            entry["rank"] = t["rank"]
            entry["alpha"] = t["alpha"]
        out.append(entry)
    return out


def serialize_blob(factors: dict, targets: list) -> bytes:
    """Serialise a delta blob: ``factors`` {key: array} + metadata.

    ``targets`` supplies the manifest-side target dicts whose
    name/kind/factor_keys(/rank/alpha) are duplicated into the
    safetensors metadata. Factor arrays are stored as float32
    (decision 2); non-float32 inputs are cast, and validation of the
    resulting blob still checks the stored dtype is F32.
    """
    from safetensors.numpy import save

    arrays = {k: np.ascontiguousarray(v, dtype=np.float32)
              for k, v in factors.items()}
    metadata = {"format": FORMAT,
                "targets": json.dumps(_metadata_targets(targets),
                                      sort_keys=True,
                                      separators=(",", ":"))}
    return save(arrays, metadata=metadata)


def _parse_safetensors_header(blob_bytes: bytes) -> tuple[dict, int]:
    """(header dict, data_start) for raw safetensors bytes, fail closed."""
    if not isinstance(blob_bytes, (bytes, bytearray)) or len(blob_bytes) < 8:
        raise DeltaValidationError("delta blob is too short to be safetensors")
    (header_len,) = struct.unpack("<Q", bytes(blob_bytes[:8]))
    if header_len > len(blob_bytes) - 8:
        raise DeltaValidationError(
            "delta blob header length exceeds blob size")
    try:
        header = json.loads(
            bytes(blob_bytes[8:8 + header_len]).decode("utf-8"))
    except Exception as err:
        raise DeltaValidationError(f"delta blob header is not JSON: {err}") from err
    if not isinstance(header, dict):
        raise DeltaValidationError("delta blob header is not an object")
    data_start = 8 + header_len
    for key, info in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(info, dict) \
                or not isinstance(info.get("data_offsets"), list) \
                or len(info["data_offsets"]) != 2:
            raise DeltaValidationError(
                f"delta blob entry {key!r} is malformed")
        start, end = info["data_offsets"]
        if not (isinstance(start, int) and isinstance(end, int)
                and 0 <= start <= end and data_start + end <= len(blob_bytes)):
            raise DeltaValidationError(
                f"delta blob entry {key!r} data_offsets lie outside the blob")
    return header, data_start


@dataclass
class ParsedDelta:
    """A parsed delta blob: factor arrays + raw header + metadata targets."""
    tensors: dict           # factor key -> np.ndarray (float32)
    header: dict            # raw safetensors header (dtype/shape/offsets)
    metadata: dict          # safetensors __metadata__ (string values)
    metadata_targets: list  # parsed JSON of metadata["targets"]


def parse_blob(blob_bytes: bytes) -> ParsedDelta:
    """Parse delta blob bytes into factors + metadata, fail closed.

    Structural parse only: header well-formedness, metadata presence,
    and factor decoding are checked here; conformance against a
    record/base is :func:`validate_transformer_delta`'s job.
    """
    header, _ = _parse_safetensors_header(blob_bytes)
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict):
        raise DeltaValidationError("delta blob carries no safetensors metadata")
    raw_targets = metadata.get("targets")
    if not isinstance(raw_targets, str):
        raise DeltaValidationError(
            "delta blob metadata is missing the 'targets' list")
    try:
        metadata_targets = json.loads(raw_targets)
    except Exception as err:
        raise DeltaValidationError(
            f"delta blob metadata 'targets' is not JSON: {err}") from err
    if not isinstance(metadata_targets, list):
        raise DeltaValidationError(
            "delta blob metadata 'targets' is not a JSON array")
    from safetensors.numpy import load

    try:
        tensors = load(bytes(blob_bytes))
    except Exception as err:
        raise DeltaValidationError(f"delta blob factors do not decode: {err}") from err
    return ParsedDelta(tensors=dict(tensors), header=header,
                       metadata=dict(metadata),
                       metadata_targets=metadata_targets)


# ---------------------------------------------------------------------------
# Record serialisation / parsing
# ---------------------------------------------------------------------------

def serialize_record(record: dict) -> bytes:
    """Canonical JSON bytes for a manifest delta record."""
    return canonical_json(record)


def parse_record(data) -> dict:
    """Parse canonical (or any) JSON bytes/str into a record dict."""
    if isinstance(data, (bytes, bytearray)):
        data = bytes(data).decode("utf-8")
    try:
        record = json.loads(data)
    except Exception as err:
        raise DeltaValidationError(f"delta record is not JSON: {err}") from err
    if not isinstance(record, dict):
        raise DeltaValidationError("delta record is not a JSON object")
    return record


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def _expected_factor_keys(index: int, kind: str) -> list:
    if kind == "lora":
        return [f"t{index}.A", f"t{index}.B"]
    return [f"t{index}.d"]


def construct_transformer_delta(base_or_manifest: dict, targets: list, *,
                                delta_id: str, parent: str,
                                created_at: str, self_edit_id: str):
    """Build a validated (record, blob_bytes) pair from factor arrays.

    ``targets`` is a list of dicts:

    - lora:     ``{"name", "kind": "lora", "A": array, "B": array,
                  "alpha": number, "rank": int (optional; inferred
                  from A's shape when omitted, must agree when given)}``
    - additive: ``{"name", "kind": "additive", "d": array}``

    Base binding fields (``base_blob``/``base_shape``/``base_dtype``)
    are copied from the base manifest — callers never supply them, so
    a constructed record cannot lie about its base. Per-target norms
    and the aggregate (root-sum-square) norm are computed from the
    factors. The pair is run through
    :func:`validate_transformer_delta` before returning: construction
    never emits an invalid delta.
    """
    base_map = base_tensor_map(base_or_manifest)
    factors: dict = {}
    record_targets: list = []
    for i, spec in enumerate(targets):
        if not isinstance(spec, dict):
            raise DeltaValidationError(f"target {i} is not an object")
        name = spec.get("name")
        kind = spec.get("kind")
        if not isinstance(name, str) or not name:
            raise DeltaValidationError(f"target {i} has no valid name")
        if name not in base_map:
            raise DeltaValidationError(
                f"target {i} names unknown base tensor {name!r} "
                "(exact name match required)")
        base_entry = base_map[name]
        keys = _expected_factor_keys(i, kind) if kind in ("lora", "additive") else []
        if kind == "lora":
            if "A" not in spec or "B" not in spec:
                raise DeltaValidationError(
                    f"lora target {name!r} needs both 'A' and 'B' factors")
            A = np.asarray(spec["A"], dtype=np.float32)
            B = np.asarray(spec["B"], dtype=np.float32)
            rank = int(spec["rank"]) if spec.get("rank") is not None \
                else int(A.shape[0]) if A.ndim == 2 else 0
            alpha = spec.get("alpha")
            factors[keys[0]] = A
            factors[keys[1]] = B
            target = {"name": name, "base_blob": base_entry["blob"],
                      "base_shape": list(base_entry["shape"]),
                      "base_dtype": base_entry["dtype"], "kind": "lora",
                      "rank": rank, "alpha": alpha,
                      "factor_keys": keys,
                      "target_norm": compute_target_norm(
                          "lora", {"A": A, "B": B}, rank=rank, alpha=alpha)
                      if rank >= 1 and alpha is not None else float("nan")}
        elif kind == "additive":
            if "d" not in spec:
                raise DeltaValidationError(
                    f"additive target {name!r} needs a 'd' factor")
            d = np.asarray(spec["d"], dtype=np.float32)
            factors[keys[0]] = d
            target = {"name": name, "base_blob": base_entry["blob"],
                      "base_shape": list(base_entry["shape"]),
                      "base_dtype": base_entry["dtype"], "kind": "additive",
                      "factor_keys": keys,
                      "target_norm": compute_target_norm("additive", {"d": d})}
        else:
            raise DeltaValidationError(
                f"target {name!r} has unknown kind {kind!r}")
        record_targets.append(target)

    blob_bytes = serialize_blob(factors, record_targets)
    record = {
        "id": delta_id,
        "parent": parent,
        "kind": KIND,
        "blob": "sha256:" + sha256_hex(blob_bytes),
        "bytes": len(blob_bytes),
        "norm": aggregate_norm([t["target_norm"] for t in record_targets]),
        "created_at": created_at,
        "self_edit_id": self_edit_id,
        "targets": record_targets,
    }
    # Self-check: construction must never emit an invalid pair.
    validate_transformer_delta(record, blob_bytes, base_or_manifest)
    return record, blob_bytes


# ---------------------------------------------------------------------------
# Validation (pure, structural, fail closed)
# ---------------------------------------------------------------------------

def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _norms_agree(recorded: float, recomputed: float) -> bool:
    return math.isfinite(recorded) and math.isfinite(recomputed) and \
        abs(recorded - recomputed) <= _NORM_RTOL * max(1.0, abs(recomputed))


def validate_transformer_delta(record: dict, blob_bytes: bytes,
                               base_or_manifest: dict) -> None:
    """Structurally validate a record/blob pair against a base manifest.

    Pure validation (RFC §6.1): no gate, no loop, no base tensor reads
    — the base is checked through its manifest only. Returns None on
    success; raises DeltaValidationError listing every problem found
    on failure (fail closed).
    """
    problems: list[str] = []

    def bad(msg: str) -> None:
        problems.append(msg)

    if not isinstance(record, dict):
        raise DeltaValidationError("delta record is not an object")
    base_map = base_tensor_map(base_or_manifest)

    # -- record-level shape -------------------------------------------------
    if record.get("kind") != KIND:
        bad(f"record kind {record.get('kind')!r} != {KIND!r}")
    for field_name in ("id", "parent", "created_at", "self_edit_id"):
        if not isinstance(record.get(field_name), str) \
                or not record[field_name]:
            bad(f"record field {field_name!r} is missing or not a string")
    blob_ref = record.get("blob")
    if not isinstance(blob_ref, str) or not blob_ref.startswith("sha256:"):
        bad(f"record blob ref {blob_ref!r} is not a sha256: ref")
    elif isinstance(blob_bytes, (bytes, bytearray)):
        if "sha256:" + sha256_hex(bytes(blob_bytes)) != blob_ref:
            bad("record blob ref does not match the blob's content hash")
    if record.get("bytes") != (len(blob_bytes)
                               if isinstance(blob_bytes, (bytes, bytearray))
                               else None):
        bad(f"record bytes {record.get('bytes')!r} != blob length "
            f"{len(blob_bytes) if isinstance(blob_bytes, (bytes, bytearray)) else '?'}")
    record_norm = record.get("norm")
    if not _is_number(record_norm) or not math.isfinite(float(record_norm)) \
            or float(record_norm) < 0:
        bad(f"record norm {record_norm!r} is not a finite non-negative number")

    targets = record.get("targets")
    if not isinstance(targets, list) or not targets:
        bad("record targets is missing or empty")
        targets = []

    # -- blob parse ----------------------------------------------------------
    parsed: ParsedDelta | None = None
    try:
        parsed = parse_blob(blob_bytes)
    except DeltaValidationError as err:
        bad(str(err))
    if parsed is not None and parsed.metadata.get("format") != FORMAT:
        bad(f"blob metadata format {parsed.metadata.get('format')!r} "
            f"!= {FORMAT!r} (metadata/manifest disagreement is tamper)")

    # -- per-target checks ----------------------------------------------------
    seen_names: set = set()
    expected_blob_keys: set = set()
    recomputed_norms: list = []
    for i, t in enumerate(targets):
        label = f"targets[{i}]"
        if not isinstance(t, dict):
            bad(f"{label} is not an object")
            continue
        name = t.get("name")
        kind = t.get("kind")
        if not isinstance(name, str) or not name:
            bad(f"{label}.name is missing or not a string")
            continue
        label = f"targets[{i}] ({name})"
        if name in seen_names:
            bad(f"{label}: duplicate target name in one delta")
        seen_names.add(name)
        if kind not in ("lora", "additive"):
            bad(f"{label}: unknown kind {kind!r}")
            continue

        # Base binding: exact name, mode-independent hash (RFC §2.4).
        base_entry = base_map.get(name)
        if base_entry is None:
            bad(f"{label}: unknown base tensor (exact name match required)")
            continue
        if t.get("base_blob") != base_entry["blob"]:
            bad(f"{label}: base_blob {t.get('base_blob')!r} does not match "
                f"the base manifest's hash for this tensor")
        base_shape = base_entry["shape"]
        if t.get("base_shape") != base_shape:
            bad(f"{label}: base_shape {t.get('base_shape')!r} does not match "
                f"the base manifest's shape {base_shape!r}")
        if t.get("base_dtype") != base_entry["dtype"]:
            bad(f"{label}: base_dtype {t.get('base_dtype')!r} does not match "
                f"the base manifest's dtype {base_entry['dtype']!r}")

        # Dimensionality + kind eligibility.
        if len(base_shape) not in (1, 2):
            bad(f"{label}: base tensor is {len(base_shape)}-D; only 1-D "
                "and 2-D base tensors are targetable")
            continue
        if kind == "lora" and len(base_shape) != 2:
            bad(f"{label}: lora target must be 2-D, base is "
                f"{len(base_shape)}-D")
            continue
        if kind == "additive" and len(base_shape) != 1:
            bad(f"{label}: additive target must be 1-D, base is "
                f"{len(base_shape)}-D")
            continue

        # Kind-specific fields.
        rank = alpha = None
        if kind == "lora":
            rank = t.get("rank")
            alpha = t.get("alpha")
            if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
                bad(f"{label}: rank {rank!r} is not an integer >= 1")
                rank = None
            if not _is_number(alpha) or not math.isfinite(float(alpha or 0)) \
                    or float(alpha or 0) <= 0:
                bad(f"{label}: alpha {alpha!r} is not a positive finite number")
                alpha = None
        elif "rank" in t or "alpha" in t:
            bad(f"{label}: additive target must not carry rank/alpha")

        # Factor keys: exactly the indexed keys for this target.
        expected_keys = _expected_factor_keys(i, kind)
        if t.get("factor_keys") != expected_keys:
            bad(f"{label}: factor_keys {t.get('factor_keys')!r} != "
                f"expected {expected_keys!r} (missing/extra factor keys)")
        expected_blob_keys.update(expected_keys)

        # Blob-side conformance: presence, dtype, shape, finiteness, norm.
        target_norm = t.get("target_norm")
        if not _is_number(target_norm) or not math.isfinite(float(target_norm)) \
                or float(target_norm) < 0:
            bad(f"{label}: target_norm {target_norm!r} is not a finite "
                "non-negative number")
            target_norm = None
        if parsed is not None:
            factors: dict = {}
            shapes_ok = True
            for key in expected_keys:
                info = parsed.header.get(key)
                if info is None:
                    bad(f"{label}: factor {key!r} missing from blob")
                    shapes_ok = False
                    continue
                if info.get("dtype") != FACTOR_DTYPE_CODE:
                    bad(f"{label}: factor {key!r} dtype "
                        f"{info.get('dtype')!r} != {FACTOR_DTYPE_CODE!r} "
                        "(factors are stored float32)")
                    shapes_ok = False
            if kind == "lora" and rank is not None:
                out_dim, in_dim = base_shape
                want = {expected_keys[0]: [rank, in_dim],
                        expected_keys[1]: [out_dim, rank]}
                for key, want_shape in want.items():
                    info = parsed.header.get(key)
                    if info is not None and info.get("shape") != want_shape:
                        bad(f"{label}: factor {key!r} shape "
                            f"{info.get('shape')!r} != {want_shape!r} "
                            "(shape lie vs record/base)")
                        shapes_ok = False
            if kind == "additive":
                info = parsed.header.get(expected_keys[0])
                if info is not None and info.get("shape") != list(base_shape):
                    bad(f"{label}: factor {expected_keys[0]!r} shape "
                        f"{info.get('shape')!r} != base shape "
                        f"{list(base_shape)!r}")
                    shapes_ok = False
            if shapes_ok:
                try:
                    if kind == "lora":
                        factors = {"A": parsed.tensors[expected_keys[0]],
                                   "B": parsed.tensors[expected_keys[1]]}
                    else:
                        factors = {"d": parsed.tensors[expected_keys[0]]}
                except KeyError:
                    factors = {}
                    shapes_ok = False
            if shapes_ok and factors:
                flat = np.concatenate(
                    [np.asarray(v, dtype=np.float64).ravel()
                     for v in factors.values()])
                if not np.all(np.isfinite(flat)):
                    bad(f"{label}: factor values are not all finite")
                elif target_norm is not None and (
                        kind == "additive"
                        or (rank is not None and alpha is not None)):
                    recomputed = compute_target_norm(
                        kind, factors, rank=rank, alpha=alpha)
                    recomputed_norms.append(recomputed)
                    if not _norms_agree(float(target_norm), recomputed):
                        bad(f"{label}: target_norm {target_norm!r} != "
                            f"recomputed {recomputed!r} from blob factors")

    # -- blob key set: no extras, none missing --------------------------------
    if parsed is not None:
        blob_keys = {k for k in parsed.header if k != "__metadata__"}
        if blob_keys != expected_blob_keys:
            bad(f"blob factor keys {sorted(blob_keys)!r} != record's "
                f"expected {sorted(expected_blob_keys)!r} "
                "(missing/extra factor keys)")

    # -- metadata vs manifest (tamper check, decision 3) ----------------------
    if parsed is not None and targets and \
            all(isinstance(t, dict) for t in targets):
        try:
            expected_meta = _metadata_targets(targets)
        except Exception:
            expected_meta = None
        if expected_meta is not None and parsed.metadata_targets != expected_meta:
            bad("blob metadata targets disagree with the manifest record's "
                "targets (metadata/manifest disagreement is tamper)")

    # -- aggregate norm: root-sum-square (decision 1) --------------------------
    if _is_number(record_norm) and targets and \
            all(isinstance(t, dict) and _is_number(t.get("target_norm"))
                for t in targets):
        aggregate = aggregate_norm([float(t["target_norm"]) for t in targets])
        if not _norms_agree(float(record_norm), aggregate):
            bad(f"record norm {record_norm!r} != root-sum-square of "
                f"target norms {aggregate!r}")

    if problems:
        raise DeltaValidationError(
            "invalid transformer delta: " + "; ".join(problems))
