"""SI-16/SI-17: the probe ``BundleContext`` for miniature bundles
(RFC SI-13 §6.1) — probe suites + gate wiring.

This module is the loop/admit-layer context that lets the gate's
transformer branch finally *measure* checks 3 (identity continuity)
and 4 (capability floor) — for miniature bundles only.

SI-17 (recorded decision, docs/SI-17-admission-flip.md): the
§6.4 verdict mapping is now LIVE for miniature bundles with a
validated probe context — probe-fail -> REJECT, all-pass ->
ADMIT-eligible. SI-16's shadow mode (verdict held at ESCALATE
regardless of outcome) is superseded for those bundles only.
Every other bundle/context keeps SI-16/SI-9b behaviour exactly:
checks 3/4 "not evaluated" and ESCALATE. One SI-17 addition: the
probe-blind guard (see :func:`derive_probe_blind_targets`) — a
candidate whose targets are exclusively probe-blind tensors
measures drift 0.0 vacuously, which is NOT a measured pass; it
ESCALATEs as blind-escalated.
- Probe results count only for the bundle they were recorded on
  (§6.2 binding, verified here from the bundle context — never
  from the candidate): the probe artifact must be present, its
  content hash must match the manifest, its recorded base binding
  (the base tensor map — names/shapes/dtypes + per-tensor content
  hashes, hashed as a unit by SI-14) must match the verified
  manifest exactly, and its architecture descriptor must match
  §3.1 exactly.
- Harness startup validation (§6.3): before any probe result
  counts, this context re-runs the base (base tensors only, no
  deltas) through the SI-15 forward path and must reproduce the
  recorded baselines within float tolerance (max baseline error
  ``<= 1e-6``; SI-15 proved the reproduction is in fact
  bit-identical on this runtime, and the recorded baselines are
  0.0 by construction). Any failure — binding mismatch, missing
  artifact, unreproducible baseline, any exception — is
  "not evaluated", exactly as the gate has always reported for
  transformer deltas: never a pass, never a candidate failure.
- The metric code is NOT forked: this context exposes the gate's
  existing ``BundleContext`` protocol (``identity_probes()``,
  ``capability_suite()``, ``capability_baseline()``,
  ``forward_candidate(x, candidate)``), so ``Gate._identity_check``
  / ``Gate._capability_check`` run unchanged. ``forward_candidate``
  is the §4.2 candidate overlay (``forward_from_bundle`` with a
  candidate), returning the last position's logits — the vector
  the suites record. The one new measurement is §5.3's echo
  accuracy arm, exposed via ``capability_targets()`` /
  ``echo_floor()`` for the gate's check-4 composition.

Scope honesty:

- This is a **miniature executor context**, not a general probe
  harness. Non-miniature bundles (the Glimmer 30B seed above all)
  carry no probe artifact and can never satisfy §6.2 here — there
  is no forward pass for them in this runtime, and this module
  does not pretend otherwise. Their gate behaviour is byte-for-byte
  what it was before SI-16.
- Nothing here trains, authors deltas, serves, or admits. The
  context is read-only: probing a candidate lands nothing.
- CPU only, existing dependencies only (numpy + stdlib here).
  This is machine learning infrastructure, not "custom LLM
  software."
"""
from __future__ import annotations

import json
import os

import numpy as np

from si.bundle import _active_chain, canonical_json, load_base_tensor, sha256_hex
from si.tiny_seed import TINY_KIND
from si.tiny_transformer import (
    EXPECTED_ARCHITECTURE,
    TinyTransformerError,
    forward,
    forward_from_bundle,
    validate_architecture,
)

__all__ = [
    "BASELINE_TOL",
    "ProbeBindingError",
    "ProbeContext",
    "derive_probe_blind_targets",
    "is_miniature_probe_bundle",
    "load_bound_probe_artifact",
]

#: Startup-validation tolerance (§6.3): the reproduced base errors
#: must match the recorded baselines within this float tolerance.
#: SI-15 proved bit-identical reproduction (error exactly 0.0) on
#: this runtime; the tolerance exists so the check is stated as the
#: RFC states it — "within float tolerance" — not as an assumption
#: that every future numpy build rounds identically.
BASELINE_TOL = 1e-6


def derive_probe_blind_targets(architecture: dict | None = None) -> set:
    """The probe-blind tensor set, derived — not a magic list.

    Derivation (mechanism, corroborated by SI-16's §5.6 per-class
    sensitivity measurement): in this architecture each layer's
    keys are ``k_j = a_j @ W_k^T + b_k``, where ``b_k`` is the
    layer's ``self_attn.k_proj.bias`` — the SAME vector added to
    every position ``j``. A query ``q_i`` therefore scores
    ``q_i . k_j = q_i . (a_j @ W_k^T) + q_i . b_k``; the second
    term does not depend on ``j``, so it adds a constant to all of
    query ``i``'s attention scores, and softmax over ``j`` is
    invariant to a constant shift. No probe input can ever detect
    a change to ``b_k`` through this architecture's outputs: drift
    is exactly 0.0 for mathematical reasons, not measurement
    noise. SI-16 measured exactly this: 28 of the 30 targetable
    tensors are measurable, and the blind set is precisely the
    ``k_proj.bias`` of both layers (drift exactly 0.0).

    The set is therefore DERIVED here from the architecture: for
    each layer index in the §3.1 descriptor, the one tensor whose
    mechanism is "a constant added identically to all keys" —
    ``model.language_model.layers.{i}.self_attn.k_proj.bias``.
    No other tensor has that mechanism: ``q_proj.bias`` shifts
    queries (scores change per-key), ``v_proj.bias`` shifts the
    values that are averaged (outputs change), ``o_proj.bias``
    shifts the residual stream directly, and every weight /
    norm tensor enters multiplicatively or per-position.
    """
    arch = architecture or EXPECTED_ARCHITECTURE
    layers = int(arch["layers"])
    return {
        f"model.language_model.layers.{i}.self_attn.k_proj.bias"
        for i in range(layers)
    }


class ProbeBindingError(ValueError):
    """The §6.2 binding (or §6.3 startup validation) does not hold
    for this bundle: probes are "not evaluated" for it. This is
    never a candidate failure and never a pass."""


def is_miniature_probe_bundle(manifest) -> bool:
    """True iff the manifest's base is the SI-14 miniature kind.

    This is only the *kind* test the admit layer uses to choose a
    context; probe evaluation itself still requires the full §6.2
    binding + §6.3 startup validation in :class:`ProbeContext`.
    """
    if not isinstance(manifest, dict):
        return False
    base = manifest.get("base")
    return isinstance(base, dict) and base.get("kind") == TINY_KIND


def load_bound_probe_artifact(bundle_dir: str, manifest: dict):
    """Verify §6.2 binding; return ``(artifact, artifact_sha256)``.

    All checks are against the bundle context's verified manifest
    (never the candidate):

    1. the manifest records a probe artifact and the file's content
       hash matches the manifest's recorded hash;
    2. the artifact's recorded base binding — SI-14's hash of the
       base tensor map (names/shapes/dtypes + per-tensor content
       hashes, hashed as a unit), plus its tensor count and total
       elements — matches the manifest exactly;
    3. the artifact's architecture descriptor matches §3.1 exactly,
       and so does the manifest base's own descriptor.

    Raises :class:`ProbeBindingError` with the specific reason on
    any mismatch.
    """
    base = manifest.get("base") if isinstance(manifest, dict) else None
    if not isinstance(base, dict):
        raise ProbeBindingError("manifest carries no base section")
    if base.get("kind") != TINY_KIND:
        raise ProbeBindingError(
            f"base kind {base.get('kind')!r} is not the miniature kind "
            f"{TINY_KIND!r}: no probe harness exists for this base")
    recorded = base.get("probe_artifact")
    if not isinstance(recorded, dict) \
            or recorded.get("file") != "probe_suite.json" \
            or not str(recorded.get("sha256", "")).startswith("sha256:"):
        raise ProbeBindingError(
            "bundle carries no probe artifact in its manifest "
            "(base.probe_artifact missing or malformed): no probe "
            "suite was recorded for this bundle")
    path = os.path.join(bundle_dir, "probe_suite.json")
    if not os.path.isfile(path):
        raise ProbeBindingError(
            "manifest records a probe artifact but probe_suite.json "
            "is missing from the bundle")
    with open(path, "rb") as f:
        raw = f.read()
    digest = "sha256:" + sha256_hex(raw)
    if digest != recorded["sha256"]:
        raise ProbeBindingError(
            f"probe artifact content hash {digest} does not match "
            f"the manifest's recorded hash {recorded['sha256']}: "
            "the suite on disk is not the suite the manifest binds")
    try:
        artifact = json.loads(raw.decode("utf-8"))
    except Exception as err:
        raise ProbeBindingError(
            f"probe artifact is not readable JSON: {err}") from err
    if not isinstance(artifact, dict) \
            or artifact.get("artifact") != "si-probe-suite-v1" \
            or artifact.get("kind") != TINY_KIND:
        raise ProbeBindingError(
            "probe artifact is not an si-probe-suite-v1 artifact for "
            "the miniature kind")

    # Rule 3 (artifact side + manifest side): §3.1 exactly.
    try:
        validate_architecture(artifact.get("architecture"))
        validate_architecture(base.get("architecture"))
    except TinyTransformerError as err:
        raise ProbeBindingError(
            f"architecture descriptor does not match SI-13 §3.1 "
            f"exactly: {err}") from err
    if artifact.get("architecture") != EXPECTED_ARCHITECTURE:
        raise ProbeBindingError(
            "probe artifact architecture descriptor differs from "
            "the SI-13 §3.1 miniature descriptor")

    # Rule 2: the base binding, hashed as a unit, must match the
    # verified manifest exactly (SI-14 recorded the hash of the
    # canonical tensor_blobs map; recompute it from the manifest).
    tensor_blobs = base.get("tensor_blobs")
    if not isinstance(tensor_blobs, dict) or not tensor_blobs:
        raise ProbeBindingError(
            "manifest base carries no per-tensor map to bind the "
            "probe artifact against")
    binding = artifact.get("base_binding")
    if not isinstance(binding, dict):
        raise ProbeBindingError("probe artifact records no base binding")
    recomputed = "sha256:" + sha256_hex(canonical_json(tensor_blobs))
    if binding.get("tensor_map_sha256") != recomputed:
        raise ProbeBindingError(
            f"probe artifact base binding {binding.get('tensor_map_sha256')!r} "
            f"does not match the bundle's verified tensor map "
            f"{recomputed}: the suite was recorded against a "
            "different base")
    total_elements = int(sum(
        int(np.prod(ref["shape"])) for ref in tensor_blobs.values()))
    if binding.get("tensor_count") != len(tensor_blobs) \
            or binding.get("total_elements") != total_elements:
        raise ProbeBindingError(
            "probe artifact base binding tensor count / total "
            "elements do not match the bundle's verified tensor map")
    return artifact, digest


class ProbeContext:
    """The gate's ``BundleContext`` for a miniature probe bundle.

    Sibling to ``si.loop.LoopContext`` (demo bundles) and
    ``si.admit._AdmissionContext`` (non-miniature transformer
    admission): same protocol, real measurements. The gate detects
    this context by the ``probe_harness`` marker and runs its
    generic ``_identity_check`` / ``_capability_check`` against
    it (SI-17: the §6.4 verdict mapping consumes the results).

    Construction never raises for a bad bundle: binding and
    startup validation run lazily in :meth:`probe_status`, which
    reports ``(False, reason)`` — the gate then reports checks 3/4
    "not evaluated" with that reason, exactly per §6.3.
    """

    #: Marker the gate's transformer branch looks for (duck-typed,
    #: so the gate grows no import of this module and demo contexts
    #: — which have no marker — keep today's behaviour untouched).
    probe_harness = True

    def __init__(self, bundle_dir: str, manifest: dict,
                 policy: dict | None = None):
        self.bundle_dir = bundle_dir
        self.manifest = manifest
        self.policy = policy
        # SI-43 (RFC SI-41 §15 B-2): the miniature loop path is the
        # LIVE tool-execution path — this context carries the gate's
        # tool-context contract with live execution on, so a tool
        # intent gated here can ADMIT for the B-2 read/write set
        # (si.gaia.executor runs it). The demo LoopContext carries
        # the same attributes with live execution OFF (B-1 shadow).
        # Task bookkeeping defaults are the out-of-task values; a
        # task driver (B-3) sets them per task.
        self.tool_live_execution = True
        self.tool_work_root: str | None = None
        self.tool_task_seq: int | None = None
        self.tool_cycles_used = 0
        self.tool_result_bytes_used = 0
        # SI-51 (SI-50 §6/§7): queue-pricing context, set per batch
        # by the task loop (see si.gaia.context.ToolContext).
        self.tool_queue_batch_seq: int | None = None
        self.tool_journal_tip: int | None = None
        self.tool_task_intent_seqs = None
        self._status: tuple[bool, str] | None = None
        self._artifact: dict | None = None
        self._artifact_hash: str | None = None
        self._startup_report: dict | None = None
        self._baseline: dict | None = None

    # -- context plumbing --------------------------------------------------
    def base_manifest(self):
        return self.manifest

    def stack_depth(self) -> int:
        return len(_active_chain(self.manifest))

    @property
    def probe_artifact_hash(self) -> str | None:
        return self._artifact_hash

    @property
    def startup_report(self) -> dict | None:
        """The §6.3 startup-validation measurements, once run."""
        return self._startup_report

    # -- §6.2 binding + §6.3 startup validation -----------------------------
    def probe_status(self) -> tuple[bool, str]:
        """``(True, "ok")`` iff probes may be evaluated for real.

        Runs (once, cached) the §6.2 binding checks and then §6.3
        startup validation: the base — base tensors only, through
        the SI-15 forward path — must reproduce every recorded
        baseline within :data:`BASELINE_TOL`, and its measured echo
        accuracy must equal the accuracy recorded at seed time.
        Any exception anywhere is a ``(False, reason)`` result:
        "not evaluated", never a pass and never a candidate
        failure.
        """
        if self._status is not None:
            return self._status
        try:
            artifact, digest = load_bound_probe_artifact(
                self.bundle_dir, self.manifest)
            report = self._validate_startup(artifact)
        except ProbeBindingError as err:
            self._status = (False, str(err))
            return self._status
        except Exception as err:  # harness failure == not evaluated (§6.3)
            self._status = (False,
                            f"probe harness startup raised "
                            f"{type(err).__name__}: {err}")
            return self._status
        self._artifact = artifact
        self._artifact_hash = digest
        self._startup_report = report
        self._status = (True, "ok")
        return self._status

    def _validate_startup(self, artifact: dict) -> dict:
        from si.tiny_seed import tiny_tensor_shapes

        weights = {name: load_base_tensor(self.bundle_dir,
                                          self.manifest, name)
                   for name in tiny_tensor_shapes()}
        worst = 0.0
        baselines = artifact.get("baselines") or {}
        for suite_key, base_key in (("identity_suite", "identity"),
                                    ("capability_suite", "capability")):
            recorded = baselines.get(base_key) or {}
            for entry in artifact[suite_key]:
                logits = forward(entry["input"], weights)[-1]
                err = float(np.max(np.abs(
                    np.asarray(logits, dtype=np.float64)
                    - np.asarray(entry["expected_logits"],
                                 dtype=np.float64))))
                want = float(recorded.get(entry["id"], 0.0))
                worst = max(worst, abs(err - want))
        if worst > BASELINE_TOL:
            raise ProbeBindingError(
                f"harness startup validation cannot reproduce the "
                f"recorded base baselines: worst deviation {worst:.6g} "
                f"> float tolerance {BASELINE_TOL:g} — the forward "
                "path does not reproduce this bundle's recorded "
                "baselines, so no probe result counts (§6.3)")
        correct = 0
        for entry in artifact["capability_suite"]:
            logits = forward(entry["input"], weights)[-1]
            if int(np.argmax(logits)) == int(entry["target_token"]):
                correct += 1
        measured_acc = correct / len(artifact["capability_suite"])
        recorded_acc = float(artifact.get("base_echo_accuracy", -1.0))
        if measured_acc != recorded_acc:
            raise ProbeBindingError(
                f"harness startup validation measured base echo "
                f"accuracy {measured_acc} != the recorded "
                f"{recorded_acc}: the base does not reproduce its "
                "recorded capability baseline (§6.3)")
        return {"max_baseline_deviation": worst,
                "tolerance": BASELINE_TOL,
                "base_echo_accuracy": measured_acc,
                "entries_checked": (len(artifact["identity_suite"])
                                    + len(artifact["capability_suite"]))}

    def _require_artifact(self) -> dict:
        ok, reason = self.probe_status()
        if not ok or self._artifact is None:
            raise ProbeBindingError(
                f"probe suite is not evaluated for this bundle: {reason}")
        return self._artifact

    # -- BundleContext protocol (real measurements) --------------------------
    def identity_probes(self) -> list:
        art = self._require_artifact()
        return [(e["id"], [int(x) for x in e["input"]],
                 np.asarray(e["expected_logits"], dtype=np.float32))
                for e in art["identity_suite"]]

    def capability_suite(self) -> list:
        art = self._require_artifact()
        return [(e["id"], [int(x) for x in e["input"]],
                 np.asarray(e["expected_logits"], dtype=np.float32))
                for e in art["capability_suite"]]

    def capability_targets(self) -> dict:
        """{cid: echo target token} — §5.3's functional expectation."""
        art = self._require_artifact()
        return {e["id"]: int(e["target_token"])
                for e in art["capability_suite"]}

    def echo_floor(self) -> float:
        """The §5.3 accuracy floor recorded at seed time
        (base accuracy − 1/16)."""
        art = self._require_artifact()
        return float(art["echo_floor"])

    def base_echo_accuracy(self) -> float:
        art = self._require_artifact()
        return float(art["base_echo_accuracy"])

    def capability_baseline(self) -> dict:
        """{cid: error} at the current head (pre-candidate).

        Measured live through the forward path over base + active
        chain — the LoopContext semantics transplanted: at head =
        base these are the recorded 0.0 baselines; on a non-base
        head they are the pre-candidate head's real errors.
        """
        if self._baseline is not None:
            return dict(self._baseline)
        baseline = {}
        for cid, x, expected in self.capability_suite():
            logits = forward_from_bundle(
                self.bundle_dir, self.manifest, x,
                policy=self.policy)[-1]
            baseline[cid] = float(np.max(np.abs(
                np.asarray(logits, dtype=np.float64)
                - np.asarray(expected, dtype=np.float64))))
        self._baseline = baseline
        return dict(baseline)

    def forward_candidate(self, x, candidate):
        """Last-position logits for ``x`` with ``candidate`` overlaid
        (§4.2: base + active chain + the candidate's validated fold,
        in memory only — probing lands nothing)."""
        return forward_from_bundle(
            self.bundle_dir, self.manifest, x,
            policy=self.policy, candidate=candidate)[-1]

    def forward_head(self, x):
        """Last-position logits for ``x`` at the current head
        (base + active chain, NO candidate) — the §6.6 VERIFY
        measurement against a landed state."""
        return forward_from_bundle(
            self.bundle_dir, self.manifest, x,
            policy=self.policy, candidate=None)[-1]

    # -- SI-17 blind-set guard ----------------------------------------------
    def probe_blind_targets(self) -> set:
        """The derived probe-blind set for this bundle's
        architecture (see :func:`derive_probe_blind_targets`)."""
        art = self._artifact
        arch = art.get("architecture") if isinstance(art, dict) else None
        return derive_probe_blind_targets(arch)

    def candidate_measurability(self, candidate) -> dict:
        """Split a candidate's targets into measurable / blind.

        Returns {"targets": [...], "measurable": [...],
        "blind": [...], "all_blind": bool}. Target names come from
        the candidate RECORD (the gate's safety suite independently
        binds them to the base manifest); an unreadable record
        yields no targets, in which case the safety suite — not
        this guard — decides the outcome.
        """
        record = candidate.get("record") if isinstance(candidate, dict) else None
        names: list = []
        if isinstance(record, dict) and isinstance(record.get("targets"), list):
            names = [t["name"] for t in record["targets"]
                     if isinstance(t, dict) and isinstance(t.get("name"), str)]
        blind_set = self.probe_blind_targets()
        blind = [n for n in names if n in blind_set]
        measurable = [n for n in names if n not in blind_set]
        return {"targets": names, "measurable": measurable,
                "blind": blind,
                "all_blind": bool(names) and not measurable}
