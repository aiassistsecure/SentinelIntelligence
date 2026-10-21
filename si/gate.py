"""The gate: every self-write is validated before it lands.

Implements the gate checklist from docs/self-write-loop.md. Ternary
verdicts: ADMIT | REJECT | ESCALATE. The five checks:

1. constitution — reason present + no policy-blocklisted patterns
2. safety bounds — rank cap, allowed modules, stack depth, delta norm cap
3. identity continuity — candidate must not drift identity probes past tol
4. capability floor — no eval regression beyond epsilon vs baseline
5. provenance completeness — parents + source seqs present

Rejections are journaled by the caller (the loop); the gate just reports.
``kind == "constitution"`` always ESCALATES. ``kind == "rollback"`` is
evaluated lightly (constitution check only) but is still journaled.

SI-9b — transformer deltas (RFC SI-9 §4, §6.2). An intent whose
``kind`` is ``"transformer-delta-v1"`` (the SI-9a record kind) is
gated by a different, honest mapping of the same five checks:

- constitution: unchanged.
- safety bounds: the §4.2 structural suite (implemented in
  :meth:`Gate._transformer_safety_check`), run against the base
  *manifest* and the delta blob only — base tensor bytes are never
  read by the gate. Each sub-check is recorded on the safety check's
  ``subchecks`` list, which is journaled with the gate report.
- identity continuity / capability floor: **cannot be evaluated**
  for a general transformer base — this runtime has no forward pass
  and no probe suite for one (RFC §4.3). Both checks are reported
  as not passed, with ``measured=None`` and the plain reason, and a
  transformer delta that passes constitution + safety + provenance
  is ESCALATED to the operator — never ADMITTED, never given a
  synthetic pass. Under SI-9b transformer deltas are
  operator-admitted, not autonomously admitted; the autonomous
  loop already halts on ESCALATE.
- SI-16/SI-17 (RFC SI-13 §6): when the bundle context is a
  probe context for a *miniature* bundle
  (``si.probes.ProbeContext`` — §6.2 binding verified, §6.3
  startup validation passed), checks 3/4 run for real through the
  gate's own generic check logic plus §5.3's echo accuracy arm,
  and their real measured results are journaled. SI-17's
  recorded decision (docs/SI-17-admission-flip.md) makes §6.4's
  verdict mapping LIVE for those bundles: a measured probe
  failure REJECTs; a measured all-pass makes the delta
  ADMIT-eligible. One guard: a candidate whose targets are
  exclusively probe-blind tensors (derived in
  ``si.probes.derive_probe_blind_targets`` — drift 0.0
  vacuously, NOT a measured pass) is blind-escalated, never
  ADMIT-eligible. Every other context (demo contexts,
  admission contexts, miniature contexts whose binding/startup
  validation fails) keeps the SI-9b behaviour above,
  byte-for-byte.
- provenance: the demo requirements plus §4.4's extension — the
  intent must declare its target list (``Intent.targets``, exact base
  tensor names) and it must match the candidate record's targets.

Candidate contract for a transformer intent (pinned here; the
candidate is an untrusted proposal):

    {"record": <SI-9a manifest delta record dict>,
     "blob_bytes": <safetensors blob bytes>,
     "norm": <optional float; if present it must agree with
              record["norm"]>}

The base manifest is taken ONLY from the bundle context
(``ctx.base_manifest()`` or a ``ctx.manifest`` dict) — never from the
candidate, so a candidate cannot validate itself against a base of
its own choosing. A context that cannot supply a base manifest fails
closed. Per-target ranks/alphas in the *record* are authoritative
for this kind; the demo-era ``Intent.rank`` / ``Intent.alpha`` fields
are not consulted. Mixed stacks are a non-goal: a transformer delta
against a demo base fails target existence (a demo base records no
per-tensor map, so every target is unknown), and a transformer
candidate submitted under a demo ``"delta"`` intent is rejected as a
kind mismatch. The demo-delta path itself is unchanged.

Policy (RFC §4.5; defaults live in ``si.bundle.DEFAULT_POLICY``).
Keys absent from a policy fall back to the §4.5 defaults pinned in
:data:`TRANSFORMER_POLICY_DEFAULTS` — empty allowlist (admits
nothing), 1 target, 4 MiB byte cap, per-target norm cap equal to the
policy's ``norm_cap``, additive off. A key present but malformed
(wrong type / negative) is treated as the most restrictive value
(0 / empty / off): a broken policy fails closed, never open.

Norm method (RFC §4.2.5, §7.2 — the choice SI-9b must pin): the gate
recomputes each LoRA target's norm from the Gram matrices in float64
— ``||B@A||_F^2 = trace((B^T B) @ (A A^T))`` — never materialising
the ``[out, in]`` product; additive norms are ``||d||_2`` in float64.
The Gram value must agree with the record's ``target_norm`` (already
checked against a direct ``B@A`` computation by SI-9a validation) to
relative tolerance 1e-6; disagreement is a safety failure. Cap
comparisons are strict (``measured > cap`` fails) on the float64
Gram-measured values, and the boundary is tested at exactly the cap
(pass) and just above it (fail) in ``tests/test_gate_transformer.py``.

Allowlist semantics (RFC §4.2.7, §7.3): ``allowed_target_patterns``
entries are ``fnmatch`` patterns matched case-sensitively against the
exact target name. A pattern with no wildcard is an exact name — no
implicit substring or prefix matching is ever applied.
"""
from __future__ import annotations

import fnmatch
import json
import math
from dataclasses import dataclass, field
from typing import Protocol

from si.transformer_delta import (
    KIND as TRANSFORMER_DELTA_KIND,
    DeltaValidationError,
    base_tensor_map,
    parse_blob,
    validate_transformer_delta,
)

#: RFC §4.5 policy defaults for transformer deltas, as pinned by
#: SI-9b (mirrors si.bundle.DEFAULT_POLICY; used when a key is absent
#: from a bundle's policy). per_target_norm_cap is resolved against
#: the policy's norm_cap instead of a constant — see _transformer_policy.
TRANSFORMER_POLICY_DEFAULTS = {
    "allowed_target_patterns": [],
    "max_targets_per_delta": 1,
    "delta_byte_cap": 4 * 1024 * 1024,
    "allow_additive": False,
}

#: Honest reason recorded on both probe checks for transformer deltas.
TRANSFORMER_PROBES_NOT_EVALUATED = (
    "not evaluated: no transformer probe harness exists — this runtime "
    "has no transformer forward pass, no probe suite, and no baseline "
    "for a transformer base (RFC SI-9 §4.3); reporting a pass would be "
    "fabrication, so the delta is escalated to the operator instead")


@dataclass
class Intent:
    """A self-write proposal. The runtime trains adapters; the mind never
    touches raw tensors."""
    kind: str  # "delta" | "memory" | "rollback" | "constitution"
    reason: str = ""  # REQUIRED — empty/missing is malformed -> reject
    # delta fields
    target_modules: list = field(default_factory=list)
    rank: int = 0
    alpha: int = 0
    parents: list = field(default_factory=list)
    source_seqs: list = field(default_factory=list)
    training_signal_id: str = ""
    # memory fields
    collection: str = ""
    record: dict = field(default_factory=dict)
    # rollback field
    target: str = ""
    # transformer-delta field (SI-9b, RFC §4.4): the intent's declared
    # target list — exact base tensor names, in record order. Required
    # (and checked against the candidate record) for transformer
    # intents; empty and ignored for every other kind.
    targets: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "reason": self.reason,
            "target_modules": list(self.target_modules),
            "rank": self.rank,
            "alpha": self.alpha,
            "parents": list(self.parents),
            "source_seqs": list(self.source_seqs),
            "training_signal_id": self.training_signal_id,
            "collection": self.collection,
            "record": self.record,
            "target": self.target,
            "targets": list(self.targets),
        }


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    measured: float | None = None
    # Optional per-sub-check breakdown (SI-9b transformer safety
    # suite). None for ordinary checks, so their journaled dict form
    # is byte-for-byte what it was before SI-9b.
    subchecks: list | None = None

    def to_dict(self) -> dict:
        out = {"name": self.name, "passed": self.passed,
               "detail": self.detail, "measured": self.measured}
        if self.subchecks is not None:
            out["subchecks"] = [c.to_dict() for c in self.subchecks]
        return out


@dataclass
class GateReport:
    verdict: str  # ADMIT | REJECT | ESCALATE
    checks: list  # list[CheckResult]

    def to_dict(self) -> dict:
        return {"verdict": self.verdict,
                "checks": [c.to_dict() for c in self.checks]}

    @property
    def admitted(self) -> bool:
        return self.verdict == "ADMIT"


class BundleContext(Protocol):
    """What the gate needs from the bundle + mind under test."""

    def identity_probes(self) -> list:
        """[(qid, x, expected), ...] — the mind must still be itself."""
        ...

    def capability_suite(self) -> list:
        """[(cid, x, expected), ...] — the seed's acceptance evals."""
        ...

    def capability_baseline(self) -> dict:
        """{cid: error} measured at the current head (pre-candidate)."""
        ...

    def forward_candidate(self, x, candidate):
        """Run input x through base + active deltas + the candidate."""
        ...

    def stack_depth(self) -> int:
        """Number of deltas currently on the active stack."""
        ...

    # Optional, transformer deltas only (SI-9b): a context MAY also
    # provide ``base_manifest()`` (or a ``manifest`` dict attribute)
    # returning the bundle's manifest dict. The gate reads the base
    # tensor map from it — manifest only, never base tensor bytes.
    # Contexts without it cannot gate transformer deltas (fail closed).


def _err(y, expected) -> float:
    import numpy as np
    return float(np.max(np.abs(np.asarray(y, dtype=np.float64)
                               - np.asarray(expected, dtype=np.float64))))


def _lora_norm_gram(B, A, scale: float) -> float:
    """``||scale * B @ A||_F`` via the Gram matrices, in float64.

    ``||B@A||_F^2 = trace((B^T B) @ (A A^T))`` — exact in exact
    arithmetic, and it never materialises the ``[out, in]`` product
    (RFC §4.2.5's method choice for SI-9b; see the module docstring
    for the tolerance story). B is [out, rank], A is [rank, in], so
    both Gram matrices are [rank, rank].
    """
    import numpy as np
    B64 = np.asarray(B, dtype=np.float64)
    A64 = np.asarray(A, dtype=np.float64)
    gram_b = B64.T @ B64
    gram_a = A64 @ A64.T
    sq = float(np.trace(gram_b @ gram_a))
    if sq < 0.0:
        # Only reachable as float64 rounding noise on a near-zero
        # norm (the true value is a sum of squares, hence >= 0);
        # clamp rather than take sqrt of a negative.
        sq = 0.0
    return abs(float(scale)) * math.sqrt(sq)


class Gate:
    def __init__(self, policy: dict):
        self.policy = dict(policy)

    # -- public ------------------------------------------------------------
    def evaluate(self, intent: Intent, candidate, ctx: BundleContext) -> GateReport:
        checks: list[CheckResult] = []

        # Malformed intents never reach the checks.
        if intent.kind not in ("delta", "memory", "rollback", "constitution",
                               TRANSFORMER_DELTA_KIND):
            return GateReport("REJECT", [CheckResult(
                "wellformed", False,
                f"unknown intent kind {intent.kind!r}", None)])
        if not intent.reason or not intent.reason.strip():
            return GateReport("REJECT", [CheckResult(
                "wellformed", False,
                "reason is required and missing: a mind that can't say why "
                "it's changing itself isn't ready to", None)])

        checks.append(self._constitution_check(intent))
        if intent.kind == "constitution":
            # Autonomous constitution amendment is never admitted in SI-0.
            return GateReport("ESCALATE", checks)
        if intent.kind == "rollback":
            # Evaluated lightly, still journaled by the caller.
            verdict = "ADMIT" if all(c.passed for c in checks) else "REJECT"
            return GateReport(verdict, checks)
        if intent.kind == "memory":
            checks.append(self._memory_check(intent))
            verdict = "ADMIT" if all(c.passed for c in checks) else "REJECT"
            return GateReport(verdict, checks)

        if intent.kind == TRANSFORMER_DELTA_KIND:
            # SI-9b: the five checks, mapped honestly for a kind this
            # runtime cannot run probes against (RFC §4).
            if candidate is None:
                return GateReport("REJECT", checks + [CheckResult(
                    "wellformed", False,
                    "transformer-delta intent arrived with no candidate", None)])
            checks.append(self._transformer_safety_check(intent, candidate, ctx))
            # §4.3: NOT passes — the checks could not be evaluated.
            # SI-16/SI-17: for a miniature probe context (§6.2
            # binding + §6.3 startup validation green) these run
            # for real and §6.4's mapping is live; every other
            # context reports them exactly as SI-9b did. See
            # _transformer_probe_checks.
            id_check, cap_check = self._transformer_probe_checks(
                intent, candidate, ctx)
            checks.append(id_check)
            checks.append(cap_check)
            checks.append(self._transformer_provenance_check(intent, candidate))
            # Verdict mapping (RFC SI-13 §6.4, SI-17):
            # - decisive checks (constitution/safety/provenance)
            #   failing -> REJECT, unchanged in all cases;
            # - probes NOT evaluated (measured is None — including
            #   the blind-escalated case): the two checks are
            #   unevaluated, not failed; they alone never cause a
            #   REJECT and they block any ADMIT -> ESCALATE,
            #   exactly the SI-9b outcome;
            # - probes genuinely evaluated: a measured failure is
            #   evidence of harm -> REJECT; a measured all-pass
            #   counts toward ADMIT -> ADMIT.
            decisive = [c for c in checks
                        if c.name not in ("identity_continuity",
                                          "capability_floor")]
            if not all(c.passed for c in decisive):
                return GateReport("REJECT", checks)
            if id_check.measured is not None and cap_check.measured is not None:
                if id_check.passed and cap_check.passed:
                    return GateReport("ADMIT", checks)
                return GateReport("REJECT", checks)
            return GateReport("ESCALATE", checks)

        # kind == "delta": the full five checks.
        if candidate is None:
            return GateReport("REJECT", checks + [CheckResult(
                "wellformed", False, "delta intent arrived with no candidate", None)])
        if isinstance(candidate, dict) and isinstance(candidate.get("record"), dict) \
                and candidate["record"].get("kind") == TRANSFORMER_DELTA_KIND:
            # Mixed kinds are a non-goal (RFC §3): a transformer
            # candidate must arrive under its own intent kind, where
            # the §4 mapping applies — not dressed as a demo delta.
            return GateReport("REJECT", checks + [CheckResult(
                "wellformed", False,
                "transformer-delta candidate submitted under a demo "
                "'delta' intent: mixed kinds are not admitted", None)])
        checks.append(self._safety_check(intent, candidate, ctx))
        checks.append(self._identity_check(intent, candidate, ctx))
        checks.append(self._capability_check(intent, candidate, ctx))
        checks.append(self._provenance_check(intent))

        if not all(c.passed for c in checks):
            return GateReport("REJECT", checks)
        if self._needs_escalation(intent, candidate):
            return GateReport("ESCALATE", checks)
        return GateReport("ADMIT", checks)

    # -- checks ------------------------------------------------------------
    def _constitution_check(self, intent: Intent) -> CheckResult:
        text = (intent.reason + " " + json.dumps(intent.to_dict())).lower()
        hits = [p for p in self.policy.get("blocklist", []) if p.lower() in text]
        return CheckResult(
            "constitution",
            passed=not hits,
            detail=("no blocklisted patterns" if not hits
                    else f"blocklisted pattern(s) in intent: {hits}"),
            measured=float(len(hits)),
        )

    def _memory_check(self, intent: Intent) -> CheckResult:
        ok = intent.collection in ("experiences",)
        return CheckResult(
            "memory_target",
            passed=ok,
            detail=(f"collection={intent.collection!r} allowed" if ok
                    else f"collection={intent.collection!r} not writable by memory intents"),
            measured=None,
        )

    def _safety_check(self, intent: Intent, candidate, ctx: BundleContext) -> CheckResult:
        p = self.policy
        problems = []
        rank, rank_cap = int(intent.rank), int(p.get("rank_cap", 0))
        if not (1 <= rank <= rank_cap):
            problems.append(f"rank {rank} outside [1,{rank_cap}]")
        allowed = set(p.get("allowed_modules", []))
        bad_modules = [m for m in intent.target_modules if m not in allowed]
        if not intent.target_modules or bad_modules:
            problems.append(f"target_modules {intent.target_modules} not subset of {sorted(allowed)}")
        depth = ctx.stack_depth() + 1
        max_depth = int(p.get("max_stack_depth", 0))
        if depth > max_depth:
            problems.append(f"resulting stack depth {depth} exceeds max {max_depth}")
        norm = float(candidate.get("norm", float("inf")))
        norm_cap = float(p.get("norm_cap", 0.0))
        if norm > norm_cap:
            problems.append(f"delta norm {norm:.6f} exceeds cap {norm_cap}")
        return CheckResult(
            "safety_bounds",
            passed=not problems,
            detail="; ".join(problems) if problems else (
                f"rank {rank}<={rank_cap}, modules ok, depth {depth}<={max_depth}, "
                f"norm {norm:.6f}<={norm_cap}"),
            measured=norm,
        )

    def _identity_check(self, intent: Intent, candidate, ctx: BundleContext) -> CheckResult:
        tol = float(self.policy.get("identity_tol", 0.0))
        worst, worst_qid = 0.0, ""
        for qid, x, expected in ctx.identity_probes():
            err = _err(ctx.forward_candidate(x, candidate), expected)
            if err > worst:
                worst, worst_qid = err, qid
        return CheckResult(
            "identity_continuity",
            passed=worst <= tol,
            detail=(f"max probe drift {worst:.6f} <= tol {tol} (worst: {worst_qid})"
                    if worst <= tol else
                    f"IDENTITY DRIFT {worst:.6f} > tol {tol} on probe {worst_qid}"),
            measured=worst,
        )

    def _capability_check(self, intent: Intent, candidate, ctx: BundleContext) -> CheckResult:
        eps = float(self.policy.get("capability_eps", 0.0))
        baseline = ctx.capability_baseline()
        worst, worst_cid = -1e18, ""
        for cid, x, expected in ctx.capability_suite():
            err = _err(ctx.forward_candidate(x, candidate), expected)
            regress = err - float(baseline.get(cid, 0.0))
            if regress > worst:
                worst, worst_cid = regress, cid
        return CheckResult(
            "capability_floor",
            passed=worst <= eps,
            detail=(f"max regression {worst:.6f} <= eps {eps} (worst: {worst_cid})"
                    if worst <= eps else
                    f"CAPABILITY REGRESSION {worst:.6f} > eps {eps} on {worst_cid}"),
            measured=worst,
        )

    def _provenance_check(self, intent: Intent) -> CheckResult:
        problems = []
        if not intent.parents:
            problems.append("parents is empty: an uncaused change is noise")
        if not intent.source_seqs:
            problems.append("source_seqs is empty: the journal can't answer why this exists")
        return CheckResult(
            "provenance_completeness",
            passed=not problems,
            detail=("; ".join(problems) if problems else
                    f"{len(intent.parents)} parent(s), {len(intent.source_seqs)} source seq(s)"),
            measured=float(len(intent.parents) + len(intent.source_seqs)),
        )

    # -- transformer deltas (SI-9b, RFC §4) --------------------------------
    def _transformer_policy(self) -> dict:
        """Resolve the §4.5 policy keys, fail closed (see module docs)."""
        p = self.policy
        d = TRANSFORMER_POLICY_DEFAULTS

        def _cap(key, default):
            if key not in p:
                return default
            v = p[key]
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                return 0  # present but malformed: most restrictive
            return v

        patterns = p.get("allowed_target_patterns",
                         d["allowed_target_patterns"])
        if not (isinstance(patterns, list)
                and all(isinstance(x, str) for x in patterns)):
            patterns = []
        if "per_target_norm_cap" not in p:
            per_target = float(p.get("norm_cap", 0.0))
        else:
            v = p["per_target_norm_cap"]
            per_target = float(v) if isinstance(v, (int, float)) \
                and not isinstance(v, bool) and float(v) >= 0 else 0.0
        return {
            "allowed_target_patterns": list(patterns),
            "max_targets_per_delta": _cap("max_targets_per_delta",
                                          d["max_targets_per_delta"]),
            "delta_byte_cap": _cap("delta_byte_cap", d["delta_byte_cap"]),
            "per_target_norm_cap": per_target,
            "allow_additive": p.get("allow_additive",
                                    d["allow_additive"]) is True,
            "rank_cap": int(p["rank_cap"]) if isinstance(p.get("rank_cap"), int)
                        and not isinstance(p.get("rank_cap"), bool)
                        and p["rank_cap"] >= 0 else 0,
            "norm_cap": float(p["norm_cap"]) if isinstance(p.get("norm_cap"), (int, float))
                        and not isinstance(p.get("norm_cap"), bool) else 0.0,
            "max_stack_depth": int(p["max_stack_depth"])
                        if isinstance(p.get("max_stack_depth"), int)
                        and not isinstance(p.get("max_stack_depth"), bool)
                        and p["max_stack_depth"] >= 0 else 0,
        }

    @staticmethod
    def _ctx_base_manifest(ctx):
        """The bundle's manifest dict from the context, or None.

        Never the candidate's word: only ``ctx.base_manifest()`` or a
        ``ctx.manifest`` dict count (see the module docstring).
        """
        if ctx is None:
            return None
        getter = getattr(ctx, "base_manifest", None)
        if callable(getter):
            try:
                manifest = getter()
            except Exception:
                return None
            return manifest if isinstance(manifest, dict) else None
        manifest = getattr(ctx, "manifest", None)
        return manifest if isinstance(manifest, dict) else None

    def _transformer_safety_check(self, intent: Intent, candidate,
                                  ctx) -> CheckResult:
        """RFC §4.2: the structural suite, as safety_bounds sub-checks.

        Manifest + delta blob only; base tensor bytes are never read.
        SI-9a's validate_transformer_delta runs as a final catch-all
        sub-check, so this suite can only be stricter than the format
        validation, never looser.
        """
        import numpy as np

        from si.transformer_delta import FACTOR_DTYPE_CODE

        pol = self._transformer_policy()
        subs: list[CheckResult] = []

        def sub(name, passed, detail, measured=None):
            subs.append(CheckResult(name, bool(passed), detail, measured))

        record = candidate.get("record") if isinstance(candidate, dict) else None
        blob_bytes = candidate.get("blob_bytes") if isinstance(candidate, dict) else None
        candidate_ok = isinstance(record, dict) \
            and isinstance(blob_bytes, (bytes, bytearray)) \
            and record.get("kind") == TRANSFORMER_DELTA_KIND
        sub("candidate_shape", candidate_ok,
            "candidate carries a transformer-delta-v1 record + blob bytes"
            if candidate_ok else
            "candidate must be {'record': <transformer-delta-v1 record>, "
            "'blob_bytes': <bytes>} with record kind "
            f"{TRANSFORMER_DELTA_KIND!r}")
        if not candidate_ok:
            for name in ("target_existence_identity", "shape_conformance",
                         "dtype_conformance", "finite_values", "norm_budgets",
                         "size_budgets", "allowlist", "additive_policy",
                         "stack_depth", "structural_validation"):
                sub(name, False,
                    "not evaluated: candidate record/blob missing or malformed")
            return self._tf_safety_result(subs, None)
        blob_bytes = bytes(blob_bytes)
        targets = record.get("targets")
        targets = targets if isinstance(targets, list) else []

        # Base manifest: from the bundle context only, fail closed.
        manifest = self._ctx_base_manifest(ctx)
        base_map, base_err = None, None
        if manifest is None:
            base_err = ("bundle context supplies no base manifest "
                        "(base_manifest()/manifest): failing closed — a "
                        "candidate may not validate against a base of its "
                        "own choosing")
        else:
            try:
                base_map = base_tensor_map(manifest)
            except DeltaValidationError as err:
                base_err = f"base manifest tensor map is malformed: {err}"

        parsed, parse_err = None, None
        try:
            parsed = parse_blob(blob_bytes)
        except DeltaValidationError as err:
            parse_err = str(err)

        existence: list[str] = []
        shape_p: list[str] = []
        dtype_p: list[str] = []
        finite_p: list[str] = []
        norm_p: list[str] = []
        size_p: list[str] = []
        allow_p: list[str] = []
        additive_p: list[str] = []
        measured_norms: list[float] = []

        if base_err is not None:
            existence.append(base_err)
        if parse_err is not None:
            shape_p.append(f"delta blob does not parse: {parse_err}")

        for i, t in enumerate(targets):
            label = f"targets[{i}]"
            if not isinstance(t, dict):
                existence.append(f"{label} is not an object")
                continue
            name, kind = t.get("name"), t.get("kind")
            if isinstance(name, str) and name:
                label = f"targets[{i}] ({name})"
            entry = base_map.get(name) if base_map is not None \
                and isinstance(name, str) else None
            # 1. existence & identity (exact name, content-hash binding)
            if base_map is not None and entry is None:
                existence.append(
                    f"{label}: target not in the base manifest's tensor "
                    "map (exact name match required; a demo base records "
                    "no tensor map, so transformer targets are all unknown)")
                continue
            if entry is not None and t.get("base_blob") != entry["blob"]:
                existence.append(
                    f"{label}: base_blob {t.get('base_blob')!r} != the base "
                    f"manifest's hash {entry['blob']!r} for this tensor")
            base_shape = entry["shape"] if entry is not None else None
            # 2. shape conformance (record vs base manifest vs blob header)
            if entry is not None and t.get("base_shape") != base_shape:
                shape_p.append(
                    f"{label}: base_shape {t.get('base_shape')!r} != base "
                    f"manifest shape {base_shape!r}")
            if base_shape is not None and len(base_shape) not in (1, 2):
                shape_p.append(
                    f"{label}: base tensor is {len(base_shape)}-D; only 1-D "
                    "and 2-D base tensors are targetable")
            if kind == "lora" and base_shape is not None and len(base_shape) != 2:
                shape_p.append(f"{label}: lora target must be 2-D")
            if kind == "additive" and base_shape is not None and len(base_shape) != 1:
                shape_p.append(f"{label}: additive target must be 1-D")
            keys = [f"t{i}.A", f"t{i}.B"] if kind == "lora" else [f"t{i}.d"]
            if parsed is not None:
                rank = t.get("rank")
                want = {}
                if kind == "lora" and base_shape is not None \
                        and len(base_shape) == 2 and isinstance(rank, int) \
                        and not isinstance(rank, bool):
                    want = {keys[0]: [rank, base_shape[1]],
                            keys[1]: [base_shape[0], rank]}
                if kind == "additive" and base_shape is not None:
                    want = {keys[0]: list(base_shape)}
                for key, want_shape in want.items():
                    info = parsed.header.get(key)
                    if info is None:
                        shape_p.append(f"{label}: factor {key!r} missing from blob")
                    elif info.get("shape") != want_shape:
                        shape_p.append(
                            f"{label}: factor {key!r} shape {info.get('shape')!r} "
                            f"!= {want_shape!r} (blob header vs record/base)")
                # 3. dtype conformance (factors float32; base dtype match)
                for key in keys:
                    info = parsed.header.get(key)
                    if info is not None and info.get("dtype") != FACTOR_DTYPE_CODE:
                        dtype_p.append(
                            f"{label}: factor {key!r} dtype {info.get('dtype')!r} "
                            f"!= {FACTOR_DTYPE_CODE!r} (factors are stored float32)")
            if entry is not None and t.get("base_dtype") != entry["dtype"]:
                dtype_p.append(
                    f"{label}: base_dtype {t.get('base_dtype')!r} != base "
                    f"manifest dtype {entry['dtype']!r}")
            # factors for the finite + norm sub-checks
            factors = None
            if parsed is not None and all(k in parsed.tensors for k in keys):
                factors = {k: parsed.tensors[k] for k in keys}
            # 4. finite values (full scan of the delta blob's factors)
            if parsed is None:
                finite_p.append(f"{label}: blob unparseable, finiteness unverifiable")
            elif factors is not None:
                flat = np.concatenate(
                    [np.asarray(v, dtype=np.float64).ravel()
                     for v in factors.values()])
                if not np.all(np.isfinite(flat)):
                    finite_p.append(f"{label}: factor values are not all finite")
            # 5. norm budgets (Gram recomputation, float64 — module docs)
            if factors is not None and kind in ("lora", "additive"):
                measured = None
                if kind == "lora":
                    rank, alpha = t.get("rank"), t.get("alpha")
                    if isinstance(rank, int) and not isinstance(rank, bool) \
                            and rank >= 1 and isinstance(alpha, (int, float)) \
                            and not isinstance(alpha, bool):
                        measured = _lora_norm_gram(factors[keys[1]],
                                                   factors[keys[0]],
                                                   float(alpha) / float(rank))
                else:
                    measured = float(np.linalg.norm(
                        np.asarray(factors[keys[0]], dtype=np.float64)))
                if measured is not None:
                    measured_norms.append(measured)
                    recorded = t.get("target_norm")
                    if not isinstance(recorded, (int, float)) \
                            or isinstance(recorded, bool) \
                            or abs(float(recorded) - measured) \
                            > 1e-6 * max(1.0, abs(measured)):
                        norm_p.append(
                            f"{label}: target_norm {recorded!r} disagrees with "
                            f"the Gram-recomputed norm {measured!r} (rtol 1e-6)")
                    if measured > pol["per_target_norm_cap"]:
                        norm_p.append(
                            f"{label}: per-target norm {measured:.6f} exceeds "
                            f"per_target_norm_cap {pol['per_target_norm_cap']}")
            # 6 (rank part). size budgets — rank against the existing cap
            if kind == "lora":
                rank = t.get("rank")
                if isinstance(rank, int) and not isinstance(rank, bool) \
                        and rank > pol["rank_cap"]:
                    size_p.append(
                        f"{label}: rank {rank} exceeds rank_cap {pol['rank_cap']}")
            # 7. allowlist (exact names / fnmatch patterns, no substring)
            if isinstance(name, str) and not any(
                    fnmatch.fnmatchcase(name, pat)
                    for pat in pol["allowed_target_patterns"]):
                allow_p.append(
                    f"{label}: target matches no allowed_target_patterns "
                    f"{pol['allowed_target_patterns']!r} (empty allowlist "
                    "admits nothing)")
            # additive policy gate
            if kind == "additive" and not pol["allow_additive"]:
                additive_p.append(
                    f"{label}: additive targets are not permitted "
                    "(policy allow_additive is false)")

        aggregate = None
        if targets and len(measured_norms) == len(targets):
            aggregate = float(math.sqrt(sum(n * n for n in measured_norms)))
            if aggregate > pol["norm_cap"]:
                norm_p.append(
                    f"aggregate norm {aggregate:.6f} exceeds norm_cap "
                    f"{pol['norm_cap']}")
            recorded_norm = record.get("norm")
            if isinstance(recorded_norm, (int, float)) \
                    and not isinstance(recorded_norm, bool) \
                    and abs(float(recorded_norm) - aggregate) \
                    > 1e-6 * max(1.0, abs(aggregate)):
                norm_p.append(
                    f"record norm {recorded_norm!r} disagrees with the "
                    f"Gram-recomputed aggregate {aggregate!r} (rtol 1e-6)")
        cand_norm = candidate.get("norm")
        if cand_norm is not None:
            recorded_norm = record.get("norm")
            if not isinstance(cand_norm, (int, float)) \
                    or isinstance(cand_norm, bool) \
                    or cand_norm != recorded_norm:
                norm_p.append(
                    f"candidate norm {cand_norm!r} != record norm "
                    f"{recorded_norm!r}")
        # 6. size budgets (bytes + breadth)
        if len(blob_bytes) > pol["delta_byte_cap"]:
            size_p.append(
                f"delta blob bytes {len(blob_bytes)} exceeds delta_byte_cap "
                f"{pol['delta_byte_cap']}")
        if len(targets) > pol["max_targets_per_delta"]:
            size_p.append(
                f"{len(targets)} target(s) exceeds max_targets_per_delta "
                f"{pol['max_targets_per_delta']}")
        if not targets:
            size_p.append("delta declares no targets")

        sub("target_existence_identity", not existence,
            "; ".join(existence) if existence else
            f"all {len(targets)} target(s) exist in the base manifest with "
            "matching content hashes")
        sub("shape_conformance", not shape_p,
            "; ".join(shape_p) if shape_p else
            "factor shapes match the blob header, the record, and the base shapes")
        sub("dtype_conformance", not dtype_p,
            "; ".join(dtype_p) if dtype_p else
            "factor dtypes are float32; base dtypes match the base manifest")
        sub("finite_values", not finite_p,
            "; ".join(finite_p) if finite_p else
            "every factor element in the delta blob is finite (full scan)")
        sub("norm_budgets", not norm_p,
            "; ".join(norm_p) if norm_p else
            (f"per-target norms <= {pol['per_target_norm_cap']}, aggregate "
             f"{aggregate:.6f} <= norm_cap {pol['norm_cap']} (Gram method, "
             "float64)" if aggregate is not None else
             "no measurable targets"), aggregate)
        sub("size_budgets", not size_p,
            "; ".join(size_p) if size_p else
            (f"blob {len(blob_bytes)} bytes <= cap {pol['delta_byte_cap']}, "
             f"{len(targets)} target(s) <= {pol['max_targets_per_delta']}, "
             f"ranks <= rank_cap {pol['rank_cap']}"))
        sub("allowlist", not allow_p,
            "; ".join(allow_p) if allow_p else
            "every target matches allowed_target_patterns")
        sub("additive_policy", not additive_p,
            "; ".join(additive_p) if additive_p else
            ("additive targets permitted by policy" if pol["allow_additive"]
             else "no additive targets (policy allow_additive is false)"))
        # 8. stack rules (unchanged semantics)
        stack_p = []
        depth = None
        try:
            depth = int(ctx.stack_depth()) + 1
        except Exception:
            stack_p.append("bundle context cannot report stack depth: "
                           "failing closed")
        if depth is not None and depth > pol["max_stack_depth"]:
            stack_p.append(
                f"resulting stack depth {depth} exceeds max "
                f"{pol['max_stack_depth']}")
        sub("stack_depth", not stack_p,
            "; ".join(stack_p) if stack_p else
            f"resulting stack depth {depth} <= {pol['max_stack_depth']}",
            float(depth) if depth is not None else None)
        # Catch-all: SI-9a pure validation (can only make this stricter)
        structural_p = []
        if manifest is None:
            structural_p.append("SI-9a validation cannot run: no base manifest")
        else:
            try:
                validate_transformer_delta(record, blob_bytes, manifest)
            except DeltaValidationError as err:
                structural_p.append(str(err))
        sub("structural_validation", not structural_p,
            "; ".join(structural_p) if structural_p else
            "SI-9a validate_transformer_delta passes (record/blob/base "
            "structural validation, incl. metadata-vs-manifest tamper check)")
        return self._tf_safety_result(subs, aggregate)

    @staticmethod
    def _tf_safety_result(subs: list, aggregate) -> CheckResult:
        failed = [s for s in subs if not s.passed]
        if failed:
            detail = "; ".join(f"[{s.name}] {s.detail}" for s in failed)
        else:
            detail = ("transformer structural suite passed: "
                      + "; ".join(f"{s.name}=pass" for s in subs))
        return CheckResult("safety_bounds", not failed, detail,
                           aggregate, subchecks=subs)

    def _transformer_probe_checks(self, intent: Intent, candidate,
                                  ctx) -> tuple:
        """Checks 3/4 for a transformer delta (RFC SI-13 §6,
        SI-16 wiring, SI-17 verdict mapping).

        Default — every context that is not a validated miniature
        probe context: the SI-9b report, unchanged (``passed=False``,
        ``measured=None``, :data:`TRANSFORMER_PROBES_NOT_EVALUATED`).

        Evaluated — only when ``ctx`` carries the probe-context
        marker AND its §6.2 binding + §6.3 startup validation pass:
        check 3 runs the §5.2 suite through this gate's own
        :meth:`_identity_check`; check 4 runs
        :meth:`_capability_check`'s regression logic PLUS §5.3's
        echo accuracy arm (both arms land in the check's detail and
        subchecks, per §6.5). Per §6.5/SI-17 labelling the details
        distinguish exactly: "measured-pass" / "measured-fail"
        (real floats in ``measured``), "not evaluated"
        (``measured=None``), and "blind-escalated"
        (``measured=None``; see below). The caller applies §6.4's
        mapping: measured fail -> REJECT, measured all-pass ->
        ADMIT, anything unevaluated -> ESCALATE.

        Blind-set guard (SI-17): a candidate whose targets are
        exclusively probe-blind tensors (derived in
        ``si.probes.derive_probe_blind_targets``) would measure
        drift 0.0 vacuously — that is NOT a measured pass. It is
        reported "blind-escalated" with ``measured=None`` (so the
        verdict mapping ESCALATEs it) and the reason stated.
        Mixed candidates (blind + measurable targets) are
        evaluated for real; their details state that
        ADMIT-eligibility rests on the measurable targets'
        evidence only, and name the blind targets the probes
        cannot measure.

        Any harness failure at any point — binding mismatch,
        unreproducible baselines, a forward exception — is reported
        "not evaluated" with the specific reason in the detail: it
        is never a pass and never the candidate's failure (§6.3).
        """
        def not_evaluated(detail):
            return (CheckResult("identity_continuity", False, detail, None),
                    CheckResult("capability_floor", False, detail, None))

        if ctx is None or not getattr(ctx, "probe_harness", False):
            return not_evaluated(TRANSFORMER_PROBES_NOT_EVALUATED)
        try:
            ok, reason = ctx.probe_status()
        except Exception as err:  # a broken harness is not a candidate failure
            ok, reason = False, (f"probe harness startup raised "
                                 f"{type(err).__name__}: {err}")
        if not ok:
            return not_evaluated(f"not evaluated: {reason}")
        # SI-17 blind-set guard: exclusively-blind targets measure
        # drift 0.0 vacuously — never a measured pass.
        measurability = None
        blind_note = ""
        try:
            measurability = ctx.candidate_measurability(candidate)
        except Exception:
            measurability = None  # guard failure never fakes a pass;
            # the evaluation below still runs and decides on evidence
        if measurability is not None and measurability["all_blind"]:
            detail = (
                "blind-escalated: target(s) not measurable by the "
                "probe suites: "
                f"{measurability['blind']!r} are probe-blind "
                "(derived set, si.probes.derive_probe_blind_targets: "
                "a constant key bias shifts all of a query's "
                "attention scores equally and softmax is "
                "shift-invariant, so drift would measure 0.0 "
                "vacuously); a vacuous 0.0 is NOT a measured pass, "
                "so this delta is not ADMIT-eligible and escalates "
                "to the operator (RFC SI-13 §6.4/SI-17 guard)")
            return not_evaluated(detail)
        if measurability is not None and measurability["blind"]:
            blind_note = (
                "; note: target(s) "
                f"{measurability['blind']!r} are probe-blind and "
                "not measurable by the probe suites; "
                "ADMIT-eligibility rests on the measurable "
                f"target(s) {measurability['measurable']!r} only")
        evaluated = ("probe suite "
                     f"{getattr(ctx, 'probe_artifact_hash', None)} "
                     "evaluated (RFC SI-13 §6.4/§6.5, SI-17: verdict "
                     "mapping live for miniature bundles)")
        try:
            identity = self._identity_check(intent, candidate, ctx)
            identity = CheckResult(
                identity.name, identity.passed,
                f"{'measured-pass' if identity.passed else 'measured-fail'}: "
                f"{identity.detail}; {evaluated}{blind_note}",
                identity.measured)
            regression = self._capability_check(intent, candidate, ctx)
            # §5.3 accuracy arm: the echo task's functional floor.
            targets = ctx.capability_targets()
            floor = float(ctx.echo_floor())
            suite = ctx.capability_suite()
            correct = 0
            for cid, x, _expected in suite:
                import numpy as np
                logits = ctx.forward_candidate(x, candidate)
                if int(np.argmax(np.asarray(logits))) == int(targets[cid]):
                    correct += 1
            accuracy = correct / len(suite) if suite else 0.0
            accuracy_passed = accuracy >= floor
            cap_passed = bool(regression.passed and accuracy_passed)
            capability = CheckResult(
                "capability_floor",
                cap_passed,
                f"{'measured-pass' if cap_passed else 'measured-fail'}: "
                f"regression arm: {regression.detail}; accuracy arm: "
                f"echo accuracy {accuracy:.4f} ({correct}/{len(suite)}) "
                f"vs floor {floor:.4f} -> "
                f"{'pass' if accuracy_passed else 'FAIL'}; "
                f"{evaluated}{blind_note}",
                regression.measured,
                subchecks=[
                    CheckResult("capability_regression",
                                regression.passed, regression.detail,
                                regression.measured),
                    CheckResult("echo_accuracy", accuracy_passed,
                                f"echo accuracy {accuracy:.4f} "
                                f"({correct}/{len(suite)}) vs floor "
                                f"{floor:.4f} (base accuracy minus "
                                "one entry, RFC SI-13 §5.3)",
                                accuracy),
                ])
            return identity, capability
        except Exception as err:  # measurement failed, not the candidate
            return not_evaluated(
                f"not evaluated: probe harness raised during "
                f"evaluation ({type(err).__name__}: {err}); the "
                "candidate did not fail — the measurement failed "
                "(RFC SI-13 §6.3)")

    def _transformer_provenance_check(self, intent: Intent,
                                      candidate) -> CheckResult:
        """RFC §4.4: demo provenance + the declared target list."""
        problems = []
        if not intent.parents:
            problems.append("parents is empty: an uncaused change is noise")
        if not intent.source_seqs:
            problems.append("source_seqs is empty: the journal can't answer "
                            "why this exists")
        declared = intent.targets
        record = candidate.get("record") if isinstance(candidate, dict) else None
        record_names = None
        if isinstance(record, dict) and isinstance(record.get("targets"), list):
            record_names = [t.get("name") if isinstance(t, dict) else None
                            for t in record["targets"]]
        if not declared or not all(isinstance(x, str) and x for x in declared):
            problems.append(
                "intent does not declare its target list (Intent.targets "
                "must name every base tensor the delta modifies, RFC §4.4)")
        elif record_names is None:
            problems.append("candidate record unavailable: declared targets "
                            "cannot be checked against the record")
        elif list(declared) != record_names:
            problems.append(
                f"declared targets {list(declared)!r} do not match the "
                f"candidate record's targets {record_names!r}")
        return CheckResult(
            "provenance_completeness",
            passed=not problems,
            detail=("; ".join(problems) if problems else
                    f"{len(intent.parents)} parent(s), "
                    f"{len(intent.source_seqs)} source seq(s), declared "
                    f"targets match the record: {list(declared)!r}"),
            measured=float(len(intent.parents) + len(intent.source_seqs)),
        )

    def _needs_escalation(self, intent: Intent, candidate) -> bool:
        """Big-but-legal changes go to the operator, not the mind alone."""
        norm = float(candidate.get("norm", 0.0))
        return norm > float(self.policy.get("escalation_norm", float("inf")))
