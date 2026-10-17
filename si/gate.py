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
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol


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
        }


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    measured: float | None = None

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed,
                "detail": self.detail, "measured": self.measured}


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


def _err(y, expected) -> float:
    import numpy as np
    return float(np.max(np.abs(np.asarray(y, dtype=np.float64)
                               - np.asarray(expected, dtype=np.float64))))


class Gate:
    def __init__(self, policy: dict):
        self.policy = dict(policy)

    # -- public ------------------------------------------------------------
    def evaluate(self, intent: Intent, candidate, ctx: BundleContext) -> GateReport:
        checks: list[CheckResult] = []

        # Malformed intents never reach the checks.
        if intent.kind not in ("delta", "memory", "rollback", "constitution"):
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

        # kind == "delta": the full five checks.
        if candidate is None:
            return GateReport("REJECT", checks + [CheckResult(
                "wellformed", False, "delta intent arrived with no candidate", None)])
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

    def _needs_escalation(self, intent: Intent, candidate) -> bool:
        """Big-but-legal changes go to the operator, not the mind alone."""
        norm = float(candidate.get("norm", 0.0))
        return norm > float(self.policy.get("escalation_norm", float("inf")))
