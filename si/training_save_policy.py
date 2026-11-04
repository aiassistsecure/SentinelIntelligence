"""SI-89: SAVE_EVERY policy gate (SI-83 section 6.4 point 3).

Implements the one section 6.4 point left unbuilt by SI-88:
"SAVE_EVERY below 500 on a full fine-tune requires its own
gated proposal naming the disk arithmetic for the new
interval -- the v12 lesson, enforced by policy rather than
memory." The flow follows the established propose -> gate ->
journal pattern of :mod:`si.training_retention` and
:mod:`si.training_fork_prune` (verdicts ADMIT / REJECT /
ESCALATE, SI-0 gate shape), journaled through the section 2
journal written by :mod:`si.training_journal` (SI-85) via
its SI-89 ``save_every_decision`` kind -- an SI-89 extension
of the section 2.5 kind list (section 6.4 states the policy
but section 2.5 names no kind for it; this module never
claims section 2.5 listed it).

Gate semantics (:func:`gate_save_every`, read-only, journals
nothing; :func:`propose_save_every` is the full flow):

- Invalid input -- ``save_every <= 0``, ``total_steps <= 0``,
  or an unknown layout -- is REJECTED.
- No reason -> refused outright: a REJECT outcome is
  returned and NOTHING is journaled (the same rule as the
  retention and prune no-reason refusals).
- Not a full fine-tune -> ADMIT, with the verdict reason
  stating section 6.4 point 3 applies only to full
  fine-tunes (honest scoping, not an exception: a LoRA /
  adapter run's checkpoints do not carry full optimizer
  state in the section 6.2 sense).
- ``save_every >= 500`` on a full fine-tune -> ADMIT
  (standing practice, section 6.1; no exception needed).
- ``save_every < 500`` on a full fine-tune -> ESCALATE,
  ALWAYS. The gate never self-admits an exception to the
  safety policy; an operator decides. The journaled row and
  the returned result both name the disk arithmetic (below)
  so the operator decides on numbers, not vibes.

Journal before return: on every verdict other than the
no-reason refusal, the ``save_every_decision`` row is
journaled BEFORE the result returns. A caller
``on_before_return`` hook runs between the journal write
and the return so tests can prove the ordering (the
retention slice's ``on_before_execute`` pattern, adapted:
this slice executes nothing, so the hook guards the
return).

Disk arithmetic (:func:`save_every_figures`): recomputed
in code from the section 6.2 assumptions, reusing the
retention module's stated constants (Imagine: 1.6B
parameters; bf16 2 B/param; fp32 4 B/param; AdamW moments
8 B/param -- two fp32 moment buffers). No total is a
hardcoded magic number: per-checkpoint Layout A (bf16
weights + bf16 grads + fp32 moments) and Layout B (fp32
master + fp32 grads + fp32 moments) sizes, all-full-
retention totals for both layouts, and the section 6.4
proposed-retention projection (weights-only bf16 for every
checkpoint + the optimizer add-on on the newest 2 only)
are all derived from those constants. Every figure carries
the literal label ``"ESTIMATE"``; nothing here is measured.
Anchors (section 6.3, test-asserted): 7,311 steps at
SAVE_EVERY 500 -> 15 checkpoints, Layout A total 288 GB,
proposed projection 80 GB; at 250 -> 30 checkpoints,
Layout A total 576 GB, proposed projection 128 GB.

What is NOT built in this slice, stated plainly:

- No trainer is launched, no training config is written or
  modified, and no file is deleted. This slice is a policy
  gate + journal only; nothing enforces the verdict in any
  trainer.
- No operator-approval execution path is built: ESCALATE
  is terminal for this slice. An operator's later decision
  is not recorded or acted on by this code.
- No live NEDB write is performed or claimed; the tests
  exercise this module through the real writer on the
  embedded backend only. No Imagine run had a SAVE_EVERY
  change proposed or applied by this code.

This is machine learning infrastructure. CPU only,
standard library only.

Honesty labels, unchanged: section 18.3 NOT earned; P5-R1 /
P5-R9 UNMEASURED; live model-authored intent OCCURRED scoped
to SI-74 only; L-3 LOCKED; A-2 / A-3 / A-4 UNAPPROVED;
unlocks []; D-F1 OPEN; F-line STOPPED.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

from si.training_journal import TrainingJournalWriter
from si.training_retention import (
    ADAMW_MOMENT_BYTES_PER_PARAM, BF16_BYTES_PER_PARAM,
    FP32_BYTES_PER_PARAM, NEWEST_KEPT_OPTIMIZER_COUNT,
    PARAM_COUNT,
)

__all__ = [
    "LAYOUT_A", "LAYOUT_B", "KNOWN_LAYOUTS",
    "MIN_SAVE_EVERY_FULL_FINE_TUNE",
    "VERDICT_ADMIT", "VERDICT_REJECT", "VERDICT_ESCALATE",
    "PARAM_COUNT", "BF16_BYTES_PER_PARAM",
    "FP32_BYTES_PER_PARAM", "ADAMW_MOMENT_BYTES_PER_PARAM",
    "SavePolicyError", "GateDecision", "SaveEveryResult",
    "save_every_figures", "gate_save_every", "propose_save_every",
]

#: Gate verdicts (SI-83 section 6.4, SI-0 gate shape).
VERDICT_ADMIT = "ADMIT"
VERDICT_REJECT = "REJECT"
VERDICT_ESCALATE = "ESCALATE"

#: Standing practice (SI-83 section 6.1): SAVE_EVERY at or
#: above this interval on a full fine-tune needs no exception.
MIN_SAVE_EVERY_FULL_FINE_TUNE = 500

#: The two checkpoint layouts section 6.2 names.
LAYOUT_A = "A"
LAYOUT_B = "B"
KNOWN_LAYOUTS = (LAYOUT_A, LAYOUT_B)

_LAYOUT_ALIASES = {
    "a": LAYOUT_A, "layout_a": LAYOUT_A, "layout a": LAYOUT_A,
    "b": LAYOUT_B, "layout_b": LAYOUT_B, "layout b": LAYOUT_B,
}

_BYTES_PER_GB = 1_000_000_000  # decimal GB (SI-83 section 6.2)


class SavePolicyError(RuntimeError):
    """Base for SAVE_EVERY policy-flow refusals (fail closed)."""


@dataclass(frozen=True)
class GateDecision:
    """The SAVE_EVERY gate's verdict, reasons, and figures."""
    verdict: str
    reasons: tuple = ()
    figures: Optional[dict] = None


@dataclass
class SaveEveryResult:
    """Outcome of one propose -> gate -> journal run."""
    verdict: str
    gate_reasons: tuple = ()
    run_id: str = ""
    save_every: object = None
    full_fine_tune: bool = False
    total_steps: object = None
    layout: Optional[str] = None
    figures: Optional[dict] = None
    decision_seq: Optional[int] = None


def _normalize_layout(layout) -> Optional[str]:
    if layout is None:
        return LAYOUT_A  # documented default: section 6.3's layout
    return _LAYOUT_ALIASES.get(str(layout).strip().lower())


def _as_int(value):
    """An int for genuine integer inputs, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _gb(byte_count) -> float:
    return round(byte_count / _BYTES_PER_GB, 6)


# -- disk arithmetic (SI-83 sections 6.2 / 6.3 / 6.4) -------------------------


def save_every_figures(save_every, total_steps, layout=None) -> dict:
    """ESTIMATE disk figures for a proposed SAVE_EVERY.

    Recomputed from the section 6.2 assumptions in code
    (parameter count times byte widths, reusing the
    retention module's constants). ``layout`` selects which
    layout the proposed run would write; both layouts'
    per-checkpoint sizes and all-full-retention totals are
    always reported, plus the section 6.4 proposed-
    retention projection. Raises ValueError on invalid
    input (the gate turns that into a REJECT)."""
    save_every = _as_int(save_every)
    total_steps = _as_int(total_steps)
    if save_every is None or save_every <= 0:
        raise ValueError("save_every must be a positive integer")
    if total_steps is None or total_steps <= 0:
        raise ValueError("total_steps must be a positive integer")
    normalized = _normalize_layout(layout)
    if normalized is None:
        raise ValueError(f"unknown layout {layout!r}: the only "
                         f"layouts are {KNOWN_LAYOUTS}")
    count = math.ceil(total_steps / save_every)
    weights_bf16 = _gb(PARAM_COUNT * BF16_BYTES_PER_PARAM)
    weights_fp32 = _gb(PARAM_COUNT * FP32_BYTES_PER_PARAM)
    grads_bf16 = _gb(PARAM_COUNT * BF16_BYTES_PER_PARAM)
    grads_fp32 = _gb(PARAM_COUNT * FP32_BYTES_PER_PARAM)
    moments = _gb(PARAM_COUNT * ADAMW_MOMENT_BYTES_PER_PARAM)
    layout_a = round(weights_bf16 + grads_bf16 + moments, 6)
    layout_b = round(weights_fp32 + grads_fp32 + moments, 6)
    weights_only = weights_bf16
    optimizer_addon = round(moments + grads_bf16, 6)
    kept = min(NEWEST_KEPT_OPTIMIZER_COUNT, count)
    proposed_total = round(
        weights_only * count + optimizer_addon * kept, 6)
    selected = layout_a if normalized == LAYOUT_A else layout_b
    return {
        "label": "ESTIMATE",
        "param_count": PARAM_COUNT,
        "save_every": save_every,
        "total_steps": total_steps,
        "layout": normalized,
        "checkpoint_count": count,
        "per_checkpoint_gb": {
            "weights_bf16": weights_bf16,
            "weights_fp32": weights_fp32,
            "gradients_bf16": grads_bf16,
            "gradients_fp32": grads_fp32,
            "adamw_moments": moments,
            "layout_a_full": layout_a,
            "layout_b_full": layout_b,
            "weights_only_bf16": weights_only,
            "optimizer_addon": optimizer_addon,
        },
        "full_retention_gb": {
            "layout_a_total": round(layout_a * count, 6),
            "layout_b_total": round(layout_b * count, 6),
            "weights_only_total": round(weights_only * count, 6),
            "selected_layout": normalized,
            "selected_layout_total": round(selected * count, 6),
        },
        "proposed_retention_gb": {
            "weights_only_total": round(weights_only * count, 6),
            "optimizer_addon_total": round(
                optimizer_addon * kept, 6),
            "newest_kept_optimizer_count": kept,
            "total": proposed_total,
        },
    }


def _figures_text(figures: dict) -> str:
    """The arithmetic, named in one sentence for an operator."""
    per = figures["per_checkpoint_gb"]
    full = figures["full_retention_gb"]
    return (
        f"disk arithmetic (ESTIMATE, SI-83 section 6.2): "
        f"{figures['total_steps']} steps at SAVE_EVERY "
        f"{figures['save_every']} = {figures['checkpoint_count']} "
        f"checkpoints; per checkpoint Layout A "
        f"{per['layout_a_full']} GB / Layout B "
        f"{per['layout_b_full']} GB; all-full retention Layout A "
        f"{full['layout_a_total']} GB / Layout B "
        f"{full['layout_b_total']} GB; proposed retention "
        f"(weights-only for all + optimizer add-on on the "
        f"newest {figures['proposed_retention_gb']['newest_kept_optimizer_count']}) "
        f"{figures['proposed_retention_gb']['total']} GB")


# -- gate (SI-83 section 6.4 point 3) ------------------------------------------


def gate_save_every(*, save_every, total_steps, full_fine_tune,
                    layout=None) -> GateDecision:
    """Run the section 6.4 point 3 gate checks (read-only;
    journals nothing) and return the verdict with reasons
    and, when the input is valid, the ESTIMATE figures."""
    problems = []
    se = _as_int(save_every)
    steps = _as_int(total_steps)
    if se is None or se <= 0:
        problems.append(
            f"save_every {save_every!r} is invalid: it must be "
            "a positive integer (SI-83 section 6.4 point 3)")
    if steps is None or steps <= 0:
        problems.append(
            f"total_steps {total_steps!r} is invalid: it must "
            "be a positive integer (SI-83 section 6.4 point 3)")
    normalized = _normalize_layout(layout)
    if layout is not None and normalized is None:
        problems.append(
            f"unknown layout {layout!r}: the only layouts "
            f"section 6.2 names are {KNOWN_LAYOUTS}")
    if problems:
        return GateDecision(VERDICT_REJECT, tuple(problems), None)
    figures = save_every_figures(se, steps, normalized)
    if not full_fine_tune:
        return GateDecision(
            VERDICT_ADMIT,
            ("SI-83 section 6.4 point 3 applies only to full "
             "fine-tunes; this proposal is not a full fine-tune, "
             "so the SAVE_EVERY >= "
             f"{MIN_SAVE_EVERY_FULL_FINE_TUNE} policy does not "
             "bind it (honest scoping, not an exception). "
             + _figures_text(figures),),
            figures)
    if se >= MIN_SAVE_EVERY_FULL_FINE_TUNE:
        return GateDecision(
            VERDICT_ADMIT,
            (f"SAVE_EVERY {se} on a full fine-tune meets the "
             f"standing practice of section 6.1 (SAVE_EVERY "
             f"{MIN_SAVE_EVERY_FULL_FINE_TUNE} or more); no "
             "exception is needed. " + _figures_text(figures),),
            figures)
    return GateDecision(
        VERDICT_ESCALATE,
        (f"SAVE_EVERY {se} is below "
         f"{MIN_SAVE_EVERY_FULL_FINE_TUNE} on a full fine-tune: "
         "section 6.4 point 3 requires its own gated proposal "
         "and an operator decision -- this gate never "
         "self-admits an exception to the safety policy. "
         + _figures_text(figures),),
        figures)


# -- flow: propose -> gate -> journal ------------------------------------------


def propose_save_every(writer: TrainingJournalWriter, *, run_id,
                       save_every, full_fine_tune, total_steps,
                       layout=None, reason=None, operator=None,
                       on_before_return: Optional[Callable] = None) \
        -> SaveEveryResult:
    """The full section 6.4 point 3 flow for one proposal.

    No reason -> REJECT returned, nothing journaled. Every
    other verdict is journaled as a ``save_every_decision``
    row (SI-89 kind) BEFORE the result returns; the optional
    ``on_before_return`` hook runs between the journal write
    and the return. Nothing is executed: no trainer is
    launched and no config is written or modified."""
    if not reason or not str(reason).strip():
        return SaveEveryResult(
            verdict=VERDICT_REJECT,
            gate_reasons=("a SAVE_EVERY proposal without a "
                          "reason is refused outright (SI-83 "
                          "section 6.4 point 3: the proposal "
                          "must name its disk arithmetic and "
                          "its reason; a reasonless proposal "
                          "is neither)",),
            run_id=str(run_id), save_every=save_every,
            full_fine_tune=bool(full_fine_tune),
            total_steps=total_steps,
            layout=_normalize_layout(layout))
    decision = gate_save_every(
        save_every=save_every, total_steps=total_steps,
        full_fine_tune=full_fine_tune, layout=layout)
    normalized = _normalize_layout(layout)
    row = writer.record_save_every_decision(
        run_id, save_every=_as_int(save_every),
        proposed_save_every=_as_int(save_every),
        full_fine_tune=bool(full_fine_tune),
        total_steps=_as_int(total_steps), layout=normalized,
        gate_verdict=decision.verdict, operator=operator,
        reason=str(reason), figures=decision.figures,
        gate_reasons=list(decision.reasons))
    result = SaveEveryResult(
        verdict=decision.verdict, gate_reasons=decision.reasons,
        run_id=str(run_id), save_every=_as_int(save_every),
        full_fine_tune=bool(full_fine_tune),
        total_steps=_as_int(total_steps), layout=normalized,
        figures=decision.figures, decision_seq=row["seq"])
    if on_before_return is not None:
        on_before_return({
            "relation": "save_every_decision",
            "decision_seq": row["seq"], "run_id": str(run_id),
            "save_every": result.save_every,
            "verdict": decision.verdict,
            "figures": decision.figures,
        })
    return result
