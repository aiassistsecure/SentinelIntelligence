"""SI-88: retention enforcement (SI-83 section 9, slice 5).

Implements SI-83 section 6.4 retention as gated
``retention_decision`` proposals, on top of the section 2
journal written by :mod:`si.training_journal` (SI-85).
Checkpoint lookup reuses slice 3's
:func:`si.training_rollback.find_manifest`, and the
optimizer file is resolved and hash-verified IN PLACE via
slice 3's :func:`si.training_rollback.verify_checkpoint`
(stream-hash, never a copy: SI never duplicates bytes).
The flow follows slice 4's pattern
(:mod:`si.training_fork_prune`): propose -> gate ->
journal -> execute, with journal-before-execution proven
by a caller hook.

The retention policy enforced here is the section 6.4
PROPOSED shape, labelled PROPOSED wherever it is stated --
a policy shape proposed by the spec, not a behaviour any
earlier code had and not a ratified limit:

1. Weights are retained for every SAVE_EVERY checkpoint.
   Deleting weights deletes the rollback/fork target.
2. Optimizer state is retained for the newest 2
   checkpoints of a run and for any checkpoint named as a
   fork parent or rollback target by an admitted proposal.
   Dropping any OTHER checkpoint's optimizer state is a
   ``retention_decision``: proposed, gated, journaled,
   with the checkpoint's new ``optimizer_blob = null``
   state appended as a NEW ``checkpoint_manifest``
   revision -- never an edit of history (slice 3's
   ``find_manifest`` returns the latest revision, so
   post-drop readers see ``optimizer_blob`` null). The
   new revision also carries ``optimizer_bytes = null``
   (this slice's documented choice: null, matching the
   journal's absent-optimizer convention, not 0).

What this slice builds, and what it does NOT build:

- :func:`execute_retention` is the full flow for one
  proposal. The proposal names a run, a checkpoint step,
  an action (``"drop_optimizer"`` or
  ``"delete_weights"``), and a reason. No reason ->
  refused outright: a REJECT outcome is returned, nothing
  is journaled, nothing is deleted.
- The gate (:func:`gate_retention`, verdicts ADMIT /
  REJECT / ESCALATE in the SI-0 gate's shape) checks, for
  ``drop_optimizer``:
  (a) the manifest (latest revision) exists for
      (run, step), is ``checkpoint_complete``, and carries
      a non-null ``optimizer_blob`` -- otherwise REJECT
      (nothing to drop / not a retention target);
  (b) PROTECTION -- the checkpoint is one of the run's
      newest 2 checkpoints by step (latest revision per
      step) -> REJECT;
  (c) PROTECTION -- the checkpoint is named as a fork
      parent (a ``training_runs`` row's ``parent_step``
      for this run as parent, or an admitted ``fork``
      proposal row) or as a rollback target (an admitted
      ``rollback`` row's ``target_step``) -> REJECT.
  Hash verification is NOT a gate verdict: it is step (d)
  of the flow (see below).
- Journal before execution: on ADMIT the
  ``retention_decision`` row (verdict ADMIT, keep=False,
  step, reason, operator) is journaled BEFORE the
  optimizer file is deleted. A caller ``on_before_execute``
  hook runs between the journal write and the deletion so
  tests can prove the ordering (slice 4's
  ``on_before_land`` pattern). On REJECT / ESCALATE the
  ``retention_decision`` row is journaled with that
  verdict (when the proposal is well-formed) and NOTHING
  is deleted.
- Verification (flow step (d)): before any deletion is
  contemplated, the checkpoint must resolve and
  hash-verify in place via slice 3's
  ``verify_checkpoint``. A missing optimizer file or a
  hash mismatch fails closed with the slice-3 typed error
  (``OptimizerFileMissing`` / ``OptimizerHashMismatch`` /
  ``WeightsHashMismatch`` / ...): nothing is deleted and
  no ADMIT is journaled.
- Execute (ADMIT only): ONLY the optimizer file is
  deleted. The weights file is never touched -- it is
  hash-checked before and after and the flow raises if
  it changed. After deletion, the NEW checkpoint_manifest
  revision described above is appended. The result
  carries the retention_decision seq, the new manifest
  revision seq, the deleted file path, and the measured
  bytes freed (the optimizer file's actual size,
  MEASURED from the filesystem -- not an estimate).
- Post-drop honesty (section 4.2 consequence of section
  6.4): after a drop, slice 3's ``verify_checkpoint`` for
  that step reports resume kind
  ``weights-exact/optimizer-fresh``, never ``exact``.
  That is the truth of a weights-only checkpoint and is
  not papered over anywhere in this module.

Weights deletion -- operator-only, never executed here:

- A proposal with action ``"delete_weights"`` is NEVER
  admitted or executed by this slice: the gate returns
  ESCALATE with a reason stating weights deletion is
  operator-only (section 6.4 point 1 / section 9 slice
  5), the ESCALATE verdict is journaled, and no file is
  deleted. There is no code path in this module that
  deletes a weights file.

Disk figures (section 9 slice 5 gate criterion + section
6.4 point 4):

- :func:`retention_figures` reports, per run: the
  checkpoint count and how many retain optimizer state
  (latest revision per step); the MEASURED bytes summed
  from the manifests' ``weights_bytes`` /
  ``optimizer_bytes`` fields where present (labelled
  ``"MEASURED"``); and the section 6.2 ESTIMATE for a
  full run at that checkpoint count, recomputed in code
  from the stated assumptions (Imagine: 1.6B parameters,
  full fine tune; bf16 = 2 B/param, fp32 = 4 B/param,
  AdamW moments = 8 B/param) -- never hardcoded totals
  without the arithmetic. Every ESTIMATE figure carries
  the literal label ``"ESTIMATE"``; a figure derived
  from manifest byte fields carries ``"MEASURED"``. An
  estimate is never presented as measured.

NOT built in this slice, stated plainly as gaps:

- Section 6.4 point 3 -- gated proposals for SAVE_EVERY
  below 500 -- is NOT built here. Nothing in this module
  proposes, gates, or journals a SAVE_EVERY change.
- Weights deletion execution is NOT built (operator-only;
  see above).

Miniature only: this slice is exercised against tiny
checkpoint files in temporary directories, journaled
through the real writer on the embedded backend. No live
NEDB write is performed or claimed, no Imagine run had
retention enforced on it by this code, and no trainer is
launched. This module deletes exactly one thing: an
admitted proposal's optimizer file. This is machine
learning infrastructure. CPU only, standard library only.

Honesty labels, unchanged: section 18.3 NOT earned; P5-R1 /
P5-R9 UNMEASURED; live model-authored intent OCCURRED scoped
to SI-74 only; L-3 LOCKED; A-2 / A-3 / A-4 UNAPPROVED;
unlocks []; D-F1 OPEN; F-line STOPPED.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

from si.training_journal import TrainingJournalWriter
from si.training_rollback import (
    CheckpointNotFound, find_manifest, verify_checkpoint,
)
from si.training_watcher import sha256_file_streaming

__all__ = [
    "ACTION_DROP_OPTIMIZER", "ACTION_DELETE_WEIGHTS",
    "VERDICT_ADMIT", "VERDICT_REJECT", "VERDICT_ESCALATE",
    "NEWEST_KEPT_OPTIMIZER_COUNT",
    "PARAM_COUNT", "BF16_BYTES_PER_PARAM", "FP32_BYTES_PER_PARAM",
    "ADAMW_MOMENT_BYTES_PER_PARAM",
    "RetentionError", "WeightsChangedDuringRetention",
    "GateDecision", "RetentionResult",
    "gate_retention", "execute_retention", "retention_figures",
]

#: The two actions a retention proposal may name.
ACTION_DROP_OPTIMIZER = "drop_optimizer"
ACTION_DELETE_WEIGHTS = "delete_weights"

#: Gate verdicts (SI-83 section 6.4, SI-0 gate shape).
VERDICT_ADMIT = "ADMIT"
VERDICT_REJECT = "REJECT"
VERDICT_ESCALATE = "ESCALATE"

#: PROPOSED policy (SI-83 section 6.4 point 2): optimizer
#: state is retained for this many newest checkpoints of a
#: run. The spec's PROPOSED shape fixes this number at 2;
#: it is labelled PROPOSED here and in gate reasons.
NEWEST_KEPT_OPTIMIZER_COUNT = 2

#: SI-83 section 6.2 assumptions, stated so the arithmetic
#: in retention_figures is checkable. Imagine: 1.6B
#: parameters, full fine tune (no adapter).
PARAM_COUNT = 1_600_000_000
BF16_BYTES_PER_PARAM = 2
FP32_BYTES_PER_PARAM = 4
#: AdamW-class: two fp32 moment buffers (m and v) per param.
ADAMW_MOMENT_BYTES_PER_PARAM = 2 * FP32_BYTES_PER_PARAM

_CHECKPOINTS = "training_checkpoints"
_PROPOSALS = "training_proposals"
_RUNS = "training_runs"

_BYTES_PER_GB = 1_000_000_000  # decimal GB (SI-83 section 6.2)


class RetentionError(RuntimeError):
    """Base for retention-flow refusals (fail closed)."""


class WeightsChangedDuringRetention(RetentionError):
    """The weights file's hash changed across an optimizer
    drop; a retention action must never touch weights."""


@dataclass(frozen=True)
class GateDecision:
    """The retention gate's verdict and its stated reasons."""
    verdict: str
    reasons: tuple = ()


@dataclass
class RetentionResult:
    """Outcome of one propose -> gate -> journal -> execute run."""
    verdict: str
    gate_reasons: tuple = ()
    run_id: str = ""
    step: int = 0
    action: str = ""
    retention_seq: Optional[int] = None
    new_manifest_seq: Optional[int] = None
    deleted_path: Optional[str] = None
    bytes_freed: Optional[int] = None  # MEASURED file size, ADMIT only


# -- journal reads ---------------------------------------------------------------


def _latest_manifests(writer: TrainingJournalWriter, run_id) -> dict:
    """Latest checkpoint_manifest revision per step, for a run."""
    latest = {}
    for row in writer.rows(_CHECKPOINTS):
        if row.get("kind") == "checkpoint_manifest" \
                and row.get("run_id") == str(run_id):
            latest[int(row["step"])] = row  # rows are seq-ordered
    return latest


def _is_fork_parent(writer: TrainingJournalWriter, run_id, step) -> bool:
    """True when (run_id, step) is named as a fork parent:
    a training_runs fork row's parent edge, or an admitted
    fork proposal row (SI-83 section 6.4 point 2)."""
    for row in writer.rows(_RUNS):
        if row.get("parent_run_id") == str(run_id) \
                and row.get("parent_step") == int(step):
            return True
    for row in writer.rows(_PROPOSALS):
        if row.get("kind") == "fork" \
                and row.get("run_id") == str(run_id) \
                and row.get("parent_step") == int(step) \
                and row.get("gate_verdict") == VERDICT_ADMIT:
            return True
    return False


def _is_rollback_target(writer: TrainingJournalWriter, run_id,
                        step) -> bool:
    """True when an ADMITTED rollback row names this step
    as its target (SI-83 section 6.4 point 2)."""
    for row in writer.rows(_PROPOSALS):
        if row.get("kind") == "rollback" \
                and row.get("run_id") == str(run_id) \
                and row.get("target_step") == int(step) \
                and row.get("gate_verdict") == VERDICT_ADMIT:
            return True
    return False


# -- gate (SI-83 section 6.4) ------------------------------------------------------


def gate_retention(writer: TrainingJournalWriter, *, run_id, step,
                   action) -> GateDecision:
    """Run the section 6.4 gate checks (read-only; journals
    nothing) and return the verdict with its reasons.

    Hash verification is deliberately NOT part of the gate:
    it is flow step (d) in :func:`execute_retention` and
    fails closed with slice 3's typed errors."""
    run_id = str(run_id)
    step = int(step)
    if action == ACTION_DELETE_WEIGHTS:
        return GateDecision(
            VERDICT_ESCALATE,
            ("weights deletion is operator-only and is never "
             "admitted or executed by this slice (SI-83 "
             "section 6.4 point 1 / section 9 slice 5): "
             "deleting weights deletes the rollback/fork "
             "target",))
    if action != ACTION_DROP_OPTIMIZER:
        return GateDecision(
            VERDICT_REJECT,
            (f"unknown retention action {action!r}: the only "
             f"actions are {ACTION_DROP_OPTIMIZER!r} and "
             f"{ACTION_DELETE_WEIGHTS!r}",))
    # (a) manifest exists, complete, carries optimizer state.
    try:
        manifest = find_manifest(writer, run_id, step)
    except CheckpointNotFound:
        return GateDecision(
            VERDICT_REJECT,
            (f"no checkpoint_manifest for run {run_id!r} step "
             f"{step}: nothing to drop, not a retention "
             "target (SI-83 6.4 check (a))",))
    if manifest.get("checkpoint_complete") is not True:
        return GateDecision(
            VERDICT_REJECT,
            (f"checkpoint run {run_id!r} step {step} is "
             "checkpoint_complete=false (partial save): not "
             "a retention target (SI-83 6.4 check (a))",))
    if not manifest.get("optimizer_blob"):
        return GateDecision(
            VERDICT_REJECT,
            (f"checkpoint run {run_id!r} step {step} carries "
             "no optimizer state (optimizer_blob is null -- "
             "already dropped or weights-only): nothing to "
             "drop (SI-83 6.4 check (a))",))
    reasons = []
    # (b) PROTECTION: newest 2 checkpoints of the run.
    latest = _latest_manifests(writer, run_id)
    newest = sorted(latest, reverse=True)[
        :NEWEST_KEPT_OPTIMIZER_COUNT]
    if step in newest:
        reasons.append(
            f"checkpoint step {step} is one of the run's "
            f"newest {NEWEST_KEPT_OPTIMIZER_COUNT} "
            f"checkpoints {newest}: optimizer state is "
            "retained for the newest "
            f"{NEWEST_KEPT_OPTIMIZER_COUNT} under the "
            "PROPOSED section 6.4 policy (check (b))")
    # (c) PROTECTION: fork parent / rollback target.
    if _is_fork_parent(writer, run_id, step):
        reasons.append(
            f"checkpoint step {step} is named as a fork "
            "parent by an admitted proposal: optimizer state "
            "is retained for fork parents under the PROPOSED "
            "section 6.4 policy (check (c))")
    if _is_rollback_target(writer, run_id, step):
        reasons.append(
            f"checkpoint step {step} is named as a rollback "
            "target by an admitted proposal: optimizer state "
            "is retained for rollback targets under the "
            "PROPOSED section 6.4 policy (check (c))")
    if reasons:
        return GateDecision(VERDICT_REJECT, tuple(reasons))
    return GateDecision(
        VERDICT_ADMIT,
        (f"checkpoint step {step} is complete, carries "
         "optimizer state, and is neither one of the newest "
         f"{NEWEST_KEPT_OPTIMIZER_COUNT} checkpoints nor a "
         "named fork parent or rollback target: the "
         "optimizer drop may proceed under the PROPOSED "
         "section 6.4 policy, subject to in-place hash "
         "verification (flow step (d))",))


# -- flow: propose -> gate -> journal -> execute -----------------------------------


def execute_retention(writer: TrainingJournalWriter, *, run_id, step,
                      action, reason=None, operator=None,
                      on_before_execute: Optional[Callable] = None) \
        -> RetentionResult:
    """The full section 6.4 flow for one retention proposal.

    Journals the ``retention_decision`` row BEFORE any file
    is deleted (ADMIT) or with the REJECT / ESCALATE verdict
    (well-formed proposals); deletes ONLY an admitted
    proposal's optimizer file; appends the new
    checkpoint_manifest revision (optimizer_blob null,
    optimizer_bytes null) after the deletion."""
    run_id = str(run_id)
    step = int(step)
    if not reason or not str(reason).strip():
        return RetentionResult(
            verdict=VERDICT_REJECT,
            gate_reasons=("a retention proposal without a "
                          "reason is refused outright (SI-83 "
                          "section 6.4: proposed, gated, "
                          "journaled -- a reasonless proposal "
                          "is none of those)",),
            run_id=run_id, step=step, action=str(action))
    decision = gate_retention(
        writer, run_id=run_id, step=step, action=action)
    if decision.verdict != VERDICT_ADMIT:
        row = writer.record_retention_decision(
            run_id, step=step, keep=True,
            verdict=decision.verdict, operator=operator,
            reason=str(reason), action=str(action),
            gate_reasons=list(decision.reasons))
        return RetentionResult(
            verdict=decision.verdict,
            gate_reasons=decision.reasons, run_id=run_id,
            step=step, action=str(action),
            retention_seq=row["seq"])
    # (d) Resolve + hash-verify in place BEFORE any deletion
    # is contemplated. A missing file or hash mismatch raises
    # the slice-3 typed error here: nothing is journaled as
    # admitted and nothing is deleted (fail closed).
    verified = verify_checkpoint(writer, run_id, step)
    manifest = find_manifest(writer, run_id, step)
    optimizer_path = verified.optimizer_path
    weights_before = sha256_file_streaming(verified.weights_path)
    bytes_freed = os.path.getsize(optimizer_path)  # MEASURED
    # Journal BEFORE execution.
    row = writer.record_retention_decision(
        run_id, step=step, keep=False, verdict=VERDICT_ADMIT,
        operator=operator, reason=str(reason),
        action=ACTION_DROP_OPTIMIZER,
        gate_reasons=list(decision.reasons),
        manifest_seq=verified.manifest_seq,
        optimizer_blob=verified.optimizer_blob,
        optimizer_path=optimizer_path,
        bytes_freed=bytes_freed)
    if on_before_execute is not None:
        on_before_execute({
            "relation": "retention_decision",
            "retention_seq": row["seq"], "run_id": run_id,
            "step": step, "action": ACTION_DROP_OPTIMIZER,
            "optimizer_path": optimizer_path,
            "weights_path": verified.weights_path,
            "bytes_freed": bytes_freed,
        })
    # Execute: delete ONLY the optimizer file.
    os.remove(optimizer_path)
    weights_after = sha256_file_streaming(verified.weights_path)
    if weights_after != weights_before \
            or weights_after != verified.weights_blob:
        raise WeightsChangedDuringRetention(
            f"weights file {verified.weights_path!r} changed "
            "during an optimizer drop for run "
            f"{run_id!r} step {step}: a retention action "
            "must never touch weights (SI-83 6.4 point 1)")
    # Append the NEW manifest revision: same weights fields,
    # optimizer_blob null, optimizer_bytes null (documented
    # choice -- the journal's absent-optimizer convention).
    extra = {}
    for key in ("weights_file", "completion_marker"):
        if manifest.get(key) is not None:
            extra[key] = manifest[key]
    new_manifest = writer.record_checkpoint_manifest(
        run_id, step, epoch=manifest.get("epoch"),
        weights_blob=manifest.get("weights_blob"),
        weights_path=manifest.get("weights_path"),
        weights_bytes=manifest.get("weights_bytes"),
        optimizer_blob=None, optimizer_bytes=None,
        corpus_hash=manifest.get("corpus_hash"),
        data_order_seed=manifest.get("data_order_seed"),
        hyperparams=manifest.get("hyperparams"),
        trainer_commit=manifest.get("trainer_commit"),
        checkpoint_complete=True,
        optimizer_dropped_by_retention_seq=row["seq"], **extra)
    return RetentionResult(
        verdict=VERDICT_ADMIT, gate_reasons=decision.reasons,
        run_id=run_id, step=step, action=ACTION_DROP_OPTIMIZER,
        retention_seq=row["seq"],
        new_manifest_seq=new_manifest["seq"],
        deleted_path=optimizer_path, bytes_freed=bytes_freed)


# -- disk figures (SI-83 sections 6.2 / 6.4 point 4) --------------------------------


def _gb(byte_count) -> float:
    return round(byte_count / _BYTES_PER_GB, 6)


def retention_figures(writer: TrainingJournalWriter, run_id) -> dict:
    """Per-run disk figures for an operator (SI-83 6.4.4).

    MEASURED figures are summed from the latest manifest
    revisions' ``weights_bytes`` / ``optimizer_bytes``
    fields where present. ESTIMATE figures are recomputed
    from the section 6.2 assumptions in code -- parameter
    count times byte widths -- for a full run at this
    run's checkpoint count. The two are never mixed: the
    measured block is labelled ``"MEASURED"`` and the
    estimate block ``"ESTIMATE"``."""
    latest = _latest_manifests(writer, str(run_id))
    count = len(latest)
    retaining = sum(1 for m in latest.values()
                    if m.get("optimizer_blob"))
    measured_weights = sum(
        m["weights_bytes"] for m in latest.values()
        if isinstance(m.get("weights_bytes"), int))
    measured_optimizer = sum(
        m["optimizer_bytes"] for m in latest.values()
        if isinstance(m.get("optimizer_bytes"), int))
    # Section 6.2 arithmetic, recomputed from the stated
    # assumptions (param count and byte widths), in GB.
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
    return {
        "run_id": str(run_id),
        "checkpoint_count": count,
        "checkpoints_retaining_optimizer": retaining,
        "checkpoints_weights_only": count - retaining,
        "measured": {
            "label": "MEASURED",
            "weights_bytes": measured_weights,
            "optimizer_bytes": measured_optimizer,
            "total_bytes": measured_weights + measured_optimizer,
            "manifests_with_weights_bytes": sum(
                1 for m in latest.values()
                if isinstance(m.get("weights_bytes"), int)),
            "manifests_with_optimizer_bytes": sum(
                1 for m in latest.values()
                if isinstance(m.get("optimizer_bytes"), int)),
        },
        "estimate": {
            "label": "ESTIMATE",
            "param_count": PARAM_COUNT,
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
            "full_run_gb": {
                "layout_a_total": round(layout_a * count, 6),
                "layout_b_total": round(layout_b * count, 6),
                "weights_only_total": round(
                    weights_only * count, 6),
                "proposed_retention_total": round(
                    weights_only * count
                    + optimizer_addon * kept, 6),
            },
        },
    }
