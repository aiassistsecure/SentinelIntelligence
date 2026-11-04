"""SI-86: training rollback / replay tooling (SI-83 section 9,
slice 3).

Implements SI-83 section 4.2 (rollback: rehydrate-and-verify)
and section 4.3 (eval replay) against the section 2 journal
written by :mod:`si.training_journal` (SI-85). It reuses that
writer and the SI-84 watcher's streaming hash
(:func:`si.training_watcher.sha256_file_streaming`); it invents
no parallel journal and no parallel hash scheme.

What this tool does, and what it does NOT do:

- :func:`verify_checkpoint` resolves a journaled
  ``checkpoint_manifest`` row to its referenced files and
  verifies them IN PLACE: the journal chain must verify, the
  manifest must exist and be ``checkpoint_complete``, the
  weights file must exist and its streamed SHA-256 must equal
  the manifest's ``weights_blob``, and -- when the manifest
  carries an ``optimizer_blob`` -- an optimizer file must be
  found and must hash-verify too. Any failure raises a typed
  error and nothing is journaled (fail closed, section 4.2).
  No file is ever copied: SI never duplicates bytes --
  rehydrate means resolve + verify the referenced file where
  it already lives.
- :func:`resume` is the rollback entry point. It verifies
  first, then journals a ``rollback`` row via
  ``writer.record_rollback`` -- target step, the caller's
  reason (required; an empty/missing reason is refused before
  anything is verified or journaled), the gate verdict and
  operator passed in by the caller, and ``parents`` = the
  target manifest's seq -- and only then returns a
  :class:`ResumeDescriptor`. The rollback row itself carries
  the resume classification (section 9 slice-3 gate). The
  original run's rows are never modified and no later
  checkpoint or row is ever deleted. This tool does NOT
  launch any trainer: it returns the verified descriptor
  only, and pointing a trainer at the checkpoint is the
  caller's act, outside this slice.
- :func:`replay_eval` re-runs a caller-supplied eval function
  against a hash-verified checkpoint and writes the result
  as a NEW ``eval_moment`` row (section 4.3; see below).

Resume classification (section 4.2, the honesty core):

- ``"exact"`` -- ONLY when the manifest's ``optimizer_blob``
  is non-null AND the optimizer file hash-verifies.
- ``"weights-exact/optimizer-fresh"`` -- when the weights
  verify and ``optimizer_blob`` is null (weights only).
  A weights-only resume is never labelled "exact" anywhere
  in this module: the weights are bit-for-bit the step's
  weights, but the optimizer starts cold, so the resumed
  trajectory is a NEW trajectory from those weights.

File resolution conventions (section 2.3 gives the manifest
``weights_path`` but NO optimizer path field; nothing is
invented silently -- the conventions are these):

- Weights file: (1) ``weights_path`` itself when it is a
  file; (2) ``weights_path`` joined with the manifest's
  ``weights_file`` extra when the watcher recorded one
  (watcher manifests name the step DIRECTORY in
  ``weights_path``); (3) otherwise a scan of the
  ``weights_path`` directory using the watcher's own
  file-picking convention (``si.training_watcher._snapshot``
  / ``_pick``, reused, not re-implemented).
- Optimizer file (only consulted when ``optimizer_blob`` is
  non-null): (1) the manifest's ``optimizer_path`` extra
  when present (absolute, or relative to the checkpoint
  directory); (2) the manifest's ``optimizer_file`` extra
  (relative to the checkpoint directory); (3) otherwise the
  watcher's directory-scan convention over the checkpoint
  directory (the directory holding the weights file), which
  is the same hint rule the watcher used to compute
  ``optimizer_blob`` in the first place. No candidate found
  means :class:`OptimizerFileMissing` -- fail closed, never
  a downgrade to a weights-only label: a manifest that
  claims optimizer state whose file cannot be verified is
  not an exact resume and not a fresh one either; it is a
  broken reference.

Eval replay provenance (section 4.3): the replay row is
written via ``writer.record_eval_moment`` with the writer's
extra-kwargs channel carrying ``replay_of_seq`` and
``replay_of_entry_hash`` (the seq and entry hash of the
ORIGINAL ``eval_moment`` for that run/step) and
``provenance_relation="replayed_from"``. "supersedes" is
deliberately NOT used: a replay never overwrites or
supersedes the original -- both rows stand, and if the
replay scores disagree with the original, the disagreement
is itself the finding. The replay always records the suite
identity of the original moment (``eval_suite`` /
``eval_suite_commit``) -- a replay under a different suite
identity would not be a replay of that moment -- and the
verified ``weights_blob`` it ran against. If NO original
``eval_moment`` exists for the checkpoint, replay fails
closed with :class:`NoOriginalEvalMoment` and writes
nothing: a replay is by definition a re-run of a journaled
moment, and a first-ever evaluation is a different act that
this tool does not dress up as a replay. A hash mismatch
means the eval function is never called and no row is
written.

Forward training replay (section 4.3) is NOT built and NOT
claimed: :func:`forward_replay` is a guard only. Where the
manifest's ``data_order_seed`` is null, forward replay is
not faithful and must not be attempted under that name, so
the guard refuses with that reason; where a seed IS
recorded it still refuses, because step-replay of training
does not exist in this slice.

Miniature only: this slice is exercised against tiny
checkpoint files in temporary directories, journaled through
the real writer on the embedded backend. No live NEDB write
is performed or claimed, and no Imagine-scale claim is made.
This is machine learning infrastructure. CPU only, standard
library only.

Honesty labels, unchanged: section 18.3 NOT earned; P5-R1 /
P5-R9 UNMEASURED; live model-authored intent OCCURRED scoped
to SI-74 only; L-3 LOCKED; A-2 / A-3 / A-4 UNAPPROVED;
unlocks []; D-F1 OPEN; F-line STOPPED.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

from si.training_journal import TrainingJournalWriter
from si.training_watcher import _pick, _snapshot, sha256_file_streaming

__all__ = [
    "RESUME_EXACT", "RESUME_WEIGHTS_EXACT",
    "RollbackError", "JournalChainInvalid", "CheckpointNotFound",
    "CheckpointIncomplete", "WeightsReferenceMissing",
    "WeightsFileMissing", "WeightsHashMismatch",
    "OptimizerFileMissing", "OptimizerHashMismatch",
    "MissingReason", "NoOriginalEvalMoment",
    "ForwardReplayRefused",
    "VerifiedCheckpoint", "ResumeDescriptor",
    "find_manifest", "verify_checkpoint", "resume", "replay_eval",
    "forward_replay",
]

#: The only two resume classifications (SI-83 section 4.2).
RESUME_EXACT = "exact"
RESUME_WEIGHTS_EXACT = "weights-exact/optimizer-fresh"

_CHECKPOINTS = "training_checkpoints"
_MOMENTS = "training_eval_moments"
_EVAL_FIELDS = ("tier0_score", "discrimination_score", "temporal_score",
                "loss", "grad_norm_mean", "grad_norm_max",
                "sample_generations")


class RollbackError(RuntimeError):
    """Base for all rollback/replay refusals (fail closed)."""


class JournalChainInvalid(RollbackError):
    """The journal chain does not verify; nothing may proceed."""


class CheckpointNotFound(RollbackError):
    """No checkpoint_manifest row for the (run_id, step)."""


class CheckpointIncomplete(RollbackError):
    """The manifest is checkpoint_complete=false (partial save)."""


class WeightsReferenceMissing(RollbackError):
    """The manifest names no weights hash and/or no weights path."""


class WeightsFileMissing(RollbackError):
    """The referenced weights file does not exist."""


class WeightsHashMismatch(RollbackError):
    """The weights file's hash != the manifest's weights_blob."""


class OptimizerFileMissing(RollbackError):
    """optimizer_blob is set but no optimizer file was found."""


class OptimizerHashMismatch(RollbackError):
    """The optimizer file's hash != the manifest's optimizer_blob."""


class MissingReason(RollbackError, ValueError):
    """A rollback without a reason is refused (SI-83 4.2)."""


class NoOriginalEvalMoment(RollbackError):
    """Replay was asked for a checkpoint with no original moment."""


class ForwardReplayRefused(RollbackError):
    """Forward training replay refused (SI-83 4.3: not built)."""


@dataclass(frozen=True)
class VerifiedCheckpoint:
    """A manifest whose referenced files hash-verified in place."""
    run_id: str
    step: int
    manifest_seq: int
    weights_path: str          # resolved weights FILE path
    weights_blob: str
    optimizer_path: Optional[str]
    optimizer_blob: Optional[str]
    resume_kind: str           # RESUME_EXACT or RESUME_WEIGHTS_EXACT


@dataclass(frozen=True)
class ResumeDescriptor(VerifiedCheckpoint):
    """Verified checkpoint + the rollback row journaled for it."""
    rollback_seq: int = 0


def find_manifest(writer: TrainingJournalWriter, run_id, step) -> dict:
    """The checkpoint_manifest row for (run_id, step), or raise."""
    rows = [r for r in writer.rows(_CHECKPOINTS)
            if r.get("kind") == "checkpoint_manifest"
            and r.get("run_id") == str(run_id)
            and r.get("step") == int(step)]
    if not rows:
        raise CheckpointNotFound(
            f"no checkpoint_manifest for run {run_id!r} step "
            f"{int(step)}; rollback/replay refused")
    return rows[-1]


def _resolve_weights_file(manifest: dict) -> str:
    weights_path = manifest.get("weights_path")
    if not weights_path or not isinstance(weights_path, str):
        raise WeightsReferenceMissing(
            "manifest carries no weights_path; nothing to "
            "rehydrate (SI-83 section 2.6: no hash/path, no blob)")
    if os.path.isfile(weights_path):
        return weights_path
    if os.path.isdir(weights_path):
        rel = manifest.get("weights_file")
        if rel:
            candidate = os.path.join(weights_path, rel)
            if os.path.isfile(candidate):
                return candidate
            raise WeightsFileMissing(
                f"manifest weights_file {rel!r} not found under "
                f"{weights_path!r}")
        picked, _opt = _pick(_snapshot(weights_path), weights_path,
                             manifest.get("completion_marker") or "")
        if picked:
            return os.path.join(weights_path, picked)
        raise WeightsFileMissing(
            f"no weights file found in checkpoint directory "
            f"{weights_path!r} by the watcher naming convention")
    raise WeightsFileMissing(
        f"weights_path {weights_path!r} is neither a file nor a "
        "directory on disk")


def _resolve_optimizer_file(manifest: dict, weights_file: str) -> str:
    ckpt_dir = os.path.dirname(os.path.abspath(weights_file))
    opt_path = manifest.get("optimizer_path")
    if opt_path:
        candidate = opt_path if os.path.isabs(opt_path) \
            else os.path.join(ckpt_dir, opt_path)
        if os.path.isfile(candidate):
            return candidate
        raise OptimizerFileMissing(
            f"manifest optimizer_path {opt_path!r} does not "
            "resolve to a file")
    opt_file = manifest.get("optimizer_file")
    if opt_file:
        candidate = os.path.join(ckpt_dir, opt_file)
        if os.path.isfile(candidate):
            return candidate
        raise OptimizerFileMissing(
            f"manifest optimizer_file {opt_file!r} not found in "
            f"checkpoint directory {ckpt_dir!r}")
    _w, picked = _pick(_snapshot(ckpt_dir), ckpt_dir,
                       manifest.get("completion_marker") or "")
    if picked:
        candidate = os.path.join(ckpt_dir, picked)
        if os.path.abspath(candidate) != os.path.abspath(weights_file):
            return candidate
    raise OptimizerFileMissing(
        "manifest carries optimizer_blob but no optimizer file "
        f"was found in checkpoint directory {ckpt_dir!r} by the "
        "watcher naming convention; exact resume refused")


def verify_checkpoint(writer: TrainingJournalWriter, run_id, step) \
        -> VerifiedCheckpoint:
    """Rehydrate-and-verify (SI-83 4.2): resolve + hash-check
    the checkpoint's files in place. Journals nothing. Raises a
    typed :class:`RollbackError` on any failure."""
    if not writer.verify_chain():
        raise JournalChainInvalid(
            "training journal chain does not verify; a journal "
            "that cannot trust its own memory must not drive a "
            "rollback (SI-83 section 2.1 rule 3)")
    manifest = find_manifest(writer, run_id, step)
    if manifest.get("checkpoint_complete") is not True:
        raise CheckpointIncomplete(
            f"checkpoint run {run_id!r} step {int(step)} is "
            "checkpoint_complete=false (partial save); never a "
            "rollback target")
    weights_blob = manifest.get("weights_blob")
    if not weights_blob or not isinstance(weights_blob, str):
        raise WeightsReferenceMissing(
            "manifest carries no weights_blob content hash; "
            "nothing verifiable to rehydrate (SI-83 2.6)")
    weights_file = _resolve_weights_file(manifest)
    actual = sha256_file_streaming(weights_file)
    if actual != weights_blob:
        raise WeightsHashMismatch(
            f"weights hash mismatch for run {run_id!r} step "
            f"{int(step)}: manifest {weights_blob} != file "
            f"{actual} ({weights_file}); fail closed")
    optimizer_blob = manifest.get("optimizer_blob")
    optimizer_file = None
    if optimizer_blob:
        optimizer_file = _resolve_optimizer_file(manifest, weights_file)
        opt_actual = sha256_file_streaming(optimizer_file)
        if opt_actual != optimizer_blob:
            raise OptimizerHashMismatch(
                f"optimizer hash mismatch for run {run_id!r} "
                f"step {int(step)}: manifest {optimizer_blob} "
                f"!= file {opt_actual} ({optimizer_file}); "
                "fail closed")
        resume_kind = RESUME_EXACT
    else:
        resume_kind = RESUME_WEIGHTS_EXACT
    return VerifiedCheckpoint(
        run_id=str(run_id), step=int(step),
        manifest_seq=manifest["seq"], weights_path=weights_file,
        weights_blob=weights_blob, optimizer_path=optimizer_file,
        optimizer_blob=optimizer_blob, resume_kind=resume_kind)


def resume(writer: TrainingJournalWriter, run_id, step, *,
           reason=None, verdict=None, operator=None) \
        -> ResumeDescriptor:
    """Rollback entry point (SI-83 4.2): verify, journal the
    ``rollback`` row BEFORE returning, then return the verified
    descriptor. Launches no trainer."""
    if not reason or not str(reason).strip():
        raise MissingReason(
            "a rollback without a reason is refused (SI-83 "
            "section 4.2: the proposal names the target step, "
            "the run, and the reason)")
    verified = verify_checkpoint(writer, run_id, step)
    row = writer.record_rollback(
        verified.run_id, target_step=verified.step,
        reason=str(reason), verdict=verdict,
        parents=[verified.manifest_seq], operator=operator,
        resume_kind=verified.resume_kind,
        weights_blob=verified.weights_blob,
        optimizer_blob=verified.optimizer_blob)
    return ResumeDescriptor(
        run_id=verified.run_id, step=verified.step,
        manifest_seq=verified.manifest_seq,
        weights_path=verified.weights_path,
        weights_blob=verified.weights_blob,
        optimizer_path=verified.optimizer_path,
        optimizer_blob=verified.optimizer_blob,
        resume_kind=verified.resume_kind, rollback_seq=row["seq"])


def _original_moment(writer: TrainingJournalWriter, run_id, step):
    """First non-replay eval_moment for (run_id, step), by seq."""
    moments = [r for r in writer.rows(_MOMENTS)
               if r.get("kind") == "eval_moment"
               and r.get("run_id") == str(run_id)
               and r.get("step") == int(step)
               and r.get("replay_of_seq") is None]
    return moments[0] if moments else None


def replay_eval(writer: TrainingJournalWriter, run_id, step,
                eval_fn: Callable[[str], Mapping], *,
                operator=None, reason=None) -> dict:
    """Eval replay (SI-83 4.3): hash-verify the checkpoint,
    re-run ``eval_fn(weights_path)``, and journal the result
    as a NEW eval_moment linked to the original by
    ``replay_of_seq`` / ``replay_of_entry_hash`` /
    ``provenance_relation="replayed_from"``, under the
    ORIGINAL moment's eval-suite identity. Scores are
    recorded exactly as ``eval_fn`` reports them. The
    original row is never modified."""
    verified = verify_checkpoint(writer, run_id, step)
    original = _original_moment(writer, run_id, step)
    if original is None:
        raise NoOriginalEvalMoment(
            f"no original eval_moment for run {run_id!r} step "
            f"{int(step)}; replay is a re-run of a journaled "
            "moment and refuses to invent one")
    result = eval_fn(verified.weights_path)
    if not isinstance(result, Mapping):
        raise TypeError(
            f"eval_fn must return a mapping of scores, got "
            f"{type(result).__name__}")
    fields = {k: result[k] for k in _EVAL_FIELDS if k in result}
    return writer.record_eval_moment(
        verified.run_id, verified.step,
        eval_suite=original.get("eval_suite"),
        eval_suite_commit=original.get("eval_suite_commit"),
        replay_of_seq=original["seq"],
        replay_of_entry_hash=original.get("entry_hash"),
        provenance_relation="replayed_from",
        replayed_weights_blob=verified.weights_blob,
        replay_operator=operator, replay_reason=reason, **fields)


def forward_replay(writer: TrainingJournalWriter, run_id, step,
                   **_kwargs):
    """Guard only: forward TRAINING replay is not built.

    SI-83 section 4.3: where ``data_order_seed`` is null,
    forward replay is not faithful and must not be attempted
    under that name. Where a seed IS recorded, this slice
    still builds no step-replay of training, so the guard
    refuses either way -- with the accurate reason."""
    manifest = find_manifest(writer, run_id, step)
    if manifest.get("data_order_seed") is None:
        raise ForwardReplayRefused(
            f"forward training replay refused for run {run_id!r} "
            f"step {int(step)}: the manifest's data_order_seed "
            "is null, so forward replay is not faithful and "
            "must not be attempted under that name (SI-83 "
            "section 4.3)")
    raise ForwardReplayRefused(
        f"forward training replay refused for run {run_id!r} "
        f"step {int(step)}: step-replay of training is not "
        "built in this slice (SI-83 section 4.3 claims only "
        "that the journal records what an attempt would need)")
