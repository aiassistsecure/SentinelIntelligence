"""SI-87: training fork tooling + gated corpus-prune flow
(SI-83 section 9, slice 4).

Implements SI-83 section 4.4 (fork: branch a new lineage, both
lines kept) and the section 5.3 propose -> gate -> journal ->
land flow for corpus prunes, on top of the section 2 journal
written by :mod:`si.training_journal` (SI-85). Checkpoint
verification reuses slice 3's machinery
(:func:`si.training_rollback.verify_checkpoint`, which
stream-hashes in place via the SI-84 watcher); no parallel
journal and no parallel hash scheme is invented here.

Fork (section 4.4):

- :func:`fork_run` verifies the source checkpoint first --
  the manifest must exist, be ``checkpoint_complete``, and
  its weights blob must hash-verify IN PLACE (stream-hash,
  never a copy: SI never duplicates bytes). Any failure
  raises the slice-3 typed error and nothing is journaled.
- A fork is a GATED proposal: a reason is required (no
  reason -> refused outright, nothing journaled); on
  success the fork is journaled via
  ``writer.record_fork`` -- a ``fork`` proposal row AND the
  new ``training_runs`` row -- BEFORE the descriptor is
  returned. The gate verdict recorded is ``ADMIT``: the
  gate for a fork is exactly "reason present + source
  checkpoint verified", and an unverified or incomplete
  checkpoint never reaches it.
- The new run references the source checkpoint's weights
  blob BY HASH -- the same ``weights_blob`` the source
  manifest references. No weights file is copied, moved,
  or read into memory. Source run rows, checkpoints, and
  eval moments are untouched.
- No trainer is launched: :func:`fork_run` returns a
  :class:`ForkDescriptor` only, exactly like slice 3's
  rollback descriptor. Pointing a trainer at the fork is
  the caller's act, outside this slice.
- Lineage is a tree: :func:`descendants` and
  :func:`ancestors` walk the journal's run rows, so a run
  forked twice and a fork of a fork are both walkable.

Prune (section 5.3) -- never an automatic edit:

- :func:`prune_corpus` runs propose -> gate -> journal ->
  land. The proposal names the source run, the traced
  ``training_windows`` row seqs (section 5.2: windows
  consumed since the last good checkpoint), the journaled
  bad ``eval_moment`` row seqs those windows are tied to,
  the exact example ids proposed for removal, and a
  reason. No reason -> malformed, rejected outright
  (the writer itself refuses a reasonless
  ``prune_proposed`` row; this flow returns a REJECT
  verdict and journals nothing).
- The gate (:func:`gate_prune`, verdicts ADMIT / REJECT /
  ESCALATE in the SI-0 gate's shape) checks:
  (a) traced-window requirement -- every proposed example
      id must trace to one of the named journaled windows
      (its index range, resolved against the corpus file)
      AND the proposal must name at least one journaled
      bad moment for the same run (a moment journaled
      with ``bad=True``); an uncaused prune is REJECTED;
  (b) bounds -- the prune fraction (proposed ids / corpus
      size) must not exceed :data:`MAX_PRUNE_FRACTION`,
      a PROPOSED bound (see below); examples outside the
      traced windows are never pruned by this proposal --
      naming one is a REJECT, not a silent widening;
  (c) order-source honesty (section 5.2) -- a
      configured-order window map supports a prune
      HYPOTHESIS only: unless EVERY traced window's
      ``order_source`` is ``observed``, the gate ESCALATES
      instead of admitting.
  Also REJECTED: unknown window/moment seqs, rows from
  another run, a window whose ``corpus_hash`` does not
  match the corpus file, example ids not in the corpus,
  and an empty proposal.
- Journal before landing: on ADMIT the ``prune_proposed``
  and ``prune_verdict`` rows are journaled BEFORE the new
  corpus file is written. The pruned corpus bytes are
  assembled in memory first so the verdict row can carry
  the new corpus's hash before the file exists; a caller
  ``on_before_land`` hook runs between the verdict and
  the write so tests can prove the ordering.
- Land: the pruned corpus is a NEW file (refusing to
  overwrite any existing path, and never the original's
  path) with its own ``corpus_hash`` (sha256 of its
  bytes). The original corpus file is NEVER edited: it
  is hash-checked before and after the flow and the land
  raises if it changed. The returned result carries the
  ``pruned_from`` provenance edge -- new corpus -> prune
  verdict seq -> traced window seqs + bad moment seqs ->
  original corpus hash -- so the chain is walkable.
- On REJECT or ESCALATE: the proposal and verdict are
  journaled (when the proposal is well-formed) and NO new
  corpus file is created.

The PROPOSED maximum prune fraction:

:data:`MAX_PRUNE_FRACTION` = 0.25 is a PROPOSED bound in
the section 5.3 sense -- the spec names the check ("the
prune does not exceed a PROPOSED maximum fraction of the
corpus per proposal") without fixing the number. 0.25 is
this slice's documented choice, labelled PROPOSED here and
in the gate's reasons; it is not a measured or ratified
limit.

Miniature corpora: a corpus is a JSONL file whose lines
are JSON objects; an example's id is its ``id`` field
when present, else its zero-based line index as a string
(the spec's minimum viable form: index in the canonical
corpus file). Window example ranges are indexes in file
order.

Miniature only: this slice is exercised against tiny
corpora and checkpoint files in temporary directories,
journaled through the real writer on the embedded
backend. No live NEDB write is performed or claimed, no
Imagine run was forked or pruned by this code, and no
trainer is launched. This is machine learning
infrastructure. CPU only, standard library only.

Honesty labels, unchanged: section 18.3 NOT earned; P5-R1 /
P5-R9 UNMEASURED; live model-authored intent OCCURRED scoped
to SI-74 only; L-3 LOCKED; A-2 / A-3 / A-4 UNAPPROVED;
unlocks []; D-F1 OPEN; F-line STOPPED.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from si.training_journal import TrainingJournalWriter
from si.training_rollback import verify_checkpoint
from si.training_watcher import sha256_file_streaming

__all__ = [
    "MAX_PRUNE_FRACTION",
    "VERDICT_ADMIT", "VERDICT_REJECT", "VERDICT_ESCALATE",
    "ForkPruneError", "MissingReason", "ForkTargetExists",
    "CorpusError", "LandRefused", "OriginalCorpusChanged",
    "ForkDescriptor", "GateDecision", "PruneResult",
    "fork_run", "descendants", "ancestors",
    "read_corpus_ids", "corpus_hash_of", "gate_prune",
    "prune_corpus",
]

#: Gate verdicts (SI-83 section 5.3, SI-0 gate shape).
VERDICT_ADMIT = "ADMIT"
VERDICT_REJECT = "REJECT"
VERDICT_ESCALATE = "ESCALATE"

#: PROPOSED maximum fraction of a corpus one prune proposal
#: may remove (SI-83 section 5.3 check (b): the spec requires
#: a maximum fraction bound but does not fix its value; 0.25
#: is this slice's documented PROPOSED choice, not a measured
#: or ratified limit).
MAX_PRUNE_FRACTION = 0.25

_RUNS = "training_runs"
_MOMENTS = "training_eval_moments"
_WINDOWS = "training_windows"


class ForkPruneError(RuntimeError):
    """Base for all fork/prune refusals (fail closed)."""


class MissingReason(ForkPruneError, ValueError):
    """A fork without a reason is refused (SI-83 4.4)."""


class ForkTargetExists(ForkPruneError):
    """The new run id already exists in the journal."""


class CorpusError(ForkPruneError):
    """The corpus file is unreadable or malformed."""


class LandRefused(ForkPruneError):
    """The land step would overwrite an existing file."""


class OriginalCorpusChanged(ForkPruneError):
    """The original corpus changed during the prune flow."""


@dataclass(frozen=True)
class ForkDescriptor:
    """A journaled fork: new run + the verified source blob
    it references BY HASH (no bytes copied, no trainer
    launched)."""
    source_run_id: str
    new_run_id: str
    step: int
    manifest_seq: int
    weights_path: str          # source file, verified in place
    weights_blob: str          # SAME hash the source manifest names
    fork_proposal_seq: int
    fork_run_seq: int
    gate_verdict: str = VERDICT_ADMIT


@dataclass(frozen=True)
class GateDecision:
    """The prune gate's verdict and its stated reasons."""
    verdict: str
    reasons: tuple = ()
    traced_example_ids: tuple = ()
    fraction: float = 0.0
    order_sources: tuple = ()


@dataclass
class PruneResult:
    """Outcome of one propose -> gate -> journal -> land run."""
    verdict: str
    gate_reasons: tuple = ()
    proposed_seq: Optional[int] = None
    verdict_seq: Optional[int] = None
    original_corpus_hash: str = ""
    new_corpus_hash: Optional[str] = None
    new_corpus_path: Optional[str] = None
    pruned_example_ids: tuple = ()
    pruned_from: Optional[dict] = None
    fraction: float = 0.0


# -- fork (SI-83 section 4.4) ---------------------------------------------------


def fork_run(writer: TrainingJournalWriter, source_run_id,
             new_run_id, step, *, reason=None, operator=None,
             hyperparams=None, corpus_hash=None) -> ForkDescriptor:
    """Fork ``source_run_id`` at ``step`` into ``new_run_id``.

    Verifies the source checkpoint in place (slice-3
    machinery: chain, completeness, streamed hash), then
    journals the fork via ``writer.record_fork`` BEFORE
    returning the descriptor. The new run references the
    source weights blob by the same hash; nothing is
    copied and no trainer is launched."""
    if not reason or not str(reason).strip():
        raise MissingReason(
            "a fork without a reason is refused outright "
            "(SI-83 section 4.4: the journal must always be "
            "able to answer 'reimagined from what, and why?')")
    if str(new_run_id) == str(source_run_id):
        raise ForkTargetExists(
            "a fork's new run id must differ from its source "
            f"run id ({source_run_id!r})")
    existing = {r.get("run_id") for r in writer.rows(_RUNS)
                if r.get("run_id")}
    if str(new_run_id) in existing:
        raise ForkTargetExists(
            f"run {new_run_id!r} already exists in the "
            "journal; a fork creates a NEW run row")
    verified = verify_checkpoint(writer, source_run_id, step)
    if corpus_hash is None:
        corpus_hash = verified_manifest_corpus_hash(
            writer, source_run_id, step)
    rows = writer.record_fork(
        str(source_run_id), str(new_run_id),
        parent_step=verified.step,
        parent_manifest_hash=verified.weights_blob,
        parents=[verified.manifest_seq], reason=str(reason),
        verdict=VERDICT_ADMIT, operator=operator,
        hyperparams=hyperparams, corpus_hash=corpus_hash,
        weights_blob=verified.weights_blob)
    return ForkDescriptor(
        source_run_id=str(source_run_id),
        new_run_id=str(new_run_id), step=verified.step,
        manifest_seq=verified.manifest_seq,
        weights_path=verified.weights_path,
        weights_blob=verified.weights_blob,
        fork_proposal_seq=rows["proposal"]["seq"],
        fork_run_seq=rows["run"]["seq"])


def verified_manifest_corpus_hash(writer, run_id, step):
    """The corpus_hash the (run_id, step) manifest recorded,
    or None when it recorded none (never invented)."""
    for row in writer.rows("training_checkpoints"):
        if row.get("kind") == "checkpoint_manifest" \
                and row.get("run_id") == str(run_id) \
                and row.get("step") == int(step):
            return row.get("corpus_hash")
    return None


def ancestors(writer: TrainingJournalWriter, run_id) -> list:
    """Run rows from the lineage root down to ``run_id``
    (root first), over the journal's parent edges."""
    return list(reversed(writer.lineage_to_root(run_id)))


def descendants(writer: TrainingJournalWriter, run_id) -> list:
    """All run rows descended from ``run_id`` (children,
    grandchildren, ...), breadth-first over the journal's
    parent edges -- lineage is a tree (SI-83 4.4)."""
    out, frontier, seen = [], [str(run_id)], {str(run_id)}
    while frontier:
        nxt = []
        for parent in frontier:
            for child in writer.children_of(parent):
                cid = child.get("run_id")
                if cid and cid not in seen:
                    seen.add(cid)
                    out.append(child)
                    nxt.append(cid)
        frontier = nxt
    return out


# -- miniature corpus -------------------------------------------------------------


def _read_corpus_lines(corpus_path) -> list:
    if not os.path.isfile(corpus_path):
        raise CorpusError(f"corpus file {corpus_path!r} not found")
    with open(corpus_path, "rb") as f:
        raw = f.read()
    lines = [ln for ln in raw.split(b"\n") if ln.strip()]
    return lines


def _line_id(line: bytes, index: int) -> str:
    try:
        obj = json.loads(line.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise CorpusError(
            f"corpus line {index} is not valid JSON: {exc}")
    if isinstance(obj, dict) and obj.get("id") is not None:
        return str(obj["id"])
    return str(index)  # index-based id: minimum viable form


def read_corpus_ids(corpus_path) -> list:
    """Example ids of a miniature JSONL corpus, in file
    order (``id`` field when present, else the line index)."""
    return [_line_id(ln, i)
            for i, ln in enumerate(_read_corpus_lines(corpus_path))]


def corpus_hash_of(corpus_path) -> str:
    """sha256 of the corpus file's bytes (streamed)."""
    return sha256_file_streaming(corpus_path)


# -- prune gate (SI-83 section 5.3) -------------------------------------------------


def _rows_by_seq(writer, collection):
    return {r.get("seq"): r for r in writer.rows(collection)}


def gate_prune(writer: TrainingJournalWriter, *, run_id,
               corpus_path, example_ids,
               window_seqs: Sequence[int],
               bad_moment_seqs: Sequence[int]) -> GateDecision:
    """Run the section 5.3 gate checks (read-only; journals
    nothing) and return the verdict with its reasons."""
    reasons = []
    run_id = str(run_id)
    proposed = [str(e) for e in (example_ids or [])]
    corpus_ids = read_corpus_ids(corpus_path)
    corpus_hash = corpus_hash_of(corpus_path)
    total = len(corpus_ids)
    fraction = (len(set(proposed)) / total) if total else 0.0

    windows = _rows_by_seq(writer, _WINDOWS)
    traced = set()
    order_sources = []
    windows_ok = True
    if not window_seqs:
        reasons.append("no traced window named: an uncaused "
                       "prune is rejected (SI-83 5.3 check (c))")
        windows_ok = False
    for seq in (window_seqs or []):
        row = windows.get(seq)
        if row is None or row.get("kind") != "window_consumed":
            reasons.append(f"window seq {seq} is not a "
                           "journaled window_consumed row")
            windows_ok = False
            continue
        if row.get("run_id") != run_id:
            reasons.append(f"window seq {seq} belongs to run "
                           f"{row.get('run_id')!r}, not {run_id!r}")
            windows_ok = False
            continue
        if row.get("corpus_hash") not in (None, corpus_hash):
            reasons.append(
                f"window seq {seq} names corpus "
                f"{row.get('corpus_hash')}, not this corpus "
                f"{corpus_hash}")
            windows_ok = False
            continue
        order_sources.append(row.get("order_source"))
        start, end = row.get("example_start"), row.get("example_end")
        extra_ids = row.get("example_ids")
        if extra_ids:
            traced.update(str(e) for e in extra_ids)
        elif isinstance(start, int) and isinstance(end, int):
            for idx in range(start, end + 1):
                if 0 <= idx < total:
                    traced.add(corpus_ids[idx])
        else:
            reasons.append(f"window seq {seq} carries no "
                           "example range or ids to trace")
            windows_ok = False

    moments = _rows_by_seq(writer, _MOMENTS)
    if not bad_moment_seqs:
        reasons.append("no journaled bad moment named: the "
                       "prune traces to no bad moment "
                       "(SI-83 5.3 check (c))")
    for seq in (bad_moment_seqs or []):
        row = moments.get(seq)
        if row is None or row.get("kind") != "eval_moment":
            reasons.append(f"bad-moment seq {seq} is not a "
                           "journaled eval_moment row")
        elif row.get("run_id") != run_id:
            reasons.append(f"bad-moment seq {seq} belongs to "
                           f"run {row.get('run_id')!r}, "
                           f"not {run_id!r}")
        elif row.get("bad") is not True:
            reasons.append(f"eval moment seq {seq} is not "
                           "journaled as a bad moment "
                           "(bad=True)")

    if not proposed:
        reasons.append("the proposal names no example ids")
    unknown = [e for e in proposed if e not in corpus_ids]
    if unknown:
        reasons.append(f"example ids not in the corpus: "
                       f"{sorted(set(unknown))}")
    if windows_ok:
        outside = [e for e in proposed if e not in traced]
        if outside:
            reasons.append(
                "example ids outside the traced windows are "
                "never pruned by this proposal: "
                f"{sorted(set(outside))} (SI-83 5.3 check (b))")
    if total and fraction > MAX_PRUNE_FRACTION:
        reasons.append(
            f"prune fraction {fraction:.3f} exceeds the "
            f"PROPOSED maximum fraction "
            f"{MAX_PRUNE_FRACTION} per proposal "
            "(SI-83 5.3 check (b))")
    if reasons:
        return GateDecision(VERDICT_REJECT, tuple(reasons),
                            tuple(sorted(traced)), fraction,
                            tuple(order_sources))
    if any(src != "observed" for src in order_sources):
        return GateDecision(
            VERDICT_ESCALATE,
            ("traced windows are not all observed-order "
             f"(order_sources={order_sources}): a "
             "configured-order window map supports a prune "
             "hypothesis only, never an admitted prune "
             "(SI-83 5.2 / 5.3 check (c))",),
            tuple(sorted(traced)), fraction,
            tuple(order_sources))
    return GateDecision(
        VERDICT_ADMIT,
        ("every pruned example traces to a journaled "
         "observed-order window and a journaled bad moment; "
         f"fraction {fraction:.3f} within the PROPOSED "
         f"maximum {MAX_PRUNE_FRACTION}",),
        tuple(sorted(traced)), fraction, tuple(order_sources))


# -- prune flow: propose -> gate -> journal -> land ---------------------------------


def prune_corpus(writer: TrainingJournalWriter, *, run_id,
                 corpus_path, new_corpus_path, example_ids,
                 window_seqs: Sequence[int],
                 bad_moment_seqs: Sequence[int], reason=None,
                 operator=None,
                 on_before_land: Optional[Callable] = None) \
        -> PruneResult:
    """The full section 5.3 flow for one prune proposal.

    Journals ``prune_proposed`` + ``prune_verdict`` BEFORE
    any new corpus file exists (on ADMIT); lands the pruned
    corpus as a NEW file only on ADMIT; never edits the
    original (hash-checked before and after). On REJECT or
    ESCALATE no new corpus file is created."""
    run_id = str(run_id)
    original_hash_before = corpus_hash_of(corpus_path)
    if not reason or not str(reason).strip():
        return PruneResult(
            verdict=VERDICT_REJECT,
            gate_reasons=("a prune proposal without a reason "
                          "is malformed and rejected outright "
                          "(SI-83 5.3 step 1)",),
            original_corpus_hash=original_hash_before,
            pruned_example_ids=tuple(
                str(e) for e in (example_ids or [])))
    decision = gate_prune(
        writer, run_id=run_id, corpus_path=corpus_path,
        example_ids=example_ids, window_seqs=window_seqs,
        bad_moment_seqs=bad_moment_seqs)
    proposed_ids = [str(e) for e in (example_ids or [])]
    step_from = step_to = None
    windows = _rows_by_seq(writer, _WINDOWS)
    spans = [(windows[s].get("step_from"), windows[s].get("step_to"))
             for s in (window_seqs or []) if s in windows]
    if spans:
        step_from = min(s[0] for s in spans if s[0] is not None) \
            if any(s[0] is not None for s in spans) else None
        step_to = max(s[1] for s in spans if s[1] is not None) \
            if any(s[1] is not None for s in spans) else None
    proposed_row = writer.record_prune_proposed(
        run_id, reason=str(reason), step_from=step_from,
        step_to=step_to, window_seqs=list(window_seqs or []),
        example_ids=proposed_ids,
        bad_moment_seqs=list(bad_moment_seqs or []),
        original_corpus_hash=original_hash_before)

    if decision.verdict != VERDICT_ADMIT:
        verdict_row = writer.record_prune_verdict(
            run_id, verdict=decision.verdict,
            parents=[proposed_row["seq"]],
            operator=operator,
            gate_reasons=list(decision.reasons))
        return PruneResult(
            verdict=decision.verdict,
            gate_reasons=decision.reasons,
            proposed_seq=proposed_row["seq"],
            verdict_seq=verdict_row["seq"],
            original_corpus_hash=original_hash_before,
            pruned_example_ids=tuple(proposed_ids),
            fraction=decision.fraction)

    # ADMIT: assemble the pruned bytes and their hash BEFORE
    # journaling the verdict, so the verdict row can carry
    # new_corpus_hash before any new corpus file exists.
    lines = _read_corpus_lines(corpus_path)
    ids = [_line_id(ln, i) for i, ln in enumerate(lines)]
    remove = set(proposed_ids)
    kept = [ln for ln, eid in zip(lines, ids) if eid not in remove]
    new_bytes = b"".join(ln + b"\n" for ln in kept)
    new_hash = "sha256:" + hashlib.sha256(new_bytes).hexdigest()
    verdict_row = writer.record_prune_verdict(
        run_id, verdict=VERDICT_ADMIT,
        parents=[proposed_row["seq"]], operator=operator,
        new_corpus_hash=new_hash,
        gate_reasons=list(decision.reasons),
        window_seqs=list(window_seqs or []),
        bad_moment_seqs=list(bad_moment_seqs or []),
        original_corpus_hash=original_hash_before)
    edge = {
        "relation": "pruned_from",
        "new_corpus_hash": new_hash,
        "original_corpus_hash": original_hash_before,
        "prune_proposed_seq": proposed_row["seq"],
        "prune_verdict_seq": verdict_row["seq"],
        "window_seqs": list(window_seqs or []),
        "bad_moment_seqs": list(bad_moment_seqs or []),
        "example_ids": proposed_ids,
        "run_id": run_id,
    }
    if on_before_land is not None:
        on_before_land(edge)
    # Land: a NEW file only -- never the original's path,
    # never over an existing file.
    if os.path.abspath(new_corpus_path) == os.path.abspath(corpus_path):
        raise LandRefused(
            "the pruned corpus must be a NEW file; refusing "
            "to land over the original corpus path")
    if os.path.exists(new_corpus_path):
        raise LandRefused(
            f"new corpus path {new_corpus_path!r} already "
            "exists; refusing to overwrite")
    with open(new_corpus_path, "wb") as f:
        f.write(new_bytes)
    landed_hash = corpus_hash_of(new_corpus_path)
    if landed_hash != new_hash:
        raise CorpusError(
            f"landed corpus hash {landed_hash} != journaled "
            f"new_corpus_hash {new_hash}")
    if corpus_hash_of(corpus_path) != original_hash_before:
        raise OriginalCorpusChanged(
            "the original corpus file changed during the "
            "prune flow; a prune never edits the original "
            "(SI-83 5.3)")
    return PruneResult(
        verdict=VERDICT_ADMIT, gate_reasons=decision.reasons,
        proposed_seq=proposed_row["seq"],
        verdict_seq=verdict_row["seq"],
        original_corpus_hash=original_hash_before,
        new_corpus_hash=new_hash,
        new_corpus_path=str(new_corpus_path),
        pruned_example_ids=tuple(proposed_ids),
        pruned_from=edge, fraction=decision.fraction)
