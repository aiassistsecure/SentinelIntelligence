"""Gaia multi-cycle capture driver — SI-109 (SI-106 §12 item 1;
SI-108 §4.1).

What this module is, stated honestly: the in-repo form of the
SI-106 §9 capture disciplines, as code. It implements:

- the §4 item shape — one item is one **task run**: cycle 0
  (unscored, it seeds the journal), journaling between cycles
  **without execution** (§9.5), and cycles 1..K entered as
  scored documents only when their capture perception carried
  a non-empty ``prior_cycles``;
- the §9.2 residency-guard interface — a poll before **every**
  attempt; a non-Resident poll (or a poll that itself fails)
  halts the whole capture, spends **no** attempt, persists
  state, and is never restarted around by this driver;
- the §9.3 attempt budgets — at most ``max_attempts`` per
  workload per cycle position, fixed when the driver is
  constructed and never extended by any code path in this
  module (there is deliberately no method that raises one);
- the §9.4 smoke-gate shape, in fixture scope
  (:func:`run_fixture_smoke_gate`);
- the §9.6 freeze — corpus and capture journal frozen
  together, with the sha256 of each recorded and independently
  recomputable (:func:`freeze_multicycle`,
  :func:`verify_freeze`).

SI-112 wiring (SI-111 §3.3; D-111-2): constructed with a
candidate registration, the driver runs the pre-flight
template lock (:mod:`si.gaia.preflight`) before its first
task — a refusal spends nothing and journals nothing — and
the §9.2 guard poll re-verifies the served template hash on
every poll, halting with no attempt spent if a mid-capture
restart changes it (§3.3(c)). With no registration the
driver behaves exactly as SI-109 left it.

Provenance and scope, stated plainly: SI-107 ran this design
live from workspace scripts under the lead's direct dispatch.
This module does **not** retroactively make SI-107 a repo-code
run — its record stands as a workspace-script run. This driver
is built and proven **only** against replay / scripted proposal
sources and fixture bundles. It ships with **no authorisation
to make a live capture** (SI-106 §9.1 authorisation remains a
separate, later, recorded act for any future run), it contacts
no model by itself, and it earns reproducibility only — no
§18.1 row, no §18.3 progress, no claim about any model
(SI-108 §4.3).

Journaling without execution (§9.5) is the §9.5-faithful form
SI-107's smoke gate corrected itself to: the queue is never
drained and no tool runs, so the proof is (1) **zero**
``queue_claim`` rows, and (2) every ``tool_result`` row is a
**non-execution disposition** — the admission path's own
REJECT / ESCALATE records (or the B-1 shadow record), carrying
a gate verdict and no measured execution fields. An ADMITted
intent journals no ``tool_result`` row at all in this walk: it
sits ``queued`` in the fold, unexecuted. See
:func:`collect_no_execution_proof`.

This module touches no gate pricing, no parser stage rule,
and no teaching text (SI-106 §12's review rule for this slice).

CPU only, stdlib only. Terminology: machine learning
infrastructure throughout.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil

from si.bundle import canonical_json, sha256_hex
from si.gaia.conformance_capture import ResidencyGuardHalt
from si.gaia.context import ToolContext
from si.gaia.drain import admit_tool_intent
from si.gaia.proposer import assemble_perception, parse_proposal
from si.gaia.preflight import PreflightRefused, check_preflight
from si.gaia.queue import record_queue_batch
from si.gaia.sibling_resolution import (
    build_resolution_refusal_record,
    journal_resolution_refusal,
    resolve_sibling_dependencies,
)
from si.gaia.task import _run_completion, record_task_entry
from si.gate import Intent
from si.journal import COLLECTIONS, open_journal

__all__ = [
    "ResidencyGuardHalt",
    "PreflightRefused",
    "SmokeGateError",
    "MultiCycleDriver",
    "journal_cycle",
    "collect_no_execution_proof",
    "is_non_execution_disposition",
    "build_multicycle_corpus",
    "scored_documents",
    "fold_rows_and_hash",
    "corpus_canonical_sha256",
    "freeze_multicycle",
    "verify_freeze",
    "run_fixture_smoke_gate",
    "GUARD_HALT_EXIT_STATUS",
]

#: The exit status a supervising unit records for a guard halt
#: (SI-106 §9.6's pattern: the status sits in
#: ``RestartPreventExitStatus`` so a halt is never restarted
#: around). This driver only raises; the constant names the
#: convention for any future unit file.
GUARD_HALT_EXIT_STATUS = 86

#: Execution fields that exist only when a tool actually ran.
#: A ``tool_result`` execution record carrying any of them is
#: an execution-shaped row, never a non-execution disposition.
MEASURED_EXECUTION_FIELDS = (
    "exit_code", "duration_ms", "wall_ms", "stdout", "stderr",
    "result_sha256", "output_sha256", "result_bytes",
)


def _hash_text(text: str) -> str:
    return "sha256:" + sha256_hex(text.encode("utf-8"))


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# §9.5 — journaling between cycles, without execution
# ---------------------------------------------------------------------------

def journal_cycle(journal, gate, ctx, task, body, *,
                  cycles_used: int, cycle_cap: int,
                  task_intent_seqs: set,
                  resume: dict | None = None,
                  on_member=None) -> tuple:
    """Journal one entered cycle document per SI-106 §9.5.

    The walk mirrors, step for step, the admission walk of
    :func:`si.gaia.task.run_task` — parse, SI-97 §6 sibling
    resolution from the walk's own map, a ``queue_batch`` entry
    per batch, :func:`si.gaia.drain.admit_tool_intent` per
    member, one cycle-budget debit per intent evaluated — with
    ONE designed difference: **the queue is never drained, so
    no tool executes.** Gate/task stop rules do not end the
    walk: a task run's cycle shape is fixed by SI-106 §4.1 and
    verdicts are journaled as fold data.

    ``task`` is ``{"seq", "ref"}`` from
    :func:`si.gaia.task.record_task_entry`; ``ctx`` is a
    :class:`si.gaia.context.ToolContext` with ``tool_task_seq``
    already set. ``resume`` (a driver interrupted mid-walk)
    carries ``{"members_done", "seq_by_position",
    "refused_positions", ...}`` so the walk continues where it
    stopped. ``on_member(outcome)`` fires after every journaled
    member so the caller can persist progress incrementally.

    Returns ``(outcome, cycles_used)``.
    """
    outcome = {
        "parse_outcome": None,
        "failing_stage": None,
        "failing_stage_name": None,
        "parse_detail": None,
        "task_signal": None,
        "batch_seq": None,
        "members": [],
        "refused": [],
        "completion": None,
        "budget_halt": False,
        "journaled_intent_seqs": [],
    }
    parse_result = parse_proposal(
        body, journal=journal, task_seq=task["seq"])
    outcome["parse_outcome"] = parse_result["outcome"]
    outcome["failing_stage"] = parse_result.get("failing_stage")
    outcome["failing_stage_name"] = parse_result.get(
        "failing_stage_name")
    outcome["parse_detail"] = parse_result.get("detail")
    outcome["task_signal"] = parse_result.get("task_signal")
    if parse_result["outcome"] != "ok":
        return outcome, cycles_used

    translation = parse_result.get("signal_translation")
    if isinstance(translation, Intent):
        # §8.7 completion signal: gated and journaled as an
        # ordinary memory intent via run_task's own completion
        # path — journaling only, no tool execution, and the
        # capture's task run is not stopped by it (SI-106 §4.1).
        comp = _run_completion(journal, gate, ctx, translation)
        outcome["completion"] = {
            "verdict": comp["verdict"],
            "edit_id": comp["edit_id"],
            "execution_status": comp["execution_status"],
        }
        return outcome, cycles_used
    if translation is not None:
        # §8.8 inability record: nothing proposed, nothing
        # journaled by the walk.
        outcome["inability"] = translation
        return outcome, cycles_used

    intents = list(parse_result.get("intents") or [])
    if not intents:
        return outcome, cycles_used

    members_done = 0
    resolution_map: dict = {}
    refused_positions: set = set()
    visited_positions: set = set()
    if resume:
        members_done = int(resume.get("members_done") or 0)
        for pos, seq in (resume.get("seq_by_position") or {}).items():
            resolution_map[int(pos)] = seq
            visited_positions.add(int(pos))
        refused_positions = {
            int(p) for p in resume.get("refused_positions") or []}
        visited_positions |= refused_positions
        outcome["members"] = list(resume.get("members") or [])
        outcome["refused"] = list(resume.get("refused") or [])
        outcome["journaled_intent_seqs"] = list(
            resume.get("journaled_intent_seqs") or [])
        outcome["batch_seq"] = resume.get("batch_seq")

    if outcome["batch_seq"] is None:
        batch_entry = record_queue_batch(
            journal, task["seq"], task["ref"], len(intents),
            proposer="mind")
        outcome["batch_seq"] = batch_entry["seq"]
        outcome["members_done"] = members_done
        outcome["seq_by_position"] = {
            str(k): v for k, v in resolution_map.items()}
        outcome["refused_positions"] = sorted(refused_positions)
        if on_member is not None:
            on_member(outcome)

    for position, member in enumerate(intents):
        if position < members_done:
            continue
        if cycles_used >= cycle_cap:
            outcome["budget_halt"] = True
            break
        ctx.tool_cycles_used = cycles_used
        ctx.tool_result_bytes_used = 0
        ctx.tool_journal_tip = journal.tip()[0]
        ctx.tool_task_intent_seqs = set(task_intent_seqs)
        resolution = resolve_sibling_dependencies(
            member, resolution_map=resolution_map,
            refused_positions=frozenset(refused_positions),
            visited_positions=frozenset(visited_positions))
        if not resolution["resolved"]:
            refused_positions.add(position)
            refusal_seq = journal_resolution_refusal(
                journal, build_resolution_refusal_record(
                    task_seq=task["seq"], position=position,
                    failed=resolution["failed"],
                    batch_seq=outcome["batch_seq"]))
            outcome["refused"].append({
                "position": position,
                "tool": member.tool,
                "failed": resolution["failed"],
                "refusal_seq": refusal_seq,
            })
            members_done = position + 1
            outcome["members_done"] = members_done
            outcome["refused_positions"] = sorted(refused_positions)
            if on_member is not None:
                on_member(outcome)
            continue
        visited_positions.add(position)
        member = resolution["intent"]
        result = admit_tool_intent(
            journal, gate, ctx, member,
            batch_seq=outcome["batch_seq"])
        cycles_used += 1
        rows = journal.backend.find(
            "self_edits", filter={"edit_id": result["edit_id"]},
            limit=1)
        seq = rows[0]["seq"] if rows else None
        if seq is not None:
            task_intent_seqs.add(seq)
            resolution_map[position] = seq
            outcome["journaled_intent_seqs"].append(seq)
        outcome["members"].append({
            "position": position,
            "tool": member.tool,
            "verdict": result["verdict"],
            "edit_id": result["edit_id"],
            "intent_seq": seq,
            "effective_class": result["effective_class"],
            "had_sibling": bool(resolution["had_sibling"]),
            "authored_depends_on": (
                member.authored_depends_on
                if getattr(member, "authored_depends_on", None)
                else [d for d in (member.depends_on or [])]),
        })
        members_done = position + 1
        outcome["members_done"] = members_done
        outcome["seq_by_position"] = {
            str(k): v for k, v in resolution_map.items()}
        if on_member is not None:
            on_member(outcome)
    outcome["members_done"] = members_done
    outcome["seq_by_position"] = {
        str(k): v for k, v in resolution_map.items()}
    outcome["refused_positions"] = sorted(refused_positions)
    return outcome, cycles_used


# ---------------------------------------------------------------------------
# §9.5 proof — journaling without execution, asserted
# ---------------------------------------------------------------------------

def is_non_execution_disposition(record: dict) -> bool:
    """True iff one ``tool_result`` row's record is a
    non-execution disposition in the §9.5-faithful form: its
    gate verdict is REJECT or ESCALATE (an ADMIT verdict's
    genuine outcome is an execution), and its execution record
    carries a verdict and **no** measured execution field —
    nothing ran, so no exit code, hash, or wall time exists to
    record. The B-1 shadow record satisfies this by the same
    reading (its verdict rides in the execution record)."""
    if not isinstance(record, dict):
        return False
    execution = record.get("execution")
    if not isinstance(execution, dict):
        return False
    verdict = record.get("gate_verdict") or execution.get("verdict")
    if verdict not in ("REJECT", "ESCALATE"):
        return False
    return not any(field in execution
                   for field in MEASURED_EXECUTION_FIELDS)


def collect_no_execution_proof(journal) -> dict:
    """Fold the §9.5 no-execution proof from a journal: the
    ``queue_claim`` row count (must be 0 — nothing was ever
    claimed for execution), every ``tool_result`` row's
    disposition, and the seqs of any execution-shaped rows
    (must be empty). A caller asserts on the returned dict; the
    proof is data, recomputed from the journal alone."""
    claims = 0
    dispositions = []
    executed_rows = []
    for row in journal.backend.find("experiences") or []:
        record = row.get("record") or {}
        kind = row.get("kind") or record.get("kind")
        if kind == "queue_claim":
            claims += 1
        elif kind == "tool_result":
            ok = is_non_execution_disposition(record)
            execution = record.get("execution") or {}
            dispositions.append({
                "seq": row.get("seq"),
                "gate_verdict": record.get("gate_verdict")
                or execution.get("verdict"),
                "status": execution.get("status"),
                "non_execution": ok,
            })
            if not ok:
                executed_rows.append(row.get("seq"))
    return {
        "queue_claim_rows": claims,
        "tool_result_rows": len(dispositions),
        "tool_result_dispositions": dispositions,
        "executed_outcome_rows": executed_rows,
        "no_execution_proven": claims == 0 and not executed_rows,
    }


# ---------------------------------------------------------------------------
# §4 — scored documents and the corpus shape
# ---------------------------------------------------------------------------

def scored_documents(documents: list) -> list:
    """The scored documents of §4.2: entered documents at cycle
    positions 1..K whose capture perception carried a non-empty
    ``prior_cycles``. Cycle-0 documents, scored-position
    documents with an empty prior fold, duplicates (never
    entered), and transport failures (never documents) are all
    absent — unscored, in no denominator."""
    return [d for d in documents
            if d.get("cycle", 0) >= 1
            and (d.get("perception") or {}).get(
                "prior_cycles_count", 0) > 0]


def build_multicycle_corpus(documents: list, entries: list, *,
                              corpus_id: str, candidate_label: str,
                              capture_date: str,
                              fold: dict | None = None) -> dict:
    """Assemble the §4/§5.1 corpus dict from a driver's frozen
    documents + capture entries: items are exactly the scored
    documents, plus transport-failure placeholders at scored
    positions (a transport failure at cycle >= 1 whose attempt
    perception had a non-empty prior fold — recorded, unscored,
    ``transport_failed`` true, never in M). ``fixed_n`` is M,
    the scored-document count. Cycle-0 documents are preserved
    in the driver's documents list and are **not** items here.
    When ``fold`` (a :func:`freeze_multicycle` record) is
    supplied it rides on the corpus as ``corpus["fold"]`` —
    callers freeze first, then rebuild with the fold attached,
    exactly as SI-107's freeze did."""
    scored = scored_documents(documents)
    items = []
    for doc in scored:
        items.append({
            "item_id": doc["doc_id"],
            "workload": doc["workload"],
            "cycle": doc["cycle"],
            "run": f"{doc['workload']}-{doc['run']}",
            "body": doc["body"],
            "body_source": "verbatim-recorded",
            "prompt_ref": {
                "task_seq": doc["task_seq"],
                "request_digest": doc.get("request_digest"),
                "perception_digest":
                    (doc.get("perception") or {}).get(
                        "perception_digest"),
            },
            "parse_task_seq": doc["task_seq"],
            "provenance": {
                "capture": (
                    "multi-cycle capture driver (SI-109, "
                    "SI-106 §9); journaling between cycles "
                    "without execution (§9.5)"),
                "residency_seq": doc.get("residency_seq"),
                "residency_state": doc.get("residency_state"),
                "prior_intent_seqs":
                    (doc.get("perception") or {}).get(
                        "prior_intent_seqs"),
                "citable_seqs_at_capture":
                    (doc.get("perception") or {}).get(
                        "citable_seqs"),
            },
        })
    tf_count = 0
    for entry in entries:
        if entry.get("outcome") != "transport-failed":
            continue
        if entry.get("cycle", 0) < 1:
            continue
        if (entry.get("perception") or {}).get(
                "prior_cycles_count", 0) <= 0:
            continue
        tf_count += 1
        items.append({
            "item_id": f"{entry['item_id']}-tf{entry['attempt']}",
            "workload": entry["workload"],
            "cycle": entry["cycle"],
            "run": f"{entry['workload']}-{entry['run']}",
            "body": None,
            "transport_failed": True,
            "body_source": "none — transport failure, no response",
            "prompt_ref": {
                "task_seq": entry.get("task_seq"),
                "request_digest": entry.get("request_digest"),
            },
            "parse_task_seq": entry.get("task_seq"),
            "provenance": {
                "capture": (
                    "multi-cycle capture driver (SI-109, "
                    "SI-106 §9); transport failure at a scored "
                    "position — recorded, unscored"),
                "residency_seq": entry.get("residency_seq"),
                "detail": entry.get("detail"),
            },
        })
    workloads = []
    for doc in documents:
        if doc["workload"] not in workloads:
            workloads.append(doc["workload"])
    cycles = sorted({d["cycle"] for d in scored})
    corpus = {
        "corpus_id": corpus_id,
        "purpose": "candidate-readiness",
        "diagnostic_label": (
            "C2 multi-cycle DIAGNOSTIC corpus (SI-106): scored "
            "items are cycle-1..K documents with non-empty "
            "prior_cycles at capture; cycle-0 documents are "
            "preserved in the capture documents and are not "
            "items of this corpus; no bar is applied to this "
            "corpus (SI-106 §3)"),
        "candidate_label": candidate_label,
        "response_origin": "candidate-recorded",
        "capture_date": capture_date,
        "fixed_n": len(scored),
        "coverage": {
            "workloads": workloads,
            "cycles_covered": cycles,
            "signals_elicited": True,
        },
        "items": items,
    }
    if fold is not None:
        corpus["fold"] = fold
    return corpus


# ---------------------------------------------------------------------------
# §9.6 / §5.1 — freeze: corpus + fold together, both hashed
# ---------------------------------------------------------------------------

def fold_rows_and_hash(journal) -> tuple:
    """``(rows, fold_sha256, counts, tip)`` for a journal: every
    collection's rows as ``{collection, row}``, sorted by
    ``(collection, seq)``, hashed as ``sha256:`` + sha256 of
    their :func:`si.bundle.canonical_json` form — the exact
    procedure SI-107's freeze used, now in-repo. Any reviewer
    can recompute it from the frozen bundle alone."""
    rows = []
    counts = {}
    for coll in COLLECTIONS:
        found = journal.backend.find(coll) or []
        counts[coll] = len(found)
        for row in found:
            rows.append({"collection": coll, "row": row})
    rows.sort(key=lambda r: (r["collection"],
                             r["row"].get("seq") or 0))
    tip = journal.tip()[0]
    digest = "sha256:" + sha256_hex(canonical_json(rows))
    return rows, digest, counts, tip


def corpus_canonical_sha256(corpus: dict) -> str:
    """``sha256:`` + sha256 of the corpus dict's canonical JSON
    form — reproducible by independent recomputation over the
    corpus value alone."""
    return "sha256:" + sha256_hex(canonical_json(corpus))


def freeze_multicycle(bundle_dir: str, corpus: dict, out_dir: str,
                      *, corpus_filename: str =
                      "multicycle-corpus.json") -> dict:
    """Freeze a capture (§9.6): copy the capture bundle to
    ``<out_dir>/capture-bundle``, write the corpus beside it,
    and return the freeze record carrying **both** hashes —
    the corpus file's raw-byte sha256 and canonical sha256,
    and the fold sha256 computed identically over the live
    journal and the frozen copy (a mismatch refuses the
    freeze). The frozen copies are the only inputs a later
    scoring step may read. Freezing twice into the same
    directory is refused: a freeze is a one-time act."""
    dst_bundle = os.path.join(out_dir, "capture-bundle")
    if os.path.exists(dst_bundle):
        raise RuntimeError(
            f"frozen bundle already exists at {dst_bundle}; "
            "freeze is a one-time act")
    os.makedirs(out_dir, exist_ok=True)
    journal = open_journal(bundle_dir)
    try:
        _rows, fold_hash, counts, tip = fold_rows_and_hash(journal)
    finally:
        journal.close()
    shutil.copytree(bundle_dir, dst_bundle)
    frozen = open_journal(dst_bundle)
    try:
        rows2, fold_hash2, _c, tip2 = fold_rows_and_hash(frozen)
    finally:
        frozen.close()
    if fold_hash2 != fold_hash or tip2 != tip:
        raise RuntimeError(
            f"fold hash mismatch: frozen {fold_hash2} != live "
            f"{fold_hash}; freeze is not faithful")
    corpus_path = os.path.join(out_dir, corpus_filename)
    with open(corpus_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(corpus, indent=2) + "\n")
    record = {
        "corpus_id": corpus.get("corpus_id"),
        "corpus_file": corpus_path,
        "corpus_file_sha256": _file_sha256(corpus_path),
        "corpus_sha256": corpus_canonical_sha256(corpus),
        "fold_sha256": fold_hash,
        "fold_bundle": dst_bundle,
        "fold_tip_seq": tip,
        "fold_row_counts": counts,
        "fold_rows_total": len(rows2),
        "fold_hash_procedure": (
            "sha256 over si.bundle.canonical_json of the frozen "
            "journal's rows across all COLLECTIONS, each as "
            "{collection, row}, sorted by (collection, seq), "
            "exactly as frozen"),
        "m_scored_documents": corpus.get("fixed_n"),
    }
    with open(os.path.join(out_dir, "freeze-record.json"), "w",
              encoding="utf-8") as fh:
        fh.write(json.dumps(record, indent=2) + "\n")
    return record


def verify_freeze(record: dict, corpus: dict,
                  frozen_bundle_dir: str) -> bool:
    """Independently recompute a freeze record's hashes: the
    corpus canonical sha256 from the corpus value, and the
    fold sha256 from the frozen bundle's journal. True iff
    both reproduce the record exactly."""
    if corpus_canonical_sha256(corpus) != record.get(
            "corpus_sha256"):
        return False
    journal = open_journal(frozen_bundle_dir)
    try:
        _rows, digest, _counts, tip = fold_rows_and_hash(journal)
    finally:
        journal.close()
    return digest == record.get("fold_sha256") \
        and tip == record.get("fold_tip_seq")


# ---------------------------------------------------------------------------
# §9.2 — the residency-guard interface
# ---------------------------------------------------------------------------

def _poll_residency(residency_guard, journal) -> tuple:
    """One guard poll -> ``(state, residency_seq)``.

    The guard interface (§9.2), deliberately small so a live
    client, a replay stub, or a test fake all fit it. The guard
    may be:

    - a callable taking the journal and returning a state
      string, a residency record dict (``{"state": ...}``), or
      a journaled-poll result (``{"record": {"state": ...},
      "seq": ...}``, the live client's shape);
    - an object with a ``poll(journal)`` method returning any
      of the above;
    - an object with a ``journal_residency(journal)`` method
      returning the journaled-poll shape.

    Any exception propagates to the driver, which prices it as
    a guard halt — a poll that failed never spends an attempt.
    """
    poll = None
    if hasattr(residency_guard, "journal_residency"):
        poll = residency_guard.journal_residency
    elif hasattr(residency_guard, "poll"):
        poll = residency_guard.poll
    elif callable(residency_guard):
        poll = residency_guard
    if poll is None:
        raise TypeError(
            "residency guard must be callable or expose poll() / "
            "journal_residency() (SI-106 §9.2 interface)")
    result = poll(journal)
    if isinstance(result, str):
        return result, None
    if isinstance(result, dict):
        record = result.get("record")
        if isinstance(record, dict) and record.get("state"):
            return record["state"], result.get("seq")
        if result.get("state"):
            return result["state"], result.get("seq")
    raise ValueError(
        f"residency guard returned an unpriceable poll result: "
        f"{result!r}")


# ---------------------------------------------------------------------------
# The driver (§9)
# ---------------------------------------------------------------------------

class MultiCycleDriver:
    """The multi-cycle capture driver (SI-106 §9).

    Constructed over an already-open capture ``journal`` and
    its ``gate``, with everything §9 fixes in advance: the
    cycle shape ``K``, the per-workload-per-cycle-position
    attempt budget, and the per-task journaling cycle cap.
    ``proposal_source`` is a callable taking one assembled
    perception dict and returning the proposal document text
    (a gateway envelope body is unwrapped with the live
    proposer's own extraction rule); raising from it is a
    transport failure — recorded, unscored, spent.
    ``residency_guard`` fits :func:`_poll_residency`'s
    interface. ``on_progress(state)`` fires after every attempt
    and every journaled member with a deep copy of the driver
    state, so a caller persists each step as it lands; passing
    a prior ``state`` resumes from it (budgets already spent
    stay spent — resume never refunds an attempt).

    The driver holds no live authorisation: both collaborators
    are injected, and this slice exercises it only with replay
    / scripted sources on fixture bundles.
    """

    def __init__(self, journal, gate, *, bundle_dir: str,
                 work_root: str, budgets: dict, cycle_cap: int,
                 proposal_source, residency_guard,
                 K: int = 2,
                 max_attempts_per_workload_cycle: int = 10,
                 operator: str = "si-109-driver",
                 on_progress=None, state: dict | None = None,
                 candidate_registration=None,
                 served_template_probe=None):
        if K < 1:
            raise ValueError("K must be >= 1 (SI-106 §4.1)")
        if max_attempts_per_workload_cycle < 1:
            raise ValueError(
                "the attempt budget must be >= 1 (SI-106 §9.3)")
        self.journal = journal
        self.gate = gate
        self.bundle_dir = bundle_dir
        self.work_root = work_root
        self.budgets = dict(budgets)
        self.cycle_cap = int(cycle_cap)
        self.proposal_source = proposal_source
        self.residency_guard = residency_guard
        self.K = int(K)
        # §9.3: fixed at construction, read at every attempt,
        # written by no method in this module.
        self._max_attempts = int(max_attempts_per_workload_cycle)
        self.operator = operator
        self.on_progress = on_progress
        # Pre-flight template lock (SI-112; SI-111 §3.3,
        # D-111-2): when a candidate registration is supplied,
        # run() pre-flights before the first task and every
        # guard poll re-verifies the served hash through
        # ``served_template_probe`` (a zero-arg callable
        # returning the currently served template's sha256,
        # or None when it cannot be resolved). The driver
        # still contacts no model by itself: the probe is the
        # caller's serving-side evidence, exactly as the
        # residency guard is.
        self.candidate_registration = candidate_registration
        self.served_template_probe = served_template_probe
        if state is not None:
            self.state = copy.deepcopy(state)
        else:
            self.state = {
                "documents": [], "entries": [], "attempts": {},
                "runs": {}, "closed": False, "halted": None,
            }
        if candidate_registration is not None \
                and "preflight" not in self.state:
            self.state["preflight"] = None
        self._seen_hashes = {
            _hash_text(d["body"]): d["doc_id"]
            for d in self.state["documents"]}

    @property
    def max_attempts_per_workload_cycle(self) -> int:
        """The pre-stated §9.3 budget (read-only by design)."""
        return self._max_attempts

    def _persist(self) -> None:
        if self.on_progress is not None:
            self.on_progress(copy.deepcopy(self.state))

    def _guard_poll(self, run_key: str, cycle: int) -> tuple:
        """Poll the guard for one attempt (§9.2). On any
        non-Resident state or poll failure: persist state, mark
        the halt in it, and raise :class:`ResidencyGuardHalt` —
        no attempt is spent and the driver never retries around
        the halt (a caller-supervised restart resumes from the
        persisted state only after the node is restored)."""
        try:
            state, seq = _poll_residency(
                self.residency_guard, self.journal)
        except ResidencyGuardHalt:
            raise
        except Exception as exc:
            self.state["halted"] = {
                "run": run_key, "cycle": cycle,
                "reason": "residency poll failed",
                "detail": f"{type(exc).__name__}: {exc}",
            }
            self._persist()
            raise ResidencyGuardHalt(
                f"residency guard halt — run {run_key!r} cycle "
                f"{cycle}: residency poll failed "
                f"({type(exc).__name__}: {exc}); no attempt spent"
            ) from exc
        if state != "Resident":
            self.state["halted"] = {
                "run": run_key, "cycle": cycle,
                "reason": "residency not Resident",
                "residency_state": state, "residency_seq": seq,
            }
            self._persist()
            raise ResidencyGuardHalt(
                f"residency guard halt — run {run_key!r} cycle "
                f"{cycle}: residency state is {state!r}, not "
                "'Resident'; no attempt spent")
        # Template re-verification (SI-112; SI-111 §3.3(c)):
        # the guard poll also carries the served template
        # hash as a checked field. A mid-capture restart that
        # changes it — or a probe that stops resolving —
        # halts here, no attempt spent, under the guard's own
        # discipline.
        if self.candidate_registration is not None \
                and self.served_template_probe is not None:
            record = self.state.get("preflight") or {}
            try:
                now_hash = self.served_template_probe()
            except Exception as exc:
                self.state["halted"] = {
                    "run": run_key, "cycle": cycle,
                    "reason": "served template probe failed",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
                self._persist()
                raise ResidencyGuardHalt(
                    f"template guard halt — run {run_key!r} "
                    f"cycle {cycle}: served template probe "
                    f"failed ({type(exc).__name__}: {exc}); "
                    f"no attempt spent") from exc
            matches = now_hash is not None and now_hash == \
                record.get("registered_template_sha256")
            record.setdefault("rechecks", []).append({
                "run": run_key, "cycle": cycle,
                "served_template_sha256": now_hash,
                "decision": "match" if matches else "MISMATCH"})
            self.state["preflight"] = record
            if not matches:
                self.state["halted"] = {
                    "run": run_key, "cycle": cycle,
                    "reason": "served template changed "
                              "mid-capture",
                    "served_template_sha256": now_hash,
                    "registered_template_sha256":
                        record.get("registered_template_sha256"),
                }
                self._persist()
                raise ResidencyGuardHalt(
                    f"template guard halt — run {run_key!r} "
                    f"cycle {cycle}: served template hash is "
                    f"{now_hash!r}, not the registered "
                    f"{record.get('registered_template_sha256')!r}; "
                    f"no attempt spent (SI-111 §3.3(c))")
            self._persist()
        return state, seq

    def _preflight(self) -> None:
        """The capture pre-flight (SI-112; SI-111 §3.3(b)).
        Runs before the first task is journaled: the served
        hash comes from the caller's probe, the comparison
        and decision are recorded in the driver state, and a
        refusal raises :class:`PreflightRefused` with the
        state already persisted — no task, no attempt. A
        resumed state already carrying a PASS record is not
        re-ruled (its guard polls still re-verify)."""
        if self.candidate_registration is None:
            return
        existing = self.state.get("preflight")
        if existing and existing.get("decision") == "PASS":
            return
        served_hash = None
        if self.served_template_probe is not None:
            served_hash = self.served_template_probe()
        record = check_preflight(
            self.candidate_registration,
            served_template_sha256=served_hash,
            served_source_label="template-probe")
        self.state["preflight"] = record
        self._persist()
        if record["decision"] != "PASS":
            raise PreflightRefused(record)

    def run(self, tasks: list) -> dict:
        """Run the capture plan: ``tasks`` is a list of
        ``{"workload", "instruction", "run"}`` dicts, executed
        in order. Returns the driver state. Raises
        :class:`ResidencyGuardHalt` on a §9.2 halt (state
        already persisted via ``on_progress`` and on
        ``self.state``); every other ending — completion or
        budget exhaustion — is recorded per run in the state."""
        from si.gaia.conformance_capture import extract_content
        journal = self.journal
        self._preflight()
        for spec in tasks:
            label = spec["workload"]
            run_idx = spec["run"]
            run_key = f"{label}#{run_idx}"
            run = self.state["runs"].get(run_key)
            if run is not None and run.get("ended"):
                continue
            if run is None:
                task = record_task_entry(
                    journal, spec["instruction"], self.work_root,
                    budgets=self.budgets, operator=self.operator)
                run = {
                    "workload": label, "run": run_idx,
                    "task_seq": task["seq"], "task_ref": task["ref"],
                    "cycles_used": 0, "intent_seqs": [],
                    "cycles": {}, "ended": None,
                }
                self.state["runs"][run_key] = run
                self._persist()
            task = {"seq": run["task_seq"], "ref": run["task_ref"]}
            ctx = ToolContext(bundle_dir=self.bundle_dir)
            ctx.tool_work_root = self.work_root
            ctx.tool_task_seq = run["task_seq"]
            intent_seqs = set(run["intent_seqs"])

            for cycle in range(self.K + 1):
                cyc_key = str(cycle)
                cyc_state = run["cycles"].get(cyc_key) or {}
                doc = None
                if cyc_state.get("doc_id"):
                    doc = next(
                        d for d in self.state["documents"]
                        if d["doc_id"] == cyc_state["doc_id"])
                if doc is not None and doc.get("journaled"):
                    continue
                if doc is None:
                    att_key = f"{label}@{cycle}"
                    while True:
                        used = self.state["attempts"].get(
                            att_key, 0)
                        if used >= self._max_attempts:
                            run["ended"] = (
                                f"attempt budget exhausted at "
                                f"cycle {cycle} ({att_key} = "
                                f"{used})")
                            self._persist()
                            break
                        # §9.2: poll BEFORE the attempt is spent.
                        residency_state, residency_seq = \
                            self._guard_poll(run_key, cycle)
                        self.state["attempts"][att_key] = used + 1
                        attempt_no = used + 1
                        doc_id = (f"c2-{label}-r{run_idx}"
                                  f"-c{cycle}")
                        perception = assemble_perception(
                            journal, task["seq"])
                        sections = perception.get("sections") or {}
                        prior = sections.get("prior_cycles") or []
                        citable = sections.get(
                            "citable_reference_set") or {}
                        perc_meta = {
                            "prior_cycles_count": len(prior),
                            "prior_intent_seqs": [
                                c.get("intent_seq") for c in prior],
                            "citable_seqs": citable.get("seqs"),
                            "perception_digest":
                                perception.get("digest"),
                            "perception_bytes":
                                perception.get("bytes"),
                            "truncated": perception.get(
                                "truncated"),
                        }
                        request_digest = _hash_text(
                            canonical_json(perc_meta).decode(
                                "utf-8"))
                        entry = {
                            "item_id": doc_id, "workload": label,
                            "run": run_idx, "cycle": cycle,
                            "attempt": attempt_no,
                            "attempt_key": att_key,
                            "residency_seq": residency_seq,
                            "residency_state": residency_state,
                            "task_seq": task["seq"],
                            "perception": perc_meta,
                            "request_digest": request_digest,
                        }
                        try:
                            raw = self.proposal_source(perception)
                        except Exception as exc:
                            entry["outcome"] = "transport-failed"
                            entry["detail"] = (
                                f"{type(exc).__name__}: {exc}")
                            self.state["entries"].append(entry)
                            self._persist()
                            continue
                        content = extract_content(
                            raw if isinstance(raw, str)
                            else json.dumps(raw))
                        digest = _hash_text(content)
                        entry["response_hash"] = digest
                        if digest in self._seen_hashes:
                            entry["outcome"] = "duplicate-capture"
                            entry["duplicate_of"] = \
                                self._seen_hashes[digest]
                            self.state["entries"].append(entry)
                            self._persist()
                            continue
                        self._seen_hashes[digest] = doc_id
                        entry["outcome"] = "captured"
                        self.state["entries"].append(entry)
                        doc = {
                            "doc_id": doc_id, "workload": label,
                            "run": run_idx, "cycle": cycle,
                            "task_seq": task["seq"],
                            "task_ref": task["ref"],
                            "body": content, "body_hash": digest,
                            "perception": perc_meta,
                            "request_digest": request_digest,
                            "residency_seq": residency_seq,
                            "residency_state": residency_state,
                            "journaled": False, "journaling": None,
                        }
                        self.state["documents"].append(doc)
                        run["cycles"][cyc_key] = {"doc_id": doc_id}
                        self._persist()
                        break
                if run.get("ended"):
                    break
                if doc is None:
                    break
                # §9.5: journaling between cycles, no execution.
                if not doc.get("journaled"):
                    prior_journaling = doc.get("journaling")
                    resume = None
                    if prior_journaling and \
                            prior_journaling.get(
                                "parse_outcome") == "ok":
                        resume = {
                            "members_done": prior_journaling.get(
                                "members_done", 0),
                            "seq_by_position": prior_journaling.get(
                                "seq_by_position", {}),
                            "refused_positions":
                                prior_journaling.get(
                                    "refused_positions", []),
                            "members": prior_journaling.get(
                                "members", []),
                            "refused": prior_journaling.get(
                                "refused", []),
                            "journaled_intent_seqs":
                                prior_journaling.get(
                                    "journaled_intent_seqs", []),
                            "batch_seq": prior_journaling.get(
                                "batch_seq"),
                        }

                    def _progress(outcome, _doc=doc):
                        _doc["journaling"] = outcome
                        self._persist()

                    out_j, new_cycles_used = journal_cycle(
                        journal, self.gate, ctx, task, doc["body"],
                        cycles_used=run["cycles_used"],
                        cycle_cap=self.cycle_cap,
                        task_intent_seqs=intent_seqs,
                        resume=resume, on_member=_progress)
                    run["cycles_used"] = new_cycles_used
                    run["intent_seqs"] = sorted(intent_seqs)
                    doc["journaling"] = out_j
                    doc["journaled"] = True
                    self._persist()
            if not run.get("ended"):
                run["ended"] = "completed"
                self._persist()
        self.state["closed"] = all(
            r.get("ended") for r in self.state["runs"].values()) \
            and len(self.state["runs"]) == len(tasks)
        self._persist()
        return self.state


# ---------------------------------------------------------------------------
# §9.4 — the smoke gate, fixture scope
# ---------------------------------------------------------------------------

class SmokeGateError(RuntimeError):
    """The fixture smoke gate failed its proof: no capture
    budget may be spent on the strength of it (§9.4)."""


def run_fixture_smoke_gate(bundle_dir: str, work_root: str, *,
                           bodies: list, residency_guard,
                           policy: dict | None = None,
                           instruction: str =
                           "list the work root, then read a file",
                           candidate_registration=None,
                           served_template_probe=None,
                           ) -> dict:
    """The §9.4 smoke gate, in fixture scope only.

    ``bodies`` is the scripted proposal source's queue: one
    body for cycle 0 and one for cycle 1 of a single dummy
    task run (K = 1) on the caller-supplied fixture bundle.
    The gate proves, end to end and from the journal alone:

    (a) the guard interface polls Resident before the run;
    (b) cycle 0 is journaled between cycles without execution;
    (c) the cycle-1 perception is rebuilt with non-empty
    ``prior_cycles`` and a non-empty citable set;
    (d) the §5.1 fold assembles from the fixture journal; and
    (e) the §9.5 no-execution proof holds (zero queue claims,
    every tool_result a non-execution disposition).

    Returns the gate summary. Raises :class:`SmokeGateError`
    naming the failed check — a gate that has not passed
    spends no budget (§9.4). No live model is contacted: the
    served-identity leg of the live gate (§9.4 part (a) in
    SI-107) is priced by a live authorisation's own pre-flight,
    not by this fixture proof, and nothing here claims it.
    """
    from si.gaia.policy import resolve_tool_policy
    from si.gate import Gate
    if policy is None:
        from si.gaia.live import fixture_policy
        policy = fixture_policy(work_root)
    tool_pol = resolve_tool_policy(policy)
    budgets = {
        "max_tool_cycles_per_task": int(
            tool_pol["max_tool_cycles_per_task"]),
        "tool_max_task_result_bytes": int(
            tool_pol["tool_max_task_result_bytes"]),
    }
    queue = list(bodies)

    def source(perception):
        if not queue:
            raise RuntimeError(
                "smoke-gate scripted source exhausted")
        return queue.pop(0)

    journal = open_journal(bundle_dir)
    try:
        driver = MultiCycleDriver(
            journal, Gate(policy), bundle_dir=bundle_dir,
            work_root=work_root, budgets=budgets,
            cycle_cap=budgets["max_tool_cycles_per_task"],
            proposal_source=source,
            residency_guard=residency_guard, K=1,
            operator="si-109-smoke-gate",
            candidate_registration=candidate_registration,
            served_template_probe=served_template_probe)
        try:
            state = driver.run([{
                "workload": "smoke-dummy", "run": 0,
                "instruction": instruction}])
        except PreflightRefused as exc:
            raise SmokeGateError(
                f"smoke gate pre-flight check failed: {exc}"
            ) from exc
        except ResidencyGuardHalt as exc:
            raise SmokeGateError(
                f"smoke gate residency check failed: {exc}"
            ) from exc
        docs = state["documents"]
        cycle1 = [d for d in docs if d["cycle"] == 1]
        checks = {
            "guard_polled_resident": all(
                e.get("residency_state") == "Resident"
                for e in state["entries"]),
            "cycle0_journaled": any(
                d["cycle"] == 0 and d.get("journaled")
                for d in docs),
            "cycle1_prior_nonempty": bool(cycle1) and all(
                (d.get("perception") or {}).get(
                    "prior_cycles_count", 0) > 0
                for d in cycle1),
            "cycle1_citable_nonempty": bool(cycle1) and all(
                bool((d.get("perception") or {}).get(
                    "citable_seqs"))
                for d in cycle1),
        }
        rows, fold_hash, counts, tip = fold_rows_and_hash(journal)
        proof = collect_no_execution_proof(journal)
        checks["fold_assembled"] = len(rows) > 0
        checks["no_execution"] = proof["no_execution_proven"]
        failed = [k for k, ok in checks.items() if not ok]
        summary = {
            "gate": "SI-109 fixture smoke gate (SI-106 §9.4, "
                    "fixture scope)",
            "checks": checks,
            "passed": not failed,
            "fold": {"rows": len(rows), "sha256": fold_hash,
                     "tip_seq": tip, "row_counts": counts},
            "no_execution_proof": proof,
            "documents": len(docs),
        }
        if state.get("preflight") is not None:
            summary["preflight"] = state["preflight"]
        if failed:
            raise SmokeGateError(
                f"smoke gate failed check(s): {failed}; "
                f"summary: {summary}")
        return summary
    finally:
        journal.close()
