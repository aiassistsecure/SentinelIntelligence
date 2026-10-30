"""Gaia seam ingest: the SI side of G-9 Flow 1 — fork watcher
proposals gated and journaled under a per-model seam task
(G-9 §8.3, §10.2, §12.1; slice S-2a).

The gaia fork's residency watcher proposes actuations (for
example ``hearth.restart`` for a model it observed lost). Those
proposals cross to SI over the seam as one JSON object (§12.1)::

    {"model": ..., "tool": ..., "args": {...}, "reason": ...,
     "caused_by": {"hash": ..., "seq": ...}}

``caused_by`` is the fork's residency-spine event the watcher
observed (a ``hash`` + spine ``seq`` pair, carried verbatim by
the fork's egress client). It is a *spine* seq, never an SI
journal seq — the two sequence spaces are never merged
(G-9 §2.2), and the pair is the proposal's dedup identity. This module is the ingest core behind the
service's ``POST /gaia/proposals`` surface (the HTTP adapter
lives in :mod:`si.gaia.service`); :func:`ingest_proposal` is
the whole flow:

1. **Validate (§8.3.1).** :func:`validate_proposal` checks the
   wire shape and itemises every problem. A malformed proposal
   is refused and journals **nothing** — validation completes
   before the bundle or its journal is opened at all. Per
   §8.2, ``args.spine_event`` duplicates ``caused_by`` in
   watcher proposals by construction, so a ``spine_event``
   that disagrees with ``caused_by`` (hash or seq) is a
   validation refusal, never a priced gate question.
2. **Dedup (§8.3.2).** The journal is scanned for a self-edit
   whose journaled intent dict carries a ``caused_by`` equal
   to this proposal's pair. A hit returns the *original* ACK
   unchanged and writes nothing — no task, no edit, no
   checkpoint: a re-delivered proposal is a no-op, because
   ``caused_by`` is the proposal's identity.
3. **Seam task (§10.2).** One seam task per model: the task
   entry for this proposal's model is found
   (:func:`find_seam_task`) or created
   (:func:`si.gaia.task.record_task_entry`, operator
   :data:`SEAM_OPERATOR`, submitter :data:`SEAM_SUBMITTER`,
   submitted via :data:`SEAM_SUBMITTED_VIA`, budgets from the
   bundle policy's cycle / byte caps), and its instruction is
   the deterministic :func:`seam_task_instruction` — discovery
   is equality on that construction, never parsing.
4. **Gate + journal (§8.3, §12.1).** An
   ``Intent(kind="tool", ...)`` is built from the proposal,
   chained to the seam task (``parents=[task_ref]``,
   ``source_seqs=[task_seq]``), and gated exactly as the
   queue-drain admission helper
   (:func:`si.gaia.drain.admit_tool_intent`) gates its
   intents — the same gate evaluation, classification,
   self-edit, provenance edges, and per-verdict result
   marking, replicated branch-for-branch. The one seam
   difference: the journaled intent dict carries
   ``caused_by`` **inside** it (§12.1 — it is a journal field
   of the intent, which is what step 2's dedup scans), so
   this module cannot call ``admit_tool_intent`` itself (it
   journals ``intent.to_dict()`` verbatim, and ``Intent``
   carries no ``caused_by`` field). The self-edit is
   journaled **with its full gate report for every verdict**
   — ADMIT leaves the edit ``pending``; REJECT / ESCALATE
   mark it ``rejected`` / ``escalated`` and journal their
   non-execution outcome (the B-1 shadow record when the
   report's execution label is the shadow label, otherwise
   the live non-execution record). After fresh journaling the
   manifest checkpoint is re-anchored and rewritten, exactly
   as :func:`si.gaia.task.run_task`'s tail does.
5. **ACK (§12.1).** Only after journaling:
   ``{"task_seq", "journal_seq", "verdict"}`` where
   ``journal_seq`` is the self-edit row's journal sequence.

**Registry gap — CLOSED by SI-55/S-2b, stated plainly.**
:data:`REGISTRY_GAP_LABEL` now records the closed state:
SI-55/S-2b registered the fork's ``hearth.*`` tools
(``hearth.restart``, ``hearth.notify``) in SI's registry
(:mod:`si.gaia.registry`) as external-class, seam-crossing
tools, so a watcher proposal now gates on its merits instead
of REJECTing at check 2's ``tool_registered`` sub-check —
under the default ``external: escalate`` class rule it
ESCALATEs; under an ``external: admit`` policy it ADMITs and
rests ``pending``. Registration is existence only: this module
still does not map, rename, or special-case ``hearth.*`` names,
no local executor exists for them, and an ADMIT submits
nothing (S-3 unbuilt — see below).

**What is NOT built here.** An ADMIT at the seam submits
nothing anywhere: S-3 work submission back to the fork is
unbuilt, so an admitted proposal is a journaled, pending
self-edit and nothing more. The operator leash does not cross
the seam in this slice, and no outcome returns to the fork —
there is no callback, no egress, no second flow. Gate setup
and contexts mirror :func:`si.gaia.task.run_task` exactly
(miniature bundles gate through the live ``ProbeContext``;
other bundles through a non-live ``ToolContext``).

**No authentication, loopback only (G-9 §14).** Nothing here
checks a credential: the seam identities are asserted
provenance fields journaled verbatim, and the surface this
core backs is a loopback service. Performance is
**UNMEASURED**: no throughput or latency claim is made for
ingest anywhere in this slice. CPU only; no secrets.

Concurrency: the service is a ``ThreadingHTTPServer``, so the
whole journal section of an ingest (dedup scan → gate →
journal → re-anchor) is serialised under a module-level
re-entrant lock — two concurrent identical POSTs cannot both
pass the dedup scan. The journal is opened per ingest call
and closed in a ``finally``.
"""
from __future__ import annotations

import os
import threading

from si.bundle import (
    _anchor_checkpoint,
    _write_manifest,
    load_bundle,
    load_policy,
)
from si.gaia.context import ToolContext
from si.gaia.executor import record_non_execution_outcome
from si.gaia.policy import resolve_tool_policy
from si.gaia.shadow import record_shadow_outcome
from si.gaia.task import record_task_entry
from si.gate import TOOL_SHADOW_EXECUTION, Gate, Intent
from si.journal import open_journal
from si.probes import ProbeContext, is_miniature_probe_bundle

#: Journaled identity of the seam task's operator (the seam
#: itself, not a human operator and not the watcher).
SEAM_OPERATOR = "gaia-seam"

#: Journaled submitter of every seam task: the gaia fork's
#: residency watcher, whose proposals this module ingests.
#: An asserted provenance field (see the module docstring),
#: not authentication.
SEAM_SUBMITTER = "gaia-fork-watcher"

#: Journaled ``submitted_via`` of every seam task, naming the
#: seam surface and the design / slice that built it.
SEAM_SUBMITTED_VIA = "gaia-seam-ingest (G-9/S-2a)"

#: Scope label carried on every successful ingest result:
#: what the seam is, what an ADMIT here does not yet do, and
#: the surface's honest limits.
SEAM_INGEST_LABEL: str = (
    "seam ingest is live SI-side (G-9 §8/§12.1): a fork watcher "
    "proposal is validated, deduplicated on caused_by, gated, "
    "and journaled under the model's seam task, and the ACK is "
    "returned only after journaling; S-3 work submission is "
    "UNBUILT — an ADMIT here submits nothing to the fork yet, "
    "it is a journaled pending self-edit and nothing more; "
    "there is no authentication — loopback only (G-9 §14); "
    "performance UNMEASURED")

#: The registry status, stated plainly (see the module
#: docstring): the S-2a gap is CLOSED by SI-55/S-2b — the
#: hearth.* names are registered (external-class, seam-crossing)
#: and watcher proposals gate on their merits. The constant
#: keeps its S-2a name for import compatibility; its text is
#: the current truth, and REGISTRY_STATUS_LABEL is the alias.
REGISTRY_GAP_LABEL: str = (
    "registry gap CLOSED by SI-55/S-2b: hearth.restart and "
    "hearth.notify are registered in SI's registry "
    "(si/gaia/registry.py) as external-class seam-crossing "
    "tools, so watcher proposals gate on their merits (default "
    "external policy ESCALATEs; an external:admit policy ADMITs "
    "to a journaled pending self-edit) — registration is "
    "existence only: no local executor exists for them, an "
    "ADMIT submits nothing (S-3 UNBUILT), and no hearth.* name "
    "is mapped, renamed, or special-cased at the seam")

#: Alias with the accurate name (SI-55/S-2b).
REGISTRY_STATUS_LABEL: str = REGISTRY_GAP_LABEL

#: Label on a refused (validation-failed) ingest: the refusal
#: is total — nothing was journaled, by construction.
_REFUSED_LABEL = (
    "a refused proposal journals nothing — validation precedes "
    "any journal write (G-9 §8.3.1)")

#: The exact wire keys of a §12.1 proposal / caused_by pair.
_PROPOSAL_KEYS = frozenset({"model", "tool", "args", "reason",
                            "caused_by"})
_CAUSED_BY_KEYS = frozenset({"hash", "seq"})

#: Serialises the journal section of every ingest in this
#: process (dedup scan through re-anchor), so concurrent
#: identical proposals cannot both pass the dedup scan.
_INGEST_LOCK = threading.RLock()


# -- validation (§8.3.1, §8.2) ----------------------------------------------

def validate_proposal(payload) -> tuple[dict | None, list[str]]:
    """Validate one §12.1 proposal's wire shape, fail closed.

    Itemises **every** problem found (never just the first) so
    the caller can refuse with the full list. Returns
    ``(normalised, [])`` on success — exactly
    ``{"model", "tool", "args", "reason",
    "caused_by": {"hash", "seq"}}`` — or ``(None, problems)``
    on any problem. Pure: touches no bundle, no journal, no
    filesystem.
    """
    if not isinstance(payload, dict):
        return None, [
            f"payload must be a JSON object (dict), got "
            f"{type(payload).__name__}"]
    problems: list[str] = []

    for key in payload:
        if key not in _PROPOSAL_KEYS:
            problems.append(
                f"unknown key {key!r}: proposal keys are exactly "
                f"{sorted(_PROPOSAL_KEYS)}")
    for key in sorted(_PROPOSAL_KEYS):
        if key not in payload:
            problems.append(f"missing required key {key!r}")

    model = payload.get("model")
    if "model" in payload and not (
            isinstance(model, str) and model):
        problems.append(
            f"model must be a non-empty string, got {model!r}")
    tool = payload.get("tool")
    if "tool" in payload and not (
            isinstance(tool, str) and tool):
        problems.append(
            f"tool must be a non-empty string, got {tool!r}")

    args = payload.get("args")
    if "args" in payload and not isinstance(args, dict):
        problems.append(
            f"args must be a JSON object (dict), got "
            f"{type(args).__name__}")

    reason = payload.get("reason")
    if "reason" in payload and not isinstance(reason, str):
        problems.append(
            f"reason must be a string (empty is allowed — the "
            f"gate prices content), got {type(reason).__name__}")

    caused_by = payload.get("caused_by")
    cb_hash = cb_seq = None
    if "caused_by" in payload:
        if not isinstance(caused_by, dict):
            problems.append(
                f"caused_by must be a JSON object (dict) with "
                f"keys exactly {sorted(_CAUSED_BY_KEYS)}, got "
                f"{type(caused_by).__name__}")
        else:
            for key in caused_by:
                if key not in _CAUSED_BY_KEYS:
                    problems.append(
                        f"caused_by: unknown key {key!r}: its "
                        f"keys are exactly {sorted(_CAUSED_BY_KEYS)}")
            for key in sorted(_CAUSED_BY_KEYS):
                if key not in caused_by:
                    problems.append(
                        f"caused_by: missing required key {key!r}")
            cb_hash = caused_by.get("hash")
            if "hash" in caused_by and not (
                    isinstance(cb_hash, str) and cb_hash):
                problems.append(
                    f"caused_by.hash must be a non-empty string, "
                    f"got {cb_hash!r}")
            cb_seq = caused_by.get("seq")
            if "seq" in caused_by and (
                    isinstance(cb_seq, bool)
                    or not isinstance(cb_seq, int) or cb_seq < 0):
                problems.append(
                    f"caused_by.seq must be an int >= 0 (a bool "
                    f"is not an int), got {cb_seq!r}")

    # §8.2: args.spine_event duplicates caused_by by
    # construction in watcher proposals; disagreement is a
    # validation refusal naming the disagreement.
    if isinstance(args, dict) and "spine_event" in args:
        spine = args["spine_event"]
        if not isinstance(spine, dict):
            problems.append(
                f"args.spine_event disagrees with caused_by: "
                f"spine_event must be an object carrying the "
                f"caused_by hash and seq, got "
                f"{type(spine).__name__}")
        elif isinstance(caused_by, dict) \
                and "hash" in caused_by and "seq" in caused_by:
            spine_seq = spine.get("seq")
            if spine.get("hash") != cb_hash \
                    or isinstance(spine_seq, bool) \
                    or spine_seq != cb_seq:
                problems.append(
                    f"args.spine_event disagrees with caused_by: "
                    f"spine_event hash/seq is "
                    f"({spine.get('hash')!r}, {spine_seq!r}) but "
                    f"caused_by is ({cb_hash!r}, {cb_seq!r}) "
                    f"(G-9 §8.2: the two must agree)")

    if problems:
        return None, problems
    normalised = {
        "model": model,
        "tool": tool,
        "args": args,
        "reason": reason,
        "caused_by": {"hash": cb_hash, "seq": cb_seq},
    }
    return normalised, []


# -- seam task (§10.2) --------------------------------------------------------

def seam_task_instruction(model: str) -> str:
    """The deterministic seam-task instruction for ``model``.

    Discovery is equality on this construction (no parsing):
    a task entry is ``model``'s seam task iff its journaled
    instruction is exactly this string (with the seam operator
    and submitted-via fields also matching — see
    :func:`find_seam_task`).
    """
    return (
        f"Gaia seam task for model {model!r} (G-9 §10.2: one "
        f"seam task per model — the fork watcher's proposals "
        f"for this model are gated and journaled under this "
        f"task)")


def find_seam_task(journal, model: str) -> dict | None:
    """``model``'s seam task as ``{"seq", "ref"}``, or None.

    Scans the journaled ``task`` experiences for a row whose
    record carries the seam operator, the seam submitted-via,
    and the deterministic seam instruction for ``model``. When
    (against construction) several rows match, the earliest
    journal sequence wins — there is one seam task per model.
    """
    instruction = seam_task_instruction(model)
    matches = []
    rows = journal.backend.find("experiences",
                                filter={"kind": "task"})
    for row in rows or []:
        record = row.get("record") or {}
        if not isinstance(record, dict):
            continue
        if record.get("operator") != SEAM_OPERATOR:
            continue
        if record.get("submitted_via") != SEAM_SUBMITTED_VIA:
            continue
        if record.get("instruction") != instruction:
            continue
        seq = row.get("seq")
        if isinstance(seq, int) and not isinstance(seq, bool):
            matches.append(seq)
    if not matches:
        return None
    seq = min(matches)
    return {"seq": seq, "ref": f"task@{seq}"}


# -- dedup (§8.3.2) -----------------------------------------------------------

def find_prior_proposal_ack(journal, caused_by: dict) -> dict | None:
    """The original ACK for ``caused_by``, or None.

    Scans the journaled self-edits for a row whose intent dict
    carries a ``caused_by`` equal to this pair — that row *is*
    the original ingest of this proposal. Returns
    ``{"task_seq", "journal_seq", "verdict", "edit_id"}``:
    ``journal_seq`` is the self-edit row's own journal
    sequence, ``verdict`` its journaled gate verdict, and
    ``task_seq`` the intent's first source sequence when that
    is an int (the seam task's sequence, by the ingest's
    chaining), else None.
    """
    if not isinstance(caused_by, dict):
        return None
    pair = {"hash": caused_by.get("hash"),
            "seq": caused_by.get("seq")}
    best = None
    rows = journal.backend.find("self_edits")
    for row in rows or []:
        intent = row.get("intent") or {}
        if not isinstance(intent, dict):
            continue
        if intent.get("caused_by") != pair:
            continue
        seq = row.get("seq")
        if not isinstance(seq, int) or isinstance(seq, bool):
            continue
        if best is not None and seq >= best["journal_seq"]:
            continue
        source_seqs = intent.get("source_seqs") or []
        task_seq = None
        if isinstance(source_seqs, list) and source_seqs:
            first = source_seqs[0]
            if isinstance(first, int) and not isinstance(first, bool):
                task_seq = first
        best = {"task_seq": task_seq, "journal_seq": seq,
                "verdict": row.get("gate_verdict"),
                "edit_id": row.get("edit_id")}
    return best


# -- ingest (§8.3) --------------------------------------------------------------

def ingest_proposal(bundle_dir: str, payload,
                    policy: dict | None = None) -> dict:
    """Ingest one fork watcher proposal (§8.3, §12.1).

    Validation runs first and opens nothing: on failure returns
    ``{"status": "refused", "problems": [...],
    "journaled": False, "label": ...}`` with the bundle and its
    journal untouched. On success the proposal is deduplicated,
    gated, and journaled under the model's seam task (see the
    module docstring for the full step mapping) and returns
    ``{"status": "acked", "ack": {"task_seq", "journal_seq",
    "verdict"}, "task_seq", "journal_seq", "verdict",
    "edit_id", "deduped", "gate_report", "label"}`` — for a
    deduplicated re-delivery, the original ingest's ACK,
    edit id, and journaled gate report, with ``deduped`` True
    and nothing newly written.
    """
    normalised, problems = validate_proposal(payload)
    if normalised is None:
        return {"status": "refused", "problems": problems,
                "journaled": False, "label": _REFUSED_LABEL}

    model = normalised["model"]
    caused_by = normalised["caused_by"]

    with _INGEST_LOCK:
        bundle_dir = os.path.abspath(bundle_dir)
        manifest = load_bundle(bundle_dir)
        pol = policy if policy is not None else load_policy(bundle_dir)
        gate = Gate(pol)
        tool_pol = resolve_tool_policy(pol)
        root = os.path.realpath(
            tool_pol.get("tool_work_root") or bundle_dir)
        journal = open_journal(bundle_dir, manifest)
        try:
            # §8.3.2 dedup: a prior ingest of this caused_by
            # returns its original ACK, writing nothing.
            prior = find_prior_proposal_ack(journal, caused_by)
            if prior is not None:
                gate_report = None
                rows = journal.backend.find(
                    "self_edits",
                    filter={"edit_id": prior["edit_id"]}, limit=1)
                if rows:
                    gate_report = rows[0].get("gate_report")
                ack = {"task_seq": prior["task_seq"],
                       "journal_seq": prior["journal_seq"],
                       "verdict": prior["verdict"]}
                return {"status": "acked", "ack": ack,
                        "task_seq": prior["task_seq"],
                        "journal_seq": prior["journal_seq"],
                        "verdict": prior["verdict"],
                        "edit_id": prior["edit_id"],
                        "deduped": True,
                        "gate_report": gate_report,
                        "label": SEAM_INGEST_LABEL}

            # §10.2: find-or-create the model's seam task.
            task = find_seam_task(journal, model)
            if task is None:
                budgets = {
                    "max_tool_cycles_per_task":
                        int(tool_pol["max_tool_cycles_per_task"]),
                    "tool_max_task_result_bytes":
                        int(tool_pol["tool_max_task_result_bytes"]),
                }
                task = record_task_entry(
                    journal, seam_task_instruction(model), root,
                    budgets=budgets, operator=SEAM_OPERATOR,
                    submitter=SEAM_SUBMITTER,
                    submitted_via=SEAM_SUBMITTED_VIA)
            task_seq = task["seq"]
            task_ref = task["ref"]

            # Gate context, mirroring run_task's setup exactly.
            miniature = is_miniature_probe_bundle(manifest)
            if miniature:
                ctx = ProbeContext(bundle_dir, manifest, policy=pol)
            else:
                ctx = ToolContext(bundle_dir=bundle_dir)
            ctx.tool_work_root = root
            ctx.tool_task_seq = task_seq

            intent = Intent(
                kind="tool", tool=normalised["tool"],
                args=normalised["args"],
                reason=normalised["reason"],
                parents=[task_ref], source_seqs=[task_seq])

            # Gate + journal, branch-for-branch the admission
            # mirror (admit_tool_intent), except the journaled
            # intent dict carries caused_by inside it (§12.1).
            report = gate.evaluate(intent, None, ctx)
            info = gate.classify_tool_intent(intent, ctx) \
                if hasattr(gate, "classify_tool_intent") else {}
            effective = info.get("effective_class") \
                if isinstance(info, dict) else None
            intent_dict = intent.to_dict()
            intent_dict["caused_by"] = {
                "hash": caused_by["hash"], "seq": caused_by["seq"]}
            edit_id = journal.record_self_edit(
                intent.kind, intent_dict, intent.reason,
                report.verdict, report.to_dict())
            if intent.parents:
                journal.record_provenance(
                    child=edit_id, parents=list(intent.parents),
                    relation="learned_from",
                    note=(intent.reason or "")[:200])
            if intent.source_seqs:
                journal.record_provenance(
                    child=edit_id, parents=list(intent.source_seqs),
                    relation="learned_from",
                    note="source journal sequences")

            if report.verdict == "ADMIT":
                journal.set_edit_result(edit_id, "pending")
            else:
                journal.set_edit_result(
                    edit_id,
                    "rejected" if report.verdict == "REJECT"
                    else "escalated")
                if report.execution == TOOL_SHADOW_EXECUTION:
                    record_shadow_outcome(
                        journal, intent, edit_id, report.verdict,
                        effective_class=effective)
                else:
                    record_non_execution_outcome(
                        journal, intent, edit_id, report.verdict,
                        report.execution or "not-evaluated",
                        effective_class=effective)

            rows = journal.backend.find(
                "self_edits", filter={"edit_id": edit_id}, limit=1)
            journal_seq = rows[0]["seq"] if rows else None

            # Re-anchor like run_task's tail, after fresh
            # journaling only (never on dedup or refusal).
            updated = _anchor_checkpoint(
                journal, manifest, manifest["head"],
                note=f"gaia seam ingest: proposal for model "
                     f"{model!r} tool {normalised['tool']!r} "
                     f"verdict {report.verdict} (edit {edit_id}, "
                     f"caused_by seq {caused_by['seq']})")
            _write_manifest(bundle_dir, updated)

            ack = {"task_seq": task_seq,
                   "journal_seq": journal_seq,
                   "verdict": report.verdict}
            return {"status": "acked", "ack": ack,
                    "task_seq": task_seq,
                    "journal_seq": journal_seq,
                    "verdict": report.verdict,
                    "edit_id": edit_id,
                    "deduped": False,
                    "gate_report": report.to_dict(),
                    "label": SEAM_INGEST_LABEL}
        finally:
            journal.close()
