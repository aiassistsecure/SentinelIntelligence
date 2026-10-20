"""``si admit-transformer-delta``: the operator-only admission path
for transformer deltas (SI-10; RFC SI-9 §4.4 / §5.5 follow-up).

THE GAP THIS CLOSES — AND THE ONE IT DOES NOT:

Under SI-9b the gate evaluates a transformer delta honestly:
identity/capability probes **cannot be evaluated** (this runtime has
no transformer probe harness — no forward pass, no probe suite, no
baseline), so a structurally valid delta ESCALATEs and the autonomous
loop halts. Until SI-10 an escalated delta had nowhere to go: no path
from ESCALATE to materialised existed at all.

This module is that path, and it is OPERATOR-ONLY by design and by
mechanism, following the SI-6 constitution-amendment precedent
(``si/amend.py``):

- The only caller is the operator-facing CLI subcommand
  ``si admit-transformer-delta``, which requires an explicit
  ``--operator`` name and a non-empty ``--reason`` (the admission
  reason). Missing either -> refused, nothing written, nothing
  journaled as success.
- The autonomous path is UNCHANGED. ``run_cycle`` still halts on
  ESCALATE and never calls this module; ``si run`` likewise. No
  flag, config key, or policy entry makes admission automatic —
  there is deliberately no such switch anywhere in this module.

WHAT ADMISSION IS (AND IS NOT), IN HONEST LABELS:

Admission re-runs the SI-9b gate, at admission time, against the
bundle as it stands NOW. Only a delta whose fresh verdict is
ESCALATE — structurally valid, with identity/capability reported
**not evaluated** — may be admitted. A REJECT-class delta can never
be admitted, by anyone, through this path: admission of a rejected
attempt is journaled as a refusal and the function raises.

The operator's admission is the operator **acting as the
identity/capability check**: a human looked at a delta whose probes
were not evaluated and decided, on the record, to let it land. It is
NOT a probe pass, and nothing in the journal, the manifest, or this
module's return value ever calls it one — the journaled gate report
keeps the identity/capability checks at passed=False,
measured=None, "not evaluated", verbatim.

FLOW (journal before materialise, append-only, existing semantics):

1. Fail closed: ``load_bundle`` must verify the bundle, and the
   candidate must be a transformer-delta candidate (SI-9b contract:
   ``{"record": ..., "blob_bytes": ...}``).
2. Re-validate: the SI-9b gate evaluates the intent + candidate
   against a manifest/stack context. For a non-miniature bundle
   that is the minimal context below, whose probe methods raise if
   touched — probes are NOT evaluated for such bundles, exactly as
   before. For a *miniature* bundle (SI-16) it is
   ``si.probes.ProbeContext``: the gate runs checks 3/4 for real
   in SHADOW MODE (verdict still ESCALATE), and the journaled gate
   report plus the ``operator_admission`` block's
   ``shadow_probe_results`` carry those measured results with the
   admission (RFC SI-13 §6.5) — the operator saw evidence, and the
   record says what it was. The basis below stays literally true
   in both cases: admission is the operator's decision, never the
   gate's ADMIT, and a shadow probe result is not an admission.
   REJECT -> the refusal is journaled (self-edit, verdict REJECT,
   result "rejected", operator attached) and anchored, exactly the
   way the loop journals rejections; then AdmitError is raised and
   nothing materialises. Any verdict other than ESCALATE is refused
   the same way — transformer deltas are never gate-ADMITted today,
   and if that ever changes, autonomous admission is a separate
   decision, not this path silently widening.
3. Journal the admission decision FIRST: a self-edit carrying the
   gate's ESCALATE report verbatim, the operator, the admission
   reason, and an ``operator_admission`` block on the intent dict
   (operator, reason, delta ids, timestamp, and the honest basis:
   the operator acting as the identity/capability check, probes not
   evaluated), plus an ``operator_admitted`` provenance record.
   Only then does anything materialise.
4. Materialise via the loop's existing semantics: the blob is
   written through ``si.loop._write_blob`` (content-addressed,
   skip-if-exists — SI never duplicates bytes), the manifest record
   is appended and the head advances, the delta is journaled, and
   the journal is checkpointed + re-anchored via
   ``si.loop._save_and_anchor``. No parallel bundle writer exists
   here. The materialised record gets a fresh runtime delta id
   (``si.loop._next_delta_id``) and ``parent`` = the head it lands
   on, exactly as the loop assigns demo-delta ids at materialise
   time; the candidate's proposed id is preserved in the journaled
   ``operator_admission`` block so the specific escalated artifact
   stays identifiable. ``self_edit_id`` on the record is the
   admission self-edit.
5. VERIFY: ``si.loop._verify_transformer_apply`` runs against the
   new head and its report is journaled under its existing honest
   label ``transformer-apply-smoke`` (a provenance record with that
   relation; the full report is the note). If VERIFY fails, the loop's
   existing rollback core (``si.loop._do_rollback``, cause
   ``verify-failed``) moves the head back to the pre-admission head,
   marks the delta and the admission self-edit ``rolled_back``, and
   journals the smoke report as the rollback's probe report — a
   failed-verify delta is never left silently active. The blob and
   the journal rows stay on disk as evidence, as rollback always
   leaves them.

Non-goals (unchanged by SI-10): no training, no delta authorship, no
inference, no probe harness, no GPU/cloud/spend. ``si serve`` still
returns 501 for transformer bundles, ``si export`` is unchanged, the
demo-delta path is unchanged, and the seed format is unchanged.

HONEST LIMIT — THE OPERATOR FLAG IS NOT AUTHENTICATION: exactly as
in SI-6, ``operator`` is an attribution / authorization *record* in
the journal, not cryptographic authentication. This local reference
runtime trusts its local operator; the journal records who claimed
to act and why.

CPU only, stdlib + the existing runtime only. This is machine
learning infrastructure, not "custom LLM software."
"""
from __future__ import annotations

import copy
import json
import os

from si.bundle import load_bundle, load_policy
from si.gate import Gate, Intent
from si.journal import open_journal
from si.transformer_delta import KIND as TRANSFORMER_DELTA_KIND

# Local module import (not `from si.loop import ...`): the VERIFY and
# rollback steps are looked up on the loop module at call time, so
# there is exactly one implementation of each and tests can patch
# ``si.loop._verify_transformer_apply`` to exercise the
# verify-failure path.
from si import loop as _loop


class AdmitError(ValueError):
    """An admission that refuses to proceed. Fail closed: nothing
    materialises, and a gate refusal leaves the head untouched."""


#: The honest basis recorded on every admission: what the operator is
#: standing in for, and what was NOT done. Kept as one constant so the
#: journal, the CLI output, and the docs cannot drift apart.
ADMISSION_BASIS = (
    "operator admission: the operator acted as the identity/capability "
    "check for this delta. Identity/capability probes were NOT "
    "evaluated — no transformer probe harness exists — and this "
    "admission is not a probe pass (RFC SI-9 §4.3/§4.4)")


class _AdmissionContext:
    """The gate's BundleContext for an admission re-evaluation.

    The SI-9b transformer path needs exactly two things from its
    context: the bundle's verified manifest (``base_manifest()`` /
    ``manifest``) and the current stack depth. The probe methods
    raise — a transformer gate evaluation that touches them would be
    faking §4.3's unevaluated checks (same fixture discipline as
    ``tests/test_gate_transformer.py``'s FakeCtx).
    """

    def __init__(self, manifest: dict, stack_depth: int):
        self.manifest = manifest
        self._depth = stack_depth

    def base_manifest(self):
        return self.manifest

    def stack_depth(self):
        return self._depth

    def identity_probes(self):
        raise AssertionError(
            "admission gate evaluation must not run identity probes: "
            "no probe harness exists for this (non-miniature) bundle")

    def capability_suite(self):
        raise AssertionError(
            "admission gate evaluation must not run capability probes: "
            "no probe harness exists for this (non-miniature) bundle")

    def capability_baseline(self):
        raise AssertionError(
            "admission gate evaluation must not read a capability "
            "baseline: no probe harness exists for this bundle")

    def forward_candidate(self, x, candidate):
        raise AssertionError(
            "admission gate evaluation must not run a forward pass: "
            "this context serves a non-miniature bundle, for which "
            "no forward pass exists")


def _stack_depth(manifest: dict) -> int:
    from si.bundle import _active_chain
    return len(_active_chain(manifest))


def admit_transformer_delta(bundle_dir: str, candidate, intent: Intent,
                            operator: str, reason: str,
                            gate: Gate | None = None) -> dict:
    """Operator-admit one escalated transformer delta. See module docs.

    ``candidate`` is the SI-9b candidate contract
    (``{"record": <transformer-delta-v1 record>, "blob_bytes": bytes}``)
    and ``intent`` the proposing ``Intent`` (kind
    ``"transformer-delta-v1"``, with its declared targets, parents,
    and source seqs). ``operator`` and ``reason`` are the admission
    attribution: both required, both journaled.

    Returns on success ``{"admitted": True, "verified": True,
    "rolled_back": False, "delta_id", "candidate_delta_id",
    "edit_id", "head", "gate_verdict": "ESCALATE", "gate_report",
    "verify_report", "operator", "reason", "admitted_at"}``.
    On VERIFY failure the delta is rolled back and the return is the
    same shape with ``"verified": False, "rolled_back": True``,
    ``"head"`` back at the pre-admission head, and
    ``"rollback_edit_id"`` set. Raises AdmitError on any refusal
    (missing operator/reason, malformed candidate/intent, a gate
    verdict other than ESCALATE) — nothing materialises then.
    """
    # -- operator-only gate: both attributions are required ----------
    if operator is None or not str(operator).strip():
        raise AdmitError(
            "--operator is required and must be non-empty: transformer "
            "delta admission is an operator-only action and the operator "
            "name is journaled as the authorization record")
    if reason is None or not str(reason).strip():
        raise AdmitError(
            "--reason is required and must be non-empty: every change "
            "has a cause, and an operator admission without a stated "
            "reason is refused")
    operator = str(operator).strip()
    reason = str(reason).strip()

    # -- candidate / intent shape (before any bundle is touched) -----
    if not isinstance(intent, Intent) \
            or intent.kind != TRANSFORMER_DELTA_KIND:
        raise AdmitError(
            "admission applies only to a transformer-delta intent "
            f"(kind {TRANSFORMER_DELTA_KIND!r}); got "
            f"{getattr(intent, 'kind', None)!r}")
    record_in = candidate.get("record") if isinstance(candidate, dict) else None
    blob_bytes = candidate.get("blob_bytes") if isinstance(candidate, dict) else None
    if not isinstance(record_in, dict) \
            or not isinstance(blob_bytes, (bytes, bytearray)):
        raise AdmitError(
            "candidate must be {'record': <transformer-delta-v1 record>, "
            "'blob_bytes': <bytes>} (the SI-9b candidate contract)")
    if record_in.get("kind") != TRANSFORMER_DELTA_KIND:
        raise AdmitError(
            f"candidate record kind {record_in.get('kind')!r} != "
            f"{TRANSFORMER_DELTA_KIND!r}: not a transformer delta")
    candidate = {"record": record_in, "blob_bytes": bytes(blob_bytes)}
    candidate_delta_id = record_in.get("id")

    bundle_dir = os.path.abspath(bundle_dir)

    # -- fail closed: a tampered bundle is never admitted into -------
    manifest = load_bundle(bundle_dir)
    prev_head = manifest["head"]
    if any(d.get("id") == candidate_delta_id for d in manifest["deltas"]):
        raise AdmitError(
            f"candidate delta id {candidate_delta_id!r} is already in "
            "the manifest's delta list: this specific delta has "
            "already been materialised (or its id collides); refusing "
            "to admit it twice")
    from si.bundle import _active_chain
    if any(d.get("blob") == record_in.get("blob")
           for d in _active_chain(manifest)):
        raise AdmitError(
            "a delta with this candidate's blob is already on the "
            "active chain: this specific delta is already admitted; "
            "refusing to stack it twice")

    # -- re-validate at admission: the SI-9b gate, fresh, now --------
    gate = gate or Gate(load_policy(bundle_dir))
    from si.probes import ProbeContext, is_miniature_probe_bundle
    if is_miniature_probe_bundle(manifest):
        # SI-16 (§6.5): a miniature bundle's admission re-evaluation
        # runs through the probe context, so shadow-mode probe
        # results are measured fresh at admission time and journaled
        # with the admission (in the gate report verbatim, and in
        # the operator_admission block below). The verdict is still
        # ESCALATE-or-REJECT: shadow probes never admit.
        ctx = ProbeContext(bundle_dir, manifest,
                           policy=load_policy(bundle_dir))
    else:
        ctx = _AdmissionContext(manifest, _stack_depth(manifest))
    report = gate.evaluate(intent, candidate, ctx)

    admitted_at = _loop._utcnow()
    intent_dict = intent.to_dict()
    admission_block = {
        "operator": operator,
        "reason": reason,
        "candidate_delta_id": candidate_delta_id,
        "admitted_at": admitted_at,
        "basis": ADMISSION_BASIS,
        "gate_verdict_at_admission": report.verdict,
    }
    # §6.5 labelling: if checks 3/4 were genuinely evaluated (a
    # miniature bundle in shadow mode), journal the measured probe
    # results alongside the basis — distinguished from both the
    # not-evaluated report and any probe pass claim. Absent for
    # non-miniature bundles, whose admission record is unchanged.
    probe_results = {
        c.name: c.to_dict() for c in report.checks
        if c.name in ("identity_continuity", "capability_floor")
        and c.measured is not None}
    if probe_results:
        admission_block["shadow_probe_results"] = {
            "mode": "shadow (RFC SI-13 §6.4: verdict held at "
                    "ESCALATE regardless of probe outcome)",
            "checks": probe_results,
            "note": "measured by the SI-16 probe harness at "
                    "admission time; the operator's admission — "
                    "not these results — is the basis for this "
                    "delta landing",
        }
    intent_dict["operator_admission"] = admission_block

    journal = open_journal(bundle_dir, manifest)
    try:
        if report.verdict != "ESCALATE":
            # REJECT-class (or any non-escalated) delta: never
            # admitted, by anyone. The refusal itself is journaled,
            # exactly the way the loop journals rejections — a
            # first-class, append-only record — then anchored, then
            # raised. Nothing materialises; the head never moves.
            edit_id = journal.record_self_edit(
                TRANSFORMER_DELTA_KIND, intent_dict, reason,
                report.verdict, report.to_dict(), operator=operator)
            journal.set_edit_result(
                edit_id,
                "rejected" if report.verdict == "REJECT" else "escalated")
            _loop._save_and_anchor(
                bundle_dir, journal, manifest,
                note=f"operator admission refused {edit_id}: "
                     f"gate {report.verdict}")
            raise AdmitError(
                f"gate verdict at admission is {report.verdict}, not "
                "ESCALATE: only a structurally valid transformer delta "
                "whose identity/capability probes were not evaluated "
                "may be operator-admitted; a REJECT-class delta can "
                f"never be admitted (refusal journaled as {edit_id})")

        # -- JOURNAL THE ADMISSION DECISION FIRST (before materialise)
        edit_id = journal.record_self_edit(
            TRANSFORMER_DELTA_KIND, intent_dict, reason,
            report.verdict, report.to_dict(), operator=operator)
        delta_id = _loop._next_delta_id(manifest)
        journal.record_provenance(
            child=delta_id,
            parents=list(intent.parents),
            relation="operator_admitted",
            note=(f"operator={operator} admitted candidate "
                  f"{candidate_delta_id} as {delta_id} at "
                  f"{admitted_at}: {reason} — {ADMISSION_BASIS}")[:1000])
        if intent.source_seqs:
            journal.record_provenance(
                child=edit_id, parents=list(intent.source_seqs),
                relation="learned_from", note="source journal sequences")

        # -- MATERIALISE (existing loop semantics, no parallel writer)
        blob_hex, byte_len = _loop._write_blob(
            bundle_dir, candidate["blob_bytes"])
        if "sha256:" + blob_hex != record_in.get("blob") \
                or byte_len != record_in.get("bytes"):
            # Unreachable after a passing gate (the structural suite
            # checks the record/blob binding), but fail closed rather
            # than materialise a record that disagrees with its blob.
            journal.set_edit_result(edit_id, "rejected")
            _loop._save_and_anchor(
                bundle_dir, journal, manifest,
                note=f"operator admission aborted {edit_id}: "
                     "record/blob binding mismatch")
            raise AdmitError(
                "candidate record/blob binding mismatch at materialise "
                f"time (record blob {record_in.get('blob')!r} vs "
                f"written sha256:{blob_hex}); nothing materialised")
        delta_rec = copy.deepcopy(record_in)
        delta_rec["id"] = delta_id
        delta_rec["parent"] = prev_head
        delta_rec["self_edit_id"] = edit_id
        manifest["deltas"].append(delta_rec)
        manifest["head"] = delta_id
        journal.record_transformer_delta(
            delta_id, prev_head, "sha256:" + blob_hex, byte_len,
            float(delta_rec["norm"]), edit_id, list(intent.source_seqs),
            delta_rec["targets"])
        journal.set_edit_result(edit_id, "materialized", delta_id=delta_id)
        manifest = _loop._save_and_anchor(
            bundle_dir, journal, manifest,
            note=f"post-materialize {delta_id} operator-admitted by "
                 f"{operator}")

        # -- VERIFY: the SI-9c apply smoke step, honestly labeled ----
        ok, verify_report = _loop._verify_transformer_apply(
            bundle_dir, manifest, delta_rec)
        journal.record_provenance(
            child=delta_id, parents=[edit_id],
            relation=verify_report.get("kind", "transformer-apply-smoke"),
            note=json.dumps(verify_report, sort_keys=True))
        if ok:
            manifest = _loop._save_and_anchor(
                bundle_dir, journal, manifest,
                note=f"post-verify {delta_id} transformer-apply-smoke "
                     "VERIFIED")
            return {"admitted": True, "verified": True,
                    "rolled_back": False, "delta_id": delta_id,
                    "candidate_delta_id": candidate_delta_id,
                    "edit_id": edit_id, "head": manifest["head"],
                    "gate_verdict": report.verdict,
                    "gate_report": report.to_dict(),
                    "verify_report": verify_report,
                    "operator": operator, "reason": reason,
                    "admitted_at": admitted_at}

        # -- VERIFY failed -> ROLLBACK (existing rollback semantics) -
        manifest, rb_edit_id, _probes_ok = _loop._do_rollback(
            bundle_dir, journal, manifest, gate,
            to=prev_head,
            reason=f"verify failed after operator-admitted {delta_id}",
            mind=None, operator=operator,
            cause="verify-failed",
            probe_report=verify_report,
            rolled_back_edit_id=edit_id)
        return {"admitted": True, "verified": False,
                "rolled_back": True, "delta_id": delta_id,
                "candidate_delta_id": candidate_delta_id,
                "edit_id": edit_id, "rollback_edit_id": rb_edit_id,
                "head": manifest["head"],
                "gate_verdict": report.verdict,
                "gate_report": report.to_dict(),
                "verify_report": verify_report,
                "operator": operator, "reason": reason,
                "admitted_at": admitted_at}
    finally:
        journal.close()


__all__ = ["AdmitError", "ADMISSION_BASIS", "admit_transformer_delta"]
