"""The self-write loop: PERCEIVE -> PROPOSE -> GATE -> JOURNAL ->
MATERIALIZE -> VERIFY, strictly serial. One cycle = one self-write.

Implements docs/self-write-loop.md for the SI-0 reference runtime.

Guarantees:
- No ungated write: the only path from "mind wants" to "bundle changed"
  passes through Gate.evaluate.
- No silent change: every transition is journaled with a reason —
  including rejections and escalations.
- Rollback is always possible: the base is frozen, history is append-only,
  rejected blobs stay on disk as evidence.
"""
from __future__ import annotations

import copy
import json
import os

from safetensors import safe_open

from si.bundle import (
    TamperError,
    _active_chain,
    load_bundle,
    load_policy,
    sha256_hex,
    _anchor_checkpoint,
    _write_manifest,
)
from si.gate import Gate, Intent
from si.journal import Journal, open_journal

# SI-16: the miniature probe context lives in si.probes (the
# loop/admit layer's third BundleContext sibling, next to
# LoopContext below and si.admit._AdmissionContext). Re-exported
# here so the loop layer names all three in one place.
# SI-17: run_cycle DOES construct it now — for miniature bundles
# only, where §6.4's verdict mapping is live (recorded decision,
# docs/SI-17-admission-flip.md). Non-miniature bundles keep the
# LoopContext path below, byte-for-byte.
from si.probes import ProbeContext  # noqa: F401


# -- bundle context for the gate -------------------------------------------
class LoopContext:
    """Implements the gate's BundleContext protocol for one cycle."""

    def __init__(self, bundle_dir: str, manifest: dict, journal: Journal, mind):
        self.bundle_dir = bundle_dir
        self.mind = mind
        self.journal = journal
        # The verified manifest, exposed for the gate's SI-9b
        # transformer-delta evaluation (manifest only; the gate never
        # reads base tensor bytes). Demo evaluation does not use it.
        self.manifest = manifest
        self.policy = load_policy(bundle_dir)
        self.head = manifest["head"]
        self.last_seq = manifest["journal"]["last_seq"]
        self.base_W = _load_base_W(bundle_dir, manifest)
        self.active = _load_stack(bundle_dir, manifest)  # [{B,A,scale}, ...]
        self._baseline = {
            cid: _probe_err(self._forward(x), expected)
            for cid, x, expected in mind.capability_suite()
        }

    # BundleContext protocol
    def identity_probes(self): return self.mind.identity_probes()
    def capability_suite(self): return self.mind.capability_suite()
    def capability_baseline(self): return dict(self._baseline)
    def stack_depth(self): return len(self.active)

    def base_manifest(self):
        """The bundle's verified manifest dict (SI-9b gate context)."""
        return self.manifest

    def forward_candidate(self, x, candidate):
        d = candidate["delta"]
        extra = [{"B": d["B"], "A": d["A"], "scale": d["alpha"] / d["rank"]}]
        return self.mind.forward(x, self.active + extra)

    def _forward(self, x):
        return self.mind.forward(x, self.active)


def _probe_err(y, expected) -> float:
    import numpy as np
    return float(np.max(np.abs(np.asarray(y, dtype=np.float64)
                               - np.asarray(expected, dtype=np.float64))))


def _load_base_W(bundle_dir: str, manifest: dict):
    # Resolves across every base layout (demo blob, pre-SI-7 whole-file
    # parts, SI-7 per-tensor blobs) via the shared bundle helper.
    from si.bundle import load_base_tensor
    return load_base_tensor(bundle_dir, manifest, "W")


def _load_stack(bundle_dir: str, manifest: dict) -> list:
    """Active delta stack: manifest deltas up to and including head.

    Resolved through ``bundle._active_chain`` so squash replacements
    (``superseded_by``) are followed: the stack holds the updates that
    are actually live, not the pre-squash history.
    """
    import numpy as np
    stack = []
    for d in _active_chain(manifest):
        blob_hex = d["blob"].split(":", 1)[1]
        with safe_open(os.path.join(bundle_dir, "blobs", "sha256",
                                    blob_hex + ".safetensors"), framework="np") as f:
            B = f.get_tensor("delta.B")
            A = f.get_tensor("delta.A")
        stack.append({"B": np.asarray(B), "A": np.asarray(A),
                      "scale": d["alpha"] / d["rank"], "id": d["id"]})
    return stack


def _next_delta_id(manifest: dict) -> str:
    n = len(manifest["deltas"]) + 1
    return f"d{n:04d}"


def _record_tool_shadow_outcome(journal: Journal, gate: Gate, intent,
                                ctx, edit_id: str, verdict: str) -> None:
    """SI-42 (SI-41 RFC §15 B-1): journal a tool cycle's shadow outcome.

    In B-1 a tool intent can only REJECT or ESCALATE (no executor
    exists), and its outcome is journaled as a ``tool_result``
    experience recording execution as not-evaluated — shadow mode,
    via the same writer si.gaia.shadow uses, so the loop path and
    the standalone path record identically shaped outcomes.
    """
    from si.gaia.shadow import record_shadow_outcome
    info = gate.classify_tool_intent(intent, ctx)
    record_shadow_outcome(journal, intent, edit_id, verdict,
                          effective_class=info.get("effective_class"))


def _record_training_hyperparams(journal: Journal, candidate,
                                 child: str, parents: list) -> None:
    """SI-21: journal a trained candidate's hyperparams, when it
    carries any, as a provenance note (relation
    ``training-hyperparams``).

    RFC SI-20 §10.2 requires every cycle's hyperparams to record
    the head trained against, the target-source head, and the
    stack depth; for a REJECTed cycle the journal is the only
    durable record of what was trained, so the hyperparams are
    journaled on BOTH verdict paths of the miniature cycle (on
    ADMIT under the landed delta id, on REJECT/ESCALATE under
    the self-edit id). Candidates without hyperparams (scripted
    fixture minds) journal nothing here — no fabricated fields.
    The note is the hyperparams dict verbatim, including its one
    labelled non-deterministic field (``wall_time_seconds``).
    """
    if not isinstance(candidate, dict):
        return
    hp = candidate.get("hyperparams")
    if not isinstance(hp, dict) or not hp:
        return
    journal.record_provenance(
        child=child, parents=list(parents),
        relation="training-hyperparams",
        note=json.dumps(hp, sort_keys=True))


def _write_blob(bundle_dir: str, blob_bytes: bytes) -> tuple[str, int]:
    """Write an immutable content-addressed blob. Returns (hex, byte_len)."""
    hexpart = sha256_hex(blob_bytes)
    dest = os.path.join(bundle_dir, "blobs", "sha256", hexpart + ".safetensors")
    if not os.path.isfile(dest):
        tmp = dest + ".tmp"
        with open(tmp, "wb") as f:
            f.write(blob_bytes)
        os.replace(tmp, dest)
    return hexpart, len(blob_bytes)


def _save_and_anchor(bundle_dir: str, journal: Journal, manifest: dict,
                     note: str) -> dict:
    """Checkpoint the journal and anchor seq/hash in the manifest, then
    persist the manifest. Returns the updated manifest dict."""
    updated = _anchor_checkpoint(journal, manifest, manifest["head"], note)
    _write_manifest(bundle_dir, updated)
    return updated


# -- the cycle --------------------------------------------------------------
def run_cycle(bundle_path: str, mind, gate: Gate | None = None) -> dict:
    """Run one serial self-write cycle. Returns a result dict:

    {"verdict": "ADMIT"|"REJECT"|"ESCALATE", "edit_id": "se_XXXX",
     "delta_id": "dXXXX"|None, "head": <head after cycle>,
     "verified": bool, "rolled_back": bool}
    """
    bundle_dir = os.path.abspath(bundle_path)
    manifest = load_bundle(bundle_dir)
    journal = open_journal(bundle_dir, manifest)
    try:
        # SI-17: a miniature probe bundle runs its transformer
        # cycles through the probe context (the LoopContext below
        # is demo-shaped: it loads a demo "W" base tensor and a
        # demo delta stack, neither of which a miniature bundle
        # has). Every other bundle takes the path below unchanged.
        from si.probes import is_miniature_probe_bundle
        if is_miniature_probe_bundle(manifest):
            return _run_miniature_cycle(bundle_dir, manifest, journal,
                                          mind, gate)
        # PERCEIVE (read-only).
        perceptions = mind.perceive(journal)
        ctx = LoopContext(bundle_dir, manifest, journal, mind)

        # PROPOSE. For delta intents the RUNTIME trains the candidate here;
        # the mind supplies the training signal, never touches raw tensors.
        intent = mind.propose(perceptions, ctx)
        candidate = None
        # Transformer-delta intents (SI-9b) also obtain a candidate so
        # the gate can evaluate it. This adds NO materialize path for
        # them: the SI-9b gate never ADMITs a transformer delta (a
        # valid one ESCALATEs to the operator, an invalid one is
        # REJECTed — both journaled and halted below), and the
        # MATERIALIZE step further down is demo-"delta"-only.
        if intent.kind in ("delta", "transformer-delta-v1") \
                and intent.reason and intent.reason.strip():
            candidate = mind.train_delta(intent)

        # GATE.
        gate = gate or Gate(load_policy(bundle_dir))
        report = gate.evaluate(intent, candidate, ctx)

        if report.verdict in ("REJECT", "ESCALATE"):
            # Rejections are journaled by the caller (the loop). No materialize.
            edit_id = journal.record_self_edit(
                intent.kind, intent.to_dict(), intent.reason,
                report.verdict, report.to_dict())
            journal.set_edit_result(
                edit_id, "rejected" if report.verdict == "REJECT" else "escalated")
            if intent.kind == "tool":
                _record_tool_shadow_outcome(
                    journal, gate, intent, ctx, edit_id, report.verdict)
            manifest = _save_and_anchor(
                bundle_dir, journal, manifest,
                note=f"{report.verdict.lower()} {edit_id}")
            return {"verdict": report.verdict, "edit_id": edit_id,
                    "delta_id": None, "head": manifest["head"],
                    "verified": False, "rolled_back": False}

        # JOURNAL (on ADMIT): intent + verdict + provenance BEFORE materialize.
        delta_id = _next_delta_id(manifest) if intent.kind == "delta" else None
        edit_id = journal.record_self_edit(
            intent.kind, intent.to_dict(), intent.reason, "ADMIT", report.to_dict())
        journal.record_provenance(
            child=delta_id or edit_id,
            parents=list(intent.parents),
            relation="learned_from",
            note=(intent.reason or "")[:200])
        if intent.source_seqs:
            journal.record_provenance(
                child=edit_id, parents=list(intent.source_seqs),
                relation="learned_from", note="source journal sequences")

        prev_head = manifest["head"]
        if intent.kind == "delta":
            # MATERIALIZE: write the immutable blob ONLY if weights changed.
            blob_hex, byte_len = _write_blob(bundle_dir, candidate["blob_bytes"])
            d = candidate["delta"]
            delta_rec = {
                "id": delta_id,
                "parent": prev_head,
                "blob": "sha256:" + blob_hex,
                "bytes": byte_len,
                "rank": d["rank"],
                "alpha": d["alpha"],
                "target_modules": list(d["target_modules"]),
                "norm": float(candidate["norm"]),
                "created_at": _utcnow(),
                "self_edit_id": edit_id,
            }
            manifest["deltas"].append(delta_rec)
            manifest["head"] = delta_id
            journal.record_delta(
                delta_id, prev_head, "sha256:" + blob_hex, byte_len,
                d["rank"], d["alpha"], list(d["target_modules"]),
                float(candidate["norm"]), edit_id, list(intent.source_seqs))
            journal.set_edit_result(edit_id, "materialized", delta_id=delta_id)
        elif intent.kind == "memory":
            # Memory intents touch the journal only: no blob, no delta record.
            rec = intent.record or {}
            journal.record_experience(
                rec.get("kind", "observation"), rec, edit_id)
            journal.set_edit_result(edit_id, "materialized")
        elif intent.kind == "rollback":
            # A mind-proposed rollback delegates fully to _do_rollback,
            # which journals the rollback as its own self-edit.
            journal.set_edit_result(edit_id, "materialized")
            target = intent.target or _previous_head(manifest, prev_head)
            manifest, rb_edit_id, _probes_ok = _do_rollback(
                bundle_dir, journal, manifest, gate,
                to=target, reason=intent.reason,
                mind=mind, operator=None, cause="mind-proposed")
            return {"verdict": "ADMIT", "edit_id": rb_edit_id,
                    "delta_id": None, "head": manifest["head"],
                    "verified": True, "rolled_back": True}

        manifest = _save_and_anchor(
            bundle_dir, journal, manifest,
            note=f"post-materialize {delta_id or edit_id}")

        # VERIFY: re-run probes against the new head.
        ok, probe_report = _verify_head(bundle_dir, manifest, mind, ctx)
        if ok:
            manifest = _save_and_anchor(
                bundle_dir, journal, manifest,
                note=f"post-verify {delta_id or edit_id} VERIFIED")
            return {"verdict": "ADMIT", "edit_id": edit_id,
                    "delta_id": delta_id, "head": manifest["head"],
                    "verified": True, "rolled_back": False}

        # VERIFY failed -> ROLLBACK. The delta's self-edit is marked
        # rolled_back; the rollback itself gets its own self-edit.
        manifest, rb_edit_id, _probes_ok = _do_rollback(
            bundle_dir, journal, manifest, gate,
            to=prev_head,
            reason=f"verify failed after {delta_id or edit_id}",
            mind=mind, operator=None,
            cause="verify-failed",
            probe_report=probe_report,
            rolled_back_edit_id=edit_id)
        return {"verdict": "ADMIT", "edit_id": rb_edit_id,
                "delta_id": delta_id, "head": manifest["head"],
                "verified": False, "rolled_back": True}
    finally:
        journal.close()


def _verify_probe_head(bundle_dir: str, manifest: dict,
                       policy: dict | None = None,
                       baseline: dict | None = None):
    """§6.6 VERIFY for a miniature bundle: re-run the REAL probes
    against the landed head (base + active chain, no candidate).

    Returns (ok, probe_report) in ``_verify_head``'s shape, with
    the report labelled ``transformer-probe-verify`` so a journal
    reader can never mistake it for the apply-smoke report or a
    demo probe report (§6.5). Identity is judged vs the seed's
    expected logits within ``identity_tol``; capability is judged
    as (a) worst regression vs ``baseline`` — the PRE-CYCLE head's
    errors, captured by the caller before materialisation (the
    demo ``_verify_head`` semantics); when no baseline is given,
    the recorded seed baselines (0.0 by construction) are used —
    plus (b) the §5.3 echo accuracy arm vs the recorded floor.
    Every measured value in the report is real, and each arm is
    labelled measured-pass / measured-fail. A harness that cannot
    run (binding/startup failure, any exception) is a FAILED
    verify (fail closed -> the caller rolls back), labelled
    not-evaluated — never a pass.
    """
    import numpy as np

    policy = policy if policy is not None else load_policy(bundle_dir)
    report: dict = {"kind": "transformer-probe-verify",
                    "head": manifest.get("head"), "passed": False}
    try:
        ctx = ProbeContext(bundle_dir, manifest, policy=policy)
        ok_status, reason = ctx.probe_status()
        report["probe_artifact_hash"] = ctx.probe_artifact_hash
        if not ok_status:
            report.update({
                "label": "not-evaluated",
                "note": f"probe harness not evaluated at verify: {reason}",
            })
            return False, report
        id_tol = float(policy.get("identity_tol", 0.0))
        worst_id, worst_id_q = 0.0, ""
        for qid, x, expected in ctx.identity_probes():
            err = _probe_err(ctx.forward_head(x), expected)
            if err > worst_id:
                worst_id, worst_id_q = err, qid
        id_passed = worst_id <= id_tol
        eps = float(policy.get("capability_eps", 0.0))
        if baseline is None:
            baseline = {cid: 0.0 for cid, _x, _e in ctx.capability_suite()}
            report["baseline_source"] = "recorded seed baselines (0.0)"
        else:
            report["baseline_source"] = "pre-cycle head (caller-captured)"
        targets = ctx.capability_targets()
        floor = float(ctx.echo_floor())
        worst_reg, worst_reg_c = -1e18, ""
        correct = 0
        suite = ctx.capability_suite()
        for cid, x, expected in suite:
            logits = ctx.forward_head(x)
            err = _probe_err(logits, expected)
            regress = err - float(baseline.get(cid, 0.0))
            if regress > worst_reg:
                worst_reg, worst_reg_c = regress, cid
            if int(np.argmax(np.asarray(logits))) == int(targets[cid]):
                correct += 1
        accuracy = correct / len(suite) if suite else 0.0
        reg_passed = worst_reg <= eps
        acc_passed = accuracy >= floor
        cap_passed = reg_passed and acc_passed
        report.update({
            "identity": {"max_drift": worst_id, "tol": id_tol,
                         "worst_probe": worst_id_q, "passed": id_passed,
                         "label": "measured-pass" if id_passed
                                  else "measured-fail"},
            "capability": {"max_regression": worst_reg, "eps": eps,
                           "worst_eval": worst_reg_c,
                           "passed": reg_passed,
                           "echo_accuracy": accuracy,
                           "echo_floor": floor,
                           "echo_passed": acc_passed,
                           "label": "measured-pass" if cap_passed
                                    else "measured-fail"},
            "passed": bool(id_passed and cap_passed),
            "note": "real probe VERIFY (RFC SI-13 §6.6, SI-17): "
                    "identity + capability (regression and echo "
                    "accuracy arms) re-run against the landed head "
                    "through the SI-15 forward path; NOT the "
                    "transformer-apply-smoke test",
        })
        return report["passed"], report
    except Exception as err:  # verify failed closed, never a pass
        report.update({
            "label": "not-evaluated",
            "note": f"probe VERIFY raised ({type(err).__name__}: {err}); "
                    "fail closed — the measurement failed, so the "
                    "landed state is not verified",
        })
        return False, report


def _run_miniature_cycle(bundle_dir: str, manifest: dict, journal,
                         mind, gate: Gate | None = None) -> dict:
    """One autonomous cycle on a miniature probe bundle (SI-17).

    Same serial shape as :func:`run_cycle` — PERCEIVE -> PROPOSE
    -> GATE -> JOURNAL -> MATERIALIZE -> VERIFY — with the probe
    context as the gate's context and transformer materialise /
    §6.6 probe VERIFY semantics:

    - REJECT / ESCALATE are journaled and halt the cycle, exactly
      as the demo path journals them (nothing materialises). A
      probe-failed delta REJECTs here (§6.4); a blind-only or
      not-evaluated delta ESCALATEs.
    - ADMIT (measured probe pass only) journals first, then
      materialises with the SI-10 transformer semantics (fresh
      runtime delta id, content-addressed blob, head advances,
      ``record_transformer_delta``), then VERIFYs with REAL
      probes (:func:`_verify_probe_head`). A failed VERIFY rolls
      back through the existing rollback core (cause
      ``verify-failed``, the probe report journaled with the
      rollback) — the delta is never left silently active.
    - Non-transformer intents keep their ordinary meanings
      (memory journals only; a mind-proposed rollback delegates
      to the rollback core with ``mind=None`` — the core's demo
      probe re-check cannot run on a miniature bundle, and
      claiming it did would be fabrication).
    - SI-21: a trained candidate's hyperparams are journaled as
      a ``training-hyperparams`` provenance note on both verdict
      paths (see :func:`_record_training_hyperparams`), so a
      second cycle's head / target-source / stack-depth record
      survives even when the gate REJECTs the cycle.
    """
    from si.transformer_delta import KIND as TF_KIND

    policy = load_policy(bundle_dir)
    gate = gate or Gate(policy)
    ctx = ProbeContext(bundle_dir, manifest, policy=policy)

    perceptions = mind.perceive(journal)
    intent = mind.propose(perceptions, ctx)
    candidate = None
    if intent.kind in ("delta", TF_KIND) \
            and intent.reason and intent.reason.strip():
        candidate = mind.train_delta(intent)

    report = gate.evaluate(intent, candidate, ctx)

    if report.verdict in ("REJECT", "ESCALATE"):
        edit_id = journal.record_self_edit(
            intent.kind, intent.to_dict(), intent.reason,
            report.verdict, report.to_dict())
        journal.set_edit_result(
            edit_id, "rejected" if report.verdict == "REJECT" else "escalated")
        if intent.kind == "tool":
            _record_tool_shadow_outcome(
                journal, gate, intent, ctx, edit_id, report.verdict)
        _record_training_hyperparams(journal, candidate, edit_id, [])
        manifest = _save_and_anchor(
            bundle_dir, journal, manifest,
            note=f"{report.verdict.lower()} {edit_id}")
        return {"verdict": report.verdict, "edit_id": edit_id,
                "delta_id": None, "head": manifest["head"],
                "verified": False, "rolled_back": False}

    # ADMIT: journal intent + verdict + provenance BEFORE materialize.
    is_tf = intent.kind == TF_KIND
    delta_id = _next_delta_id(manifest) if is_tf else None
    edit_id = journal.record_self_edit(
        intent.kind, intent.to_dict(), intent.reason, "ADMIT",
        report.to_dict())
    journal.record_provenance(
        child=delta_id or edit_id,
        parents=list(intent.parents),
        relation="learned_from",
        note=(intent.reason or "")[:200])
    if intent.source_seqs:
        journal.record_provenance(
            child=edit_id, parents=list(intent.source_seqs),
            relation="learned_from", note="source journal sequences")

    prev_head = manifest["head"]
    if is_tf:
        # Pre-cycle capability baseline for §6.6 VERIFY (measured
        # at the head the candidate lands on, before it lands).
        pre_baseline = None
        ok_status, _reason = ctx.probe_status()
        if ok_status:
            pre_baseline = ctx.capability_baseline()
        record_in = candidate["record"]
        blob_hex, byte_len = _write_blob(bundle_dir, candidate["blob_bytes"])
        if "sha256:" + blob_hex != record_in.get("blob") \
                or byte_len != record_in.get("bytes"):
            # Unreachable after a passing gate; fail closed rather
            # than materialise a record that disagrees with its blob
            # (the SI-10 admission precedent).
            journal.set_edit_result(edit_id, "rejected")
            manifest = _save_and_anchor(
                bundle_dir, journal, manifest,
                note=f"miniature cycle aborted {edit_id}: "
                     "record/blob binding mismatch")
            return {"verdict": "REJECT", "edit_id": edit_id,
                    "delta_id": None, "head": manifest["head"],
                    "verified": False, "rolled_back": False}
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
        _record_training_hyperparams(journal, candidate, delta_id,
                                     [edit_id])
        manifest = _save_and_anchor(
            bundle_dir, journal, manifest,
            note=f"post-materialize {delta_id}")

        # VERIFY (§6.6): real probes against the landed state.
        ok, probe_report = _verify_probe_head(
            bundle_dir, manifest, policy=policy, baseline=pre_baseline)
        journal.record_provenance(
            child=delta_id, parents=[edit_id],
            relation=probe_report.get("kind", "transformer-probe-verify"),
            note=json.dumps(probe_report, sort_keys=True))
        if ok:
            manifest = _save_and_anchor(
                bundle_dir, journal, manifest,
                note=f"post-verify {delta_id} transformer-probe-verify "
                     "VERIFIED")
            return {"verdict": "ADMIT", "edit_id": edit_id,
                    "delta_id": delta_id, "head": manifest["head"],
                    "verified": True, "rolled_back": False}
        manifest, rb_edit_id, _probes_ok = _do_rollback(
            bundle_dir, journal, manifest, gate,
            to=prev_head,
            reason=f"probe verify failed after {delta_id}",
            mind=None, operator=None,
            cause="verify-failed",
            probe_report=probe_report,
            rolled_back_edit_id=edit_id)
        return {"verdict": "ADMIT", "edit_id": rb_edit_id,
                "delta_id": delta_id, "head": manifest["head"],
                "verified": False, "rolled_back": True}
    if intent.kind == "memory":
        rec = intent.record or {}
        journal.record_experience(rec.get("kind", "observation"), rec, edit_id)
        journal.set_edit_result(edit_id, "materialized")
        manifest = _save_and_anchor(
            bundle_dir, journal, manifest,
            note=f"post-materialize {edit_id}")
        return {"verdict": "ADMIT", "edit_id": edit_id,
                "delta_id": None, "head": manifest["head"],
                "verified": True, "rolled_back": False}
    if intent.kind == "rollback":
        journal.set_edit_result(edit_id, "materialized")
        target = intent.target or _previous_head(manifest, prev_head)
        manifest, rb_edit_id, _probes_ok = _do_rollback(
            bundle_dir, journal, manifest, gate,
            to=target, reason=intent.reason,
            mind=None, operator=None, cause="mind-proposed")
        return {"verdict": "ADMIT", "edit_id": rb_edit_id,
                "delta_id": None, "head": manifest["head"],
                "verified": True, "rolled_back": True}
    # Any other ADMITted kind on a miniature bundle: nothing in
    # this runtime materialises it (the gate's ADMIT for a
    # transformer-shaped intent is the only landing path here).
    journal.set_edit_result(edit_id, "materialized")
    manifest = _save_and_anchor(
        bundle_dir, journal, manifest, note=f"post-materialize {edit_id}")
    return {"verdict": "ADMIT", "edit_id": edit_id,
            "delta_id": None, "head": manifest["head"],
            "verified": False, "rolled_back": False}


def _utcnow() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _previous_head(manifest: dict, head: str) -> str:
    ids = [d["id"] for d in manifest["deltas"]]
    if head == "base" or head not in ids:
        return "base"
    i = ids.index(head)
    return ids[i - 1] if i > 0 else "base"


def _verify_head(bundle_dir: str, manifest: dict, mind, ctx: LoopContext):
    """Run identity + capability probes against the current head.

    Returns (ok, probe_report). Capability is judged against the
    pre-cycle baseline: no NEW regression beyond epsilon.
    """
    policy = ctx.policy
    active = _load_stack(bundle_dir, manifest)
    id_tol = float(policy.get("identity_tol", 0.0))
    eps = float(policy.get("capability_eps", 0.0))
    baseline = ctx.capability_baseline()
    worst_id, worst_id_q = 0.0, ""
    for qid, x, expected in mind.identity_probes():
        err = _probe_err(mind.forward(x, active), expected)
        if err > worst_id:
            worst_id, worst_id_q = err, qid
    worst_reg, worst_reg_c = -1e18, ""
    for cid, x, expected in mind.capability_suite():
        err = _probe_err(mind.forward(x, active), expected)
        regress = err - float(baseline.get(cid, 0.0))
        if regress > worst_reg:
            worst_reg, worst_reg_c = regress, cid
    ok = (worst_id <= id_tol) and (worst_reg <= eps)
    report = {"identity": {"max_drift": worst_id, "tol": id_tol,
                           "worst_probe": worst_id_q, "passed": worst_id <= id_tol},
              "capability": {"max_regression": worst_reg, "eps": eps,
                             "worst_eval": worst_reg_c,
                             "passed": worst_reg <= eps}}
    return ok, report


def _verify_transformer_apply(bundle_dir: str, manifest: dict,
                              delta_record: dict):
    """Post-write VERIFY smoke step for a transformer delta (SI-9c).

    RFC §4.4 / §5.5: post-materialisation VERIFY for a transformer
    delta can only re-run the §4.2 structural suite (the gate's job,
    already done before materialisation) plus an APPLY SMOKE TEST —
    apply the just-admitted delta to each of its own targets via
    ``si.apply.load_applied_tensor`` and check the result is
    correctly shaped and finite. There are no probes to run: this
    runtime has no transformer forward pass, and this report must
    never be read as an identity or capability result.

    Returns (ok, report) in ``_verify_head``'s shape, with the report
    labeled ``transformer-apply-smoke`` so a journal reader cannot
    mistake it for a probe report.

    Wiring, stated honestly: ``run_cycle`` does NOT call this,
    because under SI-9b the gate never ADMITs a transformer delta (a
    valid one ESCALATEs to the operator and the autonomous loop
    halts before MATERIALIZE), so no transformer delta is ever
    materialised by the autonomous loop and it has no post-write
    moment to verify. Calling it from the demo VERIFY path would be
    wrong — that path serves demo deltas only, and it is unchanged.
    Since SI-10 this function IS the VERIFY step of the
    operator-admission materialise path (``si/admit.py``): after an
    operator admits an escalated delta and it materialises, that
    path runs exactly this check and journals the report under the
    ``transformer-apply-smoke`` label; a failed smoke rolls the
    admission back. The autonomous loop still never calls it:
    since SI-17 the loop's miniature branch VERIFYs with real
    probes instead (``_verify_probe_head``, §6.6), and
    non-miniature deltas are never materialised by the loop.
    """
    import numpy as np

    from si.apply import load_applied_tensor

    targets_report = []
    ok = True
    for target in delta_record.get("targets") or []:
        name = target.get("name") if isinstance(target, dict) else None
        entry = {"name": name, "passed": False}
        try:
            applied = load_applied_tensor(bundle_dir, manifest, name)
            shape_ok = list(applied.shape) == list(target["base_shape"])
            finite_ok = bool(np.all(np.isfinite(applied)))
            entry.update({"shape": list(applied.shape),
                          "shape_ok": shape_ok, "finite": finite_ok,
                          "passed": shape_ok and finite_ok})
        except Exception as err:  # smoke step: a failed apply is a failed verify
            entry["error"] = f"{type(err).__name__}: {err}"
        ok = ok and entry["passed"]
        targets_report.append(entry)
    if not targets_report:
        ok = False
    report = {"kind": "transformer-apply-smoke",
              "delta_id": delta_record.get("id"),
              "targets": targets_report, "passed": ok,
              "note": "apply smoke test only (shape + finiteness via "
                      "si.apply.load_applied_tensor); NOT identity or "
                      "capability probes — no transformer probe "
                      "harness exists (RFC SI-9 §4.3/§4.4)"}
    return ok, report


# -- rollback ----------------------------------------------------------------
def rollback(bundle_path: str, to: str, reason: str | None = None,
             operator: str | None = None, mind=None,
             cause: str = "operator", probe_report: dict | None = None) -> dict:
    """Move head back to ``to`` ("base" or a delta id). Popped deltas keep
    ``rolled_back`` status; blobs stay on disk as evidence. The rollback is
    journaled as its own self-edit with cause + probe report, checkpointed,
    and (if a mind is given) probes are re-run to confirm the mind is whole.

    Returns {"head", "edit_id", "popped", "probes_ok"}.
    """
    bundle_dir = os.path.abspath(bundle_path)
    manifest = load_bundle(bundle_dir)
    journal = open_journal(bundle_dir, manifest)
    try:
        gate = Gate(load_policy(bundle_dir))
        manifest, rb_edit_id, probes_ok = _do_rollback(
            bundle_dir, journal, manifest, gate, to=to,
            reason=reason, mind=mind, operator=operator,
            cause=cause, probe_report=probe_report)
        return {"head": manifest["head"], "edit_id": rb_edit_id,
                "probes_ok": probes_ok}
    finally:
        journal.close()


def _do_rollback(bundle_dir: str, journal: Journal, manifest: dict, gate: Gate,
                 to: str, reason: str | None, mind, operator: str | None,
                 cause: str, probe_report: dict | None = None,
                 rolled_back_edit_id: str | None = None):
    """Shared rollback core. Returns (manifest, rollback_edit_id, probes_ok).

    The rollback is journaled as its own self-edit. If ``rolled_back_edit_id``
    is given (verify-failed path), that self-edit's result is marked
    ``rolled_back``.
    """
    ids = [d["id"] for d in manifest["deltas"]]
    if to != "base" and to not in ids:
        raise ValueError(f"rollback target {to!r} unknown (head={manifest['head']!r})")
    prev_head = manifest["head"]
    reason = reason or f"{cause}: move head {prev_head} -> {to}"

    # Light gate evaluation (constitution check only), still journaled.
    intent = Intent(kind="rollback", reason=reason, target=to)
    report = gate.evaluate(intent, None, None)  # rollback path never touches ctx
    if report.verdict == "REJECT":
        raise TamperError(f"rollback intent rejected by gate: {report.to_dict()}")

    # Popped deltas: everything above the target in stack order.
    if to == "base":
        popped = [d["id"] for d in manifest["deltas"]]
    else:
        popped = ids[ids.index(to) + 1:]

    edit_id = journal.record_self_edit(
        "rollback", {**intent.to_dict(),
                     "from": prev_head, "cause": cause,
                     "probe_report": probe_report},
        reason, report.verdict, report.to_dict(), operator=operator)

    # Move head back. No bytes are deleted — blobs stay as evidence.
    manifest["head"] = to
    for did in popped:
        journal.set_delta_status(did, "rolled_back")
    if rolled_back_edit_id is not None:
        journal.set_edit_result(rolled_back_edit_id, "rolled_back")
    journal.record_provenance(child=to, parents=popped,
                              relation="rolled_back_by", note=reason[:200])

    # Re-run probes against the restored head to confirm the mind is whole.
    probes_ok, post_report = True, None
    if mind is not None:
        active = _load_stack(bundle_dir, manifest)
        policy = load_policy(bundle_dir)
        worst = 0.0
        for _qid, x, expected in mind.identity_probes():
            worst = max(worst, _probe_err(mind.forward(x, active), expected))
        probes_ok = worst <= float(policy.get("identity_tol", 0.0))
        post_report = {"max_drift": worst, "passed": probes_ok}
        journal.record_provenance(child=edit_id, parents=[to],
                                  relation="learned_from",
                                  note=f"post-rollback probes: {post_report}")

    journal.set_edit_result(edit_id, "materialized")
    # Anchor LAST: the checkpoint must be the journal tip so the manifest
    # anchor matches on the next load.
    manifest = _save_and_anchor(
        bundle_dir, journal, manifest,
        note=f"rollback {prev_head}->{to} ({cause})")
    return manifest, edit_id, probes_ok
