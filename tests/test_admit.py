"""SI-10: operator admission of escalated transformer deltas.

The missing materialise path from RFC SI-9 §4.4/§5.5: an ESCALATEd
transformer delta can be admitted by a named operator, with a reason,
after a fresh SI-9b gate re-evaluation — and by nobody/nothing else.

Proven here, on synthetic miniature fixtures (SI-9a/9c style):

- a valid escalated delta admitted by a named operator lands on the
  active chain, ``load_applied_tensor`` reflects it, the admission is
  journaled BEFORE the delta record, and the VERIFY smoke report is
  journaled under its honest ``transformer-apply-smoke`` label;
- the journal keeps saying the identity/capability probes were NOT
  evaluated — admission is the operator acting as that check, never
  a probe pass (gate verdict journaled: ESCALATE, verbatim);
- admission is refused without an operator identity or a reason
  (nothing written, nothing journaled);
- a REJECT-class delta cannot be admitted even by an operator (the
  refusal is journaled; the head never moves);
- a VERIFY failure rolls the admission back — head restored, delta
  and admission edit marked rolled_back, nothing silently active;
- the autonomous path is unchanged: ``run_cycles`` on the same valid
  delta still ESCALATEs, halts, and materialises nothing, and the
  demo-delta cycle still admits/verifies as before;
- the CLI wires the same function, operator-only.
"""
import json
import os

import numpy as np
import pytest
from safetensors.numpy import save

import si.loop
from si.admit import AdmitError, admit_transformer_delta
from si.apply import load_applied_tensor
from si.bundle import (
    _active_chain,
    _anchor_checkpoint,
    _write_manifest,
    canonical_json,
    init_bundle,
    load_bundle,
    load_policy,
    sha256_hex,
)
from si.cli import build_parser, main as cli_main
from si.gate import Gate, Intent
from si.journal import open_journal
from si.loop import run_cycle
from si.minds import DemoMind
from si.run import run_cycles
from si.seed import seed_bundle
from si.transformer_delta import KIND, construct_transformer_delta

QPROJ = "model.language_model.layers.0.self_attn.q_proj.weight"   # [4, 3]
VPROJ = "model.vision_tower.layers.0.attn.q_proj.weight"          # [2, 2]
NORM = "model.language_model.layers.0.input_layernorm.weight"     # [5]

SHAPES = {QPROJ: (4, 3), VPROJ: (2, 2), NORM: (5,)}

OPERATOR = "test-operator"
REASON = "reviewed the structural report; admitting as the operator check"


def _make_bundle(tmp_path, name, tensors=None):
    rng = np.random.default_rng(11)
    tensors = ({n: rng.standard_normal(s).astype(np.float32)
                for n, s in SHAPES.items()}
               if tensors is None else tensors)
    ckpt = tmp_path / (name + "-ckpt")
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(save(tensors))
    bundle_dir = init_bundle(str(tmp_path / (name + ".si")), name=name,
                             journal_backend="embedded")
    seed_bundle(bundle_dir, str(ckpt), importer="test")
    return bundle_dir, tensors


def _lora_spec(name=QPROJ, rank=2, alpha=4, seed=1, scale=0.05):
    rng = np.random.default_rng(seed)
    out_dim, in_dim = SHAPES[name]
    return {"name": name, "kind": "lora",
            "A": (rng.standard_normal((rank, in_dim)) * scale).astype(np.float32),
            "B": (rng.standard_normal((out_dim, rank)) * scale).astype(np.float32),
            "rank": rank, "alpha": alpha}


def _candidate(bundle_dir, specs=None, candidate_id="d0007"):
    manifest = load_bundle(bundle_dir)
    record, blob = construct_transformer_delta(
        manifest, [_lora_spec()] if specs is None else specs,
        delta_id=candidate_id, parent=manifest["head"],
        created_at="2026-10-19T00:00:00Z", self_edit_id="se_0007")
    return {"record": record, "blob_bytes": blob}


def _intent(record, **kw):
    base = dict(kind=KIND,
                reason="Adapt one named tensor with a small bounded delta.",
                parents=["base"], source_seqs=[1],
                targets=[t["name"] for t in record["targets"]])
    base.update(kw)
    return Intent(**base)


def _policy(**over):
    from si.bundle import DEFAULT_POLICY
    policy = dict(DEFAULT_POLICY)
    policy["allowed_target_patterns"] = [QPROJ]
    policy.update(over)
    return policy


def _hand_applied(base, spec):
    update = (float(spec["alpha"]) / float(spec["rank"])
              * (spec["B"] @ spec["A"])).astype(np.float32)
    return np.asarray(base, dtype=np.float32) + update


def _journal_rows(bundle_dir, manifest, collection, **flt):
    journal = open_journal(bundle_dir, manifest)
    try:
        return journal.backend.find(collection, filter=flt or None)
    finally:
        journal.close()


def _admit(bundle_dir, candidate, intent, **kw):
    kw.setdefault("operator", OPERATOR)
    kw.setdefault("reason", REASON)
    kw.setdefault("gate", Gate(_policy()))
    return admit_transformer_delta(bundle_dir, candidate, intent, **kw)


# -- the success path --------------------------------------------------------

def test_valid_escalated_delta_admitted_and_applies(tmp_path):
    bundle_dir, tensors = _make_bundle(tmp_path, "si10-ok")
    spec = _lora_spec(seed=3)
    cand = _candidate(bundle_dir, specs=[spec])
    res = _admit(bundle_dir, cand, _intent(cand["record"]))

    assert res["admitted"] is True
    assert res["verified"] is True
    assert res["rolled_back"] is False
    # The gate did NOT admit this — it escalated, and the operator did.
    assert res["gate_verdict"] == "ESCALATE"
    assert res["candidate_delta_id"] == "d0007"
    assert res["delta_id"] == "d0001"  # fresh runtime id, loop-style
    assert res["head"] == "d0001"
    assert res["operator"] == OPERATOR

    manifest = load_bundle(bundle_dir)
    assert manifest["head"] == "d0001"
    assert len(manifest["deltas"]) == 1
    rec = manifest["deltas"][0]
    assert rec["kind"] == KIND
    assert rec["parent"] == "base"
    assert rec["self_edit_id"] == res["edit_id"]
    assert [d["id"] for d in _active_chain(manifest)] == ["d0001"]

    # Apply-at-load reflects the admitted delta, hand-checked.
    applied = load_applied_tensor(bundle_dir, manifest, QPROJ)
    np.testing.assert_allclose(
        applied, _hand_applied(tensors[QPROJ], spec), rtol=1e-6, atol=1e-7)

    # Journal: admission self-edit, journaled BEFORE the delta record.
    edits = _journal_rows(bundle_dir, manifest, "self_edits",
                          edit_id=res["edit_id"])
    assert len(edits) == 1
    e = edits[0]
    assert e["kind"] == KIND
    assert e["gate_verdict"] == "ESCALATE"  # never rewritten to ADMIT
    assert e["result"] == "materialized"
    assert e["delta_id"] == "d0001"
    assert e["operator"] == OPERATOR
    assert e["reason"] == REASON
    adm = e["intent"]["operator_admission"]
    assert adm["operator"] == OPERATOR
    assert adm["reason"] == REASON
    assert adm["candidate_delta_id"] == "d0007"
    assert adm["admitted_at"] == res["admitted_at"]
    assert "NOT" in adm["basis"] and "not a probe pass" in adm["basis"]
    # The gate report is journaled verbatim: probes NOT evaluated.
    checks = {c["name"]: c for c in e["gate_report"]["checks"]}
    for name in ("identity_continuity", "capability_floor"):
        assert checks[name]["passed"] is False
        assert checks[name]["measured"] is None
        assert "not evaluated" in checks[name]["detail"]

    drows = _journal_rows(bundle_dir, manifest, "deltas", id="d0001")
    assert len(drows) == 1
    assert drows[0]["kind"] == KIND
    assert drows[0]["status"] == "active"
    assert drows[0]["targets"][0]["name"] == QPROJ
    assert e["seq"] < drows[0]["seq"]  # journal before materialise

    # Provenance: the admission decision + the VERIFY smoke report,
    # under its honest label.
    prov = _journal_rows(bundle_dir, manifest, "provenance", child="d0001")
    relations = {p["relation"] for p in prov}
    assert "operator_admitted" in relations
    assert "transformer-apply-smoke" in relations
    smoke = next(p for p in prov
                 if p["relation"] == "transformer-apply-smoke")
    report = json.loads(smoke["note"])
    assert report["kind"] == "transformer-apply-smoke"
    assert report["passed"] is True
    assert "NOT identity or capability" in report["note"]
    op = next(p for p in prov if p["relation"] == "operator_admitted")
    assert OPERATOR in op["note"] and "d0007" in op["note"]


def test_admitting_the_same_delta_twice_is_refused(tmp_path):
    bundle_dir, _ = _make_bundle(tmp_path, "si10-twice")
    cand = _candidate(bundle_dir)
    res = _admit(bundle_dir, cand, _intent(cand["record"]))
    assert res["admitted"] is True
    with pytest.raises(AdmitError, match="already"):
        _admit(bundle_dir, cand, _intent(cand["record"]))
    assert load_bundle(bundle_dir)["head"] == "d0001"
    assert len(load_bundle(bundle_dir)["deltas"]) == 1


# -- operator-only refusals ----------------------------------------------------

@pytest.mark.parametrize("operator,reason", [
    (None, REASON), ("", REASON), ("   ", REASON),
    (OPERATOR, None), (OPERATOR, ""), (OPERATOR, "   "),
])
def test_admission_refused_without_operator_or_reason(tmp_path, operator,
                                                      reason):
    bundle_dir, _ = _make_bundle(tmp_path, "si10-noattr")
    cand = _candidate(bundle_dir)
    before = load_bundle(bundle_dir)
    with pytest.raises(AdmitError):
        admit_transformer_delta(bundle_dir, cand, _intent(cand["record"]),
                                operator=operator, reason=reason,
                                gate=Gate(_policy()))
    after = load_bundle(bundle_dir)
    assert after["head"] == before["head"] == "base"
    assert after["deltas"] == []
    # Nothing journaled as an admission attempt either (amend precedent).
    assert _journal_rows(bundle_dir, after, "self_edits", kind=KIND) == []
    assert _journal_rows(bundle_dir, after, "deltas") == []


def test_wrong_kind_intent_refused(tmp_path):
    bundle_dir, _ = _make_bundle(tmp_path, "si10-kind")
    cand = _candidate(bundle_dir)
    intent = Intent(kind="delta", reason="sneak it in as a demo delta",
                    parents=["base"], source_seqs=[1])
    with pytest.raises(AdmitError, match="transformer-delta"):
        _admit(bundle_dir, cand, intent)
    assert load_bundle(bundle_dir)["head"] == "base"


# -- REJECT-class: never admitted, by anyone -----------------------------------

def test_reject_class_delta_cannot_be_admitted(tmp_path):
    bundle_dir, tensors = _make_bundle(tmp_path, "si10-reject")
    cand = _candidate(bundle_dir)
    # The bundle's gate policy (as this operator's gate resolves it)
    # does NOT allowlist the target: the fresh gate verdict is REJECT.
    rejecting_gate = Gate(_policy(allowed_target_patterns=[VPROJ]))
    with pytest.raises(AdmitError, match="REJECT"):
        admit_transformer_delta(bundle_dir, cand, _intent(cand["record"]),
                                operator=OPERATOR, reason=REASON,
                                gate=rejecting_gate)
    manifest = load_bundle(bundle_dir)
    assert manifest["head"] == "base"
    assert manifest["deltas"] == []
    assert _journal_rows(bundle_dir, manifest, "deltas") == []
    np.testing.assert_array_equal(
        load_applied_tensor(bundle_dir, manifest, QPROJ), tensors[QPROJ])
    # The refusal itself is journaled, operator attached (loop-style).
    edits = _journal_rows(bundle_dir, manifest, "self_edits", kind=KIND)
    assert len(edits) == 1
    assert edits[0]["gate_verdict"] == "REJECT"
    assert edits[0]["result"] == "rejected"
    assert edits[0]["operator"] == OPERATOR


def test_structurally_invalid_delta_cannot_be_admitted(tmp_path):
    bundle_dir, _ = _make_bundle(tmp_path, "si10-invalid")
    cand = _candidate(bundle_dir)
    cand["record"] = dict(cand["record"])
    cand["record"]["targets"] = [dict(cand["record"]["targets"][0])]
    cand["record"]["targets"][0]["base_blob"] = \
        "sha256:" + "0" * 64  # binding lie: gate structural suite rejects
    with pytest.raises(AdmitError, match="REJECT"):
        _admit(bundle_dir, cand, _intent(cand["record"]))
    assert load_bundle(bundle_dir)["head"] == "base"
    assert load_bundle(bundle_dir)["deltas"] == []


# -- VERIFY failure: rollback, never silently active ----------------------------

def test_verify_failure_rolls_back_the_admission(tmp_path, monkeypatch):
    bundle_dir, tensors = _make_bundle(tmp_path, "si10-verifyfail")
    cand = _candidate(bundle_dir)
    fake_report = {"kind": "transformer-apply-smoke",
                   "delta_id": "d0001",
                   "targets": [{"name": QPROJ, "passed": False,
                                "error": "injected failure"}],
                   "passed": False,
                   "note": "apply smoke test only (injected failure)"}
    monkeypatch.setattr(si.loop, "_verify_transformer_apply",
                        lambda bd, mf, rec: (False, fake_report))

    res = _admit(bundle_dir, cand, _intent(cand["record"]))
    assert res["admitted"] is True
    assert res["verified"] is False
    assert res["rolled_back"] is True
    assert res["head"] == "base"
    assert res["rollback_edit_id"]

    manifest = load_bundle(bundle_dir)
    assert manifest["head"] == "base"
    assert _active_chain(manifest) == []
    # History stays as evidence; the applied value is the base again.
    assert [d["id"] for d in manifest["deltas"]] == ["d0001"]
    np.testing.assert_array_equal(
        load_applied_tensor(bundle_dir, manifest, QPROJ), tensors[QPROJ])

    drows = _journal_rows(bundle_dir, manifest, "deltas", id="d0001")
    assert drows[0]["status"] == "rolled_back"
    edits = _journal_rows(bundle_dir, manifest, "self_edits",
                          edit_id=res["edit_id"])
    assert edits[0]["result"] == "rolled_back"
    rbs = _journal_rows(bundle_dir, manifest, "self_edits", kind="rollback")
    assert len(rbs) == 1
    assert rbs[0]["intent"]["cause"] == "verify-failed"
    assert rbs[0]["intent"]["probe_report"]["kind"] == \
        "transformer-apply-smoke"
    # The smoke report was also journaled under its own label.
    prov = _journal_rows(bundle_dir, manifest, "provenance", child="d0001")
    assert "transformer-apply-smoke" in {p["relation"] for p in prov}


# -- autonomous path unchanged ---------------------------------------------------

class _TransformerScriptMind:
    """One transformer proposal, then idle (test_gate_transformer style)."""

    def __init__(self, intent, candidate):
        self._intent = intent
        self._candidate = candidate
        self._proposed = False
        rng = np.random.default_rng(3)
        self._probes = [(f"script-id-{i}",
                         (v := rng.standard_normal(8).astype(np.float32)),
                         v.copy()) for i in range(2)]

    @property
    def name(self):
        return "transformer-script"

    def perceive(self, journal):
        return []

    def propose(self, perceptions, ctx):
        if not self._proposed:
            self._proposed = True
            return self._intent
        return Intent(kind="memory",
                      reason="Script exhausted: nothing more to propose.",
                      collection="experiences",
                      record={"kind": "observation", "data": {"note": "idle"}})

    def identity_probes(self):
        return self._probes

    def capability_suite(self):
        return self._probes

    def forward(self, x, active_deltas):
        return np.asarray(x, dtype=np.float64)

    def train_delta(self, intent):
        return self._candidate


def test_autonomous_run_still_escalates_and_materialises_nothing(tmp_path):
    bundle_dir, _ = _make_bundle(tmp_path, "si10-auto")
    cand = _candidate(bundle_dir)
    mind = _TransformerScriptMind(_intent(cand["record"]), cand)
    summary = run_cycles(bundle_dir, mind, 3, gate=Gate(_policy()))
    assert summary["escalated"] == 1
    assert summary["admitted"] == 0
    assert summary["halted"] is True
    assert summary["head"] == "base"
    manifest = load_bundle(bundle_dir)
    assert manifest["deltas"] == []
    # …and the SAME delta is then admissible by the operator path.
    res = _admit(bundle_dir, cand, _intent(cand["record"]))
    assert res["admitted"] is True
    assert load_bundle(bundle_dir)["head"] == "d0001"


def test_demo_delta_path_unchanged(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "si10-demo.si"),
                             name="si10-demo", journal_backend="embedded")
    res = run_cycle(bundle_dir, DemoMind(bundle_path=bundle_dir))
    assert res["verdict"] == "ADMIT"
    assert res["delta_id"] == "d0001"
    assert res["verified"] is True
    assert load_bundle(bundle_dir)["head"] == "d0001"


# -- CLI -------------------------------------------------------------------------

def _install_policy(bundle_dir, **over):
    """Write a new policy.json and re-anchor it in the manifest (the
    test-fixture stand-in for a journaled operator policy act)."""
    manifest = load_bundle(bundle_dir)
    policy = load_policy(bundle_dir)
    policy.update(over)
    data = canonical_json(policy)
    with open(os.path.join(bundle_dir, "policy.json"), "wb") as f:
        f.write(data)
    manifest["constitution"] = dict(manifest["constitution"])
    manifest["constitution"]["policy_hash"] = "sha256:" + sha256_hex(data)
    journal = open_journal(bundle_dir, manifest)
    try:
        anchored = _anchor_checkpoint(journal, manifest,
                                      head=manifest["head"],
                                      note="test policy install")
        _write_manifest(bundle_dir, anchored)
    finally:
        journal.close()
    return load_bundle(bundle_dir)


def _write_candidate_files(tmp_path, cand):
    record_file = tmp_path / "candidate-record.json"
    record_file.write_text(json.dumps(cand["record"]), encoding="utf-8")
    blob_file = tmp_path / "candidate-blob.safetensors"
    blob_file.write_bytes(cand["blob_bytes"])
    return str(record_file), str(blob_file)


def test_cli_parser_and_end_to_end(tmp_path, capsys):
    bundle_dir, _ = _make_bundle(tmp_path, "si10-cli")
    _install_policy(bundle_dir, allowed_target_patterns=[QPROJ])
    cand = _candidate(bundle_dir)
    record_file, blob_file = _write_candidate_files(tmp_path, cand)

    parser = build_parser()
    args = parser.parse_args(
        ["admit-transformer-delta", bundle_dir,
         "--record", record_file, "--blob", blob_file,
         "--operator", OPERATOR, "--reason", REASON])
    assert args.func.__name__ == "cmd_admit_transformer_delta"
    assert args.operator == OPERATOR

    rc = cli_main(["admit-transformer-delta", bundle_dir,
                   "--record", record_file, "--blob", blob_file,
                   "--operator", OPERATOR, "--reason", REASON,
                   "--source-seqs", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "admitted:  True" in out
    assert "delta_id:  d0001" in out
    assert "NOT evaluated" in out
    assert "not cryptographic authentication" in out.lower() or \
        "NOT cryptographic authentication" in out
    assert load_bundle(bundle_dir)["head"] == "d0001"

    # CLI without --operator: refused, non-zero exit, head unchanged.
    cand2 = _candidate(bundle_dir, specs=[_lora_spec(seed=9)],
                       candidate_id="d0008")
    record2, blob2 = _write_candidate_files(tmp_path, cand2)
    # Distinct file names so cand2 does not overwrite cand's evidence.
    os.replace(record2, str(tmp_path / "r2.json"))
    os.replace(blob2, str(tmp_path / "b2.safetensors"))
    rc = cli_main(["admit-transformer-delta", bundle_dir,
                   "--record", str(tmp_path / "r2.json"),
                   "--blob", str(tmp_path / "b2.safetensors"),
                   "--reason", REASON, "--source-seqs", "1"])
    assert rc == 1
    assert load_bundle(bundle_dir)["head"] == "d0001"
