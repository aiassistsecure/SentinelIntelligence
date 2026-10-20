# SI-17 — Admission flip for miniature bundles only (as-built)

> **Scope:** RFC SI-13 §6.4 (verdict mapping), §6.5 (labelling),
> §6.6 (VERIFY after materialisation), and §8's SI-17 slice.
> For miniature bundles with a validated probe context only,
> gate checks 3 (identity continuity) and 4 (capability floor)
> now decide the verdict for real: probe-fail → **REJECT**,
> all-pass → **ADMIT**-eligible. The autonomous loop materialises
> ADMITted miniature deltas and VERIFYs them with real probes,
> rolling back on failure. Every other context and bundle kind
> keeps its prior behaviour byte-for-byte: non-miniature bundles
> (the Glimmer 30B seed above all) still report checks 3/4
> "not evaluated" and ESCALATE; binding/startup failure still
> reports "not evaluated" + ESCALATE, never a pass or a fail.

## The recorded decision (RFC §8 SI-17 precondition)

SI-17 was conditional: shadow evidence reviewed + an explicit
decision recorded. That decision was **taken by the driver on
2026-10-20**, before this slice was built, and is recorded here
mirroring how SI-10 recorded its basis:

- SI-16 merged as PR #21 (merge commit
  `6ef1aedd248fec877758313c9501ed5e1ea95574`). RFC SI-13 is fully
  discharged by SI-16.
- The driver's own review of the SI-16 shadow evidence: full
  suite **270 passed, 2 skipped** on the driver's run; §5.6
  reference outcomes verified — zero delta drift exactly 0.0 /
  echo accuracy 1.0; benign rank-2 LoRA drift ≈5.8e-05 pass;
  destructive rank-8 `lm_head` fails check 3 at ≈1.02 > tol 0.25;
  echo attack fails check 4's accuracy arm 0.5 vs floor 0.9375;
  startup validation worst baseline deviation 0.0 over
  24 entries.
- Per-class sensitivity: 28 of 30 targetable tensors measurable;
  the blind set is exactly both layers' `k_proj.bias` (drift
  exactly 0.0 — mathematically necessary: a constant key bias
  shifts all of a query's attention scores equally and softmax
  is shift-invariant). The driver sanity-checked this reasoning
  and accepts it.
- **Decision: PROCEED with SI-17 for miniature bundles only,
  WITH the blind-set guard** (§"Blind-set guard" below).

This slice's own runs reproduce the shadow evidence exactly,
now with live verdicts (see "Acceptance criteria" below):
zero → ADMIT, drift 0.0; benign → ADMIT, drift
5.774e-05 / regression 1.816e-04; destructive → REJECT, drift
1.0175; echo attack → REJECT, accuracy arm 0.5 vs floor 0.9375;
startup worst baseline deviation 0.0 over 24 entries.

## What was built

- **`si/gate.py`**: the transformer branch applies §6.4's
  mapping when — and only when — both probe checks were
  genuinely evaluated (both `measured` values are real floats):
  decisive checks failing → REJECT (unchanged, all cases);
  evaluated + any probe fail → REJECT; evaluated + all pass →
  ADMIT. Any unevaluated outcome (`measured=None`: non-probe
  context, binding/startup failure, harness exception, or the
  blind guard below) → ESCALATE, exactly as SI-9b/SI-16.
  Evaluated check details are labelled `measured-pass` /
  `measured-fail` (§6.5).
- **`si/probes.py`**: `derive_probe_blind_targets()` (the
  derived blind set), `ProbeContext.candidate_measurability()`,
  and `ProbeContext.forward_head()` (landed-state forward for
  §6.6 VERIFY).
- **`si/loop.py`**: `run_cycle` routes miniature probe bundles
  to a new `_run_miniature_cycle` (the demo `LoopContext` is
  demo-shaped and cannot run on a miniature bundle at all — it
  loads a demo `W` tensor and demo delta blobs). The miniature
  cycle is the same serial shape (PERCEIVE → PROPOSE → GATE →
  JOURNAL → MATERIALIZE → VERIFY): REJECT/ESCALATE journal and
  halt as before; ADMIT journals first, materialises with the
  SI-10 transformer semantics (fresh runtime delta id,
  content-addressed blob — never duplicated bytes,
  `record_transformer_delta`), then VERIFYs via the new
  `_verify_probe_head` and rolls back through the existing
  rollback core on failure. Non-miniature bundles take the
  original `run_cycle` path unchanged.
- **`si/admit.py`**: SI-10 interplay (below). Per-case miniature
  admission bases (`MINIATURE_EVALUATED_BASIS`,
  `MINIATURE_BLIND_BASIS`, `MINIATURE_NOT_EVALUATED_BASIS`); the
  non-miniature `ADMISSION_BASIS` is unchanged verbatim.
- **`si/cli.py`**: the `admit-transformer-delta` note now has
  three honest branches — measured (miniature, with the gate
  verdict), blind-escalated, and the original NOT-evaluated
  wording for non-miniature bundles.
- **`tests/test_admission_flip.py`** (new, 10 tests) and the
  SI-17 updates to **`tests/test_probe_gate.py`** (miniature
  verdict/labelling expectations only; every not-evaluated and
  non-miniature expectation untouched).
- This doc. `si serve` is unchanged (501 for every transformer
  bundle, miniature included — tested); no policy default
  changed; the demo path is unchanged.

## Blind-set guard (driver requirement)

A candidate whose targets are **exclusively** probe-blind
tensors measures drift 0.0 *vacuously* — that is not a measured
pass, and under a naive flip it would have become ADMIT-eligible
on no evidence. The guard:

- **Derivation, not a magic list.**
  `derive_probe_blind_targets(architecture)` generates the set
  from the §3.1 descriptor: for each layer `i`, the one tensor
  whose mechanism is "a constant added identically to all keys",
  `model.language_model.layers.{i}.self_attn.k_proj.bias`.
  Mechanism: keys are `k_j = a_j @ W_k^T + b_k` with the SAME
  `b_k` at every position `j`, so a query's scores all shift by
  the constant `q · b_k`, and softmax is invariant to a constant
  shift — no probe input can ever detect a change to `b_k`
  through this architecture's outputs. No other tensor has that
  mechanism (q/v/o biases and all weights enter per-position,
  multiplicatively, or downstream of the softmax). SI-16's
  per-class sensitivity measurement corroborates the derivation
  empirically: 28/30 measurable, blind set exactly these two,
  drift exactly 0.0. A test asserts the derivation generalises
  (a 3-layer descriptor yields 3 derived names), so the set
  cannot silently be a frozen literal.
- **Blind-only → blind-escalated.** Both probe checks report
  `passed=False`, `measured=None`, and a detail beginning
  `blind-escalated: target(s) not measurable by the probe
  suites: …` with the reason stated. `measured=None` means the
  §6.4 mapping ESCALATEs it — the operator decides, exactly as
  for any unevaluated delta — and SI-10 admission of a blind
  delta journals the blind-escalated state and the blind basis
  (the operator acted as the check for those targets), never a
  pass.
- **Mixed (blind + measurable) → evaluated on the measurable
  evidence only.** The probes run for real; both checks'
  details name the blind target(s) as not measurable and state
  that ADMIT-eligibility rests on the measurable target(s)
  only. A mixed candidate whose measurable target fails REJECTs
  (tested); one whose measurable target passes may ADMIT
  (tested).

## §6.6 VERIFY (the loop)

`_verify_probe_head` re-runs the real probes against the
**landed** head (base + active chain, no candidate):
identity worst drift vs `identity_tol`; capability worst
regression vs the **pre-cycle** baseline (captured from the
gate's probe context before materialisation — the demo
`_verify_head` semantics) plus the §5.3 echo accuracy arm vs
the recorded floor. The report is journaled as a provenance
record under the relation `transformer-probe-verify`, with
each arm labelled `measured-pass` / `measured-fail`, the probe
artifact hash, and a note stating it is NOT the apply-smoke
test. A harness that cannot run at VERIFY time is a FAILED
verify (fail closed → rollback), labelled `not-evaluated`.
On failure the existing rollback core moves the head back,
marks the delta and its self-edit `rolled_back`, and journals
the probe report as the rollback's probe report (cause
`verify-failed`).

One honest note on the rollback test: gate and VERIFY measure
the same suites against the same tolerances, so a delta that
passes the gate cannot be constructed to fail VERIFY from
those tolerances alone. The rollback mechanics are therefore
proven with an injected post-land verify failure (the SI-10
precedent for its smoke-verify rollback), while the VERIFY
measurement itself is proven un-injected against base and
landed heads, and the destructive fixture is proven REJECTed
at the gate (it never lands — the §6.4 outcome for it).

## SI-10 interplay

- **Probe-failed miniature deltas are inadmissible.** Their
  fresh gate re-evaluation now REJECTs, and `si admit`'s
  existing rule (only admissible verdicts may land) refuses
  them: the refusal is journaled (REJECT, operator attached)
  and the head never moves. Tested end-to-end.
- **Probe-passing miniature deltas** have fresh verdict ADMIT.
  The operator path deliberately remains available for them
  (the operator chose it over the autonomous loop): admission
  journals `probe_results` (mode `measured`, superseding
  SI-16's `shadow_probe_results`) and the measured-pass basis.
  A non-miniature ADMIT — unreachable today — would still be
  refused: the exception is miniature + genuinely measured
  only, so this path has not silently widened.
- **Blind-escalated miniature deltas** remain operator-
  admissible from ESCALATE, journaling the blind state (above).
- The operator path's own VERIFY is still the
  `transformer-apply-smoke` step, labelled as what it is; the
  §6.6 probe VERIFY belongs to the autonomous loop. A miniature
  delta admitted by an operator therefore carries a smoke
  verify, NOT a probe verify — stated here so the journal
  record is never misread.

## §6.5 labelling summary

| State | `measured` | Detail label | Verdict (miniature, decisive pass) |
|---|---|---|---|
| measured-pass | real float | `measured-pass: …` | counts toward ADMIT |
| measured-fail | real float | `measured-fail: …` | REJECT |
| not evaluated | `None` | legacy string verbatim, or `not evaluated: <reason>` | ESCALATE |
| blind-escalated | `None` | `blind-escalated: target(s) not measurable …` | ESCALATE |

VERIFY reports carry the same vocabulary per arm, under the
`transformer-probe-verify` kind; admission records carry it in
`probe_results.mode` (`measured` / `blind-escalated`) plus the
per-case basis string.

## Acceptance criteria (RFC §8 SI-17) — status

- ✅ End-to-end autonomous cycle on a miniature bundle admits a
  benign fixture delta (ADMIT, head `d0001`), verifies it with
  real probes (§6.6 report journaled, identity label
  measured-pass, echo accuracy 1.0), and a destructive fixture
  is REJECTed at the gate with measured-fail journalling, head
  unchanged. VERIFY-failure rollback proven via injected
  post-land failure (see the honest note above): head restored,
  delta/self-edit `rolled_back`, rollback journaled with the
  probe report.
- ✅ Blind guard: blind-only ESCALATEs as blind-escalated
  (gate + loop + operator-admission journalling); mixed
  ADMITs on measurable evidence only, with the guard stated
  in the gate details; mixed-with-failing-measurable REJECTs.
- ✅ SI-10 interplay: probe-failed miniature refused by
  `si admit` (tested); probe-passing miniature admitted by an
  operator journals measured `probe_results`; blind miniature
  admitted by an operator journals the blind-escalated state.
- ✅ Non-miniature fixtures still escalate with checks 3/4
  "not evaluated" (Glimmer-shaped fixture via ProbeContext and
  the legacy-context verbatim test in `test_probe_gate.py`;
  the SI-9b/SI-10 suites unchanged and green).
- ✅ Full suite on the author's own runs: **280 passed,
  2 skipped** (272-test SI-16 baseline + 10 new SI-17 tests;
  SI-16 miniature expectations in `test_probe_gate.py` updated
  to the live mapping, nothing else). The final full run was
  clean at exactly those numbers; an earlier full run hit the
  known pre-existing nedbd flake once
  (`test_seed_refuses_history`, a stale nedbd-v2 daemon from
  the suite run holding its port): killed by specific PID,
  rerun passed.

## What is still NOT claimed

- **No Glimmer probes, no Glimmer forward path.** Nothing here
  measures any non-miniature base; the Glimmer seed's deltas
  still escalate with checks 3/4 "not evaluated", forever,
  until a future RFC designs a harness for that base.
- **No serve change.** Transformer bundles — miniature ones
  included — still return 501 from `si serve`. The probe
  forward path is a gate/loop measurement instrument, not a
  serving surface (RFC §7); §6.6 specifies VERIFY, not serving,
  and this slice does not widen it.
- **No training, no delta authorship.** The miniature cycle
  consumes hand-built fixture candidates exactly as before;
  `TinyTransformerMind.train_delta` still refuses.
- **No transformer squash, no policy default changes, no
  demo-path changes.** Probe VERIFY exists only for ADMITted
  miniature deltas in the autonomous loop; operator admissions
  still verify by apply-smoke (labelled, above).

## Terminology

Machine learning infrastructure. This document contains no
"custom LLM software" claims; the miniature is a constructed
machine-learning test instrument, not a trained model, and
probe results on it say something about deltas against the
miniature only — nothing about any other base.
