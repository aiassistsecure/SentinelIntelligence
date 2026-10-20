# SI-16 — Probe suites + gate wiring, shadow mode (as-built)

> **Scope:** RFC SI-13 §5 (probe suites), §6 (gate integration),
> and §8's SI-16 slice. This slice **consumes** the SI-14 probe
> artifacts through the SI-15 forward path: for miniature bundles
> only, gate checks 3 (identity continuity) and 4 (capability
> floor) now run for real and their measured results are
> journaled — **in shadow mode only**. The verdict is held at
> **ESCALATE** regardless of probe outcome (§6.4 staging). A
> probe-passing miniature delta is not admitted by the gate; a
> probe-failing one is not rejected by it. SI-17's admission flip
> is conditional on this shadow evidence plus a recorded decision
> and is **not** implemented here. Non-miniature bundles —
> the Glimmer 30B seed above all — behave exactly as before
> SI-16: checks 3/4 "not evaluated", valid deltas ESCALATE.

## What was built

- **`si/probes.py`** (new): `ProbeContext`, the probe
  `BundleContext` (§6.1) — a sibling to `si.loop.LoopContext`
  (demo) and `si.admit._AdmissionContext` (non-miniature
  transformer admission), re-exported from `si.loop` so the
  loop/admit layer names all three contexts in one place. It
  implements §6.2 binding (`load_bound_probe_artifact`), §6.3
  harness startup validation, the `BundleContext` protocol over
  the bundle's recorded probe artifact, live capability baselines
  at the current head, and §5.3's functional expectations
  (`capability_targets()`, `echo_floor()`).
- **`si/gate.py`**: the transformer branch's two probe checks now
  come from one new method, `Gate._transformer_probe_checks`.
  For any context without the probe marker it returns the SI-9b
  report **verbatim** (`TRANSFORMER_PROBES_NOT_EVALUATED`,
  `measured=None`). For a validated probe context it runs the
  gate's own generic `_identity_check` / `_capability_check`
  (metric code reused, not forked) plus the §5.3 echo accuracy
  arm, and labels both checks per §6.5. The verdict computation
  is untouched: probe checks remain outside the decisive set, so
  constitution/safety/provenance failures still REJECT and
  everything else transformer-shaped still ESCALATEs.
- **`si/admit.py`**: admission re-evaluation chooses its context
  by bundle kind — `ProbeContext` for a miniature bundle,
  `_AdmissionContext` otherwise (unchanged). When checks 3/4 were
  genuinely evaluated, the journaled `operator_admission` block
  carries a `shadow_probe_results` entry with the measured
  checks alongside the unchanged `ADMISSION_BASIS` (§6.5: the
  results ride *alongside* the basis string; the basis is the
  operator's decision, not the probes). The gate report is
  journaled verbatim as before, so the measured results are in
  the record twice-labelled, never as a probe pass claim.
  `_AdmissionContext`'s raising probe methods keep their
  fail-closed behaviour; their messages now scope the "no probe
  harness" claim to the non-miniature bundle they serve.
- **`si/cli.py`**: the `admit-transformer-delta` note prints the
  shadow-mode wording when the admission's gate report carries
  measured probe results, and the original "NOT evaluated"
  wording otherwise — the CLI never claims probes were not
  evaluated when they were, or were when they were not.
- **`tests/test_probe_gate.py`** (new): 18 tests covering every
  §8 SI-16 acceptance criterion (below).
- This doc. `si serve` is unchanged (501 for every transformer
  bundle, miniature included); the demo path is unchanged; no
  policy default changed.

## As-built decisions (each explicit)

1. **Context detection is a duck-typed marker, not an import.**
   `ProbeContext.probe_harness = True`; the gate checks
   `getattr(ctx, "probe_harness", False)`. The gate therefore
   grows no dependency on the probe module, demo contexts and
   test doubles keep today's behaviour byte-for-byte (asserted
   by a test that the legacy not-evaluated string is returned
   *verbatim* for a non-probe context), and a context that
   merely *looks* probe-shaped but fails binding/startup
   validation still reports "not evaluated" with the specific
   reason appended.
2. **§6.2 binding, as implemented.** Verified from the bundle
   context (never the candidate), in `load_bound_probe_artifact`:
   (1) manifest records `base.probe_artifact` and the artifact
   file's content hash matches it; (2) the artifact's recorded
   base binding matches the verified manifest exactly — the
   tensor-map hash is recomputed as `sha256(canonical_json(
   base.tensor_blobs))`, which is precisely the unit SI-14
   hashed (names/shapes/dtypes + per-tensor content hashes +
   byte counts), and the recorded tensor count / total elements
   are cross-checked against the manifest map; (3) the
   artifact's *and* the manifest base's architecture descriptors
   both match §3.1 exactly via SI-15's `validate_architecture`.
   Rule 4 (candidate targets are tensors of this base) remains
   where the RFC puts it: the §4.2 safety suite's existence
   check, unchanged.
3. **§6.3 startup validation, as implemented.** Before any probe
   result counts, the context re-runs the **base only** (per-
   tensor `load_base_tensor` reads — no deltas, no candidate)
   through `si.tiny_transformer.forward` over all 24 suite
   entries and requires (a) every entry's measured error to match
   its recorded baseline within `BASELINE_TOL = 1e-6`, and (b)
   the measured base echo accuracy to equal the recorded base
   accuracy exactly (an argmax count, not a float — there is no
   tolerance story to hide behind). On this runtime the
   reproduction is in fact exact: worst baseline deviation 0.0
   over 24 entries (asserted by test), consistent with SI-15's
   bit-identical-reproduction result. A tampered-baseline
   fixture (binding green, baselines corrupted) yields
   "not evaluated" — tested separately from the binding
   fixtures, because §6.3 is a distinct failure mode from §6.2.
   `capability_baseline()` itself follows `LoopContext`
   semantics: measured **live at the current head** (base +
   active chain, pre-candidate), so on a non-base head the
   regression arm judges against the head the candidate would
   actually land on, not against the seed-time record.
4. **Check 4's two arms.** The regression arm *is*
   `_capability_check` — same code, same worst-entry detail. The
   accuracy arm is the one new measurement: fraction of the 16
   capability entries whose candidate argmax equals the recorded
   echo target, against the recorded floor (base accuracy −
   1/16 = 0.9375). Both arms land in the check's `detail` string
   and as named `subchecks` (`capability_regression`,
   `echo_accuracy`), so the journal shows which arm decided.
   Check 4 passes iff both arms pass; its `measured` is the
   regression arm's worst regression (the demo check's measured
   semantics, unchanged).
5. **The legacy not-evaluated string is retained verbatim** for
   contexts without a validated harness (RFC §6.3 permits a
   successor string; §8's "behaves exactly as today" for
   non-miniature bundles wins). For a probe context whose
   binding/startup validation fails, the detail is
   `not evaluated: <specific reason>` — the same plainness, with
   the cause named. A harness exception during evaluation is
   reported the same way, with the exception recorded and the
   explicit note that the candidate did not fail — the
   measurement failed.
6. **Shadow verdict mechanics.** Nothing about the verdict
   computation changed: probe checks stay outside the decisive
   set in `Gate.evaluate`. Evaluated probe checks therefore
   appear in the report with real `passed`/`measured` values
   *and* an ESCALATE verdict — that juxtaposition is the shadow
   record §6.4 asks for, and every evaluated check's detail
   says so ("evaluated in SHADOW MODE (RFC SI-13 §6.4: verdict
   held at ESCALATE regardless of probe outcome…)") with the
   probe suite's artifact hash (§6.5 labelling).
7. **`run_cycle` does not construct a `ProbeContext`.** The
   autonomous loop gains no admission power from shadow probes
   (invariant 3); the context's consumers in this slice are the
   gate (via any caller that supplies it) and the SI-10
   admission path. Wiring the loop's transformer handling to it
   belongs to SI-17's scope, if SI-17 happens.
8. **Discovered blind spot, stated plainly (§5.6 evidence).**
   The per-class sensitivity test measures drift for a reference
   delta on **every one of the 30 targetable tensors**. 28 are
   measurable. The two exceptions are both layers'
   `self_attn.k_proj.bias`, with drift **exactly 0.0** — and
   that is mathematically necessary, not a harness bug: a
   constant key bias adds the same `q·b` to every position's
   attention score for a given query, and softmax is invariant
   to a constant shift across positions, so no input can ever
   detect it through this architecture's outputs. The test
   asserts the blind set is *exactly* those two tensors — any
   future change that blinds another tensor, or that makes
   these two visible (an architecture change), fails the test
   and must be explained. The §3.2 *classes* (LoRA 2-D,
   additive 1-D) are both fully measurable via their other
   members; the blind spot is one tensor per layer, named here
   rather than averaged away.

## Acceptance criteria (RFC §8 SI-16) — status

- ✅ Every §5.6 reference delta produces its required measured
  outcome (all through the real gate + `ProbeContext`):
  zero delta passes with drift **exactly 0.0** (and echo
  accuracy 1.0); benign delta (rank-2 LoRA on layer-0 `q_proj`,
  norm ≈ 0.19) passes with drift > 0 (measured ≈ 5.8e-05
  identity / ≈ 1.8e-04 regression on the author's run); a
  destructive rank-8 LoRA on `lm_head` (norm ≈ 0.51, inside all
  safety caps, so the probe check — not safety — is what fails)
  **fails check 3** with measured drift ≈ 1.02 > tol 0.25; a
  targeted echo attack (rank-8 LoRA on `lm_head` flipping the
  readout for 8 of 16 capability last-tokens) **fails check 4's
  accuracy arm**: accuracy subcheck measured **0.5** vs floor
  0.9375. (With this seed's ≈ 2.7-logit margins, any argmax flip
  also trips the regression arm — a flip needs a logit swing
  larger than the margin, which exceeds `capability_eps` by
  construction. Both arms are reported; the accuracy arm is
  named and measured, not inferred from the regression.)
  Per-class sensitivity: all measurable, with the documented
  `k_proj.bias` exception (decision 8).
- ✅ Binding-mismatch fixtures — wrong base hash (artifact's
  recorded tensor-map hash forged), wrong architecture
  (`d_model` 64 in the artifact), missing artifact (manifest
  record removed), tampered artifact hash (manifest's recorded
  hash forged) — **all yield "not evaluated" + ESCALATE, never
  a pass**. Plus the §6.3 sibling: tampered recorded baselines
  → startup validation fails → "not evaluated" + ESCALATE.
- ✅ A Glimmer-shaped fixture bundle (per-tensor base, no
  miniature kind, no artifact) behaves exactly as today —
  tested via `ProbeContext` on a synthetic seeded bundle and
  via the legacy non-probe context on the miniature itself
  (legacy string verbatim).
- ✅ Demo path journaled reports unchanged: an end-to-end
  `run_cycle` on `DemoMind` still ADMITs and its journaled gate
  report carries demo measured values with **no** probe-suite
  or shadow labelling and no subchecks (asserted); the full
  pre-existing demo/gate/loop suites are green (below).
- ✅ SI-10 admission of a shadow-escalated miniature delta
  journals the probe results with the admission: the admission
  self-edit's gate report carries the real measured checks and
  its `operator_admission` block carries `shadow_probe_results`
  (tested for a benign delta). Also tested: admission of a
  **probe-failed** (destructive) miniature delta — possible in
  shadow mode because the gate escalated rather than rejected —
  journals its measured identity failure (passed=False,
  measured > 0.25) alongside the admission, which is exactly
  the evidence trail SI-17's inadmissibility rule will consume.
- ✅ Full suite green on the author's own run: **270 passed,
  2 skipped** — the SI-15 baseline (252 passed, 2 skipped) plus
  this slice's 18 new tests, zero regressions. No `nedbd`
  flake occurred during this slice's runs (the pre-run `ps`
  check found no stale daemon).

## What is NOT in this slice

- **No SI-17 flip.** Probe-failed miniature deltas are *not*
  rejected and probe-passing ones are *not* admitted by the
  gate; operator admission (SI-10) remains possible for any
  shadow-escalated miniature delta, with its probe results
  journaled. Making probe-failed deltas inadmissible is
  SI-17's tested scope, gated on shadow evidence + a recorded
  decision that has not been taken.
- No probe VERIFY after materialisation (§6.6 is SI-17's);
  `transformer-apply-smoke` remains the transformer VERIFY,
  labelled as what it is.
- No Glimmer probes, no Glimmer forward path, no serve change
  (miniature bundles still get 501), no training, no delta
  authorship, no loop wiring (decision 7), no policy default
  changes.

## Terminology

Machine learning infrastructure. This document contains no
"custom LLM software" claims; the miniature is a constructed
machine-learning test instrument, not a trained model, and
probe results on it say something about deltas against the
miniature only — nothing about any other base.
