# SI-15 — Forward path (as-built)

> **Scope:** RFC SI-13 §4 (forward path design) and §8's SI-15
> slice. This slice builds a **library capability**: a pure-numpy
> forward for the SI-14 miniature, weight assembly through
> apply-at-load, and the §4.2 candidate overlay with shared fold
> arithmetic. It wires the forward to **nothing** — no gate, no
> loop, no serve. Gate behaviour is byte-for-byte unchanged: every
> transformer delta, miniature ones included, still reports checks
> 3/4 as "not evaluated" and still ESCALATES (asserted by test).
> Probes are still not consumed by the gate; shadow mode is SI-16,
> any admission change is SI-17, and `si serve` still returns 501
> for every transformer bundle.

## What was built

- **`si/tiny_transformer.py`** (new): `forward(token_ids, weights)`
  per the §4.1 contract (float32 logits `[len, 32]`); weight
  assembly `assemble_weights` / `assemble_weights_with_candidate` /
  `forward_from_bundle`; architecture validation
  (`validate_architecture`); and the `TinyTransformerMind` adapter
  (forward/probe surface only — see §4.4 note below).
- **`si/apply.py`** (refactored, behaviour preserved): the
  per-target fold arithmetic is extracted into one shared function,
  `fold_target`, and the §4.2 candidate overlay
  `applied_tensor_with_candidate` is added beside
  `load_applied_tensor` (re-exported from `si/tiny_transformer`).
- **`tests/test_tiny_transformer.py`** (new): 23 tests covering
  every §8 SI-15 acceptance criterion (below).
- This doc. No gate, loop, serve, bundle, seed, or policy code was
  changed; `git diff` on those paths is empty.

## As-built decisions (each explicit)

1. **Module location: RFC §4.1's proposal, confirmed.**
   `si/tiny_transformer.py` is a new module, for the reasons the
   RFC gave and the SI-9a/SI-9c precedent follows: the gate/bundle/
   seed layers gain no dependency on forward internals, and tests
   plus a future probe context (SI-16) can import the forward
   without importing the gate or the loop. SI-14's seed-time helper
   (`si/tiny_seed.py::tiny_forward`) is **not** extracted or
   replaced: it stays the seed path's internal construction aid,
   and this module instead *proves* agreement with its recordings
   (decision 3). Two implementations existing is a deliberate,
   bounded cost: the seed helper is frozen evidence of how the
   baselines were produced, and the cross-check between them is a
   test, not a hope.
2. **The 29 → 30 tensor note.** RFC §4.2 says the applied weight
   set takes "twenty-nine calls" to `load_applied_tensor`. That
   text predates SI-14's positional-table decision (RFC §3.3 left
   the choice to SI-14, which included the table). The miniature's
   manifest tensor map has **30 tensors**, so assembly makes 30
   calls, and the forward requires exactly those 30 tensors at
   their pinned shapes. The RFC's budget conclusion is unchanged.
3. **Arithmetic discipline (§4.3) — and what it proves.**
   - Weights and delta factors are **float32**: that is the dtype
     the apply path delivers and the shared fold computes in, and
     `forward` casts any input weight set to float32 first, so no
     caller can smuggle in a wider weight dtype.
   - The forward computation widens the float32 weights to
     **float64** and computes in float64 (same widening, same
     operation order as SI-14's seed-time helper), returning
     logits cast to float32 per the §4.1 contract. This follows
     the runtime's existing precedents: the gate's `_err` measures
     in float64, `DemoMind.forward` computes in float64 over a
     float32 base, and the gate's norm recomputation is float64.
   - **Proven consequence, claimed only because a test asserts
     it:** the library forward reproduces SI-14's recorded
     probe-suite expected logits for the seeded base
     **bit-identically** — all 24 entries (8 identity + 16
     capability), `array_equal`, no tolerance
     (`test_forward_reproduces_recorded_probe_logits_bit_identically`).
     SI-14's helper measured in float64 and stored a float32 cast;
     matching its discipline makes the recorded baselines exactly
     reproducible by the library path, which is what SI-16's
     startup validation (§6.3) will rely on. At 23,456 parameters
     the float64 cost is noise (§4.5), so there is no speed
     argument for a float32 compute path that would drift from
     the recorded baselines by ~1e-7 and need a tolerance story.
   - No randomness at forward time (no sampling, no dropout, no
     RNG); repeat runs are bit-identical (tested, pure and
     bundle-backed).
4. **Shared fold arithmetic — one fold, not a third copy.**
   Before SI-15, `load_applied_tensor` folded each target inline;
   SI-11's export materialise path already mirrors that arithmetic
   once (its own docstring records the duplication as a cost).
   SI-15 extracts the apply path's fold into
   `si.apply.fold_target` — the exact prior arithmetic
   (`result + (alpha/rank)·B@A` with the update cast to float32
   before the add; `result + d` for additive; the additive policy
   refusal and post-fold shape check move with it, messages
   unchanged) — and both `load_applied_tensor` and the new overlay
   call it. The overlay therefore cannot drift from the chain
   path, and the §8 overlay-vs-materialised test proves agreement
   exactly rather than to a tolerance. (Export's pre-existing
   mirror is untouched; reconciling it onto `fold_target` is a
   possible future tidy, not part of this slice.)
5. **Candidate overlay semantics (§4.2), as implemented.**
   `applied_tensor_with_candidate(bundle_dir, manifest, name,
   candidate, policy=None)`:
   - starts from `load_applied_tensor` (base + active chain);
   - for a name the candidate does not target, returns that result
     untouched — the candidate blob is not read and not validated
     by the call (one manifest lookup; tested, including with a
     deliberately tampered candidate blob);
   - for a targeted name, runs SI-9a
     `validate_transformer_delta` against the caller's verified
     manifest **first** (a tampered candidate raises
     `DeltaValidationError`, never a silent fold), then folds the
     candidate's factor(s) via `fold_target`, with
     `allow_additive` resolved exactly as the chain path resolves
     it (tested: additive overlay refused by default, and equal
     to the hand-computed `w + d` exactly when policy allows);
   - is read-only: overlaying a candidate lands nothing (tested:
     the bundle head is still `base` afterwards).
6. **Architecture refusal (§6.2 rule 3).** The forward refuses any
   descriptor that is not SI-14's recorded §3.1 descriptor exactly
   — missing key, extra key, or any differing value (parametrised
   tests mutate vocab, d_model, layers, heads, head_dim, d_ff,
   max_seq_len, positional_embedding, rms_norm_eps,
   embeddings_tied). The pure `forward` additionally refuses any
   weight set that is not exactly the 30-tensor inventory at its
   pinned shapes, and any token sequence outside
   `1 <= len <= 16`, ids in `[0, 32)`. This is a miniature
   executor, not a general one.
7. **The adapter (§4.4).** `TinyTransformerMind` is bundle-backed
   and exposes the probe/forward surface a future context needs:
   `forward` / `forward_candidate` (the §4.2 overlay),
   `identity_probes()` / `capability_suite()` /
   `capability_baseline()` read from the bundle's recorded probe
   artifact (fail closed if absent). `train_delta` **refuses
   explicitly** (`NotImplementedError`: inference-only, no
   gradient/optimiser/authorship code anywhere in this slice), and
   `propose` refuses the same way rather than fabricate an intent.
   Nothing constructs this adapter in production paths in SI-15.

## Acceptance criteria (RFC §8 SI-15) — status

- ✅ Hand-computed reference cases match **exactly**: an
  identity-block case (all branches zero; expected derived in the
  test with `math.sqrt` alone) and a single-active-layer case
  (layer 0's v/o path; single-token softmax is exactly 1, the
  contribution is one hand-derived scalar) — both `array_equal`.
- ✅ Multi-layer outputs match an independent naive
  reimplementation written in the test file from §4.1 (explicit
  Python loops over positions/heads, `math` only — not the
  module's code), within the stated tolerance: the module returns
  float32 logits while the naive path stays in float64, so
  agreement is bounded by float32 output rounding (observed max
  diff ~1e-7; asserted at rtol/atol 1e-6).
- ✅ Repeat runs are bit-identical (pure forward, bundle forward,
  and weight assembly).
- ✅ Overlay-vs-materialised agreement is **exact**: a fixture
  LoRA delta on `layers.0`'s q_proj applied both ways gives
  `array_equal` weights (all 30 tensors) and `array_equal` logits.
- ✅ Rolled-back deltas do not affect the forward: head moved back
  through a two-delta stack reproduces the one-delta and base
  forwards exactly.
- ✅ The forward refuses any architecture descriptor other than
  §3.1's (parametrised, 10 mutations + missing/extra key cases).
- ✅ Gate behaviour byte-for-byte unchanged: a valid transformer
  delta against the real miniature bundle still gets checks 3/4
  "not evaluated" (`measured=None`) and ESCALATE, with the gate's
  probe methods rigged to raise if touched — after the new
  forward path has been exercised on the same bundle.
- ✅ (Supporting) Recorded probe logits reproduced bit-identically
  (decision 3); candidate validation-first and untargeted
  discipline; additive overlay policy; adapter refusals.

Full suite on the author's own run: **252 passed, 2 skipped** —
the SI-14 baseline (229 passed, 2 skipped) plus this slice's 23
new tests, zero regressions.

## What is NOT in this slice

- No gate wiring and no shadow mode (SI-16); probes are still not
  consumed by the gate, and no probe result exists anywhere in a
  journal from this code.
- No admission change (SI-17); transformer deltas still ESCALATE
  and SI-10 remains the only admission path for them.
- No serve change: transformer bundles, miniature included, still
  return 501 from `si serve`. This forward path is not exposed on
  any HTTP surface.
- No training, no delta authorship, no Glimmer forward path of
  any kind.

## Terminology

Machine learning infrastructure. This document contains no
"custom LLM software" claims; the miniature is a constructed
machine-learning test instrument, not a trained model, and the
forward path over it says nothing about any other base.
