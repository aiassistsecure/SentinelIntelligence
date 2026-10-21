# SI-19 — Delta authorship for the miniature (as-built)

> **Scope:** RFC SI-18 §4 (candidate (b)), §7 (the SI-19
> outline), and §8 (the ten acceptance criteria — all ten are
> addressed below with their measurements; none is waived).
> `TinyTransformerMind.train_delta` now implements real
> CPU gradient-based training of ONE LoRA target on ONE named,
> deterministically constructed, probe-disjoint signal, and
> `propose` emits exactly that signal's deterministic
> transformer intent. Candidates are built through
> `construct_transformer_delta`; the gate, the probe suites,
> every policy default, and `si serve` are unchanged. Authored
> ADMIT was explicitly NOT required (RFC §8.8) — the gate's
> verdict on the authored candidate is reported below **as
> measured** (it is ADMIT, with the drift/accuracy numbers).

## What was built

- **`si/tiny_training.py`** (new module; SI-9a/SI-15
  new-module precedent): the training signal, the analytic
  gradient, the training loop, and candidate construction
  (`train_candidate`). `si/tiny_transformer.py`'s adapter
  delegates `propose` / `train_delta` to it; the SI-15 forward
  path is untouched.
- **`tests/test_tiny_training.py`** (new, 12 tests) and one
  updated assertion block in `tests/test_tiny_transformer.py`
  (SI-15's `train_delta`/`propose` refusal test now checks the
  SI-19 contract: an intent naming no signal still refuses;
  `propose` emits the named signal's intent).
- This doc. No policy default changed, no probe-suite change,
  no serve change (501 for every transformer bundle,
  miniature included — covered by the existing suite), the
  Glimmer seed untouched, the demo path byte-for-byte
  unchanged.

## The signal: `tiny-successor-lean-001` (RFC §4.3 decisions)

- **Target form: target logits (MSE)** — DemoMind's form,
  chosen and recorded per §4.3. Each example's target is the
  base's own last-position logits plus one small recorded
  correction, `v = 0.15 * (onehot(successor) - onehot(last))`,
  `successor = (last_token + 1) % 32` — the readout is asked
  to *lean toward the successor token*. Correction L2 norm
  `0.15 * sqrt(2) ≈ 0.2121`. Loss = mean over examples of the
  per-example mean squared error over the 32 logit dims.
- **Why not echo, and why not successor cross-entropy.** The
  base's echo accuracy is already 16/16 = 1.0 (SI-14) — echo
  cannot be the signal (the RFC's echo trap). A
  successor-token cross-entropy signal was measured during
  design: the base is equally imperfect there (held-out CE
  ≈ 3.87, accuracy 0/16), but flipping a ≈2.7-logit echo
  margin needs a delta norm far beyond the 1.0 policy cap, so
  at any gate-surviving size its held-out improvement is
  ~5%. The MSE successor-lean keeps the same successor
  direction with a §4.3-sized target correction (DemoMind's
  precedent: target scale recorded, small by construction),
  and its improvement at gate-surviving size is ~438x (below).
- **Construction.** One dedicated RNG stream,
  `default_rng(MINI_SEED + 11)` — distinct from the identity
  stream (`MINI_SEED + 2`) and capability stream
  (`MINI_SEED + 3`). It first draws 8 last tokens —
  `[28, 25, 31, 22, 3, 15, 1, 9]` — then draws sequences of
  lengths 2..8 with last tokens assigned round-robin from
  that set, rejecting duplicates and ANY sequence appearing
  in either probe suite. 48 examples: the first 32 are the
  training partition, the last 16 are **held out** (never
  trained on; improvement is claimed only there).
- **Base imperfection, recorded before training (§8.1).**
  The base's held-out loss on this signal is
  **0.00140625** (train partition identical), measured through
  the loss before any training step — and equal to the
  correction's closed form `2 * 0.15^2 / 32`, as it must be:
  the base is imperfect on the signal by exactly the
  correction it has not learned. A perfect score would be 0.

## Target and gradient route (RFC §4.4/§4.5 decisions)

- **One LoRA target: `lm_head.weight`**, rank **8**, alpha
  **8** (scale 1.0). Reasons: it is probe-measurable (the
  derived blind set is exactly the two `k_proj.bias`
  tensors; a test asserts the target is outside it); rank 8
  is the existing `rank_cap`; and as the final operation its
  gradient is *exactly* derivable — no gradient with respect
  to any frozen tensor is ever needed, so route (i)'s
  derivation has no downstream chain to get wrong.
- **Route (i) — analytic gradients.** With
  `W_eff = W + scale * B @ A` and `h_final` the last
  position's final RMSNorm output:
  `d loss/d logits = 2 * (logits - target) / 32`,
  `G = outer(d loss/d logits, h_final)` averaged over the
  batch, `d/dB = scale * G @ A.T`, `d/dA = scale * B.T @ G`.
  `h_final` does not depend on the readout, so it is computed
  once per sequence and cached — exactly, not approximately.
  The gradient needs `h_final`, which the SI-15 forward does
  not return, so `si/tiny_training.py` carries a forward
  replica that returns it; a test asserts the replica's
  logits are **bit-identical** (after the float32 cast) to
  the library forward's on all 24 probe inputs and all 48
  signal inputs, so the gradient is taken of the same
  function the gate measures.
- **Gradient check (§8.3).** Central finite differences of
  the same loss, computed through the existing forward (the
  float32 shared fold included), at the pinned initial
  factors over an 8-example subset, `eps = 0.01`, EVERY one
  of the 512 factor entries: **worst absolute disagreement
  3.807e-08**. The test's tolerance is pinned at **1e-5**
  (~260x the measured agreement, so platform float variation
  cannot flake it, and a wrong derivation — which disagrees
  at gradient magnitude, ~1e-3 — cannot pass it).

## Training run (measured on this VM, this slice's runs)

Full-batch gradient descent, learning rate 1.0, **2000
steps**, factors initialised `0.05 * standard_normal` from
`default_rng(MINI_SEED + 17)` (A drawn before B). A norm
guard projects any step whose scaled update would exceed
`||scale * B @ A||_F <= 0.9` back to the budget; it
**activated 0 times** in the pinned run (the trajectory
never approached the budget) — recorded in the
hyperparameters, never silent.

| Measurement (signal `tiny-successor-lean-001`) | Value |
|---|---|
| Base held-out loss (before training) | 0.00140625 |
| Base train-partition loss (before training) | 0.00140625 |
| Training loss, initial (init factors) | 0.0028961365437885785 |
| Training loss, final (step 2000, float64 path) | 1.824004756500067e-06 |
| **Held-out loss after training (float32 candidate)** | **3.2118398935406443e-06 (438x lower)** |
| Held-out examples improved | 16/16 |
| Candidate norm (`||scale * B @ A||_F`) | 0.23555380990005192 (caps: 1.0) |
| Candidate blob | 2,360 bytes (cap: 4 MiB) |
| Norm-guard activations | 0 |
| Wall time, one `train_delta` call | 0.99 s |
| Peak memory | process `ru_maxrss` 40,012 → 40,160 KiB across the call (interpreter baseline ~40 MB; training's marginal peak < 1 MB) |

Per-step cost basis (§8.10): each step is 32 cached-hidden
readouts in float64 (one `[32] x [32,32]` outer-product
accumulation per example plus the two factor-gradient
matmuls) — the transformer forward is NOT re-run per step,
because the readout's input cannot depend on the readout.
One-time costs per call: assembling the 30 applied tensors,
48 forward passes for signal targets, 32 for hidden states.

## Determinism, and one serialisation finding (§8.5)

Two `train_delta` runs on the same intent produce
**byte-identical candidate blobs** and identical records and
hyperparameters, except `wall_time_seconds` (the one
labelled non-deterministic field; no measured value depends
on it). Verified within one process and across processes
(identical blob sha256 in separate interpreter runs).

Getting there surfaced a real finding: safetensors' `save()`
emits the header's `__metadata__` mapping in hash-map order,
observed to **flip between two calls in the same process**
over bit-identical factor bytes — so two identical
constructions previously produced different blob hashes.
(SI-9a's contract never required cross-construction byte
identity; SI-19's §8.5 does.) The authored candidate is
therefore re-serialised with a pinned header (metadata keys
sorted, factor keys sorted) in `si/tiny_training.py`: the
tensor data bytes are the constructed blob's own, byte for
byte — only header key order is canonicalised — and the
rebound pair is run back through
`validate_transformer_delta` before it is returned.
`si/transformer_delta.py` itself is unchanged.

One recorded convention: the candidate record's
`created_at` is a fixed marker (`2026-10-21T00:00:00Z`), not
a wall-clock reading — §8.5 forbids wall-clock anywhere a
measured value could depend on it, and the loop re-stamps
id/parent/self-edit id and journals real timestamps when a
candidate lands.

## The gate's verdict on the authored candidate — as measured (§8.8)

No criterion requires ADMIT; this is the gate's separate,
recorded verdict on the candidate above, measured through a
fresh `ProbeContext` and reproduced exactly by the
autonomous loop's journal:

- **Verdict: ADMIT** (miniature bundle, policy allowlist
  opened by the test harness exactly as in SI-16/SI-17's
  tests — the bundle default remains an empty allowlist).
- Identity continuity: **measured-pass**, drift
  **0.17189218290150166** vs tol 0.25 (worst probe tiny-id-0).
- Capability floor: **measured-pass**, max regression
  **0.1657181391492486** vs eps 0.25 (worst tiny-cap-15);
  echo accuracy arm **1.0 (16/16)** vs floor 0.9375.
- End-to-end: `run_cycle` driven by the real
  `TinyTransformerMind` (not a script mind) ADMITs the
  candidate, materialises it as `d0001`, and the §6.6 probe
  VERIFY passes (journaled as `transformer-probe-verify`,
  measured-pass on both arms, echo accuracy 1.0). A test
  asserts the journaled verdict and measured values equal
  the independent `ProbeContext` evaluation of the same
  candidate, and that the landed delta's blob hash equals
  the independently trained candidate's.

## Refusals (§8.6)

Unknown `training_signal_id` → `ValueError` (DemoMind
precedent); a declared target list other than
`[lm_head.weight]` — blind-set targets included — →
`ValueError` (never a silently retargeted candidate); a
rank/alpha other than the pinned 8/8 → `ValueError`; a
trained candidate whose held-out loss does not improve →
`ValueError` (proven by test with the loss pinned to a
non-improving value); a mind that is not bundle-backed and
any non-miniature bundle → the pre-existing
`TinyTransformerError` refusals, unchanged.

## Acceptance criteria (RFC SI-18 §8) — status

1. ✅ **Signal validity** — deterministic construction
   (tested byte-identical across builds); base held-out
   score recorded before training: 0.00140625, measurably
   imperfect (perfect = 0.0); not the echo task.
2. ✅ **Disjointness** — test asserts train + held-out share
   no sequence with either probe suite (construction also
   rejects collisions during the draw); stream offset +11,
   distinct from +2/+3 (asserted).
3. ✅ **Gradient correctness** — route (i); worst agreement
   3.807e-08 over all 512 entries at eps=0.01; tolerance
   pinned at 1e-5 and tested.
4. ✅ **Real training, real improvement** — held-out
   0.00140625 → 3.2118398935406443e-06; training loss
   0.0028961365437885785 → 1.824004756500067e-06;
   non-improving results are refused by `train_delta`
   (tested).
5. ✅ **Determinism** — byte-identical blobs/records across
   repeat runs, in-process and cross-process (tested);
   only `wall_time_seconds` differs.
6. ✅ **Refusals intact** — unknown signal, blind/foreign
   targets, resized rank/alpha, non-bundle-backed and
   non-miniature all refuse (tested).
7. ✅ **Format and policy** — candidate validates through
   `construct_transformer_delta` /
   `validate_transformer_delta` against the verified
   manifest; exactly one allowlistable target; rank 8 ≤
   cap 8, norm 0.2356 ≤ 1.0, 2,360 bytes ≤ 4 MiB; no policy
   default changed.
8. ✅ **Gate honesty end to end** — real-mind cycle journals
   checks 3/4 genuinely measured (drift 0.17189218290150166,
   regression 0.1657181391492486, echo 1.0 vs floor 0.9375);
   verdict ADMIT + §6.6 VERIFY pass, reported as measured;
   journal-vs-independent-evaluation equality tested.
9. ✅ **No regressions, no widenings** — full suite green
   (see below); `si serve` still 501 for every transformer
   bundle; demo path unchanged; stored-bytes test: the
   base section is byte-identical after the authored delta
   lands and the blob store grows by exactly the delta's
   2,360 bytes.
10. ✅ **CPU budget published** — 2000 steps, per-step cost
    basis stated above, wall time 0.99 s per `train_delta`
    call on this VM, peak marginal memory < 1 MB.

Full suite on the author's own run: **292 passed,
2 skipped** in 66.85s — the SI-17 baseline (280 passed,
2 skipped) plus this slice's 12 new tests, zero regressions.
Focused runs: `tests/test_tiny_training.py` +
`tests/test_tiny_transformer.py`, 35 passed.

## What is still NOT claimed

- **No Glimmer anything.** No Glimmer forward path, probes,
  training, or authorship; the Glimmer seed is untouched and
  its deltas still escalate with checks 3/4 "not evaluated".
- **No authorship for any other base, target, or signal.**
  Exactly one signal, one target, one bundle kind. Additive
  targets and multi-target candidates remain unauthored.
- **No serving.** `si serve` still returns 501 for every
  transformer bundle, miniature included.
- **The trained behaviour is not claimed useful.** The
  successor-lean is a constructed task on a constructed
  model, present so "a mind wrote a delta that measurably
  does what it was trained to do, and the gate judged it"
  has a falsifiable meaning at this scale (RFC §9). The
  base remains a constructed, not trained, model; SI-19's
  deltas may honestly be described as "trained on a
  constructed signal" and nothing more (RFC §10.5).
- **Training numbers are training numbers.** Every loss
  above is labelled with the signal id and its partition;
  none of them is a gate or probe result (RFC §10.1).

## Terminology

Machine learning infrastructure. This document contains no
"custom LLM software" claims; the miniature is a constructed
machine-learning test instrument, and its authored deltas are
trained on a constructed signal — nothing here says anything
about any other base.
