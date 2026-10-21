# SI-21 — Sequential authored self-writes for the miniature (as-built)

> **Scope:** RFC SI-20 §9 (the SI-21 outline) and §10 (the ten
> acceptance criteria — each addressed below with its
> measurements; see criterion 6 for the one place where the
> measured outcome changes what could be exercised, reported
> exactly as §4.5's "or report what it measures instead"
> requires). One recorded freshness contract now governs
> which manifest trains; the SI-19 signal definition is
> unchanged; a second authored cycle was genuinely trained
> against the landed head `d0001` and genuinely judged by the
> gate. **Cycle-2 ADMIT was explicitly NOT required (RFC
> §10.4) — the gate's verdict on cycle 2 is REJECT, reported
> below as measured.** That REJECT is this slice's central
> result: the second write was really authored on top of the
> first, and the gate's cumulative identity accounting caught
> exactly the drift the RFC's §4.4 accounting predicted it
> would.

## What was built

- **`si/tiny_transformer.py`** — the freshness contract, in
  exactly one place: `TinyTransformerMind.refresh_manifest()`
  reloads the bundle's current verified manifest
  (`load_bundle`) and `train_delta` — its only caller — trains
  against that, never the construction-time snapshot.
- **`si/tiny_training.py`** — the target-anchoring semantics
  named in code (`TARGET_ANCHORING = "weights-under-training"`,
  module docstring) and recorded per cycle: `train_candidate`'s
  hyperparams now also carry `head`, `target_source_head`,
  `target_anchoring`, and `stack_depth` alongside the SI-19
  fields. The signal itself is untouched.
- **`si/loop.py`** — the miniature cycle journals a trained
  candidate's hyperparams as a `training-hyperparams`
  provenance note on both verdict paths (landed delta id on
  ADMIT, self-edit id on REJECT/ESCALATE), so a rejected
  cycle's training record survives in the journal. Candidates
  without hyperparams (scripted fixtures) journal nothing.
- **`tests/test_sequential_authorship.py`** (new, 10 tests)
  and this doc. No policy default changed, no probe-suite
  change, no tolerance moved, no serve change (501 for every
  transformer bundle), the Glimmer seed untouched, the demo
  path byte-for-byte unchanged.

## The freshness contract (§10.1)

**Contract: exactly one manifest trains in a cycle — the
bundle's current verified manifest.** Implemented in exactly
one place: `TinyTransformerMind.train_delta` calls
`refresh_manifest()`, which re-verifies and reloads the
manifest from the bundle on disk before delegating to
`train_candidate`. Every training path funnels through
`train_delta`, so every driver — the reused-mind
`run_cycles` shape, a fresh mind per cycle, a direct call —
trains at the same head the loop just verified. `propose`
already preferred the cycle context's manifest; training and
proposing now agree inside one cycle.

Proven by a test that can tell stale from fresh apart
(`test_train_delta_refreshes_a_stale_snapshot_directly`): a
mind constructed at the base *before* `d0001` lands trains
its next candidate at `head = d0001` (hyperparams assert it;
the blob differs from `d0001`'s). The control — deliberately
handing `train_candidate` the stale base manifest, Driver 1's
accident reproduced on purpose — re-derives SI-19's targets
and produces cycle 1's **byte-identical** candidate. The
distinction is therefore not asserted by construction; both
sides of it are measured in the same test.

## Target anchoring (§10.2) — the recorded choice

**Weights-under-training (head-relative).** Targets are
derived from the weight set under training — the bundle's
applied weights at the head being trained against — plus the
recorded successor-lean correction, exactly as
`build_training_signal` has always computed them. At head =
base this *is* SI-19's definition; at cycle 2 the readout is
asked to lean a further correction-step from where `d0001`
already leans. The alternative — **base-anchored** targets
(seed base's logits at every cycle, later cycles correcting
the residual toward a fixed destination) — was considered
and not chosen, per RFC §4.3.2: changing the signal
definition in the same slice that tests composition would
confound the experiment. The choice is named in code
(`si/tiny_training.TARGET_ANCHORING`), in the module
docstring, and in every cycle's hyperparams.

Consequence, stated plainly (§10.7): cycle 2's
pre-training held-out loss reads **0.00140625** again. That
figure recurs *by construction* — it is the correction's
closed form `2 · 0.15² / 32`, the loss of the weights under
training against targets derived from those same weights —
at any head. In cycle 1 it was a fact about the seed base;
in cycle 2 it is a head-relative construction artifact and
is **not** fresh evidence about the base. (The hyperparams
field keeps its SI-19 name `base_held_out_loss` for
continuity; at cycle ≥ 2 it means "the pre-training loss of
the weights under training, at their head" — the `head` and
`target_source_head` fields recorded beside it say which.)

## The two-cycle run (measured on this VM, this slice's runs)

Driven by the real `TinyTransformerMind` through
`si.run.run_cycles` with **one mind object reused across both
cycles** (the driver shape that was silently stale before
this slice). The test-harness policy opens the allowlist
exactly as in SI-19's tests; no default moved.

| | Cycle 1 | Cycle 2 |
|---|---|---|
| Head trained against | `base` | `d0001` |
| Target-source head | `base` | `d0001` |
| Stack depth at training | 0 | 1 |
| Candidate blob | `sha256:2761e8ff…6ca9e` (2,360 B) | `sha256:0f250005…2b7ea` (2,360 B) |
| Candidate norm | 0.23555380990005192 | 0.23555380625717862 |
| Train loss, initial → final | 0.0028961365437885785 → 1.824004756500067e-06 | 0.0028961362843063 → 1.8240042994718044e-06 |
| Pre-training held-out loss (at its head) | 0.00140625 | 0.00140625 (by construction — see above) |
| Held-out loss after training | 3.2118398935406443e-06 | 3.2118325217663162e-06 |
| Held-out examples improved | 16/16 | 16/16 |
| Norm-guard activations | 0 | 0 |
| `train_delta` wall time | 1.25 s | 1.28 s |
| Gate verdict | **ADMIT** → `d0001`, §6.6 VERIFY pass | **REJECT** (identity), nothing landed |
| Blob-store growth | +2,360 B | +0 B (candidate never written) |

Total two-cycle wall time **3.15 s**. Cycle 1 reproduces
SI-19 exactly — same blob `sha256:2761e8ffc139ce3e107c11da035e8a73c20792945a494f4e493dccf418b6ca9e`,
same gate numbers (§10.9). Cycle 2's candidate is genuinely
new (different blob, trained at `d0001`), and genuinely
small: as a *head-relative* improvement it succeeds (held-out
loss at its head falls ~438×, like cycle 1's). The gate does
not judge head-relative improvement; it judges the composed
head — and there the answer differs, which is the next
section.

## Cumulative vs incremental accounting (§10.5) — side by side

The gate's two probe checks accumulate differently (RFC
§4.4), and cycle 2 is where the difference first bites:

| Measurement | Cycle 1 | Cycle 2 | Accounting |
|---|---|---|---|
| Identity drift (gate, candidate overlay) | 0.17189218290150166 (pass, tol 0.25) | **0.34378434903919697 (FAIL, tol 0.25)** | **seed-absolute / cumulative** — drift of the would-be head vs the seed's recorded logits |
| Capability regression (gate) | 0.1657181391492486 (pass, eps 0.25) | 0.16571810841560364 (pass, eps 0.25) | **head-incremental** — charged only since the pre-cycle head |
| Echo accuracy | 1.0 (16/16, floor 0.9375) | 1.0 (16/16, floor 0.9375) | absolute, vs the recorded floor |
| Landed head's cumulative identity drift after the cycle | 0.17189218290150166 (§6.6 VERIFY, measured-pass) | 0.17189218290150166 (unchanged — re-measured after the REJECT) | seed-absolute, of the head that actually landed |

No incremental figure above is presented as cumulative or
vice versa. The story they tell together: the second
authored update is, by itself, the same size and quality as
the first (incremental arms pass, held-out improves) — and
the composed head has spent the identity budget twice
(cumulative arm fails). Identity continuity, not capability,
is the binding constraint on sequential authorship of this
signal, exactly as RFC §4.4's accounting said it would be.

### The §4.4 derivation: CONFIRMED by measurement

RFC §4.4 derived — labelled a derivation, from SI-19's
published drift and the architecture — that identically
stacking cycle 1's candidate would double the drift:

- **Predicted (derivation):** 2 × 0.17189218290150166 =
  **0.34378436580300332** (the RFC's rounded 0.3438).
- **Measured:** **0.34378437884151936** — the stale-manifest
  candidate (byte-identical to `d0001`) evaluated through a
  fresh `ProbeContext` against head `d0001`, verdict
  **REJECT**, identity `measured-fail`, capability
  `measured-pass`.

Agreement to 1.3e-08 — the float32 fold's last digits, the
exact caveat the RFC attached. The derivation is confirmed,
not refuted. A second, independent observation sits beside
it: the *genuinely retrained* cycle-2 candidate's cumulative
drift (0.34378434903919697) lands within 3e-08 of the
identical-stack figure. That is measured, and its reason is
visible in the setup — the head-relative signal asks for the
same correction from the head, over hidden states the
readout cannot change, so the trained second update is
nearly the first update again. This near-equality is a
property of *this* signal's anchoring, recorded here as a
measurement, not claimed as a general law.

## VERIFY and unwind (§10.6)

- **Cycle 1** ADMITted and passed the §6.6 probe VERIFY
  exactly as in SI-19 (journaled `transformer-probe-verify`,
  measured-pass on both arms, echo 1.0, baseline labelled
  "pre-cycle head (caller-captured)").
- **Cycle 2** was REJECTed at the gate and never
  materialised, so there is no cycle-2 VERIFY and no VERIFY
  rollback — nothing landed to verify or roll back. The
  would-be VERIFY rollback target, had it landed and failed,
  is `d0001` by the miniature branch's existing construction
  (`to=prev_head`); that path is unchanged and remains
  covered by SI-17's injected-failure test.
- **Unwind of the landed stack, exercised:** because cycle 2
  REJECTed, no *authored* two-delta stack exists to unwind —
  so the two-step granularity is proven on a stack of the
  authored `d0001` plus a benign **fixture** `d0002`
  (hand-built, SI-17 style, journalled as a fixture — it is
  not authored and is never described as such), rolled back
  through the existing rollback core, each restored head's
  probe state **re-measured** via `_verify_probe_head`:
  - at the two-delta head: VERIFY pass; max identity drift
    0.17189218290150166 — measured, and equal to the authored
    head's, because the max is set by `tiny-id-0`, where the
    fixture's contribution does not exceed the authored
    delta's (the composed stack is instead proven by the
    fixture target's applied tensor differing, asserted in
    the test);
  - rolled back to `d0001`: VERIFY pass, max identity drift
    **0.17189218290150166** (the authored state, exactly),
    echo 1.0, `d0002` status `rolled_back`, fixture fold gone
    from the applied tensor;
  - rolled back to base: VERIFY pass, max identity drift
    **0.0**, max capability regression **0.0**, `d0001`
    status `rolled_back`.

## Acceptance criteria (RFC SI-20 §10) — status

1. ✅ **Freshness proven** — the reused-mind two-cycle run
   trains cycle 2 at `d0001`; the stale-snapshot control in
   the same test file reproduces cycle 1's byte-identical
   candidate, so stale and fresh are told apart by
   measurement (see "The freshness contract").
2. ✅ **Semantics recorded** — anchoring named in code and
   above; per-cycle hyperparams record `head`,
   `target_source_head`, `target_anchoring`, `stack_depth`
   in the return value *and* in the journal
   (`training-hyperparams` provenance, both verdict paths,
   asserted for both cycles); signal definition byte-for-byte
   SI-19's (id, stream +11, correction 0.15, 32/16 partitions
   — pinned by test and by SI-19's unchanged suite).
3. ✅ **Determinism per cycle** — a full two-cycle run
   repeated from the seed reproduces both candidates'
   blobs/records, the landed delta record, and the journaled
   gate measurements exactly; hyperparams equal except the
   labelled `wall_time_seconds`.
4. ✅ **Gate honesty per cycle** — cycle 2's journaled
   checks 3/4 equal an independent `ProbeContext` evaluation
   of the same candidate against `d0001` (abs 1e-12); cycle
   2's verdict is **REJECT, as measured**. No ADMIT was
   required; composition is proven by genuine authorship
   against the landed head plus genuine judgement.
5. ✅ **Cumulative accounting published** — the side-by-side
   table above; the §4.4 derivation's predicted 0.3438 and
   its measured 0.34378437884151936 published together:
   **derivation CONFIRMED**.
6. ⚠️ **VERIFY and unwind — exercised as far as the measured
   outcome allows** — cycle-1 VERIFY as in SI-19; cycle 2
   never landed (REJECT), so no cycle-2 VERIFY exists to
   pass or fail — stated, not papered over. Unwind to
   `d0001` and to base exercised through the rollback core
   with re-measured probe states, on an authored-`d0001` +
   fixture-`d0002` stack (fixture labelled; no authored
   two-delta stack exists, because cycle 2 REJECTed).
7. ✅ **Held-out honesty in cycle 2** — improvement claimed
   only against cycle 2's own pre-training loss at `d0001`;
   the recurring 0.00140625 is labelled a by-construction
   artifact of head-relative target derivation, in the
   anchoring section above and in the test that asserts it.
8. ✅ **Depth and policy untouched** — exactly two authored
   cycles attempted; `max_stack_depth` 8, `identity_tol`
   0.25, `capability_eps` 0.25, empty default allowlist —
   all asserted unchanged; `si squash` still fails closed
   on the resulting transformer stack (asserted here;
   SI-12's suite green).
9. ✅ **No regressions, no widenings** — full suite green
   (below); cycle 1 as a first cycle reproduces SI-19's
   blob and gate numbers exactly; demo path unchanged;
   `si serve` still 501 for every transformer bundle
   (existing suite); no CLI wiring added.
10. ✅ **CPU budget published** — per-cycle `train_delta`
    wall times 1.25 s / 1.28 s; total two-cycle run 3.15 s;
    full suite runtime below. All on this VM, CPU only.

Full suite on the author's own run: **302 passed,
2 skipped** in 111.63s — the SI-19 baseline (292 passed,
2 skipped) plus this slice's 10 new tests, zero regressions.
Focused run: `tests/test_sequential_authorship.py`, 10 passed.

## What is still NOT claimed

- **No third cycle, no continual learning.** Two authored
  cycles on a constructed signal, on a constructed model,
  prove the loop's composition machinery and the gate's
  cumulative accounting — nothing about learning at any
  other scale or on any real task (RFC §11).
- **Cycle 2 did not land.** The bundle head after an SI-21
  run is `d0001`, by the gate's measured REJECT. Nothing in
  this doc describes the cycle-2 candidate as accepted,
  learned into the bundle, or verified (§12.3).
- **No Glimmer anything.** No Glimmer forward path, probes,
  training, or authorship; the Glimmer seed is untouched and
  its deltas still escalate with checks 3/4 "not evaluated".
- **No widening.** One signal, one target, one bundle kind;
  no base-anchored variant was built — it is recorded as the
  considered alternative, and a future slice that wants
  cycle-2 ADMITs must argue for new semantics or a new
  signal, never a wider gate (RFC §11).
- **No serving, no CLI wiring.** Transformer bundles keep
  returning 501.
- **Training numbers are training numbers.** Every loss
  above is labelled with the signal id's partitions and its
  head; none of them is a gate or probe result (§12.1).

## Terminology

Machine learning infrastructure. This document contains no
"custom LLM software" claims; the miniature is a constructed
machine-learning test instrument, its authored deltas are
trained on a constructed signal, and its second authored
delta was judged — and rejected — on cumulative identity
drift. Nothing here says anything about any other base.
