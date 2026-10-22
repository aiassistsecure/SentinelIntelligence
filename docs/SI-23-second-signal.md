# SI-23 — A second authored signal for the miniature (as-built)

> **Scope:** RFC SI-22 §9 (the SI-23 outline) and §10 (the ten
> acceptance criteria — each addressed below with its
> measurements). A signal registry now serves exactly two
> named signals; the second, `tiny-predecessor-lean-001`,
> was authored against the landed head `d0001` under the
> unchanged freshness contract and weights-under-training
> anchoring, and was genuinely judged by the unchanged gate.
> **Cycle-2 ADMIT was explicitly NOT required (RFC §10.7) —
> the gate's verdict on the decisive cycle 2 is REJECT,
> reported below as measured.** That REJECT is this slice's
> central result, and the §5.6.4 diagnostics below attribute
> it: at the worst composed probe (`tiny-id-3`) the two
> updates' movements *align* (positive inner product,
> majority sign agreement), so the composed drift adds
> almost exactly — the predecessor direction did not
> translate into opposing movement on the identity probes,
> exactly the failure mode RFC §5.6.4 said could not be
> argued away in advance.

## What was built

- **`si/tiny_training.py`** — the signal registry
  (`SIGNAL_REGISTRY`, `signal_spec`): `train_candidate`
  dispatches on `intent.training_signal_id` through the
  registry; any id outside the two named signals refuses
  with the existing `ValueError` discipline, and the
  foreign-target / foreign-rank-alpha refusals apply per
  signal, unchanged. Signal-parameterised construction:
  `signal_sequences(signal_id)`,
  `build_training_signal(weights, signal_id)`, and
  `initial_factors(signal_id)` — each defaulting to signal
  1, byte-for-byte SI-19's behaviour (proven by criterion
  9). Signal 2's new stream constants sit beside the
  existing pair (offset table below). Its rejection-draw
  construction rejects against the union of both probe
  suites, signal 1's full 48 sequences, and its own draws.
- **`si/tiny_transformer.py`** — recorded selection:
  `TinyTransformerMind` takes a constructor-level
  `training_signal_id` (default `tiny-successor-lean-001`),
  validated through the registry, and
  `set_training_signal(...)` lets a driver explicitly
  reconfigure the same mind between cycles; `propose`
  emits exactly the configured signal's intent. No
  autonomous, learned, depth-driven, or alternating
  selection exists anywhere in the slice — the mind does
  not choose; the driver configures, and the journal
  shows it.
- **`tests/test_second_signal.py`** (new, 13 tests) and
  this doc. No policy default changed, no probe-suite
  change, no tolerance moved, no anchoring change, no
  serve change (501 for every transformer bundle), no CLI
  wiring, the Glimmer seed untouched, the demo path
  unchanged, signal 1's definition byte-for-byte SI-19's.

## The second signal: `tiny-predecessor-lean-001` (RFC §5.6.2)

- **Form:** target-logits MSE, exactly signal 1's loss
  form. Each example's target is the weights-under-training's
  own last-position logits plus one recorded correction,
  `v = 0.15 * (onehot(predecessor) - onehot(last_token))`
  with `predecessor = (last_token - 1) % 32` — the readout
  is asked to *lean toward the predecessor token*.
  Correction scale **0.15, unchanged** — the family's
  recorded design, not a knob turned for this slice
  (RFC §5.6.1's gate-fitting trap is a stated non-goal).
- **Same shape as signal 1, deliberately:** target
  `lm_head.weight`, rank 8, alpha 8, 2,000 full-batch
  steps, learning rate 1.0, norm budget 0.9, 32 train /
  16 held-out examples, 8 round-robin last tokens. The
  only differences between the two signals are the
  correction direction, the stream, and the drawn
  sequences — so the two cycles' measurements are
  commensurable and any outcome is attributable.
- **Streams audited (§10.3).** The full miniature offset
  table, all six offsets pairwise distinct (asserted by
  the extended guard test):

  | Offset | Stream | Where |
  |---|---|---|
  | +2 | identity probe inputs | `si/tiny_seed.py` |
  | +3 | capability probe inputs | `si/tiny_seed.py` |
  | +11 | signal 1 sequences (`SIGNAL_SEED_OFFSET`) | `si/tiny_training.py` |
  | +17 | signal 1 factor init (`INIT_SEED_OFFSET`) | `si/tiny_training.py` |
  | +23 | signal 2 sequences (`SIGNAL2_SEED_OFFSET`) | `si/tiny_training.py` |
  | +29 | signal 2 factor init (`INIT2_SEED_OFFSET`) | `si/tiny_training.py` |

  Neither signal-2 value reuses +2, +3, +11, or +17, and
  signal 2's initial factors measurably differ from
  signal 1's (different stream, asserted by test).

## Disjointness (§10.4) — proven by construction and by test

Signal 2's draws reject against the union of both probe
suites, signal 1's full 48 sequences, and signal 2's own
draws so far. All five §5.4 obligations are asserted by
test: (1) vs the 8 identity inputs — disjoint; (2) vs the
16 capability inputs — disjoint; (3) vs signal 1's full
48 sequences (32 train + 16 held-out) — disjoint; (4)
signal 2's own 32 train / 16 held-out partitions are
pairwise distinct (48 unique sequences); (5) last-token
overlap is a **measured fact, published**: signal 1's set
is `[28, 25, 31, 22, 3, 15, 1, 9]`, signal 2's set is
`[30, 12, 2, 22, 19, 0, 18, 9]` — the overlap is exactly
**{9, 22} (2 of 8)**. Sequences, not tokens, are the unit
of disjointness; the diagnostics below are read against
this measured overlap.

## Echo-trap records (§10.5)

Signal 2's pre-training held-out loss, recorded before
training at each head (a training number, labelled with
its signal id and head wherever it appears):

| Record | Head / target source | Measured | Label |
|---|---|---|---|
| At base | `base` / `base` | **0.00140625** | measurement — **CONFIRMS** the §5.6.3 closed-form derivation `2 * 0.15² / 32` |
| At `d0001` | `d0001` / `d0001` | **0.00140625** | measurement of a **by-construction recurrence**: under weights-under-training anchoring the targets derive from `d0001`'s own logits, so the zero-delta loss is the full correction's loss at any head — **not fresh evidence about the base** |

What the at-`d0001` record proves is the narrow, sufficient
thing: signal 2 is not already satisfied at `d0001` in the
only sense authorship requires — its zero-delta loss there
is not ~0.

## Gradient check for signal 2 (§10.2)

Central finite differences of signal 2's loss through the
existing SI-15 forward (the float32 shared fold included),
at the pinned initial factors for signal 2 over an
8-example subset, `eps = 0.01`, EVERY one of the 512
factor entries — run in **both** configurations:

| Configuration | Worst absolute disagreement | Tolerance |
|---|---|---|
| Signal 2 at base | **5.5249277028738815e-08** | 1e-5 (SI-19's pin, unchanged) |
| Signal 2 at head `d0001` (the decisive configuration) | **5.062663472655505e-08** | 1e-5 |

Both pass with ~180–200× headroom, at the same order as
signal 1's published 3.807e-08. No tolerance was loosened.

## Signal 2 at base: the first-cycle sanity configuration (§9.3)

Configured with signal 2, a first cycle at base trains and
is judged there. **Verdict as measured: ADMIT** → `d0001`,
§6.6 VERIFY pass. These are at-base facts about signal 2's
definition; they are not the slice's decisive claim and
are not evidence about head `d0001`.

| Measurement (signal `tiny-predecessor-lean-001`, head `base`) | Value |
|---|---|
| Pre-training held-out loss | 0.00140625 |
| Training loss, initial → final | 0.002604702300948902 → 3.362224144792228e-06 |
| Held-out loss after training | 3.891045992787381e-06 (~361× lower) |
| Held-out examples improved | 16/16 |
| Candidate blob | `sha256:131cc242728024b4f833bc78ea627825f1e8616972fc2670603673e1814645f4` (2,360 B) |
| Candidate norm | 0.21317186144376168 |
| Norm-guard activations | 0 |
| Identity drift (gate) | 0.16672966815531254 ≤ tol 0.25 (worst: tiny-id-7) — pass |
| Capability regression (gate) | 0.2418281203135848 ≤ eps 0.25 (worst: tiny-cap-10) — pass |
| Echo accuracy | 1.0 (16/16) vs floor 0.9375 — pass |
| `train_delta` wall time | 1.04 s |
| Blob-store growth | +2,360 B |

## The decisive two-cycle run (measured on this VM, this slice's runs)

Driver shape: **one reused `TinyTransformerMind`** driven
through `si.run.run_cycles` — one `run_cycles` call for
cycle 1 (signal 1, the default selection), then the driver
explicitly reconfigures the same mind to signal 2
(`set_training_signal` — the recorded selection), then a
second `run_cycles` call for cycle 2. Exactly two authored
cycles were attempted.

| | Cycle 1 | Cycle 2 |
|---|---|---|
| Signal (recorded selection) | `tiny-successor-lean-001` (default) | `tiny-predecessor-lean-001` (explicit) |
| Head trained against | `base` | `d0001` |
| Target-source head | `base` | `d0001` |
| Stack depth at training | 0 | 1 |
| Candidate blob | `sha256:2761e8ff…6ca9e` (2,360 B) | `sha256:184cc7f0…6e6cf7` (2,360 B) |
| Candidate norm | 0.23555380990005192 | 0.21317186599953494 |
| Train loss, initial → final | 0.0028961365437885785 → 1.824004756500067e-06 | 0.0026047023050908037 → 3.3622214870436204e-06 |
| Pre-training held-out loss (at its head) | 0.00140625 | 0.00140625 (by construction — see echo-trap records) |
| Held-out loss after training | 3.2118398935406443e-06 | 3.891045006596239e-06 |
| Held-out examples improved | 16/16 | 16/16 |
| Norm-guard activations | 0 | 0 |
| `train_delta` wall time | 1.00 s | 0.98 s |
| Gate verdict | **ADMIT** → `d0001`, §6.6 VERIFY pass | **REJECT** (identity), nothing landed |
| Blob-store growth | +2,360 B | +0 B (candidate never written) |

Full candidate blob hashes: cycle 1
`sha256:2761e8ffc139ce3e107c11da035e8a73c20792945a494f4e493dccf418b6ca9e`;
cycle 2
`sha256:184cc7f06cd9cb3bd83fbed466b6a939c7e2e18cf7b8a72be70c51043a6e6cf7`.
Total two-cycle wall time **2.35 s** (both `run_cycles`
calls, gate included, on this VM). Cycle 1 reproduces
SI-19/SI-21 **exactly** — same candidate blob, same gate
numbers (§10.9) — so the registry refactor is proven
behaviour-preserving for the existing signal. Cycle 2's
candidate is genuinely new (different blob, trained at
`d0001` with signal 2's own stream, init, and sequences),
and as a *head-relative* improvement it succeeds (held-out
loss at its head falls ~361×). The gate does not judge
head-relative improvement; it judges the composed head.

## Cumulative vs incremental accounting (§10.8) — side by side

| Measurement | Cycle 1 | Cycle 2 | Accounting |
|---|---|---|---|
| Identity drift (gate, candidate overlay) | 0.17189218290150166 (pass, tol 0.25) | **0.2963416576385498 (FAIL, tol 0.25)** | **seed-absolute / cumulative** — drift of the would-be head vs the seed's recorded logits |
| Capability regression (gate) | 0.1657181391492486 (pass, eps 0.25) | 0.15023326850496233 (pass, eps 0.25) | **head-incremental** — charged only since the pre-cycle head |
| Echo accuracy | 1.0 (16/16, floor 0.9375) | 1.0 (16/16, floor 0.9375) | absolute, vs the recorded floor |
| Landed head's cumulative identity drift after the cycle | 0.17189218290150166 (§6.6 VERIFY, measured-pass) | 0.17189218290150166 (unchanged — re-measured after the REJECT) | seed-absolute, of the head that actually landed |

No incremental figure above is presented as cumulative or
vice versa. Identity continuity remains the binding
constraint — now measured for a *second, differently
directed* signal as well as for the same signal restacked
(SI-21).

## The §5.6.4 diagnostics (published whatever the verdict)

All diagnostics below are computed against head `d0001`
with the decisive cycle-2 candidate overlaid. They are
published to *attribute* the gate's verdict after the gate
has spoken; they are not gate checks and influenced no
verdict (RFC §12.12). Per-probe drift values are
measurements (max absolute logit coordinate per probe);
the incremental movement column is a **derivation** from
two measured forwards (composed forward minus the
`d0001` forward), labelled as such.

**(i) Per-probe drift contributions:**

| Probe | `d0001` drift (measured) | Signal-2 incremental movement (derivation) | Composed drift (measured) |
|---|---|---|---|
| tiny-id-0 | 0.17189218290150166 | 0.100458356551826 | 0.2092923242598772 |
| tiny-id-1 | 0.15088725090026855 | 0.07999316975474358 | 0.15134692192077637 |
| tiny-id-2 | 0.11858986946754158 | 0.14863967895507812 | 0.14906740188598633 |
| tiny-id-3 | 0.14782869815826416 | 0.14852380752563477 | **0.2963416576385498** |
| tiny-id-4 | 0.11720322235487401 | 0.15136146545410156 | 0.23394920490682125 |
| tiny-id-5 | 0.09891846403479576 | 0.11731460690498352 | 0.14004023000597954 |
| tiny-id-6 | 0.12363102938979864 | 0.14838377758860588 | 0.1332004750147462 |
| tiny-id-7 | 0.08795571327209473 | 0.16672981530427933 | 0.2018739078193903 |

Worst probes, named: `d0001`'s own drift — **tiny-id-0**
(0.17189218290150166); the incremental movement —
**tiny-id-7** (0.16672981530427933); the composed drift —
**tiny-id-3** (0.2963416576385498, equal to the gate's
measured cycle-2 drift). At tiny-id-3 the composed drift
is, to the printed digits, the *sum* of the two
contributions (0.14782869815826416 + 0.14852380752563477
≈ 0.2963) — the aligned extreme of the §5.6.4 range, at
that probe.

**(ii) Alignment diagnostic (derivation):** the two
updates' movement vectors are `m1` = (`d0001` forward −
seed expected) and `m2` = (composed forward − `d0001`
forward), per identity probe.

- At the worst composed probe (tiny-id-3): inner product
  **+0.021769049613635714** (positive — aligned, not
  cancelling), sign agreement **0.625** (20/32 coordinates).
- Aggregated over all 8 identity probes (256 coordinates):
  inner product **+0.07637087369869666**, sign agreement
  **0.54296875** (139/256); movement-vector norms
  ‖m1‖ = 0.7119678353441855, ‖m2‖ = 0.6234748221249689.

**(iii) Folded updates (measured):** Frobenius inner
product of the two folded `lm_head.weight` updates
**+0.004356018453758104** (positive — aligned in weight
space as well); norms ‖Δ1‖_F = 0.23555380990005192,
‖Δ2‖_F = 0.21317186599953494 (Δ2's norm equals its
candidate norm, as it must for a single-target LoRA).

**(iv) Composed capability (measured):** head-incremental
regression **0.15023326850496233** ≤ eps 0.25 (worst:
tiny-cap-10); echo accuracy **1.0 (16/16)** vs the
recorded floor 0.9375.

**(v) Signal 2's held-out records:** at base 0.00140625 →
3.891045992787381e-06 after at-base training; at `d0001`
0.00140625 (by construction) → 3.891045006596239e-06 after
training at `d0001`; 16/16 examples closer in both runs.

**(vi) Last-token overlap (measured):** {9, 22}, 2 of 8 —
see Disjointness above. The alignment in (ii) is not
explained away by token overlap: the two signals share
only a quarter of their last tokens, yet their movements
still align at the worst probe — consistent with RFC
§5.6.4's warning that both corrections share a component
(both move *away* from the last token itself) and that a
rank-8 readout update acts on every hidden state it meets.

**Attribution:** the REJECT is attributable to *movement
alignment*, not to an oversized second update — Δ2's own
worst incremental movement (0.1667) is *smaller* than
cycle 1's drift (0.1719) and comparable to signal 2's
at-base drift (0.1667), and its norm is smaller than
Δ1's. What failed is the composition: the remaining
headroom after cycle 1 was 0.07810781709849834 (RFC
§5.6.4 arithmetic on SI-21's numbers), Δ2's incremental
movement exceeds it at 5 of 8 probes, and at the worst
probe the movements add rather than cancel. A derivation
is not a prediction (§12.14): none of this was claimed
before the run, and the crude sufficient condition the
RFC derived (a second update whose own maximum movement
is within the headroom cannot breach tolerance) is
measurably *not* met here.

No diagnostic was uncomputable; nothing in this set is
reported "not evaluated".

## VERIFY and unwind (§10.10)

- **Cycle 1** ADMITted and passed the §6.6 probe VERIFY
  exactly as in SI-19/SI-21 (journaled
  `transformer-probe-verify`, measured-pass on both arms,
  echo 1.0, baseline labelled "pre-cycle head
  (caller-captured)").
- **Cycle 2** was REJECTed at the gate and never
  materialised, so there is no cycle-2 VERIFY and no
  VERIFY rollback — nothing landed to verify or roll
  back. The would-be VERIFY rollback target, had it
  landed and failed, is `d0001` by the miniature branch's
  existing construction; that path is unchanged and
  remains covered by SI-17's injected-failure test.
- **Unwind, honestly conditioned (§9.9):** because cycle
  2 REJECTed, **no authored two-delta stack exists** —
  nothing is unwound, and no unwind is claimed. The
  bundle head after a decisive SI-23 run is `d0001`.
  SI-21's fixture-proven unwind (authored `d0001` + a
  benign hand-built fixture `d0002`, labelled a fixture)
  remains the only unwind evidence, exactly as SI-21
  left it. The transformer-squash precondition (an
  authored two-delta stack, RFC §7.1) therefore remains
  **unmet** after SI-23 as well.

## Acceptance criteria (RFC SI-22 §10) — status

1. ✅ **Dispatch and recorded selection** — the registry
   serves exactly the two named signals; every other id
   refuses with the existing `ValueError` discipline;
   foreign-target and foreign-rank/alpha refusals are
   asserted per signal by test. Recorded selection is
   asserted per cycle on both verdict paths: cycle 1's
   journaled `training-hyperparams` (ADMIT, under `d0001`)
   names `tiny-successor-lean-001`; cycle 2's (REJECT,
   under its self-edit) names
   `tiny-predecessor-lean-001` — each equal to the signal
   the driver configured for that cycle. No autonomous
   selection exists.
2. ✅ **Gradient check for signal 2, at the decisive
   head** — re-run at `d0001` (worst 5.062663472655505e-08)
   AND at base (worst 5.5249277028738815e-08), tolerance
   1e-5, both published above.
3. ✅ **Streams audited** — the full offset table is
   published above; the extended guard asserts all four
   training offsets and both probe offsets are pairwise
   distinct (+2, +3, +11, +17, +23, +29).
4. ✅ **Disjointness proven** — all five §5.4 obligations
   hold by construction and are asserted by test,
   including signal 2 vs signal 1's full 48; the measured
   last-token overlap {9, 22} is published above.
5. ✅ **Echo-trap records** — signal 2's pre-training
   held-out loss is recorded and published at base and at
   `d0001` (both 0.00140625): the at-base record CONFIRMS
   the §5.6.3 derivation by measurement; the at-`d0001`
   record is stated, wherever it appears, to recur by
   construction under weights-under-training anchoring.
6. ✅ **Determinism per cycle** — a full decisive run
   repeated from the seed reproduces both candidates'
   blobs/records, the landed delta record, and the
   journaled gate measurements exactly; hyperparams equal
   except the labelled `wall_time_seconds`. Signal 2's
   `train_delta` alone is likewise byte-identical across
   repeats (asserted).
7. ✅ **Gate honesty per cycle** — cycle 2's journaled
   checks 3/4 equal an independent `ProbeContext`
   evaluation of the same candidate against `d0001`
   (abs 1e-12); cycle 1's equality is covered by the
   SI-19/SI-21 suites this run keeps green. Cycle 2's
   verdict is **REJECT, as measured**. No ADMIT was
   required: SI-23 succeeds by its claim — a second signal
   was genuinely authored against the landed head under
   settled semantics and genuinely judged.
8. ✅ **Cumulative accounting and diagnostics
   published** — the side-by-side table and the full
   §5.6.4 diagnostic set above, measurements and
   derivations labelled, worst probes named, nothing
   uncomputable (nothing reported "not evaluated").
9. ✅ **First-cycle reproduction** — configured with
   signal 1 (the default), a first cycle reproduces
   SI-19/SI-21 exactly: candidate blob
   `sha256:2761e8ffc139ce3e107c11da035e8a73c20792945a494f4e493dccf418b6ca9e`,
   drift 0.17189218290150166, regression
   0.1657181391492486 — asserted standalone and inside
   the decisive run.
10. ✅ **VERIFY, unwind, and untouched scope** — cycle-1
    VERIFY as in SI-19; unwind honestly conditioned (no
    authored stack exists, none unwound, none claimed —
    see VERIFY and unwind). Exactly two authored cycles
    attempted in the decisive run; `max_stack_depth` 8,
    `identity_tol` 0.25, `capability_eps` 0.25,
    `max_targets_per_delta` 1, `allow_additive` false and
    every other policy default unchanged (asserted);
    `si squash` still fails closed on the resulting
    transformer stack (asserted; SI-12's suite green);
    the full existing suite is green (below); the demo
    path is unchanged; `si serve` still returns 501 for
    every transformer bundle (existing suite); no CLI
    wiring added. Per-cycle `train_delta` wall times
    1.00 s / 0.98 s; total two-cycle wall time 2.35 s;
    full suite runtime below. All on this VM, CPU only.

Full suite on the author's own run: **315 passed,
2 skipped** in 119.40s — the SI-21 baseline (302 passed,
2 skipped) plus this slice's 13 new tests, zero
regressions. Focused run: `tests/test_second_signal.py`,
13 passed in 25.65s.

## What is still NOT claimed

- **No authored stack deeper than one.** Cycle 2
  REJECTed; the deepest landed authored stack on record
  remains depth 1 (`d0001`). Signal 2's cycle-2 candidate
  exists only as a rejected, never-written candidate
  (blob-store growth +0 B). No authored two-delta stack
  exists, and the transformer-squash precondition remains
  unmet.
- **Signal 2 ADMITs at base — and only at base.** Its
  at-base first cycle ADMITted (drift 0.1667); that is a
  fact about signal 2's definition against the seed, not
  evidence that a signal-2 delta lands on any non-base
  head, and it is never presented as such.
- **No third cycle, no continual learning.** Exactly two
  authored cycles were attempted in the decisive run, on
  constructed signals, on a constructed model.
- **No new semantics.** `TARGET_ANCHORING` stays
  `"weights-under-training"`; the base-anchored route
  (RFC §5.7) was not implemented or tested. SI-23's
  diagnostics are the measured evidence a later RFC
  would design that route against — the alignment
  finding above is a finding, not an argument this doc
  makes for any particular successor.
- **No Glimmer anything.** No Glimmer forward path,
  probes, training, or authorship; the Glimmer seed is
  untouched and its deltas still escalate with checks
  3/4 "not evaluated".
- **No widening.** Two signals, one target, one bundle
  kind; no additive or multi-target authorship; no
  policy default changed; no tolerance moved; signal 1
  is byte-for-byte SI-19's.
- **No serving, no CLI wiring.** Transformer bundles
  keep returning 501.
- **Diagnostics are not verdicts.** The §5.6.4 set above
  attributes the gate's REJECT after the fact; it is not
  a gate check, it influenced no verdict, and none of it
  was a prediction (RFC §12.12/§12.14).
- **Training numbers are training numbers.** Every loss
  above is labelled with its signal id, partition, and
  head; none of them is a gate or probe result (§12.1).
  The two signals' held-out figures are never averaged,
  summed, or blended (§12.11).
- **Recorded selection is not a choice the mind made**
  (§12.13). The driver configured each cycle's signal;
  the journal shows it. Nothing here describes the mind
  as choosing, deciding, or preferring a signal.

## Terminology

Machine learning infrastructure. This document contains
no "custom LLM software" claims; the miniature is a
constructed machine-learning test instrument, its
authored deltas are trained on constructed signals, and
its second signal's second-cycle delta was judged — and
rejected — on cumulative identity drift, with the
alignment that caused it measured and published.
