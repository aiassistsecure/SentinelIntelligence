# SI-22 RFC — The Next Frontier After Sequential Authorship
(Design Only)

> **Status: DESIGN ONLY.** This document evaluates candidate
> directions and recommends one next slice. It changes no
> production code, no test, no manifest schema, no policy
> default, and no gate behaviour. Nothing in this RFC is
> implemented unless a section explicitly says it exists today,
> in code, on `main`.
>
> **The one sentence that governs everything below:** SI-21
> proved the miniature's authored second write is genuinely
> trained against the landed head and genuinely judged — and
> the gate REJECTED it on cumulative identity drift, exactly
> as derived — and this RFC must never be cited as evidence
> that a second *signal* can be authored, that any authored
> two-delta stack exists, that cycle-2 ADMITs are achievable
> by any route, or that any slice proposed here exists.
>
> **Scope discipline:** CPU only. No GPU, no cloud, no spend,
> no downloads. Terminology: this is machine learning
> infrastructure throughout.
>
> **Relationship to prior work:** RFC SI-20 evaluated the
> frontiers open after SI-19 and recommended sequential
> authored writes; SI-21 built them (PR #26, as-built in
> `docs/SI-21-sequential-authored-writes.md`) and the gate's
> cycle-2 verdict was a measured REJECT. This RFC is SI-20's
> successor in the established cadence (SI-18 RFC → SI-19
> build → SI-20 RFC → SI-21 build): it restates the verified
> state after SI-21, re-evaluates every SI-20 candidate
> against SI-21's measured evidence — including the two
> SI-20 deferrals whose stated blockers SI-21 has now
> retired or confirmed — and gives the nearest deferred
> candidate, a second training signal, the deep design
> treatment SI-20 §5.1 deferred. SI-20's rejection of
> Glimmer probing is carried forward, not reopened (§7).
> The SI-9 RFC's §7 risks remain in force and are inherited,
> not relitigated.

---

## Table of contents

1. [Where SI stands after SI-21](#1-where-si-stands-after-si-21)
2. [Inherited invariants](#2-inherited-invariants)
3. [Candidate (a): a small real safetensors base — still deferred](#3-candidate-a-a-small-real-safetensors-base--still-deferred)
4. [Candidate (b): sequential authored self-writes — DONE as SI-21](#4-candidate-b-sequential-authored-self-writes--done-as-si-21)
5. [Candidate (c1): a second training signal, same target — the deep treatment](#5-candidate-c1-a-second-training-signal-same-target)
6. [Candidates (c2) and (c3): new target; additive / multi-target](#6-candidates-c2-and-c3-new-target-additive--multi-target)
7. [Candidate (d): transformer squash; candidate (e): Glimmer probing](#7-candidate-d-transformer-squash-candidate-e-glimmer-probing)
8. [Decision summary](#8-decision-summary)
9. [The recommended slice: SI-23 (outline)](#9-the-recommended-slice-si-23-outline)
10. [Acceptance criteria for SI-23](#10-acceptance-criteria-for-si-23)
11. [Non-goals](#11-non-goals)
12. [Honesty and labelling requirements](#12-honesty-and-labelling-requirements-for-any-future-slice)
13. [Risks and open questions](#13-risks-and-open-questions)
14. [Summary](#14-summary)

---

## 1. Where SI stands after SI-21

Everything in this section is verifiable in the code and docs
on `main` at the SI-21 merge (main tip `7ed52b2`, the merge of
PR #26). Numbers are cited to the as-built document that
measured them; this RFC measures nothing new. Where a
statement below is a *derivation* from cited facts rather
than a measurement, it is labelled a derivation, exactly as
SI-20 §1 labelled its arithmetic.

**What exists, exactly:**

- **One measurable transformer subject, unchanged.** The
  SI-14 miniature (`TinyTransformerMind`): a constructed,
  deterministically seeded, **not trained** decoder-only
  transformer — 30 tensors, 23,456 elements. Architecture
  pinned: vocabulary 32, `d_model` 32, 2 layers, 4 heads,
  head dimension 8, MLP hidden 64, maximum sequence 16
  (`si/tiny_seed.py::ARCHITECTURE`). Probe suites: 8 identity
  + 16 capability (echo-task) entries; recorded base echo
  accuracy 16/16 = 1.0, echo floor 0.9375; bundle tolerances
  both 0.25 (`identity_tol`, `capability_eps`).
- **Live probe verdicts for miniature bundles (SI-17),
  unchanged.** Probe-fail → REJECT, all-pass →
  ADMIT-eligible, anything unevaluated → ESCALATE,
  exclusively probe-blind candidates → blind-escalated
  (`si/probes.py::derive_probe_blind_targets`; the blind set
  is exactly both layers' `k_proj.bias`).
- **One authored delta, trained and ADMITted (SI-19),
  unchanged.** One named signal (`tiny-successor-lean-001`,
  target-logits MSE, stream `MINI_SEED + 11`, 32 train / 16
  held-out probe-disjoint examples), one LoRA target
  (`lm_head.weight`, rank 8, alpha 8), analytic gradients
  finite-difference checked, gate verdict ADMIT as measured:
  identity drift 0.17189218290150166 ≤ tol 0.25, capability
  regression 0.1657181391492486 ≤ eps 0.25, echo 16/16.
- **A freshness contract, implemented in exactly one place
  (SI-21).** `TinyTransformerMind.refresh_manifest()`
  (`si/tiny_transformer.py:482`) reloads the bundle's current
  verified manifest via `load_bundle` and replaces the
  construction-time snapshot; its only caller is
  `train_delta` (`si/tiny_transformer.py:521`), so every
  training path trains against the cycle's verified manifest,
  never a stale snapshot. SI-21's stale-control test proved
  the distinction by measurement: handing `train_candidate`
  the stale base manifest re-derives SI-19's targets and
  produces cycle 1's byte-identical candidate, while the
  refreshed path trains cycle 2 at head = `d0001` with a
  genuinely different blob (SI-21 as-built, "The freshness
  contract").
- **Named target anchoring (SI-21).**
  `TARGET_ANCHORING = "weights-under-training"`
  (`si/tiny_training.py:173`): targets are derived from the
  weight set under training — the bundle's applied weights at
  the head being trained against — plus the recorded
  successor-lean correction, exactly as
  `build_training_signal` computes them. Every cycle's
  hyperparams record `head`, `target_source_head`,
  `target_anchoring`, and `stack_depth` beside the SI-19
  fields, and the miniature cycle journals them as a
  `training-hyperparams` provenance note on both verdict
  paths (SI-21 as-built). The alternative — base-anchored
  targets — was considered and not chosen (SI-20 §4.3.2);
  §5.7 of this RFC returns to it as a competing route and
  defers it again, for a stated reason.
- **Sequential authorship, measured end to end (SI-21).**
  A two-cycle run driven by one reused `TinyTransformerMind`
  through `si.run.run_cycles`: cycle 1 ADMITted as `d0001`,
  reproducing SI-19 exactly (same candidate blob
  `sha256:2761e8ff…6ca9e`, same gate numbers, §6.6 VERIFY
  pass). Cycle 2 was genuinely trained against `d0001`
  (different blob, head-relative held-out loss falling ~438×
  at its head, like cycle 1's) and the gate **REJECTED** it:
  cumulative identity drift **0.34378434903919697** > tol
  0.25, while the head-incremental arms passed (capability
  regression 0.16571810841560364 ≤ eps 0.25; echo 1.0). No
  second delta landed; the bundle head after an SI-21 run is
  `d0001`. The SI-20 §4.4 derivation — identically stacking
  cycle 1's candidate doubles the drift, predicted
  2 × 0.17189218290150166 = **0.34378436580300332** — was
  **confirmed by measurement**: the stale-manifest candidate
  (byte-identical to `d0001`) evaluated against head `d0001`
  measured **0.34378437884151936**, agreement to ~1.3e-08
  (SI-21 as-built, "The §4.4 derivation: CONFIRMED by
  measurement"). The genuinely retrained cycle-2 candidate's
  cumulative drift landed within 3e-08 of the identical-stack
  figure — measured, and explained in the as-built: the
  head-relative signal asks for the same correction from the
  head, over hidden states the readout cannot change, so the
  trained second update is nearly the first update again.
- **Unwind granularity, proven on a labelled fixture stack
  (SI-21).** Because cycle 2 REJECTED, **no authored
  two-delta stack exists**. Two-step rollback was exercised
  on authored `d0001` plus a benign hand-built **fixture**
  `d0002` (journalled as a fixture, never described as
  authored), with each restored head's probe state
  re-measured: rollback to `d0001` VERIFY pass (drift
  0.17189218290150166), rollback to base VERIFY pass (drift
  0.0, regression 0.0). §7.1 states what this does and does
  not do for transformer squash.
- **Suite state:** 302 passed, 2 skipped on `main` at the
  SI-21 merge (SI-21 as-built, author's runs: the SI-19
  baseline of 292 passed, 2 skipped, plus SI-21's 10 new
  tests in `tests/test_sequential_authorship.py`, zero
  regressions).

**What does not exist, exactly (SI-21, "What is still NOT
claimed", verified against the code on `main`):**

- **No authored stack deeper than one.** The deepest landed
  authored stack on record is depth 1 (`d0001`). Cycle 2's
  candidate exists only as a rejected, never-written
  candidate (blob-store growth +0 B).
- **No second signal, no second target, no additive or
  multi-target authorship.** Dispatch is single-signal:
  `train_candidate` accepts exactly `TRAINING_SIGNAL_ID` and
  raises `ValueError` on any other id (`si/tiny_training.py`,
  the intent validation at the head of `train_candidate`);
  `TinyTransformerMind.propose` emits exactly that signal's
  intent (`si/tiny_transformer.py:438`). Policy still
  forbids authored widening: `max_targets_per_delta` is 1
  and `allow_additive` is false in the default policy
  (`si/bundle.py:46`, `si/bundle.py:49`).
- **No third cycle, no continual learning.** Exactly two
  authored cycles have ever been attempted, on a constructed
  signal, on a constructed model.
- **No Glimmer anything.** No Glimmer forward path, probes,
  training, or authorship; the Glimmer seed is untouched and
  its deltas still escalate with checks 3/4 "not evaluated"
  (§7.2).
- **No serving, no CLI wiring for the miniature.**
  `si serve` returns 501 for every transformer bundle
  (`si/serve.py`); `si cycle` / `si run` accept only
  `--mind demo|mock`.
- **No transformer squash.** SI-12's fail-closed refusal
  stands: `si/squash.py::squash_bundle` raises `SquashError`
  before any write if any record on the resolved active
  chain has kind `transformer-delta-v1` (`si/squash.py:216`).

The frontier question is therefore precise in a new way:
the loop can originate a miniature delta, judge it, land it,
and originate a *second* one on top of it — and the gate's
cumulative identity accounting has now been measured
refusing the second write of the *same* signal. SI-21's
as-built states the only honest routes to a landed second
write: "a future slice that wants cycle-2 ADMITs must argue
for new semantics or a new signal, never a wider gate."
This RFC evaluates both routes and recommends one.

---

## 2. Inherited invariants

Not negotiable inside this RFC (SI-9 §1.3, RFC SI-13 §2,
RFC SI-18 §2, RFC SI-20 §2, restated for the candidates):

1. **Base is frozen forever.** All change is a stack of
   deltas; rollback is moving `head`.
2. **Never duplicate bytes** (Mark's design rule). A blob
   whose hash exists is never rewritten or copied; seed
   imports by reference where the pattern applies
   (`stored_bytes: 0` — the SI-8 Glimmer seed stores zero
   tensor bytes against 59,553,253,376 logical bytes,
   SI-9 §0).
3. **No ungated write.** Authorship produces *candidates*.
   The gate — and only the gate — admits, rejects, or
   escalates them, and SI-17's verdict mapping and
   blind-set guard apply to authored candidates exactly as
   to fixtures, at every cycle, for every signal.
4. **Honest labels are load-bearing.** "Not evaluated" stays
   distinguishable from a measured 0.0; a training loss
   stays distinguishable from a gate result; a number
   measured at one head stays labelled with that head
   (SI-20 §12.8); incremental and cumulative figures are
   never presented as each other (SI-20 §12.9).
5. **Fail closed.** A missing, mismatched, or un-runnable
   measurement is "not evaluated" — never a pass, never a
   silent skip.
6. **CPU only, existing dependencies only.** The runtime's
   dependencies are `numpy` and `safetensors`
   (`requirements.txt`, `pyproject.toml`). No GPU, no
   network, no downloads, no spend, no new heavy dependency.
7. **Determinism where tests need it.** Seeded RNG, fixed
   inputs, no sampling, no wall-clock dependence in any
   measured or trained value — and deterministic
   *serialisation* wherever a blob hash is content
   addressing (SI-19's canonical-header lesson, SI-20
   §12.7).
8. **No tolerance is a design variable.** No candidate in
   this RFC proposes moving `identity_tol`,
   `capability_eps`, any cap, or any policy default to make
   an outcome pass. SI-21's cycle-2 REJECT is a measurement
   to design *around* — with new semantics or a new signal —
   never a number to outvote (SI-21 as-built; SI-20 §11).

---

## 3. Candidate (a): a small real safetensors base — still deferred

**Verdict up front: DEFERRED** — unchanged from SI-20 §3,
and this RFC does not reopen it. SI-20 audited SI-18 §3.7's
four preconditions (A-1 feasibility measurements for one
named family; A-2 a binding design for `reference-v1`
bases; A-3 a capability-suite design for a trained base;
A-4 a one-family executor design with SI-15-grade numeric
proof) and found all four unmet. SI-21 changed none of
them: it built sequential authorship *for the miniature*,
using the miniature's executor, the miniature's binding,
and the miniature's constructed signal. Nothing in SI-21's
freshness contract, anchoring record, or two-cycle run is a
measurement of any real checkpoint family, and citing any
of it toward A-1 would be exactly the unlabelled transfer
SI-18 §10.4 forbids.

One SI-20 point bears restating because it is the actual
blocker, not a preference: naming a candidate family and
measuring *its* forward cost requires the checkpoint
itself, and **obtaining one means a download** — outside
this project's autonomous scope (invariant 6). That is an
operator decision to flag, not a step to take, and this
RFC neither schedules nor performs one. The measurement-
only slice SI-20 §3.5 described (synthetic-shape scaling
measurements; the A-2 binding design on paper) remains a
legitimate *future* candidate; it is not recommended as
SI-23 because §5's candidate is buildable now with
instruments that already exist and retires a stated blocker
SI-21 has just cleared, while §3.5's slice retires
overheads, not any of A-1 to A-4 as stated.

| Precondition (SI-18 §3.7) | Status today | Changed by SI-21? |
|---|---|---|
| A-1 feasibility measurements, one named family | Not met — none exist | No |
| A-2 `reference-v1` binding design + tamper analysis | Not met — binding unstatable for referenced bases (`si/probes.py`) | No |
| A-3 capability-suite design for a trained base | Not met — unsolved design problem | No |
| A-4 one-family executor design + numeric proof | Not met — executor refuses all but the miniature | No |

---

## 4. Candidate (b): sequential authored self-writes — DONE as SI-21

**Verdict up front: DONE** — this was SI-20's recommended
slice, it merged as PR #26, and its result is the evidence
base for everything below. Restating the outcome without
inflation:

- The freshness contract exists and is proven by a test
  that tells stale from fresh apart (§1).
- Target anchoring is named, recorded per cycle, and its
  consequence is published: cycle 2's pre-training held-out
  loss of 0.00140625 recurs *by construction* at any head
  under weights-under-training anchoring and is not fresh
  evidence about the base (SI-21 as-built §10.7).
- Composition is proven in SI-20 §4.6's exact sense: the
  second write was genuinely authored against the landed
  head and genuinely judged. The judgement was REJECT.
- The §4.4 derivation is confirmed, not refuted (§1), and
  with it the accounting model every later candidate must
  design against: **identity drift is seed-absolute and
  cumulative; capability regression is head-incremental**
  (`si/gate.py:415` `_identity_check` compares the
  candidate-overlay forward against the suites' recorded
  seed logits; `si/gate.py:431` `_capability_check` compares
  against `ctx.capability_baseline()`, which
  `si/probes.py:413` measures live at the current head).
- What SI-21 did **not** produce: a landed second delta, an
  authored two-delta stack, or any evidence about a third
  cycle. Its own closing line binds this RFC: a future
  slice that wants cycle-2 ADMITs must argue for new
  semantics or a new signal, never a wider gate.

Candidate (b) is therefore closed, not deferred: its
question is answered. The open frontier is which of its
two named successors — new semantics (§5.7) or a new
signal (§5) — goes first.

---

## 5. Candidate (c1): a second training signal, same target

**Verdict up front: RECOMMENDED as the next slice
(SI-23)** — conditionally, and with the condition stated
before the design: **no derivation in this RFC shows that
a second signal's authored delta will ADMIT at head
`d0001`.** What the evidence supports is narrower and, this
RFC argues, sufficient: (c1) was SI-20's "nearest of the
three" widenings, explicitly blocked on §4 composition
semantics that SI-21 has now chosen, implemented, and
proven; its remaining unknowns are all measurable with
instruments that already exist; the one route that
provably *cannot* land a second write (re-leaning the same
correction, §5.6.1) is not the route (c1) takes; and both
of its possible gate outcomes are honest, publishable
results — an ADMIT would be the first authored two-delta
stack, and a REJECT would measure *why*, per probe, with
the diagnostics §5.6.4 pre-registers. As in SI-21, **no
acceptance criterion in §10 requires an ADMIT.**

### 5.1 Why the blocker is retired

SI-20 §5.1 deferred (c1) for one blocking reason and three
design debts:

- **Blocking dependency (retired).** "A second signal's
  delta, proposed on a bundle that already carries the
  first signal's delta, trains against a non-base head —
  the exact semantics §4 says are unchosen." SI-21 chose
  them: the freshness contract (exactly one manifest
  trains, the cycle's verified manifest,
  `si/tiny_transformer.py:482`/`:521`) and
  weights-under-training anchoring
  (`si/tiny_training.py:173`), recorded per cycle in
  hyperparams and journal. A second signal inherits both
  unchanged (§5.5); nothing about (c1) reopens either
  choice.
- **Design debts (still owed, designed in §5.2–§5.4).**
  A dispatch/selection rule; stream discipline for a new
  signal; a per-signal echo-trap record. SI-20 named them;
  SI-21 did not pay them, because SI-21 changed no signal.
  They are (c1)'s real work, and they are designable now
  precisely because the composition semantics they depend
  on are settled.

### 5.2 Dispatch and selection: who names the signal

Verified current state:

- `train_candidate` validates the intent's
  `training_signal_id` against the single constant
  `TRAINING_SIGNAL_ID = "tiny-successor-lean-001"` and
  raises `ValueError` on any other id; likewise it refuses
  a declared target list other than `[lm_head.weight]` and
  any rank/alpha other than the pinned 8/8
  (`si/tiny_training.py`, head of `train_candidate`).
- `TinyTransformerMind.propose` unconditionally emits that
  one signal's intent (`si/tiny_transformer.py:438`) —
  there is no selection question today because there is
  nothing to select between.
- The precedent for multi-signal service already exists in
  the demo line: `DemoMind.train_delta` serves exactly its
  one named signal (`si/minds.py:232` refuses any other id
  the same way), and `MockMind` likewise names its signal
  in its intent (`si/minds.py:214`). The discipline — *a
  mind serves exactly the signals its design names, and
  refuses all others* — generalises; the selection rule is
  the new part.

**The design SI-23 must implement (this RFC's
recommendation):**

1. **Service is a named set of two.** `train_candidate`
   dispatches on `intent.training_signal_id` through a
   signal registry — a per-signal record carrying the id,
   the stream offset (§5.3), the correction definition
   (§5.6.2), the partitions, and the pinned rank/alpha/
   target — and refuses any id not in the registry with
   the existing `ValueError` discipline. Refusal behaviour
   for foreign targets and foreign rank/alpha is unchanged
   and applies per signal.
2. **Selection is explicit, recorded, and not learned.**
   The mind is configured with the signal it will propose
   (a constructor-level named choice, defaulting to
   `tiny-successor-lean-001`), and `propose` emits exactly
   the configured signal's intent. There is no autonomous
   signal *choice* to claim: no perception-driven,
   depth-driven, or alternating selection is built in this
   slice, because any such rule would be a second
   unproven semantics stacked on the first (the SI-20
   §5.1 warning, applied to selection itself). A driver
   that wants signal 2 in cycle 2 says so, in code, and
   the journal shows it.
3. **The selection is part of provenance.** Every cycle's
   hyperparams already record `training_signal_id`
   (SI-19 field, carried through SI-21); SI-23 must
   additionally assert — by test, per cycle — that the
   journaled `training-hyperparams` note's signal id equals
   the signal the driver configured for that cycle, on both
   verdict paths (the SI-21 journaling shape). A cycle whose
   signal cannot be read back from its own journal has
   failed this design, whatever the gate said.
4. **First-cycle reproduction is the compatibility bar.**
   Configured with signal 1 (the default), a first cycle
   must reproduce SI-19/SI-21 exactly — same candidate blob
   (`sha256:2761e8ff…6ca9e`), same gate numbers — so the
   registry refactor is proven behaviour-preserving for the
   existing signal before the new signal is trusted
   (§10 criterion 9).

### 5.3 Stream discipline: offsets, audited

Streams are `numpy.random.default_rng(MINI_SEED + offset)`
with `MINI_SEED = 20261020` (`si/tiny_seed.py:84`). Every
offset in use on `main`, verified in code this run:

| Offset | Stream | Where |
|---|---|---|
| +2 | identity probe inputs | `si/tiny_seed.py:291` |
| +3 | capability probe inputs | `si/tiny_seed.py:307` |
| +11 | signal 1 sequences (`SIGNAL_SEED_OFFSET`) | `si/tiny_training.py:197` |
| +17 | signal 1 factor initialisation (`INIT_SEED_OFFSET`) | `si/tiny_training.py:201` |

(The DemoMind/MockMind streams in `si/minds.py` — seed + 2,
+ 3, + 101 — are drawn from those minds' own seed, not
`MINI_SEED`, and do not collide with the miniature streams;
the miniature's construction seed itself is `MINI_SEED`
unoffset, in `si/tiny_seed.py::build_tiny_weights`.)

**Design rule for SI-23:** signal 2 takes two new offsets,
one for its sequences and one for its factor
initialisation, chosen from offsets unused above and
pinned as named constants beside the existing pair
(`SIGNAL2_SEED_OFFSET`, `INIT2_SEED_OFFSET`), with a test
in the shape of the existing guard
(`tests/test_tiny_training.py` already asserts
`SIGNAL_SEED_OFFSET not in (2, 3)`) extended to assert all
four miniature training offsets and both probe offsets are
pairwise distinct. This RFC deliberately does not pin the
two integer values: any unused pair satisfies the
discipline, and pinning them here without the code that
audits the full table would be inventing a fact the slice
must verify. What is *not* permitted: reusing +11 (signal
2's sequences would be signal 1's), reusing +17 (signal
2's initial factors would be signal 1's — the trained
updates would start from the same point, blurring exactly
the independence §5.6 argues from), or drawing either from
a probe stream.

### 5.4 Disjointness: the proof obligations

Signal 1's construction already shows the pattern SI-23
must extend: `signal_sequences()` draws from its own
stream and rejects any draw that duplicates an earlier
signal sequence or collides with *any* probe-suite input
(identity or capability), so disjointness holds by
construction and is asserted by test
(`si/tiny_training.py::signal_sequences`).

For signal 2 the obligation set is strictly larger. SI-23
must prove, by construction *and* by test:

1. **Signal 2 vs identity suite** — no signal-2 sequence
   (train or held-out) equals any of the 8 identity inputs.
2. **Signal 2 vs capability suite** — likewise against the
   16 capability inputs.
3. **Signal 2 vs signal 1** — no signal-2 sequence equals
   any of signal 1's 48 sequences (32 train + 16 held-out).
   This obligation is new with (c1) and is the one SI-20
   §5.1 flagged ("an unexamined overlap would blur what
   each signal's held-out numbers mean"): if signal 2's
   held-out partition shared sequences with signal 1's
   training partition, a held-out improvement claim for a
   signal-2 delta stacked on `d0001` would be contaminated
   by signal 1's training, and neither signal's numbers
   could be read cleanly.
4. **Signal 2 internal partition disjointness** — its 32
   train and 16 held-out sequences are pairwise distinct,
   as signal 1's are.
5. **Last-token discipline stated, not assumed.** Signal 1
   assigns last tokens round-robin from 8 tokens drawn from
   its stream. Signal 2's last-token set is drawn from its
   own stream; whether it happens to share tokens with
   signal 1's set is a *measured fact to publish*, not a
   disjointness obligation — sequences, not tokens, are the
   unit of disjointness, and §5.6.4's diagnostics are
   interpreted against whatever overlap is measured.

The rejection-draw construction must reject against the
*union* of all four sets (both probe suites, signal 1's
full 48, signal 2's own draws so far), from signal 2's own
stream, deterministically — the same shape as signal 1's,
with a bigger reject set.

### 5.5 Anchoring at non-base heads: no new semantics

Signal 2 inherits `TARGET_ANCHORING =
"weights-under-training"` (`si/tiny_training.py:173`)
unchanged, and this RFC recommends SI-23 *not* revisit the
choice, for SI-20 §4.3.2's original reason, which applies
verbatim: changing the signal semantics in the same slice
that tests a new signal would confound the experiment.
Concretely:

- Signal 2's targets at any head are derived from the
  weights under training at that head plus signal 2's own
  recorded correction (§5.6.2). Its hyperparams record
  `head`, `target_source_head`, `target_anchoring`, and
  `stack_depth` exactly as SI-21 cycles do, plus its own
  `training_signal_id` — so a signal-2 number can never be
  mistaken for a signal-1 number or a base fact.
- The by-construction consequence must be labelled wherever
  it appears, as SI-21 labelled it for signal 1: signal 2's
  pre-training held-out loss at its head is the loss of the
  weights under training against targets derived from those
  same weights (§5.6.3 computes its closed form) — a
  head-relative construction artifact at cycle ≥ 2, not
  fresh evidence about the base.
- One interaction is genuinely new and must be tested, not
  assumed: at head `d0001`, signal 2 trains over hidden
  states produced by a readout-carrying head. The analytic
  gradient survives — it takes the readout weight as a
  parameter and the readout's input does not depend on the
  readout (the SI-19/SI-21 argument, unchanged) — but
  SI-23 must re-run the finite-difference gradient check
  *for signal 2's loss at head `d0001`*, not only at base,
  because that is the configuration the decisive cycle
  trains in (§10 criterion 2).

### 5.6 What the second signal should be — and the honest plausibility argument

#### 5.6.1 What provably cannot work

Re-leaning signal 1's correction from `d0001` cannot land:
that is no longer a hypothesis but SI-21's measurement —
cumulative drift 0.34378434903919697 (retrained) /
0.34378437884151936 (identical stack) against tol 0.25.
Any second signal whose trained update is, in effect, the
same readout movement again meets the same accounting.
This rules out, by measurement rather than taste: a second
signal that is signal 1 with a new id, a new stream, or a
new partition of the same successor-lean correction. It
also rules out the superficially attractive "same
correction, smaller scale" variant as a *primary* design:
under the linearity derivation of §5.6.4, a half-scale
successor-lean's drift contribution would be expected at
roughly half of 0.1719 ≈ 0.086 — still above the remaining
headroom computed in §5.6.4 — and a quarter-scale variant
chosen *because* the arithmetic fits the budget is
gate-fitting: a signal designed backward from the
tolerance rather than forward from a task. This RFC names
that trap explicitly so a later slice cannot walk into it
quietly (§13, risk 2).

#### 5.6.2 The recommended signal: predecessor-lean

**`tiny-predecessor-lean-001`** — the directional
counterpart of signal 1, defined with the same form and
the same discipline, differing only in the correction's
direction:

- **Form:** target-logits MSE, exactly signal 1's loss
  form. Each example's target is the weights-under-
  training's own last-position logits plus one recorded
  correction, `v = 0.15 * (onehot(predecessor) -
  onehot(last_token))` with `predecessor = (last_token -
  1) % 32` — the readout is asked to *lean toward the
  predecessor token*. Correction scale 0.15, unchanged: the
  scale is a property of the signal family's recorded
  design (SI-19), not a knob this RFC turns (§5.6.1).
- **Same target, same shape:** `lm_head.weight`, rank 8,
  alpha 8, 2,000 full-batch steps, learning rate 1.0, norm
  budget 0.9 — every training hyperparameter of signal 1,
  so the *only* differences between the two signals are
  the correction direction, the stream, and the drawn
  sequences. That minimality is deliberate: it makes the
  two cycles' measurements commensurable and any outcome
  attributable.
- **Why this signal, honestly:** it is a real, nameable,
  distinct task in the miniature's constructed world (the
  successor relation's inverse, over the same vocabulary
  ring), not a perturbation of signal 1; it is measurably
  imperfect at base and at `d0001` (§5.6.3); and its
  correction direction is the one choice available in this
  signal family whose readout movement is *not* the
  movement SI-21 measured failing — which is a plausibility
  argument, not a prediction (§5.6.4 states its limits).

#### 5.6.3 The echo trap, per signal

RFC SI-18 §4.3's rule, restated by SI-20 §5.1: each new
signal must record the score of the weights it trains
against, before training, and be measurably imperfect
there — a signal the head already satisfies authors
nothing. For signal 2, two records are required:

- **At base** (its first-cycle sanity configuration,
  §9 item 3): the base's held-out loss on signal 2.
  *Derivation:* signal 2's targets are the base's own
  logits plus a correction of the same L2 norm as signal
  1's (`0.15 * sqrt(2)`), so the zero-delta loss is the
  same closed form, `2 * 0.15**2 / 32 = 0.00140625` — the
  SI-19/SI-21 figure, derived here from the correction's
  form, to be *confirmed by measurement* in SI-23, not
  assumed from the derivation.
- **At `d0001`** (the decisive configuration): the landed
  head's pre-training held-out loss on signal 2, recorded
  in the cycle's hyperparams under the SI-21 labelling
  discipline (head-relative, target source = `d0001`). The
  same closed form is expected *by construction* under
  weights-under-training anchoring — targets derive from
  `d0001`'s own logits — and SI-23 must say so wherever the
  figure appears, exactly as SI-21 said it for signal 1.
  What this record proves is the narrow, sufficient thing:
  signal 2 is not already satisfied at `d0001` in the only
  sense authorship requires — its zero-delta loss there is
  the full correction's loss, not ~0. (A signal defined
  against a *fixed* external target could be accidentally
  satisfied by `d0001`'s lean; a head-relative signal
  cannot be, and that is a property of the anchoring SI-21
  already chose, recorded here rather than discovered
  later.)

#### 5.6.4 Cumulative identity accounting: what can and cannot be argued

The budget, from SI-21's published measurements
(arithmetic on cited numbers, labelled as such): cycle 1
spent identity drift 0.17189218290150166 of tol 0.25,
leaving **headroom 0.07810781709849834** for any second
landed delta — in the worst case. Whether a second delta
fits inside it depends on how its probe-logit movements
combine with `d0001`'s, and that combination has structure
worth stating precisely:

- *Derivation (composition is additive at the logits).*
  The readout is the final operation and its input (the
  final hidden state) does not depend on the readout — the
  fact SI-19's gradient rests on. Two LoRA updates to
  `lm_head.weight` therefore compose additively at every
  probe's logits: the composed logits at a probe are the
  seed logits plus the first update's logit movement plus
  the second update's logit movement, each movement
  computed against the same final hidden state:
  `(W + Δ1 + Δ2) h = W h + Δ1 h + Δ2 h`. The gate's
  identity check (`si/gate.py:415`, `_err` at
  `si/gate.py:239`) then takes the maximum absolute
  coordinate of the *composed* movement, over the identity
  probes — it measures the L-infinity norm of a sum of two
  movement vectors, not the sum of their norms.
- *What can be argued from that (derivation).* By the
  triangle inequality, the composed drift is at most
  cycle 1's measured drift plus the second update's own
  maximum identity-probe movement, and it can be strictly
  less — down to partial or near cancellation — where the
  two movements oppose each other at the worst probe and
  coordinate. Identical stacking is the aligned extreme
  of that range, and SI-21 measured it (§1); it is not
  the only point in the range. So a second landed delta
  inside the 0.07810781709849834 headroom is *possible in
  principle* for a differently-directed update, and the
  only sufficient condition this derivation supplies is
  the crude one: a second update whose own maximum
  identity-probe movement is itself within the headroom
  cannot push the composed drift past tolerance in exact
  arithmetic (the float32 fold may still move a measured
  value in its last digits, as SI-20 §4.4 noted).
- *What cannot be argued from that (the limits, stated as
  plainly as the derivation).* Nothing in the predecessor
  correction's definition fixes the sign, the size, or
  the inner product of the two updates' movements on the
  *identity probes*. The correction is defined on
  training sequences, at their last positions, over their
  last tokens (§5.6.2); identity probes are different
  inputs, their hidden states are different vectors, and
  a trained rank-8 readout update is a matrix applied to
  every hidden state it meets — not a per-token logit
  edit that stays where training put it. Successor and
  predecessor corrections even share one component (both
  move *away* from the last token itself, §5.6.2), so
  "opposite task" does not translate into "opposite
  movement" by any argument this RFC can make. No
  orthogonality or cancellation is claimed, estimated, or
  predicted. A Frobenius norm budget is likewise not a
  per-probe L-infinity movement bound, and this RFC does
  not use one as if it were. Capability is a separate
  accounting entirely (head-incremental regression plus
  the absolute echo arm, §4) and nothing in this
  derivation speaks to it.
- *Diagnostics pre-registered.* Because the derivation
  cannot decide the outcome, SI-23 must publish, whatever
  the gate's verdict is: (i) per identity probe, `d0001`'s
  measured drift contribution, the signal-2 candidate's
  incremental movement (the composed forward minus the
  `d0001` forward, labelled a derivation from two measured
  forwards), and the composed drift, with the worst probe
  named for each; (ii) an alignment diagnostic — the inner
  product and sign agreement of the two updates' movement
  vectors at the worst composed probe, and aggregated over
  the identity probes — labelled a derivation, published
  so a REJECT or an ADMIT can be *attributed* (alignment,
  cancellation, or a small second movement), never quoted
  as a gate check; (iii) the two folded updates' Frobenius
  inner product and each update's norm, as measured in
  SI-23; (iv) the composed candidate's head-incremental
  capability regression and echo accuracy against the
  recorded floor 0.9375; (v) signal 2's held-out records
  of §5.6.3, at base and at `d0001`; and (vi) the measured
  last-token overlap of §5.4 item 5, beside any
  interpretation of (i)–(iii). A diagnostic that cannot
  be computed is reported "not evaluated" (§2 invariant
  5), never silently omitted.

The honest bottom line of §5.6: composition at the logits
is addition of vectors, and SI-21 measured the aligned
extreme of that addition. A predecessor-lean update is
not the aligned case by construction, and its movement is
measurable with instruments that already exist. That is
a plausibility argument for running the experiment —
nothing in this section is a prediction that it ADMITs,
and §10 requires no ADMIT.

### 5.7 The competing route: new semantics (base-anchored targets) — deferred again

SI-20 §4.3.2 named the other anchoring before SI-21 chose:
targets derived from the *seed base's* logits at every
cycle, so later cycles correct the residual toward a
fixed destination instead of leaning a further step from
the head. SI-21 did not choose it, for the recorded
reason — changing the signal definition in the same slice
that tests composition would confound the experiment —
and §1 of this RFC restated it as the route not taken.
It returns here because SI-21's closing line names *new
semantics* as one of the two honest routes to a landed
second write, and this RFC must say why it recommends the
other one (§5) first, rather than leaving the choice
silent.

**What the route would be.** A cycle at head `d0001`
would train against targets anchored to base:
`target_source_head` would read `base` while `head` reads
`d0001` in the cycle's hyperparams — a combination SI-21's
records never produce, because under
weights-under-training the two fields always agree
(`si/tiny_training.py:594` and its surrounding record).
The freshness contract is untouched by the route (the
cycle's verified manifest still trains); what changes is
what the signal *asks for* at a non-base head. For
signal 1 itself, a base-anchored second cycle would ask
only for the residual of the original lean — by design a
smaller remaining correction than the full head-relative
re-lean SI-21 measured failing. That attractiveness is
real, and it is also exactly why the route needs care:
it redefines what the existing signal id means at
cycle ≥ 2.

**Why it is deferred again, for stated reasons:**

1. **Confounding, again — now in the other direction.**
   SI-23's question is whether a *new signal* lands under
   the semantics SI-21 settled. Changing the anchoring in
   the same slice would stack two unproven changes —
   exactly the SI-20 §5.1 warning this RFC applied to
   selection in §5.2 — and any outcome, ADMIT or REJECT,
   would be unattributable between them.
2. **It is a semantics change, not a parameter tweak.**
   The echo-trap record changes character (a base-anchored
   signal *can* be partially satisfied by a head that
   already leans toward the fixed target, so §5.6.3's
   by-construction argument does not transfer); the §4.4 /
   §5.6.4 accounting must be re-derived against a fixed
   origin rather than a moving one; and comparability with
   SI-21's cycle-2 numbers — the evidence base of this
   whole RFC — is lost, because the two cycles would no
   longer be the same signal wearing one id (SI-20
   §4.3.2's phrase).
3. **No measured evidence yet says it is needed.**
   The predecessor route is unmeasured. If SI-23 ADMITs
   under weights-under-training, the anchoring question
   stays open but unforced. If SI-23 REJECTs and the
   §5.6.4 diagnostics attribute the rejection to movement
   alignment under head-relative targets, base-anchoring
   becomes the *informed* next candidate — designable
   against measured diagnostics instead of against this
   RFC's speculation.
4. **Sequencing is the honesty safeguard.** Adopting new
   semantics *first*, for the *same* signal, immediately
   after that signal's measured REJECT, would have to be
   argued on its own merits in its own RFC to escape the
   reading that the semantics were redesigned backward
   from a desired verdict. Deferring it behind the
   new-signal experiment keeps that argument clean if it
   is ever made.

**Verdict: DEFERRED again — not rejected.** Base-anchored
targets remain a named, legitimate successor route.
SI-23 does not implement, test, or partially implement
it; no acceptance criterion in §10 depends on it; and
`TARGET_ANCHORING` stays `"weights-under-training"`
(`si/tiny_training.py:173`) throughout the recommended
slice.

---

## 6. Candidates (c2) and (c3): new target; additive / multi-target

SI-20 §5 split the widening candidate (c) into three
sub-directions and deferred all three (§5 of this RFC is
(c1), now recommended). The other two are re-evaluated
here against SI-21's evidence. SI-21 changed neither of
their blockers: it trained the same signal, on the same
target, through the same dispatch — deliberately. Its
freshness contract and anchoring record settle
*composition* semantics; neither is a gradient derivation
for a new tensor, and neither is evidence about authored
multi-target behaviour.

### 6.1 Candidate (c2): a new LoRA target (non-readout tensor)

**Verdict up front: DEFERRED.**

- **The gradient is still the work, unchanged.** SI-19's
  exactness rests on one fact, recorded in
  `si/tiny_training.py`'s docstring and relied on again in
  §5.6.4 above: `lm_head.weight` is the final operation,
  so no gradient with respect to any other tensor is ever
  needed and hidden states can be cached. For any earlier
  tensor — an attention projection, an MLP projection,
  the embeddings — the gradient runs backward through
  RMSNorm, causal softmax attention, and SwiGLU: the
  derivation RFC SI-18 §4.4 called error-prone in exactly
  those places. It ships, if ever, only with a
  finite-difference gradient check per target, at a
  tolerance pinned from measured agreement. None of that
  exists for any second target, and SI-21 built none of
  it: its second cycle reused SI-19's readout gradient
  verbatim.
- **The machinery a backward pass needs does not exist
  yet.** The forward replica
  (`si/tiny_training.py::forward_with_final_hidden`)
  exposes only the final hidden state; the intermediate
  activations a real backward pass through the layers
  needs are not returned anywhere in the runtime.
- **What transfers is only the judging.** Per-class
  sensitivity is 28 of 30 tensors measurable (SI-16,
  restated in §1); the blind set is exactly both layers'
  `k_proj.bias`, which are 1-D and in any case not LoRA
  targets (LoRA requires a 2-D base, enforced in the
  gate's safety check). A new-target slice would not be
  blocked by probe blindness except by its own target
  choice — but measurability was only one of SI-19's
  three recorded reasons for `lm_head.weight`. Cap fit
  partially transfers; derivability must be re-earned
  per target, with its own recorded reason, as SI-19
  recorded the readout's.
- **Why (c1) goes first.** A second signal on the same
  target is commensurable with everything already
  measured — same gradient, same fold, same accounting —
  so its outcome is attributable to the signal. A new
  target spends the slice's risk on a derivation, before
  the cheaper and more informative signal question is
  answered.

### 6.2 Candidate (c3): additive and multi-target authored candidates

**Verdict up front: REJECTED as the next slice** — the
SI-20 §5.3 verdict, carried forward unchanged, because
SI-21 produced no authored multi-target evidence that
could argue against it.

- **The format is ready; the policy is not —
  deliberately.** `construct_transformer_delta`
  (`si/transformer_delta.py:330`) already builds
  multi-target records, `si/apply.py` folds per-target
  updates in chain order, and the gate already handles
  mixed blind + measurable candidates honestly. What
  forbids authored multi-target candidates is policy:
  `max_targets_per_delta` is 1 and `allow_additive` is
  false in the default policy (`si/bundle.py:46`,
  `si/bundle.py:49`), enforced by the gate. Widening
  authorship this way still means *changing policy
  defaults to admit what the gate currently refuses* —
  the one move §2 invariant 8 and SI-20 §11 forbid as a
  route around a gate outcome. A future slice could
  argue a policy change on evidence; there is still no
  authored multi-target evidence to argue from.
- **Joint training is a new optimisation problem, not
  two copies of SI-19.** Two targets trained against one
  loss interact — each target's gradient depends on the
  other's current factors through the forward — and the
  readout-only caching trick does not survive a second,
  earlier target. Aggregate norm accounting
  (`si/transformer_delta.py:176`, `aggregate_norm`:
  root-sum-square over per-target norms, recomputed by
  the gate) would need the training loop's norm guard
  re-derived for the joint case.
- **Additive authorship's natural homes are still the
  worst targets.** Additive targets must be 1-D
  (`si/transformer_delta.py::validate_transformer_delta`);
  the miniature's 1-D tensors are norms and biases —
  including the probe-blind `k_proj.bias` pair, an
  authored delta on which could never be judged.

### 6.3 Verdicts on (c2) and (c3)

| Sub-direction | Verdict | One-line reason |
|---|---|---|
| (c2) New LoRA target | DEFERRED | Needs a full backward derivation plus a per-target gradient check; only measurability transfers from SI-19 (§6.1). |
| (c3) Additive / multi-target | REJECTED as next slice | Requires loosening `max_targets_per_delta` / `allow_additive` policy defaults with zero authored-behaviour evidence (§6.2). |

---

## 7. Candidate (d): transformer squash; candidate (e): Glimmer probing

### 7.1 Candidate (d): transformer squash for miniature deltas

**Verdict up front: DEFERRED** — and, unlike in SI-20,
the reason can now be stated in one sentence: the
precondition SI-20 §6 set is still unmet, and SI-21 is
the slice that proved it unmet by measurement rather
than by absence.

SI-20 §6.4 made squash schedulable only after SI-21
produced an authored stack and the measurements a squash
design must respect. SI-21 produced the measurements
(per-delta drift contributions, the §4.4 confirmation)
but **not the stack**: cycle 2 REJECTED, so the deepest
landed authored stack on record is still depth 1
(`d0001`), and **no authored two-delta stack exists**.
SI-21's two-step unwind was exercised on authored
`d0001` plus a benign hand-built **fixture** `d0002` —
journalled as a fixture, never described as authored
(§1). Squashing that fixture stack would prove format
handling, not authored composition: exactly the
designing-against-fixtures outcome SI-12 refused.

What SI-19 and SI-20 established still holds: for
authored same-target deltas, composition is exact
algebra — every authored delta is a single-target LoRA
on `lm_head.weight` at rank ≤ 8, two such updates compose
to a single update of rank ≤ 16 on one named tensor,
foldable by the existing `si/apply.py::fold_target`
arithmetic, with the recompression to rank ≤ 8 lossy and
therefore gated by a tolerance verify of the SI-3 kind
plus the miniature's probes (SI-20 §6.2). What still
blocks a squash slice, beyond the missing stack:

1. **The squash gate context is demo-shaped.**
   `si/squash.py::_make_squash_mind` constructs only
   `DemoMind`/`MockMind`, and `SquashContext` serves demo
   probes and a demo forward. A miniature squash candidate
   must be gated by the *probe* context — checks 3/4
   measured, the SI-17 verdict mapping, the blind-set
   guard — or it would be gated by instruments that cannot
   see it. That context is design work that does not
   exist.
2. **Provenance and identity of the squashed delta.** A
   squashed delta is not trained, so it carries no
   `hyperparams` of its own, and §12 forbids dressing it
   in its constituents' training numbers. Its journal
   story — compression of named deltas, tolerance
   measured, probes re-run — must be statable without
   borrowing authorship's labels (SI-20 §6.3).
3. **The refusal stands meanwhile.** SI-12's fail-closed
   refusal is unchanged in code:
   `si/squash.py::squash_bundle` raises `SquashError`
   before any write if any record on the resolved active
   chain has kind `transformer-delta-v1`
   (`si/squash.py:216`). §11 keeps it in force throughout
   SI-23.

If SI-23's second cycle ADMITs, the precondition is met
for the first time — an authored two-delta stack will
exist — and squash becomes a legitimate candidate for a
*later* RFC to evaluate against that stack's measured
norms and drift contributions. Meeting the precondition
does not schedule the slice, and SI-23 builds no part of
it (§11).

### 7.2 Candidate (e): probing Glimmer — REJECTED, not reopened

Glimmer probing was rejected by RFC SI-13 §7/§9,
restated by SI-18 §5 and by SI-20 §7, and is **not
reopened here**. SI-21 changes nothing about it, and
neither does this RFC's recommendation: authorship, like
probing before it, is miniature-only by construction —
the training assembles weights through the miniature
executor (`validate_architecture`,
`si/tiny_transformer.py:141`, refuses Glimmer's
descriptor), the signals are defined over a vocabulary
of 32, and the gate's probe binding cannot be stated for
the Glimmer bundle (§3).

The restated facts, so no future reader has to excavate
them: there is no Glimmer forward path, and a partial
forward over a partial stack would be a claim about
that partial computation, not about the model —
fabrication, as SI-18 §5 recorded. The Glimmer seed's
deltas still escalate with checks 3/4 "not evaluated";
its stored bytes are still 0 against 59,553,253,376
logical bytes (SI-9 §0); and one full verification of
that by-reference base still stands measured at
~29 minutes (1,756.9 s) before any forward compute
(SI-9 §0/§7.1). Any future Glimmer proposal needs its
own RFC and its own feasibility measurements; it gets
no momentum from this document.

---

## 8. Decision summary

| Candidate | Verdict | One-line reason |
|---|---|---|
| (a) Small real safetensors base | **DEFERRED** | SI-18 §3.7 audit: A-1 to A-4 all still unmet, unchanged by SI-21; naming a real base needs a download this scope forbids (§3). |
| (b) Sequential authored self-writes (miniature) | **DONE — as SI-21** | Built and measured (PR #26): cycle 1 ADMITted as `d0001`, cycle 2 genuinely trained at the head and REJECTED on cumulative identity drift (§4). |
| (c1) Second training signal, same target | **RECOMMENDED — SI-23** | Its SI-20 blocker (§4 composition semantics) is retired by SI-21; its remaining unknowns are measurable with existing instruments, and no ADMIT is required (§5). |
| (c2) New LoRA target | **DEFERRED** | Needs a full backward derivation plus a per-target finite-difference gradient check; only measurability transfers from SI-19 (§6.1). |
| (c3) Additive / multi-target | **REJECTED as next slice** | Requires loosening `max_targets_per_delta` / `allow_additive` policy defaults with zero authored-behaviour evidence (§6.2). |
| (d) Transformer squash (miniature) | **DEFERRED** | Precondition still unmet — no authored two-delta stack exists; SI-21's unwind used a labelled fixture `d0002` — and the squash gate context is still demo-shaped (§7.1). |
| (e) Glimmer probing | **REJECTED** | Rejected at SI-13 §7/§9, restated at SI-18 §5 and SI-20 §7; SI-21 changes nothing (§7.2). Not reopened. |

The two routes SI-21 named to a landed second write are
both accounted for in this table's logic: the new-signal
route is (c1), recommended; the new-semantics route is
§5.7's base-anchored targets, deferred again for stated
reasons and not counted as a separate candidate.

---

## 9. The recommended slice: SI-23 (outline)

**SI-23 — A second authored signal for the miniature:
`tiny-predecessor-lean-001`, same target, trained at the
landed head.**

In scope:

1. **A signal registry and dispatch** (§5.2):
   `train_candidate` dispatches on
   `intent.training_signal_id` through a per-signal
   registry carrying the id, stream offsets, correction
   definition, partitions, and the pinned rank/alpha/
   target; any id not in the registry is refused with the
   existing `ValueError` discipline, and the foreign-target
   and foreign-rank/alpha refusals apply per signal,
   unchanged.
2. **Audited stream discipline** (§5.3): signal 2 takes
   two new offsets — one for its sequences, one for its
   factor initialisation — pinned as named constants
   beside the existing pair, with the existing offset
   guard extended to assert all four miniature training
   offsets and both probe offsets are pairwise distinct.
   Neither value is reused from +2, +3, +11, or +17.
3. **Signal 2's at-base first-cycle sanity configuration**
   (§5.6.3): configured with signal 2, a first cycle at
   base records signal 2's pre-training held-out loss at
   base — confirming or refuting the §5.6.3 closed-form
   derivation (0.00140625) by measurement — and trains and
   is judged there, as a sanity configuration. Its
   outcomes are labelled at-base facts about signal 2's
   definition; they are not the slice's decisive claim
   and are never presented as evidence about head
   `d0001`.
4. **Disjointness by construction and by test** (§5.4):
   signal 2's rejection-draw construction rejects against
   the union of both probe suites, signal 1's full 48
   sequences, and signal 2's own draws so far; all five
   obligations of §5.4 are asserted by test, and the
   measured last-token overlap with signal 1's set is
   published as a fact, not assumed away.
5. **Per-signal echo-trap records** (§5.6.3): signal 2's
   pre-training held-out loss is recorded at base
   (item 3) and at `d0001` (item 6), each labelled with
   its head and target source, and the at-`d0001` record
   is stated to recur by construction under
   weights-under-training anchoring wherever it appears.
6. **The decisive two-cycle run**: one reused
   `TinyTransformerMind` driven through
   `si.run.run_cycles` — cycle 1 configured with signal 1
   (the default), landing `d0001`; cycle 2 configured
   with signal 2 by explicit recorded selection (§5.2),
   trained against `d0001` under the unchanged freshness
   contract and `TARGET_ANCHORING`, and judged by the
   unchanged gate. The signal-2 gradient check is re-run
   at head `d0001` (§5.5, §10 criterion 2). The verdict —
   ADMIT, REJECT, or ESCALATE — is reported as measured.
7. **The §5.6.4 diagnostics, published** whatever the
   verdict: per-probe drift contributions and worst
   probes, the alignment diagnostic, the folded updates'
   inner product and norms, composed capability
   regression and echo, and signal 2's held-out records —
   each labelled measurement or derivation, each "not
   evaluated" if it cannot be computed.
8. **Per-cycle published measurements**: gate identity
   drift (seed-absolute, cumulative), capability
   regression (head-incremental, labelled as such), echo
   accuracy per cycle; held-out numbers per cycle labelled
   with their signal id, head, and target source; §6.6
   VERIFY outcomes; stored-bytes growth per cycle; and
   per-cycle CPU cost.
9. **Unwind, honestly conditioned**: if cycle 2 ADMITs,
   rollback of the authored two-delta stack to `d0001`
   and to base is exercised through the existing rollback
   core, with each restored head's probe state reported
   as measured. If cycle 2 does not ADMIT, no authored
   two-delta stack exists, nothing is unwound, and the
   as-built doc says so — SI-21's fixture-proven unwind
   remains the only unwind evidence, labelled as such.
10. **An as-built doc** in the SI-14…SI-21 pattern,
    publishing every number §10 requires and stating
    plainly what is still not claimed.

Explicitly not in SI-23's scope: §11's non-goals.

---

## 10. Acceptance criteria for SI-23

A reviewer must be able to check each criterion against a
test or a published number. SI-23 does not merge on any
criterion being waived, tuned quietly, or reported without
its measurement.

1. **Dispatch and recorded selection.** The registry
   serves exactly the two named signals and refuses every
   other id with the existing `ValueError` discipline;
   foreign-target and foreign-rank/alpha refusals are
   unchanged and apply per signal. Selection is explicit
   and recorded: a test asserts, per cycle on both verdict
   paths, that the journaled `training-hyperparams` note's
   signal id equals the signal the driver configured for
   that cycle. No autonomous, learned, depth-driven, or
   alternating selection exists in the slice.
2. **Gradient check for signal 2, at the decisive head.**
   The finite-difference gradient check is re-run for
   signal 2's loss at head `d0001` — the configuration
   the decisive cycle trains in — and at base; the
   as-built doc publishes the measured agreement and the
   tolerance it was checked against for both. A check run
   only at base does not satisfy this criterion.
3. **Streams audited.** The as-built doc publishes the
   full offset table (§5.3) including signal 2's two new
   named constants, and the extended guard test asserts
   all four miniature training offsets and both probe
   offsets are pairwise distinct. Reuse of +11 or +17 for
   signal 2 fails this criterion by definition.
4. **Disjointness proven.** All five obligations of §5.4
   hold by construction and are asserted by test —
   including signal 2 against signal 1's full 48
   sequences (32 train + 16 held-out) — and the measured
   last-token overlap between the two signals' sets is
   published.
5. **Echo-trap records.** Signal 2's pre-training
   held-out loss is recorded and published at base and at
   `d0001`, each labelled with head and target source;
   the at-base record confirms or refutes the §5.6.3
   derivation (0.00140625) by measurement, and the
   at-`d0001` record is stated, wherever it appears, to
   recur by construction under weights-under-training
   anchoring — not as fresh evidence about the base.
6. **Determinism per cycle.** Each cycle's candidate is
   byte-identical across repeat runs of that cycle
   (canonicalised serialisation, §12), and a full
   two-cycle run repeated from the seed reproduces both
   cycles' records and blob hashes exactly, except any
   explicitly labelled wall-time field.
7. **Gate honesty per cycle.** Each cycle's journaled
   checks 3/4 equal an independent `ProbeContext`
   evaluation of the same candidate against the same
   pre-cycle head (the SI-19 §8.8 equality, per cycle).
   Cycle 2's verdict — ADMIT, REJECT, or ESCALATE — is
   reported as measured. **No criterion requires an
   ADMIT**: SI-23 succeeds by §5's claim — a second
   signal was genuinely authored against the landed head
   under settled semantics and genuinely judged —
   whatever the verdict is.
8. **Cumulative accounting and diagnostics published.**
   The as-built doc publishes, per cycle: identity drift
   (seed-absolute), capability regression
   (head-incremental, labelled as such), echo accuracy vs
   the recorded floor, and the landed head's cumulative
   identity drift after each cycle — plus the full
   §5.6.4 diagnostic set (per-probe contributions and
   worst probes, alignment, folded-update inner product
   and norms), with measurements and derivations labelled
   as such and any uncomputable diagnostic reported "not
   evaluated". No incremental figure is presented as
   cumulative or vice versa (§12).
9. **First-cycle reproduction.** Configured with signal 1
   (the default, recorded selection), a first cycle
   reproduces SI-19/SI-21 exactly — same candidate blob
   (`sha256:2761e8ff…6ca9e`), same gate numbers — so the
   registry and selection machinery is proven
   behaviour-preserving for the existing signal before
   the new signal's results are trusted (§5.2 item 4).
10. **VERIFY, unwind, and untouched scope.** An ADMITted
    cycle 2 passes (or fails and rolls back from) the
    §6.6 probe VERIFY exactly as cycle 1 did, with the
    rollback target being `d0001`; unwind follows §9
    item 9's honest conditioning (authored stack unwound
    and re-measured if one exists; no unwind claimed if
    none does). Exactly two authored cycles are attempted
    in the decisive run; `max_stack_depth`, both
    tolerances, all caps, and every policy default are
    unchanged; `si squash` still fails closed on
    transformer stacks (SI-12 tests green); the full
    existing suite is green; the demo path is unchanged;
    `si serve` still returns 501 for every transformer
    bundle; no CLI wiring is added. Per-cycle
    `train_delta` wall time, total two-cycle wall time,
    and the suite's resulting runtime on this VM are
    measured and published.

---

## 11. Non-goals

Explicit, and binding on SI-23 and on any slice this RFC
is cited to justify:

- **No Glimmer anything.** No Glimmer forward path,
  probes, training, or authorship (§7.2). No candidate in
  this RFC touches the Glimmer seed except to restate
  that it is untouched.
- **No real-base work.** Candidate (a) is deferred with
  audited preconditions (§3); SI-23 builds nothing toward
  it, and no download is performed or scheduled.
- **No anchoring change.** Base-anchored targets (§5.7)
  are deferred again; `TARGET_ANCHORING` stays
  `"weights-under-training"` and SI-23 does not implement
  or test the competing semantics.
- **No other widening.** No new LoRA target (§6.1), no
  additive or multi-target authored candidates (§6.2),
  and no policy default changes — in particular
  `max_targets_per_delta` stays 1 and `allow_additive`
  stays false.
- **No gate-fitting signal.** No successor-lean variant
  at a reduced correction scale, and no signal of any
  kind designed backward from the identity tolerance
  (§5.6.1). Signal 2's correction scale is 0.15, as
  recorded in §5.6.2.
- **No autonomous selection.** The mind does not choose
  its signal; selection is an explicit, recorded,
  constructor-level configuration (§5.2), and no
  perception-driven, depth-driven, or alternating rule
  is built or implied.
- **No transformer squash.** SI-12's fail-closed refusal
  stands throughout SI-23 (§7.1); §7.1 is a deferral
  analysis, not an implementation brief — even if SI-23
  creates the authored stack squash has been waiting
  for.
- **No serving, no CLI wiring.** Transformer bundles keep
  returning 501; `si cycle`/`si run` keep their demo/mock
  minds. SI-23 is a library-and-tests slice like SI-19
  and SI-21.
- **No tolerance loosening, no probe-suite changes, no
  redefinition of signal 1.** Signal 1's id, stream,
  correction, and partitions are byte-for-byte SI-19's.
  If signal 2's cycle is REJECTed, the answer is a later
  RFC's better argument — never a wider gate, and never
  a quieter report.
- **No third cycle, no continual-learning claim.** Exactly
  two authored cycles are attempted in the decisive run,
  on constructed signals, on a constructed model. Two
  signals at depth ≤ 2 prove something about the loop's
  machinery and its gate — nothing about learning at any
  other scale or on any real task.

---

## 12. Honesty and labelling requirements for any future slice

Binding on SI-23 and on any later slice citing this RFC.
Items 1–10 carry RFC SI-20 §12 forward unchanged in
force; items 11–14 are new, from this RFC's findings.

1. **Training numbers are training numbers.** Losses,
   residuals, and held-out signal scores must be labelled
   with the signal id and partition wherever they appear.
   They are never a gate result, never a probe result,
   and never evidence about any base but the miniature.
2. **Measured vs not evaluated survives authorship.** An
   authored candidate's checks 3/4 report exactly what
   the probes measured (`measured` a real float) or that
   they were not evaluated (`measured=None`) — the SI-17
   labelling table applies unchanged, including the
   blind-escalated state, at every cycle, for every
   signal.
3. **Authorship is not admission.** No document, journal
   entry, or report may describe a trained candidate as
   "learned into the bundle", "accepted", or "verified"
   before the gate and (on ADMIT) the §6.6 probe VERIFY
   have actually run and been journaled — per cycle, per
   signal.
4. **Every number carries its source.** A figure in an
   as-built doc is either reproduced from a test/run in
   that slice (with the test named) or cited to the
   earlier doc that measured it. Estimates are labelled
   estimates; derivations from cited facts are labelled
   derivations (as §5.6.3 and §5.6.4 label theirs).
5. **The miniature's labels travel.** "Constructed, not
   trained" describes the *base*; deltas may honestly be
   described as "trained on a constructed signal" — and
   nothing in that phrase may be allowed to blur into a
   claim that the base is a trained model or that either
   signal is a real task.
6. **Failure states are reportable successes of honesty.**
   A signal-2 REJECT, a derivation refuted by its test, a
   VERIFY rollback to `d0001` — each is a publishable,
   journalable outcome. A slice that reports only its
   passes has failed this section regardless of its code.
7. **Serialisation determinism is proven, not assumed.**
   Any code that produces a content-addressed blob —
   including a signal-2 candidate — must prove
   byte-identity across repeat constructions by test
   (SI-19's canonical-header lesson, carried from SI-20
   §12.7).
8. **Head-relative numbers name their head.** Any loss,
   baseline, or score computed against applied weights
   must be recorded with the head (and stack depth) it
   was computed at. The trap SI-20 §12.8 named is live
   for signal 2 exactly as for signal 1: the same figure
   (0.00140625) is a base fact in a first cycle and a
   head-relative construction artifact at `d0001`.
9. **Incremental and cumulative are different claims.**
   Capability regression in this runtime is measured
   against the pre-cycle head; identity drift against
   the seed. Reports must say which they are publishing,
   per number. Summing incremental regressions, or
   quoting the latest cycle's drift as the head's total
   drift, is fabrication by arithmetic.
10. **Composed artifacts do not inherit training labels.**
    A future squashed or otherwise composed delta (§7.1)
    is not "trained": it must be journaled and described
    as a compression with its own measured tolerance and
    probe results, never with its constituents'
    hyperparams presented as its own.
11. **Signal identity is part of every label (new).** A
    training or gate number recorded without its signal
    id is an incomplete record. Signal 2's numbers are
    never presented as signal 1's, as an unnamed "second
    cycle's", or as generic authorship numbers; the two
    signals' held-out figures are never averaged,
    summed, or otherwise blended into a single claim.
12. **Diagnostics are not verdicts (new).** The §5.6.4
    diagnostics — per-probe contributions, alignment,
    folded-update inner products — are published to
    *attribute* a gate outcome after the gate has spoken.
    They are not gate checks, they influence no verdict,
    and no diagnostic may be quoted before a run as a
    predicted ADMIT or REJECT.
13. **Recorded selection is not a choice the mind made
    (new).** SI-23's signal selection is an explicit
    configuration recorded in provenance (§5.2). No
    document may describe it as the mind "choosing",
    "deciding", or "preferring" a signal; the honest
    sentence is that the driver configured the signal and
    the journal shows it.
14. **A derivation is not a prediction (new).** §5.6.4's
    additive-at-the-logits derivation and §5.6.3's
    closed-form loss are labelled derivations from cited
    facts. Neither may be cited — in an as-built doc, a
    PR body, or a report — as evidence that signal 2
    will ADMIT, will REJECT, or has any measured property
    at all. Only SI-23's runs measure those.

---

## 13. Risks and open questions

Each item names what would retire it. Numbers this RFC
deliberately does **not** supply are listed as open
questions to measure — supplying them here would be
inventing them.

1. **Signal 2's gate outcome is genuinely unknown.** Its
   identity drift, capability regression, and echo
   accuracy at head `d0001` are **open questions to
   measure**. The identity budget's remaining headroom
   after cycle 1 is 0.07810781709849834 (arithmetic on
   SI-21's published drift, §5.6.4), so a REJECT is a
   live possibility the slice must be willing to publish
   (§10.7, §12.6). *Retired by:* SI-23's two-cycle run.
2. **The gate-fitting trap (the "smaller successor-lean"
   trap).** §5.6.1 named it: after a measured REJECT, the
   tempting repair is a signal designed backward from
   the tolerance — the same correction at a scale chosen
   because the arithmetic fits the headroom. Such a
   signal would convert the gate from a judge into a
   specification, and every later ADMIT in this project
   would inherit the doubt. This RFC's defence is
   structural, not rhetorical: the recommended signal's
   scale is fixed at the family's recorded 0.15 before
   any measurement, and §11 makes a reduced-scale variant
   an explicit non-goal. *Retired by:* nothing once and
   for all — it is retired per slice, by each future RFC
   naming its signal's design reasons before its results,
   as §5.6.2 does.
3. **The direction argument may fail in the measurement.**
   §5.6.4 is explicit that predecessor direction implies
   nothing about movement alignment on the identity
   probes; the two corrections share a component (both
   move away from the last token), and both updates are
   rank-8 matrices trained over similar hidden-state
   distributions. The composed drift may land near the
   aligned extreme anyway. That outcome would be a
   finding, not an embarrassment: it would say the
   binding constraint is update *size* at the probes, not
   direction. *Retired by:* SI-23's §5.6.4 diagnostics,
   published either way (§10.8).
4. **Stream or disjointness contamination would blur
   both signals' numbers.** A reused offset, or a
   signal-2 sequence colliding with signal 1's training
   partition, would make signal 2's held-out claims
   unreadable — the SI-20 §5.1 warning that motivated
   §5.4's fifth obligation in particular. *Retired by:*
   §10 criteria 3–4's audited table, guard test, and
   construction-level rejection against the union set.
5. **The registry refactor could silently alter signal
   1.** Generalising single-signal dispatch is the kind
   of change that passes its own new tests while moving
   an old behaviour in its last digits. *Retired by:*
   §10 criterion 9 — exact first-cycle reproduction of
   SI-19/SI-21's candidate blob and gate numbers under
   the default recorded selection.
6. **Weights-under-training may be the wrong default for
   a second signal.** §5.5 freezes the anchoring to keep
   SI-23's experiment clean; it does not claim the choice
   is right in general. §5.7's base-anchored route waits
   behind this slice precisely so its eventual evaluation
   can be made against SI-23's measured diagnostics
   rather than against speculation. *Retired by:*
   SI-23's published per-cycle numbers and diagnostics,
   then a later RFC's judgement.
7. **Probe blindness still bounds the judge.** RFC SI-13
   §10.1 stands, now across signals: 24 inputs, 28/30
   tensors measurable, blind set derived and guarded.
   Signal 2's target is the measurable readout, so its
   candidates are judged — but composition across signals
   does not widen the probes' view, and a two-signal
   stack is *evidence* of continuity sampled the same way
   each cycle, not a proof of it.
8. **Depth pressure arrives if SI-23 succeeds.** At
   `max_stack_depth` 8 with transformer squash still
   deferred (§7.1), a landed second authored delta makes
   the stack depth 2 and the compression question real
   for the first time — while the squash gate context
   remains unbuilt. That is a deliberate fail-closed
   posture, recorded so the next RFC meets it as a design
   problem with a real stack to measure, not a surprise.
9. **Candidate (a)'s unknowns remain unknown.** A-1 to
   A-4 are audited unmet (§3); nothing in SI-23 or this
   RFC scopes them, and the download question is an
   operator decision this RFC explicitly does not take.
   Likewise §7.2: success on the miniature renews the
   "why not Glimmer" pressure, and the answer does not
   soften because authorship has widened from one signal
   to two.

---

## 14. Summary

After SI-21, SI can originate a miniature delta, judge
it, land it, and originate a second one on top of it —
and the gate's cumulative, seed-absolute identity
accounting has been measured refusing the second write of
the *same* signal, exactly as derived. Of the candidate
frontiers this RFC evaluated against that evidence: a
small real base stays deferred, its four SI-18
preconditions audited and all still unmet, with a
checkpoint download flagged as an operator decision
rather than a step (§3); sequential authored writes are
done, their question answered by SI-21 (§4); a new target
stays deferred behind a backward derivation nobody has
written (§6.1), and additive / multi-target authorship
stays rejected as a next slice because it would be bought
with a policy loosening (§6.2); transformer squash stays
deferred, its precondition — an authored two-delta
stack — still unmet, SI-21's unwind having run on a
labelled fixture (§7.1); Glimmer probing stays rejected,
not reopened (§7.2); the competing new-semantics route,
base-anchored targets, is deferred again so the
new-signal experiment runs unconfounded (§5.7); and **a
second training signal on the same target —
`tiny-predecessor-lean-001`, explicitly selected,
stream-audited, disjointness-proven, echo-trap-recorded,
under anchoring inherited unchanged — is recommended as
SI-23** (§§5, 9, 10). Its plausibility argument is stated
with its limits in the same section that makes it: drift
composes additively at the logits, identical stacking is
the aligned extreme SI-21 measured, and no ADMIT is
predicted or required. The base stays frozen, the bytes
are never duplicated, no tolerance moves, and nothing is
claimed that a test in this repo cannot check.

---

*RFC SI-22 — design only. Implementation begins, if
approved, at SI-23 as outlined in §9. RFC SI-20 taught
the loop to write on top of its own first write and the
gate told the truth about it; this RFC proposes teaching
it a second thing to say — under the same judge, at the
same scale, with the verdict still not ours to pick.*
