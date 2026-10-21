# SI-20 RFC — The Next Frontier After Delta Authorship
(Design Only)

> **Status: DESIGN ONLY.** This document evaluates candidate
> directions and recommends one next slice. It changes no
> production code, no test, no manifest schema, no policy
> default, and no gate behaviour. Nothing in this RFC is
> implemented unless a section explicitly says it exists today,
> in code, on `main`.
>
> **The one sentence that governs everything below:** SI-19 made
> exactly one mind the author of exactly one trained miniature
> delta, on exactly one named signal, against a head that was
> the base — and this RFC must never be cited as evidence that
> authorship composes across cycles, that any other signal,
> target, or base can be authored, or that any slice proposed
> here exists.
>
> **Scope discipline:** CPU only. No GPU, no cloud, no spend,
> no downloads. Terminology: this is machine learning
> infrastructure throughout.
>
> **Relationship to prior work:** RFC SI-18 evaluated the three
> frontiers open after SI-17 and recommended delta authorship;
> SI-19 built it (PR #24, as-built in
> `docs/SI-19-delta-authorship.md`) and the gate ADMITted the
> authored candidate as measured. This RFC is SI-18's successor
> in the established cadence (SI-13 RFC → SI-14…SI-17 builds →
> SI-18 RFC → SI-19 build): it restates the verified state,
> re-evaluates SI-18's deferred candidate (a) against its own
> §3.7 preconditions, evaluates the two directions SI-19's
> landing makes concrete — sequential authored cycles, and
> wider authorship — and re-examines transformer squash now
> that authored deltas exist. SI-18's rejection of Glimmer
> probing is carried forward, not reopened (§7). The SI-9 RFC's
> §7 risks remain in force and are inherited, not relitigated.

---

## Table of contents

1. [Where SI stands after SI-19](#1-where-si-stands-after-si-19)
2. [Inherited invariants](#2-inherited-invariants)
3. [Candidate (a): a small real safetensors base — precondition audit](#3-candidate-a-a-small-real-safetensors-base--precondition-audit)
4. [Candidate (b): sequential authored self-writes on the miniature](#4-candidate-b-sequential-authored-self-writes-on-the-miniature)
5. [Candidate (c): widening authorship on the miniature](#5-candidate-c-widening-authorship-on-the-miniature)
6. [Candidate (d): transformer squash for miniature deltas](#6-candidate-d-transformer-squash-for-miniature-deltas)
7. [Candidate (e): probing Glimmer — rejected, restated](#7-candidate-e-probing-glimmer--rejected-restated)
8. [Decision summary](#8-decision-summary)
9. [The recommended slice: SI-21 (outline)](#9-the-recommended-slice-si-21-outline)
10. [Acceptance criteria for SI-21](#10-acceptance-criteria-for-si-21)
11. [Non-goals](#11-non-goals)
12. [Honesty and labelling requirements](#12-honesty-and-labelling-requirements-for-any-future-slice)
13. [Risks and open questions](#13-risks-and-open-questions)
14. [Summary](#14-summary)

---

## 1. Where SI stands after SI-19

Everything in this section is verifiable in the code and docs
on `main` at the SI-19 merge (main tip `0a155dc`, the merge of
PR #24). Numbers are cited to the as-built document or test
that produced them; this RFC measures nothing new. Where a
statement below is a *derivation* from cited facts rather than
a measurement, it is labelled a derivation, exactly as SI-18
§3.2 labelled its arithmetic.

**What exists, exactly:**

- **One measurable transformer subject, unchanged.** The
  SI-14 miniature (`TinyTransformerMind`): a constructed,
  deterministically seeded, **not trained** decoder-only
  transformer — 30 tensors, 23,456 elements (SI-14 as-built,
  restated in SI-18 §1). Architecture pinned: vocabulary 32,
  `d_model` 32, 2 layers, 4 heads, head dimension 8, MLP
  hidden 64, maximum sequence 16 (`si/tiny_seed.py::ARCHITECTURE`).
- **Live probe verdicts for miniature bundles (SI-17).**
  Gate checks 3 (identity continuity) and 4 (capability
  floor) decide for real on miniature bundles with a
  validated probe context (`si/probes.py::ProbeContext`):
  probe-fail → REJECT, all-pass → ADMIT-eligible, anything
  unevaluated → ESCALATE, exclusively probe-blind candidates
  → blind-escalated (`si/probes.py::derive_probe_blind_targets`;
  the blind set is exactly both layers' `k_proj.bias`).
  Bundle tolerances are both 0.25 (`identity_tol`,
  `capability_eps`; SI-14's measured derivation, recorded in
  `si/tiny_seed.py::seed_tiny_bundle`). The probe suites are
  8 identity + 16 capability (echo-task) entries; recorded
  base echo accuracy 16/16 = 1.0, echo floor 0.9375.
- **One authored delta, trained and ADMITted (SI-19).**
  `TinyTransformerMind.train_delta` no longer refuses: it
  performs real CPU gradient-based training in
  `si/tiny_training.py::train_candidate` — one named signal
  (`tiny-successor-lean-001`, target-logits MSE, stream
  `MINI_SEED + 11`, 32 train / 16 held-out probe-disjoint
  examples), one LoRA target (`lm_head.weight`, rank 8,
  alpha 8), analytic gradients (RFC SI-18 §4.4 route (i))
  finite-difference checked, candidate built through
  `construct_transformer_delta`, refusals intact for any
  other signal, target list, or rank/alpha, and for a
  trained candidate whose held-out loss does not improve.
  `propose` emits exactly that signal's deterministic
  transformer intent (`si/tiny_transformer.py::TinyTransformerMind.propose`).
  SI-19's published measurements (as-built doc, author's
  runs): base held-out signal loss **0.00140625** → trained
  **3.21e-06** (3.2118398935406443e-06, 438× lower, 16/16
  held-out examples improved); training loss
  0.0028961365437885785 → 1.824004756500067e-06 over 2,000
  full-batch steps; gradient check worst absolute
  disagreement **3.807e-08** over all 512 factor entries at
  the pinned tolerance 1e-5; candidate norm **0.2356**
  (0.23555380990005192; caps 1.0), blob **2,360 bytes**
  (cap 4 MiB); one `train_delta` call **0.99 s** wall on
  this VM, marginal peak memory < 1 MB.
- **The gate's verdict on the authored candidate, as
  measured (SI-19 as-built §8.8): ADMIT.** Identity drift
  **0.1719** (0.17189218290150166) ≤ tol 0.25; capability
  regression **0.1657** (0.1657181391492486) ≤ eps 0.25;
  echo accuracy **16/16** vs floor 0.9375. An end-to-end
  `run_cycle` driven by the real mind materialised the
  candidate as `d0001` and the §6.6 probe VERIFY passed; a
  test asserts the journaled verdict and measurements equal
  an independent `ProbeContext` evaluation of the same
  candidate (`tests/test_tiny_training.py::test_end_to_end_cycle_with_the_real_mind`).
- **A serialisation finding with a fix in place (SI-19).**
  safetensors' `save()` emits the header's `__metadata__`
  mapping in hash-map order, observed to flip between two
  calls in the same process over bit-identical factor bytes;
  authored candidates are therefore re-serialised with a
  pinned, sorted header in
  `si/tiny_training.py::_canonical_blob_bytes` — tensor data
  bytes untouched, the rebound pair re-validated through
  `validate_transformer_delta`. Two `train_delta` runs on
  the same intent now produce byte-identical blobs (§12
  carries the general lesson forward).
- **Suite state:** 292 passed, 2 skipped on `main` at the
  SI-19 merge (SI-19 as-built, author's runs: the SI-17
  baseline of 280 passed, 2 skipped, plus SI-19's 12 new
  tests, zero regressions).

**What does not exist, exactly (SI-19, "What is still NOT
claimed", verified against the code on `main`):**

- **No second authored cycle has ever run.** Every SI-19
  measurement is at head = base (`si/tiny_training.py`
  module docstring: "every SI-19 measurement is at head =
  base"). No test trains against a head carrying an authored
  delta; the only `d0002` records in the test suite are
  hand-built fixture stacks (`tests/test_squash_transformer.py`,
  `tests/test_tiny_transformer.py`), not authored deltas.
- **No authorship for any other base, target, or signal.**
  Exactly one signal id, one target, one bundle kind.
  Additive targets and multi-target candidates remain
  unauthored — and, at the policy layer, inadmissible as
  authored: `max_targets_per_delta` is 1 and
  `allow_additive` is false in the default policy
  (`si/bundle.py::DEFAULT_POLICY`), enforced by the gate's
  transformer safety check (`si/gate.py`, the
  `max_targets_per_delta` subcheck).
- **No CLI path drives the miniature mind.** `si cycle` and
  `si run` accept only `--mind demo|mock`
  (`si/cli.py::_make_mind` constructs `DemoMind`/`MockMind`
  only). Authored cycles today are a library surface
  (`si.loop.run_cycle`, `si.run.run_cycles`) exercised by
  tests, not a CLI surface.
- **No Glimmer anything.** No Glimmer forward path, probes,
  training, or authorship; the Glimmer seed is untouched
  and its deltas still escalate with checks 3/4 "not
  evaluated" (§7).
- **No serving.** `si serve` returns 501 for every
  transformer bundle, miniature included (`si/serve.py`).
- **No transformer squash.** SI-12's fail-closed refusal
  stands (§6): `si/squash.py::squash_bundle` raises
  `SquashError` before any write if any record on the
  resolved active chain has kind `transformer-delta-v1`.

The frontier question is therefore precise in a new way:
the loop can now *originate* a miniature delta and *judge*
it — once, from the base. It has never originated a second
one on top of the first, it can originate nothing wider,
and it still cannot judge anything larger. Which edge is
the honest next step?

---

## 2. Inherited invariants

Not negotiable inside this RFC (SI-9 §1.3, RFC SI-13 §2,
RFC SI-18 §2, restated for the candidates):

1. **Base is frozen forever.** All change is a stack of
   deltas; rollback is moving `head`.
2. **Never duplicate bytes** (Mark's design rule). A blob
   whose hash exists is never rewritten or copied; seed
   imports by reference where the pattern applies
   (`stored_bytes: 0` — the SI-8 Glimmer seed stores zero
   tensor bytes against 59,553,253,376 logical bytes,
   SI-9 §0). Any candidate below that needs a copy of a
   base has failed this rule before it starts.
3. **No ungated write.** Authorship produces *candidates*.
   The gate — and only the gate — admits, rejects, or
   escalates them, and SI-17's verdict mapping and
   blind-set guard apply to authored candidates exactly as
   to fixtures, at every cycle, not just the first.
4. **Honest labels are load-bearing.** "Not evaluated" stays
   distinguishable from a measured 0.0; a training loss
   stays distinguishable from a gate result; and — new in
   this RFC, forced by §4 — a number measured at one head
   stays labelled with that head (§12).
5. **Fail closed.** A missing, mismatched, or un-runnable
   measurement is "not evaluated" — never a pass, never a
   silent skip.
6. **CPU only, existing dependencies only.** The runtime's
   dependencies are `numpy` and `safetensors`
   (`requirements.txt`, `pyproject.toml`). No GPU, no
   network, no downloads, no spend, no new heavy dependency.
7. **Determinism where tests need it.** Seeded RNG, fixed
   inputs, no sampling, no wall-clock dependence in any
   measured or trained value — and, per SI-19's finding,
   deterministic *serialisation* wherever a blob hash is
   content addressing (§12.7).

---

## 3. Candidate (a): a small real safetensors base — precondition audit

**Verdict up front: DEFERRED** — again, and now with an
audit instead of an assertion. SI-18 §3.7 made this
candidate schedulable only when four preconditions (A-1 to
A-4) all exist. SI-19 changed none of them: it built
authorship *for the miniature*, using the miniature's
executor, the miniature's binding, and a constructed
signal. This section checks each precondition against the
code on `main` today, then states exactly what a
measurement-only slice could and could not honestly retire.

### 3.1 Precondition A-1: feasibility measurements for one named candidate base family

**Status: NOT MET — nothing has been measured, and no base
has been named.**

- No feasibility note exists in the repo. `docs/` contains
  no measurement of forward wall time, assembled memory, or
  verification cost for any real checkpoint family; SI-18
  §3.2's ladder (measure first, let the numbers choose the
  size) has not been climbed one rung.
- SI-19's CPU numbers do not transfer. Its 0.99 s
  `train_delta` wall time and < 1 MB marginal peak are
  properties of a 23,456-element miniature whose readout
  gradient needs no forward re-run per step
  (`si/tiny_training.py`: hidden states are cached because
  the readout's input cannot depend on the readout). They
  say nothing about a real base, and citing them toward A-1
  would be exactly the unlabelled transfer SI-18 §10.4
  forbids.
- The one real-base cost figure on record is unchanged:
  one full `load_bundle` verification of the by-reference
  Glimmer base took ~29 minutes (1,756.9 s) of streaming
  reads on this VM (SI-9 §0). That is Glimmer's figure, at
  Glimmer's scale; a smaller base's verification cost
  remains an open measurement, as SI-18 §3.4 recorded.

### 3.2 Precondition A-2: a binding design for `reference-v1` bases

**Status: NOT MET — the binding still cannot be stated for
a referenced base, and SI-19 did not touch it.**

Verified in `si/probes.py::load_bound_probe_artifact` on
`main` today, exactly as SI-18 §3.3 recorded:

1. **Kind gate.** `is_miniature_probe_bundle` requires
   `base.kind == "tiny-transformer-miniature-v1"`
   (`si/tiny_seed.py::TINY_KIND`); `load_bound_probe_artifact`
   raises `ProbeBindingError` for any other kind. A real
   base fails here first, correctly.
2. **Recomputation source.** The binding hash is recomputed
   from the manifest's `base.tensor_blobs` (the copy-mode
   per-tensor map) and the function raises if that map is
   absent. A `reference-v1` base carries `base.tensor_refs`
   instead (`si/bundle.py`, `si/seed.py` — per-tensor
   external refs with absolute path, byte offset/length,
   dtype, shape, content hash). The delta layer already
   reads both maps (`si/transformer_delta.py::base_tensor_map`
   accepts `tensor_refs` or `tensor_blobs`), but the *probe
   binding* has no reference-mode variant: no schema design,
   no tamper analysis, no code.
3. **Architecture exactness.** Both the artifact's and the
   manifest's descriptors must pass
   `si/tiny_transformer.py::validate_architecture` — exact
   equality with the miniature descriptor. Any real
   descriptor fails, by design.

No slice since SI-18 has proposed, let alone built, the
reference-mode binding variant. A-2 is a design problem
that remains exactly where SI-18 left it.

### 3.3 Precondition A-3: a capability-suite design for a trained base

**Status: NOT MET — still an unsolved design problem, and
SI-19 sharpens why.**

- Startup validation (`si/probes.py::ProbeContext._validate_startup`)
  still presupposes the miniature at every step: it loads
  base tensors by `tiny_tensor_shapes()` (exactly 30 names),
  requires worst baseline deviation ≤ 1e-6
  (`si/probes.py::BASELINE_TOL`) against baselines that are
  0.0 by construction, and requires the measured base echo
  accuracy to *equal* the recorded accuracy exactly. A
  trained base has no echo behaviour to reproduce and no
  construction to record baselines from (SI-18 §3.4).
- SI-19 adds a caution, not a solution: its signal works
  *because* its target is the base's own logits plus a
  recorded correction — a construction available only
  because the base is constructed and its forward is
  runnable. A trained base's capability floor needs
  behaviour specified independently of the base's own
  outputs, recorded by a process that does not yet exist.
  Nothing in SI-19's signal design generalises into that
  process, and this RFC does not pretend otherwise.

### 3.4 Precondition A-4: an executor design scoped to one family, with SI-15-grade numeric proof

**Status: NOT MET — the executor is still a miniature
executor, by refusal.**

`si/tiny_transformer.py::forward` validates the exact
30-tensor inventory at pinned shapes (`_validate_weights`)
and refuses any architecture descriptor other than the
miniature's (`validate_architecture`); its compute uses
module-level constants from that descriptor. SI-19's one
addition — the forward replica in `si/tiny_training.py`
that also returns the final hidden state — is proven
bit-identical to this same miniature forward by test and
generalises nothing. No second-family executor design,
hand-computed reference cases, or naive-reimplementation
proof obligations exist for any real architecture.

### 3.5 What a measurement-only slice could honestly retire — and what it could not

A future slice that *only measures* is the cheapest honest
step toward (a), but its scope must be stated without
inflation:

**It could retire, without any download:**

- **Scaling measurements on locally constructed safetensors
  files.** Checkpoints of candidate *shapes* can be
  generated deterministically in-repo (the runtime already
  constructs safetensors bases: `si/tiny_seed.py`,
  `si/seed.py`). Streaming-hash verification cost, per-tensor
  read cost, and assembled-memory growth under the float64
  compute discipline (SI-15) can be measured as functions of
  tensor count and element count. These are measurements of
  *this runtime's I/O and memory behaviour on synthetic
  files* — labelled synthetic, they bound the harness's
  overheads honestly. They are not measurements of any real
  model.
- **The A-2 binding design on paper.** The reference-mode
  binding variant (what is hashed, from `tensor_refs`, with
  what tamper analysis) is a design artifact; it needs no
  checkpoint at all. A design note could retire the
  *unstated* part of A-2, leaving implementation for later.
- **Nothing about Glimmer beyond what is recorded.** The
  Glimmer checkpoint already on disk is the one real
  checkpoint available without a download, and its
  verification cost is already measured (1,756.9 s, SI-9
  §0). It is a 30B multimodal base, not a "small real base"
  candidate; no A-1 ladder rung is climbed by re-measuring
  it, and §7 forbids building toward probing it.

**It could not retire:**

- **A-1 for any named real base.** Naming a candidate
  family, verifying its configuration and tokenizer against
  the actual checkpoint, and measuring *its* forward cost
  all require the checkpoint itself. **Obtaining one means
  a download** — network access and disk this project's
  autonomous scope (invariant 6: no network, no downloads)
  does not include. That is a blocker to flag for the
  operator, not a step to take: SI-18 §3.2's ladder starts
  with a checkpoint in hand, and this RFC does not schedule
  acquiring one.
- **A-3 in any form.** A capability design for a trained
  base cannot be validated against synthetic files; an
  unvalidated design is a draft, and must be labelled one.
- **Any executor claim.** Synthetic-shape timing cannot
  prove a second family's forward *correct* — SI-15-grade
  proof needs the family's real semantics and, for
  baselines, its real weights.

### 3.6 Verdict on (a): DEFERRED, preconditions audited

| Precondition (SI-18 §3.7) | Status today | Changed by SI-19? |
|---|---|---|
| A-1 feasibility measurements, one named family | Not met — none exist | No |
| A-2 `reference-v1` binding design + tamper analysis | Not met — binding unstatable for referenced bases (`si/probes.py`) | No |
| A-3 capability-suite design for a trained base | Not met — unsolved design problem | No (sharpened, §3.3) |
| A-4 one-family executor design + numeric proof | Not met — executor refuses all but the miniature | No |

Candidate (a) remains the long-term direction and remains
unschedulable. A measurement-only slice (§3.5) is a
legitimate *future* candidate — it is not recommended as
SI-21 because §4's candidate is buildable now with
instruments that already exist, while §3.5's slice retires
overheads, not any of A-1 to A-4 as stated.

---

## 4. Candidate (b): sequential authored self-writes on the miniature

**Verdict up front: RECOMMENDED as the next slice (SI-21).**
SI-19 proved a mind can author a delta the gate can judge —
once, from the base. A self-writing file's defining behaviour
is that the *second* write happens on top of the first. Today
that behaviour is unproven, and — §4.2's finding — the code's
two ways of attempting it do not even agree on what "on top
of" means. This is the only candidate whose unknowns are all
testable with instruments that already exist.

### 4.1 What composition machinery already exists

- **The chain folds oldest-first.** Apply-at-load
  (`si/apply.py::load_applied_tensor`) folds every active
  transformer delta targeting a tensor into the base value,
  in chain order, through the shared `fold_target`; the
  miniature forward consumes the assembled result
  (`si/tiny_transformer.py::assemble_weights` — one
  apply-at-load call per tensor). A second delta on
  `lm_head.weight` composes arithmetically with no format
  change: the applied readout is
  `W + s₁·B₁A₁ + s₂·B₂A₂`.
- **The loop already materialises a landed authored delta
  and can run again.** `si/loop.py::_run_miniature_cycle`
  re-stamps a fresh runtime delta id and parent onto the
  authored record, writes the blob, advances `head`,
  journals via `record_transformer_delta`, and VERIFYs with
  real probes. `si/run.py::run_cycles` reuses one mind and
  one gate across cycles, counts verdicts, and halts on the
  first REJECT or ESCALATE. Nothing in the loop *forbids* a
  second cycle.
- **Training already assembles at a head.**
  `si/tiny_training.py::train_candidate` assembles the
  bundle's applied weights via `assemble_weights` (base +
  active chain at the manifest it is given) and its module
  docstring states the intent: "Training runs against the
  bundle's applied weights at its current head." The
  analytic gradient survives composition *for this target*:
  `analytic_gradients` takes the readout weight as a
  parameter, so an applied (already-folded) `lm_head` serves
  as `W` in `W_eff = W + scale·B@A` exactly as the base
  readout did — the readout is still the final operation,
  and its input still does not depend on it.
- **The signal re-derives from whatever weights it is
  given.** `build_training_signal(weights)` computes each
  example's target as the *given* weight set's last-position
  logits plus the recorded successor-lean correction
  (`si/tiny_training.py`). At head = base that is SI-19's
  definition. At any other head it is a different target —
  §4.3's semantics question.

### 4.2 The freshness finding: two drivers, two different second cycles

Verified in the code on `main`; this is the finding SI-21
exists to close.

- **The mind holds a manifest snapshot.**
  `TinyTransformerMind.__init__` stores the manifest it is
  constructed with (`self._manifest`), and `train_delta`
  delegates to `train_candidate(self._bundle_dir,
  self._manifest, …)` — the snapshot, never refreshed.
  `propose`, asymmetrically, prefers the *context's*
  manifest for head/parents when the context carries one
  (`si/tiny_transformer.py::TinyTransformerMind.propose`
  reads `ctx.manifest` first). Training and proposing can
  therefore look at different heads inside one cycle.
- **Driver 1 — the same mind object reused** (the
  `si/run.py::run_cycles` shape): in cycle 2 the snapshot
  still says head = base with zero deltas. Training
  assembles the *base* weights, re-derives SI-19's original
  targets, starts from the same pinned initial factors
  (stream `MINI_SEED + 17`), and — derivation, from SI-19's
  tested determinism (§8.5: same intent + same manifest →
  byte-identical candidate) — produces a candidate whose
  blob is *byte-identical to `d0001`'s*. The loop re-stamps
  it as `d0002` with parent `d0001`; `_write_blob` writes
  zero new bytes (content-addressed, skip-if-exists); the
  gate then measures it *correctly composed* — the probe
  context is built fresh from the current manifest
  (`_run_miniature_cycle` constructs `ProbeContext` per
  cycle) — i.e. it judges base + `d0001` + the same update
  again. A reader of the journal would see a "second
  authored delta" that authored nothing new.
- **Driver 2 — a fresh mind per cycle** (the test suite's
  shape: `tests/test_tiny_training.py::_mind` loads the
  bundle anew): training assembles at head = `d0001`, and
  the signal's targets are re-derived from the *head's*
  logits plus the same correction — the readout is asked to
  lean a further 0.15-step toward the successor, from where
  it already leans. The recorded "base held-out loss" for
  cycle 2 would again read 0.00140625 — *derivation*: the
  loss of the training weights against targets built from
  those same weights plus the correction is the correction's
  closed form `2 · 0.15² / 32` at any head — but in cycle 2
  that number is a statement about the head and a moved
  target, not about the base. Unlabelled, it misleads.

Neither driver is "the" behaviour today; the repo has never
chosen. A multi-cycle claim made before this choice is
recorded would be a claim about whichever driver happened
to run — the precise failure §12 exists to prevent.

### 4.3 The semantics SI-21 must name before it trains anything

1. **Freshness contract.** Which manifest trains — the
   mind's snapshot or the cycle's verified manifest — must
   be decided, implemented in exactly one place, and proven
   by a test that can tell the two apart (e.g. cycle 2's
   assembled readout equals the applied tensor including
   `d0001`). Silent staleness (Driver 1 as an accident) is
   the one unacceptable outcome; if a stale-mind driver is
   ever kept deliberately, its identical-blob consequence
   must be tested and documented, not discovered.
2. **Target anchoring.** Are cycle-≥2 targets derived from
   the current head's logits (lean further, per
   `build_training_signal` as written) or anchored to the
   seed base's logits (lean to a fixed destination, with
   later cycles correcting the residual)? Both are
   definable; they are different signals wearing one id.
   This RFC recommends SI-21 keep the as-written semantics
   (targets from the weights under training) *and record
   the choice*, because changing the signal definition in
   the same slice that tests composition would confound the
   experiment: SI-21's question is "does composition work",
   not "which anchoring is better". The hyperparams must
   record the head trained against and the target-source
   head (§10 criterion 2).
3. **What cycle 2's improvement claim may say.** Held-out
   improvement in cycle 2 is improvement *of the head,
   against targets derived from that head* — never evidence
   about the base, and never additive with cycle 1's 438×
   figure. (§12.8–12.9 make this binding.)

### 4.4 The gate's split accounting — why identity is the binding constraint

Verified in `si/gate.py` and `si/probes.py`; the two probe
checks do not accumulate the same way:

- **Identity continuity is seed-absolute.**
  `Gate._identity_check` compares the candidate-overlay
  forward against the suites' *recorded expected logits*
  (the seed base's outputs) within `identity_tol` 0.25.
  Every cycle's drift is measured against the same origin,
  so drift **accumulates** across cycles. SI-19 spent
  0.1719 of the 0.25 budget on cycle 1 — leaving 0.0781 of
  headroom (arithmetic on SI-19's published numbers,
  labelled as such).
- **Capability regression is head-incremental.**
  `Gate._capability_check` compares against
  `ctx.capability_baseline()`, and
  `ProbeContext.capability_baseline` measures the errors at
  the *current head* (its docstring: at head = base these
  are the recorded 0.0 baselines; on a non-base head they
  are the pre-candidate head's real errors). A cycle is
  charged only for regression *since the previous head* —
  plus the echo-accuracy arm, which is absolute (accuracy
  vs the recorded floor 0.9375, recomputed on the
  candidate-overlay forward in
  `Gate._transformer_probe_checks`).
- **§6.6 VERIFY mirrors the split.**
  `si/loop.py::_verify_probe_head` judges identity against
  the seed's expected logits and capability against a
  baseline the caller captures at the pre-cycle head,
  labelling the report `baseline_source: "pre-cycle head
  (caller-captured)"`. A failed cycle-2 VERIFY rolls back
  to the pre-cycle head (`d0001`), not to base — the
  miniature branch of `_run_miniature_cycle` passes
  `to=prev_head` to the rollback core, with `mind=None`
  (its demo probe re-check cannot run on a miniature
  bundle, and claiming it did would be fabrication).

**A derivation SI-21 must confirm or refute.** Under
Driver 1 (identical candidate stacked), the composed
readout update is twice cycle 1's, and since the readout
is the final operation over hidden states that do not
depend on it, every probe's logit change doubles in exact
arithmetic; identity drift would therefore measure
2 × 0.17189218290150166 = **0.3438 > 0.25 → REJECT** (the
float32 fold in `si/apply.py::fold_target` may move the
measured value in its last digits — §10 criterion 5's
side-by-side publication exists to catch exactly that).
This is a *derivation* from SI-19's published drift and
the architecture (§4.1), not a measurement — no second
cycle has run. SI-21's tests must state the measured value,
whatever it is; a disagreement would mean the derivation's
premises (freshness, overlay order, fold arithmetic) are
wrong somewhere, which is itself a publishable finding.
Under Driver 2 (retrained at the head), cycle 2's drift is
**an open question to measure** (§13): the second update is
trained, not copied, and this RFC invents no figure for it.

### 4.5 What else a second cycle stresses

- **Journal and provenance.** Each cycle journals its own
  self-edit, gate report, and §6.6 VERIFY provenance
  (`_run_miniature_cycle` records the VERIFY report as a
  provenance note on the delta). Cycle 2's journal-vs-
  independent-evaluation equality (SI-19 §8.8's test, per
  cycle) is the check that the journal still tells the
  truth when the head is not the base.
- **Stack depth.** The gate's transformer safety check
  enforces `max_stack_depth` 8 (`si/bundle.py::DEFAULT_POLICY`;
  depth = `ctx.stack_depth() + 1`). Two cycles are far
  inside it; SI-21 states a depth ambition of exactly 2 and
  does not touch the cap. Depth becomes a real question
  only for a later many-cycle slice — which would also meet
  §6's squash wall.
- **Rollback granularity.** With two authored deltas
  landed, rollback to `d0001` and to base are distinct,
  testable states; SI-19 exercised neither for authored
  deltas (its VERIFY passed). SI-21 should prove the
  landed stack can be unwound one authored step at a time
  through the existing rollback core — or report what it
  measures instead.
- **Stored bytes.** Cycle 1 grew the blob store by exactly
  its delta's 2,360 bytes (SI-19 §8.9 test). Cycle 2's
  accounting differs by driver (Driver 1: zero new bytes —
  the identical blob already exists; Driver 2: the new
  candidate's bytes) and must be asserted per cycle, not
  assumed.

### 4.6 Verdict on (b): RECOMMENDED

It is the smallest slice that tests the project's defining
claim — a file that keeps writing itself — at the only
scale where the gate can judge the result. Its central
unknowns (freshness, anchoring, cumulative drift) are
correctness questions with testable answers, its
instruments (probes, gate, VERIFY, rollback) already exist
unchanged, and both of its possible outcomes are honest
results: a second ADMIT proves composition; a measured
REJECT proves the gate's cumulative accounting works
exactly as §4.4 describes. Neither outcome requires a
tolerance to move.

---

## 5. Candidate (c): widening authorship on the miniature

**Verdict up front: DEFERRED** — all three sub-directions,
for three different reasons (§5.4's table). Widening is the
natural sequel to SI-19, but each widening step either
depends on §4's unproven composition semantics, needs a new
gradient derivation, or needs a policy change this project
does not make as a shortcut. SI-19's scope — one signal,
one target — was a deliberate starting point (RFC SI-18
§4.5, §9), not an oversight.

### 5.1 Sub-direction (c1): a second training signal (same target)

The mechanically lightest widening, and still not free:

- **Dispatch is single-signal today.** `train_candidate`
  accepts exactly `TRAINING_SIGNAL_ID`
  (`"tiny-successor-lean-001"`) and raises `ValueError` on
  any other id; `propose` emits exactly that signal's
  intent. A second signal means a dispatch rule (the
  DemoMind precedent — serve exactly the named signals,
  refuse all others — generalises, but the *selection*
  question is new: which signal does an autonomous cycle
  propose, and who decides?).
- **Stream discipline is auditable but must be kept.**
  Offsets from `MINI_SEED` in use on `main`: +2 identity,
  +3 capability (`si/tiny_seed.py`), +11 signal, +17 factor
  initialisation (`si/tiny_training.py`). A second signal
  needs its own unused offset, its own disjointness proof
  against *both* probe suites (RFC SI-18 §4.3's rule), and —
  new question — a stated relationship to the first
  signal's examples (overlap between two training signals
  is not circular the way probe overlap is, but an
  unexamined overlap would blur what each signal's
  held-out numbers mean).
- **The echo-trap test applies per signal.** Each new
  signal must record the base's (or head's — §4.3) score
  before training and be measurably imperfect there.
- **The blocking dependency is §4.** A second signal's
  delta, proposed on a bundle that already carries the
  first signal's delta, trains against a non-base head —
  the exact semantics §4 says are unchosen. Widening before
  composition is understood would stack two unproven
  semantics and make any failure unattributable.

### 5.2 Sub-direction (c2): a new LoRA target (non-readout tensor)

- **The gradient is the work.** Route (i)'s exactness in
  SI-19 rests on one fact, recorded in
  `si/tiny_training.py`'s docstring: `lm_head.weight` is
  the final operation, so no gradient with respect to any
  other tensor is ever needed and hidden states can be
  cached. For *any* earlier tensor — an attention
  projection, an MLP projection, the embeddings — the
  gradient runs backward through RMSNorm, causal softmax
  attention, and SwiGLU: the derivation RFC SI-18 §4.4
  called error-prone in exactly those places. It ships, if
  ever, only with the same finite-difference gradient
  check per target, at a tolerance pinned from measured
  agreement — none of which exists for any second target.
  The forward replica (`forward_with_final_hidden`) exposes
  only the final hidden state; intermediate activations a
  real backward pass needs are not currently returned
  anywhere.
- **The judge can see most targets — that part is ready.**
  Per-class sensitivity is 28 of 30 tensors measurable
  (SI-16, restated in SI-18 §1); the blind set is exactly
  the two `k_proj.bias` tensors, which are 1-D and in any
  case not LoRA targets (LoRA requires a 2-D base,
  enforced in `si/gate.py`'s safety check). A new-target
  slice would not be blocked by probe blindness except by
  its own target choice.
- **Target choice needs its own recorded reason**, as
  SI-19 recorded `lm_head.weight`'s (measurability, cap
  fit, derivability). Only the first two reasons transfer
  to a second target; the third must be re-earned.

### 5.3 Sub-direction (c3): additive and multi-target authored candidates

- **The format is ready; the policy is not — deliberately.**
  `construct_transformer_delta` already builds multi-target
  records, `si/apply.py` folds per-target updates in chain
  order, and the gate already handles mixed
  blind + measurable candidates honestly (the blind-note
  path in `Gate._transformer_probe_checks`: ADMIT-eligibility
  rests on the measurable targets' evidence, blind targets
  named). What forbids authored multi-target candidates is
  policy: `max_targets_per_delta` = 1 and
  `allow_additive` = false (`si/bundle.py::DEFAULT_POLICY`),
  enforced by the gate. Widening authorship this way means
  *changing policy defaults to admit what the gate currently
  refuses* — the one move §11 and RFC SI-18 §9 both forbid
  as a route around a gate outcome. A future slice could
  argue a policy change on evidence; this RFC does not,
  because no evidence about authored multi-target behaviour
  exists yet.
- **Joint training is a new optimisation problem, not two
  copies of SI-19.** Two targets trained against one loss
  interact (each target's gradient depends on the other's
  current factors through the forward); the readout-only
  caching trick does not survive a second, earlier target;
  aggregate norm accounting (`aggregate_norm`: root-sum-
  square over per-target norms, recomputed by the gate)
  would need the training loop's norm guard re-derived for
  the joint case.
- **Additive authorship's natural homes are the worst
  targets.** Additive targets must be 1-D
  (`si/transformer_delta.py::validate_transformer_delta`);
  the miniature's 1-D tensors are norms and biases —
  including the probe-blind `k_proj.bias` pair, an authored
  delta on which could never be judged. An additive slice
  would start by excluding its most obvious targets.

### 5.4 Verdict on (c): DEFERRED, with sub-verdicts

| Sub-direction | Verdict | One-line reason |
|---|---|---|
| (c1) Second signal, same target | DEFERRED — nearest of the three | Blocked on §4's composition semantics; dispatch, stream, and selection rules are new design (§5.1). |
| (c2) New LoRA target | DEFERRED | Needs a full backward derivation + per-target gradient check; only measurability transfers from SI-19 (§5.2). |
| (c3) Additive / multi-target | REJECTED as a next slice | Requires loosening `max_targets_per_delta` / `allow_additive` policy defaults with zero authored-behaviour evidence (§5.3). |

---

## 6. Candidate (d): transformer squash for miniature deltas

**Verdict up front: DEFERRED** — but, unlike at SI-12,
genuinely discussable now, for one concrete reason SI-19
created (§6.2). It is not schedulable before §4 lands: today
there is no authored stack deeper than one to squash.

### 6.1 Current state, verified

`si/squash.py::squash_bundle` resolves the active chain,
returns the `{"squashed": False}` no-op when depth ≤ `keep`,
and otherwise raises `SquashError` — before any blob write,
journal open, or manifest change — if *any* record on the
resolved chain (squash segment or kept tail) has kind
`transformer-delta-v1`, naming the delta ids and citing the
deferral (RFC SI-9 §6.4). SI-12's as-built records why the
refusal covers the whole chain (the kept tail is
materialised and gate-loaded by demo-shaped machinery that
a transformer record cannot pass through) and that inactive
records do not block a demo squash. The demo squash path
itself — truncated-SVD recompression, tolerance verify
before commit, full gate, head id unchanged — is
implemented and tested for demo deltas only.

### 6.2 What SI-19 changed about the discussion

Pre-SI-19, transformer squash meant composing *arbitrary*
fixture deltas: mixed targets, mixed kinds (LoRA and
additive), per-target layouts the format does not guess at
(SI-12). Authored deltas are a far narrower population, and
for that population composition is exact algebra:

- Every authored delta is a single-target LoRA on the *same*
  tensor (`lm_head.weight`, rank ≤ 8). Two such updates
  compose as `s₁·B₁A₁ + s₂·B₂A₂` — a single update of rank
  ≤ 16 on one named tensor, foldable by the existing
  `si/apply.py::fold_target` arithmetic. No per-target
  layout needs inventing for the same-target case.
- The recompression problem is then the demo squash's own:
  rank ≤ 16 composed → rank ≤ 8 (`rank_cap`) is *lossy*, so
  a tolerance verify of the SI-3 kind (squashed head vs
  pre-squash head, max-abs below a stated tolerance, fail
  closed with nothing left behind) is the honest shape —
  with the miniature's probes as a second, stronger verify
  the demo path never had.

### 6.3 What still blocks it

1. **There is nothing authored to squash.** The deepest
   authored stack on record is depth 1 (§1). Squashing
   fixture stacks (the `tests/test_squash_transformer.py`
   population) would prove format handling, not authored
   composition. Candidate (b) is the prerequisite: it
   creates the first real authored stack and measures how
   its members were trained.
2. **The squash gate context is demo-shaped.**
   `si/squash.py::_make_squash_mind` constructs only
   `DemoMind`/`MockMind`, and `SquashContext` serves demo
   probes and a demo forward. A miniature squash candidate
   must be gated by the *probe* context (checks 3/4
   measured, SI-17 mapping) or it would be gated by
   instruments that cannot see it — a new context, with
   the §6.2 binding and the blind-set guard, is design
   work that does not exist.
3. **Provenance and identity of the squashed delta.**
   Demo squash records `squashed_from` provenance from the
   squashed deltas' journal seqs and leaves `head`
   unchanged; an authored squash must additionally answer
   what the squashed delta *is* — it is not trained, so it
   carries no `hyperparams` of its own, and §12 forbids
   dressing it in the training numbers of its constituents.
   Its journal story (compression of d0001+d0002, tolerance
   measured, probes re-run) must be statable without
   borrowing authorship's labels.
4. **Scope discipline.** Even a future slice must be
   authored-miniature-same-target only. General transformer
   squash (mixed targets, additive members, any other base)
   remains undefined and refused; the SI-12 whole-chain
   refusal stays in force for everything outside the future
   slice's proven scope.

### 6.4 Verdict on (d): DEFERRED

Discussable since SI-19 (same-target authored composition
is exact; §6.2), schedulable only after SI-21 produces an
authored stack and the measurements a squash design must
respect (per-delta norms, drift contributions, tolerance
headroom). A squash RFC written before those numbers exist
would be designing against fixtures — the thing SI-12
refused to do.

---

## 7. Candidate (e): probing Glimmer — REJECTED, restated

Glimmer probing was rejected by RFC SI-13 §7/§9, restated
by SI-18 §5, and is **not reopened here**. SI-19 changes
nothing about it: authorship, like probing before it, is
miniature-only by construction — the training assembles
weights through the miniature executor
(`validate_architecture` refuses Glimmer's descriptor), the
signal is defined over a vocabulary of 32, and the gate's
probe binding cannot be stated for the Glimmer bundle
(§3.2). The Glimmer seed's deltas still escalate with
checks 3/4 "not evaluated", its stored bytes are still 0
against 59,553,253,376 logical bytes (SI-9 §0), and one
full verification of that by-reference base still stands
measured at ~29 minutes (1,756.9 s) before any forward
compute (SI-9 §0/§7.1). A partial forward over Glimmer
would be a claim about a truncated computation, not about
the model — fabrication, as SI-18 §5 recorded. Any future
Glimmer proposal needs its own RFC and its own feasibility
measurements; it gets no momentum from this document.

---

## 8. Decision summary

| Candidate | Verdict | One-line reason |
|---|---|---|
| (a) Small real safetensors base | **DEFERRED** | SI-18 §3.7 audit: A-1 to A-4 all still unmet; naming a real base needs a download this scope forbids (§3). |
| (b) Sequential authored self-writes (miniature) | **RECOMMENDED — SI-21** | Tests the defining claim (a second write on the first) with existing gate/probe/VERIFY instruments; its unknowns are testable correctness questions (§4). |
| (c) Widening authorship | **DEFERRED** (c3 rejected as next slice) | Second signal blocked on (b)'s semantics; new target needs a new gradient derivation; additive/multi-target needs a policy loosening (§5). |
| (d) Transformer squash (miniature) | **DEFERRED** | Now discussable — authored same-target composition is exact — but needs (b)'s authored stack and a probe-based squash gate context (§6). |
| (e) Glimmer probing | **REJECTED** | Rejected at SI-13 §7/§9, restated at SI-18 §5; SI-19 changes nothing (§7). Not reopened. |

---

## 9. The recommended slice: SI-21 (outline)

**SI-21 — Sequential authored self-writes for the
miniature: a second authored cycle against the landed
head.**

In scope:

1. **A recorded freshness contract** (§4.3.1): exactly one
   manifest trains in a cycle — the cycle's verified
   manifest — implemented in one place (mind refresh at
   cycle start, or `train_candidate` receiving the cycle
   manifest), with the stale-snapshot path closed for the
   driver the slice tests. `propose`'s existing
   context-manifest preference and training must agree.
2. **Named cycle-≥2 signal semantics** (§4.3.2): the
   SI-19 signal definition unchanged (same id, stream
   `MINI_SEED + 11`, correction, partitions); targets
   derived from the weights under training, with the
   choice and its alternative (base-anchored targets)
   recorded in the as-built doc. Hyperparams for every
   cycle record the head trained against, the
   target-source head, and the stack depth.
3. **A two-cycle end-to-end run driven by the real
   `TinyTransformerMind`** through the loop (two
   `run_cycle` calls or `run_cycles`, per the freshness
   contract), landing `d0001` then attempting `d0002`,
   with per-cycle journal-vs-independent-`ProbeContext`
   equality asserted as in SI-19 §8.8.
4. **Per-cycle published measurements** (§10.5): gate
   drift/regression/echo per cycle, cumulative identity
   drift of the landed head vs the seed, §6.6 VERIFY
   outcomes, held-out numbers per cycle labelled with
   their head and target source, the §4.4 derivation's
   measured confirmation or refutation where it applies,
   stored-bytes growth per cycle, and per-cycle CPU cost.
5. **Unwind proof**: rollback from the two-delta head to
   `d0001` and to base through the existing rollback core,
   with the restored heads' probe states reported as
   measured.
6. **An as-built doc** in the SI-14…SI-19 pattern,
   publishing every number §10 requires and stating
   plainly what is still not claimed.

Explicitly not in SI-21's scope: §11's non-goals.

---

## 10. Acceptance criteria for SI-21

A reviewer must be able to check each criterion against a
test or a published number. SI-21 does not merge on any
criterion being waived, tuned quietly, or reported without
its measurement.

1. **Freshness proven.** A test distinguishes the two
   drivers of §4.2 and asserts the training in cycle 2
   assembled weights at head = `d0001` (e.g. the assembled
   readout equals `si.apply.load_applied_tensor` for
   `lm_head.weight` at that head, or cycle 2's candidate
   differs from cycle 1's in the way retraining at the
   head predicts). A driver that silently trains from a
   stale snapshot fails this criterion by definition.
2. **Semantics recorded.** The as-built doc names the
   target-anchoring choice (§4.3.2); cycle-≥2 hyperparams
   journal `head` trained against, target-source head,
   and stack depth alongside the SI-19 fields. The signal
   definition (id, stream, correction scale, partitions,
   disjointness) is byte-for-byte SI-19's.
3. **Determinism per cycle.** Each cycle's candidate is
   byte-identical across repeat runs of that cycle
   (canonicalised serialisation, §12.7), and a full
   two-cycle run repeated from the seed reproduces both
   landed records and blob hashes exactly, except any
   explicitly labelled wall-time field.
4. **Gate honesty per cycle.** Each cycle's journaled
   checks 3/4 equal an independent `ProbeContext`
   evaluation of the same candidate against the same
   pre-cycle head (the SI-19 §8.8 equality, per cycle).
   Cycle 2's verdict — ADMIT, REJECT, or ESCALATE — is
   reported as measured. **No criterion requires cycle 2
   to ADMIT**: composition succeeds by §4.6's claim —
   the second write was genuinely authored against the
   landed head and genuinely judged — whatever the
   verdict is.
5. **Cumulative accounting published.** The as-built doc
   publishes, per cycle: identity drift (seed-absolute),
   capability regression (head-incremental, labelled as
   such), echo accuracy vs floor, and the landed head's
   cumulative identity drift after each cycle. No
   incremental figure is presented as cumulative or vice
   versa (§12.9). Where the §4.4 identical-stack
   derivation applies to any tested driver, its predicted
   value (0.3438) and the measured value are published
   side by side.
6. **VERIFY and unwind.** An ADMITted cycle 2 passes (or
   fails and rolls back from) the §6.6 probe VERIFY
   exactly as cycle 1 did, with the rollback target being
   `d0001`; rollback of the landed two-delta stack to
   `d0001` and to base is exercised, and each restored
   head's state is reported as measured (probe re-runs or
   an explicit not-evaluated label — never an assumed
   pass).
7. **Held-out honesty in cycle 2.** Cycle 2's held-out
   improvement is claimed only against cycle 2's recorded
   pre-training loss at its head, with the target source
   labelled; the as-built doc states explicitly that the
   0.00140625 figure, if it recurs, recurs by construction
   (targets re-derived from the training weights) and is
   not fresh evidence about the seed base.
8. **Depth and policy untouched.** Exactly two authored
   cycles are attempted; `max_stack_depth`, both
   tolerances, all caps, and every policy default are
   unchanged; `si squash` still fails closed on the
   resulting transformer stack (SI-12 tests green).
9. **No regressions, no widenings.** The full existing
   suite is green; run as a *first* cycle, SI-19's
   outcomes reproduce unchanged (same candidate blob,
   same gate numbers as SI-19 published); the demo path
   is byte-for-byte unchanged; `si serve` still returns
   501 for every transformer bundle; the CLI's mind
   choices are unchanged (no CLI wiring is added).
10. **CPU budget published.** Per-cycle `train_delta`
    wall time, total two-cycle wall time, and the suite's
    resulting runtime on this VM are measured and
    published in the as-built doc. (No target numbers are
    set here; §13 records cost growth per cycle as an
    open measurement.)

---

## 11. Non-goals

Explicit, and binding on SI-21 and on any slice this RFC
is cited to justify:

- **No Glimmer anything.** No Glimmer forward path, probes,
  training, or authorship (§7). No candidate in this RFC
  touches the Glimmer seed except to restate that it is
  untouched.
- **No real-base work.** Candidate (a) is deferred with
  audited preconditions (§3); SI-21 builds nothing toward
  it, and no download is performed or scheduled.
- **No widening.** No second signal, no new target, no
  additive or multi-target authored candidates, no policy
  default changes (§5) in SI-21.
- **No transformer squash.** SI-12's fail-closed refusal
  stands throughout SI-21; §6 is a deferral analysis, not
  an implementation brief.
- **No serving, no CLI wiring.** Transformer bundles keep
  returning 501; `si cycle`/`si run` keep their demo/mock
  minds. SI-21 is a library-and-tests slice like SI-19.
- **No tolerance loosening, no probe-suite changes, no
  signal redefinition.** If cycle 2 is REJECTed, the
  answer is a later slice's better semantics or signal —
  never a wider gate, and never a quieter report.
- **No claim that two cycles are continual learning.** Two
  authored writes on a constructed signal, on a
  constructed model, prove composition of the loop's
  machinery — nothing about learning at any other scale
  or on any real task.

---

## 12. Honesty and labelling requirements for any future slice

Binding on SI-21 and on any later slice citing this RFC.
Items 1–6 carry RFC SI-18 §10 forward unchanged in force;
items 7–10 are new, from SI-19's and this RFC's findings.

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
   blind-escalated state, at every cycle.
3. **Authorship is not admission.** No document, journal
   entry, or report may describe a trained candidate as
   "learned into the bundle", "accepted", or "verified"
   before the gate and (on ADMIT) the §6.6 probe VERIFY
   have actually run and been journaled — per cycle.
4. **Every number carries its source.** A figure in an
   as-built doc is either reproduced from a test/run in
   that slice (with the test named) or cited to the
   earlier doc that measured it. Estimates are labelled
   estimates; derivations from cited facts are labelled
   derivations (as §1, §4.2, and §4.4 label theirs).
5. **The miniature's labels travel.** "Constructed, not
   trained" describes the *base*; deltas may honestly be
   described as "trained on a constructed signal" — and
   nothing in that phrase may be allowed to blur into a
   claim that the base is a trained model or that the
   signal is a real task.
6. **Failure states are reportable successes of honesty.**
   A cycle-2 REJECT, a derivation refuted by its test, a
   VERIFY rollback to `d0001` — each is a publishable,
   journalable outcome. A slice that reports only its
   passes has failed this section regardless of its code.
7. **Serialisation determinism is proven, not assumed
   (SI-19's lesson).** safetensors `save()` was observed
   flipping its header `__metadata__` key order between
   calls in one process over bit-identical tensor bytes;
   SI-19's fix canonicalises the authored blob's header
   (`si/tiny_training.py::_canonical_blob_bytes`, tensor
   bytes untouched, pair re-validated). Any future code
   that produces a content-addressed blob must prove
   byte-identity across repeat constructions by test —
   a hash that depends on hash-map order is a
   non-determinism bug, not a cosmetic one.
8. **Head-relative numbers name their head.** Any loss,
   baseline, or score computed against applied weights
   must be recorded with the head (and stack depth) it
   was computed at. A "base loss" recorded at a non-base
   head is a mislabel, and §4.2 shows the trap is live:
   the same figure (0.00140625) is a base fact in cycle 1
   and a head-relative construction artifact in cycle 2.
9. **Incremental and cumulative are different claims.**
   Capability regression in this runtime is measured
   against the pre-cycle head; identity drift against
   the seed. Reports must say which they are publishing,
   per number (§4.4). Summing incremental regressions, or
   quoting the latest cycle's drift as the head's total
   drift, is fabrication by arithmetic.
10. **Composed artifacts do not inherit training labels.**
    A future squashed or otherwise composed delta (§6) is
    not "trained": it must be journaled and described as
    a compression with its own measured tolerance and
    probe results, never with its constituents'
    hyperparams presented as its own.

---

## 13. Risks and open questions

Each item names what would retire it. Numbers this RFC
deliberately does **not** supply are listed as open
questions to measure — supplying them here would be
inventing them.

1. **Cycle 2's gate outcome is genuinely unknown.** Under
   the recommended semantics, cycle 2 trains a further
   lean from the head; its identity drift, capability
   regression, and echo accuracy are **open questions to
   measure**. The identity budget's remaining headroom
   after cycle 1 is 0.0781 (derivation from SI-19's
   published drift, §4.4), so a REJECT is a live
   possibility the slice must be willing to publish
   (§10.4, §12.6). *Retired by:* SI-21's two-cycle run.
2. **The §4.4 derivation could be wrong in an interesting
   way.** The exact-doubling prediction (0.3438) assumes
   Driver 1's candidate is byte-identical, the overlay
   folds it on top of `d0001`, and hidden states are
   readout-independent. Each premise is code-verified
   here, but the composition has never executed. *Retired
   by:* SI-21 criterion 5's side-by-side publication.
3. **Freshness fix placement is a real design choice.**
   Refreshing the mind's manifest inside the loop,
   passing the cycle manifest into `train_delta`, or
   making `TinyTransformerMind` re-load on each call have
   different blast radii for SI-15/SI-19's tested surface
   (including the bundle-backed refusal semantics).
   *Retired by:* SI-21's design note + criterion 1's test.
4. **Target anchoring may be the wrong default.** Head-
   derived targets make each cycle lean *further* by
   construction; base-anchored targets would make later
   cycles corrective. This RFC recommends freezing the
   as-written semantics for one slice to keep the
   composition experiment clean (§4.3.2) — it does not
   claim the choice is right for a third cycle. *Retired
   by:* SI-21's published per-cycle numbers, then a later
   RFC's judgement.
5. **Probe blindness still bounds the judge.** RFC SI-13
   §10.1 stands, now across cycles: 24 inputs, 28/30
   tensors measurable, blind set derived and guarded. A
   stack of authored deltas is *evidence* of continuity,
   sampled the same way each cycle — composition does not
   widen the probes' view.
6. **Depth pressure arrives quickly after SI-21.** At
   `max_stack_depth` 8 with no transformer squash (§6),
   an autonomous many-cycle run has a hard ceiling and no
   compression path. That is a deliberate fail-closed
   posture, recorded so a future slice meets it as a
   design problem, not a surprise.
7. **Candidate (a)'s unknowns remain unknown.** A-1 to
   A-4 are audited unmet (§3); nothing in SI-21 or this
   RFC scopes them, and the download question (§3.5) is
   an operator decision this RFC explicitly does not
   take.
8. **Scope creep toward Glimmer, again.** SI-19 renews
   the pressure ("the loop now *writes* transformer
   deltas autonomously — why not…"). The answer is §7,
   and it does not soften because authorship composes:
   a trained miniature delta is evidence about the
   miniature, full stop.

---

## 14. Summary

After SI-19, SI can originate a transformer delta honestly
at exactly one scale, against exactly one head — the base —
and has never originated the second one. Of the candidate
frontiers: generalising probes to a small real base stays
deferred, its four SI-18 preconditions audited and all
still unmet, with a checkpoint download flagged as an
operator decision rather than a step (§3); widening
authorship stays deferred — its cheapest branch is blocked
on composition semantics, its deepest needs a new gradient
derivation, and its widest needs a policy loosening this
project does not make as a shortcut (§5); transformer
squash becomes discussable for authored same-target deltas
but waits on a real authored stack and a probe-based
squash gate (§6); Glimmer probing stays rejected (§7); and
**sequential authored self-writes are recommended as
SI-21** — one freshness contract, the SI-19 signal
unchanged, a second cycle genuinely trained against the
landed head and genuinely judged by the existing gate, with
cumulative and incremental accounting published side by
side (§§4, 9, 10). The base stays frozen, the bytes are
never duplicated, and nothing is claimed that a test in
this repo cannot check.

---

*RFC SI-20 — design only. Implementation begins, if
approved, at SI-21 as outlined in §9. RFC SI-18 taught the
mind how to write; this RFC proposes proving the file can
be written on — a second time, on top of the first, with
the gate still able to tell the truth about both.*
