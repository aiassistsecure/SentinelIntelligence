# SI-18 RFC — The Next Frontier After the Miniature Admission Flip
(Design Only)

> **Status: DESIGN ONLY.** This document evaluates candidate
> directions and recommends one next slice. It changes no
> production code, no test, no manifest schema, no policy
> default, and no gate behaviour. Nothing in this RFC is
> implemented unless a section explicitly says it exists today,
> in code, on `main`.
>
> **The one sentence that governs everything below:** SI-17 made
> the gate's identity/capability verdicts live for exactly one
> honestly-scoped subject — the constructed miniature — and this
> RFC must never be cited as evidence that any other base can be
> probed, that any mind can author a transformer delta, or that
> any slice proposed here exists.
>
> **Scope discipline:** CPU only. No GPU, no cloud, no spend,
> no serving change. Terminology: this is machine learning
> infrastructure throughout.
>
> **Relationship to prior work:** RFC SI-13 (the probe harness)
> is fully discharged: SI-14 (miniature seed + probe artifacts),
> SI-15 (forward path), SI-16 (probe suites + gate wiring,
> shadow mode), and SI-17 (admission flip for miniature bundles
> only, PR #22) have all merged. SI-17's own closing section
> ("What is still NOT claimed") names the three open edges this
> RFC evaluates: no forward path for any non-miniature base, no
> training and no delta authorship, and no Glimmer probes. The
> SI-9 RFC's §7 risks (verification cost, numerics, escalation
> fatigue) remain in force and are inherited, not relitigated.

---

## Table of contents

1. [Where SI stands after SI-17](#1-where-si-stands-after-si-17)
2. [Inherited invariants](#2-inherited-invariants)
3. [Candidate (a): a small real safetensors base](#3-candidate-a-generalising-the-probe-harness-to-a-small-real-safetensors-base)
4. [Candidate (b): delta authorship for the miniature](#4-candidate-b-delta-authorship--a-mind-proposing-its-own-deltas)
5. [Candidate (c): probing Glimmer](#5-candidate-c-probing-glimmer--rejected-restated)
6. [Decision summary](#6-decision-summary)
7. [The recommended slice: SI-19 (outline)](#7-the-recommended-slice-si-19-outline)
8. [Acceptance criteria for SI-19](#8-acceptance-criteria-for-si-19)
9. [Non-goals](#9-non-goals)
10. [Honesty and labelling requirements](#10-honesty-and-labelling-requirements-for-any-future-slice)
11. [Risks and open questions](#11-risks-and-open-questions)
12. [Summary](#12-summary)

---

## 1. Where SI stands after SI-17

Everything in this section is verifiable in the code and docs
on `main` at the SI-17 merge. Numbers are cited to the as-built
document or test that produced them; this RFC measures nothing
new.

**What exists, exactly:**

- **One measurable transformer subject.** The SI-14 miniature
  (`TinyTransformerMind`): a constructed, deterministically
  seeded, **not trained** decoder-only transformer — 30 tensors,
  23,456 elements, 93,824 float32 bytes of base (SI-14 as-built,
  which also records 93,920 unique content-addressed blob bytes
  stored). Architecture pinned by RFC SI-13 §3.1 as amended by
  SI-14: vocabulary 32, `d_model` 32, 2 layers, 4 heads, head
  dimension 8, MLP hidden 64, maximum sequence 16, learned
  positional table included.
- **A real forward path for that subject only.**
  `si/tiny_transformer.py` (SI-15) executes the miniature over
  base + active deltas assembled through apply-at-load, plus the
  validated in-memory candidate overlay (RFC SI-13 §4.2). It
  refuses — by exact-match validation — any architecture
  descriptor other than the miniature's and any weight set that
  is not exactly the miniature's 30-tensor inventory at its
  pinned shapes. It is a miniature executor, not a general one.
- **Probe suites with seed-time baselines.** 8 identity entries
  + 16 capability (echo-task) entries, recorded in the bundle's
  `probe_suite.json` artifact (22,565 bytes; SI-14). The base's
  recorded echo accuracy is 16/16 = 1.0; the recorded echo floor
  is 0.9375 (base accuracy − 1/16); the miniature bundle's
  `identity_tol` and `capability_eps` are both 0.25, derived
  from measured logit margins (identity min margin ≈2.716,
  capability min margin ≈2.756) bounded by the demo-era cap
  (SI-14's measured table).
- **Live verdicts for miniature bundles only (SI-17).** With a
  validated probe context, gate checks 3 (identity continuity)
  and 4 (capability floor) decide for real: probe-fail →
  REJECT, all-pass → ADMIT-eligible, anything unevaluated →
  ESCALATE. The autonomous loop materialises ADMITted miniature
  deltas and VERIFYs them with real probes against the landed
  head, rolling back on failure (§6.6 of RFC SI-13). SI-17's
  reproduced reference outcomes, now with live verdicts: zero
  delta → ADMIT, drift exactly 0.0; benign delta → ADMIT, drift
  5.774e-05; destructive delta → REJECT, drift 1.0175 > tol
  0.25; echo attack → REJECT, accuracy arm 0.5 vs floor 0.9375;
  startup validation worst baseline deviation 0.0 over 24
  entries (SI-17 as-built).
- **A derived blind set, guarded.** Per-class sensitivity is
  28 of 30 tensors measurable; the blind set is exactly both
  layers' `k_proj.bias` (drift mathematically 0.0 — a constant
  key bias shifts all of a query's scores equally and softmax
  is shift-invariant). SI-17 derives the set from the
  architecture (`derive_probe_blind_targets`) and escalates
  blind-only candidates as *blind-escalated*, never as a
  measured pass.
- **Suite state:** 280 passed, 2 skipped on `main` at the
  SI-17 merge (SI-17 as-built, author's runs).

**What does not exist, exactly (SI-17, "What is still NOT
claimed"):**

- No Glimmer probes, no Glimmer forward path. The Glimmer 30B
  seed's deltas still escalate with checks 3/4 "not evaluated".
- No serve change: `si serve` returns 501 for every transformer
  bundle, miniature included.
- **No training, no delta authorship.** The miniature cycle
  consumes hand-built fixture candidates — in tests, a scripted
  mind (`_MiniatureScriptMind` in
  `tests/test_admission_flip.py`) returns a candidate built by
  `construct_transformer_delta` — and
  `TinyTransformerMind.train_delta` still refuses
  (`si/tiny_transformer.py`: it raises `NotImplementedError`,
  as does its `propose`).
- No transformer squash; SI-12's fail-closed refusal stands.

The frontier question is therefore precise: the loop can now
*judge* a miniature delta it is handed. It cannot *write* one,
and it cannot judge anything larger. Which edge is the honest
next step?

---

## 2. Inherited invariants

Not negotiable inside this RFC (SI-9 §1.3, RFC SI-13 §2,
restated for the candidates):

1. **Base is frozen forever.** All change is a stack of deltas;
   rollback is moving `head`.
2. **Never duplicate bytes** (Mark's design rule). A blob whose
   hash exists is never rewritten or copied; seed imports by
   reference where the pattern applies (`stored_bytes: 0` —
   the SI-8 Glimmer seed stores zero tensor bytes against
   59,553,253,376 logical bytes, SI-9 §0). Any candidate below
   that needs a copy of a base has failed this rule before it
   starts.
3. **No ungated write.** Authorship, if it ever exists,
   produces *candidates*. The gate — and only the gate —
   admits, rejects, or escalates them, and SI-17's verdict
   mapping and blind-set guard apply to authored candidates
   exactly as to fixtures.
4. **Honest labels are load-bearing.** "Not evaluated" stays
   distinguishable from a measured 0.0; an apply-smoke report
   stays distinguishable from a probe report; a training loss
   stays distinguishable from a gate result (§10).
5. **Fail closed.** A missing, mismatched, or un-runnable
   measurement is "not evaluated" — never a pass, never a
   silent skip.
6. **CPU only, existing dependencies only.** The runtime's
   dependencies are `numpy` and `safetensors`
   (`requirements.txt`, `pyproject.toml`). No GPU, no network,
   no spend, no new heavy dependency.
7. **Determinism where tests need it.** Seeded RNG, fixed
   inputs, no sampling, no wall-clock dependence in any
   measured or trained value.

---

## 3. Candidate (a): generalising the probe harness to a small real safetensors base

**Verdict up front: DEFERRED** — not rejected in principle,
not schedulable today. The gap between the miniature executor
and a real base is not a parameter tweak; it is a new executor,
a new binding variant, a new baseline story, and a set of
feasibility measurements that do not exist yet. §3.7 lists the
preconditions that would make this candidate schedulable.

### 3.1 What "a small real base" would actually require

Seed-by-reference for storage is the *solved* part: SI-8's
`reference-v1` already seeds a real checkpoint with
`stored_bytes: 0`, per-tensor refs, and full hash verification,
and SI-9a/9b/9c deltas, the structural gate suite, and
per-tensor apply-at-load already work against such a base
(the SI-9c real-checkpoint proof applied a delta to one real
Glimmer vision tensor, `[1536, 1536]`, read-only). None of
that is measurement. Probing a real base additionally needs
all of:

- a forward executor for *that* architecture (§3.5);
- probe suites defined against *that* base — inputs in its
  token space, expected outputs from its forward path (§3.2);
- baselines recorded for that base and reproducible at
  harness startup (§3.4);
- a §6.2-style binding that admits that bundle kind (§3.3).

Each is currently miniature-shaped in code, by design.

### 3.2 What base size/architecture could a CPU forward path honestly support?

**This RFC names no model and no parameter count as
supported, because no measurement supports one.** What can be
stated honestly is the bounding argument, from cited facts:

- The miniature is cheap *because* it is 23,456 elements:
  RFC SI-13 §4.5 estimates a full forward at a few million
  multiply-adds worst case (an RFC estimate, not a fresh
  measurement) and a full probe evaluation — 24 entries, one
  candidate forward each — as tens of forwards. Every cost
  scales from there: per-token compute grows with layer count,
  width, and MLP hidden size; attention scores grow with the
  square of sequence length; and this runtime's forward widens
  float32 weights to **float64** for compute (SI-15's recorded
  arithmetic discipline), so an assembled real weight set
  costs, by simple arithmetic on element counts, 4 bytes per
  element assembled and 8 bytes per element in the float64
  compute copy — before activations. Whether that discipline
  can survive at real scale, or must be re-derived (with a
  new baseline-reproduction story), is an open question
  (§11), not a detail.
- The probe artifact scales with **vocabulary**, not with the
  miniature's comfort: expected outputs are full last-position
  logit vectors. At the miniature's vocabulary of 32, 24
  entries are tens of KiB (22,565 bytes as built). By the same
  arithmetic, a base with Glimmer's vocabulary — `lm_head`
  shape `[202048, 6656]`, SI-9 §1.1 — would need
  24 × 202,048 float32 values ≈ 19.4 MB of expected logits
  alone. That is not prohibitive; it is a derivation from
  cited shapes, shown to make the scaling concrete, and it is
  *not* a measurement of any candidate base.
- Architecture, not just size, bounds feasibility: §3.5.

So the honest answer to "what size?" is a **measurement
ladder**, to be climbed by a future feasibility slice before
any RFC names a base: (i) pick a candidate real checkpoint
family whose architecture a numpy executor can express at
all; (ii) measure — on this VM, CPU only — single-forward
wall time, peak memory of the assembled weight set, and
startup-validation cost at the shortest sequence length that
could carry a probe suite; (iii) publish those numbers and
let them choose the size. Any slice that names a base before
step (ii) is guessing, and this project's rule is that
guesses do not ship as claims.

### 3.3 What breaks in the §6.2 binding at real scale

RFC SI-13 §6.2 permits probe evaluation only if all its rules
hold, verified from the bundle context, never the candidate.
As implemented (`si/probes.py::load_bound_probe_artifact`),
every rule is miniature-specific:

1. **Kind gate.** `ProbeContext` serves only bundles whose
   base kind is the miniature kind
   (`is_miniature_probe_bundle` checks
   `kind == "tiny-transformer-miniature-v1"`); any other kind
   raises the binding error that becomes "not evaluated". A
   real base fails here first, correctly.
2. **Artifact presence and hash.** A by-reference real seed
   (Glimmer's included) carries no `probe_artifact` record.
   Minting one requires baselines from a forward pass over
   that base (§3.4) — the thing that does not exist.
3. **Base binding is recomputed from the wrong map.** The
   implementation recomputes the binding hash from the
   manifest's `base.tensor_blobs` (the copy-mode per-tensor
   map) and requires tensor count / total elements to match
   it. A `reference-v1` base carries `base.tensor_refs`
   instead; there is no `tensor_blobs` map to recompute from,
   so the binding cannot even be *stated* for a referenced
   base without a binding variant — a schema-level change,
   with its own tamper analysis, not a configuration.
4. **Architecture exactness (§6.2 rule 3).** Both the
   artifact's and the manifest's descriptors must equal the
   miniature descriptor exactly
   (`validate_architecture` in `si/tiny_transformer.py`).
   Any real descriptor fails, by design. Generalising this
   rule means defining what "the architecture this executor
   runs" is for a second kind — per-kind descriptors, per-kind
   inventories — i.e. a second harness, honestly labelled as
   one, not a loosened check on the first.

### 3.4 What breaks in the §6.3 startup validation at real scale

Startup validation (`ProbeContext._validate_startup`) is what
makes a probe result count: the base alone, re-run through
the forward path, must reproduce its recorded baselines.
As implemented, it presupposes the miniature at every step:

- **Inventory.** It loads base tensors by the miniature's
  shape map (`tiny_tensor_shapes()` — exactly 30 names) and
  calls the miniature `forward`. A real base's tensor names,
  count, and shapes are a different inventory; the call
  refuses before it computes.
- **Baseline form.** It requires worst baseline deviation
  ≤ 1e-6 (`BASELINE_TOL` in `si/probes.py`; SI-15 proved the
  miniature's reproduction is in fact bit-identical on this
  runtime) against baselines that are **0.0 by construction**
  (expected outputs *are* the base's outputs, recorded at
  seed time). A real base has no construction: its baselines
  would be genuine measurements of a trained model's outputs,
  and whether a 1e-6 reproduction tolerance is attainable —
  across bf16 source weights widened to float32, a float64
  compute path, and a real depth of layers — is an open
  question to measure, not an assumption to inherit.
- **Capability baseline form.** Startup validation also
  requires the measured base echo accuracy to *equal* the
  recorded accuracy exactly. Echo is a constructed task for a
  constructed base (base accuracy 1.0, SI-14). A real base
  has no echo behaviour to reproduce; a real capability
  suite — what inputs, what expected outputs, what floor,
  recorded by whom and when — is an unsolved design problem
  for a trained base, and inventing one by renaming the
  miniature's would be fabrication.
- **Cost.** Startup validation re-runs every suite entry
  through the forward path on every harness start. At
  miniature scale that is noise. At real scale it is a full
  forward per entry over an assembled real weight set —
  *after* bundle verification, which for the by-reference
  Glimmer base alone was measured at ~29 minutes
  (1,756.9 s) for one full `load_bundle` verification
  (SI-9 §0). That figure is Glimmer's, not a candidate
  base's; a smaller base's verification cost is an open
  measurement (§11). The structural point stands: probing
  inherits verification cost plus forward cost, and both
  must be measured per base before probing is claimed.

### 3.5 What breaks in the forward executor at real scale

`si/tiny_transformer.py` computes with module-level constants
taken from the miniature descriptor (vocabulary, width,
layers, heads, head dimension, sequence cap) and implements
exactly one architecture family member: pre-norm RMSNorm,
learned positional embeddings, attention biases present,
embeddings untied, SwiGLU MLP, causal scaled dot-product
attention. Real checkpoints differ on axes this code treats
as constants — Glimmer's own recorded config (bundle
manifest, SI-9 §0/§1.1) is one example of the variance:
hidden size 6656, intermediate size 19968, head dimension
128, **no attention biases**, sliding-window and full
attention layers interleaved, per-layer rotary parameters,
bfloat16 weights, a vision tower beside the language tower.
A generic real base substitutes its own list (grouped-query
attention, tied embeddings, rotary instead of learned
positions, different norm or activation details, a tokenizer
the miniature does not have — miniature probe inputs are raw
token-id lists in a vocabulary of 32; a real base's inputs
must be produced by *its* tokenizer over *its* vocabulary).
Each substitution is executor work plus numeric-proof work
of the SI-15 kind (hand-computed reference cases, an
independent naive reimplementation, bit-identical repeat
runs). None of it is impossible; all of it is unbuilt and
unmeasured, and a "generalised harness" that quietly ran a
*wrong-family* forward would produce drift numbers about a
computation that is not the base — the exact fabrication
§6.2 exists to prevent.

### 3.6 The seed-by-reference interaction

Storage is not the blocker: a real base can be seeded by
reference with `stored_bytes: 0` (invariant 2), and probe
artifacts are derived measurement data, not base tensor
bytes — storing them beside a referenced base breaches
nothing. The costs move elsewhere: the referenced source
must remain at its recorded absolute path for any read to
succeed (SI-9 §5.2's stated price of the seed); every probe
evaluation assembles the *whole* applied weight set in
memory (the forward consumes a full weight dict), so the
per-tensor frugality of apply-at-load does not survive
contact with a full forward pass; and every source-file
read goes through the verified path, whose cost is §3.4's
open measurement.

### 3.7 Verdict on (a): DEFERRED, with preconditions

Candidate (a) is the project's long-term direction — a
measurement harness that only ever measures a constructed
toy proves measurement is possible, not that it is useful —
but it is not an honest next slice. It becomes schedulable
when **all** of the following exist:

- **A-1.** A feasibility note with real measurements (§3.2's
  ladder) for one named candidate base family: forward wall
  time, assembled memory, verification cost, on this VM.
- **A-2.** A binding design for `reference-v1` bases (the
  §3.3 rule-3 problem) with its tamper analysis.
- **A-3.** A capability-suite design for a *trained* base
  (the §3.4 problem): what behaviour is the floor under, and
  how are its baselines recorded without a construction?
- **A-4.** An executor design scoped to that one family,
  with SI-15-grade numeric proof obligations.

Until then, any "real-base probes" claim would rest on
unmeasured cost, an unstated binding, and an undefined
capability floor. Deferred — explicitly, with the path back
written down.

---

## 4. Candidate (b): delta authorship — a mind proposing its own deltas

**Verdict up front: RECOMMENDED as the next slice (SI-19).**
It is the only candidate that is buildable from machinery
that exists, measurable with instruments that exist, and
honest about the one thing SI still cannot do at any scale:
originate a self-write instead of judging hand-built ones.

### 4.1 What exists, and what refuses

- `TinyTransformerMind.train_delta` raises
  `NotImplementedError` (inference-only: no gradient code, no
  optimiser, no loss anywhere in `si/`), and its `propose`
  raises the same way (`si/tiny_transformer.py`, RFC SI-13
  §4.4's honesty rule: refuse explicitly rather than
  fabricate). This RFC proposes changing exactly those two
  refusals, for the miniature only, under §4.3–§4.6's rules.
- The miniature loop already calls both:
  `si/loop.py::_run_miniature_cycle` runs PERCEIVE → PROPOSE
  → `mind.train_delta(intent)` → GATE → JOURNAL →
  MATERIALIZE → §6.6 probe VERIFY. Today the only minds that
  can drive it are test script minds returning fixture
  candidates. Authorship is the missing plug in an existing
  socket, not a new loop.
- The candidate contract is fixed: a transformer candidate is
  a `{"record", "blob_bytes"}` pair built and validated by
  `construct_transformer_delta` against the verified manifest
  (base binding copied from the manifest; callers never
  supply it; construction never emits an invalid delta).
  `train_delta` must return exactly that shape for a
  transformer intent — not the demo adapter dict.

### 4.2 The DemoMind precedent — what transfers, what does not

`DemoMind` (`si/minds.py`) is the project's one existing
honest author, and its discipline is the template:

- **A named training signal.** `train_delta` refuses any
  intent whose `training_signal_id` is not the mind's one
  known signal (`ValueError` on an unknown id). Authorship
  for the miniature must likewise serve exactly the signal(s)
  its design names, and refuse all others.
- **A small, recorded target correction.** DemoMind learns
  `W_base + P`, where `P` is a constructed rank-1 correction
  of scale 0.03, from 32 fixed training inputs drawn from a
  dedicated RNG stream; its returned `hyperparams` record
  rank, alpha, steps, learning rate, target norm, and the
  **residual norm after training** — so a reader can see how
  well training actually did. It runs real gradient descent
  (2,000 steps, learning rate 1.0, MSE — those are DemoMind's
  recorded numbers for an 8×8 linear map; they are a
  precedent for *recording*, not values to copy onto a
  transformer).
- **Probes stay the judge.** DemoMind's training never
  touches its probe suites: identity/capability expectations
  are anchored to the seed, training inputs come from a
  separate stream, and the gate — not the trainer — decides
  whether the learned candidate lands.

What does not transfer: the demo candidate shape (keys
`delta.B`/`delta.A`, one implicit target), the linear-map
gradient (a transformer's loss surface runs through
attention, softmax, RMSNorm, and SwiGLU), and any of
DemoMind's hyperparameters. A miniature `train_delta` that
returned the demo shape, or quoted demo numbers as its own,
would be mislabelled on arrival.

### 4.3 The training signal: requirements before any gradient

The signal definition is the slice's first design act, and
it is constrained by two facts on record:

- **The echo trap.** The base's echo accuracy is already
  16/16 = 1.0 (SI-14). A signal that trains the miniature to
  echo trains it to do what it already does perfectly: the
  honest gradient is ~zero and the honest delta is ~zero.
  Any proposed signal must therefore be a behaviour the base
  is *measurably imperfect* at, with the base's score on the
  signal recorded before training begins — a signal the base
  already passes is invalid, and the slice must say so if
  its first candidate signal fails this test.
- **The gate tension.** Checks 3/4 measure *preservation*:
  drift from the base on probe inputs, and no capability
  regression. A signal that teaches a large new behaviour
  will — honestly — move probe logits and may be REJECTed.
  That is the gate working, not authorship failing.
  DemoMind's resolution is the discipline to copy: the
  target correction is **small by construction** (its `P`
  scale is 0.03 against an 8×8 base), so a well-trained
  candidate can improve on the signal while staying inside
  probe tolerances. SI-19's signal must state its target
  correction's scale and norm the same way, and must record
  the trained candidate's residual — and its probe outcome,
  whatever it is (§8).

Signal requirements (binding on SI-19):

1. **Deterministic construction** from a fixed seed, like
   the miniature itself; inputs are token-id sequences
   within the miniature's vocabulary and sequence cap.
2. **Disjointness, mechanically checked.** Training inputs
   must not overlap either probe suite's inputs, and the
   signal's RNG stream must be distinct from the identity
   stream (`MINI_SEED + 2`) and capability stream
   (`MINI_SEED + 3`) recorded in `si/tiny_seed.py`. Training
   on probe inputs would make checks 3/4 circular — the
   candidate would be judged on its homework — and any
   overlap found in review is a merge blocker.
3. **A held-out partition.** Part of the signal's examples
   are never trained on; improvement is claimed only on
   held-out examples (§4.7), never on training examples.
4. **A stated target form.** Whether the target is target
   tokens (a cross-entropy-style objective) or target logits
   (an MSE-style objective, DemoMind's form) is SI-19's
   recorded choice — either is defensible; an unstated one
   is not. The capability cross-entropy/metric distinction
   from RFC SI-13 §4.4 applies: a loss consumed by training
   is a training signal, never a gate result, and must be
   labelled as one wherever it is journaled (§10).

### 4.4 Gradient routes, evaluated

The forward path is numpy over float32 weights widened to
float64 for compute (SI-15). There is no autograd anywhere
in the dependency set, and invariant 6 forbids acquiring
one casually. Three routes:

- **(i) Analytic gradients, finite-difference checked
  (recommended).** Derive gradients of the stated loss with
  respect to the LoRA factors of the single target (§4.5)
  only — base weights are frozen, so no gradient with
  respect to any base tensor is ever needed — through the
  downstream computation from that target to the logits.
  The derivation is real work and error-prone in exactly
  the places this architecture is subtle (softmax Jacobian
  through causal attention, RMSNorm, SwiGLU). It therefore
  ships only with a **gradient-check test**: analytic
  gradients must agree with central finite differences of
  the same loss, computed through the existing forward, to
  a tolerance SI-19 pins *with the measured agreement
  published in its as-built note* (the tolerance value is
  an open question to measure, §11 — this RFC invents
  none). A gradient that fails its check is a failed
  slice, not a tunable detail.
- **(ii) Finite-difference optimisation only (fallback).**
  Derivative-free in derivation, at a derivable cost: a
  rank-`r` LoRA on a `[32, 32]` miniature target has
  `r × (32 + 32)` factor entries — 128 at rank 2, by
  arithmetic on SI-14's shapes — so one gradient estimate
  costs that many extra forwards (twice for central
  differences), each forward being the §4.5-estimated few
  million multiply-adds. Feasibility therefore hinges on
  step count, which is an open measurement (§11). This
  route is acceptable as a fallback if (i)'s check cannot
  be met, provided its step budget and wall time are
  measured and published; it is not acceptable as a silent
  substitute.
- **(iii) Add an autograd framework (rejected).** A heavy
  new dependency with its own tensor semantics, imported to
  differentiate a 23,456-element constructed model, breaks
  invariant 6 and RFC SI-13 §4.5's recorded ruling (no
  inference framework for zero measured benefit — the same
  logic covers training frameworks at this scale).
  Rejected for SI-19.

### 4.5 Candidate construction and the gate's unchanged role

- **One target, LoRA only.** Policy defaults already cap a
  transformer delta at 1 target (`max_targets_per_delta`),
  rank ≤ 8 (`rank_cap`), aggregate norm ≤ 1.0 (`norm_cap`),
  per-target norm ≤ 1.0, blob ≤ 4 MiB (`delta_byte_cap`) —
  SI-9b's as-built defaults in `si/bundle.py`. SI-19 trains
  **one named target** per candidate, chosen in the slice's
  design note with its reason recorded (target choice and
  rank are open questions to settle with measurements,
  §11). Additive targets are out of SI-19's scope (§9):
  authorship starts where the factorisation story is the
  one DemoMind proved.
- **Blind-target exclusion.** An authored candidate whose
  target is in the derived blind set (`k_proj.bias` of
  either layer — additive-only in any case) could never be
  judged by the probes; §4.5's LoRA-only, measurable-target
  scope keeps authored candidates out of the blind set by
  construction, and a test must assert the trained target
  is not blind.
- **Construction through the existing builder.** Factors
  are cast to float32 and passed to
  `construct_transformer_delta`, which binds them to the
  verified manifest and validates the pair. Training code
  never writes a blob, a manifest record, or a journal row
  directly.
- **The gate decides, unchanged.** The authored candidate
  flows through SI-17's live mapping: measured pass →
  ADMIT-eligible (loop materialises + probe VERIFYs, with
  rollback on failure); measured fail → REJECT; blind or
  unevaluated → ESCALATE. Authorship confers no admission
  privilege, and `si admit` interplay is unchanged (a
  probe-failed authored delta is as inadmissible as a
  probe-failed fixture).
- **`propose` scope.** `TinyTransformerMind.propose` should
  gain exactly one deterministic behaviour — emit the
  transformer intent for the named training signal,
  mirroring `DemoMind.propose`'s fixed-intent shape
  (declared targets, rank, alpha, parents, source
  sequences, `training_signal_id`) — so the real mind can
  drive `_run_miniature_cycle` end to end. A `propose` that
  invents intents beyond the named signal is out of scope.

### 4.6 CPU feasibility

The subject is the same 23,456-element miniature whose
forward RFC SI-13 §4.5 estimates at a few million
multiply-adds (RFC estimate, restated as such). Training
cost is that forward cost multiplied by steps and, for
route (i), by a backward pass of the same order; for route
(ii), by the §4.4 factor-count arithmetic. Whether the
resulting total is seconds or minutes on this VM is an
**open question SI-19 must measure and publish** (steps,
wall time, final loss, residual) — DemoMind's 2,000 steps
on an 8×8 map set no expectation for a transformer. No
route in §4.4 needs a GPU, a download, or a spend; the
training data is constructed, like the base.

### 4.7 What "authorship works" may claim

Exactly this, when §8's criteria are met: `train_delta`,
given the named signal's intent, deterministically produces
a structurally valid transformer candidate whose **held-out
signal loss measurably improves over the base's** (the
margin and the numbers are SI-19's to measure and publish),
and that candidate's **gate outcome — whatever the probes
measure — is journaled and reported as measured**. It may
not claim the miniature "learned" in any broader sense, that
the signal behaviour is useful (it is a constructed task,
like echo), or that authorship transfers to any other base.

### 4.8 Verdict on (b): RECOMMENDED

It closes the loop's last honest gap at the only scale the
loop can judge; it reuses the forward path, the delta
format, the gate, and the VERIFY machinery unchanged; its
risks (§11) are correctness risks that tests can catch,
not feasibility unknowns that measurements must first
retire. That is what a next slice should look like.

---

## 5. Candidate (c): probing Glimmer — REJECTED, restated

Glimmer probing was rejected by RFC SI-13 §7 (non-goal:
"No Glimmer forward path — not full, not partial, not 'just
the language tower'") and §9 (partial-forward probing and
external-engine adoption both considered and rejected).
SI-17 restated it as still not claimed. **This RFC does not
reopen it.** The reasons, restated so no future reader has
to excavate them:

- **There is no forward path, and a partial one is a
  fabrication.** Glimmer is a 1,436-tensor, bfloat16,
  multimodal checkpoint (59,553,253,376 logical bytes,
  `stored_bytes: 0`; SI-9 §0) with vision and language
  towers, hidden size 6656, and a 202,048-wide vocabulary.
  Drift measured on a truncated stack (embedding + one
  layer, say) is a claim about that truncated computation,
  not about the model's behaviour; reporting it as the
  model's identity continuity would be inventing a result.
- **The binding forbids it mechanically.** §6.2's rules
  (§3.3 above) cannot be satisfied for the Glimmer bundle:
  no probe artifact exists, none can be minted without a
  forward pass over that base, and the architecture check
  refuses its descriptor. This is the system working as
  designed, not a bug to route around.
- **Cost is measured, and it is disqualifying on CPU.**
  One full `load_bundle` verification of the by-reference
  base took ~29 minutes (1,756.9 s) of streaming reads on
  this VM (SI-9 §0/§7.1), before any forward compute over
  30B parameters in a runtime with no transformer inference
  engine. SI-9 §5.4's ruling stands: full inference is not
  "slow" — it is absent.
- **Nothing about SI-17 changes any of this.** The
  admission flip is miniature-only by its recorded decision;
  the Glimmer seed's deltas still escalate with checks 3/4
  "not evaluated", and under this RFC they continue to.

A future proposal to evaluate Glimmer would need its own
RFC, its own forward-path feasibility measurements, and its
own honesty review (RFC SI-13 §10, risk 7). It gets no
momentum from this document.

---

## 6. Decision summary

| Candidate | Verdict | One-line reason |
|---|---|---|
| (a) Small real safetensors base | **DEFERRED** | Needs a new executor, a reference-mode binding variant, a trained-base capability design, and feasibility measurements that do not exist (§3.7 preconditions). |
| (b) Delta authorship (miniature) | **RECOMMENDED — SI-19** | Plugs into the existing loop, format, and gate; judged by the existing probes; its risks are testable correctness risks (§4, §8). |
| (c) Glimmer probing | **REJECTED** | Already rejected by RFC SI-13 §7/§9; partial measurement would be fabrication; measured cost is disqualifying (§5). Not reopened. |

---

## 7. The recommended slice: SI-19 (outline)

**SI-19 — Delta authorship for the miniature: a trained,
gated candidate from `TinyTransformerMind.train_delta`.**

In scope:

1. A named, deterministically constructed **training
   signal** for the miniature, satisfying §4.3's
   requirements (base-imperfection recorded, disjointness
   mechanically checked, held-out partition, stated target
   form and target-correction scale).
2. `TinyTransformerMind.train_delta` implementing real
   gradient-based training of **one LoRA target** on CPU
   via §4.4 route (i) — or route (ii) as a documented
   fallback — returning a `construct_transformer_delta`
   candidate pair with DemoMind-grade `hyperparams`
   recording (signal id, rank, alpha, steps, learning
   rate, initial/final training loss, held-out loss before
   and after, residual/target norms, wall time).
3. `TinyTransformerMind.propose` emitting the named
   signal's deterministic transformer intent (§4.5), and
   nothing else.
4. Tests per §8, including the gradient check, determinism
   (repeat training → byte-identical candidate blob),
   unknown-signal refusal, and an end-to-end
   `_run_miniature_cycle` driven by the real
   `TinyTransformerMind` — not a script mind.
5. An as-built doc in the SI-14…SI-17 pattern, publishing
   every measured number §8 requires and stating plainly
   what is still not claimed.

Explicitly not in SI-19's scope: §9's non-goals.

---

## 8. Acceptance criteria for SI-19

A reviewer must be able to check each criterion against a
test or a published number. SI-19 does not merge on any
criterion being waived, tuned quietly, or reported without
its measurement.

1. **Signal validity.** The signal's construction is
   deterministic and documented; the base's score on the
   held-out partition is recorded *before* training and is
   measurably imperfect (not already at the signal's
   ceiling). A signal the base already passes fails this
   criterion by definition (the echo trap, §4.3).
2. **Disjointness.** A test asserts the training inputs
   (train and held-out partitions) share no sequence with
   either probe suite, and the signal's RNG stream is
   distinct from the recorded identity/capability streams.
3. **Gradient correctness.** Analytic gradients (route i)
   agree with central finite differences through the
   existing forward to the tolerance the slice pins; the
   measured worst agreement and the tolerance are published
   in the as-built doc. (Route ii fallback: this criterion
   is replaced by a published per-step forward count, step
   budget, and wall time.)
4. **Real training, real improvement.** `train_delta`
   returns a candidate for the named signal; its held-out
   signal loss is measurably lower than the base's held-out
   loss (both numbers published); training loss decreases
   over the run (initial/final published). A candidate whose
   held-out loss does not improve is a failed experiment:
   `train_delta` must refuse it (raise) rather than return
   a non-improving candidate as if it were authorship.
5. **Determinism.** Two `train_delta` runs on the same
   intent produce byte-identical candidate blobs and
   identical recorded hyperparameters (except any explicitly
   labelled wall-time field, which no measured value may
   depend on).
6. **Refusals intact.** Unknown `training_signal_id` →
   refusal (DemoMind precedent); additive-only or
   blind-set targets → refusal or construction failure,
   never a silently retargeted candidate; non-miniature
   bundles → the existing refusals, unchanged.
7. **Format and policy.** The returned candidate validates
   through `construct_transformer_delta` /
   `validate_transformer_delta` against the bundle's
   verified manifest, targets exactly one allowlistable
   tensor, and sits within the SI-9b policy caps (rank,
   norms, bytes) without any policy default changing.
8. **Gate honesty end to end.** An end-to-end miniature
   cycle driven by the real mind journals the authored
   candidate's gate report with checks 3/4 genuinely
   measured (or blind/not-evaluated, if that is the truth),
   and the outcome — ADMIT + probe VERIFY, REJECT, or
   ESCALATE — is reported in the as-built doc *as measured*,
   with the drift/accuracy numbers. No criterion requires
   ADMIT: authorship succeeds by §4.7's claim; admission is
   the gate's separate, recorded verdict. A test asserts
   the journaled verdict equals an independent ProbeContext
   evaluation of the same candidate.
9. **No regressions, no widenings.** The full existing
   suite is green with its expectations for non-miniature
   behaviour untouched; `si serve` still returns 501 for
   every transformer bundle; the demo path is byte-for-byte
   unchanged; stored-bytes accounting for a miniature bundle
   with an authored delta shows base bytes stored exactly
   as before (deltas are new bytes only).
10. **CPU budget published.** Steps, per-step cost basis,
    total wall time, and peak extra memory of one
    `train_delta` run on this VM are measured and published
    in the as-built doc. (No target numbers are set here;
    §11 records this as an open measurement. A run whose
    cost makes the test suite impractical fails the
    project's CPU-only discipline in spirit and must be
    re-scoped, not endured.)

---

## 9. Non-goals

Explicit, and binding on SI-19 and on any slice this RFC
is cited to justify:

- **No Glimmer anything.** No Glimmer forward path, probes,
  training, or authorship (§5). No candidate in this RFC
  touches the Glimmer seed except to restate that it is
  untouched.
- **No real-base probing yet.** Candidate (a) is deferred
  with preconditions (§3.7); SI-19 builds nothing toward it
  beyond keeping the miniature's machinery honest.
- **No serving.** Authorship does not wire `si serve` to
  anything; transformer bundles — miniature included —
  keep returning 501.
- **No additive-target authorship, no multi-target
  candidates, no authorship for any base other than the
  miniature** in SI-19 (§4.5).
- **No new dependencies** — in particular no autograd or
  training framework (§4.4 route iii). No GPU, no cloud,
  no downloads, no spend.
- **No policy default changes, no tolerance loosening, no
  probe-suite changes.** If an authored candidate fails
  the gate, the answer is a better signal or a better
  gradient — never a wider gate.
- **No transformer squash, no demo-path changes, no
  changes to checks 1, 2, or 5** for any delta kind.
- **No claim that the trained signal behaviour is useful.**
  It is a constructed task on a constructed model, present
  so that "a mind wrote a delta that measurably does what
  it was trained to do, and the gate judged it" has a
  falsifiable meaning at this scale.

---

## 10. Honesty and labelling requirements for any future slice

Binding on SI-19 and on any later slice citing this RFC:

1. **Training numbers are training numbers.** Losses,
   residuals, and held-out signal scores must be labelled
   with the signal id and partition wherever they appear
   (journal `hyperparams`, as-built docs, CLI output).
   They are never a gate result, never a probe result, and
   never evidence about any base but the miniature.
2. **Measured vs not evaluated survives authorship.** An
   authored candidate's checks 3/4 report exactly what the
   probes measured (`measured` a real float) or that they
   were not evaluated (`measured=None`) — the SI-17
   labelling table applies unchanged, including the
   blind-escalated state.
3. **Authorship is not admission.** No document, journal
   entry, or report may describe a trained candidate as
   "learned into the bundle", "accepted", or "verified"
   before the gate and (on ADMIT) the §6.6 probe VERIFY
   have actually run and been journaled. The DemoMind
   precedent — hyperparams record the *training*, the
   journal records the *verdict* — is the pattern.
4. **Every number carries its source.** A figure in an
   as-built doc is either reproduced from a test/run in
   that slice (with the test named) or cited to the
   earlier doc that measured it. Estimates are labelled
   estimates (as RFC SI-13 §4.5's cost figures are);
   derivations from cited shapes are labelled derivations
   (as §3.2's artifact-size arithmetic is).
5. **The miniature's labels travel.** "Constructed, not
   trained" describes the *base*; after SI-19, deltas may
   honestly be described as "trained on a constructed
   signal" — and nothing in that phrase may be allowed to
   blur into a claim that the base is a trained model or
   that the signal is a real task.
6. **Failure states are reportable successes of honesty.**
   A signal that fails §8.1, a gradient that fails §8.3, a
   candidate the gate REJECTs — each is a publishable,
   journalable outcome. A slice that reports only its
   passes has failed this section regardless of its code.

---

## 11. Risks and open questions

Each item names what would retire it. Numbers this RFC
deliberately does **not** supply are listed as open
questions to measure — supplying them here would be
inventing them.

1. **Gradient-derivation correctness (SI-19's central
   risk).** A subtly wrong analytic gradient can still
   decrease a loss — slowly, or toward the wrong factors —
   while looking like training. Mitigation is §8.3's
   finite-difference check, which is a merge blocker, not
   a nicety. *Open to measure:* the achievable
   analytic-vs-finite-difference agreement through the
   float64 forward, and the tolerance it justifies.
2. **Signal design may fail the echo-trap test or the
   gate-tension test.** A signal small enough to survive
   checks 3/4 may be too small to show held-out
   improvement; a signal large enough to improve may be
   REJECTed. DemoMind's precedent says a middle exists for
   a linear map (target scale 0.03); whether one exists
   for the miniature is **an open question to measure**,
   and §8.1/§8.4/§8.8 are written so that either answer is
   reported, not tuned away. *Open:* the signal's target
   form, target-correction scale, and target tensor/rank
   selection — SI-19 settles them with published
   measurements or reports that it could not.
3. **Training cost is unmeasured.** *Open to measure:*
   step count to a stable held-out improvement, learning
   rate, wall time per `train_delta` run, and the test
   suite's resulting runtime (§8.10). If route (i) fails
   and route (ii) is used, the per-step forward count from
   §4.4's arithmetic makes step count the whole feasibility
   question.
4. **Probe blindness bounds authorship's judge, still.**
   RFC SI-13 §10.1 stands: probes sample behaviour at 24
   inputs, per-class sensitivity is proven (28/30, blind
   set derived and guarded), but a passing authored delta
   is *evidence* of continuity, not proof. SI-19 adds a
   new way to produce deltas the probes cannot fully see;
   §10's labelling is the mitigation, and it is not a
   complete one. Recorded plainly.
5. **Overfitting to a constructed signal.** Held-out
   partitions (§4.3) bound the claim to the signal's own
   distribution. Improvement there says nothing about any
   other input distribution — there is no other distribution
   at this scale, which is precisely why §9 forbids
   usefulness claims.
6. **Candidate (a)'s unknowns remain unknown.** Forward
   wall time, assembled memory under the float64
   discipline, reference-mode verification cost, and a
   trained-base capability design for *any* real base are
   all unmeasured (§3.7's preconditions). Deferral is only
   honest if nobody later cites this RFC as having scoped
   them. It has not; it has listed them.
7. **Scope creep toward Glimmer, again.** RFC SI-13 §10.7
   named this pressure; SI-17's flip renews it ("the loop
   admits miniature deltas autonomously — why not…"). The
   answer is §5, and it does not soften because authorship
   lands: a trained miniature delta is evidence about the
   miniature, full stop.

---

## 12. Summary

After SI-17, SI can judge a transformer delta honestly at
exactly one scale, and cannot yet originate one at any
scale. Of the three candidate frontiers: probing Glimmer
stays rejected, for the recorded reasons, without
reopening (§5); generalising probes to a small real base
is the right long-term direction but is deferred until
its executor, binding, capability, and cost unknowns are
retired by measurements (§3); and **delta authorship for
the miniature is recommended as SI-19** — a named
constructed signal, real CPU gradient training of one LoRA
target with finite-difference-checked gradients,
construction through the existing delta format, and the
existing gate judging the result with the existing probes
(§§4, 7, 8). The base stays frozen, the bytes are never
duplicated, and nothing is claimed that a test in this repo
cannot check.

---

*RFC SI-18 — design only. Implementation begins, if
approved, at SI-19 as outlined in §7. RFC SI-13 taught the
gate how to measure; this RFC proposes teaching the mind
how to write — at the one scale where the gate can already
tell the truth about what was written.*
