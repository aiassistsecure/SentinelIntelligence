# SI-13 RFC — A Transformer Probe Harness: Honest Identity/Capability
Evaluation for Transformer Deltas (Design Only)

> **Status: DESIGN ONLY.** This document specifies a probe harness
> that does not exist. It changes no production code, no test, no
> manifest schema, no policy default, and no gate behaviour. Nothing
> in this RFC is implemented unless a section explicitly says it
> exists today, in code, on `main`.
>
> **The one sentence that governs everything below:** until the
> slices in §8 are implemented *and validated*, gate checks 3
> (identity continuity) and 4 (capability floor) for transformer
> deltas continue to report **"not evaluated"**, and a valid
> transformer delta continues to **ESCALATE** to the operator
> (SI-9b behaviour, SI-10 operator admission). This RFC must never
> be cited as evidence that checks 3/4 pass, ran, or can run today.
>
> **Scope discipline:** CPU only. No GPU, no cloud, no spend, no
> training, no serving change. Terminology: this is machine
> learning infrastructure, not "custom LLM software."
>
> **Relationship to prior work:** SI-9 RFC §4.3 named this harness
> as future work and deliberately did not design it ("Designing it
> here would be speculation stacked on speculation"). SI-10 built
> operator admission as the honest stopgap. SI-11 reconciled
> `si export` with transformer deltas and SI-12 made `si squash`
> fail closed on them; neither touches probes. This RFC is the
> design §4.3 deferred. It supersedes nothing in SI-9/10/11/12.

---

## Table of contents

1. [Problem restatement](#1-problem-restatement)
2. [Inherited invariants](#2-inherited-invariants)
3. [The miniature transformer mind](#3-the-miniature-transformer-mind)
4. [Forward path design](#4-forward-path-design)
5. [Probe suites](#5-probe-suites)
6. [Gate integration](#6-gate-integration)
7. [Non-goals](#7-non-goals)
8. [Implementation plan](#8-implementation-plan)
9. [Alternatives considered and rejected](#9-alternatives-considered-and-rejected)
10. [Risks and open questions](#10-risks-and-open-questions)
11. [Summary](#11-summary)

---

## 1. Problem restatement

### 1.1 What checks 3 and 4 do when they can run

The gate (`si/gate.py`) is a five-check ternary gate
(`ADMIT | REJECT | ESCALATE`). For demo deltas
(`kind == "delta"`), checks 3 and 4 are real measurements:

- **Check 3 — identity continuity** (`Gate._identity_check`): for
  each `(qid, x, expected)` in `ctx.identity_probes()`, run
  `ctx.forward_candidate(x, candidate)` and take the max-absolute
  error against `expected` (`_err`, float64). Pass iff the worst
  drift is `<= policy["identity_tol"]`.
- **Check 4 — capability floor** (`Gate._capability_check`): for
  each `(cid, x, expected)` in `ctx.capability_suite()`, compute
  the same error, subtract the baseline error recorded at the
  current head (`ctx.capability_baseline()`), and pass iff the
  worst *regression* is `<= policy["capability_eps"]`.

Both checks therefore depend on three things existing: a **forward
pass** that can run base + active deltas + candidate, a **probe
suite** of fixed inputs with expected outputs, and a **baseline**
measured before the candidate. `DemoMind` (`si/minds.py`) supplies
all three for the demo world: its forward is one linear map
`y = (W_base + Σ sᵢ·Bᵢ@Aᵢ) @ x` over an 8×8 float32 base, its
identity probes are anchored to the seed (`W_base @ x`), and its
capability suite is a held-out set of the same form. `LoopContext`
(`si/loop.py`) wires the mind to the gate's `BundleContext`
protocol (`identity_probes()`, `capability_suite()`,
`capability_baseline()`, `forward_candidate(x, candidate)`,
`stack_depth()`).

### 1.2 Why none of that exists for a transformer bundle

For a transformer delta (`kind == "transformer-delta-v1"`,
SI-9a format in `si/transformer_delta.py`), `Gate.evaluate` takes
a different branch, and it is honest about why:

- **No forward pass.** This runtime has no transformer inference
  engine of any kind. There is no attention, no embedding lookup,
  no sampling anywhere in `si/`. `si serve` returns an explicit
  `501` (`TRANSFORMER_NOT_SUPPORTED`) for transformer bundles
  rather than fabricate an output. Apply-at-load
  (`si.apply.load_applied_tensor`) can produce *one named tensor*
  with its deltas folded in — it is a library capability, not
  inference, and it cannot turn tensors into behaviour.
- **No probe suite.** No identity or capability suite is defined
  against any transformer base. There are no fixed inputs, no
  expected logits, nothing to measure drift against. The demo
  suites measure an 8-vector linear map; they say nothing about a
  transformer (SI-9 §0/§1.2 said this plainly).
- **No baseline.** A baseline is a measurement of the base (or the
  current head) taken through the forward pass that does not
  exist, over the suite that does not exist. The Glimmer seed
  (SI-8, by reference, 1,436 tensors, `stored_bytes: 0`) records
  provenance and content hashes — identity of *bytes*, not of
  *behaviour*.

The gate's transformer branch therefore appends both probe checks
as `CheckResult(name, passed=False, measured=None, detail=
TRANSFORMER_PROBES_NOT_EVALUATED)` — the constant in `si/gate.py`
reads, in part: *"not evaluated: no transformer probe harness
exists — this runtime has no transformer forward pass, no probe
suite, and no baseline for a transformer base (RFC SI-9 §4.3);
reporting a pass would be fabrication, so the delta is escalated
to the operator instead."* The two unevaluated checks are excluded
from the decisive set: they alone never cause a `REJECT`, and they
block any `ADMIT`. A transformer delta that passes constitution,
the §4.2 structural safety suite, and provenance is **ESCALATED**.

### 1.3 The stopgap, and its known cost

SI-10 (`si/admit.py`, `docs/SI-10-operator-admission.md`) built the
mechanism the escalation was waiting for: an operator-only CLI
path that re-runs the gate, admits only an `ESCALATE` verdict,
journals the decision before materialising, and runs the
`transformer-apply-smoke` VERIFY step
(`si.loop._verify_transformer_apply` — shape + finiteness via
apply-at-load, explicitly *not* a probe result). Its journaled
basis string states that *"the operator acted as the
identity/capability check"* and that probes *"were NOT evaluated."*
The admission context (`_AdmissionContext`) enforces this in code:
its `identity_probes()`, `capability_suite()`,
`capability_baseline()`, and `forward_candidate()` raise if the
gate ever touches them.

That is correct, and it does not scale as a *judgement*: SI-9 §7.3
already flagged operator escalation fatigue — a human asked to
stand in for two unmeasurable checks on every delta becomes a
rubber stamp, and loosening thresholds is not the fix. The fix is
measurement. This RFC designs the smallest measurement apparatus
that can be honest.

### 1.4 What "solving" it must mean (and not mean)

A solution is: a transformer small enough that this CPU runtime
can execute a real forward pass over it; probe suites defined
against *that* transformer, with baselines recorded when it is
seeded; and gate wiring that consumes real probe results for
bundles of *that* kind — while every other transformer bundle,
above all the Glimmer 30B seed, keeps today's behaviour exactly:
checks 3/4 "not evaluated", valid deltas ESCALATE.

A solution is **not**: a forward pass for Glimmer, a claim that
miniature probe results say anything about the 30B seed, or a
synthetic pass derived from norms, shapes, or the apply smoke
test. §4.3 of the SI-9 RFC already ruled that numeric surrogates
are not checks 3 & 4; this RFC inherits that ruling without
exception.

---

## 2. Inherited invariants

These are not negotiable inside this RFC; a proposal that breaks
one is a different project (SI-9 §1.3, restated for the harness):

1. **Base is frozen forever.** Probes read the base; they never
   modify it. All change remains a stack of deltas; rollback is
   moving `head`.
2. **Never duplicate bytes.** The harness stores no copy of any
   base. Probe artifacts (inputs, expected outputs, baselines) are
   small derived data defined in §5.5 — they are not tensor bytes
   of the base, and their size budget is stated there.
3. **No ungated write.** The harness adds measurement to the gate;
   it adds no path around the gate. The autonomous loop gains no
   admission power from this design until the final, separately
   gated slice in §8 (SI-17), and then only for miniature bundles.
4. **Honest labels are load-bearing.** Every report the harness
   produces must be distinguishable in the journal from today's
   "not evaluated" reports, from the apply smoke report, and from
   demo probe reports. A reader must never be able to mistake one
   for another (§6.5).
5. **Fail closed.** A missing, mismatched, or un-runnable probe
   suite is "not evaluated" — never a pass, never a silent skip
   (§6.3).
6. **CPU only, existing dependencies only.** The runtime's
   dependencies are `numpy` and `safetensors` (`requirements.txt`,
   `pyproject.toml`). The harness must be buildable from those two
   plus the standard library. No GPU, no network, no new heavy
   dependency (§4.5).
7. **Determinism where tests need it.** Seeded RNG, fixed inputs,
   no sampling, no wall-clock dependence in any measured value —
   the same discipline `DemoMind` and the SI-9c numeric tests
   already follow.

---

## 3. The miniature transformer mind

The harness needs a transformer the runtime can *actually
execute*. Rather than shrink Glimmer — impossible honestly; there
is no Glimmer forward path at any size in this runtime — this RFC
proposes a new, purpose-built miniature: a small decoder-only
transformer, seeded as a test/demo bundle, against which deltas,
probes, and baselines are all defined. Working name:
**`TinyTransformerMind`** (final naming is SI-14's to settle; this
RFC uses the name descriptively).

### 3.1 Architecture (proposal, pinned for SI-14)

A pre-norm, decoder-only transformer in the Llama/Glimmer family
shape — RMSNorm, multi-head causal self-attention, SwiGLU-style
MLP — reduced to dimensions this runtime can run in milliseconds.
Family resemblance is deliberate: tensor names and kinds mirror
the Glimmer-style names the SI-9 tests already use
(`model.language_model.layers.{i}.…`), so the existing exact-name
machinery (SI-9a delta binding, SI-9b allowlists, SI-9c apply,
SI-11 export) applies to the miniature *unchanged* — the harness
invents no second targeting system.

| Hyperparameter | Value | Note |
|---|---|---|
| Vocabulary `V` | 32 tokens | Inputs are token ids; there is no tokenizer (§3.4) |
| Model width `d_model` | 32 | Residual stream width everywhere |
| Layers `L` | 2 | Indexed `layers.0`, `layers.1` |
| Attention heads `H` | 4 | Head dim `d_head = 8` |
| MLP hidden `d_ff` | 64 | `gate_proj` / `up_proj` / `down_proj` |
| Max sequence `T_max` | 16 tokens | All probe inputs are `<= 16` tokens |
| Norm | RMSNorm | Weight only (1-D), no bias — Glimmer-family style |
| Attention biases | Present on `q/k/v/o_proj` | Gives `additive` targets a real home (§3.2) |
| MLP biases | None | Llama-family style |
| Embeddings | Untied | Separate `embed_tokens` and `lm_head` tensors |

Parameter budget, by construction:

| Tensors | Count | Elements |
|---|---|---|
| `embed_tokens.weight` `[32, 32]` + `lm_head.weight` `[32, 32]` | 2 | 2,048 |
| Per layer: `q/k/v/o_proj.weight` `[32, 32]` ×4 | 8 | 8,192 |
| Per layer: `q/k/v/o_proj.bias` `[32]` ×4 | 8 | 256 |
| Per layer: `gate/up_proj.weight` `[64, 32]` ×2, `down_proj.weight` `[32, 64]` | 6 | 12,288 |
| Per layer: `input_layernorm` + `post_attention_layernorm` `[32]` ×2 | 4 | 128 |
| Final `norm.weight` `[32]` | 1 | 32 |
| **Total (2 layers)** | **29 tensors** | **22,944 elements** |

At float32 that is ~90 KiB of base — three orders of magnitude
below the SI-9b `delta_byte_cap` scale of discussion, and small
enough that a full forward pass is tens of thousands of
multiply-adds per token position (see §4.5). The miniature is a
*seeded bundle* in the ordinary sense: its tensors live in a
bundle's content-addressed store via the existing per-tensor seed
machinery (SI-7 copy mode — at ~90 KiB, by-reference mode and its
external dependency would be pointless ceremony), its manifest
carries a per-tensor map (`base.tensor_blobs`), and transformer
deltas bind to it by exact name + content hash exactly as SI-9a
specifies. No seed-format change is proposed.

### 3.2 Targetable tensors

Every tensor class a transformer delta can name today is
represented, so probes can be checked for sensitivity to each:

- **`lora` targets (2-D):** all projection weights — attention
  `q/k/v/o_proj.weight`, MLP `gate/up/down_proj.weight`,
  `embed_tokens.weight`, `lm_head.weight`.
- **`additive` targets (1-D):** attention projection biases and
  RMSNorm weights — still subject to policy `allow_additive`
  (default `false`, SI-9b); the harness changes no policy default.
  A bundle that wants additive probes evaluated opts in per
  bundle, as a journaled policy act, exactly as today.

`embed_tokens` / `lm_head` are *in* scope here — the opposite of
the SI-9 choice for Glimmer (§5.4 of the SI-9 RFC excluded
`lm_head`-scale tensors on arithmetic cost). At `[32, 32]` the
cost argument does not exist, and excluding the embedding would
blind the probes to an entire tensor class for no reason. This
difference is a consequence of scale, not a disagreement.

### 3.3 Seed construction: structured, not trained

The miniature base is **constructed**, deterministically, from a
fixed seed — the same honesty pattern as the demo seed
(`si init`'s base is labelled `"demo:random-seed (not a real
model)"` in its own manifest), with one deliberate upgrade: the
construction gives the mind one real, checkable behaviour
(§5.3's echo task) so the capability floor has something to be a
floor *under*.

Construction sketch (SI-14 pins the exact numbers and proves them
with seed-time assertions; this RFC fixes the *shape* of the
construction and its honesty requirements):

1. **Residual-dominant embedding pair.** `embed_tokens.weight` is
   a scaled permutation-like matrix (token `t` maps predominantly
   to residual direction `t`) and `lm_head.weight` its approximate
   inverse readout, so that — before attention/MLP contributions —
   the model's next-token logits at a position are dominated by
   the *last consumed token*. Positional signal comes from a
   small learned-free positional embedding table
   `[T_max, d_model]`, initialised small. (If SI-14 includes the
   positional table as a base tensor, the tensor count above grows
   by one and 512 elements; the budget conclusion is unchanged.
   SI-14 must state which it does — this RFC does not smuggle the
   decision.)
2. **Small, live attention/MLP branches.** All projection weights
   are drawn from a seeded RNG at small scale (the `DemoMind`
   adapter precedent uses 0.02-scale draws). Small — so the echo
   behaviour dominates — but **non-zero**, so every branch is live:
   a delta on any projection produces a *measurable* change in the
   logits. A construction that zeroed a branch's output projection
   would make deltas on that branch's other tensors invisible to
   the probes; §5.6 makes per-class sensitivity an explicit,
   tested requirement, and zeroed branches would fail it.
3. **Seed-time assertions, fail closed.** Before any baseline is
   recorded, seed construction must verify, by running the §4
   forward path on the §5 suites: determinism (two runs, identical
   outputs), finiteness, and the echo floor (§5.3). If the
   constructed seed does not meet the floor it claims, seeding
   **fails** — SI-14 must never record baselines for a seed whose
   capability claim was not measured on that seed. This is the
   harness's first honesty test, applied to itself.

The manifest honesty label follows the demo precedent: the base's
`source` must say what it is — a constructed miniature for probe
development, **not a trained model** — and §3.4's non-claims travel
with it into every report.

### 3.4 What the miniature is NOT

Stated once, plainly, because every later section depends on it:

- It is **not Glimmer**, at any scale claim. It shares a family
  shape and tensor naming so SI's tensor machinery applies; it
  shares no weights, no tokenizer, no architecture config, and no
  behaviour with the 30B seed.
- Probe results on the miniature say something about **deltas
  against the miniature** — nothing about deltas against Glimmer.
  §6.2's binding rules exist to make that distinction mechanical,
  not editorial.
- It is **not a trained model** and does no useful work. Its one
  behaviour (§5.3) is a constructed toy, present so that
  "capability" has a falsifiable meaning at this scale.
- It is **not a serving target.** Building a forward path for
  probes does not wire `si serve` to anything (§7); transformer
  bundles — miniature ones included — keep returning `501` until
  a separate, explicit decision changes that.
- It is **not a training target in this design.** Nothing in
  §§3–6 trains the miniature or authors deltas for it. Delta
  authorship remains out of scope, exactly as in SI-9 §2.5.

---

## 4. Forward path design

### 4.1 Location and shape

Proposal: a new module, **`si/tiny_transformer.py`** — a pure
numpy forward implementation for the §3 architecture, plus a mind
adapter exposing the existing `Mind`-protocol surface the gate and
loop already consume. (Module naming follows the SI-9 precedent of
recording the location decision in the slice that makes it:
SI-9a chose `si/transformer_delta.py`, SI-9c chose `si/apply.py`
over growing `si/bundle.py`. A new module keeps the forward path
importable by tests and by a future gate context without the gate
or bundle layers depending on its internals. SI-15 confirms or
amends this with its own recorded rationale.)

The forward function's contract:

```
forward(token_ids, weights) -> logits
  token_ids : sequence of ints, 1 <= len <= T_max, each in [0, V)
  weights   : {tensor_name: float32 ndarray} — the full applied
              tensor set for the bundle at some head (§4.2)
  logits    : float32 ndarray [len(token_ids), V] — one row per
              position; probe comparisons use the last position's
              row unless a suite entry says otherwise (§5.2)
```

Computation, per position, in order: token + positional
embedding → for each layer, pre-norm RMSNorm → multi-head causal
self-attention (scaled dot-product, causal mask, softmax) →
residual add → pre-norm RMSNorm → SwiGLU MLP → residual add →
final RMSNorm → `lm_head`. No dropout (inference-only path), no
sampling, no KV cache — sequences are at most 16 tokens and the
whole sequence is recomputed per call; clarity beats speed at
this scale, and §4.5 shows speed is not a constraint.

### 4.2 Consuming base + deltas: apply-at-load, plus a candidate overlay

The forward path never reads base bytes directly. For every
tensor name in the miniature's manifest tensor map, the applied
value comes from the existing apply-at-load entry point:

```
si.apply.load_applied_tensor(bundle_dir, manifest, name, policy)
```

which starts from the hash-verified per-tensor read
(`load_base_tensor`), folds the active chain's transformer deltas
for that exact name oldest-first (`W + (alpha/rank)·B@A` for
`lora`; `w + d` for `additive` only when `allow_additive` is
literally true), re-validates each targeting delta with SI-9a
validation before trusting its factors, materialises nothing, and
returns float32. Twenty-nine calls produce the full applied
weight set; at miniature scale the per-call validation cost is
noise (§4.5).

**The candidate problem — the one new mechanism this RFC must
specify.** The gate evaluates a candidate *before* it lands: the
candidate is a `(record, blob_bytes)` pair (the SI-9b candidate
contract), not a manifest entry, so `load_applied_tensor` —
which folds the *active chain* — cannot see it. The demo world
solves the equivalent problem in `LoopContext.forward_candidate`
by appending the candidate's `{B, A, scale}` to the in-memory
active stack. The harness equivalent, proposed here:

- A helper (working name `applied_tensor_with_candidate`) that,
  for a tensor name the candidate targets, starts from
  `load_applied_tensor(...)` (base + active chain) and then folds
  the candidate's factor(s) for that exact name in, using **the
  same arithmetic and the same SI-9a validation**
  (`validate_transformer_delta` against the caller's verified
  manifest) that `si/apply.py` already performs — validation
  first, factors trusted only after it passes.
- For names the candidate does not target, the helper returns the
  `load_applied_tensor` result untouched (one manifest lookup, no
  candidate read — the §5.1 apply principle).
- The arithmetic must be **shared code, not a copy**. SI-11 already
  had to mirror apply arithmetic in export's materialize path and
  documented the duplication as a cost; the harness must not add a
  third hand-copied fold. SI-15's acceptance criteria (§8) require
  the candidate fold to reuse `si/apply.py`'s per-target fold —
  by extraction/refactor if necessary — with tests proving the
  overlay and the post-materialisation apply agree exactly.

Rejected alternative: materialising the candidate into a
temporary copy of the bundle and probing that. It duplicates
bytes (against invariant 2 in spirit — a whole applied weight set
for a ~90 KiB base is small, but the *pattern* is exactly what the
project forbids, and it would not generalise honestly), it writes
unjournaled bundle state, and it makes probe results depend on a
copy's integrity story instead of the verified manifest. The
in-memory overlay keeps probing read-only, like apply-at-load
itself.

### 4.3 Determinism

- **Inputs are token-id sequences** — no tokenizer, no text
  encoding ambiguity, no locale. A probe input is a list of ints,
  stored verbatim (§5.5).
- **No randomness at forward time.** No sampling, no dropout, no
  RNG. The same weight set and input produce the same logits.
- **Arithmetic discipline mirrors the gate's existing choices:**
  factors and weights are float32 (the apply path's stated
  dtype); error measurement against expected outputs is float64
  max-absolute error — the existing `_err` in `si/gate.py`, reused,
  not redefined. Softmax and norm computations must be written to
  be run-to-run identical on a fixed numpy build (no parallel
  reductions with non-deterministic ordering; at this size numpy's
  defaults are deterministic, and SI-15's tests must prove
  bit-identical repeat runs on this runtime rather than assume
  them).
- **Baselines are per-bundle, recorded at seed time** (§5.4), so
  cross-machine float variation — if it ever matters — shows up
  as a seed whose recorded baselines its own forward path cannot
  reproduce, which §6.3's rules treat as not-evaluated, not as a
  candidate's failure. (Re-measuring the base against its recorded
  baseline is part of harness startup validation in SI-16.)

### 4.4 No training, no authorship

The forward path is inference-only in the narrowest sense: it
evaluates. It contains no gradient code, no optimiser, no loss
for training (the capability cross-entropy in §5.3 is a *metric*
computed on outputs, not a training signal consumed here), and no
way to propose or construct a delta. `train_delta` on the
miniature mind adapter, if the adapter exposes the `Mind`
protocol's full surface, must refuse explicitly — the way
`_AdmissionContext`'s probe methods raise — rather than return a
fabricated candidate. How a delta against the miniature is
*authored* (hand-built fixtures in tests, as SI-9c's real-checkpoint
proof did, are the expected source) is unchanged by this RFC.

### 4.5 Feasibility: CPU cost with existing dependencies

Per forward call: 16 positions × 2 layers × (attention
`16×16×32`-scale score/projection work + MLP `32×64`-scale
projections) — on the order of a few million multiply-adds worst
case, i.e. single-digit milliseconds in numpy, dominated by call
overhead. A full probe evaluation (§5: 8 identity + 16 capability
inputs, each needing one candidate forward, plus baseline forwards
already recorded) is tens of forwards — comfortably under a
second. Against the SI-9 §7.1 backdrop (a full `load_bundle`
verification of the Glimmer base measured ~29 minutes), the point
is worth stating: **the harness is cheap precisely because its
subject is 22,944 parameters.** Nothing in this cost story
transfers to a real transformer base, and §6.2 ensures the gate
never pretends it does.

Dependencies: numpy (already used for every numeric path in the
repo) and safetensors (already the blob format). Softmax,
RMSNorm, and attention at this scale are a few dozen lines of
elementary numpy. No inference framework is needed, and adding
one (a transformer library, a GGUF runner, anything with its own
weight format) would import a second tensor semantics for zero
measured benefit — explicitly out (§7).

---

## 5. Probe suites

A probe suite is a set of fixed `(id, input, expected)` triples
plus recorded baselines, in direct analogy to `DemoMind`'s suites
— `(qid, x, expected)` and `(cid, x, expected)` — with `x` a
token-id sequence and `expected` a logits vector instead of an
8-vector. Both suites are defined against the miniature only.

### 5.1 Common rules

- **Fixed and frozen at seed time.** Inputs are generated by a
  stated deterministic procedure (below), expected outputs are
  computed by the §4 forward path over the **base** (head =
  `base`, no deltas), and both are recorded as probe artifacts
  (§5.5) before any delta exists. A suite never changes under a
  live bundle; changing a suite is seeding a new bundle.
- **Disjoint inputs.** Identity and capability inputs are drawn
  from separate RNG streams (the `DemoMind` precedent: separate
  seeded draws for identity vs capability probes), and neither
  overlaps the other's ids or sequences.
- **Falsifiable by construction.** Every suite entry has an id, a
  concrete input, and a numeric expected value. "The model seems
  fine" is not a probe. A check outcome names the worst entry id,
  exactly as `_identity_check` / `_capability_check` already do
  (`worst: {qid}` / `worst: {cid}` in their detail strings).
- **Small enough to audit by eye.** Every input is printable (a
  list of at most 16 small ints) and every expected output is a
  32-float vector whose argmax a reviewer can check.

### 5.2 Identity probes

**Definition.** Eight sequences (`tiny-id-0` … `tiny-id-7`)
covering the input space's corners, generated deterministically
at seed-definition time (SI-14 publishes the exact list in code
and tests; the classes are pinned here):

1. A single token (minimum length).
2. A repeated single token at maximum length (16).
3. A maximum-length sequence of distinct tokens.
4. An ascending run (`0, 1, 2, …`).
5–8. Four pseudo-random sequences from
   `numpy.random.default_rng(MINI_SEED + 2)` — mirroring
   `DemoMind`'s identity stream at `seed + 2` — of lengths
   4, 8, 12, 16.

**Expected output.** For each entry, the base forward's **last-
position logits** (`[32]` float32), recorded at seed time.

**Measurement.** Identical to the existing check 3: for the
candidate, `err = max-abs(candidate_logits − expected)` in
float64; the check's `measured` is the worst entry's error and
its detail names that entry. Nothing about the *metric* is new —
only the forward path and suite are.

**Pass condition.** `worst <= identity_tol`, where `identity_tol`
is the bundle policy's existing key (the gate already reads it;
`DEFAULT_POLICY` carries `0.25` for the demo). §5.4 covers how
the miniature bundle's value is set — it is a per-bundle policy
value set at seed time with recorded justification, not a new
global default and not the demo's number inherited by silence.

### 5.3 Capability probes: the echo task

**The toy task.** *Echo:* given a sequence, the model's
next-token prediction (argmax of the last-position logits) must
equal the sequence's **last token**. This is a real, if tiny,
sequence behaviour — the model must carry the last token's
identity through both layers to the readout — and §3.3's
construction makes the base genuinely able to do it, verified at
seed time. It is also honestly trivial: it is the transformer
analogue of `DemoMind`'s capability suite (whose "capability" is
reproducing `W_base @ x` on held-out inputs), not a claim of
usefulness. Every document and report that names this task must
call it what it is: a constructed toy floor.

**Definition.** Sixteen sequences (`tiny-cap-0` … `tiny-cap-15`)
from the capability RNG stream (`default_rng(MINI_SEED + 3)`,
mirroring `DemoMind`'s use of a stream distinct from identity's),
lengths spread over 2…16, last tokens covering at least half the
vocabulary across the suite. For each entry, two expectations are
recorded at seed time:

1. **Regression expectation** — the base forward's last-position
   logits, used exactly as check 4 uses the demo suite today:
   `regression = err_candidate − err_baseline`, where
   `err_baseline` is the baseline error at the current head
   recorded before the candidate (§5.4). At head = `base`,
   baselines are 0.0 by construction (expected *is* the base
   output); on a non-base head they are the pre-candidate head's
   measured errors, exactly as `LoopContext` computes
   `capability_baseline()` today.
2. **Functional expectation** — the echo target token itself.
   The candidate's **echo accuracy** is the fraction of the 16
   entries whose argmax equals the target. The **floor** is an
   absolute number recorded at seed time: the base's own measured
   accuracy (asserted at seed time to be `>= 15/16`; §3.3's
   fail-closed rule) — proposed floor for the candidate:
   **accuracy must not drop below the base's recorded accuracy
   minus one entry** (i.e. `>= base_accuracy − 1/16`). One entry of
   slack exists because a single argmax flip on a near-tie is a
   weaker signal than a trend; more than one flip on a 16-entry
   suite is a behaviour change, and the regression arm (below)
   independently bounds how far any single entry's logits moved.

**Pass condition (both arms required).** Check 4 passes for a
miniature candidate iff (a) worst regression `<= capability_eps`
(the existing key and semantics), **and** (b) echo accuracy meets
the floor above. Arm (a) alone would be the demo check transplanted;
arm (b) is what makes the check about a *behaviour* rather than
only about distance from the base's outputs. SI-16 must implement
both or neither — a regression-only wiring that reports itself as
"capability evaluated" would overclaim, and §6.5's report labels
are written to prevent exactly that.

### 5.4 Baselines and tolerances: recorded, justified, per-bundle

- **Baselines recorded at seed time:** for every suite entry, the
  base error terms (0.0 by construction where expected is the base
  output — recorded explicitly, not assumed), the base echo
  accuracy, and the **top-1/top-2 logit margin** per identity
  entry (the gap between the largest and second-largest expected
  logits). Recording margins is what makes the tolerance below a
  derivation instead of a guess.
- **Tolerance justification.** A max-abs logit drift smaller than
  half an entry's margin cannot change that entry's argmax. The
  miniature bundle's `identity_tol` is therefore proposed as
  `min(0.25, 0.4 × min_margin)` over the identity suite, computed
  and recorded at seed time — bounded above by the demo-era
  default, and tied to the measured scale of this seed's logits.
  `capability_eps` follows the same discipline against the
  capability suite's margins. If the computed value is absurdly
  small (a seed whose margins are near zero — a construction
  failure), seeding fails closed per §3.3 rather than recording a
  tolerance no honest delta could meet. SI-14/SI-16 must publish
  the measured margins and the resulting numbers in their as-built
  notes; **the figures are evidence in those slices, and this RFC
  invents none.**
- **Zero-delta calibration.** A candidate that is exactly zero
  (norm 0) must measure drift 0.0 and regression 0.0 and pass both
  probe checks. This is a required acceptance test in SI-16 (§8):
  it is the harness's proof that it measures the candidate, not
  its own noise.

### 5.5 Probe artifacts: what is stored, where

Proposal (SI-14 pins the mechanics): a bundle that carries probes
stores them as a **probe artifact file in the bundle**
(working name `probe_suite.json`), content-hashed into the
manifest the way `policy.json` and the constitution already are,
and journaled at seed time with its provenance (construction
seed, suite generation procedure, measured margins, resulting
tolerances, base tensor-map hash — §6.2). Contents: suite ids,
inputs, expected logits (base64 or JSON float arrays — at
24 entries × 32 floats the whole artifact is tens of KiB),
baselines, base echo accuracy, and the §6.2 binding fields.

This is derived measurement data, not base tensor bytes; storing
it does not breach the never-duplicate-bytes rule (invariant 2).
It is also **not** a seed-format change: the base section of the
manifest is untouched; the artifact is an addition alongside
policy/constitution, and bundles without it are simply bundles
without probes (§6.3).

### 5.6 Sensitivity requirement (the suite must be able to fail)

A probe suite that cannot fail is a synthetic pass with extra
steps. SI-16's acceptance criteria (§8) therefore include
**reference deltas** with known required outcomes, authored as
test fixtures against the miniature:

- A **zero delta** → both probe checks pass (§5.4).
- A **small benign delta** (LoRA at a small fraction of the
  per-target norm cap, on an allowlisted projection) → probe
  checks pass, with measured drift `> 0` (proving the measurement
  is live, not stuck at zero).
- A **large destructive delta** (norm at/near cap, directed at
  `lm_head` or the embedding) → identity check **fails** on
  measured drift.
- A **targeted capability attack** (a delta constructed to break
  echo on specific last tokens) → capability check **fails** on
  the accuracy arm even if its aggregate norm is modest.
- ** Per-class coverage:** for each targetable tensor *class* in
  §3.2, at least one reference delta on that class produces
  measurable drift — the empirical proof that no branch is blind
  (the §3.3 non-zero-branch requirement, verified rather than
  asserted).

If a reference delta does not produce its required outcome, the
harness is wrong — the suite, the tolerances, or the forward
path — and SI-16 does not merge until it does. This is the
section that keeps the whole design honest.

---

## 6. Gate integration

### 6.1 The shape of the change (future — SI-16/SI-17, not this RFC)

Today, `Gate.evaluate`'s transformer branch appends the two
not-evaluated checks unconditionally. The design keeps that branch
and adds **one conditional**: when — and only when — the bundle
under evaluation is a miniature probe bundle (§6.2) *and* a valid
probe artifact is present *and* harness startup validation (§6.3)
has reproduced the base baselines, the gate runs the real checks:

- Check 3 runs the §5.2 suite through a probe context's
  `forward_candidate(x, candidate)` — where `forward_candidate`
  is §4.2's candidate-overlay forward — and applies the existing
  `_identity_check` logic (worst max-abs drift vs `identity_tol`).
- Check 4 runs the §5.3 suite the same way, applying
  `_capability_check`'s regression logic **plus** the §5.3
  accuracy arm; both arms land in the check's detail/subchecks so
  the journal shows which arm decided.

The recommended implementation is a `BundleContext` implementation
for the miniature (working name: a probe context in the loop/admit
layer, sibling to `LoopContext` and `_AdmissionContext`) — the
gate's demo check implementations are already generic over that
protocol, and SI-16 should reuse them rather than fork the metric
code. The transformer branch's other checks (constitution, the
§4.2 structural safety suite, the §4.4 provenance extension) are
**unchanged in all cases**: probes add evidence; they replace no
existing check.

### 6.2 Binding: probe results count only for the bundle they were recorded on

This is the section that prevents miniature evidence from ever
being spent on a real transformer. Probe evaluation is permitted
only if **all** of the following hold, verified from the bundle
context (never from the candidate — the SI-9b rule that a
candidate may not validate against a base of its own choosing
applies verbatim):

1. The bundle's manifest carries a probe artifact (§5.5) whose
   content hash matches the manifest's recorded hash.
2. The artifact's recorded **base binding** — the base tensor
   map's names/shapes/dtypes and the per-tensor content hashes,
   hashed as a unit — matches the bundle's verified manifest
   exactly. (For the Glimmer seed this fails by construction: no
   probe artifact exists, and none can be minted for it by this
   design, since §5.1 requires baselines from a forward pass over
   that base, which does not exist.)
3. The artifact's architecture descriptor matches §3.1 exactly
   (hyperparameters, tensor names, shapes). A forward path asked
   to run any other shape refuses — it is a miniature executor,
   not a general one.
4. The candidate's targets are all tensors of this base (already
   guaranteed by the §4.2 safety suite's existence check — the
   probe binding adds the suite-side guarantee, not a second
   target check).

Any failure is not an error state; it is simply the absence of a
harness for this bundle, handled by §6.3.

### 6.3 "Not evaluated" remains the default — and the failure mode

Until SI-16 merges, §6.1 does not exist and this section is
vacuous: every transformer delta gets today's behaviour. After
SI-16, the rule is:

- **Not a miniature probe bundle, or no/invalid probe artifact,
  or binding mismatch (§6.2), or harness startup validation cannot
  reproduce the recorded base baselines within float tolerance,
  or the forward path raises at any point** → checks 3/4 are
  reported exactly as today (`passed=False`, `measured=None`,
  `TRANSFORMER_PROBES_NOT_EVALUATED` — or a successor string that
  states the specific reason with the same plainness), and the
  verdict logic is today's: decisive checks decide REJECT vs
  ESCALATE; ADMIT is unreachable.
- A harness exception is **never** a probe failure of the
  candidate (the candidate did not fail; the measurement failed)
  and never a pass. It is "not evaluated," with the exception
  recorded in the check detail for the journal.

### 6.4 Verdict mapping (when probes ARE evaluated)

With §6.2 satisfied and both probe checks genuinely run, the
transformer branch's verdicts become — for miniature bundles
only:

| Constitution / safety / provenance | Identity (3) | Capability (4) | Verdict |
|---|---|---|---|
| any fails | (any) | (any) | **REJECT** (unchanged — decisive checks) |
| all pass | not evaluated | not evaluated | **ESCALATE** (unchanged — §6.3) |
| all pass | evaluated, fails | (any) | **REJECT** |
| all pass | (any) | evaluated, fails | **REJECT** |
| all pass | evaluated, passes | evaluated, passes | counts toward **ADMIT** — see staging below |

Notes on the mapping, each deliberate:

- **Evaluated-and-failed is REJECT, not ESCALATE.** Today the
  unevaluated checks never cause a REJECT because absence of
  evidence is not evidence of harm. A *measured* drift beyond
  tolerance or a *measured* capability drop below the floor is
  evidence of harm, and the demo path already treats failed
  checks 3/4 as REJECT (`all(checks)` or REJECT). Escalating a
  measured-bad delta to the operator for possible admission would
  invert that meaning. (Consequence for SI-10: its admission path
  already refuses any non-ESCALATE verdict, so a probe-failed
  delta becomes inadmissible by anyone — the correct result, and
  §8's SI-17 must test it explicitly.)
- **Evaluated-and-passed staging.** A passing probe result does
  not flip admission behaviour the day it lands. §8 stages it:
  SI-16 wires probes in **shadow mode** — results are computed
  and journaled, the verdict remains ESCALATE — and only SI-17,
  gated on shadow evidence and an explicit recorded decision,
  makes a fully-passing miniature delta ADMIT-eligible. Even
  then: ADMIT-eligible means the gate *can* return ADMIT for a
  miniature bundle; the autonomous loop's handling of a
  transformer ADMIT (materialise + VERIFY, §6.6) is part of
  SI-17's scope, not an assumed consequence.
- **Nothing here touches non-miniature bundles.** For Glimmer and
  every other transformer base, the third row of the table is the
  only reachable one, forever, until some future RFC designs a
  harness for that base — which this RFC explicitly does not
  (§7).

### 6.5 Journal and report labelling

- A genuinely evaluated probe check reports `measured` as the
  real float (worst drift / worst regression), `passed` as the
  real boolean, and a detail naming the worst probe id, the
  tolerance/epsilon, the suite artifact hash, and — for check 4 —
  both arms' outcomes. This mirrors the demo checks' existing
  detail style, with the artifact hash added so a journal reader
  can tell *which suite* produced the number.
- A not-evaluated check keeps `measured=None`. The distinction
  between "measured 0.0" and "not measured" must survive in the
  journal exactly as it does today — it is the whole point.
- SI-10's admission basis string remains truthful for the bundles
  it serves: for a non-miniature bundle nothing changes. If an
  operator ever admits a miniature delta that escalated *despite*
  shadow-mode probe results, the journaled admission must carry
  those results alongside the basis string — the operator saw
  evidence, and the record must say what it was. SI-16 owns this
  labelling detail; this RFC fixes the requirement.

### 6.6 VERIFY after materialisation

For a miniature bundle, a post-materialisation VERIFY step can —
for the first time for any transformer kind — re-run the real
probes against the new head, in direct analogy to the demo loop's
`_verify_head` (identity within tol, capability no new regression
beyond eps vs the pre-cycle baseline), with rollback on failure
via the existing rollback core. This becomes possible only with
SI-17's admission flip and is specified there; until then,
`si.loop._verify_transformer_apply` (the apply smoke test) remains
the only transformer VERIFY, labelled as what it is, and nothing
in the harness work renames or upgrades it.

---

## 7. Non-goals

Explicit, and binding on §§8's slices:

- **No Glimmer forward pass.** Not full, not partial, not "just
  the language tower." The Glimmer seed remains store-and-verify
  only; `si serve` keeps returning `501` for it.
- **No GPU, no cloud, no spend.** CPU only, this VM only — the
  standing project rule, restated because a forward path is
  exactly where GPU temptation would appear.
- **No training.** No gradient training of the miniature, no
  delta authorship, no learning loop for transformer deltas.
  (§3.3's construction is fixed seed-time construction with
  assertions, not training; §4.4.)
- **No autonomous admission change in this slice (SI-13)** — and
  none in SI-14/SI-15 either. Admission behaviour changes only in
  SI-17, only for miniature bundles, only after SI-16's shadow
  evidence (§6.4, §8).
- **No serving.** The probe forward path is not exposed via
  `si serve` or any HTTP surface. A miniature bundle is still a
  transformer bundle to serve: `501`, until a separate decision
  says otherwise.
- **No changes to checks 1, 2, 5** for transformer deltas, to the
  demo-delta path in any respect, or to SI-10/11/12 behaviour for
  non-miniature bundles.
- **No seed-format change** (`reference-v1`, per-tensor maps, and
  the demo seed are untouched), **no transformer squash**, no
  mixed demo/transformer stacks, no cross-bundle probe reuse
  (§6.2 forbids it mechanically).
- **No new heavy dependencies** (§4.5) and no external inference
  engine, embedded or otherwise.

---

## 8. Implementation plan

Four slices, each independently reviewable and mergeable, each
leaving the system in an honest state if the next never lands.
**No slice fakes a check:** at every intermediate point, any
bundle whose probes have not genuinely run reports "not
evaluated" and escalates. Ordering is load-bearing — gate wiring
(SI-16) depends on a forward path (SI-15) that depends on a seeded
miniature with baselines (SI-14).

### SI-14 — Miniature seed + probe artifacts (no forward path consumer yet)

- Implement §3's construction as a seed path for the miniature
  (builder + `si seed`-adjacent CLI or test-facing constructor —
  SI-14 records the choice), producing an ordinary per-tensor
  bundle (SI-7 machinery) with §3.4's honesty label in the
  manifest.
- Record the §5 suites and §5.4 baselines/margins/tolerances as
  the §5.5 probe artifact, journaled at seed time. **Note:** this
  slice needs the forward computation to compute expected outputs
  — it may implement the §4 forward as an internal seed-time
  helper, but it wires it to *nothing*: no gate, no loop, no
  serve. If SI-14's author prefers, SI-14 and SI-15 may swap
  their internal order; the external contract (nothing consumes
  probes until SI-16) may not change.
- *Acceptance criteria:* deterministic re-seeding produces
  byte-identical base blobs and identical probe artifacts;
  parameter/tensor inventory matches §3.1 exactly; seed-time
  assertions (§3.3) run and are tested — including a constructed
  seed that *fails* the echo floor and is refused; stored-bytes
  accounting shows the base stored once, probes adding only the
  §5.5 artifact; demo and Glimmer bundle behaviour untouched
  (existing suites green).

### SI-15 — Forward path (library capability, not inference)

- Implement `si/tiny_transformer.py` (§4.1), the
  `load_applied_tensor`-based weight assembly, and the §4.2
  candidate overlay with shared fold arithmetic.
- *Acceptance criteria:* a hand-computed single-layer reference
  case (small enough to verify by hand in the test) matches the
  implementation exactly; multi-layer outputs match an independent
  naive reimplementation in the test file (written from §4.1, not
  copied from the module); repeat runs are bit-identical;
  overlay-vs-materialised agreement (§4.2) is proven by applying a
  fixture delta both ways; rolled-back deltas do not affect the
  forward; the forward refuses any architecture descriptor other
  than §3.1's (§6.2 rule 3); **gate behaviour is byte-for-byte
  unchanged** — checks 3/4 still "not evaluated" for every
  transformer delta, miniature included (test asserts it).

### SI-16 — Probe suites + gate wiring, shadow mode

- Implement the probe `BundleContext` (§6.1), harness startup
  validation (§6.3: reproduce base baselines or report not
  evaluated), the §6.2 binding checks, and the gate's conditional
  probe evaluation for miniature bundles — **with the verdict
  held at ESCALATE** regardless of probe outcome (§6.4 staging).
  Probe-failed miniature deltas in shadow mode are journaled with
  their measured failures *and* the ESCALATE verdict, so the
  shadow record shows what SI-17's REJECT would have been.
- *Acceptance criteria:* every §5.6 reference delta produces its
  required measured outcome (zero passes with drift 0.0; benign
  passes with drift > 0; destructive fails check 3; echo attack
  fails check 4's accuracy arm; per-class sensitivity all
  measurable); binding-mismatch fixtures (wrong base hash, wrong
  architecture, missing artifact, tampered artifact hash) all
  yield "not evaluated" + ESCALATE, never a pass; a Glimmer-shaped
  fixture bundle (no artifact) behaves exactly as today; demo
  path journaled reports are unchanged; SI-10 admission of a
  shadow-escalated miniature delta journals the probe results
  with the admission (§6.5).

### SI-17 — Admission flip for miniature bundles only (conditional)

- **Precondition, not a schedule:** SI-16 shadow results exist
  over a meaningful set of fixture/operator deltas and have been
  reviewed; an explicit decision to proceed is recorded (operator
  note in the slice's doc, mirroring how SI-10 recorded its basis).
  If the shadow evidence is bad, this slice does not happen and
  the harness remains a measurement instrument — a legitimate end
  state.
- Implement §6.4's full verdict mapping for miniature bundles
  (probe-fail → REJECT; all-pass → ADMIT-eligible), the loop's
  materialise + probe VERIFY (§6.6) for ADMITted miniature
  deltas, and the SI-10 interplay (probe-failed deltas are
  inadmissible; its refusal is tested).
- *Acceptance criteria:* an end-to-end autonomous cycle on a
  miniature bundle admits a benign fixture delta, verifies it
  with real probes, and rolls back a destructive one — with every
  journal entry labelled per §6.5; every non-miniature fixture
  (including a Glimmer-shaped bundle) still escalates with checks
  3/4 "not evaluated"; the full existing test suite is green with
  its expectations for non-miniature behaviour untouched.

### What "done" means for the RFC

This RFC is fully discharged when SI-16 merges: at that point SI
can *measure* identity and capability for one honestly-scoped
transformer kind, and say so. SI-17 is a separate decision that
this RFC enables but does not presume.

---

## 9. Alternatives considered and rejected

- **Numeric surrogates as checks 3/4** (norm bounds, apply-smoke
  results, structural scores reported as identity/capability):
  rejected by SI-9 §4.3 before this RFC existed; §1.4 inherits
  the ruling. They remain what they are — safety/apply evidence,
  labelled as such.
- **Probing Glimmer with a partial forward path** (e.g. embedding
  + one layer): a partial stack's outputs are not the model's
  behaviour; drift measured on them would be a fabricated
  identity claim about a 30B multimodal model from a truncated
  computation. Rejected — §7's no-Glimmer-forward rule is not
  negotiable by truncation.
- **Adopting an external inference engine** (a transformer
  library or GGUF runtime) to get a "real" forward path: imports
  a second tensor/weight semantics, a heavy dependency (§2
  invariant 6), and — for Glimmer — a multi-GB CPU inference
  problem this VM's measured costs (§4.5) already rule out.
  Rejected for this design; a future RFC could revisit it with
  measurements, for a stated base, as its own project.
- **Reusing DemoMind's suites/metrics wholesale against a
  transformer-shaped base:** the metric code *is* reused (§6.1) —
  that is the sound part. The suites cannot be: 8-vector linear
  expectations measure a different object. Rejected as a suite
  source; adopted as a metric source.
- **Materialise-to-temp-bundle probing:** rejected in §4.2
  (duplicates bytes in pattern, unjournaled state, weaker
  integrity story than the verified-manifest overlay).

---

## 10. Risks and open questions

Honestly listed; each is owned by the slice named.

1. **Probe blindness in untested directions.** Probes sample
   behaviour at 24 inputs. A delta can move the miniature in a
   direction no probe input excites and pass both checks.
   §5.6's per-class sensitivity tests bound this risk — every
   tensor *class* is measurably live — but cannot eliminate it:
   passing probes are evidence, not proof, of continuity. All
   reporting must keep that epistemics (§6.5). (SI-16.)
2. **Tolerance calibration is empirical, and could be wrong in
   either direction.** §5.4 derives tolerances from measured
   margins, but the reference deltas of §5.6 are the real test:
   if benign and destructive reference deltas land on the same
   side of a tolerance, the tolerance — or the suite — is wrong
   and SI-16 must iterate in the open, with numbers, rather than
   tune quietly until fixtures pass. (SI-14/SI-16.)
3. **The echo task is trivial — by design, and that is a risk.**
   A floor under a trivial behaviour certifies little. The
   defence is honesty (§5.3 names it a toy everywhere it appears)
   plus the regression arm, which bounds drift on the same
   inputs independently of the task. Whether a richer constructed
   task (e.g. a two-token dependency the construction can also
   guarantee) is worth its construction risk is an open question
   for SI-14 — with the rule that a richer task whose seed-time
   floor assertion is flaky is worse than a trivial one that is
   solid. (SI-14.)
4. **Float determinism across environments.** Baselines are
   recorded per bundle (§4.3), and startup validation detects a
   forward path that cannot reproduce its own bundle's baselines.
   But a bundle carried across numpy versions/machines could
   flip from "evaluated" to "not evaluated" for purely numeric
   reasons. That is the *safe* direction (fail closed, §6.3), and
   it is recorded here so nobody "fixes" it by loosening the
   reproduction tolerance without a stated justification. (SI-16.)
5. **Candidate-overlay arithmetic drift.** §4.2 requires the
   overlay fold and `si/apply.py`'s fold to be shared code.
   If SI-15 cannot extract the shared fold cleanly, the fallback
   (two implementations + an exact-agreement test) is weaker and
   must be recorded as technical debt in SI-15's as-built notes,
   not passed over. (SI-15.)
6. **Shadow-mode interpretation.** SI-16 produces probe verdicts
   that do not decide anything. There is a social risk that a
   journal full of "would have passed" reads as permission.
   §6.5's labelling and SI-10's unchanged basis string are the
   mitigation; SI-17's precondition (a recorded decision) is the
   gate. (SI-16/SI-17.)
7. **Scope creep toward Glimmer.** The predictable pressure:
   "the harness works on the miniature — run it on Glimmer."
   It cannot (§6.2 is mechanical, §7 is explicit), and any future
   proposal to evaluate a real base needs its own RFC, its own
   forward-path feasibility measurements, and its own honesty
   review. This RFC grants no momentum to that idea beyond
   proving measurement is possible at toy scale. (Standing.)
8. **Open: positional embedding as a base tensor.** §3.3 leaves
   the positional table's representation to SI-14 (tensor vs
   derived constant). It affects the tensor inventory and the
   targetable set; SI-14 must decide and record it before
   baselines are frozen. (SI-14.)
9. **Open: probe artifact format details.** JSON float arrays vs
   base64 float32 for expected logits (§5.5): JSON is auditable
   and larger; base64 is compact and opaque. At tens of KiB
   either works; SI-14 picks, preferring auditability unless a
   measured cost says otherwise. (SI-14.)
10. **Open: does the miniature need a `si serve` story ever?**
    This RFC says no (§7). If a future slice wants to demo the
    miniature interactively, that is a serving decision with its
    own honesty labels — not a consequence of the probe forward
    path existing. (Standing.)

---

## 11. Summary

Checks 3 and 4 cannot be honestly evaluated for transformer
deltas today, for three concrete reasons: no forward pass, no
probe suite, no baseline (§1). SI-10's operator admission is the
honest stopgap, and its journal says so on every admission. This
RFC designs the exit from the stopgap at the only scale this
runtime can execute: a constructed 22,944-parameter miniature
transformer (§3), a numpy forward path that consumes base +
deltas through the existing apply-at-load machinery plus a
validated candidate overlay (§4), identity and echo-capability
suites with seed-time baselines and margin-derived tolerances
(§5), and gate wiring that spends probe evidence only on the
bundle it was recorded for — shadow mode first, any admission
change last and separately decided (§§6, 8). For Glimmer, and
for every other real transformer base, nothing in this document
changes anything: the probes stay "not evaluated," and valid
deltas still escalate to the operator. That sentence remains
true until working code, reviewed and merged, makes a narrower
version of it false — for the miniature only.

---

*RFC SI-13 — design only. Implementation begins, if approved, at
SI-14. The gate is the product; this is the gate learning how to
measure what it currently, honestly, refuses to judge.*
