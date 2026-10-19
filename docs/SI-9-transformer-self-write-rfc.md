# SI-9 RFC — Self-Writes Against Real Transformer Tensors

> **Status:** Design only. This document proposes a format, a gate
> extension, and an apply path. It changes no production code, no
> manifest schema, and no behaviour. Nothing described here exists yet
> unless a section explicitly says it does today.
>
> **Scope discipline:** CPU only. No training loop, no GPU, no full
> transformer inference. Terminology: this is machine learning
> infrastructure, not "custom LLM software."
>
> **As-built note (SI-9a, PR pending at time of writing):** §6.1 is
> implemented — the §2 blob layout and §3 manifest record exist as
> data definitions plus pure validation in `si/transformer_delta.py`
> (a new module; see the notes in §2.3, §3, and §6.1). No gate, loop,
> or serve integration exists yet; those remain SI-9b / SI-9c.
>
> **As-built note (SI-9b):** §6.2 is implemented — the §4.2 structural
> suite is the gate's safety check for `transformer-delta-v1` intents,
> the §4.5 policy keys are live with fail-closed defaults, and §4.3 is
> implemented exactly as specified (identity/capability report
> "not evaluated" and a valid transformer delta ESCALATES; see the
> notes in §4.2, §4.5, and §6.2). No apply-at-load or serve integration
> exists yet; that remains SI-9c.

---

## 0. Honest state today (the baseline this RFC starts from)

Everything in this section is verifiable in the code and in the seeded
bundle on disk at the time of writing (main tip `2ef672c`, PRs #1–#10
merged).

- **SI's minds are text/linear demo minds.** `DemoMind`
  (`si/minds.py`) is a single linear map, `y = W @ x`, over an 8×8
  float32 base matrix named `W`. `MockMind` is scripted. There is no
  transformer mind.
- **The self-write loop has never operated on a real transformer
  tensor.** `LoopContext` (`si/loop.py`) loads exactly one base tensor
  — `load_base_tensor(..., "W")` — and a delta stack whose blobs must
  contain exactly the keys `delta.B` and `delta.A`. A Glimmer bundle
  has no tensor named `W` in that sense and no path through
  `run_cycle` as a transformer.
- **Seeding stores base + provenance only.** SI-8 (PR #10) seeded the
  real Muse Glimmer 30B checkpoint by reference
  (`seed_format: "reference-v1"`): architecture
  `MuseGlimmerForConditionalGeneration` (`model_type: muse_glimmer`),
  bfloat16, 1,436 tensors, 59,553,253,376 logical bytes,
  `stored_bytes: 0`. The bundle directory is ~2.9 MB. The base is
  frozen and fully hash-verified on load — and that is all it is. No
  delta has ever been proposed, gated, or applied against it.
- **`si serve` is honestly incapable here.** A transformer bundle
  returns `501` on `POST /infer`
  (`TRANSFORMER_NOT_SUPPORTED` in `si/serve.py`): SI has no
  transformer inference engine, and output is never fabricated.
- **One partial exception, stated precisely:** `si export`
  (`si/export.py`) can *materialize* a seeded checkpoint plus deltas
  into a GGUF. Its delta resolution for multi-tensor bases is a
  substring match — a delta's module string is applied to every base
  tensor whose name *contains* the string, failing only if the
  resulting shape mismatches. That is an export convenience for the
  demo-era delta shape, not a transformer self-write model: targeting
  is implicit, there is no per-tensor base-hash binding, and export
  materializes *all* base tensors as float32 — for Glimmer that means
  loading the whole 30B base. It must not be mistaken for the apply
  path proposed in §5.
- **Verification of the by-reference base is expensive, measured:**
  a full `load_bundle` re-verification of the Glimmer bundle took
  ~29 minutes (1,756.9 s) of streaming reads on this VM, because every
  load re-checks every referenced source file (existence, exact size,
  full streaming SHA-256) and every tensor's canonical hash. Any
  design that adds a second full pass per cycle is dead on arrival;
  §7 returns to this.

The gap, in one sentence: **SI can remember a transformer, prove the
memory is intact, and refuse to fake running it — but it cannot yet
learn into one, even by a single gated tensor.**

---

## 1. Problem statement

### 1.1 Shape of the mismatch

| Aspect | Demo loop today | Transformer base (Glimmer seed) |
|---|---|---|
| Base | one tensor, `W`, 8×8, float32, stored as a blob in the bundle | 1,436 named tensors, bfloat16, stored *by reference* outside the bundle (`base.tensor_refs`) |
| Tensor identity | implicit — there is only one tensor | per-tensor: name → `{blob: sha256:…, shape, dtype, dtype_code, source_file, file_offset, byte_length}` (SI-8) |
| Delta | one adapter: keys `delta.B` / `delta.A`, one implicit target | must name *which* of 1,436 tensors it modifies, and prove it was written against *those* base bytes |
| Targeting | `target_modules: ["demo_linear"]`, gate-checked against an allowlist of module strings | module strings are meaningless at tensor granularity; real names look like `model.language_model.layers.17.self_attn.q_proj.weight` (52 language layers) or `model.vision_tower.layers.3.attn.q_proj.weight` (50 vision layers) |
| Dimensions | 8×8; any rank ≤ 8 works | wildly heterogeneous: `lm_head.weight` is `[202048, 6656]`; an MLP gate projection is `[19968, 6656]`; a layernorm weight is 1-D `[6656]`; a vision bias is 1-D `[1536]` |
| Gate probes | identity/capability probes run the candidate through `forward_candidate` — cheap, real | no forward pass exists for a transformer in this runtime; probes cannot run |
| Cost of touching the base | negligible | one tensor ranges from 3,072 bytes (vision biases) to ~2.69 GB (`lm_head.weight` raw); the whole base is ~59.5 GB |
| Storage rule | blobs are content-addressed, skip-if-exists | Mark's design rule, absolute: **SI should never duplicate bytes** — the by-reference seed stores zero tensor bytes and that must stay true after deltas arrive |

### 1.2 What "solving" it must mean (and not mean)

A solution is: a delta format that binds to named base tensors by
content hash; a gate that can validate such a delta *without* an
inference engine and says so honestly; and an apply path that can
produce `base_tensor + delta` for an individual named tensor, on
demand, on CPU, with numeric tests — while base bytes are never
modified and never copied into the bundle.

A solution is **not**: training on Glimmer, running Glimmer, serving
Glimmer, or claiming the demo probes say anything about a transformer.
Those remain out of scope (§6.3).

### 1.3 Constraints inherited from existing design

These are not negotiable inside this RFC; a proposal that breaks one
is a different project:

1. **Base is frozen forever** (container spec rule 3). All learning is
   a stack of deltas; rollback is moving `head`, never surgery.
2. **Never duplicate bytes.** A blob whose hash already exists is
   never rewritten or copied. Seed imports by reference; deltas must
   be *new* bytes only, sized like adapters, not like checkpoints.
3. **No ungated write.** The only path from proposal to bundle change
   passes through the five-check ternary gate (`ADMIT` / `REJECT` /
   `ESCALATE`), and every verdict is journaled in NEDB, rejections
   included.
4. **Journal before materialize; history is append-only.** Rejected
   and rolled-back blobs stay on disk as evidence.
5. **Honest labels.** Unknowns are labeled unknown, unsupported is
   refused (501, not fabricated), and no metadata is invented.
6. **Full precision inside the bundle** (container spec rule 5):
   deltas are stored f16/bf16 minimum; the arithmetic story in §5.3
   must respect the base's bfloat16 nature rather than pretend it is
   float32.

---

## 2. Proposed write model — named-tensor delta artifacts

### 2.1 Core idea

A transformer self-write is a **delta artifact**: one new immutable,
content-addressed safetensors blob plus one manifest delta record
(§3), where the record names *every* base tensor the delta touches
and binds each touch to the base tensor's existing content hash.

Nothing about the base changes. The delta does not contain base
bytes, does not contain a modified copy of any base tensor, and cannot
be interpreted without the manifest's base section — exactly like
today's deltas cannot be interpreted without their manifest record.

### 2.2 Delta kinds

Two kinds cover the real tensor population. Conflating them is how
the demo format breaks: low-rank factorisation only makes sense for
matrices.

- **`lora` (2-D weight matrices).** For a base tensor
  `W` of shape `[out, in]`, the delta stores factors
  `A ∈ R^{rank × in}` and `B ∈ R^{out × rank}` and the applied update
  is `ΔW = (alpha / rank) · B @ A` — the same convention `DemoMind`
  and the gate already use (`scale = alpha / rank`), so existing norm
  accounting carries over. Eligible targets: any 2-D tensor, e.g.
  `…self_attn.q_proj.weight`, `…mlp.gate_proj.weight`,
  `model.vision_tower.…attn.q_proj.weight`.
- **`additive` (1-D tensors: norms, biases).** For a base vector `w`
  of shape `[n]`, the delta stores a single vector `d` of shape `[n]`
  and the applied result is `w + d`. A rank factorisation of a vector
  is a scalar multiple — dressing it up as LoRA would be theatre.
  Eligible targets: 1-D tensors such as
  `…input_layernorm.weight` (`[6656]`) or vision biases (`[1536]`).

In the seeded Glimmer checkpoint every tensor is in fact 1-D (713
tensors) or 2-D (723 tensors) — even the vision patch embedding is
stored flattened, as `[1536, 1176]`. Tensors of any other
dimensionality, should a future checkpoint contain them, are **not
targetable** in this design; the gate rejects them (§4). Embedding / `lm_head` matrices are 2-D and
therefore *format*-eligible, but §4.4 and §5.4 argue they should be
excluded by policy in the first slices on cost grounds alone.

### 2.3 Blob layout

One delta = one blob (preserving today's atomicity: one journal
entry, one manifest record, one content hash, one thing to roll back).

Inside the safetensors blob, keys are indexed, not name-derived —
base tensor names contain dots and layer indices, and embedding them
in safetensors keys invites parsing ambiguity:

```
t0.A   float32, [rank, in]        # lora targets only
t0.B   float32, [out, rank]       # lora targets only
t1.d   float32, [n]               # additive targets only
…
```

The manifest record (§3) is the authoritative map from `t{i}` to a
named base tensor. The blob also carries safetensors metadata
mirroring today's adapter metadata, extended per target:
`format: "transformer-delta-v1"`, plus the JSON target list duplicated
into metadata so a blob inspected in isolation still says what it
claims to modify. If the metadata and the manifest disagree, the blob
is invalid — the gate checks both (§4.2) and `load` paths must treat
mismatch as tamper, consistent with how `load_bundle` treats manifest
inconsistency today.

**Storage dtype.** Delta factors are stored in float32 in this
proposal. Rationale: factors are small (rank ≪ dimensions), the demo
adapters already store float32, and the gate's norm accounting is
float arithmetic. Storing bf16 factors to "match the base" would save
little and complicate exact numeric tests. This is a deliberate,
stated choice — revisiting it is an SI-9a decision (§6.1), not an
accident.

> *As-built in SI-9a:* float32 factor storage is **confirmed** —
> `si/transformer_delta.py` requires every factor's safetensors dtype
> to be `F32` and rejects anything else. The metadata duplication is
> also pinned: metadata `targets` is a JSON array carrying, per
> target, exactly `name`, `kind`, `factor_keys`, and (`lora` only)
> `rank`/`alpha`; validation compares it against the manifest record
> on exactly those fields, and any disagreement is invalid (tamper).

**Dedup.** The blob is written through the existing skip-if-exists
content-addressed path (the same pattern as `_write_blob` in
`si/loop.py`): if the SHA-256 already exists in `blobs/sha256/`, it is
not rewritten. Identical factors proposed twice therefore cost zero
additional bytes, and no code path copies base bytes — base tensors
are referenced, never read into the blob.

### 2.4 How a delta references base tensors by hash

Each target entry (§3) records:

- `name` — the exact base tensor name (exact match; **substring
  matching, as used by `si export` today, is explicitly not carried
  over** — it is ambiguous by construction: `layers.1` is a substring
  of `layers.17`'s neighbourhood in spirit and of several real names
  in fact).
- `base_blob` — the canonical-blob hash the base manifest already
  records for that tensor: `base.tensor_refs[name].blob` for a
  by-reference base, `base.tensor_blobs[name].blob` for a copy-mode
  base. SI-8 made this hash *mode-independent* — the same checkpoint
  yields the same per-tensor hashes in either storage mode — so a
  delta binds to the tensor's *content identity*, not to where the
  bytes happen to live.
- `base_shape`, `base_dtype` — copied from the same base record at
  proposal time, so the gate can check conformance against the
  manifest alone, without opening the checkpoint (§4.2).

Because the base is frozen, a `base_blob` mismatch at gate or apply
time can only mean one of: the bundle was re-seeded (which existing
code already refuses on a bundle with history), the manifest was
tampered with, or the delta was authored against a different base.
All three are fail-closed rejections, never silent adaptation.

### 2.5 What a delta is not

- Not a patch file against shard bytes (offsets into
  `model-0000X-of-00002.safetensors` are the *seed's* business; a
  delta that encoded shard offsets would break under copy-mode
  seeding of the same checkpoint, whose identity hashes are
  identical).
- Not a new base. Applying deltas never produces a stored, modified
  checkpoint (§5). Squash for transformer deltas is deferred (§6.3) —
  today's `si squash` composes demo adapters and its equivalence
  argument does not automatically transfer.
- Not training output — yet. SI-9 defines the artifact and its
  validation. Who *authors* a delta (a future training process, an
  operator tool, a test fixture) is out of scope; the gate must not
  care, and §4 is written so it doesn't.

---

## 3. Manifest schema (proposal)

Additive only, in the spirit of the container spec's versioning rule
(minor additions are new optional keys; a runtime that doesn't
understand transformer deltas must still load, verify, log, roll
back, and serve-501 a bundle that carries them — it must never
*misapply* one, and §5.5 makes application opt-in per code path).

A transformer delta record extends today's delta record
(`id`, `parent`, `blob`, `bytes`, `rank`, `alpha`, `target_modules`,
`norm`, `created_at`, `self_edit_id`) with:

```json
{
  "id": "d0001",
  "parent": "base",
  "kind": "transformer-delta-v1",
  "blob": "sha256:…",
  "bytes": 123456,
  "norm": 0.0031,
  "created_at": "…",
  "self_edit_id": "se_0007",
  "targets": [
    {
      "name": "model.vision_tower.layers.0.attn.q_proj.weight",
      "base_blob": "sha256:…",
      "base_shape": [1536, 1536],
      "base_dtype": "bfloat16",
      "kind": "lora",
      "rank": 4,
      "alpha": 8,
      "factor_keys": ["t0.A", "t0.B"],
      "target_norm": 0.0031
    }
  ]
}
```

Notes and rules:

- `kind` distinguishes transformer deltas from demo deltas. Bundles
  are expected to be homogeneous in practice (a demo base takes demo
  deltas; a transformer base takes transformer deltas), and the gate
  rejects a transformer delta against a demo base and vice versa.
  Mixed stacks are a non-goal.
- `targets` is a list because one *intent* ("adapt layer 17's
  attention") naturally touches q/v together — but §4 caps its length
  by policy, and the first slices should ship with a cap of **1**
  (see §6). A list with a cap of 1 is honest future-proofing; a scalar
  field would force a schema break later.
- Per-target `rank`/`alpha` replace the record-level pair for this
  kind; record-level `norm` remains the aggregate the existing gate
  escalation logic reads (defined in §4.3 as the root-sum-square over
  targets — a definition SI-9a must pin down, not leave implied).
  *As-built in SI-9a:* pinned — record-level `norm` is exactly
  `sqrt(sum(target_norm_i ** 2))`, where a `lora` target's norm is
  `||(alpha/rank) * B @ A||_F` and an `additive` target's norm is
  `||d||_2`, both recomputed from the blob's factors (in float64)
  during validation; a record that disagrees is invalid.
- `target_modules` is **not** reused for transformer deltas. It is a
  demo-era allowlist string; overloading it would make the gate's
  existing safety check pass or fail for the wrong reasons.
- The NEDB journal's `deltas` collection already stores `rank`,
  `alpha`, `target_modules`, `norm`, `blob`, `self_edit_id`,
  `source_seqs`, `status` per delta; the journal is schemaless by
  design (see `docs/nedb-journal-schema.md`), so `targets` rides along
  as an extra key with no schema migration. The journal spec doc
  should gain a row for it in SI-9a.

---

## 4. Gate integration

The gate's contract does not change: five checks, ternary verdict,
every verdict journaled, constitution intents still `ESCALATE`,
rollback still lightly evaluated. What changes is *what each check
can honestly compute* for a transformer delta. This section maps the
existing checks in `si/gate.py` onto transformer deltas and names the
new sub-checks — and, just as important, the checks that **cannot**
be honestly run yet.

### 4.1 Check 1 — Constitution (unchanged)

Reason required (already a malformed-intent rejection), blocklist
scan over the intent. No transformer-specific change. A delta intent
without a reason is rejected before any tensor is touched, exactly as
today.

### 4.2 Check 2 — Safety bounds (extended; the workhorse)

For demo deltas this check is rank cap, allowlisted modules, stack
depth, and a single norm cap read from the candidate dict. For
transformer deltas it becomes a **structural validation suite** that
runs against the manifest and the delta blob — deliberately *without*
reading base tensors, so gating does not inherit the base
verification cost (§7.1). Sub-checks, all required:

1. **Target existence & identity.** Every `targets[i].name` exists in
   the base's tensor map (`tensor_refs` or `tensor_blobs`), and
   `targets[i].base_blob` equals the manifest's recorded hash for
   that name. (§2.4: mismatch = reject.)
2. **Shape conformance.** For `lora`: base shape is 2-D `[out, in]`;
   `A` is `[rank, in]`, `B` is `[out, rank]`, matching the blob's
   actual safetensors header — not just the record's claim. For
   `additive`: base shape is 1-D `[n]`; `d` is `[n]`. Any other base
   dimensionality rejects.
3. **Dtype conformance.** Factor dtypes are as declared (float32 per
   §2.3); base dtype matches `base_dtype` in the record and the base
   manifest.
4. **Finite values.** Every factor element is finite (no NaN/Inf).
   This is a streaming scan over the delta blob only — deltas are
   small by budget (sub-check 6), so this is cheap and has no excuse
   to be sampled.
5. **Norm budgets.** Per-target Frobenius norm of the *scaled* update
   (`‖(alpha/rank)·B@A‖_F`, computed without materialising the full
   update where rank ≪ dimensions: `‖B@A‖_F` via the Gram matrices,
   or directly — SI-9b must pick and test one) and of additive vectors
   (`‖d‖₂`), each under a per-target cap; aggregate record norm under
   the existing `norm_cap`; existing `escalation_norm` semantics
   preserved (over the escalation threshold but under the cap →
   `ESCALATE`, not `ADMIT`).
6. **Size budgets (new policy keys, §4.5).** Delta blob bytes ≤
   `delta_byte_cap`; `len(targets)` ≤ `max_targets_per_delta`; rank ≤
   existing `rank_cap`. These budgets are what keep "small adapters on
   a frozen base" a fact rather than a slogan — an unbounded "delta"
   is a second checkpoint wearing a costume.

   *As-built in SI-9b (norm method, §4.2.5's open choice):* the gate
   recomputes per-target LoRA norms from the **Gram matrices** in
   float64 — `‖B@A‖_F^2 = trace((B^T B) @ (A A^T))` — never
   materialising the `[out, in]` product. The Gram value must agree
   with the record's `target_norm` (which SI-9a validation checks
   against a direct `B@A` computation) to relative tolerance 1e-6;
   cap comparisons are strict (`measured > cap` fails) on the float64
   Gram values, tested at exactly the cap and just above it. The
   suite is implemented as named sub-checks on the safety check
   (`subchecks` in the journaled gate report), with SI-9a's
   `validate_transformer_delta` running as a final catch-all so the
   gate can only be stricter than the format validation. Two trust
   decisions are pinned in `si/gate.py`: the base manifest comes only
   from the bundle context (never the candidate), and allowlist
   patterns are case-sensitive `fnmatch` patterns — a wildcard-free
   pattern is an exact name, with no implicit substring matching.
7. **Allowlist.** Target names must match policy
   `allowed_target_patterns` (new key, §4.5) — exact names or
   layer-scoped patterns, resolved exactly. The demo policy's
   `allowed_modules: ["demo_linear"]` grants a transformer bundle
   *nothing* by default, which is the correct default: a Glimmer
   bundle's policy must opt targets in explicitly.
8. **Stack rules (unchanged semantics).** `max_stack_depth` still
   applies; one delta per cycle still applies (the loop is serial by
   design).

Failure of any sub-check = that check fails = `REJECT`, with the
per-sub-check detail journaled in the gate report, as today's gate
already records per-check `detail`/`measured`.

### 4.3 Checks 3 & 4 — Identity continuity & capability floor (the honest problem)

Today these checks run the candidate through the mind's forward pass
against probe suites. **For a transformer bundle there is no forward
pass, no probe suite, and no baseline.** Any implementation that
returned "pass" here would be fabricating the two checks that exist
specifically to catch a mind damaging itself. This RFC therefore
proposes, in order of preference:

- **SI-9b behaviour: `ESCALATE`.** A transformer delta that passes
  constitution, safety, and provenance is escalated to the operator
  with the gate report stating plainly: *identity and capability
  checks could not be evaluated — no transformer probe harness
  exists*. The operator's admission is journaled like a constitution
  escalation. This keeps the loop's invariant — no ungated write —
  while refusing to counterfeit the gate.
- **Numeric surrogates are not checks 3 & 4.** Bounding a delta's norm
  bounds how much one tensor *can* move, and §5's apply tests prove
  the arithmetic is the arithmetic claimed. Those are safety/apply
  properties. If they are ever surfaced, they must be labelled as
  what they are and must never be reported as identity or capability
  results.
- **A real probe harness is future work** (post-SI-9): it would need
  at minimum a CPU forward path for a *small* transformer the runtime
  can actually execute, with identity/capability suites defined
  against it. Designing it here would be speculation stacked on
  speculation; this RFC only records the requirement that checks 3/4
  stay un-faked until then.

Consequence, stated plainly: **under SI-9b, transformer deltas are
operator-admitted, not autonomously admitted.** The autonomous loop
(`si run`) must halt on the `ESCALATE`, as it already halts on any
escalation. That is a limitation, and it is the correct one.

### 4.4 Check 5 — Provenance completeness (extended)

Existing requirements stand: non-empty `parents`, non-empty
`source_seqs`, a `reason`. Transformer deltas add: the intent must
declare its target list, and the journaled `self_edits` row plus the
manifest record together must answer, forever, *which base tensors,
at which content hashes, this delta modifies* — that is exactly the
`targets` array of §3, journaled with the delta (§3, journal note).
Rollback semantics are unchanged and worth restating for this kind:

- Rollback moves `head` back; the delta blob and its journal rows
  stay. Because deltas are applied at read time (§5), a rolled-back
  delta simply stops being applied — there is no "un-apply" write
  and no base bytes to restore.
- Post-materialisation `VERIFY` (loop step 6) for transformer deltas
  can only re-run the §4.2 structural suite plus an apply smoke test
  on the delta's own targets (§5.5) — not probes. The journal must
  record it as such.

### 4.5 Policy additions (proposal)

New keys in `policy.json` (content-hashed as today; changing them is
itself a gated, journaled act):

| Key | Proposed default | Meaning |
|---|---|---|
| `allowed_target_patterns` | `[]` (nothing allowed) | exact names / layer patterns a transformer delta may target |
| `max_targets_per_delta` | `1` | breadth cap per self-write |
| `delta_byte_cap` | small — SI-9b to fix a number with evidence (order: a few MB) | blob size budget |
| `per_target_norm_cap` | ≤ `norm_cap` | per-tensor movement budget |
| `allow_additive` | `false` | whether 1-D additive targets are permitted at all |

Existing keys (`norm_cap: 1.0`, `escalation_norm: 0.5`,
`rank_cap: 8`, `max_stack_depth: 8`) keep their meanings. The demo
defaults are *not* proposed as Glimmer defaults — SI-9b must set
transformer policy numbers deliberately, per bundle, at seed/policy
time, and say why.

*As-built in SI-9b:* the defaults are pinned in
`si.bundle.DEFAULT_POLICY` — `allowed_target_patterns: []`,
`max_targets_per_delta: 1`, `allow_additive: false`,
`per_target_norm_cap: 1.0` (= `norm_cap`, the loosest value the
"≤ `norm_cap`" bound allows; bundles may tighten it), and
`delta_byte_cap: 4 MiB` — fixed with evidence: a rank-8 LoRA on the
largest ordinary Glimmer matrix shape (`[19968, 6656]`) needs
~0.81 MiB of float32 factors, while a rank-8 lm_head-scale delta
(`[202048, 6656]`) would need ~6.4 MiB, so the cap admits the former
with headroom and excludes the latter (§5.4). A policy key that is
absent falls back to these defaults; a key present but malformed is
treated as its most restrictive value. These remain *runtime*
defaults, not Glimmer policy: a seeded bundle's operator still sets
its allowlist deliberately, per bundle, as a journaled policy act.

---

## 5. Read / apply path

### 5.1 Principle: apply per tensor, at read time, in stack order

The applied value of a named tensor at head `H` is:

```
applied(name, H) = base(name)  ⊕  Δ₁(name)  ⊕  …  ⊕  Δₖ(name)
```

where `Δᵢ` ranges over the active chain (`_active_chain`, oldest
first — the same resolution squash and export already use), includes
only deltas whose `targets` name this tensor, and `⊕` is
`W + (alpha/rank)·B@A` for `lora` or `w + d` for `additive`.
Deltas that don't target `name` cost one manifest lookup, not a read.

There is **no materialised modified checkpoint** — not in the bundle,
not as a cache file. A caller that wants a tensor asks for it and
gets the computed array in memory. This follows the precedent the
codebase already set: `load_base_tensor` (`si/bundle.py`) resolves a
single named tensor across all base layouts, hash-verified on read,
without loading the checkpoint. SI-9c adds a sibling —
`load_applied_tensor(bundle, manifest, name)` in spirit — that starts
from `load_base_tensor` and folds the active transformer deltas for
that name.

### 5.2 The by-reference wrinkle

For a `reference-v1` base, `load_base_tensor` reads the tensor's byte
range from the source checkpoint and re-verifies that tensor's
canonical hash on read (existing SI-8 behaviour). The apply path
inherits both the benefit (no copy of the base exists to drift) and
the dependency (the source checkpoint must remain at its recorded
absolute path — already the stated price of the seed). Delta blobs,
by contrast, live in the bundle's own store, so a bundle with deltas
is *partly* portable: manifest + deltas travel; the base still points
home. That asymmetry already exists for the seed and is not worsened.

### 5.3 Dtype and arithmetic

- Base tensors are bfloat16. The existing read path widens bf16 to
  float32 losslessly (numpy has no bf16 dtype in this runtime; raw
  bytes and hashes remain bf16-exact). Apply arithmetic is therefore:
  widen base to float32 → add the float32 update → the *result's*
  dtype is a stated decision, not an accident:
  - In-memory consumers (tests, a future probe harness) get float32.
  - Anything that writes an applied tensor back into a safetensors
    container (e.g. a future export path) must cast to the base dtype
    (bf16) explicitly and accept the rounding — and an export that
    does so is producing a *deployment artifact*, outside the bundle,
    per the container spec's quant/export rule.
- Inside the bundle, nothing is ever written back. There is no
  "applied base" on disk to round, drift, or be mistaken for the
  frozen original.

### 5.4 CPU / memory feasibility — honest numbers

This VM is CPU-only with a 100 GB home volume, ~23 GB free after the
Glimmer seed. Consequences:

- **Full-base anything is out.** Loading all Glimmer tensors as
  float32 (as `si export` does) is ~119 GB of float32 — it does not
  fit this machine's RAM story or its disk story as an intermediate,
  and nothing in SI-9 may require it. Per-tensor apply is not an
  optimisation; it is the only version of this that exists here.
- **Per-tensor apply is cheap for small targets and expensive for
  large ones — both facts matter.** Applying a rank-`r` LoRA to a
  `[1536, 1536]` vision projection touches ~4.7 MB of bf16 and two
  small factor matrices; trivial. The same operation on
  `lm_head.weight` (`[202048, 6656]`) means forming or applying
  against a 1.34-billion-element matrix — the `B@A` product alone is
  `[202048, 6656]`, and computing it naively in float32 is a
  ~5.4 GB intermediate plus billions of multiply-adds on CPU. The
  standard trick — never form `ΔW`; compute `(x @ Aᵀ) @ Bᵀ` at use
  time — only helps a consumer doing matvec/matmul against the
  tensor, which SI-9 does not have (§5.5). **Policy consequence:**
  the first slices exclude embedding/`lm_head`-scale tensors via
  `allowed_target_patterns`, on arithmetic cost, not on taste.
- **Stack depth multiplies per-tensor cost linearly** and
  `max_stack_depth` already caps it at 8. Fine.
- **Full inference remains out of scope** — not "slow", *absent*: no
  attention, no sampling, no vision tower execution exists in this
  runtime, and SI-9 adds none. `si serve` keeps returning 501 for
  transformer bundles after SI-9. If a future slice changes that, it
  will be its own RFC with its own evidence.

### 5.5 Where apply is wired (and where it is not)

- **Wired (SI-9c):** a library-level `load_applied_tensor` used by
  tests and by the loop's post-write `VERIFY` smoke step (§4.4) —
  apply the just-admitted delta to its own target(s) and check the
  result is finite, correctly shaped, and equal to a directly
  computed reference in the test.
- **Not wired:** `si serve` (still 501), `si export` (its existing
  substring-resolution path is *not* silently upgraded; reconciling
  export with exact named-tensor deltas is a follow-up decision —
  leaving two apply semantics in one codebase without a plan would
  be worse than leaving export as-is and saying so, which this RFC
  does), and the demo `LoopContext` forward path (demo deltas only).

---

## 6. Phased plan

Each phase is independently PR-sized, reviewable, and mergeable
without the next one existing. No phase trains anything.

### 6.1 SI-9a — Delta format + manifest schema (design → code, no loop wiring)

- Implement the §2 blob layout and §3 manifest record as *data
  definitions plus pure validation*: a module (or `si/bundle.py`
  extension) that can construct, serialise, parse, and structurally
  validate a transformer-delta record/blob pair against a base
  manifest — with no gate, loop, or serve integration.
- Decide and document the open format questions this RFC leaves to
  9a: aggregate-norm definition (§3), float32 factor storage
  confirmation (§2.3), metadata/manifest disagreement = tamper
  (§2.3).
- Tests: round-trip construction/parse on synthetic small tensors;
  malformed records (unknown target, wrong `base_blob`, shape lies in
  record vs blob header) all fail closed. Synthetic fixtures only —
  no Glimmer reads required.
- Docs: this RFC moves from "proposal" to "as-built" notes for the
  format sections; journal spec gains the `targets` row (§3).

*As-built:* done in `si/transformer_delta.py` — construct / serialise
/ parse / validate for the record+blob pair against a base manifest,
with the module-location decision (new module, not a `bundle.py`
extension: `bundle.py`'s contract is storage/verification, and the
format layer must be importable by the SI-9b gate and SI-9c apply
path without bundle internals), the float32 confirmation (§2.3), and
the root-sum-square aggregate norm (§3) all recorded in that
module's docstring. Tests are synthetic-fixture only
(`tests/test_transformer_delta.py`): round-trip plus a fail-closed
case for every malformed class in this section. No gate, loop, or
serve code was touched.

### 6.2 SI-9b — Gate checks for transformer deltas

- Implement §4.2's structural suite as the safety check for
  `kind: "transformer-delta-v1"` intents; §4.5 policy keys with
  fail-closed defaults (empty allowlist admits nothing).
- Implement §4.3 exactly as specified: identity/capability for
  transformer deltas produce `ESCALATE` with the honest reason, never
  a synthetic pass. Loop/`si run` halting on escalation is already
  the behaviour — test that it happens.
- Tests: each sub-check has a dedicated failing fixture (hash
  mismatch, non-finite factor, over-budget bytes/norm, disallowed
  target, wrong-dimensionality target); a fully valid small delta escalates (not
  admits, not rejects) with the expected report; demo-delta gate
  behaviour is byte-for-byte unchanged (existing `tests/test_gate.py`
  green, untouched expectations).

*As-built:* done in `si/gate.py` (the §4 mapping, the structural
suite as safety sub-checks, the §4.4 provenance extension via a new
`Intent.targets` declared-target list), `si/bundle.py` (the §4.5 keys
in `DEFAULT_POLICY`), and one deliberate loop touch in `si/loop.py`:
`LoopContext` exposes the verified manifest as the gate's base
manifest, and `run_cycle` obtains a candidate for transformer intents
so the gate can evaluate them — this adds no materialize path, since
the SI-9b gate never ADMITs a transformer delta (valid → ESCALATE,
invalid → REJECT; both journaled, and `si run` halts, tested in
`tests/test_gate_transformer.py`). The demo-delta path is unchanged,
including its journaled report shape (`subchecks` appear only on the
transformer safety check). One scope decision, recorded: the gate
does not add a demo-intent-vs-transformer-base rejection — the demo
path is byte-for-byte unchanged per this section's own test
requirement, existing seeded demo-shaped bases legitimately run demo
cycles (SI-8 tests), and a true transformer base cannot reach the
demo gate through the loop at all (`LoopContext` requires the demo
`W` tensor). Apply-at-load, serve, and squash are untouched (SI-9c).

### 6.3 SI-9c — Apply-at-load for a narrow tensor subset + numeric tests

- Implement §5's `load_applied_tensor` for `lora` and (if policy
  design in 9b allows) `additive` targets, resolving through
  `_active_chain`, via `load_base_tensor` — per tensor, on demand,
  nothing materialised.
- Numeric tests on a **synthetic miniature transformer-shaped
  checkpoint** (a handful of named tensors with Glimmer-style names
  and shapes at toy dimensions, seeded in-test): applied result
  equals a hand-computed `W + (alpha/rank)·B@A` reference to float32
  tolerance; stack order matters and is tested (two deltas, both
  orders distinguishable); rolled-back delta no longer applies;
  untargeted tensors return base-identical values.
- One **real-checkpoint proof**, read-only against the existing
  Glimmer bundle: construct a delta fixture in-test (authored in the
  test, not proposed by any mind) targeting one small real tensor —
  e.g. a vision-tower projection (`[1536, 1536]`) — apply it, and
  verify shape/finiteness and that the delta-free read is unchanged.
  This proves the path against real bf16 bytes without claiming any
  learning happened. (Target choice must respect §7.1: the test reads
  individual tensors via the verified read path, and must not trigger
  repeated full `load_bundle` verifications per assertion.)
- `si serve` still returns 501; README's honesty labels updated to
  describe apply-at-load as a library capability, not inference.

### 6.4 Explicit non-goals

- **No training loop.** Nothing in SI-9 learns a delta from data.
  Authorship of deltas is out of scope (§2.5).
- **No GPU, no cloud, no spend.** CPU only, this VM only.
- **No full transformer inference** — no forward pass, no serving
  change, no probe harness (§4.3 records it as future work).
- **No changes to the by-reference seed format (`reference-v1`)** —
  deltas bind to the identity hashes SI-8 already records. If a phase
  believes it needs a seed-format change, that belief is itself an
  escalation: stop, write the justification, get a decision. (No such
  need is known today.)
- **No transformer squash**, no cross-bundle delta sharing, no mixed
  demo/transformer stacks, no embedding/`lm_head` targets in the
  first slices (§5.4).
- **No silent export reconciliation** (§5.5).

---

## 7. Risks & open questions

### 7.1 Verification cost (the dominant operational risk)

Measured: one full `load_bundle` verification of the by-reference
Glimmer base ≈ 29 minutes of streaming reads on this VM, and reads
through the verified path re-verify per tensor. A self-write loop
that pays anything like that *per cycle* is unusable, and a gate that
quietly skipped verification would be worse. Open questions SI-9a/b
must answer with measurements, not vibes:

- The §4.2 gate suite is deliberately manifest+delta-only (no base
  reads). Is that the right trust boundary? Argument for: the base
  was fully verified at `load_bundle` when the bundle was opened, the
  base is frozen, and the delta's `base_blob` binding is checked
  against the *verified manifest*. Argument against: a long-lived
  process could gate against a manifest verified hours ago while the
  source checkpoint rots on disk. Mitigation candidates: verify-once-
  per-process at bundle open (status quo), plus per-target streaming
  re-verification at *apply* time (already the read path's behaviour)
  — so a rotted source fails at use, loudly, even if gating passed.
- **Partial/streaming verification strategies** to evaluate: per-file
  (shard-level) verification caching keyed by (path, size, mtime) is
  *rejected in advance* if it weakens the fail-closed property SI-8
  chose deliberately (no spot checks, no mtime shortcuts — see the
  SI-8 notes in `si/seed.py`). Any caching proposal must prove it
  cannot pass a tampered source. Absent such proof, the honest cost
  stands and cycle cadence adapts to it.
- Test-time cost: SI-9c's suite must not re-verify the 59.5 GB base
  per test run. The synthetic-checkpoint tests carry the logic;
  the real-checkpoint proof (§6.3) is one bounded pass, marked and
  skippable like the existing slow tests.

### 7.2 Numerics

- bf16 → float32 widening is lossless; the reverse cast is not.
  Anything that ever persists an applied tensor (§5.3) inherits
  rounding that compounds across a delta stack. SI-9 persists
  nothing — but a future export reconciliation must confront this
  explicitly.
- Norm computation for large `B@A` without materialising the product
  (§4.2.5): the Gram-matrix identity is exact in exact arithmetic and
  merely very good in float32; SI-9b must state its tolerance and
  test at the cap boundary, because a norm check that is wrong at the
  boundary is a safety check that is wrong where it matters.

### 7.3 Semantic risks

- **A gated delta is not a *good* delta.** Until a probe harness
  exists, SI-9 guarantees structural validity, bounded size, and
  provenance — not that a delta improves, or even preserves, anything
  about the base model's behaviour. Every report, README line, and
  journal entry produced under SI-9 must keep that distinction
  visible. The 501-on-serve honesty standard applies to claims, not
  just endpoints.
- **Operator escalation fatigue.** §4.3 makes the operator the
  identity/capability check for transformer deltas. If delta volume
  ever grows, that becomes a rubber stamp. Recorded here so a future
  phase treats probe-harness work as the fix, not threshold-loosening.
- **Pattern allowlists can over-grant.** A pattern like
  `layers.*.self_attn.*` spans 52 layers × 4 projections; §4.5's
  defaults start at *empty*, and widening is a journaled policy act —
  but the RFC flags that pattern granularity is a real design choice
  for SI-9b, not a detail.

### 7.4 Open questions (for the SI-9a author, not rhetorical)

1. Should `additive` deltas exist at all in v1, or should v1 ship
   `lora`-only and add vectors once matrix targeting is proven?
   (This RFC includes them because norm/bias tensors are the
   smallest, cheapest proof targets — but a lora-only v1 is a
   defensible cut.)
2. One blob per delta (§2.3) vs one blob per target: atomicity argues
   blob-per-delta; dedup granularity argues blob-per-target (two
   deltas sharing one target's factors would dedupe). §2.3 chose
   atomicity; 9a should confirm with the squash-shaped future in
   mind.
3. Where does `load_applied_tensor` live — `si/bundle.py` beside
   `load_base_tensor`, or a new `si/apply.py`? (Lean: `si/apply.py`;
   `bundle.py` is already 745 lines and its contract is storage, not
   arithmetic.)
4. Glimmer is multimodal: is the vision tower in scope as a target
   class for the first proof, or language-only? §6.3's proof uses a
   vision tensor because it is small (`[1536, 1536]`), not because
   vision matters first — 9c may equally choose a small language-side
   target if one is comparably cheap; the layernorms are 1-D and the
   language matrices are ≥ `[6656, …]`, so the honest trade-off is
   size vs subsystem. Decide in 9c and record why.

---

## 8. Summary

SI today can hold a real 30B transformer by reference, prove it is
intact, journal its provenance, and refuse to pretend it can run it.
SI-9 proposes the smallest honest next step: deltas that name tensors
and bind to their hashes (§2–§3), a gate that structurally validates
them and escalates what it cannot yet judge (§4), and a per-tensor
apply path with numeric proof on a narrow subset (§5) — phased as
SI-9a (format), SI-9b (gate), SI-9c (apply + tests), with training,
inference, GPU, and seed-format changes explicitly out (§6.4). The
base stays frozen, the bytes are never duplicated, and nothing is
claimed that a test in this repo cannot check.

---

*RFC SI-9 — design only. Implementation begins, if approved, at
SI-9a. The loop is the product; this is the loop learning where its
edges are.*
