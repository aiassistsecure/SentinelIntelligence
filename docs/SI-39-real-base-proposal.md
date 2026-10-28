# SI-39: A Real-Base Proposal — One Named Checkpoint, For a Yes or a No
(Proposal Only)

> **Status: PROPOSAL ONLY — NOT APPROVED, NOT MEASURED, NOT
> DOWNLOADED.** This document names one specific real, trained
> safetensors checkpoint and asks the operator (Mark) a single
> yes/no question about it (§8). It downloads nothing, changes
> no production code, no test, no manifest schema, no policy
> default, and no gate behaviour. No feasibility measurement
> in this document exists: every A-1 quantity (forward wall
> time, assembled memory, verification cost) is **UNMEASURED**
> and stays unmeasured unless and until the operator says
> yes and a later slice does that work. Numbers below are
> labelled **sourced** (read from the cited Hugging Face page
> or file on 2026-10-28), **derived** (arithmetic over sourced
> numbers, shown so it can be checked), or **measured on this
> VM** (the VM's own observed shape, not a model measurement).
>
> **The one sentence that governs everything below:** RFC
> SI-36 §6 laid the real-base decision before the operator
> with its preconditions (SI-18 §3.7's A-1 to A-4) and its
> costs stated as categories, and took no decision; SI-37
> and SI-38 then completed the miniature line's argued work
> (the transformer squash, merged as PR #43); this proposal
> is the follow-through the loop owes the operator — one
> concrete candidate, its sourced facts, its honest costs,
> and one question — and it must never be cited as evidence
> that a download was approved, that any base is feasible on
> this VM, that any executor, probe, or admission path for a
> real base exists, or that the Glimmer question was
> reopened.
>
> **Scope discipline:** CPU only. No GPU, no cloud, no
> spend. Terminology: this is machine learning
> infrastructure throughout. This proposal was written from
> Hugging Face web pages and metadata only; no model weights
> were downloaded to write it.

---

## 1. The decision asked, stated first

**Approve `HuggingFaceTB/SmolLM2-360M-Instruct` as the one
named candidate base for the A-1 feasibility-measurement
slice — measurement only — or do not.**

A yes approves exactly two things, in this order:

1. Downloading the checkpoint files listed in §5.1 (about
   727 MB, derived from sourced file sizes) to this VM, to
   be held as the single stored copy under the
   never-duplicate-bytes rule (§5.2).
2. One A-1 slice that *measures* — on this VM, CPU only —
   forward wall time, assembled memory, and verification
   cost for this one family, and reports the measurements
   whether they flatter the proposal or kill it.

A yes does **not** approve: any gate or admission change,
any delta authorship against this base, any probe suite, any
executor being built (A-4 is a design obligation first),
any serving, or any second candidate. A-2, A-3, and A-4
remain separate design slices, sequenced after A-1 (§5.3),
each reviewable on its own. A no closes the real-base
question until the operator reopens it; the miniature line
and the Glimmer seed are unaffected either way.

This is one proposal, not a menu. §7 lists the alternatives
that were checked and rejected, with one-line reasons, so
the rejection record is auditable — none of them is on the
table.

---

## 2. The named checkpoint and its sourced facts

**`HuggingFaceTB/SmolLM2-360M-Instruct`** — the instruct
variant of the 360M member of the SmolLM2 family, published
by Hugging Face (HuggingFaceTB).

| Fact | Value | Label / source |
|---|---|---|
| Parameters | 361,821,120, all BF16 | sourced — HF API `safetensors` block for the repo [S1] |
| Licence | Apache-2.0 | sourced — model card, "License: Apache 2.0", and repo metadata `license: apache-2.0` [S1][S2] |
| Weights file | one file, `model.safetensors`, 723,674,912 bytes (724 MB as displayed on the files page) | sourced — HF API blob listing; files page [S1][S3] |
| Architecture class | `LlamaForCausalLM`, `model_type: "llama"` | sourced — the repo's `config.json` [S4] |
| Layers | 32 (`num_hidden_layers`) | sourced — `config.json` [S4] |
| Hidden size | 960 (`hidden_size`) | sourced — `config.json` [S4] |
| Intermediate size | 2,560 | sourced — `config.json` [S4] |
| Attention | 15 heads, 5 key/value heads (grouped-query attention), `head_dim` 64 | heads sourced — `config.json` [S4]; head_dim derived: 960 / 15 = 64 exactly |
| Vocabulary | 49,152; embeddings tied (`tie_word_embeddings: true`) | sourced — `config.json` [S4] |
| Context | 8,192 (`max_position_embeddings`), RoPE θ 100,000, RMSNorm ε 1e-05, SiLU MLP | sourced — `config.json` [S4] |
| Stored dtype | bfloat16 (`torch_dtype`) | sourced — `config.json` [S4] |
| Training | 360M model trained on 4 trillion tokens; instruct version produced by supervised fine-tuning plus Direct Preference Optimization | sourced — model card, Model Summary and Training sections [S2] |

Sources (all fetched 2026-10-28; see §9 for the full list):
[S1] the repo's Hugging Face API record, [S2] the model
card page, [S3] the repo files page, [S4] the repo's raw
`config.json`.

Two facts about the files page deserve plain statement,
because they decide what "download" means in §5.1. First,
the weights are a **single** safetensors file — no sharding,
no index file — so the seed-by-reference pattern (§5.2)
deals with exactly one large object. Second, the files page
shows a **5.02 GB repo total** [S3]: that total is mostly
ONNX export variants and training-run logs that this project
has no use for. The proposal's download is the weights file
plus the small configuration/tokenizer files (§5.1), never
a whole-repo clone.

---

## 3. Why this one

The selection bars, applied in order, were the ones the
record already sets: a real, trained checkpoint (not a
construction, not random data); safetensors availability; a
permissive, verifiable licence (Apache-2.0 / MIT or
similarly clear); tensor naming and configuration the
project's existing machinery can at least *describe* without
invention; and a size at which a CPU-only forward and
verification on this VM are honestly plausible — argued
from arithmetic, not asserted. The operator's standing
preferences bind throughout: no Qwen models (excluded
without evaluation, §7), and a GLM/Zhipu candidate preferred
if one genuinely fits (§4 records that check in full).

### 3.1 It is real and trained, which is the point of the exercise

The miniature line's open question — whether the authorship
programme says anything about a *trained* base — cannot be
asked of another construction. This checkpoint's card
records 4 trillion pretraining tokens and an SFT + DPO
instruct pipeline [S2]. Its behaviour is therefore not
something this project built and can re-derive: exactly the
property A-3 (a capability floor with baselines recorded
*without* a construction) exists to handle, and exactly why
the miniature's construction-relative suites do not transfer.

### 3.2 The licence is the clear kind

Apache-2.0, stated identically on the card and in the repo
metadata [S1][S2]. No registration step, no naming mandate,
no revocability clause, no field-of-use restriction. The
project can hold it, reference it, measure it, and — if
later slices ever justify it — build against it, without a
licence question hanging over any of those steps. §4 shows
this bar, not size, is what the preferred GLM candidate
fails.

### 3.3 Architecture fit: the smallest honest executor scope, not a drop-in

Fit here is claimed narrowly and precisely:

- **The project can already describe this family without
  inventing anything.** `si/arch.py` derives export metadata
  from a stored checkpoint `config.json` generically (the
  checkpoint-config path reads `num_hidden_layers`,
  `hidden_size`, `max_position_embeddings`, head counts, and
  `vocab_size` — every one present in this `config.json`
  [S4]), and its tensor-shape-inference path already
  recognises the llama naming family
  (`model.layers.N.*`, `model.embed_tokens.weight`) that
  this checkpoint uses. No other candidate family the loop
  checked has any recognition in the codebase at all.
- **The seed pattern already fits it.** `si seed
  --by-reference` (SI-8) imports a real safetensors
  checkpoint storing zero tensor bytes; a single-file
  checkpoint is the simplest possible input to that pattern.
- **The delta format is tensor-name based.** The landed
  transformer delta machinery (`si/transformer_delta.py`)
  names targets by their base tensor names and prices
  LoRA-style factors against the base manifest; nothing in
  the format is miniature-shaped. The gate's *policies*
  (allowlist, caps) are bundle configuration, not format.

And the counterweight, stated with equal weight: **no
existing executor runs this base, and none is claimed to.**
The only forward path in the repo, `si/tiny_transformer.py`
(SI-15), refuses any architecture descriptor other than the
miniature's pinned one and any weight set that is not the
miniature's exact 30-tensor inventory. The gate's probe
checks evaluate miniature bundles only; a bundle seeded
from this checkpoint would today report checks 3/4 "not
evaluated" and escalate — the same honest state the Glimmer
seed is in. What the llama-family fit buys is *scope*: the
A-4 executor design (§5.3) covers one well-documented
decoder family — pre-norm RMSNorm blocks, RoPE, grouped-
query attention (15 query heads over 5 KV heads), a SwiGLU-
style SiLU MLP, and tied embeddings — instead of a family
with no foothold in the codebase. GQA and tied embeddings
are named here because they are real proof obligations for
A-4 (the miniature has neither), not details to discover
later.

### 3.4 Size: the CPU-plausibility argument, as arithmetic

Measured on this VM, this run: **2 vCPUs, 7.7 GiB total
RAM**, 200 GB disk with 120 GB free. Those are the
constraints; the miniature's forward discipline (SI-15:
weights delivered as float32, computation widened to
float64) is the cost model any honest executor design
starts from. Derived from the sourced parameter count
(361,821,120):

| Assembly | Derived bytes | Against 7.7 GiB RAM |
|---|---|---|
| BF16, as stored (one copy) | ≈ 0.72 GB (the sourced file size) | comfortable |
| float32 assembly (one copy) | 361,821,120 × 4 ≈ 1.45 GB | fits, with headroom |
| float32, base + candidate (what a verify step holds) | ≈ 2.90 GB | fits, with headroom |
| float64-widened compute copy (SI-15 discipline, one copy) | 361,821,120 × 8 ≈ 2.89 GB | fits alone; combined residency is an A-1/A-4 question |

Every other candidate that cleared the licence bar fails
this table or wastes it: the same family's 1.7B member
derives to ≈ 6.85 GB for a *single* float32 copy — before
any candidate copy, activations, or the operating system —
on a 7.7 GiB machine (§7). The 360M is the largest size in
its family at which the full measure → assemble → verify
shape of the existing loop remains arithmetically plausible
on the machine that actually exists, with the float64
question left open as a measured question for A-1 and a
proof obligation for A-4 rather than assumed away here.

Wall time is deliberately absent from this argument. A
32-layer, 960-wide decoder forward on 2 vCPUs will be slow
by any interactive standard; whether it is *tolerably* slow
for a measurement-and-verification loop — seconds, minutes,
or worse per forward — is precisely what A-1 exists to
measure. This proposal asserts plausibility of residency
(above) and asserts nothing about speed.

---

## 4. The GLM preference, honoured and checked

The operator prefers working with GLM (Zhipu) models. The
loop therefore checked the GLM line first, hardest, and
reports it in full. Conclusion: **no GLM checkpoint fits
all the bars at once today** — the small one fails the
licence bar and the memory table; the permissively licensed
one is not small. Evidence, all sourced 2026-10-28:

- **`zai-org/glm-edge-1.5b-chat`** (the repo formerly under
  `THUDM/`; Hugging Face redirects that id to `zai-org/`
  [S5]) is the only genuinely small GLM text checkpoint:
  1,593,427,968 BF16 parameters and a single
  `model.safetensors` of 3,186,882,328 bytes; `config.json`
  records `GlmForCausalLM` / `model_type: "glm"`, 28 layers,
  hidden 2,048 [S5][S6]. It fails twice. **Licence:** the
  repo metadata says `license: other`, and its LICENSE file
  is a custom GLM-Edge licence — a grant described in its
  own text as revocable and non-sublicensable, free for
  academic research, with commercial use conditional on
  completing a registration, plus "Built with GLM-Edge"
  display and derivative-naming mandates [S7]. That is not
  Apache-2.0 / MIT or similarly clear, and a base the whole
  real-base programme would stand on cannot carry a
  revocable licence. **Memory:** derived float32 assembly
  is 1,593,427,968 × 4 ≈ 6.37 GB for one copy — the 1.7B
  failure shape of §3.4, on the same 7.7 GiB machine. Its
  `glm` architecture also has no recognition anywhere in
  the codebase (§3.3), so it would widen A-4's scope at
  the same time.
- **`zai-org/glm-edge-4b-chat`**: 4,327,984,128 parameters,
  same `license: other` position [S8] — larger and no
  clearer.
- **`zai-org/GLM-4.7-Flash`** is the permissive case — repo
  metadata `license: mit` — at 31,221,488,576 parameters
  [S9]. That is Glimmer's feasibility class, not this
  proposal's (§6): a derived BF16 residency of ≈ 62 GB
  already exceeds this VM's RAM by an order of magnitude.

So the preference is answered with evidence, not skipped:
if a small GLM checkpoint under a clear permissive licence
appears, the naming question can be re-asked as its own
decision. On today's record, naming one would mean either
standing the programme on a revocable licence or naming a
base this VM cannot hold. The proposal names neither.

---

## 5. What it costs, in SI-36 §6's categories

RFC SI-36 §6 stated the real-base costs as four categories
and invented no figures. This section keeps the categories,
fills in only what is now sourced or derived, and leaves
the rest honestly open.

### 5.1 (i) Download

Only on a yes, and only these files (sizes sourced from the
repo blob listing [S1]):

| File | Bytes |
|---|---|
| `model.safetensors` | 723,674,912 |
| `tokenizer.json` | 2,104,556 |
| `vocab.json` | 800,662 |
| `merges.txt` | 466,391 |
| `config.json`, `generation_config.json`, `tokenizer_config.json`, `special_tokens_map.json` | 5,397 combined |
| **Total** | **≈ 727,051,918 bytes (≈ 727 MB), derived by summation** |

Not downloaded: the ONNX variants, training logs, and eval
artifacts that make up the rest of the files page's 5.02 GB
repo total [S3]. No other download is proposed anywhere in
the A-1 to A-4 sequence.

### 5.2 (i, continued) Storage under never-duplicate-bytes

The checkpoint directory holds the one stored copy
(≈ 727 MB against 120 GB free, measured on this VM this
run — disk is not a constraint at this size, and saying so
plainly is part of the cost honesty). The `.si` bundle
itself follows the SI-8 seed pattern: seeded by reference,
`stored_bytes: 0`, per-tensor references to the checkpoint
in place — the same discipline the Glimmer seed observes
against its 59,553,253,376 logical bytes. Whether
reference-v1 binding is *sound* for a base that later
slices would actually execute against — its tamper
analysis, i.e. what happens if the referenced file changes
under the bundle — is A-2's design question, not an
assumption this proposal makes. Until A-2 answers it, the
checkpoint is treated as frozen operator-held data, exactly
as the Glimmer checkpoint is today.

### 5.3 (ii) The design and measurement work, sequenced

A yes starts **only** the first slice. Each later slice is
its own PR, reviewed on its own evidence, and any of them
can return an answer that stops the line:

1. **A-1 — feasibility measurement (the slice a yes
   approves).** Download per §5.1; seed a bundle by
   reference; then measure, on this VM, CPU only: forward
   wall time for the llama family at this scale (which
   requires a measurement harness — building the *minimal*
   harness is part of the slice, and its numeric behaviour
   is reported, not presumed), assembled memory at each
   assembly width of §3.4's table, and verification cost
   (base + candidate assembly and comparison). The slice's
   deliverable is a feasibility note with real numbers —
   including if the numbers say *stop*.
2. **A-2 — `reference-v1` binding design.** How a bundle
   binds to a referenced base it executes against: content
   addressing, re-verification points, and the tamper
   analysis (referenced-file mutation, substitution, and
   partial reads), extending the SI-8 pattern's guarantees
   from store-and-verify to execute-and-verify.
3. **A-3 — capability-suite design for a trained base.**
   What behaviour is the floor, and how baselines are
   recorded when there is no construction to derive them
   from: externally authored probe inputs, recorded base
   outputs as the baseline of record, and the honesty rules
   for a suite whose expected values are *observations of
   the base*, not derivations.
4. **A-4 — executor design, llama family only.** A forward
   executor scoped to §2's configuration surface, with
   SI-15-grade numeric proof obligations: the float32 /
   float64 discipline decided against A-1's measurements,
   GQA and tied-embedding handling proven against reference
   outputs, and bit-exactness or stated-tolerance claims
   made only where a test asserts them.

No slice after A-1 is approved by this proposal's yes; the
sequence is stated so the operator can see the whole
staircase before stepping onto its first step.

### 5.4 (iii) The capability floor problem

Priced in SI-36 §6 and unchanged: a trained base's
baselines cannot be recorded the miniature's way. The
miniature's suites are construction-relative — expected
logits exist because SI-14's seed-time helper produced them
from the construction. For this checkpoint, the baseline of
record can only be the base's own measured outputs on
authored inputs, captured before any delta exists, under
A-3's design. That is a real methodological cost — the
floor is empirical, not derived — and it is the honest
price of asking questions about a mind the project did not
build. This proposal does not minimise it.

### 5.5 (iv) The CPU-only discipline's price

No GPU, no cloud, no spend — at 360M scale the discipline
costs *time*, not feasibility of residency (§3.4), and the
exchange rate is unmeasured until A-1. Every later design
(A-4 above all) must keep stating its numbers in this VM's
terms — 2 vCPUs, 7.7 GiB — not in the terms of a machine
the project does not have. If A-1's wall times say the
loop's measure-verify cadence is impracticable even at this
scale, that is a finding, and the proposal's author will
report it as one.

---

## 6. The Glimmer alternative, head-on

The obvious counter-proposal deserves its own section:
*Glimmer 30B is already here — why download anything?*

What is true in it: the Glimmer checkpoint is local
(~/workspace/glimmer-safetensors/, two safetensors shards)
and seeded by reference as `~/workspace/sentinel-glimmer.si/`
(SI-8): 1,436 tensors, 59,553,253,376 logical bytes,
`stored_bytes: 0`, full hash verification of the seed
measured at ~29 minutes in the SI-13 record. The marginal
download cost of "using Glimmer" is zero, and this proposal
does not pretend otherwise.

Why it does not answer the question:

- **There is no Glimmer forward path, and the record closed
  that door deliberately.** RFC SI-13 stated it in terms:
  a solution is *not* a forward pass for Glimmer; the
  runtime has no Glimmer forward path at any size; the seed
  remains store-and-verify. That position was carried,
  unaudited-away and unreversed, through SI-16, SI-17, and
  every successor RFC. The gate's standing behaviour for
  the Glimmer bundle is escalation with checks 3/4 "not
  evaluated" — honest, and unchanged by anything in the
  miniature line. This proposal does not reopen it.
- **The scale is a different feasibility class, by
  arithmetic, not taste.** Glimmer's text configuration
  (recorded in the seed bundle's manifest) is hidden 6,656
  across a multimodal conditional-generation architecture
  (`MuseGlimmerForConditionalGeneration`) — 59.55 GB of
  BF16 weights against this VM's 7.7 GiB of RAM. Even the
  *seed verification* — hashing, not forwarding — took
  ~29 minutes. A CPU forward-and-verify loop over that base
  on this machine is not a slower version of §3.4's table;
  it does not fit in the table.
- **The preconditions do not shrink for a local base.**
  SI-36 §6 recorded it explicitly: the local presence of
  the Glimmer checkpoint retires none of A-1 to A-4. A-1
  for Glimmer would still need a named-family executor
  design for a multimodal architecture, A-3 a capability
  floor for it, and A-4 proof obligations at 30B scale —
  the same staircase, climbed in a building the VM cannot
  hold.

Glimmer stays exactly what it is: the project's first real
seed, held by reference at zero stored bytes, awaiting
hardware and designs that do not exist yet. This proposal
is about the base the project *can* measure now. Choosing
the small base is not a judgement against Glimmer; it is a
judgement about which question can be answered honestly
first.

---

## 7. Rejected alternatives, for the record

One line each; none is proposed, and a yes in §8 approves
none of them.

- **`HuggingFaceTB/SmolLM2-1.7B-Instruct`** — same family,
  same Apache-2.0 licence, 1,711,376,384 parameters and a
  3,422,777,952-byte weights file [S10]: rejected on §3.4's
  arithmetic — a derived ≈ 6.85 GB single float32 assembly
  leaves no verify headroom on a 7.7 GiB VM.
- **`HuggingFaceTB/SmolLM2-135M-Instruct`** — same family
  and licence, 134,515,008 parameters [S11]: rejected on
  judgement, stated as judgement — the 360M fits the same
  budget with room to spare, and a capability floor (A-3)
  is more meaningful against the stronger trained base;
  nothing technical forbids the 135M.
- **`meta-llama/Llama-3.2-1B-Instruct`** — llama family,
  1B class, single safetensors file [S12]: rejected on
  licence — its repo metadata records the custom
  `llama3.2` licence, not Apache-2.0 / MIT.
- **`zai-org/glm-edge-1.5b-chat` / `glm-edge-4b-chat`** —
  rejected in §4: custom revocable licence (`license:
  other`) at both sizes, and the 1.5B already fails the
  memory table.
- **`zai-org/GLM-4.7-Flash`** — MIT-licensed but
  31,221,488,576 parameters [S9]: Glimmer's feasibility
  class (§6), not a small base.
- **Qwen models** — excluded by the operator's standing
  preference; not evaluated, and no claim about any specific
  Qwen checkpoint is made or needed.

---

## 8. The ask

One question, one answer:

> **Do you approve `HuggingFaceTB/SmolLM2-360M-Instruct`
> as the named candidate base for the A-1
> feasibility-measurement slice — download of ≈ 727 MB
> (§5.1) and measurement only (§5.3, slice 1) — yes or
> no?**

Nothing in this document is measured, admitted, feasible,
or approved until that answer is yes and the A-1 slice
returns real numbers. If the answer is no, the loop records
the no, keeps the miniature line and the Glimmer seed as
they are, and does not re-ask with a different name dressed
as a different question.

---

## 9. Sources

All fetched 2026-10-28, from Hugging Face web pages and the
Hugging Face API. No weights were downloaded; the largest
object fetched to write this proposal was a model-card page.

- [S1] `https://huggingface.co/api/models/HuggingFaceTB/SmolLM2-360M-Instruct?blobs=true`
  — licence metadata, safetensors parameter total
  (361,821,120 BF16), and per-file blob sizes including
  `model.safetensors` at 723,674,912 bytes.
- [S2] `https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct`
  — the model card: family sizes, 4-trillion-token training
  of the 360M, SFT + DPO instruct pipeline, "License:
  Apache 2.0".
- [S3] `https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/tree/main`
  — the files page: 5.02 GB repo total; `model.safetensors`
  displayed at 724 MB.
- [S4] `https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/raw/main/config.json`
  — `LlamaForCausalLM`, `model_type: "llama"`, 32 layers,
  hidden 960, intermediate 2,560, 15 / 5 heads, vocab
  49,152, tied embeddings, 8,192 context, bfloat16.
- [S5] `https://huggingface.co/api/models/zai-org/glm-edge-1.5b-chat?blobs=true`
  (the `THUDM/glm-edge-1.5b-chat` API path redirects here)
  — 1,593,427,968 BF16 parameters, `model.safetensors`
  3,186,882,328 bytes, `license: other`.
- [S6] `https://huggingface.co/zai-org/glm-edge-1.5b-chat/raw/main/config.json`
  — `GlmForCausalLM`, `model_type: "glm"`, 28 layers,
  hidden 2,048.
- [S7] `https://huggingface.co/zai-org/glm-edge-1.5b-chat/raw/main/LICENSE`
  — the GLM-Edge licence text: revocable,
  non-sublicensable grant; academic use free; commercial
  use conditional on registration; "Built with GLM-Edge"
  display and naming mandates.
- [S8] `https://huggingface.co/api/models/zai-org/glm-edge-4b-chat?blobs=true`
  — 4,327,984,128 parameters, `license: other`.
- [S9] `https://huggingface.co/api/models/zai-org/GLM-4.7-Flash?blobs=true`
  — 31,221,488,576 parameters, `license: mit`.
- [S10] `https://huggingface.co/api/models/HuggingFaceTB/SmolLM2-1.7B-Instruct?blobs=true`
  and its raw `config.json` — 1,711,376,384 parameters,
  `model.safetensors` 3,422,777,952 bytes, Apache-2.0,
  llama family.
- [S11] `https://huggingface.co/api/models/HuggingFaceTB/SmolLM2-135M-Instruct?blobs=true`
  — 134,515,008 parameters, Apache-2.0.
- [S12] `https://huggingface.co/api/models/meta-llama/Llama-3.2-1B-Instruct`
  — repo metadata recording `license: llama3.2` and a
  single `model.safetensors`.

In-repo sources: RFC SI-18 §3.7 (preconditions A-1 to A-4);
RFC SI-36 §6 (Route 3, cost categories, the local-Glimmer
point); RFC SI-13 (no Glimmer forward path; ~29-minute seed
verification; 1,436 tensors, `stored_bytes: 0`); SI-8 /
SI-9 §0 (the by-reference seed and its 59,553,253,376
logical bytes); SI-15 (`si/tiny_transformer.py`'s scope and
float32/float64 discipline); `si/arch.py` (checkpoint-config
derivation and llama-family tensor inference); the Glimmer
seed bundle's manifest on disk (text configuration, hidden
6,656, `MuseGlimmerForConditionalGeneration`).
