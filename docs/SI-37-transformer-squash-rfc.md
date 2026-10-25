# SI-37 RFC — Transformer Squash for the Landed Mixed-Target Stack
(Design Only)

> **Status: DESIGN ONLY.** This document argues the three
> design obligations RFC SI-34 §8 recorded for transformer
> squash and carried by RFC SI-36 §§11–12 — (i) the carrying
> question, (ii) the four-axis squash gate context, and
> (iii) the tolerance verify over named tensors — against
> the one authored transformer stack that exists. It
> changes no production code, no test, no manifest schema,
> no policy default, and no gate behaviour. It performs no
> squash, designs no implementation that a successor could
> execute without its own RFC-stage argument (§12), and
> states an implementation trigger that is not self-firing
> (§6). Nothing in this RFC is implemented unless a section
> explicitly says it exists today, in code, on `main`.
>
> **The one sentence that governs everything below:** the
> landed authored stack — `d0001` on `lm_head.weight` plus
> `d0002` on
> `model.language_model.layers.1.mlp.down_proj.weight` —
> is two lawful single-target deltas in sequence, and a
> squash of it would have to carry **both tensors' folded
> updates in one composite record**, which is two-target
> in its carrying whatever class the record is assigned
> to; this RFC **decides** that carrying question first
> (§3: a squashed composite is a **distinct record class**
> with its own pricing discipline, designed in §3.8, and
> is not an authored candidate for the standing budgets
> `max_targets_per_delta` 1 / `allow_additive` false,
> which continue to price authored candidates unchanged),
> then designs the gate context that would judge such a
> composite (§4) and the tolerance verify that would
> precede any judgement (§5) — and performs none of it:
> `si squash` continues to fail closed on transformer
> stacks under SI-12, throughout and after this RFC.
>
> **Scope discipline:** CPU only. No GPU, no cloud,
> no spend, no downloads. Terminology:
> machine learning infrastructure throughout.
>
> **Relationship to prior work:** RFC SI-34 §8 deferred
> transformer squash with a two-arm trigger — (a) depth,
> or (b) a future RFC arguing obligations (i)–(iii) as its
> main business, the carrying question first. RFC SI-36
> §4 evaluated both arms against SI-35's evidence: trigger
> (a) has **not fired** (depth is still 2 against `keep`
> 3, and SI-35's R2 makes a third authored delta less
> likely, not more), and trigger (b) is the route SI-36
> took in recommending this slice (§§11–12 of that RFC are
> this RFC's commission). This RFC is the (b) argument.
> It is, per RFC SI-36 §10 and §16, **the last autonomous
> slice the present record argues for**: it registers no
> SI-38, and what follows it is an implementation trigger
> firing (§6) or the operator's decision (RFC SI-36 §6),
> not a further cadence slice. SI-12's fail-closed refusal
> is the standing behaviour this RFC designs *around*,
> never against: the refusal stands until an
> implementation slice, separately argued and separately
> triggered, replaces it.

---

## Table of contents

1. [Where SI stands — the landed stack and the refusal](#1-where-si-stands--the-landed-stack-and-the-refusal)
2. [Inherited invariants](#2-inherited-invariants)
3. [The carrying question — argued and decided](#3-the-carrying-question--argued-and-decided)
4. [The gate context — designed to the four recorded axes](#4-the-gate-context--designed-to-the-four-recorded-axes)
5. [The tolerance verify over named tensors — designed as a measurement](#5-the-tolerance-verify-over-named-tensors--designed-as-a-measurement)
6. [The implementation trigger — stated precisely, not self-firing](#6-the-implementation-trigger--stated-precisely-not-self-firing)
7. [Decision summary](#7-decision-summary)
8. [Acceptance — this RFC against RFC SI-36 §12](#8-acceptance--this-rfc-against-rfc-si-36-12)
9. [Non-goals](#9-non-goals)
10. [Honesty and labelling requirements](#10-honesty-and-labelling-requirements)
11. [Risks and open questions](#11-risks-and-open-questions)
12. [Summary](#12-summary)

---

## 1. Where SI stands — the landed stack and the refusal

Everything in this section is verifiable in the code and
docs on `main` at the SI-36 merge (main tip `18169fb`,
the merge of PR #41). Numbers are cited to the as-built
document that measured them; this RFC measures nothing
new, performs no squash, and runs no verify. Where a
statement below is a *derivation* from cited facts rather
than a measurement, it is labelled a derivation, exactly
as RFC SI-36 §1 labelled its arithmetic.

**The subject, unchanged.** The SI-14 miniature
(`TinyTransformerMind`): a constructed, deterministically
seeded, **not trained** decoder-only transformer — 30
tensors, 23,456 elements. Architecture pinned: vocabulary
32, `d_model` 32, 2 layers, 4 heads, head dimension 8,
MLP hidden 64, maximum sequence 16
(`si/tiny_seed.py::ARCHITECTURE`). Probe suites: 8
identity + 16 capability (echo-task) entries; recorded
base echo accuracy 16/16 = 1.0, echo floor 0.9375; bundle
tolerances both 0.25 (`identity_tol`, `capability_eps`).

**The landed authored stack, exactly.** Depth 2, both
members admitted by the unchanged gate under SI-17's
live probe verdicts and §6.6 probe VERIFY, and unwound
and re-measured by test (SI-33 as-built):

| | `d0001` (cycle 1) | `d0002` (cycle 2) |
|---|---|---|
| Signal (recorded selection) | `tiny-successor-lean-001` | `tiny-predecessor-lean-001` |
| Anchoring (recorded selection) | `weights-under-training` | `base-anchored` |
| Composition configuration | `none` | `none` |
| Authorship target (recorded selection) | `lm-head` | `final-mlp-down` |
| Tensor acted on | `lm_head.weight` [32, 32] | `model.language_model.layers.1.mlp.down_proj.weight` [32, 64] |
| Head trained against | `base` | `d0001` |
| Candidate blob | `sha256:2761e8ff…6ca9e` (2,360 B) | `sha256:d7af7f1f…e625` (3,416 B) |
| Rank / alpha (per target) | 8 / 8 | 8 / 8 |
| Candidate norm | 0.23555380990005192 | 0.30801411790830574 |
| Gate verdict | ADMIT, VERIFY pass | ADMIT, VERIFY pass |

Full candidate blob hashes (SI-33 as-built): cycle 1
`sha256:2761e8ffc139ce3e107c11da035e8a73c20792945a494f4e493dccf418b6ca9e`;
cycle 2
`sha256:d7af7f1feaa5b0451d441a4fce8842b471fc7c4b3e70e6237f1115eba877e625`.
The second delta is labelled a non-readout-target
candidate wherever the stack is described (RFC SI-32 §18
item 24). SI-33's cycle-2 ADMIT is carried in this RFC
only with its attribution attached, as RFC SI-36 §14
item 26 requires: candidate norm 0.30801411790830574,
probe-space movement ‖m2‖ = 0.010469039280455053, F3′′′
**measured** (the destination traded away: held-out
residual 0.0027387202915082178 against the boundary
2.087217177504918e-05, ~131×), attributed by SI-33 to the
near-inertness of the branch it acts through (the final
layer's MLP branch ≈ 0.4% of the residual stream on the
measured population; SI-35 measured that branch's max
ratio on the primary cell as 0.00347083).

**The four recorded axes, as they stand.**
`AUTHORSHIP_TARGETS` serves exactly `"lm-head"` (the
default) and `"final-mlp-down"`; `TARGET_ANCHORINGS`
serves exactly `"weights-under-training"` and
`"base-anchored"`; `COMPOSITION_CONFIGS` serves exactly
`"none"`, `"movement-composition"`,
`"worst-input-composition"`, and
`"train-partition-projection"`; the signal registry
serves exactly `tiny-successor-lean-001` and
`tiny-predecessor-lean-001` (both spent in the landed
stack). There is no autonomous, learned, depth-driven,
verdict-driven, or probe-driven selection on any axis.
The landed stack exercises **two** values on the signal
axis, **two** on the anchoring axis, **one** on the
composition axis (`"none"` twice), and **two** on the
authorship-target axis. It contains **no constructed
member**: SI-31's constructed candidate (composition
`"train-partition-projection"`, rank 16) was REJECTED on
two independent lines and never landed, so no landed
stack anywhere contains one. §4 designs for the
constructed case prospectively, because RFC SI-32 §13.1
recorded it as an obligation, not because any landed
member needs it.

**The standing policy the carrying question is argued
against** (`si/bundle.py::DEFAULT_POLICY`, unchanged by
every slice since SI-9b pinned the transformer keys):
`max_targets_per_delta` 1, `allow_additive` false,
`rank_cap` 8, `norm_cap` 1.0, `per_target_norm_cap` 1.0,
`delta_byte_cap` 4 MiB (4,194,304 bytes),
`max_stack_depth` 8, `identity_tol` 0.25,
`capability_eps` 0.25. The squash entry point's own
defaults: `keep` 3, tolerance verify tolerance 1e-4
(`si/squash.py::squash_bundle`).

**The refusal that frames this RFC.** SI-12 made
`si squash` fail closed on any resolved active chain
containing a `transformer-delta-v1` record — in the
squash segment S, in the kept tail K, or both — raising
`SquashError` before any blob is written, any gate or
journal activity, or any manifest change, and refusing
to convert a transformer delta to a demo delta. Two
standing semantics of that entry point matter to every
design below and are recorded here so they are met as
facts, not rediscovered as surprises:

- **The `L <= keep` no-op precedes the refusal.** When
  the active stack is no deeper than `keep`,
  `squash_bundle` returns `{"squashed": False}` before
  every refusal, for transformer and demo stacks alike.
  At depth 2 against `keep` 3, a squash of the landed
  stack would not run even if transformer squash were
  implemented tomorrow exactly as designed here. This
  RFC designs machinery for a depth the project has not
  reached; §6 prices that honestly.
- **Demo squash is the only squash that exists.** It
  composes same-tensor demo updates by truncated SVD
  into one single-target delta, verifies the squashed
  head against the pre-squash head in weight space
  (max abs diff, tolerance 1e-4, fail closed, candidate
  blob deleted again on violation unless it pre-existed),
  and gates the result through `SquashContext` under the
  full five checks with post-squash depth accounting
  (§4.1 states that shape in full). Every use of demo
  squash below as a precedent is labelled a precedent;
  no demo behaviour is claimed for transformer stacks.

**What SI-35 changed, and what it did not.** SI-35's
pre-registered readings came back **R1 NOT measured,
R2 MEASURED, R3 NOT measured**: no in-stream non-readout
matrix satisfies R1's pair on the primary cell (signal
2's train partition at head `d0001`) — the branch-ratio
clause (≥ 0.04) fails for all 14 matrices, the largest
measured branch ratio being the final layer's attention
branch's 0.0340859, and the bound clause (≥ 0.212) fails
for exactly the two down-projections (0.0575995 and
0.0576504) — so the single-target upstream-authorship
line stands *expressibility-closed on this miniature at
the standing budgets*, quoted here only with its
qualifiers: **single-target, upstream, rank-8,
budget-0.9, this miniature, this signal family's
correction scale, derivation-scoped** (RFC SI-36 §14
item 28). R2 is a conclusion about what future
single-target candidates can express. It measures
nothing about how the two landed deltas would be
carried, gated, or verified as a composite — which is
why RFC SI-36 §4 found squash's trigger (b) untouched by
it, and why nothing in §§3–5 below cites R2 as evidence
about squash in either direction, beyond the one
depth-projection use §6 makes of it, at its stated
scope. SI-35 is a measurement-only record, not a verdict
record (RFC SI-36 §14 item 27): no figure from it is
quoted in this RFC as a gate figure or a verdict
prediction.

**The Glimmer seed is untouched.** By-reference
(`stored_bytes` 0 against 59,553,374,040 deduplicated
bytes); Glimmer-scale deltas still escalate with checks
3/4 "not evaluated"; no forward path exists and none is
faked, designed, or implied anywhere below (§4.5 records
what the designed context does at that scale: escalate,
which is a designed behaviour, not a designed forward
path).

---

## 2. Inherited invariants

Not negotiable inside this RFC (SI-9 §1.3, RFC SI-13 §2,
and the successor restatements through RFC SI-36 §2,
restated for the designs below):

1. **Base is frozen forever.** All change is a stack of
   deltas; rollback is moving `head`. A squash, if one is
   ever implemented, rewrites a stack's *representation*;
   it never edits a base tensor and never moves the head
   id (demo squash's standing semantics, §4.7).
2. **Never duplicate bytes** (Mark's design rule). A blob
   whose hash exists is never rewritten or copied; seed
   imports by reference where the pattern applies
   (`stored_bytes: 0` — the SI-8 Glimmer seed stores zero
   tensor bytes against 59,553,374,040 deduplicated
   bytes).
3. **No ungated write.** Authorship produces
   *candidates*. The gate — and only the gate — admits,
   rejects, or escalates them, and SI-17's verdict
   mapping and blind-set guard apply to authored
   candidates exactly as to fixtures, at every cycle, for
   every signal, under every anchoring, under every
   composition configuration, and under every authorship
   target. §4's design holds a squash composite to the
   same rule: a composite is a candidate, the gate judges
   it, and no verify result (§5) is a verdict.
4. **Honest labels are load-bearing.** "Not evaluated"
   stays distinguishable from a measured 0.0; a training
   loss stays distinguishable from a gate result; a
   number measured at one head stays labelled with that
   head; incremental and cumulative figures are never
   presented as each other; a number recorded under one
   anchoring, composition configuration, or authorship
   target is never presented as a number under another;
   a worst-case figure names the population it is a worst
   case *over*; a constructed candidate stays labelled
   constructed; a candidate that fails on two independent
   gate lines is reported failing on both; **an ADMIT's
   measured attribution travels with the ADMIT wherever
   the ADMIT is summarised (RFC SI-36 §14 item 26); a
   measurement-only record is not a verdict record (item
   27); and R2's derivation-scoped qualifiers travel with
   R2 wherever R2 is quoted (item 28).** §10 restates the
   items that bind this RFC's own text.
5. **Fail closed.** A missing, mismatched, or
   un-runnable measurement is "not evaluated" — never a
   pass, never a silent skip. Every design below inherits
   this in its failure semantics explicitly (§§4.7, 5.6).
6. **CPU only, existing dependencies only.** The
   runtime's dependencies are `numpy` and `safetensors`.
   No GPU, no network, no downloads, no spend, no new
   heavy dependency. Every design below is a design for
   that runtime and no other.
7. **Determinism where tests need it.** Seeded RNG,
   fixed inputs, no sampling, no wall-clock dependence
   in any measured or trained value — and deterministic
   *serialisation* wherever a blob hash is content
   addressing. A composite's serialisation, if one is
   ever built, inherits this in full (§3.8).
8. **No tolerance — and no cap — is a design variable.**
   No design in this RFC proposes moving `identity_tol`,
   `capability_eps`, `rank_cap`, the standing norm budget
   0.9, any other cap, or any policy default to make an
   outcome pass — or to make an outcome fail. In
   particular, §3's decision does **not** move
   `max_targets_per_delta` or `allow_additive`: both stay
   1 and false, pricing authored candidates, in code and
   in policy, throughout and after this RFC. What §3
   decides is a *design position* about which discipline
   a future implementation would price a composite
   under; enacting any part of it is an implementation
   slice's work, under §6's trigger, with its own
   RFC-stage argument (§12).
9. **The probes are the judge, never the training
   set.** No design in this RFC may train on, penalise
   against, project against, parameterise against, or
   otherwise shape anything using any probe input, probe
   logit, or probe-derived quantity. §5 applies this
   invariant to the verify's design in its strictest
   form: the verify's populations are allowed inputs
   only, and no probe-derived quantity — including any
   figure from SI-33's or SI-35's probe-side records —
   enters the verify's tolerance, statistic, or failure
   semantics.

---

## 3. The carrying question — argued and decided

**Verdict up front: DECIDED — reading (b). A squashed
composite is a distinct record class with its own pricing
discipline (§3.8), designed here and enacted nowhere in
this RFC. It is not an authored candidate for the
standing budgets, and it is not held inadmissible pending
a separate policy argument: the policy argument is this
section, made as RFC SI-36 §11 item 1 commissioned it —
as this document's main business, before the gate-context
and verify designs that depend on it. The standing
authored-candidate budgets (`max_targets_per_delta` 1,
`allow_additive` false) are unchanged, in code and in
policy, and continue to price every authored candidate.
Operationally, nothing changes until §6's trigger fires
and an implementation slice enacts the design: SI-12's
fail-closed refusal stands throughout, which is the same
observable state reading (c) would produce — reached
here by decision, not by deferral (§3.5).**

This section is placed first because §§4–5 depend on
its answer in kind, not in detail: a gate context cannot
be designed until it is known what record it judges, and
a verify cannot be designed until it is known what
carrying it verifies. RFC SI-34 §8 recorded the question
and RFC SI-36 §11 item 1 assigned its argument here.
Neither compressed it, and this section does not either:
all three recorded readings are stated at full strength
(§§3.3–3.5), priced against the landed stack's real
numbers (§3.2), and the chosen reading is defended
against the objections the record itself raised for its
shape (§3.6).

### 3.1 The question, precisely

A squash, in the only form the project has, takes the
oldest segment S of the active chain and replaces it
with one record whose folded effect equals S's combined
folded effect (exactly, under the carrying designed in
§3.8; approximately only under a lossy recompression
this RFC records as undesigned, §11 item 2). For demo
stacks the segment's members all act on the one base
tensor, so the composite is a single-target delta and
the question this section argues never arises. For the
landed transformer stack it arises immediately and in
its sharpest form:

- `d0001`'s folded update acts on `lm_head.weight`
  [32, 32] and on no other tensor.
- `d0002`'s folded update acts on
  `model.language_model.layers.1.mlp.down_proj.weight`
  [32, 64] and on no other tensor.
- There is **no joint matrix** the two updates are
  updates *of*. They have different shapes, different
  positions in the forward path, and no shared factor
  space; a truncated SVD "of the combined update", in
  demo squash's sense, does not exist for this segment.
  Any composite record must therefore carry the two
  updates **per tensor**: a targets list with two
  entries, each entry carrying one tensor's update.

The format already prices none of this. A
`transformer-delta-v1` record's `targets` list is a list
by construction (`si/transformer_delta.py`): validation
iterates it, per-target norms are computed per entry,
and the record-level norm is the root-sum-square over
per-target norms (`aggregate_norm`). The *policy* prices
it: the gate's transformer safety suite refuses a record
whose target count exceeds `max_targets_per_delta` and
refuses additive targets while `allow_additive` is
false. Both policy keys were written, in RFC SI-9 §4.5
and every restatement since, for **authored candidates**
— records a training (or construction) step proposes,
whose target count is a choice made before the candidate
exists. The carrying question is whether a record the
*loop itself* constructs, out of deltas already authored,
already judged, and already landed one at a time under
those budgets, is priced by the same keys in the same
way. Three readings of the record exist. Exactly three
were recorded (RFC SI-34 §8, RFC SI-36 §11 item 1):

- **(a) Authored candidate.** The composite is a delta
  like any other; the standing budgets price it as
  written. Two targets against a cap of 1 is a refusal.
- **(b) Distinct record class.** The composite is a
  record class of its own — a *squash composite* — with
  a pricing discipline designed for what it is: a
  representation rewrite of already-judged history,
  attested by its provenance, priced per tensor and in
  aggregate under §3.8, and judged by the same gate
  under the same tolerances (§4).
- **(c) Inadmissible until separately argued.** The
  composite has no lawful price today; none may be
  designed inside a squash RFC; a prior, separate policy
  argument must first decide what a composite is.

**The sentence no reading may evade** (RFC SI-36 §11
item 1, stated here as this section's ground rule): a
composite that carries two tensors' updates **is
two-target in its carrying, whatever class it is
assigned to**. Reading (b) does not make the composite
single-target, does not describe it as single-target
anywhere in this RFC, and does not borrow the
single-target budgets' authority for it. What (b)
claims is narrower and is argued on its own terms: that
the *cap* of 1 was written to price a choice, that a
composite's target count is not a choice, and that the
honest response is to price the composite under a
discipline written for it — in the open, with every
other budget still binding — rather than to pretend the
count is 1 or to leave the record permanently
unpriceable.

**The separation §8.1 of RFC SI-36 requires, preserved
verbatim in effect:** the landed stack is **not**
evidence about multi-target *authorship*. It is two
lawful single-target deltas, each authored in its own
cycle under the standing budgets and judged by the
unchanged gate in its own right. Nothing in this section
cites the stack's existence as a reason to loosen any
authorship policy, to author a multi-target candidate,
or to revisit RFC SI-36 §8.1's rejection of multi-target
authorship as a slice, which stands on its own policy
and evidence grounds and is carried, not reopened (§9).
The carrying question is about already-judged deltas
only. An authored two-target candidate proposed tomorrow
would be refused under the standing budgets exactly as
it would have been yesterday, and §3.8's discipline
gives it no route around that refusal: the class is
attested by a resolvable `squashed_from` provenance over
an active-chain segment (§3.6), which no authored
candidate can present.

### 3.2 What each reading prices — the landed stack, priced three ways

All figures below are the landed members' published
figures (SI-33 as-built, §1) or derivations from them,
labelled derivations. No blob for any composite exists;
no composite figure below is a measurement.

**The members' own prices (measurements, SI-33):**

| Price | `d0001` | `d0002` |
|---|---|---|
| Tensor | `lm_head.weight` [32, 32] | `…layers.1.mlp.down_proj.weight` [32, 64] |
| Per-target rank / alpha | 8 / 8 (scale 1) | 8 / 8 (scale 1) |
| Per-target norm (‖folded update‖_F) | 0.23555380990005192 | 0.30801411790830574 |
| Factor payload (derivation: (out + in) × rank float32) | (32 + 32) × 8 × 4 = 2,048 B | (32 + 64) × 8 × 4 = 3,072 B |
| Serialised blob (measurement) | 2,360 B | 3,416 B |

**The composite's prices (derivations from the row
above, under the exact carrying of §3.8 — each member's
factors carried verbatim, per tensor):**

- **Per-tensor folded updates:** two, unchanged. On
  `lm_head.weight`, exactly `d0001`'s folded update; on
  the down-projection, exactly `d0002`'s. Exactness here
  is a statement about the carrying's definition (§3.8),
  not a measurement: no composite has been built or
  folded.
- **Ranks:** 8 and 8, per tensor. Neither member's rank
  changes by being carried beside the other; there is no
  cross-tensor rank, because there is no cross-tensor
  matrix (§3.1).
- **Aggregate norm:** the format's own definition,
  root-sum-square over per-target norms:
  √(0.23555380990005192² + 0.30801411790830574²) =
  **0.38776061454106125** (*derivation*). The plain sum,
  0.5435679278083576, is **not** the format's aggregate
  and is recorded here only so no reader substitutes it.
  Both figures sit under `norm_cap` 1.0 and under
  `per_target_norm_cap` 1.0 per tensor — stated as
  arithmetic against standing caps, not as a gate
  result: no gate has evaluated any composite.
- **Byte size:** factor payload 2,048 + 3,072 =
  **5,120 B exactly** (*derivation* from the shapes and
  ranks above — payload is shapes × rank × 4 bytes, with
  no freedom in it). The serialised blob would add one
  safetensors header with one two-entry `targets`
  metadata list where the two landed blobs carry two
  headers with one entry each; its exact total is **not
  stated**, because no such blob exists and inventing its
  header length would be inventing a measurement. What
  is stated: the payload is exact, the total is payload
  plus one header, and under every reading the size is
  three orders of magnitude below `delta_byte_cap`
  4,194,304 B. Byte size decides nothing among the three
  readings on this stack; it is priced anyway, because a
  discipline that only prices what binds today is not a
  discipline (§3.8 prices it in general).

**What each reading then does with those prices:**

- **Reading (a)** applies the authored pricing whole:
  target count 2 > `max_targets_per_delta` 1 →
  **inadmissible**, on that line alone. Every other price
  passes as arithmetic (per-target ranks 8 ≤ `rank_cap`
  8; per-target norms ≤ 1.0; aggregate 0.38776061454106125
  ≤ 1.0; bytes ≪ cap) — and none of it matters, because
  the count line is sufficient and is reached first in
  the safety suite's own order of concerns. Under (a), a
  squash of the landed stack can never be admitted, and
  neither can a squash of **any** mixed-tensor segment of
  any future stack. Same-tensor segments are the only
  squashable ones, and then only if their per-tensor
  recompression satisfies the single-target pricing —
  a case that does not exist on any landed stack (§1).
- **Reading (b)** applies §3.8's discipline: target count
  priced as *the number of distinct tensors in the
  attested segment* (here 2, equal to the segment's
  distinct-tensor count by construction, §3.8 rule 1);
  per-target rank priced against the exact-carrying rank
  (here 8 per tensor, §3.8 rule 2); per-target norms and
  the root-sum-square aggregate priced against the
  **unchanged** caps (1.0 / 1.0); bytes priced against
  the **unchanged** byte cap. On this stack every priced
  line is within its bound *as arithmetic* — which, to
  say it once more, is not an admission, a prediction of
  one, or a gate figure of any kind (invariant 4;
  RFC SI-36 §14 item 27 applied by analogy: a pricing
  record is not a verdict record).
- **Reading (c)** applies no pricing: the composite is
  inadmissible not on any priced line but for want of a
  lawful price at all. Its cost is not a number above; it
  is the deferral itself, evaluated in §3.5.

### 3.3 Reading (a) at full strength — and why it is not chosen

The strongest case for (a) is real, and this RFC does
not pretend otherwise:

- **The policy text admits no exception on its face.**
  `max_targets_per_delta` names deltas, not "authored
  deltas". A composite is a delta record in the same
  format, judged by the same gate. Reading an unwritten
  class exception into a safety policy is exactly the
  move the project's labelling discipline exists to make
  visible — and (a) needs no such move.
- **One pricing is what makes verdicts comparable.**
  Every verdict in the record is readable because one
  judge priced every candidate under one discipline
  (RFC SI-32 §7.3's second-judge objection, §3.6). (a)
  preserves that comparability absolutely, by refusing
  to multiply disciplines at all.
- **(a) is the only reading that cannot be gamed by
  construction.** If composites are priced as authored,
  no sequence of operations — author, land, squash —
  can ever produce a record the authored pricing would
  have refused at authorship time. The invariant "(a)
  buys is total: the set of landable records is closed
  under the loop's own operations.
- **Its consequence is a decision, not an oversight.**
  Under (a), transformer squash of a mixed-tensor stack
  is not deferred; it is *refused, permanently, by
  policy*. A project that chose (a) knowingly would be
  choosing that squash exists only for same-tensor
  segments — and on the present record, where no
  same-tensor authored segment exists, that squash
  effectively does not exist at all.

Why (a) is nevertheless not chosen — two grounds, each
sufficient, neither of which denies the case above:

1. **(a) prices a choice where no choice was made.**
   `max_targets_per_delta` is a capacity budget: it
   exists so that an author — training or construction —
   cannot reach for more tensors than the gate is
   willing to price in one write. A squash composite's
   target count is not a reach. It is the cardinality of
   the set of tensors an already-landed segment happened
   to act on, fixed before the composite was proposed by
   verdicts the gate itself returned, one target at a
   time, each within the cap. Pricing that cardinality
   under the capacity budget does not conserve the
   budget's purpose; it conserves its text. RFC SI-32
   §7.2 made the identical distinction for rank — for a
   constructed candidate, rank "measures the
   construction, not the candidate's reach" — and §3.6
   answers the objection that the distinction was
   deferred there. The distinction itself was never
   found wrong; what SI-32 §7.4 found was that no verdict
   turned on it. Here, a verdict *would* turn on it:
   under (a) the carrying question is decided against
   every mixed stack in advance, forever, which is a
   considerably larger claim than the budget's text was
   ever argued to support.
2. **(a) collapses the commission.** RFC SI-36 §11
   items 2–3 commission a gate context and a verify
   design *for the landed mixed-target stack*. Under
   (a) both designs are moot for the only stack they
   were commissioned against — not deferred, moot — and
   SI-37 would be a document whose first section annuls
   its remaining sections. A reading that decides the
   question by deciding the subject away is available to
   any RFC at any time; SI-34 §8's trigger (b) was
   written for an RFC that argues the obligations, and
   an (a) verdict that never reaches them is the
   improvisation-by-refusal that trigger exists to
   prevent. This ground is about the commission, not the
   merits, and is recorded second for that reason.

### 3.4 Reading (b) at full strength

- **It prices what is actually there.** The composite's
  real risk surface is per-tensor and aggregate, and it
  is fully priced under (b): each tensor's update norm
  against the unchanged per-target cap, the
  root-sum-square aggregate against the unchanged norm
  cap, each tensor's rank at its true factorisation rank
  with its bytes counted, total bytes against the
  unchanged byte cap, and post-squash depth against the
  unchanged depth cap (§4.6). The one line (b) reprices
  is the count line, and it reprices it to a stated rule
  (§3.8 rule 1) rather than waiving it: the count must
  equal the attested segment's distinct-tensor count,
  which the gate can verify against the resolved chain —
  a composite claiming *fewer* targets than its segment
  touched, or *more*, fails. Nothing is unpriced under
  (b); one price is re-founded, in the open.
- **The project already prices composites differently
  from authored candidates — in demo squash, today.**
  A demo squashed delta is stored at the rank its
  truncated SVD produces, with `alpha = rank = r` so
  "the gate's rank-cap check applies to the true
  factorisation rank" (`si/squash.py`): rank follows the
  factorisation for a composite, while for an authored
  demo candidate rank is the author's capacity choice.
  Its provenance is `squashed_from`, its intent's
  `source_seqs` are its segment members' journal seqs,
  and its gate context reports a depth no authored
  context reports (§4.1). Demo squash is, in every way
  except the target-count line the demo format cannot
  raise, already a distinct record class with its own
  discipline, judged by the same gate. (b) extends that
  existing shape to the one line the transformer format
  does raise; it does not invent the shape.
- **The gate already prices by kind under one judge.**
  One `Gate` evaluates constitution, memory, rollback,
  demo-delta, and transformer-delta intents under
  kind-specific mappings of the same five checks
  (`si/gate.py`), including — for transformer deltas —
  an entire structural safety suite that demo deltas
  never see. Kind-specific pricing inside one gate, one
  verdict mapping, and one journal is the project's
  standing architecture, not a novelty (b) introduces.
  §3.6's second-judge answer rests on this.
- **It keeps the authored budgets at full strength.**
  Under (b), `max_targets_per_delta` 1 and
  `allow_additive` false continue to refuse every
  authored multi-target or additive candidate, with no
  exception, because the composite class is attested by
  provenance no authored candidate can present (§3.6).
  The loosening (b) is accused of is examined there and
  found to have no object: no authored candidate's price
  changes by one unit under (b).

### 3.5 Reading (c) at full strength — and why it is not chosen

- **The strongest case for (c)** is procedural, and it
  is strong: this RFC is design-only and changes no
  policy default (invariant 8; its own §9). A decision
  that a composite is a distinct class *is* a policy
  position, and there is a real argument that policy
  positions should be taken in a document whose business
  is policy, with the operator's attention on the policy
  change itself — not as §3 of a squash RFC whose later
  sections a reader reaches only after the decision has
  been spent. (c) also has the record's own caution
  behind it: RFC SI-32 §7.5 deferred its representation
  question to a future RFC "on its own, never as a rider
  on a candidate's slice", and (c) is that discipline
  applied to this question.
- **Why it is not chosen.** First, the commission:
  RFC SI-36 §11 item 1 assigns the carrying argument to
  *this* document "as the main business" and requires it
  "decided in the document". (c) is not a decision among
  the readings; it is a decision to have the decision
  elsewhere, and adopting it would fail the commission
  in the act of citing it. SI-32 §7.5's discipline is
  honoured, not breached, by arguing here: that trigger
  reserved the question for an RFC arguing it as its own
  business, and §3 is this RFC's own business — first,
  longest, and load-bearing for everything after it.
  Second, the cost: RFC SI-36 §15 item 4 named the
  failure mode (c) guarantees — design obligations
  quietly becoming permanent deferral. The carrying
  question has already been recorded unargued by RFC
  SI-30 §12.1, RFC SI-32 §13.1, RFC SI-34 §8, and RFC
  SI-36 §4 — four successive documents, each honestly.
  A fifth honest deferral is not more honest than a
  decision; it is the same deferral, and the record is
  now complete enough to decide on (§3.2 prices the real
  stack; §3.6 answers the recorded objections). What
  would justify (c) — an objection (b) cannot answer —
  is stated as (c)'s own retirement condition in §11
  item 3, so a future reader who finds such an objection
  knows exactly what it reopens.
- **What (c) gets right, kept.** Until an implementation
  enacts (b), the operational state *is* (c)'s: no
  composite is admissible, and SI-12's refusal is the
  whole of the law on transformer squash. This RFC's
  decision is a design decision — the position an
  implementation slice must enact or re-argue — not an
  enactment. The distance between "(b) decided" and "(b)
  in force" is §6's trigger plus an implementation
  slice, and §6 states that distance precisely so no
  reader closes it by citation.

### 3.6 The recorded objections, answered for reading (b)

RFC SI-36 §11 item 1 requires the chosen reading to be
argued **against** SI-32 §7.3's objections, not merely
cited beside them. Those objections were written against
re-pricing `rank_cap` for constructed candidates; this
section takes each at the strength SI-32 gave it,
transposed to the carrying question, and answers it —
including where the transposition *changes* the
objection's force, in either direction.

**Objection 1 — stack-growth: a waived budget spends the
stack's compression headroom.** SI-32 §7.3: `rank_cap`
"does not only price one delta; it prices the growth of
the whole authored stack"; a landed rank-16 delta
"occupies, in representation terms, the space of two
rank-8 deltas", and composing such a stack back down is
lossy — so the waiver spends, at admission, headroom
squash has never exercised. Transposed: a two-target
composite admitted outside `max_targets_per_delta`
occupies, in target terms, the space of two single-target
deltas — has the class spent the stack's room in the
same way?

*Answer.* The transposition reverses the direction of
the spend, and the reversal is arithmetic, not rhetoric.
SI-31's constructed candidate *grew* its stack: depth
+1, at twice the per-delta rank, before any compression
had ever been exercised on an authored stack. A squash
composite *shrinks* its stack by construction: it
replaces a segment of length |S| ≥ 2 (§4.7 refuses a
segment of 1 as no compression at all) with one record,
so post-squash depth is `L − |S| + 1 < L` — the operation
is the compression SI-32's objection was protecting
headroom *for*. Its byte price makes the same point in
the other unit: under §3.8's exact carrying, the
composite's factor payload per tensor equals the sum of
its members' payloads on that tensor (same shapes, ranks
carried verbatim — §3.2's 5,120 B is exactly 2,048 +
3,072), so the composite occupies no more factor bytes
than the segment it replaces, and strictly fewer
serialised bytes than the segment's blobs summed, by one
header's worth of overhead saved rather than spent (the
exact total is not stated, §3.2, because no blob exists;
the inequality's direction is a derivation from the
payload equality plus one-header-vs-two). What the class
does *not* do is create room for future growth: the
composite is one record against `max_stack_depth` 8
exactly as its members were |S| records, which is the
compression working, not headroom being spent. The
objection's underlying rule — *price the stack, not the
delta* — is adopted, not evaded: §3.8 prices the
composite's aggregate norm, total bytes, and post-squash
depth as stack quantities, under unchanged caps.

**Objection 2 — second judge: a class waiver creates two
safety suites selected by a label the candidate itself
carries.** SI-32 §7.3: "'judged on identity alone' does
not amend the rank line; it creates two safety suites —
the full suite for trained candidates, a reduced suite
for constructed ones — selected by a label the
candidate's own record carries", breaking the
comparability that makes the multi-run record readable
and inviting the next class and the next waiver.

*Answer, in three parts, the third being a concession
with its boundary stated.*

1. **The suite is not reduced.** Under §4, a composite
   faces the same five checks, the same probe suites,
   the same tolerances (`identity_tol`, `capability_eps`
   unmoved), the same verdict mapping including
   blind-escalation and not-evaluated escalation, and the
   same safety sub-checks per target — existence, shape,
   dtype, finiteness, per-target norm, allowlist,
   additive policy, structural validation — each applied
   per target, so a two-target composite is checked
   *more*, not less, than a single-target candidate on
   every per-target line. Exactly two pricings differ:
   the target-count line (§3.8 rule 1) and the
   per-target rank line's reference value (§3.8 rule 2),
   and both are re-foundations with stated rules, not
   waivers: each still fails a record that violates it.
   SI-32's objection was to *identity alone* — a suite
   with lines removed. No line is removed here.
2. **The selector is not a self-carried label.** The
   class is attested the way demo squash already attests
   its composites: by provenance the gate verifies
   against records the candidate does not control —
   parents equal to the segment's delta ids, `source_seqs`
   equal to those deltas' own journal seqs, a
   `squashed_from` provenance relation, and a segment
   that must resolve, in order, on the bundle's active
   chain at the depth the intent claims (§4.5, check 5).
   A record that merely *declares* itself a composite,
   without a resolvable segment and matching journal
   seqs, fails provenance and is priced as what it is:
   an authored candidate, under the standing budgets,
   where its target count refuses it. This is the
   class-definition-and-attestation answer SI-32 §7.5
   required of any future class argument — *what makes a
   record a member of the class, and who attests it* —
   given here in full: the journal attests it, against
   the manifest, neither of which the candidate writes.
3. **The concession.** Comparability is genuinely
   narrowed in one respect, and this RFC records it
   rather than arguing it away: a composite's ADMIT, if
   one ever occurs, will not be comparable to an authored
   ADMIT *on the count line*, because the two records
   were priced by different count rules. Every other line
   remains comparable, and §10's labelling items require
   a composite's verdict to be summarised with its class
   and its segment attached, so the narrowing is visible
   wherever the verdict travels. The invitation concern —
   each class argued from the candidate that needs it
   invites the next — is answered structurally: this
   class is defined by an *operation the loop performs*
   (squash of a landed segment), not by a property a
   candidate's author can claim, and §9 records that no
   further class is designed, implied, or licensed by
   this one. A future class would need its own operation,
   its own attestation, and its own RFC; none is
   foreseeable on the present record, and foreseeability
   is stated as the boundary of this answer, not as a
   proof.

**Objection 3 — timing: the question is argued
immediately after the budget first binds, which is the
gate-fitting shape.** SI-32 §7.3: a budget re-priced
after it binds, in a document whose author knows the
candidate it would have affected, "is indistinguishable
in form from repricing the identity tolerance after a
drift failure, however sound its arithmetic"; and
invariant 8 (RFC SI-36 §2) holds the same line in both
directions.

*Answer.* Three facts separate this argument from the
shape the objection names, and all three are checkable
against the record. First, **no verdict changes**: the
only candidates the count budget has ever refused are —
none. No authored multi-target candidate has ever been
proposed, trained, constructed, or judged; `d0001` and
`d0002` were each admitted under the standing budgets as
written, and §3's decision re-prices no record that
exists. The candidate the re-pricing "would have
affected", in SI-32's posture, does not exist — SI-31's
candidate, the standing example of a record a class
waiver was accused of rescuing, is not a composite, can
never become one (§3.1: it never landed, so it is on no
active chain and can be no segment's member), and its
verdict stands on both its lines permanently (§9).
Second, **the argument precedes its object**: SI-34 §8
recorded the question and SI-36 §11 commissioned its
argument *before* any composite existed or was proposed,
in documents that also recorded the depth trigger's
non-firing — the argument's timing is the cadence's
pre-registration, not a reaction to a refusal. Third,
**the decision spends nothing now**: no policy default
moves (invariant 8 is complied with, not argued around —
§3.8 is a design for an implementation slice to enact
under §6's trigger, at which point the enacting
document, not this one, carries the change and its
justification). What this RFC acknowledges the objection
gets right: the *form* is close enough to gate-fitting
that only the facts above separate them, which is why
they are stated as facts with citations rather than as
assurances.

### 3.7 What the decision does not do

Stated plainly, so no successor citation can widen it:

- It does **not** change any policy default, any code,
  or any gate behaviour. `max_targets_per_delta` remains
  1 and `allow_additive` remains false in
  `DEFAULT_POLICY`, in every bundle, and in the gate's
  transformer pricing of authored candidates.
- It does **not** admit, pre-admit, or predict the
  admission of any composite. No composite exists; §3.2's
  arithmetic is a pricing record, not a verdict record.
- It does **not** license multi-target or additive
  *authorship* in any form (§3.1's separation; §9).
  RFC SI-36 §8.1's rejection stands on its own grounds.
- It does **not** re-judge, soften, or re-describe any
  landed or refused record — not SI-33's ADMIT (carried
  only with its attribution, §1) and not SI-31's REJECT
  (standing on both lines, §3.6).
- It does **not** design lossy recompression. Demo
  squash's truncated SVD has no transformer analogue in
  this RFC: across tensors none exists (§3.1), and
  within a tensor (a segment with two members on the
  *same* tensor) it is recorded as undesigned with its
  retirement condition (§11 item 2). The carrying §3.8
  designs is exact.
- It does **not** make any composite admissible today.
  Admissibility requires §6's trigger and an
  implementation slice; until both, SI-12's refusal is
  the complete operational truth (§3.5).

### 3.8 The composite class's pricing discipline — designed

The discipline an implementation slice would enact,
stated as design rules. Each rule names what it prices,
against what bound, and what failure it produces. These
rules are a design, not code and not policy text: an
implementation slice must still argue its enactment of
each rule in its own RFC-stage document (§12), and any
rule it cannot enact as stated returns to open status
there rather than being silently weakened.

1. **Target count — priced against the segment.**
   A composite's target count must equal the number of
   distinct base tensors its attested segment's members
   act on, verified by the gate against the resolved
   active chain (not against the candidate's own list
   alone). Fewer targets than the segment touched means
   an update was dropped; more means one was invented;
   both fail the safety suite's count line. There is no
   numeric cap inherited from `max_targets_per_delta`
   (that key prices authored candidates only, unchanged),
   and there is no unboundedness either: the count is
   bounded by the segment's length, the segment's length
   by `max_stack_depth`, and each target is separately
   priced under rules 2–4. For the landed stack the count
   is 2 (§3.2).
2. **Per-target rank — priced at the true factorisation
   rank, against the exact-carrying rank.** Each target's
   carried factors must reproduce that tensor's combined
   folded update from the segment *exactly* (in the
   sense §5.2 makes precise: equality of folded weights
   in exact arithmetic, with all float discrepancy left
   to the verify). Where one member acts on a tensor,
   the exact-carrying rank is that member's rank (8 and
   8 on the landed stack). Where several members act on
   one tensor, the exact-carrying rank is at most the
   sum of their ranks (*derivation*: rank of a sum ≤ sum
   of ranks), and the carried rank must be stated and
   must be the true rank of the factors stored. The
   authored `rank_cap` 8 does not price a composite's
   per-target rank — that is rule 1's repricing applied
   to rank, for the §3.6 reasons — but rank is not
   thereby unpriced: it is priced in bytes under rule 4,
   exactly, since payload per target is
   (out + in) × rank × 4 bytes with no freedom in it.
   A composite whose author-of-record (the squash
   operation) truncates a target below its exact-carrying
   rank has left this design (§11 item 2).
3. **Norms — priced against the unchanged caps.**
   Per-target norms against `per_target_norm_cap` 1.0;
   the root-sum-square aggregate (the format's own
   `aggregate_norm`) against `norm_cap` 1.0. Neither cap
   moves, in either direction, for this class. On the
   landed stack: 0.23555380990005192 and
   0.30801411790830574 per target, 0.38776061454106125 in
   aggregate (*derivations* from SI-33's published
   per-target figures, §3.2) — arithmetic against the
   caps, not a gate result.
4. **Bytes — priced against the unchanged cap.**
   Serialised blob bytes against `delta_byte_cap`
   4,194,304 B, counted as serialised (header included),
   exactly as authored records are counted. Payload on
   the landed stack is 5,120 B (*derivation*, §3.2).
5. **Additive members — excluded, with the reason
   recorded.** A segment containing an additive target
   (a 1-D `d` factor) produces no composite under this
   design: the context refuses it, fail closed (§4.7).
   Reason: `allow_additive` false prices authored
   additive candidates, and whether a *carried* additive
   update — the verbatim vector sum of already-landed
   additive updates — stands in the same relation to that
   key as a carried LoRA update stands to the count key
   is a question this RFC does not need (no landed
   member, on any stack, is additive; both landed
   members are LoRA) and therefore does not argue.
   Recorded as an open case with its retirement
   condition (§11 item 2), never silently omitted
   (§8, criterion 3).
6. **Kind and serialisation.** The composite is a
   `transformer-delta-v1` record (the format already
   carries a targets list; no schema change is designed
   or needed) distinguished as a composite by its
   journal provenance (`squashed_from`, parents and
   `source_seqs` per §4.5 check 5), not by a new record
   kind and not by any self-declared flag the gate trusts
   without the provenance check. Serialisation is
   deterministic (invariant 7): targets in segment order
   (oldest member first; within a tensor, the tensor's
   first appearance in the segment), factors as stored,
   so the composite's blob hash is content addressing
   over a canonical form an implementation must pin and
   prove by hash equality in its tests, as every prior
   slice has.
7. **Depth — priced as a stack quantity.** Post-squash
   depth `L − |S| + 1` against `max_stack_depth` 8,
   accounted as §4.6 specifies. A squash under this
   discipline strictly reduces depth or does not happen
   (§4.7's segment rules).

---

## 4. The gate context — designed to the four recorded axes

**Verdict up front: DESIGNED — a transformer squash
context in the exact shape of demo squash's
`SquashContext` (probes from the mind, capability
baseline measured on the pre-squash stack,
`forward_candidate` running the post-squash stack,
`stack_depth` reporting the pre-candidate depth so the
gate's +1 is the true post-squash depth), extended to
represent a stack whose members carry the four recorded
axes, acted on different tensors, and may — prospectively
only — include a construction. All five checks are named
below with their transformer meaning (§4.5); none is
carried over by analogy without its meaning stated, and
the one case this design does not cover — a segment
containing an additive member — is recorded as an
exclusion with its reason (§3.8 rule 5, §4.7), never
silently omitted. Nothing in this section is code, and
no context object exists after this RFC that did not
exist before it.**

### 4.1 The shape inherited from demo squash, stated in full

Demo squash's gate context (`si/squash.py::SquashContext`)
is the only squash context the project has, and its shape
is the commission's stated template (RFC SI-36 §11
item 2). Its four commitments, quoted in substance from
the code on `main`:

- **Probes come from the mind.** `identity_probes()` and
  `capability_suite()` delegate to the mind object; the
  context invents no suite of its own.
- **The capability baseline is measured on the
  PRE-squash stack.** `capability_baseline()` returns
  errors measured by forwarding the pre-squash stack
  over the capability suite, so the capability check
  prices the squash against what it replaces, not
  against the seed.
- **`forward_candidate` runs the POST-squash stack.**
  The candidate *is* the squashed delta, already first
  in the post-squash stack (squashed segment replaced,
  kept tail untouched after it); the context forwards
  that stack.
- **Post-squash depth accounting.** The context is
  constructed with `depth = L − len(S)` — the
  pre-candidate depth, i.e. the kept tail's length — so
  the gate safety check's `ctx.stack_depth() + 1` is the
  true post-squash depth (§4.6 works the landed stack's
  arithmetic).

The pipeline around that context is also inherited as
shape, and §4.7 states the transformer pipeline in its
terms: split the resolved chain into S and K; build the
composite; **verify (§5) before anything is journaled or
committed**; gate through the context; journal REJECT /
ESCALATE exactly as the loop journals them (candidate
blob stays on disk as evidence); on ADMIT, journal the
self-edit with `squashed_from` provenance, mark every
squashed member `superseded_by`, append the composite,
checkpoint and re-anchor — with the head id deliberately
unmoved, because rollback, diff, and why semantics are
keyed on delta ids and the head is the newest delta id
(demo squash's stated rationale, adopted here unchanged).

### 4.2 The four recorded axes — what the context must represent

A transformer squash context judges a composite whose
members were each admitted under a recorded selection on
four axes. After a squash, the members are superseded:
their records remain in the manifest's history and the
journal, but the *active* stack a future reader, gate, or
squash resolves contains the composite where the members
stood. The context's first obligation is therefore
representational: **nothing the members' verdicts depended
on may become unrecoverable from the composite's own
record and its gate report.** The design:

- **Per-member axis record, carried whole.** The
  composite's gate-facing record carries, for each
  segment member, in segment order: the member's delta
  id; its signal id; its anchoring; its composition
  configuration (with its configuration's parameters
  where the configuration has them — λ for
  `"movement-composition"`, λ_w and k for
  `"worst-input-composition"`, α for
  `"train-partition-projection"` — as recorded in the
  member's own cycle record, null where the configuration
  has none, exactly as the landed members' `"none"`
  records carry them null); its authorship target; the
  tensor(s) it acts on; its per-target rank(s) and
  norm(s); its blob hash; and the head it was trained
  against. This is a *carrying* of records that already
  exist (each member's journaled cycle record carries
  every field named), not a new measurement and not a
  re-derivation: the context assembles it from the
  manifest and journal at squash time, and check 5
  (§4.5) fails the candidate if any member's record is
  incomplete. Incompleteness is fail-closed, never
  defaulted (invariant 5).
- **The landed stack, represented.** For S = [`d0001`,
  `d0002`], K = [] (the §5.7 counterfactual segment; a
  real squash at greater depth would carry a non-empty
  K untouched, §4.3): `d0001` — signal
  `tiny-successor-lean-001`, anchoring
  `weights-under-training`, composition `none`,
  authorship target `lm-head`, tensor `lm_head.weight`,
  rank 8, norm 0.23555380990005192, trained against
  `base`; `d0002` — signal `tiny-predecessor-lean-001`,
  anchoring `base-anchored`, composition `none`,
  authorship target `final-mlp-down`, tensor
  `model.language_model.layers.1.mlp.down_proj.weight`,
  rank 8, norm 0.30801411790830574, trained against
  `d0001`. The two members differ on three of the four
  axes; the context represents that difference as data.
  It does not average, merge, or vote across members on
  any axis: there is no composite signal, no composite
  anchoring, and no composite composition configuration,
  and any future text describing one is a labelling
  defect (§10).
- **No selection is made by the context.** Signal,
  anchoring, composition, and authorship target remain
  driver selections recorded per member (RFC SI-36 §14
  item 13). The squash operation itself selects only its
  segment and its `keep`, both driver-supplied through
  the existing entry point's arguments; the context
  records them, it does not choose them. There is no
  axis on which the designed context exercises discretion,
  and none on which a member's recorded value is
  re-interpreted: a member trained `"base-anchored"`
  enters the record as base-anchored even though the
  composite, once landed, will sit under later heads —
  head-relativity is preserved by carrying the head each
  member was trained against (§4.2, first bullet), not by
  rewriting the member's anchoring.

### 4.3 The mixed-tensor case — designed

The landed stack *is* the mixed-tensor case, so this is
the design's centre, not an edge:

- **The composite's targets list is the segment's
  distinct tensors, in segment order** (§3.8 rule 6):
  `lm_head.weight` (from `d0001`) then the
  down-projection (from `d0002`). The context's
  `forward_candidate` forwards the post-squash stack
  through the existing apply path (`si/apply.py`'s shared
  `fold_target`, oldest first per tensor), which already
  folds multi-target records per tensor by exact name —
  the forward path a composite needs exists today for
  authored records and is not re-designed here. What does
  not exist today, and what an implementation slice would
  build, is only the context object and the squash
  pipeline's transformer branch; §12 records that
  boundary so no successor mistakes the apply path's
  existence for the squash's.
- **Several members on one tensor** (no landed instance;
  designed because any deeper stack will present it):
  the tensor's carried update is the sum of its members'
  folded updates, in segment order, carried at the
  exact-carrying rank of §3.8 rule 2. The members remain
  individually represented in the per-member record
  (§4.2) — summation is a property of the *carrying*,
  never of the *record*: no member's axis values, norms,
  or provenance are merged into any other's.
- **The kept tail K** is represented exactly as demo
  squash represents it: untouched, in order, after the
  composite, forwarded as part of the post-squash stack,
  materialised for the verify (§5), and never priced as
  part of the candidate — its members were priced when
  they landed. A transformer record in K no longer refuses
  the squash by its presence under this design (that is
  SI-12's refusal, which stands until implementation);
  it is simply carried, as demo squash carries its tail.

### 4.4 The constructed-member case — designed, prospectively

No landed member is a construction (§1). The obligation
is prospective, recorded in RFC SI-32 §13.1 — a landed
stack's deltas "may now carry any combination of recorded
signal, anchoring, and any of four composition
configurations, one of whose values denotes a
construction" — and carried by RFC SI-34 §8. The design:

- **Representation.** A constructed member enters the
  per-member record (§4.2) with its composition
  configuration value verbatim — on the present record
  the only constructed value is
  `"train-partition-projection"` — with its construction
  parameter (α) as recorded, and with the label
  **constructed** attached to the member wherever the
  record is carried (RFC SI-36 §14 item 22). Its
  pre-construction intermediate's figures are not carried
  as the member's figures; if an implementation needs
  them, they live in the member's own cycle record, cited
  not copied.
- **Pricing.** The member's folded update is priced like
  any member's: by the tensor it acts on, at its true
  norm, inside the aggregate. Its rank enters §3.8
  rule 2's arithmetic as what it is — SI-31's candidate
  carried rank 16, so a segment containing such a member
  as sole actor on its tensor has an exact-carrying rank
  of 16 on that tensor, priced in bytes under rule 4 and
  recorded at its true value. The design does not cap
  that rank at 8 and does not hide it: the rank line's
  re-foundation (§3.6, objection 2) prices the true rank
  openly, and a reader who finds a rank-16 carried target
  in a future composite's record is reading the
  construction's arithmetic, labelled as such. Whether
  such a composite *should* be admissible is not prejudged
  here in either direction: the member hypothesised was
  REJECTED and can never be a member (§3.6), so the case
  is designed for a future constructed member that first
  lands lawfully under the unchanged authored pricing —
  whose own slice will have faced its rank under
  `rank_cap` as an authored candidate, a confrontation
  this RFC neither stages nor softens.
- **What is not designed in this case.** A constructed
  member whose construction *referenced another member of
  the same segment* (a projection against a segment-mate's
  movement, rather than against a landed predecessor as
  SI-31's did) has no record on file and no semantics in
  the current cycle machinery — constructions are defined
  against heads, not against segments. Recorded as
  undesigned with its retirement condition (§11 item 2),
  never silently omitted (§8, criterion 3).

### 4.5 The five checks — each named, with its transformer meaning

The gate's five checks (`si/gate.py`, in evaluation
order) are constitution, safety bounds, identity
continuity, capability floor, and provenance
completeness. Demo squash takes all five in their demo
mapping; transformer *authorship* takes them in the
SI-9b/SI-17 mapping (a structural safety suite, probe
checks live for miniature bundles under a verified
`ProbeContext`, escalation everywhere else). The designed
squash context takes the **transformer mapping**, with
the squash-specific meanings below. Each check is stated
with (i) its meaning for a squash composite, (ii) what
it prices on the landed stack, and (iii) its failure
consequence. No check is waived, merged, or reordered.

**Check 1 — constitution.** *Meaning, unchanged and
carried over with its meaning stated:* the intent carries
a non-empty reason and no policy-blocklisted pattern
appears in the intent's text. For a squash intent the
reason names the operation in the operation's own terms —
squash of a named segment at a named `keep`, in demo
squash's form ("squash N deltas into one (keep=…)") —
and the check prices exactly what it prices for every
intent: the stated cause's presence and lawfulness, never
its wisdom. A constitution intent still ESCALATES, as it
does for every kind. On the landed stack: a squash intent
for S = [`d0001`, `d0002`] passes or fails on its reason
text alone, as any intent does; no member's axis values
enter this check.

**Check 2 — safety bounds.** *Meaning: the transformer
structural suite, priced under §3.8's discipline.* The
SI-9b suite's sub-checks, each with its composite meaning
stated: `candidate_shape` (the candidate carries a
`transformer-delta-v1` record plus blob bytes, and — the
squash extension — a per-member axis record per §4.2);
`target_existence_identity` (each carried target names a
base tensor exactly and binds to its base blob hash —
for the composite, both landed tensors, exactly as for
an authored candidate); `shape_conformance`,
`dtype_conformance`, `finite_values` (per target, against
the base manifest and the blob header, unchanged in
meaning); `norm_budgets` (per-target norms and the
root-sum-square aggregate against the unchanged caps —
§3.8 rule 3; on the landed stack, 0.23555380990005192,
0.30801411790830574, aggregate 0.38776061454106125, as
arithmetic); `size_budgets` (per-target rank at its true
factorisation rank under §3.8 rule 2, target count under
§3.8 rule 1, serialised bytes under the unchanged byte
cap — this is the one sub-check whose *pricing rules*
differ from the authored mapping, and it differs exactly
as §3.8 states and nowhere else); `allowlist` (each
carried target matched against the bundle's
`allowed_target_patterns`, unchanged — a tensor the
bundle's policy does not allow does not become allowable
by having landed under a different bundle state, and if
the landed bundle's policy does not allowlist both landed
tensors, the composite fails here, fail-closed, until the
policy question is faced as a policy question);
`additive_policy` (no additive target is carried — §3.8
rule 5's exclusion is enforced here as a failure, not
handled as a case); `stack_depth` (post-squash depth
under §4.6 against `max_stack_depth` 8); and
`structural_validation` (SI-9a's `validate_transformer_delta`
as catch-all, unchanged — the suite can only be stricter
than format validation, never looser). Failure of any
sub-check fails the check, and the check is decisive:
failure is REJECT (§4.7's mapping).

**Check 3 — identity continuity.** *Meaning, for a
miniature bundle under a verified probe context: the
gate's own identity logic, run for real.* The post-squash
stack is forwarded over the identity probe suite (8
entries) and the max drift against the seed's expected
outputs must not exceed `identity_tol` 0.25 — the
seed-absolute, cumulative figure, in the gate's existing
`_err` form (max abs difference, float64 over the
forward's outputs), exactly as the landed members were
judged (their cumulative figures: 0.17189218290150166 at
`d0001`; 0.17195402272045612 at `d0002`). The squash
context supplies the post-squash stack to that logic the
way `ProbeContext` supplies a candidate overlay; the
baseline is the seed's, because identity continuity is a
property against the seed, not against the pre-squash
head — the pre-squash comparison is the verify's job
(§5), and confusing the two is the exact error §5.1
separates. **Blind-set guard, carried unchanged:** if the
composite's targets are exclusively probe-blind tensors
(the derived blind set is exactly both layers'
`k_proj.bias`), the candidate is blind-escalated, never
ADMIT-eligible — drift 0.0 on a blind target is vacuous,
not measured. The landed composite's targets are both
probe-visible, so the guard does not fire on this stack;
it is designed in because a context that omits it would
admit the one composite shape the authorship gate is
explicitly forbidden to. *At Glimmer scale, and for any
bundle without a verified miniature probe context, this
check has **no** autonomous meaning:* it is reported not
evaluated (`measured` null, the standing honest reason),
and the verdict mapping escalates (§4.7). That is a
designed behaviour — escalation — recorded here so it is
never mistaken for a designed forward path, of which
there is none (§9).

**Check 4 — capability floor.** *Meaning: two arms, as
the miniature authorship mapping already runs them, with
the baseline's source being the squash-specific point.*
Arm one, regression: the post-squash stack is forwarded
over the capability suite (16 echo entries) and the max
regression against the baseline must not exceed
`capability_eps` 0.25. The baseline is measured on the
**pre-squash stack** (demo `SquashContext`'s commitment,
§4.1) — not the seed's recorded baselines that an
authorship `ProbeContext` returns — because a squash
promises to preserve the behaviour of the stack it
replaces, and a regression priced against the seed would
let a squash inherit, silently, whatever regression the
segment's members had already accumulated. Arm two, echo
accuracy: the post-squash stack's echo accuracy over the
capability suite against the recorded floor 0.9375,
absolute, as in the authorship mapping (the landed heads
both measure 1.0, 16/16). Both arms' figures land in the
check's detail and sub-checks, per the existing labelling.
*At Glimmer scale:* not evaluated, escalate — the same
recorded absence as check 3, for the same reason.

**Check 5 — provenance completeness.** *Meaning: the
demo requirements plus the transformer extension plus
the squash attestation — the check §3.6's class
definition stands on.* Demo requirements, unchanged:
parents non-empty, `source_seqs` non-empty ("an uncaused
change is noise"; "the journal can't answer why this
exists"). Transformer extension, unchanged: the intent's
declared target list matches the candidate record's
targets exactly — for the composite, both named tensors,
in segment order. Squash attestation, the designed
extension: parents are exactly the segment members' delta
ids, in segment order; `source_seqs` are exactly those
members' own journal seqs (demo squash's provenance rule,
adopted verbatim — the causal events this squash exists
because of; inventing them would be lying); the
`squashed_from` relation will be journaled on ADMIT
(§4.7); and the per-member axis record (§4.2) is complete
for every member — a member whose signal, anchoring,
composition, target, or head cannot be recovered from the
manifest and journal fails this check, fail-closed,
because after the squash that record is the only place
the active stack remembers what it replaced. Failure is
decisive.

### 4.6 Post-squash depth accounting

Inherited from demo squash without alteration (§4.1):
the context reports `stack_depth()` as the
pre-candidate depth — the kept tail's length,
`L − |S|` — and the safety check's `+1` for the
candidate yields the true post-squash depth, priced
against `max_stack_depth` 8 inside check 2's
`stack_depth` sub-check and the gate's depth line.

Worked for the landed stack, as design arithmetic: a
full-segment squash (S = both members, K empty) reports
depth 0, and the gate's +1 gives post-squash depth **1**
— the stack compresses 2 → 1. At a hypothetical depth 4
with `keep` 3 (S = the oldest 1 member — refused as no
compression, §4.7 — or `keep` 2, S = 2, K = 2), the
context reports 2 and the post-squash depth is 3. The
general identity is `post = L − |S| + 1`, strictly less
than `L` whenever `|S| ≥ 2`, which §4.7 makes a
precondition. Rollback accounting is unaffected: the
head id does not move (§4.1), superseded members remain
in the manifest with `superseded_by` set, and moving
`head` back across the composite restores the pre-squash
ancestry exactly as demo squash's unwinds do — a property
an implementation must prove by test on the real stack
shape, as SI-33's unwind was proved, before any ADMIT
path it builds is trusted (§12).

### 4.7 The pipeline's failure and verdict semantics

Designed in demo squash's terms, fail-closed at every
step (invariant 5):

- **Preconditions, before any blob, journal, or manifest
  activity:** the bundle loads; the active chain resolves;
  `L > keep` and `|S| ≥ 2` (a segment of one is not a
  compression — demo squash at `|S|` = 1 would recompress
  a single delta to no purpose, and the transformer design
  refuses it as a no-op mislabelled as a squash rather
  than performing it); no segment member carries an
  additive target (§3.8 rule 5 — refusal, naming the
  member); every segment member's axis record is
  recoverable (§4.2). Each failure is a named refusal in
  SI-12's form — a `SquashError`-class refusal stating
  what was refused and why — and writes nothing anywhere.
  SI-12's transformer refusal itself remains the operative
  refusal until an implementation slice replaces this
  whole pipeline branch (§6, §12).
- **Verify before gate, before commit:** the §5 verify
  runs on the built composite against the pre-squash
  stack; a verify failure is a refusal (not a gate
  verdict), the candidate blob is deleted again unless
  the identical blob already existed (demo squash's
  blob-hygiene rule, adopted), and nothing is journaled —
  exactly the standing treatment of demo squash's
  tolerance refusals and of SI-12's refusals alike.
- **Verdict mapping, unchanged from SI-17's:** decisive
  checks (constitution, safety, provenance) fail →
  REJECT; identity and capability both measured and
  passed → ADMIT-eligible; either probe check not
  evaluated → ESCALATE; exclusively-blind targets →
  blind-ESCALATE. REJECT and ESCALATE are journaled
  exactly as the loop journals them (self-edit, verdict,
  report; result rejected/escalated; checkpoint and
  manifest re-anchor), and the candidate blob stays on
  disk as evidence, mirroring the loop's guarantee. No
  squash-specific verdict exists, and none is designed:
  a fourth verdict for composites would be the second
  judge §3.6 refused, in its purest form.
- **ADMIT commit:** journaled self-edit (ADMIT, gate
  report), `squashed_from` provenance with the segment as
  parents, the composite's delta record, result
  materialised; every segment member marked
  `superseded_by` the composite; the composite appended;
  manifest checkpointed and re-anchored; head id
  unmoved (§4.1).

### 4.8 What the context design does not do

- It does not run. No context class, pipeline branch, or
  CLI behaviour is created by this RFC; `SquashContext`
  remains demo-only in code, and the transformer refusal
  in `squash_bundle` remains exactly where SI-12 put it.
- It does not design a Glimmer context. Checks 3/4 at
  that scale are escalations (§4.5), and a context whose
  probe checks cannot be evaluated produces ESCALATE by
  the standing mapping — designed, honest, and not a
  forward path.
- It does not design the lossy case. A composite built
  by truncating any target below its exact-carrying rank
  is outside §3.8 and therefore outside this context:
  the context prices the ranks it is given at their true
  values, and §11 item 2 records the truncation design
  as open. An implementation that truncates and presents
  the result through this context without that open item
  retired has left the design, and its PR body must not
  cite this RFC as covering it.
- It does not re-open any axis. The context represents
  signal, anchoring, composition, and authorship target;
  it selects none, extends none, and adds no value to
  any of them (§9).

---

## 5. The tolerance verify over named tensors — designed as a measurement

**Verdict up front: DESIGNED — a two-part verify, run
before the gate and before any commit, whose weight-space
part checks the composite's folded weights against the
pre-squash stack's folded weights per named tensor, and
whose behavioural part forwards both stacks over
allowed-input populations only and compares logits, with
the pass/fail statistic, tolerance, and failure semantics
stated below. No probe input, probe logit, or
probe-derived quantity enters any part of this design
(invariant 9). The design states what the verify would
measure on the existing depth-2 stack if it were run
(§5.7), performs no squash and no verify, and claims no
verify result, no squash result, and no behavioural
equivalence as measured — anywhere, including in this
sentence's RFC.**

### 5.1 Two comparisons that must not be confused

The record contains two different "does the new stack
match" questions, and this design exists because a squash
needs the one the gate does not ask:

- **The gate's question** (checks 3/4, §4.5): does the
  post-squash stack still stand in the right relation to
  the **seed** (identity) and to the **pre-squash stack's
  capability errors** (regression)? Its tolerances are
  the standing judgement tolerances, 0.25 and 0.25 —
  deliberately loose, because they price *learning*.
- **The verify's question:** does the post-squash stack
  **compute the same function** as the pre-squash stack
  it replaces? A squash is a representation rewrite, not
  a learning step; the behaviour it may legitimately
  change is none. Its tolerance is therefore a *rounding*
  tolerance (§5.5), orders of magnitude tighter than any
  gate tolerance, and a verify pass is **not a verdict**:
  it precedes the gate (§4.7), it admits nothing, and a
  stack can pass the verify and still be REJECTED or
  ESCALATED by the gate on its own lines.

Demo squash already embodies the separation in weight
space: its tolerance verify compares materialised heads
(max abs diff over the base's tensors, tolerance 1e-4)
before the gate runs, and refuses — journaling nothing —
on violation. §5.2 explains why the transformer verify
needs a second, behavioural part that demo squash does
not, and §5.3–§5.6 design both parts.

### 5.2 The derivation: exact in the weights, measured in the behaviour

Two facts, one a derivation and one a property of the
model, decide this design's shape:

- **Per-tensor exactness is a derivation.** Under §3.8's
  carrying, each carried target's factors are the
  segment's factors for that tensor, verbatim (one member
  per landed tensor), or — for a tensor with several
  members — factors of their summed folded update at the
  exact-carrying rank. In exact arithmetic the composite's
  folded weight on every named tensor **equals** the
  pre-squash stack's folded weight on that tensor:
  base + Σ members' updates, summed in either grouping,
  is the same sum. Folded-weight equality is therefore
  not a hypothesis the verify tests; it is a property of
  the carrying's definition, and the weight-space part of
  the verify is a *rounding check* on that property
  (float32 factor storage, float64 fold arithmetic,
  serialisation round-trips), in demo squash's form.
- **Behavioural equivalence is not a derivation.** The
  two landed updates do not act independently in the
  forward path. `d0002`'s update acts on the final MLP's
  down-projection, whose output enters the residual stream
  and passes through the final RMSNorm; `d0001`'s update
  acts on the readout that consumes that normed stream.
  The readout is *shared downstream* of the
  down-projection: the composite's logits are a function
  of both folded weights jointly, through a normalisation
  that divides by a quantity the down-projection update
  itself perturbs. Per-tensor folded equality, even exact,
  therefore does not *arithmetically entail* a stated
  logit agreement once float rounding enters at different
  points of the two computations (fold-then-forward in
  different groupings rounds differently, and RMSNorm's
  division amplifies or damps those differences by the
  stream's scale at each position). The honest consequence
  is the commission's own (RFC SI-36 §11 item 3):
  behavioural equivalence is a **forward-path
  measurement**, with populations, a statistic, and a
  tolerance designed in advance — which is what §§5.3–5.5
  do. Any future text that cites per-tensor exactness as
  establishing behavioural equivalence without the
  measurement has confused the derivation with the
  measurement, and §10 lists that confusion as a labelling
  defect.

### 5.3 Populations — allowed inputs only

The verify's behavioural part forwards both stacks over
exactly the following populations, each reported
separately, never pooled into one figure without its
per-population lines (§10):

- **Signal 1's train partition** (32 sequences) and
  **signal 1's held-out partition** (16 sequences), as
  recorded in `si/tiny_training.py`'s partition
  construction for `tiny-successor-lean-001`.
- **Signal 2's train partition** (32 sequences) and
  **signal 2's held-out partition** (16 sequences),
  likewise for `tiny-predecessor-lean-001`.

These are the allowed populations the record already
defines: the inputs both landed members were lawfully
trained and accounted on, the same populations SI-35's
audit measured on under probe exclusion. Four populations,
96 sequences in total before any overlap between the two
signals' draws is considered — and overlap is not
considered, because each population is forwarded and
reported under its own signal and partition labels
(invariant 4; RFC SI-36 §14 item 18: allowed-input figures
and judged-population figures are never interchanged, and
two allowed populations are never silently merged either).

**Probe exclusion, stated as a design constraint, not a
preference:** no identity probe input, no capability
probe input, no probe expected output, no probe logit, and
no quantity derived from any of them — including SI-33's
probe-space ‖m2‖, any per-probe drift figure, and any
SI-35 figure measured on a judged population (SI-35's
primary cell is an allowed population, signal 2's train
partition; its *probe-side* citations are not, and none is
used here) — enters the verify's populations, its
statistic, its tolerance, or its failure semantics.
An implementation of this design must assert the exclusion
the way SI-35's audit asserted it: a bundle copy with a
perturbed probe artifact must produce a byte-identical
verify record. That assertion is a requirement this design
places on its implementation (§12); it is not a result
this RFC claims, because no verify has run.

**Why these populations are sufficient for the question
asked.** The verify prices a representation rewrite, not a
generalisation claim: it asks whether two stacks compute
the same function, and it asks it where the record can
check the answer against lawful inputs of known provenance.
A rewrite that agreed on 96 recorded sequences at rounding
scale and diverged elsewhere would be diverging through
float rounding only — the carrying introduces no other
mechanism (§5.2) — and §5.5's tolerance is set at the
rounding scale precisely so that the population question
and the tolerance question are the same question. Widening
the population (fresh draws from the signals' streams, say)
is recorded as open (§11 item 5) rather than designed in,
because the streams' lawful use for a *verify* — as opposed
to the training and audit uses the record already argues —
has never been argued, and this RFC does not argue it.

### 5.4 Heads, positions, and the statistic

- **Heads compared.** The pre-squash head — the landed
  stack's head, `d0002`, with its ancestry as resolved —
  against the post-squash head: the same head id, with
  the segment's ancestry substituted by the composite,
  materialised through the existing apply path from an
  in-memory manifest view, exactly as SI-35's
  `weights_at_head` assembled named heads without
  mutating, re-heading, or writing the verified manifest.
  The verify writes no blob except the candidate blob the
  pipeline already wrote for it (§4.7's hygiene rule
  governs that blob), appends no journal entry, and
  changes no manifest; an implementation must
  snapshot-assert bundle state byte-identical before and
  after a verify *measurement*, as SI-35's audit tests
  asserted for the audit.
- **Arithmetic discipline.** Float64 comparison over
  float32 forwards, in the library forward's operation
  order — SI-35's `forward_audit` discipline, whose logits
  are asserted bit-identical to `si.tiny_transformer.forward`'s
  by test. The verify compares the two stacks' logits; it
  does not compare either stack's logits to any expected
  output, which is what keeps it a rewrite check rather
  than a disguised probe (§5.3).
- **Positions.** Every position of every sequence is
  forwarded and compared. The **pass/fail statistic is
  the maximum, over all four populations, all sequences,
  and all positions, of the per-position max-abs logit
  difference** between the post-squash and pre-squash
  forwards (the gate's `_err` form — max abs difference,
  float64 — applied between two forwards instead of
  against an expectation). The stronger all-positions
  statistic is chosen over a last-position-only statistic
  because the carrying predicts agreement at rounding
  scale at *every* position (§5.2), so the stronger
  statistic costs the design nothing and forecloses the
  one cherry-pick this measurement could otherwise commit.
  For comparability with the record — the signals' losses
  and SI-35's audit both read last positions — the
  **last-position maximum per population is reported
  beside** the pass/fail statistic, as a reported subtotal,
  never as the statistic itself.

### 5.5 Tolerance and its derivation

**The tolerance is 1e-4, as a max-abs logit difference,
for the behavioural part; and 1e-4, as a max-abs folded-
weight difference per named tensor, for the weight-space
part.** Its derivation, stated so an implementation slice
must re-derive rather than merely inherit it:

1. **The true value is zero.** In exact arithmetic both
   parts' differences are exactly 0 (§5.2). The tolerance
   therefore prices *arithmetic*, not behaviour: float32
   factor storage (machine epsilon ≈ 1.2e-7 relative),
   float64 fold and forward accumulation over tensors of
   the miniature's recorded scales, and RMSNorm division
   at the stream scales SI-35 measured (residual norms
   ≈ 2.0 on the primary population). No behavioural slack
   is being granted, because no behavioural change is
   being proposed.
2. **The value is the standing verify tolerance, not a
   new one.** Demo squash's tolerance verify has run at
   1e-4 since SI-3, over materialised weights; adopting
   the same value for the weight-space part is continuity,
   and extending it to the behavioural part is justified
   by step 1, not by analogy alone: the behavioural
   difference is the forward image of a weight difference
   the weight-space part has already bounded at the same
   tolerance, through a forward whose operations are
   Lipschitz at the miniature's recorded scales — a
   *derivation* in outline, whose honest status is that it
   bounds the expected scale (rounding, ≈ 1e-5 or below)
   an order of magnitude inside the tolerance, without
   constituting the measurement §5.4 designs.
3. **It is not derived from any gate tolerance.** 1e-4 is
   2,500× tighter than `identity_tol` 0.25 (*derivation*
   from the cited figures). Deriving a verify tolerance
   from the judge's tolerance would make the verify a
   shadow of the gate; deriving it from rounding makes it
   what it is — a rewrite check that a *learning* step
   would fail and a faithful rewrite passes. A candidate
   that needs a looser verify tolerance is not a squash
   under this design; it is a new write, and it belongs in
   the authorship line under the standing budgets.
4. **It is not derived from any probe figure.** No
   tolerance input above is a probe input, probe logit, or
   probe-derived quantity (invariant 9, §5.3): the inputs
   are the float formats, the operation counts, and
   SI-35's allowed-population stream scales.

An implementation that measures a difference above 1e-4
on a faithfully carried composite has measured something
this design does not predict, and §5.6 — not a tolerance
revision — is its correct response: re-pricing the
tolerance after the measurement is invariant 8's forbidden
move in its purest form, and §11 item 4 records the
consequence.

### 5.6 Failure semantics — fail closed

- **Weight-space failure** (any named tensor's max-abs
  folded-weight difference > 1e-4, or NaN — the comparison
  is `not (diff <= tol)`, so NaN fails, exactly as demo
  squash's verify is coded): the squash is **refused**,
  before the gate, journaling nothing; the candidate blob
  is deleted again unless it pre-existed (§4.7).
- **Behavioural failure** (the §5.4 statistic > 1e-4, or
  any non-finite logit in either forward): refused,
  identically. There is no "close enough to gate anyway"
  path: the gate never sees a candidate whose verify
  failed, because the verify is the squash's licence to
  be judged at all.
- **Unverifiable is refused.** A population that cannot
  be assembled, a head that cannot be materialised, a
  forward that cannot be run, a member whose folded update
  cannot be recomputed — each is a refusal in SI-12's
  named form, never a pass, never a silent skip, never a
  reduced-population verify reported as the designed one
  (invariant 5). A verify that ran on three of the four
  populations is not a verify that passed; it is a verify
  that did not run, and its record must say so.
- **The verify's record** — populations with their labels,
  per-population last-position maxima, the all-positions
  statistic, both parts' tolerances, and the outcome — is
  published with any future squash's journaled report
  *whatever the outcome*, in the record's standing form:
  a refusal with its measured difference stated is a
  publishable result (§10), stated as plainly as a pass.

### 5.7 What the verify would measure on the existing depth-2 stack

RFC SI-36 §11 item 3 requires this stated, and it is
stated here as a counterfactual — **the segment below is
the measurement's subject, not a squash this RFC performs
or a depth at which one would run** (`L` = 2 ≤ `keep` 3,
§1; a real squash additionally requires §6's trigger):

> If the verify were run, it would build the §3.8
> composite for the full segment S = [`d0001`, `d0002`],
> K = [] — targets `lm_head.weight` (rank 8, factors
> verbatim from `d0001`) and
> `model.language_model.layers.1.mlp.down_proj.weight`
> (rank 8, factors verbatim from `d0002`), aggregate norm
> 0.38776061454106125 (*derivation*, §3.2) — materialise
> the pre-squash head `d0002` and the composite-substituted
> head through the apply path, check the two folded
> weights per named tensor (the only two tensors whose
> folded weights differ from base; every other tensor's
> folded weight is the base tensor in both stacks, by
> construction), and forward both heads over the four
> §5.3 populations — signal 1 train (32), signal 1
> held-out (16), signal 2 train (32), signal 2 held-out
> (16) — comparing logits at every position under §5.4's
> discipline, reporting the all-positions max-abs statistic
> and the four last-position maxima against the 1e-4
> tolerance of §5.5.

**No part of that paragraph was run.** No composite was
built, no head was materialised for it, no forward was
compared, and no difference — weight-space or behavioural
— is stated anywhere in this RFC, because none was
measured. The design's expectation, labelled a derivation
and nothing more (§5.5 step 2), is that a faithful
implementation would measure differences at rounding
scale; §5.6 already states what an implementation must do
if it measures otherwise, and that instruction binds
*before* any measurement exists to be disappointed by.

### 5.8 What the verify design does not do

- It measures nothing at Glimmer scale and designs no
  Glimmer verify: the populations, the forward, and the
  audit discipline it reuses are miniature machinery, and
  the seed's bundles have no lawful population of the
  §5.3 kind. A Glimmer squash verify is not undesigned
  here by oversight; it is outside every premise this RFC
  stands on (§9).
- It does not verify provenance, policy, or identity —
  those are the gate's checks (§4.5), and a verify pass
  lends them nothing.
- It does not certify the composite's *training* claims.
  Composed artifacts do not inherit training claims
  (RFC SI-36 §14 item 10): the composite's properties are
  what its verify and its gate report measure, not what
  its members' cycles hoped.

---

## 6. The implementation trigger — stated precisely, not self-firing

**The trigger, verbatim in effect from RFC SI-36 §11
item 4.** An implementation slice (SI-38 or later, never
SI-37 itself — and no SI-38 is registered by this RFC or
by RFC SI-36) becomes **recommendable** only when at
least one of the following holds:

- **(a) Depth.** A third authored delta lands, or the
  active authored depth otherwise exceeds `keep` 3 —
  RFC SI-34 §8's trigger (a), restated. (As of this RFC:
  depth is 2; no third delta has landed; and SI-35's R2
  — quoted only with its qualifiers: single-target,
  upstream, rank-8, budget-0.9, this miniature, this
  signal family's correction scale, derivation-scoped —
  closes the only argued upstream route by which a third
  delta was projected to land, so this arm is *less*
  likely to fire than when SI-34 stated it. The readout
  route to a third delta is not closed by R2 and is not
  argued here; §5 of RFC SI-36 states its standing
  deferral, which this RFC carries.)
- **(b) Operator direction.** The operator (Mark) directs
  the implementation work — in words, for this work, as
  his directions have always been given in this project.
  A cadence's momentum, a report's recommendation, and
  this RFC's existence are none of them operator
  direction.
- **(c) Demonstrated present-tense need.** The argued
  design itself — §§3–5 as argued — demonstrates a
  present-tense need on the depth-2 stack that RFC SI-36
  did not foresee. **The burden on that demonstration is
  the claimant's, and it is heavy by design:** the
  demonstration must name a concrete harm or loss that
  exists *today*, at depth 2, with `keep` unexceeded and
  SI-12's refusal standing — not a benefit an
  implementation would confer, not a depth the stack
  might reach, and not an elegance the design would
  complete — and must show that the harm follows from the
  *absence of an implementation*, not from the absence of
  this design (which this RFC now supplies). For the
  avoidance of doubt, recorded in advance: §§3–5 as
  argued in this RFC demonstrate **no** such need. The
  carrying question was a design debt, and it is now
  paid; the debt's payment is not a harm the stack
  suffers from. Arm (c) stands for a need nobody has yet
  stated, and the first claimant under it must state it
  against this paragraph.

**Until a trigger fires** — all three arms unmet as of
this RFC — **SI-12's fail-closed refusal stands**, in
code and in fact, and this design's existence is **not
itself a trigger**, is not evidence toward any arm, and
may not be cited as momentum toward an implementation.
An implementation slice, if one is ever triggered, still
owes its own RFC-stage argument for enacting §3.8's
pricing and §§4–5's designs (§12), must prove its unwind
and its probe-exclusion assertions by test, and leaves
every standing budget it does not explicitly carry an
enactment argument for exactly where this RFC left it.

---

## 7. Decision summary

| Question | Decision | One-line reason |
|---|---|---|
| (i) Carrying — (a) authored candidate for the standing budgets | **REJECTED as the design position** | It prices a cardinality no author chose under a budget written for an author's choice, and annuls the commission's remaining items for the only stack they concern (§3.3). |
| (i) Carrying — (b) distinct record class with its own discipline | **DECIDED — the design position** | Priced per tensor and in aggregate under §3.8, attested by journal provenance the candidate does not control, judged by the same gate under the same tolerances; the demo-squash precedent already prices composites as a class in every way the demo format can raise (§§3.4, 3.6, 3.8). |
| (i) Carrying — (c) inadmissible until a policy question is argued separately | **REJECTED as the design position; its operational state holds until implementation** | The policy argument is §3, as commissioned; a fifth successive deferral is the permanent-deferral failure RFC SI-36 §15 item 4 names. Operationally the refusal stands meanwhile — by decision, not by deferral (§3.5). |
| (ii) Gate context | **DESIGNED** | `SquashContext`'s shape, transformer mapping of all five checks with meanings stated, four-axis per-member record, mixed-tensor and prospective constructed-member cases designed; additive-member segments excluded with reason (§4). |
| (iii) Tolerance verify over named tensors | **DESIGNED** | Weight-space rounding check per named tensor plus behavioural forward comparison over four allowed populations only, all-positions max-abs statistic, 1e-4 rounding tolerance with its derivation, fail-closed semantics; counterfactual depth-2 subject stated, nothing run (§5). |
| Implementation | **NOT RECOMMENDED — trigger stated, unmet** | Depth 2 ≤ `keep` 3, no operator direction, no demonstrated present-tense need (and §§3–5 affirmatively demonstrate none); SI-12's refusal stands; this design's existence is not a trigger (§6). |
| Multi-target / additive authorship | **NOT REOPENED** | The landed stack is not evidence about authorship (§3.1); RFC SI-36 §8.1's rejection stands on its own grounds; the authored budgets are unchanged (§9). |
| Glimmer squash / verify / forward path | **NOT DESIGNED — escalations only** | No lawful population, no forward path, none faked; checks 3/4 at that scale are not-evaluated escalations by the standing mapping (§§4.5, 5.8). |

The honest shape of this RFC deserves one paragraph of
its own. §§3–5 spend a full design on machinery that, by
§6's own statement, should not now be built: the stack it
would compress is two deep against a `keep` of 3, the
route to a third delta is closed or deferred on every
argued line, and the design's completion is explicitly
not a reason to implement it. That is the posture RFC
SI-34 §8's trigger (b) was written for — obligations
argued as an RFC's main business, before any
implementation is improvised inside another slice — and
the alternative postures were both worse: leaving the
carrying question for a fifth document to record as
unargued (§3.5), or letting a future depth emergency
argue it in haste, against a live stack, with a candidate
waiting. The design now exists at the record's leisure
and the implementation waits for the record's need.
If neither §6's depth arm nor the operator's direction
ever fires, the honest closing statement for the squash
line is that it was designed, decided, and never needed
at the depths the project reached — not that it was
always about to happen (RFC SI-36 §15 item 3, carried).

---

## 8. Acceptance — this RFC against RFC SI-36 §12

Stated in the shape RFC SI-36 §12 requires: each
criterion checked against this document and the untouched
repo, without trusting RFC SI-36's or this RFC's
intentions. The reviewing run verifies each item
independently; this section is the author's self-check,
labelled as such.

1. **Exactly one deliverable, doc-only — PASS by
   construction of the slice.** The slice's diff against
   its base contains exactly one new file, this document,
   and no modification to any existing file: no production
   code, no test, no manifest schema, no policy default,
   no gate behaviour. (Diff-verifiable.)
2. **The carrying question is argued and decided —
   PASS.** §3 is this document's main business, in its
   own section, before §§4–5; all three recorded readings
   are stated at full strength (§§3.3–3.5) and priced
   against the landed stack (§3.2); the chosen reading
   (b) is named in §3's verdict and §3.4, with SI-32
   §7.3's stack-growth and second-judge objections — and
   its timing objection — answered for it in §3.6, not
   cited past; and §3.1 preserves RFC SI-36 §8.1's
   separation verbatim in effect: the landed stack is
   never treated as evidence about multi-target
   authorship, and the text says a composite is
   two-target in its carrying whatever class it is
   assigned (§3.1, ground rule).
3. **The gate context is complete against the record —
   PASS, with the one recorded exclusion.** All five
   checks are named in §4.5, each with its transformer
   meaning stated (constitution, the structural safety
   suite under §3.8, identity, capability's two arms,
   provenance with the squash attestation); the context
   represents all four recorded axes per member (§4.2),
   the mixed-tensor case (§4.3, the landed stack itself),
   and the constructed-member case (§4.4, prospective);
   the additive-member case is recorded as an exclusion
   with its reason (§3.8 rule 5, §4.7) and the
   segment-mate-referencing construction as undesigned
   with its reason (§4.4) — neither silently omitted,
   neither rendered as designed.
4. **The verify is designed as a measurement, with its
   populations lawful — PASS.** Populations (four allowed
   partitions, probe exclusion asserted as a design
   constraint, §5.3), heads (pre-squash `d0002` vs
   composite-substituted, §5.4), statistic (all-positions
   max-abs, last-position subtotals reported, §5.4),
   tolerance with its derivation (1e-4, rounding-priced,
   §5.5), and failure semantics (fail closed, unverifiable
   refused, §5.6) are all stated; §5.7 states what the
   verify would measure on the existing depth-2 stack;
   and no verify result, squash result, or behavioural
   equivalence is claimed as measured anywhere in this
   RFC (§§5, 5.7).
5. **The implementation trigger is stated and is not
   self-firing — PASS.** §6 states RFC SI-36 §11 item 4's
   conditions substantially as registered — depth /
   operator direction / demonstrated present-tense need
   with its burden stated — including the clause that the
   design's existence is not itself a trigger, and states
   SI-12's fail-closed refusal as standing throughout and
   after this RFC. §6 additionally records, against arm
   (c), that §§3–5 demonstrate no such need.
6. **R2 quoted only with its qualifiers; SI-33's ADMIT
   only with its attribution; no SI-35 figure as a gate
   figure — PASS.** R2 appears at §1 and §6 (a), each
   time with its registered scope attached (single-target,
   upstream, rank-8, budget-0.9, this miniature, this
   signal family's correction scale, derivation-scoped);
   SI-33's ADMIT appears at §1 with candidate norm
   0.30801411790830574, ‖m2‖ 0.010469039280455053, F3′′′
   measured with its figures, and the near-inert-branch
   attribution, in the same breath; SI-35's figures
   (0.0340859; 0.0575995 / 0.0576504; the branch ratios
   §5.5 does not use) are cited as audit figures on
   allowed populations, never beside gate figures without
   the distinction stated, and never as a verdict
   prediction (§§1, 5.3).
7. **Untouched scope, verified — PASS on the slice's own
   run.** The full existing suite is green on this slice's
   own run, with its real numbers reported in the slice's
   PR body (the SI-35 merge's baseline: 394 passed,
   2 skipped; this design-only slice changes no test's
   result); `si squash` still fails closed on the landed
   transformer stack (existing suite, including SI-12's
   tests untouched); `si serve` still returns 501 for
   every transformer bundle (existing suite); the Glimmer
   seed is untouched; no download was performed,
   scheduled, or implied.
8. **No implementation is claimed or smuggled — PASS,
   with §11 as the evidence.** This document contains no
   code, no schema, and no instruction a successor slice
   could execute without its own RFC-stage argument:
   §§3.8, 4, and 5 are design rules, shapes, and
   measurement specifications whose enactment §6 and §12
   expressly reserve to a triggered implementation slice's
   own argument. Every obligation this RFC cannot argue
   is recorded as open in §11, with what would retire it.

---

## 9. Non-goals

Explicit, and binding on any slice this RFC is cited to
justify:

- **No implementation of any part of §§3–5.** No code,
  no test, no schema, no policy default, no CLI
  behaviour changes under this RFC. SI-12's fail-closed
  refusal stands throughout and after it (§6).
- **No squash performed.** No composite is built, folded,
  verified, gated, or journaled, on any stack, real or
  synthetic, under this RFC. §5.7's paragraph is a
  counterfactual subject statement, not a run.
- **No policy change — in either direction.**
  `max_targets_per_delta` stays 1, `allow_additive`
  stays false, `rank_cap` stays 8, the norm budget stays
  0.9 (with `norm_cap` / `per_target_norm_cap` at 1.0),
  `identity_tol` and `capability_eps` stay 0.25,
  `max_stack_depth` stays 8, `keep` stays 3. §3's
  decision is a design position for a future enactment
  argument, not an enactment (§§3.5, 3.7).
- **No multi-target or additive authorship.** No authored
  candidate of either shape is proposed, argued, or
  licensed. RFC SI-36 §8.1's rejection is carried, not
  reopened, and §3.1's separation is preserved: this RFC
  is about carrying already-judged deltas, only.
- **No new class beyond the one designed.** The squash
  composite class (§3.8) is defined by the squash
  operation and its journal attestation. No constructed-
  candidate class, no identity-alone judging, and no
  other class-based pricing is designed, implied, or
  licensed by it (§3.6). SI-31's candidate is never
  re-judged and can never be a segment member (§3.6).
- **No lossy recompression designed.** Truncation of any
  target below its exact-carrying rank — the transformer
  analogue of demo squash's truncated SVD — is recorded
  open (§11 item 2), not designed. Across tensors it does
  not exist (§3.1).
- **No third cycle, no candidate, no training, no new
  signal, target, anchoring, or composition value.**
  Every axis keeps exactly its recorded values (§1).
- **No real-base work, no download.** RFC SI-36 §6's
  operator decision and its A-1 to A-4 preconditions are
  untouched by anything here; this RFC names no base or
  family and performs, schedules, and implies no
  download.
- **No Glimmer anything.** No Glimmer forward path,
  probes, verify populations, squash, or context beyond
  the standing not-evaluated escalation (§§4.5, 5.8).
  The Glimmer seed is untouched except to restate that
  it is untouched.
- **No probe contact in the designed verify.** §5.3's
  exclusion is a non-goal's mirror: any future design
  that routes a probe quantity into a squash verify is
  outside this RFC and outside invariant 9.
- **No SI-38 registered.** This RFC registers no
  successor slice. What follows it is §6's trigger or
  the operator's decision — not a tenth miniature route
  invented to keep a cadence fed (RFC SI-36 §13, carried).
- **No serving, no CLI wiring.** Transformer bundles keep
  returning 501; `si squash` keeps refusing them; this
  RFC adds no CLI surface of any kind.
- **No scale claim of any kind.** Every figure herein is
  a fact or a labelled derivation about a constructed,
  deterministically seeded, not-trained miniature, two
  constructed signals' partitions, and three named heads.
  Nothing is claimed about any real model, any trained
  base, or Glimmer — including by analogy, and including
  the direction "a real stack would need squash sooner",
  for which this record contains no evidence.

---

## 10. Honesty and labelling requirements

RFC SI-36 §14's twenty-eight items bind any slice that
RFC is cited to justify, and this RFC with them. The
items below restate, in compressed form, the requirements
this RFC's own text and any successor's summary of it most
directly answer to, with the numbering preserved so
existing citations resolve; the full list stands in
RFC SI-36 §14 and is not abridged by this selection.

1. **Training numbers are training numbers** — labelled
   with signal, partition, head, anchoring, composition,
   and authorship target; none is a gate or probe result.
   (§14 item 1.)
2. **"Not evaluated" is never rendered as 0.0, a pass,
   or an omission** — including, for this RFC, a designed
   check's escalation at Glimmer scale. (Item 2.)
3. **Authorship is not admission; carrying is not
   admission either.** No text describes a composite as
   landed, admissible, or admitted before a gate that
   does not yet exist for it returns such a verdict.
   (Item 3, extended in effect to composites.)
4. **Every number carries its source.** Inherited figures
   are cited to the document that measured them (§1's
   table to SI-33; §1's audit figures to SI-35).
   (Item 4.)
5. **The miniature's labels travel:** "constructed,
   deterministically seeded, not trained" accompanies the
   miniature wherever its results are summarised.
   (Item 5.)
6. **Failure states are reportable successes of the
   experiment.** A future squash refusal with its measured
   verify difference stated (§5.6) is a publishable
   result, as plainly as an ADMIT. (Item 6.)
7. **Head-relative numbers name their head** — including
   every figure in §1's table (trained against `base` /
   `d0001`; cumulative at `d0001` / `d0002`). (Item 8.)
8. **Incremental and cumulative are different numbers**,
   never presented as each other — the landed cycle-2
   identity figure in this RFC is cumulative
   (seed-absolute) wherever it appears, and is labelled
   so at §4.5. (Item 9.)
9. **Composed artifacts do not inherit training claims.**
   A composite's properties are what its verify and gate
   report measure (§5.8). (Item 10.)
10. **Recorded selection is not a choice the mind made.**
    A composite's per-member axes are driver selections
    recorded per member; no text describes the mind, the
    loop, or the squash as choosing any of them.
    (Item 13.)
11. **A derivation is not a prediction.** §3.2's
    aggregate norm, payload, and pricing arithmetic,
    §5.2's exactness, and §5.5's expected rounding scale
    are labelled derivations wherever they appear and are
    never quoted with a measurement's authority.
    (Item 14.)
12. **Anchoring, composition, and authorship target are
    part of every label** (items 15, 17, 24) — a
    composite's members keep their own values on all four
    axes (§4.2); there is no composite signal, anchoring,
    or composition configuration, and describing one is a
    labelling defect under this item.
13. **Allowed-input numbers are not probe numbers**
    (item 18), and **a worst-case figure names its
    population** (item 20) — §5's statistic is a worst
    case *over the four named allowed populations*, and
    must be quoted with them attached.
14. **A constructed candidate is labelled constructed**
    (item 22) — including prospectively, in §4.4's
    designed record, where no constructed member exists.
15. **A two-line refusal is reported on both lines**
    (item 25) — SI-31's, wherever §3 or §4 touches it,
    is safety rank *and* identity, independently.
16. **An ADMIT's attribution travels with the ADMIT**
    (item 26) — SI-33's, per §1 and §8 criterion 6.
17. **A measurement-only record is not a verdict record**
    (item 27) — SI-35's figures, and by the same rule
    §3.2's pricing record: neither is evidence that any
    candidate or composite would pass or fail any gate
    line.
18. **R2's qualifiers travel with R2** (item 28) —
    single-target, upstream, rank-8, budget-0.9, this
    miniature, this signal family's correction scale,
    derivation-scoped, at §1 and §6 (a) and wherever else
    R2 is quoted. Quoting R2 as closing squash, as
    opening squash, or as evidence about composites in
    either direction is a labelling defect under this
    item: R2 measures single-target expressibility, and
    §§3–5 cite it only for §6 (a)'s depth projection.
19. **Per-tensor exactness is not behavioural
    equivalence** (new in this RFC, in item 14's family):
    §5.2's derivation covers folded weights; behavioural
    agreement is §5.4's measurement or it is unmeasured,
    and any summary that elides the verify when citing
    the carrying's exactness commits the confusion §5.2
    names.

---

## 11. Risks and open questions

Each item names what would retire it. Numbers this RFC
deliberately does **not** supply are listed as open —
supplying them here would be inventing them.

1. **Reading (b) will be misread as "multi-target is
   allowed now."** It is the standing exposure of §3's
   decision: the class is narrow (attested squash
   composites only), the authored budgets are unchanged,
   and SI-12's refusal is still the operational law — but
   a summary that drops any of those three qualifiers
   manufactures a permission this RFC never grants, in
   the direction with the worst consequences (an authored
   multi-target candidate citing SI-37 as its licence).
   *Retired by:* nothing — §3.1's separation and §10
   item 3's labelling discipline, applied in every
   successor document and PR body; recorded here because
   successor documents are where qualifiers go to die.
2. **Obligations this RFC records as open, each with its
   retirement condition** (§8 criterion 8's evidence):
   (i) *Lossy per-tensor recompression* — truncating a
   target below its exact-carrying rank, with the
   singular-mass accounting an honest truncation record
   requires (SI-31's Phase-0 singular record is the
   shape). *Retired by:* an argued design for that
   accounting, in the slice that first needs truncation,
   with the verify (§5) unchanged as its guard — never
   by citing demo squash's truncation as a precedent that
   transfers without the accounting. (ii) *Additive-member
   segments* (§3.8 rule 5). *Retired by:* an argument, in
   its own right, about `allow_additive`'s relation to
   carried additive updates, made when a landed additive
   member exists to argue about — none does. (iii)
   *A construction referencing a segment-mate* (§4.4).
   *Retired by:* cycle semantics that define constructions
   against segments, if such semantics are ever argued
   for their own sake; the case cannot arise under the
   present cycle machinery. (iv) *The allowlist state of
   the landed bundle* (§4.5 check 2): whether the landed
   miniature bundle's `allowed_target_patterns` allowlists
   both landed tensors for a *composite* record is a fact
   an implementation run will meet, not one this design
   needs. *Retired by:* that run's own record, or by the
   policy question it would raise, faced as a policy
   question at that time.
3. **The (c) fallback.** §3.5 rejected reading (c) on the
   commission and the record's completeness. An objection
   to (b) that §3.6 does not answer — a concrete record a
   composite class would misprice, an attestation the
   journal cannot in fact verify, a comparability loss
   beyond §3.6's stated concession — reopens (c) as the
   standing position, and the objector's correct move is
   to record the objection against §3.6 in the reviewing
   run's notes, leaving SI-12's refusal (the operational
   state in either case) untouched. *Retired by:* such an
   objection, recorded; until then the decision stands as
   argued, and this item is its stated falsification
   condition rather than a hedge.
4. **The verify's tolerance may fail its first honest
   composite.** §5.5 derives 1e-4 from rounding, with the
   expected scale an order of magnitude inside it — a
   derivation, not a measurement. If a faithful
   implementation measures above tolerance, the design's
   own instruction (§5.6) is refusal and investigation,
   and the tempting move — re-derive a looser tolerance
   from the measured failure — is invariant 8's forbidden
   move after the fact. *Retired by:* a measured verify
   record, pass or fail, published under §5.6; a fail
   retires this item by converting it into a finding
   about the fold/forward arithmetic, which is what the
   verify exists to produce.
5. **Verify populations are the recorded partitions, and
   no wider.** §5.3 declined fresh stream draws because
   their lawful verify use is unargued. A future reader
   may find 96 recorded sequences a thin basis for an
   equivalence claim; the design's answer is §5.3's —
   the claim priced is a rewrite claim, whose only error
   mechanism is rounding — but the thinness is real and
   is recorded rather than argued away. *Retired by:* an
   argument for stream draws as verify inputs, in the
   implementation slice or its own RFC, if that slice
   judges the basis too thin for the claim it must make.
6. **Depth pressure may never arrive, and the design may
   age.** §6's depth arm depends on a third delta the
   record no longer projects; the operator arm depends on
   a decision RFC SI-36 §6 laid out and this RFC does not
   advance. A design that sits unimplemented while the
   codebase moves risks describing a `SquashContext`
   shape, a gate mapping, and an apply path that have
   since changed — §§4–5 cite code on `main` at `18169fb`,
   and an implementation slice that finds those citations
   stale must re-argue the affected design against the
   code as it then stands, not implement the citations.
   *Retired by:* §6's trigger firing while the citations
   are fresh, or by a future document recording the
   design's staleness honestly — the permanent-deferral
   closing statement of §7, applied to this RFC itself.
7. **The constructed-member design is untested against
   any real member.** §4.4 designs for a member class with
   zero landed instances, on the strength of RFC SI-32
   §13.1's recorded obligation. Its rank arithmetic is a
   derivation and its representation a design; neither has
   met a constructed delta that landed. *Retired by:* the
   first lawfully landed constructed member's squash, if
   one ever occurs; until then this item stays open by
   design, and §4.4 must be cited as prospective wherever
   it is cited at all.

---

## 12. Summary

RFC SI-34 §8 recorded three design obligations for
transformer squash and a trigger for arguing them; RFC
SI-36 evaluated that trigger against SI-35's evidence —
whose pre-registered readings (R1 NOT measured, **R2
MEASURED**, R3 NOT measured) close the single-target
upstream-authorship line on this miniature at the standing
budgets, with R2's qualifiers attached (single-target,
upstream, rank-8, budget-0.9, this miniature, this signal
family's correction scale, derivation-scoped) — found the
depth arm unmet and receding, and commissioned this RFC
under the argument arm. This RFC argues the obligations
in their recorded order, against the one stack that
exists. **First, the carrying question, decided (§3):**
the landed stack (`d0001` on `lm_head.weight`, norm
0.23555380990005192, 2,360 B; `d0002` on
`model.language_model.layers.1.mlp.down_proj.weight`,
norm 0.30801411790830574, 3,416 B — the second delta
labelled a non-readout-target candidate, its ADMIT
carried only with SI-33's attribution: ‖m2‖
0.010469039280455053, F3′′′ measured, the near-inert
final-MLP branch) would squash into a composite carrying
both tensors' updates — two-target in its carrying,
whatever class it is assigned, as §3.1 states before any
reading is priced — and of the three recorded readings,
this RFC decides **(b)**: a distinct record class, priced
per tensor and in aggregate under §3.8's discipline
(count equal to the attested segment's distinct tensors;
ranks at their true factorisation values, priced in
bytes; norms and bytes against the unchanged caps;
additive members excluded with reason), attested by
journal provenance no authored candidate can present,
with SI-32 §7.3's stack-growth, second-judge, and timing
objections answered in §3.6 — while the authored budgets
(`max_targets_per_delta` 1, `allow_additive` false) stand
unchanged and the landed stack is never cited as evidence
about multi-target authorship. **Second, the gate context,
designed (§4):** demo `SquashContext`'s shape — probes
from the mind, capability baseline on the pre-squash
stack, post-squash forward, depth reported so the gate's
+1 is the true post-squash depth (2 → 1 on the landed
stack) — carrying a per-member record over all four
recorded axes (the landed members differing on three of
them), the mixed-tensor case at its centre, the
constructed-member case prospectively, and all five
checks named with their transformer meanings, decisive
and probe checks alike, with not-evaluated escalation —
never a faked pass — at any scale without a verified
miniature probe context. **Third, the verify, designed
(§5):** per-tensor folded exactness, which is a
derivation, kept separate from behavioural equivalence,
which is a measurement — over four allowed populations
only (both signals' train and held-out partitions; no
probe quantity anywhere in the design), at every
position, under the max-abs statistic, at a 1e-4
rounding tolerance derived from the arithmetic rather
than from any gate tolerance, failing closed and refusing
the unverifiable — with §5.7's counterfactual stating
exactly what it would measure on the depth-2 stack and
nothing in this RFC claiming it was run. **And the
trigger, stated (§6):** implementation becomes
recommendable only on depth beyond `keep`, on the
operator's direction, or on a demonstrated present-tense
need whose burden §6 fixes and which §§3–5 affirmatively
do not demonstrate — until then SI-12's fail-closed
refusal stands, this design's existence is not a trigger,
and no SI-38 is registered. The deepest landed authored
stack is depth 2; the base stays frozen, the bytes are
never duplicated, no tolerance, cap, budget, or policy
default moves in either direction, the gate is unchanged,
`si squash` still refuses the landed stack, and nothing
is claimed that a test in this repo cannot check.

---

*RFC SI-37 — design only. RFC SI-12 made the refusal
honest; RFC SI-34 recorded what an honest design would
owe; RFC SI-36 priced the remaining routes and found
this one. This RFC pays the recorded debt in full — a
carrying question decided rather than deferred for a
fifth time, a gate context that remembers everything its
members were, and a verify that knows the difference
between a derivation and a measurement — and then stops,
on purpose, at the exact line its commission drew: the
design is the deliverable, the implementation belongs to
a trigger that has not fired, and the programme's next
move after this document is the operator's to make,
recorded here rather than taken.*
