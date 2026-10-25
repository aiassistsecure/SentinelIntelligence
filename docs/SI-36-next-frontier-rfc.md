# SI-36 RFC — The Next Frontier After the Expressibility Audit
(Design Only)

> **Status: DESIGN ONLY.** This document evaluates candidate
> directions and recommends one next slice. It changes no
> production code, no test, no manifest schema, no policy
> default, and no gate behaviour. Nothing in this RFC is
> implemented unless a section explicitly says it exists today,
> in code, on `main`.
>
> **The one sentence that governs everything below:** SI-35
> measured the expressibility audit RFC SI-34 recommended and
> its pre-registered readings came back **R1 NOT measured,
> R2 MEASURED, R3 NOT measured** — no in-stream non-readout
> matrix satisfies R1's pair on the primary cell (signal 2's
> train partition at head `d0001`): the branch-ratio clause
> (≥ 0.04) fails for all 14 matrices, the largest measured
> branch ratio on the primary cell being the final layer's
> attention branch's 0.0340859, and the bound clause
> (≥ 0.212) fails for exactly the two down-projections
> (0.0575995 and 0.0576504) — so the shortlist is **empty**
> and the single-target upstream-authorship line stands
> *expressibility-closed on this miniature at the standing
> budgets* (rank-8, budget 0.9), a derivation-scoped
> conclusion with its five qualifiers (single-target,
> upstream, rank-8, budget-0.9, this miniature, this signal
> family's correction scale) attached wherever it is quoted
> — and this RFC must never be cited as evidence that
> authorship is impossible in general, that a second target
> was approved or shortlisted, that transformer squash has
> been implemented or its design obligations argued away,
> that a third cycle has a lawful configuration, that a
> real-base download has been approved, recommended as
> approved, or scheduled, or that any slice proposed here
> exists.
>
> **Scope discipline:** CPU only. No GPU, no cloud, no spend,
> no downloads. Terminology: this is machine learning
> infrastructure throughout.
>
> **Relationship to prior work:** RFC SI-34 evaluated the
> frontiers open after SI-33 and recommended a
> measurement-only branch-contribution and expressibility
> audit; SI-35 built it (PR #40, as-built in
> `docs/SI-35-branch-contribution-expressibility-audit.md`)
> and applied RFC SI-34 §15 item 7's readings exactly as
> registered. RFC SI-34 §15 item 8 and SI-35's own closing
> sections both stated, in advance, what this RFC is: if
> the shortlist were non-empty, SI-36 would be a target RFC
> arguing one tensor from SI-35's record; the shortlist is
> empty, so SI-36 is instead the RFC that argues from R2's
> stated consequence — RFC SI-34 §8's squash trigger, §9's
> semantics, or §4's operator decision — and SI-35
> pre-wrote none of them. This RFC evaluates all three
> named routes against SI-35's evidence on their merits
> (§§4–6), evaluates the two further questions R2 genuinely
> bears on — the representation question and multi-target
> authorship (§§7–8) — and records the candidates R2 does
> not touch as carried, not reopened (§9). Its conclusion
> is stated plainly in §10 and §16: **the miniature
> authorship line is at a natural decision point for the
> operator**, and the one autonomous slice that remains
> honest to recommend is a design-only slice — SI-37,
> arguing transformer squash's three design obligations
> under RFC SI-34 §8's trigger (b), carrying question
> first, with no implementation (§§11–12). SI-34's
> rejections of additive / multi-target authorship, of
> per-probe anchoring, probe-partitioned sequencing, and
> magnitude discipline as a training dial, of a
> minimum-movement floor, and of Glimmer probing, are
> carried forward, not reopened. The SI-9 RFC's §7 risks
> remain in force and are inherited, not relitigated.

---

## Table of contents

1. [Where SI stands after SI-35](#1-where-si-stands-after-si-35)
2. [Inherited invariants](#2-inherited-invariants)
3. [What SI-35 actually changed in the evidence — the deep treatment](#3-what-si-35-actually-changed-in-the-evidence--the-deep-treatment)
4. [Route 1 (§8's squash trigger): transformer squash after R2](#4-route-1-8s-squash-trigger-transformer-squash-after-r2)
5. [Route 2 (§9's semantics): a third authored cycle after R2](#5-route-2-9s-semantics-a-third-authored-cycle-after-r2)
6. [Route 3 (§4's operator decision): a small real safetensors base](#6-route-3-4s-operator-decision-a-small-real-safetensors-base)
7. [Candidate (d): the representation question — does R2 fire its trigger?](#7-candidate-d-the-representation-question--does-r2-fire-its-trigger)
8. [Candidate (e): multi-target and budget-shaped escapes from R2](#8-candidate-e-multi-target-and-budget-shaped-escapes-from-r2)
9. [Candidates carried closed or deferred — R2 supplies no new evidence](#9-candidates-carried-closed-or-deferred--r2-supplies-no-new-evidence)
10. [Decision summary](#10-decision-summary)
11. [The recommended slice: SI-37 (outline)](#11-the-recommended-slice-si-37-outline)
12. [Acceptance criteria for SI-37](#12-acceptance-criteria-for-si-37)
13. [Non-goals](#13-non-goals)
14. [Honesty and labelling requirements](#14-honesty-and-labelling-requirements-for-any-future-slice)
15. [Risks and open questions](#15-risks-and-open-questions)
16. [Summary](#16-summary)

---

## 1. Where SI stands after SI-35

Everything in this section is verifiable in the code and
docs on `main` at the SI-35 merge (main tip `20f6194`,
the merge of PR #40). Numbers are cited to the as-built
document that measured them; this RFC measures nothing
new. Where a statement below is a *derivation* from cited
facts rather than a measurement, it is labelled a
derivation, exactly as SI-34 §1 labelled its arithmetic.

**What exists, exactly:**

- **One measurable transformer subject, unchanged.** The
  SI-14 miniature (`TinyTransformerMind`): a constructed,
  deterministically seeded, **not trained** decoder-only
  transformer — 30 tensors, 23,456 elements. Architecture
  pinned: vocabulary 32, `d_model` 32, 2 layers, 4 heads,
  head dimension 8, MLP hidden 64, maximum sequence 16
  (`si/tiny_seed.py::ARCHITECTURE`). Probe suites: 8
  identity + 16 capability (echo-task) entries; recorded
  base echo accuracy 16/16 = 1.0, echo floor 0.9375;
  bundle tolerances both 0.25 (`identity_tol`,
  `capability_eps`).
- **Live probe verdicts for miniature bundles (SI-17),
  unchanged.** Probe-fail → REJECT, all-pass →
  ADMIT-eligible, anything unevaluated → ESCALATE,
  exclusively probe-blind candidates → blind-escalated
  (`si/probes.py::derive_probe_blind_targets`; the blind
  set is exactly both layers' `k_proj.bias`).
- **Two authored signals, two landed deltas (SI-19 +
  SI-23 + SI-33), unchanged by SI-35.** The landed
  authored stack is depth 2: `d0001` (signal 1,
  `tiny-successor-lean-001`, readout target) + `d0002`
  (signal 2, `tiny-predecessor-lean-001`, final-MLP-down
  target) — the second delta labelled a
  non-readout-target candidate wherever the stack is
  described (SI-32 §18 item 24). Both signals share one
  form (target-logits MSE), correction scale 0.15, 2,000
  full-batch steps, learning rate 1.0, norm budget 0.9,
  32 train / 16 held-out examples.
- **Two recorded target anchorings (SI-25), four recorded
  composition configurations (SI-31), and two recorded
  authorship targets (SI-33), all unchanged.**
  `AUTHORSHIP_TARGETS` serves exactly `"lm-head"` (the
  default) and `"final-mlp-down"`. SI-35 added exactly
  zero new values on any of the four recorded axes —
  signal, anchoring, composition, authorship target —
  and there is no autonomous, learned, depth-driven,
  verdict-driven, or probe-driven selection on any of
  them.
- **One read-only audit module (SI-35, new).**
  `si/tiny_audit.py`: `forward_audit` repeats the library
  forward's computation in the same operation order and
  float64-over-float32 discipline, additionally exposing
  per position every branch output, the residual stream
  each branch adds into, and the activation each weight
  matrix consumes; its logits are asserted bit-identical
  to `si.tiny_transformer.forward`'s by test.
  `weights_at_head` assembles a named head's weight set
  through the existing apply path from an in-memory
  manifest view — the verified manifest is never mutated,
  re-headed, or written; nothing is materialised or
  stored, so the never-duplicate-bytes rule holds by
  construction. `run_audit` fails closed, naming the
  missing head, on any bundle whose active chain does not
  carry the landed stack. The module invokes no training
  entry point, constructs no candidate, opens no gate
  evaluation, and reads no judged-population artifact.
  No CLI wiring was added; `si serve` still returns 501
  for every transformer bundle; `si squash` still fails
  closed on the landed transformer stack.

**What SI-35 measured (cited to its as-built):**

- **The reproduction anchor held, before any other figure
  was trusted.** Under the audit's method, on signal 2's
  train partition at head `d0001` (last positions):
  ‖s‖ measured 0.04008178834376358–0.06405601655456196
  against SI-33's published 0.040–0.064 (**reproduces**),
  and ‖h_mid‖ measured 1.9678289013214776–
  2.0192666824884555 against SI-33's published 1.97–2.02
  (**reproduces**). SI-33's base representative-sequence
  figures (branch output 0.0074 against residual 2.00)
  were reconciled in print with the method difference
  named — single unnamed sequence (SI-33) vs named
  population (this audit) — the population residual mean
  1.996662501317634 matching the published 2.00 at its
  printed precision and the branch figure sitting inside
  the audit's measured envelope. The registered stop rule
  did not trigger.
- **Coverage was complete and inventoried.** All 30
  tensors appear in the audit's record: the 17 weight
  matrices (14 in-stream, the readout `lm_head.weight`
  tabled as the record's yardstick target and not as a
  candidate, and the two embedding matrices tabled
  separately under the index-input caveat) and the 13
  vector tensors, inventoried with blind-set status —
  the derived blind set is exactly both layers'
  `k_proj.bias`; no weight matrix is in the derived
  blind set. Where a figure cannot be carried it is
  recorded "not applicable" with the reason — never as
  0.0, never as an omission.
- **Branch contribution, at the primary cell.** On
  signal 2's train partition at head `d0001`, last
  position, max per-example branch ratios: layer-0
  attention 0.0303477, layer-1 attention **0.0340859**
  (the largest measured branch ratio on the primary
  cell), layer-0 MLP 0.00329234, layer-1 MLP 0.00347083.
  The embedding sum is the initial residual stream itself
  (ratio 1 / 1, an input-side row, labelled as such).
  No branch reaches the registered 0.04 anywhere on the
  primary cell.
- **Expressibility bounds, as derivations.** For each
  in-stream matrix, B(W) = 0.9 × (max input norm over the
  primary population at the named head). The bound clause
  (≥ 0.212, the signal correction's norm 0.15·√2) passes
  for 12 of 14 matrices — every matrix whose consumed
  activation is a normed stream or the concatenated
  attention output, bounds 0.463–5.091 — and fails for
  exactly the two down-projections, whose SwiGLU inputs
  are the model's smallest activations: layer 0
  0.0575995, layer 1 0.0576504, the latter beside SI-33's
  cited yardstick ≈ 0.058. **Stated limitation, in the
  body, as SI-35 stated it:** a local bound prices
  movement at the tensor's own output; downstream
  transfer to logits is *not* measured by the audit for
  any tensor except where a landed delta already measures
  it — the readout (SI-25 control: norm
  0.2515524481203623, ‖m2‖ 0.7816722548913912) and the
  final down-projection (SI-33: candidate norm
  0.30801411790830574, ‖m2‖ 0.010469039280455053). Every
  other tensor's transfer coefficient is listed as
  **unknown**, never estimated.
- **The readings, applied as registered: R1 NOT
  measured, R2 MEASURED, R3 NOT measured** — exactly one
  of R1/R2, as registered. R1's pair fails on the primary
  cell for all 14 matrices: the branch-ratio clause fails
  universally, and the bound clause fails for exactly
  the two down-projections. The shortlist is **empty**.
  R3 does not fire: the set of matrices satisfying R1's
  pair on signal 2's train partition is empty at head
  `base`, empty at head `d0001`, and empty at head
  `d0002` — qualification is head-stable *across the
  three measured heads only*, a claim scoped to those
  heads, never to depth in general. (The underlying
  figures are not head-identical — the final MLP branch's
  ratio moves 0.00347 → 0.00400 at `d0002`, where the
  landed down-projection delta acts — but no figure
  crosses a registered threshold at any head, and R3 is
  a reading about qualification sets, not about figures.)
- **No side effects, asserted.** An audit run writes no
  blob, performs no gate evaluation, appends no journal
  entry, and changes no manifest (bundle state
  snapshot-asserted byte-identical before/after by test);
  a perturbed-probe-artifact bundle copy produces a
  byte-identical audit record; both signals' definitions,
  all four composition configurations, both anchorings,
  both authorship targets, and every policy default are
  untouched. The full suite at the SI-35 merge: **394
  passed, 2 skipped** (SI-35's own file: 12 passed — the
  exact delta over the SI-33 merge's 382 passed,
  2 skipped; no prior test changed its result).

**The seven-run cycle-2 record, unchanged by SI-35**
(each figure under its own run's labels; SI-35 authored
no candidate and re-judged none):

| Run | Cycle-2 configuration | Composed identity drift (cumulative) | Verdict |
|---|---|---|---|
| SI-21 | signal 2, weights-under-training, target `"lm-head"` | 0.34378434903919697 | REJECT |
| SI-23 | signal 2, weights-under-training, target `"lm-head"` | 0.2963416576385498 | REJECT |
| SI-25 | signal 2, base-anchored, composition `"none"`, target `"lm-head"` | 0.3329036422073841 | REJECT |
| SI-27 | signal 2, base-anchored, composition `"movement-composition"`, λ = 1.0, target `"lm-head"` | 0.32509003952145576 | REJECT |
| SI-29 | signal 2, base-anchored, composition `"worst-input-composition"`, λ_w = 1.0, k = 4, target `"lm-head"` | 0.31215621903538704 | REJECT |
| SI-31 | signal 2, base-anchored, composition `"train-partition-projection"`, α = −0.8450027779470831 (constructed, rank 16) | 0.4775387533009052 | REJECT (safety rank + identity) |
| SI-33 | signal 2, base-anchored, composition `"none"`, target `"final-mlp-down"` | 0.17195402272045612 | ADMIT — with its attribution (§2 invariant 4, §14 item 26) |

SI-33's ADMIT is carried in this RFC only with its
attribution attached, as §14 item 26 requires: candidate
norm 0.30801411790830574, probe-space movement
‖m2‖ = 0.010469039280455053, F3′′′ **measured** (the
destination traded away: held-out residual
0.0027387202915082178 against the boundary
2.087217177504918e-05, ~131×), attributed by SI-33 to the
near-inertness of the branch it acts through (the final
layer's MLP branch ≈ 0.4% of the residual stream on the
measured population). The Glimmer seed is untouched;
Glimmer-scale deltas still escalate with checks 3/4 "not
evaluated"; stored tensor bytes are still 0 against
59,553,374,040 deduplicated bytes.

---

## 2. Inherited invariants

Not negotiable inside this RFC (SI-9 §1.3, RFC SI-13 §2,
RFC SI-18 §2, RFC SI-20 §2, RFC SI-22 §2, RFC SI-24 §2,
RFC SI-26 §2, RFC SI-28 §2, RFC SI-30 §2, RFC SI-32 §2,
RFC SI-34 §2, restated for the candidates):

1. **Base is frozen forever.** All change is a stack of
   deltas; rollback is moving `head`.
2. **Never duplicate bytes** (Mark's design rule). A blob
   whose hash exists is never rewritten or copied; seed
   imports by reference where the pattern applies
   (`stored_bytes: 0` — the SI-8 Glimmer seed stores zero
   tensor bytes against 59,553,374,040 deduplicated
   bytes, per the SI-35-era bundle record).
3. **No ungated write.** Authorship produces *candidates*.
   The gate — and only the gate — admits, rejects, or
   escalates them, and SI-17's verdict mapping and
   blind-set guard apply to authored candidates exactly
   as to fixtures, at every cycle, for every signal,
   under every anchoring, under every composition
   configuration, and under every authorship target.
4. **Honest labels are load-bearing.** "Not evaluated"
   stays distinguishable from a measured 0.0; a training
   loss stays distinguishable from a gate result; a
   number measured at one head stays labelled with that
   head; incremental and cumulative figures are never
   presented as each other; a number recorded under one
   anchoring, composition configuration, or authorship
   target is never presented as a number under another;
   a worst-case figure names the population it is a worst
   case *over*; a judgement-call reading is applied under
   the rule pre-registered for it, or reported as an
   observation rather than a falsification test; a
   constructed candidate stays labelled constructed; a
   candidate that fails on two independent gate lines is
   reported failing on both, never on the more convenient
   one (§14 items 22–25); **an ADMIT's measured
   attribution travels with the ADMIT wherever the ADMIT
   is summarised (§14 item 26); a measurement-only record
   is not a verdict record (§14 item 27); and R2's
   derivation-scoped qualifiers travel with R2 wherever
   R2 is quoted (§14 item 28, new in this RFC).**
5. **Fail closed.** A missing, mismatched, or
   un-runnable measurement is "not evaluated" — never a
   pass, never a silent skip.
6. **CPU only, existing dependencies only.** The
   runtime's dependencies are `numpy` and `safetensors`
   (`requirements.txt`, `pyproject.toml`). No GPU, no
   network, no downloads, no spend, no new heavy
   dependency.
7. **Determinism where tests need it.** Seeded RNG,
   fixed inputs, no sampling, no wall-clock dependence
   in any measured or trained value — and deterministic
   *serialisation* wherever a blob hash is content
   addressing.
8. **No tolerance — and no cap — is a design variable.**
   No candidate in this RFC proposes moving
   `identity_tol`, `capability_eps`, `rank_cap`, the
   standing norm budget 0.9, any other cap, or any policy
   default to make an outcome pass — and, the mirror
   image SI-34 §2 stated, none proposes moving any of
   them to make an outcome *fail* either. R2 is a
   conclusion *at the standing budgets*; §8 evaluates the
   candidates that would escape R2 by re-pricing a budget
   and does not adopt any of them. The seven cycle-2
   verdicts and SI-35's three readings are measurements
   to design *around* — with new semantics or new
   measurements — never numbers to outvote.
9. **The probes are the judge, never the training
   set.** No candidate in this RFC may train on, penalise
   against, project against, parameterise against, or
   otherwise shape a candidate using any probe input,
   probe logit, or probe-derived movement. SI-35's audit
   honoured this invariant in its stricter form —
   measurements on allowed populations only, probe
   exclusion asserted by a byte-identical record from a
   perturbed-probe-artifact bundle copy — and no
   candidate below may cite SI-35's record as a reason to
   relax it: the audit's figures are allowed-population
   figures, and that is exactly why they may shape a
   future *argument*.

---

## 3. What SI-35 actually changed in the evidence — the deep treatment

SI-33 made the question *how much any place an update
could act can move this model at the standing budgets*;
SI-35 answered it, for every weight matrix in the model,
before the fact. This section assembles what SI-35
jointly establishes with the prior record, what it
leaves unknown, and what any candidate must therefore
argue — not assume — from here. Derivations are labelled
derivations.

### 3.1 The closure is a branch-level fact, not a bound-level fact — and that distinction decides what follows

The two clauses of R1's pair fail in different patterns,
and the pattern is the finding:

- **The bound clause mostly passes.** 12 of 14 in-stream
  matrices clear B(W) ≥ 0.212, most by a wide margin —
  the matrices that consume a normed stream measure
  input-norm maxima of 5.65663 and bounds of 5.09097.
  Locally, at their own outputs, lawful updates on those
  matrices are *not* budget-starved. The clause fails
  only where SI-33 already knew it failed: the two
  down-projections, whose inputs are the SwiGLU outputs
  — the smallest activations in the model (bounds
  0.0575995 and 0.0576504 against the required 0.212).
- **The branch-ratio clause fails everywhere.** The
  ratio is a property of a *branch*, shared by every
  matrix in it: both attention branches measure max
  ratios of 0.0303477 (layer 0) and 0.0340859 (layer 1);
  both MLP branches measure 0.00329234 (layer 0) and
  0.00347083 (layer 1) — the MLP branches an order of
  magnitude below the registered 0.04, the attention
  branches at ~85% of it (0.0340859 / 0.04; *derivation*
  from the cited figures). No matrix escapes its branch's
  ratio, so no matrix satisfies the pair.

Read together, the audit's precise statement is not
"every tensor is too small to matter". It is: **on this
miniature, every residual-adding branch contributes too
small a share of the residual stream for a single-target
rank-8 update at budget 0.9 through any one of its
matrices to be argued — under the registered pair — as
expressing this signal family's correction, and the two
matrices whose branches SI-33's slice actually exercised
fail on the bound as well.** The closure is therefore
scoped exactly as R2's registered consequence scoped it
— single-target, upstream, rank-8, budget-0.9, this
miniature, this signal family's correction scale — and
§§4–8 test each candidate against that scope rather than
against a slogan version of it. In particular: R2 says
nothing about the readout (which expresses the
correction fully and is refused on composition, not
expressibility), nothing about multi-target candidates
(§8), and nothing at any other budget (§8) — and each of
those silences is a qualifier doing work, not a gap to
be filled by enthusiasm in either direction.

### 3.2 What the audit's numbers do not show — the optimism warning, restated against the new record

RFC SI-34 §19 item 1 warned that a local bound is an
upper bound at a tensor's own output, before downstream
attenuation the audit deliberately does not estimate,
and that a matrix could qualify under R1 and still
deliver a candidate whose logit movement is attenuated
below usefulness. SI-35's record makes the mirror-image
point equally binding now that *no* matrix qualified:
the large bounds on the normed-stream matrices (5.09097)
price movement at those tensors' outputs only, with
downstream transfer recorded **unknown** for 12 of the
14 in-stream matrices. Nothing in the audit measures
that a candidate on an attention matrix *would* move the
logits by any particular amount, would ADMIT, or would
reach its destination (§14 item 27) — and, symmetrically,
nothing in the audit measures that such a candidate
would REJECT: an inert candidate may well ADMIT, as
SI-33's did. R2 closes an *argument* — the argument that
a named single upstream tensor can be proposed, from
measured branch scale and a derived bound, as able to
express the correction at the standing budgets. It does
not close, predict, or price any gate verdict. Any
successor text that quotes R2 as "upstream candidates
would fail the gate" has committed exactly the confusion
§14 item 27 exists to prevent, in the pessimistic
direction.

### 3.3 The head-stability result is small, real, and scoped

R3 NOT measured is the audit's quietest reading and the
only one that *removes* a future obligation: because the
qualifying set is empty at `base`, at `d0001`, and at
`d0002`, no future argument on this record needs to
defend a head-dependent qualification, and no figure in
the audit crosses a registered threshold between heads.
The claim is scoped to the three measured heads and to
qualification *sets* (§1) — the figures themselves do
move between heads (the final MLP branch's ratio at
`d0002` is the standing example) — and it licenses no
statement about depth in general. Its one forward use in
this RFC is narrow: §5 cannot argue that a third cycle's
upstream candidate would qualify at a deeper head where
it did not qualify at `d0001`, because on the only heads
that exist, qualification did not vary. That is a
scoped, measured negative, and §5 uses it at exactly
that scope.

### 3.4 What SI-35 did not change

Stated plainly, so no candidate below can borrow a
change that did not happen:

- **No verdict moved.** SI-35 authored no candidate,
  performed no gate evaluation, and re-judged nothing.
  The seven-run cycle-2 record stands exactly as SI-33
  left it, including SI-31's two-line refusal (safety
  rank *and* identity, independently) and SI-33's ADMIT
  with its attribution (§1).
- **The composition attribution's scope is unchanged.**
  The composition line (SI-27/SI-29/SI-31) remains closed
  as a line about readout movement pairs through the
  shared fold. SI-35 measures no movement pair at all —
  it measures branches, activations, and derived bounds —
  so it neither reopens that line nor extends its closure
  (§9).
- **No policy, tolerance, cap, or gate definition
  moved.** `rank_cap` is still 8, `identity_tol` and
  `capability_eps` are still 0.25, the standing norm
  budget is still 0.9, `max_targets_per_delta` is still
  1, `allow_additive` is still false, `keep` is still 3,
  and `max_stack_depth` is still 8.
- **Depth is still 2, and the stack is untouched.** The
  audit overlaid the landed heads' weights through the
  existing apply path and wrote nothing; rollback,
  VERIFY, and SI-12's fail-closed squash refusal are
  exactly where SI-33 left them.
- **Glimmer is exactly where it was.** The seed is
  untouched, stored bytes still 0 against
  59,553,374,040 deduplicated bytes; Glimmer-scale deltas
  still escalate with checks 3/4 "not evaluated"; no
  forward path exists and none is faked (§9).
- **The real-base preconditions are exactly where they
  were.** SI-18 §3.7's A-1 to A-4 are design obligations
  SI-35 does not touch. What SI-35 changes about that
  candidate is its *context*, not its preconditions —
  §6 states the change precisely and takes no decision.

### 3.5 What is still not known — the unknowns that survive the audit

Stated as open questions a slice could retire — this
RFC invents no numbers for them (§15 carries them
forward):

- **Downstream transfer, per tensor.** For 12 of the 14
  in-stream matrices the transfer coefficient from a
  tensor's output perturbation to logit movement remains
  **unknown** — measured only for the readout and the
  final down-projection, by the landed deltas' records.
  The audit bounded and ranked; it did not, and by its
  registered method could not, manufacture transfer
  figures. Any future argument that needs one must
  measure one, on allowed populations, under its own
  pre-registration.
- **Whether the inertness map is a property of this
  construction seed or of the architecture.** The
  miniature's weights are drawn at small scale, and
  branch ratios are ratios within that one constructed
  model (RFC SI-34 §19 item 3, carried). The audit
  measures *this* model — the only one the gate has ever
  judged — and re-seeding the subject to generalise the
  measurement would change the subject every prior run
  measured. Unknown, and not an audit question.
- **The squash design for a mixed-target stack.** The
  three obligations RFC SI-34 §8 recorded — the carrying
  question, the four-axis gate context, the named-tensor
  tolerance verify — remain exactly as unargued as
  SI-34 left them. SI-35's bearing on them is indirect
  and is stated in §4: it changes whether depth will
  grow, not how a grown stack would be carried.
- **A third cycle's lawful configuration.** Both signals
  remain spent in the landed stack; re-training a landed
  signal at a deeper head, and the anchoring/target
  semantics at depth ≥ 2, remain undefined in the current
  semantics — §5 evaluates whether R2 gives that
  definition a subject to define itself *for*.

---

## 4. Route 1 (§8's squash trigger): transformer squash after R2

**Verdict up front: the design argument is RECOMMENDED
as SI-37, under RFC SI-34 §8's trigger (b) — as a
design-only RFC arguing obligations (i)–(iii), carrying
question first, with no implementation. Trigger (a) has
not fired, and R2 makes it *less* likely to fire, not
more; this section records both facts at full strength
before recommending the (b) route, because conflating
them is the one dishonest version of this candidate.**

RFC SI-34 §8 deferred transformer squash with a precise
trigger: squash becomes the recommended slice when
(a) a third authored delta lands, or the active authored
depth otherwise exceeds `keep`, **or** (b) a future RFC
argues obligations (i)–(iii) as its main business — the
carrying question first — and recommends the
implementation on that argued basis. SI-35's evidence
bears on the two arms very differently:

- **Trigger (a): not fired, and receding on the upstream
  line.** Depth is still 2 against `keep` 3 and
  `max_stack_depth` 8; no third delta has landed; and a
  squash on the only real stack would still be a no-op
  under `squash_bundle`'s standing semantics (depth 2 ≤
  `keep` 3), exactly as SI-34 §8 recorded. R2's specific
  contribution here is a *negative* derivation, labelled
  as such: the single-target upstream line was the only
  argued route by which a third authored delta was ever
  projected to land on this miniature (RFC SI-34 §5
  sequenced a second non-readout target behind the
  audit, and §9's third cycle had no other target source
  argued), and R2 closes that line at the standing
  budgets. A third delta is not thereby made impossible
  — the readout route exists and §5 discusses it — but
  no surviving argument in the record *projects* one.
  Any text that cites R2 as bringing squash's depth
  trigger closer has the derivation backwards.
- **Trigger (b): untouched by R2, and now the only arm
  left standing.** The three obligations are design
  questions about a stack that *already exists*, and
  R2 — a conclusion about what future single-target
  candidates can express — measures nothing about how
  the two landed deltas would be carried, gated, or
  verified as a composite:
  (i) *The carrying question.* The two landed deltas act
  on **different tensors** (`lm_head.weight`;
  `model.language_model.layers.1.mlp.down_proj.weight`).
  Demo squash composes same-tensor updates by truncated
  SVD into one single-target delta and gates it as such.
  A transformer squash of this stack must carry both
  tensors' folded updates in one composite record — a
  two-target delta — against policy
  `max_targets_per_delta` 1 and `allow_additive` false,
  which price authored candidates. Whether a *squashed*
  composite is an authored candidate for those budgets,
  a distinct record class with its own discipline, or
  inadmissible until the policy question is argued, has
  never been argued anywhere.
  (ii) *The gate context.* A squash candidate is judged
  by the full gate (demo squash: `SquashContext`, five
  checks, post-squash depth accounting). A transformer
  squash context must represent a stack whose members
  carry four recorded axes — signal, anchoring,
  composition configuration, authorship target — and
  whose members acted on different tensors; no such
  context exists, and SI-33's as-built asserts the
  fail-closed refusal on the real stack precisely
  because none does.
  (iii) *The tolerance verify over named tensors.* Demo
  squash verifies the squashed stack reproduces the
  unsquashed stack's behaviour within tolerance before
  gating; for deltas on different tensors the composition
  is per-tensor exact in the folded weights (a
  derivation), but the *behavioural* verify — through
  RMSNorms and the readout the two updates share
  downstream — is a forward-path measurement whose
  design (populations, tolerance, failure semantics) is
  unwritten.

**Why argue the obligations now, if depth pressure has
receded?** Three grounds, each stated at its true
strength. First, trigger (b) was written — before R2
existed — for exactly this posture: an RFC whose *main
business* is the obligations, argued before any
implementation is improvised inside another slice.
Second, squash is the only candidate in §§4–9 whose
subject matter R2 does not shrink: it operates on
already-landed, already-judged deltas, so its design
questions are as real at depth 2 as at depth 4, and its
answers would be needed by *every* future that grows
any stack — miniature or, under §6's operator decision,
otherwise. Third, the carrying question (i) is the same
*shape* as the representation question SI-32 §7 argued
for constructed candidates — whether a budget written
for authored candidates prices a composite the loop
itself constructs — and leaving both unargued while the
record is complete is how design obligations quietly
become permanent deferral (RFC SI-34 §19 item 6). The
counter-argument is recorded honestly: at depth 2 ≤
`keep` 3 the machinery, once designed, still would not
*run* on the only real stack, so SI-37 is a design
slice whose implementation remains deferred until depth
pressure or an operator direction makes it load-bearing.
That prices the recommendation correctly — design now,
implementation not claimed, not scheduled by this RFC —
and §11 scopes SI-37 to exactly that.

**What this section does not do.** It does not argue
obligations (i)–(iii) itself; that is SI-37's main
business, and compressing it into this RFC's survey
would be the improvisation trigger (b) forbids. It does
not recommend an implementation slice. And it does not
cite SI-33's stack as evidence for multi-target
*authorship* — the stack is two lawful single-target
deltas in sequence; the squash carrying question is
about already-judged deltas, argued on its own terms
(§8 keeps the two questions separate).

---

## 5. Route 2 (§9's semantics): a third authored cycle after R2

**Verdict up front: DEFERRED — R2 strengthens the
deferral for the upstream variant, leaves the readout
variant exactly as undefined and unmotivated as RFC
SI-34 §9 found it, and supplies no lawful configuration
for either. A semantics RFC for a third cycle is not
recommended, because on the present record it would be
arguing definitions for a run that has no lawful signal,
no argued target, and no question its answer would
retire.**

RFC SI-34 §9 deferred a third authored cycle on
definitional grounds, before any empirics: both signals
are spent (the registry serves exactly
`tiny-successor-lean-001` and
`tiny-predecessor-lean-001`; the landed stack consumed
signal 1 at cycle 1 and signal 2 at cycle 2); re-training
a landed signal at head `d0002` is a configuration no run
on record defines; authoring a third signal is unargued
on any ground independent of the gate (signal authoring
argued backward from the obstacle the gate measured is
the rejected signal search); and the anchoring and
target-source semantics at depth ≥ 2 are undefined —
what "base-anchored" targets mean for a signal already
landed once has no record to cite. SI-35 changes none of
those definitions. What it changes is the *target* side
of the hypothetical cycle, and the change must be stated
separately for the cycle's two conceivable variants,
because R2's scope is upstream-only:

- **The upstream variant is closed, not merely
  undefined.** A third cycle whose candidate acts on a
  single non-readout tensor was the configuration RFC
  SI-34 §5 sequenced behind the audit's shortlist. The
  shortlist is empty: under R2's registered consequence,
  no such tensor can be argued, from measured branch
  scale and derived bounds, as expressing the signal
  family's correction at the standing budgets. §3.3's
  head-stability result removes the one escape a
  semantics argument might have reached for — that
  qualification could differ at the deeper head `d0002`
  where the cycle would train: the qualifying set is
  empty at `d0002` as at `base` and `d0001`, scoped to
  those three heads. A depth-≥2 semantics RFC written
  for this variant would therefore be defining anchoring
  and target-source rules for a candidate class the
  measurement record has already closed — arguing in a
  vacuum in the literal sense: its definitions would
  have no lawful object to attach to.
- **The readout variant is untouched by R2 — and was
  already deferred without it.** R2 is scoped upstream;
  the readout target expresses the correction fully (the
  SI-25 control's held-out residual
  1.043608588752459e-05 is the standing measurement), so
  expressibility does not close a readout third cycle.
  What closes it as a *next slice* is everything RFC
  SI-34 §9 already recorded, unaltered by SI-35: no
  lawful signal exists for it (both spent; a third
  unargued; re-training a landed one undefined), and the
  composition record that would greet its candidate is
  complete and closed — six readout cycle-2 runs, the
  composition line falsified at three strengths at the
  binding probe (SI-27 drift 0.32509003952145576, SI-29
  drift 0.31215621903538704, SI-31 drift
  0.4775387533009052, all REJECT, §1's table) — with no
  new composition argument in SI-35's record, which
  measures no movement pair. Scoping a semantics RFC to
  the readout "because R2 leaves it open" would mistake
  R2's silence for a motivation: the variant's problem
  was never expressibility, and SI-35 supplies no new
  evidence in its favour.

Nothing here rejects depth in principle, and the
standing position is carried unchanged: nothing speaks
against more than two cycles someday, or against signals
authored for reasons independent of the gate. What would
retire this deferral is also unchanged in shape and is
stated so a future RFC can cite it: a signal argued on
grounds independent of the gate, *plus* — for the
upstream variant — a measurement record that reopens
single-target expressibility (a different miniature, a
different budget argued on its own merits under
invariant 8, or transfer measurements that change what
the bounds mean), *plus* the depth-≥2 semantics argued
as that RFC's main business. None of the three exists
today. Deferred.

---

## 6. Route 3 (§4's operator decision): a small real safetensors base

**Verdict up front: DEFERRED — and laid out here, as
RFC SI-34 §4 laid it out, as a decision that belongs to
the operator (Mark), not to this RFC.** Naming a real
base requires a checkpoint download; this scope forbids
downloads; and no slice recommended here proposes one,
schedules one, or treats one as approved. This section's
job is the narrower one the tasking for this RFC
assigns: to state the decision cleanly for the operator
— what it would cost, what its preconditions require,
and what SI-35 changes about its context — without
converting any of that into a recommendation that the
decision be taken, or a schedule for taking it.

**The preconditions, unchanged (SI-18 §3.7, audited in
every successor RFC since).** All four remain unmet, and
SI-35's evidence touches none of them:

- **A-1.** A feasibility note with real measurements for
  one named candidate base family: forward wall time,
  assembled memory, verification cost, on this VM.
- **A-2.** A binding design for `reference-v1` bases,
  with its tamper analysis.
- **A-3.** A capability-suite design for a *trained*
  base: what behaviour is the floor under, and how are
  its baselines recorded without a construction?
- **A-4.** An executor design scoped to that one family,
  with SI-15-grade numeric proof obligations.

A-2, A-3, and A-4 are design obligations, not download
obligations; none of them exists; and the local presence
of the Glimmer checkpoint (the SI-8 seed's referenced
source) does not retire any of them, exactly as RFC
SI-34 §4 recorded. A-1 cannot even be *measured* without
naming a family, and naming one is the operator's
decision itself, not a preliminary to it.

**What the decision would cost, stated as categories —
this RFC invents no figures for any of them.** (i) A
checkpoint download and the storage to hold it, under
the never-duplicate-bytes rule (§2 invariant 2) — the
seed pattern imports by reference where it applies, and
whether it applies to a newly named base is part of
A-2's design, not an assumption. (ii) The design work of
A-2 to A-4, each a slice-scale obligation in its own
right, in an order A-1's measurements would inform.
(iii) A capability floor for a trained base (A-3) whose
baselines cannot be recorded the way the miniature's
were — the miniature's suites are construction-relative,
and a trained base has no construction to be relative
to. (iv) The standing scope discipline's price, paid in
every prior RFC and owed here too: CPU only on this VM,
no GPU, no cloud, no spend — an executor design (A-4)
must be honest about what that discipline costs a real
family's forward path, with SI-15-grade numeric proof,
before any base is named as buildable.

**What SI-35 changes about the context — one paragraph,
in both directions.** In one direction: the miniature's
single-target upstream-authorship line is now
expressibility-closed at the standing budgets (§3), the
readout composition line was already closed (SI-32 §4),
and the third-cycle route is deferred on definitions
with its upstream variant closed (§5) — so *if* the
operator wants the authorship question itself to
continue, the miniature no longer offers an argued route
for it, and a real base is the only direction in the
record where that question remains open in principle.
In the other direction, with equal weight: SI-35's
closure is scoped to *this* constructed miniature and
*this* signal family's correction scale, and the record
contains no evidence in either direction about whether
a real base's branches are more or less inert per branch
(§14, §15; RFC SI-34 §4 made the same point from
SI-33's evidence) — the audit cannot be cited as
evidence that a real base would behave better, only as
evidence that this miniature, at these budgets, has been
measured to the end of its argued authorship routes.
Moving the question to a real base *before* that
measurement existed would have carried an unanswered
question to a more expensive room (RFC SI-34 §4); the
measurement now exists and is negative for the miniature,
which is precisely the condition under which laying the
decision before the operator — cleanly, with its costs
and preconditions attached and no thumb on the scale —
is this RFC's job. The decision, its timing, and the
naming of any base stay the operator's. This RFC does
not recommend the candidate as buildable now, does not
schedule it, and proposes no SI-37 work toward it (§11's
recommendation lies elsewhere, on autonomous ground).

---

## 7. Candidate (d): the representation question — does R2 fire its trigger?

**Verdict up front: ARGUED in SI-32, trigger EVALUATED
in SI-34 against SI-33 and again here against SI-35 and
found NOT FIRED — DEFERRED, unchanged; `rank_cap` stays
8 for trained and constructed candidates alike, and
SI-31's candidate is not re-judged.**

SI-32 §7.5's trigger is precise: the representation
question becomes arguable as a *change* when and only
when a candidate exists that is measured passing the
identity arm and the capability arm while failing *only*
on the `size_budgets` rank line. The evaluation against
SI-35's record is shorter than SI-34 §7's evaluation
against SI-33's, because SI-35 authored no candidate at
all:

- **SI-35 supplies no candidate of any kind.** It is a
  measurement-only slice (§14 item 27): its figures are
  inputs to a future argument, never evidence that a
  candidate would pass or fail any gate line. A trigger
  defined over a candidate's gate figures cannot be
  fired by a slice that performed no gate evaluation —
  any more than R1, had it been measured, would have
  approved a target (§3.2).
- **The standing candidates' positions are unchanged.**
  SI-33's candidate is rank 8 and fails on no line;
  SI-31's constructed candidate still fails identity
  independently at ~1.91× the tolerance
  (0.4775387533009052 against 0.25; *derivation* from the
  cited figures, as SI-32 §7.4 labelled it) beside its
  rank failure. No verdict on record turns on rank
  alone.
- **R2 does not smuggle the question back in through the
  budget qualifier.** R2's conclusion is scoped to
  rank-8 candidates because rank 8 is the standing cap;
  §8 evaluates the proposal to escape R2 by re-pricing
  that cap and rejects it there on invariant-8 grounds.
  The representation question's own route to a change
  remains its trigger, and only its trigger.

The trigger stays exactly as SI-32 §7.5 wrote it,
including its anti-retroactivity clause: a future rank
rule, if one is ever argued on a candidate that earns
it, never re-judges SI-31's candidate, whose verdict
stands on both lines permanently. Until the trigger
fires, any text that describes the rank line as "the
reason SI-31 failed" without the identity line beside
it is a labelling defect (§14 item 25) — and any text
that describes SI-35's R2 as bearing on the rank
question, in either direction, without noting that
SI-35 judged no candidate is the same defect's mirror
image (§14 item 26's rule, applied to a measurement
record under item 27).

---

## 8. Candidate (e): multi-target and budget-shaped escapes from R2

**Verdict up front: both REJECTED as next slices — the
first carried from SI-34 §10 with R2's bearing stated
precisely, the second rejected on invariant 8 — and
neither is reopened by being stated here. These are the
two candidates R2's *scope* genuinely invites a reader
to propose, so this RFC owes each a real evaluation
rather than a citation of its qualifier as if the
qualifier were an argument.**

### 8.1 Multi-target authorship: R2's silence is real, and it is not evidence

The honest observation first, at full strength: R2's
registered consequence is scoped **single-target**, and
that qualifier is load-bearing. The audit's pair was
applied per matrix, against per-branch ratios; nothing
in SI-35 measures what a candidate carrying *two*
upstream targets — say, one matrix in each attention
branch — could express jointly, and the branches' ratios
do not combine under any rule the audit registered (a
joint bound would need the downstream-transfer figures
§3.5 records as unknown for 12 of 14 matrices). A reader
who says "R2 does not close multi-target authorship" is
therefore *correct about the derivation's scope*, and
any successor text that quotes R2 against a multi-target
proposal without this paragraph has overstated R2 —
§14 item 28 forbids exactly that.

The candidate nevertheless stays rejected, on the
grounds that actually decide it, which are policy and
evidence grounds rather than expressibility grounds:

- **It buys its premise with a policy loosening.**
  Multi-target authored candidates require loosening
  `max_targets_per_delta` 1 / `allow_additive` false —
  the standing rejection of SI-32 §12.2, carried by
  SI-34 §10: zero authored-behaviour evidence exists
  under those defaults loosened, and the cycle-2 record
  is not evidence *for* loosening them. R2 adds no such
  evidence; it is a measurement record about
  single-target expressibility (§14 item 27).
- **SI-33's stack is not evidence for it.** The landed
  stack is a mixed-target *stack* — two lawful
  single-target deltas, each authored in its own cycle
  and judged by the unchanged gate in its own right —
  not a multi-target *delta*. The distinct question of a
  *squashed composite* carrying two targets is §4's
  obligation (i): a carrying question about
  already-judged deltas, which SI-37 argues on its own
  terms and which does not loosen any authorship policy
  by being argued.
- **Its expressibility premise is unmeasured in the
  direction the candidate needs.** The candidate's hope
  is that two branches' contributions combine toward the
  correction; the audit's record offers the branch ratios
  (largest 0.0340859) and unknown transfer coefficients,
  and nothing else. Arguing a policy loosening from an
  unmeasured hope is the exact shape §5 of SI-34 refused
  for the second single target — inverted: there,
  measurement had to *precede* the argument; here, the
  argument would precede any measurement that could
  support it, and the measurement it needs (joint
  transfer) is not SI-35's record.

Rejected as a next slice, carried, not reopened. R2's
single-target qualifier stays attached to R2 (§14 item
28); this candidate's rejection stands on its own
grounds and borrows none of R2's authority.

### 8.2 Budget-shaped escapes: raising the norm budget, the rank, or the correction's price

The second invited proposal runs: R2 is scoped
budget-0.9 and rank-8, so re-price a budget — a larger
norm budget raises every B(W) linearly (a *derivation*:
B(W) = budget × max input norm, so the down-projections
reach 0.212 only at a budget of ~3.3, roughly 3.7× the
standing 0.9; *derivation* from SI-35's cited bounds) —
and the closure dissolves. The arithmetic is correct
and is stated here so the rejection cannot be mistaken
for an arithmetic error. The rejection is invariant 8's,
in its original form: **no tolerance — and no cap — is
a design variable.** A budget re-priced after it binds,
in a document whose author knows the candidates it
would have affected, is the gate-fitting shape, whether
the re-pricing is of `identity_tol` after a drift
failure or of the norm budget after an expressibility
closure. SI-34 §12 rejected the mirror-image proposal (a
minimum-movement floor, repricing the judge in the
failing direction after a pass) on the same ground, and
this RFC holds both directions, as §2 invariant 8
states them. Two further facts price the proposal
honestly even on its own terms: the branch-ratio clause
does not move with the norm budget at all — it is a
measured property of the model's branches, not of any
budget — so the proposal must *also* re-register R1's
0.04 threshold or abandon the pair, i.e. it is not one
re-pricing but a re-founding of the audit's readings
after their outcome is known (§14 item 21); and the
rank half of the proposal is §7's representation
question, whose trigger has not fired. Rejected. The
standing budgets stay: norm budget 0.9, rank 8,
`rank_cap` 8, tolerances 0.25.

---

## 9. Candidates carried closed or deferred — R2 supplies no new evidence

**Verdict up front: every candidate in this section
keeps exactly the status SI-34 left it in. SI-35 is a
measurement of branches, activations, and bounds; it
trains nothing, composes nothing, partitions nothing,
and contacts no probe and no Glimmer artifact — so for
each candidate below, this section records the absence
of new evidence rather than re-arguing a settled
evaluation.**

- **The composition line, the penalty axes, partial
  removal: CLOSED / DEFERRED as carried (SI-32 §§4, 6,
  8; SI-34 §11).** The mean penalty (SI-27), the
  worst-case penalty (SI-29), and full-removal projection
  (SI-31) were each falsified at the binding probe at
  increasing strength, about *readout movement pairs
  through the shared fold*. SI-35 measures no movement
  pair — upstream or readout — so it neither reopens the
  line nor extends its closure, exactly as SI-34 §11
  recorded for SI-33. The λ / λ_w / k / τ axes stay
  deferred as open questions with no sweep licensed, and
  partial removal stays rejected on its two independent
  grounds (rank at every β > 0; interpolation between two
  aligned, REJECTED endpoints — derivations, labelled as
  such where they were made).
- **Per-probe anchoring; probe-partitioned sequencing;
  magnitude discipline as a training dial: REJECTED as
  carried (SI-32 §§9–11; SI-34 §12).** The first two
  reprice or read the judge (invariant 9); SI-35's audit
  is itself their lawful residue done honestly —
  measuring, on allowed inputs, what each tensor can
  express, published as a record — and supplies no
  evidence for the rejected forms. Magnitude stays
  adopted as *accounting* (candidate norm and ‖m2‖ beside
  every figure, §14) and rejected as a *dial*.
- **A minimum-movement floor: REJECTED as carried
  (SI-34 §12).** SI-35 does not revive it; if anything,
  the audit completes the argument against needing one —
  the project now *has* the expressibility map whose
  absence the floor proposal was reaching for, and a
  project that knows its map recognises a near-identity
  candidate by labelling (§14 item 26), not by
  re-pricing its judge.
- **A differently-seeded or enlarged miniature:
  NOT A CANDIDATE, recorded because R2 invites it.**
  R2 is scoped to *this* miniature (§3.5): a new
  constructed subject — re-seeded, widened, deepened —
  would have a different inertness map, and measuring it
  is mechanically easy. It is not a frontier: every prior
  run's record is a record about *this* subject, the
  gate has judged no other, and re-founding the subject
  because its measurements closed a line is choosing the
  subject by the verdict it returns — the subject-level
  form of the gate-fitting shape invariant 8 forbids.
  A future RFC that wants a new subject must argue it on
  grounds independent of R2's outcome; none is argued
  here.
- **Glimmer probing: REJECTED, not reopened (SI-13
  §7/§9; SI-34 §13).** SI-35 changes nothing: the audit
  is a miniature measurement that claims nothing at
  Glimmer scale, including by analogy and including the
  direction "real models are less inert", for which this
  record contains no evidence in either direction. The
  Glimmer seed is untouched; its deltas still escalate
  with checks 3/4 "not evaluated"; no forward path exists
  and none is faked. The standing pressure is recorded
  honestly, as every successor RFC has recorded it: the
  miniature authorship line closing (§10) renews the
  "why not Glimmer" question in a new form — *if not the
  miniature, then the seed* — and the answer does not
  soften, because a closed miniature line is not a
  Glimmer forward path, a probe suite, or a gate reading
  beyond escalation, and none may be improvised from
  SI-35's figures.

---

## 10. Decision summary

| Candidate | Verdict | One-line reason |
|---|---|---|
| Route 1 — transformer squash, trigger (a) (depth) | **NOT FIRED** | Depth is still 2 ≤ `keep` 3; no third delta has landed, and R2 closes the only argued upstream route by which one was projected to land (§4). |
| Route 1 — transformer squash, trigger (b) (argue the obligations) | **RECOMMENDED as SI-37 — design only** | The obligations (carrying, four-axis gate context, named-tensor verify) concern the stack that already exists, are untouched by R2, and were pre-registered as an RFC-first route; implementation stays deferred (§§4, 11). |
| Route 2 — a third authored cycle / depth-≥2 semantics | **DEFERRED** | No lawful configuration exists: both signals spent, semantics undefined — and R2 closes its upstream variant outright while its readout variant was already unmotivated on the closed composition record (§5). |
| Route 3 — a small real safetensors base | **DEFERRED — operator decision** | SI-18 §3.7 preconditions A-1 to A-4 all still unmet; naming a base requires a download this scope forbids and a decision this RFC explicitly does not take, schedule, or recommend as approved (§6). |
| (d) The representation question (rank / identity-alone judging) | **ARGUED (SI-32) — trigger EVALUATED, NOT FIRED — DEFERRED** | SI-35 judged no candidate; SI-33's fails on no line; SI-31's still fails identity independently at ~1.91× tol; no verdict on record turns on rank (§7). |
| (e) Multi-target authorship | **REJECTED as a next slice (carried)** | R2's single-target scope genuinely leaves it unmeasured — stated, not hidden — but it still buys its premise with a policy loosening and zero evidence under it; SI-33's mixed-target *stack* is not a multi-target *delta* (§8.1). |
| (e) Budget-shaped escapes (norm budget / rank / thresholds) | **REJECTED** | The arithmetic works and is stated; re-pricing a budget — or re-registering R1's thresholds — after the outcome is known is the gate-fitting shape invariant 8 forbids, in either direction (§8.2). |
| Composition line / penalty axes / partial removal | **CLOSED / DEFERRED as carried** | SI-35 measures no movement pair, so it neither reopens the closed readout line nor extends its closure (§9). No sweep is licensed. |
| Per-probe anchoring; sequencing; magnitude as a dial; a minimum-movement floor | **REJECTED as carried** | Each reprices or reads the judge and gains no evidence from SI-35; the audit itself is the lawful residue of the sequencing idea, done as measurement (§9). |
| A differently-seeded or enlarged miniature | **NOT A CANDIDATE** | Choosing a new subject because this subject's measurements closed a line is the subject-level gate-fitting shape; a new subject needs grounds independent of R2 (§9). |
| Glimmer probing | **REJECTED** | Rejected at SI-13 §7/§9 and restated in every successor RFC; a closed miniature line is not a Glimmer forward path, and none is faked (§9). Not reopened. |

The honest shape of the record after SI-35 deserves one
paragraph of its own, because the table's neatness could
hide it. The project has now run **eight** routes in the
authorship programme — a new signal under settled
semantics (SI-23), the same signal restacked (SI-21),
new target semantics (SI-25), a mean composition term
(SI-27), a worst-input composition term (SI-29), a
candidate constructed by full removal of the indicted
component (SI-31), a first non-readout target (SI-33),
and a measurement of every tensor's expressibility
(SI-35) — judged or registered under the same unchanged
accounting, and the eighth route was not an attempt to
land a write at all: it was the programme pricing its
own remaining room, and finding, at the standing
budgets, that the room is measured and closed for the
single-target upstream line while the readout line was
already closed on composition. That is a *completion*,
not a collapse: the miniature has yielded a complete,
honest map of what its authored self-writes can and
cannot do, which is what a constructed subject is for.
What remains genuinely open divides cleanly by who owns
it. The operator owns the scale question (§6): whether
the authorship programme continues on a real base, at
real cost, under preconditions A-1 to A-4. The project
owns one design question that needs no one's download
and no new measurement (§4): how the stack it already
landed would be carried if it ever had to be squashed.
§11 recommends that question, and only that question, as
SI-37 — and records, as this RFC's plain conclusion,
that **after SI-37 the autonomous miniature line has no
further argued slice on the present record, and
continuing it past that point is the operator's
decision to make, not a cadence's momentum to spend.**

---

## 11. The recommended slice: SI-37 (outline)

**SI-37 — Transformer squash, design only: an RFC
arguing the three design obligations RFC SI-34 §8
recorded for the landed mixed-target stack — (i) the
carrying question first, then (ii) the four-axis squash
gate context, then (iii) the tolerance verify over named
tensors — against the stack that exists (`d0001` +
`d0002`), with a stated implementation trigger, and with
no code, no policy change, and no squash performed.**

In scope:

1. **The carrying question, argued as the main business
   and decided in the document.** For the landed stack —
   `d0001` on `lm_head.weight`, `d0002` on
   `model.language_model.layers.1.mlp.down_proj.weight` —
   argue, against the standing policy
   (`max_targets_per_delta` 1, `allow_additive` false,
   written for authored candidates), which of the three
   recorded readings a squashed composite falls under:
   an authored candidate for those budgets; a distinct
   record class with its own discipline (the shape of
   SI-32 §7's representation question, in a second
   domain — argued against SI-32 §7.3's stack-growth and
   second-judge objections, not merely cited beside
   them); or inadmissible until a policy question is
   argued separately. The argument must state what each
   reading would price (per-tensor folded updates,
   ranks, aggregate norm, byte size) and must not resolve
   the question by relabelling: a composite that carries
   two tensors' updates *is* two-target in its carrying,
   whatever class it is assigned to, and the document
   must say so.
2. **The gate context, designed to the four recorded
   axes.** Specify — as design, in the shape of demo
   squash's `SquashContext` (five checks, post-squash
   depth accounting) — a transformer squash context that
   represents a stack whose members carry signal,
   anchoring, composition configuration, and authorship
   target, whose members acted on different tensors, and
   whose members may include a construction (none landed
   is one; the obligation is prospective, recorded in
   SI-32 §13.1 and carried by SI-34 §8). Each of the five
   checks must be named with its transformer meaning or
   recorded as having none, with the reason — no check
   may be carried over by analogy without its meaning
   stated.
3. **The tolerance verify over named tensors, designed
   as a measurement.** Specify the behavioural verify —
   populations (allowed inputs only; no probe input,
   probe logit, or probe-derived quantity may enter the
   verify's design, invariant 9, applied to a squash
   context exactly as SI-35's audit applied it to a
   measurement), heads, position statistic, tolerance and
   its derivation, and failure semantics (fail closed:
   an unverifiable squash is refused, never passed) —
   for a composition that is per-tensor exact in the
   folded weights (a derivation) but whose behavioural
   equivalence passes through RMSNorms and the readout
   the two updates share downstream. The design must
   state what the verify would measure on the *existing*
   depth-2 stack if it were run, while performing no
   squash and claiming no verify result.
4. **The implementation trigger, stated precisely.**
   Building on item 1–3's argued basis, state the
   conditions under which an implementation slice
   (SI-38 or later, never SI-37 itself) becomes
   recommendable: a third authored delta lands or the
   active authored depth otherwise exceeds `keep` 3
   (RFC SI-34 §8 trigger (a), restated); **or** the
   operator directs the work; **or** the argued design
   itself demonstrates a present-tense need on the
   depth-2 stack that this RFC has not foreseen — with
   the burden on that demonstration stated. Until a
   trigger fires, SI-12's fail-closed refusal stands,
   and the design's existence is not itself a trigger.
5. **Deliverable and its limits.** SI-37 ships exactly
   one document: the RFC described in items 1–4. It
   ships no production code, no test, no manifest or
   schema change, no policy default change, and no CLI
   behaviour change; `si squash` continues to fail
   closed on transformer stacks throughout, asserted by
   the existing suite's continued exact reproduction.

---

## 12. Acceptance criteria for SI-37

Stated for a design-only slice, in the shape of RFC
SI-34 §16's criteria as they apply to a document: each
criterion is checkable against the shipped RFC itself
and the untouched repo, without trusting this RFC's
intentions.

1. **Exactly one deliverable, doc-only** — the slice's
   diff against its base contains exactly one new file,
   the SI-37 RFC document, and no modification to any
   existing file: no production code, no test, no
   manifest schema, no policy default, and no gate
   behaviour changes.
2. **The carrying question is argued and decided** —
   item 1 of §11 is addressed as the document's main
   business, in its own section, before the gate-context
   and verify designs that depend on it; all three
   recorded readings are stated at full strength; the
   document's chosen reading is named, with the
   stack-growth and second-judge objections (SI-32 §7.3)
   answered for it, not cited past; and no part of the
   argument treats the landed stack as evidence about
   multi-target *authorship* (§8.1's separation is
   preserved verbatim in effect).
3. **The gate context is complete against the record** —
   all five demo-squash checks are named, each with its
   transformer meaning stated or its absence recorded
   with the reason; the context represents all four
   recorded axes (signal, anchoring, composition
   configuration, authorship target) and the
   mixed-tensor and constructed-member cases, or records
   a case as undesigned with the reason — never silently
   omitted, never rendered as designed when it is not.
4. **The verify is designed as a measurement, with its
   populations lawful** — populations, heads, statistic,
   tolerance with its derivation, and failure semantics
   are all stated; no probe input, probe logit, or
   probe-derived quantity appears anywhere in the design;
   and the document states what the verify would measure
   on the existing depth-2 stack without claiming any
   verify result, squash result, or behavioural
   equivalence as measured.
5. **The implementation trigger is stated and is not
   self-firing** — §11 item 4's conditions appear in the
   document substantially as registered here, including
   the clause that the design's existence is not itself
   a trigger; SI-12's fail-closed refusal is stated as
   standing throughout and after SI-37.
6. **R2 is quoted only with its qualifiers** — wherever
   SI-37 cites SI-35's closure (its depth projections
   depend on it), the citation carries the registered
   scope: single-target, upstream, rank-8, budget-0.9,
   this miniature, this signal family's correction scale,
   derivation-scoped (§14 item 28); and wherever SI-33's
   ADMIT is summarised, its attribution travels with it
   (§14 item 26), and no SI-35 figure is quoted as a gate
   figure or a verdict prediction (§14 item 27).
7. **Untouched scope, verified** — the full existing
   suite is green on the slice's own run, with its real
   numbers reported (the SI-35 merge's baseline: 394
   passed, 2 skipped; a design-only slice changes no
   test's result); `si squash` still fails closed on the
   landed transformer stack (existing suite); `si serve`
   still returns 501 for every transformer bundle
   (existing suite); the Glimmer seed is untouched; no
   download was performed, scheduled, or implied.
8. **No implementation is claimed or smuggled** — the
   document contains no code, no schema, and no
   instruction that a successor slice could execute
   without its own RFC-stage argument on any point
   SI-37 itself leaves open; every obligation SI-37
   cannot argue is recorded as open, with what would
   retire it (§15's discipline, applied inside SI-37).

---

## 13. Non-goals

Explicit, and binding on SI-37 and on any slice this RFC
is cited to justify:

- **No second target, no target RFC.** The shortlist is
  empty (R2 MEASURED, §1); exactly zero new
  authorship-target values are proposed by this RFC, and
  the axis keeps serving exactly `"lm-head"` and
  `"final-mlp-down"`. Any text describing a tensor as
  shortlisted, approved, or next under this RFC is a
  labelling defect.
- **No candidate, no training, no decisive run, no third
  cycle.** This RFC authors nothing and recommends no
  slice that authors anything. Both signals' ids,
  streams, corrections (scale 0.15 included), and
  partitions are byte-for-byte as SI-19/SI-23 defined
  them; no third signal is authored, argued, or searched
  for (§5).
- **No squash implementation.** SI-37 is a design RFC
  (§11); SI-12's fail-closed refusal stands throughout
  it and after it; no squash is performed on any stack,
  real or synthetic, under this RFC.
- **No real-base work, no download.** Candidate route 3
  is deferred with audited preconditions and an
  operator-owned decision (§6); this RFC builds nothing
  toward it, names no base or family, and performs,
  schedules, and implies no download.
- **No gate, policy, rank, budget, or tolerance change —
  in either direction.** `rank_cap` stays 8 (§7); the
  norm budget stays 0.9; `identity_tol` and
  `capability_eps` stay 0.25; `max_targets_per_delta`
  stays 1 and `allow_additive` stays false (§8.1); no
  minimum-movement floor is built (§9); SI-31's
  candidate is never re-judged.
- **No composition revival.** No penalty term (mean,
  worst-case, or any third statistic), no projection
  (full or partial), and no blend of any of them with
  any other change is built, run, or reported under this
  RFC (§9). The four existing composition
  configurations' semantics are untouched.
- **No new subject.** No re-seeded, widened, deepened,
  or otherwise re-founded miniature is proposed,
  measured, or argued (§9).
- **No Glimmer anything.** No Glimmer forward path,
  probes, training, authorship, audit, or squash (§9).
  The Glimmer seed is untouched except to restate that
  it is untouched.
- **No probe contact in any recommended design.** No
  probe input, probe logit, probe movement, or
  probe-derived quantity enters SI-37's verify design or
  any other design this RFC recommends (§11 item 3).
- **No autonomous selection — and no autonomous
  continuation past SI-37.** Every choice in SI-37 is a
  recorded, driver-level design argument. And per §10's
  plain conclusion: this RFC registers no SI-38. What
  follows SI-37 is either an implementation trigger
  firing (§11 item 4) or the operator's decision (§6) —
  not a ninth miniature route invented to keep the
  cadence fed.
- **No serving, no CLI wiring.** Transformer bundles
  keep returning 501; SI-37 adds no CLI surface of any
  kind.
- **No scale claim of any kind.** SI-35's figures, R2's
  closure, and this RFC's evaluations are facts and
  derivations about a constructed, deterministically
  seeded, not-trained miniature, at three named heads,
  on two constructed signals' partitions. Nothing in
  them is claimed about any real model, any trained
  base, or Glimmer — including by analogy, and including
  the direction "real models are less inert", for which
  this record contains no evidence in either direction.

---

## 14. Honesty and labelling requirements for any future slice

Items 1–25 restate RFC SI-34 §18 (itself restating
SI-32 §18, SI-30 §17, SI-26 §17, and SI-24 §12 with
additions) in compressed form, with the numbering
preserved so existing citations resolve; items 26–27
restate RFC SI-34 §18's new items **verbatim in effect**,
as this RFC is required to carry them; item 28 is new in
this RFC. All twenty-eight bind SI-37 and any slice this
RFC is cited to justify.

1. **Training numbers are training numbers.** Losses,
   penalty, objective, gradient, and construction values
   are labelled with signal, partition, head, target
   source, anchoring, composition configuration, and
   authorship target; none is a gate or probe result.
2. **Measured vs not evaluated survives authorship.**
   "Not evaluated" is never rendered as 0.0, a pass, or
   an omission.
3. **Authorship is not admission.** No document, test
   name, or journal field describes a candidate as landed
   before the gate lands it.
4. **Every number carries its source.** Each figure names
   the run, configuration, and document that measured it;
   inherited figures are cited, not re-derived silently.
5. **The miniature's labels travel.** "Constructed,
   deterministically seeded, not trained" accompanies the
   miniature wherever its results are summarised.
6. **Failure states are reportable successes of the
   experiment.** A REJECT with attribution is a
   publishable result, stated as plainly as an ADMIT.
7. **Serialisation determinism is proven, not assumed.**
   Blob-hash equality is the evidence; near-equality is a
   failure, reported as one.
8. **Head-relative numbers name their head.** Any figure
   measured against a head names that head.
9. **Incremental and cumulative are different numbers.**
   Never presented as each other, in either direction.
10. **Composed artifacts do not inherit training claims.**
    A landed stack's properties are what its VERIFY
    measured, not what its cycles' training records hoped.
11. **Signal identity is part of every label.** A number
    without its signal id is a labelling defect.
12. **Diagnostics are not verdicts.** The diagnostic set
    attributes a verdict after the gate has spoken; it is
    not a gate check, it influences no verdict, and none
    of it is a prediction — including Phase-0 diagnostics
    on already-judged candidates that point the same way
    as a later decisive run.
13. **Recorded selection is not a choice the mind made.**
    Signal, anchoring, composition configuration, and
    authorship target are driver selections; no text
    describes the mind as choosing, deciding, or
    preferring any of them.
14. **A derivation is not a prediction.** Derived
    quantities (movement columns, sums at a probe,
    interpolation arguments, rank arithmetic,
    expressibility bounds like SI-35's) are labelled
    derivations wherever they appear, and are never quoted
    with the authority of a measurement.
15. **Anchoring is part of every label.** A training or
    diagnostic number recorded under one anchoring is
    never presented as a number under the other.
16. **Controls and treatments are never blended.** Each
    figure stands under its own run's labels; a three-,
    four-, five-, six-, seven-run, or audit table keeps
    the rows separate.
17. **Composition configuration is part of every label.**
    A number recorded under one composition configuration
    is never presented as a number under another.
18. **Allowed-input numbers are not probe numbers.**
    Figures from allowed populations and figures from
    judged populations are never interchanged, averaged
    together, or quoted in each other's place; any gap
    between them is stated as the difference between two
    separately labelled records.
19. **Sign agreement never stands alone.** It is published
    only beside the magnitude-weighted figure (the inner
    product) for the same population.
20. **A worst-case figure names its population.** A worst
    *allowed input* (train partition, held-out partition)
    and the worst *probe* are different quantities over
    disjoint populations, and no text may shorten either
    to "the worst case" without its population attached.
21. **Judgement-call readings are pre-registered.** Where
    a falsification reading turns on a qualitative boundary
    in a quantitative record, the boundary is stated in
    advance, in the record's own terms, before the decisive
    run it will judge — and the reading is then applied as
    written. A reading invented after the numbers are known
    is reported as an observation, never as the
    falsification test.
22. **A constructed candidate is labelled constructed.**
    A candidate produced by a construction step after
    training is labelled with that step wherever it, its
    stack, or its figures are described; its
    pre-construction intermediate's figures are labelled an
    intermediate's and are never presented as the judged
    candidate's record.
23. **An unmeasured-as-defined reading stays unmeasured.**
    A falsification reading whose registered definition
    did not fire — F2′, F2′′, and F2′′′ are the standing
    instances — is quoted in its registered form ("not
    measured as a reading", with the clause that failed
    named) and never rounded, in summary, citation, or
    successor documents, into "measured", into "not
    measured" without its qualifier, or into the separate
    facts measured beside it.
24. **Authorship target is part of every label.**
    A number recorded under one authorship target — for
    any slice this RFC is cited to justify, `"lm-head"` or
    `"final-mlp-down"` — is never presented as a number
    under another; figures from the six-run record are
    understood, and where summarised stated, as recorded
    under `"lm-head"`. Folded-update and weight-space
    figures for a non-readout candidate are published as
    weight-space figures only: no readout-fold derivation
    (of α, of movement factorisation, or of composition)
    is claimed for a candidate whose update acts upstream
    of the readout.
25. **A two-line refusal is reported on both lines.**
    A candidate the gate refuses on two independent lines
    — SI-31's projected candidate (safety rank and
    identity) is the standing instance — is described as
    failing on both wherever its verdict is summarised;
    quoting SI-32 §7.5's deferred representation question
    as softening, explaining away, or partially overturning
    SI-31's verdict is a labelling defect, because its
    identity failure is independent of the rank question
    and stands under any resolution of it.
26. **An ADMIT's attribution travels with the ADMIT.**
    A landed candidate's verdict is never summarised
    without the measured attribution its own slice
    published for it, where one exists. The standing
    instance is SI-33's cycle-2 ADMIT: any summary of it —
    in a document, a table, a PR body, or a report — that
    does not carry, in the same breath, its candidate's
    movement scale (‖m2‖ = 0.010469039280455053 at norm
    0.30801411790830574), its F3′′′ reading (measured: the
    destination traded away, residual
    0.0027387202915082178 against boundary
    2.087217177504918e-05), and its near-inert-branch
    attribution (final-MLP branch ≈ 0.4% of the residual
    stream on the measured population) is a labelling
    defect, because the ADMIT without them asserts the
    one claim SI-33's as-built explicitly declined to
    make. The rule is general, not SI-33-specific: it
    binds any future ADMIT whose slice publishes a
    qualifying attribution, in either direction.
27. **A measurement-only record is not a verdict record.**
    Figures produced by a slice that authors no
    candidate — audit figures, bounds, and qualification
    readings in SI-35's sense — are inputs to a future
    *argument*, never evidence that a candidate would
    pass or fail the gate, and never quoted beside gate
    figures without the distinction stated. A matrix's
    qualification under R1 is a statement about measured
    branch scale and a derived bound, on named
    populations at named heads; it is not a prediction
    that a candidate trained on that matrix would ADMIT,
    would reach its destination, or would compose — and
    a matrix's failure to qualify is not a prediction
    that its candidate would REJECT in the gate's terms
    (an inert candidate may well ADMIT, as SI-33's did;
    that is precisely the confusion this item exists to
    prevent).
28. **R2's qualifiers travel with R2 (new).** SI-35's
    R2 conclusion — the single-target upstream-authorship
    line is *expressibility-closed on this miniature at
    the standing budgets* — is a derivation-scoped
    conclusion from measured inputs, and is never quoted
    without its scope attached: **single-target,
    upstream, rank-8, budget-0.9, on this constructed
    miniature, for this signal family's correction scale
    (0.212)**. Quoting it as "authorship is impossible",
    as a gate verdict about any candidate (item 27), as
    closing the readout line (refused on composition, not
    expressibility), as closing multi-target authorship
    (§8.1 records that it does not measure it — whose
    rejection stands on its own policy and evidence
    grounds), or as evidence about any real model, any
    trained base, or Glimmer, in either direction, is a
    labelling defect. RFC SI-34 §19 item 4 named this
    misreading in advance; this item makes its prevention
    a standing requirement, in the same form items 26–27
    made theirs.

---

## 15. Risks and open questions

Each item names what would retire it. Numbers this RFC
deliberately does **not** supply are listed as open
questions to measure — supplying them here would be
inventing them.

1. **R2 will be misread as "authorship is impossible"
   unless its scope travels with it.** This is RFC
   SI-34 §19 item 4's risk, now realised as a standing
   exposure: the closure's five-plus qualifiers
   (single-target, upstream, rank-8, budget-0.9, this
   miniature, this signal family's correction scale) are
   each load-bearing, and the readout target already
   falsifies any unqualified version (the SI-25 control
   expresses the correction fully; it is refused on
   composition, not expressibility). *Retired by:*
   nothing — §14 item 28's discipline, applied in every
   successor document; the risk is recorded here because
   successor documents are where qualifiers go to die.
2. **The attention branches' near-miss will be misread
   as a near-qualification.** The largest branch ratio,
   0.0340859 against the registered 0.04, is ~85% of the
   threshold (*derivation* from the cited figures), and
   a reader will be tempted to call the layer-1 attention
   matrices "almost shortlisted". They are not: R1's
   pair was registered in advance, applied as written
   (§14 item 21), and 0.0340859 < 0.04 is a failure of the
   clause, full stop; re-thresholding after the figures
   exist is §8.2's rejected move in miniature. The honest
   statement the record supports is comparative only —
   the attention branches are the least inert branches in
   the model, and the MLP branches are an order of
   magnitude more inert — and it licenses no target, no
   shortlist, and no expectation. *Retired by:* nothing
   in SI-37; recorded so the near-miss is never promoted
   after the fact.
3. **Squash's design may be argued for a stack that
   never grows — and that is acceptable only if it is
   said.** §4's trigger-(b) recommendation spends a
   design slice on machinery whose implementation
   trigger may never fire, now that the upstream route
   to a third delta is closed. The defence is §10's: the
   carrying question is a live design debt about a real,
   landed stack, and leaving it unargued has its own
   cost (item 4). But if neither §11 item 4's triggers
   nor the operator's direction ever fires, the honest
   closing statement for the squash line is that it was
   designed and never needed at the depths the project
   reached — not that it was always about to happen.
   *Retired by:* §11 item 4's trigger, if it fires;
   otherwise by a future document recording the
   permanent deferral honestly.
4. **Design obligations can quietly become permanent
   deferral — in both remaining lines.** RFC SI-34 §19
   item 6 named this for squash; SI-35's closure extends
   it to the whole miniature programme. A-1 to A-4 (§6)
   and squash's (i)–(iii) (§4) are now the record's only
   unargued design obligations, and this RFC resolves
   them asymmetrically on purpose — squash's are argued
   next because they are autonomous; A-1 to A-4 wait
   because their first step is the operator's. The risk
   is that "waiting for the operator" and "argued and
   shelved" blur into a programme that is neither closed
   nor continued, merely continued *about*. §10's plain
   conclusion is the guard: after SI-37, no further
   autonomous slice is registered by this RFC.
   *Retired by:* the operator's decision (§6), whenever
   it is taken, in either direction — a decision to stop
   retires this item as completely as a decision to
   continue.
5. **The representation trigger may still fire someday —
   and must not be improvised around.** §7 found it
   unfired. If a lawful future candidate ever passes
   identity and capability while failing only on rank,
   SI-32 §7.5's question becomes a change proposal in its
   own RFC, argued against SI-32 §7.3 with the
   candidate's figures published, never applied
   retroactively to SI-31, and never resolved inside the
   candidate's own slice. *Retired by:* that RFC, if the
   trigger ever fires; until then this item stays open
   by design.
6. **Downstream transfer remains unknown for 12 of 14
   matrices — and R2's closure could be mistaken for
   having measured it.** The audit's bound clause is a
   local, pre-attenuation bound (§3.2); the closure rests
   on the *branch-ratio* clause, which no transfer figure
   would change (the ratio is measured at the residual
   stream itself), plus the bound clause for the two
   down-projections, whose landed-delta transfer *is*
   measured (SI-33's ‖m2‖ record). A successor that
   needs a transfer coefficient for any other tensor
   must measure one, on allowed populations, under its
   own pre-registration. *Retired by:* such a
   measurement, if a future argued slice ever requires
   one; not by SI-37, whose verify design (§11 item 3)
   concerns the landed stack's behaviour, not per-tensor
   transfer.
7. **Inertness may be a property of the construction
   seed, not of the architecture** (RFC SI-34 §19 item
   3, carried). The miniature's weights are drawn at
   small scale, and branch ratios are ratios within that
   one constructed model. The audit measures *this*
   model — the only one the gate has ever judged — and
   its readings, and R2 with them, are scoped to it
   (§13). Whether a differently-seeded miniature of the
   same architecture would show the same map is unknown;
   §9 records why re-seeding is not thereby licensed.
   *Retired by:* nothing on the present record; recorded
   so the closure's scope is not inflated after the fact.
8. **A mislabelled configuration is worse than a wrong
   one.** The failure mode the cadence has named since
   SI-26 — a candidate trained under one configuration
   and journaled under another — gains a measurement-era
   variant after SI-35: an audit figure measured on one
   population or at one head and quoted under another
   would corrupt every argument that cites the audit,
   silently and at one remove; and an R2 quotation that
   drops its qualifiers (item 1) is the same failure at
   the level of a conclusion. *Retired by:* §14 items 8,
   18, 27, and 28 — per-figure population/head labels
   and qualifier discipline — asserted wherever SI-35's
   record is cited.
9. **Depth and Glimmer pressure arrive with the
   miniature line's closure, in new forms.** At
   `max_stack_depth` 8 with transformer squash still
   unimplemented (§4), the landed stack's depth 2 is now
   likely the deepest the miniature reaches — which
   converts the compression question from urgent to
   archival, and SI-37's design must be honest that it
   designs for the record, not for a growth curve. And
   per the standing item in every successor RFC: a
   closed miniature line renews the "why not Glimmer"
   and "why not a real base" questions together. §6 lays
   the second before its owner; the first stays rejected
   (§9) — there is no Glimmer forward path, none is
   proposed here, and none may be improvised by a
   successor slice.
10. **The deferred axes' unknowns remain unknown.**
    The λ / λ_w / k / τ questions (§9) remain open by
    design: each is retired only by a later RFC arguing
    a single value from the published records of the
    configurations that now exist — seven cycle-2 runs,
    two landed deltas, one complete audit — never by a
    sweep. Partial removal stays unmeasured, deliberately
    (SI-32 §8, carried at §9): its grounds are a
    derivation and rank arithmetic, labelled as such, and
    SI-35's audit — which measures what *single* tensors
    can express — supplies no β evidence in either
    direction.

---

## 16. Summary

After SI-35, SI can originate a miniature delta, judge
it, land it, originate a second one on top of it under a
second signal, under two target anchorings, under either
of two composition-aware losses, under a construction
that removes the indicted component exactly, and under a
second authorship target acting upstream of the readout —
and it has now also **measured, for every weight matrix
in the miniature, the share of the residual stream its
branch contributes, the scale of the activations it
consumes, and the derived bound on what a rank-8 update
at the standing 0.9 norm budget could express through
it**, on allowed populations only, at `base` and at both
landed heads, with SI-33's published attribution as its
reproduction anchor. The audit's pre-registered readings
came back **R1 NOT measured, R2 MEASURED, R3 NOT
measured**: the branch-ratio clause (≥ 0.04) fails for
all 14 in-stream matrices — the largest measured branch
ratio on the primary cell is the final layer's attention
branch's 0.0340859, and the MLP branches measure
0.00329234 and 0.00347083 — the bound clause (≥ 0.212)
fails for exactly the two down-projections (0.0575995
and 0.0576504), the shortlist is **empty**, and the
single-target upstream-authorship line stands
*expressibility-closed on this miniature at the standing
budgets*, with its qualifiers — single-target, upstream,
rank-8, budget-0.9, this miniature, this signal family's
correction scale — attached, derivation-scoped, beside
SI-33's ADMIT carried only with its attribution (§1:
candidate norm 0.30801411790830574, ‖m2‖
0.010469039280455053, F3′′′ measured, the final-MLP
branch ≈ 0.4% of the residual stream) and beside the
seven-run cycle-2 record it does not alter (SI-27
0.32509003952145576 REJECT, SI-29 0.31215621903538704
REJECT, SI-31 0.4775387533009052 REJECT on safety rank
and identity, independently). Of the three routes RFC
SI-34 named for an RFC in R2's wake: transformer squash's
depth trigger (a) has **not fired** and R2 makes it less
likely to fire, closing as it does the only argued
upstream route to a third delta — while its RFC trigger
(b), arguing obligations (i)–(iii) as a future RFC's
main business, is untouched by R2 and is the route this
RFC takes (§4); a third authored cycle stays deferred,
its upstream variant now closed rather than merely
undefined and its readout variant exactly as unmotivated
as SI-34 §9 left it, there being no lawful signal for
either (§5); and a small real safetensors base stays
deferred as the **operator's decision**, its four SI-18
preconditions (A-1 feasibility measurements for one named
family; A-2 a binding `reference-v1` design; A-3 a
capability-suite design for a trained base; A-4 an
executor design for that family) audited and all still
unmet, laid out in §6 with its costs as categories and
SI-35's one-paragraph change of context stated in both
directions — and not recommended as approved, not
scheduled, and not recommended as buildable now. The
representation question's trigger is evaluated against
SI-35 and found **not fired** — SI-35 judged no
candidate, SI-33's fails on no line, SI-31's still fails
identity independently — so `rank_cap` stays 8 (§7);
multi-target authorship stays rejected, with R2's
single-target silence about it stated honestly rather
than borrowed as authority, and budget-shaped escapes
from R2 are rejected on invariant 8, their arithmetic
stated and correct and beside the point (§8); the
composition line, penalty axes, partial removal,
per-probe anchoring, sequencing, magnitude as a dial,
and the minimum-movement floor stay closed, deferred, or
rejected exactly as carried, SI-35 supplying no new
evidence for any of them, and Glimmer probing stays
rejected, not reopened (§9). **One slice is recommended:
SI-37, a design-only RFC arguing transformer squash's
three obligations for the landed mixed-target stack —
the carrying question first, then the four-axis gate
context, then the tolerance verify over named tensors —
with a stated, non-self-firing implementation trigger
and no code** (§§11–12). And this RFC's plain conclusion
stands beside its recommendation, not underneath it:
**the miniature authorship line is at a natural decision
point for the operator.** The miniature has been measured
to the end of its argued authorship routes; SI-37 is the
last autonomous slice this record argues for; and what
continues the programme after it — a real base under
§6's preconditions, an implementation trigger firing
under §11 item 4, or an honest stop — is a decision for
the operator to make, recorded here rather than taken.
The deepest landed authored stack is depth 2; the base
stays frozen, the bytes are never duplicated, no
tolerance, cap, budget, or policy default moves in
either direction, the gate is unchanged, and nothing is
claimed that a test in this repo cannot check.

---

*RFC SI-36 — design only. Implementation begins, if
approved, at SI-37 as outlined in §11. RFC SI-26 changed
what the loop accounts for while it trains; RFC SI-28
changed the statistic that accounting answers to; RFC
SI-30 changed the candidate itself, removing the
component the record indicted; RFC SI-32 changed where
an authored update acts, and the gate admitted the
result — with SI-33's own document telling the truth
about that admission in the same breath; RFC SI-34
changed the order of operations, measuring what every
tensor in the model can express before arguing that any
of them should be written to — and SI-35's measurement
came back with an empty shortlist and a closed line,
published as plainly as an ADMIT would have been. This
RFC proposes changing neither the accounting, nor the
statistic, nor the candidate, nor the target, nor the
budgets any of them answer to — but the programme's
centre of gravity: from arguing where the next write
could act, on a miniature whose every acting place has
now been priced, to arguing how the writes already
landed would be carried, and to laying the one remaining
authorship question — scale — before the operator who
owns it, with its costs and preconditions attached and
no thumb on the scale.*
