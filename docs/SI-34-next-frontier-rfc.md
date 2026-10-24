# SI-34 RFC — The Next Frontier After the First Non-Readout Target
(Design Only)

> **Status: DESIGN ONLY.** This document evaluates candidate
> directions and recommends one next slice. It changes no
> production code, no test, no manifest schema, no policy
> default, and no gate behaviour. Nothing in this RFC is
> implemented unless a section explicitly says it exists today,
> in code, on `main`.
>
> **The one sentence that governs everything below:** SI-33
> landed the project's first authored two-delta stack — cycle 2
> ADMITted to head `d0002` under the unchanged gate (identity
> drift 0.17195402272045612 ≤ tol 0.25, capability regression
> 0.001467481255531311 ≤ eps 0.25, §6.6 VERIFY measured-pass) —
> and its own as-built attributes that ADMIT, in measured
> terms, to the near-inertness of the branch the second delta
> acts through: the final layer's SwiGLU output norm ‖s‖
> ranges 0.040–0.064 against a residual stream of
> ‖h_mid‖ 1.97–2.02 on the measured population, the
> candidate's probe-space movement is ‖m2‖ =
> 0.010469039280455053 at candidate norm
> 0.30801411790830574, and the pre-registered completeness
> reading F3′′′ is **measured** — the admitted candidate's
> held-out residual is 0.0027387202915082178 against a
> boundary of 2.087217177504918e-05, ~131× short of its own
> destination — so this RFC must never be cited as evidence
> that upstream authorship composes, learns, or transfers,
> that a second non-readout target has been argued, that the
> representation question's trigger has fired, that a
> real-base download has been approved, or that any slice
> proposed here exists.
>
> **Scope discipline:** CPU only. No GPU, no cloud, no spend,
> no downloads. Terminology: this is machine learning
> infrastructure throughout.
>
> **Relationship to prior work:** RFC SI-32 evaluated the
> frontiers open after SI-31 and recommended the first
> non-readout authored target; SI-33 built it (PR #38,
> as-built in `docs/SI-33-final-mlp-down-target.md`) and the
> gate's cycle-2 verdict was a measured ADMIT — the first in
> the project's seven-run cycle-2 record — with RFC SI-32
> §15's falsification readings reported exactly as
> registered: F1′′′ **not measured** (both of its clauses
> fail), F2′′′ **not measured as a reading** (its transfer
> clause holds; its maximum clause fails), F3′′′ **measured**
> (the destination counts as traded away). This RFC is the
> successor RFC that cadence requires. It restates the
> verified state after SI-33 (§1), gives SI-33's evidence —
> what it changed and what it did not — the deep treatment
> (§3), evaluates the representation question's trigger
> against SI-33's candidate and finds it unmet (§7),
> evaluates transformer squash now that its precondition
> exists for the first time (§8), records the real-base
> candidate's unchanged status as an operator decision this
> RFC does not take (§4), and recommends exactly one next
> slice (§15): a measurement-only branch-contribution and
> expressibility audit of the miniature, because SI-33's
> decisive unknown is no longer *where an update acts* but
> *how much any place an update could act can move this
> model at the standing budgets* — a quantity SI-33 measured
> for exactly one branch, after the fact, and which every
> remaining authorship candidate needs measured *before*
> the fact. SI-32's rejections of additive / multi-target
> authorship, of per-probe anchoring, probe-partitioned
> sequencing, and magnitude discipline as a training dial,
> and of Glimmer probing, are carried forward, not reopened:
> §§10–13 record, per candidate, that SI-33 supplies no new
> evidence that reopens any of them, and they stay closed.
> The SI-9 RFC's §7 risks remain in force and are inherited,
> not relitigated.

---

## Table of contents

1. [Where SI stands after SI-33](#1-where-si-stands-after-si-33)
2. [Inherited invariants](#2-inherited-invariants)
3. [What SI-33 actually changed in the evidence — the deep treatment](#3-what-si-33-actually-changed-in-the-evidence--the-deep-treatment)
4. [Candidate (a): a small real safetensors base — still deferred, still the operator's call](#4-candidate-a-a-small-real-safetensors-base--still-deferred-still-the-operators-call)
5. [Candidate (b): a second non-readout target, argued now](#5-candidate-b-a-second-non-readout-target-argued-now)
6. [Candidate (c): branch-contribution and expressibility audit — measurement first](#6-candidate-c-branch-contribution-and-expressibility-audit--measurement-first)
7. [Candidate (d): the representation question — the SI-32 §7.5 trigger, evaluated](#7-candidate-d-the-representation-question--the-si-32-75-trigger-evaluated)
8. [Candidate (e): transformer squash — precondition newly met](#8-candidate-e-transformer-squash--precondition-newly-met)
9. [Candidate (f): a third authored cycle](#9-candidate-f-a-third-authored-cycle)
10. [Candidate (g): additive / multi-target authorship](#10-candidate-g-additive--multi-target-authorship)
11. [Candidate (h): the composition line, the penalty axes, partial removal](#11-candidate-h-the-composition-line-the-penalty-axes-partial-removal)
12. [Candidate (i): per-probe anchoring; sequencing; magnitude as a dial; a minimum-movement floor](#12-candidate-i-per-probe-anchoring-sequencing-magnitude-as-a-dial-a-minimum-movement-floor)
13. [Candidate (j): Glimmer probing](#13-candidate-j-glimmer-probing)
14. [Decision summary](#14-decision-summary)
15. [The recommended slice: SI-35 (outline)](#15-the-recommended-slice-si-35-outline)
16. [Acceptance criteria for SI-35](#16-acceptance-criteria-for-si-35)
17. [Non-goals](#17-non-goals)
18. [Honesty and labelling requirements](#18-honesty-and-labelling-requirements-for-any-future-slice)
19. [Risks and open questions](#19-risks-and-open-questions)
20. [Summary](#20-summary)

---

## 1. Where SI stands after SI-33

Everything in this section is verifiable in the code and
docs on `main` at the SI-33 merge (main tip `c6985c1`,
the merge of PR #38). Numbers are cited to the as-built
document that measured them; this RFC measures nothing
new. Where a statement below is a *derivation* from cited
facts rather than a measurement, it is labelled a
derivation, exactly as SI-32 §1 labelled its arithmetic.

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
  SI-23 + SI-33).** The signal registry serves exactly
  `tiny-successor-lean-001` (SI-19) and
  `tiny-predecessor-lean-001` (SI-23) and refuses every
  other id. Both signals share one form (target-logits
  MSE), correction scale 0.15, 2,000 full-batch steps,
  learning rate 1.0, norm budget 0.9, 32 train / 16
  held-out examples; they differ only in correction
  direction, stream, and drawn sequences. The landed
  authored stack is now depth 2: `d0001` (signal 1,
  readout target) + `d0002` (signal 2, final-MLP-down
  target) — the second delta labelled a
  non-readout-target candidate wherever the stack is
  described (SI-32 §18 item 24).
- **Two recorded target anchorings (SI-25), unchanged.**
  `TARGET_ANCHORINGS` serves exactly
  `"weights-under-training"` (the default) and
  `"base-anchored"`.
- **Four recorded composition configurations (SI-31),
  unchanged.** `COMPOSITION_CONFIGS` serves exactly
  `"none"` (the default), `"movement-composition"`
  (SI-27), `"worst-input-composition"` (SI-29), and
  `"train-partition-projection"` (SI-31). The three
  mechanisms are never blended; their hyperparameter
  fields are mutually null.
- **Two recorded authorship targets (SI-33).**
  `AUTHORSHIP_TARGETS` serves exactly `"lm-head"` (the
  default; `lm_head.weight`) and `"final-mlp-down"`
  (`model.language_model.layers.1.mlp.down_proj.weight`,
  shape [32, 64]), validated by
  `validate_authorship_target(...)` with the existing
  `ValueError` discipline. Rank (8) and alpha (8) are
  pinned by each value's definition; neither is
  separately settable, and no caller supplies a tensor
  name. Composition under `"final-mlp-down"` is `"none"`
  only. There is no autonomous, learned, depth-driven,
  verdict-driven, or probe-driven selection on any of the
  four axes — signal, anchoring, composition, authorship
  target; the driver configures all four, and the journal
  shows all four.

**What SI-33 measured (cited to its as-built):**

- **Phase 0 first.** Before any decisive candidate was
  built, SI-33 published the backward derivation for
  `"final-mlp-down"` (target-logits MSE through the
  readout, the final RMSNorm — including its one new
  Jacobian — the residual add, and the LoRA fold, with
  the SwiGLU output `u` and the residual input `h_mid`
  justified as fixed inputs for the factor gradients),
  the re-run SI-25 control's per-probe inner-product
  table and allowed-input composition record (labelled
  diagnostics on an already-judged candidate), the
  pre-registered F1′′′/F2′′′/F3′′′ rules with F3′′′'s
  completeness boundary stated in advance
  (2.087217177504918e-05, 2× the re-run SI-25 control's
  held-out residual 1.043608588752459e-05), and the
  gradient-check protocol.
- **The gradient check passed, at both heads.** Central
  finite differences over every one of the 768 factor
  entries (B [32, 8] = 256, A [8, 64] = 512), tolerance
  1e-5 unchanged: worst disagreement
  3.8495586315441856e-08 at base and
  2.7961983521798964e-08 at head `d0001` — ~260–360×
  headroom, in line with the readout path's published
  figures. No decisive run was trusted before this check
  passed.
- **Probe-blindness, asserted three ways,** including a
  byte-identical new-target candidate (blob
  `sha256:d7af7f1f…e625`, the decisive candidate) from a
  perturbed-probe-artifact bundle copy, and spies
  asserting the new gradient's input set is exactly
  signal 2's train partition plus the landed head's
  weights, named as a set.
- **The at-base records, kept separately labelled.**
  (a) Signal 2, base-anchored, target `"lm-head"`, at
  base reproduced the established at-base run **exactly**
  (blob `sha256:131cc242…45f4`, identity drift
  0.16672966815531254, capability regression
  0.2418281203135848, ADMIT) — the axis's introduction
  did not move the existing path. (b) Signal 2,
  base-anchored, target `"final-mlp-down"`, at base —
  a labelled at-base fact, not a coincidence result and
  not the decisive claim: candidate blob
  `sha256:98768588…8d9a`, candidate norm
  0.29741157499219223, gate identity drift
  0.0016501061618328094, capability regression
  0.0026158951222896576, ADMIT to `d0001` with §6.6
  VERIFY measured-pass; held-out loss 0.00140625 →
  0.0013987123437990195 with 14/16 held-out examples
  closer — an improvement of less than 1%, with the
  near-inert-branch attribution already visible at base
  under the target's own labels.
- **All four controls re-ran inside the slice and
  reproduced exactly** (blob-for-blob, drift-for-drift):
  SI-25 (blob `sha256:646d1e8b…d675`, drift
  0.3329036422073841, REJECT), SI-27 (blob
  `sha256:1bb277ab…b637`, drift 0.32509003952145576,
  REJECT), SI-29 (blob `sha256:11cab02e…b8db5`, drift
  0.31215621903538704, REJECT), SI-31 (blob
  `sha256:32d13f65…a5c1`, drift 0.4775387533009052,
  REJECT on safety rank + identity). The target axis's
  refactor is proven behaviour-preserving for every
  existing semantics, including the projection
  construction.
- **The decisive two-cycle run.** One reused
  `TinyTransformerMind` through `si.run.run_cycles`:
  cycle 1 (signal 1, default anchoring, default
  composition, default target) reproduced
  SI-19/SI-21/SI-23/SI-25/SI-27/SI-29/SI-31 exactly —
  candidate blob `sha256:2761e8ff…6ca9e`, identity drift
  0.17189218290150166, capability regression
  0.1657181391492486, ADMIT to `d0001`, §6.6 VERIFY
  measured-pass. Cycle 2 (signal 2, base-anchored,
  composition `"none"`, authorship target
  `"final-mlp-down"`, trained at `d0001` against
  base-derived targets) produced candidate blob
  `sha256:d7af7f1feaa5b0451d441a4fce8842b471fc7c4b3e70e6237f1115eba877e625`
  (3,416 B, rank 8), candidate norm
  0.30801411790830574, and the unchanged gate measured:
  identity drift **0.17195402272045612** (cumulative,
  seed-absolute; PASS, tol 0.25), capability regression
  **0.001467481255531311** (head-incremental; PASS, eps
  0.25), echo accuracy 1.0 (16/16), safety suite pass —
  **ADMIT** to `d0002`, §6.6 VERIFY measured-pass on both
  arms, no rollback. Exactly two authored cycles were
  attempted and exactly one new-target decisive run was
  performed; no sweep over targets, no second non-default
  target.
- **The stack exists, and unwinds.** The authored
  two-delta stack — `d0001` (signal 1, `"lm-head"`) +
  `d0002` (signal 2, `"final-mlp-down"`) — is exercised
  by test through the existing rollback core: rollback
  to `d0001` restores the one-delta head (VERIFY
  re-measured: max drift 0.17189218290150166, cycle 1's
  own figure), and rollback to `base` restores the seed
  head. `si squash` still fails closed on the resulting
  transformer stack, asserted on the stack itself before
  the unwind (SI-12's refusal, unchanged).
- **The training record, under its own labels.** Cycle
  2's signal train loss moved 0.002750453165825476 →
  0.002734615539151513 over the standing 2,000 steps;
  pre-training held-out loss 0.002757805049077469;
  held-out loss after 0.0027387202915082178 (~0.7% lower;
  16/16 held-out examples closer); norm-guard
  activations 0. These are training numbers, not gate
  numbers (§18 item 1).

**The diagnostics (cited to SI-33's as-built; derivations
labelled as SI-33 labelled them):**

- **Where the update acts decided both figures.** The
  treatment's probe-space movement ‖m2‖ is
  0.010469039280455053 — about 75× smaller than the
  SI-25 control's ‖m2‖ (0.7816722548913912), at a
  *larger* candidate norm (0.30801411790830574 against
  the control's 0.2515524481203623). Its largest
  incremental movement at any probe is
  0.002182206604629755 (at tiny-id-4, notably not the
  binding probe), against predecessors' 0.15–0.31; its
  incremental movement exceeds cycle 1's remaining
  headroom (0.07810781709849834) at 0 of 8 probes;
  against cycle 1's own drift its composed figures
  differ by at most 0.0034 at every probe.
- **Alignment, at miniature magnitude.** In aggregate
  over the 8 identity probes the down-target movement
  weakly opposes the first: inner product
  −0.002171869843126004 (SI-25 control:
  −0.21256230517288188; SI-27: −0.3271477846470207;
  SI-29: −0.28751472160662706; SI-31:
  +0.21576810032436344), sign agreement 0.4609375
  (118/256, published beside the magnitude-weighted
  figure, never alone). At the binding probe tiny-id-0
  the inner product is −0.00031118965094582363 —
  opposing in sign, at ~1% of the magnitude of any
  predecessor's figure in either direction; the movement
  opposes at 7 of 8 probes, by amounts ~100× smaller
  than any predecessor's.
- **The falsification readings, as measured:**
  - **F1′′′ — the geometry survives the fold: NOT
    measured.** Its first clause fails (the composed
    maximum does not stay above tol:
    0.17195402272045612 ≤ 0.25 — it is, to within
    0.00006, cycle 1's own maximum) and its second
    clause also fails (the inner product at tiny-id-0
    is negative, not positive). The premise F1′′′ was
    written to falsify — that acting upstream changes
    the binding probe's alignment — is therefore not
    falsified in the registered form. What the run does
    **not** show is the stronger claim the sign change
    tempts: the opposition is ~100× weaker than any
    predecessor's alignment in either direction, the
    signature of a movement too small to align or oppose
    materially, not of a geometry reversed.
  - **F2′′′ — NOT measured as a reading: its transfer
    clause holds, its maximum clause fails.** (a)
    Transfer clause HOLDS — the treatment's inner
    product at its own worst probe (tiny-id-0,
    −0.00031118965094582363) is smaller than the SI-25
    control's at that same probe
    (+0.0338873784299374), smaller in the strong sense
    of opposite sign at ~1% of the magnitude. (b)
    Maximum clause FAILS — the composed maximum does
    not stay above tol 0.25. The proposition F2′′′ was
    written to test — the maximum as an immobile fact
    *despite* transfer — is the opposite of what this
    run measured; the maximum moved precisely because
    almost nothing was added to it. The two clauses are
    reported separately and are never rounded into one
    (§18 item 23).
  - **F3′′′ — the destination is traded away: MEASURED,**
    under the pre-registered rule, by its completeness
    clause. (a) The loss falls (0.002757805049077469 →
    0.0027387202915082178) ✓; (b) closer fraction 16/16,
    past the rule's 3/4 boundary ✓; (c) residual
    completeness FAILS — 0.0027387202915082178 >
    2.087217177504918e-05 (2× the re-run SI-25 control's
    residual 1.043608588752459e-05), by ~131×. One clause
    failing suffices under the rule as registered. The
    trade's shape: the candidate keeps every held-out
    example closer and still arrives only ~0.7% of the
    way in loss, against its primary control, which
    arrives at 1.043608588752459e-05 (~264× closer to
    the destination) and is REJECTED for it.
- **The attribution (measurements, with one labelled
  derivation — SI-33's, restated exactly).** Measured at
  head `d0001` over signal 2's train partition (last
  positions): the SwiGLU output's norm ‖s‖ ranges
  0.040–0.064 while the residual stream it perturbs has
  ‖h_mid‖ 1.97–2.02; at the base on a representative
  sequence the final MLP branch's entire output norm is
  0.0074 against a residual norm of 2.00 — the final
  layer's MLP contributes ~0.4% of the residual stream
  it adds into. *Derivation from those measurements:* an
  update of Frobenius norm ρ changes `y` by at most
  ρ·‖s‖ (≤ 0.9 × 0.064 ≈ 0.058 at the standing norm
  budget, before the final RMSNorm and readout divide it
  down further), while the signal asks for a logit
  correction of norm 0.15·√2 ≈ 0.212 — so no candidate
  within the standing budgets can express this signal's
  correction through this tensor on this miniature, in
  either direction: it cannot wreck the probes (every
  predecessor's failure mode) and it cannot reach the
  destination (F3′′′).

**The seven-run cycle-2 record** (each figure under its
own run's labels; SI-33's as-built publishes all seven
side by side):

| Run | Cycle-2 configuration | Candidate norm | Rank | Aggregate probe inner product | Inner product at worst probe | Composed identity drift (cumulative) | Worst probe | Verdict |
|---|---|---|---|---|---|---|---|---|
| SI-21 | signal 2, weights-under-training, composition n/a (pre-SI-27), target `"lm-head"` | (SI-21 as-built) | 8 | (SI-21 as-built) | (SI-21 as-built) | 0.34378434903919697 | (SI-21 as-built) | REJECT |
| SI-23 | signal 2, weights-under-training, composition n/a, target `"lm-head"` | 0.21317186599953494 | 8 | +0.07637087369869666 | (aligned at tiny-id-3, SI-23 as-built) | 0.2963416576385498 | tiny-id-3 | REJECT |
| SI-25 | signal 2, base-anchored, composition `"none"`, target `"lm-head"` | 0.2515524481203623 | 8 | −0.21256230517288188 | +0.0338873784299374 (tiny-id-0) | 0.3329036422073841 | tiny-id-0 | REJECT |
| SI-27 | signal 2, base-anchored, composition `"movement-composition"`, λ = 1.0, target `"lm-head"` | 0.2969693161862222 | 8 | −0.3271477846470207 | +0.03801222744105431 (tiny-id-0) | 0.32509003952145576 | tiny-id-0 | REJECT |
| SI-29 | signal 2, base-anchored, composition `"worst-input-composition"`, λ_w = 1.0, k = 4, target `"lm-head"` | 0.28252038537434354 | 8 | −0.28751472160662706 | +0.04944786620153699 (tiny-id-0) | 0.31215621903538704 | tiny-id-0 | REJECT |
| SI-31 | signal 2, base-anchored, composition `"train-partition-projection"`, α = −0.8450027779470831 (constructed), target `"lm-head"` | 0.2890208630043799 (pre-projection 0.25155244947498906) | **16 (constructed)** | **+0.21576810032436344** | **+0.14235826783471692 (tiny-id-0)** | **0.4775387533009052** | tiny-id-0 | **REJECT (safety rank + identity)** |
| SI-33 | signal 2, base-anchored, composition `"none"`, target **`"final-mlp-down"`** | 0.30801411790830574 | 8 | −0.002171869843126004 | **−0.00031118965094582363 (tiny-id-0)** | **0.17195402272045612** | tiny-id-0 | **ADMIT** |

Six authored second writes refused by the same unchanged
accounting, the sixth on two independent lines; the
seventh admitted — by a candidate whose movement is, in
probe space, ~1% of any predecessor's, and whose own
as-built declines to call that a composition success.
The deepest landed authored stack on record is now
depth 2 (`d0002`); the transformer-squash precondition
(an authored two-delta stack) exists for the first time
(§8 evaluates what that does and does not license). The
full suite at the SI-33 merge: **382 passed, 2 skipped**.
The Glimmer seed is untouched; Glimmer-scale deltas
still escalate with checks 3/4 "not evaluated".

---

## 2. Inherited invariants

Not negotiable inside this RFC (SI-9 §1.3, RFC SI-13 §2,
RFC SI-18 §2, RFC SI-20 §2, RFC SI-22 §2, RFC SI-24 §2,
RFC SI-26 §2, RFC SI-28 §2, RFC SI-30 §2, RFC SI-32 §2,
restated for the candidates):

1. **Base is frozen forever.** All change is a stack of
   deltas; rollback is moving `head`.
2. **Never duplicate bytes** (Mark's design rule). A blob
   whose hash exists is never rewritten or copied; seed
   imports by reference where the pattern applies
   (`stored_bytes: 0` — the SI-8 Glimmer seed stores zero
   tensor bytes against 59,553,374,040 deduplicated
   bytes, per the SI-33-era bundle record).
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
   one (§18 items 22–25); **and an ADMIT's measured
   attribution travels with the ADMIT wherever the ADMIT
   is summarised (§18 item 26, new in this RFC).**
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
   `identity_tol`, `capability_eps`, `rank_cap`, any
   other cap, or any policy default to make an outcome
   pass — and, the mirror image SI-33 now forces this
   invariant to state, none proposes moving any of them
   to make an outcome *fail* either: a budget re-priced
   after it binds, in either direction, in a document
   whose author knows the candidate it would have
   affected, is the gate-fitting shape (§12 argues the
   one candidate that asks for exactly that, in the
   failing direction, and does not adopt it). The seven
   cycle-2 verdicts are measurements to design *around*
   — with new authorship semantics or new measurements —
   never numbers to outvote.
9. **The probes are the judge, never the training
   set.** No candidate in this RFC may train on, penalise
   against, project against, parameterise against, or
   otherwise shape a candidate using any probe input,
   probe logit, or probe-derived movement. A method that
   reads the judge's inputs to shape the judged artifact
   has not passed the judge; it has memorised the
   courtroom. Every candidate below is audited against
   this invariant first — including §6's recommended
   candidate, whose measurements read only the
   populations training is allowed to read, and which
   excludes probe inputs entirely, for the reason §6
   states.

---

## 3. What SI-33 actually changed in the evidence — the deep treatment

SI-27 made the question *why a mean-priced penalty loses
at the worst probe*; SI-29 sharpened it; SI-31 completed
the composition attribution by removing the priced
quantity outright; SI-33 changed the record's one
unvaried premise — where the authored update acts — and
produced the record's first ADMIT. This section assembles
what SI-33 jointly establishes with the six prior runs,
what it leaves unknown, and what any candidate must
therefore *measure* — not assume — to aim at a second
write that both lands and arrives. Derivations are
labelled derivations.

### 3.1 The machinery result is real, and it is the whole of the positive claim

SI-33 proves, end to end and under test, that the loop's
machinery is not readout-specific: a recorded
authorship-target axis with exactly two named values and
the standing refusal discipline; a backward derivation
for an upstream tensor, checked by central finite
differences over all 768 factor entries at two heads
(worst disagreements 3.8495586315441856e-08 and
2.7961983521798964e-08 against tolerance 1e-5); training
that is probe-blind under the new path, proven by a
byte-identical candidate from a perturbed-probe-artifact
bundle; a candidate judged by the unchanged gate under
exactly the defaults every predecessor faced; an ADMIT
backed by §6.6 VERIFY on both arms; a landed two-delta
stack; and step-by-step unwind through the existing
rollback core (to `d0001`, VERIFY re-measured at cycle
1's own figure 0.17189218290150166; to `base`). Four
controls reproduced exactly under the refactored code,
including SI-31's rank-16 construction — the axis's
introduction moved no existing semantics. That is a
genuine frontier crossed, and this RFC claims it at
exactly that scope: **the self-write loop now operates
against a non-readout transformer tensor, with its
gate, its journal, and its honesty machinery intact.**
Nothing in §§3.2–3.3 extends the claim one millimetre
past the machinery.

### 3.2 The ADMIT's attribution: an inert branch, measured

The same as-built that reports the ADMIT also prices it,
and the pricing is not ambiguous. The candidate's norm
(0.30801411790830574) is the largest of the four
base-anchored `"none"`/penalty-era candidates' norms on
record, yet its probe-space movement ‖m2‖
(0.010469039280455053) is ~75× smaller than its primary
control's (0.7816722548913912) — where the update acts,
not how large it is, decided both the gate's figure and
the destination's (SI-33's diagnostics, in its own
sentence). The composed drift at the binding probe
(0.17195402272045612) is cycle 1's own drift
(0.17189218290150166) plus an increment of ~0.00006 —
the gate passed the stack because the second write added
almost nothing to what the first write had already been
judged on. The mechanism is measured, not inferred: the
final layer's MLP branch contributes ~0.4% of the
residual stream it adds into on the measured population
(branch output norm 0.0074 against residual norm 2.00 at
base; ‖s‖ 0.040–0.064 against ‖h_mid‖ 1.97–2.02 at
`d0001`), and SI-33's labelled derivation from those
measurements shows the standing budgets cannot express
the signal's correction through this tensor in either
direction (bound ≈ 0.058 against a required ≈ 0.212,
before downstream attenuation). The first cycle-2 ADMIT
of the project is therefore real, gate-measured, and
VERIFY-backed — and it is an ADMIT of a candidate that
does almost nothing, landed by a gate that correctly
measured almost nothing done. Any summary of SI-33 that
carries the ADMIT without this attribution is a
labelling defect (§18 item 26).

### 3.3 The readings, jointly: neither the geometry claim nor the immobility claim survives in its registered form — and neither reversal is earned

The three successor readings were written, in SI-32
§15, to discipline exactly the interpretations SI-33's
result tempts, and they do:

- **F1′′′ not measured** means the fold-geometry premise
  is *not* falsified: upstream authorship did not
  reproduce the binding probe's positive alignment — but
  at ~1% of the magnitude at which alignment has ever
  been measured in either direction, the reading cannot
  distinguish "the geometry changed" from "the movement
  vanished". The honest joint statement is narrower than
  either temptation: **at movement scale ‖m2‖ ≈ 0.0105,
  no alignment claim in either direction is measurable,
  because the quantity alignment is a property of has
  effectively left the record.** F1′′′'s registered
  consequence does not trigger; neither does any claim
  that the fold change solved the alignment problem.
- **F2′′′ not measured as a reading** (transfer holds,
  maximum fails) retires the proposition it was written
  to test in the opposite direction from SI-31's F2′′:
  the maximum is not immobile. It moved — down past tol —
  by the second movement standing still. A maximum that
  can be passed by adding ~0.00006 at the binding probe
  has not been *addressed*; it has been *declined*. The
  six-run attribution (§5 of SI-32) stands untouched for
  movements of predecessor scale: nothing in SI-33
  measures a readout-scale movement composing any
  differently than SI-25–SI-31 measured.
- **F3′′′ measured** is the reading with a verdict, and
  its verdict is the slice's scientific centre of
  gravity: the destination was traded away, by ~131× on
  the completeness clause, with the trade's shape
  published (loss falls, 16/16 closer, residual
  0.0027387202915082178 against boundary
  2.087217177504918e-05). Read together with §3.2, the
  gate pass and the destination failure are the same
  fact read twice (SI-33's sentence, adopted here): this
  candidate is admissible because it is, in movement
  terms, nearly the identity — and a nearly-identity
  second write does not reach its signal.

The frontier SI-33 actually opens is therefore not
"upstream authorship works" and not "upstream authorship
fails". It is a question the six readout runs never had
to ask, because the readout can always express the
correction (the SI-25 control reaches residual
1.043608588752459e-05): **which tensors of this miniature
can express a signal's correction at all, at the
standing budgets — and which are, like the final layer's
MLP branch, structurally near-inert for updates of any
lawful size?** That question is measurable today, with
instruments that already exist, on populations training
is allowed to read. §6 recommends measuring it before
any further authorship candidate is argued; §5 records
why arguing the next target without it would repeat
SI-33's one avoidable cost — a full slice spent
discovering, after the fact, a property of the model
that a measurement could have published before it.

### 3.4 What SI-33 did not change

Stated plainly, so no candidate below can borrow a
change that did not happen:

- **The composition attribution's scope is unchanged.**
  F1, F1′, and F1′′ remain readings about readout
  movement pairs; SI-33's candidate is not a readout
  movement, its diagnostics claim no readout-fold
  derivation (§18 item 24), and the composition line
  stays closed (§11). SI-33 neither reopens it nor
  extends its closure to upstream movements at
  predecessor scale — no such movement has been measured.
- **SI-31's two-line refusal stands unsoftened.** The
  projected candidate still fails on safety rank *and*
  on identity (0.4775387533009052 > 0.25), independently;
  SI-33's ADMIT is a different candidate, on a different
  tensor, at a different head configuration — a different
  measurement (§18 item 25). §7 evaluates the rank
  question's trigger against SI-33 and finds it unmet.
- **No policy, tolerance, cap, or gate definition
  moved.** `rank_cap` 8 priced SI-33's rank-8 candidate
  exactly as it priced every predecessor; `identity_tol`
  and `capability_eps` are still 0.25 and still the
  unchanged gate's.
- **Glimmer is exactly where it was.** The seed is
  untouched, stored bytes still 0 against
  59,553,374,040 deduplicated bytes; Glimmer-scale deltas
  still escalate with checks 3/4 "not evaluated"; no
  forward path exists and none is faked (§13).
- **The real-base preconditions are exactly where they
  were.** SI-18 §3.7's A-1 to A-4 are design obligations
  SI-33 does not touch (§4).
- **Depth is 2, and nothing deeper is claimed.** Exactly
  two authored cycles have ever been attempted in a
  decisive run; the landed stack's existence licenses no
  claim about cycle 3, about mixed-target stacks at
  non-base heads beyond this one stack, or about
  continual learning of any kind (§9).

### 3.5 What is still not known — the measurements and definitions that must precede any fix

Stated as open questions a slice could retire — this
RFC invents no numbers for them (§19 carries them
forward):

- **The inertness map of the miniature.** Branch
  contribution and input-activation scale are measured
  for exactly one branch of one layer: the final layer's
  MLP, on signal 2's train partition, at `d0001` (plus
  one representative sequence at base). For the other
  branch of the final layer (attention), both branches
  of layer 0, the embeddings, and every other weight
  matrix, no contribution ratio, input-norm record, or
  expressibility bound exists. Whether the final MLP's
  ~0.4% is typical of this construction or an outlier is
  **an open question to measure** — and it is the
  question §5's candidate cannot be argued without.
- **Downstream transfer, per tensor.** SI-33's bound is
  a bound at the tensor's own output (ρ·‖s‖), with
  downstream attenuation through the final RMSNorm and
  the readout stated qualitatively ("divide it down
  further") and measured only in aggregate, after the
  fact, by the landed candidate's ‖m2‖. No per-tensor
  transfer coefficient — output perturbation to logit
  movement — exists for any tensor except the two the
  record has landed deltas on (the readout, trivially,
  and the final down-projection, by SI-33's diagnostics).
  An audit can bound and rank; it cannot manufacture
  transfer figures, and §15 does not ask it to.
- **Whether any non-readout tensor can express signal
  2's correction at the standing budgets.** The
  derivation that closes the final down-projection
  (bound ≈ 0.058 < required ≈ 0.212) has no published
  analogue for any other tensor. This is §3.3's frontier
  question in its precise form, and it is unknown in
  both directions.
- **The squash design for a mixed-target stack.** The
  precondition SI-12/SI-30/SI-32 deferred on — an
  authored two-delta stack — now exists, and with it the
  obligations SI-32 §13.1 recorded (a gate context that
  represents four recorded axes; members that acted on
  different tensors) become concrete rather than
  prospective. How a squashed composite of two deltas on
  *different* tensors is to be carried — one record with
  two targets, against `max_targets_per_delta` 1 — has
  never been argued. §8 evaluates the candidate; the
  design question is recorded there as unargued, not
  papered over.
- **A third cycle's lawful configuration.** Both signals
  are spent in the landed stack; no third signal exists,
  and no run on record re-trains a signal already landed
  at a deeper head. What a third cycle would even be —
  which signal, under which anchoring, against which
  target source — is undefined in the current semantics,
  not merely unmeasured (§9).

---

## 4. Candidate (a): a small real safetensors base — still deferred, still the operator's call

**Verdict up front: DEFERRED**, unchanged from SI-32 §3,
and flagged here — as the tasking for this RFC requires
and as every predecessor RFC has recorded — as a
**decision that belongs to the operator (Mark), not to
this RFC**: naming a real base requires a checkpoint
download, this scope forbids downloads, and no slice
recommended here proposes one, schedules one, or treats
one as approved. The SI-18 §3.7 audit's four
preconditions (A-1 to A-4) remain unmet, and SI-33's
evidence does not touch any of them: the miniature's
inertness problem is a property of a constructed model's
branch scales and of the standing budgets, measured on
constructed signals; nothing about an upstream update
moving ~0.001–0.002 at the probes argues that a real base
would be more or less inert per branch, and the local
presence of the Glimmer checkpoint (the SI-8 seed's
referenced source) does not retire the preconditions —
A-2 (a binding design for `reference-v1` bases), A-3 (a
capability-suite design for a trained base), and A-4 (an
executor design scoped to one family, with SI-15-grade
numeric proof) are design obligations, not download
obligations, and none of them exists. SI-33 adds one
sentence to the audit's context, in the same direction
as SI-31's and SI-32's sentences cut: the project now
owns a landed two-delta authored stack and a completed
first non-readout target record whose behaviour at any
other scale is unclaimed — moving the authorship
question to a real base before the miniature's
expressibility question (§6) is answered would carry an
unanswered question to a more expensive room, exactly as
it would have before SI-33. Deferred; the download
question stays the operator's, and this RFC does not
recommend the candidate as buildable now.

---

## 5. Candidate (b): a second non-readout target, argued now

**Verdict up front: DEFERRED — not rejected, and not
arguable today: it is the candidate §6's audit exists to
make arguable, and arguing it in this RFC would require
precisely the measurements that do not exist.** SI-33's
as-built states the status in its own words: *"Exactly
one non-readout tensor was named, trained, checked, and
judged. Every other tensor remains a future RFC's
unargued candidate."* This section is that future RFC's
first look at the candidate, and its finding is that the
argument cannot yet be made — for a reason SI-33 itself
created.

**What the argument would need.** SI-32 §12.1 argued the
final down-projection from the backward path (shortest
honestly checkable derivation), from judge
measurability (outside the blind set), from standing
budgets, and from full finite-difference checkability.
Those four grounds are still the right grounds, and for
several tensors they can be assessed today from the
architecture alone (derivations, labelled as such): the
final layer's remaining matrices (gate/up projections,
attention projections) have longer backward chains than
the down-projection's — gate/up require the SwiGLU
derivative the down path treated as a fixed input, the
attention projections require the attention Jacobian —
and every layer-0 matrix requires the full two-layer
chain. Factor-entry counts are arithmetic (rank-8
factors on an [r, c] matrix carry 8·(r + c) entries:
512 for [32, 32], 768 for [32, 64] / [64, 32]). None of
that is the missing ground.

**The missing ground is inertness, and SI-33 is why it
is now mandatory.** The down-projection satisfied every
ground SI-32 argued — and was structurally incapable,
at the standing budgets, of expressing the signal it
was trained on, a fact discovered by the decisive run's
attribution after the slice's full price (derivation,
gradient check, controls, decisive run) had been paid.
The property that decided the outcome — the branch's
share of the residual stream, ~0.4% — was measurable
before any of that: it is a forward-path statistic on
allowed inputs, computable from the seed weights alone.
A second-target RFC that named its tensor the way
SI-32 named its — from the backward path, with inertness
unmeasured — would be repeating the one avoidable step
in SI-33's method with the lesson already published.
The candidate is therefore not rejected: it is
**sequenced**. §6's audit measures, for every weight
matrix in the model, the branch contribution and the
expressibility bound whose absence this section
records; a future RFC (SI-36, if the audit's readings
leave a candidate standing) argues one tensor from that
record, with the backward-path grounds above and the
inertness grounds §6 supplies. If the audit's readings
leave *no* tensor standing (R2, §15), this candidate is
not deferred but closed at miniature scale, and §6
states that consequence in advance rather than letting
a future RFC discover it mid-slice.

**What this section does not do.** It names no tensor.
It does not rank tensors by any judged quantity (no
probe figure has selected, or may select, a target —
invariant 9). It does not license a sweep, a shortlist
run inside one slice, or a second decisive run at a
different target under SI-33's authority: SI-32 §17's
non-goal — a future target is a future RFC's single
argued tensor — stands, and this RFC extends it: the
future argument must now include the audit's measured
grounds, or state why it proceeds without them.

---

## 6. Candidate (c): branch-contribution and expressibility audit — measurement first

**Verdict up front: RECOMMENDED — as SI-35 (§15).** A
measurement-only slice: no candidate is trained, no gate
evaluation is performed on any new artifact, no policy
or semantics change; the slice publishes, for every
weight matrix in the miniature, the branch-contribution
measurements, input-activation records, and
expressibility bounds that SI-33 measured for one branch
after its decisive run — computed before any further
authorship candidate is argued, on populations training
is allowed to read, under pre-registered readings (§15)
whose consequences are stated in advance in both
directions.

**Why this candidate, of the space.** It is the only
candidate in §§4–13 whose premise SI-33 *strengthened*:
SI-33 converted "upstream authorship" from an untested
premise into a machinery result plus a measured
attribution, and the attribution names a quantity —
branch inertness at standing budgets — that (i) decided
the slice's scientific outcome, (ii) is measurable with
existing instruments (forward-path statistics; SI-33's
own attribution is the worked example), (iii) every
remaining authorship candidate (§§5, 9) needs and none
has, and (iv) cannot be gamed toward a verdict, because
the audit authors nothing the gate could judge. It is
also the candidate SI-33's closing sections point to
without naming: the as-built's "No second target, no
sweep" paragraph leaves every other tensor "a future
RFC's unargued candidate", and §5 shows the argument
that is missing is this audit's record. Measurement
first is not a slogan here; it is the specific repair
to the specific gap SI-33's Phase 0 could not have had
— SI-32 registered no inertness reading because no
inertness measurement existed to register one against —
and which no future Phase 0 should be without.

**Why not a training slice.** Every training-shaped
candidate in the space either inherits an unargued
premise (§5: inertness; §9: a third cycle's very
configuration), asks for a policy argument first (§§7,
8), or is closed (§§10–12). The audit is the one slice
that makes the next training slice — if one survives
its readings — arguable, and its failure mode is
publishable in exactly the project's tradition: if the
audit measures that no non-readout tensor can express
the signal at standing budgets (reading R2, §15), that
is a finding about the miniature that closes a line
honestly, at the price of forward passes rather than of
a derivation, a gradient check, and a decisive run.

**The probe question, answered in advance.** The audit
shapes a future *argument* about where authorship might
act; it authors no candidate. Even so, this RFC holds
it to the stricter standard: **the audit's measurements
are taken on allowed populations only** — both signals'
train partitions, with held-out partitions reported
only, under their own labels — and no probe input,
probe logit, or probe-derived quantity enters any audit
figure, any ranking, or any reading. Probe-side branch
statistics would be diagnostics on the judge's inputs
used to select the next judged artifact's defining
configuration; the project has rejected that shape at
every reading (§12), and the audit does not need it —
SI-33's attribution, the method's worked example, was
itself measured on the train partition. §16 criterion 2
asserts the disjointness by test.

---

## 7. Candidate (d): the representation question — the SI-32 §7.5 trigger, evaluated

**Verdict up front: ARGUED in SI-32, trigger EVALUATED
here against SI-33's candidate and found NOT FIRED —
DEFERRED, unchanged; `rank_cap` stays 8 for trained and
constructed candidates alike, and SI-31's candidate is
not re-judged.** SI-32 §7 argued the representation
question in both directions and deferred it with a
precise trigger: the question becomes arguable as a
*change* when and only when a candidate exists that is
measured passing the identity arm and the capability arm
while failing *only* on the `size_budgets` rank line.
SI-33 is the first slice since that trigger was written
to land a cycle-2 candidate, so this RFC is the first
obliged to evaluate the trigger against a new record.
The evaluation is short because the figures are not
close:

- **SI-33's candidate does not meet the trigger's
  shape.** It is a *trained* candidate of rank 8 — the
  standing cap, not above it. It passes the identity arm
  (0.17195402272045612 ≤ 0.25), the capability arm
  (0.001467481255531311 ≤ 0.25), *and* the safety suite
  including the rank line (rank 8 ≤ `rank_cap` 8;
  aggregate norm 0.30801411790830574 ≤ cap 1.0). It fails
  on no line. A trigger defined by "passes identity and
  capability while failing only on rank" is not fired by
  a candidate that fails on nothing, any more than by
  one that fails on identity too.
- **No other candidate on record meets it either.** The
  only candidate that has ever failed the rank line is
  SI-31's constructed candidate, which fails identity
  independently at ~1.91× the tolerance
  (0.4775387533009052 against 0.25; derivation from the
  cited figures, as SI-32 §7.4 labelled it) — the
  deciding fact SI-32 already recorded, unchanged by
  SI-33.
- **SI-33, if anything, moves the trigger further away,
  and says so honestly in both directions.** On one
  side: the record now contains a landed cycle-2 ADMIT
  achieved entirely inside the standing rank budget, so
  "rank is what stands between the project and a landed
  second write" is no longer a sentence the record
  supports in any form. On the other: the ADMIT was
  achieved by a candidate that does almost nothing
  (§3.2), so it is not evidence that the rank budget is
  *generous* either — it prices a question (what a
  rank-8 candidate on an inert branch can do) that the
  trigger was never about. Both directions are recorded
  so the trigger's continued dormancy is not misread as
  either vindication or indictment of the cap.

The trigger stays exactly as SI-32 §7.5 wrote it,
including its anti-retroactivity clause: a future rank
rule, if one is ever argued on a candidate that earns
it, never re-judges SI-31's candidate, whose verdict
stands on both lines permanently. Until the trigger
fires, any text that describes the rank line as "the
reason SI-31 failed" without the identity line beside
it is a labelling defect (§18 item 25) — and, this RFC
adds, any text that describes SI-33's ADMIT as bearing
on the rank question in either direction without
§3.2's attribution beside it is the same defect's mirror
image (§18 item 26).

---

## 8. Candidate (e): transformer squash — precondition newly met

**Verdict up front: DEFERRED — the state change is real
and is recorded as such, but the candidate's design
obligations are unargued for the stack that now exists,
and on the only stack in existence a squash would not
yet run.** This is the one candidate besides §6 whose
status SI-33 genuinely changed, and this section records
the change at full strength before deferring:

- **The precondition is met for the first time.**
  SI-12 made `si squash` fail closed on transformer
  stacks because composition of transformer deltas was
  unimplemented; RFCs SI-30 §12.1 and SI-32 §13.1
  deferred transformer squash on the stated ground that
  its precondition — an authored two-delta stack — was
  unmet after (then) five, then six measured attempts.
  SI-33 landed that stack: `d0001` + `d0002`, VERIFY-
  backed, unwound and re-measured by test. The deferral
  ground those RFCs stated is therefore **retired**;
  this section does not pretend otherwise, and no future
  RFC may cite "no authored stack exists" again.
- **What replaces it: three design obligations, all
  unargued, each recorded by SI-32 §13.1 or created by
  SI-33's stack's specific shape.**
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
  a distinct record class with its own discipline (the
  shape of SI-32 §7's representation question, in a
  second domain), or inadmissible until the policy
  question is argued, has never been argued anywhere.
  (ii) *The gate context.* A squash candidate is judged
  by the full gate (demo squash: `SquashContext`, five
  checks, post-squash depth accounting). A transformer
  squash context must represent a stack whose members
  carry four recorded axes — signal, anchoring,
  composition configuration, authorship target — one of
  whose members is a construction (none landed is; the
  obligation is prospective, recorded in SI-32 §13.1)
  and whose members acted on different tensors; no such
  context exists, and SI-33's as-built asserts the
  fail-closed refusal on the real stack precisely
  because none does. (iii) *The tolerance verify over
  named tensors.* Demo squash verifies the squashed
  stack reproduces the unsquashed stack's behaviour
  within tolerance before gating; for deltas on
  different tensors the composition is per-tensor exact
  in the folded weights (a derivation), but the
  *behavioural* verify — through RMSNorms and the
  readout the two updates share downstream — is a
  forward-path measurement whose design (populations,
  tolerance, failure semantics) is unwritten.
- **And a plain mechanical fact:** `squash_bundle`'s
  standing semantics are a no-op when the active stack
  is no deeper than `keep` (default 3; SI-12's as-built
  records the `L <= keep` no-op as unchanged behaviour,
  asserted on transformer stacks). The only authored
  transformer stack in existence has depth 2. A squash
  slice built today would implement machinery whose
  trigger condition cannot occur on the only real stack
  it could be exercised against, and would be tested —
  as SI-12's refusal was — on synthetic fixture stacks.
  That is not a disqualification (SI-12's fixtures were
  honest work), but it prices the urgency honestly:
  squash becomes *load-bearing* when depth approaches
  `keep` or `max_stack_depth` 8, and neither is near.

**Trigger, stated precisely** (in the shape of §7's):
transformer squash becomes the recommended slice when
(a) a third authored delta lands, or the active authored
depth otherwise exceeds `keep`, **or** (b) a future RFC
argues obligations (i)–(iii) as its main business — the
carrying question first — and recommends the
implementation on that argued basis. Either trigger
makes squash an RFC-first candidate, never an
implementation improvised inside another slice. Deferred
until then; SI-12's fail-closed refusal stands
throughout, and §15 builds nothing toward squash.

---

## 9. Candidate (f): a third authored cycle

**Verdict up front: DEFERRED — no lawful configuration
for a third cycle exists in the current semantics, so
there is no candidate to evaluate, only a definition to
write, and writing it is not the next slice.** The
temptation is obvious and is stated fairly: the stack
is depth 2, `max_stack_depth` is 8, the machinery — as
SI-33 proved for a non-readout second cycle — is not
readout-specific, so a third cycle looks like "more of
the same". It is not, for reasons that are definitional
before they are empirical:

- **Both signals are spent.** The registry serves
  exactly two signals; the landed stack consumed signal
  1 at cycle 1 and signal 2 at cycle 2. A third cycle
  must either re-train a landed signal at head `d0002`
  — a configuration no run on record defines (every
  decisive run trains each signal at most once, against
  a head whose stack contains only the *other* signal
  or nothing) — or author a third signal, which SI-32
  §17 lists as a non-goal for the slices it justified
  and which no RFC has ever argued on grounds
  independent of the gate (signal authoring argued
  backward from the obstacle the gate measured is
  SI-26 §10's rejected signal search, §12).
- **Its anchoring and target source are undefined.**
  Both anchorings are defined for a candidate trained
  against a head; what "base-anchored" targets mean for
  a signal already landed once — re-derived from base,
  or re-based at its own landed delta — is a semantics
  question with no record to cite, and each answer
  changes what the cycle would measure.
- **Its stack claim would be the first mixed claim past
  depth 2.** SI-33's as-built claims exactly one
  mixed-target stack, as measured, and nothing about
  such stacks at non-base heads beyond it; a third cycle
  trains *at* a non-base, mixed-target head as its
  starting point — the precise configuration SI-33
  declined to claim anything about.

None of this rejects depth. SI-32 §10 recorded the
standing position and this RFC carries it: nothing here
speaks against more than two cycles someday, or against
signals authored for reasons independent of the gate.
What is deferred is a *third cycle as the next slice*:
it needs its own RFC arguing signal semantics at depth
≥ 2 first — and §6's audit, which prices what any
further cycle's candidates could even express, is the
measurement such an RFC would cite. Deferred behind it.

---

## 10. Candidate (g): additive / multi-target authorship

**Verdict up front: REJECTED as a next slice (carried,
not reopened).** Unchanged from SI-32 §12.2: additive
and multi-target authored candidates require loosening
`max_targets_per_delta` / `allow_additive` policy
defaults with zero authored-behaviour evidence under
them, and the cycle-2 record is not evidence *for*
loosening them. SI-33 requires one clarification, made
here so its stack is never cited for this candidate:
**the landed stack is a mixed-target *stack*, not a
multi-target *delta*.** Each of its members is a
single-target candidate, authored in its own cycle,
judged by the unchanged gate in its own right, under
`max_targets_per_delta` 1 — the sequential composition
of two lawful single-target writes is evidence about
the *loop*, and it is exactly the evidence SI-33 claims;
it is no evidence at all about a single candidate
carrying two targets, a configuration that has still
never been authored, trained, or judged. (The distinct
question of a *squashed composite* carrying two targets
is §8's obligation (i) — a carrying question about
already-judged deltas, argued there on its own terms,
and not a reopening of this candidate.) Not reopened.

---

## 11. Candidate (h): the composition line, the penalty axes, partial removal

**Verdict up front: CLOSED / DEFERRED exactly as SI-32
§§4, 6, and 8 left them — SI-33 supplies no new evidence
in either direction, and this section records that
rather than re-arguing it.** The composition line is
done as a line (SI-32 §4): the mean penalty (SI-27), the
worst-case penalty (SI-29), and full-removal projection
(SI-31) were each falsified at the binding probe at
increasing strength, F1′′ *measured* with the
construction exact on its own population. The penalty
axes (λ, λ_w, k, τ) stay deferred as open questions with
no sweep licensed (SI-32 §6), and partial-removal
projection stays rejected as a next slice on its two
independent grounds (SI-32 §8: rank at every β > 0;
linear interpolation between two aligned, REJECTED
endpoints at the binding probe — derivations, labelled
as such there and here).

SI-33's bearing on all of this is precisely delimited,
in both directions:

- **It does not reopen any of it.** SI-33's decisive run
  used composition `"none"`; no penalty, projection, or
  blend was built, run, or reported as its treatment,
  and its candidate is not a readout movement, so the
  line's readings (which are about readout movement
  pairs through the shared fold) neither gain nor lose a
  data point. A composition mechanism blended with a
  target change would have been two variables in one run
  — SI-32 §17's non-goal, honoured by SI-33 and carried
  by this RFC.
- **It does not extend the closure either.** The line's
  falsifications are scoped — *through this readout, for
  this signal family, at this scale* (SI-32 §5.1) — and
  SI-33 measures no upstream movement at predecessor
  scale, so "composition is closed upstream too" is a
  sentence no run supports. §3.3 records the scoped
  statement: at ‖m2‖ ≈ 0.0105, no alignment claim in
  either direction is measurable. If a future
  non-readout candidate ever moves at predecessor scale
  (§6's audit is how the project learns whether one
  can), the composition question for *its* movement pair
  would be a new argument on a new record — not a
  revival of this line, and not pre-judged by it.

Closed and deferred as carried; §15 recommends
elsewhere.

---

## 12. Candidate (i): per-probe anchoring; sequencing; magnitude as a dial; a minimum-movement floor

**Verdict up front: the first three REJECTED / carried
exactly as SI-32 §§9–11 left them, with SI-33 supplying
no new evidence for any; the fourth — a minimum-movement
floor, the one genuinely new candidate SI-33's result
suggests — argued here on its own merits and REJECTED.**
Per candidate:

- **Per-probe anchoring: SPLIT, carried (SI-32 §9).**
  Gate-side per-probe accounting REJECTED (it reprices
  the judge); probe-side training against per-probe
  targets REJECTED (invariant 9); per-input accounting
  on allowed inputs ADOPTED only inside a slice that
  depends on it — in SI-35 that is the audit's
  per-population, per-head accounting (§15), not a slice
  of its own. SI-33 adds nothing for the first two
  readings: its diagnostics confirm the verdict's figure
  at the binding probe, as diagnostics did before, and
  temptation is not evidence.
- **Probe-partitioned sequencing: REJECTED (carried,
  SI-32 §10).** SI-33 does not touch either reading.
  Partitioning by probe behaviour reads the judge;
  partitioning by allowed-input behaviour is SI-26
  §10's gate-fitting search in sequence form — and the
  audit §6 recommends is its lawful residue done
  honestly: measuring, on allowed inputs, what each
  tensor *can* express, published as a record — as
  opposed to sequencing writes by where they move.
- **Magnitude discipline as a training dial: REJECTED
  as a slice (carried, SI-32 §11); ADOPTED as
  accounting.** SI-33 sharpens the accounting side once
  more, in the opposite direction from SI-31: SI-31's
  candidate *grew* under construction while failing
  worse; SI-33's candidate is the record's largest
  base-anchored norm and *passed* by barely moving — so
  neither "improvement by shrinking" nor "passage by
  growing" is available as a reading of any run, and
  §16 criterion 6 requires candidate norm and ‖m2‖
  beside every figure the audit's readings rest on, as
  every prior slice required beside every drift figure.
  Magnitude as a dial aimed at the headroom remains the
  standing gate-fitting non-goal.
- **A minimum-movement floor (new in this RFC):
  REJECTED.** The candidate, stated at full strength:
  SI-33's ADMIT is, in movement terms, nearly the
  identity (F3′′′ measured: the destination traded away
  at ~131×), so — the proposal runs — the gate should
  refuse candidates whose movement falls *below* a floor
  (on ‖m2‖, on composed-drift change, or on held-out
  completeness), lest trivial writes land as successes.
  The proposal correctly identifies a real property of
  the record and then reaches for the one instrument the
  project's invariants reserve: **the judge.** Three
  grounds, each sufficient. (i) *Timing:* the floor
  would be priced in this RFC, immediately after the
  first candidate in the project's history it would have
  refused — repricing the judge after a verdict, in the
  failing direction, which invariant 8 (as §2 now
  states it in both directions) forbids for exactly the
  reason it forbids repricing after a failure: in form
  it is indistinguishable from moving `identity_tol`
  after a drift failure, however sound its motivation.
  (ii) *Semantics:* the gate's arms price what a
  candidate *breaks* (identity, capability, safety);
  completeness-toward-a-signal is a training property,
  already measured and already published under its own
  labels by the F3-family readings in every slice since
  SI-29 — the record does not hide SI-33's trade; it
  headlines it. A gate that also priced *ambition* would
  conflate the two records §18 item 1 exists to keep
  separate, and would make every future ADMIT
  uninterpretable in the opposite direction (passed
  *and* sufficiently ambitious, under a floor chosen by
  whom, from which record?). (iii) *The honest
  instrument already exists and is this RFC's
  recommendation:* what the floor proposal actually
  wants to know — how much movement is even *available*
  through each tensor at the standing budgets — is a
  measurement, and §6 takes it, before the fact, for
  every tensor, with its readings pre-registered. A
  project that knows its expressibility map does not
  need a floor to recognise a near-identity candidate;
  it needs — and has — the labelling discipline of §18
  item 26. Rejected, with the underlying observation
  adopted as accounting and as §6's motivation.

---

## 13. Candidate (j): Glimmer probing

**Verdict up front: REJECTED, not reopened.** Rejected
at SI-13 §7/§9 and restated in every successor RFC
including SI-32 §13.2; SI-33 changes nothing, and this
RFC records the standing pressure honestly rather than
letting it arrive as a surprise (SI-32 §19 item 9):
SI-33 *succeeded* on the miniature — first non-readout
target, first landed two-delta stack — and success on
the miniature is precisely the condition under which
the "why not Glimmer" question renews. The answer does
not soften: every mechanism in the seven-run record —
penalties, projection, the target axis, and §6's
proposed audit alike — is a miniature mechanism and
claims nothing at Glimmer scale; the Glimmer seed is
untouched; its deltas still escalate with checks 3/4
"not evaluated"; there is no Glimmer forward path, no
probe suite, and no gate reading beyond escalation in
the current design, and this RFC does not propose
faking one, by miniature results, by analogy, or by any
other route. Success *or* continued failure on the
miniature does not soften the answer. Not reopened.

---

## 14. Decision summary

| Candidate | Verdict | One-line reason |
|---|---|---|
| (a) Small real safetensors base | **DEFERRED — operator decision** | SI-18 §3.7 audit: A-1 to A-4 all still unmet; naming a base requires a download this scope forbids and a decision this RFC explicitly does not take; not recommended as buildable now (§4). |
| (b) A second non-readout target, argued now | **DEFERRED (sequenced behind §6)** | SI-33 leaves every other tensor an *unargued* candidate, and the ground its argument now requires — measured inertness — does not exist for any tensor; naming one now would repeat SI-33's one avoidable cost (§5). |
| (c) Branch-contribution and expressibility audit | **RECOMMENDED — SI-35** | The only candidate whose premise SI-33 strengthened: the quantity that decided SI-33's outcome (branch inertness at standing budgets) is measurable before the fact, on allowed inputs, for every tensor, at the price of forward passes (§6). |
| (d) The representation question (rank / identity-alone judging) | **ARGUED (SI-32) — trigger EVALUATED, NOT FIRED — DEFERRED** | SI-33's candidate is rank 8 and fails on no line; SI-31's still fails identity independently at ~1.91× tol; no verdict on record turns on rank (§7). |
| (e) Transformer squash (miniature) | **DEFERRED with a trigger** | Precondition newly met (first authored two-delta stack) — that deferral ground is retired — but the mixed-target carrying question, the four-axis gate context, and the named-tensor tolerance verify are all unargued, and at depth 2 ≤ `keep` 3 a squash would not yet run on the only real stack (§8). |
| (f) A third authored cycle | **DEFERRED** | No lawful configuration exists: both signals are spent, re-training a landed signal is undefined in the current semantics, and a third signal is unargued on any ground independent of the gate (§9). |
| (g) Additive / multi-target authorship | **REJECTED as a next slice (carried)** | Still buys its premise with a policy loosening and zero evidence under it; SI-33's mixed-target *stack* is two lawful single-target deltas in sequence, not evidence about a multi-target *delta* (§10). |
| (h) Composition line / penalty axes / partial removal | **CLOSED / DEFERRED as carried** | SI-33 used composition `"none"` and measures no readout movement pair, so it neither reopens the closed line nor extends its closure upstream (§11). No sweep is licensed. |
| (i) Per-probe anchoring; sequencing; magnitude as a dial; a minimum-movement floor | **REJECTED (first three carried); floor REJECTED as new** | The first three reprice or read the judge and gain no evidence from SI-33; the floor reprices the judge after a *pass*, conflates gate arms with training completeness, and asks for a measurement §6 already takes honestly (§12). |
| (j) Glimmer probing | **REJECTED** | Rejected at SI-13 §7/§9 and restated in every successor RFC; SI-33's miniature success renews the pressure and does not soften the answer; no forward path exists and none is faked (§13). Not reopened. |

The seven routes to a landed second write that the
project has now run — a new signal under settled
semantics (SI-23), the same signal restacked (SI-21),
new target semantics (SI-25), a mean composition term
(SI-27), a worst-input composition term (SI-29), a
candidate constructed by full removal of the indicted
component (SI-31), and a first non-readout target
(SI-33) — have been judged by the same unchanged
accounting, and the seventh landed. What the seventh
established is narrower than its verdict line and more
useful than a plain failure would have been: the loop's
machinery operates end to end against an upstream
tensor, and the tensor it operated on cannot express
the signal it was given, at budgets no candidate may
exceed — a property of the *model*, measurable without
training anything. §6's route is the eighth the project
takes, and the first that is not an attempt to land a
write at all: it prices, for every place an authored
update could act, how much acting there can move this
model — so that the ninth route, whichever §15's
readings leave standing, is argued from a map instead of
from a hope. No ninth route is proposed by this RFC,
and §§10–12 record the tempting ones (a looser policy,
a partitioned sequence, a judge with a floor) and why —
on SI-33's evidence as on SI-31's — they are still not
candidates.

---

## 15. The recommended slice: SI-35 (outline)

**SI-35 — Branch-contribution and expressibility audit
of the miniature: for every weight matrix in the model,
the measured share of the residual stream its branch
contributes, the measured scale of the activations it
consumes, and the derived bound on what a rank-8 update
at the standing norm budget could express through it —
on allowed populations only, at named heads, under
pre-registered readings — with no candidate trained, no
gate evaluation performed, and no target approved.**

In scope:

1. **Method and populations, stated before any figure is
   trusted (the Phase-0 discipline, applied to a
   measurement slice).** (i) *Populations:* allowed
   inputs only — signal 1's train partition (32
   sequences) and signal 2's train partition (32
   sequences) as the primary populations; both signals'
   held-out partitions (16 each) reported only, under
   their own labels, never merged into a primary figure.
   No probe input, probe logit, or probe-derived
   quantity is read by any audit measurement, ranking,
   or reading (§6). (ii) *Heads:* every figure is
   measured at a named head — `base`, `d0001`, and
   `d0002` (the landed stack, weights overlaid through
   the existing apply path) — and no figure is presented
   without its head. (iii) *Positions:* the last position
   is primary (the loss reads last-position logits only,
   as in every prior slice, and SI-33's attribution is a
   last-position record); all-position statistics are
   reported separately, labelled, and no reading rests
   on them. (iv) *Stop rule:* if item 3's reproduction
   of SI-33's published attribution cannot be reconciled
   with the audit's method, the slice stops and
   publishes the discrepancy as its result — the
   Phase-0 stop discipline of SI-29/SI-31/SI-33, applied
   a fifth time.
2. **Branch-contribution measurements (measurements).**
   For each residual-adding branch — per layer, the
   attention block's additive output and the MLP block's
   additive output (4 branches), plus the embedding sum
   that forms the initial residual stream (1 input-side
   row, labelled as such) — over each population × head
   cell: the branch output norm and the residual-stream
   norm it adds into (per-example, last position;
   mean and max published), and their ratio. The final
   layer's MLP branch on the primary population at
   `d0001` is the method's yardstick row: its figures
   must sit beside SI-33's (§16 criterion 3).
3. **Reproduction of SI-33's attribution (acceptance
   anchor).** Under the audit's method, on signal 2's
   train partition at head `d0001`: ‖s‖ and ‖h_mid‖
   ranges reproducing SI-33's published ranges
   (0.040–0.064; 1.97–2.02), and at base on SI-33's
   representative sequence the MLP branch output norm
   and residual norm reproducing 0.0074 and 2.00 — or
   any difference reconciled in print, with the method
   difference named (population, position, or statistic),
   before any other figure is trusted.
4. **Per-matrix input-activation records
   (measurements).** For each of the model's 17 weight
   matrices — the 14 in-stream matrices (per layer:
   `q_proj`, `k_proj`, `v_proj`, `o_proj` [32, 32];
   `gate_proj`, `up_proj` [64, 32]; `down_proj`
   [32, 64]), the readout `lm_head.weight` [32, 32]
   (tabled as the record's yardstick target, not as a
   candidate), and the two embedding matrices
   (`embed_tokens` [32, 32], `embed_positions` [16, 32],
   tabled separately under item 6's caveat) — the norm
   of the activation the matrix consumes, per population
   × head cell (min / mean / max, last position): for
   `down_proj`, the SwiGLU output `s`; for
   `gate_proj`/`up_proj`, the post-attention norm
   output; for `q/k/v_proj`, the input-layernorm output;
   for `o_proj`, the concatenated attention output; for
   `lm_head`, the final-norm output. The 13 vector
   tensors (per layer: 4 attention biases + 2 norm
   weights; plus the final norm weight) are inventoried
   with their blind-set status — the derived blind set
   is exactly both layers' `k_proj.bias` — and are not
   silently dropped from the record.
5. **Expressibility bounds (derivations, labelled as
   such wherever they appear).** For each in-stream
   matrix W, the local bound B(W) = 0.9 × (max input
   norm over the primary population at the named head)
   — the standing norm budget (0.9, as SI-19/SI-33
   trained under it) times the largest activation the
   matrix is measured to consume, bounding ‖ΔW·x‖ for
   any lawful candidate exactly as SI-33's derivation
   bounded the down-projection's (0.9 × 0.064 ≈ 0.058).
   Each bound is published beside the two cited
   yardsticks: the signal correction's logit-space norm
   0.15·√2 ≈ 0.212 and SI-33's down-projection bound
   ≈ 0.058. **Stated limitation, in the as-built, not
   the footnotes:** a local bound prices movement at the
   tensor's own output; downstream transfer to logits is
   *not* measured by the audit for any tensor except
   where a landed delta already measures it (the
   readout, and the final down-projection by SI-33's
   ‖m2‖ record) — every other tensor's transfer
   coefficient is listed as unknown, never estimated.
   The embedding matrices are excluded from the bound
   table's form: their input is an index, not an
   activation, so ρ·‖x‖ does not describe them; their
   rows state this and give the row-selection form of
   the bound instead (a derivation, labelled).
6. **Backward-path and checkability inventory
   (definitions and arithmetic, labelled).** For each
   in-stream matrix: the chain elements a future
   backward derivation must traverse between the tensor
   and the loss (RMSNorm Jacobians, the SwiGLU
   derivative, the attention Jacobian, layer traversals —
   counted, not argued); the rank-8 factor-entry count
   8·(rows + cols) (512 for [32, 32]; 768 for [32, 64]
   and [64, 32]; 384 for [16, 32]); blind-set membership
   (no weight matrix is in the derived blind set — the
   blind set is biases only — stated as the derived fact
   it is); and gate-measurability under SI-17's mapping.
   This is SI-32 §12.1's argument grounds, tabled for
   every tensor so a future target RFC cites a record
   instead of re-deriving one.
7. **The pre-registered readings** (stated here, before
   any audit figure exists; applied as written, §18
   item 21). The *primary cell* for every reading is
   signal 2's train partition at head `d0001` — the
   decisive configuration's population and head, the
   configuration any second-target argument must first
   survive; all other cells are published for the
   record and for R3.
   - **R1 — a less-inert single target is arguable.**
     R1 is **measured** iff at least one in-stream
     non-readout matrix satisfies BOTH, on the primary
     cell: (i) its branch's contribution ratio (max
     per-example, item 2's statistic) is at least 0.04
     — stated as ≈10× the final-MLP branch's measured
     ~0.4% headline (0.0074 / 2.00 = 0.0037; *derivation*
     from SI-33's cited figures, the multiplier chosen
     here, in advance, not from any audit figure); and
     (ii) its local bound B(W) ≥ 0.212, the signal
     correction's norm (SI-33's derivation). Consequence
     stated in advance: the qualifying matrices — and
     only they — form the shortlist a future RFC
     (SI-36) may argue a single second target from,
     using item 6's inventory for the backward-path
     grounds; R1 measured is **not** an approval of any
     target, a prediction about any candidate, or a
     gate figure of any kind (§18 item 27).
   - **R2 — inertness is uniform at the standing
     budgets.** R2 is **measured** iff NO in-stream
     non-readout matrix satisfies R1's pair on the
     primary cell. R1 and R2 are complementary by
     construction — exactly one of them is measured,
     stated here so the audit cannot end in an
     unregistered middle — and R2's consequence is
     stated in advance with its scope attached: the
     single-target upstream-authorship line is then
     *expressibility-closed on this miniature at the
     standing budgets* (a derivation-scoped conclusion
     from measured inputs, labelled as such), and the
     next frontier must be argued from a different
     premise — §8's squash trigger, §9's semantics, or
     §4's operator decision — in a future RFC this one
     does not pre-write.
   - **R3 — qualification is head-dependent.** R3 is
     **measured** iff the set of matrices satisfying
     R1's pair on signal 2's train partition differs
     between head `base` and head `d0001` (the `d0002`
     cell is published beside them). Consequence stated
     in advance: if R3 is measured, every future target
     argument must name its head and may not quote a
     base-measured qualification at a landed head, or
     the reverse (§18 item 8); if R3 is not measured,
     qualification is head-stable *across the three
     measured heads only* — a claim scoped to those
     heads, never to depth in general.
8. **Deliverables and the shortlist's semantics.** The
   slice ships: a read-only measurement module (forward
   passes and weight overlays only — it invokes no
   training entry point, writes no blob, opens no gate
   evaluation, and leaves any bundle it is run against
   byte-identical, all asserted by test); its tests
   (§16); and an as-built document publishing the full
   tables (items 2, 4, 5, 6) and the readings applied as
   registered. The audit's output is at most a
   *shortlist for an argument*: SI-35 names no approved
   target, performs no decisive run, and claims no
   candidate-shaped result of any kind. If the shortlist
   is non-empty, SI-36 is a target RFC arguing one
   tensor from this record; if it is empty (R2), SI-36
   is an RFC arguing from R2's stated consequence.
   Either way the cadence continues from measurements,
   which is the point of the slice.

---

## 16. Acceptance criteria for SI-35

1. **Coverage, complete and inventoried** — all 30
   tensors appear in the audit's record: the 17 weight
   matrices tabled with shapes, the 14 in-stream
   matrices carried through every figure (branch,
   input, bound, inventory), the readout and the two
   embedding matrices tabled under their own labels, and
   the 13 vector tensors inventoried with blind-set
   status. No tensor is silently omitted; any figure a
   tensor cannot carry is recorded as "not applicable"
   with the reason — never as 0.0, never as an omission
   (§18 item 2).
2. **Populations and probe exclusion, asserted** — the
   audit's input sets are exactly the named partitions
   (asserted by test, in the shape of the prior slices'
   probe-blindness spies): signal 1 + signal 2 train
   partitions as primary populations, held-out partitions
   reported only, and a perturbed-probe-artifact bundle
   copy produces a **byte-identical** audit record. No
   probe input, probe logit, held-out sequence from
   either probe suite, or probe-derived quantity enters
   any figure, ranking, or reading.
3. **Reproduction of SI-33's attribution** — the audit
   reproduces, under its stated method, SI-33's
   published ranges on signal 2's train partition at
   `d0001` (‖s‖ 0.040–0.064; ‖h_mid‖ 1.97–2.02) and its
   base representative-sequence figures (branch output
   0.0074; residual 2.00), or reconciles every difference
   in print with the method difference named — before
   any non-yardstick figure is presented as trusted.
   If reconciliation fails, the slice's stop rule
   (§15 item 1 (iv)) applies and the discrepancy is the
   published result.
4. **Measurements and derivations, labelled** — branch
   ratios and input norms are published as measurements
   with population, head, and position-statistic on
   every figure; bounds are published as derivations
   (formula, budget, and yardsticks stated: 0.9;
   0.15·√2 ≈ 0.212; ≈ 0.058); downstream transfer is
   recorded as measured only for the readout and the
   final down-projection (by the landed deltas' records)
   and as **unknown** for every other tensor.
5. **Readings applied as registered** — R1/R2/R3 are
   applied under §15 item 7's rules exactly as written
   (primary cell, thresholds, and populations unchanged
   after the figures exist); exactly one of R1/R2 is
   reported measured, with the qualification table that
   decides it published in full; R3's base-vs-`d0001`
   comparison is published either way. No reading is
   rounded, re-thresholded, or reported without its
   registered qualifier (§18 items 21, 23).
6. **Magnitude accounting** — wherever an audit figure is
   compared against a landed candidate's record, the
   candidate's norm and ‖m2‖ stand beside it (SI-33:
   norm 0.30801411790830574, ‖m2‖ 0.010469039280455053;
   SI-25 control: norm 0.2515524481203623, ‖m2‖
   0.7816722548913912), so no bound is ever quoted as if
   it were a movement and no movement as if it were a
   bound (§18 items 14, 27).
7. **No side effects, asserted** — an audit run writes
   no blob, performs no gate evaluation, appends no
   journal entry, and changes no manifest: bundle state
   is snapshot-asserted byte-identical before/after, in
   the shape of SI-12's unchanged-bundle tests; the
   audit module invokes no training entry point
   (asserted); both signals' definitions, all four
   composition configurations, both anchorings, both
   authorship targets, and every policy default are
   untouched (asserted by the controls' continued exact
   reproduction under the existing suite).
8. **Determinism** — the audit repeated from the seed
   reproduces every published figure exactly (the audit
   is a deterministic function of the seed weights, the
   landed heads' overlays, and the named partitions; no
   sampling, no wall-clock dependence in any figure).
9. **Untouched scope** — the full existing suite is
   green; `si squash` still fails closed on the landed
   transformer stack (existing suite); `si serve` still
   returns 501 for every transformer bundle (existing
   suite); the demo path is unchanged; no CLI wiring is
   added; the Glimmer seed is untouched.

---

## 17. Non-goals

Explicit, and binding on SI-35 and on any slice this RFC
is cited to justify:

- **No candidate, no training, no decisive run.** SI-35
  authors nothing, trains nothing, and judges nothing.
  It is a measurement slice; any text describing its
  shortlist as an approved target, its bounds as gate
  figures, or its readings as verdicts is a labelling
  defect (§18 item 27).
- **No second target under this RFC.** Exactly zero new
  authorship-target values are added by SI-35; the axis
  keeps serving exactly `"lm-head"` and
  `"final-mlp-down"`. A second non-readout target is a
  future RFC's (SI-36's) single argued tensor, argued
  from SI-35's record if that record leaves one
  standing (§5).
- **No Glimmer anything.** No Glimmer forward path,
  probes, training, authorship, or audit (§13). The
  audit is a miniature measurement that claims nothing
  at Glimmer scale, and no candidate in this RFC touches
  the Glimmer seed except to restate that it is
  untouched.
- **No real-base work.** Candidate (a) is deferred with
  audited preconditions and an operator-owned download
  decision (§4); SI-35 builds nothing toward it, and no
  download is performed, scheduled, or implied.
- **No gate, policy, rank, or tolerance change — in
  either direction.** `rank_cap` stays 8 (§7);
  `identity_tol` and `capability_eps` stay 0.25; no
  minimum-movement floor is built (§12); no candidate is
  "judged on identity alone" under this RFC, and
  SI-31's candidate is never re-judged.
- **No composition revival.** No penalty term (mean,
  worst-case, or any third statistic), no projection
  (full or partial), and no blend of any of them with a
  target change is built, run, or reported under this
  RFC (§11). The four existing composition
  configurations' semantics are untouched.
- **No parameter tuning, no sweep.** No sweep of λ, λ_w,
  k, τ, β, rank, alpha, step count, learning rate, norm
  budget, or correction scale; no second decisive run at
  a different value of any of them; and no sweep *of the
  audit's own thresholds* — R1's 0.04 and 0.212 are
  registered in §15 before the figures exist and are
  not re-chosen after them.
- **No third cycle, no third signal, no signal
  redefinition.** Both signals' ids, streams,
  corrections (scale 0.15 included), and partitions are
  byte-for-byte as SI-19/SI-23 defined them; SI-35 reads
  their partitions as populations and changes nothing
  about them (§9).
- **No transformer squash.** SI-12's fail-closed refusal
  stands throughout SI-35 (§8); §8 is a deferral
  analysis with a stated trigger, not an implementation
  brief — even though its precondition now exists.
- **No probe contact in measurement.** No probe input,
  probe logit, probe movement, or probe-derived quantity
  enters any audit figure, ranking, or reading (§§6, 12;
  §16 criterion 2). The audit is held to the training
  standard, not the diagnostic standard, because its
  record shapes a future candidate's defining
  configuration.
- **No autonomous selection — of signal, of anchoring,
  of composition configuration, of authorship target, or
  of audit population.** Every population, head, and
  statistic in SI-35 is a recorded, driver-level,
  pre-registered choice; no perception-driven,
  verdict-driven, or figure-driven rule chooses any of
  them after the figures exist.
- **No serving, no CLI wiring.** Transformer bundles
  keep returning 501; `si cycle`/`si run` keep their
  demo/mock minds. SI-35 is a library-and-tests slice
  like SI-19, SI-21, SI-23, SI-25, SI-27, SI-29, SI-31,
  and SI-33.
- **No scale claim of any kind.** The audit's figures
  are facts about a constructed, deterministically
  seeded, not-trained miniature, at three named heads,
  on two constructed signals' partitions. Nothing in
  them is claimed about any real model, any trained
  base, or Glimmer — including by analogy, and including
  the direction "real models are less inert", for which
  this record contains no evidence in either direction.

---

## 18. Honesty and labelling requirements for any future slice

Items 1–25 restate RFC SI-32 §18 (itself restating
SI-30 §17, SI-26 §17, and SI-24 §12 with additions) in
compressed form, with the numbering preserved so
existing citations resolve; items 26–27 are new in this
RFC. All twenty-seven bind SI-35 and any slice this RFC
is cited to justify.

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
    interpolation arguments like SI-32 §8.3's, rank
    arithmetic like SI-32 §8.2's, expressibility bounds
    like §15 item 5's) are labelled derivations wherever
    they appear, and are never quoted with the authority
    of a measurement.
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
26. **An ADMIT's attribution travels with the ADMIT
    (new).** A landed candidate's verdict is never
    summarised without the measured attribution its own
    slice published for it, where one exists. The standing
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
27. **A measurement-only record is not a verdict record
    (new).** Figures produced by a slice that authors no
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

---

## 19. Risks and open questions

Each item names what would retire it. Numbers this RFC
deliberately does **not** supply are listed as open
questions to measure — supplying them here would be
inventing them.

1. **The audit's bounds could mislead in the direction
   of optimism.** A local bound is an upper bound at a
   tensor's own output, before downstream attenuation
   that the audit deliberately does not estimate (§15
   item 5's stated limitation). A matrix can qualify
   under R1 — bound ≥ 0.212, branch ratio ≥ 0.04 — and
   still deliver a candidate whose logit movement is
   attenuated below usefulness, exactly as the
   down-projection's ≈ 0.058 pre-attenuation bound
   became ‖m2‖ ≈ 0.0105 after it. R1 measured therefore
   licenses an *argument* (SI-36), never an expectation;
   a slice that reports R1 as "a workable target exists"
   has misread its own RFC. *Retired by:* nothing in
   SI-35 — only by the future target slice's own Phase 0
   and decisive record, if one is ever argued.
2. **The audit's thresholds are judgement calls, and are
   registered as such.** R1's pair (branch ratio ≥ 0.04,
   ≈10× SI-33's measured headline; bound ≥ 0.212, the
   signal's own correction norm) is argued from SI-33's
   cited figures in §15 item 7, but the 10× multiplier
   is this RFC's choice, not a measurement. A different
   multiplier would draw the shortlist's boundary
   elsewhere. The defence is procedural, and it is the
   project's standing one (§18 item 21): the boundary is
   stated in advance, in the record's own terms, applied
   as written, and the full qualification table is
   published so any reader can re-draw the boundary and
   see what would change. *Retired by:* SI-35's
   published table; the choice itself stays on this
   RFC's record permanently.
3. **Inertness may be a property of the construction
   seed, not of the architecture.** The miniature's
   weights are drawn at small scale (construction
   `small(...)` draws at 0.02 in `si/tiny_seed.py`), and
   branch ratios are ratios within that one constructed
   model. The audit measures *this* model — the only one
   the gate has ever judged — and its readings are
   scoped to it (§17). Whether a differently-seeded
   miniature of the same architecture would show the
   same map is unknown and is **not** an audit question:
   re-seeding the subject to make the measurement more
   general would change the subject every prior run
   measured. *Retired by:* nothing in SI-35; recorded so
   the audit's scope is not inflated after the fact.
4. **R2, if measured, will be misread as "authorship is
   impossible" unless its scope travels with it.** R2's
   registered consequence is expressibility-closure for
   *single-target, upstream, rank-8, budget-0.9*
   candidates on *this* miniature for *this* signal
   family's correction scale — five qualifiers, each
   load-bearing. The readout target already falsifies
   any unqualified version (the SI-25 control expresses
   the correction fully; it is refused on composition,
   not expressibility). *Retired by:* §18 item 27's
   discipline and R2's own registered wording; the risk
   is recorded here because successor documents are
   where qualifiers go to die.
5. **The representation trigger may still fire someday —
   and must not be improvised around.** §7 found it
   unfired. If a lawful future candidate ever passes
   identity and capability while failing only on rank,
   SI-32 §7.5's question becomes a change proposal in
   its own RFC, argued against SI-32 §7.3 with the
   candidate's figures published, and never applied
   retroactively to SI-31, and never resolved inside the
   candidate's own slice. *Retired by:* that RFC, if the
   trigger ever fires; until then this item stays open
   by design.
6. **Squash's deferral now rests on design obligations,
   not on a missing precondition — and obligations can
   quietly become permanent deferral.** §8 retired the
   precondition ground and replaced it with three
   unargued obligations plus a depth observation. The
   stated trigger (a third landed delta, depth exceeding
   `keep`, or an RFC arguing the obligations as its main
   business) is the guard: if none ever fires, the
   honest statement is that squash was never needed at
   the depths the project reached — not that it was
   always about to happen. *Retired by:* §8's trigger,
   if it fires; otherwise by a future RFC recording the
   permanent-deferral honestly.
7. **A mislabelled configuration is worse than a wrong
   one.** The failure mode the cadence has named since
   SI-26 — a candidate trained under one configuration
   and journaled under another — gains a measurement-era
   variant in SI-35: an audit figure measured on one
   population or at one head and quoted under another
   would corrupt the shortlist every future target
   argument cites, silently and at one remove.
   *Retired by:* §16 criteria 2, 4, and 8 — asserted
   input sets, per-figure population/head labels, and
   exact determinism on repetition.
8. **Probe blindness still bounds the judge.** RFC SI-13
   §10.1 stands, now across signals, anchorings, four
   composition configurations, and two authorship
   targets: 24 inputs, 28/30 tensors measurable, blind
   set derived and guarded. A landed two-delta stack is
   *evidence* of continuity sampled the same way each
   cycle, not a proof of it. SI-35 does not widen the
   judge; it measures the subject the judge judges, on
   the populations training may read, and the distinction
   between those two measurement programmes is §6's
   probe ruling.
9. **Depth and Glimmer pressure arrive with SI-33's
   success already here.** At `max_stack_depth` 8 with
   transformer squash still deferred (§8), the landed
   second authored delta makes the stack depth 2 and the
   compression question concrete for the first time —
   while the squash gate context remains unbuilt and
   must now be designed against four recorded axes and a
   mixed-target stack. And per the standing item in
   every successor RFC: success on the miniature renews
   the "why not Glimmer" pressure, and the answer does
   not soften because the miniature's authorship has
   gained a target and a stack. There is no Glimmer
   forward path; none is proposed here, and none may be
   improvised by a successor slice (§13).
10. **The deferred axes' unknowns remain unknown.**
    Candidate (a)'s A-1 to A-4 are audited unmet (§4);
    nothing in SI-35 or this RFC scopes them, and the
    download decision is the operator's, explicitly not
    taken here. The λ / λ_w / k / τ questions (§11)
    likewise remain open by design: each is retired only
    by a later RFC arguing a single value from the
    published records of the configurations that now
    exist — seven cycle-2 runs, two landed deltas — never
    by a sweep. Partial removal stays unmeasured,
    deliberately (SI-32 §8, carried at §11): its grounds
    are a derivation and rank arithmetic, labelled as
    such, and SI-35's audit — which measures what
    *single* tensors can express — supplies no β
    evidence in either direction.

---

## 20. Summary

After SI-33, SI can originate a miniature delta, judge
it, land it, originate a second one on top of it under a
second signal, under two target anchorings, under either
of two composition-aware losses, under a construction
that removes the indicted component exactly, and —
the seventh route — under a second authorship target
acting upstream of the readout, trained by a derivation
checked entry by entry over 768 factor entries at two
heads, judged by the same unchanged gate — and that
seventh second write **landed**: candidate
`sha256:d7af7f1f…e625`, norm 0.30801411790830574,
identity drift 0.17195402272045612 ≤ 0.25, capability
regression 0.001467481255531311 ≤ 0.25, head `d0002`,
the first authored two-delta stack in the project's
record, VERIFY-backed and unwound by test. And its own
as-built prices the landing in the same document that
reports it: F1′′′ not measured (both clauses fail — at
movement scale ‖m2‖ = 0.010469039280455053, no alignment
claim in either direction is measurable), F2′′′ not
measured as a reading (transfer clause holds, maximum
clause fails — the maximum moved because almost nothing
was added to it), F3′′′ **measured** (the destination
traded away: residual 0.0027387202915082178 against
boundary 2.087217177504918e-05, ~131×), with the
attribution measured on the record — the final layer's
MLP branch contributes ~0.4% of the residual stream it
adds into (‖s‖ 0.040–0.064 against ‖h_mid‖ 1.97–2.02),
and no candidate within the standing budgets can
express this signal's correction through that tensor in
either direction (bound ≈ 0.058 against required
≈ 0.212, a derivation from the measured inputs). Of the
candidate frontiers this RFC evaluated against that
evidence: a small real base stays deferred, its four
SI-18 preconditions audited and all still unmet, and
its download decision stays the operator's, explicitly
not taken and not recommended as buildable now (§4); a
second non-readout target is deferred, sequenced behind
the measurement its argument now requires, because
SI-33 leaves every other tensor an unargued candidate
and the missing ground is inertness (§5); the
representation question's trigger (SI-32 §7.5) is
evaluated against the first new candidate since it was
written and found **not fired** — SI-33's candidate is
rank 8 and fails on no line, SI-31's still fails
identity independently at ~1.91× the tolerance — so
`rank_cap` stays 8 and the question stays deferred with
its trigger intact (§7); transformer squash's
precondition is met for the first time and that
deferral ground is retired, but the candidate stays
deferred with a precise trigger, because the
mixed-target carrying question, the four-axis squash
gate context, and the named-tensor tolerance verify are
unargued, and at depth 2 against `keep` 3 no squash
would yet run on the only real stack (§8); a third
authored cycle is deferred, there being no lawful
configuration for one in the current semantics (§9);
additive / multi-target authorship stays rejected, the
landed mixed-target *stack* being two lawful
single-target deltas in sequence and no evidence about
a multi-target *delta* (§10); the composition line,
penalty axes, and partial removal stay closed or
deferred exactly as carried, SI-33 measuring no readout
movement pair in either direction (§11); per-probe
anchoring, partitioned sequencing, and magnitude as a
dial stay rejected as carried, and the one new candidate
SI-33 suggests — a minimum-movement floor on the gate —
is argued and rejected, as a repricing of the judge
after a pass that conflates the gate's arms with
training completeness, its honest observation adopted
instead as §6's motivation and §18 item 26's labelling
rule (§12); Glimmer probing stays rejected, not
reopened, its pressure renewed by miniature success and
its answer unsoftened (§13); and **a measurement-only
branch-contribution and expressibility audit — every
weight matrix in the miniature priced, on allowed
populations only, at base and at both landed heads, for
the share of the residual stream its branch contributes,
the scale of the activations it consumes, and the
derived bound on what a rank-8 update at the standing
0.9 norm budget could express through it, with SI-33's
published attribution as its reproduction anchor, its
backward-path inventory tabled for every tensor, and
its three readings (R1 a less-inert target is arguable;
R2 inertness is uniform at the standing budgets —
exactly one of R1/R2 is measured by construction; R3
qualification is head-dependent) registered in advance
with their consequences stated in both directions — is
recommended as SI-35** (§§6, 15, 16). Its plausibility
argument is stated with its limits in the same sections
that make it: it is the cheapest slice in the space
(forward passes on populations that already exist, no
derivation, no training, no gate contact), it is the
only one whose premise SI-33 strengthened, and it is
the prerequisite every authorship candidate after it
needs — because the one avoidable cost in SI-33's
method was discovering after a full slice a property of
the model that was measurable before it. Whether the
audit leaves a second target standing is genuinely
unknown in both directions, and §15 registers the
readings so the answer is publishable either way:
a shortlist argued in SI-36, or an honest closure of
the single-target upstream line at miniature scale.
The deepest landed authored stack is depth 2; the base
stays frozen, the bytes are never duplicated, no
tolerance or cap moves in either direction, the gate is
unchanged, and nothing is claimed that a test in this
repo cannot check.

---

*RFC SI-34 — design only. Implementation begins, if
approved, at SI-35 as outlined in §15. RFC SI-26 changed
what the loop accounts for while it trains; RFC SI-28
changed the statistic that accounting answers to; RFC
SI-30 changed the candidate itself, removing the
component the record indicted; RFC SI-32 changed where
an authored update acts, and the gate admitted the
result — and SI-33's own document told the truth about
that admission in the same breath: the branch the update
acted through barely moves the model it writes to, and
a write that barely moves its model has not reached its
signal, however cleanly it lands. This RFC proposes
changing neither the accounting, nor the statistic, nor
the candidate, nor the target — but the order of
operations: measure what every tensor in the model can
express before arguing that any of them should be
written to, so the next authorship slice, if the
measurements leave one standing, begins from a map of
the model it is actually writing — drawn on the inputs
it is allowed to read, at the heads it will actually
train against, with its readings registered before the
first figure exists.*
