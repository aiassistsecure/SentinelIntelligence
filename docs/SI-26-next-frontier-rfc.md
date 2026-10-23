# SI-26 RFC — The Next Frontier After Base-Anchored Targets
(Design Only)

> **Status: DESIGN ONLY.** This document evaluates candidate
> directions and recommends one next slice. It changes no
> production code, no test, no manifest schema, no policy
> default, and no gate behaviour. Nothing in this RFC is
> implemented unless a section explicitly says it exists today,
> in code, on `main`.
>
> **The one sentence that governs everything below:** SI-25
> proved the miniature's second signal can be authored at the
> landed head against the frozen base's targets — exactly one
> variable changed against SI-23's control run — and the gate
> REJECTED it again on cumulative identity drift, with
> published diagnostics attributing the rejection to
> *movement composition at the binding probe*: the
> base-anchored update opposes the first update in aggregate
> and at the control's worst probe, yet at `tiny-id-0` the
> two movements still add almost exactly — and this RFC must
> never be cited as evidence that any composition-aware
> training has been implemented or tested, that any authored
> two-delta stack exists, that a cycle-2 ADMIT is achievable
> by any route, or that any slice proposed here exists.
>
> **Scope discipline:** CPU only. No GPU, no cloud, no spend,
> no downloads. Terminology: this is machine learning
> infrastructure throughout.
>
> **Relationship to prior work:** RFC SI-24 evaluated the
> frontiers open after SI-23 and recommended base-anchored
> targets; SI-25 built it (PR #30, as-built in
> `docs/SI-25-base-anchored-targets.md`) and the gate's
> cycle-2 verdict was a measured REJECT under the new
> anchoring as it had been under the old, now attributed
> per probe under both. This RFC is SI-24's successor in
> the established cadence (SI-18 RFC → SI-19 build →
> SI-20 RFC → SI-21 build → SI-22 RFC → SI-23 build →
> SI-24 RFC → SI-25 build): it restates the verified state
> after SI-25, re-evaluates every SI-24 candidate against
> SI-25's measured evidence, gives the question SI-25's
> diagnostics have made precise — why movements that oppose
> in aggregate still add at the binding probe — the deep
> treatment (§6), and recommends exactly one next slice
> (§14). SI-24's rejections of Glimmer probing, of a reduced
> correction scale, and of additive / multi-target
> authorship are carried forward, not reopened, unless
> SI-25's evidence genuinely reopens them; §§10–12 record,
> per candidate, that it does not. The SI-9 RFC's §7 risks
> remain in force and are inherited, not relitigated.

---

## Table of contents

1. [Where SI stands after SI-25](#1-where-si-stands-after-si-25)
2. [Inherited invariants](#2-inherited-invariants)
3. [Candidate (a): a small real safetensors base — still deferred](#3-candidate-a-a-small-real-safetensors-base--still-deferred)
4. [Candidate (b): a second training signal — DONE as SI-23, closed](#4-candidate-b-a-second-training-signal--done-as-si-23-closed)
5. [Candidate (c): base-anchored targets — DONE as SI-25](#5-candidate-c-base-anchored-targets--done-as-si-25)
6. [The movement-composition question — the deep treatment](#6-the-movement-composition-question--the-deep-treatment)
7. [Candidate (d): movement-composition-aware training](#7-candidate-d-movement-composition-aware-training)
8. [Candidate (e): per-probe movement accounting — as diagnostics and as a gate change](#8-candidate-e-per-probe-movement-accounting--as-diagnostics-and-as-a-gate-change)
9. [Candidate (f): orthogonalization at the binding probe](#9-candidate-f-orthogonalization-at-the-binding-probe)
10. [Candidate (g): a third or different signal](#10-candidate-g-a-third-or-different-signal)
11. [Candidates (h) and (i): new target; additive / multi-target](#11-candidates-h-and-i-new-target-additive--multi-target)
12. [Candidate (j): transformer squash; candidate (k): Glimmer probing](#12-candidate-j-transformer-squash-candidate-k-glimmer-probing)
13. [Decision summary](#13-decision-summary)
14. [The recommended slice: SI-27 (outline)](#14-the-recommended-slice-si-27-outline)
15. [Acceptance criteria for SI-27](#15-acceptance-criteria-for-si-27)
16. [Non-goals](#16-non-goals)
17. [Honesty and labelling requirements](#17-honesty-and-labelling-requirements-for-any-future-slice)
18. [Risks and open questions](#18-risks-and-open-questions)
19. [Summary](#19-summary)

---

## 1. Where SI stands after SI-25

Everything in this section is verifiable in the code and
docs on `main` at the SI-25 merge (main tip `195b12c`,
the merge of PR #30). Numbers are cited to the as-built
document that measured them; this RFC measures nothing
new. Where a statement below is a *derivation* from cited
facts rather than a measurement, it is labelled a
derivation, exactly as SI-24 §1 labelled its arithmetic.

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
- **Two authored signals, one landed delta (SI-19 +
  SI-23), unchanged.** The signal registry
  (`SIGNAL_REGISTRY`, `signal_spec` in
  `si/tiny_training.py`) serves exactly two named signals
  — `tiny-successor-lean-001` (SI-19) and
  `tiny-predecessor-lean-001` (SI-23) — and refuses every
  other id with the existing `ValueError` discipline.
  Selection is explicit and recorded; no autonomous
  selection exists. Both signals share one form
  (target-logits MSE), one LoRA target
  (`lm_head.weight`, rank 8, alpha 8), correction scale
  0.15, 2,000 full-batch steps, learning rate 1.0, norm
  budget 0.9, 32 train / 16 held-out examples; they differ
  only in correction direction, stream, and drawn
  sequences.
- **Two recorded target anchorings (SI-25).**
  `TARGET_ANCHORINGS` serves exactly
  `"weights-under-training"` (the default, SI-21's
  semantics) and `"base-anchored"` (SI-25), validated by
  `validate_target_anchoring(...)` with the existing
  refusal discipline (`si/tiny_training.py`). Anchoring
  is a recorded per-cycle configuration beside recorded
  signal selection: a constructor-level
  `target_anchoring` on `TinyTransformerMind`, settable
  between cycles by the explicit driver call
  `set_target_anchoring(...)`, carried on the intent
  (`Intent.target_anchoring` in `si/gate.py`), and
  recorded in every cycle's hyperparams with
  `target_source_head`. Under `"base-anchored"`,
  `train_candidate` splits only the target-source role:
  hidden states, the readout under training, candidate
  construction, and every loss evaluation still use the
  weights under training at the cycle's verified head;
  the signal's targets alone derive from
  `assemble_base_weights(...)` — the base weight set
  assembled from the cycle's verified manifest with its
  active chain truncated to empty, through the existing
  hash-verified reads, the verified manifest never
  mutated, no tensor copied or stored
  (`si/tiny_transformer.py`). A failed base assembly
  propagates and refuses training; there is no fallback
  to head-derived targets anywhere in the code. The
  module constant `TARGET_ANCHORING` survives as the
  *default's* name. No gate check, verdict mapping,
  tolerance, cap, or policy default changed in SI-25.
- **Sequential authorship of one signal, measured
  (SI-21), unchanged.** Cycle 1 (signal 1 at base)
  ADMITted as `d0001`, reproducing SI-19 exactly
  (candidate blob `sha256:2761e8ff…6ca9e`, identity drift
  0.17189218290150166 ≤ tol 0.25, capability regression
  0.1657181391492486 ≤ eps 0.25, echo 16/16, §6.6 VERIFY
  pass). Cycle 2 (signal 1 retrained at `d0001`,
  head-relative) was **REJECTED**: cumulative identity
  drift **0.34378434903919697** > tol 0.25.
- **A second signal, authored and measured (SI-23),
  unchanged.** Cycle 1 reproduced SI-19/SI-21 exactly.
  Cycle 2 (signal 2, head-relative, at `d0001`) trained a
  genuinely new candidate (blob `sha256:184cc7f0…6e6cf7`,
  norm 0.21317186599953494) that succeeded as a
  head-relative improvement (held-out loss at its head
  fell ~361×, 16/16 examples closer) and was
  **REJECTED**: cumulative identity drift
  **0.2963416576385498** > tol 0.25 (worst probe
  `tiny-id-3`), capability regression
  0.15023326850496233 ≤ eps 0.25 (pass), echo 1.0 (pass).
  Signal 2 at base ADMITted in its own run (drift
  0.16672966815531254, worst `tiny-id-7`) — an at-base
  fact, never evidence about head `d0001`.
- **Base-anchored targets, authored and measured
  (SI-25).** The decisive two-cycle run — one reused
  mind, cycle 1 signal 1 under the default anchoring,
  cycle 2 signal 2 under explicit recorded
  base-anchoring, exactly one configuration changed
  against SI-23's decisive run:
  - Cycle 1 reproduced SI-19/SI-21/SI-23 **exactly under
    either anchoring** — same candidate blob
    `sha256:2761e8ffc139ce3e107c11da035e8a73c20792945a494f4e493dccf418b6ca9e`,
    identity drift 0.17189218290150166, capability
    regression 0.1657181391492486 — so the anchoring
    machinery is proven behaviour-preserving for the
    existing semantics.
  - The coincidence configuration reproduced SI-23's
    at-base signal-2 run **exactly** (blob
    `sha256:131cc242728024b4f833bc78ea627825f1e8616972fc2670603673e1814645f4`,
    drift 0.16672966815531254, regression
    0.2418281203135848) — the base assembly and the split
    target-source path are proven correct where the two
    semantics coincide.
  - Cycle 2's candidate was genuinely new (blob
    `sha256:646d1e8ba3978d858d4173d561f8799bce37350f88ff2de65f40c223d9e4d675`,
    2,360 B, norm 0.2515524481203623, trained at `d0001`
    against base-derived targets) and as a
    destination-relative improvement it succeeded
    (held-out loss against the base-anchored targets fell
    ~264×, from the genuine pre-training measurement
    0.002757805049077469 to 1.043608588752459e-05, 16/16
    examples closer). The gate **REJECTED** it:
    cumulative identity drift
    **0.3329036422073841** > tol 0.25 (worst probe
    `tiny-id-0`), capability regression
    **0.10096919978968799** ≤ eps 0.25 (pass), echo 1.0
    (pass). Nothing landed; head stayed `d0001`;
    blob-store growth +0 B. Total two-cycle wall time
    2.43 s on the author's VM, CPU only.
  - Signal 2's base-anchored gradient was
    finite-difference checked in both configurations
    (worst absolute disagreement
    5.5249277028738815e-08 at base,
    4.8377706524291005e-08 at head `d0001`, tolerance
    1e-5, unchanged).
  - **The §5.5 residual finding.** Signal 1's
    base-anchored residual held-out loss at `d0001`
    measured **3.2118398935406443e-06** — 0.23% of the
    closed form, exactly cycle 1's own
    `held_out_loss_after`, as it must be: the landed
    head essentially *occupies* signal 1's fixed
    destination on held-out sequences. Training against
    it did not improve (trained held-out loss
    6.234076521114708e-06, worse than the pre-training
    record), so the existing non-improvement refusal
    fired and no residual candidate exists to judge. A
    base-anchored destination the head already occupies
    authors nothing — reported as a finding, exactly as
    RFC SI-24 §5.5 prescribed.
- **The attribution SI-25's diagnostics bought (the
  evidence this RFC designs against).** RFC SI-24 §5.6
  pre-registered a diagnostic set; SI-25 published it in
  full, recomputed for the base-anchored candidate
  against SI-23's control run. Its findings, stated
  against the control wherever SI-25 stated them:
  - **In aggregate the designed mechanism appears.**
    Over all 8 identity probes (256 coordinates), the
    two updates' movement inner product is
    **−0.21256230517288188** (negative — opposing) where
    the control's was **+0.07637087369869666**
    (aligned); sign agreement **0.37109375** (95/256)
    where the control's was 0.54296875 (139/256);
    movement-vector norms ‖m1‖ = 0.7119678353441855,
    ‖m2‖ = 0.7816722548913912. In weight space the
    folded updates likewise oppose: Frobenius inner
    product **−0.011457914936582613** where the
    control's was +0.004356018453758104; norms
    ‖Δ1‖_F = 0.23555380990005192, ‖Δ2‖_F =
    0.2515524481203623 — the base-anchored update is
    *larger* than the head-relative one (0.2516 vs
    0.2132), because it corrects the head's existing
    lean as well as adding its own.
  - **At the control's worst probe it appears
    decisively.** At `tiny-id-3`: inner product
    **−0.02224599986114801** (opposing; control:
    +0.021769049613635714, aligned), sign agreement 0.5
    (16/32); composed drift **0.1489148810505867**
    where the control composed 0.2963416576385498.
  - **At the treatment's worst probe it fails.** At
    `tiny-id-0`: inner product
    **+0.0338873784299374** (positive — aligned), sign
    agreement 0.5 (16/32); `d0001`'s own drift there is
    0.17189218290150166 (cycle 1's largest), the
    base-anchored incremental movement there is
    0.16173836588859558, and the composed drift is
    **0.3329036422073841** — to the printed digits the
    *sum* of the two contributions (0.17189218290150166
    + 0.16173836588859558 ≈ 0.3329), the aligned
    extreme, at that probe. `tiny-id-0` was already
    cycle 1's worst probe, and it is where the gate
    binds.
  - **Per probe, the treatment improves on the control
    at 5 of 8 probes** (`tiny-id-1`, `-3`, `-4`, `-5`,
    `-7` — dramatically at the control's worst probe)
    and worsens at 3 (`tiny-id-0` substantially,
    `tiny-id-6`, and `tiny-id-2` marginally: 0.14907 →
    0.14936). The maximum — the only figure the gate's
    identity arm prices — worsens: 0.3329036422073841
    against the control's 0.2963416576385498.
  - **The §5.6 derivation is not confirmed.** RFC
    SI-24 §5.6 priced *exact arrival* at the
    destination's at-base drift
    (0.16672966815531254); the measured composed drift
    is 0.3329036422073841. Arrival on the training
    sequences is per-probe uneven — near-exact
    opposition at some probes, none at the binding
    one — so the derivation's limit is not approached
    where it matters.
  - **The larger update exceeds the remaining headroom
    almost everywhere.** The headroom after cycle 1 is
    **0.07810781709849834** (tol 0.25 −
    0.17189218290150166; arithmetic on cited numbers).
    The treatment's incremental movement exceeds it at
    **7 of 8** probes (control: 5 of 8). No diagnostic
    was uncomputable in SI-25; nothing in its set is
    reported "not evaluated".
  - **The capability arm did not bind.** The
    treatment's head-incremental regression
    (0.10096919978968799) is *better* than the
    control's (0.15023326850496233), and echo accuracy
    is 1.0 in both runs. RFC SI-24 §5.6's capability
    warning — that the larger base-anchored update
    might bind on capability first — did not
    materialise in this configuration.
- **Unwind, still fixture-only (SI-21, unchanged by
  SI-23 and SI-25).** Because cycle 2 REJECTED in
  SI-21, SI-23, and SI-25, **no authored two-delta
  stack exists**; nothing was unwound in SI-25 and no
  unwind is claimed. SI-21's fixture-proven unwind
  (authored `d0001` + a benign hand-built fixture
  `d0002`, labelled a fixture) remains the only unwind
  evidence. The transformer-squash precondition (an
  authored two-delta stack) remains **unmet** (§12.1).
- **Suite state:** 328 passed, 2 skipped on `main` at
  the SI-25 merge (SI-25 as-built, author's run: the
  SI-23 baseline of 315 passed, 2 skipped, plus
  SI-25's 13 new tests in `tests/test_base_anchored.py`,
  zero regressions).

**What does not exist, exactly (SI-25, "What is still
NOT claimed", verified against the code on `main`):**

- **No authored stack deeper than one.** The deepest
  landed authored stack on record remains depth 1
  (`d0001`). All three attempted second cycles (SI-21
  same-signal head-relative, SI-23 second-signal
  head-relative, SI-25 second-signal base-anchored)
  exist only as rejected, never-written candidates
  (blob-store growth +0 B each).
- **Base-anchoring is not shown to be the better
  semantics.** The treatment's composed maximum drift
  (0.3329036422073841) is *worse* than the control's
  (0.2963416576385498), even though its aggregate
  alignment opposes, its capability regression
  improves, and 5 of 8 probes improve. One decisive
  configuration was measured under each anchoring; no
  claim is made about base-anchoring at other heads,
  other depths, or for signal 1 as a decisive matter.
- **No third cycle, no continual learning, no
  mixed-anchoring stack claim.** Exactly two authored
  cycles were attempted in each decisive run, on
  constructed signals, on a constructed model. No stack
  whose cycles mix anchorings at non-base heads was
  trained or is claimed.
- **No composition-aware training of any kind.** Every
  candidate on record was trained by the signal loss
  alone (head-relative or base-anchored). No loss term,
  constraint, projection, or selection rule has ever
  read the first update's movement, the composed
  head's movement, or any probe's movement while
  training. §7 designs the first such candidate; it
  does not exist today.
- **No movement accounting inside training or inside
  the gate beyond the existing checks.** The gate's
  identity arm remains the seed-absolute cumulative
  maximum over the identity probes
  (`identity_tol` 0.25, unchanged since the miniature
  was seeded); the per-probe tables in SI-23 and SI-25
  are post-hoc diagnostics, published after the gate
  has spoken, and are not gate checks (§8).
- **No Glimmer anything.** No Glimmer forward path,
  probes, training, or authorship; the Glimmer seed is
  untouched and its deltas still escalate with checks
  3/4 "not evaluated" (§12.2).
- **No widening.** Two signals, two anchorings, one
  target, one bundle kind; no additive or multi-target
  authorship; no policy default changed; no tolerance
  moved; both signals are byte-for-byte as SI-19/SI-23
  defined them; the default anchoring is unchanged.
- **No serving, no CLI wiring for the miniature.**
  `si serve` returns 501 for every transformer bundle
  (`si/serve.py`); `si cycle` / `si run` accept only
  `--mind demo|mock`.
- **No transformer squash.** SI-12's fail-closed refusal
  stands: `si/squash.py::squash_bundle` raises
  `SquashError` before any write if any record on the
  resolved active chain has kind `transformer-delta-v1`.

The frontier question is therefore precise in a new
way. Three second cycles have now been authored against
the landed head and genuinely judged, under two signals
and two anchorings, and the gate's cumulative identity
accounting has refused all three. The first two refusals
were *aggregate* alignment failures: the second update's
movement aligned with the first's in aggregate, in
weight space, and at the worst probe. The third refusal
is a different, narrower failure: the fixed-destination
mechanism SI-24 designed is measurably present — in
aggregate, in weight space, and at the previous worst
probe — and the composed maximum still worsens,
because at one probe, the probe where cycle 1's own
drift was already largest, the movements add almost
exactly. What has never been tried is a training
procedure that *knows* the first update's movement
exists and is shaped by it — not by aiming at a
different destination, but by accounting for the
movement composition itself while it trains (§§6–7).

---

## 2. Inherited invariants

Not negotiable inside this RFC (SI-9 §1.3, RFC SI-13 §2,
RFC SI-18 §2, RFC SI-20 §2, RFC SI-22 §2, RFC SI-24 §2,
restated for the candidates):

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
   blind-set guard apply to authored candidates exactly
   as to fixtures, at every cycle, for every signal,
   under every anchoring, under every training
   configuration.
4. **Honest labels are load-bearing.** "Not evaluated"
   stays distinguishable from a measured 0.0; a training
   loss stays distinguishable from a gate result; a
   number measured at one head stays labelled with that
   head (SI-20 §12.8); incremental and cumulative
   figures are never presented as each other
   (SI-20 §12.9); a number recorded under one anchoring
   is never presented as a number under the other
   (SI-24 §12.15); a number recorded under one training
   configuration is never presented as a number under
   another (§17 of this RFC).
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
   addressing (SI-19's canonical-header lesson,
   SI-24 §12.7).
8. **No tolerance is a design variable.** No candidate
   in this RFC proposes moving `identity_tol`,
   `capability_eps`, any cap, or any policy default to
   make an outcome pass. SI-21's, SI-23's, and SI-25's
   cycle-2 REJECTs are measurements to design *around*
   — with new training semantics or a new signal —
   never numbers to outvote, and never numbers to
   re-price by changing what the gate measures
   (§8.2).
9. **The probes are the judge, never the training
   set** (new in statement, in force in fact since
   SI-19: every signal's sequences are drawn disjoint
   from both probe suites and the disjointness is
   asserted by test). No candidate in this RFC may
   train on, penalise against, project against, or
   otherwise shape a candidate using any probe input,
   probe logit, or probe-derived movement. A method
   that reads the judge's inputs to shape the judged
   artifact has not passed the judge; it has memorised
   the courtroom. §9's rejection rests on this
   invariant, and §7's design is constrained by it at
   every step.

---

## 3. Candidate (a): a small real safetensors base — still deferred

**Verdict up front: DEFERRED** — unchanged from
SI-22 §3 and SI-24 §3, and this RFC does not reopen it.
SI-20 audited SI-18 §3.7's four preconditions (A-1
feasibility measurements for one named family; A-2 a
binding design for `reference-v1` bases; A-3 a
capability-suite design for a trained base; A-4 a
one-family executor design with SI-15-grade numeric
proof) and found all four unmet; SI-22 and SI-24
re-audited them unchanged after SI-21 and SI-23.
SI-25 changed none of them: it built a second
*anchoring* for the miniature, using the miniature's
executor, the miniature's binding, and the miniature's
constructed signals. Nothing in SI-25's recorded
configuration, base assembly, or diagnostics is a
measurement of any real checkpoint family, and citing
any of it toward A-1 would be exactly the unlabelled
transfer SI-18 §10.4 forbids.

The actual blocker bears restating once more, because
it is not a preference: naming a candidate family and
measuring *its* forward cost requires the checkpoint
itself, and **obtaining one means a download** —
outside this project's autonomous scope (invariant 6).
That is an operator decision to flag, not a step to
take, and this RFC neither schedules nor performs one.
The measurement-only slice SI-20 §3.5 described
(synthetic-shape scaling measurements; the A-2 binding
design on paper) remains a legitimate *future*
candidate; it is not recommended as SI-27 because §7's
candidate is buildable now with instruments that
already exist and answers the question SI-25's
diagnostics have just made precise, while §3.5's slice
retires overheads, not any of A-1 to A-4 as stated.

| Precondition (SI-18 §3.7) | Status today | Changed by SI-25? |
|---|---|---|
| A-1 feasibility measurements, one named family | Not met — none exist | No |
| A-2 `reference-v1` binding design + tamper analysis | Not met — binding unstatable for referenced bases (`si/probes.py`) | No |
| A-3 capability-suite design for a trained base | Not met — unsolved design problem | No |
| A-4 one-family executor design + numeric proof | Not met — executor refuses all but the miniature | No |

---

## 4. Candidate (b): a second training signal — DONE as SI-23, closed

**Verdict up front: DONE and CLOSED** — this was
SI-22's recommended slice, it merged as PR #28, and
SI-24 §4 already restated its outcome. SI-25 did not
touch it: the registry still serves exactly the two
named signals, byte-for-byte as SI-19/SI-23 defined
them, and SI-25's decisive run used the second of them
under a new anchoring. Nothing in SI-25's evidence
reopens candidate (b)'s question — whether a second
signal can be authored against the landed head under
settled semantics and genuinely judged — because that
question was answered by SI-23 (it can; the judgement
was REJECT) and SI-25 answered the successor question
about the same signal under new semantics (it can; the
judgement was REJECT again, §5). Whether a *third*
signal should be authored is a different candidate,
evaluated on its own merits in §10 — and rejected
there, on SI-25's evidence, not on this section's.

---

## 5. Candidate (c): base-anchored targets — DONE as SI-25

**Verdict up front: DONE** — this was SI-24's
recommended slice, it merged as PR #30, and its result
is the evidence base for everything below. Restating
the outcome without inflation:

- Anchoring is a recorded per-cycle configuration
  serving exactly the two named anchorings; selection
  is explicit, recorded per cycle, and asserted by
  test on both verdict paths (§1). The default is
  unchanged: `"weights-under-training"`.
- Target-source assembly is fail-closed and
  byte-honest: the base view is assembled from the
  cycle's verified manifest with its chain truncated
  to empty, the manifest is never mutated, a broken
  base assembly refuses training with no fallback, and
  blob-store growth is attributable only to a landed
  candidate (+2,360 B cycle 1, +0 B cycle 2).
- The coincidence configuration and first-cycle
  reproduction hold exactly under the new code path
  (§1), and the base-anchored loss's gradient was
  checked at the decisive head (worst
  4.8377706524291005e-08, tolerance 1e-5).
- Composition across anchorings is proven in
  SI-24 §9's exact sense: the second write was
  genuinely authored against the landed head under a
  *different target semantics* and genuinely judged.
  The judgement was REJECT — under base-anchoring as
  under head-relative anchoring, on the same arm
  (cumulative identity drift), with the capability and
  echo arms passing in both runs.
- The §5.6 diagnostics were published in full and did
  the work they were pre-registered to do: the REJECT
  is *attributed* — the fixed-destination mechanism
  appears in aggregate and fails at the binding probe
  — not merely recorded. §6 is this RFC's reading of
  that attribution.
- Two findings stand beside the verdict. First, the
  §5.5 residual record (§1): signal 1's base-anchored
  destination at `d0001` is already occupied, its
  training does not improve, and the existing refusal
  stands — the residual-cycle route SI-24 §5.7
  considered as an alternative decisive configuration
  is thereby closed by measurement, not by preference:
  there is no residual candidate to judge. Second,
  base-anchoring is **not** shown to be the better
  semantics (§1): on the one figure the gate prices,
  the treatment is worse than the control.
- What SI-25 did **not** produce: a landed second
  delta, an authored two-delta stack, or any evidence
  about a third cycle. Its own closing section binds
  this RFC: no authored stack deeper than 1 exists;
  the §5.6 derivation is not confirmed; diagnostics
  are not verdicts.

Candidate (c) is therefore closed, not deferred: its
question is answered. The open frontier is the failure
both (b) and (c) exposed and neither was designed to
address — the composition of two movements at the
probes — against the alternatives evaluated in
§§7–12.

---

## 6. The movement-composition question — the deep treatment

This section is the centre of the RFC. It states what
three decisive runs have measured about how two
authored movements compose, what the composition
arithmetic does and does not explain, why the
base-anchored mechanism succeeded in aggregate and
failed at the binding probe, and what question remains
that a next slice could actually answer. Every number
is cited to SI-21, SI-23, or SI-25; every statement
beyond them is labelled a derivation.

### 6.1 The three-run record

Three times, a second authored candidate has been
trained at head `d0001` and judged by the unchanged
gate. The identity arm's record, side by side (each
figure under its own run's labels — signal, anchoring,
accounting — per §17):

| Decisive cycle 2 | Signal / anchoring | Composed identity drift (cumulative, seed-absolute) | Worst probe | Capability regression (head-incremental) | Verdict |
|---|---|---|---|---|---|
| SI-21 | signal 1 / weights-under-training | 0.34378434903919697 | (as published in SI-21) | 0.16571810841560364 — pass | REJECT |
| SI-23 (control) | signal 2 / weights-under-training | 0.2963416576385498 | `tiny-id-3` | 0.15023326850496233 — pass | REJECT |
| SI-25 (treatment) | signal 2 / base-anchored | 0.3329036422073841 | `tiny-id-0` | 0.10096919978968799 — pass | REJECT |

Cycle 1 is common to all three runs and reproduces
exactly in each: drift 0.17189218290150166, headroom
**0.07810781709849834** remaining under tol 0.25
(derivation: 0.25 − 0.17189218290150166, as first
computed in RFC SI-22 §5.6.4). Echo accuracy is 1.0 in
every cycle of every run. The capability arm has never
bound. The identity arm has bound three times, and —
this is the point SI-25 adds — it has now bound for
two *different reasons at the level of composition*,
which §§6.2–6.4 separate.

### 6.2 Composition arithmetic: what the gate prices

The gate's identity arm prices exactly one figure: the
maximum, over the 8 identity probes and their 32 logit
coordinates, of the composed head's absolute drift from
the seed's recorded logits. For a candidate trained at
`d0001`, write `m1` for the first update's movement at
a probe (the `d0001` forward minus the seed expected
forward — a measurement) and `m2` for the second
update's incremental movement at the same probe (the
composed forward minus the `d0001` forward — a
derivation from two measured forwards, labelled as
such in SI-23 and SI-25). The composed drift at a probe
is the maximum absolute coordinate of `m1 + m2` there.
Three consequences of that definition are arithmetic,
not interpretation (derivations, stated once and used
throughout this RFC):

- **The composed maximum is bounded by the sum and
  floored by the difference.** Per probe, the composed
  drift lies between |drift₁ − movement₂| and
  drift₁ + movement₂, where drift₁ and movement₂ are
  that probe's measured maxima — with the caveat that
  the bounding maxima may be attained at different
  coordinates, so the interval is a bound, not an
  identity. Both SI-23's worst probe and SI-25's worst
  probe sit at the interval's *top*, to the printed
  digits: the maxima are attained at coordinates where
  the two movements align.
- **Aggregate alignment does not price the maximum.**
  An inner product summed over 256 coordinates, or a
  sign-agreement fraction counted over them, is a
  statement about the bulk of the movement vectors.
  The gate prices the single largest coordinate of the
  sum. A candidate can oppose in aggregate — negative
  total inner product, minority sign agreement — and
  still compose to a *larger* maximum than an aligned
  candidate, if its opposition is concentrated where
  the first movement was small and its alignment where
  the first movement was largest. SI-25 is the measured
  instance of exactly that sentence (§6.3).
- **Sign agreement counts coordinates; the gate
  prices magnitudes.** At `tiny-id-0` under
  base-anchoring, sign agreement is 0.5 (16/32) — a
  coin flip by count — while the inner product is
  +0.0338873784299374, positive. Both are published
  derivations from the same two forwards; they do not
  contradict each other. The inner product is
  magnitude-weighted: the coordinates where the
  movements agree in sign include the large ones, and
  the composed maximum is attained at one of them.
  Any future diagnostic that reports sign agreement
  without the magnitude-weighted figure beside it
  invites a misreading this RFC explicitly warns
  against (§17).

### 6.3 Why base-anchoring opposed in aggregate and failed at `tiny-id-0`

RFC SI-24 §5.2's design sentence: a base-anchored
cycle asks the composed head to equal what the seed
base's outputs would be after the signal's lean — a
fixed destination, independent of where the head
stands. SI-25 measured that mechanism appearing
(§1): aggregate inner product −0.21256230517288188
against the control's +0.07637087369869666; at the
control's worst probe, `tiny-id-3`, the composed drift
collapses from 0.2963416576385498 to
0.1489148810505867. The destination logic is not
speculation any more; on most probes, the
base-anchored update demonstrably travels *back
toward* the base's side of the first update's
movement while adding its own lean.

At `tiny-id-0` it does not. The published per-probe
table (SI-25 §5.6(i)) is the whole evidence, and this
RFC reads it as follows — interpretation labelled as
interpretation, over measurements:

- `tiny-id-0` is the probe where `d0001` itself moved
  most: drift 0.17189218290150166, the largest of the
  8, equal to the gate's cycle-1 figure. Whatever a
  second update does there, it starts from the least
  headroom of any probe — 0.07810781709849834, the
  same at every probe by the tolerance's definition,
  but consumed furthest there (derivation: the
  headroom is a property of the maximum, and
  `tiny-id-0` *is* the maximum's probe in cycle 1).
- The base-anchored incremental movement at
  `tiny-id-0` is 0.16173836588859558 — not the largest
  incremental movement in the treatment's table by
  much (`tiny-id-6` is 0.16010307148098946,
  `tiny-id-2` is 0.14893579483032227), and comparable
  to the control's movements. The treatment did not
  fail because its second update is enormous at the
  binding probe; it failed because at that probe the
  movement it does make points the *same way* as the
  first update's, at the coordinates that matter.
- The destination argument (§6.4) explains why this
  can happen without the mechanism "not working":
  training optimises the destination on signal 2's 32
  training sequences, at their last positions. The
  identity probes are different inputs, disjoint by
  construction. A rank-8 readout update trained to
  move the head toward the destination on the training
  inputs does whatever it does on a probe input
  according to that input's hidden state — and the
  readout is the final operation, so the hidden states
  a probe presents are the *base architecture's*
  hidden states, unaffected by either readout delta
  (a fact about the miniature's shape, relied on since
  SI-19: the readout cannot change the hidden states
  it reads). Per-probe unevenness of arrival is
  therefore not an anomaly to be explained away; it
  is the expected behaviour of a destination argument
  applied through a low-rank readout to inputs the
  training never saw. SI-24 §5.6 named this gap in
  advance (its limits (i)–(iii)); SI-25 measured it.

One further reading must be refused explicitly,
because the numbers invite it: that `tiny-id-0` is
"the problem probe" and a future slice should attend
to it. `tiny-id-0` is a probe *input the gate judges
on*. Attending to it — training against it,
penalising movement on it, projecting movement off it
— is training on the judge (invariant 9), and §9
rejects that candidate on exactly that ground. The
honest statement is distributional: among 8 judged
inputs the training never sees, the composition was
worst at the input where the first update had already
moved most. A next slice may address the *distribution*
— how a second update's movement composes with the
first's on inputs in general, measured on inputs the
training is allowed to see — and be judged, as always,
on the probes it never touched.

### 6.4 The destination derivation, after measurement

RFC SI-24 §5.6's derivation: if a base-anchored
signal-2 cycle trained its composed head to reproduce,
on the identity probes, exactly the outputs signal 2's
at-base training produced at base, the composed drift
would equal **0.16672966815531254** ≤ tol 0.25 — the
destination's drift replacing the first cycle's in the
accounting instead of adding to it. SI-25's verdict on
that derivation, in its own words: "The derivation
remains what it was labelled in the RFC: a statement
about a hypothetical exact arrival, now measured to be
far from the binding probe's reality." The measured
composed drift is 0.3329036422073841 — almost exactly
*double* the derivation's price (derivation: 2 ×
0.16672966815531254 = 0.33345933631062508, within
0.0006 of the measured figure; the closeness is a
numerical coincidence of the aligned extreme — at the
binding probe the composed drift is the *sum* of two
near-0.17 movements, each comparable to the
destination's at-base drift — and this RFC claims no
identity behind it).

The derivation is not thereby refuted as arithmetic;
it is refuted as a *model of training*. Exact arrival
was always a hypothetical, and the measurement now
prices the hypothetical's distance at the probe that
matters: arrival on the training sequences was ~264×
loss reduction with 16/16 held-out examples closer,
and arrival on `tiny-id-0` was nil — the incremental
movement there added instead of correcting. Any
future destination-style argument in this project
must be read with SI-25's measurement attached: a
fixed destination bounds the destination's drift, and
says nothing, by itself, about the journey's per-probe
composition. That sentence is SI-25's permanent
contribution to the project's design knowledge,
whatever §14's slice measures.

### 6.5 What is still not known about composition

Stated as open questions a slice could retire — this
RFC invents no numbers for them (§18 carries them
forward):

- **Composition on non-probe inputs.** Every published
  alignment diagnostic in SI-23 and SI-25 is computed
  on the 8 identity probes — the judge's inputs,
  published after the verdict. The same diagnostic
  computed on the signals' own training and held-out
  sequences (inputs training is allowed to see) has
  never been published for any candidate. Whether the
  SI-25 candidate's movements oppose on the inputs it
  was trained on — and, if they do, how large the
  train-to-probe composition gap is — is **an open
  question to measure**. It is also the load-bearing
  unknown for §7: a penalty computed on training
  inputs can only be an informed design if training-
  input composition is itself measured, before or
  inside the slice that uses it.
- **Whether composition is shapeable at all through
  the readout.** Both authored updates are rank-8
  LoRA factors on one readout matrix. A composition
  penalty's gradient with respect to those factors
  exists or does not as a matter of derivation
  (§7.3); whether optimising it changes probe-space
  composition in the intended direction, on inputs
  the penalty never saw, is **an open question to
  measure** — the same generalisation gap as §6.4,
  one level down.
- **Whether the binding probe is stable.** SI-23's
  binding probe was `tiny-id-3`; SI-25's is
  `tiny-id-0`; SI-21's cycle-1 maximum was also at
  `tiny-id-0`. Whether "the probe where cycle 1 moved
  most binds cycle 2" is a regularity or a two-sample
  coincidence is **an open question** that only
  further decisive runs can inform — and designing a
  slice around the assumption that it is a regularity
  would be designing around the judge (§6.3).

---

## 7. Candidate (d): movement-composition-aware training

**Verdict up front: RECOMMENDED as the next slice
(SI-27)** — conditionally, and with the condition
stated before the design: **no derivation in this RFC
shows that a composition-aware second cycle will ADMIT
at head `d0001`.** What the evidence supports is
narrower and, this RFC argues, sufficient: the route
addresses the measured cause of the SI-25 REJECT —
per-probe movement composition — rather than a proxy
for it; it is the only candidate that changes exactly
one variable against SI-25's decisive run (the training
loss gains one term; signal, anchoring, target,
hyperparameters, streams, and head are held fixed), so
its outcome, ADMIT or REJECT, is attributable to the
composition term and to nothing else; it can be built
with the existing gradient machinery's shape (§7.3)
and judged by the unchanged gate on probes the
training never reads (invariant 9); and both of its
possible gate outcomes are honest, publishable
results. As in SI-21, SI-23, and SI-25, **no acceptance
criterion in §15 requires an ADMIT.**

### 7.1 Why the prior deferrals do not bind this candidate

Composition-aware training was never previously a
candidate in this project's RFCs — SI-24 §5.1 disposed
of two unnamed routes (a third head-relative signal; a
reduced correction scale) and neither is this one — so
there is no deferral to retire. What must be argued
instead is that this candidate is not one of the
disposed routes in a new costume. It is not, for
reasons checkable against the designs:

- **It is not a third signal.** The signal is signal 2,
  byte-for-byte as SI-23 defined it: same correction,
  same scale 0.15, same streams, same partitions. What
  changes is the *loss the factors are optimised
  against*, not what the signal asks for (§7.2). The
  gate-fitting objection to a third signal — searching
  a family for a member whose movements happen to fit
  the headroom (§10) — does not apply: nothing is
  searched, and the one new constant is fixed in
  advance by a stated rule (§7.2), not swept.
- **It is not a reduced scale.** The correction scale
  is untouched, the norm budget is untouched, and the
  design does not aim at a smaller `m2`: SI-25's `m2`
  at the binding probe (0.16173836588859558) is
  comparable to the control's movements, and a smaller
  aligned movement still adds. The design aims at the
  movement's *direction relative to `m1`*, on the
  inputs training may see — the variable SI-25's
  diagnostics isolated (§6.3).
- **It is not a semantics change to the destination.**
  The anchoring stays base-anchored, exactly as in
  SI-25's decisive cycle: the signal loss term is
  SI-25's base-anchored loss, unchanged. The new term
  is *added beside* it, recorded as its own
  configuration axis (§7.4), so the control/treatment
  attribution against SI-25 survives.

### 7.2 What composition-aware training means, precisely

Verified current state, so the change is statable in
one paragraph: in SI-25's decisive cycle,
`train_candidate` optimises signal 2's LoRA factors
against one loss — the target-logits MSE between the
candidate head's last-position logits and the
base-anchored targets (targets derived from the frozen
base's logits plus signal 2's recorded correction), on
signal 2's 32 training sequences — with the targets
held constant throughout, hidden states cached from
the weights under training, and the norm guard as the
only constraint beside the loss. The first update,
`d0001`, enters that optimisation nowhere: not its
folded matrix, not its movements, not its existence.

**The design SI-27 must implement (this RFC's
recommendation):** add exactly one term to that loss,
and change nothing else about the optimisation.

- **Definitions.** At a cycle trained against head
  `d0001`, the **existing movement** at an input `x` is
  `m1(x)` = (the landed head's last-position logits at
  `x`) − (the base's last-position logits at `x`) —
  the first update's realised movement at `x`, a
  measurement from two forwards through the existing
  SI-15 forward. The **candidate movement** at `x` is
  `m2(x)` = (the candidate head's last-position logits
  at `x`) − (the landed head's last-position logits at
  `x`), which changes as the factors train. Both are
  defined at last positions, on a named input set, in
  logit space — the same space, position, and
  movement definitions SI-23/SI-25's diagnostics use,
  so the training term and the published diagnostics
  are commensurable by construction.
- **The input set is the signal's train partition —
  and nothing else.** The penalty is computed over
  signal 2's 32 training sequences, which are disjoint
  from both probe suites by SI-23's proven
  construction (§2, invariant 9). No probe input,
  probe logit, or probe-derived quantity enters the
  loss, the penalty, factor initialisation, early
  stopping, or any selection among candidates. The
  held-out partition does not enter the penalty
  either: it remains what it has been since SI-19, a
  training number recorded before and after, never an
  optimisation input — so the slice's held-out records
  stay honest as generalisation evidence for the
  signal loss, and §14 additionally requires the
  penalty's value to be *reported* (not optimised) on
  the held-out partition, as the first measurement of
  the train-to-probe composition gap's near end
  (§6.5).
- **The penalty term.** For each training input `x`,
  the per-coordinate mean product
  `p(x) = (1/32) Σ_c m1_c(x) · m2_c(x)` — the
  magnitude-weighted alignment of the two movements at
  `x`, per logit coordinate, in units of logit², the
  same units as the MSE loss beside it. The penalty is
  the mean of `p(x)` over the train partition, and the
  total loss is:

  `loss = signal_MSE + λ · mean_x p(x)`, with
  **λ = 1.0**.

  Three properties of this exact form are load-bearing
  and are stated here so the as-built can be checked
  against them:
  (i) *The penalty is linear in `m2`.* Minimising it
  rewards opposition (negative `p(x)`) without bound
  in the term itself — the signal MSE and the
  unchanged norm budget are what anchor the candidate
  near the destination. A squared or positive-part
  form would price opposition and orthogonality
  differently; this RFC chooses the linear form
  because its gradient is exact and trivial (§7.3)
  and because its sign convention matches the
  diagnostic the project already publishes (the inner
  product): a negative penalty contribution *is*
  aggregate opposition, in the diagnostic's own units,
  on the training inputs.
  (ii) *λ is fixed by parity of units, not by search.*
  λ = 1.0 weights one unit of per-coordinate mean
  movement product exactly as one unit of
  per-coordinate mean squared target error. That is a
  design statement, made here, before any penalised
  measurement exists — the same discipline as the
  correction scale's (SI-23: "the family's recorded
  design, not a knob turned for this slice"). SI-27
  runs exactly λ = 1.0, once, in the decisive
  configuration. A λ sweep — running several weights
  and reporting the best — is gate-fitting with extra
  steps and is an explicit non-goal (§16); if λ = 1.0
  fails, a later RFC may argue a different single
  value from SI-27's published measurements, in the
  open, as this RFC argues from SI-25's.
  (iii) *The penalty is signed, so it can make the
  total loss negative and can fight the signal loss.*
  Both are expected and must be reported, not hidden:
  the as-built publishes the two terms separately, at
  initialisation and at the end of training, on the
  train partition (optimised) and on the held-out
  partition (reported only). If the penalised training
  cannot improve the *signal* held-out loss — the
  destination-relative record SI-25 published — the
  slice has traded the destination away for
  opposition, and §15's criteria treat that as a
  finding to publish, not a defect to tune around.
- **What the term asks for, in one sentence:** among
  candidates that approach signal 2's base-anchored
  destination on the training sequences, prefer the
  one whose movement on those sequences *opposes* the
  first update's realised movement there — where
  SI-25's unpunished optimiser was indifferent to that
  composition on every input, and the gate priced the
  indifference at the probes.
- **What does not move.** The freshness contract,
  candidate construction (against the verified
  manifest, parent = the cycle's head), the gate
  (unchanged code, unchanged accounting — it judges
  the composed head against the seed exactly as
  today), §6.6 VERIFY, rollback, the norm guard and
  budget, the step count, the learning rate, and every
  policy default. Composition-awareness is a property
  of how a candidate is *trained*, never of how it is
  *judged* — the same sentence SI-24 §5.2 wrote about
  anchoring, applied one axis over.

### 7.3 Gradient obligations under the penalised loss

SI-19's exactness rests on targets being *constants*
during optimisation, and SI-25 preserved that shape by
changing only the weight set the targets derive from.
The penalty preserves it too, for a reason specific to
the miniature's shape — stated as a derivation, to be
proven by SI-27's gradient check, not assumed:

- `m1(x)` is constant during a cycle's optimisation:
  both of its forwards (landed head, base) are forwards
  of *fixed* weight sets, computed once, exactly as
  the targets are computed once.
- `m2(x)` is the candidate's readout movement at `x`:
  the folded candidate update applied to `x`'s final
  hidden state. Because the LoRA target is the readout
  — the final operation — the hidden states the
  penalty reads are the same cached hidden states the
  signal loss reads; they do not change as the factors
  train. The penalty is therefore *linear* in the
  folded update, and its gradient with respect to the
  LoRA factors has the same shape as the signal loss's
  gradient (a fixed vector — the `m1`-weighted
  combination of cached hidden states — carried through
  the readout fold). No new backward machinery, no
  intermediate activations, and no derivative through
  attention, RMSNorm, or SwiGLU is needed (§11.1's
  blocker for a new target does not apply here, and
  this paragraph is the reason).
- What is needed, and §15 criterion 3 requires, is the
  proof re-run where the loss actually changed: the
  finite-difference gradient check for signal 2's
  **base-anchored, composition-penalised** loss **at
  head `d0001`** (base-derived targets, head hidden
  states, head readout, landed `m1` on the train
  partition), at SI-19's pinned tolerance 1e-5, over
  EVERY factor entry, with the measured agreement
  published; and at base, where `m1` is identically
  zero (the landed head *is* the base — a derivation
  from the definitions in §7.2), the penalised loss
  must coincide with SI-25's unpenalised base-anchored
  loss, and the check must be re-run under the new code
  path so the coincidence is measured, not asserted.
  That at-base coincidence is also the slice's
  correctness anchor for the penalty machinery
  (§7.5): a first cycle at base has no existing
  movement to oppose, so composition-aware training at
  base *is* SI-25 training at base, byte-for-byte, or
  the penalty code is wrong.

### 7.4 The representation: composition as a recorded configuration

How the choice is represented decides whether records
under the two training configurations can be told
apart forever after. This RFC applies SI-24 §5.3's
settled reasoning a second time, and reaches the same
shape of answer, for the same reasons:

1. **Flip the training loss globally.** The penalty
   becomes part of `train_candidate` for every cycle.
   *Rejected.* It rewrites the meaning of every future
   cycle silently, makes the unpenalised control
   irreproducible without a code revert, and renders
   SI-25's control run non-comparable going forward.
2. **A new signal id for the penalised variant**
   (e.g. a `…-composition-001` twin). *Rejected.* The
   penalty is not signal identity: the signal — its
   correction, stream, partitions, destination — is
   unchanged (§7.1). Twinning the id would mislabel a
   training axis as a signal axis, exactly as SI-24
   §5.3 rejected twinning for anchoring.
3. **An explicit, recorded, per-cycle configuration —
   RECOMMENDED.** Composition-awareness joins signal
   selection and anchoring as a driver-level
   configuration of the mind, mirroring both existing
   axes: a constructor-level named choice on
   `TinyTransformerMind` (defaulting to the
   unpenalised configuration, whose recorded name this
   RFC fixes as `"none"` for the configuration field),
   settable between cycles by an explicit driver call
   in the shape of `set_training_signal(...)` /
   `set_target_anchoring(...)`, validated against
   exactly the named values this slice introduces
   (`"none"` and `"movement-composition"`) with the
   existing refusal discipline (any other value
   refuses), carried on the intent, and recorded in
   every cycle's hyperparams — the configuration, the
   λ it ran with, and the penalty's measured values
   beside the signal loss's — asserted by test, per
   cycle, on both verdict paths. There is no
   autonomous, learned, depth-driven, or
   verdict-driven configuration choice: a configuration
   a gate verdict could influence would be the
   gate-fitting trap given a mechanism (SI-24 §5.3,
   applied verbatim). λ is part of the configuration's
   *definition* for this slice (λ = 1.0, §7.2), not a
   per-cycle dial: the recorded configuration names
   the penalty form and its weight together, and a
   driver cannot set λ independently in SI-27.

One consequence must be stated plainly, mirroring
SI-24 §5.3's consequence for anchoring: with the
configuration defaulting to `"none"`, the unpenalised
path must remain byte-for-byte the SI-25 path.
"Composition-aware" describes a cycle's recorded
configuration, never the runtime as a whole.

### 7.5 The decisive configuration: signal 2, base-anchored, composition-penalised, at `d0001`

**The design SI-27 must implement:** the decisive run
is SI-25's decisive run with exactly one configuration
changed — one reused `TinyTransformerMind`, cycle 1
signal 1 under the default anchoring and the default
composition configuration, landing `d0001`; the driver
then explicitly reconfigures the same mind to signal 2,
to base-anchored targets, *and* to the
movement-composition configuration (three recorded
selections, each already an established axis or
modelled on one); cycle 2 trains at `d0001` under the
unchanged freshness contract and is judged by the
unchanged gate. Two sanity configurations accompany
it, labelled as SI-25 labelled its own:

- **Signal 2, base-anchored, composition-penalised at
  base (the coincidence configuration).** At head =
  `base`, `m1` is identically zero (§7.3), the penalty
  term is identically zero, and the run must reproduce
  SI-25's at-base run **exactly** — candidate blob
  `sha256:131cc242728024b4f833bc78ea627825f1e8616972fc2670603673e1814645f4`,
  drift 0.16672966815531254, regression
  0.2418281203135848. Any mismatch invalidates the
  decisive run's attribution until explained as a
  finding.
- **The unpenalised control, re-run inside the slice.**
  SI-25's decisive cycle 2 (configuration `"none"`)
  must reproduce its published candidate exactly
  (blob `sha256:646d1e8b…d675`) under the refactored
  training path, so the treatment's difference is
  attributable to the penalty term and not to the
  refactor that introduced it — the first-cycle
  reproduction discipline of SI-23/SI-25, applied to
  the control itself.
- **The composition record on allowed inputs
  (diagnostic, published whatever the verdict).**
  Before the penalised run is interpreted, the slice
  measures and publishes the §6.5 unknown for the
  *control* candidate: the alignment diagnostic
  (inner products, per-input and aggregate) computed
  on signal 2's train and held-out partitions, under
  the same movement definitions as the probe
  diagnostics. This record is what the penalty's
  design premise can be checked against: if the
  control's movements already oppose on the training
  inputs while aligning at the binding probe, the
  penalty is pushing on an already-pushed direction
  and the slice must say so; if they align on the
  training inputs, the penalty has something real to
  correct there. Either record is a finding; neither
  changes the decisive run, which is run as designed
  regardless (§15 criterion 5).

### 7.6 What would falsify the design's premise

A recommendation that cannot be wrong is not a design;
it is a hope. This slice's premise is falsifiable in
three named ways, and §15's criteria require the
measurements that would falsify it to be published
whatever they show:

- **F1 — the penalty does not transfer.** The
  penalised candidate's training-input composition
  improves (inner products fall, per §7.5's record)
  while its probe composition at the worst probe is
  unchanged or worse than the control's — the
  train-to-probe gap of §6.5 swallows the term, as it
  swallowed the destination argument at `tiny-id-0`
  (§6.4). Falsified premise: that composition measured
  on allowed inputs shapes composition on judged
  inputs, through this readout, at this scale.
- **F2 — the penalty transfers but the maximum does
  not move.** Probe composition improves in aggregate
  (inner product falls against −0.21256230517288188)
  yet the composed maximum stays above tol 0.25 —
  opposition purchased where the first movement was
  small, alignment surviving where it was large, a
  second measured instance of §6.2's arithmetic.
  Falsified premise: that aggregate composition is
  the binding variable. (SI-25 already half-falsified
  this premise once; F2 would settle it.)
- **F3 — the destination is traded away.** The
  penalised candidate composes better but its signal
  held-out record (§7.2(iii)) shows the destination
  no longer approached — opposition achieved by
  abandoning the lean the cycle exists to author.
  Such a candidate, even if the gate ADMITted it,
  would not be evidence that composition-aware
  *authorship* works; it would be evidence that the
  gate can be approached by a candidate that does
  less of its task, and the as-built must say so in
  those words. §15 criterion 8 makes the record
  mandatory for exactly this reason.

---

## 8. Candidate (e): per-probe movement accounting — as diagnostics and as a gate change

This candidate names two different proposals that
share a phrase, and SI-25's evidence treats them
oppositely. They are separated here so neither can
borrow the other's verdict.

### 8.1 As diagnostics: ADOPTED, inside SI-27 — not as a slice of its own

Per-probe movement accounting already exists in the
project — as the post-hoc diagnostic tables of SI-23
and SI-25, which named worst probes, separated
measurements from derivations, and made both RFCs'
designs possible. SI-25 leaves two accounting gaps
that §7.5's design already closes inside SI-27: the
same accounting computed on the signals' own
partitions (§6.5), and the penalised run's per-probe
table published against *both* controls (SI-23
head-relative, SI-25 base-anchored) rather than one.
Beyond those, a standalone diagnostics slice — more
tables, finer decompositions, no training change —
is **not recommended**: it would spend a cycle
measuring a system whose next measurement SI-27 will
change anyway, and the diagnostics SI-27 needs are
enumerable now (§15 criterion 8). Diagnostics in this
project have always ridden with the slice whose
outcome they attribute (SI-22 §5.6.4, SI-24 §5.6);
this RFC keeps that discipline.

### 8.2 As a gate change: REJECTED

The stronger reading of this candidate — that the
gate itself should account identity *per probe*
(budgets per probe, an incremental identity arm, a
mean or percentile in place of the maximum, headroom
carried per probe from cycle 1) — is **rejected as
the next slice and as any slice justified by this
RFC**, on grounds SI-25 strengthens rather than
weakens:

- **Every variant reprices the judge after three
  measured failures.** The identity arm's definition
  — seed-absolute, cumulative, maximum over probes —
  is the contract all three decisive runs were judged
  under and the contract the project's honesty rules
  exist to keep fixed (invariant 8). Changing what
  the arm measures *because* the maximum keeps binding
  is the tolerance-loosening move with the tolerance
  left numerically intact: 0.25 would remain printed
  in the policy while the thing it bounds quietly
  became easier. SI-21's, SI-23's, and SI-25's
  as-builts each closed with the same sentence in
  substance — if the cycle is REJECTed, the answer is
  a later RFC's better argument, never a wider gate —
  and this RFC is the later RFC; its better argument
  is §7, aimed at the candidate, not at the judge.
- **The maximum is not an accounting accident.** A
  per-probe or averaged arm would declare the composed
  head continuous with the seed on the strength of
  probes it passes while the worst probe's drift —
  the measured 0.3329036422073841, a real output
  change on a real judged input — went unpriced. The
  miniature's identity claim is only ever as strong
  as its worst judged input; an accounting that
  cannot see its worst input cannot carry the claim
  (§17, item 9's rule applied to the gate itself).
- **SI-25 gives the change no evidence to stand on.**
  Nothing in SI-25 suggests the gate mismeasured:
  cycle 2's journaled checks equal an independent
  evaluation (criterion 7 of SI-25's acceptance), the
  diagnostics *confirm* the verdict's figure at the
  binding probe, and the capability arm — the one
  head-incremental arm the gate has — passed in every
  run. The failure is in the candidates, three times
  measured. Redesigning the instrument that correctly
  reported three failures is not a frontier; it is an
  evasion.

---

## 9. Candidate (f): orthogonalization at the binding probe

**Verdict up front: REJECTED** — the only candidate in
this RFC rejected on an invariant rather than on
sequencing, and SI-25's numbers are not what reject
it; they merely make it tempting, which is why it
gets its own section.

The proposal, stated fairly: SI-25 identifies
`tiny-id-0` as the binding probe, with `m1` and `m2`
published there to full precision. A post-training
projection — or a training penalty — that removes the
component of the candidate's movement parallel to
`m1` *at that probe* (or at the 8 identity probes)
would directly attack the composed maximum the gate
prices. Variants differ in machinery; all share one
property: **they read probe inputs or probe movements
to shape the candidate.** That is training on the
judge (invariant 9), and it voids the very measurement
it aims to improve: a candidate orthogonalised against
the probes' movements and then ADMITted by those
probes has demonstrated that a projection can satisfy
8 known inputs, not that two authored deltas compose.
The ADMIT would be uninterpretable — worse than a
REJECT, because a REJECT at least measures something.

Two narrower variants are disposed of explicitly, so
no later reader revives them as "not quite (f)":

- **Weight-space orthogonalization** (project the
  folded Δ2 orthogonal to Δ1 in Frobenius geometry,
  no probe contact): invariant-clean, and **refuted
  as a next step by SI-25's own table** — the folded
  updates already oppose in weight space
  (−0.011457914936582613) while the binding probe
  aligns (+0.0338873784299374). Weight-space geometry
  is not the geometry the gate prices (§6.2), and a
  projection that strengthens an already-negative
  weight-space inner product has no argued path to
  the probe-space maximum. Rejected on evidence.
- **Orthogonalization on the train partition**
  (penalise or project the component of `m2` parallel
  to `m1` on allowed inputs): invariant-clean and
  *not* rejected — it is §7's design in a harder form.
  This RFC recommends the penalty form over the
  projection form for one stated reason: a projection
  applied after training produces a candidate whose
  factors were never optimised as a whole (its signal
  loss, norm, and destination arrival are whatever the
  projection leaves behind, unmeasured during
  training), while §7's penalty keeps a single
  optimisation whose every published record — signal
  loss, penalty value, gradient check — describes the
  candidate actually judged. If SI-27's F1/F2 outcomes
  (§7.6) falsify the penalty's premise, the projection
  form is a legitimate subject for a later RFC to
  argue from SI-27's measurements; it is not
  recommended now, and it inherits invariant 9 in full
  either way.

---

## 10. Candidate (g): a third or different signal

**Verdict up front: REJECTED as the next slice** —
SI-24 §5.1's verdict on a third head-relative signal,
extended by SI-25's evidence to a third signal under
either anchoring, and to a differently *formed* signal
on the same target.

- **The family's composition behaviour is the
  measured obstacle, not any member of it.** SI-24
  §5.1 argued this from SI-23 alone: both corrections
  share the away-from-the-last-token component, and
  both trained updates are rank-8 readout matrices
  applied to every hidden state the probes present.
  SI-25 adds the decisive datum: even when the second
  update is trained to a *fixed destination* — the
  strongest composition-favourable semantics the
  project has designed — its movement still adds at
  the binding probe. A third lean (successor,
  predecessor, or any other fixed neighbour offset)
  has no argued reason to compose differently at the
  probes, and searching the family for one whose
  movements happen to fit the 0.07810781709849834
  headroom remains the gate-fitting trap of
  SI-24 §5.1/§13 in the same costume.
- **A different *form* (a new loss shape, a new
  correction structure) confounds two questions.**
  SI-27's design (§7) changes the loss by adding a
  composition term to a *known* signal whose control
  runs exist under both anchorings. A new form would
  arrive with no control: its outcome, either way,
  would be attributable to the form, the composition,
  or their interaction, with no published run to
  separate them. Sequencing is the honesty safeguard
  (SI-24 §5.7 reason 4, applied again): measure the
  composition variable on the signal the project
  understands, or understand a new signal first —
  not both at once.
- **The residual route within this candidate is
  already closed by measurement.** The one "different
  signal" configuration SI-24 §5.7 left open as a
  diagnostic — signal 1 base-anchored at `d0001`, the
  residual cycle — was measured in SI-25: the
  destination is occupied (residual
  3.2118398935406443e-06), training does not improve,
  the refusal stands (§§1, 5). There is no residual
  candidate, and re-arguing one would require new
  evidence this RFC does not have.

---

## 11. Candidates (h) and (i): new target; additive / multi-target

SI-24 §6 evaluated these as its (d) and (e); they are
renumbered (h) and (i) here because §§7–10's
candidates intervene. SI-25 changed neither
candidate's blockers: it trained the same signal, on
the same target, through the same readout gradient,
under a new target-source semantics. Its recorded
anchoring and SI-27's proposed recorded composition
configuration settle *which signal*, *which
anchoring*, and *which loss*; none of them is a
gradient derivation for a new tensor, and none is
evidence about authored multi-target behaviour.

### 11.1 Candidate (h): a new LoRA target (non-readout tensor)

**Verdict up front: DEFERRED.**

- **The gradient is still the work, unchanged.**
  SI-19's exactness rests on one fact: `lm_head.weight`
  is the final operation, so no gradient with respect
  to any other tensor is ever needed and hidden states
  can be cached. For any earlier tensor — an attention
  projection, an MLP projection, the embeddings — the
  gradient runs backward through RMSNorm, causal
  softmax attention, and SwiGLU: the derivation RFC
  SI-18 §4.4 called error-prone in exactly those
  places. It ships, if ever, only with a
  finite-difference gradient check per target, at a
  tolerance pinned from measured agreement. None of
  that exists for any second target, and SI-25 built
  none of it.
- **The machinery a backward pass needs does not
  exist yet.** The forward replica
  (`si/tiny_training.py::forward_with_final_hidden`)
  exposes only the final hidden state; the
  intermediate activations a real backward pass
  through the layers needs are not returned anywhere
  in the runtime.
- **SI-25's evidence does not ask this candidate's
  question — it asks §7's.** A new target's update
  composes with `d0001` through *changed hidden
  states*, a composition the current diagnostics do
  not even describe (every published movement is a
  readout movement over readout-invariant hidden
  states, §6.3). Spending SI-27 on a derivation would
  leave the measured composition question untouched
  and, worse, make it unanswerable in the new
  target's terms until a new diagnostic set exists.
  §7, by contrast, is answerable with the existing
  gradient's shape (§7.3), the existing diagnostics,
  and two published controls.
- **What transfers is only the judging.** Per-class
  sensitivity is 28 of 30 tensors measurable (SI-16);
  the blind set is exactly both layers' `k_proj.bias`,
  which are 1-D and in any case not LoRA targets. If
  a later slice lands a second delta by any route, a
  new target becomes the natural next widening for a
  later RFC to evaluate — against a landed stack,
  with the composition question settled or priced.

### 11.2 Candidate (i): additive and multi-target authored candidates

**Verdict up front: REJECTED as the next slice** —
the SI-22 §6.2 / SI-24 §6.2 verdict, carried forward
unchanged, because SI-25 produced no authored
multi-target evidence that could argue against it.

- **The format is ready; the policy is not —
  deliberately.** `construct_transformer_delta`
  (`si/transformer_delta.py:330`) already builds
  multi-target records, `si/apply.py` folds per-target
  updates in chain order, and the gate already handles
  mixed blind + measurable candidates honestly. What
  forbids authored multi-target candidates is policy:
  `max_targets_per_delta` is 1 and `allow_additive`
  is false in the default policy (`si/bundle.py:46`,
  `si/bundle.py:49`), enforced by the gate. Widening
  authorship this way still means *changing policy
  defaults to admit what the gate currently refuses*
  — the one move §2 invariant 8 and §8.2 of this RFC
  forbid as a route around a gate outcome.
- **Joint training is a new optimisation problem, not
  two copies of SI-19.** Two targets trained against
  one loss interact — each target's gradient depends
  on the other's current factors through the forward
  — and the readout-only caching trick does not
  survive a second, earlier target. A composition
  term (§7) defined over readout movements does not
  transfer to it either: with an earlier target in
  play, hidden states move, `m1` and `m2` are no
  longer movements over a shared hidden state, and
  §7.3's derivation fails. Citing §7 toward (i) would
  be the unlabelled transfer this RFC's labelling
  rules exist to prevent.
- **Additive authorship's natural homes are still the
  worst targets.** Additive targets must be 1-D
  (`si/transformer_delta.py::validate_transformer_delta`);
  the miniature's 1-D tensors are norms and biases —
  including the probe-blind `k_proj.bias` pair, an
  authored delta on which could never be judged.

### 11.3 Verdicts on (h) and (i)

| Sub-direction | Verdict | One-line reason |
|---|---|---|
| (h) New LoRA target | DEFERRED | Needs a full backward derivation plus a per-target gradient check; SI-25's evidence asks the composition question on the existing target, whose diagnostics and controls already exist (§11.1). |
| (i) Additive / multi-target | REJECTED as next slice | Requires loosening `max_targets_per_delta` / `allow_additive` policy defaults with zero authored-behaviour evidence (§11.2). |

---

## 12. Candidate (j): transformer squash; candidate (k): Glimmer probing

### 12.1 Candidate (j): transformer squash for miniature deltas

**Verdict up front: DEFERRED** — its precondition
remains unmet, and SI-25 is now the third slice to
confirm that by measurement rather than by absence.

SI-20 §6.4 made squash schedulable only after a slice
produced an authored stack and the measurements a
squash design must respect. SI-21, SI-23, and SI-25
all produced the measurements (per-delta drift
contributions, the alignment diagnostics, two
control/treatment pairs) but **not the stack**: all
three second cycles REJECTED, so the deepest landed
authored stack on record is still depth 1 (`d0001`),
and **no authored two-delta stack exists**. SI-21's
two-step unwind was exercised on authored `d0001`
plus a benign hand-built **fixture** `d0002` —
journalled as a fixture, never described as authored
(§1). Squashing that fixture stack would prove format
handling, not authored composition: exactly the
designing-against-fixtures outcome SI-12 refused.

What SI-19 and SI-20 established still holds: for
authored same-target deltas, composition is exact
algebra — every authored delta is a single-target
LoRA on `lm_head.weight` at rank ≤ 8, two such updates
compose to a single update of rank ≤ 16 on one named
tensor, foldable by the existing
`si/apply.py::fold_target` arithmetic, with the
recompression to rank ≤ 8 lossy and therefore gated
by a tolerance verify of the SI-3 kind plus the
miniature's probes (SI-20 §6.2). What still blocks a
squash slice, beyond the missing stack:

1. **The squash gate context is demo-shaped.**
   `si/squash.py::_make_squash_mind` constructs only
   `DemoMind`/`MockMind`, and `SquashContext` serves
   demo probes and a demo forward. A miniature squash
   candidate must be gated by the *probe* context —
   checks 3/4 measured, the SI-17 verdict mapping, the
   blind-set guard — or it would be gated by
   instruments that cannot see it. That context is
   design work that does not exist.
2. **Provenance and identity of the squashed delta.**
   A squashed delta is not trained, so it carries no
   `hyperparams` of its own — and after SI-25 its
   constituents may carry *different anchorings*, and
   after SI-27 (if built) *different composition
   configurations*, so §17 forbids dressing it in its
   constituents' training numbers, either anchoring's
   labels, or either configuration's labels. Its
   journal story — compression of named deltas,
   tolerance measured, probes re-run — must be
   statable without borrowing authorship's labels
   (SI-20 §6.3).
3. **The refusal stands meanwhile.** SI-12's
   fail-closed refusal is unchanged in code:
   `si/squash.py::squash_bundle` raises `SquashError`
   before any write if any record on the resolved
   active chain has kind `transformer-delta-v1`. §16
   keeps it in force throughout SI-27.

If SI-27's second cycle ADMITs, the precondition is
met for the first time — an authored two-delta stack
will exist — and squash becomes a legitimate candidate
for a *later* RFC to evaluate against that stack's
measured norms and drift contributions. Meeting the
precondition does not schedule the slice, and SI-27
builds no part of it (§16).

### 12.2 Candidate (k): probing Glimmer — REJECTED, not reopened

Glimmer probing was rejected by RFC SI-13 §7/§9,
restated by SI-18 §5, SI-20 §7, SI-22 §7.2, and
SI-24 §7.2, and is **not reopened here**. SI-25
changes nothing about it, and neither does this RFC's
recommendation: authorship, like probing before it,
is miniature-only by construction — the training
assembles weights through the miniature executor
(`validate_architecture`, `si/tiny_transformer.py:146`,
refuses Glimmer's descriptor), the signals are defined
over a vocabulary of 32, the gate's probe binding
cannot be stated for the Glimmer bundle (§3), the
base-assembly design of SI-24 §5.4 is a 30-tensor
miniature mechanism, and §7's penalty is defined over
that mechanism's cached hidden states and a landed
miniature delta — every one of its terms is
unstable at Glimmer scale, and none is claimed there.

The restated facts, so no future reader has to
excavate them: there is no Glimmer forward path, and
a partial forward over a partial stack would be a
claim about that partial computation, not about the
model — fabrication, as SI-18 §5 recorded. The
Glimmer seed's deltas still escalate with checks 3/4
"not evaluated"; its stored bytes are still 0 against
59,553,253,376 logical bytes (SI-9 §0); and one full
verification of that by-reference base still stands
measured at ~29 minutes (1,756.9 s) before any forward
compute (SI-9 §0/§7.1). Any future Glimmer proposal
needs its own RFC and its own feasibility
measurements; it gets no momentum from this document.

---

## 13. Decision summary

| Candidate | Verdict | One-line reason |
|---|---|---|
| (a) Small real safetensors base | **DEFERRED** | SI-18 §3.7 audit: A-1 to A-4 all still unmet, unchanged by SI-25; naming a real base needs a download this scope forbids (§3). |
| (b) Second training signal (miniature) | **DONE — as SI-23, closed** | Built and measured (PR #28): signal 2 authored at base (ADMIT) and at `d0001` (REJECT, attributed to movement alignment) (§4). |
| (c) Base-anchored targets | **DONE — as SI-25** | Built and measured (PR #30): signal 2 base-anchored at `d0001` REJECTED on composed drift 0.3329036422073841, mechanism present in aggregate, failing at `tiny-id-0` (§5). |
| (d) Movement-composition-aware training | **RECOMMENDED — SI-27** | The only candidate aimed at SI-25's measured cause — per-probe composition — changing exactly one variable against SI-25's control, on inputs training may see, with λ fixed in advance and no ADMIT required (§7). |
| (e) Per-probe movement accounting | **SPLIT: diagnostics ADOPTED inside SI-27; gate change REJECTED** | The accounting already attributes verdicts and SI-27 extends it to allowed inputs; repricing the gate's identity arm after three measured REJECTs is tolerance-loosening by redefinition (§8). |
| (f) Orthogonalization at the binding probe | **REJECTED** | Reads the judge's inputs to shape the judged candidate (invariant 9); its weight-space variant is separately refuted by SI-25's table (§9). |
| (g) Third or different signal | **REJECTED as next slice** | The family's composition behaviour is the measured obstacle; a new member or form has no argued reason to compose differently and would confound §7's variable (§10). |
| (h) New LoRA target | **DEFERRED** | Needs a full backward derivation plus a per-target finite-difference gradient check; its composition through changed hidden states is not even described by the current diagnostics (§11.1). |
| (i) Additive / multi-target | **REJECTED as next slice** | Requires loosening `max_targets_per_delta` / `allow_additive` policy defaults with zero authored-behaviour evidence (§11.2). |
| (j) Transformer squash (miniature) | **DEFERRED** | Precondition still unmet — no authored two-delta stack exists after three measured attempts — and the squash gate context is still demo-shaped (§12.1). |
| (k) Glimmer probing | **REJECTED** | Rejected at SI-13 §7/§9, restated at SI-18 §5, SI-20 §7, SI-22 §7.2, and SI-24 §7.2; SI-25 changes nothing (§12.2). Not reopened. |

The three routes to a landed second write that the
project has named across SI-20 to SI-25 have now all
been run: a new signal under settled semantics (SI-23,
REJECT, attributed to aggregate alignment), the same
signal restacked (SI-21, REJECT), and new target
semantics for the second signal (SI-25, REJECT,
attributed to per-probe composition). §7's route is
the fourth, and the first aimed inside the training
loss at the composition itself rather than at what
the second update asks for or where its targets come
from. No fifth route is proposed by this RFC, and
§§8–10 record the three unnamed or tempting routes
(a repriced gate; the binding probe itself; a third
signal) and why they are not candidates.

---

## 14. The recommended slice: SI-27 (outline)

**SI-27 — Movement-composition-aware training for the
miniature: signal 2, base-anchored, trained at the
landed head with a composition penalty on its own
train partition.**

In scope:

1. **Composition as a recorded configuration**
   (§7.4): `TinyTransformerMind` is configured with
   the composition configuration it will train under
   (constructor-level named choice, defaulting to
   `"none"`), settable between cycles by an explicit
   driver call in the shape of
   `set_training_signal(...)` /
   `set_target_anchoring(...)`; the value is validated
   against exactly the two named configurations
   (`"none"`, `"movement-composition"`) and any other
   value refuses with the existing discipline; the
   intent carries it; `train_candidate` dispatches on
   it. λ = 1.0 is part of the
   `"movement-composition"` configuration's definition
   (§7.2), recorded in the hyperparams, and is not a
   separately settable value in this slice. No
   autonomous, learned, depth-driven, or
   verdict-driven configuration selection exists in
   the slice, and no λ sweep of any kind is run.
2. **The penalised loss** (§7.2): under
   `"movement-composition"`, `train_candidate` adds
   exactly one term to SI-25's base-anchored signal
   loss — λ · mean over the signal's 32 training
   sequences of the per-coordinate mean product
   `p(x)` between the landed head's existing movement
   `m1(x)` (computed once, from the cycle's verified
   manifest's landed head and its base assembly, both
   through the existing forwards) and the candidate
   movement `m2(x)` — with hidden states, target
   source, candidate construction, step count,
   learning rate, and the norm guard exactly as in
   SI-25. Under `"none"` the training path is
   byte-for-byte SI-25's. No probe input, probe logit,
   held-out sequence, or probe-derived quantity enters
   the loss under either configuration (asserted by
   test, §15 criterion 4).
3. **The coincidence configuration** (§7.5):
   configured with signal 2, base-anchoring, and the
   movement-composition configuration, a first cycle
   at base records the penalty's value at
   initialisation (identically zero, measured — not
   asserted from the derivation alone) and trains and
   is judged there. It must reproduce SI-25's at-base
   run exactly — candidate blob
   `sha256:131cc242…45f4`, same gate numbers — proving
   the penalty machinery correct where it must vanish,
   before the decisive cycle is trusted. Its outcomes
   are labelled at-base facts; they are not the
   slice's decisive claim.
4. **The control, re-run inside the slice** (§7.5):
   configured with signal 2, base-anchoring, and
   composition `"none"`, a cycle at `d0001` (in a run
   whose cycle 1 landed `d0001` as usual) reproduces
   SI-25's decisive cycle-2 candidate exactly — blob
   `sha256:646d1e8b…d675`, same gate numbers — proving
   the refactored training path behaviour-preserving
   for the unpenalised semantics.
5. **The composition record on allowed inputs**
   (§7.5): for the control candidate, the alignment
   diagnostic (per-input and aggregate inner products,
   sign agreements, movement norms — the SI-23/SI-25
   diagnostic definitions, unchanged) is computed on
   signal 2's train partition and on its held-out
   partition, and published with each input set
   labelled. For the treatment candidate, the same
   record is published for both partitions, together
   with the penalty term's value at initialisation
   and at the end of training on each partition
   (optimised on train, reported-only on held-out).
6. **The gradient check for the penalised loss**
   (§7.3): the finite-difference gradient check is
   re-run for signal 2's base-anchored,
   composition-penalised loss at head `d0001` (the
   decisive configuration) and at base under the new
   code path; the as-built doc publishes the measured
   agreement and the tolerance (1e-5, unchanged) for
   both.
7. **The decisive two-cycle run** (§7.5): one reused
   `TinyTransformerMind` driven through
   `si.run.run_cycles` — cycle 1 configured with
   signal 1 under the default anchoring and the
   default composition configuration, landing
   `d0001`; cycle 2 configured with signal 2,
   base-anchored targets, *and* the
   movement-composition configuration by explicit
   recorded selections, trained against `d0001` under
   the unchanged freshness contract, and judged by
   the unchanged gate. The verdict — ADMIT, REJECT,
   or ESCALATE — is reported as measured.
8. **The diagnostics, published** whatever the
   verdict: the SI-24 §5.6 set recomputed for the
   penalised candidate; the **three-run
   control/treatment table** — SI-23 head-relative,
   SI-25 base-anchored, SI-27 base-anchored +
   penalised, side by side with each figure under its
   own run's labels (signal, anchoring, composition
   configuration) — per-probe composed drift, worst
   probe, gate numbers, and candidate norms; the
   §7.5 allowed-input composition records; the
   penalised run's signal held-out records (the F3
   check, §7.6); and the falsification readings of
   §7.6 (F1/F2/F3), each stated as measured, refuted,
   or not evaluated. Each diagnostic is labelled
   measurement or derivation, per §17.
9. **Per-cycle published measurements**: gate identity
   drift (seed-absolute, cumulative), capability
   regression (head-incremental, labelled as such),
   echo accuracy per cycle; held-out numbers per
   cycle labelled with their signal id, head, target
   source, anchoring, and composition configuration;
   the penalty term's values per §14 item 5; §6.6
   VERIFY outcomes; stored-bytes growth per cycle;
   and per-cycle CPU cost.
10. **Unwind, honestly conditioned**: if cycle 2
    ADMITs, rollback of the authored two-delta stack
    to `d0001` and to base is exercised through the
    existing rollback core, with each restored head's
    probe state reported as measured. If cycle 2
    does not ADMIT, no authored two-delta stack
    exists, nothing is unwound, and the as-built doc
    says so — SI-21's fixture-proven unwind remains
    the only unwind evidence, labelled as such.
11. **An as-built doc** in the SI-14…SI-25 pattern,
    publishing every number §15 requires and stating
    plainly what is still not claimed.

Explicitly not in SI-27's scope: §16's non-goals.

---

## 15. Acceptance criteria for SI-27

A reviewer must be able to check each criterion
against a test or a published number. SI-27 does not
merge on any criterion being waived, tuned quietly,
or reported without its measurement.

**What success and failure mean for this slice, stated
before the criteria.** SI-27 *succeeds as a slice* if
a composition-penalised second cycle was genuinely
authored against the landed head under the semantics
of §7.2, genuinely judged by the unchanged gate, and
its outcome — ADMIT, REJECT, or ESCALATE — attributed
by the §14 diagnostics against *both* published
controls (SI-23 and SI-25), with the §7.6
falsification readings stated. **No criterion
requires an ADMIT.** SI-27 *fails as a slice* if any
criterion below fails — in particular if the
composition configuration in a cycle's record cannot
be read back from its own journal (criterion 1), if
cycle 1, the coincidence configuration, or the
re-run control does not reproduce its published blob
exactly (criteria 2 and 9), if any probe-derived
quantity can be shown to enter the training loss
(criterion 4), or if a number is published without
its composition configuration (criterion 8, §17). A
gate REJECT with all criteria met is a *successful
slice with a negative experimental result*, in the
SI-21/SI-23/SI-25 pattern; a gate ADMIT with any
criterion waived is a failed slice.

1. **Recorded composition configuration.** The
   configuration serves exactly the two named values
   (`"none"`, `"movement-composition"`) and refuses
   every other value with the existing refusal
   discipline, at the constructor, the setter, the
   intent, and the `train_candidate` dispatch.
   Selection is explicit and recorded: a test asserts,
   per cycle on both verdict paths, that the
   journaled `training-hyperparams` note's composition
   configuration, its λ (= 1.0 under
   `"movement-composition"`), its signal id, and its
   anchoring equal the configuration the driver set
   for that cycle — including the decisive cycle's
   full triple (signal 2, `target_anchoring =
   "base-anchored"`, composition =
   `"movement-composition"`). No autonomous, learned,
   depth-driven, or verdict-driven selection exists,
   λ is not independently settable, and exactly one
   penalised decisive run is performed (no sweep).
2. **Coincidence at base.** Configured with signal 2,
   base-anchoring, and the movement-composition
   configuration, a first cycle at base reproduces
   SI-25's at-base run exactly — candidate blob
   `sha256:131cc242728024b4f833bc78ea627825f1e8616972fc2670603673e1814645f4`,
   identity drift 0.16672966815531254, capability
   regression 0.2418281203135848 — with the penalty
   term measured at initialisation and published
   (identically zero, §7.3), and configured with
   signal 1 under the default composition
   configuration, a first cycle reproduces
   SI-19/SI-21/SI-23/SI-25's `d0001` exactly
   (criterion 9). A mismatch at base invalidates the
   decisive run's attribution and fails the slice
   until explained as a finding, not tuned away.
3. **Gradient check for the penalised loss, at the
   decisive head.** The finite-difference gradient
   check is re-run for signal 2's base-anchored,
   composition-penalised loss at head `d0001` and at
   base under the new code path, over EVERY factor
   entry, at the pinned initial factors; the as-built
   doc publishes the measured worst absolute
   disagreement and the tolerance it was checked
   against (1e-5, SI-19's pin, unchanged) for both. A
   check run only at base, or only for the unpenalised
   loss, does not satisfy this criterion.
4. **Probe-blindness of training, asserted.** A test
   establishes that the training loss under
   `"movement-composition"` is computed from exactly
   the signal's train partition and the cycle's
   landed/base movements on that partition: the
   penalty's input set equals the train partition
   (asserted as a set), the train partition's proven
   disjointness from both probe suites is re-asserted
   under the new code path, and a training run whose
   probe suites are perturbed in a copy of the bundle
   (probe inputs replaced, expectations untouched in
   the original) produces a byte-identical candidate
   — or an equivalent test the reviewer accepts as
   proving the same independence. Any path by which a
   probe input, probe logit, or probe-derived
   movement can reach the loss, the penalty, factor
   initialisation, or candidate selection fails this
   criterion and the slice with it.
5. **The allowed-input composition record,
   published.** The as-built doc publishes, for the
   re-run control candidate and for the treatment
   candidate: the alignment diagnostic on signal 2's
   train partition and on its held-out partition
   (per-input and aggregate inner products, sign
   agreements, movement norms, each input set
   labelled, measurements and derivations
   distinguished), and the penalty term's value at
   initialisation and at the end of training on each
   partition. The record is published whatever it
   shows — including if the control already opposes
   on the train partition (§7.5) — and no figure from
   it is presented as a probe figure or vice versa.
6. **Determinism per cycle.** Each cycle's candidate
   is byte-identical across repeat runs of that
   cycle (canonicalised serialisation, §17), and a
   full decisive run repeated from the seed
   reproduces both cycles' records and blob hashes
   exactly, except any explicitly labelled wall-time
   field. The penalised `train_delta` alone is
   likewise byte-identical across repeats.
7. **Gate honesty per cycle.** Each cycle's journaled
   checks 3/4 equal an independent `ProbeContext`
   evaluation of the same candidate against the same
   pre-cycle head (the SI-19 §8.8 equality, per
   cycle). Cycle 2's verdict — ADMIT, REJECT, or
   ESCALATE — is reported as measured. **No criterion
   requires an ADMIT**: SI-27 succeeds by §7's claim
   — a composition-penalised candidate was genuinely
   authored against the landed head under the new
   configuration and genuinely judged — whatever the
   verdict is.
8. **Cumulative accounting, diagnostics, the
   three-run table, and the falsification readings
   published.** The as-built doc publishes, per
   cycle: identity drift (seed-absolute), capability
   regression (head-incremental, labelled as such),
   echo accuracy vs the recorded floor, and the
   landed head's cumulative identity drift after each
   cycle — plus the full §14 diagnostic set (per-probe
   contributions and worst probes, alignment, folded-
   update inner product and norms), the §7.5 records,
   the penalised run's signal held-out records
   (pre-training and after training, labelled with
   signal, head, target source, anchoring, and
   composition configuration — the F3 evidence), the
   three-run control/treatment table with the three
   runs' figures kept distinct and labelled, and the
   F1/F2/F3 readings of §7.6, each stated as
   measured, refuted, or not evaluated. Any
   uncomputable diagnostic is reported "not
   evaluated". No incremental figure is presented as
   cumulative or vice versa, no penalised figure as
   an unpenalised one, and no train- or held-out-
   partition figure as a probe figure (§17).
9. **First-cycle and control reproduction.**
   Configured with signal 1 (the default signal,
   default anchoring, default composition
   configuration), a first cycle reproduces
   SI-19/SI-21/SI-23/SI-25 exactly — same candidate
   blob
   (`sha256:2761e8ffc139ce3e107c11da035e8a73c20792945a494f4e493dccf418b6ca9e`),
   same gate numbers — and the re-run unpenalised
   control at `d0001` reproduces SI-25's decisive
   cycle-2 candidate exactly (blob
   `sha256:646d1e8ba3978d858d4173d561f8799bce37350f88ff2de65f40c223d9e4d675`),
   so the configuration machinery is proven
   behaviour-preserving for both existing semantics
   before the new configuration's results are trusted
   (§§7.4–7.5).
10. **VERIFY, unwind, and untouched scope.** An
    ADMITted cycle 2 passes (or fails and rolls back
    from) the §6.6 probe VERIFY exactly as cycle 1
    did, with the rollback target being `d0001`;
    unwind follows §14 item 10's honest conditioning
    (authored stack unwound and re-measured if one
    exists; no unwind claimed if none does). Exactly
    two authored cycles are attempted in the decisive
    run, and exactly one penalised decisive run is
    performed; `max_stack_depth`, both tolerances,
    all caps, and every policy default are unchanged;
    both signals' ids, streams, corrections, and
    partitions are unchanged; both anchorings'
    definitions are unchanged; `si squash` still
    fails closed on transformer stacks (SI-12 tests
    green); the full existing suite is green; the
    demo path is unchanged; `si serve` still returns
    501 for every transformer bundle; no CLI wiring
    is added. Per-cycle `train_delta` wall time,
    total two-cycle wall time, and the suite's
    resulting runtime on this VM are measured and
    published. All on this VM, CPU only.

---

## 16. Non-goals

Explicit, and binding on SI-27 and on any slice this
RFC is cited to justify:

- **No Glimmer anything.** No Glimmer forward path,
  probes, training, or authorship (§12.2). No
  candidate in this RFC touches the Glimmer seed
  except to restate that it is untouched, and §7's
  penalty is a miniature mechanism that claims
  nothing at Glimmer scale.
- **No real-base work.** Candidate (a) is deferred
  with audited preconditions (§3); SI-27 builds
  nothing toward it, and no download is performed or
  scheduled.
- **No signal redefinition, no third signal.** Both
  signals' ids, streams, corrections (scale 0.15
  included), partitions, target, rank, and alpha are
  byte-for-byte as SI-19/SI-23 defined them. The
  composition penalty changes the loss a known
  signal's factors are optimised against; it does
  not change what either signal *is*, and no third
  signal is introduced (§10).
- **No probe contact in training.** No probe input,
  probe logit, probe movement, or probe-derived
  quantity enters any training loss, penalty,
  projection, initialisation, stopping rule, or
  candidate selection (§§2, 9). The binding probe is
  never named in training code. This non-goal has no
  exceptions and no "diagnostic-only" carve-out for
  the training path: diagnostics read probes after
  the gate has spoken, as they always have; training
  never does.
- **No gate change of any kind.** The gate's checks,
  verdict mapping, accounting (seed-absolute
  cumulative identity maximum; head-incremental
  capability), tolerances, caps, and every policy
  default are unchanged (§8.2). If the penalised
  cycle is REJECTed, the answer is a later RFC's
  better argument — never a wider, repriced, or
  quieter gate.
- **No λ tuning.** λ = 1.0 is the configuration's
  definition for SI-27 (§7.2). No sweep, no second
  penalised decisive run at a different λ, no
  reporting of an un-run λ's hypothetical outcome as
  anything but an open question. A future λ is a
  future RFC's argued, single, pre-registered value.
- **No gate-fitting variant of any other kind.** No
  reduced correction scale, no reduced step count or
  norm budget chosen to shrink `m2`, no signal,
  anchoring, or configuration chosen backward from
  the identity tolerance, and no selection rule a
  gate verdict can influence (§§7.1, 7.4).
- **No other widening.** No new LoRA target (§11.1),
  no additive or multi-target authored candidates
  (§11.2), and no policy default changes — in
  particular `max_targets_per_delta` stays 1 and
  `allow_additive` stays false.
- **No autonomous selection — of signal, of
  anchoring, or of composition configuration.** The
  mind chooses none of them; all three are explicit,
  recorded, driver-level configurations (§7.4), and
  no perception-driven, depth-driven, alternating,
  or verdict-driven rule is built or implied for any
  axis.
- **No transformer squash.** SI-12's fail-closed
  refusal stands throughout SI-27 (§12.1); §12.1 is
  a deferral analysis, not an implementation brief —
  even if SI-27 creates the authored stack squash
  has been waiting for.
- **No serving, no CLI wiring.** Transformer bundles
  keep returning 501; `si cycle`/`si run` keep their
  demo/mock minds. SI-27 is a library-and-tests slice
  like SI-19, SI-21, SI-23, and SI-25.
- **No third cycle, no continual-learning claim.**
  Exactly two authored cycles are attempted in the
  decisive run, on constructed signals, on a
  constructed model, under the recorded configurations
  §14 scopes. Two anchorings and a penalty term at
  depth ≤ 2 prove something about the loop's
  machinery and its gate — nothing about learning at
  any other scale or on any real task.
- **No mixed-configuration stack claim.** The
  decisive run's cycle 1 is unpenalised-at-base only
  in the degenerate sense that the penalty vanishes
  there (§7.3); SI-27 does not train an unpenalised
  cycle at a non-base head inside the decisive run
  as a claimed stack member (the §14 item 4 control
  is a separate run, labelled a control), and no
  claim is made about stacks whose cycles mix
  composition configurations at non-base heads.

---

## 17. Honesty and labelling requirements for any future slice

Binding on SI-27 and on any later slice citing this
RFC. Items 1–14 carry RFC SI-22 §12 forward unchanged
in force; items 15–16 carry RFC SI-24 §12.15–12.16
forward; items 17–19 are new, from this RFC's
findings. (SI-24 §12.17, the residual-record rule, is
retired with its subject: the residual record was
measured in SI-25 and its rule is preserved in
SI-25's as-built; no SI-27 record is a residual
record.)

1. **Training numbers are training numbers.** Losses,
   residuals, penalty values, and held-out signal
   scores must be labelled with the signal id and
   partition wherever they appear. They are never a
   gate result, never a probe result, and never
   evidence about any base but the miniature.
2. **Measured vs not evaluated survives authorship.**
   An authored candidate's checks 3/4 report exactly
   what the probes measured (`measured` a real float)
   or that they were not evaluated (`measured=None`) —
   the SI-17 labelling table applies unchanged,
   including the blind-escalated state, at every
   cycle, for every signal, under every anchoring and
   every composition configuration.
3. **Authorship is not admission.** No document,
   journal entry, or report may describe a trained
   candidate as "learned into the bundle", "accepted",
   or "verified" before the gate and (on ADMIT) the
   §6.6 probe VERIFY have actually run and been
   journaled — per cycle, per signal, per anchoring,
   per composition configuration.
4. **Every number carries its source.** A figure in
   an as-built doc is either reproduced from a test
   or run in that slice (with the test named) or
   cited to the earlier doc that measured it.
   Estimates are labelled estimates; derivations from
   cited facts are labelled derivations (as §6 labels
   theirs).
5. **The miniature's labels travel.** "Constructed,
   not trained" describes the *base*; deltas may
   honestly be described as "trained on a constructed
   signal" — and nothing in that phrase may be
   allowed to blur into a claim that the base is a
   trained model or that either signal is a real task.
6. **Failure states are reportable successes of
   honesty.** A composition-penalised REJECT, a
   derivation refuted by its test, a premise failure
   under §7.6 (F1, F2, or F3 measured), a VERIFY
   rollback to `d0001` — each is a publishable,
   journalable outcome. A slice that reports only its
   passes has failed this section regardless of its
   code.
7. **Serialisation determinism is proven, not
   assumed.** Any code that produces a
   content-addressed blob — including a
   composition-penalised candidate — must prove
   byte-identity across repeat constructions by test
   (SI-19's canonical-header lesson, carried from
   SI-22 §12.7).
8. **Head-relative numbers name their head.** Any
   loss, baseline, movement, or score computed
   against applied weights must be recorded with the
   head (and stack depth) it was computed at. Under
   two anchorings and two composition configurations
   the trap SI-22 §12.8 named multiplies: the same
   signal's pre-training loss is a closed-form
   artifact at one head under one anchoring and a
   genuine measurement under another (§5.5 of
   SI-24), and only its full label — signal, head,
   target source, anchoring, composition
   configuration — says which.
9. **Incremental and cumulative are different
   claims.** Capability regression in this runtime is
   measured against the pre-cycle head; identity
   drift against the seed. Reports must say which
   they are publishing, per number. Summing
   incremental regressions, or quoting the latest
   cycle's drift as the head's total drift, is
   fabrication by arithmetic.
10. **Composed artifacts do not inherit training
    labels.** A future squashed or otherwise composed
    delta (§12.1) is not "trained": it must be
    journaled and described as a compression with its
    own measured tolerance and probe results, never
    with its constituents' hyperparams presented as
    its own — and, after SI-25 and SI-27, never with
    either constituent's anchoring or composition
    configuration presented as its own.
11. **Signal identity is part of every label.** A
    training or gate number recorded without its
    signal id is an incomplete record. Either
    signal's numbers are never presented as the
    other's, as an unnamed "second cycle's", or as
    generic authorship numbers; the two signals'
    held-out figures are never averaged, summed, or
    otherwise blended into a single claim.
12. **Diagnostics are not verdicts.** The diagnostic
    sets — per-probe contributions, alignment, folded-
    update inner products, allowed-input composition
    records — are published to *attribute* a gate
    outcome after the gate has spoken. They are not
    gate checks, they influence no verdict, and no
    diagnostic may be quoted before a run as a
    predicted ADMIT or REJECT.
13. **Recorded selection is not a choice the mind
    made.** Signal selection (SI-22 §12.13),
    anchoring selection (SI-24 §12.13), and
    composition-configuration selection (§7.4) are
    explicit configurations recorded in provenance.
    No document may describe any of them as the mind
    "choosing", "deciding", or "preferring"; the
    honest sentence is that the driver configured
    them and the journal shows it.
14. **A derivation is not a prediction.** §6's
    composition arithmetic, §7.3's gradient-shape
    argument, and §7.2's statements about what the
    penalty rewards are labelled derivations from
    cited facts and the miniature's shape. None may
    be cited — in an as-built doc, a PR body, or a
    report — as evidence that a penalised cycle will
    ADMIT, will REJECT, or has any measured property
    at all. Only SI-27's runs measure those.
15. **Anchoring is part of every label.** A training
    or gate number recorded without its anchoring
    (`target_anchoring` and `target_source_head`) is
    an incomplete record, exactly as a number without
    its signal id is (item 11). A base-anchored figure
    is never presented as a head-relative figure or
    vice versa, and the at-base coincidence of the two
    anchorings is stated wherever an at-base figure is
    used to compare them.
16. **Controls and treatments are never blended.**
    SI-23's head-relative cycle 2 and SI-25's
    base-anchored cycle 2 are SI-27's two control
    runs. The three-run table (§14 item 8) publishes
    the runs side by side with each figure under its
    own run's labels; no figure is averaged,
    differenced into a single headline without its
    per-probe table, or quoted without the anchoring
    and composition configuration that produced it. A
    difference between the treatment and SI-25's
    control is attributable to the penalty only
    because everything else was held fixed — any
    accidental difference (a changed hyperparameter,
    stream, partition, step count, or λ other than
    the pre-registered 1.0) voids the attribution and
    must be reported as voiding it.
17. **Composition configuration is part of every
    label (new).** A training or gate number recorded
    without its composition configuration (and, under
    `"movement-composition"`, its λ) is an incomplete
    record, exactly as a number without its anchoring
    is (item 15). A penalised figure is never
    presented as an unpenalised figure or vice versa,
    and the at-base coincidence of the two
    configurations — the penalty identically zero
    where `m1` is zero — is stated wherever an at-base
    figure is used to compare them.
18. **Allowed-input numbers are not probe numbers
    (new).** A movement, inner product, or penalty
    value computed on the train or held-out partition
    is labelled with that partition wherever it
    appears, and is never presented as, averaged
    with, or substituted for the same diagnostic
    computed on the identity probes — nor the reverse.
    The train-to-probe composition gap (§6.5) can
    only ever be *stated* as the difference between
    two separately labelled records; a blended figure
    would hide exactly the quantity SI-27 exists to
    measure.
19. **Sign agreement never stands alone (new).** Any
    published sign-agreement figure is accompanied by
    the magnitude-weighted inner product over the same
    coordinates (§6.2). A sign agreement of 0.5 at the
    binding probe coexisted, in SI-25, with a positive
    inner product and a composed maximum at the
    aligned extreme; a future report that quotes the
    count without the weighting repeats a misreading
    this RFC has named.

---

## 18. Risks and open questions

Each item names what would retire it. Numbers this
RFC deliberately does **not** supply are listed as
open questions to measure — supplying them here would
be inventing them.

1. **The penalised gate outcome is genuinely
   unknown.** Signal 2's base-anchored,
   composition-penalised identity drift, capability
   regression, and echo accuracy at head `d0001` are
   **open questions to measure**. The identity
   budget's remaining headroom after cycle 1 is
   0.07810781709849834 under the unchanged accounting,
   SI-25's unpenalised treatment composed to
   0.3329036422073841, and a REJECT is a live
   possibility the slice must be willing to publish
   (§15.7, §17.6). *Retired by:* SI-27's decisive run.
2. **The train-to-probe composition gap may swallow
   the penalty (F1).** §6.4's lesson — a
   destination argument priced on training inputs
   did not price the binding probe — applies to §7's
   term with full force: the penalty shapes
   composition on 32 training sequences, and the gate
   prices composition on 8 disjoint probes. SI-27's
   defence is not a counter-argument; it is the
   §7.5 record, which measures the gap's near end
   (train and held-out partitions) for both control
   and treatment, so an F1 outcome is attributable
   rather than mysterious. *Retired by:* SI-27's
   §15 criterion 5 and criterion 8 publications.
3. **Aggregate opposition may again not move the
   maximum (F2).** SI-25 is already one measured
   instance of §6.2's arithmetic: bulk opposition,
   worst-coordinate alignment. The penalty rewards
   per-*input* opposition on allowed inputs, which is
   finer-grained than SI-25's aggregate — but the
   gate prices per-*coordinate* maxima on judged
   inputs, and nothing in the penalty's form speaks
   to coordinates individually. *Retired by:*
   SI-27's per-probe table and F2 reading, published
   either way.
4. **The penalty may trade the destination away
   (F3).** The linear term rewards opposition without
   bound in itself (§7.2); an optimiser can in
   principle satisfy it by moving against `m1` on the
   training inputs at the expense of the signal's
   destination, producing a candidate that composes
   better and authors less. The norm budget (0.9,
   unchanged) and the signal term bound the trade but
   do not forbid it. *Retired by:* SI-27's signal
   held-out records for the penalised run (§15
   criterion 8) — and, if F3 is measured, by the
   as-built saying so in §7.6's words, even on an
   ADMIT.
5. **λ = 1.0 is a parity argument, not a measured
   optimum.** The weight is fixed by units (§7.2)
   precisely so it cannot be accused of being fitted
   — which also means it may be badly scaled for the
   actual loss landscape: the penalty's magnitude
   relative to the signal term at initialisation is
   **an open question to measure** (SI-27 publishes
   both terms' values, §14 item 5). If the penalty is
   negligible at λ = 1.0, the decisive run measures
   the control again and the slice's finding is that
   parity-of-units under-weights composition; if it
   dominates, F3 is the risk. Either is publishable;
   neither licenses a second λ inside SI-27 (§16).
   *Retired by:* SI-27's published term values; any
   successor λ belongs to a later RFC arguing from
   them.
6. **A mislabelled configuration is worse than a
   wrong one.** The slice's worst failure mode is not
   a REJECT; it is a candidate trained unpenalised
   and journaled penalised (a configuration dropped
   between `propose` and `train_delta`, an `m1`
   computed against the wrong head, a penalty
   silently evaluated on the held-out partition) —
   every downstream comparison would then attribute a
   composition effect to a run that never ran the
   composition term. *Retired by:* §15 criteria 1
   and 4 — per-cycle journal assertions on both
   verdict paths, the coincidence configuration's
   exact reproduction, the re-run control's exact
   reproduction, and the probe-perturbation
   independence test.
7. **The split-movement path could silently alter
   signal behaviour under the existing
   configurations.** Generalising `train_candidate`
   from one movement-free loss to a loss that
   assembles two additional forwards (landed head and
   base, on the train partition) is the kind of
   change that passes its own new tests while moving
   an old behaviour in its last digits. *Retired
   by:* §15 criteria 2 and 9 — exact reproduction of
   *three* published blobs (signal 1's `d0001`,
   signal 2's at-base `d0001`, and SI-25's decisive
   cycle-2 candidate) under the refactored path,
   before the decisive run is trusted.
8. **Probe blindness still bounds the judge.** RFC
   SI-13 §10.1 stands, now across signals, anchorings,
   and composition configurations: 24 inputs, 28/30
   tensors measurable, blind set derived and guarded.
   The penalty changes nothing the probes can see
   *about*; it changes only how one measurable
   target's candidates are trained. A landed
   two-delta stack under any configuration is
   *evidence* of continuity sampled the same way each
   cycle, not a proof of it.
9. **Depth pressure arrives if SI-27 succeeds.** At
   `max_stack_depth` 8 with transformer squash still
   deferred (§12.1), a landed second authored delta
   makes the stack depth 2 and the compression
   question real for the first time — while the
   squash gate context remains unbuilt, and the
   stack's two deltas would carry the anchoring and
   composition-configuration questions into any
   squash design (§12.1 item 2). That is a deliberate
   fail-closed posture, recorded so the next RFC
   meets it as a design problem with a real stack to
   measure, not a surprise.
10. **Candidate (a)'s unknowns remain unknown.**
    A-1 to A-4 are audited unmet (§3); nothing in
    SI-27 or this RFC scopes them, and the download
    question is an operator decision this RFC
    explicitly does not take. Likewise §12.2: success
    on the miniature renews the "why not Glimmer"
    pressure, and the answer does not soften because
    the miniature's training has gained a third
    recorded configuration axis.

---

## 19. Summary

After SI-25, SI can originate a miniature delta,
judge it, land it, and originate a second one on top
of it under a second signal and under two target
anchorings — and the gate's cumulative, seed-absolute
identity accounting has been measured refusing all
three second writes, the third with its cause
attributed more precisely than the first two: the
base-anchored update *opposes* the first update in
aggregate, in weight space, and at the previous worst
probe, and the composed maximum still worsens,
because at `tiny-id-0` — the probe where cycle 1 had
already moved most — the two movements add almost
exactly (0.17189218290150166 + 0.16173836588859558 ≈
0.3329036422073841 > tol 0.25, tolerances unchanged).
Of the candidate frontiers this RFC evaluated against
that evidence: a small real base stays deferred, its
four SI-18 preconditions audited and all still unmet,
with a checkpoint download flagged as an operator
decision rather than a step (§3); a second signal and
base-anchored targets are done, their questions
answered by SI-23 and SI-25 (§§4–5);
per-probe movement accounting is adopted as
diagnostics inside the recommended slice and rejected
as a gate change, because repricing the judge after
three measured failures is loosening by redefinition
(§8); orthogonalization at the binding probe is
rejected outright — it trains on the judge — and its
weight-space variant is refuted by SI-25's own table
(§9); a third or different signal is rejected, the
family's composition behaviour being the measured
obstacle (§10); a new target stays deferred behind a
backward derivation nobody has written (§11.1), and
additive / multi-target authorship stays rejected as
a next slice because it would be bought with a
policy loosening (§11.2); transformer squash stays
deferred, its precondition — an authored two-delta
stack — still unmet after three measured attempts
(§12.1); Glimmer probing stays rejected, not reopened
(§12.2); and **movement-composition-aware training —
signal 2, base-anchored, at the landed head, with one
penalty term of pre-registered weight λ = 1.0 computed
only over the signal's own train partition, recorded
as a third configuration axis beside signal and
anchoring, judged by the unchanged gate on probes the
training never reads, with its premise stated in
three falsifiable forms — is recommended as SI-27**
(§§7, 14, 15). Its plausibility argument is stated
with its limits in the same sections that make it:
SI-25 proved a second update's movement *can* oppose
the first's on most judged inputs when its
destination is fixed, and proved just as plainly that
indifference to composition on the training inputs
leaves the binding probe to chance. Whether
accounting for the first movement while training —
on the only inputs training may see — moves the one
figure the gate prices is measurable with instruments
that already exist, against two published controls,
with exactly one variable changed. That is a
plausibility argument for running the experiment —
nothing in this section is a prediction that it
ADMITs, and §15 requires no ADMIT. The deepest landed
authored stack remains depth 1; no two-delta authored
stack exists; the base stays frozen, the bytes are
never duplicated, no tolerance moves, the gate is
unchanged, and nothing is claimed that a test in this
repo cannot check.

---

*RFC SI-26 — design only. Implementation begins, if
approved, at SI-27 as outlined in §14. RFC SI-24
changed what the loop aims at — a fixed destination
instead of a further step — and the gate told the
truth about it, per probe: the aim was reached in
aggregate and missed where it binds. This RFC
proposes changing not what the loop says, nor what it
aims at, but what it remembers while it aims: the
first movement it already made, accounted for on the
inputs it is allowed to see, under the same judge, at
the same scale, with the verdict still not ours to
pick.*
