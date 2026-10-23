# SI-24 RFC — The Next Frontier After the Second Signal
(Design Only)

> **Status: DESIGN ONLY.** This document evaluates candidate
> directions and recommends one next slice. It changes no
> production code, no test, no manifest schema, no policy
> default, and no gate behaviour. Nothing in this RFC is
> implemented unless a section explicitly says it exists today,
> in code, on `main`.
>
> **The one sentence that governs everything below:** SI-23
> proved the miniature's second *signal* can be authored
> against the landed head under settled semantics and
> genuinely judged — and the gate REJECTED it on cumulative
> identity drift, with published diagnostics attributing the
> rejection to *movement alignment* between the two updates —
> and this RFC must never be cited as evidence that a
> base-anchored target has been implemented or tested, that
> any authored two-delta stack exists, that a cycle-2 ADMIT
> is achievable by any route, or that any slice proposed here
> exists.
>
> **Scope discipline:** CPU only. No GPU, no cloud, no spend,
> no downloads. Terminology: this is machine learning
> infrastructure throughout.
>
> **Relationship to prior work:** RFC SI-22 evaluated the
> frontiers open after SI-21 and recommended a second
> training signal; SI-23 built it (PR #28, as-built in
> `docs/SI-23-second-signal.md`) and the gate's cycle-2
> verdict was a measured REJECT, attributed by SI-23's own
> pre-registered diagnostics. This RFC is SI-22's successor
> in the established cadence (SI-18 RFC → SI-19 build →
> SI-20 RFC → SI-21 build → SI-22 RFC → SI-23 build): it
> restates the verified state after SI-23, re-evaluates
> every SI-22 candidate against SI-23's measured evidence,
> and gives the route RFC SI-22 §5.7 deferred — base-anchored
> targets — the deep design treatment that section promised
> once SI-23's diagnostics existed to design against.
> SI-22's rejection of Glimmer probing is carried forward,
> not reopened (§7). The SI-9 RFC's §7 risks remain in force
> and are inherited, not relitigated.

---

## Table of contents

1. [Where SI stands after SI-23](#1-where-si-stands-after-si-23)
2. [Inherited invariants](#2-inherited-invariants)
3. [Candidate (a): a small real safetensors base — still deferred](#3-candidate-a-a-small-real-safetensors-base--still-deferred)
4. [Candidate (b): a second training signal — DONE as SI-23](#4-candidate-b-a-second-training-signal--done-as-si-23)
5. [Candidate (c): base-anchored targets — the deep treatment](#5-candidate-c-base-anchored-targets--the-deep-treatment)
6. [Candidates (d) and (e): new target; additive / multi-target](#6-candidates-d-and-e-new-target-additive--multi-target)
7. [Candidate (f): transformer squash; candidate (g): Glimmer probing](#7-candidate-f-transformer-squash-candidate-g-glimmer-probing)
8. [Decision summary](#8-decision-summary)
9. [The recommended slice: SI-25 (outline)](#9-the-recommended-slice-si-25-outline)
10. [Acceptance criteria for SI-25](#10-acceptance-criteria-for-si-25)
11. [Non-goals](#11-non-goals)
12. [Honesty and labelling requirements](#12-honesty-and-labelling-requirements-for-any-future-slice)
13. [Risks and open questions](#13-risks-and-open-questions)
14. [Summary](#14-summary)

---

## 1. Where SI stands after SI-23

Everything in this section is verifiable in the code and docs
on `main` at the SI-23 merge (main tip `3294235`, the merge of
PR #28). Numbers are cited to the as-built document that
measured them; this RFC measures nothing new. Where a
statement below is a *derivation* from cited facts rather
than a measurement, it is labelled a derivation, exactly as
SI-22 §1 labelled its arithmetic.

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
- **Two authored signals, one landed delta (SI-19 + SI-23).**
  The signal registry (`SIGNAL_REGISTRY`, `signal_spec` in
  `si/tiny_training.py`) serves exactly two named signals —
  `tiny-successor-lean-001` (SI-19) and
  `tiny-predecessor-lean-001` (SI-23) — and refuses every
  other id with the existing `ValueError` discipline.
  Selection is explicit and recorded: the mind is configured
  with the signal it will propose (constructor-level
  `training_signal_id`, defaulting to signal 1, or
  `set_training_signal(...)` between cycles,
  `si/tiny_transformer.py`); no autonomous selection exists.
  Both signals share one form (target-logits MSE), one LoRA
  target (`lm_head.weight`, rank 8, alpha 8), correction
  scale 0.15, 2,000 full-batch steps, learning rate 1.0,
  norm budget 0.9, 32 train / 16 held-out examples; they
  differ only in correction direction, stream, and drawn
  sequences. The audited stream table now has six offsets
  (+2 identity probes, +3 capability probes, +11/+17 signal
  1 sequences/init, +23/+29 signal 2 sequences/init), all
  pairwise distinct, asserted by test.
- **Named target anchoring, unchanged (SI-21).**
  `TARGET_ANCHORING = "weights-under-training"`
  (`si/tiny_training.py:214`): targets are derived from the
  weight set under training — the bundle's applied weights
  at the head being trained against — plus the signal's
  recorded correction, exactly as `build_training_signal`
  computes them. Every cycle's hyperparams record `head`,
  `target_source_head`, `target_anchoring`, and
  `stack_depth`; under the current anchoring the two head
  fields always agree (`si/tiny_training.py`, the
  hyperparams record in `train_candidate`: under
  `TARGET_ANCHORING` the target source IS the head trained
  against). The alternative — base-anchored targets — was
  considered and not chosen (SI-20 §4.3.2), returned to and
  deferred again (SI-22 §5.7); §5 of this RFC is that
  deferred treatment, now due.
- **Sequential authorship of one signal, measured (SI-21).**
  Cycle 1 (signal 1 at base) ADMITted as `d0001`, reproducing
  SI-19 exactly (candidate blob
  `sha256:2761e8ff…6ca9e`, identity drift
  0.17189218290150166 ≤ tol 0.25, capability regression
  0.1657181391492486 ≤ eps 0.25, echo 16/16, §6.6 VERIFY
  pass). Cycle 2 (signal 1 retrained at `d0001`,
  head-relative) was **REJECTED**: cumulative identity drift
  **0.34378434903919697** > tol 0.25, head-incremental arms
  passing (capability regression 0.16571810841560364, echo
  1.0), head stayed `d0001`, blob-store growth +0 B.
- **A second signal, authored and measured (SI-23).** The
  decisive two-cycle run — one reused mind, cycle 1 signal 1
  (default selection), cycle 2 signal 2 (explicit recorded
  selection), both under the unchanged freshness contract
  and weights-under-training anchoring:
  - Cycle 1 reproduced SI-19/SI-21 **exactly** — same
    candidate blob
    `sha256:2761e8ffc139ce3e107c11da035e8a73c20792945a494f4e493dccf418b6ca9e`,
    same gate numbers — so the registry refactor is proven
    behaviour-preserving for the existing signal.
  - Cycle 2's candidate was genuinely new (blob
    `sha256:184cc7f06cd9cb3bd83fbed466b6a939c7e2e18cf7b8a72be70c51043a6e6cf7`,
    2,360 B, norm 0.21317186599953494, trained at `d0001`
    with signal 2's own stream, init, and sequences) and as
    a *head-relative* improvement it succeeded (held-out
    loss at its head fell ~361×, 16/16 examples closer).
    The gate **REJECTED** it: cumulative identity drift
    **0.2963416576385498** > tol 0.25 (worst probe
    `tiny-id-3`), capability regression
    **0.15023326850496233** ≤ eps 0.25 (pass), echo 1.0
    (pass). Nothing landed; head stayed `d0001`; blob-store
    growth +0 B.
  - Signal 2 at base (the first-cycle sanity configuration)
    **ADMITted** → `d0001` in its own run: identity drift
    0.16672966815531254 ≤ tol 0.25 (worst `tiny-id-7`),
    capability regression 0.2418281203135848 ≤ eps 0.25,
    candidate blob
    `sha256:131cc242728024b4f833bc78ea627825f1e8616972fc2670603673e1814645f4`,
    norm 0.21317186144376168. These are at-base facts about
    signal 2's definition — never evidence about head
    `d0001` (SI-23, "What is still NOT claimed").
  - Signal 2's gradient was finite-difference checked in
    both configurations (worst absolute disagreement
    5.5249277028738815e-08 at base, 5.062663472655505e-08 at
    head `d0001`, tolerance 1e-5, unchanged). Disjointness
    was proven against both probe suites and signal 1's
    full 48 sequences; the measured last-token overlap is
    exactly {9, 22} (2 of 8).
- **The attribution SI-23's diagnostics bought (the evidence
  this RFC designs against).** RFC SI-22 §5.6.4 pre-registered
  a diagnostic set; SI-23 published it in full, and it
  attributes the cycle-2 REJECT to *movement alignment*, not
  to an oversized second update:
  - At the worst composed probe (`tiny-id-3`), the composed
    drift is, to the printed digits, the *sum* of the two
    contributions: `d0001`'s drift 0.14782869815826416 +
    signal 2's incremental movement 0.14852380752563477 ≈
    0.2963416576385498 — the aligned extreme of the §5.6.4
    range, at that probe.
  - The alignment diagnostic (a *derivation* from two
    measured forwards, labelled as such in SI-23): at
    `tiny-id-3`, inner product **+0.021769049613635714**
    (positive — aligned, not cancelling), sign agreement
    **0.625** (20/32 coordinates); aggregated over all 8
    identity probes, inner product
    **+0.07637087369869666**, sign agreement **0.54296875**
    (139/256); movement-vector norms ‖m1‖ =
    0.7119678353441855, ‖m2‖ = 0.6234748221249689.
  - The folded updates align in weight space too
    (measured): Frobenius inner product
    **+0.004356018453758104**; ‖Δ1‖_F =
    0.23555380990005192, ‖Δ2‖_F = 0.21317186599953494.
  - Δ2 was *not* oversized: its own worst incremental
    movement (0.16672981530427933, at `tiny-id-7`) is
    *smaller* than cycle 1's drift (0.17189218290150166),
    and its norm is smaller than Δ1's. What failed is the
    composition: the remaining headroom after cycle 1 was
    **0.07810781709849834** (tol 0.25 − 0.17189218290150166),
    Δ2's incremental movement exceeds it at **5 of 8**
    probes, and at the worst probe the movements add rather
    than cancel.
  - The alignment is not explained away by token overlap:
    the two signals share only 2 of 8 last tokens, yet their
    movements still align — consistent with SI-22 §5.6.4's
    warning that both corrections share a component (both
    move *away* from the last token itself) and that a
    rank-8 readout update acts on every hidden state it
    meets.
- **Unwind, still fixture-only (SI-21, unchanged by SI-23).**
  Because cycle 2 REJECTED in both SI-21 and SI-23, **no
  authored two-delta stack exists**; nothing was unwound in
  SI-23 and no unwind is claimed. SI-21's fixture-proven
  unwind (authored `d0001` + a benign hand-built fixture
  `d0002`, labelled a fixture) remains the only unwind
  evidence. The transformer-squash precondition (an authored
  two-delta stack) remains **unmet** (§7.1).
- **Suite state:** 315 passed, 2 skipped on `main` at the
  SI-23 merge (SI-23 as-built, author's run: the SI-21
  baseline of 302 passed, 2 skipped, plus SI-23's 13 new
  tests in `tests/test_second_signal.py`, zero regressions).

**What does not exist, exactly (SI-23, "What is still NOT
claimed", verified against the code on `main`):**

- **No authored stack deeper than one.** The deepest landed
  authored stack on record remains depth 1 (`d0001`). Both
  attempted second cycles (SI-21 same-signal, SI-23
  second-signal) exist only as rejected, never-written
  candidates (blob-store growth +0 B each).
- **Signal 2 ADMITs at base — and only at base.** Its
  at-base first cycle ADMITted (drift 0.1667); that is a
  fact about signal 2's definition against the seed, not
  evidence that a signal-2 delta lands on any non-base
  head, and it is never presented as such.
- **No new semantics.** `TARGET_ANCHORING` stays
  `"weights-under-training"`; the base-anchored route
  (SI-22 §5.7) was not implemented or tested in SI-23.
  SI-23's diagnostics are the measured evidence a later RFC
  would design that route against — this RFC is that later
  RFC, and §5 is that design.
- **No third cycle, no continual learning.** Exactly two
  authored cycles were attempted in each decisive run, on
  constructed signals, on a constructed model.
- **No Glimmer anything.** No Glimmer forward path, probes,
  training, or authorship; the Glimmer seed is untouched and
  its deltas still escalate with checks 3/4 "not evaluated"
  (§7.2).
- **No widening.** Two signals, one target, one bundle kind;
  no additive or multi-target authorship; no policy default
  changed; no tolerance moved; signal 1 is byte-for-byte
  SI-19's and signal 2 is as SI-23 defined it.
- **No serving, no CLI wiring for the miniature.**
  `si serve` returns 501 for every transformer bundle
  (`si/serve.py`); `si cycle` / `si run` accept only
  `--mind demo|mock`.
- **No transformer squash.** SI-12's fail-closed refusal
  stands: `si/squash.py::squash_bundle` raises `SquashError`
  before any write if any record on the resolved active
  chain has kind `transformer-delta-v1`.

The frontier question is therefore precise in a new way:
the loop can originate a miniature delta, judge it, land it,
and originate a second one on top of it under two different
signals — and the gate's cumulative identity accounting has
now been measured refusing both second writes, the second
time with the cause attributed per probe. Twice, the second
update was asked to lean *further from the head*; twice,
its movement *added* to the first update's movement at the
identity probes. The one route SI-21 named that has never
been tried asks for something structurally different: not a
further step from the head, but the residual toward a fixed
destination (§5).

---

## 2. Inherited invariants

Not negotiable inside this RFC (SI-9 §1.3, RFC SI-13 §2,
RFC SI-18 §2, RFC SI-20 §2, RFC SI-22 §2, restated for the
candidates):

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
   to fixtures, at every cycle, for every signal, under
   every anchoring.
4. **Honest labels are load-bearing.** "Not evaluated" stays
   distinguishable from a measured 0.0; a training loss
   stays distinguishable from a gate result; a number
   measured at one head stays labelled with that head
   (SI-20 §12.8); incremental and cumulative figures are
   never presented as each other (SI-20 §12.9); a number
   recorded under one anchoring is never presented as a
   number under the other (§12 of this RFC).
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
   addressing (SI-19's canonical-header lesson, SI-22
   §12.7).
8. **No tolerance is a design variable.** No candidate in
   this RFC proposes moving `identity_tol`,
   `capability_eps`, any cap, or any policy default to make
   an outcome pass. SI-21's and SI-23's cycle-2 REJECTs are
   measurements to design *around* — with new semantics or
   a new signal — never numbers to outvote (SI-21 as-built;
   SI-22 §11).

---

## 3. Candidate (a): a small real safetensors base — still deferred

**Verdict up front: DEFERRED** — unchanged from SI-22 §3,
and this RFC does not reopen it. SI-20 audited SI-18 §3.7's
four preconditions (A-1 feasibility measurements for one
named family; A-2 a binding design for `reference-v1`
bases; A-3 a capability-suite design for a trained base;
A-4 a one-family executor design with SI-15-grade numeric
proof) and found all four unmet; SI-22 re-audited them
unchanged after SI-21. SI-23 changed none of them: it built
a second signal *for the miniature*, using the miniature's
executor, the miniature's binding, and the miniature's
constructed signals. Nothing in SI-23's registry, stream
audit, or diagnostics is a measurement of any real
checkpoint family, and citing any of it toward A-1 would be
exactly the unlabelled transfer SI-18 §10.4 forbids.

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
SI-25 because §5's candidate is buildable now with
instruments that already exist and answers the question
SI-23's diagnostics have just made precise, while §3.5's
slice retires overheads, not any of A-1 to A-4 as stated.

| Precondition (SI-18 §3.7) | Status today | Changed by SI-23? |
|---|---|---|
| A-1 feasibility measurements, one named family | Not met — none exist | No |
| A-2 `reference-v1` binding design + tamper analysis | Not met — binding unstatable for referenced bases (`si/probes.py`) | No |
| A-3 capability-suite design for a trained base | Not met — unsolved design problem | No |
| A-4 one-family executor design + numeric proof | Not met — executor refuses all but the miniature | No |

---

## 4. Candidate (b): a second training signal — DONE as SI-23

**Verdict up front: DONE** — this was SI-22's recommended
slice, it merged as PR #28, and its result is the evidence
base for everything below. Restating the outcome without
inflation:

- The registry serves exactly two signals; selection is
  explicit, recorded per cycle, and asserted by test on
  both verdict paths (§1).
- The freshness contract and weights-under-training
  anchoring were inherited unchanged; signal 2's gradient
  was checked at the decisive head; disjointness and the
  stream audit hold, with the last-token overlap measured
  and published ({9, 22}, 2 of 8).
- Composition across signals is proven in SI-22 §5's exact
  sense: the second write was genuinely authored against
  the landed head under a *different* signal and genuinely
  judged. The judgement was REJECT.
- The §5.6.4 diagnostics were published in full and did
  the work they were pre-registered to do: the REJECT is
  *attributed* — movement alignment at the identity
  probes, composed drift adding almost exactly at the
  worst probe — not merely recorded. Identity continuity
  remains the binding constraint, now measured for two
  differently directed signals as well as for the same
  signal restacked (SI-21).
- What SI-23 did **not** produce: a landed second delta,
  an authored two-delta stack, or any evidence about a
  third cycle. Its own closing section binds this RFC:
  signal 2 ADMITs at base only; no authored stack deeper
  than 1 exists; the base-anchored route was not
  implemented or tested, and SI-23's diagnostics "are the
  measured evidence a later RFC would design that route
  against".

Candidate (b) is therefore closed, not deferred: its
question is answered. The open frontier is the route both
SI-20 and SI-22 named alongside it and deferred twice —
new semantics (§5) — against the alternatives re-evaluated
in §§6–7.

---

## 5. Candidate (c): base-anchored targets — the deep treatment

**Verdict up front: RECOMMENDED as the next slice
(SI-25)** — conditionally, and with the condition stated
before the design: **no derivation in this RFC shows that
a base-anchored second cycle will ADMIT at head `d0001`.**
What the evidence supports is narrower and, this RFC
argues, sufficient: the route was named before either
REJECT existed (SI-20 §4.3.2), deferred twice for stated,
checkable reasons (§5.1), and SI-23 has now retired the
last of those reasons by publishing the diagnostics the
deferral asked for; the route changes exactly one variable
against SI-23's decisive run — the target source — so its
outcome, ADMIT or REJECT, is attributable to the semantics
change and to nothing else; its mechanism speaks directly
to the measured cause of both REJECTs (head-relative
targets ask for a further step from the head, and the
further step's movement *adds*); and both of its possible
gate outcomes are honest, publishable results. As in
SI-21 and SI-23, **no acceptance criterion in §10
requires an ADMIT.**

### 5.1 Why the deferral is retired

SI-22 §5.7 deferred base-anchored targets "again — not
rejected", for four stated reasons. Each is re-examined
here against SI-23's evidence, because a deferral whose
reasons have expired is a decision, and this RFC must
make it explicitly:

1. **"Confounding, again — now in the other direction"**
   (§5.7 reason 1). SI-23's question was whether a *new
   signal* lands under the semantics SI-21 settled;
   changing the anchoring in that slice would have stacked
   two unproven changes. *Status: retired by SI-23's
   completion.* The new-signal question is now answered
   under settled semantics (§4). The next slice can change
   the anchoring *alone*, holding signal, target,
   hyperparameters, streams, and head fixed against
   SI-23's decisive cycle 2 — the control run now exists,
   which is precisely what was missing before.
2. **"It is a semantics change, not a parameter tweak"**
   (§5.7 reason 2). The echo-trap record changes character;
   the accounting must be re-derived against a fixed
   origin; comparability with the head-relative cycles
   must be argued, not assumed. *Status: not retired —
   discharged by design.* §5.5 re-derives the echo-trap
   record for the new semantics; §5.6 re-derives the
   accounting; §5.3 designs the representation so the two
   semantics can never be confused in a record; §12 makes
   the labelling binding. A semantics change is a reason
   to design carefully, and after two deferrals it is no
   longer a reason to defer.
3. **"No measured evidence yet says it is needed"**
   (§5.7 reason 3). The predecessor route was unmeasured;
   SI-22 wrote that if SI-23 REJECTED "and the §5.6.4
   diagnostics attribute the rejection to movement
   alignment under head-relative targets, base-anchoring
   becomes the *informed* next candidate — designable
   against measured diagnostics instead of against this
   RFC's speculation." *Status: retired, on SI-22's own
   terms.* SI-23 REJECTED, and its diagnostics attribute
   the rejection to movement alignment under head-relative
   targets (§1: positive inner products in logit space and
   in weight space, composed drift equal to the sum at the
   worst probe, an un-oversized second update). The
   condition SI-22 set for this candidate is met verbatim.
4. **"Sequencing is the honesty safeguard"** (§5.7 reason
   4). Adopting new semantics *first*, for the *same*
   signal, immediately after that signal's measured
   REJECT, would invite the reading that the semantics
   were redesigned backward from a desired verdict.
   *Status: retired by the sequence itself.* The semantics
   were not adopted first: the new-signal experiment ran
   first, under unchanged semantics, and REJECTED — and
   that REJECT is published, attributed, and stands
   whatever SI-25 measures. Base-anchoring is also not a
   post-hoc invention: it was named in SI-20 §4.3.2,
   before SI-21 ran, as one of the two definable
   anchorings, and its design reasons are stated in §5.2
   before any base-anchored measurement exists. The
   argument is made here, in its own RFC, on evidence —
   the clean argument §5.7 reason 4 asked for.

Two further routes to a second cycle were considered and
are disposed of here, so no later reader has to excavate
them:

- **A third signal under head-relative anchoring** (a new
  member of the lean family, or a differently formed
  signal on the same target): SI-23's attribution is
  about the *family's* composition behaviour, not about
  predecessor-lean in particular — both corrections share
  the away-from-the-last-token component, and both trained
  updates are rank-8 readout matrices applied to every
  hidden state the probes present. No third signal in this
  family has an argued reason to compose differently, and
  searching the family for one whose movements happen to
  fit the 0.07810781709849834 headroom is the gate-fitting
  trap of §5.6.1/§13 in a new costume: a signal chosen
  backward from the tolerance. Rejected as the next slice.
- **The same signals at a reduced correction scale:** the
  SI-22 §5.6.1 gate-fitting trap, unchanged. A scale
  chosen because the arithmetic fits the budget converts
  the gate from a judge into a specification. §11 keeps
  it an explicit non-goal.

### 5.2 What base-anchored means, precisely

Verified current state, so the change is statable in one
paragraph: `train_candidate` assembles **one** weight set
— the bundle's applied weights at the manifest's head, via
`assemble_weights(bundle_dir, manifest, policy)`
(`si/tiny_transformer.py:297`, one hash-verified
`load_applied_tensor` call per tensor) — and uses that
single set for everything: `build_training_signal(weights,
signal_id)` derives each example's targets from *those*
weights' last-position logits plus the signal's recorded
correction (`si/tiny_training.py:356`); the cached final
hidden states and the readout under training come from the
same set; the pre-training losses are that set's losses
against its own derived targets; and the trained factors
are evaluated folded into that same set. Under
weights-under-training anchoring the identity "target
source = weights under training = head" is therefore not
a policy layered over the code — it is the code's shape:
one weight set, four roles.

**The design SI-25 must implement (this RFC's
recommendation):** split the *target-source* role from the
other three, and nothing else.

- **Definitions.** At any cycle, the **weights under
  training** are the bundle's applied weights at the head
  being trained against (the freshness contract's verified
  manifest, head = `d0001` in the decisive run) — unchanged
  in every role they hold today: hidden states, the readout
  being updated, candidate construction against the
  verified manifest, the gate, VERIFY. The **target source**
  is the weight set whose last-position logits the signal's
  targets are derived from. Under `weights-under-training`
  it is the weights under training (today's behaviour,
  byte-for-byte). Under the new anchoring —
  **`base-anchored`** — it is the bundle's **frozen seed
  base**: the applied weight set at head `base`, i.e. the
  base tensors with an empty active chain, at every cycle.
- **The target formula does not change; its input does.**
  Each example's target remains the target source's own
  last-position logits plus the signal's recorded
  correction (`v = 0.15 * (onehot(neighbour) -
  onehot(last_token))`, neighbour per the signal's
  direction). Signal definitions, ids, streams, partitions,
  and corrections are untouched (§11). What a base-anchored
  cycle at `d0001` *asks for* is therefore statable in one
  sentence: **make the composed head's outputs equal what
  the seed base's outputs would be after the signal's lean
  — a fixed destination, independent of where the head
  currently stands** — where a head-relative cycle asks
  for one more lean-step from wherever the head stands.
  Later base-anchored cycles correct the *residual*
  toward that destination; they do not re-ask for the
  full step.
- **The records say which happened, in fields that already
  exist.** Every cycle's hyperparams already record
  `head`, `target_source_head`, `target_anchoring`, and
  `stack_depth` (SI-21). A base-anchored cycle 2 records
  `head = d0001`, `target_source_head = base`,
  `target_anchoring = "base-anchored"`, `stack_depth = 1`
  — the first record in the project's history where the
  two head fields disagree, which is exactly what the
  fields were recorded for. §12 makes the anchoring part
  of every number's label.
- **What does not move.** The freshness contract (exactly
  one manifest trains — the cycle's verified manifest;
  `refresh_manifest` remains its only mechanism), candidate
  construction (against the verified manifest, parent =
  the cycle's head), the gate (unchanged code, unchanged
  accounting — it judges the composed head against the
  seed exactly as today), §6.6 VERIFY, rollback, and every
  policy default. Base-anchoring is a property of how a
  candidate is *trained*, never of how it is *judged*.

### 5.3 The representation: anchoring as a recorded configuration

How the choice is represented decides whether the two
semantics can be told apart in every record forever after.
Three representations were considered:

1. **Flip the global constant.** `TARGET_ANCHORING`
   becomes `"base-anchored"` for the whole runtime.
   *Rejected.* It rewrites the meaning of every future
   cycle silently — a later head-relative experiment
   becomes impossible without another flip — and it makes
   the semantics a property of the code version rather
   than of the cycle, so a journal spanning the flip
   cannot be read without knowing which code wrote each
   entry. It also destroys the control/treatment design of
   §5.7: SI-25's decisive comparison requires both
   anchorings to exist in one runtime, selectable per
   cycle, with the head-relative default proven untouched.
2. **New signal ids for anchored variants** (e.g. a
   `…-anchored-001` twin per signal). *Rejected.*
   Anchoring is not signal identity: SI-21 already records
   it as a separate axis beside the signal id, and the two
   axes vary independently (either signal can, in
   principle, be trained under either anchoring). Twinning
   the ids would mislabel a semantics axis as a signal
   axis, double the registry for no new signal, and invite
   exactly the blended reporting §12 forbids.
3. **An explicit, recorded, per-cycle configuration —
   RECOMMENDED.** Anchoring joins signal selection as a
   driver-level configuration of the mind, mirroring
   SI-22 §5.2's recorded-selection design exactly: a
   constructor-level named choice on `TinyTransformerMind`
   (defaulting to `"weights-under-training"`), settable
   between cycles by an explicit driver call in the shape
   of `set_training_signal(...)`, validated against the
   two named values with the existing refusal discipline
   (any other value refuses), carried on the intent, and
   recorded in every cycle's hyperparams as
   `target_anchoring` / `target_source_head` — asserted by
   test, per cycle, on both verdict paths, to equal the
   anchoring the driver configured. There is no autonomous,
   learned, depth-driven, or verdict-driven anchoring
   choice, for SI-22 §5.2's reason applied to anchoring
   itself: any such rule would be a second unproven
   semantics stacked on the first, and a configuration the
   gate's outcome could influence would be a feedback loop
   from verdict to training semantics — the gate-fitting
   trap given a mechanism. A driver that wants
   base-anchored targets in cycle 2 says so, in code, and
   the journal shows it.
   
   One consequence of this representation must be stated
   plainly: the module-level `TARGET_ANCHORING` constant
   survives as the *default's* name, not as a description
   of every cycle. SI-25 must keep the constant, the
   docstrings, and the hyperparams field coherent — a
   cycle's anchoring is read from its own record, never
   inferred from the constant (§10 criterion 1).

**First-cycle compatibility is the bar, as in SI-23.** At
head = `base` the two anchorings coincide — the weights
under training *are* the base — so a first cycle must
reproduce SI-19/SI-21/SI-23 exactly under *either*
configuration: same candidate blob
(`sha256:2761e8ff…6ca9e` for signal 1), same gate numbers.
That coincidence is not a triviality; it is the strongest
available correctness check on the new target-source
assembly (§5.4), because any error in assembling the base
weight set shows up at cycle 1 as a changed blob, before
the decisive cycle is trusted (§10 criteria 2 and 9).

### 5.4 Target-source assembly: the base, from the verified manifest

The one piece of genuinely new machinery the route needs
is a second assembled weight set — the seed base's — inside
a cycle whose manifest head is `d0001`. The design
constraints, each load-bearing:

- **Derived from the cycle's verified manifest, in exactly
  one place.** The base weight set is the applied weight
  set of *the same manifest the freshness contract
  refreshed*, with its active chain truncated to empty —
  the state rollback to `base` already defines and
  `_verify_probe_head` already measures (drift 0.0 at base,
  SI-21). The derivation lives beside `assemble_weights`
  in `si/tiny_transformer.py` and nowhere else; the
  verified manifest itself is never mutated, re-headed, or
  written — the base view is an in-memory assembly input,
  not a bundle state. A test must be able to tell the two
  assemblies apart (the base view's readout equals the
  recorded base tensor; the head view's readout equals
  base + `d0001`'s fold), in the shape of SI-21's
  stale-control test.
- **Hash-verified reads only, no new bytes.** Assembly goes
  through the existing apply-at-load path
  (`load_applied_tensor`, `si/apply.py:175`) with the
  truncated chain — every tensor value still comes from a
  hash-verified per-tensor read of the frozen base, and a
  rolled-back or active delta simply does not fold. No
  tensor is copied, materialised to disk, or stored: the
  never-duplicate-bytes rule (invariant 2) is satisfied by
  construction, and the slice's blob-store accounting must
  show it (growth attributable only to a landed candidate,
  exactly as in SI-21/SI-23). This is a miniature-local
  assembly over 30 small tensors; nothing here is a
  Glimmer-scale mechanism and none is claimed (§7.2).
- **Fail closed.** A bundle whose base cannot be assembled
  and verified — tampered manifest, missing base tensor,
  architecture mismatch — refuses training, exactly as
  head assembly refuses today. There is no fallback from a
  failed base assembly to head-derived targets: a silent
  fallback would train a head-relative candidate wearing a
  base-anchored label, the worst labelling failure this
  design can produce (§13, risk 4).
- **Cost stated, not hidden.** Base assembly costs one
  extra 30-tensor assembly per base-anchored cycle (the
  head assembly still happens; the targets' forwards run
  over the base set once per example). On the miniature
  this is the same order as SI-23's per-cycle
  `train_delta` wall times (~1 s); SI-25 measures and
  publishes its own per-cycle wall times (§10 criterion
  10) rather than inheriting that estimate.

### 5.5 The echo trap under base-anchoring: the record changes character

RFC SI-18 §4.3's rule, restated by SI-20 §5.1 and SI-22
§5.6.3: each signal must record the score of the weights
it trains, before training, and be measurably imperfect
there — a signal the head already satisfies authors
nothing. Under weights-under-training anchoring this
record degenerated, honestly and by construction, into a
closed form: targets derive from the weights under
training, so the pre-training held-out loss at any head is
`2 * 0.15**2 / 32 = 0.00140625` — confirmed by measurement
at base and recurring by construction at `d0001` in both
SI-21 and SI-23, and labelled a construction artifact
wherever it appeared.

**Under base-anchoring that closed form does not
transfer, and SI-25 must never quote it for a non-base
head.** The pre-training held-out loss of a base-anchored
cycle at `d0001` is the loss of the *head's* weights
against targets derived from the *base* — two different
weight sets — and it is a genuine measurement carrying
real information:

- **For signal 2 at `d0001` (the decisive configuration):**
  the record measures how far the landed head stands from
  signal 2's fixed destination (base outputs leaned toward
  the predecessor). It contains, inseparably, `d0001`'s
  successor-lean movement and the predecessor correction's
  own size; no closed form is derived for it here, because
  `d0001`'s movement on signal 2's *held-out sequences* is
  a per-sequence quantity no cited number fixes. SI-25
  measures it and publishes it labelled with signal id,
  head, target source, and anchoring. What the record must
  show for authorship to be meaningful is the narrow,
  sufficient thing (SI-22 §5.6.3's phrasing, carried over):
  the loss is *not ~0* — the destination is not already
  occupied.
- **For signal 1 at `d0001` (the residual record):** a
  base-anchored signal-1 record at `d0001` measures the
  *residual* of the original lean — how much of signal 1's
  destination the landed head has not reached on held-out
  sequences. This record is genuinely new information the
  head-relative semantics could never produce (under
  head-relative anchoring the same record is the closed
  form by construction). §5.7 assigns it its role: a
  published diagnostic, not the decisive run.
- **The already-satisfied case is a premise failure, and
  is handled as one.** If a base-anchored signal's
  pre-training held-out loss at its head measures ~0, or
  training cannot improve it, the existing machinery
  already refuses the candidate (`train_candidate` raises
  when held-out loss did not improve — a non-improving
  candidate is "a failed experiment, not authorship",
  RFC SI-18 §8.4). SI-25 does not soften that refusal, does
  not substitute another signal quietly, and reports the
  measurement as the slice's finding (§10 criterion 5).
  A base-anchored destination the head already occupies
  authors nothing, and saying so is a result.

Two labelling consequences follow, and §12 makes them
binding: the hyperparams field `base_held_out_loss` — kept
under its SI-19 name, already re-interpreted once by
SI-21 — at a base-anchored non-base cycle means "the
pre-training held-out loss of the weights under training
(at their head) against targets derived from the recorded
target source"; the `head` and `target_source_head` fields
recorded beside it are what make that reading unambiguous,
and the as-built doc must state the reading wherever the
figure appears. And the coincidence at base must be stated
wherever the at-base record appears: at head = `base` the
record *is* the closed form under either anchoring,
because the two weight sets are one set there.

### 5.6 Cumulative identity accounting: what can and cannot be argued

The budget, from SI-23's published measurements
(arithmetic on cited numbers, labelled as such): cycle 1
spent identity drift 0.17189218290150166 of tol 0.25,
leaving **headroom 0.07810781709849834** for any second
landed delta — under the accounting that does not change:
identity drift is seed-absolute and cumulative
(`si/gate.py:415`, `_identity_check` compares the
candidate-overlay forward against the suites' recorded
seed logits), capability regression is head-incremental
(`si/gate.py:431` against `ctx.capability_baseline()`,
`si/probes.py:413`), and base-anchoring changes neither
check, neither tolerance, and neither baseline. What
changes is what the second update is *trained to be*, and
that change has structure worth stating precisely:

- *Derivation (a fixed destination bounds the destination's
  drift, not the journey's).* Suppose a base-anchored
  signal-2 cycle at `d0001` trained its composed head to
  reproduce, on the identity probes, exactly the outputs
  that signal 2's at-base training produced at base. Then
  the composed drift would equal signal 2's measured
  at-base drift, **0.16672966815531254** ≤ tol 0.25 — the
  destination's drift replaces the first cycle's drift in
  the accounting instead of adding to it, because the
  gate compares the composed head to the *seed*, and the
  destination is defined from the seed. This is the
  route's whole plausibility argument, and it is a
  statement about a hypothetical exact arrival, labelled
  a derivation.
- *What cannot be argued from that (the limits, stated as
  plainly as the derivation).* Training optimises the
  destination on signal 2's *training sequences*, at their
  last positions — not on the identity probes. The
  composed head's probe outputs are whatever a rank-8
  readout update, trained to move the head from
  `d0001`'s outputs toward the base-anchored targets on
  32 training sequences, does to 8 different inputs. Three
  named gaps stand between the derivation and any
  prediction: (i) the at-base figure 0.16672966815531254
  was measured for an update trained *at base* and applied
  *at base*; the decisive update is trained *at `d0001`*
  against base-derived targets and its folded matrix will
  differ — the destination is the same sentence, the
  matrix that reaches for it is not the same matrix;
  (ii) arrival on training sequences does not bound
  per-probe L-infinity movement (SI-22 §5.6.4's warning,
  unchanged: a Frobenius norm budget is not a per-probe
  movement bound, and the norm budget 0.9 is not a drift
  bound); (iii) the second update must *undo* part of the
  first update's movement on some inputs while *adding*
  the predecessor lean on others, and nothing in the
  signal's definition fixes the sign or size of the
  composed movement at any particular probe. No arrival,
  cancellation, or ADMIT is claimed, estimated, or
  predicted.
- *The capability arm deserves its own warning.* The
  head-incremental capability check passed comfortably in
  both REJECTed cycles (0.1657, 0.1502 against eps 0.25),
  and it is tempting to read identity as the only arm that
  can bind. Signal 2's *at-base* capability regression,
  however, measured **0.2418281203135848** — within eps
  0.25 by less than 0.009. That figure is an at-base,
  against-base-baseline measurement and does not transfer
  to the decisive cycle's head-incremental accounting
  (baseline measured live at `d0001`); it is cited here
  for one narrow reason: it proves signal 2's updates
  *can* move capability-probe outputs by nearly the full
  epsilon, so a base-anchored cycle 2 — whose update must
  travel further in weight space than a head-relative one
  (it corrects the head's existing lean as well as adding
  its own) — cannot be assumed capability-safe. The arm is
  measured, per cycle, as always (§10 criterion 8).
- *Diagnostics pre-registered.* Because the derivation
  cannot decide the outcome, SI-25 must publish, whatever
  the gate's verdict is: (i) the full SI-22 §5.6.4 set,
  recomputed for the base-anchored candidate — per-probe
  drift contributions with worst probes named, the
  alignment diagnostic, the folded updates' inner product
  and norms, composed capability and echo, the signal's
  held-out records; (ii) **the control/treatment table**:
  SI-23's head-relative cycle 2 and SI-25's base-anchored
  cycle 2 side by side — same signal, same head, same
  target, same hyperparameters, differing only in target
  source — per-probe composed drift, worst probe, gate
  numbers, and candidate norms, with measurements and
  derivations labelled and neither run's figures blended
  into the other's (§12); (iii) the base-anchored
  pre-training held-out records of §5.5, for signal 2 at
  `d0001` and the signal-1 residual record at `d0001`;
  and (iv) the alignment diagnostic read against its
  head-relative counterpart: under base-anchoring the
  second update's movement is *expected by the design* to
  oppose the first update's movement on the component
  where `d0001` overshoots signal 2's destination — an
  expectation stated here, before the run, as a
  derivation-level reading of the design, so that a
  measured negative inner product can be reported as the
  mechanism appearing and a measured positive one as the
  mechanism failing, neither as a surprise (§12.14's rule
  applies: the expectation is not a prediction of the
  verdict). A diagnostic that cannot be computed is
  reported "not evaluated" (invariant 5), never silently
  omitted.

The honest bottom line of §5.6: both measured second
cycles asked the readout to step *further from the seed*
and the gate's seed-absolute accounting charged them for
the sum. Base-anchoring is the one named semantics under
which a second cycle's destination is priced, in the
limit of exact arrival, at its own at-base drift instead
of at the sum — and whether training on constructed
sequences arrives anywhere near that limit on the probes
is measurable with instruments that already exist. That
is a plausibility argument for running the experiment —
nothing in this section is a prediction that it ADMITs,
and §10 requires no ADMIT.

### 5.7 The decisive configuration: signal 2, base-anchored, at `d0001`

Two base-anchored cycles at `d0001` are definable, and
the slice must choose which one is decisive — the choice
determines what an ADMIT would prove:

- **Signal 1, base-anchored, at `d0001` (the residual
  cycle).** Asks only for what remains of the original
  lean. Its trained update is by construction the
  *residual* of an already-landed update; if the residual
  is small, an ADMIT would prove that a small residual
  fits the headroom — a weak claim, close kin to the
  smaller-scale gate-fitting trap (the smallness would be
  inherited from cycle 1's arrival, not designed, which
  keeps it honest — but the claim it buys is still
  weak), and if the residual measures ~0 the cycle
  authors nothing at all (§5.5). Its pre-training record,
  however, is valuable regardless (§5.5), and costs one
  training run to obtain.
- **Signal 2, base-anchored, at `d0001` (RECOMMENDED as
  decisive).** Asks for *new content* — a destination the
  head has never been trained toward — under the new
  semantics, against the exact control SI-23 published:
  same signal, same head, same everything except target
  source. An ADMIT here is the first authored two-delta
  stack, bought by a semantics argument rather than by a
  smaller ask; a REJECT here is attributable, per probe,
  against the control (did the composition improve? by
  how much? on which probes did the mechanism appear or
  fail?) — publishable either way, and decisive about the
  route in a way the residual cycle cannot be: if new
  content cannot land base-anchored either, the residual
  cycle's narrow ADMIT would not have rescued the route's
  general claim.

**The design SI-25 must implement:** the decisive run is
SI-23's decisive run with exactly one configuration
changed — one reused `TinyTransformerMind`, cycle 1 signal
1 under the default anchoring, landing `d0001`; the driver
then explicitly reconfigures the same mind to signal 2
*and* to base-anchored targets (both recorded selections,
§5.3); cycle 2 trains at `d0001` under the unchanged
freshness contract and is judged by the unchanged gate.
Two sanity configurations accompany it, labelled as
SI-23 labelled its at-base run: signal 2 base-anchored at
base (must coincide with SI-23's at-base run exactly —
same blob `sha256:131cc242…45f4`, §5.3) and the signal-1
base-anchored residual record at `d0001` (§5.5 — a record
and, if it trains, a judged candidate reported as the
diagnostic it is, never as the slice's decisive claim and
never stacked into the decisive run's bundle).

### 5.8 Gradient obligations under the new semantics

SI-19's exactness rests on targets being *constants*
during optimisation: `build_training_signal` computes
them once, from a weight set, and `analytic_gradients`
differentiates the loss with respect to the LoRA factors
through the readout, over hidden states the readout
cannot change. Base-anchoring preserves that shape — the
targets are still computed once and held fixed; only the
weight set they are computed *from* differs from the set
being trained — so no new derivative is needed. What is
needed, and §10 criterion 3 requires, is the proof
re-run where the loss actually changed: the
finite-difference gradient check for signal 2's
base-anchored loss **at head `d0001`** (base-derived
targets, head hidden states, head readout), at SI-19's
pinned tolerance 1e-5, with the measured agreement
published; and at base, where the anchored loss coincides
with the head-relative loss, the existing SI-23 check
stands but must be re-run under the new code path so the
coincidence is measured, not asserted. A gradient check
run only at base does not satisfy the criterion, for
SI-22 §10 criterion 2's reason carried over: the decisive
configuration is the one that must be checked.

---

## 6. Candidates (d) and (e): new target; additive / multi-target

SI-22 §6 split the remaining widenings into (c2) a new
LoRA target and (c3) additive / multi-target authorship;
they are renumbered (d) and (e) here because §5's
candidate (c) is the anchoring route SI-22 treated inside
its §5.7. SI-23 changed neither candidate's blockers: it
trained a second signal, on the same target, through the
same readout gradient, deliberately. Its registry and
recorded selection settle *which signal* and (after §5)
*which anchoring*; neither is a gradient derivation for a
new tensor, and neither is evidence about authored
multi-target behaviour.

### 6.1 Candidate (d): a new LoRA target (non-readout tensor)

**Verdict up front: DEFERRED.**

- **The gradient is still the work, unchanged.** SI-19's
  exactness rests on one fact, recorded in
  `si/tiny_training.py`'s docstring and relied on again in
  §5.6 above: `lm_head.weight` is the final operation, so
  no gradient with respect to any other tensor is ever
  needed and hidden states can be cached. For any earlier
  tensor — an attention projection, an MLP projection, the
  embeddings — the gradient runs backward through RMSNorm,
  causal softmax attention, and SwiGLU: the derivation RFC
  SI-18 §4.4 called error-prone in exactly those places. It
  ships, if ever, only with a finite-difference gradient
  check per target, at a tolerance pinned from measured
  agreement. None of that exists for any second target,
  and SI-23 built none of it: its second cycle reused
  SI-19's readout gradient verbatim, under a new signal.
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
  choice — but measurability was only one of SI-19's three
  recorded reasons for `lm_head.weight`. Cap fit partially
  transfers; derivability must be re-earned per target,
  with its own recorded reason, as SI-19 recorded the
  readout's.
- **Why (c) goes first.** The anchoring question is
  *asked by measured evidence* — two REJECTs and an
  attributed cause — and is answerable with the existing
  gradient, the existing fold, and the existing
  accounting, so its outcome is attributable to the
  semantics. A new target spends the slice's risk on a
  derivation, answers no question SI-23 posed (a new
  target's update composes through changed hidden states,
  a composition the current diagnostics do not even
  describe), and would confound any later anchoring
  comparison. If SI-25 lands a second delta, a new target
  becomes the natural next widening for a later RFC to
  evaluate — against a landed stack, with the anchoring
  axis settled.

### 6.2 Candidate (e): additive and multi-target authored candidates

**Verdict up front: REJECTED as the next slice** — the
SI-22 §6.2 verdict, carried forward unchanged, because
SI-23 produced no authored multi-target evidence that
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
  the one move §2 invariant 8 and SI-22 §11 forbid as a
  route around a gate outcome. A future slice could argue
  a policy change on evidence; there is still no authored
  multi-target evidence to argue from.
- **Joint training is a new optimisation problem, not two
  copies of SI-19.** Two targets trained against one loss
  interact — each target's gradient depends on the other's
  current factors through the forward — and the
  readout-only caching trick does not survive a second,
  earlier target. Aggregate norm accounting
  (`si/transformer_delta.py:176`, `aggregate_norm`:
  root-sum-square over per-target norms, recomputed by the
  gate) would need the training loop's norm guard
  re-derived for the joint case.
- **Additive authorship's natural homes are still the
  worst targets.** Additive targets must be 1-D
  (`si/transformer_delta.py::validate_transformer_delta`);
  the miniature's 1-D tensors are norms and biases —
  including the probe-blind `k_proj.bias` pair, an
  authored delta on which could never be judged.
- **Anchoring does not rescue it.** Base-anchored targets
  (§5) are defined per signal over the readout; nothing in
  the anchoring design speaks to joint or additive
  training, and citing §5 toward (e) would be the
  unlabelled transfer this RFC's labelling rules exist to
  prevent.

### 6.3 Verdicts on (d) and (e)

| Sub-direction | Verdict | One-line reason |
|---|---|---|
| (d) New LoRA target | DEFERRED | Needs a full backward derivation plus a per-target gradient check; only measurability transfers from SI-19, and no measured evidence asks its question yet (§6.1). |
| (e) Additive / multi-target | REJECTED as next slice | Requires loosening `max_targets_per_delta` / `allow_additive` policy defaults with zero authored-behaviour evidence (§6.2). |

---

## 7. Candidate (f): transformer squash; candidate (g): Glimmer probing

### 7.1 Candidate (f): transformer squash for miniature deltas

**Verdict up front: DEFERRED** — its precondition remains
unmet, and SI-23 is now the second slice to confirm that
by measurement rather than by absence.

SI-20 §6.4 made squash schedulable only after a slice
produced an authored stack and the measurements a squash
design must respect. SI-21 and SI-23 both produced the
measurements (per-delta drift contributions, the §4.4
confirmation, the §5.6.4 diagnostics) but **not the
stack**: both second cycles REJECTED, so the deepest
landed authored stack on record is still depth 1
(`d0001`), and **no authored two-delta stack exists**.
SI-21's two-step unwind was exercised on authored `d0001`
plus a benign hand-built **fixture** `d0002` — journalled
as a fixture, never described as authored (§1). Squashing
that fixture stack would prove format handling, not
authored composition: exactly the designing-against-
fixtures outcome SI-12 refused.

What SI-19 and SI-20 established still holds: for authored
same-target deltas, composition is exact algebra — every
authored delta is a single-target LoRA on `lm_head.weight`
at rank ≤ 8, two such updates compose to a single update
of rank ≤ 16 on one named tensor, foldable by the existing
`si/apply.py::fold_target` arithmetic, with the
recompression to rank ≤ 8 lossy and therefore gated by a
tolerance verify of the SI-3 kind plus the miniature's
probes (SI-20 §6.2). What still blocks a squash slice,
beyond the missing stack:

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
   `hyperparams` of its own — and after SI-25, its
   constituents may carry *different anchorings*, so §12
   forbids dressing it in its constituents' training
   numbers or either anchoring's labels. Its journal story
   — compression of named deltas, tolerance measured,
   probes re-run — must be statable without borrowing
   authorship's labels (SI-20 §6.3).
3. **The refusal stands meanwhile.** SI-12's fail-closed
   refusal is unchanged in code:
   `si/squash.py::squash_bundle` raises `SquashError`
   before any write if any record on the resolved active
   chain has kind `transformer-delta-v1`. §11 keeps it in
   force throughout SI-25.

If SI-25's second cycle ADMITs, the precondition is met
for the first time — an authored two-delta stack will
exist — and squash becomes a legitimate candidate for a
*later* RFC to evaluate against that stack's measured
norms and drift contributions. Meeting the precondition
does not schedule the slice, and SI-25 builds no part of
it (§11).

### 7.2 Candidate (g): probing Glimmer — REJECTED, not reopened

Glimmer probing was rejected by RFC SI-13 §7/§9, restated
by SI-18 §5, SI-20 §7, and SI-22 §7.2, and is **not
reopened here**. SI-23 changes nothing about it, and
neither does this RFC's recommendation: authorship, like
probing before it, is miniature-only by construction —
the training assembles weights through the miniature
executor (`validate_architecture`,
`si/tiny_transformer.py:146`, refuses Glimmer's
descriptor), the signals are defined over a vocabulary of
32, the gate's probe binding cannot be stated for the
Glimmer bundle (§3), and the base-assembly design of §5.4
is a 30-tensor miniature mechanism whose cost and
verification shape say nothing about a 1,436-tensor
by-reference base.

The restated facts, so no future reader has to excavate
them: there is no Glimmer forward path, and a partial
forward over a partial stack would be a claim about that
partial computation, not about the model — fabrication,
as SI-18 §5 recorded. The Glimmer seed's deltas still
escalate with checks 3/4 "not evaluated"; its stored bytes
are still 0 against 59,553,253,376 logical bytes
(SI-9 §0); and one full verification of that by-reference
base still stands measured at ~29 minutes (1,756.9 s)
before any forward compute (SI-9 §0/§7.1). Any future
Glimmer proposal needs its own RFC and its own
feasibility measurements; it gets no momentum from this
document.

---

## 8. Decision summary

| Candidate | Verdict | One-line reason |
|---|---|---|
| (a) Small real safetensors base | **DEFERRED** | SI-18 §3.7 audit: A-1 to A-4 all still unmet, unchanged by SI-23; naming a real base needs a download this scope forbids (§3). |
| (b) Second training signal (miniature) | **DONE — as SI-23** | Built and measured (PR #28): signal 2 authored at base (ADMIT) and at `d0001` (REJECT, attributed to movement alignment) (§4). |
| (c) Base-anchored targets | **RECOMMENDED — SI-25** | SI-22 §5.7's four deferral reasons are retired or discharged by SI-23's published diagnostics; the route changes exactly one variable against SI-23's control run, and no ADMIT is required (§5). |
| (d) New LoRA target | **DEFERRED** | Needs a full backward derivation plus a per-target finite-difference gradient check; only measurability transfers from SI-19 (§6.1). |
| (e) Additive / multi-target | **REJECTED as next slice** | Requires loosening `max_targets_per_delta` / `allow_additive` policy defaults with zero authored-behaviour evidence (§6.2). |
| (f) Transformer squash (miniature) | **DEFERRED** | Precondition still unmet — no authored two-delta stack exists; SI-21's unwind used a labelled fixture `d0002` — and the squash gate context is still demo-shaped (§7.1). |
| (g) Glimmer probing | **REJECTED** | Rejected at SI-13 §7/§9, restated at SI-18 §5, SI-20 §7, and SI-22 §7.2; SI-23 changes nothing (§7.2). Not reopened. |

The two routes SI-21 named to a landed second write have
now both been run or designed to the point of running:
the new-signal route is (b), done, its outcome measured
and attributed; the new-semantics route is (c),
recommended, its design discharged against the diagnostics
(b) produced. No third route is proposed by this RFC, and
§5.1 records the two unnamed routes (a third head-relative
signal; a reduced correction scale) and why they are not
candidates.

---

## 9. The recommended slice: SI-25 (outline)

**SI-25 — Base-anchored targets for the miniature: signal 2
trained at the landed head against the frozen base's
targets.**

In scope:

1. **Anchoring as a recorded configuration** (§5.3):
   `TinyTransformerMind` is configured with the anchoring
   it will train under (constructor-level named choice,
   defaulting to `"weights-under-training"`), settable
   between cycles by an explicit driver call in the shape
   of `set_training_signal(...)`; the value is validated
   against exactly the two named anchorings and any other
   value refuses with the existing discipline; the intent
   carries it; `train_candidate` dispatches on it. No
   autonomous, learned, depth-driven, or verdict-driven
   anchoring selection exists in the slice.
2. **Target-source assembly** (§5.4): the base weight set
   is assembled from the cycle's verified manifest with
   its active chain truncated to empty, in exactly one
   place beside `assemble_weights`, through the existing
   hash-verified apply-at-load reads; the verified
   manifest is never mutated; a failed base assembly
   refuses training (no fallback to head-derived targets).
   Under `weights-under-training` the target source is the
   weights under training, exactly as today.
3. **Signal 2 base-anchored at base: the coincidence
   configuration** (§5.7): configured with signal 2 and
   base-anchoring, a first cycle at base records signal
   2's pre-training held-out loss at base (the closed form
   0.00140625 under either anchoring there, confirmed by
   measurement) and trains and is judged there. It must
   reproduce SI-23's at-base run exactly — candidate blob
   `sha256:131cc242…45f4`, same gate numbers — proving the
   base assembly and the split target-source path correct
   where the two semantics coincide, before the decisive
   cycle is trusted. Its outcomes are labelled at-base
   facts; they are not the slice's decisive claim.
4. **Echo-trap records under the new semantics** (§5.5):
   signal 2's pre-training held-out loss at `d0001` under
   base-anchoring is recorded in the cycle's hyperparams
   (head and target source recorded beside it) and
   published as the genuine measurement it is — no closed
   form quoted for it; and the signal-1 base-anchored
   residual record at `d0001` is measured and published as
   a diagnostic. If a record measures ~0 or training
   cannot improve on it, the slice reports the premise
   failure (§5.5) and the existing non-improvement refusal
   stands.
5. **The gradient check for the base-anchored loss**
   (§5.8): the finite-difference gradient check is re-run
   for signal 2's base-anchored loss at head `d0001` (the
   decisive configuration) and at base under the new code
   path; the as-built doc publishes the measured agreement
   and the tolerance (1e-5, unchanged) for both.
6. **The decisive two-cycle run** (§5.7): one reused
   `TinyTransformerMind` driven through
   `si.run.run_cycles` — cycle 1 configured with signal 1
   under the default anchoring, landing `d0001`; cycle 2
   configured with signal 2 *and* base-anchored targets by
   explicit recorded selections, trained against `d0001`
   under the unchanged freshness contract, and judged by
   the unchanged gate. The verdict — ADMIT, REJECT, or
   ESCALATE — is reported as measured.
7. **The §5.6 diagnostics, published** whatever the
   verdict: the SI-22 §5.6.4 set recomputed for the
   base-anchored candidate; the control/treatment table
   against SI-23's head-relative cycle 2; the §5.5
   records; the alignment diagnostic read against its
   head-relative counterpart — each labelled measurement
   or derivation, each "not evaluated" if it cannot be
   computed.
8. **Per-cycle published measurements**: gate identity
   drift (seed-absolute, cumulative), capability
   regression (head-incremental, labelled as such), echo
   accuracy per cycle; held-out numbers per cycle labelled
   with their signal id, head, target source, and
   anchoring; §6.6 VERIFY outcomes; stored-bytes growth
   per cycle; and per-cycle CPU cost.
9. **Unwind, honestly conditioned**: if cycle 2 ADMITs,
   rollback of the authored two-delta stack to `d0001`
   and to base is exercised through the existing rollback
   core, with each restored head's probe state reported
   as measured. If cycle 2 does not ADMIT, no authored
   two-delta stack exists, nothing is unwound, and the
   as-built doc says so — SI-21's fixture-proven unwind
   remains the only unwind evidence, labelled as such.
10. **An as-built doc** in the SI-14…SI-23 pattern,
    publishing every number §10 requires and stating
    plainly what is still not claimed.

Explicitly not in SI-25's scope: §11's non-goals.

---

## 10. Acceptance criteria for SI-25

A reviewer must be able to check each criterion against a
test or a published number. SI-25 does not merge on any
criterion being waived, tuned quietly, or reported without
its measurement.

**What success and failure mean for this slice, stated
before the criteria.** SI-25 *succeeds as a slice* if a
base-anchored second cycle was genuinely authored against
the landed head under the semantics of §5.2, genuinely
judged by the unchanged gate, and its outcome — ADMIT,
REJECT, or ESCALATE — attributed by the §5.6 diagnostics
against SI-23's control run. **No criterion requires an
ADMIT.** SI-25 *fails as a slice* if any criterion below
fails — in particular if the anchoring in a cycle's record
cannot be read back from its own journal (criterion 1),
if cycle 1 or the coincidence configuration does not
reproduce its published blob exactly (criteria 2 and 9),
if the base assembly can fall back silently to
head-derived targets (criterion 4), or if a number is
published without its anchoring (criterion 8, §12). A gate
REJECT with all criteria met is a *successful slice with a
negative experimental result*, in the SI-21/SI-23 pattern;
a gate ADMIT with any criterion waived is a failed slice.

1. **Recorded anchoring.** The configuration serves
   exactly the two named anchorings
   (`"weights-under-training"`, `"base-anchored"`) and
   refuses every other value with the existing refusal
   discipline. Selection is explicit and recorded: a test
   asserts, per cycle on both verdict paths, that the
   journaled `training-hyperparams` note's
   `target_anchoring` and `target_source_head` equal the
   anchoring and target source the driver configured for
   that cycle — including the decisive cycle's
   `head = d0001` / `target_source_head = base`
   disagreement. No autonomous, learned, depth-driven, or
   verdict-driven anchoring selection exists in the
   slice, and a cycle whose anchoring cannot be read back
   from its own journal has failed this criterion whatever
   the gate said.
2. **Coincidence at base.** Configured with signal 2 and
   base-anchoring, a first cycle at base reproduces
   SI-23's at-base run exactly — candidate blob
   `sha256:131cc242728024b4f833bc78ea627825f1e8616972fc2670603673e1814645f4`,
   identity drift 0.16672966815531254, capability
   regression 0.2418281203135848 — and configured with
   signal 1 under either anchoring, a first cycle
   reproduces SI-19/SI-21/SI-23's `d0001` exactly
   (criterion 9). A mismatch at base invalidates the
   decisive run's attribution and fails the slice until
   explained as a finding, not tuned away.
3. **Gradient check for the base-anchored loss, at the
   decisive head.** The finite-difference gradient check
   is re-run for signal 2's base-anchored loss at head
   `d0001` and at base under the new code path; the
   as-built doc publishes the measured agreement and the
   tolerance it was checked against (1e-5, SI-19's pin,
   unchanged) for both. A check run only at base does not
   satisfy this criterion.
4. **Target-source assembly, fail-closed and
   byte-honest.** A test tells the two assemblies apart
   (base view's readout equals the recorded base tensor;
   head view's readout equals base + `d0001`'s fold); the
   verified manifest is never mutated by the base view
   (asserted); a tampered or unassemblable base refuses
   training with no fallback to head-derived targets
   (asserted); blob-store growth per cycle is attributable
   only to a landed candidate (asserted, as in
   SI-21/SI-23).
5. **Echo-trap records under the new semantics.** Signal
   2's pre-training held-out loss at `d0001` under
   base-anchoring is recorded and published, labelled with
   signal id, head, target source, and anchoring, with no
   closed form quoted for it; the signal-1 base-anchored
   residual record at `d0001` is likewise published as a
   diagnostic. If either record measures ~0 or its
   training cannot improve on it, the existing
   non-improvement refusal stands, no substitute signal or
   configuration is quietly run as the decisive cycle,
   and the measurement is reported as the slice's finding.
6. **Determinism per cycle.** Each cycle's candidate is
   byte-identical across repeat runs of that cycle
   (canonicalised serialisation, §12), and a full
   decisive run repeated from the seed reproduces both
   cycles' records and blob hashes exactly, except any
   explicitly labelled wall-time field.
7. **Gate honesty per cycle.** Each cycle's journaled
   checks 3/4 equal an independent `ProbeContext`
   evaluation of the same candidate against the same
   pre-cycle head (the SI-19 §8.8 equality, per cycle).
   Cycle 2's verdict — ADMIT, REJECT, or ESCALATE — is
   reported as measured. **No criterion requires an
   ADMIT**: SI-25 succeeds by §5's claim — a base-anchored
   candidate was genuinely authored against the landed
   head under the new semantics and genuinely judged —
   whatever the verdict is.
8. **Cumulative accounting, diagnostics, and the
   control/treatment table published.** The as-built doc
   publishes, per cycle: identity drift (seed-absolute),
   capability regression (head-incremental, labelled as
   such), echo accuracy vs the recorded floor, and the
   landed head's cumulative identity drift after each
   cycle — plus the full §5.6 diagnostic set (per-probe
   contributions and worst probes, alignment, folded-
   update inner product and norms), the §5.5 records, and
   the control/treatment table against SI-23's
   head-relative cycle 2 with the two runs' figures kept
   distinct and labelled by anchoring. Any uncomputable
   diagnostic is reported "not evaluated". No incremental
   figure is presented as cumulative or vice versa, and
   no base-anchored figure is presented as a head-relative
   one (§12).
9. **First-cycle reproduction.** Configured with signal 1
   (the default signal, default anchoring), a first cycle
   reproduces SI-19/SI-21/SI-23 exactly — same candidate
   blob
   (`sha256:2761e8ffc139ce3e107c11da035e8a73c20792945a494f4e493dccf418b6ca9e`),
   same gate numbers — so the anchoring machinery is
   proven behaviour-preserving for the existing semantics
   before the new semantics' results are trusted (§5.3).
10. **VERIFY, unwind, and untouched scope.** An ADMITted
    cycle 2 passes (or fails and rolls back from) the
    §6.6 probe VERIFY exactly as cycle 1 did, with the
    rollback target being `d0001`; unwind follows §9 item
    9's honest conditioning (authored stack unwound and
    re-measured if one exists; no unwind claimed if none
    does). Exactly two authored cycles are attempted in
    the decisive run; `max_stack_depth`, both tolerances,
    all caps, and every policy default are unchanged;
    both signals' ids, streams, corrections, and
    partitions are unchanged; `si squash` still fails
    closed on transformer stacks (SI-12 tests green); the
    full existing suite is green; the demo path is
    unchanged; `si serve` still returns 501 for every
    transformer bundle; no CLI wiring is added. Per-cycle
    `train_delta` wall time, total two-cycle wall time,
    and the suite's resulting runtime on this VM are
    measured and published.

---

## 11. Non-goals

Explicit, and binding on SI-25 and on any slice this RFC
is cited to justify:

- **No Glimmer anything.** No Glimmer forward path,
  probes, training, or authorship (§7.2). No candidate in
  this RFC touches the Glimmer seed except to restate
  that it is untouched, and §5.4's base assembly is a
  miniature mechanism that claims nothing at Glimmer
  scale.
- **No real-base work.** Candidate (a) is deferred with
  audited preconditions (§3); SI-25 builds nothing toward
  it, and no download is performed or scheduled.
- **No signal redefinition.** Both signals' ids, streams,
  corrections (scale 0.15 included), partitions, target,
  rank, and alpha are byte-for-byte as SI-19/SI-23 defined
  them. Base-anchoring changes which weight set a signal's
  targets are derived from; it does not change what either
  signal *is*, and no third signal is introduced.
- **No gate-fitting variant.** No reduced correction
  scale, no signal or anchoring chosen backward from the
  identity tolerance (§5.1), and no anchoring selection
  rule that a gate verdict can influence (§5.3).
- **No other widening.** No new LoRA target (§6.1), no
  additive or multi-target authored candidates (§6.2),
  and no policy default changes — in particular
  `max_targets_per_delta` stays 1 and `allow_additive`
  stays false.
- **No autonomous selection — of signal or of anchoring.**
  The mind chooses neither; both are explicit, recorded,
  driver-level configurations (§5.3), and no
  perception-driven, depth-driven, alternating, or
  verdict-driven rule is built or implied for either axis.
- **No transformer squash.** SI-12's fail-closed refusal
  stands throughout SI-25 (§7.1); §7.1 is a deferral
  analysis, not an implementation brief — even if SI-25
  creates the authored stack squash has been waiting for.
- **No serving, no CLI wiring.** Transformer bundles keep
  returning 501; `si cycle`/`si run` keep their demo/mock
  minds. SI-25 is a library-and-tests slice like SI-19,
  SI-21, and SI-23.
- **No tolerance loosening, no probe-suite changes.** If
  the base-anchored cycle is REJECTed, the answer is a
  later RFC's better argument — never a wider gate, and
  never a quieter report.
- **No third cycle, no continual-learning claim.** Exactly
  two authored cycles are attempted in the decisive run,
  on constructed signals, on a constructed model, under
  each anchoring at most as §9 scopes. Two anchorings at
  depth ≤ 2 prove something about the loop's machinery
  and its gate — nothing about learning at any other scale
  or on any real task.
- **No mixed-anchoring stack claim.** The decisive run's
  cycle 1 is head-relative-at-base only in the degenerate
  sense that the anchorings coincide there; SI-25 does
  not train a head-relative cycle at a non-base head
  inside the decisive run, and no claim is made about
  stacks whose cycles mix anchorings at non-base heads.

---

## 12. Honesty and labelling requirements for any future slice

Binding on SI-25 and on any later slice citing this RFC.
Items 1–14 carry RFC SI-22 §12 forward unchanged in force;
items 15–17 are new, from this RFC's findings.

1. **Training numbers are training numbers.** Losses,
   residuals, and held-out signal scores must be labelled
   with the signal id and partition wherever they appear.
   They are never a gate result, never a probe result, and
   never evidence about any base but the miniature.
2. **Measured vs not evaluated survives authorship.** An
   authored candidate's checks 3/4 report exactly what the
   probes measured (`measured` a real float) or that they
   were not evaluated (`measured=None`) — the SI-17
   labelling table applies unchanged, including the
   blind-escalated state, at every cycle, for every
   signal, under every anchoring.
3. **Authorship is not admission.** No document, journal
   entry, or report may describe a trained candidate as
   "learned into the bundle", "accepted", or "verified"
   before the gate and (on ADMIT) the §6.6 probe VERIFY
   have actually run and been journaled — per cycle, per
   signal, per anchoring.
4. **Every number carries its source.** A figure in an
   as-built doc is either reproduced from a test/run in
   that slice (with the test named) or cited to the
   earlier doc that measured it. Estimates are labelled
   estimates; derivations from cited facts are labelled
   derivations (as §5.5 and §5.6 label theirs).
5. **The miniature's labels travel.** "Constructed, not
   trained" describes the *base*; deltas may honestly be
   described as "trained on a constructed signal" — and
   nothing in that phrase may be allowed to blur into a
   claim that the base is a trained model or that either
   signal is a real task.
6. **Failure states are reportable successes of honesty.**
   A base-anchored REJECT, a derivation refuted by its
   test, a premise failure under §5.5, a VERIFY rollback
   to `d0001` — each is a publishable, journalable outcome.
   A slice that reports only its passes has failed this
   section regardless of its code.
7. **Serialisation determinism is proven, not assumed.**
   Any code that produces a content-addressed blob —
   including a base-anchored candidate — must prove
   byte-identity across repeat constructions by test
   (SI-19's canonical-header lesson, carried from SI-22
   §12.7).
8. **Head-relative numbers name their head.** Any loss,
   baseline, or score computed against applied weights
   must be recorded with the head (and stack depth) it was
   computed at. Under two anchorings the trap SI-22 §12.8
   named doubles: the same signal's pre-training loss is a
   closed-form artifact at one head under one anchoring
   and a genuine measurement under the other (§5.5), and
   only its full label — signal, head, target source,
   anchoring — says which.
9. **Incremental and cumulative are different claims.**
   Capability regression in this runtime is measured
   against the pre-cycle head; identity drift against the
   seed. Reports must say which they are publishing, per
   number. Summing incremental regressions, or quoting the
   latest cycle's drift as the head's total drift, is
   fabrication by arithmetic.
10. **Composed artifacts do not inherit training labels.**
    A future squashed or otherwise composed delta (§7.1)
    is not "trained": it must be journaled and described
    as a compression with its own measured tolerance and
    probe results, never with its constituents' hyperparams
    presented as its own — and, after SI-25, never with
    either constituent's anchoring presented as its own.
11. **Signal identity is part of every label.** A training
    or gate number recorded without its signal id is an
    incomplete record. Either signal's numbers are never
    presented as the other's, as an unnamed "second
    cycle's", or as generic authorship numbers; the two
    signals' held-out figures are never averaged, summed,
    or otherwise blended into a single claim.
12. **Diagnostics are not verdicts.** The §5.6 diagnostics
    — per-probe contributions, alignment, folded-update
    inner products — are published to *attribute* a gate
    outcome after the gate has spoken. They are not gate
    checks, they influence no verdict, and no diagnostic
    may be quoted before a run as a predicted ADMIT or
    REJECT.
13. **Recorded selection is not a choice the mind made.**
    Signal selection (SI-22 §12.13) and anchoring selection
    (§5.3) are explicit configurations recorded in
    provenance. No document may describe either as the
    mind "choosing", "deciding", or "preferring"; the
    honest sentence is that the driver configured them and
    the journal shows it.
14. **A derivation is not a prediction.** §5.6's
    fixed-destination derivation and §5.5's statements
    about what the new records contain are labelled
    derivations from cited facts. Neither may be cited —
    in an as-built doc, a PR body, or a report — as
    evidence that a base-anchored cycle will ADMIT, will
    REJECT, or has any measured property at all. Only
    SI-25's runs measure those. The §5.6 expectation about
    the alignment diagnostic's sign is covered by this
    item in full: it is a reading of the design stated in
    advance so the measurement can be interpreted, never
    a forecast of the verdict.
15. **Anchoring is part of every label (new).** A training
    or gate number recorded without its anchoring
    (`target_anchoring` and `target_source_head`) is an
    incomplete record, exactly as a number without its
    signal id is (item 11). A base-anchored figure is
    never presented as a head-relative figure or vice
    versa, and the at-base coincidence of the two
    anchorings is stated wherever an at-base figure is
    used to compare them.
16. **Control and treatment are never blended (new).**
    SI-23's head-relative cycle 2 is SI-25's control run.
    The control/treatment table (§5.6) publishes the two
    runs side by side with each figure under its own
    run's labels; no figure is averaged, differenced into
    a single headline without its per-probe table, or
    quoted without the anchoring that produced it. A
    difference between the runs is attributable to the
    anchoring only because everything else was held fixed
    — any accidental difference (a changed hyper
    parameter, stream, or partition) voids the attribution
    and must be reported as voiding it.
17. **The residual record is a diagnostic (new).** The
    signal-1 base-anchored record at `d0001` (§5.5)
    measures a residual toward a destination a landed
    delta already approached. It is never presented as a
    second cycle of signal 1, as evidence that signal 1
    can be stacked, or as the decisive run's result; if
    its candidate is judged, its verdict is reported under
    its own labels as a diagnostic outcome.

---

## 13. Risks and open questions

Each item names what would retire it. Numbers this RFC
deliberately does **not** supply are listed as open
questions to measure — supplying them here would be
inventing them.

1. **The base-anchored gate outcome is genuinely
   unknown.** Signal 2's base-anchored identity drift,
   capability regression, and echo accuracy at head
   `d0001` are **open questions to measure**. The identity
   budget's remaining headroom after cycle 1 is
   0.07810781709849834 under the unchanged accounting, the
   §5.6 derivation prices *exact arrival* at the
   destination's at-base drift (0.16672966815531254) and
   nothing between, and a REJECT is a live possibility the
   slice must be willing to publish (§10.7, §12.6).
   *Retired by:* SI-25's decisive run.
2. **The gate-fitting trap, in its anchoring costume.**
   §5.1 disposed of the reduced-scale variant; the same
   trap reappears if base-anchoring is *adopted as the
   runtime default* on the strength of one favourable
   run, or if a future slice shops anchorings per cycle
   for whichever passes. This RFC's defences are
   structural: the default stays
   `"weights-under-training"` (§5.3), selection is
   recorded and verdict-blind (§5.3, §12.13), and §11
   makes any verdict-influenced selection a non-goal.
   *Retired by:* nothing once and for all — it is retired
   per slice, by each future RFC naming its semantics'
   design reasons before its results, as §5.1–§5.2 do.
3. **The destination argument may fail in the
   measurement.** §5.6 is explicit that exact arrival is a
   hypothetical: training optimises training sequences,
   the gate judges probes, and the decisive update must
   simultaneously undo part of Δ1 and add a lean of its
   own through one rank-8 matrix. The composed drift may
   land near SI-23's figure anyway, and the capability arm
   — warned about in §5.6 on the strength of signal 2's
   at-base regression 0.2418281203135848 — may bind where
   identity did not. Either outcome is a finding: the
   first would say the binding constraint is probe-space
   movement per se, not target semantics; the second
   would reprice which arm a semantics change must argue
   against. *Retired by:* SI-25's §5.6 diagnostics,
   published either way (§10.8).
4. **A mislabelled anchoring is worse than a wrong one.**
   The slice's worst failure mode is not a REJECT; it is a
   candidate trained head-relative and journaled
   base-anchored (a silent fallback, a stale target-source
   assembly, a configuration dropped between `propose` and
   `train_delta`) — every downstream comparison would then
   attribute a semantics effect to a run that never ran
   the semantics. *Retired by:* §10 criteria 1 and 4 —
   per-cycle journal assertions on both verdict paths, a
   test that tells the assemblies apart, and the
   fail-closed refusal of any fallback.
5. **The split weight-set path could silently alter
   signal behaviour under the old anchoring.**
   Generalising `train_candidate` from one weight set to
   two is the kind of change that passes its own new tests
   while moving an old behaviour in its last digits.
   *Retired by:* §10 criteria 2 and 9 — exact reproduction
   of *both* published first-cycle blobs (signal 1's
   `d0001` and signal 2's at-base `d0001`) under the
   refactored path, before the decisive run is trusted.
6. **Weights-under-training may still be the right
   default for other heads.** §5 recommends base-anchoring
   for one decisive configuration; it does not claim the
   anchoring is right in general, at other depths, or for
   signal 1 at `d0001` as a decisive matter (§5.7). The
   default is unchanged (§5.3) precisely so the question
   stays empirical. *Retired by:* SI-25's published
   per-cycle numbers and diagnostics, then a later RFC's
   judgement — not by this RFC's preference.
7. **Probe blindness still bounds the judge.** RFC SI-13
   §10.1 stands, now across signals and anchorings: 24
   inputs, 28/30 tensors measurable, blind set derived and
   guarded. Both anchorings' candidates target the
   measurable readout, so both are judged — but a change
   of training semantics does not widen the probes' view,
   and a landed two-delta stack under either semantics is
   *evidence* of continuity sampled the same way each
   cycle, not a proof of it.
8. **Depth pressure arrives if SI-25 succeeds.** At
   `max_stack_depth` 8 with transformer squash still
   deferred (§7.1), a landed second authored delta makes
   the stack depth 2 and the compression question real for
   the first time — while the squash gate context remains
   unbuilt, and the stack's two deltas would carry the
   same anchoring question into any squash design (§7.1
   item 2). That is a deliberate fail-closed posture,
   recorded so the next RFC meets it as a design problem
   with a real stack to measure, not a surprise.
9. **Candidate (a)'s unknowns remain unknown.** A-1 to
   A-4 are audited unmet (§3); nothing in SI-25 or this
   RFC scopes them, and the download question is an
   operator decision this RFC explicitly does not take.
   Likewise §7.2: success on the miniature renews the
   "why not Glimmer" pressure, and the answer does not
   soften because the miniature's semantics have widened
   from one anchoring to two.

---

## 14. Summary

After SI-23, SI can originate a miniature delta, judge it,
land it, and originate a second one on top of it under a
second signal — and the gate's cumulative, seed-absolute
identity accounting has been measured refusing both second
writes, the second with its cause attributed per probe:
the second update's movement *aligned* with the first's,
because a head-relative target asks for a further step
from the head and the further step adds. Of the candidate
frontiers this RFC evaluated against that evidence: a
small real base stays deferred, its four SI-18
preconditions audited and all still unmet, with a
checkpoint download flagged as an operator decision rather
than a step (§3); a second signal is done, its question
answered by SI-23 (§4); a new target stays deferred behind
a backward derivation nobody has written (§6.1), and
additive / multi-target authorship stays rejected as a
next slice because it would be bought with a policy
loosening (§6.2); transformer squash stays deferred, its
precondition — an authored two-delta stack — still unmet
after two measured attempts (§7.1); Glimmer probing stays
rejected, not reopened (§7.2); and **base-anchored
targets — recorded as a per-cycle configuration beside
recorded signal selection, target source assembled from
the verified manifest's frozen base with no bytes
duplicated, echo-trap records re-derived for semantics
under which they are genuine measurements again, and a
decisive run that changes exactly one variable against
SI-23's published control — are recommended as SI-25**
(§§5, 9, 10). Their plausibility argument is stated with
its limits in the same section that makes it: a fixed
destination is priced, in the limit of exact arrival, at
its own at-base drift rather than at the sum of two steps,
and no ADMIT is predicted or required. The base stays
frozen, the bytes are never duplicated, no tolerance
moves, and nothing is claimed that a test in this repo
cannot check.

---

*RFC SI-24 — design only. Implementation begins, if
approved, at SI-25 as outlined in §9. RFC SI-22 taught
the loop a second thing to say and the gate told the
truth about it, per probe; this RFC proposes changing
not what the loop says, but what it aims at — a fixed
destination instead of a further step — under the same
judge, at the same scale, with the verdict still not
ours to pick.*
