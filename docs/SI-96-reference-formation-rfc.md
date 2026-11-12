# SI-96 — Reference Formation: What §9.3 Cannot Currently
# Express (RFC)

## The honest headline

SI-95 established what SI-94's dominant failure class is made of: six stage-4 failures,
every one a `depends_on` entry citing something other than an already-journaled tool intent
of its task — in a cycle-0 capture fold that contains **no tool-intent rows at all**, so no
`depends_on` entry in that corpus could have verified under §9.3, and the only reference
form that passed was no reference (all 20 parse-ok bodies are single-intent with an empty
`depends_on`). This RFC frames the problem that follows — a proposer planning dependent
steps inside one proposal document has, under the current contract, **no form in which to
say so** — and lays out the three option families that bear on it: **(A)** a perception
assembly and rendering intervention, inside the one D-F1 class the lead opened in SI-91 §5;
**(B)** a protocol amendment adding an intra-document dependency form to §9; **(C)**
measurement changes (fold-supplied scoring; multi-cycle capture). Each option is analysed —
what it can fix, what it cannot, what a re-measurement under it would price, what stays
UNMEASURED — and the resulting decisions are posed to the lead as either/or questions (§9).
**This RFC recommends none of them.** SI-94's verdict — **BELOW THRESHOLD** under D-C1 —
stands exactly as recorded; no option here re-grades SI-94, and none is framed as doing so.

## Table of contents

- [0. Honesty table — no row changes](#0-honesty-table--no-row-changes)
- [1. Fingerprint](#1-fingerprint)
- [2. The problem, stated exactly](#2-the-problem-stated-exactly)
- [3. The contract as it stands](#3-the-contract-as-it-stands)
- [4. Option A — perception assembly and rendering](#4-option-a--perception-assembly-and-rendering)
- [5. Option B — protocol amendment: an intra-document dependency form](#5-option-b--protocol-amendment-an-intra-document-dependency-form)
- [6. Option C — measurement changes](#6-option-c--measurement-changes)
- [7. Interactions between options](#7-interactions-between-options)
- [8. What stays UNMEASURED under every option](#8-what-stays-unmeasured-under-every-option)
- [9. Decisions posed to the lead](#9-decisions-posed-to-the-lead)
- [10. What this RFC does not do](#10-what-this-rfc-does-not-do)

## 0. Honesty table — no row changes

| Row | State, unchanged by this slice |
|---|---|
| SI-63 §18.3 claim | **NOT earned** |
| SI-63 §18.1 rows P5-R1, P5-R9 | **UNMEASURED** |
| SI-63 §18.1 rows P5-R2–R8, R10 | As promoted, scoped, by SI-74; cited, not re-derived |
| Live model-authored tool intent | **OCCURRED, scoped to SI-74 only** (2026-11-02, P-5 live run only). This slice extends that scope by nothing: it makes no live attempt of any kind |
| SI-94 verdict (D-C1) | **BELOW THRESHOLD** — stands exactly as recorded in `docs/SI-94-c2-conformance-record.md`; not re-cut, re-scored, or re-based here |
| SI-92 record | **INSUFFICIENT** at 27/30 — stands; a serving-configuration artifact per SI-93, not a candidate result |
| SI-79 verdict (v12) | **BELOW THRESHOLD** at 10/30 — stands |
| C-4 for v13 | **Does not proceed** on the SI-94 record |
| L-3 residency-loss drill | **LOCKED** |
| A-2 / A-3 / A-4 | **UNAPPROVED** |
| Readiness unlocks | `[]` |
| D-F1 classes opened | Perception assembly and rendering only (SI-91 §5). §4.2 catalogue rendering, §4.3 standing instructions, §4.4 output-contract instruction, §4.5 gateway-request assembly: **NOT opened** |

The served model is a specialised machine learning model. This slice is machine learning
infrastructure throughout: a design document, written offline, that ran no parser, no
harness, no capture, and no live call. Its claims about code are read from source at the
fingerprint commit (§1) and cite functions and spec sections, not line numbers.

## 1. Fingerprint

| Field | Value |
|---|---|
| SI commit | `55f638ccdcfa2f952cfb7e8239e34c5568d656d2` (main, PR #103 / SI-95 merged) |
| Inputs | `docs/SI-95-failure-taxonomy.md`, `docs/SI-95-taxonomy.json`, `docs/SI-94-c2-conformance-record.md`, `docs/SI-91-decision-record-and-v13-conformance-plan.md` (§5/§6), `docs/SI-82-f2-admissibility-question.md` (§4), `docs/SI-63-gaia-propose-hearth-proposer-rfc.md` (§6/§8/§9) — all on main, read only |
| Code read at the same commit | `si/gaia/proposer.py` (`assemble_perception`, `render_perception_text`, `build_gateway_request`, `parse_proposal`, `_task_intent_seqs`), `si/gaia/conformance.py` (`score_corpus`), `si/gaia/queue.py` (`queue_summary`, the Q-1/Q-2 fold) |
| Candidate under discussion | Imagine v13, corrected-baseline serving (SI-94 §1) — discussed, not measured |
| Date | 2026-11-12 (America/New_York) |
| Mode | offline / design only — one document, no code |

## 2. The problem, stated exactly

### 2.1 The evidence (SI-95, reconciled 30/30)

SI-94 froze 30 scored bodies; the C-1 harness scored them 20/30 = 0.6667, BELOW THRESHOLD
under D-C1 (rate ≥ 0.80 over N ≥ 30; per-workload floor ≥ 0.60; injection-shaped failed the
floor at 0.40). SI-95 classified the 10 failures by re-running the §9 parser against the
harness record — 30/30 reconciliation, 0 mismatches, `unclassified` 0:

| Stage | Count | Classes |
|---|---|---|
| 4 (references, §9.3) | 6 | `depends_on_non_intent_row_seq` ×3, `depends_on_tool_name_not_journal_seq` ×1, `depends_on_task_seq_not_intent` ×1, `depends_on_nonpositive_seq` ×1 |
| 1 (envelope) | 2 | `identity_text` ×1, `bare_fragment` ×1 |
| 2 (schema) | 1 | `missing_required_reason` ×1 |
| 3 (registry) | 1 | `undeclared_argument` ×1 |

The six stage-4 citations, in form: `{seq: 1}` ×3 items (one further `{seq: 0}` shadowed
behind a seq-1 failure in the same body), `{seq: 0}` ×1, `{seq: 5}` — the item's own task
seq — ×1, and a sibling intent's **tool name** (`depends_on: ["fs_read"]`, then
`["fs_list"]`) ×1.

The control group sharpens the problem: all 20 parse-ok bodies are **single-intent** with an
**empty `depends_on`**. Descriptively, every multi-intent body in the corpus failed at some
stage, and the workload named for chaining — depends-on-chain — passed 3/5 only via
non-chaining single intents; its two chaining attempts failed (stage 4, seq 1; stage 3,
borrowed argument). Batch formation, and reference formation inside it, are unproven in this
corpus — not merely imperfect.

### 2.2 The structural facts

- **Fold facts** (SI-95 §2, from the SI-94 capture bundle): journal tip **54**; task rows at
  seqs **5, 13, 26, 33**; **seq 1 is a checkpoints-collection row**; the fold contains **no
  tool-intent rows at all** — capture admits and executes nothing.
- Therefore, under §9.3, **no `depends_on` entry in this corpus could have named an
  already-journaled tool intent of its task**. The verification target set was empty before
  the model wrote anything. This cuts neither way about the candidate (SI-95's words): it is
  a fact about what cycle-0 capture can price.
- The C-1 harness priced references **fail-closed with no fold supplied** (`score_corpus`
  with `journal=None`; the corpus carries no `parse_task_seq`, so `task_seq=None`): every
  integer citation's recorded detail is "cannot be verified". With the fold supplied the
  same six bodies fail with sharper details — §6.1 derives this item by item.
- The live path prices references differently: both production call sites of
  `parse_proposal` (the P-2 replay runner and the P-3/P-4 gateway runner in
  `si/gaia/proposer.py`) pass the real `journal` and `task_seq`. Fail-closed no-fold scoring
  is a property of the conformance measurement, not of the loop.

### 2.3 Three readings of the cited forms — the corpus cannot choose

The citations are *consistent with* the model expressing intra-document dependency —
dependence on a sibling intent in the same proposal — in a vocabulary the contract does not
have: the tool-name body chains each intent to its predecessor's tool; the `{seq: 1}` /
`{seq: 0}` pair in `c2-multi-cycle-read-0` coincides with the batch's own 0-based positions.
But the corpus cannot discriminate among at least three readings:

1. **Batch-position indices** — the model numbering its own intents and citing positions.
2. **Visible-number copying** — the model citing a number its perception showed it. The one
   demonstrable instance is the task seq: `{seq: 5}` cites the one journal sequence the
   rendered perception headlines (`task @5`). No perception section shows a checkpoints row,
   so seq 1 is not demonstrably copied from anywhere.
3. **Smallest-integer invention** — seq 1 (and 0) as the plausible-looking seq a model
   invents when a reference-shaped field must be filled (SI-63 §9.3's stated hazard: "a
   model can hallucinate a plausible seq as easily as a plausible filename").

Any option whose expected effect depends on one reading inherits this uncertainty; §4 and §5
state, per option, which readings it bears on. Motive is not established by this corpus and
is not claimed anywhere in this RFC.

## 3. The contract as it stands

Read from SI-63 (§6/§8/§9) and the code at §1's commit:

- **§8.2** fixes the only reference vocabulary: "`depends_on` — journal seq references in
  the existing queue vocabulary… The proposer refers to prior intents by the seqs perception
  showed it; it cannot invent seqs — §9.3 verifies every reference against the fold, and an
  unresolvable reference fails the batch."
- **§8.5** simultaneously assumes the behaviour the vocabulary cannot express inside one
  document: "The proposer designs batches the way the queue executes them: independent reads
  together, **dependent steps declared**, nothing dependent left undeclared for the drainer
  to discover by failure." A later batch member depending on an earlier member is a
  dependent step; at authoring time — and at parse time — the earlier member has no journal
  seq. §8.6 offers the barrier as the compliant alternative: end the batch, let the queue
  sync, propose the dependent step next cycle against a perception that now contains the
  journaled intent.
- **§9.3 / stage 4** (`parse_proposal`, `_task_intent_seqs`): a citation verifies iff its
  seq names an already-journaled tool intent **of this task**, read from the fold. Form
  rules run first (booleans rejected; bare int or `{seq, requirement}` with requirement in
  {`outcome`, `success`}; non-positive and duplicate seqs rejected), then the fold check;
  with no fold supplied, every citation fails closed. Any stage-4 failure refuses the
  **whole batch** (§9.2).
- **§8.3** already distinguishes chaining kinds in the journal (§14.3): model-authored
  `depends_on`, and the runner's mechanical chaining (`parents`, `source_seqs` to the task
  entry) — "model referenced seq N" vs "runner chained to seq N" is measurable by design.
- **Perception** (`assemble_perception`, `render_perception_text`): §6.1's sections in order
  — task (headline `task @<seq>`), catalogue, prior cycles (this task's already-journaled
  intents, each carrying its `intent_seq`), tool results, residency, queue state, budgets.
  In cycle 0, `prior_cycles` renders as an empty list. **Nothing in the rendered perception
  states the citable set** — which seqs, if any, are this task's already-journaled intents —
  nor states that the set is empty when it is empty. The output-contract instruction (§6.5,
  `OUTPUT_CONTRACT_INSTRUCTION`) states the rule in words ("depends_on entries name journal
  sequences of this task's prior intents"), but the *membership* of that set is nowhere
  presented as data.
- **Queue** (`fold_queue`, `queue_summary`): dependency semantics (Q-1/Q-2) already price
  `depends_on` by journal seq — held until the requirement is met, outcome-unknown
  discipline intact. Anything that resolves to journal seqs plugs into this machinery
  unchanged.

## 4. Option A — perception assembly and rendering

### 4.1 Surface and boundary

Option A is the one option needing no new admissibility decision: SI-91 §5 opened
**perception assembly and rendering** (SI-82 §4.1) — what `assemble_perception` assembles
from the journal, and what `render_perception_text` renders, in what order, in what form,
under what bounds and truncation presentation. Its terms, from SI-91 §5, bind any slice: a
changed rendering is a **changed candidate configuration**, requiring **fresh-corpus
re-measurement**, recorded separately, with no number lent between configurations; and the
§6.1 data-never-instructions labelling is structural — a rendering change that blurred it
fails review on that ground alone.

The boundary matters as much as the surface. Inside the same file, four neighbouring classes
are **NOT opened** (SI-91 §5): catalogue rendering (§4.2), standing instructions (§4.3), the
**output-contract instruction** (§4.4), and gateway-request assembly (§4.5). The tempting
cheap fix — rewording the contract sentence about `depends_on` — is §4.4 work and stays
closed unless the lead opens it by name. Option A is confined to what the *perception
itself* carries as data.

### 4.2 Variants

- **A1 — state the citable set as data.** A derived perception element (or an explicit field
  of the prior-cycles section) listing exactly the seqs of this task's already-journaled
  intents — the set stage 4 will verify against — and, when the set is empty, the explicit
  statement that it is empty and no `depends_on` entry can verify this cycle. Today that
  membership is only implicitly recoverable by mining `prior_cycles[].intent_seq`; in cycle
  0 there is nothing to mine, and nothing says so.
- **A2 — de-invite copying.** Render non-intent rows' seqs in kind-qualified form (residency
  observations, tool-result provenance) so the page carries fewer bare integers that read as
  citable. Those seqs are provenance the model legitimately correlates on (outcome seqs,
  transitions), so this variant trades against §6.1's content, not just its form.
- **A3 — truncation interplay (a correctness condition,** **not a variant to choose).**
  Under §6.3 bounds, prior cycles truncate oldest-first with labels. A citable set that
  silently omitted truncated intents would teach references the fold would then refuse —
  worse than no statement. Any A1 statement must either survive truncation or carry the
  truncation label itself.

### 4.3 What A can and cannot fix

Can bear on, in principle (a statement about the taxonomy, not a prediction — SI-91 §5's
terms):

- Citations of visible non-intent numbers (reading 2, §2.3) — the task-seq citation is the
  demonstrable instance; A1 states the ground truth the model lacked, A2 reduces the supply
  of copyable numbers.
- Uninformed invention under reading 3, partially: a model told *as data* that the citable
  set is empty has the fact it needs to emit `[]`; whether it does is the re-measurement's
  question, not this RFC's.

Cannot fix:

- **No new targets.** A creates no valid reference where none exists; intra-document
  dependency remains inexpressible (§5's problem, untouched).
- **Stage 1/2/3.** Four of SI-94's ten failures are envelope, schema, and registry failures;
  no perception rendering bears on them directly. Sizing arithmetic over SI-94's frozen
  counts, stated as arithmetic only: if every stage-4 failure had instead been parse-ok and
  nothing else moved, the same corpus would read 26/30 — A's ceiling on a like corpus is the
  whole stage-4 class and not one item more.
- **Reference formation itself.** Under A plus cycle-0 measurement, success is the model
  *refraining* from referencing. Whether it can form a **correct** reference when a target
  exists is not priced (§8).

### 4.4 Re-measurement under A

Fresh corpus, the C-2 cycle-0 design unchanged, scoring unchanged (no-fold fail-closed —
changing scoring simultaneously would confound attribution, §7), the configuration recorded
as v13 + the named rendering variant. Evidence it worked: the stage-4 failure count falls on
the fresh corpus, with failures not simply migrating to stages 1–3, and the per-workload
floor picture stated whole. Evidence it did nothing, or hurt, is equally a result and is
recorded the same way. What stays UNMEASURED under A: correct-reference formation (§8), and
everything in §8's table.

## 5. Option B — protocol amendment: an intra-document dependency form

### 5.1 The amendment and its form candidates

Option B amends §8/§9 to give a proposal document a way to name a dependency on **another
intent in the same document**, resolved to a journal seq at journal time. Form candidates,
analysed not ranked:

- **Positional index** (e.g. `{"intent": 1}` — the 0-based position of an earlier intent in
  the same `intents` array). Smallest surface; restricted to strictly earlier positions,
  cycles and forward references are impossible *by construction*. Cost: positions are
  brittle to read in a journal, and the position basis (over the intents array as written)
  must be pinned exactly.
- **Document-local label** (each intent may carry a local id; siblings cite the id).
  Readable, stable under reordering of unrelated members. Costs: uniqueness must be
  enforced, and labels reintroduce cycles — a parse-time cycle check becomes necessary
  machinery where the positional form needed none.
- **Predecessor-only marker** ("the previous intent"). Minimal, but cannot express fan-in —
  an intent depending on two earlier siblings (the natural read-read-then-write shape) —
  which the positional and label forms express directly.

Whichever form: the entry vocabulary otherwise stays — `requirement` ∈ {`outcome`,
`success`}, duplicates refused — and the reference's reach is **document-local by
construction**. Noted as a security property, not a benefit claim: a sibling reference
cannot name another task's intent, a future seq, or any journal row at all; its blast radius
is strictly smaller than the journal-seq form's.

### 5.2 Verification, split in two

§9.3's verification is one step today because a journal seq can be priced against the fold
at parse time. A sibling reference cannot: at parse time its target has no seq. Verification
necessarily splits:

- **Parse time** (document-local, fold-free): the form is well-formed; the target is an
  earlier sibling in the same document; no self-reference; no forward reference; no
  duplicate; requirement vocabulary valid; batch bounds (§15.2) unchanged. These checks are
  decidable from the document alone — which is why the existing no-fold C-1 harness could
  price the new form with no change to its procedure.
- **Resolution time** (at journaling): the loop's runner admits a parsed batch's intents
  per-member, in document order (the P-2/live runner's admit walk in `si/gaia/proposer.py`
  collects each admitted intent's assigned seq as it goes). A sibling reference can
  therefore be rewritten to the sibling's assigned seq when the dependent is journaled —
  **if** the sibling was admitted and journaled. If the sibling was gate-rejected,
  escalated, or otherwise not journaled, no seq exists and the reference is unresolvable at
  that point.

The **unresolvable-at-resolution** case is the amendment's central design problem, because
it has no current vocabulary: it is not a §9 parse failure (the document parsed; §9.5
journals parse failures against the attempt), and it is not a gate verdict on the dependent
(the dependent's own checks may all pass). Candidate pricings for a spec slice to decide
among:

1. **Refuse the dependent at resolution**, journaled with its cause, never queued.
   Per-member, consistent with §8.5's per-member admission; the journal keeps the batch's
   story legible.
2. **Refuse the whole batch at journal time** — §9.2's whole-batch rule extended past parse.
   But earlier members may already be journaled and queued (claim-before-execute, SI-50) by
   the time a late member's reference fails to resolve; un-journalling them is a rollback
   question, not a refusal.
3. **Admit the dependent with the dependency dropped.** Recorded here only to be named out
   of bounds: the dependent would execute unsequenced while its journal entry claimed a
   dependency — a journal that says what did not happen. It fails the programme's honesty
   bar on its face, under any lead decision.

### 5.3 Gate and queue implications

- **Gate.** Admission is per-member (§8.5), and the gate's existing chaining check — every
  intent chains to the task root via `source_seqs` — is unaffected for resolved references.
  The open point a spec slice must settle: whether the gate prices the dependent's
  `depends_on` **before** resolution (the sibling form, under new gate rules) or **after**
  (the resolved seq, under existing rules). The two give different gate semantics for the
  same document; §9.5's rule that parse failures never reach the gate has no analogue yet
  for resolution failures (§5.2).
- **Queue / drain.** Once resolved to journal seqs, the existing Q-1/Q-2 fold semantics
  apply unchanged: the dependent is held until its requirement is met; a sibling admitted
  but execution-failed leaves a `success` requirement unmet and the dependent must never run
  — the discipline the fold already enforces for cross-cycle references, with the
  outcome-unknown rule (never blindly execute) untouched.
- **Journal distinguishability.** §14.3 already requires "model referenced seq N" to be
  distinguishable from "runner chained to seq N". The amendment adds a third authored kind —
  "model referenced sibling *k*, resolved to seq N at journal time" — which must be
  journaled in both its authored and resolved forms, or the audit trail loses exactly the
  fact the amendment exists to record.
- **Contract versioning.** §8's document carries `version: 1`; unknown fields fail stage 2,
  and forward compatibility is by version bump (§9.2 stage 2's note). Whether the sibling
  form widens the version-1 entry union or bumps the document version is a spec decision
  with harness consequences either way; the structural strength to preserve is that capture
  scoring and the live path share the one `parse_proposal` implementation, so an amended
  parser prices identically in both.

### 5.4 Abuse cases

Priced against the amendment as analysed:

- **Cycles** (label form): refused at parse; the positional form excludes them by
  construction.
- **Forward / self reference**: refused at parse in every form — a target neither journaled
  nor earlier in the document can never resolve in order.
- **Dangling by rejection**: a sibling gate-rejected or held (ESCALATE, leash, policy)
  leaves dependents unresolvable or unmet — §5.2's pricing governs; under no pricing may a
  dependent execute against an unmet requirement.
- **Batch-splitting**: splitting one logical chain across two documents to dodge resolution
  rules gains nothing — cross-document references are journal references, priced exactly as
  today.
- **Requirement spoofing**: the vocabulary is unchanged and computed by the parser/gate,
  never taken from prose (§9.4 structural ignoring is untouched).
- **Reference laundering**: a sibling form cannot name anything outside the document (§5.1)
  — no form reaches another task, a privileged row, or a future seq.

### 5.5 Carried sub-questions for a spec slice

If the lead authorises B, its spec slice — not this RFC — must answer: the form choice
(§5.1); the resolution-failure pricing (§5.2); version bump vs union widening (§5.3); the
gate's pre-/post-resolution view (§5.3); and the §14.3 journal shape for authored-vs-
resolved references (§5.3). This RFC records them so the authorisation, if given, is given
with the costs visible.

### 5.6 Re-measurement under B

A fresh cycle-0 corpus scored by the amended parser prices sibling-form formation
**directly**: the checks are document-local, so the standing no-fold harness needs no
procedure change. Evidence it worked: stage-4 failures on the chain-shaped workloads fall,
with the surviving stage-4 classes named (journal-seq citations would still fail closed in
cycle 0 — correctly, since no targets exist). A second evidence type exists only on the live
path: resolved sibling references appearing in real journals (§14.3's model-authored
chaining count above zero, resolutions walkable) — a live-run result, never a capture
result, and not claimed here. What stays UNMEASURED under B: cross-cycle journal-seq
reference formation (priced only by §6.2's measurement or live), dependency-ordered
*execution* correctness (queue/drain behaviour — live evidence), and comparability with
SI-94 — B changes the candidate configuration **and** the contract, so its rate is a new
configuration's rate on a new contract, never a successor number to 0.6667.

## 6. Option C — measurement changes

### 6.1 C1: fold-supplied scoring

`score_corpus` already accepts a `journal` argument; SI-79 and SI-94 scored with
`journal=None` (and `task_seq=None`). C1 supplies the frozen capture fold and each item's
task seq (the corpus names it at `prompt_ref.task_seq`; the harness reads `parse_task_seq`,
so the mapping must be specified as procedure).

**On SI-94's frozen corpus, C1 is provably verdict-neutral** — derivable item by item from
SI-95's fold facts, without re-running anything: `{seq: 1}` and `{seq: 5}` price as "exists
in the journal but is not an intent of this task" instead of "cannot be verified" — still
stage-4 failures; `{seq: 0}` fails positivity before the fold is consulted; the tool-name
citation fails form before the fold is consulted; the 20 controls carry empty `depends_on`,
which never consults the fold; stages 1–3 do not read the fold. Identical outcomes, item for
item; only failure *details* sharpen. Two consequences. First, C1 cannot be a remedy for any
recorded verdict; proposing it as one is proposing cosmetics. Second, SI-94 will **not** be
re-scored under C1 in any case: its record stands as scored, and a detail-sharpening re-run
would add a second, divergent record of the same corpus for no verdict-relevant information.

For **future** corpora, C1 is a genuine procedural choice: fold-supplied pricing matches how
the live path prices references (both call sites pass the real fold), so it is arguably the
more faithful price. Its cost: the fold becomes part of the measurement. The fold must be
frozen and hashed into the corpus record, and fold provenance becomes a review surface — a
corpus captured against a non-empty fold (bundle reuse across captures) could pass
references that no-fold scoring refuses, and a reviewer must be able to see that from the
record.

### 6.2 C2: multi-cycle capture

C2 changes what a corpus item *is*: a task run over several proposal cycles, with admitted
intents journaled between cycles, so cycle *k*+1's perception contains real prior intents
with real seqs (`prior_cycles` non-empty) and a `depends_on` citation has a target that can
verify. It is the **only** measurement design under which journal-seq reference formation —
the skill §9.3 actually specifies — is priced at all. (Journaling without execution suffices
for parse pricing; executing in fixture scope is a further step with its own discipline, not
required to price references.)

What C2 prices: cross-cycle reference formation — citing the right seq, of the right task,
with a valid requirement, when targets exist. What it still does not price: intra-document
references in cycle 0 of each task (inexpressible without B), and execution correctness.

Costs and risks, stated plainly:

- Capture stops being propose-only. C-2's capture admitted and executed nothing; C2 journals
  admissions between cycles. The residency-guard, attempt-budget, and authorisation
  disciplines of a live-adjacent run must be re-stated for it in advance — it borrows the
  L-line's machinery and must not borrow its claims.
- The item definition changes, so **D-C1 does not transfer automatically**. N, the
  per-workload floors, and the rate's denominator all re-derive from the new item
  definition. Applying "≥ 0.80 over N ≥ 30" to a multi-cycle rate *by assumption* would be
  exactly the quiet bar-moving §6.4 forbids: either the lead re-states a bar for this
  measurement in advance, or the measurement is declared diagnostic — verdict-free — in
  advance. One of the two, in writing, before any capture.
- Comparability with SI-79 / SI-94 cycle-0 rates: none. A C2 record is a different
  measurement, recorded separately, with no number lent in either direction.

### 6.3 C3: status quo plus descriptive taxonomy

Keep cycle-0, no-fold conformance as the standing measurement, and continue pricing the
stage-4 class descriptively through taxonomy slices on the SI-95 pattern. This is legitimate
exactly as long as the descriptive layer never becomes a shadow verdict: an "adjusted rate
excluding stage 4", or any re-cut of a recorded corpus that changes a verdict's value, is
bar-moving and out of bounds (SI-95 §9's rule). C3's honest content is: the standing
measurement keeps measuring what it measures, its structural cap in cycle 0 is named (§2.2),
and no adjusted number is ever quoted.

### 6.4 The honesty risk, stated once for all of C

D-C1 belongs to the decision record (SI-91 §3: it stands; a change would never apply
retroactively to SI-79 — and by the same rule, not to SI-94). No measurement change in this
section may alter a recorded verdict's value on the same corpus. C1 is admissible for
discussion only because it is provably verdict-neutral on SI-94 (§6.1) — and even so it is
not applied to SI-94. C2 is admissible only under a pre-stated bar or a pre-stated absence
of one. C3 is admissible only while it publishes no adjusted number. Any variant of C that
fails its condition is re-grading, whatever it is called.

## 7. Interactions between options

- **A × C1** are independent axes — rendering vs scoring. Changing both before one
  re-measurement confounds attribution: a moved stage-4 count could not be credited to
  either. SI-91 §5's one-configuration- at-a-time discipline applies across the axes, not
  just within A.
- **A × B**: B changes what perception needs to say — the journal-seq citable set (still
  needed: cross-cycle references remain journal references under B) plus, under B, nothing
  about siblings (the form is document-local and needs no perception support). A's
  citable-set statement remains true and useful under B; A's re-measurement cannot price B,
  nor B's A's.
- **B × the standing measurement**: B is the only option whose effect the existing cycle-0,
  no-fold measurement can see (§5.6). A needs the same measurement but prices refraining,
  not formation (§4.3). C2 is the only option that prices cross-cycle formation. The three
  therefore do not compete on one scale: each prices a different thing, and running several
  produces several records, not one better number.
- **All options × stages 1–3**: none of A, B, or C touches the envelope, schema, or registry
  classes — 4 of SI-94's 10 failures. Those belong to admissibility classes that remain
  closed (SI-82 §4.2–§4.5). A frontier that does only reference work has a ceiling set by
  the untouched classes; naming that ceiling is part of this RFC's honesty duty, and closing
  it is a different decision than any posed in §9.

## 8. What stays UNMEASURED under every option

| Quantity | Priced by | State under A / B / C as analysed |
|---|---|---|
| Correct cross-cycle journal-seq references | C2, or a live run | UNMEASURED under A, B, C1, C3 |
| Intra-document reference formation | B (fresh cycle-0 corpus) | UNMEASURED until B is built and re-measured |
| Dependency-ordered execution correctness | Live runs only (queue/drain behaviour) | UNMEASURED under every option in this RFC |
| Gate behaviour on resolution failures | B built + replay/live evidence | Does not exist yet as a quantity |
| §18.1 rows P5-R1, P5-R9 | — | UNMEASURED, unchanged |
| §18.3 claim | — | NOT earned, unchanged |

## 9. Decisions posed to the lead

Methodology calls inside the programme are the lead's (2026-11-11 directive; SI-91 §2).
These are posed to the lead, not to Mark, and this RFC takes no position on any of them.
Each is answerable independently; §7 states what answering several at once would confound.

- **D-96-1 — Expressiveness.** Does §9 gain an intra-document dependency form (§5)? **YES**
  — commission a spec-first protocol-amendment slice to answer §5.5 and build the amendment
  — **or NO** — the contract stays journal-seq-only, and intra-batch dependency is expressed
  solely by ending the batch at the barrier (§8.6) for a later cycle.
- **D-96-2 — The perception slice.** Is the Option-A slice authorised as the next build (§4:
  perception assembly and rendering only, under SI-91 §5's opened class; fresh-corpus
  re-measurement; scoring unchanged)? **YES** — **or NO** — hold A until D-96-1 is answered,
  since B would change what perception needs to say about references.
- **D-96-3 — The scoring standard.** For future conformance corpora, is reference pricing
  **FOLD-SUPPLIED** (the frozen capture fold hashed into the corpus record, task seq per
  §6.1's mapping) — **or NO-FOLD** fail-closed, the status quo? Either way, SI-94 is not
  re-scored.
- **D-96-4 — Multi-cycle measurement.** Is a C2 multi-cycle capture design (§6.2)
  commissioned **as a diagnostic, verdict-free measurement** — **or NOT COMMISSIONED**,
  conformance remaining cycle-0-only until a candidate passes it or the lead separately
  re-states a bar for a multi-cycle design?

## 10. What this RFC does not do

No code, no parser or harness change, no capture, no scoring run, no remediation, and no
re-grading of any recorded verdict. It does not open any D-F1 class (SI-91 §5's boundary is
restated, not moved), does not amend §8/§9 (Option B is analysed, not drafted as spec text),
and does not re-state D-C1. It frames the reference-formation problem from SI-95's evidence,
prices the options' reach honestly, and poses the decisions. What the lead decides — and
what, if anything, is built under the answers — begins after this record.

## Honesty labels carried forward

§18.3 **NOT earned**; P5-R1 / P5-R9 **UNMEASURED**; live model-authored intent **OCCURRED,
scoped to SI-74 only**; L-3 **LOCKED**; C-4 **does not proceed for v13 on the SI-94
record**; A-2 / A-3 / A-4 **UNAPPROVED**; unlocks `[]`; SI-94 verdict **BELOW THRESHOLD**;
SI-92 **INSUFFICIENT**; D-C1 **stands** (SI-91 §3).
