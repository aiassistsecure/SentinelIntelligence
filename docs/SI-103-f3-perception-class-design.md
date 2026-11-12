# SI-103 — F-3 Perception-Class Design: The Citable Set as Data

## The honest headline

This is a **design only** for the one harness/prompt class the lead
opened in SI-91 §5 — perception assembly and rendering — sequenced
after the Imagine v13 baseline (SI-91 §8.4). It designs exactly one
variant from SI-96 §4.2: **A1, state the citable set as data**, with
SI-96 §4.2's A3 truncation condition incorporated as a correctness
requirement, and A2 rejected (§6). It changes no code, runs no model,
captures nothing, and claims no conformance result.

The baselines this design is sequenced after both stand, unchanged:
SI-94 measured Imagine v13 under the v1 teaching at **20/30,
BELOW THRESHOLD**; SI-100 measured the v2 teaching at **9/30,
BELOW THRESHOLD**, and that teaching was **withdrawn by SI-102**
(PR #110, merged). The live configuration is back to the v1 teaching
— SI-94's measured configuration. A future re-measurement under the
rendering designed here would be a **new, changed configuration**,
recorded separately, with no number lent to or from either baseline.

## Table of contents

- [0. Honesty table — no row changes](#0-honesty-table--no-row-changes)
- [1. Fingerprint](#1-fingerprint)
- [2. Authority and sequencing](#2-authority-and-sequencing)
- [3. The problem this variant addresses](#3-the-problem-this-variant-addresses)
- [4. Chosen variant A1 — precise design](#4-chosen-variant-a1--precise-design)
- [5. What does not change](#5-what-does-not-change)
- [6. Variants and weakenings rejected](#6-variants-and-weakenings-rejected)
- [7. Failure-class mapping — SI-95 and SI-101](#7-failure-class-mapping--si-95-and-si-101)
- [8. Re-measurement plan](#8-re-measurement-plan)
- [9. Future implementation slice — scope sketch](#9-future-implementation-slice--scope-sketch)
- [10. Risks and open points for the build slice](#10-risks-and-open-points-for-the-build-slice)
- [11. What this design does not do](#11-what-this-design-does-not-do)

## 0. Honesty table — no row changes

| Row | State, unchanged by this slice |
|---|---|
| SI-63 §18.3 claim | **NOT earned** |
| SI-63 §18.1 rows P5-R1, P5-R9 | **UNMEASURED** |
| Live model-authored tool intent | **OCCURRED, scoped to SI-74 only** |
| SI-94 verdict (v13, v1 teaching) | **BELOW THRESHOLD — 20/30 = 0.6667** |
| SI-100 verdict (v13, v2 teaching) | **BELOW THRESHOLD — 9/30 = 0.30** |
| SI-99 v2 teaching | **WITHDRAWN by SI-102** (PR #110); live configuration is v1 teaching |
| C-4 for v13 | **Does not proceed** on the SI-94 or SI-100 record |
| L-3 residency-loss drill | **LOCKED** |
| A-2 / A-3 / A-4 | **UNAPPROVED** |
| Readiness unlocks | `[]` |
| D-C1 | **Stands** (≥0.80 / N≥30 / per-workload floor ≥0.60) |
| D-F1 classes opened | Perception assembly and rendering only (SI-91 §5) |

The candidate is a specialised machine learning model. This slice is
machine learning infrastructure throughout: one design document,
written offline on CPU only, that ran no parser, harness, capture,
or live call in its production.

## 1. Fingerprint

| Field | Value |
|---|---|
| SI commit | `7fe1ce610b1a2d7b61d3fce7327265d590500ff7` (main, SI-102 / PR #110 merged) |
| Inputs | `docs/SI-91-decision-record-and-v13-conformance-plan.md` (§5, §7.6, §8.4), `docs/SI-96-reference-formation-rfc.md` (§4), `docs/SI-95-failure-taxonomy.md`, `docs/SI-101-failure-taxonomy.md`, `docs/SI-94-c2-conformance-record.md`, `docs/SI-100-v13-v2-conformance-record.md` — all on main, read only |
| Code read at the same commit | `si/gaia/proposer.py` (`assemble_perception`, `render_perception_text`, `build_gateway_request`, `OUTPUT_CONTRACT_INSTRUCTION` as restored by SI-102) — read only |
| Corpora | Referenced by record and hash in SI-94 / SI-100 / SI-95 / SI-101; no corpus body is copied into this document (never duplicate bytes) |
| Candidate under discussion | Imagine v13 — discussed, not measured |
| Date | 2026-11-12 (America/New_York) |
| Mode | offline / design only — one document, no code, no tests |

## 2. Authority and sequencing

- **SI-91 §5 (D-F1, lead decision 2026-11-11):** harness/prompt fixes
  are admissible for **one class only** — perception assembly and
  rendering (`assemble_perception` / `render_perception_text` in
  `si/gaia/proposer.py`) — as a changed configuration requiring
  fresh-corpus re-measurement, recorded separately. No other SI-82
  §4 class is opened: catalogue rendering (§4.2), standing
  instructions (§4.3), the output-contract instruction (§4.4), and
  gateway-request assembly (§4.5) stay **NOT opened** for this slice.
- **SI-91 §8.4:** the F-3 perception-class slice is sequenced
  **after** the v13 baseline, so the two configurations are never
  confounded. That sequencing condition is now met twice over:
  SI-94 is the v1-teaching baseline and SI-100 is the (withdrawn)
  v2-teaching measurement. Neither is re-graded by this design.
- **SI-96 §4** is the starting design surface. SI-96 recommended
  none of its options; this document makes the F-3 choice SI-96
  left open, inside Option A only. It does not adopt SI-96's
  Option B (protocol amendment — the sibling form's history is
  SI-97/SI-98/SI-99/SI-100/SI-102, and its teaching stands
  withdrawn) or Option C (measurement changes — §8 keeps scoring
  and fold discipline exactly as they stand, for attribution).

## 3. The problem this variant addresses

SI-96 §3 states the gap exactly, and this design takes it verbatim
in substance: perception's sections (§6.1 order — task, catalogue,
prior cycles, tool results, residency, queue state, budgets) render
each prior intent with its `intent_seq`, but **nothing in the
rendered perception states the citable set** — which seqs, if any,
are this task's already-journaled intents that a `depends_on` entry
may name under §9.3 — **nor states that the set is empty when it
is empty**. In cycle 0, `prior_cycles` renders as an empty list
and there is nothing to mine. The output-contract instruction
states the rule in words; the *membership* of the set is nowhere
presented as data. A1's whole content is to present that membership
as data, in the perception itself, and nothing else.

## 4. Chosen variant A1 — precise design

### 4.1 The derived element

`assemble_perception` gains one derived element, computed from the
same journal fold it already reads, by the same rule §9.3 verifies
against — this task's already-journaled tool intents, in journal
order (the rows `_task_intent_seqs` prices; no new source, no new
inference):

```text
citable_reference_set:
  seqs: [<intent_seq>, ...]        # sorted ascending, may be empty
  empty: true | false
  basis: "this task's already-journaled tool intents only (§9.3)"
  data_not_instructions: <the §6.1 DATA_NOT_INSTRUCTIONS label>
```

Placement and form, pinned so a build slice has no discretion left
on the points that matter:

1. **Assembly:** a new top-level perception section named
   `citable_reference_set`, assembled inside `assemble_perception`
   from the already-computed prior-intent rows — the `intent_seq`
   values of this task's tool-intent rows, and nothing else. Task
   rows, checkpoints rows, residency rows, tool-result rows, and
   any other task's rows are **never** members, by construction.
2. **Rendering:** `render_perception_text` renders it as its own
   labelled block, immediately **after** the `prior_cycles` line
   and before `tool_results` — the position where a reader who has
   just seen the prior intents meets the set derived from them.
   The other sections' order, names, and contents do not move.
3. **Empty set (the cycle-0 case, and the measured corpora's
   case):** the block renders the empty list explicitly, in this
   form (illustrative rendering text, not a corpus body):

   ```text
   citable_reference_set: {"seqs": [], "empty": true,
     "note": "no journal sequence can be named in depends_on
     this cycle; depends_on must be []"}
   ```

   The note is **rendered data derived from the set's emptiness**,
   not a new instruction: it states a verifiable fact about the
   fold (§9.3 will refuse any entry this cycle), it lives inside
   the perception's data-never-instructions labelling, and it does
   not touch the output-contract instruction (§5).
4. **Non-empty set:** the block renders the seq list explicitly,
   e.g. `{"seqs": [12, 27], "empty": false}` with the same basis
   and label fields. No seq is rendered in this block that is not
   a member; no member is omitted except under §4.2's truncation
   rule, which must then say so in the block itself.
5. **Digest and size:** the new section is inside the perception's
   canonical form, so the perception digest and byte count change
   — expected, and recorded as part of the changed configuration
   (§8). The catalogue digest does not change. The section counts
   toward the §6.3 perception bound like every other section.

### 4.2 Truncation condition (SI-96 A3, incorporated)

A3 is not a second variant; it is A1's correctness condition.
Prior cycles truncate oldest-first under §6.3 with labels. A
citable set that silently omitted truncated intents would teach
references the fold would then refuse — worse than no statement.
Therefore:

- If any prior-intent row whose seq would be a member has been
  truncated from `prior_cycles`, the rendered set must either
  still list that seq (the set is small: one integer per intent)
  or carry the §6.3 truncation label inside its own block stating
  that the set is partial and which seqs were omitted.
- The design's preference, stated for the build slice: **list the
  seqs in full** — the set's size is bounded by the batch/cycle
  counts, not by argument or result payloads — and apply the
  truncation label only if the bound is still exceeded. A silent
  partial set fails review on this section alone.
- The set itself never invents a seq to fill the space a truncated
  member left: membership is fold-derived or labelled absent.

### 4.3 Worked cycle shapes (design examples, not evidence)

- **Cycle 0, any task:** `prior_cycles: []`,
  `citable_reference_set: {seqs: [], empty: true, …}`. This is the
  shape every SI-94 and SI-100 corpus item was captured under —
  except that, under A1, the emptiness is stated instead of left
  to be inferred from an empty list.
- **Cycle k>0, two admitted prior intents:** `prior_cycles`
  carries both intents with their `intent_seq` values, and the set
  lists exactly those two values. A citation of the task's own
  seq, a checkpoints seq, or a result seq is visibly **not** in
  the set the block presents.

## 5. What does not change

Each item below is a boundary of this design, not an omission:

- **Gate:** untouched. No gate check, verdict, or chaining rule
  moves; the gate prices only journal-seq references, as today.
- **Parser (§9):** untouched. All six stages, the §9.3 fold rule,
  the entry vocabulary (bare int, or `{seq, requirement}` with
  requirement in {`outcome`, `success`}), and whole-batch refusal
  stand exactly as at the fingerprint commit.
- **Output-contract instruction (§4.4, NOT opened):** untouched.
  The v1 statement restored by SI-102 — including its closing
  `depends_on` sentence — is byte-for-byte the live text after
  this design is built. A1 adds data to perception; it rewords
  no instruction to the model anywhere.
- **Standing instructions (§4.3), catalogue rendering (§4.2),
  gateway-request assembly (§4.5):** untouched, each for the same
  reason — its class was not opened (SI-91 §5).
- **Journal:** no new record kind, no schema change. The attempt
  record already carries the perception digest; the changed
  digest is the journal's evidence that this configuration ran.
- **§17 perception-poisoning posture — must not weaken, and does
  not:** (a) the set is derived **only** from the journal fold's
  tool-intent rows for this task — never parsed from, or
  influenced by, perceived content (tool results, file contents,
  search matches), so perceived data cannot inject a citable seq;
  (b) the block carries the §6.1 data-never-instructions label
  like every other perceived element — perceived records remain
  data, never instructions, and the labelling is not blurred;
  (c) no perceived string is copied into the set — its only values
  are integers from the fold, the basis string, and the label.
  Any build whose set could be content-influenced fails review
  on this section alone (SI-91 §5's review rule, applied).

## 6. Variants and weakenings rejected

- **A2 (de-invite copying) — rejected.** Rendering non-intent
  rows' seqs in kind-qualified form trades against §6.1's content,
  not just its form: residency and tool-result seqs are provenance
  the model legitimately correlates on. A2 also misses the
  dominant evidence — seq 1, SI-95's largest stage-4 form, is
  shown nowhere in perception, so removing bare integers from
  view cannot explain or address it — while degrading auditable
  provenance for a speculative gain. Not built, not combined.
- **Any variant that rewords the contract sentence** about
  `depends_on` — rejected as out of class: that is §4.4 work,
  NOT opened (SI-91 §5; SI-96 §4.1 names this the tempting cheap
  fix and excludes it).
- **Any variant that weakens §17** — e.g. deriving the set from
  model-visible text, letting perceived content suggest members,
  or dropping the data-never-instructions labelling for salience
  — rejected outright, with the reasons in §5.
- **Combining A1 with a scoring or fold change** — rejected for
  this slice: changing rendering and scoring before one
  re-measurement confounds attribution (SI-96 §7). §8 holds
  scoring fixed for exactly this reason.

## 7. Failure-class mapping — SI-95 and SI-101

A statement about the taxonomies, not a prediction (SI-91 §5's
terms). "Plausibly addresses" means: the class's cited form is one
the stated-as-data set speaks to directly. Whether the candidate
acts on it is the re-measurement's question (§8).

### 7.1 Classes A1 could plausibly address

| Class (source) | Count | Why A1 speaks to it |
|---|---|---|
| `depends_on_task_seq_not_intent` — `{seq: 5}`, its own task (SI-95) | 1 | The demonstrable visible-number case: the task seq is headlined in perception. A1 states, as data, that the task seq is **not** a member and (cycle 0) that no seq is. |
| `depends_on_nonpositive_seq` — `{seq: 0}` (SI-95 ×1; SI-101 ×7, always on a first intent) | 1 + 7 | Seq 0 is in no set. The explicit empty-set statement gives the fact the citation contradicts; SI-101's concentration of this form on first/only intents is exactly the cycle-0 shape A1 renders for. Partial only: A1 also cannot show *why* 0 was chosen. |
| `depends_on_non_intent_row_seq` — `{seq: 1}`, a checkpoints row (SI-95 ×3; SI-101 ×1) | 3 + 1 | Partial only, stated honestly: seq 1 appears nowhere in perception, so a copying account is not established (SI-96 §2.3's three readings). A1 bears on the invention reading — a model told as data that the set is empty has the fact it needs to emit `[]` — and on no other reading. |

### 7.2 Classes A1 cannot address — stated explicitly

| Class (source) | Count | Why it is outside A1's reach |
|---|---|---|
| `depends_on_tool_name_not_journal_seq` — `depends_on: ["fs_read"]`, `["fs_list"]` (SI-95) | 1 | Intra-document dependency expressed in a vocabulary the v1 contract does not have. A1 creates **no new reference form and no new target**; making a tool name valid is protocol work, not perception rendering. Cross-tool / intra-document chaining is **not in this class** — say so plainly: A1 does not touch it. |
| Cross-tool argument bleeding, stage 3 — `max_bytes` on `fs_search`/`fs_write`, `content` on `fs_edit` (SI-101 ×9; SI-95's single `undeclared_argument` is the same genus) | 9 + 1 | A registry-stage defect about argument schemas. A1 changes no catalogue rendering (§4.2 NOT opened) and no args schema presentation. **Not in this class.** |
| Stage 1 envelope — `identity_text`, `bare_fragment` (SI-95 ×2); version narration (SI-101 ×3) | 2 + 3 | No proposal document, or a narration line before one. Perception's citable-set block bears on neither shape. |
| Stage 2 schema — `missing_required_reason` (SI-95 ×1); hyphenated `depends-on` unknown field (SI-101 ×1) | 1 + 1 | Document-shape defects priced before references are read. Untouched by rendering the reference set. |
| Sibling self-references, shadowed (SI-101 audit: 2 entries, 0 correct uses) | 0 priced | The sibling form is not taught (SI-102 withdrew the teaching) and A1 neither teaches nor mentions it. Its entries failed behind stage 3 and were never priced; A1 changes nothing about them. |
| Correct-reference formation when targets exist | unmeasured | Under A1 plus cycle-0 measurement, success is the model **refraining** from referencing. Whether it can form a correct reference when the set is non-empty is **not priced** by this design's measurement (§8) and stays UNMEASURED. |

### 7.3 Ceiling arithmetic — arithmetic only

Over the frozen counts, stated as arithmetic and nothing more:
if every SI-94 stage-4 failure had instead been parse-ok and
nothing else moved, that corpus would read 26/30 — A1's ceiling
on a like corpus is the stage-4 class and not one item more.
On SI-101's counts the same arithmetic reads 17/30, still below
D-C1, because SI-100's dominant mass is the stage-3 bleeding
A1 cannot reach (§7.2). A design whose ceiling on one baseline's
shape cannot cross the bar is designed anyway because the class
it prices — refraining when no reference can verify — is real,
named by SI-95/SI-96, and is the only class the lead opened.

## 8. Re-measurement plan

Exactly SI-91 §7.6 / C-1 harness discipline, applied to the
changed configuration. This plan is fixed in advance; the build
slice does not improvise it, and this document runs none of it.

1. **Fresh corpus.** Captured for this measurement and no other.
   No SI-94 or SI-100 item, body, or response is reused. Shape
   fixed before capture, as SI-94 fixed it: the six L-1 v2
   workloads × 5 distinct-body responses = **30 scored items**;
   at most 10 attempts per workload; transport failures recorded,
   unscored, never counted in N; duplicates logged, not entered.
2. **Changed configuration, recorded separately.** The record
   names the configuration as Imagine v13 + the A1 rendering
   (this document's §4, by slice/PR once built), served as SI-94
   served v13, with the v1 teaching live (SI-102's state) and
   every other element of the configuration held at SI-94's:
   same seat discipline, residency guard in force, cycle-0
   capture, no execution. The perception digest difference from
   the A1 section is part of the configuration record. No number
   from this measurement is quoted as a number of SI-94's or
   SI-100's configuration, in either direction (**NO-FOLD**
   between configurations, in SI-91 §5's sense).
3. **Scoring — unmodified harness, NO-FOLD reference pricing.**
   Scoring runs through the C-1 conformance harness (SI-77) and
   the §9 parser, **both unmodified**, on a fresh checkout of
   main, with the harness's standing call — `score_corpus` with
   `journal=None`, fail-closed on every citation. Fold-supplied
   scoring is not adopted (SI-96 §6.1/§7: it would confound
   attribution and add fold provenance as a review surface).
4. **Bar — D-C1, unmodified (NO-FOLD against D-C1).** Rate
   **≥ 0.80** over **N ≥ 30**, ≥5 per workload, **per-workload
   floor ≥ 0.60**. The verdict is stated in SI-94's form (MEETS
   THRESHOLD / BELOW THRESHOLD / INSUFFICIENT if N < 30), with
   the rate and per-workload rates whole. D-C1 is not re-stated,
   rounded, subset, or re-based by this plan.
5. **Evidence criteria, stated in advance.** Evidence A1 worked:
   the stage-4 failure count falls on the fresh corpus, with
   failures not simply migrating to stages 1–3, and the
   per-workload floor picture stated whole. Evidence it did
   nothing, or hurt, is equally a result and is recorded the
   same way, with a taxonomy slice on the SI-95 pattern to say
   which. No adjusted rate (e.g. excluding stage 4) is ever
   quoted (SI-96 §6.3's rule).
6. **What stays UNMEASURED under this plan:** correct cross-cycle
   journal-seq reference formation (needs a multi-cycle design
   or a live run — neither is commissioned here), dependency-
   ordered execution correctness, and §18.1 P5-R1/P5-R9.

## 9. Future implementation slice — scope sketch

Not this slice. For the build that follows, if the lead keeps it:
`assemble_perception` + `render_perception_text` in
`si/gaia/proposer.py` only (plus their tests) — the derived
section (§4.1), its rendering position (§4.1), the truncation
rule (§4.2), and tests pinning: empty-set rendering at cycle 0;
exact membership at cycle k>0; exclusion of task/checkpoints/
result/residency seqs; the §17 properties of §5 (fold-only
derivation, labelling intact); digest change confined to the
perception digest; and byte-stability for a byte-stable fold.
No parser, gate, contract-instruction, catalogue, standing-
instruction, gateway-assembly, or journal change is in scope;
a diff touching any of those fails this design at review.

## 10. Risks and open points for the build slice

- **Salience vs labelling:** the empty-set note must be plain
  enough to be used and must stay inside the data labelling —
  §5's line, not a wording free-for-all. The build slice pins
  the exact rendered string in tests; this design pins its
  content and position (§4.1), not its every character.
- **Digest churn:** every perception digest changes when A1
  lands. Anything comparing digests across the change must
  compare within a configuration (§8.2) — the build slice
  should grep for cross-configuration digest assumptions.
- **Bound pressure:** the set adds bytes to every perception.
  On truncation-shaped workloads this could itself trigger
  §6.3 truncation earlier; §4.2's rule and the re-measurement's
  perception-truncation workload are the checks, in that order.

## 11. What this design does not do

No code, no tests, no capture, no scoring run, no live model
contact, no conformance claim, and no re-grading of SI-94 or
SI-100. It does not open any D-F1 class beyond the one SI-91 §5
opened, does not amend §8/§9, does not revive the withdrawn v2
teaching, and does not re-state D-C1. It picks the F-3 variant,
designs it precisely, maps its reach honestly — including the
classes, cross-tool argument bleeding among them, that lie
outside perception rendering's reach — and fixes the
re-measurement that would price it. The build, if any, is a
later slice under §9.

## Honesty labels carried forward

§18.3 **NOT earned**; P5-R1 / P5-R9 **UNMEASURED**; live
model-authored intent **OCCURRED, scoped to SI-74 only**; L-3
**LOCKED**; C-4 **does not proceed for v13**; A-2 / A-3 / A-4
**UNAPPROVED**; unlocks `[]`; SI-94 verdict **BELOW THRESHOLD
20/30**; SI-100 verdict **BELOW THRESHOLD 9/30**, v2 teaching
**withdrawn by SI-102**; D-C1 **stands**.
