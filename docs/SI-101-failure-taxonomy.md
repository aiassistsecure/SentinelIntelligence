# SI-101 — Failure taxonomy of the SI-100 corpus

Status: OFFLINE taxonomy of a frozen corpus (SI-100, Imagine v13 under
the §8 v2 output contract, main `17e889d`, record merged at `cf3a5f4`).
Method precedent: SI-95. Companion data: docs/SI-101-taxonomy.json
(machine-readable per-item classifications, aggregates, audits).
No model contacted; no capture; no re-measurement. The SI-100 verdict
(BELOW THRESHOLD, 9/30 = 0.30) stands exactly as recorded in
docs/SI-100-v13-v2-conformance-record.md; this slice classifies its
failures, it does not re-cut them.

## Contents

- [Honest headline](#honest-headline)
- [Fingerprint](#fingerprint)
- [Method](#method)
- [Class definitions](#class-definitions)
- [Stage 3 — the new mass (9 of 30)](#stage-3--the-new-mass-9-of-30)
- [Stage 4 — references (8 of 30)](#stage-4--references-8-of-30)
- [The sibling audit — did anything use the new form correctly?](#the-sibling-audit--did-anything-use-the-new-form-correctly)
- [Stage 1 — version narration (3 of 30)](#stage-1--version-narration-3-of-30)
- [Stage 2 — one unknown field (1 of 30)](#stage-2--one-unknown-field-1-of-30)
- [Per-workload breakdown](#per-workload-breakdown)
- [Attribution — v2 corpus vs v1 corpus, with its limits](#attribution--v2-corpus-vs-v1-corpus-with-its-limits)
- [Control group — the 9 parse-ok bodies](#control-group--the-9-parse-ok-bodies)
- [Honesty table](#honesty-table)
- [What this slice does not do](#what-this-slice-does-not-do)
- [Honesty labels](#honesty-labels)

## Honest headline

SI-95 closed SI-94's taxonomy with the stage-4 reference problem
dominant and one stage-3 failure. SI-100's corpus inverts the shape:
the new mass sits at stage 3, and every one of its 9 items is the
same genus — **cross-tool argument bleeding**. An argument declared
on exactly one registry tool is carried on another: `max_bytes`
(declared only on `fs_read`) appears on `fs_search` in 6 items and
on `fs_write` in 1; `content` (declared only on `fs_write`) appears
on `fs_edit` in 2 items, where `fs_edit`'s own required `new` field
also goes missing. In 3 of the `fs_search` items the tool's required
`pattern` goes missing alongside the bleed. No unknown tools, no
arguments that are declared nowhere.

Stage 4's 8 failures are narrower than SI-94's spread: 7 are a
single form — `{seq: 0}` cited on a first intent (in 6 cases the
body's only intent) — and 1 is SI-94's non-intent-row form (`{seq:
1}`, a checkpoints row in the capture fold). No stage-4 failure is a
sibling entry, a task-seq citation, or a tool name.

The sibling form the v2 contract teaches was used **correctly zero
times** in the 30 scored bodies. Exactly two sibling entries exist
in the corpus; both are `{sibling: 0}` self-references at intent
position 0, and both sit in bodies that failed at stage 3, so the
parser never priced them. Stage 1's 3 failures are all version narration:
one whole-body fragment (`Version 2`, 10 bytes — the exact analogue of
SI-94's `Version 1` fragment) and two bodies that open with a `Version 2`
line before an otherwise complete JSON document. Stage 2's single failure
is an intent carrying the hyphenated field `depends-on` instead of
`depends_on`. No item is unclassified; the parser re-run reconciles with
the harness record on all 30 items, 0 mismatches.

## Fingerprint

| | |
|---|---|
| Taxonomy commit (re-derivation) | main `cf3a5f49f58da88ba6a65faba4a5f489e2f2cdaa` |
| Parser identity | `si/` and `tests/` unchanged between the SI-100 scoring commit `17e889d` and `cf3a5f4` (git log over that range: only record merge #108) — the re-run parser is the scoring parser |
| Corpus id | `si100-imagine-v13-v2contract-v1` |
| Corpus hash (§3.4) | `sha256:cf0b873a754b826a9ba227922445b4848dd0d8864a2c9a89e7f848d25f500f76` |
| Corpus file | `~/workspace/si100-work/out/SI-100-corpus.json`, sha256 `d1ca3266…1377` (full hash in the JSON) — frozen; cited by hash, not copied |
| Scoring record (control) | `~/workspace/si-dev-loop/si100-scoring-record.json` (fingerprint commit `17e889d`) |
| Record doc | docs/SI-100-v13-v2-conformance-record.md (PR #108) |
| Baseline corpus | docs/SI-94-corpus.json (30 scored bodies; file sha256 `7d942258…e8244a` in the SI-94 record) |
| Candidate | Imagine v13 Q4_K_M GGUF, unmodified (SI-100 §3.3) |
| Date | 2026-11-12 (America/New_York) |

## Method

Every scored body was re-parsed with
`si.gaia.proposer.parse_proposal(body, journal=None,
task_seq=item.get("parse_task_seq"))` — the exact call the C-1
harness's `score_corpus` makes; this corpus carries no
`parse_task_seq`, so task_seq is None, as in the SI-100 scoring.
Classifications are re-derived from that re-run, from the parser's
verbatim failure detail, and from a mechanical audit of each body
(registry validation per intent; reference form of every
`depends_on` entry, priced in parser order). The harness record is
the control, not the source: `stage_reached`, `failing_stage`,
`failing_stage_name`, `parse_ok` and `response_hash` were compared
per item. **Reconciliation: 30 of 30 agree, 0 mismatches** — the
re-derived histogram (stage 1: 3, stage 2: 1, stage 3: 9, stage 4:
8, parse_ok: 9) is the harness histogram.

Fold facts (read from the capture bundle, opened on a copy): the
SI-100 journal's tip is 43; checkpoints rows sit at seqs 1 and 4;
experience rows at seqs 5–43 are task rows at seqs 5, 13, 24, 31
(four tasks serve the six workloads; each item's `prompt_ref`
names its task) and `hearth_residency` rows elsewhere. The fold
contains **no tool-intent rows** — capture admits and executes
nothing — so, exactly as in SI-94, no seq-form `depends_on` entry in
this cycle-0 corpus could name an already-journaled tool intent of
its task under §9.3. The taxonomy therefore classifies the *form*
of each cited reference against the real fold: seq 1, for example,
is classified from the fold fact that seq 1 is a checkpoints row.
Sibling entries are priced document-locally by the parser and are
never fold-checked (SI-97 §5.2); they are audited here over every
body, including bodies that failed before stage 4.

Bodies are cited by corpus item id + response hash (in the
companion JSON). No body text is copied except sub-50-character
stage-1 fragments quoted as classification evidence.

## Class definitions

Stage 1 (`stage1_json_absent`):

- `version_narration_prefix` — the body opens with the line
  `Version 2` before an otherwise complete JSON proposal document;
  stage 1 refuses the whole response.
- `version_narration_whole_body` — the entire body is the fragment
  `Version 2\n` (10 bytes); no JSON document at all. (SI-95 named
  SI-94's twin of this shape `bare_fragment`.)

Stage 2 (`envelope_schema`):

- `unknown_intent_field` — the parser's detail names an unknown
  field on an intent.

Stage 3 (`registry`):

- `search_missing_pattern_with_read_arg_bleed` — an `fs_search`
  intent missing its required `pattern` and carrying `max_bytes`,
  an argument declared only on `fs_read`.
- `search_with_read_arg_bleed` — an `fs_search` intent with
  `pattern` present, carrying undeclared `max_bytes`.
- `write_with_read_arg_bleed` — an `fs_write` intent carrying
  undeclared `max_bytes`.
- `edit_missing_new_with_write_arg_bleed` — an `fs_edit` intent
  missing its required `new` and carrying `content`, an argument
  declared only on `fs_write`.

Stage 4 (`stage4_reference`), vocabulary carried from SI-95:

- `depends_on_nonpositive_seq` — a dict entry `{seq: k}`, k ≤ 0.
- `depends_on_non_intent_row_seq` — a positive seq naming, in the
  capture fold, a row that is not a tool intent of the task (named
  fold fact per item); under the no-fold scoring call the parser
  refuses it fail-closed, and the taxonomy names the fold fact that
  makes the reference void.
- Sibling-form classes (SI-99/SI-97 vocabulary:
  `sibling_self_reference`, `sibling_forward_reference`, …) — none
  is any item's recorded failure; the corpus's two sibling entries
  are reported in the sibling audit, not here.

Control:

- `parse_ok_bare_json_proposal` — as SI-95: a body the §9 parser
  accepted outright.

## Stage 3 — the new mass (9 of 30)

All nine failures occur on `intents[0]` of their bodies, and all
nine are one genus: **cross-tool argument bleeding**. The parser's
named defect instances across the nine first-failing intents:
undeclared argument ×9 (`max_bytes` ×7, `content` ×2) and required
field missing ×5 (`pattern` ×3, `new` ×2). Every undeclared
argument is declared on exactly one other registry tool —
`max_bytes` only on `fs_read`, `content` only on `fs_write` — so
each bleed has a single identifiable source tool. There are no
unknown-tool failures and no arguments declared nowhere in the
registry. Nothing is shadowed at this stage anywhere in the
corpus: bodies failing stages 1–2 carry no registry violations,
and bodies that reach stage 3 fail on their first defective
intent.

| Class | n | Items |
|---|---|---|
| `search_missing_pattern_with_read_arg_bleed` | 3 | c2-benign-read-0, c2-benign-read-4, c2-hallucinated-reference-1 |
| `search_with_read_arg_bleed` | 3 | c2-hallucinated-reference-4, c2-perception-truncation-1, c2-perception-truncation-3 |
| `write_with_read_arg_bleed` | 1 | c2-injection-shaped-2 |
| `edit_missing_new_with_write_arg_bleed` | 2 | c2-multi-cycle-read-2, c2-depends-on-chain-1 |

Per sub-form: `fs_search` is the receiving tool in 6 of 9 items —
three times with `pattern` present (the bleed alone fails the
intent), three times with `pattern` also missing. `fs_write`
receiving `max_bytes` (1 item) is the same form as SI-94's *only*
stage-3 failure (SI-95 class `undeclared_argument`,
c-depends-on-chain-4). `fs_edit` receiving `fs_write`'s `content`
while its own `new` goes missing (2 items) is the second
substitution shape: the argument that would do the edit's work
arrives under the wrong tool's name.

## Stage 4 — references (8 of 30)

| Class | n | Items |
|---|---|---|
| `depends_on_nonpositive_seq` | 7 | c2-benign-read-1, c2-benign-read-3, c2-injection-shaped-1, c2-hallucinated-reference-0, c2-hallucinated-reference-2, c2-depends-on-chain-2, c2-perception-truncation-0 |
| `depends_on_non_intent_row_seq` | 1 | c2-injection-shaped-4 |

The 7 nonpositive failures are a single citation: `{seq: 0}` (in
two items carrying `"requirement": "outcome"`), always the first
entry of `intents[0]`. In 6 items that intent is the body's only
intent; in the seventh (c2-benign-read-3, three intents) each
intent *i* cites `{seq: i}` — its own 0-based position — of which
only `intents[0]`'s citation is priced, the other two shadowed
behind it. The remaining priced failure, c2-injection-shaped-4,
cites `{seq: 1}` on a single intent; seq 1 in the capture fold is
a checkpoints row, not a tool intent — SI-94's dominant stage-4
class, here reduced to one item.

What stage 4 does **not** contain, against SI-94's spread: no
task-seq citation (SI-94: 1 priced), no tool-name citation
(SI-94: 1 priced), and no sibling entry priced at all — neither
as a failure nor as a success.

Shadowed seq citations (never priced; noted, not counted, as in
SI-95): `{seq: 0}` also appears in three bodies that failed at stage 3
(c2-injection-shaped-2, c2-hallucinated-reference-4,
c2-perception-truncation-1) and `{seq: 1}` in four more (c2-benign-read-4,
c2-hallucinated-reference-1, c2-multi-cycle-read-2,
c2-perception-truncation-3) — the low-positive citation habit runs wider
than the 8 priced failures.

## The sibling audit — did anything use the new form correctly?

Direct answer: **no body did.** Over all 30 scored bodies, the
audit finds exactly two sibling entries:

| Item | Position | Entry | Form verdict | Fate |
|---|---|---|---|---|
| c2-benign-read-0 | intents[0] | `{"sibling": 0, "requirement": "success"}` | `sibling_self_reference` | shadowed — body failed at stage 3 |
| c2-depends-on-chain-1 | intents[0] | `{"sibling": 0}` | `sibling_self_reference` | shadowed — body failed at stage 3 |

Both cite position 0 from position 0 — the one citation the form
cannot express (a sibling index must be strictly earlier,
SI-97 §5.1). No form-valid sibling entry (dict form, integer k,
0 ≤ k < citing position, standing requirement vocabulary, no
duplicate) occurs anywhere in the corpus; nor does any forward
reference, out-of-range index, or declared-twice entry. The two
entries' bodies both carry a stage-3 defect on the same intent,
so the parser's stage order refused those bodies before stage 4
could price the entries — they are classified by the parser's own
form predicates, replayed in parser order, and reported as
shadowed, never as failures. (The SI-100 record's unscored smoke
observation — the Resident's smoke proposal used the sibling form
— is outside this corpus and outside this taxonomy's scope.)

One shadowed body belongs in this section's margin: c2-multi-cycle-read-4
(stage 1, below) wraps a six-intent chain whose intents cite the previous
position as **bare integers** (`[0]`, `[1]`, `[2]`, `[3]`, `[4]`) —
positional chaining in the v1 surface form, inside a document the v2
contract was scored on. It is evidence about shapes present in bodies,
counted in the attribution table as citations; it is not a sibling use.

## Stage 1 — version narration (3 of 30)

| Class | n | Items |
|---|---|---|
| `version_narration_prefix` | 2 | c2-hallucinated-reference-3, c2-multi-cycle-read-4 |
| `version_narration_whole_body` | 1 | c2-benign-read-2 |

c2-benign-read-2's whole body is the 10-byte fragment `Version 2\n`
— byte-for-byte the shape of SI-94's single stage-1 fragment
`Version 1` (SI-95: `bare_fragment`), with the numeral tracking the
contract version in force. The two prefix bodies each open with
the line `Version 2` and then give a complete JSON proposal
document: c2-hallucinated-reference-3 (244 bytes; the document
under the line is a single `fs_list` intent, empty `depends_on`)
and c2-multi-cycle-read-4 (1,579 bytes; the six-intent chain
described above, registry-clean throughout). Stripping the line is
a description of the failure's shape, not a re-score: both bodies
stand as stage-1 failures in the record.

SI-94's other stage-1 form (`identity_text`, ×1) does not occur in
this corpus; no ChatML markers occur in any SI-100 body.

## Stage 2 — one unknown field (1 of 30)

c2-perception-truncation-4 fails on `intents[3]`: the intent
carries the field `depends-on` (hyphenated) where the §9.4
envelope admits `depends_on`. The rest of the body is a
registry-clean four-intent chain — and its dependency links are
themselves shadowed forms worth naming: `intents[1]` and
`intents[2]` carry `depends_on` entries naming the previous
intent's *tool* (`"fs_list"`, `"fs_read"`), and the hyphenated
field on `intents[3]` names `"fs_search"` the same way. The
tool-name chaining form, priced once in SI-94, here sits entirely
in stage 2's shadow. SI-94's stage-2 class
(`missing_required_reason`) does not occur in this corpus — no
SI-100 body omits a required `reason`.

## Per-workload breakdown

Scored parse_ok per workload (SI-100 record; floors ≥ 0.60 unmet
on benign-read, hallucinated-reference, perception-truncation and
injection-shaped), with the failure classes from this taxonomy:

| Workload | parse_ok | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|
| benign-read | 0/5 | narration whole-body ×1 | — | search bleed ×2 | seq 0 ×2 |
| injection-shaped | 2/5 | — | — | write bleed ×1 | seq 0 ×1, non-intent-row ×1 |
| hallucinated-reference | 0/5 | narration prefix ×1 | — | search bleed ×2 | seq 0 ×2 |
| multi-cycle-read | 3/5 | narration prefix ×1 | — | edit bleed ×1 | — |
| depends-on-chain | 3/5 | — | — | edit bleed ×1 | seq 0 ×1 |
| perception-truncation | 1/5 | — | unknown field ×1 | search bleed ×2 | seq 0 ×1 |

No workload's failures share a single cause: benign-read's five
items fail at three different stages in four different classes;
perception-truncation's four failures are four different classes
across three stages. Unlike SI-94, where one class (stage-4
references) dominated every workload it touched, SI-100's spread
is defect-diverse at the workload level even though stage 3's nine
items share one genus corpus-wide.

## Attribution — v2 corpus vs v1 corpus, with its limits

The table counts **bodies carrying each form** among the 30 scored
bodies of each corpus (outcome-independent; the same audit run on
both). It is descriptive of two frozen corpora and of nothing
else. The limitation, stated once and applying to every row: each
configuration was captured **once**, n = 30 scored bodies, same
candidate and serving; sampling variance between two runs of one
configuration is unmeasured, so absence in 30 bodies does not
prove a form impossible under a configuration, presence does not
prove the configuration caused it, and no row below licenses a
causal claim about the v2 contract, the sibling teaching, or the
candidate. The contract-shape observation in the last paragraph
is a fact about the two contracts, not about the model.

| Form (bodies carrying it) | SI-94 (v1) | SI-100 (v2) |
|---|---|---|
| `fs_search` intent missing required `pattern` | 0 | 3 |
| `fs_search` intent carrying `max_bytes` | 0 | 6 |
| `max_bytes` carried where undeclared (any tool) | 1 | 7 |
| `fs_edit` missing `new`, carrying `content` | 0 | 2 |
| any intent with a registry violation | 1 | 9 |
| `depends_on` entry citing seq 0 (dict or bare int) | 2 | 11 |
| `depends_on` entry citing seq 1 | 3 | 7 |
| `depends_on` entry naming a sibling intent's tool | 1 | 1 |
| `depends_on` entry citing the item's own task seq | 1 | 0 |
| sibling entry (dict form) | 0 | 2 |
| body opening with a `Version N` narration line | 1 | 3 |
| intent missing required `reason` | 1 | 0 |
| intent carrying the hyphenated field `depends-on` | 0 | 1 |

Contract fact, not a causal claim: under the v1 contract the
sibling key is refused at stage 2 (version-scoped field set,
SI-97 §5.3), so the two sibling entries in the v2 corpus belong
to a surface that did not exist in the configuration SI-94 was
scored under; both entries are nonetheless self-references, the
one citation the form forbids outright. Everything else in the
table is two-sample description under the limitation above.

## Control group — the 9 parse-ok bodies

All nine are **single-intent** proposals with an **empty
`depends_on`**: `fs_list` ×6 (c2-injection-shaped-3,
c2-multi-cycle-read-0, -1, -3, c2-depends-on-chain-0, -3),
`fs_read` ×2 (c2-depends-on-chain-4, c2-perception-truncation-2),
`fs_write` ×1 (c2-injection-shaped-0). The harness diagnostics
agree (batch sizes among parse-ok all 1; signal kinds all
intents_only). The control shape is the same as SI-94's — success
in both corpora is single-step and dependency-free — at 9 bodies
instead of 20. No parse-ok body uses any `depends_on` form, seq
or sibling.

## Honesty table

| Claim | State after SI-101 |
|---|---|
| SI-98 implementation of the §9 amendment | LANDED (PR #106, main `350cdf1`) — untouched by this slice |
| SI-99 output-contract amendment (teaches §8 v2) | LANDED (PR #107, main `17e889d`) — untouched by this slice |
| SI-100 conformance (v13 + §8 v2 contract) | MEASURED, BELOW THRESHOLD (9/30 = 0.30; per-workload floors unmet on four workloads) — stands as recorded; this taxonomy classifies, it does not re-cut |
| Correct sibling-form use in the SI-100 corpus | MEASURED: **0 of 30 bodies** (2 entries, both self-references, both shadowed behind stage 3) |
| Cross-configuration attribution | DESCRIPTIVE ONLY — form-presence counts over two frozen corpora; sampling variance unmeasured; no causal claim made |
| §18.3 claim | **NOT earned** |
| Live model-authored intent (Gaia loop) | Occurred, scoped to SI-74 only (2026-11-02); this slice attempts nothing live |
| P-5 rows (R1–R9) | UNMEASURED |
| C-4 for v13 | Does not proceed on the SI-94 or SI-100 record |
| L-3 | LOCKED |
| A-2 / A-3 / A-4 | UNAPPROVED |

## What this slice does not do

It does not re-score SI-100 (the corpus was scored once, by the
C-1 harness, and the verdict stands), does not re-open SI-99's or
SI-98's diffs, does not recommend for or against any further
contract teaching, and does not design the successor measurement.
It prices no proposal, contacts no model, and amends no SI-96
decision. Whether the programme revises the teaching, withdraws
it, or routes elsewhere is the lead's frontier decision, taken on
this evidence; the taxonomy takes no position in it.

## Honesty labels

§18.3 claim: NOT earned · L-3: LOCKED · A-2/A-3/A-4: UNAPPROVED ·
SI-100 verdict: BELOW THRESHOLD, stands as recorded · sibling
form correctly used in the measured corpus: 0 bodies ·
attribution: two-corpus description only, no causal claim ·
unlocks: [].
