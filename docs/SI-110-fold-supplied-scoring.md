# SI-110 — Fold-supplied scoring support, in-repo (SI-106 §12 item 2)

## What this slice is

SI-106 §12 sequenced, as its second future slice,
fold-supplied scoring support implementing §5.1's procedure.
SI-108 §4.2 proposed it as SI-110, and lead decision D-108-1
(2026-11-13) answered CONTINUE — SI-109 first (the multi-cycle
capture driver, merged), then this slice. It lands in the
conformance harness (`si/gaia/conformance.py`), sequenced
after SI-109 only because §12 orders it so; it shares no code
path with the driver beyond the frozen artefacts' shapes.

SI-107 scored its frozen corpus fold-supplied from workspace
scripts under the lead's direct dispatch. **Landing this
support does not retroactively make SI-107 a repo-code run,
and SI-107's frozen corpus is not re-scored here** — its
workspace scoring stands as its record (SI-108 §4.2
non-goals). This slice earns reproducibility only: no §18.1
row, no §18.3 progress, no rate for any candidate, and no
claim about any model (SI-108 §4.3).

## What landed

- **The §5.1 procedure** (`score_corpus_fold_supplied`,
  `build_fold_supplied_record`): the corpus record carries
  the frozen fold's sha256 beside the corpus sha256 (the
  SI-109 freeze record's shape, riding on the corpus as
  `corpus["fold"]`); scoring supplies the frozen fold — the
  capture's own journal, opened by the caller, and no other —
  as the `journal` of every scoring call; both hashes are
  recorded in the scoring record's fingerprint and its
  `fold_supplied` section.
- **The task_seq mapping as procedure** (`parse_task_seq`,
  §5.1 step 4): for each scored document, scoring (a) reads
  the owning item's `prompt_ref.task_seq` from the frozen
  corpus record, (b) passes it as `task_seq` together with
  the frozen fold as `journal` to the scoring call, and
  (c) records both values' hashes in the scoring record —
  the fold sha256, and per item the mapped task seq with its
  hash.
- **Fail-closed on fold-hash mismatch** (§5.1 step 2;
  §4.2 acceptance 1): the supplied fold's sha256 is
  recomputed over the supplied journal itself (the SI-109
  `fold_rows_and_hash` procedure), never taken on trust. Any
  mismatch — recomputed vs corpus-carried, or caller-stated
  vs recomputed — raises `FoldScoringRefused` before any
  document is scored; the record form is `SCORING REFUSED`
  with `results: None` and no rate published anywhere. There
  is no fallback to no-fold scoring, and a corpus carrying
  no fold hash cannot be scored under this support at all.
- **Corpus defects** (§5.1 step 4; §4.2 acceptance 2): a
  scored document whose item carries no `prompt_ref.task_seq`
  is recorded as a corpus defect and excluded from M —
  never silently scored task-less. M is the §3.2 fold's `n`
  over the documents actually scored; the defects ride in
  the record beside it.
- **Both reference forms, priced separately** (§6;
  §4.2 acceptances 3–4): the record's `reference_forms`
  split counts seq-form entries (priced against the frozen
  fold), sibling-form entries (priced document-locally at
  parse, SI-97 §5 — the fold is never consulted for them),
  and other-shaped entries, each with its own
  occurrences / passed / refused line, attributed from the
  parser's own results in iteration order (entries after a
  stage-4 refusal are unpriced, never estimated; references
  of a document refused before stage 4 are unpriced, §6.4).
  A sibling occurrence count of 0 is reported as 0, never
  folded into the seq-form line. A foreign seq — a fold row
  this capture did not journal for this task — fails stage 4
  exactly as a nonexistent seq does (§13, first risk); the
  parser's scope check is what prices it, unchanged.
- **Comparability, stated in the record itself** (§5.1
  step 5; §4.2 acceptance 6): every fold-supplied record
  carries, in its `fold_supplied` section and its honesty
  block, the statement that fold-supplied pricing is a
  different quantity from every no-fold cycle-0 rate —
  comparability none, in either direction. No bar, no
  verdict, and no unlock is computed for a fold-supplied
  corpus (SI-106 §3).

## What did not change

No-fold scoring is byte-for-byte unchanged (§4.2
acceptance 5): `score_corpus` and `build_conformance_record`
are untouched, the existing conformance tests pass
unmodified, and SI-94 / SI-100 / SI-105 are never re-scored
under this support (SI-106 §5, first paragraph). This diff
touches no gate pricing, no parser stage rule, and no
teaching text (`OUTPUT_CONTRACT_INSTRUCTION` untouched) —
per SI-106 §12, such a diff would fail this slice's own
review.

## Evidence (this slice's own runs)

New tests: `tests/test_gaia_conformance_si110.py` — 8
passed, covering §4.2 acceptances 1–6 as numbered in the
test file's docstring (the fold is a fixture capture journal
journaled by the SI-109 driver from scripted sources — no
live model, no live capture). Affected suites re-run
(conformance C-1, conformance capture C-2, multi-cycle
SI-109, proposer P-1/P-2, SI-104, SI-98) — counts in the
build evidence file outside this repo
(`~/workspace/si-dev-loop/si110-build-evidence.md`).

## Non-goals and honesty labels

No re-scoring of SI-107's frozen corpus as a new
measurement; no live capture and no authorisation to make
one; no new number for any candidate is published by this
slice. Carried forward, unweakened: §18.3 **NOT earned**;
P5-R1 / P5-R9 **UNMEASURED**; live model-authored intent
**OCCURRED, scoped to SI-74 only**; L-3 **LOCKED**;
A-2/A-3/A-4 **UNAPPROVED**; unlocks `[]`; D-C1 **stands**
for cycle-0 conformance and is not applied by anything
here.
