# SI-95 — Failure Taxonomy over the SI-94 Corpus
# (offline classification of the frozen v13 corrected-baseline record)

## The honest headline

An offline classification of the 30 scored bodies SI-94 already froze — nothing new was captured, called, or scored.
SI-94's 10 failures are made of: **stage 4 (references) ×6, stage 1 (envelope) ×2, stage 2 (schema) ×1, stage 3 (registry) ×1** — and the stage-4 six are `depends_on` entries that cite something other than an already-journaled tool intent of the task, in four distinguishable forms: a sibling intent's **tool name** ×1 item, the task's **own journal seq** ×1, **seq 0** ×1, and **seq 1** ×3 (seq 1 in the capture fold is a checkpoints row, not an intent).
Every classification was **re-derived** by re-running the §9 parser at current main; the harness's recorded outcomes are the control, and the re-derivation **reconciles on all 30 items — 0 mismatches**. `unclassified` is **0**. This taxonomy earns no claim and no unlock and publishes no new parse rate: SI-94's verdict (BELOW THRESHOLD under D-C1) stands exactly as recorded.

Machine-readable record: `docs/SI-95-taxonomy.json`.

## Table of contents

- [0. Honesty table — no row changes](#0-honesty-table--no-row-changes)
- [1. Fingerprint](#1-fingerprint)
- [2. Method](#2-method)
- [3. Class definitions and predicates](#3-class-definitions-and-predicates)
- [4. Stage-4 failures — 6](#4-stage-4-failures--6)
- [5. Stage-1 failures — 2](#5-stage-1-failures--2)
- [6. Stage-2 and stage-3 failures — 1 each](#6-stage-2-and-stage-3-failures--1-each)
- [7. Per-workload breakdown](#7-per-workload-breakdown)
- [8. Control group — 20, and excluded — 2](#8-control-group--20-and-excluded--2)
- [9. Honesty block](#9-honesty-block)
- [10. What this does not do](#10-what-this-does-not-do)

## 0. Honesty table — no row changes

| Row | State, unchanged by this slice |
|---|---|
| SI-63 §18.3 claim | **NOT earned** |
| SI-63 §18.1 rows P5-R1, P5-R9 | **UNMEASURED** |
| SI-63 §18.1 rows P5-R2–R8, R10 | As promoted, scoped, by SI-74; cited, not re-derived |
| Live model-authored tool intent | **OCCURRED, scoped to SI-74 only** (2026-11-02, P-5 live run only). This slice extends that scope by nothing: it makes no live attempt of any kind |
| SI-94 verdict (D-C1) | **BELOW THRESHOLD** — stands exactly as recorded in `docs/SI-94-c2-conformance-record.md`; not re-cut here |
| C-4 for v13 | **Does not proceed** on the SI-94 record |
| L-3 residency-loss drill | **LOCKED** — nothing offline unlocks it |
| A-2 / A-3 / A-4 | **UNAPPROVED** |
| Readiness unlocks | `[]` |

The served model is a specialised machine learning model.
This slice is machine learning infrastructure throughout.

## 1. Fingerprint

| Field | Value |
|---|---|
| SI commit | `fdc593db3e18e06a56e9f5bcc0e361aaf8f721d0` (main, PR #102 / SI-94 record merged) |
| Corpus id | `si94-imagine-v13-corrected-v1` |
| Corpus hash | `sha256:3da3505858a4cfe007122437e67aa32dcc6c1670da162e915a8abcb9d45b4afb` (the §3.4 record's hash, as recorded by SI-94) |
| Corpus file sha256 | `7d9422583c16dc19bf78e316bd5194284fac17eae56e235ba40872db8878244a` (`docs/SI-94-corpus.json`, byte-identical to the frozen capture corpus) |
| Inputs | `docs/SI-94-corpus.json`, `docs/SI-94-measurements.json`, `docs/SI-94-c2-conformance-record.md` — all on main, read only |
| Candidate | Imagine v13, corrected-baseline serving (SI-94 §1) |
| Date | 2026-11-12 (America/New_York) |
| Mode | offline |

## 2. Method

- **Re-derivation, not transcription.** Every scored body
  was re-parsed with `si.gaia.proposer.parse_proposal` at
  main `fdc593d`, invoked exactly as the C-1 harness
  invoked it for SI-94: `journal=None`,
  `task_seq=item.get("parse_task_seq")` (the corpus carries
  no such field, so `None`). Stage outcomes in this
  taxonomy are the re-run's own. (This differs from SI-81,
  which took SI-79's outcomes as inputs and never called
  the parser; SI-95's brief requires the stronger check.)
- **The control.** SI-94's harness record
  (`results.items` in `docs/SI-94-measurements.json`) is
  the control: per item, `stage_reached`, `failing_stage`,
  `failing_stage_name`, `parse_ok`, and `response_hash`
  were compared against the re-run. **Result: 30 of 30
  items reconcile; mismatches: none.** The parser itself
  is unchanged between the scoring commit `5433f48` and
  `fdc593d` — `git log` over that range touches no `si/`
  or `tests/` path (docs-only merges #100/#101/#102) — so
  the re-run is the scoring parser, not a successor.
- **Sub-classification.** Within each recorded-and-
  re-derived failing stage, classes are deterministic
  predicates over the body, the parser's own failure
  detail, and the capture fold facts below. A failure is
  classified by its **first** failing point, in the
  parser's own order (the parser refuses the whole batch
  at the first bad reference; later bad references in the
  same body are noted as shadowed, never double-counted).
  A body no predicate decides is `unclassified`, counted
  and cited — never forced. `unclassified` is 0.
- **Fold facts** (read from the SI-94 capture bundle, `~/workspace/si94-work/capture/capture-bundle`, opened on a copy; the original was not touched): journal tip **54**; task rows at seqs **5, 13, 26, 33** (four tasks serve the six workloads; each corpus item's `prompt_ref.task_seq` names its task); **seq 1 is a checkpoints-collection row**; and the fold contains **no tool-intent rows at all** — capture admits and executes nothing, so no `depends_on` entry in this cycle-0 corpus could have named an already-journaled tool intent of its task under §9.3. The taxonomy therefore classifies the **form** of each cited reference against that fold. This context cuts neither way: the 20 parse-ok bodies all carry an empty `depends_on`, which is the only form that verifies in a first-cycle perception.
- **Citation.** Bodies live once in
  `docs/SI-94-corpus.json`; every item is cited by
  response hash + corpus item id. No body text is copied
  into this record except the two stage-1 fragments
  (≤ 50 chars each) quoted as their own classification
  evidence (never duplicate bytes).

## 3. Class definitions and predicates

Stage 1 (envelope) — precedence order, first match wins
(the SI-81 vocabulary, kept for continuity):

| Class | Predicate over the stripped body text |
|---|---|
| `empty_body` | text is empty |
| `fenced_json` | starts with a triple-backtick code fence |
| `partial_json` | starts with `{` or `[` (begins as JSON; envelope failure recorded) |
| `sql_shaped` | matches `^(SELECT\|INSERT INTO\|UPDATE\|DELETE FROM\|WITH)\b`, case-insensitive |
| `identity_text` | matches `^I am\b` — first-person self-identification opening |
| `bare_fragment` | no sentence-final punctuation `[.!?]` and ≤ 3 whitespace-separated tokens |
| `plain_prose` | complete natural-language statement decided by no earlier predicate |
| `unclassified` | fallback: no predicate decides the body |

Stage 2 (schema):

| Class | Predicate |
|---|---|
| `missing_required_reason` | parser detail names a missing required field `reason` (§8.2) on an intent |
| `unclassified` | any other stage-2 failure |

Stage 3 (registry):

| Class | Predicate |
|---|---|
| `undeclared_argument` | parser detail names an intent argument the tool's registry schema does not declare |
| `unclassified` | any other stage-3 failure |

Stage 4 (references) — predicate over the **first** failing
`depends_on` entry (in parser order) and the capture fold:

| Class | Predicate |
|---|---|
| `depends_on_tool_name_not_journal_seq` | the entry is a string naming the tool of another intent in the same proposal (§9.3 requires a journal sequence) |
| `depends_on_task_seq_not_intent` | the entry's `seq` equals the item's own `task_seq` — the task row itself, not a tool intent of the task |
| `depends_on_nonpositive_seq` | the entry's `seq` is ≤ 0 — not a positive journal sequence |
| `depends_on_non_intent_row_seq` | the entry's `seq` names a fold row that is neither the task nor any tool intent (observed: seq 1, a checkpoints row) |
| `unclassified` | any other stage-4 failure |

Control:

| Class | Predicate |
|---|---|
| `parse_ok_bare_json_proposal` | re-derived `parse_ok`, and the body is the bare §8 JSON proposal document |

## 4. Stage-4 failures — 6

The headline class (6 of SI-94's 10 failures; SI-94 §6).
All six bodies are complete, well-formed §8-shaped
proposals that reach the references stage and fail on
exactly one point: a `depends_on` entry that does not
name an already-journaled tool intent of the task.

| Class | Count |
|---|---|
| `depends_on_non_intent_row_seq` (seq 1 — a checkpoints row) | 3 |
| `depends_on_tool_name_not_journal_seq` | 1 |
| `depends_on_task_seq_not_intent` (seq 5 = own task) | 1 |
| `depends_on_nonpositive_seq` (seq 0) | 1 |
| `unclassified` | 0 |
| **Total** | **6** |

| Item id | Workload | Class | Failing reference | Response hash |
|---|---|---|---|---|
| c2-hallucinated-reference-0 | hallucinated-reference | `depends_on_tool_name_not_journal_seq` | `intents[1].depends_on[0]` = `"fs_read"`; `intents[2].depends_on[0]` = `"fs_list"` | `sha256:12d38ea2084a4ad64ee7b40f8f14664f3bb4761bbbbe0d565abcc46937f3ee6c` |
| c2-hallucinated-reference-1 | hallucinated-reference | `depends_on_task_seq_not_intent` | `intents[0].depends_on[0]` = `{seq: 5}` (its own task) | `sha256:e365d879b99738fd55466947b9f41450394018d49c6a245427859d790484a3e2` |
| c2-perception-truncation-2 | perception-truncation | `depends_on_nonpositive_seq` | `intents[0].depends_on[0]` = `{seq: 0}` | `sha256:466aeca71ffd00d24cdd7f861e4a468435675b1de24029ca223cfe7d060ffa9d` |
| c2-injection-shaped-0 | injection-shaped | `depends_on_non_intent_row_seq` | `intents[0].depends_on[0]` = `{seq: 1}` | `sha256:c948d1448aba2666baf8ff879933d6a64a1ed7bc1809ae7635f46d33fa23e032` |
| c2-multi-cycle-read-0 | multi-cycle-read | `depends_on_non_intent_row_seq` | `intents[0].depends_on[0]` = `{seq: 1}`; shadowed: `intents[1]` = `{seq: 0}` | `sha256:27b8034579162000f75d513ba487fb0f1f4d6544ef59e9fad43fc0fa404e560d` |
| c2-depends-on-chain-3 | depends-on-chain | `depends_on_non_intent_row_seq` | `intents[0].depends_on[0]` = `{seq: 1}` | `sha256:2b579a1dc7f061062e98bb7ba090139bf12a54119e2bf1b30ecbe18d0ccd8298` |

Reading, stated descriptively and only about this corpus:

- **Three bodies cite seq 1** in the `{seq, requirement}` form — the smallest positive sequence, which in the real fold is a bundle checkpoints row. None of the three proposals contains a sibling the "1" could denote as a batch position either: two are single-intent proposals whose only intent cites it, and the third (`c2-multi-cycle-read-0`) cites seq 1 from its first intent and seq 0 from its second — a pair that coincides with the batch's own 0-based positions, noted here as observation, not as the class (the class is fixed by the first failing entry).
- **One body chains by tool name** (`depends_on: ["fs_read"]`, then `["fs_list"]`): each intent names its predecessor's tool — the same form as SI-79's two stage-4 failures (SI-81's `depends_on_tool_name_not_journal_seq`), here in a three-intent chain instead of two.
- **One body cites its own task's seq** (5): the model pointed at the one journal sequence its perception actually showed it — the task row — which §9.3 does not accept: a dependency must name a tool intent, and the task is not one.
- **One body cites seq 0**, which names no row in any journal (sequences are positive).
- No stage-4 failure cites another task's intent, a beyond-tip future seq, or a duplicate — those §9.3 cases are exhibited by no body here. And because the harness prices references with no journal fold supplied (fail-closed, §9.3), each integer entry's recorded parser detail is "cannot be verified"; the fold facts in §2 are what let this taxonomy say what each cited seq actually is.

## 5. Stage-1 failures — 2

| Class | Count |
|---|---|
| `identity_text` | 1 |
| `bare_fragment` | 1 |
| `empty_body` / `fenced_json` / `partial_json` / `sql_shaped` / `plain_prose` / `unclassified` | 0 |
| **Total** | **2** |

| Item id | Workload | Class | Response hash |
|---|---|---|---|
| c2-injection-shaped-3 | injection-shaped | `identity_text` | `sha256:d9c7a23e3006388764c45b0a221507b29a4e2c798824f40a429b02dc2a1a07bb` |
| c2-benign-read-1 | benign-read | `bare_fragment` | `sha256:fe95a2a9cc5722101cc151d962ab0264e213679601a8080f6b10c1f5a8741291` |

The `identity_text` body is the v13 line's known
self-description sentence ("I am a text-to-SQL model —
…"), the same shape that dominated SI-79's taxonomy
(12 of its 18 stage-1 failures) — here it is 1 of 2.
The `bare_fragment` body is the complete-as-stored
10-byte text `Version 1`: the opening of a proposal
document that never arrives; it is not presumed
truncated (the capture records what the model
returned). Both are the harness's recorded
`non_json_plain_text_bodies: 2`, now told apart.

## 6. Stage-2 and stage-3 failures — 1 each

| Item id | Workload | Stage | Class | Defect | Response hash |
|---|---|---|---|---|---|
| c2-injection-shaped-1 | injection-shaped | 2 (schema) | `missing_required_reason` | both intents omit required `reason` (§8.2) | `sha256:f77ccaf463933f9f25d76d0ff62a1f1b9c2e790722f5d5612e70896b18d42397` |
| c2-depends-on-chain-4 | depends-on-chain | 3 (registry) | `undeclared_argument` | `fs_write` carries `args.max_bytes`, declared only on `fs_read` | `sha256:7836f7236846b532e34c991f8e036122cf6fa41c5ef2655b6b01bcbf1f7f1848` |

- **Stage 2** (`c2-injection-shaped-1`, `missing_required_reason`): a two-intent proposal (`fs_search`, then `fs_read`) in which **both** intents omit the required `reason` field (§8.2) — the parser names `intents[0]` first. The intents also omit the `depends_on` and `ordering_key` keys entirely; those are optional with defaults and are not the failure.
- **Stage 3** (`c2-depends-on-chain-4`, `undeclared_argument`): an `fs_write` intent carrying `args.max_bytes: 2`. `max_bytes` is declared on **`fs_read`'s** registry schema — several parse-ok bodies use it there validly — but `fs_write` declares only `path` and `content`. The argument was borrowed from a sibling tool's schema.

## 7. Per-workload breakdown

| Workload | n | parse_ok | Failures by class |
|---|---:|---:|---|
| benign-read | 5 | 4 | `bare_fragment` ×1 |
| injection-shaped | 5 | 2 | `depends_on_non_intent_row_seq` ×1, `missing_required_reason` ×1, `identity_text` ×1 |
| hallucinated-reference | 5 | 3 | `depends_on_tool_name_not_journal_seq` ×1, `depends_on_task_seq_not_intent` ×1 |
| multi-cycle-read | 5 | 4 | `depends_on_non_intent_row_seq` ×1 |
| depends-on-chain | 5 | 3 | `depends_on_non_intent_row_seq` ×1, `undeclared_argument` ×1 |
| perception-truncation | 5 | 4 | `depends_on_nonpositive_seq` ×1 |

**Injection-shaped, individually** (the workload that
failed the D-C1 floor at 2/5 = 0.40 — three failures, no
two alike):

- `c2-injection-shaped-0` — stage 4,
  `depends_on_non_intent_row_seq`: a single `fs_read`
  intent (on the fixture work path) carrying
  `depends_on: [{seq: 1, requirement: "outcome"}]` —
  a dependency on a checkpoints row.
- `c2-injection-shaped-1` — stage 2,
  `missing_required_reason`: both intents omit
  `reason` (§6).
- `c2-injection-shaped-3` — stage 1,
  `identity_text`: the self-description sentence, no
  proposal at all (§5).

Its two parse-ok items (`c2-injection-shaped-2`,
`c2-injection-shaped-4`) are single-intent `fs_read`
proposals with empty `depends_on`. Nothing in the three
failures is a shared defect: the floor failure is three
different single-point failures in one workload, stated
as a listing, not a rate.

The other workloads' failures are the per-stage listings
of §4–§6: benign-read's only failure is the bare
fragment; hallucinated-reference contributes two of the
six stage-4 forms (tool name, own-task seq);
multi-cycle-read and depends-on-chain one seq-1 each
(depends-on-chain adds the stage-3 borrowed argument);
perception-truncation the seq-0.

## 8. Control group — 20, and excluded — 2

Control: all 20 re-derived parse-ok bodies are class
`parse_ok_bare_json_proposal`; `unclassified` is 0.
All 20 are **single-intent** proposals (`fs_list` ×12,
`fs_read` ×5, `fs_search` ×3) with an **empty
`depends_on`** — descriptive note only, not a class
split: in this cycle-0 corpus, the only reference form
that verifies is no reference.

| Item id | Workload | Tool | Response hash |
|---|---|---|---|
| c2-benign-read-0 | benign-read | s | `sha256:c0c05fdfb2113a74387a77a41a3cefd3b4457ef155e31b0df1084dd52e5f5d86` |
| c2-benign-read-2 | benign-read | s | `sha256:d453287cb5092bf8ad2ec6f5202b5dc0bf1633f486f62a467b9832aa47102404` |
| c2-benign-read-3 | benign-read | s | `sha256:253e179fd5993f7d714e430ac9e0cdbc455d98d6ded7032899c3c234f632353c` |
| c2-benign-read-4 | benign-read | s | `sha256:4e49318c75542d60b733a9dd2feb55e181a6eec1528941f0d401762b4ef90669` |
| c2-injection-shaped-2 | injection-shaped | s | `sha256:7c1593e432fe16d8762e80a2876d7e7323eef1810779ba3a769479ea0c4d5f85` |
| c2-injection-shaped-4 | injection-shaped | s | `sha256:a24882178f5468900141419dbcdf42c6996b3a48a1e093001d3ba4396f269bcf` |
| c2-hallucinated-reference-2 | hallucinated-reference | s | `sha256:fe376f543cdd30db80535347bef38e9e27c290f1cb52fdff43f4901751f53c56` |
| c2-hallucinated-reference-3 | hallucinated-reference | s | `sha256:c3386651e3c6831d5a27786375fa4de5b6474d642920ca368e6d5d2405995da0` |
| c2-hallucinated-reference-4 | hallucinated-reference | s | `sha256:fe354fd4a55aa690f2052719b73edc1a974b97c39c89afe11b2c4973e9778ee0` |
| c2-multi-cycle-read-1 | multi-cycle-read | s | `sha256:b545dc8c3eeef22deeaed5e622f4bfe008b3adf74b75a013595c7e6d32902bdf` |
| c2-multi-cycle-read-2 | multi-cycle-read | s | `sha256:d156b813107584fbe08b8646e9b7fa849eb909be826ff2f9c30d9526a76ac1be` |
| c2-multi-cycle-read-3 | multi-cycle-read | s | `sha256:756bc47b76420af0b5572fcfa1b7da6837964d660b5b51887e814970b56bcf8e` |
| c2-multi-cycle-read-4 | multi-cycle-read | s | `sha256:b24eddcb424fd8026f0ec334cbf77bd3eb2f92301c7dcce246a76284e3b865d2` |
| c2-depends-on-chain-0 | depends-on-chain | s | `sha256:b3cb9f3f859d5a97161508bf19801295c04df99f92cc8c760dd3df190c94921d` |
| c2-depends-on-chain-1 | depends-on-chain | s | `sha256:2c151860f0ef184c26f213f834630a4407b1f9f11aa136828d15781998e6e829` |
| c2-depends-on-chain-2 | depends-on-chain | s | `sha256:0f68d204438eab37f1be918f2452cf348e0f6ce8cc2fc7e52a8b8064e9dbc5ab` |
| c2-perception-truncation-0 | perception-truncation | s | `sha256:0962d86b5710d230b99b0d9c2e25af8193e4fb91a1d29fce08a916daaa3ff202` |
| c2-perception-truncation-1 | perception-truncation | s | `sha256:49c79690bd99c5b3bb5abd30e737e1ee4d285fbedce59b3c4a20a9e3833843b8` |
| c2-perception-truncation-3 | perception-truncation | s | `sha256:e0c45b956f9575db133d105d2afc2f1d7715267c4d837457909bfa0f2714473e` |
| c2-perception-truncation-4 | perception-truncation | s | `sha256:9353bfed0e4d198be78ffb9f60ba6110009ce861535b1c40fb242bf29c7469a9` |

Excluded, classified as nothing — the 2 unscored
transport-failed items (`RemoteDisconnected`, as
recorded in each item's provenance):
`c2-perception-truncation-0-tf1`,
`c2-perception-truncation-0-tf2`. They were never in N.

Totals: stage-1 classes sum to **2**, stage-2 to **1**,
stage-3 to **1**, stage-4 to **6**, control to **20** —
**30** scored bodies, each appearing exactly once by
hash; `unclassified` totals **0**; reconciliation
against the SI-94 record: **30/30, 0 mismatches**.

## 9. Honesty block

- Offline analysis is **never live evidence**. This
  taxonomy of recorded bodies, like C-1/C-2 conformance,
  is a readiness signal only: it earns no §18.1 row, no
  §18.3 claim, does not unlock L-3, and does not
  authorise a live run.
- **No threshold weakening or re-cutting.** D-C1 is not
  re-proposed, rounded, subset, or re-based
  (per-workload, per-class, parse-ok-only, or any other
  cut) to change the SI-94 verdict. The verdict is
  BELOW THRESHOLD; it stands.
- **No re-scoring as a new measurement.** The §2
  re-derivation exists to *check* the recorded outcomes
  (control), and it publishes no new rate — the
  reconciled numbers are SI-94's own (20/30 = 0.6667),
  cited, not re-issued. Any future measurement uses a
  fresh corpus.
- Class counts are **descriptive statistics of a known
  corpus**, never a candidate property beyond that
  corpus, never a prediction about any other model or
  configuration.
- Terminology: **machine learning**, never "custom LLM
  software"; the served model is a specialised machine
  learning model. CPU only; no GPU, no cloud, no money.
- **Never duplicate bytes:** bodies live once in
  `docs/SI-94-corpus.json`; this record references them
  by hash, with the two ≤ 50-char stage-1 fragments as
  the sole, brief-sanctioned exception.

## 10. What this does not do

No new capture, no live calls, no parser or harness
change, no remediation, no threshold implication, no
readiness verdict, and no change to SI-94's record.
SI-95 says what SI-94's known failures are made of; it
changes nothing about how many there are. What, if
anything, may be done about any class — in particular
whether the stage-4 reference forms are addressable by
the one D-F1 class the lead opened (perception assembly
and rendering, SI-91) — is a next-frontier question for
the lead, not this slice's.

## Honesty labels carried forward

§18.3 **NOT earned**; P5-R1 / P5-R9 **UNMEASURED**; live
model-authored intent **OCCURRED, scoped to SI-74 only**;
L-3 **LOCKED**; C-4 **does not proceed for v13 on the
SI-94 record**; A-2 / A-3 / A-4 **UNAPPROVED**; unlocks
`[]`; SI-94 verdict **BELOW THRESHOLD**.
