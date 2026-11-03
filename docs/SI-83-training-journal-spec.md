# SI-83 — NEDB-Journaled Training: Design Spec (SPEC ONLY)

> **Status: SPEC ONLY — this document builds nothing, changes no code,
> and runs no training.** It contains no code, no test, no corpus
> change, no capture, no live call, and no measurement. It designs a
> journal for training runs; it does not journal any run. Nothing in
> this doc is a claim that any training journal exists, runs, or has
> been verified. Where an interface does not exist in the repo today,
> this doc says so and designs it as PROPOSED.
> **Scope:** this slice runs nothing and spends nothing. The training
> run it designs for is Mark's live Imagine v13 run on the B200 box,
> which proceeds on its own runbook in parallel (§8); this doc does
> not run, steer, or claim to observe that run. This is machine
> learning infrastructure throughout; the served model is a
> specialised machine learning model, 1.6B parameters, full fine
> tune. Terminology rule, stated once and negated: this doc never
> calls what SI builds "custom LLM software".
> **Relationship:** Mark directed this work on 2026-11-03 (see
> `~/workspace/si-dev-loop/training-journal-mission-brief.md` in the
> loop's working notes; the mission brief itself is not a repo
> file): if NEDB tracked everything during a training run, then
> when a run "works but not how we wanted" we should be able to
> undo — roll back to a checkpoint, prune the training corpus from
> there, inspect, and reimagine. This spec is the first slice of
> that mission and nothing more. It presumes no answer to D-F1
> (§10) and proposes no F-line work of any kind.

## Table of contents

- [0. Honesty table](#0-honesty-table--carried-forward-no-row-changes)
- [1. Fingerprint and scope statement](#1-fingerprint-and-scope-statement)
- [2. Journal schema](#2-journal-schema)
- [3. Watcher design (non-invasive)](#3-watcher-design-non-invasive)
- [4. Rollback / replay / fork semantics](#4-rollback--replay--fork-semantics)
- [5. Corpus-window prune lineage](#5-corpus-window-prune-lineage)
- [6. Disk budget vs the v12 lesson](#6-disk-budget-vs-the-v12-lesson)
- [7. Database naming, explicit](#7-database-naming-explicit)
- [8. Adoption split: what the live v13 run can adopt vs what waits](#8-adoption-split-what-the-live-v13-run-can-adopt-vs-what-waits)
- [9. Proposed slice plan after this spec](#9-proposed-slice-plan-after-this-spec)
- [10. Honesty and status](#10-honesty-and-status)
- [11. Sources and code references](#11-sources-and-code-references)

## 0. Honesty table — carried forward, no row changes

SI-80 §3.2 requires every slice doc to update this table row by row
with evidence, or to state explicitly that no row changes. **No
honesty row changes in this slice.** This slice designs a journal;
a design is not evidence, and it earns, promotes, weakens, and
re-labels nothing. The rows below are carried forward from SI-82
§0, unchanged.

| Element | State | Evidence / scope |
|---|---|---|
| SI-63 §18.3 claim | NOT earned | SI-74, SI-75, SI-79 §0: no live task completed |
| Live model-authored tool intent | OCCURRED, scoped to SI-74 only | 2026-11-02, P-5 live run only; 2 parsed intents, `fs_list` ×2; 1 ADMIT + executed + verified, 1 ESCALATE + not executed. No other run, and no sentence here, extends that scope |
| §18.1 row P5-R1 | UNMEASURED | SI-74 §4 |
| §18.1 row P5-R9 | UNMEASURED | SI-74 §4; SI-79 §4 notes a Resident death as an observation only |
| §18.1 rows P5-R2–R8, R10 | As promoted, scoped, by SI-74 | Cited by SI-77/SI-79/SI-81, promoted by none of them, and by none of this |
| Readiness unlocks (SI-79) | `[]` | `docs/SI-79-measurements.json`: `readiness.unlocks`; this slice adds none — unlocks remain `[]` |
| C-2 conformance measurement | DONE, BELOW THRESHOLD | SI-79 (PR #86); 10/30 (0.3333) against D-C1 |
| L-3 residency-loss drill | LOCKED | SI-69 §4.5; SI-76 §3.6; nothing offline unlocks it |
| C-5 conditional L-3 path | LOCKED | SI-76 §4.7; no real L-2 live success |
| A-2 / A-3 / A-4 | UNAPPROVED | Not approved by SI-40, SI-63, SI-69, SI-76, SI-80, SI-81, SI-82, or this doc |
| D-F1 | OPEN | SI-80 §4; SI-82 §5 poses it; Mark's decision, not this slice's |
| F-line | STOPPED at D-F1 | No F-3+ work is built, designed, or proposed by this doc |

## 1. Fingerprint and scope statement

| Field | Value |
|---|---|
| SI commit (base) | `2e4758784e01e1a743961b32817e7b1edb3e9e33` (main, PR #89 / SI-82 merged) |
| Spec | This doc; first slice of Mark's 2026-11-03 training-journal directive |
| Informed by | `docs/nedb-journal-schema.md`, `docs/si-container-spec.md`, `docs/self-write-loop.md`, `si/journal.py`, `si/backends/nedb.py`, `si/seed.py` (by-reference seeding), `docs/SI-82-f2-admissibility-question.md` (§0 honesty format) |
| Date | 2026-11-03 (America/New_York) |
| Mode | design only — offline, no run of any kind |

Plainly, as SI-82's format requires this slice to state: **this
slice builds nothing, changes no rendering, and runs nothing.** No
file other than this doc is added or modified by the slice. In
particular, nothing in this doc is a training journal, a watcher,
a rollback tool, or a corpus prune. Those are PROPOSED slices in
§9, each needing its own build and its own verification before
any claim about it is made.

What exists today, and what does not — stated so the design is
not mistaken for a description of the present:

- EXISTS: the SI-0 journal (`si/journal.py`): five collections
  (`self_edits`, `deltas`, `checkpoints`, `provenance`,
  `experiences`), a runtime-assigned logical clock (`seq`), a
  hash chain (`prev_hash`, `entry_hash`), fail-closed chain
  verification, and a real-NEDB backend over pgwire
  (`si/backends/nedb.py`) with an embedded sqlite fallback.
- EXISTS: by-reference seeding (`si/seed.py`, SI-8): a manifest
  that records per-tensor external references
  (`seed_format="reference-v1"`, `base.tensor_refs`) with
  `stored_bytes` 0, re-verified on every load. The Glimmer seed
  bundle is the standing example: stored bytes 0, referenced
  bytes 59,553,253,376.
- EXISTS: miniature training (`si/tiny_training.py`, SI-19):
  gradient-based training of one LoRA target on a miniature
  model, gated exactly like any other candidate. This is NOT an
  Imagine training loop and NOT an eval suite for a 1.6B model.
- DOES NOT EXIST: any training-run journal, any checkpoint
  watcher, any training checkpoint manifest, any training eval
  moment, any corpus-window map, any rollback/replay/fork tool
  for training, any training-journal database. Every such item
  in §§2–9 is PROPOSED.

## 2. Journal schema

### 2.1 Design rules, inherited, not re-invented

The training journal extends the SI-0 journal; it does not fork
it. Four rules carry over verbatim in intent:

1. **NEDB stores pointers, not weights.**
   `docs/nedb-journal-schema.md` states it plainly: "NEDB never
   stores raw weight bytes — it stores hashes that point at
   them." A training checkpoint manifest therefore records the
   weights blob's content hash and its external path, never the
   weights. SI never duplicates bytes (Mark's design rule): a
   checkpoint that is already on disk is referenced, not copied.
2. **One logical clock.** Every entry carries `seq`, a
   monotonically increasing integer assigned by the runtime
   (not the trainer, not the model). Wall-clock `ts` is
   recorded alongside but never orders anything; `seq` is what
   rollback and "as of" queries use.
3. **Hash chain, fail closed.** Every entry carries `prev_hash`
   and `entry_hash = sha256(canonical(prev_hash | seq |
   payload))`, exactly as `si/journal.py::_entry_hash` computes
   it, with `GENESIS_PREV = "genesis"` for the first entry.
   Operational state (`result`, `delta_id`, `status` in the
   SI-0 journal's `MUTABLE_KEYS`) is excluded from the hashed
   payload and may change without rewriting history; the causal
   record (intent, reason, gate verdict/report) is
   chain-protected and immutable. The chain is verified on load
   (`Journal.verify_chain`); a broken chain refuses to run.
   A training journal that cannot trust its own memory must
   not drive a rollback.
4. **Gate before landing, for every state-changing proposal.**
   The self-write loop (`docs/self-write-loop.md` §3) gates
   before anything is written, with ternary verdicts
   `ADMIT` / `REJECT` / `ESCALATE`, and journals the intent and
   the verdict *before* materialization (§4). A corpus prune
   (§5) and a retention decision (§6) are proposals in exactly
   this sense. A checkpoint observation, by contrast, is not a
   proposal: the watcher journals what the trainer did; it does
   not ask permission for the trainer's save, and it never
   blocks one.

### 2.2 Proposed collections (NEW — none exists today)

The SI-0 journal's five collections do not cover training.
This spec PROPOSES five new collections, in the same NEDB
journal (see §7 for which database), with the same seq/chain
machinery. Names are PROPOSED; a later implementation slice
may rename them, but it must not silently merge training rows
into the SI-0 collections, because training moments and
self-edit decisions answer different questions and must stay
separately queryable.

| Collection | One row per | Key purpose |
|---|---|---|
| `training_runs` | training run (one lineage root) | Identity of a run: model, corpus, hyperparams, trainer commit, lineage parent |
| `training_checkpoints` | checkpoint manifest (one per saved step) | Where the weights were, by reference, at a step |
| `training_eval_moments` | eval moment (one per checkpoint evaluated, or per eval event) | What the model scored, and what it generated, at that moment |
| `training_windows` | consumed example window (one per step range) | Which exact examples the run consumed, in which order, between two steps |
| `training_proposals` | gated proposal (prune, retention, fork) | The proposal, the gate verdict, the operator, the outcome — the audit trail |

A `training_runs` row is the anchor every other row points at
(`run_id`). A fork (§4) creates a new `training_runs` row whose
`parent` is the source run's checkpoint, not a copy of the
source run's rows: both lines are kept, and provenance edges
(§2.5) link them.

### 2.3 Checkpoint manifest, by reference

One `training_checkpoints` row per saved checkpoint. Required
fields (all PROPOSED; the field names are this spec's design,
not an existing API):

| Field | Type | Notes |
|---|---|---|
| `run_id` | string | FK → `training_runs.run_id` |
| `step` | int | trainer step at save; the checkpoint's identity within the run |
| `epoch` | float | epoch at save (fractional epochs allowed; the trainer's own numbering, recorded verbatim, never renormalised) |
| `weights_blob` | string | `sha256:<hex>` over the checkpoint's weight bytes — the content address. The journal stores the hash, never the bytes |
| `weights_path` | string | external path of the referenced blob (trainer checkpoint dir, e.g. `step<N>/`); the reference, not a copy |
| `weights_bytes` | int | size of the referenced weights file, in bytes |
| `optimizer_blob` | string \| null | `sha256:<hex>` of the optimizer state file, or `null` when the checkpoint retains weights only (§6: optimizer state is the disk cost, and not every checkpoint keeps it) |
| `optimizer_bytes` | int \| null | size of the optimizer state file, or `null` |
| `corpus_hash` | string | `sha256:<hex>` of the canonical corpus file consumed by this run (the corpus as ordered at run start) |
| `data_order_seed` | int | the seed that produced the example order actually consumed (not the seed that was intended — the one the trainer used, as recorded by the trainer's own log or checkpoint metadata; if the trainer does not record it, the field is `null` and the run's ordering is UNVERIFIED — see §2.6) |
| `hyperparams` | object | learning rate, batch size, epochs, scheduler, and any other trainer hyperparams, recorded verbatim from the trainer config at run start; later config changes journal a new `training_runs` revision note, never an edit of this row |
| `trainer_commit` | string | git commit of the trainer code that produced this checkpoint (trainer repo + commit hash; if the trainer is uncommitted or its commit is unknown, the field is the literal string `"unknown"` — never an invented hash) |
| `checkpoint_complete` | bool | `true` only when the watcher observed a complete, quiescent checkpoint (§3); a row with `false` is a partial-save observation, never a rollback target |
| `seq` / `prev_hash` / `entry_hash` | as §2.1 | assigned by the journal runtime, not the trainer |

The manifest is the training analogue of the `.si`
`manifest.json` (`docs/si-container-spec.md`): canonical,
content-addressed, and the only mutable truth about "where
the weights were" — except that here the manifest itself is
append-only history: a new checkpoint appends a row; no row is
ever edited to change what a past step was.

### 2.4 Journaled eval moments

One `training_eval_moments` row per evaluation of a
checkpoint. Required fields (PROPOSED):

| Field | Type | Notes |
|---|---|---|
| `run_id` / `step` | string / int | FK to the checkpoint row this moment describes |
| `eval_suite` | string | name + version of the eval suite run (e.g. the conformance suite that produces the Tier-0 / discrimination / temporal scores below; the suite's identity must be recorded, because a score without its suite is not comparable) |
| `eval_suite_commit` | string | git commit of the eval suite, or `"unknown"` (same rule as `trainer_commit`) |
| `tier0_score` | float \| null | Tier-0 gate-eval score for this checkpoint, or `null` when the suite did not run that family |
| `discrimination_score` | float \| null | discrimination-family score, or `null` |
| `temporal_score` | float \| null | temporal-family score, or `null` |
| `loss` | float \| null | training loss at this step, as reported by the trainer, or `null` |
| `grad_norm_mean` / `grad_norm_max` | float \| null | gradient statistics as reported by the trainer, or `null` |
| `sample_generations` | object[] | sample generations at this moment: `{prompt, generation, generation_hash}`; generations are text and are stored verbatim up to a PROPOSED cap of 4,096 characters per generation, with the full text's `sha256` always recorded, so a truncated sample is detectable and never mistaken for complete |
| `eval_wall_ts` | timestamptz | when the eval ran (informational; ordering still uses `seq`) |

Scores are recorded exactly as the suite reported them. The
journal never re-scores, never normalises, and never fills a
`null` with a neighbouring step's value: a missing family is a
gap in the record, and a gap must be visible as a gap (§3.4).

There is NO existing SI API that produces Tier-0 /
discrimination / temporal scores for a 1.6B model. The
miniature probes (`si/probes.py`, SI-13/SI-14) are identity
and capability probes for miniature models, not this suite.
The binding between an eval suite and these three score
fields is PROPOSED work for the journal-writer slice (§9,
slice 2), and until that slice lands, these fields are a
schema design, not a measurement pipeline.

### 2.5 Event kinds

Every journaled training event is one of these kinds
(PROPOSED). The kind is a required field on every row in the
five new collections, so a reader can filter the journal by
what happened without parsing payloads.

| Event kind | Collection | Meaning |
|---|---|---|
| `run_started` | `training_runs` | A run began; hyperparams, corpus hash, trainer commit recorded |
| `checkpoint_manifest` | `training_checkpoints` | A checkpoint save was observed and hashed (§3) |
| `eval_moment` | `training_eval_moments` | An eval suite result was journaled for a checkpoint |
| `window_consumed` | `training_windows` | A step range consumed a stated example window (§5) |
| `prune_proposed` / `prune_verdict` | `training_proposals` | A corpus prune was proposed / gated (§5) |
| `rollback` | `training_proposals` | A rollback was proposed, gated, and (if admitted) executed (§4) |
| `fork` | `training_proposals` + `training_runs` | A new lineage branched off a past checkpoint (§4) |
| `retention_decision` | `training_proposals` | A checkpoint's optimizer state was kept or dropped under the §6 policy, gated |
| `watcher_gap` | `training_runs` (note row) | The watcher lagged, died, or missed steps; the gap is recorded, never hidden (§3.4) |

Provenance edges (following `docs/nedb-journal-schema.md`'s
`provenance` collection, extended to training): every
`rollback`, `fork`, and `prune_verdict` row carries
`parents` (the checkpoint seqs / window rows it derives
from) and a `relation` (`rolled_back_to`,
`forked_from`, `pruned_from`, `supersedes`). A fork's new
`training_runs` row carries `parent_run_id`,
`parent_step`, and `parent_manifest_hash`, so the lineage
chain is walkable in both directions without a separate
index.

### 2.6 What the schema refuses to fake

- No weights bytes in NEDB. Ever. A row that cannot name a
  content hash names no blob.
- No invented `data_order_seed`, `trainer_commit`, or
  `eval_suite_commit`: `"unknown"` / `null` is the honest
  value, and any downstream claim (replay exactness, §4)
  degrades explicitly when those fields are unknown.
- No edited history: a corrected score appends a new
  `eval_moment` row that `supersedes` the old one by
  provenance edge; the old row stays.
- No silent defaults for the database name (§7).

## 3. Watcher design (non-invasive)

### 3.1 The watcher observes; it never operates the trainer

The watcher is a separate process that watches the trainer's
checkpoint output. It does not patch, wrap, import, or
otherwise modify the trainer. There is NO trainer surgery in
this design: no trainer code changes, no trainer flag changes,
no trainer restart to attach. If the watcher were deleted
mid-run, the trainer run would be bit-for-bit unaffected; only
the journal would have a gap, and the gap itself would be
journaled on the watcher's return (§3.4).

PROPOSED watcher steps, per checkpoint:

1. **Discover.** Poll the trainer's checkpoint root for new
   `step<N>` directories (a filesystem poll at a PROPOSED
   30-second interval, plus a directory-listing diff; an
   OS filesystem-notification mechanism MAY supplement the
   poll but must never be the only discovery path, because
   notifications are droppable and a dropped notification
   must not become a dropped checkpoint).
2. **Quiesce.** A checkpoint is journal-eligible only when it
   is complete: the directory's file set and file sizes are
   unchanged across two consecutive observations AND the
   trainer's own completion signal for that save is present
   (whatever the trainer writes last — a manifest file, a
   lock release, or the appearance of `step<N+1>` progress;
   the concrete signal for a given trainer is a PROPOSED
   per-trainer adapter decision, recorded in the
   `training_runs` row, never guessed silently). A checkpoint
   observed mid-write is journaled, if at all, with
   `checkpoint_complete = false` and is never a rollback
   target.
3. **Hash.** Stream-hash the weights file and (when present)
   the optimizer state file with SHA-256. Hashing is
   read-only and happens after quiescence, so the hash names
   the bytes the trainer actually finished writing.
4. **Journal.** Append the `checkpoint_manifest` row (§2.3)
   and, when the trainer has reported them, the
   `window_consumed` rows covering the steps since the last
   journaled checkpoint (§5).
5. **Score.** When an eval moment is available — the trainer
   ran an eval, or an operator ran one against the
   checkpoint — append the `eval_moment` row (§2.4). The
   watcher does not itself run evals in this design; scoring
   automation is a later slice (§9). A checkpoint with no
   eval moment is a checkpoint with no scores, and the
   journal says exactly that.

### 3.2 What the watcher writes, and where

The watcher writes ONLY to the training journal (§7's
database). It never writes into trainer checkpoint
directories, never renames or moves checkpoint files, and
never deletes optimizer state: retention (§6) is a gated
proposal executed by an operator or an admitted tool, not a
side effect of observation.

### 3.3 Ordering guarantee

Journal `seq` order is watcher-observation order. For
checkpoints discovered in the same poll, observation order
is step order (the watcher sorts by `step` before
journaling). A checkpoint discovered late (watcher lag,
§3.4) is journaled when discovered, with its true `step`;
its position in `seq` is later than its step-neighbours,
and any reader must therefore order checkpoints by `step`,
using `seq` for provenance, not as a step proxy. This is
stated here so no future reader builds a step-order
assumption the journal does not guarantee.

### 3.4 Failure modes: what is lost, what is recoverable

| Failure | Journal state | What is lost | What is recoverable |
|---|---|---|---|
| Watcher lags (trainer saves faster than the watcher polls/hashes) | Manifests arrive late; `seq` shows a burst after the save's wall time | Nothing about the checkpoints themselves (they are on disk, content-addressed) | The full manifest set, once the watcher catches up; per-step timing precision degrades to poll granularity and is recorded as such |
| Watcher dies mid-run | Last journaled checkpoint stands; no rows for later saves until restart | Eval moments and window rows that existed only in the watcher's memory (none should: the watcher journals before it forgets — a window/eval fact the watcher has not journaled and cannot re-derive from disk is lost, and the loss is recorded as a `watcher_gap` row on restart, naming the step range it covers) | On restart, the watcher re-scans the checkpoint root from the last journaled step, hashes every `step<N>` it finds, and journals the missing manifests; the `watcher_gap` row records the interval so no reader mistakes the run for continuously observed |
| Watcher observes a partial save (trainer died mid-checkpoint) | A `checkpoint_complete = false` row, or no row if quiescence never held | The partial checkpoint is unusable as a rollback target — by design | Nothing needs recovery: the last complete checkpoint remains the rollback target; the partial directory is the trainer's to clean, never the watcher's |
| Checkpoint deleted on disk after journaling | The manifest row remains, its hash now pointing at nothing | The ability to roll back to that checkpoint | Detectable, not silently: any rollback/replay attempt re-verifies the referenced blob's existence and hash before proceeding and fails closed on a mismatch (§4). The journal keeps the history; it does not pretend the bytes still exist |
| Two watchers run at once (operator error) | Risk of duplicate manifests for one step | Nothing if detected | Prevention is procedural in this design (one watcher per run, enforced by a watcher lock file OUTSIDE the trainer tree, in the journal's own directory); duplicates, if they occur, are detectable by (`run_id`, `step`, `weights_blob`) and the second row is superseded by provenance edge, never deleted |

The governing principle: **the watcher may lose precision,
and may lose moments that existed only in flight, but it
must never lose the distinction between what it observed
and what it did not.** Every gap above ends as a row in the
journal, not as an absence a reader could misread as
"nothing happened".

## 4. Rollback / replay / fork semantics

Honest granularity, stated exactly as the mission brief
states it, because every other sentence in this section
depends on it:

> **Inspection is exact at every journaled moment.
> Rollback is exact only at retained checkpoints —
> optimizer state is the disk cost. Between checkpoints,
> replay forward with journaled order.**

### 4.1 Inspection (exact at every journaled moment)

For any journaled `seq`, a reader can reconstruct exactly
what the journal knew: which checkpoints existed, their
hashes and manifests, every eval moment and its scores,
every window consumed up to that point, and every proposal
and verdict. This is the SI-0 journal's time-travel
property (`docs/nedb-journal-schema.md`, "Time travel")
applied to training rows: the journal at seq N plus the
blobs its manifests pointed at is the full record of the
run as of N. Inspection never moves anything.

### 4.2 Rollback (exact only at retained checkpoints)

Rollback to step N means: rehydrate the weights blob for
step N — resolve `weights_blob` to its referenced file,
verify the file's SHA-256 equals the manifest's hash
(fail closed on any mismatch), and resume training from
that checkpoint.

- When the checkpoint at step N retains optimizer state
  (`optimizer_blob` non-null, and that file also
  hash-verifies), resume is **exact**: weights, optimizer
  moments, and step count are all the trainer's own, and
  the resumed run continues the same trajectory the
  original run would have taken (up to the trainer's own
  nondeterminism, which this design does not claim to
  eliminate).
- When step N retains weights only, resume is
  **weights-exact, optimizer-fresh**: the weights are bit
  for bit the step-N weights; the optimizer starts cold.
  The trajectory after resume is therefore a NEW trajectory
  from step N's weights, not a continuation of the original
  one, and the journal records it as such (the resumed run
  is a fork in the §4.4 sense, parented at step N). No doc
  or tool built on this spec may call a weights-only resume
  an exact rollback.
- A rollback is a gated proposal (§2.1 rule 4): the
  proposal names the target step, the run, and the reason;
  the gate verdict and the operator are journaled BEFORE
  the trainer is pointed at the rehydrated checkpoint; the
  original run's rows are never modified by the rollback.
- Rollback never deletes the later checkpoints or their
  rows. What the run did after step N stays in the journal
  and on disk (subject to §6 retention) as evidence;
  rollback moves the head, it does not rewrite the past.
  This mirrors the SI-0 loop's rollback
  (`docs/self-write-loop.md`, "Rollback — the path back"):
  move `head`, journal the rollback as its own event,
  delete no bytes.

### 4.3 Replay (re-run the eval suite, as of a journal moment)

Replay means: take the checkpoint for step N, and re-run
the eval suite against it **as of its journal moment** —
the same suite identity (`eval_suite` +
`eval_suite_commit`) the original `eval_moment` recorded,
on the same checkpoint bytes (hash-verified first).

- A replay result is a NEW `eval_moment` row, linked by
  provenance to the original moment it replays. It never
  overwrites the original score: if the replay disagrees
  with the original moment, both rows stand and the
  disagreement is itself the finding.
- Replay is the only exact operation available for a step
  BETWEEN checkpoints in one specific sense: the eval
  suite can be re-run against the nearest retained
  checkpoint. A step that was never checkpointed has no
  weights to evaluate; nothing in this design pretends
  otherwise.
- Forward replay of TRAINING between checkpoints —
  reconstructing step N+k's weights by re-running steps
  N+1…N+k — is possible only when the journal recorded
  the corpus hash, the data-order seed, and the window map
  (§5) for that range, AND the trainer is deterministic
  enough for the re-run to land on the same weights. This
  spec does NOT claim forward training replay reproduces
  weights exactly: GPU training nondeterminism is real, and
  the honest claim is that the journal records everything
  needed to ATTEMPT a faithful re-run and to verify its
  result hash against the next checkpoint's manifest when
  one exists. Where `data_order_seed` is `null` (§2.3),
  forward replay is not faithful and must not be attempted
  under that name.

### 4.4 Fork (branch a new lineage, both lines kept)

Forking at step N means: create a new `training_runs` row
whose parent is (source `run_id`, step N, the step-N
manifest's `weights_blob` hash); the new run starts from
step N's verified weights and proceeds with its own
hyperparams, its own corpus (possibly pruned, §5), and its
own journal rows.

- Both lines are kept. The source run's rows, checkpoints,
  and eval moments are untouched; the fork's rows reference
  the source checkpoint by hash and provenance edge,
  never by copying its weights (SI never duplicates bytes:
  the fork's step-N manifest references the same blob the
  source run's manifest references).
- A fork is a gated proposal, journaled before the forked
  trainer starts, with its reason recorded — the fork is
  the "reimagine" step of Mark's directive, and the
  journal must always be able to answer "reimagined from
  what, and why?"
- Lineage is a tree, not a chain: a run may be forked more
  than once, and a fork may itself be forked. Every node's
  parent edge is in its `training_runs` row (§2.2), so the
  full tree is one recursive query over the journal.

## 5. Corpus-window prune lineage

### 5.1 The window map: step ranges → exact example windows

The journal maps step ranges to the exact example windows
the run consumed, so a bad moment can be traced to the
data that produced it.

One `training_windows` row per consumed window (PROPOSED
fields):

| Field | Type | Notes |
|---|---|---|
| `run_id` | string | FK → `training_runs.run_id` |
| `step_from` / `step_to` | int | inclusive step range this window covers |
| `corpus_hash` | string | the corpus this window indexes into (must equal the run's `corpus_hash`; a mismatch means the run changed corpora mid-flight and the change itself must be a journaled event, not a silent swap) |
| `data_order_seed` | int \| null | the order seed in force for this range (same honesty rule as §2.3) |
| `example_start` / `example_end` | int | inclusive index range into the ORDERED corpus (the corpus as ordered by `data_order_seed`) |
| `example_ids_hash` | string | `sha256` over the ordered list of example ids in this window — compact proof of exactly which examples, without storing the corpus in the journal |

An example id is the corpus's own stable identifier for an
example (its index in the canonical corpus file, hashed
with the file, is the minimum viable form; a corpus with
real ids uses those). The window map plus `corpus_hash`
plus `data_order_seed` is sufficient to enumerate the
exact examples consumed between any two journaled steps —
which is the precondition for a prune that means anything.

### 5.2 Tracing a bad moment to its data

Given an eval moment at step N that went wrong (a score
regression, a bad sample generation), the trace is: sum
the `training_windows` rows with `step_to <= N` since the
last checkpoint whose eval moment was good, enumerate
their example ranges against the corpus named by
`corpus_hash`, and inspect those examples. The journal
turns "the run went bad somewhere in the middle" into a
named set of examples. This is inspection (§4.1): exact,
because the window map is exact — PROVIDED the trainer
actually consumed examples in the journaled order. Where
the order seed is unknown or the trainer's true order was
never observed, the window map records the order the
trainer was CONFIGURED to consume, and the row must say
so (`order_source`: `observed` / `configured` —
PROPOSED field). A configured-order map supports a prune
hypothesis; only an observed-order map supports a prune
claim.

### 5.3 A prune is a gated proposal — never an automatic edit

A prune NEVER edits the corpus automatically. The flow is
the self-write loop's flow, applied to data
(`docs/self-write-loop.md` §§2–4):

1. **Propose.** A `prune_proposed` row names: the source
   run, the step range and window rows the bad examples
   were traced to (§5.2), the exact example ids proposed
   for removal, and a `reason` — a statement of why these
   examples, in the proposer's own words. A proposal
   without a reason is malformed and rejected outright,
   exactly as a self-write intent without a reason is.
2. **Gate.** The gate runs BEFORE the prune lands, with
   verdicts `ADMIT` / `REJECT` / `ESCALATE` as in the SI-0
   gate. PROPOSED prune checks, mirroring the SI-0
   checklist's shape: (a) constitution check — the prune
   does not violate the bundle constitution; (b) bounds —
   the prune does not exceed a PROPOSED maximum fraction
   of the corpus per proposal and never prunes examples
   outside the traced windows without a separate,
   explicitly-scoped proposal; (c) provenance completeness
   — every pruned example traces to a journaled window and
   a journaled bad moment; an uncaused prune is
   indistinguishable from noise and is rejected; (d)
   capability floor — the gate MAY require an eval plan
   for the post-prune run (the fork that consumes the
   pruned corpus gets eval moments at its first
   checkpoints; those moments are the prune's evidence,
   recorded when they exist, never predicted in advance).
3. **Journal before landing.** On `ADMIT`, the proposal
   and the verdict are journaled (`prune_verdict`) BEFORE
   any new corpus exists. The pruned corpus is a NEW
   corpus file with its own `corpus_hash`; the original
   corpus file is never edited. The fork (§4.4) that
   trains on the pruned corpus names the new
   `corpus_hash` in its `training_runs` row, and a
   provenance edge (`pruned_from`) links new corpus →
   prune verdict → traced windows → original corpus.
4. **Reversible in record.** Because nothing is edited —
   the original corpus, the window map, the proposal, and
   the verdict all persist — a prune decision is fully
   auditable after the fact and reversible in record: a
   later fork can train on the ORIGINAL corpus hash from
   the same checkpoint, and the journal shows both lines
   and the decision that separated them. "Reversible in
   record" is the exact claim; this spec does not claim
   the pruned run's compute is recoverable — spent steps
   are spent.

## 6. Disk budget vs the v12 lesson

### 6.1 The v12 lesson

Imagine v12's training taught the rule the hard way, now
standing practice: **SAVE_EVERY 500 or more on full
fine-tunes.** A full fine-tune checkpoint is not a LoRA
adapter; it carries optimizer state, and saving too often
turns a training run into a disk-exhaustion event. This
section does the arithmetic for the current model size so
the retention policy below rests on numbers, not vibes.
All sizes in this section are ESTIMATES computed from the
stated assumptions; none is a measured size, and none may
be quoted as one.

### 6.2 Assumptions (stated, so the arithmetic is checkable)

- Model: Imagine, **1.6B parameters, full fine tune**
  (no adapter: every parameter trains, every parameter is
  in the checkpoint).
- Parameter count used: 1.6 × 10⁹.
- Byte widths: bf16 = 2 bytes/param; fp32 = 4 bytes/param.
- Optimizer: AdamW-class, two fp32 moment buffers
  (m and v) per parameter = 2 × 4 = 8 bytes/param.
- A "full checkpoint" = weights + gradients + optimizer
  moments + master weights where the layout keeps them;
  a "weights-only checkpoint" = weights alone.
- Decimal GB (10⁹ bytes) throughout.

Derived per-checkpoint sizes (arithmetic from the
assumptions above):

| Component | Arithmetic | Size (ESTIMATE) |
|---|---|---|
| Weights, bf16 | 1.6×10⁹ × 2 B | 3.2 GB |
| Weights / master weights, fp32 | 1.6×10⁹ × 4 B | 6.4 GB |
| Gradients, bf16 | 1.6×10⁹ × 2 B | 3.2 GB |
| Gradients, fp32 | 1.6×10⁹ × 4 B | 6.4 GB |
| AdamW moments (m + v), fp32 | 1.6×10⁹ × 8 B | 12.8 GB |
| **Layout A full checkpoint** (bf16 weights + bf16 grads + fp32 moments) | 3.2 + 3.2 + 12.8 | **19.2 GB** |
| **Layout B full checkpoint** (fp32 master + fp32 grads + fp32 moments) | 6.4 + 6.4 + 12.8 | **25.6 GB** |
| Weights-only checkpoint, bf16 | — | **3.2 GB** |
| Optimizer add-on per checkpoint (moments + bf16 grads) | 12.8 + 3.2 | **16.0 GB** |

The journal itself is not a disk factor: a manifest row is
hashes, scores, and config — kilobytes per checkpoint,
megabytes per run, against tens of gigabytes of blobs.
NEDB stores pointers (§2.1 rule 1); the disk budget below
is entirely blob budget.

### 6.3 The v13-shaped example (ESTIMATES throughout)

The live v13 stage-1 run's shape, as recorded in the
loop's notes at this spec's writing: 7,311 steps, 3
epochs, learning rate 1e-5, SAVE_EVERY 500.

- Saves at steps 500, 1000, …, 7000 = 14 saves, plus the
  final step 7,311 = **15 checkpoints**.
- If all 15 were retained as Layout A full checkpoints:
  15 × 19.2 GB = **288 GB** (ESTIMATE). Layout B:
  15 × 25.6 GB = **384 GB** (ESTIMATE). This is the v12
  lesson in numbers: full retention of full checkpoints
  is not a policy, it is an outage on a schedule.
- Weights-only for all 15: 15 × 3.2 GB = **48 GB**
  (ESTIMATE) — every step remains inspectable and
  forkable (§4.4), and weights-exact rollback (§4.2)
  remains possible everywhere.
- Optimizer state on the newest 2 checkpoints only:
  48 + 2 × 16.0 = **80 GB** (ESTIMATE) — exact resume
  (§4.2) stays available where it is most likely to be
  wanted (the head), at roughly a quarter of the
  all-full cost.

### 6.4 Retention policy — as gated proposals

PROPOSED default retention shape (a policy to be adopted,
not a behaviour of any existing code):

1. Weights are retained for every SAVE_EVERY checkpoint.
   Deleting weights deletes the rollback/fork target; a
   weights deletion is therefore itself a gated proposal,
   never a watcher side effect (§3.2).
2. Optimizer state is retained for the newest 2
   checkpoints and for any checkpoint named as a fork
   parent or rollback target by an admitted proposal.
   Dropping any other checkpoint's optimizer state is a
   `retention_decision` row (§2.5): proposed, gated,
   journaled, with the checkpoint's new `optimizer_blob =
   null` state appended as a new manifest revision, never
   an edit of history.
3. SAVE_EVERY below 500 on a full fine-tune requires its
   own gated proposal naming the disk arithmetic for the
   new interval — the v12 lesson, enforced by policy
   rather than memory.
4. Every retention figure quoted to an operator uses this
   section's method: state the layout, state the
   assumption, label the number an ESTIMATE. A measured
   size, when one exists, replaces the estimate in the
   manifest's `weights_bytes` / `optimizer_bytes` fields
   (those fields are measured at hash time, §3.1 step 3 —
   the one place this design does produce real numbers).

## 7. Database naming, explicit

### 7.1 The lesson that forces this section

On the B200 box (2026-11-03), the NEDB seed materialises
as database **`stealth_shop`**: the gate proof passes
**68/68** against `dbname=stealth_shop`. The sukuna-imagine
setup script exports `dbname=shop` — that name is **WRONG**:
the same proof passes only **53/68** against it, with every
must-accept case failing at L2 bind. A wrong database name
did not fail loudly; it failed as a subtly broken bind.
The training journal must never be in a position to make
that mistake: a journal written into the wrong database is
worse than no journal, because it looks like a record.

### 7.2 The training-journal database, named

- The training journal's database name is
  **`si_training_journal`**. This name is PROPOSED by this
  spec (no such database exists yet); once adopted by the
  journal-writer slice (§9, slice 2), it is the ONLY name
  that slice may use, recorded verbatim in the
  `training_runs` row and in the journal manifest.
- The existing SI-0 journal database for a bundle is a
  DIFFERENT database, derived today by
  `si/backends/nedb.py::_dbname_for` as `si_<bundle_name>`.
  The training journal does not silently inherit that
  derivation. If a future slice co-locates training rows
  with a bundle journal, it must say so explicitly and
  name the shared database explicitly; until then, the
  training journal is its own database with its own name.
- `stealth_shop` is the pre-existing seed database on the
  B200 box. It is NOT the training-journal database, and
  nothing in this design writes to it. Any future slice
  that reads it must name `stealth_shop` explicitly in its
  own spec and re-run the gate proof against that name
  before relying on the read. `dbname=shop` is wrong, is
  never an acceptable alias, and must never appear as a
  default, a fallback, or an example value in training
  journal code or docs after this spec. (It appears in
  this section once, as the named wrong value, so a
  reader can recognise and reject it.)

### 7.3 Verification before any write — never silently default

The name is verified before the journal's first write,
every time a writer opens the journal (PROPOSED procedure,
to be implemented by the journal-writer slice):

1. **Explicit configuration, no derivation.** The writer
   takes the database name from explicit configuration
   whose value must be the literal `si_training_journal`
   (or a future name adopted by a later spec and recorded
   the same way). An absent or empty configuration is a
   hard failure. There is NO fallback derivation — not
   `_dbname_for`, not an environment default, not the
   bundle name — because a fallback is exactly the
   mechanism that turns a naming mistake into silent
   wrong-database writes.
2. **Create-or-refuse.** The database must exist before
   the first write (NEDB requires the database to be
   created first, via the daemon's HTTP API — see
   `si/backends/nedb.py`: "Database must be created first
   (POST /v1/databases)"). Creating it is an explicit,
   journaled-elsewhere operator action naming
   `si_training_journal`; the writer itself never creates
   a differently-named database to "make progress".
3. **Read-back handshake.** After connecting and BEFORE
   any journal write, the writer reads back the connected
   database's name from the server and compares it,
   byte-for-byte, with the configured name. A mismatch —
   or an inability to read the name back — fails closed:
   no write, no retry against a different name, and the
   failure is reported to the operator with both names
   (configured vs actual) stated.
4. **Genesis record.** The journal's first entry (its
   `run_started`-era genesis row) records the verified
   database name, the pgwire and HTTP ports in use
   (which `si/backends/nedb.py::_ports_for` derives
   deterministically from the dbname's SHA-256 — recording
   them makes a port collision with another database
   visible instead of mysterious), and the verification
   timestamp. Any later reader can therefore audit WHICH
   database a run's journal claims to be, from inside the
   record itself.

## 8. Adoption split: what the live v13 run can adopt vs what waits

The live Imagine v13 run on the B200 box proceeds on its
current runbook, in parallel with this spec. Nothing in
this section makes the v13 run wait for SI, and nothing in
this spec claims the v13 run IS journaled: a run is
journaled only when a watcher is actually built, attached,
and verified journaling it — and no watcher exists today
(§1). The split:

**Can adopt live (add-on, non-blocking, only if Mark
directs it on the box):**

- The §3 watcher in OBSERVE-ONLY form, attached before
  stage 1 as a read-only observer of the checkpoint root —
  journaling checkpoint manifests (§2.3) and nothing else:
  no gate, no prune, no retention action. Observe-only is
  the one form whose failure (§3.4) cannot touch the run.
- The §7 discipline, immediately and without code: every
  script, config, and note for the run names its NEDB
  database explicitly (`stealth_shop` for the seed),
  verifies the name by read-back before writing, and
  never defaults — the `dbname=shop` export is corrected
  wherever it appears, as its own fix, not as part of any
  SI slice.
- The §6 retention SHAPE as an operator habit (weights
  for every checkpoint, optimizer for the newest few,
  SAVE_EVERY 500+) — it already matches the run's
  runbook; the journal is not required for the habit to
  help.

**Waits for later slices (NOT adoptable live):**

- The journal writer and the five PROPOSED collections
  (§2.2) — they do not exist; until slice 2 (§9) lands
  and is verified, there is no training journal to write
  into, and observe-only manifests live wherever that
  slice's evidence says they lived, not retroactively in
  a journal built later.
- Gated corpus prune (§5.3), rollback tooling (§4.2),
  replay tooling (§4.3), and fork tooling (§4.4) — each
  is its own slice with its own verification.
- Eval-moment automation (§2.4) — the score fields are a
  schema design until a suite binding exists. Manual
  scores, journaled by hand with their suite identity
  recorded, are honest; automated scores from an unbuilt
  pipeline are not available to claim.

If the v13 run finishes before any of this is built, that
is a normal outcome, not a failure of either line: the
run's checkpoints remain on disk, content-hashable after
the fact, and a later slice MAY journal them
retrospectively — recording every manifest with the
honest `order_source` / observation caveats of §§3–5,
and never backdating the observation.

## 9. Proposed slice plan after this spec

Each slice below is **PROPOSED — not scheduled, not
approved, and not built.** Each needs its own brief, its
own PR, and the loop's normal review before any claim
about it is made. Order is the order of dependency, not a
commitment to build in that order.

1. **Watcher implementation (observe-only).** Build the
   §3 watcher: discovery, quiescence, hashing, manifest
   journaling to a plain record (the §2 schema in file
   form), `watcher_gap` rows on failure. Gate/verification:
   against a synthetic trainer that saves miniature
   checkpoints, the watcher's manifests must match
   independently computed hashes for every complete
   checkpoint, must mark a deliberately truncated save
   `checkpoint_complete = false`, and must journal a gap
   — not silence — when killed and restarted mid-sequence.
   No NEDB write path is required for this slice to prove
   the observation half.
2. **Journal writer + database.** Implement the five
   PROPOSED collections (§2.2) on the existing
   `si/journal.py` seq/hash-chain machinery, in the
   database named and verified per §7 (`si_training_journal`,
   read-back handshake, fail closed). Gate/verification:
   chain verification passes over a journal containing
   every §2.5 event kind; a writer configured with any
   other database name — including `shop` and
   `stealth_shop` — refuses to write; `verify_chain`
   failure refuses to open.
3. **Rollback / replay tooling.** Implement §4.2
   rehydrate-and-verify (fail closed on hash mismatch)
   and §4.3 eval replay (new moment rows, provenance to
   the originals), against miniature checkpoints first.
   Gate/verification: a weights-only resume is labelled
   weights-exact/optimizer-fresh in its journal row; an
   exact resume is claimed only where optimizer state
   hash-verifies; a replay that disagrees with its
   original moment leaves both rows standing.
4. **Fork + gated prune proposals.** Implement §4.4 fork
   lineage rows and the §5.3 propose→gate→journal→land
   flow for corpus prunes, on miniature corpora first.
   Gate/verification: a prune with no traced window or no
   reason is rejected at the gate; an admitted prune
   produces a NEW corpus hash and a `pruned_from` edge,
   and the original corpus file is byte-identical after
   the flow (hash-checked).
5. **Retention enforcement.** Implement §6.4 as gated
   `retention_decision` proposals (optimizer-drop first;
   weights deletion remains operator-only even here).
   Gate/verification: no retention action occurs without
   a journaled verdict; the disk figures an operator is
   shown are recomputed from the §6.2 assumptions and
   labelled ESTIMATE unless measured bytes are available
   from the manifests.

No slice above is an F-line slice. None of them presumes,
tests, or advances D-F1's question, and none may be
re-scoped into one without Mark's answer to D-F1 first
(§10).

## 10. Honesty and status

- **§18.3 is NOT claimed by this spec** — and this spec
  earns nothing toward it. §18.3 remains NOT earned (§0).
- **No capability is stated as built.** The training
  journal, the watcher, the rollback/replay/fork tooling,
  the prune flow, and the retention enforcement DO NOT
  EXIST. §§2–9 are design. The only present-tense claims
  in this doc are about artifacts that already exist and
  are cited in §11 (the SI-0 journal, by-reference
  seeding, the self-write gate, the miniature trainer),
  plus the B200 database-naming measurements recorded in
  the mission brief (§7.1).
- **Honesty labels are unchanged** (§0): §18.3 NOT
  earned; P5-R1 / P5-R9 UNMEASURED; live model-authored
  intent OCCURRED, scoped to SI-74 only; L-3 LOCKED;
  A-2 / A-3 / A-4 UNAPPROVED; unlocks `[]`.
- **D-F1 remains OPEN.** This spec does not answer it,
  presume its answer, or arrange its contents around
  either answer. No SI-side, non-model intervention is
  proposed anywhere in §§2–9; the watcher observes a
  TRAINER's files, which is training infrastructure, not
  an intervention in Gaia's perception, catalogue,
  instructions, output contract, or gateway assembly
  (the classes SI-82 §4 names), and this spec claims no
  conformance effect for it.
- **The F-line remains STOPPED.** No F-3+ work is built,
  designed, or proposed by this doc, and the §9 slice
  plan contains no F-line slice.
- **Imagine is stated honestly:** 1.6B parameters, full
  fine tune — in the disk arithmetic (§6), in the scope
  header, and everywhere else it appears.
- **The v13 run is not claimed as journaled** (§8): not
  by this spec, and not by any slice until a watcher is
  built, attached to that run, and verified journaling it.
- **Numbers discipline:** every size in §6 is an ESTIMATE
  from stated assumptions; the only measured numbers in
  this doc are the ones it cites from prior merged work
  (Glimmer referenced bytes; the 68/68 vs 53/68 gate
  proof, which this doc cites from the mission brief and
  does not re-measure).
- CPU only for this slice's own work, which ran nothing;
  no secrets were read, used, or recorded; no code was
  written or changed.

## 11. Sources and code references

In-repo (at base `2e4758784e01e1a743961b32817e7b1edb3e9e33`):

- `docs/nedb-journal-schema.md` — the SI-0 journal schema:
  collections, `seq` logical clock, hash chain, "NEDB
  never stores raw weight bytes", time travel. §§2–4 of
  this spec extend its conventions.
- `docs/si-container-spec.md` — the `.si` bundle:
  content-addressed immutable blobs (`blobs/sha256/`),
  frozen base, canonical manifest. The model for §2.3's
  by-reference checkpoint manifest.
- `docs/self-write-loop.md` — propose/gate/journal/
  materialize/verify; the five gate checks; ternary
  verdicts; rollback as a first-class, journaled
  operation. The model for §§2.1, 4.2, and 5.3.
- `si/journal.py` — `Journal._append` (seq + `prev_hash` +
  `entry_hash`), `GENESIS_PREV`, `MUTABLE_KEYS`,
  `verify_chain`, `COLLECTIONS` (the five existing
  collections — training collections are NOT among them,
  §2.2). Cited for mechanism, not modified.
- `si/backends/nedb.py` — the real-NEDB backend:
  `_dbname_for` (the existing `si_<bundle_name>`
  derivation §7.2 declines to inherit silently),
  `_ports_for` (ports derived deterministically from the
  dbname's SHA-256), database-must-be-created-first. Cited
  for mechanism, not modified.
- `si/seed.py` — by-reference seeding (SI-8):
  `seed_format="reference-v1"`, `base.tensor_refs`,
  `stored_bytes` 0, re-verification on load. The precedent
  for §2.3's by-reference manifests.
- `si/tiny_training.py` — miniature training (SI-19).
  Cited in §1 for what it is and is not: it is not an
  Imagine training loop and provides no Tier-0 /
  discrimination / temporal eval binding (§2.4).
- `docs/SI-82-f2-admissibility-question.md` — §0 honesty
  table format and the D-F1 posing (§§0, 10 here).
- `README.md` — terminology ("machine learning
  infrastructure", not the negated phrase of §10's rule).

Out-of-repo (loop working notes, cited as sources, not as
repo evidence):

- Mark's training-journal mission brief, 2026-11-03
  (`~/workspace/si-dev-loop/training-journal-mission-brief.md`):
  the mission, the honest-granularity sentence (§4), the
  first-slice scope, and the `stealth_shop` 68/68 vs
  `shop` 53/68 naming measurements (§7.1).
- The loop's run notes for the live v13 stage-1 run shape
  (7,311 steps, 3 epochs, LR 1e-5, SAVE_EVERY 500) used in
  §6.3, and the v12 SAVE_EVERY lesson (§6.1).
