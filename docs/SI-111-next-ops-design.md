# SI-111 RFC — Next Ops: Reference Discipline, Template Lock, Live Escalation, Checkpoint Diffs (Design Only)

> **Status: DESIGN ONLY.** Derived from merged evidence on `main` at
> `f9656939dea78dcd854fc1d25f690371e5e92d4b` (Merge PR #119, SI Intel
> Briefing dossier). Changes no code, no test, no policy, and no gate
> behaviour. Nothing here is implemented, measured, captured, or run
> live. This RFC runs no graded conformance run, names no new
> candidate, trains nothing, and resolves no escalation.
> **Scope:** CPU only. No GPU, no cloud, no spend, no downloads. This
> is machine learning infrastructure throughout; the served model is
> a specialised machine learning model. Where Imagine is mentioned,
> it is stated as 1.6B parameters, full fine tune.
> **Authority:** Mark's direction of 2026-11-13 ~15:18 EST, minutes
> after the v1.0.0 release ("cook up a reference-discipline linter to
> clear the 24/30 bar and pre-flight template lock", plus three
> patterns from wrapper-style agent-gating projects), recorded by
> the lead as decision **D-111-0** (loop STATE.md, 2026-11-13
> ~15:30 EST): this RFC is the design lane that direction opens. It
> poses its decisions to the **lead** (§7), per the standing rule
> that methodology calls are the lead's; nothing here is built
> until the lead rules.
> **Relationship:** SI-108 closed the v13 conformance frontier and
> recorded D-108-1; the lead answered D-108-1 (continue), and SI-109
> (PR #117) and SI-110 (PR #118) executed it — the Gaia line stands
> COMPLETE at its stated scopes. This RFC does not reopen that
> line. It designs four new slices (§§2–5) the closed frontier's
> own evidence calls for, sequences them (§6), and poses the
> decisions they need (§7).
> **Honesty labels carried forward verbatim, unweakened:** §18.3
> **NOT earned**; P5-R1 **UNMEASURED**; P5-R9 **UNMEASURED**; live
> model-authored intent **OCCURRED, scoped to SI-74 only**; L-3
> **LOCKED**; C-5 **LOCKED**; A-2/A-3/A-4 **UNAPPROVED**; unlocks
> `[]`; D-C1 **stands**; D-90-2 / D-90-3 **DEFERRED**.

## Table of contents

- [0. Honesty table: state at this RFC](#0-honesty-table-state-at-this-rfc)
- [1. Context: a closed frontier and a new direction](#1-context-a-closed-frontier-and-a-new-direction)
- [2. RDL-1 — reference-discipline linter (training-side)](#2-rdl-1--reference-discipline-linter-training-side)
- [3. PFL-1 — pre-flight template lock](#3-pfl-1--pre-flight-template-lock)
- [4. ESC-LIVE-1 — approval trigger and the live escalation exercise](#4-esc-live-1--approval-trigger-and-the-live-escalation-exercise)
- [5. DIFFVIEW-1 — checkpoint diff views over SSE replay](#5-diffview-1--checkpoint-diff-views-over-sse-replay)
- [6. Sequencing](#6-sequencing)
- [7. Decisions posed to the lead](#7-decisions-posed-to-the-lead)
- [8. Rejected alternatives](#8-rejected-alternatives)
- [9. Non-goals](#9-non-goals)
- [10. Honesty rules binding any successor slice](#10-honesty-rules-binding-any-successor-slice)
- [11. Sources and code references](#11-sources-and-code-references)

## 0. Honesty table: state at this RFC

No row is promoted, weakened, or re-labelled by this RFC, and it
adds no unlock: unlocks remain `[]`.

| Element | State | Evidence / scope |
|---|---|---|
| D-C1 | **Stands** — rate ≥ 0.80 over N ≥ 30, ≥ 5 per workload, per-workload floor ≥ 0.60 (24/30 with no workload under 0.60) | Decided 2026-11-02; confirmed standing by SI-91 §3 |
| SI-94 (v13, v1 teaching, plain rendering) | **20/30 = 0.6667, BELOW THRESHOLD** | `docs/SI-94-c2-conformance-record.md` |
| SI-100 (v13, v2 teaching) | **9/30 = 0.30, BELOW THRESHOLD** | Teaching **withdrawn by SI-102**; SI-98 parser amendment stays |
| SI-105 (v13, v1 teaching + A1 rendering) | **18/30 = 0.60, BELOW THRESHOLD** | A1 did not beat the SI-94 baseline |
| SI-107 (C2 multi-cycle diagnostic) | **Complete, INCOMPLETE-DIAGNOSTIC** — no verdict, unlocks `[]` | M = 31 of 60 target shape; `docs/SI-107-c2-multicycle-diagnostic-record.md` |
| v13 conformance frontier | **CLOSED** (2026-11-12, lead synthesis) | F-line harness investigation for v13 COMPLETE; no new harness classes for v13 |
| Gaia line | **COMPLETE at stated scopes** | D-108-1 answered (continue); SI-109 + SI-110 merged (PRs #117, #118) |
| SI-63 §18.3 claim | NOT earned | No merged record earns it; the live escalation exercise that could bear on it has never been run (§4) |
| §18.1 rows P5-R1, P5-R9 | UNMEASURED | Carried unchanged |
| Live model-authored tool intent | OCCURRED, scoped to SI-74 only | No later run extends that scope |
| Escalation resolution machinery (PR #99) | Merged, miniature/embedded scope | Exercised by tests only; **never exercised live end-to-end** |
| L-3 / C-5 | LOCKED | Nothing here unlocks either |
| A-2 / A-3 / A-4 | UNAPPROVED | Separate approvals; none granted here |
| D-90-2 (trainer wiring) / D-90-3 (live-NEDB proof) | DEFERRED — recorded, not decided, not activated | SI-91 §6; SI-108 §3; §7 poses their status under this RFC explicitly (D-111-6) |

## 1. Context: a closed frontier and a new direction

### 1.1 Where the programme stands

Three configurations of Imagine v13 were graded under one unchanged
apparatus — the C-1 harness, the §9 parser, fresh corpora, D-C1 as
decided — and all three stand below the bar (§0). The lead's
synthesis of 2026-11-12 closed the frontier on the evidence: harness
variation moved v13 within roughly 9–20/30, no harness lever reached
D-C1, and **the bottleneck is the candidate, not the harness**. The
F-line's work is complete — it found and fixed a serving artifact
(SI-93), built and priced a protocol extension (SI-96 to SI-102),
and priced the perception lever (SI-103 to SI-105). SI-106's C2
diagnostic ran (SI-107) and its §12 remainder landed in-repo
(SI-109, SI-110 under D-108-1). What remains genuinely open is not
more measurement of v13. It is:

- the **training-side** gap the measurements priced — reference
  discipline (§2);
- a **harness-integrity** gap one incident exposed and
  hand-practice currently covers — the chat template (§3);
- a **live-evidence** gap the merged machinery was built for and
  never used to close — the escalation walk (§4);
- a **debugging** gap every incident paid for in hand-read journal
  archaeology (§5).

### 1.2 Mark's direction, 2026-11-13

Mark's words, in chat, after reading the declassified dossier:
"cook up a reference-discipline linter to clear the 24/30 bar and
pre-flight template lock" — naming the journal-first gate, the
zero-copy `.si` setup, and SQL rewind (the NEDB journal's
queryable, prefix-foldable history) as the architecture's strengths.
He then set this RFC's raw material: three patterns observed in
wrapper-style agent-gating projects he had reviewed —

1. **HumanLayer-style approval triggers** — a human approval,
   arriving over a channel (CLI / webhook), unblocking a gated
   action;
2. **agent-gate-style pre-flight linting** — checks that run
   *before* the agent acts and refuse a wrongly-set-up run;
3. **LangGraph-style checkpoint diffs** — comparing recorded
   state at two checkpoints to see exactly what changed.

### 1.3 The architectural distinction this RFC keeps

Those projects bolt **external wrappers onto tool calls**: an
approval proxy in front of the call, a lint shim beside the runtime,
a checkpoint store the framework maintains alongside execution. SI's
architecture is different in kind, and the difference is the reason
the patterns must be re-derived rather than adopted:

- The **gate** is inside the `.si` container and prices every intent
  before materialisation; there is no tool call outside it to wrap.
- The **constitution** travels with the bundle and is unamendable by
  the mind it governs; policy changes are journaled operator acts.
- The **NEDB journal is the queue** (SI-50) and the source of
  truth; the **SSE spine is a projection** of that journal (event
  ids *are* journal seqs, SI-47), not a second store.
- State is **prefix-foldable**: any prefix of the journal yields the
  state as of that prefix (SI-92 §1.2), which is what "SQL rewind"
  names from the operator's side.

So each borrowed pattern lands as an **in-container discipline**:
an approval becomes a journaled resolution act through the merged
PR #99 machinery, transported — never decided — by a trigger (§4);
pre-flight linting becomes a fail-closed harness precondition
whose check and result are themselves journaled (§3), and a
training-side grader reading the parser's own pricing vocabulary
(§2); checkpoint diffs become a read-only fold over two journal
prefixes (§5). No external service, no second store, no wrapper
around the gate is introduced anywhere in this RFC (§8.1).

## 2. RDL-1 — reference-discipline linter (training-side)

### 2.1 Purpose

The graded records agree on where v13's remaining headroom sits.
SI-95's taxonomy of SI-94: of 10 failures, **6 are stage-4 reference
refusals** — and SI-103 §7.3's ceiling arithmetic over those frozen
counts reads: had every stage-4 failure been parse-ok and nothing
else moved, the corpus would read **26/30** — above the 24/30 bar.
The reference class is therefore the one class whose perfection, on
that corpus shape, crosses D-C1. But SI-108 §1.3 and the lead's
synthesis are equally clear that no harness lever teaches it: the
v2-teaching episode (SI-99 to SI-102) priced contract-teaching as a
*harm*, and the parser amendment that remains cannot supply
discipline the candidate never learned. Reference discipline is a
**training-side** property. RDL-1 is the training-side tool that
grades it **in the data, before training**: a linter over candidate
corpora whose rules are the recorded failure classes, so the next
candidate's corpus can be required to teach — and checked for
actually containing — the forms the measurements priced as missing.

### 2.2 The rules: recorded failure classes only

Every rule below is derived from a class a merged taxonomy or
record actually counted. No rule is invented from intuition; where
a class was defined but never exercised, the rule says so.

| Rule | What it flags | Recorded evidence |
|---|---|---|
| `REF-SEQ-NONPOSITIVE` | A `depends_on` entry citing journal seq ≤ 0 (`seq_nonpositive`) | SI-95: seq 0 ×1 (SI-94). SI-101: seq 0 ×7 across five workloads (SI-100). SI-107 §7: 3 × seq 0, priced-and-refused |
| `REF-SEQ-NOT-INTENT` | A seq citation naming a journal row that is not a tool-intent row of the same task (`seq_not_intent_row`) — the item's own task row, a checkpoints row, another task's row | SI-95: own task seq {5} ×1; checkpoints-row seq 1 ×3 (SI-94). SI-107 §7: seq 181, the run's own *task* row, ×1 |
| `REF-TOOL-NAME` | `depends_on` naming a tool (`"fs_read"`) instead of a journal seq | SI-95: ×1 item, a three-intent chain (SI-94); the genus is SI-81's `depends_on_tool_name_not_journal_seq` from SI-79 |
| `REF-SIBLING-SELF` | A sibling entry citing the citing intent's own index (`sibling_self_reference`) | SI-101 sibling audit: the corpus's **only two** sibling entries, both `{sibling: 0}` self-references at `intents[0]`; correct sibling use **zero** (SI-100) |
| `REF-SIBLING-FORWARD` | A sibling entry citing an index ≥ the citing intent's index | Defined by SI-97's positional rule (strictly-earlier only) and carried in SI-101's class vocabulary; **exercised 0 times** in any recorded corpus — included because the form's spec makes it decidable, labelled unexercised |
| `REF-ENTRY-MALFORMED` | A `depends_on` entry that is neither seq-form nor sibling-form (`entry_malformed`) | SI-107 §7: 1 occurrence, priced-and-refused at stage 4 |
| `ARG-BLEED` | An argument declared on exactly one registry tool carried on a different tool | SI-101: undeclared argument ×9 — `max_bytes` (declared only on `fs_read`) ×7, `content` (declared only on `fs_write`) ×2; classes `search_with_read_arg_bleed` ×3, `write_with_read_arg_bleed` ×1, `edit_missing_new_with_write_arg_bleed` ×2, `search_missing_pattern_with_read_arg_bleed` ×3 |
| `ARG-MISSING-REQUIRED` | A required argument absent (priced where it co-occurs with the bleed genus) | SI-101: `fs_search` missing required `pattern` ×3; `fs_edit` missing required `new` ×2 |
| `SHAPE-PROSE-PREFIX` | Target text opening with prose narration before the §8 document | SI-101: `version_narration_prefix` — bodies opening with a "Version 2" narration line (SI-100) |

Two scoping facts travel with the table. First, **SI-95's
structural finding**: the SI-94 capture fold contains no tool-intent
rows, so under §9.3 no cycle-0 `depends_on` could have verified —
all 20 parse-ok bodies carried an empty `depends_on`. The linter
therefore grades *form and citability against a stated reference
set*, never "would the gate have admitted this" (§2.4). Second,
**SI-107's sibling count**: across the whole 66-document C2
capture, the sibling form occurred **0 times** — the form the
parser supports (SI-98) and the contract no longer teaches (SI-102)
is absent from candidate behaviour entirely; a corpus containing
no correct sibling examples is exactly the gap RDL-1's report
exists to make visible (§2.3).

### 2.3 Inputs, outputs, and the eligibility report

**Inputs.** (i) A candidate corpus in the C-2 corpus shape —
prompt/target pairs or target documents alone; (ii) the tool
registry as data (`si/gaia/registry.py` on `main` — argument
declarations are read from the registry, never copied into the
linter, so the bleed rules price against the same surface the gate
prices); (iii) optionally, the fold or citable set the corpus's
references are priced against (the A1 `citable_reference_set`
shape, SI-104), where seq-form citability — not merely seq form —
is to be graded.

**Outputs.** Per-row verdicts: for every flagged row, the rule id,
the location (intent index, entry index), and the cited value —
rows are cited by corpus id and position, bodies never duplicated
(never duplicate bytes). Plus a **corpus eligibility report**: the
count of rows exhibiting each rule, the count of rows clean, the
count of rows exhibiting each *correct* form (a seq citation inside
the citable set; a sibling citation strictly-earlier and non-self),
and — stated as its own line, never folded into a pass figure —
the count of rows whose references are absent where the corpus's
own task shape calls for a dependency.

**Vocabulary discipline.** The linter reuses the §9 parser's
stage-4 refusal names (`seq_nonpositive`, `seq_not_intent_row`,
`sibling_self_reference`, `entry_malformed`) and SI-95/SI-101's
taxonomy class names rather than inventing synonyms: a lint verdict
and a parser refusal for the same shape carry the same name, so a
training-side fix is checkable against the live pricing vocabulary
by construction. RDL-1 modifies neither the parser, the gate, the
registry, nor the contract text.

### 2.4 The hard label

> **RDL-1 grades data, never live conformance.** No conformance
> claim may cite lint results. A lint-clean corpus is not evidence
> about any model; a lint-dirty corpus is not a verdict on one. The
> 24/30 bar falls only to a **trained candidate re-measured under
> D-C1** — fresh corpus, C-1 harness, D-C1 as decided — which is a
> training-side event in Mark's lane. RDL-1's entire claim is:
> *the data stopped teaching the recorded mistakes*, stated per rule,
> over a named corpus, by hash.

The ceiling context, stated as arithmetic only (SI-103 §7.3):
perfect references on SI-94's shape give 26/30, above the bar;
on SI-100's shape the same arithmetic gives 17/30, because stage-3
bleeding dominated there. That is why the rule set prices the
bleed genus (`ARG-BLEED`, `ARG-MISSING-REQUIRED`) alongside the
reference forms instead of references alone — and why even a
perfect RDL-1 pass promises nothing: it removes the *data-side*
excuse, and the candidate still has to be trained and measured.

### 2.5 What RDL-1 is not

Not a harness change (the closed F-line stays closed; no live
proposal is linted by it), not trainer wiring (it reads corpus
files and emits a report — no trainer called, no training journal
written, no live NEDB touched; D-90-2/D-90-3 untouched, §7
D-111-6), and not a conformance instrument (§2.4 binds its output).

## 3. PFL-1 — pre-flight template lock

### 3.1 The incident this prices

SI-93's diagnosis, on the record: the Imagine v13 Q4_K_M GGUF's
metadata contains **no `tokenizer.chat_template` field at all**
(the v12 GGUF embeds the 1,041-byte DeepSeek-Coder template).
With no embedded template and no `--chat-template` / `--jinja`
anywhere in the gaia spawn path, llama-server fell back to its
built-in ChatML template. The candidate was measured in a chat
format its training never used: SI-92 closed **INSUFFICIENT at
27/30**, and the lead's offline scoring of the frozen bodies found
parse_ok **0/27** — every body ChatML-scaffold debris (27/27
carrying ChatML markers; SI-79's corpus: 0/30). SI-93's finding
stands verbatim: **the 0/27 was a serving artifact, NOT a model
verdict.** The failure was silent: nothing in the capture path
checked which template the server would actually apply, and the
budget was spent before anyone knew.

### 3.2 Current practice: correct, manual, unguarded

Since SI-94, serving is corrected by hand: the node unit's
`--binary` points to a wrapper (`~/workspace/gaia-node/llama-server-v13.sh`)
that execs llama-server with `--jinja --chat-template-file
chat-template-v13.jinja` appended. The template file is 1,041
bytes, sha256
`993c0af704bad76aeba4263a5318f7873a06420a8eb7d40b72a49c39599c0a66`,
verified byte-identical to the template embedded in the v12 GGUF,
and SI-94 verified by hand that the child's argv still carried the
template flags after a restart during capture. Correct — and
held together by operator attention. PFL-1 makes it a lock.

### 3.3 Design

**(a) Candidate registration probe.** When a candidate is
registered for measurement, a GGUF metadata probe (a real GGUF
reader, as SI-93 used — never filename or size inference) records:
`tokenizer.chat_template` **present or absent**; if present, the
sha256 of the embedded template; if absent, the sha256 of the
externally supplied template the serving configuration will use,
or `none`. A candidate with neither an embedded template nor a
recorded external template is registered **TEMPLATE-UNRESOLVED**
and is ineligible for graded or diagnostic capture until resolved.

**(b) Capture pre-flight, fail-closed.** Before the first attempt
of any capture, the harness resolves the template the server will
actually apply — from the served GGUF's metadata or the spawn
path's `--chat-template-file` target — hashes it, and compares it
to the registration record. **Mismatch, absence, or an
unresolvable serving configuration refuses the capture**: no
attempt is spent, no corpus item is consumed, and the refusal is
the run's recorded outcome. The check and its result (both hashes,
the comparison, the decision) are **journaled** as a pre-flight
record in the capture journal, so the proof travels with the
corpus it guarded.

**(c) Restart re-verification.** The residency-guard poll (SI-106
§9.2 interface) gains the template hash as a checked field: a
mid-capture node restart that changes the served template hash
**halts the capture and spends no attempt** — the SI-94
hand-check, mechanised, under the same halt discipline the guard
already enforces for residency.

### 3.4 Scope and honesty

PFL-1 is harness-integrity infrastructure. It changes no recorded
verdict — SI-92's INSUFFICIENT stands exactly as PR #100 records
it — and earns no §18.1 row and no §18.3 progress. Its claim is
narrow and complete: *the SI-93 incident class — a capture spent
against a silently wrong chat template — cannot recur unnoticed*,
because the next capture either starts against the candidate's
recorded trained template or does not start.

## 4. ESC-LIVE-1 — approval trigger and the live escalation exercise

### 4.1 What already exists, and the one gap

The resolution machinery is merged (PR #99; `docs/SI-92-escalation-resolution.md`;
`si/gaia/resolution.py`) and test-proven at miniature/embedded
scope: the closed vocabulary (`admit-after-review` / `reject`);
`resolve_escalation` folds first and journals **nothing** on an
unknown, non-escalated, or already-resolved intent, an invalid
value, or an empty operator; `admit-after-review` is the **only**
transition out of `escalated`, to `queued`; the claim is journaled
**before** execution; the executor re-verifies the resolution from
the journal itself; the resolution buys **exactly one run**, whose
record carries `resolution_seq`. What has never happened: **the
walk, live, end to end.** No real escalation on the live node has
ever been resolved and executed. That is the gap ESC-LIVE-1
closes — the trigger carrying the operator's act to the journal,
and the protocol that exercises it once, for real.

### 4.2 The trigger: transport, never decision

The SSE spine already surfaces escalations as journal events
(event ids are journal seqs; filtered subscriptions, SI-48, can
project exactly the escalation class). The trigger is the return
path: a **CLI command** taking `(intent_seq, resolution, operator,
basis)` that performs `resolve_escalation` against the node's SI
journal (today served by `si-seam-service` on the live node). Its
properties are inherited, not re-decided:

- The trigger **decides nothing**. Every verification — folded
  state, closed vocabulary, single-resolution, operator non-empty
  — stays inside `resolve_escalation` and the executor, exactly as
  merged. A trigger call that fails verification journals nothing.
- **No policy key makes a resolution automatic** (PR #99's design,
  kept): no timer, no allowlist shortcut, no auto-admit. A
  resolution is an explicit operator act or it does not exist.
- The webhook form is **designed but deferred**: the B-5 service
  is localhost-shaped until an auth story exists (SI-41 §19), and
  an unauthenticated webhook that can journal an operator act is
  an approval channel anyone on the network could ring. §7 poses
  the split (D-111-4).

### 4.3 The live exercise protocol

**Preconditions (all verified and recorded before step 1).**
The live gaia node is Resident with Imagine v13 (ctx pool 4096,
parallel 1); seam egress is live (the node passes `--si-addr
127.0.0.1:11437`; `si-seam-service` active on current `main`); the
SI journal's allowlist is exactly `hearth.notify` /
`hearth.restart` at ESCALATE (the journaled operator policy act of
2026-11-11); the journal's chain verification passes at baseline;
the baseline journal tip seq is recorded. The operator of record
is the lead.

**Steps.**

1. **Baseline.** Record the journal tip and a full chain
   verification result. (A DIFFVIEW-1 diff, §5, is the natural
   evidence form for steps 2–6 if it has landed; the raw journal
   export serves until then.)
2. **Propose.** Drive one real `hearth.restart`-shaped proposal
   through seam Flow 1 ingest. Expected, on the 2026-11-11 probe's
   precedent (se_0007): the proposal is journaled, the gate prices
   it **ESCALATE**, nothing executes.
3. **Observe.** Confirm the escalation is visible on the SSE spine
   with event id equal to the intent's journal seq.
4. **Resolve.** Via the new CLI trigger: `admit-after-review`,
   operator and basis recorded. Verify the `escalation_resolution`
   row journals and the fold moves the intent `escalated` →
   `queued` — the only path that transition has.
5. **Execute exactly once.** `execute_resolved_escalation` (or the
   drainer): the claim journals before execution; the execution
   record carries `resolution_seq`. The execution's real effect is
   stated plainly: the node restarts its inference child — Gaia's
   first job is itself (SI-41), and the exercise is scheduled so
   the Resident's return is part of the observation, not a
   surprise.
6. **Verify.** Exactly one genuine execution outcome exists for
   the intent; chain verification over the whole journal passes;
   the fold reads the intent executed with its resolution
   provenance.
7. **Negative controls, run deliberately.** A second resolution
   attempt on the same intent refuses and journals nothing; a
   second execution attempt refuses. Both refusals are recorded in
   the evidence — they are half of what the exercise proves.
8. **Publish evidence.** A record document in the SI-9x record
   style: the journal seqs of proposal, verdict, resolution,
   claim, and outcome; the hashes; the before/after tips; the
   negative-control results; the Resident's restart-and-return
   timeline.

**Abort conditions.** Node not Resident at any checkpoint; seam
egress down; baseline chain verification fails (stop — the journal
is the evidence, and nothing writes to a journal that does not
verify); any execution observed before a resolution (stop; the
journal stands; the anomaly is recorded as an incident, not
worked around).

### 4.4 The label

> **§18.3 remains NOT earned until this exercise (or a successor)
> completes and is recorded.** A completed exercise earns exactly
> its own scope, stated in its record: one live escalation,
> driven, resolved through a journaled operator act, and executed
> exactly once on the live node, chain-verified. No broader
> §18.3 reading is pre-claimed by this RFC; the record will state
> what it earns, and the honesty table will move only on that
> record.

## 5. DIFFVIEW-1 — checkpoint diff views over SSE replay

### 5.1 Design

A **read-only** diff of journaled state between two journal seqs.
Given `(seq_a, seq_b)` with `seq_a < seq_b`, the view folds both
prefixes — prefix folds are already the journal's semantics
(SI-92 §1.2: any prefix yields the state as of that prefix) — and
reports the delta, every row cited by its journal seq:

- proposals journaled in `(seq_a, seq_b]`, by intent kind and tool;
- gate verdicts in the window, counted by verdict (ADMIT / REJECT
  / ESCALATE);
- resolutions and dispositions journaled in the window, with their
  operators;
- claims, execution outcomes, and materialisations, each tied to
  its intent seq and, where present, its `resolution_seq`;
- policy acts and integrity faults in the window, unabridged.

The view is a projection over the one journal — the same source
the SSE spine streams (SI-47) — and opens the journal read-only.
**It has no mutation surface**: it cannot journal, resolve, claim,
execute, or amend. There is no write path to audit because there
is no write path.

### 5.2 Purpose, stated against the incidents

Debugging failed runs is today journal archaeology by hand.
SI-107's capture interleaved two residency-guard halts, four unit
wipes, transport failures, and scored work across 371 fold rows;
reconstructing "what changed between the last good seq and the
failure" cost operator hours when time mattered most. DIFFVIEW-1
makes that question one query. It earns no §18.1 row and no §18.3
progress; its claim is only that the journal — already the source
of truth — becomes cheap to interrogate between two points in its
own history (the SQL-rewind property, given a diff's shape).

## 6. Sequencing

Four slices, each its own PR, builder never merges, lead reviews —
the standing disciplines bind all four (subset tests for builders;
one full suite per PR, the reviewer's; honest labels intact).

| Order | Slice | Depends on | Why here |
|---|---|---|---|
| 1 | **PFL-1** | Nothing | Smallest surface; guards every future capture. The SI-93 class must be locked before any new measurement spends a budget — including any diagnostic the other slices provoke. |
| 2 | **ESC-LIVE-1** | The trigger (§4.2); PR #99 machinery (merged) | The §18.3-bearing gap is the programme's oldest unexercised live path. DIFFVIEW-1 would ease its evidence but is a soft dependency only — the raw journal export serves. |
| 3 | **DIFFVIEW-1** | Nothing | Read-only and dependency-free; sequenced third because its first real customer is the debugging of slices 1–2 and of SI-107-class incidents, not because anything blocks it. It may be pulled forward by the lead without disturbing the order of 1–2. |
| 4 | **RDL-1** | Nothing in-repo | Training-side: its deadline is not this repo's critical path but the **next candidate's corpus freeze** in Mark's lane. It must land before that freeze to be of use; sequencing it fourth is a statement about urgency, not importance — on SI-94's arithmetic it is the only slice aimed at the bar itself. |

## 7. Decisions posed to the lead

Posed to the lead, not to Mark: methodology calls inside the
programme are the lead's by standing directive (2026-11-11), and
D-111-0 opened this lane on that basis. Each is one question with
a recommended answer.

- **D-111-1 — RDL-1's home.** Does RDL-1 land in this repo as
  offline corpus tooling (beside the SI-83 training-journal
  file-form tooling), or in Mark's training lane outside this
  repo? *Recommended: in this repo.* Its rules are this repo's
  recorded failure classes, priced in this repo's parser
  vocabulary, against this repo's registry; it calls no trainer
  and so activates nothing deferred.
- **D-111-2 — PFL-1's registration gate.** Is a recorded template
  hash (embedded or external, §3.3a) a hard eligibility
  precondition for any future graded or diagnostic capture —
  TEMPLATE-UNRESOLVED candidates refused, fail-closed?
  *Recommended: yes.* A precondition that can be waived in the
  moment is the SI-93 incident with extra steps.
- **D-111-3 — ESC-LIVE-1's target.** Does the exercise run
  directly against the live node with one real
  `hearth.restart`-shaped proposal, or against a miniature bundle
  first? *Recommended: the live node.* The miniature walk is
  already test-proven (PR #99); the unexercised thing is
  precisely the live path, and the abort conditions (§4.3) bound
  the blast radius to one restart of the node's own child.
- **D-111-4 — Trigger surface.** CLI only in the first slice,
  webhook deferred until an auth story exists (SI-41 §19)?
  *Recommended: CLI only.* An unauthenticated operator-act
  channel contradicts the gate's own discipline.
- **D-111-5 — DIFFVIEW-1's form.** An in-repo, read-only CLI over
  journal prefix folds (§5.1) — no fork-side work, no new store?
  *Recommended: yes.* The journal is the source of truth; the
  view is a projection of it, like the SSE spine it complements.
- **D-111-6 — D-90-2 / D-90-3 status under this RFC.** Record
  explicitly that both remain **DEFERRED** and are activated by
  no SI-111 slice — RDL-1 is trainer-free by design (§2.5) and
  ESC-LIVE-1 writes to the seam journal, not the training
  journal? *Recommended: yes, record it.* SI-108 §5's lesson is
  that unrecorded deferral is how this programme once lost a
  week; the status is stated here so no successor has to infer it.

## 8. Rejected alternatives

- **8.1 Adopt the wrapper projects as dependencies.** Rejected
  (§1.3): HumanLayer / agent-gate / LangGraph libraries would put
  approval, linting, and checkpoint state in external services
  beside the runtime — second stores, outside the container,
  around a gate that is not theirs. The patterns are borrowed;
  the code is not.
- **8.2 RDL-1 as a live harness lint.** Rejected: refusing or
  annotating live proposals by lint rule is a new harness class,
  and the F-line is closed for v13 (SI-108 §3). Worse, it would
  let a lint verdict masquerade as a conformance signal — §2.4's
  hard label exists to forbid exactly that.
- **8.3 Webhook-first trigger.** Rejected (§4.2): without an auth
  story, a webhook is an unauthenticated operator-act channel.
  CLI first; the webhook is designed, deferred, and gated on
  D-111-4's successor decision.
- **8.4 Automatic resolution (timer- or policy-based admit).**
  Rejected: PR #99's design is explicit that no policy key makes
  a resolution automatic. An approval trigger that approves by
  itself is the wrapper anti-pattern in §1.3 wearing our clothes.
- **8.5 Re-run SI-92 under PFL-1 to repair its record.** Rejected:
  records stand. PFL-1 is prospective; SI-92's INSUFFICIENT and
  SI-93's diagnosis are the history that justifies it, not inputs
  to be re-graded.
- **8.6 Fold RDL-1 into the conformance harness as a pre-score
  filter.** Rejected: scoring corpora by lint and reporting the
  filtered rate would manufacture a number D-C1 never priced —
  §2.4 again. Lint output and conformance output never share a
  page.

## 9. Non-goals

- Design only: no code, tests, configuration, capture,
  measurement, training, or live call in this RFC, and no new
  number of any kind is published here — §§1–5 report only what
  merged slices and records already reported, cited to them.
- No candidate is named, no graded run is proposed, and no
  training work is assigned or claimed (§1.1, §2.4).
- D-90-2 and D-90-3 are not decided, activated, or pre-designed
  (§7, D-111-6). A-2/A-3/A-4 remain UNAPPROVED; L-3 and C-5
  remain LOCKED; C-4 does not proceed.
- No change is proposed to §8 or §9 of the proposer contract, any
  §15.2 policy key, the gate, the registry, the queue, the drain,
  the parser, or any training-journal module. PFL-1 adds a
  pre-flight *around* captures; it touches no pricing.
- No fork-repo or Hearth-side work is assigned. The webhook form
  of the §4 trigger is designed and deferred, not built (§8.3).

## 10. Honesty rules binding any successor slice

These bind RDL-1, PFL-1, ESC-LIVE-1, DIFFVIEW-1, and any slice
built under a §7 decision, exactly as SI-108 §8 bound its
successors:

- **Offline and miniature work is never live evidence.** Fixture
  corpora, replay transports, and embedded-backend tests earn no
  §18.1 row, no §18.3 progress, no L-3 unlock, and no live
  authorisation. ESC-LIVE-1's live claim extends exactly as far
  as its recorded exercise, and no further (§4.4).
- **Lint results are never conformance results** (§2.4). Any doc
  citing an RDL-1 report carries the hard label with the citation.
- **Scope labels travel with the claim.** A result obtained on
  miniature bundles, fixture corpora, or a single live exercise is
  cited with that scope attached, in the citing doc's own honesty
  table.
- **No threshold weakening or re-cutting.** D-C1 is not
  re-proposed, rounded, subset, or re-based to change any
  recorded verdict — including by lint-filtering a corpus (§8.6).
- **No re-scoring old bodies as a new measurement.** Recorded
  corpora are inputs to offline analysis, never a new rate.
- **Verdict-free stays verdict-free.** No fold-supplied or
  multi-cycle corpus acquires a bar, a verdict, or an unlock by
  proximity to the graded records (SI-106 §3).
- **Never fake admission, probes, or inference; never invent
  metadata.** A GGUF probe that cannot read a field records
  `absent` / `unknown`, never a guess (SI-93's reader discipline);
  a pre-flight that cannot resolve the served template refuses
  (§3.3b) rather than assuming.
- **Never duplicate bytes.** Corpus, fold, template, and model
  records reference bodies by hash and path, stored once. This
  RFC copies no corpus body and no template body (§§2–3 state
  counts and hashes only).
- **Terminology and means:** machine learning throughout;
  Imagine, wherever mentioned, is 1.6B parameters, full fine
  tune. CPU only; no GPU, no cloud, no money, no secrets.

## 11. Sources and code references

All on `main` at `f9656939dea78dcd854fc1d25f690371e5e92d4b` unless
a workspace path is given. Where an earlier doc and this RFC
disagree, the earlier doc's evidence wins.

- Loop STATE.md (outside this repo, `~/workspace/si-dev-loop/STATE.md`)
  — lead decision **D-111-0** (2026-11-13 ~15:30 EST) opening this
  design lane on Mark's direction of 2026-11-13 ~15:18 EST; lead
  frontier synthesis, 2026-11-12 ~16:30 EST (frontier closed;
  bottleneck is the candidate); the standing note that a Resolved
  escalation has never been exercised live end-to-end.
- `docs/SI-108-next-frontier-rfc.md` — §0 (the honesty table §0
  inherits), §1 (the closed frontier), §3 (D-90-2 / D-90-3
  deferred, restated at §7 D-111-6), §5 (D-108-1, answered:
  continue — SI-109/SI-110 executed it), §8 (honesty rules §10
  inherits).
- `docs/SI-94-c2-conformance-record.md` — 20/30, BELOW THRESHOLD;
  §serving-correction: the wrapper practice §3.2 prices (template
  file 1,041 bytes, sha256 `993c0af7…a66`, byte-identical to the
  v12 GGUF's embedded template; child argv re-verified by hand
  after a mid-capture restart).
- `docs/SI-95-failure-taxonomy.md` — SI-94's 10 failures: stage 4
  ×6 (tool name ×1 item, own task seq ×1, seq 0 ×1, seq 1 ×3 — a
  checkpoints row), stage 1 ×2, stage 2 ×1, stage 3 ×1; §2's
  structural finding (no tool-intent rows in the capture fold;
  all 20 passes carry empty `depends_on`).
- `docs/SI-96-reference-formation-rfc.md` — §2 (the reference
  problem stated exactly), §4 (Option A / A1, the class the lead
  opened), §5 (Option B, the sibling form SI-97 specified).
- `docs/SI-97-intra-document-dependency-spec.md` — the sibling
  form: positional, 0-based, strictly-earlier; `sibling_not_journaled`
  refusal; §8 v2. `docs/SI-98*` (parser amendment, merged),
  `docs/SI-102*` (v2 teaching withdrawn, parser stays).
- `docs/SI-101-failure-taxonomy.md` — SI-100's taxonomy: the
  sibling audit (two entries, both `{sibling: 0}` self-references,
  correct use zero); cross-tool argument bleeding ×9
  (`max_bytes` ×7, `content` ×2); `version_narration_prefix`;
  seq 0 ×7 across five workloads.
- `docs/SI-103-f3-perception-class-design.md` — §7.3 ceiling
  arithmetic (26/30 on SI-94's shape; 17/30 on SI-101's counts),
  quoted at §2.4 as arithmetic only; §4 (A1, the citable set as
  data — the reference-set shape §2.3 prices against).
- `docs/SI-107-c2-multicycle-diagnostic-record.md` — §1–§2
  (INCOMPLETE-DIAGNOSTIC, M = 31, no verdict), §7 (reference
  forms priced separately: seq form 4 occurrences, 0 passed —
  3 × `seq_nonpositive`, 1 × `seq_not_intent_row` at seq 181;
  sibling form 0 occurrences in the whole 66-document capture;
  1 × `entry_malformed`).
- `docs/SI-93-v13-failure-taxonomy-and-template-diagnosis.md` —
  the template diagnosis §3.1 prices: no `tokenizer.chat_template`
  in the v13 GGUF (real GGUF reader), llama-server's built-in
  ChatML fallback, 27/27 bodies ChatML debris vs 0/30 for SI-79;
  finding: SI-92's 0/27 is a serving artifact, NOT a model
  verdict.
- `docs/SI-92-escalation-resolution.md` and
  `si/gaia/resolution.py` — the merged PR #99 machinery §4 builds
  on: `RESOLUTIONS` (`admit-after-review` / `reject`),
  `resolve_escalation` (fail-closed, journals nothing on refusal),
  `execute_resolved_escalation` (claim before execute, exactly
  one run, `resolution_seq` provenance); scope stated in both:
  miniature/embedded, no live exercise.
- `docs/SI-41-agentic-powers-rfc.md` — §5.4 (operator admission
  of escalated intents), §15.6 (new tool classes and policy
  defaults remain Mark's — none is proposed here), §19 (the
  service is localhost-shaped until an auth story exists — the
  ground of §8.3 / D-111-4).
- `docs/SI-47-gaia-sse-event-spine.md`,
  `docs/SI-48-gaia-sse-filtered-subscriptions.md` — the spine
  §§4–5 project over: event ids are journal seqs; filtered
  subscriptions are projections of the one stream.
  `si/gaia/queue.py` (the prefix-stable fold), `si/gaia/events.py`,
  `si/gaia/registry.py` (the argument declarations §2.3 reads).
- `~/workspace/gaia-node/` (workspace, live node) — `NODE.md`
  (runbook), `llama-server-v13.sh` (the §3.2 wrapper),
  `chat-template-v13.jinja` (1,041 bytes, sha256 `993c0af7…`);
  seam egress wired 2026-11-11 (`--si-addr 127.0.0.1:11437`),
  allowlist `hearth.notify` / `hearth.restart` at ESCALATE by
  journaled operator policy act; the `hearth.restart`-shaped
  probe priced ESCALATE (se_0007) — §4.3 step 2's precedent.

End of RFC SI-111. The frontier is closed and stays closed (§1);
four slices are designed — a training-side linter that grades the
recorded mistakes out of the data (§2), a template lock making the
SI-93 incident class impossible to repeat silently (§3), the live
escalation walk the merged machinery has always waited for (§4),
and a read-only diff over the journal's own history (§5) —
sequenced at §6, gated on the lead's rulings at §7.
