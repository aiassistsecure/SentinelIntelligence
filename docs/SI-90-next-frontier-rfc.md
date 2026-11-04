# SI-90 RFC — The Next Frontier After the Training-Journal Line (Design Only)

> **Status: DESIGN ONLY.** Derived from merged evidence on `main` at `ee0a2474c920a714a86ed4782f7a66276ae0c3c6`
> (PR #96 merge, SI-89). Changes no code, no test, no policy, and no gate behaviour. Nothing here is
> implemented, measured, captured, or run live. This RFC runs nothing live, journals no training run, enforces
> no retention, forks nothing, prunes nothing, and scores nothing.
> **Scope:** CPU only. No GPU, no cloud, no spend, no downloads. This is machine learning infrastructure
> throughout; the served model is a specialised machine learning model. Where Imagine is mentioned, it is
> stated as 1.6B parameters, full fine tune.
> **Relationship:** SI-83 specified the NEDB-journaled training line in its §9 (slices 1–5) and §6.4 point 3.
> Those slices are now merged (SI-84 to SI-89, PRs #91–#96; §1). No unconditional build slice remains in any
> merged spec (§3), so, following the repo precedent for an exhausted line (SI-18/20/22/24/26/28/30/32/34/36/
> 69/76/80), this RFC states what the completed line earned, inventories the other completed lines, names what
> is genuinely open and whose decision each item is, and poses the decision points for Mark (§4). It proposes
> no new line.
> **Honesty labels carried forward verbatim, unweakened:** §18.3 **NOT earned**; P5-R1 **UNMEASURED**;
> P5-R9 **UNMEASURED**; live model-authored intent **OCCURRED, scoped to SI-74 only**; L-3 **LOCKED**;
> A-2/A-3/A-4 **UNAPPROVED**; unlocks `[]`; D-F1 **OPEN**; F-line **STOPPED**.
>
> Where this RFC and an earlier doc disagree, the earlier doc's evidence wins until a merged slice updates
> it with new evidence.

## Table of contents

- [0. Honesty table: state after SI-89](#0-honesty-table-state-after-si-89)
- [1. What the training-journal line earned and closed](#1-what-the-training-journal-line-earned-and-closed)
- [2. Inventory of the other completed lines](#2-inventory-of-the-other-completed-lines)
- [3. What remains genuinely open](#3-what-remains-genuinely-open)
- [4. Decision points for Mark](#4-decision-points-for-mark)
- [5. Rejected alternatives](#5-rejected-alternatives)
- [6. Non-goals](#6-non-goals)
- [7. Honesty rules binding any successor line](#7-honesty-rules-binding-any-successor-line)
- [8. Sources and code references](#8-sources-and-code-references)

## 0. Honesty table: state after SI-89

No row changes. This RFC strengthens, weakens, and re-labels nothing, and it adds no unlock: unlocks remain `[]`.

| Element | State after SI-89 | Evidence / scope |
|---|---|---|
| SI-63 §18.3 claim | NOT earned | SI-74, SI-75, SI-79 §0: no live task completed |
| §18.1 row P5-R1 | UNMEASURED | SI-74 §4; no later slice promotes it |
| §18.1 row P5-R9 | UNMEASURED | SI-74 §4; SI-79 §4 notes a Resident death as an observation only |
| Live model-authored tool intent | OCCURRED, scoped to SI-74 only | 2026-11-02, P-5 live run only; 2 parsed intents, `fs_list` x2; 1 ADMIT + executed + verified, 1 ESCALATE + not executed. No other run extends that scope |
| §18.1 rows P5-R2–R8, R10 | As promoted, scoped, by SI-74 | Cited by later slices, promoted by none of them |
| L-3 residency-loss drill | LOCKED | SI-69 §4.5; SI-76 §3.6; nothing offline unlocks it |
| C-5 conditional L-3 path | LOCKED | SI-76 §4.7; no real L-2 live success |
| C-4 conditional L-2 retry | NOT ELIGIBLE | SI-76 §4.6; its conditions fail on SI-79 |
| C-2 conformance measurement | DONE, BELOW THRESHOLD | SI-79 (PR #86); 10/30 (0.3333) against D-C1 |
| Readiness unlocks (SI-79) | `[]` | `docs/SI-79-measurements.json`: `readiness.unlocks` |
| A-2 / A-3 / A-4 | UNAPPROVED | Not approved by SI-40, SI-63, SI-69, SI-76, SI-80, SI-83, or this RFC |
| D-F1 | OPEN | SI-80 §4; posed by SI-82 §5; Mark's decision |
| F-line | STOPPED at D-F1 | SI-80 §4; SI-82; SI-83 §10. No F-3+ work is built, designed, or proposed here |
| Training-journal line (§1) | COMPLETE, miniature/embedded scope | SI-83 §9 slices 1–5 plus §6.4 point 3: SI-84 to SI-89, PRs #91–#96 |
| Live-NEDB write for the training journal | NOT PERFORMED, NOT CLAIMED | SI-85 to SI-89 module scope notes: tests on the embedded backend only |
| Any Imagine run journaled / retention-enforced / forked / pruned by §1 code | NOT CLAIMED | Each §1 module states the negative plainly; Imagine is 1.6B parameters, full fine tune |

Consequences: (1) The training-journal line is complete as the spec wrote it — as miniature, embedded-backend
work (§1.8). Completeness of the slices is not a claim about any real training run. (2) Every remaining item
in §3 is gated on a named person's decision, is a stated non-build by design, or is an untested claim. None
of them is an unconditional next slice, and this RFC invents none.

## 1. What the training-journal line earned and closed

The spec is SI-83, `docs/SI-83-training-journal-spec.md` (PR #90): journal schema (§2), a non-invasive watcher
(§3), rollback/replay/fork semantics (§4), corpus-window prune lineage (§5), a disk/retention policy stated as
gated proposals (§6), and explicit database naming (§7). Each slice below is one PR, built by a builder and
reviewed and merged by the dev-loop driver on a later run, in the loop's standing pattern. Module docstrings
are cited as the slices' own scope statements; where a module limits itself, this section repeats the limit
rather than the headline.

### 1.1 SI-84 — the watcher (PR #91)

`si/training_watcher.py`. The observation half of SI-83 §3: polls a trainer's checkpoint root for `step<N>`
directories, treats a checkpoint as complete only after quiescence (stable file set and sizes, a non-empty
weights file, and the completion marker), stream-hashes weights (and optimizer, when present) with SHA-256
after quiescence, and appends `checkpoint_manifest` records in the §2.3 shape to a plain file record (JSON
lines) carrying the SI-0 hash chain. A stabilised but incomplete save is journaled with
`checkpoint_complete = false` and is never a rollback target; a restart that finds unjournaled checkpoints
first appends a `watcher_gap` row — a gap is a row, never silence. Proven against synthetic trainers in tests:
manifests match independently computed hashes, truncated saves are marked incomplete, gaps are journaled. Not
built here, by the module's own statement: no NEDB write path exists in this slice, the watcher never writes
into the trainer's tree, and it journals no real training run.

### 1.2 SI-85 — the journal writer and database (PR #92)

`si/training_journal.py`. The five §2.2 collections — `training_runs`, `training_checkpoints`,
`training_eval_moments`, `training_windows`, `training_proposals` — as separate collections, written through
the SI-0 journal's machinery (same `seq` clock, same hash chain, same backend protocol); no parallel scheme.
Database naming is fail-closed per SI-83 §7: the only accepted name is the literal `si_training_journal` from
explicit configuration; any other name — including `shop` and `stealth_shop`, and any derived name — refuses
to write, and every open performs the §7.3 read-back handshake before the first write. Proven in tests: chain
verification over every §2.5 event kind, wrong-name refusal, handshake failure closing the writer. Scope
stated in the module: for the real NEDB backend the read-back is the backend's opened `dbname` attribute, not
an independent server query, and **no live-NEDB write is performed or claimed by this slice's tests (embedded
only)**.

### 1.3 SI-86 — rollback and replay tooling (PR #93)

`si/training_rollback.py`. SI-83 §4.2 rehydrate-and-verify and §4.3 eval replay over the SI-85 journal,
reusing the SI-84 streaming hash. Rollback verifies a manifest in place — chain, completeness, weights hash,
optimizer hash when carried — fails closed on any mismatch, journals a `rollback` row with a required caller
reason, and returns a resume descriptor only; it launches no trainer. A weights-only resume is labelled
`weights-exact/optimizer-fresh`, never `exact`. Replay re-runs a caller-supplied eval function against a
hash-verified checkpoint and journals a new moment row; a disagreeing replay leaves both rows standing.
Forward training replay is explicitly NOT built in this slice, per its module docstring. Proven against
miniature checkpoints in tests; no real run was rolled back or replayed.

### 1.4 SI-87 — fork tooling and gated prune proposals (PR #94)

`si/training_fork_prune.py`. SI-83 §4.4 fork and the §5.3 propose → gate → journal → land flow for corpus
prunes. A fork verifies its source checkpoint first (complete and hash-verified in place, never copied),
requires a reason, and journals the fork proposal row and the new run row before returning a descriptor; the
new run references the source weights blob by the same hash. Lineage is walkable (`descendants` / `ancestors`,
including a fork of a fork). A prune with no traced window or no reason is rejected at the gate; an admitted
prune lands as a NEW corpus file with a new hash and a `pruned_from` edge, and the original corpus file is
byte-identical after the flow, hash-checked. Proven on miniature corpora in tests; no trainer is launched and
no real corpus was pruned.

### 1.5 SI-88 — retention enforcement (PR #95)

`si/training_retention.py`. SI-83 §6.4 retention as gated `retention_decision` proposals over the SI-85
journal, in the SI-87 flow pattern. The one deletion this code executes is an admitted proposal's optimizer
file, after journaling; the post-drop state is appended as a NEW manifest revision with
`optimizer_blob = null`, never an edit of history. Disk figures are recomputed from the §6.2 assumptions and
labelled `ESTIMATE`, or summed from manifest byte fields and labelled `MEASURED`; an estimate is never
presented as measured. Two limits are the slice's own: weights deletion is operator-only and is never admitted
or executed here (a `delete_weights` proposal ESCALATEs and no file is deleted; no code path in the module
deletes a weights file), and the slice is exercised on tiny checkpoint files through the embedded backend —
no live-NEDB write, and no run of Imagine (1.6B parameters, full fine tune) had retention enforced on it by
this code.

### 1.6 SI-89 — the SAVE_EVERY policy gate (PR #96)

`si/training_save_policy.py`, plus the `save_every_decision` journal kind in `si/training_journal.py`. The one
§6.4 item SI-88 left unbuilt — point 3: SAVE_EVERY below 500 on a full fine-tune requires its own gated
proposal naming the disk arithmetic for the new interval (the v12 lesson, enforced by policy rather than
memory). The gate is read-only; `propose_save_every` journals the verdict before returning. Invalid input
REJECTs; no reason refuses outright and journals nothing; a non-full-fine-tune or SAVE_EVERY ≥ 500 ADMITS;
SAVE_EVERY < 500 on a full fine-tune ESCALATEs, always — the gate never self-admits an exception. The module's
stated non-builds: no operator-approval execution path is built (ESCALATE is terminal for this slice), no
trainer is launched, no training config is written or modified, and nothing enforces the verdict in any
trainer. Tests run through the real writer on the embedded backend only; no live-NEDB write is performed or
claimed, and no run of Imagine (1.6B parameters, full fine tune) had a SAVE_EVERY change proposed or applied
by this code.

### 1.7 The spec item that closed the line

SI-83 §9 listed five slices in dependency order; SI-89 was not a sixth §9 slice but the §6.4 point 3 item SI-88
named as its remaining gap. With PR #96 merged, §9 slices 1–5 and §6.4 point 3 are all built. The lead's
full-suite run on the PR #96 tree is recorded in the loop's STATE.md, outside this repo; this RFC cites no
suite count of its own, because it ran nothing.

### 1.8 The line's common scope, stated once

Every §1 slice shares the same honest scope, stated in each module and repeated here so no later citation can
drop it: miniature checkpoints and corpora in temporary directories, the real writer on the embedded (sqlite)
backend, synthetic trainers for the watcher, and caller-supplied eval functions for replay. No slice writes to
a live NEDB, launches a trainer, enforces anything inside a trainer's configuration, or touches a real
training run. The journal, the gate shape, and the by-reference rule (hashes and paths, never weight bytes)
are real; the scale at which they have been exercised is miniature, and the backend is embedded.

## 2. Inventory of the other completed lines

Each entry cites the merged slice doc and the module that carries the work. "Complete" means the line's
sequenced slices are merged; the scope label each slice gave itself travels with the entry.

### 2.1 Gaia B-line — tool use (SI-42 to SI-46)

SI-41's §15 order, all in the `si/gaia/` subpackage. B-1 (SI-42): the `tool` intent kind, the full v1 registry
as data (`si/gaia/registry.py`), the §5 gate mapping, and shadow mode — no executor, every passing intent
ESCALATEd, execution recorded as not-evaluated. B-2 (SI-43): a real executor (`si/gaia/executor.py`) and the
read/write-class filesystem tools live on miniature bundles only; the execute class still ESCALATEd. B-3
(SI-44): `shell_exec`, `code_exec`, and `test_run` live through the same executor under its resource bounds,
plus the §10 task loop (`si/gaia/task.py`) — still miniature-scoped, still escalating any intent priced
`external`. B-4 (SI-45): the Hearth interface, SI-side only (`si/gaia/hearth.py`) — declaration record,
residency perception, gateway client refusal logic — exercised against labelled replay/stub transports; no
real Hearth node was contacted and the Hearth path's performance was UNMEASURED. B-5 (SI-46): agentic work as
a service, SI-side adapter (`si/gaia/service.py`) — a submitted task runs through the B-3 machinery and returns
its result with the provenance trail serialised from the journal. With B-5, the B-line the RFC sequenced is
complete on the SI side, at the scopes above.

### 2.2 SSE spine (SI-47 to SI-49, SI-58)

SI-47: the event spine (`si/gaia/events.py`) — every journal row serialised as a server-sent event whose id IS
the journal seq, resumable via `Last-Event-ID`, exposed as `GET /events` on the B-5 service; the journal stays
the source of truth. SI-48: server-side filtered subscriptions (`collection` / `event` / `kind` / `task`) as a
projection of that same stream, plus the stdlib reference consumer (`si/gaia/events_client.py`). SI-49: the
consumer's resilient following layer — reconnect with `Last-Event-ID` at the last seq actually received,
bounded backoff, dedup by seq, fail-closed on malformed resume. SI-58 later fixed the live-tail path of the
same spine (see its slice doc). No entry here adds a second store; filtering and following are projections and
clients of the one journal stream.

### 2.3 Queue-drain Q-line (SI-50 to SI-54)

SI-50 is the design spec (the journal IS the queue); Q-1 to Q-4 built it. Q-1 (SI-51): the queue core — batch
proposals with per-member verdicts, claim-before-execute, the serial drainer, batched perception
(`si/gaia/queue.py`, `si/gaia/drain.py`). Q-2 (SI-52): `depends_on` requirement semantics, barrier sync
points, claim-time residence refusals, and the bounded read pool — concurrency confined to unkeyed read-class
intents; writes and executions stay serial. Q-3 (SI-53): the leash against a live drain, operator dispositions
for outcome-unknown claims (`accept-unknown` / `compensate` / `re-propose`; there is no `mark-executed`), and
queue exposure. Q-4 (SI-54): the measurement, on the record (`si/gaia/measure.py`,
`docs/SI-54-measurements.json`) — scripted miniature workloads only; its headline is honestly negative for the
queue at that scale (the serial cycle was faster at every bound for every workload measured), with the
read-bound benefit plateauing at bound 4 in its focused sweep.

### 2.4 Seam S-line (S-2a to S-7)

The SI half of the G-9 seam design (the design itself lives in the gaia fork repo; the SI-side docs are the
evidence here). S-2a: proposal ingest — fork watcher proposals gated and journaled on the B-5 service, ACK
after journal (`si/gaia/seam.py`). S-2b (SI-55): the `hearth.*` tools registered, so watcher proposals gate on
their merits; local execution of `hearth.*` does not exist. S-3 (SI-56): work submission as an explicit step
(`si/gaia/seam_submit.py`) — a submitted item is `submitted/awaiting outcome`, never "completed" by submission.
S-4 (SI-57): leash crossing (`si/gaia/seam_leash.py`). S-5 (SI-59): outcome return and reconciliation shipped
together (`si/gaia/seam_outcome.py`, `si/gaia/seam_reconcile.py`) — outcomes journaled as provenance,
disagreements journaled as integrity faults, never rewritten. S-6 (SI-60): SI journal events forwarded toward
the fork's SSE `source=si` buffer, SI side only (`si/gaia/seam_events.py`); the fork half is not built by the
SI-side slice. S-7: the measurement design (SI-61, design only) and its canonical run (SI-62,
`si/gaia/seam_measure.py`, `docs/S-7-seam-measurements.md`) — single-run, miniature fixture, embedded journal,
with live fork interop performance still UNMEASURED in every row.

### 2.5 Proposer P-line (SI-64 to SI-68)

SI-63's §19 order, in `si/gaia/proposer.py` unless noted. P-1 (SI-64): perception assembly (§6) and catalogue
rendering (§7) as pure journal-derived functions; `propose` returned the labelled not-evaluated outcome and
made no gateway call. P-2 (SI-65): the §8 output contract and the §9 six-stage parser, replay only. P-3
(SI-66): gateway wiring under preconditions (§10/§11), the §6.5 request assembly, and the §6.6 standing
instructions — exercised only against replay transports and refusal fixtures; the real Hearth path remained
UNMEASURED. P-4 (SI-67): first live proposer — the wiring (`si/gaia/live.py`) completed live discovery, a live
residency poll, and live served-model discovery, then the fixture suite did not run because the node was lost
mid-slice; its doc records the honest failure variant, not a completion. P-5 (SI-68): the measurement harness
(`si/gaia/propose_measure.py`) and hardening, exercised only against fakes and the labelled replay transport,
carrying every §18.1 row forward as UNMEASURED in `docs/SI-68-measurements.json`. The later live evidence for
this line is the L-line's, in §2.6 — in particular SI-74, the sole scope of the OCCURRED label in §0.

### 2.6 L-line and C-line — status per SI-80 §0–§1

This RFC re-opens nothing here and re-cuts no number; it carries SI-80's statement forward. The L-line (SI-69
§4) produced: L-1, the replay corpus v2 (SI-70); the L-5 seam label reconciliation (SI-71); L-2 live re-runs
(SI-72, superseded as the live record by SI-74) — SI-74 is the run in which live model-authored intent
OCCURRED, scoped to SI-74 only (§0), completing 0 of 3 fixture tasks end-to-end, so no L-2 live success exists;
L-4 tuning from live evidence (SI-75), whose list was deliberately empty (§2.7's root cause is a
model-output-content failure, and no §15.2 policy key governs what the model says); and L-3, which never ran
and stays LOCKED. The C-line (SI-76 §4) produced: C-1, the offline conformance harness (SI-77); C-3, the
threshold proposal adopted by Mark on 2026-11-02 as D-C1 (SI-78: success rate ≥ 0.80 over N ≥ 30, ≥ 5 runs per
workload, per-workload floor ≥ 0.60); and C-2, the conformance record (SI-79): 10/30 (0.3333), BELOW THRESHOLD
against D-C1, unlocks `[]`. C-4 is NOT ELIGIBLE on its original conditions and C-5 stays LOCKED (§0). F-1
(SI-81) then taxonomised the SI-79 corpus offline (stage-1 classes 12 / 5 / 1 summing to 18, the stage-4 listing
of 2, control group of 10, `unclassified` 0) and F-2 (SI-82) posed D-F1. That is where the F-line stopped (§3.1).

### 2.7 What the inventory does not say

No line above is a live-evidence line beyond its own stated scope. The B-line and Q-line execute on miniature
bundles; the seam measurements are single-run on a miniature fixture; the P-line's live moment is SI-74's,
scoped exactly as §0 states; and the C-line's one measured verdict is a refusal at the decided bar. An
inventory of completed slices is a record of what was built and verified, at the scale it was verified at —
never a claim that the same work has been exercised at production scale, against a live NEDB, or on a real
training run, except where a cited doc says exactly that, with its evidence.

## 3. What remains genuinely open

Each item is labelled with whose decision it is, or that it is a non-build by design. An item's presence here
is not a proposal to build it; §4 poses the items that are questions.

### 3.1 D-F1 — Mark's decision, OPEN

The admissibility of any SI-side, non-model intervention (perception, catalogue, instructions, output contract,
or gateway assembly — the classes SI-82 §4 names) is undecided. SI-82 posed the question and answered nothing;
SI-83 §10 recorded that the training-journal line presumes no answer to it. Until Mark answers, the F-line
stays STOPPED and no slice may touch those classes beyond posing the question exactly as SI-80 §4 and SI-82 §5
pose it (§4, D-90-1).

### 3.2 Trainer integration for the training journal — unbuilt, unapproved as a slice

The §1 slices are a journal, a watcher, and gated tools around a trainer; none of them is integrated with one.
SI-89's gate journals a verdict that nothing enforces in any trainer; SI-86 and SI-87 return descriptors that
a caller must act on outside the slices; SI-84 observes a checkpoint tree it does not operate. Wiring any of
this into a real trainer's configuration or run loop is unbuilt, and no merged doc approves it as a slice.
Whether it is ever built, and against which trainer, is Mark's decision (§4, D-90-2) — not a gap this RFC
schedules a fix for.

### 3.3 Live-NEDB verification for the training journal — an untested claim, not a capability

Every §1 test ran on the embedded backend. SI-85's handshake is real code, but for the real NEDB backend its
read-back is the backend's opened `dbname` attribute, not an independent server query, and no §1 slice
performed or claimed a live-NEDB write. A training journal verified against a live NEDB — writes, chain
verification, and read-back over the real backend — does not exist as evidence. Until a slice runs it and
reports it, "the training journal works on live NEDB" is an untested claim, and no doc may state it as a
capability (§4, D-90-3).

### 3.4 A future conformance candidate — a question for Mark, nothing named

SI-79 measured one candidate against D-C1 and the verdict was BELOW THRESHOLD; that verdict stands and is not
re-opened here. Whether any future candidate is ever named for a fresh conformance measurement, and if so
which one and when, is Mark's decision (§4, D-90-4). This RFC names no candidate, authorises none, and asserts
none exists, is ready, or would pass. In particular it makes no assertion about any model that it cannot cite
to an in-repo source.

### 3.5 Stated non-builds, by design

These are closed by the slices that stated them, not open work awaiting a builder:

- **Weights-deletion execution** (SI-88): operator-only. The gate ESCALATEs a `delete_weights` proposal and no code path executes it. That is the design, not a defect.
- **Operator-approval execution path for ESCALATE** (SI-89): terminal by design. An operator's later decision on an escalated SAVE_EVERY proposal is not recorded or acted on by that code.
- **Forward training replay** (SI-86): explicitly not built; §4.3 eval replay is what exists.
- **Local execution of `hearth.*` tools** (S-2b): does not exist; the registry admits the names for gating, and the executor refuses them by construction.
- **The fork half of S-6 event ingest**: fork-repo work, not an SI-repo slice, and not assigned by this RFC.

## 4. Decision points for Mark

Mark's decisions; this RFC decides none, and no successor slice may pre-empt them. All are OPEN as of this RFC.

- **D-90-1 — D-F1, re-posed verbatim in substance (SI-80 §4, as posed by SI-82 §5).** *Is perception/prompt-side
  remediation in scope at all? May the loop test SI-side, non-model interventions (e.g. perception/prompt
  rendering changes) as a route from the SI-79 result — accepting that a changed rendering is a changed
  candidate configuration requiring fresh-corpus re-measurement?* The intervention classes the question could
  cover are exactly SI-82 §4's: perception assembly and rendering, tool catalogue rendering, standing
  instructions, the output-contract instruction, and gateway-request assembly. *If no:* the F-line ends at
  F-2's question, answered; F-1's taxonomy stands as the record, and the rendering as it stands remains the
  only rendering any existing measurement describes. *If yes, for a named class:* the conditional slice SI-80
  §3.5 defines opens for that class only — an answer in scope for one named class is not an answer for any
  other, and is not a measurement, a rate, or a claim. This RFC adds no class, no recommendation, and no
  predicted effect, in any direction.
- **D-90-2 — Trainer integration.** Should any §1 component be integrated with a real trainer — a verdict
  enforced in a trainer's configuration, a descriptor acted on by a run loop, a watcher attached to a real
  run — and if so, which component, which trainer, and under what gate? *If no:* the training journal stays a
  verified miniature/embedded toolkit, honestly labelled as such wherever it is cited. *If yes, for a named
  component:* that integration is a new slice requiring its own brief and its own verification against the
  real trainer; no §1 test result transfers to it as evidence.
- **D-90-3 — Live-NEDB verification.** Should a slice verify the training journal against a live NEDB — real
  writes, chain verification, and read-back over the real backend — and on which named database (SI-83 §7's
  discipline: the name is `si_training_journal`, verified before any write, never silently defaulted)? *If
  no:* the embedded-only label in §1.8 stands as the permanent scope of the claim. *If yes:* the slice reports
  what the live backend actually did, including any failure, and only its own run may retire the "untested
  claim" label in §3.3.
- **D-90-4 — A future conformance candidate.** If a future candidate is ever to be conformance-measured, which
  one, named by Mark under a D-C3-like decision, and when? Naming none parks the question without penalty;
  this RFC names none (§3.4). *If a candidate is named:* any measurement is a separate slice run exactly as
  SI-79 ran — a fresh corpus captured for that measurement, scored through the existing C-1 harness and §9
  parser unmodified, and compared against D-C1 as decided — and it inherits no expectation from SI-79's
  verdict, in either direction. *If none is named:* SI-79 remains the last conformance record, and its verdict
  stands as the line's answer.
- **D-90-5 — Does D-C1 stand?** SI-80's D-F4, carried forward unanswered: D-C1 (≥ 0.80 over N ≥ 30, ≥ 5 per
  workload, floor ≥ 0.60) is assumed to stand as decided on 2026-11-02 until Mark decides otherwise.
  *Confirming it* costs nothing and removes all doubt for any future measurement. *Changing it* is a new
  decision, stated as such, and never applies retroactively to SI-79's verdict.

## 5. Rejected alternatives

- **5.1 Invent a successor line to keep building.** Rejected. Every prior next-frontier RFC derived its line
  from merged evidence with an unconditional first slice. Here the evidence supports no such slice: the open
  items are decisions (§4), non-builds by design (§3.5), or an untested claim (§3.3). Naming a line anyway
  would be a backlog invented to avoid stopping, and §7 forbids it.
- **5.2 Treat the training-journal line's completion as a claim about a real run.** Rejected (§1.8): miniature
  checkpoints, synthetic trainers, and the embedded backend prove the machinery at that scale. They do not
  prove a journaled, rolled-back, forked, pruned, or retention-enforced run of Imagine (1.6B parameters, full
  fine tune), and no wording may slide from one to the other.
- **5.3 Describe live NEDB as the training journal's backend.** Rejected (§3.3): the tests are embedded-only
  and SI-85's real-backend read-back is an attribute check, not a live write. The honest label is untested, not
  implied-verified.
- **5.4 Answer D-F1 inside this RFC, or design around an assumed answer.** Rejected (SI-80 §6.5, SI-82 §7): the
  question is Mark's, and a roadmap that only makes sense under one answer presumes it. §4 poses it with each
  answer's consequence and nothing more.
- **5.5 Re-open D-C1 or re-cut SI-79's numbers.** Rejected (SI-80 §5, §6.2): a threshold moved after the
  measurement it judges is not a threshold, and subsets of the same 30 bodies are the same move in smaller
  pieces. The verdict is BELOW THRESHOLD; it stands.
- **5.6 Name or design around a future conformance candidate.** Rejected (§3.4): none has been named by Mark,
  and designing slices that only make sense for a specific future model would fabricate the premise any future
  measurement waits on.
- **5.7 Approve or pre-design A-2/A-3/A-4 here.** Rejected (SI-76 §6.7, SI-80 §6.7): they remain separate
  approvals no merged document grants, and a design-only RFC must not smuggle one. They stay UNAPPROVED.
- **5.8 Run C-4 or retry L-2 on the strength of this RFC.** Rejected (SI-80 §6.1): C-4's conditions fail on the
  SI-79 verdict, and nothing in the training-journal line changes a single one of them.

## 6. Non-goals

- Design only: no code, tests, configuration, capture, measurement, or live call in this RFC, and no new number
  of any kind is published here — §1 and §2 report only what merged slices already reported, cited to them.
- No successor line is named, sequenced, or scheduled. Unlisted slices need Mark's decision (§4) or a successor
  RFC, never an implied extension of this one.
- No decision in §4 is answered, recommended, or presented as expected. D-F1 in particular is posed, not argued.
- No candidate is named as existing, available, or expected to pass (§3.4, D-90-4). No threshold is decided,
  re-decided, or weakened (§5.5, D-90-5).
- C-4 stays NOT ELIGIBLE and C-5 / L-3 stay LOCKED (§0); nothing here is their approval, scheduling, or
  pre-design. A-2/A-3/A-4 remain UNAPPROVED and out of scope.
- No trainer integration, live-NEDB run, node relaunch, fork-repo work, or Hearth-side change is assigned,
  designed in detail, or claimed (§3.2, §3.3, §3.5).
- No change is proposed to §8, §9, any §15.2 key, any gate, the registry, the queue, the SSE spine, or any
  training journal module.

## 7. Honesty rules binding any successor line

These bind every slice built under any §4 decision or any successor RFC, exactly as SI-80 §5 bound the F-line:

- **Offline and miniature work is never live evidence.** Embedded-backend tests, replay transports, synthetic
  trainers, and miniature fixtures earn no §18.1 row, no §18.3 progress, no L-3 unlock, and no live
  authorisation, whatever line they are built under.
- **Scope labels travel with the claim.** A result obtained on miniature bundles, miniature checkpoints, or the
  embedded backend is cited with that scope attached, in the citing doc's own honesty table — never summarised
  into an unscoped capability.
- **No threshold weakening or re-cutting.** D-C1 is not re-proposed, rounded, subset, or re-based to change any
  recorded verdict. A changed threshold is a new decision by Mark, prospective only (D-90-5).
- **No re-scoring old bodies as a new measurement.** Any future measurement uses a fresh corpus captured for
  it (SI-76 §3.3(6)); recorded bodies are inputs to offline analysis, never a new rate.
- **A changed configuration is a changed candidate.** Anything altered on SI's side — rendering, integration,
  backend — makes prior measurements describe the prior configuration only (SI-80 §3.4 risk 1).
- **No candidate is claimed to exist, be available, or pass** without a naming decision by Mark and a fresh
  measurement under the unchanged apparatus (D-90-4).
- **Never fake admission, probes, or inference; never invent metadata.** Unknown fields stay `unknown` / `None`
  (the §1 modules' standing practice); a gap is a journaled row, never silence; an environmental failure is a
  reportable outcome, never worked around with replay (SI-67, SI-68).
- **Never duplicate bytes.** Checkpoint and corpus records reference weights and bodies by hash and path,
  stored once; no successor slice copies model, GGUF, tensor, or corpus bytes to make a claim easier to state.
- **Terminology and means:** machine learning throughout; Imagine, wherever mentioned, is 1.6B parameters,
  full fine tune. CPU only; no GPU, no cloud, no money, no secrets — unless a §4 decision by Mark explicitly
  says otherwise for a named slice.

## 8. Sources and code references

All on `main` at `ee0a2474c920a714a86ed4782f7a66276ae0c3c6` unless a doc number is given. Where an earlier doc
and this RFC disagree, the earlier doc's evidence wins (see header).

- `docs/SI-83-training-journal-spec.md` — the training journal spec (PR #90): §2 schema, §6.4 retention points, §7 database naming, §9 slice plan, §10 honesty / D-F1 / F-line status.
- `si/training_watcher.py` — SI-84 (PR #91): watcher scope, plain-file record, no NEDB write path, no real run journaled, in its module docstring.
- `si/training_journal.py` — SI-85 (PR #92) and SI-89's journal kind (PR #96): five collections, `si_training_journal` fail-closed naming, §7.3 handshake, embedded-only tests, no live-NEDB write performed or claimed.
- `si/training_rollback.py` — SI-86 (PR #93): §4.2 verify / rollback, §4.3 eval replay, forward training replay NOT built, weights-only labelling.
- `si/training_fork_prune.py` — SI-87 (PR #94): §4.4 fork, §5.3 gated prune, by-hash reference, original corpus byte-identical after an admitted prune.
- `si/training_retention.py` — SI-88 (PR #95): §6.4 gated retention, optimizer-drop only, weights deletion operator-only and never executed, ESTIMATE / MEASURED labelling.
- `si/training_save_policy.py` — SI-89 (PR #96): §6.4 point 3 gate, ESCALATE always below 500 on a full fine tune, no operator-approval execution path, no trainer enforcement, embedded only.
- `docs/SI-42-gaia-b1-tool-intent-shadow.md` to `docs/SI-46-gaia-b5-service-adapter.md` — the B-line; `si/gaia/registry.py`, `si/gaia/executor.py`, `si/gaia/task.py`, `si/gaia/hearth.py`, `si/gaia/service.py` as the code it built.
- `docs/SI-47-gaia-sse-event-spine.md`, `docs/SI-48-gaia-sse-filtered-subscriptions.md`, `docs/SI-49-gaia-sse-resilient-follower.md`, `docs/SI-58-sse-live-tail-fix.md` — the SSE spine; `si/gaia/events.py`, `si/gaia/events_client.py`.
- `docs/SI-50-queue-drain-spec.md`, `docs/SI-51-q1-queue-core.md`, `docs/SI-52-q2-concurrency-deps.md`, `docs/SI-53-q3-leash-dispositions.md`, `docs/SI-54-q4-measurement.md`, `docs/SI-54-measurements.json` — the Q-line; `si/gaia/queue.py`, `si/gaia/drain.py`, `si/gaia/measure.py`.
- `docs/S-2a-proposal-ingest.md`, `docs/S-2b-hearth-registry.md`, `docs/S-3-work-submission.md`, `docs/S-4-leash-crossing.md`, `docs/S-5-outcome-return.md`, `docs/S-6-si-events-forwarder.md`, `docs/S-7-seam-measurement-design.md`, `docs/S-7-seam-measurements.md` — the seam S-line; `si/gaia/seam.py`, `si/gaia/seam_submit.py`, `si/gaia/seam_leash.py`, `si/gaia/seam_outcome.py`, `si/gaia/seam_reconcile.py`, `si/gaia/seam_events.py`, `si/gaia/seam_measure.py`.
- `docs/SI-64-p1-perception-catalogue.md` to `docs/SI-68-p5-measurement-hardening.md`, `docs/SI-68-measurements.json` — the P-line; `si/gaia/proposer.py`, `si/gaia/live.py`, `si/gaia/propose_measure.py`.
- `docs/SI-80-next-frontier-rfc.md` — §§0–1 for the L-line and C-line status carried in §2.6, §4 for D-F1 (re-posed in §4 as D-90-1) and D-F4 (carried as D-90-5), §§5–7 for the rejected alternatives and honesty rules §7 inherits.
- `docs/SI-82-f2-admissibility-question.md` — F-2: §4's intervention classes, §5's posing of D-F1, §6's honesty risks.
- `docs/SI-81-f1-failure-taxonomy.md`, `docs/SI-81-taxonomy.json` — F-1: the SI-79 corpus classes cited in §2.6.
- `docs/SI-79-c2-conformance-record.md`, `docs/SI-79-measurements.json` — C-2 as measured: 10/30 (0.3333), BELOW THRESHOLD against D-C1, unlocks `[]`.
- `docs/SI-74-l2-live-rerun.md`, `docs/SI-74-measurements.json` — the sole scope of the OCCURRED label; P5-R1 / P5-R9 UNMEASURED.
- `docs/SI-40-a1-feasibility-measurement.md` — A-1 measured; A-2/A-3/A-4 unapproved.

End of RFC SI-90. The training-journal line is complete at its stated scope (§1); no unconditional slice
follows from it. What happens next is one of the decisions in §4, made by the person each decision names — or
nothing, which §5.1 records as a complete outcome, not a stall to be filled.
