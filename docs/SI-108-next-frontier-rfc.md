# SI-108 RFC — The Next Frontier After the Closed v13 Conformance Frontier (Design Only)

> **Status: DESIGN ONLY.** Derived from merged evidence on `main` at
> `98846f6fe1bdb0b5a6f3dac67dfe41a2a5110f22` (Merge PR #115, SI-107 record).
> Changes no code, no test, no policy, and no gate behaviour. Nothing here is
> implemented, measured, captured, or run live. This RFC runs no graded
> conformance run, captures no corpus, scores nothing, and names no new
> candidate.
> **Scope:** CPU only. No GPU, no cloud, no spend, no downloads. This is
> machine learning infrastructure throughout; the served model is a
> specialised machine learning model. Where Imagine is mentioned, it is
> stated as 1.6B parameters, full fine tune.
> **Relationship:** SI-90 posed the decisions the v13 line then executed;
> SI-91 recorded D-C1 standing, named Imagine v13 as the candidate, admitted
> the perception-rendering class only, and deferred D-90-2 / D-90-3. SI-94,
> SI-100, and SI-105 graded three v13 configurations; SI-106 designed the
> C2 multi-cycle diagnostic and SI-107 ran it. The lead's synthesis of
> 2026-11-12 ~16:30 EST and merge note of 2026-11-13 ~02:55 EST close the
> v13 conformance frontier. This RFC states what that closure earned,
> inventories the Gaia lines as merged, derives the one sequenced
> in-repo remainder (§4), and records — without inventing — where no
> unambiguous next Gaia slice exists (§5).
> **Honesty labels carried forward verbatim, unweakened:** §18.3 **NOT
> earned**; P5-R1 **UNMEASURED**; P5-R9 **UNMEASURED**; live
> model-authored intent **OCCURRED, scoped to SI-74 only**; L-3
> **LOCKED**; A-2/A-3/A-4 **UNAPPROVED**; unlocks `[]`; D-C1 **stands**.

## Table of contents

- [0. Honesty table: state after SI-107](#0-honesty-table-state-after-si-107)
- [1. The closed v13 conformance frontier](#1-the-closed-v13-conformance-frontier)
- [2. Inventory of the merged Gaia lines](#2-inventory-of-the-merged-gaia-lines)
- [3. What is genuinely not next](#3-what-is-genuinely-not-next)
- [4. The derived remainder: SI-106 §12, in RFC order](#4-the-derived-remainder-si-106-12-in-rfc-order)
- [5. Where no unambiguous next Gaia slice can be derived](#5-where-no-unambiguous-next-gaia-slice-can-be-derived)
- [6. Rejected alternatives](#6-rejected-alternatives)
- [7. Non-goals](#7-non-goals)
- [8. Honesty rules binding any successor slice](#8-honesty-rules-binding-any-successor-slice)
- [9. Sources and code references](#9-sources-and-code-references)

## 0. Honesty table: state after SI-107

No row is promoted, weakened, or re-labelled by this RFC, and it adds no
unlock: unlocks remain `[]`.

| Element | State after SI-107 | Evidence / scope |
|---|---|---|
| D-C1 | **Stands** — rate ≥ 0.80 over N ≥ 30, ≥ 5 per workload, per-workload floor ≥ 0.60 | Decided 2026-11-02; confirmed standing by SI-91 §3 (lead decision, 2026-11-11) |
| SI-94 (v13, v1 teaching, plain rendering) | **20/30 = 0.6667, BELOW THRESHOLD** | `docs/SI-94-c2-conformance-record.md` |
| SI-100 (v13, v2 teaching) | **9/30 = 0.30, BELOW THRESHOLD** | `docs/SI-100-v13-v2-conformance-record.md`; v2 teaching **withdrawn by SI-102**, SI-98 parser amendment stays |
| SI-105 (v13, v1 teaching + A1 rendering) | **18/30 = 0.60, BELOW THRESHOLD** | `docs/SI-105-v13-a1-conformance-record.md`; A1 did not beat the SI-94 baseline |
| SI-107 (C2 multi-cycle diagnostic) | **Complete, INCOMPLETE-DIAGNOSTIC** per SI-106 §9.7 — no verdict, unlocks `[]` | `docs/SI-107-c2-multicycle-diagnostic-record.md` (§1–§2) |
| SI-63 §18.3 claim | NOT earned | SI-94, SI-100, SI-105, SI-107 — no record earns it |
| §18.1 rows P5-R1, P5-R9 | UNMEASURED | Carried unchanged through SI-107 |
| Live model-authored tool intent | OCCURRED, scoped to SI-74 only | No later run extends that scope |
| L-3 residency-loss drill | LOCKED | Nothing in the v13 line unlocks it |
| C-4 for v13 | Does not proceed | On the SI-94, SI-100, or SI-105 record |
| A-2 / A-3 / A-4 | UNAPPROVED | Separate approvals; no merged document grants them |
| F-line harness investigation for v13 | COMPLETE | Lead synthesis, 2026-11-12 ~16:30 EST (§1) |
| D-90-2 (trainer wiring) | DEFERRED — recorded, not decided, not activated | SI-91 §6; lead merge note, 2026-11-13 ~02:55 EST |
| D-90-3 (live-NEDB proof) | DEFERRED — recorded, not decided, not activated | SI-91 §6; lead merge note, 2026-11-13 ~02:55 EST |

Consequence: every graded v13 configuration stands **below** the bar that
stands. That is the frontier's answer, not a prompt to move the bar, re-cut
a corpus, or invent a new harness lever (§3, §6).

## 1. The closed v13 conformance frontier

### 1.1 The three graded configurations

Three configurations of one candidate were measured under one unchanged
apparatus — the C-1 harness and the §9 parser, fresh corpora, D-C1 as
decided — and each measurement is a separate record of a separate
configuration. No number is lent between them:

- **SI-94 — 20/30.** V1 teaching, plain perception rendering, serving
  corrected per SI-93's diagnosis (the v13 GGUF as originally served
  carried no chat template; the trained template was supplied at
  serving). BELOW THRESHOLD.
- **SI-100 — 9/30.** V2 teaching (the §8 intra-document sibling form,
  SI-97 spec, SI-98 parser amendment, SI-99 teaching). BELOW THRESHOLD.
  The teaching was **withdrawn by SI-102** — the contract text restored
  to exactly the v1 text SI-94 measured — while the parser amendment
  stays in force for a future candidate trained with the form.
- **SI-105 — 18/30.** V1 teaching plus the A1 perception rendering
  (SI-103 design, SI-104 implementation: the citable reference set
  rendered as data). BELOW THRESHOLD; the perception-truncation workload
  again failed the floor (0.20), and A1 did not beat the SI-94 baseline.

### 1.2 The diagnostic that closed the measurement designs

SI-106 designed C2, the multi-cycle diagnostic (D-96-4), as a
**verdict-free** measurement: no bar, no verdict language, no unlocks,
fold-supplied scoring as the first corpus under D-96-3. SI-107 ran it
under that design. Its record states, whole:

- Capture closed with 30/30 task runs ended (19 completed,
  11 budget-exhausted), 66 documents, and **M = 31** scored documents —
  against the 60 a full-shape capture yields, with perception-truncation
  contributing no scored documents. The corpus therefore closes
  **INCOMPLETE-DIAGNOSTIC per SI-106 §9.7**: counts stated whole, no
  headline rate over M quoted, no verdict, unlocks `[]`.
- **17 of 31** scored documents were parse-ok. The stage histogram over
  the 31 was 1:9, 2:0, 3:0, 4:5, 5:0, 6:0.
- Fold-supplied scoring ran end to end: the frozen fold was **371 rows**,
  tip seq 371, sha256
  `10756d341b9a447ec5db21c11192e822e0d554057475efdaa84366300697f0a3`,
  re-verified before scoring.
- Reference forms, priced separately: the journal-seq form occurred
  **4 times, 0 priced-and-passed** (3 × `seq 0`, 1 × the item's own task
  seq); the sibling form occurred **0 times** in the whole capture.

These are diagnostic counts beside the graded records, never a fourth
grade (§10 of SI-106: comparability is none, in either direction).

### 1.3 Why the frontier is closed, in the lead's own synthesis

The lead's synthesis (2026-11-12 ~16:30 EST, recorded in the loop's
STATE.md and applied through SI-106/SI-107) is adopted here without
embellishment: harness-side variation moved v13 within roughly
9–20/30 and no harness lever reached D-C1; the F-line's harness
investigation for v13 is **complete** — it found and fixed a serving
artifact (SI-93), built and priced a protocol extension
(SI-96 to SI-102), and priced the perception lever (SI-103 to SI-105).
C2 was the last open measurement design, and it is now run and recorded.
**No new harness classes are to be invented for v13.**

The next graded candidate, if one is ever graded, is a **training-side
event — Mark's lane**. This RFC names no candidate, proposes no new
graded v13 run, and designs nothing that only makes sense for a specific
future model (§6).

## 2. Inventory of the merged Gaia lines

Gaia is the agentic layer in the `si/gaia/` subpackage: tool intents in
the self-write loop, the gate before execution, the NEDB journal as the
source of truth, and the SSE spine derived from that journal (standing
priority, Mark, 2026-10-29: expand and expose the tool surface slice by
slice, gate and journal before execution). "Complete" below means the
line's sequenced slices are merged at the scope each slice stated for
itself; the scope label travels with the entry.

### 2.1 B-line — tool use (SI-42 to SI-46, from SI-41 §15)

B-1 (SI-42): the `tool` intent kind, the full v1 registry as data
(`si/gaia/registry.py`), the gate mapping, shadow mode — no executor.
B-2 (SI-43): a real executor (`si/gaia/executor.py`) and the
read/write-class filesystem and house tools live, miniature-scoped.
B-3 (SI-44): `shell_exec`, `code_exec`, and `test_run` live through the
same executor under its resource bounds, plus the task loop
(`si/gaia/task.py`). B-4 (SI-45): the Hearth interface, SI-side only
(`si/gaia/hearth.py`), exercised against replay/stub transports; no
real Hearth node was contacted by that slice. B-5 (SI-46): agentic work
as a service, SI-side adapter (`si/gaia/service.py`) — a submitted task
returns its result with the provenance trail serialised from the
journal. The registry on `main` today holds exactly the SI-41 §8 v1
surface (12 tools) plus the two seam-crossing `hearth.*` names
registered by SI-55 (§2.4), which have no local executor by design.

### 2.2 SSE spine (SI-47 to SI-49, SI-58)

SI-47: every journal row serialised as a server-sent event whose id is
the journal seq, resumable via `Last-Event-ID`, exposed as `GET /events`
on the B-5 service; the journal stays the source of truth. SI-48:
server-side filtered subscriptions as a projection of that stream, plus
the stdlib reference consumer (`si/gaia/events_client.py`). SI-49: the
consumer's resilient following layer — reconnect at the last seq
actually received, bounded backoff, dedup by seq, fail-closed on
malformed resume. SI-58: the live-tail fix on the same spine. Filtering
and following are projections and clients of the one journal stream;
no second store exists.

### 2.3 Q-line — queue drain (SI-50 to SI-54)

SI-50 is the design (the journal IS the queue). Q-1 (SI-51): queue core,
claim-before-execute, the serial drainer, batched perception
(`si/gaia/queue.py`, `si/gaia/drain.py`). Q-2 (SI-52): `depends_on`
requirement semantics, barrier sync points, claim-time residence
refusals, a bounded read pool — concurrency confined to unkeyed
read-class intents. Q-3 (SI-53): the leash against a live drain and
operator dispositions for outcome-unknown claims (there is no
`mark-executed`). Q-4 (SI-54): the measurement, honestly negative for
the queue at miniature scale — the serial cycle was faster at every
bound for every workload measured.

### 2.4 Seam S-line (S-2a to S-7)

The SI half of the seam design. S-2a: proposal ingest, ACK after journal
(`si/gaia/seam.py`). S-2b (SI-55): the `hearth.*` tools registered so
watcher proposals gate on their merits; local execution of `hearth.*`
does not exist. S-3 (SI-56): work submission as an explicit step — a
submitted item is awaiting outcome, never "completed" by submission.
S-4 (SI-57): leash crossing. S-5 (SI-59): outcome return and
reconciliation — disagreements journaled as integrity faults, never
rewritten. S-6 (SI-60): SI journal events forwarded toward the fork's
SSE `source=si` buffer, SI side only; the fork half is fork-repo work,
not an SI-repo slice. S-7 (SI-61 design, SI-62 run): a single-run,
miniature-fixture, embedded-journal measurement, with live fork
interop performance UNMEASURED in every row.

### 2.5 P-line — the thinking proposer (SI-64 to SI-68, from SI-63 §19)

P-1 (SI-64): perception assembly and catalogue rendering as pure
journal-derived functions; no gateway call. P-2 (SI-65): the output
contract and the six-stage parser, replay only. P-3 (SI-66): gateway
wiring under preconditions, exercised only against replay transports
and refusal fixtures. P-4 (SI-67): first live proposer wiring; the
fixture suite did not run because the node was lost mid-slice, and its
doc records that failure honestly. P-5 (SI-68): the measurement harness
and hardening, exercised only against fakes and the labelled replay
transport, every §18.1 row carried as UNMEASURED. The later live
evidence for this line is the L-line's (§2.6).

### 2.6 L-line, C-line, and the lines that followed them

Per SI-80 and SI-90, carried forward without re-cutting any number:
L-1 replay corpus v2 (SI-70); L-5 seam label reconciliation (SI-71);
L-2 live re-runs (SI-72, superseded as the live record by SI-74 — the
run in which live model-authored intent OCCURRED, scoped to SI-74
only, completing 0 of 3 fixture tasks end-to-end); L-4 tuning from live
evidence (SI-75), its list deliberately empty; L-3 LOCKED, never run.
C-1 offline conformance harness (SI-77); C-3 threshold proposal adopted
as D-C1 (SI-78); C-2 conformance record for v12 (SI-79): 10/30, BELOW
THRESHOLD, unlocks `[]`. C-4 NOT ELIGIBLE on its original conditions;
C-5 LOCKED. Then the F-line and the v13 line of §1: F-1 taxonomy
(SI-81), F-2 admissibility question (SI-82), the SI-91 decisions, and
SI-92 to SI-107 as §1 recounts. The escalation-resolution slice
(SI-92, `si/gaia/resolution.py`) is also merged: escalated intents have
an operator resolution path, recorded in the journal.

### 2.7 What the inventory does not say

No line above is a live-evidence line beyond its own stated scope. The
B-line and Q-line execute on miniature bundles; the seam measurements
are single-run on a miniature fixture; the P-line's live moment is
SI-74's, scoped exactly as §0 states. An inventory of completed slices
is a record of what was built and verified, at the scale it was
verified at — never a claim of production-scale, real-base, or
live-NEDB exercise, except where a cited doc says exactly that.

## 3. What is genuinely not next

Each item is a non-next by a recorded decision, a lock, or an approval
that does not exist. Listing them here is the derivation's negative
half: a successor RFC that ignored any of them would be inventing work.

- **No new graded v13 run.** All three graded configurations are below
  D-C1, D-C1 stands, and §1.3 closes the harness investigation. A
  re-run, a re-cut subset, or a new harness class for v13 is not a
  slice; it is the move SI-90 §5.5 and SI-106 §3 forbid.
- **No new SI-82 §4 harness class for v13.** SI-91 §5 opened exactly
  one class — perception assembly and rendering — and that class has
  been exercised (SI-103 to SI-105). Tool catalogue rendering, standing
  instructions, the output-contract instruction, and gateway-request
  assembly were NOT opened by SI-91 and remain unopened for v13. (The
  v2 teaching episode, SI-99 to SI-102, is closed: teaching withdrawn,
  parser amendment retained, no re-opening proposed here.)
- **No training-side work.** The next graded candidate is a
  training-side event in Mark's lane. This repo's loop does not train
  the candidate, name it, or schedule it.
- **D-90-2 and D-90-3 stay deferred.** Trainer wiring (D-90-2) and
  live-NEDB proof for the training journal (D-90-3) are recorded
  bridges in SI-91 §6. This RFC records them again, verbatim in
  substance, and does **not** activate, design, or pre-build either:
  - *D-90-2:* should any SI-83 training-journal component be integrated
    with a real trainer — and if so, which component, which trainer,
    and under what gate?
  - *D-90-3:* should a slice verify the training journal against a
    live NEDB — real writes, chain verification, and read-back over the
    real backend — on the named database `si_training_journal`?
  Until a later recorded decision answers one, the honest labels stand:
  the training journal is file-form tooling at miniature,
  embedded-backend scope, and no live-NEDB training write has been
  performed or claimed.
- **A-2/A-3/A-4 stay UNAPPROVED**, L-3 and C-5 stay LOCKED, C-4 does
  not proceed. None is a next slice.
- **Fork-side work is not SI-repo work.** The fork half of S-6 event
  ingest, and the Hearth-side proposals H-1/H-2/H-3 of SI-41 §12.6,
  belong to the gaia fork and to Mark's merge authority respectively.
  No SI-repo slice is gated on them and none is assigned here.

## 4. The derived remainder: SI-106 §12, in RFC order

Exactly one sequenced Gaia-adjacent remainder can be derived from the
merged RFCs, and it comes from the most recent design in the line:
**SI-106 §12, "Future slices — scope sketch."** SI-106 sequenced, in
order, each as its own slice and review:

1. a multi-cycle capture driver implementing SI-106 §9, with
   replay/fixture evidence only, no live capture;
2. fold-supplied scoring support implementing SI-106 §5.1's procedure
   (frozen fold, `prompt_ref.task_seq` / `parse_task_seq` mapping,
   fold hash in the record), with tests pinning fail-closed behaviour
   on a fold-hash mismatch and on a missing `task_seq`;
3. only then, under SI-106 §9.1's separate authorisation, a live
   capture and its §11 record.

Step (3) has happened: SI-107 is that capture and record, authorised by
the lead's dispatch of 2026-11-12 ~16:45 EST. But steps (1) and (2)
**never landed in this repo as slices.** The SI-107 record's own §9
artefact list places the driver, smoke gate, freeze, and scoring
scripts in a workspace directory (`~/workspace/si107-work/`), not in
`si/gaia/`; on `main` today there is no multi-cycle capture driver
module, no test file for a multi-cycle driver, and no fold-hash
verification in the conformance harness — `score_corpus` in
`si/gaia/conformance.py` accepts a `journal` argument, but no merged
code computes a frozen fold's hash, compares it against a corpus
record, or refuses scoring on a mismatch. SI-107's run verified its
fold hash in its own workspace scripts and recorded the verification
in its evidence; that verification is real, and it is not a repo
capability a reviewer can re-run from `main`.

That gap — exercised once, in workspace scripts, never landed under
review with tests — is the next buildable work. It is proposed here as
two slices, in SI-106 §12's order. Neither slice re-runs, re-grades,
or extends SI-107; both are CPU-only, offline, and miniature/fixture
scoped, exactly as §12 sketched them.

### 4.1 Proposed SI-109 — the multi-cycle capture driver, in-repo (SI-106 §12 item 1)

**Scope.** Land a multi-cycle capture driver in `si/gaia/` implementing
SI-106 §9's disciplines as code: the §4 item shape (cycle 0 unscored,
journaling between cycles **without execution** per §9.5, cycles 1..K
scored), the §9.2 residency-guard interface (poll before every attempt;
a non-Resident poll halts and spends no attempt), the §9.3 budget
accounting (per workload per cycle position, never extended by the
driver), the §9.4 smoke-gate shape, and the §9.6 freeze (corpus and
capture journal frozen together, sha256 of each recorded). The driver
is built and proven against replay transports and fixture bundles
only. **No live capture is part of this slice**, and the driver ships
with no authorisation to make one (§9.1 authorisation remains a
separate, later, recorded act for any future run).

**Acceptance tests (the slice is reviewable against exactly these).**

1. A fixture task run of K = 2 produces exactly the §4 accounting:
   cycle 0 present and unscored; cycles 1..2 entered only with
   non-empty `prior_cycles`; duplicates logged and not entered;
   transport failures recorded unscored and in no denominator.
2. Journaling-without-execution is asserted, not assumed: across the
   fixture corpus, zero queue claims, zero tool executions, and every
   journaled outcome a non-execution disposition — the §9.5-faithful
   proof form SI-107's smoke gate corrected itself to.
3. A residency-guard halt mid-run spends no attempt, persists state,
   and is never restarted around by the driver.
4. Budget exhaustion at a cycle position ends that run under the
   pre-stated budget; the driver contains no path that extends a
   budget.
5. The freeze emits corpus and fold hashes that an independent
   recomputation over the frozen artefacts reproduces exactly.
6. A diff touching gate pricing, parser stage rules, or teaching text
   fails this slice's own review rule (SI-106 §12, last sentence).

**Non-goals.** No live model contact; no new workload family; no
change to SI-106's item definition, denominators, or verdict-free
discipline; no claim that landing the driver retroactively makes
SI-107 a repo-code run — SI-107's record stands as a workspace-script
run, stated as such wherever this slice cites it.

### 4.2 Proposed SI-110 — fold-supplied scoring support, in-repo (SI-106 §12 item 2)

**Scope.** Land SI-106 §5.1's procedure in the conformance harness:
the corpus record carries the frozen fold's sha256 beside the corpus
sha256; scoring reads each scored item's `prompt_ref.task_seq` through
the `parse_task_seq` mapping and supplies the frozen fold as the
scoring journal; both hashes are recorded in the scoring record.
Sequenced after SI-109 only because §12 orders it so; it shares no
code path with the driver beyond the frozen artefacts' shapes.

**Acceptance tests (the slice is reviewable against exactly these).**

1. A fold-hash mismatch between the corpus record and the fold
   supplied at scoring **refuses scoring** — no partial record, no
   rate, no fallback to no-fold scoring.
2. A scored document whose item carries no `prompt_ref.task_seq` is
   recorded as a corpus defect and excluded from M, never silently
   scored task-less (SI-106 §5.1 step 4).
3. A foreign seq — a fold row this capture did not journal — fails
   stage 4 exactly as a nonexistent seq does (SI-106 §13, first risk).
4. Seq-form and sibling-form outcomes are priced and reported
   separately, per SI-106 §6, and a sibling occurrence count of 0 is
   reported as 0, never folded into the seq-form line.
5. No-fold scoring of the existing cycle-0 corpora is byte-for-byte
   unchanged: SI-94, SI-100, and SI-105 are never re-scored under this
   support (SI-106 §5, first paragraph), and the existing conformance
   tests pass unmodified except for this slice's own additions.
6. The scoring record states, in itself, that fold-supplied pricing
   is a different quantity from every no-fold cycle-0 rate (SI-106
   §5.1 step 5) — comparability none, in either direction.

**Non-goals.** No re-scoring of SI-107's frozen corpus as a new
measurement (its workspace scoring stands as its record); no bar, no
verdict language, and no unlock computation for any fold-supplied
corpus — SI-106 §3 binds this support exactly as it bound the design;
no parser stage-rule change.

### 4.3 What §4 earns, stated honestly

SI-109 and SI-110 earn **reproducibility and reviewability, nothing
else**: machinery that was exercised once from workspace scripts
becomes repo code a reviewer can run, with the fail-closed tests §12
always required. They earn no §18.1 row, no §18.3 progress, no rate,
and no claim about any model. If the lead judges that post-hoc landing
is not worth two slices for a diagnostic line that is already closed,
that judgement is a recorded decision under §5 — not a silent skip of
§12, which is the failure mode this RFC exists to prevent.

## 5. Where no unambiguous next Gaia slice can be derived

Beyond §4's remainder, the derivation stops, and this section says so
plainly instead of filling the space:

- **The B/Q/SSE/P/L/seam sequences are exhausted.** Every slice those
  RFCs sequenced is merged (§2). No merged RFC sequences a B-6, Q-5,
  P-6, or a further SSE slice.
- **A new tool is not derivable.** The v1 registry is the SI-41 §8
  surface, complete in B-1's design. SI-41 §15.6 reserves any future
  tool class beyond §8's four classes, and any change to the shipped
  policy defaults, to Mark. A new tool name inside an existing class
  is likewise named by no merged RFC; inventing one here would be a
  capability fabricated to keep building, which §6 rejects.
- **Exposure beyond the B-5 adapter is not derivable.** The service is
  localhost-shaped until an auth story is designed on evidence
  (SI-41 §19); exposure through the Hearth gateway is proposal H-2,
  Mark's to build (§3). No SI-repo slice sequences it.
- **Real-base agency is not derivable.** It is blocked on A-2/A-3/A-4,
  which are UNAPPROVED (§3).

**The smallest decision the lead must record — D-108-1.** One question,
posed so a one-line answer closes it:

> *Does the Gaia line continue with §4's remainder (SI-109, then
> SI-110, in that order), or is the Gaia line recorded COMPLETE at
> SI-107 — with SI-106 §12 items 1–2 recorded as exercised-in-workspace
> and deliberately not landed — until a decision outside this RFC
> (a training-side candidate from Mark's lane, a Mark-approved new
> tool class, or a decided D-90-2 / D-90-3) opens a new line?*

Either answer is a complete outcome. Answering "SI-109 first" makes
§4.1 the next slice and §4.2 its successor. Answering "complete at
SI-107" closes the line honestly, with §12's status stated in the
open. What is not an outcome is silence: SI-91 §1 records what a week
of unrecorded deferral cost this programme, and this RFC does not
repeat it by implication.

## 6. Rejected alternatives

- **6.1 Propose a new graded v13 run under a new harness idea.**
  Rejected (§1.3, §3): the harness investigation is complete, no new
  harness classes for v13, and a changed configuration measured again
  would still be graded against a bar no harness lever has approached.
- **6.2 Invent a new tool, tool class, or service surface to keep the
  Gaia line moving.** Rejected (§5): no merged RFC names one, §15.6
  reserves new classes to Mark, and a roadmap invented to avoid
  stopping is the backlog SI-90 §5.1 forbids.
- **6.3 Activate D-90-2 or D-90-3 inside this RFC.** Rejected: SI-91
  §6 deferred both openly, with their specifics unnamed; a design-only
  RFC that quietly wired a trainer or claimed a live-NEDB proof would
  be exactly the unrecorded activation §3 forbids. They are recorded,
  not decided.
- **6.4 Present SI-107's workspace scripts as merged repo slices.**
  Rejected (§4): the record's own artefact list places them outside
  the repo. The honest statement is the gap, and §4 is its remedy or
  §5's recorded decision is its closure — never a relabelling.
- **6.5 Re-score SI-107, SI-105, SI-100, or SI-94 under any new
  support built by §4.** Rejected (SI-106 §5, §10): recorded corpora
  are inputs to offline analysis, never a new rate, and fold-supplied
  numbers are comparable with no cycle-0 number in either direction.
- **6.6 Move, round, or re-base D-C1.** Rejected: D-C1 stands (§0);
  a threshold moved after the measurements it judges is not a
  threshold, and no verdict in §1 changes under any future decision
  Mark might make — such a decision would be prospective only.

## 7. Non-goals

- Design only: no code, tests, configuration, capture, measurement, or
  live call in this RFC, and no new number of any kind is published
  here — §§1–2 report only what merged slices already reported, cited
  to them, plus the code-state observations of §4, each verifiable on
  `main` at the fingerprint commit.
- No candidate is named, no graded run is proposed, and no training
  work is assigned, designed, or claimed (§1.3, §3).
- No decision in §5 (D-108-1) is answered, recommended, or presented
  as expected by this RFC; §4 states the slices the "continue" answer
  would open, in the form a builder's brief would need, and nothing
  more.
- D-90-2 and D-90-3 are not decided, activated, or pre-designed (§3).
  A-2/A-3/A-4 remain UNAPPROVED; L-3 and C-5 remain LOCKED.
- No change is proposed to §8 or §9 of the proposer contract, any
  §15.2 policy key, any gate, the registry, the queue, the SSE spine,
  the parser, or any training-journal module.
- No fork-repo or Hearth-side work is assigned (§3).

## 8. Honesty rules binding any successor slice

These bind SI-109, SI-110, and any slice built under a §5 decision or a
successor RFC, exactly as SI-90 §7 bound its successors:

- **Offline and miniature work is never live evidence.** Fixture
  drivers, replay transports, and embedded-backend tests earn no
  §18.1 row, no §18.3 progress, no L-3 unlock, and no live
  authorisation.
- **Scope labels travel with the claim.** A result obtained on
  miniature bundles, fixture corpora, or workspace scripts is cited
  with that scope attached, in the citing doc's own honesty table.
- **No threshold weakening or re-cutting.** D-C1 is not re-proposed,
  rounded, subset, or re-based to change any recorded verdict.
- **No re-scoring old bodies as a new measurement.** Any future
  measurement uses a fresh corpus captured for it; a changed
  configuration is a changed candidate, and prior measurements
  describe the prior configuration only.
- **Verdict-free stays verdict-free.** No fold-supplied or multi-cycle
  corpus acquires a bar, a verdict, or an unlock by proximity to the
  graded records (SI-106 §3).
- **Never fake admission, probes, or inference; never invent
  metadata.** Unknown fields stay `unknown` / `None`; a gap is a
  journaled row, never silence; an environmental failure is a
  reportable outcome, never worked around with replay.
- **Never duplicate bytes.** Corpus, fold, checkpoint, and model
  records reference bodies by hash and path, stored once; no successor
  slice copies model, GGUF, tensor, or corpus bytes to make a claim
  easier to state. This RFC copies no corpus body (§1.2 states counts
  and hashes only, per SI-106 §11).
- **Terminology and means:** machine learning throughout; Imagine,
  wherever mentioned, is 1.6B parameters, full fine tune. CPU only;
  no GPU, no cloud, no money, no secrets.

## 9. Sources and code references

All on `main` at `98846f6fe1bdb0b5a6f3dac67dfe41a2a5110f22` unless a
workspace path is given. Where an earlier doc and this RFC disagree,
the earlier doc's evidence wins (see header).

- `docs/SI-90-next-frontier-rfc.md` — the prior next-frontier RFC:
  §2 for the Gaia inventory's first form, §4 for D-90-1 to D-90-5,
  §§5–7 for the rejected alternatives and honesty rules §6 and §8
  inherit.
- `docs/SI-91-decision-record-and-v13-conformance-plan.md` — §3 (D-C1
  stands), §4 (v13 named), §5 (perception class only), §6 (D-90-2 /
  D-90-3 deferred, restated in §3), §7 (the procedure SI-94, SI-100,
  and SI-105 ran).
- `docs/SI-94-c2-conformance-record.md` — 20/30, BELOW THRESHOLD.
- `docs/SI-100-v13-v2-conformance-record.md` — 9/30, BELOW THRESHOLD;
  SI-102 (contract revert) for the withdrawn v2 teaching.
- `docs/SI-105-v13-a1-conformance-record.md` — 18/30, BELOW THRESHOLD;
  §6 for the honest comparison across the three graded records.
- `docs/SI-106-c2-multicycle-diagnostic-design.md` — §3 (verdict-free),
  §4 (item and denominators), §5 (fold-supplied scoring), §6 (both
  reference forms), §9.7 (INCOMPLETE-DIAGNOSTIC), §10 (comparability
  none), §12 (the future-slice sketch §4 derives from).
- `docs/SI-107-c2-multicycle-diagnostic-record.md` — §1 (diagnostic
  statement), §2 (INCOMPLETE-DIAGNOSTIC characterisation), §5 (freeze
  and fold: 371 rows, sha256 `10756d34…`), §6 (M = 31, 17 parse-ok,
  histogram), §7 (seq-form 4/0, sibling-form 0), §9 (artefacts in
  `~/workspace/si107-work/` — the workspace placement §4 relies on).
- `docs/SI-41-agentic-powers-rfc.md` — §8 (the v1 tool surface),
  §12.5–§12.6 (exposure use cases; H-1/H-2/H-3 as Mark's), §15
  (B-1 to B-5 sequence), §15.6 (new tool classes and policy defaults
  remain Mark's), §19 (localhost-shaped service until an auth story
  exists).
- `docs/SI-42-gaia-b1-tool-intent-shadow.md` to
  `docs/SI-46-gaia-b5-service-adapter.md` — the B-line;
  `si/gaia/registry.py` (the §8 surface plus the SI-55 `hearth.*`
  names, no local executor for the latter), `si/gaia/executor.py`,
  `si/gaia/task.py`, `si/gaia/hearth.py`, `si/gaia/service.py`.
- `docs/SI-47-gaia-sse-event-spine.md`,
  `docs/SI-48-gaia-sse-filtered-subscriptions.md`,
  `docs/SI-49-gaia-sse-resilient-follower.md`,
  `docs/SI-58-sse-live-tail-fix.md` — the SSE spine;
  `si/gaia/events.py`, `si/gaia/events_client.py`.
- `docs/SI-50-queue-drain-spec.md` to
  `docs/SI-54-q4-measurement.md` — the Q-line; `si/gaia/queue.py`,
  `si/gaia/drain.py`, `si/gaia/measure.py`.
- `docs/S-2a-proposal-ingest.md` to `docs/S-7-seam-measurements.md` —
  the seam S-line; `si/gaia/seam.py`, `si/gaia/seam_submit.py`,
  `si/gaia/seam_leash.py`, `si/gaia/seam_outcome.py`,
  `si/gaia/seam_reconcile.py`, `si/gaia/seam_events.py`,
  `si/gaia/seam_measure.py`.
- `docs/SI-63-gaia-propose-hearth-proposer-rfc.md` (§19, the P-line
  order) and `docs/SI-64-p1-perception-catalogue.md` to
  `docs/SI-68-p5-measurement-hardening.md` — the P-line;
  `si/gaia/proposer.py`, `si/gaia/live.py`,
  `si/gaia/propose_measure.py`.
- `docs/SI-69-next-frontier-rfc.md`, `docs/SI-76-next-frontier-rfc.md`,
  `docs/SI-80-next-frontier-rfc.md` — the L-line and C-line status §2.6
  carries; `docs/SI-79-c2-conformance-record.md` (v12, 10/30, BELOW
  THRESHOLD); `docs/SI-92-escalation-resolution.md` and
  `si/gaia/resolution.py` — the merged escalation-resolution path.
- `si/gaia/conformance.py` — `score_corpus(corpus, *, journal=None)`:
  the fold argument exists; the fold-hash verification, fold
  provenance, and `prompt_ref.task_seq` record fields of SI-106 §5.1
  do not — the code-state observation at the heart of §4, verifiable
  by reading the module at the fingerprint commit.
- Loop STATE.md (outside this repo, `~/workspace/si-dev-loop/STATE.md`)
  — lead frontier synthesis, 2026-11-12 ~16:30 EST (F-line complete;
  C2 the last open measurement design; no new harness classes for
  v13), and lead SI-107 merge note, 2026-11-13 ~02:55 EST (PR #115
  merged to `98846f6…`; frontier CLOSED; next graded candidate is a
  training-side event; D-90-2/D-90-3 remain deferred bridges).

End of RFC SI-108. The v13 conformance frontier is closed (§1); the
Gaia sequences are complete at their stated scopes (§2); the one
sequenced remainder is SI-106 §12 items 1–2, proposed as SI-109 and
SI-110 (§4); and everything beyond that remainder waits on the single
recorded decision D-108-1 (§5) — or on Mark's lane, which this RFC
does not enter.
