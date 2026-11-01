# SI-69 RFC — The Next Frontier After the PROPOSE Line (Design Only)

> **Status: DESIGN ONLY.** This document derives the next frontier from merged evidence on
> `main` at `b0268e900612cb75df6ca6a8b2af3f4f67007392` (74 PRs merged; SI-68 / PR #74 was the
> last merged slice). It changes no code, no test, no policy default, and no gate behaviour.
> Nothing proposed here is implemented unless a section explicitly says it exists today, in
> code, on `main`.
> **Scope discipline:** CPU only. No GPU, no cloud, no spend, no downloads. Terminology: this
> is machine learning infrastructure throughout.
> **Relationship to prior work:** RFC SI-63 defined the PROPOSE line in its §19 as exactly
> five slices, P-1 through P-5. Those five slices are built and merged as SI-64 through SI-68.
> SI-63 §19 defines no P-6. This RFC is therefore not a continuation by inertia: it states
> what P-1 to P-5 actually built (§1), what they did not earn (§0 and §2), which seam items
> remain UNBUILT on code evidence (§3), and what line should follow (§4).
> **Honesty labels carried forward verbatim, unweakened:**
> - live model-authored intent NEVER YET OCCURRED
> - Live model-authored tool intent: **NEVER YET OCCURRED**.
> - Real Hearth/gaia path performance: **UNMEASURED**.
> - real Hearth/gaia path UNMEASURED
> - Residency-loss behaviour (§16.6 of SI-63): **UNMEASURED**.
> - residency-loss UNMEASURED
>
> No sentence in this RFC strengthens any of those labels. Where
> this RFC and an earlier slice doc disagree, the earlier doc's evidence wins until a merged
> slice updates it with new evidence, in the project's established style.

## Table of contents

- [0. Honesty table: state after SI-68](#0-honesty-table-state-after-si-68)
- [1. What P-1 to P-5 actually built](#1-what-p-1-to-p-5-actually-built)
- [2. What remains blocked and why](#2-what-remains-blocked-and-why)
- [3. Remaining UNBUILT seam items, with code evidence](#3-remaining-unbuilt-seam-items-with-code-evidence)
- [4. The next line: the L-line](#4-the-next-line-the-l-line)
- [5. Rejected alternatives](#5-rejected-alternatives)
- [6. Non-goals](#6-non-goals)
- [7. Sources and code references](#7-sources-and-code-references)
- [Appendix A. Carry-forward checklist for §18.1 rows](#appendix-a-carry-forward-checklist-for-181-rows)

## 0. Honesty table: state after SI-68

Every later section relies on this table. No row is strengthened relative to SI-63 §0 as
updated by SI-67, or relative to SI-68's §0 statement ("No §0 row changes").

| Element | State after SI-68 | Evidence / scope |
|---|---|---|
| Tool-intent format, registry, gate checks | LIVE | SI-42 / B-1; `si/gate.py`, `si/gaia/registry.py` |
| Executor, task loop, queue-drain, SSE spine | LIVE, miniature-scoped where so labelled | SI-43, SI-44, SI-51 to SI-53, SI-47 to SI-49, SI-58 |
| Perception assembly + catalogue rendering (P-1) | BUILT, pure functions, no gateway call | SI-64; `si/gaia/proposer.py` |
| Proposal output contract + parser (P-2) | BUILT, replay only | SI-65; `si/gaia/proposer.py` |
| Gateway wiring under preconditions (P-3) | BUILT, replay + refusal paths only | SI-66; `si/gaia/proposer.py` |
| First live proposer wiring (P-4) | WIRING BUILT, live run NOT COMPLETED | SI-67; `si/gaia/live.py` |
| Measurement harness + preflight hardening (P-5) | BUILT, exercised on fakes / replay only | SI-68; `si/gaia/propose_measure.py`, `si/gaia/live.py` |
| Real Hearth node contact | PARTIAL, scoped (SI-67) | Discovery, residency poll, served-model discovery succeeded 2026-10-31; node then lost mid-slice |
| Live model-authored tool intent | NEVER YET OCCURRED | No merged evidence claims one |
| Real Hearth/gaia path performance | UNMEASURED | Every §18.1 row carried forward UNMEASURED by SI-68 |
| Proposer cycle latency (live) | UNMEASURED | SI-68 row P5-R2; replay observation is REPLAY ONLY |
| Proposal parse-success rate (live) | UNMEASURED | SI-68 row P5-R3; replay observation attached |
| Gate verdict distribution for model intents | UNMEASURED | SI-68 row P5-R4; replay observation attached |
| Task completion rate (model-proposed) | UNMEASURED | SI-68 row P5-R5; replay observation attached |
| Hallucinated-reference rate (§9.3) | UNMEASURED | SI-68 row P5-R6; replay observation attached |
| Injection-attempt success (to proposal, live) | UNMEASURED | SI-68 row P5-R7; replay observation attached |
| Injection success (to execution, live) | UNMEASURED live | SI-68 row P5-R8; gate-construction argument stands, live verification still owed |
| Residency-loss frequency / recovery | UNMEASURED | SI-68 row P5-R9; not exercised, no eviction induced |
| Journal growth per model-proposed task (live) | UNMEASURED | SI-68 row P5-R10; replay observation attached |
| Model quality / task success rate | UNMEASURED | No benchmark has been run live |
| S-5 outcome return (seam label) | UNBUILT as labelled in seam code | See §3.1 for the precise, evidence-scoped reading |
| Fork half of S-6 event ingest | UNBUILT | See §3.2; fork-side work in `aiassistsecure/gaia` |
| A-2 / A-3 / A-4 (SI-39 line) | UNAPPROVED | Not approved by SI-40, SI-63, SI-68, or this RFC |

Three consequences follow immediately:

1. The PROPOSE line is complete as a *wiring and harness* line. It is not complete as a *live
evidence* line, and this RFC must never be cited as evidence that it is. 2. Any demo before a
live re-run slice lands is still a scripted, miniature, or replay demo and must be labelled as
such. 3. Replay numbers from SI-68 measure the parser and the loop. They never measure the
model or the Hearth path, and no later slice may re-label them as live numbers.

## 1. What P-1 to P-5 actually built

Each subsection cites the merged slice doc and the module that carries the work. Claims are
scoped exactly as the slice docs scope them.

### 1.1 P-1 — Perception assembly and catalogue rendering (SI-64)

- Doc: `docs/SI-64-p1-perception-catalogue.md`.
- Module: `si/gaia/proposer.py` — perception assembly (SI-63 §6) and catalogue rendering (§7)
  as pure, journal-derived functions, with digests and the §6.3 bounds and truncation labels.
- `HearthProposer.propose` in P-1 returned the labelled `not-evaluated` outcome and invoked no
  transport.
- Explicitly not built in P-1: §6.5 gateway-request assembly and §6.6 standing-instructions
  prompt text.

### 1.2 P-2 — Output contract and parser, replay only (SI-65)

- Doc: `docs/SI-65-p2-output-contract-parser.md`.
- Module: `si/gaia/proposer.py` — §8's proposal output contract as code, §9's six-stage
  fail-closed parser, reference verification against the fold (§9.3), signal translation (§8.7,
  §8.8), and §14 attempt / response record journaling against the labelled replay transport.
- Hallucinated, cross-task, and future-seq references fail at stage 4 with the batch wholly
  refused.
- Everything in P-2 is labelled REPLAY ONLY. No live claim was made or implied by the slice.

### 1.3 P-3 — Gateway wiring under preconditions (SI-66)

- Doc: `docs/SI-66-p3-gateway-wiring.md`.
- Module: `si/gaia/proposer.py` — the §10 call path and the §11.3 precondition check wired into
  `propose`; §6.5 request assembly (`build_gateway_request`) and §6.6 standing instructions
  (`STANDING_INSTRUCTIONS`); §13.2 halts, §13.3 leash-at-call discard, §15.2 policy keys with
  `proposer_enabled` defaulting off, and the §16.2 to §16.4 failure behaviours.
- Exercised only against replay transports and refusal fixtures. Every precondition failure
  produces zero transport calls and a journaled `not-evaluated` record.
- No real node was contacted by P-3.

### 1.4 P-4 — First live proposer: wiring built, run not completed (SI-67)

- Doc: `docs/SI-67-p4-first-live-proposer.md`.
- Module: `si/gaia/live.py` — runtime discovery of the ephemeral inference base (env override,
  gaia log, or NODE.md; never a guessed port), `GaiaResidencyClient` polling the real residency
  API, served-model discovery with exact §11.1 identity, the fixture bundle builder, the
  journaled `operator_act` enable act, the fixture policy (allowlist `fs_list` / `fs_read` /
  `fs_search`, execute class `deny`, external `escalate`), the suite orchestrator, and
  journal-folded statistics.
- Verified live before the node was lost (2026-10-31): residency `smollm2: resident`; inference
  discovery from the gaia log; served-model discovery returning the SmolLM2-360M-Instruct GGUF
  identity named in SI-67.
- Not completed: the fixture suite did not run. The gaia node was lost mid-slice during another
  lane's rebuild, and relaunch from the builder's sandbox failed with `Operation not permitted`
  on model start (SI-67 §"What then happened"). The one live attempt failed at served-model
  discovery with connection refused.
- P-4's required numbers all stayed UNMEASURED: gateway attempts N=0, no parse rate, no verdict
  distribution, no completion number. The §18.3 claim was NOT earned.

### 1.5 P-5 — Measurement harness and hardening, no live measurement (SI-68)

- Doc: `docs/SI-68-p5-measurement-hardening.md`; record: `docs/SI-68-measurements.json`.
- Module: `si/gaia/propose_measure.py` — S-7-pattern harness: environment fingerprint, named
  workloads (`benign-read`, `injection-shaped`, `hallucinated-reference`), N and spread folded
  from the journal, replay mode by default and a live mode gated on `SI_P5_LIVE=1` that runs
  preflight first.
- Replay exercise (REPLAY ONLY, SI-68's own run): 5 gateway calls, 5 parses, parse rate 0.8,
  stage-4 reference-error rate 0.2, verdicts ADMIT 3 / REJECT 1, tasks completed 2 of 3,
  executions of injected intents 0, journal growth per task median 9 entries.
- Hardening in `si/gaia/live.py`: `preflight_live(...)` with stage-labelled `DiscoveryError`,
  and `journal_environmental_failure(...)` journaling a `gaia_environmental_failure` experience
  with `measurement_taken: False` and `tasks_run: 0`, so an environmental failure is a walkable
  journal record.
- Hardening, second part: `live.reanchor_bundle(...)` fixes the manifest re-anchor gap P-4's
  code path had — out-of-loop journaling now re-anchors before `run_task`.
- Live mode in SI-68: `status: not_attempted`, N=0 live gateway attempts. All ten §18.1 rows
  (P5-R1 to P5-R10) carried forward UNMEASURED. No tuning change landed: the record's
  `tuning_changes` is an explicit empty list.

### 1.6 What the P-line did not build

- No live fixture task has ever completed end to end through a model proposal. That is the
  central gap.
- No §18.1 row has been replaced by a measured live statement. Every replacement still requires
  live data.
- No tuning of perception size or attempt budgets has landed, because SI-63 §19 P-5 permits
  tuning only from live evidence, and there is none.
- No residency-loss behaviour has been exercised: no eviction has been induced on a live node
  during a proposer task, so §16.6 recovery is undesigned-by-data, however well it is designed
  on paper.

## 2. What remains blocked and why

### 2.1 The blocker, stated plainly

The P-4 / P-5 live re-run requires a running gaia node serving a real base as `Resident` under
a known identity. That node does not currently run on this VM: it was lost during SI-67 and,
per SI-67 and SI-68, relaunch is main-chat / fork-lane work following the NODE.md recipe from a
shell where gaia can start its supervised child — not dev-loop work. The loop's sandbox is
precisely the environment in which the relaunch already failed with `Operation not permitted`
(SI-67). The loop must not retry that relaunch, simulate it, or work around it with a better
replay (SI-63 §19.1).

### 2.2 The one decision for Mark

This RFC presents exactly one decision, as a yes/no:

> **Decision D-L0:** Relaunch the gaia node on the VM (NODE.md recipe, main-chat / fork lane),
> confirm `smollm2: resident`, and authorise the L-2 live re-run of the P-4 fixture suite
> followed by the P-5 live measurement — yes or no?

- **Yes** unblocks L-2 and, after it, L-3 and L-4 (§4). The loop then runs the already-built
  commands (`SI_P4_LIVE=1 python -m si.gaia.live`, then `SI_P5_LIVE=1 python -m
  si.gaia.propose_measure`), whose preflight fails closed and journals any environmental
  failure instead of fabricating a run.
- **No** (or no answer) parks the live line without penalty: L-1 and L-5 (§4) are SI-side and
  CPU-only, require no node, and proceed in the meantime. The honesty labels in §0 stay exactly
  as they are.
- There is no third option in this RFC, and no menu of alternative models, hosts, or serving
  stacks. Picking a different served model is an operator act under SI-63 §11, not a decision
  this RFC pre-empts.

### 2.3 What a successful re-run would — and would not — earn

- It would earn, at most, the SI-63 §18.3 claim, scoped to the named served model, gateway, and
  date, with k of n fixture tasks completed end to end and the full §14 attempt chain walkable
  in the journal.
- It would replace only those §18.1 rows the journaled live data actually measures, row by row,
  with N.
- It would not earn general agency, autonomy in the wild, model-quality claims beyond the
  fixtures, or any comparison to other agent systems (SI-63 §18.3).
- A failed re-run is also a reportable outcome: preflight or environmental failure is journaled
  (SI-68), the rows stay UNMEASURED, and the loop reports the failure rather than repairing the
  labels retroactively.

## 3. Remaining UNBUILT seam items, with code evidence

This section inventories only items with direct code or merged-doc evidence on `main`. No other
item is claimed UNBUILT here; absence from this section means "not evidenced as UNBUILT", not
"proven complete".

### 3.1 S-5 outcome return — the UNBUILT label in seam code

- Evidence: `si/gate.py` defines `TOOL_SEAM_PENDING_EXECUTION` ending "and S-5 outcome return
  is UNBUILT"; `si/gaia/seam.py`'s `REGISTRY_GAP_LABEL` states "S-5 outcome return UNBUILT";
  `si/gaia/seam_submit.py` carries `SUBMITTED_STATE = "submitted/awaiting outcome (S-5
  unbuilt)"` and a `WORK_SUBMISSION_LABEL` repeating the label; `si/gaia/seam_leash.py` states
  "S-5 outcome return remains UNBUILT".
- Scoped reading (do not over-claim in either direction): the SI-side outcome poller and
  reconciler shipped in the S-5 slice (`docs/S-5-outcome-return.md`,
  `si/gaia/seam_outcome.py`), tested against labelled replay / stub transports and a loopback
  double. What remains UNBUILT / UNMEASURED, per that doc's own §0, is live fork interop: no
  live fork endpoint was exercised, outcome latency is UNMEASURED except the S-7 scoped
  conditional form, and no outcome returns to the fork — submission-side state still rests at
  "submitted / awaiting outcome" under the labels above.
- Consequence: a seam-crossing ADMIT's full round trip — submit, execute on the fork, outcome
  back, fold closed — has never been observed live. L-5 (§4) addresses the label
  reconciliation; live closure depends on a live fork endpoint, which is fork-lane work, not
  loop work.

### 3.2 Fork half of S-6 event ingest — UNBUILT

- Evidence: `docs/S-6-si-events-forwarder.md` states the fork ingest half (`POST
  /gaia/v1/events/ingest`) is "UNBUILT in this repo / slice — it is fork-side work in
  `aiassistsecure/gaia`"; `si/gaia/seam_measure.py` lists deferred quantities F-E-INGEST and
  F-E-SSE with reason "fork half UNBUILT (design §8.5)"; the S-7 record
  (`docs/S-7-seam-measurements.md`) carries fork ingest service time and ingest-to-SSE
  visibility as UNMEASURED for the same reason.
- Scoped reading: the SI-side forwarder is built and measured in Class S / Class L (SI-62). The
  receiving half — the fork endpoint that accepts forwarded SI journal events into the fork's
  `source=si` SSE buffer — does not exist in either repo's merged evidence.
- Consequence: forwarded-to-visible timing on the fork SSE stream is UNMEASURED and cannot be
  measured from the SI repo alone. Building the fork half is work in `aiassistsecure/gaia`
  under the fork lane's direction, not a slice this RFC can assign to the SI loop.

### 3.3 Items deliberately not listed

- The G-9 "UNBUILT after S-6" cumulative note in `docs/S-6-si-events-forwarder.md` (escalation
  and related items) is a historical slice-time statement. Later slices (S-7 / SI-62 and the
  PROPOSE line) changed parts of that picture; this RFC does not re-derive a full seam census,
  and inventing one without fresh code evidence would violate its own honesty rules.
- A-2 / A-3 / A-4 are not "UNBUILT items" in this section's sense: they are UNAPPROVED (§0,
  §6). An unapproved design step is not a backlog item, and this RFC does not schedule,
  recommend, or pre-design them.

## 4. The next line: the L-line

### 4.1 Naming and justification

The next line is called the **L-line**, for *live-earned*: every slice in it either earns a
live claim from journaled evidence, prepares a specific live claim so it can be earned without
improvisation, or reconciles SI-side labels that a live claim would otherwise collide with. The
P-line built the proposer; the L-line earns (or honestly fails to earn) what the proposer can
claim. The name is deliberately not "P-6+": SI-63 §19 closed the P-line at P-5, and reusing its
lettering would imply an authority SI-63 never granted.

### 4.2 Line order and the no-stall rule

- Order: L-1, then L-2, L-3, L-4 as the live decision (D-L0) allows, with L-5 available at any
  point after L-1.
- No-stall rule: L-1 is first precisely because it is SI-side, CPU-only, and requires no live
  node. If D-L0 is parked, the loop does L-1, then L-5, then stops and reports — it does not
  invent replacement live work.
- Every slice is one PR, built by a builder, reviewed and merged by the dev-loop driver on a
  later run. No slice merges itself. Every slice's doc updates §0's honesty table row by row,
  with evidence, or states explicitly that no row changes.
- Acceptance criteria below are phrased as journaled or system outcomes — records in a bundle's
  journal, files on disk, exit codes, folded statistics — never as conversation quality or demo
  impressions.

### 4.3 L-1 — Replay evaluation corpus v2 (SI-side, CPU-only, no node)

- **Scope.** Extend `si/gaia/propose_measure.py`'s replay workloads with a second fixture
  family: multi-cycle tasks, a valid `depends_on` batch chain, a perception- truncation case at
  the §6.3 bounds, and a repeat of the SI-68 workloads for comparability. All against the
  labelled `ReplayTransport`; all statistics folded from the journal; a new machine-readable
  record beside `docs/SI-68-measurements.json`'s pattern.
- **Acceptance (journaled / system outcomes).** - The record exists on disk, names its
  workloads, N, environment fingerprint, and SI commit, and labels every number REPLAY ONLY. -
  Every admitted replay batch's §14 attempt → response → intent chain is walkable in its
  bundle's journal. - Executions of injected intents in the journal: 0. - The record's live
  section states N=0 live attempts and every §18.1 row UNMEASURED — unchanged from SI-68. -
  Full test suite green on the reviewer's own run.
- **Honesty gate.** Any live label in the L-1 record or doc fails review. L-1 earns no live
  claim, by design.

### 4.4 L-2 — Live re-run of P-4, then P-5 live measurement (conditional on D-L0 = yes)

- **Scope.** No new code by default. Run the SI-67 suite command, then the SI-68 live-mode
  harness command, on the relaunched node; publish the slice doc as the §18.3 claim (or its
  honest failure variant, as SI-67 did) plus the live measurement record.
- **Acceptance (journaled / system outcomes).** - Preflight passes and is journaled, or the run
  stops: the `gaia_environmental_failure` record is journaled, live N stays 0, and the slice
  doc reports the failure. - On success: at least one fixture task completes end to end with
  its §14 chain walkable from the journal; parse rate, stage-4 rate, verdict distribution, and
  stop causes are reported from the journal, with N. - Injection fixtures: journaled executions
  of injected intents = 0; proposal-level behaviour reported. - §18.1 rows are promoted only
  where live journaled data measures them; all other rows stay UNMEASURED with the reason
  stated.
- **Honesty gate.** Numbers come only from the journal fold. A number sourced from logs,
  memory, or the model's own report fails review.

### 4.5 L-3 — Residency-loss drill (conditional on L-2 success)

- **Scope.** During a live fixture task, the operator induces one eviction of the served model;
  the loop's §16.6 behaviour (precondition refresh allowance D-8, then halt) is observed and
  journaled; recovery or non-recovery is reported as observed.
- **Acceptance (journaled / system outcomes).** - The journal contains the residency
  transition, the refresh attempt (at most one, per D-8), and the resulting halt or resumed
  cycle, in order. - No intent is admitted against a non-`Resident` model at any point in the
  journal. - The residency-loss row leaves UNMEASURED only if the drill actually ran; otherwise
  it stays UNMEASURED with the reason stated. No induced eviction, no claim.
- **Honesty gate.** A drill simulated by replay or by a fake transport does not satisfy this
  slice.

### 4.6 L-4 — Tuning from live evidence only (conditional on L-2 data)

- **Scope.** Perception-size and attempt-budget tuning of the proposer policy keys (§15.2),
  only where L-2's (and L-3's, if run) journaled live data motivates a change.
- **Acceptance (journaled / system outcomes).** - Every tuning change cites its before / after
  journal evidence (record ids or seq ranges) in the slice doc. - A change with no live
  evidence behind it does not land; an empty tuning list, stated explicitly as SI-68 did, is a
  valid and complete L-4 outcome. - Re-running the L-2 workloads after tuning produces a new
  journal-folded record; no prior record is edited.
- **Honesty gate.** Replay evidence alone never justifies a tuning change in this slice (SI-63
  §19 P-5 rule).

### 4.7 L-5 — Seam label reconciliation and fork-ingest contract (SI-side, CPU-only)

- **Scope.** Two SI-side parts. (a) Reconcile the stale S-5 wording identified in §3.1: new
  label constants or doc errata that state the S-5 SI-side truth (poller and reconciler built,
  live fork interop UNMEASURED, no outcome returns to the fork) without rewriting any
  already-journaled record or the tests that pin historical strings. (b) Write the SI-side
  contract spec for the fork ingest half of S-6 (request / response shape, 200/409 dedup
  contract already exercised against the S-7 double, `source=si` visibility requirement) as a
  design doc the fork lane can build against.
- **Acceptance (journaled / system outcomes).** - No already-journaled string changes meaning:
  existing tests pinning historical labels pass unmodified, or a test change is justified in
  the slice doc as pinning a *new* constant, never editing history's meaning. - The contract
  spec names the deferred S-7 quantities (F-E-INGEST, F-E-SSE) and states they stay UNMEASURED
  until a live fork endpoint exists and is exercised. - The slice doc states plainly: the fork
  half remains UNBUILT after L-5; L-5 builds no fork code and claims no fork behaviour.
- **Honesty gate.** Any wording that lets a reader believe the fork ingest endpoint exists
  fails review.

## 5. Rejected alternatives

Following SI-63 §21's pattern: each alternative was considered for this frontier and rejected
for cause.

### 5.1 Continue numbering the PROPOSE line (P-6, P-7, …)

Rejected: SI-63 §19 defines P-1 to P-5 only. Extending its numbering would borrow authority the
RFC never granted and blur which claims belong to which design.

### 5.2 Treat SI-68's replay numbers as live evidence

Rejected: SI-63 §18.2 and SI-68 both label replay numbers REPLAY ONLY forever. Re-labelling
them would be exactly the faked measurement this project's honesty rules exist to prevent. The
numbers measure the parser and the loop.

### 5.3 Relaunch the node from the dev loop

Rejected: SI-67 records that relaunch from the loop's sandbox fails on the supervisor / spine
path (`Operation not permitted`), while the recipe works from a proper shell. Node relaunch is
main-chat / fork-lane work (§2). A loop that retries a known-blocked system action is not being
resourceful; it is stalling loudly.

### 5.4 Build a SI-local inference path to bypass the node

Rejected, as in SI-63 §21.3: SI runs as a model under Hearth governance. A local inference
shortcut rebuilds what Hearth provides, strands the design off the serving path, and would
manufacture "live" evidence against the wrong stack — worse than no evidence.

### 5.5 Approve or pre-design A-2 / A-3 / A-4 inside this RFC

Rejected: A-1 was approved and measured (SI-40); A-2, A-3, and A-4 remain separate approvals
that no merged document grants. Folding them into the L-line would smuggle an approval through
a docs-only RFC.

### 5.6 A second model as live-evidence judge

Rejected, as in SI-63 §21.6: completion is judged by the operator against fixture expectations,
and statistics are folded from the journal. A judge model would add a second unmeasured surface
and could not certify the first one.

### 5.7 Skip L-1 and wait for the live decision

Rejected: waiting stalls the loop on a decision that is legitimately Mark's to park. L-1 and
L-5 are real, evidenced, SI-side work that a later live claim needs (a broader replay corpus;
reconciled seam labels). Doing them first is progress, not displacement activity — provided
neither is ever presented as live progress.

## 6. Non-goals

- No code, tests, or configuration in this RFC; it is design only. Implementation belongs to
  L-1 to L-5.
- No live claim of any kind: no live run is reported, scheduled by this document alone, or
  implied.
- No node relaunch, fork-repo work, or Hearth-side change is assigned or performed by this RFC.
- No A-2 / A-3 / A-4 work, and nothing in this RFC reads as approval of them. They remain
  UNAPPROVED.
- No new tool surface, registry change, gate change, queue change, or SSE change is proposed.
- No model selection advice: whichever model the node serves under SI-63 §11 is an operator
  act; the L-line measures the path, not the model's worth.
- No tuning change is proposed here; L-4's evidence rule (§4.6) governs all future tuning.
- No backlog beyond the L-line: slices not listed in §4 require a successor RFC, not an implied
  extension.

## 7. Sources and code references

All references are to `main` at `b0268e900612cb75df6ca6a8b2af3f4f67007392` unless a doc number
is given. Module and doc references are used in place of line-level pointers, which do not
survive refactors.

- `docs/SI-63-gaia-propose-hearth-proposer-rfc.md` — the PROPOSE RFC: §0 honesty table, §18.1
  labels, §18.3 first honest live claim, §19 slices P-1 to P-5, §21 rejected alternatives, §22
  non-goals.
- `docs/SI-64-p1-perception-catalogue.md` — P-1 as-built.
- `docs/SI-65-p2-output-contract-parser.md` — P-2 as-built.
- `docs/SI-66-p3-gateway-wiring.md` — P-3 as-built.
- `docs/SI-67-p4-first-live-proposer.md` — P-4 as-built: live discovery evidence, node loss,
  failed relaunch, the un-earned §18.3 claim, and the re-run command.
- `docs/SI-68-p5-measurement-hardening.md` and `docs/SI-68-measurements.json` — P-5 as-built:
  harness, preflight, environmental-failure journaling, re-anchor fix, all §18.1 rows
  UNMEASURED, live N=0 / not_attempted.
- `si/gaia/proposer.py` — P-1 to P-3: assembly, rendering, contract, parser, gateway wiring
  under preconditions.
- `si/gaia/live.py` — P-4 live wiring; P-5 preflight, environmental-failure journaling, bundle
  re-anchoring.
- `si/gaia/propose_measure.py` — P-5 measurement harness (replay mode; live mode gated on
  `SI_P5_LIVE=1`).
- `docs/S-5-outcome-return.md` and `si/gaia/seam_outcome.py` — S-5 SI-side as-built scope.
- `si/gate.py`, `si/gaia/seam.py`, `si/gaia/seam_submit.py`, `si/gaia/seam_leash.py` — the S-5
  UNBUILT label strings inventoried in §3.1.
- `docs/S-6-si-events-forwarder.md`, `si/gaia/seam_measure.py` (deferred list F-E-INGEST,
  F-E-SSE), `docs/S-7-seam-measurements.md` — the fork half of S-6 as UNBUILT, and its
  UNMEASURED quantities.
- `docs/SI-40-a1-feasibility-measurement.md` — A-1 measured feasible; A-2 / A-3 / A-4
  unapproved.

## Appendix A. Carry-forward checklist for §18.1 rows

For the next slice that touches any §18.1 row, in the SI-68 pattern: a row is promoted only by
journaled live data naming its workload, N, and environment, recorded in a new machine-readable
record — never by editing a prior record, and never from replay data.

| Row (SI-68 id) | Quantity | State entering the L-line |
|---|---|---|
| P5-R1 | Real Hearth path latency / throughput | UNMEASURED |
| P5-R2 | Proposer cycle latency (gateway + parse) | UNMEASURED |
| P5-R3 | Proposal parse-success rate | UNMEASURED |
| P5-R4 | Gate verdict distribution for model intents | UNMEASURED |
| P5-R5 | Task completion rate (model-proposed) | UNMEASURED |
| P5-R6 | Hallucinated-reference rate (§9.3) | UNMEASURED |
| P5-R7 | Injection-attempt success (to proposal) | UNMEASURED |
| P5-R8 | Injection success (to execution) | UNMEASURED live |
| P5-R9 | Residency-loss frequency / recovery | UNMEASURED |
| P5-R10 | Journal growth per model-proposed task | UNMEASURED |

End of RFC SI-69. The L-line starts at L-1 (§4.3) on the next dev-loop run after this RFC is
reviewed and merged, whether or not decision D-L0 (§2.2) has been answered.
