# SI-50 — The Queue-Drain Executor: The Journal Is the Queue
(Design Only)

> **Status: DESIGN ONLY.** This document specifies a new
> execution shape for Gaia — the agentic layer of SI built by
> the SI-41 line — in which tool intents are drained from a
> queue instead of executed one per loop cycle. It changes no
> production code, no test, no manifest schema, no journal
> schema, no policy default, and no gate behaviour. Nothing
> in this spec is implemented unless a section explicitly
> says it exists today, in code, on `main` (787b9bf, PRs
> #1–#53 merged). Every quantity this design would need
> measured — drain throughput, concurrency benefit, queue
> residence times — is **UNMEASURED** unless labelled
> measured with its source. The numbers pinned in §8 and §12
> are *policy defaults chosen by this spec and justified in
> place*; they are starting values for the implementation
> slices to measure against (§16, Q-4), not measurements.
>
> **The operator directive (Mark, 2026-10-30, verbatim):**
> "one tool per call is going to blunt the daemon dont you
> think a queue to drain would be more elegant" — given in
> chat while reviewing the merged Gaia line, and adopted the
> same morning as the next Gaia architecture slice. The
> design points agreed with the directive, and recorded in
> the dev-loop STATE.md the same day, are the skeleton of
> this spec: the journal is the queue; pipelined, not
> serial; batched perception; the operator leash preempts
> the queue; backpressure and budgets remain.
>
> **The one sentence that governs everything below:** a
> gated, journaled tool intent *is* an enqueued intent —
> SI already writes its queue to the NEDB journal as a
> by-product of its governance, so this spec builds a
> drainer for the journal it has, and no second queue, in
> any store, anywhere.
>
> **Scope discipline:** CPU only on this VM for every slice
> proposed here. No GPU, no cloud, no spend, no downloads.
> Terminology: this is machine learning infrastructure
> throughout.
>
> **Relationship to prior work:** this spec refines exactly
> one section of one prior document — SI-41 §10's
> one-tool-per-cycle chaining rule — and inherits everything
> else in SI-41 unchanged (§2, §3). The pieces it composes
> with are built and merged: the tool registry and shadow
> gate (SI-42), the executor and filesystem tools (SI-43),
> the execution tools and task loop (SI-44), the Hearth
> interface (SI-45), the service adapter (SI-46), the SSE
> event spine (SI-47), filtered subscriptions (SI-48), and
> the resilient following consumer (SI-49, in flight as
> PR #54 at the time of writing). §15 treats each in turn.

---

## Table of contents

1. [Where Gaia stands: a serial task loop, stated plainly](#1-where-gaia-stands-a-serial-task-loop-stated-plainly)
2. [The directive, and what "refines §10" means](#2-the-directive-and-what-refines-10-means)
3. [Inherited invariants, unchanged](#3-inherited-invariants-unchanged)
4. [The journal is the queue](#4-the-journal-is-the-queue)
5. [Intent lifecycle: states as journal events](#5-intent-lifecycle-states-as-journal-events)
6. [Batch proposal: how intents enter the queue](#6-batch-proposal-how-intents-enter-the-queue)
7. [Drain semantics: ordering keys and dependencies](#7-drain-semantics-ordering-keys-and-dependencies)
8. [Concurrency: bounded reads, serial writes](#8-concurrency-bounded-reads-serial-writes)
9. [Pipelining: admission overlaps execution](#9-pipelining-admission-overlaps-execution)
10. [Batched perception](#10-batched-perception)
11. [The operator leash against a queue](#11-the-operator-leash-against-a-queue)
12. [Backpressure and budgets](#12-backpressure-and-budgets)
13. [Crash, restart, and the partially drained queue](#13-crash-restart-and-the-partially-drained-queue)
14. [Idempotency honesty](#14-idempotency-honesty)
15. [Interaction with the existing pieces](#15-interaction-with-the-existing-pieces)
16. [Phased implementation plan: Q-1 to Q-4](#16-phased-implementation-plan-q-1-to-q-4)
17. [Acceptance criteria for Q-1](#17-acceptance-criteria-for-q-1)
18. [Risks and failure modes](#18-risks-and-failure-modes)
19. [Rejected alternatives](#19-rejected-alternatives)
20. [Open questions](#20-open-questions)
21. [Non-goals](#21-non-goals)
22. [Sources](#22-sources)

---

## 1. Where Gaia stands: a serial task loop, stated plainly

What exists on `main` today (787b9bf, PRs #1–#53, suite
602 passed / 2 skipped on `main`; the SI-49 branch carries
it to 621):

- **Tool intents in the self-write loop** (SI-41, SI-42):
  a third intent kind, `tool`, evaluated by the existing
  gate against the 12-tool registry (`fs_list`, `fs_read`,
  `fs_search`, `fs_write`, `fs_edit`, `fs_mkdir`,
  `journal_query`, `bundle_status`, `probe_run`,
  `shell_exec`, `code_exec`, `test_run`), journaled in NEDB
  before anything runs, with the journaled-args rule
  (SI-41 §9.2): the executor executes the journaled intent,
  re-read from the journal and re-validated, never a
  mutable in-memory copy.
- **A live executor** (SI-43, SI-44): on miniature bundles
  the full registry executes for real, inside the work
  root, under the executor's bounds (per-intent timeout
  `tool_timeout_seconds`, default 120s; per-stream capture
  caps; environment scrubbing). Intents whose effective
  class prices to `external` still ESCALATE and never
  execute.
- **A task loop** (SI-44, `si/gaia/task.py`): an operator
  opens a task with a journaled instruction; the loop then
  runs **one tool per cycle** — PERCEIVE → PROPOSE (one
  intent) → GATE → JOURNAL → EXECUTE → VERIFY → PERCEIVE —
  until a §10.4 stop rule fires. Budgets in force today
  (`si/gaia/policy.py`, mirroring SI-41 §6.2):
  `max_tool_cycles_per_task` 16, `tool_max_result_bytes`
  64 KiB inline per result, `tool_max_task_result_bytes`
  1 MiB per task.
- **Exposure** (SI-46, SI-47, SI-48): the service adapter
  returns work results with their gate-journaled provenance
  trail; the SSE spine streams journal-derived events;
  filtered subscriptions and a reference consumer exist,
  with the resilient follower (reconnect + Last-Event-ID
  resume) in flight as SI-49 / PR #54.

The serial shape was deliberate. SI-41 §10.2 chose
one-tool-per-cycle so every step is "individually gated,
journaled, priced, and attributable," mirroring the delta
stack's serial discipline while the gate's tool mapping
was new and unproven. It was, in the word used when the
directive was recorded, a **correctness scaffold**: the
simplest execution order under which gate-before-execution
and journal-before-execution are trivially true, because
nothing is ever in flight while anything else is decided.

The cost of the scaffold is also plain, and it is the
operator's critique: a daemon that pays a full loop cycle
— a perception, a proposal, a gate evaluation, a journal
round-trip, an execution, a verification — **per tool
call** spends its latency budget on ceremony. A task that
needs thirty tool calls takes thirty sequential laps, each
waiting for the previous lap's execution before the next
intent is even *priced*, although pricing intent N+1 does
not depend on the result of executing intent N unless the
mind wrote it that way. For an acting layer whose whole
purpose is sustained work on computer systems, that is
the bluntness the directive names. This spec removes the
scaffold and keeps the governance.

## 2. The directive, and what "refines §10" means

Mark's ruling, applied:

1. **A queue to drain is the execution shape.** Tool
   intents are admitted into a queue and an executor
   *drains* the queue. The daemon's steady state is a
   drain in progress, not a sequence of isolated cycles.
2. **The queue is the journal** (§4). No queue service, no
   queue table, no in-memory list that matters. This is
   the project's standing design rule — never duplicate
   bytes, never duplicate truth — applied to a work queue:
   the journal already records every admitted intent, in
   order, durably, with provenance. A second store would
   have to be kept consistent with the first, and every
   inconsistency between them would be a governance bug.
3. **The invariants do not move** (§3). Gate before
   execution. Journal before execution. The executor runs
   exactly the journaled args. The queue changes *when*
   admitted intents run relative to each other and to
   later admissions; it changes nothing about *whether* an
   intent may run.

"Refines §10" is therefore a precise claim, and this spec
makes no larger one:

- **Superseded:** SI-41 §10.2's "one tool per cycle" and
  "the chain is linear and serial — no parallel tool
  execution" as the *only* execution order; §10.3's budget
  accounting where it counts cycles (re-derived in §12.3
  of this spec).
- **Unchanged:** §10.1 (tasks are operator-opened; a mind
  cannot open a task for itself), §10.4's stop rules (each
  is re-expressed against a queue in §11 and §12.4, with
  its firing condition intact), §9.1's work-root scope,
  §9.2's journaled-args rule (extended, not replaced, by
  §13's recovery rules), §9.3's result bounds, §11's
  VERIFY and rollback honesty, and the whole of §§4–8
  (intent format, gate mapping, registry, classes).

Any implementation slice that finds it cannot honour the
unchanged list returns `stop` and says so, in the SI-39
§5.3 style this project has used before.

## 3. Inherited invariants, unchanged

These are not re-decided here. They are restated because
the queue puts pressure on exactly these points, and each
later section names the invariant it is spending care on.

From `docs/self-write-loop.md` and SI-41 §3:

- **No ungated execution.** There is no code path from
  "mind wants" to "tool ran" that bypasses the gate. Under
  a queue this becomes: there is no code path from "intent
  exists" to "tool ran" that bypasses the gate — in
  particular, the drainer is not a gate bypass. The drainer
  executes only intents whose journaled verdict is ADMIT
  (§5), and re-checks that verdict itself, exactly as the
  SI-43 executor already does.
- **No silent change.** Every transition is journaled with
  a reason. Under a queue, *queue transitions are
  transitions*: admission, claim, refusal at claim,
  outcome, skip, hold, resume, and disposition are each
  journal events (§5), never mutations of a row nobody can
  audit afterwards.
- **Journal before execution.** An intent's gated record
  exists in the NEDB journal before its tool runs. Under a
  queue this strengthens into **claim before execution**
  (§5.3): the drainer's claim on an intent is also
  journaled before the tool runs, so a crash mid-execution
  leaves a journal that says precisely what was in flight
  (§13).
- **Fail closed.** A malformed intent, an unknown tool, a
  broken policy key, or an unevaluable check never
  produces an ADMIT — and, the queue adds, never produces
  a claim. A drainer that cannot establish an intent's
  state from the journal does not execute it (§13.2).
- **Measured or labelled, never fabricated.** Queue
  behaviour that has not been measured is labelled
  UNMEASURED in this spec and must stay labelled in every
  slice doc until Q-4 (§16) measures it.
- **Never duplicate bytes** (Mark's design rule,
  2026-10-17). The queue stores nothing: it is a *reading*
  of the journal (§4). Results and pre-images keep using
  the bundle's content-addressed blob store, as today.

From SI-41 §11.2, restated because recovery (§13) leans
on it hardest:

- **The journal is a record, not an undo for the world.**
  One-way effects cannot be rolled back by any operation
  SI has. The queue's crash semantics are built on this,
  not around it: what recovery may and may not do follows
  from what the world does and does not permit (§13.3,
  §14).

---

## 4. The journal is the queue

### 4.1 The definition

An intent is **enqueued** when its gate verdict ADMIT and
its intent record are journaled — which is to say, at the
exact moment the existing loop's JOURNAL step completes
for it. There is no enqueue operation distinct from
journaling an admitted intent, and no dequeue operation
distinct from journaling the events that follow it (§5).
The **queue for a task** is the derived set: every ADMITted
tool intent chained to the task whose lifecycle (§5) has
not reached a terminal state.

This is not a metaphor and not a storage trick. The NEDB
journal already has every property a governed work queue
needs, and the queue needs every property for governance
rather than for speed:

- **Durable.** Journaled intents survive process death;
  §13's recovery is a journal fold, not a salvage
  operation.
- **Totally ordered.** Journal sequence numbers give the
  admission order — the FIFO backbone of §7 — with no
  clock trust anywhere.
- **Provenance-complete.** Every intent already carries
  its reason, its parents, and its chain to the operator's
  task instruction (SI-41 §5.2, check 5). A queue entry
  that knows *why it exists* is the entire difference
  between this design and a job list.
- **Queryable and streamable.** `journal_query` reads it;
  the SI-47 SSE spine already streams journal-derived
  events, so queue state is observable live, and resumably
  (SI-48 filters, SI-49 Last-Event-ID resume), with no new
  observability built and nothing extra to keep in sync.

### 4.2 What the drainer is

The **drainer** is a new component in `si/gaia/` (working
name `si/gaia/drain.py`; the name is this spec's, the
module layout is Q-1's to settle) with exactly three
responsibilities:

1. **Fold** the journal to derive each open task's queue
   state (§5.4).
2. **Claim** eligible intents — journaling the claim
   first (§5.3) — in the order §7 prescribes and within
   the concurrency §8 permits.
3. **Execute** each claimed intent through the existing
   SI-43/SI-44 executor, unchanged, under the
   journaled-args rule, and journal the outcome through
   the executor's existing outcome records.

The drainer proposes nothing, gates nothing, and decides
nothing about admissibility. It is a reader of verdicts
and a caller of the executor. Any slice that finds the
drainer growing a policy opinion has put the code in the
wrong component (§19).

### 4.3 One drainer per bundle

In v1 there is **one drainer per bundle**, and it is the
only claimer of that bundle's intents. This is an
assumption this spec makes explicitly rather than a fact
it inherits: nothing in the current code runs two task
loops against one bundle, and the claim protocol (§5.3)
is specified so that a second claimer would fail closed
(§13.2), but multi-drainer coordination — leases contested
between live processes, claim fencing tokens — is **not
designed here** and is named as an open question (§20).
Slices must not silently assume it away: a drainer that
starts against a bundle whose journal shows a live claim
by another drainer (§5.3) refuses to drain and journals
the refusal.

### 4.4 The queue is per-task, the drain is per-bundle

Queues are scoped to tasks (budgets, stop rules, and
perception are all task-shaped in SI-41 §10), but a
bundle may hold several open tasks over its life. The
drainer serves the bundle's open tasks **round-robin by
task journal order**: within a drain pass it claims at
most one intent per task before offering the next task
its claim, so one task's deep queue cannot starve
another's single intent. This is the only fairness rule
in v1; priority lanes are a non-goal (§21) and an open
question (§20).

## 5. Intent lifecycle: states as journal events

### 5.1 The principle: state is folded, never stored

An intent's queue state is **derived by folding the
journal entries that name it**, in journal sequence
order — the same way the bundle's head is derived from
its self-edit chain. No entry is ever edited; a state
transition is a new journal entry. The fold is total:
any prefix of the journal yields the state as of that
prefix, which is what makes crash recovery (§13) and
SSE replay (SI-47/49) the same operation.

### 5.2 The states

| State | Meaning | Entered by (journal event) |
|---|---|---|
| `proposed` | Intent journaled at PROPOSE, gate evaluation pending or in progress | intent entry |
| `rejected` | Gate verdict REJECT — terminal; never queued | verdict entry |
| `escalated` | Gate verdict ESCALATE — terminal for the queue; the operator's to admit or refuse per SI-41 §5.4 | verdict entry |
| `queued` | Verdict ADMIT journaled; no claim yet | verdict entry (ADMIT) |
| `claimed` | A drainer has journaled a claim; execution may begin (§5.3) | claim entry |
| `executed` | An outcome entry exists — success **or** failure; both are outcomes (SI-41 §11.1) — terminal | outcome entry |
| `refused-at-claim` | Claim-time re-validation failed (§7.5); never executed — terminal | refusal entry |
| `skipped` | A `depends_on` requirement was unmet (§7.3); never executed — terminal | skip entry |
| `held` | The operator leash stopped the drain before this intent was claimed (§11) | (derived: stop entry with no claim) |
| `outcome-unknown` | Was `claimed` at a crash; no outcome exists; recovery could not establish one (§13.3) — terminal pending operator disposition | recovery entry |
| `compensated` | A later, ordinary compensation intent (SI-41 §11.2) names this intent — an annotation folded onto `executed`, never a rewrite of it | compensation intent's provenance |

Three honesty notes on this table:

- **`executed` does not mean "succeeded."** It means an
  execution happened and its outcome — exit status,
  result, post-state observation — is on the record. The
  outcome entry carries the detail; the state claims only
  that the world was touched and the touch was recorded.
- **`compensated` is not a rollback state.** It records
  that a compensating action was proposed, gated, and
  executed as its own intent. The original effect's
  history is untouched, per SI-41 §11.2.
- **Derived states are labelled derived.** `queued` and
  `held` are folds, not entries; `si why` and
  `journal_query` answers about them must be able to show
  the entries they were folded from. No mystery (§3).

### 5.3 The claim

Before executing an intent, the drainer journals a
**claim entry**: the intent's journal sequence, the
drainer's identity (a per-process identifier recorded in
the entry), the claim's own sequence, and a **claim
lease** — a wall-clock duration (policy
`queue_claim_lease_seconds`, default 300s, §12.2) after
which an outcome-less claim is treated as suspect by
recovery (§13.3). The claim is the queue's analogue of
journal-before-materialize: it makes "what was in flight
when the process died" a journal fact instead of a guess.

The claim is also the executor's ticket. The executor's
existing discipline — re-read the intent from the
journal, re-validate against registry and scope, re-check
the verdict (SI-43) — is extended by one check: the
intent it is handed must be the intent the claim names,
at the sequence the claim names. A claim/intent mismatch
refuses execution and journals the refusal, under the
same integrity rule as SI-41 §9.2.

### 5.4 Folding rules

- Entries fold in journal sequence order; a later entry
  about an intent supersedes the state implied by earlier
  ones, per the table's transitions. An entry that implies
  an impossible transition (an outcome for a never-claimed
  intent, a second terminal entry) is a **journal
  integrity fault**: the fold stops at that intent,
  marks it `outcome-unknown` with the fault recorded, and
  the drainer refuses to drain that task until the
  operator dispositions it (§13.4). Fail closed (§3),
  applied to bookkeeping.
- The fold is over the bundle's journal only. A drainer
  never reconstructs queue state from the filesystem, from
  executor logs, or from its own memory of a previous
  process (§19).

## 6. Batch proposal: how intents enter the queue

Pipelining requires the mind to be able to put more than
one intent in flight. The minimal change to SI-41 §10.2
that achieves it: **a PROPOSE step may return a batch** —
an ordered list of tool intents — instead of exactly one.

### 6.1 Batch mechanics

- A batch is journaled as a **batch entry** (task,
  proposer provenance, declared size) followed by the
  member intents in declared order. Each member is then
  gated and journaled **individually, in order**, exactly
  as a singleton proposal is today. Admission order is
  therefore still serial and total: batching changes who
  waits for execution, never the order in which intents
  are priced.
- **A batch is not atomic.** Partial admission is the
  normal case, not an error: some members ADMIT (and are
  queued), some REJECT, some ESCALATE, each with its own
  verdict entry and reason. No member's verdict is
  influenced by another's except through the shared
  budgets priced in order (§12.3). Any slice that makes a
  batch all-or-nothing has changed the gate's semantics
  and is out of spec (§19).
- A batch of one is legal and behaves exactly as today's
  single proposal. The SI-44 task loop's behaviour is the
  degenerate case of this section, which is what makes
  the migration in §16 incremental rather than a rewrite.
- The proposer declares, per member, the fields §7 needs:
  optional `ordering_key`, optional `depends_on`, optional
  `barrier`. Absent fields take §7's defaults; the gate
  prices the *declared* fields (a dependency on a sequence
  that cannot exist is a malformed intent, §6.2).

### 6.2 Malformed batches

Fail closed, per member where possible: a member whose
`depends_on` names a sequence not yet journaled, a
sequence outside the task, or its own sequence is
malformed and REJECTs on that cause; the rest of the
batch is unaffected. A batch entry whose declared size
does not match its member count is a journal integrity
fault (§5.4) — the batch's members fold as individuals
(their own entries are complete records), and the fault
is recorded against the batch entry.

### 6.3 What batching does not do

A batch is **speculative**: the mind writes later members
without having seen earlier members' results. That is
the point — and the limit. `depends_on` (§7.3) orders
execution; it does **not** pipe one intent's result into
another's args. Args are fixed at proposal, journaled,
and executed as journaled (§3). A step whose args depend
on a previous step's *result* cannot be in the same
batch; it belongs to a later proposal, after the batched
perception (§10) returns. Specs and slices that blur
this — templated args, result references, "the executor
will fill it in" — are rejected by this section in
advance: they would break the journaled-args rule that
the whole governance rests on.

## 7. Drain semantics: ordering keys and dependencies

### 7.1 The drain order, stated as an algorithm

Within a task, an intent is **claimable** when all of
the following hold:

1. Its folded state is `queued`.
2. Every intent in its `depends_on` has reached a state
   satisfying the declared requirement (§7.3).
3. No unexecuted earlier intent shares its ordering key
   (§7.2) — FIFO per key.
4. Its concurrency class (§8) has a free slot.
5. No stop condition is in force for the task (§11,
   §12.4).

The drainer repeatedly claims the **lowest-sequence
claimable intent** across the bundle's open tasks
(round-robin per §4.4 refines this across tasks: lowest
sequence within each task's turn) until nothing is
claimable or a drain budget (§12.2) is exhausted. The
algorithm is deliberately this small: any drain order a
reviewer cannot re-derive from the journal and this
section is a bug, however good its intentions.

### 7.2 Ordering keys

An intent's **ordering key** names the resource whose
order it must respect. Two intents with the same key
execute in journal sequence order, never concurrently.
Defaults, derived at gate time and recorded in the
verdict entry (the priced key is part of the gated
record, not recomputed at drain time):

- `write`-class filesystem tools (`fs_write`, `fs_edit`,
  `fs_mkdir`): `fs:` + the canonical resolved target path
  (resolution per SI-41 §9.1 — symlinks resolved before
  classification, so the key names the real target).
- `execute`-class tools (`shell_exec`, `code_exec`,
  `test_run`): `exec:` + the work root. All execution in a
  task serialises on the work root in v1 — commands can
  touch anything under it, and pretending a finer key is
  knowable would be fabrication (§14).
- House tools (`journal_query`, `bundle_status`,
  `probe_run`): `house` — they read the bundle's own
  state, which the queue itself is changing; serialising
  them keeps their answers fold-consistent.
- `read`-class tools: **no key by default** — the
  parallel pool (§8). A proposer may declare a key for a
  read whose target it knows is being written by a queued
  sibling; the gate records the declared key and §7.1
  rule 3 then applies to it.
- A proposer may declare an explicit `ordering_key` for
  any intent to force a shared order across tools (two
  writes to different files that must not interleave, a
  read that must follow an unrelated write). Declared keys
  are priced as part of the intent and journaled with it.

### 7.3 Dependencies: `depends_on`

`depends_on` is a list of dependency declarations, each
naming an earlier intent **by journal sequence** in the
same task, with a requirement:

- `outcome` (the default): the dependency is satisfied
  when the named intent reaches `executed`,
  `refused-at-claim`, or `skipped` — any recorded outcome.
  The dependent's args were fixed at proposal; it runs
  knowing the dependency *concluded*, not how.
- `success`: satisfied only by `executed` with a
  successful outcome entry. If the named intent concludes
  any other way, the dependent is `skipped` — journaled
  with the cause and the dependency's outcome sequence —
  and never executes.

Cycles are impossible by construction: a dependency must
name a strictly earlier sequence (§6.2), so the dependency
graph is a DAG whose topological order is sequence order.
Any implementation that adds a cycle check has
misunderstood the construction; any that needs one has
broken §6.2.

### 7.4 Head-of-line honesty

FIFO per key means a slow write to a file blocks later
writes to the same file for as long as it takes (bounded
by `tool_timeout_seconds`). That is correctness, not a
defect — the alternative is interleaved writes to one
target — but this spec names it because it bounds what
the queue can promise: **drain latency for a key is the
sum of its members' execution times**, and no slice may
advertise per-intent latency the key structure cannot
deliver (§18).

### 7.5 Claim-time re-validation

The gate priced the world at proposal time; a queued
intent may sit while the world moves. At claim, the
drainer hands the intent to the executor, whose existing
re-validation (SI-41 §9.2: re-read from journal,
re-validate against registry, scope, and the world's
current classification) runs **as it does today** — the
queue widens the proposal-to-execution window, which
makes §9.2 more load-bearing, not less. An intent that
fails re-validation is `refused-at-claim`: journaled
with its cause, never executed under a stale price, and
its dependents see a concluded outcome (§7.3). Residence
time is therefore a *correctness* non-issue and a
*freshness* fact: §12.2's residence bound exists so
stale-priced work is refused by policy rather than
accumulating until re-validation catches it one by one.

---

## 8. Concurrency: bounded reads, serial writes

### 8.1 The rule

- **Read-class intents with no ordering key** may execute
  concurrently, up to a bound (§8.2).
- **Everything else executes serially per ordering key**,
  and keyed execution never overlaps another execution on
  the same key. In v1, write-class and execute-class
  intents are always keyed (§7.2), so the entire
  state-changing surface drains serially per target. The
  queue's concurrency is, by design, confined to the
  class that cannot change the world.

This asymmetry is the spec's central safety choice, and
it is chosen from the consequence classes SI-41 §7
already defined rather than invented here: reads are
`reversible` by construction (they change nothing),
writes are `compensable` at best, executions are
`one-way`. Concurrency is spent only where a mistake is
free.

### 8.2 The bound: 4, and why

Policy `queue_max_concurrent_reads`, **default 4**, hard
cap 16 (a policy value above the cap resolves to the
cap, per the project's most-restrictive-resolution
practice for malformed policy, SI-41 §3).

Justification, from evidence rather than taste:

- The deployment target measured by SI-40 (A-1) is a
  **2 vCPU, 7.7 GiB VM**. Read tools are I/O-bound
  (filesystem reads, searches, journal queries), so a
  small multiple of the core count overlaps latency
  without contending for CPU.
- The journal serialises outcomes anyway: outcome entries
  append in sequence order, and result bytes are bounded
  per intent (§9.3 of SI-41). Past a small bound, extra
  readers mostly queue at the journal and at PERCEIVE's
  result budget — contention, not throughput.
- Four is also the largest bound whose failure surface
  stays reviewable in v1: four concurrent readers, each
  with journaled claim and outcome, is a fold a human can
  audit by eye in `si log`.

This number is **UNMEASURED**. It is a starting policy,
and Q-4 (§16) exists to replace it with evidence —
sweeping 1/2/4/8 on scripted read-heavy miniature tasks
and publishing the drain wall times. A slice that tunes
the default without that measurement is out of spec.

### 8.3 What concurrency does not change

- **Admission order** is untouched: gating and journaling
  remain serial (§6.1, §9).
- **Outcome journaling** is serial: concurrent executions
  finish in any order, but their outcome entries append
  in completion order, each naming its intent and claim —
  the fold (§5.4) does not care, because state is per
  intent.
- **Perception order** is outcome-sequence order (§10),
  never proposal order and never completion-time order
  reconstructed from clocks.

## 9. Pipelining: admission overlaps execution

### 9.1 The pipeline shape

```
proposer ──batch──▶ GATE ──▶ JOURNAL ──▶ QUEUE (folded)
                                              │
                                              ▼
                            DRAINER: claim ──▶ EXECUTOR
                                              │
                                              ▼
                            outcome ──▶ JOURNAL ──▶ (fold)
                                              │
                                              ▼
                            PERCEIVE (batched, §10)
```

While the drainer executes intent N, the proposer may
already be offering batch N+1, whose members are gated
and journaled — enqueued — concurrently with N's
execution. Admission (gate + journal) is CPU-and-journal
work; execution is tool work; the pipeline overlaps the
two because they share no mutable state: the gate prices
the new intents against policy and the journal as of
their proposal, the executor re-validates at claim
(§7.5), and the journal's total order keeps every fact
attributable to a sequence.

### 9.2 The pipeline's honest limit

Pipelining overlaps *admission* with *execution*. It
does not overlap *perception* with either: the mind does
not see results until a perception sync point (§10.2).
A mind that proposes batch N+1 before perceiving batch N
is speculating twice deep, and §12.1's depth cap is what
bounds that speculation. The spec's position is that
two-deep speculation with a hard depth cap is the useful
regime — admission is cheap and executions are the long
pole — and deeper speculation buys stale pricing (§7.5)
and wasted gate work on premises the next perception
would have corrected. Q-4 measures the two-deep regime;
anything deeper is an open question (§20), not a default.

## 10. Batched perception

### 10.1 The observation set

When perception returns for a task, it returns **one
observation set**: the outcomes of every intent that
concluded since the last perception, in outcome-sequence
order, each observation carrying — per SI-41 §4.5 — its
provenance (task sequence, intent sequence, claim
sequence, outcome sequence, tool, verdict reference) and
its result under the existing bounds (inline cap, blob
reference for overflow, `truncated` labelled). Refusals
at claim, skips, and escalations concluded in the window
are observations too: silence about a proposed intent is
a provenance hole, and this design does not ship those.

### 10.2 Sync points: when perception returns

Perception for a task returns at the first of:

1. **Queue empty** — every queued intent for the task
   has concluded.
2. **Barrier** — an intent declared `barrier: true` has
   concluded. A barrier is the proposer's way of saying
   "I need to see results before anything after this is
   even proposed"; barrier outcomes flush the set
   immediately.
3. **Stop rule** — any SI-41 §10.4 stop rule fires (§12.4
   re-expresses them); the partial set returns with the
   stop's cause.
4. **Drain budget exhausted** (§12.2) — the partial set
   returns labelled `drain-budget-exhausted`, a priced
   outcome, not an error.

The result budget (`tool_max_task_result_bytes`, 1 MiB)
applies to the **set in aggregate**, as it applied to the
task before: batching perception does not multiply the
mind's result allowance, it only delays its delivery.
Overflow follows SI-41 §9.3 — blob reference, inline
prefix, `truncated: true` — per observation.

### 10.3 What the mind is told about the queue

The observation set is accompanied by a **queue summary**:
counts by folded state for the task, the depth at the
sync point, and any `outcome-unknown` or `held` intents
named by sequence. A mind planning its next batch
against a queue it cannot see would be guessing at its
own situation; the summary is perception of the queue
itself, and it is data, tagged like every other
observation (SI-41 §4.5) — a tool result inside it is
never an instruction (§18).

## 11. The operator leash against a queue

The leash — a constitution-level operator STOP the mind
cannot gate, veto, or route around — was adopted into
Gaia's scope with the SI-41 line. Against a serial loop
it is simple (there is at most one thing in flight).
Against a queue it needs exact semantics, which are
these:

### 11.1 The stop

- The operator's stop is a **journaled operator entry**
  (the task loop's existing stop record is the precedent,
  `si/gaia/task.py`'s `record_task_stop`). It is not a
  gate verdict and the gate has no say in it: source
  `operator`, effective at its journal sequence.
- On folding a stop for a task, the drainer **claims
  nothing further for that task**, from that sequence on.
  Queued intents derive to `held` (§5.2): still journaled,
  still visible in `si log` and the SSE stream, and
  **resumable only by an operator resume entry** naming
  the task. A held intent is never silently dropped and
  never silently run later; resume is a new operator
  cause, journaled like the stop was.
- The stop is per-task in v1. A bundle-wide stop is a
  sequence of per-task stops in this spec; a single
  bundle-scope stop entry is an open question (§20), not
  an assumed feature.

### 11.2 In-flight work at the stop

- **Execute-class, in flight:** the drainer kills the
  process group, using the executor's existing timeout
  kill machinery — a leash kill is a timeout whose cause
  is the operator. The outcome is journaled as killed by
  operator stop, partial output labelled partial
  (SI-41 §9.3's precedent for killed commands).
- **Write-class, in flight:** a filesystem tool call is a
  single bounded operation; it runs to its conclusion
  (bounded by `tool_timeout_seconds`) and its outcome is
  journaled normally. Interrupting a half-written file to
  honour a stop would trade a recorded, compensable
  effect for an unrecorded, torn one.
- **Read-class, in flight:** may finish or be abandoned
  at the drainer's convenience; if it finishes, its
  outcome is journaled (a read's record harms nothing);
  if abandoned before execution began, the claim folds
  to `refused-at-claim` with cause `operator-stop`.

### 11.3 The leash and the fold

Because the stop is a journal entry and queue state is a
fold, the leash composes with recovery for free: a
drainer that restarts after a stop (§13) folds the stop
and resumes *not draining* that task. There is no
in-memory "stopped" flag to lose. This is the journal-is
-the-queue design paying for itself, and §17's criteria
test it explicitly.

## 12. Backpressure and budgets

### 12.1 Queue depth cap

Policy `queue_max_depth_per_task`, **default 16**. The
depth of a task's queue is its count of `queued` +
`claimed` intents. The cap is priced at gate check 2 —
the budget check — alongside the existing budgets: an
intent whose admission would exceed the cap REJECTs
with cause `queue-depth-exceeded`.

Why 16, and why at the gate:

- 16 is the existing default task size
  (`max_tool_cycles_per_task`). A queue that can hold a
  whole default task can never be the binding constraint
  on a default task; operators who raise the cycle budget
  raise the depth cap in step, as one sizing decision.
  A larger default would size the buffer for tasks the
  budget forbids.
- Pricing at check 2 keeps one mechanism: depth is a
  budget, budgets are the gate's, and a depth refusal is
  a verdict with a reason on the record — never a silent
  block, a dropped intent, or a proposer that hangs
  waiting for a queue nobody told it about. The mind
  perceives the refusal in the next observation set and
  routes around it (propose less, or wait for the drain);
  throttling becomes a planning fact, not a deadlock.
- Depth is also the speculation bound of §9.2: cap 16
  means at most 16 admitted intents are priced against a
  world the mind has not yet perceived the effects of.

### 12.2 Drain budgets

Policy additions, all journaled as drawn down:

- `queue_drain_budget_seconds` — wall clock per task per
  drain engagement, **default 1920s**, derived rather
  than invented: the worst honest case of a full default
  task draining serially is 16 intents × 120s
  `tool_timeout_seconds` = 1920s. An operator who raises
  either factor re-derives or overrides this in step.
  Exhaustion is a sync point (§10.2), not a fault.
- `queue_claim_lease_seconds` — §5.3's claim lease,
  **default 300s**: 2.5× the per-intent timeout, so a
  claim whose execution is merely slow (at the 120s cap,
  plus outcome journaling) is never suspect, while a
  claim orphaned by a crash is recognisable within
  minutes rather than hours.
- `queue_max_residence_seconds` — **default 3600s**. An
  intent still `queued` this long after admission is
  refused at claim with cause `stale-admission` when the
  drainer next reaches it, however it re-validates: past
  an hour, the priced world is old enough that the honest
  move is a fresh proposal at fresh prices, not
  §7.5's case-by-case catches. (Residence is measured
  from journal timestamps recorded at admission; the
  journal's timestamps are operator-visible record, and
  §18 names the clock trust this does and does not need.)

Per-intent bounds are unchanged: `tool_timeout_seconds`
still kills a runaway execution; the result caps still
bound every outcome.

### 12.3 Budget accounting under the queue (the §10.3 refinement)

SI-41 §10.3 counted *cycles*. Under batching, this spec
redefines the accounting, and flags the redefinition as
one of the two §10 changes this document makes (the
other is §10.2's serial order, §2):

- `max_tool_cycles_per_task` now debits **per intent
  evaluated** — proposed and gated — not per execution.
  Gate and journal work is spent on every proposal,
  admitted or not, and a proposer that could spray
  unlimited rejected proposals at the gate would be
  spending the daemon's resources unpriced. Executions
  are bounded by admissions, and admissions by
  evaluations, so one counter still bounds the whole
  task.
- The result budget is unchanged in size and meaning,
  re-expressed against the batched set (§10.2).

### 12.4 Stop rules, re-expressed

SI-41 §10.4's five stop rules, against a queue:

1. **Budget exhausted** — unchanged in spirit: the next
   intent fails check 2 (cycle, result, or §12.1's depth
   budget). Intents already queued **continue to drain**:
   the budget priced their admission, and un-executing
   admitted work because a later proposal was refused
   would make admission meaningless. The task stops
   *proposing*; it does not stop *draining*.
2. **ESCALATE** — the escalated intent was never queued
   (§5.2). Its dependents by key and by `depends_on`
   cannot proceed past it; in v1 the task halts
   proposing and the drain completes only intents
   independent of the escalation, then the task stops
   for the operator's admission decision. (Per-key halt
   without task halt is an open question, §20 — v1 keeps
   §10.4's conservatism.)
3. **REJECT streak** — two consecutive REJECT verdicts
   *in admission order* halt proposing, as today.
   Verdicts are the streak's unit, exactly as in the
   serial loop; drain outcomes do not reset it.
4. **Execution integrity failure** — a journaled-args
   mismatch or integrity fault (§5.4, §9.2 of SI-41)
   halts the **drain** immediately, task-wide: integrity
   failures are about the record, and the record is
   shared. This is the one stop rule that reaches into
   the queue and freezes it mid-drain.
5. **Mind-reported completion** — closes the task when
   the queue is empty. If intents remain queued, the
   completion is journaled and the task closes **after**
   the drain finishes or the operator stops it: a mind
   may not abandon its own admitted work by declaring
   victory, and the label remains `mind-reported-complete`
   with the same honesty as §10.4 — the mind reported it;
   the journal shows what was actually done.

---

## 13. Crash, restart, and the partially drained queue

A daemon that cannot say what it was doing when it died
will eventually do some of it twice and swear it didn't.
This section is the spec's answer, and it is deliberately
ungenerous: recovery gets less power than intuition
wants, because the world does not give the journal an
undo (SI-41 §11.2) and a queue must not pretend
otherwise.

### 13.1 The recovery fold

On startup, before claiming anything, the drainer folds
the bundle's journal (§5.4) and classifies every
non-terminal intent:

- `queued` → claimable as normal (§7.1), subject to
  §12.2's residence rule.
- `held` → stays held (§11.1). A restart never clears a
  stop.
- `claimed`, lease unexpired → **left alone**. The
  claim's drainer may still be alive (this process may be
  the interloper, §4.3); v1's rule is that a fresh drainer
  facing an unexpired foreign claim refuses to drain the
  task and journals the refusal.
- `claimed`, lease expired, no outcome → the recovery
  procedure of §13.3. This is the only hard case, and it
  is hard because it is genuinely ambiguous: the tool may
  or may not have run.

### 13.2 The ambiguity, stated exactly

A claim with no outcome means one of three things, and
the journal cannot distinguish them: the process died
before executing; it died *during* execution (the effect
is partial or complete, unknowable from here); or it
executed fully and died before journaling the outcome.
Any recovery rule that picks one of these and acts on it
will sometimes be wrong. This spec's rules are therefore
chosen to be wrong **safely** per consequence class —
the cost of a wrong guess is priced by what the class
can do to the world.

### 13.3 Recovery by class

- **Read class: re-claim and re-execute.** A read changes
  nothing; running it again with exactly the journaled
  args (§3) yields a fresh, honest observation. Recovery
  journals a **recovery re-claim** entry naming the
  original claim, then proceeds as a normal claim. The
  original claim stays in the fold as superseded by the
  recovery entry — history is not rewritten, it is
  continued.
- **Write class: `outcome-unknown`, never blindly
  re-executed.** The write may have landed. Re-running it
  could be harmless (same bytes, same target) or could
  clobber a change made since; the journal cannot know.
  Recovery journals the intent `outcome-unknown` and
  notifies (§13.4). The executor's post-state machinery
  (SI-41 §11.1: for `fs_write`, the target's observed
  hash and size) may be run **as a read** and journaled
  as a `recovery-observation` — evidence for the
  operator, labelled as observation, never silently
  promoted to "it ran."
- **Execute class: `outcome-unknown`, never replayed.**
  A command that may have run is never run again on a
  recovery path. No exceptions in v1. `shell_exec` and
  `code_exec` are one-way (SI-41 §11.2); the recovery
  rule for one-way effects is this section's title in
  miniature: what may already have happened is not
  repeated on a guess.

### 13.4 Operator disposition

An `outcome-unknown` intent blocks its ordering key and
its dependents (they can never see a satisfying outcome,
§7.3) and appears in every queue summary (§10.3) until
the operator dispositions it with a journaled
**disposition entry**, one of:

- `accept-unknown` — the intent folds to a terminal
  `outcome-unknown` permanently; dependents with
  requirement `outcome` are satisfied by the disposition
  (the record says the outcome is *unknown*, and they
  proceed on that honest basis); `success`-dependents
  are `skipped`.
- `compensate` — the operator directs a compensation
  intent (SI-41 §11.2's mechanism, an ordinary gated
  intent naming the unknown one); the unknown intent
  stays `outcome-unknown` in history, annotated by the
  compensation's provenance.
- `re-propose` — the operator (or the mind, through the
  operator) proposes the work anew as a fresh intent at
  fresh prices. The unknown original is never revived;
  the new intent is a new fact.

There is no fourth disposition, and in particular there
is no `mark-executed`: a state the journal cannot
establish does not get established by preference.

## 14. Idempotency honesty

What this design guarantees, and what it does not:

- **At-most-once per claim.** A claim leads to at most
  one execution attempt by its drainer. Claims are
  journaled before execution (§5.3), so two executions of
  one intent require two claims, and two claims are two
  journal facts any auditor can see.
- **At-least-once is never promised for world effects.**
  Across a crash, an intent's effect is zero-or-one
  times with a genuinely unknowable middle (§13.2). This
  spec does not use the phrase "exactly once" about
  anything that touches the world, and slices must not
  either.
- **Idempotency is a property of tools, not of the
  queue.** Some tools are repeat-safe by nature
  (`fs_write` of the same bytes to the same target,
  `fs_mkdir` of an existing directory, every read). Some
  are not (`shell_exec` of a non-idempotent command,
  `fs_edit` whose match text the first application
  consumed). The queue neither knows nor guesses which:
  recovery (§13.3) treats the *class* as the unit of
  safety because class is what the gate priced and the
  journal can prove.
- **No idempotency keys are invented.** The journal
  sequence of an intent is its identity; the claim
  sequence is an attempt's identity. A future design that
  wants effect-level deduplication must say what store
  remembers the effects — and §4 has already answered
  where SI keeps such truth, so it will have to argue
  with this spec, in the open, in a successor document.

## 15. Interaction with the existing pieces

### 15.1 SI-42 — registry and gate

Unchanged. The registry's 12 tools, their schemas, and
their consequence classes are the queue's vocabulary;
the gate's tool mapping is the queue's only admission
path. The queue adds exactly three pricings to gate
check 2's budget family (depth, §12.1; the redefined
cycle debit, §12.3) and three policy keys (§12.2), all
resolved under the project's fail-closed policy practice.
Shadow mode (SI-42) composes trivially and usefully: a
shadow gate ADMITs nothing, so a shadow bundle's queue
is empty by construction — the drainer on a shadow
bundle is a fold that finds no work, which is a correct
and testable behaviour, not a special case.

### 15.2 SI-43 / SI-44 — executor and task loop

The executor is called by the drainer exactly as the
task loop calls it today — `execute_journaled`, verdict
re-checked, args re-validated — and its outcome records
are the queue's outcome entries. The task loop
(`si/gaia/task.py`) is the migration surface: `run_task`'s
serial cycle becomes propose-batch → drain → perceive
(§16, Q-1), with the loop's existing task entry, stop
record, and budget-reject plumbing reused. The `si task`
CLI keeps its shape; an operator watching a task sees a
drain where they used to see laps, and `si log` shows
claims between intents and outcomes where it used to
show only outcomes.

### 15.3 SI-45 — the Hearth interface

Orthogonal in v1, deliberately. The queue lives in the
bundle's journal and drains through the SI-side executor;
Hearth's residency governance (the mind as a model,
served) is about where the *proposer* runs, not how its
admitted intents execute. When a Hearth-served mind
proposes batches through the B-4 interface, they enter
this queue unchanged. Nothing in this spec requires a
Hearth-side queue, and §19 rejects building one there.

### 15.4 SI-46 — the service adapter

The adapter's submit-and-return shape gains a queue
reading: a submission opens (or names) a task; the
service's result for a task is its terminal queue summary
with the provenance trail SI-46 already returns. A
service caller polling mid-drain reads the folded queue
state — which is the journal, which the adapter already
exposes — so no new status endpoint is specified here;
Q-3 (§16) settles the adapter's exact queue vocabulary
against this spec's state names (§5.2), and until then
the adapter's existing responses stand.

### 15.5 SI-47 / SI-48 / SI-49 — the SSE spine

Queue transitions are journal entries, and the spine
streams journal entries: **the queue is observable over
SSE with zero new plumbing**. Claims, outcomes, refusals,
skips, stops, and dispositions arrive as events with
journal sequences for ids, filterable per task (SI-48)
and resumable after disconnect by Last-Event-ID (SI-49).
A subscriber can reconstruct any queue state by folding
the event stream from any sequence — because the stream
*is* the journal's edge (§5.1). This spec names one
convention for Q-3 to pin: queue event payloads carry the
§5.2 state name they transition *to*, so consumers do not
each re-implement the fold to answer "what is running."

### 15.6 The Gaia fork

The Gaia fork of Hearth (created 2026-10-30 on the
operator's direction) carries agency onto the runtime
layer as its own directed steps. This spec is SI-side:
the queue is a property of the bundle's journal and the
`si/gaia/` package wherever the bundle lives. If a later
fork-side step hosts a drainer, it hosts *this* drainer
semantics against the same journal — the spec travels
with the journal, and a second, fork-native queue
semantics would be a §19 violation in a new repo instead
of this one.

## 16. Phased implementation plan: Q-1 to Q-4

Each slice is one PR, built by a builder who does not
merge it, reviewed and merged by the dev-loop driver on a
later run — the project's standing pattern. Slices are
ordered so each is independently true: nothing below
requires a later slice to be honest.

### Q-1 — Queue core: fold, claim, serial drain, batched perception

The state machine of §5 as journal entry kinds and a
fold; batch proposal (§6) in the task loop; the drainer
(§4.2) claiming in §7.1 order **serially** (concurrency
limit fixed at 1 by policy, whatever §8's default later
becomes); claim-before-execute wired through the
executor; batched perception with §10.2 sync points 1, 3,
and 4 (barrier is Q-2, with `depends_on`); recovery fold
for the `queued` case (§13.1) and read-class re-claim
(§13.3). Ordering keys enforced (§7.2) — serial drain
makes this natural — with `depends_on` parsed, priced,
and journaled but its satisfaction treated as sequence
order (full requirement semantics land in Q-2; Q-1's doc
must label this precisely). Acceptance: §17.

### Q-2 — Dependencies and bounded read concurrency

`depends_on` requirement semantics (`outcome` /
`success`, skips), `barrier` sync points, declared
ordering keys from proposers, and the read pool at
`queue_max_concurrent_reads` (§8). Claim-time residence
refusals (§12.2). This is the slice where the pipeline
of §9 becomes real: admission of batch N+1 overlapping
drain of batch N is exercised here, under test, with the
journal as the witness.

### Q-3 — Leash, dispositions, and exposure

Operator stop/resume entries against a live drain
(§11), `outcome-unknown` recovery for write and execute
classes with operator dispositions (§13.3–13.4), the
service adapter's queue vocabulary (§15.4), and the SSE
queue-event convention (§15.5). After Q-3, every state
in §5.2 is reachable and every transition in §5 is
implemented; a slice report that says less must say so
in its own doc, in this spec's style.

### Q-4 — Measurement: drain versus serial, on the record

A measurement slice in the SI-40 tradition: scripted
miniature tasks (read-heavy, write-heavy, mixed,
dependency-chained), run under the serial loop and under
the queue at read bounds 1/2/4/8, on this VM, CPU only.
Deliverable is a measurements document with wall times,
queue residence distributions, and journal growth per
task — and a recommendation, from those numbers, for
§8.2's default and §12's caps. Until Q-4 lands, every
performance word in this spec stays UNMEASURED, and any
slice claiming a speedup without Q-4's method is
overclaiming.

## 17. Acceptance criteria for Q-1

Q-1 (§16) is acceptable when **all** of the following
hold, each demonstrated by a test or by a journal
transcript in the slice's evidence, not by assertion:

1. **Journal-only queue.** A task's queue state, folded
   from the journal alone by a fresh process, matches the
   state the drainer acted on — demonstrated by killing
   the drainer mid-task and re-folding.
2. **No ungated execution.** No test or transcript shows
   an execution whose intent lacks a journaled ADMIT;
   REJECTed and ESCALATEd batch members are never claimed
   and never executed.
3. **Claim before execution.** Every outcome entry is
   preceded, in journal sequence, by its claim entry
   naming the same intent sequence; a claim/intent
   mismatch refuses execution and journals the refusal.
4. **Batch honesty.** A mixed batch (admit / reject /
   escalate members) journals each verdict individually,
   in proposal order; no member's verdict references
   another member's.
5. **Serial drain order.** With concurrency fixed at 1,
   executions occur in §7.1 order: per-key FIFO respected
   across a scripted multi-key task, verified from the
   journal.
6. **Batched perception.** A multi-intent task returns
   one observation set per sync point, observations in
   outcome-sequence order, each provenance-complete
   (task, intent, claim, outcome sequences), with the
   §10.3 queue summary attached.
7. **Stop-rule parity.** The §12.4 re-expressions behave
   as specified: budget refusal stops proposing but not
   draining; an integrity failure freezes the drain;
   completion with a non-empty queue closes only after
   drain or operator stop.
8. **Leash fold.** A stop entry journaled while a task is
   queued (drainer not running) is honoured by the next
   drainer from the fold alone: nothing is claimed until
   a resume entry exists.
9. **Recovery of the queued.** A drainer restarted with
   `queued` intents resumes them; a read-class intent
   `claimed` at a simulated crash is recovery-re-claimed
   with the original claim referenced, and executes
   exactly the journaled args.
10. **Suite and scope.** The full suite passes with Q-1's
    new tests added and no prior test weakened; live
    scope remains miniature bundles, `external` still
    escalates, and the slice doc states — in the SI-43/44
    style — exactly what is live and what is not.

## 18. Risks and failure modes

### 18.1 Speculative waste

Batches are written against stale perceptions by design.
A mind that batches badly spends gate, journal, and
execution budget on premises the first result would have
killed. The depth cap (§12.1) bounds the waste per task;
it does not make the waste wise. Q-4's mixed workload
exists partly to size this honestly.

### 18.2 Stale pricing at scale

§7.5 catches class changes at claim, but a long queue is
a portfolio of prices struck against an older world:
files changed by others, directories filled, the work
root itself reorganised. Re-validation refuses the
dangerous cases (class changes, scope breaks); it cannot
refuse the merely *unwise* ones. Residence caps (§12.2)
bound the staleness window by policy; the residual risk
— executing exactly what was asked, later than it should
have been — is inherent to any queue and is stated here
rather than designed around fictitiously.

### 18.3 Perception delay versus reactive work

Batching trades latency for throughput. Work that is
genuinely reactive — each step chosen from the previous
result — gains nothing from the queue and loses the
sync-point wait. The design's answer is that batching is
the proposer's choice: batches of one (§6.1) are the old
loop, always available. A slice that *requires* batching,
or that penalises singleton proposals, is out of spec.

### 18.4 Journal growth rate

A drained task journals strictly more than a serial one:
batch entries, claims, recovery entries, dispositions,
queue summaries' references. SI-41 §17.4's journal-bloat
concern compounds: the result budget bounds perception,
not bookkeeping. Q-4 measures journal growth per task so
the cost is a number, and §20 carries the compaction
question this spec does not answer.

### 18.5 `outcome-unknown` accumulation

Every crash near an execution can mint an
`outcome-unknown` that blocks a key until a human
dispositions it. A daemon that crashes often could
accumulate blocked keys faster than an operator clears
them — a liveness risk created by this spec's honesty,
accepted deliberately: the alternative (guessing) is
§13.2's unsafe wrong. The mitigation is operational, and
named as such: crash frequency is the thing to fix;
§10.3's summary surfaces the backlog where the operator
already looks.

### 18.6 Tool-result poisoning, queued edition

SI-41 §17.2's risk — tool results returning as PERCEIVE
data that a mind might treat as instructions — grows a
queue-specific edge: a batched observation set delivers
many results at once, including results of intents
proposed *before* the poisoned one was seen. The
mitigations are unchanged and restated: observations are
data, provenance-tagged, never instructions; the
proposer's authority comes from the operator's task
instruction and nothing downstream of a tool. Batching
changes the dose, not the rule.

### 18.7 The drainer as a confused deputy

The drainer executes with the executor's authority on
the strength of journal entries. A party that could
write forged ADMIT entries to a bundle's journal could
drive the drainer — but that party already owns the
journal, and with it the bundle's head and history; the
queue widens no trust boundary that was not already the
journal's integrity, which the project's tamper-evidence
work (the NEDB journal's fail-closed tamper tests)
already guards. Stated so a reviewer does not have to
take it on faith — and so nobody claims the queue *adds*
a defence it does not.

## 19. Rejected alternatives

- **An external queue (a broker, a database table, a
  Redis list).** A second store of the project's most
  governance-critical facts, to be kept consistent with
  the journal forever, in exchange for throughput the
  journal already provides at this scale. Rejected by
  §4's whole argument; also rejected by the design rule
  it descends from — never duplicate bytes, never
  duplicate truth.
- **An in-memory queue.** Fast, simple, and amnesiac:
  every crash turns §13 into archaeology. The daemon's
  queue must survive the daemon.
- **Parallel everything.** Draining writes and executions
  concurrently would spend the safety §8.1 exists to
  keep, for throughput Q-4 has not shown anyone needs.
  If measurement ever argues otherwise, it argues with
  numbers, in a successor spec.
- **Atomic batches.** All-or-nothing admission would
  make one bad member veto a batch of good work and,
  worse, would couple verdicts the gate is designed to
  reach independently (§6.1). Partial admission with
  per-member provenance is strictly more honest.
- **Auto-replay of interrupted executions.** §13.3's
  write/execute rule, rejected as an alternative here so
  the record shows it was considered: replay is the
  recovery strategy that feels most like a queue and is
  least compatible with a world that has no undo.
- **A drainer with policy opinions.** Rate-limiting by
  vibes, reordering for efficiency, dropping stale
  intents unjournaled — each is governance leaking into
  the one component that must have none (§4.2). The
  drainer's whole virtue is that it only reads verdicts
  and calls the executor.
- **Result piping between batched intents.** §6.3:
  breaks the journaled-args rule. The queue orders work;
  it does not author it.

## 20. Open questions

Stated as questions, with the spec's current position
where it has one — none of these block Q-1, and none are
answered by pretending:

1. **Per-key halt on ESCALATE.** §12.4 keeps SI-41's
   task-level conservatism for v1. A queue makes a finer
   rule thinkable — halt only the escalated intent's key
   and dependents, drain the rest, and let the task
   continue proposing around the hold. Which behaviour
   serves operators better is an evidence question for
   after Q-3.
2. **Priority lanes.** v1 has none (§4.4 is fairness, not
   priority). The leash is the only preemption. Whether
   an operator "urgent" lane belongs in the design — and
   what it does to FIFO's auditability — is open.
3. **Multi-drainer bundles.** §4.3 assumes one drainer
   per bundle and fails closed otherwise. Contested
   claims, fencing, and lease theft are undesigned.
4. **Depth and bound tuning.** §8.2's 4 and §12's
   defaults are justified starting points, UNMEASURED.
   Q-4 may move all of them; this spec's claims about
   them are claims about their *justification*, not
   their optimality.
5. **Journal compaction for queue bookkeeping.** §18.4's
   growth is priced but not managed. Whether claim and
   fold bookkeeping ever earns a compaction or summarised
   form — without violating append-only honesty — is
   open and probably belongs to the journal's own roadmap
   rather than Gaia's.
6. **Bundle-scope stop.** §11.1 specifies per-task stops.
   A single operator entry stopping every task on a
   bundle is an obvious convenience with non-obvious
   fold semantics (which sequence does it bind at, per
   task?) — open.
7. **Where the drainer lives long-term.** This spec puts
   it in `si/gaia/` beside the executor it drives. The
   Gaia fork (§15.6) raises a real alternative — a
   runtime-resident drainer — whose merits depend on fork
   work that has not happened yet. The journal semantics
   are specified to survive either home.

## 21. Non-goals

- **No code.** This document specifies; Q-1 to Q-4 (§16)
  implement, each as its own reviewed slice.
- **No change to the gate's verdict algebra, the intent
  format's existing fields, the registry, or any policy
  default in force today.** New fields (`ordering_key`,
  `depends_on`, `barrier`) and new policy keys (§12) are
  *specified* here and *landed* by slices, or not at all.
- **No cross-bundle or cross-machine queue.** A queue is
  a fold of one bundle's journal for one bundle's tasks.
  Federating queues is not designed, implied, or
  promised.
- **No exactly-once claims, anywhere, about world
  effects** (§14).
- **No new proposer.** The queue serves whichever mind
  proposes — miniature today, a Hearth-served model when
  that path is measured (SI-45's real-Hearth path remains
  UNMEASURED) — and this spec's claims do not depend on
  the proposer's cleverness. A queue drained by
  construction-derived proposals is still a queue, tested
  as one.
- **No scheduling of humans.** The operator dispositions
  of §13.4 and the leash of §11 are operator *powers*;
  nothing here pages, nudges, or deadlines the operator.

## 22. Sources

- SI-41, *Agentic Powers for the SI Mind* (RFC, merged
  PR #45) — §§3, 4.5, 5, 6.2, 7, 9, 10, 11, 17.2, 17.4
  are this spec's inheritance; §10.2 and §10.3 are the
  sections it refines.
- SI-42 (PR #47) — tool-intent format, registry of 12
  tools, shadow-mode gate. SI-43 (PR #48) — executor and
  filesystem tools, live miniature-scoped. SI-44
  (PR #49) — execution tools and the task loop
  (`si/gaia/task.py`, `record_task_stop`). SI-45
  (PR #50) — Hearth interface, replay/stub, real path
  UNMEASURED. SI-46 (PR #51) — agentic work as a service.
  SI-47 (PR #52) — SSE event spine, journal-derived.
  SI-48 (PR #53) — SSE filtered subscriptions + reference
  consumer. SI-49 (PR #54, open at the time of writing) —
  the resilient following consumer, reconnect +
  Last-Event-ID resume.
- `si/gaia/policy.py` on `main` (787b9bf) —
  `TOOL_POLICY_DEFAULTS`: `tool_timeout_seconds` 120,
  `max_tool_cycles_per_task` 16,
  `tool_max_result_bytes` 65536,
  `tool_max_task_result_bytes` 1048576 — the defaults
  §12's arithmetic derives from.
- SI-40 (PR #46) — the A-1 feasibility measurement:
  the 2 vCPU / 7.7 GiB VM figures §8.2's bound is
  justified against.
- The operator directive and the design points agreed
  with it, recorded 2026-10-30 in the dev-loop STATE.md
  ("UPDATE 2026-10-30 11:27 EDT — QUEUE-DRAIN
  EXECUTOR"), including the operator leash's adoption
  into Gaia's scope (STATE.md, 2026-10-29 19:11 EDT
  update, item 1) and Mark's design rule (2026-10-17):
  SI should never duplicate bytes.
