# SI-51 — Gaia Q-1: the queue-drain core (batch proposal, the serial drainer, claim-before-execute, batched perception, queued/read recovery)

**Status:** implemented on `si-51/q1-queue-core` against the
SI-50 queue-drain spec (PR #55, merged at `59ed1be`). This is
§16's **Q-1** slice, built exactly to that scope — no Q-2, Q-3,
or Q-4 behaviour is smuggled in, and §0 below says plainly what
is live and what is not. Q-1 turns the journal into the queue:
a proposal is a batch, admission is per-member gating into a
journaled queue, a serial drainer claims before it executes,
and perception returns once per sync point instead of once per
tool.

| SI-50 element | Q-1 state |
| --- | --- |
| §4 journal-as-queue, drainer in `si/gaia/` | **live** — `si/gaia/queue.py` (entry kinds + fold), `si/gaia/drain.py` (admission + drainer) |
| §5 intent lifecycle as journal entry kinds + fold | **live** — states folded from the journal alone, never stored mutably (§5.1/§5.4) |
| §6 batch proposal (non-atomic, per-member verdicts, no result piping) | **live** — task loop + `si task` / service script converters (nested array = one batch) |
| §7.1 claim order, serial | **live** — concurrency fixed at **1 by policy** (`Q1_CONCURRENCY`); concurrency itself is Q-2 |
| §5.3 / §9.2 claim-before-execute through the executor | **live** — claim journaled before execution; a claim naming a different intent seq refuses, journaled |
| §7.2 ordering keys | **live** — derived at gate time, priced into the verdict entry, enforced as per-key FIFO by the drainer |
| §7.3 / §6.2 `depends_on` | **parsed, priced, journaled — satisfaction treated as sequence order (see §3)**; requirement semantics (`outcome` vs `success`) and barrier satisfaction are **Q-2** |
| §10.2 sync points 1 (queue empty), 3 (stop rule), 4 (drain budget exhausted) | **live** |
| §10.2 sync point 2 (barrier) | **not live** — `barrier` is parsed, priced, and journaled; it creates no sync point in Q-1 (Q-2) |
| §10.3 queue summary in perception | **live** |
| §11 leash (operator stop / resume as journal entries) | **live at the fold** — a stop journaled while no drainer runs holds the task's queued intents for the next drainer (§17.8); operator-initiated terminal dispositions (cancel / compensate / force-release) are **Q-3** |
| §12.1 queue depth cap priced at gate check 2 | **not priced in Q-1** — the Q-1 build list (§16) does not include it; `queue_max_depth_per_task` remains a named default for the slice that prices it. (The cycle/result budgets of §12.3 are live, debited per intent evaluated.) |
| §12.4 stop rules | **live** — budget refusal and reject streak halt *proposing*, not draining; escalation halts proposing, queued members drain; integrity failure freezes the drain; completion closes only after the drain |
| §13.1 recovery fold, `queued` case | **live** — a restarted drainer resumes queued intents from the fold alone |
| §13.3 read-class re-claim | **live** — lease-expired read claims are recovery-re-claimed with the original claim referenced, executing exactly the journaled args |
| §13.3 write/execute outcome-unknown + operator dispositions | **not live — Q-3.** Q-1 *detects and defers*: an expired write/execute claim is reported as deferred and left claimed; it is never blindly replayed and never silently resolved |
| §13.1 foreign live claim | **live** — an unexpired claim by another drainer is reported as foreign and left alone (fail closed) |
| §15 measured-or-labelled | **all queue performance is UNMEASURED** — no latency, throughput, or contention number appears in this slice; measurement is Q-4 |
| Miniature scope / `external` | **unchanged** — the executor's miniature scope stands; an external-effective intent still ESCALATEs and is never claimed or executed |

---

## 1. What was built

### 1.1 The queue's journal entry kinds and fold — `si/gaia/queue.py` (new)

The §5 state machine as journal entries (all `experiences`
records, written only through `Journal._append`'s hash chain):
`queue_batch` (§6.1), `queue_claim` (§5.3), `queue_recovery_reclaim`
(§13.3), `queue_refusal` (§5.2/§7.5), `queue_stop` and
`queue_resume` (§11.1). Writers return `{seq, row_id, ...}` with
the entry's own journal seq read back the way the task loop
reads its entries back.

`fold_queue(journal, task_seq=None)` reconstructs queue state
from the journal alone: per intent — verdict-derived state
(`proposed` / `rejected` / `escalated` / `queued`), claim fields,
outcome, folded state (`executed` / `refused-at-claim` /
`skipped` / `held` / `outcome-unknown`), the priced queue block
(ordering key, normalised `depends_on`, barrier, batch seq) read
from the journaled gate report — never recomputed — and
per-task leash bookkeeping (latest stop vs resume). Integrity
faults (§5.4) are detected for queue-managed intents: an outcome
with no prior claim, or a second terminal entry, folds the
intent to `outcome-unknown` and records the fault; legacy
direct-execution journals (B-2/B-3's `execute_tool_cycle` path)
fold without retroactive faults. `queue_summary(fold, task_seq)`
produces the §10.3 summary (counts by folded state, depth, named
outcome-unknown and held seqs, integrity faults).

One integration rule worth naming: queue entries name their
intent by the record's `edit_id`, **not** by the journal row's
`self_edit_id` column — that column names the self-edit an
*outcome* belongs to, and existing readers (task outcome lookup,
provenance walks, the B-3 portability contract) find outcomes by
it. Claims are provenance, not outcomes.

### 1.2 Gate pricing — `si/gate.py`, `si/probes.py`, `si/gaia/context.py`

`Intent` gains `ordering_key` / `depends_on` / `barrier`
(defaults round-trip byte-stably for pre-Q-1 intents; a B-1 test
pins `to_dict()` equality). Check 2 prices the queue fields:

- the **ordering key** (§7.2): a declared key wins; otherwise the
  class default is derived at gate time — write-class fs tools:
  `fs:` + the canonical resolved target; execute-class:
  `exec:` + the work root; house tools (`journal_query`,
  `bundle_status`, `probe_run`): `house`; read-class: no key;
- **`depends_on`** (§7.3/§6.2): normalised to
  `{seq, requirement}` entries and validated fail-closed against
  journal facts the task loop supplies on the gate context
  (`tool_journal_tip`, `tool_task_intent_seqs`) — malformed
  shape, a not-yet-journaled (future or own) sequence, or a
  sequence outside the task REJECTs that member only;
- the priced block (`ordering_key`, `depends_on`, `barrier`,
  `batch_seq`, `effective_class`) is recorded on the report as
  `queue` and journaled with the verdict — present on tool
  reports only, so every other report's dict form is unchanged.

### 1.3 Admission — `si/gaia/drain.py: admit_tool_intent`

Mirrors `execute_tool_cycle`'s journaling (gate → self-edit →
provenance → non-execution outcome for REJECT/ESCALATE) **minus**
execution: an ADMIT leaves the intent `queued` with its result
`pending`. This is the §15.2 split — gating/journaling is the
admission half; the executor is called later, by the drainer,
exactly as the task loop called it before.

### 1.4 The drainer — `si/gaia/drain.py: Drainer`

Serial (Q-1 policy: `Q1_CONCURRENCY = 1`). `drain_task`:

1. **recovers first** (§5 below), then loops on a fresh fold;
2. executes its own outstanding claims first (e.g. a recovery
   re-claim), otherwise claims the lowest-seq **claimable**
   intent — folded state `queued`, no stop in force, per-key
   FIFO clear (no earlier non-terminal intent on the same
   ordering key), `depends_on` satisfied per §3 — journaling the
   claim (lease from `queue_claim_lease_seconds`: absent → 300,
   malformed → 0, the project's most-restrictive reading);
3. executes through `Executor.execute_journaled` with the claim
   as its ticket (§1.5);
4. stops at a sync point — queue empty / stop rule / drain
   budget exhausted (`queue_drain_budget_seconds`: absent →
   1920, malformed → 0) — or freezes on an integrity refusal;
5. returns **one observation set** (§10): the round's outcomes
   in outcome-seq order, each provenance-complete (task, intent,
   claim, outcome seqs), with the §10.3 queue summary attached.

### 1.5 Claim-before-execute — `si/gaia/executor.py`

`execute_journaled` gains optional claim parameters
(`claim_intent_seq`, `claim_seq`), checked before anything else:
if the journaled self-edit's own seq is not the seq the claim
names, execution is refused — *"claim/intent mismatch"* — and
the refusal is journaled as a `refused` outcome (an execution
that did not happen, saying why, like every executor refusal).
Direct callers (B-2/B-3 paths, tests) pass no claim and are
unchanged. The executor also round-trips the new Intent fields
from the journaled dict, so the journaled-args rule covers them.

### 1.6 The task loop — `si/gaia/task.py`

`run_task`'s cycle is now propose-batch → admit → drain →
perceive (the §15.2 migration). A proposer returns one intent
(the degenerate batch of one — every B-3 behaviour is preserved
for singleton proposals) or a list. Per batch: one
`queue_batch` entry; members gated and journaled individually
in proposal order; stop-rule halts (§12.4) stop *admission* and
the drain still runs; the observation set is appended to the
result's `perceptions`, and the journaled outcome records feed
the next PERCEIVE exactly as before. Budgets debit per intent
**evaluated** (§12.3). `task_complete` in a batch is processed
after the batch drains (§12.4.5). A new stop reason,
`operator-stop`, labels a drain held by the leash — NOT one of
§10.4's five rules, labelled as itself like
`proposer-exhausted`. Return dict: all B-3 keys unchanged, plus
`perceptions` and `batch_seqs`.

Script surfaces follow: `si task` and the B-5 service converter
accept a nested JSON array as one batch and pass
`ordering_key` / `depends_on` / `barrier` through to the Intent
(service script validation flattens batches for its per-entry
checks). The B-5 provenance trail gains a `queue` section (the
task's batch/claim/stop entries) — trail membership now matches
the SSE task filter's, which counts any record naming the task
seq; SI-48's chain test was extended accordingly (see §5).

---

## 2. Pinned semantics (where the spec left the slice a choice)

- **Claim lease and drain budget resolution** (§12.2 leaves the
  malformed-value reading to the implementation): absent →
  spec default (300s / 1920s); a valid non-negative int → itself;
  anything else → **0** (most restrictive — an unparseable lease
  is already expired; an unparseable drain budget drains
  nothing). The task loop's *cycle/result* budgets keep their
  B-3 resolution, unchanged.
- **Claim timestamps** come from the claim row's journal `ts`;
  an unparseable timestamp is treated as *unexpired* (fail
  closed — a foreign claim is never stolen on a guess).
- **Executor `refused` outcomes fold to `refused-at-claim`** —
  in the queue flow every executor refusal happens after a
  claim, so the folded name is the true one.
- **Recovery of an expired write/execute claim defers** (does
  not mark `outcome-unknown`): §13.3's disposition machinery is
  Q-3, and Q-1 does not pre-decide it; the deferral is reported
  in `recover()`'s result, never silent.

## 3. The `depends_on` label, precisely (§16 Q-1)

In Q-1, `depends_on` is **parsed** (int or `{seq, requirement}`
entries), **priced** (check 2 validates shape, journaled-ness,
and same-task scope, failing closed), and **journaled** (the
normalised declarations ride the verdict entry). Its
**satisfaction is treated as sequence order**: the drainer
considers a dependency satisfied when the named earlier-seq
intent has reached *any* terminal folded state. Q-1 does not
distinguish the `outcome` requirement from `success`, does not
skip a dependent whose dependency failed, and does not treat
`barrier` as a sync point — all of that is Q-2 (§10.2 point 2,
§7.3 requirement semantics). Because admission is in-order and
the drainer is serial, sequence order is the order Q-1 actually
executes in; the label exists so no reader mistakes Q-1's
treatment for the full §7.3 semantics.

## 4. What is NOT in this slice

- **Concurrency** (§8) — Q-2. The drainer is serial by policy,
  not by accident; nothing here measures or tunes parallelism.
- **Barrier sync points and `depends_on` requirement
  semantics** — Q-2 (see §3).
- **Write/execute outcome-unknown recovery and operator
  dispositions** (cancel / compensate / force-release) — Q-3.
  Q-1 detects, defers, and reports; it never replays a write or
  execute intent blind.
- **Queue depth-cap pricing** (`queue_max_depth_per_task` at
  check 2) and **residence limits** (§12.1/§12.2) — not in the
  §16 Q-1 build list; the names and defaults stand in the spec
  for the slice that prices them.
- **All performance** — Q-4. Every number in this slice is a
  functional count (tests, seqs, bytes of journal); no latency
  or throughput claim is made or implied. UNMEASURED.
- **A thinking proposer** — unchanged from B-3/B-4: proposals
  still come from scripted proposers or the miniature mind.

## 5. Evidence (§17, criterion by criterion)

Suite discipline (standing rule, 2026-10-30): the builder ran
the NEW tests plus the affected existing subsets; the single
full-suite run is the reviewer's. Subset results on this
branch (venv `si2-venv`, leftover `nedbd-v2` processes killed
before runs):

- new: `tests/test_gaia_q1_queue.py` 12 passed,
  `tests/test_gaia_q1_drain.py` 12 passed,
  `tests/test_gaia_q1_task.py` 11 passed (35 new tests);
- affected subsets, one combined run: all `tests/test_gaia_*.py`
  (B-1…B-5, SSE, SSE filter, Q-1) + `test_gate.py`,
  `test_probe_gate.py`, `test_admission_flip.py`, `test_loop.py`,
  `test_journal_ordering.py`, `test_serve.py`, `test_si1.py`,
  `test_run.py`, `test_admit.py`, `test_bundle.py` —
  **293 passed**.

| §17 criterion | Demonstrated by |
| --- | --- |
| 1. Journal-only queue | `test_gaia_q1_drain.py::test_restarted_drainer_drains_a_queued_intent` (a fresh drainer instance — no carried state — folds the journal and completes the queued work); `test_gaia_q1_queue.py::test_fold_of_prefix_is_stable`; `test_gaia_q1_task.py::test_batch_members_journaled_individually_in_proposal_order` (post-run fold agrees with the acted state) |
| 2. No ungated execution | `test_gaia_q1_drain.py::test_mixed_batch_reject_and_escalate_members_never_drain` (REJECT + ESCALATE members: never claimed, never executed); `test_gaia_q1_task.py::test_escalated_member_halts_proposing_queued_member_drains` |
| 3. Claim before execution | `test_gaia_q1_drain.py::test_batch_of_three_writes_admitted_without_executing_then_drain` (every outcome preceded by its claim naming the same intent seq), `test_claim_intent_mismatch_refuses_and_writes_nothing` (mismatch refuses, refusal journaled, nothing written); `test_gaia_q1_task.py::test_one_observation_set_per_batch_with_queue_summary` (claim seq < outcome seq per observation) |
| 4. Batch honesty | `test_gaia_q1_task.py::test_batch_members_journaled_individually_in_proposal_order` (each verdict journaled individually, in proposal order), `test_no_verdict_references_another_member` |
| 5. Serial drain order | `test_gaia_q1_drain.py::test_ordering_keys_drain_same_path_writes_in_seq_order`, `test_claimable_fifo_dependencies_and_stop`; `test_gaia_q1_task.py::test_gate_prices_default_ordering_keys` (claims in intent-seq order across keys; same-key writes land in seq order) and `test_declared_ordering_key_and_barrier_are_priced` |
| 6. Batched perception | `test_gaia_q1_drain.py::test_drain_returns_one_batched_observation_set`; `test_gaia_q1_task.py::test_one_observation_set_per_batch_with_queue_summary` (one set per sync point, observations in outcome-seq order, provenance-complete, §10.3 summary attached) |
| 7. Stop-rule parity (§12.4) | `test_gaia_q1_task.py::test_cycle_cap_mid_batch_stops_proposing_but_drains` (budget refusal stops proposing, not draining), `test_escalated_member_halts_proposing_queued_member_drains`, `test_completion_member_closes_only_after_batch_drains`; `test_gaia_q1_drain.py::test_integrity_failure_freezes_the_drain` (TOCTOU mid-drain freezes; next intent never claimed) |
| 8. Leash fold | `test_gaia_q1_drain.py::test_queue_stop_holds_the_drain_until_resume` (stop journaled while no drainer runs; the next drainer claims nothing until a resume entry exists); `test_gaia_q1_queue.py::test_stop_holds_queued_and_resume_unholds` |
| 9. Recovery of the queued | `test_gaia_q1_drain.py::test_restarted_drainer_drains_a_queued_intent` (§13.1 queued case), `test_recovery_reclaims_expired_read_claim_and_drain_executes` (§13.3: re-claim references the original claim, executes exactly the journaled args), `test_recovery_expired_write_claim_is_deferred_not_unknown` (Q-3 deferral, §2), `test_recovery_foreign_unexpired_claim_is_left_alone`; `test_gaia_q1_queue.py::test_recovery_reclaim_supersedes_original_claim` |
| 10. Suite and scope | subset runs above (full suite = reviewer's); miniature scope and `external`-escalates pinned by the unchanged B-2/B-3/B-5 tests in the same subsets; this doc's §0/§4 |

Integration notes (found by the subsets, fixed on this branch):
the queue writers initially took the journal row's
`self_edit_id` column, which existing outcome readers own —
claims now name their intent by record `edit_id` only (§1.1);
and SI-48's task-chain test predated queue entries — the B-5
trail now carries the task's queue entries and the test's
expected chain includes them (§1.6), keeping "filter == trail"
true of the enlarged, honest chain.

## 6. Files

| File | Change |
| --- | --- |
| `si/gaia/queue.py` | **new** — queue entry kinds, writers, fold, queue summary |
| `si/gaia/drain.py` | **new** — batch admission + the serial drainer + batched perception + recovery |
| `si/gate.py` | Intent queue fields; check-2 ordering-key derivation + `depends_on` pricing; report `queue` block |
| `si/gaia/executor.py` | claim ticket on `execute_journaled` (mismatch refuses, journaled); Intent field round-trip |
| `si/gaia/task.py` | propose-batch → admit → drain → perceive; §12.4 halts; `operator-stop` reason; `perceptions`/`batch_seqs` in the result |
| `si/gaia/context.py`, `si/probes.py` | queue-pricing context fields (batch seq, journal tip, task intent scope) |
| `si/cli.py`, `si/gaia/service.py` | batch + queue-field script conversion; service validation flattens batches; provenance trail gains `queue` |
| `si/gaia/__init__.py` | package summary for Q-1 |
| `tests/test_gaia_q1_queue.py` | **new** — 12 tests (writers + fold + summary) |
| `tests/test_gaia_q1_drain.py` | **new** — 12 tests (admission, drain order, claims, mismatch, recovery, leash, perception, integrity freeze) |
| `tests/test_gaia_q1_task.py` | **new** — 11 tests (task-level §17: batches, pricing, depends_on, §12.4) |
| `tests/test_gaia_sse_filter.py` | expected task chain includes the task's queue entries (SI-51) |
| `docs/SI-51-q1-queue-core.md` | this doc |
