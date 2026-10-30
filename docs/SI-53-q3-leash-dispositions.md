# SI-53 — Gaia Q-3: the leash against a live drain, operator dispositions for outcome-unknown claims, and queue exposure

**Status:** implemented on `si-53/q3-leash-dispositions-exposure`
against the SI-50 queue-drain spec (PR #55, merged), the SI-51
Q-1 queue core (PR #56, merged) and the SI-52 Q-2
concurrency/dependency slice (PR #57, merged). This is §16's
**Q-3** slice, built exactly to that scope — no Q-4
(measurement) behaviour is smuggled in, and §0 below says
plainly what is live and what is not. After Q-3, every state
in §5.2 is reachable and every transition in §5 is
implemented, with the two honest qualifications in §0's last
rows (read-class abandonment is the spec's *permitted* option
not taken; bundle-scope stop remains §20's open question).

| SI-50 element | Q-3 state |
| --- | --- |
| §11.1 operator stop/resume entries | **live (unchanged writers, now exercised against a live drain)** — `queue_stop` / `queue_resume` were journaled and folded in Q-1; Q-3's drainer acts on them mid-drain (§1.1) |
| §11.2 execute-class in flight at a stop | **live** — the executor polls the fold during a process run (drainer-installed `leash_check`); on a stop it kills the process group — a timeout whose cause is the operator — and the outcome is journaled `timed-out`, `partial: true`, `kill_cause: "operator-stop"`, partial output recorded |
| §11.2 write-class in flight at a stop | **live** — a filesystem tool call runs to its conclusion and its outcome is journaled normally; the drain then stops (`stop-in-force`) and the remaining queued intents fold to `held` |
| §11.2 read-class in flight at a stop | **live, finish option** — reads in flight finish and their outcomes are journaled (the spec permits finish *or* abandon; this slice takes finish, uniformly — see §3) |
| §11.3 leash and the fold | **unchanged, now load-bearing live** — the stop the live drain honours is the same journal fold; a restarted drainer folds the stop and does not drain the task |
| §13.3 read-class recovery | **unchanged (Q-1)** — recovery re-claim and re-execution with exactly the journaled args |
| §13.3 write-class recovery | **live** — an orphaned write claim (lease expired, no outcome) is journaled `outcome-unknown` (`queue_outcome_unknown`), NEVER blindly re-executed; the target's post-state is gathered AS A READ and journaled as a `queue_recovery_observation`, labelled observation-only, never promoted to an outcome |
| §13.3 execute-class recovery | **live** — an orphaned execute claim is journaled `outcome-unknown`, NEVER replayed; no recovery observation is gathered for execute class (there is no file target to observe — §3) |
| §13.4 operator dispositions | **live** — `accept-unknown` / `compensate` / `re-propose`, journaled as `queue_disposition` by an explicit operator act (`disposition_outcome_unknown`) that first verifies the folded intent is `outcome-unknown` and undispositioned; there is no fourth disposition and no `mark-executed` (an invalid disposition raises and journals nothing) |
| §13.4 dependents of a disposition | **live, pinned (§2)** — undispositioned: dependents wait (neither satisfied nor skipped); `accept-unknown`: `outcome` dependents are satisfied, `success` dependents are skipped; `compensate` / `re-propose`: dependents still wait |
| §15.4 service queue vocabulary | **live** — `queue_state_view` / `GET /tasks/<seq>/queue` speak §5.2's state names; a POST /tasks response and `submit_task`'s return carry the task's `queue`; the provenance trail's queue section includes the three new entry kinds; `POST /tasks/<seq>/dispositions` is the disposition act over the service (400 + nothing journaled on a non-unknown intent; the manifest is re-anchored after a journaled disposition) |
| §15.5 SSE queue-event convention | **live, pinned (§2)** — queue event payloads carry `queue_state`, the §5.2 state name the row transitions its intent TO, on the event dict and in the serialised SSE envelope; non-queue rows carry no `queue_state` key; event names are unchanged from SI-47/48 |
| §16 Q-4 measurement | **not live — Q-4.** No latency, throughput, residence, or contention number appears in this slice; every performance quantity remains UNMEASURED |
| §20 bundle-scope stop | **not live — open question, unchanged** — stops remain per-task in v1 (§11.1) |
| Miniature scope / `external` | **unchanged** — an external-effective intent still ESCALATEs and is never claimed or executed; live agency claims remain miniature-scoped |

---

## 1. What was built

### 1.1 The leash against a live drain — `si/gaia/drain.py`, `si/gaia/executor.py`

Q-1's leash was fold-level: a stop journaled *before* a drain
held it. Q-3 makes the same journaled stop act on a drain
that is already running, in §11.2's three arms:

- **Between rounds** the drainer already re-folds after every
  step (Q-1's discipline: the fold is the state, nothing is
  cached across an execution). A stop journaled while round N
  executes is therefore seen at round N+1: the drain ends
  `stop-in-force`, nothing further is claimed, and the task's
  still-queued intents fold to `held` until an operator
  resume — the resume path is Q-1's, unchanged.
- **Execute-class, in flight:** `Drainer.execute_claim`
  installs a `leash_check` on the executor for the duration of
  that one claim (and always removes/restores it after). The
  check re-folds the journal for the claim's task and reports
  whether its stop is in force. `Executor._run_process` —
  which previously made one `communicate(timeout)` call — now
  polls in 50 ms slices **only when a leash check is
  installed** (with none installed it is one call with the
  full timeout, exactly as before). On a stop it kills the
  process group with the existing timeout-kill machinery
  (own session, `killpg`, the `TIMEOUT_EXIT_CODE` sentinel):
  a leash kill *is* a timeout whose cause is the operator.
  The handlers' detail reads "killed by operator stop (the
  leash, SI-50 §11.2)…", and the execution record — closed as
  status `timed-out` with `partial: true` like any kill —
  additionally carries `kill_cause: "operator-stop"`, so a
  reader never has to parse prose to tell the two kills apart.
- **Write-class, in flight:** filesystem handlers never
  consult the leash check — there is no process to kill, and
  interrupting a half-written file would trade a recorded,
  compensable effect for an unrecorded, torn one (§11.2).
  The write concludes, its outcome is journaled normally,
  and the stop takes effect at the next round.
- **Read-class, in flight:** reads are filesystem calls in
  this executor (and pool executions in Q-2's read pool);
  they finish and their outcomes are journaled. §11.2's
  abandon option is not taken (§3).

The leash remains what §11.1 says it is: a journaled
operator entry, source `operator`, not a gate verdict; the
gate has no say in it, and nothing in this slice adds an
in-memory stopped flag (§11.3).

### 1.2 Outcome-unknown recovery, write/execute — `si/gaia/drain.py`, `si/gaia/queue.py`

`Drainer.recover` keeps Q-1's foreign-claim and read arms
verbatim and replaces the Q-1/Q-2 *deferred* arm with §13.3's
rule. For an orphaned (lease-expired or own) write/execute
claim with no outcome it journals, in order:

1. a `queue_outcome_unknown` naming the claim sequence, the
   effective class, and a cause that states §13.2's ambiguity
   and the rule: NOT re-executed, NOT replayed, pending an
   operator disposition; then
2. for write class, a `queue_recovery_observation` — the
   target's observed existence/kind/size/sha256 under the
   drainer's work root, gathered as a read, labelled
   "observation only — evidence for the operator, never an
   outcome (SI-50 §13.3)" — when the intent names an in-scope
   file target.

The state is journaled before the evidence (§2), and both
are journaled before `recover` returns.

The intent folds to `outcome-unknown` with
`outcome_unknown_seq` / `outcome_unknown_cause` /
`outcome_unknown_claim_seq` and its `recovery_observations`
carried on the folded state. `recover` returns the intent
seqs under the new key `outcome_unknown`; the historical
`deferred` key is still returned, now always empty — Q-3 is
the slice that arm was deferred *to*. The one Q-1 test that
pinned the deferred reading now asserts the Q-3 reading,
with the supersession labelled in the test (the SI-52
pattern); no other prior test was changed by this slice.

An `outcome-unknown` intent is never claimed again: the
fold's claim handling already refuses terminal states, and
`outcome-unknown` is terminal pending disposition (§5.2) —
so "never blindly replayed" is a property of the fold, not
of drainer politeness.

### 1.3 Operator dispositions — `si/gaia/queue.py`

`disposition_outcome_unknown(journal, intent_seq, disposition, …)`
is the operator act. It folds the journal first and fails
closed — raising `ValueError` and journaling NOTHING — when
the intent does not exist, does not fold to
`outcome-unknown`, is already dispositioned, or the
disposition is not one of `DISPOSITIONS`
(`accept-unknown`, `compensate`, `re-propose`). Only then
does it journal the `queue_disposition` entry via the
low-level writer `record_queue_disposition` (which validates
the disposition name on its own, so `mark-executed` cannot
be journaled through either path).

The fold records `disposition` / `disposition_seq` /
`disposition_operator` / `disposition_detail` on the intent
and — deliberately — does **not** change its `state`: the
intent stays `outcome-unknown` in history, annotated by the
disposition (§13.4's `compensate` wording is the general
rule). What a disposition changes is the *dependents'*
reading, in the drainer (§2).

### 1.4 Queue exposure — `si/gaia/service.py`, `si/gaia/queue.py`, `si/gaia/events.py`

- **Service (§15.4).** `queue_state_view(fold, task_seq)` is
  the adapter's vocabulary: the §10.3 summary (now also
  carrying `dispositions` — intent seq → disposition name —
  and `stop_in_force`) plus every intent's folded fields in
  intent order, all in §5.2's own state names (the full
  `ALL_STATES` vocabulary rides on the view as `states`).
  It is exposed as `GET /tasks/<seq>/queue` (404 for an
  unknown task, the trail's discipline), as the `queue`
  field of a POST /tasks response, and via
  `build_queue_view` / `queue_view_for_bundle`. The trail's
  queue section now includes the three new kinds. The
  disposition act is `POST /tasks/<seq>/dispositions` with
  `{intent_seq, disposition, detail?, operator?}`: 200 with
  the journaled entry plus the task's post-disposition queue
  view; 400 (and nothing journaled) for a malformed body, an
  invalid disposition, or an intent that does not fold to
  `outcome-unknown` on that task. A journaled disposition
  re-anchors the manifest, as the task machinery does, so
  the bundle loads cleanly afterwards. `QUEUE_EXPOSURE_LABEL`
  rides on `/health` and `/`.
- **SSE (§15.5).** `queue_state_for_row(collection, row)`
  pins the convention: a queue row's payload carries
  `queue_state`, the §5.2 state it transitions its intent
  TO — self-edit tool rows by their journaled verdict
  (ADMIT → `queued`, REJECT → `rejected`, ESCALATE →
  `escalated`, otherwise `proposed`), claims and recovery
  re-claims → `claimed`, refusals → `refused-at-claim`,
  skips → `skipped`, outcome-unknown / recovery-observation
  / disposition rows → `outcome-unknown` (the latter two
  annotate that state without moving it), stops → `held`,
  resumes → `queued`, and `tool_result` rows → `executed`
  (or `refused-at-claim` when the execution status is
  `refused`). It appears on the event dicts from
  `read_journal_events` and inside the serialised SSE
  envelope beside `collection`/`row`. Non-queue rows carry
  no `queue_state` key at all, and no event name changed —
  SI-47/48 consumers see the same stream, plus one field on
  the rows that are queue transitions. No row carries
  `compensated`: in the fold compensation is an annotation,
  never a state (§5.2), and the convention names states.

## 2. Pinned semantics (where the spec left the slice a choice)

- **Only `accept-unknown` concludes an outcome-unknown
  intent for its dependents.** §13.4 says it verbatim for
  `accept-unknown` (`outcome` satisfied, `success` skipped)
  and says nothing of the kind for `compensate` /
  `re-propose` — both leave the original unknown and direct
  *new* work. So: undispositioned, compensated, and
  re-proposed unknowns all leave dependents waiting (not
  claimable, not skipped); the Q-2 skip propagator gained
  exactly one guard for this (an outcome-unknown dependency
  skips its `success` dependents only once dispositioned
  `accept-unknown`).
- **A second disposition is refused.** §13.4 gives the
  operator one journaled act per unknown; a second would
  either rewrite the first's meaning or stack two
  contradictory conclusions on one record. Fail closed,
  journal nothing — the operator's remedy for a wrong
  disposition is a fresh intent, not a revised history.
- **The outcome-unknown entry precedes the recovery
  observation in the journal** (the state first, then the
  evidence gathered for its disposition — an observation
  journaled for an intent the journal had not yet marked
  unknown would float free of the state it evidences), and
  the observation is gathered for **write class only**:
  §13.3 names the write target's post-state as the
  observable; an execute-class command has no target whose
  observation would be evidence rather than noise.
- **The leash poll slice is 50 ms** while a leash check is
  installed. It is an implementation constant of the kill
  path, not a performance claim: how *fast* a stop lands is
  a Q-4 measurement question and is UNMEASURED here.
- **`queue_stop` carries `queue_state: "held"`** in the SSE
  convention although the stop row names a task, not an
  intent: §5.2's transition the stop effects on the task's
  still-queued intents is exactly `held`, and a consumer
  answering "what is running" needs that word on that row.

## 3. What is NOT in this slice

- **Q-4** — all measurement. The leash's 50 ms poll slice,
  like Q-2's bound of 4, is a policy/implementation starting
  point labelled UNMEASURED; this slice's tests demonstrate
  *that* an in-flight execution is killed and *that* the
  queue holds, never how quickly.
- **Read-class abandonment at a stop.** §11.2 permits the
  drainer to abandon an in-flight read (folding its claim to
  `refused-at-claim` with cause `operator-stop`). This slice
  takes the spec's other permitted option — reads finish,
  outcomes journaled — uniformly, because a read's record
  harms nothing and a uniform rule is auditable. The
  abandonment spelling is therefore the one §11.2 behaviour
  not implemented; it is named here rather than implied.
- **Bundle-scope stop** (§20.6) and **per-key halt on
  ESCALATE** (§20.1) — open questions, untouched.
- **A compensation *executor*.** The `compensate`
  disposition records the operator's direction; building and
  gating the compensation intent itself remains the
  operator/mind's ordinary work through the existing
  SI-41 §11.2 mechanism (the fold's `compensated_by`
  annotation, Q-1). This slice does not auto-propose one.
- **NEDB-backend concurrency claims** — unchanged from Q-2:
  the live-leash and recovery paths are built and tested on
  the embedded (sqlite) journal backend.
- **A thinking proposer** — unchanged from B-3/B-4.

## 4. Evidence (acceptance, criterion by criterion)

Suite discipline (standing rule, 2026-10-30): the builder
ran the NEW tests plus the affected existing subsets; the
single full-suite run is the reviewer's. Subset results on
this branch (venv `si2-venv`):

- new: `tests/test_gaia_q3.py` — **12 passed**;
- affected subsets, one combined run: all
  `tests/test_gaia_*.py` + `test_gate.py`,
  `test_probe_gate.py`, `test_admission_flip.py`,
  `test_journal_ordering.py`, `test_loop.py`, `test_serve.py`,
  `test_si1.py`, `test_run.py`, `test_admit.py`,
  `test_bundle.py` — **334 passed** (the Q-1 recovery test
  updated per §1.2 included; counts are the run's own).

| §16 Q-3 element | Demonstrated by |
| --- | --- |
| Leash kills an in-flight execute; partial labelled; queue held; resume releases (§11.2/§11.1) | `test_gaia_q3.py::test_live_leash_kills_in_flight_execute_and_holds_queue` (a real `shell_exec` sleeping 30 s is stopped mid-flight: outcome `timed-out`, `partial`, `kill_cause == "operator-stop"`, stdout holds the pre-kill output only; the queued write folds `held`, runs only after resume; chain verifies) |
| Leash lets an in-flight write conclude (§11.2) | `::test_live_leash_lets_in_flight_write_conclude` (stop journaled during a slow write execution: the write completes and is journaled, the next intent is held unexecuted, sync `stop-in-force`) |
| Write recovery: outcome-unknown + recovery observation, never re-executed (§13.3) | `::test_recovery_write_claim_folds_outcome_unknown_with_observation` (the write *had* landed before the crash — recovery still does not claim it ran: observation journaled as labelled evidence, a drain touches nothing, file byte-identical) |
| Execute recovery: outcome-unknown, never replayed (§13.3) | `::test_recovery_execute_claim_is_never_replayed` (the one-way command's side-effect file is never created by recovery or by a drain) |
| Dependents wait undispositioned; `accept-unknown` releases `outcome`, skips `success` (§13.4/§7.3) | `::test_dependents_wait_then_accept_unknown_releases_and_skips` (pre-disposition drain is `blocked` with no executions and no skips; post-disposition the outcome dependent executes, the success dependent is journaled `skipped`, the unknown's state stays `outcome-unknown` annotated) |
| Disposition refusals journal nothing; no `mark-executed`; no second disposition (§13.4) | `::test_disposition_refusals_journal_nothing` (journal tip is byte-identical after each refused act), `::test_compensate_disposition_annotates_without_satisfying` |
| Service queue vocabulary + disposition endpoint (§15.4) | `::test_service_queue_view_and_task_response_carry_queue`, `::test_service_disposition_endpoint_journals_the_act`, `::test_submit_task_response_carries_queue_vocabulary`, `::test_http_queue_and_disposition_routes` (real HTTP: queue 200 in §5.2 names, disposition of an executed intent 400, unknown task queue 404, label on /health) |
| SSE queue-event convention (§15.5) | `::test_sse_queue_events_carry_queue_state` (event dicts and the serialised payload carry `queue_state` for claim/outcome/outcome-unknown/disposition/stop rows; non-queue rows carry no key; the mapping's verdict/refusal arms pinned as units) |

## 5. Files

| File | Change |
| --- | --- |
| `si/gaia/queue.py` | `record_queue_outcome_unknown` / `record_queue_recovery_observation` / `record_queue_disposition` writers; `DISPOSITIONS`; the fail-closed operator act `disposition_outcome_unknown`; fold carries outcome-unknown + observation + disposition fields; summary gains `dispositions` + `stop_in_force`; `queue_state_view` (§15.4 vocabulary) |
| `si/gaia/drain.py` | §13.3 write/execute recovery arm (observation + outcome-unknown; `deferred` kept, always empty); §13.4 dependent reading (only `accept-unknown` satisfies `outcome` / skips `success`); live-leash wiring in `execute_claim` (installs/removes the executor's `leash_check` per claim); drain result gains `outcome_unknown` |
| `si/gaia/executor.py` | `leash_check` hook; `_run_process` polls and leash-kills the process group (6th return: leash-killed), handlers name the operator cause in their detail; execution records carry `kill_cause: "operator-stop"` on a leash kill |
| `si/gaia/events.py` | `queue_state_for_row` (§15.5 convention); `queue_state` on event dicts and in the SSE envelope for queue rows only |
| `si/gaia/service.py` | `QUEUE_EXPOSURE_LABEL`; `build_queue_view` / `queue_view_for_bundle` / `submit_disposition`; `GET /tasks/<seq>/queue` + `POST /tasks/<seq>/dispositions` routes; `queue` on task responses; trail queue section includes the three new kinds; labels on health/description |
| `si/gaia/task.py`, `si/gaia/__init__.py` | labels/summaries updated: Q-3 is built, not future |
| `tests/test_gaia_q3.py` | **new** — 12 tests (live leash ×2, recovery ×2, dispositions ×3, service ×4, SSE ×1) |
| `tests/test_gaia_q1_drain.py` | the one recovery assertion that pinned Q-1's superseded deferred reading now asserts Q-3's, labelled (§1.2) |
| `docs/SI-53-q3-leash-dispositions.md` | this doc |
