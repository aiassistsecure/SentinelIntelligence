# SI-66 — Gaia PROPOSE P-3: gateway wiring under preconditions (replay + refusal paths)

**Status:** implemented on `si-66/p3-gateway-wiring` against the merged
SI-63 RFC (docs/SI-63-gaia-propose-hearth-proposer-rfc.md, PR #69), the merged
P-1 slice (SI-64, PR #70) and the merged P-2 slice (SI-65, PR #71, main
`07949b9`). This is §19's **P-3** slice, built exactly to that scope: the §10
gateway call path and the §11 precondition check wired into
`HearthProposer.propose` in `si/gaia/proposer.py` (grown, not replaced);
§13.2's two-failure halts; §15.2's policy keys with `proposer_enabled`
defaulting OFF (in `si/gaia/policy.py`); §13.3's leash-at-call discard; and
the §16.2–§16.4 failure behaviours. **Exercised only against the replay
transport and refusal fixtures** (no transport present; residency not
`Resident`; identity mismatch). No real Hearth node was contacted, no live
model was contacted, and **the real Hearth path remains UNMEASURED**.

---

## 0. Honesty table — the SI-63 §0 rows this slice changes, and those it does not

P-3 updates exactly the rows its evidence touches. Every other row stands
exactly as SI-65 left it; this slice strengthens no other claim.

| SI-63 §0 element | State before P-3 | State after P-3 (this slice's evidence) |
|---|---|---|
| Prompt construction in §6–§7 | PARTIALLY BUILT, scoped (P-1: §6 assembly + §7 rendering; §6.5/§6.6 UNBUILT) | **BUILT as code, replay-scoped:** §6.5's request assembly (`build_gateway_request`: verified model id, system message, rendered perception + §8 contract instruction, no gateway tool-calling per D-1) and §6.6's standing instructions (`STANDING_INSTRUCTIONS`, items 1–7 with 2/3/4 unweakened) ship in `si/gaia/proposer.py` §8 and are exercised only against replay transports |
| A thinking proposer (SI-63) | UNBUILT beyond P-2 replay parsing | **UNBUILT beyond P-3 replay/refusal wiring** — §10's call path, §11.3's precondition check, §13.2 halts, §13.3 discard, §15.2 keys and §16.2–§16.4 behaviours are implemented and tested (tests/test_gaia_proposer_p3.py); any live model contact is P-4 and has not occurred |
| Live model-authored tool intent | NEVER YET OCCURRED | **Unchanged — NEVER YET OCCURRED.** Every intent P-3 returns comes from a scripted replay corpus in tests; none is live or model-authored in any deployed sense |
| Model quality / task success rate | UNMEASURED | **Unchanged — UNMEASURED** |
| Real Hearth path performance | UNMEASURED | **Unchanged — UNMEASURED.** P-3 contacted no Hearth node; its transports are labelled replay/stub fixtures, its one latency figure is client-observed against the stub and labelled as measuring the stub, and no number in this slice is a Hearth-path measurement |
| Gateway inference leg (B-4 conditionality) | CONDITIONAL, not evaluated live | **Unchanged in posture, wired in the proposer:** the proposer's call is conditional on the full §11.3 precondition check (a strictly stronger gate than B-4's two-leg rule, per §10.2); the leg remains not evaluated live |
| Proposal attempt record (§14.2) | Journaling BUILT, replay-scoped (P-2) | **Gateway-scoped journaling BUILT, replay-exercised:** gateway attempts journal the resolved §15.2 policy, the verified model identity (configured + discovered served ids), the §11.3 leg outcomes, and the request digest — before the call (§14.2); the digest verifies against the exact body the replay transport received (asserted by test) |
| Proposal response record (§14.3) | BUILT, replay-scoped (P-2) | **Gateway records added:** kind stays `gaia_proposal_response`, marked `gateway_attempt`, carrying transport outcome + status, gateway-reported usage or its labelled absence (§10.3), the §9 parse outcome, and the `discarded-under-leash` shape (§13.3). One ordering deviation from P-2's replay cycle, pinned in §4 below |

Three consequences, mirroring SI-63 §0's discipline:

1. This slice cannot be cited as evidence that SI "has a thinking agent".
   It earns the narrower claim: *when a bundle's policy explicitly enables
   the proposer, `propose` prices every §11.3 precondition from the journal
   fold before any gateway call, journals the attempt, makes at most one
   call per cycle through the injected transport, journals the response
   before anything is admitted, and returns §5.1's batch — demonstrated
   here only against replay transports and refusal fixtures.*
2. Any demo performed with this code is a replay demo and is labelled as
   such in code, in every journaled record (`P3_SCOPE_LABEL`), and in
   this doc.
3. Nothing downstream changed: the gate, queue, drainer, executor, SSE
   spine, service, seam modules and `run_task` are untouched, and the
   proposer remains unused by default (§15.2, §3 regression evidence).

---

## 1. What was built

### 1.1 The §15.2 policy keys (`si/gaia/policy.py`)

`resolve_proposer_policy` resolves exactly the seven §15.2 keys under the
existing fail-closed discipline, additively (it reads no §6.2 key and
changes no §6.2 resolution — asserted by test):

| Key | Default | Malformed resolves to |
|---|---|---|
| `proposer_enabled` | `False` (absent = OFF) | `False` — only the literal boolean `True` enables |
| `proposer_max_attempts_per_task` | 8 | 0 |
| `proposer_max_batch_intents` | 16 | 0 |
| `proposer_request_timeout_s` | 30 | 0 |
| `proposer_residency_freshness_s` | 60 | 0 |
| `proposer_max_perception_bytes` | 262144 | 0 |
| `proposer_max_response_bytes` | 262144 | 0 |

No unlimited setting exists for any bound (§15.2). `DEFAULT_POLICY` in
`si/bundle.py` is deliberately **not** extended: absent keys already
resolve to these defaults, and leaving stored policies byte-identical is
part of the no-behaviour-change evidence (§3).

### 1.2 The precondition check (§11.3) — `check_preconditions`

A pure pricer (no I/O beyond reading the journal it is handed and an
injected clock). Legs are priced in §11.3's order, with the transport's
existence first (P-3's added refusal fixture), stopping at the first
failure so the attempt record names exactly one failed leg; legs after it
are recorded as *not evaluated*, never as passed:

- **transport** — a transport is present and a gateway base URL is
  configured; otherwise `not-evaluated`, leg `transport`.
- **leash** — the Q-3 fold (re-folded at check time), the perception's
  queue summary, and an optional live `leash_check` callable are all
  clear; a callable that *raises* prices the leash as engaged (a stop
  that cannot be priced is not priced as clear).
- **budgets** — journaled gateway attempts for the task (response
  records, §1.4) below `proposer_max_attempts_per_task`, and the
  assembled perception within `proposer_max_perception_bytes`.
- **identity** — the configured model appears *exactly* in the
  supplied §11.1 discovery list, and the latest residency observation
  names the same identifier; a disagreement is `identity-mismatch`
  (§16.4). No discovery supplied = unverified = `not-evaluated`.
- **residency** — the latest observation prices `Resident`, names the
  configured model, and is no older than
  `proposer_residency_freshness_s` against the injected clock; an
  unparseable poll time fails closed. An intervening non-`Resident`
  transition is *noted on the leg* (§11.2's re-verification is this
  check itself: identity leg + fresh `Resident` observation), not hidden.
- **declaration** — priced only where both digests exist (bundle-side
  B-4 digest vs the fleet-side assertion); disagreement is §16.4's
  `identity-mismatch`; single-sided/absent is a labelled absence (§14.2).
- **catalogue** — re-rendered at check time; a digest differing from the
  perception's is a re-render (the fresh rendering rides the request and
  the attempt record names its digest), never a failure (§11.3's own
  landing for this leg).

Identity discovery is the caller's act (§5.2 construction input
`served_models`): `propose` performs no `GET /v1/models` call of its own,
because a discovery call inside the check would spend the
zero-transport-call evidence §10.2 requires of every refusal.

### 1.3 The gateway call path (§10) — `run_gateway_proposal`

One cycle, in order (the §5.1-shaped adapter is
`HearthProposer._propose_gateway`):

1. **§16.9 first:** with no journal handle, no call is made and nothing
   is journaled — the outcome names leg `journal`.
2. **§13.2 halts priced from the fold** (§1.4): exhausted attempt
   budget, two consecutive parse failures, or two consecutive transport
   failures journal a halt attempt record (cause verbatim) and return
   `halted` with no call.
3. **§11.3 check** (§1.2). Failure journals the attempt record naming
   the failed leg and returns `not-evaluated` (§16.2, zero transport
   calls) or `identity-mismatch` (§16.4, halted immediately, both
   identities recorded). A *second consecutive* precondition failure
   spends D-8's single refresh allowance and halts — as `residency-lost`
   with the observed state named where the leg is residency (§16.6
   step 4).
4. **§14.2 attempt record journaled before the call**, carrying the
   resolved policy, verified identity, leg outcomes and the §6.5
   request digest. If journaling raises, no call is made (§16.9).
5. **The call (§10.3):** one `POST /v1/chat/completions` through the
   injected transport with the policy timeout; no in-cycle retry.
   Exceptions and non-200 statuses are `transport-failed` *data*, never
   raised; a body beyond `proposer_max_response_bytes` is a
   transport-class failure (§15.2); gateway-reported `usage` is
   journaled as reported or as a labelled absence — never estimated.
   A 200 with an unparseable envelope hands its raw body text to §9's
   parser, which prices it as a parse failure (§10.3).
6. **§13.3 leash at call:** the leash is re-priced when the response
   arrives (fold + callable). Engaged, the response is journaled as
   `discarded-under-leash` with its digest and **discarded whole** —
   its content is never parsed for admission, and nothing is admitted.
7. **§9 parse + §14.3 response record**, journaled before any intent
   from the batch can be admitted; parse failure returns `parse-failed`
   with the stage named (§9.5).

On success `propose` returns §5.1's contract — the parsed batch as a
list of `Intent` objects (a §8.7 completion as its one-member memory
batch) — and on inability (§8.8), any failure, or any halt it returns
`None`, with the full labelled outcome on `last_outcome` and in the
journal. **The proposer admits nothing itself** (§12): admission stays
the loop's business, and the happy-path test asserts no self-edit exists
after a successful `propose`.

### 1.4 Halt and budget counters are journal-derived (§5.3)

`_gateway_history` folds this task's gateway attempt/response records:
response records count attempts spent (§15.2 — precondition failures
spend none, §16.2); trailing consecutive events of one class price
§13.2's streaks. The proposer keeps no authoritative counters between
cycles; a replaced proposer continues from the same journal facts.
Halt records are consequences of the history, not new failures, and are
not counted as events (a halt cannot extend its own streak).

### 1.5 The §6.5 request and §6.6 instructions

`build_gateway_request` builds the ordinary OpenAI-shaped body:
`model` = the §11-verified identifier; `messages` = a system message
carrying `STANDING_INSTRUCTIONS` (§6.6 items 1–7, with items 2/3/4 —
gating honesty, format exclusivity, perceived-data discipline — stated
unweakened) and a user message carrying `render_perception_text`
(the task section, the §7 catalogue text, prior cycles / tool results /
residency / queue / budgets as canonical JSON under their §6.1 names,
truncation labels verbatim) plus `OUTPUT_CONTRACT_INSTRUCTION` (§8's
document shape in brief). No gateway tool-calling feature is used
(D-1); streaming is not used (§10.3).

---

## 2. What was NOT built (and remains labelled so)

- No live model contact of any kind — P-4, and only on its Mark-side
  preconditions (a real Hearth node reachable, a real base `Resident`
  under a known identity, a bundle policy explicitly enabling the
  proposer). The HTTP transport path in code is inherited from B-4's
  transport shape and is **not exercised** by this slice's evidence.
- No `GET /v1/models` discovery inside `propose` (§1.2); no Hearth-side
  change of any kind (§10.4).
- No change to `si/gaia/task.py`, the gate, queue, drainer, executor,
  registry, SSE, service, or seam modules. `run_task` does not
  construct a `HearthProposer`; a task gains one only when an operator
  wires one explicitly under an enabling policy.
- §16.5 (proposer crash / process loss) adds no code: the proposer is
  stateless across cycles (§5.3, §1.4), an attempt with no response
  record prices as an interrupted cycle from the fold, and recovery
  never re-calls the gateway — the next cycle is a new attempt with a
  new record. §16.8 is unchanged by construction (nothing downstream
  changed).

---

## 3. Acceptance evidence (§19 P-3, criterion by criterion)

All tests are in `tests/test_gaia_proposer_p3.py`, run by the builder
with the workspace venv; the reviewer's full-suite run belongs to the
dev-loop driver per standing builder discipline.

| §19 P-3 criterion | Evidence |
|---|---|
| Every precondition failure produces zero transport calls and a journaled `not-evaluated` (or §16.4) record naming the failed leg | `test_no_transport_is_a_journaled_zero_call_refusal` (leg `transport`); `test_residency_not_resident_refuses_with_zero_calls` (`Loading` named), `test_residency_lost_state_is_named_in_the_refusal` (`Lost{Evicted}` named), `test_stale_residency_refuses_with_zero_calls`, `test_missing_residency_refuses_with_zero_calls` (leg `residency`); `test_identity_unverified_refuses_with_zero_calls` (leg `identity`); `test_leash_in_force_refuses_with_zero_calls` (leg `leash`); §16.4: `test_identity_mismatch_halts_with_zero_calls` (both identities recorded, halted), `test_residency_model_mismatch_is_identity_mismatch`, `test_declaration_mismatch_is_identity_mismatch` (leg `declaration`). Each asserts `transport.calls == []` (or no transport) and reads the journaled attempt record back from the journal |
| Two-failure halts (§13.2) fire for parse, transport, and precondition classes | `test_two_parse_failures_halt_proposing` (3rd cycle halted, 2 calls total, stages journaled); `test_two_transport_failures_halt_proposing` (exception + HTTP 503, 3rd cycle halted, 2 calls); `test_second_precondition_failure_halts_proposing` (D-8 refresh spent, halts as `residency-lost` with `Loading` named, 0 calls); budget form: `test_attempt_budget_exhaustion_halts_without_a_call` |
| With `proposer_enabled` absent / off, task behaviour is byte-identical to today's for scripted and miniature proposers (regression) | `test_disabled_proposer_makes_no_call_and_returns_p1_outcome` (parametrised: absent / `{}` / `False` / malformed `"yes"` — P-1's exact outcome dict, zero calls, *nothing journaled*); `test_run_task_scripted_proposer_unchanged` and `test_run_task_mind_proposer_unchanged` (real `run_task` runs on a seeded miniature bundle with a policy carrying no proposer keys: same stop reasons and verdicts as B-3); `test_tool_policy_resolution_unchanged_by_proposer_keys`; plus the P-1/P-2 suites and the B-1/B-3/B-4/Q-1 affected subsets staying green (§5) |
| Leash-engaged-at-call (§13.3): a replay response arriving under leash is journaled, discarded whole, nothing admitted | `test_response_arriving_under_leash_is_discarded_whole` (transport journals a queue stop mid-call, then returns a valid proposal: response record is `discarded-under-leash` with digest, `batch` null — never parsed for admission — and zero self-edits exist); `test_leash_callable_engaged_at_call_discards` (same via the live leash callable) |
| Still no real node is contacted; the slice doc states the Hearth path remains UNMEASURED | Every transport in the suite is `si.gaia.hearth.ReplayTransport` or a labelled replay fixture class; `P3_SCOPE_LABEL` rides every P-3 record and outcome; §0's UNMEASURED row above |

Test runs (builder, `~/workspace/si2-venv`, Python 3.12, CPU only):

- New: `tests/test_gaia_proposer_p3.py` — **34 passed**.
- Affected subsets: `test_gaia_proposer_p1.py`, `test_gaia_proposer_p2.py`,
  `test_gaia_b1.py`, `test_gaia_b3.py`, `test_gaia_b4.py`,
  `test_gaia_q1_queue.py`, `test_gaia_q1_task.py`,
  `test_gaia_s2b_hearth_registry.py` together with the new file —
  **225 passed**, 0 failed.
- The full suite was **not** run by the builder (standing discipline:
  one full suite per PR belongs to the reviewer).

---

## 4. Decisions pinned by this slice (SI-63 §20)

- **D-8** — exactly one precondition refresh cycle (the RFC default):
  the first precondition failure returns `not-evaluated`; a second
  *consecutive* one halts, as `residency-lost` where the failed leg is
  residency. Parse and transport classes carry no refresh allowance:
  two consecutive failures end proposing from the next cycle.
- **§5.1 return contract — the P-1 deviation is closed for the enabled
  path only.** Disabled, `propose` returns P-1's labelled outcome dict
  exactly as before (returning a batch would claim a proposal
  happened). Enabled, it returns §5.1's batch / `None` contract, with
  the labelled outcome on `last_outcome`. SI-64's §5.1 deviation note
  anticipated exactly this growth ("P-2/P-3 grow this method into the
  batch return when a real proposal exists to return"); a replay
  proposal is returned *labelled as a replay proposal*, never as a
  live one.
- **§11.1 discovery placement** — discovery is a construction input
  (`served_models`), not a call inside `propose` (§1.2). A slice that
  wires live discovery must journal the discovery's digest in the
  first attempt record of the task, as §11.1 requires; P-3's attempt
  records already carry `served_ids` for that chain.
- **§14.3 ordering (gateway path)** — the response record is journaled
  *before* admission (this RFC's §14.3 ordering), so its `intent_seqs`
  is a labelled absence; P-2's replay cycle journals *after* caller
  admission to carry the seqs (SI-65 §1.6 deviation i). Both orderings
  keep the attempt → response link walkable via `attempt_seq`; the
  gateway path prefers the RFC's stated order because its admission
  happens later, inside the task loop, with no callback to this module.
- **Declaration leg scope** — priced only where both digests exist
  (§1.2); §11.3's "(if any)" read literally, with single-sided states
  recorded as labelled absences rather than silently passed over.

---

## 5. Next slices (unchanged from SI-63 §19)

P-4 (first live proposer — only on Mark-side preconditions: a real
Hearth node, a real base `Resident` under a known identity, an
explicitly enabling bundle policy, fixture tasks including §17.4's
injection fixtures), P-5 (measurement + hardening). Until P-4 lands its
own evidence, the thinking proposer remains UNBUILT beyond P-3
replay/refusal wiring, live model-authored intent has NEVER YET
OCCURRED, and the real Hearth path remains UNMEASURED.

— End of SI-66 (P-3). —
