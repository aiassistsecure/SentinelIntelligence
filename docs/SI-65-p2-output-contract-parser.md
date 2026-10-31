# SI-65 — Gaia PROPOSE P-2: output contract + parser (REPLAY ONLY)

**Status:** implemented on `si-65/p2-output-contract-parser` against the merged
SI-63 RFC (docs/SI-63-gaia-propose-hearth-proposer-rfc.md, PR #69) and the merged
P-1 slice (SI-64, PR #70, main `22fc0a0`). This is §19's **P-2** slice, built
exactly to that scope, in `si/gaia/proposer.py` (grown, not replaced — P-1's
module, per the RFC's single-component design): §8's proposal output contract
as code; §9's six-stage parser; reference verification against the fold (§9.3);
signal translation (§8.7–§8.8); attempt / response record journaling (§14)
against the labelled replay transport. **Everything in this slice is
REPLAY ONLY** — the parser is exercised only against scripted replay
responses; no gateway call exists in this slice, no live model was contacted,
and no live claim is made or implied.

---

## 0. Honesty table — the SI-63 §0 rows this slice changes, and those it does not

P-2 updates exactly the rows its evidence touches. Every other row stands
exactly as SI-64 left it; this slice strengthens no other claim.

| SI-63 §0 element | State before P-2 | State after P-2 (this slice's evidence) |
|---|---|---|
| Prompt construction in §6–§7 | PARTIALLY BUILT, scoped (P-1: §6 assembly + §7 rendering; §6.5/§6.6 UNBUILT) | **Unchanged.** P-2 adds no prompt text and constructs no gateway request; §6.5/§6.6 remain UNBUILT |
| A thinking proposer (SI-63) | UNBUILT beyond P-1 | **UNBUILT beyond P-2 replay parsing** — §8's contract, §9's parser, §9.3 reference verification, §8.7/§8.8 signal translation, and §14 replay journaling are implemented and tested (tests/test_gaia_proposer_p2.py); `HearthProposer.propose` still returns P-1's labelled `not-evaluated` outcome and invokes no transport; precondition-gated gateway wiring (§10/§11) is P-3 and any live model contact is P-4 |
| Live model-authored tool intent | NEVER YET OCCURRED | **Unchanged — NEVER YET OCCURRED.** Every intent P-2 constructs comes from a scripted replay corpus in tests; none is live or model-authored in any deployed sense |
| Model quality / task success rate | UNMEASURED | **Unchanged — UNMEASURED** |
| Real Hearth path performance | UNMEASURED | **Unchanged — UNMEASURED.** P-2 contacted no Hearth node; its transport is a labelled replay/stub and no number in this slice is a Hearth-path measurement |
| Proposal attempt record (§14.2) | Shape pinned as code, journaling UNBUILT (P-1) | **Journaling BUILT, replay-scoped:** `journal_attempt_record` journals the attempt as a `gaia_proposal_attempt` experience before the replay body is parsed, carrying P-2's replay labels; live-attempt journaling remains P-3/P-4 business |
| Proposal response record (§14.3) | Designed only | **BUILT, replay-scoped:** kind pinned as `gaia_proposal_response`; `build_response_record` / `journal_response_record` journal attempt linkage, the labelled replay transport outcome, response digest, parse outcome (stage reached / failing stage), batch + chaining summary, admitted intent seqs, signal verbatim, and bounded labelled excerpts on failure |

Three consequences, mirroring SI-63 §0's discipline:

1. This slice cannot be cited as evidence that SI "has a thinking agent".
   It earns the narrower claim: *a model-shaped response, when one
   eventually exists, will be parsed by a deterministic six-stage
   fail-closed parser whose refusals and admissions are journaled with a
   walkable attempt → response → intent chain — demonstrated here only
   against scripted replay bodies.*
2. Any demo performed with this code is a replay demo and is labelled as
   such in code, in every journaled record, and in this doc.
3. Nothing downstream changed: the gate, queue, drainer, executor, SSE
   spine, service, and `run_task` are untouched, and the proposer remains
   unused by default (§15.2). P-2 supplies no gate and fakes no verdict:
   admission in the replay cycle is the caller's callback (§3).

---

## 1. What was built

### 1.1 The output contract as code (§8)

Pinned in `si/gaia/proposer.py` §5: `PROPOSAL_VERSION = 1`; the closed field
sets `PROPOSAL_TOP_FIELDS` (`{proposal}`), `PROPOSAL_FIELDS` (`{version,
intents, task_signal}`), `PROPOSAL_INTENT_FIELDS` (`{tool, args, reason,
depends_on, ordering_key}`), `PROPOSAL_SIGNAL_FIELDS` (`{kind, detail}`),
`PROPOSAL_DEPENDS_ON_FIELDS` (`{seq, requirement}`); the two signal kinds
(`task_complete`, `cannot_proceed`); and P-2's pinned §15.2 bounds as
defaults-with-arguments — `DEFAULT_MAX_BATCH_INTENTS = 16`,
`DEFAULT_MAX_REASON_CHARS = 2000`, `DEFAULT_MAX_ARGS_BYTES = 16384`
(canonical bytes), `DEFAULT_MAX_RESPONSE_BYTES = 262144`. No unlimited
setting exists for any bound (§15.2).

Field rules follow §8.2 exactly: `intents` may be empty only with a
non-null `task_signal`; `tool` must name a registry entry (no "closest
tool" fallback); `args` is a JSON object validated at parse against the
registry schema (the gate's later validation remains authoritative —
§8.2's two-pass discipline is unchanged); `reason` must be a non-empty
string; `ordering_key` is carried verbatim or `None` — the parser never
invents a key the model omitted.

### 1.2 The six-stage parser (§9) — `parse_proposal`

Deterministic and fail closed (§9.1): stages run in §9.2's order —
1 **envelope** (one bare JSON document; strict per decision D-5's
default: prose, fences, multiple documents, truncated JSON all fail),
2 **schema** (known version, known fields only, correct types — unknown
fields at any level fail; forward compatibility is by version bump),
3 **registry** (every tool exists; every args object validates via
`si.gaia.registry.validate_args`), 4 **references** (§1.3),
5 **bounds** (§15.2 bounds above; an over-size batch is a failure, never
a truncation opportunity), 6 **signals** (§1.4). Any stage's failure
refuses the **whole** response: partial salvage is rejected (§21.4),
`intents` is empty, and the failure never reaches the gate (§9.5).
A result dict is returned for every malformed response — the parser
raises for none of them. Each failure carries the failing stage, the
itemised problems, the response digest, and a bounded (500-char),
labelled excerpt (§9.5/§14.4).

### 1.3 Reference verification (§9.3)

Every `depends_on` entry (bare seq = `outcome` requirement, or
`{seq, requirement}` with `outcome`/`success`) is verified against the
current journal fold for the task (`fold_queue` unioned with the
perception's own intent rows): the seq must name an already-journaled
tool intent **of this task**. The failure detail distinguishes the
cases §9.3 names: hallucinated (no such seq in the journal — including
beyond the tip), future (beyond the journal tip), cross-task (an intent
of another task), and non-intent (a journal row that is not this task's
intent). Malformed entries (non-integer seq, unknown requirement,
duplicate seq, non-positive seq) fail at stage 4 as reference-form
failures, mirroring the gate's §7.3 normalisation vocabulary. With no
journal/task supplied, any declared reference fails closed — the parser
does not guess. References to intents proposed in the *same* batch are
not accepted: batch members have no seqs yet, and §8.2's references are
to prior intents perception showed.

### 1.4 Signal translation (§8.7–§8.8)

- **Completion** (`task_complete`) translates to the existing completion
  form: a `memory` Intent with `collection="experiences"` and record
  `{"kind": "task_complete", ...}` carrying the signal detail — gated
  downstream as an ordinary memory intent, closing a task only as
  `mind-reported-complete`, judged by the operator (§8.7/§13.4). No new
  stop rule is created.
- **Inability** (`cannot_proceed`, non-empty `detail` required)
  translates to **no intent**: a labelled record in the
  proposer-exhaustion family with decision D-4's pinned sub-label
  `model-reported-inability`, its detail verbatim — never completion,
  never a §10.4 stop rule.
- Consistency is stage 6: a signal carrying intents, an unknown signal
  kind, an empty proposal with no signal, and an inability without a
  detail all fail the whole response.

### 1.5 Structural ignoring (§9.4)

Only §8's fields are read, and the schema is closed, so authority-shaped
material has no channel: verdict words, gate-report shapes, or policy
values inside free-text fields (`reason`, `detail`) are carried verbatim
as data — a reason is a claim (§8.4), never an admission input — and
authority-shaped *fields* fail stage 2. Constructed Intents carry exactly
tool/args/reason/depends_on/ordering_key plus the runner's mechanical
task chaining (§8.3: `parents=[task@<seq>]`, `source_seqs=[<task_seq>]`,
distinguishable from model-authored `depends_on` in the response
record's chaining counts). No value the model can write sets a verdict,
a consequence class, or a policy value anywhere in this slice; tests
assert all three (§3).

### 1.6 Attempt / response journaling (§14) — the replay cycle

`run_replay_proposal(journal, raw, task_seq=..., admit=None)` performs
one journaled replay cycle: assemble perception (P-1) →
`journal_attempt_record` journals the §14.2 attempt (P-1's builder, with
P-2's replay labels, the labelled replay transport
`REPLAY_TRANSPORT_LABEL_P2`, precondition outcome honestly recorded as
`replay-only` — no precondition check exists in P-2 — and
`request_digest` a labelled absence: no gateway request exists) →
`parse_proposal` → the caller's optional `admit(intent) -> seq`
callback → `journal_response_record` journals the §14.3 response
(attempt seq linkage, replay transport outcome with latency/usage as
labelled absences — a replay has neither and none is invented —
response digest, parse outcome, batch members/tools/chaining, admitted
intent seqs, signal verbatim, failure excerpt).
`walk_attempt_chain(journal, response_seq)` walks response → attempt →
admitted intents from the journal alone (§14.6) and reports any link
that does not resolve. Two pinned deviations, both forced by §14.3's
own content list: (i) the response record is journaled *after* the
caller's admission so it can carry the admitted seqs §14.3 requires —
in replay there is no execution between parse and admission for this
ordering to misdescribe; (ii) digests-always + bounded-excerpt is the
D-6 mechanism (§14.4): full replay bodies live in the test corpus, and
the journaled digest verifies against them (asserted by test).
`HearthProposer.propose_replay` is the same cycle through the P-1
skeleton's configured task/registry; `propose` itself is unchanged.

---

## 2. What was NOT built (and remains labelled so)

- No gateway wiring, no precondition check (§11.3), no §15.2 policy keys
  (`proposer_enabled` included) — P-3. No residency freshness pricing.
- No live model contact of any kind — P-4, and only on its Mark-side
  preconditions. No §13.2 two-failure halt machinery: §9.5's failure
  *perception* and halt counting belong to the loop wiring of P-3; P-2
  journals each failure with its stage, which is what that machinery
  will count.
- No change to `si/gaia/task.py`, the gate, queue, drainer, executor,
  registry, policy, SSE, service, or seam modules — the diff is P-1's
  module grown by one section, one new test file, and this doc.

---

## 3. Acceptance evidence (§19 P-2, criterion by criterion)

All tests are in `tests/test_gaia_proposer_p2.py`, run by the builder
with the workspace venv; the reviewer's full-suite run belongs to the
dev-loop driver per standing builder discipline.

| §19 P-2 criterion | Evidence |
|---|---|
| Scripted replay corpus covers valid singleton, valid batch with `depends_on`, completion signal, inability signal, and every §9.2 failure stage — each journaled with the designed outcome, none admitted on failure | `test_valid_singleton_parses_to_a_tool_intent`; `test_valid_batch_with_depends_on_and_ordering_key`; `test_completion_signal_translates_to_task_complete_intent`; `test_inability_signal_proposes_no_intent`; stage corpora `test_stage1_envelope_failures` (5 cases), `test_stage2_schema_failures` (8), `test_stage3_registry_failures` (6), stage-4/5/6 tests below; journaling + non-admission: `test_replay_parse_failure_journaled_and_nothing_admitted` (admit callback raises if ever called; response record carries stage 4, `batch: null`, labelled excerpt) |
| Hallucinated / cross-task / future seq references fail at stage 4 with the batch wholly refused | `test_stage4_hallucinated_reference_wholly_refused` (a valid first member is refused with the batch — §21.4), `test_stage4_future_seq_reference`, `test_stage4_cross_task_reference`, `test_stage4_non_intent_row_reference`, `test_stage4_unverifiable_without_journal_fails_closed` |
| Authority-shaped content in model output (§9.4) sets no verdict, class, or policy value in any test | `test_authority_shaped_reason_is_data_and_sets_nothing` (verdict/class/policy prose in `reason` parses as data; the Intent dict contains no verdict/effective_class/policy/gate_report key); `test_authority_shaped_fields_fail_schema_and_set_nothing` (top-level `verdict`/`effective_class` fields fail stage 2, nothing admitted) |
| Attempt → response → intent linkage (§14.3) is walkable in the journal for every admitted replay batch | `test_replay_cycle_journals_attempt_response_and_linkage` — a real-gate admission callback (the bundle `Gate` over `corpus_policy`, journaling self-edits exactly as the loop does) admits a 2-member batch; `walk_attempt_chain` resolves attempt + both intent seqs/edit ids from the journal alone; the response digest verifies against the replayed body |
| Everything labelled REPLAY ONLY in code, records, and slice doc; no live claim made or implied | `REPLAY_ONLY_LABEL` on every parse result, attempt record, response record, and replay outcome (asserted in tests); `REPLAY_TRANSPORT_LABEL_P2` names the transport a scripted replay; this doc's §0 |

Test runs (builder, `~/workspace/si2-venv`, Python 3.12.3, CPU only):

- New: `tests/test_gaia_proposer_p2.py` — **42 passed**.
- Affected subsets: `test_gaia_proposer_p1.py`, `test_gaia_b1.py`,
  `test_gaia_b3.py`, `test_gaia_b4.py`, `test_gaia_q1_queue.py`,
  `test_gaia_q1_task.py`, `test_gaia_s2b_hearth_registry.py` together
  with the new file — **191 passed** (149 P-1-era subset + 42 new),
  0 failed.
- The full suite was **not** run by the builder (standing discipline:
  one full suite per PR belongs to the reviewer).

---

## 4. Decisions pinned by this slice (SI-63 §20)

- **D-4** — inability maps onto the proposer-exhaustion family with the
  sub-label `model-reported-inability` (the RFC's default), pinned as
  `INABILITY_STOP_FAMILY` / `INABILITY_SUB_LABEL`.
- **D-5** — strict envelope (the RFC's default): exactly one bare JSON
  document; no fenced-block tolerance is applied or journaled.
- **D-6** — body storage is digests-always + bounded labelled excerpts
  in the journal; full replay bodies are retained in the test corpus
  and verify against the journaled digests (§1.6 deviation note ii).
- **D-7 (P-2 part)** — response record kind pinned as
  `gaia_proposal_response` (Appendix A's name); parse stages pinned as
  `PARSE_STAGES`; bounds pinned as in §1.1.
- **§14.3 ordering** — in the replay cycle the response record is
  journaled after caller admission so it can carry the admitted seqs
  (§1.6 deviation note i).

---

## 5. Next slices (unchanged from SI-63 §19)

P-3 (gateway wiring under preconditions; replay + refusal paths;
`proposer_enabled` defaulting off), P-4 (first live proposer — only on
Mark-side preconditions: a real Hearth node, a real base `Resident`, an
explicitly enabling bundle policy), P-5 (measurement + hardening).
Until P-4 lands its own evidence, the thinking proposer remains
UNBUILT beyond P-2 replay parsing, live model-authored intent has
NEVER YET OCCURRED, and the real Hearth path remains UNMEASURED.

— End of SI-65 (P-2). —
