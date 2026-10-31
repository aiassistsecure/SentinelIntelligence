# SI-64 — Gaia PROPOSE P-1: perception assembly + catalogue rendering (no gateway call)

**Status:** implemented on `si-64/p1-perception-catalogue` against the merged
SI-63 RFC (docs/SI-63-gaia-propose-hearth-proposer-rfc.md, PR #69, main
`26d12dd`). This is §19's **P-1** slice, built exactly to that scope:
`si/gaia/proposer.py` — perception assembly (SI-63 §6) and catalogue
rendering (§7) as pure, journal-derived functions; digests (§6.4, §7.4);
bounds and truncation labels (§6.3). The proposer's `propose` exists but
returns the labelled not-evaluated outcome — P-1 makes **no** gateway call,
by construction (a test asserts the transport is never invoked).

---

## 0. Honesty table — the SI-63 §0 rows this slice changes, and those it does not

P-1 updates exactly one §0 row, with the evidence in §3 below. Every other
row stands exactly as SI-63 left it; this slice strengthens no claim.

| SI-63 §0 element | State before P-1 | State after P-1 (this slice's evidence) |
|---|---|---|
| Prompt construction in §6–§7 | UNBUILT — design only | **PARTIALLY BUILT, scoped:** perception assembly (§6) and catalogue rendering (§7) are implemented as pure functions in `si/gaia/proposer.py` and tested (tests/test_gaia_proposer_p1.py). This row's §6.5 gateway-request assembly and §6.6 standing-instructions prompt text remain **UNBUILT** — no prompt string ships in this slice, and no gateway request is constructed |
| A thinking proposer (SI-63) | UNBUILT | **UNBUILT beyond P-1** — `HearthProposer.propose` returns the labelled `not-evaluated` outcome and invokes no transport; parsing (§8/§9), precondition-gated gateway wiring (§10/§11), and any model contact are P-2/P-3/P-4 |
| Live model-authored tool intent | NEVER YET OCCURRED | **Unchanged — NEVER YET OCCURRED.** P-1 authors no intent |
| Model quality / task success rate | UNMEASURED | **Unchanged — UNMEASURED** |
| Real Hearth path performance | UNMEASURED | **Unchanged — UNMEASURED.** P-1 contacted no Hearth node; no number in this slice is a Hearth-path measurement |
| Proposal attempt record (§14.2) | Designed only | **Shape pinned as code, journaling UNBUILT:** `build_attempt_record` returns the *would-be* attempt record (digests, truncation labels, labelled absences); it journals nothing — journaling an attempt belongs to the slice that makes a call (P-2/P-3) |

Three consequences, mirroring SI-63 §0's discipline:

1. This slice cannot be cited as evidence that SI "has a thinking agent".
   It earns the narrower claim: *what a proposer would be shown, and the
   catalogue it would be shown it from, are now deterministic,
   digest-verifiable functions of the journal and the registry.*
2. Any demo performed with this code is still a scripted or miniature
   demo and must be labelled as such.
3. Nothing downstream changed: the gate, queue, drainer, executor, SSE
   spine, service, and `run_task` are untouched, and the proposer is
   unused by default (§4 below).

---

## 1. What was built

### 1.1 Catalogue rendering (§7) — `render_catalogue`

- The catalogue is a **rendering of the registry at call time** (§7.1).
  With no explicit registry argument, `si.gaia.registry.REGISTRY` is
  consulted on every call; there is no hand-maintained tool list in
  `proposer.py` to drift from the gate's list. A test mutates the
  registry in place and observes the rendering and its digest change,
  then observes both revert when the mutation is removed.
- Each entry carries, from the `ToolSpec`: `name`, `version`,
  `description` verbatim, the **complete** `args_schema` the gate will
  validate against, a **summarised** result schema (required fields +
  per-property types — §7.1: the model plans around result shapes; full
  result schemas are gate business), `consequence_class`,
  `reversibility`, and `default_timeout_seconds`.
- **Seam-crossing tools** (`hearth.restart`, `hearth.notify`) carry
  `seam_crossing: true` and the §7.2 label (`SEAM_CROSSING_LABEL`):
  they do not execute locally; an ADMIT submits work across the seam
  under S-3's rules and outcomes return under S-5's. A rendering that
  presented them as ordinary local tools would be a slice defect; a
  test asserts the label is present on exactly the seam set.
- **Filtering (§7.3) labels, never deletes.** A caller may supply a
  live-execution scope (`live_tools`) and policy-disabled names
  (`disabled_tools`): out-of-scope tools render with the
  shadow/escalate-only label, disabled tools with the disabled label,
  and the tool count never changes. No competence-based filtering
  exists in this module (§7.3 forbids it).
- `render_catalogue_text` renders the same record as deterministic
  text (registry order, no timestamps) — the form a later slice
  embeds in a gateway request (§6.5). Byte-stable for a byte-stable
  catalogue, asserted by test.
- The catalogue **digest** (§7.4) is `sha256:` over the record's
  canonical form without the digest field; `verify_catalogue_digest`
  recomputes it. A tampered description fails verification (test).

### 1.2 Perception assembly (§6) — `assemble_perception`

A pure function of `(journal, task_seq)`: it reads no clock, no
filesystem, and no state outside the journal rows and the registry, so
the same journal fold produces a **byte-identical** perception (§6.4) —
asserted by assembling twice and comparing whole records, and by
`verify_perception_digest` recomputing the digest from the record's own
sections + truncations (tampering fails, test).

Sources, in §6.1's precedence order:

1. **Task entry** — instruction verbatim, work root, operator, budgets
   in force, and submitter provenance where the task entry journaled
   it. No task entry at the seq → `ValueError`, fail closed (I8).
2. **Tool catalogue** — §1.1's rendering; its digest is carried on the
   perception as `catalogue_digest`.
3. **Prior cycles of this task** — per prior tool intent: tool, args
   (inline only when their canonical form is ≤ 1,024 bytes; larger
   args ride by `args_digest` with a label — §6.1 item 3 / §14.4),
   reason, gate verdict, the gate's **stated reasons** on failure
   (every failed check/sub-check's name + detail, from the journaled
   gate report), execution outcome and outcome seq.
4. **Tool results as tagged data** — each with its `source` tag,
   seqs, edit id, and provenance (`task_seq`, `edit_id`) intact, plus
   the data-never-instructions label (I4).
5. **Residency observations** — the latest journaled
   `hearth_residency` observation (state, model, node, poll time,
   observation-of-a-moment label) plus the transition walk
   (`from_state → to_state` with seq and poll time).
6. **Queue state** — `queue_summary` over `fold_queue` for the task,
   in the fold's own vocabulary.
7. **Budgets remaining** — cycles used (= this task's journaled tool
   intents), result bytes used (summed from journaled executions),
   and per-cap remaining numbers where the task entry journaled caps.

Cross-task isolation (§6.2) is structural and tested: intents and
outcomes are selected only when chained to this task
(`source_seqs` / `parents` naming it), so another task's rows never
enter the perception. Secrets, other tasks, gate policy internals,
and raw weights are never assembled — they are not among the sources
above.

### 1.3 Bounds and truncation (§6.3)

- The bound is a byte bound on the canonical perception (decision
  D-2's default), taken as an explicit argument
  (`max_perception_bytes`, default `DEFAULT_MAX_PERCEPTION_BYTES` =
  262,144). The policy key of §15.2 is P-3's to pin; P-1 hardcodes no
  policy behaviour.
- Truncation follows §6.3's order, pinned as data in
  `TRUNCATION_ORDER` and recorded per truncation: **oldest tool
  results first** (each replaced by a labelled stub keeping its tags,
  seqs, provenance, status, and original byte count), **then older
  residency transitions** (the latest observation never truncates),
  **then the oldest prior cycles' bulky fields** (args/execution/
  gate-reason details stubbed; seq, tool, verdict, args digest kept).
  Steps 2–3 are P-1's pinned completion of §6.3's stated order, which
  names only its first step and its never-truncate set.
- The **task entry and the catalogue never truncate**. If they alone
  exceed the bound, the perception carries a final `bound` truncation
  entry stating that fact — the bound is reported exceeded rather
  than silently violated.
- Every truncation is **labelled twice**: on the truncated element in
  the assembled perception, and in the perception's `truncations`
  list, which `build_attempt_record` carries into the would-be
  attempt record verbatim (§6.3: "carried into the attempt record").
  A test drives a bound of 1 byte and asserts the recorded order —
  tool results oldest-first, then residency, then prior cycles, then
  the bound label — and a moderate bound and asserts the labels reach
  the attempt record.

### 1.4 The would-be attempt record (§14.2) — `build_attempt_record`

Returns the record a later slice will journal before a gateway call:
task + cycle, proposer identity (`si.gaia.proposer` v1, transport
label), model identity and declaration digest — **each a labelled
absence in P-1**, never an invented value — the residency observation
seq relied on, perception digest + bytes, catalogue digest, the
truncation labels, and the precondition outcome. In P-1 that outcome
is always `not-evaluated`, and `request_digest` is absent with its
label: *no gateway request exists in P-1; no call was made and none
is claimed.* The builder journals nothing (assembly stays pure);
journaling is P-2/P-3's step, at the moment a call exists to record.

### 1.5 The proposer skeleton (§5.1) — `HearthProposer`

- `perceive(journal)` returns the assembled perception for the
  configured task as a one-element contribution, or `[]` when no task
  is configured (§5.1's allowed empty contribution).
- `propose(perceptions, ctx)` returns the **labelled not-evaluated
  outcome** (§16.2's shape): `outcome: "not-evaluated"`, the P-1
  label, `transport_called: false`, `intents: []`, the would-be
  attempt record, and the perception/catalogue digests and truncation
  labels carried from the assembled perception when one is supplied
  (a freshly rendered catalogue otherwise). It is deliberately **not**
  a batch of `Intent` objects: returning one would claim a proposal
  happened. This is a pinned P-1 deviation from §5.1's eventual
  return contract, recorded here per §5.1's own rule: P-2/P-3 grow
  this method into the batch return when a real proposal exists to
  return; until then the honest return is the labelled outcome.
- **No gateway call, by construction:** `propose` contains no
  transport invocation at all. The constructor accepts a transport
  only so P-3's wiring and P-1's zero-call evidence share one
  construction surface. The acceptance test injects a transport that
  records calls and raises on use, calls `propose` twice (with and
  without an assembled perception), and asserts `calls == []` both
  times.

---

## 2. What was NOT built (and remains labelled so)

- No gateway client use, no `/v1/models` discovery, no precondition
  check (§11.3) — P-3. No residency *freshness* pricing — P-3.
- No §8 output contract, no §9 parser, no attempt/response
  **journaling** — P-2.
- No §15.2 policy keys (`proposer_enabled` included) — P-3. P-1 adds
  no key to `si/gaia/policy.py` and none to any bundle policy.
- No prompt text (§6.6's content requirements are unimplemented as
  text; prompts are code and belong to the slice that sends them).
- No change to `si/gaia/task.py`, the gate, queue, drainer, executor,
  registry, policy, SSE, service, or seam modules — the diff is one
  new module, one new test file, and this doc.

---

## 3. Acceptance evidence (§19 P-1, criterion by criterion)

All tests are in `tests/test_gaia_proposer_p1.py` (20 tests), run by
the builder with the workspace venv; the reviewer's full-suite run
belongs to the dev-loop driver per standing builder discipline.

| §19 P-1 criterion | Evidence |
|---|---|
| Assembly from a fixture journal reproduces byte-stable perception; digests verify against a re-fold | `test_assembly_is_byte_stable_and_digest_verifies_on_refold` — two assemblies of one fixture journal compare equal as whole records; `verify_perception_digest` passes on the re-fold and fails on a tampered copy |
| Catalogue renders from the registry at call time; a registry mutation changes rendering + digest | `test_registry_mutation_changes_rendering_and_digest`; `test_catalogue_renders_every_registry_tool_verbatim`; `test_catalogue_digest_detects_tampering` |
| Truncation follows §6.3's order and is labelled in the perception and the would-be attempt record | `test_truncation_order_is_oldest_tool_results_first`; `test_truncation_labels_reach_the_would_be_attempt_record` |
| Seam-crossing tools render with §7.2's labels | `test_seam_crossing_tools_render_with_seam_labels` |
| No gateway call (P-1 scope) | `test_propose_returns_not_evaluated_and_never_invokes_transport`; `test_propose_without_perception_still_makes_no_call` |
| No existing behaviour changes (proposer unused by default, §15.2) | `test_proposer_is_unused_by_default` (the task loop neither imports nor names `HearthProposer`); no existing file is modified by this slice; affected subsets below stay green |

Test runs (builder, `~/workspace/si2-venv`, Python 3.12.3, CPU only):

- New: `tests/test_gaia_proposer_p1.py` — **20 passed**.
- Affected subsets (combined run): `test_gaia_b1.py`,
  `test_gaia_s2b_hearth_registry.py`, `test_gaia_b4.py`,
  `test_gaia_q1_queue.py`, `test_gaia_q1_task.py`, `test_gaia_b3.py`
  together with the new file — **149 passed** (129 pre-existing +
  20 new), 0 failed.
- The full suite was **not** run by the builder (standing discipline:
  one full suite per PR belongs to the reviewer).

---

## 4. Decisions pinned by this slice (SI-63 §20)

- **D-2** — perception size unit is bytes at assembly (the bound and
  the recorded `bytes` are canonical-JSON byte counts; no token
  estimate is presented as metering).
- **D-7 (P-1 part)** — module name `si/gaia/proposer.py`; record
  kinds pinned: perception `si.gaia.proposer.perception`, catalogue
  `si.gaia.proposer.catalogue`, outcome `si.gaia.proposer.proposal_outcome`,
  and the attempt record kind `gaia_proposal_attempt` (SI-63
  Appendix A's illustrative name, adopted). Journal *writing* of
  these kinds remains P-2/P-3.
- **§6.3 completion** — truncation steps 2–3 (residency history, then
  prior cycles) pinned as in §1.3 above; step 1 and the
  never-truncate set are §6.3 verbatim.
- **§5.1 deviation** — `propose` returns the labelled not-evaluated
  outcome dict in P-1 rather than a batch; recorded in §1.5.

---

## 5. Next slices (unchanged from SI-63 §19)

P-2 (output contract + parser, replay only), P-3 (gateway wiring
under preconditions; `proposer_enabled` defaulting off), P-4 (first
live proposer — only on Mark-side preconditions: a real Hearth node,
a real base `Resident`, an explicitly enabling bundle policy), P-5
(measurement + hardening). Until P-4 lands its own evidence, the
thinking proposer remains UNBUILT beyond P-1's assembly/rendering and
the real Hearth path remains UNMEASURED.

— End of SI-64 (P-1). —
