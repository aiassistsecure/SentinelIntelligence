# SI-42 — Gaia B-1: Tool-Intent Format, Full Registry, and Gate
Checks in Shadow Mode (SI-41 RFC §15 B-1, §16)

> **Status: BUILT, in review (PR opened by the B-1 builder; merging
> is the dev-loop driver's, on a later run).** This slice implements
> SI-41 §15's B-1 exactly: the `tool` intent kind, the complete v1
> registry as data, the §5 gate mapping, the §6.2 policy keys, and
> journaling of tool intents, verdicts, and shadow outcomes.
>
> **Shadow discipline (RFC §15 B-1, SI-16 precedent):** B-1 has
> **no executor**. Every tool intent that passes the decisive gate
> checks ESCALATEs; every report and every journaled outcome records
> execution as `not-evaluated — shadow mode`. Nothing in this slice
> runs a tool, writes a file on a mind's behalf, or spawns a
> process, and the test suite asserts exactly that across the whole
> scripted corpus (§4 below). Claiming an execution path before one
> is built and measured would be fabrication.
>
> **Gaia's home:** the new code lives in the `si/gaia/` subpackage
> (Mark, 2026-10-29): `registry.py` (the §8 surface as data),
> `policy.py` (§6.2 resolution), `context.py` (the gate-context
> contract), `corpus.py` (the scripted verdict corpus),
> `shadow.py` (task-fixture + shadow journaling + the `si why`
> walk), plus the gate's tool branch in `si/gate.py`, the loop's
> shadow-outcome journaling in `si/loop.py`, the §6.2 keys in
> `si/bundle.py`'s `DEFAULT_POLICY`, and the `si why` tool walk in
> `si/cli.py`. This slice also adds the repo-root Apache-2.0
> `LICENSE` file (the repo was previously unlicensed; Mark,
> 2026-10-29).
>
> **Terminology:** machine learning infrastructure throughout.
> CPU only; no GPU, no cloud, no spend, no downloads. No bytes are
> duplicated by this slice (Mark's design rule): the registry is
> code, the journal records are the loop's own records.

---

## 1. What was built

### 1.1 The tool intent (RFC §4, §16.1)

`Intent` gains two fields for `kind == "tool"` only — `tool` (the
registered tool name) and `args` (the declared args object, the
RFC §4.1 JSON example's `"args"`) — and `to_dict()` carries both,
so a tool intent round-trips through the intent machinery exactly
as the other kinds do. `"tool"` is added to the gate's known-kind
set; genuinely unknown kinds are still rejected as malformed, and
a tool proposal dressed as a `delta` gets the delta checks (and
REJECTs) — the §5.3 not-applicable mapping is kind-scoped and
cannot be reached from another kind (both directions tested).

### 1.2 The registry (RFC §6.1, §8; §16.2)

`si/gaia/registry.py` defines all **12** v1 tools as `ToolSpec`
records — name, version, description, args schema, result schema,
consequence class, reversibility, default timeout,
`introduced_by` — in RFC §8 order:

| § | Tools | Classes |
|---|---|---|
| §8.1 | `fs_list`, `fs_read`, `fs_search` | read / reversible |
| §8.1 | `fs_write`, `fs_edit`, `fs_mkdir` | write / compensable |
| §8.2 | `journal_query`, `bundle_status`, `probe_run` | read / reversible |
| §8.3 | `shell_exec`, `code_exec`, `test_run` | execute / one-way |

A test asserts the registry's name set equals the RFC §8 table
(the table is pinned as data in `RFC_V1_TOOL_NAMES`, so the test
compares the registry against the RFC, not against itself), and
that every spec is fully specified. Schemas use exactly the §6.1
subset (`type`, `required`, `properties`, `items`,
`minimum`/`maximum`, `minLength`/`maxLength`, `enum`) plus one
recorded extension — `type` may be a *list* (a result field that
is a number or null) — validated by the in-repo validator in
`registry.py`, which itemises every violation with its field
path. Objects reject undeclared properties: an arg the registry
did not declare is an arg the gate cannot price. §8.4 holds: no
bundle-mutation tool, no tool-registration tool, no SI-repo tool,
no network-specific tool exists in the registry.

Deviations/open fields the RFC left open, pinned by this slice
(said here, per the RFC's own honesty rules, rather than silently):

- **`introduced_by`** records registry introduction (`SI-41/B-1`),
  not the execution slice; §6.1's example value (`SI-41/B-2` for
  `fs_write`) described the then-planned execution landing.
- **`shell_exec` command-length cap:** §5.2 names a "policy
  command-length cap" but §6.2 defines no key for it. B-1 pins the
  cap at the schema (`maxLength` 8192 on `command`) instead of
  inventing a policy key; if Mark wants it as policy, that is a
  §6.2 amendment for a later slice.
- **`code_exec` source-XOR-path:** the §6.1 subset cannot express
  "exactly one of `source`/`path`", so the gate prices it as an
  additional itemised args rule (in `validate_args` and as the
  `command_bounds` sub-check), stated in the spec's description.
- **Result-schema fields** beyond §8's summaries (e.g.
  `journal_query`'s `count`/`truncated`, `test_run`'s `parse` /
  `output`, `probe_run`'s `label`/`measured`/`detail`) are pinned
  from the RFC's own §8 notes (the suite's numbers as reported;
  `parse: failed` on parse failure; the probe harness's
  measured/not-evaluated labels) — summaries made field-exact,
  nothing added that §8 does not describe.

### 1.3 The gate mapping (RFC §5; §16.3, §16.4)

`Gate.evaluate` gains a tool branch with the five checks in order:

1. **Constitution** — unchanged mechanism: the reason *and the
   full serialised intent, tool name and args included*, are
   scanned against the blocklist. A blocklisted shell command
   REJECTs here (corpus row `blocklisted-command`).
2. **Safety bounds → tool safety** — a sub-check suite journaled
   with the report, SI-9b-style: `tool_registered`,
   `allowed_tools`, `args_schema` (with one sub-sub-check per
   schema violation), `effective_class`, `execution_scope`,
   `class_policy`, `command_bounds`, `budgets`. Unknown tool,
   not-allowlisted tool, schema violations, a `deny` class rule,
   and exhausted budgets all fail this check, itemised.
3. **Identity continuity** — recorded **not-applicable** per
   §5.3: `passed=True`, `measured=None`, a distinct `label`
   field carrying `"not-applicable"`, and the §5.3 detail ("no
   bundle state change is proposed by this intent; the mind the
   probes measure is the mind that remains"). The label exists
   only on tool reports' checks 3/4 — every other kind's
   journaled check form is byte-for-byte what it was.
4. **Capability floor** — same not-applicable record as check 3.
5. **Provenance completeness** — parents/source_seqs non-empty
   (unchanged requirement), plus, when the gate context names
   the current task (`tool_task_seq`), the cited seqs must
   include the task seq and the parents must reference the task
   entry as `task@<seq>` (§5.2's chain-to-task rule). Outside a
   task, cited causes are priced exactly as a delta's are.

**Per-intent classification (§7.2)** is computed by
`Gate.classify_tool_intent`: the registry class is the floor; any
path-bearing arg (or a `shell_exec`/`code_exec` `cwd`) whose
*resolved* absolute path (symlinks resolved first, §9.1) falls
outside the work root raises the intent to `external`. The work
root resolves from the context, then the bundle dir, then the
policy's `tool_work_root`, then the working directory.

**Verdict algebra (§5.4 as shadowed by §15):** any decisive
failure REJECTs; a passing intent whose effective class rule is
`escalate` ESCALATEs by class policy; every other passing intent
ESCALATEs by shadow discipline. No tool intent ADMITs in B-1 —
an ADMIT would claim an execution path that does not exist.
The two ESCALATE routes are distinguishable on the record: the
`class_policy` sub-check states the class rule that applied,
and every tool `GateReport` carries
`execution: "not-evaluated — shadow mode"` (a tool-only field;
other kinds' report form is unchanged). Delta and
transformer-delta behaviour is untouched — the existing gate
suites pass unmodified (§5 below).

**Gate-context contract:** the tool branch reads only optional
context attributes (`tool_work_root`, `tool_task_seq`,
`tool_cycles_used`, `tool_result_bytes_used`; `bundle_dir` as a
scope fallback) and never calls the demo BundleContext methods,
so any context — loop, probe, or scripted — can gate a tool
intent. The RFC pins the budget *caps* (§6.2) and the budgets
themselves (§10.3) but leaves the runtime spend bookkeeping to
the B-3 task loop; `si/gaia/context.py` pins this attribute
shape as the gate-facing form of that bookkeeping (an open
field, recorded here).

### 1.4 Policy resolution (RFC §6.2; §16.5)

`si/gaia/policy.py::resolve_tool_policy` resolves the seven
§6.2 keys fail-closed: absent → the RFC defaults (which
`DEFAULT_POLICY` now also ships, so fresh bundles carry them);
malformed `tool_class_policy` entries → `deny` per class (a
wholly non-dict class policy denies every class); malformed
`allowed_tools` → `[]`; malformed numeric caps → 0; malformed
`tool_work_root` → `None`. The shipped default `allowed_tools`
is empty and admits nothing at the gate (tested).

### 1.5 Journaling and `si why` (RFC §4.5–4.6, §10.1; §16.7)

All journaling uses the existing methods in the existing
collections — no new collection, no schema change:

- `si/gaia/shadow.py::record_task_fixture` journals a scripted
  operator task entry via `record_experience` (kind `task`,
  `source: "operator"`, instruction, work root, budgets) and
  returns `{seq, ref: "task@<seq>", row_id}`. It is a **fixture**:
  the real task surface (`si task`) is B-3 and is not built here.
- `journal_tool_intent` gates one intent and journals the
  self-edit (`record_self_edit`, kind `tool`, full gate report),
  its result (`escalated`/`rejected` — never `materialized`), the
  provenance edges (`record_provenance`), and the shadow outcome
  (`record_experience`, kind `tool_result`, `source: tool:<name>`,
  carrying the §4.4-shaped execution record with status/execution
  `not-evaluated — shadow mode` and no fabricated measured
  fields — no exit code, hash, or wall time exists to record).
- The self-write loop's own tool branch (both the demo and
  miniature paths' REJECT/ESCALATE journaling) records the same
  shadow outcome through the same writer, so a mind proposing a
  tool intent through `run_cycle` leaves the identical record.
- `si why <edit-id>` (a tool self-edit id, e.g. `se_0003`) walks
  the intent's provenance — self-edit, gate checks, provenance
  edges, cited source seqs resolved across all five collections —
  to its task entry and prints it; a chain reaching no task is
  reported as none, never invented (`walk_tool_provenance`).

### 1.6 LICENSE

The repo-root `LICENSE` is the standard Apache-2.0 text
(Mark's 2026-10-29 direction: Apache-2.0 for the whole
SentinelIntelligence repo, previously unlicensed).

---

## 2. §16 acceptance criteria, one by one

| # | Criterion | Where it is met |
|---|---|---|
| 1 | `Intent(kind="tool", ...)` round-trips through `to_dict`; malformed-kind handling still rejects unknown kinds | `si/gate.py` (`tool`/`args` fields); tests `test_tool_intent_round_trips_through_to_dict`, `test_unknown_kind_still_rejected_as_malformed`, `test_tool_kind_cannot_be_redressed_as_delta` |
| 2 | Registry defines every §8 tool with args schema, result schema, class, reversibility; a test asserts the name set equals the §8 table | `si/gaia/registry.py`; tests `test_registry_name_set_equals_rfc_section8_table`, `test_every_tool_is_fully_specified`, `test_registry_classes_and_reversibility_match_rfc` |
| 3 | Verdict corpus produces §5.4's verdicts (unknown-tool, schema with itemised sub-checks, blocklisted check-1, missing-reason, provenance REJECTs; outside-scope → `external`; class-policy ESCALATE) | `si/gaia/corpus.py` + the §3 table below; tests `test_corpus_verdicts`, `test_schema_violation_subchecks_are_itemised`, `test_blocklisted_command_rejects_at_check1`, `test_outside_scope_prices_to_external`, `test_class_policy_escalate_is_distinguishable_from_shadow`, `test_budget_exhausted_names_the_budget` |
| 4 | Checks 3/4 carry the §5.3 not-applicable label with `measured=None`; delta / transformer-delta behaviour byte-for-byte unchanged | `si/gate.py` tool branch (`label="not-applicable"`, tool-only field); test `test_corpus_check_shape_for_passing_rows`; existing gate suites pass unmodified (§5) |
| 5 | Policy resolution: absent → §6.2 defaults; malformed class entries → `deny`; `allowed_tools` default admits nothing | `si/gaia/policy.py`; tests `test_policy_absent_keys_resolve_to_rfc_defaults`, `test_policy_malformed_entries_fail_closed`, `test_allowed_tools_default_admits_nothing_at_the_gate` |
| 6 | Shadow guarantee: executions = 0 across the corpus (filesystem + process-spawn sentinel); every report records execution as not-evaluated/shadow | Test `test_shadow_guarantee_zero_executions_across_corpus` (Popen/run/os.system sentinel + work-root tree diff), `test_no_tool_intent_ever_admits_in_b1`, and the execution assertion inside `test_corpus_verdicts` |
| 7 | Journal via existing methods/collections; `si why` walks a tool intent to its task entry (scripted fixture; `si task` is B-3, not built) | `si/gaia/shadow.py`, `si/cli.py` (`_cmd_why_tool`); tests `test_journal_and_why_walk_to_task`, `test_loop_journals_tool_shadow_outcome` |
| 8 | Full suite green in a workspace venv, baseline stated honestly | §5 below (numbers from the builder's own run on the PR head; the driver re-runs independently) |

---

## 3. The verdict corpus (scripted; executed by the tests)

Corpus policy: every registered tool allowlisted, RFC §6.2 class
defaults, scripted task seq 7 with parent ref `task@7`, unless a
row overrides. Work root and the outside path are the test's temp
directories. Every row's verdict below is the gate's actual output
as asserted by `test_corpus_verdicts`.

| Case | Tool | Effective class | Verdict | Failing check | Notes |
|---|---|---|---|---|---|
| `valid-fs_list` | `fs_list` | read | **ESCALATE** | — | Valid in-scope read; §5.4 would ADMIT, B-1 shadow ESCALATEs. |
| `valid-fs_read` | `fs_read` | read | **ESCALATE** | — | Valid in-scope read; shadow ESCALATE. |
| `valid-fs_search` | `fs_search` | read | **ESCALATE** | — | Valid in-scope search; shadow ESCALATE. |
| `valid-journal_query` | `journal_query` | read | **ESCALATE** | — | Valid house read; shadow ESCALATE. |
| `valid-bundle_status` | `bundle_status` | read | **ESCALATE** | — | Valid house read; shadow ESCALATE. |
| `valid-probe_run` | `probe_run` | read | **ESCALATE** | — | Valid house read; shadow ESCALATE. |
| `valid-fs_write` | `fs_write` | write | **ESCALATE** | — | Valid in-scope write; shadow ESCALATE. The zero-execution test asserts this file is never created. |
| `valid-fs_edit` | `fs_edit` | write | **ESCALATE** | — | Valid in-scope edit; shadow ESCALATE. |
| `valid-fs_mkdir` | `fs_mkdir` | write | **ESCALATE** | — | Valid in-scope mkdir; shadow ESCALATE. |
| `valid-shell_exec` | `shell_exec` | execute | **ESCALATE** | — | Valid in-scope command; shadow ESCALATE. |
| `valid-code_exec` | `code_exec` | execute | **ESCALATE** | — | Valid code execution; shadow ESCALATE. |
| `valid-test_run` | `test_run` | execute | **ESCALATE** | — | Valid test run; shadow ESCALATE. |
| `class-policy-escalate-write` | `fs_write` | write | **ESCALATE** | — | All decisive checks pass; class policy sets write=escalate (§5.4 class-policy route). |
| `outside-scope-fs_read` | `fs_read` | external | **ESCALATE** | — | Path resolves outside the work root: prices as external (§7.2); the default external rule escalates it. |
| `outside-scope-shell-cwd` | `shell_exec` | external | **ESCALATE** | — | An outside-scope declared cwd raises shell_exec from execute to external (§7.2). |
| `unknown-tool` | `fs_delete_everything` | — (unknown tool) | **REJECT** | safety_bounds | Unregistered name: unknown-tool REJECT at check 2 (§5.4). |
| `schema-violation-fs_write` | `fs_write` | write | **REJECT** | safety_bounds | Missing required `content`, wrong-typed `path`, undeclared arg: itemised sub-checks (§5.2). |
| `schema-violation-shell-empty` | `shell_exec` | execute | **REJECT** | safety_bounds | Empty command violates minLength 1. |
| `schema-violation-code_exec-both` | `code_exec` | execute | **REJECT** | safety_bounds | Both source and path: the exactly-one rule the subset cannot express (§1.2). |
| `blocklisted-command` | `shell_exec` | execute | **REJECT** | constitution | The command contains a blocklisted pattern: REJECT at check 1 (§5.2). |
| `missing-reason` | `fs_list` | read | **REJECT** | wellformed | No reason: malformed, REJECT before the checks (§4.1). |
| `provenance-incomplete` | `fs_list` | read | **REJECT** | provenance_completeness | No parents, no source seqs (§5.4). |
| `provenance-missing-task-link` | `fs_list` | read | **REJECT** | provenance_completeness | Cited seqs omit the task seq: the chain does not reach the task (§5.2). |
| `not-allowlisted` | `fs_list` | read | **REJECT** | safety_bounds | Removed from allowed_tools: existence is not permission (§6.3). |
| `class-policy-deny-read` | `fs_list` | read | **REJECT** | safety_bounds | Class policy read=deny: REJECT at check 2 (§7.3). |
| `budget-exhausted-cycles` | `fs_list` | read | **REJECT** | safety_bounds | 16 used of `max_tool_cycles_per_task` 16: budget named (§10.3). |
| `budget-exhausted-results` | `fs_list` | read | **REJECT** | safety_bounds | 1048576 used of `tool_max_task_result_bytes`: budget named (§10.3). |

A symlink inside the work root pointing outside it also prices
`external` (resolved-target classification, §9.1) —
`test_symlink_out_of_scope_prices_external`.

---

## 4. The shadow guarantee, as evidenced

Across all 27 corpus rows, evaluated against a real temp work
root: the filesystem tree of the work root is byte-identical
before and after (the valid `fs_write` row's target is never
created), and a process-spawn sentinel monkeypatched over
`subprocess.Popen`, `subprocess.run`, and `os.system` records
**zero** spawns. No tool intent in the corpus — or anywhere in
B-1 — returns ADMIT. These are test assertions, re-run by the
driver on review, not prose claims.

---

## 5. Test results

Baseline on main at this slice's base (`cee5a76`, PR #46 merged):
**419 passed, 2 skipped**. This slice adds
`tests/test_gaia_b1.py` (29 tests). Full-suite numbers from the
builder's own run on the PR head are stated in the PR body; the
existing gate/loop/journal suites are unmodified and pass as
part of that run.

---

## 6. Explicitly NOT built (do not read this slice as claiming them)

- **No executor and no live execution of any tool** — B-2 (fs /
  house tools) and B-3 (execution tools) per RFC §15.
- **No `si task`** and no task loop, chaining, or stop rules —
  B-3. B-1 journals a scripted task *fixture* only.
- **No Hearth fork, no Hearth bridge, no SSE stream** — the fork
  is its own step on Mark's direction; B-4 is the Hearth
  interface slice. (Gaia's additions noted by Mark on 2026-10-29
  — operator leash, SSE replay from journal seqs, powers travelling
  with the bundle, earned autonomy — are later-slice design inputs,
  not B-1 deliverables; nothing here claims them.)
- **No A-2 / A-3 / A-4** and no real-base agency — unapproved,
  untouched (§14.2). Real-base tool-use claims remain honestly
  scoped: everything here is exercised with scripted intents
  against the gate and journal, which is what B-1's evidence is.
- **No policy-default changes beyond §6.2's own defaults**, no
  constitution change, no new journal collection, no NEDB schema
  change.

---

*The gate is not the price of the powers; the gate is what makes
them SI's. B-1 gives the loop a way to* want *to act — priced,
journaled, and, until an executor exists to be measured, honestly
not executed.*
