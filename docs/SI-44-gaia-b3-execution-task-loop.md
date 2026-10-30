# SI-44 — Gaia B-3: The Execution Tools and the Task Loop
(SI-41 RFC §15 B-3)

> **Status: BUILT, in review (PR opened by the B-3 builder; merging
> is the dev-loop driver's, on a later run).** This slice implements
> SI-41 §15's B-3: the execute-class tools (`shell_exec`,
> `code_exec`, `test_run`) live through the same executor B-2 built,
> with the executor's resource bounds (timeout kill, per-stream
> caps, environment scrubbing by explicit allowlist); and the task
> loop of RFC §10 — real operator task entries, chaining, budgets,
> the §10.4 stop rules, and the `si task` CLI — plus the §12.5(d)
> portability check (bundle moved, trail queried on the far side).
>
> **The live-scope boundary, stated exactly.** LIVE (gate-ADMITs
> and the executor runs them, on miniature bundles): everything
> B-2 made live (`fs_list`, `fs_read`, `fs_search`, `fs_write`,
> `fs_edit`, `fs_mkdir`, `journal_query`, `bundle_status`,
> `probe_run`) plus `shell_exec`, `code_exec`, `test_run` — the
> registry's `LIVE_TOOLS_B3` set, which is `LIVE_TOOLS_B2` ∪ the
> execute class; `LIVE_TOOLS_B2` itself is unchanged. STILL
> ESCALATE (nothing executes): every intent whose effective class
> prices to `external` (a path or cwd resolving outside the work
> root), every tool intent on a demo (non-miniature) bundle, and
> every tool intent on any non-live gate context. Nothing executes
> without a gate ADMIT on the journaled record — the executor
> re-checks the verdict itself, exactly as in B-2.
>
> **Terminology:** machine learning infrastructure throughout.
> CPU only; no GPU, no cloud, no spend, no downloads. No bytes are
> duplicated: oversized results and pre-images are content-addressed
> in the bundle's existing blob store and referenced by hash
> (Mark's design rule).

---

## 1. What was built

### 1.1 The gate: the live set grows to the execute class (§5, §15 B-3)

`Gate._evaluate_tool`'s live branch now prices against
`LIVE_TOOLS_B3`:

- **Non-live contexts**: B-1 behaviour **byte-for-byte** — every
  passing intent ESCALATEs, every tool report records
  `not-evaluated — shadow mode`. The B-1 suite passes unmodified;
  the demo loop path is not flipped.
- **Live contexts** (only the miniature loop's `ProbeContext`
  carries the flag): decisive checks unchanged (constitution,
  tool safety with its itemised sub-checks, provenance with the
  chain-to-task rule; checks 3/4 still the §5.3 not-applicable
  label). Then:
  - decisive failure → **REJECT** (execute-class intents carry the
    B-3 rejected label, `not executed — rejected at gate
    (SI-44/B-3)`; read/write keep B-2's);
  - tool in `LIVE_TOOLS_B3`, effective class `read`/`write`/
    `execute`, class rule `admit` → **ADMIT**. Read/write ADMITs
    keep B-2's authorisation label — B-2 is the slice that
    authorised them and its suite asserts that label; execute-class
    ADMITs carry the new label `live execution authorised —
    SI-44/B-3 (miniature-scoped, read/write/execute classes)`
    (`TOOL_B3_LIVE_EXECUTION` in `si/gate.py`);
  - class rule `escalate` → **ESCALATE** by class policy (the
    §7.3 dial; execute-class escalations carry the B-3 escalated
    label);
  - anything else passing — in B-3 that is exactly an
    external-effective intent → **ESCALATE**,
    `not-evaluated — outside SI-44/B-3 live scope …`
    (`TOOL_B3_NOT_LIVE_EXECUTION`): external still holds in B-3
    (§2).

### 1.2 The executor: process tools under the B-3 bounds (§9, §11)

`si/gaia/executor.py` keeps B-2's discipline entire — journaled
intent only (§9.2), args re-validated at EXECUTE, effective class
re-priced (a change is a TOCTOU refusal), refusals journaled — and
its live scope grows from read/write to read/write/execute on the
same miniature scope. The three new handlers are real:

- **`shell_exec`** runs `["/bin/sh", "-c", command]` with
  `cwd` = the intent's declared `cwd` resolved in scope (default:
  the work root) and an effective timeout of
  `min(declared timeout_seconds or the registry default,
  policy tool_timeout_seconds)`.
- **`code_exec`** runs the runtime's own interpreter
  (`sys.executable`): `["python", "-c", source]` or
  `["python", resolved in-scope path]` — python only, as the
  registry's schema pins.
- **`test_run`** runs `[sys.executable, "-m", "pytest",
  resolved target, "-q"]` with the work root as cwd and parses
  pytest's own summary line for passed/failed/skipped **as
  reported** (§2 pins the parse rule).

Process discipline (RFC §15 B-3, §9.3, §17.6):

- **Environment scrubbing.** The child environment is an explicit
  allowlist — `CHILD_ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL",
  "LC_CTYPE", "TZ")`, taken from the runtime's environment only
  where present — never the runtime's own environment, so no
  credential the runtime holds leaks into a command's environment
  by default. The B-3 suite proves it: a sentinel variable set in
  the parent environment is invisible to the child's `env`.
- **Stream caps.** stdout/stderr (and `test_run`'s combined
  output) are capped per stream at the policy's
  `tool_max_result_bytes`; truncation is flagged on the execution
  record, and `_bound_result` — extended in B-3 to bound the
  `stdout`/`stderr`/`output` string fields alongside B-2's
  `content` — blob-stores any still-oversized full result exactly
  as B-2 does.
- **Timeout.** The child runs in its own session; on timeout the
  whole process group is killed, partial output is kept and the
  record is labelled `partial: true` (the tools' result schemas
  carry no such field, so the label lives on the record), and
  the status is `timed-out` (§4.4).
- **Exit codes.** The record-level `exit_code` is now the
  process's **real** exit code for process tools (B-2 tools spawn
  no process and keep `null`, as B-2 recorded). A process that
  ran to completion is status `completed` **even on a non-zero
  exit** — that is data (§11.1.3), recorded, returned to the mind
  as an observation, never dressed as an executor failure.
  Handler errors remain status `failed`.
- **Post-state (§11.1).** Process tools record the observed
  `{exit_code, timed_out, cwd}`; `test_run`'s post-state is the
  suite's own parsed summary. Result conformance is the existing
  `validate_result` against the registry's result schemas.
- Execution records for execute-class tools carry the B-3 label
  `live — executed by the SI-44/B-3 executor (miniature-scoped)`;
  B-2 tools' records keep the B-2 label (B-2 made them live).
  One-way actions say so on the record: `shell_exec`/`code_exec`/
  `test_run` records carry `reversibility: "one-way"` from the
  registry (RFC §15 B-3's evidence point).

### 1.3 The task loop (`si/gaia/task.py`, RFC §10)

- **`record_task_entry(journal, instruction, work_root, budgets,
  operator)`** journals a REAL operator task entry — the §10.1
  shape (kind `task`, `experiences`, `source: "operator"`,
  instruction, work root, budgets in force) through the journal's
  existing `record_experience`. It is the shape of B-1's
  `record_task_fixture` **without the fixture label**: an operator
  actually opened this task.
- **`run_task(bundle_dir, proposer, instruction, work_root=None,
  cycles=None, operator="operator", policy=None)`** opens the
  task BEFORE any cycle, then runs serial cycles (§10.2): per
  cycle it sets the context's `tool_task_seq` / `tool_cycles_used`
  / `tool_result_bytes_used` and chains one intent through
  `execute_tool_cycle` (gate → journal → execute → outcome).
  Budgets come from the policy's §6.2 keys
  (`max_tool_cycles_per_task`, `tool_max_task_result_bytes`);
  the `cycles` argument caps total tool cycles at the minimum of
  the two. Perceptions handed to a mind proposer are the task
  entry plus the tool_result observations so far, provenance
  tags intact (§4.5 — observations are data, never instructions),
  plus whatever the mind's own `perceive(journal)` returns.
  Before returning, the manifest checkpoint is re-anchored and
  the manifest rewritten, as `run_tool_intent` does, so the
  bundle stays loadable with the whole trail in its journal.
- **The proposer** has the two honest forms of §14.1: a mind-like
  object (`perceive`/`propose`), or an iterable of `Intent`
  objects — the scripted proposer, the miniature's evidence
  vehicle. Scripted intents whose `parents`/`source_seqs` are
  empty get the §10.2 task chain filled by the runner
  (`parents = [task_ref] + [prior edit]`, `source_seqs =
  [task_seq] + [prior outcome seq]`), so a script need not
  predict journal seqs; chains a script authors itself are left
  exactly as authored and priced by the gate as authored.
- **Stop rules (§10.4)**, each journaled as a `task_stop`
  experience with its cause before control returns:
  1. `budget-exhausted` — either budget reached (checked before
     proposing; a gate REJECT whose budgets sub-check names an
     exhaustion also maps here);
  2. `escalated` — any ESCALATE verdict halts the task at the
     escalation;
  3. `reject-streak` — two consecutive REJECTs (a non-REJECT
     resets the streak; one REJECT is information the proposer
     may route around, and the suite proves both halves);
  4. `integrity-failure` — an executor refusal whose detail is a
     journaled-args re-validation failure or a class-changing
     TOCTOU halts immediately, no streak arithmetic;
  5. `mind-reported-complete` — a memory intent whose record is
     `{"kind": "task_complete", ...}`, gated as an ordinary
     memory intent (gate + `record_self_edit` + the memory
     experience, mirroring `si/loop.py`'s memory handling);
     ADMIT closes the task under this label — the mind *reported*
     completion; the operator judges whether it is true.
- **No rule 6.** A scripted proposer running out of intents (or
  a mind returning None) stops the task as `proposer-exhausted`,
  explicitly labelled in the stop detail as NOT one of §10.4's
  five rules: an operator-visible incomplete task, never dressed
  as a §10.4 stop.
- `run_task` returns `{task_seq, task_ref, stop_reason,
  stop_detail, cycles_used, result_bytes_used, steps, head}`;
  each step is `{edit_id, verdict, tool, effective_class,
  execution_status}`.

### 1.4 The CLI: `si task`

```
si task <bundle> --instruction "..." [--root DIR] [--cycles N]
        [--script FILE] [--operator NAME]
```

`--script` is a JSON file holding an array of intent dicts
(`{"kind": "tool", "tool": ..., "args": {...}, "reason": ...}`,
or a `{"kind": "memory", "record": {"kind": "task_complete",
...}}` completion report); `parents`/`source_seqs` may be
omitted — the runner fills the task chain (§1.3). The CLI prints
the task seq/ref, each step, the stop reason, cycles/bytes used,
and the head. Without `--script` the CLI exits 2 with an honest
message (§2): B-3's `si task` drives a scripted proposer, because
no trained mind proposes tool intents yet (§14) — the CLI does
not invent a default mind.

Contexts: miniature bundles gate tasks through the live
`ProbeContext`; demo bundles gate through a non-live
`ToolContext`, so a task on a demo bundle ESCALATEs its first
step and halts `escalated` with the B-1 shadow record — honest:
no demo-bundle execution exists in B-3 either (suite-proven).

---

## 2. Pinned deviations and open fields (stated, never silent)

- **Executor location (carried forward from SI-43).** RFC §4.3
  named `si/tools.py`; the executor lives in `si/gaia/executor.py`
  per Mark's 2026-10-29 directive placing all Gaia code in the
  `si/gaia/` subpackage. The task loop likewise lives in
  `si/gaia/task.py`. The RFC's component design is followed; only
  the module paths follow the later directive.
- **External held at ESCALATE in B-3 (carried forward from
  SI-43).** §7.3 makes `external` admittable by policy in
  principle; B-3's live scope remains work-root-contained
  execution, so an external-effective intent ESCALATEs in this
  slice even under `external: admit`. A slice-scoped hold like
  B-2's — stated on the gate label and here — for a later slice
  to lift on its own evidence, not a change to §7.3.
- **`test_run`'s runner is pinned to pytest.** The registry
  (§8.3) says "run a test target and report the suite's own
  summary"; the v1 runner is pinned to
  `[sys.executable, "-m", "pytest", <target>, "-q"]` — the
  runtime's own interpreter running the suite this project is
  itself tested with. The parse rule: passed/failed/skipped are
  read from pytest's summary line **as reported** (a category
  pytest does not report is 0 by pytest's own convention); a run
  whose output carries no recognisable count at all — including
  an error-only summary the result schema cannot represent —
  returns `parse: "failed"`, counts 0, and the raw output.
  Counts are never invented.
- **`exit_code: -1` is the timeout sentinel.** A killed process
  produced no exit code; the record and the result carry `-1`,
  documented here and at `TIMEOUT_EXIT_CODE` in the executor as
  the no-exit sentinel — never a claim about a real exit code.
  Processes that exit record their real code, including
  non-zero codes as `completed`.
- **`--script` pinning of the `si task` surface.** RFC §10.1's
  proposed surface is `si task <bundle> --instruction "..."
  [--root <dir>] [--cycles <n>]` with no script argument; the
  miniature's proposer is scripted per §14.1, so B-3 pins the
  proposer input as `--script FILE` (JSON intent dicts). A mined
  proposer arrives with the A-line, not by inventing one here.
- **The two B-2 test changes (the only ones).** B-3 supersedes
  exactly two assertions in `tests/test_gaia_b2.py`, both because
  the behaviour they asserted was B-2's slice boundary and B-3 is
  the slice that moves it:
  1. `test_live_context_execute_tools_escalate_to_b3` →
     `test_live_context_execute_tools_admit_in_b3`: execute-class
     intents on a live context now ADMIT with
     `TOOL_B3_LIVE_EXECUTION` (they ESCALATEd in B-2 *because*
     they were B-3's).
  2. In `test_executor_refuses_non_admitted_and_b3_journaled_intents`:
     the fabricated-ADMIT `shell_exec` case — refused in B-2 as
     out-of-slice — now **executes** (`echo hi`, completed,
     exit_code 0); the refusal case is replaced with a still-valid
     one, a fabricated ADMIT for an **unknown** tool, refused as
     not a registered tool. The test file's module docstring was
     updated to point here.
  Every other B-1/B-2 test is unmodified and passes.
- **Read/write ADMITs keep the B-2 gate label** (§1.1): the
  authorising slice for read/write is B-2; rebadging its ADMITs
  would break B-2's asserted record for no honesty gain. The
  B-3 label text names the full B-3 live scope
  (read/write/execute classes) because that is the scope B-3's
  *execute-class* ADMITs are issued under.
- **Completion-intent `collection` default.** A `task_complete`
  memory intent whose `collection` is unset is filled to
  `"experiences"` by the task runner before gating — the only
  collection a memory intent may write, and the value the loop's
  minds set explicitly. A completion report is an experience by
  construction; REJECTing it for an undeclared default would
  price paperwork, not the act.
- **Budget-REJECT mapping is detail-checked.** The gate journals
  a failed `budgets` sub-check as a "not evaluated" placeholder
  for unknown tools; the task loop maps a REJECT to
  `budget-exhausted` only when the sub-check's detail names an
  exhaustion (or a malformed spend, which fails closed the same
  way) — never on the placeholder.

---

## 3. Evidence (all measured by the builder on the PR head)

- **Process tools, real runs**: `shell_exec` of `echo` +
  `env` completes with exit_code 0 and the output captured;
  `exit 3` completes with the real code 3; `code_exec` runs both
  the `source` form (`6 * 7` → `source-form 42`) and the `path`
  form against a script in the work root.
- **Environment scrubbing**: with a sentinel variable exported
  in the parent environment, the child's `env` output contains
  `PATH=` and neither the sentinel's name nor its value anywhere
  in the journaled record.
- **`test_run` honesty**: against a suite file with exactly
  2 passing / 1 failing / 1 skipped tests, the result parses
  `parse: "ok"` with counts (2, 1, 1) — pytest's own numbers —
  and exit_code 1 recorded as a completed execution; against a
  plain text file the result is `parse: "failed"`, counts
  (0, 0, 0), raw output carried.
- **Timeout**: `echo before; sleep 5` with `timeout_seconds: 1`
  closes `timed-out`, `partial: true`, exit_code −1 (the
  sentinel), the partial stdout (`before`) kept, wall time
  under the sleep's length.
- **Stream caps**: 5000 bytes of shell output under a 256-byte
  result cap completes truncated, flagged, inline stdout within
  the cap.
- **Integrity**: a `test_run` target priced `execute` at the
  gate and swapped for an outside-pointing symlink before
  EXECUTE is **refused** by the executor's re-pricing (TOCTOU),
  and the task loop's integrity test recognises the refusal as
  §10.4.4 material.
- **Stop rules, one task each**: completion (write + shell +
  `task_complete`, every step ADMITted, the task entry a real
  operator entry with no fixture label, the stop journaled);
  budget-exhausted twice over (policy cycle cap 2 against a
  3-step script — the third step never runs; operator
  `--cycles`-style cap 1 against a 2-step script); escalated
  (an external-cwd shell step halts the task; the following step
  never runs); reject-streak (unknown tool + schema violation
  halt at two; a second task proves one REJECT followed by
  success and completion does NOT halt); proposer-exhausted
  (labelled, and asserted not to be any of §10.4's five).
- **A mind proposer** (scripted class with `perceive`/`propose`)
  receives the task entry as its first perception and, on the
  next cycle, the fs_write observation tagged
  `source: "tool:fs_write"` (§4.5), then reports completion.
- **Portability (§12.5(d))**: after a task completes, the bundle
  directory is copied to a new path; the copy `load_bundle`s and
  verifies its journal chain, and the far side answers the whole
  trail — the task entry, the `task_stop`, the step's execution
  record with its captured stdout, and the provenance walk from
  the step's self-edit to the task entry.
- **Demo honesty**: a task on a demo bundle ESCALATEs its first
  step with the B-1 shadow record and halts `escalated`; nothing
  executes.

## 4. Test results

Baseline on main at this slice's base (`46fc2de`, PR #48
merged): **468 passed, 2 skipped** (builder's own run, matching
the tasking). This slice adds `tests/test_gaia_b3.py` (24 tests)
and modifies exactly two assertions in `tests/test_gaia_b2.py`
(§2). Full-suite numbers from the builder's own run on the PR
head are stated in the PR body; the B-1 suite and every other
existing suite are unmodified and pass as part of that run.

---

## 5. Explicitly NOT built (do not read this slice as claiming them)

- **No external execution** — outside-work-root effects hold at
  ESCALATE in B-3 (§2). No intent in this slice executes with an
  external effective class, whatever the policy says.
- **No demo-bundle live execution** — the demo loop path and
  demo-bundle tasks keep B-1 shadow behaviour; live execution
  is miniature-scoped only.
- **No Hearth interface, no SSE stream, no service adapter** —
  B-4 and B-5 respectively; nothing here speaks to Hearth.
- **No A-2 / A-3 / A-4** and no real-base agency — unapproved,
  untouched (§14.2). Everything live here is exercised on the
  miniature bundle with scripted proposers, which is what B-3's
  evidence is; no trained mind's judgement is claimed, and
  `si task` without `--script` refuses rather than inventing a
  default mind.
- **No policy-default changes, no constitution change, no new
  journal collection, no NEDB schema change** — tasks, stops,
  and outcomes journal through the existing methods in the
  existing collections (`experiences` carries `task` /
  `task_stop` / `tool_result` kinds, as B-1/B-2 established).
- **No new tools** — the registry surface is still exactly
  RFC §8's twelve; B-3 changes which of them are live, not what
  exists. No shell sandbox beyond the stated bounds (cwd scope,
  timeouts, stream caps, env allowlist, budgets, journal):
  §17.3's non-controls remain non-controls, named as such.

---

*The gate is not the price of the powers; the gate is what makes
them SI's. B-2 gave the loop hands. B-3 gives it a shell, an
interpreter, and a test runner — the full harness-class surface,
live — and gives the work a shape: an operator's task, opened in
the journal before anything runs, advanced one gated cycle at a
time, and closed by a stop rule with its cause on the record.
The powers grew; the law they run under did not move.*
