# SI-43 — Gaia B-2: The Executor and the Filesystem/House Tools,
Live but Miniature-Scoped (SI-41 RFC §15 B-2)

> **Status: BUILT, in review (PR opened by the B-2 builder; merging
> is the dev-loop driver's, on a later run).** This slice implements
> SI-41 §15's B-2: a real executor (`si/gaia/executor.py`), the
> journaled-args rule (§9.2), result bounding and blob overflow
> (§9.3), §4.4 execution records with measured fields, VERIFY as
> result conformance + post-state observation (§11.1), and
> pre-image journaling for compensable writes (§11.2) — with live
> execution of the `read`/`write`-class tools (§8.1, §8.2) on
> **miniature bundles only**.
>
> **The live-scope boundary, stated exactly.** LIVE (gate-ADMITs
> and the executor runs them, on miniature bundles): `fs_list`,
> `fs_read`, `fs_search`, `fs_write`, `fs_edit`, `fs_mkdir`,
> `journal_query`, `bundle_status`, `probe_run`. STILL ESCALATE
> (nothing executes): `shell_exec`, `code_exec`, `test_run`
> (execute class — B-3), every intent whose effective class prices
> to `external` (outside the work root), every tool intent on a
> demo (non-miniature) bundle, and every tool intent on any
> non-live gate context. Nothing executes without a gate ADMIT on
> the journaled record — the executor re-checks the verdict itself.
>
> **Terminology:** machine learning infrastructure throughout.
> CPU only; no GPU, no cloud, no spend, no downloads. No bytes are
> duplicated: overflow results and large pre-images are
> content-addressed in the bundle's existing blob store and
> referenced by hash (Mark's design rule).

---

## 1. What was built

### 1.1 The gate: §5.4's algebra, live for B-2's set (RFC §5, §15)

`Gate._evaluate_tool` now branches on the gate context's
`tool_live_execution` flag (`si/gaia/context.py`):

- **Non-live contexts** (the demo `LoopContext`, scripted
  contexts, the B-1 corpus): B-1 behaviour **byte-for-byte** —
  every passing intent ESCALATEs, every tool report records
  `not-evaluated — shadow mode`. The existing B-1 suite passes
  unmodified; the demo loop path is not flipped (RFC §15 B-2
  prices the paths separately, SI-17 precedent).
- **Live contexts** (only `si.probes.ProbeContext` — the
  miniature loop's context — carries the flag): decisive checks
  unchanged (constitution, tool safety with its itemised
  sub-checks, provenance with the chain-to-task rule; checks 3/4
  still the §5.3 not-applicable label). Then:
  - decisive failure → **REJECT**, execution label
    `not executed — rejected at gate (SI-43/B-2)`;
  - tool in the B-2 live set (`LIVE_TOOLS_B2` in
    `si/gaia/registry.py`), effective class `read`/`write`,
    class rule `admit` → **ADMIT**, execution label
    `live execution authorised — SI-43/B-2 (miniature-scoped,
    read/write classes)`;
  - class rule `escalate` → **ESCALATE** by class policy
    (the §7.3 dial, distinguishable on the record);
  - anything else passing (execute-class tools; external-effective
    intents) → **ESCALATE**, `not-evaluated — outside SI-43/B-2
    live scope …`.

### 1.2 The executor (`si/gaia/executor.py`)

One runtime component, as RFC §4.3 designs it (registry +
executor + execution recorder — here the registry stays in
`registry.py` from B-1, the executor and recorder live together):

- **`Executor.execute_journaled(edit_id, expected_effective_class)`**
  executes **the journaled intent** (§9.2): it re-reads the
  self-edit row from the journal, and refuses — journaling the
  refusal as a `refused` execution record — when the row's gate
  verdict is not ADMIT, the tool is unknown or outside the B-2
  live set, the bundle is not miniature-scoped, the journaled
  args no longer validate against the registry schema, the
  effective class has **changed since admission** (the TOCTOU
  rule: priced as `read`, re-priced as `external` after a symlink
  swap → refused, never executed under a stale price), the
  effective class is outside `read`/`write`, or any resolved path
  lands outside the work root. Defence in depth: the gate is the
  authority, and the executor re-prices anyway.
- **Handlers** for the nine live tools, all real: filesystem
  tools resolve paths against the work root with symlinks
  resolved before containment (the resolved target must be
  inside the root — §9.1); `fs_read` returns text or base64
  (flagged) truncated at the declared `max_bytes` and the policy
  result cap; `fs_search` is a literal-pattern line search over
  the scope (match-capped, binary files skipped); `fs_write`
  writes atomically (temp + replace); `fs_edit` is exact-match
  and **fails** when `old` is absent or ambiguous; `fs_mkdir`
  creates one level and reports `created: false` as a completed
  no-op when the directory already exists; `journal_query` reads
  the mind's **own** journal through its backend (the five
  collections, scalar exact-match filters, limit capped at 50);
  `bundle_status` derives `{head, deltas, last_seq, policy_hash}`
  from the verified manifest loaded at cycle start (a mid-cycle
  `load_bundle` re-load would fail the journal-anchor check for
  the wrong reason — the journal tip has legitimately moved —
  so handlers never re-load; the record states `last_seq` is the
  manifest-anchored seq); `probe_run` runs the **real** SI-15
  probe harness through `ProbeContext` on miniature bundles
  (identity: measured max drift vs `identity_tol`; capability:
  max regression vs the recorded seed baselines plus the echo-
  accuracy arm vs the recorded floor — labelled
  `measured-pass`/`measured-fail`/`not-evaluated` exactly as
  produced) and returns an honest `not-evaluated` on bundles
  with no harness (RFC §8.2).
- **Execution records (§4.4)** are journaled via the existing
  `record_experience` (`kind: tool_result`, `source: tool:<name>`)
  in the existing collections — no new collection, no schema
  change (§4.6). Fields: `tool`, `intent_id`, `class` (effective),
  `reversibility` (from the registry), `status`
  (`completed`/`failed`/`refused`), `exit_code` (**null** — no
  B-2 tool spawns a process; the field is present so the §4.4
  shape is uniform when B-3's process tools land), `result`
  (the inline bounded result), `result_hash`/`result_bytes`
  (over the exact inline result PERCEIVE hands the mind),
  `truncated`, `result_blob` + `overflow_result_hash`/
  `overflow_result_bytes` when §9.3 overflow blob-stores the full
  result, `wall_time_seconds` (a real `perf_counter` measurement),
  `post_state` (observed after the act — for writes, the file's
  re-read size and SHA-256), `pre_image` (writes; §1.4), and
  `verify` (§11.1: `result_conformance` and `post_state`, each
  `passed`/`failed`/`not-evaluated`). A handler failure is a
  **failed execution record**, never an exception past the
  executor (§11.1.3); the self-edit's result becomes `executed`
  or `failed`/`refused` accordingly.
- **Result bounding (§9.3):** the inline cap is the bundle
  policy's `tool_max_result_bytes`. Handler results are bounded
  by construction (list cap 512 per §8.1, search match cap,
  read cap); a result still over the cap is blob-stored whole
  (content-addressed, deduped — an identical payload already in
  the store is referenced, never rewritten) and the inline form
  is honestly shrunk (tail items dropped / content prefixed)
  with `truncated: true` and the overflow hash/bytes recorded.
- **`execute_tool_cycle`** is the B-2 form of B-1's
  `journal_tool_intent` (gate → journal self-edit + provenance
  **before** execute → executor → outcome), and
  **`run_tool_intent`** is the on-disk scripted-scenario driver
  (loads/verifies the bundle, builds the miniature or demo
  context by bundle kind, re-anchors the manifest checkpoint so
  the bundle loads cleanly afterwards). The operator task
  surface (`si task`, task loop, stop rules) remains B-3;
  `record_task_fixture` (B-1) still supplies task entries for
  scripted scenarios.

### 1.3 The loop (RFC §4.2: EXECUTE replaces MATERIALIZE)

- **Miniature path** (`_run_miniature_cycle`): `ProbeContext`
  now carries the tool gate-context contract with
  `tool_live_execution = True`. An ADMITted tool intent is
  journaled (intent + verdict + provenance, as every kind),
  then **executed** by the executor from its journaled record;
  the cycle returns the execution record, `head` does not move,
  no delta is recorded, and `verified` reports §11.1's outcome
  (execution completed + conformance + post-state passed). A
  failed execution does **not** roll anything back — world
  effects have no rollback (§11.2) — it returns as its journaled
  record. REJECT/ESCALATE journal the B-2 non-execution outcome
  (the gate report's live label, no measured fields).
- **Demo path** (`run_cycle`): `LoopContext` carries
  `tool_live_execution = False` explicitly; tool intents
  REJECT/ESCALATE with B-1's shadow record, byte-for-byte
  (asserted by the unmodified B-1 loop test and a new B-2 demo
  test at ESCALATE).

### 1.4 Pre-images and compensation (RFC §11.2)

Every `fs_write`/`fs_edit` execution journals its **pre-image**
before writing: existence, SHA-256, and byte size always; the
bytes themselves inline when they fit the result cap (text when
UTF-8-decodable, base64 otherwise) and blob-stored when they do
not. `si.gaia.executor.compensation_intent` builds the
compensating intent — a **new, ordinary `fs_write` intent** of
the pre-image back to the same path, gated and journaled like
any other action — exactly where the v1 surface can honestly
carry it, and returns `None` with the reason where it cannot:
a created file's prior absence (the v1 registry has **no
delete tool**, §8.4), a binary pre-image (`fs_write`'s content
is text), a directory creation (no inverse tool). Compensation
is a new action in the world, not a rewind, and the record
never calls it rollback.

---

## 2. Pinned deviations and open fields (stated, never silent)

- **Executor location.** RFC §4.3 named `si/tools.py`; the
  executor lives in `si/gaia/executor.py` because Mark's
  2026-10-29 directive placed all Gaia code in the `si/gaia/`
  subpackage (where B-1 already put the registry, policy, and
  shadow journaling). The RFC's component design is followed;
  only the module path follows the later directive.
- **External held at ESCALATE in B-2.** §7.3 makes `external`
  admittable by policy in principle; B-2's live scope is
  work-root-contained execution (RFC §15 B-2: live execution of
  the read/write tools *in scope*), so an external-effective
  intent ESCALATEs in this slice even under `external: admit`.
  This is a slice-scoped hold like B-1's shadow discipline —
  stated on the gate report's label and here — for a later
  slice to lift on its own evidence, not a change to §7.3.
- **`exit_code: null`.** §4.4's record shape carries
  `exit_code`; no B-2 tool spawns a process, so the field is
  recorded as null (a measured absence) rather than omitted or
  filled with a fabricated 0.
- **`journal_query` filters** are the journal backend's
  exact-match filters over scalar values; a non-scalar filter
  value is a failed execution with the reason, not a silent
  empty result.
- **`fs_search` pattern** is a literal substring (the RFC does
  not specify regex); excerpts are capped at 200 chars.
- **The B-1 corpus is unchanged**: its contexts are non-live,
  so its verdict table (all ESCALATE/REJECT, shadow label)
  stands as the shadow discipline's record.

---

## 3. Evidence (all measured by the builder on the PR head)

- **The scripted multi-step scenario (RFC §14.1)** —
  `fs_list → fs_read → fs_write → fs_edit → journal_query` —
  completes against a real miniature bundle (embedded journal)
  opened by a scripted task fixture, every step ADMITted and
  `completed`, every step's provenance walking via
  `walk_tool_provenance` to the same task entry, every
  execution record's `result_hash`/`result_bytes` recomputing
  from its journaled result, the journal chain verifying.
- **Escape/classification cases**: an absolute path outside
  the work root and a symlink inside the root pointing outside
  both price to `external`, ESCALATE, and the outside file's
  content appears nowhere in the journaled record.
- **TOCTOU**: a file priced `read` at the gate and swapped for
  an outside-pointing symlink before EXECUTE is **refused** by
  the executor's re-pricing; the outside content is never read.
- **Journaled-args discipline**: an ESCALATEd intent handed to
  the executor is refused (`not ADMIT`); a fabricated ADMIT for
  `shell_exec` is refused (B-3 scope) — nothing executes without
  a real gate ADMIT for a live tool.
- **Pre-image + compensation**: overwriting `plan.md` journals
  the original bytes and their SHA-256; the compensating
  `fs_write` built by `compensation_intent`, run through the
  same gated cycle, restores the file byte-for-byte.
- **Failures are results**: `fs_edit` absent/ambiguous,
  `fs_write` into a missing parent, `fs_read` of a directory,
  `fs_list` of a file — each a journaled `failed` execution,
  the filesystem unchanged.
- **House tools**: `bundle_status` matches the verified
  manifest field-for-field; `probe_run` returns
  `measured-pass` with a real measured float for both suites
  on the miniature, and `not-evaluated` (measured null) on a
  demo bundle.
- **Result bounding**: with a 400-byte result cap, an
  `fs_list` of 32 entries completes truncated, the full result
  blob-stored and hash-bound by the overflow fields.

## 4. Test results

Baseline on main at this slice's base (`b55f616`, PR #47
merged): **448 passed, 2 skipped** (builder's own run, matching
the driver's). This slice adds `tests/test_gaia_b2.py`
(20 tests). Full-suite numbers from the builder's own run on
the PR head are stated in the PR body; the existing B-1, gate,
loop, and probe suites are unmodified and pass as part of that
run.

---

## 5. Explicitly NOT built (do not read this slice as claiming them)

- **No execution of `shell_exec` / `code_exec` / `test_run`** —
  B-3. They remain registered, gate-priced, and ESCALATE-only,
  and the executor refuses them even on a fabricated ADMIT.
- **No `si task`**, no task loop, chaining driver, budgets as a
  running loop, or §10.4 stop rules — B-3. B-2's evidence uses
  scripted cycles and the B-1 task fixture, as the RFC's B-2
  evidence list specifies.
- **No external-scope execution** — outside-work-root effects
  hold at ESCALATE in B-2 (§2 above).
- **No demo-bundle execution** — the demo loop path keeps B-1
  shadow behaviour; live execution is miniature-scoped only.
- **No Hearth fork, no Hearth bridge, no SSE stream** — the fork
  is its own step on Mark's direction; B-4 is the Hearth
  interface slice.
- **No A-2 / A-3 / A-4** and no real-base agency — unapproved,
  untouched (§14.2). Everything live here is exercised on the
  miniature bundle with scripted intents, which is what B-2's
  evidence is; no trained mind's judgement is claimed.
- **No policy-default changes, no constitution change, no new
  journal collection, no NEDB schema change**, no new tool in
  the registry (the surface is still exactly RFC §8's twelve).

---

*The gate is not the price of the powers; the gate is what makes
them SI's. B-1 gave the loop a way to* want *to act. B-2 gives it
hands — small, scoped, journaled hands: every act re-priced at
the moment it happens, every result measured, every overwrite
carrying its pre-image, and nothing outside the work root, the
execute class, or the miniature's scope pretending to be live
before its slice.*
