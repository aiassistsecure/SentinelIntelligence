# SI-46 — Gaia B-5: Agentic Work as a Service (the SI-side adapter)
(SI-41 RFC §15 B-5; §12.5(a), §12.5(e), §12.6 H-2)

> **Status: BUILT, in review (PR opened by the B-5 builder; merging
> is the dev-loop driver's, on a later run).** This slice implements
> SI-41 §15's B-5, SI-side only, in `si/gaia/service.py` (plus an
> additive submitter-provenance extension to `si/gaia/task.py` and
> a `si gaia-serve` CLI surface): a stdlib-HTTP service in
> `si serve`'s pattern that accepts a task — with submitter
> identity as a mandatory journaled provenance field, no identity,
> no task — runs it through the B-3 machinery, and returns the
> result **with the provenance trail** serialised from the
> journal. B-5 is the last slice in §15's order; with it, the
> B-line the RFC sequenced is complete on the SI side.
>
> **The live-scope boundary, stated exactly.** LIVE (real code,
> exercised this slice): submission validation that fails closed
> before any journal write; a submitted task executed by the
> existing B-3 task loop (live execution miniature-scoped, demo
> bundles shadow — B-3's scope, unchanged); the submitter identity
> journaled on the task entry; the provenance trail serialised
> from the journal's existing collections (task entry, every
> chained intent with its full gate report, provenance edges,
> execution outcomes, and the stop, in seq order, with the
> journal's hash-chain verdict); the HTTP surface
> (`GET /`, `GET /health`, `POST /tasks`, `GET /tasks/<seq>`)
> exercised over real localhost HTTP in the tests.
> **Exposure through the Hearth gateway is NOT built** — it is
> proposal **H-2** (§12.6), Mark's to build in the Hearth repo;
> this service is exposed *beside* a gateway on the same node,
> never through one, and imports no Hearth code. The service has
> **no authentication**: the submitter identity is asserted and
> journaled, never verified (§2.1). No real Hearth node was
> contacted by this slice; nothing about the Hearth path is
> measured here either.
>
> **Terminology:** machine learning infrastructure throughout.
> CPU only; no GPU, no cloud, no spend, no downloads. No bytes
> are duplicated: the trail serialises journal rows, never
> tensor bytes (Mark's design rule).

---

## 1. What was built

### 1.1 Submission and the submitter-identity rule (§12.5(a))

`validate_submission` prices a submission (a JSON object)
fail-closed, itemising every problem:

- `submitter` — **mandatory**: a non-empty string naming the
  submitting party. Missing, blank, or non-string is a refusal:
  *no identity, no task*. §12.5(a) prices the risk exactly — a
  task from the network is an operator instruction only if the
  service's auth story says so — so the identity is designed in
  as a journaled provenance field (§1.2), which is what §15 B-5
  requires of this slice.
- `instruction` — mandatory non-empty string (the §10.1 task
  instruction).
- `script` — mandatory non-empty array of intent dicts, the
  scripted proposer of §14.1 in the same shape `si task
  --script` accepts. There is no other proposer: no trained
  mind proposes tool intents yet (§14), and the adapter does
  not invent a default mind — a submission without a script is
  refused, exactly as `si task` without `--script` refuses.
- `work_root` / `cycles` — optional, type-checked.

A refused submission returns `{"status": "refused", "problems":
[…], "journaled": false}` and — enforced structurally, asserted
by test — **nothing is journaled and nothing runs**: validation
completes before the bundle is even loaded. Over HTTP the same
refusal is a 400; a body over the labelled 1 MiB cap is refused
before parsing on the same terms.

### 1.2 Submitter identity as a journaled provenance field

`run_task` / `record_task_entry` in `si/gaia/task.py` gained an
additive optional pair — `submitter` / `submitted_via` — present
on the task entry only when a service submitted the task:

```json
{"kind": "task", "source": "operator", "operator": "operator",
 "instruction": "…", "work_root": "…", "budgets": {…},
 "submitter": "<asserted identity>",
 "submitted_via": "gaia-service (SI-46/B-5)",
 "submitter_identity_label": "submitter identity is an asserted
   provenance field, journaled verbatim — it is NOT
   authentication: no credential was checked, and the identity
   is not verified (SI-41 §12.5(a); the si admit --operator
   precedent)"}
```

Operator-opened tasks (`si task`) pass neither field and their
entries are byte-unchanged (asserted by test). The label is a
single-sourced constant (`SUBMITTER_IDENTITY_LABEL` in
`si.gaia.task`), carried identically on the task entry, the
service response, and the trail, so the three can never drift.

### 1.3 Running the task: the B-3 machinery, reused whole

`submit_task` converts the script with `intents_from_script`
(mirroring `si.cli.cmd_task`'s conversion exactly) and calls
`run_task` — gate, journal-before-execution, executor, budgets,
and the §10.4 stop rules are B-3's, unmodified. Submission is
**synchronous**: the POST blocks until the task stops and then
returns. There is no job queue, no async handle, and no
streaming in B-5 (§2.4).

### 1.4 The provenance trail (§12.5(a)'s deliverable)

`build_provenance_trail(journal, task_seq)` serialises, from the
journal's existing collections only (no new collection, no NEDB
schema change, §4.6):

- the **task entry** row, submitter identity included;
- every **step** chained to the task — found by following the
  §10.2 chain itself (an intent citing the task seq, or citing
  the outcome seq of a step already in the trail, transitively)
  — each with its full self-edit: the intent, the reason, the
  gate verdict **and the full gate report** (itemised checks),
  the result, its **provenance edges**, and its **execution
  outcomes** (the `tool_result` records);
- the **task_stop** row with its §10.4 cause;
- `journal_chain_verified`: the journal's own hash-chain
  verdict at read time.

An unknown task seq raises `KeyError` — reported as unknown
(HTTP 404), never fabricated. The trail is also the §12.5(d)
portability story's read side: it is built purely from journal
rows, so a moved bundle answers it identically (B-3 measured
the move; B-5's trail reads the same rows).

### 1.5 The HTTP surface and the CLI

`make_service_server` follows `si serve`'s pattern (stdlib
`ThreadingHTTPServer`, state on `server.state`, `port=0`
ephemeral for tests): `GET /` (description + labels),
`GET /health` (bundle id + the *current* head read fresh — the
service is not read-only, §2.3), `POST /tasks`,
`GET /tasks/<seq>`. `si gaia-serve <bundle> [--host] [--port]`
is the CLI surface, localhost by default, with the no-auth and
H-2 labels in its help text and its startup banner.

## 2. Pinned deviations and open fields (stated, never silent)

1. **The submitter identity is asserted, not authenticated.**
   §15 B-5 requires the identity *designed as a journaled
   provenance field*; it does not — and B-5 does not — create
   an auth story. No credential is checked, no signature is
   verified, and any party who can reach the port can assert
   any identity. That is precisely §12.5(a)'s confused-deputy
   risk, priced and labelled rather than solved: the journal
   records who a task *claimed* to be from, which is what makes
   a bad task attributable after the fact; it does not make a
   bad task impossible. Multi-tenancy stays out of scope until
   a real auth story exists (§19). Localhost-by-default and the
   no-security-claim label are the operating discipline until
   then (the `si admit` --operator and `si serve` precedents).
2. **Scripted proposer only.** A service whose tasks are
   scripts is a service for *submitted, reviewable* work — the
   honest v1 of §12.5(a) while no trained mind proposes intents
   (§14). A mind-proposer service surface would ride on
   `run_task`'s existing mind-shaped slot unchanged; B-5 does
   not build a proposer and does not pretend one exists.
3. **The service moves the head; `si serve` does not.** B-5's
   surface exists to *do* work, so accepted tasks journal and
   re-anchor the manifest exactly as `si task` does.
   `ServiceState` therefore snapshots identity and the startup
   head only, and `/health` reads the current head fresh —
   carrying `si serve`'s snapshot discipline over unchanged
   would have made `/health` lie after the first task.
4. **Synchronous submission.** POST /tasks runs the task in the
   request. The §10.3 budgets bound a single task (16 cycles by
   default), which is what makes a synchronous v1 defensible;
   an async/queued surface is a future slice's design, not a
   silent omission.
5. **One bundle per service process.** The adapter serves the
   bundle it was started for — the node-side shape §12.5(a)
   describes ("the SI service exposed beside the gateway on
   the same node"). Fleet routing across bundles is the
   gateway's concern (H-2), not built.

## 3. Hearth-side assumptions, listed against §12.6

| # | Assumption this adapter makes | §12.6 proposal it waits on |
|---|---|---|
| A-1 | The SI service can be exposed beside a Hearth gateway on the same node (PIN already routes services) — the adapter needs nothing from Hearth to run | **H-2** — the integrated gateway route for task submission; not built, not depended on |
| A-2 | Nothing in B-5 presents a bundle as a model or submits tasks through the gateway | **H-2** (and its §12.5(c) sibling) — Mark's |

No part of B-5 depends on H-1/H-2/H-3 landing. The Hearth fork
("Hearth becomes Gaia") remains Mark's step; no fork was
created and the hearth repo was not touched.

## 4. Evidence (the builder's own runs on the PR head)

- **End-to-end submission (miniature, scripted):** a submission
  (`fs_write` → `shell_exec` → `task_complete`) returns
  `task-returned` with stop `mind-reported-complete`, 2 tool
  cycles used, the output file really written in the work root,
  and a trail whose task entry carries the submitter identity,
  whose 3 steps each carry intent + full gate report +
  provenance edges + outcomes in seq order, and whose
  `journal_chain_verified` is true. `si why`'s existing
  provenance walk reaches the same task entry from the first
  step's edit — the trail and the tooling agree.
- **No identity, no task (priced, not prose):** five submitter
  mutations (missing / empty / blank / non-string / null) all
  refuse with the mandatory-identity problem itemised; over
  both the direct API and real HTTP, a refused submission
  leaves every journal collection's row count unchanged and
  writes no file.
- **Demo bundles stay shadow through the service:** a submitted
  `fs_write` against a demo bundle returns stop `escalated`
  with the B-1 shadow execution label, no file is written, and
  the submitter identity is still journaled on the task entry.
- **Trail retrieval:** `GET /tasks/<seq>` rebuilds the same
  trail from the journal (same step edit ids, same stop);
  unknown seq → 404; non-integer seq → 400.
- **Tests:** `tests/test_gaia_b5.py` — 24 tests (validation
  corpus, refusal-journals-nothing, end-to-end trail, trail
  rebuild, operator-entry compatibility, demo shadow, HTTP
  surface, CLI wiring, honesty guards). Full suite numbers in
  §5.

## 5. Test results

- Baseline on main `189ca0f` (builder's own run, workspace
  venv): **530 passed, 2 skipped**.
- B-5 file alone: **24 passed**.
- Full suite on the PR head: **554 passed, 2 skipped**
  (530 baseline + 24 new; no existing test modified — the
  `si/gaia/task.py` change is additive and its compatibility is
  asserted by a new B-5 test, not by editing B-3's).
- **Flake disclosure (verification history, for the reviewer):**
  two earlier full-suite runs on this head each reported
  553 passed / 1 failed / 2 skipped, the failure rotating
  between pre-existing real-NEDB-backend tests
  (`test_journal_ordering.py::test_journal_before_materialize`,
  then `test_seed.py::test_seeded_bundle_runs_cycle`). The
  journal-ordering failure reproduced **identically on
  unmodified main** in the same daemon-polluted environment
  (leftover `nedbd-v2` processes accumulate across long runs
  on this VM — the project's known stale-daemon flake class),
  and both tests pass on main and on this head once orphaned
  daemons are cleared. The numbers above are from the final
  verification: solo full runs in a fresh environment, main
  `189ca0f` = 530 passed / 2 skipped and this head = 554
  passed / 2 skipped, no failures. This slice's diff touches
  no journal, bundle, seed, or loop code.

## 6. Explicitly NOT built (do not read this slice as claiming them)

- **No exposure through the Hearth gateway** (proposal H-2),
  no gateway route, no bundle-as-model presentation
  (§12.5(c)), and no Hearth code of any kind — the adapter
  imports none. The Hearth fork is Mark's step; the hearth
  repo was not touched.
- **No authentication or authorization**: no credentials, no
  tokens, no submitter verification, no multi-tenancy (§19).
  The submitter identity is a journaled claim, labelled as
  such everywhere it appears.
- **No new proposer**: no trained or scripted-by-default mind;
  submissions carry their script or are refused. No async job
  queue, no task cancellation surface, no SSE stream changes
  (the journal remains the source of truth the trail is
  serialised from).
- **No new execution scope**: live execution remains
  miniature-scoped (B-2/B-3); demo bundles remain shadow;
  no `hearth_*` tool family (§12.5(b) self-maintaining serving
  remains a future slice on operating evidence).
- **A-2/A-3/A-4 and any real-base agency** — unapproved, not
  built. Real-base agency claims stay honestly scoped until
  measured. The Hearth path remains UNMEASURED (B-4's label,
  unchanged by this slice).
