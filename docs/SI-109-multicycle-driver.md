# SI-109 — The multi-cycle capture driver, in-repo (SI-106 §12 item 1)

## What this slice is

SI-106 §12 sequenced, as its first future slice, a multi-cycle
capture driver implementing §9 with replay/fixture evidence
only and no live capture. SI-108 §4.1 proposed it as SI-109,
and lead decision D-108-1 (2026-11-13) answered CONTINUE.
This slice lands that driver as `si/gaia/multicycle.py`, with
its acceptance tests in `tests/test_gaia_multicycle_si109.py`.

SI-107 ran this design live from workspace scripts
(`~/workspace/si107-work/`) under the lead's direct dispatch.
**Landing this driver does not retroactively make SI-107 a
repo-code run** — its record stands as a workspace-script run,
stated as such wherever this slice cites it. This slice earns
reproducibility and reviewability only: no §18.1 row, no
§18.3 progress, no rate, and no claim about any model
(SI-108 §4.3).

## What landed

- **§4 item shape** — one item is one task run: cycle 0
  present and unscored (it seeds the journal); cycles 1..K
  are scored documents only when their capture perception
  carried a non-empty `prior_cycles`
  (`scored_documents`, `build_multicycle_corpus`). Duplicates
  are logged and never entered (bodies stored once);
  transport failures are recorded unscored — carried as
  placeholders at scored positions, in no denominator
  (`fixed_n` is M, the scored-document count).
- **§9.5 journaling without execution** — `journal_cycle`
  mirrors `run_task`'s admission walk (parse → SI-97 §6
  sibling resolution → `queue_batch` → `admit_tool_intent`,
  budget debit per intent evaluated) with the one designed
  difference: the queue is never drained and no tool runs.
  `collect_no_execution_proof` asserts the §9.5-faithful form
  SI-107's smoke gate corrected itself to: zero `queue_claim`
  rows, and every `tool_result` row a non-execution
  disposition (REJECT/ESCALATE verdict, no measured execution
  fields). An ADMITted intent journals no `tool_result` row in
  this walk; it sits queued, unexecuted.
- **§9.2 residency guard** — the driver polls an injected
  guard before every attempt (`_poll_residency`'s interface
  accepts a callable, a `poll()` object, or a
  `journal_residency()` live-client shape). A non-Resident
  poll or a failed poll persists state, marks the halt in it,
  and raises `ResidencyGuardHalt` spending no attempt; the
  driver never retries or restarts around a halt.
- **§9.3 budgets** — at most `max_attempts_per_workload_cycle`
  attempts per workload per cycle position, shared across that
  workload's runs, fixed at construction and exposed
  read-only. No code path in the module extends, refunds, or
  resets a budget; resume seeds from prior state with attempts
  already spent left spent.
- **§9.4 smoke gate, fixture scope** —
  `run_fixture_smoke_gate` runs one dummy task run (K = 1) on
  a fixture bundle with a scripted source and proves: guard
  Resident, cycle 0 journaled, cycle-1 perception rebuilt
  with non-empty `prior_cycles` and citable set, the §5.1
  fold assembled from the fixture journal, and the §9.5
  no-execution proof. It raises `SmokeGateError` naming the
  failed check. The live gate's served-identity leg is **not**
  claimed here — it belongs to a live authorisation's own
  pre-flight (§9.1), which this slice does not grant.
- **§9.6 freeze** — `freeze_multicycle` freezes corpus and
  capture journal together: the bundle is copied, the corpus
  written beside it, and the record carries the corpus file
  sha256, the corpus canonical sha256, and the fold sha256 —
  the fold computed identically over the live journal and the
  frozen copy (a mismatch refuses the freeze), over all
  journal collections' rows as `{collection, row}` sorted by
  `(collection, seq)` in canonical JSON form, exactly the
  procedure SI-107's workspace freeze used. `verify_freeze`
  recomputes both hashes independently. A second freeze into
  the same directory is refused.

## Evidence (this slice's own runs)

New tests: `tests/test_gaia_multicycle_si109.py` — 10 passed.
Affected suites re-run (conformance C-1, conformance capture,
proposer P-1/P-2, SI-104, SI-98, Q-1 queue/drain/task) — 166
passed. Exact commands and counts are in the build evidence
file outside this repo (`~/workspace/si-dev-loop/
si109-build-evidence.md`).

## Non-goals and honesty labels

No live capture, no live model contact, and no authorisation
to make one — the driver takes both its proposal source and
its residency guard by injection, and this slice exercises
them only with scripted sources on fixture bundles. No new
workload family; no change to SI-106's item definition,
denominators, or verdict-free discipline. This diff touches
no gate pricing, no parser stage rule, and no teaching text
(`OUTPUT_CONTRACT_INSTRUCTION` untouched) — per SI-106 §12,
such a diff would fail this slice's own review.

Carried forward, unweakened: §18.3 **NOT earned**; P5-R1 /
P5-R9 **UNMEASURED**; live model-authored intent **OCCURRED,
scoped to SI-74 only**; L-3 **LOCKED**; A-2/A-3/A-4
**UNAPPROVED**; unlocks `[]`; D-C1 **stands** for cycle-0
conformance and is not applied by anything here.
