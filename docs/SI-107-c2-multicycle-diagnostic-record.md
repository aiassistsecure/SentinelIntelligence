# SI-107 — C2 multi-cycle diagnostic: run record

This document is the record of the SI-107 run: the live
capture and fold-supplied scoring of the C2 multi-cycle
diagnostic designed in SI-106
(`docs/SI-106-c2-multicycle-diagnostic-design.md`). It
states what was run, what was frozen, and the counts the
frozen corpus produced under fold-supplied scoring. It is
a measurement record, not a conformance record.

## 1. Diagnostic statement (SI-106 §3, printed in the record per §11)

This is a diagnostic measurement. No bar was applied to
this corpus, no verdict is stated anywhere in this record,
and no unlock is computed or earned by it; unlocks remain
`[]`. D-C1 does not transfer to multi-cycle measurement
and was not applied. No earlier record (SI-79, SI-94,
SI-100, SI-105) is re-graded by anything here, and the
counts below are comparable with none of them in either
direction (SI-106 §10): they are rates over scored cycle
documents in multi-cycle task runs, a different quantity
from the cycle-0 corpora of the earlier records. The
§18.3 end-to-end claim is not earned by any outcome of
this run.

## 2. Corpus characterisation (SI-106 §9.7)

The capture executed all 30 task runs of the §4 shape and
every run ended under the pre-registered disciplines
(§4 below). The scored corpus nevertheless stands below
the §4.2 reference shape — 31 scored documents against
the 60 a full-shape capture yields, and one workload
family (perception-truncation) contributing no scored
documents at all. Under §9.7 the corpus is therefore
characterised as **INCOMPLETE-DIAGNOSTIC**: the frozen
partial corpus is preserved, all counts in this record
are stated whole with their denominators, and no headline
rate over M is quoted. The label describes the corpus,
never the candidate. A fresh attempt would be a new slice
under the same design, never a quiet extension of this
one; no extension was made here — budgets fixed before
capture were never extended, and no position was retried
around a guard halt.

## 3. Configuration (SI-106 §8, stated in full)

- Candidate: Imagine v13, unmodified GGUF
  (`imagine-v13-Q4_K_M.gguf`, sha256
  `12c5794948228a04196d040dfb898dd34ea871eaf61166831068b42c30684e60`),
  Resident on the live Gaia node (agency 127.0.0.1:11435,
  seat imagine-v13, ctx pool 4096, parallel 1,
  --total-gib 5), served model id verified equal to the
  GGUF path before capture and at the smoke gate.
- Trained template at serving (the SI-94 §1 correction):
  the gaia-node unit's `--binary` names the wrapper
  `llama-server-v13.sh`, which execs llama-server with
  `--jinja --chat-template-file chat-template-v13.jinja`
  (1,041 B, sha256
  `993c0af704bad76aeba4263a5318f7873a06420a8eb7d40b72a49c39599c0a66`),
  verified present in the child argv at pre-flight and
  after every node restart during the capture.
- Output contract: v1 teaching (the SI-102 state). The
  §8 v2 sibling form is **not taught**; it is priced when
  it appears (§6).
- Perception: A1 citable-set rendering **in** (SI-104) —
  the live configuration SI-105 measured — because C2 is
  the first design that exercises A1 where it has content
  (a non-empty citable set at cycles k ≥ 1).
- Gateway requests built by SI main
  `3a2060cf974642cad4dd249696ccc0f498e0a958` code,
  unmodified. Residency guard in force throughout: a
  guard poll before every attempt; a non-Resident poll is
  a capture halt, never retried around.
- Between cycles: **journaling without execution**
  (§9.5). Each entered body was parsed by the unmodified
  §9 parser against the capture journal's fold at that
  moment, sibling-resolved (SI-97 §6 walk), gate-priced
  and journaled through the live loop's own admission
  path (`si.gaia.drain.admit_tool_intent`, the exact call
  `run_task` makes) with a `queue_batch` entry per batch.
  The queue was never drained; no tool ran; no fixture
  state changed. The fold's `tool_result` rows are the
  admission path's own non-execution dispositions for
  REJECT/ESCALATE verdicts (`record_non_execution_outcome`
  / `record_shadow_outcome`: status `not-evaluated —
  shadow mode`, no measured fields); the fold contains
  zero `queue_claim` rows.
- Fixture: the C-1 fixture bundle machinery
  (`live.build_fixture_bundle`, `live.fixture_policy`:
  reads admitted by policy shape, writes denied,
  execution denied, external escalated; allowed tools
  fs_list / fs_read / fs_search), policy
  `max_tool_cycles_per_task` = 16 debited per intent
  evaluated across a task run, exactly as `run_task`
  debits.
- Authorisation: the lead's dispatch recorded in
  `~/workspace/si-dev-loop/STATE.md` (2026-11-12 ~16:45
  EST), naming this design, this configuration, and the
  budgets below, recorded before the smoke gate.

## 4. Capture accounting (§9)

Capture window: 2026-11-12 18:00:58 EST →
2026-11-13 02:25:57 EST (America/New_York), driver under
systemd (`si107-capture.service`, Restart=on-failure,
guard-halt exit status 86 in RestartPreventExitStatus),
per-attempt persistence throughout.

- Task runs: 30 executed (6 workload families × 5), each
  one journaled task identity, cycles 0..2 (K = 2).
  Endings: **19 completed** the full cycle shape
  (benign-read ×5, injection-shaped ×5, multi-cycle-read
  ×5, hallucinated-reference #0/#1, depends-on-chain
  #0/#1); **11 ended on attempt-budget exhaustion** at
  a cycle position: depends-on-chain #2/#3/#4 at
  cycle 0, hallucinated-reference #2/#3 at cycle 2 and
  #4 at cycle 1, perception-truncation #0/#1/#2/#3 at
  cycle 1 and #4 at cycle 0.
- Attempts: **130** POSTs, each preceded by a passing
  guard poll. Outcomes: **66 captured** documents
  entered, **47 duplicate-capture** (byte-identical
  bodies; logged, never entered), **17 transport-failed**
  (unscored; 10 occurred at scored positions and are
  carried in the corpus as transport-failure placeholder
  items). Attempt budget: at most 10 per workload per
  cycle position, fixed before capture, shared across
  that workload's runs (the SI-79 budget shape), never
  extended. Per-position spend (attempts): benign-read
  5/9/6, injection-shaped 7/8/9, hallucinated-reference
  5/10/10, multi-cycle-read 10/7/8, depends-on-chain
  10/3/3, perception-truncation 10/10/0 (cycles 0/1/2).
- Residency guard halts: **2** (2026-11-12 20:39 EST and
  2026-11-13 01:46 EST), each on a Lost poll after a
  llama-server child death; at each halt no attempt was
  spent, state was persisted and backed up, and the
  capture resumed only after the node was restored to
  Resident and its seat re-verified. Infrastructure
  interruptions beyond the halts: the capture unit was
  wiped from /etc by environment recycles four times and
  reinstalled from its workspace source each time; the
  driver resumed from persisted state after every
  interruption. No scored document and no attempt budget
  was lost to any interruption; the only budget consumed
  by an interruption is the transport-failed attempts
  that had already POSTed when a child died (unscored,
  above).
- Smoke gate (§9.4), passed before any capture budget
  was spent: (a) a proposer-shaped request proved the
  served model is the exact candidate under §3's
  configuration (A1 cycle-0 citable set rendered; no
  ChatML markers in the body); (b) a complete dummy task
  run in a separate smoke fixture proved the multi-cycle
  machinery end to end — cycle 0 journaled, cycle-1
  perception rebuilt with non-empty `prior_cycles` (1)
  and non-empty citable set ([10]), cycle 1 journaled,
  and the §5.1 fold assembled from the fixture journal
  (19 rows, sha256
  `9a99e4e458d0d760ff3b6acdb4978c6205b03d5e49aaae4fbdfb7cb03f46f3f1`;
  queue_claim rows 0; its 2 tool_result rows both
  non-execution dispositions). One correction is part of
  this record: the gate's first run failed a no-execution
  assertion the coordinator had written as
  `tool_result rows == 0`; journal forensics and the
  admission path's source showed those rows are the
  path's designed non-execution dispositions (§3 above),
  so the assertion — not the machinery — was wrong. The
  proof was corrected to its §9.5-faithful form (zero
  queue_claim rows; every tool_result row a
  non-execution disposition) and the gate was re-run
  end-to-end on a fresh fixture and passed in that form;
  the failed first run's artifacts are preserved in the
  run's evidence.

## 5. Freeze and fold (§5.1)

At capture close the corpus and the capture journal were
frozen together; the frozen copies are the only inputs
scoring read.

- Corpus: 41 items — 31 scored documents + 10
  transport-failure placeholders at scored positions.
  Harness corpus fingerprint:
  `sha256:9411efc739b02271c2f074f8d7113f644945907019fa6a2043e45bf00c6164be`.
  Corpus file sha256:
  `dc399e9b4a0add27b5bf1ac91afdae8b542e90d33e4b9f5e7e591c3e6aa00fc2`.
  `validate_corpus` reported no deficiencies.
- Fold: the capture journal itself — **371 rows**
  (experiences 247, provenance 82, self_edits 40,
  checkpoints 2, deltas 0), tip seq 371. Fold sha256,
  computed over the fold's canonical bytes exactly as
  frozen (all journal collections' rows as
  `{collection, row}`, sorted by (collection, seq),
  serialised with `si.bundle.canonical_json`):
  `sha256:10756d341b9a447ec5db21c11192e822e0d554057475efdaa84366300697f0a3`.
  The hash was computed identically over the live journal
  and the frozen copy at freeze time, and re-computed by
  the scoring run before any document was scored; a
  mismatch would have refused scoring (§5.1 step 2). It
  matched.
- Fold provenance (§5.1): every fold row originated in
  this capture's own journaling — the capture fixture was
  created fresh for this run, and every intent row in it
  was journaled by the §3 admission path from a body this
  capture recorded. Per item, the corpus record states
  the task identity (`prompt_ref.task_seq`, read into
  `parse_task_seq` by the §5.1 step-4 mapping for every
  item; no item lacked it), the journaled cycle-0..k−1
  intent seqs the item's scored cycles could cite, and
  the citable set as rendered at capture.
- Scoring: the unmodified C-1 harness fold
  (`si.gaia.conformance.score_corpus`) with the frozen
  fold supplied as `journal` — the first fold-supplied
  scoring under D-96-3. An independent per-item re-parse
  agreed with the harness on all 31 scored items
  (0 mismatches).

## 6. Counts (descriptive; §9.7 — stated whole, no headline rate)

M = **31** scored documents (cycle-1..2 documents whose
capture perception had non-empty `prior_cycles`).
Parse-ok: **17** of 31. Stage histogram over the 31:
stage 1 = **9**, stage 2 = **0**, stage 3 = **0**,
stage 4 = **5**, stage 5 = **0**, stage 6 = **0**
(22 documents reached stage 4; 5 were refused there).

Per workload family (parse-ok / scored documents):
benign-read 4/7 · injection-shaped 8/10 ·
hallucinated-reference 1/5 · multi-cycle-read 2/6 ·
depends-on-chain 2/3 · perception-truncation 0/0
(no scored documents: the family entered only four
documents, all at cycle 0 — three of them journaled
intents — but no cycle-1 body was ever entered for the
family within the shared cycle-1 attempt budget, so no
document of that family stood at a scored position).

Per cycle position (parse-ok / scored documents):
cycle 1 = 8/14 · cycle 2 = 9/17.

Strict per-item counts (labelled as such, per §11): 18
of the 30 task runs carry at least one scored document;
in **8** of those 18, every scored document of the run
is parse-ok. 12 task runs carry no scored document.
All 17 parse-ok documents are single-intent documents
with intents-only signals (no `task_complete` and no
`cannot_proceed` signal among them).

## 7. Reference forms, priced separately (§6)

Five `depends_on` entries occur in the scored corpus,
in five documents. Every entry was priced; none was
estimated.

- **Journal-seq form** (priced against the frozen fold):
  **4 occurrences, 0 priced-and-passed, 4
  priced-and-refused** at stage 4 — 3 × `seq 0`
  (`seq_nonpositive`) and 1 × `seq 181`
  (`seq_not_intent_row`: seq 181 exists in the fold as
  that run's own *task* row — the citation names the
  task, not an intent of it). None of the four targets
  was in its item's citable set at capture.
- **Sibling form** (§8 v2, priced document-locally):
  **0 occurrences** — in the scored documents and in
  the whole 66-document capture. The form was not
  taught (§3); its occurrence count is reported as
  exactly 0.
- **Other shapes**: 1 occurrence, priced-and-refused at
  stage 4 (`entry_malformed`).

The 26 scored documents carrying no `depends_on` entry
are the remainder; a document that failed before stage 4
(9 stage-1 failures) had its references, if any,
unpriced by construction — the five entries above all
occurred in documents that reached stage 4.

## 8. What this measurement does not price (§7)

Unchanged from the design: execution correctness of any
intent (nothing executed); the §18.1 rows P5-R1/P5-R9;
dependency-ordered execution and queue/drain behaviour;
gate behaviour beyond admission pricing at journaling;
resolution correctness in a live walk; correct
refraining at cycle 0 (SI-105's quantity); the §18.3
claim. Corpus and fold are referenced by hash only; no
corpus body is copied into this record.

## 9. Artefacts

- Frozen corpus and fold, freeze record, and the
  fold-supplied scoring record:
  `~/workspace/si107-work/freeze/` (corpus
  `SI-107-corpus.json`, fold bundle `capture-bundle/`,
  `freeze-record.json`, `si107-scoring-record.json`).
- Capture documents, capture log, driver state and
  backups: `~/workspace/si107-work/out/` and
  `~/workspace/si107-work/backup-*/`.
- Driver, smoke gate, freeze and scoring scripts:
  `~/workspace/si107-work/` (`run_capture.py`,
  `smoke_gate.py`, `c2_common.py`, `freeze.py`,
  `score.py`; unit sources `si107-capture.service`,
  `si107-smoke.service`).
- Run evidence, per phase:
  `~/workspace/si-dev-loop/si107-c2-evidence.md`.
