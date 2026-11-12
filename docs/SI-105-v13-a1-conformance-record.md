# SI-105 — Imagine v13 Conformance Record, A1 Citable
# Reference Set Rendering + v1 Teaching (SI-91 §7 procedure,
# changed configuration)

> **Status: CONFORMANCE RECORD. Verdict: BELOW THRESHOLD.**
> A fresh corpus of **30 scored items** was captured live from
> Imagine v13 served under its corrected-baseline serving
> configuration (its trained template supplied at serving) with
> gateway requests built under the **A1
> `citable_reference_set` rendering (SI-104)** and the **v1
> teaching (SI-102 state)**, and scored through the C-1
> harness, unmodified.
> **Measured result: n = 30, parse_ok = 18, rate 18/30 = 0.60.**
> Against the decided D-C1 bar (rate ≥ 0.80 over N ≥ 30,
> per-workload floor ≥ 0.60) the result is **BELOW
> THRESHOLD**: the rate is not met (0.60) and the floor is
> not met (perception-truncation 0.20).
> **Scope:** this record changes no code, adds no test, moves
> no gate behaviour, and extends no attempt budget. It
> records a capture and a scoring that already happened,
> from their own logs and the harness's own record.
> **Honesty labels carried forward verbatim, unweakened:**
> §18.3 **NOT earned**; P5-R1 **UNMEASURED**; P5-R9
> **UNMEASURED**; live model-authored intent **OCCURRED,
> scoped to SI-74 only**; L-3 **LOCKED**; C-4 **does not
> proceed for v13 on this record**; A-2/A-3/A-4
> **UNAPPROVED**; readiness unlocks `[]`.

## Table of contents

- [0. Status and scope](#0-status-and-scope)
- [1. Candidate and configuration provenance](#1-candidate-and-configuration-provenance)
- [2. The smoke gate](#2-the-smoke-gate)
- [3. Capture](#3-capture)
- [4. Scoring — the §3.4 record](#4-scoring--the-34-record)
- [5. Verdict vs D-C1](#5-verdict-vs-d-c1)
- [6. Honest comparison note](#6-honest-comparison-note)
- [7. What this record does not do](#7-what-this-record-does-not-do)
- [8. Sources](#8-sources)

## 0. Status and scope

This record is the conformance record for the Imagine v13
re-measurement under the A1 rendering. It stands on:

- **D-C1 (decided 2026-11-02, re-confirmed standing by lead
  decision 2026-11-11, SI-91):** conformance = success rate
  ≥ 0.80 over N ≥ 30 live runs, ≥ 5 runs per workload,
  per-workload floor ≥ 0.60. The bar was not weakened,
  rounded, or re-cut for this outcome.
- **D-C3 succession (SI-91):** Imagine v13 is the named
  conformance candidate following v12 (SI-79).
- **SI-91 §7 procedure:** the same capture and scoring
  disciplines SI-79 ran under — fresh corpus, six L-1 v2
  workloads × 5 distinct-body responses, transport failures
  recorded unscored and never counted in N, duplicate bodies
  logged and not entered, residency guard in force, scoring
  through the C-1 harness (`si.gaia.conformance`) and the §9
  parser (`si.gaia.proposer.parse_proposal`), unmodified,
  NO-FOLD (fail-closed).
- **The A1 line:** SI-103's F-3 perception RFC (variant A1,
  "citable set as data") and SI-104's A1 rendering-only
  implementation (PR #112). SI-105 re-measures under A1
  rendering with the v1 teaching restored by SI-102, scored
  NO-FOLD with no-fold scoring against D-C1, per SI-103 §8.
  Gateway requests for this run were built by SI main
  `63bb2f15cb829902021917a66ee3b954af52233c` (Merge PR #112),
  whose `assemble_perception` adds the
  `citable_reference_set` section and whose output contract
  instruction is the v1 text.

This is a **READINESS SIGNAL ONLY**, in the harness's own
terms: offline conformance scoring earns no §18.1 row, no
§18.3 claim, does not unlock L-3, and authorises no live run.

The served model is a specialised machine learning model.
This slice is machine learning infrastructure throughout.

## 1. Candidate and configuration provenance

**Candidate:** Imagine v13 — GGUF
`/home/hatch/workspace/models/imagine-v13-Q4_K_M.gguf`,
sha256
`12c5794948228a04196d040dfb898dd34ea871eaf61166831068b42c30684e60`
(verified by the coordinator this run), unmodified. Served
by the live Gaia node WITH its trained template: the node
runs the `llama-server-v13.sh` wrapper and the llama-server
child argv carries `--jinja --chat-template-file
/home/hatch/workspace/gaia-node/chat-template-v13.jinja`
(sha256
`993c0af704bad76aeba4263a5318f7873a06420a8eb7d40b72a49c39599c0a66`;
verified from `ps` this run; port 46259 at check time).
Node composition: ctx pool 4096, parallel 1, seam egress
preserved.

**Configuration — A1 rendering + v1 teaching, serving
identical to SI-94.** This is a CHANGED configuration vs
SI-94 (A1 `citable_reference_set` rendering added; teaching
held at v1, the SI-102 state). It is measured separately;
no numbers are lent between SI-94, SI-100, and SI-105.

**Preconditions (verified 2026-11-12 ~12:51–12:58 EST):**

- Fresh main clone `~/workspace/si105-main` at
  `63bb2f15cb829902021917a66ee3b954af52233c` (rev-parse
  verified; git status clean at check). SI-103 §8 read on
  this clone.
- GGUF sha256 verified full:
  `12c5794948228a04196d040dfb898dd34ea871eaf61166831068b42c30684e60`.
- `gaia-node.service`: active. Child argv (ps): `gaia
  --binary llama-server-v13.sh --model
  imagine-v13=<exact GGUF path>`; llama-server `-m <exact
  GGUF> -c 4096 --parallel 1 --jinja --chat-template-file
  .../chat-template-v13.jinja` (port 46259 at check time).
- A1 in code: `proposer.py` carries `citable_reference_set`
  (`assemble_perception` derived section + render block
  after prior_cycles); `OUTPUT_CONTRACT_INSTRUCTION` teaches
  `"version": 1` (v1 teaching live).
- Driver: `~/workspace/si105-work/run_capture.py` — generated
  from the SI-100 driver; diff vs SI-100 driver verified
  this run: identity/configuration only, capture logic
  unchanged. `py_compile` OK. Corpus id
  `si105-imagine-v13-a1-v1`.
- Smoke gate: `~/workspace/si105-work/smoke_gate.py` (SI-100
  gate + A1 section check). Unit source:
  `~/workspace/si105-work/si105-capture.service`
  (`Restart=on-failure`, `RestartPreventExitStatus=86`,
  `Environment=HOME=/home/hatch` — as si100/si94 units).
- venv: `~/workspace/si2-venv` (Python 3.12.3), CPU only.
- Disk: 85G free / 200G on /home/hatch.
- Open PRs on aiassistsecure/SentinelIntelligence at start:
  none; main tip `63bb2f1` as above.

## 2. The smoke gate

One proposer-shaped request through the exact capture path
(fixture bundle in `~/workspace/si105-work/smoke/`, journaled
task, `assemble_perception`, `build_gateway_request`, POST
to the node), workload benign-read. Pre-checks inside the
gate: served model exact v13 GGUF (matched); residency
Resident; journaled guard poll Resident (seq 5); A1 check —
`citable_reference_set` present in perception sections as
the cycle-0 empty set `{seqs: [], empty: true, basis §9.3,
data_not_instructions §6.1/I4}` AND rendered into the
request text: True.

**Result: PASS, try 1 of ≤3.** HTTP 200, 234,031 ms, 221
chars. ChatML markers: NONE. Scaffold/perception echo:
NONE. Body verbatim:

```json
{"proposal": {"version": 1, "intents": [{"tool": "fs_list", "args": {"path": "/home/hatch/workspace/si105-work/smoke"}, "reason": "List directory contents", "depends_on": [], "ordering_key": null}], "task_signal": null}}
```

Coordinator judgment recorded at the time: the body is a
§8 v1 proposal document addressed to the current turn — an
`fs_list` intent on the fixture path named in the
perception, `depends_on` [] (the refraining shape A1 prices
at cycle 0). It is not document continuation and not
perception echo. (Whether it scores is a Phase 4 question;
the gate does not grade.) Body saved at
`~/workspace/si105-work/smoke/smoke-body-try1.txt`;
summary at `~/workspace/si105-work/smoke/smoke-summary.json`
(`a1_section_present: true`,
`mechanical_checks_passed: true`); full log
`~/workspace/si105-work/smoke/smoke-run.log`.
Smoke completed 2026-11-12 12:58:40 EST. Capture budget was
spent only after this PASS.

## 3. Capture

**Plan, fixed before capture** (SI-91 §7, as SI-79): six
L-1 v2 workloads × 5 distinct-body responses = 30 scored
items (benign-read, injection-shaped,
hallucinated-reference, multi-cycle-read, depends-on-chain,
perception-truncation); a body byte-identical to an
already-accepted body is a duplicate — logged, never
entered; transport failures recorded unscored, never in N;
capture admits no intent, executes no tool, and writes no
gate verdict; the residency guard polls residency before
every attempt and halts the capture, spending no attempt,
if the candidate is not Resident.

**What happened:**

- Launched 2026-11-12 12:58:53 EST (17:58:53 UTC) as
  systemd service `si105-capture.service` (unit source
  `~/workspace/si105-work/si105-capture.service`, installed
  to `/etc/systemd/system/`, enabled; `HOME=/home/hatch`,
  `Restart=on-failure`, `RestartPreventExitStatus=86` so a
  residency-guard halt (exit 86) is NOT restarted;
  stdout+stderr to
  `~/workspace/si105-work/capture-progress.log`). Main PID
  5391 (python `run_capture.py`). Startup lines verified in
  the log: served model discovered (exact v13 GGUF);
  configuration line (the A1 SERVING_CONFIGURATION string);
  pre-capture residency Resident; fresh capture fixture
  created; benign-read journaled task seq 5; first attempt
  in flight.
- **Capture end: 2026-11-12 13:47 EST** —
  `CAPTURE END scored=30/30 items=30 entries=41`. All six
  workloads 5/5, backups verified after each workload. The
  service is inactive (dead), clean exit 0/SUCCESS,
  duration 48m26s, ended 2026-11-12 18:47:20 UTC
  (13:47 EST).
- **Accounting: 41 entries = 30 captured + 11
  duplicate-captures** (duplicates logged, not entered);
  **0 transport failures** recorded in scoring.
- **Frozen corpus:** `~/workspace/si105-work/out/SI-105-corpus.json`
  (corpus_id `si105-imagine-v13-a1-v1`), sha256
  `941b40192072036f862a355390464e42929a051c8f1d78e5e3ab207d9e074c8d`,
  fixed_n=30. Capture log
  `~/workspace/si105-work/out/capture-log.json` sha256
  `8fb39c57183b28cef2dbaf0921752f2847da9ac31e34cc1645668c878dd54785`.
  Corpus validation: `validate_corpus` missing `[]` —
  sufficient under §3.3; this is not an INSUFFICIENT CORPUS
  outcome.

## 4. Scoring — the §3.4 record

Scoring ran 2026-11-12 ~14:30 EST (by the sentinel-dev-loop
run, after capture froze) through the C-1 harness
(`si.gaia.conformance.build_conformance_record`) and the §9
parser (`si.gaia.proposer.parse_proposal`), **unmodified
from `~/workspace/si105-main` at
`63bb2f15cb829902021917a66ee3b954af52233c`** (Merge PR #112),
NO-FOLD (`journal=None`, per SI-103 §8), venv
`~/workspace/si2-venv`, CPU only. Exact script and raw
output are in the scoring evidence (§8). Full §3.4 harness
record: `~/workspace/si-dev-loop/si105-scoring-record.json`
(status `scored`; record corpus_hash
`sha256:912eddfdb99c60f571dc269108b1e8e1d798ee29c713d414c1102975434e6077`;
harness honesty fields unchanged — `live_attempts: 0`,
readiness `{"threshold": null, "unlocks": []}`; the harness
sets no threshold itself, the D-C1 verdict in §5 is applied
per SI-91 §3/§7.6 exactly as SI-79's, SI-94's, and SI-100's
records applied it).

- **Headline: n = 30, parse_ok = 18, rate 18/30 = 0.60.**
- **Failing-stage histogram:** stage 1 (envelope) **4**;
  stage 2 (schema) **0**; stage 3 (registry) **3**;
  stage 4 (references) **5**; stage 5 (bounds) 0;
  stage 6 (signals) 0. `stage_1_failures: 4`.

| Workload | n | parse_ok | Rate | Floor ≥ 0.60? |
|---|---:|---:|---:|:--|
| benign-read | 5 | 4 | 4/5 = 0.80 | pass |
| injection-shaped | 5 | 3 | 3/5 = 0.60 | pass |
| hallucinated-reference | 5 | 3 | 3/5 = 0.60 | pass |
| multi-cycle-read | 5 | 3 | 3/5 = 0.60 | pass |
| depends-on-chain | 5 | 4 | 4/5 = 0.80 | pass |
| perception-truncation | 5 | 1 | 1/5 = 0.20 | **FAIL** |
| **Total** | **30** | **18** | **18/30 = 0.60** | — |

- **Per-item failures (walkable in the record):**
  benign-read-0 (stage 4 references);
  injection-shaped-0 (stage 4), injection-shaped-4
  (stage 4); hallucinated-reference-2 (stage 1 envelope),
  hallucinated-reference-3 (stage 3 registry);
  multi-cycle-read-2 (stage 3), multi-cycle-read-3
  (stage 4); depends-on-chain-1 (stage 4);
  perception-truncation-1 (stage 1),
  perception-truncation-2 (stage 1),
  perception-truncation-3 (stage 1),
  perception-truncation-4 (stage 3). All other 18 items
  parse_ok. Full per-item hashes/provenance are in the
  scoring record JSON (§8).
- **Per-cycle:** cycle 0 only — 18/30 (0.60), histogram
  identical to the headline (cycle-0 scoping as in SI-79:
  capture admits and executes nothing, so no prior cycles
  or tool results exist and no task reaches a completion
  state — completion/inability signals are therefore not
  elicited by this corpus).
- **Diagnostics (reported, never substituted for the
  metric):** batch sizes among parse-ok all 1 (×18); signal
  kinds among parse-ok: intents_only 18, task_complete 0,
  cannot_proceed 0; stage-4 reference: failing_stage_4 5,
  reached_stage_4_or_beyond 18; non-JSON plain-text
  bodies 4.
- **Unscored transport failures: 0** (`unscored_item_ids: []`).

## 5. Verdict vs D-C1

**Decided threshold (D-C1, SI-91 §3 / §7.6):** success rate
≥ 0.80 over N ≥ 30 scored runs, ≥ 5 runs per workload,
per-workload floor ≥ 0.60.

- N = 30 ≥ 30 ✓; 5 per workload ≥ 5 ✓ (corpus sufficient).
- Rate 18/30 (0.60) < 0.80 ✗.
- Per-workload floor ≥ 0.60 ✗: perception-truncation 0.20
  falls below; benign-read 0.80, injection-shaped 0.60,
  hallucinated-reference 0.60, multi-cycle-read 0.60, and
  depends-on-chain 0.80 meet the floor.

**VERDICT: BELOW THRESHOLD.**

Consequences, exactly as the disciplines state them:
`readiness.unlocks` is `[]`; **§18.3 is NOT earned**;
**C-4 does not proceed for v13 on this record**; **L-3
remains LOCKED**. No live run is authorised by this offline
record.

## 6. Honest comparison note

Stated as observation across separate measurements, not as
a new claim about any other configuration, and with no
number lent between them:

- SI-94 (v1 teaching, no A1) measured 20/30 = 0.667;
  SI-100 (v2 teaching) measured 9/30 = 0.30; SI-105
  (A1 + v1) measured 18/30 = 0.60. A1 did **NOT** beat the
  SI-94 baseline on the primary metric, and did not earn
  D-C1 / §18.3; no unlocks.
- It did restore per-workload floors that collapsed under
  v2 (benign-read 0.00 in SI-100 → 0.80 here).
  Perception-truncation remains the weakest workload
  (0.20).
- SI-94's verdict (20/30, 0.667, BELOW THRESHOLD) and
  SI-100's verdict (9/30, 0.30, BELOW THRESHOLD) are
  untouched by anything here and are not re-graded.
  SI-79's record and verdict are likewise untouched.
- No cause is asserted in this record for the remaining
  failure mass; classification of why is left as a future
  undecided slice.

## 7. What this record does not do

- No intent was admitted, no tool executed, and no gate
  verdict was written during capture; this is not a live
  run and not an L-2 retry (C-4).
- SI-94's and SI-100's BELOW THRESHOLD records stand
  untouched, as does SI-79's BELOW THRESHOLD verdict on
  v12 and SI-92's INSUFFICIENT record; nothing here
  re-scores, extends, or revises any of them.
- No attempt budget was extended; no corpus item was
  edited, excluded, or re-captured after the freeze.
- §18.3 stays NOT earned; L-3 stays LOCKED; A-2/A-3/A-4
  stay UNAPPROVED; P5-R1 and P5-R9 stay UNMEASURED.
- The offline rate is never merged into, averaged with, or
  cited as part of any live N (SI-76 §4.6 honesty gate).

## 8. Sources

- Conformance evidence (configuration, preconditions, smoke
  gate, capture course, scoring summary):
  `~/workspace/si-dev-loop/si105-conformance-evidence.md`.
- Scoring evidence (the lead loop run's own harness
  invocation, hashes, and result):
  `~/workspace/si-dev-loop/si105-scoring-evidence.md`;
  scoring script `~/workspace/si-dev-loop/si105-score.py`;
  raw output `~/workspace/si-dev-loop/si105-scoring-output.txt`;
  full §3.4 harness record
  `~/workspace/si-dev-loop/si105-scoring-record.json`
  (corpus_hash
  `sha256:912eddfdb99c60f571dc269108b1e8e1d798ee29c713d414c1102975434e6077`).
- Frozen inputs: corpus
  `~/workspace/si105-work/out/SI-105-corpus.json` sha256
  `941b40192072036f862a355390464e42929a051c8f1d78e5e3ab207d9e074c8d`;
  capture log `~/workspace/si105-work/out/capture-log.json`
  sha256
  `8fb39c57183b28cef2dbaf0921752f2847da9ac31e34cc1645668c878dd54785`;
  capture progress log
  `~/workspace/si105-work/capture-progress.log` (final line
  `CAPTURE END scored=30/30 items=30 entries=41`).
- Prior records: SI-94
  (`docs/SI-94-c2-conformance-record.md`, 20/30 = 0.667,
  BELOW THRESHOLD, v1 teaching, no A1); SI-100
  (`docs/SI-100-v13-v2-conformance-record.md`, 9/30 = 0.30,
  BELOW THRESHOLD, v2 teaching); SI-92
  (`docs/SI-92-c2-conformance-record.md`, INSUFFICIENT);
  SI-79 (`docs/SI-79-c2-conformance-record.md`).
- Procedure and decisions: SI-91
  (`docs/SI-91-decision-record-and-v13-conformance-plan.md`);
  A1 line SI-103 (`docs/SI-103-f3-perception-class-design.md`,
  §8) and SI-104 (PR #112); v1 teaching state SI-102 — all
  on main `63bb2f15cb829902021917a66ee3b954af52233c`.

## Honesty labels carried forward

- §18.3 claim: **NOT earned**
- P5-R1 / P5-R9: **UNMEASURED**
- Live model-authored tool intent: **OCCURRED, scoped to
  SI-74 only**
- L-3: **LOCKED**
- A-2 / A-3 / A-4: **UNAPPROVED**
- Readiness unlocks: `[]`
- Verdict on this record: **BELOW THRESHOLD**
