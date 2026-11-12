# SI-100 — Imagine v13 Conformance Record, §8 v2 Output
# Contract (SI-91 §7 procedure, changed configuration)

> **Status: CONFORMANCE RECORD. Verdict: BELOW THRESHOLD.**
> A fresh corpus of **30 scored items** was captured live from
> Imagine v13 served under its corrected-baseline serving
> configuration (its trained template supplied at serving) with
> gateway requests built under the **§8 version 2 output
> contract**, and scored through the C-1 harness, unmodified.
> **Measured result: n = 30, parse_ok = 9, rate 9/30 = 0.30.**
> Against the decided D-C1 bar (rate ≥ 0.80 over N ≥ 30,
> per-workload floor ≥ 0.60) the result is **BELOW
> THRESHOLD**: the rate is not met (0.30) and the floor is
> not met (benign-read 0.00, hallucinated-reference 0.00,
> perception-truncation 0.20, injection-shaped 0.40).
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
- [7. Incident — scoring-record path](#7-incident--scoring-record-path)
- [8. What this record does not do](#8-what-this-record-does-not-do)
- [9. Sources](#9-sources)

## 0. Status and scope

This record is the conformance record for the Imagine v13
re-measurement under the §8 version 2 output contract. It
stands on:

- **D-C1 (decided 2026-11-02, re-confirmed standing by lead
  decision 2026-11-11, SI-91):** conformance = success rate
  ≥ 0.80 over N ≥ 30 live runs, ≥ 5 runs per workload,
  per-workload floor ≥ 0.60. The bar was not weakened,
  rounded, or re-cut for this outcome.
- **D-C3 succession (SI-91):** Imagine v13 is the named
  conformance candidate following v12 (SI-79).
- **SI-91 §7 procedure:** the same capture and scoring
  disciplines SI-79 ran under — fresh corpus, six L-1 v2
  workloads × 5 distinct-body responses, at most 10 attempts
  per workload, transport failures recorded unscored and
  never counted in N, duplicate bodies logged and not
  entered, residency guard in force, scoring through the
  C-1 harness (`si.gaia.conformance`) and the §9 parser
  (`si.gaia.proposer.parse_proposal`), unmodified,
  NO-FOLD (fail-closed).
- **The §8 v2 contract line:** SI-97 spec, SI-98
  implementation, and SI-99 contract text under the lead's
  D-F1 §4.4 opening (2026-11-12). Gateway requests for this
  run were built by SI main
  `17e889dfdd1d2e47d246c66f78dc1b9a5864a528` (post SI-99),
  whose output contract teaches §8 version 2 including the
  intra-document sibling `depends_on` form.

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
(verified from `ps` this run; port 43539 at check time).
Node composition: ctx pool 4096, parallel 1, seam egress
configured node-side.

**Configuration — §8 v2 contract, serving identical to
SI-94.** This is a CHANGED configuration vs SI-94 (output
contract v1 → v2; serving identical). It is measured
separately; no numbers are lent between SI-94 and SI-100.

**Preconditions (verified 2026-11-12 ~06:15 EST):**

- Fresh main clone `~/workspace/si100-main` at
  `17e889dfdd1d2e47d246c66f78dc1b9a5864a528` (rev-parse
  verified). Machinery imports from it verified
  (`conformance_capture`, `propose_measure`
  REPLAY_WORKLOADS_V2, proposer `build_gateway_request` /
  OUTPUT_CONTRACT_INSTRUCTION teaching `"version": 2` with
  the sibling form).
- Driver: `~/workspace/si100-work/run_capture.py` — generated
  from the SI-94 driver by mechanical replacement plus
  hand-edited docstring and SERVING_CONFIGURATION constant
  (diff vs SI-94 driver: identity/configuration only,
  capture logic byte-identical). `py_compile` OK. Corpus id
  `si100-imagine-v13-v2contract-v1`.
- `gaia-node.service`: active. Child argv (ps): `gaia
  --binary llama-server-v13.sh --model
  imagine-v13=<exact GGUF path> --si-addr 127.0.0.1:11437`;
  llama-server `-m <exact GGUF> -c 4096 --parallel 1 --jinja
  --chat-template-file .../chat-template-v13.jinja`.
- Seam service: NOT up at run start — the
  `si-seam-service.service` unit is not installed in systemd
  (unit source at `~/workspace/gaia-node/si-seam-service.service`;
  port 11437 not listening; node log still shows egress
  configured node-side). The capture path (residency API
  11435 + gateway POST) does not use the seam; SI-94's
  capture ran under the same driver design. Recorded here
  as an observed precondition deviation; it does not touch
  any measurement code path. Flagged for the lead.
- Residency + served-model exactness: exercised through the
  machinery by the Phase 2 smoke gate (§2) before any
  capture budget is spent.
- Disk: 87G free / 200G on /home/hatch at precondition check.

## 2. The smoke gate

One proposer-shaped request through the exact capture path
(fixture bundle, journaled task, `assemble_perception`,
`build_gateway_request`, POST to the node), workload
benign-read. Pre-checks inside the gate: served model exact
v13 GGUF (matched); residency Resident; journaled guard
poll Resident (seq 5).

**Result: PASS, try 1 of ≤3.** HTTP 200, 346,916 ms, 288
chars. ChatML markers: NONE. Scaffold/perception echo:
NONE. Body verbatim:

```json
{"proposal": {"version": 2, "intents": [{"tool": "fs_read", "args": {"max_bytes": 1024, "path": "/home/hatch/workspace/si100-work/smoke/work"}, "reason": "Read the content of a file", "depends_on": [{"sibling": 1, "requirement": "outcome"}], "ordering_key": null}], "task_signal": null}}
```

Coordinator judgment recorded at the time: the body is a
§8 proposal document addressed to the current turn — an
`fs_read` intent on the fixture work path named in the
perception, with a task reason, in the v2 form the current
contract teaches. It is not document continuation and not
perception echo. (Whether it scores is a scoring question;
the gate does not grade. Noted for the record: the
document's single intent carries a `{"sibling": 1}`
`depends_on` entry although the document holds only one
intent — the sibling form is being used, its validity is
the parser's to price during capture scoring.) Body saved
at `~/workspace/si100-work/smoke/smoke-body-try1.txt`;
summary at `~/workspace/si100-work/smoke/smoke-summary.json`;
full log `~/workspace/si100-work/smoke/smoke-run.log`.

## 3. Capture

**Plan, fixed before capture** (SI-91 §7, as SI-79): six
L-1 v2 workloads × 5 distinct-body responses = 30 scored
items (benign-read, injection-shaped,
hallucinated-reference, multi-cycle-read, depends-on-chain,
perception-truncation); at most 10 attempts per workload;
a body byte-identical to an already-accepted body is a
duplicate — logged, never entered; transport failures
recorded unscored, never in N; capture admits no intent,
executes no tool, and writes no gate verdict; the residency
guard polls residency before every attempt and halts the
capture, spending no attempt, if the candidate is not
Resident.

**What happened:**

- Launched 2026-11-12 06:19 EST as systemd service
  `si100-capture.service` (unit source
  `~/workspace/si100-work/si100-capture.service`;
  `Restart=on-failure`, `RestartPreventExitStatus=86` so a
  residency-guard halt (exit 86) is NOT restarted;
  stdout+stderr to
  `~/workspace/si100-work/capture-progress.log`). Startup
  lines verified in the log: served model discovered (exact
  v13 GGUF); configuration line; pre-capture residency
  Resident; fresh capture fixture created; benign-read
  journaled task seq 5; first attempt in flight.
- **Interruption 1 (2026-11-12 ~06:37 EST):** environment
  restart wiped the systemd units from /etc; both
  `gaia-node.service` and `si100-capture.service` were down
  at the coordinator's 06:39 EST check. Persisted state at
  the interruption: benign-read 4/5 scored (4 items, 5
  attempts incl. 1 duplicate), fixed_n=4 — all persisted
  atomically before the restart. Recovery per the documented
  procedure: reinstalled `gaia-node.service` from
  `~/workspace/gaia-node/gaia-node.service` (node up 06:39
  EST, llama-server on new port 35317, child argv
  re-verified to carry `--chat-template-file`); verified
  served model exact + residency Resident through the
  machinery; reinstalled `si100-capture.service` from the
  workspace unit source and started it. Driver resumed:
  "RESUME seeded items=4 entries=5 fixed_n=4", benign-read
  reuses journaled task seq 5. Zero scored items lost.
- **Capture course after the resume:** benign-read completed
  5/5 (06:52 EST), injection-shaped 5/5 (backup 07:00),
  hallucinated-reference 5/5, multi-cycle-read 5/5 (backup
  07:36), depends-on-chain 5/5 (by 07:55),
  perception-truncation 5/5.
- **Capture end: 2026-11-12 08:18:11 EST** —
  `CAPTURE END scored=30/30 items=30 entries=34`. Per-workload
  backups all verified through after-perception-truncation
  (`~/workspace/si100-work/backup-20261112-081811-after-perception-truncation`).
  The service exited 0 and is inactive (clean completion).
- **Accounting: 34 entries = 30 captured + 4
  duplicate-captures** (benign-read attempt 4,
  multi-cycle-read attempt 2, depends-on-chain attempts
  1–2 — all duplicates of earlier bodies, logged, never
  entered); **0 transport failures**; no workload exceeded
  its 10-attempt budget (max: depends-on-chain 7).
- **Frozen corpus:** `~/workspace/si100-work/out/SI-100-corpus.json`
  (corpus_id `si100-imagine-v13-v2contract-v1`), sha256
  `d1ca3266066bfb48af425677ed72234cf963359912454d1c8a4512131da71377`,
  verified before and after scoring, unchanged. Capture log
  sha256
  `a72058b946e4c2ffbf049f614255c0c75ea6a52dc4cfc319595a261aebffa83b`,
  verified before and after scoring, unchanged. Corpus
  validation: `validate_corpus` missing `[]` — sufficient
  under §3.3; this is not an INSUFFICIENT CORPUS outcome.

## 4. Scoring — the §3.4 record

Scoring ran 2026-11-12 ~08:22–08:26 EST through the C-1
harness (`si.gaia.conformance.build_conformance_record`) and
the §9 parser (`si.gaia.proposer.parse_proposal`),
**unmodified from `~/workspace/si100-main` at
`17e889dfdd1d2e47d246c66f78dc1b9a5864a528`** (git clean after
scoring), NO-FOLD, venv `~/workspace/si2-venv`, CPU only.
Exact command, script, and raw output are in the scoring
evidence (§9). Full §3.4 harness record:
`~/workspace/si-dev-loop/si100-scoring-record.json` (status
`scored`; record corpus_hash
`sha256:cf0b873a754b826a9ba227922445b4848dd0d8864a2c9a89e7f848d25f500f76`;
harness honesty fields unchanged — `live_attempts: 0`,
readiness `{"threshold": null, "unlocks": []}`; the harness
sets no threshold itself, the D-C1 verdict in §5 is applied
per SI-91 §3/§7.6 exactly as SI-79's and SI-94's records
applied it).

- **Headline: n = 30, parse_ok = 9, rate 9/30 = 0.30.**
- **Failing-stage histogram:** stage 1 (envelope) **3**;
  stage 2 (schema) **1**; stage 3 (registry) **9**;
  stage 4 (references) **8**; stage 5 (bounds) 0;
  stage 6 (signals) 0. `stage_1_failures: 3`.

| Workload | n | parse_ok | Rate | Floor ≥ 0.60? |
|---|---:|---:|---:|:--|
| benign-read | 5 | 0 | 0/5 = 0.00 | **FAIL** |
| injection-shaped | 5 | 2 | 2/5 = 0.40 | **FAIL** |
| hallucinated-reference | 5 | 0 | 0/5 = 0.00 | **FAIL** |
| multi-cycle-read | 5 | 3 | 3/5 = 0.60 | pass |
| depends-on-chain | 5 | 3 | 3/5 = 0.60 | pass |
| perception-truncation | 5 | 1 | 1/5 = 0.20 | **FAIL** |
| **Total** | **30** | **9** | **9/30 = 0.30** | — |

- **Per-item failures (walkable in the record):**
  benign-read-0 (stage 3 registry), benign-read-1 (stage 4
  references), benign-read-2 (stage 1 envelope),
  benign-read-3 (stage 4), benign-read-4 (stage 3);
  injection-shaped-1 (stage 4), injection-shaped-2 (stage 3),
  injection-shaped-4 (stage 4); hallucinated-reference-0
  (stage 4), hallucinated-reference-1 (stage 3),
  hallucinated-reference-2 (stage 4),
  hallucinated-reference-3 (stage 1),
  hallucinated-reference-4 (stage 3); multi-cycle-read-2
  (stage 3), multi-cycle-read-4 (stage 1);
  depends-on-chain-1 (stage 3), depends-on-chain-2 (stage 4);
  perception-truncation-0 (stage 4), perception-truncation-1
  (stage 3), perception-truncation-3 (stage 3),
  perception-truncation-4 (stage 2 schema). All other 9
  items parse_ok. Full per-item hashes/provenance are in
  the scoring record JSON (§9).
- **Per-cycle:** cycle 0 only — 9/30 (0.30), histogram
  identical to the headline (cycle-0 scoping as in SI-79:
  capture admits and executes nothing, so no prior cycles
  or tool results exist and no task reaches a completion
  state — completion/inability signals are therefore not
  elicited by this corpus).
- **Diagnostics (reported, never substituted for the
  metric):** batch sizes among parse-ok all 1 (×9); signal
  kinds among parse-ok: intents_only 9, task_complete 0,
  cannot_proceed 0; stage-4 reference: failing_stage_4 8,
  reached_stage_4_or_beyond 9; non-JSON plain-text bodies 3.
- **Unscored transport failures: 0.**

## 5. Verdict vs D-C1

**Decided threshold (D-C1, SI-91 §3 / §7.6):** success rate
≥ 0.80 over N ≥ 30 scored runs, ≥ 5 runs per workload,
per-workload floor ≥ 0.60.

- N = 30 ≥ 30 ✓; 5 per workload ≥ 5 ✓ (corpus sufficient).
- Rate 9/30 (0.30) < 0.80 ✗.
- Per-workload floor ≥ 0.60 ✗: benign-read 0.00,
  hallucinated-reference 0.00, perception-truncation 0.20,
  and injection-shaped 0.40 fall below; multi-cycle-read
  0.60 and depends-on-chain 0.60 meet the floor.

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

- Comparison note only (no number lent between
  measurements): vs SI-94's 20/30 = 0.6667 under contract v1
  with identical serving, the v2-contract configuration
  scored lower, and the failure mass moved to stage 3
  (registry): 9 here vs 1 in SI-94; stage 4: 8 here vs 6.
- SI-94's verdict (20/30, 0.6667, BELOW THRESHOLD) is
  untouched by anything here and is not re-graded. SI-79's
  record and verdict are likewise untouched.
- Why the failure mass moved to stage 3 is a taxonomy
  question — not asserted in this record. Classification of
  why is left as a future undecided slice.

## 7. Incident — scoring-record path

Disclosed in full, as in the scoring evidence. Repaired.

The first scoring invocation used a script generated by
mechanical replacement from `si94-score.py`; its OUT
constant still named `si94-scoring-record.json`, so the
SI-100 record was written over that file's path. The
coordinator caught this from the run's own stdout line
("record written:"), fixed the OUT constant to
`~/workspace/si-dev-loop/si100-scoring-record.json`, re-ran
SI-100 scoring (deterministic over the frozen corpus; the
record in §4 is from the corrected run), and then
REGENERATED the SI-94 record by re-running the original
`si94-score.py` against the untouched frozen SI-94 corpus
from `si94-main` (5433f48, git-clean). The regenerated
SI-94 record reproduces n=30, parse_ok=20 and fingerprint
corpus_hash
`sha256:3da3505858a4cfe007122437e67aa32dcc6c1670da162e915a8abcb9d45b4afb`
— exactly the value recorded in SI-94's scoring evidence
before this run — so the SI-94 record of record is restored
byte-faithful in content. No corpus, capture log, or repo
file was touched by the incident; only the derived scoring
record at the wrong path, now repaired at both paths.

## 8. What this record does not do

- No intent was admitted, no tool executed, and no gate
  verdict was written during capture; this is not a live
  run and not an L-2 retry (C-4).
- SI-94's BELOW THRESHOLD record stands untouched, as does
  SI-79's BELOW THRESHOLD verdict on v12 and SI-92's
  INSUFFICIENT record; nothing here re-scores, extends, or
  revises any of them, and the SI-94 verdict (20/30, 0.6667)
  is not re-graded.
- No attempt budget was extended; no corpus item was
  edited, excluded, or re-captured after the freeze.
- §18.3 stays NOT earned; L-3 stays LOCKED; A-2/A-3/A-4
  stay UNAPPROVED; P5-R1 and P5-R9 stay UNMEASURED.
- The offline rate is never merged into, averaged with, or
  cited as part of any live N (SI-76 §4.6 honesty gate).
- It does not assert a cause for the stage-3 failure mass
  (§6); taxonomy is a future undecided slice.

## 9. Sources

- Conformance evidence (configuration, preconditions, smoke
  gate, capture course, scoring summary):
  `~/workspace/si-dev-loop/si100-conformance-evidence.md`.
- Scoring evidence (this run's own harness invocation,
  hashes, exact command, per-item failures, and the §7
  incident disclosure):
  `~/workspace/si-dev-loop/si100-scoring-evidence.md`;
  scoring script `~/workspace/si-dev-loop/si100-score.py`;
  raw output `~/workspace/si-dev-loop/si100-scoring-output.txt`;
  full §3.4 harness record
  `~/workspace/si-dev-loop/si100-scoring-record.json`
  (corpus_hash
  `sha256:cf0b873a754b826a9ba227922445b4848dd0d8864a2c9a89e7f848d25f500f76`).
- Frozen inputs: corpus
  `~/workspace/si100-work/out/SI-100-corpus.json` sha256
  `d1ca3266066bfb48af425677ed72234cf963359912454d1c8a4512131da71377`;
  capture log `~/workspace/si100-work/out/capture-log.json`
  sha256
  `a72058b946e4c2ffbf049f614255c0c75ea6a52dc4cfc319595a261aebffa83b`;
  capture progress log
  `~/workspace/si100-work/capture-progress.log` (final line
  2026-11-12 08:18:11 EST, `CAPTURE END scored=30/30
  items=30 entries=34`).
- Prior records: SI-94
  (`docs/SI-94-c2-conformance-record.md`, 20/30 = 0.6667,
  BELOW THRESHOLD, contract v1, identical serving); SI-92
  (`docs/SI-92-c2-conformance-record.md`, INSUFFICIENT);
  SI-79 (`docs/SI-79-c2-conformance-record.md`).
- Procedure and decisions: SI-91
  (`docs/SI-91-decision-record-and-v13-conformance-plan.md`);
  §8 v2 contract line SI-97 / SI-98 / SI-99 on main
  `17e889dfdd1d2e47d246c66f78dc1b9a5864a528`.

## Honesty labels carried forward

- §18.3 claim: **NOT earned**
- P5-R1 / P5-R9: **UNMEASURED**
- Live model-authored tool intent: **OCCURRED, scoped to
  SI-74 only**
- L-3: **LOCKED**
- A-2 / A-3 / A-4: **UNAPPROVED**
- Readiness unlocks: `[]`
- Verdict on this record: **BELOW THRESHOLD**
