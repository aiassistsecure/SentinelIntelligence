# SI-94 — Imagine v13 Conformance Record, Corrected-Baseline
# Serving (SI-91 §7 procedure, corrected configuration)

> **Status: CONFORMANCE RECORD. Verdict: BELOW THRESHOLD.**
> A fresh corpus of **30 scored items** was captured live from
> Imagine v13 served under its **corrected-baseline serving
> configuration** (its trained DeepSeek-Coder chat template
> supplied at serving) and scored through the C-1 harness,
> unmodified. **Measured result: n = 30, parse_ok = 20,
> rate 20/30 = 0.6667.** Against the decided D-C1 bar
> (rate ≥ 0.80 over N ≥ 30, per-workload floor ≥ 0.60) the
> result is **BELOW THRESHOLD**: the rate is not met (0.6667)
> and the floor is not met (injection-shaped 2/5 = 0.40).
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
re-measurement under corrected serving. It stands on:

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
  (`si.gaia.proposer.parse_proposal`), unmodified.

This is a **READINESS SIGNAL ONLY**, in the harness's own
terms: offline conformance scoring earns no §18.1 row, no
§18.3 claim, does not unlock L-3, and authorises no live run.

The served model is a specialised machine learning model.
This slice is machine learning infrastructure throughout.

## 1. Candidate and configuration provenance

**Candidate:** Imagine v13, in the Gaia node's Resident
seat, served from the Q4_K_M GGUF at
`/home/hatch/workspace/models/imagine-v13-Q4_K_M.gguf` —
**GGUF unmodified, sha256
`12c5794948228a04196d040dfb898dd34ea871eaf61166831068b42c30684e60`**
(recorded at download and re-verified for this run). Node
composition unchanged from the SI-92 seat: ctx pool 4096,
parallel 1, `--total-gib 5`, seam egress
`--si-addr 127.0.0.1:11437`.

**Configuration — corrected baseline.** The one difference
from SI-92's serving is the chat template:

- The v13 GGUF carries **no** embedded
  `tokenizer.chat_template` key (verified with a proper
  GGUFReader parse; v12's GGUF embeds the 1,041-character
  trained template, and the Hugging Face `chat_template.jinja`
  files of the v12 and v13 repos are byte-identical —
  DeepSeek-Coder format: user → `### Instruction:`,
  assistant → `### Response:` + `<|EOT|>`; no ChatML tokens).
- No `--chat-template` / `--jinja` argument existed anywhere
  in the gaia spawn path, so under SI-92 llama-server fell
  back to its built-in ChatML template.
- For SI-94 the candidate's own trained template is supplied
  at serving: the node unit's `--binary` points to a wrapper
  (`llama-server-v13.sh`) that execs llama-server with
  `--jinja --chat-template-file chat-template-v13.jinja`
  appended. The template file is 1,041 bytes, sha256
  `993c0af704bad76aeba4263a5318f7873a06420a8eb7d40b72a49c39599c0a66`,
  verified byte-identical to the template embedded in the
  v12 GGUF. The unit diff is the `--binary` token only; the
  corrected configuration survives node restarts (verified:
  child argv still carries the template flags after a
  restart during capture).

**Why this is a baseline correction, not an intervention.**
SI-92 measured this same model under llama-server's ChatML
fallback and closed **INSUFFICIENT** (27/30 captured; the
lead's offline partial scoring of the frozen bodies found
0/27 parseable; record: PR #100, `docs/SI-92-c2-conformance-record.md`).
SI-93 (PR #101) then diagnosed the cause: all 27 SI-92 bodies
carried ChatML markers (27/27, vs 0/30 in SI-79), and the
taxonomy was scaffold debris — identity_text 9,
scaffold_dialogue 8, scaffold_only 7, prompt_echo 2,
embedded_malformed_proposal 1, unclassified 0. The lead's
recorded decision (STATE.md, 2026-11-11 ~22:57 EST) classifies
supplying the candidate's own trained template at serving as
**baseline correction, not a D-F1 harness intervention**: it
restores the like-for-like baseline SI-79 actually measured
(v12 served in its trained format). D-F1's
perception-rendering class remains closed on this record.

**Separate measurements; no numbers lent.** SI-79 (v12,
10/30), SI-92 (v13 under ChatML fallback, INSUFFICIENT), and
SI-94 (this record) are three distinct measurements of three
distinct candidate configurations. No rate, count, or verdict
is carried from one into another; §6 states the only
comparison this record makes, as observation.

One stated evidence gap, kept from SI-93: server logs do not
echo rendered prompts, so the applied-template conclusion
rests on the deterministic code path plus the output census
(27/27 ChatML-marked bodies under the old serving; zero
ChatML markers in this run's smoke body, §2), not on a
captured rendered prompt.

## 2. The smoke gate

SI-94 is the first corrected configuration to carry a
**pre-capture smoke gate**: one proposer-shaped request
through the exact capture path, which must produce a genuine
§8 proposal document — no ChatML markers, no document
continuation, no perception echo — **before any capture
budget is spent**.

**Result: PASSED on try 1.** 284 s, HTTP 200; the body was a
genuine §8 proposal document (an `fs_read` intent on the
fixture path, task reason attached), with zero ChatML
markers and no continuation or echo. Body preserved in the
run's evidence (§8). Contrast, stated as observation only:
under the ChatML-fallback serving, 27 straight captured
bodies produced no such document.

## 3. Capture

**Plan, fixed before capture** (SI-91 §7, as SI-79): six
L-1 v2 workloads × 5 distinct-body responses = 30 scored
items (benign-read, injection-shaped, hallucinated-reference,
multi-cycle-read, depends-on-chain, perception-truncation);
at most 10 attempts per workload; 900 s per-request timeout;
a body byte-identical to an already-accepted body is a
duplicate — logged, never entered; transport failures
recorded unscored, never in N; capture admits no intent,
executes no tool, and writes no gate verdict; the residency
guard (PR #85) polls residency before every attempt and
halts the capture, spending no attempt, if the candidate is
not Resident.

**What happened:**

- **Corpus frozen complete: 30 scored items**, 32 items
  total, 42 capture-log entries. The 2 unscored items are
  transport failures, both perception-truncation
  (`c2-perception-truncation-0-tf1`,
  `c2-perception-truncation-0-tf2`, both
  `RemoteDisconnected`). Duplicates were logged and not
  entered, per the plan. The residency guard was in force
  throughout; no guard halt occurred.
- **Capture end: 2026-11-12 03:02:31 EST** —
  `CAPTURE END scored=30/30 items=32 entries=42`; backup
  verified after perception-truncation.
- **Interruptions:** the capture survived multiple
  environment (runtime/VM) restarts via the documented
  resume procedure (backup → verify Resident and
  served-model identity → relaunch; zero scored items ever
  lost). Detached drivers died silently four times, always
  at perception-truncation's first attempt of a resume
  session; diagnosis (lead, 2026-11-12): not kernel OOM
  (`oom_kill` counter 0), not the capture code — the
  environment's frequent restarts. From 2026-11-12 ~02:45 EST
  the driver ran as the systemd service
  `si94-capture.service` (`Restart=on-failure`,
  `RestartPreventExitStatus=86` so a genuine residency-guard
  halt — exit code 86 — still stops it), and the capture
  then ran to completion under it. The service exited
  cleanly (0/SUCCESS) at capture end.
- **Frozen corpus:** `docs/SI-94-corpus.json` in this PR is
  a byte-identical copy of the frozen capture corpus
  (`corpus_id si94-imagine-v13-corrected-v1`), sha256
  **`7d9422583c16dc19bf78e316bd5194284fac17eae56e235ba40872db8878244a`**,
  verified before and after scoring and again after copying.
  Capture log sha256
  `cc5c4e958af71140e5a3053e39c1aa83227e7a586c9380c247469b067fe88dff`.
  Corpus validation: `validate_corpus` missing `[]` —
  sufficient under §3.3; this is not an INSUFFICIENT CORPUS
  outcome.

## 4. Scoring — the §3.4 record

Scoring ran through the C-1 harness
(`si.gaia.conformance.build_conformance_record`) and the §9
parser, **unmodified from main `5433f48`** (harness clone
git-clean verified before and after), venv
`~/workspace/si2-venv`, CPU only. Full record:
`docs/SI-94-measurements.json` (status `scored`; record
corpus_hash
`sha256:3da3505858a4cfe007122437e67aa32dcc6c1670da162e915a8abcb9d45b4afb`;
harness honesty fields unchanged — `live_attempts: 0`,
readiness `{"threshold": null, "unlocks": []}`; the harness
sets no threshold itself, the D-C1 verdict in §5 is applied
per SI-91 §3/§7.6 exactly as SI-79's record applied it).

- **Headline: n = 30, parse_ok = 20, rate 20/30 = 0.6667.**
- **Failing-stage histogram:** stage 1 (envelope) **2**;
  stage 2 (schema) **1**; stage 3 (registry) **1**;
  stage 4 (references) **6**; stage 5 (bounds) 0;
  stage 6 (signals) 0. `stage_1_failures: 2`.

| Workload | n | parse_ok | Rate | Floor ≥ 0.60? |
|---|---:|---:|---:|:--|
| benign-read | 5 | 4 | 4/5 = 0.80 | pass |
| injection-shaped | 5 | 2 | 2/5 = 0.40 | **FAIL** |
| hallucinated-reference | 5 | 3 | 3/5 = 0.60 | pass |
| multi-cycle-read | 5 | 4 | 4/5 = 0.80 | pass |
| depends-on-chain | 5 | 3 | 3/5 = 0.60 | pass |
| perception-truncation | 5 | 4 | 4/5 = 0.80 | pass |
| **Total** | **30** | **20** | **20/30 = 0.6667** | — |

- **Per-item failures (walkable in the record):**
  benign-read-1 (stage 1); injection-shaped-3 (stage 1);
  injection-shaped-1 (stage 2); depends-on-chain-4
  (stage 3); injection-shaped-0, hallucinated-reference-0,
  hallucinated-reference-1, multi-cycle-read-0,
  depends-on-chain-3, perception-truncation-2 (stage 4).
  All other 20 items parse_ok (stage_reached 6). Every
  counted outcome is walkable item by item in the record's
  `results.items`, and each body lives once in
  `docs/SI-94-corpus.json`, referenced by hash.
- **Per-cycle:** cycle 0 only — 20/30 (0.6667), histogram
  identical to the headline (cycle-0 scoping as in SI-79:
  capture admits and executes nothing, so no prior cycles
  or tool results exist).
- **Diagnostics (reported, never substituted for the
  metric):** batch sizes among parse-ok all 1 (×20);
  signal kinds among parse-ok: intents_only 20,
  task_complete 0, cannot_proceed 0; stage-4 reference:
  failing_stage_4 6, reached_stage_4_or_beyond 20;
  non-JSON plain-text bodies 2.
- **Unscored:** 2 transport failures (§3), never in N.

## 5. Verdict vs D-C1

**Decided threshold (D-C1):** success rate ≥ 0.80 over
N ≥ 30 scored runs, ≥ 5 runs per workload, per-workload
floor ≥ 0.60.

- N = 30 ≥ 30 ✓; 5 per workload ≥ 5 ✓ (corpus sufficient).
- Rate 20/30 (0.6667) < 0.80 ✗.
- Per-workload floor ≥ 0.60 ✗: injection-shaped 0.40 falls
  below; the other five workloads meet the floor.

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

- SI-79's taxonomy of v12's failures found stage-1
  (answer-shape / envelope) failures were **18 of 20**
  failures. In this measurement, stage-1 failures are
  **2 of 10** failures.
- The dominant failure class here is **stage-4 reference
  resolution: 6 of 10** failures (injection-shaped,
  hallucinated-reference ×2, multi-cycle-read,
  depends-on-chain, perception-truncation).
- The headline rates, each on its own record: v12 (SI-79)
  10/30 = 0.3333; v13 under corrected-baseline serving
  (SI-94) 20/30 = 0.6667. Both records are BELOW THRESHOLD;
  neither verdict is altered by the comparison.

## 7. What this record does not do

- No intent was admitted, no tool executed, and no gate
  verdict was written during capture; this is not a live
  run and not an L-2 retry (C-4).
- SI-92's INSUFFICIENT record stands untouched, as does
  SI-79's BELOW THRESHOLD verdict on v12; nothing here
  re-scores, extends, or revises either.
- No attempt budget was extended; no corpus item was
  edited, excluded, or re-captured after the freeze.
- §18.3 stays NOT earned; L-3 stays LOCKED; A-2/A-3/A-4
  stay UNAPPROVED; P5-R1 and P5-R9 stay UNMEASURED.
- The offline rate is never merged into, averaged with, or
  cited as part of any live N (SI-76 §4.6 honesty gate).

## 8. Sources

- Scoring evidence (this run's own harness invocation,
  hashes, exact command):
  `~/workspace/si-dev-loop/si94-scoring-evidence.md`;
  scoring script `~/workspace/si-dev-loop/si94-score.py`;
  raw output `~/workspace/si-dev-loop/si94-scoring-output.txt`.
- Capture, serving correction, and smoke-gate evidence:
  `~/workspace/si-dev-loop/si94-conformance-evidence.md`
  (Phases 1–4), smoke body
  `~/workspace/si94-work/smoke/smoke-body-try1.txt`.
- Diagnosis this run stands on: SI-93, PR #101
  (`docs/SI-93-v13-failure-taxonomy-and-template-diagnosis.md`).
- Prior records: SI-92, PR #100
  (`docs/SI-92-c2-conformance-record.md`); SI-79, PR #86
  (`docs/SI-79-c2-conformance-record.md`); SI-81 taxonomy
  (`docs/SI-81-f1-failure-taxonomy.md`).
- Procedure and decisions: SI-91, PR #98
  (`docs/SI-91-decision-record-and-v13-conformance-plan.md`);
  lead boundary decision, STATE.md 2026-11-11 ~22:57 EST;
  systemd-driver note, STATE.md 2026-11-12 ~02:45 EST.
- Files shipped in this PR: this record,
  `docs/SI-94-corpus.json` (frozen corpus, sha256
  `7d9422583c16dc19bf78e316bd5194284fac17eae56e235ba40872db8878244a`),
  `docs/SI-94-measurements.json` (the §3.4 harness record).

## Honesty labels carried forward

- §18.3 claim: **NOT earned**
- P5-R1 / P5-R9: **UNMEASURED**
- Live model-authored tool intent: **OCCURRED, scoped to
  SI-74 only**
- L-3: **LOCKED**
- A-2 / A-3 / A-4: **UNAPPROVED**
- Readiness unlocks: `[]`
- Verdict on this record: **BELOW THRESHOLD**
