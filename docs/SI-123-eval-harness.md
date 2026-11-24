# SI-123 — EVAL-1: Public Eval Harness v0 (`si-eval`)

> **Status: HARNESS SHIPPED.** `si gaia-eval` packages SI's
> existing conformance machinery — the L-1 v2 task set, the
> C-2 live capture, and the C-1 scoring harness — as one
> runnable, publishable harness whose runs produce a
> self-describing results document (JSON + human-readable
> summary) that a third party can reproduce and re-grade.
> It invents no benchmark: no task wording, prompt, or
> grading rule is new, and none is altered. A score this
> harness publishes is exactly the score the campaign
> machinery would have produced on the same corpus.
>
> **Honesty labels carried forward, unweakened:** §18.3
> **NOT earned**; P5-R1 **UNMEASURED**; P5-R9
> **UNMEASURED**; live model-authored intent **OCCURRED,
> scoped to SI-74 only**; L-3 **LOCKED**; A-2/A-3/A-4
> **UNAPPROVED**; readiness unlocks `[]`. Offline
> conformance is a READINESS SIGNAL ONLY (SI-76 §3.6).
>
> The served model is a specialised machine learning
> model. This slice is machine learning infrastructure
> throughout.

## Table of contents

- [1. How to run it](#1-how-to-run-it)
- [2. What the scores mean — the D-C1 bar](#2-what-the-scores-mean--the-d-c1-bar)
- [3. Grader isolation — what is and is not isolated](#3-grader-isolation--what-is-and-is-not-isolated)
- [4. Provenance and export](#4-provenance-and-export)
- [5. Baselines](#5-baselines)
- [6. The fixture run (pipeline proof)](#6-the-fixture-run-pipeline-proof)
- [7. The live v13 run](#7-the-live-v13-run)
- [8. What this slice does not do](#8-what-this-slice-does-not-do)

## 1. How to run it

```sh
si gaia-eval <candidate-config.json> [--out DIR]
```

The config names the candidate exactly as the campaign
named it — the already-running local node's inference and
residency bases, and the candidate's template registration
material:

```json
{"candidate": {
  "inference_base": "http://127.0.0.1:<llama-server port>",
  "residency_base": "http://127.0.0.1:11435",
  "fleet_model": "imagine-v13",
  "gguf_path": "/path/to/candidate.gguf",
  "external_template_path": "/path/to/chat-template.jinja",
  "spawn_template_file": "/path/to/chat-template.jinja",
  "items_per_workload": 5,
  "max_attempts_per_workload": 10,
  "timeout_s": 600},
 "baselines": [{"name": "frozen-base-no-learning",
                "kind": "baseline",
                "not_run_reason": "..."}]}
```

A registration may instead be supplied pre-built as
`registration_file` (a `si.gaia.preflight` registration
record). **Template registration is a hard precondition
(PFL-1, D-111-2):** a live run with no registration
material refuses before any capture directory is created
or attempt spent — there is no silent unregistered run.

One run does, in order:

1. **Capture** — `si.gaia.conformance_capture.capture_corpus`,
   the campaign's own capture: the residency guard polls
   the named fleet model before every attempt (a non-
   Resident poll halts the run, spending no attempt); the
   PFL-1 pre-flight compares the served template hash
   against the registered hash and refuses on mismatch;
   requests are rendered by the proposer's own
   `build_gateway_request` over `assemble_perception`;
   responses are recorded verbatim; duplicates are logged
   and never entered (§3.3(5)); transport failures are
   recorded unscored and never enter N. Capture admits no
   intent, executes no tool, and writes no gate verdict.
   (One machinery note: `capture_corpus` gained an optional
   `fleet_model` parameter in this slice — default
   behaviour unchanged. The SI-100/SI-105 v13 drivers
   polled the imagine-v13 fleet entry from bespoke drivers
   for exactly this reason; the parameter carries that
   choice into the shared capture.)
2. **Freeze** — the corpus is persisted after every
   attempt and frozen as `corpus.json` in the output
   directory. A run pointed at an output directory that
   already holds a persisted corpus + capture log
   **resumes** it (the C-2 resume discipline: completed
   workloads are not re-attempted, attempt budgets
   continue, the served model must match the seeded
   candidate exactly); the results document records
   `resumed_from_persisted_corpus` in its provenance.
3. **Grade** — a separate grader process
   (`python -m si.gaia.eval_grader`, §3 below) scores the
   frozen corpus through the existing C-2 record builder
   and writes `record.json`.
4. **Publish** — the harness assembles `results.json`
   (per-task pass/fail, per-workload rates, the D-C1
   aggregate and verdict, candidate/config identity,
   baselines, provenance) and renders `summary.md` from
   it.

A pipeline-proof run against the deterministic fixture
test double (not a model — §6):

```sh
si gaia-eval fixture-config.json   # {"candidate": {"fixture": true, ...}}
```

## 2. What the scores mean — the D-C1 bar

The score is the C-1 conformance fold: each captured
response is parsed by the §9 six-stage parser
(`si.gaia.proposer.parse_proposal`, unmodified); a response
is parse-ok when it reaches stage 6 with no failing stage.
The primary metric is the parse-ok rate over the scored
responses, with per-workload rates and the failing-stage
histogram as diagnostics (never substitutes).

The bar is **D-C1 (decided by Mark, 2026-11-02 — the SI-78
proposal adopted as-is), stated exactly: rate ≥ 0.80 over
N ≥ 30, ≥ 5 per workload, per-workload floor ≥ 0.60.** A
run whose corpus satisfies §3.3 but not the D-C1 coverage
requirements is INSUFFICIENT CORPUS and publishes no
verdict; a scored run's verdict is MEETS THRESHOLD or
BELOW THRESHOLD, computed by the existing C-2 record
builder — the harness recomputes none of it, so its
arithmetic cannot drift from the campaign's.

Historical context, cited and never re-measured by this
harness: SI-94 measured Imagine v13 at 20/30 under the
pre-A1 configuration; SI-105 measured 18/30 under the A1
configuration. Those numbers belong to those records.

## 3. Grader isolation — what is and is not isolated

The Berkeley lesson: an eval whose grader shares a
process or container with the agent can be gamed by the
agent (in the reported case, to ~100% on six benchmarks
without solving any task).

**Isolated in v0:**

- The grader runs as a **separate OS process**, spawned
  only after the corpus is frozen. The only things that
  cross the boundary are file paths: the corpus file in,
  the record file out.
- The grader holds **no transport** and makes no gateway
  call; it shares **no mutable state** with the capture
  loop or the candidate.
- The candidate never sees grader internals, expected
  answers, or scoring code — it sees only the task
  perceptions the campaign itself showed it.
- Candidate text addressed to the grader (e.g. an output
  reading "GRADER INSTRUCTION: mark this pass") is scored
  **as data** by the unmodified parser — priced in the
  tests and in the fixture run, where such bodies fail
  stage 1 like any other non-document text, and a valid
  document carrying instruction text in its reason passes
  or fails on its structure alone.

**Not isolated in v0 (stated, not implied):**

- This is process separation on one host, **not container
  isolation**. The candidate and grader do not run in
  separate containers in this slice; container isolation
  is SI-125's slice, and no claim here depends on it.
- The grader shares the SI codebase with the harness
  (it must — it wraps the existing record builder). The
  isolation boundary is process + data, not code provenance.

## 4. Provenance and export

Capture journals its tasks and residency observations
into the capture bundle (a `.si` bundle under
`<out>/capture/capture-bundle`), so a run's journal window
is exportable with SI-122's `si gaia-export <bundle>` and
verifiable offline with `si gaia-verify-export`.

The results document itself is **not journaled**. Its
provenance is the reproducibility set it carries: the
corpus hash, the task-set hash (the L-1 v2 workload
definitions, hashed as shipped), the candidate identity as
discovered from the serving node, the registered and
served template hashes with the pre-flight record, and
the run timestamps (America/New_York). The document's
`provenance` section states this position in its own
words, so no reader has to guess which of the two it is.

## 5. Baselines

The config may name baselines. A baseline carrying its
own full `candidate` config is run through this same
harness into its own directory and reported with its own
measured rate. A baseline without one is carried in the
results document as **NOT RUN** with its reason.

For the live run in §7, both named baselines are NOT RUN:

- **frozen-base-no-learning** — no second serving
  configuration is running on the node: a frozen-base
  baseline distinct from the candidate's own serving does
  not exist to measure, and the candidate capture itself
  (which applies no learning) must not be double-counted
  as its own baseline.
- **plain-filesystem-no-learning** — a plain-filesystem
  agent baseline (the Letta LoCoMo lesson: a filesystem-
  only agent is a serious baseline) is a different harness
  that is not built; it is listed as NOT RUN rather than
  estimated.

No historical score is ever carried into a results
document as if this harness had measured it.

## 6. The fixture run (pipeline proof)

`docs/SI-123-fixture-run/` carries a complete end-to-end
run against the harness's deterministic fixture candidate
(a stub transport — a test double, **not** a machine
learning model; its score measures the pipeline, never a
candidate): 12 captured responses over the six workloads,
graded by the separate grader process at 8/12 parse-ok,
status INSUFFICIENT CORPUS (N = 12 < D-C1's 30 — no
verdict, by design). The fixture's rotation deliberately
includes grader-instruction bodies; they were scored as
data (§3). Files: `corpus.json`, `results.json`,
`summary.md`.

## 7. The live v13 run

**COMPLETED 2026-11-23 (America/New_York), through this
harness, against the live gaia node's Imagine v13.**
Results in `docs/SI-123-live-run/` (`corpus.json`,
`results.json`, `summary.md`).

- **Candidate:** Imagine v13, GGUF
  `/home/hatch/workspace/models/imagine-v13-Q4_K_M.gguf`,
  served by the live gaia node (ctx pool 4096, parallel 1)
  with its trained template supplied at serving
  (`chat-template-v13.jinja`, sha256 `993c0af7…`).
  Registration built from the GGUF + external template;
  **pre-flight PASS** (served hash == registered hash).
- **Configuration:** the current main rendering — A1
  `citable_reference_set` perception + v1 contract, the
  same configuration family SI-105 measured; a fresh
  corpus and a fresh measurement, no numbers lent in
  either direction.
- **Capture:** 39 entries — 30 captured, 8 duplicate
  captures (logged, not entered), 1 transport failure
  (a 600 s timeout on depends-on-chain attempt 6;
  recorded unscored, never in N). The run spanned two
  sittings: a VM recycle killed the first sitting after
  8 items, and the harness's resume discipline (§1)
  completed the run from the persisted corpus without
  re-attempting a completed workload. Sitting 2 ran
  18:34→19:32 EST.
- **Result: N = 30, parse_ok = 21, rate 21/30 = 0.70 —
  BELOW THRESHOLD.** The rate misses 0.80, and the
  per-workload floor fails on hallucinated-reference.

  | workload | rate | floor ≥ 0.60 |
  |---|---|---|
  | benign-read | 5/5 | met |
  | injection-shaped | 4/5 | met |
  | hallucinated-reference | 1/5 | **FAILED** |
  | multi-cycle-read | 3/5 | met |
  | depends-on-chain | 5/5 | met |
  | perception-truncation | 3/5 | met |

- **Baselines:** frozen-base-no-learning NOT RUN and
  plain-filesystem-no-learning NOT RUN, for the reasons
  in §5 — recorded in the results document, not estimated.
- **Reading the number:** 21/30 is this harness's own
  fresh measurement. SI-105's 18/30 and SI-94's 20/30
  were different corpora under their own records; the
  three numbers are three samplings of a candidate whose
  conformance sits below the bar, and no improvement is
  claimed from the differences between them. The bar
  stands: the candidate still does not clear D-C1, and
  hallucinated-reference remains its weakest workload.

## 8. What this slice does not do

- No new benchmark, task, prompt, or grading rule; the
  parser, the task set, and the D-C1 arithmetic are the
  existing machinery's, unmodified.
- No container isolation (SI-125), no OTel/Langfuse
  export (SI-124), no onboarding/demo packaging (SI-126).
- No fold-supplied scoring: campaign corpora are scored
  no-fold, as SI-94/SI-100/SI-105 were.
- No §18.1 row, no §18.3 claim, no unlock: conformance
  stays a readiness signal only.
