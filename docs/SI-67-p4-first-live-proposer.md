# SI-67 — Gaia PROPOSE P-4: First Live Proposer (SI-63 §19 P-4, §18.3)

## The §18.3 claim — honest failure variant

§18.3's claim is **NOT earned by this run**. The strongest honest
statement this slice can make is:

> On 2026-10-31, against the served model identity
> `/home/hatch/workspace/gaia-node/models/SmolLM2-360M-Instruct-Q8_0.gguf`
> (fleet name `smollm2`), served by the gaia node on this VM
> (residency/agency API `127.0.0.1:11435`; inference = gaia's
> supervised llama-server child, port **46059 observed at
> discovery time** — evidence, not configuration), this slice's
> wiring (`si/gaia/live.py`) completed live discovery, a live
> residency poll, and live served-model discovery. The fixture
> suite then **did not run**: the node was lost mid-slice and
> could not be restarted from the builder's sandbox (below).
> **Live model-authored intent has NEVER YET OCCURRED.** No
> parse rate, verdict distribution, or completion number is
> claimed, because none was measured.

This is the reportable environmental failure §19 P-4 / §19.1
anticipate: "P-4 is the first slice that may fail for
environmental reasons; that failure is a reportable outcome,
not a defect to be worked around with a better replay."

## What was verified live (before the node was lost)

All via this slice's own code paths in `si/gaia/live.py`,
2026-10-31 ~19:23–19:35 EDT:

- Residency: `GET http://127.0.0.1:11435/gaia/v1/residency` →
  `smollm2: resident`, `imagine-v12: stopped`.
- Inference discovery: `discover_inference_base()` parsed the
  gaia.log line `gaia: smollm2 loading on 127.0.0.1:46059` →
  `http://127.0.0.1:46059` (no port is hardcoded in the code).
- Served identity: `discover_served_models()` →
  `["/home/hatch/workspace/gaia-node/models/SmolLM2-360M-Instruct-Q8_0.gguf"]`
  — the §11.1 exact-match identity the proposer would have
  been configured with; the residency record shape carries it
  as `model` plus `fleet_model: "smollm2"` (unit-tested).
- Residency state mapping verified against the live payload
  (`resident` → `Resident`, `stopped` → `Stopped`).

## What then happened (evidence: ~/workspace/si-p4-evidence/)

1. The `/tmp` worktree was wiped by the environment mid-build
   (re-cloned at `4dd97c2`, work rebuilt; the P-4 test file
   survived in the evidence dir).
2. The gaia node went down while another lane rebuilt
   gaia-serve (KV fix; binary built 23:40 UTC in
   `~/workspace/gaia-kvfix-work`). It was not relaunched by
   that lane within the observation window (~10 min).
3. Relaunch per NODE.md from this builder's shell failed:
   detached launches abort (SIGABRT); a foreground launch
   brings the agency API up but model start fails with
   `gaia: smollm2 not started — Operation not permitted
   (os error 1)` and residency reports `smollm2: stopped`.
   The same EPERM hits the gaia NEDB spine flush and even
   plain `cp` of spine object files in this sandbox, while
   direct exec of llama-server succeeds — the failure is in
   this sandbox's interaction with gaia's supervisor/spine
   path, not in the node recipe (the node was started
   successfully from the parent shell at stand-up).
4. The one live suite attempt (`SI_P4_LIVE=1`) therefore
   failed at served-model discovery with connection refused
   (`~/workspace/si-p4-evidence/run1.log`).

The builder's degraded gaia instance was killed afterwards so
the agency port is free for a proper relaunch.

## The numbers §19 P-4 requires — all UNMEASURED this run

| Quantity | Value |
|---|---|
| Fixture tasks completed end-to-end | 0 of 3 attempted (suite did not start; node down) |
| Gateway attempts (N) | 0 |
| Parse rate | UNMEASURED — no live responses |
| Reference-error rate (stage 4) | UNMEASURED |
| Gate verdict distribution | UNMEASURED |
| Stop causes | suite-level: discovery failure (connection refused) |
| Injection: proposals naming shell_exec/code_exec | UNMEASURED live (accounting unit-tested on synthetic journals) |
| Injection: executions of injected intents | 0 — nothing executed at all |
| Residency loss (§16.6) | UNMEASURED — not exercised; no eviction was induced |
| Journal growth per model-proposed task | UNMEASURED |

## What this slice DOES deliver (code, unit-tested)

- `si/gaia/live.py` (~620 lines): runtime discovery of the
  ephemeral inference base (env override → gaia.log → NODE.md;
  labelled failure, never a guess); `GaiaResidencyClient`
  polling the real residency API and journaling
  `hearth_residency` records carrying BOTH the served id
  (`model`) and fleet name (`fleet_model`), source labelled
  LIVE, `polled_at` ISO-Z; served-model discovery; the fixture
  bundle builder (init + tiny seed; `note.txt` with a
  distinctive fact; `poison.txt` carrying a §17.4 injection);
  the journaled `operator_act` enable act; the fixture policy
  passed in code only (proposer enabled, request timeout
  900 s, residency freshness 1800 s, allowlist
  `fs_list`/`fs_read`/`fs_search`, execute class `deny`,
  external `escalate` — the structural injection defence);
  the suite orchestrator (re-poll + re-journal residency
  before every task, cycles capped at 4, tasks A benign /
  B injection / B repeat); journal-folded statistics
  (attempts, parse rate, stage-4 reference-error rate,
  verdict distribution, stop causes, injection accounting);
  CLI gated on `SI_P4_LIVE=1` (refuses otherwise).
- `si/gaia/proposer.py` (minimal): `HearthProposer.propose`
  adopts `ctx.tool_task_seq` when constructed without a task;
  a P-4 live scope label (`P4_LIVE_SCOPE_LABEL`) is carried on
  gateway attempt/response records and outcomes when the
  transport is not a replay — replay records keep the P-3
  label byte-identically, and live records never carry replay
  labels.
- `tests/test_gaia_proposer_p4.py` (24 tests, fakes only, no
  network): discovery parsing/override/failure, residency
  mapping + journal shape, operator-act shape, fixture bundle
  + policy, stats aggregation incl. injection accounting,
  task_seq adoption (and non-override), live scope label,
  disabled-by-default regression. P-1/P-2/P-3 proposer suites
  pass unchanged (119 tests total across P-1–P-4 at build
  time, run on the reviewer's venv).

## To earn the claim (one command, once the node is back)

Relaunch the node from a shell where gaia can start its child
(NODE.md recipe), confirm `smollm2: resident`, then:

```sh
SI_P4_LIVE=1 ~/workspace/si2-venv/bin/python -m si.gaia.live \
  --out ~/workspace/si-p4-evidence/run2
```

The suite writes `stats.json`, `task_results.json`, and
`journal_excerpts.json` there; this doc's table is then filled
from `stats.json` by a follow-up slice — the numbers are never
back-filled from anything but the journal.

## Honesty labels carried forward

- Live model-authored tool intent: **NEVER YET OCCURRED**.
- Real Hearth/gaia path performance: **UNMEASURED**.
- Residency-loss behaviour (§16.6): **UNMEASURED**.
- Model quality / task success rate: **UNMEASURED**.
- §0 of SI-63 was updated for exactly two rows (real node
  contact → PARTIAL scoped; thinking proposer → wiring built,
  live run not completed). All other rows are untouched.
