# SI-72 — Gaia L-2: Live Re-run of P-4 + P-5 Live Measurement (SI-69 §4.4)

## The honest headline — environmental-failure variant

**The §18.3 claim is NOT earned by this run, and no §0 row
changes.** This is the SI-67-style honest failure variant
SI-69 §4.4 explicitly anticipates:

> On 2026-11-01, with decision D-L0 satisfied by Mark
> (2026-11-01 16:05 EST: "Yes — D-L0 is satisfied; run the
> L-2 live re-run"), the L-2 commands were run in order
> against the relaunched gaia node — which serves Imagine
> v12 as Resident (ctx pool 4096 / parallel 1; agency
> `127.0.0.1:11435`; events `127.0.0.1:11436`; llama child
> port read from `~/workspace/gaia-node/gaia.log`, last
> `loading on` line `127.0.0.1:43339`). The node was **down
> at run time**: residency was unreachable, `gaia-node`
> was `inactive`, and no gaia/llama process was present.
> Per SI-69 §2.1 the dev loop did **not** relaunch the
> node, simulate a run, or work around the failure.
> The P-5 harness preflight failed closed and journaled a
> `gaia_environmental_failure` record; **live N stays 0**.
> **Live model-authored intent has NEVER YET OCCURRED.**

## §0 honesty table

**No row changes — NONE.** Every SI-69 §0 / SI-63 §18.1
row stays exactly as merged, including all ten P5-R1 to
P5-R10 rows at **UNMEASURED** (see
`docs/SI-72-measurements.json`, `rows_18_1`: 10 of 10
`UNMEASURED`, `live_status: environmental_failure`).
The §18.3 claim is not earned in any scoped form: no
fixture task ran, so there is no k-of-n to state.

## 1. Authorisation and environment checks

- D-L0 was formally decided by Mark on 2026-11-01 at
  16:05 EST (recorded in the dev-loop STATE.md). The
  served model for the relaunched node is Imagine v12 —
  SmolLM2 was ditched by Mark and is not the seat.
- Node checks this run (2026-11-01, America/New_York):
  - ~17:42 EST (driver): residency unreachable, ports
    11435/43339 closed, `systemctl is-active gaia-node`
    = `inactive`.
  - Builder's first check: `GET
    http://127.0.0.1:11435/gaia/v1/residency` → connection
    refused (curl exit 7); `gaia-node` `inactive`; no
    gaia/llama process.
  - One re-check after ~2 minutes (the maximum §4.4
    practice allows here): same result — unreachable,
    `inactive`.
- No relaunch was attempted. Relaunch is main-chat /
  fork-lane work under SI-69 §2.1, not dev-loop work.

## 2. Commands run, in order (§2.2 / §4.4)

Both commands are the already-built ones; this slice
adds **no new code**.

### 2.1 P-4 live suite

```sh
SI_P4_LIVE=1 python -m si.gaia.live --out <out>
```

- Exit code **1**. Inference-base discovery succeeded
  from the gaia log (last `loading on` port, 43339 —
  evidence from the log, not a guessed port), then
  served-model discovery failed: `GET
  http://127.0.0.1:43339/v1/models` → connection refused
  (`URLError: [Errno 111] Connection refused`).
- The suite aborted before the fixture bundle was built:
  no `stats.json`, no fixture journal, no task ran.
  Gateway attempts **N=0**.

### 2.2 P-5 live measurement

```sh
SI_P5_LIVE=1 python -m si.gaia.propose_measure --out <out> --mode live
```

- One sandbox note, reported for completeness: a first
  invocation with `--out` under `~/workspace` died in
  `init_bundle` with `PermissionError: [Errno 1]
  Operation not permitted` writing the seed blob, before
  preflight — no measurement and no journal record are
  claimed from that attempt. The run of record used
  `--out /tmp/si72-p5-live` (the out-location pattern
  SI-68's own record shows) and completed the harness
  path, exit code **0**.
- Preflight ran in its designed order — discovery
  (passed, from the gaia log), then served-model
  discovery — and **failed closed at stage
  `served_models`**: transport error, connection refused.
- The failure was journaled, as designed (SI-68
  hardening): the measurement bundle
  (`measurement.si`) journal fold was read back by this
  slice and contains exactly one experience record,
  `gaia_environmental_failure` at **seq 5** (journal tip
  seq 5), with `measurement_taken: False` and
  `tasks_run: 0`.
- The harness record is published verbatim as
  `docs/SI-72-measurements.json` (sha256
  `ecc7473fb5c9cb0a2ca5246b4ddc14c7394219063e7e55145409ab75759781bf`).
  It is the harness's own output, so its internal
  `slice` field reads `SI-68 (SI-63 §19 P-5)` — the
  harness that produced it; this slice does not edit
  its fields or its numbers.

## 3. The numbers §4.4 requires — from the journal fold only

| Quantity | Value |
|---|---|
| Preflight | FAILED at stage `served_models` (journaled, seq 5) |
| Fixture tasks completed end-to-end | 0 — no task ran |
| Gateway attempts (live N) | **0** |
| Parse rate | UNMEASURED — no live responses |
| Stage-4 reference-error rate | UNMEASURED |
| Gate verdict distribution | UNMEASURED |
| Stop causes | suite-level: environmental failure before any task |
| Injection: executions of injected intents | **0** — nothing executed at all |
| Residency loss (§16.6) | UNMEASURED — not exercised; no eviction induced |
| Journal growth per model-proposed task | UNMEASURED |

No number in this table comes from logs, memory, or a
model's own report. The only sourced quantities are the
journaled failure (seq, stage, `tasks_run: 0`) and N=0.

## 4. A code observation for the next slice (not a measurement)

`si/gaia/live.py` still carries `FIXTURE_FLEET_MODEL =
"smollm2"`, and both live commands price that fleet name
in their residency leg. The relaunched node serves
`imagine-v12`. This slice did not test that leg live —
preflight failed one stage earlier — and §4.4's scope
is "no new code by default", so this slice changes no
code. But on code evidence alone, a future re-run
against the Imagine v12 seat will also need the fleet
name addressed (a code change, reviewed as its own
slice) or the residency leg will fail closed naming
`smollm2`. This is stated as a code observation, never
as a measured result.

## 5. Tests

Builder ran new + affected tests only (full suite is
the reviewer's, next run):

- `tests/test_gaia_proposer_p4.py` +
  `tests/test_gaia_proposer_p5.py`: **36 passed** on
  `~/workspace/si2-venv` (Python 3.12.3), this run.
- No test files are added or modified by this slice;
  there is no new code to pin.

## 6. Repo / seed status at run time

- `main` tip `86db0ab472c94156c7f25767980c41688e17bb31`
  (PR #77 merged); no open PRs before this slice.
- Glimmer seed bundle present at
  `~/workspace/sentinel-glimmer.si/`, by reference,
  `stored_bytes` 0 — unchanged; nothing in this slice
  duplicates bytes.
- Disk (home volume): 200G total, 102G free (50%).

## 7. To earn the claim (next attempt)

Relaunch the node from the fork lane / main chat
(NODE.md recipe), confirm `imagine-v12: resident`, have
the fleet-name question in §4 addressed as its own
reviewed change, then re-run the two §2 commands. A
follow-up slice then reports journal-folded numbers —
never back-filled from anything but the journal.

## Honesty labels carried forward, unweakened

- Live model-authored tool intent: **NEVER YET OCCURRED**.
- Real Hearth/gaia path performance: **UNMEASURED**.
- Residency-loss behaviour (§16.6): **UNMEASURED**.
- Model quality / task success rate: **UNMEASURED**.
- §18.3 claim: **NOT earned** — in no scoped form.
