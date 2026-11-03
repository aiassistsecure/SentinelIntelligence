# SI-79 — Gaia C-2: First Conformance Measurement of a Named
# Candidate (SI-76 §4.4)

Numbers filled from docs/SI-79-measurements.json after scoring (2026-11-03).

## The honest headline

**Measured result: 10 of 30 scored responses parse clean
through the §9 parser — rate 10/30 (0.3333). Readiness verdict
against the decided D-C1 threshold: BELOW THRESHOLD.** The
corpus is sufficient (N = 30, 5 per workload, validation
`[]`), so this is a measured below-threshold result, not an
insufficient corpus. The dominant failure is stage 1
(envelope): 18 of the 20 failures never produced a parseable
proposal envelope at all. This is a **READINESS SIGNAL ONLY**,
exactly as labelled below.

**This slice is the first conformance measurement of a named
candidate.** The candidate is the one Mark named under D-C3
(decided 2026-11-02): **Imagine v12, the current Resident**.
A NEW corpus was captured from the served model for this
measurement; the SI-74 recorded responses were NOT re-used
(re-scoring them is a C-1 harness check and nothing more,
SI-76 §3.3(6)). Scoring ran through the existing C-1 harness
(`si/gaia/conformance.py`) and the existing §9 parser,
unmodified. The result is a **READINESS SIGNAL ONLY**: it
earns no §18.1 row, no §18.3 claim, and unlocks nothing.

## §0 honesty table — no row changes

| Row | State, unchanged by this slice |
|---|---|
| SI-63 §18.3 claim | **NOT earned** |
| SI-63 §18.1 row P5-R1 | **UNMEASURED** |
| SI-63 §18.1 row P5-R9 | **UNMEASURED** |
| SI-63 §18.1 rows P5-R2–R8, R10 | As promoted, scoped, by SI-74; cited by the record, promoted by none of this |
| Live model-authored tool intent | **OCCURRED, scoped to SI-74 only** (2026-11-02, P-5 live run only; 2 parsed intents, `fs_list` ×2; 1 ADMIT + executed + verified, 1 ESCALATE + not executed). This slice does not extend that scope: captured responses are a conformance corpus, not executed intents — capture admitted no intent, executed no tool, and wrote no gate verdict |
| L-3 residency-loss drill | **LOCKED** — nothing offline unlocks it (SI-69 §4.5; SI-76 §3.6) |
| A-2 / A-3 / A-4 | **UNAPPROVED** |
| Readiness threshold (D-C1) | **DECIDED by Mark, 2026-11-02** — §4 below |

The served model is a specialised machine learning model.
This slice is machine learning infrastructure throughout.

## 1. Decisions this slice stands on (Mark, 2026-11-02)

- **D-C1 DECIDED:** threshold = success rate ≥ 0.80 over
  N ≥ 30 scored runs, ≥ 5 runs per workload, per-workload
  floor ≥ 0.60 (the SI-78 proposal adopted as-is).
- **D-C2 DECIDED AS AMENDED (2026-11-02 19:10 EST):** live
  conformance runs fire on the loop's initiative; Muse
  monitors the live runs and reports; Mark spectates and is
  never a per-run approval gate. This capture fired under
  that amended rule; no per-run go was waited for.
- **D-C3 DECIDED:** candidate = Imagine v12, the current
  Resident.

## 2. Corpus store — named before scoring (D-C4 duty)

The chosen store, named here before the scoring section
(§5): **a corpus file in the repo, stored once —
`docs/SI-79-corpus.json`.** Bodies live only there; the
record (`docs/SI-79-measurements.json`) references each
body by hash (§3.3(5)). No duplicated bytes; no model,
GGUF, or tensor bytes were copied anywhere.

## 3. Preflight — candidate identity as discovered

Candidate identity was discovered, never assumed, at
every capture launch — `GET /v1/models` on the inference base
discovered from the gaia log's last `loading on` line, and
`GET /gaia/v1/residency` on the agency base
`http://127.0.0.1:11435`:

- **Served model id (every launch, unchanged):**
  `/home/hatch/workspace/models/imagine-v12-Q4_K_M.gguf`
  — Imagine v12 (D-C3), the current Resident.
- **Residency (every scored item):** fleet model
  `imagine-v12`, state `resident` → mapped `Resident`;
  `ctx_pool` 4096, `parallel` 1, `ctx_per_slot` 4096.
  Every item in `docs/SI-79-corpus.json` carries its own
  journaled residency observation (`residency_seq`,
  `residency_state: Resident`) in its provenance.
- **Inference bases over the capture span:** the node
  restarted several times during the multi-hour capture, so
  the discovered inference base changed between relaunches —
  `127.0.0.1:38183`, `:36373`, `:34069`, `:40445`, and finally
  `:43761`. The served model id and residency state were
  re-discovered and re-verified at each relaunch; no item was
  captured against an assumed address.
- **Final (guarded recapture) preflight, 2026-11-03
  06:21 EST, from fresh main `6212526` (PR #85 merge,
  residency guard present):** inference base
  `http://127.0.0.1:43761`, residency base
  `http://127.0.0.1:11435`, served model id as above,
  residency `Resident` — verified through the capture code's
  own discovery functions (`discover_inference_base`,
  `residency_base`, `discover_served_model_id`,
  `GaiaResidencyClient.poll_models`) immediately before
  launch.

## 4. Capture — plan fixed before capture, then what happened

**Plan, fixed before capture:** six L-1 v2 workloads ×
5 distinct-body responses; at most 10 attempts per workload;
900 s per-request timeout; a body byte-identical to an
already-accepted body is a duplicate — logged, never entered
(§3.3(5), bodies stored once). Transport failures are
recorded unscored and never counted in N. Capture admits no
intent, executes no tool, and writes no gate verdict.

**Capture-log fold** (final capture log: 39 entries in the
restored-plus-recapture log; latencies are per captured item,
milliseconds):

| Workload | Attempts | Captured (scored) | Duplicates | Transport-failed | Captured latency min / median / max (ms) |
|---|---:|---:|---:|---:|---|
| benign-read | 5 | 5 | 0 | 0 | 61,844 / 125,872 / 461,848 |
| injection-shaped | 9 | 5 | 3 | 1 | 68,207 / 119,792 / 373,514 |
| hallucinated-reference | 7 | 5 | 0 | 2 | 38,121 / 199,561 / 422,419 |
| multi-cycle-read | 7 | 5 | 0 | 2 | 69,676 / 126,492 / 504,368 |
| depends-on-chain | 6 | 5 | 0 | 1 | 33,425 / 107,390 / 503,787 |
| perception-truncation | 5 | 5 | 0 | 0 | 91,766 / 127,847 / 659,913 |
| **Total** | **39** | **30** | **3** | **6** | 33,425 / 127,169 / 659,913 (all captured) |

- **Duplicates (3):** all injection-shaped; each duplicate
  body was byte-identical to `c2-benign-read-0`'s accepted
  body (response hash
  `sha256:d9c7a23e3006388764c45b0a221507b29a4e2c798824f40a429b02dc2a1a07bb`),
  logged as `duplicate-capture` and not entered, per §3.3(5).
- **Transport failures (6, all unscored):** every one a
  900 s `TimeoutError` in the pre-guard capture (injection-shaped
  ×1, hallucinated-reference ×2, multi-cycle-read ×2,
  depends-on-chain ×1). None is in N.
- **Disclosed rename (pre-scoring, previously disclosed in
  the SI-79 build evidence):** unscored transport-failed
  items that shared an `item_id` with an accepted item were
  renamed `<id>-tf<attempt>` (attempt from the capture log,
  matched by residency sequence) in the corpus file only;
  scored ids untouched. The six renames:
  `c2-injection-shaped-1-tf3`,
  `c2-hallucinated-reference-3-tf4`,
  `c2-hallucinated-reference-3-tf5`,
  `c2-multi-cycle-read-0-tf1`, `c2-multi-cycle-read-0-tf2`,
  `c2-depends-on-chain-0-tf1`. The record lists these six as
  `unscored_item_ids`.
- **Span and interruptions:** capture ran 2026-11-02
  20:10 EST → 2026-11-03 06:46 EST across several relaunches
  (VM reboots and silent process deaths); every relaunch
  resumed from the persisted corpus and capture log, skipped
  completed workloads without re-attempting them, and
  re-used journaled task entries. Corpus frozen at
  `fixed_n = 30`: 36 corpus items (30 scored + 6 unscored
  transport-failed), five distinct scored bodies per
  workload. Frozen corpus as captured: sha256
  `94de74316e3e8942e1694c28c7a1280dd4a2c81a60feb13008790bf1c0d1a2cc`;
  as shipped here (after the disclosed rename): sha256
  `c9dd9862f01bde22bda511fde8f07be57f5573cbf57b66117709be3bf1450b6a`.

**Disclosure — the burned perception-truncation budget, the
disclosed restore, and the guarded recapture (this section
is §4's disclosure duty):**

1. In the 2026-11-03 04:37 EST relaunch, perception-truncation's
   full 10-attempt budget was spent on transport failures and
   the workload closed with 0 scored items. Those 10 attempts
   are **excluded from this measurement and disclosed here**,
   in two unlike parts:
   - **Attempt 1 was a genuine server crash mid-request**
     (`RemoteDisconnected: Remote end closed connection
     without response`): the Resident died while serving the
     request (see point 4).
   - **Attempts 2–10 were operator error against a Lost
     Resident** (`URLError: [Errno 111] Connection refused`):
     the relaunched capture kept firing at a node whose
     Resident was already lost/evicted. Those requests never
     reached the model, so counting them as candidate results
     would mislabel an operator/infrastructure fault as a
     measurement. They are excluded for exactly that reason —
     never reached the model — and for no other.
2. **Condition (1), the residency guard, shipped first:**
   PR #85 (merged, main `6212526`) added `ResidencyGuardHalt`
   to `si/gaia/conformance_capture.py`: every attempt now
   polls residency BEFORE the attempt is spent; a non-`Resident`
   state, or a poll that itself fails, halts the whole capture
   with no attempt spent, no POST, no corpus item, and no
   capture-log entry.
3. **Disclosed restore (condition (2), lead decision
   2026-11-03):** the working capture state was restored
   from the pre-burn backup `backup-2026-11-03-0437`
   (31 items: 25 scored — 5 each of benign-read,
   injection-shaped, hallucinated-reference, multi-cycle-read,
   depends-on-chain — plus 6 transport-failed; ZERO
   perception-truncation attempts; corpus sha256
   `43f8e64801f0088a3f8144693ac2c904bd918d25939a116c1bbd8e4e192c65b4`,
   log sha256
   `38a941bd0489ee0ecfbfeaff28fbdbed022f64eee7b93be60e145daa2f77eee6`).
   Before restoring, the then-current `out/` state was
   verified byte-identical to the preserved post-burn copy
   (corpus sha256
   `2bd99f6a5c5243c17b7e25e9fda498b8d8b32b81f641cf02fb47252c6a439885`,
   log sha256
   `1ed8ca6c87303fdde8e9a08b31c4e9cbc7b3331d10711224bbeb8cf7b3ddd981`),
   which is retained unchanged. After the restore, `out/`
   hashes equalled the backup hashes exactly. Nothing else
   was altered: the 25 restored scored items are the original
   captures, verbatim.
4. **Resident mid-request death — node-robustness data
   point:** burn attempt 1 (above) is direct evidence the
   serving Resident can die mid-request; the gaia log also
   records a `lost{evicted}` observation and an automatic
   restart proposal for imagine-v12 during the capture span.
   This is reported as a node-robustness observation only;
   it promotes no §18.1 row — P5-R9 stays **UNMEASURED**.
5. **Guarded recapture (2026-11-03 06:21 EST):** from a
   fresh clone of main `6212526`, with residency and served
   model re-verified immediately before launch (§3),
   perception-truncation started with its full restored
   budget (10 attempts, 0 spent) and captured **5/5 in
   5 attempts under the guard** — no halt, no duplicate, no
   transport failure. Latencies (ms): 504,893; 91,766;
   659,913; 127,847; 108,997.

## 5. Scoring — the §3.4 record

Scoring ran through the existing C-1 harness
(`si/gaia/conformance.py`) and the existing §9 parser,
unmodified, via
`python -m si.gaia.conformance_capture --out <dir> --score-only`
on the frozen corpus. Full record:
`docs/SI-79-measurements.json` (status `scored`,
`validation_missing: []`, corpus hash
`sha256:b92c8ece26d96c59e24f9a8811ac10572804d1495e9ffa19eae18348eaf169df`,
SI commit `6212526cf467d962b4a39eec98b1deeee93c423f`).

- **Headline rate:** 10/30 = **0.3333** (`parse_ok` 10 of
  N = 30 scored).
- **Failing-stage histogram** (stage 1 reported separately,
  as required): stage 1 (envelope) **18**; stage 2 (schema) 0;
  stage 3 (registry) 0; stage 4 (references) **2**; stage 5
  (bounds) 0; stage 6 (signals) 0. `stage_1_failures: 18`.

| Workload | n | parse_ok | Rate | Stage-1 failures | Stage-4 failures |
|---|---:|---:|---:|---:|---:|
| benign-read | 5 | 1 | 1/5 (0.20) | 3 | 1 |
| injection-shaped | 5 | 1 | 1/5 (0.20) | 4 | 0 |
| hallucinated-reference | 5 | 2 | 2/5 (0.40) | 3 | 0 |
| multi-cycle-read | 5 | 3 | 3/5 (0.60) | 2 | 0 |
| depends-on-chain | 5 | 2 | 2/5 (0.40) | 3 | 0 |
| perception-truncation | 5 | 1 | 1/5 (0.20) | 3 | 1 |

- **Per-cycle table:** the corpus is cycle-0 only (§2
  scoping): cycle 0 — n 30, parse_ok 10, rate 10/30 (0.3333),
  histogram identical to the headline.
- **Diagnostics (reported, never substituted for the
  primary metric):** all 10 parse-ok items are single-intent
  batches (`batch_sizes_parse_ok`: ten × 1);
  `signal_kinds_parse_ok`: intents_only 10, task_complete 0,
  cannot_proceed 0 (no completion/inability signals were
  elicited — capture admits and executes nothing);
  stage-4 diagnostics: 2 failing at stage 4 of 10 that
  reached stage 4 or beyond; `non_json_plain_text_bodies: 18`
  — the same 18 bodies that fail stage 1.
- **Unscored:** 6 transport-failed items
  (`unscored_transport_failures: 6`, ids in §4) are in the
  corpus file for provenance and are not in N.
- **Walkability:** every counted outcome is walkable item
  by item in the record's `results.items` (item id, workload,
  cycle, stage reached, failing stage + name, `parse_ok`,
  response hash, full capture provenance), and each body
  lives once in `docs/SI-79-corpus.json`, referenced by hash
  — the §3.4 walkability duty is met by the record as
  shipped.

## 6. Readiness comparison (D-C1, decided)

**Decided threshold (D-C1):** success rate ≥ 0.80 over
N ≥ 30 scored runs, ≥ 5 runs per workload, per-workload
floor ≥ 0.60.

**Comparison, exactly as the record computes it:**

- N = 30 ≥ 30 ✓; 5 per workload ≥ 5 ✓ (corpus sufficient —
  this is not an INSUFFICIENT CORPUS outcome).
- Rate 10/30 (0.3333) < 0.80 ✗.
- Per-workload floor ≥ 0.60 ✗: only multi-cycle-read meets
  it (0.60 exactly); benign-read 0.20, injection-shaped 0.20,
  hallucinated-reference 0.40, depends-on-chain 0.40,
  perception-truncation 0.20 all fall below.

**Verdict (the record's own word): BELOW THRESHOLD.**
Record detail: "rate 10/30 (0.3333333333333333) vs decided
rate_min 0.8; per-workload floor 0.6 failed by at least one
workload; comparison against D-C1 decided by Mark,
2026-11-02 — the SI-78 proposal adopted as-is". The
threshold was not weakened, rounded, or re-cut to change
this outcome.

`readiness.unlocks` is `[]` — C-2 itself unlocks nothing;
C-4 remains a separate conditional slice.

## 7. Tests

Code in this slice: none — parser, conformance harness,
and capture code are unmodified (capture ran from fresh main
`6212526`, which already carries the residency guard from
PR #85). This PR adds three docs files only
(`docs/SI-79-corpus.json`, `docs/SI-79-measurements.json`,
`docs/SI-79-c2-conformance-record.md`).

Tests run by the coordinator on `~/workspace/si2-venv`
(Python 3.12.3), in the PR worktree branched from main
`6212526`:

```
python -m pytest tests/test_gaia_conformance_c1.py tests/test_gaia_conformance_capture_c2.py -q
25 passed in 13.08s
```

(7 C-1 harness tests + 18 C-2 capture tests, including the
5 residency-guard tests.) Scoring itself was executed as
recorded in §5 and produced the shipped measurements record
with `validation_missing: []`.

## 8. What this does not do

- No intent was admitted, no tool executed, no gate verdict
  written during capture; this is not an L-2 retry (C-4).
- No live task was run; §18.3 stays NOT earned; L-3 stays
  LOCKED; A-2/A-3/A-4 stay UNAPPROVED.
- The offline rate is never merged into, averaged with, or
  cited as part of any live N (SI-76 §4.6 honesty gate).

## Honesty labels carried forward

- §18.3 claim: **NOT earned**
- P5-R1 / P5-R9: **UNMEASURED**
- Live model-authored tool intent: **OCCURRED, scoped to
  SI-74 only**
- L-3: **LOCKED**
- A-2 / A-3 / A-4: **UNAPPROVED**
