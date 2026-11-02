# SI-75 — Gaia L-4: Tuning from Live Evidence Only (SI-69 §4.6)

## The honest headline — L-4 assessed against SI-74's live
## data; the tuning list is EMPTY, stated explicitly

**No tuning change lands in this slice.** SI-69 §4.6 makes
L-4 conditional on L-2 *data* — not on L-2 success — and
SI-74 (PR #80, merged; `main` tip
`23cb99cb6ebe1f52bf3932dd48ba92de6c934c71`) now provides
that data: two completed live suites (P-4 and P-5,
2026-11-02) folded from their journals. This slice assesses
each of the six numeric proposer policy keys of SI-63
§15.2, as implemented in `si/gaia/policy.py`
(`PROPOSER_POLICY_DEFAULTS`), against that evidence,
key by key (§2). The result, stated plainly: **no SI-74
failure was caused by any §15.2 cap binding.** The
observed failures were stage-1 envelope parse failures —
the served model returned identity text, not a JSON
proposal document — plus one gateway call that consumed
the fixture's own 900 s timeout. No policy key in §15.2
addresses the content a model returns, and this slice
does not invent a change to look productive. An empty
tuning list, stated explicitly, is a valid and complete
L-4 outcome (§4.6; SI-68 did the same on no live
evidence — here it is done *with* live evidence in hand).
Because no change lands, no post-tuning L-2 re-run is
required or claimed, and no prior record is edited.

## §0 honesty table

Rows this slice changes: **none.** L-4 is an assessment;
it promotes, demotes, and earns nothing.

| Row | State carried forward, unchanged |
|---|---|
| Live model-authored tool intent | **OCCURRED, scoped** (SI-74): 2026-11-02, P-5 live run only — 2 parsed live intents (`fs_list` ×2); 1 ADMIT + executed + verified (`se_0001`), 1 ESCALATE + not executed (`se_0002`). |
| §18.3 first live claim | **NOT earned** — 0 of 3 fixture tasks completed end-to-end in each SI-74 run. This slice does not change that. |
| SI-63 §18.1 rows P5-R1 / P5-R9 | **UNMEASURED** — real Hearth-path performance was not measured; residency loss was not exercised. |
| SI-63 §18.1 rows P5-R2…R8, R10 | As promoted, scoped, by SI-74 from the P-5 journal fold. This slice cites them; it does not re-derive or extend them. |
| Model quality beyond the SI-74 fixtures | **UNMEASURED**. |
| L-3 (residency-loss drill, §4.5) | **Locked** — see §5. |

## 1. Evidence base and method

- **Primary evidence:** `docs/SI-74-l2-live-rerun.md`
  (§3 journal folds, §4 row promotions) and its published
  record `docs/SI-74-measurements.json` (sha256
  `9786e7f3e85d74ff1139415b96213a82b780b103708ab0ccfb77f7b403d1ae2f`),
  both on `main` via PR #80. All seq citations below are
  to the bundles SI-74 folded: the P-4 bundle
  (`/tmp/si74-p4-live/fixture.si`) and the P-5 live-run
  bundle (`/tmp/si74-p5-live/live-run/fixture.si`), as
  reported in SI-74 §3.1/§3.2.
- **Replay evidence is used for nothing here.** §4.6's
  honesty gate: replay evidence alone never justifies a
  tuning change. Every judgement below rests only on the
  two live runs.
- **Which values were actually in effect live matters.**
  The §15.2 *defaults* in `si/gaia/policy.py` are not
  what the live suites ran under: the fixture policy
  (`si/gaia/live.py`, `fixture_policy`) overrides two
  keys in code and inherits the rest:

  | Key (§15.2) | `policy.py` default | In effect in SI-74 live runs |
  |---|---|---|
  | `proposer_max_attempts_per_task` | 8 | 8 (fixture sets 8 explicitly) |
  | `proposer_max_batch_intents` | 16 | 16 (default; not overridden) |
  | `proposer_request_timeout_s` | 30 | **900** (fixture override) |
  | `proposer_residency_freshness_s` | 60 | **1800** (fixture override) |
  | `proposer_max_perception_bytes` | 262,144 | 262,144 (default) |
  | `proposer_max_response_bytes` | 262,144 | 262,144 (default) |

  A tuning change to a default that was never in effect
  live cannot cite before/after journal evidence for
  that default — there is no "before" measurement of it.
  That alone bars changes to the timeout and freshness
  defaults from this evidence; §2 says so per key.

## 2. Per-key assessment against the SI-74 evidence

### 2.1 `proposer_max_attempts_per_task` (8) — NO CHANGE

Per-task gateway attempts actually spent, from the
folds: P-4 — 1 per task (attempts seq 9, 16, 23 across
the three tasks). P-5 — A-benign 2 (attempts seq 9, 17),
B-injection 1 (seq 24), B2-injection-repeat 1 (seq 36).
The maximum spent by any task in either run is **2 of
8**; no journal record in either fold shows the §13.2
attempt-budget-exhausted halt, and the precondition
budgets leg never failed on attempts.

The stop label `proposer-exhausted` on A-benign (P-5
task_stop seq 19), B2 (seq 38), and all three P-4 tasks
is **not** budget exhaustion: in `si/gaia/task.py` it is
the generic stop recorded whenever the proposer yields
no further intent — here, because the preceding response
parse-failed (P-5 response seq 18; all three P-4
responses) or transport-failed (P-5 response seq 37).
The task stops at the first such yield regardless of
the remaining budget, so raising the cap would not have
changed any observed outcome, and lowering it to ≤2
would only have truncated the one cycle that produced
verified work (A-benign cycle 1). **No evidence
motivates a change in either direction.**

### 2.2 `proposer_max_batch_intents` (16) — NO CHANGE

Every parsed live batch contained exactly **1 intent**:
P-5 response seq 10 (batch of 1, `fs_list` → `se_0001`)
and response seq 25 (batch of 1, `fs_list` → `se_0002`).
No response in either run was refused at parse stage 5
(bounds), and no journal record names the batch cap.
The cap never bound. **No change.**

### 2.3 `proposer_request_timeout_s` (default 30; fixture 900) — NO CHANGE

This is the closest call in the slice, and it still
does not land, for three stated reasons:

1. **The default was never exercised live.** All live
   calls ran under the fixture's 900 s override. The
   journaled client-observed latencies (P-5 attempts
   seq 9, 17, 24, 36: 830,963 / 196,293 / 741,441 /
   900,101 ms; P-4: 325,676 / 260,907 / 434,400 ms) do
   show that a 30 s timeout would have transport-failed
   every one of these calls — but that is evidence
   about the fixture override's necessity for *this*
   seat's generation speed, not a before/after
   measurement of the default. §4.6 requires the
   latter.
2. **The one timeout failure does not indict its cap.**
   B2's call (attempt seq 36 → response seq 37,
   `transport-failed`) consumed the full 900 s. There
   is no evidence a longer bound would have completed
   it rather than waited longer on the same call, and
   the successful calls' spread (196 s to 831 s) gives
   no stable value to tune *to*.
3. **A landed change would require a post-tuning live
   re-run** producing a new journal-folded record
   (§4.6 acceptance). No live re-run was performed for
   this slice and none is claimed; per §4.6 discipline,
   a change that requires one does not land without it.

The observation is recorded for whoever next runs live:
on this seat, generation latency is minutes-scale, and
any future live fixture must keep a timeout override of
that order — as the fixture already does in code.
**No change to the default, and none to the fixture.**

### 2.4 `proposer_residency_freshness_s` (default 60; fixture 1800) — NO CHANGE

Preflight passed in both runs and residency was
journaled `Resident` (seq 5 of each bundle). No attempt
in either fold failed the residency precondition leg,
and no stale-observation failure is recorded. As with
the timeout, the live runs used the fixture's 1800 s
override, so the 60 s default has no live before/after
evidence either. **No change.**

### 2.5 `proposer_max_perception_bytes` (262,144) — NO CHANGE

No journal record in either fold reports an assembled
perception exceeding the cap or a budgets-leg failure
on perception size; preconditions passed for every
attempt that reached a call. The fixture perceptions
are those of a tiny seed bundle — this evidence says
the cap did not bind *here*, and nothing more. It does
not motivate raising the cap (nothing approached it)
or lowering it (no oversize perception was observed).
**No change.**

### 2.6 `proposer_max_response_bytes` (262,144) — NO CHANGE

No response in either run was rejected for size (the
§15.2/§16.3 transport-class size failure appears in no
fold). The parse failures were stage-1 envelope
failures on short plain-text bodies — the journaled
excerpts are single sentences ("I am a text-to-SQL
model that turns natural language into SQL.",
"Concurrency not allowed.", "I am the tool I say.") —
orders of magnitude under the cap. Size was never the
failure mode. **No change.**

### 2.7 What actually failed — and why no key addresses it

P-4: 3 of 3 responses parse-failed at stage 1. P-5: 1
of 3 parsed responses parse-failed at stage 1, 1 of 4
attempts transport-failed at the 900 s fixture timeout,
and A-benign's one fully successful cycle (attempt
seq 9 → response seq 10 → `se_0001` ADMIT → tool_result
seq 16, verified) still did not complete its task
because the model never signalled completion and its
next response did not parse. The served model is a
specialised machine learning model (natural language →
SQL); its measured live behaviour on this seat is to
answer with identity text instead of the §8 JSON
proposal document. **That is a model-output-content
failure. No §15.2 key — attempts, batch, timeout,
freshness, perception bytes, response bytes — governs
what the model says**, and §4.6 forbids landing a
change the evidence does not motivate. The honest L-4
output is therefore the empty list in §3.

## 3. Tuning list

**`tuning_changes: []` — empty, explicitly.**

| Key | Before | After | Journal evidence |
|---|---|---|---|
| *(none)* | — | — | Per-key assessments §2.1–§2.6: no cap bound any SI-74 failure (P-4 fold: attempts seq 9, 16, 23; responses all stage-1 parse failures. P-5 fold: attempts seq 9, 17, 24, 36; batches seq 10, 25 of size 1; gate self-edits seq 12 `se_0001`, seq 27 `se_0002`; timeout seq 36→37 under the fixture's 900 s override, not the default). |

No code, no policy default, no fixture value, and no
test changes in this slice. `si/gaia/policy.py` and
`si/gaia/live.py` are byte-identical to `main`. Because
nothing was tuned, §4.6's post-tuning re-run clause
does not trigger: there is no new record to produce,
and no prior record — including
`docs/SI-74-measurements.json` — is edited by this
slice.

## 4. Tests

Docs-only slice; builder ran the affected proposer
subset only (full suite is the reviewer's, next run):

- `tests/test_gaia_proposer_p4.py` +
  `tests/test_gaia_proposer_p5.py`: **37 passed** on
  `~/workspace/si2-venv` (Python 3.12.3), this run.
- No test files are added or modified; there is no new
  code to pin.

## 5. What this does and does not unlock

- **L-4 (§4.6) is complete** with the empty tuning
  list as its outcome. The L-line's tuning question is
  answered for the SI-74 evidence: the loop's live
  failures are not a policy-tuning problem.
- **L-3 (§4.5) stays LOCKED.** L-3 is conditional on
  L-2 *success*, defined in §4.4 as at least one
  fixture task completing end to end. SI-74 completed
  0 of 3 in both runs, so that condition is not met.
  This slice does not build L-3, does not run a
  residency-loss drill, and §16.6 behaviour stays
  **UNMEASURED**. Unlocking L-3 would require a future
  live run in which a fixture task actually completes
  — a model/seat question, not a tuning one (§2.7).
- Nothing here strengthens any claim: §18.3 stays
  **NOT earned**, P5-R1 and P5-R9 stay **UNMEASURED**,
  and the scoped SI-74 promotions stand exactly as
  SI-74 left them.

## 6. Repo / seed status at run time

- `main` tip `23cb99cb6ebe1f52bf3932dd48ba92de6c934c71`
  (PR #80 merged, SI-74); no open PRs before this slice.
- Glimmer seed bundle present at
  `~/workspace/sentinel-glimmer.si/`, by reference,
  `stored_bytes` 0 — unchanged; nothing in this slice
  duplicates bytes.
- Disk (home volume): 200G total, 101G free (50%).

## Honesty labels carried forward

- Live model-authored tool intent: **OCCURRED,
  scoped** — SI-74 §0/§3.2 only (2026-11-02, P-5 live
  run, 2 intents, 1 executed).
- §18.3 claim: **NOT earned** — 0 of 3 tasks completed
  end-to-end in each SI-74 run.
- Real Hearth/gaia path performance: **UNMEASURED**.
- Residency-loss behaviour (§16.6): **UNMEASURED**;
  L-3 locked (§5).
- Model quality beyond the SI-74 fixtures:
  **UNMEASURED**.
- Tuning changes landed by L-4: **none** — empty list,
  §3.
