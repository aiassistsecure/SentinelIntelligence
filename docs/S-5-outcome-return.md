# S-5 — Gaia seam Flow 3, SI side: outcome return + reconciliation (SI-59)

**Status:** implemented on `s-5/outcome-return` against main
`f8721e9` (SI-58 merged as PR #64). This is the SI half of G-9
§11 / §15's **S-5**: execution outcomes flow back across the
seam from the gaia fork and are journaled SI-natively, as
provenance. §11.3's rule is honoured literally — the poller
and the reconciler ship together or not at all; here they ship
together. §0 says plainly what is live and what is not.

| G-9 element | S-5 state |
| --- | --- |
| §11.2 primary path: SI pulls status | **live, SI side, explicit step** — `si/gaia/seam_outcome.py:poll_seam_outcomes` GETs `{work_url}/{work_id}` for every journaled submission with no journaled outcome; no daemon, no timer, the poll interval is an UNMEASURED policy constant this slice does not set |
| Terminal test | **strict** — HTTP 200, `state` `executed`, and a `result` object whose `outcome` is `executed`/`refused`/`failed` with a string `tool` (the fork's `finish` shape); anything else journals no outcome — the poller never completes a work record by inference (§13.3) |
| §12.5 outcome envelope | **live** — an `experiences` row of kind `tool_result`, `source` `gaia-seam`, the seam pair `(work_id, journal_seq)`, `gate_verdict` `ADMIT`, the fork result **verbatim** under `fork_result`, and under `execution` the same result plus SI's one fold key `status` (= `outcome`); the row's `self_edit_id` is the intent's edit, so `build_provenance_trail` (unmodified) collects it and the queue fold learns it through `_apply_outcome` |
| Dedup (§7.2) | **live** — at most one journaled outcome per `work_id`; a repeat poll spends no transport call, a repeat reconciliation cross-checks |
| §11.3 reconciler: tail `outcomes.jsonl` | **live, read-only** — `reconcile_seam_outcomes` tails the fork's durable outcome file (opened read-only, never written) from a caller-held byte cursor that advances only past journaled / cross-checked / faulted / unmatched records; agreements cross-check, disagreements and distinct second outcomes journal as `gaia_outcome` integrity faults and never rewrite the journaled outcome; unmatched `work_id`s are counted, never journaled; a torn or unparseable line stops the tail at its offset, is counted, and is retried next tail |
| §11.4 outcome-unknown | **live as an explicit operator act** — `mark_seam_outcome_unknown` verifies a journaled submission and no journaled outcome, then journals the intent `outcome-unknown` via the existing SI-50 §13.3 writer (`queue_outcome_unknown`, pair `(work_id, journal_seq, caused_by)` named, evidence journaled as a `queue_recovery_observation`); never re-executes, never resubmits; `seam_reproposal_blocked` reports from the fold whether a `caused_by` is cited by an undisposed seam outcome-unknown |
| Dispositions (§15 S-5 pin) | **pinned: existing acts only** — dispositions remain SI-50 §13.4's operator acts (`si.gaia.queue.disposition_outcome_unknown`, already exposed by the B-5 service); S-5 adds **no** new disposition writer. No third option was taken |
| §11.5 ordering claims | **as spec'd** — at most one journaled outcome per `work_id`; every readable fork outcome is journaled or faulted; outcome journaling never precedes the fork's own outcome record. NOT claimed: outcome latency, cross-work ordering |
| Service surface | **wrappers only** — `si.gaia.service:poll_outcomes` / `reconcile_outcomes` / `mark_outcome_unknown` and a `seam_outcome_return` description key; no new HTTP route (outcome return is outbound / file-mediated and operator-invoked, like S-3/S-4) |
| Live fork interop | **UNMEASURED** — no live fork endpoint was exercised; evidence is the labelled replay/stub transport and a loopback HTTP test double through the real `HttpTransport`, plus operator-supplied `outcomes.jsonl` fixtures in the fork's documented record shape. An env-gated live test (`SI_S5_WORK_URL`) follows the S-3/S-4 pattern and skips when unset |
| Performance / outcome latency | **UNMEASURED** |

---

## 1. What was built

### 1.1 The outcome core — `si/gaia/seam_outcome.py` (new)

- **Poller (§11.2).** Scans the journaled `gaia_work`
  submissions, skips any with a journaled outcome, polls the
  rest once each, and journals terminal outcomes with
  manifest re-anchoring per outcome (the S-3/S-4 pattern).
  Non-terminal states report `awaiting`, 404 reports
  `not-found` (§13.2's restart signal — evidence for §11.4,
  never auto-marked), and non-200 / transport failures report
  `poll-failed`; all three journal nothing and retry free.
- **Reconciler (§11.3).** As the table states. The cursor is
  returned, not stored: the caller persists it, and this
  module invents no cursor store. Fault dedup is by
  `(work_id, line_offset)` — the same conflicting line
  re-tailed faults once; a distinct conflicting line faults
  again, as §11.2's "distinct second outcome" requires.
- **Outcome-unknown (§11.4).** As the table states.
  Fail-closed throughout: an unknown `work_id`, or one with
  a journaled outcome, is refused with nothing journaled; a
  repeat mark returns the existing entry deduped.

### 1.2 Tests — `tests/test_gaia_s5_outcome_return.py` (new)

12 tests (11 passed, 1 skipped — the env-gated live test):
poll + journaling with the verbatim fork result and spine
refs; provenance-trail collection and fold-to-executed;
second-poll no-op with zero transport calls; non-terminal /
404 / 500 / transport-failure honesty with a free retry; a
leash-refusal result folding `refused-at-claim`; reconciler
post-restart recovery from `outcomes.jsonl` alone, with
re-tail cross-check; disagreement and distinct-second-outcome
integrity faults (journaled outcome unrewritten, fault dedup
by offset); unmatched `work_id` counting; the torn final line
(cursor held at its offset, counted, still torn-class after a
bogus completion); outcome-unknown marking, the `caused_by`
block, dedup, and the existing §13.4 `accept-unknown`
disposition lifting the block; outcome-unknown refusals; and
the loopback HTTP test double through the real HTTP
transport.

## 2. Honest gaps and what remains unbuilt

- **Ingest enforcement of the §11.4 re-proposal block is not
  changed by S-5.** `seam_reproposal_blocked` reports the
  block as a journal fact; proposal ingest itself still
  dedups by `caused_by` (§8.3.2 returns the original ACK, so
  no second intent is ever created for the same `caused_by`
  while the original stands), but ingest does not consult
  the outcome-unknown fold. Named here so no reader assumes
  an enforcement this slice did not build.
- **The historical S-3 label strings are untouched.**
  `SUBMITTED_STATE` still reads `submitted/awaiting outcome
  (S-5 unbuilt)` in already-journaled `gaia_work` rows and in
  S-3's label — that was the truth at submission time, and
  S-5 does not rewrite journaled history or the tests that
  pin it. The S-5 truth lives in this doc, in
  `OUTCOME_RETURN_LABEL`, and in the service description.
- S-6 (SI journal events into fork SSE, `source=si`), live
  fork interop, heartbeat/poll cadence automation (outcomes
  are polled when invoked; no daemon), and all performance
  numbers: **UNBUILT / UNMEASURED**, per G-9 §15's
  "UNBUILT after S-5" list.
