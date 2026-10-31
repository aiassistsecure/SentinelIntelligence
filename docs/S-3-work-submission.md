# S-3 — Gaia seam Flow 2, SI side: work submission (SI-56)

**Status:** implemented on `s-3/work-submission` against main
`aa02b07` (S-2b merged as PR #61). This is the SI half of G-9
§9 / §15's **S-3**: a seam-crossing intent that was ADMITted at
ingest and rests `pending` is submitted across the seam to the
gaia fork's work-submission surface, and the submission is
journaled. §0 says plainly what is live and what is not.

| G-9 element | S-3 state |
| --- | --- |
| §9 Flow 2: work submission, SI → fork | **live, SI side, explicit step** — `si/gaia/seam_submit.py:submit_seam_edit` / `submit_pending_seam_work`, wrapped by `si.gaia.service:submit_work`; ingest does NOT auto-submit |
| Fork contract relied on | **202 + stable work ID only** — the contract stated for this slice; the SI repo documents no gaia-api path and no response field name, so the submission URL is operator-supplied and no path is invented |
| Journaling the submission | **live** — an `experiences` row of kind `gaia_work` (the kind S-2a anticipated), via the existing journal machinery, carrying edit id, self-edit seq, tool, `caused_by`, fork work ID + the field it came from, 202 status, transport label, verbatim payload/response; manifest re-anchored after journaling |
| Submitted item's state | **`submitted/awaiting outcome (S-5 unbuilt)`** — never "completed"; the self-edit's own `result` stays `pending` |
| Dedup | **live** — one edit, one submission: a repeat attempt returns the journaled submission and calls no transport |
| Failures | **honest** — non-202, a 202 without a work ID (`submission-unconfirmed`), and transport failures journal no submission; the edit stays pending and a retry is free |
| §10 leash crossing (S-4) | **built after this slice by SI-57** (docs/S-4-leash-crossing.md) — at S-3 time no leash record crossed with a submission and fork-side leash state was not consulted; S-4 now refuses submission SI-side while a stop is in force and carries the §10.3.3 leash tip in the payload |
| §11 Flow 3: outcome return (S-5) | **not built** — no outcome returns; no fork outcome is claimed anywhere |
| Live fork interop | **UNMEASURED** — no live fork endpoint was exercised; evidence is the labelled replay/stub transport and a loopback HTTP test double through the real `HttpTransport`. An env-gated live test (`SI_S3_WORK_SUBMISSION_URL`) follows the S-2a egress-probe pattern and skips when unset |
| Performance | Measured under S-7 (SI-62): median 120 ms (range 108–164, N=30) for Q-B5 work submission invocation (success, loopback) on the miniature fixture, embedded journal, Class L transport, double service median 0.3 ms, host fingerprint in docs/S-7-seam-measurements.md; single-run; live fork interop performance remains UNMEASURED. Live fork acceptance remains **UNMEASURED**. |

---

## 1. What was built

### 1.1 The submission core — `si/gaia/seam_submit.py` (new)

- **Eligibility, fail-closed.** A self-edit is spendable only
  if its journaled row shows: verdict `ADMIT`, `result`
  `pending`, tool in `SEAM_CROSSING_TOOLS`, gate execution
  label exactly `TOOL_SEAM_PENDING_EXECUTION`. Anything else
  is refused before any transport call, journaling nothing.
- **Payload from the journal.** `build_work_payload` reads
  the journaled self-edit row (never an in-memory intent).
  The shape is SI's design, not a claimed fork schema.
- **Transport, B-4 pattern.** `si.gaia.hearth`'s transports:
  the real `HttpTransport` for an operator-supplied
  `submission_url`, or any injected callable. Every result
  carries its transport label, so a test-double submission
  can never be read as a fork one. No transport and no URL
  is a refusal, not a default.
- **Confirmation rule.** Only HTTP 202 with a work ID under
  `work_id` / `id` / `workId` (checked in that order; the
  field found is journaled) counts as submitted. See the
  table above for the failure states.
- **Journal + re-anchor.** On confirmation, a `gaia_work`
  experience is recorded against the edit id and the manifest
  checkpoint is re-anchored, as seam ingest does. Ordering is
  named in the module docstring: the call precedes journaling
  (the work ID does not exist before the fork answers), so a
  crash in between could orphan fork-side work SI has no
  record of; dedup covers every journaled submission.
- **Scan.** `submit_pending_seam_work` submits every
  spendable pending seam ADMIT in journal order.

### 1.2 Labels revised — only where S-3 changes the truth

`TOOL_SEAM_PENDING_EXECUTION` and `TOOL_SEAM_ESCALATED_EXECUTION`
(`si/gate.py`), `SEAM_INGEST_LABEL` / `REGISTRY_GAP_LABEL`
(`si/gaia/seam.py`), the `hearth.*` registry descriptions and
the pending result-schema comment (`si/gaia/registry.py`), the
B-5 service description (new `seam_work_submission` key), and
the S-2a/S-2b docs now say: an ADMIT rests pending until S-3's
explicit step submits it; ingest submits nothing; escalated
intents are never submitted; S-5 remains UNBUILT. The
registered tools' result schema is unchanged — its `pending`
shape is the ingest-time record, and a submission is recorded
in the `gaia_work` row, not by rewriting that schema.

### 1.3 Tests — `tests/test_gaia_s3_work_submission.py` (new)

14 tests (13 passed, 1 skipped — the env-gated live test):
submission + journaling with payload provenance; dedup with
zero second transport calls; the pending-scan submitting only
spendable ADMITs; the service wrapper; refusals for ESCALATE,
REJECT, unknown edits, and a missing endpoint; non-202 / 202-
without-ID / transport-failure honesty (including a free
retry after failure); the alternate `id` work-ID spelling,
recorded with its field; and the loopback HTTP test double
through the real HTTP transport.

## 2. What remains unbuilt

S-5 (until it lands a submitted item is
`submitted/awaiting outcome` forever, by label and by fact;
S-4 leash crossing has since been built SI-side by SI-57,
docs/S-4-leash-crossing.md),
any SI-service HTTP route for submission (outbound and
operator-invoked here), live fork interop, and all performance
numbers: **UNMEASURED**.
