# S-4 — Gaia seam Flow 2b, SI side: leash crossing (SI-57)

**Status:** implemented on `s-4/leash-crossing` against main
`8430b43` (S-3 merged as PR #62). This is the SI half of G-9
§10 / §15's **S-4**: the operator leash — journaled
`queue_stop` / `queue_resume` acts on a model's seam task —
crosses the seam to the gaia fork, and is honoured on the SI
side before any seam work is submitted. The fork half (S-4a:
`POST /gaia/v1/leash` + `POST /gaia/v1/leash/heartbeat`,
watermark application, staleness refusal in the fork's
actuator) was built fork-side first, per §15's "fork first"
ordering; this slice does not re-implement or re-claim it.
§0 says plainly what is live and what is not.

| G-9 element | S-4 state |
| --- | --- |
| §10.1 leash lives in SI's journal | **unchanged truth, now read by the crossing** — `si/gaia/seam_leash.py` derives every record from journaled `queue_stop`/`queue_resume` entries on seam tasks; it mints no record |
| §10.2 scope mapping (one seam task per model) | **live** — seam tasks are discovered by the seam operator/submitted-via identity plus equality on the deterministic §10.2 instruction; a stop on that task names exactly that model |
| §10.3.1 event records → `POST /gaia/v1/leash` | **live, SI side, explicit step** — `sync_seam_leash` sends each uncrossed record as `{model, kind, journal_seq, reason}` (§12.3, field-for-field the fork's `LeashRecord`); no daemon, no timer |
| §10.3.2 watermark | **fork-side application; SI-side respect** — a 200 `{applied: false, last_leash_seq}` is a confirmed crossing, not an error; sync dedups by `journal_seq` and never sends a model's later record past an unconfirmed earlier one |
| §10.3.3 watermark piggyback | **live** — every S-3 work payload now carries `leash_tip` (latest leash seq for the model, or the seam task's creation seq when none) |
| §10.3.4 heartbeat → `POST /gaia/v1/leash/heartbeat` | **live, SI side, explicit step** — `send_leash_heartbeat` sends `{model, leash_tip, journal_tip}` (§12.3) and journals the confirmed delivery; the 10 s / 30 s interval/TTL are UNMEASURED fork-side policy constants — this slice claims no cadence and adds no timer |
| §10.5 snapshot re-sync | **live, SI side** — `send_leash_snapshot` sends the model's current folded state as the single latest record; a model with no leash record has nothing to snapshot and none is invented |
| Leash honoured SI-side | **live** — `submit_seam_edit` folds the journal before submitting: a stop in force for the edit's seam task returns `leash-held` (no transport call, nothing journaled by the refusal, the edit stays `pending`); a journaled resume re-enables submission. §10.3.5's `hearth.notify` exemption is a fork-actuation rule and is **not** applied to SI-side submission — named asymmetry, see the module docstring |
| Journaling a crossing | **live, confirmed-only** — an `experiences` row of kind `gaia_leash` (payload + ACK verbatim, transport label) is written only on a 200 carrying a boolean `applied` and an int `last_leash_seq`; non-200 / malformed 200 / transport failure journals no crossing and a retry is free; manifest re-anchored after journaling |
| §11 Flow 3: outcome return (S-5) | **not built** — no outcome returns; a submitted item is still `submitted/awaiting outcome (S-5 unbuilt)`, and a fork-side leash refusal would be an outcome — SI claims none |
| Live fork interop | **UNMEASURED** — no live fork endpoint was exercised; evidence is the labelled replay/stub transport and a loopback HTTP test double through the real `HttpTransport`. An env-gated live test (`SI_S4_LEASH_URL`) follows the S-2a/S-3 pattern and skips when unset |
| Performance | **UNMEASURED** |

---

## 1. What was built

### 1.1 The crossing core — `si/gaia/seam_leash.py` (new)

- **Journal-derived records.** `seam_tasks` finds the seam
  tasks; `leash_records` turns a task's journaled
  `queue_stop`/`queue_resume` entries into §12.3 records in
  journal order (`reason`: a stop's journaled `cause`, else
  its `detail`; a resume's `detail`).
- **Three explicit sends.** `sync_seam_leash` (the §10.3.1
  event feed, in-order per model, gap-stopping),
  `send_leash_snapshot` (§10.5), `send_leash_heartbeat`
  (§12.3) — each operator-invoked via an injected transport or
  an operator-supplied URL; the heartbeat URL is derived from
  the leash URL only when it ends with the documented
  `/gaia/v1/leash` path. No host is defaulted; no path beyond
  §12.3's two documented routes is used.
- **Confirmed-only journaling + dedup**, exactly as the table
  above states, serialised per process like S-3's submit.

### 1.2 The leash honoured in submission — `si/gaia/seam_submit.py`

`submit_seam_edit` now consults the fold after eligibility and
before any transport call (see the table), and
`build_work_payload` carries the §10.3.3 `leash_tip`
piggyback. `WORK_SUBMISSION_LABEL` and the module docstring
now state the S-4 truth; S-5 stays labelled UNBUILT.

### 1.3 Tests — `tests/test_gaia_s4_leash_crossing.py` (new)

12 tests (11 passed, 1 skipped — the env-gated live test):
journal reading of seam tasks/records; stop-blocks-submission
+ resume-re-enables (with the piggybacked tip naming the
resume's seq, and the task-creation-seq default); sync +
confirmed journaling + dedup; the watermark no-op ACK counted
as confirmed; unconfirmed crossings journaling nothing with a
free retry; in-order sync stopping at a gap; the snapshot
sending only the current folded state (and refusing to invent
one); the heartbeat's tips; and the loopback HTTP test double
through the real HTTP transport. No service wrapper or HTTP
route is added: crossing is outbound and operator-invoked,
like S-3 submission (whose `submit_work` docstring now notes
the leash refusal).

## 2. What remains unbuilt

S-5 (outcome return — until it lands, a submitted item is
`submitted/awaiting outcome` forever, by label and by fact,
and no fork actuation outcome, leash refusal included, is
claimed at SI), any SI-service HTTP route for crossing
(outbound and operator-invoked here), heartbeat cadence
automation (heartbeats are sent when invoked; no daemon),
live fork interop, and all performance numbers:
**UNMEASURED**.
