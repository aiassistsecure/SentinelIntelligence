# SI-116 — P-1: the resolution-spend bridge

**Build record.** SI-115 diagnosed that nothing consumes a
journaled escalation resolution: leg (a) resolution → fork
awareness was designed NO (G-9 §9.4/§16.1 — ESCALATE terminal
at ingest, resolution UNBUILT), S-3's caller refuses any
verdict but `ADMIT` and speaks SI's own payload design, and
the SI Executor refuses the `external` class by
construction. SI-115 §4 prescribed the smallest bridge, and
the lead's rulings bound it: D-115-1 the spend trigger is an
explicit CLI act (no watcher, no auto-spend, no timer);
D-115-2 `se_0012`'s completion is its designed `stale_state`
refusal, journaled in P-2; D-115-3 the fork's as-built
`parse_intent` shape is schema authority and SI adapts.
This slice builds the bridge. It runs nothing live: P-2,
the lead-run spend against `se_0012`, is a separate
operational act after this PR merges.

## 1. What was built

`si/gaia/seam_spend.py` (new), with thin service wrappers
(`service.submit_resolved_work` / `poll_spent_outcome` /
`reconcile_spent_outcome`) and CLI wiring on the existing
`si gaia-escalation` family (§3).

### The spend (`spend_resolved_escalation`)

Given an intent seq and a named operator:

1. **Fold-verify** (`verify_spendable`, reusing the fold's
   own fields — no checks invented): the intent folds
   `queued`, its `resolution` is `admit-after-review`, and
   `resolution_seq` is present. Unknown intent, unresolved
   escalation, `reject`-resolved intent, an intent no
   longer `queued` (claimed / executed / held / concluded),
   a non-seam-crossing tool, an empty operator, or a
   missing journaled self-edit row ⇒ `ValueError`, and
   **nothing is journaled** — the `resolve_escalation`
   discipline, verbatim. A leash stop in force for the
   intent's seam task refuses as `leash-held` (S-4's check,
   reused), also journaling nothing.
2. **Dedup**: a journaled `gaia_work` submission already
   naming this intent seq returns that submission
   (`deduped: true`) and calls no transport. The fork's
   S-1 dedup (`journal_seq` → `work_id`) is the second
   layer: a resubmission that somehow reaches the fork
   returns the existing record, so nothing actuates twice.
3. **Payload** in the fork's as-built shape (§2), POSTed
   to the operator-supplied work URL through an injected
   transport (or `HttpTransport`); the library invents no
   default endpoint (S-3's discipline).
4. **Journal**: on 202 + `work_id`, a `gaia_work`
   experience row exactly as S-3 journals submissions —
   `status: submitted`, `state: submitted/awaiting
   outcome` — carrying the bridge's additions: `intent_seq`,
   `resolution` (`admit-after-review`), `resolution_seq`,
   `operator`, `basis`, `task_seq`, `submitted_via`
   (`gaia-seam-spend (G-9/SI-116 P-1)`), the payload and the
   fork's response verbatim; then the manifest re-anchors,
   as S-3/S-5 do. The row's `(intent_seq, resolution_seq,
   fork_work_id)` triple is the Flow 3 reconciliation
   anchor.
5. **Failure is evidence**: a non-202 answer, a 202 without
   a `work_id`, or a transport exception journals a
   `gaia_work` row of `status: spend-failed` with the
   status / body / failure, the payload sent, the operator,
   and the intent + resolution seqs, and reports it. The
   act never retries silently. A failed row is not a
   submission: a later operator re-invocation is free, and
   fork-side S-1 dedup guards the resubmission.

### The outcome leg (`poll_spent_outcome` / `reconcile_spent_outcome`)

SI-115 found leg (d)'s SI poller library-only with no
caller for a spent resolution. These are intent-scoped
wrappers over S-5's existing machinery
(`si.gaia.seam_outcome`): the poller GETs
`{work_url}/{work_id}` for the one work id the spend
journaled and journals a terminal fork status through
S-5's own `_journal_outcome` — once per `work_id`, in
S-5's §12.5 envelope, unmodified (this slice adds no
outcome writer); the reconciler runs S-5's read-only
`outcomes.jsonl` tail (shared cursor, caller's to persist)
and filters its view to the intent's work id. An
already-journaled outcome returns deduped without polling;
non-terminal / 404 / failure answers journal nothing and
report as data; an intent with no journaled spend is
refused.

## 2. The fork wire shape as adopted (D-115-3)

Read from the fork's source (`gaia-api/src/gateway.rs:170
parse_intent`, `work.rs`), not from SI's prior assumptions:

- Route: `POST /gaia/v1/work` (status: `GET
  /gaia/v1/work/{id}`). The live agency address is
  `http://127.0.0.1:11435/gaia/v1/work`.
- Payload: `{"type": "tool", "tool", "args", "reason",
  "task_id", "journal_seq", "verdict": "admit"}` —
  `type`/`tool`/`reason`/`task_id` required strings,
  `journal_seq` a required non-negative integer (the
  intent's self-edit seq, G-9 §9.2 — never the resolution
  seq), `args` verbatim (defaults to null when absent),
  `verdict` the lowercase wire spelling `admit`.
- No `class` field is sent: the parser defaults an absent
  class to `execute`, which the running store carries —
  SI-115 §2c's priced path (claim → `stale_state` refusal,
  not a class 409).
- `task_id` is `task@<task_seq>` (SI-115 §4's priced
  example: `"task_id": "task@43"` for the seam task).
- Answer: 202 `{"work_id", "state": "accepted",
  "journal_seq"}`. Status answers are `status_body`:
  `{work_id, state, provenance {si_journal_seqs,
  spine_event_refs}, result?}` with `state` ∈
  accepted / awaiting_verdict / admitted / rejected /
  executed; `result` present only once executed, carrying
  the fork's actuation result verbatim (`outcome` ∈
  executed / refused / failed) — the shape S-5's
  `_terminal_result` already prices.

## 3. CLI surface (design choice, recorded)

The spend extends the `gaia-escalation` family rather
than a new command: the escalation lifecycle is one
operator story (list → resolve → spend → poll), and
D-115-1 keeps every step an explicit, named operator act
on the one surface the operator already uses for these
intents. Transport, never decision, exactly as SI-113
built the trigger: every verification lives inside the
acts (`spend_resolved_escalation` / S-5), the trigger adds
none.

```
si gaia-escalation <bundle> spend <intent_seq> \
    --operator NAME [--basis TEXT] [--work-url URL] \
    [--task SEQ] [--json]
si gaia-escalation <bundle> poll-outcome <intent_seq> \
    [--work-url URL] [--json]
si gaia-escalation <bundle> reconcile-outcome <intent_seq> \
    --outcomes PATH [--cursor N] [--json]
```

`--work-url` defaults (CLI only) to the live node's route
above; the library requires the URL or a transport
explicitly.

## 4. Tests

`tests/test_gaia_si116_spend_bridge.py` (new): the
eligibility table (unresolved / reject-resolved / unknown /
non-queued / non-seam tool / anonymous — each refused,
journal unchanged); the golden payload (field-for-field
against `parse_intent`'s required set, built from a real
journaled + resolved seam intent); dedup (second spend:
same `work_id`, no second transport call); fork-refusal
and transport-failure journaling as `spend-failed`
evidence (and a later operator retry then succeeding);
the gaia_work row's tie fields; CLI parsing for the three
new subcommands on both parsers; and a loopback
end-to-end: an in-process HTTP stub implementing
`parse_intent`'s rules + the 202 ACK + S-1 dedup by
`journal_seq` + the status route (claim → `stale_state`
refusal → executed with a `refused` result), driven over
real loopback HTTP — spend → poll → outcome journaled
once, folding `refused-at-claim`, a second poll deduped,
a second spend deduped, and the intent-scoped reconciler
cross-checking the same outcome from an `outcomes.jsonl`.

## 5. Honest labels and residue

- This bridge spends resolutions; it does not itself
  execute tools. A spend ACK (202 + `work_id`) is not an
  execution outcome; outcomes return through the S-5 leg.
- Live interop is **UNMEASURED**: nothing in this slice
  called the real node; the fork double is a loopback stub
  implementing the fork's as-built rules from source, and
  transports are labelled (replay / loopback).
- No fork change was needed or made. The one residue,
  recorded rather than posed as a decision: S-5's §12.5
  outcome envelope carries `gate_verdict: "ADMIT"` as its
  pinned literal (S-5 as built); a spent resolution's
  outcome row therefore reads the resolution's *effect*
  (admitted) — the resolution's provenance is carried
  exactly by the `gaia_work` row's `resolution_seq` and
  the outcome row's `(work_id, journal_seq)` pair, and the
  self-edit's ESCALATE verdict is never rewritten anywhere.
  Changing S-5's pinned envelope is a separate slice's
  question, not P-1's.
- §18.3 remains **NOT earned**. P-2 (the lead-run live
  spend against `se_0012`, priced to earn the designed
  `stale_state` refusal per D-115-2) follows after this PR
  merges; until then `se_0012` rests exactly as SI-114 left
  it: queued, shot intact.

— SI-116 coordinator, 2026-11-13. Build only; the lead
reviews, merges, and runs P-2.
