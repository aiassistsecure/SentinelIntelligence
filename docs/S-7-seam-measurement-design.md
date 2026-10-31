# S-7 — Gaia seam measurement: design (SI-61)

**Status:** design only. This document designs the seam measurement slice
proposed as **S-7** in G-9 §15. It does not implement a harness, does not
run a measurement, and does not change any number, label, or behaviour in
the merged seam code. Every seam latency, throughput, and interop-
performance quantity named here remains **UNMEASURED** until the
implementation slice designed here (the SI-62 candidate, §13–§14) runs and
reports under the conditions this design specifies.

**Provenance of the proposal.** G-9 §15 proposes S-7 (seam measurements) but
does not design it. That is the state recorded at the end of the S-6 slice
document: `docs/S-6-si-events-forwarder.md` lists, under "UNBUILT after
S-6", "seam measurements — every latency and throughput number for the seam
is UNMEASURED; the proposed S-7 measurement slice is proposed, not designed,
in G-9." This document is the missing design. G-9 itself lives in the gaia
fork (`aiassistsecure/gaia`); this design cites G-9 only where the merged
SI-side slice documents already cite it (§4.5, §7, §8, §9, §10, §11, §12,
§14, §15) and invents no G-9 text.

**What S-2 through S-6 built (all merged, SI side).**

| Slice | PR / doc | Module |
| --- | --- | --- |
| S-2a proposal ingest | `docs/S-2a-proposal-ingest.md` | `si/gaia/seam.py` |
| S-2b hearth registry | `docs/S-2b-hearth-registry.md` | `si/gaia/registry.py` |
| S-3 work submission | `docs/S-3-work-submission.md` | `si/gaia/seam_submit.py` |
| S-4 leash crossing | `docs/S-4-leash-crossing.md` | `si/gaia/seam_leash.py` |
| S-5 outcome return | `docs/S-5-outcome-return.md` | `si/gaia/seam_outcome.py` |
| S-6 event forwarder | `docs/S-6-si-events-forwarder.md` | `si/gaia/seam_events.py` |

All five seam hops are explicit, operator-invoked steps. None of them runs a
daemon, a timer, or a follow loop. That single fact shapes almost every
measurement decision in this design (§2.4, §17 D-3).

**Reading rule.** Where this design names a function, constant, label, or
test, the name is taken from the merged tree at SI main `d7059bc` (PR #66
merged). Where it names a future file, function, or field, the name is a
*proposal for SI-62*, marked "(proposed)" at first use in its section.

---

## Table of contents

- §0 Status and honesty table — what is UNMEASURED today
- §1 Scope, audience, and source map
- §2 Invariants the measurements must not break
- §3 Measurement taxonomy and statistical discipline
- §4 Hop A — S-2: proposal ingest to verdict ACK
- §5 Hop B — S-3: work submission round-trip
- §6 Hop C — S-4: leash check and leash crossing cost
- §7 Hop D — S-5: outcome poll and reconcile latency
- §8 Hop E — S-6: event-forward lag and catch-up rate
- §9 The end-to-end chain — a derived view, not a clock
- §10 Harness design
- §11 Sampling, warmup, and reproducibility
- §12 Environment fingerprint and results format
- §13 Execution plan for the implementation slice (SI-62 candidate)
- §14 Acceptance criteria for SI-62
- §15 Label replacement rules
- §16 Explicit non-goals
- §17 Risks and open questions — decisions SI-62 must make
- §18 Alternatives considered and rejected
- §19 Glossary
- Appendix A Per-hop measurement checklists
- Appendix B Results document skeleton
- Document tail and verification

---

## §0 Status and honesty table — what is UNMEASURED today

This table is the contract the rest of the design fulfils. Each row names
one seam hop, the quantity a reader of the S-2..S-6 documents would
reasonably want, where the UNMEASURED label lives today, and what kind of
measurement this design specifies for it: SI-side only, SI-side with a
loopback double, or requiring a fork half that does not yet exist in the
required form.

### §0.1 Hop A — S-2 proposal → verdict

| Quantity | Today | Design class |
| --- | --- | --- |
| Ingest latency, validation phase only | UNMEASURED | SI-side only (§4.2 Q-A1) |
| Ingest latency, dedup scan | UNMEASURED | SI-side only (§4.2 Q-A2) |
| Ingest latency, gate evaluation | UNMEASURED | SI-side only (§4.2 Q-A3) |
| Ingest latency, journal + re-anchor | UNMEASURED | SI-side only (§4.2 Q-A4) |
| Ingest latency, end to end in-process | UNMEASURED | SI-side only (§4.2 Q-A5) |
| Ingest latency over HTTP (`POST /gaia/proposals`) | UNMEASURED | SI-side + loopback (§4.2 Q-A6) |
| Ingest throughput (proposals / second, serial caller) | UNMEASURED | SI-side + loopback (§4.2 Q-A7) |
| Dedup-retry latency (original ACK rebuilt) | UNMEASURED | SI-side only (§4.2 Q-A8) |
| Verdict mix under the fixture corpus | not a latency; unreported for seam fixtures | SI-side only (§4.3) |
| Live fork egress → SI ingest latency | UNMEASURED | Requires fork half (§4.5) |

Label locations today: `SEAM_INGEST_LABEL` in `si/gaia/seam.py` ends
"performance UNMEASURED"; `docs/S-2a-proposal-ingest.md` §0 repeats
"Performance: UNMEASURED — no latency/throughput number is claimed for
ingest". S-2b closed the registry gap; it added no performance claim.

### §0.2 Hop B — S-3 work submission round-trip

| Quantity | Today | Design class |
| --- | --- | --- |
| Eligibility + leash-fold check cost | UNMEASURED | SI-side only (§5.2 Q-B1) |
| Payload construction cost (`build_work_payload`) | UNMEASURED | SI-side only (§5.2 Q-B2) |
| Transport round-trip to a 202 + work ID | UNMEASURED | Loopback double (§5.2 Q-B3) |
| Journal submission + re-anchor cost | UNMEASURED | SI-side only (§5.2 Q-B4) |
| Submission end to end, per edit | UNMEASURED | SI-side + loopback (§5.2 Q-B5) |
| Pending-scan cost vs pending count (`submit_pending_seam_work`) | UNMEASURED | SI-side + loopback (§5.2 Q-B6) |
| Dedup short-circuit cost (repeat submission attempt) | UNMEASURED | SI-side only (§5.2 Q-B7) |
| Live fork work-submission latency | UNMEASURED | Requires fork half (§5.5) |

Label locations today: `WORK_SUBMISSION_LABEL` in `si/gaia/seam_submit.py`
ends "live interop and performance UNMEASURED"; `docs/S-3-work-
submission.md` §0 and §2 repeat both labels.

### §0.3 Hop C — S-4 leash check cost

| Quantity | Today | Design class |
| --- | --- | --- |
| Leash fold cost inside `submit_seam_edit` (`seam_leash_block`) | UNMEASURED | SI-side only (§6.2 Q-C1) |
| `leash_records` / `leash_tip` derivation cost vs journal size | UNMEASURED | SI-side only (§6.2 Q-C2) |
| One leash record crossing (`sync_seam_leash`, per record) | UNMEASURED | Loopback double (§6.2 Q-C3) |
| Heartbeat send cost (`send_leash_heartbeat`) | UNMEASURED | Loopback double (§6.2 Q-C4) |
| Snapshot send cost (`send_leash_snapshot`) | UNMEASURED | Loopback double (§6.2 Q-C5) |
| Heartbeat interval / TTL (10 s / 30 s in G-9 §10.3.4) | UNMEASURED fork-side policy constants | Not measurable SI-side (§6.4) |
| Stop journaled → fork actuator refuses (true crossing latency) | UNMEASURED | Requires fork half (§6.5) |

Label locations today: `LEASH_CROSSING_LABEL` in `si/gaia/seam_leash.py`
ends "live interop and performance UNMEASURED"; `docs/S-4-leash-crossing.md`
notes the 10 s / 30 s interval/TTL are "UNMEASURED fork-side policy
constants".

### §0.4 Hop D — S-5 outcome poll / reconcile latency

| Quantity | Today | Design class |
| --- | --- | --- |
| Poll pass cost per outstanding submission (`poll_seam_outcomes`) | UNMEASURED | Loopback double (§7.2 Q-D1) |
| Poll transport round-trip per `work_id` | UNMEASURED | Loopback double (§7.2 Q-D2) |
| Outcome journaling + re-anchor cost per outcome | UNMEASURED | SI-side only (§7.2 Q-D3) |
| Reconciler cost per line / per KiB (`reconcile_seam_outcomes`) | UNMEASURED | SI-side only, file fixture (§7.2 Q-D4) |
| Reconciler cost vs `outcomes.jsonl` size and cursor position | UNMEASURED | SI-side only (§7.2 Q-D5) |
| Outcome latency: fork finishes → SI journals the outcome | UNMEASURED | Requires fork half + a defined poll cadence (§7.5) |
| Outcome-unknown marking cost (`mark_seam_outcome_unknown`) | UNMEASURED | SI-side only (§7.2 Q-D6) |

Label locations today: `OUTCOME_RETURN_LABEL` in `si/gaia/seam_outcome.py`
ends "live interop, outcome latency, and performance UNMEASURED";
`docs/S-5-outcome-return.md` §0 names the poll interval "an UNMEASURED
policy constant this slice does not set", and §11.5 explicitly does not
claim outcome latency or cross-work ordering.

### §0.5 Hop E — S-6 event-forward lag

| Quantity | Today | Design class |
| --- | --- | --- |
| `build_stream_event` construction cost per event | UNMEASURED | SI-side only (§8.2 Q-E1) |
| Journal read cost per event (`read_journal_events` window) | UNMEASURED | SI-side only (§8.2 Q-E2) |
| Forward cost per event (one POST per event) | UNMEASURED | Loopback double (§8.2 Q-E3) |
| Forward pass throughput (events / second, one pass) | UNMEASURED | Loopback double (§8.2 Q-E4) |
| Catch-up rate: backlog of N events drained in one pass / repeated passes | UNMEASURED | Loopback double (§8.2 Q-E5) |
| Event-forward lag per event (journaled → fork-acked) | UNMEASURED; also *undefined* until a push cadence exists | SI-side component + fork half (§8.4) |
| Fork ingest → SSE visibility lag (`source=si`) | UNMEASURED | Requires fork half — fork ingest is UNBUILT (§8.5) |
| Push cadence itself | No daemon/timer exists; cadence is caller policy | Not a measurement (§8.4) |

Label locations today: `EVENT_FORWARD_LABEL` in `si/gaia/seam_events.py`
ends "live interop and performance UNMEASURED" and states the fork half is
"out of scope / not built by this slice"; `docs/S-6-si-events-forwarder.md`
§0 marks the push cadence "an UNMEASURED policy constant this slice does not
set" and performance UNMEASURED.

### §0.6 What §0 does not list

- No quantity for a trained mind proposing intents. No such proposer
   exists (the B-5 service refuses a submission without a script; see
   `si/gaia/service.py`). Seam measurement measures the seam machinery, not
   a mind, and is not evidence about machine learning inference speed.
- No quantity on a non-miniature bundle. Every seam fixture in the merged
   tests is a miniature bundle (`init_bundle` + `seed_tiny_bundle`); S-7
   inherits that scope and says so on every result (§12.3).

---

## §1 Scope, audience, and source map

### §1.1 Scope

S-7, as designed here, measures the five merged SI-side seam hops and
nothing else:

1. proposal ingest (S-2a, `si/gaia/seam.py`),
2. work submission (S-3, `si/gaia/seam_submit.py`),
3. leash derivation, honouring, and crossing (S-4,
   `si/gaia/seam_leash.py`),
4. outcome polling and reconciliation (S-5, `si/gaia/seam_outcome.py`), and
5. journal-event forwarding (S-6, `si/gaia/seam_events.py`),

plus the thin service wrappers in `si/gaia/service.py` (`submit_proposal`,
`submit_work`, `forward_events`, `poll_outcomes`, `reconcile_outcomes`,
`mark_outcome_unknown`) where a hop is exposed through them.

Out of scope by construction: gate internals as a subject in themselves
(they appear only as a phase inside Hop A), the Q-1..Q-3 queue machinery
(already measured on its own terms by SI-54 / Q-4), the SSE event spine
inside SI (SI-47..SI-49; it is not a seam hop), and everything fork-side
except as an explicitly named missing half (§2.5).

### §1.2 Audience

- The SI-62 implementer, who needs per-hop quantities, fixture recipes,
   sample sizes, and acceptance tests precise enough to build without
   re-designing.
- The dev-loop reviewer, who needs to tell, from the results document
   alone, whether a number may replace an UNMEASURED label (§14, §15).

### §1.3 Source map — where each hop actually lives

| Concern | File | Entry points (merged) |
| --- | --- | --- |
| Ingest | `si/gaia/seam.py` | `validate_proposal`, `find_prior_proposal_ack`, `find_seam_task`, `ingest_proposal` |
| Transports | `si/gaia/hearth.py` | `HttpTransport`, `ReplayTransport` |
| Submission | `si/gaia/seam_submit.py` | `build_work_payload`, `submit_seam_edit`, `submit_pending_seam_work` |
| Leash | `si/gaia/seam_leash.py` | `seam_tasks`, `leash_records`, `leash_tip`, `seam_leash_block`, `sync_seam_leash`, `send_leash_snapshot`, `send_leash_heartbeat` |
| Outcome | `si/gaia/seam_outcome.py` | `poll_seam_outcomes`, `reconcile_seam_outcomes`, `mark_seam_outcome_unknown`, `seam_reproposal_blocked` |
| Events | `si/gaia/seam_events.py` | `build_stream_event`, `pending_event_seqs`, `forward_journal_events`, `SeamProtocolError` |
| Journal read for events | `si/gaia/events.py` | `read_journal_events` |
| Service wrappers | `si/gaia/service.py` | wrappers named in §1.1; `ThreadingHTTPServer` handler |
| Miniature fixture | `si/tiny_seed.py`, tests | `seed_tiny_bundle`; `init_bundle(..., journal_backend="embedded")` |
| Prior measurement precedent | `docs/SI-54-q4-measurement.md` | Q-4 harness shape, reporting style, honesty table |

### §1.4 Precedent: SI-54 / Q-4

Q-4 is the project's existing measurement slice and S-7 deliberately follows
its discipline:

- a dedicated harness module, run as a command;
- fresh miniature bundles per repetition, with setup excluded from timed
   regions;
- scripted, named workloads rather than ad-hoc calls;
- median + range reporting, never a single run;
- a machine-readable results record committed beside a human-readable
   results document;
- an honesty table that keeps whatever was not measured labelled
   UNMEASURED; and
- an explicit statement of host contention when the measuring VM is shared
   with other work.

Where S-7 must differ from Q-4, it differs because the seam differs: Q-4
timed in-process machinery with no network at all, while four of the five
seam hops cross an HTTP boundary whose far side, in S-7, is a loopback test
double. §2.5 and §10.3 exist to keep that difference honest.

---

## §2 Invariants the measurements must not break

A measurement that changes the thing it measures, or that can be misread as
a stronger claim than it is, fails this design even if its numbers are
accurate.

### §2.1 Two sequence spaces, never merged

SI has a journal sequence. The fork has a spine sequence. G-9 §2 invariant
2, honoured by every merged slice, keeps them distinct:

- S-2's ACK carries the SI `journal_seq` (and the seam `task_seq`); the
   proposal's `caused_by` carries the fork spine pair `{hash, seq}`.
   Neither is renamed into the other.
- S-6's `build_stream_event` sets `seq` to the SI journal seq and never
   presents it as a spine seq; there is no `hash` key in the `from_si`
   shape.

Consequence for measurement: there is no single sequence axis along which an
end-to-end latency can be read off. Any cross-space duration must be built
from timestamps (§2.2), never from subtracting sequences, and the results
document must state which space each reported sequence belongs to (§12.2).

### §2.2 Clocks: what legitimately lives where

Three kinds of time appear in the seam, and they are not interchangeable.

1. **Monotonic process clock (SI side).** Durations inside one SI process —
   a phase of `ingest_proposal`, a transport call, a forward pass — are
   measured with a monotonic clock (`time.perf_counter_ns`, proposed for
   SI-62). This is the only clock from which S-7 durations may be computed
   in the SI-side and loopback classes.
2. **Journal timestamps (SI side).** Journal rows carry the journal's own
   recorded times where the journal records them. Q-4 already used
   journaled timestamps for residence (claim minus intent). S-7 may use
   journal timestamps only for quantities the journal itself defines (for
   example, time between two journaled rows for the same seam item), and
   must name the fields used. Journal timestamps are wall-clock records
   written at journaling time; they are evidence about ordering and coarse
   lag, not a substitute for monotonic phase timing.
3. **Fork-side times (fork side).** Spine event times, fork work-state
   transition times, and any timestamp inside a fork response body or an
   `outcomes.jsonl` record belong to the fork's clock domain. S-7 treats
   fork-supplied times as *verbatim data*: it may record them, and it may
   report a fork-internal duration if the fork record itself carries both
   endpoints, but it must never subtract an SI wall time from a fork wall
   time (or the reverse) and present the result as a latency. Clock skew
   between two processes — let alone two repositories' test doubles — is
   unmeasured and unbounded in S-7.

Loopback exception, stated narrowly: when both ends run in the same harness
process or on the same host under the loopback double (§10.3), a round-trip
timed entirely on the SI side (send → response received) is a legitimate SI-
side duration. It measures SI + the double + loopback, and is labelled
exactly that way.

### §2.3 Journal-first ordering is part of the quantity

Several seam costs exist *because* of safety ordering that must not be
optimised away or hidden:

- S-2 ACKs only after journaling (§8.3.5 discipline in the S-2a doc):
   ingest latency therefore includes durable journaling and manifest re-
   anchoring. A variant measurement with journaling removed is not an S-7
   quantity.
- S-3 journals a submission only after the fork's 202 (the work ID does
   not exist earlier); its doc names the crash window this creates.
   Submission latency includes both sides of that ordering.
- S-4 journals a crossing only on a confirmed 200 ACK.
- S-5 journals an outcome only for a strict terminal fork status; the
   reconciler never rewrites a journaled outcome.
- S-6 advances its caller-held cursor only after an HTTP 200 per event.

The harness times the real code paths, with those orderings intact. If a
phase breakdown (§3.3) makes a safety cost visible, the results document
reports it as the cost of the guarantee, not as overhead to be apologised
for.

### §2.4 No daemon, no cadence — so some "latencies" are undefined

Every seam hop is invoked explicitly. Therefore:

- There is no background ingest, submission, poll, heartbeat, or forward
   latency to measure. What exists is the cost of one invocation and its
   phases.
- "Outcome latency" in the user-visible sense — fork finishes a work item,
   how long until SI knows — is a function of *when an operator next
   polls*, which is policy, not code. S-7 measures the poll pass itself
   (§7) and defines outcome latency only relative to a stated polling
   schedule (§7.5). It does not invent a schedule and call the result a
   system property.
- The same applies to S-6 event-forward lag (§8.4) and S-4 heartbeat
   freshness (§6.4).

### §2.5 The fork half: named, not assumed

As of this design, on the SI side of the record:

- The fork half of S-6 — `POST /gaia/v1/events/ingest` feeding the fork's
   `source=si` buffer, in `aiassistsecure/gaia` — is **UNBUILT** in the S-6
   slice's own statement. S-7 therefore cannot measure any quantity that
   needs it, and §8.5 lists those quantities as deferred, not estimated.
- Fork halves for S-2 egress, S-3 work submission, S-4 leash endpoints,
   and S-5 status/outcomes are relied on by the merged SI code as
   documented contracts (202 + work ID; `{applied, last_leash_seq}`; the
   `finish` shape; `outcomes.jsonl` records), and the SI tests exercise
   them through `ReplayTransport` and loopback HTTP doubles. Whether a live
   fork endpoint is running and reachable at SI-62 time is an environment
   fact SI-62 must check and record; this design does not assert it. If no
   live fork is available, SI-62 completes in the loopback class (§3.2) and
   the live-interop rows of §0 stay UNMEASURED. That is a successful
   outcome of the design, not a failure of the run.

### §2.6 Measurement must not journal, submit, or forward for real

The harness works on throwaway miniature bundles in temporary directories
(the merged tests' pattern: `tmp_path` + `init_bundle` +
`seed_tiny_bundle`). It must never point a seam function at the Glimmer seed
bundle, at any operator bundle, or at any non-loopback URL. Env-gated live
tests in the merged suite (`SI_S3_WORK_SUBMISSION_URL`, `SI_S4_LEASH_URL`,
`SI_S5_WORK_URL`, `SI_S6_INGEST_URL`, and the S-2a egress-probe gate) stay
gated and unset during S-7 runs unless §17 D-8 is decided otherwise.

### §2.7 Honesty labels are load-bearing

UNMEASURED labels live in code constants (`SEAM_INGEST_LABEL`,
`WORK_SUBMISSION_LABEL`, `LEASH_CROSSING_LABEL`, `OUTCOME_RETURN_LABEL`,
`EVENT_FORWARD_LABEL`), in service description keys, and in the S-2..S-6
docs. S-7 (this design) changes none of them. SI-62 may propose label
changes only through the replacement rules of §15, in its own PR, reviewed
on its own evidence.

---

## §3 Measurement taxonomy and statistical discipline

### §3.1 Quantity kinds

Every S-7 quantity is one of four kinds. The kind is recorded with the
result (§12.2) because the kinds support different conclusions.

- **Phase duration** — monotonic time spent in one named phase of one
   invocation, in-process. Supports: "where ingest time goes at miniature
   scale".
- **Invocation duration** — monotonic time for one complete seam call
   (function entry to return), including any loopback transport it makes.
- **Pass throughput** — items completed divided by pass duration, for
   scan/pass-shaped functions (`submit_pending_seam_work`,
   `poll_seam_outcomes`, a `forward_journal_events` pass, a reconcile
   tail). Reported with the item count; a throughput without its N is not a
   result.
- **Journal-derived lag** — a difference of two journal timestamps for the
   same item (§2.2 kind 2), reported as a lag distribution, never mixed
   with monotonic durations in one statistic.

### §3.2 Evidence classes

Every result carries exactly one evidence class:

- **Class S — SI-side only.** No transport, or a `ReplayTransport` with
   scripted responses where the transport cost is by construction ~zero and
   is excluded or separately named. Claims: SI code cost.
- **Class L — loopback double.** The real `HttpTransport` against a
   loopback HTTP test double implementing the documented fork contract, of
   the kind the merged S-3/S-4/S-5/S-6 tests already run. Claims: SI code
   cost + real HTTP serialisation + loopback, against a double whose own
   service time is measured and reported (§10.3). Never claims: fork
   performance, live interop performance.
- **Class F — live fork.** The real fork endpoint. Available to SI-62 only
   if §2.5's check finds one, and only through the existing env-gated
   pattern. This design specifies Class F quantities so the fork-side
   follow-up has a target list, but SI-62 does not need Class F to succeed
   (§14.4).

### §3.3 Phase breakdown rule

For each hop, the design names phases that partition the invocation. Phases
are measured by instrumenting the *harness's* call sequence where the merged
code already exposes the boundary (separate functions), and — where a phase
boundary lives inside one merged function — by timing the closest exposed
boundaries and reporting the remainder as a named residual. SI-62 may add
timing instrumentation inside seam modules only if §17 D-1 decides for it;
the default is harness-side timing with residuals, so the seam code under
measurement is the seam code that ships.

Phases must sum-check: the results record includes the invocation duration
and the sum of phases; a discrepancy beyond the stated timer overhead is
reported as the residual, not silently absorbed.

### §3.4 Statistics

Following Q-4's reporting shape, tightened:

- Report **median, minimum, maximum, and N** for every duration and lag.
   Report p95 as well when N ≥ 30 for that quantity (§11 sets sample sizes
   so the headline quantities meet that bar).
- Never report a mean without the median beside it; seam durations on a
   contended VM are skewed.
- Never compare two configurations on medians alone when their ranges
   fully overlap; say "indistinguishable at this scale and contention", as
   Q-4 did for write-heavy bounds.
- Rounding: durations in milliseconds with one decimal below 100 ms, whole
   milliseconds above; throughputs to three significant figures. Raw
   nanosecond/monotonic readings stay in the machine-readable record
   (§12.2) so rounding is presentation, not data loss.
- Outliers are never dropped silently. A repetition may be excluded only
   for a recorded cause (for example, the harness itself logged an
   interruption), and exclusions are counted in the results doc.

### §3.5 Units

| Quantity kind | Unit |
| --- | --- |
| Phase / invocation duration | milliseconds (ms) |
| Journal-derived lag | seconds (s), with ms in raw record |
| Throughput | items per second (items/s), with N and window |
| File-size scaling (reconciler) | per KiB and per 1,000 lines, as specified in §7 |
| Journal growth (if reported) | rows added per invocation |

### §3.6 What a number means — the conditions sentence

No S-7 number is valid without its conditions sentence: evidence class,
fixture, bundle scale, journal backend, host fingerprint (§12.1), sample
size, and contention note. §15.3 gives the exact sentence template label
replacements must use. A number quoted without it — in a doc, a PR body, or
a report — is a labelling defect, and the reviewer should treat it as an
unmeasured claim.

---

## §4 Hop A — S-2: proposal ingest to verdict ACK

Code under measurement: `ingest_proposal` in `si/gaia/seam.py`, and the HTTP
route `POST /gaia/proposals` in `si/gaia/service.py`.

### §4.1 The flow being timed

Per the S-2a doc, one ingest is, in order: validate (before the bundle or
journal is opened) → dedup scan of journaled `self_edits` by `caused_by`
(under the module lock) → find-or-create the model's seam task → construct
gate + context → evaluate → journal the self-edit with its full gate report
(+ provenance edges + per-verdict result marking) → re-anchor the manifest →
form the ACK `{task_seq, journal_seq, verdict}`. A dedup hit short-circuits
after the scan and writes nothing.

### §4.2 Quantities

- **Q-A1 Validation duration.** Phase duration, Class S. Timed over both a
   valid fixture proposal and the merged tests' malformed mutations
   (validation cost must not depend on refusing early in a way the fixture
   hides; report valid and refused separately).
- **Q-A2 Dedup scan duration.** Phase duration, Class S, as a function of
   journaled self-edit count. Measured at the journal sizes in §4.3 — the
   scan is a journal scan, so a single size would mislead.
- **Q-A3 Gate evaluation duration.** Phase duration, Class S, split by
   verdict reached (ADMIT / ESCALATE / REJECT) and by tool class, because
   the gate's work differs by path — registered `fs_*` tools, external
   `hearth.*` tools under the S-2b registry, and an unknown tool that
   REJECTs at the `tool_registered` sub-check.
- **Q-A4 Journal + re-anchor duration.** Phase duration, Class S:
   `record_self_edit` with the full gate report, provenance edges, result
   marking, and manifest re-anchor/rewrite, as one phase (the merged code
   does not expose a finer honest boundary) plus the invocation residual
   per §3.3.
- **Q-A5 Ingest invocation duration.** Invocation duration, Class S, per
   verdict class in §4.3.
- **Q-A6 HTTP ingest duration.** Invocation duration, Class L: the real
   service handler on loopback, client-timed send → ACK received, and
   server-side Q-A5 for the same requests, so the HTTP+JSON layer's share
   is visible as a difference of distributions, not of single runs.
- **Q-A7 Serial ingest throughput.** Pass throughput in the degenerate
   single-item sense: sustained sequential `ingest_proposal` calls by one
   caller, items/s over the §11 window. No concurrency claim — the service
   serialises the journal section under a lock by design, and measuring
   "concurrent ingest" as if it parallelised would misdescribe the system.
   (A separate contention observation — two concurrent identical POSTs, per
   the S-2a lock rationale — is an optional correctness-timed observation,
   §17 D-6.)
- **Q-A8 Dedup-retry duration.** Invocation duration, Class S: re-
   ingesting an identical `caused_by` returns the rebuilt original ACK and
   journals nothing; measure it against Q-A5 so the retry cost the fork
   pays after a lost ACK is a number, and verify zero journal growth per
   retry (a correctness assertion riding the measurement, as Q-4 rode
   completeness assertions).

### §4.3 Fixtures and workload shape

- Bundle: fresh miniature bundle per repetition set (`init_bundle` +
   `seed_tiny_bundle`, embedded journal), the S-2a test pattern.
- Proposals: the watcher fixture family from the merged seam tests —
   `hearth.restart` with `args.spine_event` agreeing with `caused_by` —
   plus: a registered-tool control proposal (the `fs_list` ADMIT control
   from S-2a §3), an external-policy ESCALATE case, an unknown-tool REJECT
   case, and the malformed set for Q-A1. Policies: `corpus_policy` and the
   all-admit tool-class policy used by the S-6 tests (`external: admit`
   etc.), named in the record.
- Journal-size axis for Q-A2/Q-A5: pre-ingest 0, 10, and 50 distinct
   proposals (distinct `caused_by`) before timing, so scan and journal
   costs are reported against size, not at a single point.
- Every timed ingest uses a fresh `caused_by` (except Q-A8), or the dedup
   short-circuit silently replaces the measurement with Q-A8. This is the
   hop's main fixture hazard and the harness must assert verdict + journal
   growth per timed call.

### §4.4 Clocks

All Q-A quantities are SI-side monotonic (§2.2 kind 1), except Q-A6's client
leg, which is same-host loopback timed on the client process's monotonic
clock. The `caused_by` spine seq appears in fixtures as data; it is never
used as a time or an axis (§2.1).

### §4.5 Deferred to the fork half

- Fork egress send → SI ACK received, timed on the fork (Class F): the
   true Flow-1 crossing latency.
- Fork restart re-send behaviour under timing (S-2 acceptance item 5 is
   fork-side): not an SI quantity. These stay UNMEASURED after SI-62 unless
   a fork-side run measures them; §15 forbids SI-62's loopback numbers from
   being relabelled as either.

---

## §5 Hop B — S-3: work submission round-trip

Code under measurement: `submit_seam_edit` and `submit_pending_seam_work` in
`si/gaia/seam_submit.py`; transport via `si/gaia/hearth.py` (`HttpTransport`
/ `ReplayTransport`).

### §5.1 The flow being timed

Eligibility check on the journaled self-edit row (verdict ADMIT, result
pending, tool in `SEAM_CROSSING_TOOLS`, execution label exactly
`TOOL_SEAM_PENDING_EXECUTION`) → leash fold check (S-4's `seam_leash_block`;
costed here as a phase and in depth in §6) → payload construction from the
journaled row, including the §10.3.3 `leash_tip` piggyback → transport POST
to the operator-supplied submission URL → on HTTP 202 with a work ID under
`work_id` / `id` / `workId`, journal a `gaia_work` `experiences` row and re-
anchor. Failures (non-202, 202 without a work ID = `submission-unconfirmed`,
transport error) journal nothing and retry free. Dedup: a repeat attempt
returns the journaled submission with no transport call.

### §5.2 Quantities

- **Q-B1 Eligibility + leash-fold duration.** Phase duration, Class S.
   Includes `_eligibility_problems` and the §6 fold; reported separately
   from the rest so a leash-held refusal's cost is known (a refusal is a
   real seam outcome, and its cost is the cost of *not* crossing unsafely).
- **Q-B2 Payload construction duration.** Phase duration, Class S:
   `build_work_payload` over the journaled row, at the §5.3 journal sizes.
- **Q-B3 Transport round-trip duration.** Invocation duration for the
   transport leg only, Class L: real `HttpTransport` to the loopback double
   that answers 202 with each of the three work-ID spellings across
   repetitions (spelling is recorded; the SI code checks them in order, and
   the record should show whether spelling changes anything — expected:
   nothing beyond noise, stated as such if ranges overlap). Class S
   companion: `ReplayTransport` leg, reported as the transport floor (~zero
   by construction).
- **Q-B4 Journal submission + re-anchor duration.** Phase duration, Class
   S.
- **Q-B5 Submission invocation duration.** Invocation duration, Class L
   (and Class S companion), success path; plus the three failure paths
   timed separately (non-202, 202-without-ID, transport exception) —
   failure honesty is a feature of this hop, and its cost is part of the
   seam's real profile.
- **Q-B6 Pending-scan scaling.** Pass throughput and invocation duration
   for `submit_pending_seam_work` at 1, 10, and 50 spendable pending
   ADMITs, with non-spendable rows mixed in (ESCALATE / REJECT / already-
   submitted), because the scan's real cost is finding the spendable ones.
   Report per-item derived cost only alongside the pass totals (§3.1).
- **Q-B7 Dedup short-circuit duration.** Invocation duration, Class S,
   with a zero-transport-call assertion (the loopback double counts
   requests).

### §5.3 Fixtures

- ADMITted seam edits produced the honest way: ingest proposals through
   `ingest_proposal` under the all-admit policy for a seam-crossing tool
   (`hearth.restart`), as the S-3 tests do — not by hand-writing journal
   rows. The setup cost (ingest) is excluded from B timings and reported as
   setup.
- Loopback double: the S-3 tests' double pattern — a loopback HTTP server
   answering the documented contract (202 + work ID; scriptable to the
   failure modes) through the real `HttpTransport`.
- The submission URL is a loopback URL supplied by the harness, exactly as
   an operator would supply it; the module invents no endpoint, and neither
   does the harness (§2.6).

### §5.4 Clocks

SI-side monotonic throughout; Q-B3/Q-B5 Class L legs are same-host loopback
round-trips (§2.2). The fork work ID is an identifier, not a time or a
sequence in either space; it is recorded verbatim for provenance and used as
the join key in §9.

### §5.5 Deferred to the fork half

- Live fork acceptance cost: 202 latency of the real gaia-api work surface
   under real fork state (Class F).
- Any fork-side queueing between 202 and actuation. SI cannot see it; §7
   covers the SI-visible part of outcome return, and the gap between them
   stays named, not interpolated.

---

## §6 Hop C — S-4: leash check and leash crossing cost

Code under measurement: `si/gaia/seam_leash.py`, and its call site inside
`submit_seam_edit`.

### §6.1 The flow being timed

The leash lives in SI's journal as `queue_stop` / `queue_resume` entries on
a model's seam task. SI-side, three things cost time: deriving records from
the journal (`seam_tasks`, `leash_records`, `leash_tip`,
`seam_leash_block`'s fold), honouring a stop before submission (the Q-B1
phase), and crossing records to the fork (`sync_seam_leash` sending `{model,
kind, journal_seq, reason}` per record to `POST /gaia/v1/leash`;
`send_leash_snapshot`; `send_leash_heartbeat` sending `{model, leash_tip,
journal_tip}`), journaling a crossing only on a confirmed 200 `{applied,
last_leash_seq}`.

### §6.2 Quantities

- **Q-C1 Leash-fold check duration.** Phase duration, Class S:
   `seam_leash_block` as called from submission, in three journal states —
   no leash record (tip = task creation seq), stop in force, stop-then-
   resume — at the §6.3 journal sizes. This is the per-submission safety
   tax; it is expected to be small and the design requires reporting it
   even (especially) if it is indistinguishable from noise.
- **Q-C2 Derivation scaling.** Phase durations, Class S, for
   `leash_records` and `leash_tip` against seam tasks carrying 0, 10, and
   100 leash entries, and against a journal whose seam task must be found
   among other models' seam tasks (1 vs 10 models).
- **Q-C3 Per-record crossing duration.** Invocation duration, Class L, for
   one record through `sync_seam_leash`, split into transport leg and
   confirmed-journaling leg; plus the watermark no-op case (`applied:
   false`) timed separately — it is a confirmed crossing with different
   journal meaning. Also: a sync pass over N uncrossed records (pass
   throughput), and the gap-stop case (an unconfirmed earlier record stops
   the pass) timed as a pass that crosses a prefix.
- **Q-C4 Heartbeat duration.** Invocation duration, Class L, for
   `send_leash_heartbeat` including its confirmed-delivery journaling.
- **Q-C5 Snapshot duration.** Invocation duration, Class L, for
   `send_leash_snapshot` (single latest folded record), contrasted with
   Q-C3's per-record cost so the §10.5 re-sync strategy has a price.

### §6.3 Fixtures

Leash entries journaled through the real queue vocabulary on real seam tasks
(the S-4 tests' pattern), never hand-forged rows. Loopback double implements
the documented leash contract: 200 with `{applied: true|false,
last_leash_seq}`, scriptable non-200 and malformed-200 for the (untimed or
separately timed) failure paths. The heartbeat URL derivation rule (from a
leash URL ending in the documented path) is exercised as the merged code
implements it.

### §6.4 The cadence constants are not SI quantities

G-9 §10.3.4's 10 s heartbeat interval and 30 s TTL are, in the S-4 doc's
words, UNMEASURED fork-side policy constants. S-7 measures what one
heartbeat *costs* (Q-C4), not what interval is *right*. Any claim that a
cadence is adequate needs fork-side staleness evidence; §17 D-5 records the
decision SI-62 must make about even proposing one.

### §6.5 Deferred to the fork half

- Stop crossing → fork actuator actually refusing: the safety-relevant
   end-to-end latency. Its SI-side components (journaling the stop is
   ordinary journal cost; Q-C3 crossing) are measured here; the fork's
   watermark application and actuator check are not.
- Fork-side staleness refusal timing under §10's stale-state rules. Both
   stay UNMEASURED after SI-62 unless measured fork-side, and §15 keeps
   S-7's numbers from standing in for them: a fast Q-C3 must never be
   written up as a fast stop.

---

## §7 Hop D — S-5: outcome poll and reconcile latency

Code under measurement: `poll_seam_outcomes`, `reconcile_seam_outcomes`, and
`mark_seam_outcome_unknown` in `si/gaia/seam_outcome.py`.

### §7.1 The flows being timed

Poll: scan journaled `gaia_work` submissions, skip any with a journaled
outcome, GET `{work_url}/{work_id}` once per outstanding submission, and
journal a terminal outcome (strict test: HTTP 200, `state` `executed`,
`result.outcome` in executed/refused/failed with a string `tool`) as the
§12.5 `tool_result` envelope, re-anchoring per outcome. Non-terminal →
`awaiting`, 404 → `not-found`, failures → `poll-failed`; all journal nothing
and retry free.

Reconcile: tail the fork's `outcomes.jsonl` read-only from a caller-held
byte cursor; agreements cross-check, disagreements and distinct second
outcomes journal as `gaia_outcome` integrity faults, unmatched `work_id`s
are counted, a torn final line stops the tail at its offset.

### §7.2 Quantities

- **Q-D1 Poll pass duration vs outstanding count.** Pass duration +
   throughput, Class L, at 1, 10, and 50 outstanding submissions, in three
   mixes: all terminal, all non-terminal (`awaiting`), and mixed with 404s.
   The mixes matter: only terminal outcomes pay the journaling cost, so a
   single-mix number would describe a workload nobody runs.
- **Q-D2 Per-work poll round-trip.** The GET leg only, Class L, per
   terminal/non-terminal/404 answer.
- **Q-D3 Outcome journaling duration.** Phase duration, Class S: the §12.5
   envelope journaling + re-anchor per outcome, including provenance
   collection preconditions (the row keyed by the intent's edit) — timed as
   the merged code does it.
- **Q-D4 Reconciler unit cost.** Invocation duration, Class S, per 1,000
   lines and per KiB of `outcomes.jsonl` tail, split by line outcome:
   agreement, disagreement (fault journaling — a real journal write,
   costed), distinct-second-outcome, unmatched, malformed/torn (tail stops;
   cost is the partial pass, reported as such).
- **Q-D5 Reconciler scaling.** Pass duration against file size (cursor at
   0 over 100 / 1,000 / 10,000 lines) and against cursor position (cursor
   near EOF over the same files), because a byte-cursor tail's steady-state
   cost is the near-EOF case and its recovery cost is the from-0 case; both
   are operationally real.
- **Q-D6 Outcome-unknown marking duration.** Invocation duration, Class S,
   including the verification reads and the §13.3 journaling; plus
   `seam_reproposal_blocked` read cost as a named sub-phase (it is a fold
   read other callers pay).

### §7.3 Fixtures

- Submissions created through §5's honest path (ingest → submit against
   the loopback double), so poll fixtures carry real `gaia_work` rows.
- Status double: loopback server answering the documented status contract
   in the fork's `finish` shape, scriptable per `work_id`.
- `outcomes.jsonl` fixtures generated in the fork's documented record
   shape (as the S-5 tests use), with fixed seeds (§11.3): known counts of
   each line class in §7.2 Q-D4, deterministic byte sizes, and a torn final
   line produced by truncating a real line at a fixed byte, not by hand-
   typing a fragment.

### §7.4 Clocks

Poll and journaling phases: SI monotonic. The reconciler is file-mediated
and in-process: SI monotonic, no network. Timestamps inside `outcomes.jsonl`
records are fork-clock data (§2.2 kind 3): recorded verbatim, never
subtracted from SI times.

### §7.5 Outcome latency, defined honestly

S-7 defines exactly one outcome-latency quantity, and it is conditional:

> **Conditional outcome latency (proposed):** given a stated polling
> schedule (first poll at offset *o* seconds after submission, then every
> *p* seconds, both recorded as harness parameters, not system properties),
> the distribution of (outcome journaled time − fork terminal time) — where
> the fork terminal time is knowable only in the loopback class, from the
> double's own log, on the same host clock.

In Class L this is measurable and worth measuring: it prices the polling
discipline itself (expected value ≈ o + p/2 plus Q-D1's pass cost) and
verifies that model against reality. In Class F it additionally needs the
fork's terminal timestamp, and §2.2 forbids cross-clock subtraction unless
both sides' clocks are the same host's — a condition a live-fork run must
state, not assume. Absent a schedule, the quantity does not exist, and the
S-5 label "outcome latency UNMEASURED" is replaced, if at all, by the
conditional form (§15.2), never by a bare number.

---

## §8 Hop E — S-6: event-forward lag and catch-up rate

Code under measurement: `build_stream_event`, `pending_event_seqs`, and
`forward_journal_events` in `si/gaia/seam_events.py`, reading via
`read_journal_events` in `si/gaia/events.py`.

### §8.1 The flow being timed

One pass: read journal events after the caller-held cursor in ascending seq
order, build each event's `StreamEvent::from_si` payload (source `si`, SI
journal seq, kind, collection, full row verbatim, `row_kind`, `facts`,
derivable `model`/`task` only, no `hash`), POST one event per call to the
operator-supplied ingest URL, and advance the cursor only after each HTTP
200. HTTP 409 raises `SeamProtocolError` (loud, no retry, cursor held);
other failures return `forward-failed` with the cursor at the last acked
event. The forwarder journals nothing. `limit` bounds one pass's window.

### §8.2 Quantities

- **Q-E1 Payload construction duration.** Phase duration, Class S:
   `build_stream_event` per event, across the row kinds a real miniature
   journal contains (the S-6 tests seed via `submit_proposal`, producing
   genuine row variety), reported per collection/kind grouping if counts
   allow, else as one distribution with the mix recorded.
- **Q-E2 Journal read duration.** Phase duration, Class S:
   `read_journal_events` windows of 1, 10, 100, and 500 events at cursor
   positions near the journal's start and near its tip (reads are of a
   live-shaped journal; the S-6 doc's note that the read path does not
   demand manifest-anchor == journal-tip is part of the path being timed).
- **Q-E3 Per-event forward duration.** Invocation duration, Class L: one
   event's POST → 200 through the real `HttpTransport` against the loopback
   ingest double (200 per event; 409 on duplicate seq, per the documented
   contract the S-6 tests implement).
- **Q-E4 Forward pass throughput.** Pass throughput, Class L: events/s for
   passes of 10, 100, and 500 events, unlimited and with `limit` set to the
   pass size, plus the failure-interrupted pass (double answers 500 at
   event k; pass returns `forward-failed`, cursor at k−1) timed as the
   partial pass it is.
- **Q-E5 Catch-up rate.** The operationally named quantity of this hop:
   given a backlog of N events journaled while no forwarding happened (the
   fork "unreachable" interval simulated by simply not forwarding), time
   for repeated passes — and for a single unlimited pass — to drive the
   cursor to the tip, at N = 100 and N = 500. Reported as events/s and as
   passes-to-tip + total wall. Catch-up is where one-POST-per-event (§8.1)
   shows its price, and the design requires the number even if it is
   unflattering.
- **Q-E6 Duplicate-push failure cost.** Invocation behaviour, Class L: re-
   forwarding from a stale cursor raises `SeamProtocolError` on the first
   duplicate; time-to-raise is recorded, but the quantity of record is
   correctness (cursor, pushed list, call count — the S-6 tests' pins),
   with the duration as supporting data.

### §8.3 Fixtures

- A miniature journal grown to known sizes by real activity: repeated
   proposal ingests (§4 fixtures) and task activity, so forwarded rows are
   genuine journal rows, not synthetic dicts. `pending_event_seqs` is the
   harness's ground truth for backlog N before each catch-up run.
- Loopback ingest double: the S-6 tests' double pattern (200 per event,
   409 on duplicate seq) through the real `HttpTransport`. The double
   counts and logs every POST (seq, arrival order) so the harness can
   assert exact seq fidelity across the whole measured pass, not just on
   test-sized runs.

### §8.4 Event-forward lag, defined honestly

Like outcome latency (§7.5), per-event lag needs a cadence to exist. S-7
defines:

> **Per-event forward lag (components).** For one event: (a) *queueing lag*
> — journaled time → pass start — a property of the caller's cadence, not
> measured as a system quantity; (b) *in-pass lag* — pass start → that
> event's 200 — measured, Class L, as a function of the event's position in
> the pass (position 1 vs position N differ by construction in a serial one-
> POST-per-event pass, and the results must show lag *by position decile*,
> not a single average that hides the serial shape); (c) *fork visibility
> lag* — 200 → event visible on the fork's `source=si` SSE — fork-half,
> §8.5.

Journal timestamps (§2.2 kind 2) may be used to report a coarse journal-
derived lag distribution for (a)+(b) combined under a stated harness cadence
(for example, "passes driven back-to-back by the harness"), labelled
journal-derived (§3.1) and never merged with the monotonic (b) statistics.

### §8.5 Deferred to the fork half — the UNBUILT ingest

The fork ingest half of S-6 is UNBUILT (§2.5). Therefore deferred, by name:

- fork ingest service time per event (the real counterpart of the double's
   service time, §10.3);
- ingest → `SiEventBuffer` → SSE emission lag;
- SI's own `events_client` consuming the merged fork stream through drop-
   and-reconnect at measured rates (§15 S-6 acceptance 1 is fork-side);
- any combined multi-source (hearth + si) stream timing. All stay
   UNMEASURED after SI-62.

---

## §9 The end-to-end chain — a derived view, not a clock

A reader will inevitably ask: "proposal appears at the fork → outcome
journaled at SI: how long?" This design's answer is disciplined refusal plus
a well-defined partial product.

### §9.1 Why there is no single end-to-end latency

The chain crosses two processes, two sequence spaces (§2.1), at least two
clock domains (§2.2), one file-mediated return path, and two operator-
invoked gaps (submission is an explicit step after ingest; polling is an
explicit step after fork completion) whose spacing is policy (§2.4). A
single number would encode the harness's own invocation spacing as if it
were the system's latency. That is precisely the kind of invented figure the
seam's honesty labels exist to prevent.

### §9.2 What S-7 does provide: the chain worksheet

The results document includes a **chain worksheet** (proposed): a table
whose rows are the chain's legs, each filled from that leg's own measured
distribution under its own class, with the unmeasured legs shown as
UNMEASURED in place:

| Leg | Source quantity | Class |
| --- | --- | --- |
| Proposal validation → ACK | Q-A5 / Q-A6 | S / L |
| ACK → submission invoked | operator policy | not a system quantity |
| Submission round-trip | Q-B5 | L |
| Fork acceptance → fork terminal | fork half | UNMEASURED (Class F list) |
| Terminal → poll discovers it | conditional, §7.5 schedule | L (conditional) |
| Poll journaling | Q-D3 | S |
| Journal event → forwarded 200 | Q-E3 / §8.4(b) | L |
| Forwarded → visible on fork SSE | fork half (UNBUILT ingest) | UNMEASURED |

A worked *sum of medians* may be shown in the worksheet only with the
caption that a sum of medians is not a median of sums, that policy gaps are
excluded, and that the total describes the harness scenario, not a
deployment. §17 D-7 decides whether even that captioned sum is more
misleading than useful; the default is to publish the worksheet without a
total row.

### §9.3 Correlation keys

Legs are joined for provenance, never for timing arithmetic across clocks,
by the keys the merged code already journals: `caused_by` (proposal), the
self-edit / edit id, the fork `work_id` (with the field spelling it arrived
under), the seam pair `(work_id, journal_seq)` on outcome rows, and the SI
journal seq on forwarded events. The harness asserts these joins hold
across its measured runs — a broken join is a seam defect to report, not a
timing footnote.

---

## §10 Harness design

### §10.1 Shape

One harness module (proposed: `si/gaia/seam_measure.py`, mirroring Q-4's
`si/gaia/measure.py` precedent), invoked as a command (proposed: `python -m
si.gaia.seam_measure --out <dir>`), which:

1. builds fixtures (§10.2) in a fresh temp tree;
2. starts the loopback doubles (§10.3) on ephemeral loopback ports;
3. runs the per-hop measurement plans (§4–§8) at the §11 sample sizes;
4. writes the machine-readable record and a draft results summary (§12.2)
   to the output directory;
5. asserts the correctness riders each hop specifies (journal growth, dedup
   counts, seq fidelity, cursor positions) and records assertion outcomes
   in the record — a run whose riders fail is a failed run, and its timings
   are not publishable.

The harness is a development tool, like Q-4's: it is not added to the
service, exposes no route, and runs no daemon.

### §10.2 Fixtures

- **Miniature bundle.** `init_bundle` + `seed_tiny_bundle`, embedded
   journal backend — the merged tests' fixture, reused, not reinvented.
   Bundle creation and seeding are setup: untimed, but their wall time is
   recorded for context.
- **Activity-grown journals.** Journal-size axes (§4.3, §5.2 Q-B6, §6.2,
   §8.3) are reached by running real seam operations (ingests, submissions,
   leash entries) on the fixture bundle, so every row the measurement
   touches was written by the code under measurement's own family.
- **Policies.** `corpus_policy` and the all-admit tool-class policy from
   the merged tests, named per run in the record.
- **Work roots.** Fresh temp work roots per repetition set (Q-4 pattern),
   so filesystem state from one repetition cannot warm or pollute the next
   beyond what §11.2's warmup deliberately does.
- **Outcomes files.** §7.3's generated `outcomes.jsonl` fixtures, written
   to the temp tree, opened read-only by the reconciler exactly as in
   production shape.

### §10.3 Loopback doubles — and measuring the double

Each double extends the corresponding merged test's double rather than
starting fresh:

| Double | Contract implemented | Source pattern |
| --- | --- | --- |
| Ingest/status/work server (service-level) | the B-5 handler itself on loopback for Q-A6 | `si/gaia/service.py` |
| Work submission double | 202 + work ID (3 spellings), scriptable failures | S-3 tests |
| Leash double | 200 `{applied, last_leash_seq}`, watermark no-op, scriptable malformed | S-4 tests |
| Status double | fork `finish` shape per `work_id`, scriptable state | S-5 tests |
| Events ingest double | 200 per event, 409 on duplicate seq, full POST log | S-6 tests |

Rules for doubles:

- The double's own service time (request parsed → response written, timed
   inside the double) is measured and reported beside every Class L result.
   If the double's median service time is not small against the measured
   round-trip, the result is reported as double-bound and the hop's Class L
   claim is downgraded in the results doc — the reviewer does not have to
   guess.
- Doubles add no artificial delay by default. A fixed-delay variant
   (proposed: 5 ms service time) may be run as a sensitivity check that SI-
   side costs are not an artefact of a zero-cost peer; §17 D-4 decides
   whether it is required.
- Doubles bind loopback only, on port 0 (ephemeral), and log every request
   to the run's evidence dir.

### §10.4 Transports

Class S runs use `ReplayTransport` with scripted responses where a transport
is structurally required but not the subject; Class L runs use the real
`HttpTransport`. Every record names the transport class and, for Class L,
the double and its measured service time (§10.3). A result whose transport
label is missing is invalid — this mirrors the merged code's own discipline
of carrying a transport label on every seam result so a double can never be
read as a fork.

### §10.5 Where the harness may not go

No measurement against: a live fork (unless §2.5's check + env gates are
deliberately engaged under D-8), a non-loopback address, the Glimmer bundle,
a real operator bundle, or any URL taken from anywhere but the harness's own
doubles. The harness refuses non-loopback URLs outright (proposed), in code,
not by convention.

---

## §11 Sampling, warmup, and reproducibility

### §11.1 Environment of record

CPU only, on this VM class: the dev-loop VM (the fingerprint is captured per
run, §12.1 — this design does not hardcode a CPU model or core count as the
requirement, because the VM's allocation has changed before and the
fingerprint, not a spec sheet, is the evidence). No GPU, no cloud hosts, no
other machines.

Python runs from a workspace venv (the loop's standing discipline; Q-4 used
`~/workspace/si2-venv`). The venv path, Python version, and install state
are fingerprint fields, not assumptions.

### §11.2 Warmup

- Each hop's plan begins with an untimed warmup set: at least 3 full
   invocations of the measured call on the same fixture shape, discarded,
   so first-call costs (imports already paid at harness start, JIT — none,
   filesystem cache, journal open) do not land in repetition 1. Q-4's
   ranges show this VM's noise; warmup is necessary and explicitly not
   sufficient.
- The loopback doubles are warmed with at least 3 untimed requests each,
   including one full request/response cycle through `HttpTransport`.
- Bundle/journal setup for a repetition is untimed (§10.2), but a *fresh*
   bundle per repetition set means "warm" never means "the journal file is
   already open in-process across repetitions" unless the merged code
   itself holds it open — the harness calls the seam functions the way the
   service and tests call them, per-invocation journal opens included,
   because that is the cost operators pay.

### §11.3 Sample sizes

Minimums (proposed; §17 D-2 owns the final call):

| Quantity class | Minimum timed repetitions |
| --- | --- |
| Phase durations (per fixture variant) | 30 |
| Invocation durations | 30 |
| HTTP (Class L) invocations | 30 |
| Pass measurements (per pass size) | 10 passes |
| Catch-up runs (per backlog N) | 5 runs |
| Scaling-axis points (per axis point) | the per-quantity minimum above |

Thirty is the floor that makes the §3.4 p95 honest. Where a quantity is
expensive at scale (the 500-event pass, the 10,000-line reconcile), the pass
minimum of 10 / catch-up minimum of 5 governs, and the record states N
plainly so the reviewer can weigh it.

Interleaving: where two variants are compared (verdict classes, ID
spellings, applied true/false), repetitions are interleaved A/B/A/B rather
than run in blocks, so slow host drift during the run is shared between
variants instead of being assigned to one of them.

### §11.4 Fixed inputs and seeds

- All fixture payloads are fixed literals or generated from a fixed seed
   recorded in the run record (proposed seed: a single integer in the
   harness defaults, overridable by flag, always recorded).
- Generated `outcomes.jsonl` fixtures derive from the same seed; line-
   class counts are parameters, stated in the record.
- `caused_by` hashes in fixtures are fixed strings of the merged tests'
   shape (`hash-…` fixture spellings), extended with a per-repetition
   counter so dedup behaves exactly as designed (§4.3's hazard).
- No wall-clock value is ever a fixture input: fixture *content* must not
   depend on when the run happens.

### §11.5 Contention honesty

This VM also runs the agent runtime and the dev loop itself (Q-4 noted the
same). Therefore:

- The harness records a contention note: what else the operator knows was
   running (at minimum: "dev-loop VM, shared with the agent runtime"), plus
   the run's start/end wall times so a reader can correlate.
- Runs are not repeated until a pretty number appears. One canonical run
   per configuration, reported whole. If a run is invalidated (rider
   failure, §10.1), the invalidation and its cause are recorded and the
   replacement run is the canonical one — the record keeps the invalidation
   note.
- A second confirmation run on a different occasion is recommended (not
   required) for the headline quantities, and §14.3 makes the label
   consequence explicit either way.

### §11.6 Reproducibility package

A reader with the two repos should be able to rerun SI-62's numbers from:
the harness at the recorded SI commit, the recorded command line and flags,
the recorded seed, and a host matching the fingerprint class. The results
record (§12.2) carries all four. Byte-identical timings are not promised —
timing on a contended VM is a distribution, and reproducibility here means
"same procedure, same fixture, comparable distribution", stated in those
words in the results doc.

---

## §12 Environment fingerprint and results format

### §12.1 Environment fingerprint (required fields)

Every run record opens with a fingerprint object (proposed field names):

- `si_commit` — full SHA of the SI tree measured.
- `gaia_commit` — full SHA of the gaia fork tree, or the string `"not-
   used"` for Class S/L runs; a Class F run must record the real SHA.
   Recording `"not-used"` is mandatory honesty, not an omission: it is what
   stops a loopback number acquiring a fork SHA by association.
- `python_version`, `python_executable`, `venv_path`.
- `os` (kernel + distro string), `cpu_model`, `cpu_count_os`
   (`os.cpu_count()`), `ram_bytes_total`.
- `journal_backend` — `"embedded"` for all S-7 runs (the fixture backend);
   any deviation invalidates comparability with §0's baselines-to-be and
   must be justified in the results doc. (NEDB-backend seam performance is
   a separate, unmeasured question — the Q-4 precedent labelled the same
   gap.)
- `host_note` — hostname + contention note (§11.5).
- `started_at` / `finished_at` — wall clock, with timezone; for record-
   keeping only, never a duration source (§2.2).
- `harness` — module name, command line, flags, seed.
- `transports` — per hop: class, double identity, double service-time
   summary (§10.3).

### §12.2 Machine-readable record

One JSON document per canonical run (proposed filename pattern:
`docs/S-7-measurements.json` for the canonical SI-62 run, siblings for any
confirmation run), containing: the fingerprint (§12.1); per quantity: the §3
fields (id like `Q-A5`, kind, evidence class, fixture id, unit) and the
summary statistics (median, min, max, p95 where §3.4 allows, N, exclusions
with causes); per-pass records for pass quantities; the correctness-rider
outcomes (§10.1 step 5); and the deferred-quantities list (§4.5, §5.5, §6.5,
§7.5 Class F, §8.5) copied in as explicit `UNMEASURED` entries — so the
record itself, not only the prose, refuses to imply coverage it does not
have. Q-4's compact single-line-JSON commit style is the precedent for
committing the record.

### §12.3 Human-readable results document

SI-62's results document (proposed: `docs/S-7-seam-measurements.md`, written
by SI-62, not by this design) mirrors this design's §0 table as its §0:
every row either gains a measured value with its conditions sentence (§3.6,
§15.3) or stays UNMEASURED with the reason (deferred class, double-bound,
insufficient N). It carries: the fingerprint in prose form; per-hop sections
following §4–§8; the §9.2 chain worksheet; the contention note; and a "what
would change these numbers" paragraph per hop (journal size, bundle scale, a
live fork, a quieter host) so no reader mistakes miniature-loopback numbers
for deployment numbers.

### §12.4 Evidence retention

The raw per-repetition readings are retained with the run's evidence (the
loop's backups-directory pattern); only the compact record is committed.
If a summary is ever questioned, the raw readings are the appeal court.

---

## §13 Execution plan for the implementation slice (SI-62 candidate)

Numbering note: "SI-62" is the candidate number this design uses for its
implementation slice; the dev loop assigns the real number when it spawns
the work.

### §13.1 Deliverables of SI-62

1. The harness (proposed `si/gaia/seam_measure.py`) implementing §10, with
   harness-side timing and residuals per §3.3's default.
2. Tests for the harness itself (proposed
   `tests/test_gaia_s7_seam_measure.py`): fixture shapes, rider assertions,
   record schema, the non-loopback refusal (§10.5), and small-N smoke runs
   of each hop plan — tests assert the harness *behaves*, never that
   timings meet a target (there are no targets, §16).
3. One canonical run at §11's sample sizes, producing the §12.2 record and
   §12.3 results document.
4. Label/doc updates strictly per §15, in the same PR, each tied to a
   quantity id in the record.
5. Evidence (raw readings, double logs, run logs) in the slice's backups
   directory, per loop practice.

### §13.2 Order of work inside SI-62

1. Harness skeleton + fingerprint + record writer.
2. Class S plans for all five hops (no doubles yet) — these alone retire
   the SI-side rows of §0.
3. Doubles + Class L plans, hop by hop in S-2→S-6 order, each with its
   §10.3 service-time check.
4. The §7.5 conditional outcome-latency plan and the §8.4/§8.5 lag-by-
   position reporting.
5. Canonical run, results document, label pass (§15).
6. (Only if §2.5 finds a live fork and D-8 approves) Class F env-gated
   runs, reported as their own section, never merged into Class L tables.

### §13.3 What SI-62 must not need

No new seam behaviour, no protocol change, no fork commit, no Mark decision
— unless a §17 decision is resolved in a way that asks for one. If SI-62
discovers it *does* need a seam change (for example, a phase boundary that
cannot be timed without instrumentation, D-1), it stops at the design's
default (residual reporting) rather than improvising the change inside a
measurement PR.

---

## §14 Acceptance criteria for SI-62

SI-62 is accepted — and its numbers may replace UNMEASURED labels per §15 —
only if all of §14.1–§14.3 hold. §14.4 states what is *not* required.

### §14.1 Coverage criteria

- Every quantity id in §4–§8 (Q-A1…Q-E6) is either reported from the
   canonical run at §11's minimum sample sizes, or appears in the results
   document as UNMEASURED with a named reason from this list: deferred fork
   half (§2.5), double-bound (§10.3), rider failure invalidating that
   quantity, or a §17 decision taken to defer it. Silent omission of a
   quantity is a review failure.
- The deferred lists (§4.5, §5.5, §6.5, §8.5, and Class F outcome latency)
   appear in both the record (§12.2) and the results doc as UNMEASURED
   entries.
- The chain worksheet (§9.2) is present, with UNMEASURED legs shown in
   place.

### §14.2 Integrity criteria

- All correctness riders (§10.1) passed in the canonical run: journal-
   growth assertions (Q-A5, Q-A8, Q-B7), seq-fidelity over measured forward
   passes (Q-E4/Q-E5), cursor assertions (Q-E4), dedup counts (Q-A8, Q-B7,
   Q-D1 second-poll no-op behaviour where the plan exercises it).
- Phase sum-checks (§3.3) reported; residuals named.
- Double service times reported beside every Class L result; any double-
   bound result downgraded in the text, not buried.
- Full test suite green on the reviewer's own run (the loop's standing
   review rule), including the new harness tests; no existing test changed
   or weakened; no existing seam behaviour changed.
- No secrets, no non-loopback URL, no run against a non-fixture bundle
   anywhere in the evidence.

### §14.3 Reporting criteria

- Every reported number carries its conditions sentence (§3.6) at first
   use in the results doc.
- Medians with ranges everywhere; no bare means; no cross-variant median
   claim where ranges fully overlap (§3.4).
- The contention note (§11.5) is present. If only one canonical run was
   made, the results doc says so and labels its numbers "single-run" in the
   conditions sentence; a confirmation run, if made, is reported beside it,
   not averaged into it.
- The fingerprint (§12.1) is complete; `gaia_commit` is a real SHA only
   for Class F content.

### §14.4 Explicitly not required

- No Class F (live fork) measurement. Its absence keeps the live-interop
   rows UNMEASURED and does not block acceptance (§2.5).
- No performance target, threshold, or improvement over any baseline.
   There is no baseline; S-7 *creates* the first numbers. A slow number
   honestly reported passes; a fast number without its conditions fails.
- No optimisation work. If a quantity looks bad, SI-62 reports it; fixing
   it is a future slice with its own design, and §15.4 forbids SI-62 from
   softening the wording because the number is unflattering.

---

## §15 Label replacement rules

### §15.1 The rule

An UNMEASURED label may be replaced only by the slice that measured the
quantity (SI-62), only for the quantities it actually reported under §14,
and only in the scoped form of §15.2/§15.3. This design (SI-61) replaces
nothing. A quantity measured in Class L replaces a label only with a Class L
wording; it never discharges a Class F row.

### §15.2 Replacement targets, per label

| Label / doc line | May gain (from SI-62) | Must keep UNMEASURED |
| --- | --- | --- |
| `SEAM_INGEST_LABEL` + S-2a doc "Performance" | Q-A5/Q-A6 scoped sentence | live fork egress latency (§4.5) |
| `WORK_SUBMISSION_LABEL` + S-3 doc | Q-B5 scoped sentence | live fork acceptance (§5.5) |
| `LEASH_CROSSING_LABEL` + S-4 doc | Q-C1/Q-C3 scoped sentence | stop→actuator-refusal latency; cadence adequacy (§6.4/§6.5) |
| `OUTCOME_RETURN_LABEL` + S-5 doc | Q-D1/Q-D4 scoped sentence; §7.5 conditional latency in conditional wording only | unconditional outcome latency |
| `EVENT_FORWARD_LABEL` + S-6 doc | Q-E4/Q-E5 scoped sentence; §8.4(b) in-pass lag | fork ingest + SSE visibility (§8.5 — fork half UNBUILT) |

### §15.3 The conditions sentence (template)

> "Measured under S-7 (SI-62): median {value} {unit} (range {min}–{max},
> N={n}) for {quantity} on the miniature fixture, embedded journal,
> {evidence class} transport{, double service median {d} ms}, host
> fingerprint in docs/S-7-seam-measurements.md; {single-run | confirmed by a
> second run}; live fork interop performance remains UNMEASURED."

Label constants carry a shortened form pointing at the results doc; docs
carry the full sentence. Any shortening that drops the evidence class or the
remaining-UNMEASURED clause is a labelling defect.

### §15.4 No retroactive softening, no forward claims

- Journaled historical label strings (for example S-3's `SUBMITTED_STATE`
   in already-journaled rows) are history and are never rewritten — the S-5
   doc's precedent on exactly this point stands.
- A measured miniature/loopback number never licenses wording about
   production, real-base, non-loopback, or live-fork performance, in any
   doc, PR body, or report derived from SI-62.

---

## §16 Explicit non-goals

1. **No implementation.** This slice is design only: no harness code, no
   tests, no measurement runs, and no numbers — including no "indicative"
   or "expected" numbers presented as findings.
2. **No live fork interop claims.** Nothing here asserts a live fork
   endpoint exists, was run, or performed in any way.
3. **No invented performance targets.** No SLO, budget, threshold, or
   "should be under X" figure appears as a requirement anywhere in this
   design. Where a constant exists in the merged system (heartbeat
   interval/TTL, poll/forward cadence as caller policy), S-7 measures costs
   around it and judges nothing.
4. **No seam protocol or behaviour changes.** No change to payload shapes,
   ACK shapes, journal kinds, cursor disciplines, dedup rules, or any
   merged seam function is designed or implied.
5. **No GPU, no cloud, no spend.** CPU only, this VM class, loopback only
   (§10.5).
6. **No new daemon or automation.** S-7 does not design a poller loop, a
   forwarder loop, or a heartbeat timer; cadence remains caller policy,
   measured conditionally (§7.5) where defined.
7. **No non-miniature scope.** No real-base bundle, no Glimmer bundle, no
   NEDB-backend journal measurement (the backend gap is named in §12.1, not
   closed).
8. **No Hearth-path measurement.** B-4's UNMEASURED Hearth quantities are
   untouched by this design.
9. **No weakening of any UNMEASURED label** — by this slice at all, and by
   SI-62 except under §15.
10. **No mind claims.** S-7 measures seam machinery; it says nothing about
   machine learning model quality or any trained proposer.

---

## §17 Risks and open questions — decisions SI-62 must make

Each item is a concrete decision, with this design's default stated. SI-62
records its resolution in its results document; unresolved items fall back
to the default, and the fallback is itself reported.

- **D-1 Instrumentation vs residuals.** *Decision:* may SI-62 add timing
   hooks inside seam modules, or must it time exposed boundaries +
   residuals (§3.3)? *Default:* harness-side only. *Risk if changed:* the
   measured code diverges from the shipped code, and every future seam PR
   must maintain the hooks.
- **D-2 Sample sizes vs run wall time.** *Decision:* adopt §11.3's
   minimums, or trim expensive axes (500-event passes, 10,000-line
   reconciles)? *Default:* adopt; trim only by reducing axis points, never
   below N=30 for headline durations. *Risk:* a multi-hour canonical run on
   a contended VM accumulates drift; mitigated by interleaving (§11.3), not
   by shrinking N silently.
- **D-3 Cadence scenarios for conditional quantities.** *Decision:* which
   polling schedule(s) §7.5 is evaluated at, and which harness cadence
   §8.4's journal-derived lag uses. *Default:* one schedule each (poll
   offset 0 s, period 1 s; back-to-back forward passes), labelled as
   harness parameters. *Risk:* readers mistake the scenario for a
   recommended deployment cadence — the conditions sentence (§15.3) is the
   mitigation.
- **D-4 Fixed-delay double.** *Decision:* is the §10.3 5 ms-delay
   sensitivity variant required or optional? *Default:* optional, run for
   Q-B3 and Q-E3 only. *Risk if skipped:* a zero-cost double could flatter
   round-trip shares attributed to SI.
- **D-5 Heartbeat cadence commentary.** *Decision:* may SI-62's results
   doc even *discuss* whether the 10 s / 30 s constants look plausible
   given Q-C4's heartbeat cost? *Default:* report Q-C4's cost only;
   adequacy is fork-side (§6.4). *Risk:* a cost number invites an adequacy
   conclusion the evidence cannot carry.
- **D-6 Concurrent-ingest observation.** *Decision:* include Q-A7's
   optional two-concurrent-POST contention observation, or leave
   concurrency wholly out? *Default:* out — the lock serialises by design,
   and a contention number invites a parallelism reading. *Risk if
   included:* misreading, per §4.2 Q-A7's caution.
- **D-7 Chain worksheet total row.** *Decision:* publish the §9.2
   captioned sum-of-medians, or the worksheet without a total? *Default:*
   no total. *Risk:* even captioned, a total is the number that gets
   quoted, captionless, elsewhere.
- **D-8 Class F attempt.** *Decision:* if a live fork appears reachable at
   SI-62 time, attempt env-gated Class F runs, or complete in Class L and
   leave Class F to a fork-side slice? *Default:* complete in Class L;
   Class F only on an explicit loop decision recorded in STATE.md, because
   fork-side work is main-chat directed in the loop's standing split. *Risk
   of attempting casually:* an unowned fork state (unknown commit, unknown
   load) makes the numbers unreproducible by §11.6's standard.
- **D-9 Results record location.** *Decision:* commit the §12.2 JSON
   beside the results doc (Q-4 precedent) or keep it in backups only?
   *Default:* commit it — Q-4's record is why its numbers are auditable.
   *Risk:* record churn in the repo if runs are repeated; mitigated by one
   canonical record + named siblings.
- **D-10 Journal-timestamp trust for lag quantities.** *Decision:* are the
   miniature journal's timestamps fine-grained and monotonic enough for
   §8.4's journal-derived lag, or is that quantity reported as "not
   resolvable at journal timestamp grain"? *Default:* measure the grain
   first (two adjacent rows written in one operation); if grain ≥ the lags
   of interest, report unresolvable — an honest non-number beats a precise-
   looking artefact.
- **D-11 Double-bound downgrade threshold.** *Decision:* what ratio of
   double service time to round-trip triggers §10.3's downgrade? *Default:*
   if the double's median service time exceeds 25% of the round-trip
   median, the quantity is reported as double-bound. *Risk:* the threshold
   is a judgement; it is recorded so the reviewer can override it knowingly
   rather than discover it later.
- **D-12 p95 publication.** *Decision:* publish p95 for all N≥30
   quantities (§3.4) or only headline ones? *Default:* all — selective
   tails look like selection. *Risk:* none material beyond record size.

---

## §18 Alternatives considered and rejected

- **Benchmark the fork from SI.** Rejected: the fork is another repository
   with its own loop and its own measurement questions; SI-side numbers
   about fork internals would be unreproducible from SI evidence and would
   blur the §2.5 ownership line. S-7 defines the Class F quantity list and
   stops there.
- **Synthetic in-memory journal rows as fixtures.** Rejected: hand-made
   rows can drift from what the seam code actually writes (the S-5 doc's
   untouched historical-strings episode shows how real journal content
   surprises). All fixtures are activity-grown (§10.2).
- **End-to-end single-number latency.** Rejected in §9.1.
- **Microbenchmarks of individual helpers in isolation (no journal, no
   bundle).** Rejected as the primary evidence: seam costs live in journal
   I/O, locking, and serialisation, which isolation removes. (Isolated
   timings may appear as supporting diagnostics if a phase residual is
   large; they cannot carry a label replacement.)

---

## §19 Glossary

- **Seam** — the SI ↔ gaia-fork boundary built by S-2..S-6 under G-9.
- **Class S / L / F** — evidence classes of §3.2: SI-side only, loopback
   double, live fork.
- **Double** — a loopback HTTP test double implementing a documented fork
   contract (§10.3). A double is never called "the fork" in S-7 text.
- **Conditions sentence** — the §15.3 template binding a number to its
   evidence class, fixture, and the UNMEASURED remainder.
- **Miniature bundle** — the tiny-seed bundle of the merged tests
   (`init_bundle` + `seed_tiny_bundle`), the only bundle class S-7
   measures.
- **UNMEASURED** — the project's honesty label: no number exists for this
   quantity under any stated conditions. It is a statement about evidence,
   not an apology, and this design uses it in that sense throughout.

---

## Appendix A Per-hop measurement checklists

### A.1 Hop A (S-2)

- [ ] Q-A1 validation, valid + refused fixtures
- [ ] Q-A2 dedup scan at 0 / 10 / 50 prior self-edits
- [ ] Q-A3 gate duration by verdict and tool class
- [ ] Q-A4 journal + re-anchor phase, residual named
- [ ] Q-A5 invocation by verdict class, growth asserted
- [ ] Q-A6 HTTP leg, client + server distributions
- [ ] Q-A7 serial throughput, window + N recorded
- [ ] Q-A8 dedup retry, zero growth, zero extra rows

### A.2 Hop B (S-3)

- [ ] Q-B1 eligibility + leash fold, incl. leash-held
- [ ] Q-B2 payload construction at scan sizes
- [ ] Q-B3 transport leg, 3 ID spellings, Class L + S
- [ ] Q-B4 journal submission + re-anchor
- [ ] Q-B5 invocation: success + 3 failure paths
- [ ] Q-B6 pending scan at 1 / 10 / 50 spendable
- [ ] Q-B7 dedup short-circuit, zero transport calls

### A.3 Hop C (S-4)

- [ ] Q-C1 leash fold in 3 journal states × sizes
- [ ] Q-C2 derivation scaling, records + tip, 1 vs 10 models
- [ ] Q-C3 per-record crossing + watermark no-op + gap pass
- [ ] Q-C4 heartbeat incl. confirmed journaling
- [ ] Q-C5 snapshot vs per-record contrast

### A.4 Hop D (S-5)

- [ ] Q-D1 poll pass at 1 / 10 / 50, 3 answer mixes
- [ ] Q-D2 per-work GET leg by answer class
- [ ] Q-D3 outcome journaling per outcome
- [ ] Q-D4 reconciler per 1,000 lines / KiB by line class
- [ ] Q-D5 reconcile scaling: file size × cursor position
- [ ] Q-D6 outcome-unknown marking + reproposal-block read
- [ ] §7.5 conditional latency at the D-3 schedule

### A.5 Hop E (S-6)

- [ ] Q-E1 payload construction across row kinds
- [ ] Q-E2 journal read windows 1 / 10 / 100 / 500, 2 cursor positions
- [ ] Q-E3 per-event forward, Class L
- [ ] Q-E4 pass throughput at 10 / 100 / 500 + interrupted pass
- [ ] Q-E5 catch-up at backlog 100 / 500, passes-to-tip + wall
- [ ] Q-E6 duplicate-push protocol failure behaviour
- [ ] §8.4(b) in-pass lag by position decile

---

## Appendix B Results document skeleton (for SI-62)

```text
# S-7 — Gaia seam measurements (SI-62)

§0 Honesty table — §0 of the S-7 design, each row now:
   measured (quantity id, conditions sentence) or
   UNMEASURED (reason: deferred / double-bound / D-n).
§1 Fingerprint — §12.1 of the design, in prose + the
   committed record's filename.
§2 Method deviations — every deviation from the design,
   or "none". (An unlisted deviation a reviewer finds in
   the record invalidates the affected quantities.)
§3–§7 Per-hop results — design §4–§8 order; each
   quantity: table of median / min / max / p95 / N,
   then prose that obeys §3.4's overlap rule.
§8 Chain worksheet — design §9.2, no total row
   unless D-7 was resolved otherwise, with caption.
§9 Decisions — D-1…D-12 as resolved, one line each.
§10 Label changes made — per §15.2, quoting old and
   new wording, or "none" per label.
```

---

## Document tail and verification

This document is the complete SI-61 design for S-7. It contains: the §0
honesty table for all five seam hops; per-hop quantities with units, clocks,
and evidence classes; the harness, fixture, sampling, and reproducibility
design; the environment fingerprint and results format; SI-62's execution
plan and acceptance criteria; the label replacement rules; the explicit non-
goals; and twelve numbered decisions (D-1…D-12) the implementation slice
must resolve or default.

Nothing in this document is a measurement. No seam quantity's status changes
because this document exists: every quantity listed as UNMEASURED in §0
remains UNMEASURED until SI-62 measures it under this design and a reviewer
accepts the evidence.

— End of S-7 seam measurement design (SI-61). —
