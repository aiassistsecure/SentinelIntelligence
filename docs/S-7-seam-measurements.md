# S-7 — Gaia seam measurements (SI-62)

**Status:** canonical run VALID — all correctness riders passed (391 riders, 0 failed), single-run, CPU-only on the dev-loop VM. Harness: `si/gaia/seam_measure.py` (`python -m si.gaia.seam_measure --out <dir>`); machine-readable record: `docs/S-7-measurements.json`. Design: `docs/S-7-seam-measurement-design.md` (SI-61).

## §0 Honesty table

| Hop | Measured (this run) | Still UNMEASURED |
| --- | --- | --- |
| A — S-2 proposal → verdict | Measured under S-7 (SI-62): median 192 ms (range 179–246, N=30) for Q-A5 ingest invocation (ESCALATE) on the miniature fixture, embedded journal, Class S transport, host fingerprint in docs/S-7-seam-measurements.md; single-run; live fork interop performance remains UNMEASURED. Also: Measured under S-7 (SI-62): median 187 ms (range 173–321, N=30) for Q-A6 HTTP ingest on the miniature fixture, embedded journal, Class L transport, host fingerprint in docs/S-7-seam-measurements.md; single-run; live fork interop performance remains UNMEASURED. | live fork egress → SI ingest latency (Class F, §4.5) |
| B — S-3 work submission | Measured under S-7 (SI-62): median 120 ms (range 108–164, N=30) for Q-B5 work submission invocation (success, loopback) on the miniature fixture, embedded journal, Class L transport, double service median 0.3 ms, host fingerprint in docs/S-7-seam-measurements.md; single-run; live fork interop performance remains UNMEASURED. | live fork acceptance + fork-side queueing (Class F, §5.5) |
| C — S-4 leash | Measured under S-7 (SI-62): median 124 ms (range 103–156, N=30) for Q-C3 leash per-record crossing on the miniature fixture, embedded journal, Class L transport, double service median 0.4 ms, host fingerprint in docs/S-7-seam-measurements.md; single-run; live fork interop performance remains UNMEASURED. Also: Measured under S-7 (SI-62): median 0.1 ms (range 0.1–0.4, N=30) for Q-C1 leash fold check (stop in force) on the miniature fixture, embedded journal, Class S transport, host fingerprint in docs/S-7-seam-measurements.md; single-run; live fork interop performance remains UNMEASURED. | stop → fork actuator refusal latency; heartbeat cadence adequacy (§6.4/§6.5) |
| D — S-5 outcome return | Measured under S-7 (SI-62): median 1139 ms (range 1087–1648, N=10) for Q-D1 outcome poll pass (10 outstanding, all terminal) on the miniature fixture, embedded journal, Class L transport, host fingerprint in docs/S-7-seam-measurements.md; single-run; live fork interop performance remains UNMEASURED. Conditional form only: Measured under S-7 (SI-62): median 0.76 s (range 0.75–0.77, N=5) for conditional outcome latency at the stated schedule (offset 0 s, period 1 s, terminal at +0.35 s) on the miniature fixture, embedded journal, Class L transport, host fingerprint in docs/S-7-seam-measurements.md; single-run; unconditional outcome latency and live fork interop remain UNMEASURED. | unconditional outcome latency; Class F outcome latency (§7.5) |
| E — S-6 event forwarding | Measured under S-7 (SI-62): median 152 ms (range 138–158, N=10) for Q-E4 event forward pass (100 events) on the miniature fixture, embedded journal, Class L transport, double service median 0.3 ms, host fingerprint in docs/S-7-seam-measurements.md; single-run; live fork interop performance remains UNMEASURED. Also: Measured under S-7 (SI-62): median 0.55 s (range 0.50–0.58, N=5) for Q-E5 event catch-up (backlog 500, single unlimited pass) on the miniature fixture, embedded journal, Class L transport, host fingerprint in docs/S-7-seam-measurements.md; single-run; fork ingest (UNBUILT) and SSE visibility remain UNMEASURED. | fork ingest service time and ingest → SSE visibility — fork half UNBUILT (§8.5) |

Every other quantity in §3–§7 below is reported from the canonical run at the design's §11 sample sizes (N=30 headline durations, 10 passes, 5 catch-up runs), or listed in the deferred table at the end of this section. Fork-dependent quantities were not estimated.

| Deferred quantity | Status | Reason |
| --- | --- | --- |
| fork egress send -> SI ACK received (Class F) | UNMEASURED | deferred fork half (design §4.5); no live fork run (D-8) |
| live fork work-acceptance 202 latency + fork queueing (Class F) | UNMEASURED | deferred fork half (design §5.5); no live fork run (D-8) |
| stop journaled -> fork actuator refuses; fork staleness timing | UNMEASURED | deferred fork half (design §6.5); SI-side components measured (Q-C1/Q-C3) |
| heartbeat interval/TTL (10 s / 30 s) adequacy | UNMEASURED | fork-side policy constants (design §6.4, D-5): cost measured (Q-C4), adequacy not judged |
| unconditional outcome latency; Class F outcome latency | UNMEASURED | undefined without a poll cadence (design §2.4/§7.5); only the conditional form is measured |
| fork ingest service time per event (POST /gaia/v1/events/ingest) | UNMEASURED | fork half UNBUILT (design §2.5/§8.5) |
| fork ingest -> source=si SSE visibility lag; merged-stream timing | UNMEASURED | fork half UNBUILT (design §8.5) |
| Q-A7 concurrent-ingest contention observation | UNMEASURED | deferred by D-6 default (out): the lock serialises by design |

## §1 Fingerprint

- SI commit measured: `fb1aa47da115723a734dcffce113b7f5e8227d86` (seam code at main b7e312a plus this slice's harness)
- gaia fork commit: `not-used` (mandatory for Class S/L runs, §12.1 — no fork SHA by association)
- Python 3.12.3 at `/home/hatch/workspace/si2-venv/bin/python` (venv `/home/hatch/workspace/si2-venv`)
- OS: Linux 7.0.0-39-generic (Linux-7.0.0-39-generic-x86_64-with-glibc2.39); CPU: AMD EPYC 9D25 126-Core Processor (4 CPUs per os.cpu_count()); RAM total 8319901696 bytes
- Journal backend: embedded. Host: htch-runtime; dev-loop VM, shared with the agent runtime (contention note, §11.5)
- Contention: this canonical run shared the VM with the agent runtime and other project loops (§11.5); medians are reported as measured under that contention, not re-run for prettier numbers.
- Run window (wall, record-keeping only): 2026-10-31T07:28:40.457974+00:00 → 2026-10-31T07:56:18.701808+00:00; harness wall 1658 s; seed 20261031
- Transports: A: Class S in-process + Class L real B-5 service handler on loopback (Q-A6); B: Class S ReplayTransport + Class L loopback work double via HttpTransport; C: Class S journal reads + Class L loopback leash double; D: Class S file/replay + Class L loopback status double; E: Class S journal reads + Class L loopback ingest double (200/409 contract; fork ingest UNBUILT)

Reproducibility (§11.6): same procedure, same fixtures, comparable distribution — byte-identical timings are not promised on a contended VM.

## §2 Method deviations

- D-1 defaults applied: phases inside one merged function are named residuals (Q-A4, Q-B4, and the Q-C3 journaling share as a difference of distributions), not instrumentation added to seam code.
- Each Q-A5/Q-B5 repetition set runs on one fixture bundle, so its journal grows across the 30 repetitions; the Q-A5 size axis (pre-0/10/50) names starting sizes explicitly.
- Q-D4 unit costs per 1,000 lines are derived arithmetic over measured passes at the stated line counts (§7.2), labelled derived in the record.
- One canonical run only: every number below is single-run, and its conditions sentence says so (§14.3). No run was repeated until a prettier number appeared (§11.5); exclusions: none.
- Two earlier attempts at this canonical run were invalidated by infrastructure, not by the harness: a VM reboot killed the first attempt mid-run (Hop D) and wiped the /tmp worktree the second attempt was launched from. Neither produced a record; the run reported here is the first completed canonical run, executed from persistent workspace storage.
- The Q-C3 transport-leg-only quantity is flagged double-bound under D-11 (double median 0.27 ms against a 0.93 ms bare leg — the double is a large share of a bare transport leg by construction); all other Class L headline quantities are not double-bound.
- No other deviation from the design.

## §3 Hop A — S-2 proposal ingest (design §4)

Validation is ~free; the dedup scan and the ingest invocation grow with journaled self-edit count (Q-A2, Q-A5 size axis) — journaling and re-anchoring, the residual Q-A4, dominate the invocation, which is the cost of the journal-first guarantee (§2.3), reported as such. Verdict classes are indistinguishable at this scale where their ranges overlap (§3.4). What would change these numbers: a larger journal (the size axis shows the direction), a non-miniature bundle, a quieter host.

### Q-A1

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid | S | 0.0 ms | 0.0 | 0.0 | 0.0 | 30 | — | — |
| refused | S | 0.0 ms | 0.0 | 0.0 | 0.0 | 30 | — | — |


### Q-A2

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prior-edits-0 | S | 0.0 ms | 0.0 | 1.1 | 0.1 | 30 | — | — |
| prior-edits-10 | S | 0.3 ms | 0.3 | 1.3 | 0.9 | 30 | — | — |
| prior-edits-50 | S | 1.4 ms | 1.4 | 1.5 | 1.5 | 30 | — | — |


### Q-A3

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADMIT-fs_list | S | 0.1 ms | 0.1 | 0.1 | 0.1 | 30 | — | — |
| ESCALATE-hearth.restart | S | 0.1 ms | 0.1 | 0.1 | 0.1 | 30 | — | — |
| REJECT-hearth.reap | S | 0.1 ms | 0.1 | 0.1 | 0.1 | 30 | — | — |
| ADMIT-hearth.restart | S | 0.1 ms | 0.1 | 0.1 | 0.1 | 30 | — | — |


### Q-A4

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| journal+reanchor-residual | S | 191 ms | 191 | 191 | — | 1 | — | — |

_journal+reanchor-residual: derived residual: median Q-A5(ESCALATE) minus the medians of Q-A1(valid), Q-A2(0 prior) and Q-A3(ESCALATE) measured at exposed boundaries on separate fixtures (D-1 harness-side timing); a single derived value, not a distribution_

### Q-A5

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ESCALATE | S | 192 ms | 179 | 246 | 243 | 30 | — | — |
| ADMIT-external | S | 173 ms | 157 | 222 | 220 | 30 | — | — |
| ADMIT-fs_list | S | 171 ms | 159 | 227 | 199 | 30 | — | — |
| REJECT | S | 193 ms | 182 | 258 | 223 | 30 | — | — |
| ESCALATE-pre10 | S | 199 ms | 180 | 231 | 230 | 30 | — | — |
| ESCALATE-pre50 | S | 217 ms | 195 | 300 | 266 | 30 | — | — |


### Q-A6

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| POST /gaia/proposals | L | 187 ms | 173 | 321 | 261 | 30 | — | — |

_POST /gaia/proposals: double is the real B-5 service handler itself; its in-process cost distribution is Q-A5 (ESCALATE) — the HTTP+JSON share is the difference of those distributions (§4.2)_

### Q-A7

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| window-30 | S | 5.3 items/s | 4.96 | 5.46 | — | 5 | — | — |


### Q-A8

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| retry | S | 5.4 ms | 3.7 | 19.6 | 8.5 | 30 | — | — |


## §4 Hop B — S-3 work submission (design §5)

The work-ID spelling does not change the transport leg beyond noise (ranges overlap, §3.4). Failure paths cost roughly the eligibility + scan work and journal nothing — the riders assert the empty gaia_work set after every failure repetition. The 5 ms-delay sensitivity variant (D-4) shows the SI-side share is not an artefact of a zero-cost peer. What would change these numbers: pending-set size (Q-B6), a live fork's real acceptance cost (UNMEASURED, §5.5).

### Q-B1

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eligible | S | 0.1 ms | 0.1 | 0.1 | 0.1 | 30 | — | — |
| leash-held | S | 0.1 ms | 0.1 | 0.3 | 0.2 | 30 | — | — |


### Q-B2

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| payload | S | 0.0 ms | 0.0 | 0.0 | 0.0 | 30 | — | — |


### Q-B3

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| spelling-work_id | L | 0.8 ms | 0.6 | 1.1 | 1.1 | 30 | 0.2 | yes |
| spelling-id | L | 0.8 ms | 0.6 | 1.0 | 1.0 | 30 | 0.2 | yes |
| spelling-workId | L | 0.8 ms | 0.6 | 1.1 | 1.0 | 30 | 0.2 | yes |
| replay | S | 0.0 ms | 0.0 | 0.0 | 0.0 | 30 | — | — |
| delay-5ms-sensitivity | L | 6.1 ms | 5.9 | 6.4 | 6.4 | 30 | 5.3 | yes |

_delay-5ms-sensitivity: D-4 sensitivity variant: double adds a fixed 5 ms service delay_

### Q-B4

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| journal-submission-residual | S | 118 ms | 118 | 118 | — | 1 | — | — |

_journal-submission-residual: derived residual: median Q-B5(replay) minus medians of Q-B1(eligible), Q-B2 and Q-B3(replay) (D-1)_

### Q-B5

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| success-loopback | L | 120 ms | 108 | 164 | 156 | 30 | 0.3 | no |
| success-replay | S | 118 ms | 109 | 177 | 167 | 30 | — | — |
| non-202 | S | 4.0 ms | 3.9 | 4.2 | 4.2 | 30 | — | — |
| 202-no-work-id | S | 4.0 ms | 3.9 | 4.2 | 4.1 | 30 | — | — |
| transport-exception | S | 4.0 ms | 3.9 | 5.0 | 4.7 | 30 | — | — |


### Q-B6

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pass-1-spendable | S | 106 ms | 98.8 | 114 | — | 10 | — | — |
| pass-1-spendable | S | 9.4 items/s | 8.75 | 10.1 | — | 10 | — | — |
| pass-10-spendable | S | 1095 ms | 895 | 1761 | — | 10 | — | — |
| pass-10-spendable | S | 9.14 items/s | 5.68 | 11.2 | — | 10 | — | — |
| pass-50-spendable | S | 6971 ms | 6630 | 8391 | — | 10 | — | — |
| pass-50-spendable | S | 7.17 items/s | 5.96 | 7.54 | — | 10 | — | — |


### Q-B7

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dedup | S | 26.0 ms | 25.3 | 36.6 | 29.6 | 30 | — | — |


## §5 Hop C — S-4 leash (design §6)

The per-submission leash fold (Q-C1) is sub-millisecond in all three journal states — the safety tax is small at miniature scale, reported even though it is near noise (§6.2). leash_records derivation grows with leash-entry count (Q-C2); crossings are dominated by confirmed journaling, not the transport leg (Q-C3 distributions). No cadence adequacy is judged (D-5). What would change these numbers: journal size, a live fork's watermark application (UNMEASURED, §6.5).

### Q-C1

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no-leash-record | S | 0.1 ms | 0.1 | 0.1 | 0.1 | 30 | — | — |
| stop-in-force | S | 0.1 ms | 0.1 | 0.4 | 0.2 | 30 | — | — |
| stop-then-resume | S | 0.1 ms | 0.1 | 0.6 | 0.2 | 30 | — | — |


### Q-C2

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leash_records | S | 0.0 ms | 0.0 | 0.0 | 0.0 | 30 | — | — |
| leash_tip | S | 0.0 ms | 0.0 | 0.0 | 0.0 | 30 | — | — |
| leash_records | S | 0.1 ms | 0.1 | 0.1 | 0.1 | 30 | — | — |
| leash_tip | S | 0.1 ms | 0.1 | 0.1 | 0.1 | 30 | — | — |
| leash_records | S | 0.6 ms | 0.6 | 1.1 | 1.0 | 30 | — | — |
| leash_tip | S | 0.6 ms | 0.6 | 1.1 | 0.8 | 30 | — | — |
| seam_tasks-scan | S | 0.0 ms | 0.0 | 0.2 | 0.1 | 30 | — | — |
| seam_tasks-scan | S | 0.3 ms | 0.3 | 10.1 | 0.7 | 30 | — | — |


### Q-C3

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| per-record-applied-true | L | 124 ms | 103 | 156 | 155 | 30 | 0.4 | no |
| transport-leg-only | L | 0.9 ms | 0.7 | 1.9 | 1.5 | 30 | 0.3 | yes |
| watermark-applied-false | L | 125 ms | 104 | 171 | 165 | 30 | 0.2 | no |
| pass-10-records | L | 1055 ms | 1001 | 1125 | — | 10 | 0.4 | no |
| pass-10-records | L | 9.48 items/s | 8.89 | 9.99 | — | 10 | 0.4 | no |
| gap-pass-first-record-500 | S | 4.3 ms | 4.3 | 4.8 | — | 10 | — | — |

_transport-leg-only: journaling share of per-record crossing is the difference of the per-record and transport-leg distributions (§3.3)_

### Q-C4

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heartbeat | L | 123 ms | 103 | 186 | 161 | 30 | 0.4 | no |


### Q-C5

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| snapshot | L | 146 ms | 119 | 950 | 390 | 30 | 0.4 | no |


## §6 Hop D — S-5 outcome poll and reconcile (design §7)

Only terminal answers pay journaling, so the mixes differ by design (§7.2). Fault journaling in the reconciler (disagreement / distinct-second-outcome classes of Q-D4) is the expensive line class — each fault is a real journal write with re-anchor. The conditional outcome latency (Q-D-COND) is reported only in its conditional form at the D-3 schedule; unconditional outcome latency remains UNMEASURED. What would change these numbers: outstanding-set size, outcomes-file size and cursor position (Q-D5), the poll schedule itself.

### Q-D1

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pass-1-terminal | L | 105 ms | 100 | 168 | — | 10 | — | — |
| pass-1-terminal | L | 9.52 items/s | 5.95 | 9.98 | — | 10 | — | — |
| pass-1-awaiting | L | 6.4 ms | 6.1 | 8.3 | — | 10 | — | — |
| pass-1-awaiting | L | 157 items/s | 120 | 164 | — | 10 | — | — |
| pass-1-mixed | L | 104 ms | 95.5 | 121 | — | 10 | — | — |
| pass-1-mixed | L | 9.65 items/s | 8.27 | 10.5 | — | 10 | — | — |
| pass-10-terminal | L | 1139 ms | 1087 | 1648 | — | 10 | — | — |
| pass-10-terminal | L | 8.78 items/s | 6.07 | 9.2 | — | 10 | — | — |
| pass-10-awaiting | L | 21.7 ms | 19.5 | 25.0 | — | 10 | — | — |
| pass-10-awaiting | L | 460 items/s | 399 | 512 | — | 10 | — | — |
| pass-10-mixed | L | 567 ms | 548 | 662 | — | 10 | — | — |
| pass-10-mixed | L | 17.6 items/s | 15.1 | 18.2 | — | 10 | — | — |
| pass-50-terminal | L | 7427 ms | 7242 | 7880 | — | 10 | — | — |
| pass-50-terminal | L | 6.73 items/s | 6.35 | 6.9 | — | 10 | — | — |
| pass-50-awaiting | L | 81.0 ms | 74.1 | 156 | — | 10 | — | — |
| pass-50-awaiting | L | 617 items/s | 321 | 675 | — | 10 | — | — |
| pass-50-mixed | L | 3589 ms | 3421 | 3720 | — | 10 | — | — |
| pass-50-mixed | L | 13.9 items/s | 13.4 | 14.6 | — | 10 | — | — |


### Q-D2

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| terminal | L | 0.9 ms | 0.7 | 2.0 | 1.9 | 30 | 0.2 | no |
| awaiting | L | 1.0 ms | 0.8 | 2.0 | 1.6 | 30 | 0.1 | no |
| notfound | L | 1.0 ms | 0.7 | 1.3 | 1.2 | 30 | 0.1 | no |


### Q-D3

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| journal-outcome-replay | S | 98.6 ms | 90.9 | 303 | 138 | 30 | — | — |

_journal-outcome-replay: poll of exactly one outstanding submission over the replay floor: scan + journaling + re-anchor, with the transport leg ~zero by construction (§3.2)_

### Q-D4

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unmatched-per-1000-lines | S | 10.4 ms | 10.1 | 11.6 | — | 10 | — | — |
| agreement-per-20-lines | S | 24.0 ms | 22.9 | 37.7 | — | 10 | — | — |
| disagreement-per-20-lines | S | 2410 ms | 2301 | 3218 | — | 10 | — | — |
| distinct-second-outcome-per-40-lines | S | 4676 ms | 4524 | 5039 | — | 10 | — | — |
| torn-partial-pass | S | 4.8 ms | 4.8 | 4.8 | — | 1 | — | — |

_unmatched-per-1000-lines: file bytes 403890; per-KiB derived in the results document_
_agreement-per-20-lines: per-1,000-line unit cost is derived arithmetic (pass median / lines x 1000), reported in the results document (§7.2)_
_disagreement-per-20-lines: per-1,000-line unit cost is derived arithmetic (pass median / lines x 1000), reported in the results document (§7.2)_
_distinct-second-outcome-per-40-lines: per-1,000-line unit cost is derived arithmetic (pass median / lines x 1000), reported in the results document (§7.2)_

### Q-D5

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| from-0-100-lines | S | 4.7 ms | 4.5 | 44.4 | — | 10 | — | — |
| from-0-1000-lines | S | 10.4 ms | 10.3 | 10.6 | — | 10 | — | — |
| from-0-10000-lines | S | 67.0 ms | 66.3 | 70.9 | — | 10 | — | — |
| near-eof-1-line | S | 4.0 ms | 4.0 | 7.8 | 6.1 | 30 | — | — |


### Q-D6

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mark-outcome-unknown | S | 116 ms | 103 | 136 | 133 | 30 | — | — |
| reproposal-block-read | S | 13.2 ms | 5.0 | 35.9 | 32.8 | 30 | — | — |


### Q-D-COND

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conditional-offset0-period1s | L | 0.76 s | 0.75 | 0.77 | — | 5 | — | — |

_conditional-offset0-period1s: conditional outcome latency (§7.5): outcome journaled time minus the double's terminal time, both on this host's monotonic clock, under the stated harness schedule (first poll at offset 0 s, then every 1 s; terminal at +0.35 s). Not an unconditional outcome latency._

## §7 Hop E — S-6 event forwarding (design §8)

One POST per event is the shape of this hop: per-event forward cost (Q-E3) is dominated by the journal read the forwarder performs, and catch-up (Q-E5) prices draining a backlog with that shape — reported even where unflattering (§8.2). In-pass lag grows linearly by position decile (Q-E-LAG), as a serial pass must. The journal-derived lag (Q-E-LAGJ) is dominated by queueing under the harness cadence (growth first, one pass after) and is not a system property. Fork ingest and SSE visibility remain UNMEASURED — the fork half is UNBUILT (§8.5).

### Q-E0

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| journal-ts-adjacent-delta (D-10 grain) | S | 40.7 ms | 17.4 | 151 | 96.9 | 554 | — | — |

_journal-ts-adjacent-delta (D-10 grain): timestamp grain measured first per D-10: the median adjacent-row delta bounds what the journal-derived lag (Q-E-LAGJ) can resolve_

### Q-E1

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| build_stream_event-per-event | S | 0.0 ms | 0.0 | 0.0 | 0.0 | 30 | — | — |


### Q-E2

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| window-1-@start | S | 30.4 ms | 28.2 | 46.8 | 46.0 | 30 | — | — |
| window-1-@near-tip | S | 13.6 ms | 13.1 | 24.5 | 18.3 | 30 | — | — |
| window-10-@start | S | 29.9 ms | 28.0 | 46.0 | 42.4 | 30 | — | — |
| window-10-@near-tip | S | 13.6 ms | 13.2 | 24.8 | 24.8 | 30 | — | — |
| window-100-@start | S | 30.3 ms | 28.7 | 42.6 | 42.5 | 30 | — | — |
| window-100-@near-tip | S | 13.6 ms | 13.2 | 24.9 | 14.8 | 30 | — | — |
| window-500-@start | S | 30.9 ms | 30.1 | 44.3 | 44.2 | 30 | — | — |
| window-500-@near-tip | S | 13.6 ms | 13.1 | 26.6 | 15.2 | 30 | — | — |


### Q-E3

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| per-event | L | 48.1 ms | 46.4 | 114 | 61.8 | 30 | 0.4 | no |
| per-event-delay5ms | L | 55.7 ms | 50.8 | 75.0 | 68.6 | 30 | 5.3 | no |

_per-event-delay5ms: D-4 sensitivity variant_

### Q-E4

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pass-10 | L | 60.4 ms | 57.2 | 70.6 | — | 10 | 0.3 | no |
| pass-10 | L | 166 items/s | 142 | 175 | — | 10 | 0.3 | no |
| pass-100 | L | 152 ms | 138 | 158 | — | 10 | 0.3 | no |
| pass-100 | L | 656 items/s | 634 | 724 | — | 10 | 0.3 | no |
| pass-500 | L | 532 ms | 468 | 596 | — | 10 | 0.3 | no |
| pass-500 | L | 940 items/s | 839 | 1.07e+03 | — | 10 | 0.3 | no |
| interrupted-pass-100-at-50 | L | 98.4 ms | 94.4 | 117 | — | 10 | — | — |


### Q-E5

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| catchup-100-repeated-limit50 | L | 0.23 s | 0.19 | 0.41 | — | 5 | — | — |
| catchup-100-repeated-limit50 | L | 433 items/s | 245 | 522 | — | 5 | — | — |
| catchup-100-repeated-limit50-passes-to-tip | L | 2 passes | 2 | 2 | — | 5 | — | — |
| catchup-100-single-unlimited | L | 0.15 s | 0.15 | 0.17 | — | 5 | — | — |
| catchup-100-single-unlimited | L | 653 items/s | 606 | 666 | — | 5 | — | — |
| catchup-100-single-unlimited-passes-to-tip | L | 1 passes | 1 | 1 | — | 5 | — | — |
| catchup-500-repeated-limit50 | L | 0.92 s | 0.91 | 1.19 | — | 5 | — | — |
| catchup-500-repeated-limit50 | L | 542 items/s | 419 | 548 | — | 5 | — | — |
| catchup-500-repeated-limit50-passes-to-tip | L | 10 passes | 10 | 10 | — | 5 | — | — |
| catchup-500-single-unlimited | L | 0.55 s | 0.50 | 0.58 | — | 5 | — | — |
| catchup-500-single-unlimited | L | 914 items/s | 858 | 992 | — | 5 | — | — |
| catchup-500-single-unlimited-passes-to-tip | L | 1 passes | 1 | 1 | — | 5 | — | — |


### Q-E6

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| duplicate-push-time-to-raise | L | 50.8 ms | 47.4 | 90.8 | 88.2 | 30 | 0.3 | no |

_duplicate-push-time-to-raise: quantity of record is correctness (cursor/pushed/call count, pinned by the rider); duration is supporting data (§8.2)_

### Q-E-LAG

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| in-pass-lag-decile-1 | L | 87.7 ms | 49.5 | 130 | 123 | 150 | — | — |
| in-pass-lag-decile-2 | L | 140 ms | 102 | 189 | 182 | 150 | — | — |
| in-pass-lag-decile-3 | L | 195 ms | 151 | 247 | 239 | 150 | — | — |
| in-pass-lag-decile-4 | L | 253 ms | 199 | 303 | 295 | 150 | — | — |
| in-pass-lag-decile-5 | L | 312 ms | 250 | 355 | 352 | 150 | — | — |
| in-pass-lag-decile-6 | L | 369 ms | 297 | 411 | 405 | 150 | — | — |
| in-pass-lag-decile-7 | L | 424 ms | 337 | 466 | 460 | 150 | — | — |
| in-pass-lag-decile-8 | L | 477 ms | 388 | 530 | 522 | 150 | — | — |
| in-pass-lag-decile-9 | L | 535 ms | 432 | 585 | 578 | 150 | — | — |
| in-pass-lag-decile-10 | L | 587 ms | 481 | 637 | 629 | 150 | — | — |

_in-pass-lag-decile-1: §8.4(b): pass start -> that event's 200, by position decile in a serial one-POST-per-event pass_
_in-pass-lag-decile-2: §8.4(b): pass start -> that event's 200, by position decile in a serial one-POST-per-event pass_
_in-pass-lag-decile-3: §8.4(b): pass start -> that event's 200, by position decile in a serial one-POST-per-event pass_
_in-pass-lag-decile-4: §8.4(b): pass start -> that event's 200, by position decile in a serial one-POST-per-event pass_
_in-pass-lag-decile-5: §8.4(b): pass start -> that event's 200, by position decile in a serial one-POST-per-event pass_
_in-pass-lag-decile-6: §8.4(b): pass start -> that event's 200, by position decile in a serial one-POST-per-event pass_
_in-pass-lag-decile-7: §8.4(b): pass start -> that event's 200, by position decile in a serial one-POST-per-event pass_
_in-pass-lag-decile-8: §8.4(b): pass start -> that event's 200, by position decile in a serial one-POST-per-event pass_
_in-pass-lag-decile-9: §8.4(b): pass start -> that event's 200, by position decile in a serial one-POST-per-event pass_
_in-pass-lag-decile-10: §8.4(b): pass start -> that event's 200, by position decile in a serial one-POST-per-event pass_

### Q-E-LAGJ

| Variant | Class | Median | Min | Max | p95 | N | Double median | Double-bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| journal-derived-queueing-plus-inpass | L | 87182 ms | 85386 | 88941 | 88716 | 100 | — | — |

_journal-derived-queueing-plus-inpass: §8.4 journal-derived lag under the harness cadence (one pass driven immediately after journal growth): queueing lag is a property of that cadence, not of the system; never merged with the monotonic Q-E-LAG statistics_

## §8 Chain worksheet (design §9.2 — no total row, D-7)

| Leg | Source quantity | Class |
| --- | --- | --- |
| Proposal validation → ACK | Q-A5 / Q-A6 | S / L |
| ACK → submission invoked | operator policy — not a system quantity | — |
| Submission round-trip | Q-B5 | L |
| Fork acceptance → fork terminal | fork half | UNMEASURED (Class F list) |
| Terminal → poll discovers it | conditional, §7.5 schedule (Q-D-COND) | L (conditional) |
| Poll journaling | Q-D3 | S |
| Journal event → forwarded 200 | Q-E3 / Q-E-LAG | L |
| Forwarded → visible on fork SSE | fork half (UNBUILT ingest) | UNMEASURED |

## §9 Decisions (design §17)

- **D-1:** harness-side timing only; internal phases as named residuals (Q-A4, Q-B4, Q-C3)
- **D-10:** journal timestamp grain measured first (Q-E0); journal-derived lag reported at that grain (Q-E-LAGJ)
- **D-11:** double-bound threshold 25% of round-trip median, computed per Class L quantity
- **D-12:** p95 published for all N>=30 quantities
- **D-2:** §11.3 minimums adopted (30 / 10 passes / 5 catch-up runs)
- **D-3:** one schedule each: poll offset 0 s, period 1 s; forward passes back-to-back
- **D-4:** 5 ms-delay sensitivity variant run for Q-B3 and Q-E3
- **D-5:** Q-C4 cost reported only; cadence adequacy not judged
- **D-6:** concurrent-ingest observation OUT (deferred F-A-CONC)
- **D-7:** chain worksheet published without a total row
- **D-8:** Class L only; no Class F attempt
- **D-9:** record committed beside the results document (docs/S-7-measurements.json)

## §10 Label changes made (design §15.2)

Each label constant keeps its full prior text; the scoped measured sentence is appended and the live-fork remainder stays UNMEASURED. Journaled historical label strings are history and were not rewritten (§15.4).

- `SEAM_INGEST_LABEL` gains: "Measured under S-7 (SI-62): median 192 ms (range 179–246, N=30) for Q-A5 ingest invocation (ESCALATE) on the miniature fixture, embedded journal, Class S transport, host fingerprint in docs/S-7-seam-measurements.md; single-run; live fork interop performance remains UNMEASURED."
- `SEAM_INGEST_LABEL_A6` gains: "Measured under S-7 (SI-62): median 187 ms (range 173–321, N=30) for Q-A6 HTTP ingest on the miniature fixture, embedded journal, Class L transport, host fingerprint in docs/S-7-seam-measurements.md; single-run; live fork interop performance remains UNMEASURED."
- `WORK_SUBMISSION_LABEL` gains: "Measured under S-7 (SI-62): median 120 ms (range 108–164, N=30) for Q-B5 work submission invocation (success, loopback) on the miniature fixture, embedded journal, Class L transport, double service median 0.3 ms, host fingerprint in docs/S-7-seam-measurements.md; single-run; live fork interop performance remains UNMEASURED."
- `LEASH_CROSSING_LABEL` gains: "Measured under S-7 (SI-62): median 124 ms (range 103–156, N=30) for Q-C3 leash per-record crossing on the miniature fixture, embedded journal, Class L transport, double service median 0.4 ms, host fingerprint in docs/S-7-seam-measurements.md; single-run; live fork interop performance remains UNMEASURED."
- `LEASH_CROSSING_LABEL_C1` gains: "Measured under S-7 (SI-62): median 0.1 ms (range 0.1–0.4, N=30) for Q-C1 leash fold check (stop in force) on the miniature fixture, embedded journal, Class S transport, host fingerprint in docs/S-7-seam-measurements.md; single-run; live fork interop performance remains UNMEASURED."
- `OUTCOME_RETURN_LABEL` gains: "Measured under S-7 (SI-62): median 1139 ms (range 1087–1648, N=10) for Q-D1 outcome poll pass (10 outstanding, all terminal) on the miniature fixture, embedded journal, Class L transport, host fingerprint in docs/S-7-seam-measurements.md; single-run; live fork interop performance remains UNMEASURED."
- `OUTCOME_RETURN_LABEL_COND` gains: "Measured under S-7 (SI-62): median 0.76 s (range 0.75–0.77, N=5) for conditional outcome latency at the stated schedule (offset 0 s, period 1 s, terminal at +0.35 s) on the miniature fixture, embedded journal, Class L transport, host fingerprint in docs/S-7-seam-measurements.md; single-run; unconditional outcome latency and live fork interop remain UNMEASURED."
- `EVENT_FORWARD_LABEL` gains: "Measured under S-7 (SI-62): median 152 ms (range 138–158, N=10) for Q-E4 event forward pass (100 events) on the miniature fixture, embedded journal, Class L transport, double service median 0.3 ms, host fingerprint in docs/S-7-seam-measurements.md; single-run; live fork interop performance remains UNMEASURED."
- `EVENT_FORWARD_LABEL_E5` gains: "Measured under S-7 (SI-62): median 0.55 s (range 0.50–0.58, N=5) for Q-E5 event catch-up (backlog 500, single unlimited pass) on the miniature fixture, embedded journal, Class L transport, host fingerprint in docs/S-7-seam-measurements.md; single-run; fork ingest (UNBUILT) and SSE visibility remain UNMEASURED."

---

Correctness riders: 391 checks, 0 failed. Phase sum-checks: Q-A4 and Q-B4 are the named residuals of §3.3. Raw per-repetition readings are in `docs/S-7-measurements.json`.

— End of S-7 seam measurements (SI-62). —
