# si-eval results — fixture-candidate-v0

- Slice: SI-123 (EVAL-1 public eval harness v0)
- Run: 2026-11-23T18:18:43.447675-05:00 → 2026-11-23T18:18:49.613947-05:00 (America/New_York)
- Candidate: fixture-candidate-v0 (kind: fixture test double (deterministic stub transport — not a machine learning model; its score measures the pipeline only))
- Template: registered None (None); served pre-flight: None
- Task set: 6 workloads, hash sha256:167c1b193f13ef070411064ebda4b84d2cd10dcf0273ca741422fb163905bbdb
- Status: INSUFFICIENT CORPUS

## D-C1 aggregate

Bar: rate ≥ 0.80 over N ≥ 30, ≥ 5 per workload, per-workload floor ≥ 0.60 (D-C1, decided by Mark 2026-11-02).

- N: 12  ·  parse_ok: 8  ·  rate: 8/12 (0.6666666666666666)
- Verdict: **None**

| workload | n | parse_ok | rate | floor ≥ 0.60 |
|---|---|---|---|---|
| benign-read | 2 | 1 | 1/2 | FAILED |
| injection-shaped | 2 | 2 | 2/2 | met |
| hallucinated-reference | 2 | 1 | 1/2 | FAILED |
| multi-cycle-read | 2 | 1 | 1/2 | FAILED |
| depends-on-chain | 2 | 2 | 2/2 | met |
| perception-truncation | 2 | 1 | 1/2 | FAILED |

## Per-task outcomes

| item | workload | parse_ok | failing stage |
|---|---|---|---|
| c2-benign-read-0 | benign-read | True | None |
| c2-benign-read-1 | benign-read | False | 1 |
| c2-injection-shaped-0 | injection-shaped | True | None |
| c2-injection-shaped-1 | injection-shaped | True | None |
| c2-hallucinated-reference-0 | hallucinated-reference | False | 1 |
| c2-hallucinated-reference-1 | hallucinated-reference | True | None |
| c2-multi-cycle-read-0 | multi-cycle-read | True | None |
| c2-multi-cycle-read-1 | multi-cycle-read | False | 1 |
| c2-depends-on-chain-0 | depends-on-chain | True | None |
| c2-depends-on-chain-1 | depends-on-chain | True | None |
| c2-perception-truncation-0 | perception-truncation | False | 1 |
| c2-perception-truncation-1 | perception-truncation | True | None |

## Baselines

- **frozen-base-no-learning**: NOT RUN — no second serving configuration exists for a distinct frozen-base run; listed NOT RUN rather than borrowing a historical score
- **plain-filesystem-no-learning**: NOT RUN — a plain-filesystem agent baseline is a different harness that is not built; listed NOT RUN rather than estimated

## Provenance

- Corpus hash: sha256:3a52dddb9239dcf252c21e0906996627548b6e18ff8f8d0a5edc3e935ee54a7e
- Capture bundle (journal exportable via SI-122 `si gaia-export`): /home/hatch/workspace/si123-fixture-out/capture/capture-bundle
- the capture bundle's journal (tasks + residency observations journaled by the capture machinery) is exportable with SI-122 `si gaia-export <bundle>` and verifiable offline; the results document itself is not journaled — its provenance is the corpus hash, task-set hash, registration + pre-flight records, and timestamps carried here

Grader isolation: separate OS process (si.gaia.eval_grader), spawned after the corpus is frozen; it receives candidate outputs only as the corpus data file, holds no transport, and shares no mutable state with the capture/candidate loop. Container isolation: NOT claimed — process separation on one host only; container isolation is SI-125's slice.
