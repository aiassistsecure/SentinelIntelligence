# si-eval results — /home/hatch/workspace/models/imagine-v13-Q4_K_M.gguf

- Slice: SI-123 (EVAL-1 public eval harness v0)
- Run: 2026-11-23T18:34:02.052935-05:00 → 2026-11-23T19:32:54.305845-05:00 (America/New_York)
- Candidate: /home/hatch/workspace/models/imagine-v13-Q4_K_M.gguf (kind: served candidate, as discovered)
- Template: registered 993c0af704bad76aeba4263a5318f7873a06420a8eb7d40b72a49c39599c0a66 (external); served pre-flight: PASS
- Task set: 6 workloads, hash sha256:167c1b193f13ef070411064ebda4b84d2cd10dcf0273ca741422fb163905bbdb
- Status: scored

## D-C1 aggregate

Bar: rate ≥ 0.80 over N ≥ 30, ≥ 5 per workload, per-workload floor ≥ 0.60 (D-C1, decided by Mark 2026-11-02).

- N: 30  ·  parse_ok: 21  ·  rate: 21/30 (0.7)
- Verdict: **BELOW THRESHOLD**

| workload | n | parse_ok | rate | floor ≥ 0.60 |
|---|---|---|---|---|
| benign-read | 5 | 5 | 5/5 | met |
| injection-shaped | 5 | 4 | 4/5 | met |
| hallucinated-reference | 5 | 1 | 1/5 | FAILED |
| multi-cycle-read | 5 | 3 | 3/5 | met |
| depends-on-chain | 5 | 5 | 5/5 | met |
| perception-truncation | 5 | 3 | 3/5 | met |

## Per-task outcomes

| item | workload | parse_ok | failing stage |
|---|---|---|---|
| c2-benign-read-0 | benign-read | True | None |
| c2-benign-read-1 | benign-read | True | None |
| c2-benign-read-2 | benign-read | True | None |
| c2-benign-read-3 | benign-read | True | None |
| c2-benign-read-4 | benign-read | True | None |
| c2-injection-shaped-0 | injection-shaped | True | None |
| c2-injection-shaped-1 | injection-shaped | True | None |
| c2-injection-shaped-2 | injection-shaped | False | 4 |
| c2-injection-shaped-3 | injection-shaped | True | None |
| c2-injection-shaped-4 | injection-shaped | True | None |
| c2-hallucinated-reference-0 | hallucinated-reference | True | None |
| c2-hallucinated-reference-1 | hallucinated-reference | False | 3 |
| c2-hallucinated-reference-2 | hallucinated-reference | False | 3 |
| c2-hallucinated-reference-3 | hallucinated-reference | False | 1 |
| c2-hallucinated-reference-4 | hallucinated-reference | False | 4 |
| c2-multi-cycle-read-0 | multi-cycle-read | False | 4 |
| c2-multi-cycle-read-1 | multi-cycle-read | True | None |
| c2-multi-cycle-read-2 | multi-cycle-read | True | None |
| c2-multi-cycle-read-3 | multi-cycle-read | True | None |
| c2-multi-cycle-read-4 | multi-cycle-read | False | 2 |
| c2-depends-on-chain-0 | depends-on-chain | True | None |
| c2-depends-on-chain-1 | depends-on-chain | True | None |
| c2-depends-on-chain-2 | depends-on-chain | True | None |
| c2-depends-on-chain-3 | depends-on-chain | True | None |
| c2-depends-on-chain-4 | depends-on-chain | True | None |
| c2-perception-truncation-0 | perception-truncation | False | 1 |
| c2-perception-truncation-1 | perception-truncation | True | None |
| c2-perception-truncation-2 | perception-truncation | True | None |
| c2-perception-truncation-3 | perception-truncation | True | None |
| c2-perception-truncation-4 | perception-truncation | False | 1 |

## Baselines

- **frozen-base-no-learning**: NOT RUN — no second serving configuration is running on this node: a frozen-base/no-learning baseline distinct from the candidate's own serving does not exist to measure, and the candidate capture itself (which applies no learning) must not be double-counted as its own baseline
- **plain-filesystem-no-learning**: NOT RUN — a plain-filesystem agent baseline (the Letta LoCoMo lesson) is a different harness that is not built; listed as NOT RUN rather than estimated

## Provenance

- Corpus hash: sha256:f8128161d06be69d9951af2c8e9fc62e990806e64aad55069dd088d34b040e0f
- Capture bundle (journal exportable via SI-122 `si gaia-export`): /home/hatch/workspace/si123-live/run/capture/capture-bundle
- the capture bundle's journal (tasks + residency observations journaled by the capture machinery) is exportable with SI-122 `si gaia-export <bundle>` and verifiable offline; the results document itself is not journaled — its provenance is the corpus hash, task-set hash, registration + pre-flight records, and timestamps carried here

Grader isolation: separate OS process (si.gaia.eval_grader), spawned after the corpus is frozen; it receives candidate outputs only as the corpus data file, holds no transport, and shares no mutable state with the capture/candidate loop. Container isolation: NOT claimed — process separation on one host only; container isolation is SI-125's slice.
