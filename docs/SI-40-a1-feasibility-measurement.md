# SI-40: A-1 feasibility measurement — as built, with real numbers

The A-1 slice of RFC SI-39 §5.3, approved by the operator
(Mark) on 2026-10-29 (§8's yes), built and measured on this
VM, CPU only. SI-39 named the candidate, priced the
categories, and labelled every quantity UNMEASURED; this
note replaces those labels with measurements — or says, in
the open, where a measurement could not be taken.

Every number below is **measured** (produced by a run of
the code in this slice on 2026-10-29, on this VM: 2 vCPUs,
7.7 GiB RAM, 200 GB disk), **sourced** (read from the
Hugging Face record the same day), or **derived**
(arithmetic over measured/sourced numbers, shown so it can
be checked). The raw measurement records are in
`docs/SI-40-measurements.json`. Nothing was carried from
SI-39's derived column into a result column without being
re-measured: where SI-39 derived and A-1 measured, both
figures are shown.

> **The headline, first:** at this scale the CPU-only
> discipline costs seconds, not feasibility. A full forward
> over the named checkpoint takes 0.51 – 2.71 s for a 5-token
> sequence (1.36 – 3.92 s at 64 tokens); the heaviest assembly
> the verify loop needs — two float32 copies, base +
> candidate — peaks at ≈ 3.86 GB RSS of 7.7 GiB; a full
> by-reference verification of the seed takes 3.09 – 7.09 s. The
> A-1 verdict is **FEASIBLE on this VM** for the
> measure → assemble → verify cadence, in the terms §5.5
> demanded. What A-1 does *not* establish is listed in §6,
> and it is not a short list by accident.

## 1. What landed

- `si/llama_measure.py` (new) — the minimal measurement
  harness: llama-config validation (llama family surface
  only, refused otherwise), weight assembly at the four
  §3.4 widths, a self-contained llama decoder forward
  (pre-norm RMSNorm, RoPE rotate-half, grouped-query
  attention, SwiGLU MLP, tied embeddings), the verify
  compare, and a phase CLI that runs each measurement in a
  fresh subprocess so peak-RSS accounting is per-phase.
- `tests/test_llama_measure.py` (new, 13 tests) — a tiny
  locally constructed llama checkpoint (real safetensors
  format, honestly NOT trained) proves the harness's
  arithmetic against an independent naive float64
  implementation written in the test file (per-head loops,
  explicit per-position attention; max logit difference
  < 2e-4, the stated float32 tolerance), plus assembly
  byte-count, determinism, refusal, and compare tests.
- The checkpoint on disk (not in the repo):
  `~/workspace/smollm2-360m-safetensors/` — §5.1's file
  list only.
- The seeded bundle on disk (not in the repo):
  `~/workspace/sentinel-smollm2.si/` — seeded by reference
  (SI-8 machinery), §3 below.
- This note, and `docs/SI-40-measurements.json` (the raw
  records of the canonical measurement run).

**Not landed, by design:** no executor design (A-4), no
`reference-v1` binding design (A-2), no capability suite
(A-3), no gate change, no delta authorship against this
base, no serving. A bundle seeded from this checkpoint
still reports gate checks 3/4 "not evaluated" and
escalates — unchanged honest state.

## 2. Download (SI-39 §5.1, executed exactly)

Only the eight §5.1 files were downloaded from
`HuggingFaceTB/SmolLM2-360M-Instruct`. Sizes re-verified
against the Hugging Face API record fetched the same day;
the weights file's SHA-256 was verified against the LFS
hash in that record.

| File | Bytes (measured on disk) | HF record |
|---|---|---|
| `model.safetensors` | 723,674,912 | 723,674,912; SHA-256 `e6bffe74…bf86e` — **match** |
| `tokenizer.json` | 2,104,556 | 2,104,556 |
| `vocab.json` | 800,662 | 800,662 |
| `merges.txt` | 466,391 | 466,391 |
| `config.json` + `generation_config.json` + `tokenizer_config.json` + `special_tokens_map.json` | 5,397 combined | 5,397 combined |
| **Total** | **727,051,918** | ≈ 727,051,918 (derived in SI-39) |

Not downloaded: the ONNX variants, training logs, and
eval artifacts (≈ 4.3 GB of the repo's 5.02 GB total).
Disk: 120 GB free before, 119 GB free after (`df`,
measured both times; the difference also includes the
measurement venvs built for this slice).

The downloaded `config.json` re-confirms SI-39 §2's
sourced facts on disk: `LlamaForCausalLM`, 32 layers,
hidden 960, intermediate 2,560, 15 query / 5 KV heads,
vocab 49,152, tied embeddings, BF16. The safetensors
header records 290 tensors / 361,821,120 parameters
(measured during assembly, §4).

## 3. Seed by reference (SI-8 pattern)

`init_bundle` (NEDB journal backend, as the Glimmer seed)
+ `seed_bundle(..., by_reference=True)` against the
checkpoint directory:

- seed wall time **5.97 s** (measured, including the
  streaming canonical hashing of all 290 tensors);
- `seed_format: reference-v1`, **`stored_bytes: 0`**,
  `referenced_bytes: 723,642,240`, 290 tensor refs;
- bundle directory total 584 KB on disk, of which the only
  stored blob is the 328-byte demo blob `si init` leaves
  behind (unreferenced by the seeded base; blobs are
  append-only evidence, per the SI-8 record — it is not a
  copy of any checkpoint tensor);
- checkpoint left in place at its recorded absolute path,
  which the by-reference base depends on by design.

## 4. Assembly: the §3.4 table, measured

Each width assembled in its own subprocess; `peak RSS` is
that subprocess's `ru_maxrss`. Assembled bytes are the
live tensors' summed element count × width, and in every
row equal SI-39's derived figure exactly (361,821,120
elements × 2 / 4 / 4×2 / 8).

| Assembly (§3.4 row) | Assembled bytes (measured) | Assembly wall, range over 3 runs (s) | Peak RSS, range (bytes) |
|---|---|---|---|
| BF16, as stored (one copy) | 723,642,240 | 0.02 – 1.66 | 246,267,904 – 250,638,336 |
| float32 (one copy) | 1,447,284,480 | 2.11 – 7.98 | 1,627,201,536 – 2,408,792,064 |
| float32, base + candidate (two copies) | 2,894,568,960 | 5.04 – 21.31 | 3,107,581,952 – 3,857,412,096 |
| float64-widened compute copy (one copy) | 2,894,568,960 | 3.78 – 6.12 | 3,522,846,720 – 3,856,203,776 |

Method notes, because the RSS column needs them: the BF16
row's peak RSS is far below its assembled bytes because
safetensors loads are memory-mapped and this phase never
touches most pages — logical residency is the assembled
bytes; RSS is what the kernel actually had resident. The
float32 rows convert (real copies), so their RSS tracks
assembled bytes plus the source mapping and runtime. Even
the heaviest row leaves more than half the VM's RAM
uncommitted at peak.

## 5. Forward wall time, and what the forward is worth

The harness's forward is this project's own code (§1), so
the wall times measure a forward SI runs, not a serving
stack's. Sequences are full prefill passes, float32
compute, 2 threads, no KV cache (none is needed to price a
single forward; a cache is an executor-design question).

| Sequence (token ids) | Wall, range over runs (s) | Peak RSS (bytes) |
|---|---|---|
| 1 token (`504`) | 0.28 – 0.41 | ≈ 2,408,800,000 |
| 5 tokens — "The capital of France is" (`504, 3575, 282, 4649, 314`) | 0.51 – 2.71 | ≈ 2,408,900,000 |
| 13 tokens (fixed id list, first run) | 0.86 (single run) | 2,407,997,440 |
| 64 tokens (the 5-token prompt cycled) | 1.36 – 3.92 | ≈ 2,409,500,000 |

The wide wall ranges are host contention, measured not
hidden: this VM also runs the agent runtime, and concurrent
work (including this slice's own test-suite run) shares its
2 vCPUs. The *uncontended-class* figures are the low ends;
the verdict in §7 rests on the high ends. Two invariants
held across every run: the 5-token logits checksum
(521539.893, float64 sum) and last-position argmax (7042)
were identical in all three measurement runs — the
arithmetic is deterministic; only the clock moves.

**Numeric validation — reported, not presumed.** The same
5 token ids were run through Hugging Face `transformers`
5.22.0's `LlamaForCausalLM` (float32, CPU, this VM;
installed only in this slice's measurement venv, not a
project dependency) and through this harness:

- max absolute logit difference over all 5 × 49,152
  logits: **2.10e-05** (mean 2.20e-06) — float32
  implementation-order noise, stated as measured;
- argmax agreement: **5 / 5 positions**; last-position
  top-5 identical in order;
- both implementations' greedy next token: **`" Paris"`**
  (token 7042);
- reference forward wall in the same session: 0.759 s vs
  the harness's 0.543 s in-process (both include no load
  time; the harness number here is the in-process call,
  the table above is the clean-subprocess phase).

Together with the unit tests' naive-float64 proof on the
tiny checkpoint, this is the basis for calling the
harness's forward a llama forward. It is a *measurement*
basis: no bit-exactness is claimed, and A-4's proof
obligations (float32/float64 discipline, GQA and
tied-embedding proofs) remain A-4's to discharge properly.

## 6. Verification cost

The two costs a self-write cycle would pay per verify, on
the seeded base:

| Verify step | Wall, range over 3 runs (s) | Peak RSS, range (bytes) | Result |
|---|---|---|---|
| `load_bundle` full verification (streaming SHA-256 of the referenced file + per-tensor canonical hashes, journal + manifest checks) | 3.09 – 7.09 | 52,154,368 – 52,293,632 | passed; `stored_bytes 0`, 290 tensors |
| Assemble base + candidate (both float32) and compare all 290 tensors elementwise | 1.83 – 6.98 (compare step only; the phase's assemblies are the §4 pair row) | 3,854,352,384 – 3,857,244,160 | max abs diff **0.0**, as constructed |

Honest labels on the second row: no delta exists against
this base, so the "candidate" is an independent second
assembly of the same checkpoint — the row prices the
assemble + compare machinery a verify would run, and its
0.0 is by construction (the test suite separately proves
the compare detects and names a perturbed tensor). The
assembly half of that row is the `float32_pair` row of
§4. One full verification *pass* — bundle verification +
pair assembly + compare, summed from the measured rows —
costs ≈ 10 – 35 s of wall on this VM across the observed
range (derived by summation of measured rows, not itself a
single measured run).

Seed-time verification for scale: the initial `seed`
itself (which hashes everything once, §3) took 5.97 s,
and a standalone `load_bundle` in the seeding session
took 4.82 s — inside the table's range.

## 7. Verdict

**FEASIBLE — at this scale, on this VM, for the
measure → assemble → verify cadence A-1 was asked to
price.** The numbers behind the sentence:

- forward: sub-second at the uncontended end, and even
  the worst contended ladder point (64 tokens, 3.92 s)
  is seconds-class;
- residency: worst measured peak 3,857,412,096 bytes
  (≈ 3.86 GB) RSS against 7.7 GiB — every §3.4 row fits
  with headroom, including the float64-widened copy;
- verification: 3 – 7 s for full hash verification,
  seconds for assemble + compare;
- the exchange rate §5.5 left unmeasured is therefore:
  **a full measure-verify cycle is priced in seconds,
  not minutes** — the CPU-only discipline at 360M costs
  time the loop can afford.

What this verdict is *not* (the honesty boundary):

- It is not an executor: the harness forward is a
  measurement instrument. It has no KV cache, no batching,
  no delta application, and it is wired to nothing —
  not the gate, not the loop, not `si serve`.
- It is not an admission path: nothing about the gate's
  behaviour toward this bundle changed (checks 3/4 "not
  evaluated", ESCALATE).
- It does not price generation (autoregressive decode
  over long outputs), long-context forwards (the ladder
  stops at 64 tokens; attention cost grows quadratically
  and 8,192-token residency of activations was **not
  measured**), or training/authorship against this base.
- The float32-vs-float64 compute question is narrowed,
  not settled: float32 assembly + forward is what was
  validated against the reference (2.10e-05); a float64
  *compute* pass was priced for residency only (§4) and
  its wall time was **not measured**. A-4 decides the
  discipline with these numbers in hand.
- Timings are this VM under this slice's conditions
  (2 vCPUs shared with the host runtime and, during some
  runs, this slice's own test suite). Ranges over three
  full measurement runs are reported throughout instead
  of any single run's figures; the canonical run's raw
  record is `docs/SI-40-measurements.json`. The orders of
  magnitude — seconds, single-digit GB — which are what
  the verdict rests on, were stable across runs even
  when individual figures moved by ~5x.

**Recommended next step (for the operator / a later
slice, not approved by this one):** A-2, the
`reference-v1` binding design — the tamper analysis for a
base a future executor would actually run against. A-1's
numbers retire the feasibility unknown that sequenced it;
nothing in this note approves starting it.

## 8. Reproduce

```
# checkpoint at ~/workspace/smollm2-360m-safetensors (§2)
# bundle at ~/workspace/sentinel-smollm2.si (§3)
cd <repo>
~/workspace/si2-venv/bin/python -m si.llama_measure \
  --checkpoint ~/workspace/smollm2-360m-safetensors \
  --bundle ~/workspace/sentinel-smollm2.si \
  --tokens 504,3575,282,4649,314
~/workspace/si2-venv/bin/python -m pytest tests/test_llama_measure.py -q
```
