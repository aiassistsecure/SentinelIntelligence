# SI-14 — Miniature seed + probe artifacts (as-built)

> **Scope:** RFC SI-13 §3 (miniature construction) and §5 (probe
> suites, baselines, probe artifact), per §8's SI-14 slice. This
> slice **records** probes; it **consumes** none. Gate behaviour is
> byte-for-byte unchanged from SI-9b/SI-10: every transformer delta —
> miniature ones included — still reports checks 3/4 as "not
> evaluated" and still ESCALATES. Nothing in this slice wires a
> forward path to the gate, the loop, or `si serve`.

## As-built decisions (divergences from RFC defaults, each explicit)

1. **Positional embedding table: INCLUDED.** RFC §3.3 left the
   decision to SI-14. The table is part of the base as
   `model.language_model.embed_positions.weight` (`[16, 32]`).
   Inventory is therefore **30 tensors / 23,456 elements** — §3.1's
   budget (29 / 22,944) plus exactly the +1 tensor / +512 elements
   the RFC flagged for this variant. Total base: 93,824 raw bytes
   float32; 93,920 unique content-addressed blob bytes (dedup
   included below).
2. **Tensor names** mirror the Glimmer-style `model.language_model.`
   family used by the SI-9 tests, so SI-9a delta binding, SI-9b
   allowlists, SI-9c apply, and SI-11 export apply to the miniature
   unchanged — no second targeting system.
3. **Seed path:** a new module, `si/tiny_seed.py`
   (`seed_tiny_bundle`), plus the CLI subcommand `si seed-tiny`.
   The §4 forward is an **internal seed-time helper** in that same
   module — it computes expected outputs, margins, and the echo
   floor at seed time and is wired to nothing (no gate, no loop,
   no serve). SI-15 owns the library forward path and may replace
   or extract this helper; SI-16 owns any consumption.
4. **`readout_shift` construction parameter** (default 0, honest
   construction) exists so tests can build a seed that *fails* the
   echo floor deterministically — the fail-closed proof (§8
   acceptance criterion), not a serving knob.

## Construction (deterministic, NOT trained)

`si/tiny_seed.py::build_tiny_weights` from one
`numpy.random.default_rng(20261020)` stream, in fixed order:

- `embed_tokens.weight` `[32,32]`: `2.0 * I + 0.01·N` —
  residual-dominant (token *t* → direction *t*).
- `embed_positions.weight` `[16,32]`: `0.01·N`.
- `lm_head.weight` `[32,32]`: `0.5 * P_shift + 0.002·N` where
  `P_shift[(t+shift) % 32, t] = 1` — the approximate inverse
  readout; `shift = 0` normally, `1` to construct the failing
  fixture.
- All 5 RMSNorm weights: exactly `1.0`.
- All 8 attention projection biases: exactly `0.0` (present as
  tensors so `additive` deltas have a real home).
- Attention `q/k/v/o_proj` weights and MLP
  `gate/up/down_proj` weights: `0.02·N` — small but non-zero, so
  every branch is live and no projection class is blind to probes.

Internal forward helper (seed-time only): float64 arithmetic over
the float32 weights; token + positional embedding → 2 ×
(pre-norm RMSNorm → causal multi-head attention, 4 heads, head
dim 8 → residual → pre-norm RMSNorm → SwiGLU MLP → residual) →
final RMSNorm → `lm_head`; full `[len, 32]` logits. Sequences are
int token ids, `1 <= len <= 16`, no tokenizer. `rms_norm_eps =
1e-5` (recorded in the manifest architecture descriptor).

## Seed-time assertions, fail closed (RFC §3.3)

Run **before anything is written**, on the in-memory
construction, against the §5 suites:

- construction twice → byte-identical tensors;
- forward twice → bit-identical logits;
- finiteness of all tensors and all expected outputs;
- echo floor: base echo accuracy >= 15/16;
- tolerance sanity: derived tolerances positive.

## Measured on this seed (evidence, not figures of speech)

| Measurement | Value |
|---|---|
| Tensor inventory | 30 tensors, 23,456 elements |
| Unique blob bytes stored | 93,920 |
| Deduplicated (not rewritten) bytes | 2,200 (5 identical norm-ones blobs, 8 identical bias-zero blobs → 11 dedup events) |
| Blob files in store | 19 miniature + 1 inert demo base blob from `si init` (store is append-only; unreferenced) |
| Base echo accuracy (seed-time, 16 capability entries) | **16/16 = 1.0** |
| Identity min margin (top1−top2 logit gap) | **2.7158792540431023** |
| Capability min margin | **2.755554586648941** |
| `identity_tol` = min(0.25, 0.4 × 2.7159) | **0.25** (margin-bound → capped by demo-era default) |
| `capability_eps` = min(0.25, 0.4 × 2.7556) | **0.25** (same) |
| Echo floor recorded | 0.9375 (base accuracy − 1/16) |
| Probe artifact size (`probe_suite.json`) | 22,565 bytes |

The margins are large because the construction is
residual-dominant by design: the last token's direction survives
the normalisation layers at ~5.6 units against ~0.01-scale branch
noise, and `0.5 × lm_head` amplifies that gap to ~2.7 logits. The
0.25 caps are therefore loose for benign drift — SI-16's
reference deltas will prove empirically whether the tolerances
separate benign from destructive, per §5.6.

## Probe artifact (RFC §5.5)

`probe_suite.json` in the bundle, canonical JSON (no wall-clock,
so re-seeds are byte-identical), hashed as `sha256:` in
`base.probe_artifact` and **verified on every `load_bundle`**
(tamper/missing fails closed — a small additive check; demo and
Glimmer bases have no `probe_artifact` field and are untouched).
Contents: artifact kind, honesty label, construction seed, suite
generation procedure, architecture descriptor, base binding
(tensor-map `sha256`, 30 tensors, 23,456 elements), 8 identity +
16 capability entries with inputs / expected logits /
baseline 0.0 / margins, echo accuracy + floor, margins and the
derived tolerances. Journaled at seed time with provenance
(`seeded_from` on `base`, `probe_suite_recorded` on
`probe_suite`, naming the artifact hash, margins, tolerances,
and that probes are recorded, not consumed).

The per-bundle `policy.json` carries `identity_tol: 0.25` and
`capability_eps: 0.25` with the manifest `policy_hash` updated
at seed time — no policy default is changed, and no new global
default is introduced.

## Acceptance criteria (RFC §8 SI-14) — status

- ✅ Deterministic re-seeding → byte-identical base blobs +
  identical probe artifacts (tested).
- ✅ Inventory matches §3.1 exactly modulo the stated
  positional-table variant (tested: 30 / 23,456).
- ✅ Seed-time assertions run and are tested, including a
  constructed seed that fails the echo floor (`readout_shift=1` →
  0/16) and is refused with the bundle untouched (tested).
- ✅ Stored-bytes accounting: base stored once, probes add only
  the artifact (tested).
- ✅ Demo and Glimmer bundle behaviour untouched; full suite
  green on the author's own run: **229 passed, 2 skipped** —
  the pre-SI-14 baseline (216 passed, 2 skipped) plus this
  slice's 13 new tests, zero regressions.
- ✅ Miniature transformer deltas still ESCALATE with checks 3/4
  "not evaluated" (tested with the real bundle manifest and
  raising probe methods — the gate never touches probes in
  SI-14).

## Terminology

Machine learning. This document contains no "custom LLM software"
claims; the miniature is a constructed machine-learning test
instrument, not a trained model, and every manifest, artifact,
and journal note says so.
