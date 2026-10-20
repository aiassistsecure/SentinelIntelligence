# SI-12 — Squash reconciliation for transformer deltas: fail closed (as-built)

**Status: implemented.** `si/squash.py` (explicit transformer-delta
refusal in `squash_bundle`), and `tests/test_squash_transformer.py`.
This is NOT transformer squash: composition/SVD over named tensors
is not implemented, here or anywhere in SI. RFC SI-9 §6.4 lists "No
transformer squash" as an explicit non-goal, and
`docs/SI-11-export-reconciliation.md` recorded "Transformer squash
remains deferred (RFC §6.4)". SI-12 makes `si squash` honour that
honestly, with a named error, instead of failing by accident.

## The gap (reproduced before the fix)

`squash_bundle` had no explicit check for
`kind == "transformer-delta-v1"`. Its pipeline assumed demo deltas:
it read `d["rank"]`, `d.get("target_modules")`, wrote a demo
`delta.B`/`delta.A` candidate blob, and loaded the stack through
`si.loop._load_stack` (which expects demo blob keys and record-level
`rank`/`alpha`). A transformer record carries none of those (rank and
alpha are per-target, in `targets`; blob keys are `t{i}.A`/`t{i}.B`/
`t{i}.d`). What actually happened depended on the stack — every case
below was reproduced on synthetic miniature bundles
(`tests/test_admit.py` / `tests/test_export_transformer.py` fixture
style) before any fix was written:

1. **Multi-tensor base, transformer deltas in the squash segment:**
   refused with `SquashError`, but only by the generic SI-3
   single-tensor scope error ("the base has N tensors; refusing to
   guess a squash layout"). The transformer problem was never named.
   Manifest, journal, and blob store unchanged.
2. **Single-tensor base, transformer delta in S, default rank:**
   crashed with a raw `KeyError: 'rank'` at the
   `max(int(d["rank"]) for d in S)` line. Nothing was written — but
   only by accident of ordering (the crash precedes the candidate
   blob write), not by design, and the error was not `SquashError`.
3. **Single-tensor base, transformer delta in S with an explicit
   `--rank`, or a transformer delta only in the kept tail K**
   (demo delta in S, base tensor `W` `[8, 8]` so the demo gate mind
   can load): the run got *further*, and worse. It computed the
   combined update (SI-11's materialize path applies transformer
   deltas correctly), wrote a **demo-format candidate blob**
   (`delta.B`/`delta.A`) into the blob store, passed the tolerance
   verify — and then crashed with a raw
   `safetensors._safetensors_rust.SafetensorError: File does not
   contain tensor delta.B` from `_load_stack` trying to read the
   transformer blob as a demo blob for the gate. Manifest and journal
   were unchanged, but **the orphaned demo-format candidate blob was
   left in the blob store**. No completed silent conversion to a
   demo delta was observed in any reproduced path — the gate stack
   load crashed first — but a demo-format candidate for a transformer
   stack was written, and nothing in the code *intended* to stop the
   conversion; it stopped by crashing.
4. **Mixed stack, transformer in S + demo in K (default rank):**
   case 2's `KeyError: 'rank'`.

So: a transformer delta was never *explicitly* refused, the errors
that did occur were raw library/dict errors rather than the named
`SquashError`, and one reachable path left a written artifact behind
before failing.

## Design decision

**Refuse explicitly, early, and over the whole resolved active
chain.**

- **Check:** in `squash_bundle`, immediately after the active chain
  is resolved (`bundle._active_chain`) and split into S/K, and
  before the single-tensor scope check, any SVD, any blob write, any
  journal open, or any manifest change: if any record on the chain
  has `kind == "transformer-delta-v1"`, raise `SquashError` naming
  the delta id(s) and stating that transformer squash is not
  implemented / is deferred (RFC §6.4). The CLI therefore surfaces
  it as a clean `si: error: …` (SquashError is a `ValueError`),
  never a raw `KeyError`/`SafetensorError`.
- **Scope — the whole chain (S ∪ K), not only S.** A squash
  *involves* the kept tail: K is materialized for the tolerance
  verify and loaded as demo factors by `_load_stack` for the gate's
  baseline and post-squash stacks, and the candidate's provenance
  and gate context are built over the full stack. Case 3 above is
  exactly a K-only failure (orphan blob + raw crash). Refusing only
  S would leave that path live. Mixed stacks are therefore refused
  in both directions (transformer in S + demo in K; demo in S +
  transformer in K), which the tests cover separately.
- **Scope — resolved chain only.** Records that are not on
  `_active_chain` — rolled back, superseded history, or beyond the
  current head — are never loaded, applied, or gated by a squash,
  so they do not block one. A test proves a demo squash proceeds
  normally with an inactive transformer record sitting in the
  manifest's history, and that the record is left untouched (not
  marked `superseded_by`).
- **Scope — the `L <= keep` no-op is unchanged.** When the active
  stack is no deeper than `keep`, no squash would run and no delta
  of any kind is involved; `squash_bundle` returns
  `{"squashed": False}` exactly as before, for transformer and demo
  stacks alike. The refusal applies when a squash would actually be
  attempted (`L > keep`).
- **No composition, no conversion.** SI-12 does not implement
  transformer-delta composition/SVD over named tensors, does not
  change the demo squash path's behaviour (existing
  `tests/test_squash.py` passes unchanged), and never converts a
  transformer record into a demo record.

## Tests

`tests/test_squash_transformer.py` (7 tests, synthetic miniature
bundles only, CPU only):

- transformer delta in S refused with `SquashError` — default rank
  and with an explicit `rank=2` (the pre-fix orphan-blob path) —
  with bundle/manifest/journal/blob store provably unchanged
  (canonical manifest bytes, journal collection counts, and the
  blob-file list are snapshotted before/after);
- multi-tensor transformer stack refused with the transformer
  message, not the generic single-tensor scope message;
- transformer delta only in K refused, unchanged;
- mixed stack (transformer in S, demo in K) refused, unchanged;
- CLI surfaces the refusal as a clean named error (no
  `KeyError`/`SafetensorError` in stderr), unchanged;
- `L <= keep` remains the no-op return on a transformer stack;
- an inactive (not-on-active-chain) transformer record does not
  block a demo squash, and is left untouched by it.

Full suite (this run, `~/workspace/si2-venv`): baseline on main
`209 passed, 2 skipped` → after SI-12 `216 passed, 2 skipped`
(+7 new tests; one baseline run first hit the known stale-`nedbd`
flake in `test_seed_multifile`, which passed on rerun after the
stale daemon was killed — the clean numbers above are the rerun).

## Honest limits — what SI-12 does not do

- **No transformer squash.** Composition of transformer deltas
  remains deferred (RFC §6.4). This change only refuses honestly.
- **No serve change** — `si serve` is untouched and still returns
  501 for transformer bundles. No export, gate, admission, apply,
  seed, or loop change.
- **No training, no inference, no probe harness.** Nothing here
  runs a transformer.
- **No full-Glimmer squash attempt.** Tests are synthetic miniature
  bundles only; the seeded Glimmer bundle and its checkpoint were
  not touched.
- A refusal is **not journaled**: like the existing tolerance /
  scope refusals that precede any journal activity in `si squash`,
  a transformer refusal writes nothing anywhere. (Gate REJECT /
  ESCALATE verdicts, which occur later in the pipeline for demo
  squashes, remain journaled exactly as before.)

This is machine learning infrastructure, not "custom LLM software."
CPU only; no GPU, cloud, or spend.
