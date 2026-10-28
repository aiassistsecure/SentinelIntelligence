# SI-38: transformer squash — as built

Implementation of RFC SI-37's design (`docs/SI-37-transformer-squash-rfc.md`):
the §3.8 composite pricing discipline, the §4 gate context, and the §5
tolerance verify, for the landed `d0001`+`d0002` mixed-target delta stack.
Built under RFC SI-37 §6 trigger (b), operator direction. The trigger's
other arms did not fire and are not claimed: stack depth never exceeded
`keep`, and §§3–5 of the RFC affirmatively demonstrated no other
present-tense need.

Every number below is either **measured** (produced by a run of the code
in this slice, asserted in `tests/test_transformer_squash.py`) or
**derived** (RFC SI-37's own arithmetic, labelled as such). Nothing was
carried over from a design document into a result column.

## What landed

- `si/transformer_squash.py` (new) — the segment attestation recovered
  from the journal, the exact composite carrying, the §5 verify, the
  `TransformerSquashContext` gate context, and the §4.7 pipeline:
  preconditions → verify → gate → commit.
- `si/squash.py` — `squash_bundle` classifies the resolved chain:
  mixed demo/transformer chains fail closed; an all-transformer chain
  delegates to the SI-38 pipeline; an explicit `--rank` on a
  transformer chain refuses (lossy recompression is undesigned,
  RFC §3.5). `squash_bundle` accepts an optional driver `Gate`, the
  `run_cycles` precedent.
- `si/gate.py` — the gate verifies a squash attestation **from primary
  records** (the context-supplied verified manifest plus journaled
  per-member axis records) before any composite pricing applies, and
  prices a verified composite under §3.8: attested distinct-tensor
  count, per-target exact-carrying rank, norms under the unchanged
  caps, bytes under the unchanged cap, additive members excluded.
  A claimed-but-unverifiable attestation fails decisively; a
  self-declared two-target authored candidate gets none of it (the
  authored budgets `max_targets_per_delta` 1 / `allow_additive` false
  are unchanged and still price authored candidates).
- Tests: `tests/test_transformer_squash.py` (new, 12 tests);
  `tests/test_squash_transformer.py` updated to the new refusal
  points; the superseded SI-12 assertions in
  `tests/test_final_mlp_down_target.py` and
  `tests/test_branch_expressibility_audit.py` updated in place to the
  SI-38 behaviour, with the change noted inline in both files.

## The counterfactual subject, as built (§5.7)

Full segment S = [`d0001`, `d0002`], kept tail K = [], run at `keep=0`
on the depth-2 landed stack (signal 1 `lm-head` delta `d0001`,
norm 0.23555380990005192, blob 2,360 B; signal 2 base-anchored
`final-mlp-down` delta `d0002`, norm 0.30801411790830574, blob 3,416 B;
cumulative identity drift at `d0002` 0.17195402272045612).

## Measured results

**Composite carrying.** One record, kind `transformer-delta-v1`,
parent `base`, targets in segment order
[`lm_head.weight`, `model.language_model.layers.1.mlp.down_proj.weight`],
per-target rank/alpha 8/8 (the members' own; scale 1, verbatim
factors). Aggregate norm **0.38776061454106125** — measured, and equal
to RFC §3.2's derivation √(0.23555380990005192² + 0.30801411790830574²).
Serialised blob **5,704 B**, vs the segment's 5,776 B summed — one
safetensors header instead of two; the factor payload is 5,120 B
(derived, RFC §3.2). Two independently landed stacks squash to the
identical composite blob hash (§3.8 rule 6) — which required one
real fix, recorded here: the safetensors serialiser emits
`__metadata__` in hash-map order, varying per call, so
`construct_transformer_delta`'s raw output is NOT byte-deterministic
(measured during this slice: two hash variants over repeated
identical constructions, payload always identical). The composite
therefore canonicalises its blob header (metadata first, keys
sorted, total header length preserved, equivalence re-parsed and
proven) and takes its content address over the canonical bytes.
Authored deltas' serialisation is pre-existing landed behaviour
and is untouched by this slice.

**§5 verify: EXACT, not merely within tolerance.** Weight-space
max-abs folded difference **0.0** on both named tensors; behavioural
statistic (max over all positions, all four allowed populations —
signal 1 train 32 + held-out 16, signal 2 train 32 + held-out 16)
**0.0**; per-population last-position maxima all **0.0**. RFC §5.5's
derivation expected rounding scale (~1e-5 or below); the measurement
is stronger and is reported as measured: the exact carrying folds and
forwards bit-identically on this miniature. The verify writes nothing,
reads no probe data, and a bundle copy with a perturbed probe artifact
produces a byte-identical verify record (asserted).

**Gate, driver policy** (the policy the stack landed under:
`allowed_target_patterns` opened to `["*"]` by the driver, every other
key the bundle's own): **ADMIT**. Constitution pass; safety pass with
the composite pricing stated in the report (attested count 2,
exact-carrying ranks 8/8, aggregate 0.38776061454106125 ≤ norm cap 1.0,
blob 5,704 B ≤ 4 MiB, no additive targets, resulting depth 1 ≤ 8);
identity **measured-pass at 0.17195402272045612** — `d0002`'s own
cumulative drift, exactly, because the post-squash stack computes the
same function; capability **measured-pass**, regression 0.0, echo
16/16 (1.0 vs floor 0.9375); provenance pass with the attestation
verified. The composite lands as `d0003`: both members journaled and
marked `superseded_by` `d0003`, `squashed_from` + verify provenance
journaled, **head id unmoved at `d0002`**.

**Gate, bundle's own policy** (empty allowlist — the fail-closed
default): **REJECT**, on the allowlist line alone. Every other check,
including both measured probe checks, passes; nothing lands, nothing
is superseded; the rejection is journaled. This is RFC §11 item
2(iv)'s policy question met in practice: composite admission today
requires the driver policy under which the stack itself was admitted.

## The resolver consequence, stated plainly

The head does not move (§4.7), and the bundle's standing
`_active_chain` semantics say a head carrying `superseded_by` resolves
**true historical ancestry** (the time-travel rule). So immediately
after this full-segment squash, the resolved chain at head `d0002` is
[`d0001`, `d0002`] — the historical stack, which the verify proved
computes identically to the composite (bit-identical assembled
weights, asserted both ways). The composite becomes the resolved
ancestry for any future head written above the segment. This is the
demo squash's own unexercised head-in-segment property, inherited,
not introduced; it is safe precisely because admission required the
§5 exactness proof first, and it is recorded here rather than
discovered later.

**Unwind (§4.6's required proof, on the real stack shape):** after the
squash, `rollback` to `d0001` restores that head with bit-identical
weights; `rollback` to `base` restores the bare base. Asserted.

## What still refuses (fail closed, nothing written)

Segment of one ("not a compression"); any additive member in the
segment (named, §3.8 rule 5); a non-miniature head (the §5 verify has
no lawful population — Glimmer scale is untouched and unreachable
from this path); mixed demo/transformer chains; explicit `--rank` on
a transformer chain; a failed verify (candidate blob deleted again,
nothing journaled); an axis record that cannot be recovered whole
from the journal; a tampered member axis record or mis-ordered
parents at the gate (attestation fails, decisive REJECT, no composite
pricing).

## Test accounting

Full suite after this slice: **406 passed, 2 skipped** (baseline
394 passed, 2 skipped; +12 new tests in
`tests/test_transformer_squash.py`, no test removed). CPU only; the
repository venv; no GPU, cloud, or spend.
