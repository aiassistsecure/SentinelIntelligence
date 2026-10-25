# SI-35 — Branch-contribution and expressibility audit of the miniature (as-built)

> **Scope:** RFC SI-34 §6 (the audit candidate), §15 (the
> SI-35 outline, items 1–8), and §16 (the nine acceptance
> criteria — each addressed below with its measurements),
> plus §17 non-goals and §18 honesty / labelling (items
> 26–27 are new in that RFC and bind this slice). SI-35 is
> a **measurement-only** slice: it trains nothing, authors
> nothing, performs no gate evaluation, and makes no probe
> contact. For every weight matrix in the miniature it
> publishes the branch-contribution measurements, the
> input-activation records, and the derived rank-8
> expressibility bounds that SI-33 measured for one branch
> after its decisive run — computed on allowed populations
> only, at the named heads `base`, `d0001`, and `d0002`,
> under the readings RFC SI-34 §15 item 7 registered
> before any audit figure existed.
> **The readings, applied as registered: R1 NOT measured,
> R2 MEASURED, R3 NOT measured — exactly one of R1/R2, as
> registered.** No in-stream non-readout matrix satisfies
> R1's pair on the primary cell (signal 2's train
> partition at head `d0001`): the branch-ratio clause
> (≥ 0.04) fails for all 14 matrices — the largest
> measured branch ratio on the primary cell is the final
> layer's attention branch's 0.0341 — and the bound clause
> (≥ 0.212) fails for exactly the two down-projections.
> The shortlist is therefore **empty**, and R2's
> registered consequence stands as this slice's result:
> the single-target upstream-authorship line is
> *expressibility-closed on this miniature at the
> standing budgets* — a derivation-scoped conclusion from
> measured inputs, labelled as such, with its five
> qualifiers (single-target, upstream, rank-8,
> budget-0.9, this miniature, this signal family's
> correction scale) attached wherever it is quoted. If a
> future RFC exists in this line, it is SI-36 arguing
> from R2's consequence (RFC SI-34 §8's squash trigger,
> §9's semantics, or §4's operator decision), not a
> target RFC — this slice pre-writes none of them.
> **Attribution that travels (§18 item 26):** SI-33's
> cycle-2 ADMIT — the stack this audit measures at
> `d0002` — is an ADMIT of a candidate that does almost
> nothing: candidate norm 0.30801411790830574, probe-space
> movement ‖m2‖ = 0.010469039280455053, F3′′′ **measured**
> (the destination traded away: held-out residual
> 0.0027387202915082178 against the boundary
> 2.087217177504918e-05, ~131×), attributed by SI-33 to
> the near-inertness of the branch it acts through (the
> final layer's MLP branch ≈ 0.4% of the residual stream
> on the measured population). Nothing in this audit
> softens, extends, or re-judges that record; the audit
> generalises its *measurement*, not its verdict.
> **A measurement-only record is not a verdict record
> (§18 item 27):** a matrix's failure to qualify under R1
> is not a prediction that a candidate trained on it
> would REJECT (an inert candidate may well ADMIT, as
> SI-33's did), and no audit figure below is a gate
> figure of any kind.

## What was built

- **`si/tiny_audit.py`** (new) — the read-only measurement
  module. `forward_audit` repeats the library forward's
  computation in the same operation order and
  float64-over-float32 discipline, additionally exposing
  per position every branch output, the residual stream
  each branch adds into, and the activation each weight
  matrix consumes; its logits are asserted bit-identical
  to `si.tiny_transformer.forward`'s by test, so the
  audit measures the same function every prior slice
  measured. `weights_at_head` assembles a named head's
  weight set through the existing apply path from an
  in-memory manifest view (the `assemble_base_weights`
  precedent: the verified manifest is never mutated,
  re-headed, or written; nothing is materialised or
  stored — the never-duplicate-bytes rule holds by
  construction). `run_audit` fails closed, naming the
  missing head, on any bundle whose active chain does
  not carry the landed stack. The module invokes no
  training entry point, constructs no candidate, opens
  no gate evaluation, and reads no judged-population
  artifact (all asserted by test — see §16 criteria 2
  and 7 below). Its one probe-adjacent import is
  `derive_probe_blind_targets`, a pure derivation from
  the architecture descriptor that reads no input of any
  kind, needed for the item-6 inventory's blind set
  exactly as the gate needs it.
- **`tests/test_branch_expressibility_audit.py`** (new,
  12 tests) and this doc. No prior file is modified: no
  production code, policy default, signal definition,
  anchoring, composition configuration, or authorship
  target changed. No CLI wiring is added (SI-35 is a
  library-and-tests slice, §17); `si serve` still returns
  501 for every transformer bundle; the Glimmer seed is
  untouched.

## Method, populations, heads, positions, stop rule (§15 item 1)

- **Populations (allowed inputs only):** signal 1's train
  partition (32 sequences) and signal 2's train partition
  (32 sequences) as the primary populations; both
  signals' held-out partitions (16 each) reported only,
  under their own labels, never merged into a primary
  figure. No probe input, probe logit, or probe-derived
  quantity enters any figure, ranking, or reading in
  this document or in the audit record: the record from
  a perturbed-probe-artifact bundle copy is
  byte-identical to the unperturbed record (criterion 2).
- **Heads:** every figure is measured at a named head —
  `base`, `d0001` (signal 1, readout target), and `d0002`
  (signal 2, final-MLP-down target — the second delta
  labelled a non-readout-target candidate wherever the
  stack is described) — weights overlaid through the
  existing apply path. No figure is presented without
  its head.
- **Positions:** the last position is primary (the loss
  reads last-position logits only, as in every prior
  slice, and SI-33's attribution is a last-position
  record). All-position statistics are published in
  their own table, labelled, and no reading rests on
  them.
- **Stop rule:** if the reproduction below could not be
  reconciled with the audit's method, the slice would
  have stopped and published the discrepancy as its
  result. It did not trigger: the ranges reproduce, and
  the one figure that cannot reproduce exactly is
  reconciled in print with its method difference named,
  before any other figure is trusted.

## Reproduction of SI-33's attribution (§15 item 3 — the acceptance anchor, before any other figure is trusted)

Under the audit's method, on signal 2's train partition
at head `d0001` (last positions):

| Quantity | SI-33 published | SI-35 measured | Status |
|---|---|---|---|
| ‖s‖ range (final down-projection's input) | 0.040–0.064 | 0.04008178834376358–0.06405601655456196 | **reproduces** (the measured range rounds to the published range at its printed precision) |
| ‖h_mid‖ range (residual entering the final MLP block) | 1.97–2.02 | 1.9678289013214776–2.0192666824884555 | **reproduces** (same, at printed precision) |

At `base`, SI-33 published, for one representative
sequence it does not name, the final MLP branch's output
norm 0.0074 against a residual norm of 2.00. Exact
per-sequence reproduction is not possible from the
published record — the sequence is unnamed — so the
audit reconciles the difference in print, as §16
criterion 3 provides, with the method difference named:
**single unnamed sequence (SI-33) vs named population
(this audit), last position, mean/max published.** On
signal 2's train partition at `base` the same branch
measures branch-output norm mean 0.0056088839327087725 /
max 0.006931974601408632 and residual-stream norm mean
1.996662501317634 — the published residual 2.00 matches
the population mean to SI-33's printed precision, and
the published branch output 0.0074 sits inside the
envelope of the same branch's base measurements across
the audit's allowed populations (per-population maxima
0.00693–0.00857 over the four partitions; full table
below). The yardstick row — the final layer's MLP branch
on the primary population at `d0001` — measures
branch-output norm mean 0.0056088839327087725 (identical
to base: the `d0001` delta acts on the readout, downstream
of every branch the audit measures) and contribution
ratio max per-example 0.0034708334734584786, beside
SI-33's ~0.4% headline (0.0074 / 2.00 = 0.0037, a
derivation from SI-33's cited figures). The anchor
holds; the figures below are presented as trusted on
that basis, and on no other.

## The full tables (§15 items 2, 4, 5, 6)

The inventory first, in brief (criterion 1): all **30
tensors** appear in the audit's record — the **17 weight
matrices** (14 in-stream, the readout `lm_head.weight`
tabled as the record's yardstick target and not as a
candidate, and the two embedding matrices tabled
separately under the index-input caveat) and the **13
vector tensors**, inventoried with blind-set status:
the derived blind set is exactly both layers'
`k_proj.bias` (`model.language_model.layers.0.self_attn.k_proj.bias`,
`model.language_model.layers.1.self_attn.k_proj.bias`);
the other 11 vector tensors (the 6 remaining attention
biases, the 4 layer norm weights, and the final norm
weight — the inventory counts, per layer, 4 attention
biases + 2 norm weights, plus the final norm weight)
are measurable (not in the derived blind set). No weight matrix is in
the derived blind set — the blind set is biases only,
stated as the derived fact it is. Where a figure cannot
be carried it is recorded "not applicable" with the
reason — never as 0.0, never as an omission: the two
embedding matrices' input-activation rows (their input
is an index, not an activation) and the readout's
candidate bound (it is the yardstick, not a candidate).

**Stated limitation, in the body, not the footnotes
(§15 item 5):** a local bound prices movement at the
tensor's own output; downstream transfer to logits is
*not* measured by this audit for any tensor except
where a landed delta already measures it — the readout
(SI-25 control: norm 0.2515524481203623, ‖m2‖
0.7816722548913912) and the final down-projection
(SI-33: candidate norm 0.30801411790830574, ‖m2‖
0.010469039280455053, printed beside its bound wherever
the bound appears, §16 criterion 6). Every other
tensor's transfer coefficient is listed as **unknown**,
never estimated. A bound is never quoted as if it were
a movement, and no movement as if it were a bound.

### Branch-contribution measurements (measurements; last position)

| Branch | Population | Head | Branch output norm (mean / max) | Residual norm (mean / max) | Ratio (mean / max per-example) |
|---|---|---|---|---|---|
| embedding-sum | tiny-successor-lean-001:train (primary) | base | 1.99907 / 2.02265 | 1.99907 / 2.02265 | 1 / 1 |
| embedding-sum | tiny-successor-lean-001:train (primary) | d0001 | 1.99907 / 2.02265 | 1.99907 / 2.02265 | 1 / 1 |
| embedding-sum | tiny-successor-lean-001:train (primary) | d0002 | 1.99907 / 2.02265 | 1.99907 / 2.02265 | 1 / 1 |
| embedding-sum | tiny-predecessor-lean-001:train (primary) | base | 1.9965 / 2.02238 | 1.9965 / 2.02238 | 1 / 1 |
| embedding-sum | tiny-predecessor-lean-001:train (primary) | d0001 | 1.9965 / 2.02238 | 1.9965 / 2.02238 | 1 / 1 |
| embedding-sum | tiny-predecessor-lean-001:train (primary) | d0002 | 1.9965 / 2.02238 | 1.9965 / 2.02238 | 1 / 1 |
| embedding-sum | tiny-successor-lean-001:held_out (reported-only) | base | 1.99851 / 2.02021 | 1.99851 / 2.02021 | 1 / 1 |
| embedding-sum | tiny-successor-lean-001:held_out (reported-only) | d0001 | 1.99851 / 2.02021 | 1.99851 / 2.02021 | 1 / 1 |
| embedding-sum | tiny-successor-lean-001:held_out (reported-only) | d0002 | 1.99851 / 2.02021 | 1.99851 / 2.02021 | 1 / 1 |
| embedding-sum | tiny-predecessor-lean-001:held_out (reported-only) | base | 1.9944 / 2.01406 | 1.9944 / 2.01406 | 1 / 1 |
| embedding-sum | tiny-predecessor-lean-001:held_out (reported-only) | d0001 | 1.9944 / 2.01406 | 1.9944 / 2.01406 | 1 / 1 |
| embedding-sum | tiny-predecessor-lean-001:held_out (reported-only) | d0002 | 1.9944 / 2.01406 | 1.9944 / 2.01406 | 1 / 1 |
| layer-0-attention | tiny-successor-lean-001:train (primary) | base | 0.0330749 / 0.0612363 | 1.99907 / 2.02265 | 0.0165426 / 0.0304044 |
| layer-0-attention | tiny-successor-lean-001:train (primary) | d0001 | 0.0330749 / 0.0612363 | 1.99907 / 2.02265 | 0.0165426 / 0.0304044 |
| layer-0-attention | tiny-successor-lean-001:train (primary) | d0002 | 0.0330749 / 0.0612363 | 1.99907 / 2.02265 | 0.0165426 / 0.0304044 |
| layer-0-attention | tiny-predecessor-lean-001:train (primary) | base | 0.0367575 / 0.0604823 | 1.9965 / 2.02238 | 0.0184167 / 0.0303477 |
| layer-0-attention | tiny-predecessor-lean-001:train (primary) | d0001 | 0.0367575 / 0.0604823 | 1.9965 / 2.02238 | 0.0184167 / 0.0303477 |
| layer-0-attention | tiny-predecessor-lean-001:train (primary) | d0002 | 0.0367575 / 0.0604823 | 1.9965 / 2.02238 | 0.0184167 / 0.0303477 |
| layer-0-attention | tiny-successor-lean-001:held_out (reported-only) | base | 0.0384199 / 0.0574437 | 1.99851 / 2.02021 | 0.0192319 / 0.0290092 |
| layer-0-attention | tiny-successor-lean-001:held_out (reported-only) | d0001 | 0.0384199 / 0.0574437 | 1.99851 / 2.02021 | 0.0192319 / 0.0290092 |
| layer-0-attention | tiny-successor-lean-001:held_out (reported-only) | d0002 | 0.0384199 / 0.0574437 | 1.99851 / 2.02021 | 0.0192319 / 0.0290092 |
| layer-0-attention | tiny-predecessor-lean-001:held_out (reported-only) | base | 0.0404472 / 0.0540468 | 1.9944 / 2.01406 | 0.0202963 / 0.0272035 |
| layer-0-attention | tiny-predecessor-lean-001:held_out (reported-only) | d0001 | 0.0404472 / 0.0540468 | 1.9944 / 2.01406 | 0.0202963 / 0.0272035 |
| layer-0-attention | tiny-predecessor-lean-001:held_out (reported-only) | d0002 | 0.0404472 / 0.0540468 | 1.9944 / 2.01406 | 0.0202963 / 0.0272035 |
| layer-0-mlp | tiny-successor-lean-001:train (primary) | base | 0.00576636 / 0.00932336 | 1.9965 / 2.02153 | 0.00288836 / 0.00466391 |
| layer-0-mlp | tiny-successor-lean-001:train (primary) | d0001 | 0.00576636 / 0.00932336 | 1.9965 / 2.02153 | 0.00288836 / 0.00466391 |
| layer-0-mlp | tiny-successor-lean-001:train (primary) | d0002 | 0.00576636 / 0.00932336 | 1.9965 / 2.02153 | 0.00288836 / 0.00466391 |
| layer-0-mlp | tiny-predecessor-lean-001:train (primary) | base | 0.00522422 / 0.00661739 | 1.99381 / 2.01558 | 0.00261965 / 0.00329234 |
| layer-0-mlp | tiny-predecessor-lean-001:train (primary) | d0001 | 0.00522422 / 0.00661739 | 1.99381 / 2.01558 | 0.00261965 / 0.00329234 |
| layer-0-mlp | tiny-predecessor-lean-001:train (primary) | d0002 | 0.00522422 / 0.00661739 | 1.99381 / 2.01558 | 0.00261965 / 0.00329234 |
| layer-0-mlp | tiny-successor-lean-001:held_out (reported-only) | base | 0.00573315 / 0.0089839 | 1.99532 / 2.01927 | 0.00287537 / 0.00455314 |
| layer-0-mlp | tiny-successor-lean-001:held_out (reported-only) | d0001 | 0.00573315 / 0.0089839 | 1.99532 / 2.01927 | 0.00287537 / 0.00455314 |
| layer-0-mlp | tiny-successor-lean-001:held_out (reported-only) | d0002 | 0.00573315 / 0.0089839 | 1.99532 / 2.01927 | 0.00287537 / 0.00455314 |
| layer-0-mlp | tiny-predecessor-lean-001:held_out (reported-only) | base | 0.00523391 / 0.0065284 | 1.98974 / 2.00743 | 0.00262975 / 0.00325536 |
| layer-0-mlp | tiny-predecessor-lean-001:held_out (reported-only) | d0001 | 0.00523391 / 0.0065284 | 1.98974 / 2.00743 | 0.00262975 / 0.00325536 |
| layer-0-mlp | tiny-predecessor-lean-001:held_out (reported-only) | d0002 | 0.00523391 / 0.0065284 | 1.98974 / 2.00743 | 0.00262975 / 0.00325536 |
| layer-1-attention | tiny-successor-lean-001:train (primary) | base | 0.0388148 / 0.0771513 | 1.99639 / 2.02109 | 0.019448 / 0.0389777 |
| layer-1-attention | tiny-successor-lean-001:train (primary) | d0001 | 0.0388148 / 0.0771513 | 1.99639 / 2.02109 | 0.019448 / 0.0389777 |
| layer-1-attention | tiny-successor-lean-001:train (primary) | d0002 | 0.0388148 / 0.0771513 | 1.99639 / 2.02109 | 0.019448 / 0.0389777 |
| layer-1-attention | tiny-predecessor-lean-001:train (primary) | base | 0.037513 / 0.0674319 | 1.99413 / 2.01526 | 0.018822 / 0.0340859 |
| layer-1-attention | tiny-predecessor-lean-001:train (primary) | d0001 | 0.037513 / 0.0674319 | 1.99413 / 2.01526 | 0.018822 / 0.0340859 |
| layer-1-attention | tiny-predecessor-lean-001:train (primary) | d0002 | 0.037513 / 0.0674319 | 1.99413 / 2.01526 | 0.018822 / 0.0340859 |
| layer-1-attention | tiny-successor-lean-001:held_out (reported-only) | base | 0.0426025 / 0.0650403 | 1.99519 / 2.0189 | 0.0213479 / 0.0327044 |
| layer-1-attention | tiny-successor-lean-001:held_out (reported-only) | d0001 | 0.0426025 / 0.0650403 | 1.99519 / 2.0189 | 0.0213479 / 0.0327044 |
| layer-1-attention | tiny-successor-lean-001:held_out (reported-only) | d0002 | 0.0426025 / 0.0650403 | 1.99519 / 2.0189 | 0.0213479 / 0.0327044 |
| layer-1-attention | tiny-predecessor-lean-001:held_out (reported-only) | base | 0.0380198 / 0.0712648 | 1.9901 / 2.00829 | 0.019113 / 0.0360241 |
| layer-1-attention | tiny-predecessor-lean-001:held_out (reported-only) | d0001 | 0.0380198 / 0.0712648 | 1.9901 / 2.00829 | 0.019113 / 0.0360241 |
| layer-1-attention | tiny-predecessor-lean-001:held_out (reported-only) | d0002 | 0.0380198 / 0.0712648 | 1.9901 / 2.00829 | 0.019113 / 0.0360241 |
| layer-1-mlp | tiny-successor-lean-001:train (primary) | base | 0.00589736 / 0.00856694 | 1.99676 / 2.02436 | 0.00295255 / 0.00425786 |
| layer-1-mlp | tiny-successor-lean-001:train (primary) | d0001 | 0.00589736 / 0.00856694 | 1.99676 / 2.02436 | 0.00295255 / 0.00425786 |
| layer-1-mlp | tiny-successor-lean-001:train (primary) | d0002 | 0.00631237 / 0.00883103 | 1.99676 / 2.02436 | 0.00316025 / 0.00438912 |
| layer-1-mlp | tiny-predecessor-lean-001:train (primary) | base | 0.00560888 / 0.00693197 | 1.99666 / 2.01927 | 0.00281051 / 0.00347083 |
| layer-1-mlp | tiny-predecessor-lean-001:train (primary) | d0001 | 0.00560888 / 0.00693197 | 1.99666 / 2.01927 | 0.00281051 / 0.00347083 |
| layer-1-mlp | tiny-predecessor-lean-001:train (primary) | d0002 | 0.00604641 / 0.00797378 | 1.99666 / 2.01927 | 0.00303013 / 0.00400029 |
| layer-1-mlp | tiny-successor-lean-001:held_out (reported-only) | base | 0.00587074 / 0.00851472 | 1.9969 / 2.02687 | 0.00293884 / 0.00420092 |
| layer-1-mlp | tiny-successor-lean-001:held_out (reported-only) | d0001 | 0.00587074 / 0.00851472 | 1.9969 / 2.02687 | 0.00293884 / 0.00420092 |
| layer-1-mlp | tiny-successor-lean-001:held_out (reported-only) | d0002 | 0.00628793 / 0.00877326 | 1.9969 / 2.02687 | 0.00314829 / 0.00432848 |
| layer-1-mlp | tiny-predecessor-lean-001:held_out (reported-only) | base | 0.00563681 / 0.00698691 | 1.99187 / 2.00991 | 0.00283192 / 0.00356097 |
| layer-1-mlp | tiny-predecessor-lean-001:held_out (reported-only) | d0001 | 0.00563681 / 0.00698691 | 1.99187 / 2.00991 | 0.00283192 / 0.00356097 |
| layer-1-mlp | tiny-predecessor-lean-001:held_out (reported-only) | d0002 | 0.00608276 / 0.00804165 | 1.99187 / 2.00991 | 0.00305679 / 0.00409853 |

### Branch-contribution, all positions (measurements; reported separately — no reading rests on these)

| Branch | Population | Head | Ratio (mean / max per-example, all positions) |
|---|---|---|---|
| embedding-sum | tiny-successor-lean-001:train (primary) | base | 1 / 1 |
| embedding-sum | tiny-successor-lean-001:train (primary) | d0001 | 1 / 1 |
| embedding-sum | tiny-successor-lean-001:train (primary) | d0002 | 1 / 1 |
| embedding-sum | tiny-predecessor-lean-001:train (primary) | base | 1 / 1 |
| embedding-sum | tiny-predecessor-lean-001:train (primary) | d0001 | 1 / 1 |
| embedding-sum | tiny-predecessor-lean-001:train (primary) | d0002 | 1 / 1 |
| embedding-sum | tiny-successor-lean-001:held_out (reported-only) | base | 1 / 1 |
| embedding-sum | tiny-successor-lean-001:held_out (reported-only) | d0001 | 1 / 1 |
| embedding-sum | tiny-successor-lean-001:held_out (reported-only) | d0002 | 1 / 1 |
| embedding-sum | tiny-predecessor-lean-001:held_out (reported-only) | base | 1 / 1 |
| embedding-sum | tiny-predecessor-lean-001:held_out (reported-only) | d0001 | 1 / 1 |
| embedding-sum | tiny-predecessor-lean-001:held_out (reported-only) | d0002 | 1 / 1 |
| layer-0-attention | tiny-successor-lean-001:train (primary) | base | 0.0231824 / 0.0516502 |
| layer-0-attention | tiny-successor-lean-001:train (primary) | d0001 | 0.0231824 / 0.0516502 |
| layer-0-attention | tiny-successor-lean-001:train (primary) | d0002 | 0.0231824 / 0.0516502 |
| layer-0-attention | tiny-predecessor-lean-001:train (primary) | base | 0.022971 / 0.0493436 |
| layer-0-attention | tiny-predecessor-lean-001:train (primary) | d0001 | 0.022971 / 0.0493436 |
| layer-0-attention | tiny-predecessor-lean-001:train (primary) | d0002 | 0.022971 / 0.0493436 |
| layer-0-attention | tiny-successor-lean-001:held_out (reported-only) | base | 0.024844 / 0.0448886 |
| layer-0-attention | tiny-successor-lean-001:held_out (reported-only) | d0001 | 0.024844 / 0.0448886 |
| layer-0-attention | tiny-successor-lean-001:held_out (reported-only) | d0002 | 0.024844 / 0.0448886 |
| layer-0-attention | tiny-predecessor-lean-001:held_out (reported-only) | base | 0.0260431 / 0.043204 |
| layer-0-attention | tiny-predecessor-lean-001:held_out (reported-only) | d0001 | 0.0260431 / 0.043204 |
| layer-0-attention | tiny-predecessor-lean-001:held_out (reported-only) | d0002 | 0.0260431 / 0.043204 |
| layer-0-mlp | tiny-successor-lean-001:train (primary) | base | 0.00283517 / 0.00466391 |
| layer-0-mlp | tiny-successor-lean-001:train (primary) | d0001 | 0.00283517 / 0.00466391 |
| layer-0-mlp | tiny-successor-lean-001:train (primary) | d0002 | 0.00283517 / 0.00466391 |
| layer-0-mlp | tiny-predecessor-lean-001:train (primary) | base | 0.00293557 / 0.00462417 |
| layer-0-mlp | tiny-predecessor-lean-001:train (primary) | d0001 | 0.00293557 / 0.00462417 |
| layer-0-mlp | tiny-predecessor-lean-001:train (primary) | d0002 | 0.00293557 / 0.00462417 |
| layer-0-mlp | tiny-successor-lean-001:held_out (reported-only) | base | 0.00299238 / 0.00458885 |
| layer-0-mlp | tiny-successor-lean-001:held_out (reported-only) | d0001 | 0.00299238 / 0.00458885 |
| layer-0-mlp | tiny-successor-lean-001:held_out (reported-only) | d0002 | 0.00299238 / 0.00458885 |
| layer-0-mlp | tiny-predecessor-lean-001:held_out (reported-only) | base | 0.00280959 / 0.00458759 |
| layer-0-mlp | tiny-predecessor-lean-001:held_out (reported-only) | d0001 | 0.00280959 / 0.00458759 |
| layer-0-mlp | tiny-predecessor-lean-001:held_out (reported-only) | d0002 | 0.00280959 / 0.00458759 |
| layer-1-attention | tiny-successor-lean-001:train (primary) | base | 0.023821 / 0.0498679 |
| layer-1-attention | tiny-successor-lean-001:train (primary) | d0001 | 0.023821 / 0.0498679 |
| layer-1-attention | tiny-successor-lean-001:train (primary) | d0002 | 0.023821 / 0.0498679 |
| layer-1-attention | tiny-predecessor-lean-001:train (primary) | base | 0.0229511 / 0.0498679 |
| layer-1-attention | tiny-predecessor-lean-001:train (primary) | d0001 | 0.0229511 / 0.0498679 |
| layer-1-attention | tiny-predecessor-lean-001:train (primary) | d0002 | 0.0229511 / 0.0498679 |
| layer-1-attention | tiny-successor-lean-001:held_out (reported-only) | base | 0.0259191 / 0.0498679 |
| layer-1-attention | tiny-successor-lean-001:held_out (reported-only) | d0001 | 0.0259191 / 0.0498679 |
| layer-1-attention | tiny-successor-lean-001:held_out (reported-only) | d0002 | 0.0259191 / 0.0498679 |
| layer-1-attention | tiny-predecessor-lean-001:held_out (reported-only) | base | 0.025771 / 0.0498679 |
| layer-1-attention | tiny-predecessor-lean-001:held_out (reported-only) | d0001 | 0.025771 / 0.0498679 |
| layer-1-attention | tiny-predecessor-lean-001:held_out (reported-only) | d0002 | 0.025771 / 0.0498679 |
| layer-1-mlp | tiny-successor-lean-001:train (primary) | base | 0.00286179 / 0.00440884 |
| layer-1-mlp | tiny-successor-lean-001:train (primary) | d0001 | 0.00286179 / 0.00440884 |
| layer-1-mlp | tiny-successor-lean-001:train (primary) | d0002 | 0.00305134 / 0.00453215 |
| layer-1-mlp | tiny-predecessor-lean-001:train (primary) | base | 0.00282826 / 0.00440884 |
| layer-1-mlp | tiny-predecessor-lean-001:train (primary) | d0001 | 0.00282826 / 0.00440884 |
| layer-1-mlp | tiny-predecessor-lean-001:train (primary) | d0002 | 0.00301954 / 0.00453215 |
| layer-1-mlp | tiny-successor-lean-001:held_out (reported-only) | base | 0.00285145 / 0.00420092 |
| layer-1-mlp | tiny-successor-lean-001:held_out (reported-only) | d0001 | 0.00285145 / 0.00420092 |
| layer-1-mlp | tiny-successor-lean-001:held_out (reported-only) | d0002 | 0.0030506 / 0.00432848 |
| layer-1-mlp | tiny-predecessor-lean-001:held_out (reported-only) | base | 0.00292646 / 0.00418335 |
| layer-1-mlp | tiny-predecessor-lean-001:held_out (reported-only) | d0001 | 0.00292646 / 0.00418335 |
| layer-1-mlp | tiny-predecessor-lean-001:held_out (reported-only) | d0002 | 0.00313092 / 0.0042973 |

### Per-matrix input-activation records (measurements; last position, min / mean / max)

| Matrix | Population | Head | Input norm (min / mean / max) |
|---|---|---|---|
| model.language_model.embed_tokens.weight | tiny-successor-lean-001:train (primary) | base | not applicable — index input (see inventory note) |
| model.language_model.embed_tokens.weight | tiny-successor-lean-001:train (primary) | d0001 | not applicable — index input (see inventory note) |
| model.language_model.embed_tokens.weight | tiny-successor-lean-001:train (primary) | d0002 | not applicable — index input (see inventory note) |
| model.language_model.embed_tokens.weight | tiny-predecessor-lean-001:train (primary) | base | not applicable — index input (see inventory note) |
| model.language_model.embed_tokens.weight | tiny-predecessor-lean-001:train (primary) | d0001 | not applicable — index input (see inventory note) |
| model.language_model.embed_tokens.weight | tiny-predecessor-lean-001:train (primary) | d0002 | not applicable — index input (see inventory note) |
| model.language_model.embed_tokens.weight | tiny-successor-lean-001:held_out (reported-only) | base | not applicable — index input (see inventory note) |
| model.language_model.embed_tokens.weight | tiny-successor-lean-001:held_out (reported-only) | d0001 | not applicable — index input (see inventory note) |
| model.language_model.embed_tokens.weight | tiny-successor-lean-001:held_out (reported-only) | d0002 | not applicable — index input (see inventory note) |
| model.language_model.embed_tokens.weight | tiny-predecessor-lean-001:held_out (reported-only) | base | not applicable — index input (see inventory note) |
| model.language_model.embed_tokens.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0001 | not applicable — index input (see inventory note) |
| model.language_model.embed_tokens.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0002 | not applicable — index input (see inventory note) |
| model.language_model.embed_positions.weight | tiny-successor-lean-001:train (primary) | base | not applicable — index input (see inventory note) |
| model.language_model.embed_positions.weight | tiny-successor-lean-001:train (primary) | d0001 | not applicable — index input (see inventory note) |
| model.language_model.embed_positions.weight | tiny-successor-lean-001:train (primary) | d0002 | not applicable — index input (see inventory note) |
| model.language_model.embed_positions.weight | tiny-predecessor-lean-001:train (primary) | base | not applicable — index input (see inventory note) |
| model.language_model.embed_positions.weight | tiny-predecessor-lean-001:train (primary) | d0001 | not applicable — index input (see inventory note) |
| model.language_model.embed_positions.weight | tiny-predecessor-lean-001:train (primary) | d0002 | not applicable — index input (see inventory note) |
| model.language_model.embed_positions.weight | tiny-successor-lean-001:held_out (reported-only) | base | not applicable — index input (see inventory note) |
| model.language_model.embed_positions.weight | tiny-successor-lean-001:held_out (reported-only) | d0001 | not applicable — index input (see inventory note) |
| model.language_model.embed_positions.weight | tiny-successor-lean-001:held_out (reported-only) | d0002 | not applicable — index input (see inventory note) |
| model.language_model.embed_positions.weight | tiny-predecessor-lean-001:held_out (reported-only) | base | not applicable — index input (see inventory note) |
| model.language_model.embed_positions.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0001 | not applicable — index input (see inventory note) |
| model.language_model.embed_positions.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0002 | not applicable — index input (see inventory note) |
| lm_head.weight | tiny-successor-lean-001:train (primary) | base | 5.65662 / 5.65663 / 5.65663 |
| lm_head.weight | tiny-successor-lean-001:train (primary) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| lm_head.weight | tiny-successor-lean-001:train (primary) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| lm_head.weight | tiny-predecessor-lean-001:train (primary) | base | 5.65662 / 5.65663 / 5.65663 |
| lm_head.weight | tiny-predecessor-lean-001:train (primary) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| lm_head.weight | tiny-predecessor-lean-001:train (primary) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| lm_head.weight | tiny-successor-lean-001:held_out (reported-only) | base | 5.65662 / 5.65663 / 5.65663 |
| lm_head.weight | tiny-successor-lean-001:held_out (reported-only) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| lm_head.weight | tiny-successor-lean-001:held_out (reported-only) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| lm_head.weight | tiny-predecessor-lean-001:held_out (reported-only) | base | 5.65662 / 5.65663 / 5.65663 |
| lm_head.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| lm_head.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.q_proj.weight | tiny-successor-lean-001:train (primary) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.q_proj.weight | tiny-successor-lean-001:train (primary) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.q_proj.weight | tiny-successor-lean-001:train (primary) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.q_proj.weight | tiny-predecessor-lean-001:train (primary) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.q_proj.weight | tiny-predecessor-lean-001:train (primary) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.q_proj.weight | tiny-predecessor-lean-001:train (primary) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.q_proj.weight | tiny-successor-lean-001:held_out (reported-only) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.q_proj.weight | tiny-successor-lean-001:held_out (reported-only) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.q_proj.weight | tiny-successor-lean-001:held_out (reported-only) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.q_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.q_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.q_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.k_proj.weight | tiny-successor-lean-001:train (primary) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.k_proj.weight | tiny-successor-lean-001:train (primary) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.k_proj.weight | tiny-successor-lean-001:train (primary) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.k_proj.weight | tiny-predecessor-lean-001:train (primary) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.k_proj.weight | tiny-predecessor-lean-001:train (primary) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.k_proj.weight | tiny-predecessor-lean-001:train (primary) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.k_proj.weight | tiny-successor-lean-001:held_out (reported-only) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.k_proj.weight | tiny-successor-lean-001:held_out (reported-only) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.k_proj.weight | tiny-successor-lean-001:held_out (reported-only) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.k_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.k_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.k_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.v_proj.weight | tiny-successor-lean-001:train (primary) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.v_proj.weight | tiny-successor-lean-001:train (primary) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.v_proj.weight | tiny-successor-lean-001:train (primary) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.v_proj.weight | tiny-predecessor-lean-001:train (primary) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.v_proj.weight | tiny-predecessor-lean-001:train (primary) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.v_proj.weight | tiny-predecessor-lean-001:train (primary) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.v_proj.weight | tiny-successor-lean-001:held_out (reported-only) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.v_proj.weight | tiny-successor-lean-001:held_out (reported-only) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.v_proj.weight | tiny-successor-lean-001:held_out (reported-only) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.v_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.v_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.v_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.self_attn.o_proj.weight | tiny-successor-lean-001:train (primary) | base | 0.205579 / 0.290808 / 0.482364 |
| model.language_model.layers.0.self_attn.o_proj.weight | tiny-successor-lean-001:train (primary) | d0001 | 0.205579 / 0.290808 / 0.482364 |
| model.language_model.layers.0.self_attn.o_proj.weight | tiny-successor-lean-001:train (primary) | d0002 | 0.205579 / 0.290808 / 0.482364 |
| model.language_model.layers.0.self_attn.o_proj.weight | tiny-predecessor-lean-001:train (primary) | base | 0.194704 / 0.324403 / 0.514342 |
| model.language_model.layers.0.self_attn.o_proj.weight | tiny-predecessor-lean-001:train (primary) | d0001 | 0.194704 / 0.324403 / 0.514342 |
| model.language_model.layers.0.self_attn.o_proj.weight | tiny-predecessor-lean-001:train (primary) | d0002 | 0.194704 / 0.324403 / 0.514342 |
| model.language_model.layers.0.self_attn.o_proj.weight | tiny-successor-lean-001:held_out (reported-only) | base | 0.24772 / 0.338472 / 0.477777 |
| model.language_model.layers.0.self_attn.o_proj.weight | tiny-successor-lean-001:held_out (reported-only) | d0001 | 0.24772 / 0.338472 / 0.477777 |
| model.language_model.layers.0.self_attn.o_proj.weight | tiny-successor-lean-001:held_out (reported-only) | d0002 | 0.24772 / 0.338472 / 0.477777 |
| model.language_model.layers.0.self_attn.o_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | base | 0.19489 / 0.350865 / 0.557716 |
| model.language_model.layers.0.self_attn.o_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0001 | 0.19489 / 0.350865 / 0.557716 |
| model.language_model.layers.0.self_attn.o_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0002 | 0.19489 / 0.350865 / 0.557716 |
| model.language_model.layers.0.mlp.gate_proj.weight | tiny-successor-lean-001:train (primary) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.mlp.gate_proj.weight | tiny-successor-lean-001:train (primary) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.mlp.gate_proj.weight | tiny-successor-lean-001:train (primary) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.mlp.gate_proj.weight | tiny-predecessor-lean-001:train (primary) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.mlp.gate_proj.weight | tiny-predecessor-lean-001:train (primary) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.mlp.gate_proj.weight | tiny-predecessor-lean-001:train (primary) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.mlp.gate_proj.weight | tiny-successor-lean-001:held_out (reported-only) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.mlp.gate_proj.weight | tiny-successor-lean-001:held_out (reported-only) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.mlp.gate_proj.weight | tiny-successor-lean-001:held_out (reported-only) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.mlp.gate_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.mlp.gate_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.mlp.gate_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.mlp.up_proj.weight | tiny-successor-lean-001:train (primary) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.mlp.up_proj.weight | tiny-successor-lean-001:train (primary) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.mlp.up_proj.weight | tiny-successor-lean-001:train (primary) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.mlp.up_proj.weight | tiny-predecessor-lean-001:train (primary) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.mlp.up_proj.weight | tiny-predecessor-lean-001:train (primary) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.mlp.up_proj.weight | tiny-predecessor-lean-001:train (primary) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.mlp.up_proj.weight | tiny-successor-lean-001:held_out (reported-only) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.mlp.up_proj.weight | tiny-successor-lean-001:held_out (reported-only) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.mlp.up_proj.weight | tiny-successor-lean-001:held_out (reported-only) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.mlp.up_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.mlp.up_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.mlp.up_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.0.mlp.down_proj.weight | tiny-successor-lean-001:train (primary) | base | 0.0415839 / 0.0518285 / 0.0696454 |
| model.language_model.layers.0.mlp.down_proj.weight | tiny-successor-lean-001:train (primary) | d0001 | 0.0415839 / 0.0518285 / 0.0696454 |
| model.language_model.layers.0.mlp.down_proj.weight | tiny-successor-lean-001:train (primary) | d0002 | 0.0415839 / 0.0518285 / 0.0696454 |
| model.language_model.layers.0.mlp.down_proj.weight | tiny-predecessor-lean-001:train (primary) | base | 0.0361516 / 0.0479358 / 0.0639994 |
| model.language_model.layers.0.mlp.down_proj.weight | tiny-predecessor-lean-001:train (primary) | d0001 | 0.0361516 / 0.0479358 / 0.0639994 |
| model.language_model.layers.0.mlp.down_proj.weight | tiny-predecessor-lean-001:train (primary) | d0002 | 0.0361516 / 0.0479358 / 0.0639994 |
| model.language_model.layers.0.mlp.down_proj.weight | tiny-successor-lean-001:held_out (reported-only) | base | 0.0412932 / 0.0515944 / 0.067344 |
| model.language_model.layers.0.mlp.down_proj.weight | tiny-successor-lean-001:held_out (reported-only) | d0001 | 0.0412932 / 0.0515944 / 0.067344 |
| model.language_model.layers.0.mlp.down_proj.weight | tiny-successor-lean-001:held_out (reported-only) | d0002 | 0.0412932 / 0.0515944 / 0.067344 |
| model.language_model.layers.0.mlp.down_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | base | 0.0362972 / 0.0480065 / 0.0640699 |
| model.language_model.layers.0.mlp.down_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0001 | 0.0362972 / 0.0480065 / 0.0640699 |
| model.language_model.layers.0.mlp.down_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0002 | 0.0362972 / 0.0480065 / 0.0640699 |
| model.language_model.layers.1.self_attn.q_proj.weight | tiny-successor-lean-001:train (primary) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.q_proj.weight | tiny-successor-lean-001:train (primary) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.q_proj.weight | tiny-successor-lean-001:train (primary) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.q_proj.weight | tiny-predecessor-lean-001:train (primary) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.q_proj.weight | tiny-predecessor-lean-001:train (primary) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.q_proj.weight | tiny-predecessor-lean-001:train (primary) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.q_proj.weight | tiny-successor-lean-001:held_out (reported-only) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.q_proj.weight | tiny-successor-lean-001:held_out (reported-only) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.q_proj.weight | tiny-successor-lean-001:held_out (reported-only) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.q_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.q_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.q_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.k_proj.weight | tiny-successor-lean-001:train (primary) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.k_proj.weight | tiny-successor-lean-001:train (primary) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.k_proj.weight | tiny-successor-lean-001:train (primary) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.k_proj.weight | tiny-predecessor-lean-001:train (primary) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.k_proj.weight | tiny-predecessor-lean-001:train (primary) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.k_proj.weight | tiny-predecessor-lean-001:train (primary) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.k_proj.weight | tiny-successor-lean-001:held_out (reported-only) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.k_proj.weight | tiny-successor-lean-001:held_out (reported-only) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.k_proj.weight | tiny-successor-lean-001:held_out (reported-only) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.k_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.k_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.k_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.v_proj.weight | tiny-successor-lean-001:train (primary) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.v_proj.weight | tiny-successor-lean-001:train (primary) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.v_proj.weight | tiny-successor-lean-001:train (primary) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.v_proj.weight | tiny-predecessor-lean-001:train (primary) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.v_proj.weight | tiny-predecessor-lean-001:train (primary) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.v_proj.weight | tiny-predecessor-lean-001:train (primary) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.v_proj.weight | tiny-successor-lean-001:held_out (reported-only) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.v_proj.weight | tiny-successor-lean-001:held_out (reported-only) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.v_proj.weight | tiny-successor-lean-001:held_out (reported-only) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.v_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.v_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.v_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.self_attn.o_proj.weight | tiny-successor-lean-001:train (primary) | base | 0.199521 / 0.320146 / 0.514963 |
| model.language_model.layers.1.self_attn.o_proj.weight | tiny-successor-lean-001:train (primary) | d0001 | 0.199521 / 0.320146 / 0.514963 |
| model.language_model.layers.1.self_attn.o_proj.weight | tiny-successor-lean-001:train (primary) | d0002 | 0.199521 / 0.320146 / 0.514963 |
| model.language_model.layers.1.self_attn.o_proj.weight | tiny-predecessor-lean-001:train (primary) | base | 0.185547 / 0.325393 / 0.669848 |
| model.language_model.layers.1.self_attn.o_proj.weight | tiny-predecessor-lean-001:train (primary) | d0001 | 0.185547 / 0.325393 / 0.669848 |
| model.language_model.layers.1.self_attn.o_proj.weight | tiny-predecessor-lean-001:train (primary) | d0002 | 0.185547 / 0.325393 / 0.669848 |
| model.language_model.layers.1.self_attn.o_proj.weight | tiny-successor-lean-001:held_out (reported-only) | base | 0.186699 / 0.352506 / 0.545968 |
| model.language_model.layers.1.self_attn.o_proj.weight | tiny-successor-lean-001:held_out (reported-only) | d0001 | 0.186699 / 0.352506 / 0.545968 |
| model.language_model.layers.1.self_attn.o_proj.weight | tiny-successor-lean-001:held_out (reported-only) | d0002 | 0.186699 / 0.352506 / 0.545968 |
| model.language_model.layers.1.self_attn.o_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | base | 0.237783 / 0.331198 / 0.493032 |
| model.language_model.layers.1.self_attn.o_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0001 | 0.237783 / 0.331198 / 0.493032 |
| model.language_model.layers.1.self_attn.o_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0002 | 0.237783 / 0.331198 / 0.493032 |
| model.language_model.layers.1.mlp.gate_proj.weight | tiny-successor-lean-001:train (primary) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.mlp.gate_proj.weight | tiny-successor-lean-001:train (primary) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.mlp.gate_proj.weight | tiny-successor-lean-001:train (primary) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.mlp.gate_proj.weight | tiny-predecessor-lean-001:train (primary) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.mlp.gate_proj.weight | tiny-predecessor-lean-001:train (primary) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.mlp.gate_proj.weight | tiny-predecessor-lean-001:train (primary) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.mlp.gate_proj.weight | tiny-successor-lean-001:held_out (reported-only) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.mlp.gate_proj.weight | tiny-successor-lean-001:held_out (reported-only) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.mlp.gate_proj.weight | tiny-successor-lean-001:held_out (reported-only) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.mlp.gate_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.mlp.gate_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.mlp.gate_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.mlp.up_proj.weight | tiny-successor-lean-001:train (primary) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.mlp.up_proj.weight | tiny-successor-lean-001:train (primary) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.mlp.up_proj.weight | tiny-successor-lean-001:train (primary) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.mlp.up_proj.weight | tiny-predecessor-lean-001:train (primary) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.mlp.up_proj.weight | tiny-predecessor-lean-001:train (primary) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.mlp.up_proj.weight | tiny-predecessor-lean-001:train (primary) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.mlp.up_proj.weight | tiny-successor-lean-001:held_out (reported-only) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.mlp.up_proj.weight | tiny-successor-lean-001:held_out (reported-only) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.mlp.up_proj.weight | tiny-successor-lean-001:held_out (reported-only) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.mlp.up_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | base | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.mlp.up_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0001 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.mlp.up_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0002 | 5.65662 / 5.65663 / 5.65663 |
| model.language_model.layers.1.mlp.down_proj.weight | tiny-successor-lean-001:train (primary) | base | 0.0396724 / 0.0508623 / 0.0612513 |
| model.language_model.layers.1.mlp.down_proj.weight | tiny-successor-lean-001:train (primary) | d0001 | 0.0396724 / 0.0508623 / 0.0612513 |
| model.language_model.layers.1.mlp.down_proj.weight | tiny-successor-lean-001:train (primary) | d0002 | 0.0396724 / 0.0508623 / 0.0612513 |
| model.language_model.layers.1.mlp.down_proj.weight | tiny-predecessor-lean-001:train (primary) | base | 0.0400818 / 0.0503698 / 0.064056 |
| model.language_model.layers.1.mlp.down_proj.weight | tiny-predecessor-lean-001:train (primary) | d0001 | 0.0400818 / 0.0503698 / 0.064056 |
| model.language_model.layers.1.mlp.down_proj.weight | tiny-predecessor-lean-001:train (primary) | d0002 | 0.0400818 / 0.0503698 / 0.064056 |
| model.language_model.layers.1.mlp.down_proj.weight | tiny-successor-lean-001:held_out (reported-only) | base | 0.0402068 / 0.0508616 / 0.0610376 |
| model.language_model.layers.1.mlp.down_proj.weight | tiny-successor-lean-001:held_out (reported-only) | d0001 | 0.0402068 / 0.0508616 / 0.0610376 |
| model.language_model.layers.1.mlp.down_proj.weight | tiny-successor-lean-001:held_out (reported-only) | d0002 | 0.0402068 / 0.0508616 / 0.0610376 |
| model.language_model.layers.1.mlp.down_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | base | 0.0402894 / 0.0505437 / 0.0628508 |
| model.language_model.layers.1.mlp.down_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0001 | 0.0402894 / 0.0505437 / 0.0628508 |
| model.language_model.layers.1.mlp.down_proj.weight | tiny-predecessor-lean-001:held_out (reported-only) | d0002 | 0.0402894 / 0.0505437 / 0.0628508 |

### Expressibility bounds (derivations; B(W) = 0.9 × max input norm over the named primary population at the named head)

| Matrix | Population (primary) | Head | Max input norm | B(W) | Downstream transfer |
|---|---|---|---|---|---|
| model.language_model.layers.0.self_attn.q_proj.weight | tiny-successor-lean-001:train | base | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.self_attn.q_proj.weight | tiny-successor-lean-001:train | d0001 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.self_attn.q_proj.weight | tiny-successor-lean-001:train | d0002 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.self_attn.q_proj.weight | tiny-predecessor-lean-001:train | base | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.self_attn.q_proj.weight | tiny-predecessor-lean-001:train | d0001 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.self_attn.q_proj.weight | tiny-predecessor-lean-001:train | d0002 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.self_attn.k_proj.weight | tiny-successor-lean-001:train | base | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.self_attn.k_proj.weight | tiny-successor-lean-001:train | d0001 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.self_attn.k_proj.weight | tiny-successor-lean-001:train | d0002 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.self_attn.k_proj.weight | tiny-predecessor-lean-001:train | base | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.self_attn.k_proj.weight | tiny-predecessor-lean-001:train | d0001 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.self_attn.k_proj.weight | tiny-predecessor-lean-001:train | d0002 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.self_attn.v_proj.weight | tiny-successor-lean-001:train | base | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.self_attn.v_proj.weight | tiny-successor-lean-001:train | d0001 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.self_attn.v_proj.weight | tiny-successor-lean-001:train | d0002 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.self_attn.v_proj.weight | tiny-predecessor-lean-001:train | base | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.self_attn.v_proj.weight | tiny-predecessor-lean-001:train | d0001 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.self_attn.v_proj.weight | tiny-predecessor-lean-001:train | d0002 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.self_attn.o_proj.weight | tiny-successor-lean-001:train | base | 0.482364 | 0.434128 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.self_attn.o_proj.weight | tiny-successor-lean-001:train | d0001 | 0.482364 | 0.434128 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.self_attn.o_proj.weight | tiny-successor-lean-001:train | d0002 | 0.482364 | 0.434128 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.self_attn.o_proj.weight | tiny-predecessor-lean-001:train | base | 0.514342 | 0.462908 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.self_attn.o_proj.weight | tiny-predecessor-lean-001:train | d0001 | 0.514342 | 0.462908 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.self_attn.o_proj.weight | tiny-predecessor-lean-001:train | d0002 | 0.514342 | 0.462908 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.mlp.gate_proj.weight | tiny-successor-lean-001:train | base | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.mlp.gate_proj.weight | tiny-successor-lean-001:train | d0001 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.mlp.gate_proj.weight | tiny-successor-lean-001:train | d0002 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.mlp.gate_proj.weight | tiny-predecessor-lean-001:train | base | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.mlp.gate_proj.weight | tiny-predecessor-lean-001:train | d0001 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.mlp.gate_proj.weight | tiny-predecessor-lean-001:train | d0002 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.mlp.up_proj.weight | tiny-successor-lean-001:train | base | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.mlp.up_proj.weight | tiny-successor-lean-001:train | d0001 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.mlp.up_proj.weight | tiny-successor-lean-001:train | d0002 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.mlp.up_proj.weight | tiny-predecessor-lean-001:train | base | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.mlp.up_proj.weight | tiny-predecessor-lean-001:train | d0001 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.mlp.up_proj.weight | tiny-predecessor-lean-001:train | d0002 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.mlp.down_proj.weight | tiny-successor-lean-001:train | base | 0.0696454 | 0.0626808 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.mlp.down_proj.weight | tiny-successor-lean-001:train | d0001 | 0.0696454 | 0.0626808 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.mlp.down_proj.weight | tiny-successor-lean-001:train | d0002 | 0.0696454 | 0.0626808 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.mlp.down_proj.weight | tiny-predecessor-lean-001:train | base | 0.0639994 | 0.0575995 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.mlp.down_proj.weight | tiny-predecessor-lean-001:train | d0001 | 0.0639994 | 0.0575995 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.0.mlp.down_proj.weight | tiny-predecessor-lean-001:train | d0002 | 0.0639994 | 0.0575995 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.self_attn.q_proj.weight | tiny-successor-lean-001:train | base | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.self_attn.q_proj.weight | tiny-successor-lean-001:train | d0001 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.self_attn.q_proj.weight | tiny-successor-lean-001:train | d0002 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.self_attn.q_proj.weight | tiny-predecessor-lean-001:train | base | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.self_attn.q_proj.weight | tiny-predecessor-lean-001:train | d0001 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.self_attn.q_proj.weight | tiny-predecessor-lean-001:train | d0002 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.self_attn.k_proj.weight | tiny-successor-lean-001:train | base | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.self_attn.k_proj.weight | tiny-successor-lean-001:train | d0001 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.self_attn.k_proj.weight | tiny-successor-lean-001:train | d0002 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.self_attn.k_proj.weight | tiny-predecessor-lean-001:train | base | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.self_attn.k_proj.weight | tiny-predecessor-lean-001:train | d0001 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.self_attn.k_proj.weight | tiny-predecessor-lean-001:train | d0002 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.self_attn.v_proj.weight | tiny-successor-lean-001:train | base | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.self_attn.v_proj.weight | tiny-successor-lean-001:train | d0001 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.self_attn.v_proj.weight | tiny-successor-lean-001:train | d0002 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.self_attn.v_proj.weight | tiny-predecessor-lean-001:train | base | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.self_attn.v_proj.weight | tiny-predecessor-lean-001:train | d0001 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.self_attn.v_proj.weight | tiny-predecessor-lean-001:train | d0002 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.self_attn.o_proj.weight | tiny-successor-lean-001:train | base | 0.514963 | 0.463467 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.self_attn.o_proj.weight | tiny-successor-lean-001:train | d0001 | 0.514963 | 0.463467 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.self_attn.o_proj.weight | tiny-successor-lean-001:train | d0002 | 0.514963 | 0.463467 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.self_attn.o_proj.weight | tiny-predecessor-lean-001:train | base | 0.669848 | 0.602863 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.self_attn.o_proj.weight | tiny-predecessor-lean-001:train | d0001 | 0.669848 | 0.602863 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.self_attn.o_proj.weight | tiny-predecessor-lean-001:train | d0002 | 0.669848 | 0.602863 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.mlp.gate_proj.weight | tiny-successor-lean-001:train | base | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.mlp.gate_proj.weight | tiny-successor-lean-001:train | d0001 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.mlp.gate_proj.weight | tiny-successor-lean-001:train | d0002 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.mlp.gate_proj.weight | tiny-predecessor-lean-001:train | base | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.mlp.gate_proj.weight | tiny-predecessor-lean-001:train | d0001 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.mlp.gate_proj.weight | tiny-predecessor-lean-001:train | d0002 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.mlp.up_proj.weight | tiny-successor-lean-001:train | base | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.mlp.up_proj.weight | tiny-successor-lean-001:train | d0001 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.mlp.up_proj.weight | tiny-successor-lean-001:train | d0002 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.mlp.up_proj.weight | tiny-predecessor-lean-001:train | base | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.mlp.up_proj.weight | tiny-predecessor-lean-001:train | d0001 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.mlp.up_proj.weight | tiny-predecessor-lean-001:train | d0002 | 5.65663 | 5.09097 | unknown — the audit measures no transfer coefficient for this tensor |
| model.language_model.layers.1.mlp.down_proj.weight | tiny-successor-lean-001:train | base | 0.0612513 | 0.0551261 | measured (SI-33's landed delta record) |
| model.language_model.layers.1.mlp.down_proj.weight | tiny-successor-lean-001:train | d0001 | 0.0612513 | 0.0551261 | measured (SI-33's landed delta record) |
| model.language_model.layers.1.mlp.down_proj.weight | tiny-successor-lean-001:train | d0002 | 0.0612513 | 0.0551261 | measured (SI-33's landed delta record) |
| model.language_model.layers.1.mlp.down_proj.weight | tiny-predecessor-lean-001:train | base | 0.064056 | 0.0576504 | measured (SI-33's landed delta record) |
| model.language_model.layers.1.mlp.down_proj.weight | tiny-predecessor-lean-001:train | d0001 | 0.064056 | 0.0576504 | measured (SI-33's landed delta record) |
| model.language_model.layers.1.mlp.down_proj.weight | tiny-predecessor-lean-001:train | d0002 | 0.064056 | 0.0576504 | measured (SI-33's landed delta record) |

### Backward-path and checkability inventory (definitions and arithmetic, labelled)

| Matrix | Shape | Chain count | Chain elements (downstream order) | Rank-8 factor entries | Blind | Gate-measurability | Downstream transfer |
|---|---|---|---|---|---|---|---|
| model.language_model.layers.0.self_attn.q_proj.weight | [32, 32] | 17 | layer-0 attention Jacobian → layer-0 o_proj matmul → layer-0 attention residual add → layer-0 post-attention RMSNorm → layer-0 SwiGLU derivative → layer-0 down_proj matmul → layer-0 MLP residual add → layer-1 input RMSNorm → layer-1 attention Jacobian → layer-1 o_proj matmul → layer-1 attention residual add → layer-1 post-attention RMSNorm → layer-1 SwiGLU derivative → layer-1 down_proj matmul → layer-1 MLP residual add → final RMSNorm → lm_head matmul (readout) | 512 | False | measurable under SI-17's mapping (not in the derived blind set) | unknown |
| model.language_model.layers.0.self_attn.k_proj.weight | [32, 32] | 17 | layer-0 attention Jacobian → layer-0 o_proj matmul → layer-0 attention residual add → layer-0 post-attention RMSNorm → layer-0 SwiGLU derivative → layer-0 down_proj matmul → layer-0 MLP residual add → layer-1 input RMSNorm → layer-1 attention Jacobian → layer-1 o_proj matmul → layer-1 attention residual add → layer-1 post-attention RMSNorm → layer-1 SwiGLU derivative → layer-1 down_proj matmul → layer-1 MLP residual add → final RMSNorm → lm_head matmul (readout) | 512 | False | measurable under SI-17's mapping (not in the derived blind set) | unknown |
| model.language_model.layers.0.self_attn.v_proj.weight | [32, 32] | 17 | layer-0 attention Jacobian → layer-0 o_proj matmul → layer-0 attention residual add → layer-0 post-attention RMSNorm → layer-0 SwiGLU derivative → layer-0 down_proj matmul → layer-0 MLP residual add → layer-1 input RMSNorm → layer-1 attention Jacobian → layer-1 o_proj matmul → layer-1 attention residual add → layer-1 post-attention RMSNorm → layer-1 SwiGLU derivative → layer-1 down_proj matmul → layer-1 MLP residual add → final RMSNorm → lm_head matmul (readout) | 512 | False | measurable under SI-17's mapping (not in the derived blind set) | unknown |
| model.language_model.layers.0.self_attn.o_proj.weight | [32, 32] | 15 | layer-0 attention residual add → layer-0 post-attention RMSNorm → layer-0 SwiGLU derivative → layer-0 down_proj matmul → layer-0 MLP residual add → layer-1 input RMSNorm → layer-1 attention Jacobian → layer-1 o_proj matmul → layer-1 attention residual add → layer-1 post-attention RMSNorm → layer-1 SwiGLU derivative → layer-1 down_proj matmul → layer-1 MLP residual add → final RMSNorm → lm_head matmul (readout) | 512 | False | measurable under SI-17's mapping (not in the derived blind set) | unknown |
| model.language_model.layers.0.mlp.gate_proj.weight | [64, 32] | 13 | layer-0 SwiGLU derivative → layer-0 down_proj matmul → layer-0 MLP residual add → layer-1 input RMSNorm → layer-1 attention Jacobian → layer-1 o_proj matmul → layer-1 attention residual add → layer-1 post-attention RMSNorm → layer-1 SwiGLU derivative → layer-1 down_proj matmul → layer-1 MLP residual add → final RMSNorm → lm_head matmul (readout) | 768 | False | measurable under SI-17's mapping (not in the derived blind set) | unknown |
| model.language_model.layers.0.mlp.up_proj.weight | [64, 32] | 13 | layer-0 SwiGLU derivative → layer-0 down_proj matmul → layer-0 MLP residual add → layer-1 input RMSNorm → layer-1 attention Jacobian → layer-1 o_proj matmul → layer-1 attention residual add → layer-1 post-attention RMSNorm → layer-1 SwiGLU derivative → layer-1 down_proj matmul → layer-1 MLP residual add → final RMSNorm → lm_head matmul (readout) | 768 | False | measurable under SI-17's mapping (not in the derived blind set) | unknown |
| model.language_model.layers.0.mlp.down_proj.weight | [32, 64] | 11 | layer-0 MLP residual add → layer-1 input RMSNorm → layer-1 attention Jacobian → layer-1 o_proj matmul → layer-1 attention residual add → layer-1 post-attention RMSNorm → layer-1 SwiGLU derivative → layer-1 down_proj matmul → layer-1 MLP residual add → final RMSNorm → lm_head matmul (readout) | 768 | False | measurable under SI-17's mapping (not in the derived blind set) | unknown |
| model.language_model.layers.1.self_attn.q_proj.weight | [32, 32] | 9 | layer-1 attention Jacobian → layer-1 o_proj matmul → layer-1 attention residual add → layer-1 post-attention RMSNorm → layer-1 SwiGLU derivative → layer-1 down_proj matmul → layer-1 MLP residual add → final RMSNorm → lm_head matmul (readout) | 512 | False | measurable under SI-17's mapping (not in the derived blind set) | unknown |
| model.language_model.layers.1.self_attn.k_proj.weight | [32, 32] | 9 | layer-1 attention Jacobian → layer-1 o_proj matmul → layer-1 attention residual add → layer-1 post-attention RMSNorm → layer-1 SwiGLU derivative → layer-1 down_proj matmul → layer-1 MLP residual add → final RMSNorm → lm_head matmul (readout) | 512 | False | measurable under SI-17's mapping (not in the derived blind set) | unknown |
| model.language_model.layers.1.self_attn.v_proj.weight | [32, 32] | 9 | layer-1 attention Jacobian → layer-1 o_proj matmul → layer-1 attention residual add → layer-1 post-attention RMSNorm → layer-1 SwiGLU derivative → layer-1 down_proj matmul → layer-1 MLP residual add → final RMSNorm → lm_head matmul (readout) | 512 | False | measurable under SI-17's mapping (not in the derived blind set) | unknown |
| model.language_model.layers.1.self_attn.o_proj.weight | [32, 32] | 7 | layer-1 attention residual add → layer-1 post-attention RMSNorm → layer-1 SwiGLU derivative → layer-1 down_proj matmul → layer-1 MLP residual add → final RMSNorm → lm_head matmul (readout) | 512 | False | measurable under SI-17's mapping (not in the derived blind set) | unknown |
| model.language_model.layers.1.mlp.gate_proj.weight | [64, 32] | 5 | layer-1 SwiGLU derivative → layer-1 down_proj matmul → layer-1 MLP residual add → final RMSNorm → lm_head matmul (readout) | 768 | False | measurable under SI-17's mapping (not in the derived blind set) | unknown |
| model.language_model.layers.1.mlp.up_proj.weight | [64, 32] | 5 | layer-1 SwiGLU derivative → layer-1 down_proj matmul → layer-1 MLP residual add → final RMSNorm → lm_head matmul (readout) | 768 | False | measurable under SI-17's mapping (not in the derived blind set) | unknown |
| model.language_model.layers.1.mlp.down_proj.weight | [32, 64] | 3 | layer-1 MLP residual add → final RMSNorm → lm_head matmul (readout) | 768 | False | measurable under SI-17's mapping (not in the derived blind set) | measured (SI-33's landed delta record: candidate norm 0.30801411790830574, probe-space movement norm 0.010469039280455053) |

### Qualification table — Primary cell (signal 2 train @ d0001)

| Matrix | Branch | Branch ratio (max per-example) | Ratio clause (≥ 0.04) | B(W) | Bound clause (≥ 0.212) | Qualifies |
|---|---|---|---|---|---|---|
| model.language_model.layers.0.self_attn.q_proj.weight | layer-0-attention | 0.0303477 | False | 5.09097 | True | False |
| model.language_model.layers.0.self_attn.k_proj.weight | layer-0-attention | 0.0303477 | False | 5.09097 | True | False |
| model.language_model.layers.0.self_attn.v_proj.weight | layer-0-attention | 0.0303477 | False | 5.09097 | True | False |
| model.language_model.layers.0.self_attn.o_proj.weight | layer-0-attention | 0.0303477 | False | 0.462908 | True | False |
| model.language_model.layers.0.mlp.gate_proj.weight | layer-0-mlp | 0.00329234 | False | 5.09097 | True | False |
| model.language_model.layers.0.mlp.up_proj.weight | layer-0-mlp | 0.00329234 | False | 5.09097 | True | False |
| model.language_model.layers.0.mlp.down_proj.weight | layer-0-mlp | 0.00329234 | False | 0.0575995 | False | False |
| model.language_model.layers.1.self_attn.q_proj.weight | layer-1-attention | 0.0340859 | False | 5.09097 | True | False |
| model.language_model.layers.1.self_attn.k_proj.weight | layer-1-attention | 0.0340859 | False | 5.09097 | True | False |
| model.language_model.layers.1.self_attn.v_proj.weight | layer-1-attention | 0.0340859 | False | 5.09097 | True | False |
| model.language_model.layers.1.self_attn.o_proj.weight | layer-1-attention | 0.0340859 | False | 0.602863 | True | False |
| model.language_model.layers.1.mlp.gate_proj.weight | layer-1-mlp | 0.00347083 | False | 5.09097 | True | False |
| model.language_model.layers.1.mlp.up_proj.weight | layer-1-mlp | 0.00347083 | False | 5.09097 | True | False |
| model.language_model.layers.1.mlp.down_proj.weight | layer-1-mlp | 0.00347083 | False | 0.0576504 | False | False |

### Qualification table — Base cell (signal 2 train @ base)

| Matrix | Branch | Branch ratio (max per-example) | Ratio clause (≥ 0.04) | B(W) | Bound clause (≥ 0.212) | Qualifies |
|---|---|---|---|---|---|---|
| model.language_model.layers.0.self_attn.q_proj.weight | layer-0-attention | 0.0303477 | False | 5.09097 | True | False |
| model.language_model.layers.0.self_attn.k_proj.weight | layer-0-attention | 0.0303477 | False | 5.09097 | True | False |
| model.language_model.layers.0.self_attn.v_proj.weight | layer-0-attention | 0.0303477 | False | 5.09097 | True | False |
| model.language_model.layers.0.self_attn.o_proj.weight | layer-0-attention | 0.0303477 | False | 0.462908 | True | False |
| model.language_model.layers.0.mlp.gate_proj.weight | layer-0-mlp | 0.00329234 | False | 5.09097 | True | False |
| model.language_model.layers.0.mlp.up_proj.weight | layer-0-mlp | 0.00329234 | False | 5.09097 | True | False |
| model.language_model.layers.0.mlp.down_proj.weight | layer-0-mlp | 0.00329234 | False | 0.0575995 | False | False |
| model.language_model.layers.1.self_attn.q_proj.weight | layer-1-attention | 0.0340859 | False | 5.09097 | True | False |
| model.language_model.layers.1.self_attn.k_proj.weight | layer-1-attention | 0.0340859 | False | 5.09097 | True | False |
| model.language_model.layers.1.self_attn.v_proj.weight | layer-1-attention | 0.0340859 | False | 5.09097 | True | False |
| model.language_model.layers.1.self_attn.o_proj.weight | layer-1-attention | 0.0340859 | False | 0.602863 | True | False |
| model.language_model.layers.1.mlp.gate_proj.weight | layer-1-mlp | 0.00347083 | False | 5.09097 | True | False |
| model.language_model.layers.1.mlp.up_proj.weight | layer-1-mlp | 0.00347083 | False | 5.09097 | True | False |
| model.language_model.layers.1.mlp.down_proj.weight | layer-1-mlp | 0.00347083 | False | 0.0576504 | False | False |

### Qualification table — d0002 cell (signal 2 train @ d0002)

| Matrix | Branch | Branch ratio (max per-example) | Ratio clause (≥ 0.04) | B(W) | Bound clause (≥ 0.212) | Qualifies |
|---|---|---|---|---|---|---|
| model.language_model.layers.0.self_attn.q_proj.weight | layer-0-attention | 0.0303477 | False | 5.09097 | True | False |
| model.language_model.layers.0.self_attn.k_proj.weight | layer-0-attention | 0.0303477 | False | 5.09097 | True | False |
| model.language_model.layers.0.self_attn.v_proj.weight | layer-0-attention | 0.0303477 | False | 5.09097 | True | False |
| model.language_model.layers.0.self_attn.o_proj.weight | layer-0-attention | 0.0303477 | False | 0.462908 | True | False |
| model.language_model.layers.0.mlp.gate_proj.weight | layer-0-mlp | 0.00329234 | False | 5.09097 | True | False |
| model.language_model.layers.0.mlp.up_proj.weight | layer-0-mlp | 0.00329234 | False | 5.09097 | True | False |
| model.language_model.layers.0.mlp.down_proj.weight | layer-0-mlp | 0.00329234 | False | 0.0575995 | False | False |
| model.language_model.layers.1.self_attn.q_proj.weight | layer-1-attention | 0.0340859 | False | 5.09097 | True | False |
| model.language_model.layers.1.self_attn.k_proj.weight | layer-1-attention | 0.0340859 | False | 5.09097 | True | False |
| model.language_model.layers.1.self_attn.v_proj.weight | layer-1-attention | 0.0340859 | False | 5.09097 | True | False |
| model.language_model.layers.1.self_attn.o_proj.weight | layer-1-attention | 0.0340859 | False | 0.602863 | True | False |
| model.language_model.layers.1.mlp.gate_proj.weight | layer-1-mlp | 0.00400029 | False | 5.09097 | True | False |
| model.language_model.layers.1.mlp.up_proj.weight | layer-1-mlp | 0.00400029 | False | 5.09097 | True | False |
| model.language_model.layers.1.mlp.down_proj.weight | layer-1-mlp | 0.00400029 | False | 0.0576504 | False | False |
## The pre-registered readings, applied as written (§15 item 7)

The primary cell for every reading is signal 2's train
partition at head `d0001` — the decisive configuration's
population and head. R1's pair, registered in RFC SI-34
before any audit figure existed: (i) the matrix's
branch's contribution ratio (max per-example, last
position) at least 0.04 — stated there as ≈10× the final
MLP branch's measured ~0.4% headline (0.0074 / 2.00 =
0.0037, a derivation from SI-33's cited figures; the
multiplier was the RFC's choice, in advance); and (ii)
its local bound B(W) ≥ 0.212, the signal correction's
norm (0.15·√2, a derivation). The qualification tables
above publish the deciding facts in full for the primary
cell and, for R3, for the `base` and `d0002` cells on
the same population.

- **R1 — a less-inert single target is arguable: NOT
  measured.** No in-stream non-readout matrix satisfies
  both clauses on the primary cell. The branch-ratio
  clause fails for all 14 matrices: the attention
  branches measure max ratios 0.0303 (layer 0) and
  0.0341 (layer 1) — the closest approach to the
  registered 0.04 anywhere in the model — and the MLP
  branches 0.00329 (layer 0) and 0.00347 (layer 1), an
  order below it. The bound clause passes for 12 of 14
  (every matrix whose consumed activation is a normed
  stream or the concatenated attention output: bounds
  0.463–5.091) and fails for exactly the two
  down-projections, whose SwiGLU inputs are the model's
  smallest activations (bounds 0.0576 and 0.0577, the
  latter beside SI-33's cited yardstick ≈ 0.058 and
  beside the SI-33 candidate's own norm
  0.30801411790830574 and ‖m2‖ 0.010469039280455053,
  criterion 6). The shortlist is **empty**.
- **R2 — inertness is uniform at the standing budgets:
  MEASURED.** R1 and R2 are complementary by
  construction, and exactly one of them is measured, as
  registered. R2's consequence, quoted with its scope
  attached (RFC SI-34 §19 item 4 names the misreading
  this guards against): the single-target
  upstream-authorship line is *expressibility-closed on
  this miniature at the standing budgets* — scoped to
  single-target, upstream, rank-8, budget-0.9 candidates
  on this constructed miniature for this signal family's
  correction scale (0.212), and stated as the
  derivation-scoped conclusion from measured inputs
  that it is. It is **not** a claim that authorship is
  impossible: the readout target already falsifies any
  unqualified version (the SI-25 control expresses the
  correction fully; it is refused on composition, not
  expressibility), and it is not a gate verdict about
  any candidate (§18 item 27). The next frontier must
  be argued from a different premise — §8's squash
  trigger, §9's semantics, or §4's operator decision —
  in a future RFC (SI-36) this slice does not pre-write.
- **R3 — qualification is head-dependent: NOT
  measured.** The set of matrices satisfying R1's pair
  on signal 2's train partition is empty at head `base`
  and empty at head `d0001` (and, published beside them,
  empty at `d0002`): the sets do not differ, so R3 does
  not fire. Under its registered consequence-if-not-
  measured, qualification is head-stable *across the
  three measured heads only* — a claim scoped to those
  heads, never to depth in general. (The underlying
  figures are not head-identical — the final MLP
  branch's ratio moves 0.00347 → 0.00400 at `d0002`,
  where the landed down-projection delta acts — but no
  figure crosses a registered threshold at any head, and
  R3 is a reading about qualification sets, not about
  figures.)

## Acceptance criteria (RFC SI-34 §16) — status

1. **Coverage, complete and inventoried — PASS.** All
   30 tensors appear in the audit's record: the 17
   weight matrices tabled with shapes, the 14 in-stream
   matrices carried through every figure (branch, input,
   bound, inventory), the readout and the two embedding
   matrices tabled under their own labels (the
   embeddings' bound rows state the row-selection form
   of the bound — a derivation, labelled: an update's
   effect enters through the single selected row, so
   the movement is at most the update's Frobenius norm,
   itself at most the standing budget 0.9), and the 13
   vector tensors inventoried with blind-set status.
   The "not applicable" rows carry their reasons.
2. **Populations and probe exclusion, asserted —
   PASS.** The audit's input sets are exactly the named
   partitions (asserted by test); a
   perturbed-probe-artifact bundle copy produces a
   byte-identical audit record (asserted by test, in
   the prior slices' shape). No probe input, probe
   logit, held-out sequence from either probe suite, or
   probe-derived quantity enters any figure, ranking,
   or reading.
3. **Reproduction of SI-33's attribution — PASS**, as
   published above: the ranges reproduce under the
   audit's method, and the base representative figures
   are reconciled in print with the method difference
   named, before any non-yardstick figure is presented
   as trusted. The stop rule did not trigger.
4. **Measurements and derivations, labelled — PASS.**
   Branch ratios and input norms are published as
   measurements with population, head, and
   position-statistic on every figure; bounds are
   published as derivations (formula, budget 0.9, and
   yardsticks 0.15·√2 ≈ 0.212 and ≈ 0.058 stated);
   downstream transfer is recorded as measured only for
   the readout and the final down-projection and as
   **unknown** for every other tensor.
5. **Readings applied as registered — PASS.** R1/R2/R3
   are applied under §15 item 7's rules exactly as
   written (primary cell, thresholds 0.04 and 0.212,
   and populations unchanged after the figures exist);
   exactly one of R1/R2 is reported measured (R2), with
   the qualification table that decides it published in
   full; R3's base-vs-`d0001` comparison is published
   either way (both sets empty), with the `d0002` cell
   beside them. No reading is rounded, re-thresholded,
   or reported without its registered qualifier.
6. **Magnitude accounting — PASS.** Wherever an audit
   figure is compared against a landed candidate's
   record, the candidate's norm and ‖m2‖ stand beside
   it (SI-33: norm 0.30801411790830574, ‖m2‖
   0.010469039280455053; SI-25 control: norm
   0.2515524481203623, ‖m2‖ 0.7816722548913912) — in the
   record's magnitude-accounting block, beside the
   down-projection bounds above, and in this doc's scope
   and readings sections.
7. **No side effects, asserted — PASS.** An audit run
   writes no blob, performs no gate evaluation, appends
   no journal entry, and changes no manifest: bundle
   state is snapshot-asserted byte-identical
   before/after by test, in the shape of SI-12's
   unchanged-bundle tests; spies assert the audit
   invokes no training entry point and constructs no
   probe context during a run; both signals' definitions,
   all four composition configurations, both anchorings,
   both authorship targets, and every policy default are
   untouched (asserted by the controls' continued exact
   reproduction under the existing suite — the fixture
   stack this audit measures *is* the SI-33 decisive
   run, reproduced blob-for-blob: `d0001`
   `sha256:2761e8ff…6ca9e`, `d0002`
   `sha256:d7af7f1f…e625`).
8. **Determinism — PASS.** The audit repeated from the
   same bundle reproduces the record byte-identically
   (asserted by test over the canonical record bytes;
   the audit is a deterministic function of the seed
   weights, the landed heads' overlays, and the named
   partitions — no sampling, no wall-clock dependence
   in any figure).
9. **Untouched scope — PASS.** The full existing suite
   is green (see below); `si squash` still fails closed
   on the landed transformer stack (asserted by test on
   the stack itself); `si serve` still returns 501 for
   every transformer bundle (existing suite); the demo
   path is unchanged; no CLI wiring is added; the
   Glimmer seed is untouched.

**Full suite (this slice's own run, on this VM):**
`394 passed, 2 skipped` (full suite, author's run in
the venv; the slice's own file: 12 passed). The 12 new
tests are exactly the delta over the SI-33 merge's
`382 passed, 2 skipped` — no prior test changed its
result.

## What is still NOT claimed

- **No candidate, no verdict, no prediction.** The
  shortlist is empty and R2 is measured; neither fact
  predicts any candidate's gate verdict in either
  direction (§18 item 27). SI-36, if it comes, argues
  from R2's stated consequence — this slice pre-writes
  no part of it.
- **No approved second target.** Exactly zero new
  authorship-target values are added; the axis keeps
  serving exactly `"lm-head"` and `"final-mlp-down"`.
- **No transfer figures.** For 12 of the 14 in-stream
  matrices the bound is a local, pre-attenuation bound
  and downstream transfer is recorded **unknown**. In
  particular the large bounds on the normed-stream
  matrices (5.091) price movement at those tensors'
  outputs only; nothing here says a candidate on any of
  them would move the logits by any particular amount,
  would ADMIT, or would reach its destination — RFC
  SI-34 §19 item 1's optimism warning stands in full.
- **No Glimmer anything, no real-base work.** The audit
  is a miniature measurement; the Glimmer seed is
  untouched; no download was performed, scheduled, or
  implied, and candidate (a)'s decision remains the
  operator's, not taken here.
- **No gate, policy, rank, or tolerance change.**
  `rank_cap` stays 8; the representation question's
  trigger remains as SI-34 §7 evaluated it (not fired);
  SI-31's candidate still fails on both of its lines
  (safety rank and identity, independently) wherever
  its verdict is summarised (§18 item 25), and nothing
  in this audit re-judges it.
- **No scale claim of any kind.** The figures are facts
  about a constructed, deterministically seeded,
  not-trained miniature, at three named heads, on two
  constructed signals' partitions. Nothing in them is
  claimed about any real model, any trained base, or
  Glimmer — including by analogy, and including the
  direction "real models are less inert", for which
  this record contains no evidence in either direction.

## Terminology

Machine learning infrastructure throughout. The
miniature is constructed, deterministically seeded, and
not trained; its labels travel with every figure above.
Admission and authorship, where mentioned, are
miniature-scale facts about the landed stack this audit
measures — cited, never extended. CPU only; no GPU, no
cloud, no spend, no downloads.
