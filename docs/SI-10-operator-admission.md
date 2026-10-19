# SI-10 — Operator admission for transformer deltas (as-built)

**Status: implemented.** `si/admit.py`, a `record_transformer_delta`
journal method, the `si admit-transformer-delta` CLI subcommand, and
`tests/test_admit.py`. This is the follow-up the SI-9 RFC left open
in §4.4 and §5.5: *an operator-admission materialise path* for
transformer deltas.

## The gap

SI-9 built, in order: the delta format (9a), the gate (9b — a valid
transformer delta ESCALATEs, never ADMITs, because identity and
capability probes cannot honestly be evaluated without a transformer
probe harness), and apply-at-load plus a VERIFY smoke step (9c,
`si.loop._verify_transformer_apply`, deliberately not wired into
`run_cycle` because nothing ever materialised a transformer delta).

So an escalated delta had nowhere to go. The escalation said "ask the
operator", and no mechanism existed for the operator to answer.
SI-10 is that mechanism — and only that mechanism.

## Design

The precedent is SI-6 (`si/amend.py`, constitution amendment):
operator-only actions live outside the autonomous loop, are reached
only through an explicit CLI subcommand, require an operator name
and a non-empty reason, journal the decision with the gate's
ESCALATE report verbatim, and state plainly that the `--operator`
flag is an attribution/authorization record, not cryptographic
authentication. SI-10 follows that shape exactly.

### Entry point

```
si admit-transformer-delta <bundle> \
    --record candidate-record.json --blob candidate.safetensors \
    --operator NAME --reason "..." \
    [--parents a,b] [--source-seqs 1,2] [--intent-reason "..."]
```

Programmatically: `si.admit.admit_transformer_delta(bundle_dir,
candidate, intent, operator, reason, gate=None)`, where `candidate`
is the SI-9b candidate contract (`{"record", "blob_bytes"}`) and
`intent` is the proposing `Intent` (kind `transformer-delta-v1`).
The CLI takes the intent's declared target list from the record
itself — the gate's provenance check refuses any disagreement, and a
CLI that let the operator type a second, divergent list would only
manufacture such disagreements. `--parents` defaults to the current
head, the delta's true parent at admission time.

### Steps (as built)

1. **Refusals before anything is touched.** Missing/empty operator
   or reason → `AdmitError`, nothing written, nothing journaled
   (the amend precedent). A non-transformer intent or a malformed
   candidate is refused the same way. `load_bundle` must verify the
   bundle — a tampered bundle is never admitted into.
2. **Re-validation at admission.** The SI-9b gate evaluates the
   intent + candidate fresh, against the bundle as it stands now
   (minimal context: verified manifest + active-chain stack depth;
   its probe methods raise if touched — the gate's transformer path
   never calls them). Admission proceeds only on **ESCALATE**:
   structurally valid, probes not evaluated. A REJECT-class delta
   can never be admitted, by anyone — the refusal is journaled as a
   self-edit (verdict REJECT, result `rejected`, operator attached,
   anchored, exactly the way the loop journals rejections) and
   `AdmitError` is raised; the head never moves. Any other non-
   ESCALATE verdict is refused identically: if the gate ever learns
   to ADMIT a transformer delta, autonomous admission will be its
   own decision, not this path silently widening.
3. **Journal before materialise.** The admission decision is a
   self-edit journaled first: kind `transformer-delta-v1`, the
   gate's ESCALATE report verbatim, the operator (the self-edit's
   `operator` field), the admission reason as the self-edit
   reason, and an `operator_admission` block on the journaled intent
   dict — operator, reason, candidate delta id, timestamp, and the
   basis string (below) — followed by an `operator_admitted`
   provenance record naming both delta ids. Only after those rows
   exist does anything materialise (tests assert the self-edit's
   seq precedes the delta row's seq).
4. **Materialise with the loop's own semantics.** The blob is
   written via `si.loop._write_blob` (content-addressed,
   skip-if-exists — SI never duplicates bytes; a blob-vs-record
   binding mismatch at this point aborts before the manifest moves).
   The manifest record is the candidate record with three fields
   re-assigned the way the loop assigns demo records at materialise
   time: a fresh runtime delta id (`si.loop._next_delta_id`), parent
   = the head it lands on, and `self_edit_id` = the admission
   self-edit. The candidate's proposed id is preserved in the
   journaled `operator_admission` block and the `operator_admitted`
   provenance note, so the specific escalated artifact stays
   identifiable. The delta is journaled via the new
   `Journal.record_transformer_delta` (the demo-shaped
   `record_delta` cannot describe this kind honestly: rank/alpha are
   per-target, so the row carries the SI-9a `targets` array verbatim
   and no single rank/alpha is fabricated), its result is marked
   `materialized`, and the journal is checkpointed + re-anchored via
   `si.loop._save_and_anchor`. There is no parallel bundle writer.
   Admitting a delta whose id collides, or whose blob is already on
   the active chain, is refused — the same delta cannot be stacked
   twice.
5. **VERIFY.** `si.loop._verify_transformer_apply` runs against the
   new head. Its report is journaled as a provenance record whose
   relation is the report's own kind, `transformer-apply-smoke`,
   with the full report as the note. On failure, the loop's existing
   rollback core (`si.loop._do_rollback`, cause `verify-failed`,
   the smoke report as its probe report) moves the head back to the
   pre-admission head and marks both the delta and the admission
   self-edit `rolled_back`. A failed-verify delta is never left
   silently active; the blob and journal rows stay as evidence, as
   rollback always leaves them.

### The honest labels (the point of the whole slice)

The basis string journaled on every admission, word for word:

> operator admission: the operator acted as the identity/capability
> check for this delta. Identity/capability probes were NOT
> evaluated — no transformer probe harness exists — and this
> admission is not a probe pass (RFC SI-9 §4.3/§4.4)

Consequences, stated plainly:

- The journaled gate verdict for an admitted transformer delta is
  **ESCALATE**, forever. It is never rewritten to ADMIT, and the
  identity/capability checks inside the report stay `passed=False`,
  `measured=None`, "not evaluated".
- **Operator admission ≠ probe pass.** The operator is a human
  standing in for checks the runtime cannot run. If a probe harness
  ever exists, it will supersede this judgment call with evidence;
  until then the record says a person decided, and why.
- **Autonomous admission is still impossible.** `run_cycle` /
  `si run` halt on ESCALATE and have no path to `si/admit.py`. No
  flag, config key, or policy entry makes admission automatic —
  there is deliberately no such switch.

### Non-goals (unchanged)

No training, no delta authorship, no inference, no probe harness,
no GPU/cloud/spend. `si serve` still returns 501 for transformer
bundles. `si export` is unchanged. The demo-delta path is unchanged
(a test re-runs a demo cycle to prove it). No seed-format change.

## Tests

`tests/test_admit.py` (15 tests): the success path (chain, apply,
journal-before-materialise seq order, smoke report journaling,
honest gate report), duplicate-admission refusal, missing
operator/reason refusals (parametrised, nothing journaled),
wrong-kind refusal, REJECT-class refusal (policy-rejected and
structurally-invalid variants; refusal journaled, head unmoved),
VERIFY-failure rollback (the smoke step is patched to fail; head,
applied value, delta status, edit result, and the rollback record
are all checked), autonomous-run-unchanged (the same delta that
`run_cycles` escalates on is then operator-admitted), the demo path
unchanged, and CLI parser + end-to-end (including the
missing-operator CLI refusal).

This is machine learning infrastructure, not "custom LLM software."
