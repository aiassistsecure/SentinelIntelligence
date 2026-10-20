# SI-11 — Export reconciliation for transformer deltas (as-built)

**Status: implemented.** `si/export.py` (`ExportError`,
transformer-delta handling in `materialize_head`), and
`tests/test_export_transformer.py`. This is the follow-up RFC SI-9
§5.5 explicitly left open: *reconciling `si export` with exact
named-tensor (transformer) deltas.*

## The gap (reproduced before the fix)

SI-10 made transformer deltas admittable by an operator, so a
bundle's active chain can contain a record with
`kind="transformer-delta-v1"`, a per-target `targets` array, and a
blob with indexed keys `t{i}.A`/`t{i}.B`/`t{i}.d`
(`si/transformer_delta.py`).

But `si export` was never reconciled. `materialize_head()` iterated
the active chain and called the demo-era `_load_delta()` for every
record, which expects top-level `alpha`/`rank` and blob tensors
`delta.B`/`delta.A`, plus `target_modules` substring resolution.

Reproduction (synthetic miniature bundle, `tests/test_admit.py`
fixture style; operator-admitted LoRA delta to
`model.language_model.layers.0.self_attn.q_proj.weight`), then
`materialize_head()`:

```
File "si/export.py", in materialize_head
    B, A, scale = _load_delta(bundle_dir, d)
File "si/export.py", in _load_delta
    B = np.asarray(f.get_tensor("delta.B"), dtype=np.float32)
safetensors._safetensors_rust.SafetensorError: File does not contain tensor delta.B
```

That is neither correct application nor an honest fail-closed
error. Exporting a head that crashes with a raw library error — or
could silently misapply — is an integrity gap once admission is
live.

## Design decision

**Implement correct application in export's materialize path for
transformer-delta-v1 deltas, using exact tensor names only.**

- **Branch by kind, in chain order.** `materialize_head` walks
  `_active_chain` oldest-first as before. A record with
  `kind == "transformer-delta-v1"` goes through the new transformer
  path; every other record goes through the original demo path,
  unchanged. Rolled-back / superseded deltas are not on
  `_active_chain`, so they are never applied (proven by a test).
- **Exact names only — never substring.** Substring resolution was
  explicitly rejected for transformer deltas in RFC §2.4/§5.5
  (ambiguous by construction). A transformer target applies only to
  the base tensor whose name equals `target["name"]` exactly.
- **Reuse validated code, don't duplicate arithmetic.**
  Before any factor is trusted, the record/blob pair is re-run
  through `si.transformer_delta.validate_transformer_delta`
  against the caller's manifest (content hash, metadata-vs-manifest
  agreement, exact-name base binding, factor shapes/dtypes,
  finiteness, recomputed norms) and parsed with
  `si.transformer_delta.parse_blob`. Factor math mirrors
  `si.apply.load_applied_tensor` exactly, in float32:
  `W + (alpha/rank) * B@A` for `lora` (factors from
  `target["factor_keys"]`, in order A then B), `w + d` for
  `additive`. `allow_additive` resolution reuses
  `si.apply._resolve_allow_additive` directly, so export and
  apply-at-load cannot drift: the `policy` argument when given,
  else the bundle's `policy.json`, else `False`; only the literal
  `True` permits additive. Tests assert export's materialized
  tensors equal `load_applied_tensor`'s results for the same head.
  (Export materializes *all* tensors at once because it already
  loads the whole base into memory; the per-target application
  inside that loop is deliberately consistent with the per-tensor
  apply path, not a second semantics.)
- **Additive: refuse, never skip.** If a transformer delta has an
  `additive` target and `allow_additive` is not literally true,
  export raises `ExportError` explaining why — it does not silently
  skip the target (which would export a wrong head labelled as the
  head) and does not apply it against policy.
- **Fail closed with a named error.** Any transformer delta that
  cannot be validated/applied — unknown target, shape mismatch,
  metadata/record disagreement, missing factor keys, missing blob,
  invalid rank/alpha — raises `si.export.ExportError` (a
  `ValueError` subclass). No raw `SafetensorError`/`KeyError` from
  the transformer path leaks. Because `export_bundle` materializes
  *before* writing any file or journaling, a failure leaves **no
  partial GGUF file and no journal entry claiming a successful
  export** (proven by a test that counts provenance rows).
- **Mixed stacks: apply both, in chain order.** If a bundle somehow
  contains both demo and transformer deltas, export applies both in
  `_active_chain` order, each by its own kind's rules. Rationale:
  each kind's resolution is deterministic and fail-closed, so
  chain-order application is well defined; refusing would strand a
  head that apply-at-load partly understands, and silently dropping
  one kind would be worse. This choice is documented here and in
  `si/export.py`, and proven by a mixed-stack test. (RFC §3 still
  calls mixed stacks a non-goal for *authorship*; export's job is
  only to not misrepresent a head that exists.)
- **Demo path unchanged.** The demo-delta branch is the original
  code, untouched in behaviour; `tests/test_export.py` passes with
  its existing expectations.

New public surface (additive only): `ExportError`, and optional
`policy=None` parameters on `materialize_head` / `export_bundle`
(forwarded for the additive rule; `None` means the bundle's
`policy.json`, exactly as in `si.apply`).

## Honest limits — what export still does not do

- **No full-Glimmer export.** Materialisation loads ALL base
  tensors as float32 (~119 GB for Glimmer, RFC §5.4). SI-11 fixes
  correctness for transformer deltas; it does not change that cost,
  and no test here attempts a full-Glimmer export. Tests are
  synthetic miniature bundles only.
- **No inference, no training, no probe harness.** Nothing here
  runs a transformer. `si serve` is unchanged and still returns
  501 for transformer bundles.
- **No seed-format change, no autonomous-admission change, no
  squash change.** Transformer squash remains deferred (RFC §6.4).
- Export remains **not gated** and does not move the head, as
  before; it journals `relation="export"` provenance only after a
  successful materialisation and write.

## Tests

`tests/test_export_transformer.py` (7 tests): (a) admitted LoRA
delta materializes to the hand-computed value, differs from base,
matches `load_applied_tensor`, and exports to GGUF; (b) untargeted
tensors base-identical; (c) two stacked transformer deltas apply
oldest-first matching `load_applied_tensor`; (d) rolled-back delta
not applied; (e) additive refused when `allow_additive` is false
(materialize and export, no file), applied when true and matching
`load_applied_tensor`; (f) unknown-target and tampered-norm deltas
fail closed with `ExportError`, no output file, no export journal
entry; plus the mixed-stack test. Demo export (g) is covered by the
unchanged `tests/test_export.py` in the full suite.

This is machine learning infrastructure, not "custom LLM software."
CPU only; no GPU, cloud, or spend.
