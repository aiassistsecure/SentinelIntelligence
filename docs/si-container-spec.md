# The `.si` Container — SI-0 Bundle Format Spec

> AI shipped models as files. SI ships a mind as a bundle that can rewrite
> itself without losing itself. This spec defines the bundle. Nothing here is
> aspirational — every field exists because the self-write loop needs it.

## What a `.si` bundle is

A `.si` bundle is a **directory** with a fixed layout. Not a bespoke binary
format — directories compose, diff, sync, and survive. (A future single-file
`.sia` is just a tar of this directory. Don't build it until someone needs it.)

```
sentinel-001.si/
├── manifest.json          # the bundle's identity and current state
├── blobs/
│   └── sha256/
│       ├── 9f2c…a1.safetensors   # frozen base weights (full precision)
│       ├── 41bd…77.safetensors   # delta 0001 (LoRA-style adapter)
│       └── c803…e9.safetensors   # delta 0002
├── journal/               # NEDB journal snapshot / live data dir
│   └── … (see docs/nedb-journal-schema.md)
├── constitution.md        # the gate constitution, human-readable
└── constitution.sha256    # hash of constitution.md at bundle time
```

## Design rules

1. **Tensors are adopted, not invented.** Every weight blob is a standard
   `safetensors` file. Safe loading, mmap, existing tooling — all free.
2. **Blobs are immutable and content-addressed.** Filename is
   `<sha256-of-bytes>.safetensors`. If the bytes change, the name changes.
   There is no "edit a blob," only "add a blob."
3. **The base is frozen forever.** The seed model's weights never change.
   All learning is a stack of deltas. (Why: rollback is `pop`, not surgery.)
4. **The manifest is the only mutable truth**, and it mutates only through
   the self-write loop (see `docs/self-write-loop.md`).
5. **Full precision inside the bundle.** Deltas are stored f16/bf16 minimum.
   Quants (GGUF etc.) are *exports* for deployment, never the learning
   substrate. You don't do brain surgery with a lossy copy.

## `manifest.json` — field spec

All keys required unless marked optional. Canonical form: UTF-8 JSON,
keys sorted, no whitespace beyond separators `,` and `:` — the bundle's
identity hash is computed over this canonical form.

```json
{
  "si_version": "0.1.0",
  "name": "sentinel-001",
  "created_at": "2026-10-16T22:00:00Z",
  "parent_manifest": null,

  "base": {
    "source": "hf://Interchained/pocket-spark-v100-small",
    "revision": "sha256:…",
    "blob": "sha256:9f2c…a1",
    "bytes": 17179869184,
    "dtype": "bf16",
    "format": "safetensors",
    "checkpoint_config": { "model_type": "llama", "…": "…" },
    "checkpoint_config_sha256": "sha256:…",
    "arch_source": "checkpoint-config"
  },

  "deltas": [
    {
      "id": "d0001",
      "parent": "base",
      "blob": "sha256:41bd…77",
      "bytes": 134217728,
      "rank": 16,
      "alpha": 32,
      "target_modules": ["q_proj", "v_proj"],
      "norm": 0.0042,
      "created_at": "2026-10-17T03:12:00Z",
      "self_edit_id": "se_0007"
    }
  ],
  "head": "d0001",

  "journal": {
    "backend": "nedb",
    "snapshot": "journal/snap-0007/",
    "last_seq": 128,
    "last_seq_hash": "sha256:…"
  },

  "constitution": {
    "hash": "sha256:…",
    "gate_version": "0.1.0",
    "policy_hash": "sha256:…"
  }
}
```

Field notes:

- `si_version` — version of **this spec**, semver. A runtime MUST refuse a
  bundle whose major version it doesn't understand. Minor additions are
  additive-only (new optional keys).
- `parent_manifest` — sha256 of the canonical manifest this bundle was
  derived from, or `null` at genesis. Bundles form a lineage chain, git-style.
- `base.blob` — content hash of the frozen seed weights. The seed MUST be a
  full-precision safetensors master. A quant here is a spec violation.
- `deltas[].parent` — `"base"` or a previous delta id. The stack is a chain;
  `head` points at the tip. Rollback = move `head` back (see self-write loop).
- `deltas[].norm` — Frobenius norm of the delta, recorded by the gate.
  Norm caps are enforced pre-write (see gate checklist).
- `base.checkpoint_config` (optional) — the checkpoint's `config.json`,
  captured by `si seed` when it sits beside the weights, plus its
  `checkpoint_config_sha256`. `si export` derives real GGUF architecture
  KV entries (`<arch>.block_count`, `<arch>.embedding_length`,
  `<arch>.attention.head_count`, …) from it.
- `base.arch_source` — how export labels the GGUF's architecture:
  `"checkpoint-config"` (values from the author's config.json),
  `"tensor-shape-inference"` (conservative inference from tensor
  names/shapes when no config was captured; export fails closed on
  ambiguity), or `"demo-unknown"` (demo seed data — not a model;
  the GGUF keeps `general.architecture = "unknown"`). Unknown values
  are omitted from the GGUF, never invented.
- `deltas[].self_edit_id` — foreign key into the NEDB journal's `self_edits`
  collection. Every delta traces to the exact self-write that proposed it.
- `journal.last_seq` / `last_seq_hash` — the journal's logical clock and a
  hash-chained checkpoint, so a tampered journal is detectable.
- `constitution.hash` — the constitution is versioned by content. Change the
  constitution, you change the hash, and the journal records *who authorized
  the change*. The mind can't quietly rewrite its own values.

## Hashing scheme

- **Blobs:** `SHA-256` over raw file bytes.
- **Manifest identity:** `SHA-256` over the canonical JSON form.
  `bundle_id = "si1:" + hex(sha256(canonical_manifest))`.
- **Journal chain:** each journal entry hashes the previous entry's hash
  (hash-chained log). `last_seq_hash` in the manifest anchors it.
- No exotic hashes. SHA-256 everywhere, hex-encoded, lowercase.

## Versioning

| What          | Scheme | Example        | Rule                                    |
|---------------|--------|---------------|------------------------------------------|
| Spec (`si_version`) | semver | `0.1.0` | Major bump = breaking manifest change; runtime refuses unknown majors |
| Bundle lineage | content hash chain | `parent_manifest` | Every derived bundle points at its parent |
| Constitution  | content hash | `constitution.hash` | Values change only explicitly, journaled |
| Gate policy   | content hash | `policy_hash` | Policy change is itself a gated self-edit |

## Non-goals (SI-0)

- No encryption at rest in SI-0 (add a sealed-box layer later, don't half-do it).
- No cross-bundle delta sharing yet (federation is SI-1).
- No quant storage inside the bundle. Exports live outside.

---

*Honest footnote: this spec is small on purpose. The epic part isn't the
container — it's what the loop does inside it.*
