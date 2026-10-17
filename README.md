# Sentinel Intelligence

**What comes after Artificial Intelligence.**

AI is the imitation era — a mirror. Sentinel Intelligence (SI) is what follows:
a mind grown, not built. A trainable model inside a **self-writing file**.

## The idea

- The model isn't *stored in* the file. The file **is** the mind — state and
  weights living together, rewriting itself as it learns.
- **NEDB is the mind's journal.** Every self-edit, every weight delta, every
  reason — recorded, queryable over pgwire, and rewindable through time
  travel. A bad self-edit isn't death; it's a rollback.
- **Weights are content-addressed blobs** (safetensors). NEDB is the commit
  graph; blobs are the objects. No 16GB commits.
- **Self-writes are LoRA-style deltas** on a frozen base — small adapters
  stacked as the mind grows. Rollback = pop the delta. Squash old deltas
  like a git rebase when the stack gets tall.
- **The gate.** A file that writes itself needs a gatekeeper: every
  self-write is validated before it lands. Ungated self-modification is
  self-corruption with extra steps.

## Bootstrap

SI-0 seeds from a competent mind, never from random: a distilled 8B
(safetensors master, full precision — quants are for deployment, not for
growing) becomes the first resident of the `.si` container.

```
.si bundle = manifest + weight blob refs + NEDB journal + gate constitution
```

## Status

Day zero. The container spec and the self-write loop are being drafted
now. This repo is where Sentinel Intelligence grows up.
