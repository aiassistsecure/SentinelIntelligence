# The NEDB Journal — Schema Spec (SI-0)

> NEDB is the mind's memory: every self-edit recorded, queryable over
> pgwire, rewindable via time travel. NEDB stays schemaless — this doc
> defines *conventions*, not constraints. Required keys are listed; extra
> keys are always allowed. The journal must never reject a truth for not
> fitting a table.

## The split (read this first)

```
NEDB (the journal)                      Disk (the luggage)
───────────────────                     ──────────────────
self_edits    ──what was proposed        blobs/sha256/*.safetensors
deltas        ──what changed             (immutable, content-addressed)
checkpoints   ──where the head was
provenance    ──why it exists
experiences   ──what was learned
```

NEDB is the **commit graph**; blobs are the **objects**. NEDB never stores
raw weight bytes — it stores hashes that point at them. This keeps time
travel cheap: rewinding the journal rewinds *pointers*, not gigabytes.

## Logical clock

Every journal entry carries `seq` — a monotonically increasing integer,
assigned by the runtime (not the model). `seq` is the mind's own time.
Wall-clock `ts` is recorded too, but `seq` is what ordering, rollback, and
"time travel to seq 412" use. The manifest's `journal.last_seq` anchors the
current tip.

## Collections

### `self_edits` — one row per proposed self-write

The complete biography of the mind's decisions, including the ones it
refused to make.

| Key | Type | Notes |
|-----|------|-------|
| `_id` | auto | NEDB-assigned |
| `seq` | int | logical clock, unique |
| `ts` | timestamptz | wall clock |
| `edit_id` | string | `se_0007` — stable external id |
| `kind` | string | `delta` \| `memory` \| `rollback` \| `constitution` |
| `intent` | object | the proposed intent, verbatim |
| `reason` | string | why, in the mind's own words — required |
| `gate_verdict` | string | `ADMIT` \| `REJECT` \| `ESCALATE` |
| `gate_report` | object | per-check pass/fail + measurements |
| `operator` | string \| null | who escalated/authorized, if anyone |
| `result` | string | `materialized` \| `rejected` \| `rolled_back` \| `pending` |
| `delta_id` | string \| null | FK → `deltas.id`, when materialized |

### `deltas` — metadata for each weight delta

Small rows pointing at big blobs.

| Key | Type | Notes |
|-----|------|-------|
| `_id` | auto | |
| `id` | string | `d0001` — matches manifest |
| `parent` | string | `base` or previous delta id |
| `blob` | string | `sha256:<hex>` — content address on disk |
| `bytes` | int | blob size |
| `rank` / `alpha` | int | LoRA hyperparameters |
| `target_modules` | string[] | e.g. `["q_proj","v_proj"]` |
| `targets` | object[] | transformer deltas only (`kind: "transformer-delta-v1"`, SI-9a): one entry per named base tensor — `{name, base_blob, base_shape, base_dtype, kind (lora\|additive), rank/alpha (lora only), factor_keys, target_norm}`. `base_blob` binds the delta to the base tensor's content hash (mode-independent, RFC §2.4). Absent on demo deltas. |
| `norm` | float | Frobenius norm at materialization; for transformer deltas, the root-sum-square over per-target norms (SI-9a) |
| `self_edit_id` | string | FK → `self_edits.edit_id` — the proposal that caused it |
| `source_seqs` | int[] | journal seqs this delta learned from |
| `status` | string | `active` \| `superseded` \| `rolled_back` |

### `checkpoints` — where `head` was, over time

| Key | Type | Notes |
|-----|------|-------|
| `_id` | auto | |
| `seq` | int | journal seq at checkpoint |
| `ts` | timestamptz | |
| `manifest` | string | `sha256:<hex>` of canonical manifest |
| `head` | string | delta id or `base` |
| `note` | string | e.g. `"post-verify d0001"`, `"operator rollback"` |

A checkpoint is written after every materialization, every rollback, and
every constitution change. `si log` is just this collection, pretty-printed.

### `provenance` — the causal edges

| Key | Type | Notes |
|-----|------|-------|
| `_id` | auto | |
| `child` | string | delta id / edit id / checkpoint seq |
| `parents` | string[] | ids or seqs it derives from |
| `relation` | string | `learned_from` \| `supersedes` \| `rolled_back_by` \| `authorized_by` |
| `note` | string | optional human/machine-readable detail |

Provenance is what lets the mind answer "why am I like this?" — recursively,
all the way back to genesis.

### `experiences` — memory intents (optional but encouraged)

| Key | Type | Notes |
|-----|------|-------|
| `_id` | auto | |
| `seq` | int | |
| `ts` | timestamptz | |
| `kind` | string | `observation` \| `conclusion` \| `operator_note` |
| `record` | object | schemaless payload |
| `self_edit_id` | string | FK → the memory intent that wrote it |

## pgwire: the mind, queryable

Because the journal is NEDB, the whole biography is plain SQL. SI-0 ships
these as saved views; ad-hoc queries are expected and encouraged.

```sql
-- Every self-edit that touched attention weights last week
SELECT e.edit_id, e.ts, e.reason, d.target_modules, d.norm
FROM self_edits e JOIN deltas d ON d.self_edit_id = e.edit_id
WHERE e.ts > now() - interval '7 days'
  AND d.target_modules && ARRAY['q_proj','v_proj','k_proj','o_proj']
ORDER BY e.seq;

-- Where was the mind's head on a given date?
SELECT head, manifest FROM checkpoints
WHERE ts <= '2026-10-20T00:00:00Z'
ORDER BY seq DESC LIMIT 1;

-- Every rollback and why
SELECT e.edit_id, e.ts, e.reason, e.gate_report
FROM self_edits e
WHERE e.kind = 'rollback'
ORDER BY e.seq DESC;

-- Full causal chain behind a delta (recursive)
WITH RECURSIVE chain(child) AS (
  SELECT 'd0042'
  UNION
  SELECT p.parents FROM provenance p JOIN chain c ON p.child = c.child
)
SELECT * FROM chain;
```

## Time travel

NEDB's time travel rewinds any collection to any `seq`. Combined with the
frozen base + immutable blobs, this means: **any past self of the mind is
fully reconstructible** — journal state at seq N plus the blobs its
manifest pointed at. Rollback moves `head`; time travel lets you *visit*
without moving.

## Integrity

- `journal.last_seq_hash` (manifest) anchors a hash chain over journal
  entries: each entry includes `prev_hash`. A tampered journal is detectable
  on load.
- The runtime verifies the chain at startup and refuses to run on a broken
  chain (fail closed — a mind that can't trust its memory shouldn't act).
- Backups: snapshot the journal directory + copy any blobs referenced by
  retained checkpoints. GC of unreferenced blobs is SI-1.

## Non-goals (SI-0)

- No vector index inside NEDB for experience recall (SI-1; flat scan is
  honest at this scale).
- No multi-writer. One runtime owns the journal. Serial or nothing.

---

*NEDB doesn't store the mind's weights. It stores the mind's life — which
turns out to be the more interesting half.*
