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

## SI-0 runtime

The reference implementation of the self-write loop lives in `si/`:
`bundle` (the `.si` container), `journal` (hash-chained logical-clock
journal), `gate` (the five-check ternary gate), `loop`
(perceive → propose → gate → journal → materialize → verify, strictly
serial), `minds` (a scripted `MockMind` and a real CPU `DemoMind` that
learns rank-2 LoRA-style adapters), and `cli` (the `si` command).

Two honesty labels, stated in code and in user-visible strings: the demo
seed created by `si init` is **random data, not a trained model** (use
`si seed` to import a real safetensors checkpoint as the frozen base), and
the default journal backend is **the real NEDB engine** (`si/backends/nedb.py`,
dogfooding actual NEDB over pgwire). Embedded sqlite remains as the
zero-service fallback (`--journal-backend embedded`).

```bash
python3 -m venv /tmp/prfactory-si0/venv
/tmp/prfactory-si0/venv/bin/pip install -r requirements.txt
/tmp/prfactory-si0/venv/bin/pip install -e .

si init ./sentinel-001.si --name sentinel-001   # journal: real NEDB (default)
si seed ./sentinel-001.si ./my-checkpoint/   # real safetensors base (or skip: demo random seed)
si cycle ./sentinel-001.si --mind demo     # one self-write: trains a real adapter, gates it, journals it
si run ./sentinel-001.si --cycles 5        # autonomous multi-cycle run (halts on REJECT/ESCALATE)
si log ./sentinel-001.si
si why ./sentinel-001.si d0001             # why does this delta exist?
si diff ./sentinel-001.si base..d0001
si rollback ./sentinel-001.si --to base --reason "operator: trim stack"
si squash ./sentinel-001.si --keep 3    # compress oldest deltas into one gated delta (head id unchanged)
si serve ./sentinel-001.si --port 8000  # serve the head for inference (local HTTP, read-only)
si amend-constitution ./sentinel-001.si --from-file ./new-constitution.md --reason "why" --operator "your-name"  # operator-only, journaled
```

`si serve` API (localhost only, no auth — do not expose it beyond
localhost): `GET /health` returns the bundle id, head, mind kind, and
arch source. `POST /infer` with `{"input": [8 numbers]}` returns
`{"output": [8 numbers]}` — real inference from the demo mind
materialized at the current head (frozen base + active deltas). SI has
no transformer inference engine, so a bundle seeded from a real
transformer checkpoint returns an explicit `501` on `/infer` (never a
fabricated output); only demo-mind inference is supported. Serving is
read-only: no journal writes, and the head never moves.

`si seed` (SI-7) imports a real multi-tensor transformer checkpoint
**per tensor**: every tensor becomes its own content-addressed blob
(a valid single-tensor safetensors file under the canonical key
`"tensor"`; the tensor's name lives in the manifest's
`base.tensor_blobs`), so content addressing addresses tensor *data* —
two tensors with identical bytes under different names are stored
exactly once. **SI never duplicates bytes**: a blob whose hash already
exists in the store is never rewritten or copied (dedup is at blob
level, skip-if-exists; per-tensor re-serialization means source files
are never hardlinked, and none is claimed). Re-seeding the exact
checkpoint that is already the base is an idempotent no-op (zero bytes
written, reported as `already_seeded`). Sharded HuggingFace layouts
(`model.safetensors.index.json` + shard files) are validated
fail-closed — missing shards, missing mapped tensors, and tensors
claimed by two shards are all refused — and a captured `config.json`
drives `si export`'s GGUF architecture metadata end to end (tied
embeddings / a missing `lm_head` included). Pre-SI-7 bundles
(whole-file `parts` bases, demo bases) still load, verify, and export.
The seeded import is journaled with full provenance accounting:
tensor count, total bytes, deduplicated bytes, source path, and the
config sha256 when present.

`si amend-constitution` (SI-6) is the **operator-only** way to change
the gate constitution. Autonomous amendment is still impossible: the
gate returns ESCALATE for every constitution intent, and the autonomous
loop has no path to this command. An amendment requires an explicit
`--operator` name and a non-empty `--reason`; it is **journaled and
hash-versioned** — the journal records the gate's ESCALATE report, the
operator, the reason, and the hash chain link (old hash → new hash) —
and it never moves the head, the delta stack, or the frozen base.
Empty and no-op (identical) replacement files are refused, a tampered
bundle is never amended, and the bundle is re-verified after the write.
Honest limit: `--operator` is an **attribution/authorization record in
the journal, not cryptographic authentication** — this local reference
runtime trusts its operator; the flag records who claimed to act, it
does not prove it. Policy (`policy.json`) amendment is out of scope.

Embedded sqlite fallback (zero services, no daemon):

```bash
si init ./sentinel-embedded.si --name sentinel-embedded --journal-backend embedded
si cycle ./sentinel-embedded.si --mind demo   # same loop, journal in sqlite
si log ./sentinel-embedded.si
```

The NEDB backend autostarts the `nedbd-v2` daemon (prebuilt binary in the
`nedb-engine` wheel — no Rust toolchain needed) on bundle-derived ports,
keeps its data inside the bundle (`journal/nedb-data`), and stops only
daemons it started itself. Caveats: NEDB's pgwire is simple-query protocol
only (`psycopg2`, not `psycopg3`); pgwire is cleartext, loopback only.

Terminology: this is machine learning infrastructure, not "custom LLM
software." No GPU, no network calls; everything deterministic where tests
need it (seeded RNG).
