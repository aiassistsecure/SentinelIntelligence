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
```

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
