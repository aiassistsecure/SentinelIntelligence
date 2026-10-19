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
si seed ./sentinel-001.si ./huge-checkpoint/ --by-reference   # SI-8: store zero tensor bytes; reference the checkpoint in place
si cycle ./sentinel-001.si --mind demo     # one self-write: trains a real adapter, gates it, journals it
si run ./sentinel-001.si --cycles 5        # autonomous multi-cycle run (halts on REJECT/ESCALATE)
si log ./sentinel-001.si
si why ./sentinel-001.si d0001             # why does this delta exist?
si diff ./sentinel-001.si base..d0001
si rollback ./sentinel-001.si --to base --reason "operator: trim stack"
si squash ./sentinel-001.si --keep 3    # compress oldest deltas into one gated delta (head id unchanged)
si serve ./sentinel-001.si --port 8000  # serve the head for inference (local HTTP, read-only)
si amend-constitution ./sentinel-001.si --from-file ./new-constitution.md --reason "why" --operator "your-name"  # operator-only, journaled
si admit-transformer-delta ./sentinel-001.si --record ./candidate.json --blob ./candidate.safetensors --reason "why" --operator "your-name"  # SI-10: operator-only admission of an escalated transformer delta
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

`si seed --by-reference` (SI-8) is for checkpoints too large to copy
into the bundle store — the never-duplicate-bytes rule taken to its
end state. It stores **zero tensor bytes**: the manifest records
`seed_format: "reference-v1"` and, per tensor, an external reference
(absolute source path, absolute byte offset + length inside that
file, dtype, shape, and a content hash). The hash is the *same*
canonical-blob SHA-256 a copy seed would store — checkpoint identity
is mode-independent, so re-seeding the same checkpoint in either mode
is an idempotent no-op and never silently converts storage modes —
but it is computed **streaming** (bounded memory: no whole shard and
no all-tensor plan is ever held in RAM), and copy mode's source
hashes are streaming now too. Verification stays fail-closed at full
strength: every `load_bundle` re-checks each referenced source file
(existence, exact size, full streaming SHA-256) and each tensor's
streaming canonical hash, and reads via `si export` /
`load_base_tensor` re-verify the tensor they read — a tampered,
truncated, or missing source raises rather than passing silently.
(That makes loading a by-reference bundle cost two streaming passes
over its checkpoint; there are no spot checks or mtime shortcuts.)
**The dependency, plainly: a by-reference base depends on the source
checkpoint remaining at its recorded absolute path. Moving or
deleting the source breaks the base — that is the design price of
never duplicating bytes.** Copy mode remains the choice for portable
bundles, and it now refuses up front — before writing anything — when
the new blob bytes clearly cannot fit in the free space at the
bundle. One read-path note: bfloat16 tensors referenced in place are
widened losslessly to float32 when read (numpy has no bf16 dtype in
this runtime; their raw bytes and hashes are bf16-exact regardless).

Transformer deltas (SI-9, RFC `docs/SI-9-transformer-self-write-rfc.md`)
are how a self-write names real transformer tensors: a delta artifact
(SI-9a, `si/transformer_delta.py`) binds each target by exact name and
base content hash — never substring matching — and the gate (SI-9b)
structurally validates such deltas and **escalates** them to the
operator, because identity/capability probes cannot honestly be run
without a transformer probe harness; a valid transformer delta is
never autonomously admitted. **Apply-at-load** (SI-9c,
`si.apply.load_applied_tensor`) is a **library capability, not
inference**: given a bundle, a manifest, and one tensor name, it
returns that tensor in memory with the active chain's deltas for that
exact name folded in oldest-first (`W + (alpha/rank)·B@A` for `lora`
targets; `w + d` for `additive` targets only when policy
`allow_additive` is true — it defaults to false), starting from the
hash-verified per-tensor read. Nothing is materialised to disk, the
base is never modified or copied, and a rolled-back delta simply
stops being applied. Nothing in SI-9 trains or authors deltas,
nothing runs a transformer, and `si serve` still returns an explicit
`501` for transformer bundles; `si export`'s existing demo-era delta
resolution is unchanged (reconciling it with exact named-tensor
deltas is a separate, future decision — see RFC §5.5).

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

`si admit-transformer-delta` (SI-10, `docs/SI-10-operator-admission.md`)
is the **operator-only** materialise path for transformer deltas —
the answer an escalated delta (SI-9b) was waiting for. It re-runs the
SI-9b gate at admission time and admits only a delta whose fresh
verdict is ESCALATE; a REJECT-class delta can never be admitted, by
anyone (the refusal is journaled and the head never moves). Admission
requires an explicit `--operator` name and a non-empty `--reason`,
is journaled *before* anything materialises (gate ESCALATE report
verbatim, operator, reason, timestamp), materialises through the
loop's existing blob/head semantics, and then runs the SI-9c VERIFY
smoke step — journaled under its honest `transformer-apply-smoke`
label, with rollback on failure. Two honesty labels, stated plainly:
**operator admission is not a probe pass** — identity/capability
probes were *not evaluated* (no transformer probe harness exists);
the operator acted as the identity/capability check, and the journal
says exactly that. And **autonomous admission is still impossible**:
`si run` / `run_cycle` halt on ESCALATE and have no path to this
command; no flag, config, or policy entry makes admission automatic.
As with SI-6, `--operator` is an attribution record, not
cryptographic authentication.

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
