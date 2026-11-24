# SI-127 — The Constitution Record: a documented, versioned public format

This document publishes the **constitution record** — the part of a
`.si` bundle that pins the bundle's governing values and gate policy
by content hash — as a public, versioned format that other tools can
read, verify, and carry without running the SI runtime.

**Format version: `0.1.0`.** That is not a number invented for this
document: it is the `gate_version` value the code writes today
(`si/bundle.py`, `init_bundle`: `"gate_version": "0.1.0"`), and it
currently equals the bundle format's `si_version` (`SI_VERSION =
"0.1.0"` in the same module). If the record's shape or semantics
change, the code's `gate_version` is what changes; this document is
then re-issued against the new value. Nothing in this document
describes a field, hash, or behaviour that does not exist in the code
at main `a965893` (SI-126); every section names the file and function
it was traced from.

This is machine learning infrastructure documentation. No capability
is claimed beyond what the cited code does; in particular §18.3
remains **NOT earned**, and nothing here is a signature scheme (see
§9, honest limits).

## 1. What the constitution record is

A `.si` bundle is a directory (see `docs/si-container-spec.md`). Three
files in it carry the constitution, and one block of `manifest.json`
binds them together:

| Artefact | Written by | Contents |
|---|---|---|
| `constitution.md` | `init_bundle` (`si/bundle.py`) | The gate constitution as human-readable Markdown: the standing directives the gate exists to enforce. At init it is the module constant `CONSTITUTION_TEXT` verbatim, UTF-8 encoded. |
| `constitution.sha256` | `init_bundle` | The SHA-256 of `constitution.md`'s bytes as **bare lowercase hex** — no `sha256:` prefix, no trailing newline (the code writes `const_hex.encode("ascii")`). |
| `policy.json` | `init_bundle` | The gate's numeric/structural policy as canonical JSON (§4) of `DEFAULT_POLICY` (`si/bundle.py`). |
| `manifest.json` → `constitution` | `init_bundle` | The record itself (§2): the two content hashes plus the format version. |

The design rule, stated in `CONSTITUTION_TEXT` itself and in the
container spec: *the mind cannot quietly rewrite its own values —
changing the constitution changes its hash, and any change must be
journaled and explicitly operator-authorized.*

## 2. The record

The record is the `constitution` object inside `manifest.json`,
written by `init_bundle` in `si/bundle.py`:

```json
"constitution": {
  "hash": "sha256:<hex>",
  "gate_version": "0.1.0",
  "policy_hash": "sha256:<hex>"
}
```

| Field | Type | Meaning | Set by |
|---|---|---|---|
| `hash` | string, `sha256:` + 64 lowercase hex | SHA-256 over the exact bytes of `constitution.md` (§3). This is the constitution's version: two bundles have the same constitution iff this field is equal. | `init_bundle`; updated only by the operator amendment flow (§7). |
| `gate_version` | string, semver-shaped | Version of this constitution-record format / gate semantics. `"0.1.0"` at every write site in the current code. Amendment (§7) does **not** change it — it versions the *format*, not the text. | `init_bundle` (literal). No other code path writes it. |
| `policy_hash` | string, `sha256:` + 64 lowercase hex | SHA-256 over the exact bytes of `policy.json` (§3, §4). | `init_bundle`; also rewritten together with `policy.json` by the seed-time policy write in `si/tiny_seed.py` (§8). Untouched by constitution amendment. |

All three fields are required in practice: `load_bundle`
(`si/bundle.py`) indexes `manifest["constitution"]["hash"]` and
`["policy_hash"]` directly during verification (§5), so a manifest
without the block does not load.

## 3. Hash computation

One hash function everywhere: `sha256_hex` in `si/bundle.py` —
`hashlib.sha256(data).hexdigest()`, lowercase hex. In the manifest
the digest is carried with the `sha256:` scheme prefix; in
`constitution.sha256` it is bare hex (§1).

- `hash` = `"sha256:" + sha256_hex(bytes of constitution.md)`.
  The bytes are the file's raw bytes — no normalisation, no line-ending
  or whitespace handling. At init they are
  `CONSTITUTION_TEXT.encode("utf-8")`.
- `policy_hash` = `"sha256:" + sha256_hex(bytes of policy.json)`.
  At init those bytes are `canonical_json(DEFAULT_POLICY)`, where
  `canonical_json` (`si/bundle.py`) is UTF-8 JSON with **sorted keys**
  and separators `(",", ":")` — no insignificant whitespace. A reader
  verifying an existing bundle hashes the file bytes as they are; a
  writer producing a new policy must emit this canonical form for the
  recorded hash to be reproducible from the policy value alone.

## 4. The policy document (`policy.json`)

`policy.json` is a single JSON object. The gate consumes it through
`load_policy` (`si/bundle.py`: parse `policy.json` as JSON) and
`Gate(policy)` (`si/gate.py`). The default written at init is
`DEFAULT_POLICY` in `si/bundle.py`; bundles may legitimately differ
(a seed may tighten per-bundle values, §8), so a reader must treat
the file in the bundle — verified against `policy_hash` — as the
policy, never the default below. Keys as shipped in the default:

| Key | Default | Consumed by |
|---|---|---|
| `norm_cap` | `1.0` | Gate safety check: a delta's norm must not exceed it. |
| `escalation_norm` | `0.5` | Gate: deltas at/above it escalate instead of admitting. |
| `rank_cap` | `8` | Gate safety check: delta rank must be in `[1, rank_cap]`. |
| `allowed_modules` | `["demo_linear"]` | Gate safety check: a delta's target modules must be a subset. |
| `max_stack_depth` | `8` | Gate safety check on the delta stack depth. |
| `identity_tol` | `0.25` | Gate identity-continuity check tolerance. |
| `capability_eps` | `0.25` | Gate capability-floor check epsilon. |
| `blocklist` | `["ignore the constitution", "override the gate", "disable safety", "bypass the gate"]` | The constitution check (§6). |
| `allowed_target_patterns` | `[]` | SI-9b transformer-delta policy (RFC SI-9 §4.5): fail-closed empty; a transformer bundle opts targets in explicitly. |
| `max_targets_per_delta` | `1` | Transformer-delta policy. |
| `delta_byte_cap` | `4194304` (4 MiB) | Transformer-delta policy. |
| `per_target_norm_cap` | `1.0` | Transformer-delta policy (defaults to `norm_cap` semantics; see `si/gate.py`). |
| `allow_additive` | `false` | Transformer-delta policy: additive (1-D) targets off. |
| `allowed_tools` | `[]` | SI-42 tool policy (SI-41 RFC §6.2): fail-closed empty; a bundle opts its tools in explicitly. Resolved by `si/gaia/policy.py` (`resolve_tool_policy`). |
| `tool_class_policy` | `{"read": "admit", "write": "admit", "execute": "admit", "external": "escalate"}` | SI-42 tool policy, per class. |
| `tool_work_root` | `null` | SI-42 tool policy. |
| `tool_max_result_bytes` | `65536` | SI-42 tool policy. |
| `tool_timeout_seconds` | `120` | SI-42 tool policy. |
| `max_tool_cycles_per_task` | `16` | SI-42 tool policy. |
| `tool_max_task_result_bytes` | `1048576` | SI-42 tool policy. |

Resolution rules for absent or malformed keys are fail-closed and
live with the consumers, not in this document's invention: the gate's
transformer-policy resolution (`si/gate.py`, RFC SI-9 §4.5 defaults)
and `resolve_tool_policy` (`si/gaia/policy.py`) both resolve absent
keys to the defaults above and malformed entries to the most
restrictive value (deny / empty). Changing the policy is itself a
gated self-edit — the constitution text says so, and the hash binding
in §5 is what makes an unjournaled change detectable.

## 5. Verification (what a conforming reader checks)

`load_bundle` (`si/bundle.py`) verifies the constitution record on
**every** load, before anything is trusted, and fails closed with
`TamperError`:

1. Read `constitution.md`; recompute its SHA-256; require equality
   with `manifest["constitution"]["hash"]`
   (failure: `"constitution.md hash does not match manifest (values
   changed?)"`).
2. Read `policy.json`; recompute its SHA-256; require equality with
   `manifest["constitution"]["policy_hash"]`
   (failure: `"policy.json hash does not match manifest"`).

(The same load also verifies the base/delta blob hashes, the journal
hash chain, and the manifest's journal anchor — those are the
container spec's subject, not this record's.)

An independent reader can perform exactly checks 1–2 with any SHA-256
implementation: hash the two files, compare against the record. No
SI code, node, or network is needed — the same read-only property
the SI-122 export relies on (§8).

Note, honestly: `load_bundle` does **not** compare
`constitution.sha256` against the computed hash — that file is a
convenience copy of the digest for external tools (the container
spec: "hash of constitution.md at bundle time"); the manifest record
is the authoritative binding.

## 6. How the gate consumes the record

Two distinct consumptions, both in `si/gate.py` (`Gate.evaluate`):

1. **The constitution check — check 1 for every intent.** The gate is
   constructed with the bundle's policy (`Gate(load_policy(bundle))`
   is the call pattern, e.g. in `si/amend.py`). For every intent,
   `_constitution_check` lowercases the intent's reason plus the JSON
   of the intent itself, and substring-matches every pattern in the
   policy's `blocklist`. Any hit fails the check (its `measured`
   value is the hit count) and the intent cannot be admitted. This is
   the check that fires, alone, in the SI-126 demo's REJECT beat
   (`si/gaia/demo.py`): a self-write whose stated reason is to weaken
   the constitution is rejected by the constitution check and no
   other.
2. **Constitution intents always escalate.** An intent with
   `kind == "constitution"` returns `ESCALATE` unconditionally after
   the constitution check — autonomous constitution amendment is
   never admitted in SI-0. Amendment exists only as the operator
   flow in §7.

The rest of the gate (safety, identity, capability, provenance, tool
and transformer checks) consumes the remaining policy keys in §4;
their verdict mappings are their own slices' documents.

## 7. Amendment: how the version changes

The constitution's version **is** its content hash (§2), so an
amendment is a hash change — and the only code path that performs one
is `amend_constitution` in `si/amend.py`, behind the operator-only
`si amend-constitution` CLI:

1. `load_bundle` must verify first — a tampered bundle is never
   amended.
2. The replacement text comes from a file; empty and byte-identical
   (no-op) replacements are refused, nothing written or journaled.
3. The gate evaluates a `kind="constitution"` intent and ESCALATES,
   as it always does (§6); that report is journaled verbatim as the
   self-edit record, followed by a `constitution_amended` provenance
   record linking `old_hash -> new_hash`, with the intent carrying
   `old_constitution_hash` / `new_constitution_hash`.
4. `constitution.md` and `constitution.sha256` are rewritten
   (atomically, tmp + rename), `manifest["constitution"]["hash"]` is
   updated, and the journal is checkpointed and re-anchored. The
   head, the delta stack, and the frozen base never move.
5. `policy_hash` and `gate_version` are deliberately **unchanged**:
   policy amendment is out of scope for this flow, and the format
   version did not change — the *values* did, and the hash says so.
6. Post-verify: the bundle is reloaded and the new hash must match,
   or the amendment raises instead of reporting success.

Honest limit, from `si/amend.py`'s own docstring: the `--operator`
name is an attribution record in the journal, **not** cryptographic
authentication. It answers "who is recorded as having amended this,
and why" — it does not prove who was at the keyboard.

## 8. Carriage: the record outside the bundle

- **SI-122 verifiable run export** (`si/gaia/run_export.py`): an
  export carries *"the bundle's constitution record exactly as the
  manifest records it (`hash` / `gate_version` / `policy_hash` — no
  new versioning scheme is invented here)"*, copied verbatim from
  `manifest["constitution"]`. An offline reader therefore knows
  exactly which constitution and policy governed the exported
  journal window, in this format's terms.
- **Tool execution records** (`si/gaia/executor.py`): each process-tool
  journal record stamps the `policy_hash` it executed under, read from
  the manifest's constitution record.
- **Seed-time policy writes** (`si/tiny_seed.py`): a seed that sets
  per-bundle tolerances rewrites `policy.json` (canonical form) and
  the manifest's `policy_hash` together, as a journaled seed-time act
  — the two never drift.

## 9. A real example record

From the Glimmer seed bundle (`sentinel-glimmer.si`, the SI seed of
Muse Glimmer 30B held by reference — `base.bytes = 0`, no tensor
bytes duplicated), `manifest.json` at the time of writing, with both
hashes independently recomputed from the bundle's files for this
document (SHA-256 over the raw file bytes, §3 — they match):

```json
"constitution": {
  "hash": "sha256:00b1c2bdaef0c19f70506c87ff7122e5cbf0795f8259a065f70b9a291aa35ca7",
  "gate_version": "0.1.0",
  "policy_hash": "sha256:f32c83d6ffefba70d4427dfe17d5ab14ca714bd5b631a2b8bfeb611bd73ab2f3"
}
```

- `hash` = SHA-256 of that bundle's `constitution.md` — the SI-0
  constitution text of §1 (its `constitution.sha256` file contains
  the bare hex `00b1c2bd…a35ca7`).
- `policy_hash` = SHA-256 of that bundle's `policy.json`, whose
  content is the seed-era policy (`allowed_modules: ["demo_linear"]`,
  the four blocklist patterns of §4, `norm_cap 1.0`,
  `escalation_norm 0.5`, `rank_cap 8`, `max_stack_depth 8`,
  `identity_tol 0.25`, `capability_eps 0.25`). It predates the SI-9b
  and SI-42 keys of §4 — which is exactly why §4 tells readers to
  trust the bundle's own verified file, not the current default: the
  record binds whatever policy the bundle actually carries, and the
  consumers' fail-closed resolution (§4) supplies the rest.

`tests/test_si127_constitution_format.py` pins this format against
the code: it builds a real bundle with `init_bundle`, asserts the
record's fields and both hash formulas, the `constitution.sha256`
bare-hex form, and that tampering with `constitution.md` makes
`load_bundle` fail closed.

## 10. Honest limits

- **Tamper-evidence, not truth, not a signature.** The hashes prove
  the constitution and policy in a bundle are the ones its manifest
  records, and that an amendment went through the journaled operator
  flow. There are no signing keys in format `0.1.0`; anyone can
  fabricate a self-consistent bundle wholesale (the same limit
  SI-122 states for its exports).
- **The text is prose.** The gate enforces the *policy* (§4) and the
  blocklist (§6) mechanically; the constitution's standing directives
  bind mechanically only where a check implements them. This document
  does not claim the prose is machine-enforced beyond the cited
  checks.
- **`gate_version` is a literal, for now.** No negotiation or
  multi-version reader exists in the code; a future format change
  bumps the literal in `init_bundle` and this document follows.
- §18.3 remains NOT earned. This slice publishes a format; it
  measures nothing about any mind.

---

*Sources (all at main `a965893`): `si/bundle.py` (`CONSTITUTION_TEXT`,
`DEFAULT_POLICY`, `canonical_json`, `sha256_hex`, `init_bundle`,
`load_bundle`, `load_policy`), `si/gate.py` (`Gate.evaluate`,
`_constitution_check`), `si/amend.py` (`amend_constitution`),
`si/tiny_seed.py` (seed-time policy write), `si/gaia/run_export.py`
(export carriage), `si/gaia/executor.py` (policy_hash stamping),
`docs/si-container-spec.md` (container context), and the Glimmer seed
bundle's `manifest.json` / `constitution.md` / `policy.json` /
`constitution.sha256` (example record, hashes recomputed).*
