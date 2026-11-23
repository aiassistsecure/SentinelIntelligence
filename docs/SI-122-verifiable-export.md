# SI-122 — EXPORT-1: verifiable run bundle export + offline verifier

**Status: BUILT.** One command exports a window of a
bundle's journal as ONE self-contained file; anyone can
re-verify it offline — no bundle, no node, no network —
against the journal's own hash chain. Implementation:
`si/gaia/run_export.py`, wired into the main CLI as
`si gaia-export` and `si gaia-verify-export`. Tests:
`tests/test_gaia_si122_export.py`.

This is the roadmap's FREE TURN hero item (SI-120 lane):
publish one verifiable, tamper-evident Gaia run bundle
and invite the public to forge it.

| Element | SI-122 state |
| --- | --- |
| Export (`si gaia-export <bundle> [--from SEQ] [--to SEQ] [--out FILE]`) | **built** — one self-contained JSON file: journal entries in order, the chain data to re-verify them, the manifest's constitution record + head + manifest id, export metadata (exported_at, source tip seq + tip hash, recorded si_version) |
| Offline verifier (`si gaia-verify-export <file>`) | **built** — opens NO bundle; recomputes the journal's existing hash chain over the exported entries and checks internal consistency; clear verdict; exit non-zero on ANY mismatch |
| Read-only on the source bundle | **built and test-pinned** — the journal is opened the `_open_read` way (chain verified before export, fail closed); bundle bytes are identical after export |
| Signing | **NOT built — no signing keys in v1** (see honest scope, §4) |

---

## 1. Usage

Export the whole journal:

```
si gaia-export path/to/bundle.si --out run-export.json
```

Export a window — entries with `from_seq < seq <= to_seq`,
the same `(a, b]` window convention as `si gaia-diff`
(SI-118). `--to` defaults to the journal tip; a window
past the tip is refused, as is a reversed window. A
zero-width window is the explicit empty export. Without
`--out`, the export is printed to stdout.

Verify it anywhere:

```
si gaia-verify-export run-export.json
# VERIFIED: N entries, window (0, T], chain recomputes
# to the recorded tip (seq T, sha256:...)
# bundle: <name>  head=<head>  manifest=si1:...
```

Exit code 0 iff the export verifies; 1 with a `FAILED:`
line naming the problem otherwise.

## 2. The export format (format_version 1)

```json
{
  "format": "si-gaia-run-export",
  "format_version": 1,
  "label": "<the honest-scope label>",
  "export": {
    "exported_at": "2026-…Z",
    "tool": "si gaia-export (SI-122/EXPORT-1)",
    "si_version": "0.1.0",
    "source_bundle": "/abs/path/to/bundle.si",
    "window": {"from_seq": 0, "to_seq": 74},
    "window_start_prev_hash": "genesis",
    "window_tip": {"seq": 74, "entry_hash": "sha256:…"},
    "entry_count": 74,
    "source_tip": {"seq": 74, "entry_hash": "sha256:…"}
  },
  "bundle": {
    "name": "…",
    "manifest_id": "si1:…",
    "head": "…",
    "constitution": {"hash": "sha256:…",
                     "gate_version": "0.1.0",
                     "policy_hash": "sha256:…"},
    "journal_anchor": {"backend": "…", "last_seq": …,
                       "last_seq_hash": "…"}
  },
  "entries": [
    {"collection": "checkpoints",
     "entry": {"seq": 1, "prev_hash": "genesis",
               "entry_hash": "sha256:…", "…": "…"}}
  ]
}
```

- **Entries** are the journal rows in sequence order,
  each tagged with its collection. A row carries every
  field the journal recorded for it, including its `seq`,
  `prev_hash`, and `entry_hash`. Backend storage detail
  (sqlite row ids, NEDB native `_seq`/`_hash`) is NOT
  carried: the journal's own chain verification excludes
  those keys, so they are not part of the verifiable
  record.
- **The constitution record is the manifest's own** —
  `hash` / `gate_version` / `policy_hash`, copied
  verbatim. No new versioning scheme is invented.
- **The manifest head** is the manifest's `head` plus
  its `manifest_id` (`si1:` + sha256 of the canonical
  manifest), so an export is tied to the exact manifest
  it was taken at.

## 3. How a third party verifies

The verifier reuses the journal's EXISTING hash chain —
no second hash scheme:

1. For each entry, in file order: its `seq` must strictly
   increase, lie inside the declared window, and its
   `prev_hash` must equal the preceding entry's
   `entry_hash` (the first entry's must equal the
   recorded `window_start_prev_hash` — `genesis` for a
   full export).
2. Each `entry_hash` must recompute as
   `sha256(canonical{prev_hash, seq, payload})` — the
   journal's own `_entry_hash` — where `payload` excludes
   exactly what `Journal.verify_chain` excludes: the
   chain keys, the native backend keys, the mutable
   operational-state keys (`result` / `delta_id` /
   `status`), and `ts`.
3. Internal consistency: `entry_count` matches the
   entries present; the chain must end exactly at
   `window_tip` and at the window's `to_seq`; when the
   window reaches the recorded source tip, the final
   hash must equal the recorded source tip hash.

Failure modes covered (each test-pinned): a tampered
entry payload, two reordered entries, a truncated entry
list, an altered entry hash, an altered recorded tip
hash.

## 4. Honest scope

- The chain proves the export is **internally
  consistent and tamper-evident against the recorded
  tip**: alter any chain-protected byte after export and
  verification fails.
- It does **not** prove the outside world happened as
  recorded. The journal is the record; the export proves
  the record wasn't rewritten, not that the record was
  true.
- It is **not a signature**. V1 has no signing keys:
  anyone can fabricate a journal wholesale and export a
  self-consistent file from it. What they cannot do is
  alter THIS export's chain-protected record
  undetectably, or pass off an altered copy as the
  export whose tip hash was published. Publishing the
  tip hash (seq + entry hash) alongside the export is
  what anchors it — the same discipline as a checksum
  next to a download.
- The operational-state keys (`result` / `delta_id` /
  `status`) and the wall-clock `ts` are carried in each
  entry but are **not chain-protected** — that is the
  journal's own documented semantics (`si/journal.py`):
  they are bookkeeping that changes as a self-edit moves
  pending → materialized; the causal record — intent,
  reason, gate verdict/report — is chain-protected and
  immutable.

## 5. What it earns

An export is a projection of the journal, like the
SI-118 diff view: it makes the journal's existing
tamper-evidence portable and third-party-checkable. It
earns no §18.1 row and no §18.3 progress; §18.3 remains
not earned.
