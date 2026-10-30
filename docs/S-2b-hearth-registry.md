# S-2b — Registering the `hearth.*` tools in SI's registry (SI-55)

**Status:** implemented on `s-2b/hearth-registry` against main
`c81895f` (S-2a merged as PR #59). This slice closes the registry
gap S-2a documented as its headline honesty item: SI's registry
registered only the 12 SI-41 tools, so every fork watcher
proposal gated REJECT at check 2's `tool_registered` sub-check
regardless of its merits, and S-3's ADMIT path had nothing it
could ever submit. §0 says plainly what is live and what is not.

| Element | S-2b state |
| --- | --- |
| `hearth.restart` registered | **live** — external class, compensable, schema from the S-2a watcher fixture's args |
| `hearth.notify` registered | **live** — external class, one-way, only `model` required (see §2) |
| Watcher proposals gate on their merits | **live** — `tool_registered` passes; verdict is the class policy's: default `external: escalate` → ESCALATE; `external: admit` → ADMIT |
| ADMIT semantics at the seam | **unchanged from S-2a** — the self-edit rests `pending`, gate report journaled; nothing is submitted and nothing executes locally |
| Local execution of `hearth.*` | **does not exist** — the names are in no LIVE set; the B-3 executor refuses them by construction |
| S-3 work submission to the fork | **not built in this slice** — an ADMIT was a journaled pending self-edit and nothing more; SI-56/S-3 has since built submission as a separate explicit step (docs/S-3-work-submission.md) |
| S-4 leash crossing | **not built** |
| S-5 outcome return | **not built** — the registered result schema is the SI-side pending shape, not a fork outcome |
| Performance | **UNMEASURED** — no latency/throughput claim is made for registration, gating, or ingest |

---

## 1. What is registered, and the evidence for each name

Exactly two names — no others were evidenced in-repo, and none
were invented:

| Tool | Evidence in-repo |
| --- | --- |
| `hearth.restart` | `si/gaia/seam.py` module docstring (the watcher's example proposal); the S-2a test fixture `WATCHER` in `tests/test_gaia_seam_ingest.py` — `tool: "hearth.restart"`, args `{"model": "muse", "spine_event": {"hash": ..., "seq": 41}}` |
| `hearth.notify` | `docs/S-2a-proposal-ingest.md` §3 — named alongside `hearth.restart` as one of "the fork watcher's candidate names" from fork `watcher.rs` |

## 2. Schemas, classes, policy — and why

Both specs live in `si/gaia/registry.py` §8.4, built with the
same `_spec` helper and the same §6.1 schema subset as the
existing 12 tools, `version=1`, `introduced_by="SI-55/S-2b"`.

- **Consequence class `external` for both.** This answers the
  class-vocabulary question G-9 §16.4 left open for these names:
  no new class was invented. A restart/notify's effect lands on
  the fork's managed runtime — a system outside SI's work root —
  which is exactly what the existing `external` class prices.
  The class is a policy dial (§7.3), not a cage: the shipped
  default (`external: escalate`) makes a watcher proposal
  ESCALATE for an operator; a bundle whose policy admits the
  external class ADMITs it. Registration changed *existence*;
  permission remains the bundle policy's.
- **`hearth.restart` — compensable, timeout 120.** Args:
  required `model` (non-empty string); optional `spine_event`
  (`{hash, seq}`, seq an int ≥ 0) — the fixture's exact shape;
  the seam already validates `spine_event`/`caused_by`
  agreement before the gate. Compensable because a restarted
  model can be stopped/restarted again by a new gated intent —
  there is no SI-side compensation mechanism, and none is
  claimed; the label records the actuation's nature, priced by
  the gate like every other tool's.
- **`hearth.notify` — one-way, timeout 30.** Args: required
  `model`; optional `message` and `spine_event`. No in-repo
  evidence pins a notify payload beyond the model it concerns,
  so `message` is declared **optional rather than invented as
  required** — an undeclared arg still REJECTs at
  `args_schema`, so the gate can price everything the registry
  declares, and a future fork-side fixture can tighten the
  schema in its own slice if it evidences more. One-way: a
  surfaced notification cannot be un-surfaced.
- **Result schema — the pending shape, honestly.** Both tools
  share `{status: "pending", model, detail?}`. That is the
  SI-side record of an admitted seam intent at this slice, NOT
  a fork outcome: S-5 outcome return is unbuilt, so no result
  claiming a restart happened exists anywhere in this slice.
- **Gate seam branch (`si/gate.py`).** On a live context, a
  registered seam-crossing tool with class rule `admit` ADMITs
  with the new execution label `TOOL_SEAM_PENDING_EXECUTION`
  ("not executed locally — seam-crossing tool … ADMIT rests
  pending; S-3 work submission … is UNBUILT"). Escalated and
  rejected seam intents get their own seam labels. This branch
  is an *admission* exception to B-3's external hold for
  registry-class-external tools only: path/cwd-priced
  external-effective local tools keep the B-3 hold exactly.
  The gate also prices a per-model ordering key (`hearth:<model>`)
  for seam tools, so restarts of one model serialise in the
  queue fold instead of racing across the seam.
- **Deliberately NOT in any live set.** `LIVE_TOOLS_B2` /
  `LIVE_TOOLS_B3` are unchanged; `hearth.*` joins the new
  `HEARTH_SEAM_TOOL_NAMES` / `SEAM_CROSSING_TOOLS` sets instead.
  The B-3 executor refuses any tool outside `LIVE_TOOLS_B3`, so
  even a caller that tried to execute an admitted seam intent
  locally would be refused and journaled — no code path pretends
  SI restarted a Hearth model.

Labels updated to stay true: `REGISTRY_GAP_LABEL` in
`si/gaia/seam.py` (kept under its S-2a name for import
compatibility, aliased as `REGISTRY_STATUS_LABEL`) now records
the gap as CLOSED with S-3 still UNBUILT; the seam module
docstring and `docs/S-2a-proposal-ingest.md` §3/§5 carry the
same correction. The B-1 registry acceptance now asserts
`REGISTRY == RFC_V1_TOOL_NAMES ∪ HEARTH_SEAM_TOOL_NAMES`
(14 tools), and the scripted corpus gains one valid row per
hearth tool (29 rows; both ESCALATE in the corpus's non-live
shadow context, as every valid row does there).

## 3. Tests

New file `tests/test_gaia_s2b_hearth_registry.py` (15 tests):
registration exactness and evidence schemas; not-in-any-live-set;
the pending result shape; the corrected gap label; gate merits
(ADMIT under `external: admit` with the seam pending label and
the `hearth:<model>` ordering key, ESCALATE under the default
policy, shadow ESCALATE on a non-live context, REJECT of an
unregistered `hearth.*` name at `tool_registered`, REJECT of an
undeclared arg at `args_schema`); and seam end-to-end (watcher
ADMITs to `pending` with the gate report journaled and no
`tool_result`/`gaia_work` artefacts, watcher ESCALATEs under the
corpus policy, notify ADMITs to pending, unregistered name still
REJECTs at the seam). S-2a's suite was updated where it pinned
the old gap truth (the watcher fixture's verdict is now the
merits ESCALATE; its REJECT-artefacts test now proposes a
still-unregistered name so it still tests a true REJECT).

## 4. What remains unbuilt

S-3 (work submission of ADMIT seam intents to the fork —
built after this slice by SI-56, docs/S-3-work-submission.md), S-4
(operator leash crossing the seam), S-5 (outcome return — until
then no `hearth.*` execution result exists or is claimed).
Registration performance and ingest performance remain
UNMEASURED. The fork side is untouched by this slice.
