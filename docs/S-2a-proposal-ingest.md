# S-2a — Gaia seam Flow 1, SI side: proposal ingest (fork watcher proposals gated + journaled, ACK after journal)

**Status:** implemented on `s-2a/proposal-ingest` against the
G-9 seam design (gaia repo, `docs/G-9-seam-design.md`, merged at
gaia `580aeb1`). This is the SI half of G-9 §15's **S-2**; the
fork half (egress client + egress log, §8.4) is already merged
fork-side (gaia PR #11, main `283d517`). The branch was built on
SI main `ea9b079` and merged with Q-3 (PR #58, main `9318381`)
before opening — the one shared file, `si/gaia/service.py`,
carries both slices (the merge was four additive hunks: imports,
description labels, `do_POST` routing, `__all__`). §0 says
plainly what is live and what is not.

| G-9 element | S-2a state |
| --- | --- |
| §8 Flow 1: proposals, fork → SI | **live, SI side** — `POST /gaia/proposals` on the B-5 service (§8.3 semantics); fork egress is merged fork-side |
| §8.3.1 validate; malformed → 400, journaling nothing | **live** — `si/gaia/seam.py:validate_proposal`; validation completes before the bundle/journal is opened |
| §8.2 `args.spine_event` must agree with `caused_by` | **live** — disagreement is a validation refusal, never a gate question |
| §8.3.2 dedup by `caused_by`; retry returns the original ACK | **live** — journal scan for a self-edit whose journaled intent carries the pair; a hit writes nothing at all |
| §10.2 one seam task per model | **live** — found or created per model (`record_task_entry`, operator `gaia-seam`); discovery is equality on the deterministic instruction, never parsing |
| §8.3.4 gate + journal the self-edit with its full gate report, **every verdict** | **live** — journaling replicates `admit_tool_intent` branch-for-branch, with `caused_by` inside the journaled intent dict (§12.1) |
| §8.3.5 ACK after journaling: `{task_seq, journal_seq, verdict}` | **live** — `journal_seq` is the self-edit row's seq; the HTTP 200 body is exactly the ACK |
| §8.4 fork egress log / dispositions | **fork-side, already merged** (gaia PR #11) — nothing SI-side in this slice |
| §9 Flow 2: work submission through the seam (S-3) | **built by SI-56/S-3** (docs/S-3-work-submission.md) as a separate explicit step — an ADMIT at ingest still submits nothing to the fork; the admitted self-edit rests `pending` until S-3 submits it |
| §10 leash crossing (S-4) | **not built in this slice** — no leash record crossed at S-2a time; built later SI-side by SI-57 (docs/S-4-leash-crossing.md) |
| §11 Flow 3: outcome return (S-5) | **not built** — no outcome returns to the fork |
| §14 security posture | **as designed** — no authentication on the route; loopback only; identities are asserted provenance, journaled verbatim |
| Performance | Measured under S-7 (SI-62): median 192 ms (range 179–246, N=30) for Q-A5 ingest invocation (ESCALATE) on the miniature fixture, embedded journal, Class S transport, host fingerprint in docs/S-7-seam-measurements.md; single-run; live fork interop performance remains UNMEASURED. Measured under S-7 (SI-62): median 187 ms (range 173–321, N=30) for Q-A6 HTTP ingest on the miniature fixture, embedded journal, Class L transport, host fingerprint in docs/S-7-seam-measurements.md; single-run; live fork interop performance remains UNMEASURED. Live fork egress latency remains **UNMEASURED**. |

---

## 1. What was built

### 1.1 The ingest core — `si/gaia/seam.py` (new)

`ingest_proposal(bundle_dir, payload, policy=None)` is the whole
flow, in §8.3's order:

1. **Validate** (`validate_proposal`): the §12.1 shape — keys
   exactly `{model, tool, args, reason, caused_by}`; `model` /
   `tool` non-empty strings; `args` an object; `reason` a string;
   `caused_by` exactly `{hash, seq}` with `hash` a non-empty
   string and `seq` a non-negative int (a bool is not an int);
   and the §8.2 agreement check on `args.spine_event` when
   present. Every problem is itemised; any problem refuses the
   proposal **before the bundle is loaded or the journal
   opened** — a refused proposal journals nothing, exactly the
   B-5 service's discipline for refused submissions.
2. **Dedup**: under a module-level lock (the service is a
   `ThreadingHTTPServer`; two concurrent identical POSTs must
   not both pass the scan), the journal's `self_edits` are
   scanned for an intent dict carrying this `caused_by` pair.
   A hit rebuilds the original ACK from that row (`journal_seq`
   = the row's seq, `verdict` = its `gate_verdict`, `task_seq`
   = the intent's cited task seq) and returns it — writing
   nothing: no task, no edit, no checkpoint.
3. **Seam task**: the model's seam task is found
   (`find_seam_task`: an `experiences` row of kind `task` whose
   record carries operator `gaia-seam`, the seam
   `submitted_via`, and the deterministic
   `seam_task_instruction(model)`) or created with
   `record_task_entry` (budgets from the bundle policy's cycle /
   byte caps; submitter `gaia-fork-watcher`, an asserted
   provenance identity, not authentication).
4. **Gate + journal**: gate and context are constructed exactly
   as `run_task` constructs them (bundle policy or the caller's
   override; `ProbeContext` for a miniature bundle — the live
   path — else `ToolContext`; work root from the resolved tool
   policy, else the bundle dir; `tool_task_seq` = the seam
   task). The `Intent(kind="tool", parents=[task_ref],
   source_seqs=[task_seq])` satisfies the gate's provenance
   check (check 5) by construction. Journaling then replicates
   `si.gaia.drain.admit_tool_intent` branch-for-branch —
   evaluate, classify, `record_self_edit` with the full gate
   report, provenance edges for parents and source seqs,
   per-verdict result marking, and on REJECT/ESCALATE the
   non-execution outcome (the B-1 shadow record when the
   report's execution label is the shadow label, else the live
   non-execution record). The single seam difference is the
   reason this module exists as its own journaling path:
   §12.1 requires `caused_by` **inside** the journaled intent
   record (it is the dedup index of §8.3.2), and
   `admit_tool_intent` journals `intent.to_dict()` verbatim —
   `Intent` has no `caused_by` field and `si/gate.py` is not
   changed by this slice. After fresh journaling, the manifest
   checkpoint is re-anchored and rewritten, as `run_task`'s
   tail does.
5. **ACK**: `{"task_seq", "journal_seq", "verdict"}` — formed
   only after the self-edit is durable (§7.1: journal first,
   ACK second; an ACK-then-journal crash window would
   manufacture phantom state on the fork).

An ADMIT rests `pending` in the journal (admission semantics,
unchanged): nothing executes SI-side, and nothing is submitted
to the fork — S-3 is what spends an ADMIT, and it is not built.

Budgets are priced per ingest with a fresh context
(`tool_cycles_used` = 0): the gate's budget sub-check sees one
proposal against the bundle's caps, not the seam task's
cumulative lifetime spend — the seam task is perpetual per
model and no cumulative seam budget exists in G-9's design.
Named here so it is a known simplification, not a discovery.

### 1.2 The HTTP surface — `si/gaia/service.py`

- `submit_proposal(bundle_dir, payload, policy=None)`: the
  service-level wrapper (the `submit_task` precedent), returning
  the ingest result dict unchanged.
- `POST /gaia/proposals` (and the trailing-slash spelling) on
  the B-5 handler, routed before the `/tasks` guard, with the
  same body discipline as `/tasks` (1 MiB cap, invalid JSON →
  400). A refused proposal → **400** with the refusal dict
  (`{status, problems, journaled: False, label}`); an acked
  proposal → **200** with a body that is **exactly** the ACK
  dict — the three §12.1 fields, nothing else (the fork's
  egress `parse_ack` requires `journal_seq` + `verdict` and
  tolerates the optional `task_seq`; the seam sends all three).
- `GET /` description gains the route, plus `seam_ingest` and
  `seam_registry_gap` label keys; the module docstring and the
  `serve_service` banner name the route. Existing routes are
  behaviourally unchanged (the B-5 suite passes unmodified).

### 1.3 Tests — `tests/test_gaia_seam_ingest.py` (new)

24 tests: the watcher fixture gated + journaled with its full
report (and the chain verified); retry dedup with an identical
ACK and zero journal growth; 16 malformed mutations refused
with cumulative zero growth on one bundle; REJECT creating no
submission artefacts (no `gaia_work` rows — that kind is S-3's —
no claims, no executed outcome); the `fs_list` ADMIT control
(§3 below); per-model seam-task creation and reuse; the HTTP
wire shape (200 body is exactly the ACK; 400s; description
lists the route); a raw-socket replay of the fork egress
client's exact request bytes (§4); and the env-gated live
interop test (§4).

---

## 2. §15 S-2 acceptance, SI half — item by item

The fork half (egress + restart behaviour) was proven in gaia
PR #11; together the two halves close §15 S-2 items 1–5, with
item 5 (fork restart re-send semantics) fork-side only.

1. **A watcher proposal is gated and journaled at SI; the
   ACK's `journal_seq` names the self-edit row; the gate
   report is in the journal** — `test_watcher_proposal_is_
   gated_and_journaled`: the ACK's `journal_seq` equals the
   citing row's seq; the row carries the full gate report;
   `journal.verify_chain()` holds.
2. **Killing the endpoint between POST and ACK, then retrying,
   yields exactly one self-edit and the identical ACK** —
   `test_retry_returns_identical_ack_and_journals_nothing_
   more` (dedup by `caused_by`, counts across all five journal
   collections unchanged), and over the wire with the real
   egress client in the §4 demo (two probe runs, identical
   `Acked(...)`).
3. **A REJECT-verdict proposal produces no fork work
   submission and an egress-log disposition of `rejected`** —
   SI half: `test_reject_creates_no_submission_artefacts` (no
   `gaia_work`, no claims, no executed outcome; the proposal
   rests as a journaled REJECT). Fork half (no submission,
   `rejected` disposition): gaia PR #11's acceptance.
4. **Malformed proposal → 400, zero journal growth** —
   `test_malformed_proposals_are_refused_and_journal_nothing`
   (16 mutations incl. `args.spine_event` disagreeing with
   `caused_by` on hash and on seq) + the HTTP 400 cases in
   `test_http_wire_shape_and_ack_body`.
5. **Fork restart with an un-ACKed proposal re-sends it; with
   an ACKed one does not** — fork-side only (egress log is the
   fork's store), proven in gaia PR #11. Nothing in this slice
   changes that behaviour; SI's dedup is what makes the
   re-send free.

Interop (brief criterion 5) is §4 below.

---

## 3. The registry gap — the slice's headline honesty item

> **Superseded by SI-55/S-2b (docs/S-2b-hearth-registry.md):**
> the gap described below is CLOSED. `hearth.restart` and
> `hearth.notify` are now registered in SI's registry as
> external-class seam-crossing tools; watcher proposals gate
> on their merits (default external policy ESCALATEs, an
> `external: admit` policy ADMITs to a journaled pending
> self-edit), and S-3's ADMIT path is unblocked on this point.
> S-3 has since been built by SI-56 (docs/S-3-work-submission.md)
> and S-4 by SI-57 (docs/S-4-leash-crossing.md); S-5 remains
> unbuilt. The text below is the S-2a-era
> record, kept for provenance.

SI's tool registry (`si/gaia/registry.py`) registers **only
the 12 SI-41 tools** (`fs_*`, `journal_query`, `bundle_status`,
`probe_run`, `shell_exec`, `code_exec`, `test_run`). The fork
watcher's candidate names — `hearth.restart`, `hearth.notify`
(fork `watcher.rs` calls them "exactly as the SI registry
spells them"; the registry spells no such names — a
repo-wide grep of the SI tree finds no `hearth.*` tool name
anywhere) — are **not registered**. The gate therefore prices
every watcher proposal to **REJECT** at check 2's
`tool_registered` sub-check ("unknown tool 'hearth.restart':
not in the v1 registry … it fails closed"), whatever the
bundle policy allowlists: registration precedes permission in
the gate's own order.

Consequences, stated plainly:

- The ingest pipeline is complete and correct — validation,
  dedup, seam tasks, gating, journaling, ACKs all behave per
  §8.3 — but **no watcher proposal can ADMIT** until a future
  slice registers the `hearth.*` tools in SI's registry (with
  arg schemas, consequence classes, and a policy story — the
  class-vocabulary question G-9 §16.4 already names as open).
  That registry work is **not** in this slice: the registry is
  pinned by B-1's RFC-comparison acceptance, and growing it is
  its own slice with its own review, not a fold-in.
- **S-3's ADMIT path is blocked on that registry slice**: S-3
  submits ADMIT seam intents to the fork, and until `hearth.*`
  can ADMIT there is nothing for it to submit on the watcher
  path. S-3's other dispositions (REJECT/ESCALATE never
  submit) are unaffected.
- The tests pin the current truth rather than a wished-for
  one: the watcher fixture asserts REJECT **with the
  `tool_registered` sub-check named in the journaled report**,
  and a control proposal naming a registered tool
  (`fs_list`, the corpus's own valid row shape) under the
  corpus policy asserts **ADMIT** with the edit resting
  `pending` — proving the pipeline admits what the registry
  knows, and the REJECTs are the registry gap, not an ingest
  defect. `si/gaia/seam.py` carries the gap as
  `REGISTRY_GAP_LABEL`, surfaced on the service description.

---

## 4. Interop with the real fork egress client

The counterparty is not a stub: the fork's egress client
(`crates/gaia-serve/src/egress_client.rs`, merged in gaia
PR #11) was compiled **unmodified** from a pristine copy of
the merged gaia tree, with one added example harness
(`egress_probe`, coordinator-side, in the copy only — the
gaia repo is untouched) that POSTs the fixture proposal
through the real `HttpProposalSink` and prints the
`SendOutcome`.

- Always-on: `test_egress_client_byte_replay` reproduces the
  client's exact request bytes (request line, headers, and
  `Proposal::to_json` field order) over a raw socket and
  asserts the response parses under the client's `parse_ack`
  rules (`journal_seq` a u64, `verdict` one of the three §8.3
  spellings, `task_seq` a u64) — i.e. the real client would
  return `Acked`.
- Live (run at integration; log in the slice backups as
  `interop-live.log`): the probe against the live service
  returned, **twice in a row**,
  `Acked(Ack { task_seq: Some(5), journal_seq: 6, verdict: Reject })`
  — the second run is the dedup retry: the real client
  re-POSTed, SI returned the original ACK, no second
  self-edit exists. In-suite, the same proof is
  `test_live_egress_interop`, env-gated on
  `SI_S2A_EGRESS_PROBE_BIN` so the suite carries no Rust
  dependency; with the probe wired in, the file runs
  **24 passed**.

---

## 5. Scope, labels, and what is not claimed

- **Files:** `si/gaia/seam.py` (new, ingest core),
  `si/gaia/service.py` (route + wrapper + description labels;
  merged with Q-3's service changes — both slices' suites pass
  on the merged tree), `tests/test_gaia_seam_ingest.py` (new),
  this doc. No changes to `si/gate.py`, `si/journal.py`,
  `si/gaia/queue.py`, `si/gaia/drain.py`, `si/gaia/task.py`, or
  the registry.
- **Tests run (coordinator, not the full suite — the loop
  reviewer's run is the full one):** the S-2a file, **24
  passed** (interop env wired in); affected subsets on the
  post-merge tree — B-5, Q-3, Q-1 task/queue/drain, Q-2 drain,
  SSE (SI-47/48/49), gate, B-1…B-4 — **269 passed**. Logs in
  the slice backups (`subset-tests-premerge.log`,
  `subset-tests-postmerge.log`, `interop-live.log`).
- **Labels held:** seam ingest live SI-side; S-3 work
  submission was **UNBUILT** at this slice (an ADMIT submitted
  nothing yet; SI-56/S-3 has since built it as a separate
  explicit step — ingest still submits nothing); leash
  crossing and outcome return unbuilt; watcher proposals
  REJECT at `tool_registered` until a registry slice registers
  `hearth.*` (§3 — that slice is SI-55/S-2b, merged after S-2a;
  watcher proposals now gate on their merits); no authentication, loopback v1 (G-9 §14);
  all ingest performance **UNMEASURED**; miniature scope
  unchanged — nothing executes at ingest, on any verdict.
