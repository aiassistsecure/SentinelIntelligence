# SI-113 — ESC-LIVE-1: CLI approval trigger and the live escalation exercise

**Status: the trigger is BUILT; the live exercise ABORTED at
the §4.3 precondition gate — no proposal was sent, nothing
was resolved, nothing executed.** The abort is the protocol
working as designed (§4.3: "if any precondition fails, stop —
no proposal, no improvisation"). The failed precondition is
an environment fact, recorded below with its evidence and
its unblock. §0 states the label disposition plainly.

**SI-114 re-run (2026-11-13 ~16:37 EST): the exercise was
re-run against the repaired seam and reached a PARTIAL
CLOSE.** One resting escalation was resolved through the
CLI trigger (journal seq 69) and folds `queued`,
chain-verified, and the §4.3 step-7 resolution-refusal
control was proven live; the fork-side execution leg and
the S-5 outcome return were **not observed** inside the
observation window, so §18.3 remains **NOT earned**. The
full re-run record is §4 of this document.

| Element | SI-113 state |
| --- | --- |
| CLI approval trigger (§4.2) | **built** — `si/gaia/escalation_cli.py`, wired as `si gaia-escalation <bundle> list\|show\|resolve`; transport only, resolution recorded exclusively through the existing `resolve_escalation` act (PR #99) |
| Trigger read path, live | **exercised read-only** against the live seam journal during precondition verification (fold + chain verification; journaled nothing) |
| Live exercise (§4.3 steps 1–8) | **NOT RUN — aborted at preconditions**: `si-seam-service` inactive (see §2). No proposal, no resolution, no execution, no negative controls |
| Live exercise re-run (SI-114) | **PARTIAL CLOSE** — one resting escalation resolved live via the CLI trigger (seq 69; fold `escalated` → `queued`); negative control (a) proven; actuation and outcome return not observed in the window; §18.3 NOT earned (see §4) |
| §18.3 label | **NOT earned** (§4.4, unchanged): the exercise did not complete; this record claims no §18.3 progress of any scope |

---

## 1. The trigger (Part 1, built)

Per SI-111 §4.2 and lead ruling D-111-4 (CLI only — no
webhook; no auth story exists, SI-41 §19):

- `list` / `show` fold the bundle's journal
  (`si.gaia.queue.fold_queue`) and present the pending
  escalations: folded `escalated`, no resolution journaled.
  Reads open the journal directly and verify the hash chain
  before folding — the SI-58 lesson (no anchor == tip demand
  on a live journal; no folding an unverified chain).
- `resolve` verifies the bundle (`load_bundle`, fail
  closed), calls the existing verifying act
  `si.gaia.resolution.resolve_escalation` — the only write
  path; the trigger journals nothing itself, rewrites no
  verdict, and adds no verification of its own (the CLI
  parser deliberately carries no `choices=` for the
  resolution vocabulary: the closed vocabulary, folded-state,
  single-resolution, and operator checks all stay inside the
  act) — then re-anchors the manifest checkpoint, exactly as
  seam ingest and seam submission do after journaling, so
  the bundle is left fully verified for its next opener.
- No resolution is ever automatic through the trigger: no
  timer, no allowlist shortcut, no auto-admit (§4.2, kept).

Tests: `tests/test_gaia_escalation_cli_si113.py` (6) — fold
reads; admit-after-review journals exactly one resolution
(operator + basis) and folds the intent to `queued`; reject
annotates and un-pends; invalid value / anonymous operator /
double resolution refuse and journal nothing (journal tip
compared before and after); the `si` CLI wiring parses and
runs end to end. Builder runs: 6 new + 8 SI-92 resolution +
133 affected (CLI, queue, seam ingest, SSE) passed, 1
skipped. The full suite is the reviewer's one run.

One design fact the exercise plan must carry (found while
reading the executor for §4.3 step 5): the SI-side
`Executor` refuses `hearth.restart` by construction — the
registry keeps seam-crossing tools out of every live set,
and the `external` class is outside the B-3 live scope. A
resolved seam intent's execution leg is therefore the G-9
Flow 2 work lifecycle on the fork (verdict-less submission;
the verdict route carries the resolution's Admit; the fork
claims and actuates; S-5 returns the outcome) — the
journaled path this exercise was to walk once the
resolution landed. It was not reached.

## 2. The live exercise — precondition verification (2026-11-13 ~16:10 EST)

Protocol §4.3, verified in order, before step 1. The
operator of record is the lead; this run was executed by the
SI-113 coordinator under the lead's dispatch (D-111-3).

| Precondition (§4.3) | Result | Evidence |
| --- | --- | --- |
| Live gaia node Resident with Imagine v13, ctx pool 4096, parallel 1 | **PASS** | `GET 127.0.0.1:11435/gaia/v1/residency`: `imagine-v13` resident, `ctx_pool` 4096, `parallel` 1 (the `smollm2: resident` line is the known stale spine record; imagine-v12 stopped). Child: llama-server pid 2100 on 127.0.0.1:41583 via the locked template wrapper `llama-server-v13.sh` |
| Seam egress live: node passes `--si-addr 127.0.0.1:11437` | **PASS** | gaia process argv carries `--si-addr 127.0.0.1:11437` |
| `si-seam-service` active | **FAIL** | `systemctl is-active si-seam-service` → inactive; the unit file is absent from /etc/systemd/system (evaporated on a VM recycle — the documented failure mode, NODE.md); `GET 127.0.0.1:11437/health` → connection refused |
| Allowlist exactly `hearth.notify` / `hearth.restart`, external at ESCALATE | **PASS** | Seam bundle policy: `allowed_tools == ['hearth.notify', 'hearth.restart']`; `tool_class_policy.external == 'escalate'` (the journaled operator policy act of 2026-11-11) |
| Journal chain verification at baseline | **PASS** | `verify_chain()` → True |
| Baseline journal tip recorded | **RECORDED** | tip seq **42**, hash `sha256:2d72cd26ce0d55c8eae0417cb9897b7c9df54b58061cfb823d36c6884e489022` |

**Abort.** With the seam service down there is no ingest
surface to drive a proposal through and no live journal
service to resolve against; §4.3 and the dispatch both bind:
stop, no proposal, no improvisation. Per the dispatch's hard
rules the coordinator did not reinstall or start the service
(the exercise uses the seam as it stands, and the node's
configuration was not touched). Journal sequences for
proposal / verdict / resolution / claim / outcome therefore
**do not exist for this exercise** — none were created.

Baseline state left exactly as found: the journal still tips
at seq 42; its one pending escalation is the 2026-11-11
policy-fix probe (`se_0007`, journal seq 38,
`hearth.restart` for imagine-v12, synthetic `caused_by` seq
900102), resting unresolved — listed read-only by the new
trigger during verification, and not resolved by this run:
resolving a probe that is not the exercise's own proposal
would be exactly the improvisation the protocol forbids.

**Observation (fork side, recorded, not acted on):** while
the seam is down the fork watcher keeps proposing —
`~/workspace/gaia-node/home/proposals.jsonl` has grown from
5 to 10 lines since 2026-11-11, with egress passes logging
proposals still pending and `egress.jsonl` unchanged at 5
records. On the seam's return, the five new proposals will
ingest and price on their merits (ESCALATE under the current
policy); nothing about that is claimed here beyond the
counts observed.

**Unblock (for the lead):** reinstall `si-seam-service` from
`~/workspace/gaia-node/si-seam-service.service` (the
documented manual recovery — the gaia-node watchdog does not
cover it), daemon-reload, enable, start; then re-verify the
§4.3 table. One wrinkle to settle at reinstall: the shipped
unit's `PYTHONPATH` points at `~/workspace/si97-review`, so
"active on current `main`" (§4.3) needs the unit pointed at
a current-main tree as part of the same act. After that,
this exercise re-runs from precondition verification; the
trigger and this protocol need no changes.

## 3. The label (§4.4, verbatim disposition)

> **§18.3 remains NOT earned.** This exercise did not
> complete — it aborted at its precondition gate, honestly
> recorded. No live escalation was driven, resolved, or
> executed by SI-113; no broader or narrower §18.3 reading
> is claimed. The honesty table does not move. A completed
> successor run of this same protocol earns exactly its own
> scope, stated in its own record, and nothing here
> pre-claims it.

---

## 4. SI-114 — the re-run (2026-11-13 ~16:36–16:47 EST)

After the lead reinstalled `si-seam-service` (unit now
running from the standing checkout `~/workspace/sentinel-main`
at main `5eeece87`, PR #122 merged), the exercise was
re-run once by the SI-114 coordinator under the lead's
dispatch. **Outcome: PARTIAL CLOSE** — the resolution half
of the walk is proven live; the execution half was not
observed. This section is the record; §4.7 states the
label disposition for exactly this outcome.

### 4.1 Protocol adaptations — stated plainly

This re-run is **not** the unmodified §4.3 walk. Two
deltas, both deliberate and recorded here so no reader
mistakes it for one:

1. **Subject.** §4.3 step 2 drives a fresh synthetic
   proposal. The re-run instead took as its subject a
   **real, fork-generated resting escalation** already in
   the journal: during the seam-down window the fork's
   watcher kept proposing, and on the seam's return five
   genuine `hearth.restart` proposals for imagine-v13
   (task 43, residency churn) ingested and priced ESCALATE.
   No synthetic proposal was needed — the journal already
   held real ones. The subject was selected by rule, not
   by hand: the highest-seq entry standing `escalated`
   with tool `hearth.restart` whose args name imagine-v13.
2. **Execution leg.** §4.3 step 5 names
   `execute_resolved_escalation`. As SI-113 §1 recorded
   (verified against the executor), the SI `Executor`
   refuses `hearth.restart` by construction — the registry
   keeps seam-crossing tools out of every live set. A
   resolved seam intent's execution leg is therefore the
   **fork's G-9 Flow 2 lifecycle** (verdict route → claim /
   actuate → S-5 outcome return), and the re-run's steps
   5–6 were an **observation** of that leg, not a call into
   the SI executor.

### 4.2 Preconditions (§4.3, re-verified before any act)

| Precondition (§4.3) | Result | Evidence |
| --- | --- | --- |
| Live gaia node Resident with Imagine v13, ctx pool 4096, parallel 1 | **PASS** | `GET 127.0.0.1:11435/gaia/v1/residency`: `imagine-v13` resident, `ctx_pool` 4096, `parallel` 1 (the `smollm2: resident` line is the known stale spine record). gaia argv carries `--binary ~/workspace/gaia-node/llama-server-v13.sh` (the locked template wrapper); the child (llama-server pid 2100, 127.0.0.1:41583) runs with `--jinja --chat-template-file ~/workspace/gaia-node/chat-template-v13.jinja` |
| Seam egress live: node passes `--si-addr 127.0.0.1:11437` | **PASS** | gaia process argv carries `--si-addr 127.0.0.1:11437` |
| `si-seam-service` active | **PASS** | `systemctl is-active si-seam-service` → active; `GET 127.0.0.1:11437/health` → `{"status": "ok", ...}` (bundle `gaia-seam`) |
| Allowlist exactly `hearth.notify` / `hearth.restart`, external at ESCALATE | **PASS** | Seam bundle policy (read via the seam's own `load_policy` + `resolve_tool_policy`): `allowed_tools == ['hearth.notify', 'hearth.restart']`; `tool_class_policy.external == 'escalate'` |
| Journal chain verification at baseline | **PASS** | The CLI's read path verifies the chain before folding (fail closed); fold succeeded |
| Baseline journal tip recorded | **RECORDED** | tip seq **68**, hash `sha256:b80ae938d3267e092b0beddd6b083785b3a75e1eadfcf384bc945aba5db869cd` |

### 4.3 Pre-state census (folded, chain-verified)

Resting escalations at baseline — six, all
`hearth.restart`, all verdict ESCALATE, all unresolved:

| Intent seq | Edit | Task | Model | State (before) |
| --- | --- | --- | --- | --- |
| 38 | se_0007 | 5 | imagine-v12 (synthetic spine seq 900102 — the 2026-11-11 policy-fix probe) | escalated |
| 44 | se_0008 | 43 | imagine-v13 (spine seq 509, `lost{evicted}`) | escalated |
| 49 | se_0009 | 43 | imagine-v13 (spine seq 538, `lost{evicted}`) | escalated |
| 54 | se_0010 | 43 | imagine-v13 (spine seq 547, `lost{evicted}`) | escalated |
| 59 | se_0011 | 43 | imagine-v13 (spine seq 556, `lost{evicted}`) | escalated |
| 64 | se_0012 | 43 | imagine-v13 (spine seq 565, `lost{evicted}`) | escalated |

The six older task-5 intents (se_0001–se_0006, seqs
6/11/16/21/26/31) fold `rejected` — the 2026-11-11
empty-allowlist era — and were not escalations. Node
residency at baseline: imagine-v13 resident, inference
child pid 2100 on 127.0.0.1:41583. Fork-side baselines:
`gaia.log` 18,970,685 bytes (mtime 21:08:34Z);
`home/egress.jsonl` 10 lines (mtime 21:08:34Z);
`home/mirror.jsonl` 452 lines (mtime 15:07:37Z);
`home/proposals.jsonl` 10 lines.

**Selected subject: se_0012, intent seq 64** — the
highest-seq resting escalation naming imagine-v13, per
the selection rule in §4.1. se_0007 was expressly not
selected (the synthetic v12 probe).

### 4.4 The resolution (the exercise's one operator act)

At **2026-11-13T21:37:22Z**, via the merged CLI trigger
(`si gaia-escalation <bundle> resolve 64`), operator
`SI-114 ESC-LIVE-1 exercise coordinator (lead-dispatched)`,
basis recorded on the act (SI-111 §4.3 live exercise
re-run; genuine fork-generated proposal from the
seam-down window). The act journaled exactly one row:

- **seq 69** — `escalation_resolution`,
  `resolution: admit-after-review`, intent seq 64,
  edit se_0012, task 43.
- **seq 70** — the trigger's manifest re-anchor
  checkpoint (the CLI's documented post-act step, as at
  seam ingest), noting the resolution.

The fold moved se_0012 `escalated` → **`queued`** —
the only transition out of `escalated`, carrying
`resolution_seq: 69`. Exactly one successful resolution
was performed in the whole run.

### 4.5 Observation of the fork leg (21:37:22Z → 21:47:22Z)

Watched continuously for the protocol's 10-minute
window: the seam journal fold, `gaia.log`, the node's
residency API, the fork SSE spine, and the fork's home
state files. Polls at 21:37:31Z, 21:38:39Z, 21:40:15Z,
21:43:03Z, and 21:47:13Z all read identically:

- Seam journal: tip unchanged at seq 70; se_0012 folds
  `queued`; **no claim row, no outcome row** — nothing
  journaled after the resolution.
- Inference child: pid **2100 throughout** — no restart,
  no new port; residency `imagine-v13: resident` at
  every poll.
- `gaia.log`: byte-identical at every poll
  (18,970,685 bytes, mtime 21:08:34Z — its last write is
  the backlog flush, ~29 minutes before the resolution).
- `home/egress.jsonl` (mtime 21:08:34Z),
  `home/mirror.jsonl` (mtime 15:07:37Z),
  `home/proposals.jsonl`: all unchanged.

**Where the trail goes cold:** at the fork's Flow 2
intake. The resolution is fully visible in the SI
journal — the same journal the fork's S-6 forwarder
mirrors — yet no fork surface shows any consumption of
seqs 69/70 within the window: no verdict-route traffic,
no claim, no actuation, no S-5 outcome return. Whether
the fork's resolution-consumption path is unbuilt,
unwired on this node build, or on a cadence longer than
the window is **not established by this record** — the
fork wrote nothing anywhere observable after 21:08:34Z,
so its internal state cannot be read from the outside
surfaces the runbook names. Recorded as a finding for a
successor slice; per the protocol it was not worked
around: nothing was hand-actuated, nothing was
restarted, no second entry was resolved.

### 4.6 Negative controls (§4.3 step 7)

Run deliberately after the observation window closed:

- **(a) Second resolution attempt — RUN, REFUSED as
  designed.** At 21:47:26Z the same CLI resolve was
  issued against intent seq 64. Exit 1: *"intent seq 64
  is already resolved ('admit-after-review' at seq 69):
  a second resolution would rewrite the operator's
  record (nothing journaled)."* Journal tip before and
  after: seq 70, hash
  `sha256:3d3b4fdeb44731c942b26180b5ec7b13ae728acf187d189afc6e40c6cab32df3`
  — unchanged. The single-resolution rule is proven
  live, not only in tests.
- **(b) Second execution attempt — NOT RUN, premise
  absent.** Control (b) presupposes an execution to
  repeat; none occurred. It also cannot be probed
  safely in the partial-close state: se_0012 still
  stands `queued`, so the one reachable execution
  handle (the SI-side `execute_resolved_escalation`)
  would not refuse — it would journal a claim and
  consume the resolution's single shot with the SI
  executor's by-construction refusal of
  `hearth.restart`, destroying the very state the
  control exists to probe and falsifying §4.5's
  finding. No verified non-actuating fork-side
  execution handle exists among the runbook's surfaces,
  and no speculative calls were made against the live
  node. Control (b) remains owed by whichever successor
  slice completes the fork leg; it is recorded here as
  outstanding, not as passed.

### 4.7 Post-state and the label (§4.4 disposition)

Post-state, verified 21:47:34Z: journal tip seq **70**,
hash
`sha256:3d3b4fdeb44731c942b26180b5ec7b13ae728acf187d189afc6e40c6cab32df3`;
**chain verification passes** over the whole journal;
se_0012 folds `queued` (`resolution_seq` 69); the other
five resting escalations (se_0007–se_0011) fold exactly
as found — escalated, unresolved; se_0001–se_0006
unchanged. Node: imagine-v13 resident (ctx pool 4096,
parallel 1), child pid 2100, both `gaia-node` and
`si-seam-service` active. Nothing else on the node was
touched.

> **§18.3 remains NOT earned.** The SI-114 re-run earns
> exactly this scope and no more: **one live escalation
> on the live node, resolved through a journaled
> operator act via the merged CLI trigger, folding
> `escalated` → `queued` on a chain-verified journal,
> with the single-resolution refusal proven live.** The
> remaining half of the §4.4 scope — *executed exactly
> once on the live node* — did not occur inside the
> observation window and is not claimed; no outcome was
> returned or journaled. The honesty table does not
> move on §18.3. A successor slice that completes the
> fork Flow 2 leg earns its own scope in its own record;
> nothing here pre-claims it.
