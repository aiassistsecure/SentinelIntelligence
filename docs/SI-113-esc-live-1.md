# SI-113 — ESC-LIVE-1: CLI approval trigger and the live escalation exercise

**Status: the trigger is BUILT; the live exercise ABORTED at
the §4.3 precondition gate — no proposal was sent, nothing
was resolved, nothing executed.** The abort is the protocol
working as designed (§4.3: "if any precondition fails, stop —
no proposal, no improvisation"). The failed precondition is
an environment fact, recorded below with its evidence and
its unblock. §0 states the label disposition plainly.

| Element | SI-113 state |
| --- | --- |
| CLI approval trigger (§4.2) | **built** — `si/gaia/escalation_cli.py`, wired as `si gaia-escalation <bundle> list\|show\|resolve`; transport only, resolution recorded exclusively through the existing `resolve_escalation` act (PR #99) |
| Trigger read path, live | **exercised read-only** against the live seam journal during precondition verification (fold + chain verification; journaled nothing) |
| Live exercise (§4.3 steps 1–8) | **NOT RUN — aborted at preconditions**: `si-seam-service` inactive (see §2). No proposal, no resolution, no execution, no negative controls |
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
