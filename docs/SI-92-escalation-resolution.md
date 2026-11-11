# SI-92 — Gaia escalation resolution: the operator's answer to a gate ESCALATE

**Status:** implemented on `si-92/escalation-resolution` against
SI-41 §5.4 (the operator-admission pricing for escalated tool
intents), SI-50 §5.2 / §12.4 rule 2 (the escalated state is
"the operator's to admit or refuse"), the SI-10
operator-admission precedent, and SI-53's §13.4 disposition
pattern. This slice builds the resolution path those texts
always described and no slice had built. §0 says plainly
what is live and what is not.

| Element | SI-92 state |
| --- | --- |
| §5.4 resolution entry | **live** — `escalation_resolution` experience journaled by `record_escalation_resolution` (source `operator`; operator, basis, task, edit id, intent seq, resolution, detail) |
| Resolutions vocabulary | **closed** — exactly `admit-after-review` / `reject` (`RESOLUTIONS`); no third value, and none that rewrites the gate's verdict |
| Verifying act | **live** — `resolve_escalation` folds first; unknown / non-escalated / already-resolved intents, invalid values, and an empty operator raise and journal NOTHING |
| Fold | **live** — the resolution annotates the intent; `reject` changes no state; `admit-after-review` is the one transition out of `escalated`, to `queued`, in journal sequence order (prefix folds unaffected) |
| The one execution | **live, miniature-scoped** — claim journaled before execution (§5.3); the executor verifies the resolution ticket from the journal itself, plus no genuine outcome already journaled; all §9.2 disciplines unchanged; the execution record carries `resolution_seq` |
| Drain path | **live** — a later drain claims the resolved intent under §7.1; `Drainer.execute_claim` passes the folded `resolution_seq` with the claim |
| Serial/task-loop path | **unchanged** — `execute_tool_cycle` still journals the non-execution outcome on ESCALATE and halts; resolution execution goes through `execute_resolved_escalation` (or the drainer), never through a fresh gate cycle |
| Gate behaviour | **unchanged** — the gate escalates exactly what it escalated; no policy key makes a resolution automatic; a resolution is an explicit operator act |
| Live deployment | **not claimed** — machinery at the existing miniature/embedded scope, exercised by this slice's tests; no live node was touched and no real-world escalation was resolved in building it |

---

## 1. What was built

### 1.1 The resolution act — `si/gaia/resolution.py`

SI-41 §5.4: "An ESCALATEd tool intent is journaled and
halted, and the operator's later admission of it — if a
B-slice builds one — follows the SI-10 operator-admission
precedent (admission materialises *the journaled intent*,
with the admitter and basis recorded), never a fresh,
ungated execution." SI-92 is that B-slice.

`resolve_escalation(journal, intent_seq, resolution, ...)`
is the act, shaped exactly like SI-53's
`disposition_outcome_unknown`: fold the journal, verify the
named intent is folded `escalated` and carries no prior
resolution, then journal. Every refusal — unknown intent,
non-escalated intent, **double resolution**, an invalid
resolution value (`mark-executed` and friends do not exist),
an empty operator — raises `ValueError` and journals
nothing. The journaled record carries the operator's name
and a basis: the caller's, or the pinned default stating
that the operator reviewed the journaled intent and its
ESCALATE report, that the verdict is not re-evaluated, and
that the resolution is a separate act. The self-edit keeps
its ESCALATE verdict forever; history is not rewritten.

### 1.2 The fold — `si/gaia/queue.py`

An `escalation_resolution` entry annotates its intent
(`resolution`, `resolution_seq`, `resolution_operator`,
`resolution_detail` — the first resolution in journal order
is the record). `reject` changes no state: the intent stays
`escalated`, terminal, and the annotation is the record of
its death. `admit-after-review` moves the intent from
`escalated` to `queued` — the only transition out of
`escalated` anywhere in the fold, driven by the journaled
operator act and applied in sequence order, so any prefix
of the journal still yields the state as of that prefix
(§5.1).

One fold rule needed care: an escalated intent already has
a journaled **non-execution** outcome (the shadow / live
record of its halting, which carries its `verdict` in the
record). When the resolution's one genuine execution later
lands, that outcome supersedes the non-execution outcome in
the fold's outcome fields — the record itself stays in the
journal. A *further* genuine outcome faults exactly as
before (§5.4 integrity): the supersede is scoped to
`resolution == "admit-after-review"` with a non-execution
record on the intent and a genuine record arriving, and to
nothing else.

`queue_summary` gains a `resolutions` mapping beside
`dispositions`; `queue_state_view` carries `resolution` /
`resolution_seq` per intent.

### 1.3 The one execution — `si/gaia/resolution.py`, `si/gaia/executor.py`, `si/gaia/drain.py`

`execute_resolved_escalation(...)` folds and refuses
(nothing journaled) unless the intent stands `queued`
*because* of an `admit-after-review` resolution: unresolved
escalations never execute, `reject`-resolved intents are
dead, and an intent already claimed or executed is not
re-run (§13: never blindly replay). It then journals a
**claim** (§5.3) and calls `Executor.execute_journaled` with
the claim and the resolution sequence as tickets.

The executor's verdict check is the enforcement point: a
journaled ESCALATE passes only when `_resolution_admits`
verifies, **from the journal itself**, an
`escalation_resolution` entry at exactly the passed
sequence naming this intent and resolving it
`admit-after-review`, and no genuine execution outcome
already journaled for the intent (the non-execution outcome
of the escalation does not count; any refusal or execution
record does — the resolution buys exactly one run, and a
direct executor call that journals a refusal spends it,
which the tests pin deliberately). Every other executor
discipline — journaled args, live set, miniature scope,
schema re-validation, TOCTOU re-pricing, work-root
containment — runs unchanged on the journaled intent. The
execution (or refusal) record carries `resolution_seq` as
its §5.4 provenance.

The drainer reaches the same path with no new machinery: a
resolved intent folds `queued`, is claimable under §7.1,
and `Drainer.execute_claim` passes the folded
`resolution_seq` alongside the claim when the executor
supports it (signature-checked, as the claim parameters
already were). A crash between the claim and the outcome
is §13's existing business: the claim folds
`outcome-unknown` at recovery and is never blindly replayed
— a resolution is not a replay licence.

## 2. Tests

`tests/test_gaia_resolution_si92.py` (8 tests): unresolved
escalations never execute (drainer, driver, fabricated
ticket); reject kills the intent, journaled with operator
and basis; admit-after-review executes exactly the
journaled args once, in resolution < claim < outcome
journal order, with `resolution_seq` on the record; the
drain path executes a resolved escalation and the summary /
state view carry the resolution; the executor runs an
escalated intent under a verified ticket; a wrong ticket
refuses and spends the one shot; double resolution refuses
and journals nothing; and the fail-closed battery (unknown
intent, ADMITted intent, REJECTed intent, invalid values,
empty operator) journals nothing at all.
