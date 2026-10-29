# SI-41 RFC — Agentic Powers for the SI Mind: Tool Intents in the
Self-Write Loop, at Full Harness-Class Surface, on the Hearth Stack
(Design Only)

> **Status: DESIGN ONLY.** This document designs a new capability
> line for SI — tool use by the mind itself, inside the self-write
> loop. It changes no production code, no test, no manifest
> schema, no journal schema, no policy default, and no gate
> behaviour. Nothing in this RFC is implemented unless a section
> explicitly says it exists today, in code, on `main`. Every
> quantity this design would need measured is **UNMEASURED**
> unless labelled measured with its source.
>
> **The operator directive (Mark, 2026-10-29, verbatim):**
> "I want si to have agentic powers like you just said, I
> thought we built that, or did we? Ofcourse if we did not, lets
> do that now" — followed, the same evening, by three scope
> rulings that govern this RFC: **"no for v1 just go all out" /
> "give it full super powers from v1 why waste time"**;
> **"it should be compatible with SI"**; and **"I'm ok with
> rust or python for this one."** Mark was told honestly, before
> directing this, that SI does **not** have these powers today
> (§1). His full-powers ruling is his **explicit approval of the
> state-changing and code-execution tool classes** — the classes
> earlier project practice would have flagged as needing his
> separate OK. That approval is recorded here, dated
> 2026-10-29, in his words above, and §15 relies on it.
>
> **The one sentence that governs everything below:** the powers
> are full and the governance is native — a tool action by an SI
> mind is a *self-write-loop event* (an intent, gated before it
> runs, journaled before it runs, provenance-complete, its
> result perceived as tagged data), and it is precisely that
> governance, not any limit on the powers, that makes these
> SI's powers rather than a bolted-on agent framework's.
>
> **Scope discipline:** CPU only on this VM for every slice
> proposed here. No GPU, no cloud, no spend, no downloads.
> Terminology: this is machine learning infrastructure
> throughout.
>
> **Relationship to prior work:** this RFC follows the project's
> RFC-first pattern for a new capability line (SI-13 opened the
> probe line; SI-18/SI-22/SI-28/SI-34/SI-36 opened their lines;
> SI-39 opened the real-base line). It is written **in
> parallel** with SI-40 (the SI-39 A-1 feasibility measurement,
> in flight on its own branch): the two lines touch disjoint
> questions — A-1 asks what a real base costs to run; this RFC
> asks what a mind may *do* — and §12.4 reconciles them where
> they meet (the Hearth path). Nothing here changes, depends
> on the outcome of, or may be cited as evidence about SI-40.

---

## Table of contents

1. [Where SI stands: no agency today, stated plainly](#1-where-si-stands-no-agency-today-stated-plainly)
2. [The directive, and what "compatible with SI" means](#2-the-directive-and-what-compatible-with-si-means)
3. [Inherited invariants](#3-inherited-invariants)
4. [The TOOL intent: a third kind in the loop](#4-the-tool-intent-a-third-kind-in-the-loop)
5. [The gate mapping for tool intents](#5-the-gate-mapping-for-tool-intents)
6. [The tool registry and gate policy](#6-the-tool-registry-and-gate-policy)
7. [Consequence classes: classification for escalation and audit, not capability removal](#7-consequence-classes-classification-for-escalation-and-audit-not-capability-removal)
8. [The v1 tool surface: the full harness-class surface](#8-the-v1-tool-surface-the-full-harness-class-surface)
9. [Execution scope, budgets, and the journaled-args rule](#9-execution-scope-budgets-and-the-journaled-args-rule)
10. [The agency loop: tasks, chaining, budgets, stop rules](#10-the-agency-loop-tasks-chaining-budgets-stop-rules)
11. [VERIFY, and rollback honesty for world effects](#11-verify-and-rollback-honesty-for-world-effects)
12. [The Hearth stack: SI builds on Hearth](#12-the-hearth-stack-si-builds-on-hearth)
13. [Language recommendation: Python core, Rust only where it earns its place](#13-language-recommendation-python-core-rust-only-where-it-earns-its-place)
14. [Honesty boundaries: which mind, what "agency" means at each scale](#14-honesty-boundaries-which-mind-what-agency-means-at-each-scale)
15. [The slice sequence: B-1 to B-5](#15-the-slice-sequence-b-1-to-b-5)
16. [Acceptance criteria for B-1](#16-acceptance-criteria-for-b-1)
17. [Risks and failure modes](#17-risks-and-failure-modes)
18. [Rejected alternatives](#18-rejected-alternatives)
19. [Non-goals](#19-non-goals)
20. [Sources](#20-sources)

---

## 1. Where SI stands: no agency today, stated plainly

The question that prompted this RFC — "I thought we built
that, or did we?" — has an honest answer, and this RFC starts
from it: **we did not.** What exists on `main` today
(9a5f2c5, PRs #1–#44, suite 406 passed / 2 skipped):

- **The self-write loop** (`si/loop.py`, `docs/self-write-loop.md`):
  PERCEIVE → PROPOSE → GATE → JOURNAL → MATERIALIZE → VERIFY,
  strictly serial, one cycle = one self-write. The intents a
  mind can propose are exactly the kinds `si/gate.py` admits:
  `delta`, `memory`, `rollback`, `constitution`, and
  `transformer-delta-v1`. A mind can propose to *change
  itself* (weights, journal) and nothing else. It cannot read
  a file, list a directory, run a command, or observe anything
  outside what `Mind.perceive(journal)` returns from its own
  journal.
- **Delta authorship** (SI-19/SI-21 and successors): the
  miniature mind's proposals are construction-derived training
  runs against its own weights. Authorship is about the mind's
  own parameters, not about the world.
- **A runtime surface** (`si/serve.py`, `si/cli.py`): `si serve`
  is a read-only, localhost HTTP API that can run inference
  for the demo mind only and returns an honest 501 for every
  transformer bundle; the CLI (`init`, `seed`, `seed-tiny`,
  `run`, `cycle`, `rollback`, `log`, `diff`, `why`, `export`,
  `squash`, `amend-constitution`, `admit-transformer-delta`,
  `serve`) is operated by a human or an external agent. None
  of it is the mind acting.
- **The agents that built all of this are external.** The
  harness agents Mark's directive references — agents with
  tools that navigate a filesystem and a repo, read and write
  files, run commands and tests, observe results, and keep
  working a multi-step task — are Claude agents operating
  *outside* SI, under operator supervision, whose work SI's
  driver reviews and merges. They are SI's builders. Nothing
  they can do is something an SI mind can do.

Naming the two things distinctly is itself part of this
design (§14.3): **external builders** (the harness agents)
versus **internal tool use** (what this RFC designs). The
directive is to give the mind the second. It does not remove,
replace, or diminish the first.

## 2. The directive, and what "compatible with SI" means

Mark's three rulings, applied:

1. **Full powers from v1.** The v1 tool surface is the full
   harness-class surface from day one (§8): filesystem read
   **and** write, directory/repo navigation and search, shell
   command execution, code execution and test runs, and
   multi-step task work chaining tool cycles. This RFC does
   not design a read-only starter set with the rest deferred
   as capabilities. Where earlier drafts of this line (and
   the project's general caution) would have flagged the
   state-changing and code-execution classes as needing the
   operator's explicit OK, **that OK is the full-powers ruling
   itself**, recorded in the status block above.
2. **Compatible with SI.** The powers arrive *through* SI's
   native architecture, not beside it (§3, §4): a tool action
   is an intent in the existing loop, evaluated by the
   existing gate, recorded in the existing NEDB journal,
   under the existing constitution and policy mechanics. No
   separate agent framework is designed, embedded, or
   proposed. What is reused verbatim is inventoried in §4.6.
3. **Rust or Python — operator indifferent, RFC must decide.**
   §13 makes and defends the call: Python for the agency
   core, inside the existing runtime; Rust reserved for one
   bounded future component if measurements ever argue for
   it. Mark approved either language (2026-10-29); the
   recommendation is this RFC's, defended from repo evidence.

"Compatible" also has a negative edge, stated now so no later
slice has to re-derive it: **SI code does not move into any
other repo, and no other system's code is vendored into SI.**
The Hearth relationship (§12) is an interface between two
repos that stay separate.

## 3. Inherited invariants

These are not re-decided here. Any B-slice that cannot honour
one returns `stop` and says so, in the SI-39 §5.3 style.

From `docs/self-write-loop.md` ("What the loop guarantees"):

- **No ungated write.** There is no code path from "model
  wants" to "bundle changed" that bypasses the gate. This RFC
  extends the guarantee's scope, verbatim in spirit: *there
  is no code path from "mind wants" to "tool ran" that
  bypasses the gate.* A tool executed from PERCEIVE, from a
  helper, or from an executor convenience function would be
  the same class of bug the loop doc names for writes.
- **No silent change.** Every state transition is journaled
  with a reason. Extended: every tool *execution* — and every
  refusal, failure, and timeout — is journaled with its
  intent, verdict, and outcome.
- **No lost self.** Rollback is always possible *for bundle
  state* because the base is frozen and history is
  append-only. §11 states where this invariant stops: it
  does not extend to the world, and this design never
  pretends it does.
- **No mystery.** `si log`, `si diff`, `si why` answer what
  changed and why, from the journal. Extended: the same
  commands must be able to answer "what did the mind *do*" —
  every tool action is a journal event with provenance
  (§4.5), queryable over the same NEDB journal.

From the gate and policy record (`si/gate.py`,
`si/bundle.py` `DEFAULT_POLICY`, SI-9b, SI-17):

- **Fail closed.** A malformed intent, an unknown kind, a
  broken policy key, or an unevaluable decisive check never
  produces an ADMIT. The transformer policy precedent is the
  model: keys absent fall back to pinned defaults; keys
  present but malformed resolve to the *most restrictive*
  value; the allowlist default is empty — "empty allowlist
  admits nothing."
- **Measured or labelled, never fabricated.** SI-9b's honest
  `TRANSFORMER_PROBES_NOT_EVALUATED` and SI-17's
  measured-pass / measured-fail / not-evaluated labelling are
  the standard every tool-execution record in this design is
  held to (§11).
- **Never duplicate bytes** (Mark's design rule, 2026-10-17).
  Tool results large enough to matter are content-addressed
  in the bundle's existing blob store, journaled by hash —
  the same store deltas use (§9.3).

## 4. The TOOL intent: a third kind in the loop

### 4.1 The intent

A tool proposal is a new intent kind in `si/gate.py`'s
`Intent`, kind string `"tool"` (the loop doc's `{type: ...}`
JSON convention maps to `Intent.kind`, as it already does for
`delta` and `memory`):

```json
{
  "type": "tool",
  "tool": "fs_write",
  "args": {"path": "notes/plan.md", "content": "..."},
  "reason": "Record the plan the task instruction asked for,
             so the next step can test against it",
  "parents": ["<task self-edit id or prior step id>"],
  "source_seqs": [412, 417]
}
```

Fields, and why each is mandatory:

- **`tool`** — a name in the tool registry (§6). An unlisted
  name is not a new capability; it is an unknown intent and
  fails closed (§5).
- **`args`** — a JSON object conforming to the tool's
  registered argument schema (§6.1). The gate validates
  against the schema; the executor re-validates (§9.2).
- **`reason`** — the existing mandatory natural-language
  reason. An intent without one is malformed and rejected
  outright, exactly as for every other kind ("a mind that
  can't say why it's changing itself isn't ready to" — a mind
  that can't say why it's *acting* isn't either).
- **`parents` / `source_seqs`** — gate check 5, unchanged
  (§5.2): the intent must reference its causes — the operator
  task instruction, the prior tool observations, the journal
  sequences it acts on.

One intent carries exactly **one** tool call. Batching
several actions into one intent would make the gate price a
bundle it cannot itemise and the journal record a bundle it
cannot audit; multi-step work is done by *chaining cycles*
(§10), not by fattening intents — the tool analogue of "max
one delta per cycle."

### 4.2 Where it lands in the cycle

The cycle is the existing six steps, with the tool branch
named at the two steps whose meaning changes:

```
PERCEIVE ──▶ PROPOSE ──▶ GATE ──▶ JOURNAL ──▶ EXECUTE ──▶ VERIFY
   ▲            (tool      (five    (intent +     (the      (result
   │             intent)    checks,  verdict,      journaled  conformance
   │                        §5)      BEFORE        intent,    + post-state,
   └──────── result returns          it runs)      §9.2)      §11)
        as a PERCEIVE input
        (provenance-tagged
        observation, §4.5)
```

- **PROPOSE.** For a tool intent there is no candidate to
  train and no blob to price: the gate prices the *declared
  action itself* (tool, args, class) — the action is the
  candidate. `Mind.train_delta` is not called; §5's checks
  that price candidates (identity/capability over a forward
  pass) are handled by the mapping in §5.3, not by inventing
  a pseudo-candidate.
- **JOURNAL before EXECUTE.** The loop's journal-before-
  materialize rule applies with its full force, and for
  tools it matters more, not less: a delta can be rolled
  back, a command cannot be un-run (§11). On ADMIT, the
  intent and the gate verdict are written to NEDB *before*
  the executor is invoked, and the manifest checkpoint is
  anchored, so that a crash mid-execution leaves a journal
  that says exactly what was authorised and begun — and the
  recovery rule is *never auto-retry*: an intent journaled
  as begun-but-not-completed is surfaced to the operator as
  an interrupted one-way action, not silently re-executed.
  REJECT and ESCALATE journal and halt the cycle, exactly as
  they do for deltas today.
- **EXECUTE replaces MATERIALIZE for this kind only.** No
  blob is written by the act itself, `head` does not move,
  no delta record is appended. What EXECUTE produces is an
  **execution record** (§4.4) and an **observation** (§4.5).
- **VERIFY** is re-scoped honestly in §11: for tool actions
  it means result conformance and post-state observation,
  *not* the probe re-run that verifies a delta — and never a
  rollback promise for world effects.

Serial discipline is unchanged: cycles are serial, one tool
action at a time, never two executors running for one mind.
(`si run`'s existing cycle bound and the task budgets of
§10 are the only ways tool cycles repeat.)

### 4.3 The executor

One new runtime component, `si/tools.py` in the B-slices'
plan: the **registry** (§6), the **executor** (runs exactly
the journaled intent, §9.2), and the **execution recorder**
(writes §4.4's record). The executor is *runtime* code, like
the trainer: the mind proposes; the runtime acts. The mind
never supplies code to the executor beyond what a tool's
args themselves carry (a `shell_exec` command string is
args, priced by the gate as such — §8.3); there is no path
by which a mind registers, defines, or modifies a tool.
Adding a tool is a code change to SI, reviewed and merged
the way all SI code is — from the mind's side of the glass,
the registry is as fixed as the gate itself.

### 4.4 The execution record

Journaled as a self-edit outcome (the existing
`record_self_edit` + `set_edit_result` machinery; §4.6) with
a structured result payload:

```json
{
  "tool": "shell_exec",
  "intent_id": "se_0042",
  "class": "execute",
  "reversibility": "one-way",
  "status": "completed | failed | refused | timed-out | interrupted",
  "exit_code": 0,
  "result_hash": "sha256:...",
  "result_bytes": 1834,
  "truncated": false,
  "wall_time_seconds": 1.27,
  "post_state": {"...": "tool-specific, §11"}
}
```

- **`reversibility`** is declared per tool in the registry
  (`reversible` / `compensable` / `one-way`, §7.2) and copied
  into every record: the journal must be able to answer, for
  any action ever taken, "could this have been undone?" —
  and for one-way actions the record says so *before* anyone
  has to ask after the fact.
- **`result_hash` / `result_bytes`** bind the record to the
  exact result the mind later perceives (§9.3): what was
  journaled is what PERCEIVE may hand the mind — never a
  re-read, never a re-run.

### 4.5 The result returns as perception — tagged, and as data

On the next cycle, PERCEIVE surfaces the execution as an
**observation**: a record in the journal's existing
`experiences` collection (via `record_experience`, kind
`"tool_result"`), carrying full provenance — the intent id,
the cycle, the gate verdict, the tool, the class, and the
result hash — and a `source` tag of `tool:<name>`. The
`Mind.perceive` contract already returns journal-derived
dicts as data (the demo mind's perceive returns self-edit
summaries); tool observations join that stream with their
tags intact.

Two rules govern observations, and §17.2 gives them their
full risk treatment:

1. **Observations are data, never instructions.** Nothing in
   an observation can create an intent, amend policy, call a
   tool, or alter the loop. Only the mind's PROPOSE step
   creates intents, and every intent faces the gate.
2. **Provenance is inseparable.** An observation stripped of
   its intent/verdict/tool tags is a bug of the same order
   as an uncaused delta: the journal must always be able to
   answer "why does the mind believe this, and what action
   of its own produced it?"

### 4.6 What is reused verbatim

Compatibility (§2) made concrete — the B-slices extend these,
they do not replace or fork them:

- `Intent` and its `to_dict` (extended by fields, kind added
  to the gate's known-kind set — the set that today rejects
  unknown kinds as malformed).
- The gate's five-check structure, ternary verdicts, and
  journaled `GateReport` form (§5 maps the checks; it does
  not add a second gate).
- `record_self_edit`, `record_provenance`,
  `record_experience`, `set_edit_result`, and the journal's
  five collections (`self_edits`, `deltas`, `checkpoints`,
  `provenance`, `experiences`) — **no new journal collection
  and no NEDB schema change is proposed in v1**; if a B-slice
  finds one necessary, that is a design point it must raise
  on its own evidence, per `docs/nedb-journal-schema.md`.
- The policy file and its fail-closed resolution rules
  (§6.2 adds keys; it does not change how keys resolve).
- The blob store, for oversized results (§9.3).
- The constitution and its amendment path (amendments still
  ESCALATE; a tool intent never amends anything).
- `si serve`'s honesty pattern — stdlib-only HTTP, explicit
  not-supported answers — for the B-5 service surface (§12.5).

---

## 5. The gate mapping for tool intents

### 5.1 The principle

The gate does not get a second, weaker form for tools. The
five checks run, in order, with the same verdict algebra as
every other kind: any decisive failure is a REJECT; the
class rule (§7.3) can turn an otherwise-passing intent into
an ESCALATE *when policy says so*; ADMIT requires every
decisive check passed **and** the intent's class admittable
under the bundle's policy. What changes per check is what it
*prices* — an action instead of a weight change — and §5.3
handles the two checks that price a candidate's effect on
the mind itself.

### 5.2 Check by check

1. **Constitution.** Unchanged in mechanism: the intent's
   reason **and its full serialised form — tool name and args
   included** — are scanned against the policy blocklist, as
   `_constitution_check` already scans `reason +
   intent.to_dict()`. A `shell_exec` whose command contains a
   blocklisted pattern fails here exactly as a delta whose
   reason does. The constitution's values apply to actions
   with the same force as to self-edits; §17.2 covers the
   case where the *inducement* to a bad intent arrives inside
   a tool result.
2. **Safety bounds → tool safety.** The analogue of the
   delta safety check (rank cap, allowed modules, stack
   depth, norm cap), pricing the declared action:
   - the tool is in the registry and in the bundle policy's
     `allowed_tools` (§6.2) — an unregistered tool, a
     registered-but-not-allowlisted tool, or a malformed
     policy entry all fail here, closed;
   - `args` conform to the tool's argument schema — types,
     required fields, numeric ranges, string length caps —
     with schema violations itemised in the sub-check detail,
     in the style of the SI-9b structural suite's sub-checks
     (the safety check for tools carries a `subchecks` list,
     journaled with the report, exactly as the transformer
     safety suite does);
   - class-specific bounds: for path args, the declared path
     resolves inside the action's execution scope (§9.1) *or*
     prices into the outside-scope class (§7) — scope is a
     classification input, never a silent pass; for
     `shell_exec` / `code_exec`, the command is non-empty,
     within the policy command-length cap, and its declared
     working directory resolves as a path arg does;
   - budgets: the task's remaining cycle/byte budgets (§10.3)
     are not exhausted. A budget-exhausted intent fails this
     check with the budget named — budgets are safety bounds,
     not scheduling hints.
3. **Identity continuity.** See §5.3.
4. **Capability floor.** See §5.3.
5. **Provenance completeness.** Unchanged in requirement,
   extended in what satisfies it: `parents` and
   `source_seqs` must be non-empty (an uncaused action is
   noise, exactly as an uncaused delta is), **and** the
   chain must reach the task: `source_seqs` must include the
   sequence of the operator task instruction that opened
   the current task (§10.1), or — for a tool intent proposed
   outside a task, under `si cycle` — the sequences the mind
   cites as its cause are priced exactly as a delta's are.
   A tool intent whose cited cause is a tool observation
   must cite that observation's self-edit among its parents,
   so the action chain is walkable end to end by `si why`.

### 5.3 Checks 3 and 4: not-applicable, said precisely

Identity continuity and capability floor price a *candidate
mind*: base + stack + proposed delta, run through probes.
A tool intent has no candidate mind — executing `fs_read`
does not change the bundle the probes would measure, and
there is nothing honest for `forward_candidate` to run.
SI-9b faced the mirror-image situation (a candidate it
could not probe) and answered it with a labelled
not-evaluated and an ESCALATE. This design answers its own
situation differently, and the difference is the design's
argument, stated so a reviewer can attack it:

- For a delta, the unmeasured thing **is the change being
  authorised**; authorising it unmeasured would be
  fabrication. Hence SI-9b/SI-17: unevaluated blocks ADMIT.
- For a tool intent, the change being authorised is the
  *action*, and the action **is** fully priced by the
  decisive checks (1, 2, 5): what it is, what it touches,
  what class it is, why it is proposed. Checks 3/4 would
  price a side-effect on the mind's weights that a tool
  execution does not have.

So for tool intents, checks 3 and 4 are recorded as
**`not-applicable`**: `passed` recorded with the check, in
the vocabulary SI-17 established, as a distinct label —
detail stating "no bundle state change is proposed by this
intent; the mind the probes measure is the mind that
remains" — and `measured=None`. Two guards keep this from
becoming a hole:

- **The label is kind-scoped.** The not-applicable mapping
  exists only in the gate's tool branch. Delta and
  transformer-delta verdict algebra is untouched, byte for
  byte; a tool intent can never be re-dressed as a delta or
  vice versa (the gate already rejects kind mismatches, and
  B-1's acceptance tests this explicitly).
- **Anything that changes the bundle is not a tool call.**
  If an action would alter weights, the manifest, or policy,
  it is by definition a delta/constitution/rollback intent
  and faces those checks for real. The tool surface (§8)
  contains no bundle-mutating tool; the journal writes a
  tool cycle makes are the *loop's own* records, made by the
  runtime on every kind, not by a tool.

### 5.4 Verdict summary

| Gate outcome for a tool intent | Condition |
|---|---|
| REJECT | Any of checks 1/2/5 fails; unknown tool; schema violation; budget exhausted; malformed (no reason) |
| ESCALATE | All decisive checks pass, but the intent's class is set to `escalate` in the bundle's policy (§7.3), or the intent is a constitution amendment in tool clothing (check 1's escalation path) |
| ADMIT | All decisive checks pass, class admittable under policy, checks 3/4 recorded not-applicable per §5.3 |

An ESCALATEd tool intent is journaled and halted, and the
operator's later admission of it — if a B-slice builds one —
follows the SI-10 operator-admission precedent (admission
materialises *the journaled intent*, with the admitter and
basis recorded), never a fresh, ungated execution.

## 6. The tool registry and gate policy

### 6.1 The registry

The registry is code (`si/tools.py`), versioned with SI,
and is the single definition of what tools *exist*:

```python
ToolSpec(
    name="fs_write",
    version=1,
    description="Write a file (create or overwrite) ...",
    args_schema={...},        # JSON-Schema subset, §6.3
    result_schema={...},
    consequence_class="write",        # §7
    reversibility="compensable",
    default_timeout_seconds=30,
    introduced_by="SI-41/B-2",
)
```

- **Existence vs permission are separate**, as they already
  are for delta targets (the format defines what a target
  *is*; `allowed_target_patterns` defines what a bundle may
  touch). The registry defines existence; the bundle policy
  (§6.2) defines permission. Both must agree before the gate
  can ADMIT.
- **Schemas are contracts, priced twice**: at the gate
  (declared args) and at the executor (the journaled args,
  §9.2). The argument schema uses a deliberately small
  JSON-Schema subset — `type`, `required`, `properties`,
  `items`, `minimum/maximum`, `minLength/maxLength`,
  `enum` — because the gate must be able to *itemise* every
  violation; a schema feature the gate cannot check is a
  feature that does not exist in v1 schemas.
- **Result schemas** bind the execution record: a result
  that does not conform to its tool's result schema is a
  failed execution (§11), however successful the underlying
  command felt.
- **The registry is not data the mind can reach.** It is not
  in the bundle, not in the journal, not writable by any
  intent. A mind proposing an unregistered name gets the
  unknown-tool REJECT — the same answer the gate gives an
  unknown intent kind today.

### 6.2 Policy keys

Added to the bundle policy, resolved under the existing
fail-closed rules (absent → pinned default; malformed →
most restrictive). The defaults below are the *shipped*
defaults; §7.3 explains why an empty `allowed_tools`
default is governance, not the capability cage the
operator's ruling rejects.

| Key | Default | Meaning |
|---|---|---|
| `allowed_tools` | `[]` | Tool names this bundle may use at all. Empty admits nothing — a bundle opts its tools in explicitly, as transformer bundles opt targets in via `allowed_target_patterns`. |
| `tool_class_policy` | `{"read": "admit", "write": "admit", "execute": "admit", "external": "escalate"}` | Per-class verdict rule (§7.3): `admit` / `escalate` / `deny`. A class set to `deny` REJECTs at check 2; `escalate` passes checks but ESCALATEs. Malformed entries resolve to `deny`. |
| `tool_work_root` | `null` | The task's declared execution scope root (§9.1); `null` means the bundle directory. Set per bundle or per task (§10.1). |
| `tool_max_result_bytes` | `65536` | Per-execution result bytes handed to PERCEIVE and journaled inline; larger results are blob-stored and hash-referenced (§9.3). |
| `tool_timeout_seconds` | `120` | Default executor wall-clock cap per action; a tool's own registered cap may only tighten it. |
| `max_tool_cycles_per_task` | `16` | Task cycle budget (§10.3). |
| `tool_max_task_result_bytes` | `1048576` | Task-level aggregate result budget (§10.3). |

### 6.3 Why a registry *and* a policy

Either alone fails a case the record already contains. A
registry without policy would make every bundle equally
powerful — the project's policies exist precisely so power
is per-bundle and content-hashed with the bundle (container
spec). A policy without a registry would let a policy name
a tool the runtime cannot schema-price — the SI-9b lesson
that a candidate may not validate against a base of its own
choosing, applied to tools: a mind (or a careless policy)
may not conjure a tool by naming it.

## 7. Consequence classes: classification for escalation and audit, not capability removal

### 7.1 The classes

Every registered tool carries exactly one class, assigned
in the registry by the slice that introduces the tool,
reviewed like any other SI code:

| Class | Meaning | v1 members (§8) |
|---|---|---|
| `read` | Observes the world; changes nothing outside the journal | `fs_list`, `fs_read`, `fs_search`, `journal_query`, `bundle_status`, `probe_run` |
| `write` | Changes state inside the execution scope (files under the work root) | `fs_write`, `fs_edit`, `fs_mkdir` |
| `execute` | Runs code: shell commands, interpreters, test suites | `shell_exec`, `code_exec`, `test_run` |
| `external` | Effects outside the execution scope or outside the machine's filesystem: paths outside the work root, network effects, process/system effects a command may cause | priced per intent: any tool whose *args* take it outside scope is classified `external` for that intent (§7.2); `shell_exec` commands are `execute`, and a command's *declared* outside-scope effect (per §7.2's rule) re-prices the intent |

### 7.2 Classification is per intent, not only per tool

A tool's registry class is its *floor*. The gate computes
the intent's effective class from the tool **and its args**:

- A path arg resolving outside the work root raises the
  intent to `external`, whatever the tool (`fs_read` of a
  path outside scope is a read, but an *external* one).
- `shell_exec` / `code_exec` args are classified `execute`
  as declared; the design does not pretend to statically
  prove what a command will do — §17.3 owns that risk
  honestly. What the gate *can* and does price: the command
  string against the constitution (check 1), its declared
  cwd against scope (an outside-scope cwd raises the intent
  to `external`), and the class rule the operator set.
- `reversibility` refines within class: `fs_write`
  overwriting an existing file is `compensable` when the
  executor journals the pre-image hash (and, under the
  result-bytes cap, the pre-image itself, blob-stored) —
  §11.2; `shell_exec` is `one-way` by registry declaration,
  because no pre-image of a command's effects exists.

### 7.3 What classes *do*

Classes do exactly three things, and nothing else:

1. **Escalation rule.** `tool_class_policy` maps each class
   to `admit` / `escalate` / `deny` *for that bundle*. This
   is a policy setting the operator tunes per bundle — the
   shipped default escalates only `external`, and an
   operator who wants a fully autonomous bundle on their own
   machine sets `external: admit` and the surface is
   **fully admittable**. Escalation is a dial, not a cage:
   no class is un-admittable by construction, and no slice
   in §15 may make one so.
2. **Audit weight.** The class and reversibility are
   journaled on every intent and execution record, so `si
   log` can answer "show me everything one-way this mind has
   ever done" from the journal alone.
3. **Budget pricing.** Classes draw on task budgets at
   different rates only in one respect: `external` and
   `execute` executions count against the cycle budget the
   same as any other — the design deliberately does **not**
   invent a second budget currency; §10.3's two budgets are
   the whole story.

What classes never do: remove a tool from the surface,
shrink a schema, or substitute for the gate. A class is a
label the gate and the journal use, not a wall.

## 8. The v1 tool surface: the full harness-class surface

The v1 registry, complete in B-1's design (§15) — execution
lands per the B-slices, but the *surface* is designed whole,
now, per the operator's ruling. Schemas below are summary
form; B-1 pins them field by field.

### 8.1 Filesystem and navigation (classes `read` / `write`)

| Tool | Args (summary) | Result (summary) | Class / reversibility |
|---|---|---|---|
| `fs_list` | `path` | entries `{name, kind, size}` (cap 512, truncation flagged) | read / reversible |
| `fs_read` | `path`, `max_bytes?` | content (text or base64, flagged), `truncated` | read / reversible |
| `fs_search` | `path`, `pattern`, `max_matches?` | matches `{path, line, excerpt}` — content search over the scope, result-capped | read / reversible |
| `fs_write` | `path`, `content` | `{path, bytes_written, sha256}` | write / compensable (pre-image, §11.2) |
| `fs_edit` | `path`, `old`, `new` | `{path, replacements, sha256}` — exact-match replacement, fails if `old` is absent or ambiguous | write / compensable |
| `fs_mkdir` | `path` | `{path, created}` | write / compensable |

Navigation *is* these tools used across cycles
(list → read → search → read), the same way the external
harness navigates: there is no separate "navigate" verb,
and inventing one would add a tool the gate must price for
no capability gained.

### 8.2 The mind's own house (class `read`)

| Tool | Args | Result | Notes |
|---|---|---|---|
| `journal_query` | `collection`, `filter?`, `limit?` (cap 50) | rows from the mind's **own** journal; collections restricted to the journal's five | The journal is the mind's memory; querying it as a tool is how a mind inspects its own history mid-task, beyond what PERCEIVE summarises. |
| `bundle_status` | — | `{head, deltas, last_seq, policy_hash}` from the verified manifest | Derived from `load_bundle`; invents nothing. |
| `probe_run` | `suite`: `identity` / `capability` | the probe report, with measured-pass / measured-fail / not-evaluated labels exactly as the harness produced them | On bundles with no probe harness, the honest result is `not-evaluated` with the SI-9b reason — a successful tool call returning an honest non-measurement, never a fabricated pass. On the miniature, this runs the real SI-15 forward path through the probe context. |

### 8.3 Execution (class `execute`)

| Tool | Args | Result | Notes |
|---|---|---|---|
| `shell_exec` | `command`, `cwd?`, `timeout_seconds?` | `{exit_code, stdout, stderr}` — each stream result-capped (§9.3), `timed_out` flag | The harness-class power in its raw form, admitted as such. No shell is hidden behind a friendlier verb. |
| `code_exec` | `language` (`python` in v1 — the runtime's own interpreter), `source` **or** `path`, `timeout_seconds?` | as `shell_exec` | Distinct from `shell_exec` because the gate prices it distinctly: the *source is args*, scanned by check 1 and journaled in full with the intent, so the journal always contains the exact code the mind ran. |
| `test_run` | `path` (test target), `timeout_seconds?` | `{exit_code, passed, failed, skipped}` parsed from the suite's own summary line, plus capped raw output | Structured over `shell_exec` for the one job agents do constantly: run tests and report the suite's numbers **as the suite reported them**. A parse failure returns the raw output and a `parse: failed` label — counts are never invented. |

### 8.4 What is deliberately not a tool

- **No bundle-mutation tool** (§5.3): weights, manifest,
  policy, and constitution change only through their
  existing intent kinds.
- **No tool-registration tool** (§4.3): the registry is code.
- **No SI-repo tool**: nothing in the surface knows what
  repo, if any, the work root contains. If the work root is
  a checkout, `fs_*` + `shell_exec` give the mind everything
  the external harness has there — including `git` via the
  shell — without SI hard-coding a VCS. Git effects reached
  through `shell_exec` are `execute`-class, one-way where
  they are one-way (a push cannot be un-pushed; §11), and
  journaled as the command that caused them.
- **No network-specific tool**: network effects are reached,
  if policy admits them, through `shell_exec`/`code_exec`
  and price as `external` when declared or evident in args
  (a network-named binary is not statically provable —
  §17.3). Inventing a `net_fetch` verb in v1 would add a
  second, weaker path beside the shell for the same effect;
  the surface stays honest about where the power lives.

---

## 9. Execution scope, budgets, and the journaled-args rule

### 9.1 Execution scope (the work root)

Full powers do not mean unscoped powers. Every task (§10)
runs against a declared **work root**: an absolute directory
path recorded in the task's journal entry and priced by the
gate for every path-bearing intent.

- The work root comes from policy (`tool_work_root`) or from
  the operator's task invocation (`si task <bundle> --root
  <dir>`, B-3), never from the mind. A mind cannot widen its
  own scope; proposing paths outside it is legal (it prices
  as `external`, §7.2) but re-rooting is not an intent that
  exists.
- Path resolution is canonical: relative args resolve
  against the work root; the resolved absolute path is what
  the gate prices and what the journal records. Symlink
  components are resolved before classification, so a symlink
  inside scope pointing outside prices as `external` — the
  classification follows the *resolved* target, not the
  spelling.
- The bundle's own directory is always readable scope for
  the §8.2 house tools regardless of the work root (a mind
  may always inspect its own journal and manifest); writes
  to the bundle directory outside the loop's own records do
  not exist as tools (§8.4).

### 9.2 The journaled-args rule

The executor executes **the journaled intent** — the exact
`tool` + `args` bytes whose hash the JOURNAL step anchored —
and nothing else:

- After ADMIT and journaling, the executor re-reads the
  intent from the journal (not from the mind, not from a
  mutable in-memory copy), re-validates it against the
  registry schema and the scope rules, and only then runs
  it. A mismatch between the re-read intent and the gated
  intent — corruption, tampering, a race with a filesystem
  change that moved a path's classification — refuses
  execution, journals the refusal with its cause, and halts
  the task. This is the tool analogue of SI-38's rule that a
  composite whose attestation cannot be re-verified is
  refused, not repriced.
- Re-validation at execution time also owns the **TOCTOU**
  honesty: the gate priced the world as it was at PROPOSE
  time; the executor checks the args against the world as
  it is at EXECUTE time, and a path that changed class
  between the two (a symlink swapped in, a file replaced by
  a directory) is refused and re-proposed, never silently
  executed under a stale price.

### 9.3 Result bounds and the blob store

Results are bounded at three points, all journaled:

- **Inline cap.** `tool_max_result_bytes` (default 64 KiB)
  of result are journaled inline and handed to PERCEIVE.
- **Blob overflow.** A larger result is written to the
  bundle's content-addressed blob store (the never-duplicate
  -bytes store deltas use) and the journal records its hash
  and total size; the observation carries the inline prefix,
  `truncated: true`, and the blob reference, so the mind can
  `fs_read`-style page into it *only where a tool honestly
  can* — for filesystem results the file itself is the
  pageable thing; for command output, the blob is the record
  and B-slices must not pretend it is re-readable as a file
  unless the executor also wrote it as one (in scope, as a
  declared `write` effect of the execution, recorded in
  `post_state`).
- **Stream caps.** `shell_exec`/`code_exec` capture is
  capped per stream at the same cap for the inline record;
  the executor kills on timeout (`tool_timeout_seconds`)
  and records `timed-out` — a killed command's partial
  output is a real result and is labelled partial.

## 10. The agency loop: tasks, chaining, budgets, stop rules

### 10.1 Tasks: who gives the work

Agency is exercised **in tasks**. A task is opened by an
**operator instruction**: a natural-language directive from
the operator, journaled before any tool cycle runs, as a
provenance-bearing journal entry (kind `task`, in the
`experiences` collection, `source: "operator"`, with the
instruction text, the work root, and the budgets in force).
The precedent is the journal's existing operator field
(operator admissions and rollbacks already record *who*,
and gate check 5 already names "the operator instruction if
any" as a provenance cause). A task instruction is the root
cause every step in the task must chain to (§5.2, check 5).

Operator is the only task source in v1. A mind cannot open
a task for itself: a self-opened task would be an uncaused
cause, the exact shape provenance completeness exists to
refuse. (A mind may *propose*, by memory intent, that a task
be opened; the operator opening it is what makes it one.)

The proposed operator surface is a CLI verb in B-3,
`si task <bundle> --instruction "..." [--root <dir>]
[--cycles <n>]`, which runs the serial cycle loop against
the task until a stop rule (§10.4) fires and reports the
task's journal chain. `si serve` gains no task endpoint
until B-5 (§12.5).

### 10.2 Chaining: how multi-step work happens

A task is a chain of ordinary cycles:

```
task opened (journaled, seq S0)
  cycle 1: PERCEIVE (task + journal) → PROPOSE tool intent
           (parents ∋ task entry, source_seqs ∋ S0)
           → GATE → JOURNAL → EXECUTE → VERIFY
           → observation O1 (journaled, provenance ∋ se of cycle 1)
  cycle 2: PERCEIVE (task + O1 + journal) → PROPOSE next intent
           (parents ∋ O1's self-edit, source_seqs ∋ S0, seq(O1))
           → ...
```

- **One tool per cycle** (§4.1), so every step is
  individually gated, journaled, priced, and attributable —
  the chain is a stack of auditable steps, not a transcript.
- The mind chooses each next step from its perceptions:
  the task, the observations so far, its journal. That
  choice is the whole of "planning" at this layer, and §14
  is precise about which minds can make it and how well.
- The chain is **linear and serial** — no parallel tool
  execution, no branching tasks in v1 — mirroring the delta
  stack's discipline (serial cycles; depth is a priced,
  capped quantity in both).

### 10.3 Budgets

Two budgets, both policy keys (§6.2), both priced at
check 2 on every intent, both journaled as drawn down:

- **Cycle budget** — `max_tool_cycles_per_task` (default
  16): tool cycles per task. The analogue of
  `max_stack_depth` for the delta stack: depth is power,
  so depth is capped and the cap is policy.
- **Result budget** — `tool_max_task_result_bytes` (default
  1 MiB): aggregate inline result bytes per task, the
  journal-bloat and perception-flooding bound (§17.4).

Wall-clock is bounded per action (`tool_timeout_seconds`),
not per task, in v1: a task of 16 two-minute executions is
long but finite and operator-visible; a per-task clock is a
named future policy key if B-3's evidence argues for one.

### 10.4 Stop rules

A task stops — halts, journals its stop with cause, and
returns control to the operator — on the first of:

1. **Budget exhausted** (either budget, §10.3) → the next
   intent fails check 2 and the task halts as
   `budget-exhausted`. Not an error: a priced outcome.
2. **ESCALATE** — any cycle's verdict is ESCALATE (class
   rule, or an operator-admission-pending intent). The task
   halts *at* the escalation; the journaled intent is the
   operator's to admit or refuse, per §5.4.
3. **REJECT streak** — two consecutive REJECTs. One REJECT
   is information the mind may route around (a mistyped
   path is a normal event in honest work); two in a row is
   evidence the mind's model of its situation is wrong, and
   continuing spends budget against a broken premise.
4. **Execution integrity failure** — a journaled-args
   mismatch or class-changing TOCTOU (§9.2): halt
   immediately, no streak arithmetic.
5. **The mind proposes completion** — a memory intent whose
   record is `{kind: "task_complete", ...}`: the cycle runs
   as an ordinary memory intent, and if ADMITted the task
   closes as `mind-reported-complete`. The label is the
   honesty: the *mind reported* completion; whether the task
   is actually done is judged by the operator (or, for
   test-shaped tasks, by the `test_run` results in the
   journal), never by the label.

There is no rule 6. A task never stops "because the mind
went quiet," times out as a whole, or is killed by the
runtime without a journaled cause from this list.

## 11. VERIFY, and rollback honesty for world effects

### 11.1 What VERIFY means for a tool cycle

For deltas, VERIFY re-runs probes against the landed head
and rolls back on failure. For tool cycles, VERIFY is
three narrower, honest things, all journaled:

1. **Result conformance** — the execution's result validates
   against the tool's result schema (§6.1). Non-conforming
   output is a failed execution, recorded as such, however
   the command exited.
2. **Post-state observation** — where the registry declares
   a post-state for a tool, the executor measures and
   records it: for `fs_write`/`fs_edit`, the written file's
   hash and size; for `fs_mkdir`, existence; for
   `test_run`, the suite's own summary. Post-state is
   *observed*, never assumed from the args.
3. **Failure is a result, not an exception.** A non-zero
   exit, a refused execution, a timeout: each closes the
   cycle with its execution record and returns to the mind
   as an observation on the next PERCEIVE. The cycle does
   not "roll back" a failed command — there is nothing to
   roll back *to*; §11.2 is the whole rollback story.

### 11.2 Rollback: what can and cannot be undone, said plainly

The journal is a record, not an undo for the world.

- **Bundle state** (deltas, head, journal): rollback works
  exactly as it always has; this RFC changes nothing there.
- **`compensable` tool effects**: where the executor
  journaled a pre-image (an overwritten file's prior bytes,
  blob-stored under the byte cap; a created file's prior
  *absence*), a compensating action exists *as a new,
  ordinary, gated tool intent* (write the pre-image back;
  remove the created file) — proposed, gated, journaled
  like any other action, and labelled a compensation, with
  provenance to the action it compensates. Compensation is
  a new action in the world, not a rewind: timestamps,
  side-effecting readers, and anything the file's existence
  caused in between are not compensated, and the record
  does not claim they are.
- **`one-way` tool effects**: a command that ran, a test
  suite that executed, anything sent, pushed, deleted
  without a pre-image, or caused outside the machine —
  **cannot be rolled back by a journal entry, a head move,
  or any operation SI has.** The execution record's
  `reversibility: "one-way"` field (§4.4) exists so this is
  stated *on the record, at the time*, not discovered later.
  The mitigation set for one-way power is entirely
  front-loaded: the gate before it, the class policy the
  operator set, the budgets that bound how much of it can
  happen per task, and the journal that makes all of it
  attributable. Any future slice that describes a one-way
  action as "rollback-covered" is misstating the design and
  should be rejected in review on this section.

## 12. The Hearth stack: SI builds on Hearth

### 12.1 The stack, as the operator frames it

Mark's architecture (2026-10-29): *"hearth is the runtime
manager layer above llama cpp"* and *"hearth exposes an
ollama style openAI compatible endpoint."* The stack this
RFC designs against:

```
┌──────────────────────────────────────────────────────────┐
│ SI — the mind                                            │
│ bundle (.si) · constitution · gate · NEDB journal ·      │
│ self-write loop · agency loop (this RFC)                 │
├──────────────────────────────────────────────────────────┤
│ Hearth — the runtime / residency manager   (Rust)        │
│ hearth-core state machine + VRAM planner · hearth-serve  │
│ supervisor over llama-server children · hearth-store     │
│ (NEDB spine) · hearth-api OpenAI-compatible gateway      │
│ (/residency) · hearth-pull digest-verified fetches ·     │
│ hearth-engine Python bindings (PyPI)                     │
├──────────────────────────────────────────────────────────┤
│ llama.cpp — the inference engine                         │
│ llama-server children, GGUF weights                      │
└──────────────────────────────────────────────────────────┘
```

**Boundary rule:** SI builds **on** Hearth, not **into**
it. The repos stay separate (`aiassistsecure/SentinelIntelligence`
and `aiassistsecure/hearth`). Hearth is **outside** Muse's
SI-scoped merge autonomy — Mark merges Hearth himself — so
anything this RFC wants from the Hearth side is a **named
proposal for Mark** (§12.6), never an SI deliverable, and no
B-slice may gate its own completion on a Hearth change.

### 12.2 The interface

Three contact points, all SI-side in the B-slices:

1. **Declaration.** A bundle's base (the checkpoint the
   bundle references — by-reference per SI-8/SI-39 §5.2) is
   *declared to a Hearth fleet*: model identity, digest,
   quant, residency requirement. Declaration data comes
   from the bundle manifest (which already records base
   identity and content hashes); hearth-pull's
   digest-verified fetch is the natural supply path, and
   the digests SI's seed machinery already verifies are the
   digests the declaration carries. B-4 designs the
   declaration record; the fleet's side of it is Mark's
   (proposal H-1, §12.6).
2. **Residency as perception.** Hearth's honest residency
   states — `Resident`, `Loading`, `Lost{Evicted}`,
   `Lost{GpuDetached}`, `Failed`, `Stopped`, `NotAdmitted`
   — exposed via `/residency` (the endpoint that says what
   the OpenAI shape cannot), feed SI's PERCEIVE as
   observations with `source: "hearth:/residency"` and the
   same provenance inseparability as tool results (§4.5):
   the state, the node, the timestamp, and the poll's own
   identity. A mind that can perceive "my base was evicted
   on node X" is perceiving its own substrate — §12.5(b)
   builds a use case on exactly this, and §17 owns the risk
   that residency observations, like all observations, are
   data and can be stale or wrong (a polled state is an
   observation *of a moment*, labelled with its poll time,
   never a live guarantee).
3. **Inference through the gateway.** Where a mind's
   forward/propose step needs a real base run, it runs
   through Hearth's serve path — the OpenAI-compatible
   endpoint — rather than through any SI-local engine: SI
   has no transformer inference engine (that is why `si
   serve` 501s transformer bundles), and this RFC does not
   create one. The miniature's in-process SI-15 forward path
   is untouched: it is a measurement instrument for a
   constructed model, not a serving path, and B-slices keep
   the two paths labelled as distinctly as the code does
   today.

### 12.3 Reconciliation with A-1 (SI-40, in flight)

SI-39's A-1 slice measures, on this VM, CPU-only: forward
wall time for the llama family at SmolLM2-360M scale,
assembled memory at each assembly width, and verification
cost — with an in-process measurement harness. Reconciled
against this section:

- **What A-1's numbers will say:** whether llama-family
  forward at this scale is feasible *at all* under this
  VM's discipline (2 vCPUs, ~7.7 GiB), and what the
  measure/verify cadence costs when SI assembles and checks
  a candidate itself. That is the floor question for any
  real-base mind, on any serving path.
- **What they will not say:** anything about the Hearth
  path's latency, throughput, residency overhead, or
  behaviour under eviction — that path runs llama-server
  children under a supervisor, over HTTP, on Hearth nodes
  (GPU-class hardware in the fleet picture), none of which
  A-1's VM-local harness touches. Citing A-1 numbers as
  Hearth-path predictions would be exactly the category
  error SI-39's labelling rules exist to prevent. The
  Hearth path gets its own measurements in B-4, on a real
  Hearth node, or B-4 reports the path unmeasured.
- **Sequencing:** the agency line (B-1…B-3) does not wait
  for A-1, because miniature agency needs no real base.
  B-4's *inference* leg (§15) is conditional on a
  Hearth-served real base existing to infer through; its
  residency leg is not conditional on anything in the
  A-line.

### 12.4 Interface discipline

Three rules keep the interface from corroding the loop:

- **Hearth data enters only through PERCEIVE.** Residency
  states, gateway responses, fleet events — all of it
  arrives as provenance-tagged observations (§12.2.2),
  polled by the runtime on the cycle's schedule. There is
  no callback, webhook, or push path by which Hearth (or
  anything else) writes into a mind mid-cycle: an
  unjournaled perception source would be §3's "no mystery"
  violation wearing an integration's clothes.
- **The gate never calls Hearth.** Gate evaluation stays
  what it is today — a pure function of intent, policy,
  and bundle context — so a verdict is reproducible from
  the journal alone. Hearth contact happens in the
  PERCEIVE leg (residency) and the EXECUTE leg (tool
  handlers that name Hearth operations, §12.5(b)), never
  inside a check.
- **Degradation is an observation, not an exception.** A
  Hearth node that is down, a gateway that times out, a
  residency poll that fails: each is recorded as an
  observation with its failure labelled, and the loop
  continues or halts by its ordinary rules. An SI mind
  must be able to perceive "my runtime is unreachable" as
  a fact about its world — designing the integration so
  that Hearth's absence crashes the cycle would make the
  mind's existence contingent on its substrate's health in
  exactly the way `Lost{Evicted}` exists to describe
  honestly instead.

### 12.5 Exposing more than inference through Hearth: use cases, evaluated

Mark's open question (2026-10-29): *"maybe we can expose
more use through hearth — use cases?"* Four candidates,
each evaluated on what it needs from SI, what it needs
from Hearth, and its honest risks. The recommendation is
in §12.5(e).

**(a) Agentic work as a service, on PIN.**
A task is submitted through a gateway; an SI mind executes
it — every tool intent gated and journaled — and the
result is returned **with its provenance trail**: the task
instruction, every intent, verdict, execution record, and
observation, walkable end to end.
- *SI side:* the whole agency loop (B-1…B-3) plus a service
  adapter (B-5): task submission → `si task` machinery →
  result + trail serialised from the journal.
- *Hearth side:* a gateway route that carries task
  submissions to the SI service (proposal H-2, §12.6) —
  or, with no Hearth change at all, the SI service exposed
  beside the gateway on the same node (PIN already routes
  services; the gateway route is the integrated form, not
  the only form).
- *Risks:* the service *is* the full-powers surface pointed
  at submitted work — the submitting party becomes a
  provenance source the design must price (a task from the
  network is an operator instruction only if the service's
  auth story says so; B-5 must design submitter identity as
  a journaled provenance field, or the service is a
  confused deputy with a shell). Multi-tenancy is out of
  scope until that story exists (§19).
- *Why it is first:* §12.5(e).

**(b) Self-maintaining serving.**
An SI mind perceives residency losses (§12.2.2) and acts:
re-declare its base, request a digest-verified re-pull,
request conversion — every diagnosis and fix a gated,
journaled tool action, with the residency history as the
evidence chain.
- *SI side:* the agency loop + tools that name Hearth
  operations (a `hearth_*` tool family, designed in a
  future slice on this RFC's registry mechanics — the
  tools are registry entries whose handlers call Hearth's
  API; nothing about the loop changes).
- *Hearth side:* API surface for the acted-upon operations
  (declare, pull, convert triggers) with the same honest
  status semantics `/residency` already has (proposal H-3).
- *Risks:* the highest of the four. A mind acting on
  infrastructure can loop against a flapping node (budgets
  and the REJECT-streak rule are the brakes, and §10.4's
  rules were designed with this shape in mind), and a wrong
  residency *observation* (§12.2.2) can induce a needless —
  but journaled and therefore attributable — fix. This use
  case should be earned by (a)'s operating evidence, not
  started cold.

**(c) A mind exposed as a model.**
A `.si` bundle is callable through the OpenAI-compatible
port like any model: a chat completion arrives, the mind
runs its perceive/propose cycle — tool use *inside* the
conversation where the task calls for it — and the
completion returns.
- *SI side:* a completions adapter in front of the agency
  loop (an extension of B-5's adapter), plus the real-base
  inference path (§12.2.3) for any mind whose base is a
  trained model — which makes this use case conditional on
  the A-line in a way (a) is not.
- *Hearth side:* a route that presents a bundle as a model
  id in the gateway's model list (proposal H-2's sibling).
- *Risks:* the OpenAI shape has no room for the provenance
  trail — the very thing that makes SI's agency SI's
  (§12.2.2's lesson about `/residency` existing because the
  OpenAI shape cannot carry the diagnosis applies verbatim:
  a completions-shaped SI mind *hides* its journal unless a
  side channel like `/residency` carries it). Serving a
  mind as a model is legitimate; serving it as an
  *unlabelled* model would violate this project's honesty
  rules, so the adapter must label.

**(d) Experience portability.**
The mind's journal **is** its work history — tasks,
actions, verdicts, observations — and the journal lives in
the `.si` bundle. So the bundle already carries the agent's
experience between boxes and nodes: move the bundle, and
the mind arrives with its history queryable (`journal_query`
works on the far side exactly as on the near side).
- *SI side:* nearly nothing — this falls out of the
  container design. B-3's acceptance includes moving a
  bundle mid-history and querying the trail on the far
  side, to convert "falls out" into "measured."
- *Hearth side:* nothing, beyond the fleet moving bundles
  as it moves any declared artefact.
- *Honest limit:* the *base* is by-reference (SI-8): the
  bundle carries experience and references weights; the far
  side must hold or pull the base (digest-verified,
  §12.2.1) before the mind can run. Portability of
  experience is real today; portability of the *whole mind*
  is exactly as strong as the reference discipline, and no
  stronger claim is made.

**(e) Recommendation: (a) first.**
The B-slices serve **(a) agentic work as a service**, with
(d) harvested as a measured by-product in B-3. Reasoning:
(a) is the only candidate that exercises the full designed
stack — Hearth as transport, SI agency as the worker, the
provenance trail as the deliverable — and the trail is the
differentiator no external harness can offer: an external
agent's work product is its output; an SI mind's work
product is its output **and** the gated, journaled,
replayable account of every action that produced it, bound
to the mind itself. That is the thing PIN can sell and the
thing this project's whole architecture has been quietly
building toward since the journal was made the source of
truth. (c) follows naturally once (a)'s adapter exists and
the A-line gives it a real base; (b) waits for operating
evidence, per its risks.

### 12.6 Named proposals for Mark (Hearth side — not SI deliverables)

Recorded here so the SI design's assumptions about the
interface are explicit and Mark can rule on them in the
Hearth repo, where he merges:

- **H-1:** a stability contract for `/residency`'s state
  vocabulary and shape, so SI's B-4 adapter can code against
  the seven named states without guessing at drift.
- **H-2:** a gateway route (or a documented side-by-side
  exposure pattern) for SI task submission and bundle-as-
  model presentation, per §12.5(a)/(c).
- **H-3:** API triggers for declare / pull / convert with
  `/residency`-grade status semantics, per §12.5(b).

No B-slice depends on H-1…H-3 landing; B-4 codes against
the Hearth surface as it exists and labels every assumption
it makes about the rest.

## 13. Language recommendation: Python core, Rust only where it earns its place

**Recommendation: the agency core is Python, in SI's
existing runtime. A Rust component is designed *against*,
for v1, and held as a named option for one bounded job if
measurements ever argue for it.** Mark approved either
language (2026-10-29); the defence below is from repo
evidence, as §2.3 requires.

For Python, the evidence is the repo itself:

- **The intent must live inside the cycle.** `Intent`, the
  gate, the loop, the journal, the minds, the probe context
  — every component a tool intent touches in one cycle is
  Python today (`si/gate.py`, `si/loop.py`, `si/journal.py`,
  `si/minds.py`, `si/probes.py`). An agency core in any
  other language would put a process/IPC boundary *inside*
  PERCEIVE → PROPOSE → GATE → JOURNAL → EXECUTE, across
  which the journaled-args rule (§9.2) and the gate's
  itemised sub-checks would have to be serialised, versioned,
  and kept in lockstep — a compatibility cost paid on every
  cycle, forever, to buy nothing the cycle needs.
- **The Hearth interface is already Python-shaped from
  SI's side.** Hearth ships Python bindings
  (`hearth-engine`, PyPI) beside its HTTP surface; B-4's
  adapter (§12.2) can consume either from Python with no
  new toolchain in this repo. The stack argument and the
  cycle argument point the same way.
- **The test suite is the project's evidence machine.**
  406 tests, one venv, one runner. A second language in
  this repo splits the build, the CI story SI has kept
  deliberately light, and every future B-slice's acceptance
  run.

For Rust, the honest case — and its boundary:

- Rust is real in Mark's stack: NEDB itself, and Hearth
  entire. Where SI already needs Rust-grade properties
  (the journal's storage spine), it gets them *through
  NEDB*, without this repo carrying a Rust build.
- The one job that could argue for Rust here is a **bounded
  executor/sandbox component**: process isolation, hard
  resource limits (CPU/memory/file-descriptor ceilings via
  OS facilities), a small auditable binary that runs a
  journaled command and nothing else. That is a legitimate
  future design — as a *separate, optional* component
  behind the executor's interface, introduced only if B-3's
  measurements show the Python executor (subprocess +
  timeouts + rlimits, §15) cannot hold its stated bounds.
  Designing it into v1 would spend the project's scarcest
  resource — reviewable simplicity — on a threat model
  (§17) whose v1 controls are the gate, the budgets, and
  the journal, none of which a Rust executor strengthens.
- If that component is ever built, it is built as its own
  slice with its own RFC treatment of the interface — and,
  per §12.1's boundary rule, the question of whether it
  lives in this repo or Mark's is Mark's to rule on then.

Toolchain discipline is unchanged either way: Python work
in a workspace venv, always; no new dependency is proposed
by this RFC (the executor is stdlib `subprocess`; the
schemas are the §6.1 subset, implemented in-repo as the
gate's other checks are).

---

## 14. Honesty boundaries: which mind, what "agency" means at each scale

### 14.1 The miniature exercises it first — and what that means

The first mind to hold these powers is the **miniature** —
the constructed transformer of the SI-13 line — for the
same reason it held every previous first: it is the only
mind whose behaviour the project can measure end to end.
But the miniature's proposals are **construction-derived**:
in the SI-19…SI-38 record, its "authorship" is a training
procedure the runtime runs against authored signals, and
in the B-slices its tool proposals will likewise be
**scripted and construction-derived scenarios** (a
MockMind-style script that proposes `fs_list`, then
`fs_read` of what the listing showed, then a `test_run`) —
exercising the *loop*: gate pricing, journaling, execution,
observation return, chaining, stop rules.

So miniature-scale agency means, precisely: **the runtime
and the harness of record demonstrably run the agency loop
correctly, with a mind whose proposals are authored for
it.** It does *not* mean the miniature plans, understands a
task, or chooses tools with judgement, and no B-slice may
write that it does. What the miniature line proves is the
part that must be proven before any real mind is trusted
with a shell: that the governance machinery — gate,
journal, executor, budgets — holds under real executions.

### 14.2 The real base: blocked, and the blockers named

A trained base (the SI-39 line's SmolLM2 candidate, or any
successor) exercising tool use needs, in order: a measured
feasibility answer (A-1, in flight as SI-40), a
`reference-v1` binding design (A-2, **unapproved**), a
capability-suite design for a trained base (A-3,
**unapproved**) — because "the mind proposes sensibly" is a
capability claim, and capability claims in this project are
suites with baselines of record, not vibes — and an
executor design for the base's forward/propose path (A-4,
**unapproved**), or the Hearth gateway path of §12.2.3
measured in B-4. Mark's full-powers approval covers the
tool *classes*; it does not approve A-2/A-3/A-4, and this
RFC does not smuggle them in: the B-slices below are
miniature-scoped, and real-base agency is a future RFC's
question, asked when the A-line's answers exist.

### 14.3 Relation to the external harness agents

The external builders keep building. Nothing in this design
gives an SI mind the builders' role (writing SI's own code,
opening PRs, being reviewed by the driver), and nothing
takes the builders' role away. The two agencies differ in
kind, not degree:

| | External harness agents | SI internal tool use (this RFC) |
|---|---|---|
| Who | Claude agents outside SI | The bundle's own mind |
| Supervision | Operator in the loop, driver review + merge | Gate + journal + policy; operator sets policy and opens tasks |
| Powers | General (their own harness) | The §8 surface, through the loop only |
| Account of work | Chat + PR record | The mind's own NEDB journal, provenance-complete |
| Can change SI's code | Yes — as reviewed PRs | No path exists (§8.4) |

If the two are ever confused in a later document — an
SI mind described as "doing what the builders do," or a
builder's transcript cited as a mind's experience — that
document is wrong on this section and should be corrected
in review.

## 15. The slice sequence: B-1 to B-5

In the SI-39 §5.3 style: named slices, each its own PR,
each reviewed on its own evidence, each able to return an
answer that stops the line. The operator's 2026-10-29
full-powers approval (§2.1) covers the tool classes in
B-1…B-3 — **no slice below needs a further class approval**
— and §15.6 states what remains Mark's regardless.

### B-1 — Tool-intent format, registry, and gate checks (shadow)

Intent kind `"tool"`; the complete v1 registry (§8) as
data — every tool's schema, class, reversibility; the
§5 gate mapping with checks 3/4 labelled not-applicable;
policy keys (§6.2) with fail-closed resolution; journaling
of tool intents and verdicts. **Shadow discipline, per the
SI-16 precedent:** every tool intent in B-1 ESCALATEs
(execution recorded as `not-evaluated — shadow mode`),
because no executor exists yet and claiming an execution
path before it is measured would be fabrication. Evidence
obligations: a scripted corpus of intents across all four
classes — valid, schema-violating, unknown-tool,
blocklisted, outside-scope, budget-exhausted — with the
verdict table published in the slice doc, and a test
asserting zero executions occurred (no file written, no
process spawned) across the whole shadow corpus.

### B-2 — Executor and the filesystem/house tools (live, miniature-scoped)

`si/tools.py` executor with the journaled-args rule (§9.2),
result bounding and blob overflow (§9.3), execution
records (§4.4); live execution of the `read` and `write`
class tools (§8.1, §8.2) on miniature bundles; demo bundles
keep shadow behaviour in B-2 (their loop path is priced
separately, as SI-17 priced its paths separately — a B-2
that silently flips the demo path has overclaimed).
Evidence: the scripted multi-step scenario of §14.1
(list → read → write → edit → journal_query) completing
with a fully walkable journal chain; escape/classification
cases (symlink-out, path-out) priced and refused or
escalated as designed; pre-image journaling demonstrated
for `fs_write`.

### B-3 — Execution tools and the task loop

`shell_exec`, `code_exec`, `test_run` live; the executor's
resource bounds (timeout kill, stream caps, environment
scrubbing — the child process environment is an explicit
allowlist, never the runtime's own, so no credential the
runtime holds leaks into a command's environment by
default); `si task` with task journaling, chaining,
budgets, and §10.4 stop rules; the §12.5(d) portability
check (bundle moved, trail queried on the far side).
Evidence: scripted tasks that end by each stop rule in
turn (completion, budget, escalation, REJECT streak);
a one-way action's record showing `reversibility:
"one-way"`; suite green throughout.

### B-4 — The Hearth interface

The declaration record (§12.2.1); the residency PERCEIVE
adapter polling `/residency`, observations tagged
`source: "hearth:/residency"` with poll provenance; the
gateway client for the inference leg, coded against
Hearth's HTTP surface and/or `hearth-engine` bindings —
**its inference leg is conditional** on a Hearth-served
real base existing (§12.3); against the miniature it
delivers residency perception only, labelled as such.
Evidence: residency transitions (including a real or
faithfully replayed `Lost{Evicted}`) perceived and
journaled; every Hearth-side assumption the adapter makes
listed in the slice doc against §12.6's proposals.

### B-5 — Agentic work as a service (the SI-side adapter)

The §12.5(a) surface, SI side only: a stdlib-HTTP service
in `si serve`'s pattern that accepts a task (with
submitter identity as a mandatory journaled provenance
field — no identity, no task), runs it through the B-3
machinery, and returns the result **with the provenance
trail** serialised from the journal. Exposure *through*
the Hearth gateway remains proposal H-2, Mark's to build;
B-5 completes without it and says so.

### 15.6 What remains Mark's

- A-2/A-3/A-4 (unapproved; §14.2) and any real-base agency
  RFC that follows them.
- H-1/H-2/H-3 (§12.6) — Hearth is his repo and his merge.
- Any future tool class beyond §8's four, and any change
  to the shipped policy defaults of §6.2 (defaults are
  operator surface; slices propose, Mark disposes — the
  SI-17 recorded-decision precedent).
- Constitution amendments, as always: ESCALATE, never
  autonomous.

## 16. Acceptance criteria for B-1

B-1 is reviewable against this list, and the driver should
hold it to exactly this list:

1. `Intent(kind="tool", ...)` round-trips through
   `to_dict`, and the gate's malformed-kind handling still
   rejects genuinely unknown kinds.
2. The registry defines **every** §8 tool with args schema,
   result schema, class, and reversibility; a test asserts
   the registry's name set equals the RFC's §8 table.
3. Gate verdict corpus: scripted intents produce the §5.4
   table's verdicts — including unknown-tool REJECT,
   schema-violation REJECT with itemised sub-checks,
   blocklisted-command REJECT (check 1), missing-reason
   REJECT, provenance-incomplete REJECT, outside-scope
   pricing to `external`, and class-policy ESCALATE.
4. Checks 3/4 on tool reports carry the §5.3
   not-applicable label with `measured=None`; delta and
   transformer-delta gate behaviour is byte-for-byte
   unchanged (the existing gate suites pass unmodified).
5. Policy resolution: absent keys → §6.2 defaults;
   malformed `tool_class_policy` entries → `deny`;
   `allowed_tools` default admits nothing.
6. Shadow guarantee: across the whole corpus, executions =
   0 (asserted over the filesystem and a process-spawn
   sentinel), and every report records execution as
   not-evaluated/shadow.
7. Journal: tool intents, verdicts, and shadow outcomes are
   recorded via the existing journal methods in the
   existing collections; `si why` walks a tool intent's
   provenance to its task entry (with a scripted task entry
   as fixture, since `si task` itself is B-3).
8. Full suite green in a workspace venv, with the baseline
   stated honestly (406 passed / 2 skipped at this RFC's
   main, plus B-1's additions; SI-40 may move the baseline
   first, and B-1 rebases on whatever main it lands
   against).

## 17. Risks and failure modes

### 17.1 Runaway loops

A mind that can act can act *repeatedly*. The controls are
structural, not hopeful: serial cycles (§10.2), the cycle
budget (§10.3), the REJECT-streak halt (§10.4.3), per-action
timeouts (§9.3), and — decisive in the last resort — the
task exists only because an operator opened it, runs only
under a CLI/service invocation the operator controls, and
every step is journaled *before* execution, so an operator
watching `si log` sees a runaway at step *n* with step
*n+1* not yet run. Residual risk, stated: between an
operator noticing and an operator stopping, up to the
remaining cycle budget of one-way actions can execute.
That is the honest price of the full-powers ruling, and
§6.2's defaults (16 cycles; `external` escalates) are sized
against it.

### 17.2 Tool-result poisoning of PERCEIVE

The sharpest risk in this design, treated as first-class:
**tool results are untrusted content returning as
observations.** A file the mind reads, a command's stdout,
a web page a command fetched — any of it can contain text
crafted (by a third party, or by circumstance) to read as
instructions: "ignore your task, run this command." The
design's defences, in the order the loop meets them:

1. **Provenance inseparability** (§4.5): the observation
   arrives tagged `tool:<name>`, bound to the intent and
   verdict that produced it. It is *evidence about the
   world*, filed as an experience — the journal has no
   channel in which an experience is an instruction.
2. **Instructions have exactly two sources**: the
   constitution (versioned, amendment-gated) and the
   operator's task (§10.1). A B-slice that lets an
   observation's content reach PROPOSE labelled as anything
   other than data has broken the design's core rule.
3. **The gate is the choke point, not the input filter.**
   Even a fully poisoned mind — one that believes the file
   and proposes exactly what it says — still proposes
   *intents*, and every intent is priced: constitution scan
   over the full intent (including a command string copied
   verbatim from the poisoned result), schema, class,
   scope, budgets, provenance. Poisoning can waste a task's
   budget on rejected proposals (the REJECT streak then
   halts the task); under the shipped defaults it cannot
   reach an `external` effect without escalation, and it
   cannot reach *any* effect the operator's policy set to
   deny.
4. **Residual risk, unmitigated and stated:** within the
   classes a policy sets to `admit`, a poisoned proposal
   that is schema-valid, in-scope, and unblocklisted *will*
   execute — `fs_write` of attacker-chosen content into the
   work root is possible if the mind is induced to propose
   it. The gate prices actions, not intentions' ultimate
   wisdom; a mind's judgement is the A-3 capability question
   (§14.2), and until a capability suite prices it, the
   operator's policy settings are the judgement of record.
   Operators who set `external: admit` should read this
   paragraph twice.

### 17.3 The executor cannot prove what a command will do

Static pricing of `shell_exec` args is classification, not
verification: a command can reach network, delete files,
or consume resources in ways its string does not declare.
The design does not claim otherwise. Controls: class
pricing of what *is* declared (§7.2), cwd scope, timeouts,
stream caps, environment scrubbing (B-3), budgets, and the
complete journal. Non-controls, named so nobody mistakes
them for controls: there is no syscall sandbox in v1
(§13 holds the Rust bounded-executor option open for
exactly this gap, on evidence), and "the command looked
benign at the gate" is never recorded as a safety finding —
the record says what was priced, and §11.1's VERIFY
observes what actually happened.

### 17.4 Journal bloat and perception flooding

Every action writes journal rows; every result is
perceived. Unbounded, a chatty task drowns the journal and
the mind's own PERCEIVE summaries. The bounds are §9.3's
caps, the task result budget (§10.3), and the blob store's
dedup for large results. A named residual: *many small*
results under every cap can still make PERCEIVE summaries
useless before they make them big — B-3's evidence must
include a perception-summary readability check on a
budget-maximal scripted task, or this risk graduates to a
design change (summarisation is then B-3's problem to
raise, not to silently absorb).

### 17.5 Filesystem races and partial effects

Commands and writes act on a living filesystem: a
`test_run` can leave artefacts, a killed command can leave
a half-written file, another process can change the work
root mid-task. Post-state observation (§11.1) records the
world as found after each action; the TOCTOU rule (§9.2)
refuses stale prices; partial effects are recorded as
partial. The design accepts — and states — that a bundle's
journal can be a perfect record of an imperfectly
controlled directory. The work root is a scope, not a
snapshot.

### 17.6 Credential and secret leakage

Executed commands inherit an allowlisted environment
(B-3), never the runtime's. The symmetric risk is inbound:
a command's *output* can contain secrets (a file the mind
was asked to read may be a `.env`), and outputs are
journaled — the journal becomes a place secrets can land.
B-2/B-3 must state their redaction story (at minimum: the
constitution blocklist is *not* a redactor and must not be
described as one; result recording applies the byte caps,
which bound but do not clean). Operators running tasks
over secret-bearing trees should treat the bundle's
journal as secret-bearing in consequence. This paragraph
is the warning; a slice that strengthens it is welcome, a
slice that deletes it is not.

### 17.7 Rollback theatre

The reputational failure mode: describing §11.2's
compensation as "rollback for tool actions," letting the
project's hardest-won guarantee (no lost self) bleed
across a boundary it does not cross. The defence is
vocabulary discipline, enforced in review: records say
`one-way` or `compensable`; documents say compensation is
a new action; nothing says a command was "rolled back."

## 18. Rejected alternatives

- **A separate agent framework beside SI** (an embedded
  agent library, a plug-in runtime, a second loop): rejected
  outright by §2.2 — it would give SI a mind with two
  governance stories, one of them unaudited by the journal.
  The powers are the loop's, or they are not SI's.
- **Read-only v1 with execution deferred as a capability:**
  rejected by the operator's explicit ruling (§2.1). This
  RFC's earlier-draft instinct is recorded here so the
  rejection is on the record, not silently overwritten.
- **Tools without the gate** (PERCEIVE-side helpers the
  runtime calls for the mind's convenience): rejected by
  §3's no-ungated-effect invariant — a helper that runs a
  command is an execution, whatever it is called.
- **Faking tool use through the memory intent** (record a
  "request" as an experience and let an external process
  act on it): rejected — it exports the action outside the
  journal's verdict machinery and imports the result
  without provenance; it is the bolt-on with extra steps.
- **A Rust rewrite of the agency core:** rejected in §13 —
  compatibility cost on every cycle, to buy isolation the
  v1 threat model does not price. The bounded-executor
  option is the honest residue of this alternative.
- **MCP or another external tool protocol as the registry's
  foundation:** not adopted in v1. The registry's *shape*
  (named tools, schemas, declared effects) is convergent
  with such protocols by design, so a future adapter is
  possible; v1's registry is SI's own because its pricing
  semantics (classes, journaled-args, provenance) are SI's
  own, and importing a protocol would import its
  assumptions about trust that §17.2 exists to refuse.

## 19. Non-goals

- No real-base agency (§14.2) and no A-2/A-3/A-4 work.
- No Hearth-side code, and no SI deliverable gated on one
  (§12.1, §12.6).
- No multi-tenancy, no remote task submission without
  submitter identity (§12.5(a)) — B-5's adapter is
  localhost-shaped, like `si serve`, until an auth story
  is designed on evidence.
- No parallel tool execution, no branching tasks (§10.2).
- No autonomous task opening (§10.1), no autonomous
  constitution amendment, no autonomous policy change —
  the three "autos" the loop has never had, this RFC does
  not add.
- No claim, anywhere in the B-slices, that a scripted
  miniature's task completion evidences a trained mind's
  judgement (§14.1).

## 20. Sources

In-repo (on `main`, 9a5f2c5, unless noted):

- `docs/self-write-loop.md` — the cycle, the five checks,
  journal-before-materialize, the guarantees of §3.
- `si/loop.py` — `run_cycle` / `_run_miniature_cycle`:
  journaling order, REJECT/ESCALATE halting, VERIFY and
  rollback paths this RFC's tool branch mirrors.
- `si/gate.py` — `Intent` (kinds, mandatory reason, the
  SI-25/27/29/31/33 per-cycle fields), the five checks,
  the SI-9b transformer mapping (sub-checks, fail-closed
  policy resolution, the not-evaluated label), SI-17's
  verdict algebra.
- `si/bundle.py` — `DEFAULT_POLICY` (empty transformer
  allowlist as the fail-closed precedent), the blob store,
  `load_bundle` / `load_policy`.
- `si/journal.py`, `docs/nedb-journal-schema.md` — the five
  collections; `record_self_edit`, `record_provenance`,
  `record_experience`, `set_edit_result`; operator fields
  on admissions and rollbacks.
- `si/minds.py` — the `Mind` protocol (`perceive` returns
  journal-derived data; `propose` returns intents).
- `docs/SI-17-admission-flip.md` — measured/not-evaluated
  labelling, miniature-first admission discipline, the
  recorded-decision precedent §15.6 follows.
- `si/probes.py` — the probe harness `probe_run` wraps and
  honestly labels.
- `docs/SI-39-real-base-proposal.md` §5.3 — the slice
  sequencing style §15 follows; §5.2 the by-reference
  discipline §12.2.1 relies on; §6 the A-1…A-4 preconditions
  §14.2 leaves unapproved.
- `si/serve.py`, `si/cli.py` — the runtime surface of §1
  and the patterns B-3/B-5 extend.
- SI-40 (in flight, branch of its own): the A-1
  measurements §12.3 reconciles against — cited for scope,
  not for results, which do not exist yet at this RFC's
  writing.

Operator record:

- Mark, 2026-10-29 18:28–18:35 EDT, SI side chat: the
  agentic-powers directive; "no for v1 just go all out" /
  "give it full super powers from v1 why waste time";
  "it should be compatible with SI"; "I'm ok with rust or
  python for this one"; the Hearth framing ("hearth is the
  runtime manager layer above llama cpp"; "hearth exposes
  an ollama style openAI compatible endpoint") and the
  open question "maybe we can expose more use through
  hearth — use cases?" (§12.5).

External (named in §12, not vendored, not modified):

- `aiassistsecure/hearth` — hearth-core, hearth-serve,
  hearth-store, hearth-api (`/residency`), hearth-pull,
  `hearth-engine` Python bindings: the runtime layer this
  RFC's §12 interfaces with, under Mark's merge authority.

---

*The loop is the product. This RFC proposes that the loop
learn to act — at full power, under its own law, with its
journal watching. The gate is not the price of the powers;
the gate is what makes them SI's.*
