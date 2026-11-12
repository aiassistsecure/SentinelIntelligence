# SI-97 — §9 Intra-Document Dependency Amendment (SPEC)

## The honest headline

This is the spec slice the lead commissioned in D-96-1 (SI-96 §9, decided
2026-11-12): §9 gains an intra-document dependency form, **spec-first**. It
decides every sub-question SI-96 §5.5 carried, as rules an implementation
slice (SI-98) will be built from — not a menu; where a genuine fork
remained it takes the conservative branch and says why (§2's table is the
five answers; the body is their justification). The decided shape: a
`depends_on` entry may name an **earlier sibling intent in the same
document** by 0-based position under a dedicated key —
`{"sibling": k, "requirement": …}` — a vocabulary disjoint from journal
seqs by construction. Parse time prices the form document-locally (stage 4
gains sibling checks; the document version bumps to **2**). The serial
admit walk resolves each sibling entry to the journal seq the walk itself
assigned the sibling, **before the gate**, and the gate prices only the
resolved form under its existing §6.2 rules — no gate change. The journal
records both the authored form and the resolved seq (§8). A dependent
whose sibling was never journaled by its walk is **refused at
resolution** — a pre-gate, per-member refusal in the §9.5 family,
journaled with its cause, never queued, never admitted with the
dependency dropped. **Nothing here is built or measured**: no parser,
walk, gate, or harness code changes in this slice, and SI-94's verdict —
**BELOW THRESHOLD** under D-C1 — stands exactly as recorded.

## Table of contents

- [0. Honesty table — no row changes](#0-honesty-table--no-row-changes)
- [1. Fingerprint](#1-fingerprint)
- [2. Authority, and the five answers](#2-authority-and-the-five-answers)
- [3. The amendment in one page](#3-the-amendment-in-one-page)
- [4. The intra-document form](#4-the-intra-document-form)
- [5. Parse time — stage 4 under version 2](#5-parse-time--stage-4-under-version-2)
- [6. Resolution at journal time](#6-resolution-at-journal-time)
- [7. The gate's view — post-resolution](#7-the-gates-view--post-resolution)
- [8. Journal shape — authored and resolved](#8-journal-shape--authored-and-resolved)
- [9. Backwards compatibility](#9-backwards-compatibility)
- [10. Abuse cases, repriced against the decided rules](#10-abuse-cases-repriced-against-the-decided-rules)
- [11. The implementation slice (SI-98)](#11-the-implementation-slice-si-98)
- [12. Re-measurement under this amendment](#12-re-measurement-under-this-amendment)
- [13. What this spec does not do](#13-what-this-spec-does-not-do)

## 0. Honesty table — no row changes

| Row | State, unchanged by this slice |
|---|---|
| SI-63 §18.3 claim | **NOT earned** |
| SI-63 §18.1 rows P5-R1, P5-R9 | **UNMEASURED** |
| SI-63 §18.1 rows P5-R2–R8, R10 | As promoted, scoped, by SI-74; cited, not re-derived |
| Live model-authored tool intent | **OCCURRED, scoped to SI-74 only** (2026-11-02, P-5 live run only). This slice extends that scope by nothing: it makes no live attempt of any kind |
| SI-94 verdict (D-C1) | **BELOW THRESHOLD** — stands exactly as recorded in `docs/SI-94-c2-conformance-record.md`; not re-cut, re-scored, or re-based here |
| SI-92 record | **INSUFFICIENT** at 27/30 — stands; a serving-configuration artifact per SI-93, not a candidate result |
| SI-79 verdict (v12) | **BELOW THRESHOLD** at 10/30 — stands |
| C-4 for v13 | **Does not proceed** on the SI-94 record |
| L-3 residency-loss drill | **LOCKED** |
| A-2 / A-3 / A-4 | **UNAPPROVED** |
| Readiness unlocks | `[]` |
| D-F1 classes opened | Perception assembly and rendering only (SI-91 §5). §4.2 catalogue rendering, §4.3 standing instructions, §4.4 output-contract instruction, §4.5 gateway-request assembly: **NOT opened**. §12.3 states the consequence for re-measurement |
| Contract version priced by every existing record | §8 version **1**. This spec defines version 2 on paper; no record prices it until an implementation and a fresh corpus exist |

The served model is a specialised machine learning model. This slice is
machine learning infrastructure throughout: a design document, written
offline, that ran no parser, no harness, no capture, and no live call.
Its claims about code are read from source at the fingerprint commit
(§1) and cite functions and spec sections, not line numbers.

## 1. Fingerprint

| Field | Value |
|---|---|
| SI commit | `1017b0c690de90326055cdaf832f4ff3649c8ca1` (main, PR #104 / SI-96 merged) |
| Inputs | `docs/SI-96-reference-formation-rfc.md` (§5 in full, §2–§3, §7–§9), `docs/SI-95-failure-taxonomy.md` (§4), `docs/SI-94-c2-conformance-record.md`, `docs/SI-91-decision-record-and-v13-conformance-plan.md` (§5/§6), `docs/SI-63-gaia-propose-hearth-proposer-rfc.md` (§6/§8/§9) — all on main, read only |
| Code read at the same commit | `si/gaia/proposer.py` (`parse_proposal` stages 1–6, `_task_intent_seqs`, `run_replay_proposal`, `build_response_record`, `walk_attempt_chain`), `si/gaia/drain.py` (`admit_tool_intent`, Q-2 §7.3 dependency semantics, `_dep_satisfied`), `si/gate.py` (`_normalize_depends_on`, §6.2 check 2), `si/gaia/queue.py` (`_normalise_depends_on`, fold states) |
| Authority | Lead decisions D-96-1…D-96-4, recorded 2026-11-12 in `~/workspace/si-dev-loop/STATE.md` under the 2026-11-11 directive (methodology calls are the lead's; SI-91 §2) |
| Candidate under discussion | Imagine v13, corrected-baseline serving (SI-94 §1) — discussed, not measured |
| Date | 2026-11-12 (America/New_York) |
| Mode | offline / design only — one document, no code |

## 2. Authority, and the five answers

D-96-1 answered SI-96's expressiveness question **YES**: §9 gains an
intra-document dependency form, commissioned spec-first. (D-96-2 held
Option A; D-96-3 set fold-supplied scoring for future corpora, with the
B-line re-measurement keeping no-fold scoring; D-96-4 commissioned the
C2 diagnostic after this spec.) This document is that spec. SI-96 §5.5
carried five sub-questions; each is answered with a decided rule:

| # | Sub-question (SI-96 §5.5) | Decided answer |
|---|---|---|
| SQ1 | The form choice (§5.1) | **Positional index**, dict-only, under the dedicated key `sibling`: `{"sibling": k, "requirement": …}`, k the 0-based position of a strictly earlier intent in the same document's `intents` array (§4) |
| SQ2 | The resolution-failure pricing (§5.2) | **Per-member refusal at resolution** (§6.4): the dependent is refused before the gate, journaled with its cause, never queued. Whole-batch refusal (candidate 2) is rejected — earlier members are already journaled and un-journalling is not an operation this programme has. Admit-with-dependency-dropped (candidate 3) remains out of bounds, refused by construction (§6.4, §10) |
| SQ3 | Version bump vs union widening (§5.3) | **Version bump**: §8 document version **2** (§9.1). The entry union widens *inside* v2 only; v1's schema is untouched |
| SQ4 | The gate's pre-/post-resolution view (§5.3) | **Post-resolution** (§7): the gate prices only resolved `{seq, requirement}` entries under its existing §6.2 normalisation. The gate changes by zero lines |
| SQ5 | The §14.3 journal shape (§5.3) | **Both forms journaled** (§8): the intent record carries the effective resolved `depends_on` and a verbatim `authored_depends_on`; the batch chaining summary gains the third authored kind, `sibling_resolved`; resolution refusals journal their own record kind |

Each answer takes the conservative branch where a fork remained:
SQ1 because the positional form makes cycles, forward references, and
duplicate designators impossible or parse-refused with **no new
machinery**; SQ3 because §9.2's own pinned rule is that forward
compatibility is by version bump, not tolerance; SQ4 because one pricing
of references, against journal facts, in one place — the gate — is the
existing architecture, and a second pricing would create two authorities
for one fact.

## 3. The amendment in one page

- **Form.** A `depends_on` entry in a version-2 document is one of:
  a bare positive int (a journal seq, `outcome` requirement — unchanged);
  `{"seq": n, "requirement": r}` (unchanged); or
  `{"sibling": k, "requirement": r}` (new), naming the intent at 0-based
  position k of the same document's `intents` array. A bare int **never**
  means a sibling. An entry naming both `seq` and `sibling` is refused.
- **Parse.** Stage 4 prices sibling entries document-locally: dict form
  only, integer non-bool k, k ≥ 0, k strictly less than the citing
  intent's own position, no duplicate k per intent, requirement in
  {`outcome`, `success`}. Violations refuse the whole batch (§9.2),
  exactly like seq-form defects today — with distinct detail classes so a
  taxonomy can split them (§5).
- **Resolution.** The admit walk, in document order, keeps a per-walk map
  from position to the journal seq it assigned that member; at a
  dependent's turn each sibling entry is rewritten in place to
  `{"seq": <assigned seq>, "requirement": r}`. Resolution is
  **verdict-blind**: a REJECTED or ESCALATED sibling was still journaled
  and holds a seq, so the reference resolves; what the verdict means
  for execution is the queue's existing §7.3 semantics, unchanged (§6.3).
- **Gate.** Sees only resolved entries; §6.2 pricing untouched (§7).
- **Journal.** The intent record's `depends_on` is the resolved list;
  `authored_depends_on` preserves the model's entries verbatim (§8).
- **Failure.** If the walk reaches a dependent whose sibling has no
  assigned seq in this walk, the dependent is **refused at resolution**
  (§6.4): one journaled refusal record with a named cause, no gate
  evaluation, no queue entry, no execution — and no path exists that
  admits it with the dependency dropped.

## 4. The intra-document form

### 4.1 Syntax — the v2 entry union

Under §8 document version 2, a `depends_on` entry is exactly one of:

| Form | Meaning | Status |
|---|---|---|
| `n` (bare int) | Journal seq n, requirement `outcome` | unchanged from v1 |
| `{"seq": n, "requirement": r}` | Journal seq n, requirement r ∈ {`outcome`, `success`} (`requirement` optional, default `outcome`) | unchanged from v1 |
| `{"sibling": k, "requirement": r}` | The intent at position k of this document's `intents` array, requirement r (optional, default `outcome`) | **new in v2** |

Rules of the form:

- **Disjoint vocabularies.** The sibling designator lives under its own
  key, `sibling`, never under `seq`, so the two can never be confused by
  the parser, the gate, the fold, or a human reader — SI-96 §2.3's
  three-way ambiguity has no representation in v2. The asymmetry is
  load-bearing: journal seqs are positive (seq 0 names no row in any
  journal), while sibling **0 is valid** and names the first intent.
  Any shared key would make one of those two facts a lie.
- **Dict-only.** A sibling reference exists only in the dict form. A bare
  int in `depends_on` is a journal sequence, full stop, in v1 and v2
  alike — giving bare ints a positional reading in multi-intent
  documents would recreate the ambiguity this amendment exists to
  remove.
- **Exactly one designator.** A dict entry carrying both `seq` and
  `sibling` is not a third form; it is refused (§5). An entry carrying
  neither is refused, as today.
- **Closed schema.** The v2 entry field set is
  {`seq`, `sibling`, `requirement`}. Any other key fails stage 2, as
  unknown fields do today (§9.4 structural ignoring is untouched: no
  value in a sibling entry is copied into a verdict, a consequence
  class, or a policy field).

### 4.2 The index basis

- k is the **0-based position in the `intents` array as written** in the
  one document that is parsed and then walked. Positions count every
  member of the array, in document order; there are no other members —
  a document carrying a `task_signal` carries no intents (§8.7/§8.8,
  stage 6 unchanged), so no position ever names a signal, a task row,
  or anything outside the array.
- Parse and the admit walk consume the **same array**: the walk iterates
  the parser's constructed intents in document order; no re-serialisation
  intervenes in which positions could shift. (§11 makes walk-order ==
  document-order a test obligation, not an assumption.)

### 4.3 Order, cardinality, depth

- **Strictly earlier.** Every sibling entry must satisfy k < i, where i
  is the citing intent's own position. Self-reference (k == i) and
  forward reference (k > i) are parse refusals (§5). It follows that
  **cycles are impossible by construction**: every dependency edge
  points to a strictly smaller position, so document order is already a
  topological order of the batch's dependency graph. No cycle detector
  is specified, and SI-98 must not build one — there is nothing for it
  to detect.
- **Fan-in is expressible.** An intent may carry several sibling
  entries naming distinct earlier positions — the read–read–then-write
  shape the predecessor-only form of SI-96 §5.1 cannot express (one of
  the two grounds it was not chosen; the other is that it still needs a
  position basis to say *which* predecessor when several exist).
  Duplicate k within one intent is refused (§5), mirroring the
  duplicate-seq rule; a differing `requirement` does not excuse it.
- **No new bounds.** The amendment adds no depth cap, no per-intent
  entry cap, and no chain counter. The longest constructible chain is
  n − 1 edges in a batch of n intents, and n is already bounded by
  §15.2's batch bound (default 16, stage 5, unchanged). A future change
  to that bound is a §15.2 policy matter; SI-98 must not introduce a
  second, silent depth knob alongside it.

### 4.4 Why positional, recorded once

SI-96 §5.1 analysed three candidates without ranking them. The spec
chooses **positional** on two grounds. First, machinery: labels
reintroduce exactly what the positional form excludes by construction —
a uniqueness obligation (duplicate designators become a parse concern)
and cycles (a parse-time cycle check becomes necessary machinery where
the positional form needs none). Second, evidence-shape: the corpus's
own citations include a `{seq: 1}` / `{seq: 0}` pair coinciding with a
batch's 0-based positions (SI-95 §4) — positions are a vocabulary at
least one observed body was already reaching for, though SI-96 §2.3's
caution stands: the corpus cannot prove motive, and this choice does not
claim it does. The form's known cost — positions are brittle to read in
a journal — is retired by §8: the journal records the resolved seq
beside the authored form, so no reader interprets a bare position
against a document they may not have.

## 5. Parse time — stage 4 under version 2

Verification splits in two (SI-96 §5.2). This section pins the
parse-time half; §6 pins resolution. The split's load-bearing property:
**every parse-time sibling check is decidable from the document alone**
— no fold, no journal, no task — which is why the standing no-fold C-1
harness can price the new form with no procedure change (§12.1).

### 5.1 What stage 4 gains

For each sibling entry in `intents[i].depends_on[j]`, in the parser's
existing per-entry order, stage 4 refuses the batch (whole-batch
refusal, §9.2 — unchanged) on the first of:

| Detail class (problem-string prefix) | Condition |
|---|---|
| `sibling_not_an_index` | k is a bool, or not an int |
| `sibling_negative_index` | k < 0 |
| `sibling_requirement_invalid` | `requirement` present and not in {`outcome`, `success`} |
| `entry_names_both_seq_and_sibling` | the dict carries both designator keys (either one's value validity is irrelevant — the entry is ambiguous as written) |
| `sibling_self_reference` | k == i |
| `sibling_forward_reference` | i < k < n (n = number of intents in the document) |
| `sibling_out_of_range` | k ≥ n (names no member of this document at all) |
| `sibling_declared_twice` | the same k already declared by this intent (any requirement) |

The three range/order classes are kept distinct — self, forward, and
out-of-range are different authoring failures, and SI-95's discipline
applies: a taxonomy that cannot split them cannot price a fix. The
problem strings follow the existing stage-4 format
(`intents[i].depends_on[j]: … (§9.3)`), extended by this amendment;
SI-98 pins the exact strings in code and tests them verbatim.

### 5.2 What stage 4 does not gain

- Seq-form verification is **byte-for-byte unchanged**: form rules first
  (bools refused; bare int or {seq, requirement}; non-positive and
  duplicate seqs refused), then the fold check against this task's
  already-journaled intents, failing closed with no fold (§9.3).
- A sibling entry is **never** fold-checked, at parse or anywhere else:
  its target is not a journal row and has no seq at parse time.
  Conversely a seq entry is never position-checked. Each entry is priced
  by exactly one rule-set — its own form's.
- No cross-form duplicate is constructible, so none is checked: a seq
  entry verifies against the fold of *prior* intents, and a sibling's
  seq does not exist until this document's walk assigns it — after
  parse. A seq entry naming the number a sibling will later be assigned
  names, at parse time, a future seq, and fails closed as one. Stated
  so SI-98 does not invent a reconciliation rule for a collision the
  ordering already forbids.

### 5.3 Stage 2 under version 2

- The version check's shape is unchanged (an exact match against a
  known version, stage 2): the v2 parser knows versions {1, 2}
  (§9.1). The `depends_on` entry field set is **version-scoped**: under
  version 1 it remains {`seq`, `requirement`}, and a `sibling` key in a
  v1 document fails stage 2 as an unknown field, exactly as any
  unknown field does today; under version 2 it is
  {`seq`, `sibling`, `requirement`}.
- Both-designators is **not** a stage-2 matter under v2 (both keys are
  known fields); it is priced at stage 4 with the other form rules
  (§5.1), mirroring how a non-integer `seq` is a stage-4 failure today
  rather than a schema failure. Stages 1, 3, 5, 6 are untouched in both
  versions, and §9.5's whole-batch, never-reaches-the-gate refusal is
  unchanged.

## 6. Resolution at journal time

### 6.1 Where resolution lives

Resolution is an act of the **runner's admit walk** — the serial walk
that already takes a parsed batch's intents in document order and hands
each to admission (`admit(intent) -> intent_seq`, the loop's business,
§12; in the live path, `admit_tool_intent` gates and journals it via
`record_self_edit`). It is not part of parsing (the parser is fold-free
for sibling entries and assigns no seqs) and not part of the gate (§7).
The walk is the only place holding, at once, the document's order and
the seqs being assigned — the two facts a sibling reference needs.

### 6.2 The algorithm, pinned

The walk maintains a per-walk **resolution map** M: position → journal
seq, built by the walk itself from its own admission results, in
document order, held for the walk's duration and no longer. For each
member at position i, in order:

1. Partition the member's parsed `depends_on` into seq entries
   (fold-verified at parse; carried unchanged) and sibling entries.
2. For each sibling entry {`sibling`: k, `requirement`: r}: if k ∈ M,
   the entry resolves to {`seq`: M[k], `requirement`: r}. If k ∉ M,
   the member is **refused at resolution** (§6.4) — it is not gated,
   not journaled as an intent, and the walk continues at i + 1.
3. If every entry resolved, construct the member's effective intent:
   `depends_on` is the authored list **in authored order**, sibling
   entries replaced in place by their resolved form; the authored list
   is preserved verbatim alongside (§8). Hand it to admission as today.
4. If admission returns a journal seq, set M[i] to it. If it returns
   none (a caller that declines to journal a member — the replay walk
   with no admit callback is the standing example), M gains no entry
   for i, and any later member depending on i refuses under §6.4 with
   that sub-case named in its record's detail.

One source of truth per walk: the walk resolves **only** from M. It
does not search the journal mid-walk for a sibling's seq — a seq the
walk did not itself assign in this walk does not resolve a reference in
this walk, however findable after the fact. (Journal archaeology during
admission is how two walks come to disagree about one batch; the serial
design exists to prevent exactly that.)

### 6.3 Resolution is verdict-blind

Admission journals **every** gated member: ADMIT, REJECT, and ESCALATE
each produce the member's journal row and seq (the serial cycle's
standing behaviour — a non-ADMIT verdict journals its non-execution
outcome, it does not erase the intent). Step 4 therefore records M[i]
for every member the walk gated, whatever the verdict. Consequences:

- A dependent whose sibling was gate-**REJECTED** resolves normally.
  It is *not* refused at resolution — its sibling exists in the
  journal, which is all resolution prices.
- A dependent whose sibling was **ESCALATED** resolves normally, on
  the same ground. If an operator later admits that sibling after
  review (SI-92's resolution acts), the dependent's already-journaled
  reference names the same seq throughout — no rewriting, no revival
  mechanics, because nothing was left unresolved.
- What a sibling's verdict means for the dependent's **execution** is
  not this amendment's business: once resolved to journal seqs, the
  existing Q-2 fold semantics (§7.3) apply unchanged. An `outcome`
  requirement is satisfied by any recorded conclusion — executed,
  refused at claim, skipped, rejected, or escalated. A `success`
  requirement is satisfied only by executed-and-completed; a queued
  dependent whose `success` dependency concludes any other way is
  **skipped** at drain (a `queue_skip` is journaled naming the cause;
  the skip cascades). A dependent that must not run unless its sibling
  truly executed declares `success` — the vocabulary already says
  exactly this, and the amendment adds no second way to say it.

### 6.4 Refusal at resolution — the disposition, argued

The disposition is **refusal**: a pre-gate, per-member refusal in the
§9.5 family. The dependent is journaled as refused (§8.3), never
gated, never queued, never executed. The two neighbouring dispositions
were considered and closed:

- **Not a gate REJECT.** The gate never evaluated this intent; no gate
  report exists. Recording one would put a verdict in the journal that
  no gate produced — the journal saying what did not happen, the precise
  failure SI-96 §5.2 named out of bounds for the drop-the-dependency
  pricing. A refusal and a rejection are different facts; the audit
  trail keeps them different.
- **Not an ESCALATE.** Escalation exists for judgements an operator
  can actually make — policy and consequence ambiguity on an otherwise
  admissible intent. A missing seq inside a completed walk has no
  judgement content, and no operator act can cure it: admit-after-review
  (SI-92) journals a new act for the escalated intent itself; it does
  not reach back into a finished batch walk and assign the sibling a
  place in it. Escalating the dependent would strand it awaiting an act
  that cannot fix it. If the work is still wanted, the proposer's next
  cycle re-proposes it — by then the sibling, if ever journaled, is
  citable by journal seq: the §8.6 barrier pattern, arrived at by the
  walk's own arithmetic rather than by authoring discipline.

The cause vocabulary is a closed set of one code with named sub-cases:
**`sibling_not_journaled`** — at the dependent's turn, position k held
no assigned seq in this walk — the record's detail naming which
sub-case obtained: the sibling was itself refused at resolution earlier
in this walk (the cascade, §6.5); its admission returned no seq; or the
caller admitted no member at that position. One code, because the
dependent's situation is one situation; the sub-case is provenance,
carried as detail, not as a second taxonomy.

### 6.5 The enumeration — every way resolution can fail

SI-96 named unresolvable-at-resolution the amendment's central design
problem. Priced against the decided rules, the failure space is:

| Candidate failure | Level | Behaviour |
|---|---|---|
| Sibling index out of range (k ≥ n) | **Parse** | Refused at stage 4 (`sibling_out_of_range`); resolution never sees it |
| Self / forward reference | **Parse** | Refused at stage 4 (distinct classes, §5.1) |
| Sibling "not an intent" | **Unconstructible** | Every array member is an intent; signal documents carry no intents and no positions exist to cite. There is no document in which a position names a non-intent |
| Duplicate designators | **Parse / unconstructible** | Duplicate k refused at stage 4; the label form's uniqueness problem does not exist — positions have no designators to duplicate |
| Sibling's journal write **raised** (journal failure mid-walk) | **Cycle** | The walk aborts — an infrastructure failure of the cycle, not a per-member pricing. Members already journaled stay journaled; members not yet reached get **no records at all** (no refusal is fabricated for a member the walk never priced), and the batch's missing response record (§8.4) is itself the mark of an interrupted walk |
| Sibling admitted but **no seq returned** (caller skipped it / no admit callback) | **Resolution** | The dependent is refused under §6.4, cause `sibling_not_journaled`, sub-case named |
| Sibling **refused at resolution** earlier in the walk | **Resolution** | The dependent refuses under the same rule and code; its detail names the sibling's refusal as the sub-case. The cascade is the same rule applied again — no special case is specified, and none is needed, because document order guarantees the sibling's outcome is already decided when the dependent is reached |

### 6.6 Atomicity

- Journal-before-materialize is unchanged: nothing executes from a
  batch whose members are still being walked.
- Per member, the walk appends **exactly one** of: the member's intent
  record (carrying a fully-resolved `depends_on` — §8), or its
  resolution-refusal record. No member is ever journaled with a
  partially-resolved `depends_on`; no journal state exists in which
  some of a member's sibling entries are resolved and others absent.
- A refusal never reaches backwards: members already journaled are not
  un-journaled, edited, or re-priced by a later member's refusal. The
  journal is append-only; rollback exists elsewhere in SI as a
  journaled operation and refusal does not invoke it.
- A batch walk is **not resumable**. An interrupted walk (§6.5, cycle
  row) ends the cycle; unadmitted members are simply never admitted,
  and the next cycle re-proposes from perception. No slice may add
  mid-batch resume: a resumed walk would hold a resolution map it did
  not build, against §6.2's one-source rule.

## 7. The gate's view — post-resolution

SQ4 decided: **the gate prices the dependent after resolution** — the
resolved `{seq, requirement}` entries, under the gate's existing §6.2
normalisation (check 2 fails closed on any malformed entry) and its
existing chaining check (every intent chains to the task root via
`source_seqs`). Grounds:

- The gate prices references against **journal facts**. A sibling
  designator is not a journal fact; the resolved seq is. Pricing the
  sibling form at the gate would require the gate to hold the walk's
  resolution map — a second authority for the one fact the walk
  already owns.
- The existing defence is real and stays load-bearing as a backstop,
  not a mechanism: a `sibling` entry reaching the gate anyway fails
  §6.2 closed (a dict whose `seq` is absent is a normalisation problem,
  and check 2 fails on any problem). The walk is specified so it cannot
  happen (§6.2) — but if an implementation bug ever produced it, the
  failure mode is refusal, not silent admission.

**Invariant (SI-98 test obligation, §11):** no `sibling` key ever
reaches `gate.evaluate`, and no journaled intent record's effective
`depends_on` contains one. The gate changes by zero lines under this
amendment; the queue fold and the drainer change by zero lines —
both already read exactly the normalised resolved form the walk now
journals.

## 8. Journal shape — authored and resolved

### 8.1 The intent record

A journaled intent that declared any sibling entry carries, in its
intent record:

- `depends_on` — the **effective** list: authored order preserved,
  each sibling entry replaced in place by {`seq`, `requirement`}.
  This is the only `depends_on` any consumer reads — gate report,
  queue fold, drainer, perception's prior-cycles section — and it is
  indistinguishable in form from a v1 record's.
- `authored_depends_on` — the model's `depends_on` entries
  **verbatim** as authored (both forms, authored order). Present on
  any intent whose authored list contains a sibling entry; absent
  otherwise (a v1-shaped record gains no new field).

§14.3 already requires "model referenced seq N" to be distinguishable
from "runner chained to seq N". This amendment adds the third authored
kind — **"model referenced sibling k, resolved to seq N at journal
time"** — and the pair of fields above makes it walkable from the
journal alone: the authored entry says what the model wrote, the
effective entry says what it became, and neither masquerades as the
other. An audit presenting a resolved seq as if the model had cited it
would falsify the proposer's measured behaviour; one keeping only the
position would lose the journal fact. Both are kept; that is the rule.

### 8.2 Parse-result representation

On parse success under v2, the parser's constructed intents carry
sibling entries in normalised authored form —
{`sibling`: k, `requirement`: r} — visibly unresolved, beside seq
entries normalised as today. The parser resolves nothing. Any consumer
that requires the resolved form and is handed an unresolved one must
refuse closed: the gate already does (§7); SI-98 must keep it so for
every consumer it touches.

### 8.3 The refusal record

A resolution refusal journals one experience record, kind
**`proposal_resolution_refusal`**, carrying: the task seq; the member's
position i in its document; the authored entries that failed to
resolve, verbatim; the cause code `sibling_not_journaled` with its
sub-case detail (§6.4); and linkage to its cycle's attempt/response
records where the caller's journaling provides them. It is not a
self-edit, carries no gate report and no verdict, and is never an
input to the queue fold. (Field-level pinning is SI-98's.)

### 8.4 The response record and the batch summary

The §14.3 response record's batch summary is extended, not reshaped:

- `batch.chaining` gains **`sibling_resolved`**: the count of members
  carrying at least one sibling entry resolved at journal time —
  beside `model_authored` (journal-seq authored) and `runner_filled`,
  whose meanings are unchanged. A member with both authored forms
  counts in both authored kinds; the label text must say so.
- `batch` gains **`resolution_refused`**: the positions refused at
  resolution in this walk, in document order (absent or empty when
  none). `batch.intent_seqs` keeps its meaning.
- §14.6's attempt chain stays walkable, and the walk now also surfaces
  the refusal records — a reviewer reconstructs the batch's whole
  story, admissions and refusals alike, from the journal alone. A
  batch whose walk was interrupted (§6.5) has no response record;
  that absence is the interruption's record, priced as one.

## 9. Backwards compatibility

### 9.1 The version bump (SQ3)

The §8 document version goes from 1 to **2**. Grounds, from the
contract's own text: §9.2 stage 2 pins that forward compatibility is by
version bump, not by tolerance — the parser's unknown-field problem
string says so verbatim, and the version check is an exact match.
Widening the v1 entry union instead (SI-96 §5.3's other branch) would
make two contracts answer to one version number, and every record that
says "version 1" ambiguous about which priced it. The bump keeps every
record's claim exact.

### 9.2 Compatibility rules

- **v1 documents under the v2 parser** parse **identically** to today
  — every stage, every problem string, every constructed intent. A v1
  document cannot contain a sibling entry (the key fails its stage 2),
  so there is nothing new for it to mean.
- **v2 documents under a v1 parser** fail stage 2 on the version
  mismatch — refused whole, never partially read. Fail-closed is the
  only cross-version behaviour either direction.
- **Version is document-level.** There are no mixed-version batches;
  a document's entries are all priced by its own version's rules.
- **Existing corpora and records are untouched.** SI-94's corpus is a
  v1 corpus priced by the v1 contract; its record stands (§0). The
  conformance record of any future corpus names the contract version
  it priced (§12.2).
- **Seq-form semantics are unchanged in v2** — same form rules, same
  fold verification, same fail-closed absence-of-fold behaviour,
  same queue meaning after journaling.

### 9.3 Mixed forms in one intent — allowed

Decided: a single intent's `depends_on` **may mix** seq entries and
sibling entries. The fan-in that motivates the amendment routinely spans
the boundary — depend on a prior cycle's journaled read *and* on this
document's earlier read, then write — and refusing the mixture would
force a barrier (§8.6) for a purely syntactic reason. Each entry keeps
its own form's pricing: seq entries fold-verified at parse, sibling
entries document-checked at parse and resolved in the walk; the
effective list preserves authored order across the mixture (§6.2).

## 10. Abuse cases, repriced against the decided rules

SI-96 §5.4's cases, each with the rule that defeats it under this
spec:

| Abuse | Defeating rule |
|---|---|
| Cycles | Impossible by construction — every sibling edge points to a strictly smaller position (§4.3). No detector exists to evade |
| Forward / self reference | Parse-refused at stage 4, three distinct detail classes (§5.1) |
| Dangling by rejection (sibling gate-REJECTED or ESCALATED) | The reference **resolves** — the sibling was journaled under its verdict (§6.3). Execution is then §7.3's business: `success` dependents of a concluded-but-not-executed sibling are skipped at drain, never run; no pricing under this amendment lets a dependent execute against an unmet `success` requirement |
| Dangling by walk failure | Resolution refusal (§6.4): never gated, never queued, and — the load-bearing clause — **no path admits the dependent with the dependency dropped**. The journaled effective `depends_on` of any admitted member is exactly its resolved declared set, or the member was not admitted (§6.6) |
| Batch-splitting (one logical chain across two documents to dodge resolution rules) | Gains nothing: cross-document references are journal references, priced by the unchanged seq rules. The sibling form's reach ends at its document's edge |
| Requirement spoofing | The vocabulary is unchanged ({`outcome`, `success`}), computed by parser and gate from the entry's own field, never taken from prose; §9.4 structural ignoring is untouched, and the sibling entry's closed schema carries no authority-shaped field |
| Reference laundering (naming something the author may not name) | A sibling entry can name only an earlier member of the same document of the same task — not another task's intent, not the task row, not a privileged row, not a future seq. Its blast radius is strictly smaller than the journal-seq form's (§5.1 of SI-96, carried) |
| Position-shifting (reordering between parse and walk so a position names a different intent) | Parse and walk consume the same array in the same order (§4.2); walk order == document order is a tested invariant (§11), not a hope |
| Provenance erasure (presenting a resolved seq as model-cited, or a position as a seq) | `authored_depends_on` is journaled verbatim beside the effective list (§8.1); the §14.3 chaining summary counts the sibling-resolved kind separately (§8.4) |

## 11. The implementation slice (SI-98)

One slice, one PR, code plus tests. Its contents, against this spec:

1. **`si/gaia/proposer.py` — parser.** Accept versions {1, 2} (§9.1);
   version-scoped `depends_on` entry field set (§5.3); stage-4 sibling
   checks with §5.1's detail classes and pinned problem strings;
   parse-output normalisation for sibling entries (§8.2); the batch
   summary's `sibling_resolved` count (§8.4). Seq-form code paths are
   not refactored — unchanged means unchanged, and the diff shows it.
2. **The admit walk** (replay and live gateway runners alike): the
   resolution map and §6.2's algorithm; the
   `proposal_resolution_refusal` record (§8.3) via the existing
   experience-record path; the response record's `resolution_refused`
   extension (§8.4); `authored_depends_on` on the intent record (§8.1).
3. **Replay corpus** (L-1 line): scripted v2 documents exercising the
   form deterministically — resolution exercised end-to-end (parse →
   walk → journal → §14.6 chain walk) against stub admission, labelled
   replay-only as that corpus is. This is SI-98's positive evidence;
   it is not a live claim of any kind.
4. **Explicitly not in SI-98:** gate changes (zero lines, §7);
   queue-fold or drainer changes (zero lines, §7); perception or
   output-contract text changes (§12.3 — a closed class); the C2
   multi-cycle design (D-96-4, specified separately); any capture.

Test obligations (each maps to a rule above; the suite's existing
discipline — new tests plus affected subsets for the builder, one full
run for the reviewer — applies):

- **v1 regression:** the existing parser test corpus passes with
  outcomes identical in kind and count; the SI-94 frozen corpus,
  re-scored through the amended parser as a *parser regression check*,
  reproduces its recorded stage outcomes item-for-item (20/30, same
  failing stages). A regression assertion about the parser on v1
  input — it does not re-grade or amend the SI-94 record, and its
  output is a test result, not a record.
- **Form:** each §5.1 detail class has a refusing case; a both-keys
  entry refuses; a `sibling` key in a v1 document fails stage 2; a
  v2 document under v1 rules fails stage 2 (version); valid fan-in,
  maximal chain (n − 1 edges at the batch bound), and mixed-form
  (§9.3) documents parse.
- **Resolution:** with a stub admission assigning seqs — sibling
  entries resolve in place, authored order preserved,
  `authored_depends_on` verbatim; a REJECTED sibling and an ESCALATED
  sibling each still resolve (§6.3); admission returning no seq
  produces exactly one refusal record and the cascade reaches the next
  dependent (§6.5); walk order == document order (§4.2).
- **Gate invariant:** a spy gate asserts no `sibling` key in any intent
  it evaluates, across every walk test (§7).
- **Journal shape:** the §14.6 chain walk surfaces refusal records;
  chaining counts split three ways (§8.4); the queue fold prices a
  resolved dependent exactly as a seq-authored one, including the
  `success` skip cascade against a rejected sibling (§6.3).

## 12. Re-measurement under this amendment

### 12.1 What it prices

When SI-98 exists, a **fresh cycle-0 corpus** — the C-2 design
unchanged, same six workloads, fresh capture — scored by the amended
parser under **no-fold** scoring (D-96-3's sequencing guard: the B-line
re-measurement keeps the standing measurement so the amendment is the
only thing that moved) prices **sibling-form formation directly**: the
parse-time checks are document-local, so a body forming a valid sibling
reference passes stage 4 with no fold supplied, and a body still citing
journal seqs in cycle 0 still fails closed — correctly, since no
targets exist. Evidence it worked: stage-4 failures on the chain-shaped
workloads fall, with surviving stage-4 classes named by form.

### 12.2 The record it would make

The corpus record names its configuration exactly: candidate Imagine
v13, contract **version 2**, parser at the SI-98 commit, no-fold
scoring, D-C1 applied as it stands. Per SI-96 §5.6, that rate is **a new
configuration's rate on a new contract — never a successor number to
0.6667**, recorded separately with no number lent in either direction.

### 12.3 The precondition this spec does not supply

A fresh-corpus run prices the model's formation of a form **under
whatever model-facing contract text is then in force**. The
output-contract instruction (§6.5's text, which today states the
journal-seq rule) belongs to the §4.4 class SI-91 §5 did **not** open;
whether and how that text comes to describe the v2 form is a separate
admissibility decision for the lead, **not authorised by this spec and
not assumed by it**. A re-measurement run before any such decision would
price the absence of instruction, not the presence of capability, and a
record that failed to say which it priced would be dishonest. SI-98
therefore ships with replay evidence only (§11 item 3); a live
conformance re-measurement is commissioned, if at all, by a later lead
decision that also settles the §4.4 question.

### 12.4 What stays UNMEASURED under this amendment

| Quantity | Priced by | State |
|---|---|---|
| Correct cross-cycle journal-seq reference formation | C2 (D-96-4) or a live run | UNMEASURED |
| Resolution correctness in a live walk (§6.2 against a real gate and journal) | Live runs only | UNMEASURED — SI-98's evidence is replay-path |
| Dependency-ordered **execution** correctness | Live runs only | UNMEASURED |
| Gate behaviour on resolution refusals | — | Nothing to price: refusals never reach the gate, by design (§6.4) |
| §18.1 rows P5-R1, P5-R9 | — | UNMEASURED, unchanged |
| §18.3 claim | — | NOT earned, unchanged |

## 13. What this spec does not do

No code, no parser or walk change, no capture, no scoring run, no
re-grading of any recorded verdict. It does not open any D-F1 class —
SI-91 §5's boundary is restated, not moved (§12.3) — and it does not
amend §8/§9 in force: it *specifies* the amendment SI-98 will enact,
and until that slice merges, the contract in force is version 1 exactly
as the code at §1's fingerprint implements it. It does not commission
the C2 design (D-96-4 sequences it after this spec, as its own slice)
and does not restate D-C1. What it does: answer SI-96 §5.5's five
questions with decided rules and state SI-98's contents and test
obligations — so the implementation, when built, is a transcription of
decisions already taken, not a second design.

## Honesty labels carried forward

§18.3 **NOT earned**; P5-R1 / P5-R9 **UNMEASURED**; live
model-authored intent **OCCURRED, scoped to SI-74 only**; L-3
**LOCKED**; C-4 **does not proceed for v13 on the SI-94 record**;
A-2 / A-3 / A-4 **UNAPPROVED**; unlocks `[]`; SI-94 verdict
**BELOW THRESHOLD**; SI-92 **INSUFFICIENT**; D-C1 **stands**
(SI-91 §3); contract in force remains §8 version **1** until SI-98
merges.
