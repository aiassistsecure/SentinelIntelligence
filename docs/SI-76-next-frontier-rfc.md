# SI-76 RFC — The Next Frontier After the L-Line (Design Only)

> **Status: DESIGN ONLY.** Derived from merged evidence on `main` at
> `754d4d50d3de1bedb9ccd36ded818af8d90cc2a2` (PR #81 merge, SI-75). Changes
> no code, no test, no policy default, no gate behaviour. Nothing here is
> implemented, measured, or run live unless a section says it exists today
> on `main`.
> **Scope:** CPU only. No GPU, no cloud, no spend, no downloads. This is
> machine learning infrastructure throughout; the served model is a
> specialised machine learning model.
> **Relationship:** SI-69 defined the L-line (L-1 to L-5) in its §4. That
> line is complete except L-3, which is locked (§0, §1). SI-69 defines no
> successor, so this RFC states what the L-line earned (§1), why its live
> failures happened (§2), what must be measured offline before any further
> live attempt (§3), and what line should follow (§4).
> **Honesty labels carried forward verbatim, unweakened:**
> - §18.3 claim (SI-63): **NOT earned**
> - SI-63 §18.1 row P5-R1: **UNMEASURED**
> - SI-63 §18.1 row P5-R9: **UNMEASURED**
> - Live model-authored tool intent: **OCCURRED, scoped to SI-74 only**
>   (2026-11-02, P-5 live run only; 2 parsed intents, `fs_list` x2; 1 ADMIT
>   + executed + verified, 1 ESCALATE + not executed). No other run, and
>   no sentence here, extends that scope.
>
> Where this RFC and an earlier slice doc disagree, the earlier doc's
> evidence wins until a merged slice updates it with new evidence.

## Table of contents

- [0. Honesty table](#0-honesty-table-state-after-si-75)
- [1. What the L-line built, and did not earn](#1-what-the-l-line-built-and-did-not-earn)
- [2. Root cause: a model-output-content failure](#2-root-cause-a-model-output-content-failure)
- [3. The proposer-conformance readiness gate (offline)](#3-the-proposer-conformance-readiness-gate-offline)
- [4. The next line: the C-line](#4-the-next-line-the-c-line)
- [5. Decision points for Mark](#5-decision-points-for-mark)
- [6. Rejected alternatives](#6-rejected-alternatives)
- [7. Non-goals](#7-non-goals)
- [8. Sources and code references](#8-sources-and-code-references)

## 0. Honesty table: state after SI-75

No row is strengthened relative to SI-69 §0 as updated by SI-74/SI-75.

| Element | State after SI-75 | Evidence / scope |
|---|---|---|
| L-1 replay corpus v2 | DONE | SI-70 (PR #76); REPLAY ONLY, live N=0 |
| L-5 seam reconciliation | DONE | SI-71 (PR #77); fork half of S-6 remains UNBUILT |
| L-2 live re-run | ATTEMPTED TWICE, no task completed | SI-72 (PR #78, environmental failure, N=0, no task ran); SI-74 (PR #80, 0 of 3 end-to-end in P-4 run and 0 of 3 in P-5 run) |
| L-4 tuning | DONE, empty list | SI-75 (PR #81) §3: `tuning_changes: []`, explicitly |
| L-3 residency-loss drill | LOCKED | SI-69 §4.5, conditional on L-2 success (§4.4), which has not happened |
| SI-63 §18.3 claim | NOT earned | SI-74 §0/§3, SI-75 §0: 0 of 3 in each SI-74 run |
| Live model-authored intent | OCCURRED, scoped to SI-74 only | Header; SI-74 §0/§3.2, P-5 run only |
| §18.1 rows P5-R2–R8, R10 | As promoted, scoped, by SI-74 | SI-74 §4, P-5 journal fold only; not re-derived here |
| §18.1 row P5-R1 | UNMEASURED | SI-74 §4: latencies are client-observed, not Hearth-path performance |
| §18.1 row P5-R9 | UNMEASURED | SI-74 §4, SI-75 §5: no eviction induced; L-3 locked |
| Model quality beyond SI-74 fixtures | UNMEASURED | No other live benchmark has been run |
| A-2 / A-3 / A-4 | UNAPPROVED | Not approved by SI-40, SI-63, SI-69, or this RFC; out of scope (§7) |

Consequences: (1) The L-line is complete as a wiring/corpus/label/tuning
line, not as a task-completion line; never cite this RFC otherwise.
(2) The unmet condition — ≥1 live task end-to-end — is blocked on the
evidence by what the served model returns, not any policy cap (§2); an
unmeasured retry spends minutes-scale live calls blind (§2.2). (3) Any
§3 offline number measures a recorded corpus through the parser, never
a live path (§3.6).

## 1. What the L-line built, and did not earn

### 1.1 L-1 — Replay corpus v2 (SI-70)

Doc `docs/SI-70-l1-replay-corpus-v2.md`, record
`docs/SI-70-measurements.json`, module `si/gaia/propose_measure.py`
(`REPLAY_WORKLOADS_V2`, labelled `ReplayTransport`,
`_ChainReplayTransport`). Six workloads: the SI-68 three verbatim plus
`multi-cycle-read`, `depends-on-chain`, `perception-truncation` (§6.3
bounds). All REPLAY ONLY; live N=0; every §18.1 row UNMEASURED. Reuse in
this RFC: the v2 perception set is the prompt-side basis of §3.3 — the
same perceptions, catalogue, and standing instructions a live cycle
sends, so a recorded response is scored against the real input shape.

### 1.2 L-5 — Seam label reconciliation (SI-71)

Doc `docs/SI-71-l5-seam-label-reconciliation.md`, module
`si/gaia/seam_reconcile.py`. SI-side poller/reconciler BUILT, live fork
interop UNMEASURED, fork half of S-6 UNBUILT, F-E-INGEST/F-E-SSE
UNMEASURED. No journaled record changed meaning. L-5 is closed.

### 1.3 L-2 — Attempted twice (SI-72, SI-74)

SI-72: D-L0 satisfied by Mark (2026-11-01), node down, preflight failed
closed at `served_models`, `gaia_environmental_failure` journaled, N=0,
no task ran. SI-74: node serving Imagine v12 as Resident, preflight
passed, both suites completed. P-4: 3 attempts, 3 parses, 0 parse-ok,
0/3 tasks. P-5: 4 attempts, 3 parses, 2 parse-ok, ADMIT 1 / ESCALATE 1,
0/3 tasks. §18.3 NOT earned in either run. SI-73 (PR #79, fleet name
`smollm2` -> `imagine-v12`) sits between them and is not a frontier item.

### 1.4 L-4 — Tuning from live evidence (SI-75)

Doc `docs/SI-75-l4-tuning-from-live-evidence.md`. All six numeric §15.2
keys in `si/gaia/policy.py` (`PROPOSER_POLICY_DEFAULTS`) assessed against
the SI-74 folds (§2.1–§2.6 there); no cap bound any failure; tuning list
empty, explicitly (§3 there). Rule carried forward: replay alone never
justifies tuning; no tuning is proposed anywhere in this RFC.

### 1.5 L-3 — Locked, and why the lock is the design working

SI-69 §4.4 defines L-2 success as ≥1 fixture task end-to-end with its
§14 chain walkable; §4.5 makes L-3 conditional on that success. SI-74
was 0/3 in both runs, so L-3 stays LOCKED (SI-75 §5). A drill against a
proposer that has never completed one task live would measure a halt
without the baseline it needs. This RFC does not unlock L-3, weaken its
condition, or propose a substitute; the only path is §4.6 (§4.7 here).

### 1.6 The one gap this RFC closes — on paper

No offline measure of *proposer conformance* exists on `main`: the
replay corpora score scripted responses through the parser and say
nothing about any real model's output. Nothing scores a candidate's
recorded responses against §8 as a readiness signal before a live run.
Closing exactly that gap — and nothing larger — is §3.

## 2. Root cause: a model-output-content failure

### 2.1 What actually failed (SI-75 §2.7)

Adopted verbatim in substance from SI-75 §2.7: P-4, 3 of 3 responses
parse-failed at stage 1 (envelope, SI-63 §9.2). P-5, 1 of 3 parsed
responses parse-failed at stage 1; 1 of 4 attempts transport-failed at
the fixture's 900 s timeout. Journaled excerpts are single sentences of
identity text — e.g. "I am a text-to-SQL model that turns natural
language into SQL." — not the one JSON proposal document SI-63 §8
requires. The served model is a specialised machine learning model
(natural language -> SQL); its measured behaviour on this seat is to
return identity text instead of the §8 document. That is a
model-output-content failure. A-benign's one successful P-5 cycle
(attempt seq 9 -> response seq 10 -> `se_0001` ADMIT -> tool_result
seq 16, verified) still did not complete its task: the model never
signalled completion (§8.7) and its next response did not parse.
No §15.2 key — attempts, batch, timeout, freshness, perception bytes,
response bytes — governs what the model says (SI-75 §2.1–§2.7).

### 2.2 The cost fact (SI-74 §3)

Live calls are minutes-scale: journaled client-observed latencies
196,293–900,101 ms per attempt in P-5 (last = 900 s timeout) and
260,907–434,400 ms in P-4 (SI-74 §3.1/§3.2, folded from the journals).
A failed live re-run is expensive in wall-clock on the seat. A successor
line that proposes another live run before measuring offline whether
the candidate's output is even §8-shaped spends that cost blind. That
is the problem §3 removes.

### 2.3 What §2 does not claim

Conformance on this evidence is *intermittent*, not absent (P-5: 2 of 3
parses parse-ok) — hence a rate over a corpus is the right readiness
quantity, not one anecdotal response. No other model is claimed to
behave better, worse, or at all: none has been measured, and this RFC
names none as existing, available, or expected to pass. The content
failure is not converted into a tuning, prompt, or parser change;
weakening §8/§9 to accept identity text would manufacture conformance
and is rejected in §6.

## 3. The proposer-conformance readiness gate (offline)

### 3.1 Purpose and posture

An SI-side, CPU-only, offline measurement. Given recorded responses
from one named candidate served model, paired with the L-1 v2
perceptions they answer (§3.3), score every response through the
existing §9 six-stage parser (`parse_proposal` in
`si/gaia/proposer.py`) and fold the outcomes into a machine-readable
record (§3.4). Output is a **readiness signal only**: how often that
candidate's recorded output is a §8-conformant proposal document,
measured *before* anyone proposes another minutes-scale live run.
The gate makes no gateway call, contacts no node, admits no intent,
executes no tool, writes no gate verdict. Scoring is deterministic:
same corpus, parser, and SI commit must produce the same fold.

### 3.2 Conformance metric

- **Primary — §8-conformance rate:** for N recorded responses,
  `parse_ok / N`, where `parse_ok` counts exactly responses with
  `stage_reached == 6` and `failing_stage is None` — i.e. exactly one
  JSON document (§8.1) passing §8.2 field rules, registry and reference
  checks (§9.2 stages 3–4), §15.2 bounds (stage 5), and §8.7/§8.8
  signal rules (stage 6). No partial credit, for the same reason §9
  refuses partial salvage of a batch.
- **Diagnostics (reported, never substituted):** failing-stage
  histogram (stages 1–6), stage-1 count stated separately (the SI-74
  failure mode), parse-ok rate per workload and per cycle position,
  and — among parse-ok responses — batch sizes, signal kinds (intents
  only / `task_complete` / `cannot_proceed`), stage-4 reference
  behaviour, and the count of non-JSON plain-text bodies.
- **Not the metric:** not a task-completion rate, verdict distribution,
  quality score, or live-behaviour prediction. A conformant proposal
  can still be ESCALATE/REJECT, name the wrong tool, or fail to
  complete a task — A-benign (§2.1) is the standing example of
  conformance without completion. The metric measures output *shape*
  against §8, and nothing else.

### 3.3 Corpus requirements

A corpus missing any requirement is reported as INSUFFICIENT CORPUS
naming the missing requirement; no rate is published from it.

1. **Prompt side from L-1 v2:** each item pairs a response with one
   L-1 v2 perception (SI-70: SI-68 three + `multi-cycle-read`,
   `depends-on-chain`, `perception-truncation`), rendered as the live
   proposer renders it (perception, catalogue, standing instructions,
   SI-63 §6/§7).
2. **Responses recorded verbatim, once:** each body is the candidate's
   own output, recorded at capture time with prompt-item id, candidate
   identity as an operator-supplied label exactly as discovered (never
   invented), capture date, and journal/file provenance. Scripted,
   replay-transport, hand-written, edited, or another model's responses
   never enter the corpus; the SI-68/SI-70 scripted corpora measure the
   parser and are explicitly *not* conformance corpora.
3. **Coverage stated:** per-workload counts, cycles covered, and whether
   completion/inability signals were elicited. Single-workload or
   single-cycle corpora are labelled as such; their rate is scoped to
   that coverage in the slice doc.
4. **N fixed before scoring:** adding responses after seeing a rate is
   a new measurement with a new record, never an edit of the old one.
5. **No duplicated bytes:** bodies stored once, referenced by hash; no
   seed tensors, GGUF files, or model bytes are copied (standing design
   rule), and SI-74 journal responses are referenced by bundle and seq,
   not copied.
6. **Today's recorded responses, honestly:** the only candidate
   responses in merged evidence are SI-74's seven attempts (P-4: 3
   responses, 0 parse-ok; P-5: 3 parsed, 2 parse-ok, plus 1 transport
   failure with no response to score). Re-scoring them offline is a
   valid *harness check* for C-1 (§4.3) and nothing more: N is tiny,
   the outcome is already known live, and it earns no new claim.

### 3.4 Journal and record shape

Following the SI-68/SI-70/SI-74 record pattern, each measurement makes
one new machine-readable record, never an edit of a prior record, with:
`fingerprint` (SI commit, parser identity, corpus id + hash, candidate
label per §3.3(2), environment fingerprint, date America/New_York,
`mode: offline`); `honesty` (§0 labels verbatim, §3.6 wording,
`live_attempts: 0` for the measurement itself); `corpus` (per-workload
counts, cycle coverage, signal elicitation, per-item provenance, any
§3.3(3) scoping label); `results` (N, `parse_ok`, rate as exact
fraction + decimal, stage histogram with stage 1 separate, per-workload
and per-cycle breakdowns, fold provenance of every counted outcome);
`rows_18_1` (every row exactly as SI-75 left it, each noted as promoted
by none of this); `readiness` (`threshold: null` until D-C1, comparison
only in §3.5 forms, `unlocks: []`).

Where the measurement runs through a journalled bundle (C-1's choice),
each scored item is an experience record naming corpus item id, parser
outcome (`stage_reached`, `failing_stage`, `failing_stage_name` in the
vocabulary `si/gaia/proposer.py` already journals), and response hash —
walkable item by item, as §14 chains are in the live bundles. The body
itself is held once in the corpus (§3.3(5)), not repeated per record.

### 3.5 How a threshold would be proposed — not decided here

This RFC proposes **no binding threshold number as decided**.

1. A C-slice with ≥1 conformance record (§4.4) may *propose* a threshold
   in its doc: one rate, the corpus/N it applies over, any per-workload
   floor, and its evidence — including what SI-74's intermittence
   (2 of 3 in P-5; 0 of 3 in P-4) implies about corpus size.
2. The proposal is labelled PROPOSED, NOT DECIDED everywhere, including
   records (`threshold: null` until step 3).
3. It becomes binding only when Mark decides it (D-C1, §5). Only then
   may a later record state MEETS THRESHOLD or BELOW THRESHOLD against
   the decided value, naming the decision and its date.
4. A decided threshold governs readiness to *propose* a live re-run
   only. It is not a quality bar, not a §18.1 row, and not a property
   of the model beyond the measured corpus.

### 3.6 Honesty rules for the gate (binding on every C-slice)

- Offline/replay conformance is a **READINESS signal only**: never live
  evidence, earns no §18.1 row, earns no §18.3 claim, and no §3 record
  or doc may be cited as live measurement.
- It **does not unlock L-3**. L-3 unlocks only on a real L-2 live
  success as defined in SI-69 §4.4/§4.5 (≥1 live task end-to-end, §14
  chain walkable). Nothing offline substitutes, at any rate/threshold.
- It does not authorise a live run. A live re-run additionally needs
  measured readiness against a Mark-decided threshold *and* Mark's
  explicit go for that run (D-C1/D-C2), because of §2.2's cost.
- No candidate is claimed to exist, be available, or pass. With no
  recorded corpus for a candidate, the honest result is INSUFFICIENT
  CORPUS (§3.3), not a rate.
- Numbers come only from the fold of the recorded corpus through the
  pinned parser. A number from logs, memory, a capture summary, or a
  model's own report fails review (same rule as SI-69 §4.4).

## 4. The next line: the C-line

### 4.1 Naming and justification

The **C-line**, for *conformance*: each slice builds or runs the §3
offline readiness measurement, or — only after readiness is measured
and Mark has decided — retries the live work the L-line left locked.
Not "L-6+": SI-69 closed the L-line at L-5 (L-3 locked inside it), and
reusing its lettering would imply authority SI-69 never granted. The
C-line ends by *re-entering* L-2/L-3 under their original SI-69
definitions; it does not redefine them.

### 4.2 Order and no-stall rule

Order: C-1, C-2, C-3, then conditionally C-4, C-5. C-1 to C-3 are
SI-side, CPU-only, no live node, no live call. If the conditional
slices never activate, the loop does C-1 to C-3, stops, and reports —
it does not invent replacement live work. Every slice is one PR, built
by a builder, reviewed and merged by the dev-loop driver on a later
run; no slice merges itself; every slice doc updates §0 row by row
with evidence or states explicitly that no row changes. Acceptance
below is journaled/system outcomes only, never demo impressions.

### 4.3 C-1 — Conformance harness over recorded responses (CPU-only)

- **Scope:** implement §3 scoring — consume a §3.3 corpus, score via
  the existing §9 parser (no parser change), fold §3.2, emit §3.4.
  Exercise it as a harness check on the SI-74 recorded responses
  (referenced by bundle/seq, §3.3(6)) and prove it refuses to label
  the SI-70 scripted corpus as candidate evidence.
- **Acceptance:** the check reproduces SI-74's known outcomes from the
  recorded responses (P-4 stage-1 failures; P-5 parsed items 2 parse-ok,
  1 stage-1 failure), folded, N stated; the record matches §3.4 with
  `live_attempts: 0`, `threshold: null`, `unlocks: []`, no row promoted;
  a §3.3-failing corpus yields INSUFFICIENT CORPUS naming the
  requirement and no rate; full suite green on reviewer's own run; new
  tests pin metric, histogram, and the refusal.
- **Honesty gate:** any wording presenting the harness check as a
  candidate measurement, live result, or readiness verdict fails
  review. C-1 measures the *harness* against known responses.

### 4.4 C-2 — First conformance measurement of a named candidate

- **Condition:** Mark names a candidate (D-C3) and a §3.3 corpus for it
  exists or is captured under a Mark-approved arrangement. If none is
  named, C-2 does not run; the line reports that state rather than
  substituting any model, corpus, or scripted stand-in.
- **Scope:** run the C-1 harness over that corpus; publish the §3.4
  record and a doc stating rate, histogram, coverage scoping, §3.6
  wording — no threshold comparison unless D-C1 is already decided.
- **Acceptance:** record on disk naming candidate exactly as recorded,
  corpus id/hash, N fixed before scoring, environment fingerprint;
  every counted outcome walkable item by item (§3.4); headline states
  readiness-signal-only, no §18.1/§18.3 claim, unlocks nothing.
- **Honesty gate:** any live label, undecided-threshold verdict, or
  L-3 implication in record or doc fails review.

### 4.5 C-3 — Threshold proposal and readiness review (docs, CPU-only)

- **Scope:** from C-2's record (and C-1 evidence on corpus size), write
  the §3.5 proposal for Mark: one proposed rate, corpus/N basis, any
  per-workload floor, and a wall-clock cost estimate for one further
  live run derived from SI-74 §3 latencies, stated as an estimate with
  its basis.
- **Acceptance:** proposal labelled PROPOSED, NOT DECIDED in doc and
  any record it touches; `threshold` stays `null` until D-C1; D-C1 and
  D-C2 presented as Mark's decisions with each answer's consequence
  stated; no live run scheduled, attempted, or implied.
- **Honesty gate:** presenting the proposed value as decided, or a
  dependent readiness verdict as earned, fails review.

### 4.6 C-4 — Conditional L-2 retry (readiness + Mark's go only)

- **Conditions, all three:** (a) a C-2 record exists for the candidate
  to be served; (b) its rate MEETS the threshold Mark decided under
  D-C1, stated against that decision; (c) Mark's explicit go for this
  specific run (D-C2). Missing any one, C-4 does not run.
- **Scope:** no new code by default; re-run the L-2 commands as SI-69
  §4.4 / SI-74 §2 define them, on the node serving the measured
  candidate as Resident; publish the §18.3 claim or its honest failure
  variant plus the live record, in SI-74's form.
- **Acceptance:** as SI-69 §4.4, unchanged — preflight journaled or the
  run stops with a journaled environmental failure; on completion,
  parse rate, histogram, verdicts, stop causes from the journal with
  N; injected executions = 0; rows promoted only where live journaled
  data measures them.
- **Honesty gate:** numbers only from the journal fold. The C-2 offline
  rate is never merged into, averaged with, or cited as part of the
  live Ns.

### 4.7 C-5 — Conditional L-3 path (real L-2 success only)

- **Condition:** C-4 (or any L-2 retry under SI-69 §4.4) completes ≥1
  fixture task end-to-end live. That event — only that event —
  satisfies SI-69 §4.5 and unlocks L-3.
- **Scope/acceptance:** the §4.5 drill exactly as SI-69 defines it —
  one operator-induced eviction during a live task, §16.6 behaviour
  journaled in order (transition, at most one refresh, halt/resume),
  no intent admitted against a non-`Resident` model, P5-R9 leaving
  UNMEASURED only if the drill actually ran.
- **Honesty gate:** a drill simulated by replay, recorded responses, or
  a fake transport does not satisfy this slice. Offline conformance
  plays no part in it.

## 5. Decision points for Mark

Mark's decisions; this RFC decides none, no C-slice may pre-empt them.

- **D-C1 — Conformance threshold:** if C-3 proposes one (§3.5), what
  rate (and any per-workload floor), over what minimum N, counts as
  readiness to propose a live re-run? Until Mark sets it, `threshold`
  is `null` and no MEETS/BELOW verdict exists.
- **D-C2 — Spend another live run:** given a MEETS-THRESHOLD record and
  the §2.2 cost estimate, authorise one specific L-2 retry (C-4) —
  yes/no, per run. A yes for one run is not a standing yes.
- **D-C3 — Which candidate(s), and corpus capture:** which candidate
  served model (when one exists) gets a conformance corpus, and under
  what capture arrangement may its responses be recorded for offline
  scoring? Naming none parks C-2 without penalty; C-1/C-3 still stand.
- **D-C4 — Corpus store:** if capture is approved under D-C3, where do
  recorded responses live (bundle journal vs. corpus file) under
  §3.3(5)? The C-2 doc must name the chosen store before scoring.

## 6. Rejected alternatives

- **6.1 Retry L-2 live now.** Rejected: SI-74 ran that experiment twice
  in one day against the same behaviour at minutes-scale cost (§2.2);
  SI-75 showed no policy change alters it (§2.1). A third unmeasured
  retry re-learns a known result at the seat's expense.
- **6.2 Treat offline conformance as live evidence / L-3 unlock.**
  Rejected (§3.6): SI-63 §18.2's REPLAY ONLY rule is inherited in full;
  L-3's condition is a live completion (SI-69 §4.4/§4.5), and no
  offline rate is a task, live or otherwise.
- **6.3 Weaken §8 or teach §9 to accept identity text.** Rejected:
  salvage-parsing prose would redefine conformance as whatever the
  model already emits. §8/§9 are the governance boundary (SI-63
  §9.1/§9.4); the gate measures distance from it, not moves it.
- **6.4 Tune policy/perception/prompts until live passes.** Rejected:
  SI-75 closed §15.2 on live evidence (empty list); SI-69 §4.6 forbids
  replay-justified tuning. Model-specific prompt work is not designed,
  pre-approved, or smuggled in here as "readiness".
- **6.5 A second model as conformance judge.** Rejected (SI-63 §21.6,
  SI-69 §5.6): conformance is deterministic against §8 via the existing
  parser. A judge adds a second unmeasured surface and certifies none.
- **6.6 SI-local inference to generate responses offline.** Rejected
  (SI-63 §21.3, SI-69 §5.4): SI runs as a model under Hearth
  governance; a local shortcut would measure the shortcut, not the
  served candidate, against the wrong stack.
- **6.7 Approve/pre-design A-2/A-3/A-4 here.** Rejected: A-1 was
  approved and measured (SI-40); A-2/A-3/A-4 remain separate approvals
  no merged document grants. A docs-only RFC must not smuggle one.

## 7. Non-goals

- Design only: no code, tests, or configuration in this RFC.
- No live claim, live run, or live measurement — reported, scheduled,
  performed, or implied. No conformance rate is published here either:
  §3 designs the gate; C-1 builds it, C-2 runs it.
- No threshold decided here (§3.5, D-C1).
- A-2/A-3/A-4 remain UNAPPROVED, out of scope; nothing here is their
  approval, scheduling, or pre-design.
- No candidate named as existing, available, or expected to pass
  (§2.3, §3.6); naming one is Mark's act under D-C3.
- No node relaunch, fork-repo work, or Hearth-side change assigned.
- No change proposed to §8, §9, any §15.2 key, gate, registry, queue,
  or SSE spine. No backlog beyond the C-line: unlisted slices need a
  successor RFC, not an implied extension.

## 8. Sources and code references

All on `main` at `754d4d50d3de1bedb9ccd36ded818af8d90cc2a2` unless a
doc number is given. Module/doc references replace line pointers.

- `docs/SI-69-next-frontier-rfc.md` — L-line: §0 table, §4.3 L-1,
  §4.4 L-2 (success = ≥1 task end-to-end), §4.5 L-3, §4.6 L-4, §4.7 L-5.
- `docs/SI-70-l1-replay-corpus-v2.md`, `docs/SI-70-measurements.json`
  — L-1 as-built: corpus v2, REPLAY ONLY, live N=0.
- `docs/SI-71-l5-seam-label-reconciliation.md` — L-5 as-built.
- `docs/SI-72-l2-live-rerun.md`, `docs/SI-72-measurements.json` —
  first L-2 attempt: environmental failure, N=0, §18.3 NOT earned.
- `docs/SI-74-l2-live-rerun.md`, `docs/SI-74-measurements.json` —
  second L-2 attempt: §0 scoped OCCURRED label, §3 folds + latencies
  (196 s–900 s per attempt), §4 promotions (R1/R9 stay UNMEASURED).
- `docs/SI-75-l4-tuning-from-live-evidence.md` — L-4: §2.7 root cause
  (model-output-content; no §15.2 key addresses it), §3 empty list,
  §5 L-3 stays LOCKED.
- `docs/SI-63-gaia-propose-hearth-proposer-rfc.md` — §8 contract, §9
  parser, §14 journaling, §15.2 keys, §18.1 rows, §18.2 discipline,
  §18.3 first honest live claim.
- `si/gaia/proposer.py` — §8 as code; `parse_proposal`, `PARSE_STAGES`
  (1 envelope, 2 schema, 3 registry, 4 references, 5 bounds, 6 signals).
- `si/gaia/propose_measure.py` — `REPLAY_WORKLOADS(_V2)`, replay
  transports, record shape §3.4 follows.
- `si/gaia/policy.py` — `PROPOSER_POLICY_DEFAULTS` (§15.2 keys).
- `si/gaia/live.py` — L-2 wiring; fixture policy (900 s override).
- `docs/SI-40-a1-feasibility-measurement.md` — A-1 measured; A-2/A-3/A-4
  unapproved.

End of RFC SI-76. The C-line starts at C-1 (§4.3) on the next dev-loop
run after this RFC is reviewed and merged, whether or not any §5
decision has been answered.
