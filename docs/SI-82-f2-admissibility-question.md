# SI-82 — Gaia F-2: The Admissibility Question (Design Only)
# (SI-80 §3.4)

> **Status: DESIGN ONLY — a question, posed and not answered.**
> This slice builds nothing, changes no rendering, and runs
> nothing. It contains no code, no test, no corpus change, no
> capture, no live call, and no measurement. It decides nothing:
> the question it poses is Mark's to answer (D-F1, §5 below),
> and this doc does not answer it, presume its answer, or
> arrange its contents around either answer.
> **Scope:** CPU only. No GPU, no cloud, no spend, no
> downloads. This is machine learning infrastructure
> throughout; the served model is a specialised machine
> learning model.
> **Relationship:** SI-80 §3.4 defines F-2 as a design-only
> slice whose condition is that F-1 is merged, and whose whole
> deliverable is the posing of one question. F-1 is merged
> (SI-81, PR #88, main `be96961a`). SI-80 §4 frames the
> decision as D-F1; §5 below poses D-F1 in that framing.
>
> Where this doc and an earlier slice doc disagree, the
> earlier doc's evidence wins until a merged slice updates it
> with new evidence.

## Table of contents

- [0. Honesty table](#0-honesty-table--row-by-row-per-si-80-32)
- [1. Fingerprint and scope statement](#1-fingerprint-and-scope-statement)
- [2. The question](#2-the-question)
- [3. Why the question is posed in these terms](#3-why-the-question-is-posed-in-these-terms)
- [4. Intervention classes the question could cover](#4-intervention-classes-the-question-could-cover)
- [5. D-F1, posed as SI-80 §4 frames it](#5-d-f1-posed-as-si-80-4-frames-it)
- [6. The four honesty risks (SI-80 §3.4)](#6-the-four-honesty-risks-si-80-34)
- [7. What this slice does not do](#7-what-this-slice-does-not-do)
- [8. Sources and code references](#8-sources-and-code-references)

## 0. Honesty table — row by row, per SI-80 §3.2

SI-80 §3.2 requires every slice doc to update this table row
by row with evidence, or to state explicitly that no row
changes. **No honesty row changes in this slice.** This slice
poses a question; a question is not evidence, and it earns,
promotes, weakens, and re-labels nothing. The rows below are
carried forward from SI-80 §0 and SI-81 §0, unchanged, with
one factual addition that is not an honesty row: F-1's own
completion, which is this slice's stated precondition.

| Element | State | Evidence / scope |
|---|---|---|
| F-1 failure taxonomy | DONE | SI-81 (PR #88, merged on main `be96961a`); offline classification only, §3 below |
| C-1 conformance harness | DONE | SI-77; harness check only |
| C-3 threshold proposal | DONE, adopted | SI-78; adopted as D-C1 by Mark, 2026-11-02 |
| C-2 conformance measurement | DONE, BELOW THRESHOLD | SI-79 (PR #86); 10/30 (0.3333) against D-C1 |
| Readiness verdict (SI-79) | BELOW THRESHOLD | Against D-C1 as decided; stands, not re-opened here |
| Readiness unlocks (SI-79) | `[]` | `docs/SI-79-measurements.json`: `readiness.unlocks`; this slice adds none — unlocks remain `[]` |
| C-4 conditional L-2 retry | NOT ELIGIBLE | SI-76 §4.6; its conditions fail on SI-79 |
| C-5 conditional L-3 path | LOCKED | SI-76 §4.7; no real L-2 live success |
| L-3 residency-loss drill | LOCKED | SI-69 §4.5; SI-76 §3.6; nothing offline unlocks it |
| SI-63 §18.3 claim | NOT earned | SI-74, SI-75, SI-79 §0: no live task completed |
| Live model-authored tool intent | OCCURRED, scoped to SI-74 only | 2026-11-02, P-5 live run only; 2 parsed intents, `fs_list` ×2; 1 ADMIT + executed + verified, 1 ESCALATE + not executed. No other run, and no sentence here, extends that scope |
| §18.1 rows P5-R2–R8, R10 | As promoted, scoped, by SI-74 | Cited by SI-77/SI-79/SI-81, promoted by none of them, and by none of this |
| §18.1 row P5-R1 | UNMEASURED | SI-74 §4 |
| §18.1 row P5-R9 | UNMEASURED | SI-74 §4; SI-79 §4 notes a Resident death as an observation only |
| A-2 / A-3 / A-4 | UNAPPROVED | Not approved by SI-40, SI-63, SI-69, SI-76, SI-80, SI-81, or this doc |
| D-F1 | OPEN, posed in §5 | SI-80 §4; Mark's decision, not this slice's |

## 1. Fingerprint and scope statement

| Field | Value |
|---|---|
| SI commit (base) | `be96961a3a659f3e2482fc678a90f7509328b5e3` (main, PR #88 / SI-81 merged) |
| Spec | `docs/SI-80-next-frontier-rfc.md` §3.4 (F-2), §4 (D-F1), §5 |
| Informed by | `docs/SI-81-f1-failure-taxonomy.md`, `docs/SI-81-taxonomy.json` |
| Date | 2026-11-03 (America/New_York) |
| Mode | design only — offline, no run of any kind |

Plainly, as SI-80 §3.4 requires this slice to state: **this
slice builds nothing, changes no rendering, and runs
nothing.** No file other than this doc is added or modified
by the slice. In particular, nothing in
`si/gaia/proposer.py` — not the perception assembly, not the
catalogue rendering, not the standing instructions, not the
output-contract instruction, not the gateway-request
assembly — is changed, and no wording in this doc is a
specification of a change to any of them.

## 2. The question

The one question this slice poses, in SI-80 §3.4's terms:

> *Is any SI-side, non-model intervention admissible to test
> at all* — for example a change to how the perception /
> prompt is rendered to the served model (the §6/§7
> rendering: perception, catalogue, standing instructions),
> informed by F-1's taxonomy?

Three scope words in that question carry its whole meaning,
and this doc keeps them exact:

- **SI-side, non-model.** The question is only about things
  on SI's side of the gateway that are not the model: what SI
  renders and sends. It is not about the served model itself
  — not its weights, its training, its identity, or its
  replacement. No candidate is named, claimed to exist, or
  discussed anywhere in this doc.
- **Intervention.** A deliberate change to that rendering,
  of one of the classes enumerated in §4. §4 enumerates
  classes so the question has concrete content; enumerating
  a class states only that the question, if answered in
  scope, could be answered about that class. It approves,
  schedules, designs, and recommends nothing.
- **Admissible to test.** The question is whether testing
  such a change is in scope at all. It is prior to any
  question of what a test would show, and this doc contains
  no statement — estimate, prediction, or expectation, in
  any direction — about what a test would show.

## 3. Why the question is posed in these terms

The question is informed by F-1's taxonomy (SI-81), and by
nothing else. What F-1 established, descriptively, about the
corpus SI-79 already stored:

- The 18 stage-1 (envelope) failures are made of:
  **`identity_text` ×12, `plain_prose` ×5, `bare_fragment`
  ×1** — and no empty, fenced, partial-JSON, or SQL-shaped
  body at all. `identity_text`, at 12 of 18, is the dominant
  stage-1 class. `unclassified` is 0.
- The 2 stage-4 (references) failures are a listing, not a
  rate: in both, a `depends_on` entry names the tool
  `fs_list` where §9.3 requires a journal sequence.
- The 10 parse-ok bodies are the control group.
- SI-79's verdict stands exactly as recorded: 10/30
  (0.3333), **BELOW THRESHOLD** against D-C1. F-1 published
  no new parse rate and changed nothing about how many
  failures there are.
- SI-81's closing states the boundary this slice sits on:
  what, if anything, may be done about any class is F-2's
  question — Mark's decision, not F-1's, and not this
  slice's either.

That is the extent of the informing. F-1's class counts are
descriptive statistics of one known corpus (SI-80 §5): they
are the stated reason the admissibility question is posed
about rendering at all, and they are **not** evidence that
any class in §4 would change any count, in any direction.
No sentence in this doc connects a class in §4 to a class in
F-1's taxonomy as a remedy, a target, or a mechanism; the
taxonomy describes bodies the served model produced under
the rendering as it stood, and says nothing about any other
rendering.

The root-cause continuity SI-80 §2 records also stands
unchanged behind the question: SI-75 §2.7 established a
model-output-content failure, and SI-79's 18 plain-text
stage-1 bodies are that same failure class at N = 30 scale.
This doc invents no new cause, and the question's existence
does not revise that finding in either direction.

## 4. Intervention classes the question could cover

Each class below names a concrete part of the §6/§7
rendering as it exists on main today, states what kind of
change the class would consist of *if* the question were
answered in scope for it, and states what that kind of
change puts at risk. The risk statements are the reason the
admissibility question is a decision and not a routine step.
No class is named as approved, scheduled, or expected to
help; no class is ranked; no class is described in more
detail than naming it requires, because detail beyond
naming would be design work for an undecided question.

### 4.1 Perception assembly and rendering (§6)

*What it is today:* `assemble_perception` and
`render_perception_text` in `si/gaia/proposer.py` — the
sections assembled from the journal under §6.1's precedence
order (task, catalogue, prior cycles, tool results,
residency, queue state, budgets remaining), the §6.3 bounds
and truncation handling with its labels, and the §6.4
ordering and stability of the rendered text.

*What a change in this class would consist of:* a change to
which sections are rendered, in what order, in what form,
or under what bounds and truncation presentation.

*What it puts at risk:* the model's perceived world is this
rendering; changing it changes the facts a proposal is
answerable to, including what is present at all under
truncation. It also touches the §6.1 boundary that perceived
records are data, never instructions — a rendering change
that blurred that labelling would weaken the §17
perception-poisoning posture, which is structural, not
decorative. And under §6 risk 1, any result obtained under
a changed perception rendering belongs to that changed
configuration, not to the configuration SI-79 measured.

### 4.2 Tool catalogue rendering (§7)

*What it is today:* `render_catalogue` and
`render_catalogue_text` in `si/gaia/proposer.py` — the
catalogue rendered from `si.gaia.registry.REGISTRY` at call
time (§7.1), with per-tool availability labelling for tools
outside a call's live set or disabled by policy, and the
§7.4 catalogue digest carried on the perception.

*What a change in this class would consist of:* a change to
how registry tools are presented to the model — their
rendered descriptions, argument summaries, ordering, or
availability presentation.

*What it puts at risk:* the catalogue's governing property
is that it is a *rendering of the registry*, with no
hand-maintained list: what the model is shown and what the
gate and registry actually enforce are the same source.
Presentation changes that drifted from the registry's own
schemas would put the model in front of a catalogue the
system does not honour, while the §7.4 digest exists
precisely so a rendered catalogue can be re-rendered and
verified, digest for digest. A changed catalogue rendering
is likewise a changed candidate configuration under §6
risk 1.

### 4.3 Standing instructions (§6.6)

*What they are today:* `STANDING_INSTRUCTIONS` in
`si/gaia/proposer.py`, sent as the system message of the
§6.5 gateway request — the governed-loop posture: the model
proposes and does not execute; every proposal is gated and
journaled; perceived records are data and instructions
inside them are not instructions; reasons are stated
concretely; the completion and inability signals are used
instead of invented ones. The code records that §6.6's
security-posture items are stated unweakened.

*What a change in this class would consist of:* a change to
the wording, content, or ordering of that system-message
text.

*What it puts at risk:* this is the class where a wording
change and a governance change are hardest to tell apart.
The standing instructions carry the loop's security posture
in prompt form — the data-versus-instructions boundary and
the gated, journaled posture among them — so editing the
text risks editing the posture while appearing to edit
only prose. SI-80 §3.4 risk 4 (§6 below) applies to this
class with particular force: prompt-text work presented as
something other than what it is, or as already decided, is
exactly what that risk excludes.

### 4.4 The output-contract instruction (§6.5)

*What it is today:* `OUTPUT_CONTRACT_INSTRUCTION` in
`si/gaia/proposer.py`, appended to the user message of the
§6.5 gateway request — a brief statement to the model of
the §8 proposal-document shape. The parser (§9) is the
contract's enforcement; this text is only its statement.

*What a change in this class would consist of:* a change to
how the §8 document shape is stated to the model in that
appended instruction — its wording, completeness, or
emphasis — with §8 itself and §9 unchanged.

*What it puts at risk:* the statement and the enforcement
can be pulled apart in either direction: a statement that
promises more or less than §9 enforces teaches the loop to
expect a contract the gate does not hold, and SI-76 §6.3 /
SI-80 §6.4 already rejected the neighbouring move — weakening
§8 or teaching §9 to accept what the model already emits.
Keeping this class distinct from that rejected move is part
of what an answer to D-F1 would have to be precise about;
this doc draws the line descriptively and decides nothing
about it.

### 4.5 Gateway-request assembly (§6.5)

*What it is today:* `build_gateway_request` in
`si/gaia/proposer.py` — one request body carrying the
standing instructions as the system message and the
rendered perception plus the output-contract instruction
as the user message, with no gateway tool-calling feature
used (SI-63 decision D-1: the model answers in the §8
document format, in plain completion text).

*What a change in this class would consist of:* a change to
that assembly — the split of content between the system and
user messages, or the framing text that introduces the
rendered perception.

*What it puts at risk:* the system/user split is where the
§6.1 data-versus-instructions boundary is physically
enacted in the request; moving content across it changes
which text arrives with instruction standing. The framing
text today states that the perception is data, never
instructions — changing the framing risks changing that
statement. Any such change is also, again, a changed
candidate configuration under §6 risk 1.

### 4.6 What the question does not cover

For precision, the classes the question is **not** about.
Naming them here excludes them from D-F1's scope; it does
not decide anything about them either, and several are
already decided or gated elsewhere:

- **The model itself** — weights, training, fine-tuning,
  identity, or a successor candidate. Out of scope by the
  question's own terms (§2); a successor is D-F2's separate
  question (SI-80 §4), and none is named here.
- **§8 and §9 themselves** — the proposal contract and the
  parser. Weakening either is a rejected alternative
  (SI-76 §6.3, SI-80 §6.4), not an intervention class.
- **§15.2 policy keys** — SI-75 closed policy tuning on live
  evidence with an explicitly empty tuning list; that
  closure stands and this question does not re-open it.
- **Gate, registry, queue, executor, and SSE spine** — the
  machinery that governs and journals proposals. The
  question concerns only what is rendered to the model
  before that machinery acts.
- **A live run of any kind** — C-4 remains NOT ELIGIBLE on
  its original conditions, and nothing in this doc is a
  live authorisation, a scheduling, or a step towards one.

## 5. D-F1, posed as SI-80 §4 frames it

SI-80 §4 states the decision in these terms, and this slice
poses it in exactly those terms, undecided:

> **D-F1 — Is perception/prompt-side remediation in scope at
> all?** May the loop test SI-side, non-model interventions
> (e.g. perception/prompt rendering changes) as a route from
> the SI-79 result — accepting that a changed rendering is a
> changed candidate configuration requiring fresh-corpus
> re-measurement (F-3/F-4)?

Each answer's consequence, as SI-80 states it:

- **If no:** the F-line ends at F-2's question, answered;
  F-1's taxonomy stands as the record. The rendering as it
  stands remains the only rendering any existing measurement
  describes, and no intervention slice of any class in §4
  exists.
- **If yes, for a named class:** F-3 opens for that class
  only. Under SI-80 §3.5, F-3 additionally requires F-2
  merged and the intervention specified narrowly enough to
  be one PR; under §3.6, any measurement of a changed
  configuration is F-4's separate, conditional work over a
  fresh corpus under the unchanged apparatus. An answer in
  scope for one named class is not an answer in scope for
  any other class, and is not a measurement, a rate, or a
  claim of any kind.

D-F1 is **OPEN**. So, for completeness, are SI-80 §4's
remaining decisions, which this slice does not pose and
does not pre-empt: D-F2 (whether and when to name a
successor candidate — none is named), D-F3 (another live
conformance capture spend, after any F-line intervention —
until answered, no F-4 capture fires), and D-F4 (whether
D-C1 stands as the F-line's threshold — this doc assumes
nothing about it beyond recording that D-C1 stands as Mark
decided it on 2026-11-02 and that SI-79's verdict under it
is not re-opened here).

## 6. The four honesty risks (SI-80 §3.4)

SI-80 §3.4 requires the F-2 slice to state these risks, not
soften them. They are carried here verbatim in substance:

1. A changed rendering is a **changed candidate
   configuration**. A result under it is a result for
   (model + rendering), not for the model SI-79 measured;
   the two are never conflated or averaged.
2. Nothing measured under a changed configuration is known
   until it is **re-measured under the C-1 harness over a
   fresh corpus** — F-1's taxonomy and SI-79's corpus can
   motivate a question; they cannot score an answer.
3. **SI-76 §3.3(6) applies in full:** re-scoring old bodies
   is a harness check and nothing more. SI-79 bodies were
   answers to the old rendering; they are not evidence
   about a new one.
4. The line SI-76 §6.4 drew still stands: SI-75 closed
   policy tuning on live evidence, and "readiness" work
   must not smuggle in prompt engineering as a decided
   plan. F-2 exists precisely because admissibility is
   undecided — the doc presents it as an open question with
   its risks, **not as a decided plan and with no predicted
   improvement, in any direction, anywhere**.

The wider F-line honesty rules (SI-80 §5) bind this slice
as they bind every F-slice, in substance: offline analysis
is never live evidence and earns no §18.1 row, no §18.3
claim, no L-3 unlock, and no live authorisation; D-C1 is
not weakened or re-cut and the SI-79 verdict stands; no
re-scoring of old bodies is presented as a new measurement;
class counts from F-1 are descriptive statistics of a known
corpus, never a candidate property beyond that corpus and
never a prediction about any other model or configuration;
no candidate is claimed to exist, be available, or pass;
bodies live once in `docs/SI-79-corpus.json` and are
referenced by hash, never duplicated. Terminology:
**machine learning**, never "custom LLM software"; the
served model is a specialised machine learning model. CPU
only; no GPU, no cloud, no money.

## 7. What this slice does not do

- It does not answer D-F1, recommend an answer, or present
  either answer as the expected one.
- It names no intervention as approved, scheduled, or
  expected to help, and contains no effect estimate and no
  predicted improvement, in any direction, anywhere.
- It builds nothing, changes no rendering, runs nothing,
  captures nothing, and measures nothing. No code, test,
  corpus, configuration, §8/§9 contract, §15.2 key, gate,
  registry, queue, or SSE change rides along in this slice.
- It earns no claim and no unlock: no §18.1 row, no §18.3
  progress, no readiness verdict, no threshold implication.
  Unlocks remain `[]`.
- It is not a plan for what happens after D-F1 is answered.
  SI-80 §3.2 already states the standing rule for the line
  as a whole — if the conditional slices never activate, the
  loop does F-1, poses F-2, stops, and reports; it does not
  invent replacement work — and this doc adds nothing to
  that rule in either direction.

The question now stands posed, with its classes, its risks,
and its consequences stated. Answering it is Mark's.

## 8. Sources and code references

All on `main` at
`be96961a3a659f3e2482fc678a90f7509328b5e3` unless a doc
number is given.

- `docs/SI-80-next-frontier-rfc.md` — F-line: §3.2 order and
  slice discipline, §3.4 F-2 (scope, the four honesty risks,
  acceptance, honesty gate), §3.5 F-3 conditions, §3.6 F-4
  conditions, §4 D-F1–D-F4, §5 honesty rules, §6.4 rejected
  prompt/perception tuning as a decided plan.
- `docs/SI-81-f1-failure-taxonomy.md`,
  `docs/SI-81-taxonomy.json` — F-1 as merged (PR #88):
  stage-1 classes 12 / 5 / 1 summing to 18, stage-4 listing
  of 2, control group of 10, `unclassified` 0; §9 of SI-81
  hands the admissibility question to F-2.
- `docs/SI-79-c2-conformance-record.md`,
  `docs/SI-79-measurements.json`, `docs/SI-79-corpus.json` —
  C-2 as measured: 10/30 (0.3333), BELOW THRESHOLD against
  D-C1, histogram 18/0/0/2/0/0, unlocks `[]`.
- `docs/SI-76-next-frontier-rfc.md` — §3.3(6) no-reuse of
  recorded responses beyond a harness check, §6.3 §8/§9
  boundary, §6.4 tuning rejection.
- `docs/SI-75-l4-tuning-from-live-evidence.md` — §2.7 root
  cause (model-output-content failure), §3 empty tuning
  list: policy tuning closed on live evidence.
- `docs/SI-63-gaia-propose-hearth-proposer-rfc.md` — §6
  perception assembly (§6.1 sources, §6.3 bounds, §6.5
  gateway request, §6.6 standing instructions), §7 tool
  catalogue rendering, §8 contract, §9 parser, §17
  perception-poisoning posture, §18.1 rows, §18.3 claim.
- `si/gaia/proposer.py` — the rendering as code:
  `assemble_perception`, `render_perception_text`,
  `render_catalogue`, `render_catalogue_text`,
  `STANDING_INSTRUCTIONS`, `OUTPUT_CONTRACT_INSTRUCTION`,
  `build_gateway_request`. Read for this doc; modified by
  nothing in this slice.
- `docs/SI-40-a1-feasibility-measurement.md` — A-1 measured;
  A-2/A-3/A-4 unapproved.

End of SI-82. The deliverable is the question, posed:
D-F1 (§5), over the classes in §4, under the risks in §6.
Nothing here answers it.

## Honesty labels carried forward

§18.3 **NOT earned**; P5-R1 / P5-R9 **UNMEASURED**; live
model-authored intent **OCCURRED, scoped to SI-74 only**;
L-3 **LOCKED**; A-2 / A-3 / A-4 **UNAPPROVED**; unlocks `[]`.
