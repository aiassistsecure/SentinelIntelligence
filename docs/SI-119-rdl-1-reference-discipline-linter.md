# SI-119 — RDL-1: the reference-discipline linter

**Build record.** SI-111 §2 designed this slice; lead
decision **D-111-1 is resolved per the RFC's
recommendation**: RDL-1 lands **in this repo** as offline
corpus tooling, beside the SI-83 training-journal
file-form tooling. Its rules are this repo's recorded
failure classes, priced in this repo's parser vocabulary,
against this repo's registry; it calls no trainer and so
activates nothing deferred. Recorded here under the
2026-11-11 standing directive (methodology calls are the
lead's; Mark spectates) and SI-111 §7.

**D-111-6, restated for this slice:** D-90-2 and D-90-3
remain **DEFERRED** and are activated by nothing in
SI-119. RDL-1 is trainer-free by design (§2.5 of the
RFC): it reads corpus files and emits a report — no
trainer called, no training journal written, no live
NEDB touched.

## 0. Honesty table: state at this slice

| Claim | State |
|---|---|
| The linter exists and grades fixture corpora offline | **Built and fixture-tested here** (`si/gaia/ref_lint.py`, `tests/test_gaia_si119_rdl.py`) |
| Any real candidate corpus has been linted | **No** — no corpus is named, captured, or graded by this slice; no lint result over a real corpus is reported |
| Any conformance claim | **None** — §2.4's hard label binds this doc and the tool's output (below); fixture tests earn no §18.1 row and no §18.3 progress (SI-111 §10) |
| Parser, gate, registry, contract text | **Unmodified** — the linter reads the registry as data and reuses the parser's stage-4 names; it changes none of them |
| Live anything | **None** — CPU only, no network, no node, no NEDB, no secrets |

> **RDL-1 grades data, never live conformance.** No
> conformance claim may cite lint results. A lint-clean
> corpus is not evidence about any model; a lint-dirty
> corpus is not a verdict on one. The 24/30 bar falls
> only to a **trained candidate re-measured under
> D-C1** — fresh corpus, C-1 harness, D-C1 as decided —
> which is a training-side event in Mark's lane.
> RDL-1's entire claim is: *the data stopped teaching
> the recorded mistakes*, stated per rule, over a named
> corpus, by hash. (SI-111 §2.4, verbatim; the label is
> carried in the tool's report record and both
> renderings, and any doc citing an RDL-1 report carries
> it with the citation, per §10.)

## 1. What was built

- `si/gaia/ref_lint.py` (new) — the linter: corpus
  loading, the nine §2.2 rules, the §2.3 eligibility
  report, text/JSON renderings, and a module CLI.
- `si/cli.py` (wiring only) — `si gaia-lint <corpus>
  [--reference-set FILE] [--json]`, beside `gaia-diff`,
  sharing the module's `execute`.
- `tests/test_gaia_si119_rdl.py` (new) — fixture tests
  for every rule, the report lines, the label, and both
  CLI surfaces.

Nothing else is touched. The linter is a reader: it
opens corpus files and an optional reference-set file,
and writes only its report to stdout.

## 2. Inputs (§2.3)

**(i) Candidate corpus, C-2 shape.** A JSON object with
an `items` list (the capture shape of
`docs/SI-94-corpus.json`), a bare JSON list of rows, or
JSONL. Each row is a prompt/target pair or a target
document alone: the target is read from the first
present of `target` / `body` / `response` / `completion`
/ `document`, and may be the §8 document as text or as
an already-decoded object. A row carrying no target
fails the load, closed — an unlintable row is never
silently clean.

**(ii) The registry as data.** Argument declarations —
which arguments each tool declares, and which are
required — are read from `si/gaia/registry.py`'s
`REGISTRY` at lint time. No declaration is copied into
the linter, so the bleed rules price against the same
surface the gate prices, as it stands on the day of the
run.

**(iii) Optional citable reference set.** The A1
`citable_reference_set` shape (SI-103 §4, SI-104):
`{"seqs": [...], "empty": bool, ...}`, supplied
run-wide via `--reference-set FILE` or carried on a row
(a row's own set wins). Its members are by construction
exactly one task's tool-intent rows, which is what makes
seq-form *citability* — not merely seq form — gradable.

## 3. The rules (§2.2, implemented exactly)

| Rule | Verdict prices | Recorded name carried |
|---|---|---|
| `REF-SEQ-NONPOSITIVE` | a `depends_on` entry citing seq ≤ 0 (bare int or `{seq}` form) | `seq_nonpositive` |
| `REF-SEQ-NOT-INTENT` | a positive seq outside the supplied citable set — a non-intent row, another task's row, the task's own row, or no row; the A1 shape does not distinguish these, and the verdict says so | `seq_not_intent_row` |
| `REF-TOOL-NAME` | a string entry naming a registered tool | `depends_on_tool_name_not_journal_seq` (SI-81's genus, via SI-95) |
| `REF-SIBLING-SELF` | a sibling entry citing the citing intent's own index | `sibling_self_reference` |
| `REF-SIBLING-FORWARD` | a sibling entry citing an index > the citing intent's index — **labelled unexercised**: no recorded corpus exercised this class; it is included because SI-97's positional spec makes it decidable | `sibling_forward_reference` (or `sibling_out_of_range` when the index names no member of the document — both are the parser's own names) |
| `REF-ENTRY-MALFORMED` | an entry that is neither seq-form nor sibling-form: a boolean, a non-tool string, a float, an array, a dict naming neither designator, a dict naming both, or a non-integer designator | `entry_malformed`; where the parser has a more specific name (`entry_names_both_seq_and_sibling`, `sibling_not_an_index`, `sibling_negative_index`) the verdict's detail carries it, under this rule |
| `ARG-BLEED` | an argument **declared on exactly one registry tool** carried on a different tool; the verdict names the arg and the declaring tool, both read from the registry | the SI-101 bleed genus; its per-combination class names (`search_with_read_arg_bleed`, …) are **not** reconstructed — the combination is in the verdict's fields |
| `ARG-MISSING-REQUIRED` | a registry-required argument absent from the intent's args | SI-101 priced this where it co-occurred with the bleed genus; the rule as §2.2 states it is *a required argument absent*, and it is implemented as stated |
| `SHAPE-PROSE-PREFIX` | target text opening with prose narration before the §8 document (first non-whitespace character does not open the document) | `version_narration_prefix` — the recorded instance is a `Version N` narration line; §2.2's rule text is the general prose-opening shape, and the implementation follows the rule text |

Two scoping consequences, stated plainly. An undeclared
argument declared on **zero** tools, or on **two or
more**, is not `ARG-BLEED` — §2.2's definition is the
exactly-one-owner shape the record counted (`max_bytes`
→ `fs_read` only; `content` → `fs_write` only), and no
other argument class is invented for the remainder. And
an intent naming an **unregistered tool** draws no
argument verdicts: the registry declares nothing for it,
§2.2 has no unknown-tool rule, and none is invented
(its `depends_on` entries are still priced — the
reference rules do not need its schema).

## 4. Outputs (§2.3)

**Per-row verdicts.** For every flagged row: the rule
id, the recorded name, the location (intent index,
entry index), and the cited value — a seq, a sibling
index, a tool name, an argument name (with its declaring
tool for bleed). Rows are cited by corpus id and
position and by the target's `sha256:` hash. **Bodies
are never duplicated**: a dict-shaped malformed entry is
cited by its sorted key names, an array by its length;
no body text, prompt text, or argument *value* appears
in any verdict or count.

**Corpus eligibility report.** Corpus id and the corpus
file's hash; rows total / clean / unparseable; per-rule
counts as **rows exhibiting each rule** (with verdict
counts beside them); the correct-form counts — rows
exhibiting a seq citation **inside the citable set**, and
rows exhibiting a sibling citation **strictly-earlier
and non-self** (entry counts beside them); and the
absent-references count **as its own line, never folded
into a pass figure**: rows whose own task shape declares
`expects_dependency` (row-level, or in the row's
`task_shape`) and whose document carries no `depends_on`
entry at all. A corpus declaring no expectation reports
that line as 0 *by construction, not by measurement*,
and the report says which of the two it is.

**Ungraded, never assumed.** A positive seq citation
with no citable set to price it against is counted as
*seq-form, citability ungraded* — not correct, not
flagged. The linter grades form and citability against
a stated reference set, never "would the gate have
admitted this" (§2.2's scoping fact, from SI-95's
structural finding).

**The document under a prose prefix is still linted.**
A prose-prefixed target draws its `SHAPE-PROSE-PREFIX`
verdict, and the §8 document after the prefix, when
present and decodable, is graded on its own forms
beside it. Describing the shape is not a re-score: no
verdict is suppressed because another fired first, and
no stage ordering is imported from the parser — the
linter is not the parser and claims no stage semantics.

## 5. What this slice is not (§2.5, §10)

Not a harness change (no live proposal is linted), not
trainer wiring (no trainer, no training journal, no
live NEDB; D-90-2/D-90-3 untouched and still DEFERRED),
not a conformance instrument (§2.4 binds). No threshold
is weakened, re-cut, or re-based; no old body is
re-scored as a new measurement — recorded corpora, if
ever passed to this tool, are inputs to offline
analysis only, and no such run is reported here.
Terminology is machine learning throughout; the work
was CPU-only, with no GPU, no cloud, no money, and no
secrets.

## 6. Tests

`tests/test_gaia_si119_rdl.py`: every rule fires on its
recorded shape with its recorded name; bleed prices
against the live registry (the expectation recomputed
from `REGISTRY` in the test) including the
zero/multi-owner scoping; correct forms count only when
priced; row-set precedence over the run-wide set; the
absent-references line's isolation; the hard label
verbatim in the record and both renderings, with body
text proven absent from the report; the accepted corpus
shapes and the fail-closed load; an unparseable target
counted, never cleaned; both CLI surfaces, with a dirty
corpus exiting 0 (a report, not a failure) and an
unreadable corpus exiting 1. Builder ran the new file
plus the registry/gaia neighbour subsets; the one full
suite per PR belongs to the reviewer.

End of SI-119 build record. RDL-1 is the fourth of
SI-111's four slices; with it, the RFC's designed set
is built. What it makes possible is stated by §2.4 and
no further: when the next candidate's corpus is frozen
in Mark's lane, the data can be required to have stopped
teaching the recorded mistakes — per rule, over a named
corpus, by hash.
