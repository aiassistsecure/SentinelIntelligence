# SI-118 — DIFFVIEW-1: checkpoint diff views over the journal

**Build record.** SI-111 §5 designed this slice and ruling
D-111-5 fixed its form: an in-repo, read-only CLI over
journal prefix folds — no fork-side work, no new store.
It is the third of SI-111's four slices, sequenced after
ESC-LIVE-1 because its first real customer is the
debugging of slices 1–2 and of SI-107-class incidents
(§5.2): reconstructing "what changed between the last
good seq and the failure" was journal archaeology by
hand. That question is now one query. This slice builds
the view and nothing else: it journals nothing, resolves
nothing, claims nothing, executes nothing, and amends
nothing.

## 1. What was built

`si/gaia/diffview.py` (new), wired into the main CLI as
`si gaia-diff` beside `si gaia-escalation` (§2).

### The mechanism: prefix folds, unmodified

Prefix folds are already the journal's semantics (SI-92
§1.2: any prefix yields the state as of that prefix), and
:func:`si.gaia.queue.fold_queue` reads only its backend.
The view therefore takes a prefix fold by handing the
fold a seq-filtering read-only backend view
(`_PrefixBackend`: `find` returns only rows at or below
the prefix; writes raise). `fold_queue`, the gate, the
queue writers, and the journal are untouched — there is
no second fold implementation to drift from the first.

The journal's hash chain is verified before anything is
folded (`verify_chain`, the escalation trigger's SI-58
read discipline: no manifest-anchor demand, but no fold
from an unverified chain either). A broken chain refuses
the view, fail closed.

### What a diff renders

Given `(seq_a, seq_b]`, every row cited by its journal
seq:

- **proposals** — self-edits journaled in the window, by
  kind, with their tools — and **gate verdicts counted
  by verdict** (ADMIT / REJECT / ESCALATE);
- **resolutions and dispositions** in the window, with
  their operators;
- **claims** (`queue_claim` / `queue_recovery_reclaim`),
  **`gaia_work` submissions** (with `resolution_seq` and
  the fork work id), **execution outcomes**, and
  **materialisations** (delta rows), each tied to its
  intent seq where the row states one;
- **policy acts and integrity faults, unabridged**
  (§5.1): `operator_act` / policy rows and journaled
  `gaia_outcome` fault rows carry their full records;
  fold-derived faults (which have no row of their own)
  are reported as the fold derives them, labelled
  `kind: fold`;
- **checkpoints** in the window (the re-anchor rows are
  journaled state, and in a seam lifecycle they are the
  acts' own punctuation);
- **per-intent state transitions**: the view folds at
  every journal seq in the window at which the fold can
  change (a tool self-edit, or a queue/resolution/
  outcome experience) and diffs successive snapshots, so
  each transition is attributed to the seq that caused
  it — `escalated -> queued` at the resolution's seq,
  `queued -> refused-at-claim` at the outcome's seq.
  Resolution and disposition annotations that change no
  state (a `reject` resolution) are transitions too,
  rendered as such: the annotation is a journaled change
  even when the state stands.

### The empty window

SI-111 §5.1 poses the pair as `seq_a < seq_b`. This
slice admits `seq_a == seq_b` as the explicit empty
case — a deliberate, recorded extension: a zero-width
window (or any window whose rows move nothing) renders
`no journaled change between seq A and seq B` in text
and `"changed": false` in JSON, at exit 0. An empty
result is never silence that could be mistaken for a
successful non-empty diff, and never an error.
`seq_b < seq_a` remains a usage error. Seqs past the
tip fold as the tip.

## 2. CLI surface (design choice, recorded)

```
si gaia-diff <bundle> <seq_a> <seq_b> [--task SEQ] [--json]
python -m si.gaia.diffview <bundle> <seq_a> <seq_b> [--task SEQ] [--json]
```

One operation, so no subcommand family: the two seqs
are the command. It sits beside `gaia-escalation` in
the same conventions — a scope label printed with every
run, text and `--json` forms, exit 0 on success
(including the explicit empty window), exit 1 on a
refused view (broken chain, reversed window). `--task`
scopes the folded transitions to one task, as the
trigger's reads scope their folds; the window's raw
rows are journal-wide and cited by seq either way.

## 3. The acceptance case

The test fixture replays the `se_0012` lifecycle shape
end to end, built entirely from the programme's own
writers on a miniature embedded bundle (seam submission
under an escalate policy → the SI-113 resolution act →
the SI-116 spend over a replay transport → the S-5
outcome leg answering `refused` / `stale_state`):

- the diff across the full window renders every leg
  with its seq: proposal ESCALATE, verdict count
  `ESCALATE=1`, the resolution with its operator, the
  `gaia_work` submission tied to its `resolution_seq`,
  both outcome rows (the escalation's journaled
  non-execution answer, which precedes the resolution
  and is not rewritten, and the terminal refused
  outcome), the re-anchor checkpoints, and the state
  legs `(none) -> escalated -> queued ->
  refused-at-claim`, each at its causing seq;
- the sub-window resolution → outcome renders only the
  spend/outcome legs; proposal → resolution renders
  only the resolution leg;
- the zero-width window at the tip renders the
  explicit no-change result.

**Smoke read (one, read-only, reported as such):**
after the tests, the CLI was run once against the live
seam bundle — `gaia-diff si-seam.si 63 74` — and
rendered `se_0012`'s real lifecycle (proposal seq 64,
resolution 69, `gaia_work` 71, refused outcome 73;
chain verified, tip 74). The bundle's journal and
manifest hashes were identical before and after. A
smoke read is not a measurement and earns nothing; it
is recorded because it is the slice's first real
customer arriving on schedule.

## 4. Tests

`tests/test_gaia_si118_diffview.py` (new, 10 tests):
the acceptance case (full window, both sub-windows,
empty window); the read-only proof — every byte of the
fixture bundle identical after library and CLI runs in
both formats; chain refusal on a tampered journal;
queue-level prefix diffing from the Q-1 writers
(claim/outcome legs with causing seqs, a materialisation,
verdict counts); a fold-derived second-outcome integrity
fault surfacing in its window; a policy act rendered
unabridged; window validation; both output formats and
both parsers (module + `si` CLI).

Builder runs: new tests 10 passed; affected batches
(every test module importing `si.cli`, plus the queue,
escalation-CLI, and spend suites) 242 passed / 1
skipped. The one full suite is the reviewer's, per the
standing discipline.

## 5. Honest labels and residue

- A diff is a **projection of the journal, never new
  evidence** (SI-111 §5.2, §10). It earns no §18.1 row
  and no §18.3 progress, and every run prints that
  label. §18.3 remains **NOT earned**; P-3 (the
  fresh-loss sitting) is untouched by this slice.
- Read-only means read-only: the tool opens the journal
  directly, takes no writer's lock beyond the backend's
  own snapshot read, journals nothing, and never
  re-anchors a checkpoint or touches a manifest — the
  bundle-hash proof in §4 covers the fixture, and the
  §3 smoke read verified the same on the live bundle.
- Cost, stated plainly: transitions cost one prefix
  fold per fold-relevant seq in the window. That is the
  right price for an operator debugging view over
  incident-sized windows; it is not a hot path and is
  not priced as one.
- A seam lifecycle's fork-side claim is not journaled
  SI-side (the fork claims its own work; S-5 returns the
  outcome), so the diff shows the `gaia_work` submission
  and the outcome leg — the claim section is populated
  by the queue's own claims, where the journal has
  them. The view renders what the journal records, not
  what a tidier lifecycle would have recorded.

— SI-118 coordinator, 2026-11-13. Build only; the lead
reviews and merges.
