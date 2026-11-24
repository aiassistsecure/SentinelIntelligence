# SI-130 — MEMBENCH-1: the memory benchmark slice v0

`si gaia-membench` · implementation: `si/gaia/membench.py` ·
fixture: `si/gaia/membench_fixture.json` · tests:
`tests/test_gaia_si130_membench.py` · measured run:
`docs/SI-130-results/` (`results.json` + `summary.md`)

## Purpose

SI-123's results document carried a plain-filesystem baseline
as **NOT RUN**. This slice runs that baseline for one domain —
memory retrieval — and, in doing so, gives the SI journal its
first published retrieval measurement of any kind: store a set
of conversation sessions, ask questions against the whole
corpus, and measure whether the session carrying the answer is
retrieved.

The measurement is deliberately narrow, and this document
states its limits in full (see **Honesty** below): it is
retrieval-only top-k session recall over a small self-authored
synthetic fixture. No model is involved, no answer is
generated, and nothing here is comparable to published
memory-benchmark figures.

## Fixture design

The fixture is fixed and shipped in the repo
(`si/gaia/membench_fixture.json`, sha256 recorded in every
results document). It was authored for this slice; every
person, fact, and conversation in it is fictional.

- **4 personas** — Priya Nair, Marcus Bell, Tomoko Abe, Diego
  Fuentes — each with **4 conversation sessions** (16 sessions
  total). Each session is a short dialogue (6–10 turns)
  embedding several concrete facts: city, job, pet, hobby,
  family, plans, preferences.
- **40 questions**, each naming its persona, each carrying one
  or more `gold_facts`: short strings that appear **verbatim**
  in the cited `source_session` text. Grading is a
  case-insensitive substring check, so the verbatim bar is
  stronger than the grading bar. The retrieval corpus for
  every question is **all 16 sessions in one shared store** —
  no per-persona partitioning.
- **Categories:**
  - **single-hop (14)** — a fact stated once, asked directly.
    For these (and cross-session) questions the gold string
    appears in exactly one session corpus-wide; the fixture
    tests pin that.
  - **cross-session (10)** — the gold fact sits in session
    j < 4 of its persona while that persona's later sessions
    talk about other topics: the fact is buried under the
    persona's own later conversation.
  - **temporal-update (8)** — an attribute (city, employer,
    phone carrier, car, gym, sport) is stated in an early
    session and **changed** in a later one. `gold_facts` is
    the latest value, `source_session` is the latest session,
    and the question additionally carries `superseded_fact`:
    the old value's string, present in an earlier session.
    The fixture tests pin that the gold string appears in no
    earlier session of that persona.
  - **distractor-resistance (8)** — another persona (or the
    same persona, a different attribute) holds a very similar
    fact with a different value: two dogs (a golden retriever
    named Biscuit vs a black Labrador named Pepper), two cats,
    two languages being learned, two Sunday volunteering
    posts. Each question carries `distractor_fact` and
    `distractor_session` naming the similar-but-wrong string
    and where it lives.

The sessions are written to be keyword-retrievable — this is a
measurement, not a trap — but the distractors genuinely compete
on keywords, and the temporal questions' stale values are
stated in the persona's own voice in earlier sessions.

## The two retrievers

Both implement the same interface —
`retrieve(question_text, k) -> [(session_id, text), …]` with
length ≤ k — and both feed the ranker the corpus in fixture
order. What differs is only the store.

- **FilesystemRetriever** materializes the sessions as
  `<session-id>.txt` files in a fresh directory, and on every
  `retrieve` reads all 16 files back from disk and ranks their
  contents. The directory is deleted when the retriever closes.
- **JournalRetriever** builds a throwaway bundle with
  `init_bundle(..., journal_backend="embedded")` in a temp
  dir and stores each session with
  `journal.record_experience("membench_session",
  {"persona": …, "session_index": …, "text": …}, "")`.
  Retrieval is one exact-match
  `journal.backend.find("experiences",
  filter={"kind": "membench_session"})`; the returned rows
  carry the payload under their `record` key (the journal's
  stored shape is `{kind, record, self_edit_id}` plus chain
  fields), and the payloads are mapped back to session ids by
  (persona, session_index) and ranked in memory. The journal
  is closed and the temp bundle deleted when the retriever
  closes. **The journal backend offers exact-match find only —
  there is no native text search anywhere in the repo, and
  this slice adds none.**

## The shared ranking function, stated exactly

One ranker serves both systems, by design:

1. Tokenize the question and each session text identically:
   lowercase, replace every character that is not `[a-z0-9 ]`
   with a space, split on whitespace, drop the fixed stopword
   set defined in code in `si/gaia/membench.py` (`STOPWORDS`:
   articles, pronouns, question words, common prepositions and
   auxiliaries, and `"s"` — the residue of stripped
   possessives).
2. Score each session as the tuple **(distinct matched query
   tokens, total matched occurrences)**.
3. Sort by score descending. Ties keep fixture order (Python's
   sort is stable, and both retrievers present the corpus in
   fixture order), so the ranking is fully deterministic.
4. The top-k of that order is the retrieval. Default k = 3.

## Grading rule

A question is a **hit** for a system iff any of its gold
strings is a case-insensitive substring of any of that
system's top-k retrieved session texts. Per-question results
also record the retrieved session ids, `gold_in_top1` (any
gold string in the first retrieved text), and
`distractor_retrieved`: for distractor-resistance questions,
whether the distractor string is present in the top-k texts;
for temporal-update questions, whether the **superseded**
(stale) string is present — the same field name, recording
"the wrong value rode along in the retrieval" for both
categories. It is False for the other categories, which carry
neither string.

## Re-running

```
si gaia-membench --out docs/SI-130-results
```

Options: `--fixture FILE` (default: the fixture shipped with
`si.gaia.membench`), `--out DIR` (writes `results.json` +
`summary.md`; without it, only the headline line is printed),
`--k N` (default 3). The command prints the two headline
rates. No network, no randomness, stdlib only.

## Results summary

From the measured run shipped in `docs/SI-130-results/`
(2026-11-24, America/New_York; the generated `summary.md`
there is the record — this section quotes its headlines):

| system | hits | n | rate |
|---|---|---|---|
| journal | 40 | 40 | 1.0000 |
| filesystem | 40 | 40 | 1.0000 |

Per-category, both systems are identical: single-hop 14/14,
cross-session 10/10, temporal-update 8/8,
distractor-resistance 8/8.

Read that result for exactly what it is:

- **The two systems cannot differ on this measurement unless
  a store returns different texts than it was given.** Both
  feed the identical corpus to the identical ranker; the run
  verifies end-to-end that the journal round-trips the
  sessions losslessly through `record_experience` + exact-match
  `find`, which is the store-level claim this slice exists to
  make. It is not evidence that one store retrieves "better".
- **Top-3 recall saturates on this fixture.** 16 short,
  topically distinct sessions are an easy corpus for keyword
  overlap. The less saturated signals in the same run:
  `gold_in_top1` is 26/40 for both systems, and in **all 8
  temporal-update questions the superseded (stale) value's
  session rides along in the top-3** — a ranker or a reader
  that trusts retrieval order alone would meet the old value
  alongside the new one every time. A harder fixture (more
  sessions, longer sessions, k=1 reporting) is the obvious v1
  direction; v0 reports what it measured.

## Honesty

- Self-authored synthetic slice, small N: 4 fictional
  personas, 16 sessions, 40 questions, all written for this
  slice. Nothing here is a sample of real conversations.
- NOT LoCoMo and NOT comparable to published LoCoMo figures.
  Published memory-benchmark numbers (Mem0, Zep, Letta, and
  others) are mutually contested and there is no neutral
  leaderboard; do not place a rate from this slice next to
  one of theirs.
- Retrieval-only: the measurement is top-k session recall
  (does a gold fact string appear in a retrieved session
  text). It is not end-to-end QA — no model reads the
  retrieved sessions and no answer is generated or judged.
- No model is involved in v0: no inference, no embeddings,
  no learned ranker, no judge.
- The ranking function is SHARED and identical for both
  systems by design (keyword overlap, defined in
  `si.gaia.membench`); what differs is the store — a directory
  of plain .txt files vs the SI journal.
- The journal has no native text search: its backend offers
  exact-match find only, and this slice adds none. The journal
  retriever fetches all session records by exact-match kind
  and ranks their payloads in memory with the shared scorer.
- This closes SI-123's 'plain-filesystem baseline: NOT RUN'
  line for the memory-retrieval domain only. It says nothing
  about the SI-123 conformance domain.
