# si-membench results — memory benchmark slice v0

- Slice: SI-130 (MEMBENCH-1 memory benchmark slice v0)
- Run: 2026-11-24T00:15:27.224046-05:00 → 2026-11-24T00:15:28.542492-05:00 (America/New_York)
- Fixture: 16 sessions, 40 questions, 4 personas; sha256 sha256:392ffd9e80332d5ed37a940fb831b1cfe576f13b1a7445a708d22323755eb4c8
- Retrieval depth: top-3

## Headline — top-k session recall

| system | hits | n | rate |
|---|---|---|---|
| journal | 40 | 40 | 1.0000 |
| filesystem | 40 | 40 | 1.0000 |

## Per-category

| category | n | journal hits | journal rate | filesystem hits | filesystem rate |
|---|---|---|---|---|---|
| single-hop | 14 | 14 | 1.0000 | 14 | 1.0000 |
| cross-session | 10 | 10 | 1.0000 | 10 | 1.0000 |
| temporal-update | 8 | 8 | 1.0000 | 8 | 1.0000 |
| distractor-resistance | 8 | 8 | 1.0000 | 8 | 1.0000 |

## Methods

- **journal**: Sessions are stored in a throwaway .si bundle (init_bundle, journal_backend='embedded') via journal.record_experience('membench_session', {persona, session_index, text}, ''). Retrieval fetches every session record with one exact-match backend.find('experiences', filter={'kind': 'membench_session'}) and ranks the record payloads with the shared scorer. The bundle is deleted after the run.
- **filesystem**: Sessions are materialized as one <session-id>.txt file per session in a fresh directory. Retrieval reads every file back from disk and ranks the contents with the shared scorer. The directory is deleted after the run.
- **ranking**: SHARED, identical for both systems: lowercase the question and the session text, strip punctuation, drop the fixed stopword set defined in si.gaia.membench; score each session as the tuple (distinct matched query tokens, total matched occurrences); sort by score descending; ties break by the session's order in the fixture. Top-k of that order is the retrieval.

## Honesty

- Self-authored synthetic slice, small N: 4 fictional personas, 16 sessions, 40 questions, all written for this slice. Nothing here is a sample of real conversations.
- NOT LoCoMo and NOT comparable to published LoCoMo figures. Published memory-benchmark numbers (Mem0, Zep, Letta, and others) are mutually contested and there is no neutral leaderboard; do not place a rate from this slice next to one of theirs.
- Retrieval-only: the measurement is top-k session recall (does a gold fact string appear in a retrieved session text). It is not end-to-end QA — no model reads the retrieved sessions and no answer is generated or judged.
- No model is involved in v0: no inference, no embeddings, no learned ranker, no judge.
- The ranking function is SHARED and identical for both systems by design (keyword overlap, defined in si.gaia.membench); what differs is the store — a directory of plain .txt files vs the SI journal.
- The journal has no native text search: its backend offers exact-match find only, and this slice adds none. The journal retriever fetches all session records by exact-match kind and ranks their payloads in memory with the shared scorer.
- This closes SI-123's 'plain-filesystem baseline: NOT RUN' line for the memory-retrieval domain only. It says nothing about the SI-123 conformance domain.

## Per-question outcomes

| question | category | persona | journal hit | journal retrieved | filesystem hit | filesystem retrieved |
|---|---|---|---|---|---|---|
| q01 | single-hop | Priya Nair | True | priya-s1, priya-s3, priya-s2 | True | priya-s1, priya-s3, priya-s2 |
| q02 | single-hop | Priya Nair | True | priya-s1, priya-s3, priya-s2 | True | priya-s1, priya-s3, priya-s2 |
| q03 | single-hop | Priya Nair | True | priya-s3, priya-s4, priya-s1 | True | priya-s3, priya-s4, priya-s1 |
| q04 | single-hop | Priya Nair | True | priya-s4, priya-s1, priya-s3 | True | priya-s4, priya-s1, priya-s3 |
| q05 | cross-session | Priya Nair | True | priya-s1, priya-s2, priya-s3 | True | priya-s1, priya-s2, priya-s3 |
| q06 | cross-session | Priya Nair | True | priya-s2, priya-s1, priya-s3 | True | priya-s2, priya-s1, priya-s3 |
| q07 | temporal-update | Priya Nair | True | priya-s1, priya-s3, diego-s1 | True | priya-s1, priya-s3, diego-s1 |
| q08 | temporal-update | Priya Nair | True | priya-s4, priya-s1, priya-s2 | True | priya-s4, priya-s1, priya-s2 |
| q09 | distractor-resistance | Priya Nair | True | priya-s1, priya-s3, priya-s2 | True | priya-s1, priya-s3, priya-s2 |
| q10 | distractor-resistance | Priya Nair | True | priya-s1, priya-s3, tomoko-s1 | True | priya-s1, priya-s3, tomoko-s1 |
| q11 | single-hop | Marcus Bell | True | marcus-s1, marcus-s2, marcus-s3 | True | marcus-s1, marcus-s2, marcus-s3 |
| q12 | single-hop | Marcus Bell | True | marcus-s1, marcus-s2, marcus-s3 | True | marcus-s1, marcus-s2, marcus-s3 |
| q13 | single-hop | Marcus Bell | True | marcus-s1, marcus-s4, marcus-s2 | True | marcus-s1, marcus-s4, marcus-s2 |
| q14 | cross-session | Marcus Bell | True | marcus-s2, marcus-s1, marcus-s3 | True | marcus-s2, marcus-s1, marcus-s3 |
| q15 | cross-session | Marcus Bell | True | marcus-s2, marcus-s1, marcus-s3 | True | marcus-s2, marcus-s1, marcus-s3 |
| q16 | cross-session | Marcus Bell | True | marcus-s3, marcus-s1, marcus-s2 | True | marcus-s3, marcus-s1, marcus-s2 |
| q17 | temporal-update | Marcus Bell | True | marcus-s1, marcus-s3, marcus-s2 | True | marcus-s1, marcus-s3, marcus-s2 |
| q18 | temporal-update | Marcus Bell | True | marcus-s4, marcus-s1, marcus-s2 | True | marcus-s4, marcus-s1, marcus-s2 |
| q19 | distractor-resistance | Marcus Bell | True | marcus-s1, marcus-s3, marcus-s2 | True | marcus-s1, marcus-s3, marcus-s2 |
| q20 | distractor-resistance | Marcus Bell | True | marcus-s1, marcus-s2, marcus-s3 | True | marcus-s1, marcus-s2, marcus-s3 |
| q21 | single-hop | Tomoko Abe | True | tomoko-s1, tomoko-s2, tomoko-s3 | True | tomoko-s1, tomoko-s2, tomoko-s3 |
| q22 | single-hop | Tomoko Abe | True | tomoko-s2, tomoko-s1, tomoko-s4 | True | tomoko-s2, tomoko-s1, tomoko-s4 |
| q23 | single-hop | Tomoko Abe | True | tomoko-s4, tomoko-s1, tomoko-s2 | True | tomoko-s4, tomoko-s1, tomoko-s2 |
| q24 | cross-session | Tomoko Abe | True | tomoko-s1, tomoko-s2, tomoko-s4 | True | tomoko-s1, tomoko-s2, tomoko-s4 |
| q25 | cross-session | Tomoko Abe | True | tomoko-s2, tomoko-s1, tomoko-s3 | True | tomoko-s2, tomoko-s1, tomoko-s3 |
| q26 | cross-session | Tomoko Abe | True | tomoko-s1, tomoko-s3, tomoko-s4 | True | tomoko-s1, tomoko-s3, tomoko-s4 |
| q27 | temporal-update | Tomoko Abe | True | tomoko-s1, tomoko-s3, tomoko-s2 | True | tomoko-s1, tomoko-s3, tomoko-s2 |
| q28 | temporal-update | Tomoko Abe | True | tomoko-s1, tomoko-s4, diego-s1 | True | tomoko-s1, tomoko-s4, diego-s1 |
| q29 | distractor-resistance | Tomoko Abe | True | tomoko-s1, tomoko-s2, tomoko-s3 | True | tomoko-s1, tomoko-s2, tomoko-s3 |
| q30 | distractor-resistance | Tomoko Abe | True | tomoko-s1, tomoko-s2, tomoko-s3 | True | tomoko-s1, tomoko-s2, tomoko-s3 |
| q31 | single-hop | Diego Fuentes | True | diego-s1, diego-s2, diego-s3 | True | diego-s1, diego-s2, diego-s3 |
| q32 | single-hop | Diego Fuentes | True | diego-s2, diego-s1, diego-s4 | True | diego-s2, diego-s1, diego-s4 |
| q33 | single-hop | Diego Fuentes | True | diego-s4, diego-s1, diego-s2 | True | diego-s4, diego-s1, diego-s2 |
| q34 | single-hop | Diego Fuentes | True | diego-s1, diego-s3, diego-s2 | True | diego-s1, diego-s3, diego-s2 |
| q35 | cross-session | Diego Fuentes | True | diego-s2, diego-s1, diego-s3 | True | diego-s2, diego-s1, diego-s3 |
| q36 | cross-session | Diego Fuentes | True | diego-s1, diego-s2, diego-s3 | True | diego-s1, diego-s2, diego-s3 |
| q37 | temporal-update | Diego Fuentes | True | diego-s1, diego-s3, priya-s4 | True | diego-s1, diego-s3, priya-s4 |
| q38 | temporal-update | Diego Fuentes | True | diego-s1, diego-s2, diego-s4 | True | diego-s1, diego-s2, diego-s4 |
| q39 | distractor-resistance | Diego Fuentes | True | diego-s1, diego-s4, diego-s2 | True | diego-s1, diego-s4, diego-s2 |
| q40 | distractor-resistance | Diego Fuentes | True | diego-s1, diego-s2, diego-s3 | True | diego-s1, diego-s2, diego-s3 |
