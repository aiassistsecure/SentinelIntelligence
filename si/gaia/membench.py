"""SI-130 (MEMBENCH-1) — the memory-benchmark slice v0
(``si gaia-membench``).

What this module is, stated honestly: a small, self-authored,
synthetic memory-RETRIEVAL measurement. It takes one fixed
fixture — 4 fictional personas x 4 conversation sessions
(16 sessions, :data:`membench_fixture.json` shipped beside
this module) and 40 questions about facts stated in those
sessions — stores the 16 sessions in two different stores,
and asks every question against the whole 16-session corpus
in each store. A question is a HIT when any of its gold fact
strings appears (case-insensitive substring) in any of the
top-k retrieved session texts. That is the whole measurement.

What it is NOT:

- **Retrieval-only.** It measures top-k session recall. There
  is no model, no answer generation, and no judge: nothing
  reads the retrieved sessions and composes an answer, so
  this is not end-to-end question answering and says nothing
  about one.
- **NOT LoCoMo, and NOT comparable to published LoCoMo
  figures.** The fixture is self-authored and synthetic, and
  N is small (40 questions). Published memory-benchmark
  numbers (Mem0, Zep, Letta, and others) are mutually
  contested and there is no neutral leaderboard; a rate from
  this slice must never be placed next to one of theirs.
- **Not a store-vs-store ranking contest.** The ranking
  function is SHARED and identical for both systems by
  design — a plain keyword-overlap scorer defined in this
  module. What differs is the store: a directory of plain
  ``.txt`` files vs the SI journal. Any rate difference is a
  difference in what the store handed the shared ranker,
  nothing more.
- **No text search was added anywhere.** The journal backend
  offers exact-match ``find`` only — there is no native text
  search in the journal, and this slice adds none. The
  journal retriever fetches every stored session record with
  one exact-match find on the record kind and ranks the
  payloads in memory with the shared scorer, exactly as the
  filesystem retriever ranks the file contents it read back
  from disk.

This slice closes SI-123's "plain-filesystem baseline:
NOT RUN" line for the memory-retrieval domain only. It says
nothing about the SI-123 conformance domain.

Deterministic throughout: no randomness, no network, stdlib
only, CPU only. Terminology: machine learning infrastructure
throughout.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo

SLICE = "SI-130 (MEMBENCH-1 memory benchmark slice v0)"

DEFAULT_K = 3

#: The fixture as shipped beside this module.
DEFAULT_FIXTURE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "membench_fixture.json")

#: The record kind under which sessions are stored in the
#: journal (see :meth:`JournalRetriever._build`).
JOURNAL_KIND = "membench_session"

#: The fixed stopword set for the shared tokenizer. Defined
#: in code, identical for every run and both systems.
STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "what", "which", "who", "whom", "whose", "where", "when",
    "why", "how", "does", "do", "did", "has", "have", "had",
    "in", "on", "at", "of", "to", "for", "from", "by", "with",
    "and", "or", "as", "it", "its", "his", "her", "their",
    "he", "she", "they", "now", "any", "every", "s",
})

_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")

#: The honesty block. These strings ride in every results
#: document and every summary, verbatim.
HONESTY = [
    "Self-authored synthetic slice, small N: 4 fictional "
    "personas, 16 sessions, 40 questions, all written for "
    "this slice. Nothing here is a sample of real "
    "conversations.",
    "NOT LoCoMo and NOT comparable to published LoCoMo "
    "figures. Published memory-benchmark numbers (Mem0, "
    "Zep, Letta, and others) are mutually contested and "
    "there is no neutral leaderboard; do not place a rate "
    "from this slice next to one of theirs.",
    "Retrieval-only: the measurement is top-k session "
    "recall (does a gold fact string appear in a retrieved "
    "session text). It is not end-to-end QA — no model "
    "reads the retrieved sessions and no answer is "
    "generated or judged.",
    "No model is involved in v0: no inference, no "
    "embeddings, no learned ranker, no judge.",
    "The ranking function is SHARED and identical for both "
    "systems by design (keyword overlap, defined in "
    "si.gaia.membench); what differs is the store — a "
    "directory of plain .txt files vs the SI journal.",
    "The journal has no native text search: its backend "
    "offers exact-match find only, and this slice adds "
    "none. The journal retriever fetches all session "
    "records by exact-match kind and ranks their payloads "
    "in memory with the shared scorer.",
    "This closes SI-123's 'plain-filesystem baseline: "
    "NOT RUN' line for the memory-retrieval domain only. "
    "It says nothing about the SI-123 conformance domain.",
]

METHODS = {
    "journal": (
        "Sessions are stored in a throwaway .si bundle "
        "(init_bundle, journal_backend='embedded') via "
        "journal.record_experience('membench_session', "
        "{persona, session_index, text}, ''). Retrieval "
        "fetches every session record with one exact-match "
        "backend.find('experiences', filter={'kind': "
        "'membench_session'}) and ranks the record payloads "
        "with the shared scorer. The bundle is deleted "
        "after the run."),
    "filesystem": (
        "Sessions are materialized as one <session-id>.txt "
        "file per session in a fresh directory. Retrieval "
        "reads every file back from disk and ranks the "
        "contents with the shared scorer. The directory is "
        "deleted after the run."),
    "ranking": (
        "SHARED, identical for both systems: lowercase the "
        "question and the session text, strip punctuation, "
        "drop the fixed stopword set defined in "
        "si.gaia.membench; score each session as the tuple "
        "(distinct matched query tokens, total matched "
        "occurrences); sort by score descending; ties break "
        "by the session's order in the fixture. Top-k of "
        "that order is the retrieval."),
}


# ---------------------------------------------------------------------------
# The shared tokenizer + scorer
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list:
    """Lowercase, strip punctuation, drop stopwords.

    The one tokenizer for questions and session texts, in
    both systems. Possessives collapse with the punctuation
    strip ("Fuentes's" -> tokens "fuentes" + "s", and "s"
    is a stopword)."""
    cleaned = _PUNCT_RE.sub(" ", text.lower())
    return [tok for tok in cleaned.split() if tok not in STOPWORDS]


def score_session(query_tokens: list, session_text: str) -> tuple:
    """The shared score: (distinct matched query tokens,
    total matched occurrences) for one session text."""
    session_tokens = tokenize(session_text)
    counts = {}
    for tok in session_tokens:
        counts[tok] = counts.get(tok, 0) + 1
    distinct = 0
    total = 0
    for tok in set(query_tokens):
        n = counts.get(tok, 0)
        if n:
            distinct += 1
            total += n
    return (distinct, total)


def rank_sessions(query_text: str, sessions: list, k: int) -> list:
    """Rank ``sessions`` (a list of (session_id, text) in
    fixture order) for ``query_text`` with the shared scorer
    and return the top-k as a list of (session_id, text).

    Sort is by score descending; Python's sort is stable, so
    equal scores keep fixture order — the deterministic
    tie-break, identical for both systems because both feed
    the ranker the corpus in fixture order."""
    query_tokens = tokenize(query_text)
    scored = [(score_session(query_tokens, text), sid, text)
              for sid, text in sessions]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [(sid, text) for _score, sid, text in scored[:k]]


# ---------------------------------------------------------------------------
# The two retrievers — same interface, different stores
# ---------------------------------------------------------------------------

class FilesystemRetriever:
    """The plain-filesystem store: one ``<id>.txt`` file per
    session in a directory; retrieval reads the files back
    from disk and ranks their contents with the shared
    scorer."""

    name = "filesystem"

    def __init__(self, sessions: list):
        self._order = [s["id"] for s in sessions]
        self._dir = tempfile.mkdtemp(prefix="si-membench-fs-")
        for s in sessions:
            path = os.path.join(self._dir, s["id"] + ".txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(s["text"])

    def retrieve(self, question_text: str, k: int = DEFAULT_K) -> list:
        sessions = []
        for sid in self._order:
            path = os.path.join(self._dir, sid + ".txt")
            with open(path, encoding="utf-8") as fh:
                sessions.append((sid, fh.read()))
        return rank_sessions(question_text, sessions, k)

    def close(self) -> None:
        shutil.rmtree(self._dir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class JournalRetriever:
    """The SI journal store: sessions are recorded as
    experiences in a throwaway bundle on the embedded
    backend; retrieval fetches them with one exact-match
    ``find`` on the record kind and ranks the payloads with
    the shared scorer. No text search exists in the journal
    and none is added here."""

    name = "journal"

    def __init__(self, sessions: list):
        from si.bundle import init_bundle, load_bundle
        from si.journal import open_journal

        self._order = [s["id"] for s in sessions]
        self._id_by_key = {(s["persona"], s["session_index"]): s["id"]
                           for s in sessions}
        self._tmp = tempfile.mkdtemp(prefix="si-membench-journal-")
        bundle_dir = init_bundle(
            os.path.join(self._tmp, "membench.si"),
            name="membench", journal_backend="embedded")
        self._journal = open_journal(bundle_dir, load_bundle(bundle_dir))
        for s in sessions:
            self._journal.record_experience(
                JOURNAL_KIND,
                {"persona": s["persona"],
                 "session_index": s["session_index"],
                 "text": s["text"]},
                "")

    def _corpus(self) -> list:
        rows = self._journal.backend.find(
            "experiences", filter={"kind": JOURNAL_KIND})
        by_id = {}
        for row in rows:
            record = row["record"]
            sid = self._id_by_key[(record["persona"],
                                   record["session_index"])]
            by_id[sid] = record["text"]
        return [(sid, by_id[sid]) for sid in self._order]

    def retrieve(self, question_text: str, k: int = DEFAULT_K) -> list:
        return rank_sessions(question_text, self._corpus(), k)

    def close(self) -> None:
        try:
            self._journal.close()
        finally:
            shutil.rmtree(self._tmp, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def _contains(haystacks: list, needle: str) -> bool:
    lowered = needle.lower()
    return any(lowered in text.lower() for text in haystacks)


def grade_question(question: dict, retrieved: list) -> dict:
    """Grade one system's retrieval for one question.

    ``retrieved`` is the retriever's top-k list of
    (session_id, text). Hit iff any gold string is a
    case-insensitive substring of any retrieved text.
    ``distractor_retrieved`` records whether the
    similar-but-wrong string rode along in the top-k: the
    question's ``distractor_fact`` for the
    distractor-resistance category, its ``superseded_fact``
    for temporal-update (the stale value surfacing next to —
    or instead of — the current one), and False for the
    other categories, which carry neither field."""
    texts = [text for _sid, text in retrieved]
    hit = any(_contains(texts, gold) for gold in question["gold_facts"])
    wrong = question.get("distractor_fact") or question.get(
        "superseded_fact")
    return {
        "hit": hit,
        "gold_in_top1": bool(retrieved) and any(
            gold.lower() in retrieved[0][1].lower()
            for gold in question["gold_facts"]),
        "retrieved": [sid for sid, _text in retrieved],
        "distractor_retrieved": bool(wrong) and _contains(texts, wrong),
    }


def _aggregate(per_question: list, system: str) -> dict:
    n = len(per_question)
    hits = sum(1 for q in per_question if q[system]["hit"])
    per_category = {}
    for q in per_question:
        bucket = per_category.setdefault(q["category"],
                                         {"hits": 0, "n": 0})
        bucket["n"] += 1
        if q[system]["hit"]:
            bucket["hits"] += 1
    for bucket in per_category.values():
        bucket["rate"] = (bucket["hits"] / bucket["n"]
                          if bucket["n"] else None)
    return {"hits": hits, "n": n,
            "rate": (hits / n if n else None),
            "per_category": per_category}


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------

def load_fixture(fixture_path: str = None) -> dict:
    path = fixture_path or DEFAULT_FIXTURE_PATH
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def fixture_stats(fixture: dict, fixture_bytes: bytes) -> dict:
    per_category = {}
    for q in fixture["questions"]:
        per_category[q["category"]] = per_category.get(
            q["category"], 0) + 1
    return {
        "sessions": len(fixture["sessions"]),
        "questions": len(fixture["questions"]),
        "personas": len(fixture.get("personas", [])),
        "per_category": per_category,
        "fixture_sha256": "sha256:" + hashlib.sha256(
            fixture_bytes).hexdigest(),
    }


def _now_ny() -> datetime:
    return datetime.now(ZoneInfo("America/New_York"))


def run_membench(fixture_path: str = None, out_dir: str = None,
                 k: int = DEFAULT_K) -> dict:
    """Run the slice: both stores over every question of the
    fixture, graded by the shared rule. Returns the results
    dict; when ``out_dir`` is given, also writes
    ``results.json`` and ``summary.md`` there."""
    path = fixture_path or DEFAULT_FIXTURE_PATH
    with open(path, "rb") as fh:
        fixture_bytes = fh.read()
    fixture = json.loads(fixture_bytes.decode("utf-8"))
    sessions = fixture["sessions"]
    questions = fixture["questions"]

    started = _now_ny()
    per_question = []
    retrievers = [JournalRetriever(sessions),
                  FilesystemRetriever(sessions)]
    try:
        for q in questions:
            row = {"id": q["id"], "category": q["category"],
                   "persona": q["persona"],
                   "source_session": q["source_session"]}
            for retriever in retrievers:
                row[retriever.name] = grade_question(
                    q, retriever.retrieve(q["question"], k))
            per_question.append(row)
    finally:
        for retriever in retrievers:
            retriever.close()
    finished = _now_ny()

    results = {
        "record": "si.gaia.membench.results",
        "slice": SLICE,
        "run": {
            "started_america_new_york": started.isoformat(),
            "finished_america_new_york": finished.isoformat(),
        },
        "fixture": fixture_stats(fixture, fixture_bytes),
        "k": k,
        "systems": {
            "journal": _aggregate(per_question, "journal"),
            "filesystem": _aggregate(per_question, "filesystem"),
        },
        "per_question": per_question,
        "honesty": list(HONESTY),
        "methods": dict(METHODS),
    }
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "results.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
            fh.write("\n")
        with open(os.path.join(out_dir, "summary.md"), "w",
                  encoding="utf-8") as fh:
            fh.write(render_summary(results))
    return results


def render_summary(results: dict) -> str:
    """The human-readable summary of one results document,
    in the SI-123 style: headline table, per-category table,
    methods, honesty block, per-question table."""
    fx = results["fixture"]
    systems = results["systems"]
    lines = [
        "# si-membench results — memory benchmark slice v0",
        "",
        f"- Slice: {results['slice']}",
        f"- Run: {results['run']['started_america_new_york']} "
        f"→ {results['run']['finished_america_new_york']} "
        f"(America/New_York)",
        f"- Fixture: {fx['sessions']} sessions, "
        f"{fx['questions']} questions, {fx['personas']} "
        f"personas; sha256 {fx['fixture_sha256']}",
        f"- Retrieval depth: top-{results['k']}",
        "",
        "## Headline — top-k session recall",
        "",
        "| system | hits | n | rate |",
        "|---|---|---|---|",
    ]
    for name in ("journal", "filesystem"):
        s = systems[name]
        lines.append(f"| {name} | {s['hits']} | {s['n']} | "
                     f"{s['rate']:.4f} |")
    lines += ["", "## Per-category", "",
              "| category | n | journal hits | journal rate | "
              "filesystem hits | filesystem rate |",
              "|---|---|---|---|---|---|"]
    for category, count in fx["per_category"].items():
        j = systems["journal"]["per_category"][category]
        fsys = systems["filesystem"]["per_category"][category]
        lines.append(
            f"| {category} | {count} | {j['hits']} | "
            f"{j['rate']:.4f} | {fsys['hits']} | "
            f"{fsys['rate']:.4f} |")
    lines += ["", "## Methods", ""]
    for name in ("journal", "filesystem", "ranking"):
        lines.append(f"- **{name}**: {results['methods'][name]}")
    lines += ["", "## Honesty", ""]
    for statement in results["honesty"]:
        lines.append(f"- {statement}")
    lines += ["", "## Per-question outcomes", "",
              "| question | category | persona | journal hit | "
              "journal retrieved | filesystem hit | "
              "filesystem retrieved |",
              "|---|---|---|---|---|---|---|"]
    for q in results["per_question"]:
        lines.append(
            f"| {q['id']} | {q['category']} | {q['persona']} | "
            f"{q['journal']['hit']} | "
            f"{', '.join(q['journal']['retrieved'])} | "
            f"{q['filesystem']['hit']} | "
            f"{', '.join(q['filesystem']['retrieved'])} |")
    lines.append("")
    return "\n".join(lines)


def execute(args) -> int:
    """`si gaia-membench` wiring: run the slice and print the
    two headline rates."""
    results = run_membench(fixture_path=args.fixture,
                           out_dir=args.out, k=args.k)
    j = results["systems"]["journal"]
    fsys = results["systems"]["filesystem"]
    print(f"si-membench: journal {j['hits']}/{j['n']} "
          f"({j['rate']:.4f}) · filesystem {fsys['hits']}/"
          f"{fsys['n']} ({fsys['rate']:.4f}) · "
          f"k={results['k']}"
          + (f" → {args.out}" if args.out else ""))
    return 0
