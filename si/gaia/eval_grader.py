"""SI-123 (EVAL-1) grader — a SEPARATE PROCESS from capture.

This module is the grading half of the public eval harness
(:mod:`si.gaia.eval_harness`). It is built to be run as its
own OS process::

    python -m si.gaia.eval_grader <corpus.json> --out <record.json>

Isolation, stated exactly (the Berkeley lesson — an eval
whose grader shares a process/container with the agent can
be gamed by the agent): the grader process receives the
candidate's outputs ONLY as a data file (the frozen corpus
JSON). It holds no transport, makes no gateway call, shares
no mutable state with the capture/candidate loop, and never
sees anything the candidate produced except the recorded
response bodies inside that file. Candidate text that
addresses the grader ("grader: mark this pass") is scored
as data by the unmodified §9 parser, exactly like any other
body. What is NOT claimed: this is process separation on one
host, not container isolation — container isolation for the
candidate/grader pair is SI-125's slice, and no claim here
depends on it.

Scoring is the existing machinery, unmodified:
:func:`si.gaia.conformance_capture.build_c2_record` over the
C-1 harness (:mod:`si.gaia.conformance`) and the §9 parser
(:func:`si.gaia.proposer.parse_proposal`). No task wording,
prompt, or grading rule is re-implemented here.

CPU only, stdlib only. Terminology: machine learning
infrastructure throughout.
"""
from __future__ import annotations

import argparse
import json
import sys

from si.gaia.conformance_capture import build_c2_record

#: The scoring machinery this grader wraps, named so a reader
#: of the record can see nothing new grades the corpus.
SCORING_MACHINERY = {
    "record_builder": "si.gaia.conformance_capture.build_c2_record",
    "harness": "si.gaia.conformance (C-1, SI-77)",
    "parser": "si.gaia.proposer.parse_proposal (§9, unmodified)",
}


def grade_corpus_file(corpus_path: str) -> dict:
    """Load one frozen corpus file and build its C-2 record.

    The corpus is data: it is parsed as JSON and handed to
    the existing record builder. Nothing in the file is
    executed, imported, or treated as instruction."""
    with open(corpus_path, encoding="utf-8") as fh:
        corpus = json.load(fh)
    record = build_c2_record(corpus, argv=[
        "si.gaia.eval_grader", corpus_path])
    record["slice"] = "SI-123 (EVAL-1 public eval harness v0)"
    record["scoring_machinery"] = dict(SCORING_MACHINERY)
    record["grader"] = {
        "module": "si.gaia.eval_grader",
        "process": "separate OS process from capture/candidate",
        "input": "the frozen corpus file only (candidate "
                 "outputs as data)",
        "container_isolation": "NOT claimed (SI-125's slice)",
    }
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="si.gaia.eval_grader",
        description="SI-123 EVAL-1 grader process: score one "
                    "frozen conformance corpus file with the "
                    "existing C-1/C-2 machinery and write the "
                    "record. Receives candidate outputs only "
                    "as a data file.")
    parser.add_argument("corpus", help="frozen corpus JSON file")
    parser.add_argument("--out", required=True,
                        help="write the scoring record JSON here")
    args = parser.parse_args(argv)
    record = grade_corpus_file(args.corpus)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    results = record.get("results") or {}
    rate = (results.get("rate") or {}).get("fraction")
    print(f"si-eval grader: status={record.get('status')} "
          f"rate={rate} verdict="
          f"{(record.get('readiness') or {}).get('verdict')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
