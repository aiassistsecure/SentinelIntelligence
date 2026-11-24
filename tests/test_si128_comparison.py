"""SI-128: pin the honest comparison page against the tree.

docs/SI-128-comparison.md makes load-bearing claims about
what SI itself ships today. Competitor claims on the page
are dated research and cannot be pinned to code — but SI's
own anchors can, and these tests fail if they drift:

- every repo path the page cites in backticks (``si/...``,
  ``docs/...``, ``tests/...``) exists in the tree;
- every ``si <command>`` the page cites is a registered
  subcommand in si/cli.py;
- the eval record the page prints (21/30, 0.70, BELOW
  THRESHOLD, hallucinated-reference floor failure) matches
  the graded record in docs/SI-123-live-run/results.json;
- the page's honesty markers (baselines NOT RUN, export
  not a signature) are still on the page.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "SI-128-comparison.md"
RESULTS = ROOT / "docs" / "SI-123-live-run" / "results.json"
CLI = ROOT / "si" / "cli.py"

_PATH_RE = re.compile(r"`((?:si|docs|tests)/[^`\s]+\.(?:py|md|json))`")
_CMD_RE = re.compile(r"`si ([a-z][a-z0-9-]+)`")
_PARSER_RE = re.compile(r'add_parser\("([a-z0-9-]+)"')


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def test_cited_repo_paths_exist():
    cited = sorted(set(_PATH_RE.findall(_page())))
    assert cited, "page cites no repo paths — anchors missing?"
    missing = [p for p in cited if not (ROOT / p).is_file()]
    assert not missing, f"page cites paths not in the tree: {missing}"


def test_cited_commands_are_registered():
    cited = sorted(set(_CMD_RE.findall(_page())))
    assert cited, "page cites no si commands — anchors missing?"
    registered = set(_PARSER_RE.findall(CLI.read_text(encoding="utf-8")))
    unknown = [c for c in cited if c not in registered]
    assert not unknown, f"page cites unregistered commands: {unknown}"


def test_eval_record_matches_graded_results():
    dc1 = json.loads(RESULTS.read_text(encoding="utf-8"))["dc1"]
    assert dc1["n"] == 30
    assert dc1["parse_ok"] == 21
    assert dc1["rate"]["fraction"] == "21/30"
    assert dc1["rate"]["decimal"] == 0.7
    assert dc1["verdict"] == "BELOW THRESHOLD"
    assert dc1["floor_failures"] == ["hallucinated-reference"]
    assert dc1["per_workload"]["hallucinated-reference"]["parse_ok"] == 1

    page = _page()
    for needle in ("21/30", "0.70", "BELOW", "hallucinated-reference",
                   "1/5"):
        assert needle in page, f"page lost the eval fact: {needle}"


def test_honesty_markers_present():
    page = _page()
    assert "NOT RUN" in page  # the eval baselines, stated not estimated
    assert "not a signature" in page.lower()  # the export's honest scope
    assert "OWED" in page  # the live Docker smoke, still outstanding
