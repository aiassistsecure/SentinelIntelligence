"""Gaia seam label reconciliation (SI-69 §3.1/§3.2, §4.7 L-5; SI-71).

Four historical label strings still say S-5 outcome return is
UNBUILT, because that was the truth when they were written and,
for the journaled one, when the record was journaled:

- ``si.gate.TOOL_SEAM_PENDING_EXECUTION`` (ends "and S-5
  outcome return is UNBUILT") — its text is also the eligibility
  key S-3 submission checks against the journaled gate report;
- ``si.gaia.seam.REGISTRY_GAP_LABEL`` ("S-5 outcome return
  UNBUILT");
- ``si.gaia.seam_submit.SUBMITTED_STATE`` ("submitted/awaiting
  outcome (S-5 unbuilt)") — journaled verbatim in ``gaia_work``
  rows — and ``WORK_SUBMISSION_LABEL``, which repeats it;
- ``si.gaia.seam_leash.LEASH_CROSSING_LABEL`` and that module's
  docstring ("S-5 outcome return remains UNBUILT").

None of those strings is edited by this slice. Editing one
would rewrite already-journaled history or break the tests
that pin it, and SI-69 §4.7 forbids both. The S-5 doc itself
(``docs/S-5-outcome-return.md`` §2) already took this position:
"The historical S-3 label strings are untouched."

What this module adds is the reconciled, current reading,
carried beside the historical strings — the same pattern S-5
used when it put its truth in ``OUTCOME_RETURN_LABEL`` rather
than rewriting S-3's labels:

- the SI-side S-5 poller and reconciler are BUILT
  (S-5/SI-59, :mod:`si.gaia.seam_outcome`), tested against
  labelled replay/stub transports and a loopback double;
- live fork interop is UNMEASURED — no live fork endpoint
  was exercised by S-5, by S-7, or by this slice;
- no outcome returns to the fork: S-5 is the fork-to-SI return
  leg; nothing in S-5 or here sends an outcome back across
  the seam in the SI-to-fork direction, and a seam-crossing
  ADMIT's full live round trip has never been observed.

This module also carries the label for the second L-5 part:
the fork ingest half of S-6 remains UNBUILT after L-5. The
contract spec in ``docs/SI-71-l5-seam-label-reconciliation.md``
§3 is a design spec for the fork lane to build against in
``aiassistsecure/gaia``; it is not a claim that any fork
endpoint exists. The deferred S-7 quantities F-E-INGEST and
F-E-SSE stay UNMEASURED until a live fork endpoint exists
and is exercised.

CPU only; no secrets; no fork code; no live work.
"""
from __future__ import annotations

#: The historical label constants whose S-5 wording this
#: slice reconciles — named, never edited. Each entry is the
#: defining module and constant, so a reader can find the
#: historical string and then read the reconciled truth below.
HISTORICAL_S5_LABEL_SOURCES: tuple[str, ...] = (
    "si.gate.TOOL_SEAM_PENDING_EXECUTION",
    "si.gaia.seam.REGISTRY_GAP_LABEL",
    "si.gaia.seam_submit.SUBMITTED_STATE",
    "si.gaia.seam_submit.WORK_SUBMISSION_LABEL",
    "si.gaia.seam_leash.LEASH_CROSSING_LABEL",
)

#: The reconciled S-5 truth (SI-69 §3.1, §4.7 L-5 part (a)).
#: This label supplements the historical strings in
#: :data:`HISTORICAL_S5_LABEL_SOURCES`; it does not replace,
#: rewrite, or re-mean any of them, and it rewrites no
#: already-journaled record.
S5_RECONCILIATION_LABEL: str = (
    "S-5 seam label reconciliation (SI-69 §3.1, SI-71/L-5): "
    "the SI-side S-5 outcome poller and reconciler are BUILT "
    "(S-5/SI-59, si.gaia.seam_outcome) and were tested "
    "against labelled replay/stub transports and a loopback "
    "double; live fork interop is UNMEASURED (no live fork "
    "endpoint was exercised); no outcome returns to the fork "
    "— S-5 is the fork-to-SI return leg only; the historical "
    "label strings naming S-5 UNBUILT (si.gate, si.gaia.seam, "
    "si.gaia.seam_submit, si.gaia.seam_leash) are "
    "journal-time/slice-time truth, are not rewritten by this "
    "reconciliation, and no already-journaled record changes "
    "meaning")

#: The L-5 part (b) label: the fork ingest contract is a
#: design spec only. Nothing here claims a fork endpoint
#: exists, and the deferred quantities stay UNMEASURED.
FORK_INGEST_CONTRACT_LABEL: str = (
    "fork ingest half of S-6 (POST /gaia/v1/events/ingest) "
    "remains UNBUILT after SI-71/L-5: L-5 builds no fork code "
    "and claims no fork behaviour; "
    "docs/SI-71-l5-seam-label-reconciliation.md §3 is an "
    "SI-side contract spec for the fork lane to build "
    "against in aiassistsecure/gaia (request/response shape, "
    "200/409 dedup contract already exercised against the "
    "S-7 double, source=si SSE visibility requirement); the "
    "deferred S-7 quantities F-E-INGEST and F-E-SSE stay "
    "UNMEASURED until a live fork endpoint exists and is "
    "exercised")

__all__ = [
    "FORK_INGEST_CONTRACT_LABEL",
    "HISTORICAL_S5_LABEL_SOURCES",
    "S5_RECONCILIATION_LABEL",
]
