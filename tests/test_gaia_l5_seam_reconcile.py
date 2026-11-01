"""SI-71 / L-5 (SI-69 §4.7): seam label reconciliation.

Pins ONLY the new constants added by SI-71
(:mod:`si.gaia.seam_reconcile`). No historical label string is
pinned, edited, or re-meant here: the justification is in
``docs/SI-71-l5-seam-label-reconciliation.md`` — a new test
may pin a new constant, never rewrite history's meaning.
Existing tests pinning the historical labels pass unmodified
alongside this file.
"""
import os

from si.bundle import init_bundle
from si.gaia.seam_reconcile import (
    FORK_INGEST_CONTRACT_LABEL,
    HISTORICAL_S5_LABEL_SOURCES,
    S5_RECONCILIATION_LABEL,
)
from si.gaia.service import ServiceState
from si.tiny_seed import seed_tiny_bundle


def test_s5_reconciliation_label_states_the_si_side_truth():
    label = S5_RECONCILIATION_LABEL
    # Poller and reconciler BUILT, with their slice identity.
    assert "BUILT" in label
    assert "S-5/SI-59" in label
    assert "si.gaia.seam_outcome" in label
    assert "replay/stub" in label
    assert "loopback" in label
    # Live fork interop was never exercised.
    assert "live fork interop is UNMEASURED" in label
    # Direction honesty: S-5 is fork-to-SI only.
    assert "no outcome returns to the fork" in label
    # History is supplemented, never rewritten.
    assert "not rewritten" in label
    assert "no already-journaled record changes" in label


def test_historical_label_sources_name_every_stale_constant():
    assert HISTORICAL_S5_LABEL_SOURCES == (
        "si.gate.TOOL_SEAM_PENDING_EXECUTION",
        "si.gaia.seam.REGISTRY_GAP_LABEL",
        "si.gaia.seam_submit.SUBMITTED_STATE",
        "si.gaia.seam_submit.WORK_SUBMISSION_LABEL",
        "si.gaia.seam_leash.LEASH_CROSSING_LABEL",
    )


def test_fork_ingest_contract_label_keeps_fork_half_unbuilt():
    label = FORK_INGEST_CONTRACT_LABEL
    assert "remains UNBUILT" in label
    assert "builds no fork code" in label
    assert "claims no fork behaviour" in label
    assert "200/409 dedup contract" in label
    assert "source=si" in label
    assert "F-E-INGEST" in label and "F-E-SSE" in label
    assert ("UNMEASURED until a live fork endpoint exists "
            "and is exercised") in label


def test_service_description_carries_reconciliation_label(tmp_path):
    bundle_dir = init_bundle(
        os.path.join(str(tmp_path), "l5.si"), name="gaia-l5",
        journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    description = ServiceState(bundle_dir).description()
    assert (description["seam_label_reconciliation"]
            == S5_RECONCILIATION_LABEL)
