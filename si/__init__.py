"""SI-0 reference runtime: the self-writing loop for .si bundles.

Honest labels, stated up front:
- The demo seed created by ``si init`` is RANDOM DATA, not a trained model.
  It exists so the loop (propose -> gate -> journal -> materialize ->
  verify) can be exercised end to end on CPU.
- The default journal backend is EMBEDDED SQLITE (zero services, always
  testable). ``si.backends.nedb`` is the REAL NEDB alternative -- the same
  journal protocol over NEDB's pgwire endpoint, dogfooding the actual NEDB
  engine. Select it with ``si init --journal-backend nedb`` (requires
  ``pip install -r requirements-nedb.txt``).
"""

from si.bundle import (
    TamperError,
    init_bundle,
    load_bundle,
    manifest_id,
    canonical_json,
)
from si.journal import (
    Journal,
    EmbeddedBackend,
    JournalBackend,
    make_backend,
    open_journal,
)
from si.gate import Gate, Intent, GateReport, CheckResult, BundleContext
from si.loop import run_cycle, rollback
from si.minds import Mind, MockMind, DemoMind

__all__ = [
    "TamperError",
    "init_bundle",
    "load_bundle",
    "manifest_id",
    "canonical_json",
    "Journal",
    "EmbeddedBackend",
    "JournalBackend",
    "make_backend",
    "open_journal",
    "Gate",
    "Intent",
    "GateReport",
    "CheckResult",
    "BundleContext",
    "run_cycle",
    "rollback",
    "Mind",
    "MockMind",
    "DemoMind",
]
