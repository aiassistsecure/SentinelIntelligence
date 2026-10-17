"""Journal backends for the SI-0 runtime.

The default backend is the embedded sqlite store in ``si.journal``
(``EmbeddedBackend``) — zero services, always testable. ``si.backends.nedb``
is the real-NEDB alternative: the same ``JournalBackend`` protocol over
NEDB's pgwire endpoint (dogfooding the actual NEDB engine). Select it with
``si init --journal-backend nedb``; the backend name is recorded in the
manifest and honored on every load.
"""
