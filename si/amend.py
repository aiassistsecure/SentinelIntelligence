"""``si amend-constitution``: operator-only, hash-versioned amendment of
the gate constitution.

OPERATOR-ONLY, by design and by mechanism:

- Autonomous constitution amendment is still impossible. The gate
  (``si/gate.py``) returns ESCALATE for every ``kind="constitution"``
  Intent, unconditionally — this module does not change that, and the
  autonomous loop / ``run_cycle`` has no path to the function below.
- The only caller is the operator-facing CLI subcommand
  ``si amend-constitution``, which requires an explicit ``--operator``
  name and a non-empty ``--reason``. Both are recorded in the journal.

HONEST LIMITS — THE OPERATOR FLAG IS NOT AUTHENTICATION:

The ``--operator`` value is an attribution / authorization *record* in
this local reference runtime: a name the operator typed, journaled so
``si log`` / ``si why``-style audits can answer "who amended the
constitution, and why". It is NOT cryptographic authentication — nothing
here verifies that the person at the keyboard is who the flag claims.
Do not treat it as an identity proof; treat it as a signed-in-ink note
in an append-only journal, in a runtime that trusts its local operator.

What an amendment does:

1. Fail closed first: ``load_bundle`` must verify the bundle (a tampered
   bundle is never amended).
2. The replacement text comes from ``--from-file`` (UTF-8). Empty files
   and files identical to the current constitution (no-op amendments)
   are refused — nothing written, nothing journaled as success.
3. The gate is evaluated on a constitution Intent and ESCALATES, as it
   always does; that escalation report is journaled verbatim as the
   self-edit record (``kind="constitution"``, operator + reason
   attached), followed by a ``constitution_amended`` provenance record
   carrying the hash chain link: old_hash -> new_hash.
4. ``constitution.md`` and ``constitution.sha256`` are rewritten
   (tmp + rename), the manifest's ``constitution.hash`` is updated
   (``policy_hash`` is untouched — policy amendment is out of scope),
   and the journal is checkpointed + re-anchored via the same
   ``_anchor_checkpoint`` mechanism every other journaled operation
   uses, so ``load_bundle`` verifies afterwards.
5. The head, the delta stack, and the frozen base blob NEVER move.
   Only the constitution and its manifest hash change.
6. Post-verify: the bundle is reloaded with ``load_bundle`` and the new
   hash must match; any failure raises loudly instead of returning a
   success summary.

CPU only, stdlib only. This is machine learning infrastructure, not
"custom LLM software."
"""
from __future__ import annotations

import os
import tempfile

from si.bundle import (
    _anchor_checkpoint,
    _write_manifest,
    load_bundle,
    sha256_hex,
    utc_now,
)
from si.gate import Gate, Intent
from si.journal import open_journal


class AmendError(ValueError):
    """An amendment that refuses to proceed. Fail closed: nothing
    half-written, nothing journaled as success."""


def _write_file_atomic(path: str, data: bytes) -> None:
    """Write ``data`` to ``path`` via tmp + rename in the same directory,
    mirroring ``bundle._write_manifest``'s atomicity pattern."""
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path),
                                    prefix=".amend.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def amend_constitution(bundle_dir: str, from_file: str, reason: str,
                       operator: str) -> dict:
    """Amend the bundle's constitution. Operator-only; see module docstring.

    Returns ``{"amended": True, "old_hash", "new_hash", "operator",
    "reason", "head", "edit_id", "seq"}`` — hashes are bare hex (no
    ``sha256:`` prefix); ``head`` is the unchanged head id. Raises
    AmendError (a ValueError) on any refusal — missing/empty operator or
    reason, unreadable/empty/no-op replacement file — and propagates
    TamperError from ``load_bundle`` for a tampered bundle. On any
    refusal, nothing is written and nothing is journaled.
    """
    # -- operator-only gate: both attributions are required -------------
    if operator is None or not str(operator).strip():
        raise AmendError(
            "--operator is required and must be non-empty: constitution "
            "amendment is an operator-only action and the operator name "
            "is journaled as the authorization record")
    if reason is None or not str(reason).strip():
        raise AmendError(
            "--reason is required and must be non-empty: every change "
            "has a cause, and a constitution amendment without a stated "
            "reason is refused")
    operator = str(operator).strip()
    reason = str(reason).strip()

    # -- read + validate the replacement text ---------------------------
    try:
        with open(from_file, "rb") as f:
            new_bytes = f.read()
    except OSError as e:
        raise AmendError(f"cannot read replacement constitution "
                         f"{from_file!r}: {e}")
    try:
        new_text = new_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise AmendError(f"replacement constitution {from_file!r} is not "
                         f"valid UTF-8: {e}")
    if not new_text.strip():
        raise AmendError("replacement constitution is empty; refusing to "
                         "amend the constitution to nothing")

    bundle_dir = os.path.abspath(bundle_dir)

    # -- fail closed: a tampered bundle is never amended ----------------
    manifest = load_bundle(bundle_dir)
    old_hash = manifest["constitution"]["hash"].split(":", 1)[1]
    const_path = os.path.join(bundle_dir, "constitution.md")
    with open(const_path, "rb") as f:
        current_bytes = f.read()
    if new_bytes == current_bytes:
        raise AmendError(
            "replacement constitution is identical to the current one "
            f"(sha256:{old_hash}); a no-op amendment is refused")
    new_hex = sha256_hex(new_bytes)
    new_hash = "sha256:" + new_hex

    head_before = manifest["head"]
    deltas_before = [dict(d) for d in manifest["deltas"]]

    # -- journal first: the gate ESCALATES (as it always does for
    #    constitution intents); the operator's direct CLI action is what
    #    carries the amendment through, and both facts are recorded. ---
    from si.bundle import load_policy  # local: keeps import graph flat
    intent = Intent(kind="constitution", reason=reason)
    report = Gate(load_policy(bundle_dir)).evaluate(intent, None, None)
    assert report.verdict == "ESCALATE", (
        "gate invariant broken: a constitution intent must ESCALATE, "
        f"got {report.verdict!r}; refusing to amend")

    intent_dict = intent.to_dict()
    intent_dict["old_constitution_hash"] = "sha256:" + old_hash
    intent_dict["new_constitution_hash"] = new_hash

    journal = open_journal(bundle_dir, manifest)
    try:
        edit_id = journal.record_self_edit(
            "constitution", intent_dict, reason,
            report.verdict, report.to_dict(), operator=operator)
        journal.record_provenance(
            child=new_hash,
            parents=["sha256:" + old_hash],
            relation="constitution_amended",
            note=(f"sha256:{old_hash} -> {new_hash}; "
                  f"operator={operator}: {reason}")[:200])
        journal.set_edit_result(edit_id, "amended")

        # -- write the new constitution + hash file, update the
        #    manifest hash, then checkpoint + re-anchor so the journal
        #    tip and the manifest agree (the standard mechanism). -----
        _write_file_atomic(const_path, new_bytes)
        _write_file_atomic(os.path.join(bundle_dir, "constitution.sha256"),
                           new_hex.encode("ascii"))
        manifest = dict(manifest)
        manifest["constitution"] = dict(manifest["constitution"])
        manifest["constitution"]["hash"] = new_hash
        # policy_hash deliberately unchanged: policy.json amendment is
        # out of scope for this flow.
        anchored = _anchor_checkpoint(
            journal, manifest, head=head_before,
            note=f"constitution amended by {operator} "
                 f"(sha256:{old_hash[:12]}.. -> sha256:{new_hex[:12]}..)")
        _write_manifest(bundle_dir, anchored)
        seq = anchored["journal"]["last_seq"]
    finally:
        journal.close()

    # -- post-verify: reload and demand the new hash --------------------
    verified = load_bundle(bundle_dir)
    if verified["constitution"]["hash"] != new_hash:
        raise AmendError(
            "post-amend verification failed: manifest constitution hash "
            f"is {verified['constitution']['hash']!r}, expected "
            f"{new_hash!r} — the bundle may be inconsistent; investigate "
            "before running anything")
    if verified["head"] != head_before or verified["deltas"] != deltas_before:
        raise AmendError(
            "post-amend verification failed: head or deltas moved during "
            "a constitution amendment, which must never happen")

    return {
        "amended": True,
        "old_hash": old_hash,
        "new_hash": new_hex,
        "operator": operator,
        "reason": reason,
        "head": verified["head"],
        "edit_id": edit_id,
        "seq": seq,
        "amended_at": utc_now(),
    }


__all__ = ["AmendError", "amend_constitution"]
