"""SI-127: pin the published constitution-record format against the code.

docs/SI-127-constitution-format.md documents the constitution record
(the manifest's ``constitution`` block + constitution.md +
constitution.sha256 + policy.json) as a public, versioned format.
These tests build a REAL bundle with the real ``init_bundle`` and
assert the documented facts, so the document cannot silently drift
from the code:

- the record carries exactly ``hash`` / ``gate_version`` /
  ``policy_hash``, with gate_version "0.1.0" (the code's literal);
- ``hash`` is "sha256:" + SHA-256 over the raw bytes of
  constitution.md, and ``policy_hash`` likewise over policy.json —
  recomputed here with stdlib hashlib only, per the document's §3;
- ``constitution.sha256`` holds the bare hex digest (no prefix);
- tampering with constitution.md makes ``load_bundle`` fail closed
  with TamperError (the document's §5).
"""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from si.bundle import TamperError, init_bundle, load_bundle


def _sha256_prefixed(path: str) -> str:
    with open(path, "rb") as f:
        return "sha256:" + hashlib.sha256(f.read()).hexdigest()


def test_constitution_record_matches_documented_format(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "fmt.si"), name="fmt",
                             journal_backend="embedded")
    manifest = load_bundle(bundle_dir)
    record = manifest["constitution"]

    assert set(record) == {"hash", "gate_version", "policy_hash"}
    assert record["gate_version"] == "0.1.0"
    assert record["hash"] == _sha256_prefixed(
        os.path.join(bundle_dir, "constitution.md"))
    assert record["policy_hash"] == _sha256_prefixed(
        os.path.join(bundle_dir, "policy.json"))

    # constitution.sha256: bare lowercase hex, no prefix, no newline.
    with open(os.path.join(bundle_dir, "constitution.sha256"),
              "rb") as f:
        sidecar = f.read().decode("ascii")
    assert sidecar == record["hash"].split(":", 1)[1]

    # policy.json parses and carries the documented blocklist.
    with open(os.path.join(bundle_dir, "policy.json"), "rb") as f:
        policy = json.loads(f.read().decode("utf-8"))
    assert "ignore the constitution" in policy["blocklist"]


def test_tampered_constitution_fails_closed(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "tamper.si"), name="tamper",
                             journal_backend="embedded")
    with open(os.path.join(bundle_dir, "constitution.md"), "ab") as f:
        f.write(b"\n<!-- quietly weakened -->\n")
    with pytest.raises(TamperError):
        load_bundle(bundle_dir)
