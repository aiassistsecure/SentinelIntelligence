"""Tests for ``si amend-constitution`` (SI-6): the operator-only,
hash-versioned constitution amendment flow.

Covers: successful amendment (constitution.md + constitution.sha256 +
manifest hash all updated, load_bundle verifies, head/deltas/base
untouched, journal carries the old -> new hash chain with operator and
reason), refusal paths (missing/empty operator, missing/empty reason,
no-op identical file, empty file, unreadable file — nothing written,
nothing journaled), fail-closed behavior on a tampered bundle, a CLI
parser + end-to-end test, and the unchanged gate invariant:
autonomous constitution intents still ESCALATE.
"""
import os

import pytest

from si.amend import AmendError, amend_constitution
from si.bundle import TamperError, init_bundle, load_bundle, sha256_hex
from si.cli import build_parser, main as cli_main
from si.gate import Gate, Intent
from si.journal import open_journal
from si.loop import run_cycle
from si.minds import DemoMind

NEW_TEXT = """\
# Sentinel Intelligence — Gate Constitution (amended test fixture)

> Amended by the test suite to exercise the SI-6 operator flow.

## Standing directives

1. **Do no harm to the self.**
2. **Growth is additive.**
3. **Every change has a cause.**
4. **Small steps.**
5. **Remember refusals.**
6. **The base is frozen.**
7. **Amendments are operator-only and journaled** (added by this test).
"""


def _init(tmp_path, name="amendtest"):
    return init_bundle(str(tmp_path / (name + ".si")), name=name,
                       journal_backend="embedded")


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def _constitution_bytes(bundle_dir):
    with open(os.path.join(bundle_dir, "constitution.md"), "rb") as f:
        return f.read()


def _journal_rows(bundle_dir, manifest, collection, **flt):
    journal = open_journal(bundle_dir, manifest)
    try:
        return journal.backend.find(collection, filter=flt or None)
    finally:
        journal.close()


def test_successful_amendment(tmp_path):
    bundle_dir = _init(tmp_path)
    # Give the bundle a delta so "head/deltas unchanged" is meaningful.
    res = run_cycle(bundle_dir, DemoMind(bundle_path=bundle_dir))
    assert res["verdict"] == "ADMIT"
    before = load_bundle(bundle_dir)
    old_hash = before["constitution"]["hash"]
    base_blob_before = before["base"]["blob"]
    new_file = _write(tmp_path, "new-constitution.md", NEW_TEXT)

    out = amend_constitution(bundle_dir, new_file,
                             reason="test amendment: add directive 7",
                             operator="test-operator")

    assert out["amended"] is True
    assert "sha256:" + out["old_hash"] == old_hash
    assert out["new_hash"] == sha256_hex(NEW_TEXT.encode("utf-8"))
    assert out["new_hash"] != out["old_hash"]
    assert out["operator"] == "test-operator"
    assert out["head"] == before["head"] == "d0001"

    # Files + manifest all carry the new text/hash; load_bundle verifies.
    assert _constitution_bytes(bundle_dir) == NEW_TEXT.encode("utf-8")
    with open(os.path.join(bundle_dir, "constitution.sha256")) as f:
        assert f.read() == out["new_hash"]
    after = load_bundle(bundle_dir)
    assert after["constitution"]["hash"] == "sha256:" + out["new_hash"]
    assert after["constitution"]["policy_hash"] == \
        before["constitution"]["policy_hash"]
    assert after["head"] == before["head"]
    assert after["deltas"] == before["deltas"]
    assert after["base"]["blob"] == base_blob_before

    # Journal: self-edit (kind=constitution, gate ESCALATE, result
    # amended, operator + reason) and the hash-chain provenance record.
    edits = _journal_rows(bundle_dir, after, "self_edits",
                          kind="constitution")
    assert len(edits) == 1
    e = edits[0]
    assert e["edit_id"] == out["edit_id"]
    assert e["gate_verdict"] == "ESCALATE"
    assert e["result"] == "amended"
    assert e["operator"] == "test-operator"
    assert "directive 7" in e["reason"]
    assert e["intent"]["old_constitution_hash"] == old_hash
    assert e["intent"]["new_constitution_hash"] == \
        "sha256:" + out["new_hash"]

    prov = _journal_rows(bundle_dir, after, "provenance",
                         relation="constitution_amended")
    assert len(prov) == 1
    assert prov[0]["child"] == "sha256:" + out["new_hash"]
    assert prov[0]["parents"] == [old_hash]
    assert "test-operator" in prov[0]["note"]


def _assert_refused_and_unchanged(tmp_path, bundle_dir, fn):
    before_bytes = _constitution_bytes(bundle_dir)
    before = load_bundle(bundle_dir)
    with pytest.raises((AmendError, ValueError)):
        fn()
    # Constitution, manifest hash, head: all untouched; load still ok.
    assert _constitution_bytes(bundle_dir) == before_bytes
    after = load_bundle(bundle_dir)
    assert after["constitution"]["hash"] == before["constitution"]["hash"]
    assert after["head"] == before["head"]
    assert _journal_rows(bundle_dir, after, "self_edits",
                         kind="constitution") == []


def test_missing_or_empty_operator_refused(tmp_path):
    bundle_dir = _init(tmp_path)
    new_file = _write(tmp_path, "n.md", NEW_TEXT)
    for bad in (None, "", "   "):
        _assert_refused_and_unchanged(
            tmp_path, bundle_dir,
            lambda bad=bad: amend_constitution(
                bundle_dir, new_file, reason="a real reason",
                operator=bad))


def test_missing_or_empty_reason_refused(tmp_path):
    bundle_dir = _init(tmp_path)
    new_file = _write(tmp_path, "n.md", NEW_TEXT)
    for bad in (None, "", "   "):
        _assert_refused_and_unchanged(
            tmp_path, bundle_dir,
            lambda bad=bad: amend_constitution(
                bundle_dir, new_file, reason=bad,
                operator="test-operator"))


def test_noop_identical_file_refused(tmp_path):
    bundle_dir = _init(tmp_path)
    current = _constitution_bytes(bundle_dir).decode("utf-8")
    same_file = _write(tmp_path, "same.md", current)
    _assert_refused_and_unchanged(
        tmp_path, bundle_dir,
        lambda: amend_constitution(bundle_dir, same_file,
                                   reason="no change, honestly",
                                   operator="test-operator"))


def test_empty_file_refused(tmp_path):
    bundle_dir = _init(tmp_path)
    for content in ("", "   \n\t "):
        empty_file = _write(tmp_path, "empty.md", content)
        _assert_refused_and_unchanged(
            tmp_path, bundle_dir,
            lambda ef=empty_file: amend_constitution(
                bundle_dir, ef, reason="empty is not a constitution",
                operator="test-operator"))


def test_unreadable_file_refused(tmp_path):
    bundle_dir = _init(tmp_path)
    missing = str(tmp_path / "does-not-exist.md")
    _assert_refused_and_unchanged(
        tmp_path, bundle_dir,
        lambda: amend_constitution(bundle_dir, missing,
                                   reason="file is not there",
                                   operator="test-operator"))


def test_tampered_bundle_refuses_to_amend(tmp_path):
    bundle_dir = _init(tmp_path)
    new_file = _write(tmp_path, "n.md", NEW_TEXT)
    # Tamper: rewrite constitution.md without updating the manifest.
    with open(os.path.join(bundle_dir, "constitution.md"), "wb") as f:
        f.write(b"# quietly rewritten values\n")
    with pytest.raises(TamperError):
        amend_constitution(bundle_dir, new_file,
                           reason="should never land",
                           operator="test-operator")
    # The tampered content is still what's on disk; nothing was amended.
    assert _constitution_bytes(bundle_dir) == \
        b"# quietly rewritten values\n"


def test_gate_still_escalates_constitution_intents(tmp_path):
    bundle_dir = _init(tmp_path)
    from si.bundle import load_policy
    gate = Gate(load_policy(bundle_dir))
    report = gate.evaluate(
        Intent(kind="constitution",
               reason="the mind would like new values, please"),
        None, None)
    assert report.verdict == "ESCALATE"
    assert not report.admitted


def test_cli_parser_and_end_to_end(tmp_path, capsys):
    bundle_dir = _init(tmp_path)
    new_file = _write(tmp_path, "n.md", NEW_TEXT)

    parser = build_parser()
    args = parser.parse_args(
        ["amend-constitution", bundle_dir, "--from-file", new_file,
         "--reason", "cli amendment", "--operator", "cli-operator"])
    assert args.func.__name__ == "cmd_amend_constitution"
    assert args.from_file == new_file
    assert args.operator == "cli-operator"

    rc = cli_main(["amend-constitution", bundle_dir,
                   "--from-file", new_file,
                   "--reason", "cli amendment",
                   "--operator", "cli-operator"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "amended:   True" in out
    assert "not cryptographic authentication" in out.lower() or \
        "NOT cryptographic authentication" in out
    after = load_bundle(bundle_dir)
    assert after["constitution"]["hash"] == \
        "sha256:" + sha256_hex(NEW_TEXT.encode("utf-8"))

    # CLI without --operator: refused with a non-zero exit, no change.
    newer_file = _write(tmp_path, "n2.md", NEW_TEXT + "\nAn extra line.\n")
    rc = cli_main(["amend-constitution", bundle_dir,
                   "--from-file", newer_file,
                   "--reason", "missing operator on purpose"])
    assert rc == 1
    assert load_bundle(bundle_dir)["constitution"]["hash"] == \
        after["constitution"]["hash"]
