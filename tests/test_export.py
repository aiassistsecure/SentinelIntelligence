"""``si export``: materialize head weights to a portable GGUF file.

Covers: F16 roundtrip (tensor names/shapes read back from GGUF match
``materialize_head()`` output, on-disk dtype f16 vs materialized float32),
export determinism, the quant path (honest upstream error propagation when
llama-quantize exists -- the demo arch is genuinely unquantizable, so a
real failure must surface, never a faked success; clear hearth/binary
error otherwise), provenance journaling on the nedb backend (dogfood),
head immobility (export journals provenance + checkpoint only), base-only
export, and the CLI.
"""
import json
import os

import gguf
import numpy as np
import pytest

from si.bundle import init_bundle, load_bundle
from si.export import export_bundle, materialize_head
from si.journal import open_journal

#: hearth's quantize binary; its presence selects the quant smoke test.
LLAMA_QUANTIZE = os.path.expanduser("~/.hearth/runtime/bin/llama-quantize")


def _init(tmp_path, name, backend="embedded"):
    return init_bundle(str(tmp_path / (name + ".si")), name=name,
                       journal_backend=backend)


def _demo_cycles(bundle_dir, n):
    """Run n real DemoMind cycles; every one must ADMIT."""
    from si.loop import run_cycle
    from si.minds import DemoMind
    for _ in range(n):
        res = run_cycle(bundle_dir, DemoMind(bundle_path=bundle_dir))
        assert res["verdict"] == "ADMIT"
    return load_bundle(bundle_dir)


def _gguf_tensors(path):
    return {t.name: t for t in gguf.GGUFReader(path).tensors}


def _bare(ref):
    """Strip scheme prefixes (e.g. 'gguf:sha256:' or 'sha256:'), if present."""
    return str(ref).rsplit(":", 1)[-1]


def _require_keys(res):
    assert {"head", "head_hash", "out_path", "bytes", "sha256", "quant",
            "journal_seq"} <= set(res), f"missing return keys: {sorted(set(res))}"


# -- roundtrip ---------------------------------------------------------------

def test_export_roundtrip(tmp_path):
    bundle_dir = _init(tmp_path, "export-rt")
    manifest = _demo_cycles(bundle_dir, 2)
    assert manifest["head"] == "d0002"

    expected = materialize_head(bundle_dir)
    assert expected, "expected materialized tensors for a 2-cycle head"
    for name, arr in expected.items():
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32, f"{name}: materialized dtype must be float32"

    out = str(tmp_path / "head.gguf")
    res = export_bundle(bundle_dir, out)
    _require_keys(res)
    assert res["head"] == "d0002"
    assert res["quant"] is None
    assert res["out_path"] == os.path.abspath(out)
    assert res["bytes"] == os.path.getsize(out) > 0
    assert _bare(res["sha256"]) and len(_bare(res["sha256"])) == 64

    on_disk = _gguf_tensors(out)
    assert set(on_disk) == set(expected), \
        f"GGUF tensor names {sorted(on_disk)} != materialized {sorted(expected)}"
    for name, t in on_disk.items():
        assert tuple(int(x) for x in t.shape) == tuple(expected[name].shape), \
            f"{name}: shape mismatch"
        assert t.tensor_type.name == "F16", \
            f"{name}: on-disk dtype {t.tensor_type.name} != F16"


# -- determinism --------------------------------------------------------------

def test_export_determinism(tmp_path):
    bundle_dir = _init(tmp_path, "export-det")
    _demo_cycles(bundle_dir, 2)

    a = str(tmp_path / "a.gguf")
    b = str(tmp_path / "b.gguf")
    ra = export_bundle(bundle_dir, a)
    rb = export_bundle(bundle_dir, b)
    _require_keys(ra)
    _require_keys(rb)

    from si.bundle import sha256_hex
    with open(a, "rb") as f:
        ha = sha256_hex(f.read())
    with open(b, "rb") as f:
        hb = sha256_hex(f.read())
    assert ha == hb, "two exports of the same head must be byte-identical"
    assert _bare(ra["sha256"]) == _bare(rb["sha256"]) == ha


# -- quant --------------------------------------------------------------------

def test_export_quant(tmp_path):
    bundle_dir = _init(tmp_path, "export-q")
    _demo_cycles(bundle_dir, 1)

    if os.path.isfile(LLAMA_QUANTIZE) and os.access(LLAMA_QUANTIZE, os.X_OK):
        # Binary present: prove we really shell out and never fake a quant.
        # The demo export declares general.architecture="unknown" (honest:
        # it is a demo mind, not a transformer), and upstream llama-quantize
        # refuses unknown architectures. That genuine upstream error must
        # propagate — a fake "success" here would be a lie.
        import subprocess
        q = str(tmp_path / "q40.gguf")
        with pytest.raises(subprocess.CalledProcessError):
            export_bundle(bundle_dir, q, quant="Q4_0")
        assert not os.path.isfile(q), \
            "a failed quant must not leave a fake quantized file behind"
    else:
        # Binary missing: a clear error, never a faked quant.
        out = str(tmp_path / "q40.gguf")
        with pytest.raises(Exception,
                           match=r"(?i)hearth|llama-quantize") as ei:
            export_bundle(bundle_dir, out, quant="Q4_0")
        assert str(ei.value).strip(), "quant error must carry a message"


# -- provenance on the dogfood backend -----------------------------------------

@pytest.mark.slow
def test_export_journaled_nedb(tmp_path):
    """Dogfood: one real cycle + export against the real NEDB journal."""
    bundle_dir = _init(tmp_path, "export-nedb", backend="nedb")
    manifest = _demo_cycles(bundle_dir, 1)
    head = manifest["head"]
    assert head == "d0001"

    out = str(tmp_path / "nedb.gguf")
    res = export_bundle(bundle_dir, out)
    _require_keys(res)
    assert res["head"] == "d0001"
    assert isinstance(res["journal_seq"], int) and res["journal_seq"] > 0

    manifest = load_bundle(bundle_dir)  # still verifies cleanly (re-anchor)
    assert manifest["head"] == "d0001"
    journal = open_journal(bundle_dir, manifest)
    try:
        assert journal.verify_chain() is True
        rows = journal.backend.find("provenance", filter={"relation": "export"},
                                    sort="-seq")
        assert rows, "expected an export provenance entry"
        row = rows[0]
        assert _bare(row["child"]) == _bare(res["sha256"]), \
            "export provenance child must carry the output sha"
        assert row["parents"] == [head]
        note = json.loads(row["note"])
        assert note["quant"] is None
        assert note["bytes"] > 0
        assert note["exported_at"], "exported_at timestamp missing"
    finally:
        journal.close()


# -- export moves nothing -------------------------------------------------------

def test_export_moves_nothing(tmp_path):
    bundle_dir = _init(tmp_path, "export-still")
    before = _demo_cycles(bundle_dir, 2)
    head_before = before["head"]
    delta_ids_before = [d["id"] for d in before["deltas"]]

    journal = open_journal(bundle_dir, before)
    try:
        n_self_edits = journal.backend.count("self_edits")
        n_deltas = journal.backend.count("deltas")
        n_prov = journal.backend.count("provenance")
        n_ckpt = journal.backend.count("checkpoints")
    finally:
        journal.close()

    res = export_bundle(bundle_dir, str(tmp_path / "still.gguf"))
    assert res["head"] == head_before

    after = load_bundle(bundle_dir)  # verifies cleanly: re-anchor worked
    assert after["head"] == head_before, "export must not move the head"
    assert [d["id"] for d in after["deltas"]] == delta_ids_before, \
        "export must not create deltas"

    journal = open_journal(bundle_dir, after)
    try:
        assert journal.backend.count("self_edits") == n_self_edits, \
            "export must not journal self-edits (never gates)"
        assert journal.backend.count("deltas") == n_deltas, \
            "export must not add delta rows"
        # Only provenance + checkpoint are appended.
        assert journal.backend.count("provenance") == n_prov + 1
        assert journal.backend.count("checkpoints") == n_ckpt + 1
    finally:
        journal.close()


# -- base only -------------------------------------------------------------------

def test_export_base_only(tmp_path):
    """A fresh bundle (zero cycles) exports its frozen base."""
    bundle_dir = _init(tmp_path, "export-base")
    expected = materialize_head(bundle_dir)
    assert expected, "base must materialize to the demo seed tensors"

    out = str(tmp_path / "base.gguf")
    res = export_bundle(bundle_dir, out)
    _require_keys(res)
    assert res["head"] == "base"

    on_disk = _gguf_tensors(out)
    assert set(on_disk) == set(expected)
    for name, t in on_disk.items():
        assert tuple(int(x) for x in t.shape) == tuple(expected[name].shape)
        assert t.tensor_type.name == "F16"

    manifest = load_bundle(bundle_dir)
    assert manifest["head"] == "base"


# -- CLI ---------------------------------------------------------------------------

def test_export_cli(tmp_path):
    from si.cli import main

    bundle_dir = _init(tmp_path, "export-cli")
    _demo_cycles(bundle_dir, 1)
    out = str(tmp_path / "cli.gguf")

    rc = main(["export", bundle_dir, "--out", out])
    assert rc == 0, "si export should exit 0"
    assert os.path.isfile(out), "CLI must write the requested --out file"
    assert _gguf_tensors(out), "CLI output must be a readable GGUF"


def test_export_cli_quant_without_binary_fails(tmp_path):
    from si.cli import main

    if os.path.isfile(LLAMA_QUANTIZE) and os.access(LLAMA_QUANTIZE, os.X_OK):
        pytest.skip("llama-quantize present: no missing-binary path to test")
    bundle_dir = _init(tmp_path, "export-cli-q")
    _demo_cycles(bundle_dir, 1)
    rc = main(["export", bundle_dir, "--out", str(tmp_path / "cli-q.gguf"),
               "--quant", "Q4_0"])
    assert rc != 0, "si export --quant must fail loudly without the quant binary"
