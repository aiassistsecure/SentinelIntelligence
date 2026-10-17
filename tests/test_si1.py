"""SI-1 integration: NEDB is the default journal backend; `si seed` and
`si run` work end to end through the real CLI.
"""
import numpy as np
import pytest

from safetensors.numpy import save as _st_save

from si.bundle import init_bundle, load_bundle
from si.cli import main as cli_main
from si.journal import open_journal
from si.minds import MockMind
from si.run import run_cycles


def _tiny_checkpoint(path):
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(99)
    blob = _st_save({
        "W": rng.standard_normal((8, 8)).astype(np.float32),
        "b": rng.standard_normal((8,)).astype(np.float32),
    })
    (path / "model.safetensors").write_bytes(blob)
    return str(path)


def test_nedb_is_the_default_backend(tmp_path):
    """Dogfood directive: init with no backend flag journals in real NEDB."""
    bundle_dir = init_bundle(str(tmp_path / "default.si"), name="si1-default")
    manifest = load_bundle(bundle_dir)
    assert manifest["journal"]["backend"] == "nedb"

    res = run_cycles(bundle_dir, MockMind(), 1)
    assert res["admitted"] == 1 and res["halted"] is False

    journal = open_journal(bundle_dir, manifest)
    try:
        assert journal.verify_chain() is True
    finally:
        journal.close()


def test_cli_seed_then_run(tmp_path):
    """`si seed` imports a real checkpoint; `si run` learns on top of it."""
    bundle_dir = str(tmp_path / "cli.si")
    assert cli_main(["init", bundle_dir, "--name", "cli-e2e",
                     "--journal-backend", "embedded"]) == 0

    ckpt = _tiny_checkpoint(tmp_path / "ckpt")
    assert cli_main(["seed", bundle_dir, ckpt]) == 0
    manifest = load_bundle(bundle_dir)
    assert manifest["base"]["source"].startswith("seed:")
    assert set(manifest["base"]["tensors"]) == {"W", "b"}

    assert cli_main(["run", bundle_dir, "--cycles", "2",
                     "--mind", "demo"]) == 0
    manifest = load_bundle(bundle_dir)
    assert manifest["head"] == "d0002"
    assert len(manifest["deltas"]) == 2

    journal = open_journal(bundle_dir, manifest)
    try:
        assert journal.verify_chain() is True
        prov = journal.backend.find("provenance",
                                    filter={"relation": "seeded_from"})
        assert len(prov) == 1 and prov[0]["child"] == "base"
    finally:
        journal.close()


def test_cli_seed_rejects_garbage(tmp_path):
    bundle_dir = str(tmp_path / "cli-bad.si")
    assert cli_main(["init", bundle_dir, "--name", "cli-bad",
                     "--journal-backend", "embedded"]) == 0
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"this is not a safetensors file at all")
    assert cli_main(["seed", bundle_dir, str(bad)]) == 1
