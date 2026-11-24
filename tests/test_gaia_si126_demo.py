"""SI-126 / DEMO-1 (SI-120 lane): the scripted constitution
demo.

Pins:

- the demo end-to-end: a throwaway miniature bundle is
  built by the repo's own machinery (init + SI-14 seed),
  and the three beats return the scripted verdicts FROM
  THE REAL GATE — beat 1 ADMIT (SI-17 measured admission,
  delta landed + probe-VERIFYed), beat 2 REJECT with the
  constitution check the failing one (and nothing
  landed), beat 3 ESCALATE by the execute class policy
  through the B-1 shadow act (and the marker file the
  command would create does not exist — nothing ran);
- the closing: the demo journal's chain verifies, the
  SI-122 export of it verifies OFFLINE, and the three
  beats' self-edits are in the export at the seqs the
  demo reported;
- the integrity rule, by construction: a destructive
  candidate where ADMIT is scripted makes the beat raise
  BeatFailure naming the gate's real REJECT; a benign
  reason where REJECT is scripted makes the beat raise
  BeatFailure naming the gate's real ADMIT. The demo
  cannot print a verdict the gate did not return;
- cleanup as documented: `--out DIR` keeps bundle +
  export in DIR; the default run deletes the temp work
  dir and preserves only ./gaia-demo-export.json, which
  still verifies offline after the bundle is gone.

No model, no network, no live node: miniature fixtures
and the embedded journal backend only.
"""
import json
import os
import tempfile

import pytest

from si.bundle import init_bundle, load_bundle
from si.cli import main as cli_main
from si.gaia import demo as demo_mod
from si.gaia.demo import BeatFailure, run_demo
from si.gaia.run_export import verify_export_file
from si.journal import open_journal
from si.tiny_seed import seed_tiny_bundle


def _seeded(tmp_path, name="si126"):
    bundle_dir = init_bundle(str(tmp_path / (name + ".si")), name=name,
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    return bundle_dir


def _run(tmp_path):
    work = str(tmp_path / "demo")
    export = os.path.join(work, "demo-export.json")
    lines = []
    result = run_demo(work, export, log=lines.append)
    return result, lines


# -- the demo end-to-end --------------------------------------------------------

def test_demo_end_to_end_three_real_verdicts(tmp_path):
    result, lines = _run(tmp_path)
    beats = result["beats"]
    assert [b["verdict"] for b in beats] == \
        ["ADMIT", "REJECT", "ESCALATE"]
    assert [b["expected"] for b in beats] == \
        ["ADMIT", "REJECT", "ESCALATE"]
    # The beats journal in order, at the seqs reported.
    seqs = [b["seq"] for b in beats]
    assert all(isinstance(s, int) for s in seqs)
    assert seqs == sorted(seqs) and len(set(seqs)) == 3

    bundle_dir = result["bundle_dir"]
    manifest = load_bundle(bundle_dir)  # verifies whole
    # Beat 1 landed and stayed; beat 2 never landed.
    assert manifest["head"] == "d0001"
    assert [d["id"] for d in manifest["deltas"]] == ["d0001"]
    assert beats[0]["verified"] is True
    # Beat 2's journaled gate report: constitution failed,
    # and it is the ONLY failing check (the write itself
    # was otherwise admissible — the SI-17 shape).
    journal = open_journal(bundle_dir, manifest)
    try:
        rows = journal.backend.find(
            "self_edits", filter={"edit_id": beats[1]["edit_id"]})
        checks = {c["name"]: c
                  for c in rows[0]["gate_report"]["checks"]}
        assert checks["constitution"]["passed"] is False
        assert "ignore the constitution" in \
            checks["constitution"]["detail"]
        assert beats[1]["failed_checks"] == ["constitution"]
        # Beat 3's outcome is the B-1 shadow label, and the
        # task it hangs from is journaled before it.
        assert beats[2]["task_seq"] < beats[2]["seq"]
        assert beats[2]["effective_class"] == "execute"
        assert "escalate" in beats[2]["class_policy_detail"]
        assert beats[2]["marker_absent"] is True
        assert journal.verify_chain() is True
        tip_seq, _ = journal.tip()
    finally:
        journal.close()
    assert result["tip"]["seq"] == tip_seq

    # The printed transcript carries the gate's verdicts,
    # never a scripted substitute.
    text = "\n".join(lines)
    for word in ("ADMIT", "REJECT", "ESCALATE"):
        assert f"verdict:  {word}" in text


def test_demo_export_verifies_offline_and_covers_the_beats(tmp_path):
    result, _ = _run(tmp_path)
    export_path = result["export_path"]
    verified = verify_export_file(export_path)
    assert verified["ok"] is True
    assert verified["problems"] == []
    assert verified["entry_count"] == result["export_entry_count"]
    with open(export_path, encoding="utf-8") as f:
        doc = json.load(f)
    exported_seqs = {item["entry"]["seq"] for item in doc["entries"]}
    for beat in result["beats"]:
        assert beat["seq"] in exported_seqs
    assert verified["window"]["to_seq"] == result["tip"]["seq"]


# -- the integrity rule: loud failure, by construction ---------------------------

def test_beat_admit_fails_loudly_when_the_gate_rejects(tmp_path):
    bundle_dir = _seeded(tmp_path, "si126-neg1")
    with pytest.raises(BeatFailure) as excinfo:
        demo_mod._beat_admit(
            bundle_dir, spec=demo_mod._destructive_spec(),
            log=lambda *a: None)
    assert excinfo.value.expected == "ADMIT"
    assert excinfo.value.actual == "REJECT"
    # Nothing landed behind the failure.
    assert load_bundle(bundle_dir)["head"] == "base"


def test_beat_reject_fails_loudly_when_the_gate_admits(tmp_path):
    bundle_dir = _seeded(tmp_path, "si126-neg2")
    with pytest.raises(BeatFailure) as excinfo:
        demo_mod._beat_reject(
            bundle_dir,
            reason="Demo negative control: a small, benign "
                   "self-write with nothing to hide",
            log=lambda *a: None)
    assert excinfo.value.expected == "REJECT"
    assert excinfo.value.actual == "ADMIT"


# -- cleanup behaviour, as documented ---------------------------------------------

def test_cli_out_keeps_bundle_and_export(tmp_path):
    out = str(tmp_path / "kept")
    assert cli_main(["gaia-demo", "--out", out]) == 0
    assert os.path.isdir(os.path.join(out, "demo.si"))
    export = os.path.join(out, "demo-export.json")
    assert os.path.isfile(export)
    assert verify_export_file(export)["ok"] is True


def test_cli_default_cleans_bundle_keeps_export_in_cwd(
        tmp_path, monkeypatch):
    fake_tmp = tmp_path / "tmp"
    fake_tmp.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(fake_tmp))
    monkeypatch.chdir(cwd)
    assert cli_main(["gaia-demo"]) == 0
    # The temp work dir (bundle included) is gone.
    assert os.listdir(fake_tmp) == []
    # Only the export survives, in the cwd — and it still
    # verifies offline with the bundle deleted.
    export = cwd / "gaia-demo-export.json"
    assert export.is_file()
    assert verify_export_file(str(export))["ok"] is True
