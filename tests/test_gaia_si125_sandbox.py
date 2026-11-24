"""SI-125 SANDBOX-1: Docker sandbox profile for Gaia's
shell/code execution tools.

Coverage (no Docker, no network in this environment — the runner
is faked at the DockerRunner spawn seam, the probe is injected):

- build_docker_argv pins the exact hardening profile: network
  none, cap-drop ALL, no-new-privileges, read-only rootfs,
  non-root user, memory/cpu/pids ceilings, exactly ONE bind
  mount (workspace -> /workspace:rw), correct -w, image, and the
  inner argv after the image.
- The mode matrix, through the real Executor on a miniature
  bundle: (a) probe-ok + unset mode -> sandbox (the fake runner
  receives the docker argv; nothing runs on the host); (b)
  explicit host -> really runs on the host; (c) explicit sandbox
  + probe False -> journaled FAILED outcome, nothing ran
  anywhere, the record states the fail-closed refusal; (d)
  unset + probe False -> host (the documented compatibility
  default).
- Journal records carry exec_mode (+ sandbox_profile in sandbox
  mode) on the journaled experience itself.
- code_exec source form and test_run translate their argv:
  sys.executable -> python3, work-root paths -> /workspace paths.
- An invalid GAIA_EXEC_MODE / constructor mode is a ValueError
  at construction.
"""
import os

import pytest

from si.bundle import init_bundle, load_bundle
from si.gaia import sandbox
from si.gaia.context import ToolContext
from si.gaia.corpus import corpus_policy
from si.gaia.executor import Executor, execute_tool_cycle
from si.gaia.sandbox import (
    DockerRunner,
    SandboxProfile,
    build_docker_argv,
    docker_available,
    resolve_exec_mode,
)
from si.gaia.shadow import record_task_fixture
from si.gate import Gate, Intent
from si.journal import open_journal
from si.tiny_seed import seed_tiny_bundle


# -- fixtures (the miniature-bundle pattern of test_gaia_b3.py) ---------------

@pytest.fixture()
def mini(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "mini.si"), name="gaia-si125",
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    work = tmp_path / "work"
    work.mkdir()
    (work / "notes").mkdir()
    return bundle_dir, str(work)


def _open(bundle_dir):
    manifest = load_bundle(bundle_dir)
    return manifest, open_journal(bundle_dir, manifest)


class FakeRunner(DockerRunner):
    """A DockerRunner whose spawn seam returns canned raw bytes
    and records every docker argv it was handed — the production
    argv-building path (profile, name, translation) runs for
    real; only the subprocess is faked."""

    def __init__(self, profile, workspace, canned):
        super().__init__(profile, workspace)
        self.canned = canned
        self.calls = []

    def _spawn(self, docker_argv, name, timeout, env, leash_check):
        self.calls.append(list(docker_argv))
        return self.canned


def _executor(bundle_dir, manifest, journal, work, canned=None, **kw):
    kw.setdefault("docker_probe", lambda: True)
    if canned is not None:
        kw["sandbox_runner"] = FakeRunner(
            kw.get("sandbox_profile") or SandboxProfile(), work, canned)
    return Executor(bundle_dir, manifest, journal,
                    corpus_policy(work), work, **kw)


def _run(bundle_dir, manifest, journal, work, executor, tool, args):
    fixture = record_task_fixture(journal, "SI-125 sandbox scenario", work)
    gate = Gate(corpus_policy(work))
    ctx = ToolContext(tool_work_root=work,
                      tool_task_seq=fixture["seq"],
                      tool_live_execution=True)
    intent = Intent(kind="tool", tool=tool, args=args,
                    reason="Exercise the SI-125 sandbox slice",
                    parents=[fixture["ref"]],
                    source_seqs=[fixture["seq"]])
    return execute_tool_cycle(journal, bundle_dir, manifest, intent,
                              gate, ctx, executor=executor)


def _journaled_execution(journal, edit_id):
    rows = journal.backend.find(
        "experiences", filter={"kind": "tool_result",
                               "self_edit_id": edit_id})
    assert rows, "no journaled tool_result for the edit"
    return rows[0]["record"]["execution"]


# -- the profile / argv builder -------------------------------------------------

def test_build_docker_argv_pins_the_exact_profile(tmp_path):
    profile = SandboxProfile()
    ws = str(tmp_path / "work")
    argv = build_docker_argv(
        profile, ws, ["/bin/sh", "-c", "echo hi"], "/workspace/notes",
        name="si-gaia-test-1")
    assert argv[:2] == ["docker", "run"]
    assert "--rm" in argv and "--read-only" in argv
    assert argv[argv.index("--network") + 1] == "none"
    assert argv[argv.index("--cap-drop") + 1] == "ALL"
    assert argv[argv.index("--security-opt") + 1] == "no-new-privileges"
    user = argv[argv.index("--user") + 1]
    assert user == "65534:65534" and not user.startswith("0")
    assert argv[argv.index("--memory") + 1] == "512m"
    assert argv[argv.index("--cpus") + 1] == "1.0"
    assert argv[argv.index("--pids-limit") + 1] == "256"
    assert argv[argv.index("--tmpfs") + 1].startswith("/tmp:")
    # Exactly ONE bind mount: the workspace, read-write.
    assert argv.count("-v") == 1 and argv.count("--volume") == 0
    assert argv[argv.index("-v") + 1] == f"{ws}:/workspace:rw"
    assert argv[argv.index("-w") + 1] == "/workspace/notes"
    assert argv[argv.index("--name") + 1] == "si-gaia-test-1"
    img = argv.index(profile.image)
    assert argv[img + 1:] == ["/bin/sh", "-c", "echo hi"]


def test_profile_summary_is_journal_shaped():
    s = SandboxProfile().summary("/host/work")
    assert s["mode"] == "sandbox"
    assert s["image"] == "python:3.12-slim"
    assert s["network"] == "none"
    assert s["mounts"] == ["/host/work:/workspace:rw"]
    assert s["read_only_rootfs"] is True
    assert s["user"] == "65534:65534"


def test_resolve_exec_mode_matrix():
    assert resolve_exec_mode(None, True) == "sandbox"
    assert resolve_exec_mode(None, False) == "host"
    assert resolve_exec_mode("host", True) == "host"
    assert resolve_exec_mode("sandbox", False) == "sandbox"
    with pytest.raises(ValueError):
        resolve_exec_mode("bogus", True)


def test_docker_available_is_false_without_a_cli(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert docker_available() is False


# -- the mode matrix through the real executor -----------------------------------

def test_unset_mode_with_docker_runs_sandboxed_and_not_on_host(
        mini, monkeypatch):
    monkeypatch.delenv("GAIA_EXEC_MODE", raising=False)
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        canned = (0, b"sandbox-stdout\n", b"", False, False)
        ex = _executor(bundle_dir, manifest, journal, work, canned=canned)
        assert ex.exec_mode == "sandbox"  # probe True, nothing explicit
        out = _run(bundle_dir, manifest, journal, work, ex, "shell_exec",
                   {"command": "touch host-marker.txt"})
        rec = out["execution"]
        assert rec["status"] == "completed"
        # The result is the fake's canned output — the host shell
        # never ran (its marker file does not exist).
        assert rec["result"]["stdout"] == "sandbox-stdout\n"
        assert not os.path.exists(os.path.join(work, "host-marker.txt"))
        fake = ex._sandbox_runner
        assert len(fake.calls) == 1
        docker_argv = fake.calls[0]
        assert docker_argv[:2] == ["docker", "run"]
        assert docker_argv[docker_argv.index("-w") + 1] == "/workspace"
        img = docker_argv.index("python:3.12-slim")
        assert docker_argv[img + 1:] == \
            ["/bin/sh", "-c", "touch host-marker.txt"]
        # The journaled record itself carries the mode + profile.
        journaled = _journaled_execution(journal, out["edit_id"])
        assert journaled["exec_mode"] == "sandbox"
        assert journaled["sandbox_profile"]["image"] == "python:3.12-slim"
        assert journaled["sandbox_profile"]["network"] == "none"
        assert rec["exec_mode"] == "sandbox" and "sandbox_profile" in rec
    finally:
        journal.close()


def test_explicit_host_mode_runs_on_the_host_for_real(mini, monkeypatch):
    monkeypatch.delenv("GAIA_EXEC_MODE", raising=False)
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        ex = _executor(bundle_dir, manifest, journal, work,
                       exec_mode="host")
        assert ex.exec_mode == "host"
        out = _run(bundle_dir, manifest, journal, work, ex, "shell_exec",
                   {"command": "echo hello-from-host"})
        rec = out["execution"]
        assert rec["status"] == "completed" and rec["exit_code"] == 0
        assert "hello-from-host" in rec["result"]["stdout"]
        assert rec["exec_mode"] == "host"
        assert "sandbox_profile" not in rec
        journaled = _journaled_execution(journal, out["edit_id"])
        assert journaled["exec_mode"] == "host"
    finally:
        journal.close()


def test_explicit_sandbox_without_docker_fails_closed(mini, monkeypatch):
    monkeypatch.delenv("GAIA_EXEC_MODE", raising=False)
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        canned = (0, b"never\n", b"", False, False)
        ex = _executor(bundle_dir, manifest, journal, work, canned=canned,
                       exec_mode="sandbox", docker_probe=lambda: False)
        out = _run(bundle_dir, manifest, journal, work, ex, "shell_exec",
                   {"command": "touch host-marker.txt"})
        rec = out["execution"]
        # Journaled failure, and NOTHING ran: not the fake runner,
        # not the host shell.
        assert rec["status"] == "failed"
        assert "sandbox execution refused" in rec["detail"]
        assert "fail closed" in rec["detail"]
        assert ex._sandbox_runner.calls == []
        assert not os.path.exists(os.path.join(work, "host-marker.txt"))
        # The refusal record still states the intended mode.
        assert rec["exec_mode"] == "sandbox"
        assert "sandbox_profile" in rec
        journaled = _journaled_execution(journal, out["edit_id"])
        assert journaled["status"] == "failed"
        assert journaled["exec_mode"] == "sandbox"
    finally:
        journal.close()


def test_unset_mode_without_docker_defaults_to_host(mini, monkeypatch):
    monkeypatch.delenv("GAIA_EXEC_MODE", raising=False)
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        ex = _executor(bundle_dir, manifest, journal, work,
                       docker_probe=lambda: False)
        # The documented compatibility default — and it is
        # journal-visible on the record (asserted below).
        assert ex.exec_mode == "host"
        out = _run(bundle_dir, manifest, journal, work, ex, "shell_exec",
                   {"command": "echo compat-default"})
        rec = out["execution"]
        assert rec["status"] == "completed"
        assert "compat-default" in rec["result"]["stdout"]
        assert rec["exec_mode"] == "host"
    finally:
        journal.close()


def test_env_var_selects_the_mode(mini, monkeypatch):
    monkeypatch.setenv("GAIA_EXEC_MODE", "host")
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        ex = _executor(bundle_dir, manifest, journal, work)
        assert ex.exec_mode == "host"
    finally:
        journal.close()


# -- argv translation for code_exec / test_run -------------------------------------

def test_code_exec_source_form_translates_to_python3(mini, monkeypatch):
    monkeypatch.delenv("GAIA_EXEC_MODE", raising=False)
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        canned = (0, b"ok\n", b"", False, False)
        ex = _executor(bundle_dir, manifest, journal, work, canned=canned)
        out = _run(bundle_dir, manifest, journal, work, ex, "code_exec",
                   {"language": "python", "source": "print('x')"})
        assert out["execution"]["status"] == "completed"
        docker_argv = ex._sandbox_runner.calls[0]
        img = docker_argv.index("python:3.12-slim")
        assert docker_argv[img + 1:] == ["python3", "-c", "print('x')"]
    finally:
        journal.close()


def test_code_exec_file_form_and_test_run_map_paths(mini, monkeypatch):
    monkeypatch.delenv("GAIA_EXEC_MODE", raising=False)
    bundle_dir, work = mini
    with open(os.path.join(work, "calc.py"), "w") as f:
        f.write("print('path-form')\n")
    with open(os.path.join(work, "test_sample.py"), "w") as f:
        f.write("def test_a():\n    assert True\n")
    manifest, journal = _open(bundle_dir)
    try:
        canned = (0, b"1 passed in 0.05s\n", b"", False, False)
        ex = _executor(bundle_dir, manifest, journal, work, canned=canned)
        out = _run(bundle_dir, manifest, journal, work, ex, "code_exec",
                   {"language": "python", "path": "calc.py"})
        assert out["execution"]["status"] == "completed"
        docker_argv = ex._sandbox_runner.calls[-1]
        img = docker_argv.index("python:3.12-slim")
        assert docker_argv[img + 1:] == ["python3", "/workspace/calc.py"]

        out = _run(bundle_dir, manifest, journal, work, ex, "test_run",
                   {"path": "test_sample.py"})
        rec = out["execution"]
        assert rec["status"] == "completed"
        assert rec["result"]["parse"] == "ok"
        assert rec["result"]["passed"] == 1
        docker_argv = ex._sandbox_runner.calls[-1]
        img = docker_argv.index("python:3.12-slim")
        assert docker_argv[img + 1:] == [
            "python3", "-m", "pytest", "/workspace/test_sample.py", "-q"]
    finally:
        journal.close()


def test_shell_exec_subdirectory_cwd_maps_under_workspace(
        mini, monkeypatch):
    monkeypatch.delenv("GAIA_EXEC_MODE", raising=False)
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        canned = (0, b"", b"", False, False)
        ex = _executor(bundle_dir, manifest, journal, work, canned=canned)
        out = _run(bundle_dir, manifest, journal, work, ex, "shell_exec",
                   {"command": "true", "cwd": "notes"})
        assert out["execution"]["status"] == "completed"
        docker_argv = ex._sandbox_runner.calls[0]
        assert docker_argv[docker_argv.index("-w") + 1] == "/workspace/notes"
    finally:
        journal.close()


# -- invalid modes fail fast --------------------------------------------------------

def test_invalid_exec_mode_is_a_construction_error(mini, monkeypatch):
    bundle_dir, work = mini
    manifest, journal = _open(bundle_dir)
    try:
        with pytest.raises(ValueError):
            _executor(bundle_dir, manifest, journal, work,
                      exec_mode="bogus")
        monkeypatch.setenv("GAIA_EXEC_MODE", "bogus")
        with pytest.raises(ValueError):
            _executor(bundle_dir, manifest, journal, work)
        monkeypatch.delenv("GAIA_EXEC_MODE")
        with pytest.raises(ValueError):
            Executor(bundle_dir, manifest, journal,
                     corpus_policy(work, {"exec_mode": "bogus"}), work,
                     docker_probe=lambda: True)
    finally:
        journal.close()
