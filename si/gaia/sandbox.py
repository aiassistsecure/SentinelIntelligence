"""SI-125 SANDBOX-1: a Docker sandbox profile for Gaia's
shell/code execution tools (RFC §8.3 process tools).

The executor's process tools (``shell_exec``, ``code_exec``,
``test_run``) historically ran on the host with an allowlist
environment and a process-group kill discipline — containment by
convention, not by isolation. This module gives the executor a
real isolation boundary: when the resolved execution mode is
``sandbox``, the same inner argv runs inside a hardened,
single-use Docker container instead of on the host.

The profile is deliberately small and fixed in shape:

- a read-only root filesystem with exactly ONE bind mount — the
  executor's work root at ``/workspace``, read-write — so a
  command can write only where the executor already allows it to
  write on the host path;
- no network (``--network none``);
- no capabilities (``--cap-drop ALL``), no privilege escalation
  (``no-new-privileges``), a non-root user, and hard memory /
  CPU / PID ceilings;
- a size-capped tmpfs at ``/tmp`` (the read-only rootfs leaves
  no other writable scratch);
- ``--rm``: the container is single-use and leaves nothing
  behind but the journaled execution record.

Mode selection is fail-closed where it matters: an *explicit*
``sandbox`` mode with Docker unavailable refuses the execution
(journaled as a failure) rather than silently falling back to
the host. The one documented compatibility exception is the
*unset* mode: with no explicit choice anywhere, Docker's
presence selects sandbox and its absence selects host — and
every execution record states which mode actually ran, so the
fallback is never silent in the journal.

This module is the profile + runner layer only. The executor
(:mod:`si.gaia.executor`) owns mode resolution, argv/cwd
translation, stream capping, and journaling; runners here return
raw bytes and leave capping to that caller, exactly as the host
path's discipline is owned by the executor.
"""
from __future__ import annotations

import itertools
import os
import shutil
import signal
import subprocess
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass

#: Where the work root is mounted inside the container. Fixed:
#: inner argv/cwd translation in the executor targets this path.
CONTAINER_WORKSPACE = "/workspace"

#: The execution modes the executor understands.
EXEC_MODES = ("sandbox", "host")


@dataclass
class SandboxProfile:
    """The container hardening profile for one sandboxed run.

    Only the operator-tunable knobs are fields; the hardening
    shape itself (read-only rootfs, cap-drop ALL,
    no-new-privileges, --rm, single workspace bind mount) is
    fixed by :func:`build_docker_argv` and is not negotiable per
    profile — a profile that could relax it would not be a
    profile, it would be an escape hatch.
    """

    #: Tag-pinned image. Digest pinning is stronger and is an
    #: operator hardening step (see docs/SI-125-sandbox.md); the
    #: default stays a tag so a fresh Docker host works without
    #: a registry lookup ritual. NOTE: the slim image carries no
    #: pytest — ``test_run`` in sandbox mode needs an operator
    #: image that does, and fails honestly inside the container
    #: (journaled) until one is configured.
    image: str = "python:3.12-slim"
    #: nobody:nobody — never root inside the container.
    user: str = "65534:65534"
    memory: str = "512m"
    cpus: str = "1.0"
    pids_limit: int = 256
    network: str = "none"
    #: Size cap for the writable tmpfs at /tmp.
    tmpfs_size: str = "64m"

    def summary(self, workspace_host_path: str | None = None) -> dict:
        """The journal-facing summary of this profile (§4.4):
        measured configuration facts only, safe to journal."""
        mount = (f"{workspace_host_path}:{CONTAINER_WORKSPACE}:rw"
                 if workspace_host_path else
                 f"work_root:{CONTAINER_WORKSPACE}:rw")
        return {
            "mode": "sandbox",
            "image": self.image,
            "user": self.user,
            "network": self.network,
            "mounts": [mount],
            "memory": self.memory,
            "cpus": self.cpus,
            "pids_limit": self.pids_limit,
            "read_only_rootfs": True,
        }


def build_docker_argv(profile: SandboxProfile, workspace_host_path: str,
                      inner_argv, inner_cwd: str, *, name: str) -> list:
    """Build the full ``docker run`` argv (a list — never a shell
    string) for one sandboxed execution.

    ``inner_cwd`` is a container path (``/workspace`` or a
    subpath); ``inner_argv`` is the already-translated inner
    command. The result runs ``inner_argv`` after the image name,
    with exactly one bind mount: the workspace.
    """
    return [
        "docker", "run",
        "--rm",
        "--name", name,
        "--network", profile.network,
        "--read-only",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--user", profile.user,
        "--memory", profile.memory,
        "--cpus", profile.cpus,
        "--pids-limit", str(profile.pids_limit),
        "--tmpfs", f"/tmp:size={profile.tmpfs_size}",
        "-v", f"{workspace_host_path}:{CONTAINER_WORKSPACE}:rw",
        "-w", inner_cwd,
        profile.image,
        *[str(a) for a in inner_argv],
    ]


def docker_available() -> bool:
    """Cheap, honest Docker probe: the CLI exists AND the daemon
    answers a version query within a short timeout. Any failure
    — no CLI, no daemon, timeout, non-zero exit — is False.
    Module-level so tests (and the executor) can substitute it.
    """
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, timeout=5)
    except Exception:
        return False
    return proc.returncode == 0 and bool((proc.stdout or b"").strip())


def resolve_exec_mode(explicit, docker_ok: bool) -> str:
    """Resolve the execution mode: an explicit choice (constructor
    arg, ``GAIA_EXEC_MODE``, or policy ``exec_mode`` — the caller
    applies that precedence) always wins; with no explicit
    choice, sandbox when Docker is available, host when it is
    not (the documented compatibility default). An explicit
    value outside :data:`EXEC_MODES` is a ``ValueError`` — fail
    fast and honest at construction, never a silent coercion.
    """
    if explicit is not None:
        if explicit not in EXEC_MODES:
            raise ValueError(
                f"invalid exec mode {explicit!r}: expected one of "
                f"{EXEC_MODES} (constructor arg, GAIA_EXEC_MODE, or "
                "policy 'exec_mode')")
        return explicit
    return "sandbox" if docker_ok else "host"


class ProcessRunner(ABC):
    """Runs one (already container-translated) inner argv and
    returns raw results: ``(exit_code, stdout_bytes,
    stderr_bytes, timed_out, leash_killed)``. Stream capping and
    decoding are the caller's (the executor's) job, mirroring the
    host path where ``_run_process`` owns the caps.
    """

    @abstractmethod
    def run(self, argv, cwd, timeout, env, leash_check=None):
        raise NotImplementedError


class DockerRunner(ProcessRunner):
    """Production runner: builds the ``docker run`` argv via
    :func:`build_docker_argv` under a unique container name and
    drives the Docker client with the same poll/leash discipline
    as the executor's host path — short poll slices while a leash
    check is installed; on timeout or leash stop, a best-effort
    ``docker kill <name>`` (the container is the real process
    boundary) followed by killing the client process group.
    A timed-out/leash-killed run reports exit code -1, the
    executor's documented no-exit sentinel (TIMEOUT_EXIT_CODE).
    """

    def __init__(self, profile: SandboxProfile, workspace_host_path: str):
        self.profile = profile
        self.workspace_host_path = workspace_host_path
        self._counter = itertools.count(1)
        #: Introspection for tests/operators: the last container
        #: name and full docker argv this runner built.
        self.last_container_name = None
        self.last_docker_argv = None

    def _container_name(self) -> str:
        return (f"si-gaia-{os.getpid()}-{next(self._counter)}-"
                f"{uuid.uuid4().hex[:8]}")

    def run(self, argv, cwd, timeout, env, leash_check=None):
        name = self._container_name()
        docker_argv = build_docker_argv(
            self.profile, self.workspace_host_path, argv, cwd,
            name=name)
        self.last_container_name = name
        self.last_docker_argv = docker_argv
        return self._spawn(docker_argv, name, timeout, env, leash_check)

    def _spawn(self, docker_argv, name, timeout, env, leash_check):
        proc = subprocess.Popen(
            docker_argv, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True)
        timed_out = False
        leash_killed = False
        deadline = time.monotonic() + timeout
        out_b = err_b = b""
        exit_code = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            slice_s = remaining if leash_check is None \
                else min(remaining, 0.05)
            try:
                out_b, err_b = proc.communicate(timeout=slice_s)
                exit_code = proc.returncode
                break
            except subprocess.TimeoutExpired:
                if leash_check is not None:
                    try:
                        stopped = bool(leash_check())
                    except Exception:
                        stopped = False
                    if stopped:
                        leash_killed = True
                        timed_out = True
                        break
                continue
        if timed_out:
            # The container is the boundary: kill it by name
            # first (best-effort), then the client process group,
            # mirroring the host path's group kill.
            try:
                subprocess.run(["docker", "kill", name],
                               capture_output=True, timeout=10)
            except Exception:
                pass
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    proc.kill()
                except (ProcessLookupError, PermissionError):
                    pass
            out_b, err_b = proc.communicate()
            exit_code = -1  # TIMEOUT_EXIT_CODE: no exit exists
        return exit_code, out_b or b"", err_b or b"", \
            timed_out, leash_killed
