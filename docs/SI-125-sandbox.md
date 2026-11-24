# SI-125 SANDBOX-1 — Docker sandbox profile for Gaia's shell/code execution

Gaia's process tools (`shell_exec`, `code_exec`, `test_run` — RFC §8.3)
previously ran on the host: an allowlist environment, per-stream caps,
a timeout, and a process-group kill. That is discipline, not isolation.
SI-125 adds the isolation boundary: in **sandbox mode** the same inner
command runs inside a hardened, single-use Docker container, through
the executor's existing single choke point (`Executor._run_process`),
so every discipline the host path already had — stream caps, timeout,
leash kill, journaled §4.4 records — applies unchanged, and every
execution record now states which mode actually ran.

Code: `si/gaia/sandbox.py` (profile, argv builder, probe, runners,
mode resolution) + a surgical integration in `si/gaia/executor.py`.
Tests: `tests/test_gaia_si125_sandbox.py` (13 tests, all faked at the
runner/probe seams — no Docker, no network).

## 1. The sandbox profile

`SandboxProfile` (dataclass) carries the operator-tunable knobs; the
hardening *shape* is fixed in `build_docker_argv` and cannot be
relaxed per profile.

| Flag | Value | Defends against |
|---|---|---|
| `--network none` | no network interfaces beyond loopback-less isolation | exfiltration, calling home, reaching LAN/cloud metadata services, downloading payloads |
| `--read-only` (rootfs) | container root filesystem immutable | persistence, tampering with the image/system binaries, writing outside the one allowed mount |
| `-v <work_root>:/workspace:rw` | exactly ONE bind mount | the command can write only where the executor already permits host writes — the work root; the rest of the host filesystem is simply absent |
| `-w /workspace[/sub]` | workdir = translated inner cwd | cwd escapes; the process starts inside the mount, never at `/` |
| `--cap-drop ALL` | zero Linux capabilities | privilege escalation via cap-gated syscalls (NET_RAW, SYS_ADMIN, DAC_OVERRIDE, …) even if a setuid binary is found |
| `--security-opt no-new-privileges` | exec cannot gain privileges | setuid/setgid escalation inside the container |
| `--user 65534:65534` | nobody:nobody, never root | root-in-container damage; files written to the mount are owned by nobody, and a container escape lands as an unprivileged uid |
| `--memory 512m` | hard RAM ceiling (profile field) | memory exhaustion of the host by a runaway command |
| `--cpus 1.0` | CPU quota (profile field) | CPU starvation of the host / the resident mind |
| `--pids-limit 256` | PID ceiling (profile field) | fork bombs |
| `--tmpfs /tmp:size=64m` | size-capped writable tmpfs | the read-only rootfs leaves no scratch space; tools that need a temp dir get one that cannot grow into a disk-fill |
| `--rm` | container removed on exit | leftover state, writable-layer accumulation, cross-run contamination |

The child environment is the executor's existing allowlist
(`CHILD_ENV_ALLOWLIST`) — unchanged; no runtime credential reaches the
container either. Timeout and operator-leash stops kill the container
by name (`docker kill <name>`, best-effort) and then the client
process group, mirroring the host path's group kill; a killed run
records the same `TIMEOUT_EXIT_CODE` (-1) no-exit sentinel.

### Image pinning

The default image is **tag-pinned**: `python:3.12-slim`. A tag is a
deliberate trade: a fresh Docker host can pull and run it with no
extra ritual, and the tag tracks 3.12 patch releases. **Digest
pinning** (`python:3.12-slim@sha256:<digest>`) is the stronger
posture — it makes the image bit-for-bit reproducible and immune to
tag re-pointing — and is the recommended operator hardening: set
`sandbox_profile` / construct `SandboxProfile(image=...)` with the
digest form on any host where the extra supply-chain guarantee
matters. The slice does not digest-pin the default because the
correct digest is a fact about a registry at a moment in time, and
this environment has no Docker and no network to measure it with;
inventing one would be worse than the honest tag.

**The slim image has no pytest.** `shell_exec` and `code_exec` work
against the default image as-is. `test_run` in sandbox mode needs an
operator image that carries pytest (e.g. a project image built on
`python:3.12` with pytest installed, digest-pinned per above). Until
one is configured, a sandboxed `test_run` fails *honestly inside the
container* — pytest's absence is the command's real stderr/exit, and
the outcome is journaled like any other completed/failed execution.
Nothing silently reroutes it to the host.

## 2. Mode selection

Resolution order (first hit wins), applied once at `Executor`
construction; all new constructor parameters are optional kwargs, so
no existing call site changed:

1. constructor kwarg `exec_mode=`
2. environment variable `GAIA_EXEC_MODE`
3. policy key `"exec_mode"`
4. default: `resolve_exec_mode(None, docker_available())` —
   **sandbox if Docker answers the probe, host if it does not.**

An explicit value outside `{"sandbox", "host"}` is a `ValueError` at
construction — fail fast, never a silent coercion.

| Explicit mode | Docker at construction | Docker at execution | Behaviour |
|---|---|---|---|
| `sandbox` | any | available | runs in the container |
| `sandbox` | any | **unavailable** | **fail closed**: `ToolExecutionError`, journaled as a *failed* execution whose detail states the refusal; the command never runs — not on the host, not anywhere |
| `host` | any | any | runs on the host, exactly the pre-SI-125 discipline |
| unset | available | available | sandbox (as above) |
| unset | available | unavailable at execution | fail closed (as above) — availability is re-probed at execution time, never trusted from construction |
| unset | **unavailable** | — | **host** — the one documented exception (below) |

**The fail-closed rule:** once sandbox is the resolved mode — by any
route — a missing Docker refuses the execution. There is no fallback
path from a sandbox decision to a host run.

**The one documented exception:** when *no* explicit mode is set
anywhere and Docker is unavailable, the mode resolves to `host`.
This is a compatibility default, not a silent one: SI-125 exists so
that existing deployments (including this VM, which has no Docker by
design) keep working unchanged, and **every** process-tool execution
record carries `exec_mode` (plus `sandbox_profile` in sandbox mode),
so the mode that actually ran is journal-visible on every record,
forever. An operator who wants the fail-closed guarantee on a
Docker-less host sets `GAIA_EXEC_MODE=sandbox` — and gets refusals,
journaled, until Docker exists.

Non-process tools (`fs_*`, `journal_query`, `bundle_status`,
`probe_run`) are untouched: they never spawn processes, and their
records carry no `exec_mode` (nothing ran in a mode).

## 3. Docker-absent behaviour on this VM

This VM has no Docker (by design; nothing was installed). Therefore:

- `docker_available()` returns `False` (no `docker` on PATH — the
  probe short-circuits before any subprocess).
- Every default-constructed `Executor` resolves to `host` mode and
  behaves byte-for-byte as before SI-125; the full pre-existing
  executor test surface passes unchanged.
- Setting `GAIA_EXEC_MODE=sandbox` here produces journaled failed
  executions with the fail-closed refusal detail — the honest,
  testable behaviour, pinned by
  `test_explicit_sandbox_without_docker_fails_closed`.
- The container path itself is tested through `FakeRunner`, a
  `DockerRunner` subclass that runs the production argv-building
  code and fakes only the subprocess spawn.

## 4. LIVE-NODE IMPACT

Who constructs `Executor`: the self-write loop's miniature branch
(`si/loop.py`), the queue drainer (`si/gaia/drain.py`), and the
measure / resolution tooling (`si/gaia/measure.py`,
`si/gaia/resolution.py`), plus `execute_tool_cycle`'s default. All
construct it positionally; SI-125's kwargs are optional, so none of
these call sites changed or need to.

What serves the live node on this VM (verified against the systemd
units in `~/workspace/gaia-node/` and `/etc/systemd/system/`):

- **`gaia-node.service`** runs the **Gaia fork** binary
  (`aiassistsecure/gaia`, Imagine v13 Resident) — not this repo. Its
  seam work is executed by the fork itself, submitted via the seam's
  HTTP spend path (`si/gaia/seam_spend.py` is the SI side of the
  accounting, not the execution). **The fork's own executor is out
  of scope for this slice and is untouched.**
- **`si-seam-service.service`** runs `si gaia-serve` — the
  `si/gaia/service.py` SSE spine (proposal ingest / Flow 1) — with
  `PYTHONPATH` pointing at a *different checkout*
  (`~/workspace/sentinel-main`), not this clone. `gaia-serve` can
  also run a submitted task through B-3's `run_task` →
  `execute_tool_cycle` → `Executor`; that path remains
  miniature-scoped exactly as in B-5.
- **No systemd unit on this VM invokes `si loop`, the queue
  drainer, measure, or resolution tooling.**

Precisely, then: merging SI-125 changes **nothing on the live node
on any restart**. `si-seam-service` does not load this clone's code;
even where an `Executor` is constructed (a `gaia-serve` task
submission, an operator-run loop/drain), this VM has no Docker, so
the default mode resolves to `host` and behaviour is identical —
with `exec_mode: "host"` now recorded on each process-tool record.
Sandbox mode first becomes *reachable* on a host that has Docker:
there, an unset mode resolves to sandbox by default (see §2), which
is the slice's intent — isolation by default where the boundary
exists, honesty in the journal everywhere.

## 5. OWED LIVE DOCKER SMOKE — **OWED / unavailable in this environment**

Docker does not exist on this VM, so the end-to-end container run
has **not** been executed. This section is owed, not done. On the
first Docker host, before sandbox mode is trusted live:

1. Set `GAIA_EXEC_MODE=sandbox`; run one `shell_exec`
   (`echo hello; touch marker.txt`) and one `code_exec` (source
   form) against a miniature bundle.
2. Verify the §4.4 records carry `exec_mode: "sandbox"` and the
   expected `sandbox_profile`, and the commands' stdout matches.
3. Verify **network none actually blocks egress**: from inside a
   sandboxed `shell_exec`, attempt an outbound connection (e.g.
   `python3 -c` opening a socket to a public IP) — it must fail.
4. Verify **writes land only in the workspace mount**: `marker.txt`
   exists under the work root; an attempt to write outside it
   (e.g. `/etc/x`, `/root/x`) fails against the read-only rootfs /
   absent paths; no container remains afterwards (`docker ps -a`).
5. Verify a timeout and (where wired) an operator leash stop kill
   the container, not just the client.
6. Record the results — host, image digest used, dates, outcomes —
   in this section, replacing this owed notice.
