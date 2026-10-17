"""``si serve``: serve the bundle's current head for inference over HTTP.

Minimal, localhost-oriented, read-only HTTP API (stdlib
``http.server`` only — no new dependencies):

    GET  /health  -> 200 JSON:
        {"status": "ok", "bundle_id": "si1:...", "bundle_name": ...,
         "head": "base"|"dXXXX", "mind_kind": "demo"|"transformer",
         "arch": ..., "arch_source": ..., "input_dim": 8|null,
         "inference_supported": true|false}
    GET  /        -> 200 JSON describing the endpoints above.
    POST /infer   -> demo bundles only. Body: {"input": [8 numbers]}
        (``"x"`` is accepted as an alias for ``"input"``.)
        200 JSON: {"output": [8 numbers], "head": ..., "bundle_id": ...,
                   "mind_kind": "demo"}
        The output is REAL inference: the demo mind materialized at the
        bundle head (frozen base + every active LoRA-style delta, exactly
        as ``si export`` materializes it) applied to the input —
        ``y = W_head @ x`` via ``DemoMind.forward`` over the active
        delta stack loaded from the bundle. No canned responses.
    POST /infer on a seeded transformer bundle -> 501 JSON with an
        explicit message: SI has no transformer inference engine, so
        transformer inference is not supported. Output is never
        fabricated for a checkpoint SI cannot actually run.
    Malformed / wrong-shape inputs -> 400; unknown paths -> 404.

Honest limits:
- Only the demo mind (``demo-linear``, 8-dim linear machine learning
  model) can be served. Real transformer checkpoints can be seeded
  and exported, but not run by this runtime.
- Read-only serving: the head is loaded and verified once at startup
  (``load_bundle``) and snapshotted; handlers perform no journal
  writes, create no deltas or provenance, and never move the head.
  Restart the server to serve a head that changed after startup.
- Localhost by default (``--host 127.0.0.1``). There is NO auth and
  no security claim: do not expose this server beyond localhost
  without putting your own controls in front of it.
"""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from si.bundle import load_bundle, manifest_id

#: Message returned (501) for any non-demo bundle. Shared with tests.
TRANSFORMER_NOT_SUPPORTED = (
    "transformer inference is not supported: SI has no transformer "
    "inference engine; only demo-mind (demo-linear) inference is "
    "supported by `si serve`"
)


def classify_mind(manifest: dict) -> str:
    """Return ``"demo"`` for the demo-seed base, else ``"transformer"``.

    The demo base is honestly labeled in its own manifest
    (``base.source`` starts with ``"demo:"`` and/or
    ``base.arch_source == "demo-unknown"``). Anything seeded via
    ``si seed`` is a real checkpoint base; this runtime has no engine
    that can run it, so it is classified ``"transformer"`` for serving
    purposes and inference on it is refused (501), never faked.
    """
    base = manifest.get("base", {}) or {}
    source = str(base.get("source", ""))
    if source.startswith("demo:") or base.get("arch_source") == "demo-unknown":
        return "demo"
    return "transformer"


def _arch_info(manifest: dict, mind_kind: str) -> tuple[str, str]:
    """(arch, arch_source) for /health — unknowns labeled, never invented."""
    base = manifest.get("base", {}) or {}
    arch_source = base.get("arch_source")
    if mind_kind == "demo":
        return "unknown", str(arch_source or "demo-unknown")
    if arch_source:
        cfg = base.get("checkpoint_config") or {}
        arch = cfg.get("model_type") if isinstance(cfg, dict) else None
        return str(arch or "unknown"), str(arch_source)
    # Seeded without a captured config: ask derive_arch (a pure function
    # of manifest + base tensor shapes recorded in the manifest); if it
    # cannot infer honestly, report "unknown" rather than guessing or
    # failing /health.
    try:
        from si.arch import derive_arch

        shapes = _load_base_tensors_for_shapes(manifest)
        arch = derive_arch(manifest, shapes)
        return str(arch["arch"]), str(arch["arch_source"])
    except Exception:
        return "unknown", "unknown"


def _load_base_tensors_for_shapes(manifest: dict) -> dict:
    """Base tensor name -> shape tuple, from the manifest when recorded.

    Seeded manifests record per-tensor shapes (``base["tensors"]`` or
    per-part ``tensors``); use those instead of loading tensor data.
    """
    base = manifest.get("base", {}) or {}
    shapes: dict = {}
    if isinstance(base.get("tensors"), dict):
        for name, info in base["tensors"].items():
            if isinstance(info, dict) and "shape" in info:
                shapes[name] = tuple(info["shape"])
    for part in base.get("parts") or []:
        for name, info in (part.get("tensors") or {}).items():
            if isinstance(info, dict) and "shape" in info:
                shapes[name] = tuple(info["shape"])
    return shapes


class ServeState:
    """Snapshot of the bundle head to serve. Built once, then read-only.

    For a demo bundle this holds a real ``DemoMind`` bound to the
    bundle plus its active delta stack (loaded via
    ``si.loop._load_stack``, the same resolution — including squash
    replacements — that the self-write loop verifies against), so
    ``infer`` computes exactly what calling the mind directly computes.
    For a transformer bundle no mind is constructed (constructing a
    DemoMind over transformer tensors would fail / mislead); inference
    is refused instead.
    """

    def __init__(self, bundle_dir: str):
        self.bundle_dir = os.path.abspath(bundle_dir)
        # Fail closed first: tampered bundles are never served.
        self.manifest = load_bundle(self.bundle_dir)
        self.bundle_id = manifest_id(self.manifest)
        self.head = self.manifest["head"]
        self.mind_kind = classify_mind(self.manifest)
        self.arch, self.arch_source = _arch_info(self.manifest, self.mind_kind)
        self.mind = None
        self.active: list = []
        self.input_dim: int | None = None
        if self.mind_kind == "demo":
            from si.loop import _load_stack
            from si.minds import DIM, DemoMind

            self.mind = DemoMind(bundle_path=self.bundle_dir)
            self.active = _load_stack(self.bundle_dir, self.manifest)
            self.input_dim = int(DIM)

    @property
    def inference_supported(self) -> bool:
        return self.mind_kind == "demo"

    def health(self) -> dict:
        return {
            "status": "ok",
            "bundle_id": self.bundle_id,
            "bundle_name": self.manifest.get("name"),
            "head": self.head,
            "mind_kind": self.mind_kind,
            "arch": self.arch,
            "arch_source": self.arch_source,
            "input_dim": self.input_dim,
            "inference_supported": self.inference_supported,
        }

    def infer(self, x) -> list:
        """Real demo-mind inference: y = W_head @ x. Demo bundles only."""
        import numpy as np

        if self.mind is None:
            raise NotSupportedError(TRANSFORMER_NOT_SUPPORTED)
        vec = _validate_input(x, self.input_dim)
        y = self.mind.forward(vec, self.active)
        return [float(v) for v in np.asarray(y, dtype=np.float64).tolist()]


class NotSupportedError(Exception):
    """Raised when inference is requested for an unsupported mind kind."""


def _validate_input(x, dim: int):
    """Validate a JSON-decoded input vector; ValueError on anything off."""
    import numpy as np

    if isinstance(x, (str, bytes)) or not isinstance(x, (list, tuple)):
        raise ValueError(f"input must be a JSON array of {dim} numbers")
    if len(x) != dim:
        raise ValueError(
            f"input must have exactly {dim} numbers (the demo mind is "
            f"{dim}-dimensional); got {len(x)}"
        )
    for v in x:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"input entries must be numbers; got {v!r}")
    vec = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(vec)):
        raise ValueError("input entries must be finite numbers")
    return vec


def make_server(bundle_dir: str, host: str = "127.0.0.1",
                port: int = 8000) -> ThreadingHTTPServer:
    """Build (but do not start) an HTTP server serving ``bundle_dir``'s head.

    ``port=0`` picks an ephemeral port (used by tests). The returned
    server carries its ``ServeState`` as ``server.state``.
    """
    state = ServeState(bundle_dir)

    class Handler(BaseHTTPRequestHandler):
        server_version = "si-serve/0.1"
        # Quieter logs: keep test / terminal output readable.
        def log_message(self, fmt, *args):  # noqa: A003 - stdlib name
            pass

        def _send_json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - stdlib name
            if self.path in ("/health", "/health/"):
                self._send_json(200, state.health())
            elif self.path == "/":
                self._send_json(200, {
                    "service": "si serve",
                    "bundle_id": state.bundle_id,
                    "head": state.head,
                    "mind_kind": state.mind_kind,
                    "endpoints": {
                        "GET /health": "bundle id, head, mind kind, arch_source",
                        "POST /infer": (
                            'demo bundles only: body {"input": [8 numbers]} '
                            "-> {\"output\": [8 numbers]}; seeded transformer "
                            "bundles return 501 (not supported)"
                        ),
                    },
                })
            else:
                self._send_json(404, {"error": f"unknown path {self.path!r}"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib name
            if self.path not in ("/infer", "/infer/"):
                self._send_json(404, {"error": f"unknown path {self.path!r}"})
                return
            if not state.inference_supported:
                self._send_json(501, {
                    "error": TRANSFORMER_NOT_SUPPORTED,
                    "mind_kind": state.mind_kind,
                    "head": state.head,
                    "bundle_id": state.bundle_id,
                })
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                payload = json.loads(raw.decode("utf-8")) if raw else {}
            except (ValueError, UnicodeDecodeError) as e:
                self._send_json(400, {"error": f"invalid JSON body: {e}"})
                return
            if not isinstance(payload, dict):
                self._send_json(400, {"error": "body must be a JSON object"})
                return
            x = payload.get("input", payload.get("x"))
            if x is None:
                self._send_json(400, {
                    "error": 'missing "input": expected '
                             '{"input": [8 numbers]}'})
                return
            try:
                output = state.infer(x)
            except NotSupportedError as e:
                self._send_json(501, {"error": str(e),
                                      "mind_kind": state.mind_kind})
                return
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return
            self._send_json(200, {
                "output": output,
                "head": state.head,
                "bundle_id": state.bundle_id,
                "mind_kind": state.mind_kind,
            })

    server = ThreadingHTTPServer((host, int(port)), Handler)
    server.state = state  # type: ignore[attr-defined]
    return server


def serve_bundle(bundle_dir: str, host: str = "127.0.0.1",
                 port: int = 8000) -> dict:
    """Serve ``bundle_dir``'s head until interrupted (Ctrl-C).

    Returns a small summary dict after shutdown. Read-only: no journal
    writes, head never moves (see module docstring).
    """
    server = make_server(bundle_dir, host=host, port=port)
    state = server.state  # type: ignore[attr-defined]
    actual_host, actual_port = server.server_address[:2]
    print(f"serving bundle {state.bundle_id}")
    print(f"head: {state.head}  mind_kind: {state.mind_kind}  "
          f"arch_source: {state.arch_source}")
    print(f"listening on http://{actual_host}:{actual_port} "
          f"(GET /health, POST /infer)")
    if not state.inference_supported:
        print(f"note: {TRANSFORMER_NOT_SUPPORTED}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return {"bundle_id": state.bundle_id, "head": state.head,
            "mind_kind": state.mind_kind, "host": actual_host,
            "port": actual_port}


__all__ = [
    "TRANSFORMER_NOT_SUPPORTED",
    "NotSupportedError",
    "ServeState",
    "classify_mind",
    "make_server",
    "serve_bundle",
]
