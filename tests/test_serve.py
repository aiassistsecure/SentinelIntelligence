"""SI-5: ``si serve`` — serve the bundle head for inference.

Covers: /health fields for a demo bundle; POST /infer round-trip over
real HTTP asserted against calling the demo mind directly (same bundle,
same active stack — so the server must be computing the materialized
head, deltas included, not returning canned data); invalid inputs ->
400; unknown paths -> 404; the seeded-transformer case (tiny llama
fixture synthesized on the fly, like tests/test_export_arch.py) where
/health reports the transformer honestly and /infer returns 501 with
an explicit not-supported message; and read-only serving (head,
deltas, and manifest bytes unchanged after requests).

The sandbox sets HTTP(S)_PROXY env vars that break localhost urllib
calls, so NO_PROXY is set for loopback before any request is made.
"""
import json
import os
import threading
import urllib.error
import urllib.request

os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

import numpy as np
from safetensors.numpy import save

from si.bundle import init_bundle, load_bundle, manifest_id
from si.seed import seed_bundle
from si.serve import make_server


def _init(tmp_path, name):
    return init_bundle(str(tmp_path / (name + ".si")), name=name,
                       journal_backend="embedded")


def _demo_cycles(bundle_dir, n):
    from si.loop import run_cycle
    from si.minds import DemoMind
    for _ in range(n):
        res = run_cycle(bundle_dir, DemoMind(bundle_path=bundle_dir))
        assert res["verdict"] == "ADMIT"
    return load_bundle(bundle_dir)


class _Running:
    def __init__(self, bundle_dir):
        self.server = make_server(bundle_dir, host="127.0.0.1", port=0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"


def _get(running, path):
    try:
        with urllib.request.urlopen(running.url(path), timeout=10) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def _post(running, path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(running.url(path), data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def _direct_output(bundle_dir, x):
    """What calling the mind directly computes at the current head."""
    from si.loop import _load_stack
    from si.minds import DemoMind
    manifest = load_bundle(bundle_dir)
    mind = DemoMind(bundle_path=bundle_dir)
    active = _load_stack(bundle_dir, manifest)
    return [float(v) for v in mind.forward(np.asarray(x, dtype=np.float64),
                                            active)]


# -- demo bundle: health + real inference --------------------------------------

def test_serve_health_demo(tmp_path):
    bundle_dir = _init(tmp_path, "serve-health")
    manifest = _demo_cycles(bundle_dir, 1)
    with _Running(bundle_dir) as srv:
        status, body = _get(srv, "/health")
    assert status == 200
    assert body["status"] == "ok"
    assert body["bundle_id"] == manifest_id(manifest)
    assert body["head"] == "d0001" == manifest["head"]
    assert body["mind_kind"] == "demo"
    assert body["arch_source"] == "demo-unknown"
    assert body["input_dim"] == 8
    assert body["inference_supported"] is True


def test_serve_infer_demo_roundtrip(tmp_path):
    bundle_dir = _init(tmp_path, "serve-infer")
    manifest = _demo_cycles(bundle_dir, 2)
    assert manifest["head"] == "d0002", "infer must serve the delta stack"

    x = [0.5, -1.0, 2.0, 0.25, 0.0, 1.5, -0.75, 3.0]
    expected = _direct_output(bundle_dir, x)

    with _Running(bundle_dir) as srv:
        status, body = _post(srv, "/infer", {"input": x})
    assert status == 200
    assert body["head"] == "d0002"
    assert body["mind_kind"] == "demo"
    assert body["bundle_id"] == manifest_id(manifest)
    assert len(body["output"]) == 8
    np.testing.assert_allclose(body["output"], expected, rtol=1e-9, atol=1e-9)

    # A different input must give the (different) directly computed
    # output — guards against a canned / constant response.
    x2 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    expected2 = _direct_output(bundle_dir, x2)
    assert not np.allclose(expected, expected2)
    with _Running(bundle_dir) as srv:
        status, body = _post(srv, "/infer", {"input": x2})
    assert status == 200
    np.testing.assert_allclose(body["output"], expected2, rtol=1e-9, atol=1e-9)


def test_serve_infer_base_only(tmp_path):
    bundle_dir = _init(tmp_path, "serve-base")
    x = [0.1] * 8
    expected = _direct_output(bundle_dir, x)
    with _Running(bundle_dir) as srv:
        status, body = _post(srv, "/infer", {"input": x})
    assert status == 200
    assert body["head"] == "base"
    np.testing.assert_allclose(body["output"], expected, rtol=1e-9, atol=1e-9)


def test_serve_infer_invalid_and_unknown(tmp_path):
    bundle_dir = _init(tmp_path, "serve-bad")
    with _Running(bundle_dir) as srv:
        # Wrong length.
        status, body = _post(srv, "/infer", {"input": [1.0, 2.0]})
        assert status == 400 and "exactly 8" in body["error"]
        # Non-numeric entries.
        status, body = _post(srv, "/infer", {"input": ["a"] * 8})
        assert status == 400
        # Missing input.
        status, body = _post(srv, "/infer", {"nope": []})
        assert status == 400
        # Unknown paths.
        status, body = _get(srv, "/nope")
        assert status == 404
        status, body = _post(srv, "/nope", {"input": [0.0] * 8})
        assert status == 404


def test_serve_is_read_only(tmp_path):
    bundle_dir = _init(tmp_path, "serve-ro")
    _demo_cycles(bundle_dir, 1)
    manifest_path = os.path.join(bundle_dir, "manifest.json")
    with open(manifest_path, "rb") as f:
        manifest_bytes_before = f.read()
    before = load_bundle(bundle_dir)

    with _Running(bundle_dir) as srv:
        assert _get(srv, "/health")[0] == 200
        assert _post(srv, "/infer", {"input": [1.0] * 8})[0] == 200

    with open(manifest_path, "rb") as f:
        assert f.read() == manifest_bytes_before, \
            "serving must not rewrite the manifest"
    after = load_bundle(bundle_dir)  # still verifies cleanly
    assert after["head"] == before["head"]
    assert [d["id"] for d in after["deltas"]] == \
        [d["id"] for d in before["deltas"]]


# -- seeded transformer: honest not-supported -----------------------------------

HIDDEN = 16
VOCAB = 32
LAYERS = 2


def _llama_ckpt(tmp_path, name):
    rng = np.random.default_rng(1234)
    tensors = {
        "model.embed_tokens.weight": (VOCAB, HIDDEN),
        "model.norm.weight": (HIDDEN,),
        "lm_head.weight": (VOCAB, HIDDEN),
    }
    for i in range(LAYERS):
        p = f"model.layers.{i}"
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            tensors[f"{p}.self_attn.{proj}.weight"] = (HIDDEN, HIDDEN)
        tensors[f"{p}.mlp.gate_proj.weight"] = (32, HIDDEN)
        tensors[f"{p}.mlp.up_proj.weight"] = (32, HIDDEN)
        tensors[f"{p}.mlp.down_proj.weight"] = (HIDDEN, 32)
        tensors[f"{p}.input_layernorm.weight"] = (HIDDEN,)
        tensors[f"{p}.post_attention_layernorm.weight"] = (HIDDEN,)
    arrays = {n: rng.standard_normal(s).astype(np.float32)
              for n, s in tensors.items()}
    ckpt = tmp_path / name
    ckpt.mkdir(exist_ok=True)
    (ckpt / "model.safetensors").write_bytes(save(arrays))
    (ckpt / "config.json").write_text(json.dumps({
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "num_hidden_layers": LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "hidden_size": HIDDEN,
        "vocab_size": VOCAB,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-5,
    }))
    return str(ckpt)


def test_serve_transformer_not_supported(tmp_path):
    bundle_dir = _init(tmp_path, "serve-xfmr")
    seed_bundle(bundle_dir, _llama_ckpt(tmp_path, "ckpt-serve"),
                importer="test")
    manifest = load_bundle(bundle_dir)
    with _Running(bundle_dir) as srv:
        status, health = _get(srv, "/health")
        assert status == 200
        assert health["mind_kind"] == "transformer"
        assert health["arch_source"] == "checkpoint-config"
        assert health["arch"] == "llama"
        assert health["head"] == manifest["head"] == "base"
        assert health["bundle_id"] == manifest_id(manifest)
        assert health["inference_supported"] is False

        status, body = _post(srv, "/infer", {"input": [0.0] * 8})
    assert status == 501, f"transformer /infer must be 501, got {status}: {body}"
    assert "not supported" in body["error"]
    assert "transformer" in body["error"]
    assert "output" not in body, "a 501 must never carry fabricated output"


def test_serve_cli_parser():
    from si.cli import build_parser
    args = build_parser().parse_args(
        ["serve", "some.si", "--host", "127.0.0.1", "--port", "0"])
    assert args.cmd == "serve"
    assert args.host == "127.0.0.1"
    assert args.port == 0
