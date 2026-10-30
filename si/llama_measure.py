"""SI-40 (A-1): the minimal llama-family measurement harness.

Implements slice 1 of RFC SI-39 §5.3 — the A-1 feasibility
measurement the operator approved on 2026-10-29. This module
MEASURES one named real checkpoint
(``HuggingFaceTB/SmolLM2-360M-Instruct``, llama family,
361,821,120 BF16 parameters) on this VM, CPU only:

- **assembly**: build the full weight set from the checkpoint
  file at each assembly width of RFC SI-39 §3.4's table
  (BF16 as stored, float32, float32 base + candidate, and a
  float64-widened compute copy in the SI-15 discipline);
- **forward**: a minimal, self-contained llama decoder
  forward (pre-norm RMSNorm, RoPE, grouped-query attention,
  SwiGLU MLP, tied embeddings) so forward wall time at this
  scale is a measurement of a forward this project runs,
  not of a third-party serving stack;
- **verification**: the two costs the self-write loop would
  pay per cycle — the SI-8 by-reference ``load_bundle``
  streaming-hash verification, and a base + candidate
  assembly-and-compare pass.

What this is and is not (honest labels):

- It is a **measurement harness**, not an executor design
  (that is A-4, unapproved), not serving (nothing here is
  wired to ``si serve``), not a probe suite (A-3,
  unapproved), and not a gate path: a bundle seeded from
  this checkpoint still reports gate checks 3/4 "not
  evaluated" and escalates, exactly as before this slice.
- Every number the CLI emits is **measured** in the process
  that emits it (wall time via ``time.perf_counter``, peak
  memory via ``resource.getrusage`` ``ru_maxrss`` of the
  measuring subprocess). Nothing is derived except where a
  function says so (assembled byte counts are element
  counts x width, checked against the live tensors).
- The forward's numeric behaviour is **reported, not
  presumed**: the A-1 feasibility note
  (``docs/SI-40-a1-feasibility-measurement.md``) records its
  logits against a reference implementation's logits for
  the same token ids on the real checkpoint, with the
  measured max-abs difference. The unit tests validate the
  forward against an independent naive float64
  implementation on a tiny constructed llama checkpoint.
- Arithmetic discipline, stated plainly: weights are
  assembled in float32 (the dtype the SI-15 path's storage
  delivers) and the forward computes in float32. The
  miniature's float64-widened compute is NOT used here —
  §3.4 prices a float64 copy's residency (measured in the
  assembly phase) and SI-39 §5.3 leaves the float32/float64
  decision to A-4, decided against these measurements.
- No randomness at forward time: no sampling, no dropout.
  Repeat runs over the same weights and token ids are
  bit-identical (asserted by test on the tiny checkpoint).
- torch is imported lazily inside the functions that need
  it: the project's declared dependencies stay numpy +
  safetensors, torch is available in the measurement venv,
  and a caller without torch gets a clear refusal from the
  measuring functions rather than an import-time failure of
  the whole package.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np
from safetensors import safe_open

__all__ = [
    "ASSEMBLY_WIDTHS",
    "LlamaMeasureError",
    "assemble_weights",
    "assembled_bytes",
    "compare_weight_sets",
    "llama_forward",
    "load_llama_config",
    "measure_all",
    "peak_rss_bytes",
    "run_phase",
    "validate_llama_config",
]


class LlamaMeasureError(ValueError):
    """The harness refuses this input (config, weights, or
    token sequence). Subclasses ValueError, like the other
    fail-closed errors in this runtime."""


#: The assembly widths of RFC SI-39 SS3.4's table, in order:
#: bytes per element for each named assembly.
ASSEMBLY_WIDTHS = {
    "bf16": 2,        # as stored, one copy
    "float32": 4,     # float32 assembly, one copy
    "float32_pair": 4,  # float32, base + candidate (two copies)
    "float64": 8,     # float64-widened compute copy, one copy
}

_DTYPES = {}


def _torch():
    try:
        import torch
    except ImportError as e:  # pragma: no cover - env dependent
        raise LlamaMeasureError(
            "torch is required for the SI-40 measurement harness "
            "and is not importable in this environment; refusing "
            "rather than substituting a different arithmetic") from e
    return torch


def _dtype_map():
    if not _DTYPES:
        torch = _torch()
        _DTYPES.update({
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "float64": torch.float64,
        })
    return _DTYPES


def peak_rss_bytes() -> int:
    """Peak resident set size of THIS process so far, in bytes
    (``ru_maxrss`` is KiB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


# -- config -------------------------------------------------------------------


def load_llama_config(checkpoint_dir: str) -> dict:
    """Read and validate ``config.json`` beside the weights."""
    path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.isfile(path):
        raise LlamaMeasureError(f"no config.json in {checkpoint_dir!r}")
    with open(path, "rb") as f:
        config = json.loads(f.read().decode("utf-8"))
    return validate_llama_config(config)


def validate_llama_config(config) -> dict:
    """Return ``config`` iff it describes the llama-family
    configuration surface this harness implements: pre-norm
    RMSNorm decoder, RoPE (no scaling), GQA, SiLU MLP, no
    biases. Anything else is refused, not approximated."""
    if not isinstance(config, dict):
        raise LlamaMeasureError(f"config is not an object: {config!r}")
    if config.get("model_type") != "llama":
        raise LlamaMeasureError(
            f"model_type {config.get('model_type')!r} != 'llama' — "
            f"this harness measures the llama family only")
    problems = []
    if config.get("attention_bias", False):
        problems.append("attention_bias is true (harness assumes none)")
    if config.get("mlp_bias", False):
        problems.append("mlp_bias is true (harness assumes none)")
    if config.get("rope_scaling"):
        problems.append("rope_scaling is set (harness implements plain RoPE)")
    hidden = config.get("hidden_size")
    heads = config.get("num_attention_heads")
    if not hidden or not heads or hidden % heads:
        problems.append(
            f"hidden_size {hidden!r} not divisible by num_attention_heads "
            f"{heads!r}")
    for key in ("num_hidden_layers", "intermediate_size", "vocab_size",
                "num_key_value_heads", "rms_norm_eps", "rope_theta",
                "max_position_embeddings"):
        if key not in config:
            problems.append(f"missing key {key!r}")
    if problems:
        raise LlamaMeasureError(
            "llama config outside the harness's surface: " + "; ".join(problems))
    return config


# -- assembly -------------------------------------------------------------------


def checkpoint_tensor_names(checkpoint_path: str) -> list:
    """Tensor names in a single-file safetensors checkpoint."""
    with safe_open(checkpoint_path, framework="pt") as f:
        return sorted(f.keys())


def assemble_weights(checkpoint_path: str, width: str) -> dict:
    """Assemble the full weight set of a single-file checkpoint
    at one named width (a key of ``ASSEMBLY_WIDTHS``).

    BF16 tensors are read through torch (numpy has no bf16);
    conversion happens tensor by tensor so peak memory stays
    the assembled set plus one source tensor. Returns a dict
    name -> torch tensor in the width's dtype, on CPU."""
    torch = _torch()
    if width not in ASSEMBLY_WIDTHS:
        raise LlamaMeasureError(
            f"unknown assembly width {width!r}; expected one of "
            f"{sorted(ASSEMBLY_WIDTHS)}")
    dtype = _dtype_map()[width if width != "float32_pair" else "float32"]
    weights = {}
    with safe_open(checkpoint_path, framework="pt") as f:
        for name in sorted(f.keys()):
            weights[name] = f.get_tensor(name).to(dtype)
    return weights


def assembled_bytes(weights: dict) -> int:
    """Live byte count of an assembled weight set (element
    counts x dtype width, summed over the actual tensors)."""
    return int(sum(t.numel() * t.element_size() for t in weights.values()))


def count_parameters(weights: dict) -> int:
    return int(sum(t.numel() for t in weights.values()))


# -- forward -------------------------------------------------------------------


def _rms_norm(torch, x, weight, eps):
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight


def llama_forward(weights: dict, config: dict, token_ids) -> "object":
    """One full forward pass: logits, float32 ``[len, vocab]``.

    ``weights`` must be a float32 assembly (see
    ``assemble_weights``); the computation is float32
    throughout (see the module docstring for the discipline
    statement). Tied embeddings: when the checkpoint carries
    no ``lm_head.weight`` (the named checkpoint ties), the
    embedding matrix is the output projection."""
    torch = _torch()
    ids = [int(t) for t in token_ids]
    if not ids:
        raise LlamaMeasureError("empty token sequence — refusing")
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = hidden // heads
    eps = float(config["rms_norm_eps"])
    theta = float(config["rope_theta"])
    if kv_heads <= 0 or heads % kv_heads:
        raise LlamaMeasureError(
            f"num_attention_heads {heads} not a multiple of "
            f"num_key_value_heads {kv_heads}")
    embed = weights["model.embed_tokens.weight"]
    if embed.dtype != torch.float32:
        raise LlamaMeasureError(
            f"forward requires a float32 assembly; embed tensor is "
            f"{embed.dtype}")
    vocab = int(config["vocab_size"])
    if embed.shape != (vocab, hidden):
        raise LlamaMeasureError(
            f"embed shape {tuple(embed.shape)} != (vocab {vocab}, "
            f"hidden {hidden})")

    x = embed[torch.tensor(ids, dtype=torch.long)]  # [S, H]
    seq = len(ids)
    # RoPE tables for this sequence (plain RoPE, non-interleaved:
    # rotate_half convention, matching the llama reference).
    inv_freq = 1.0 / (theta ** (
        torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(seq, dtype=torch.float32), inv_freq)
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)  # [S, D]
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)

    def rope(t):  # t: [S, heads, D]
        half = head_dim // 2
        rotated = torch.cat([-t[..., half:], t[..., :half]], dim=-1)
        return t * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)

    causal = torch.full((seq, seq), float("-inf")).triu(1)
    for layer in range(int(config["num_hidden_layers"])):
        p = f"model.layers.{layer}"
        h = _rms_norm(torch, x, weights[p + ".input_layernorm.weight"], eps)
        q = (h @ weights[p + ".self_attn.q_proj.weight"].t())
        k = (h @ weights[p + ".self_attn.k_proj.weight"].t())
        v = (h @ weights[p + ".self_attn.v_proj.weight"].t())
        q = rope(q.view(seq, heads, head_dim))
        k = rope(k.view(seq, kv_heads, head_dim))
        v = v.view(seq, kv_heads, head_dim)
        # GQA: each KV head serves heads // kv_heads query heads.
        k = k.repeat_interleave(heads // kv_heads, dim=1)
        v = v.repeat_interleave(heads // kv_heads, dim=1)
        scores = (q.transpose(0, 1) @ k.transpose(0, 1).transpose(1, 2))
        scores = scores / (head_dim ** 0.5) + causal
        attn = torch.softmax(scores, dim=-1)
        ctx = (attn @ v.transpose(0, 1)).transpose(0, 1).reshape(seq, hidden)
        x = x + ctx @ weights[p + ".self_attn.o_proj.weight"].t()
        h = _rms_norm(torch, x,
                      weights[p + ".post_attention_layernorm.weight"], eps)
        gate = h @ weights[p + ".mlp.gate_proj.weight"].t()
        up = h @ weights[p + ".mlp.up_proj.weight"].t()
        x = x + (torch.nn.functional.silu(gate) * up) \
            @ weights[p + ".mlp.down_proj.weight"].t()
    x = _rms_norm(torch, x, weights["model.norm.weight"], eps)
    lm_head = weights.get("lm_head.weight", embed)
    return x @ lm_head.t()


# -- verification compare -------------------------------------------------------


def compare_weight_sets(base: dict, candidate: dict) -> dict:
    """The verify loop's compare step over two assembled weight
    sets: every tensor compared elementwise; returns the max
    absolute difference and the first differing tensor (or
    None). Refuses mismatched inventories rather than
    comparing a subset."""
    if sorted(base) != sorted(candidate):
        raise LlamaMeasureError(
            "weight-set inventories differ — refusing a partial compare")
    torch = _torch()
    max_abs = 0.0
    first_diff = None
    for name in sorted(base):
        if base[name].shape != candidate[name].shape:
            raise LlamaMeasureError(f"shape mismatch on {name}")
        diff = (base[name].float() - candidate[name].float()).abs().max()
        d = float(diff) if base[name].numel() else 0.0
        if d > 0.0 and first_diff is None:
            first_diff = name
        max_abs = max(max_abs, d)
    return {"max_abs_diff": max_abs, "first_differing_tensor": first_diff,
            "tensors_compared": len(base)}


# -- measurement phases (each run in a clean subprocess) ------------------------


def _weights_file(checkpoint_dir: str) -> str:
    path = os.path.join(checkpoint_dir, "model.safetensors")
    if not os.path.isfile(path):
        raise LlamaMeasureError(f"no model.safetensors in {checkpoint_dir!r}")
    return path


def run_phase(phase: str, checkpoint_dir: str, token_ids=None) -> dict:
    """Run one measurement phase IN THIS PROCESS and return its
    record. ``measure_all`` invokes each phase in a fresh
    subprocess so ``peak_rss_bytes`` is that phase's own peak,
    uncontaminated by the phases before it."""
    torch = _torch()
    torch.set_num_threads(os.cpu_count() or 1)
    ckpt = _weights_file(checkpoint_dir)
    config = load_llama_config(checkpoint_dir)
    rec = {"phase": phase}
    if phase.startswith("assemble:"):
        width = phase.split(":", 1)[1]
        t0 = time.perf_counter()
        weights = assemble_weights(ckpt, width)
        if width == "float32_pair":
            candidate = assemble_weights(ckpt, "float32")
            rec["assemblies_held"] = 2
            rec["assembled_bytes"] = assembled_bytes(weights) \
                + assembled_bytes(candidate)
            del candidate
        else:
            rec["assemblies_held"] = 1
            rec["assembled_bytes"] = assembled_bytes(weights)
        rec["wall_s"] = time.perf_counter() - t0
        rec["parameters"] = count_parameters(weights)
        del weights
    elif phase == "forward":
        ids = list(token_ids or [])
        if not ids:
            raise LlamaMeasureError("forward phase needs token ids")
        weights = assemble_weights(ckpt, "float32")
        t0 = time.perf_counter()
        logits = llama_forward(weights, config, ids)
        rec["wall_s"] = time.perf_counter() - t0
        rec["sequence_length"] = len(ids)
        rec["logits_shape"] = list(logits.shape)
        rec["logits_checksum"] = float(logits.double().sum())
        rec["last_position_argmax"] = int(logits[-1].argmax())
        del weights, logits
    elif phase == "verify_compare":
        base = assemble_weights(ckpt, "float32")
        candidate = assemble_weights(ckpt, "float32")
        t0 = time.perf_counter()
        rec["compare"] = compare_weight_sets(base, candidate)
        rec["wall_s"] = time.perf_counter() - t0
        rec["assembled_bytes"] = assembled_bytes(base) \
            + assembled_bytes(candidate)
        del base, candidate
    elif phase == "verify_bundle":
        raise LlamaMeasureError(
            "verify_bundle needs --bundle; use measure_all or the "
            "phase CLI")
    else:
        raise LlamaMeasureError(f"unknown phase {phase!r}")
    rec["peak_rss_bytes"] = peak_rss_bytes()
    return rec


def _phase_cli(argv) -> int:
    p = argparse.ArgumentParser(prog="si.llama_measure phase")
    p.add_argument("--phase", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bundle", default=None)
    p.add_argument("--tokens", default=None,
                   help="comma-separated token ids for the forward phase")
    args = p.parse_args(argv)
    if args.phase == "verify_bundle":
        if not args.bundle:
            raise SystemExit("verify_bundle phase requires --bundle")
        from si.bundle import load_bundle
        t0 = time.perf_counter()
        manifest = load_bundle(args.bundle)
        rec = {"phase": "verify_bundle",
               "wall_s": time.perf_counter() - t0,
               "stored_bytes": manifest["base"].get("stored_bytes"),
               "referenced_bytes": manifest["base"].get("referenced_bytes"),
               "tensor_count": manifest["base"].get("tensor_count"),
               "peak_rss_bytes": peak_rss_bytes()}
    else:
        tokens = ([int(t) for t in args.tokens.split(",") if t.strip()]
                  if args.tokens else None)
        rec = run_phase(args.phase, args.checkpoint, tokens)
    print(json.dumps(rec))
    return 0


def measure_all(checkpoint_dir: str, token_ids, bundle_dir=None,
                widths=("bf16", "float32", "float32_pair", "float64")) -> dict:
    """Run every phase in its own subprocess (clean peak-RSS
    accounting) and return the collected records."""
    env = dict(os.environ)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    phases = [f"assemble:{w}" for w in widths] + ["forward", "verify_compare"]
    if bundle_dir:
        phases.append("verify_bundle")
    out = {"checkpoint": checkpoint_dir, "bundle": bundle_dir,
           "token_ids": list(token_ids), "phases": {}}
    for phase in phases:
        cmd = [sys.executable, "-m", "si.llama_measure", "phase",
               "--phase", phase,
               "--checkpoint", checkpoint_dir,
               "--tokens", ",".join(str(t) for t in token_ids)]
        if bundle_dir:
            cmd += ["--bundle", bundle_dir]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise LlamaMeasureError(
                f"phase {phase} failed ({proc.returncode}): "
                f"{proc.stderr.strip()[-2000:]}")
        rec = json.loads(proc.stdout.strip().splitlines()[-1])
        rec["subprocess_wall_s"] = time.perf_counter() - t0
        out["phases"][phase] = rec
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "phase":
        return _phase_cli(argv[1:])
    p = argparse.ArgumentParser(
        prog="si.llama_measure",
        description="SI-40 (A-1) llama-family feasibility measurement")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bundle", default=None)
    p.add_argument("--tokens", required=True,
                   help="comma-separated token ids for the forward phase")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    tokens = [int(t) for t in args.tokens.split(",") if t.strip()]
    result = measure_all(args.checkpoint, tokens, args.bundle)
    text = json.dumps(result, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
