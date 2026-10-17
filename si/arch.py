"""Architecture metadata for ``si export``.

A GGUF that only carries ``si.*`` metadata is dishonestly generic: it
claims nothing about the model it holds, so no inference engine can
load it. When the bundle's base came from a real transformer
checkpoint, ``si export`` must write real ``<arch>.*`` KV entries.

Two honest sources, in order of trust:

1. ``checkpoint-config`` — ``si seed`` stored the checkpoint's
   ``config.json`` in the manifest at seed time. Values come straight
   from the checkpoint author. A value that is unknown is OMITTED,
   never invented.
2. ``tensor-shape-inference`` — the base was seeded from a real
   checkpoint but no ``config.json`` was captured. Conservative
   inference from tensor names and shapes: only well-known layer
   families with fully consistent shapes. Anything ambiguous raises
   ``ValueError`` — export fails closed rather than mislabeling the
   file. (Caveat: families are naming conventions, not fingerprints —
   e.g. mistral-style checkpoints share llama-style names. The
   ``tensor-shape-inference`` label is the honesty mechanism: it says
   the arch name was guessed from shapes, not read from the author.)

Demo bases (random seed data, not a trained model) keep
``general.architecture = "unknown"`` and are labeled
``si.arch_source = "demo-unknown"``.

``derive_arch`` is a pure function of the manifest and the base tensor
shapes, so exports stay deterministic: the same head always yields the
same bytes.
"""
from __future__ import annotations

import re

#: Fixed write order for the numeric arch KVs (deterministic output).
ARCH_FIELD_ORDER = (
    "block_count",
    "embedding_length",
    "context_length",
    "head_count",
    "head_count_kv",
    "layer_norm_rms_eps",
    "rope_dimension_count",
    "vocab_size",
)


def _blank_result(arch: str, arch_source: str) -> dict:
    return {"arch": arch, "arch_source": arch_source,
            **{k: None for k in ARCH_FIELD_ORDER}}


# -- source 1: checkpoint config.json ------------------------------------------
def _pos_int(cfg: dict, key: str):
    """Positive-int config value, or None when the key is absent.

    Fail closed on a present-but-garbage value: a corrupt stored config
    must not silently produce a half-labeled file.
    """
    v = cfg.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
        raise ValueError(
            f"stored checkpoint config: {key}={v!r} is not a positive int; "
            "refusing to derive architecture metadata from it"
        )
    return v


def _pos_float(cfg: dict, key: str):
    v = cfg.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
        raise ValueError(
            f"stored checkpoint config: {key}={v!r} is not a positive "
            "number; refusing to derive architecture metadata from it"
        )
    return float(v)


#: (arch field, config keys in priority order)
_INT_FIELDS = (
    ("block_count", ("num_hidden_layers", "num_layers", "n_layer")),
    ("embedding_length", ("hidden_size", "n_embd")),
    ("context_length", ("max_position_embeddings", "n_positions")),
    ("head_count", ("num_attention_heads", "n_head")),
    ("head_count_kv", ("num_key_value_heads", "n_head_kv")),
    ("vocab_size", ("vocab_size",)),
)


def _arch_name_from_config(cfg: dict) -> str:
    """Arch name from the config, or "unknown" when it carries none.

    Prefers ``model_type`` (e.g. "llama"); falls back to the
    ``architectures`` class name with the task suffix stripped
    ("LlamaForCausalLM" -> "llama"). Never guesses beyond what the
    config states.
    """
    model_type = cfg.get("model_type")
    if isinstance(model_type, str) and model_type.strip():
        return model_type.strip().lower()
    archs = cfg.get("architectures")
    if isinstance(archs, list) and archs and isinstance(archs[0], str):
        name = archs[0]
        for suffix in ("ForCausalLM", "ForSequenceClassification",
                       "ForTokenClassification", "ForMaskedLM", "Model"):
            if name.endswith(suffix) and len(name) > len(suffix):
                name = name[: -len(suffix)]
                break
        name = name.strip().lower()
        if name:
            return name
    return "unknown"


def _rope_dim_from_config(cfg: dict, hidden: int | None,
                          heads: int | None) -> int | None:
    """RoPE dimension count, derived exactly — never invented.

    ``head_dim`` when the config states it; otherwise
    ``hidden_size // num_attention_heads`` when that division is exact.
    """
    head_dim = _pos_int(cfg, "head_dim")
    if head_dim is not None:
        return head_dim
    if hidden and heads and hidden % heads == 0:
        return hidden // heads
    return None


def _arch_from_config(cfg: dict) -> dict:
    """Derive arch metadata from a stored checkpoint config.json dict."""
    if not isinstance(cfg, dict):
        raise ValueError(
            "stored checkpoint_config is not a JSON object; "
            "refusing to export"
        )
    res = _blank_result(_arch_name_from_config(cfg), "checkpoint-config")
    for field, keys in _INT_FIELDS:
        for key in keys:
            v = _pos_int(cfg, key)
            if v is not None:
                res[field] = v
                break
    # head_count_kv: omit when the config does not state it. Copying
    # head_count in would fabricate a GQA layout the author never
    # declared.
    eps = _pos_float(cfg, "rms_norm_eps")
    if eps is not None:
        res["layer_norm_rms_eps"] = eps
    res["rope_dimension_count"] = _rope_dim_from_config(
        cfg, res["embedding_length"], res["head_count"])
    return res


# -- source 2: tensor-shape inference --------------------------------------------
_LLAMA_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")
_GPT2_LAYER_RE = re.compile(r"^transformer\.h\.(\d+)\.")


def _layer_indices(shapes: dict, pattern: re.Pattern) -> list | None:
    """Sorted layer indices for a naming pattern, or None if no match.

    Fail closed on non-contiguous indices: gaps mean the layer count is
    genuinely ambiguous.
    """
    idx = {int(m.group(1)) for name in shapes
           for m in [pattern.match(name)] if m}
    if not idx:
        return None
    if idx != set(range(max(idx) + 1)):
        raise ValueError(
            f"non-contiguous layer indices {sorted(idx)}; refusing to "
            "infer block_count"
        )
    return sorted(idx)


def _shape2(shapes: dict, name: str) -> tuple:
    """2-D tensor shape for ``name``; ValueError when absent or not 2-D."""
    if name not in shapes:
        raise ValueError(
            f"cannot infer architecture: {name!r} missing from base tensors"
        )
    s = tuple(shapes[name])
    if len(s) != 2:
        raise ValueError(
            f"cannot infer architecture: {name!r} has shape {s}, expected 2-D"
        )
    return s


def _check_norm(shapes: dict, name: str, hidden: int) -> None:
    """Final norm, when present, must agree on the hidden size."""
    if name in shapes:
        s = tuple(shapes[name])
        if len(s) != 1 or s[0] != hidden:
            raise ValueError(
                f"ambiguous architecture: {name!r} has shape {s}, expected "
                f"({hidden},)"
            )


def _infer_llama(shapes: dict, indices: list) -> dict:
    """HF llama-style naming: model.layers.N.*, model.embed_tokens."""
    res = _blank_result("llama", "tensor-shape-inference")
    res["block_count"] = len(indices)
    # Every layer must carry at least one attention weight: a bare
    # "model.layers.N." prefix with no attention tensors is not a
    # transformer block we recognize.
    for i in indices:
        prefix = f"model.layers.{i}.self_attn."
        if not any(n.startswith(prefix) for n in shapes):
            raise ValueError(
                f"ambiguous architecture: model.layers.{i} has no "
                "self_attn.* tensors; refusing to label it a llama block"
            )
    vocab, hidden = _shape2(shapes, "model.embed_tokens.weight")
    res["vocab_size"] = vocab
    res["embedding_length"] = hidden
    if "lm_head.weight" in shapes:
        lh_vocab, lh_hidden = _shape2(shapes, "lm_head.weight")
        if (lh_vocab, lh_hidden) != (vocab, hidden):
            raise ValueError(
                "ambiguous architecture: lm_head.weight shape "
                f"({lh_vocab}, {lh_hidden}) disagrees with embed_tokens "
                f"({vocab}, {hidden})"
            )
    _check_norm(shapes, "model.norm.weight", hidden)
    # Not inferable from shapes alone: head counts, context length,
    # rope dims, norm eps. Omitted, never invented.
    return res


def _infer_gpt2(shapes: dict, indices: list) -> dict:
    """GPT-2-style naming: transformer.h.N.*, transformer.wte/wpe."""
    res = _blank_result("gpt2", "tensor-shape-inference")
    res["block_count"] = len(indices)
    for i in indices:
        prefix = f"transformer.h.{i}.attn."
        if not any(n.startswith(prefix) for n in shapes):
            raise ValueError(
                f"ambiguous architecture: transformer.h.{i} has no attn.* "
                "tensors; refusing to label it a gpt2 block"
            )
    vocab, hidden = _shape2(shapes, "transformer.wte.weight")
    res["vocab_size"] = vocab
    res["embedding_length"] = hidden
    if "transformer.wpe.weight" in shapes:
        ctx, wpe_hidden = _shape2(shapes, "transformer.wpe.weight")
        if wpe_hidden != hidden:
            raise ValueError(
                "ambiguous architecture: transformer.wpe.weight hidden dim "
                f"{wpe_hidden} disagrees with wte {hidden}"
            )
        res["context_length"] = ctx
    _check_norm(shapes, "transformer.ln_f.weight", hidden)
    return res


def _arch_from_tensors(shapes: dict) -> dict:
    """Conservative arch inference from base tensor names/shapes."""
    llama_idx = _layer_indices(shapes, _LLAMA_LAYER_RE)
    gpt2_idx = _layer_indices(shapes, _GPT2_LAYER_RE)
    families = [f for f, idx in (("llama", llama_idx), ("gpt2", gpt2_idx))
               if idx is not None]
    if not families:
        raise ValueError(
            "cannot infer architecture: base tensors match no recognized "
            "transformer layer naming (seeded checkpoint without "
            "config.json); refusing to label the export"
        )
    if len(families) > 1:
        raise ValueError(
            "ambiguous architecture: base tensors match multiple layer "
            f"families {families}; refusing to label the export"
        )
    if families[0] == "llama":
        return _infer_llama(shapes, llama_idx)
    return _infer_gpt2(shapes, gpt2_idx)


# -- entry point -------------------------------------------------------------------
def derive_arch(manifest: dict, shapes: dict) -> dict:
    """Derive GGUF architecture metadata for an export.

    ``manifest`` is the verified bundle manifest; ``shapes`` maps base
    tensor name -> shape tuple. Pure function: the same manifest and
    shapes always yield the same result, so exports stay deterministic.

    Returns {"arch", "arch_source", <ARCH_FIELD_ORDER keys>}; unknown
    values are None (omitted from the GGUF, never invented).

    Raises ValueError when inference is ambiguous or a stored config is
    corrupt — export fails closed rather than mislabeling the file.
    """
    base = manifest.get("base", {}) or {}
    cfg = base.get("checkpoint_config")
    if cfg is not None:
        return _arch_from_config(cfg)
    source = str(base.get("source", ""))
    if source.startswith("demo:") or base.get("arch_source") == "demo-unknown":
        return _blank_result("unknown", "demo-unknown")
    return _arch_from_tensors(shapes)


__all__ = ["ARCH_FIELD_ORDER", "derive_arch"]
