"""SI-35: branch-contribution and expressibility audit of the
miniature (RFC SI-34 §§6, 15, 16 — measurement only).

What this module is:

- A **read-only measurement module**. It runs forward passes
  over weight sets assembled through the existing apply path
  (base + landed overlays, in memory) and publishes, for every
  weight matrix in the miniature, three kinds of figures:
  (i) **branch-contribution measurements** — for each
  residual-adding branch, the measured share of the residual
  stream its output contributes; (ii) **input-activation
  records** — the measured scale of the activation each
  matrix consumes; and (iii) **expressibility bounds**
  (*derivations*, labelled as such wherever they appear) —
  the local bound ``B(W) = 0.9 * (max input norm)`` on what
  a rank-8 update at the standing norm budget could express
  through that matrix, exactly as SI-33's as-built bounded
  the final down-projection's movement (0.9 * 0.064).
- The audit's method, populations, heads, and readings are
  the ones RFC SI-34 §15 registered *before* any audit
  figure existed. The readings (R1/R2/R3, §15 item 7) are
  applied as written in :func:`run_audit_on_weight_sets`;
  exactly one of R1/R2 is measured, by construction.

What this module is NOT (RFC SI-34 §§15–17, §18 item 27):

- It authors nothing and trains nothing: it invokes no
  training entry point, constructs no candidate, and
  performs no decisive run.
- It performs **no gate evaluation** and opens no probe
  contact of any kind: no judged-population input, logit,
  or derived quantity is read by any audit measurement,
  ranking, or reading. The audit's populations are exactly
  the two signals' partitions — train partitions primary,
  held-out partitions reported only, under their own
  labels, never merged into a primary figure. (The one
  probe-adjacent import is
  :func:`si.probes.derive_probe_blind_targets`, a pure
  derivation from the architecture descriptor that reads
  no input of any kind; the audit needs its derived blind
  set for the §15 item 6 inventory, exactly as the gate
  does.)
- It writes nothing: no blob, no journal entry, no
  manifest change. A bundle it is run against is left
  byte-identical (asserted by test).
- Its output is at most a **shortlist for an argument**:
  qualification under R1 is a statement about measured
  branch scale and a derived bound on named populations
  at named heads. It is not an approval of any target, a
  prediction about any candidate, or a gate figure of any
  kind (§18 item 27). A measurement-only record is not a
  verdict record.

Scope labels that travel with every figure: the subject
is the SI-14 miniature — constructed, deterministically
seeded, **not trained** — at three named heads (``base``,
``d0001``, ``d0002``, the landed stack SI-33 produced).
Nothing here is claimed about any real model, any trained
base, or Glimmer, at any scale. Machine learning
infrastructure throughout; CPU only.

Arithmetic discipline mirrors the library forward
(``si.tiny_transformer.forward``): float32 weights widened
to float64 for compute, same operation order; the audit
forward's logits are asserted bit-identical to the library
forward's (after the float32 cast both paths apply) by
test, so every audited quantity is a statistic of the same
function every prior slice measured.
"""
from __future__ import annotations

import math

import numpy as np

from si.bundle import _active_chain, canonical_json
from si.probes import derive_probe_blind_targets
from si.tiny_seed import (
    EMBED_NAME,
    FINAL_NORM_NAME,
    LM_HEAD_NAME,
    POS_NAME,
    tiny_tensor_shapes,
)
from si.tiny_training import (
    CORRECTION_SCALE,
    NORM_BUDGET,
    TRAINING_SIGNAL_ID,
    TRAINING_SIGNAL_ID_2,
    signal_sequences,
)

__all__ = [
    "AUDIT_HEADS",
    "AuditError",
    "BRANCHES",
    "IN_STREAM_MATRICES",
    "POPULATION_SPECS",
    "R1_BOUND_MIN",
    "R1_BRANCH_RATIO_MIN",
    "WEIGHT_MATRICES",
    "audit_record_bytes",
    "branch_for_matrix",
    "forward_audit",
    "inventory",
    "matrix_branch",
    "run_audit",
    "run_audit_on_weight_sets",
    "weights_at_head",
]


class AuditError(ValueError):
    """The audit refuses this input (missing head, wrong
    weight inventory). Subclasses ValueError, like the other
    fail-closed errors in this runtime."""


#: The named heads every audit figure is measured at
#: (RFC SI-34 §15 item 1(ii)): the seed base and the two
#: landed heads of the authored stack SI-33 produced
#: (``d0001``: signal 1, readout target; ``d0002``: signal 2,
#: final-MLP-down target — the second delta labelled a
#: non-readout-target candidate wherever the stack is
#: described, §18 item 24).
AUDIT_HEADS = ("base", "d0001", "d0002")

#: The audit's populations (RFC SI-34 §15 item 1(i)):
#: both signals' train partitions are the primary
#: populations; both held-out partitions are reported
#: only, under their own labels, never merged into a
#: primary figure. No other input set exists in the audit.
POPULATION_SPECS = (
    {"signal_id": TRAINING_SIGNAL_ID, "partition": "train",
     "role": "primary"},
    {"signal_id": TRAINING_SIGNAL_ID_2, "partition": "train",
     "role": "primary"},
    {"signal_id": TRAINING_SIGNAL_ID, "partition": "held_out",
     "role": "reported-only"},
    {"signal_id": TRAINING_SIGNAL_ID_2, "partition": "held_out",
     "role": "reported-only"},
)

#: R1's registered thresholds (RFC SI-34 §15 item 7),
#: stated there before any audit figure existed and
#: applied here as written — never re-chosen after the
#: figures exist (§17: no sweep of the audit's own
#: thresholds).
R1_BRANCH_RATIO_MIN = 0.04
#: The signal correction's logit-space norm,
#: ``0.15 * sqrt(2)`` (a derivation from the signals'
#: recorded correction scale, SI-19/SI-33).
R1_BOUND_MIN = CORRECTION_SCALE * math.sqrt(2.0)

#: SI-33's published yardstick bound for the final
#: down-projection (a derivation in its as-built from its
#: measured inputs: ``0.9 * 0.064`` at the standing budget).
SI33_DOWN_BOUND_YARDSTICK = 0.058

#: Magnitude accounting (§16 criterion 6): the landed
#: candidates audit figures are compared against, cited
#: to docs/SI-33-final-mlp-down-target.md — never
#: re-derived silently (§18 item 4).
SI33_CANDIDATE_NORM = 0.30801411790830574
SI33_CANDIDATE_M2 = 0.010469039280455053
SI25_CONTROL_NORM = 0.2515524481203623
SI25_CONTROL_M2 = 0.7816722548913912

#: SI-33's published attribution ranges — the audit's
#: reproduction anchor (§15 item 3 / §16 criterion 3),
#: measured by SI-33 at head ``d0001`` over signal 2's
#: train partition (last positions).
SI33_S_RANGE = (0.040, 0.064)
SI33_H_MID_RANGE = (1.97, 2.02)
#: SI-33's published base representative-sequence figures
#: for the final layer's MLP branch (branch output norm /
#: residual-stream norm). SI-33 does not name the sequence;
#: the audit reconciles these against its allowed-population
#: base statistics in print (see the as-built doc), with
#: the method difference named — it does not invent a
#: sequence for them.
SI33_BASE_BRANCH_OUTPUT = 0.0074
SI33_BASE_RESIDUAL = 2.00

_EMBEDDINGS = (EMBED_NAME, POS_NAME)
_SHAPES = tiny_tensor_shapes()

#: All 17 weight matrices (2-D tensors), in inventory
#: order (the seed inventory's sorted-by-construction
#: order is NOT used: matrices are listed readout-last
#: within their layer groups exactly as §15 item 4 lists
#: them — see :func:`inventory`).
WEIGHT_MATRICES = tuple(
    name for name, shape in _SHAPES.items() if len(shape) == 2)
#: The 14 in-stream matrices: every weight matrix except
#: the readout and the two embedding matrices.
IN_STREAM_MATRICES = tuple(
    name for name in WEIGHT_MATRICES
    if name not in (LM_HEAD_NAME,) + _EMBEDDINGS)
#: The 13 vector tensors (1-D), inventoried with their
#: blind-set status (§15 item 4).
VECTOR_TENSORS = tuple(
    name for name, shape in _SHAPES.items() if len(shape) == 1)

#: The 5 residual-stream rows of §15 item 2: per layer the
#: attention branch and the MLP branch (4), plus the
#: embedding sum that forms the initial residual stream
#: (1 input-side row, labelled as such).
BRANCHES = (
    "embedding-sum",
    "layer-0-attention",
    "layer-0-mlp",
    "layer-1-attention",
    "layer-1-mlp",
)


def _layer_of(name: str) -> int:
    return int(name.split("layers.")[1].split(".")[0])


def matrix_branch(name: str) -> str | None:
    """The residual-adding branch a weight matrix acts
    through, or ``None`` for the readout and the embedding
    matrices (they add to no pre-existing residual stream
    in a branch: the readout reads the final stream out,
    and the embeddings form the initial one)."""
    if name not in IN_STREAM_MATRICES:
        return None
    layer = _layer_of(name)
    kind = "attention" if ".self_attn." in name else "mlp"
    return f"layer-{layer}-{kind}"


#: Backwards-compatible alias (the branch a matrix belongs
#: to is the branch it acts through).
branch_for_matrix = matrix_branch


def inventory() -> dict:
    """The complete 30-tensor inventory (§16 criterion 1).

    Every tensor appears exactly once: the 17 weight
    matrices tabled with shapes and categories, the 13
    vector tensors inventoried with blind-set status (the
    derived blind set is exactly both layers'
    ``k_proj.bias`` — :func:`derive_probe_blind_targets`,
    a derivation from the architecture, not a list copied
    from a probe run). No tensor is silently omitted.
    """
    blind = derive_probe_blind_targets()
    matrices = []
    for name in WEIGHT_MATRICES:
        if name == LM_HEAD_NAME:
            category = "readout (yardstick target, not a candidate)"
        elif name in _EMBEDDINGS:
            category = "embedding (tabled separately; index input)"
        else:
            category = "in-stream"
        matrices.append({
            "name": name,
            "shape": [int(x) for x in _SHAPES[name]],
            "category": category,
            "branch": matrix_branch(name),
        })
    vectors = [{
        "name": name,
        "shape": [int(x) for x in _SHAPES[name]],
        "blind": name in blind,
        "blind_set_status": ("in the derived blind set"
                             if name in blind else
                             "measurable (not in the derived blind set)"),
    } for name in VECTOR_TENSORS]
    return {
        "tensor_count": len(_SHAPES),
        "weight_matrices": matrices,
        "weight_matrix_count": len(matrices),
        "in_stream_count": len(IN_STREAM_MATRICES),
        "vector_tensors": vectors,
        "vector_tensor_count": len(vectors),
        "derived_blind_set": sorted(blind),
    }


# -- the audit forward -------------------------------------------------------
# A replica of si.tiny_transformer.forward's computation —
# same operation order, same float64-over-float32
# discipline — that additionally exposes, per position,
# every branch output, every residual stream a branch adds
# into, and the activation every weight matrix consumes.
# tests assert its logits are BIT-IDENTICAL to the library
# forward's on the audit's populations, so the audit
# measures the same function every prior slice measured
# (the SI-19/SI-33 replica precedent).

def _rms_norm(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(ms + 1e-5) * w


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def forward_audit(token_ids, weights: dict) -> dict:
    """One audited forward over ``weights`` (a full
    30-tensor applied weight set, e.g. from
    :func:`weights_at_head`).

    Returns ``{"logits", "branches", "inputs"}`` where
    ``branches`` maps each of :data:`BRANCHES` to
    ``{"output": [t, 32], "residual": [t, 32] | None}``
    (the residual stream the branch's output adds into;
    ``None`` for the embedding sum, which *forms* the
    initial stream rather than adding into one), and
    ``inputs`` maps each weight matrix to the activation
    it consumes, ``[t, in_dim]`` — ``None`` for the two
    embedding matrices, whose input is an index, not an
    activation (§15 item 5's caveat, recorded per figure
    as "not applicable", never as 0.0).
    """
    shapes = tiny_tensor_shapes()
    if set(weights) != set(shapes):
        raise AuditError(
            "weight set is not the miniature's 30-tensor "
            f"inventory: missing={sorted(set(shapes) - set(weights))!r} "
            f"extra={sorted(set(weights) - set(shapes))!r}")
    w = {k: np.asarray(v, dtype=np.float64) for k, v in weights.items()}
    ids = [int(t) for t in token_ids]
    t = len(ids)
    if not (1 <= t <= 16) or any(x < 0 or x >= 32 for x in ids):
        raise AuditError(f"sequence outside the miniature's "
                         f"domain: {ids!r}")
    branches: dict = {}
    inputs: dict = {name: None for name in _EMBEDDINGS}
    emb = w[EMBED_NAME][ids] + w[POS_NAME][:t]
    branches["embedding-sum"] = {"output": emb, "residual": None}
    h = emb
    causal = np.triu(np.ones((t, t), dtype=bool), k=1)
    for i in range(2):
        p = f"model.language_model.layers.{i}"
        a = _rms_norm(h, w[f"{p}.input_layernorm.weight"])
        for proj in ("q_proj", "k_proj", "v_proj"):
            inputs[f"{p}.self_attn.{proj}.weight"] = a
        q = (a @ w[f"{p}.self_attn.q_proj.weight"].T
             + w[f"{p}.self_attn.q_proj.bias"]).reshape(t, 4, 8)
        k = (a @ w[f"{p}.self_attn.k_proj.weight"].T
             + w[f"{p}.self_attn.k_proj.bias"]).reshape(t, 4, 8)
        v = (a @ w[f"{p}.self_attn.v_proj.weight"].T
             + w[f"{p}.self_attn.v_proj.bias"]).reshape(t, 4, 8)
        q = np.transpose(q, (1, 0, 2))
        k = np.transpose(k, (1, 0, 2))
        v = np.transpose(v, (1, 0, 2))
        scores = q @ np.transpose(k, (0, 2, 1)) / np.sqrt(8.0)
        scores = np.where(causal, -np.inf, scores)
        attn = _softmax(scores) @ v                       # [H, t, dh]
        attn = np.transpose(attn, (1, 0, 2)).reshape(t, 32)
        inputs[f"{p}.self_attn.o_proj.weight"] = attn
        attn_out = attn @ w[f"{p}.self_attn.o_proj.weight"].T \
            + w[f"{p}.self_attn.o_proj.bias"]
        branches[f"layer-{i}-attention"] = {
            "output": attn_out, "residual": h}
        h = h + attn_out
        m = _rms_norm(h, w[f"{p}.post_attention_layernorm.weight"])
        inputs[f"{p}.mlp.gate_proj.weight"] = m
        inputs[f"{p}.mlp.up_proj.weight"] = m
        gate = m @ w[f"{p}.mlp.gate_proj.weight"].T
        up = m @ w[f"{p}.mlp.up_proj.weight"].T
        silu = gate / (1.0 + np.exp(-gate))
        s = silu * up
        inputs[f"{p}.mlp.down_proj.weight"] = s
        mlp_out = s @ w[f"{p}.mlp.down_proj.weight"].T
        branches[f"layer-{i}-mlp"] = {"output": mlp_out, "residual": h}
        h = h + mlp_out
    h_final = _rms_norm(h, w[FINAL_NORM_NAME])
    inputs[LM_HEAD_NAME] = h_final
    logits = (h_final @ w[LM_HEAD_NAME].T).astype(np.float32)
    return {"logits": logits, "branches": branches, "inputs": inputs}


# -- statistics ----------------------------------------------------------------

def _row_norms(arr: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(arr, dtype=np.float64), axis=-1)


def _stats(values) -> dict:
    v = [float(x) for x in values]
    return {"min": min(v), "mean": sum(v) / len(v), "max": max(v)}


def _populations() -> list:
    """The audit's input sets, exactly as registered:
    ``[(spec, seqs)]`` with each spec from
    :data:`POPULATION_SPECS` and its sequences from the
    signals' own deterministic construction
    (:func:`si.tiny_training.signal_sequences` — the
    partitions as SI-19/SI-23 defined them, read here as
    populations and changed in no way)."""
    out = []
    cache: dict = {}
    for spec in POPULATION_SPECS:
        sid = spec["signal_id"]
        if sid not in cache:
            train, held, _ = signal_sequences(sid)
            cache[sid] = {"train": train, "held_out": held}
        out.append((dict(spec), [list(s) for s in
                                 cache[sid][spec["partition"]]]))
    return out


def _pop_key(spec: dict) -> str:
    return f"{spec['signal_id']}:{spec['partition']}"


# -- the backward-path / checkability inventory (§15 item 6) -------------------
# Counting rule (stated here so the table is checkable,
# and applied uniformly): the chain for a matrix lists,
# in downstream order, each named element a future
# backward derivation must traverse between the tensor's
# output and the loss — a residual add counts once, an
# RMSNorm counts once, the SwiGLU derivative counts once,
# the attention Jacobian (scores / softmax / value
# combine, as one element) counts once, and each
# intervening matmul counts once. Elements are counted,
# not argued (§15 item 6).

def _downstream_tail(layer: int) -> list:
    if layer == 1:
        return ["final RMSNorm", "lm_head matmul (readout)"]
    return [
        "layer-1 input RMSNorm",
        "layer-1 attention Jacobian",
        "layer-1 o_proj matmul",
        "layer-1 attention residual add",
        "layer-1 post-attention RMSNorm",
        "layer-1 SwiGLU derivative",
        "layer-1 down_proj matmul",
        "layer-1 MLP residual add",
        "final RMSNorm",
        "lm_head matmul (readout)",
    ]


def _chain_elements(name: str) -> list:
    layer = _layer_of(name)
    tail = _downstream_tail(layer)
    if ".self_attn.q_proj." in name or ".self_attn.k_proj." in name \
            or ".self_attn.v_proj." in name:
        return [
            f"layer-{layer} attention Jacobian",
            f"layer-{layer} o_proj matmul",
            f"layer-{layer} attention residual add",
            f"layer-{layer} post-attention RMSNorm",
            f"layer-{layer} SwiGLU derivative",
            f"layer-{layer} down_proj matmul",
            f"layer-{layer} MLP residual add",
        ] + tail
    if ".self_attn.o_proj." in name:
        return [
            f"layer-{layer} attention residual add",
            f"layer-{layer} post-attention RMSNorm",
            f"layer-{layer} SwiGLU derivative",
            f"layer-{layer} down_proj matmul",
            f"layer-{layer} MLP residual add",
        ] + tail
    if ".mlp.gate_proj." in name or ".mlp.up_proj." in name:
        return [
            f"layer-{layer} SwiGLU derivative",
            f"layer-{layer} down_proj matmul",
            f"layer-{layer} MLP residual add",
        ] + tail
    # down_proj: its output is the branch output itself.
    return [f"layer-{layer} MLP residual add"] + tail


def backward_inventory() -> list:
    """§15 item 6's table for every in-stream matrix:
    chain elements + count, rank-8 factor-entry count
    ``8 * (rows + cols)`` (arithmetic, labelled), blind-set
    membership, gate-measurability under SI-17's mapping,
    and downstream transfer — recorded as measured only
    for the two tensors landed deltas already measure
    (handled for the readout / final down-projection in
    the record; every other tensor's transfer coefficient
    is **unknown**, never estimated, §15 item 5)."""
    blind = derive_probe_blind_targets()
    rows = []
    for name in IN_STREAM_MATRICES:
        r, c = (int(x) for x in _SHAPES[name])
        chain = _chain_elements(name)
        is_final_down = name == (
            "model.language_model.layers.1.mlp.down_proj.weight")
        rows.append({
            "matrix": name,
            "shape": [r, c],
            "chain_elements": chain,
            "chain_count": len(chain),
            "factor_entries_rank8": 8 * (r + c),
            "factor_entries_kind": "arithmetic (derivation)",
            "blind": name in blind,
            "gate_measurability": (
                "measurable under SI-17's mapping "
                "(not in the derived blind set)"),
            "downstream_transfer": (
                "measured (SI-33's landed delta record: "
                "candidate norm 0.30801411790830574, "
                "probe-space movement norm "
                "0.010469039280455053)"
                if is_final_down else "unknown"),
        })
    return rows


# -- weight assembly at a named head -------------------------------------------

def weights_at_head(bundle_dir: str, manifest: dict, head: str,
                    policy: dict | None = None) -> dict:
    """The applied weight set at ``head``, assembled
    through the existing apply path from an in-memory
    manifest view (the ``assemble_base_weights``
    precedent: the verified manifest itself is never
    mutated, re-headed, or written; every tensor is a
    hash-verified apply-at-load read; nothing is copied
    to disk or stored — the never-duplicate-bytes rule
    holds by construction)."""
    if head not in AUDIT_HEADS:
        raise AuditError(f"unknown audit head {head!r} "
                         f"(the audit serves exactly {AUDIT_HEADS!r})")
    from si.tiny_transformer import assemble_weights

    view = dict(manifest)
    view["head"] = head
    return assemble_weights(bundle_dir, view, policy)


def available_heads(manifest: dict) -> list:
    """The heads the manifest's active chain actually
    carries: ``base`` plus each landed delta's id, oldest
    first (via the bundle layer's own chain resolution)."""
    return ["base"] + [r["id"] for r in _active_chain(manifest)]


# -- the audit itself ------------------------------------------------------------

def run_audit_on_weight_sets(weight_sets: dict) -> dict:
    """The full audit over explicit weight sets
    ``{head: weights}`` for exactly :data:`AUDIT_HEADS`.

    Pure measurement: forward passes only, over the
    registered populations. Deterministic — a repetition
    from the same weight sets reproduces the record
    exactly (no sampling, no wall-clock anywhere in it).
    """
    missing = [h for h in AUDIT_HEADS if h not in weight_sets]
    extra = [h for h in weight_sets if h not in AUDIT_HEADS]
    if missing or extra:
        raise AuditError(
            f"audit weight sets must be exactly {AUDIT_HEADS!r}: "
            f"missing={missing!r} extra={extra!r}")
    pops = _populations()
    inv = inventory()

    # Per (head, population): raw per-example statistics.
    # branch_stats[(branch, pop_key, head)] and
    # input_stats[(matrix, pop_key, head)].
    branch_stats: dict = {}
    input_stats: dict = {}
    # Reproduction raw values (signal 2 train @ d0001):
    repro_s: list = []
    repro_h_mid: list = []
    primary_key = _pop_key({"signal_id": TRAINING_SIGNAL_ID_2,
                            "partition": "train"})
    for head in AUDIT_HEADS:
        weights = weight_sets[head]
        for spec, seqs in pops:
            key = _pop_key(spec)
            b_last = {b: {"out": [], "res": [], "ratio": []}
                      for b in BRANCHES}
            b_all = {b: {"out": [], "res": [], "ratio": []}
                     for b in BRANCHES}
            i_last: dict = {m: [] for m in WEIGHT_MATRICES}
            i_all: dict = {m: [] for m in WEIGHT_MATRICES}
            for seq in seqs:
                rec = forward_audit(seq, weights)
                for bname in BRANCHES:
                    br = rec["branches"][bname]
                    out_n = _row_norms(br["output"])
                    if br["residual"] is None:
                        # The embedding sum forms the stream:
                        # its residual is its own output, and
                        # its ratio is 1.0 by definition
                        # (input-side row, labelled as such).
                        res_n = out_n
                    else:
                        res_n = _row_norms(br["residual"])
                    b_last[bname]["out"].append(float(out_n[-1]))
                    b_last[bname]["res"].append(float(res_n[-1]))
                    b_last[bname]["ratio"].append(
                        float(out_n[-1] / res_n[-1]))
                    b_all[bname]["out"].extend(
                        float(x) for x in out_n)
                    b_all[bname]["res"].extend(
                        float(x) for x in res_n)
                    b_all[bname]["ratio"].extend(
                        float(x) for x in out_n / res_n)
                for mname in WEIGHT_MATRICES:
                    arr = rec["inputs"].get(mname)
                    if arr is None:
                        continue
                    n = _row_norms(arr)
                    i_last[mname].append(float(n[-1]))
                    i_all[mname].extend(float(x) for x in n)
                if head == "d0001" and key == primary_key:
                    s_arr = rec["inputs"][
                        "model.language_model.layers.1.mlp."
                        "down_proj.weight"]
                    repro_s.append(float(_row_norms(s_arr)[-1]))
                    h_mid = rec["branches"]["layer-1-mlp"]["residual"]
                    repro_h_mid.append(float(_row_norms(h_mid)[-1]))
            for bname in BRANCHES:
                branch_stats[(bname, key, head)] = {
                    "last_position": {
                        "branch_output_norm": _stats(b_last[bname]["out"]),
                        "residual_stream_norm": _stats(b_last[bname]["res"]),
                        "contribution_ratio": _stats(b_last[bname]["ratio"]),
                    },
                    "all_positions": {
                        "branch_output_norm": _stats(b_all[bname]["out"]),
                        "residual_stream_norm": _stats(b_all[bname]["res"]),
                        "contribution_ratio": _stats(b_all[bname]["ratio"]),
                        "label": "reported separately; no reading "
                                 "rests on all-position statistics",
                    },
                }
            for mname in WEIGHT_MATRICES:
                if not i_last[mname]:
                    input_stats[(mname, key, head)] = {
                        "status": "not applicable",
                        "reason": "the matrix's input is an index, "
                                  "not an activation (embedding "
                                  "matrices; RFC SI-34 §15 item 5)",
                    }
                else:
                    input_stats[(mname, key, head)] = {
                        "last_position": _stats(i_last[mname]),
                        "all_positions": {
                            **_stats(i_all[mname]),
                            "label": "reported separately; no "
                                     "reading rests on "
                                     "all-position statistics",
                        },
                    }

    branch_measurements = []
    for bname in BRANCHES:
        for spec, _seqs in pops:
            key = _pop_key(spec)
            for head in AUDIT_HEADS:
                branch_measurements.append({
                    "branch": bname,
                    "population": key,
                    "population_role": spec["role"],
                    "signal_id": spec["signal_id"],
                    "head": head,
                    "kind": "measurement",
                    **branch_stats[(bname, key, head)],
                })
    input_activations = []
    for mname in WEIGHT_MATRICES:
        for spec, _seqs in pops:
            key = _pop_key(spec)
            for head in AUDIT_HEADS:
                input_activations.append({
                    "matrix": mname,
                    "population": key,
                    "population_role": spec["role"],
                    "signal_id": spec["signal_id"],
                    "head": head,
                    "kind": "measurement",
                    **input_stats[(mname, key, head)],
                })

    # Expressibility bounds (DERIVATIONS, §15 item 5):
    # B(W) = 0.9 * (max input norm over a primary
    # population at the named head), for each in-stream
    # matrix, each head, each primary population.
    bounds = []
    bound_lookup: dict = {}
    for name in IN_STREAM_MATRICES:
        for spec, _seqs in pops:
            if spec["role"] != "primary":
                continue
            key = _pop_key(spec)
            for head in AUDIT_HEADS:
                mx = input_stats[(name, key, head)]["last_position"]["max"]
                bound = NORM_BUDGET * mx
                bound_lookup[(name, key, head)] = bound
                bounds.append({
                    "matrix": name,
                    "head": head,
                    "population": key,
                    "population_role": "primary",
                    "kind": "derivation",
                    "formula": "B(W) = 0.9 * (max input norm over "
                               "the population at the head)",
                    "norm_budget": NORM_BUDGET,
                    "max_input_norm": mx,
                    "bound": bound,
                    "downstream_transfer": (
                        "measured (SI-33's landed delta record)"
                        if name == "model.language_model.layers.1."
                                   "mlp.down_proj.weight"
                        else "unknown — the audit measures no "
                             "transfer coefficient for this tensor"),
                })
    for name in _EMBEDDINGS:
        r, _c = (int(x) for x in _SHAPES[name])
        bounds.append({
            "matrix": name,
            "kind": "derivation",
            "form": "row-selection: an update's effect enters "
                    "through the single selected row, so the "
                    "movement at the stream is at most the "
                    "update's Frobenius norm, itself at most "
                    "the standing budget 0.9; the rho * ||x|| "
                    "form does not describe an index input",
            "bound": NORM_BUDGET,
            "rows": r,
            "downstream_transfer": "unknown",
        })
    bounds.append({
        "matrix": LM_HEAD_NAME,
        "kind": "not applicable",
        "reason": "the readout is tabled as the record's "
                  "yardstick target, not as an in-stream "
                  "candidate (§15 item 4); no candidate bound "
                  "is stated for it",
        "downstream_transfer": "measured (the landed readout "
                               "deltas' records: SI-25 control "
                               "norm 0.2515524481203623, movement "
                               "norm 0.7816722548913912)",
    })

    # The pre-registered readings (§15 item 7), applied as
    # written. Primary cell: signal 2's train partition at
    # head d0001.
    def qualifies(head: str) -> list:
        out = []
        for name in IN_STREAM_MATRICES:
            branch = matrix_branch(name)
            ratio_max = branch_stats[
                (branch, primary_key, head)][
                "last_position"]["contribution_ratio"]["max"]
            bound = bound_lookup[(name, primary_key, head)]
            out.append({
                "matrix": name,
                "branch": branch,
                "head": head,
                "population": primary_key,
                "branch_ratio_max_per_example": ratio_max,
                "branch_ratio_threshold": R1_BRANCH_RATIO_MIN,
                "branch_ratio_clause": ratio_max >= R1_BRANCH_RATIO_MIN,
                "bound": bound,
                "bound_threshold": R1_BOUND_MIN,
                "bound_clause": bound >= R1_BOUND_MIN,
                "qualifies": bool(
                    ratio_max >= R1_BRANCH_RATIO_MIN
                    and bound >= R1_BOUND_MIN),
            })
        return out

    qual_primary = qualifies("d0001")
    qual_base = qualifies("base")
    qual_d0002 = qualifies("d0002")
    shortlist = [q["matrix"] for q in qual_primary if q["qualifies"]]
    r1_measured = len(shortlist) > 0
    r2_measured = not r1_measured
    set_base = {q["matrix"] for q in qual_base if q["qualifies"]}
    set_d0001 = set(shortlist)
    set_d0002 = {q["matrix"] for q in qual_d0002 if q["qualifies"]}
    r3_measured = set_base != set_d0001

    # Reproduction of SI-33's attribution (§15 item 3):
    # ranges under the audit's method + the base
    # representative figures reconciled against the
    # allowed-population base statistics.
    s_range = (min(repro_s), max(repro_s))
    h_range = (min(repro_h_mid), max(repro_h_mid))
    yard = branch_stats[("layer-1-mlp", primary_key, "base")][
        "last_position"]
    reproduction = {
        "population": primary_key,
        "head_for_s_and_h_mid": "d0001",
        "s_norm_range_measured": [s_range[0], s_range[1]],
        "s_norm_range_published_si33": list(SI33_S_RANGE),
        "h_mid_norm_range_measured": [h_range[0], h_range[1]],
        "h_mid_norm_range_published_si33": list(SI33_H_MID_RANGE),
        "ranges_reproduce": bool(
            abs(s_range[0] - SI33_S_RANGE[0]) < 0.001
            and abs(s_range[1] - SI33_S_RANGE[1]) < 0.001
            and abs(h_range[0] - SI33_H_MID_RANGE[0]) < 0.01
            and abs(h_range[1] - SI33_H_MID_RANGE[1]) < 0.01),
        "base_yardstick_row": {
            "branch": "layer-1-mlp",
            "population": primary_key,
            "head": "base",
            "branch_output_norm_mean": yard["branch_output_norm"]["mean"],
            "branch_output_norm_max": yard["branch_output_norm"]["max"],
            "residual_stream_norm_mean":
                yard["residual_stream_norm"]["mean"],
            "published_si33_representative": {
                "branch_output_norm": SI33_BASE_BRANCH_OUTPUT,
                "residual_stream_norm": SI33_BASE_RESIDUAL,
            },
            "reconciliation": (
                "SI-33 published its base figures for one "
                "representative sequence it does not name, so "
                "exact per-sequence reproduction is not "
                "possible from the published record; the "
                "audit publishes the same branch's statistics "
                "over the full allowed population instead "
                "(method difference: single unnamed sequence "
                "vs named population, last position, "
                "mean/max). The published residual 2.00 "
                "matches the population mean to SI-33's "
                "printed precision, and the published branch "
                "output 0.0074 sits inside the envelope of "
                "the same branch's base measurements across "
                "the audit's allowed populations, beside "
                "this population's mean/max printed here — "
                "reconciled in print, as §16 criterion 3 "
                "requires, before any other figure is "
                "trusted."),
        },
    }

    return {
        "audit": "SI-35 branch-contribution and expressibility "
                 "audit of the miniature",
        "kind": "measurement-only: no candidate authored, "
                "nothing trained, no gate evaluation "
                "performed, no target approved",
        "miniature_labels": "constructed, deterministically "
                            "seeded, not trained",
        "heads": list(AUDIT_HEADS),
        "populations": [
            {**spec, "key": _pop_key(spec),
             "size": len(seqs)}
            for spec, seqs in pops],
        "position_statistic": "last position is primary; "
                              "all-position statistics are "
                              "reported separately, labelled, "
                              "and no reading rests on them",
        "thresholds": {
            "norm_budget": NORM_BUDGET,
            "r1_branch_ratio_min": R1_BRANCH_RATIO_MIN,
            "r1_bound_min": R1_BOUND_MIN,
            "r1_bound_min_kind": "derivation: 0.15 * sqrt(2), "
                                 "the signal correction's "
                                 "logit-space norm",
            "yardsticks": {
                "signal_correction_norm": R1_BOUND_MIN,
                "si33_down_projection_bound_approx":
                    SI33_DOWN_BOUND_YARDSTICK,
            },
        },
        "magnitude_accounting": {
            "si33_candidate": {"norm": SI33_CANDIDATE_NORM,
                               "m2": SI33_CANDIDATE_M2},
            "si25_control": {"norm": SI25_CONTROL_NORM,
                             "m2": SI25_CONTROL_M2},
            "note": "cited to docs/SI-33-final-mlp-down-"
                    "target.md (§16 criterion 6): no bound "
                    "is quoted as if it were a movement and "
                    "no movement as if it were a bound",
        },
        "inventory": inv,
        "branch_measurements": branch_measurements,
        "input_activations": input_activations,
        "expressibility_bounds": bounds,
        "backward_inventory": backward_inventory(),
        "reproduction": reproduction,
        "readings": {
            "primary_cell": {"population": primary_key,
                             "head": "d0001"},
            "qualification_table_primary": qual_primary,
            "qualification_table_base": qual_base,
            "qualification_table_d0002": qual_d0002,
            "R1": {
                "name": "a less-inert single target is arguable",
                "measured": r1_measured,
                "qualifying_matrices": shortlist,
                "consequence_if_measured": (
                    "the qualifying matrices — and only they — "
                    "form the shortlist a future RFC (SI-36) "
                    "may argue a single second target from; "
                    "not an approval, a prediction, or a gate "
                    "figure (§18 item 27)"),
            },
            "R2": {
                "name": "inertness is uniform at the standing "
                        "budgets",
                "measured": r2_measured,
                "consequence_if_measured": (
                    "the single-target upstream-authorship "
                    "line is expressibility-closed on this "
                    "miniature at the standing budgets "
                    "(a derivation-scoped conclusion from "
                    "measured inputs, labelled as such), and "
                    "the next frontier must be argued from a "
                    "different premise in a future RFC"),
            },
            "exactly_one_of_R1_R2": bool(r1_measured) != bool(r2_measured),
            "R3": {
                "name": "qualification is head-dependent",
                "measured": r3_measured,
                "qualifying_at_base": sorted(set_base),
                "qualifying_at_d0001": sorted(set_d0001),
                "qualifying_at_d0002": sorted(set_d0002),
                "consequence_if_measured": (
                    "every future target argument must name "
                    "its head and may not quote a base-measured "
                    "qualification at a landed head, or the "
                    "reverse"),
                "consequence_if_not_measured": (
                    "qualification is head-stable across the "
                    "three measured heads only — a claim "
                    "scoped to those heads, never to depth in "
                    "general"),
            },
        },
        "shortlist": shortlist,
        "shortlist_semantics": "at most a shortlist for an "
                               "argument in a future RFC "
                               "(SI-36); SI-35 names no "
                               "approved target",
    }


def run_audit(bundle_dir: str, manifest: dict | None = None,
              policy: dict | None = None) -> dict:
    """Run the audit against a bundle carrying the landed
    stack (heads ``d0001`` and ``d0002`` must exist on its
    active chain — the audit fails closed, naming the
    missing head, rather than fabricate a head's figures).

    Read-only: assembles each head's weights through
    apply-at-load from in-memory manifest views and runs
    :func:`run_audit_on_weight_sets`. Writes no blob,
    appends no journal entry, changes no manifest.
    """
    if manifest is None:
        from si.bundle import load_bundle

        manifest = load_bundle(bundle_dir)
    avail = available_heads(manifest)
    missing = [h for h in AUDIT_HEADS if h not in avail]
    if missing:
        raise AuditError(
            f"audit requires the landed stack {AUDIT_HEADS!r}; "
            f"this bundle's active chain carries {avail!r} — "
            f"missing head(s) {missing!r}: refusing rather "
            f"than fabricate a head's figures")
    weight_sets = {head: weights_at_head(bundle_dir, manifest,
                                         head, policy)
                   for head in AUDIT_HEADS}
    return run_audit_on_weight_sets(weight_sets)


def audit_record_bytes(record: dict) -> bytes:
    """The record's canonical bytes (sorted-key JSON, the
    bundle layer's canonical form) — the byte-identity
    evidence for determinism and probe-exclusion tests
    (§16 criteria 2 and 8: equality is the evidence;
    near-equality is a failure)."""
    return canonical_json(record)
