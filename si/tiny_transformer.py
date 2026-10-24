"""SI-15: the miniature's forward path — a library capability, not inference.

Implements RFC SI-13 §4 for the SI-14 miniature (``TinyTransformerMind``,
§3.1 as amended by SI-14: the positional table is included, so the
inventory is **30 tensors / 23,456 elements** — §3.1's 29 / 22,944
plus the +1 tensor / +512 elements SI-14 recorded).

Module-location decision (RFC §4.1 proposed this module and asked
SI-15 to confirm or amend with its own rationale): **confirmed** —
the forward path lives in this NEW module, ``si/tiny_transformer.py``.
Rationale, following the SI-9a/SI-9c precedent: the gate, bundle, and
seed layers keep no dependency on forward internals; tests and a
future probe context (SI-16) can import the forward without importing
the gate or the loop; and ``si/tiny_seed.py``'s seed-time helper
stays what it was — an internal construction aid owned by the seed
path — while this module is the library forward path the RFC designs.
SI-14's as-built note explicitly left SI-15 free to replace or
extract that helper; SI-15 instead keeps it untouched (it is the
seed's own measured record of how its baselines were produced) and
proves below that this path reproduces its recordings exactly.

What this is and is not (honest labels):

- This is a **library capability**: ``forward`` evaluates the
  miniature over a weight set; the bundle entry points assemble that
  weight set through apply-at-load (base + active chain, plus an
  optional candidate overlay). It is NOT serving: nothing here is
  wired to ``si serve`` (transformer bundles, miniature included,
  still return 501), to the gate, or to the loop. Gate behaviour is
  byte-for-byte unchanged — transformer deltas still report checks
  3/4 as "not evaluated" and ESCALATE (asserted by test).
- It is **not a Glimmer forward path**, at any size claim, and not a
  general transformer executor: any architecture descriptor other
  than §3.1's (as recorded by SI-14) is refused, and the pure
  ``forward`` refuses any weight set that is not exactly the
  miniature's 30-tensor inventory at its pinned shapes.
- **Authorship arrived in SI-19, scoped exactly as RFC SI-18
  §4/§7 designs it**: the mind adapter's ``train_delta`` now
  trains ONE LoRA target (``lm_head.weight``) on ONE named,
  deterministically constructed signal with real analytic
  gradients (finite-difference checked), and ``propose`` emits
  exactly that signal's deterministic transformer intent. The
  training itself lives in ``si/tiny_training.py``; this module's
  forward path is unchanged, and anything outside the named
  signal still refuses explicitly rather than fabricate.
  **SI-23 (RFC SI-22 §5.2)** widens the named set to exactly
  two signals served through a registry in
  ``si/tiny_training.py``; the mind is configured with the
  signal it proposes (explicit recorded selection, default
  signal 1) and serves no other signal.
- **SI-21 freshness contract (RFC SI-20 §4.2/§4.3.1, §9.1):**
  exactly one manifest trains in a cycle — the bundle's current
  verified manifest. ``train_delta`` refreshes the mind's
  manifest from the bundle on disk (``load_bundle``, fully
  verified) before delegating, in exactly one place
  (:meth:`TinyTransformerMind.refresh_manifest`, called only
  from ``train_delta``). The SI-19 behaviour — training from
  the construction-time ``self._manifest`` snapshot — is
  closed: a mind object reused across cycles (the
  ``si.run.run_cycles`` shape) can no longer silently retrain
  from the base in cycle 2. ``propose`` already preferred the
  cycle context's manifest; after the refresh, training and
  proposing look at the same head inside one cycle.

Arithmetic discipline (§4.3, recorded per the SI-15 brief):

- Weights and delta factors are **float32** — that is the dtype the
  apply path delivers (``load_applied_tensor`` returns float32; the
  shared fold in ``si.apply.fold_target`` computes its update and
  result in float32), and ``forward`` casts its input weight set to
  float32 before computing, so a caller cannot smuggle in a wider
  weight dtype.
- The forward computation itself widens the float32 weights to
  **float64** and computes in float64, returning logits cast to
  float32 (the §4.1 contract: ``float32 [len, V]``). This mirrors
  SI-14's seed-time helper exactly — same widening, same operation
  order — and follows the runtime's existing precedents for
  float64 compute over float32 storage: the gate's ``_err`` measures
  in float64, ``DemoMind.forward`` computes in float64, and the
  gate's norm recomputation is float64. The consequence is a
  *proven* one, not a hoped-for one: this path reproduces SI-14's
  recorded probe-suite expected logits for the seeded base
  **bit-identically** (after the float32 storage cast both paths
  apply), asserted by ``tests/test_tiny_transformer.py`` over all
  24 probe entries. At 23,456 parameters the float64 cost is noise
  (§4.5), so there is no speed argument for a second, float32
  arithmetic that would drift from the recorded baselines.
- No randomness at forward time: no sampling, no dropout, no RNG.
  Repeat runs are bit-identical (asserted by test).
"""
from __future__ import annotations

import json
import os

import numpy as np

from si.apply import applied_tensor_with_candidate, load_applied_tensor
from si.tiny_seed import (
    ARCHITECTURE,
    EMBED_NAME,
    FINAL_NORM_NAME,
    LM_HEAD_NAME,
    POS_NAME,
    TINY_MIND_NAME,
    tiny_tensor_shapes,
)

__all__ = [
    "EXPECTED_ARCHITECTURE",
    "TinyTransformerError",
    "TinyTransformerMind",
    "applied_tensor_with_candidate",
    "assemble_base_weights",
    "assemble_weights",
    "assemble_weights_with_candidate",
    "forward",
    "forward_from_bundle",
    "validate_architecture",
]

#: The one architecture descriptor this forward path will run:
#: RFC §3.1 exactly as SI-14 recorded it (positional table included).
EXPECTED_ARCHITECTURE = dict(ARCHITECTURE)

_V = EXPECTED_ARCHITECTURE["vocab_size"]
_D = EXPECTED_ARCHITECTURE["d_model"]
_LAYERS = EXPECTED_ARCHITECTURE["layers"]
_HEADS = EXPECTED_ARCHITECTURE["heads"]
_HEAD_DIM = EXPECTED_ARCHITECTURE["head_dim"]
_T_MAX = EXPECTED_ARCHITECTURE["max_seq_len"]
_EPS = float(EXPECTED_ARCHITECTURE["rms_norm_eps"])


class TinyTransformerError(ValueError):
    """The forward path refuses this input (architecture, weights,
    or token sequence). Subclasses ValueError, like the other
    fail-closed errors in this runtime."""


def _layer_prefix(i: int) -> str:
    return f"model.language_model.layers.{i}"


# -- architecture refusal (§6.2 rule 3) --------------------------------------

def validate_architecture(architecture) -> dict:
    """Return ``architecture`` iff it is §3.1's descriptor exactly.

    Fail closed: a non-dict, any missing key, any extra key, or any
    value differing from the descriptor SI-14 recorded (vocab 32,
    d_model 32, 2 layers, 4 heads, head_dim 8, d_ff 64, max_seq_len
    16, positional_embedding True, and the rest of that record)
    raises TinyTransformerError. This forward is a miniature
    executor, not a general one (RFC §6.2 rule 3).
    """
    if not isinstance(architecture, dict):
        raise TinyTransformerError(
            f"architecture descriptor is not an object: "
            f"{architecture!r} — refusing (only the SI-13 §3.1 "
            f"miniature descriptor is runnable)")
    problems = []
    for key, want in EXPECTED_ARCHITECTURE.items():
        if key not in architecture:
            problems.append(f"missing key {key!r} (expected {want!r})")
        elif architecture[key] != want:
            problems.append(
                f"{key}={architecture[key]!r} != expected {want!r}")
    for key in architecture:
        if key not in EXPECTED_ARCHITECTURE:
            problems.append(f"unexpected key {key!r}")
    if problems:
        raise TinyTransformerError(
            "architecture descriptor is not the SI-13 §3.1 miniature "
            "(as amended by SI-14): " + "; ".join(problems))
    return architecture


def _manifest_architecture(manifest: dict) -> dict:
    base = manifest.get("base") if isinstance(manifest, dict) else None
    arch = base.get("architecture") if isinstance(base, dict) else None
    if arch is None:
        raise TinyTransformerError(
            "manifest carries no base architecture descriptor — "
            "refusing (not a miniature bundle; fail closed)")
    return validate_architecture(arch)


def _validate_weights(weights) -> dict:
    """The exact 30-tensor inventory at pinned shapes, as float32."""
    shapes = tiny_tensor_shapes()
    if not isinstance(weights, dict):
        raise TinyTransformerError("weights is not a {name: array} dict")
    missing = [n for n in shapes if n not in weights]
    extra = [n for n in weights if n not in shapes]
    if missing or extra:
        raise TinyTransformerError(
            f"weight set is not the miniature's 30-tensor inventory: "
            f"missing={missing!r} extra={extra!r}")
    out = {}
    for name, shape in shapes.items():
        arr = np.asarray(weights[name])
        if list(arr.shape) != list(shape):
            raise TinyTransformerError(
                f"tensor {name!r} has shape {list(arr.shape)} != "
                f"pinned {list(shape)}")
        out[name] = np.ascontiguousarray(arr, dtype=np.float32)
    return out


# -- the pure forward (§4.1) ---------------------------------------------------

def _rms_norm(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(ms + _EPS) * w


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def forward(token_ids, weights: dict,
            architecture: dict | None = None) -> np.ndarray:
    """``forward(token_ids, weights) -> logits`` (RFC §4.1 contract).

    ``token_ids``: sequence of ints, ``1 <= len <= 16``, each in
    ``[0, 32)``. ``weights``: the full applied tensor set for the
    miniature (exactly its 30 tensors; see ``assemble_weights``).
    ``architecture``: when given, must be §3.1's descriptor exactly
    or this refuses before computing. Returns float32 logits of
    shape ``[len(token_ids), 32]`` — one row per position.

    Computation, per position, in order (§4.1): token + positional
    embedding → per layer: pre-norm RMSNorm → multi-head causal
    self-attention (scaled dot-product, causal mask, softmax) →
    residual add → pre-norm RMSNorm → SwiGLU MLP → residual add →
    final RMSNorm → ``lm_head``. No dropout, no sampling, no KV
    cache — the whole (<= 16-token) sequence is recomputed per call.
    Arithmetic discipline: see the module docstring (float32
    weights widened to float64 for compute; float32 logits out).
    """
    if architecture is not None:
        validate_architecture(architecture)
    w32 = _validate_weights(weights)
    ids = [int(t) for t in token_ids]
    if not (1 <= len(ids) <= _T_MAX):
        raise TinyTransformerError(
            f"sequence length {len(ids)} outside [1, {_T_MAX}]")
    if any(t < 0 or t >= _V for t in ids):
        raise TinyTransformerError(f"token id outside [0, {_V}): {ids!r}")
    w = {k: np.asarray(v, dtype=np.float64) for k, v in w32.items()}
    t = len(ids)
    h = w[EMBED_NAME][ids] + w[POS_NAME][:t]
    causal = np.triu(np.ones((t, t), dtype=bool), k=1)
    for i in range(_LAYERS):
        p = _layer_prefix(i)
        a = _rms_norm(h, w[f"{p}.input_layernorm.weight"])
        q = (a @ w[f"{p}.self_attn.q_proj.weight"].T
             + w[f"{p}.self_attn.q_proj.bias"]).reshape(t, _HEADS, _HEAD_DIM)
        k = (a @ w[f"{p}.self_attn.k_proj.weight"].T
             + w[f"{p}.self_attn.k_proj.bias"]).reshape(t, _HEADS, _HEAD_DIM)
        v = (a @ w[f"{p}.self_attn.v_proj.weight"].T
             + w[f"{p}.self_attn.v_proj.bias"]).reshape(t, _HEADS, _HEAD_DIM)
        q = np.transpose(q, (1, 0, 2))
        k = np.transpose(k, (1, 0, 2))
        v = np.transpose(v, (1, 0, 2))
        scores = q @ np.transpose(k, (0, 2, 1)) / np.sqrt(float(_HEAD_DIM))
        scores = np.where(causal, -np.inf, scores)
        attn = _softmax(scores) @ v                       # [H, t, dh]
        attn = np.transpose(attn, (1, 0, 2)).reshape(t, _D)
        h = h + attn @ w[f"{p}.self_attn.o_proj.weight"].T \
            + w[f"{p}.self_attn.o_proj.bias"]
        m = _rms_norm(h, w[f"{p}.post_attention_layernorm.weight"])
        gate = m @ w[f"{p}.mlp.gate_proj.weight"].T
        up = m @ w[f"{p}.mlp.up_proj.weight"].T
        silu = gate / (1.0 + np.exp(-gate))
        h = h + (silu * up) @ w[f"{p}.mlp.down_proj.weight"].T
    h = _rms_norm(h, w[FINAL_NORM_NAME])
    return (h @ w[LM_HEAD_NAME].T).astype(np.float32)


# -- weight assembly: apply-at-load, plus the candidate overlay (§4.2) --------

def _check_tensor_map(manifest: dict) -> None:
    from si.transformer_delta import base_tensor_map

    names = set(base_tensor_map(manifest))
    want = set(tiny_tensor_shapes())
    if names != want:
        raise TinyTransformerError(
            f"manifest tensor map is not the miniature's inventory: "
            f"missing={sorted(want - names)!r} "
            f"extra={sorted(names - want)!r}")


def assemble_weights(bundle_dir: str, manifest: dict,
                     policy: dict | None = None) -> dict:
    """The full applied weight set at the manifest's head.

    One ``load_applied_tensor`` call per tensor in the miniature's
    manifest tensor map — **30 calls** (RFC §4.2's text says
    twenty-nine, which predates SI-14's positional-table decision;
    the as-built doc records the 29 -> 30 note). The forward path
    never reads base bytes directly: every value comes through
    apply-at-load's hash-verified, delta-folding read. Refuses any
    bundle whose architecture descriptor or tensor map is not the
    miniature's.
    """
    _manifest_architecture(manifest)
    _check_tensor_map(manifest)
    return {name: load_applied_tensor(bundle_dir, manifest, name, policy)
            for name in tiny_tensor_shapes()}


def assemble_base_weights(bundle_dir: str, manifest: dict,
                          policy: dict | None = None) -> dict:
    """The frozen seed base's full applied weight set (SI-25).

    RFC SI-24 §5.4: the base view of the cycle's OWN verified
    manifest — the applied weight set at head ``base``, i.e.
    the base tensors with the active chain truncated to empty.
    This is the ONE place the base weight set is assembled for
    training's target-source role; it lives beside
    :func:`assemble_weights` and nowhere else.

    - Derived, never stored: the derivation is an in-memory
      manifest view (a shallow copy of ``manifest`` with its
      ``head`` set to ``"base"``); the verified manifest itself
      is never mutated, re-headed, or written.
    - Hash-verified reads only, no new bytes: every tensor
      still comes through :func:`si.apply.load_applied_tensor`
      — with the truncated (empty) chain it folds nothing, so
      each value is a hash-verified read of the frozen base.
      No tensor is copied, materialised to disk, or stored:
      the never-duplicate-bytes rule holds by construction.
    - Fail closed: the same architecture / tensor-map refusals
      as :func:`assemble_weights`, and any failed base read
      raises — callers must NOT fall back to head-derived
      targets (RFC §5.4).
    """
    _manifest_architecture(manifest)
    _check_tensor_map(manifest)
    base_manifest = dict(manifest)
    base_manifest["head"] = "base"
    return {name: load_applied_tensor(bundle_dir, base_manifest, name,
                                      policy)
            for name in tiny_tensor_shapes()}


def assemble_weights_with_candidate(bundle_dir: str, manifest: dict,
                                    candidate,
                                    policy: dict | None = None) -> dict:
    """The applied weight set with a not-yet-landed candidate overlaid.

    Same assembly as :func:`assemble_weights`, but every tensor goes
    through :func:`si.apply.applied_tensor_with_candidate`: tensors
    the candidate targets get the candidate's validated fold on top
    of base + active chain; untargeted tensors are the
    ``load_applied_tensor`` result untouched.
    """
    _manifest_architecture(manifest)
    _check_tensor_map(manifest)
    return {name: applied_tensor_with_candidate(
                bundle_dir, manifest, name, candidate, policy)
            for name in tiny_tensor_shapes()}


def forward_from_bundle(bundle_dir: str, manifest: dict, token_ids,
                        policy: dict | None = None,
                        candidate=None) -> np.ndarray:
    """Forward over a bundle: assemble weights (§4.2), then ``forward``.

    With ``candidate=None`` this is base + active chain at the
    manifest's head — a rolled-back delta is not on the chain, so it
    does not affect the result. With a candidate (the SI-9b
    ``{"record", "blob_bytes"}`` pair), the candidate is overlaid
    in memory only; it does not land.
    """
    if candidate is None:
        weights = assemble_weights(bundle_dir, manifest, policy)
    else:
        weights = assemble_weights_with_candidate(
            bundle_dir, manifest, candidate, policy)
    return forward(token_ids, weights)


# -- mind adapter (§4.1 surface, §4.4 honesty) ---------------------------------

class TinyTransformerMind:
    """A bundle-backed adapter exposing the probe/forward surface.

    This adapter exists so a future probe context (SI-16) has the
    §4.1 surface to consume: ``forward`` / ``forward_candidate``
    over token-id sequences, and the probe suites recorded in the
    bundle's ``probe_suite.json``. It is wired to NOTHING in SI-15:
    no gate, loop, or serve path constructs it.

    SI-19 honesty (RFC SI-18 §4): this mind CAN now train and
    propose — for exactly one named signal and one LoRA target,
    implemented in ``si.tiny_training``. ``train_delta`` refuses
    (``ValueError``) any other signal, any other declared target
    list, and any candidate whose held-out signal loss does not
    improve; ``propose`` emits the named signal's deterministic
    transformer intent and nothing else. Authorship is not
    admission: the gate judges an authored candidate exactly as
    it judges a fixture.

    SI-23: the named-signal set is now exactly two (the SI-23
    registry in ``si.tiny_training``); this mind proposes the
    ONE signal it is configured with (see ``__init__`` /
    ``set_training_signal``) and refuses every other signal
    through the same dispatch.

    SI-25: the mind is likewise configured with the target
    ANCHORING it will train under (see ``__init__`` /
    ``set_target_anchoring``) — an explicit, recorded,
    per-cycle driver configuration beside the signal
    selection, defaulting to ``"weights-under-training"``.
    The anchoring is carried on the proposed intent and
    recorded in every cycle's hyperparams. No autonomous,
    learned, depth-driven, or verdict-driven anchoring
    selection exists.

    SI-27: the mind is likewise configured with the
    COMPOSITION configuration it will train under (see
    ``__init__`` / ``set_composition_config``) — a third
    explicit, recorded, per-cycle driver configuration
    beside signal selection and anchoring, defaulting to
    ``"none"`` (the unpenalised SI-25 path). The
    configuration is carried on the proposed intent and
    recorded in every cycle's hyperparams with its λ and
    the penalty's measured values. No autonomous, learned,
    depth-driven, or verdict-driven composition selection
    exists, and λ is not independently settable.

    SI-29 (RFC SI-28 §6.4): exactly three composition
    configurations are served, the third being
    ``"worst-input-composition"`` — the worst-input term
    (λ_w = 1.0, k = 4) entering the loss in place of the
    mean-penalty term, prepared and recorded exactly as
    above. Still no autonomous selection, and λ_w/k are
    not independently settable.

    SI-31 (RFC SI-30 §7.3): exactly four composition
    configurations are served, the fourth being
    ``"train-partition-projection"`` — the training step
    is SI-25's and the judged candidate is constructed by
    full removal of the train-partition-parallel component
    (coefficient α measured per run in
    ``si.tiny_training``), prepared and recorded exactly
    as above. Still no autonomous selection, and α is not
    independently settable: neither the constructor nor
    the setter takes such an argument.
    """

    def __init__(self, bundle_dir: str | None = None,
                 manifest: dict | None = None,
                 policy: dict | None = None,
                 training_signal_id: str | None = None,
                 target_anchoring: str | None = None,
                 composition_config: str | None = None):
        if bundle_dir is not None and manifest is None:
            from si.bundle import load_bundle
            manifest = load_bundle(bundle_dir)
        if manifest is not None:
            _manifest_architecture(manifest)
        self._bundle_dir = bundle_dir
        self._manifest = manifest
        self._policy = policy
        # SI-23 (RFC SI-22 §5.2): explicit, recorded selection.
        # The mind is configured with the signal it will propose;
        # the default is signal 1. Validation is through the
        # registry (unknown id -> ValueError, the existing
        # discipline). There is no autonomous selection.
        from si.tiny_training import (
            DEFAULT_COMPOSITION_CONFIG,
            TARGET_ANCHORING,
            TRAINING_SIGNAL_ID,
            signal_spec,
            validate_composition_config,
            validate_target_anchoring,
        )

        chosen = training_signal_id if training_signal_id is not None \
            else TRAINING_SIGNAL_ID
        signal_spec(chosen)
        self._training_signal_id = chosen
        # SI-25 (RFC SI-24 §5.3): explicit, recorded anchoring
        # configuration, mirroring the signal selection above.
        # The default is TARGET_ANCHORING ("weights-under-
        # training"); validation serves exactly the two named
        # anchorings and refuses every other value.
        anchoring = target_anchoring if target_anchoring is not None \
            else TARGET_ANCHORING
        validate_target_anchoring(anchoring)
        self._target_anchoring = anchoring
        # SI-27 (RFC SI-26 §7.4): explicit, recorded composition
        # configuration, mirroring both axes above. The default
        # is "none"; validation serves exactly the two named
        # configurations and refuses every other value.
        comp = composition_config if composition_config is not None \
            else DEFAULT_COMPOSITION_CONFIG
        validate_composition_config(comp)
        self._composition_config = comp

    @property
    def name(self) -> str:
        return TINY_MIND_NAME

    @property
    def training_signal_id(self) -> str:
        """The explicitly configured signal this mind proposes
        (SI-23 recorded selection; see :meth:`set_training_signal`)."""
        return self._training_signal_id

    def set_training_signal(self, signal_id: str) -> None:
        """Explicitly reconfigure the proposed signal (SI-23).

        This is the driver saying, in code, which signal the
        next cycle proposes — the journal then records it in
        that cycle's ``training-hyperparams``. It is not a
        choice the mind makes: no perception, depth, or
        alternation rule calls this. Unknown ids refuse with
        the registry's ``ValueError`` discipline.
        """
        from si.tiny_training import signal_spec

        signal_spec(signal_id)
        self._training_signal_id = signal_id

    @property
    def target_anchoring(self) -> str:
        """The explicitly configured target anchoring this mind
        trains under (SI-25 recorded configuration; see
        :meth:`set_target_anchoring`)."""
        return self._target_anchoring

    def set_target_anchoring(self, anchoring: str) -> None:
        """Explicitly reconfigure the target anchoring (SI-25).

        This is the driver saying, in code, which anchoring the
        next cycle trains under — the journal then records it
        in that cycle's ``training-hyperparams`` as
        ``target_anchoring`` / ``target_source_head``. It is
        not a choice the mind makes: no perception, depth,
        alternation, or verdict rule calls this. Any value
        outside the two named anchorings refuses with the
        existing ``ValueError`` discipline.
        """
        from si.tiny_training import validate_target_anchoring

        validate_target_anchoring(anchoring)
        self._target_anchoring = anchoring

    @property
    def composition_config(self) -> str:
        """The explicitly configured composition configuration
        this mind trains under (SI-27 recorded configuration;
        see :meth:`set_composition_config`)."""
        return self._composition_config

    def set_composition_config(self, config: str) -> None:
        """Explicitly reconfigure the composition
        configuration (SI-27).

        This is the driver saying, in code, which composition
        configuration the next cycle trains under — the journal
        then records it in that cycle's ``training-hyperparams``
        as ``composition_config`` / ``composition_lambda`` with
        the penalty's measured values. It is not a choice the
        mind makes: no perception, depth, alternation, or
        verdict rule calls this. Any value outside the four
        named configurations refuses with the existing
        ``ValueError`` discipline. λ cannot be set through
        this (or any) call: it is the configuration's
        definition (1.0 under ``"movement-composition"``).
        α likewise cannot be set through this (or any)
        call: under ``"train-partition-projection"`` it is
        measured from the run's own candidates on the
        train partition (SI-31).
        """
        from si.tiny_training import validate_composition_config

        validate_composition_config(config)
        self._composition_config = config

    def _require_bundle(self) -> None:
        if self._bundle_dir is None or self._manifest is None:
            raise TinyTransformerError(
                "this TinyTransformerMind is not bundle-backed: the "
                "adapter's forward/probe surface assembles weights "
                "from a bundle via apply-at-load (the pure "
                "si.tiny_transformer.forward takes an explicit "
                "weight set instead)")

    def forward(self, token_ids, candidate=None) -> np.ndarray:
        """Logits for ``token_ids`` at the bundle head (+ overlay)."""
        self._require_bundle()
        return forward_from_bundle(
            self._bundle_dir, self._manifest, token_ids,
            policy=self._policy, candidate=candidate)

    def forward_candidate(self, token_ids, candidate) -> np.ndarray:
        """Logits with ``candidate`` overlaid (the §4.2 mechanism)."""
        return self.forward(token_ids, candidate=candidate)

    def _probe_artifact(self) -> dict:
        self._require_bundle()
        path = os.path.join(self._bundle_dir, "probe_suite.json")
        if not os.path.isfile(path):
            raise TinyTransformerError(
                f"bundle carries no probe artifact ({path} missing): "
                f"no probe suite exists for this bundle — fail closed")
        with open(path, "rb") as f:
            return json.loads(f.read().decode("utf-8"))

    def identity_probes(self) -> list:
        """[(qid, token_ids, expected_last_position_logits), ...]"""
        art = self._probe_artifact()
        return [(e["id"], [int(x) for x in e["input"]],
                 np.asarray(e["expected_logits"], dtype=np.float32))
                for e in art["identity_suite"]]

    def capability_suite(self) -> list:
        """[(cid, token_ids, expected_last_position_logits), ...]"""
        art = self._probe_artifact()
        return [(e["id"], [int(x) for x in e["input"]],
                 np.asarray(e["expected_logits"], dtype=np.float32))
                for e in art["capability_suite"]]

    def capability_baseline(self) -> dict:
        """{cid: error} recorded at seed time (0.0 at head=base)."""
        art = self._probe_artifact()
        return dict(art["baselines"]["capability"])

    def perceive(self, journal) -> list:
        return []  # read-only; nothing to perceive in this adapter

    def propose(self, perceptions: list, ctx):
        """The deterministic intent this mind proposes: the
        CONFIGURED training signal's transformer intent (SI-19
        shape; SI-23 recorded selection — the signal is the
        constructor / :meth:`set_training_signal` choice, never
        an autonomous one; SI-25 recorded anchoring — the
        anchoring is likewise the constructor /
        :meth:`set_target_anchoring` choice, carried on the
        intent), mirroring ``DemoMind.propose``'s
        fixed-intent shape. Head / last journal seq are read from
        the context's manifest when it carries one (the loop's
        ``ProbeContext`` does), else from this mind's own bundle,
        else the seed-time defaults. SI-27: the intent also
        carries the configured composition configuration
        (``composition_config``), exactly as it carries the
        configured signal and anchoring."""
        from si.gate import Intent
        from si.tiny_training import (
            ALPHA,
            RANK,
            TARGET_NAME,
        )
        from si.transformer_delta import KIND as TF_KIND

        signal_id = self._training_signal_id
        manifest = getattr(ctx, "manifest", None)
        if not isinstance(manifest, dict):
            manifest = self._manifest
        head = "base"
        last_seq = 0
        if isinstance(manifest, dict):
            head = manifest.get("head", "base")
            journal = manifest.get("journal")
            if isinstance(journal, dict):
                last_seq = int(journal.get("last_seq", 0) or 0)
        return Intent(
            kind=TF_KIND,
            reason=("Train the named signal "
                    f"{signal_id} as a small rank-{RANK} "
                    f"LoRA correction on {TARGET_NAME} (CPU "
                    "gradient descent on a constructed, "
                    "probe-disjoint signal; the gate judges the "
                    "candidate)."),
            target_modules=[TARGET_NAME],
            rank=RANK,
            alpha=ALPHA,
            parents=[head],
            source_seqs=[last_seq] if last_seq else [],
            training_signal_id=signal_id,
            target_anchoring=self._target_anchoring,
            composition_config=self._composition_config,
            targets=[TARGET_NAME],
        )

    def refresh_manifest(self) -> dict:
        """Reload the verified manifest from the bundle on disk.

        THE freshness contract's one place (SI-21, RFC SI-20
        §4.3.1): the manifest that trains is the bundle's
        current verified manifest — never the construction-time
        snapshot this adapter was built with. ``load_bundle``
        re-verifies the bundle (journal anchor, hashes) on every
        call, so a tampered or half-written bundle fails closed
        here rather than training against a fantasy head. The
        refreshed manifest replaces ``self._manifest``, so the
        adapter's forward surface also reads at the fresh head
        after a training call. Called only from
        :meth:`train_delta`; no other code path in this adapter
        chooses a training manifest.
        """
        self._require_bundle()
        from si.bundle import load_bundle

        self._manifest = load_bundle(self._bundle_dir)
        return self._manifest

    def train_delta(self, intent) -> dict:
        """Train the named signal's candidate (SI-19; SI-21
        freshness contract).

        Refreshes the manifest first (see
        :meth:`refresh_manifest` — the cycle's verified manifest
        is the one that trains), then delegates to
        :func:`si.tiny_training.train_candidate`: real CPU
        gradient-based training of one LoRA target on the named
        signal, returned as a ``construct_transformer_delta``
        candidate pair. Refuses explicitly — never fabricates —
        when this mind is not bundle-backed, the signal is
        unknown, the intent declares other targets or a
        different rank/alpha, or the trained candidate's
        held-out loss does not improve.
        """
        self._require_bundle()
        manifest = self.refresh_manifest()
        from si.tiny_training import train_candidate

        return train_candidate(
            self._bundle_dir, manifest, intent,
            policy=self._policy)
