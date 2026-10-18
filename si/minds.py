"""Minds: the thing that perceives, proposes, and supplies training signals.

The runtime trains adapters; a mind never touches raw tensors. A mind
supplies: perceptions, a structured Intent, probe suites, a forward pass
over (base + active deltas), and — for delta intents — a training routine
that returns a candidate adapter.

- ``MockMind``: scripted and deterministic, for loop/gate tests. Its
  candidates are canned (norm set by the script); its probes trivially
  pass (identity forward).
- ``DemoMind``: REAL machine learning on CPU. A tiny linear model
  ``y = (W_base + sum(s_i * B_i @ A_i)) @ x`` with a frozen random 8x8
  ``W_base`` (loaded from the bundle's base blob when a bundle path is
  given). ``train_delta`` runs gradient descent on MSE to learn a small
  new input->output mapping as a rank-2 LoRA-style adapter, with identity
  probes and a capability suite that must survive the write.
"""
from __future__ import annotations

import json
from typing import Protocol

import numpy as np
from safetensors.numpy import save  # 0.8.0: save() takes ndarrays -> bytes

from si.gate import Intent

DIM = 8
BASE_SEED = 20261016  # must match si.bundle.DEMO_SEED


class Mind(Protocol):
    @property
    def name(self) -> str: ...
    def perceive(self, journal) -> list: ...
    def propose(self, perceptions: list, ctx) -> Intent: ...
    def identity_probes(self) -> list:
        """[(qid, x, expected), ...]"""
        ...
    def capability_suite(self) -> list:
        """[(cid, x, expected), ...]"""
        ...
    def forward(self, x, active_deltas) -> np.ndarray:
        """y for input x given active_deltas=[{B,A,scale}, ...]."""
        ...
    def train_delta(self, intent: Intent) -> dict:
        """Train the candidate adapter. Returns
        {'delta': {'B','A','rank','alpha','target_modules'},
         'blob_bytes': real .safetensors bytes,
         'norm': Frobenius norm of the scaled delta,
         'hyperparams': {...}}."""
        ...


def _adapter_blob(name: str, B: np.ndarray, A: np.ndarray, rank: int,
                  alpha: int, target_modules: list,
                  training_signal_id: str) -> bytes:
    """Real .safetensors bytes for one LoRA-style adapter."""
    meta = {
        "mind": name,
        "rank": str(rank),
        "alpha": str(alpha),
        "target_modules": json.dumps(list(target_modules)),
        "training_signal_id": training_signal_id,
    }
    return save({"delta.B": np.ascontiguousarray(B, dtype=np.float32),
                 "delta.A": np.ascontiguousarray(A, dtype=np.float32)},
                metadata=meta)


class MockMind:
    """Scripted, deterministic mind for loop/gate tests.

    ``script`` is a list of {"intent": Intent, "candidate_norm": float}.
    propose() pops them in order; train_delta() returns a canned candidate
    with that Frobenius norm. When the script runs out, propose() emits
    idle memory intents.
    """

    def __init__(self, script: list | None = None, name: str = "mock"):
        self._name = name
        if script is None:
            script = [{
                "intent": Intent(
                    kind="delta",
                    reason="MockMind smoke delta: tiny adapter to exercise the loop.",
                    target_modules=["demo_linear"], rank=2, alpha=4,
                    parents=["base"], source_seqs=[1],
                    training_signal_id="mock-signal"),
                "candidate_norm": 0.01,
            }]
        self._script = list(script)
        self._pending_norm = 0.01
        rng = np.random.default_rng(7)
        self._id_probes = [(f"mock-id-{i}",
                            (v := rng.standard_normal(DIM).astype(np.float32)), v.copy())
                           for i in range(3)]
        self._cap_suite = [(f"mock-cap-{i}",
                            (v := rng.standard_normal(DIM).astype(np.float32)), v.copy())
                           for i in range(4)]

    @property
    def name(self) -> str:
        return self._name

    def perceive(self, journal) -> list:
        return []  # read-only; nothing to perceive in the mock

    def propose(self, perceptions: list, ctx) -> Intent:
        if self._script:
            item = self._script.pop(0)
            self._pending_norm = float(item.get("candidate_norm", 0.01))
            return item["intent"]
        return Intent(kind="memory",
                      reason="MockMind idle: nothing to learn this cycle.",
                      collection="experiences",
                      record={"kind": "observation", "data": {"note": "idle cycle"}})

    def identity_probes(self) -> list:
        return self._id_probes

    def capability_suite(self) -> list:
        return self._cap_suite

    def forward(self, x, active_deltas):
        return np.asarray(x, dtype=np.float64)  # identity: probes trivially pass

    def train_delta(self, intent: Intent) -> dict:
        rank = intent.rank or 2
        alpha = intent.alpha or 4
        scale = alpha / rank
        norm = self._pending_norm
        # B,A chosen so that ||scale * B @ A||_F == norm exactly.
        c = norm / (scale * 16.0) if norm > 0 else 0.0
        B = np.full((DIM, rank), c, dtype=np.float32)
        A = np.ones((rank, DIM), dtype=np.float32)
        actual = float(np.linalg.norm(scale * B @ A))
        return {
            "delta": {"B": B, "A": A, "rank": rank, "alpha": alpha,
                      "target_modules": list(intent.target_modules or ["demo_linear"])},
            "blob_bytes": _adapter_blob(self._name, B, A, rank, alpha,
                                        intent.target_modules or ["demo_linear"],
                                        intent.training_signal_id or "mock-signal"),
            "norm": actual,
            "hyperparams": {"mock": True, "target_norm": norm},
        }


class DemoMind:
    """A real tiny linear mind: y = (W_base + sum s_i*B_i@A_i) @ x.

    W_base is the frozen demo seed (loaded from the bundle's base blob when
    ``bundle_path`` is given, else regenerated from BASE_SEED — identical
    bytes either way). train_delta() learns a small new mapping with real
    gradient descent as a rank-2 adapter. CPU only, fully deterministic.
    """

    TRAINING_SIGNAL = "demo-mapping-001"

    def __init__(self, bundle_path: str | None = None, seed: int = BASE_SEED,
                 name: str = "demo-linear"):
        self._name = name
        self._seed = seed
        if bundle_path is not None:
            from si.bundle import load_base_tensor, load_bundle
            manifest = load_bundle(bundle_path)
            self.W_base = load_base_tensor(
                bundle_path, manifest, "W").astype(np.float32)
        else:
            rng = np.random.default_rng(seed)
            self.W_base = rng.standard_normal((DIM, DIM)).astype(np.float32)

        # The new mapping to learn: W_base + P, with P small and rank-1 so a
        # rank-2 adapter can represent it while barely moving the probes.
        trng = np.random.default_rng(seed + 3)
        u = trng.standard_normal(DIM); u /= np.linalg.norm(u)
        v = trng.standard_normal(DIM); v /= np.linalg.norm(v)
        self._P = (0.03 * np.outer(u, v)).astype(np.float32)
        self._train_x = [trng.standard_normal(DIM).astype(np.float32)
                         for _ in range(32)]

        # Fixed held-out probes, anchored to the SEED (identity = the base).
        prng = np.random.default_rng(seed + 2)
        id_x = [prng.standard_normal(DIM).astype(np.float32) for _ in range(3)]
        cap_x = [prng.standard_normal(DIM).astype(np.float32) for _ in range(4)]
        self._id_probes = [(f"demo-id-{i}", x, self.W_base @ x)
                           for i, x in enumerate(id_x)]
        self._cap_suite = [(f"demo-cap-{i}", x, self.W_base @ x)
                           for i, x in enumerate(cap_x)]

    @property
    def name(self) -> str:
        return self._name

    # -- Mind protocol ------------------------------------------------------
    def perceive(self, journal) -> list:
        """Read-only: summarize recent self-edits (the mind's own time)."""
        edits = journal.backend.find("self_edits", sort="-seq", limit=3)
        return [{"edit_id": e.get("edit_id"), "kind": e.get("kind"),
                 "result": e.get("result"), "seq": e.get("seq")} for e in edits]

    def propose(self, perceptions: list, ctx) -> Intent:
        head = getattr(ctx, "head", "base")
        last_seq = getattr(ctx, "last_seq", 0)
        return Intent(
            kind="delta",
            reason=("Learn the demo input->output mapping correction as a "
                    "small rank-2 adapter, keeping identity probes fixed."),
            target_modules=["demo_linear"],
            rank=2,
            alpha=4,
            parents=[head],
            source_seqs=[last_seq] if last_seq else [],
            training_signal_id=self.TRAINING_SIGNAL,
        )

    def identity_probes(self) -> list:
        return self._id_probes

    def capability_suite(self) -> list:
        return self._cap_suite

    def forward(self, x, active_deltas):
        W = self.W_base.astype(np.float64)
        for d in active_deltas:
            B = np.asarray(d["B"], dtype=np.float64)
            A = np.asarray(d["A"], dtype=np.float64)
            W = W + float(d["scale"]) * (B @ A)
        return W @ np.asarray(x, dtype=np.float64)

    def train_delta(self, intent: Intent) -> dict:
        if intent.training_signal_id != self.TRAINING_SIGNAL:
            raise ValueError(f"unknown training signal {intent.training_signal_id!r}")
        rank = intent.rank or 2
        alpha = intent.alpha or 4
        scale = alpha / rank
        rng = np.random.default_rng(self._seed + 101)
        B = (rng.standard_normal((DIM, rank)) * 0.02)
        A = (rng.standard_normal((rank, DIM)) * 0.02)
        X = np.stack(self._train_x).T  # (8, 32)
        W64 = self.W_base.astype(np.float64)
        P64 = self._P.astype(np.float64)
        Y = (W64 + P64) @ X
        lr, steps = 1.0, 2000
        for _ in range(steps):
            E = (W64 + scale * B @ A) @ X - Y
            G = (2.0 / E.size) * (E @ X.T)          # dL/dD
            gB = scale * (G @ A.T)
            gA = scale * (B.T @ G)
            B -= lr * gB
            A -= lr * gA
        B32 = B.astype(np.float32)
        A32 = A.astype(np.float32)
        norm = float(np.linalg.norm(scale * B32 @ A32))
        residual = float(np.linalg.norm(scale * B32 @ A32 - self._P))
        target_modules = list(intent.target_modules or ["demo_linear"])
        return {
            "delta": {"B": B32, "A": A32, "rank": rank, "alpha": alpha,
                      "target_modules": target_modules},
            "blob_bytes": _adapter_blob(self._name, B32, A32, rank, alpha,
                                        target_modules, self.TRAINING_SIGNAL),
            "norm": norm,
            "hyperparams": {"rank": rank, "alpha": alpha, "scale": scale,
                            "steps": steps, "lr": lr,
                            "target_norm": float(np.linalg.norm(self._P)),
                            "residual_norm": residual,
                            "training_signal_id": self.TRAINING_SIGNAL},
        }
