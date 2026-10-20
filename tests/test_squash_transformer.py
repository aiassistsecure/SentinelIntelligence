"""SI-12: ``si squash`` fails closed on transformer deltas.

Transformer squash (composition over named tensors) is NOT implemented
— it is an explicit non-goal (RFC SI-9 §6.4). Before SI-12, squash had
no explicit ``kind == "transformer-delta-v1"`` check; what actually
happened depended on the stack shape (all reproduced on synthetic
miniature bundles before the fix; see
``docs/SI-12-squash-transformer-fail-closed.md``):

- multi-tensor base: refused, but only by the generic SI-3
  single-tensor scope error, never naming transformer deltas;
- single-tensor base, transformer delta in the squash segment S,
  default rank: raw ``KeyError: 'rank'`` (transformer records carry
  per-target rank/alpha, no record-level ``rank``);
- single-tensor base, transformer delta in S with an explicit
  ``--rank``, or only in the kept tail K: a demo-format candidate blob
  (``delta.B``/``delta.A``) was WRITTEN to the blob store, tolerance
  passed, and the run then crashed with a raw ``SafetensorError``
  (``_load_stack`` reading the transformer blob as a demo blob),
  leaving the orphan candidate blob behind;
- mixed stacks hit whichever of the above came first.

SI-12's rule (documented in ``si/squash.py`` and the SI-12 doc): if
the RESOLVED ACTIVE CHAIN contains any transformer-delta-v1 record —
in S, in K, or both — and a squash would actually run (``L > keep``),
raise SquashError before any blob, journal, or manifest write.
Records NOT on the resolved active chain are not involved and do
not block a demo squash; ``L <= keep`` remains the unchanged no-op.
Demo-only squash is unchanged (``tests/test_squash.py``).
"""
import os

import numpy as np
import pytest
from safetensors.numpy import save

from si.bundle import (
    _active_chain,
    _anchor_checkpoint,
    _write_manifest,
    canonical_json,
    init_bundle,
    load_bundle,
)
from si.journal import open_journal
from si.loop import _write_blob
from si.minds import _adapter_blob
from si.seed import seed_bundle
from si.squash import SquashError, squash_bundle
from si.transformer_delta import KIND, construct_transformer_delta

QPROJ = "model.language_model.layers.0.self_attn.q_proj.weight"   # [4, 3]
VPROJ = "model.vision_tower.layers.0.attn.q_proj.weight"          # [2, 2]
SHAPES = {QPROJ: (4, 3), VPROJ: (2, 2)}


def _make_bundle(tmp_path, name, shapes):
    rng = np.random.default_rng(11)
    tensors = {n: rng.standard_normal(s).astype(np.float32)
               for n, s in shapes.items()}
    ckpt = tmp_path / (name + "-ckpt")
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(save(tensors))
    (ckpt / "config.json").write_text(
        '{"model_type": "muse_glimmer", "architectures": '
        '["MuseGlimmerForConditionalGeneration"]}')
    bundle_dir = init_bundle(str(tmp_path / (name + ".si")), name=name,
                             journal_backend="embedded")
    seed_bundle(bundle_dir, str(ckpt), importer="test")
    return bundle_dir, tensors


def _lora_spec(name, shape, seed=1):
    rng = np.random.default_rng(seed)
    out_dim, in_dim = shape
    return {"name": name, "kind": "lora",
            "A": (rng.standard_normal((2, in_dim)) * 0.05).astype(np.float32),
            "B": (rng.standard_normal((out_dim, 2)) * 0.05).astype(np.float32),
            "rank": 2, "alpha": 4}


def _persist(bundle_dir, manifest, note):
    journal = open_journal(bundle_dir, manifest)
    try:
        anchored = _anchor_checkpoint(journal, manifest,
                                      head=manifest["head"], note=note)
        _write_manifest(bundle_dir, anchored)
    finally:
        journal.close()
    return load_bundle(bundle_dir)


def _stack_transformer(bundle_dir, name, shape, delta_id, seed=1):
    manifest = load_bundle(bundle_dir)
    record, blob = construct_transformer_delta(
        manifest, [_lora_spec(name, shape, seed=seed)], delta_id=delta_id,
        parent=manifest["head"], created_at="2026-10-19T00:00:00Z",
        self_edit_id="se_" + delta_id)
    assert record["kind"] == KIND
    _write_blob(bundle_dir, blob)
    manifest["deltas"].append(record)
    manifest["head"] = delta_id
    return _persist(bundle_dir, manifest, f"stack {delta_id}")


def _demo_record(delta_id, parent, blob_hex, byte_len, B, A):
    return {"id": delta_id, "parent": parent, "blob": "sha256:" + blob_hex,
            "bytes": byte_len, "rank": 1, "alpha": 1,
            "target_modules": ["demo_linear"],
            "norm": float(np.linalg.norm(B @ A)),
            "created_at": "2026-10-19T00:00:00Z",
            "self_edit_id": "se_" + delta_id}


def _small_demo_factors(seed):
    rng = np.random.default_rng(seed)
    B = (rng.standard_normal((8, 1)) * 0.05).astype(np.float32)
    A = (rng.standard_normal((1, 8)) * 0.05).astype(np.float32)
    return B, A


def _stack_demo(bundle_dir, delta_id, seed=1, journaled=False):
    B, A = _small_demo_factors(seed)
    blob = _adapter_blob("test", B, A, rank=1, alpha=1,
                         target_modules=["demo_linear"],
                         training_signal_id="si12-test")
    blob_hex, byte_len = _write_blob(bundle_dir, blob)
    manifest = load_bundle(bundle_dir)
    record = _demo_record(delta_id, manifest["head"], blob_hex, byte_len,
                          B, A)
    manifest["deltas"].append(record)
    manifest["head"] = delta_id
    if not journaled:
        return _persist(bundle_dir, manifest, f"stack {delta_id}")
    # Journal the delta row in the same session, before anchoring, so
    # a later squash finds its provenance source seqs (loop-style).
    journal = open_journal(bundle_dir, manifest)
    try:
        journal.record_delta(delta_id, record["parent"], record["blob"],
                             byte_len, 1, 1, ["demo_linear"],
                             record["norm"], record["self_edit_id"], [1])
        anchored = _anchor_checkpoint(journal, manifest, head=delta_id,
                                      note=f"stack {delta_id}")
        _write_manifest(bundle_dir, anchored)
    finally:
        journal.close()
    return load_bundle(bundle_dir)


def _move_head(bundle_dir, to):
    manifest = load_bundle(bundle_dir)
    manifest["head"] = to
    return _persist(bundle_dir, manifest, f"head -> {to}")


def _snapshot(bundle_dir):
    manifest = load_bundle(bundle_dir)
    journal = open_journal(bundle_dir, manifest)
    try:
        counts = {c: journal.backend.count(c)
                  for c in ("self_edits", "deltas", "provenance",
                            "checkpoints")}
    finally:
        journal.close()
    blobs = sorted(os.listdir(os.path.join(bundle_dir, "blobs", "sha256")))
    return canonical_json(manifest), counts, blobs


def _assert_refused_unchanged(bundle_dir, **kw):
    before = _snapshot(bundle_dir)
    with pytest.raises(SquashError, match="transformer squash is not "
                                           "implemented"):
        squash_bundle(bundle_dir, **kw)
    assert _snapshot(bundle_dir) == before, \
        "transformer refusal must leave manifest, journal, and blob " \
        "store provably unchanged (no new blob, no journal entry)"


# -- transformer delta in the squash segment S --------------------------------

def test_transformer_in_squash_segment_refused_unchanged(tmp_path, capsys):
    bundle_dir, _ = _make_bundle(tmp_path, "si12-s", {QPROJ: (4, 3)})
    _stack_transformer(bundle_dir, QPROJ, (4, 3), "d0001", seed=1)
    _stack_transformer(bundle_dir, QPROJ, (4, 3), "d0002", seed=2)
    assert [d["id"] for d in _active_chain(load_bundle(bundle_dir))] == \
        ["d0001", "d0002"]

    _assert_refused_unchanged(bundle_dir, keep=0)

    # The CLI surfaces the same refusal as a clean named error.
    from si.cli import main
    before = _snapshot(bundle_dir)
    rc = main(["squash", bundle_dir, "--keep", "0"])
    assert rc != 0
    err = capsys.readouterr().err.lower()
    assert "error" in err and "transformer" in err
    assert "keyerror" not in err and "safetensor" not in err
    assert _snapshot(bundle_dir) == before


def test_transformer_in_segment_refused_with_explicit_rank(tmp_path):
    # Pre-fix, an explicit rank bypassed the KeyError, wrote a
    # demo-format candidate blob, and then crashed raw in _load_stack.
    bundle_dir, _ = _make_bundle(tmp_path, "si12-s-rank", {"W": (8, 8)})
    _stack_transformer(bundle_dir, "W", (8, 8), "d0001", seed=1)
    _stack_transformer(bundle_dir, "W", (8, 8), "d0002", seed=2)
    _assert_refused_unchanged(bundle_dir, keep=0, rank=2)


def test_multi_tensor_transformer_refusal_names_transformer(tmp_path):
    # Pre-fix this was refused only by the generic SI-3 single-tensor
    # scope error. The SI-12 check runs first and names the real reason.
    bundle_dir, _ = _make_bundle(tmp_path, "si12-multi", SHAPES)
    _stack_transformer(bundle_dir, QPROJ, (4, 3), "d0001", seed=1)
    _stack_transformer(bundle_dir, QPROJ, (4, 3), "d0002", seed=2)
    before = _snapshot(bundle_dir)
    with pytest.raises(SquashError) as ei:
        squash_bundle(bundle_dir, keep=0)
    assert "transformer" in str(ei.value)
    assert "single-tensor scope" not in str(ei.value)
    assert _snapshot(bundle_dir) == before


# -- transformer delta only in the kept tail K ---------------------------------

def test_transformer_only_in_kept_tail_refused_unchanged(tmp_path):
    # Demo delta in S, transformer delta in K. Pre-fix (W-shaped
    # fixture): candidate blob written, then raw SafetensorError from
    # _load_stack, orphan blob left behind. K is involved in a squash
    # (tolerance verify + gate stack), so K-only is refused too.
    bundle_dir, _ = _make_bundle(tmp_path, "si12-k", {"W": (8, 8)})
    _stack_demo(bundle_dir, "d0001", seed=1)
    _stack_transformer(bundle_dir, "W", (8, 8), "d0002", seed=3)
    manifest = load_bundle(bundle_dir)
    assert [d["kind"] if "kind" in d else "demo"
            for d in _active_chain(manifest)] == ["demo", KIND]
    _assert_refused_unchanged(bundle_dir, keep=1)


# -- mixed stacks ----------------------------------------------------------------

def test_mixed_transformer_in_segment_demo_in_tail_refused(tmp_path):
    bundle_dir, _ = _make_bundle(tmp_path, "si12-mixed", {QPROJ: (4, 3)})
    _stack_transformer(bundle_dir, QPROJ, (4, 3), "d0001", seed=1)
    # Demo delta on top (its factors are [4,1]x[1,3] for this shape).
    rng = np.random.default_rng(5)
    B = (rng.standard_normal((4, 1)) * 0.05).astype(np.float32)
    A = (rng.standard_normal((1, 3)) * 0.05).astype(np.float32)
    blob = _adapter_blob("test", B, A, rank=1, alpha=1,
                         target_modules=["q_proj"],
                         training_signal_id="si12-mixed")
    blob_hex, byte_len = _write_blob(bundle_dir, blob)
    manifest = load_bundle(bundle_dir)
    manifest["deltas"].append(
        _demo_record("d0002", "d0001", blob_hex, byte_len, B, A))
    manifest["head"] = "d0002"
    _persist(bundle_dir, manifest, "stack mixed demo")
    _assert_refused_unchanged(bundle_dir, keep=1)


# -- documented scope edges --------------------------------------------------------

def test_nothing_to_squash_is_still_a_noop_with_transformer(tmp_path):
    # L <= keep: no squash would run and no delta is involved, so the
    # unchanged no-op return applies (documented SI-12 decision).
    bundle_dir, _ = _make_bundle(tmp_path, "si12-noop", {QPROJ: (4, 3)})
    _stack_transformer(bundle_dir, QPROJ, (4, 3), "d0001", seed=1)
    before = _snapshot(bundle_dir)
    assert squash_bundle(bundle_dir, keep=3) == {"squashed": False}
    assert _snapshot(bundle_dir) == before


def test_inactive_transformer_does_not_block_demo_squash(tmp_path):
    # A transformer record that is NOT on the resolved active chain
    # (head moved back past it) is not involved in the squash: the
    # demo squash of the active chain proceeds exactly as before.
    bundle_dir, _ = _make_bundle(tmp_path, "si12-inactive", {"W": (8, 8)})
    _stack_demo(bundle_dir, "d0001", seed=1, journaled=True)
    _stack_demo(bundle_dir, "d0002", seed=2, journaled=True)
    _stack_transformer(bundle_dir, "W", (8, 8), "d0003", seed=3)
    manifest = _move_head(bundle_dir, "d0002")
    assert [d["id"] for d in _active_chain(manifest)] == ["d0001", "d0002"]

    res = squash_bundle(bundle_dir, keep=1)
    assert res["squashed"] is True, f"demo squash blocked by inactive " \
                                     f"transformer record: {res}"
    assert res["delta_id"] == "d0004"
    manifest = load_bundle(bundle_dir)
    assert manifest["head"] == "d0002"
    assert [d["id"] for d in _active_chain(manifest)] == ["d0004", "d0002"]
    by_id = {d["id"]: d for d in manifest["deltas"]}
    assert by_id["d0003"]["kind"] == KIND
    assert "superseded_by" not in by_id["d0003"]
