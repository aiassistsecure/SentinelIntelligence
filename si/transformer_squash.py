"""SI-38: transformer squash — the implementation of RFC SI-37's
design for the landed mixed-target stack.

RFC SI-37 (PR #42) argued and decided the three obligations
RFC SI-34 §8 recorded for transformer squash, and SI-12's
fail-closed refusal stood until an implementation slice was
triggered (§6: depth / operator direction / demonstrated
present-tense need). This slice is that implementation, built
under §6's operator-direction arm. It enacts:

- **§3.8 — the composite class's pricing discipline.** A squash
  composite is a ``transformer-delta-v1`` record carrying the
  segment's distinct tensors in segment order, each tensor's
  update carried EXACTLY (factors verbatim for a single member;
  scale-folded factor blocks concatenated for several members
  on one tensor, at the exact-carrying rank = the sum of the
  members' ranks, stored with ``alpha = rank`` so the scale is
  1). It is priced by the gate as a composite — count against
  the attested segment's distinct tensors, per-target rank at
  the attested exact-carrying rank, norms and bytes against the
  UNCHANGED caps — attested by journal provenance no authored
  candidate can present (see ``si.gate``'s
  ``_verified_squash_attestation``). The authored budgets
  (``max_targets_per_delta`` 1, ``allow_additive`` false,
  ``rank_cap`` 8) are unchanged and price every authored
  candidate exactly as before. Lossy recompression (truncating
  a target below its exact-carrying rank) is NOT implemented —
  it is recorded open at RFC SI-37 §11 item 2, and
  ``si squash --rank`` on a transformer chain refuses.
- **§4 — the gate context.** :class:`TransformerSquashContext`
  takes demo ``SquashContext``'s shape into the transformer
  mapping: probes from the miniature's probe artifact, the
  capability baseline measured on the PRE-squash stack,
  ``forward_candidate`` running the POST-squash stack, and
  ``stack_depth`` reporting the kept tail's length so the
  gate's +1 is the true post-squash depth. The five checks run
  in the gate's existing transformer mapping (SI-9b/SI-17),
  with check 2 priced under §3.8 and check 5 extended by the
  squash attestation. The per-member axis record (§4.2) is
  assembled from the members' own journaled cycle records —
  nothing is re-derived or defaulted; an unrecoverable field
  refuses the squash before any blob is written.
- **§5 — the tolerance verify.** Before the gate and before
  any commit, the built composite is verified against the
  pre-squash stack: per named tensor, the max-abs difference
  of folded weights; and behaviourally, both stacks are
  forwarded over the four allowed populations ONLY (both
  signals' recorded train and held-out partitions — no probe
  input, probe logit, or probe-derived quantity enters any
  part of it; invariant 9) and compared at every position
  under the all-positions max-abs statistic, tolerance 1e-4
  (a rounding tolerance, derived at RFC SI-37 §5.5 — never
  from a gate tolerance). A verify failure — or an
  unverifiable verify — is a refusal in SI-12's form: a
  :class:`~si.squash.SquashError`, the candidate blob deleted
  again unless it pre-existed, nothing journaled.

Pipeline failure/verdict semantics are §4.7's: preconditions
fail closed before any write; REJECT/ESCALATE are journaled
exactly as the loop journals them (candidate blob stays as
evidence), with the §5 verify record journaled beside the
verdict (§5.6: the verify record is published whatever the
outcome the pipeline reaches); on ADMIT the self-edit is
journaled with ``squashed_from`` provenance, every segment
member is marked ``superseded_by``, the composite is
appended, and the head id deliberately does NOT move (demo
squash's standing semantics — rollback, diff, and why are
keyed on delta ids). One consequence of the standing chain
resolution is stated plainly because the landed stack
exercises it: when the segment contains the head itself
(the full-segment squash of a depth-2 stack), the resolver's
time-travel rule resolves that head's HISTORICAL ancestry —
which the §5 verify has just proven computes identically to
the composite stack — and the composite is the resolved
ancestry for every head above the segment. Nothing about
that is hidden: the as-built record
(``docs/SI-38-transformer-squash.md``) and this slice's
tests state and prove both resolutions.

Scope, honestly: miniature bundles only. For any other
bundle the §5 verify has no lawful population (RFC SI-37
§5.8) and the pipeline refuses — unverifiable is refused,
and no Glimmer forward path is designed, implied, or faked.
Segments containing an additive member are excluded
(§3.8 rule 5). Mixed demo/transformer active chains are
refused by ``si.squash`` before this module is reached.
CPU only, existing dependencies only. This is machine
learning infrastructure, not "custom LLM software."
"""
from __future__ import annotations

import copy
import json
import os
import struct

import numpy as np

from si.bundle import blob_path, load_policy, sha256_hex, utc_now
from si.gate import Gate, Intent
from si.journal import open_journal
from si.loop import _next_delta_id, _save_and_anchor, _write_blob
from si.probes import ProbeContext, is_miniature_probe_bundle
from si.squash import SquashError
from si.transformer_delta import (
    KIND as TRANSFORMER_DELTA_KIND,
    construct_transformer_delta,
    parse_blob,
)

#: The §5 verify tolerance (RFC SI-37 §5.5): a rounding
#: tolerance, the standing demo-verify value, 2,500x tighter
#: than identity_tol. Never derived from a gate tolerance.
VERIFY_TOL = 1e-4


# ---------------------------------------------------------------------------
# Composite construction (§3.8 exact carrying)
# ---------------------------------------------------------------------------

def distinct_tensors(segment: list) -> list:
    """The segment's distinct target tensor names, in segment
    order (oldest member first; within a member, record order)."""
    names: list = []
    for d in segment:
        for t in d.get("targets") or []:
            if isinstance(t, dict) and t.get("name") not in names:
                names.append(t["name"])
    return names


def _read_blob_bytes(bundle_dir: str, blob_ref: str) -> bytes:
    path = blob_path(bundle_dir, blob_ref)
    if not os.path.isfile(path):
        raise SquashError(
            f"transformer squash refused: delta blob missing from "
            f"the bundle store: {path}")
    with open(path, "rb") as f:
        return f.read()


def _canonical_blob_bytes(blob_bytes: bytes) -> bytes:
    """Canonicalise a transformer-delta blob's safetensors
    header: ``__metadata__`` first with its keys sorted, then
    tensor entries sorted by name. Total header length is
    preserved exactly (key permutation cannot change the
    serialised length; the tail is space padding), so the
    data section does not move and the record's byte count
    is unchanged.

    Why this exists: the safetensors serialiser emits
    ``__metadata__`` in hash-map order, which varies per
    call — two serialisations of the SAME composite differ
    in header bytes (payload always identical), as measured
    during SI-38 (two hash variants over repeated identical
    constructions). RFC SI-37 §3.8 rule 6 binds the
    composite to a deterministic serialisation, so the
    composite canonicalises what
    :func:`construct_transformer_delta` returns rather than
    inheriting the variance. Authored deltas are untouched:
    their serialisation is pre-existing landed behaviour,
    outside this slice's scope.
    """
    (hlen,) = struct.unpack("<Q", blob_bytes[:8])
    header = json.loads(blob_bytes[8:8 + hlen])
    meta = header.get("__metadata__") or {}
    canon = {"__metadata__": {k: meta[k] for k in sorted(meta)}}
    for key in sorted(k for k in header if k != "__metadata__"):
        canon[key] = header[key]
    raw = json.dumps(canon, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    if len(raw) > hlen:  # cannot happen for a key permutation
        raise SquashError(
            "canonicalised blob header exceeds the original "
            "header length — refusing to emit a shifted blob")
    raw = raw + b" " * (hlen - len(raw))
    return blob_bytes[:8] + raw + blob_bytes[8 + hlen:]


def build_composite(bundle_dir: str, manifest: dict, segment: list, *,
                    delta_id: str, self_edit_id: str):
    """Build the §3.8 composite (record, blob_bytes) for a segment.

    Exact carrying, per tensor: every member acting on the
    tensor contributes its folded update; the carried factors
    are the members' factor blocks concatenated in segment
    order, each block's scale folded into its B, stored at the
    exact-carrying rank (the sum of the members' ranks) with
    ``alpha = rank`` (scale 1) — so the composite's folded
    update equals the segment's combined folded update in
    exact arithmetic, and all float discrepancy is left to
    the §5 verify. For the common one-member-per-tensor case
    the factors are the member's, verbatim (scale 1 members
    pass through float64 scaling bit-exactly). Serialisation
    is the format's deterministic one (targets in segment
    order, canonical metadata), so the blob hash is content
    addressing over a canonical form (§3.8 rule 6).
    Construction self-validates (SI-9a): an invalid pair is
    never emitted.
    """
    targets_spec = []
    for name in distinct_tensors(segment):
        b_blocks, a_blocks = [], []
        for d in segment:
            for t in d.get("targets") or []:
                if not isinstance(t, dict) or t.get("name") != name:
                    continue
                if t.get("kind") != "lora":
                    raise SquashError(
                        f"transformer squash refused: segment member "
                        f"{d['id']} carries a non-LoRA target on "
                        f"{name!r}; additive members are excluded "
                        "(RFC SI-37 §3.8 rule 5)")
                parsed = parse_blob(_read_blob_bytes(bundle_dir,
                                                     d["blob"]))
                A = np.asarray(parsed.tensors[t["factor_keys"][0]],
                               dtype=np.float32)
                B = np.asarray(parsed.tensors[t["factor_keys"][1]],
                               dtype=np.float32)
                scale = float(t["alpha"]) / float(t["rank"])
                b_blocks.append(
                    (B.astype(np.float64) * scale).astype(np.float32))
                a_blocks.append(A)
        B_cat = b_blocks[0] if len(b_blocks) == 1 else np.concatenate(
            b_blocks, axis=1)
        A_cat = a_blocks[0] if len(a_blocks) == 1 else np.concatenate(
            a_blocks, axis=0)
        rank = int(B_cat.shape[1])
        targets_spec.append({"name": name, "kind": "lora",
                             "A": np.ascontiguousarray(A_cat),
                             "B": np.ascontiguousarray(B_cat),
                             "rank": rank, "alpha": rank})
    record, blob_bytes = construct_transformer_delta(
        manifest, targets_spec, delta_id=delta_id,
        parent=segment[0]["parent"], created_at=utc_now(),
        self_edit_id=self_edit_id)
    # §3.8 rule 6: the composite's serialisation is canonical
    # (see _canonical_blob_bytes). The canonical bytes are
    # the blob; the record's content address is recomputed
    # over them. Equivalence is proven, not assumed: same
    # parsed factors, same metadata targets.
    canonical = _canonical_blob_bytes(blob_bytes)
    before = parse_blob(blob_bytes)
    after = parse_blob(canonical)
    if before.metadata_targets != after.metadata_targets or \
            set(before.tensors) != set(after.tensors) or \
            any(not np.array_equal(before.tensors[k], after.tensors[k])
                for k in before.tensors):
        raise SquashError(
            "blob canonicalisation changed the composite's "
            "semantics — refusing to proceed (RFC SI-37 §3.8 "
            "rule 6)")
    record = dict(record)
    record["blob"] = "sha256:" + sha256_hex(canonical)
    record["bytes"] = len(canonical)
    return record, canonical


# ---------------------------------------------------------------------------
# The per-member axis record (§4.2) and the attestation (§4.5 check 5)
# ---------------------------------------------------------------------------

def member_axis_records(journal, segment: list) -> list:
    """Assemble each segment member's §4.2 axis record from the
    journal: the member's delta row (its seq) and its
    ``training-hyperparams`` provenance note (the cycle record
    SI-21 journals for every authored cycle). Carried whole —
    assembled from records that already exist, never
    re-derived, never defaulted: any unrecoverable field
    raises SquashError (fail closed, §4.7), because after the
    squash this record is the only place the active stack
    remembers what it replaced.
    """
    records = []
    for d in segment:
        rows = journal.backend.find("deltas", filter={"id": d["id"]},
                                    limit=1)
        if not rows or not rows[0].get("seq"):
            raise SquashError(
                f"transformer squash refused: segment member "
                f"{d['id']} has no journaled delta row with a seq — "
                "its provenance cannot be attested (RFC SI-37 "
                "§4.5 check 5), fail closed")
        prov = journal.backend.find("provenance",
                                    filter={"child": d["id"]})
        notes = [p for p in prov
                 if p.get("relation") == "training-hyperparams"]
        if len(notes) != 1:
            raise SquashError(
                f"transformer squash refused: segment member "
                f"{d['id']} has {len(notes)} training-hyperparams "
                "journal records (expected exactly 1) — its axis "
                "record is not recoverable (RFC SI-37 §4.2), "
                "fail closed")
        try:
            hp = json.loads(notes[0]["note"])
        except Exception as err:
            raise SquashError(
                f"transformer squash refused: segment member "
                f"{d['id']}'s journaled hyperparams are not "
                f"readable JSON ({err}) — its axis record is not "
                "recoverable (RFC SI-37 §4.2), fail closed") from err
        required = ("training_signal_id", "target_anchoring",
                    "composition_config", "authorship_target", "head")
        missing = [f for f in required if not hp.get(f)]
        if missing:
            raise SquashError(
                f"transformer squash refused: segment member "
                f"{d['id']}'s journaled cycle record is missing "
                f"axis field(s) {missing!r} — not recoverable, "
                "never defaulted (RFC SI-37 §4.2), fail closed")
        cfg = hp["composition_config"]
        if cfg == "movement-composition":
            params = {"lambda": hp.get("composition_lambda")}
        elif cfg == "worst-input-composition":
            params = {"lambda_w": hp.get("composition_lambda_w"),
                      "k": hp.get("composition_worst_k")}
        elif cfg == "train-partition-projection":
            params = {"alpha": hp.get("projection_alpha")}
        elif cfg == "none":
            params = None
        else:
            raise SquashError(
                f"transformer squash refused: segment member "
                f"{d['id']}'s recorded composition configuration "
                f"{cfg!r} is not one of the four recorded values — "
                "its axis record cannot be carried (RFC SI-37 "
                "§4.2), fail closed")
        if params is not None and any(v is None for v in params.values()):
            raise SquashError(
                f"transformer squash refused: segment member "
                f"{d['id']}'s recorded composition parameters for "
                f"{cfg!r} are incomplete ({params!r}) — not "
                "recoverable, never defaulted (RFC SI-37 §4.2), "
                "fail closed")
        records.append({
            "delta_id": d["id"],
            "seq": rows[0]["seq"],
            "signal_id": hp["training_signal_id"],
            "anchoring": hp["target_anchoring"],
            "composition_config": cfg,
            "composition_params": params,
            "authorship_target": hp["authorship_target"],
            "targets": [{"name": t["name"], "kind": t["kind"],
                         "rank": t.get("rank"),
                         "norm": t["target_norm"]}
                        for t in d.get("targets") or []],
            "blob": d["blob"],
            "head_trained_against": hp["head"],
            "constructed": cfg == "train-partition-projection",
        })
    return records


def build_attestation(segment: list, member_records: list) -> dict:
    """The squash attestation the context offers the gate
    (§4.5 check 5): segment ids and journal seqs in segment
    order, the distinct tensors, the per-tensor exact-carrying
    ranks (sum of member ranks; None where a member is
    additive — priced as an exclusion at the gate), and the
    per-member axis records (without their seqs, which are
    carried by ``source_seqs``). Every value is recomputed by
    the gate from the manifest before it prices anything."""
    ranks: dict = {}
    for d in segment:
        for t in d.get("targets") or []:
            name = t["name"]
            if name not in ranks:
                ranks[name] = 0
            if t.get("kind") == "lora":
                ranks[name] += int(t["rank"])
            else:
                ranks[name] = None
    return {
        "segment_ids": [d["id"] for d in segment],
        "source_seqs": [mr["seq"] for mr in member_records],
        "distinct_tensors": distinct_tensors(segment),
        "per_tensor_exact_ranks": ranks,
        "member_records": [
            {k: v for k, v in mr.items() if k != "seq"}
            for mr in member_records],
    }


# ---------------------------------------------------------------------------
# The post-squash manifest view (§5.4 heads, §4.3 kept tail)
# ---------------------------------------------------------------------------

def post_squash_view(manifest: dict, segment: list,
                     composite_record: dict) -> dict:
    """An in-memory manifest view whose resolved chain is the
    post-squash stack: the composite where the segment stood,
    the kept tail after it (its first member's parent copy
    re-anchored onto the composite). The verified manifest
    itself is never mutated, re-headed, or written (the SI-35
    ``weights_at_head`` precedent): segment members are simply
    unreachable from the head in this view. One view-only
    fiction, stated plainly: when the segment contains the
    head itself (a full-segment squash, kept tail empty), the
    VIEW's head is the composite — the post-squash stack's
    top — because the view exists to materialise that stack;
    the real manifest's head id does not move (see
    :func:`run_transformer_squash`)."""
    seg_ids = {d["id"] for d in segment}
    view = dict(manifest)
    deltas = []
    for d in manifest["deltas"]:
        if d["id"] in seg_ids:
            continue
        c = dict(d)
        if c.get("parent") == segment[-1]["id"]:
            c["parent"] = composite_record["id"]
        deltas.append(c)
    deltas.append(dict(composite_record))
    view["deltas"] = deltas
    if view.get("head") in seg_ids:
        view["head"] = composite_record["id"]
    return view


# ---------------------------------------------------------------------------
# The §5 tolerance verify (a measurement, never a verdict)
# ---------------------------------------------------------------------------

def verify_transformer_squash(bundle_dir: str, manifest: dict,
                              segment: list, composite_record: dict,
                              tol: float = VERIFY_TOL,
                              policy: dict | None = None) -> dict:
    """RFC SI-37 §5: verify a built composite against the
    pre-squash stack, before the gate and before any commit.

    Weight-space part: per named tensor (the segment's
    distinct tensors), the max-abs difference between the
    post-squash stack's folded weight and the pre-squash
    stack's, assembled through the existing apply path from
    in-memory manifest views (float64 comparison).

    Behavioural part: both stacks forwarded over the four
    allowed populations ONLY — signal 1's and signal 2's
    recorded train (32) and held-out (16) partitions, as
    constructed by ``si.tiny_training.signal_sequences``.
    No probe input, probe logit, or probe-derived quantity
    is read anywhere in this function (invariant 9): a
    bundle copy with a perturbed probe artifact produces a
    byte-identical verify record, and the function writes
    nothing — bundle state is byte-identical before and
    after a verify measurement. The pass/fail statistic is
    the maximum, over all four populations, all sequences,
    and all positions, of the per-position max-abs logit
    difference; each population's last-position maximum is
    reported beside it, never as the statistic.

    Returns the verify record (published with the squash's
    journaled report whatever the outcome, §5.6). Raises
    SquashError when the verify cannot run at all — a
    population that cannot be assembled or a head that
    cannot be materialised is unverifiable, and unverifiable
    is refused, never passed (invariant 5).
    """
    from si.tiny_training import (
        TRAINING_SIGNAL_ID,
        TRAINING_SIGNAL_ID_2,
        signal_sequences,
    )
    from si.tiny_transformer import assemble_weights, forward

    try:
        post_view = post_squash_view(manifest, segment, composite_record)
        pre_weights = assemble_weights(bundle_dir, manifest, policy)
        post_weights = assemble_weights(bundle_dir, post_view, policy)
    except Exception as err:
        raise SquashError(
            f"transformer squash verify could not materialise the "
            f"pre/post-squash heads ({type(err).__name__}: {err}) — "
            "unverifiable is refused (RFC SI-37 §5.6)") from err

    named = distinct_tensors(segment)
    weight_diffs: dict = {}
    finite = True
    for name in named:
        diff = float(np.max(np.abs(
            post_weights[name].astype(np.float64)
            - pre_weights[name].astype(np.float64))))
        weight_diffs[name] = diff
        finite = finite and bool(np.isfinite(diff))

    populations = []
    statistic = 0.0
    for sid in (TRAINING_SIGNAL_ID, TRAINING_SIGNAL_ID_2):
        try:
            train, held_out, _ = signal_sequences(sid)
        except Exception as err:
            raise SquashError(
                f"transformer squash verify could not assemble "
                f"signal {sid!r}'s recorded partitions "
                f"({type(err).__name__}: {err}) — unverifiable is "
                "refused (RFC SI-37 §5.6)") from err
        for partition, seqs in (("train", train),
                                ("held_out", held_out)):
            last_max = 0.0
            all_max = 0.0
            for seq in seqs:
                pre_logits = forward(seq, pre_weights)
                post_logits = forward(seq, post_weights)
                if not (np.all(np.isfinite(pre_logits))
                        and np.all(np.isfinite(post_logits))):
                    finite = False
                d = np.abs(post_logits.astype(np.float64)
                           - pre_logits.astype(np.float64))
                all_max = max(all_max, float(d.max()))
                last_max = max(last_max, float(d[-1].max()))
            populations.append({
                "signal_id": sid, "partition": partition,
                "sequences": len(seqs),
                "last_position_max_abs": last_max,
                "all_positions_max_abs": all_max})
            statistic = max(statistic, all_max)

    weight_max = max(weight_diffs.values()) if weight_diffs else 0.0
    passed = bool(finite
                  and (weight_max <= tol)
                  and (statistic <= tol))
    return {
        "kind": "transformer-squash-verify",
        "tolerance": tol,
        "weight_space_max_abs_by_tensor": weight_diffs,
        "weight_space_max_abs": weight_max,
        "behavioural_statistic_all_positions_max_abs": statistic,
        "populations": populations,
        "finite": finite,
        "passed": passed,
        "populations_note": (
            "the two signals' recorded train/held-out partitions "
            "only (RFC SI-37 §5.3); no probe input, probe logit, "
            "or probe-derived quantity enters this verify"),
    }


# ---------------------------------------------------------------------------
# The gate context (§4)
# ---------------------------------------------------------------------------

class TransformerSquashContext:
    """The gate's BundleContext for a transformer squash
    composite (RFC SI-37 §4), in demo ``SquashContext``'s
    shape, on the miniature probe harness:

    - probes come from the mind (the bundle's bound probe
      artifact, via :class:`si.probes.ProbeContext` — the
      §6.2 binding and §6.3 startup validation are the probe
      context's own, unchanged);
    - the capability baseline is measured on the PRE-squash
      stack (a probe context over the verified manifest);
    - ``forward_candidate`` runs the POST-squash stack (a
      probe context over the post-squash manifest view; the
      candidate IS the composite, already first in that
      stack, exactly as demo squash's context forwards its
      post stack);
    - ``stack_depth`` reports the kept tail's length — the
      pre-candidate depth — so the gate's +1 is the true
      post-squash depth (§4.6);
    - ``squash_attestation()`` offers the §4.5 check-5
      attestation, which the gate verifies against the
      verified manifest before pricing anything from it.
    """

    #: Markers the gate's transformer branch looks for
    #: (duck-typed, as with ProbeContext): probe evaluation
    #: and composite pricing.
    probe_harness = True
    squash_harness = True

    def __init__(self, bundle_dir: str, pre_manifest: dict,
                 post_view: dict, attestation: dict, depth: int,
                 policy: dict | None = None):
        self.bundle_dir = bundle_dir
        self.manifest = pre_manifest
        self._pre = ProbeContext(bundle_dir, pre_manifest,
                                 policy=policy)
        self._post = ProbeContext(bundle_dir, post_view,
                                  policy=policy)
        self._attestation = attestation
        self._depth = depth

    # -- context plumbing ---------------------------------------------------
    def base_manifest(self):
        return self.manifest

    def squash_attestation(self):
        return self._attestation

    def stack_depth(self) -> int:
        return self._depth

    def probe_status(self):
        return self._post.probe_status()

    @property
    def probe_artifact_hash(self):
        return self._post.probe_artifact_hash

    # -- BundleContext protocol ---------------------------------------------
    def identity_probes(self):
        return self._post.identity_probes()

    def capability_suite(self):
        return self._post.capability_suite()

    def capability_targets(self):
        return self._post.capability_targets()

    def echo_floor(self):
        return self._post.echo_floor()

    def capability_baseline(self):
        """Measured on the PRE-squash stack (§4.1/§4.5)."""
        return self._pre.capability_baseline()

    def forward_candidate(self, x, candidate):
        """Last-position logits of the POST-squash stack."""
        return self._post.forward_head(x)

    def candidate_measurability(self, candidate):
        return self._post.candidate_measurability(candidate)


# ---------------------------------------------------------------------------
# The pipeline (§4.7)
# ---------------------------------------------------------------------------

def _remove_blob_file(bundle_dir: str, blob_ref: str) -> None:
    path = blob_path(bundle_dir, blob_ref)
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def run_transformer_squash(bundle_dir: str, manifest: dict,
                           chain: list, S: list, K: list, *,
                           keep: int, tol: float = VERIFY_TOL,
                           gate: Gate | None = None) -> dict:
    """The transformer branch of ``si squash`` (RFC SI-37 §§4–5).

    Called by :func:`si.squash.squash_bundle` only when the
    resolved active chain is entirely transformer-delta-v1.
    Refusals raise SquashError before any blob, journal, or
    manifest write. Gate REJECT/ESCALATE and ADMIT follow
    §4.7's journaling exactly (see the module docstring).
    """
    # -- preconditions, before any write (§4.7) ------------------------------
    if len(S) < 2:
        raise SquashError(
            f"transformer squash refused: the squash segment has "
            f"{len(S)} delta(s) — a segment of one is not a "
            "compression, and a no-op mislabelled as a squash is "
            "refused rather than performed (RFC SI-37 §4.7)")
    for d in S:
        for t in d.get("targets") or []:
            if isinstance(t, dict) and t.get("kind") == "additive":
                raise SquashError(
                    f"transformer squash refused: segment member "
                    f"{d['id']} carries an additive target on "
                    f"{t.get('name')!r} — additive members are "
                    "excluded from squash composites (RFC SI-37 "
                    "§3.8 rule 5), fail closed")
    if not is_miniature_probe_bundle(manifest):
        raise SquashError(
            "transformer squash refused: this bundle is not the "
            "miniature probe bundle, so the RFC SI-37 §5 verify "
            "has no lawful population for it (§5.8) and checks "
            "3/4 could not be measured — unverifiable is "
            "refused, fail closed, and no forward path is faked")

    policy = load_policy(bundle_dir)
    journal = open_journal(bundle_dir, manifest)
    try:
        # Axis records + source seqs: journal reads only; a
        # refusal here has written nothing anywhere.
        member_records = member_axis_records(journal, S)
        source_seqs = [mr["seq"] for mr in member_records]
        attestation = build_attestation(S, member_records)

        delta_id = _next_delta_id(manifest)
        record, blob_bytes = build_composite(
            bundle_dir, manifest, S, delta_id=delta_id,
            self_edit_id="si38-squash-uncommitted")
        blob_ref = "sha256:" + sha256_hex(blob_bytes)
        blob_preexisted = os.path.isfile(blob_path(bundle_dir, blob_ref))
        _write_blob(bundle_dir, blob_bytes)  # content-addressed

        # -- §5 verify, before the gate and before any commit ---------------
        try:
            verify = verify_transformer_squash(
                bundle_dir, manifest, S, record, tol=tol,
                policy=policy)
        except SquashError:
            if not blob_preexisted:
                _remove_blob_file(bundle_dir, blob_ref)
            raise
        if not verify["passed"]:
            if not blob_preexisted:
                _remove_blob_file(bundle_dir, blob_ref)
            raise SquashError(
                f"transformer squash refused: tolerance verify "
                f"failed — weight-space max abs "
                f"{verify['weight_space_max_abs']:.6e}, "
                f"behavioural statistic (max over all positions, "
                f"four allowed populations) "
                f"{verify['behavioural_statistic_all_positions_max_abs']:.6e}, "
                f"tolerance {tol:.1e}; the gate never sees a "
                "candidate whose verify failed, and nothing is "
                "journaled (RFC SI-37 §5.6)")

        # -- gate: the full five checks in the transformer mapping ----------
        post_view = post_squash_view(manifest, S, record)
        ctx = TransformerSquashContext(
            bundle_dir, manifest, post_view, attestation,
            depth=len(K), policy=policy)
        intent = Intent(
            kind=TRANSFORMER_DELTA_KIND,
            reason=f"squash {len(S)} deltas into one (keep={keep})",
            parents=[d["id"] for d in S],
            source_seqs=source_seqs,
            targets=distinct_tensors(S),
        )
        candidate = {"record": record, "blob_bytes": blob_bytes,
                     "norm": float(record["norm"]),
                     "member_records": attestation["member_records"]}
        gate = gate or Gate(policy)
        report = gate.evaluate(intent, candidate, ctx)

        if report.verdict in ("REJECT", "ESCALATE"):
            # Journaled exactly as the loop journals them; the
            # §5 verify record is journaled beside the verdict
            # (§5.6). The candidate blob stays as evidence.
            edit_id = journal.record_self_edit(
                TRANSFORMER_DELTA_KIND, intent.to_dict(),
                intent.reason, report.verdict, report.to_dict())
            journal.record_provenance(
                child=edit_id, parents=[d["id"] for d in S],
                relation="transformer-squash-verify",
                note=json.dumps(verify, sort_keys=True))
            journal.set_edit_result(
                edit_id,
                "rejected" if report.verdict == "REJECT" else "escalated")
            _save_and_anchor(
                bundle_dir, journal, manifest,
                note=f"{report.verdict.lower()} {edit_id}")
            return {"squashed": False, "verdict": report.verdict,
                    "edit_id": edit_id, "verify": verify,
                    "gate_report": report.to_dict()}

        # -- ADMIT: journal, mark superseded, append, re-anchor -------------
        edit_id = journal.record_self_edit(
            TRANSFORMER_DELTA_KIND, intent.to_dict(), intent.reason,
            "ADMIT", report.to_dict())
        journal.record_provenance(
            child=delta_id, parents=[d["id"] for d in S],
            relation="squashed_from",
            note=(f"squash of {len(S)} deltas into {delta_id} "
                  f"(keep={keep})")[:200])
        journal.record_provenance(
            child=delta_id, parents=[edit_id],
            relation="transformer-squash-verify",
            note=json.dumps(verify, sort_keys=True))
        delta_rec = copy.deepcopy(record)
        delta_rec["self_edit_id"] = edit_id
        journal.record_transformer_delta(
            delta_id, delta_rec["parent"], delta_rec["blob"],
            delta_rec["bytes"], float(delta_rec["norm"]), edit_id,
            list(source_seqs), delta_rec["targets"])
        journal.set_edit_result(edit_id, "materialized",
                                delta_id=delta_id)

        seg_ids = {d["id"] for d in S}
        for d in manifest["deltas"]:
            if d["id"] in seg_ids:
                d["superseded_by"] = delta_id
        manifest["deltas"].append(delta_rec)
        # Head id deliberately UNCHANGED (module docstring).
        manifest = _save_and_anchor(
            bundle_dir, journal, manifest,
            note=f"squash {len(S)}->{delta_id} (keep={keep})")
        return {"squashed": True, "delta_id": delta_id,
                "edit_id": edit_id, "verdict": "ADMIT",
                "head": manifest["head"], "verify": verify,
                "gate_report": report.to_dict()}
    finally:
        journal.close()


__all__ = [
    "VERIFY_TOL",
    "TransformerSquashContext",
    "build_attestation",
    "build_composite",
    "distinct_tensors",
    "member_axis_records",
    "post_squash_view",
    "run_transformer_squash",
    "verify_transformer_squash",
]
