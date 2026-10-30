"""Gaia B-4: the Hearth interface (SI-41 RFC §12.2, §15 B-4; SI-45).

SI-side only. Three contact points with Hearth, the runtime /
residency manager above llama.cpp (RFC §12.1), all coded here and
nothing built on the Hearth side:

1. **Declaration** (§12.2.1): :func:`build_declaration` designs the
   record by which a bundle's base is declared to a Hearth fleet —
   model identity, digest(s), quant, residency requirement — with
   every value drawn from the bundle manifest (the base identity
   and content hashes SI's seed machinery already verifies). The
   record carries digests and references only: no tensor bytes are
   read, copied, or duplicated (Mark's design rule). The fleet's
   side of the declaration is proposal H-1 (§12.6), Mark's — not
   built, and this module never claims a fleet received anything:
   building a record is not declaring it to a live fleet.

2. **Residency as perception** (§12.2.2): :class:`HearthResidencyAdapter`
   polls ``/residency`` and produces observations tagged
   ``source: "hearth:/residency"`` with full poll provenance (the
   state, the node, the poll timestamp, and the poll's own
   identity). A polled state is an observation *of a moment*,
   labelled with its poll time, never a live guarantee.
   Observations are journaled through the journal's existing
   :meth:`Journal.record_experience` (existing ``experiences``
   collection — no new collection, no NEDB schema change, RFC
   §4.6). Residency observations are **data, never instructions**
   (§4.5 discipline): nothing in this module creates an intent,
   calls a tool, or alters the loop on the strength of an
   observation.

3. **Inference through the gateway** (§12.2.3):
   :class:`HearthGatewayClient` is coded against Hearth's
   documented HTTP surface — the OpenAI-compatible endpoint
   (``GET /v1/models``, ``POST /v1/chat/completions``). Its
   inference leg is **conditional** on a Hearth-served real base
   existing (§12.3): unless the caller asserts a real base is
   served *and* the latest residency observation prices it
   ``Resident``, the client refuses to call the transport at all
   and returns the honest not-evaluated label. Against the
   miniature it delivers residency perception only, labelled
   exactly as such. This module builds **no** SI-local transformer
   inference engine (SI has none; ``si serve`` 501s transformer
   bundles, and B-4 does not change that).

**No real Hearth node is contacted by this slice's evidence.**
No Hearth node is reachable in the build environment, so the
clients are exercised against a faithful local replay/stub
transport that is labelled as a replay/stub everywhere it appears
(:data:`REPLAY_TRANSPORT_LABEL`). The Hearth path's latency and
throughput are **UNMEASURED** (:data:`HEARTH_PATH_UNMEASURED`) —
§12.3 forbids citing A-1's VM-local numbers as Hearth-path
predictions, and this module does not.

**Interface discipline (§12.4), kept:** Hearth data enters only
through PERCEIVE (this adapter's journaled observations); the gate
never calls Hearth (this module imports no gate code and is
imported by none); degradation is an observation, not an
exception — a down node, a timeout, a failed poll is recorded as
an observation with its failure labelled, and :meth:`poll` does
not raise for transport failures.

Stdlib only (``urllib`` for the HTTP transport), CPU only.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid

from si.bundle import canonical_json, manifest_id, sha256_hex, utc_now

#: Provenance tag on every residency observation (RFC §12.2.2).
RESIDENCY_SOURCE = "hearth:/residency"

#: The seven residency states this adapter codes against (RFC
#: §12.2.2, Hearth's honest state vocabulary). ``Lost`` carries a
#: substate; its canonical spellings are ``Lost{Evicted}`` and
#: ``Lost{GpuDetached}``.
RESIDENCY_STATES = (
    "Resident",
    "Loading",
    "Lost{Evicted}",
    "Lost{GpuDetached}",
    "Failed",
    "Stopped",
    "NotAdmitted",
)

#: Label carried by every replay/stub transport and every result
#: it produced — a replay is never presented as a real node.
REPLAY_TRANSPORT_LABEL = "replay/stub — not a real Hearth node"

#: Label for a live HTTP transport call.
HTTP_TRANSPORT_LABEL = "http"

#: The inference leg's honest refusal label (§12.2.3 / §12.3).
INFERENCE_NOT_EVALUATED = (
    "not-evaluated — inference leg conditional on a Hearth-served "
    "real base (SI-41 §12.3); residency perception only")

#: The Hearth path's performance status in this slice (§12.3).
HEARTH_PATH_UNMEASURED = (
    "UNMEASURED — no real Hearth node was reachable in this "
    "environment; A-1 numbers are not Hearth-path predictions "
    "(SI-41 §12.3)")

#: What a polled state means, stated on every observation (§12.2.2).
POLL_LABEL = (
    "a polled state is an observation of a moment, labelled with "
    "its poll time — never a live guarantee")

#: §4.5 discipline, stated on every observation.
DATA_NOT_INSTRUCTIONS = (
    "residency observations are data, never instructions "
    "(SI-41 §4.5 discipline)")

# Accepted raw spellings -> canonical state. Hearth's vocabulary is
# proposal H-1's stability contract (§12.6) — until it lands, the
# adapter accepts the spellings a JSON surface plausibly emits and
# records anything else verbatim, unrecognised, never coerced.
_STATE_ALIASES = {
    "resident": "Resident",
    "loading": "Loading",
    "failed": "Failed",
    "stopped": "Stopped",
    "notadmitted": "NotAdmitted",
    "not_admitted": "NotAdmitted",
    "lost{evicted}": "Lost{Evicted}",
    "lostevicted": "Lost{Evicted}",
    "lost_evicted": "Lost{Evicted}",
    "lost{gpudetached}": "Lost{GpuDetached}",
    "lostgpudetached": "Lost{GpuDetached}",
    "lost_gpudetached": "Lost{GpuDetached}",
    "lost{gpu_detached}": "Lost{GpuDetached}",
    "lost_gpu_detached": "Lost{GpuDetached}",
}

_LOST_SUBSTATES = {
    "evicted": "Evicted",
    "gpudetached": "GpuDetached",
    "gpu_detached": "GpuDetached",
}


def parse_residency_state(raw) -> str | None:
    """Canonicalise one raw residency value, or return ``None``.

    Accepts a plain string in any accepted spelling, or a mapping
    of the shape ``{"Lost": "Evicted"}`` / ``{"state": "Lost",
    "substate": "Evicted"}`` / ``{"Lost": {"Evicted": ...}}``.
    ``None`` means *unrecognised*: callers record the raw value
    verbatim with the parse failure labelled — an unknown state is
    priced honestly, never guessed into a known one.
    """
    if isinstance(raw, dict):
        state = raw.get("state", raw.get("residency", raw.get("status")))
        sub = raw.get("substate", raw.get("reason"))
        if state is None:
            for key, val in raw.items():
                if str(key).strip().lower() == "lost":
                    if isinstance(val, dict) and val:
                        sub = next(iter(val))
                    else:
                        sub = val
                    state = "Lost"
                    break
        if state is None:
            return None
        raw = str(state) if sub is None else f"{state}{{{sub}}}"
    if not isinstance(raw, str):
        return None
    key = raw.strip().lower().replace(" ", "").replace("-", "_")
    if key in _STATE_ALIASES:
        return _STATE_ALIASES[key]
    if key.startswith("lost"):
        sub_key = key[4:].strip("{}_").lower()
        sub = _LOST_SUBSTATES.get(sub_key)
        if sub is not None:
            return f"Lost{{{sub}}}"
    return None


# ---------------------------------------------------------------------------
# 1. The declaration record (§12.2.1)
# ---------------------------------------------------------------------------

def _base_digests(base: dict) -> list:
    """Every content digest the manifest records for the base.

    Drawn from the three reference shapes SI's loader verifies —
    the single ``base.blob``, whole-file ``base.parts[*]``, and
    per-tensor ``base.tensor_blobs`` — deduplicated by digest, in
    manifest order. Digests only: no bytes are read.
    """
    out: list = []
    seen: set = set()

    def add(digest, *, ref, bytes_=None, file=None):
        if not digest or digest in seen:
            return
        seen.add(digest)
        row = {"ref": ref, "sha256": digest}
        if bytes_ is not None:
            row["bytes"] = bytes_
        if file is not None:
            row["file"] = file
        out.append(row)

    add(base.get("blob"), ref="base.blob", bytes_=base.get("bytes"))
    for part in base.get("parts") or []:
        if isinstance(part, dict):
            add(part.get("sha256") or part.get("blob"),
                ref="base.parts", bytes_=part.get("bytes"),
                file=part.get("file"))
    tensor_blobs = base.get("tensor_blobs")
    if isinstance(tensor_blobs, dict):
        for tname in sorted(tensor_blobs):
            tref = tensor_blobs[tname]
            if isinstance(tref, dict):
                add(tref.get("blob") or tref.get("sha256"),
                    ref=f"base.tensor_blobs[{tname}]")
    return out


def _base_dtype(base: dict) -> str | None:
    if base.get("dtype"):
        return base["dtype"]
    cfg = base.get("checkpoint_config")
    if isinstance(cfg, dict):
        return cfg.get("torch_dtype") or cfg.get("dtype")
    return None


def _base_quant(base: dict) -> dict:
    """The quant story, exactly as strong as the manifest's record.

    No SI bundle manifest records a GGUF quant for its base today
    (bases are referenced in their source safetensors format), so
    the honest default is an explicit *none recorded* — never an
    invented quant name.
    """
    quant = base.get("quant") or base.get("quantization")
    if quant:
        return {"value": quant,
                "label": "recorded in the bundle manifest's base record"}
    return {
        "value": None,
        "label": ("no quantisation recorded in the bundle manifest — "
                  "the base is referenced in its source format and "
                  "dtype; no GGUF quant is claimed"),
    }


def build_declaration(manifest: dict, *,
                      residency_requirement: dict | None = None) -> dict:
    """Build the §12.2.1 declaration record from a bundle manifest.

    The record is the deliverable of B-4's declaration leg: it is
    designed, digestible, and validatable here; *sending* it is a
    fleet-side act (proposal H-1) that this slice does not perform
    and does not claim. ``declaration_digest`` is the sha256 of the
    record's canonical form without the digest and timestamp
    fields, so the same manifest + requirement always produces the
    same digest.
    """
    base = manifest.get("base") or {}
    digests = _base_digests(base)
    cfg = base.get("checkpoint_config")
    architecture = None
    if isinstance(cfg, dict):
        architecture = {
            "architectures": cfg.get("architectures"),
            "model_type": cfg.get("model_type"),
        }
    referenced = base.get("deduplicated_bytes")
    if referenced is None:
        referenced = base.get("imported_bytes")
    if referenced is None:
        parts = base.get("parts") or []
        if parts:
            referenced = sum(p.get("bytes") or 0 for p in parts
                             if isinstance(p, dict))
    record = {
        "record": "si.gaia.hearth.declaration",
        "version": 1,
        "declared_at": utc_now(),
        "bundle": {
            "name": manifest.get("name"),
            "bundle_id": manifest_id(manifest),
            "si_version": manifest.get("si_version"),
            "head": manifest.get("head"),
        },
        "model": {
            "identity": base.get("source"),
            "revision": base.get("revision"),
            "digest": digests[0]["sha256"] if digests else None,
            "digests": digests,
            "format": base.get("format"),
            "dtype": _base_dtype(base),
            "quant": _base_quant(base),
            "architecture": architecture,
            "arch_source": base.get("arch_source"),
            "seed_format": base.get("seed_format"),
            "by_reference": base.get("seed_format") is not None
                            or (base.get("bytes") == 0 and bool(digests)),
            "bytes": {
                "stored_in_bundle": base.get("bytes"),
                "referenced": referenced,
                "deduplicated_bytes": base.get("deduplicated_bytes"),
            },
        },
        "residency_requirement": dict(residency_requirement) if
            residency_requirement is not None else {
                "state": "Resident",
                "label": ("declared requirement — a request to the "
                          "fleet, not a guarantee and not an "
                          "admission; the fleet side is proposal "
                          "H-1 (SI-41 §12.6), not built"),
            },
        "fleet_side": ("not built — proposal H-1 (SI-41 §12.6); this "
                       "record is the SI-side declaration only, and "
                       "building it does not declare anything to a "
                       "live fleet"),
        "supply_path": ("hearth-pull digest-verified fetch is the "
                        "natural supply path (§12.2.1); the digests "
                        "above are the digests SI's seed machinery "
                        "already verifies"),
    }
    digest_payload = {k: v for k, v in record.items()
                      if k not in ("declared_at",)}
    record["declaration_digest"] = (
        "sha256:" + sha256_hex(canonical_json(digest_payload)))
    return record


def validate_declaration(record: dict) -> list:
    """Fail-closed validation of a declaration record.

    Returns a list of problems (empty = valid). A declaration with
    no identity or no digest is not a declaration — the fleet could
    not pull or verify anything from it — so both are required, and
    every digest must carry the ``sha256:`` label SI's machinery
    verifies.
    """
    problems: list = []
    if record.get("record") != "si.gaia.hearth.declaration":
        problems.append("record kind is not si.gaia.hearth.declaration")
    bundle = record.get("bundle") or {}
    if not bundle.get("bundle_id", "").startswith("si1:"):
        problems.append("bundle.bundle_id missing or not an si1: id")
    model = record.get("model") or {}
    if not model.get("identity"):
        problems.append("model.identity missing (manifest base.source)")
    digests = model.get("digests") or []
    if not digests:
        problems.append("model.digests empty — nothing verifiable declared")
    for row in digests:
        if not str(row.get("sha256", "")).startswith("sha256:"):
            problems.append(f"digest for {row.get('ref')!r} lacks sha256: label")
    if model.get("digest") and digests and \
            model["digest"] != digests[0]["sha256"]:
        problems.append("model.digest does not match the first recorded digest")
    if not (record.get("residency_requirement") or {}).get("state"):
        problems.append("residency_requirement.state missing")
    payload = {k: v for k, v in record.items()
               if k not in ("declared_at", "declaration_digest")}
    expect = "sha256:" + sha256_hex(canonical_json(payload))
    if record.get("declaration_digest") != expect:
        problems.append("declaration_digest does not match the record body")
    return problems


def declaration_from_bundle(bundle_dir: str) -> dict:
    """Load (and thereby verify) a bundle, then build its record.

    Uses :func:`si.bundle.load_bundle`, so a tampered bundle fails
    closed here exactly as it does everywhere else — B-4 declares
    only what SI's existing verification has already checked.
    """
    from si.bundle import load_bundle
    return build_declaration(load_bundle(bundle_dir))


# ---------------------------------------------------------------------------
# Transports: real HTTP, and a labelled replay/stub
# ---------------------------------------------------------------------------

class HttpTransport:
    """The real transport: stdlib ``urllib`` against a base URL.

    Coded against Hearth's documented HTTP surface. Constructing it
    contacts nothing; each call is one HTTP request. Transport
    failures raise, and the clients convert them into labelled
    observations/results (§12.4) — this class does not hide them.
    """

    label = HTTP_TRANSPORT_LABEL
    is_replay = False

    def __call__(self, method: str, url: str, *,
                 json_body: dict | None = None,
                 timeout_s: float = 5.0) -> tuple[int, bytes]:
        data = None
        headers = {"Accept": "application/json"}
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers,
                                     method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()


class ReplayTransport:
    """A faithful local replay/stub transport — labelled as such.

    Fed a scripted list of responses (``(status, body)`` tuples,
    dict bodies, or :class:`Exception` instances to raise), it
    answers one per call, in order, and records every call it
    received. It stands in for a Hearth node **only** so the
    clients' parsing, labelling, and journaling can be exercised
    without one; every observation or result it touches carries
    :data:`REPLAY_TRANSPORT_LABEL`. Exhaustion raises, and the
    clients label that as a failed poll/call — a replay never
    silently falls through to a real network call.
    """

    label = REPLAY_TRANSPORT_LABEL
    is_replay = True

    def __init__(self, responses: list):
        self._responses = list(responses)
        self.calls: list = []

    def __call__(self, method: str, url: str, *,
                 json_body: dict | None = None,
                 timeout_s: float = 5.0) -> tuple[int, bytes]:
        self.calls.append({"method": method, "url": url,
                           "json_body": json_body})
        if not self._responses:
            raise RuntimeError(
                "replay/stub transport exhausted — no scripted "
                "response remains (labelled failure, not a retry "
                "against a real node)")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, tuple):
            status, body = item
        else:
            status, body = 200, item
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        return status, body


def _join(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path


# ---------------------------------------------------------------------------
# 2. The residency PERCEIVE adapter (§12.2.2)
# ---------------------------------------------------------------------------

def _extract_residency(payload):
    """(raw_state, node, model) from a /residency payload.

    Accepted shapes (assumptions A-1/A-2 in the slice doc, against
    proposal H-1): a single entry object, an object wrapping an
    ``entries``/``residency`` list, or a bare list. For a list, the
    first entry is priced and the full payload is preserved on the
    observation — the adapter never discards what it was told.
    """
    entry = payload
    if isinstance(payload, list):
        entry = payload[0] if payload else {}
    elif isinstance(payload, dict):
        for key in ("entries", "residency", "nodes"):
            val = payload.get(key)
            if isinstance(val, list) and val:
                entry = val[0]
                break
    if not isinstance(entry, dict):
        return entry, None, None
    raw = entry.get("state", entry.get("residency", entry.get("status")))
    if raw is None and ("Lost" in entry or "lost" in entry):
        raw = entry
    node = entry.get("node", entry.get("node_id", entry.get("host")))
    model = entry.get("model", entry.get("model_id", entry.get("identity")))
    return raw, node, model


class HearthResidencyAdapter:
    """Polls ``/residency``; produces provenance-tagged observations.

    One adapter prices one endpoint. ``transport`` is injectable so
    tests and replays exercise the identical code path as the HTTP
    transport; the default is :class:`HttpTransport`.
    """

    def __init__(self, base_url: str, *, node: str | None = None,
                 transport=None, timeout_s: float = 5.0):
        self.base_url = base_url
        self.node = node
        self.transport = transport if transport is not None \
            else HttpTransport()
        self.timeout_s = timeout_s
        self._poll_n = 0

    @property
    def transport_label(self) -> str:
        return getattr(self.transport, "label", HTTP_TRANSPORT_LABEL)

    @property
    def is_replay(self) -> bool:
        return bool(getattr(self.transport, "is_replay", False))

    def poll(self) -> dict:
        """One poll -> one observation. Never raises for a failed
        poll: failure is an observation with its failure labelled
        (§12.4, degradation is an observation, not an exception)."""
        self._poll_n += 1
        poll_id = f"hearth-poll:{self._poll_n}:{uuid.uuid4().hex[:12]}"
        obs = {
            "kind": "residency",
            "source": RESIDENCY_SOURCE,
            "state": None,
            "raw_state": None,
            "state_parse": None,
            "node": self.node,
            "model": None,
            "polled_at": utc_now(),
            "poll_id": poll_id,
            "endpoint": _join(self.base_url, "/residency"),
            "transport": self.transport_label,
            "observation_status": None,
            "failure": None,
            "http_status": None,
            "label": POLL_LABEL,
            "data_not_instructions": DATA_NOT_INSTRUCTIONS,
            "raw": None,
        }
        try:
            status, body = self.transport(
                "GET", obs["endpoint"], timeout_s=self.timeout_s)
            obs["http_status"] = status
            payload = json.loads(body.decode("utf-8"))
            obs["raw"] = payload
            raw_state, node, model = _extract_residency(payload)
            obs["raw_state"] = raw_state
            if node is not None:
                obs["node"] = node
            obs["model"] = model
            state = parse_residency_state(raw_state)
            obs["state"] = state
            if status != 200:
                obs["observation_status"] = (
                    f"poll-failed — /residency returned HTTP {status}; "
                    "any state in the body is recorded, not trusted "
                    "as a successful poll")
                obs["failure"] = f"HTTP {status}"
            elif state is None:
                obs["observation_status"] = (
                    "observed — state unrecognised, recorded "
                    "verbatim, never coerced to a known state")
                obs["state_parse"] = "unrecognised"
            else:
                obs["observation_status"] = "observed"
                obs["state_parse"] = "recognised"
            return obs
        except Exception as e:  # §12.4: degradation is an observation
            obs["observation_status"] = (
                "poll-failed — residency poll failed; the failure "
                "is the observation")
            obs["failure"] = f"{type(e).__name__}: {e}"
            obs["state_parse"] = "not-evaluated — poll failed"
            return obs

    def poll_and_journal(self, journal) -> dict:
        """Poll once and journal the observation; returns both."""
        obs = self.poll()
        ref = journal_residency_observation(journal, obs)
        return {"observation": obs, **ref}


def journal_residency_observation(journal, observation: dict) -> dict:
    """Journal one residency observation via existing methods.

    Lands in the existing ``experiences`` collection via
    :meth:`Journal.record_experience` (kind ``hearth_residency``) —
    no new collection, no schema change (RFC §4.6). Returns
    ``{"seq", "row_id"}`` read back from the journal, the pattern
    the task loop uses for its entries.
    """
    record = dict(observation)
    record["kind"] = "hearth_residency"
    row_id = journal.record_experience("hearth_residency", record, "")
    rows = journal.backend.find(
        "experiences", filter={"kind": "hearth_residency"},
        sort="-seq", limit=1)
    seq = rows[0]["seq"] if rows else journal.tip()[0]
    return {"seq": seq, "row_id": row_id}


def residency_history(journal) -> list:
    """Every journaled residency observation, oldest first."""
    rows = journal.backend.find(
        "experiences", filter={"kind": "hearth_residency"}, sort="seq")
    return [r.get("record") or {} for r in rows]


# ---------------------------------------------------------------------------
# 3. The gateway client for the inference leg (§12.2.3)
# ---------------------------------------------------------------------------

class HearthGatewayClient:
    """Client for Hearth's OpenAI-compatible gateway surface.

    Coded against the documented shape — ``GET /v1/models`` and
    ``POST /v1/chat/completions`` — and nothing else. The inference
    leg is conditional (§12.3): :meth:`chat_completion` calls the
    transport only when the caller asserts a Hearth-served real
    base exists (``served_real_base=True``) *and* supplies a
    residency state of ``Resident``; otherwise it returns the
    not-evaluated label without touching the transport, so the
    miniature path can never be misread as inference.
    """

    def __init__(self, base_url: str, *, model: str | None = None,
                 transport=None, timeout_s: float = 30.0):
        self.base_url = base_url
        self.model = model
        self.transport = transport if transport is not None \
            else HttpTransport()
        self.timeout_s = timeout_s

    @property
    def transport_label(self) -> str:
        return getattr(self.transport, "label", HTTP_TRANSPORT_LABEL)

    @property
    def is_replay(self) -> bool:
        return bool(getattr(self.transport, "is_replay", False))

    def list_models(self) -> dict:
        """``GET /v1/models`` — labelled result, failure as data."""
        url = _join(self.base_url, "/v1/models")
        result = {
            "operation": "list_models",
            "endpoint": url,
            "transport": self.transport_label,
            "hearth_path_performance": HEARTH_PATH_UNMEASURED,
        }
        try:
            status, body = self.transport(
                "GET", url, timeout_s=self.timeout_s)
            result["http_status"] = status
            payload = json.loads(body.decode("utf-8"))
            data = payload.get("data") if isinstance(payload, dict) \
                else payload
            result["status"] = "observed" if status == 200 \
                else f"failed — gateway returned HTTP {status}"
            result["models"] = data
            return result
        except Exception as e:  # §12.4: degradation is data here too
            result["status"] = (
                "failed — gateway unreachable; the failure is the "
                "result, not an exception")
            result["failure"] = f"{type(e).__name__}: {e}"
            result["models"] = None
            return result

    def chat_completion(self, messages: list, *,
                        residency_state: str | None = None,
                        served_real_base: bool = False,
                        model: str | None = None) -> dict:
        """The conditional inference leg (§12.2.3, §12.3).

        Refuses — without calling the transport — unless a real
        base is asserted served *and* residency prices it Resident.
        A successful call's content is returned with its transport
        labelled; a replay-produced completion is labelled a
        replay, never a real base's output.
        """
        url = _join(self.base_url, "/v1/chat/completions")
        result = {
            "operation": "chat_completion",
            "endpoint": url,
            "transport": self.transport_label,
            "transport_called": False,
            "hearth_path_performance": HEARTH_PATH_UNMEASURED,
            "residency_state_supplied": residency_state,
            "served_real_base_asserted": bool(served_real_base),
        }
        if not served_real_base or residency_state != "Resident":
            result["status"] = INFERENCE_NOT_EVALUATED
            result["inference"] = INFERENCE_NOT_EVALUATED
            result["content"] = None
            result["detail"] = (
                "no Hearth-served real base is asserted Resident; "
                "against the miniature this client delivers "
                "residency perception only, labelled as such")
            return result
        body = {"model": model or self.model, "messages": messages}
        result["transport_called"] = True
        try:
            status, raw = self.transport(
                "POST", url, json_body=body, timeout_s=self.timeout_s)
            result["http_status"] = status
            payload = json.loads(raw.decode("utf-8"))
            choices = payload.get("choices") or []
            content = None
            if choices:
                content = (choices[0].get("message") or {}).get("content")
            result["content"] = content
            result["usage"] = payload.get("usage")
            if status != 200:
                result["status"] = (
                    f"failed — gateway returned HTTP {status}")
            elif self.is_replay:
                result["status"] = (
                    "completed — via replay/stub transport "
                    f"({REPLAY_TRANSPORT_LABEL}); the completion is "
                    "a scripted replay payload, not a real base's "
                    "output")
            else:
                result["status"] = (
                    "completed — via the Hearth gateway (http)")
            return result
        except Exception as e:  # §12.4
            result["status"] = (
                "failed — gateway call failed; the failure is the "
                "result, not an exception")
            result["failure"] = f"{type(e).__name__}: {e}"
            result["content"] = None
            return result
