"""Gaia capture pre-flight template lock — SI-112 (SI-111
§3.3; lead decision D-111-2).

What this module is, stated honestly: harness-integrity
infrastructure that makes the SI-93 incident class — a
capture spent against a silently wrong chat template —
impossible to repeat unnoticed. SI-93's diagnosis stands:
the Imagine v13 GGUF carries **no** ``tokenizer.chat_template``
metadata, llama-server fell back to its built-in ChatML
template, and SI-92's 0/27 was a serving artifact, not a
model verdict. Nothing in the capture path checked which
template the server would apply, and the budget was spent
before anyone knew.

Three pieces, per SI-111 §3.3:

(a) **Candidate registration.** :func:`register_candidate`
    probes the candidate GGUF with a real GGUF metadata
    reader (never filename or size inference) and records:
    ``tokenizer.chat_template`` present or absent; if
    present, the sha256 of the embedded template; if absent,
    the sha256 of the externally supplied template the
    serving configuration will use, or ``none``. A candidate
    with neither is registered **TEMPLATE-UNRESOLVED**.

(b) **Capture pre-flight, fail-closed.** :func:`check_preflight`
    resolves the template the server will actually apply —
    the spawn path's ``--chat-template-file`` target when one
    is in play (it is what llama-server applies), else the
    served GGUF's embedded template, else a caller-supplied
    served hash — hashes it, and compares it to the
    registration record. Mismatch, absence, an unresolved
    registration, or an unresolvable serving configuration
    REFUSES the capture: no attempt is spent and the refusal
    is the run's recorded outcome (D-111-2: template-hash
    registration is a hard eligibility precondition for any
    graded or diagnostic capture; no waivers).

(c) **The record travels.** The check and its result (both
    hashes, the comparison, the decision) are returned as a
    plain dict; the capture entry points
    (:mod:`si.gaia.conformance_capture`,
    :mod:`si.gaia.multicycle`) carry it in the capture log /
    driver state and re-verify the served hash at every
    residency-guard poll (§3.3(c)), halting with no attempt
    spent if a mid-capture restart changes it.

Scope and honesty (§3.4): this slice prevents a
serving-artifact class. It changes no recorded verdict —
SI-92's INSUFFICIENT stands exactly as recorded — improves
no model score, earns no §18.1 row and no §18.3 progress,
and touches no scoring semantics: :func:`score_corpus`
behaviour for already-captured corpora is unchanged (this
module is not imported by the scoring path at all).

CPU only, stdlib only. Terminology: machine learning
infrastructure throughout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

__all__ = [
    "GGUFProbeError",
    "PreflightRefused",
    "probe_gguf_template",
    "read_template_file",
    "register_candidate",
    "resolve_served_template",
    "check_preflight",
    "require_preflight",
    "TEMPLATE_KEY",
]

#: The GGUF metadata key this lock prices.
TEMPLATE_KEY = "tokenizer.chat_template"

_SLICE = "SI-112 (SI-111 §3.3; D-111-2)"

_RULE = ("SI-111 §3.3(b) / D-111-2: template-hash "
         "registration is a hard eligibility precondition "
         "for any graded or diagnostic capture — mismatch, "
         "absence, or an unresolvable serving configuration "
         "refuses the capture, no attempt spent, no waivers")


class GGUFProbeError(RuntimeError):
    """The GGUF probe could not read the file as a GGUF with
    a well-formed metadata section. Registration fails
    loudly with this error — it never guesses a template
    from a filename, a size, or a partial parse."""


class PreflightRefused(RuntimeError):
    """The pre-flight template lock refused a capture,
    fail-closed. ``record`` carries the full pre-flight
    check record (both hashes, the comparison, the decision)
    so the refusal is itself evidence in the capture log."""

    def __init__(self, record: dict):
        self.record = record
        super().__init__(
            f"pre-flight template lock REFUSED the capture — "
            f"{record.get('reason')}")


def _now_ny() -> datetime:
    return datetime.now(ZoneInfo("America/New_York"))


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# (a) The GGUF metadata probe — a real reader
# ---------------------------------------------------------------------------

# GGUF value types (gguf-py GGUFValueType numbering).
_T_UINT8, _T_INT8, _T_UINT16, _T_INT16 = 0, 1, 2, 3
_T_UINT32, _T_INT32, _T_FLOAT32, _T_BOOL = 4, 5, 6, 7
_T_STRING, _T_ARRAY, _T_UINT64, _T_INT64, _T_FLOAT64 = \
    8, 9, 10, 11, 12

_FIXED_SIZES = {
    _T_UINT8: 1, _T_INT8: 1, _T_UINT16: 2, _T_INT16: 2,
    _T_UINT32: 4, _T_INT32: 4, _T_FLOAT32: 4, _T_BOOL: 1,
    _T_UINT64: 8, _T_INT64: 8, _T_FLOAT64: 8,
}


class _Reader:
    """A cursor over the GGUF metadata section. Only the
    header and the metadata key-value block are read; tensor
    infos and tensor data are never touched, so probing an
    873 MB candidate costs the metadata bytes only."""

    def __init__(self, fh):
        self._fh = fh

    def read(self, n: int) -> bytes:
        data = self._fh.read(n)
        if len(data) != n:
            raise GGUFProbeError(
                "unexpected end of file inside the GGUF "
                "metadata section")
        return data

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> bytes:
        return self.read(self.u64())

    def skip_value(self, vtype: int) -> None:
        if vtype in _FIXED_SIZES:
            self.read(_FIXED_SIZES[vtype])
        elif vtype == _T_STRING:
            self.string()
        elif vtype == _T_ARRAY:
            elem = self.u32()
            count = self.u64()
            if elem in _FIXED_SIZES:
                self.read(_FIXED_SIZES[elem] * count)
            elif elem == _T_STRING:
                for _ in range(count):
                    self.string()
            else:
                raise GGUFProbeError(
                    f"unsupported GGUF array element type "
                    f"{elem} in metadata")
        else:
            raise GGUFProbeError(
                f"unsupported GGUF metadata value type {vtype}")

    def read_value(self, vtype: int):
        """Read one value, returning it for scalar/string
        types (arrays are skipped and return None — the
        probe prices one string key, nothing else)."""
        if vtype == _T_STRING:
            return self.string()
        if vtype in _FIXED_SIZES:
            raw = self.read(_FIXED_SIZES[vtype])
            if vtype == _T_UINT8:
                return raw[0]
            if vtype == _T_BOOL:
                return bool(raw[0])
            fmt = {_T_INT8: "<b", _T_UINT16: "<H",
                   _T_INT16: "<h", _T_UINT32: "<I",
                   _T_INT32: "<i", _T_FLOAT32: "<f",
                   _T_UINT64: "<Q", _T_INT64: "<q",
                   _T_FLOAT64: "<d"}[vtype]
            return struct.unpack(fmt, raw)[0]
        self.skip_value(vtype)
        return None


def probe_gguf_template(path: str) -> dict:
    """Probe one GGUF file's metadata for
    ``tokenizer.chat_template`` with a real GGUF reader.

    Returns ``{"present": bool, "template": str | None,
    "template_sha256": hex | None, "template_bytes":
    int | None, "gguf_version": int, "metadata_keys": int}``.
    Raises :class:`GGUFProbeError` on anything that is not a
    well-formed GGUF metadata section."""
    try:
        fh = open(path, "rb")
    except OSError as exc:
        raise GGUFProbeError(
            f"cannot open GGUF candidate {path!r}: {exc}"
        ) from exc
    with fh:
        reader = _Reader(fh)
        magic = reader.read(4)
        if magic != b"GGUF":
            raise GGUFProbeError(
                f"{path!r} is not a GGUF file (magic "
                f"{magic!r}) — the probe never infers a "
                "template from a filename or a size")
        version = reader.u32()
        if version == 1:
            reader.u32()  # v1 tensor count is u32
            n_kv = reader.u32()
        elif version in (2, 3):
            reader.u64()  # tensor count — not priced here
            n_kv = reader.u64()
        else:
            raise GGUFProbeError(
                f"unsupported GGUF version {version} in "
                f"{path!r}")
        found = None
        for _ in range(n_kv):
            key = reader.string().decode("utf-8", "replace")
            vtype = reader.u32()
            if key == TEMPLATE_KEY:
                if vtype != _T_STRING:
                    raise GGUFProbeError(
                        f"{TEMPLATE_KEY} in {path!r} has "
                        f"non-string value type {vtype}")
                found = reader.string()
            else:
                reader.skip_value(vtype)
    if found is None:
        return {"present": False, "template": None,
                "template_sha256": None, "template_bytes": None,
                "gguf_version": version, "metadata_keys": n_kv}
    return {"present": True,
            "template": found.decode("utf-8", "replace"),
            "template_sha256": _sha256_hex(found),
            "template_bytes": len(found),
            "gguf_version": version, "metadata_keys": n_kv}


def read_template_file(path: str) -> tuple:
    """Read an externally supplied chat-template file (the
    serving configuration's ``--chat-template-file`` target)
    -> ``(sha256 hex, byte count)``. Raises OSError on a
    missing/unreadable file — callers price that as an
    unresolvable serving configuration, never as a pass."""
    with open(path, "rb") as fh:
        data = fh.read()
    return _sha256_hex(data), len(data)


# ---------------------------------------------------------------------------
# (a) Candidate registration (SI-111 §3.3(a); D-111-2)
# ---------------------------------------------------------------------------

def register_candidate(candidate_label: str, *,
                       gguf_path: str = None,
                       external_template_path: str = None) -> dict:
    """Register one candidate for measurement: probe its
    GGUF, record the template hashes, and rule its
    eligibility under D-111-2.

    The registered template hash is the **embedded**
    template's when the GGUF carries one; when it does not,
    the externally supplied template the serving
    configuration will use; when neither exists the
    candidate is **TEMPLATE-UNRESOLVED** and ineligible for
    graded or diagnostic capture until resolved. A GGUF
    that cannot be probed raises :class:`GGUFProbeError` —
    registration never guesses."""
    record = {
        "record": "si.gaia.preflight.candidate_registration",
        "slice": _SLICE,
        "candidate_label": candidate_label,
        "registered_at_america_new_york":
            _now_ny().isoformat(),
        "gguf_path": gguf_path,
        "probe": None,
        "external_template": None,
        "registered_template_sha256": None,
        "registered_template_source": "none",
    }
    probe = None
    if gguf_path is not None:
        probe = probe_gguf_template(gguf_path)
        record["probe"] = {
            "tokenizer_chat_template":
                "present" if probe["present"] else "absent",
            "embedded_template_sha256":
                probe["template_sha256"],
            "embedded_template_bytes":
                probe["template_bytes"],
            "gguf_version": probe["gguf_version"],
            "metadata_keys": probe["metadata_keys"],
        }
    if external_template_path is not None:
        digest, nbytes = read_template_file(
            external_template_path)
        record["external_template"] = {
            "path": external_template_path,
            "sha256": digest, "bytes": nbytes}
    if probe is not None and probe["present"]:
        record["registered_template_sha256"] = \
            probe["template_sha256"]
        record["registered_template_source"] = "embedded"
    elif record["external_template"] is not None:
        record["registered_template_sha256"] = \
            record["external_template"]["sha256"]
        record["registered_template_source"] = "external"
    if record["registered_template_sha256"] is not None:
        record["status"] = "TEMPLATE-RESOLVED"
        record["eligibility"] = (
            "eligible for graded or diagnostic capture "
            "(D-111-2), subject to the capture pre-flight "
            "matching this registered template hash")
    else:
        record["status"] = "TEMPLATE-UNRESOLVED"
        record["eligibility"] = (
            "INELIGIBLE for graded or diagnostic capture "
            "until resolved — fail-closed, no waivers "
            "(D-111-2)")
    return record


# ---------------------------------------------------------------------------
# (b) Served-template resolution + the pre-flight check
# ---------------------------------------------------------------------------

def resolve_served_template(*, spawn_template_file: str = None,
                            served_gguf_path: str = None,
                            served_template_sha256: str = None,
                            served_source_label: str = None) -> tuple:
    """Resolve the template the server will actually apply
    -> ``(sha256 | None, source, detail)``.

    Precedence mirrors the serving stack: a spawn-path
    ``--chat-template-file`` target is what llama-server
    applies when one is passed, so it wins; else the served
    GGUF's embedded template; else a caller-supplied served
    hash (asserted by the caller — for harnesses whose
    serving side cannot be probed in-process). ``(None,
    "unresolvable", detail)`` when no source resolves — the
    ChatML-fallback shape of SI-92 prices exactly here."""
    if spawn_template_file is not None:
        try:
            digest, nbytes = read_template_file(
                spawn_template_file)
        except OSError as exc:
            return None, "spawn-template-file", (
                f"spawn template file {spawn_template_file!r} "
                f"could not be read ({exc}) — the serving "
                "configuration is unresolvable")
        return digest, "spawn-template-file", (
            f"--chat-template-file target "
            f"{spawn_template_file!r} ({nbytes} bytes)")
    if served_gguf_path is not None:
        try:
            probe = probe_gguf_template(served_gguf_path)
        except GGUFProbeError as exc:
            return None, "served-gguf", str(exc)
        if not probe["present"]:
            return None, "served-gguf", (
                f"served GGUF {served_gguf_path!r} carries no "
                f"embedded {TEMPLATE_KEY} and no spawn "
                "template file was supplied — the server "
                "would fall back to a built-in template (the "
                "SI-93 incident shape)")
        return probe["template_sha256"], "served-gguf-embedded", (
            f"embedded {TEMPLATE_KEY} of served GGUF "
            f"{served_gguf_path!r} ({probe['template_bytes']} "
            "bytes)")
    if served_template_sha256 is not None:
        return served_template_sha256, (
            served_source_label or "caller-supplied-hash"), (
            "served template hash supplied by the caller")
    return None, "unresolvable", (
        "no spawn template file, no served GGUF to probe, "
        "and no caller-supplied served template hash — the "
        "template the server will apply cannot be resolved")


def check_preflight(registration: dict, *,
                    spawn_template_file: str = None,
                    served_gguf_path: str = None,
                    served_template_sha256: str = None,
                    served_source_label: str = None) -> dict:
    """The pre-flight check (§3.3(b)): compare the served
    template's hash against the registration record. Returns
    the check record — the function itself never raises, so
    a caller can record a refusal as evidence; entry points
    that must not proceed use :func:`require_preflight`."""
    served_hash, source, detail = resolve_served_template(
        spawn_template_file=spawn_template_file,
        served_gguf_path=served_gguf_path,
        served_template_sha256=served_template_sha256,
        served_source_label=served_source_label)
    registered_hash = registration.get(
        "registered_template_sha256")
    record = {
        "record": "si.gaia.preflight.check",
        "slice": _SLICE,
        "checked_at_america_new_york": _now_ny().isoformat(),
        "candidate_label": registration.get("candidate_label"),
        "registration_status": registration.get("status"),
        "registered_template_sha256": registered_hash,
        "registered_template_source": registration.get(
            "registered_template_source"),
        "served_template_sha256": served_hash,
        "served_template_source": source,
        "served_detail": detail,
        "rule": _RULE,
        "rechecks": [],
    }
    if registration.get("status") != "TEMPLATE-RESOLVED" \
            or registered_hash is None:
        record["decision"] = "REFUSED"
        record["reason"] = (
            "candidate is TEMPLATE-UNRESOLVED — no trained "
            "template hash is registered, so no capture may "
            "start (D-111-2)")
    elif served_hash is None:
        record["decision"] = "REFUSED"
        record["reason"] = (
            f"the served template could not be resolved: "
            f"{detail} — an unresolvable serving "
            "configuration refuses the capture, fail-closed")
    elif served_hash != registered_hash:
        record["decision"] = "REFUSED"
        record["reason"] = (
            f"served template hash {served_hash} does not "
            f"match the registered template hash "
            f"{registered_hash} — the candidate would be "
            "measured under a template it was not registered "
            "with; capture refused before any attempt")
    else:
        record["decision"] = "PASS"
        record["reason"] = (
            "served template hash matches the registered "
            "template hash — the capture starts against the "
            "candidate's recorded trained template")
    return record


def require_preflight(registration: dict, **served) -> dict:
    """Run the pre-flight and return its record, or raise
    :class:`PreflightRefused` carrying it. This is the form
    capture entry points call before their first attempt."""
    record = check_preflight(registration, **served)
    if record["decision"] != "PASS":
        raise PreflightRefused(record)
    return record


# ---------------------------------------------------------------------------
# Operator CLI — registration is an act, so it is a command
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="si.gaia.preflight",
        description="SI-112 pre-flight template lock: probe "
                    "GGUF chat templates and register "
                    "candidates (SI-111 §3.3; D-111-2)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_probe = sub.add_parser(
        "probe", help="probe one GGUF's tokenizer.chat_template")
    p_probe.add_argument("gguf")
    p_reg = sub.add_parser(
        "register", help="register a candidate and write its "
                         "registration record as JSON")
    p_reg.add_argument("--candidate-label", required=True)
    p_reg.add_argument("--gguf", default=None)
    p_reg.add_argument("--external-template", default=None)
    p_reg.add_argument("--out", default=None,
                       help="write the record here (default: "
                            "stdout only)")
    args = parser.parse_args(argv)
    if args.cmd == "probe":
        probe = probe_gguf_template(args.gguf)
        print(json.dumps({
            "gguf": args.gguf,
            "tokenizer_chat_template":
                "present" if probe["present"] else "absent",
            "template_sha256": probe["template_sha256"],
            "template_bytes": probe["template_bytes"],
            "gguf_version": probe["gguf_version"],
            "metadata_keys": probe["metadata_keys"]},
            indent=2))
        return 0
    record = register_candidate(
        args.candidate_label, gguf_path=args.gguf,
        external_template_path=args.external_template)
    text = json.dumps(record, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    return 0 if record["status"] == "TEMPLATE-RESOLVED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
