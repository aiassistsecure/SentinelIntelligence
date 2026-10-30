"""Gaia v1 tool registry (SI-41 RFC §6.1, §8; slice SI-42 / B-1).

The registry is *code*, versioned with SI, and is the single
definition of what tools exist. Existence vs permission are separate
(RFC §6.3): this module defines existence (name, schemas, consequence
class, reversibility); the bundle policy (see :mod:`si.gaia.policy`)
defines permission. A name not in :data:`REGISTRY` does not exist as
a tool — proposing one is an unknown-tool REJECT at the gate, exactly
as the gate treats an unknown intent kind.

B-1 scope, stated honestly: the registry is data plus the schema
validator the gate prices args with. **No tool in this module is an
executor** — there is no handler here, no filesystem access, no
process spawn. Execution lands in B-2/B-3 (RFC §15). The
``introduced_by`` field therefore records the slice that introduced
the tool *into the registry* (SI-41/B-1, built as SI-42); the RFC's
§6.1 example showing ``introduced_by="SI-41/B-2"`` for ``fs_write``
described the then-planned execution slice, and this slice pins the
field to registry introduction instead — the slice doc
(docs/SI-42-gaia-b1-tool-intent-shadow.md) records the deviation.

Schemas use exactly the deliberately small JSON-Schema subset of
RFC §6.1 — ``type``, ``required``, ``properties``, ``items``,
``minimum``/``maximum``, ``minLength``/``maxLength``, ``enum`` —
because the gate must be able to *itemise* every violation. A schema
feature the validator cannot check does not exist in v1 schemas.
Objects with a ``properties`` map reject undeclared properties: an
arg the registry did not declare is an arg the gate cannot price.
``type`` may be a string or a list of strings (the one extension the
result schemas need, e.g. a measured value that is a number or
``None``); every other feature stays inside the RFC's subset.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: Consequence classes (RFC §7.1). Classification is for escalation
#: and audit, never capability removal.
TOOL_CLASSES = ("read", "write", "execute", "external")

#: Reversibility labels (RFC §7.2, §11.2), journaled on every record.
REVERSIBILITIES = ("reversible", "compensable", "one-way")

# Schema fragments reused across tools.
_STR = {"type": "string", "minLength": 1}
_PATH = {"type": "string", "minLength": 1}
_TIMEOUT = {"type": "integer", "minimum": 1, "maximum": 3600}
_EXEC_RESULT = {
    "type": "object",
    "required": ["exit_code", "stdout", "stderr", "timed_out"],
    "properties": {
        "exit_code": {"type": "integer"},
        "stdout": {"type": "string"},
        "stderr": {"type": "string"},
        "timed_out": {"type": "boolean"},
    },
}


@dataclass(frozen=True)
class ToolSpec:
    """One registered tool (RFC §6.1's record, field for field)."""

    name: str
    version: int
    description: str
    args_schema: dict
    result_schema: dict
    consequence_class: str  # registry floor class (RFC §7.2)
    reversibility: str
    default_timeout_seconds: int
    introduced_by: str = "SI-41/B-1"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "args_schema": self.args_schema,
            "result_schema": self.result_schema,
            "consequence_class": self.consequence_class,
            "reversibility": self.reversibility,
            "default_timeout_seconds": self.default_timeout_seconds,
            "introduced_by": self.introduced_by,
        }


def _spec(name, description, args_schema, result_schema, cls, rev,
          timeout=30):
    return ToolSpec(
        name=name, version=1, description=description,
        args_schema=args_schema, result_schema=result_schema,
        consequence_class=cls, reversibility=rev,
        default_timeout_seconds=timeout)


# The complete v1 surface (RFC §8) — 12 tools, no more, no fewer.
# A test asserts this name set equals the RFC §8 table exactly.
_TOOLS = [
    # -- §8.1 filesystem and navigation (read / write) ----------------
    _spec(
        "fs_list",
        "List a directory inside the execution scope. Entries are "
        "capped at 512 and truncation is flagged in the result.",
        {"type": "object", "required": ["path"],
         "properties": {"path": _PATH}},
        {"type": "object", "required": ["entries", "truncated"],
         "properties": {
             "entries": {"type": "array", "items": {
                 "type": "object",
                 "required": ["name", "kind", "size"],
                 "properties": {
                     "name": {"type": "string"},
                     "kind": {"type": "string",
                              "enum": ["file", "dir", "other"]},
                     "size": {"type": "integer", "minimum": 0}}}},
             "truncated": {"type": "boolean"}}},
        "read", "reversible"),
    _spec(
        "fs_read",
        "Read a file inside the execution scope. Content is text, or "
        "base64 when flagged; truncation is flagged in the result.",
        {"type": "object", "required": ["path"],
         "properties": {"path": _PATH,
                        "max_bytes": {"type": "integer", "minimum": 1}}},
        {"type": "object",
         "required": ["content", "encoding", "truncated"],
         "properties": {
             "content": {"type": "string"},
             "encoding": {"type": "string", "enum": ["text", "base64"]},
             "truncated": {"type": "boolean"}}},
        "read", "reversible"),
    _spec(
        "fs_search",
        "Search file contents over the execution scope for a pattern; "
        "matches are result-capped and truncation is flagged.",
        {"type": "object", "required": ["path", "pattern"],
         "properties": {"path": _PATH, "pattern": _STR,
                        "max_matches": {"type": "integer", "minimum": 1}}},
        {"type": "object", "required": ["matches", "truncated"],
         "properties": {
             "matches": {"type": "array", "items": {
                 "type": "object",
                 "required": ["path", "line", "excerpt"],
                 "properties": {
                     "path": {"type": "string"},
                     "line": {"type": "integer", "minimum": 1},
                     "excerpt": {"type": "string"}}}},
             "truncated": {"type": "boolean"}}},
        "read", "reversible"),
    _spec(
        "fs_write",
        "Write a file (create or overwrite) inside the execution "
        "scope. Compensable: the executor journals the pre-image "
        "(RFC §11.2) — in B-1 nothing executes (shadow mode).",
        {"type": "object", "required": ["path", "content"],
         "properties": {"path": _PATH, "content": {"type": "string"}}},
        {"type": "object",
         "required": ["path", "bytes_written", "sha256"],
         "properties": {
             "path": {"type": "string"},
             "bytes_written": {"type": "integer", "minimum": 0},
             "sha256": {"type": "string"}}},
        "write", "compensable"),
    _spec(
        "fs_edit",
        "Exact-match replacement in a file inside the execution "
        "scope; fails if `old` is absent or ambiguous.",
        {"type": "object", "required": ["path", "old", "new"],
         "properties": {"path": _PATH, "old": _STR,
                        "new": {"type": "string"}}},
        {"type": "object",
         "required": ["path", "replacements", "sha256"],
         "properties": {
             "path": {"type": "string"},
             "replacements": {"type": "integer", "minimum": 0},
             "sha256": {"type": "string"}}},
        "write", "compensable"),
    _spec(
        "fs_mkdir",
        "Create a directory inside the execution scope.",
        {"type": "object", "required": ["path"],
         "properties": {"path": _PATH}},
        {"type": "object", "required": ["path", "created"],
         "properties": {"path": {"type": "string"},
                        "created": {"type": "boolean"}}},
        "write", "compensable"),
    # -- §8.2 the mind's own house (read) ------------------------------
    _spec(
        "journal_query",
        "Query rows from the mind's OWN journal; collections are "
        "restricted to the journal's five, limit capped at 50.",
        {"type": "object", "required": ["collection"],
         "properties": {
             "collection": {"type": "string", "enum": [
                 "self_edits", "deltas", "checkpoints", "provenance",
                 "experiences"]},
             "filter": {"type": "object"},
             "limit": {"type": "integer", "minimum": 1, "maximum": 50}}},
        {"type": "object", "required": ["rows", "count", "truncated"],
         "properties": {
             "rows": {"type": "array", "items": {"type": "object"}},
             "count": {"type": "integer", "minimum": 0},
             "truncated": {"type": "boolean"}}},
        "read", "reversible"),
    _spec(
        "bundle_status",
        "Report {head, deltas, last_seq, policy_hash} from the "
        "verified manifest. Derived from load_bundle; invents nothing.",
        {"type": "object", "required": [], "properties": {}},
        {"type": "object",
         "required": ["head", "deltas", "last_seq", "policy_hash"],
         "properties": {
             "head": {"type": "string"},
             "deltas": {"type": "integer", "minimum": 0},
             "last_seq": {"type": "integer", "minimum": 0},
             "policy_hash": {"type": "string"}}},
        "read", "reversible"),
    _spec(
        "probe_run",
        "Run one probe suite through the real probe harness and "
        "return its report with measured-pass / measured-fail / "
        "not-evaluated labels exactly as produced — on bundles with "
        "no harness the honest result is not-evaluated (RFC §8.2).",
        {"type": "object", "required": ["suite"],
         "properties": {
             "suite": {"type": "string",
                       "enum": ["identity", "capability"]}}},
        {"type": "object", "required": ["suite", "label", "detail"],
         "properties": {
             "suite": {"type": "string",
                       "enum": ["identity", "capability"]},
             "label": {"type": "string", "enum": [
                 "measured-pass", "measured-fail", "not-evaluated"]},
             "measured": {"type": ["number", "null"]},
             "detail": {"type": "string"}}},
        "read", "reversible", timeout=120),
    # -- §8.3 execution (execute) ---------------------------------------
    _spec(
        "shell_exec",
        "Run a shell command — the harness-class power in its raw "
        "form. One-way: no pre-image of a command's effects exists. "
        "The command string is priced by gate check 1 (constitution) "
        "and journaled in full with the intent. The schema pins an "
        "8192-char command cap: RFC §5.2 names a policy command-length "
        "cap but §6.2 defines no key for it, so B-1 pins the cap at "
        "the schema (recorded as an open field in the slice doc).",
        {"type": "object", "required": ["command"],
         "properties": {
             "command": {"type": "string", "minLength": 1,
                         "maxLength": 8192},
             "cwd": _PATH,
             "timeout_seconds": _TIMEOUT}},
        _EXEC_RESULT,
        "execute", "one-way", timeout=120),
    _spec(
        "code_exec",
        "Execute code with the runtime's own interpreter (python in "
        "v1). The source is args: scanned by check 1 and journaled in "
        "full with the intent. Exactly one of `source` / `path` must "
        "be present — the §6.1 schema subset cannot express that, so "
        "the gate prices it as an additional itemised sub-check "
        "(recorded as an open field in the slice doc).",
        {"type": "object", "required": ["language"],
         "properties": {
             "language": {"type": "string", "enum": ["python"]},
             "source": _STR,
             "path": _PATH,
             "timeout_seconds": _TIMEOUT}},
        _EXEC_RESULT,
        "execute", "one-way", timeout=120),
    _spec(
        "test_run",
        "Run a test target and report the suite's own summary counts "
        "as the suite reported them; a parse failure returns the raw "
        "output with parse='failed' — counts are never invented.",
        {"type": "object", "required": ["path"],
         "properties": {"path": _PATH, "timeout_seconds": _TIMEOUT}},
        {"type": "object",
         "required": ["exit_code", "passed", "failed", "skipped",
                      "parse", "output"],
         "properties": {
             "exit_code": {"type": "integer"},
             "passed": {"type": "integer", "minimum": 0},
             "failed": {"type": "integer", "minimum": 0},
             "skipped": {"type": "integer", "minimum": 0},
             "parse": {"type": "string", "enum": ["ok", "failed"]},
             "output": {"type": "string"}}},
        "execute", "one-way", timeout=120),
]

#: name -> ToolSpec, in RFC §8 order.
REGISTRY: dict = {t.name: t for t in _TOOLS}

#: The RFC §8 table's name set, pinned as data so the acceptance
#: test compares the registry against the RFC, not against itself.
RFC_V1_TOOL_NAMES = frozenset({
    "fs_list", "fs_read", "fs_search",
    "fs_write", "fs_edit", "fs_mkdir",
    "journal_query", "bundle_status", "probe_run",
    "shell_exec", "code_exec", "test_run",
})


def get_tool(name) -> ToolSpec | None:
    """The spec for ``name``, or None — an unlisted name is not a tool."""
    if not isinstance(name, str):
        return None
    return REGISTRY.get(name)


# -- the §6.1 schema-subset validator ------------------------------------
_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float))
    and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def validate_against_schema(value, schema, path: str = "args") -> list:
    """Itemised violations of ``value`` against a §6.1-subset schema.

    Returns a list of human-readable violation strings (empty = the
    value conforms). Every violation is separate and names its field
    path, in the itemised style of the SI-9b structural suite — the
    gate journals these with the safety check.
    """
    violations: list = []
    if not isinstance(schema, dict):
        return [f"{path}: schema itself is malformed (not an object)"]
    declared = schema.get("type")
    if declared is not None:
        types = declared if isinstance(declared, list) else [declared]
        if not any(_TYPE_CHECKS.get(t, lambda v: False)(value)
                   for t in types):
            violations.append(
                f"{path}: expected type {declared!r}, got "
                f"{type(value).__name__}")
            return violations  # deeper checks would be noise
    if "enum" in schema and value not in schema["enum"]:
        violations.append(f"{path}: {value!r} not in enum {schema['enum']!r}")
    if isinstance(value, dict):
        for req in schema.get("required") or []:
            if req not in value:
                violations.append(f"{path}.{req}: required field missing")
        props = schema.get("properties")
        if isinstance(props, dict):
            for key in sorted(value):
                if key not in props:
                    violations.append(
                        f"{path}.{key}: undeclared argument (the registry "
                        "schema declares no such field)")
            for key, subschema in props.items():
                if key in value:
                    violations.extend(validate_against_schema(
                        value[key], subschema, f"{path}.{key}"))
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value):
            violations.extend(validate_against_schema(
                item, schema["items"], f"{path}[{i}]"))
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            violations.append(
                f"{path}: string length {len(value)} below minLength "
                f"{schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            violations.append(
                f"{path}: string length {len(value)} above maxLength "
                f"{schema['maxLength']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            violations.append(
                f"{path}: value {value} below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            violations.append(
                f"{path}: value {value} above maximum {schema['maximum']}")
    return violations


def validate_args(spec: ToolSpec, args) -> list:
    """Itemised violations of a tool intent's declared args."""
    if not isinstance(args, dict):
        return [f"args: expected object, got {type(args).__name__}"]
    violations = validate_against_schema(args, spec.args_schema, "args")
    if spec.name == "code_exec":
        # The §6.1 subset cannot express source-XOR-path; priced here
        # as an additional itemised rule (see the spec's description).
        has_source, has_path = "source" in args, "path" in args
        if has_source == has_path:
            violations.append(
                "args: code_exec requires exactly one of 'source' / "
                "'path' (the schema subset cannot express this; priced "
                "as an additional gate rule, RFC §8.3)")
    return violations


def validate_result(spec: ToolSpec, result) -> list:
    """Itemised violations of an execution result (RFC §11.1.1)."""
    return validate_against_schema(result, spec.result_schema, "result")
