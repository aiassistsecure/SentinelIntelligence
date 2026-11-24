"""Gaia A2A Agent Card (SI-129): discovery-only well-known document.

An A2A-style Agent Card for the Gaia node, in the shape the A2A
protocol publishes at ``/.well-known/agent-card.json``: name,
description, url, provider, version, capabilities, default
input/output modes, and one skill per agent capability.

**Every skill is derived 1:1 from the live tool registry**
(:data:`si.gaia.registry.REGISTRY`) at build time — this module
contains no hand-written skill list. If the registry gains or
loses a tool, the card changes with it, and the SI-129 tests pin
that equality so the card and the registry cannot drift apart.

**Honest scope — DISCOVERY ONLY.** No A2A JSON-RPC task endpoints
exist in this codebase: there is no ``tasks/send``, no
``tasks/sendSubscribe``, no ``tasks/get`` / ``tasks/cancel``, and
no push-notification endpoint. Accordingly the card's
``capabilities`` are all ``false``:

- ``streaming: false`` — A2A streaming (JSON-RPC
  ``tasks/sendSubscribe``) is not implemented. Gaia *does* have a
  separate SSE event spine (``GET /events`` on the Gaia service,
  SI-47/SI-48: journal rows as server-sent events, resumable via
  ``Last-Event-ID``); that is described, as it actually exists,
  under ``x-gaia-sse-events`` and is **not** claimed as A2A
  streaming.
- ``pushNotifications: false`` — no push endpoint exists.
- ``stateTransitionHistory: false`` — no A2A task state history
  endpoint exists. (Gaia's own provenance trails, ``GET
  /tasks/<seq>`` on the Gaia service, are a Gaia-native surface,
  not an A2A one, and are listed as such.)

What a client *can* do with this card today: discover that the
Gaia agent exists, where its Gaia-native HTTP service lives
(``url``), and exactly which registered tools it can price
through its gate — no more. Actual work submission today goes
through the Gaia-native surfaces named in
``x-gaia-service-endpoints`` (``POST /tasks`` on ``si gaia-serve``,
or the SI-121 MCP server over stdio), never through A2A.

The card also carries no authentication claim: the Gaia service
has no auth (see :data:`DISCOVERY_ONLY_LABEL` and the service's
own ``NO_AUTH_LABEL`` discipline) — ``authentication.schemes``
is therefore the empty list, stated, not omitted.

Stdlib only. CPU only. No network, no secrets, no bundle is
opened to build the card (the registry is code, not bundle
state), so generating a card journals nothing.
"""
from __future__ import annotations

import json

from si.gaia.registry import REGISTRY

#: The well-known path the card is served at on the Gaia service.
WELL_KNOWN_PATH = "/.well-known/agent-card.json"

#: Card / package version: the SI package version (pyproject
#: ``version = "0.1.0"``). Kept as a constant here because the
#: registry is likewise code-versioned; the SI-129 tests pin it
#: against pyproject so it cannot silently drift.
AGENT_CARD_VERSION = "0.1.0"

#: Default base URL: the Gaia service's own defaults
#: (``si gaia-serve`` binds 127.0.0.1:8000). Callers serving on
#: another host/port pass their real base URL.
DEFAULT_BASE_URL = "http://127.0.0.1:8000/"

#: Provider identity, as in the repo itself.
PROVIDER = {
    "organization": "Interchained / aiassistsecure",
    "url": "https://github.com/aiassistsecure/SentinelIntelligence",
}

#: The honest scope label, carried on the card itself.
DISCOVERY_ONLY_LABEL = (
    "DISCOVERY ONLY (SI-129): this card describes the Gaia agent "
    "for discovery. No A2A JSON-RPC task endpoints are implemented "
    "(no tasks/send, tasks/sendSubscribe, tasks/get or "
    "tasks/cancel, and no push notifications), so no A2A task "
    "execution is claimed. Work submission today uses the "
    "Gaia-native surfaces in x-gaia-service-endpoints.")

#: The SSE event spine, described only as it actually exists.
SSE_EVENTS_LABEL = (
    "Gaia-native, NOT A2A streaming: GET /events on the Gaia "
    "service (SI-47/SI-48) streams journal rows as server-sent "
    "events whose id is the journal seq, resumable via the "
    "Last-Event-ID header (or ?after=), replaying then following "
    "live by poll-the-journal. It is not the A2A "
    "tasks/sendSubscribe method, which is not implemented.")

#: Gaia-native endpoints that actually exist on `si gaia-serve`
#: (si.gaia.service), plus the stdio MCP server (SI-121). Listed
#: so a discovering client is pointed at real surfaces, not at
#: A2A endpoints that do not exist.
SERVICE_ENDPOINTS = {
    "GET /": "Gaia service description + honesty labels",
    "GET /health": "bundle id, current head, service labels",
    "GET /.well-known/agent-card.json": (
        "this Agent Card (SI-129, discovery only)"),
    "POST /tasks": (
        "submit one task (scripted intents) through the B-3 "
        "machinery; returns the result with its journal-"
        "serialised provenance trail"),
    "GET /tasks/<seq>": "provenance trail for one journaled task",
    "GET /tasks/<seq>/queue": "one task's folded queue view",
    "POST /tasks/<seq>/dispositions": (
        "operator disposition act on an outcome-unknown intent"),
    "POST /gaia/proposals": (
        "seam ingest of one fork watcher proposal (gated, "
        "journaled, ACKed after journaling)"),
    "GET /events": (
        "the SSE event spine (SI-47/SI-48) — see x-gaia-sse-events"),
    "stdio (si gaia-mcp)": (
        "SI-121 MCP server over stdio: gaia_journal_query and "
        "gaia_escalations_list (read-only) plus gaia_submit_intent "
        "in escalate-only shadow form (journaled, never executed)"),
}


def _skill_for(spec) -> dict:
    """One A2A skill, derived field-for-field from one ToolSpec."""
    return {
        "id": spec.name,
        "name": spec.name,
        "description": spec.description,
        "tags": [spec.consequence_class, spec.reversibility,
                 spec.introduced_by],
        "inputModes": ["application/json"],
        "outputModes": ["application/json"],
        "x-gaia-tool": spec.to_dict(),
    }


def build_agent_card(base_url: str | None = None) -> dict:
    """Build the Agent Card from the live registry.

    ``base_url`` is the Gaia service's base URL (the A2A ``url``
    field); it defaults to the service's own localhost default.
    Skills are emitted in registry order, one per registered
    tool, with no additions and no omissions.
    """
    url = base_url or DEFAULT_BASE_URL
    if not url.endswith("/"):
        url = url + "/"
    return {
        "name": "Gaia (Sentinel Intelligence)",
        "description": (
            "Gaia is the agentic layer of Sentinel Intelligence: "
            "a gated, journaled tool-use agent. Every tool intent "
            "is priced by the SI gate against the bundle policy "
            "and journaled in the NEDB journal before it runs; "
            "work is returned with its provenance trail. This "
            "card is discovery only — see x-gaia-discovery-only."),
        "url": url,
        "provider": dict(PROVIDER),
        "version": AGENT_CARD_VERSION,
        "documentationUrl": (
            "https://github.com/aiassistsecure/SentinelIntelligence"
            "/blob/main/docs/SI-129-a2a-agent-card.md"),
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": False,
        },
        "authentication": {"schemes": []},
        "defaultInputModes": ["application/json"],
        "defaultOutputModes": ["application/json"],
        "skills": [_skill_for(spec) for spec in REGISTRY.values()],
        "x-gaia-discovery-only": DISCOVERY_ONLY_LABEL,
        "x-gaia-sse-events": SSE_EVENTS_LABEL,
        "x-gaia-service-endpoints": dict(SERVICE_ENDPOINTS),
        "x-gaia-auth": (
            "No authentication: the Gaia service checks no "
            "credential and carries no security claim; it binds "
            "localhost by default. authentication.schemes is "
            "empty because no scheme exists, not by omission."),
    }


def validate_agent_card(card) -> list:
    """Itemised violations of ``card`` against what SI-129 claims.

    Empty list = the card is well-formed for this slice: the
    required A2A fields are present and typed, capabilities are
    exactly the honest all-false set, and the skills match the
    live registry 1:1 (ids, descriptions, and the embedded
    ``x-gaia-tool`` records). This is the check the tests pin;
    it validates SI-129's own claims, not the full A2A spec.
    """
    violations: list = []
    if not isinstance(card, dict):
        return ["card: expected object, got "
                f"{type(card).__name__}"]
    for field_name in ("name", "description", "url", "version"):
        if not isinstance(card.get(field_name), str) \
                or not card[field_name]:
            violations.append(f"card.{field_name}: required "
                              "non-empty string missing")
    provider = card.get("provider")
    if not isinstance(provider, dict) \
            or not provider.get("organization"):
        violations.append("card.provider.organization: required")
    caps = card.get("capabilities")
    if caps != {"streaming": False, "pushNotifications": False,
                "stateTransitionHistory": False}:
        violations.append(
            "card.capabilities: must be exactly the honest "
            f"all-false set, got {caps!r}")
    for field_name in ("defaultInputModes", "defaultOutputModes"):
        if not isinstance(card.get(field_name), list) \
                or not card[field_name]:
            violations.append(f"card.{field_name}: required "
                              "non-empty list missing")
    skills = card.get("skills")
    if not isinstance(skills, list):
        violations.append("card.skills: required list missing")
        return violations
    expected = {name: spec.to_dict()
                for name, spec in REGISTRY.items()}
    seen = {}
    for i, skill in enumerate(skills):
        if not isinstance(skill, dict) or not skill.get("id"):
            violations.append(f"card.skills[{i}]: id missing")
            continue
        seen[skill["id"]] = skill
        for field_name in ("name", "description", "tags"):
            if not skill.get(field_name):
                violations.append(
                    f"card.skills[{i}].{field_name}: required")
    if set(seen) != set(expected):
        violations.append(
            "card.skills: ids do not match the live registry "
            f"(card-only: {sorted(set(seen) - set(expected))}, "
            f"registry-only: {sorted(set(expected) - set(seen))})")
    for name, skill in seen.items():
        if name in expected \
                and skill.get("x-gaia-tool") != expected[name]:
            violations.append(
                f"card.skills[{name}]: x-gaia-tool does not "
                "match the live registry record")
        if name in expected \
                and skill.get("description") \
                != expected[name]["description"]:
            violations.append(
                f"card.skills[{name}]: description does not "
                "match the live registry description")
    if "DISCOVERY ONLY" not in str(
            card.get("x-gaia-discovery-only", "")):
        violations.append(
            "card.x-gaia-discovery-only: honest discovery-only "
            "label missing")
    return violations


def execute(args) -> int:
    """`si gaia-agent-card` (SI-129): print or write the card.

    No bundle argument: the card is derived from the registry
    (code), so nothing is opened and nothing is journaled.
    ``--url`` sets the card's ``url`` field (the Gaia service
    base URL being advertised); ``--out`` writes the JSON to a
    file instead of stdout.
    """
    card = build_agent_card(base_url=getattr(args, "url", None))
    violations = validate_agent_card(card)
    if violations:
        for violation in violations:
            print(f"error: {violation}")
        return 1
    text = json.dumps(card, indent=2) + "\n"
    out = getattr(args, "out", None)
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote Agent Card ({len(card['skills'])} skills, "
              f"discovery only) to {out}")
    else:
        print(text, end="")
    return 0
