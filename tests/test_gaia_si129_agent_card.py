"""SI-129: the Gaia A2A Agent Card — discovery only, registry-true.

Coverage:

- the card validates against SI-129's own claimed fields
  (``validate_agent_card`` returns no violations);
- skills match the live registry 1:1 — ids, descriptions, and
  the embedded ``x-gaia-tool`` records — so the test fails if
  the registry and the card ever drift;
- capabilities are exactly the honest all-false set (no A2A
  streaming / push / state-history is claimed, because no A2A
  JSON-RPC task endpoint exists), the discovery-only label is
  present, and no skill is invented beyond the registry;
- the version is pinned against pyproject's project version;
- the CLI path (``si gaia-agent-card``) emits the same card,
  honours ``--url`` / ``--out``;
- the Gaia service serves the card at
  ``GET /.well-known/agent-card.json`` with the request's own
  Host as the advertised url, and its ``GET /`` description
  names the route.
"""
import json
import os
import threading
import urllib.request

import pytest

from si.bundle import init_bundle
from si.cli import build_parser, cmd_gaia_agent_card
from si.gaia.agent_card import (
    AGENT_CARD_VERSION,
    WELL_KNOWN_PATH,
    build_agent_card,
    validate_agent_card,
)
from si.gaia.registry import REGISTRY
from si.gaia.service import make_service_server
from si.tiny_seed import seed_tiny_bundle


def test_card_validates():
    assert validate_agent_card(build_agent_card()) == []


def test_skills_match_live_registry_exactly():
    card = build_agent_card()
    skills = card["skills"]
    assert [s["id"] for s in skills] == list(REGISTRY)
    for skill in skills:
        spec = REGISTRY[skill["id"]]
        assert skill["name"] == spec.name
        assert skill["description"] == spec.description
        assert skill["x-gaia-tool"] == spec.to_dict()
        assert skill["tags"] == [spec.consequence_class,
                                 spec.reversibility,
                                 spec.introduced_by]


def test_capabilities_are_honest_all_false():
    card = build_agent_card()
    assert card["capabilities"] == {
        "streaming": False,
        "pushNotifications": False,
        "stateTransitionHistory": False,
    }
    assert "DISCOVERY ONLY" in card["x-gaia-discovery-only"]
    assert "not implemented" in card["x-gaia-discovery-only"] or \
        "No A2A JSON-RPC" in card["x-gaia-discovery-only"]
    # No auth scheme is claimed: none exists on the service.
    assert card["authentication"] == {"schemes": []}
    # The SSE spine is described as Gaia-native, not A2A streaming.
    assert "NOT A2A streaming" in card["x-gaia-sse-events"]


def test_card_identity_fields():
    card = build_agent_card()
    assert card["name"] == "Gaia (Sentinel Intelligence)"
    assert card["provider"]["organization"] == \
        "Interchained / aiassistsecure"
    assert card["url"] == "http://127.0.0.1:8000/"
    custom = build_agent_card(base_url="http://example.test:9000")
    assert custom["url"] == "http://example.test:9000/"


def test_version_matches_pyproject():
    root = os.path.join(os.path.dirname(__file__), "..")
    with open(os.path.join(root, "pyproject.toml"),
              encoding="utf-8") as fh:
        text = fh.read()
    assert f'version = "{AGENT_CARD_VERSION}"' in text
    assert build_agent_card()["version"] == AGENT_CARD_VERSION


def test_validator_catches_drift():
    card = build_agent_card()
    card["skills"] = card["skills"][:-1]
    assert validate_agent_card(card)
    card = build_agent_card()
    card["skills"][0]["description"] = "invented"
    assert validate_agent_card(card)
    card = build_agent_card()
    card["capabilities"]["streaming"] = True
    assert validate_agent_card(card)


def test_cli_emits_card(capsys):
    parser = build_parser()
    args = parser.parse_args(["gaia-agent-card"])
    assert cmd_gaia_agent_card(args) == 0
    card = json.loads(capsys.readouterr().out)
    assert validate_agent_card(card) == []
    assert [s["id"] for s in card["skills"]] == list(REGISTRY)


def test_cli_out_and_url(tmp_path, capsys):
    parser = build_parser()
    out = tmp_path / "agent-card.json"
    args = parser.parse_args(
        ["gaia-agent-card", "--url", "http://127.0.0.1:8123",
         "--out", str(out)])
    assert cmd_gaia_agent_card(args) == 0
    card = json.loads(out.read_text(encoding="utf-8"))
    assert card["url"] == "http://127.0.0.1:8123/"
    assert validate_agent_card(card) == []


@pytest.fixture()
def mini(tmp_path):
    bundle_dir = init_bundle(str(tmp_path / "mini.si"),
                             name="gaia-si129",
                             journal_backend="embedded")
    seed_tiny_bundle(bundle_dir)
    return bundle_dir


def test_serve_route_returns_card(mini):
    server = make_service_server(mini, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever,
                              daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}{WELL_KNOWN_PATH}",
                timeout=30) as resp:
            assert resp.status == 200
            card = json.loads(resp.read().decode())
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/",
                timeout=30) as resp:
            description = json.loads(resp.read().decode())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert validate_agent_card(card) == []
    assert card["url"] == f"http://127.0.0.1:{port}/"
    assert [s["id"] for s in card["skills"]] == list(REGISTRY)
    endpoints = description["endpoints"]
    assert any("agent-card" in key for key in endpoints)
