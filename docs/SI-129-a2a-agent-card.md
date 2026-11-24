# SI-129 — A2A Agent Card for the Gaia node

Packaging item (c) from the SI-128 comparison follow-on: SI-128
recorded "No A2A support yet — no Agent Card". This slice adds
the Agent Card, and only the Agent Card.

Code: `si/gaia/agent_card.py` (card builder + validator + CLI
execution) + one read-only route in `si/gaia/service.py` +
`si gaia-agent-card` wiring in `si/cli.py`.
Tests: `tests/test_gaia_si129_agent_card.py`.

## What the card is

An A2A-style Agent Card: a well-known JSON document describing
an agent for discovery — `name`, `description`, `url`,
`provider`, `version`, `documentationUrl`, `capabilities`,
`authentication`, `defaultInputModes`, `defaultOutputModes`,
and `skills`.

- **Name:** Gaia (Sentinel Intelligence)
- **Provider:** Interchained / aiassistsecure
  (`https://github.com/aiassistsecure/SentinelIntelligence`)
- **Version:** `0.1.0` — the SI package version from
  `pyproject.toml`; a test pins the two together.
- **URL:** the Gaia service base URL being advertised (the
  service's own localhost default, `http://127.0.0.1:8000/`,
  unless another base is supplied; the serve route advertises
  the request's own `Host`).
- **Skills:** one per registered Gaia tool, derived **1:1 from
  the live tool registry** (`si.gaia.registry.REGISTRY`) at
  build time. There is no hand-written skill list: each skill's
  `id`/`name` is the tool name, its `description` is the
  registry description verbatim, its `tags` are the tool's
  consequence class, reversibility, and introducing slice, and
  its `x-gaia-tool` extension carries the full registry record
  (schemas, class, reversibility, timeout). At this slice the
  registry holds 14 tools (the 12 RFC §8 tools plus the two
  seam-crossing `hearth.*` tools); if the registry changes, the
  card changes with it, and the SI-129 tests fail if the two
  ever drift.

## How to fetch or generate it

From a running Gaia service (`si gaia-serve <bundle>`):

```bash
curl http://127.0.0.1:8000/.well-known/agent-card.json
```

The route builds the card per request from the live registry,
so what is served can never go stale against the code serving
it. Reading it journals nothing.

Without a server (no bundle is opened; nothing is journaled):

```bash
si gaia-agent-card                              # JSON to stdout
si gaia-agent-card --url http://host:port/      # advertise another base
si gaia-agent-card --out agent-card.json        # write to a file
```

Design choice: both surfaces, because each is the smallest fit
for an existing pattern — the well-known route rides the
existing Gaia HTTP service (the natural A2A discovery
location), and the CLI follows the repo's established
`si gaia-*` pattern for offline/static generation (e.g. for
publishing the card as a static file). Both call the same
builder, so they cannot disagree.

## Honest limits — discovery only

This is the load-bearing section. **The card is DISCOVERY
ONLY.**

- **No A2A JSON-RPC task endpoints exist.** There is no
  `tasks/send`, no `tasks/sendSubscribe`, no `tasks/get` /
  `tasks/cancel`, and no push-notification endpoint in this
  codebase. A client cannot submit work to Gaia over A2A
  today, and the card does not claim otherwise:
  `capabilities` is exactly
  `{"streaming": false, "pushNotifications": false,
  "stateTransitionHistory": false}`, and the card carries the
  limit in its own `x-gaia-discovery-only` field.
- **The SSE event spine is not A2A streaming.** Gaia does have
  a server-sent-events surface — `GET /events` on the Gaia
  service (SI-47/SI-48): journal rows as SSE, id = journal seq,
  resumable via `Last-Event-ID`, replay-then-follow by
  poll-the-journal. The card describes it, as it actually
  exists, under `x-gaia-sse-events`, explicitly labelled
  Gaia-native and *not* the A2A `tasks/sendSubscribe` method.
- **No authentication is claimed.** The Gaia service checks no
  credential and binds localhost by default;
  `authentication.schemes` is the empty list, stated rather
  than omitted, with the reason in `x-gaia-auth`.
- **Skills describe registered tools, not a promise of
  execution.** A skill's presence means the tool exists in the
  registry and the gate can price an intent for it. Whether an
  intent executes, escalates, or is rejected is the gate's and
  the bundle policy's decision at submission time, under the
  execution scopes the underlying slices state (e.g. live
  execution is miniature-scoped; the two `hearth.*` skills are
  seam-crossing names with no local executor — their registry
  descriptions say so). The card invents no execution claim
  beyond the registry records it embeds.
- **Where work actually happens today** is listed on the card
  in `x-gaia-service-endpoints`: the Gaia-native HTTP service
  (`POST /tasks`, provenance trails, queue, dispositions, seam
  ingest, SSE events) and the SI-121 MCP server over stdio
  (`si gaia-mcp`; its one state-changing tool is escalate-only
  shadow form — journaled, never executed).

## Verification

`tests/test_gaia_si129_agent_card.py` pins: the card validates
against the fields this slice claims; skills match the live
registry exactly (ids, descriptions, embedded records — the
test fails on drift in either direction); capabilities are the
honest all-false set with the discovery-only label present;
the version matches `pyproject.toml`; the CLI emits the same
card and honours `--url`/`--out`; and the serve route returns
a valid card advertising the request's own Host, with `GET /`
naming the route.
