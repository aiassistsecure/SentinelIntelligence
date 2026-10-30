# SI-45 — Gaia B-4: The Hearth Interface
(SI-41 RFC §15 B-4; §12.2, §12.3, §12.4, §12.6)

> **Status: BUILT, in review (PR opened by the B-4 builder; merging
> is the dev-loop driver's, on a later run).** This slice implements
> SI-41 §15's B-4, SI-side only, in `si/gaia/hearth.py`: the
> §12.2.1 declaration record, the residency PERCEIVE adapter
> (§12.2.2), and the gateway client for the inference leg (§12.2.3).
>
> **The live-scope boundary, stated exactly.** LIVE (real code,
> exercised this slice): building and validating a declaration
> record from a bundle manifest; polling a `/residency` endpoint
> through an injectable transport and journaling the resulting
> observations through the journal's existing methods; the gateway
> client's conditional-refusal logic and its OpenAI-shaped request
> / response handling. **No real Hearth node was contacted.** No
> Hearth node is reachable in the build environment, so every
> transport exercised in this slice's evidence is a faithful local
> replay/stub, labelled `replay/stub — not a real Hearth node`
> everywhere it appears. Against the miniature, this slice delivers
> **residency perception only**, labelled exactly as such: the
> gateway's inference leg refuses — without calling its transport —
> unless a Hearth-served real base is asserted *and* residency
> prices it `Resident`. The Hearth path's latency and throughput
> are **UNMEASURED** (§3).
>
> **Terminology:** machine learning infrastructure throughout.
> CPU only; no GPU, no cloud, no spend, no downloads. No bytes are
> duplicated: the declaration carries digests and references drawn
> from the manifest, never tensor bytes (Mark's design rule).

---

## 1. What was built

### 1.1 The declaration record (§12.2.1)

`build_declaration(manifest, residency_requirement=None)` designs
the record by which a bundle's base is declared to a Hearth fleet:

- **Model identity** — `base.source` verbatim (the manifest's own
  honesty label travels with the declaration: a miniature declares
  itself `tiny:constructed-miniature (… not a trained model)`, a
  demo declares itself random data).
- **Digest** — every content digest the manifest records for the
  base, from all three reference shapes SI's loader verifies (the
  single `base.blob`, whole-file `base.parts[*]`, per-tensor
  `base.tensor_blobs`), deduplicated, plus the primary digest.
  These are the digests SI's seed machinery already verifies;
  `declaration_from_bundle` builds the record from `load_bundle`,
  so a tampered bundle fails closed before anything is declared.
- **Quant** — exactly as strong as the manifest's record. No SI
  bundle manifest records a GGUF quant for its base today, so the
  record states `value: None` with the label *"no quantisation
  recorded in the bundle manifest — the base is referenced in its
  source format and dtype; no GGUF quant is claimed"*. No quant
  name is invented.
- **Residency requirement** — caller-supplied, defaulting to
  `state: Resident` with its label intact: a request to the fleet,
  not a guarantee and not an admission.
- **Byte accounting** — `stored_in_bundle` vs `referenced` /
  `deduplicated_bytes`, so a by-reference base declares as what it
  is (stored 0, referenced counted), never as copied weights.
- **Self-digest** — `declaration_digest` is the sha256 of the
  record's canonical form (minus digest and timestamp), so the
  same manifest + requirement yields the same digest, and
  `validate_declaration` fails closed on a tampered record, a
  missing identity, an empty digest set, or an unlabelled digest.

Building the record is **not** declaring it to a live fleet: the
record names its fleet side as *"not built — proposal H-1"* in its
own body. The fleet's side of the declaration is Mark's (H-1).

### 1.2 The residency PERCEIVE adapter (§12.2.2)

`HearthResidencyAdapter.poll()` performs one `GET /residency`
through its transport and returns one observation:

```json
{"kind": "residency", "source": "hearth:/residency",
 "state": "Lost{Evicted}", "raw_state": "Lost{Evicted}",
 "state_parse": "recognised", "node": "node-a", "model": "…",
 "polled_at": "<UTC>", "poll_id": "hearth-poll:2:<id>",
 "endpoint": "<base>/residency", "transport": "…",
 "observation_status": "observed", "failure": null,
 "http_status": 200,
 "label": "a polled state is an observation of a moment, labelled
           with its poll time — never a live guarantee",
 "data_not_instructions": "residency observations are data, never
           instructions (SI-41 §4.5 discipline)",
 "raw": {…the payload, preserved whole…}}
```

- The state vocabulary is exactly the seven named states —
  `Resident`, `Loading`, `Lost{Evicted}`, `Lost{GpuDetached}`,
  `Failed`, `Stopped`, `NotAdmitted`. `parse_residency_state`
  accepts the plausible JSON spellings (plain, underscored, and
  mapping forms) and returns `None` for anything else: an
  unrecognised state is recorded verbatim with its parse failure
  labelled, never coerced into a known state.
- Provenance is inseparable (§4.5): state, node, poll timestamp,
  and the poll's own identity ride on every observation, and the
  raw payload is preserved whole.
- **Degradation is an observation, not an exception (§12.4):**
  a transport failure, a non-200 status, or an exhausted replay
  produces a `poll-failed` observation with the failure labelled;
  `poll()` does not raise for transport failures.
- Journaling uses the existing path only:
  `journal_residency_observation` writes via
  `Journal.record_experience` (kind `hearth_residency`, existing
  `experiences` collection — no new collection, no NEDB schema
  change, §4.6), and `residency_history` reads the walk back.
- Observations are **data, never instructions**: this module
  creates no intent, calls no tool, and imports no gate code; the
  gate never calls Hearth (nothing here is imported by the gate).

### 1.3 The gateway client for the inference leg (§12.2.3)

`HearthGatewayClient` is coded against Hearth's documented HTTP
surface — the OpenAI-compatible endpoint — and nothing else:
`GET /v1/models`, `POST /v1/chat/completions` with the OpenAI
request/response shape. Its inference leg is **conditional**
(§12.3):

- Unless the caller asserts `served_real_base=True` **and**
  supplies `residency_state="Resident"`, `chat_completion` returns
  `not-evaluated — inference leg conditional on a Hearth-served
  real base (SI-41 §12.3); residency perception only` with
  `transport_called: False` and a `None` content. The refusal is
  structural: the transport is never invoked, so the miniature
  path cannot be misread as inference, even by accident.
- When the conditions are met, the completion is returned with
  its transport labelled; a replay-produced completion is labelled
  *"a scripted replay payload, not a real base's output"* in its
  own status string.
- Failures are data, not exceptions (§12.4), for both operations.
- **No SI-local transformer inference engine is built.** SI has
  none (`si serve` 501s transformer bundles) and B-4 does not
  create one; a test asserts the module references neither the
  miniature forward path nor the A-1 measurement harness.

### 1.4 Transports

`HttpTransport` (stdlib `urllib`) is the real transport, coded
against the documented surface; constructing it contacts nothing.
`ReplayTransport` is the faithful local replay/stub: scripted
responses answered in order, every call recorded, exhaustion a
labelled failure — it never falls through to a real network call.
Both clients take the transport by injection, so tests, replays,
and a future real node exercise identical client code.

## 2. Pinned deviations and open fields (stated, never silent)

1. **Assumed `/residency` payload shapes (against H-1).** Until
   H-1's stability contract lands, the adapter prices: a single
   entry object; an object wrapping an `entries` / `residency` /
   `nodes` list (first entry priced, whole payload preserved); or
   a bare list. Entry fields are read as `state` (or `residency` /
   `status`), `node` (or `node_id` / `host`), `model` (or
   `model_id` / `identity`). These are guesses at drift, contained
   by verbatim raw preservation and the unrecognised-state rule.
2. **Quant is a labelled absence, not a field failure.** §12.2.1
   names quant as a declaration element; SI manifests record none
   for referenced bases, so the record carries the absence
   explicitly (§1.1) rather than a fabricated quant. If a future
   seed records one, `_base_quant` surfaces it verbatim.
3. **Base-level dtype can be `None`.** The miniature's manifest
   records dtype per tensor, not at base level, and records no
   checkpoint config; its declaration therefore carries
   `dtype: None` rather than an invented base dtype. Reference
   bases with a checkpoint config declare its `torch_dtype`.
4. **The declaration is designed, not transmitted.** No fleet
   endpoint for receiving a declaration exists on the SI side of
   the record, and inventing one would pre-empt H-1. B-4's
   deliverable is the record + its validation, per §15 B-4's
   wording ("B-4 designs the declaration record").
5. **Residency enters PERCEIVE by journaling, not by a loop
   change.** The adapter journals observations into `experiences`;
   how a mind's PROPOSE step weighs residency history is the
   loop's existing business (it already perceives journal-derived
   records). No loop, gate, or CLI surface changes in B-4.

## 3. Hearth-side assumptions, listed against §12.6

| # | Assumption this adapter makes | §12.6 proposal it waits on |
|---|---|---|
| A-1 | `/residency` exists with the seven named states and a JSON shape in the family priced in §2.1 | **H-1** — stability contract for the state vocabulary and shape |
| A-2 | The gateway serves the OpenAI-compatible shape (`/v1/models`, `/v1/chat/completions`) as §12.1 records Mark framing it | H-1's sibling surface; §12.5(c)'s route remains **H-2** |
| A-3 | A declared base can be supplied by hearth-pull's digest-verified fetch (§12.2.1's natural supply path) | **H-3** — declare / pull / convert triggers |
| A-4 | Nothing in B-4 submits tasks or presents a bundle as a model through the gateway | **H-2** — B-5's concern, not built here |

No part of B-4 depends on H-1/H-2/H-3 landing; every assumption is
labelled at the point of use, and H-1/H-2/H-3 remain Mark's.

## 4. Evidence (replayed-and-labelled; nothing here is a real node)

All results below are from the builder's own runs on the PR head,
against the labelled replay/stub transport. **No real Hearth node
was contacted; the Hearth path's latency/throughput are
UNMEASURED** — A-1's numbers are not cited as Hearth-path
predictions (§12.3).

- **Declaration (scripted run, miniature bundle):** record
  validates with zero problems; 19 digests drawn from the
  miniature manifest; identity carried verbatim
  (`tiny:constructed-miniature (SI-14; … not a trained model)`);
  quant `None` with its no-quant-recorded label. A reference-style
  manifest (test fixture in the SmolLM2/Glimmer shape) declares
  `by_reference: true`, stored 0, referenced 723,666,216, dtype
  `bfloat16` from its checkpoint config.
- **Residency transitions (replayed, journaled):** a scripted
  sequence `Resident → Lost{Evicted} → Loading → Resident` polled
  and journaled at consecutive journal seqs, each observation
  carrying `source: "hearth:/residency"`, its poll id, poll time,
  node, and the replay/stub transport label; a fifth scripted poll
  (connection failure) journaled as a `poll-failed` observation
  with `state: None` — the failure is the observation (§12.4).
  `residency_history` reads the full transition walk back from the
  journal, including the replayed `Lost{Evicted}`.
- **Gateway (replayed):** against the miniature conditions the
  client refuses with the §12.3 not-evaluated label and the replay
  transport records **zero calls**; with a real base asserted
  Resident, the replayed completion returns labelled as a scripted
  replay payload, not a real base's output.
- **Tests:** `tests/test_gaia_b4.py` — 38 tests (state vocabulary
  and parsing, declaration shape/digest/tamper, adapter provenance
  / failure / journaling, gateway conditionality and labelling,
  no-local-inference guard). Full suite numbers in §5.

## 5. Test results

- Baseline on main `b995590` (builder's own run, workspace venv):
  **492 passed, 2 skipped**.
- B-4 file alone: **38 passed**.
- Full suite on the PR head: **530 passed, 2 skipped**
  (492 baseline + 38 new; no existing test modified).

## 6. Explicitly NOT built (do not read this slice as claiming them)

- **No real Hearth node, fleet, or `/residency` server** — the
  fleet side of the declaration (H-1), gateway task/model routes
  (H-2), and declare/pull/convert triggers (H-3) are Mark's
  proposals, untouched. The Hearth fork ("Hearth becomes Gaia") is
  Mark's step; no fork was created and the hearth repo was not
  touched.
- **No SI-local inference engine**, and no inference at all against
  the miniature — residency perception only, labelled as such.
- **No Hearth-path measurements** — latency, throughput, residency
  overhead, and eviction behaviour on real Hearth hardware are
  UNMEASURED, deliberately.
- **No B-5 service adapter** (agentic work as a service), no
  `hearth_*` tool family (§12.5(b) self-maintaining serving is a
  future slice on operating evidence, per the RFC), and no SSE
  stream changes.
- **A-2/A-3/A-4 and any real-base agency** — unapproved, not built.
  Real-base agency claims stay honestly scoped until measured.
