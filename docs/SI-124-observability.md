# SI-124 / OBS-1 — OpenTelemetry span mapping from the journal + OTLP export

Lane SI-120, slice SI-124. Builds on SI-122 (the
read-only, chain-verified bundle open) — the exporter
reads a bundle exactly the way EXPORT-1 does and
writes nothing to it.

## The command

```
si gaia-export-traces <bundle> [--out FILE] [--otlp-endpoint URL]
```

Emits the bundle's journal as ONE OTLP/JSON payload
(the standard `resourceSpans` structure) to `--out`
or stdout. **Nothing is sent anywhere by default.**
Passing `--otlp-endpoint URL` is the explicit operator
act that turns network export on: the same payload
bytes are then POSTed to that URL.

Pointing it at Langfuse: Langfuse ingests OTLP
natively at

```
https://cloud.langfuse.com/api/public/otel/v1/traces
```

(self-hosted: `https://<your-host>/api/public/otel/v1/traces`).
Langfuse authenticates that endpoint with HTTP Basic
over the project's public/secret key pair; supply the
pair through the environment, never on the command
line and never in code:

```
SI_LANGFUSE_PUBLIC_KEY=pk-lf-... \
SI_LANGFUSE_SECRET_KEY=sk-lf-... \
si gaia-export-traces mybundle.si \
    --otlp-endpoint https://cloud.langfuse.com/api/public/otel/v1/traces
```

For any other OTLP collector, `SI_OTLP_AUTHORIZATION`
is used verbatim as the `Authorization` header when
set (it wins over the Langfuse pair). No credential of
any kind is needed — or read — for file/stdout export.

No OpenTelemetry SDK is vendored or required: the
mapping targets the OTLP wire format directly, stdlib
only (`urllib` for the one POST). The wire format is
the standard; an SDK would add a dependency to produce
the same bytes.

## The span mapping

One span per journal entry, named by collection +
kind. Gaia (the acting layer) is `gaia.*`; the mind's
own journal machinery is `si.*`.

| Journal entry | Span name | Parent | Notable attributes |
|---|---|---|---|
| `experiences`, kind `task` | `gaia.task` | — (trace root) | `si.journal.seq` |
| `self_edits`, kind `tool` | `gaia.tool_intent` | its task | `tool.name`, `si.gate.verdict`, `si.edit_id`, `si.gate.checks`, `si.caused_by.seq` / `si.caused_by.hash` |
| `self_edits`, other kinds | `si.self_edit` | its task (when chained) | `si.gate.verdict`, `si.self_edit.result`, `si.delta_id` |
| `experiences`, kind `tool_result` | `gaia.tool_result` | its intent | `si.execution.outcome` / `.status` / `.tool` / `.refusal`; span status Error when the outcome is refused/failed |
| `experiences`, kind `escalation_resolution` | `gaia.escalation_resolution` | its intent | `si.intent.seq`, `si.edit_id` |
| `experiences`, kind `gaia_work` | `gaia.work_submission` | its intent | `si.record.fork_work_id` |
| `experiences`, kind `queue_*` | `gaia.queue.*` | its intent, else its task | `si.task.seq`, `si.intent.seq` |
| `experiences`, other kinds | `si.experience` | its intent/task when referenced | `si.experience.kind` |
| `deltas` | `si.delta` | its self-edit | `si.delta.id`, `si.delta.status`, `si.delta.bytes` |
| `checkpoints` | `si.checkpoint` | — | `si.checkpoint.head`, `si.checkpoint.manifest` |
| `provenance` | `si.provenance` | the entry it derives (when a self-edit) | `si.provenance.child`, `si.provenance.relation`, `si.provenance.parents` |

Every span also carries `si.journal.collection`,
`si.journal.seq`, `si.journal.entry_hash`,
`si.journal.prev_hash`, and `si.journal.kind`.

**Traces.** One trace per task the journal records,
plus ONE journal-scope trace for everything the
journal does not tie to a task (seed checkpoints, head
provenance, unchained checkpoints). Task membership is
read from the journal's own linkage — `record.task_seq`
on queue records, `self_edit_id` / `record.edit_id` /
`record.intent_seq` references, and provenance rows
chaining a self-edit to `task@<seq>`. Where none of
that reaches, an entry is journal-scope; the mapping
says so rather than guessing.

**Parentage** follows the same references: an outcome
hangs from the intent it answers, an intent from its
task, a delta from its self-edit. A seam intent's
`caused_by` points OUTSIDE the journal (the fork's
residency spine), so it is carried as span attributes,
never as parentage — there is no journal span to
parent to. A span is never parented across traces.

**Ids (deterministic).**

- trace id = first 32 hex of
  `sha256("si-trace:" + manifest_id + ":" + scope)`,
  where scope is `task@<seq>` or `journal`;
- span id = first 16 hex of
  `sha256("si-span:" + entry_hash)`.

Re-exporting the same journal yields the same ids and
byte-identical payload — the payload carries no export
timestamp; it is a pure function of journal + manifest.

**Timestamps.** From each entry's journaled `ts`
(append time). The journal records appends, not
durations, so every span is a zero-duration point at
its append instant (`endTimeUnixNano ==
startTimeUnixNano`). No duration is invented.

**Span status.** Error only for refused/failed tool
outcomes — an execution failure. Gate verdicts are
attributes: an ESCALATE or REJECT is the gate working
as designed, not a tracing error.

## Honest limits

- **Journal granularity.** One span per journaled
  entry; nothing finer exists in the source to map.
  This is a projection of the audit journal, not
  in-process instrumentation.
- **Append-time points, not durations.** See above.
- **Not a metrics pipeline, no dashboard** — by
  design (the roadmap's OBS-1 is an export mapping;
  Langfuse or any OTLP backend provides the UI).
- **Whole journal only.** No window flag in v1;
  SI-122's `gaia-export` remains the tool for
  verifiable journal windows.
- **Task grouping is only as complete as the
  journal's linkage.** Entries the writers did not
  chain to a task are journal-scope, labelled as such
  by their trace — not silently attached to a
  plausible task.
