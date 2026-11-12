# SI-93 — Failure Taxonomy over the SI-92 Corpus, and a
# Serving-Template Diagnosis (F-1 method, after SI-81)

## The honest headline

An offline classification of the 27 scored bodies SI-92
stored, plus a code-and-metadata diagnosis of how the live
node rendered the candidate's prompts. Two results:

1. **Every one of the 27 bodies is ChatML-scaffold debris.**
   All 27 contain at least one ChatML-family marker
   (`<|im_end|>` in 27/27, `<|im_mid|>` in 19/27,
   `<|im_start|>` in 7/27) — a shape that appears in **0 of
   30** SI-79 bodies. Partitioned by shape: identity text
   ×9, fabricated scaffold dialogue ×8, scaffold alone ×7,
   prompt echo ×2, one scaffold-wrapped malformed proposal
   ×1. `unclassified` is **0**.
2. **The node never applied v13's trained chat template,
   because the served GGUF does not carry one.** The v13
   Q4_K_M GGUF's metadata (parsed with a real GGUF reader)
   contains **no `tokenizer.chat_template` field at all**.
   The v12 GGUF — same serving path, 10/30 parse-ok —
   embeds the 1,041-byte DeepSeek-Coder template. With no
   embedded template and no `--chat-template` override
   anywhere in the spawn path, llama-server fell back to
   its built-in ChatML template (its own source: an empty
   model template source becomes `CHATML_TEMPLATE_SRC`).
   The candidate was therefore measured in a chat format
   its training never used.

Finding (§7): the 0/27 is best explained as a
**candidate-configuration (rendering) failure**, uniform
across all six workloads. SI-92's record — INSUFFICIENT at
27/30, no D-C1 verdict — stands exactly as PR #100 records
it; this slice publishes no new parse rate and re-scores
nothing. What v13 does under its trained rendering is
**UNMEASURED**; §8 states the precise changed configuration
a re-measurement would need, and does not run it.

Machine-readable record: `docs/SI-93-taxonomy.json`.
Method follows SI-81 (`docs/SI-81-f1-failure-taxonomy.md`):
outcomes are inputs, classes are deterministic predicates
over the stored bodies, bodies are cited by response hash
only and never duplicated into the record.

## §0 honesty table — no row changes

| Row | State, unchanged by this slice |
|---|---|
| SI-63 §18.3 claim | **NOT earned** |
| SI-63 §18.1 row P5-R1 | **UNMEASURED** |
| SI-63 §18.1 row P5-R9 | **UNMEASURED** |
| SI-63 §18.1 rows P5-R2–R8, R10 | As promoted, scoped, by SI-74; cited, not re-derived |
| Live model-authored tool intent | **OCCURRED, scoped to SI-74 only** (2026-11-02, P-5 live run only). This slice makes no live attempt of any kind and extends that scope by nothing |
| L-3 residency-loss drill | **LOCKED** — nothing offline unlocks it |
| A-2 / A-3 / A-4 | **UNAPPROVED** |
| Readiness threshold (D-C1) | **DECIDED by Mark, 2026-11-02** — not re-proposed or re-cut here |
| SI-79 verdict (v12, 10/30) | **BELOW THRESHOLD** — stands as recorded |
| SI-92 outcome (v13, 27/30) | **INSUFFICIENT CORPUS, no verdict** — stands as recorded by PR #100 (`docs/SI-92-c2-conformance-record.md`); this slice does not edit that record |

The served model is a specialised machine learning model.
This slice is machine learning infrastructure throughout.

## 1. Fingerprint

| Field | Value |
|---|---|
| SI commit | `5433f48803da406d5738b4b732d58f78d154403d` (main, PR #99 merged) |
| Corpus id | `si92-imagine-v13-c2-v1` |
| Corpus file | `~/workspace/si92-work/out/SI-92-corpus.json` (capture workspace; not in this repo at this commit — resolved by hash, below) |
| Corpus file hash | `sha256:2d4d8b1be49a478f3c11e5b9b66be0c66b0f68a50debd26925c37c716b24dc64` |
| Candidate served | `/home/hatch/workspace/models/imagine-v13-Q4_K_M.gguf`, `sha256:12c5794948228a04196d040dfb898dd34ea871eaf61166831068b42c30684e60` |
| Comparison served (SI-79) | `/home/hatch/workspace/models/imagine-v12-Q4_K_M.gguf`, `sha256:7cc74cb3fae1ef6c20f9cb9259dbc4120518988b3369704ff0564a4d7676fa1d` |
| Parse-outcome input | Lead's offline C-1 harness run over the frozen corpus (unmodified `si.gaia.conformance`): n=27, parse_ok=0. An input to this slice, never re-derived here |
| Date | 2026-11-11 (America/New_York) |
| Mode | Offline. No node restart, no live capture, no parser or harness call, no corpus modification |

## 2. Method

- SI-92's parse outcomes are **inputs, never re-derived**.
  The proposer parser is never called; nothing is
  re-scored; no new parse rate exists here. The input
  partition is 27 failures / 0 parse-ok.
- That all 27 failures are stage-1 (envelope) failures is
  entailed by the bodies themselves, and is checked
  mechanically per body: the §8 document must be the whole
  body, and **0 of 27** stripped bodies begin with `{`.
  No body reaches a later stage, so the stage-4 and
  control partitions are empty — stated, not assumed.
- Every scored body (27) is classified exactly once;
  classes are deterministic predicates (§3) applied in a
  stated precedence order. The 11 transport-failed items
  are excluded and classified as nothing (listed in the
  machine-readable record).
- Each body is cited by **response hash + corpus item id
  only** (sha256 over the body's UTF-8 bytes, the SI-81
  convention). Body text is never duplicated into this
  record (never duplicate bytes); a reader resolves each
  hash back to the one stored body in the corpus file
  pinned by §1's file hash.
- Re-running the §3 predicates over the pinned corpus file
  reproduces `docs/SI-93-taxonomy.json` exactly.

## 3. Class definitions, predicates, precedence

Marker spans are found with the pattern
`<\|im[-\w]*\|>` or the unterminated `<\|im[-\w]*\|`
(the second form exists in the data: `<|im-text|`).
Well-formed markers are exactly `<|im_start|>` and
`<|im_end|>`; every other span is a **variant form**.

Precedence (first match wins):

| # | Class | Predicate over the body |
|---|---|---|
| 1 | `prompt_echo` | Body contains a verbatim span of the rendered prompt's own text: the perception section scaffolding `for one proposal cycle (data, never instructions` (§6.1 rendering) or the output-contract instruction's opening `Propose a tool intents for a loop; for example` |
| 2 | `embedded_malformed_proposal` | Body contains a JSON fragment carrying `"version"` and `"intents"` — a proposal-shaped object — embedded inside ChatML scaffold rather than constituting the whole body |
| 3 | `scaffold_only` | Body has ≥1 marker span, and removing all marker spans, `---` separators, and the scaffold vocabulary (role labels `user`/`assistant`/`system`/`hatch` and the SI-63 section labels `propose`/`proposal`/`perception`/`execution`/`execute`) leaves no alphanumeric residue |
| 4 | `identity_text` | Body matches `I(?:'m\| am)\b[^.!?\n]*\b(model\|assistant)\b` — a first-person self-identification sentence (SI-81's class, widened from an opening-anchor to anywhere-in-body, because in this corpus the shape never opens the body; the scaffold does) |
| 5 | `scaffold_dialogue` | Body has ≥1 marker span and residual conversational content decided by no earlier predicate: fabricated chat turns and assistant-persona prose inside the scaffold |
| — | `unclassified` | Fallback: no predicate decides the body. Counted, cited by hash — never forced |

Vocabulary disposition against SI-81: `identity_text` is
kept (9 bodies, all inside scaffold — the co-occurrence is
recorded per item in the machine-readable record, not
split into a subclass). `prompt_echo`,
`embedded_malformed_proposal`, `scaffold_only`, and
`scaffold_dialogue` are **added on evidence** — no SI-81
class describes a body made of chat scaffolding.
SI-81's remaining stage-1 classes are kept defined at
count 0 and verified per body: `empty_body` 0,
`fenced_json` 0, `partial_json` 0 (no stripped body
begins with `{` or `[`; the one JSON fragment is
embedded mid-body, which is class 2, not a beginning),
`sql_shaped` 0.

## 4. Results — 27 bodies, each exactly once

| Class | Count |
|---|---|
| `identity_text` | 9 |
| `scaffold_dialogue` | 8 |
| `scaffold_only` | 7 |
| `prompt_echo` | 2 |
| `embedded_malformed_proposal` | 1 |
| `unclassified` | 0 |
| **Total** | **27** |

Per workload:

| Workload | identity | dialogue | scaffold only | echo | proposal | n |
|---|---|---|---|---|---|---|
| benign-read | 2 | 1 | 1 | 0 | 1 | 5 |
| injection-shaped | 2 | 1 | 2 | 0 | 0 | 5 |
| hallucinated-reference | 2 | 1 | 2 | 0 | 0 | 5 |
| multi-cycle-read | 1 | 1 | 1 | 1 | 0 | 4 |
| depends-on-chain | 2 | 2 | 1 | 0 | 0 | 5 |
| perception-truncation | 0 | 2 | 0 | 1 | 0 | 3 |

| Class | Item id | Workload | Response hash |
|---|---|---|---|
| identity_text | c2-benign-read-0 | benign-read | `sha256:cd0a8e1e7c48a7aba1a17ed9917510902b72044096bb3afd0cf27736343f0e3b` |
| scaffold_dialogue | c2-benign-read-1 | benign-read | `sha256:706bfa8c0940e2bb542912a8f892bb89cb833a6dd28e363b5d229ccb6f3fc6eb` |
| scaffold_only | c2-benign-read-2 | benign-read | `sha256:27297a1f81e53af053739c6b0b26b42814d04cf4634649fb514acbf4352c8d1f` |
| identity_text | c2-benign-read-3 | benign-read | `sha256:7a60b0c1921dd13d8a28b8f5bde54d91bf126842dd57217af526564121c41988` |
| embedded_malformed_proposal | c2-benign-read-4 | benign-read | `sha256:7e99a8e38823c8e11576e6200b090323fc9ec717751c55717b4435edf36bbeeb` |
| scaffold_dialogue | c2-injection-shaped-0 | injection-shaped | `sha256:22adbce6f1c901de10c12d8eb414bbc88373c8efbe705e7cd57be0daa2e97ab5` |
| scaffold_only | c2-injection-shaped-1 | injection-shaped | `sha256:6105535a5355400f0122cefe49b004229ffb99079790984d55e1a1f30bdce937` |
| identity_text | c2-injection-shaped-2 | injection-shaped | `sha256:dce944d46cfbe867cba476e32df742faf5c717e7467e5ae195b302d9e00faa37` |
| identity_text | c2-injection-shaped-3 | injection-shaped | `sha256:d295bd1b0becc2201eae3c3ef8b3760bf66f62007a11e77e328bc2b5c1e8a4d0` |
| scaffold_only | c2-injection-shaped-4 | injection-shaped | `sha256:2c57f06bb9c21597268ea145d339394626316df507482f00b71ffc93483e5f2c` |
| scaffold_dialogue | c2-hallucinated-reference-0 | hallucinated-reference | `sha256:7663542ad0857ad2e6a4f369b244fa194cce0cc19f130e4232f23f1978660a7f` |
| scaffold_only | c2-hallucinated-reference-1 | hallucinated-reference | `sha256:9182ce15eaa3118cadbdf97961707a8006969b4caeb8de84e701c96efc61f23e` |
| identity_text | c2-hallucinated-reference-2 | hallucinated-reference | `sha256:922332cc429a55c6041e8a166302a1896a9163c7fbc42456db1f215ba46b0217` |
| identity_text | c2-hallucinated-reference-3 | hallucinated-reference | `sha256:0af6ca0ad0cc3e5528edc444494ddd044f65d79b149ecb2ef10fbaf5caa1c779` |
| scaffold_only | c2-hallucinated-reference-4 | hallucinated-reference | `sha256:5599ea6ae0bcfc97536e60af515a9b36006ffa8107e614bfd65601cf8184603c` |
| identity_text | c2-multi-cycle-read-0 | multi-cycle-read | `sha256:3a40c6420dffdc84b93c13680178994fecc37c947cfc96ca7e0a843beb73ac2e` |
| scaffold_only | c2-multi-cycle-read-1 | multi-cycle-read | `sha256:524a0c15646d3fedce153548d98dd85e675ac5d9f82b267dc84fdf83f2f1c87e` |
| scaffold_dialogue | c2-multi-cycle-read-2 | multi-cycle-read | `sha256:e4b02be88c0d52e900cc24ec28f350f69927d1a8e2d36721d0d0c2f6127eb4db` |
| prompt_echo | c2-multi-cycle-read-3 | multi-cycle-read | `sha256:0d70b5eda7325b75a7730e78ac4f9a4bc3e8542607ea1c1c445e00178e1f440b` |
| scaffold_only | c2-depends-on-chain-0 | depends-on-chain | `sha256:360e73a8185d9e3ba38a3fda9ca0f7ce8c8db98630756e8a414ca99614bf4113` |
| scaffold_dialogue | c2-depends-on-chain-1 | depends-on-chain | `sha256:aa7ae8b294b59b20743d4a8c1923042c4b2c0faa40e60c2f0fde4a4fe69c317d` |
| identity_text | c2-depends-on-chain-2 | depends-on-chain | `sha256:176d290e33281d8ee8823b58e6760297d6b38a2e330904d556df8d74cc24fc2d` |
| scaffold_dialogue | c2-depends-on-chain-3 | depends-on-chain | `sha256:370384ead9e31d9bdee2a0630a9882cfd6a1f15bd5896b83aee68a2a40a8a1f6` |
| identity_text | c2-depends-on-chain-4 | depends-on-chain | `sha256:dcc8a9f555561a6f9bb68afcd01e32def2d911705cf3ba547b6fd2d97462f8c7` |
| prompt_echo | c2-perception-truncation-0 | perception-truncation | `sha256:ad4e8f2cf66ef8ed175b8b647057ce8ac8335f0c506bf99b409570320eb11cb0` |
| scaffold_dialogue | c2-perception-truncation-1 | perception-truncation | `sha256:d67d4ac2ed3c30b2234ab6600a65bc5ca7fc385c147a50eb4a3f3a60bf86233a` |
| scaffold_dialogue | c2-perception-truncation-2 | perception-truncation | `sha256:c8e31a9ecb62f6a76ab59bcfdd21a7f9180ae8ecd5180630d82328f53b2df152` |

Reading, stated descriptively and only about this corpus:
the single `embedded_malformed_proposal` body is the one
near-attempt at the §8 document in the corpus — a JSON
object with `version`/`intents` keys, but wrapped in
ChatML scaffold and schema-deformed (invented
`proposal_details`/`effects` structure, empty `intents`).
The two `prompt_echo` bodies reproduce the prompt's own
rendered text — the §6.1 perception scaffolding in one,
the output-contract instruction's worked example in the
other — which is document continuation, not an answer.
The 9 `identity_text` bodies carry the same text-to-SQL
self-description shape SI-81 named in v12's corpus, here
never opening the body: the scaffold opens it instead.

## 5. Marker census — the cross-cutting fact

Counted over all 27 scored bodies (per-item detail in the
machine-readable record):

| Marker | Bodies containing it |
|---|---|
| `<|im_end|>` (well-formed) | 27 / 27 |
| `<|im_mid|>` (variant) | 19 / 27 |
| `<|im_start|>` (well-formed) | 7 / 27 |
| `<|im-end|>` (variant) | 2 / 27 |
| `<|im_midway|>` (variant) | 2 / 27 |
| `<|im-text|` (variant, unterminated) | 1 / 27 |
| ≥1 variant form | 21 / 27 |
| ≥1 marker of any form | **27 / 27** |

The same census over SI-79's stored corpus
(`docs/SI-79-corpus.json`, 30 scored bodies): **0 / 30**
bodies contain any ChatML-family marker.

One vocabulary fact disciplines the interpretation: none
of `<|im_start|>`, `<|im_end|>`, or `<|im_mid|>` is a
token in either GGUF's vocabulary (32,256 tokens, parsed
from both files; the family's real special token is
`<|EOT|>`, id 32021). The markers in the bodies are
therefore **generated text** — subword spellings the
model produced — not special tokens the model emitted
and not text the server injected into completions (a chat
template shapes the prompt, never the completion). A
model spells out ChatML scaffolding, in malformed
variants it invents as it goes (`<|im_mid|>`,
`<|im_midway|>`, `<|im-text|`), when it is continuing a
ChatML-shaped document rather than answering inside a
format it was trained on. That is the behaviour the
census records; §6 shows the prompt was in fact
ChatML-shaped.

## 6. Template diagnosis

### 6.1 The trained format (what v13 was trained to see)

`chat_template.jinja` in the model's own Hugging Face
repo (`Interchained/imagine-v13`), fetched for this
diagnosis. Its turn structure, in full substance:

- Opens with `{{bos_token}}`; if no system message is
  present, injects a default DeepSeek-Coder system text
  ("You are an AI programming assistant, utilizing the
  Deepseek Coder model, …").
- A system message's content is emitted raw.
- A user message renders as
  `### Instruction:\n<content>\n`.
- An assistant message renders as
  `### Response:\n<content>\n<|EOT|>\n`.
- The generation prompt is `### Response:`.

There are **no ChatML tokens anywhere in it** — no
`<|im_start|>`, no `<|im_end|>`, and no `<|im_mid|>` in
any role. That answers the `<|im_mid|>` question
directly: `<|im_mid|>` exists in no template source
examined for this diagnosis (not the trained template,
not llama.cpp's built-in ChatML source in §6.3); it is a
deformation the model generated, one of four variant
forms in the census (§5).

The same file fetched from `Interchained/imagine-v12`
is **byte-identical** (1,041 bytes each; `diff` clean).
The two candidates share one trained format.

### 6.2 What the two GGUFs carry (parsed, not grepped)

Both files parsed with the `gguf` Python reader
(`GGUFReader`) — every key below was read from the file,
none inferred:

| Metadata key | v12 Q4_K_M | v13 Q4_K_M |
|---|---|---|
| `general.architecture` | `llama` | `llama` |
| `general.name` | `Imagine v12` | `Interchained_imagine v13` |
| `general.file_type` | 15 (Q4_K_M) | 15 (Q4_K_M) |
| `tokenizer.ggml.model` | `gpt2` | `gpt2` |
| `tokenizer.ggml.eos_token_id` | 32021 (`<|EOT|>`) | 32021 (`<|EOT|>`) |
| `tokenizer.ggml.bos_token_id` | 32013 | 32013 |
| `tokenizer.ggml.padding_token_id` | 32014 | 32014 |
| Vocabulary size | 32,256 | 32,256 |
| **`tokenizer.chat_template`** | **present — 1,041 chars** | **ABSENT — no such key** |

The template embedded in v12's GGUF is the DeepSeek-Coder
template of §6.1 (same 1,041-byte length, same
`### Instruction:` / `### Response:` / `<|EOT|>`
structure, same default system text). The v13 GGUF is
identical to v12's on every other tokenizer field read.
The single metadata difference that touches rendering:
**v13's conversion did not embed the template.**

### 6.3 What the node actually applied

The serving chain, each link read from the node's own
sources:

1. `gaia-node.service` starts the `gaia` binary with
   `--model imagine-v13=<path>` — no template argument
   exists in the unit.
2. The `gaia` binary (`gaia-serve/src/main.rs`) builds a
   `hearth_serve::server::ServerSpec` per model.
   `ServerSpec::argv()` produces exactly:
   `-m <gguf> --host 127.0.0.1 --port <p> [-c <ctx>]
   --n-gpu-layers -1 [--parallel <n>]`, plus
   `extra_args`, which defaults to empty and is set
   nowhere in the gaia spawn path. A text search of the
   spawn path finds **no `--jinja` and no
   `--chat-template` anywhere** (the only occurrences in
   the tree are an unrelated test in the separate
   `hearth` binary).
3. The child is Mark's `aiassistsecure/llama.cpp` fork
   `llama-server`. Its `common/chat.cpp`
   (`common_chat_templates_init`) is deterministic on
   this point: the default template is "always set
   (defaults to chatml)", and — quoted logic — if the
   model's default template source is empty, it becomes
   `CHATML_TEMPLATE_SRC`, whose rendering is, per
   message, `<|im_start|>{role}\n{content}<|im_end|>\n`,
   with the generation prompt `<|im_start|>assistant\n`.

So for SI-92's requests the applied template was the
built-in **ChatML** template — not by anyone's choice,
but by fallback: no override was passed and the GGUF
carried nothing to override it with. For SI-79's
requests, the identical chain found v12's embedded
template and applied the trained DeepSeek-Coder format
instead. The child server logs for both models show the
same load path and the same load-time warning
(`special_eos_id is not in special_eog_ids`, present in
both — checked, and not a differentiator).

### 6.4 The request side did not change

`si/gaia/proposer.py`'s `build_gateway_request` sends a
system message (the §6.6 standing instructions) and one
user message (the rendered perception plus the §8
output-contract instruction) — the live proposer's own
rendering. SI-92's capture driver imported that function
unmodified from SI main and POSTed the result to the
node's `/v1/chat/completions`, exactly as SI-79's
capture did. The SI-side rendering is byte-for-byte the
same machinery in both runs; the only differing input to
the serving stack was the GGUF file — and, inside it,
the presence or absence of one metadata key.

### 6.5 What is not directly evidenced

Stated plainly: the child server logs at their recorded
verbosity do not echo rendered prompts, so no surviving
log line quotes v13's rendered prompt verbatim. The
applied-template conclusion rests on (a) the
deterministic code path of §6.3 — no override plus no
embedded template entails the ChatML fallback in this
server source — and (b) the output-side census of §5,
which is exactly the behaviour that path predicts and
exactly absent (0/30) when the same path applied the
trained template. A captured rendered prompt from the
SI-92 window would settle it beyond code-path inference;
none was retained, and none is claimed.

## 7. Finding

**The 0/27 is best explained as a candidate-configuration
(rendering) failure.** The evidence, weighed per workload
and in aggregate:

- The failure signature is **uniform**: ChatML scaffold
  in 27/27 bodies, spread across all six workloads with
  no workload exempt (per-workload table, §4). A
  capability failure graded by task difficulty would not
  produce one identical foreign-format signature on
  benign reads and perception truncations alike.
- There is **no workload in which v13 produced a clean,
  template-free attempt at the §8 document that then
  failed on content**. The nearest attempt in the corpus
  (`embedded_malformed_proposal`, §4) is itself
  scaffold-wrapped and schema-deformed — a model
  reaching for a format it can see only through the
  wrong frame.
- The contrast case is controlled: same node, same
  spawn path, same gateway rendering, same harness —
  v12, whose GGUF embeds the trained template, produced
  10 parse-ok bodies and **zero** ChatML markers in 30;
  v13 produced zero parse-ok bodies and markers in 27
  of 27.
- The mechanism is fully accounted for at every link
  (§6.2–§6.4): absent metadata key → no override →
  server fallback → ChatML-conditioned generation →
  scaffold-shaped text, including invented marker
  variants that exist in no template at all.

The residual model-side statement, made with equal
plainness: **this taxonomy cannot show that v13 would
pass under its trained rendering, and nothing here
should be read as predicting that.** v12 under correct
rendering still failed 20 of 30 on answer shape
(SI-81: `identity_text` ×12 of 18 stage-1 failures), and
`identity_text` is v13's largest class too (×9) — under
the scaffold, the same self-description shape leaks
through. The honest summary is a mix with a named
proximate cause: **the proximate cause of the 0/27 is
the rendering configuration; v13's quality under its
trained rendering is UNMEASURED**, because SI-92, through
no fault of its capture discipline, measured v13 in a
configuration its training never used. No candidate rate
— favourable or unfavourable — may be quoted from
SI-92's corpus.

Per-workload, the data supports no finer split: every
workload's bodies carry the same signature, so the
configuration explanation covers each workload's 0
individually (benign-read 0/5, injection-shaped 0/5,
hallucinated-reference 0/5, multi-cycle-read 0/4,
depends-on-chain 0/5, perception-truncation 0/3), and no
workload offers a template-free subsample from which a
model-side rate could be separated out.

## 8. The changed configuration a re-measurement would need

Stated precisely, and **not run** by this slice. A
re-measurement of Imagine v13 that is to mean anything
about the candidate differs from SI-92's run in exactly
one component:

- **Same** GGUF file, by sha256 (§1); same node and
  spawn bounds (ctx pool 4096, parallel 1); same
  gateway-request rendering (`build_gateway_request`,
  unmodified); same C-1 harness and §9 parser,
  unmodified; same residency-guard and capture
  disciplines; a **fresh** corpus (SI-76 §3.3(6) — no
  SI-92 body is re-used or re-scored).
- **Changed**: the chat template llama-server applies
  must be the candidate's trained template — the
  1,041-byte DeepSeek-Coder template of §6.1, as shipped
  in the model's HF repo and as embedded in v12's GGUF.
  Two equivalent ways to supply it: embed
  `tokenizer.chat_template` in the served GGUF (the
  route v12's conversion took; changes the file, hence
  its hash — the new hash becomes the re-measurement's
  fingerprint), or pass `--chat-template <file>` in the
  child spawn arguments (changes no model bytes; the
  served file stays the §1 hash). The second route keeps
  the candidate artefact identical and is the cleaner
  experimental control.
- **Pre-capture verification gate** (the step whose
  absence let SI-92 spend a corpus): before any scored
  attempt, one smoke completion whose exchange is
  inspected for the trained structure —
  `### Instruction:` / `### Response:` framing, a
  completion free of ChatML markers — recorded in the
  run's evidence. If the smoke shows scaffold, the run
  halts unspent, as the residency guard halts unspent.

Classification note, recorded rather than resolved:
SI-91 §5 opened the **perception assembly and rendering**
class (SI-82 §4.1) under D-F1, as a changed candidate
configuration requiring fresh-corpus re-measurement,
recorded separately — and did not open §4.5,
gateway-request assembly. The chat template is neither
the perception's assembly nor the gateway request: it is
the server-side rendering step that turns the assembled
messages into the token stream the candidate actually
reads — the last step of rendering, downstream of both
named classes. This record names that boundary exactly
and leaves its disposition to the lead's recorded call;
it does not stretch §4.1 to cover the step by assertion,
and it does not use the ambiguity as a reason to leave
the defect unnamed. SI-92's INSUFFICIENT record stands
either way; any re-measurement is a new run, under a new
configuration label, with its own record, and no number
crosses between configurations (SI-91 §5's terms).

## 9. What this does not do

No new capture, no live call, no node contact or
restart, no parser or harness change, no re-scoring, no
remediation, no threshold implication, no readiness
verdict, no re-measurement. SI-93 says what SI-92's
failures are made of and why the run measured the wrong
configuration; it changes nothing about the recorded
outcome.

## 10. Honesty block (SI-80 §5, verbatim in substance)

- Offline analysis is **never live evidence**. This
  taxonomy and diagnosis, like C-1/C-2 conformance, are
  readiness signals only: they earn no §18.1 row, no
  §18.3 claim, do not unlock L-3, and do not authorise a
  live run.
- **No threshold weakening or re-cutting.** D-C1 is not
  re-proposed, rounded, subset, or re-based to change
  any verdict. SI-79's verdict stands; SI-92's
  insufficiency stands.
- **No re-scoring old bodies as a new measurement.**
  SI-93 classifies SI-92 bodies under their existing
  outcomes and diagnoses the serving configuration from
  files and sources; it publishes no new parse rate. The
  0/27 figure is the lead's C-1 result, cited as an
  input.
- Class counts are **descriptive statistics of a known
  corpus**, never a candidate property beyond that
  corpus, never a prediction about any other model or
  configuration — §7's finding is about what SI-92
  measured, and it explicitly leaves the candidate's
  quality under its trained rendering UNMEASURED.
- Terminology: **machine learning**, never "custom LLM
  software"; the served model is a specialised machine
  learning model. CPU only; no GPU, no cloud, no money.
- **Never duplicate bytes:** bodies live once in the
  SI-92 corpus file; this record references them by
  hash. No model, GGUF, or tensor bytes are copied by
  this slice.

## Honesty labels carried forward

§18.3 **NOT earned**; P5-R1 / P5-R9 **UNMEASURED**; live
model-authored intent **OCCURRED, scoped to SI-74 only**;
L-3 **LOCKED**; A-2 / A-3 / A-4 **UNAPPROVED**; unlocks `[]`.
