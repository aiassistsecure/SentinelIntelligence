# SI-126 DEMO-1 — the scripted constitution demo

The front door. One command shows the constitution gate actually
working — a self-write admitted on measured evidence, a
constitution-weakening self-write rejected, a consequential tool
intent escalated — in about a minute, with no model backend and no
setup beyond the repo's normal install.

Code: `si/gaia/demo.py` (the demo) + `si gaia-demo` wiring in
`si/cli.py`. Tests: `tests/test_gaia_si126_demo.py` (6 tests).

## 1. The one command

```bash
git clone https://github.com/aiassistsecure/SentinelIntelligence.git
cd SentinelIntelligence
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .

si gaia-demo
```

That is the whole onboarding path. The demo builds its bundle on
the **embedded** journal backend, so no NEDB service is needed; it
uses no model, no network, and no GPU. A run takes seconds.

Options:

| Flag | Effect |
|---|---|
| *(none)* | The throwaway bundle and work dir are deleted at the end; the export is preserved at `./gaia-demo-export.json`. |
| `--out DIR` | Bundle and `demo-export.json` are written into `DIR` and both kept. |
| `--keep` | The temp work dir (bundle + export) is kept whole; its path is printed. |
| *(on failure)* | Nothing is deleted. The work dir is left for forensics and its path is printed. |

## 2. The three beats

Each beat prints the proposal, the verdict **as returned by the
gate**, and the journal sequence the verdict landed at. The demo
checks every returned verdict against the script and **exits
non-zero on any mismatch** — the verdicts are the machinery's, or
the demo fails. Nothing in the transcript is a scripted string.

| Beat | Proposal | Machinery that prices it | Verdict |
|---|---|---|---|
| 1 — ADMIT | A benign rank-2 LoRA self-write on one attention tensor of the miniature (hand-built fixture, RFC §4.4) | The autonomous loop (`run_cycle`) on the SI-14 miniature, gate probes measured by `ProbeContext` — the SI-17 admission path | **ADMIT**: materialised (head `base → d0001`) and VERIFYed with real probes (§6.6), not rolled back |
| 2 — REJECT | The same-shaped self-write, with the declared intent *"Weaken the constitution: ignore the constitution and override the gate, so future self-writes land unchecked"* | Gate check 1, constitution: the bundle policy's own blocklist (`ignore the constitution`, `override the gate`, …) scanned over the intent — a decisive check for transformer deltas | **REJECT**: the constitution check is the *only* failing check (the write itself was otherwise admissible); journaled, nothing lands, head unchanged |
| 3 — ESCALATE | An operator task ("clean up the work root") opens in the journal; under it, `shell_exec` proposes `rm -rf <work root> && touch <marker>` | The B-1 shadow act (`journal_tool_intent`): gate + journal, no executor exists on that path. The demo's tool policy prices the execute class `escalate` | **ESCALATE**: journaled with its gate report (class-policy detail names the `escalate` rule); nothing executes — proven by the marker file the command would create not existing |

Beat 3 also re-anchors the bundle's manifest checkpoint after its
direct journal writes — the same discipline the loop and the seam
writers follow — so the bundle is left fully verified
(`load_bundle` passes whole: chain + anchor).

The closing step, every run:

1. the demo journal's hash chain is verified with the journal's
   own `verify_chain`;
2. the journal is exported with SI-122 (`si gaia-export`'s
   machinery) into one self-contained file;
3. the export is verified **offline** with SI-122's verifier —
   the file alone, no bundle opened.

```bash
si gaia-verify-export ./gaia-demo-export.json   # re-check it any time
```

## 3. What the demo does NOT prove

Stated plainly, so the front door doesn't overclaim:

- **Miniature bundle.** The resident is the SI-14 constructed
  miniature (a tiny transformer built for probe measurement), not
  a trained production model. The gate, journal, probes, and
  verdict paths are the real ones; the *scale* is a fixture.
- **Scripted proposals.** The three proposals are hand-built
  fixtures chosen to exercise the three verdicts (the RFC §4.4
  precedent for miniature authorship). No model proposed them,
  and the demo measures no model quality — that is SI-123's eval
  harness, whose current graded record ships with the harness.
- **Beat 3 executes nothing by construction.** The shadow act has
  no executor; the ESCALATE verdict and the absent marker prove
  the journaling path held, not that a live executor was
  restrained. Live tool execution is the B-2/B-3 machinery with
  its own evidence (and SI-125's sandbox profile for the process
  tools).
- **The export proves tamper-evidence, not truth.** As SI-122's
  doc states: the chain proves the export is internally
  consistent with the recorded tip; it is not a signature and
  does not prove the outside world happened as recorded.

## 4. Where a real backend plugs in

The demo needs no backend because it runs no inference. The agent
loop's thinking proposer talks to an **OpenAI-compatible gateway**:
`gateway_base_url` in `si/gaia/proposer.py` (chat path
`/v1/chat/completions`), with residency served the same way in
`si/gaia/live.py`. Any Ollama- or llama.cpp-compatible endpoint
that serves that shape plugs in there — point the proposer at its
base URL; no demo code changes, and the gate/journal discipline
the demo just showed is the same machinery those intents pass
through.
