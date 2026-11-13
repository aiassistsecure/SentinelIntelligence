# SI-112 — PFL-1: the pre-flight template lock

Implements SI-111 §3.3 as designed, under lead decision
D-111-2 (template-hash registration is a **hard**
eligibility precondition for any graded or diagnostic
capture; TEMPLATE-UNRESOLVED candidates are refused,
fail-closed; no waivers).

## The incident this prices

SI-93's diagnosis: the Imagine v13 GGUF carries no
`tokenizer.chat_template`; llama-server silently fell back
to its built-in ChatML template; SI-92 measured 0/27
parse-ok against a chat format the candidate never trained
on — a serving artifact, not a model verdict, discovered
only after the budget was spent. Since SI-94 the serving
side has been corrected by hand (a wrapper passing
`--chat-template-file`). PFL-1 makes it a lock.

## What landed

**`si/gaia/preflight.py`** (new, stdlib only):

- **GGUF metadata probe** — a real GGUF reader (header +
  metadata KV walk; every value type, arrays included,
  skipped correctly; tensor data never read). It prices
  exactly one key, `tokenizer.chat_template`, and refuses
  loudly (`GGUFProbeError`) on anything that is not a
  well-formed GGUF — never filename or size inference.
  Verified against the recorded SI-93 ground truth: v13
  Q4_K_M → absent; v12 Q4_K_M → present, 1,041 bytes,
  sha256 `993c0af7…`, byte-identical to the external file
  the node serves v13 with.
- **Candidate registration** (§3.3(a)) —
  `register_candidate` records embedded presence + hash;
  when absent, the externally supplied template's hash;
  when neither, **TEMPLATE-UNRESOLVED** with an INELIGIBLE
  eligibility line. Operator CLI:
  `python -m si.gaia.preflight register|probe`.
- **Pre-flight check** (§3.3(b)) — `check_preflight`
  resolves the template the server will actually apply
  (spawn-path `--chat-template-file` target wins, as it
  does in serving; else the served GGUF's embedded
  template; else a caller-supplied served hash), hashes
  it, and compares it to the registration. Mismatch,
  absence, an unresolved registration, or an unresolvable
  serving configuration → **REFUSED**, with both hashes,
  the comparison, and the decision in the check record.
  `require_preflight` raises `PreflightRefused` carrying
  that record, so a refusal is itself evidence.

**Wiring — both capture entry points, no scoring change:**

- `si/gaia/conformance_capture.py`: `capture_corpus`
  accepts a registration + served-template sources, runs
  the pre-flight before anything is built or spent, and
  carries the record in the capture log and every progress
  payload. The operator CLI refuses a fresh capture with
  no `--registration` (exit 2); a resume is bound by its
  original capture's PASS record, carried forward in the
  log. `si/gaia/conformance.py` is **not modified** —
  `score_corpus` behaviour for already-captured corpora is
  byte-for-byte identical.
- `si/gaia/multicycle.py`: `MultiCycleDriver` accepts a
  registration + a `served_template_probe` callable (the
  caller's serving-side evidence; the driver still contacts
  no model itself). `run()` pre-flights before the first
  task is journaled.
- **Restart re-verification** (§3.3(c)): in both capture
  paths the residency-guard poll re-resolves / re-probes
  the served template hash on every poll; a mid-capture
  change halts the capture with **no attempt spent**, under
  the guard's existing halt discipline, and every recheck
  is appended to the pre-flight record.

## Mechanical choices (recorded per the build brief)

- Registration and served hashes are bare sha256 hex (the
  SI-111 §3.2 citation form), not the `sha256:`-prefixed
  form used for body hashes elsewhere.
- When a registration is supplied to `capture_corpus`, the
  lock is mandatory and fail-closed; with none, the
  function keeps its historical behaviour so recorded
  corpora (SI-94/100/105) remain resumable — the hard
  precondition binds at the operator CLI, where graded and
  diagnostic captures are actually launched, and for any
  caller that registers (D-111-2's population).
- The multicycle driver takes the served hash from an
  injected probe callable rather than probing files
  itself, matching its injected-collaborator design
  (residency guard, proposal source).

## Honesty (§3.4)

PFL-1 is harness-integrity infrastructure. It changes no
recorded verdict — SI-92's INSUFFICIENT stands — improves
no model score, earns no §18.1 row and no §18.3 progress,
and does not reopen the closed F-line: it does not touch
the parser, the output contract, or any teaching text. Its
claim is narrow and complete: *the SI-93 incident class — a
capture spent against a silently wrong chat template —
cannot recur unnoticed*, because the next capture either
starts against the candidate's recorded trained template
or does not start.

## Tests

`tests/test_gaia_preflight_si112.py` (21 tests): probe
correctness over synthetic GGUFs (typed values, string
arrays, garbage/truncation refused), registration's three
outcomes, the check's PASS / mismatch / unresolved /
unresolvable refusals (including the exact SI-93 fallback
shape), the recorded SI-93 ground truth against the real
v12/v13 GGUFs where present, both capture wirings
(refusal spends no attempt; a mid-capture template swap
halts at the guard), and the CLI's no-registration refusal.
Affected existing subsets re-run green: conformance capture
(SI-79), multicycle driver (SI-109), C-1 harness, SI-110
fold-supplied scoring.
