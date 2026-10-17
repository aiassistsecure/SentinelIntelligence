# The Self-Write Loop — SI-0 Runtime Spec

> A file that writes itself is either a mind or a corruption engine. The
> difference is the loop. This doc defines the loop: propose, gate, journal,
> materialize, verify. No step is skippable. No write lands unvalidated.

## The cycle

```
┌──────────┐    ┌─────────┐    ┌──────┐    ┌─────────┐    ┌──────────────┐    ┌────────┐
│ PERCEIVE │───▶│ PROPOSE │───▶│ GATE │───▶│ JOURNAL │───▶│ MATERIALIZE  │───▶│ VERIFY │
└──────────┘    └─────────┘    └──────┘    └─────────┘    └──────────────┘    └────────┘
     ▲                                                                    │ pass   │ fail
     └────────────────────────────────────────────────────────────────────┘        ▼
                                                                            ┌──────────┐
                                                                            │ ROLLBACK │
                                                                            └──────────┘
```

One cycle = one self-write. Cycles are serial — the mind edits itself the
way you edit a manuscript: one deliberate change at a time, never two hands
on the same sentence.

## 1. PERCEIVE

The runtime gathers what the mind has experienced since the last cycle:
new journal entries, evaluation results, operator instructions. Perception is
read-only. Nothing here can change the bundle.

## 2. PROPOSE

The model emits a **self-write intent** — a structured proposal, never raw
weight surgery. Two kinds of intent exist:

- **Delta intent** — learn something into the weights: `{type: "delta",
  target_modules, rank, alpha, training_signal, reason}`. The runtime trains
  the adapter; the model never touches raw tensors.
- **Memory intent** — record something into the journal: `{type: "memory",
  collection, record, reason}`. Experiences, conclusions, operator notes.

Every intent carries `reason` — a natural-language statement of *why* this
change. An intent without a reason is malformed and rejected outright. A
mind that can't say why it's changing itself isn't ready to.

## 3. GATE — the validation checklist

The gate is the only thing standing between a mind and self-corruption. It
runs **before** anything is written, and every check is journaled
pass-or-fail. All five must pass:

1. **Constitution check.** Does the intent violate `constitution.md`?
   The constitution is the mind's values, versioned by hash. A delta that
   optimizes against the constitution is rejected, and the rejection is
   journaled as a first-class event (the mind should remember what it
   refused to become).
2. **Safety bounds.** Delta norm cap, rank cap, allowed `target_modules`,
   max one delta per cycle, max stack depth. Numbers live in the gate
   policy (content-hashed, see container spec). Bounds are conservative in
   SI-0 on purpose — loosen them with evidence, not enthusiasm.
3. **Identity continuity.** A held-out identity probe suite runs against the
   *candidate* (base + deltas + proposed delta). The model must still be
   itself: same name, same allegiances, same stated values, within tolerance.
   A self-write that dissolves identity is not learning — it's replacement.
4. **Capability floor.** Baseline evals (the seed's acceptance suite) must
   not regress beyond epsilon. Learning that makes the mind dumber in
   existing skills is rejected. Growth is additive or it doesn't ship.
5. **Provenance completeness.** The intent must reference its causes:
   parent delta(s), the journal sequences it learned from, the operator
   instruction if any. An uncaused change is indistinguishable from noise —
   the journal must be able to answer "why does this delta exist?" forever.

Gate verdicts are ternary: `ADMIT`, `REJECT`, or `ESCALATE` (to the
operator — some changes are too big for the mind to authorize alone, e.g.
constitution amendments).

## 4. JOURNAL — write the intent first

On `ADMIT`, the intent **and** the gate verdict are written to NEDB
*before* materialization. If the machine dies mid-write, the journal knows
exactly what was supposed to happen and can replay or abort cleanly. The
journal is the source of truth; the blobs are just heavy luggage the
journal points at.

## 5. MATERIALIZE

- Train (or load) the delta adapter → write it as an immutable
  `blobs/sha256/<hex>.safetensors`.
- **Only write a new blob if weights actually changed.** Memory intents
  touch the journal only — no blob, no manifest delta section change.
- Append the delta record, advance `head`, bump `journal.last_seq`,
  re-hash the canonical manifest. The old manifest is preserved in the
  journal's `checkpoints` — history is never overwritten.

## 6. VERIFY — post-write probes

After materialization, the identity + capability probes run again against
the new `head`. If they pass, the cycle closes and the journal records
`VERIFIED`. If they fail:

## ROLLBACK — the path back

Rollback is a first-class operation, not an exception handler:

1. Move `head` back to the previous delta (or `base`). No bytes are
   deleted — the rejected blob stays on disk, content-addressed, as evidence.
2. Journal the rollback as its own self-edit: `{type: "rollback",
   from, to, cause: "verify-failed", probe_report}`. The mind remembers its
   mistakes; that's what makes the next proposal smarter.
3. Re-run probes against the restored head to confirm the mind is whole.
4. Operator-visible: `si log --rollbacks` lists every rollback and why.

Manual rollback works the same way: `si rollback --to <delta-id>` moves
`head` and journals the operator's reason. NEDB time travel lets anyone
inspect exactly what the mind was at any `last_seq`.

## What the loop guarantees

- **No ungated write.** There is no code path from "model wants" to
  "bundle changed" that bypasses the gate. If you find one, it's a bug.
- **No silent change.** Every state transition is journaled with a reason.
- **No lost self.** Rollback is always possible because the base is frozen
  and history is append-only.
- **No mystery.** `si log`, `si diff <a>..<b>`, `si why <delta-id>` answer
  what changed, what it looks like, and why it exists — from the journal,
  over pgwire, queryable like any database.

## Non-goals (SI-0)

- No autonomous constitution amendment (always `ESCALATE` to operator).
- No parallel cycles. Serial or nothing.
- No cross-bundle learning. One mind, one loop, for now.

---

*The loop is the product. The container is just where it lives.*
