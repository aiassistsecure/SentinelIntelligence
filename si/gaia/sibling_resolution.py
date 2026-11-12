"""Gaia PROPOSE — resolution at journal time (SI-97 §6,
SI-98).

The intra-document dependency amendment's walk machinery,
shared by every admit walk (the replay runner in
:mod:`si.gaia.proposer` and the live task walk in
:mod:`si.gaia.task` alike): a per-walk resolution map built
from the walk's own admission results rewrites each member's
``{"sibling": k, ...}`` dependency entries to the journal
seq the walk assigned that sibling, before the member is
handed to admission — so the gate prices only the resolved
form under its existing rules (SI-97 §7), and a member whose
sibling was never journaled by its walk is refused at
resolution (SI-97 §6.4): one journaled refusal record, never
gated, never queued, never executed, and never admitted with
the dependency dropped.

This module is deliberately dependency-light (stdlib only,
the journal handle passed in): it is imported by both the
proposer and the task loop, and it prices nothing itself —
parsing is :func:`si.gaia.proposer.parse_proposal`'s
business, gating is the gate's, execution is the queue's.

CPU only, stdlib only. Terminology: machine learning
infrastructure throughout.
"""
from __future__ import annotations

import copy

#: Journal kind of the §6.4 (SI-97) resolution-refusal record,
#: pinned by SI-98 (decision D-7 discipline): a dependent whose
#: sibling was never journaled by its walk is refused at
#: resolution — a pre-gate, per-member refusal in the §9.5
#: family. It is not a self-edit, carries no gate report and
#: no verdict, and is never an input to the queue fold.
RESOLUTION_REFUSAL_KIND = "proposal_resolution_refusal"

#: The closed cause vocabulary of a resolution refusal (SI-97
#: §6.4): one code — ``sibling_not_journaled`` — with the
#: sub-case carried as detail, not as a second taxonomy.
RESOLUTION_REFUSAL_CAUSE = "sibling_not_journaled"
SUBCASE_SIBLING_REFUSED = "sibling_refused_at_resolution"
SUBCASE_ADMISSION_NO_SEQ = "sibling_admission_returned_no_seq"
SUBCASE_NO_MEMBER_ADMITTED = "no_member_admitted_at_position"

#: The refusal's standing label (SI-97 §6.4): a refusal and a
#: gate rejection are different facts, and the record says so.
RESOLUTION_REFUSAL_LABEL = (
    "refused at resolution before the gate (SI-97 §6.4) — "
    "not a gate REJECT, not an ESCALATE; never gated, never "
    "queued, never executed; no path admits this member with "
    "the dependency dropped")


def resolve_sibling_dependencies(intent, *, resolution_map,
                                 refused_positions=frozenset(),
                                 visited_positions=frozenset()):
    """Resolve one parsed intent's sibling entries against a
    walk's resolution map (SI-97 §6.2) — the per-member step
    every admit walk applies, in document order, before
    handing the member to admission.

    ``resolution_map`` is the walk's own map M (position ->
    journal seq it assigned that member, built by the walk
    itself from its admission results — §6.2's one-source
    rule: the journal is never searched mid-walk).
    ``refused_positions`` / ``visited_positions`` are the
    walk's bookkeeping used only to name a failure's sub-case
    (§6.4): a position the walk refused at resolution, a
    position the walk visited but whose admission returned no
    seq, or a position the walk never admitted at all.

    Returns a dict:

    - ``resolved`` True: ``intent`` is the member's effective
      intent — its ``depends_on`` is the authored list in
      authored order with each sibling entry replaced in
      place by ``{"seq": M[k], "requirement": r}`` — and, when
      the authored list carried a sibling entry, its
      ``authored_depends_on`` preserves the authored entries
      verbatim (§8.1). An intent with no sibling entries is
      returned unchanged (``had_sibling`` False).
    - ``resolved`` False: the member is refused at resolution
      (§6.4); ``failed`` lists every entry that could not
      resolve, verbatim, each with its ``sibling`` position
      and its named sub-case. The caller journals exactly one
      refusal record for the member and does not admit it.
    """
    authored = list(getattr(intent, "depends_on", None) or [])
    if not any(isinstance(e, dict) and "sibling" in e
               for e in authored):
        return {"resolved": True, "intent": intent,
                "had_sibling": False, "failed": []}
    resolved_deps = []
    failed = []
    for entry in authored:
        if isinstance(entry, dict) and "sibling" in entry:
            k = entry["sibling"]
            if k in resolution_map:
                resolved_deps.append({
                    "seq": resolution_map[k],
                    "requirement": entry.get(
                        "requirement", "outcome")})
                continue
            if k in refused_positions:
                sub_case = SUBCASE_SIBLING_REFUSED
            elif k in visited_positions:
                sub_case = SUBCASE_ADMISSION_NO_SEQ
            else:
                sub_case = SUBCASE_NO_MEMBER_ADMITTED
            failed.append({"entry": copy.deepcopy(entry),
                           "sibling": k, "sub_case": sub_case})
        else:
            resolved_deps.append(copy.deepcopy(entry))
    if failed:
        return {"resolved": False, "intent": None,
                "had_sibling": True, "failed": failed}
    effective = copy.deepcopy(intent)
    effective.depends_on = resolved_deps
    effective.authored_depends_on = copy.deepcopy(authored)
    return {"resolved": True, "intent": effective,
            "had_sibling": True, "failed": []}


def build_resolution_refusal_record(*, task_seq, position,
                                    failed, attempt_seq=None,
                                    response_seq=None,
                                    batch_seq=None,
                                    replay_only=False,
                                    replay_label=None) -> dict:
    """The §8.3 (SI-97) resolution-refusal record for one
    member (pure builder — journals nothing).

    Carries: the task seq; the member's position in its
    document; the authored entries that failed to resolve,
    verbatim, each with its sub-case; the closed cause code
    ``sibling_not_journaled`` with the first failure's
    sub-case as the record's headline sub-case; and linkage
    to the cycle's attempt / response / batch records where
    the caller provides them. It is not a self-edit, carries
    no gate report and no verdict, and is never an input to
    the queue fold (§8.3)."""
    first = failed[0] if failed else {}
    record = {
        "kind": RESOLUTION_REFUSAL_KIND,
        "task_seq": task_seq,
        "position": position,
        "cause": RESOLUTION_REFUSAL_CAUSE,
        "sub_case": first.get("sub_case"),
        "detail": (
            f"member at position {position} refused at "
            f"resolution: sibling {first.get('sibling')} held "
            f"no assigned journal seq in this walk "
            f"({first.get('sub_case')}) — cause "
            f"{RESOLUTION_REFUSAL_CAUSE} (SI-97 §6.4)"),
        "unresolved_entries": copy.deepcopy(list(failed)),
        "label": RESOLUTION_REFUSAL_LABEL,
        "replay_only": bool(replay_only),
    }
    if replay_only and replay_label is not None:
        record["replay_label"] = replay_label
    if attempt_seq is not None:
        record["attempt_seq"] = attempt_seq
    if response_seq is not None:
        record["response_seq"] = response_seq
    if batch_seq is not None:
        record["batch_seq"] = batch_seq
    return record


def journal_resolution_refusal(journal, record: dict) -> int:
    """Journal one §8.3 refusal record via the existing
    experience-record path; returns its journal seq."""
    journal.record_experience(
        RESOLUTION_REFUSAL_KIND, copy.deepcopy(record), "")
    rows = journal.backend.find(
        "experiences", filter={"kind": RESOLUTION_REFUSAL_KIND},
        sort="-seq", limit=1)
    return rows[0]["seq"] if rows else journal.tip()[0]
