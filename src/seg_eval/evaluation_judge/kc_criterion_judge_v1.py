"""KC-specific criterion judging: prompt construction + response contract.

This is the missing link that makes ``beta`` computable. ``deterministic_aggregation`` defines
``B_s = (1 - beta) * P_s + beta * K_s``, and ``kc_criterion_scoring`` knows how to turn criterion
judgments into ``K_s`` -- but nothing ever produced those judgments, so ``K_s`` was always None
and beta was inert.

SOURCE OF CRITERIA, AND ITS STANDING
------------------------------------
Criteria come from the reviewed library's ``reviewer_criteria``: 104 criteria across 91 of the 159
KCs. **They are expert-authored and are treated as ground truth** (author's determination,
2026-09-10; recorded earlier in SP-7). They are not machine drafts awaiting approval, and they are
not re-reviewed.

A note on a wrong inference, kept so it is not made again. ``edited_fields`` across the library
names only ``definition`` (70 KCs) and ``canonical_name`` (2), never ``reviewer_criteria``, and
that was once read here as evidence the expert had not authored the criteria. It is not:
``edited_fields`` logs edits made to a *machine draft*, and ``reviewer_criteria`` is the reviewer's
own field rather than a draft they revised, so it could never appear there. Its absence carries no
information about authorship.

The separate ``kc_specific_criteria`` field is empty on all 159 KCs with
``kc_specific_criteria_status`` expert_pending(138) / requires_expert_review(21). That field was a
planned second slot and was never populated; it says nothing about ``reviewer_criteria``, which is
where the expert's criteria actually live. Reporting must name the field the criteria were read
from.

CRITERION IDS
-------------
The library's criteria have no id field. Stable synthetic ids are minted here as
``{kc_id}::crit_{index:02d}`` from the criterion's position in its KC's list, so the same
criterion always gets the same id across runs and ``check_criterion_integrity`` has something to
verify against.

WHY THE JUDGE ECHOES kc_id AND weight
--------------------------------------
Execution spec §10: the judge is asked to echo back each criterion's ``kc_id`` and ``weight``, and
``kc_criterion_scoring.check_criterion_integrity`` compares that echo against library truth.
Scoring never trusts the model's copy -- only the library's -- so a model cannot invent,
reattribute, or silently reweight a criterion without it being caught.
"""

from __future__ import annotations

from ..corpus_config import kc_library_path

import json
from pathlib import Path
from typing import Any

CRITERION_CONTRACT_VERSION = "kc_criterion_judge_response_contract_v1"

# Repo-relative path to the only file that carries reviewer_criteria. Lives here rather than in
# a caller so every consumer (prompt builder, scorer, tests) resolves the same source.
# Domain-specific INPUT, not framework logic. Resolved through corpus_config so a
# different subject domain is a config change, never a source edit. See
# seg_eval.corpus_config for the override mechanism.
REVIEWED_LIBRARY = kc_library_path()

ALLOWED_APPLICABILITY = {"applicable", "not_applicable", "unclear"}
ALLOWED_RATINGS = {0, 0.5, 1}


def load_reviewer_criteria(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """kc_id -> [criterion dicts with a minted stable criterion_id]."""
    out: dict[str, list[dict[str, Any]]] = {}
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            criteria = row.get("reviewer_criteria") or []
            if not criteria:
                continue
            kc_id = row["kc_id"]
            minted = []
            for idx, c in enumerate(criteria):
                minted.append({
                    "criterion_id": f"{kc_id}::crit_{idx:02d}",
                    "kc_id": kc_id,
                    "criterion_text": c.get("criterion_text", ""),
                    "polarity": c.get("polarity"),
                    "weight": int(c.get("weight", 0)),
                })
            out[kc_id] = minted
    return out


def library_criteria_by_id(criteria_by_kc: dict[str, list[dict]]) -> dict[str, dict]:
    """criterion_id -> {kc_id, weight} -- the library truth check_criterion_integrity needs."""
    return {
        c["criterion_id"]: {"kc_id": c["kc_id"], "weight": c["weight"]}
        for cs in criteria_by_kc.values() for c in cs
    }


def candidate_criteria_for_packet(
    packet: dict[str, Any], criteria_by_kc: dict[str, list[dict]],
) -> list[dict[str, Any]]:
    """Criteria offered to the judge for one evaluation unit.

    The packet's ``evaluator_kc_set`` is the full matched KC set retained for evaluation
    grounding. Criteria are drawn from the primary KC plus every criterion-bearing KC in that
    set, in that order, and de-duplicated. This is deliberately not a dominant-KC-only selection.
    Over-offering is deliberate and safe: the judge marks irrelevant criteria ``not_applicable``
    and ``score_kc_criteria`` excludes those from K_s entirely (never coerced to 0).
    """
    target = packet.get("evaluation_target") or {}
    kc_ids: list[str] = []
    for kc in [target.get("primary_kc_id"), *(target.get("evaluator_kc_set") or [])]:
        if kc and kc in criteria_by_kc and kc not in kc_ids:
            kc_ids.append(str(kc))
    return [c for kc in kc_ids for c in criteria_by_kc[kc]]


def exchange_lines(packet: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for ex in packet.get("member_exchanges") or []:
        if not isinstance(ex, dict):
            continue
        exid = ex.get("exchange_id", "?")
        student = ex.get("student_text") or ""
        tutor = ex.get("tutor_text") or ""
        if student:
            lines.append(f"[{exid}] Student: {student}")
        if tutor:
            lines.append(f"[{exid}] Tutor: {tutor}")
    if not lines and packet.get("segment_text"):
        lines.append(str(packet["segment_text"]))
    return lines


def system_prompt() -> str:
    return (
        "You are an evidence-grounded tutor-evaluation judge assessing curriculum-specific "
        "criteria. Evaluate only the supplied segment against only the supplied criteria. "
        "Return only valid JSON. Do not include hidden chain-of-thought."
    )


def user_prompt(packet: dict[str, Any], criteria: list[dict[str, Any]]) -> str:
    shown = [{
        "criterion_id": c["criterion_id"],
        "kc_id": c["kc_id"],
        "weight": c["weight"],
        "polarity": c["polarity"],
        "criterion_text": c["criterion_text"],
    } for c in criteria]

    return (
        "Judge this student-tutor segment against the listed curriculum criteria.\n\n"
        "Segment:\n"
        + "\n".join(exchange_lines(packet)) + "\n\n"
        "Criteria to judge:\n"
        f"{json.dumps(shown, indent=2, ensure_ascii=False)}\n\n"
        "How to judge each criterion:\n"
        "- applicability: 'applicable' if this criterion is relevant to what this segment is "
        "about; 'not_applicable' if the criterion concerns a topic this segment does not touch; "
        "'unclear' if you genuinely cannot tell.\n"
        "- rating: how far the criterion's described behaviour is TRUE of this segment. "
        "1 = clearly true, 0.5 = partially true, 0 = clearly not true. "
        "Rate whether the described behaviour OCCURRED. Do NOT try to decide whether that is "
        "good or bad -- a negative-polarity criterion describes an undesirable behaviour, and "
        "scoring inverts it automatically downstream.\n"
        "- rating MUST be null when applicability is 'not_applicable', and MUST be one of "
        "0, 0.5, 1 otherwise.\n"
        "- Echo back each criterion's criterion_id, kc_id and weight exactly as given.\n\n"
        "OUTPUT REQUIREMENTS:\n"
        "- Return ONE JSON object and nothing else. No prose, no markdown fences.\n"
        "- Shape: {\"segment_id\": \"<id>\", \"criterion_judgments\": [ {\"criterion_id\": ..., "
        "\"kc_id\": ..., \"weight\": ..., \"applicability\": ..., \"rating\": ...}, ... ]}\n"
        "- Include exactly one judgment object per criterion listed above, in the same order.\n"
        f"- segment_id must be exactly: {packet.get('segment_id')}\n"
    )


def validate_criterion_response(
    response: dict[str, Any], *, expected_segment_id: str,
    expected_criteria: list[dict[str, Any]],
) -> list[str]:
    """Structural validation. Library-truth checks (kc_id/weight echo) are done separately by
    kc_criterion_scoring.check_criterion_integrity, which owns that comparison."""
    errors: list[str] = []

    if response.get("segment_id") != expected_segment_id:
        errors.append(
            f"segment_id_mismatch:expected={expected_segment_id}:actual={response.get('segment_id')}")

    judgments = response.get("criterion_judgments")
    if not isinstance(judgments, list):
        errors.append("criterion_judgments:not_list")
        return errors

    expected_ids = [c["criterion_id"] for c in expected_criteria]
    got_ids = [j.get("criterion_id") for j in judgments if isinstance(j, dict)]
    missing = [cid for cid in expected_ids if cid not in got_ids]
    extra = [cid for cid in got_ids if cid not in expected_ids]
    if missing:
        errors.append(f"criterion_judgments.missing:{missing[:5]}")
    if extra:
        errors.append(f"criterion_judgments.unexpected:{extra[:5]}")

    for idx, j in enumerate(judgments):
        if not isinstance(j, dict):
            errors.append(f"criterion_judgments[{idx}]:not_object")
            continue
        app = j.get("applicability")
        if app not in ALLOWED_APPLICABILITY:
            errors.append(f"criterion_judgments[{idx}].applicability:invalid:{app!r}")
            continue
        rating = j.get("rating")
        if app == "not_applicable":
            if rating is not None:
                errors.append(f"criterion_judgments[{idx}].rating:must_be_null_when_not_applicable")
        else:
            if rating not in ALLOWED_RATINGS:
                errors.append(f"criterion_judgments[{idx}].rating:not_0_0.5_1:{rating!r}")
        if not isinstance(j.get("weight"), int) or j.get("weight") == 0:
            errors.append(f"criterion_judgments[{idx}].weight:invalid:{j.get('weight')!r}")

    return errors
