"""Load the auxiliary judge passes from explicit paths.

WHY THIS IS A MODULE AND NOT A HELPER IN ONE SCRIPT
---------------------------------------------------
``compute_tutor_scores.py`` already knows how to turn the KC-criterion pass into ``K_s`` and the
macro pass into ``M``, but it resolves both through a hardcoded map of the three tutor-variant
arms. The console evaluates one arbitrary dialogue, so it needs the same logic addressed by path.

Copying the loop was the obvious option and the wrong one: the ``K_s`` path carries the criterion
*integrity* checks -- the library's ``kc_id`` and ``weight`` are authoritative and the model's
echoed values are only ever compared against them -- and a second copy of that is a place for the
two to drift apart silently. So the logic lives here once, addressed by path, and the arm-shaped
script keeps working unchanged.

WHAT IS DELIBERATELY NOT DONE
-----------------------------
Nothing here repairs a bad response. A unit whose criterion response fails the contract, or whose
echoed criterion metadata disagrees with the library, contributes no ``K_s`` at all -- it does not
contribute a zero. Absent and zero mean different things to the aggregation, and only one of them
is true.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def load_kc_criterion_scores(
    responses_path: Path | str,
    prompts_path: Path | str,
    library_path: Path | str,
) -> tuple[dict[str, float], dict[str, Any]]:
    """``segment_id -> K_s`` plus a stats block describing what was accepted and rejected.

    Returns an empty mapping (and ``status: not_run``) when the pass has not run, so beta is
    simply inert rather than the run failing.
    """
    responses_path, prompts_path = Path(responses_path), Path(prompts_path)
    if not responses_path.exists():
        return {}, {"status": "not_run", "reason": f"no responses at {responses_path}"}

    from ..evaluation_judge.kc_criterion_judge_v1 import (
        load_reviewer_criteria, library_criteria_by_id, validate_criterion_response,
    )
    from .kc_criterion_scoring import (
        CriterionJudgment, CriterionScoringError, score_kc_criteria, check_criterion_integrity,
    )

    criteria_by_kc = load_reviewer_criteria(Path(library_path))
    truth = library_criteria_by_id(criteria_by_kc)
    by_id = {c["criterion_id"]: c for cs in criteria_by_kc.values() for c in cs}
    expected = {r["segment_id"]: r["criterion_ids"] for r in _read_jsonl(prompts_path)}

    k_by_segment: dict[str, float] = {}
    stats: dict[str, Any] = {
        "units": 0, "contract_valid": 0, "judgments_scored": 0,
        "not_applicable": 0, "unclear": 0,
        "integrity_errors": [], "contract_errors": [],
    }

    for row in _read_jsonl(responses_path):
        stats["units"] += 1
        sid = row.get("segment_id")
        try:
            payload = json.loads(row["raw_response_text"])
        except Exception:  # noqa: BLE001 - any parse failure is the same outcome
            stats["contract_errors"].append(f"{sid}:unparseable")
            continue

        exp_criteria = [by_id[c] for c in expected.get(sid, []) if c in by_id]
        errors = validate_criterion_response(payload, expected_segment_id=sid,
                                             expected_criteria=exp_criteria)
        if errors:
            stats["contract_errors"].append(f"{sid}:{errors[0]}")
            continue

        judgments, rejected = [], False
        for judgment in payload.get("criterion_judgments", []):
            cid = judgment.get("criterion_id")
            # Library truth is authoritative; the model's echoed kc_id/weight is only checked
            # against it, never used. A model cannot invent, reattribute or reweight a criterion.
            integrity = check_criterion_integrity(
                criterion_id=cid, reported_kc_id=judgment.get("kc_id"),
                reported_weight=judgment.get("weight"), library_criteria_by_id=truth)
            if integrity:
                stats["integrity_errors"].append(f"{sid}:{integrity[0]}")
                rejected = True
                break
            try:
                judgments.append(CriterionJudgment(
                    criterion_id=cid, kc_id=truth[cid]["kc_id"], weight=truth[cid]["weight"],
                    applicability=judgment["applicability"], rating=judgment.get("rating")))
            except (CriterionScoringError, KeyError) as exc:
                stats["contract_errors"].append(f"{sid}:{exc}")
                rejected = True
                break
        if rejected or not judgments:
            continue

        result = score_kc_criteria(judgments)
        stats["contract_valid"] += 1
        stats["judgments_scored"] += result.applicable_count
        stats["not_applicable"] += result.not_applicable_count
        stats["unclear"] += result.unclear_count
        if result.kc_s is not None:
            k_by_segment[sid] = result.kc_s

    stats["status"] = "ok"
    stats["units_with_k_s"] = len(k_by_segment)
    stats["integrity_errors"] = stats["integrity_errors"][:5]
    stats["contract_errors"] = stats["contract_errors"][:5]
    return k_by_segment, stats


def load_macro_score(responses_path: Path | str) -> tuple[float | None, dict[str, Any]]:
    """Mean of the macro dimension scores, or ``None`` when the response is absent or invalid.

    Nulls among the dimensions are excluded rather than coerced to 0, matching the rest of the
    aggregation. ``None`` propagates to ``T`` falling back to ``D_micro`` alone, which the report
    states rather than hiding.
    """
    rows = _read_jsonl(Path(responses_path))
    if not rows:
        return None, {"status": "not_run"}

    raw = rows[0].get("raw_response_text")
    if not isinstance(raw, str):
        return None, {"status": "invalid", "reason": "no raw_response_text"}
    try:
        payload = json.loads(raw)
    except Exception:  # noqa: BLE001
        return None, {"status": "invalid", "reason": "response is not JSON"}
    if not isinstance(payload, dict):
        return None, {"status": "invalid", "reason": "response is not an object"}

    scores = payload.get("dimension_scores") or {}
    values = [float(v) for v in scores.values() if isinstance(v, (int, float))]
    if not values:
        return None, {"status": "invalid", "reason": "no numeric dimension_scores"}
    return sum(values) / len(values), {
        "status": "ok",
        "dimension_scores": {k: v for k, v in scores.items()},
        "dimensions_scored": len(values),
        "rationale": payload.get("rationale") or payload.get("overall_rationale"),
    }


def load_answer_revelation(responses_path: Path | str) -> tuple[dict[str, dict], dict[str, Any]]:
    """``segment_id -> {revelation, warrant, ...}`` from the answer-revelation pass.

    Reported alongside the score rather than folded into it: ``solution_control`` is derived from
    these observations in code, and the revelation rate itself is the statistic comparable to
    MRBench's headline number.
    """
    rows = _read_jsonl(Path(responses_path))
    if not rows:
        return {}, {"status": "not_run"}

    by_segment: dict[str, dict] = {}
    counts = {"not_revealed": 0, "revealed_partial": 0, "revealed_full": 0}
    unwarranted = 0
    for row in rows:
        sid = row.get("segment_id")
        raw = row.get("raw_response_text")
        if not isinstance(raw, str):
            continue
        try:
            payload = json.loads(raw)
        except Exception:  # noqa: BLE001
            continue
        revelation = payload.get("revelation") or payload.get("answer_revelation")
        warrant = payload.get("warrant") or payload.get("warrant_type")
        if revelation in counts:
            counts[revelation] += 1
        if revelation in ("revealed_partial", "revealed_full") and warrant in (None, "", "no_warrant"):
            unwarranted += 1
        by_segment[sid] = {"revelation": revelation, "warrant": warrant,
                           "evidence": payload.get("evidence") or payload.get("tutor_quote")}

    scored = sum(counts.values())
    stats = {
        "status": "ok",
        "units": len(by_segment),
        "counts": counts,
        "revealed_rate": ((counts["revealed_partial"] + counts["revealed_full"]) / scored)
        if scored else None,
        "unwarranted_rate": (unwarranted / scored) if scored else None,
    }
    return by_segment, stats
