from pathlib import Path
import json

from seg_eval.segmentation_candidate_diagnostic_v1 import (
    classify_failure_mode,
    build_candidate_diagnostic,
)


def test_formula_context_rescue_is_detected():
    fields = {
        "resolved_kc_id": "KC_WRONG",
        "resolved_kc_name": "F-Measure",
        "confidence_band": "low",
        "assignment_scope": "branch_context",
        "resolution_action": "pass2_left_context_continuation",
        "evaluator_kc_set": [],
    }
    candidates = [{"kc_id": "KC_CI", "kc_name": "Confidence Interval for Accuracy", "score": 1.0}]
    text = {
        "student_text": "Calculate the interval for p=0.85 and N=200.",
        "tutor_text": "Use the formula.",
        "combined_text": "Calculate the interval for p=0.85 and N=200. Use the formula.",
    }
    modes, notes = classify_failure_mode(fields, candidates, text, "hard_review_required")
    assert "formula_or_calculation_turn" in modes
    assert "context_rescue_or_continuation_used" in modes


def test_recap_turn_is_detected():
    fields = {
        "resolved_kc_id": "KC_X",
        "resolved_kc_name": "K-Means Algorithm",
        "confidence_band": "low",
        "assignment_scope": "single_kc",
        "resolution_action": "pass1_closed_set",
        "evaluator_kc_set": [],
    }
    candidates = []
    text = {
        "student_text": "Can you connect all these exercises together?",
        "tutor_text": "Here is a recap.",
        "combined_text": "Can you connect all these exercises together? Here is a recap.",
    }
    modes, notes = classify_failure_mode(fields, candidates, text, "hard_review_required")
    assert "recap_or_strategy_turn" in modes
    assert "candidate_trace_missing_or_not_exported" in modes


def test_build_candidate_diagnostic_fixture(tmp_path):
    run = tmp_path / "run"
    v9c = run / "v9c"
    v9e = run / "v9e"
    q = run / "quality_audit"
    v9c.mkdir(parents=True)
    v9e.mkdir(parents=True)
    q.mkdir(parents=True)

    (v9c / "summary.json").write_text('{"segment_count":1}', encoding="utf-8")
    (v9e / "summary.json").write_text('{"segment_count":1}', encoding="utf-8")

    (v9e / "contactless_exchange_assignments.jsonl").write_text(
        json.dumps({
            "exchange_id": "ex1",
            "resolved_kc_id": "KC_WRONG",
            "resolved_kc_name": "F-Measure",
            "confidence_band": "low",
            "assignment_scope": "branch_context",
            "resolution_action": "pass2_left_context_continuation",
            "student_text": "Calculate the interval.",
            "tutor_text": "Use the formula."
        }) + "\n",
        encoding="utf-8",
    )

    (v9c / "contactless_exchange_assignments.jsonl").write_text(
        json.dumps({
            "exchange_id": "ex1",
            "candidates": [
                {"kc_id": "KC_CI", "kc_name": "Confidence Interval for Accuracy", "score": 1.0},
                {"kc_id": "KC_WRONG", "kc_name": "F-Measure", "score": 0.5}
            ]
        }) + "\n",
        encoding="utf-8",
    )

    (q / "segmentation_quality_audit.json").write_text(
        json.dumps({
            "exchange_assessments": [
                {"exchange_id": "ex1", "evaluation_readiness": "hard_review_required", "hard_reasons": ["low_confidence"]}
            ]
        }),
        encoding="utf-8",
    )

    diag = build_candidate_diagnostic(
        run_root=run,
        quality_json_path=q / "segmentation_quality_audit.json",
    )

    assert diag["diagnosed_exchange_count"] == 1
    row = diag["diagnostics"][0]
    assert row["candidate_trace_available"] is True
    assert "resolved_assignment_not_top_local_candidate" in row["diagnostic_modes"]
