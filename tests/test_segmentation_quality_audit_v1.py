from seg_eval.segmentation_quality_audit_v1 import classify_exchange, build_quality_audit


def test_low_confidence_is_hard_review_required():
    row = {
        "exchange_id": "ex1",
        "resolved_kc_id": "KC_X",
        "confidence_band": "low",
        "assignment_scope": "single_kc",
        "resolution_action": "pass1_closed_set",
        "student_text": "F1?",
        "tutor_text": "F1 is 2PR/(P+R).",
    }
    assessed = classify_exchange(row)
    assert assessed["evaluation_readiness"] == "hard_review_required"
    assert "low_confidence" in assessed["hard_reasons"]


def test_broad_kc_set_is_hard_review_required():
    row = {
        "exchange_id": "ex2",
        "resolved_kc_id": "KC_X",
        "confidence_band": "medium",
        "assignment_scope": "broad_kc_set",
        "resolution_action": "pass2_broad_multibranch_closed_set",
        "student_text": "Can you connect all exercises?",
        "tutor_text": "Here is a recap.",
    }
    assessed = classify_exchange(row)
    assert assessed["evaluation_readiness"] == "hard_review_required"
    assert "broad_kc_set" in assessed["hard_reasons"]


def test_context_rescue_with_formula_gets_suspicious_reason():
    row = {
        "exchange_id": "ex3",
        "resolved_kc_id": "KC_X",
        "confidence_band": "medium",
        "assignment_scope": "branch_context",
        "resolution_action": "pass2_context_dependent_anaphora_bridge_rescue",
        "student_text": "Calculate the interval for p=0.85 and N=200.",
        "tutor_text": "Use the formula.",
    }
    assessed = classify_exchange(row)
    assert assessed["evaluation_readiness"] == "review_recommended"
    assert "formula_or_calculation_turn_with_context_rescue" in assessed["suspect_reasons"]


def test_meta_orientation_gets_hard_review_required():
    row = {
        "exchange_id": "ex4",
        "resolved_kc_id": "KC_X",
        "confidence_band": "medium",
        "assignment_scope": "single_kc",
        "resolution_action": "pass1_closed_set",
        "student_text": "I want this to feel like a conversation with a real human student and an LLM tutor.",
        "tutor_text": "I will respond like an LLM tutor.",
    }
    assessed = classify_exchange(row)
    assert assessed["evaluation_readiness"] == "hard_review_required"
    assert "meta_or_orientation_turn_forced_to_kc" in assessed["suspect_reasons"]


def test_audit_builds_on_fixture_run(tmp_path):
    run = tmp_path / "v9e"
    run.mkdir()
    (run / "summary.json").write_text(
        '{"exchange_count":1,"segment_count":1,"pipeline":"test","confidence_band_counts":{"low":1},"assignment_scope_counts":{"broad_kc_set":1},"resolution_action_counts":{"pass2_broad_multibranch_closed_set":1}}',
        encoding="utf-8",
    )
    (run / "contactless_exchange_assignments.jsonl").write_text(
        '{"exchange_id":"ex1","resolved_kc_id":"KC_X","resolved_kc_name":"X","confidence_band":"low","assignment_scope":"broad_kc_set","resolution_action":"pass2_broad_multibranch_closed_set","student_text":"Final strategy?","tutor_text":"Use four passes."}\n',
        encoding="utf-8",
    )
    (run / "segments.jsonl").write_text(
        '{"segment_id":"seg1","member_exchange_ids":["ex1"],"resolved_kc_id":"KC_X","resolved_kc_name":"X"}\n',
        encoding="utf-8",
    )
    audit = build_quality_audit(run)
    assert audit["exchange_status_counts"]["hard_review_required"] == 1
    assert audit["segment_status_counts"]["hard_review_required"] == 1
