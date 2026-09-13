from pathlib import Path

from seg_eval.conversation_region_extractor_v1 import extract_conversation_region


def test_auto_student_tutor_turn_scan_excludes_header_footer() -> None:
    result = extract_conversation_region(
        "tests/fixtures/mixed_input_auto_student_tutor_001.txt",
        conversation_id="mixed_auto_001",
    )
    assert result.ready_for_normalization is True
    assert result.extraction_method == "auto_student_tutor_turn_scan"
    assert result.extracted_turn_count == 4
    assert result.student_turn_count == 2
    assert result.tutor_turn_count == 2
    assert "Exercise 1 description" not in result.conversation_text
    assert "after-analysis" not in result.conversation_text
    assert "Student: I am confused about classification" in result.conversation_text
    assert "Tutor: Not exactly" in result.conversation_text


def test_auto_student_tutor_turn_scan_blocks_noise_only() -> None:
    tmp = Path("tests/fixtures/tmp_noise_only_no_student_tutor.txt")
    tmp.write_text("Notes only\nNo dialogue labels here\n", encoding="utf-8")
    try:
        result = extract_conversation_region(tmp, conversation_id="tmp")
        assert result.ready_for_normalization is False
        assert result.extracted_turn_count == 0
    finally:
        tmp.unlink(missing_ok=True)
