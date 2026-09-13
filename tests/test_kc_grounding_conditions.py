"""Tests for the KC-grounding ablation conditions (execution spec §29).

The integration tests at the bottom run against the real built prompt sets when present and
are skipped otherwise, matching the pattern used elsewhere in this suite.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from seg_eval.evaluation_judge.kc_grounding_conditions import (  # noqa: E402
    GROUNDING_CONDITIONS, DEFAULT_CONDITION, EVALUATION_SUPPORT_FIELDS,
    condition_shows_kc_context, condition_adds_curriculum_grounding, grounding_block_for_kc,
    load_reviewed_library_grounding, REVIEWED_LIBRARY_PATH,
)
from seg_eval.evaluation_judge.prompt_builder import (  # noqa: E402
    compact_packet_context, profile_summary,
)

PROMPT_ROOT = REPO_ROOT / "data/processed/judge_prompts"
TAG = "kcgrd_20260829"
FAMILIES = {
    "dm1_local": "v35_dm1_kc_segment_local_20260820",
    "dm1_arc": "v35_dm1_topic_rollup_arc_20260820",
    "dm2_local": "v35_dm2_kc_segment_local_20260827",
    "dm2_arc": "v35_dm2_topic_rollup_arc_20260827",
}

built = all((PROMPT_ROOT / f"{TAG}_A_ungrounded_{f}/segment_judge_prompts.jsonl").exists() for f in FAMILIES)
requires_built = pytest.mark.skipif(not built, reason="ablation prompt sets not built")


# ---------------------------------------------------------------------------
# Unit: condition semantics
# ---------------------------------------------------------------------------

def test_default_condition_is_baseline():
    assert DEFAULT_CONDITION == "B_baseline"


def test_only_a_hides_kc_context():
    assert not condition_shows_kc_context("A_ungrounded")
    assert condition_shows_kc_context("B_baseline")
    assert condition_shows_kc_context("C_curriculum_grounded")


def test_only_c_adds_curriculum_grounding():
    assert not condition_adds_curriculum_grounding("A_ungrounded")
    assert not condition_adds_curriculum_grounding("B_baseline")
    assert condition_adds_curriculum_grounding("C_curriculum_grounded")


def test_unknown_condition_raises():
    with pytest.raises(ValueError, match="unknown grounding condition"):
        condition_shows_kc_context("D_made_up")


def test_all_conditions_are_distinct_and_known():
    assert len(set(GROUNDING_CONDITIONS)) == 3


# ---------------------------------------------------------------------------
# Unit: grounding block construction
# ---------------------------------------------------------------------------

def test_grounding_block_none_for_unknown_kc():
    assert grounding_block_for_kc("KC_DOES_NOT_EXIST", {}) is None


def test_grounding_block_none_when_entry_has_no_content():
    index = {"KC_X": {"scope_status": "grounded"}}
    assert grounding_block_for_kc("KC_X", index) is None


def test_grounding_block_carries_all_populated_fields():
    index = {"KC_X": {
        "scope_status": "grounded",
        "what_tutor_should_explain": ["a"],
        "red_flags": ["b"],
    }}
    block = grounding_block_for_kc("KC_X", index)
    assert block["what_tutor_should_explain"] == ["a"]
    assert block["red_flags"] == ["b"]
    assert block["scope_status"] == "grounded"
    assert "grounding_caution" not in block


def test_abstained_kc_gets_explicit_caution():
    index = {"KC_X": {"scope_status": "abstained", "red_flags": ["do not evaluate from this"]}}
    block = grounding_block_for_kc("KC_X", index)
    assert "grounding_caution" in block
    assert "abstained" in block["grounding_caution"]


# ---------------------------------------------------------------------------
# Unit: profile_summary / compact_packet_context gating
# ---------------------------------------------------------------------------

SAMPLE_PACKET = {
    "segment_id": "d::seg_0000",
    "evaluation_target": {"mode": "single_kc_primary", "primary_kc_id": "KC_X", "evaluator_kc_set": ["KC_X"]},
    "segmentation_metadata": {"segment_scope": "single_kc"},
    "kc_context": [{"unit_id": "KC_X", "canonical_name": "Thing", "definition_or_summary": "def"}],
    "member_exchanges": [{
        "exchange_id": "ex_1", "student_text": "s", "tutor_text": "t",
        "assignment": {"resolved_kc_id": "KC_X", "resolved_kc_name": "Thing"},
    }],
}


def test_profile_summary_unchanged_when_no_grounding_index():
    out = profile_summary("KC_X", {}, {"KC_X": {"canonical_name": "Thing"}})
    assert "curriculum_grounding" not in out


def test_profile_summary_adds_grounding_when_index_given():
    index = {"KC_X": {"scope_status": "grounded", "red_flags": ["r"]}}
    out = profile_summary("KC_X", {}, {"KC_X": {"canonical_name": "Thing"}}, index)
    assert out["curriculum_grounding"]["red_flags"] == ["r"]


def test_condition_a_empties_kc_context():
    ctx = compact_packet_context(SAMPLE_PACKET, {}, "A_ungrounded")
    assert ctx["primary_kc"] is None
    assert ctx["primary_kc_id"] is None
    assert ctx["evaluator_kc_set"] == []


def test_condition_a_strips_resolved_kc_from_segmentation_context():
    """Regression: resolved_kc_id/resolved_kc_name leaked a KC identifier into every
    'ungrounded' prompt before this was gated."""
    ctx = compact_packet_context(SAMPLE_PACKET, {}, "A_ungrounded")
    exchange_ctx = ctx["segmentation_context"]["member_exchange_contexts"][0]
    assert exchange_ctx["resolved_kc_id"] is None
    assert exchange_ctx["resolved_kc_name"] is None
    assert "KC_X" not in json.dumps(ctx)


def test_condition_b_keeps_resolved_kc_in_segmentation_context():
    ctx = compact_packet_context(SAMPLE_PACKET, {}, "B_baseline")
    exchange_ctx = ctx["segmentation_context"]["member_exchange_contexts"][0]
    assert exchange_ctx["resolved_kc_id"] == "KC_X"


def test_condition_b_default_matches_explicit_b():
    assert compact_packet_context(SAMPLE_PACKET, {}) == compact_packet_context(SAMPLE_PACKET, {}, "B_baseline")


def test_condition_a_uses_ungrounded_mode_policy():
    a = compact_packet_context(SAMPLE_PACKET, {}, "A_ungrounded")
    b = compact_packet_context(SAMPLE_PACKET, {}, "B_baseline")
    assert a["mode_policy"] != b["mode_policy"]
    assert "No curriculum knowledge-component reference" in a["mode_policy"]


def test_condition_c_equals_b_plus_grounding_only():
    index = {"KC_X": {"scope_status": "grounded", "red_flags": ["r"]}}
    b = compact_packet_context(SAMPLE_PACKET, {}, "B_baseline")
    c = compact_packet_context(SAMPLE_PACKET, {}, "C_curriculum_grounded", index)
    assert c["primary_kc"]["curriculum_grounding"]["red_flags"] == ["r"]
    stripped = copy.deepcopy(c)
    stripped["primary_kc"].pop("curriculum_grounding")
    for entry in stripped["evaluator_kc_set"]:
        entry.pop("curriculum_grounding", None)
    assert stripped == b


def test_grounding_index_ignored_for_condition_b():
    index = {"KC_X": {"scope_status": "grounded", "red_flags": ["r"]}}
    with_index = compact_packet_context(SAMPLE_PACKET, {}, "B_baseline", index)
    without = compact_packet_context(SAMPLE_PACKET, {}, "B_baseline")
    assert with_index == without


# ---------------------------------------------------------------------------
# Real library
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (REPO_ROOT / REVIEWED_LIBRARY_PATH).exists(), reason="reviewed library not present")
def test_real_reviewed_library_loads_159_kcs():
    index = load_reviewed_library_grounding(REPO_ROOT / REVIEWED_LIBRARY_PATH)
    assert len(index) == 159


@pytest.mark.skipif(not (REPO_ROOT / REVIEWED_LIBRARY_PATH).exists(), reason="reviewed library not present")
def test_real_library_never_exposes_kc_specific_criteria():
    """kc_specific_criteria is empty on all 159 KCs and reviewer_criteria is 0% expert-approved;
    neither may be surfaced as curriculum truth by the grounding layer."""
    index = load_reviewed_library_grounding(REPO_ROOT / REVIEWED_LIBRARY_PATH)
    for entry in index.values():
        assert "kc_specific_criteria" not in entry
        assert "reviewer_criteria" not in entry
        for key in entry:
            assert key in EVALUATION_SUPPORT_FIELDS or key == "scope_status"


# ---------------------------------------------------------------------------
# Integration against the real built prompt sets
# ---------------------------------------------------------------------------

def _rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8")]


@requires_built
def test_each_condition_has_115_units_total():
    for condition in ("A_ungrounded", "C_curriculum_grounded"):
        total = sum(
            len(_rows(PROMPT_ROOT / f"{TAG}_{condition}_{fam}/segment_judge_prompts.jsonl"))
            for fam in FAMILIES
        )
        assert total == 115, f"{condition} has {total} units"


@requires_built
def test_ungrounded_condition_leaks_no_kc_identifier_anywhere():
    for fam in FAMILIES:
        for row in _rows(PROMPT_ROOT / f"{TAG}_A_ungrounded_{fam}/segment_judge_prompts.jsonl"):
            assert "KC_" not in row["user_prompt"], f"{fam}:{row['segment_id']}"


@requires_built
def test_grounded_condition_differs_from_baseline_only_by_grounding():
    for fam, baseline_run in FAMILIES.items():
        b_rows = _rows(PROMPT_ROOT / baseline_run / "segment_judge_prompts.jsonl")
        c_rows = _rows(PROMPT_ROOT / f"{TAG}_C_curriculum_grounded_{fam}/segment_judge_prompts.jsonl")
        assert len(b_rows) == len(c_rows)
        for rb, rc in zip(b_rows, c_rows):
            assert rb["segment_id"] == rc["segment_id"]
            cb = json.loads(rb["user_prompt"].split("Segment packet:\n", 1)[1])
            cc = json.loads(rc["user_prompt"].split("Segment packet:\n", 1)[1])
            if cc.get("primary_kc"):
                cc["primary_kc"].pop("curriculum_grounding", None)
            for entry in cc.get("evaluator_kc_set") or []:
                entry.pop("curriculum_grounding", None)
            assert cc == cb, f"{fam}:{rb['segment_id']} differs beyond curriculum_grounding"


@requires_built
def test_segment_ids_align_across_all_three_conditions():
    """The analysis joins by review_item_id; misaligned ids would silently compare different
    segments across conditions."""
    for fam, baseline_run in FAMILIES.items():
        ids = []
        for path in (
            PROMPT_ROOT / f"{TAG}_A_ungrounded_{fam}/segment_judge_prompts.jsonl",
            PROMPT_ROOT / baseline_run / "segment_judge_prompts.jsonl",
            PROMPT_ROOT / f"{TAG}_C_curriculum_grounded_{fam}/segment_judge_prompts.jsonl",
        ):
            ids.append([r["segment_id"] for r in _rows(path)])
        assert ids[0] == ids[1] == ids[2], f"{fam}: segment ids differ across conditions"
