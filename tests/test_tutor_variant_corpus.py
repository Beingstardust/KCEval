"""Tests for the tutor-variant corpus construction.

The experimental validity of the whole variant study rests on a few structural properties that
are easy to break silently, so they are asserted here rather than eyeballed:

  - the student side is identical across all three arms (controlled comparison)
  - T3 is a genuine minimal pair with T1 (differs ONLY at injected exchanges)
  - T2 introduces no factual errors (it is the false-positive control for the trust flags)
  - no dialogue id leaks the experimental condition into a judge prompt

That last one is a regression test for a real defect: the first build used descriptive dialogue
ids, and because dialogue_id is embedded in segment_id and every exchange id, the strings
"subtly_wrong" and "answer_dumping" appeared 13x and 5x per judge prompt -- disclosing the
condition to a judge that was supposed to be blind.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INBOX = REPO_ROOT / "data/input/dialogues/inbox"
GROUND_TRUTH = REPO_ROOT / "data/gold/tutor_variant_injected_errors.json"
SOURCE = INBOX / "dm1_human_student_llm_tutor_teaching_rich_dialogue.json"

built = all((INBOX / f).exists() for f in ("tv_a_source.json", "tv_b_source.json", "tv_c_source.json"))
requires_built = pytest.mark.skipif(not built, reason="tutor variant corpus not built")

# Tokens that would disclose the experimental condition if they appeared in an identifier.
CONDITION_CUES = ("subtly", "wrong", "dump", "strong", "degrad", "inject", "bad")


def load(name: str) -> list[dict]:
    return json.loads((INBOX / name).read_text(encoding="utf-8"))


@requires_built
def test_all_arms_have_same_exchange_count():
    a, b, c = load("tv_a_source.json"), load("tv_b_source.json"), load("tv_c_source.json")
    assert len(a) == len(b) == len(c) == 63


@requires_built
def test_student_turns_identical_across_arms():
    """The controlled-comparison premise: only tutor behaviour varies."""
    a, b, c = load("tv_a_source.json"), load("tv_b_source.json"), load("tv_c_source.json")
    sa = [x["student"]["text"] for x in a]
    assert sa == [x["student"]["text"] for x in b]
    assert sa == [x["student"]["text"] for x in c]


@requires_built
def test_t1_arm_matches_original_dm1_tutor_text():
    """tv_a must be dm1's real tutor, unmodified -- it is the reference arm."""
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    a = load("tv_a_source.json")
    assert [x["tutor"]["text"] for x in a] == [x["tutor"]["text"] for x in source]


@requires_built
def test_t3_is_a_minimal_pair_with_t1():
    """T3 differs from T1 at exactly the injected exchanges and nowhere else. Without this the
    T1-vs-T3 comparison cannot attribute score differences to the injected errors."""
    gt = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    injected_indices = {e["exchange_index"] for e in gt["injected_errors"]}
    a, c = load("tv_a_source.json"), load("tv_c_source.json")
    differing = {i for i in range(len(a)) if a[i]["tutor"]["text"] != c[i]["tutor"]["text"]}
    assert differing == injected_indices


@requires_built
def test_t2_rewrites_every_turn():
    a, b = load("tv_a_source.json"), load("tv_b_source.json")
    differing = [i for i in range(len(a)) if a[i]["tutor"]["text"] != b[i]["tutor"]["text"]]
    assert len(differing) == len(a)


@requires_built
def test_t3_length_matches_t1_so_it_is_not_separable_by_verbosity():
    """A length gap would be an alternative explanation for any T1-vs-T3 score difference."""
    a, c = load("tv_a_source.json"), load("tv_c_source.json")
    mean_a = sum(len(x["tutor"]["text"]) for x in a) / len(a)
    mean_c = sum(len(x["tutor"]["text"]) for x in c) / len(c)
    assert abs(mean_a - mean_c) < 15


@pytest.mark.skipif(not GROUND_TRUTH.exists(), reason="ground truth not built")
def test_ground_truth_records_every_injection_with_both_texts():
    gt = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    assert gt["t3_injected_error_count"] == len(gt["injected_errors"]) == 15
    for e in gt["injected_errors"]:
        assert e["exchange_id"].startswith("ex_")
        assert e["original_tutor_text"] and e["variant_tutor_text"]
        assert e["original_tutor_text"] != e["variant_tutor_text"]
        assert e["injected_error"]


@pytest.mark.skipif(not GROUND_TRUTH.exists(), reason="ground truth not built")
def test_ground_truth_uses_opaque_dialogue_ids():
    gt = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    for variant, spec in gt["variants"].items():
        opaque = spec["opaque_dialogue_id"] if isinstance(spec, dict) else str(spec)
        low = opaque.lower()
        assert not any(cue in low for cue in CONDITION_CUES), f"{variant}: id {opaque} leaks condition"


# ---------------------------------------------------------------------------
# Regression: no condition cue may reach a judge prompt through an identifier
# ---------------------------------------------------------------------------

PROMPT_DIRS = sorted((REPO_ROOT / "data/processed/judge_prompts").glob("v35_tv_*_20260829"))


@pytest.mark.skipif(not PROMPT_DIRS, reason="variant judge prompts not built")
def test_no_condition_cue_in_any_prompt_identifier():
    checked = 0
    for d in PROMPT_DIRS:
        for path in d.glob("*.jsonl"):
            for line in path.open(encoding="utf-8"):
                row = json.loads(line)
                checked += 1
                ident = (str(row.get("segment_id", "")) + " "
                          + str(row.get("judge_prompt_id", "")) + " "
                          + str(row.get("dialogue_id", ""))).lower()
                hits = [c for c in CONDITION_CUES if c in ident]
                assert not hits, f"{path.name}: {row.get('segment_id')} leaks {hits}"
    assert checked > 0
