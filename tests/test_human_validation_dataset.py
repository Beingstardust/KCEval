"""Tests against the built canonical human-validation dataset.

These are integration checks against the real, already-built dataset file (built by
scripts/build_human_validation_dataset.py), not unit tests of pure functions -- skipped
gracefully if the dataset hasn't been built yet in a given environment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DATASET_PATH = Path(__file__).resolve().parents[1] / "data/processed/human_validation/dataset_v1.jsonl"
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "data/processed/human_validation/manifest_v1.json"

pytestmark = pytest.mark.skipif(not DATASET_PATH.exists(), reason="human validation dataset not built")


def load_rows():
    return [json.loads(l) for l in DATASET_PATH.open(encoding="utf-8")]


def test_exactly_115_rows():
    assert len(load_rows()) == 115


def test_review_item_ids_unique():
    rows = load_rows()
    ids = [r["review_item_id"] for r in rows]
    assert len(ids) == len(set(ids))


def test_both_dialogues_represented():
    rows = load_rows()
    dialogues = {r["dialogue_id"] for r in rows}
    assert len(dialogues) == 2


def test_both_families_represented():
    rows = load_rows()
    families = {r["family"] for r in rows}
    assert families == {"local", "arc"}


def test_dm1_count_and_dm2_count():
    rows = load_rows()
    counts = {}
    for r in rows:
        d = "dm1" if r["dialogue_id"].startswith("dm1") else "dm2"
        counts[d] = counts.get(d, 0) + 1
    assert counts["dm1"] == 52
    assert counts["dm2"] == 63


def test_every_row_has_focus_target_fields():
    for r in load_rows():
        ft = r["focus_target"]
        assert "target_type" in ft
        assert "pattern" in ft
        assert "focus_span" in ft


def test_manifest_has_sha256_for_every_source():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for key, entry in manifest["source_artifacts"].items():
        assert len(entry["sha256"]) == 64, f"{key} missing a real sha256"


def test_manifest_unit_count_matches_dataset():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["evaluation_unit_count"] == len(load_rows())


def test_invalid_responses_still_present_not_dropped():
    # dm2 has known contract-invalid responses -- they must still appear as rows, not be
    # silently excluded from the dataset.
    rows = load_rows()
    n_invalid = sum(1 for r in rows if not r["judge_response_contract_valid"])
    assert n_invalid > 0


def test_valid_rows_carry_judge_output_invalid_rows_do_not():
    for r in load_rows():
        if r["judge_response_contract_valid"]:
            assert r["judge_output"] is not None
        else:
            assert r["judge_output"] is None
