"""Tests for the Selene client.

The first test is the important one: it guards against the exact silent failure that cost the
KC-library pipeline all 61 calls of its first real run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from seg_eval.evaluation_judge.selene_client_v1 import (  # noqa: E402
    build_payload, SeleneResult, SERVED_MODEL_NAME,
)

SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string"}},
    "required": ["verdict"],
    "additionalProperties": False,
}


def _payload(**kw):
    base = dict(model=SERVED_MODEL_NAME, system_message="sys", user_prompt="user",
                schema=SCHEMA, temperature=0.0, top_p=1.0, seed=None, max_tokens=100)
    base.update(kw)
    return build_payload(**base)


def test_response_format_is_top_level_not_under_extra_body():
    """THE regression test. Nesting response_format under extra_body fails SILENTLY: the request
    succeeds, the model returns plausible free-form JSON, and the schema is never enforced. The
    KC-library pipeline lost 61/61 calls to this."""
    p = _payload()
    assert "response_format" in p, "response_format must be a top-level key"
    assert "extra_body" not in p, "extra_body is an SDK convention and is silently ignored here"
    assert "guided_json" not in p


def test_schema_is_marked_strict():
    """Without strict:True the server treats the schema as advisory and generation is not
    constrained, which is the whole reason for adopting grammar enforcement."""
    p = _payload()
    rf = p["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"] == SCHEMA


def test_seed_omitted_at_temperature_zero():
    """At temperature 0 generation is greedy and a seed changes nothing. Sending one would imply
    a reproducibility guarantee the parameter is not providing."""
    assert "seed" not in _payload(temperature=0.0, seed=20260812)


def test_seed_included_when_sampling():
    p = _payload(temperature=0.7, seed=20260812)
    assert p["seed"] == 20260812


def test_no_schema_means_no_response_format():
    assert "response_format" not in _payload(schema=None)


def test_messages_carry_system_then_user_in_order():
    p = _payload()
    assert [m["role"] for m in p["messages"]] == ["system", "user"]


def test_sampling_params_are_passed_through():
    p = _payload(temperature=0.3, top_p=0.9, max_tokens=1234)
    assert (p["temperature"], p["top_p"], p["max_tokens"]) == (0.3, 0.9, 1234)


# ---------------------------------------------------------------------------
# two-tier validity
# ---------------------------------------------------------------------------

def test_fully_valid_requires_both_schema_and_contract():
    assert SeleneResult("t", {"a": 1}, True, []).fully_valid
    # schema satisfied but a contract rule the grammar cannot express was broken
    assert not SeleneResult("t", {"a": 1}, True, ["quote_not_in_source"]).fully_valid
    # grammar itself failed
    assert not SeleneResult("t", None, False, []).fully_valid
    # transport failure
    assert not SeleneResult("", None, False, [], error="http_500:boom").fully_valid


def test_schema_and_contract_failures_stay_distinguishable():
    """A single boolean would conflate two problems with different fixes: a grammar failure means
    the request was malformed, a contract failure means the judgment was."""
    grammar = SeleneResult("t", None, False, [])
    contract = SeleneResult("t", {"a": 1}, True, ["verdict_contradicts_own_evidence"])
    assert grammar.schema_ok is False and grammar.contract_errors == []
    assert contract.schema_ok is True and contract.contract_errors
