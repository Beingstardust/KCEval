"""Response contract for the full-dialogue macro judge (macro_rubric_v1).

Mirrors response_contract_v2's validation style (strict required fields, no coercion of
judgments, verbatim-quote checking) at the macro grain instead of the per-segment grain. Kept as
a separate module rather than folded into response_contract_v2 because the macro judge answers
about a whole dialogue, not one segment -- different id space, different evidence source
(per-exchange quotes, not one segment's text), different dimension set.
"""

from __future__ import annotations

import re
from typing import Any

from .macro_rubric_v1 import MACRO_DIMENSION_IDS

ALLOWED_SCORES = {0, 0.5, 1}
ALLOWED_APPLICABILITY = {"applicable", "not_applicable", "unclear"}

REQUIRED_TOP_LEVEL_FIELDS = [
    "dialogue_id", "dimension_scores", "dimension_applicability",
    "evidence_pointers", "rationale_short",
]


def _normalize_for_quote_match(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    return text.strip()


def validate_macro_response(
    response: dict[str, Any],
    *,
    expected_dialogue_id: str | None = None,
    exchange_text_by_id: dict[str, str] | None = None,
) -> list[str]:
    """Validate one macro judge response. Returns a list of error strings (empty = valid).

    ``exchange_text_by_id`` maps exchange_id -> that exchange's text (student+tutor turns
    concatenated), used the same way response_contract_v2 checks segment evidence_quotes: every
    evidence_pointers[i]["quote"] must be an exact, normalized substring of the exchange it
    claims to be from.
    """
    errors: list[str] = []

    for key in REQUIRED_TOP_LEVEL_FIELDS:
        if key not in response:
            errors.append(f"missing_required_field:{key}")
    if errors:
        return errors

    extra = sorted(set(response.keys()) - set(REQUIRED_TOP_LEVEL_FIELDS))
    if extra:
        errors.append(f"unexpected_top_level_fields:{extra}")

    dialogue_id = response.get("dialogue_id")
    if not isinstance(dialogue_id, str):
        errors.append("dialogue_id:expected_str")
    elif expected_dialogue_id and dialogue_id != expected_dialogue_id:
        errors.append(f"dialogue_id_mismatch:expected={expected_dialogue_id}:actual={dialogue_id}")

    applicability_raw = response.get("dimension_applicability")
    applicability: dict[str, str] = {}
    if not isinstance(applicability_raw, dict):
        errors.append("dimension_applicability:expected_dict")
    else:
        extra_dims = sorted(set(applicability_raw.keys()) - set(MACRO_DIMENSION_IDS))
        if extra_dims:
            errors.append(f"dimension_applicability.unexpected_dimensions:{extra_dims}")
        for dim_id in MACRO_DIMENSION_IDS:
            value = applicability_raw.get(dim_id)
            if value not in ALLOWED_APPLICABILITY:
                errors.append(f"dimension_applicability.{dim_id}:invalid:{value}")
            else:
                applicability[dim_id] = value

    scores = response.get("dimension_scores")
    if not isinstance(scores, dict):
        errors.append("dimension_scores:expected_dict")
    else:
        extra_dims = sorted(set(scores.keys()) - set(MACRO_DIMENSION_IDS))
        if extra_dims:
            errors.append(f"dimension_scores.unexpected_dimensions:{extra_dims}")
        for dim_id in MACRO_DIMENSION_IDS:
            if dim_id not in scores:
                errors.append(f"dimension_scores.missing:{dim_id}")
                continue
            value = scores[dim_id]
            app = applicability.get(dim_id)
            if app == "not_applicable":
                if value is not None:
                    errors.append(f"dimension_scores.{dim_id}:must_be_null_when_not_applicable")
            else:
                if value is None:
                    errors.append(f"dimension_scores.{dim_id}:null_only_allowed_when_not_applicable")
                elif not isinstance(value, (int, float)) or value not in ALLOWED_SCORES:
                    errors.append(f"dimension_scores.{dim_id}:score_not_allowed_0_0.5_1:{value}")

    pointers = response.get("evidence_pointers")
    if not isinstance(pointers, list):
        errors.append("evidence_pointers:expected_list")
    else:
        for i, p in enumerate(pointers):
            if not isinstance(p, dict):
                errors.append(f"evidence_pointers[{i}]:not_object")
                continue
            dim = p.get("dimension")
            if dim not in MACRO_DIMENSION_IDS:
                errors.append(f"evidence_pointers[{i}].dimension:unknown:{dim}")
            exid = p.get("exchange_id")
            quote = p.get("quote")
            if not isinstance(exid, str) or not exid:
                errors.append(f"evidence_pointers[{i}].exchange_id:missing_or_invalid")
            if not isinstance(quote, str) or len(quote.strip()) < 4:
                errors.append(f"evidence_pointers[{i}].quote:too_short_or_missing")
            elif exchange_text_by_id is not None:
                source_text = exchange_text_by_id.get(exid, "")
                if _normalize_for_quote_match(quote) not in _normalize_for_quote_match(source_text):
                    errors.append(f"evidence_pointers[{i}].quote:not_found_in_exchange:{exid}")

        # every dimension marked "applicable" needs at least one evidence pointer -- an
        # applicable macro judgment with zero supporting evidence is exactly the kind of
        # ungrounded rationale the proposal's grounding concerns (sec 4.4/6.4) warn against.
        dims_with_evidence = {p.get("dimension") for p in pointers if isinstance(p, dict)}
        for dim_id in MACRO_DIMENSION_IDS:
            if applicability.get(dim_id) == "applicable" and dim_id not in dims_with_evidence:
                errors.append(f"evidence_pointers:missing_for_applicable_dimension:{dim_id}")

    rationale = response.get("rationale_short")
    if not isinstance(rationale, str):
        errors.append("rationale_short:expected_str")
    else:
        if len(rationale.strip()) < 20:
            errors.append("rationale_short:too_short")
        if len(rationale) > 1400:
            errors.append("rationale_short:too_long")
        if any(m in rationale.lower() for m in
               ["step-by-step reasoning", "chain of thought", "hidden reasoning"]):
            errors.append("rationale_short:mentions_hidden_reasoning")

    return errors
