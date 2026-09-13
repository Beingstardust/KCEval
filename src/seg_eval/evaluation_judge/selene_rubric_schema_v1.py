"""Grammar-enforced rubric response for Selene, with the two observed contract violations made
structurally impossible rather than merely discouraged.

MEASURED PROBLEM
----------------
Selene answered all 115 rubric prompts with parseable JSON but only 62 satisfied
``response_contract_v2``. Every failure was one of two mechanical issues, neither of which is a
judgment-quality problem:

  35x  raw_dimension_scores / dimension_scores gave a NUMBER for a dimension it had marked
       ``not_applicable``. The contract requires null there. A flat schema cannot express that
       linkage -- applicability and score are separate objects, so nothing stops them
       disagreeing.

  68x  kc_grounding.context_kcs_used cited KC identifiers that do not appear in the packet.
       Nothing in a flat ``array of string`` constrains which strings are legal.

A 46% instrument failure rate would contaminate every downstream number: the surviving 62 units
are not a random sample, so any agreement statistic computed on them is measuring "units where
Selene happened to comply" rather than "how well Selene judges". Fixing the instrument is a
precondition for the comparison meaning anything.

THE FIX
-------
Two structural changes, both enforced by the grammar at generation time:

1. **Paired dimension objects.** Each dimension becomes one object judged by a ``oneOf``:
   either ``{applicability: "not_applicable", raw: null, adjusted: null}`` or
   ``{applicability: "applicable"|"unclear", raw: 0|0.5|1, adjusted: 0|0.5|1}``. The
   contradictory combination is not a representable state, so the model cannot emit it.

2. **Per-packet KC enum.** ``context_kcs_used`` items are constrained to an ``enum`` of exactly
   the KC ids present in that packet. Citing an absent identifier becomes ungeneratable.

The wire shape is then mapped back to the canonical ``response_contract_v2`` shape by
:func:`to_contract_shape`, so nothing downstream changes and the contract validator still has
the final say. The wire format is chosen for what a grammar can guarantee; the canonical format
remains the project's contract.

WHY THIS IS NOT "SILENTLY REPAIRING" OUTPUT
--------------------------------------------
This project's standing rule is never to repair a judgment after the fact. Nothing here repairs
anything: the mapping is a lossless rename of fields the model chose under a grammar that only
permits self-consistent answers. No value is altered, defaulted, or inferred. If the model wants
to call a dimension not_applicable it still can -- it simply cannot also score it.
"""

from __future__ import annotations

from typing import Any

from .rubric_v2 import PARTIAL_DIMENSIONS, TRUST_FLAGS
from .response_contract_v2 import REQUIRED_FLAGS, ALLOWED_EVALUATION_MODES

WIRE_SCHEMA_VERSION = "selene_rubric_wire_v1_paired_dimensions"


def _obj(props: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": props,
        "required": required if required is not None else list(props),
        "additionalProperties": False,
    }


def _dimension_object() -> dict:
    """One dimension: applicability and both scores in a single object, with the two legal
    combinations expressed as a oneOf so no contradictory state exists."""
    return {
        "oneOf": [
            _obj({
                "applicability": {"const": "not_applicable"},
                "raw": {"type": "null"},
                "adjusted": {"type": "null"},
            }),
            _obj({
                "applicability": {"type": "string", "enum": ["applicable", "unclear"]},
                "raw": {"type": "number", "enum": [0, 0.5, 1]},
                "adjusted": {"type": "number", "enum": [0, 0.5, 1]},
            }),
        ]
    }


def build_wire_schema(allowed_kc_ids: list[str]) -> dict[str, Any]:
    """Per-packet schema. ``allowed_kc_ids`` constrains context_kcs_used to identifiers that
    actually exist in this packet."""
    kc_items: dict[str, Any] = {"type": "string"}
    if allowed_kc_ids:
        kc_items["enum"] = sorted(set(allowed_kc_ids))

    return _obj({
        "segment_id": {"type": "string"},
        "evaluation_mode": {"type": "string", "enum": sorted(ALLOWED_EVALUATION_MODES)},
        "dimensions": _obj({d: _dimension_object() for d in PARTIAL_DIMENSIONS}),
        "trust_flags": _obj({f: {"type": "integer", "enum": [0, 1]} for f in TRUST_FLAGS}),
        "dependency_adjustments": {
            "type": "array",
            "items": _obj({
                "rule_id": {"type": "string"},
                "applied": {"type": "boolean"},
                "reason": {"type": "string"},
                "affected_dimensions": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(PARTIAL_DIMENSIONS)},
                },
            }),
        },
        "micro_score_raw": {"type": "number"},
        "micro_score_dependency_adjusted": {"type": "number"},
        "trust_adjusted_score": {"type": "number"},
        "kc_grounding": _obj({
            "primary_kc_used": {"type": "boolean"},
            "context_kcs_used": {"type": "array", "items": kc_items},
            "unsupported_or_wrong_kc_claims": {"type": "array", "items": {"type": "string"}},
        }),
        "flags": _obj({f: {"type": "boolean"} for f in REQUIRED_FLAGS}),
        "evidence_quotes": {"type": "array", "items": {"type": "string"}},
        "rationale_short": {"type": "string"},
    })


def to_contract_shape(wire: dict[str, Any]) -> dict[str, Any]:
    """Lossless rename from the wire shape to response_contract_v2's shape.

    Splits each paired dimension object back into the three parallel maps the contract expects.
    No value is altered, defaulted or inferred -- if a field is missing from the wire payload it
    stays missing, so the contract validator still reports it rather than this function hiding it.
    """
    dims = wire.get("dimensions") or {}
    raw, adjusted, applicability = {}, {}, {}
    for name, entry in dims.items():
        if not isinstance(entry, dict):
            continue
        applicability[name] = entry.get("applicability")
        raw[name] = entry.get("raw")
        adjusted[name] = entry.get("adjusted")

    out = {
        "segment_id": wire.get("segment_id"),
        "evaluation_mode": wire.get("evaluation_mode"),
        "raw_dimension_scores": raw,
        "dimension_applicability": applicability,
        "dependency_adjustments": wire.get("dependency_adjustments"),
        "dimension_scores": adjusted,
        "trust_flags": wire.get("trust_flags"),
        "micro_score_raw": wire.get("micro_score_raw"),
        "micro_score_dependency_adjusted": wire.get("micro_score_dependency_adjusted"),
        "trust_adjusted_score": wire.get("trust_adjusted_score"),
        "kc_grounding": wire.get("kc_grounding"),
        "flags": wire.get("flags"),
        "evidence_quotes": wire.get("evidence_quotes"),
        "rationale_short": wire.get("rationale_short"),
    }
    return {k: v for k, v in out.items() if v is not None}


WIRE_FORMAT_INSTRUCTION = """

OUTPUT FORMAT NOTE:
Report each rubric dimension as ONE object under "dimensions", with three fields:
  applicability : "applicable" | "unclear" | "not_applicable"
  raw           : the score before dependency adjustments
  adjusted      : the score after dependency adjustments

When applicability is "not_applicable", both raw and adjusted MUST be null.
When applicability is "applicable" or "unclear", both MUST be 0, 0.5 or 1.

Under kc_grounding, "context_kcs_used" may only contain KC identifiers that appear in this
packet. Do not name a KC that was not supplied to you."""


def allowed_kc_ids_for_packet(packet: dict[str, Any]) -> list[str]:
    """Every KC identifier legitimately citable for this packet."""
    ids: list[str] = []
    target = packet.get("evaluation_target") or {}
    for key in ("primary_kc_id", "resolved_kc_id", "dominant_kc_id"):
        v = target.get(key)
        if isinstance(v, str) and v:
            ids.append(v)
    for v in target.get("evaluator_kc_set") or []:
        if v:
            ids.append(str(v))
    for row in packet.get("kc_context") or []:
        if isinstance(row, dict):
            v = row.get("unit_id") or row.get("kc_id") or row.get("id")
            if v:
                ids.append(str(v))
    return sorted(set(ids))
