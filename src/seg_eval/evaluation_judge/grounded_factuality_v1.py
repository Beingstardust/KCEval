"""Grounded factuality pass: does the tutor contradict the curriculum reference?

Design rationale and citations: see
``local_audits/evaluation_completion_20260824T093426Z/27_FACTUALITY_REDESIGN_RESEARCH_AND_CITATIONS.md``.

WHY THIS EXISTS
---------------
``rubric_v2`` asks the judge for ``curriculum_hallucination_present: 0|1`` while supplying **no
curriculum** in the prompt. A judge asked to detect contradiction against a reference it was
never given falls back on plausibility, and fluent-but-wrong content scores well. Measured here:
3 of 15 planted material errors caught, and a tutor carrying all 15 scored a perfect 1/1/1/1 at
the dialogue level.

The reference the framework needs already exists -- the reviewed library's ``evaluation_support``
block (``what_tutor_should_explain``, ``common_confusions_or_errors``, ``red_flags``) -- and
reaches the judge nowhere.

THREE DESIGN DECISIONS, EACH FORCED BY EVIDENCE
-----------------------------------------------
1. **Holistic, not atomic decomposition.** The intuitive fix -- split the turn into atomic claims
   and verify each -- is contraindicated: arXiv 2603.28005 finds decomposition often *lowers*
   correlation with human judgment through error accumulation and context loss. DeepMind's FACTS
   Grounding likewise judges groundedness holistically against a supplied context document. So
   this asks one grounded question about the whole segment, and requires verbatim quotes for any
   contradiction so the verdict stays auditable.

2. **A separate, narrow pass -- not the main rubric prompt.** Putting this same reference content
   into the main prompt was already measured harmful here (contract validity 91.3% -> 82.6%,
   `21_KC_GROUNDING_ABLATION.md`), and prompt-size sensitivity has been measured twice
   independently in this repo. This pass therefore carries no rubric dimensions, no dependency
   rules and no aggregate arithmetic -- only the segment, the reference, and one question.

3. **Absence is not error.** The single largest false-positive risk: a tutor legitimately
   covering something the reference does not mention. The prompt states explicitly that content
   absent from the reference is NOT a contradiction, and only direct conflict counts. Without
   this, the pass would measure "coverage of the reference" rather than "correctness".
"""

from __future__ import annotations

from ..corpus_config import kc_library_path

import json
from pathlib import Path
from typing import Any

FACTUALITY_CONTRACT_VERSION = "grounded_factuality_response_contract_v1"

# Prompt revision. r1 measured: recall 0.733 on T3's planted errors (all 11 detections verified
# against the actual injected text -- genuine, not artifacts), but false positives on both clean
# arms (T1 5/37, T2 7/36) with two diagnosed causes:
#
#   1. `what_tutor_should_explain` was used as a contradiction reference. It is a COVERAGE field
#      -- it says what a complete explanation contains -- so a terse-but-correct tutor reads as
#      "contradicting" it. 15 of 30 flagged contradictions cited this field.
#   2. Topic matching without polarity checking. The judge flagged tutors for DISCUSSING the
#      topic of a red flag even when they stated the correct opposite. Real example: the tutor
#      said "Gain Ratio reduces that bias" and was flagged against the red flag "Claiming Gain
#      Ratio increases the preference" -- the tutor said the opposite of the error.
#
# r2 splits the reference into ERROR PATTERNS (the contradiction basis) and BACKGROUND (context
# only, explicitly not a checklist), and states the polarity rule directly.
# r2 measured: recall improved (0.733 -> 0.800) but precision got WORSE, not better
# (T1 contradiction rate 0.135 -> 0.250). Diagnosis: all 12 of T1's flags cited genuine library
# error patterns, but every flagged tutor statement was CORRECT -- e.g. the tutor stating Bayes'
# theorem correctly was flagged as "Confusing the likelihood with the posterior". Making error
# patterns the headline made the judge hunt harder for them.
#
# The root cause is the TASK FRAMING, not the wording. "Here is a list of plausible mistakes --
# does the tutor make any?" is generative, open-ended and suggestible: an 8B judge asked to find
# errors will find them, and polarity instructions do not survive that pull. Two revisions of
# prompt wording failed to fix it, which is the signal to change the task rather than the words.
#
# r3 replaces error-hunting with per-pattern STANCE CLASSIFICATION: for each supplied error
# pattern, does the tutor ASSERT it, DENY it, or NOT ADDRESS it? That is discriminative rather
# than generative, it makes polarity the actual question instead of a caveat, and "not_addressed"
# is an easy, low-effort default rather than something the judge must resist finding.
#
# This is NOT the atomic decomposition contraindicated by arXiv 2603.28005 -- that finding is
# about decomposing the CANDIDATE ANSWER into atomic claims. Here the tutor text stays whole and
# is judged holistically; only the REFERENCE, which is already a discrete list, supplies the
# items. The verdict is then derived in code from the stances rather than self-reported, on the
# same principle that removed reliance on the judge's trust_adjusted_score.
# r3 measured: recall 0.000 at 8B, and ~0 at 14B too (2 asserts in 278 stances). Identical
# behaviour at two model sizes ruled OUT a simple capability ceiling and pointed at the prompt.
#
# A 15-item direct probe settled it. Stripping all framing -- show the tutor's claim and the
# reference, ask "is anything factually wrong here?" -- both models identified the planted errors:
# 8B 11/14, 14B 14/15. They correctly named reversed learning/classification phases, swapped
# ordinal/nominal, even the exact arithmetic slip ("denominator 0.0110 instead of 0.0198").
#
# So the models KNOW these claims are false. r3 suppressed the answer. Two causes:
#   - The candidate-mistake list framed the task as LIST MATCHING, so the judge asked "does the
#     tutor make mistake P4?" rather than "is the tutor right?", and defaulted to not_addressed.
#   - Nothing forced engagement with the actual fact. A stance label is cheap to guess; stating
#     the CORRECTION is not.
#
# r4 keeps r1's direct question but adds the probe's two working ingredients: no candidate list,
# and a required `correct_version` for every flagged claim. Naming the correction forces the
# model to reason about the fact rather than pattern-match a flag, which is also what makes the
# output auditable -- a wrong correction is visible, a wrong flag alone is not.
PROMPT_REVISION = "r4_direct_with_required_correction"

# Fields that describe what an ERROR looks like -- the only legitimate basis for a contradiction.
ERROR_PATTERN_FIELDS = ("common_confusions_or_errors", "red_flags")
# Describes what a COMPLETE explanation covers. Context only; never a contradiction basis.
COVERAGE_FIELDS = ("what_tutor_should_explain",)

# Domain-specific INPUT, not framework logic. Resolved through corpus_config so a
# different subject domain is a config change, never a source edit. See
# seg_eval.corpus_config for the override mechanism.
REVIEWED_LIBRARY = kc_library_path()

REFERENCE_FIELDS = ("what_tutor_should_explain", "common_confusions_or_errors", "red_flags")

ALLOWED_VERDICTS = {"grounded", "minor_deviation", "contradicted", "insufficient_reference"}

# Severity of a contradiction, kept coarse on purpose: a finer scale would invite the judge to
# split hairs, and the downstream cap only needs "material or not".
ALLOWED_SEVERITY = {"material", "minor"}


def load_reference_by_kc(path: str | Path) -> dict[str, dict[str, list[str]]]:
    """kc_id -> {reference field -> entries}. Only the three curriculum-content fields; nothing
    about review status, provenance, or matching cues belongs in a factuality reference."""
    out: dict[str, dict[str, list[str]]] = {}
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            support = row.get("evaluation_support") or {}
            block = {f: list(support.get(f) or []) for f in REFERENCE_FIELDS if support.get(f)}
            if block:
                block["scope_status"] = row.get("scope_status")
                out[str(row["kc_id"])] = block
    return out


def reference_for_packet(
    packet: dict[str, Any], reference_by_kc: dict[str, dict], max_kcs: int = 3,
) -> list[dict[str, Any]]:
    """Curriculum reference for one unit: primary KC first, then evaluator_kc_set, capped.

    The cap exists because this pass's whole advantage is a small prompt (decision 2 above); a
    unit whose evaluator_kc_set names nine KCs would otherwise reintroduce exactly the prompt
    bloat that broke the grounding ablation. Primary KC is always included when it has content.
    """
    target = packet.get("evaluation_target") or {}
    ordered: list[str] = []
    for kc in [target.get("primary_kc_id"), *(target.get("evaluator_kc_set") or [])]:
        if kc and kc in reference_by_kc and kc not in ordered:
            ordered.append(str(kc))
    out = []
    for kc in ordered[:max_kcs]:
        block = dict(reference_by_kc[kc])
        block["kc_id"] = kc
        out.append(block)
    return out


def tutor_lines(packet: dict[str, Any]) -> list[str]:
    """Tutor turns only. The student's own statements are not the tutor's claims and must not be
    scored as tutor error -- but student text is still shown as context by the caller when the
    tutor's turn only makes sense against it."""
    lines = []
    for ex in packet.get("member_exchanges") or []:
        if isinstance(ex, dict) and (ex.get("tutor_text") or "").strip():
            lines.append(f"[{ex.get('exchange_id','?')}] Tutor: {ex['tutor_text']}")
    return lines


def segment_lines(packet: dict[str, Any]) -> list[str]:
    lines = []
    for ex in packet.get("member_exchanges") or []:
        if not isinstance(ex, dict):
            continue
        exid = ex.get("exchange_id", "?")
        if (ex.get("student_text") or "").strip():
            lines.append(f"[{exid}] Student: {ex['student_text']}")
        if (ex.get("tutor_text") or "").strip():
            lines.append(f"[{exid}] Tutor: {ex['tutor_text']}")
    return lines


def system_prompt() -> str:
    return (
        "You are a curriculum fact-checker. You are given a tutoring segment and the curriculum "
        "reference for the knowledge components it covers. Judge ONLY whether the tutor's "
        "statements contradict the supplied curriculum reference. Do not judge teaching quality, "
        "style, helpfulness, or completeness. Return only valid JSON. Do not include hidden "
        "chain-of-thought."
    )


def split_reference(reference: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """(error_patterns, background). Only error patterns may ground a contradiction."""
    errors, background = [], []
    for block in reference:
        kc = block.get("kc_id")
        e = {f: block[f] for f in ERROR_PATTERN_FIELDS if block.get(f)}
        b = {f: block[f] for f in COVERAGE_FIELDS if block.get(f)}
        if e:
            errors.append({"kc_id": kc, **e})
        if b:
            background.append({"kc_id": kc, **b})
    return errors, background


def numbered_error_patterns(reference: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flat, numbered list of candidate error patterns -- the items to classify stance on."""
    error_patterns, _ = split_reference(reference)
    out = []
    for block in error_patterns:
        for field in ERROR_PATTERN_FIELDS:
            for text in block.get(field, []):
                out.append({"pattern_id": f"P{len(out) + 1}", "kc_id": block.get("kc_id"),
                             "mistake": text})
    return out


def user_prompt(packet: dict[str, Any], reference: list[dict[str, Any]]) -> str:
    """r4: direct factual check with a REQUIRED correction for every flagged claim.

    Deliberately does NOT show a list of candidate mistakes. r3 did, and the judge switched from
    "is the tutor right?" to "does the tutor make mistake P4?", defaulting to not_addressed and
    scoring 0.000 recall. The reference is supplied as curriculum content, not as a checklist.
    """
    curriculum: list[str] = []
    for block in reference:
        for field in REFERENCE_FIELDS:
            curriculum.extend(block.get(field) or [])

    return (
        "CURRICULUM REFERENCE (what the course material says about this topic):\n- "
        + "\n- ".join(curriculum[:14]) + "\n\n"
        "TUTORING SEGMENT:\n"
        + "\n".join(segment_lines(packet)) + "\n\n"
        "Question: is anything the TUTOR says factually WRONG according to the curriculum "
        "reference above?\n\n"
        "Check the specific values, directions, definitions and formulas the tutor states. "
        "For each thing that is wrong, you must say what the tutor claimed AND what it should "
        "say instead. If you cannot state the correction, it is not an error.\n\n"
        "Rules:\n"
        "- Judge the TUTOR's words only. The student's own statements are never a tutor error.\n"
        "- A tutor agreeing with or endorsing a student's incorrect claim IS a tutor error.\n"
        "- Being brief, or not explaining something fully, is NOT an error. Only a statement "
        "that is actually WRONG counts. You are not judging teaching quality.\n"
        "- Different wording, extra detail, or a different worked example are not errors.\n"
        "- Most segments contain no error at all. Saying so is the expected answer.\n\n"
        "OUTPUT REQUIREMENTS:\n"
        "- Return ONE JSON object and nothing else. No prose, no markdown fences.\n"
        "- Shape:\n"
        "{\n"
        '  "segment_id": "<id>",\n'
        '  "errors": [\n'
        '    {"tutor_quote": "<the tutor exact words, copied VERBATIM>",\n'
        '     "what_is_wrong": "<why it is wrong>",\n'
        '     "correct_version": "<what it should say instead>",\n'
        '     "severity": "material | minor"}\n'
        "  ],\n"
        '  "rationale_short": "<1-3 sentences>"\n'
        "}\n"
        "- errors MUST be [] when the tutor says nothing factually wrong.\n"
        "- severity is 'material' if a student acting on the claim would be wrong.\n"
        f"- segment_id must be exactly: {packet.get('segment_id')}\n"
    )


def validate_error_response(
    response: dict[str, Any], *, expected_segment_id: str, allowed_tutor_text: str | None = None,
) -> list[str]:
    """Validate an r4 response. The required correction is the load-bearing check: a flag whose
    `correct_version` is missing is exactly the unreasoned pattern-match r1/r2 produced."""
    errors: list[str] = []

    if response.get("segment_id") != expected_segment_id:
        errors.append(
            f"segment_id_mismatch:expected={expected_segment_id}:actual={response.get('segment_id')}")

    found = response.get("errors")
    if not isinstance(found, list):
        errors.append("errors:not_list")
        return errors

    norm_tutor = " ".join((allowed_tutor_text or "").lower().split())
    for i, e in enumerate(found):
        if not isinstance(e, dict):
            errors.append(f"errors[{i}]:not_object")
            continue
        q = e.get("tutor_quote")
        if not isinstance(q, str) or len(q.strip()) < 4:
            errors.append(f"errors[{i}].tutor_quote:missing_or_too_short")
        elif norm_tutor and " ".join(q.lower().split()) not in norm_tutor:
            errors.append(f"errors[{i}].tutor_quote:not_found_in_tutor_text")
        if not isinstance(e.get("correct_version"), str) or not e.get("correct_version", "").strip():
            errors.append(f"errors[{i}].correct_version:required")
        if not isinstance(e.get("what_is_wrong"), str) or not e.get("what_is_wrong", "").strip():
            errors.append(f"errors[{i}].what_is_wrong:required")
        if e.get("severity") not in ALLOWED_SEVERITY:
            errors.append(f"errors[{i}].severity:invalid:{e.get('severity')!r}")

    rationale = response.get("rationale_short")
    if not isinstance(rationale, str) or len(rationale.strip()) < 10:
        errors.append("rationale_short:missing_or_too_short")

    return errors


def derive_verdict_from_errors(response: dict[str, Any]) -> tuple[str, int]:
    """(verdict, material_count) computed in code from the reported errors."""
    found = response.get("errors")
    if not isinstance(found, list):
        return "insufficient_reference", 0
    material = sum(1 for e in found if isinstance(e, dict) and e.get("severity") == "material")
    if material:
        return "contradicted", material
    if found:
        return "minor_deviation", 0
    return "grounded", 0


ALLOWED_STANCES = {"asserts", "denies", "not_addressed"}


def validate_stance_response(
    response: dict[str, Any], *, expected_segment_id: str,
    expected_pattern_ids: list[str], allowed_tutor_text: str | None = None,
) -> list[str]:
    """Validate an r3 stance-classification response.

    The verbatim-quote requirement carries the real weight: 'asserts' without a quote the tutor
    actually said is exactly the failure mode r1/r2 exhibited (a plausible-sounding error matched
    on topic), so an unquotable assertion is rejected rather than scored.
    """
    errors: list[str] = []

    if response.get("segment_id") != expected_segment_id:
        errors.append(
            f"segment_id_mismatch:expected={expected_segment_id}:actual={response.get('segment_id')}")

    stances = response.get("stances")
    if not isinstance(stances, list):
        errors.append("stances:not_list")
        return errors

    got_ids = [s.get("pattern_id") for s in stances if isinstance(s, dict)]
    missing = [p for p in expected_pattern_ids if p not in got_ids]
    extra = [p for p in got_ids if p not in expected_pattern_ids]
    if missing:
        errors.append(f"stances.missing:{missing[:5]}")
    if extra:
        errors.append(f"stances.unexpected:{extra[:5]}")

    norm_tutor = " ".join((allowed_tutor_text or "").lower().split())
    for i, s in enumerate(stances):
        if not isinstance(s, dict):
            errors.append(f"stances[{i}]:not_object")
            continue
        stance = s.get("stance")
        if stance not in ALLOWED_STANCES:
            errors.append(f"stances[{i}].stance:invalid:{stance!r}")
            continue
        if stance == "asserts":
            q = s.get("tutor_quote")
            if not isinstance(q, str) or len(q.strip()) < 4:
                errors.append(f"stances[{i}].tutor_quote:required_when_asserts")
            elif norm_tutor and " ".join(q.lower().split()) not in norm_tutor:
                errors.append(f"stances[{i}].tutor_quote:not_found_in_tutor_text")
            if s.get("severity") not in ALLOWED_SEVERITY:
                errors.append(f"stances[{i}].severity:invalid:{s.get('severity')!r}")

    rationale = response.get("rationale_short")
    if not isinstance(rationale, str) or len(rationale.strip()) < 10:
        errors.append("rationale_short:missing_or_too_short")

    return errors


def derive_verdict_from_stances(response: dict[str, Any]) -> tuple[str, int]:
    """(verdict, material_count) computed IN CODE from the stances.

    The judge is never asked for the verdict directly. It reports per-pattern stances; the
    conclusion is derived. Same principle that removed reliance on the judge's self-reported
    trust_adjusted_score -- a model should supply observations, not the summary judgment those
    observations are supposed to support.
    """
    stances = response.get("stances")
    if not isinstance(stances, list) or not stances:
        return "insufficient_reference", 0

    asserted = [s for s in stances
                 if isinstance(s, dict) and s.get("stance") == "asserts"]
    material = sum(1 for s in asserted if s.get("severity") == "material")

    if material:
        return "contradicted", material
    if asserted:
        return "minor_deviation", 0
    return "grounded", 0


def stance_response_to_contradictions(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Render asserted stances in the shape the rest of the pipeline already consumes."""
    return [
        {"tutor_quote": s.get("tutor_quote"), "conflicts_with": s.get("pattern_id"),
         "severity": s.get("severity")}
        for s in (response.get("stances") or [])
        if isinstance(s, dict) and s.get("stance") == "asserts"
    ]


def validate_factuality_response(
    response: dict[str, Any], *, expected_segment_id: str, allowed_tutor_text: str | None = None,
) -> list[str]:
    """Structural validation, plus a verbatim check on quotes when the segment text is supplied.

    The quote check is what stops a contradiction being asserted against words the tutor never
    said -- the same discipline response_contract_v2 applies to evidence_quotes.
    """
    errors: list[str] = []

    if response.get("segment_id") != expected_segment_id:
        errors.append(
            f"segment_id_mismatch:expected={expected_segment_id}:actual={response.get('segment_id')}")

    verdict = response.get("verdict")
    if verdict not in ALLOWED_VERDICTS:
        errors.append(f"verdict:invalid:{verdict!r}")

    contradictions = response.get("contradictions")
    if not isinstance(contradictions, list):
        errors.append("contradictions:not_list")
        contradictions = []
    else:
        for i, c in enumerate(contradictions):
            if not isinstance(c, dict):
                errors.append(f"contradictions[{i}]:not_object")
                continue
            quote = c.get("tutor_quote")
            if not isinstance(quote, str) or len(quote.strip()) < 4:
                errors.append(f"contradictions[{i}].tutor_quote:missing_or_too_short")
            elif allowed_tutor_text is not None:
                norm = " ".join(quote.lower().split())
                if norm and norm not in " ".join(allowed_tutor_text.lower().split()):
                    errors.append(f"contradictions[{i}].tutor_quote:not_found_in_tutor_text")
            if c.get("severity") not in ALLOWED_SEVERITY:
                errors.append(f"contradictions[{i}].severity:invalid:{c.get('severity')!r}")
            if not isinstance(c.get("conflicts_with"), str) or not c.get("conflicts_with", "").strip():
                errors.append(f"contradictions[{i}].conflicts_with:missing")

    if verdict == "contradicted" and not contradictions:
        errors.append("verdict_contradicted_but_no_contradictions_listed")
    if verdict in ("grounded", "insufficient_reference") and contradictions:
        errors.append(f"verdict_{verdict}_but_contradictions_listed")

    rationale = response.get("rationale_short")
    if not isinstance(rationale, str) or len(rationale.strip()) < 10:
        errors.append("rationale_short:missing_or_too_short")

    return errors


def material_contradiction_count(response: dict[str, Any]) -> int:
    """How many material contradictions this response asserts. The deterministic trust cap keys
    off this rather than off the verdict label alone."""
    return sum(
        1 for c in (response.get("contradictions") or [])
        if isinstance(c, dict) and c.get("severity") == "material"
    )
