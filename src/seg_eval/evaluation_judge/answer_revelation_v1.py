"""Replacement for ``solution_control``: separate the observation from the appropriateness call.

MEASURED PROBLEM
----------------
``solution_control`` agrees with human labels at 0.79 but Cohen's kappa is -0.01 (Qwen) and -0.07
(Selene) -- below chance for *both* models, which means the defect is in the dimension, not the
judge. The confusion matrix on the 76 human-labelled local units shows why::

    human marginal : {1: 62, 0.5: 14}
    judge marginal : {1: 73, 0.5:  3}
    (human 0.5, judge 1) = 14      <- every single partial the human found, the judge scored 1
    (human 1,   judge 0.5) =  3

The judge is close to a constant predictor. On the deliberately answer-dumping tutor variant it
flags only 7 of 35 segments. It is not blind -- it does move in the right direction -- but its
threshold is far too lenient to be usable.

WHY: THE QUESTION HAS TWO ESCAPE HATCHES
----------------------------------------
The live question is::

    "Does the tutor control revealingness, avoiding UNNECESSARY answer dumping
     WHEN GUIDED REASONING IS APPROPRIATE?"

That packs three judgements into one 0/0.5/1 score: (1) did the tutor reveal the answer, (2) was
revealing unnecessary, (3) was guided reasoning appropriate here. A judge can score 1 on any
segment by deciding revealing was necessary, or that guided reasoning was not appropriate. Neither
decision has to be stated or defended, so the escape is silent and unfalsifiable.

THE FIX
-------
Split it, following MRBench's treatment of the same construct. Their "Revealing of the Answer" is a
pure observation -- Yes(correct) / Yes(incorrect) / No, desideratum *No* -- and appropriateness
lives in a separate dimension. It is one of their most discriminative measures (they report GPT-4
revealing 47% of the time), precisely because an observation has no escape hatch.

Here that becomes two judgements the model must make separately and defend:

1. ``revelation`` -- what was actually said. Requires a VERBATIM quote when anything was revealed,
   the same device the r4 factuality contract uses: a flag you cannot quote is not a flag.
2. ``warrant`` -- if something was revealed, *why it was legitimate*. The escape hatch still
   exists, because sometimes revealing genuinely is correct teaching, but it must now be named
   from a closed list and quoted, so it can be audited and counted.

``solution_control`` is then DERIVED in code from the two, never asked for directly. The scoring
policy is stated once, in :func:`derive_solution_control`, instead of living implicitly inside a
model's interpretation of a compound question.

This also yields the answer-revelation rate as a first-class statistic, directly comparable to
MRBench's headline number.

GENERICITY
----------
Nothing here names a subject domain. "The answer" means whatever the segment's own task is; the
warrant categories are pedagogical, not topical.
"""

from __future__ import annotations

from typing import Any

CONTRACT_VERSION = "answer_revelation_v1"
PROMPT_REVISION = "r1_observation_then_warrant"

# what was revealed -- an observation about the text, not a judgement about teaching
REVELATION_NOT = "not_revealed"
REVELATION_PARTIAL = "revealed_partial"
REVELATION_FULL = "revealed_full"
ALLOWED_REVELATION = (REVELATION_NOT, REVELATION_PARTIAL, REVELATION_FULL)

# why revealing was legitimate -- a closed list, so the escape hatch is named and countable
WARRANT_NONE = "no_warrant"
ALLOWED_WARRANT = (
    "student_asked_directly_after_genuine_effort",
    "student_stuck_after_multiple_attempts",
    "segment_is_expository_by_design",
    "answer_was_already_known_to_the_student",
    WARRANT_NONE,
)

# derived score policy -- stated once, here, rather than implied by a compound question
SCORE_NOT_REVEALED = 1.0
SCORE_REVEALED_WARRANTED = 1.0
SCORE_PARTIAL_UNWARRANTED = 0.5
SCORE_FULL_UNWARRANTED = 0.0


def _branch(props: dict[str, Any]) -> dict[str, Any]:
    base = {
        "segment_id": {"type": "string"},
        "reasoning": {"type": "string", "minLength": 10},
    }
    base.update(props)
    return {"type": "object", "properties": base, "required": list(base),
            "additionalProperties": False}


def build_schema() -> dict[str, Any]:
    """Grammar in which an unquoted revelation and an unnamed warrant are ungeneratable.

    The nullable fields are expressed as a ``oneOf`` over the two legal shapes rather than as
    ``{"type": ["string", "null"]}``. That earlier form let the model satisfy the grammar by
    emitting the *string* ``"null"`` -- which it did on 7 of the first 15 units, a pure
    serialization artifact with a correct judgement behind it. Splitting the branches removes the
    ambiguity at generation time instead of teaching the validator to forgive it, which would have
    meant repairing output.
    """
    return {
        "oneOf": [
            # nothing revealed: the quote and warrant fields have no legal content at all
            _branch({
                "revelation": {"const": REVELATION_NOT},
                "revealed_quote": {"type": "null"},
                "warrant": {"const": WARRANT_NONE},
                "warrant_evidence": {"type": "null"},
            }),
            # something revealed and NOT excused -- the branch that costs the tutor points, so it
            # carries the same evidence burden as the branch that excuses them
            _branch({
                "revelation": {"type": "string",
                                "enum": [REVELATION_PARTIAL, REVELATION_FULL]},
                "revealed_quote": {"type": "string", "minLength": 8},
                "warrant": {"const": WARRANT_NONE},
                "warrant_evidence": {"type": "string", "minLength": 8},
            }),
            # something revealed and excused: both the quote and the evidence must exist
            _branch({
                "revelation": {"type": "string",
                                "enum": [REVELATION_PARTIAL, REVELATION_FULL]},
                "revealed_quote": {"type": "string", "minLength": 8},
                "warrant": {"type": "string",
                             "enum": [w for w in ALLOWED_WARRANT if w != WARRANT_NONE]},
                "warrant_evidence": {"type": "string", "minLength": 8},
            }),
            # revealed in a segment with no student turn to quote. The grammar has to PERMIT this
            # shape or the model cannot express it: without the branch it wrote the placeholder
            # string "null - no student turn is applicable" to satisfy a minLength on a field it
            # had nothing to put in. Whether the claim is true is not taken on trust -- the
            # validator checks the packet for student turns and rejects this when any exist.
            _branch({
                "revelation": {"type": "string",
                                "enum": [REVELATION_PARTIAL, REVELATION_FULL]},
                "revealed_quote": {"type": "string", "minLength": 8},
                "warrant": {"const": "segment_is_expository_by_design"},
                "warrant_evidence": {"type": "null"},
            }),
        ]
    }


def derive_solution_control(response: dict[str, Any]) -> tuple[float | None, str]:
    """(score, reason) computed in code from the observation and the warrant.

    Never asks the model for the score. A model that wants to excuse a revelation must name the
    warrant, and the arithmetic consequence of that choice is fixed here rather than negotiated
    inside the model's own head.
    """
    rev = response.get("revelation")
    if rev not in ALLOWED_REVELATION:
        return None, f"unusable revelation value: {rev!r}"

    if rev == REVELATION_NOT:
        return SCORE_NOT_REVEALED, "nothing was revealed; guided reasoning was preserved"

    warrant = response.get("warrant")
    if warrant not in ALLOWED_WARRANT:
        return None, f"unusable warrant value: {warrant!r}"

    if warrant != WARRANT_NONE:
        return SCORE_REVEALED_WARRANTED, f"revealed, but warranted: {warrant}"

    if rev == REVELATION_PARTIAL:
        return SCORE_PARTIAL_UNWARRANTED, "a decisive intermediate result was given unprompted"
    return SCORE_FULL_UNWARRANTED, "the final answer was given with no warrant"


def student_text_of(packet: dict[str, Any]) -> str:
    """Everything the student said in this segment, for checking warrant evidence against.

    Warrants are claims about the STUDENT -- whether they asked, were stuck, already knew, or were
    mid-reasoning. Checking evidence against the student's own words rather than the whole segment
    is what stops the model justifying a warrant by quoting the tutor, or by writing a description
    of the segment instead of quoting anything at all.
    """
    parts = []
    for ex in packet.get("member_exchanges") or []:
        if isinstance(ex, dict):
            t = ex.get("student_text")
            if isinstance(t, str) and t.strip():
                parts.append(t)
    return "\n".join(parts)


def validate_response(response: dict[str, Any], *, expected_segment_id: str,
                      allowed_tutor_text: str | None = None,
                      allowed_student_text: str | None = None) -> list[str]:
    """Contract check. The verbatim-quote requirement carries the weight.

    A revelation the model cannot quote from the tutor's own words is the same unreasoned
    pattern-match the r1/r2 factuality prompts produced, and is rejected rather than scored.
    """
    errors: list[str] = []

    if response.get("segment_id") != expected_segment_id:
        errors.append(f"segment_id_mismatch:expected={expected_segment_id}:"
                      f"actual={response.get('segment_id')}")

    rev = response.get("revelation")
    if rev not in ALLOWED_REVELATION:
        errors.append(f"revelation:not_allowed:{rev}")

    warrant = response.get("warrant")
    if warrant not in ALLOWED_WARRANT:
        errors.append(f"warrant:not_allowed:{warrant}")

    quote = response.get("revealed_quote")
    if rev in (REVELATION_PARTIAL, REVELATION_FULL):
        if not isinstance(quote, str) or len(quote.strip()) < 8:
            errors.append("revealed_quote:required_when_revealed")
        elif allowed_tutor_text is not None and quote.strip() not in allowed_tutor_text:
            errors.append("revealed_quote:not_found_in_segment_text")
        # symmetric burden: no_warrant is the branch that costs the tutor points, so it must be
        # evidenced exactly as a claimed warrant is. Requiring evidence only for the excusing
        # branch made no_warrant the path of least resistance, and every measured false flag rode
        # on that default.
        ev = response.get("warrant_evidence")
        # A segment with no student text at all IS structurally expository -- there was no student
        # reasoning to interrupt. That is verified here from the packet rather than accepted on the
        # model's say-so, so it cannot become a free escape hatch.
        no_student_activity = (allowed_student_text is not None
                               and not allowed_student_text.strip())
        if warrant == "segment_is_expository_by_design" and ev is None:
            if not no_student_activity:
                errors.append("warrant_evidence:student_turns_exist_so_expository_needs_a_quote")
        elif not isinstance(ev, str) or len(ev.strip()) < 8:
            errors.append("warrant_evidence:required_when_warrant_claimed" if warrant != WARRANT_NONE
                          else "warrant_evidence:required_to_support_no_warrant")
        elif allowed_student_text is not None:
            # every warrant is a claim about the student, so its evidence must be the student's
            # own words -- not the tutor's, and not a description of the segment
            if ev.strip() not in allowed_student_text:
                errors.append("warrant_evidence:not_found_in_student_text")
        elif allowed_tutor_text is not None and ev.strip() not in allowed_tutor_text:
            errors.append("warrant_evidence:not_found_in_segment_text")
    elif rev == REVELATION_NOT:
        if quote not in (None, "") :
            errors.append("revealed_quote:must_be_null_when_not_revealed")
        if warrant != WARRANT_NONE:
            errors.append("warrant:must_be_no_warrant_when_not_revealed")

    if not isinstance(response.get("reasoning"), str) or len(response.get("reasoning", "")) < 10:
        errors.append("reasoning:required")

    return errors


SYSTEM_PROMPT = (
    "You observe tutoring dialogue and report exactly what the tutor did. You do not grade, and "
    "you do not decide whether the teaching was good. Report only what the text supports, and "
    "quote the tutor verbatim whenever you claim they revealed something."
)


def build_user_prompt(*, segment_id: str, segment_text: str, task_context: str = "") -> str:
    """Two questions, asked in order, neither of which can be dodged.

    The observation is deliberately asked FIRST and on its own terms: whether revealing was
    *appropriate* is a separate question, asked afterwards, so it cannot contaminate the record of
    what was actually said.
    """
    parts = [
        "Read the tutoring segment below and answer two separate questions.",
        "",
        "QUESTION 1 -- OBSERVATION. Did the tutor state the answer to the problem the student is "
        "working on?",
        f'  "{REVELATION_NOT}"      the tutor did not state it; the student still has to work it out',
        f'  "{REVELATION_PARTIAL}"  the tutor gave a decisive intermediate result that removes most '
        "of the remaining work",
        f'  "{REVELATION_FULL}"     the tutor stated the final answer outright',
        "",
        "This is a question about what the text says, NOT about whether it was good teaching. "
        "If you answer anything other than not_revealed, you MUST copy the tutor's exact words "
        "into revealed_quote. If you cannot quote it, it was not revealed.",
        "",
        "QUESTION 2 -- WARRANT. Only if something was revealed: was there a legitimate reason?",
        f'  "student_asked_directly_after_genuine_effort"',
        f'  "student_stuck_after_multiple_attempts"',
        f'  "segment_is_expository_by_design"  (the tutor is presenting new material, not probing '
        "-- this is common and legitimate; a tutor explaining a concept nobody was mid-way through "
        "working out has not taken anything away from the student)",
        f'  "answer_was_already_known_to_the_student"',
        f'  "{WARRANT_NONE}"  the student was actively working the problem out and could have '
        "continued, and the tutor cut that short",
        "",
        "EVERY answer to Question 2 needs evidence in warrant_evidence, including no_warrant, and "
        "that evidence must be a VERBATIM quote of something the STUDENT said. Every one of these "
        "is a claim about the student -- that they asked, were stuck, already knew, opened the "
        "topic, or were still working -- so the student's own words are what settles it. Do not "
        "quote the tutor, and do not describe the segment: copy the student's words exactly.",
        "  for a named warrant, quote the student turn that shows it applied",
        f'  for "{WARRANT_NONE}", quote the student turn showing they were still working the '
        "problem and had somewhere left to go",
        "no_warrant is a finding against the tutor, so it carries the same burden as excusing "
        "them: if you cannot point to a student who was actively working, the tutor interrupted "
        "nothing. If the segment contains no student turn at all, use "
        '"segment_is_expository_by_design" with null evidence.',
        "If nothing was revealed, use null for revealed_quote and warrant_evidence, and "
        "\"no_warrant\" for warrant.",
        "",
        f"SEGMENT ID: {segment_id}",
    ]
    if task_context:
        parts += ["", "WHAT THE STUDENT IS WORKING ON:", task_context]
    parts += ["", "SEGMENT:", segment_text]
    return "\n".join(parts)


def revelation_rate(responses: list[dict[str, Any]]) -> dict[str, Any]:
    """Answer-revelation rate, directly comparable to MRBench's headline statistic."""
    counts = {r: 0 for r in ALLOWED_REVELATION}
    unwarranted = 0
    for resp in responses:
        rev = resp.get("revelation")
        if rev in counts:
            counts[rev] += 1
            if rev != REVELATION_NOT and resp.get("warrant") == WARRANT_NONE:
                unwarranted += 1
    n = sum(counts.values())
    revealed = counts[REVELATION_PARTIAL] + counts[REVELATION_FULL]
    return {
        "n": n,
        "counts": counts,
        "revealed_rate": revealed / n if n else None,
        "unwarranted_revelation_rate": unwarranted / n if n else None,
    }
