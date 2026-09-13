"""Focus-target selection — proposal section 6.4 / Figure 4, implemented as a deterministic,
rule-based classifier over a segment's member exchanges.

WHAT THIS IS
------------
A high-precision layer that tries to identify the specific turn(s) within a segment that should
anchor curriculum-grounded CORRECTNESS checking, so the judge isn't distracted by the rest of
the segment's content when checking one factual claim. When no pattern matches with confidence,
this falls back to a broad target (all tutor content in the segment) rather than guessing --
per the user's instruction, "high-precision... with broad correctness fallback when no precise
target exists," and per execution-spec policy: never let "no clear pattern" become "skip
correctness checking".

WHAT THIS IS NOT
----------------
- Not a segmentation change. Operates read-only on an already-built packet's `member_exchanges`;
  never touches segment boundaries, KC assignment, or the full `segment_text` used for
  dialogue-dependent dimensions (scaffolding, error handling, etc.) -- those keep seeing the
  full segment exactly as before.
- Not a text-synthesis step. The proposal's four scenarios (Figure 4) all turn out to be
  SELECTION and, in one case, CONCATENATION of verbatim turn text -- never a paraphrase or a
  synthesized sentence blending both parties' words. This module only ever extracts or
  concatenates existing text, matching the same verbatim-quote discipline already enforced
  everywhere else in this codebase (response_contract_v2's evidence_quotes).
- Not an LLM call. Classification is deterministic lexical pattern matching so it is
  reproducible run to run without depending on generation settings or a model revision.

FOUR PATTERNS (proposal wording; Figure 4 scenario mapping in parentheses)
----------------------------------------------------------------------
- confirmation (Scenario 1, "use approved student turn"): tutor turn is a short, pure
  acknowledgement with no new substantive content -- the student's own claim was already
  correct. focus_span = the student's turn.
- self_correction (Scenario 2, "use student turn after correction"): a later student turn (not
  the first exchange in the segment) contains a self-correction marker ("oh wait", "actually I
  mean", ...). focus_span = that later student turn, verbatim. The original error stays in the
  full segment text for the diagnostic dimensions, untouched.
- tutor_correction (Scenario 3, "use combination of student and tutor turn after correction"):
  the tutor's turn contains an explicit correction marker responding to a substantive student
  claim in the same exchange. focus_span = the student's claim concatenated with the tutor's
  correction, both verbatim.
- tutor_explanation (Scenario 4, "use tutor answer"): the student's turn is a generic question
  with no specific claim to check; the tutor's turn carries all the instructional content.
  focus_span = the tutor's turn, verbatim.

Anything not matching one of these with a clear lexical signal falls back to
``broad_tutor_claim_target`` (concatenation of every tutor turn in the segment) or, in the
genuine edge cases, ``no_domain_claim`` (segment has no tutor content at all) or
``insufficient_or_ambiguous_target`` (segment has no exchanges to classify).

This is a first version. The signal lists are intentionally explicit and testable rather than
exhaustive; expanding them is safe (it only ever shifts more segments from broad to precise,
never the reverse) and should be done by adding to the marker lists and their tests, not by
loosening the structural conditions (exchange position, presence of a correction alongside an
assertion, etc.) that keep this conservative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

TARGET_TYPES = (
    "precise_focus_target",
    "broad_tutor_claim_target",
    "no_domain_claim",
    "insufficient_or_ambiguous_target",
)

PATTERNS = ("confirmation", "self_correction", "tutor_correction", "tutor_explanation")

FOCUS_TARGET_SCHEMA_VERSION = "focus_target_v1"

_SELF_CORRECTION_RE = re.compile(
    r"\boh,?\s*wait\b|\bwait[,.]|\bactually,?\s*i\b|\bi mean\b|\bscratch that\b|"
    r"\bsorry,?\s*i meant\b|\blet me redo\b|\bi made a mistake\b|\bcorrection:\b|"
    r"\boh!?\s*(?:it|i)\s*should\b|\bhold on\b|\bwait,?\s*that'?s wrong\b",
    re.IGNORECASE,
)

_CORRECTION_RE = re.compile(
    r"\bactually\b|\bnot quite\b|\bnot correct\b|\bthat'?s not right\b|\bshould be\b|"
    r"\brather than\b|\binstead of\b|\bclose,?\s*but\b|\balmost,?\s*but\b|"
    r"\bsmall (?:mistake|correction|issue)\b|\bthe issue is\b|\byou need to\b|"
    r"^\s*no[,.]|\bincorrect\b|\bthat'?s a common (?:mistake|error)\b",
    re.IGNORECASE,
)

_CONFIRMATION_RE = re.compile(
    r"\bexactly\b|\bthat'?s right\b|\bthat'?s correct\b|\bperfect\b|\bwell done\b|"
    r"\bgood job\b|\byou got it\b|\bspot on\b|\bprecisely\b|\bthat'?s it\b|^\s*yes[,.!]?\s*$|^\s*yes,",
    re.IGNORECASE,
)

_PURE_QUESTION_RE = re.compile(
    r"^\s*(?:what|how|why|when|where|which)\b.*\?\s*$|"
    r"^\s*(?:can|could)\s+you\b.*\?\s*$|"
    r"\bi don'?t understand\b|\bi'?m confused\b|\bhelp me understand\b|"
    r"^\s*explain\b",
    re.IGNORECASE,
)

_CLAIM_MARKER_RE = re.compile(r"[=<>]|\d|\bi think\b|\bi believe\b|\bso\b|\bthen\b", re.IGNORECASE)

_CONFIRMATION_MAX_CHARS = 140
_MIN_ASSERTION_CHARS = 15


@dataclass(frozen=True)
class FocusTarget:
    target_type: str
    pattern: str | None
    focus_span: str
    source_exchange_ids: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.target_type not in TARGET_TYPES:
            raise ValueError(f"invalid target_type: {self.target_type!r}")
        if self.pattern is not None and self.pattern not in PATTERNS:
            raise ValueError(f"invalid pattern: {self.pattern!r}")
        if self.target_type == "precise_focus_target" and self.pattern is None:
            raise ValueError("precise_focus_target requires a pattern")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FOCUS_TARGET_SCHEMA_VERSION,
            "target_type": self.target_type,
            "pattern": self.pattern,
            "focus_span": self.focus_span,
            "source_exchange_ids": list(self.source_exchange_ids),
            "signals": list(self.signals),
        }


def _text(ex: dict[str, Any], key: str) -> str:
    value = ex.get(key)
    return value.strip() if isinstance(value, str) else ""


def _looks_like_assertion(student_text: str) -> bool:
    if len(student_text) < _MIN_ASSERTION_CHARS:
        return False
    if _PURE_QUESTION_RE.search(student_text) and not _CLAIM_MARKER_RE.search(student_text):
        return False
    return True


def _looks_like_pure_question(student_text: str) -> bool:
    if not student_text:
        return False
    if _CLAIM_MARKER_RE.search(student_text):
        return False
    return bool(_PURE_QUESTION_RE.search(student_text))


def _is_short_confirmation(tutor_text: str) -> bool:
    if not tutor_text or len(tutor_text) > _CONFIRMATION_MAX_CHARS:
        return False
    if "?" in tutor_text:
        return False
    if _CORRECTION_RE.search(tutor_text):
        return False
    return bool(_CONFIRMATION_RE.search(tutor_text))


def classify_segment_focus(member_exchanges: list[dict[str, Any]]) -> FocusTarget:
    """Classify one segment's focus target from its member exchanges. Pure function, no I/O."""
    if not member_exchanges:
        return FocusTarget(
            target_type="insufficient_or_ambiguous_target", pattern=None, focus_span="",
            signals=["empty_segment_no_exchanges"],
        )

    # 1. self_correction: scan exchanges after the first for a self-correction marker in the
    #    STUDENT's turn. Checked first because it is the most structurally distinctive pattern
    #    (requires a specific later-turn marker) and, per the proposal, should take priority when
    #    the segment's own error-then-fix arc is unambiguous.
    if len(member_exchanges) >= 2:
        for ex in member_exchanges[1:]:
            student = _text(ex, "student_text")
            if student and _SELF_CORRECTION_RE.search(student):
                return FocusTarget(
                    target_type="precise_focus_target", pattern="self_correction",
                    focus_span=student,
                    source_exchange_ids=[ex.get("exchange_id")] if ex.get("exchange_id") else [],
                    signals=["self_correction_marker_in_later_student_turn"],
                )

    # 2. tutor_correction: an explicit correction responding to a substantive student claim,
    #    within the same exchange. Checked next because an explicit correction marker is a very
    #    strong, low-false-positive signal.
    for ex in member_exchanges:
        student = _text(ex, "student_text")
        tutor = _text(ex, "tutor_text")
        if tutor and _CORRECTION_RE.search(tutor) and _looks_like_assertion(student):
            return FocusTarget(
                target_type="precise_focus_target", pattern="tutor_correction",
                focus_span=f"Student: {student}\nTutor: {tutor}",
                source_exchange_ids=[ex.get("exchange_id")] if ex.get("exchange_id") else [],
                signals=["correction_marker_in_tutor_turn", "assertion_in_student_turn"],
            )

    # 3. confirmation: short, pure acknowledgement with no correction marker and no question.
    for ex in member_exchanges:
        student = _text(ex, "student_text")
        tutor = _text(ex, "tutor_text")
        if student and _is_short_confirmation(tutor):
            return FocusTarget(
                target_type="precise_focus_target", pattern="confirmation",
                focus_span=student,
                source_exchange_ids=[ex.get("exchange_id")] if ex.get("exchange_id") else [],
                signals=["short_affirming_tutor_turn"],
            )

    # 4. tutor_explanation: a generic student question with no specific claim, answered with
    #    substantive tutor content.
    for ex in member_exchanges:
        student = _text(ex, "student_text")
        tutor = _text(ex, "tutor_text")
        if tutor and _looks_like_pure_question(student):
            return FocusTarget(
                target_type="precise_focus_target", pattern="tutor_explanation",
                focus_span=tutor,
                source_exchange_ids=[ex.get("exchange_id")] if ex.get("exchange_id") else [],
                signals=["generic_question_student_turn"],
            )

    # Fallback: broad target over all tutor content, or the rarer no-content edge cases.
    tutor_parts = [f"Tutor: {_text(ex, 'tutor_text')}" for ex in member_exchanges if _text(ex, "tutor_text")]
    if tutor_parts:
        return FocusTarget(
            target_type="broad_tutor_claim_target", pattern=None,
            focus_span="\n".join(tutor_parts),
            source_exchange_ids=[ex.get("exchange_id") for ex in member_exchanges if ex.get("exchange_id")],
            signals=["no_precise_pattern_matched"],
        )

    return FocusTarget(
        target_type="no_domain_claim", pattern=None, focus_span="",
        source_exchange_ids=[ex.get("exchange_id") for ex in member_exchanges if ex.get("exchange_id")],
        signals=["no_tutor_content_in_segment"],
    )


def compute_focus_target(member_exchanges: list[dict[str, Any]]) -> dict[str, Any]:
    """Packet-embeddable dict form. Use this from packet builders."""
    return classify_segment_focus(member_exchanges).to_dict()
