"""Deterministic analysis of a filled-in Focus Target human-validation sheet
(build_judge_validation_workbook.py's "Focus Target" sheet).

Three metrics, each answering a different question from the task this was built for:

- target_status_agreement: does the human's own, independently-made (blind) classification of
  each segment's target type agree with focus_target_v1's classification? Raw agreement AND
  Cohen's kappa (chance-corrected -- raw agreement alone is misleading when one category
  dominates the sample, which is common here since only 2 of 4 categories occur in practice).
- precise_target_correctness: of the segments the SYSTEM classified as precise_focus_target,
  what fraction did the human confirm the selected span was actually the right one to check?
- missed_material_claim_rate: across ALL rated segments regardless of target type, how often did
  the human find that the system's focus_target missed some real claim entirely?

Every function here is pure and takes plain rows (dicts), not an openpyxl worksheet directly --
see focus_target_validation_row_from_worksheet() for the one place that touches openpyxl, kept
separate so the analysis itself has no dependency on the file format.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FocusValidationRow:
    segment_id: str
    dialogue: str
    system_target_type: str
    human_target_type: str | None = None  # blind STEP1 answer; None = not yet rated
    precise_span_correct: str | None = None  # "Y" | "N" | "NA" | None
    broad_fallback_appropriate: str | None = None  # "Y" | "N" | "NA" | None
    material_claim_missed: str | None = None  # "Y" | "N" | None


def _normalize_yn(value: str | None) -> str | None:
    if value is None:
        return None
    v = str(value).strip().upper()
    return v if v in ("Y", "N", "NA") else None


@dataclass(frozen=True)
class AgreementResult:
    n_rated: int
    n_agree: int
    raw_agreement: float | None
    cohens_kappa: float | None
    confusion: dict[tuple[str, str], int] = field(default_factory=dict)


def cohens_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Chance-corrected agreement over (rater_a, rater_b) category pairs. None if <2 pairs or
    every category present has zero variance in the confusion matrix (kappa undefined)."""
    if len(pairs) < 2:
        return None
    categories = sorted({c for pair in pairs for c in pair})
    n = len(pairs)
    counts = {c: [0, 0] for c in categories}  # [as_a, as_b] marginal counts
    agree = 0
    for a, b in pairs:
        counts[a][0] += 1
        counts[b][1] += 1
        if a == b:
            agree += 1
    po = agree / n
    pe = sum((counts[c][0] / n) * (counts[c][1] / n) for c in categories)
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def target_status_agreement(rows: list[FocusValidationRow]) -> AgreementResult:
    pairs = [
        (r.human_target_type, r.system_target_type)
        for r in rows if r.human_target_type is not None
    ]
    if not pairs:
        return AgreementResult(n_rated=0, n_agree=0, raw_agreement=None, cohens_kappa=None)
    n_agree = sum(1 for a, b in pairs if a == b)
    confusion: dict[tuple[str, str], int] = {}
    for a, b in pairs:
        confusion[(a, b)] = confusion.get((a, b), 0) + 1
    return AgreementResult(
        n_rated=len(pairs), n_agree=n_agree, raw_agreement=n_agree / len(pairs),
        cohens_kappa=cohens_kappa(pairs), confusion=confusion,
    )


@dataclass(frozen=True)
class CorrectnessResult:
    n_applicable: int  # rows where the question was answerable (system's type matched + rated Y/N, not NA/blank)
    n_correct: int
    rate: float | None
    n_not_applicable: int  # explicitly marked NA
    n_unrated: int  # system's type matched but the column was left blank


def precise_target_correctness(rows: list[FocusValidationRow]) -> CorrectnessResult:
    candidates = [r for r in rows if r.system_target_type == "precise_focus_target"]
    return _yn_na_correctness(candidates, lambda r: _normalize_yn(r.precise_span_correct))


def broad_fallback_appropriateness(rows: list[FocusValidationRow]) -> CorrectnessResult:
    candidates = [r for r in rows if r.system_target_type == "broad_tutor_claim_target"]
    return _yn_na_correctness(candidates, lambda r: _normalize_yn(r.broad_fallback_appropriate))


def _yn_na_correctness(candidates: list[FocusValidationRow], getter) -> CorrectnessResult:
    n_na = sum(1 for r in candidates if getter(r) == "NA")
    n_unrated = sum(1 for r in candidates if getter(r) is None)
    applicable = [r for r in candidates if getter(r) in ("Y", "N")]
    n_correct = sum(1 for r in applicable if getter(r) == "Y")
    rate = n_correct / len(applicable) if applicable else None
    return CorrectnessResult(
        n_applicable=len(applicable), n_correct=n_correct, rate=rate,
        n_not_applicable=n_na, n_unrated=n_unrated,
    )


@dataclass(frozen=True)
class MissedClaimResult:
    n_rated: int
    n_missed: int
    rate: float | None


def missed_material_claim_rate(rows: list[FocusValidationRow]) -> MissedClaimResult:
    rated = [r for r in rows if _normalize_yn(r.material_claim_missed) in ("Y", "N")]
    if not rated:
        return MissedClaimResult(n_rated=0, n_missed=0, rate=None)
    n_missed = sum(1 for r in rated if _normalize_yn(r.material_claim_missed) == "Y")
    return MissedClaimResult(n_rated=len(rated), n_missed=n_missed, rate=n_missed / len(rated))


def focus_target_validation_row_from_worksheet_row(values: dict[str, object]) -> FocusValidationRow:
    """Build a FocusValidationRow from one worksheet row's {header: value} dict (openpyxl-agnostic
    -- the caller reads the sheet and passes plain values, this function does no file I/O)."""
    return FocusValidationRow(
        segment_id=str(values.get("Segment ID") or ""),
        dialogue=str(values.get("Dialogue") or ""),
        system_target_type=str(values.get("System target_type") or ""),
        human_target_type=(str(values["STEP1: Your target type"])
                            if values.get("STEP1: Your target type") else None),
        precise_span_correct=values.get("Is the selected precise span correct? (Y/N/NA)"),
        broad_fallback_appropriate=values.get("Is broad fallback appropriate? (Y/N/NA)"),
        material_claim_missed=values.get("Was a material claim missed? (Y/N)"),
    )
