from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Turn:
    dialogue_id: str
    turn_id: int
    role: str
    raw_text: str
    norm_text: str
    noise_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Exchange:
    dialogue_id: str
    exchange_id: str
    exchange_index: int
    turn_start: int
    turn_end: int
    roles_pattern: str
    exchange_text: str
    student_text: str = ""
    tutor_text: str = ""
    turn_ids: list[int] = field(default_factory=list)
    left_context_exchange_ids: list[str] = field(default_factory=list)
    right_context_exchange_ids: list[str] = field(default_factory=list)
    noise_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LibraryRecord:
    unit_id: str
    unit_type: str
    canonical_name: str
    definition: str = ""
    summary: str = ""
    topic_path: list[str] = field(default_factory=list)
    source_hierarchy_path: list[str] = field(default_factory=list)
    criteria: list[dict[str, Any]] = field(default_factory=list)
    source_packet_hash: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def searchable_text(self) -> str:
        parts = [
            self.canonical_name,
            " > ".join(self.topic_path),
            " > ".join(self.source_hierarchy_path),
            self.definition,
            self.summary,
        ]
        return "\n".join(p for p in parts if p)


@dataclass
class CandidateHit:
    unit_id: str
    unit_type: str
    canonical_name: str
    score_by_query: dict[str, float]
    best_score: float
    local_best_score: float = 0.0
    context_best_score: float = 0.0
    topic_path: list[str] = field(default_factory=list)
    source_hierarchy_path: list[str] = field(default_factory=list)
    source_packet_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExchangeAssignment:
    dialogue_id: str
    exchange_id: str
    assignment_label: str
    dominant_kc_id: str | None
    dominant_kc_name: str | None
    secondary_kc_ids: list[str]
    top1_score: float
    top2_score: float
    margin: float
    assignment_score: float
    decision_reasons: list[str]
    candidate_hits: list[dict[str, Any]]
    review_required: bool = False
    review_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Segment:
    segment_id: str
    dialogue_id: str
    exchange_start: str
    exchange_end: str
    turn_start: int
    turn_end: int
    dominant_kc_id: str | None
    dominant_kc_name: str | None
    secondary_kc_ids: list[str]
    member_exchange_ids: list[str]
    topic_path: list[str]
    boundary_reasons: dict[str, list[str]]
    confidence_summary: dict[str, Any]
    review_required: bool
    review_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
