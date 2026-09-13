"""Rank-fusion helpers for v21.

Design decisions recorded in v19_architecture_research_report.md:
- RRF uses source-family rank positions, not raw score arithmetic.
- The default `k` is not the literature default 60. It is a data-derived
  v19 constant chosen from this repo's own rank-depth distributions.
- For the current 63-exchange, 144-profile setting, the chosen value is 11.

v21 adds deterministic tie-breaking so exact RRF ties are reproducible across
Python processes regardless of hash seed.

The helper intentionally stays generic and domain-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


RRF_K_V21 = 11
FAMILY_SUPPORT_RANK_CUTOFF_V21 = 25


@dataclass(frozen=True)
class FusedRank:
    unit_id: str
    rank: int
    rrf_score: float
    contributing_sources: tuple[str, ...]
    source_count: int
    best_source_rank: int | None
    source_ranks: dict[str, int]


def reciprocal_rank(rank: int, k: int = RRF_K_V21) -> float:
    return 1.0 / float(k + rank)


def reciprocal_rank_fuse(
    source_rank_maps: dict[str, dict[str, int]],
    universe_ids: Iterable[str] | None = None,
    k: int = RRF_K_V21,
) -> list[FusedRank]:
    if universe_ids is None:
        all_ids: set[str] = set()
        for rank_map in source_rank_maps.values():
            all_ids.update(rank_map)
    else:
        all_ids = set(universe_ids)

    fused: list[FusedRank] = []
    for uid in all_ids:
        source_ranks = {
            source_name: rank
            for source_name, rank_map in source_rank_maps.items()
            if (rank := rank_map.get(uid)) is not None
        }
        if not source_ranks:
            continue

        rrf_score = sum(reciprocal_rank(rank, k=k) for rank in source_ranks.values())
        contributing = tuple(sorted(source_ranks, key=lambda name: source_ranks[name]))
        fused.append(
            FusedRank(
                unit_id=uid,
                rank=0,
                rrf_score=rrf_score,
                contributing_sources=contributing,
                source_count=len(source_ranks),
                best_source_rank=min(source_ranks.values()) if source_ranks else None,
                source_ranks=source_ranks,
            )
        )

    fused.sort(
        key=lambda row: (
            -row.rrf_score,
            -row.source_count,
            row.best_source_rank or 10**9,
            row.unit_id,
        )
    )

    out: list[FusedRank] = []
    for idx, row in enumerate(fused, start=1):
        out.append(
            FusedRank(
                unit_id=row.unit_id,
                rank=idx,
                rrf_score=row.rrf_score,
                contributing_sources=row.contributing_sources,
                source_count=row.source_count,
                best_source_rank=row.best_source_rank,
                source_ranks=row.source_ranks,
            )
        )
    return out


def fused_rank_map(
    source_rank_maps: dict[str, dict[str, int]],
    universe_ids: Iterable[str] | None = None,
    k: int = RRF_K_V21,
) -> dict[str, FusedRank]:
    return {row.unit_id: row for row in reciprocal_rank_fuse(source_rank_maps, universe_ids=universe_ids, k=k)}
