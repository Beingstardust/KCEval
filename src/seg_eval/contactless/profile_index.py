from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from seg_eval.matching_profiles import load_matching_profiles


def _field_values(profile: dict[str, Any], field: str) -> list[str]:
    if field == "definition_or_summary":
        value = profile.get("definition_or_summary")
    elif field == "leaf_topic":
        value = (profile.get("context_profile") or {}).get("leaf_topic")
    elif field in {
        "canonical_terms",
        "matching_cues",
        "likely_dialogue_surface_forms",
        "formula_or_symbol_forms",
        "matched_surface_terms_from_evidence",
        "child_surface_terms",
    }:
        value = (profile.get("positive_profile") or {}).get(field)
    else:
        value = (profile.get("negative_profile") or {}).get(field)

    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


def branch_signature(topic_path: list[str], depth: int = 3) -> str:
    if not topic_path:
        return ""
    return " > ".join(topic_path[: min(depth, len(topic_path))])


def leaf_topic(topic_path: list[str]) -> str:
    return topic_path[-1] if topic_path else ""


@dataclass
class LoadedProfiles:
    path: str
    profiles: list[dict[str, Any]]
    by_id: dict[str, dict[str, Any]]
    branch_by_id: dict[str, str]
    leaf_by_id: dict[str, str]
    profile_token_df: Counter
    profile_vocab: set[str]

    @classmethod
    def from_path(cls, path: str):
        profiles = load_matching_profiles(path)
        by_id = {p["unit_id"]: p for p in profiles}
        branch_by_id = {p["unit_id"]: branch_signature(p.get("topic_path") or []) for p in profiles}
        leaf_by_id = {p["unit_id"]: leaf_topic(p.get("topic_path") or []) for p in profiles}

        profile_token_df: Counter = Counter()
        vocab: set[str] = set()

        for profile in profiles:
            tokens_seen: set[str] = set()
            text_parts: list[str] = [str(profile.get("canonical_name") or "")]
            text_parts.extend(profile.get("topic_path") or [])
            for field in [
                "canonical_terms",
                "matching_cues",
                "likely_dialogue_surface_forms",
                "formula_or_symbol_forms",
                "matched_surface_terms_from_evidence",
                "child_surface_terms",
                "definition_or_summary",
                "leaf_topic",
            ]:
                text_parts.extend(_field_values(profile, field))

            for raw in text_parts:
                for tok in str(raw).lower().replace("-", " ").replace("_", " ").split():
                    tok = "".join(ch for ch in tok if ch.isalnum())
                    if len(tok) >= 3:
                        tokens_seen.add(tok)
                        vocab.add(tok)

            profile_token_df.update(tokens_seen)

        return cls(
            path=str(path),
            profiles=profiles,
            by_id=by_id,
            branch_by_id=branch_by_id,
            leaf_by_id=leaf_by_id,
            profile_token_df=profile_token_df,
            profile_vocab=vocab,
        )

    def profile(self, unit_id: str) -> dict[str, Any] | None:
        return self.by_id.get(unit_id)

    def branch(self, unit_id: str | None) -> str:
        return self.branch_by_id.get(unit_id or "", "")

    def leaf(self, unit_id: str | None) -> str:
        return self.leaf_by_id.get(unit_id or "", "")
