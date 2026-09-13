from __future__ import annotations

from .schema import Turn, Exchange


def _role_pattern(turns: list[Turn]) -> str:
    return "".join("S" if t.role == "student" else "T" if t.role == "tutor" else "U" for t in turns)


def _exchange_text(turns: list[Turn]) -> str:
    parts = []
    for t in turns:
        label = "Student" if t.role == "student" else "Tutor" if t.role == "tutor" else t.role.title()
        if t.norm_text:
            parts.append(f"{label}: {t.norm_text}")
    return "\n".join(parts)


def build_exchanges(turns: list[Turn], left_context: int = 1, right_context: int = 1) -> list[Exchange]:
    """
    Build non-overlapping exchange units.

    Rule priority:
    1. TST: tutor prompt, student attempt, tutor feedback
    2. ST: student query/attempt, tutor answer/feedback
    3. single orphan turn
    """
    exchanges: list[Exchange] = []
    i = 0
    ex_idx = 0
    while i < len(turns):
        chosen: list[Turn]

        if i + 2 < len(turns) and turns[i].role == "tutor" and turns[i+1].role == "student" and turns[i+2].role == "tutor":
            chosen = [turns[i], turns[i+1], turns[i+2]]
            i += 3
        elif i + 1 < len(turns) and turns[i].role == "student" and turns[i+1].role == "tutor":
            chosen = [turns[i], turns[i+1]]
            i += 2
        else:
            chosen = [turns[i]]
            i += 1

        student_text = "\n".join(t.norm_text for t in chosen if t.role == "student" and t.norm_text)
        tutor_text = "\n".join(t.norm_text for t in chosen if t.role == "tutor" and t.norm_text)
        flags = sorted({flag for t in chosen for flag in t.noise_flags})

        ex = Exchange(
            dialogue_id=chosen[0].dialogue_id,
            exchange_id=f"{chosen[0].dialogue_id}::ex_{ex_idx:04d}",
            exchange_index=ex_idx,
            turn_start=chosen[0].turn_id,
            turn_end=chosen[-1].turn_id,
            roles_pattern=_role_pattern(chosen),
            exchange_text=_exchange_text(chosen),
            student_text=student_text,
            tutor_text=tutor_text,
            turn_ids=[t.turn_id for t in chosen],
            noise_flags=flags,
        )
        exchanges.append(ex)
        ex_idx += 1

    for idx, ex in enumerate(exchanges):
        lo = max(0, idx - left_context)
        hi = min(len(exchanges), idx + right_context + 1)
        ex.left_context_exchange_ids = [e.exchange_id for e in exchanges[lo:idx]]
        ex.right_context_exchange_ids = [e.exchange_id for e in exchanges[idx+1:hi]]

    return exchanges
