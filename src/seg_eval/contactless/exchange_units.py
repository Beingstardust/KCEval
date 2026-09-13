from __future__ import annotations

from seg_eval.schema import Turn, Exchange


def _exchange_text(student: Turn, tutor: Turn) -> str:
    parts = []
    if student.norm_text:
        parts.append(f"Student: {student.norm_text}")
    if tutor.norm_text:
        parts.append(f"Tutor: {tutor.norm_text}")
    return "\n".join(parts)


def build_strict_st_exchanges(turns: list[Turn], left_context: int = 1, right_context: int = 1) -> list[Exchange]:
    """Build deterministic Student->Tutor exchanges.

    Target setting: an LLM tutor responds once to each student query.
    Malformed logs fail loudly instead of being silently repaired.
    """

    if len(turns) % 2 != 0:
        raise ValueError(f"Strict Student->Tutor pairing requires an even number of turns; got {len(turns)}")

    exchanges: list[Exchange] = []

    for i in range(0, len(turns), 2):
        student = turns[i]
        tutor = turns[i + 1]

        if student.role != "student" or tutor.role != "tutor":
            raise ValueError(
                f"Strict S->T violation at turn pair {i}/{i+1}: "
                f"got roles {student.role!r}/{tutor.role!r}, expected 'student'/'tutor'"
            )

        flags = sorted(set(student.noise_flags + tutor.noise_flags))
        idx = len(exchanges)

        exchanges.append(
            Exchange(
                dialogue_id=student.dialogue_id,
                exchange_id=f"{student.dialogue_id}::ex_{idx:04d}",
                exchange_index=idx,
                turn_start=student.turn_id,
                turn_end=tutor.turn_id,
                roles_pattern="ST",
                exchange_text=_exchange_text(student, tutor),
                student_text=student.norm_text,
                tutor_text=tutor.norm_text,
                turn_ids=[student.turn_id, tutor.turn_id],
                noise_flags=flags,
            )
        )

    for idx, exchange in enumerate(exchanges):
        lo = max(0, idx - left_context)
        hi = min(len(exchanges), idx + right_context + 1)
        exchange.left_context_exchange_ids = [e.exchange_id for e in exchanges[lo:idx]]
        exchange.right_context_exchange_ids = [e.exchange_id for e in exchanges[idx + 1:hi]]

    return exchanges
