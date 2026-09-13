from seg_eval.dialogue_parser import parse_dialogue
from seg_eval.exchange_builder import build_exchanges


def test_non_overlapping_exchanges():
    dialogue = {
        "dialogue_id": "d",
        "turns": [
            {"role": "student", "content": "Question one?"},
            {"role": "tutor", "content": "Answer one."},
            {"role": "student", "content": "Question two?"},
            {"role": "tutor", "content": "Answer two."},
        ],
    }
    turns = parse_dialogue(dialogue)
    exchanges = build_exchanges(turns)
    assert len(exchanges) == 2
    used = [tid for ex in exchanges for tid in ex.turn_ids]
    assert used == [0, 1, 2, 3]
    assert len(used) == len(set(used))
