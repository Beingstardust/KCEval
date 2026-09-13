from __future__ import annotations


def boundary_vector_from_segments(num_exchanges: int, segment_ranges: list[tuple[int, int]]) -> list[int]:
    if num_exchanges <= 1:
        return []
    boundaries = [0] * (num_exchanges - 1)
    for lo, hi in segment_ranges:
        if hi < num_exchanges - 1:
            boundaries[hi] = 1
    return boundaries


def boundary_precision_recall_f1(pred: list[int], gold: list[int]) -> dict[str, float]:
    if len(pred) != len(gold):
        raise ValueError("pred and gold boundary vectors must have equal length")
    tp = sum(1 for p, g in zip(pred, gold) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(pred, gold) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(pred, gold) if p == 0 and g == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def windowdiff(pred: list[int], gold: list[int], k: int | None = None) -> float:
    if len(pred) != len(gold):
        raise ValueError("pred and gold boundary vectors must have equal length")
    n = len(gold)
    if n == 0:
        return 0.0
    if k is None:
        num_gold_segments = sum(gold) + 1
        avg_seg_len = (n + 1) / max(num_gold_segments, 1)
        k = max(1, round(avg_seg_len / 2))
    if k >= n:
        k = max(1, n - 1)
    penalties = 0
    total = 0
    for i in range(0, n - k + 1):
        if sum(pred[i:i+k]) != sum(gold[i:i+k]):
            penalties += 1
        total += 1
    return penalties / total if total else 0.0
