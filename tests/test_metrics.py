from seg_eval.metrics import boundary_precision_recall_f1, windowdiff


def test_boundary_metrics():
    pred = [0, 1, 0, 1]
    gold = [0, 1, 1, 0]
    m = boundary_precision_recall_f1(pred, gold)
    assert m["tp"] == 1
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert 0.0 <= windowdiff(pred, gold) <= 1.0
