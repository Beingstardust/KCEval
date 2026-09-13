"""Deterministic scoring and aggregation for the segment/episode/macro tutor evaluator.

Nothing in this package calls a model. Everything here consumes judge-produced judgments
(ratings, applicability, trust flags, macro dimension scores) and computes scores from them with
pure, tested arithmetic — so a reported number can always be traced back to a specific formula
and a specific set of judge outputs, not re-derived by asking a model to also do the arithmetic.

See ``kc_criterion_scoring`` for signed-weight KC-specific criterion scoring, and
``deterministic_aggregation`` for the segment -> episode -> dialogue -> tutor-score chain.
"""
