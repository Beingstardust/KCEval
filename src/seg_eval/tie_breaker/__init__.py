"""Generative tie-breaker for topic-vs-mention disambiguation.

Held in reserve throughout this project and invoked only after every
surface-statistics approach was measured and failed on the same cases
(v28 cross-encoder, v29 adapted dense family, v30 ColBERT, v31 funnel,
the real-data fine-tuned head, and contextual query expansion -- see
data/gold/context_expansion_probe_report.md).
"""
