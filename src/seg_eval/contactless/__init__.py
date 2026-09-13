"""Contactless two-pass KC segmentation package.

This package is the clean assignment/segmentation rewrite:
Pass 0: strict Student->Tutor exchange units
Pass 1: high-recall local candidate pool
Pass 1b: closed-set-first local assignment
Pass 2: context rescue over the exchange sequence
Novelty gate: missing-KC-only review
Segmenter: merge contactless resolved KC continuity
"""
