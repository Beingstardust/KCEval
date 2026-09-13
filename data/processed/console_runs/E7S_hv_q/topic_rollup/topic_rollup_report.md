# Topic Rollup v1 Report

## Purpose
Post-hoc topic-level rollup over the frozen v15 baseline for evaluation-judge dimensions that need multi-turn arcs: `scaffolding_quality`, `student_level_calibration`, and `student_context_hallucination_present` (source of truth: `src/seg_eval/evaluation_judge/rubric_v2.py`).

## Inputs
- Baseline run dir: `R:\Thesis Project\Segmentation and Evaluation\kc_segmentation_kernel_v0_20260601\data\processed\console_runs\E7S_hv_q\segmentation`
- Frozen profiles: `R:\Thesis Project\Segmentation and Evaluation\kc_segmentation_kernel_v0_20260601\data\processed\console_runs\E7S_hv_q\library\frozen_matching_profiles.jsonl`
- This rollup reads already-produced segment output and frozen hierarchy metadata only.
- It does not edit segmentation boundaries, dominant_kc_id selection, or any v9-v18 runtime file.
- The same script can be rerun against a later baseline run dir without logic changes.
- Frozen hierarchy depth check: `29` topic nodes; minimum KC-leaf coverage under any topic node is `4` (`Data Mining > Clustering > Clustering Concepts`), so topic-level rollup remains meaningful.

## Rollup logic
- Primary merge: merge consecutive v15 segments when their dominant KCs share the same immediate parent topic node (`topic_path[-1]`).
- Fallback merge: when adjacent units disagree on parent topic, allow a conservative adjacency fallback only if a short low-confidence unit (<=2 exchanges and <=2 source segments) shares at least two content tokens with a neighboring unit that still has a stable single-KC anchor.
- If both neighbors qualify, fallback picks the stronger neighbor by shared-token count, then stable topic-vote mass, then total topic-vote mass. Topic choice uses runtime topic-vote aggregation from constituent exchanges (high=3, medium=2, low=1 by confidence band on each resolved KC topic), not gold labels.

## Summary stats
- Topic-level unit count: `22`
- Exchange-count distribution: `{1: 11, 2: 5, 3: 3, 4: 2, 6: 1}`
- Exchanges per topic unit: min `1`, median `1.5`, mean `2.00`, max `6`
- Original v15 segments absorbed into larger topic units: `31`
- Original v15 segments staying standalone: `11`
- Units formed purely from pipeline dominant-parent chaining: `22`
- Units formed with adjacency fallback: `0`
- Units showing fragmentation risk: `1`

## Notes
- Frozen v15 segment count in the source run: `42`
- Excluded meta exchanges inherited from v15 baseline: `[]`
- Gold was used only for annotation in the output JSON/report (score fields, gold KC labels, and post-hoc validation fields).
- Showcase output: `R:\Thesis Project\Segmentation and Evaluation\kc_segmentation_kernel_v0_20260601\data\processed\console_runs\E7S_hv_q\topic_rollup\topic_rollup.json`
