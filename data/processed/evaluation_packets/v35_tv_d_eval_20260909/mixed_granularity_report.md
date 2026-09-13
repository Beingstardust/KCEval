# Mixed-Granularity Judge Packet Prep v1

- Source run: `data\processed\segmentation_runs\tv_d_20260909\v35_20260909`
- Topic rollup: `data\processed\topic_rollup\v35_tv_d_topic_rollup.json`
- Local packet family: `kc_segment_local_dimensions_v1` with `37` packets
- Topic packet family: `topic_rollup_arc_dimensions_v1` with `25` packets
- Local focus dimensions: `['error_detection', 'error_reasoning', 'error_localisation', 'solution_control', 'actionability_moving_forward', 'local_coherence_relevance', 'clarity_cognitive_load', 'proactive_clarification']`
- Local focus trust flags: `['curriculum_hallucination_present', 'domain_knowledge_hallucination_present']`
- Topic focus dimensions: `['scaffolding_quality', 'student_level_calibration']`
- Topic focus trust flags: `['student_context_hallucination_present']`
- Topic fragmentation-risk units carried forward for review: `0`

## Notes

- Local packets preserve the original v15 segment granularity.
- Topic packets reuse the frozen topic rollup and keep explicit formation/fallback trace fields.
- No gold labels are used in the production packet logic.
