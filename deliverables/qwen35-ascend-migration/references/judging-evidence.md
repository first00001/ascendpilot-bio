# Judging Evidence Map

| Competition concern | Required artifact |
|---|---|
| Automatic migration | Environment report, generated policy and one-command deployment log |
| Ascend adaptation | Device mapping, torch-npu/CANN versions, NPU smoke output |
| Optimization | Baseline/candidate table with latency, throughput, memory and negative results |
| Accuracy | Fixed prompt set, deterministic settings, grounding violations and raw responses |
| Performance | Warm-up policy, request count, P50/P95, errors, concurrency and NPU snapshot |
| Runnable Skill | `SKILL.md`, scripts, schemas, health check and rollback path |
| Application value | PlantCell demo, real ZH11 input, riceFM result and evidence neighbors |
| Agent intelligence | JSON planner, tool whitelist, evidence IDs and grounded report |
| Reproducibility | Model/checkpoint/data hashes, timestamps, commands and immutable JSON |

Recommended presentation order: migration problem, automatic workflow, measured
policy selection, Ascend deployment proof, performance/quality evidence, then the
PlantCell scenario. Do not open with domain background for more than one slide.
