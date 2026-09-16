---
name: plantcell-ascend-analysis
description: Analyze rice single-cell expression data with Qwen3.5 orchestration and riceFM embeddings on Ascend NPU. Use for ZH11 quality review, reference-label transfer, evidence inspection, and bounded scientific report generation.
---

# PlantCell Ascend Analysis

Use this Skill for rice single-cell analysis requests. Qwen3.5 plans the work and
explains results; riceFM generates cell embeddings; ZH11 supplies the labeled
reference. Keep measured facts, predictions, and biological interpretation visibly
separate.

## Workflow

1. Check `/health` and `/version`. Stop if an artifact is missing or either model is
   not assigned to its expected device (`Qwen3.5: npu:0`, `riceFM: npu:1`).
2. For a natural-language request, call `/analyze/plan` and show the selected steps.
3. For expression-based annotation, accept finite non-negative counts whose width
   exactly matches `genes`, then call `/ricefm/annotate`. Treat the returned label as
   a reference-transfer prediction, not a ground-truth diagnosis.
4. Show confidence and nearest labeled ZH11 neighbors with every annotation.
5. Use `/analyze/report` only after structured results exist. Do not invent marker,
   differential-expression, pathway, or causal conclusions that were not computed.
6. State the evaluation boundary: the riceFM checkpoint may have seen overlapping
   ZH11 profiles during pretraining, so the recorded score is not an independent
   external validation result.

Read [references/API.md](references/API.md) when constructing requests. Validate
payloads against [schemas/annotation-request.json](schemas/annotation-request.json)
and [schemas/annotation-response.json](schemas/annotation-response.json) when
integrating another client.

## Operations

Run `scripts/healthcheck.sh` before a demonstration. Use `scripts/demo.sh` for a
deterministic, real-cell smoke test. Installation and service lifecycle are handled
by `scripts/install.sh`, `scripts/start.sh`, `scripts/stop.sh`, and
`scripts/status.sh`.

