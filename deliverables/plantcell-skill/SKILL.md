---
name: plantcell-ascend-analysis
description: Analyze public rice single-cell evidence with Qwen3.5 orchestration and riceFM embeddings on Ascend NPU. Use for public-data evidence inspection, compatibility-pilot execution, and bounded scientific report generation.
---

# PlantCell Ascend Analysis

Use this Skill for public rice single-cell analysis requests. Qwen3.5 plans the
work and explains results; riceFM generates cell embeddings from a validated
25-gene anchor subset of `GSE232863`; `E-ENAD-52` supplies public UMAP, cluster,
and marker evidence. Keep measured facts, model outputs, and biological
interpretation visibly separate.

## Workflow

1. Check `/health` and `/version`. Stop if an artifact is missing or either model is
   not assigned to its expected device (`Qwen3.5: npu:0`, `riceFM: npu:1`).
2. For a natural-language request, call `/analyze/plan` and show the selected steps.
3. For the riceFM compatibility pilot, accept finite non-negative counts whose
   width exactly matches `genes`. The gene list must use only the validated 25
   riceFM anchor IDs returned by `/ricefm/demo-samples`.
4. Treat `/ricefm/annotate` as embedding execution. Show the 256-dimensional
   output hash, norm, device, and timing; do not present a cell-type prediction,
   confidence score, or nearest labeled reference cell.
5. Use `/analyze/report` only after structured results exist. Do not invent marker,
   differential-expression, pathway, or causal conclusions that were not computed.
6. State the evaluation boundary: the checked-in mapping covers 25 genes, not the
   whole transcriptome, and the checkpoint has no validated classification head
   for this public dataset. The output proves execution compatibility only.

Read [references/API.md](references/API.md) when constructing requests. Validate
payloads against [schemas/annotation-request.json](schemas/annotation-request.json)
and [schemas/annotation-response.json](schemas/annotation-response.json) when
integrating another client.

## Operations

Run `scripts/healthcheck.sh` before a demonstration. Use `scripts/demo.sh` for a
deterministic, real-cell smoke test. Installation and service lifecycle are handled
by `scripts/install.sh`, `scripts/start.sh`, `scripts/stop.sh`, and
`scripts/status.sh`.

For reviewers without the model artifacts or an Ascend environment, run
`python deliverables/offline_demo.py`. This review mode uses a checked-in public
subset of EMBL-EBI `E-ENAD-52` / NCBI `GSE146035`, is visibly labeled in the UI,
and does not claim to execute Qwen, riceFM, or NPU inference.
Atlas cluster IDs are shown as unsupervised clusters, not curated cell types. Use
the real service for performance or private-environment scientific claims.
