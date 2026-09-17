# API Contract

Base URL defaults to `http://127.0.0.1:8000`. Health, version, demo, Skill,
migration-report, and `/ricefm/*` public-pilot endpoints are public. Qwen report,
chat, upload, and analysis endpoints require either `x-api-key: <token>` or
`Authorization: Bearer <token>`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Artifact and device readiness |
| GET | `/version` | Immutable service/model identity |
| GET | `/skills` | Available analysis capabilities |
| GET | `/ricefm/metrics` | Public Atlas evidence and riceFM pilot execution record |
| GET | `/ricefm/umap` | Deterministically sampled public E-ENAD-52 UMAP points |
| GET | `/ricefm/demo-samples` | Public GSE232863 25-gene anchor inputs |
| POST | `/ricefm/annotate` | Live 256-dimensional riceFM embedding on Ascend |
| POST | `/chat` | Qwen3.5 answer with token and generation timing |
| POST | `/agent/run` | Qwen JSON planner, whitelisted tools, evidence store and grounded report |
| POST | `/analyze/plan` | Deterministic analysis routing |
| POST | `/analyze/report` | Evidence-bounded Qwen report |

The embedding endpoint accepts at most 256 cells per request and `k` from 1 to 25.
Counts must be finite, non-negative, aligned to the supplied gene list, and limited
to the validated 25 anchor IDs. The response has no classification head or
cell-type label. Uploads accept CSV/TSV/TXT up to 20 MiB.
