# API Contract

Base URL defaults to `http://127.0.0.1:8000`. `/health`, `/version`, `/docs`, and
`/demo` are public. All data and inference endpoints require either
`x-api-key: <token>` or `Authorization: Bearer <token>`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Artifact and device readiness |
| GET | `/version` | Immutable service/model identity |
| GET | `/skills` | Available analysis capabilities |
| GET | `/ricefm/metrics` | Recorded ZH11 transfer metrics |
| GET | `/ricefm/umap` | Deterministically sampled ZH11 UMAP points |
| GET | `/ricefm/demo-samples` | Four compact real-cell demo inputs |
| POST | `/ricefm/annotate` | riceFM embedding and weighted 5-NN transfer |
| POST | `/chat` | Qwen3.5 answer with token and generation timing |
| POST | `/agent/run` | Qwen JSON planner, whitelisted tools, evidence store and grounded report |
| POST | `/analyze/plan` | Deterministic analysis routing |
| POST | `/analyze/report` | Evidence-bounded Qwen report |

The annotation endpoint accepts at most 128 cells per request and `k` from 1 to 25.
Counts must be finite, non-negative, and aligned to the supplied gene list. Uploads
accept CSV/TSV/TXT up to 20 MiB.
