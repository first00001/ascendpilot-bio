#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-/usr/local/python3.12.13/bin/python3}
MODEL=${QWEN_MODEL:-/workspace/shared_assets/models/Qwen/Qwen3.5-4B}
RESULTS=${MIGRATION_RESULTS:-/opt/plantcell/results/migration}
mkdir -p "$RESULTS"

$PYTHON "$ROOT/qwen35-ascend-migration/scripts/inspect_environment.py" \
  --model "$MODEL" --output "$RESULTS/environment.json"
$PYTHON "$ROOT/scripts/benchmark_qwen_dtype.py" --model "$MODEL" --device npu:1 \
  --runs 3 --max-new-tokens 128 --output "$RESULTS/precision-benchmark.json"
$PYTHON "$ROOT/qwen35-ascend-migration/scripts/select_policy.py" \
  --benchmark "$RESULTS/precision-benchmark.json" --max-memory-gb 12 \
  --objective balanced --output "$RESULTS/policy.json"
bash "$ROOT/plantcell-skill/scripts/install.sh"
bash "$ROOT/plantcell-skill/scripts/healthcheck.sh"
$PYTHON "$ROOT/scripts/evaluate_qwen_grounding.py" --model "$MODEL" --device npu:1 \
  --output "$RESULTS/grounding-ab.json"
echo "Migration evidence written to $RESULTS"
