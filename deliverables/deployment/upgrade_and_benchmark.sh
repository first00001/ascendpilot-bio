#!/usr/bin/env bash
set -euo pipefail

SOURCE=${1:-/tmp/plantcell-public}
PYTHON=${PYTHON:-/usr/local/python3.12.13/bin/python}
API=${API:-http://127.0.0.1:8000}
STAMP=$(date +%Y%m%d-%H%M%S)

bash "$SOURCE/deployment/deploy_and_verify.sh" "$SOURCE"

set -a
source /etc/qwen35.env
set +a
export PLANTCELL_API_TOKEN=$QWEN_API_TOKEN

install -d -m 0755 /opt/plantcell/results/benchmarks
if [[ -f "$SOURCE/benchmark.py" ]]; then
  install -m 0644 "$SOURCE/benchmark.py" /opt/plantcell/benchmark.py
  $PYTHON /opt/plantcell/benchmark.py --mode chat --runs 3 --warmup 1 \
    --max-new-tokens 64 \
    --output "/opt/plantcell/results/benchmarks/public-chat-$STAMP.json"
fi

curl -fsS --max-time 300 \
  -H "x-api-key: $QWEN_API_TOKEN" \
  -H 'content-type: application/json' \
  -d '{"prompt":"简要说明 E-ENAD-52 公开证据、Qwen 实时执行和 GSE232863 riceFM 25 基因 embedding pilot 的边界。","max_new_tokens":96}' \
  "$API/chat" | tee "/opt/plantcell/results/benchmarks/public-chat-smoke-$STAMP.json"
echo

npu-smi info > "/opt/plantcell/results/benchmarks/npu-smi-$STAMP.txt" || true
supervisorctl status qwen35
echo "BENCHMARK_STAMP=$STAMP"
