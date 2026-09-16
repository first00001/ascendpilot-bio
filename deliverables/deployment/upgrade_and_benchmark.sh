#!/usr/bin/env bash
set -euo pipefail

SOURCE=${1:-/tmp/plantcell-upgrade}
PYTHON=/usr/local/python3.12.13/bin/python3
API=http://127.0.0.1:8000
STAMP=$(date +%Y%m%d-%H%M%S)

$PYTHON -m py_compile "$SOURCE/deployment/qwen35_server.py"
install -d -m 0755 /opt/plantcell/demo /opt/plantcell/plantcell-skill \
  /opt/plantcell/results/benchmarks
cp -a /opt/qwen35_server.py "/opt/qwen35_server.py.pre-${STAMP}"
install -m 0644 "$SOURCE/deployment/qwen35_server.py" /opt/qwen35_server.py
install -m 0644 "$SOURCE/demo/index.html" /opt/plantcell/demo/index.html
install -m 0644 "$SOURCE/data/ZH11_riceFM_eval/demo_samples.json" \
  /opt/plantcell/data/ZH11_riceFM_eval/demo_samples.json
install -m 0644 "$SOURCE/benchmark.py" /opt/plantcell/benchmark.py
install -m 0644 "$SOURCE/scripts/benchmark_qwen_dtype.py" \
  /opt/plantcell/benchmark_qwen_dtype.py
cp -a "$SOURCE/plantcell-skill/." /opt/plantcell/plantcell-skill/
chmod 0755 /opt/plantcell/plantcell-skill/scripts/*.sh

supervisorctl restart qwen35
ready=0
for _ in $(seq 1 120); do
  if curl -fsS "$API/health" >/dev/null; then ready=1; break; fi
  sleep 2
done
if [[ "$ready" != 1 ]]; then
  tail -100 /var/log/qwen35/stderr.log || true
  exit 1
fi

set -a
source /etc/qwen35.env
set +a
export PLANTCELL_API_TOKEN=$QWEN_API_TOKEN

echo "== PUBLIC ENDPOINTS =="
curl -fsS "$API/health"
echo
curl -fsS "$API/version"
echo
test "$(curl -sS -o /dev/null -w '%{http_code}' "$API/demo")" = 200
test "$(curl -sS -o /dev/null -w '%{http_code}' "$API/ricefm/umap")" = 401

echo "== AUTHENTICATED DATA =="
curl -fsS -H "x-api-key: $QWEN_API_TOKEN" "$API/ricefm/umap?limit=100" \
  | $PYTHON -c 'import json,sys; x=json.load(sys.stdin); assert len(x["points"])==100; print({"umap_points":len(x["points"]),"total":x["total_cells"]})'

echo "== REAL RICEFM CELL =="
$PYTHON - <<'PY' | curl -fsS --max-time 600 \
  -H "x-api-key: $QWEN_API_TOKEN" -H 'content-type: application/json' \
  --data-binary @- "$API/ricefm/annotate" | tee "/opt/plantcell/results/benchmarks/annotate-${STAMP}.json"
import json
p = "/opt/plantcell/data/ZH11_riceFM_eval/demo_samples.json"
s = json.load(open(p, encoding="utf-8"))["samples"][0]
print(json.dumps({"genes": s["genes"], "counts": s["counts"], "k": 5}))
PY
echo

echo "== QWEN GENERATION =="
curl -fsS --max-time 600 -H "x-api-key: $QWEN_API_TOKEN" \
  -H 'content-type: application/json' \
  -d '{"prompt":"请用一句话说明ZH11标签迁移的主要限制。","max_new_tokens":64}' \
  "$API/chat" | tee "/opt/plantcell/results/benchmarks/chat-${STAMP}.json"
echo

npu-smi info > "/opt/plantcell/results/benchmarks/npu-smi-${STAMP}.txt" || true
$PYTHON /opt/plantcell/benchmark.py --mode ricefm --runs 5 --warmup 1 \
  --output "/opt/plantcell/results/benchmarks/ricefm-${STAMP}.json"
$PYTHON /opt/plantcell/benchmark.py --mode chat --runs 3 --warmup 1 \
  --max-new-tokens 64 --output "/opt/plantcell/results/benchmarks/chat-benchmark-${STAMP}.json"
$PYTHON /opt/plantcell/benchmark_qwen_dtype.py --runs 2 --max-new-tokens 64 \
  --output "/opt/plantcell/results/benchmarks/qwen-dtype-${STAMP}.json"
supervisorctl status qwen35
echo "BENCHMARK_STAMP=$STAMP"
