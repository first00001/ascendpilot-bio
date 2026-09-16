#!/usr/bin/env bash
set -euo pipefail

PYTHON=/usr/local/python3.12.13/bin/python3
API=http://127.0.0.1:8000

$PYTHON -m py_compile /opt/qwen35_server.py.new
cp -a /opt/qwen35_server.py /opt/qwen35_server.py.pre-zh11
mv /opt/qwen35_server.py.new /opt/qwen35_server.py
supervisorctl restart qwen35

ready=0
for _ in $(seq 1 60); do
  if curl -fsS "$API/health" >/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "$ready" != 1 ]]; then
  tail -100 /var/log/qwen35/stderr.log || true
  exit 1
fi

set -a
source /etc/qwen35.env
set +a

echo HEALTH
curl -fsS "$API/health"
echo
echo VERSION
curl -fsS "$API/version"
echo
echo UNAUTHORIZED_METRICS_HTTP
curl -sS -o /dev/null -w '%{http_code}\n' "$API/ricefm/metrics"
echo AUTHORIZED_METRICS
curl -fsS -H "x-api-key: $QWEN_API_TOKEN" "$API/ricefm/metrics"
echo
echo RICEFM_ANNOTATE_SMOKE
curl -fsS --max-time 600 \
  -H "x-api-key: $QWEN_API_TOKEN" \
  -H "content-type: application/json" \
  -d '{"genes":["ZH01G00010","ZH01G00020","ZH01G00030"],"counts":[[1,0,2]],"k":3}' \
  "$API/ricefm/annotate"
echo

tar -czf /opt/plantcell/results/zh11-results.tar.gz \
  -C /opt/plantcell/results/zh11 .
sha256sum /opt/plantcell/results/zh11-results.tar.gz
supervisorctl status qwen35
