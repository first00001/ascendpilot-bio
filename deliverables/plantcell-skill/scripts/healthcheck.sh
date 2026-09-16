#!/usr/bin/env bash
set -euo pipefail
API=${PLANTCELL_ENDPOINT:-http://127.0.0.1:8000}
TOKEN=${PLANTCELL_API_TOKEN:-}
if [[ -z "$TOKEN" && -r /etc/qwen35.env ]]; then
  set -a
  source /etc/qwen35.env
  set +a
  TOKEN=${QWEN_API_TOKEN:-}
fi
curl -fsS "$API/health" | python3 -m json.tool
curl -fsS "$API/version" | python3 -m json.tool
test "$(curl -sS -o /dev/null -w '%{http_code}' "$API/ricefm/metrics")" = 401
curl -fsS -H "x-api-key: $TOKEN" "$API/ricefm/metrics" | python3 -m json.tool

