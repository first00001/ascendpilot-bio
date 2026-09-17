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
curl -fsS "$API/health" | python3 -c '
import json, sys
x = json.load(sys.stdin)
assert x["status"] == "ok"
assert x["ricefm_runtime"] == "public-anchor-pilot-ready"
assert x["artifacts"]["ricefm_public_pilot_input"] is True
assert x["artifacts"]["ricefm_public_pilot_result"] is True
'
curl -fsS "$API/ricefm/metrics" | python3 -c '
import json, sys
x = json.load(sys.stdin)
assert x["ricefm_execution"] == "public-anchor-pilot-on-ascend"
assert x["ricefm_pilot"]["embedding_shape"] == [64, 256]
print(json.dumps({"ricefm_execution": x["ricefm_execution"], "cells_per_second": x["cells_per_second"]}))
'
