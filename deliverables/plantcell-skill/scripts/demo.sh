#!/usr/bin/env bash
set -euo pipefail
API=${PLANTCELL_ENDPOINT:-http://127.0.0.1:8000}
if [[ -r /etc/qwen35.env ]]; then set -a; source /etc/qwen35.env; set +a; fi
TOKEN=${PLANTCELL_API_TOKEN:-${QWEN_API_TOKEN:-}}
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
curl -fsS "$API/ricefm/demo-samples" > "$tmp"
python3 - "$tmp" <<'PY' | curl -fsS --max-time 600 \
  -H 'content-type: application/json' \
  --data-binary @- "$API/ricefm/annotate" | python3 -m json.tool
import json, sys
sample = json.load(open(sys.argv[1], encoding="utf-8"))["samples"][0]
print(json.dumps({"genes": sample["genes"], "counts": sample["counts"], "k": 5}))
PY
