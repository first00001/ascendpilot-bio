#!/usr/bin/env bash
set -euo pipefail

SOURCE=${1:-/tmp/plantcell-public}
PYTHON=${PYTHON:-/usr/local/python3.12.13/bin/python}
API=${API:-http://127.0.0.1:8000}
STAMP=$(date +%Y%m%d-%H%M%S)

SERVER="$SOURCE/deployment/qwen35_server.py"
DEMO="$SOURCE/demo/index.html"
PUBLIC_JSON="$SOURCE/public_data/e_enad_52_review_subset.json"

for file in "$SERVER" "$DEMO" "$PUBLIC_JSON"; do
  test -f "$file" || { echo "missing deployment file: $file" >&2; exit 1; }
done

$PYTHON -m py_compile "$SERVER"
$PYTHON - <<PY
import json
p = json.load(open("$PUBLIC_JSON", encoding="utf-8"))
assert p["dataset"]["atlas_accession"] == "E-ENAD-52"
assert p["total_cells"] == 28856
assert p["dataset"]["cluster_count"] == 28
assert len(p["points"]) == 1120
PY

BACKUP="/opt/plantcell/backups/$STAMP-pre-public-evidence"
install -d -m 0755 "$BACKUP" /opt/plantcell/demo /opt/plantcell/data/E-ENAD-52
cp -a /opt/qwen35_server.py "$BACKUP/" 2>/dev/null || true
cp -a /opt/plantcell/demo/index.html "$BACKUP/" 2>/dev/null || true

install -m 0644 "$PUBLIC_JSON" \
  /opt/plantcell/data/E-ENAD-52/e_enad_52_review_subset.json
install -m 0644 "$DEMO" /opt/plantcell/demo/index.html
install -m 0644 "$SERVER" /opt/qwen35_server.py
supervisorctl restart qwen35

ready=0
for _ in $(seq 1 120); do
  if curl -fsS "$API/health" >/dev/null; then ready=1; break; fi
  sleep 2
done
if [[ "$ready" != 1 ]]; then
  tail -100 /var/log/qwen35/qwen35.err.log || true
  exit 1
fi

curl -fsS "$API/health" | $PYTHON -c '
import json, sys
x = json.load(sys.stdin)
assert x["status"] == "ok"
assert x["mode"] == "public-evidence"
assert x["qwen_device"] == "npu:0"
assert x["ricefm_runtime"] == "disabled-no-validated-public-gene-map"
print(x)
'

curl -fsS "$API/ricefm/metrics" | $PYTHON -c '
import json, sys
x = json.load(sys.stdin)
assert x["dataset"] == "E-ENAD-52"
assert x["total_cells"] == 28856
assert x["cluster_count"] == 28
print({"dataset": x["dataset"], "cells": x["total_cells"], "clusters": x["cluster_count"]})
'

curl -fsS -H 'content-type: application/json' \
  -d '{"genes":["ATLAS_CLUSTER_1"],"counts":[[1.0]],"k":5}' \
  "$API/ricefm/annotate" | $PYTHON -c '
import json, sys
x = json.load(sys.stdin)
assert x["ricefm_executed"] is False
assert x["reference"] == "E-ENAD-52"
print({"cluster": x["predictions"][0]["celltype_L1"], "ricefm_executed": False})
'

set -a
source /etc/qwen35.env
set +a
test -n "${QWEN_API_TOKEN:-}"

curl -fsS --max-time 300 \
  -H "x-api-key: $QWEN_API_TOKEN" \
  -H 'content-type: application/json' \
  -d '{"query":"解释 E-ENAD-52 cluster、marker 和模型执行边界。"}' \
  "$API/agent/run" | $PYTHON -c '
import json, sys
x = json.load(sys.stdin)
assert x["status"] == "completed"
assert x["verification"]["passed"] is True
assert {"E1", "E2", "E3"}.issubset(x["verification"]["cited_evidence"])
assert x["report_generation"]["device"] == "npu:0"
print({"status": x["status"], "device": x["report_generation"]["device"], "citations": x["verification"]["cited_evidence"]})
'

supervisorctl status qwen35
echo "PUBLIC_DEPLOY_BACKUP=$BACKUP"
