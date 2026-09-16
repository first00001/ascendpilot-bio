#!/usr/bin/env bash
set -euo pipefail

pid="$(cat /opt/qwen35_08b/smoke.pid 2>/dev/null || true)"
if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
  echo "smoke=running pid=$pid"
  ps -p "$pid" -o pid,etime,stat,cmd
else
  echo "smoke=stopped pid=${pid:-missing}"
fi
echo "--- launcher ---"
tail -160 /opt/qwen35_08b/logs/smoke_launcher.log 2>/dev/null || true
echo "--- training ---"
latest="$(find /opt/qwen35_08b/logs/smoke -type f -name 'train_*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
if [[ -n "$latest" ]]; then
  echo "log=$latest"
  tail -220 "$latest"
fi
echo "--- NPU processes ---"
npu-smi info | tail -20
