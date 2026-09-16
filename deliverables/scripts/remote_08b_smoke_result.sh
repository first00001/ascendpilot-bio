#!/usr/bin/env bash
set -euo pipefail

pid="$(cat /opt/qwen35_08b/smoke.pid 2>/dev/null || true)"
if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
  echo "smoke=running pid=$pid"
  ps -p "$pid" -o pid,etime,stat,cmd
else
  echo "smoke=stopped pid=${pid:-missing}"
fi
latest="$(find /opt/qwen35_08b/logs/smoke -type f -name 'train_*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
echo "log=${latest:-missing}"
if [[ -n "$latest" ]]; then
  grep -E -i 'iteration|loss|samples/sec|throughput|finished|saving|saved|traceback|error|exception|failed' "$latest" | tail -100 || true
  echo "--- tail ---"
  tail -40 "$latest"
fi
echo "triton_cache=$(du -sh /opt/MindSpeed-MM/triton_cache 2>/dev/null | cut -f1)"
echo "checkpoint_files=$(find /opt/qwen35_08b/save -type f 2>/dev/null | wc -l)"
