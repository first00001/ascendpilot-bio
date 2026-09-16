#!/usr/bin/env bash
set -euo pipefail

for task in data smoke; do
  pid="$(cat "/opt/qwen35_08b/${task}.pid" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "$task=running pid=$pid"
    ps -p "$pid" -o pid,etime,stat,cmd
  else
    echo "$task=stopped pid=${pid:-missing}"
  fi
done
echo "--- data size ---"
du -sh /opt/qwen35_08b/data 2>/dev/null || true
find /opt/qwen35_08b/data -maxdepth 2 -type f -printf '%p %s bytes\n' 2>/dev/null | head -30
echo "--- data log ---"
tail -30 /opt/qwen35_08b/logs/prepare_data.log 2>/dev/null || true
echo "--- smoke launcher ---"
tail -120 /opt/qwen35_08b/logs/smoke_launcher.log 2>/dev/null || true
echo "--- smoke train log ---"
find /opt/qwen35_08b/logs/smoke -type f -name 'train_*.log' -print -exec tail -160 {} \; 2>/dev/null || true
