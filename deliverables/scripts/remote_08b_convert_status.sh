#!/usr/bin/env bash
set -euo pipefail

pid="$(cat /opt/qwen35_08b/convert.pid 2>/dev/null || true)"
echo "pid=${pid:-missing}"
if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
  ps -p "$pid" -o pid,etime,stat,cmd
else
  echo "conversion_process=stopped"
fi
echo "--- convert.log ---"
tail -120 /opt/qwen35_08b/logs/convert.log 2>/dev/null || true
echo "--- dcp files ---"
find /opt/qwen35_08b/dcp -maxdepth 3 -type f -printf '%p %s bytes\n' 2>/dev/null | head -80
