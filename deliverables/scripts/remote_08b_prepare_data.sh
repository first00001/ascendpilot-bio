#!/usr/bin/env bash
set -euo pipefail

mkdir -p /opt/qwen35_08b/logs
if [[ -f /opt/qwen35_08b/data.pid ]] && kill -0 "$(cat /opt/qwen35_08b/data.pid)" 2>/dev/null; then
  echo "data preparation already running: pid=$(cat /opt/qwen35_08b/data.pid)"
  exit 0
fi
nohup bash -lc '
  set -e
  /usr/local/python3.12.13/bin/python3 -m pip install modelscope==1.38.1
  bash /opt/qwen35_08b/tooling/scripts/prepare_data.sh
' > /opt/qwen35_08b/logs/prepare_data.log 2>&1 < /dev/null &
echo "$!" > /opt/qwen35_08b/data.pid
echo "data_pid=$!"
