#!/usr/bin/env bash
set -euo pipefail

mkdir -p /opt/qwen35_08b/logs
if [[ -f /opt/qwen35_08b/convert.pid ]] && kill -0 "$(cat /opt/qwen35_08b/convert.pid)" 2>/dev/null; then
  echo "conversion already running: pid=$(cat /opt/qwen35_08b/convert.pid)"
  exit 0
fi
rm -rf /opt/qwen35_08b/dcp
nohup bash -lc '
  set -e
  source /usr/local/Ascend/cann/set_env.sh
  cd /opt/MindSpeed-MM
  mm-convert Qwen35Converter hf_to_dcp \
    --hf_dir /workspace/shared_assets/HC2026/helloworld/model/Qwen3.5-0.8B \
    --dcp_dir /opt/qwen35_08b/dcp \
    --num_workers 0
' > /opt/qwen35_08b/logs/convert.log 2>&1 < /dev/null &
echo "$!" > /opt/qwen35_08b/convert.pid
echo "conversion_pid=$!"
