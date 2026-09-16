#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/qwen35_08b
pid="$(cat "$ROOT/triton_pilot.pid" 2>/dev/null || true)"
if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
  echo "pilot=running pid=$pid"
  ps -p "$pid" -o pid,etime,stat,cmd
else
  echo "pilot=stopped pid=${pid:-missing}"
fi
echo "--- metrics/errors ---"
grep -E -i 'iteration +[0-9]+/|Loaded checkpoint|Saved checkpoint|traceback|error|exception|failed' "$ROOT/logs/triton_pilot/train.log" | tail -70 || true
echo "--- tail ---"
tail -60 "$ROOT/logs/triton_pilot/train.log" || true
echo "--- checkpoint ---"
find "$ROOT/save_triton_pilot" -maxdepth 3 -type f -printf '%p %s bytes\n' 2>/dev/null | head -30
echo "--- AscendC imports ---"
sed -n '565,625p' /opt/MindSpeed-MM/mindspeed_mm/fsdp/models/qwen3_5/modeling_qwen3_5.py
grep -R -n 'fla_npu\|flash.linear.attention.npu' /opt/MindSpeed-MM/examples/qwen3_5 /opt/MindSpeed-MM/mindspeed_mm/fsdp/ops/gdn | head -70 || true
