#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/qwen35_08b
BASE="$ROOT/tooling/configs/qwen3_5_0.8B_config.yaml"
CONFIG="$ROOT/qwen3_5_0.8B_triton_pilot.yaml"
LOG_DIR="$ROOT/logs/triton_pilot"
mkdir -p "$LOG_DIR" "$ROOT/save_triton_pilot"

test -s "$ROOT/data/output_llava_coco_data.json"
test -f "$ROOT/dcp/release/.metadata"

/usr/local/python3.12.13/bin/python3 - "$BASE" "$CONFIG" <<'PY'
from pathlib import Path
import sys

source, destination = map(Path, sys.argv[1:])
text = source.read_text(encoding="utf-8")
replacements = {
    "      max_samples: null": "      max_samples: 64",
    "  gdn_implementation: ascendc": "  gdn_implementation: triton",
    "    num_workers: 8": "    num_workers: 4",
    "  micro_batch_size: 4": "  micro_batch_size: 1",
    "  train_iters: 100": "  train_iters: 5",
    "  save_interval: 100": "  save_interval: 5",
    "  save: /opt/qwen35_08b/save": "  save: /opt/qwen35_08b/save_triton_pilot",
}
for old, new in replacements.items():
    if old not in text:
        raise SystemExit(f"missing expected config line: {old}")
    text = text.replace(old, new, 1)
destination.write_text(text, encoding="utf-8")
PY

if [[ -f "$ROOT/triton_pilot.pid" ]] && kill -0 "$(cat "$ROOT/triton_pilot.pid")" 2>/dev/null; then
  echo "triton pilot already running: pid=$(cat "$ROOT/triton_pilot.pid")"
  exit 0
fi

nohup bash -lc "
  set -e
  source /usr/local/Ascend/cann/set_env.sh
  cd /opt/MindSpeed-MM
  export ASCEND_RT_VISIBLE_DEVICES=1
  export NON_MEGATRON=true
  export MULTI_STREAM_MEMORY_REUSE=2
  export TASK_QUEUE_ENABLE=2
  export ASCEND_LAUNCH_BLOCKING=0
  export ACLNN_CACHE_LIMIT=100000
  export CPU_AFFINITY_CONF=1
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=/opt/MindSpeed-MM/triton_cache
  torchrun --nproc_per_node 1 --master_addr localhost --master_port 6002 \
    mindspeed_mm/fsdp/train/trainer.py '$CONFIG'
" > "$LOG_DIR/train.log" 2>&1 < /dev/null &
echo "$!" > "$ROOT/triton_pilot.pid"
echo "triton_pilot_pid=$!"
