#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/qwen35_08b
BASE="$ROOT/tooling/configs/qwen3_5_0.8B_config.yaml"
SMOKE_DIR="$ROOT/data/smoke"
SMOKE_CONFIG="$ROOT/qwen3_5_0.8B_triton_smoke.yaml"
mkdir -p "$SMOKE_DIR" "$ROOT/logs"

cat > "$SMOKE_DIR/smoke.json" <<'JSON'
[
  {"images": [], "messages": [{"role": "user", "content": "Describe one benefit of running AI models on Ascend NPUs."}, {"role": "assistant", "content": "Ascend NPUs provide high-throughput tensor computation with an optimized software stack."}]},
  {"images": [], "messages": [{"role": "user", "content": "What is a reproducible benchmark?"}, {"role": "assistant", "content": "It fixes the environment, data, parameters, warm-up policy, and measurement method."}]},
  {"images": [], "messages": [{"role": "user", "content": "Why compare loss curves?"}, {"role": "assistant", "content": "Loss curves help verify numerical alignment and training stability across implementations."}]},
  {"images": [], "messages": [{"role": "user", "content": "What does FSDP do?"}, {"role": "assistant", "content": "FSDP shards model states across devices to reduce per-device memory usage."}]},
  {"images": [], "messages": [{"role": "user", "content": "Name a useful performance metric."}, {"role": "assistant", "content": "Samples per second measures end-to-end training throughput."}]},
  {"images": [], "messages": [{"role": "user", "content": "Why record software versions?"}, {"role": "assistant", "content": "Version records make experimental results traceable and reproducible."}]},
  {"images": [], "messages": [{"role": "user", "content": "What is an operator optimization?"}, {"role": "assistant", "content": "It improves the implementation of a compute kernel while preserving expected numerical behavior."}]},
  {"images": [], "messages": [{"role": "user", "content": "Why use a smoke test first?"}, {"role": "assistant", "content": "A smoke test quickly validates dependencies, data loading, model loading, forward pass, and backward pass."}]}
]
JSON

/usr/local/python3.12.13/bin/python3 - "$BASE" "$SMOKE_CONFIG" <<'PY'
from pathlib import Path
import sys

source, destination = map(Path, sys.argv[1:])
text = source.read_text(encoding="utf-8")
replacements = {
    "      dataset_dir: /opt/qwen35_08b/data/coco": "      dataset_dir: /opt/qwen35_08b/data/smoke",
    "      dataset: /opt/qwen35_08b/data/output_llava_coco_data.json": "      dataset: /opt/qwen35_08b/data/smoke/smoke.json",
    "      max_samples: null": "      max_samples: 8",
    "  gdn_implementation: ascendc": "  gdn_implementation: triton",
    "    num_workers: 8": "    num_workers: 0",
    "  micro_batch_size: 4": "  micro_batch_size: 1",
    "  train_iters: 100": "  train_iters: 1",
    "  save_interval: 100": "  save_interval: 1",
}
for old, new in replacements.items():
    if old not in text:
        raise SystemExit(f"missing expected config line: {old}")
    text = text.replace(old, new, 1)
destination.write_text(text, encoding="utf-8")
PY

if [[ -f "$ROOT/smoke.pid" ]] && kill -0 "$(cat "$ROOT/smoke.pid")" 2>/dev/null; then
  echo "smoke test already running: pid=$(cat "$ROOT/smoke.pid")"
  exit 0
fi

nohup bash -lc "
  set -e
  source /usr/local/Ascend/cann/set_env.sh
  cd /opt/MindSpeed-MM
  export ASCEND_RT_VISIBLE_DEVICES=1
  NPUS_PER_NODE=1 MASTER_PORT=6001 \
    bash /opt/qwen35_08b/tooling/scripts/run_training.sh \
    '$SMOKE_CONFIG' '$ROOT/logs/smoke'
" > "$ROOT/logs/smoke_launcher.log" 2>&1 < /dev/null &
echo "$!" > "$ROOT/smoke.pid"
echo "smoke_pid=$!"
