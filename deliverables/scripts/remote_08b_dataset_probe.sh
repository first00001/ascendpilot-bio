#!/usr/bin/env bash
set -euo pipefail

sed -n '1,260p' /opt/MindSpeed-MM/mindspeed_mm/fsdp/tools/data_tool/llava_instruct_2_mllm_demo_format.py
echo "--- example dataset configs ---"
grep -R -n 'output_llava\|dataset_dir:\|dataset:' /opt/MindSpeed-MM/examples/qwen3_5 /opt/MindSpeed-MM/examples 2>/dev/null | head -100
echo "--- data log ---"
tail -80 /opt/qwen35_08b/logs/prepare_data.log || true
