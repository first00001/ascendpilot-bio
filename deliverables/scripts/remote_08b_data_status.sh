#!/usr/bin/env bash
set -euo pipefail

echo "--- tools ---"
command -v modelscope || true
command -v torchrun || true
echo "--- shared datasets ---"
find /workspace/shared_assets/datasets -maxdepth 3 -type d -o -type f 2>/dev/null | head -120
echo "--- local data ---"
find /opt/qwen35_08b/data -maxdepth 3 -type f -printf '%p %s bytes\n' 2>/dev/null | head -80
echo "--- checkpoint marker ---"
cat /opt/qwen35_08b/dcp/latest_checkpointed_iteration.txt
