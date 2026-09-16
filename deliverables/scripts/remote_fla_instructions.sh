#!/usr/bin/env bash
set -euo pipefail

sed -n '70,110p' /opt/MindSpeed-MM/examples/qwen3_5/README.md
echo "--- connectivity ---"
git ls-remote https://github.com/flashserve/flash-linear-attention-npu HEAD 2>&1 | head -10 || true
echo "--- installed packages ---"
/usr/local/python3.12.13/bin/python3 -m pip list | grep -E -i 'fla|flash|triton' || true
