#!/usr/bin/env bash
set -euo pipefail

npu-smi info
supervisorctl status qwen35 2>/dev/null || true
ps -ef | grep -E '[q]wen35_server|[r]iceFM' || true
