#!/usr/bin/env bash
set -euo pipefail

echo "--- process tree ---"
ps -eo pid,ppid,etime,stat,pcpu,pmem,cmd --sort=ppid | grep -E '[t]orchrun|[t]rainer.py|[b]isheng|[c]cec|[t]riton|[s]moke' | tail -80
echo "--- cache ---"
du -sh /opt/MindSpeed-MM/triton_cache /root/.triton 2>/dev/null || true
find /opt/MindSpeed-MM/triton_cache /root/.triton -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -40
echo "--- latest training lines ---"
latest="$(find /opt/qwen35_08b/logs/smoke -type f -name 'train_*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
[[ -z "$latest" ]] || tail -60 "$latest"
