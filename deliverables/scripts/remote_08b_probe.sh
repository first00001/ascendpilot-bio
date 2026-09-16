#!/usr/bin/env bash
set -euo pipefail

source /usr/local/Ascend/cann/set_env.sh
/usr/local/python3.12.13/bin/python3 - <<'PY'
import importlib.util
import numpy
import torch
import torch_npu
import triton

print("triton", triton.__version__)
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("numpy", numpy.__version__)
print("npu", torch.npu.is_available(), torch.npu.device_count())
print("fla_npu", importlib.util.find_spec("fla_npu"))
PY
command -v mm-convert
grep -R -n -E 'fla_npu|triton' /opt/MindSpeed-MM/mindspeed_mm/fsdp/models/qwen3_5 | head -40 || true
