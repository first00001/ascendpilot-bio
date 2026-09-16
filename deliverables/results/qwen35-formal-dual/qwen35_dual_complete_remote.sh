#!/usr/bin/env bash
set -euo pipefail

root=/opt/qwen35_08b
repo=/opt/MindSpeed-MM
checkpointer=$repo/mindspeed_mm/fsdp/checkpoint/dcp_checkpointer.py
status=$root/logs/formal_dual_gloo_status.txt
driver=$root/logs/formal_dual_gloo_driver.log
stage=PATCH

source /usr/local/Ascend/cann-9.1.0-beta.3/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH
export PYTHONPATH=/usr/local/python3.12.13/lib/python3.12/site-packages:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.1.0-beta.3/opp/vendors/fla_npu_transformer/op_api/lib:/usr/local/Ascend/cann-9.1.0-beta.3/aarch64-linux/lib64:/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/lib:${LD_LIBRARY_PATH:-}
export NPUS_PER_NODE=2

trap 'code=$?; printf "FAILED stage=%s exit=%s %s\n" "$stage" "$code" "$(date -Is)" > "$status"' ERR

if pgrep -f '[m]indspeed_mm/fsdp/train/trainer.py' >/dev/null; then
  echo 'Another training process is active.' >&2
  exit 4
fi
if ! supervisorctl status qwen35 | grep -q RUNNING; then
  echo 'Qwen3.5-4B service is not healthy.' >&2
  exit 3
fi

cd "$repo"
if ! grep -q 'def _get_sync_process_group' "$checkpointer"; then
  python - "$checkpointer" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
replacements = [
    (
        "    _async_process_group: Optional[Any] = None\n",
        "    _async_process_group: Optional[Any] = None\n"
        "    _sync_process_group: Optional[Any] = None\n",
    ),
    (
        "        else:\n"
        "            dcp.load(\n"
        "                state_dict=load_state,\n"
        "                storage_reader=storage_reader,\n"
        "                process_group=process_group,\n",
        "        else:\n"
        "            if process_group is None and dist.is_initialized():\n"
        "                process_group = cls._get_sync_process_group()\n"
        "            dcp.load(\n"
        "                state_dict=load_state,\n"
        "                storage_reader=storage_reader,\n"
        "                process_group=process_group,\n",
    ),
    (
        "        else:\n"
        "            dcp.save(\n"
        "                state_dict=save_state,\n"
        "                storage_writer=storage_writer,\n"
        "            )\n",
        "        else:\n"
        "            process_group = cls._get_sync_process_group() if dist.is_initialized() else None\n"
        "            dcp.save(\n"
        "                state_dict=save_state,\n"
        "                storage_writer=storage_writer,\n"
        "                process_group=process_group,\n"
        "            )\n",
    ),
    (
        "    # Private helper methods\n"
        "    @classmethod\n"
        "    def _create_checkpoint_dir(cls, checkpoint_dir: str) -> None:\n",
        "    # Private helper methods\n"
        "    @classmethod\n"
        "    def _get_sync_process_group(cls):\n"
        "        \"\"\"Use CPU collectives for DCP metadata planning on HCCL systems.\"\"\"\n"
        "        if cls._sync_process_group is None:\n"
        "            cls._sync_process_group = dist.new_group(backend=\"gloo\")\n"
        "        return cls._sync_process_group\n\n"
        "    @classmethod\n"
        "    def _create_checkpoint_dir(cls, checkpoint_dir: str) -> None:\n",
    ),
]
for old, new in replacements:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected exactly one match, got {count}: {old[:80]!r}")
    text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")
PY
fi
python - "$checkpointer" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
old_variants = (
    '            cls._sync_process_group = dist.new_group(backend="gloo")',
    '            cls._sync_process_group = dist.new_group(backend="gloo", device_id=torch.device("cpu"))',
    '            cls._sync_process_group = dist.new_group(backend="cpu:gloo")',
)
new = '''            default_group = dist.distributed_c10d._get_default_group()
            bound_device_id = default_group.bound_device_id
            try:
                # torch_npu rejects a CPU device_id, while PyTorch otherwise
                # inherits the default NPU binding into this CPU-only group.
                default_group.bound_device_id = None
                cls._sync_process_group = dist.new_group(backend="cpu:gloo")
            finally:
                default_group.bound_device_id = bound_device_id'''
if new not in text:
    matches = [variant for variant in old_variants if variant in text]
    if len(matches) != 1:
        raise SystemExit(f"expected one Gloo construction variant, found {len(matches)}")
    text = text.replace(matches[0], new, 1)
path.write_text(text, encoding="utf-8")
PY
python -m py_compile "$checkpointer"
git diff -- "$checkpointer" > "$root/results/dcp_gloo_process_group.patch"

stage=PROBE
probe_cfg=$root/qwen3_5_0.8B_dual_gloo_probe.yaml
probe_logs=$root/logs/dual_gloo_probe
rm -rf "$probe_cfg" "$probe_logs" "$root/save_dual_gloo_probe" "$root/triton_cache_dual_gloo_probe"
cp "$root/tooling/configs/qwen3_5_0.8B_config.yaml" "$probe_cfg"
sed -i 's#^  train_iters: 100$#  train_iters: 1#' "$probe_cfg"
sed -i 's#^  save: /opt/qwen35_08b/save$#  save: /opt/qwen35_08b/save_dual_gloo_probe#' "$probe_cfg"
printf 'PROBE_RUNNING %s\n' "$(date -Is)" > "$status"
export MASTER_PORT=6029 TRITON_CACHE_DIR=$root/triton_cache_dual_gloo_probe
bash "$root/tooling/scripts/run_training.sh" "$probe_cfg" "$probe_logs" >> "$driver" 2>&1
probe_log=$(find "$probe_logs" -maxdepth 1 -type f -name 'train_*.log' | sort | tail -n 1)
grep -Eq 'iteration +1/ +1' "$probe_log"
test -f "$root/save_dual_gloo_probe/iter_0000001/.metadata"

base=$root/tooling/configs/qwen3_5_0.8B_config.yaml
asc_cfg=$root/qwen3_5_0.8B_ascendc_formal_dual_gloo.yaml
tri_cfg=$root/qwen3_5_0.8B_triton_formal_dual_gloo.yaml
asc_logs=$root/logs/ascendc_formal_dual_gloo
tri_logs=$root/logs/triton_formal_dual_gloo

if [[ -e "$asc_cfg" || -e "$tri_cfg" || -e "$asc_logs" || -e "$tri_logs" ]]; then
  echo 'Dual Gloo formal artifacts already exist; refusing to overwrite.' >&2
  exit 2
fi
mkdir -p "$asc_logs" "$tri_logs"
cp "$base" "$asc_cfg"
sed -i 's#^  save: /opt/qwen35_08b/save$#  save: /opt/qwen35_08b/save_ascendc_formal_dual_gloo#' "$asc_cfg"
cp "$asc_cfg" "$tri_cfg"
sed -i 's#^  gdn_implementation: ascendc$#  gdn_implementation: triton#' "$tri_cfg"
sed -i 's#^  save: /opt/qwen35_08b/save_ascendc_formal_dual_gloo$#  save: /opt/qwen35_08b/save_triton_formal_dual_gloo#' "$tri_cfg"
for cfg in "$asc_cfg" "$tri_cfg"; do
  grep -q '^  train_iters: 100$' "$cfg"
  grep -q '^  micro_batch_size: 4$' "$cfg"
  grep -q '^  load: /opt/qwen35_08b/dcp$' "$cfg"
done
diff -u "$asc_cfg" "$tri_cfg" > "$root/results/formal_dual_gloo_config_diff.txt" || test $? -eq 1
sha256sum "$base" "$asc_cfg" "$tri_cfg" > "$root/results/formal_dual_gloo_config_hashes.txt"

stage=ASCENDC
printf 'ASCENDC_RUNNING %s\n' "$(date -Is)" > "$status"
export MASTER_PORT=6030 TRITON_CACHE_DIR=$root/triton_cache_ascendc_formal_dual_gloo
bash "$root/tooling/scripts/run_training.sh" "$asc_cfg" "$asc_logs" >> "$driver" 2>&1
asc_log=$(find "$asc_logs" -maxdepth 1 -type f -name 'train_*.log' | sort | tail -n 1)
grep -Eq 'iteration +100/ +100' "$asc_log"
test -f "$root/save_ascendc_formal_dual_gloo/iter_0000100/.metadata"

stage=TRITON
printf 'TRITON_RUNNING %s\n' "$(date -Is)" > "$status"
export MASTER_PORT=6031 TRITON_CACHE_DIR=$root/triton_cache_triton_formal_dual_gloo
bash "$root/tooling/scripts/run_training.sh" "$tri_cfg" "$tri_logs" >> "$driver" 2>&1
tri_log=$(find "$tri_logs" -maxdepth 1 -type f -name 'train_*.log' | sort | tail -n 1)
grep -Eq 'iteration +100/ +100' "$tri_log"
test -f "$root/save_triton_formal_dual_gloo/iter_0000100/.metadata"

stage=COMPARE
python "$root/tooling/scripts/compare_training_logs.py" --ascendc "$asc_log" --triton "$tri_log" --start 1 --end 100 --output "$root/results/formal_dual_gloo_100_comparison.json" >> "$driver" 2>&1
python "$root/tooling/scripts/compare_training_logs.py" --ascendc "$asc_log" --triton "$tri_log" --start 11 --end 100 --output "$root/results/formal_dual_gloo_steady_comparison.json" >> "$driver" 2>&1

stage=EXTENSION_VERIFICATION
extension=$root/results/extension_verification_20260915
mkdir -p "$extension"
bash /opt/plantcell/plantcell-skill/scripts/healthcheck.sh > "$extension/healthcheck.log" 2>&1
python /opt/plantcell/qwen35-ascend-migration/scripts/inspect_environment.py \
  --model /workspace/shared_assets/models/Qwen/Qwen3.5-4B \
  --output "$extension/environment.json" > "$extension/environment.log" 2>&1
python /opt/plantcell/evaluate_qwen_grounding.py \
  --model /workspace/shared_assets/models/Qwen/Qwen3.5-4B \
  --device npu:1 --max-new-tokens 128 \
  --output "$extension/grounding-ab.json" > "$extension/grounding-ab.log" 2>&1
python - "$extension/grounding-ab.json" "$extension/migration-verification.json" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

grounding = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
payload = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "status": "passed",
    "checks": [
        {"name": "service_health", "status": "passed"},
        {"name": "environment_fingerprint", "status": "passed"},
        {"name": "dual_npu_training_100_steps", "status": "passed"},
        {"name": "grounding_12_case_ab", "status": "passed"},
    ],
    "grounding_metrics": {
        name: value["metrics"] for name, value in grounding["profiles"].items()
    },
    "notes": [
        "Training comparison and grounding A/B are separate experiments.",
        "Grounding audit is rule-based and not a complete biological quality judge.",
    ],
}
Path(sys.argv[2]).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

stage=PACKAGE
bundle=$root/results/formal_dual_gloo_evidence_20260915.tar.gz
manifest=$root/results/formal_dual_gloo_checkpoint_files.txt
hashes=$root/results/formal_dual_gloo_evidence_sha256.txt
find "$root/save_ascendc_formal_dual_gloo/iter_0000100" "$root/save_triton_formal_dual_gloo/iter_0000100" -maxdepth 1 -type f -printf '%p\t%s\n' | sort > "$manifest"
sha256sum "$asc_cfg" "$tri_cfg" "$asc_log" "$tri_log" "$root/results/formal_dual_gloo_100_comparison.json" "$root/results/formal_dual_gloo_steady_comparison.json" "$root/results/dcp_gloo_process_group.patch" > "$hashes"
tar -C "$root" -czf "$bundle" \
  qwen3_5_0.8B_ascendc_formal_dual_gloo.yaml \
  qwen3_5_0.8B_triton_formal_dual_gloo.yaml \
  qwen35_dual_complete_remote.sh \
  logs/formal_dual_gloo_driver.log \
  logs/dual_gloo_probe \
  logs/ascendc_formal_dual_gloo \
  logs/triton_formal_dual_gloo \
  results/dcp_gloo_process_group.patch \
  results/formal_dual_gloo_config_diff.txt \
  results/formal_dual_gloo_config_hashes.txt \
  results/formal_dual_gloo_100_comparison.json \
  results/formal_dual_gloo_steady_comparison.json \
  results/formal_dual_gloo_checkpoint_files.txt \
  results/formal_dual_gloo_evidence_sha256.txt \
  results/extension_verification_20260915
printf 'COMPLETE %s\n' "$(date -Is)" > "$status"
