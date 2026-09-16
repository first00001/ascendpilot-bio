#!/usr/bin/env bash
set -euo pipefail

job_root=/opt/qwen35_08b
baseline=$job_root/tooling/configs/qwen3_5_0.8B_config.yaml
ascendc_cfg=$job_root/qwen3_5_0.8B_ascendc_formal_single.yaml
triton_cfg=$job_root/qwen3_5_0.8B_triton_formal_single.yaml
status=$job_root/logs/formal_single_training_status.txt
driver_log=$job_root/logs/formal_single_training_driver.log
ascendc_logs=$job_root/logs/ascendc_formal_single
triton_logs=$job_root/logs/triton_formal_single

source /usr/local/Ascend/cann-9.1.0-beta.3/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH
export PYTHONPATH=/usr/local/python3.12.13/lib/python3.12/site-packages:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.1.0-beta.3/opp/vendors/fla_npu_transformer/op_api/lib:/usr/local/Ascend/cann-9.1.0-beta.3/aarch64-linux/lib64:/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/lib:${LD_LIBRARY_PATH:-}
export ASCEND_RT_VISIBLE_DEVICES=1
export NPUS_PER_NODE=1

if [[ -e "$ascendc_cfg" || -e "$triton_cfg" || -e "$status" || -e "$ascendc_logs" || -e "$triton_logs" ]]; then
  echo 'Single-NPU formal run artifacts already exist; refusing to overwrite.' >&2
  exit 2
fi
if ! supervisorctl status qwen35 | grep -q RUNNING; then
  echo 'Qwen3.5-4B service is not healthy; refusing to start.' >&2
  exit 3
fi
if pgrep -f '[m]indspeed_mm/fsdp/train/trainer.py' >/dev/null; then
  echo 'Another training process is active; refusing to start.' >&2
  exit 4
fi

mkdir -p "$ascendc_logs" "$triton_logs" "$job_root/results"
cp "$baseline" "$ascendc_cfg"
sed -i 's#^  save: /opt/qwen35_08b/save$#  save: /opt/qwen35_08b/save_ascendc_formal_single#' "$ascendc_cfg"
cp "$ascendc_cfg" "$triton_cfg"
sed -i 's#^  gdn_implementation: ascendc$#  gdn_implementation: triton#' "$triton_cfg"
sed -i 's#^  save: /opt/qwen35_08b/save_ascendc_formal_single$#  save: /opt/qwen35_08b/save_triton_formal_single#' "$triton_cfg"

for cfg in "$ascendc_cfg" "$triton_cfg"; do
  grep -q '^  train_iters: 100$' "$cfg"
  grep -q '^  micro_batch_size: 4$' "$cfg"
  grep -q '^  load: /opt/qwen35_08b/dcp$' "$cfg"
done
grep -q '^  gdn_implementation: ascendc$' "$ascendc_cfg"
grep -q '^  gdn_implementation: triton$' "$triton_cfg"
sha256sum "$baseline" "$ascendc_cfg" "$triton_cfg" > "$job_root/results/formal_single_config_hashes.txt"
diff -u "$ascendc_cfg" "$triton_cfg" > "$job_root/results/formal_single_config_diff.txt" || test $? -eq 1

stage=ASCENDC
trap 'code=$?; printf "FAILED stage=%s exit=%s %s\n" "$stage" "$code" "$(date -Is)" > "$status"' ERR

cd /opt/MindSpeed-MM
printf 'ASCENDC_RUNNING %s\n' "$(date -Is)" > "$status"
export MASTER_PORT=6026
export TRITON_CACHE_DIR=$job_root/triton_cache_ascendc_formal_single
bash "$job_root/tooling/scripts/run_training.sh" "$ascendc_cfg" "$ascendc_logs" >> "$driver_log" 2>&1
ascendc_log=$(find "$ascendc_logs" -maxdepth 1 -type f -name 'train_*.log' | sort | tail -n 1)
grep -Eq 'iteration +100/ +100' "$ascendc_log"
printf 'ASCENDC_COMPLETE %s\n' "$(date -Is)" > "$status"

stage=TRITON
printf 'TRITON_RUNNING %s\n' "$(date -Is)" > "$status"
export MASTER_PORT=6027
export TRITON_CACHE_DIR=$job_root/triton_cache_triton_formal_single
bash "$job_root/tooling/scripts/run_training.sh" "$triton_cfg" "$triton_logs" >> "$driver_log" 2>&1
triton_log=$(find "$triton_logs" -maxdepth 1 -type f -name 'train_*.log' | sort | tail -n 1)
grep -Eq 'iteration +100/ +100' "$triton_log"

stage=COMPARISON
python "$job_root/tooling/scripts/compare_training_logs.py" \
  --ascendc "$ascendc_log" --triton "$triton_log" \
  --start 1 --end 100 \
  --output "$job_root/results/formal_single_100_comparison.json" >> "$driver_log" 2>&1
python "$job_root/tooling/scripts/compare_training_logs.py" \
  --ascendc "$ascendc_log" --triton "$triton_log" \
  --start 11 --end 100 \
  --output "$job_root/results/formal_single_steady_comparison.json" >> "$driver_log" 2>&1
printf 'COMPLETE %s\n' "$(date -Is)" > "$status"
