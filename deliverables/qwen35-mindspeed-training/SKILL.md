---
name: qwen35-mindspeed-training
description: Migrate, run, compare, and report Qwen3.5 training on MindSpeed-MM/FSDP for the Ascend competition.
---

# Qwen3.5 MindSpeed-MM Training Skill

## Purpose

Use this Skill as the official competition mainline. It verifies that Qwen3.5 training is implemented under `mindspeed_mm/fsdp/models/qwen3_5/`, runs the official two-NPU training command, and compares AscendC and Triton logs with identical steps and hyperparameters.

## Required inputs

- MindSpeed-MM 26.1.0 checkout
- Qwen3.5-0.8B HF and converted DCP weights
- Converted LLaVA/COCO training manifest
- AscendC and Triton configuration variants
- Two Ascend NPUs

## Procedure

1. Verify CANN, PyTorch, torch_npu, MindSpeed-MM commit, model paths and dataset paths.
2. Set `gdn_implementation` to `ascendc` or `triton` while keeping all other training parameters identical.
3. Run `scripts/run_training.sh CONFIG_PATH LOG_DIR` for each variant.
4. Run `scripts/compare_logs.py` over the two logs.
5. Report both the official 100-step end-to-end result and a separately labeled warm steady-state window.
6. Accept precision alignment only when the full per-step loss series, maximum absolute loss difference and final-step difference are recorded. Do not infer equivalence from a single step.

## Output evidence

- Original training logs
- Machine-readable comparison JSON
- Environment and configuration hashes
- PR URL pointing to the actual MindSpeed-MM changes
- Accuracy, performance analysis and performance test reports

## Restrictions

- Do not substitute Qwen3.5-4B inference results for the official 0.8B training task.
- Do not claim AscendC is universally faster. The provided logs show a 100-step end-to-end gain but a different warm steady-state result.
- Do not fabricate a PR URL or copy Huawei-confidential input documents into a public repository.
