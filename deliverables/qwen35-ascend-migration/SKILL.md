---
name: qwen35-ascend-migration
description: Automatically inspect, migrate, optimize, deploy, and verify Qwen3.5 on Huawei Ascend NPU. Use when converting an existing Qwen3.5 model service to torch-npu, selecting a measured precision policy, or producing reproducible migration evidence.
---

# Qwen3.5 Ascend Migration

Make Qwen3.5 migration the primary task. Industry applications are verification
scenarios and must not replace the migration report.

## Required Workflow

1. Run `scripts/inspect_environment.py` and record OS, architecture, CANN, NPU,
   Python, PyTorch, torch-npu and Transformers versions. Do not infer compatibility.
2. Verify the source model directory contains a readable config and tokenizer.
3. Establish a baseline before changing deployment. Preserve raw output and failure
   logs.
4. Benchmark only compatible candidates. At minimum compare FP32 and FP16 using the
   same prompt, decoding settings and device. Never call a memory tradeoff a speedup.
5. Run `scripts/select_policy.py` with the deployment memory budget and objective.
   Use its JSON decision as the deployment input rather than choosing silently.
6. Deploy through the repository's versioned installer. Keep the stable
   Transformers + torch-npu path available while experimental backends are tested
   in isolation.
7. Verify health, authentication, deterministic generation, restart recovery and
   the application scenario. Roll back when a required check fails.
8. Produce a migration report conforming to
   [schemas/migration-report.json](schemas/migration-report.json). Clearly label
   measured, inferred, failed and not-run items.

## Scoring Evidence

Read [references/judging-evidence.md](references/judging-evidence.md) before preparing
competition materials. Include source-to-target mapping, automated decisions, raw
benchmarks, accuracy/grounding checks, negative results, environment fingerprints,
and a one-command reproduction path.

For the PlantCell demonstration, load the sibling `plantcell-skill` only after the
Qwen migration checks pass. The biological model is evidence of extensibility, not
the Qwen migration mechanism itself.

