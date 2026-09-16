# Public Release Notes

## Scope

Official mainline: Qwen3.5-0.8B two-NPU MindSpeed-MM/FSDP AscendC/Triton training comparison. Qwen3.5-4B and riceFM/ZH11 are extension scenarios, not substitutes for the official training task.

Current formal evidence is under `deliverables/results/qwen35-formal-dual/`. Each variant completed 100 steps with global batch size 8. AscendC throughput is +16.02% over steps 1-100 and -41.50% over steps 11-100. No universal speedup or official precision pass is claimed.

Grounding has 12 cases per profile. The reported zero rule violations are detector-specific: the gene detector recognizes selected rice-style identifiers, not all gene symbols, pathways or unsupported statements. The baseline output contains unsupported scientific narrative despite its zero detector score. These numbers do not prove absence of hallucinations.

`deliverables/results/migration/` contains historical extension records. Its older migration-report and training comparison do not supersede the dual-NPU formal evidence. The latest extension checks are in `deliverables/results/qwen35-formal-dual/results/extension_verification_20260915/`.

## External Requirements

MindSpeed-MM baseline: `2de94dc0f7453e21015821629036f515673d32e9`. Runtime requirements: CANN 9.1.0-beta.3, Python 3.12.13, PyTorch 2.7.1, torch_npu 2.7.1.post10, AscendC custom operator package and fla_npu wheel. Version/patch records are under `dependencies/fla/`; binaries are intentionally excluded.

The third_party MindSpeed-MM files retain upstream copyright and licenses. They are references, not a complete framework and not a claim of original model architecture. Do not assign the legacy Thief project's GPL license to this separate work without confirming ownership and license compatibility. No blanket license for project-authored files has been chosen; the project owner must decide it before offering reuse rights.

Local path defaults in scripts/configuration describe the verified target. Adapt model, data, service and output paths to a new authorized environment before running. Preserve A/B input parity.

## Known Limitations

- Raw company data, private expression demo samples and private per-cell derived outputs are excluded. Reviewers can run the public-data demonstration with the checked-in `E-ENAD-52` / `GSE146035` subset; the full riceFM/Qwen NPU mode still requires separately authorized local assets and model weights.
- The original static test `test_demo_sample_matrix_widths` depends on excluded real expression samples; do not interpret its absence as a failed model run.
- The recorded DCP patch is preserved as measured evidence and validates synchronous save only. Its async CPU device_id change was not exercised and must not be advertised as async-save support.
- DOCX files are updated but have not passed automatic visual rendering QA.
- No repository push or PR has been performed; README PR placeholders are not real links.

Reports are in `reports/pdf/` and `reports/docx/`, not the private workspace's `output/` paths mentioned in historical notes. Do not upload private source snapshots or the full competition ZIP as additional repository contents.
