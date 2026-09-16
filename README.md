# README

## 项目

AscendPilot-Bio：Qwen3.5-0.8B MindSpeed-MM/FSDP 训练迁移 Skill，附 Qwen3.5-4B Agent 与 riceFM/ZH11 创新验证场景。

## 环境版本

- 操作系统：openEuler 24.03 ARM64
- 硬件：2 x Ascend 910，单卡 64 GB HBM
- CANN：9.1.0-beta.3
- Python：3.12.13（`/usr/local/python3.12.13/bin/python3`）
- PyTorch：2.7.1
- torch_npu：2.7.1.post10
- Qwen：Qwen3.5-4B，运行于 `npu:0`
- riceFM：用户 checkpoint，运行于 `npu:1`

## 运行启动脚本

官方训练主线：

```bash
bash deliverables/qwen35-mindspeed-training/scripts/run_training.sh \
  examples/qwen3_5/qwen3_5_0.8B_config.yaml logs/ascendc
python deliverables/scripts/compare_training_logs.py \
  --ascendc logs/ascendc/train.log --triton logs/triton/train.log \
  --start 1 --end 100 --output results/training-log-comparison.json
```

AscendC 与 Triton 两次运行除算子实现外必须保持数据、随机种子、batch size、步数和训练超参数一致。

扩展演示服务：

目标机服务由 supervisor 管理，启动入口为 `/opt/start_qwen35.sh`，服务监听 `127.0.0.1:8000`，并启用 API Token、日志轮转、自动重启和 `/version`。

```bash
ssh -N -L 8000:127.0.0.1:8000 \
  -J 'JUMP_USER:JUMP_TOKEN@JUMP_HOST:JUMP_PORT' root@TARGET_HOST
export PLANTCELL_API_TOKEN='<与目标机 /etc/qwen35.env 相同的 Token>'
python demo.py
python benchmark.py --mode ricefm --runs 20 --warmup 2
python benchmark.py --mode chat --runs 10 --warmup 2 --max-new-tokens 128
```

浏览器访问 `http://127.0.0.1:8000/demo`。生产接口除 `/health`、`/version`、`/demo` 外均需 `x-api-key` 或 Bearer Token。

## 代码结构说明

- `deliverables/qwen35-ascend-migration/`：环境检查、候选精度实测、部署策略选择、迁移验证和报告 Schema。
- `deliverables/qwen35-mindspeed-training/`：官方赛题主 Skill，包含双卡训练启动和 AscendC/Triton 日志对齐。
- `deliverables/plantcell-skill/`：标准 Skill、API 文档、JSON Schema、安装/启停/状态/健康检查/演示脚本。
- `deliverables/skill/`：Qwen Agent 与 riceFM 适配器实现。
- `deliverables/deployment/`：Qwen 服务端、部署脚本和性能升级脚本。
- `deliverables/scripts/`：ZH11 构建、精度评测、性能压测、Agent 验收和环境检查脚本。
- `deliverables/demo/`：无额外前端依赖的评审演示页。
- `deliverables/results/`：迁移报告、性能原始 JSON、npu-smi 快照、精度指标和逐细胞预测。

## 代码逻辑

请求先进入 Qwen JSON Planner，经过工具白名单过滤后调用 `validate_matrix`、`ricefm_annotate`、`zh11_metrics` 或 `qwen_report`。Evidence Store 保存 E0 计划、E1 注释和 E2 指标；Verifier 检查证据引用、未知证据 ID 和可疑基因符号，失败时自动重写一次。缺少 `genes/counts` 时返回 `input_required`，不会编造 marker、差异基因或通路结论。

## 官方训练实测结果

双芯片正式 A/B 两份日志各完成 100 step，global batch size 均为 8。按 step 1-100 同口径统计：AscendC 2.716775 samples/s，Triton 2.341653 samples/s，AscendC 全流程吞吐提升 16.02%；逐 step loss 平均绝对差 0.000884，最大 0.002954，第 100 step 差 0.000212。step 11-100 稳态窗口中 AscendC 为 4.036069 samples/s、Triton 为 6.898838 samples/s，AscendC 低 41.50%，因此不能把全流程收益外推为普遍加速。最终精度通过标准以赛事验收脚本为准。

## 扩展场景实测结果

Qwen 在线平均 3.226 s，P95 3.247 s，19.86 tok/s，错误率 0；riceFM 热态单细胞接口平均 9.55 ms，P95 9.98 ms，错误率 0；ZH11 5,188 细胞批量推理耗时 25.1301 s，吞吐 206.45 cells/s。Qwen FP16 峰值显存 8.60 GB，相比 FP32 的 16.91 GB 下降 49.14%，但本次短文本吞吐低 11.77%，因此仅作为显存和双模型驻留优化。Agent 真机固定任务 5/5 通过，完成率 100%。

## 数据说明

ZH11 表达矩阵来自公司集群。未经数据所有者书面授权，不得将 `matrix.mtx.gz` 上传到竞赛平台或公开仓库；可保留脚本、哈希、元数据摘要和已生成的结果文件。

## PR 链接

- 项目仓库：<https://github.com/first00001/ascendpilot-bio>
- MindSpeed-MM PR：<https://github.com/Ascend/MindSpeed-MM/pull/8>
- PR 分支：`first00001:qwen35-08b-ascendc-gloo`
