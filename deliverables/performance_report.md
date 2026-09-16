# 性能测试报告

## 1. 环境

- 设备：2 x Ascend 910，64 GB HBM/卡
- 系统：openEuler 24.03 ARM64
- CANN：9.1.0-beta.3
- PyTorch：2.7.1 + torch-npu 2.7.1.post10
- NumPy：2.5.0
- Qwen：Qwen3.5-4B，`npu:0`
- 生物模型：riceFM，6 层、4 头、256 维，`npu:1`

## 2. 测试方法

使用 `scripts/evaluate_zh11.py` 对 5,188 个真实 ZH11 细胞执行完整流程：
流式读取 9,515,606 个非零值、51 箱预处理、最长 1,200 token 截断、批量大小
16、riceFM 前向和 embedding 回传。固定随机种子为 2026。

服务接口测试对 `/ricefm/annotate` 与 `/chat` 先预热，再顺序发起真实请求，记录端到端延迟、P95 和错误率。Qwen 精度对照在同一张 `npu:1` 上依次加载 FP32 与 FP16 权重，使用相同 prompt、贪心解码和 64-token 上限；峰值显存来自当前 PyTorch 进程的 NPU allocator 统计。

```bash
python scripts/evaluate_zh11.py \
  --data-dir /opt/plantcell/data/ZH11_riceFM_eval \
  --model-dir /opt/riceFM_save/eval-Nov05-18-46-2025 \
  --repo-dir /opt/riceFM --output-dir /opt/plantcell/results/zh11 \
  --device npu:1 --batch-size 16
```

## 3. 实测结果

### 官方训练迁移日志证据

随附日志显示，Qwen3.5 训练配置在 2 卡 Ascend 环境分别完成 AscendC 与 Triton 正式 100 step，使用 FSDP、bf16、激活卸载与重计算，global batch size 均为 8；两套配置除算子实现和输出目录外保持一致。按 step 1-100 全口径统计，AscendC 平均 2944.668 ms/step、2.716775 samples/s，Triton 平均 3416.390 ms/step、2.341653 samples/s，AscendC 吞吐提升 16.02%，平均 step time 下降 13.81%。step 11-100 稳态统计则为 AscendC 4.036069 samples/s、Triton 6.898838 samples/s，AscendC 低 41.50%，表明编译、缓存和动态执行阶段会显著影响不同统计窗口。逐 step loss 平均绝对差 0.000884，最大 0.002954，最后一步 0.000212；是否达到官方精度阈值仍以赛事评测程序为准。

| 项目 | 结果 |
|---|---:|
| 数据与标签元信息加载 | 0.0598 s |
| riceFM 全量前向 | 25.1301 s |
| 处理细胞数 | 5,188 |
| 推理吞吐量 | 206.45 cells/s |
| embedding 维度 | 256 |
| 非有限 embedding | 0 |

### 在线接口实测（2026-09-14）

| 接口 | 请求数 | 平均延迟 | P95 | 错误率 | 吞吐 |
|---|---:|---:|---:|---:|---:|
| riceFM 单细胞注释（热态） | 5 | 9.55 ms | 9.98 ms | 0 | - |
| Qwen3.5，64-token 上限 | 3 | 3.226 s | 3.247 s | 0 | 19.86 tok/s |

### Qwen FP32/FP16 对照

| 精度 | 加载时间 | 生成时间 | 吞吐 | 峰值分配显存 |
|---|---:|---:|---:|---:|
| FP32 | 5.248 s | 2.862 s | 22.36 tok/s | 16.91 GB |
| FP16 | 2.947 s | 3.244 s | 19.73 tok/s | 8.60 GB |

FP16 将峰值分配显存降低 49.14%、加载时间降低 43.83%，但本次短文本测试吞吐为 FP32 的 88.23%。因此当前线上选择 FP16 的理由是降低显存占用、支持 Qwen 与 riceFM 双模型稳定驻留，而不是宣称其生成更快。原始 JSON 与 `npu-smi` 快照位于 `results/benchmarks/20260914-034732`。

### Agent 端到端真机验收

Qwen Agent v2.2.1 在目标环境执行 5 条固定任务，覆盖结构化规划、工具白名单、证据引用、报告验证、自动重写和缺少表达矩阵处理。最终 5/5 通过，任务完成率 100%，失败数 0，平均端到端耗时 9.17 秒。其中短结论任务首次缺少证据编号，Verifier 触发一次自动重写后通过；无表达矩阵的注释请求正确返回 `input_required`，未调用 riceFM 或编造预测。原始结果位于 `results/migration/agent-tasks-v221.json`。

| Agent 指标 | 实测结果 |
|---|---:|
| 固定任务数 | 5 |
| 成功任务数 | 5 |
| 任务完成率 | 100% |
| 失败数 | 0 |
| 平均端到端耗时 | 9.17 s |
| Verifier 自动重写验证 | 通过 |
| 缺输入安全终止 | 通过 |

原始结果位于
`/opt/plantcell/results/zh11/metrics.json`，embedding 位于
`embeddings.npy`，其 SHA-256 为
`c1f722039cea8656af258ea6598755b502e84c022586a0e8620710e6ce6b48b4`。

## 4. 测试边界

riceFM 9.55 ms 是单细胞、512 个高表达非零基因、模型与参考库已加载后的热态接口延迟；25.13 秒是 5,188 个细胞的离线批量前向，二者不可直接互换。Agent 9.17 秒包含 Planner 与 Reporter 等多次串行生成，不能与单次 `/chat` 延迟直接比较。当前接口为非流式，因此未报告 TTFT；并发大于 1 的稳态曲线和冷启动方差尚未采集。12 题 grounding A/B 已在双芯片正式训练完成后执行并生成机器可读结果。服务由 supervisor 管理，日志目录为 `/var/log/qwen35/`。

### 12 题 Grounding A/B

Qwen3.5-4B 在 `npu:1` 完成 12 个固定问题的 baseline 与 evidence-guarded 对照。两组均为 12/12 响应，unsupported gene response、unsupported citation response 和 transparent rule violation 均为 0。Evidence-guarded 组对缺失证据的明确承认率为 83.33%，baseline 为 50.00%；平均生成时间分别为 6.345 s 与 6.860 s。该规则审计用于验证证据边界，不等同于完整的生物学质量评估。
