# 性能分析报告

## 性能优化

### 性能优化汇总

本项目的性能优化围绕显存预算、双 NPU 资源隔离和可复现证据展开。Qwen3.5-4B 与 riceFM 分别固定在 `npu:0` 和 `npu:1`，服务由 supervisor 管理，并启用 API Token、日志轮转、自动重启和 `/version`。

| 优化点摘要 | 性能收益 |
|---|---|
| Qwen3.5 MindSpeed-MM 双芯片训练算子对照 | 两份正式 100-step 日志同口径统计：AscendC 2.716775 samples/s，Triton 2.341653 samples/s，AscendC 全流程吞吐提升 16.02%，平均 step time 下降 13.81%。step 11-100 稳态窗口中 Triton 更快，说明收益包含编译与执行全流程差异。 |
| 基于实测显存预算的 FP32/FP16 策略选择 | FP16 峰值分配显存 8.60 GB，相比 FP32 的 16.91 GB 下降 49.14%；加载时间由 5.248 s 降至 2.947 s，下降 43.83%。短文本吞吐下降 11.77%，因此收益定位为显存和双模型驻留。 |
| Qwen Agent 结构化规划与证据约束 | JSON Planner、工具白名单、Evidence Store、Verifier 和一次自动重写形成闭环；目标环境固定 5 项任务 5/5 通过，完成率 100%，失败 0，平均端到端 9.17 s。 |

### 优化点说明

优化点 1：系统先在同一 Ascend NPU、同一输入和相同 64 token 上限下实测 FP32/FP16，再根据 12 GB 显存预算自动选择部署精度，拒绝“只报加速、不报代价”的单指标结论。

优化点 2：Qwen 不直接生成无依据的科研结论。请求先经过 JSON Planner 和工具白名单，只能调用矩阵校验、riceFM 注释、ZH11 指标或报告生成工具；Verifier 校验证据引用、未知证据 ID 和可疑基因符号，缺少表达矩阵时明确返回 `input_required`。

## 官方训练迁移对齐补充

官方赛题主线是 Qwen3.5-0.8B 在 MindSpeed-MM/FSDP 中的训练流程迁移、精度对齐和训练性能优化。双芯片 AscendC 与 Triton 正式运行各完成 100 step，global batch size 均为 8。按 step 1-100 同口径解析：AscendC 平均 step time 2944.668 ms、2.716775 samples/s；Triton 平均 step time 3416.390 ms、2.341653 samples/s；AscendC 全流程吞吐提升 16.02%。逐 step loss 的平均绝对差为 0.000884，最大绝对差 0.002954，最后一步绝对差 0.000212。step 11-100 的稳态窗口中 AscendC 为 4.036069 samples/s、Triton 为 6.898838 samples/s，因此报告不将全流程结果外推为所有阶段的普遍加速。正式精度验收阈值仍应以赛事评测脚本为准。
