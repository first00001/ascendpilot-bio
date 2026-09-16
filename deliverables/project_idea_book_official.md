# AscendPilot-Bio：Qwen3.5 昇腾训练迁移 Skill 与水稻单细胞验证

## 一、创意描述

本项目面向官方 Qwen3.5 训练迁移赛题，构建可审计的 `qwen35-mindspeed-training` Skill。主任务是将 Qwen3.5-0.8B 训练流程迁移到 MindSpeed-MM/FSDP，在双卡 Ascend NPU 上运行 AscendC 与 Triton 两种算子方案，自动完成环境检查、训练启动、逐 step loss 对齐、吞吐统计和迁移报告。Qwen3.5-4B Agent 与 riceFM/ZH11 水稻单细胞分析作为扩展创新验证，证明迁移能力可以支撑专业科研应用，但不替代官方 0.8B 训练主线。

一句话关键创新点：用证据驱动的自动迁移闭环完成 Qwen3.5 训练适配，并以真实生物模型验证迁移后的工程价值。

## 二、需求与市场价值

大模型从 GPU 迁移到国产 NPU 时，开发者必须同时处理 CANN、torch_npu、算子兼容、权重转换、分布式并行、显存、精度和性能验收，过程高度依赖人工经验。高校团队与行业研发常遇到“能够启动但没有基线”“只报告加速不报告精度”“环境和步骤无法复现”等问题。

AscendPilot-Bio 把迁移过程固化为 Skill，目标用户包括昇腾生态开发者、高校实验室、模型适配团队和生物信息科研人员。其环境指纹、候选策略、日志比较和证据报告机制可复用于其他 Transformers 模型；riceFM 场景展示其在高维科研数据上的落地能力。

## 三、技术方案

官方训练主链路为：环境指纹采集 → Qwen3.5-0.8B HF/DCP 权重和数据路径检查 → MindSpeed-MM 模型结构加载 → 双卡 FSDP 训练 → AscendC/Triton 候选 A/B → 逐 step loss 与吞吐对照 → JSON 迁移报告。

训练配置使用 2 卡 Ascend、bf16、FSDP、激活卸载、梯度重计算和 chunk loss。AscendC 方案使用 GatedDeltaNet 融合算子，causal convolution 使用 NPU Triton；对照方案将 GatedDeltaNet 切换为 Triton，其他数据、超参数与训练步数保持一致。统一脚本设置 `NON_MEGATRON`、内存复用、任务队列、NPU allocator 和 Triton cache，并使用 `torchrun` 启动两卡训练。

日志比较器按相同步号提取 elapsed time、global batch size 和 loss，分别报告官方 100-step 全流程口径与去掉前 10 步编译预热后的稳态口径。所有负收益和统计边界均写入 JSON，不挑选单一有利指标。

## 四、实测精度与性能

双芯片 AscendC 与 Triton 正式运行均完成 100 step，global batch size 均为 8。按 step 1-100 同口径统计，AscendC 平均 2944.668 ms/step、2.716775 samples/s；Triton 平均 3416.390 ms/step、2.341653 samples/s；AscendC 全流程吞吐提升 16.02%，平均 step time 下降 13.81%。

两条 loss 序列平均绝对差为 0.000884，最大绝对差 0.002954，第 100 step 绝对差 0.000212，平均 loss 分别为 1.375924 和 1.375591。结果表明曲线高度接近，但最终精度是否通过必须以赛事评测阈值或验收脚本为准，项目不自行发明阈值。

step 11-100 稳态窗口中，AscendC 为 4.036069 samples/s，Triton 为 6.898838 samples/s，AscendC 低 41.50%。该结果说明编译、缓存和动态执行阶段会显著改变统计结论，因此报告同时披露全流程与稳态结果，不将 16.02% 外推为所有阶段的普遍加速。

## 五、扩展创新验证

Qwen3.5-4B 服务运行于 `npu:0`，riceFM 运行于 `npu:1`，由 supervisor 管理并启用 API Token、日志轮转、自动重启、健康检查和 `/version`。Qwen Agent 使用 JSON Planner、工具白名单、Evidence Store 和 Verifier，禁止补写未计算的 marker、差异基因和通路；目标环境固定五项任务 5/5 通过，失败数为 0。

riceFM 在 `npu:1` 对 5,188 个 ZH11 细胞生成 256 维 embedding，耗时 25.1301 秒，吞吐 206.45 cells/s。基于标注参考集的 5-NN 标签迁移 Accuracy 0.6464、Macro-F1 0.6449、Weighted-F1 0.6445。该结果属于标注参考迁移，不宣称 checkpoint 自带分类头或完成独立外部验证。

## 六、最小可运行 Demo

执行 `qwen35-mindspeed-training/scripts/run_training.sh` 启动官方双卡训练，执行 `deliverables/scripts/compare_training_logs.py` 生成机器可读对照。通过 SSH 隧道访问 `/demo`，可查看环境版本、真实 ZH11 样本、riceFM 近邻证据和 Qwen 受约束报告。

## 七、排期与提交边界

已完成双芯片正式训练日志对照、12 题 grounding A/B、迁移 Skill、服务工程化和 riceFM/ZH11 验证。提交前需由项目方补齐学校、团队、联系人、签名信息，确认 ZH11 外发授权，推送代码并填写真实 PR 链接，并使用赛事评测程序复核精度阈值。
