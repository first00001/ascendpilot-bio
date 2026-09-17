# 5 分钟复赛演示流程

## 演示原则

官方赛题主线是 Qwen3.5-0.8B 在 MindSpeed-MM/FSDP 上的双 NPU 训练与 AscendC/Triton 对比。PlantCell、Qwen3.5-4B 和 riceFM/ZH11 是迁移能力的扩展验证，不能替代官方主线。现场优先展示已保存的正式结果，不重新跑完整 100 step。

## 0:00-0:30 项目定位

打开项目 README，用一句话说明：项目完成 Qwen3.5-0.8B 双 NPU 训练迁移、AscendC/Triton 同条件 A/B 验证，以及面向植物单细胞分析的可信 Agent 扩展。展示项目仓库和 MindSpeed-MM PR #8。

## 0:30-1:30 官方训练闭环

展示两份正式训练日志末尾的第 100 step 和 `iter_0000100` 保存记录。说明两次实验都使用 2 x Ascend 910、global batch size 8、相同数据、随机种子、步数和训练超参数，唯一主要变量是 AscendC 或 Triton 算子实现。

建议打开：

- `results/qwen35-formal-dual/logs/ascendc_formal_dual_gloo/train_20260915_103446.log`
- `results/qwen35-formal-dual/logs/triton_formal_dual_gloo/train_20260915_104105.log`
- `results/qwen35-formal-dual/results/formal_dual_gloo_checkpoint_files.txt`

## 1:30-2:30 性能和精度对齐

展示 `formal_dual_gloo_100_comparison.json`：step 1-100 中 AscendC 为 2.716775 samples/s，Triton 为 2.341653 samples/s，全流程吞吐高 16.02%。两条 loss 序列平均绝对差 0.000884，最大差 0.002954，第 100 step 差 0.000212。

随后主动展示稳态结果：step 11-100 中 AscendC 为 4.036069 samples/s，Triton 为 6.898838 samples/s，AscendC 低 41.50%。结论只能表述为“全流程窗口包含初始化与编译成本时 AscendC 占优，但本次稳态吞吐未占优”，不能宣称普遍加速。最终精度是否通过以赛事验收阈值或脚本为准。

## 2:30-3:05 工程修复和贡献

打开 MindSpeed-MM PR #8，展示仅有 1 个提交。说明在 HCCL 双 NPU 训练中，DCP 元数据规划需要 CPU Gloo process group；补丁解决同步 checkpoint load/save 的进程组兼容问题，两条正式训练均成功保存第 100 step checkpoint。

## 3:05-4:20 PlantCell 扩展演示

打开 `http://127.0.0.1:8000/demo`。普通电脑离线运行时确认顶部显示“公开数据证据模式”；连接 Ascend 服务时确认显示“公开数据 + Ascend NPU”。依次完成：

1. 两种模式都可在 E-ENAD-52 UMAP 上悬停真实公开细胞，展示 Atlas cluster；强调 cluster 编号不是人工细胞类型。
2. 普通电脑离线模式点击“查看 cluster 证据”展示官方 marker；Ascend 真机模式选择一个 `GSE232863` 公开细胞并点击“运行 riceFM embedding”，展示 256 维输出哈希、norm、NPU 设备和耗时。
3. Ascend 真机模式：输入 API Token 后点击“生成分析”，展示 Qwen3.5-4B 在 `npu:0` 上的实时生成耗时和 tokens/s。
4. 展示 Agent 的 `[E1]` 数据集、`[E2]` marker、`[E3]` 执行边界引用。明确 riceFM 仅执行 `GSE232863` 的 25 基因锚点 compatibility pilot，不是全转录组映射或细胞类型分类。

这一段只演示一次注释和一次问答，不在现场跑多轮 benchmark。

## 4:20-5:00 总结和边界

总结三点：官方 0.8B 双 NPU 训练完成；DCP/Gloo 修复形成上游 PR；扩展场景验证了迁移、部署和证据约束闭环。公开演示使用 E-ENAD-52/GSE146035 与 GSE232863/GSM8865415，私有 ZH11 原始数据仍受授权限制；仓库不包含私有原始矩阵、逐细胞预测、模型权重或 checkpoint 分片。

## 现场准备

- 提前开启 SSH 隧道并访问 `/health`、`/version` 和 `/demo`。
- 确认 `/health` 返回 `mode=public-evidence`、`qwen_device=npu:0`、`ricefm_device=npu:1`，并确认 `ricefm_runtime=public-anchor-pilot-ready`。
- API Token 只在本机输入，不出现在幻灯片、终端历史或录屏中。
- 提前打开 README、PR #8、两份比较 JSON、两份日志末尾和演示页面，按顺序放在浏览器标签页中。
- 关闭聊天软件通知，终端字体调大，浏览器缩放保持 100%。
- 不要现场重跑完整 100 step；若必须展示实时执行，只运行 3-5 step 的独立 smoke test，并明确它不是正式性能结果。

## 断网兜底

网络或服务器不可用时，按相同顺序展示本地 README、正式 JSON、日志末尾、checkpoint 文件清单和报告 PDF。不要用静态结果冒充现场运行；直接说明这是 2026 年 9 月 15 日保存的正式双 NPU 实验记录，并展示对应哈希和原始日志。

## 建议口播结论

“我们完成的不是单一页面演示，而是一条可审计的迁移闭环：Qwen3.5-0.8B 在双 Ascend NPU 上完成 100 step AscendC/Triton 对照训练，两次都保存了 checkpoint；我们如实报告全流程与稳态窗口的不同结论，并把 DCP/Gloo 兼容修复提交为 MindSpeed-MM PR。PlantCell 是这条迁移能力在科研场景中的扩展验证。”
