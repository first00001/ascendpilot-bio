# 精度分析报告

## Qwen3.5 训练精度对齐

对双芯片 AscendC 与 Triton 正式 100-step 训练日志按相同步号解析。两次运行的 global batch size 均为 8；两条 loss 序列的平均绝对差为 0.000884，最大绝对差为 0.002954，第 100 step 绝对差为 0.000212；平均 loss 分别为 1.375924 与 1.375591。上述结果证明两条曲线高度接近，但最终“通过/不通过”必须依据赛事提供的精度阈值或验收脚本，当前文档不自行发明阈值。

![Qwen3.5-0.8B AscendC 与 Triton Loss 精度对齐](results/qwen35-formal-dual/results/formal_dual_gloo_loss_alignment.png)

上图由两份正式训练原始日志逐 step 生成。上半部分为 AscendC 与 Triton 的 loss 叠加曲线，下半部分为相同步号的绝对 loss 差值。制图脚本为 `scripts/plot_loss_alignment.py`，可由原始日志重新生成 SVG 和 PNG。

## 1. 测试目标

验证 riceFM 预训练 checkpoint 对水稻单细胞表达谱的表征能力，以及基于 ZH11
标注参考集进行一级细胞类型标签迁移的可用性。本报告只陈述已执行的真实实验。

## 2. 数据集与划分

| 数据集 | 物种 | 细胞数 | 基因数 | 一级类型 | 划分 |
|---|---|---:|---:|---:|---|
| ZH11_riceFM_eval | 水稻 ZH11 | 5,188 | 39,914 | 52 | 分层留出，80%/20% |

数据来自公司集群
`/share/home/bgi_wangbz/data_keep_zh11-SC/ZH11_SC_ALL.h5ad`。原始数据包含
876,980 个细胞。评测集使用固定随机种子 2026，对每个非 `Unknown`
`celltype_L1` 最多抽取 100 个细胞，共含 9,515,606 个非零表达值。
训练参考集 4,150 个细胞，测试集 1,038 个细胞。

## 3. 方法

1. 使用 checkpoint 对每个细胞执行 51 分位箱表达预处理。
2. 保留 `<cls>`，按 checkpoint 配置将序列限制为 1,200 token。
3. 在 Ascend `npu:1` 上生成 256 维 riceFM cell embedding。
4. 对 embedding 做 L2 归一化，以训练参考集执行余弦相似度加权 5-NN。
5. 对留出测试细胞计算 Accuracy、Macro-F1 和 Weighted-F1。

## 4. 实测结果

| 指标 | 结果 |
|---|---:|
| Accuracy | 0.6464 |
| Macro-F1 | 0.6449 |
| Weighted-F1 | 0.6445 |
| 测试细胞数 | 1,038 |
| 一级细胞类型数 | 52 |

原始指标、逐类 Precision/Recall/F1、混淆矩阵和逐细胞预测保存在目标环境
`/opt/plantcell/results/zh11/metrics.json` 与 `predictions.tsv`。

## 5. 误差分析

F1 较高的类型包括 Pollen 0.9756、Ovary 0.9268、Tapetum 0.9231 和
Stamen 0.9000；较低的类型包括 Epidermis 0.2759、Embryo_unknown 0.2941、
Vascular_cell 0.3448、Lemma 0.3556 和 Procambium 0.3590。结果说明高度特化的
生殖细胞类型更易被当前 embedding 分离，而广泛分布或发育连续的细胞类型存在混淆。
下一轮应结合组织/时期分层参考库、层级分类和 marker 证据校准，而不是只调大 k 值。

## 6. 结论与限制

结果证明 checkpoint 能在昇腾 NPU 上稳定生成有限值 embedding，并能支撑真实
ZH11 标签迁移，但不能据此声称 checkpoint 自带 52 类分类能力。该 checkpoint
可能在预训练阶段见过与 ZH11 重叠的表达谱，因此本实验属于标注参考迁移评测，
不是独立外部验证。ARI、NMI、差异表达准确性和跨数据集泛化尚未执行，不列为已有结果。
