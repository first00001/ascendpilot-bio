# Qwen3.5-0.8B 目标环境状态

更新时间：2026-09-15

## 已确认

- HF 权重已预置：`/workspace/shared_assets/HC2026/helloworld/model/Qwen3.5-0.8B`
- 权重文件约 1.7 GB，配置为 24 层、hidden size 1024、bf16。
- MindSpeed-MM `26.1.0` 已克隆：`/opt/MindSpeed-MM`
- Git commit：`2de94dc0f7453e21015821629036f515673d32e9`
- 分支已包含 `mindspeed_mm/fsdp/models/qwen3_5/` 和 `examples/qwen3_5/`，但没有官方 0.8B 示例配置。
- 0.8B 配置与脚本已上传：`/opt/qwen35_08b/tooling/`
- 两张 Ascend 910 可见，单卡 64 GB HBM；`torch.npu.is_available()` 为 true。
- LLaVA 标注已预置于 `/workspace/shared_assets/datasets/LLaVA-Instruct-150K/`。
- MindSpeed Core `0.12.1` 与 MindSpeed-MM 已完成安装，`mm-convert` 可用。
- `triton-ascend 3.2.1` 已从昇腾专用源安装；其 Python 模块内部版本号报告为 `3.2.0`。
- 原有 Qwen3.5-4B 服务保持健康，未被本次训练环境改动替换。
- `flash-linear-attention-npu` 已从提交 `60a791f` 加补丁
  `50cba07` 构建并安装，Python 包版本为 `fla-npu 1.0.0`；面向
  `ascend910_93` 的自定义算子包也已安装。
- 0.8B HF 权重已成功转换为 DCP，耗时约 22.8 秒：
  - `/opt/qwen35_08b/dcp/release/__0_0.distcp`（约 1.75 GB）
  - `/opt/qwen35_08b/dcp/release/.metadata`
  - `/opt/qwen35_08b/dcp/latest_checkpointed_iteration.txt`
- COCO 2017 训练集已完成下载和解压：压缩包为 19,336,861,798 字节，数据目录总计约 37 GB。
- LLaVA 训练标注已完成转换：`/opt/qwen35_08b/data/output_llava_coco_data.json`，大小约 205 MB。
- 单卡 Triton 冒烟训练已在物理 NPU 1 完整通过：
  - 8 条官方消息格式的纯文本样例已成功预处理；
  - DCP checkpoint 已成功从 `/opt/qwen35_08b/dcp/release` 加载；
  - FSDP、重计算与激活卸载初始化成功；
  - 完成 1 次前向、反向和参数更新，`loss=3.537025`，`grad norm=224.062`；
  - 首次编译迭代耗时约 157.083 秒，不能作为热态性能指标；
  - checkpoint 已保存至 `/opt/qwen35_08b/save/iter_0000001`，共 19 个文件；
  - Triton 缓存约 448 MB。
- 基于真实 COCO/LLaVA 图文数据的 5 步 Triton 预实验已提交至物理 NPU 1，PID 为 `67256`：
  - 配置：`/opt/qwen35_08b/qwen3_5_0.8B_triton_pilot.yaml`
  - 日志：`/opt/qwen35_08b/logs/triton_pilot/train.log`
  - 输出：`/opt/qwen35_08b/save_triton_pilot`
- 5 步 Triton 与 AscendC 预实验均已完成；比较结果位于
  `/opt/qwen35_08b/pilot-5step-comparison.json`。该结果包含首次编译开销，
  只作为 pilot，不作为正式性能结论。
- 在不停止 Qwen3.5-4B 服务的前提下，正式单卡 100-step AscendC/Triton
  对比已于 2026-09-15 17:15:27（Asia/Shanghai）完成。两套配置均为
  `micro_batch_size: 4`、`train_iters: 100`、相同 seed、数据和 DCP，除输出
  目录外仅 `gdn_implementation` 不同。
- 正式 checkpoint：
  - `/opt/qwen35_08b/save_ascendc_formal_single/iter_0000100`
  - `/opt/qwen35_08b/save_triton_formal_single/iter_0000100`
- 正式原始日志：
  - `/opt/qwen35_08b/logs/ascendc_formal_single/train_20260915_090211.log`
  - `/opt/qwen35_08b/logs/triton_formal_single/train_20260915_090840.log`
- 正式比较结果：
  - 1-100：AscendC `1.30070 samples/s`，Triton `1.23232 samples/s`，
    AscendC throughput change `+5.55%`；loss 平均绝对差 `0.00126029`，
    最大绝对差 `0.004655`，最终步绝对差 `0.001393`。
  - 11-100：AscendC `1.89031 samples/s`，Triton `2.74501 samples/s`，
    AscendC throughput change `-31.14%`；loss 平均绝对差 `0.00113323`，
    最大绝对差 `0.003936`，最终步绝对差 `0.001393`。
  - 结果文件：`/opt/qwen35_08b/results/formal_single_100_comparison.json` 与
    `/opt/qwen35_08b/results/formal_single_steady_comparison.json`。
- 第一次双卡正式尝试在迭代 1 前的 DCP 加载阶段失败：HCCL
  `scatter_object_list` 触发 AICPU error `507018`。该失败不是 `fla_npu`
  算子错误，也未纳入正式性能结果；原日志保留在
  `/opt/qwen35_08b/logs/ascendc_formal/`。
- DCP metadata planning 已切换到 CPU/Gloo process group。兼容修复在创建 Gloo
  进程组时临时清空默认 HCCL 组的 NPU `bound_device_id`，创建完成后立即恢复；
  双卡 1-step 探针已通过加载、前向、反向、参数更新和 checkpoint 保存。
- 双卡 AscendC/Triton 正式 A/B 已于 2026-09-15 18:55:04（Asia/Shanghai）完成，
  两边均完成 100 step、global batch size 为 8，并保存 `iter_0000100` checkpoint。
- 双卡 1-100：AscendC `2.716775 samples/s`，Triton `2.341653 samples/s`，
  AscendC throughput change `+16.02%`；loss 平均绝对差 `0.00088404`，最大
  绝对差 `0.002954`，最终步绝对差 `0.000212`。
- 双卡 11-100：AscendC `4.036069 samples/s`，Triton `6.898838 samples/s`，
  AscendC throughput change `-41.50%`；loss 平均绝对差 `0.00080524`，最大
  绝对差 `0.002770`，最终步绝对差 `0.000212`。
- 正式 checkpoint：
  - `/opt/qwen35_08b/save_ascendc_formal_dual_gloo/iter_0000100`
  - `/opt/qwen35_08b/save_triton_formal_dual_gloo/iter_0000100`
- 12 题 grounding A/B、服务健康检查和环境指纹采集已完成；最终远端证据包为
  `/opt/qwen35_08b/results/formal_dual_gloo_evidence_20260915.tar.gz`。

## 当前状态与本地归档

- 官方 0.8B 双卡 100-step AscendC/Triton 对比已完成，无训练阻塞点。
- Qwen3.5-4B 服务在训练期间未被主动停止或重启；完成后再次检查为
  `RUNNING`，PID `359`。
- 远端原始日志、正式配置、配置 diff、环境记录、服务/NPU 状态、结果 JSON、
  checkpoint 文件清单及 SHA-256 已打包并下载到
  `deliverables/results/qwen35-formal-single/`。
- 证据包 `formal_single_evidence_20260915.tar.gz` 的远端和本地 SHA-256 均为
  `60b7b0039c7418152fca63184737e9e491d3e9e84eec11eb920bdd511f03da40`。
- 本地两份比较 JSON 与证据包中的远端原件逐字节一致。AscendC 与 Triton
  正式日志均核得 100 条迭代记录；双卡失败日志为 0 条有效迭代。
- 两套多 GB checkpoint 分片仍保留在远端，本地归档保存了逐文件大小和
  SHA-256，未复制 checkpoint 分片本体。
- 远端提交快照已下载至
  `deliverables/results/qwen35-submission-snapshot/`，共 102 个文件，
  102/102 SHA-256 校验通过；包括 MindSpeed-MM Qwen3.5 源码、FLA wheel/run/source、
  正式配置、训练日志、结果 JSON、环境记录、部署脚本和 checkpoint/DCP 哈希清单。
- 双卡最终证据包已下载到本地并完成 SHA-256 核验。
- 本地提交审计见 `deliverables/SUBMISSION_AUDIT_20260915.md`；当前项目方仍须确认
  团队/签名、真实 PR 和 ZH11 外发授权等非技术事项。

## 下次登录首先执行

```bash
python deliverables/scripts/target_ssh.py \
  --command-file deliverables/scripts/remote_npu_status.sh
python deliverables/scripts/target_ssh.py \
  --command-file deliverables/scripts/remote_08b_status.sh
python deliverables/scripts/target_ssh.py \
  --command-file deliverables/scripts/remote_08b_pilot_status.sh
```

不要重新启动已经完成的正式任务。下一次登录只下载
`formal_dual_gloo_evidence_20260915.tar.gz` 并核验其 SHA-256；若后续需要迁移
checkpoint，可按证据包中的文件清单逐文件验证。
