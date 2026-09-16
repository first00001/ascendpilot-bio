# PR 提交准备

## 目标

PR 必须提交到项目方实际 fork 的 MindSpeed-MM 仓库，基线固定为赛事指定版本。README 中只能填写真实可访问的 PR URL。

## 应包含的改动

- `mindspeed_mm/fsdp/models/qwen3_5/`：Qwen3.5 模型结构、配置注册和算子选择。
- `examples/qwen3_5/`：0.8B 配置、HF 到 DCP 权重转换入口和双卡训练脚本。
- 训练配置：`gdn_implementation` 的 AscendC/Triton 候选，其他超参数保持一致。
- 精度验证：逐 step loss 比较与明确阈值来源。
- 性能验证：step 1-100 全流程和 step 11-100 稳态两个窗口。
- 单元测试和 README：最小运行命令、环境版本、数据路径占位符与已知限制。

## 不应提交

- 公司集群的 ZH11 `matrix.mtx.gz`，除非已有书面外发授权。
- `/etc/qwen35.env`、API Token、SSH 密码或短期跳板凭据。
- 华为标记为保密、未经授权禁止扩散的原始说明文件。
- 伪造的性能收益、精度阈值或 PR 地址。

## 创建 PR 后回填

1. 在 `README_OFFICIAL.md` 与 `README_已填充.docx` 的“PR 链接”章节写入真实 URL。
2. 在性能与精度报告中记录 PR commit SHA。
3. 重新生成 PDF 与公开 ZIP，并记录 SHA-256。
