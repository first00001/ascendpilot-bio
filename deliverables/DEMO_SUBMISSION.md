# 演示系统上交说明

## 建议提交内容

| 提交项 | 内容 |
| --- | --- |
| 源代码 | GitHub 仓库主分支，包含训练 Skill、迁移 Skill、PlantCell Skill、公开数据离线演示和可复核实验记录 |
| 代码附件 | `github_project_20260917_v2.zip`，用于平台附件上传或评委离线检查 |
| 项目文档 | 官方项目创意书或复赛要求的项目说明 PDF |
| 演示视频 | 建议 5 分钟 MP4，按 `DEMO_GUIDE.md` 展示正式训练证据、PR、riceFM embedding 和 Qwen 分析 |
| 上游贡献 | MindSpeed-MM 真实 PR 链接，不填写占位地址 |

## 评委如何运行

### 无 Ascend 环境

在仓库根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File deliverables\run_offline_demo.ps1
```

然后打开 `http://127.0.0.1:8890/demo`。该模式只展示仓库内公开数据证据，页面会明确标记没有执行 Qwen、riceFM 或 NPU 实时推理。

### Ascend 真机演示

使用赛事平台当次提供的临时跳板凭据建立本地转发：

```text
ssh -N -L 8000:127.0.0.1:8000 -J <temporary-jump-user>@<jump-host>:<jump-port> root@<target-host>
```

然后打开 `http://127.0.0.1:8000/demo`。服务通过短期 HttpOnly 本机会话完成页面鉴权，不需要把长期 API Token 填入网页。临时 SSH 凭据和服务器 Token 不得写入提交材料。

## 平台表单填写建议

- **作品地址**：填写 GitHub 仓库地址。
- **演示地址**：若没有长期公网部署，不填写 localhost；填写演示视频地址，并注明“现场真机演示通过赛事临时 SSH 隧道访问”。
- **运行说明**：指向仓库 `README.md`、本文件和 `DEMO_GUIDE.md`。
- **技术贡献**：填写真实 MindSpeed-MM PR 地址。
- **附件**：上传项目说明 PDF、代码 ZIP、演示视频或平台允许的视频链接。

## 不应提交

- `jt_...` 跳板凭据、root 密码、API Token、私钥或 `/etc/qwen35.env`。
- 私有 ZH11 原始表达矩阵或未经数据方书面授权的逐细胞结果。
- Qwen/riceFM 模型权重、DCP 或训练 checkpoint 大文件。
- 将离线证据页面描述为实时 NPU 推理的材料。

## 提交前检查

1. GitHub 主分支包含本地最新修改，工作树干净。
2. README 中所有仓库、PR、数据来源和演示说明均为真实地址。
3. 新机器解压代码 ZIP 后，离线演示可以启动。
4. 真机演示依次通过 `/health`、riceFM embedding 和 Qwen 分析。
5. 视频画面中没有 SSH 密钥、Token、密码、私有数据或通知弹窗。
6. 项目说明中的性能、精度和完成状态均能指向仓库内对应日志或 JSON。
