# PaperBanana Manual Bridge

这是一个可独立运行的版本，专门用于“文本阶段自动/手动 + 可视化阶段手动出图上传”的 PaperBanana 流程，并可按需发布到 GitHub。

## 基于项目与致谢

- PaperBanana: https://github.com/dwzhu-pku/PaperBanana
- PaperVizAgent: https://github.com/google-research/papervizagent

感谢原作者团队提供完整的多智能体学术绘图框架。

## Documentation

Please read:

- Chinese: [docs/PROJECT_FLOW_ZH.md](docs/PROJECT_FLOW_ZH.md)
- English: [docs/PROJECT_FLOW_EN.md](docs/PROJECT_FLOW_EN.md)

文档包含：
- 项目完整工作流程
- 发布与合规要点（许可证/署名/变更声明）
- 结构图说明与使用步骤

## 安装与启动

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run tools/chat_bridge/web_app.py
```

## 数据准备（与原项目一致）

1. 从 HuggingFace 下载 `PaperBananaBench`。
2. 放到：`data/PaperBananaBench/`，结构示例：

```text
data/
  PaperBananaBench/
    diagram/
      ref.json
      images/
    plot/
      ref.json
      images/
```

## 可选：生成内置参考图库（本地）

```bash
python scripts/prepare_reference_gallery.py
```

复制目标：

`tools/chat_bridge/reference_gallery/PaperBananaBench/...`

## GitHub 上传建议

本项目默认保持轻量化提交：
- 忽略 `data/`
- 忽略 `results/`
- 忽略 `tools/chat_bridge/reference_gallery` 大图目录

标准上传命令：

```bash
git init
git add .
git commit -m "init manual-bridge project"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

## License

Apache-2.0（见 `LICENSE`）
