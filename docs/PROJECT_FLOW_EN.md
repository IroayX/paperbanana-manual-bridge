# Project Guide (English)

## 1. Project Positioning

This project is a **Manual Visualizer Bridge** variant of PaperBanana:
- Text stages (Retriever / Planner / Stylist / Critic) can be run manually or via Text API.
- The Visualizer stage is manual: generate images with your own chat/agent model, then upload back.
- The original multi-stage state machine and iterative loop are preserved.
- It targets users who only have chat UI access, or APIs that support text but not image generation.

## 2. Credits and Upstream Links

This project is adapted from:
- **PaperBanana**: https://github.com/dwzhu-pku/PaperBanana
- **PaperVizAgent** (upstream mentioned by PaperBanana): https://github.com/google-research/papervizagent

Thanks to the original authors for the open-source multi-agent illustration framework.

## 3. Project Structure Diagram

The following figure is the overall workflow diagram you provided:

![Manual Visualizer Workflow](assets/visualizer_workflow_diagram.png)

## 4. End-to-End Workflow

### Stage Order

- Retriever (optional)
- Planner
- Stylist (`demo_full` mode)
- Visualizer (manual image generation + upload)
- Critic
- Repeat until completion

### Core Logic

1. Initialize run with task settings, retrieval mode, rounds, and interaction mode (`chat_only` or `agent`).
2. Retriever stage:
   - `auto`: one-click API loop for all chunks + final top10.
   - `manual`: paste top10 JSON manually.
3. Planner / Stylist / Critic can run via Text API Assist.
4. Visualizer stage:
   - system outputs strict prompt;
   - you generate image externally;
   - upload image back to continue.
5. Critic returns JSON (`critic_suggestions`, `revised_description`), then state machine enters next visualizer round.

## 5. Quick Start

1. Install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Start web app:

```bash
streamlit run tools/chat_bridge/web_app.py
```

3. Dataset setup (same strategy as upstream):
- download from: https://huggingface.co/datasets/dwzhu/PaperBananaBench
- place `PaperBananaBench` under `data/PaperBananaBench/`.

4. Optional: prepare embedded local gallery:

```bash
python scripts/prepare_reference_gallery.py
```
