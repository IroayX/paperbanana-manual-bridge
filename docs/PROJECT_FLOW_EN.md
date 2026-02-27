# Project Guide (English)

## 1. Project Positioning

This project is a **Manual Visualizer Bridge** variant of PaperBanana:
- Text stages (Retriever / Planner / Stylist / Critic) can be run manually or via Text API.
- The Visualizer stage is manual: generate images with your own chat/agent model, then upload back.
- The original multi-stage state machine and iterative loop are preserved.

## 2. Credits and Upstream Links

This project is adapted from:
- **PaperBanana**: https://github.com/dwzhu-pku/PaperBanana
- **PaperVizAgent** (upstream mentioned by PaperBanana): https://github.com/google-research/papervizagent

Thanks to the original authors for the open-source multi-agent illustration framework.

## 3. Release and Compliance Notes

This repository keeps `LICENSE` (Apache-2.0). Before publishing to GitHub, make sure:
1. Keep original license and copyright notices.
2. Clearly state upstream sources in README/docs.
3. Clearly state your modifications.
4. Never commit private keys, private data, or unauthorized assets.

## 4. Project Structure Diagram

The following figure is the overall workflow diagram you provided:

![Manual Visualizer Workflow](assets/visualizer_workflow_diagram.png)

## 5. End-to-End Workflow

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

## 6. Quick Start

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
- place `PaperBananaBench` under `data/PaperBananaBench/`.

4. Optional: prepare embedded local gallery:

```bash
python scripts/prepare_reference_gallery.py
```

## 7. Safety and Publishing Tips

- Do not commit real API keys.
- Keep `results/` clean (`.gitkeep` only).
- Ensure `.gitignore` is effective before commit.
- Keep source links and acknowledgements in public docs.
- Reference gallery images are large. If you must commit them, prefer Git LFS.
