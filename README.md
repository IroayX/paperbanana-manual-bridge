# PaperBanana Manual Bridge

PaperBanana Manual Bridge is an independent variant of the PaperBanana pipeline.
It keeps text-stage automation/manual control (Retriever / Planner / Stylist / Critic),
while making the Visualizer stage manual (you generate externally, then upload back).

## Project Purpose

This project is designed for users who:

- only have a chat interface (no direct image API workflow), or
- only have text-capable API access.

It splits the original PaperBanana workflow so that:

- text-heavy stages still follow the same pipeline/state-machine logic,
- image generation is done manually outside, then uploaded back for iterative refinement.

## Credits

This project is adapted from:

- PaperBanana: https://github.com/dwzhu-pku/PaperBanana
- PaperVizAgent: https://github.com/google-research/papervizagent

Thanks to the original authors for open-sourcing the multi-agent academic illustration workflow.

## Documentation

- English guide: [docs/PROJECT_FLOW_EN.md](docs/PROJECT_FLOW_EN.md)
- Chinese guide: [docs/PROJECT_FLOW_ZH.md](docs/PROJECT_FLOW_ZH.md)

## Quick Start

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run tools/chat_bridge/web_app.py
```

## Dataset Setup (same strategy as upstream)

1. Download `PaperBananaBench` from HuggingFace:
   - https://huggingface.co/datasets/dwzhu/PaperBananaBench
2. Place it under `data/PaperBananaBench/`:

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

## Optional: Build Embedded Local Reference Gallery

```bash
python scripts/prepare_reference_gallery.py
```

This copies data into:

`tools/chat_bridge/reference_gallery/PaperBananaBench/...`

## License

Apache-2.0 (see `LICENSE`)
