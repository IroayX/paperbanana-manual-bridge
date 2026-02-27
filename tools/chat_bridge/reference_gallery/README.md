# Embedded Reference Gallery (Optional)

This folder is used by the chat-bridge web app as an optional local reference gallery root:

`tools/chat_bridge/reference_gallery/PaperBananaBench/{diagram,plot}`

By default this repository **does not** commit large reference images to GitHub.

If you already have `data/PaperBananaBench` locally, run:

```bash
python scripts/prepare_reference_gallery.py
```

This will copy `diagram/plot` `ref.json` and `images/` into this folder for local use.
