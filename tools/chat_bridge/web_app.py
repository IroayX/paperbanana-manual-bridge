from __future__ import annotations

import json
import io
import zipfile
import re
import base64
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
import json_repair
import httpx
from PIL import Image

from tools.chat_bridge.session import (
    build_next_prompt,
    get_status,
    init_run,
    load_input,
    load_state,
    submit_image_for_visualizer,
    submit_text_output,
)

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    from google import genai as google_genai
except Exception:  # pragma: no cover
    google_genai = None  # type: ignore

try:
    from anthropic import Anthropic
except Exception:  # pragma: no cover
    Anthropic = None  # type: ignore


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = ROOT_DIR / "results" / "chat_bridge_web_run"
EMBEDDED_REF_ROOT = ROOT_DIR / "tools" / "chat_bridge" / "reference_gallery" / "PaperBananaBench"
DEFAULT_REF_ROOT = ROOT_DIR / "data" / "PaperBananaBench"

TEXT_API_PRESETS = {
    "ClaudeCode Proxy": {
        "kind": "openai_compat",
        "base_url": "https://api.aicodemirror.com/api/claudecode",
    },
    "Codex Proxy": {
        "kind": "openai_compat",
        "base_url": "https://api.aicodemirror.com/api/codex/backend-api/codex",
    },
    "Gemini Proxy": {
        "kind": "openai_compat",
        "base_url": "https://api.aicodemirror.com/api/gemini",
    },
    "OpenAI Official": {
        "kind": "openai_compat",
        "base_url": "https://api.openai.com/v1",
    },
    "Gemini Official": {
        "kind": "gemini_official",
        "base_url": "",
    },
    "Anthropic Official": {
        "kind": "anthropic_official",
        "base_url": "",
    },
    "Custom (OpenAI-Compatible)": {
        "kind": "openai_compat",
        "base_url": "",
    },
}

PRESET_FALLBACK_MODELS = {
    "Gemini Proxy": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"],
    "ClaudeCode Proxy": [
        "claude-opus-4-1",
        "claude-sonnet-4-5",
        "claude-3-7-sonnet-latest",
    ],
    "Codex Proxy": ["gpt-5", "gpt-5-mini", "o3", "gpt-4.1"],
    "OpenAI Official": ["gpt-5", "gpt-5-mini", "o3", "gpt-4.1"],
    "Gemini Official": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-pro"],
    "Anthropic Official": [
        "claude-opus-4-1",
        "claude-sonnet-4-5",
        "claude-3-7-sonnet-latest",
    ],
    "Custom (OpenAI-Compatible)": [],
}


def _load_builtin_example_from_demo() -> tuple[str, str]:
    demo_file = ROOT_DIR / "demo.py"
    if not demo_file.exists():
        raise FileNotFoundError(f"demo.py not found: {demo_file}")
    text = demo_file.read_text(encoding="utf-8")
    method_match = re.search(r'example_method\s*=\s*r"""(.*?)"""', text, re.S)
    caption_match = re.search(
        r'example_caption\s*=\s*"([\s\S]*?)"\n\s*\n\s*col_input1', text
    )
    if not method_match or not caption_match:
        raise RuntimeError("Failed to parse built-in example from demo.py")
    return method_match.group(1), caption_match.group(1)


def _safe_decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


def _text_api_state() -> dict[str, Any]:
    return st.session_state.get("text_api_config", {})


def _save_text_api_state(payload: dict[str, Any]) -> None:
    st.session_state["text_api_config"] = payload


def _is_text_model_name(name: str) -> bool:
    lowered = name.lower()
    blocked = ["image", "embedding", "tts", "whisper", "moderation", "vision-preview"]
    return not any(x in lowered for x in blocked)


def _model_score(name: str) -> int:
    lowered = name.lower()
    priority = [
        "gpt-5",
        "o3",
        "o1",
        "claude-opus-4",
        "claude-sonnet-4",
        "claude-3.7",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gpt-4.1",
        "gpt-4o",
        "claude-3.5",
        "gemini-1.5-pro",
        "deepseek-reasoner",
        "deepseek-chat",
    ]
    for idx, tag in enumerate(priority):
        if tag in lowered:
            return 1000 - idx
    return 0


def _pick_best_model(models: list[str]) -> str:
    text_models = [m for m in models if _is_text_model_name(m)]
    if not text_models:
        return models[0] if models else ""
    ranked = sorted(text_models, key=lambda x: (_model_score(x), x), reverse=True)
    return ranked[0]


def _extract_model_ids(payload: Any) -> list[str]:
    models: list[str] = []
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("id"):
                    models.append(str(item["id"]))
                elif isinstance(item, str):
                    models.append(item)
        alt = payload.get("models")
        if isinstance(alt, list):
            for item in alt:
                if isinstance(item, dict) and item.get("id"):
                    models.append(str(item["id"]))
                elif isinstance(item, str):
                    models.append(item)
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and item.get("id"):
                models.append(str(item["id"]))
            elif isinstance(item, str):
                models.append(item)
    return sorted(set([m for m in models if str(m).strip()]))


def _fetch_openai_compat_models_http(base_url: str, api_key: str) -> list[str]:
    base = (base_url or "").rstrip("/")
    if not base:
        raise ValueError("Base URL is required for openai-compatible providers")
    headers = {"Authorization": f"Bearer {api_key}"}
    candidates = [f"{base}/models", f"{base}/v1/models"]
    errors = []
    for url in candidates:
        try:
            with httpx.Client(timeout=20) as client:
                resp = client.get(url, headers=headers)
            if resp.status_code >= 400:
                errors.append(f"{url}: HTTP {resp.status_code}")
                continue
            try:
                data = resp.json()
            except Exception:
                errors.append(f"{url}: non-JSON response")
                continue
            models = _extract_model_ids(data)
            if models:
                return models
            errors.append(f"{url}: no model ids in response schema")
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError(" ; ".join(errors))


def _extract_openai_compat_text(payload: Any) -> str:
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return ""
        lower = text.lower()
        if lower.startswith("<!doctype html") or lower.startswith("<html"):
            return ""
        if text.startswith("{") or text.startswith("["):
            try:
                parsed = json.loads(text)
                return _extract_openai_compat_text(parsed)
            except Exception:
                return text
        return text

    if isinstance(payload, dict):
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        return content.strip()
                    if isinstance(content, list):
                        chunks = []
                        for part in content:
                            if not isinstance(part, dict):
                                continue
                            if isinstance(part.get("text"), str):
                                chunks.append(part["text"])
                        if chunks:
                            return "\n".join(chunks).strip()
                if isinstance(first.get("text"), str):
                    return first["text"].strip()

        if isinstance(payload.get("output_text"), str):
            return payload["output_text"].strip()

        output = payload.get("output")
        if isinstance(output, list):
            chunks = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            chunks.append(part["text"])
            if chunks:
                return "\n".join(chunks).strip()

        if isinstance(payload.get("text"), str):
            return payload["text"].strip()

    return ""


def _call_openai_compat_http(
    prompt: str, cfg: dict[str, Any], *, image_bytes: bytes | None = None
) -> str:
    base_url = str(cfg.get("base_url", "") or "").rstrip("/")
    api_key = str(cfg.get("api_key", "") or "")
    model = str(cfg.get("model", "") or "")
    temperature = float(cfg.get("temperature", 0))
    max_tokens_unlimited = bool(cfg.get("max_tokens_unlimited", False))
    max_tokens_raw = cfg.get("max_tokens", 4096)
    max_tokens = None if max_tokens_unlimited else int(max_tokens_raw)

    if not base_url:
        raise ValueError("Base URL is required")
    if not api_key:
        raise ValueError("Missing API key")
    if not model:
        raise ValueError("Missing model")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    user_content: Any = prompt
    if image_bytes:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        user_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]

    req: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "temperature": temperature,
    }
    if max_tokens is not None:
        req["max_tokens"] = max_tokens

    endpoints = [
        f"{base_url}/chat/completions",
        f"{base_url}/v1/chat/completions",
    ]
    errors: list[str] = []
    for url in endpoints:
        try:
            with httpx.Client(timeout=120) as client:
                resp = client.post(url, headers=headers, json=req)
            if resp.status_code >= 400:
                body = resp.text[:300].replace("\n", " ")
                errors.append(f"{url}: HTTP {resp.status_code} {body}")
                continue
            try:
                data = resp.json()
            except Exception:
                data = resp.text
            text = _extract_openai_compat_text(data)
            if text:
                return text
            errors.append(f"{url}: empty/unsupported response schema")
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError(" ; ".join(errors))


def _list_models_via_api(cfg: dict[str, Any]) -> list[str]:
    kind = cfg.get("kind", "")
    api_key = cfg.get("api_key", "")
    base_url = cfg.get("base_url", "")
    if not api_key:
        raise ValueError("Missing API key")

    if kind == "openai_compat":
        # Use raw HTTP listing for better compatibility with proxy providers.
        return _fetch_openai_compat_models_http(base_url=base_url, api_key=api_key)

    if kind == "gemini_official":
        if google_genai is None:
            raise RuntimeError("google-genai package is unavailable")
        client = google_genai.Client(api_key=api_key)
        out = []
        for m in client.models.list():
            name = getattr(m, "name", "")
            if name:
                out.append(name.replace("models/", ""))
        return sorted(set(out))

    if kind == "anthropic_official":
        return [
            "claude-sonnet-4-5",
            "claude-opus-4-1",
            "claude-3-7-sonnet-latest",
        ]

    return []


def _call_text_api(
    prompt: str, cfg: dict[str, Any], *, image_bytes: bytes | None = None
) -> str:
    kind = cfg.get("kind", "")
    api_key = cfg.get("api_key", "")
    model = cfg.get("model", "")
    base_url = cfg.get("base_url", "")
    temperature = float(cfg.get("temperature", 0))
    max_tokens_unlimited = bool(cfg.get("max_tokens_unlimited", False))
    max_tokens_raw = cfg.get("max_tokens", 4096)
    max_tokens = None if max_tokens_unlimited else int(max_tokens_raw)

    if not api_key:
        raise ValueError("Missing API key")
    if not model:
        raise ValueError("Missing model name")

    if kind == "openai_compat":
        return _call_openai_compat_http(
            prompt=prompt, cfg=cfg, image_bytes=image_bytes
        ).strip()

    if kind == "gemini_official":
        if google_genai is None:
            raise RuntimeError("google-genai package is unavailable")
        client = google_genai.Client(api_key=api_key)
        contents: Any = prompt
        if image_bytes:
            contents = [prompt, google_genai.types.Part.from_bytes(data=image_bytes, mime_type="image/png")]
        resp = client.models.generate_content(model=model, contents=contents)
        text = getattr(resp, "text", "")
        return str(text or "").strip()

    if kind == "anthropic_official":
        if Anthropic is None:
            raise RuntimeError("anthropic package is unavailable")
        client = Anthropic(api_key=api_key)
        # Anthropic requires max_tokens; if "unlimited", use a large safe cap.
        anthropic_max_tokens = max_tokens if max_tokens is not None else 8192
        content_blocks: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image_bytes:
            content_blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(image_bytes).decode("utf-8"),
                    },
                }
            )
        resp = client.messages.create(
            model=model,
            temperature=temperature,
            max_tokens=int(anthropic_max_tokens),
            messages=[{"role": "user", "content": content_blocks}],
        )
        chunks = []
        for block in getattr(resp, "content", []):
            txt = getattr(block, "text", "")
            if txt:
                chunks.append(txt)
        return "\n".join(chunks).strip()

    raise ValueError(f"Unsupported API kind: {kind}")


def _render_status(run_dir: Path) -> dict[str, Any]:
    status = get_status(run_dir)
    st.subheader("Current Stage")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("stage", status.get("stage", ""))
    col2.metric("round", str(status.get("current_round", 0)))
    col3.metric("completed", str(status.get("completed", False)))
    col4.metric("stop_reason", status.get("stop_reason", "") or "-")
    return status


def _render_stage_order(run_dir: Path, current_stage: str) -> None:
    try:
        state = load_state(run_dir)
    except Exception:
        return

    order = []
    if state.get("retrieval_setting") != "none":
        order.append("retriever")
    order.append("planner")
    if state.get("exp_mode") == "demo_full":
        order.append("stylist")
    order.extend(["visualizer", "critic"])

    styled = []
    for item in order:
        if item == current_stage:
            styled.append(f"[{item.upper()}]")
        else:
            styled.append(item)
    st.caption("Pipeline Order: " + " -> ".join(styled))


def _render_text_api_sidebar() -> dict[str, Any]:
    st.sidebar.header("Text API Assist")
    enabled = st.sidebar.checkbox(
        "Enable Text API Assist",
        value=bool(_text_api_state().get("enabled", False)),
    )
    preset_names = list(TEXT_API_PRESETS.keys())
    default_preset = _text_api_state().get("preset", "Gemini Proxy")
    if default_preset not in preset_names:
        default_preset = "Gemini Proxy"
    preset = st.sidebar.selectbox(
        "Preset",
        preset_names,
        index=preset_names.index(default_preset),
    )
    preset_cfg = TEXT_API_PRESETS[preset]
    kind = preset_cfg["kind"]

    default_base = _text_api_state().get("base_url", preset_cfg.get("base_url", ""))
    if preset != "Custom (OpenAI-Compatible)":
        default_base = preset_cfg.get("base_url", "")
    base_url = st.sidebar.text_input("Base URL", value=default_base)
    api_key = st.sidebar.text_input(
        "API Key",
        value=_text_api_state().get("api_key", ""),
        type="password",
    )

    col_m1, col_m2 = st.sidebar.columns(2)
    if col_m1.button("Load Models", use_container_width=True):
        try:
            models = _list_models_via_api(
                {
                    "kind": kind,
                    "api_key": api_key,
                    "base_url": base_url,
                }
            )
            st.session_state["text_api_models"] = models
            if models:
                st.session_state["text_api_model"] = _pick_best_model(models)
            st.sidebar.success(f"Loaded {len(models)} models")
        except Exception as exc:
            fallback = PRESET_FALLBACK_MODELS.get(preset, [])
            if fallback:
                st.session_state["text_api_models"] = fallback
                st.session_state["text_api_model"] = _pick_best_model(fallback)
                st.sidebar.warning(
                    f"Load models failed: {exc}\nUsing preset fallback models."
                )
            else:
                st.sidebar.error(f"Load models failed: {exc}")

    if col_m2.button("Best Model", use_container_width=True):
        models = st.session_state.get("text_api_models", [])
        if models:
            st.session_state["text_api_model"] = _pick_best_model(models)

    models = st.session_state.get("text_api_models", [])
    model_default = st.session_state.get("text_api_model", _text_api_state().get("model", ""))

    if models:
        if model_default not in models:
            model_default = _pick_best_model(models)
        model = st.sidebar.selectbox(
            "Model",
            models,
            index=models.index(model_default),
        )
    else:
        model = st.sidebar.text_input("Model", value=model_default)

    temperature = st.sidebar.number_input(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=float(_text_api_state().get("temperature", 0.0)),
        step=0.1,
    )
    saved_unlimited = bool(_text_api_state().get("max_tokens_unlimited", False))
    max_tokens_unlimited = st.sidebar.checkbox(
        "Max Tokens Unlimited",
        value=saved_unlimited,
    )
    saved_tokens_raw = _text_api_state().get("max_tokens", 4096)
    try:
        saved_tokens = int(saved_tokens_raw if saved_tokens_raw is not None else 4096)
    except Exception:
        saved_tokens = 4096
    if not max_tokens_unlimited:
        max_tokens = st.sidebar.number_input(
            "Max Tokens",
            min_value=256,
            max_value=32768,
            value=saved_tokens,
            step=256,
        )
    else:
        max_tokens = None
        st.sidebar.caption(
            "Unlimited mode: no max_tokens sent where provider allows."
        )

    cfg = {
        "enabled": enabled,
        "preset": preset,
        "kind": kind,
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "temperature": float(temperature),
        "max_tokens_unlimited": bool(max_tokens_unlimited),
        "max_tokens": (int(max_tokens) if max_tokens is not None else None),
    }
    _save_text_api_state(cfg)
    return cfg


def _render_prompt_box(run_dir: Path, stage: str) -> None:
    st.subheader("Stage Prompt")
    if st.button("Generate Prompt", use_container_width=True):
        prompt = build_next_prompt(run_dir)
        prompt_path = run_dir / f"next_prompt_{stage}.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        st.success(f"Saved: {prompt_path}")
        st.code(prompt, language="text")
        st.session_state["last_prompt"] = prompt
        st.session_state["last_prompt_stage"] = stage
    elif st.session_state.get("last_prompt"):
        if st.session_state.get("last_prompt_stage") == stage:
            st.code(st.session_state["last_prompt"], language="text")
        else:
            st.caption("Prompt cache belongs to another stage. Click `Generate Prompt`.")
    else:
        st.caption("Click `Generate Prompt` to produce the exact stage prompt.")


def _iter_reference_roots(state: dict[str, Any]) -> list[Path]:
    roots: list[Path] = []
    explicit = str(state.get("reference_gallery_dir", "")).strip()
    if explicit:
        roots.append(Path(explicit))

    work_dir = Path(state.get("work_dir", str(ROOT_DIR)))
    roots.extend(
        [
            work_dir / "data" / "PaperBananaBench",
            DEFAULT_REF_ROOT,
            EMBEDDED_REF_ROOT,
            ROOT_DIR / "reference_gallery" / "PaperBananaBench",
        ]
    )

    seen: set[str] = set()
    out: list[Path] = []
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key in seen:
            continue
        seen.add(key)
        out.append(root)
    return out


def _resolve_ref_task_dir(state: dict[str, Any]) -> Path:
    task_name = str(state.get("task_name", "diagram"))
    roots = _iter_reference_roots(state)
    for root in roots:
        task_dir = root / task_name
        if (task_dir / "ref.json").exists():
            return task_dir
    if roots:
        return roots[0] / task_name
    return DEFAULT_REF_ROOT / task_name


def _resolve_ref_json_path(state: dict[str, Any]) -> Path:
    return _resolve_ref_task_dir(state) / "ref.json"


def _collect_reference_images(run_dir: Path) -> list[dict[str, str]]:
    state = load_state(run_dir)
    ids = [str(x) for x in state.get("top10_references", []) if str(x).strip()]
    if not ids:
        return []
    ref_path = _resolve_ref_json_path(state)
    if not ref_path.exists():
        return []
    pool = json.loads(ref_path.read_text(encoding="utf-8"))
    id_map = {str(item.get("id", "")): item for item in pool}
    base_dir = _resolve_ref_task_dir(state)

    out: list[dict[str, str]] = []
    for ref_id in ids:
        item = id_map.get(ref_id)
        if not item:
            continue
        rel = str(item.get("path_to_gt_image", "")).strip()
        if not rel:
            continue
        abs_path = base_dir / rel
        if not abs_path.exists():
            continue
        out.append({"id": ref_id, "path": str(abs_path), "rel": rel})
    return out


def _copy_reference_gallery(src_root: Path, dst_root: Path) -> None:
    for task in ("diagram", "plot"):
        src_task = src_root / task
        if not src_task.exists():
            raise FileNotFoundError(f"Missing source task directory: {src_task}")
        dst_task = dst_root / task
        dst_task.mkdir(parents=True, exist_ok=True)

        src_ref = src_task / "ref.json"
        if not src_ref.exists():
            raise FileNotFoundError(f"Missing source ref.json: {src_ref}")
        shutil.copy2(src_ref, dst_task / "ref.json")

        src_images = src_task / "images"
        if not src_images.exists():
            raise FileNotFoundError(f"Missing source images directory: {src_images}")
        dst_images = dst_task / "images"
        if dst_images.exists():
            shutil.rmtree(dst_images)
        shutil.copytree(src_images, dst_images)


def _render_reference_gallery_panel(task_name: str) -> None:
    st.sidebar.markdown("### Reference Gallery")
    active_ref = (
        EMBEDDED_REF_ROOT
        if (EMBEDDED_REF_ROOT / task_name / "ref.json").exists()
        else DEFAULT_REF_ROOT
    )
    st.sidebar.caption(f"Active preferred root: `{active_ref}`")
    st.sidebar.caption(
        "Resolution fallback: explicit config -> work_dir/data -> project/data -> embedded/tools/chat_bridge."
    )

    src_ready = all((DEFAULT_REF_ROOT / t / "ref.json").exists() for t in ("diagram", "plot"))
    embedded_ready = all((EMBEDDED_REF_ROOT / t / "ref.json").exists() for t in ("diagram", "plot"))
    st.sidebar.write(f"Project data ready: {'Yes' if src_ready else 'No'}")
    st.sidebar.write(f"Embedded data ready: {'Yes' if embedded_ready else 'No'}")

    if st.sidebar.button("Embed Gallery Into Project", use_container_width=True):
        try:
            _copy_reference_gallery(DEFAULT_REF_ROOT, EMBEDDED_REF_ROOT)
            st.sidebar.success(f"Embedded gallery updated: {EMBEDDED_REF_ROOT}")
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"Embed failed: {exc}")


def _pick_preferred_reference_root(task_name: str) -> Path:
    if (EMBEDDED_REF_ROOT / task_name / "ref.json").exists():
        return EMBEDDED_REF_ROOT
    return DEFAULT_REF_ROOT


def _build_reference_zip_bytes(refs: list[dict[str, str]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in refs:
            src = Path(item["path"])
            arc = f"{item['id']}_{src.name}"
            zf.write(src, arcname=arc)
    buf.seek(0)
    return buf.read()


def _render_planner_reference_panel(run_dir: Path) -> None:
    refs = _collect_reference_images(run_dir)
    st.subheader("Planner Reference Images")
    if not refs:
        st.warning("No local reference images found for current top10_references.")
        return

    st.info(
        "For strict parity with original pipeline, send planner prompt together with these reference images."
    )
    st.caption(f"Found {len(refs)} images linked by retriever top10 IDs.")

    zip_bytes = _build_reference_zip_bytes(refs)
    st.download_button(
        label="Download Reference Image Pack (.zip)",
        data=zip_bytes,
        file_name="planner_reference_images.zip",
        mime="application/zip",
        use_container_width=True,
    )

    with st.expander("Reference IDs and Paths", expanded=False):
        st.table([{"id": x["id"], "path": x["path"]} for x in refs])

    with st.expander("Preview", expanded=False):
        cols = st.columns(2)
        for i, item in enumerate(refs[:10]):
            with cols[i % 2]:
                st.image(item["path"], caption=item["id"], use_container_width=True)

    chat_helper = (
        "[MULTIMODAL INPUT PROTOCOL - CHAT MODEL]\n"
        "I will provide reference images in one or more upload batches.\n"
        "Do not produce any final answer until I explicitly send: [ALL_REFERENCES_UPLOADED].\n"
        "For each batch, only reply: ACK_RECEIVED_BATCH.\n"
        "After [ALL_REFERENCES_UPLOADED], read all uploaded images together with the planner prompt,\n"
        "then output only the final planner description text.\n"
    )
    agent_paths = "\n".join([f"- {x['id']}: {x['path']}" for x in refs])
    agent_helper = (
        "[MULTIMODAL INPUT PROTOCOL - AGENT MODEL]\n"
        "Read all local reference images below before answering.\n"
        "Do not output final content until all images are read.\n"
        "Then process planner prompt jointly with these images and output only final planner description text.\n\n"
        "Reference image paths:\n"
        f"{agent_paths}\n"
    )

    st.markdown("**Helper Prompt (Chat Model Version)**")
    st.code(chat_helper, language="text")
    st.markdown("**Helper Prompt (Agent Version)**")
    st.code(agent_helper, language="text")


def _submit_text_area(run_dir: Path) -> None:
    st.subheader("Submit Text Output")
    current_stage = ""
    planner_requires_ref_confirm = False
    try:
        st_state = load_state(run_dir)
        current_stage = str(get_status(run_dir).get("stage", ""))
        has_refs = len(st_state.get("top10_references", [])) > 0
        planner_requires_ref_confirm = current_stage == "planner" and has_refs
    except Exception:
        planner_requires_ref_confirm = False

    if planner_requires_ref_confirm:
        st.info(
            "Planner parity gate: please ensure top reference images were provided to the model "
            "(uploaded in chat_only mode or read via local paths in agent mode)."
        )
        planner_ref_confirm = st.checkbox(
            "I confirm planner stage has read the top reference images",
            value=False,
            key="planner_ref_confirm",
        )
    else:
        planner_ref_confirm = True

    upload = st.file_uploader(
        "Upload model output text/json",
        type=["txt", "md", "json"],
        key="submit_text_file",
    )
    manual_text = st.text_area(
        "Or paste output text",
        value="",
        height=220,
        key="submit_text_manual",
    )
    if st.button("Submit Text", use_container_width=True):
        if not planner_ref_confirm:
            st.error("Please confirm planner reference images were provided before submit.")
            return
        payload = ""
        if upload is not None:
            payload = _safe_decode(upload.getvalue())
        elif manual_text.strip():
            payload = manual_text

        if not payload.strip():
            st.error("No text payload provided.")
            return

        try:
            submit_text_output(run_dir, payload)
            st.success("Text submitted. Stage advanced by state machine.")
            st.rerun()
        except Exception as exc:
            st.error(f"Submit failed: {exc}")


def _submit_retriever_area(run_dir: Path) -> None:
    st.subheader("Submit Retriever Output")
    try:
        state = load_state(run_dir)
    except Exception as exc:
        st.error(f"Cannot load state: {exc}")
        return

    task_name = state.get("task_name", "diagram")
    retrieval_setting = state.get("retrieval_setting", "none")
    key = "top10_diagrams" if task_name == "diagram" else "top10_plots"

    if retrieval_setting == "auto":
        auto = state.get("retrieval_auto", {})
        phase = auto.get("phase", "final")
        if phase == "chunk":
            chunk_idx = int(auto.get("chunk_index", 0)) + 1
            total_chunks = int(auto.get("total_chunks", 1))
            st.info(
                f"Auto retrieval chunk phase: {chunk_idx}/{total_chunks}. "
                'Expected JSON key: "top3_ids".'
            )
            st.code('{"top3_ids":["ref_1","ref_2","ref_3"]}', language="json")
        else:
            st.info(
                f"Auto retrieval final phase. Expected JSON key: \"{key}\"."
            )
            st.code(
                json.dumps({key: ["ref_1", "ref_2", "..."]}, ensure_ascii=False, indent=2),
                language="json",
            )
    elif retrieval_setting == "manual":
        st.info(f"Manual retrieval mode. Expected JSON key: \"{key}\".")
        st.code(
            json.dumps({key: ["ref_1", "ref_2", "..."]}, ensure_ascii=False, indent=2),
            language="json",
        )
        ids_str = st.text_input(
            "Quick fill IDs (comma-separated, optional)",
            value="",
            key="retriever_ids_quickfill",
        )
        if st.button("Submit Quick IDs", use_container_width=True):
            ids = [x.strip() for x in ids_str.split(",") if x.strip()]
            if not ids:
                st.error("No IDs provided.")
            else:
                payload = json.dumps({key: ids}, ensure_ascii=False)
                try:
                    submit_text_output(run_dir, payload)
                    st.success("Retriever IDs submitted.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Submit failed: {exc}")
    else:
        st.warning(
            "retrieval_setting is none, retriever stage should not appear in this run."
        )

    upload = st.file_uploader(
        "Upload retriever output text/json",
        type=["txt", "json", "md"],
        key="submit_retriever_file",
    )
    manual_text = st.text_area(
        "Or paste retriever output JSON",
        value="",
        height=180,
        key="submit_retriever_manual",
    )
    if st.button("Submit Retriever Output", use_container_width=True):
        payload = ""
        if upload is not None:
            payload = _safe_decode(upload.getvalue())
        elif manual_text.strip():
            payload = manual_text
        if not payload.strip():
            st.error("No retriever payload provided.")
            return
        try:
            latest_state = load_state(run_dir)
            latest_auto = latest_state.get("retrieval_auto", {})
            latest_mode = latest_state.get("retrieval_setting", "none")
            expected_key = "top10_diagrams" if task_name == "diagram" else "top10_plots"
            if latest_mode == "auto" and latest_auto.get("phase") == "chunk":
                expected_key = "top3_ids"
            parsed = json_repair.loads(payload.strip())
            if not isinstance(parsed, dict) or expected_key not in parsed:
                st.error(
                    f"Payload key mismatch. Current phase expects `{expected_key}`. "
                    "Please click `Generate Prompt` and follow its exact JSON contract."
                )
                return
        except Exception as exc:
            st.error(f"Payload parse failed: {exc}")
            return
        try:
            submit_text_output(run_dir, payload)
            st.success("Retriever output submitted.")
            st.rerun()
        except Exception as exc:
            st.error(f"Submit failed: {exc}")


def _get_or_build_stage_prompt(run_dir: Path, stage: str, *, force_refresh: bool = False) -> str:
    cached_prompt = st.session_state.get("last_prompt", "")
    cached_stage = st.session_state.get("last_prompt_stage", "")
    if (not force_refresh) and cached_prompt and cached_stage == stage:
        return str(cached_prompt)
    prompt = build_next_prompt(run_dir)
    prompt_path = run_dir / f"next_prompt_{stage}.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    st.session_state["last_prompt"] = prompt
    st.session_state["last_prompt_stage"] = stage
    return prompt


def _get_latest_visualizer_image_bytes_for_critic(run_dir: Path) -> bytes | None:
    try:
        state = load_state(run_dir)
    except Exception:
        return None
    round_idx = int(state.get("current_round", 0))
    rounds = state.get("rounds", [])
    if round_idx < 0 or round_idx >= len(rounds):
        return None
    rel = str(rounds[round_idx].get("visualizer_image_rel_path", "")).strip()
    if not rel:
        return None
    path = run_dir / rel
    if not path.exists() or not path.is_file():
        return None
    try:
        return path.read_bytes()
    except Exception:
        return None


def _parse_ratio_value(ratio_text: str) -> float | None:
    txt = str(ratio_text or "").strip()
    if not txt:
        return None
    if ":" in txt:
        parts = txt.split(":")
        if len(parts) != 2:
            return None
        try:
            a = float(parts[0].strip())
            b = float(parts[1].strip())
            if a <= 0 or b <= 0:
                return None
            return a / b
        except Exception:
            return None
    try:
        v = float(txt)
        return v if v > 0 else None
    except Exception:
        return None


def _check_image_specs(image_bytes: bytes, opts: dict[str, Any]) -> tuple[dict[str, int], list[str]]:
    issues: list[str] = []
    with Image.open(io.BytesIO(image_bytes)) as im:
        width, height = int(im.width), int(im.height)
    info = {"width": width, "height": height}

    expected_ratio = _parse_ratio_value(str(opts.get("aspect_ratio", "")))
    if expected_ratio and height > 0:
        actual_ratio = width / height
        # Allow 3% ratio deviation.
        if abs(actual_ratio - expected_ratio) / expected_ratio > 0.03:
            issues.append(
                f"Aspect ratio mismatch: expected {opts.get('aspect_ratio')} but got {width}:{height}."
            )

    size_map = {"1K": 900, "2K": 1700, "4K": 3000}
    expected_size = str(opts.get("image_size", "")).upper()
    min_long_edge = size_map.get(expected_size)
    if min_long_edge:
        long_edge = max(width, height)
        if long_edge < min_long_edge:
            issues.append(
                f"Resolution too small for {expected_size}: long edge {long_edge}px < required {min_long_edge}px."
            )

    return info, issues


def _run_retriever_auto_via_api(run_dir: Path, cfg: dict[str, Any], *, max_steps: int = 512) -> dict[str, Any]:
    traces: list[dict[str, Any]] = []
    last_output = ""

    for i in range(max_steps):
        status = get_status(run_dir)
        if status.get("stage") != "retriever":
            break

        state = load_state(run_dir)
        if state.get("retrieval_setting") != "auto":
            raise ValueError("Auto retriever loop only applies to retrieval_setting=auto")

        auto = state.get("retrieval_auto", {})
        phase = str(auto.get("phase", "final"))
        chunk_index = int(auto.get("chunk_index", 0))
        total_chunks = int(auto.get("total_chunks", 1))

        prompt = _get_or_build_stage_prompt(run_dir, "retriever", force_refresh=True)
        response_text = _call_text_api(prompt, cfg).strip()
        if not response_text:
            raise RuntimeError(f"Empty API response at retriever iteration {i + 1}")

        submit_text_output(run_dir, response_text)
        last_output = response_text

        item: dict[str, Any] = {
            "iteration": i + 1,
            "phase": phase,
        }
        if phase == "chunk":
            item["chunk"] = f"{chunk_index + 1}/{total_chunks}"
        traces.append(item)

    final_status = get_status(run_dir)
    if final_status.get("stage") == "retriever":
        raise RuntimeError(f"Retriever did not finish within {max_steps} API iterations")

    return {
        "iterations": len(traces),
        "traces": traces,
        "last_output": last_output,
        "final_stage": str(final_status.get("stage", "")),
    }


def _render_text_api_actions(run_dir: Path, stage: str, cfg: dict[str, Any]) -> None:
    st.subheader("Text API Assist")
    if not cfg.get("enabled", False):
        st.caption("Enable Text API Assist from sidebar to use this section.")
        return
    if stage not in {"retriever", "stylist", "critic"}:
        st.caption(
            "Text API Assist automation is enabled only for retriever, stylist and critic stages."
        )
        return
    if not cfg.get("api_key") or not cfg.get("model"):
        st.warning("Please provide API key and model in sidebar first.")
        return

    auto_submit = st.checkbox(
        "Auto submit API result to state machine",
        value=True,
        key=f"api_auto_submit_{stage}",
    )
    retrieval_setting = ""
    if stage == "retriever":
        try:
            retrieval_setting = str(load_state(run_dir).get("retrieval_setting", ""))
        except Exception:
            retrieval_setting = ""

    retriever_auto_mode = stage == "retriever" and retrieval_setting == "auto"
    if retriever_auto_mode:
        st.info(
            "Retriever auto mode: one click will run all chunk iterations plus final top10 selection."
        )
        if not auto_submit:
            st.warning("Retriever auto mode requires auto-submit to advance chunk state.")

    if stage == "critic":
        st.info(
            "For parity with original pipeline, critic should review the current visualizer image."
        )
        critic_attach_image = st.checkbox(
            "Attach latest visualizer image to critic API call (multimodal)",
            value=True,
            key="critic_api_attach_image",
        )
        if not critic_attach_image:
            st.warning("Image is not attached. Critic may over-simplify and drift from visual fidelity.")
    else:
        critic_attach_image = False
    run_label = (
        "Run retriever via Text API (one click all iterations)"
        if retriever_auto_mode
        else f"Run {stage} via Text API"
    )
    if st.button(run_label, use_container_width=True):
        try:
            if retriever_auto_mode:
                if not auto_submit:
                    raise ValueError("Please enable auto-submit for retriever auto mode.")
                run_info = _run_retriever_auto_via_api(run_dir, cfg)
                st.session_state[f"api_last_output_{stage}"] = str(run_info.get("last_output", ""))
                st.session_state[f"api_last_trace_{stage}"] = run_info.get("traces", [])
                st.success(
                    f"Retriever finished in {run_info.get('iterations', 0)} API calls. "
                    f"Current stage: {run_info.get('final_stage', '')}"
                )
                st.rerun()
            else:
                prompt = _get_or_build_stage_prompt(
                    run_dir, stage, force_refresh=(stage == "retriever")
                )
                image_bytes: bytes | None = None
                if stage == "critic" and critic_attach_image:
                    image_bytes = _get_latest_visualizer_image_bytes_for_critic(run_dir)
                    if not image_bytes:
                        raise RuntimeError(
                            "No current-round visualizer image found to attach for critic."
                        )
                response_text = _call_text_api(
                    prompt, cfg, image_bytes=image_bytes
                ).strip()
                if not response_text:
                    raise RuntimeError("Empty API response")
                st.session_state[f"api_last_output_{stage}"] = response_text
                st.success("API call succeeded.")
                st.code(response_text, language="text")
                if auto_submit:
                    submit_text_output(run_dir, response_text)
                    st.success("API result submitted to state machine.")
                    st.rerun()
        except Exception as exc:
            st.error(f"API run failed: {exc}")

    last_output = st.session_state.get(f"api_last_output_{stage}", "")
    last_trace = st.session_state.get(f"api_last_trace_{stage}", [])
    if retriever_auto_mode and isinstance(last_trace, list) and last_trace:
        with st.expander("Last Retriever Auto-Run Trace", expanded=False):
            st.table(last_trace)
    if last_output:
        st.markdown("**Last API Output**")
        st.code(last_output, language="text")
        if st.button(f"Submit Last API Output ({stage})", use_container_width=True):
            try:
                submit_text_output(run_dir, str(last_output))
                st.success("Submitted.")
                st.rerun()
            except Exception as exc:
                st.error(f"Submit failed: {exc}")


def _submit_image_area(run_dir: Path) -> None:
    st.subheader("Submit Visualizer Image")
    opts: dict[str, Any] = {}
    try:
        state = load_state(run_dir)
        opts = state.get("visualizer_options", {})
        if isinstance(opts, dict) and opts:
            st.caption(
                "Current visualizer options: "
                f"aspect_ratio={opts.get('aspect_ratio', '')}, "
                f"image_size={opts.get('image_size', '')}, "
                f"use_reference_images={bool(opts.get('use_reference_images', False))}, "
                f"reference_input_mode={opts.get('reference_input_mode', 'agent')}, "
                f"enforce_output_specs={bool(opts.get('enforce_output_specs', True))}, "
                f"candidates_per_round={int(opts.get('candidates_per_round', 1) or 1)}"
            )
    except Exception:
        opts = {}

    candidates_expected = max(1, int(opts.get("candidates_per_round", 1) or 1))
    enforce_specs = bool(opts.get("enforce_output_specs", True))

    if candidates_expected > 1:
        uploads = st.file_uploader(
            f"Upload generated images (up to {candidates_expected} candidates)",
            type=["png", "jpg", "jpeg", "webp"],
            key="submit_image_file_multi",
            accept_multiple_files=True,
        )
    else:
        single = st.file_uploader(
            "Upload generated image",
            type=["png", "jpg", "jpeg", "webp"],
            key="submit_image_file",
        )
        uploads = [single] if single is not None else []

    uploads = [u for u in uploads if u is not None]
    selected_upload = None
    if uploads:
        if candidates_expected > 1:
            st.caption(
                f"Uploaded {len(uploads)} candidate(s). "
                f"Expected {candidates_expected} for this round."
            )
            labels = [f"{idx+1}. {u.name}" for idx, u in enumerate(uploads)]
            chosen_label = st.radio(
                "Choose the best candidate to submit",
                labels,
                horizontal=False,
                key="submit_image_pick",
            )
            chosen_idx = labels.index(chosen_label)
            selected_upload = uploads[chosen_idx]
            cols = st.columns(min(3, len(uploads)))
            for idx, up in enumerate(uploads[:3]):
                with cols[idx]:
                    st.image(up, caption=f"Candidate {idx+1}", use_container_width=True)
        else:
            selected_upload = uploads[0]
            st.image(selected_upload, caption="Preview", use_container_width=True)

    spec_info = {}
    spec_issues: list[str] = []
    if selected_upload is not None:
        try:
            img_bytes = selected_upload.getvalue()
            spec_info, spec_issues = _check_image_specs(img_bytes, opts if isinstance(opts, dict) else {})
            st.caption(
                f"Selected image size: {spec_info.get('width', '?')} x {spec_info.get('height', '?')}"
            )
            if spec_issues:
                for issue in spec_issues:
                    st.warning(issue)
        except Exception as exc:
            st.warning(f"Image spec check failed: {exc}")

    if st.button("Submit Image", use_container_width=True):
        if selected_upload is None:
            st.error("Please upload and select an image first.")
            return
        if candidates_expected > 1 and len(uploads) < candidates_expected:
            st.warning(
                f"Only {len(uploads)} candidate(s) uploaded, expected {candidates_expected}. "
                "Proceeding with selected candidate."
            )
        if spec_issues and enforce_specs:
            st.error(
                "Selected image does not satisfy visualizer output specs. "
                "Please regenerate or disable strict enforcement in Run settings."
            )
            return
        uploads_dir = run_dir / "manual_uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        ext = Path(selected_upload.name).suffix or ".png"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = uploads_dir / f"visualizer_round_upload_{ts}{ext}"
        dst.write_bytes(selected_upload.getvalue())
        try:
            submit_image_for_visualizer(run_dir, dst)
            st.success(f"Image submitted: {dst}")
            st.rerun()
        except Exception as exc:
            st.error(f"Submit failed: {exc}")


def _show_run_context(run_dir: Path) -> None:
    with st.expander("Run Context", expanded=False):
        try:
            input_payload = load_input(run_dir)
            st.markdown("**visual_intent**")
            st.code(str(input_payload.get("visual_intent", "")), language="text")
            st.markdown("**content**")
            st.code(str(input_payload.get("content", ""))[:8000], language="text")
            if isinstance(input_payload.get("additional_info"), dict):
                st.markdown("**additional_info**")
                st.json(input_payload.get("additional_info"))
        except Exception as exc:
            st.warning(f"Cannot load input payload: {exc}")


def _init_run_panel() -> Path:
    st.sidebar.header("Run")
    if "pending_run_dir" in st.session_state:
        st.session_state["run_dir_input"] = str(st.session_state.pop("pending_run_dir"))
    if "run_dir_input" not in st.session_state:
        st.session_state["run_dir_input"] = str(
            st.session_state.get("run_dir", DEFAULT_RUN_DIR)
        )
    run_dir_text = st.sidebar.text_input("Run Directory", key="run_dir_input")
    run_dir = Path(run_dir_text)
    st.session_state["run_dir"] = str(run_dir)
    _render_reference_gallery_panel(str(st.session_state.get("init_task_name", "diagram")))

    with st.sidebar.expander("Create New Run", expanded=False):
        for k, v in {
            "init_task_name": "diagram",
            "init_exp_mode": "demo_full",
            "init_retrieval_setting": "none",
            "init_interaction_mode": "chat_only",
            "init_max_critic_rounds": 3,
            "init_visual_intent": "",
            "init_content": "",
            "init_aspect_ratio": "16:9",
            "init_image_size": "1K",
            "init_visualizer_use_refs": False,
            "init_visualizer_enforce_specs": True,
            "init_visualizer_candidates": 1,
        }.items():
            if k not in st.session_state:
                st.session_state[k] = v

        c1, c2 = st.columns(2)
        if c1.button("Load Built-in Example", use_container_width=True):
            try:
                method_text, caption_text = _load_builtin_example_from_demo()
                st.session_state["init_content"] = method_text
                st.session_state["init_visual_intent"] = caption_text
                st.session_state["pending_run_dir"] = str(
                    ROOT_DIR / "results" / "chat_bridge_builtin_example"
                )
                st.success("Loaded built-in example from demo.py")
                st.rerun()
            except Exception as exc:
                st.error(f"Load built-in example failed: {exc}")
        if c2.button("Clear Inputs", use_container_width=True):
            st.session_state["init_content"] = ""
            st.session_state["init_visual_intent"] = ""
            st.rerun()

        task_options = ["diagram", "plot"]
        exp_options = ["demo_planner_critic", "demo_full"]
        retrieval_options = ["none", "manual", "auto"]
        interaction_mode_options = ["chat_only", "agent"]
        aspect_options = ["1:1", "4:3", "3:2", "16:9", "21:9"]
        size_options = ["1K", "2K", "4K"]

        task_name = st.selectbox(
            "task_name",
            task_options,
            index=task_options.index(
                st.session_state["init_task_name"]
                if st.session_state["init_task_name"] in task_options
                else "diagram"
            ),
            key="init_task_name",
        )
        exp_mode = st.selectbox(
            "exp_mode",
            exp_options,
            index=exp_options.index(
                st.session_state["init_exp_mode"]
                if st.session_state["init_exp_mode"] in exp_options
                else "demo_full"
            ),
            key="init_exp_mode",
        )
        retrieval = st.selectbox(
            "retrieval_setting",
            retrieval_options,
            index=retrieval_options.index(
                st.session_state["init_retrieval_setting"]
                if st.session_state["init_retrieval_setting"] in retrieval_options
                else "none"
            ),
            key="init_retrieval_setting",
        )
        interaction_mode = st.selectbox(
            "Model Interaction Mode",
            interaction_mode_options,
            index=interaction_mode_options.index(
                st.session_state["init_interaction_mode"]
                if st.session_state["init_interaction_mode"] in interaction_mode_options
                else "chat_only"
            ),
            key="init_interaction_mode",
            help=(
                "chat_only: prompts expect image uploads in chat batches. "
                "agent: prompts provide local file paths for agent runtime."
            ),
        )
        max_rounds = st.number_input(
            "max_critic_rounds",
            min_value=1,
            max_value=10,
            value=int(st.session_state.get("init_max_critic_rounds", 3)),
            step=1,
            key="init_max_critic_rounds",
        )
        visual_intent = st.text_area(
            "Caption / Visual Intent",
            height=80,
            key="init_visual_intent",
        )
        content = st.text_area(
            "Method / Data Content",
            height=200,
            key="init_content",
        )

        st.markdown("**Visualizer Options**")
        aspect_ratio = st.selectbox(
            "Target Aspect Ratio",
            aspect_options,
            index=aspect_options.index(
                st.session_state["init_aspect_ratio"]
                if st.session_state["init_aspect_ratio"] in aspect_options
                else "16:9"
            ),
            key="init_aspect_ratio",
        )
        image_size = st.selectbox(
            "Target Image Size",
            size_options,
            index=size_options.index(
                st.session_state["init_image_size"]
                if st.session_state["init_image_size"] in size_options
                else "1K"
            ),
            key="init_image_size",
        )
        use_visualizer_refs = st.checkbox(
            "Visualizer uses retrieved reference images (optional, non-original behavior)",
            value=bool(st.session_state.get("init_visualizer_use_refs", False)),
            key="init_visualizer_use_refs",
        )
        st.caption(
            "Prompt mode source: `Model Interaction Mode` above "
            f"(current: `{interaction_mode}`)."
        )
        enforce_specs = st.checkbox(
            "Enforce uploaded image aspect ratio/resolution",
            value=bool(st.session_state.get("init_visualizer_enforce_specs", True)),
            key="init_visualizer_enforce_specs",
        )
        visualizer_candidates = st.number_input(
            "Visualizer candidates per round",
            min_value=1,
            max_value=6,
            value=int(st.session_state.get("init_visualizer_candidates", 1)),
            step=1,
            key="init_visualizer_candidates",
            help="If >1, upload multiple generated images in Visualizer stage and choose best one to submit.",
        )

        if st.button("Init Run", use_container_width=True):
            if not visual_intent.strip() or not content.strip():
                st.error("Both content and caption are required.")
            else:
                ref_root = _pick_preferred_reference_root(task_name)
                if retrieval != "none":
                    ref_json = ref_root / task_name / "ref.json"
                    if not ref_json.exists():
                        st.error(
                            "Reference gallery is not ready for retrieval mode.\n"
                            f"Missing file: {ref_json}\n\n"
                            "Options:\n"
                            "1) Switch `retrieval_setting` to `none` for quick start.\n"
                            "2) Prepare `data/PaperBananaBench` and retry."
                        )
                        return run_dir
                try:
                    init_run(
                        run_dir=run_dir,
                        task_name=task_name,
                        exp_mode=exp_mode,
                        retrieval_setting=retrieval,
                        max_critic_rounds=int(max_rounds),
                        content=content,
                        visual_intent=visual_intent,
                        work_dir=ROOT_DIR,
                        plot_visualizer_route="code",
                        auto_chunk_size=30,
                        visualizer_aspect_ratio=aspect_ratio,
                        visualizer_image_size=image_size,
                        visualizer_use_reference_images=use_visualizer_refs,
                        visualizer_reference_input_mode=interaction_mode,
                        visualizer_enforce_specs=enforce_specs,
                        visualizer_candidates_per_round=int(visualizer_candidates),
                        reference_gallery_dir=ref_root,
                    )
                except Exception as exc:
                    st.error(f"Init run failed: {exc}")
                    return run_dir
                st.success(f"Initialized run: {run_dir}")
                st.rerun()
    return run_dir


def _show_raw_state(run_dir: Path) -> None:
    with st.expander("Raw State JSON", expanded=False):
        try:
            st.json(load_state(run_dir))
        except Exception as exc:
            st.warning(f"Cannot load state: {exc}")


def _get_final_image_info(run_dir: Path) -> dict[str, Any]:
    state = load_state(run_dir)
    rounds = state.get("rounds", [])
    best = None
    for idx, rd in enumerate(rounds):
        rel = str(rd.get("visualizer_image_rel_path", "")).strip()
        if rel:
            abs_path = run_dir / rel
            if abs_path.exists():
                best = {"round": idx, "rel_path": rel, "abs_path": str(abs_path)}
    return best or {}


def _render_completion_panel(run_dir: Path) -> None:
    st.success("Pipeline completed.")
    try:
        state = load_state(run_dir)
    except Exception as exc:
        st.warning(f"Cannot load final state: {exc}")
        return

    info = _get_final_image_info(run_dir)
    if info:
        st.subheader("Recommended Final Image")
        st.caption(
            f"Selected by pipeline rule: latest valid visualizer output (round {info['round']})."
        )
        st.image(info["abs_path"], use_container_width=True)
        img_bytes = Path(info["abs_path"]).read_bytes()
        st.download_button(
            "Download Final Image",
            data=img_bytes,
            file_name=f"final_round{info['round']}{Path(info['abs_path']).suffix}",
            mime="image/png",
            use_container_width=True,
        )
    else:
        st.warning("No visualizer image found in artifacts.")

    rows = []
    for idx, rd in enumerate(state.get("rounds", [])):
        rows.append(
            {
                "round": idx,
                "image": rd.get("visualizer_image_rel_path", ""),
                "critic_done": bool(rd.get("critic_done", False)),
                "critic_suggestions": str(rd.get("critic_suggestions", ""))[:120],
            }
        )
    if rows:
        with st.expander("Round Summary", expanded=False):
            st.table(rows)


def main() -> None:
    st.set_page_config(
        page_title="PaperBanana Chat-Bridge Manual UI",
        page_icon="PB",
        layout="wide",
    )
    st.title("PaperBanana Chat-Bridge Manual UI")
    st.caption(
        "This UI does not alter pipeline logic. It only wraps the existing chat_bridge state machine."
    )

    text_api_cfg = _render_text_api_sidebar()
    run_dir = _init_run_panel()
    if not run_dir.exists():
        st.warning(
            "Run directory does not exist yet. Create one from sidebar `Create New Run`."
        )
        st.stop()

    try:
        status = _render_status(run_dir)
    except Exception as exc:
        st.error(f"Failed to read run status: {exc}")
        st.stop()

    stage = status.get("stage", "")
    _render_stage_order(run_dir, stage)
    _render_prompt_box(run_dir, stage)
    _show_run_context(run_dir)

    if stage == "retriever":
        _render_text_api_actions(run_dir, stage, text_api_cfg)
        _submit_retriever_area(run_dir)
    elif stage == "planner":
        _render_planner_reference_panel(run_dir)
        _submit_text_area(run_dir)
    elif stage == "visualizer":
        _submit_image_area(run_dir)
    elif stage == "critic":
        _render_text_api_actions(run_dir, stage, text_api_cfg)
        _submit_text_area(run_dir)
    elif stage == "stylist":
        _render_text_api_actions(run_dir, stage, text_api_cfg)
        _submit_text_area(run_dir)
    elif stage == "completed":
        _render_completion_panel(run_dir)
    else:
        st.warning(f"Unknown stage: {stage}")

    _show_raw_state(run_dir)


if __name__ == "__main__":
    main()
