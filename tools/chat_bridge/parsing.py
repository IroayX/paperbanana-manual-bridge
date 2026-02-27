from __future__ import annotations

from typing import Any, Dict, List

import json_repair


def _clean_json_text(text: str) -> str:
    raw = (text or "").strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return raw


def _load_json(text: str) -> Dict[str, Any]:
    cleaned = _clean_json_text(text)
    try:
        obj = json_repair.loads(cleaned)
    except Exception as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("JSON payload must be an object")
    return obj


def _normalize_id_list(values: Any, key_name: str) -> List[str]:
    if not isinstance(values, list):
        raise ValueError(f"{key_name} must be a list")
    out: List[str] = []
    for item in values:
        if not isinstance(item, str):
            raise ValueError(f"{key_name} must contain string ids")
        v = item.strip()
        if v:
            out.append(v)
    return out


def parse_critic_json(text: str) -> Dict[str, str]:
    obj = _load_json(text)
    if "critic_suggestions" not in obj or "revised_description" not in obj:
        raise ValueError("Critic JSON must include critic_suggestions and revised_description")
    critic_suggestions = str(obj["critic_suggestions"]).strip()
    revised_description = str(obj["revised_description"]).strip()
    return {
        "critic_suggestions": critic_suggestions,
        "revised_description": revised_description,
    }


def parse_retriever_top10_json(text: str, task_name: str) -> List[str]:
    obj = _load_json(text)
    key = "top10_diagrams" if task_name == "diagram" else "top10_plots"
    if key not in obj:
        raise ValueError(f"Retriever JSON must include {key}")
    return _normalize_id_list(obj[key], key)


def parse_retriever_chunk_json(text: str) -> List[str]:
    obj = _load_json(text)
    if "top3_ids" not in obj:
        raise ValueError("Retriever chunk JSON must include top3_ids")
    return _normalize_id_list(obj["top3_ids"], "top3_ids")

