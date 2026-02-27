from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from tools.chat_bridge.parsing import (
    parse_critic_json,
    parse_retriever_chunk_json,
    parse_retriever_top10_json,
)
from tools.chat_bridge.state_machine import (
    get_active_description_for_current_round,
    apply_critic_result,
    apply_planner_description,
    apply_retriever_top10,
    apply_stylist_description,
    apply_visualizer_result,
    create_initial_state,
    get_current_stage,
    get_round_description_for_visualizer,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
STATE_FILE = "chat_bridge_state.json"
INPUT_FILE = "input_target.json"


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _state_path(run_dir: Path) -> Path:
    return run_dir / STATE_FILE


def _input_path(run_dir: Path) -> Path:
    return run_dir / INPUT_FILE


def load_state(run_dir: Path) -> Dict[str, Any]:
    path = _state_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"State file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(run_dir: Path, state: Dict[str, Any]) -> None:
    _state_path(run_dir).write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_input(run_dir: Path) -> Dict[str, Any]:
    path = _input_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _save_input(run_dir: Path, payload: Dict[str, Any]) -> None:
    _input_path(run_dir).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _append_artifact_text(run_dir: Path, name: str, text: str) -> Path:
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    path = artifacts / name
    path.write_text(text, encoding="utf-8")
    return path


def _serialize_content_for_prompt(content: Any) -> str:
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _resolve_ref_json_path(state: Dict[str, Any]) -> Path:
    return _resolve_ref_task_dir(state) / "ref.json"


def _iter_ref_roots(state: Dict[str, Any]) -> List[Path]:
    roots: List[Path] = []
    explicit = str(state.get("reference_gallery_dir", "")).strip()
    if explicit:
        roots.append(Path(explicit))

    work_dir = Path(state.get("work_dir", str(ROOT_DIR)))
    roots.extend(
        [
            work_dir / "data" / "PaperBananaBench",
            ROOT_DIR / "data" / "PaperBananaBench",
            ROOT_DIR / "tools" / "chat_bridge" / "reference_gallery" / "PaperBananaBench",
            ROOT_DIR / "reference_gallery" / "PaperBananaBench",
        ]
    )

    seen: set[str] = set()
    deduped: List[Path] = []
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(root)
    return deduped


def _resolve_ref_task_dir(state: Dict[str, Any]) -> Path:
    task_name = str(state.get("task_name", "diagram"))
    roots = _iter_ref_roots(state)
    for root in roots:
        task_dir = root / task_name
        if (task_dir / "ref.json").exists():
            return task_dir
    if roots:
        return roots[0] / task_name
    return ROOT_DIR / "data" / "PaperBananaBench" / task_name


def _load_ref_pool(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    ref_path = _resolve_ref_json_path(state)
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference pool not found: {ref_path}")
    pool = json.loads(ref_path.read_text(encoding="utf-8"))
    if state["task_name"] == "diagram":
        limit = min(len(pool), int(state.get("retrieval_auto", {}).get("candidate_limit", 200)))
        return pool[:limit]
    return pool


def _compact_ref_line(item: Dict[str, Any], idx: int) -> str:
    visual_intent = str(item.get("visual_intent", "")).replace("\n", " ").strip()
    content = _serialize_content_for_prompt(item.get("content", ""))
    content = content.replace("\n", " ").strip()
    if len(content) > 220:
        content = content[:220] + "..."
    if len(visual_intent) > 140:
        visual_intent = visual_intent[:140] + "..."
    return f"{idx}) id={item.get('id','')} | visual_intent={visual_intent} | content={content}"


def _build_top_reference_block(state: Dict[str, Any]) -> str:
    ids = state.get("top10_references", [])
    if not ids:
        return "[]"
    pool = _load_ref_pool(state)
    id_map = {str(item.get("id")): item for item in pool}
    lines = []
    for i, ref_id in enumerate(ids, start=1):
        item = id_map.get(ref_id)
        if item is None:
            lines.append(f"{i}) id={ref_id} | visual_intent=NA | content=NA")
        else:
            lines.append(_compact_ref_line(item, i))
    return "\n".join(lines)


def _normalize_reference_input_mode(state: Dict[str, Any]) -> str:
    opts = state.get("visualizer_options", {})
    mode = str(opts.get("reference_input_mode", "agent")).strip().lower()
    if mode not in {"agent", "chat_only"}:
        return "agent"
    return mode


def _build_retriever_prompt(state: Dict[str, Any], input_data: Dict[str, Any]) -> str:
    task = state["task_name"]
    retrieval_setting = state["retrieval_setting"]
    if retrieval_setting == "manual":
        key = "top10_diagrams" if task == "diagram" else "top10_plots"
        return (
            "[STAGE: RETRIEVER]\n"
            "Manual mode: provide strict JSON only.\n"
            f'Expected format: {{"{key}":["ref_1","ref_2", "..."]}}'
        )
    if retrieval_setting != "auto":
        raise ValueError("Retriever prompt requested for non-retrieval mode")

    auto = state["retrieval_auto"]
    pool = _load_ref_pool(state)
    content_text = _serialize_content_for_prompt(input_data["content"])
    caption_text = str(input_data["visual_intent"])

    if auto["phase"] == "chunk":
        chunk_idx = int(auto["chunk_index"])
        chunk_size = int(auto["chunk_size"])
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(pool))
        chunk_items = pool[start:end]
        lines = [_compact_ref_line(item, i + 1) for i, item in enumerate(chunk_items)]
        return (
            "[STAGE: RETRIEVER]\n"
            f"Task: shortlist from chunk {chunk_idx + 1}/{auto['total_chunks']}.\n"
            "Return strict JSON only:\n"
            '{"top3_ids":["ref_1","ref_2","ref_3"]}\n\n'
            f"Target visual intent:\n{caption_text}\n\n"
            f"Target source content:\n{content_text}\n\n"
            "Candidate chunk:\n"
            + "\n".join(lines)
        )

    # final phase
    shortlisted_ids = auto.get("shortlisted_ids", [])
    id_set = set(shortlisted_ids)
    shortlisted_items = [item for item in pool if str(item.get("id")) in id_set]
    key = "top10_diagrams" if task == "diagram" else "top10_plots"
    lines = [_compact_ref_line(item, i + 1) for i, item in enumerate(shortlisted_items)]
    return (
        "[STAGE: RETRIEVER]\n"
        "Task: choose final top 10 from shortlisted references.\n"
        "Return strict JSON only:\n"
        f'{{"{key}":["ref_1","ref_2","..."]}}\n\n'
        f"Target visual intent:\n{caption_text}\n\n"
        f"Target source content:\n{content_text}\n\n"
        "Shortlisted references:\n"
        + ("\n".join(lines) if lines else "(empty)")
    )


def _build_planner_prompt(state: Dict[str, Any], input_data: Dict[str, Any]) -> str:
    task_name = state["task_name"]
    content_text = _serialize_content_for_prompt(input_data["content"])
    visual_intent = str(input_data["visual_intent"])
    top_refs = _build_top_reference_block(state)
    ref_mode = _normalize_reference_input_mode(state)
    refs_path_block = _build_visualizer_reference_block(state)

    if ref_mode == "chat_only":
        multimodal_block = (
            "\n\nReference input protocol (CHAT-ONLY mode):\n"
            "- I will upload top reference images in one or more batches.\n"
            "- Before all uploads finish, reply only: ACK_RECEIVED_BATCH.\n"
            "- Wait until I send [ALL_REFERENCES_UPLOADED], then produce final planner description.\n"
            "- Do not ignore uploaded references.\n"
        )
    else:
        multimodal_block = (
            "\n\nReference input protocol (AGENT mode):\n"
            "- Read all local top-reference images below before answering.\n"
            "- Use these references jointly with target methodology/caption.\n"
            "Reference image paths:\n"
            f"{refs_path_block}\n"
        )

    if task_name == "diagram":
        return (
            "[STAGE: PLANNER]\n"
            "Generate a detailed target diagram description from:\n"
            "- target methodology\n"
            "- target caption\n"
            "- top references\n\n"
            "Requirements:\n"
            "- preserve logic exactly\n"
            "- explicit geometry/layout/arrows/labels/colors\n"
            "- no figure title in image\n\n"
            "Output:\nOnly final detailed description text.\n\n"
            f"Target caption:\n{visual_intent}\n\n"
            f"Target methodology:\n{content_text}\n\n"
            f"Top references:\n{top_refs}"
            f"{multimodal_block}"
        )

    return (
        "[STAGE: PLANNER]\n"
        "Generate a detailed plot description from:\n"
        "- target raw data\n"
        "- target visual intent\n"
        "- top references\n\n"
        "Requirements:\n"
        "- include all required data points\n"
        "- define mappings (x, y, color, marker, group)\n"
        "- specify style and readability constraints\n\n"
        "Output:\nOnly final detailed description text.\n\n"
        f"Target raw data:\n{content_text}\n\n"
        f"Target visual intent:\n{visual_intent}\n\n"
        f"Top references:\n{top_refs}"
        f"{multimodal_block}"
    )


def _load_style_guide(state: Dict[str, Any]) -> str:
    work_dir = Path(state.get("work_dir", str(ROOT_DIR)))
    task_name = state["task_name"]
    style_path = work_dir / "style_guides" / f"neurips2025_{task_name}_style_guide.md"
    if not style_path.exists():
        return "(Style guide not found)"
    return style_path.read_text(encoding="utf-8")


def _build_stylist_prompt(state: Dict[str, Any], input_data: Dict[str, Any]) -> str:
    planner_desc = state.get("planner_description", "")
    style_guide = _load_style_guide(state)
    content_text = _serialize_content_for_prompt(input_data["content"])
    visual_intent = str(input_data["visual_intent"])
    if state["task_name"] == "diagram":
        return (
            "[STAGE: STYLIST]\n"
            "Refine the description using style guide, without changing semantics.\n\n"
            "Output:\nOnly polished description text.\n\n"
            f"Detailed description:\n{planner_desc}\n\n"
            f"Style guide:\n{style_guide}\n\n"
            f"Methodology context:\n{content_text}\n\n"
            f"Caption context:\n{visual_intent}"
        )
    return (
        "[STAGE: STYLIST]\n"
        "Refine the plot description using style guide, without changing quantitative content.\n\n"
        "Output:\nOnly polished description text.\n\n"
        f"Detailed description:\n{planner_desc}\n\n"
        f"Style guide:\n{style_guide}\n\n"
        f"Raw data context:\n{content_text}\n\n"
        f"Visual intent context:\n{visual_intent}"
    )


def _build_visualizer_reference_block(state: Dict[str, Any]) -> str:
    ids = [str(x).strip() for x in state.get("top10_references", []) if str(x).strip()]
    if not ids:
        return "(No top10_references available)"

    ref_path = _resolve_ref_json_path(state)
    if not ref_path.exists():
        return f"(Reference pool not found: {ref_path})"

    pool = json.loads(ref_path.read_text(encoding="utf-8"))
    id_map = {str(item.get("id", "")): item for item in pool}

    base_dir = _resolve_ref_task_dir(state)

    lines: List[str] = []
    for ref_id in ids:
        item = id_map.get(ref_id)
        if not item:
            continue
        rel = str(item.get("path_to_gt_image", "")).strip()
        if not rel:
            continue
        abs_path = (base_dir / rel).resolve()
        lines.append(f"- {ref_id}: {abs_path}")

    return "\n".join(lines) if lines else "(No valid local paths resolved from top10_references)"


def _build_visualizer_prompt(state: Dict[str, Any]) -> str:
    desc = get_round_description_for_visualizer(state)
    opts = state.get("visualizer_options", {})
    aspect_ratio = str(opts.get("aspect_ratio", "")).strip()
    image_size = str(opts.get("image_size", "")).strip()
    use_refs = bool(opts.get("use_reference_images", False))
    candidates_per_round = int(opts.get("candidates_per_round", 1) or 1)
    reference_input_mode = _normalize_reference_input_mode(state)

    output_prefs: List[str] = []
    if aspect_ratio:
        output_prefs.append(f"- target aspect ratio: {aspect_ratio}")
    if image_size:
        output_prefs.append(f"- target resolution level: {image_size}")

    style_ref_block = ""
    if use_refs:
        if reference_input_mode == "chat_only":
            style_ref_block = (
                "\n\nOptional style-reference guidance (CHAT-ONLY mode):\n"
                "- Read all reference images uploaded in this chat before generating.\n"
                "- If references are uploaded in multiple batches, wait until user sends [ALL_REFERENCES_UPLOADED].\n"
                "- Before [ALL_REFERENCES_UPLOADED], reply only: ACK_RECEIVED_BATCH.\n"
                "- Use references as style/layout cues only. Do NOT copy semantic content.\n"
            )
        else:
            refs_text = _build_visualizer_reference_block(state)
            style_ref_block = (
                "\n\nOptional style-reference guidance (AGENT mode):\n"
                "- If your model supports multimodal/image inputs, review these reference images for visual style cues only.\n"
                "- Do NOT copy their semantic content. Preserve the current description's logic.\n"
                "- If local paths cannot be accessed in your environment, ask user to upload these references first.\n"
                "Reference image paths:\n"
                f"{refs_text}"
            )

    prefs_block = ""
    if output_prefs:
        prefs_block = "\nOutput preferences:\n" + "\n".join(output_prefs)
    candidate_block = ""
    if candidates_per_round > 1:
        candidate_block = (
            "\nCandidate policy:\n"
            f"- Generate {candidates_per_round} distinct candidate images for this same description.\n"
            "- Keep semantics identical across candidates; vary only style/layout details.\n"
        )

    if state["task_name"] == "diagram":
        return (
            "[STAGE: VISUALIZER_PROMPT]\n"
            "Render a scientific diagram from this description.\n\n"
            "Hard constraints:\n"
            "- no figure title text in image\n"
            "- clear labels\n"
            "- unambiguous arrow directions\n"
            "- professional academic style"
            f"{prefs_block}"
            f"{candidate_block}"
            f"{style_ref_block}\n\n"
            f"Description:\n{desc}"
        )
    route = state.get("plot_visualizer_route", "code")
    if route == "image":
        return (
            "[STAGE: VISUALIZER_PROMPT]\n"
            "Render a statistical plot image from this description directly (no code)."
            f"{prefs_block}\n\n"
            f"Description:\n{desc}"
        )
    return (
            "[STAGE: VISUALIZER_PROMPT]\n"
            "Use python matplotlib to generate a statistical plot based on the following detailed description.\n"
            "Only provide the code without any explanations."
            f"{prefs_block}\n\n"
            f"Description:\n{desc}"
        )


def _build_critic_prompt(state: Dict[str, Any], input_data: Dict[str, Any]) -> str:
    round_idx = state["current_round"]
    desc = state["rounds"][round_idx]["effective_description"] or get_active_description_for_current_round(state)
    image_rel = state["rounds"][round_idx].get("visualizer_image_rel_path", "")
    visual_intent = str(input_data["visual_intent"])
    content_text = _serialize_content_for_prompt(input_data["content"])

    if state["task_name"] == "diagram":
        return (
            "[STAGE: CRITIC]\n"
            "Attach the generated image in chat before sending this prompt.\n"
            "Critique against methodology + caption + current description.\n\n"
            "Check:\n"
            "1) semantic fidelity\n"
            "2) text/label correctness\n"
            "3) readability/layout\n"
            "4) remove redundant legend text\n"
            "5) ensure caption text is not drawn inside image\n\n"
            "Return strict JSON only:\n"
            '{"critic_suggestions":"...","revised_description":"..."}\n\n'
            f"Current round image path:\n{image_rel}\n\n"
            f"Methodology:\n{content_text}\n\n"
            f"Caption:\n{visual_intent}\n\n"
            f"Current description:\n{desc}"
        )

    return (
        "[STAGE: CRITIC]\n"
        "Attach the generated plot image in chat before sending this prompt.\n"
        "Critique this plot against raw data + visual intent + current description.\n\n"
        "Check:\n"
        "1) numeric/data fidelity\n"
        "2) label/text correctness\n"
        "3) readability/layout\n"
        "4) legend clarity and redundancy\n"
        "5) avoid drawing caption text inside image\n\n"
        "Return strict JSON only:\n"
        '{"critic_suggestions":"...","revised_description":"..."}\n\n'
        f"Current round image path:\n{image_rel}\n\n"
        f"Raw data:\n{content_text}\n\n"
        f"Visual intent:\n{visual_intent}\n\n"
        f"Current description:\n{desc}"
    )


def init_run(
    *,
    run_dir: Path,
    task_name: str,
    exp_mode: str,
    retrieval_setting: str,
    max_critic_rounds: int,
    content: Any,
    visual_intent: str,
    work_dir: Path | None = None,
    plot_visualizer_route: str = "code",
    auto_chunk_size: int = 30,
    visualizer_aspect_ratio: str = "16:9",
    visualizer_image_size: str = "1K",
    visualizer_use_reference_images: bool = False,
    visualizer_reference_input_mode: str = "agent",
    visualizer_enforce_specs: bool = True,
    visualizer_candidates_per_round: int = 1,
    reference_gallery_dir: Path | None = None,
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "prompts").mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    state = create_initial_state(
        task_name=task_name,
        exp_mode=exp_mode,
        retrieval_setting=retrieval_setting,
        max_critic_rounds=max_critic_rounds,
    )
    state["work_dir"] = str(work_dir or ROOT_DIR)
    if reference_gallery_dir is not None:
        state["reference_gallery_dir"] = str(reference_gallery_dir)
    else:
        embedded_root = (
            ROOT_DIR / "tools" / "chat_bridge" / "reference_gallery" / "PaperBananaBench"
        )
        if (embedded_root / task_name / "ref.json").exists():
            state["reference_gallery_dir"] = str(embedded_root)
    state["plot_visualizer_route"] = plot_visualizer_route
    state["visualizer_options"] = {
        "aspect_ratio": str(visualizer_aspect_ratio or "").strip(),
        "image_size": str(visualizer_image_size or "").strip(),
        "use_reference_images": bool(visualizer_use_reference_images),
        "reference_input_mode": str(visualizer_reference_input_mode or "agent").strip(),
        "enforce_output_specs": bool(visualizer_enforce_specs),
        "candidates_per_round": max(1, int(visualizer_candidates_per_round)),
    }

    if retrieval_setting == "auto":
        pool = _load_ref_pool(state)
        chunk_size = max(1, int(auto_chunk_size))
        total_chunks = (len(pool) + chunk_size - 1) // chunk_size
        state["retrieval_auto"] = {
            "phase": "chunk",
            "chunk_size": chunk_size,
            "chunk_index": 0,
            "total_chunks": total_chunks,
            "shortlisted_ids": [],
            "candidate_limit": 200 if task_name == "diagram" else len(pool),
        }

    payload = {
        "content": content,
        "visual_intent": visual_intent,
        "additional_info": {
            "rounded_ratio": str(visualizer_aspect_ratio or "").strip(),
            "target_image_size": str(visualizer_image_size or "").strip(),
            "visualizer_use_reference_images": bool(visualizer_use_reference_images),
            "visualizer_reference_input_mode": str(
                visualizer_reference_input_mode or "agent"
            ).strip(),
            "visualizer_enforce_specs": bool(visualizer_enforce_specs),
            "visualizer_candidates_per_round": max(1, int(visualizer_candidates_per_round)),
        },
    }
    _save_input(run_dir, payload)
    save_state(run_dir, state)

    _append_artifact_text(
        run_dir,
        "input_target.md",
        f"visual_intent:\n{visual_intent}\n\ncontent:\n{_serialize_content_for_prompt(content)}\n",
    )
    return state


def build_next_prompt(run_dir: Path) -> str:
    state = load_state(run_dir)
    input_data = load_input(run_dir)
    stage = get_current_stage(state)
    save_state(run_dir, state)

    if stage == "completed":
        return f"[STAGE: COMPLETED]\nstop_reason={state.get('stop_reason','')}"
    if stage == "retriever":
        prompt = _build_retriever_prompt(state, input_data)
    elif stage == "planner":
        prompt = _build_planner_prompt(state, input_data)
    elif stage == "stylist":
        prompt = _build_stylist_prompt(state, input_data)
    elif stage == "visualizer":
        prompt = _build_visualizer_prompt(state)
    elif stage == "critic":
        prompt = _build_critic_prompt(state, input_data)
    else:
        raise ValueError(f"Unknown stage: {stage}")

    prompt_path = run_dir / "prompts" / f"{_now_stamp()}_{stage}.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    return prompt


def submit_text_output(run_dir: Path, text: str) -> Dict[str, Any]:
    state = load_state(run_dir)
    stage = get_current_stage(state)
    task = state["task_name"]
    payload = (text or "").strip()

    if stage == "retriever":
        if state["retrieval_setting"] == "manual":
            top10 = parse_retriever_top10_json(payload, task_name=task)
            apply_retriever_top10(state, top10)
            _append_artifact_text(run_dir, "retrieval_top10.json", json.dumps(top10, ensure_ascii=False, indent=2))
        elif state["retrieval_setting"] == "auto":
            auto = state["retrieval_auto"]
            if auto["phase"] == "chunk":
                top3 = parse_retriever_chunk_json(payload)
                auto["shortlisted_ids"] = list(dict.fromkeys(auto["shortlisted_ids"] + top3))
                chunk_idx = int(auto["chunk_index"])
                _append_artifact_text(
                    run_dir,
                    f"retrieval_chunk_{chunk_idx}.json",
                    json.dumps({"top3_ids": top3}, ensure_ascii=False, indent=2),
                )
                auto["chunk_index"] = chunk_idx + 1
                if auto["chunk_index"] >= auto["total_chunks"]:
                    auto["phase"] = "final"
            else:
                top10 = parse_retriever_top10_json(payload, task_name=task)
                apply_retriever_top10(state, top10)
                _append_artifact_text(
                    run_dir,
                    "retrieval_top10.json",
                    json.dumps(top10, ensure_ascii=False, indent=2),
                )
        else:
            raise ValueError("retriever stage reached with retrieval_setting=none")

    elif stage == "planner":
        apply_planner_description(state, payload)
        _append_artifact_text(run_dir, "planner_desc.md", payload + "\n")
    elif stage == "stylist":
        apply_stylist_description(state, payload)
        _append_artifact_text(run_dir, "stylist_desc.md", payload + "\n")
    elif stage == "critic":
        parsed = parse_critic_json(payload)
        round_idx = state["current_round"]
        apply_critic_result(
            state,
            critic_suggestions=parsed["critic_suggestions"],
            revised_description=parsed["revised_description"],
        )
        _append_artifact_text(
            run_dir,
            f"critic_round{round_idx}.json",
            json.dumps(parsed, ensure_ascii=False, indent=2),
        )
        effective_desc = state["rounds"][round_idx].get("effective_description", "")
        _append_artifact_text(run_dir, f"critic_desc_round{round_idx}.md", effective_desc + "\n")
    else:
        raise ValueError(f"submit_text_output not valid at stage={stage}")

    save_state(run_dir, state)
    return state


def submit_image_for_visualizer(run_dir: Path, image_path: Path) -> Dict[str, Any]:
    state = load_state(run_dir)
    stage = get_current_stage(state)
    if stage != "visualizer":
        raise ValueError(f"submit_image_for_visualizer not valid at stage={stage}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    round_idx = state["current_round"]
    suffix = image_path.suffix if image_path.suffix else ".png"
    rel = Path("artifacts") / f"visual_round{round_idx}{suffix}"
    dst = run_dir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, dst)
    apply_visualizer_result(state, image_rel_path=str(rel).replace("\\", "/"))
    save_state(run_dir, state)
    return state


def get_status(run_dir: Path) -> Dict[str, Any]:
    state = load_state(run_dir)
    stage = get_current_stage(state)
    save_state(run_dir, state)
    return {
        "stage": stage,
        "current_round": state.get("current_round", 0),
        "completed": state.get("completed", False),
        "stop_reason": state.get("stop_reason", ""),
        "retrieval_setting": state.get("retrieval_setting", ""),
        "exp_mode": state.get("exp_mode", ""),
        "task_name": state.get("task_name", ""),
    }
