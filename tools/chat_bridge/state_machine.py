from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


def create_initial_state(
    task_name: str,
    exp_mode: str,
    retrieval_setting: str,
    max_critic_rounds: int,
) -> Dict[str, Any]:
    if exp_mode not in {"demo_planner_critic", "demo_full"}:
        raise ValueError(f"Unsupported exp_mode: {exp_mode}")
    if task_name not in {"diagram", "plot"}:
        raise ValueError(f"Unsupported task_name: {task_name}")
    if retrieval_setting not in {"none", "manual", "auto"}:
        raise ValueError(f"Unsupported retrieval_setting: {retrieval_setting}")
    if max_critic_rounds <= 0:
        raise ValueError("max_critic_rounds must be > 0")

    rounds = []
    for _ in range(max_critic_rounds):
        rounds.append(
            {
                "visualizer_done": False,
                "visualizer_image_rel_path": "",
                "critic_done": False,
                "critic_suggestions": "",
                "revised_description": "",
                "effective_description": "",
            }
        )

    return {
        "task_name": task_name,
        "exp_mode": exp_mode,
        "retrieval_setting": retrieval_setting,
        "max_critic_rounds": max_critic_rounds,
        "current_round": 0,
        "completed": False,
        "stop_reason": "",
        "retriever_done": retrieval_setting == "none",
        "top10_references": [],
        "planner_done": False,
        "planner_description": "",
        "stylist_done": False,
        "stylist_description": "",
        "rounds": rounds,
    }


def _active_description_for_round(state: Dict[str, Any], round_idx: int) -> str:
    if round_idx == 0:
        if state["exp_mode"] == "demo_full":
            return state.get("stylist_description", "") or state.get(
                "planner_description", ""
            )
        return state.get("planner_description", "")

    prev = state["rounds"][round_idx - 1]
    return prev.get("effective_description", "") or prev.get("revised_description", "")


def get_current_stage(state: Dict[str, Any]) -> str:
    if state.get("completed"):
        return "completed"

    if not state.get("retriever_done", False):
        return "retriever"
    if not state.get("planner_done", False):
        return "planner"
    if state.get("exp_mode") == "demo_full" and not state.get("stylist_done", False):
        return "stylist"

    round_idx = state.get("current_round", 0)
    max_rounds = state.get("max_critic_rounds", 0)
    if round_idx >= max_rounds:
        state["completed"] = True
        state["stop_reason"] = "max_critic_rounds_reached"
        return "completed"

    round_data = state["rounds"][round_idx]
    if not round_data.get("visualizer_done", False):
        return "visualizer"
    if not round_data.get("critic_done", False):
        return "critic"

    suggestions = (round_data.get("critic_suggestions") or "").strip()
    if suggestions == "No changes needed.":
        state["completed"] = True
        state["stop_reason"] = "critic_no_changes"
        return "completed"

    next_round = round_idx + 1
    if next_round >= state["max_critic_rounds"]:
        state["completed"] = True
        state["stop_reason"] = "max_critic_rounds_reached"
        return "completed"

    state["current_round"] = next_round
    return get_current_stage(state)


def apply_retriever_top10(state: Dict[str, Any], top10_references: list[str]) -> None:
    if get_current_stage(state) != "retriever":
        raise ValueError("Current stage is not retriever")
    state["top10_references"] = deepcopy(top10_references)
    state["retriever_done"] = True


def apply_planner_description(state: Dict[str, Any], planner_description: str) -> None:
    if get_current_stage(state) != "planner":
        raise ValueError("Current stage is not planner")
    state["planner_description"] = (planner_description or "").strip()
    state["planner_done"] = True


def apply_stylist_description(state: Dict[str, Any], stylist_description: str) -> None:
    if get_current_stage(state) != "stylist":
        raise ValueError("Current stage is not stylist")
    state["stylist_description"] = (stylist_description or "").strip()
    state["stylist_done"] = True


def apply_visualizer_result(state: Dict[str, Any], image_rel_path: str) -> None:
    if get_current_stage(state) != "visualizer":
        raise ValueError("Current stage is not visualizer")
    round_idx = state["current_round"]
    round_data = state["rounds"][round_idx]
    round_data["visualizer_done"] = True
    round_data["visualizer_image_rel_path"] = (image_rel_path or "").strip()


def apply_critic_result(
    state: Dict[str, Any],
    critic_suggestions: str,
    revised_description: str,
) -> None:
    if get_current_stage(state) != "critic":
        raise ValueError("Current stage is not critic")
    round_idx = state["current_round"]
    round_data = state["rounds"][round_idx]
    round_data["critic_done"] = True
    round_data["critic_suggestions"] = (critic_suggestions or "").strip()
    round_data["revised_description"] = (revised_description or "").strip()

    if round_data["revised_description"] == "No changes needed.":
        round_data["effective_description"] = _active_description_for_round(state, round_idx)
    else:
        round_data["effective_description"] = round_data["revised_description"]

    # Trigger stage transition logic.
    _ = get_current_stage(state)


def get_round_description_for_visualizer(state: Dict[str, Any]) -> str:
    if get_current_stage(state) != "visualizer":
        raise ValueError("Current stage is not visualizer")
    return _active_description_for_round(state, state["current_round"])


def get_active_description_for_current_round(state: Dict[str, Any]) -> str:
    round_idx = int(state.get("current_round", 0))
    return _active_description_for_round(state, round_idx)
