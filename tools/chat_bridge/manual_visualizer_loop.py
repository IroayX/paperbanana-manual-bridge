from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.chat_bridge.session import (  # noqa: E402
    build_next_prompt,
    get_status,
    submit_image_for_visualizer,
    submit_text_output,
)


def _write_prompt(out_dir: Path, filename: str, prompt: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    out_path.write_text(prompt, encoding="utf-8")
    return str(out_path)


def _print(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manual visualizer loop helper for chat_bridge runs."
    )
    parser.add_argument("--run-dir", required=True, help="chat_bridge run directory")
    parser.add_argument(
        "--image-path",
        default="",
        help="Path to a newly generated visualizer image (used when stage=visualizer)",
    )
    parser.add_argument(
        "--critic-file",
        default="",
        help="Path to critic JSON text output (used when stage=critic)",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Directory to write next prompt files (default: run-dir)",
    )
    parser.add_argument(
        "--emit-current-prompt",
        action="store_true",
        help="If no submission input is provided, emit the current stage prompt file",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir

    status = get_status(run_dir)
    stage = status["stage"]

    if stage == "completed":
        _print(
            {
                "status": "completed",
                "stage": stage,
                "current_round": status.get("current_round", 0),
                "stop_reason": status.get("stop_reason", ""),
            }
        )
        return 0

    if stage == "visualizer":
        if args.critic_file:
            _print(
                {
                    "status": "error",
                    "message": "stage=visualizer does not accept --critic-file",
                }
            )
            return 2

        if args.image_path:
            submit_image_for_visualizer(run_dir, Path(args.image_path))
            status = get_status(run_dir)
            if status["stage"] != "critic":
                _print(
                    {
                        "status": "error",
                        "message": f"Expected stage=critic after image submit, got {status['stage']}",
                    }
                )
                return 2

            prompt = build_next_prompt(run_dir)
            prompt_path = _write_prompt(out_dir, "next_prompt_critic.txt", prompt)
            _print(
                {
                    "status": "need_critic_json",
                    "stage": "critic",
                    "current_round": status.get("current_round", 0),
                    "prompt_path": prompt_path,
                }
            )
            return 0

        payload = {
            "status": "need_visualizer_image",
            "stage": "visualizer",
            "current_round": status.get("current_round", 0),
            "message": "Provide --image-path after generating the image manually.",
        }
        if args.emit_current_prompt:
            prompt = build_next_prompt(run_dir)
            payload["prompt_path"] = _write_prompt(
                out_dir, "next_prompt_visualizer.txt", prompt
            )
        _print(payload)
        return 0

    if stage == "critic":
        if args.image_path:
            _print(
                {
                    "status": "error",
                    "message": "stage=critic does not accept --image-path",
                }
            )
            return 2

        if args.critic_file:
            critic_text = _read_text(Path(args.critic_file))
            submit_text_output(run_dir, critic_text)
            status = get_status(run_dir)
            next_stage = status["stage"]

            if next_stage == "visualizer":
                prompt = build_next_prompt(run_dir)
                prompt_path = _write_prompt(out_dir, "next_prompt_visualizer.txt", prompt)
                _print(
                    {
                        "status": "need_new_visualizer_image",
                        "stage": "visualizer",
                        "current_round": status.get("current_round", 0),
                        "prompt_path": prompt_path,
                    }
                )
                return 0

            if next_stage == "completed":
                _print(
                    {
                        "status": "completed",
                        "stage": "completed",
                        "current_round": status.get("current_round", 0),
                        "stop_reason": status.get("stop_reason", ""),
                    }
                )
                return 0

            _print(
                {
                    "status": "error",
                    "message": f"Unexpected next stage after critic submit: {next_stage}",
                }
            )
            return 2

        payload = {
            "status": "need_critic_json",
            "stage": "critic",
            "current_round": status.get("current_round", 0),
            "message": "Provide --critic-file with strict critic JSON output.",
        }
        if args.emit_current_prompt:
            prompt = build_next_prompt(run_dir)
            payload["prompt_path"] = _write_prompt(out_dir, "next_prompt_critic.txt", prompt)
        _print(payload)
        return 0

    _print(
        {
            "status": "error",
            "stage": stage,
            "message": "This helper only supports visualizer/critic/completed stages.",
        }
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
