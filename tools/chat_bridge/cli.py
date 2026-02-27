from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.chat_bridge.session import (
    build_next_prompt,
    get_status,
    init_run,
    submit_image_for_visualizer,
    submit_text_output,
)


def _safe_text(text: str) -> str:
    return str(text).lstrip("\ufeff")


def _print_safe(text: str) -> None:
    payload = _safe_text(text)
    out = sys.stdout
    encoding = getattr(out, "encoding", None) or "utf-8"
    try:
        out.write(payload + ("\n" if not payload.endswith("\n") else ""))
    except UnicodeEncodeError:
        out.buffer.write(payload.encode(encoding, errors="replace"))
        if not payload.endswith("\n"):
            out.buffer.write(b"\n")
        out.flush()


def _read_text_from_args(file_path: str, inline_text: str) -> str:
    if file_path:
        return _safe_text(Path(file_path).read_text(encoding="utf-8-sig"))
    return _safe_text(inline_text)


def _load_input_payload(input_json: str, content_file: str, caption_file: str) -> Dict[str, Any]:
    if input_json:
        obj = json.loads(Path(input_json).read_text(encoding="utf-8-sig"))
        if "content" not in obj or "visual_intent" not in obj:
            raise ValueError("input-json must include content and visual_intent")
        return {"content": obj["content"], "visual_intent": obj["visual_intent"]}

    if not content_file or not caption_file:
        raise ValueError("Either --input-json or both --content-file and --caption-file are required")
    return {
        "content": _safe_text(Path(content_file).read_text(encoding="utf-8-sig")),
        "visual_intent": _safe_text(
            Path(caption_file).read_text(encoding="utf-8-sig")
        ).strip(),
    }


def cmd_init(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    payload = _load_input_payload(args.input_json, args.content_file, args.caption_file)
    state = init_run(
        run_dir=run_dir,
        task_name=args.task_name,
        exp_mode=args.exp_mode,
        retrieval_setting=args.retrieval_setting,
        max_critic_rounds=args.max_critic_rounds,
        content=payload["content"],
        visual_intent=payload["visual_intent"],
        work_dir=Path(args.work_dir) if args.work_dir else ROOT_DIR,
        plot_visualizer_route=args.plot_visualizer_route,
        auto_chunk_size=args.auto_chunk_size,
    )
    _print_safe(
        json.dumps(
            {"status": "ok", "run_dir": str(run_dir), "state": state},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    status = get_status(run_dir)
    _print_safe(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


def cmd_next_prompt(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    prompt = build_next_prompt(run_dir)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(prompt, encoding="utf-8")
    _print_safe(prompt)
    return 0


def cmd_submit_text(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    text = _read_text_from_args(args.file, args.text)
    state = submit_text_output(run_dir, text)
    _print_safe(
        json.dumps(
            {"status": "ok", "next_stage": get_status(run_dir)["stage"], "state": state},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_submit_image(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    image_path = Path(args.image_path)
    state = submit_image_for_visualizer(run_dir, image_path)
    _print_safe(
        json.dumps(
            {"status": "ok", "next_stage": get_status(run_dir)["stage"], "state": state},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PaperBanana chat-bridge CLI (no API-key chat workflow)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Initialize a chat-bridge run")
    p_init.add_argument("--run-dir", required=True)
    p_init.add_argument("--task-name", default="diagram", choices=["diagram", "plot"])
    p_init.add_argument("--exp-mode", default="demo_full", choices=["demo_planner_critic", "demo_full"])
    p_init.add_argument("--retrieval-setting", default="none", choices=["none", "manual", "auto"])
    p_init.add_argument("--max-critic-rounds", type=int, default=3)
    p_init.add_argument("--input-json", default="")
    p_init.add_argument("--content-file", default="")
    p_init.add_argument("--caption-file", default="")
    p_init.add_argument("--work-dir", default=str(ROOT_DIR))
    p_init.add_argument("--plot-visualizer-route", default="code", choices=["code", "image"])
    p_init.add_argument("--auto-chunk-size", type=int, default=30)
    p_init.set_defaults(func=cmd_init)

    p_status = sub.add_parser("status", help="Show current run status")
    p_status.add_argument("--run-dir", required=True)
    p_status.set_defaults(func=cmd_status)

    p_next = sub.add_parser("next-prompt", help="Print the next prompt for chat model")
    p_next.add_argument("--run-dir", required=True)
    p_next.add_argument("--out", default="")
    p_next.set_defaults(func=cmd_next_prompt)

    p_submit_text = sub.add_parser("submit-text", help="Submit text output from chat model")
    p_submit_text.add_argument("--run-dir", required=True)
    p_submit_text.add_argument("--file", default="")
    p_submit_text.add_argument("--text", default="")
    p_submit_text.set_defaults(func=cmd_submit_text)

    p_submit_image = sub.add_parser("submit-image", help="Submit generated image for visualizer stage")
    p_submit_image.add_argument("--run-dir", required=True)
    p_submit_image.add_argument("--image-path", required=True)
    p_submit_image.set_defaults(func=cmd_submit_image)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
