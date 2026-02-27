from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def copy_task(src_root: Path, dst_root: Path, task: str) -> None:
    src_task = src_root / task
    if not src_task.exists():
        raise FileNotFoundError(f"Missing source task directory: {src_task}")

    src_ref = src_task / "ref.json"
    src_images = src_task / "images"
    if not src_ref.exists():
        raise FileNotFoundError(f"Missing ref.json: {src_ref}")
    if not src_images.exists():
        raise FileNotFoundError(f"Missing images directory: {src_images}")

    dst_task = dst_root / task
    dst_task.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_ref, dst_task / "ref.json")

    dst_images = dst_task / "images"
    if dst_images.exists():
        shutil.rmtree(dst_images)
    shutil.copytree(src_images, dst_images)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy PaperBananaBench references into embedded gallery path"
    )
    parser.add_argument(
        "--source",
        default="data/PaperBananaBench",
        help="Source dataset root containing diagram/plot",
    )
    parser.add_argument(
        "--dest",
        default="tools/chat_bridge/reference_gallery/PaperBananaBench",
        help="Destination embedded gallery root",
    )
    args = parser.parse_args()

    src_root = Path(args.source).resolve()
    dst_root = Path(args.dest).resolve()

    for task in ("diagram", "plot"):
        copy_task(src_root, dst_root, task)

    print(f"Embedded gallery prepared at: {dst_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
