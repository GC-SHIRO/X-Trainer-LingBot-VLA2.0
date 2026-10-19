#!/usr/bin/env python
"""Create an orientation-corrected copy of a video-backed X-trainer dataset.

The source is a LeRobot v2.1 dataset and is never modified. The output keeps
top and left-wrist videos unchanged and rotates right-wrist videos 180 degrees.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

import numpy as np


CAMERA_KEYS = (
    "observation.images.top",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)
RIGHT_WRIST_FILTER = "vflip,hflip"


def transform_frame(camera_key: str, image: np.ndarray) -> np.ndarray:
    """Apply the dataset camera-orientation rule to a single HWC RGB image."""
    if camera_key in CAMERA_KEYS[:2]:
        return np.ascontiguousarray(image)
    if camera_key == CAMERA_KEYS[2]:
        return np.ascontiguousarray(image[::-1, ::-1])
    raise KeyError(f"Unsupported camera key: {camera_key}")


def _video_paths(root: Path, camera_key: str) -> dict[Path, Path]:
    videos_root = root / "videos"
    paths = sorted(videos_root.glob(f"chunk-*/{camera_key}/episode_*.mp4"))
    return {path.relative_to(videos_root).parent.parent / path.name: path for path in paths}


def _validate_source(root: Path) -> dict[str, dict[Path, Path]]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing dataset metadata: {info_path}")
    with info_path.open(encoding="utf-8") as file:
        info = json.load(file)

    features = info.get("features", {})
    invalid = [key for key in CAMERA_KEYS if features.get(key, {}).get("dtype") != "video"]
    if invalid:
        raise ValueError(
            "This tool supports only video-backed X-trainer v2.1 datasets; "
            f"these camera features are missing or not videos: {', '.join(invalid)}"
        )

    videos = {key: _video_paths(root, key) for key in CAMERA_KEYS}
    expected = set(videos[CAMERA_KEYS[0]])
    if not expected:
        raise RuntimeError("No top-camera MP4 files found under videos/chunk-*/.")
    for key, paths in videos.items():
        found = set(paths)
        if found != expected:
            missing = sorted(str(path) for path in expected - found)
            extra = sorted(str(path) for path in found - expected)
            raise ValueError(f"Video files for {key} do not match the top camera (missing={missing}, extra={extra})")
    return videos


def _assert_safe_output(source: Path, output: Path) -> None:
    if source == output or output in source.parents or source in output.parents:
        raise ValueError("--output-root must be separate from --input-root and cannot be its parent or child")


def _prepare_output(output: Path, overwrite: bool) -> None:
    if not output.exists():
        return
    if not overwrite:
        raise FileExistsError(f"Output path already exists: {output}. Pass --overwrite-output to replace it.")
    if output.is_dir():
        shutil.rmtree(output)
    else:
        output.unlink()


def _rotate_video(path: Path, crf: int, preset: str) -> None:
    temporary = path.with_name(f".{path.stem}.transforming.mp4")
    try:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-vf",
                RIGHT_WRIST_FILTER,
                "-c:v",
                "libx264",
                "-crf",
                str(crf),
                "-preset",
                preset,
                "-an",
                "-movflags",
                "+faststart",
                str(temporary),
            ],
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"ffmpeg failed ({completed.returncode}) while transforming {path}")
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg created no video output for {path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def transform_dataset(
    input_root: Path,
    output_root: Path,
    *,
    overwrite_output: bool = False,
    crf: int = 18,
    preset: str = "medium",
    dry_run: bool = False,
) -> None:
    input_root = input_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input dataset directory does not exist: {input_root}")
    _assert_safe_output(input_root, output_root)
    videos = _validate_source(input_root)
    episode_count = len(videos[CAMERA_KEYS[0]])

    if dry_run:
        print(f"Validated {episode_count} episodes in {input_root}")
        print("Would copy the dataset, preserve top and left videos, and rotate right videos 180 degrees.")
        return

    _prepare_output(output_root, overwrite_output)
    shutil.copytree(input_root, output_root)
    for source_video in videos[CAMERA_KEYS[2]].values():
        output_video = output_root / "videos" / source_video.relative_to(input_root / "videos")
        _rotate_video(output_video, crf, preset)

    print(f"Created transformed dataset: {output_root}")
    print(f"Episodes: {episode_count}; top=unchanged, left=unchanged, right=vflip+hflip")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an X-trainer dataset copy with corrected camera orientation.")
    parser.add_argument("--input-root", required=True, help="Existing video-backed LeRobot v2.1 dataset directory")
    parser.add_argument("--output-root", required=True, help="New output dataset directory; source is never modified")
    parser.add_argument("--overwrite-output", action="store_true", help="Replace an existing output directory")
    parser.add_argument("--crf", type=int, default=18, help="H.264 quality setting for transformed videos")
    parser.add_argument("--preset", default="medium", help="FFmpeg libx264 preset")
    parser.add_argument("--dry-run", action="store_true", help="Validate only; do not create output")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    transform_dataset(
        Path(args.input_root),
        Path(args.output_root),
        overwrite_output=args.overwrite_output,
        crf=args.crf,
        preset=args.preset,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
