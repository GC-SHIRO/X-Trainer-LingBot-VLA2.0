#!/usr/bin/env python
"""Convert raw Dobot X-trainer recordings to a LeRobot Dataset v3.0 tree (lerobot==0.4.2).

Expected raw layout::

    collect_data/<episode_id>/
      topImg/<frame_id>.jpg
      leftImg/<frame_id>.jpg
      rightImg/<frame_id>.jpg
      observation/<frame_id>.pkl  # {"joint_positions": ..., "control": ...}

Adapted from XVLA scripts/xtrainer/convert_raw_to_lerobot_2_1.py.
The output is local-only. Each episode has its own data/video file, with
v3.0 Parquet task and episode metadata for LingBot training.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import datasets
import numpy as np
import pandas as pd
from PIL import Image

from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.utils import (
    DEFAULT_FEATURES,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    embed_images,
    flatten_dict,
    get_hf_features_from_features,
    serialize_dict,
    write_episodes,
    write_tasks,
)
from lerobot.datasets.video_utils import encode_video_frames, get_video_info


CODEBASE_VERSION = "v3.0"
CHUNK_SIZE = 1000
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
# ``encode_video_frames`` discovers RGB frames using ``frame-000000.png``.
TEMP_IMAGE_PATH = "images/{image_key}/episode_{episode_index:06d}/frame-{frame_index:06d}.png"
CAMERAS = {
    "observation.images.top": "topImg",
    "observation.images.left_wrist": "leftImg",
    "observation.images.right_wrist": "rightImg",
}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4, ensure_ascii=False)


def _frame_id(path: Path) -> int | None:
    return int(path.stem) if path.stem.isdigit() else None


def _frame_ids(folder: Path, suffixes: tuple[str, ...]) -> list[int]:
    ids = {
        frame_id
        for suffix in suffixes
        for path in folder.glob(f"*{suffix}")
        if (frame_id := _frame_id(path)) is not None
    }
    return sorted(ids)


def _resolve_image_path(folder: Path, frame_id: int) -> Path:
    for suffix in (".jpg", ".jpeg", ".png"):
        path = folder / f"{frame_id}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"Image for frame {frame_id} not found in {folder}")


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Could not decode image: {path}")
    return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _build_joint_names(dim: int) -> list[str]:
    if dim == 14:
        return (
            [f"left_joint{i}.pos" for i in range(1, 7)]
            + ["left_gripper.pos"]
            + [f"right_joint{i}.pos" for i in range(1, 7)]
            + ["right_gripper.pos"]
        )
    return [f"joint_{index}.pos" for index in range(dim)]


def _first_observation(episode_dirs: list[Path]) -> dict:
    for episode_dir in episode_dirs:
        for path in sorted((episode_dir / "observation").glob("*.pkl")):
            with path.open("rb") as file:
                payload = pickle.load(file)
            if "joint_positions" in payload and "control" in payload:
                return payload
    raise RuntimeError("No observation .pkl containing joint_positions and control was found.")


def _first_image_shape(episode_dirs: list[Path], raw_camera_dir: str) -> tuple[int, int, int]:
    for episode_dir in episode_dirs:
        image_dir = episode_dir / raw_camera_dir
        for path in sorted(image_dir.glob("*")) if image_dir.is_dir() else []:
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            image = _read_rgb(path)
            return (int(image.shape[0]), int(image.shape[1]), 3)
    raise RuntimeError(f"No valid RGB image found for raw camera directory {raw_camera_dir!r}.")


def _build_features(
    action_dim: int, state_dim: int, image_shapes: dict[str, tuple[int, int, int]], use_videos: bool
) -> dict:
    if action_dim != state_dim:
        raise ValueError(f"Action/state dimensions differ: {action_dim} vs {state_dim}")
    names = _build_joint_names(action_dim)
    image_dtype = "video" if use_videos else "image"
    features = {
        "action": {"dtype": "float32", "shape": (action_dim,), "names": names},
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": names},
    }
    for key, shape in image_shapes.items():
        features[key] = {
            "dtype": image_dtype,
            "shape": shape,
            "names": ["height", "width", "channels"],
        }
    return features


def _episode_data_path(root: Path, episode_index: int) -> Path:
    return root / DATA_PATH.format(chunk_index=episode_index // CHUNK_SIZE, file_index=episode_index % CHUNK_SIZE)


def _episode_video_path(root: Path, episode_index: int, camera_key: str) -> Path:
    return root / VIDEO_PATH.format(
        chunk_index=episode_index // CHUNK_SIZE,
        file_index=episode_index % CHUNK_SIZE,
        video_key=camera_key,
    )


def _temporary_image_path(root: Path, episode_index: int, camera_key: str, frame_index: int) -> Path:
    return root / TEMP_IMAGE_PATH.format(
        image_key=camera_key, episode_index=episode_index, frame_index=frame_index
    )


def _cleanup_images(root: Path, episode_index: int) -> None:
    for camera_key in CAMERAS:
        image_dir = _temporary_image_path(root, episode_index, camera_key, 0).parent
        if image_dir.is_dir():
            shutil.rmtree(image_dir)


def _write_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    Image.fromarray(rgb).save(temporary, format="PNG")
    temporary.replace(path)


def _write_verified_png(path: Path, source_path: Path, repair_retries: int) -> None:
    for attempt in range(repair_retries + 1):
        _write_png(path, _read_rgb(source_path))
        decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if decoded is not None:
            return
        if attempt < repair_retries:
            print(f"[warn] Repair temporary image {attempt + 1}/{repair_retries}: {path}")
    raise OSError(f"Could not write a readable temporary image: {path}")


def _encode_video(
    image_dir: Path, video_path: Path, fps: int, vcodec: str, use_subprocess: bool, quiet: bool
) -> None:
    """Encode in a child process to isolate FFmpeg/torchcodec native crashes."""
    if not use_subprocess:
        encode_video_frames(
            imgs_dir=image_dir,
            video_path=video_path,
            fps=fps,
            vcodec=vcodec,
            overwrite=True,
            log_level=None,
        )
        return
    child = (
        "from pathlib import Path\n"
        "import sys\n"
        "from lerobot.datasets.video_utils import encode_video_frames\n"
        "encode_video_frames(imgs_dir=Path(sys.argv[1]), video_path=Path(sys.argv[2]), "
        "fps=int(sys.argv[3]), vcodec=sys.argv[4], "
        "overwrite=True, log_level=None)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", child, str(image_dir), str(video_path), str(fps), vcodec],
        check=False,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.DEVNULL if quiet else None,
    )
    if completed.returncode:
        raise RuntimeError(f"Video encoding failed for {video_path} (exit code {completed.returncode}).")


def convert(args: argparse.Namespace) -> None:
    raw_root = Path(args.raw_root).resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw root does not exist or is not a directory: {raw_root}")
    episode_dirs = sorted(path for path in raw_root.iterdir() if path.is_dir())
    if not episode_dirs:
        raise RuntimeError(f"No episode directories found under {raw_root}")

    first_observation = _first_observation(episode_dirs)
    action_dim = int(np.asarray(first_observation["control"]).reshape(-1).shape[0])
    state_dim = int(np.asarray(first_observation["joint_positions"]).reshape(-1).shape[0])
    image_shapes = {key: _first_image_shape(episode_dirs, name) for key, name in CAMERAS.items()}
    features = {**_build_features(action_dim, state_dim, image_shapes, args.use_videos), **DEFAULT_FEATURES}
    hf_features = get_hf_features_from_features(features)

    output_root = Path(args.output_root).resolve()
    if output_root == raw_root or output_root in raw_root.parents or raw_root in output_root.parents:
        raise ValueError("Raw and output directories must not contain each other.")
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite_output:
            raise FileExistsError(
                f"Output root is non-empty: {output_root}. Pass --overwrite-output to replace it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_stats: list[dict] = []
    episode_rows: list[dict] = []
    total_frames = 0
    saved_episodes = 0

    for raw_episode in episode_dirs:
        observation_dir = raw_episode / "observation"
        camera_dirs = {key: raw_episode / name for key, name in CAMERAS.items()}
        if not observation_dir.is_dir() or not all(path.is_dir() for path in camera_dirs.values()):
            print(f"[skip] Missing observation or camera directory: {raw_episode}")
            continue

        ids = set(_frame_ids(observation_dir, (".pkl",)))
        for camera_dir in camera_dirs.values():
            ids &= set(_frame_ids(camera_dir, (".jpg", ".jpeg", ".png")))
        frame_ids = sorted(ids)[args.skip_first_frames :]
        if len(frame_ids) < args.min_frames:
            print(f"[skip] Too few aligned frames ({len(frame_ids)}): {raw_episode.name}")
            continue

        episode_index = saved_episodes
        actions: list[np.ndarray] = []
        states: list[np.ndarray] = []
        image_paths = {key: [] for key in CAMERAS}
        skipped_bad = 0
        for raw_frame_id in frame_ids:
            try:
                with (observation_dir / f"{raw_frame_id}.pkl").open("rb") as file:
                    observation = pickle.load(file)
                action = np.asarray(observation["control"], dtype=np.float32).reshape(-1)
                state = np.asarray(observation["joint_positions"], dtype=np.float32).reshape(-1)
                if action.shape != (action_dim,) or state.shape != (state_dim,):
                    raise ValueError(f"Expected action/state {action_dim}/{state_dim}; got {action.size}/{state.size}")
                if not np.isfinite(action).all() or not np.isfinite(state).all():
                    raise ValueError("Action or state contains NaN/Inf")

                frame_index = len(actions)
                frame_paths = {}
                for key, camera_dir in camera_dirs.items():
                    destination = _temporary_image_path(output_root, episode_index, key, frame_index)
                    _write_verified_png(
                        destination, _resolve_image_path(camera_dir, raw_frame_id), args.repair_retries
                    )
                    with Image.open(destination) as image:
                        if (image.height, image.width, 3) != image_shapes[key]:
                            raise ValueError(f"Image shape changed for {key}: {destination}")
                    frame_paths[key] = str(destination)
                for key, path in frame_paths.items():
                    image_paths[key].append(path)
                actions.append(action)
                states.append(state)
            except Exception as error:
                # Remove partial camera writes so an invalid final frame cannot
                # become an extra frame in an encoded video.
                for key in CAMERAS:
                    _temporary_image_path(output_root, episode_index, key, len(actions)).unlink(missing_ok=True)
                skipped_bad += 1
                if not args.skip_bad_frames:
                    _cleanup_images(output_root, episode_index)
                    raise RuntimeError(f"Invalid frame {raw_episode.name}/{raw_frame_id}") from error
                print(f"[warn] Skip invalid frame {raw_episode.name}/{raw_frame_id}: {error}")

        if len(actions) < args.min_frames:
            print(f"[skip] Too few valid frames after filtering ({len(actions)}): {raw_episode.name}")
            _cleanup_images(output_root, episode_index)
            continue

        length = len(actions)
        episode_buffer = {
            "action": np.stack(actions).astype(np.float32),
            "observation.state": np.stack(states).astype(np.float32),
            "frame_index": np.arange(length, dtype=np.int64),
            "timestamp": np.arange(length, dtype=np.float32) / float(args.fps),
            "episode_index": np.full(length, episode_index, dtype=np.int64),
            "index": np.arange(total_frames, total_frames + length, dtype=np.int64),
            "task_index": np.zeros(length, dtype=np.int64),
            **image_paths,
        }
        parquet_payload = {key: episode_buffer[key] for key in hf_features}
        episode_dataset = datasets.Dataset.from_dict(parquet_payload, features=hf_features, split="train")
        if not args.use_videos:
            episode_dataset = embed_images(episode_dataset)
        data_path = _episode_data_path(output_root, episode_index)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        episode_dataset.to_parquet(data_path)

        stats = compute_episode_stats(episode_buffer, features)
        all_stats.append(stats)
        episode_row = {
            "episode_index": episode_index,
            "tasks": [args.task],
            "length": length,
            "data/chunk_index": episode_index // CHUNK_SIZE,
            "data/file_index": episode_index % CHUNK_SIZE,
            "dataset_from_index": total_frames,
            "dataset_to_index": total_frames + length,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
            **flatten_dict({"stats": serialize_dict(stats)}),
        }
        episode_rows.append(episode_row)

        if args.use_videos:
            for camera_key, paths in image_paths.items():
                video_path = _episode_video_path(output_root, episode_index, camera_key)
                for attempt in range(args.encode_retries + 1):
                    try:
                        _encode_video(
                            Path(paths[0]).parent,
                            video_path,
                            args.fps,
                            args.vcodec,
                            args.encode_in_subprocess,
                            args.quiet_encoder,
                        )
                        break
                    except RuntimeError:
                        if attempt == args.encode_retries:
                            raise
                        print(
                            f"[warn] Retry video encoding {attempt + 1}/{args.encode_retries}: "
                            f"{raw_episode.name}/{camera_key}"
                        )
                features[camera_key]["info"] = get_video_info(video_path)
                episode_row.update({
                    f"videos/{camera_key}/chunk_index": episode_index // CHUNK_SIZE,
                    f"videos/{camera_key}/file_index": episode_index % CHUNK_SIZE,
                    f"videos/{camera_key}/from_timestamp": 0.0,
                    f"videos/{camera_key}/to_timestamp": length / float(args.fps),
                })
            if not args.keep_images_for_video:
                _cleanup_images(output_root, episode_index)

        total_frames += length
        saved_episodes += 1
        print(f"[ok] Saved episode {saved_episodes}: {raw_episode.name} ({length} frames; skipped={skipped_bad})")

    if not saved_episodes:
        raise RuntimeError("No episodes were converted; inspect the raw layout and frame files.")

    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": args.robot_type,
        "total_episodes": saved_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "data_files_size_in_mb": DEFAULT_DATA_FILE_SIZE_IN_MB,
        "video_files_size_in_mb": DEFAULT_VIDEO_FILE_SIZE_IN_MB,
        "chunks_size": CHUNK_SIZE,
        "fps": args.fps,
        "splits": {"train": f"0:{saved_episodes}"},
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH if args.use_videos else None,
        "features": features,
    }
    _write_json(output_root / "meta/info.json", info)
    _write_json(output_root / "meta/stats.json", serialize_dict(aggregate_stats(all_stats)))
    write_tasks(pd.DataFrame({"task_index": [0]}, index=[args.task]), output_root)
    write_episodes(datasets.Dataset.from_list(episode_rows), output_root)
    print(f"Done. Converted {saved_episodes} episodes to: {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw Dobot X-trainer recordings to LeRobot v3.0.")
    parser.add_argument("--raw-root", "--raw_root", dest="raw_root", required=True, help="Raw collect_data directory.")
    parser.add_argument(
        "--output-root",
        "--output_root",
        dest="output_root",
        required=True,
        help="Destination LeRobot Dataset v3.0 directory.",
    )
    parser.add_argument("--repo-id", "--repo_id", dest="repo_id", default="local/dobot_xtrainer_converted_v3")
    parser.add_argument("--task", default="Insert the test tube on the desktop into the rack.")
    parser.add_argument("--robot-type", "--robot_type", dest="robot_type", default="dobot_xtrainer_bimanual")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--use-videos",
        "--use_videos",
        dest="use_videos",
        action="store_true",
        help="Encode camera frames as MP4 (default).",
    )
    parser.add_argument(
        "--no-videos", "--no_videos", dest="use_videos", action="store_false", help="Embed images in Parquet instead."
    )
    parser.set_defaults(use_videos=True)
    parser.add_argument("--vcodec", default="h264", choices=["h264", "hevc", "libsvtav1"])
    parser.add_argument("--repair-retries", "--repair_retries", dest="repair_retries", type=int, default=2)
    parser.add_argument("--encode-retries", "--encode_retries", dest="encode_retries", type=int, default=1)
    parser.add_argument(
        "--encode-in-subprocess", "--encode_in_subprocess", dest="encode_in_subprocess", action="store_true"
    )
    parser.add_argument("--encode-in-process", "--encode_in_process", dest="encode_in_subprocess", action="store_false")
    parser.set_defaults(encode_in_subprocess=True)
    parser.add_argument("--skip-first-frames", "--skip_first_frames", dest="skip_first_frames", type=int, default=0)
    parser.add_argument("--min-frames", "--min_frames", dest="min_frames", type=int, default=10)
    parser.add_argument(
        "--skip-bad-frames",
        "--skip_bad_frames",
        dest="skip_bad_frames",
        action="store_true",
        help="Skip corrupt frames (default).",
    )
    parser.add_argument("--fail-on-bad-frames", "--fail_on_bad_frames", dest="skip_bad_frames", action="store_false")
    parser.set_defaults(skip_bad_frames=True)
    parser.add_argument(
        "--keep-images-for-video",
        "--keep_images_for_video",
        dest="keep_images_for_video",
        action="store_true",
        help="Keep temporary PNGs after MP4 encoding.",
    )
    parser.add_argument(
        "--quiet-encoder",
        "--quiet_encoder",
        dest="quiet_encoder",
        action="store_true",
        help="Hide encoder output (default).",
    )
    parser.add_argument("--verbose-encoder", "--verbose_encoder", dest="quiet_encoder", action="store_false")
    parser.set_defaults(quiet_encoder=True)
    parser.add_argument(
        "--overwrite-output",
        "--overwrite_output",
        dest="overwrite_output",
        action="store_true",
        help="Replace a non-empty output directory.",
    )
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.min_frames <= 0:
        parser.error("--min-frames must be positive")
    if args.skip_first_frames < 0:
        parser.error("--skip-first-frames must be non-negative")
    if args.encode_retries < 0:
        parser.error("--encode-retries must be non-negative")
    if args.repair_retries < 0:
        parser.error("--repair-retries must be non-negative")
    return args


if __name__ == "__main__":
    convert(parse_args())
