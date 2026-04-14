"""Convert a simple_pick_raw episode to a LeRobot v2.1 dataset.

Usage (from the Isaac Sim install root):
    python standalone_examples/user/isaac_table/utils/convert_to_lerobot.py

Options:
    --raw-dir   Path to raw episode dir  (default: auto-detect latest in data/simple_pick_raw/)
    --output    Repo-id style name       (default: local/piper_simple_pick)
    --root      Where to write dataset   (default: data/lerobot/)
    --fps       Override FPS             (default: inferred from timestamps)
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_RAW_ROOT = PROJECT_DIR / "data" / "simple_pick_raw"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "data" / "lerobot"


def find_latest_episode(raw_root: Path) -> Path:
    """Return the most-recently-modified episode directory."""
    episodes = sorted(
        [d for d in raw_root.iterdir() if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    if not episodes:
        sys.exit(f"No episode directories found in {raw_root}")
    return episodes[-1]


def infer_fps(timestamps: np.ndarray) -> int:
    dt = np.median(np.diff(timestamps))
    return int(round(1.0 / dt))


def load_raw_episode(episode_dir: Path):
    """Load raw episode data and return a dict of arrays + metadata."""
    data = dict(np.load(episode_dir / "data.npz"))

    meta_path = episode_dir / "metadata.json"
    metadata = {}
    if meta_path.exists():
        raw = meta_path.read_bytes()
        # handle truncated json gracefully
        try:
            metadata = json.loads(raw)
        except json.JSONDecodeError:
            print(f"[warn] metadata.json is truncated, using defaults")

    # detect image format
    top_dir = episode_dir / "images" / "top"
    sample = next(top_dir.iterdir())
    img_ext = sample.suffix  # .png or .jpg

    return data, metadata, img_ext


def convert(raw_dir: Path, repo_id: str, output_root: Path, fps_override: int | None):
    data, metadata, img_ext = load_raw_episode(raw_dir)

    timestamps = data["timestamp"]
    fps = fps_override or infer_fps(timestamps)
    num_steps = len(timestamps)
    task_name = metadata.get("task", "simple_pick_red_to_blue")

    print(f"Episode:  {raw_dir.name}")
    print(f"Steps:    {num_steps}")
    print(f"FPS:      {fps}")
    print(f"Task:     {task_name}")

    # --- read one image to get resolution ---
    sample_img = Image.open(raw_dir / "images" / "top" / f"000000{img_ext}")
    img_h, img_w = sample_img.height, sample_img.width
    print(f"Image:    {img_w}x{img_h}")

    # --- build dataset via LeRobot API ---
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # root= is used as the full dataset directory by LeRobotDataset.create().
    ds_path = output_root / repo_id
    if ds_path.exists():
        shutil.rmtree(ds_path)

    # joint_position is (N, 8) — 6 arm joints + 2 gripper joints
    # ee_delta is (N, 6) — 3 translation + 3 rotation (the commanded action)
    # we store observation.state = joint_position (8)
    # and action = ee_delta (6) + gripper_action (1) = 7
    joint_dim = data["joint_position"].shape[1]
    action_dim = data["ee_delta"].shape[1] + 1  # +1 for gripper

    features = {
        "observation.images.top": {
            "dtype": "image",
            "shape": (img_h, img_w, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist": {
            "dtype": "image",
            "shape": (img_h, img_w, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (joint_dim,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["action"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=ds_path,
        robot_type="piper_x",
        use_videos=False,
        image_writer_threads=4,
    )

    print("Converting frames...")
    for i in range(num_steps):
        top_img = np.array(Image.open(raw_dir / "images" / "top" / f"{i:06d}{img_ext}"))
        wrist_img = np.array(Image.open(raw_dir / "images" / "wrist" / f"{i:06d}{img_ext}"))

        # action = ee_delta (6) + gripper_action (1)
        action = np.concatenate([
            data["ee_delta"][i],
            data["gripper_action"][i:i+1].astype(np.float32),
        ])

        dataset.add_frame(
            {
                "observation.images.top": top_img,
                "observation.images.wrist": wrist_img,
                "observation.state": data["joint_position"][i],
                "action": action,
            },
            task=task_name,
        )

        if (i + 1) % 100 == 0 or i == num_steps - 1:
            print(f"  {i+1}/{num_steps}")

    dataset.save_episode()

    print(f"\nDone! LeRobot v2.1 dataset written to:\n  {ds_path}")
    return ds_path


def main():
    parser = argparse.ArgumentParser(description="Convert raw episode to LeRobot v2.1")
    parser.add_argument("--raw-dir", type=str, default=None,
                        help="Path to raw episode dir (default: latest in data/simple_pick_raw/)")
    parser.add_argument("--output", type=str, default="local/piper_simple_pick",
                        help="Repo-id style dataset name")
    parser.add_argument("--root", type=str, default=str(DEFAULT_OUTPUT_ROOT),
                        help="Root dir for output dataset")
    parser.add_argument("--fps", type=int, default=None,
                        help="Override FPS (default: inferred from timestamps)")
    args = parser.parse_args()

    if args.raw_dir:
        raw_dir = Path(args.raw_dir)
    else:
        raw_dir = find_latest_episode(DEFAULT_RAW_ROOT)

    if not raw_dir.exists():
        sys.exit(f"Episode dir not found: {raw_dir}")

    convert(raw_dir, args.output, Path(args.root), args.fps)


if __name__ == "__main__":
    main()
