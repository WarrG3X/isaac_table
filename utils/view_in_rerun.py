"""View a LeRobot v2.1 dataset in Rerun.

Usage:
    python standalone_examples/user/isaac_table/utils/view_in_rerun.py

Options:
    --dataset   Repo-id style name  (default: local/piper_simple_pick)
    --root      Dataset root dir    (default: data/lerobot/)
    --episode   Episode index       (default: 0)
    --raw-dir   Optional path to raw episode dir (for ee_position 3D trajectory)
    --save      Save to .rrd file instead of spawning viewer
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import rerun as rr
import rerun.blueprint as rrb
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_ROOT = PROJECT_DIR / "data" / "lerobot"
DEFAULT_RAW_ROOT = PROJECT_DIR / "data" / "simple_pick_raw"


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    """Convert CHW float32 [0,1] torch tensor to HWC uint8 numpy array."""
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expected CHW, got {chw_float32_torch.shape}"
    if chw_float32_torch.dtype == torch.float32:
        return (chw_float32_torch * 255).to(torch.uint8).permute(1, 2, 0).numpy()
    return chw_float32_torch.permute(1, 2, 0).numpy()


def find_latest_episode(raw_root: Path) -> Path | None:
    if not raw_root.exists():
        return None
    episodes = sorted(
        [d for d in raw_root.iterdir() if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    return episodes[-1] if episodes else None


def load_ee_positions(raw_dir: Path) -> np.ndarray | None:
    """Load ee_position from raw episode data.npz if available."""
    npz_path = raw_dir / "data.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    if "ee_position" in data:
        return data["ee_position"]
    return None


def load_raw_episode(raw_dir: Path) -> dict | None:
    t0 = time.perf_counter()
    npz_path = raw_dir / "data.npz"
    meta_path = raw_dir / "metadata.json"
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    metadata = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    top_dir = raw_dir / "images" / "top"
    wrist_dir = raw_dir / "images" / "wrist"
    image_ext = metadata.get("image_format")
    if image_ext:
        image_ext = "." + str(image_ext).lstrip(".")
    elif top_dir.exists():
        sample = next(top_dir.iterdir(), None)
        image_ext = sample.suffix if sample else ".png"
    else:
        image_ext = ".png"
    result = {
        "timestamp": np.asarray(data["timestamp"], dtype=np.float64),
        "frame_index": np.asarray(data["step_index"], dtype=np.int64),
        "state": np.asarray(data["joint_position"], dtype=np.float32),
        "action": np.concatenate(
            [
                np.asarray(data["ee_delta"], dtype=np.float32),
                np.asarray(data["gripper_action"], dtype=np.float32).reshape(-1, 1),
            ],
            axis=1,
        ),
        "ee_position": np.asarray(data["ee_position"], dtype=np.float32) if "ee_position" in data else None,
        "top_dir": top_dir,
        "wrist_dir": wrist_dir,
        "image_ext": image_ext,
    }
    t1 = time.perf_counter()
    print(f"[RerunView] raw episode loaded in {t1 - t0:.2f}s from {raw_dir}")
    return result


def view(repo_id: str, root: Path, episode_idx: int, raw_dir: Path | None, save_path: Path | None = None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    overall_t0 = time.perf_counter()
    raw_episode = load_raw_episode(raw_dir) if raw_dir else None
    ds_path = root / repo_id
    print(f"[RerunView] dataset={repo_id}")
    print(f"[RerunView] dataset_path={ds_path}")
    print(f"[RerunView] raw_dir={raw_dir if raw_dir else 'none'}")
    print(f"[RerunView] save_path={save_path if save_path else 'viewer'}")
    dataset_t0 = time.perf_counter()
    dataset = LeRobotDataset(repo_id=repo_id, root=ds_path)
    dataset_t1 = time.perf_counter()
    print(f"[RerunView] LeRobotDataset init took {dataset_t1 - dataset_t0:.2f}s")

    # find episode bounds using episode_data_index
    index_t0 = time.perf_counter()
    ep_start = dataset.episode_data_index["from"][episode_idx].item()
    ep_end = dataset.episode_data_index["to"][episode_idx].item()
    num_frames = ep_end - ep_start
    index_t1 = time.perf_counter()
    print(f"[RerunView] episode={episode_idx} frames={num_frames} index_lookup={index_t1 - index_t0:.2f}s")

    # load 3D trajectory from raw data
    ee_positions = None
    if raw_episode is not None:
        ee_positions = raw_episode["ee_position"]
        if ee_positions is not None:
            print(f"[RerunView] ee_position shape={ee_positions.shape}")
    elif raw_dir:
        ee_positions = load_ee_positions(raw_dir)
        if ee_positions is not None:
            print(f"[RerunView] ee_position shape={ee_positions.shape}")

    # --- blueprint layout ---
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            # top row: cameras side by side
            rrb.Horizontal(
                rrb.Spatial2DView(name="Top Camera", origin="camera/top"),
                rrb.Spatial2DView(name="Wrist Camera", origin="camera/wrist"),
                column_shares=[1, 1],
            ),
            # middle row: 3D trajectory + joint states
            rrb.Horizontal(
                rrb.Spatial3DView(name="EE Trajectory", origin="trajectory"),
                rrb.TimeSeriesView(
                    name="Joint Positions",
                    origin="state",
                ),
                column_shares=[1, 1],
            ),
            # bottom row: action deltas and gripper (separate plots)
            rrb.Horizontal(
                rrb.TimeSeriesView(
                    name="Action Deltas",
                    origin="action/delta",
                ),
                rrb.TimeSeriesView(
                    name="Gripper",
                    origin="action/gripper",
                ),
                column_shares=[3, 1],
            ),
            row_shares=[4, 3, 2],
        ),
        rrb.SelectionPanel(state="collapsed"),
        rrb.BlueprintPanel(state="collapsed"),
    )

    rr.init("piper_simple_pick", spawn=save_path is None)
    rr.send_blueprint(blueprint)

    # --- log full trajectory upfront (static, not per-frame) ---
    if ee_positions is not None:
        rr.log("trajectory/path", rr.LineStrips3D(
            [ee_positions],
            colors=[[100, 100, 255]],
            radii=[0.001],
        ), static=True)

    frame_loop_t0 = time.perf_counter()
    print("[RerunView] begin frame logging")
    for idx in range(ep_start, ep_end):
        local_i = idx - ep_start
        if raw_episode is not None:
            frame_i = int(raw_episode["frame_index"][local_i])
            timestamp = float(raw_episode["timestamp"][local_i])
            state = raw_episode["state"][local_i]
            action = raw_episode["action"][local_i]
        else:
            sample = dataset[idx]
            frame_i = sample["frame_index"].item()
            timestamp = sample["timestamp"].item()
            state = sample["observation.state"].numpy()
            action = sample["action"].numpy()

        rr.set_time_sequence("frame", frame_i)
        rr.set_time_seconds("time", timestamp)

        # --- images ---
        if raw_episode is not None:
            top_path = raw_episode["top_dir"] / f"{local_i:06d}{raw_episode['image_ext']}"
            wrist_path = raw_episode["wrist_dir"] / f"{local_i:06d}{raw_episode['image_ext']}"
            if top_path.exists():
                rr.log("camera/top", rr.Image(np.asarray(Image.open(top_path))))
            if wrist_path.exists():
                rr.log("camera/wrist", rr.Image(np.asarray(Image.open(wrist_path))))
        else:
            if "observation.images.top" in sample:
                rr.log("camera/top", rr.Image(to_hwc_uint8_numpy(sample["observation.images.top"])))

            if "observation.images.wrist" in sample:
                rr.log("camera/wrist", rr.Image(to_hwc_uint8_numpy(sample["observation.images.wrist"])))

        # --- joint state ---
        for j in range(len(state)):
            rr.log(f"state/joint_{j}", rr.Scalar(float(state[j])))

        # --- actions: deltas separate from gripper ---
        rr.log("action/delta/dx", rr.Scalar(float(action[0])))
        rr.log("action/delta/dy", rr.Scalar(float(action[1])))
        rr.log("action/delta/dz", rr.Scalar(float(action[2])))
        if len(action) > 3:
            rr.log("action/delta/drx", rr.Scalar(float(action[3])))
            rr.log("action/delta/dry", rr.Scalar(float(action[4])))
            rr.log("action/delta/drz", rr.Scalar(float(action[5])))
        if len(action) > 6:
            rr.log("action/gripper/value", rr.Scalar(float(action[6])))

        # --- 3D: current EE position as a moving point ---
        if ee_positions is not None:
            if local_i < len(ee_positions):
                pos = ee_positions[local_i]
                rr.log("trajectory/current", rr.Points3D(
                    [pos],
                    colors=[[255, 50, 50]],
                    radii=[0.005],
                ))

        if (idx - ep_start + 1) % 100 == 0 or idx == ep_end - 1:
            now = time.perf_counter()
            done = idx - ep_start + 1
            elapsed = now - frame_loop_t0
            rate = done / elapsed if elapsed > 0 else 0.0
            print(f"[RerunView] logged {done}/{num_frames} frames in {elapsed:.2f}s ({rate:.1f} fps)")

    if save_path:
        save_t0 = time.perf_counter()
        print("[RerunView] begin rrd save")
        rr.save(str(save_path))
        save_t1 = time.perf_counter()
        print(f"[RerunView] save complete in {save_t1 - save_t0:.2f}s")
        print(f"[RerunView] saved to {save_path}")
        print(f"[RerunView] total elapsed {save_t1 - overall_t0:.2f}s")
        print(f"Open with: rerun {save_path}")
    else:
        overall_t1 = time.perf_counter()
        print(f"[RerunView] viewer logging complete in {overall_t1 - overall_t0:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="View LeRobot dataset in Rerun")
    parser.add_argument("--dataset", type=str, default="local/piper_simple_pick",
                        help="Repo-id style dataset name")
    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT),
                        help="Dataset root dir")
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode index to view")
    parser.add_argument("--raw-dir", type=str, default=None,
                        help="Path to raw episode dir for ee_position 3D trajectory (default: auto-detect)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save to .rrd file instead of spawning viewer (e.g. --save episode.rrd)")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir) if args.raw_dir else find_latest_episode(DEFAULT_RAW_ROOT)
    if raw_dir:
        print(f"Raw episode for 3D trajectory: {raw_dir}")

    save_path = Path(args.save) if args.save else None
    view(args.dataset, Path(args.root), args.episode, raw_dir, save_path)


if __name__ == "__main__":
    main()
