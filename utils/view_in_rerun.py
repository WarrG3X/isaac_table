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
from pathlib import Path

import numpy as np
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


def view(repo_id: str, root: Path, episode_idx: int, raw_dir: Path | None, save_path: Path | None = None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds_path = root / repo_id
    print(f"Loading dataset: {repo_id} (path={ds_path})")
    dataset = LeRobotDataset(repo_id=repo_id, root=ds_path)

    # find episode bounds using episode_data_index
    ep_start = dataset.episode_data_index["from"][episode_idx].item()
    ep_end = dataset.episode_data_index["to"][episode_idx].item()
    num_frames = ep_end - ep_start
    print(f"Episode {episode_idx}: {num_frames} frames")

    # load 3D trajectory from raw data
    ee_positions = None
    if raw_dir:
        ee_positions = load_ee_positions(raw_dir)
        if ee_positions is not None:
            print(f"Loaded ee_position trajectory: {ee_positions.shape}")

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

    print("Logging to Rerun...")
    for idx in range(ep_start, ep_end):
        sample = dataset[idx]
        frame_i = sample["frame_index"].item()
        timestamp = sample["timestamp"].item()

        rr.set_time_sequence("frame", frame_i)
        rr.set_time_seconds("time", timestamp)

        # --- images ---
        if "observation.images.top" in sample:
            rr.log("camera/top", rr.Image(to_hwc_uint8_numpy(sample["observation.images.top"])))

        if "observation.images.wrist" in sample:
            rr.log("camera/wrist", rr.Image(to_hwc_uint8_numpy(sample["observation.images.wrist"])))

        # --- joint state ---
        state = sample["observation.state"].numpy()
        for j in range(len(state)):
            rr.log(f"state/joint_{j}", rr.Scalar(float(state[j])))

        # --- actions: deltas separate from gripper ---
        action = sample["action"].numpy()
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
            local_i = idx - ep_start
            if local_i < len(ee_positions):
                pos = ee_positions[local_i]
                rr.log("trajectory/current", rr.Points3D(
                    [pos],
                    colors=[[255, 50, 50]],
                    radii=[0.005],
                ))
                # trail up to current frame
                rr.log("trajectory/trail", rr.LineStrips3D(
                    [ee_positions[:local_i + 1]],
                    colors=[[255, 100, 100]],
                    radii=[0.002],
                ))

        if (idx - ep_start + 1) % 100 == 0 or idx == ep_end - 1:
            print(f"  {idx - ep_start + 1}/{num_frames}")

    if save_path:
        rr.save(str(save_path))
        print(f"Done! Saved to {save_path}")
        print(f"Open with: rerun {save_path}")
    else:
        print("Done! Rerun viewer should be open.")


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
