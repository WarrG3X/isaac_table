from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import os
import sys
import time

import numpy as np
import yaml
from isaacsim.core.api import World
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.camera import Camera
import omni.ui as ui
from pxr import UsdGeom


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from scenes.registry import get_scene_builder


def build_camera_window(title, provider, width, height, pos_x, pos_y):
    window = ui.Window(title, width=width + 20, height=height + 40, visible=True)
    window.position_x = pos_x
    window.position_y = pos_y
    with window.frame:
        with ui.VStack():
            ui.ImageWithProvider(provider, width=width, height=height)
    return window


def update_camera_provider_from_rgba(provider, rgba):
    if rgba.ndim != 3 or rgba.shape[0] == 0 or rgba.shape[1] == 0:
        return
    if rgba.shape[2] == 3:
        alpha = np.full((rgba.shape[0], rgba.shape[1], 1), 255, dtype=np.uint8)
        rgba = np.concatenate([rgba, alpha], axis=2)
    provider.set_bytes_data(bytearray(rgba.tobytes()), [int(rgba.shape[1]), int(rgba.shape[0])])


def initialize_scene_objects(scene_info):
    robot = scene_info.robot
    robot.initialize()
    for key in ("clutter_objects",):
        for entry in scene_info.extras.get(key, []):
            rigid = entry.get("rigid")
            if rigid is not None:
                rigid.initialize()
    cube = scene_info.extras.get("cube")
    if cube is not None:
        cube.initialize()


def reset_scene_objects(scene_info):
    cube = scene_info.extras.get("cube")
    if cube is not None:
        cube.set_world_pose(
            position=scene_info.extras["initial_cube_position"],
            orientation=scene_info.extras["initial_cube_orientation"],
        )
        cube.set_linear_velocity(np.zeros(3, dtype=np.float32))
        cube.set_angular_velocity(np.zeros(3, dtype=np.float32))
    for entry in scene_info.extras.get("clutter_objects", []):
        entry["rigid"].set_world_pose(position=entry["initial_position"], orientation=entry["initial_orientation"])
        entry["rigid"].set_linear_velocity(np.zeros(3, dtype=np.float32))
        entry["rigid"].set_angular_velocity(np.zeros(3, dtype=np.float32))


parser = argparse.ArgumentParser()
parser.add_argument("--episode-dir", type=str, required=True)
parser.add_argument("--camera-panels", action="store_true")
parser.add_argument("--auto-start", action="store_true")
parser.add_argument("--mode", type=str, choices=["auto", "robot_only", "full_state"], default="auto")
args = parser.parse_args()

episode_dir = os.path.abspath(args.episode_dir)
scene_yaml_path = os.path.join(episode_dir, "scene.yaml")
data_path = os.path.join(episode_dir, "data.npz")
if not os.path.exists(scene_yaml_path):
    raise FileNotFoundError(f"Missing scene.yaml: {scene_yaml_path}")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Missing data.npz: {data_path}")

with open(scene_yaml_path, "r", encoding="utf-8") as f:
    scene_config = yaml.safe_load(f)
episode_data = np.load(data_path)
joint_positions = np.asarray(episode_data["joint_position"], dtype=np.float32)
timestamps = np.asarray(episode_data["timestamp"], dtype=np.float64)
object_positions = np.asarray(episode_data["object_position"], dtype=np.float32) if "object_position" in episode_data else None
object_orientations = np.asarray(episode_data["object_orientation_wxyz"], dtype=np.float32) if "object_orientation_wxyz" in episode_data else None
if joint_positions.ndim != 2 or joint_positions.shape[0] == 0:
    raise RuntimeError("Episode has no joint_position trajectory")

scene_name = scene_config["scene_name"]
scene_builder = get_scene_builder(scene_name)

world = World(stage_units_in_meters=1.0)
stage = world.stage
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
scene_info = scene_builder(world, stage, scene_config)

topdown_provider = ui.ByteImageProvider() if args.camera_panels else None
robot_provider = ui.ByteImageProvider() if args.camera_panels else None
topdown_window = None
robot_window = None
topdown_sensor_camera = None
robot_sensor_camera = None

world.reset()
initialize_scene_objects(scene_info)
reset_scene_objects(scene_info)

robot = scene_info.robot
articulation_controller = robot.get_articulation_controller()
topdown_sensor_camera = Camera(prim_path=scene_info.topdown_camera_prim_path, frequency=30, resolution=(640, 480))
robot_sensor_camera = Camera(prim_path=scene_info.robot_camera_prim_path, frequency=30, resolution=(640, 480))
topdown_sensor_camera.initialize()
robot_sensor_camera.initialize()

if args.camera_panels:
    topdown_window = build_camera_window("Playback Top Down", topdown_provider, 640, 480, 980, 40)
    robot_window = build_camera_window("Playback Wrist", robot_provider, 640, 480, 980, 560)

initial_targets = joint_positions[0]
articulation_controller.apply_action(ArticulationAction(joint_positions=initial_targets))
for _ in range(5):
    world.step(render=True)

frame_dt = None
if timestamps.shape[0] > 1:
    diffs = np.diff(timestamps)
    valid_diffs = diffs[diffs > 0.0]
    if valid_diffs.size > 0:
        frame_dt = float(np.median(valid_diffs))

print(f"[Playback] scene={scene_name} episode={episode_dir}")
print(f"[Playback] frames={joint_positions.shape[0]} dof={joint_positions.shape[1]}")
full_state_available = object_positions is not None and object_orientations is not None
playback_mode = args.mode
if playback_mode == "auto":
    playback_mode = "full_state" if full_state_available else "robot_only"
print(f"[Playback] mode={playback_mode}")
if frame_dt is not None:
    print(f"[Playback] nominal_dt={frame_dt:.4f}s")
if not args.auto_start:
    input("[Playback] Press Enter to start replay...")

try:
    while simulation_app.is_running():
        reset_scene_objects(scene_info)
        articulation_controller.apply_action(ArticulationAction(joint_positions=initial_targets))
        for _ in range(5):
            world.step(render=True)
        for idx, target in enumerate(joint_positions):
            if not simulation_app.is_running():
                break
            frame_start = time.perf_counter()
            articulation_controller.apply_action(ArticulationAction(joint_positions=target))
            if playback_mode == "full_state" and full_state_available:
                cube = scene_info.extras.get("cube")
                if cube is not None and object_positions.ndim == 2:
                    cube.set_world_pose(position=object_positions[idx], orientation=object_orientations[idx])
                    cube.set_linear_velocity(np.zeros(3, dtype=np.float32))
                    cube.set_angular_velocity(np.zeros(3, dtype=np.float32))
                clutter_entries = scene_info.extras.get("clutter_objects", [])
                if clutter_entries and object_positions.ndim == 3:
                    for obj_idx, entry in enumerate(clutter_entries):
                        entry["rigid"].set_world_pose(position=object_positions[idx, obj_idx], orientation=object_orientations[idx, obj_idx])
                        entry["rigid"].set_linear_velocity(np.zeros(3, dtype=np.float32))
                        entry["rigid"].set_angular_velocity(np.zeros(3, dtype=np.float32))
            world.step(render=True)
            if args.camera_panels:
                top_rgba = np.asarray(topdown_sensor_camera.get_rgba(device="cpu"), dtype=np.uint8)
                wrist_rgba = np.asarray(robot_sensor_camera.get_rgba(device="cpu"), dtype=np.uint8)
                if top_rgba.ndim == 3:
                    update_camera_provider_from_rgba(topdown_provider, top_rgba)
                if wrist_rgba.ndim == 3:
                    update_camera_provider_from_rgba(robot_provider, wrist_rgba)
            if frame_dt is not None:
                elapsed = time.perf_counter() - frame_start
                remaining = frame_dt - elapsed
                if remaining > 0.0:
                    time.sleep(remaining)
            if idx == 0 or (idx + 1) % 100 == 0 or idx == joint_positions.shape[0] - 1:
                print(f"[Playback] frame {idx + 1}/{joint_positions.shape[0]}")

        print("[Playback] replay complete")
        if not simulation_app.is_running():
            break
        input("[Playback] Press Enter to replay again (Ctrl+C to quit)...")
except KeyboardInterrupt:
    pass
finally:
    simulation_app.close()
