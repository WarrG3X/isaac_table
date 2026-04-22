from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import os
import sys
import threading
from datetime import datetime

import carb
import numpy as np
import omni.appwindow
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path
from omni.kit.viewport.utility import get_active_viewport
from pxr import Gf, Sdf, UsdGeom, UsdLux
from PIL import Image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scenes.common import (
    apply_static_collision,
    bind_material,
    build_tabletop_with_cutouts,
    build_tray,
    create_omnipbr_material,
    define_box,
    define_uv_plane,
    offset_prim_translate_z,
)
from scenes.clutter_pick import default_scene_config, get_floor_top_z


DEFAULT_TABLE_WIDTH = 1.90
DEFAULT_TABLE_DEPTH = 1.55
DEFAULT_TRAY_WIDTH = 0.60
DEFAULT_TRAY_DEPTH = 0.42
DEFAULT_TRAY_HEIGHT = 0.1275
DEFAULT_TRAY_WALL = 0.012
DEFAULT_VIEW_EYE = [-0.248822, 0.555326, 0.355019]
DEFAULT_VIEW_TARGET = [-0.248822, -0.233189, -0.259997]
REALSENSE_CAMERA_PRIM_PATH = "/World/ProbeRig/Realsense"
REALSENSE_RESOLUTION = (640, 480)
REALSENSE_FREQUENCY = 30
REALSENSE_FOCAL_LENGTH = 1.93
REALSENSE_HORIZONTAL_APERTURE = 3.728
REALSENSE_VERTICAL_APERTURE = 2.799
REALSENSE_CLIPPING_RANGE = (0.01, 100.0)
CAPTURE_DIR = os.path.join(REPO_ROOT, "debug_captures")


dump_requested = False
capture_requested = False


def stdin_loop():
    global dump_requested
    while True:
        try:
            input()
        except EOFError:
            return
        dump_requested = True


def save_camera_frame(camera, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    rgba = np.asarray(camera.get_rgba(device="cpu"))
    if rgba.size == 0:
        raise RuntimeError("Camera returned an empty RGBA frame")
    rgb = np.clip(rgba[..., :3], 0, 255).astype(np.uint8)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = os.path.join(output_dir, f"realsense_{timestamp}.png")
    Image.fromarray(rgb).save(output_path)
    return output_path


def sync_sensor_camera_to_viewport(stage, camera_prim_path):
    report = get_camera_report(stage)
    set_camera_view(
        eye=report["eye"].tolist(),
        target=report["target_guess"].tolist(),
        camera_prim_path=camera_prim_path,
    )
    return report


def on_keyboard_event(event, *args, **kwargs):
    global capture_requested
    if event.type == carb.input.KeyboardEventType.KEY_PRESS and event.input == carb.input.KeyboardInput.S:
        capture_requested = True
        return True
    return True


def get_camera_report(stage):
    viewport = get_active_viewport()
    camera_path = viewport.camera_path
    camera_prim = stage.GetPrimAtPath(camera_path)
    if not camera_prim.IsValid():
        raise RuntimeError(f"Active viewport camera prim is invalid: {camera_path}")

    xform = UsdGeom.Xformable(camera_prim)
    world_tf = xform.ComputeLocalToWorldTransform(0.0)
    eye = np.array(world_tf.ExtractTranslation(), dtype=np.float64)
    rot_only = np.array(world_tf.ExtractRotationMatrix(), dtype=np.float64)
    forward = rot_only @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
    up = rot_only @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
    target_guess = eye + forward
    return {
        "camera_path": camera_path,
        "eye": eye,
        "forward": forward,
        "up": up,
        "target_guess": target_guess,
    }


world = World(stage_units_in_meters=1.0)
stage = world.stage
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
for path in ["/World", "/World/Looks", "/World/Floor", "/World/Table", "/World/SourceTray", "/World/TargetTray", "/World/Lights"]:
    UsdGeom.Xform.Define(stage, path)

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    raise RuntimeError("Could not find Isaac Sim assets folder for Simple_Room")
add_reference_to_stage(
    usd_path=assets_root_path + "/Isaac/Environments/Simple_Room/simple_room.usd",
    prim_path="/World/Room",
)
room_table = stage.GetPrimAtPath("/World/Room/table_low_327")
if room_table.IsValid():
    room_table.SetActive(False)

bamboo_texture_path = os.path.join(REPO_ROOT, "materials", "nv_bamboo_desktop.jpg")
granite_texture_path = os.path.join(REPO_ROOT, "materials", "nv_granite_tile.jpg")
floor_material_base = create_omnipbr_material(stage, "/World/Looks/FloorBase", color=(0.44, 0.46, 0.48), roughness=0.72)
floor_material = create_omnipbr_material(stage, "/World/Looks/Floor", color=(0.85, 0.85, 0.85), roughness=0.62, texture_path=granite_texture_path if os.path.exists(granite_texture_path) else None)
wood_material = create_omnipbr_material(stage, "/World/Looks/Wood", color=(0.72, 0.66, 0.54), roughness=0.38, texture_path=bamboo_texture_path if os.path.exists(bamboo_texture_path) else None)
wood_base_material = create_omnipbr_material(stage, "/World/Looks/WoodBase", color=(0.49, 0.35, 0.20), roughness=0.48)
leg_material = create_omnipbr_material(stage, "/World/Looks/Legs", color=(0.19, 0.12, 0.08), roughness=0.5)
source_tray_material = create_omnipbr_material(stage, "/World/Looks/SourceTray", color=(0.16, 0.24, 0.44), roughness=0.60, metallic=0.06)
target_tray_material = create_omnipbr_material(stage, "/World/Looks/TargetTray", color=(0.46, 0.26, 0.16), roughness=0.62, metallic=0.04)

floor_prim = define_box(stage, "/World/Floor/Base", size=(7.0, 7.0, 0.02), translate=(0.0, 0.0, -0.01))
bind_material(floor_prim, floor_material_base)
floor_surface = define_uv_plane(stage, "/World/Floor/Surface", size=(6.92, 6.92), translate=(0.0, 0.0, 0.001), uv_scale=(6.0, 6.0))
bind_material(floor_surface, floor_material)

scene_config = default_scene_config(
    repo_root=REPO_ROOT,
    robot_usd=os.path.join(REPO_ROOT, "dummy.usd"),
    ee_frame="link6",
    robot_x=0.0,
    robot_y=-0.05,
    robot_yaw=0.0,
    table_width=DEFAULT_TABLE_WIDTH,
    table_depth=DEFAULT_TABLE_DEPTH,
    tray_width=DEFAULT_TRAY_WIDTH,
    tray_depth=DEFAULT_TRAY_DEPTH,
    tray_height=DEFAULT_TRAY_HEIGHT,
    tray_wall=DEFAULT_TRAY_WALL,
    num_objects=0,
    seed=42,
    max_volume=0.002,
)
table_cfg = scene_config["table"]
tray_cfg = scene_config["trays"]
table_size = (float(table_cfg["width"]), float(table_cfg["depth"]), float(table_cfg["thickness"]))
table_top_z = float(table_cfg["top_z"])
leg_height = float(table_cfg["leg_height"])
leg_dx = table_size[0] * 0.5 - 0.08
leg_dy = table_size[1] * 0.5 - 0.08
leg_center_z = leg_height * 0.5
floor_top_z = get_floor_top_z(stage, scene_config["environment"]["floor_support_prim"])
if floor_top_z is not None:
    leg_center_z = floor_top_z + leg_height * 0.5
    table_top_z = floor_top_z + leg_height + table_size[2] * 0.5
table_surface_z = table_top_z + table_size[2] * 0.5

table_collision_paths = [str(floor_prim.GetPath())]
for name, position in {
    "LegFL": (leg_dx, leg_dy, leg_center_z),
    "LegFR": (leg_dx, -leg_dy, leg_center_z),
    "LegBL": (-leg_dx, leg_dy, leg_center_z),
    "LegBR": (-leg_dx, -leg_dy, leg_center_z),
}.items():
    leg_prim = define_box(stage, f"/World/Table/{name}", size=(0.06, 0.06, leg_height), translate=position)
    bind_material(leg_prim, leg_material)
    table_collision_paths.append(str(leg_prim.GetPath()))

source_tray_center = tuple(float(v) for v in tray_cfg["source_center"])
target_tray_center = tuple(float(v) for v in tray_cfg["target_center"])
source_tray_paths = build_tray(stage, "/World/SourceTray", center_xy=source_tray_center, surface_z=table_surface_z, outer_size=(float(tray_cfg["width"]), float(tray_cfg["depth"])), height=float(tray_cfg["height"]), wall=float(tray_cfg["wall"]), material=source_tray_material)
target_tray_paths = build_tray(stage, "/World/TargetTray", center_xy=target_tray_center, surface_z=table_surface_z, outer_size=(float(tray_cfg["width"]), float(tray_cfg["depth"])), height=float(tray_cfg["height"]), wall=float(tray_cfg["wall"]), material=target_tray_material)
for path in source_tray_paths + target_tray_paths:
    offset_prim_translate_z(stage, path, -float(tray_cfg["height"]))
tabletop_paths = build_tabletop_with_cutouts(
    stage,
    "/World/Table",
    table_size=table_size,
    table_top_z=table_top_z,
    hole_specs=[
        (source_tray_center[0], source_tray_center[1], float(tray_cfg["width"]), float(tray_cfg["depth"])),
        (target_tray_center[0], target_tray_center[1], float(tray_cfg["width"]), float(tray_cfg["depth"])),
    ],
    base_material=wood_base_material,
    surface_material=wood_material,
)
table_collision_paths.extend(tabletop_paths)
apply_static_collision(table_collision_paths + source_tray_paths + target_tray_paths)

dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Lights/Dome"))
dome_light.CreateIntensityAttr(950.0)
dome_light.CreateColorAttr(Gf.Vec3f(0.92, 0.95, 1.0))
sun_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Lights/Sun"))
sun_light.CreateIntensityAttr(3200.0)
sun_light.CreateAngleAttr(0.6)
sun_light.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
UsdGeom.Xformable(sun_light.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(315.0, 0.0, 35.0))
table_fill = UsdLux.RectLight.Define(stage, Sdf.Path("/World/Lights/TableFill"))
table_fill.CreateIntensityAttr(2000.0)
table_fill.CreateWidthAttr(2.4)
table_fill.CreateHeightAttr(1.2)
table_fill.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))
UsdGeom.Xformable(table_fill.GetPrim()).AddTranslateOp().Set(Gf.Vec3f(0.3, 0.05, 1.95))
UsdGeom.Xformable(table_fill.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-90.0, 0.0, 0.0))

set_camera_view(
    eye=DEFAULT_VIEW_EYE,
    target=DEFAULT_VIEW_TARGET,
    camera_prim_path="/OmniverseKit_Persp",
)

for prim_path in ["/World/Floor/Base", "/World/Floor/Surface", "/World/Lights/Dome", "/World/Lights/Sun"]:
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        prim.SetActive(False)

world.reset()
UsdGeom.Xform.Define(stage, "/World/ProbeRig")
realsense_camera = world.scene.add(
    Camera(
        prim_path=REALSENSE_CAMERA_PRIM_PATH,
        name="probe_realsense",
        frequency=REALSENSE_FREQUENCY,
        resolution=REALSENSE_RESOLUTION,
    )
)
set_camera_view(
    eye=DEFAULT_VIEW_EYE,
    target=DEFAULT_VIEW_TARGET,
    camera_prim_path=REALSENSE_CAMERA_PRIM_PATH,
)
world.reset()
realsense_camera.initialize()
realsense_camera.set_focal_length(REALSENSE_FOCAL_LENGTH)
realsense_camera.set_horizontal_aperture(REALSENSE_HORIZONTAL_APERTURE)
realsense_camera.set_vertical_aperture(REALSENSE_VERTICAL_APERTURE)
realsense_camera.set_clipping_range(*REALSENSE_CLIPPING_RANGE)
realsense_camera.resume()

appwindow = omni.appwindow.get_default_app_window()
input_interface = carb.input.acquire_input_interface()
keyboard = appwindow.get_keyboard()
keyboard_subscription = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)
threading.Thread(target=stdin_loop, daemon=True).start()
print("[TempViewCameraProbe] Adjust the main viewer camera, then press Enter in this terminal to dump its pose.")
print("[TempViewCameraProbe] Press 's' in the app window to save a RealSense RGB frame.")

while simulation_app.is_running():
    world.step(render=True)
    if capture_requested:
        capture_requested = False
        try:
            report = sync_sensor_camera_to_viewport(stage, REALSENSE_CAMERA_PRIM_PATH)
            world.step(render=True)
            output_path = save_camera_frame(realsense_camera, CAPTURE_DIR)
            print(
                "[TempViewCameraProbe] synced_realsense_to_view "
                f"eye=[{report['eye'][0]:.6f}, {report['eye'][1]:.6f}, {report['eye'][2]:.6f}] "
                f"target=[{report['target_guess'][0]:.6f}, {report['target_guess'][1]:.6f}, {report['target_guess'][2]:.6f}]"
            )
            print(
                "[TempViewCameraProbe] realsense_config "
                f"resolution={REALSENSE_RESOLUTION} "
                f"focal_length={REALSENSE_FOCAL_LENGTH:.6f} "
                f"horizontal_aperture={REALSENSE_HORIZONTAL_APERTURE:.6f} "
                f"vertical_aperture={REALSENSE_VERTICAL_APERTURE:.6f} "
                f"clipping_range=({REALSENSE_CLIPPING_RANGE[0]:.6f}, {REALSENSE_CLIPPING_RANGE[1]:.6f})"
            )
            print(f"[TempViewCameraProbe] saved_realsense_frame={output_path}")
        except Exception as exc:
            print(f"[TempViewCameraProbe] failed_to_save_realsense_frame: {exc}")
    if dump_requested:
        dump_requested = False
        report = get_camera_report(stage)
        eye = report["eye"]
        forward = report["forward"]
        up = report["up"]
        target_guess = report["target_guess"]
        print(f"[TempViewCameraProbe] camera_path={report['camera_path']}")
        print(f"[TempViewCameraProbe] eye=[{eye[0]:.6f}, {eye[1]:.6f}, {eye[2]:.6f}]")
        print(f"[TempViewCameraProbe] forward=[{forward[0]:.6f}, {forward[1]:.6f}, {forward[2]:.6f}]")
        print(f"[TempViewCameraProbe] up=[{up[0]:.6f}, {up[1]:.6f}, {up[2]:.6f}]")
        print(f"[TempViewCameraProbe] target_guess=[{target_guess[0]:.6f}, {target_guess[1]:.6f}, {target_guess[2]:.6f}]")

input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_subscription)
simulation_app.close()
