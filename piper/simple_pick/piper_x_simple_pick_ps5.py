from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import concurrent.futures
import copy
import json
import os
import sys
import time

import numpy as np
import omni.kit.commands
import omni.ui as ui
import pygame
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import GeometryPrim, SingleArticulation
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats, rot_matrices_to_quats
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from PIL import Image
from isaacsim.sensors.camera import Camera
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt
import yaml


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
ISAAC_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
DEFAULT_PIPER_USD = os.path.join(ISAAC_ROOT, "piper_isaac_sim", "USD", "piper_x_v1.usd")
DEFAULT_LULA_YAML = os.path.join(REPO_ROOT, "desc", "piper_x_robot_description.yaml")
DEFAULT_LULA_URDF = os.path.join(
    ISAAC_ROOT, "piper_isaac_sim", "piper_x_description", "urdf", "piper_x_description_d435.urdf"
)
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data", "simple_pick_raw")
from scenes.simple_pick import default_scene_config


def define_box(stage, prim_path, size, translate):
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(*translate))
    cube = UsdGeom.Cube.Define(stage, f"{prim_path}/Geom")
    cube.CreateSizeAttr(1.0)
    UsdGeom.Xformable(cube.GetPrim()).AddScaleOp().Set(Gf.Vec3f(*size))
    return cube.GetPrim()


def define_uv_plane(stage, prim_path, size, translate, uv_scale=(1.0, 1.0)):
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(*translate))
    root.AddScaleOp().Set(Gf.Vec3f(size[0], size[1], 1.0))
    mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/Geom")
    mesh.CreatePointsAttr([Gf.Vec3f(-0.5, -0.5, 0.0), Gf.Vec3f(0.5, -0.5, 0.0), Gf.Vec3f(0.5, 0.5, 0.0), Gf.Vec3f(-0.5, 0.5, 0.0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, 1.0)] * 4)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateExtentAttr([Gf.Vec3f(-0.5, -0.5, 0.0), Gf.Vec3f(0.5, 0.5, 0.0)])
    primvars_api = UsdGeom.PrimvarsAPI(mesh)
    st = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray([Gf.Vec2f(0.0, 0.0), Gf.Vec2f(uv_scale[0], 0.0), Gf.Vec2f(uv_scale[0], uv_scale[1]), Gf.Vec2f(0.0, uv_scale[1])]))
    return mesh.GetPrim()


def create_omnipbr_material(stage, material_path, color=None, roughness=0.5, metallic=0.0, texture_path=None):
    created = []
    omni.kit.commands.execute("CreateAndBindMdlMaterialFromLibrary", mdl_name="OmniPBR.mdl", mtl_name="OmniPBR", mtl_created_list=created)
    material_prim = stage.GetPrimAtPath(created[0])
    if created[0] != material_path:
        omni.kit.commands.execute("MovePrim", path_from=created[0], path_to=material_path)
        material_prim = stage.GetPrimAtPath(material_path)
    if color is not None:
        omni.usd.create_material_input(material_prim, "diffuse_color_constant", Gf.Vec3f(*color), Sdf.ValueTypeNames.Color3f)
    omni.usd.create_material_input(material_prim, "reflection_roughness_constant", float(roughness), Sdf.ValueTypeNames.Float)
    omni.usd.create_material_input(material_prim, "metallic_constant", float(metallic), Sdf.ValueTypeNames.Float)
    if texture_path:
        omni.usd.create_material_input(material_prim, "diffuse_texture", texture_path, Sdf.ValueTypeNames.Asset)
    return UsdShade.Material(material_prim)


def bind_material(prim, material):
    UsdShade.MaterialBindingAPI(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)


def apply_static_collision(prim_paths):
    for prim_path in prim_paths:
        GeometryPrim(prim_path).apply_collision_apis()


def deactivate_embedded_environment(stage, root_prim_path):
    candidate_tokens = {"environment", "groundplane", "ground_plane", "ground", "floor", "physicsscene", "camerasettings", "viewport_l", "viewport_r", "data"}
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if prim_path.startswith(root_prim_path + "/") and prim.GetName().lower() in candidate_tokens:
            prim.SetActive(False)


def set_translate(prim, translate):
    tx, ty, tz = float(translate[0]), float(translate[1]), float(translate[2])
    xformable = UsdGeom.Xformable(prim)
    translate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if translate_ops:
        translate_ops[0].Set(Gf.Vec3d(tx, ty, tz))
    else:
        xformable.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))


def create_robot_camera(stage, prim_path):
    mount = UsdGeom.Xform.Define(stage, prim_path)
    mount_xform = UsdGeom.Xformable(mount.GetPrim())
    mount_xform.AddTranslateOp().Set(Gf.Vec3d(0.03, 0.0, 0.0))
    mount_xform.AddRotateXYZOp().Set(Gf.Vec3f(-90.0, 0.0, 90.0))

    camera = UsdGeom.Camera.Define(stage, f"{prim_path}/Sensor")
    camera_xform = UsdGeom.Xformable(camera.GetPrim())
    camera_xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 180.0))
    camera.CreateFocalLengthAttr(2.79)
    camera.CreateHorizontalApertureAttr(3.84)
    camera.CreateVerticalApertureAttr(2.88)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.02, 8.0))
    return camera.GetPrim()


def create_topdown_camera(stage, prim_path, center, coverage_xy):
    camera = UsdGeom.Camera.Define(stage, prim_path)
    xform = UsdGeom.Xformable(camera.GetPrim())
    focal_length = 18.0
    horizontal_aperture = 20.955
    vertical_aperture = 15.2908
    required_height_x = float(coverage_xy[0]) * focal_length / horizontal_aperture
    required_height_y = float(coverage_xy[1]) * focal_length / vertical_aperture
    camera_height = 2.0 * (max(required_height_x, required_height_y) + 0.10)
    xform.AddTranslateOp().Set(Gf.Vec3d(float(center[0]), float(center[1]), float(center[2] + camera_height)))
    camera.CreateFocalLengthAttr(focal_length)
    camera.CreateHorizontalApertureAttr(horizontal_aperture)
    camera.CreateVerticalApertureAttr(vertical_aperture)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.02, 12.0))
    return camera.GetPrim()


def build_status_window(initial_position):
    text_model = ui.SimpleStringModel("")
    save_model = ui.SimpleStringModel("idle")
    window = ui.Window("Simple Pick Teleop", width=460, height=140, visible=True)
    window.position_x = 40
    window.position_y = 60
    with window.frame:
        with ui.VStack(spacing=8, height=0):
            ui.Label("PS5 teleop for the simple-pick scene.")
            ui.Label("Cross: gripper | Circle: reset | Right stick: roll/pitch | L1/R1: yaw")
            with ui.HStack(height=24):
                ui.Label("Target", width=50)
                ui.StringField(model=text_model, read_only=True)
            with ui.HStack(height=24):
                ui.Label("Saves", width=50)
                ui.StringField(model=save_model, read_only=True)
    text_model.set_value(f"({initial_position[0]:.3f}, {initial_position[1]:.3f}, {initial_position[2]:.3f})")
    return window, text_model, save_model


def build_camera_window(title, provider, width, height, pos_x, pos_y):
    window = ui.Window(title, width=width + 20, height=height + 40, visible=True)
    window.position_x = pos_x
    window.position_y = pos_y
    with window.frame:
        with ui.VStack():
            ui.ImageWithProvider(provider, width=width, height=height)
    return window


def update_camera_provider(camera, provider):
    rgba = np.asarray(camera.get_rgba(device="cpu"), dtype=np.uint8)
    update_camera_provider_from_rgba(provider, rgba)


def update_camera_provider_from_rgba(provider, rgba):
    if rgba.ndim != 3 or rgba.shape[0] == 0 or rgba.shape[1] == 0:
        return
    if rgba.shape[2] == 3:
        alpha = np.full((rgba.shape[0], rgba.shape[1], 1), 255, dtype=np.uint8)
        rgba = np.concatenate([rgba, alpha], axis=2)
    provider.set_bytes_data(bytearray(rgba.tobytes()), [int(rgba.shape[1]), int(rgba.shape[0])])


def create_episode_buffer():
    return {
        "started_at": time.time(),
        "steps": [],
        "top_rgb": [],
        "wrist_rgb": [],
        "success": False,
    }


def save_rgb_frame(rgb, path, image_format, jpeg_quality):
    image = Image.fromarray(rgb)
    if image_format == "jpg":
        image.save(path, format="JPEG", quality=int(jpeg_quality), optimize=True)
    else:
        image.save(path, format="PNG")


def save_episode(output_dir, episode, task_name, source_pad_center, target_pad_center, cube_start, ee_frame, image_format, jpeg_quality, scene_config):
    os.makedirs(output_dir, exist_ok=True)
    episode_id = time.strftime("episode_%Y%m%d_%H%M%S")
    episode_dir = os.path.join(output_dir, episode_id)
    suffix = 0
    while os.path.exists(episode_dir):
        suffix += 1
        episode_dir = os.path.join(output_dir, f"{episode_id}_{suffix:02d}")
    os.makedirs(episode_dir, exist_ok=False)
    top_dir = os.path.join(episode_dir, "images", "top")
    wrist_dir = os.path.join(episode_dir, "images", "wrist")
    os.makedirs(top_dir, exist_ok=True)
    os.makedirs(wrist_dir, exist_ok=True)
    image_ext = "jpg" if image_format == "jpg" else "png"

    for idx, rgb in enumerate(episode["top_rgb"]):
        save_rgb_frame(rgb, os.path.join(top_dir, f"{idx:06d}.{image_ext}"), image_format=image_format, jpeg_quality=jpeg_quality)
    for idx, rgb in enumerate(episode["wrist_rgb"]):
        save_rgb_frame(rgb, os.path.join(wrist_dir, f"{idx:06d}.{image_ext}"), image_format=image_format, jpeg_quality=jpeg_quality)

    np.savez_compressed(
        os.path.join(episode_dir, "data.npz"),
        step_index=np.asarray([step["step_index"] for step in episode["steps"]], dtype=np.int32),
        timestamp=np.asarray([step["timestamp"] for step in episode["steps"]], dtype=np.float64),
        joint_position=np.asarray([step["joint_position"] for step in episode["steps"]], dtype=np.float32),
        ee_position=np.asarray([step["ee_position"] for step in episode["steps"]], dtype=np.float32),
        ee_delta=np.asarray([step["ee_delta"] for step in episode["steps"]], dtype=np.float32),
        gripper_action=np.asarray([step["gripper_action"] for step in episode["steps"]], dtype=np.int8),
        gripper_open=np.asarray([step["gripper_open"] for step in episode["steps"]], dtype=np.int8),
    )

    metadata = {
        "format": "simple_pick_raw_v1",
        "task": task_name,
        "controller": "ps5",
        "ee_frame": ee_frame,
        "scene_name": scene_config["scene_name"],
        "success": bool(episode["success"]),
        "num_steps": len(episode["steps"]),
        "source_pad_center": list(source_pad_center),
        "target_pad_center": list(target_pad_center),
        "cube_start_position": list(cube_start),
        "images": {
            "top": "images/top",
            "wrist": "images/wrist",
        },
        "image_format": image_ext,
        "data_file": "data.npz",
        "scene_file": "scene.yaml",
    }
    with open(os.path.join(episode_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(episode_dir, "scene.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(scene_config, f, sort_keys=False)
    return episode_dir


def quat_multiply_wxyz(q1, q2):
    w1, x1, y1, z1 = [float(v) for v in q1]
    w2, x2, y2, z2 = [float(v) for v in q2]
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def normalize_quat_wxyz(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def quat_wxyz_to_rot_matrix(q):
    w, x, y, z = [float(v) for v in normalize_quat_wxyz(q)]
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def ray_aabb_intersection(origin, direction, bounds_min, bounds_max):
    tmin = -np.inf
    tmax = np.inf
    for axis in range(3):
        if abs(direction[axis]) < 1e-9:
            if origin[axis] < bounds_min[axis] or origin[axis] > bounds_max[axis]:
                return None
            continue
        inv_d = 1.0 / direction[axis]
        t1 = (bounds_min[axis] - origin[axis]) * inv_d
        t2 = (bounds_max[axis] - origin[axis]) * inv_d
        t_near = min(t1, t2)
        t_far = max(t1, t2)
        tmin = max(tmin, t_near)
        tmax = min(tmax, t_far)
        if tmax < tmin:
            return None
    if tmax < 0.0:
        return None
    return tmin if tmin >= 0.0 else tmax


def init_controller(index):
    pygame.init()
    pygame.joystick.init()
    count = pygame.joystick.get_count()
    print(f"[PS5] pygame joysticks={count}")
    if count == 0:
        raise RuntimeError("No controller detected by pygame")
    if index >= count:
        raise RuntimeError(f"Controller index {index} out of range")
    joystick = pygame.joystick.Joystick(index)
    joystick.init()
    print(
        f"[PS5] name={joystick.get_name()} "
        f"axes={joystick.get_numaxes()} buttons={joystick.get_numbuttons()} hats={joystick.get_numhats()}"
    )
    return joystick


parser = argparse.ArgumentParser()
parser.add_argument("--controller", type=int, default=0)
parser.add_argument("--translation-scale", type=float, default=0.0018)
parser.add_argument("--rotation-scale", type=float, default=0.0007)
parser.add_argument("--yaw-scale", type=float, default=0.012)
parser.add_argument("--stick-deadband", type=float, default=0.10)
parser.add_argument("--trigger-deadband", type=float, default=0.05)
parser.add_argument("--robot-usd", type=str, default=DEFAULT_PIPER_USD)
parser.add_argument("--lula-yaml", type=str, default=DEFAULT_LULA_YAML)
parser.add_argument("--lula-urdf", type=str, default=DEFAULT_LULA_URDF)
parser.add_argument("--robot-x", type=float, default=0.0)
parser.add_argument("--robot-y", type=float, default=-0.18)
parser.add_argument("--robot-yaw", type=float, default=90.0)
parser.add_argument("--ee-frame", type=str, default="Link6")
parser.add_argument("--table-width", type=float, default=1.40)
parser.add_argument("--table-depth", type=float, default=1.55)
parser.add_argument("--cube-size", type=float, default=0.05)
parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
parser.add_argument("--camera-panels", action="store_true")
parser.add_argument("--image-format", type=str, choices=["jpg", "png"], default="png")
parser.add_argument("--jpeg-quality", type=int, default=85)
args, _ = parser.parse_known_args()
scene_config = default_scene_config(
    repo_root=REPO_ROOT,
    robot_usd=args.robot_usd,
    ee_frame=args.ee_frame,
    robot_x=args.robot_x,
    robot_y=args.robot_y,
    robot_yaw=args.robot_yaw,
    table_width=args.table_width,
    table_depth=args.table_depth,
    cube_size=args.cube_size,
)

world = World(stage_units_in_meters=1.0)
stage = world.stage
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.Xform.Define(stage, "/World")
UsdGeom.Xform.Define(stage, "/World/Looks")
UsdGeom.Xform.Define(stage, "/World/Floor")
UsdGeom.Xform.Define(stage, "/World/Table")
UsdGeom.Xform.Define(stage, "/World/SourcePad")
UsdGeom.Xform.Define(stage, "/World/TargetPad")
UsdGeom.Xform.Define(stage, "/World/Lights")
UsdGeom.Xform.Define(stage, "/World/RobotMount")
UsdGeom.Xform.Define(stage, "/World/Debug")

bamboo_texture_path = os.path.join(REPO_ROOT, "materials", "nv_bamboo_desktop.jpg")
granite_texture_path = os.path.join(REPO_ROOT, "materials", "nv_granite_tile.jpg")

floor_material_base = create_omnipbr_material(stage, "/World/Looks/FloorBase", color=(0.44, 0.46, 0.48), roughness=0.72)
floor_material = create_omnipbr_material(stage, "/World/Looks/Floor", color=(0.85, 0.85, 0.85), roughness=0.62, texture_path=granite_texture_path if os.path.exists(granite_texture_path) else None)
wood_material = create_omnipbr_material(stage, "/World/Looks/Wood", color=(0.72, 0.66, 0.54), roughness=0.38, texture_path=bamboo_texture_path if os.path.exists(bamboo_texture_path) else None)
wood_base_material = create_omnipbr_material(stage, "/World/Looks/WoodBase", color=(0.49, 0.35, 0.20), roughness=0.48)
leg_material = create_omnipbr_material(stage, "/World/Looks/Legs", color=(0.19, 0.12, 0.08), roughness=0.5)
source_pad_material = create_omnipbr_material(stage, "/World/Looks/SourcePad", color=(0.78, 0.16, 0.16), roughness=0.42)
target_pad_material = create_omnipbr_material(stage, "/World/Looks/TargetPad", color=(0.16, 0.28, 0.82), roughness=0.42)
target_material = create_omnipbr_material(stage, "/World/Looks/Target", color=(0.82, 0.16, 0.12), roughness=0.25)

floor_base_prim = define_box(stage, "/World/Floor/Base", size=(7.0, 7.0, 0.02), translate=(0.0, 0.0, -0.01))
bind_material(floor_base_prim, floor_material_base)
floor_surface_prim = define_uv_plane(stage, "/World/Floor/Surface", size=(6.92, 6.92), translate=(0.0, 0.0, 0.001), uv_scale=(6.0, 6.0))
bind_material(floor_surface_prim, floor_material)

table_size = (args.table_width, args.table_depth, 0.06)
table_top_z = 0.75
table_surface_z = table_top_z + table_size[2] * 0.5
top_prim = define_box(stage, "/World/Table/Top", size=table_size, translate=(0.0, 0.0, table_top_z))
bind_material(top_prim, wood_base_material)
table_surface_prim = define_uv_plane(stage, "/World/Table/TopSurface", size=(table_size[0] - 0.04, table_size[1] - 0.04), translate=(0.0, 0.0, table_surface_z + 0.001))
bind_material(table_surface_prim, wood_material)

leg_height = 0.72
leg_dx = table_size[0] * 0.5 - 0.08
leg_dy = table_size[1] * 0.5 - 0.08
table_leg_paths = []
for name, position in {
    "LegFL": (leg_dx, leg_dy, leg_height * 0.5),
    "LegFR": (leg_dx, -leg_dy, leg_height * 0.5),
    "LegBL": (-leg_dx, leg_dy, leg_height * 0.5),
    "LegBR": (-leg_dx, -leg_dy, leg_height * 0.5),
}.items():
    leg_prim = define_box(stage, f"/World/Table/{name}", size=(0.06, 0.06, leg_height), translate=position)
    bind_material(leg_prim, leg_material)
    table_leg_paths.append(str(leg_prim.GetPath()))

pad_size = (0.18, 0.18)
pad_z = table_surface_z + 0.002
pad_y = 0.22
source_pad_center = (-0.16, pad_y, pad_z)
target_pad_center = (0.16, pad_y, pad_z)
task_center = (0.5 * (source_pad_center[0] + target_pad_center[0]), 0.5 * (source_pad_center[1] + target_pad_center[1]), table_surface_z)
task_coverage_xy = (abs(target_pad_center[0] - source_pad_center[0]) + pad_size[0] + 0.18, pad_size[1] + 0.22)
source_pad = define_uv_plane(stage, "/World/SourcePad/Plane", size=pad_size, translate=source_pad_center)
bind_material(source_pad, source_pad_material)
target_pad = define_uv_plane(stage, "/World/TargetPad/Plane", size=pad_size, translate=target_pad_center)
bind_material(target_pad, target_pad_material)

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

apply_static_collision([str(floor_base_prim.GetPath()), str(top_prim.GetPath()), *table_leg_paths])

mount_prim = UsdGeom.Xform.Define(stage, "/World/RobotMount/PiperX")
mount_xform = UsdGeom.Xformable(mount_prim.GetPrim())
mount_xform.AddTranslateOp().Set(Gf.Vec3d(args.robot_x, args.robot_y, table_surface_z))
mount_xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, args.robot_yaw))
add_reference_to_stage(usd_path=os.path.abspath(args.robot_usd), prim_path="/World/RobotMount/PiperX/Robot")
deactivate_embedded_environment(stage, "/World/RobotMount/PiperX/Robot")
robot = SingleArticulation(prim_path="/World/RobotMount/PiperX/Robot", name="piper_x")
robot_camera_prim = create_robot_camera(stage, "/World/RobotMount/PiperX/Robot/piper_x_camera/camera_link/DebugCamera")
topdown_camera_prim = create_topdown_camera(stage, "/World/Debug/TopDownCamera", center=task_center, coverage_xy=task_coverage_xy)
topdown_sensor_camera = None
robot_sensor_camera = None
topdown_provider = ui.ByteImageProvider() if args.camera_panels else None
robot_provider = ui.ByteImageProvider() if args.camera_panels else None
topdown_window = None
robot_window = None

cube_center = np.array([source_pad_center[0], source_pad_center[1], table_surface_z + args.cube_size * 0.5 + 0.002], dtype=np.float32)
cube = DynamicCuboid(
    prim_path="/World/ObjectCube",
    name="simple_pick_cube",
    position=cube_center,
    scale=np.array([args.cube_size, args.cube_size, args.cube_size], dtype=np.float32),
    color=np.array([0.18, 0.76, 0.28], dtype=np.float32),
)
world.scene.add(cube)
initial_cube_position = cube_center.copy()
initial_cube_orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

target_marker = UsdGeom.Sphere.Define(stage, "/World/Debug/TeleopTarget")
target_marker.CreateRadiusAttr(0.018)
bind_material(target_marker.GetPrim(), target_material)
pointer_marker = UsdGeom.Sphere.Define(stage, "/World/Debug/ToolProjection")
pointer_marker.CreateRadiusAttr(0.014)
bind_material(pointer_marker.GetPrim(), target_pad_material)

set_camera_view(eye=[0.0, 1.85, 1.55], target=[0.0, 0.02, 0.82], camera_prim_path="/OmniverseKit_Persp")
if args.camera_panels:
    topdown_window = build_camera_window("Simple Pick Top Down", topdown_provider, 640, 480, 980, 40)
    robot_window = build_camera_window("Simple Pick Wrist", robot_provider, 640, 480, 980, 560)

world.reset()
robot.initialize()
robot_sensor_camera = Camera(prim_path=str(robot_camera_prim.GetPath()), frequency=30, resolution=(640, 480))
topdown_sensor_camera = Camera(prim_path=str(topdown_camera_prim.GetPath()), frequency=30, resolution=(640, 480))
robot_sensor_camera.initialize()
topdown_sensor_camera.initialize()
articulation_controller = robot.get_articulation_controller()
joint_names = list(robot.dof_names)
joint_name_to_index = {name: idx for idx, name in enumerate(joint_names)}
dof_props = robot.dof_properties
joint7_idx = joint_name_to_index["joint7"]
joint8_idx = joint_name_to_index["joint8"]
gripper_open_target = np.array([float(dof_props["upper"][joint7_idx]), float(dof_props["lower"][joint8_idx])], dtype=np.float32)
gripper_closed_target = np.array([float(dof_props["lower"][joint7_idx]), float(dof_props["upper"][joint8_idx])], dtype=np.float32)
lula_solver = LulaKinematicsSolver(robot_description_path=os.path.abspath(args.lula_yaml), urdf_path=os.path.abspath(args.lula_urdf))
robot_base_position = np.array([args.robot_x, args.robot_y, table_surface_z], dtype=np.float64)
robot_base_orientation = euler_angles_to_quats(np.array([0.0, 0.0, args.robot_yaw]), degrees=True)
lula_solver.set_robot_base_pose(robot_base_position, robot_base_orientation)
art_kinematics = ArticulationKinematicsSolver(robot, lula_solver, args.ee_frame)
print(f"[SimplePick] source pad center: {source_pad_center}")
print(f"[SimplePick] target pad center: {target_pad_center}")
print(f"[SimplePick] cube start: {cube_center.tolist()}")
print(f"[SimplePick] Lula active joints: {lula_solver.get_joint_names()}")
print(f"[SimplePick] camera panels: {'enabled' if args.camera_panels else 'disabled'}")
if args.camera_panels:
    print(f"[SimplePick] top-down camera panel: {topdown_window.title}")
    print(f"[SimplePick] wrist camera panel: {robot_window.title}")

stiffnesses = np.asarray(dof_props["stiffness"], dtype=np.float32).copy()
dampings = np.asarray(dof_props["damping"], dtype=np.float32).copy()
max_efforts = np.asarray(dof_props["maxEffort"], dtype=np.float32).copy()
for gripper_joint in ("joint7", "joint8"):
    idx = joint_name_to_index[gripper_joint]
    stiffnesses[idx] = max(stiffnesses[idx], 4000.0)
    dampings[idx] = max(dampings[idx], 120.0)
    max_efforts[idx] = max(max_efforts[idx], 1000.0)
robot._articulation_view.set_gains(kps=stiffnesses.reshape(1, -1), kds=dampings.reshape(1, -1))
robot._articulation_view.set_max_efforts(max_efforts.reshape(1, -1))

ee_position, ee_rotation = art_kinematics.compute_end_effector_pose()
target_position = np.array(ee_position, dtype=np.float32)
target_orientation = rot_matrices_to_quats(np.asarray(ee_rotation))
downward_pitch_quat = euler_angles_to_quats(np.deg2rad(np.array([-90.0, 0.0, 0.0], dtype=np.float64)), degrees=False)
target_orientation = normalize_quat_wxyz(quat_multiply_wxyz(downward_pitch_quat, target_orientation))
default_target_rotation = quat_wxyz_to_rot_matrix(target_orientation)
candidate_axes = {
    "+X": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "-X": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "+Y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "-Y": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "+Z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    "-Z": np.array([0.0, 0.0, -1.0], dtype=np.float64),
}
tool_axis_name, tool_local_axis = min(
    candidate_axes.items(),
    key=lambda item: float((default_target_rotation @ item[1])[2]),
)
initial_target_position = target_position.copy()
initial_target_orientation = np.asarray(target_orientation, dtype=np.float64).copy()
last_valid_targets = np.asarray(robot.get_joint_positions(), dtype=np.float32).copy()
set_translate(target_marker.GetPrim(), target_position)
set_translate(pointer_marker.GetPrim(), np.array([0.0, 0.0, -10.0], dtype=np.float32))
status_window, status_model, save_status_model = build_status_window(target_position)
print(f"[SimplePick] tool projection axis: {tool_axis_name}")

target_half_extent_x = pad_size[0] * 0.5
target_half_extent_y = pad_size[1] * 0.5
joystick = init_controller(args.controller)
last_buttons = []
gripper_open = True
success_latched = False
recording = False
episode = None
episode_step_index = 0
save_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="episode_save")
save_futures = []
last_pending_save_count = 0
print("[PS5] controls:")
print("[PS5] left stick -> X/Y")
print("[PS5] triggers -> Z down/up")
print("[PS5] right stick -> roll/pitch")
print("[PS5] L1/R1 -> yaw -/+")
print("[PS5] Cross -> gripper toggle | Circle -> reset/discard | Square -> start/stop recording")
print(f"[PS5] recording image format: {args.image_format}")

try:
    while simulation_app.is_running():
        still_pending = []
        for future in save_futures:
            if future.done():
                episode_dir = future.result()
                print(f"[PS5] save complete -> {episode_dir}")
            else:
                still_pending.append(future)
        save_futures = still_pending
        pending_save_count = len(save_futures)
        if pending_save_count != last_pending_save_count:
            if pending_save_count > 0:
                save_status_model.set_value(f"{pending_save_count} pending")
                print(f"[PS5] saves pending -> {pending_save_count}")
            else:
                save_status_model.set_value("idle")
                if last_pending_save_count > 0:
                    print("[PS5] saves pending -> 0")
            last_pending_save_count = pending_save_count

        pygame.event.pump()

        lx = float(joystick.get_axis(0)) if joystick.get_numaxes() > 0 else 0.0
        ly = float(joystick.get_axis(1)) if joystick.get_numaxes() > 1 else 0.0
        rx = float(joystick.get_axis(2)) if joystick.get_numaxes() > 2 else 0.0
        ry = float(joystick.get_axis(3)) if joystick.get_numaxes() > 3 else 0.0
        l2 = float(joystick.get_axis(4)) if joystick.get_numaxes() > 4 else -1.0
        r2 = float(joystick.get_axis(5)) if joystick.get_numaxes() > 5 else -1.0

        if abs(lx) < args.stick_deadband:
            lx = 0.0
        if abs(ly) < args.stick_deadband:
            ly = 0.0
        if abs(rx) < args.stick_deadband:
            rx = 0.0
        if abs(ry) < args.stick_deadband:
            ry = 0.0

        l2_norm = max(0.0, min(1.0, (l2 + 1.0) * 0.5))
        r2_norm = max(0.0, min(1.0, (r2 + 1.0) * 0.5))
        if l2_norm < args.trigger_deadband:
            l2_norm = 0.0
        if r2_norm < args.trigger_deadband:
            r2_norm = 0.0

        delta = np.array(
            [
                -lx * args.translation_scale,
                ly * args.translation_scale,
                (r2_norm - l2_norm) * args.translation_scale,
            ],
            dtype=np.float32,
        )
        target_position += delta

        buttons = [i for i in range(joystick.get_numbuttons()) if joystick.get_button(i)]
        yaw_delta = 0.0
        if 9 in buttons:
            yaw_delta -= args.yaw_scale
        if 10 in buttons:
            yaw_delta += args.yaw_scale
        rot_delta = np.array(
            [
                -ry * args.rotation_scale,
                -rx * args.rotation_scale,
                yaw_delta,
            ],
            dtype=np.float64,
        )
        if np.any(rot_delta != 0.0):
            delta_quat = euler_angles_to_quats(rot_delta, degrees=False)
            target_orientation = normalize_quat_wxyz(quat_multiply_wxyz(delta_quat, target_orientation))

        target_position[0] = np.clip(target_position[0], -0.50, 0.50)
        target_position[1] = np.clip(target_position[1], -0.05, 0.75)
        target_position[2] = np.clip(target_position[2], table_surface_z + 0.03, 1.35)
        set_translate(target_marker.GetPrim(), target_position)
        status_model.set_value(f"({target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f})")

        gripper_action = 0
        if 0 in buttons and 0 not in last_buttons:
            gripper_open = not gripper_open
            gripper_action = 1 if gripper_open else -1
            print(f"[PS5] gripper toggled -> {'open' if gripper_open else 'closed'}")
        if 2 in buttons and 2 not in last_buttons:
            if not recording:
                cube_pose = cube.get_world_pose()
                scene_config["cube"]["start_position"] = np.asarray(cube_pose[0], dtype=np.float32).tolist()
                scene_config["cube"]["start_orientation_wxyz"] = np.asarray(cube_pose[1], dtype=np.float32).tolist()
                episode = create_episode_buffer()
                episode_step_index = 0
                recording = True
                print("[PS5] recording started")
            else:
                recording = False
                episode["success"] = bool(episode["success"] or success_latched)
                save_futures.append(
                    save_executor.submit(
                        save_episode,
                        output_dir=os.path.abspath(args.data_dir),
                        episode=episode,
                        task_name="simple_pick_red_to_blue",
                        source_pad_center=source_pad_center,
                        target_pad_center=target_pad_center,
                        cube_start=np.asarray(scene_config["cube"]["start_position"], dtype=np.float32),
                        ee_frame=args.ee_frame,
                        image_format=args.image_format,
                        jpeg_quality=args.jpeg_quality,
                        scene_config=copy.deepcopy(scene_config),
                    )
                )
                print("[PS5] recording stopped -> saving in background")
                save_status_model.set_value(f"{len(save_futures)} pending")
                episode = None
        if 1 in buttons and 1 not in last_buttons:
            if recording:
                recording = False
                episode = None
                print("[PS5] recording discarded")
            target_position = initial_target_position.copy()
            target_orientation = initial_target_orientation.copy()
            cube.set_world_pose(position=initial_cube_position, orientation=initial_cube_orientation)
            cube.set_linear_velocity(np.zeros(3, dtype=np.float32))
            cube.set_angular_velocity(np.zeros(3, dtype=np.float32))
            print("[PS5] target reset -> initial pose | cube reset")

        action, success = art_kinematics.compute_inverse_kinematics(target_position=target_position, target_orientation=target_orientation)
        finger_targets = gripper_open_target if gripper_open else gripper_closed_target
        if success:
            full_targets = np.asarray(robot.get_joint_positions(), dtype=np.float32).copy()
            lula_targets = np.asarray(action.joint_positions, dtype=np.float32).reshape(-1)
            for lula_idx, joint_name in enumerate(lula_solver.get_joint_names()):
                full_targets[joint_name_to_index[joint_name]] = lula_targets[lula_idx]
            full_targets[joint7_idx] = finger_targets[0]
            full_targets[joint8_idx] = finger_targets[1]
            last_valid_targets = full_targets.copy()
            articulation_controller.apply_action(ArticulationAction(joint_positions=full_targets))
        else:
            fallback_targets = last_valid_targets.copy()
            fallback_targets[joint7_idx] = finger_targets[0]
            fallback_targets[joint8_idx] = finger_targets[1]
            articulation_controller.apply_action(ArticulationAction(joint_positions=fallback_targets))

        ee_world_position, ee_world_rotation = art_kinematics.compute_end_effector_pose()
        ee_world_position = np.asarray(ee_world_position, dtype=np.float64)
        ee_world_rotation = np.asarray(ee_world_rotation, dtype=np.float64)
        ray_direction = ee_world_rotation @ tool_local_axis
        best_t = None
        hit_point = None

        cube_pos = np.asarray(cube.get_world_pose()[0], dtype=np.float64)
        cube_half = np.array([args.cube_size * 0.5, args.cube_size * 0.5, args.cube_size * 0.5], dtype=np.float64)
        cube_t = ray_aabb_intersection(ee_world_position, ray_direction, cube_pos - cube_half, cube_pos + cube_half)
        if cube_t is not None and cube_t > 0.0:
            best_t = cube_t
            hit_point = ee_world_position + cube_t * ray_direction

        if abs(ray_direction[2]) > 1e-6:
            table_t = (table_surface_z - ee_world_position[2]) / ray_direction[2]
            if table_t > 0.0 and (best_t is None or table_t < best_t):
                best_t = table_t
                hit_point = ee_world_position + table_t * ray_direction

        if hit_point is not None:
            set_translate(pointer_marker.GetPrim(), hit_point)
        else:
            set_translate(pointer_marker.GetPrim(), np.array([0.0, 0.0, -10.0], dtype=np.float32))

        cube_pos = cube_pos.astype(np.float32)
        in_target = (
            abs(float(cube_pos[0] - target_pad_center[0])) <= target_half_extent_x
            and abs(float(cube_pos[1] - target_pad_center[1])) <= target_half_extent_y
            and float(cube_pos[2]) <= table_surface_z + args.cube_size
        )
        if in_target and not success_latched:
            print(f"[SimplePick] success: cube entered target square at {cube_pos}")
            success_latched = True
            if recording and episode is not None:
                episode["success"] = True
        elif not in_target:
            success_latched = False

        if buttons != last_buttons:
            print(
                f"[PS5] buttons={buttons or ['NONE']} "
                f"target=({target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}) ik={success}"
            )
            last_buttons = buttons

        world.step(render=True)
        top_rgba = np.asarray(topdown_sensor_camera.get_rgba(device="cpu"), dtype=np.uint8)
        wrist_rgba = np.asarray(robot_sensor_camera.get_rgba(device="cpu"), dtype=np.uint8)
        if args.camera_panels and top_rgba.ndim == 3:
            update_camera_provider_from_rgba(topdown_provider, top_rgba)
        if args.camera_panels and wrist_rgba.ndim == 3:
            update_camera_provider_from_rgba(robot_provider, wrist_rgba)
        if recording and episode is not None and top_rgba.ndim == 3 and wrist_rgba.ndim == 3:
            episode["top_rgb"].append(top_rgba[:, :, :3].copy())
            episode["wrist_rgb"].append(wrist_rgba[:, :, :3].copy())
            episode["steps"].append(
                {
                    "step_index": episode_step_index,
                    "timestamp": time.time(),
                    "joint_position": np.asarray(robot.get_joint_positions(), dtype=np.float32).copy(),
                    "ee_position": np.asarray(ee_world_position, dtype=np.float32).copy(),
                    "ee_delta": np.array(
                        [delta[0], delta[1], delta[2], rot_delta[0], rot_delta[1], rot_delta[2]],
                        dtype=np.float32,
                    ),
                    "gripper_action": gripper_action,
                    "gripper_open": 1 if gripper_open else 0,
                }
            )
            episode_step_index += 1
finally:
    save_executor.shutdown(wait=False)
    joystick.quit()
    pygame.joystick.quit()
    pygame.quit()
    simulation_app.close()
