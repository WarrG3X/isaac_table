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
from isaacsim.core.prims import GeometryPrim, SingleArticulation, SingleRigidPrim
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats, rot_matrices_to_quats
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from isaacsim.storage.native import get_assets_root_path
from PIL import Image
from isaacsim.sensors.camera import Camera
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, UsdPhysics, Vt
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
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data", "clutter_pick_raw")
from scenes.clutter_pick import default_scene_config
ASSET_ROOT = r"C:\Users\Warra\Downloads\Assets\Isaac\5.1"
YCB_AXIS_ALIGNED_DIR = os.path.join(ASSET_ROOT, "Isaac", "Props", "YCB", "Axis_Aligned")


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


def list_ycb_assets():
    entries = [
        os.path.join(YCB_AXIS_ALIGNED_DIR, name)
        for name in sorted(os.listdir(YCB_AXIS_ALIGNED_DIR))
        if name.endswith(".usd") and not name.startswith(".")
    ]
    if not entries:
        raise RuntimeError(f"No YCB USD files found in: {YCB_AXIS_ALIGNED_DIR}")
    return entries


def compute_bbox_range(usd_path):
    temp_stage = Usd.Stage.Open(usd_path)
    if temp_stage is None:
        return None
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    world = Gf.Range3d()
    found = False
    for prim in temp_stage.Traverse():
        if prim.IsA(UsdGeom.Imageable):
            bbox = bbox_cache.ComputeWorldBound(prim)
            rng = bbox.ComputeAlignedRange()
            if not rng.IsEmpty():
                if not found:
                    world = Gf.Range3d(rng.GetMin(), rng.GetMax())
                    found = True
                else:
                    world.UnionWith(rng)
    if not found:
        return None
    return world


def compute_bbox_volume(usd_path):
    rng = compute_bbox_range(usd_path)
    if rng is None:
        return None
    size = rng.GetSize()
    return float(size[0]) * float(size[1]) * float(size[2])


def find_meshes(root_prim):
    return [prim for prim in Usd.PrimRange(root_prim) if prim.IsA(UsdGeom.Mesh)]


def auto_generate_convex_colliders(root_prim):
    meshes = find_meshes(root_prim)
    for mesh in meshes:
        UsdPhysics.CollisionAPI.Apply(mesh)
        mesh_api = UsdPhysics.MeshCollisionAPI.Apply(mesh)
        mesh_api.CreateApproximationAttr().Set("convexHull")
    return meshes


def build_tray(stage, root_path, center_xy, surface_z, outer_size, height, wall, material):
    outer_w, outer_d = outer_size
    tray_floor_z = surface_z + wall * 0.5
    tray_center_z = surface_z + height * 0.5
    inner_w = outer_w - 2.0 * wall
    inner_d = outer_d - 2.0 * wall
    parts = {
        "Base": ((outer_w, outer_d, wall), (center_xy[0], center_xy[1], tray_floor_z)),
        "WallL": ((wall, outer_d, height), (center_xy[0] - outer_w * 0.5 + wall * 0.5, center_xy[1], tray_center_z)),
        "WallR": ((wall, outer_d, height), (center_xy[0] + outer_w * 0.5 - wall * 0.5, center_xy[1], tray_center_z)),
        "WallB": ((inner_w, wall, height), (center_xy[0], center_xy[1] - outer_d * 0.5 + wall * 0.5, tray_center_z)),
        "WallF": ((inner_w, wall, height), (center_xy[0], center_xy[1] + outer_d * 0.5 - wall * 0.5, tray_center_z)),
    }
    paths = []
    for name, (size, translate) in parts.items():
        prim = define_box(stage, f"{root_path}/{name}", size=size, translate=translate)
        bind_material(prim, material)
        paths.append(str(prim.GetPath()))
    return paths


def build_tabletop_with_cutouts(stage, root_path, table_size, table_top_z, hole_specs, base_material, surface_material):
    table_half_x = table_size[0] * 0.5
    table_half_y = table_size[1] * 0.5
    table_surface_z = table_top_z + table_size[2] * 0.5
    x_breaks = [-table_half_x]
    for center_x, _, hole_w, _ in hole_specs:
        x_breaks.extend([center_x - hole_w * 0.5, center_x + hole_w * 0.5])
    x_breaks.append(table_half_x)
    x_breaks = sorted(set(round(v, 6) for v in x_breaks))

    box_paths = []
    for slab_idx in range(len(x_breaks) - 1):
        x0 = x_breaks[slab_idx]
        x1 = x_breaks[slab_idx + 1]
        slab_width = x1 - x0
        if slab_width <= 1e-6:
            continue

        active_holes = []
        for center_x, center_y, hole_w, hole_d in hole_specs:
            hole_min_x = center_x - hole_w * 0.5
            hole_max_x = center_x + hole_w * 0.5
            if x0 >= hole_min_x - 1e-6 and x1 <= hole_max_x + 1e-6:
                active_holes.append((center_y, hole_d))

        if not active_holes:
            piece = define_box(
                stage,
                f"{root_path}/Top_{slab_idx}",
                size=(slab_width, table_size[1], table_size[2]),
                translate=((x0 + x1) * 0.5, 0.0, table_top_z),
            )
            bind_material(piece, base_material)
            top_plane = define_uv_plane(
                stage,
                f"{root_path}/TopSurface_{slab_idx}",
                size=(max(0.0, slab_width - 0.01), max(0.0, table_size[1] - 0.01)),
                translate=((x0 + x1) * 0.5, 0.0, table_surface_z + 0.001),
            )
            bind_material(top_plane, surface_material)
            box_paths.append(str(piece.GetPath()))
            continue

        center_y, hole_d = active_holes[0]
        hole_min_y = center_y - hole_d * 0.5
        hole_max_y = center_y + hole_d * 0.5
        y_segments = [
            ((-table_half_y + hole_min_y) * 0.5, hole_min_y - (-table_half_y)),
            ((hole_max_y + table_half_y) * 0.5, table_half_y - hole_max_y),
        ]
        for seg_idx, (seg_center_y, seg_depth) in enumerate(y_segments):
            if seg_depth <= 1e-6:
                continue
            piece = define_box(
                stage,
                f"{root_path}/Top_{slab_idx}_{seg_idx}",
                size=(slab_width, seg_depth, table_size[2]),
                translate=((x0 + x1) * 0.5, seg_center_y, table_top_z),
            )
            bind_material(piece, base_material)
            top_plane = define_uv_plane(
                stage,
                f"{root_path}/TopSurface_{slab_idx}_{seg_idx}",
                size=(max(0.0, slab_width - 0.01), max(0.0, seg_depth - 0.01)),
                translate=((x0 + x1) * 0.5, seg_center_y, table_surface_z + 0.001),
            )
            bind_material(top_plane, surface_material)
            box_paths.append(str(piece.GetPath()))

    return box_paths


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


def offset_prim_translate_z(stage, prim_path, delta_z):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    target_prim = prim.GetParent()
    if not target_prim.IsValid():
        target_prim = prim
    xformable = UsdGeom.Xformable(target_prim)
    translate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if not translate_ops:
        return
    current = translate_ops[0].Get()
    translate_ops[0].Set(Gf.Vec3d(float(current[0]), float(current[1]), float(current[2] + delta_z)))


def get_floor_top_z(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(prim)
    rng = bbox.GetRange()
    if rng.IsEmpty():
        return None
    return float(rng.GetMax()[2])


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
        "object_position": [],
        "object_orientation_wxyz": [],
    }


def save_rgb_frame(rgb, path, image_format, jpeg_quality):
    image = Image.fromarray(rgb)
    if image_format == "jpg":
        image.save(path, format="JPEG", quality=int(jpeg_quality), optimize=True)
    else:
        image.save(path, format="PNG")


def save_episode(output_dir, episode, task_name, source_tray_center, target_tray_center, object_info, ee_frame, image_format, jpeg_quality, scene_config):
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
        object_position=np.asarray(episode["object_position"], dtype=np.float32),
        object_orientation_wxyz=np.asarray(episode["object_orientation_wxyz"], dtype=np.float32),
    )

    metadata = {
        "format": "clutter_pick_raw_v1",
        "task": task_name,
        "controller": "ps5",
        "ee_frame": ee_frame,
        "scene_name": scene_config["scene_name"],
        "success": bool(episode["success"]),
        "num_steps": len(episode["steps"]),
        "source_tray_center": list(source_tray_center),
        "target_tray_center": list(target_tray_center),
        "object_info": object_info,
        "images": {
            "top": "images/top",
            "wrist": "images/wrist",
        },
        "image_format": image_ext,
        "data_file": "data.npz",
        "scene_file": "scene.yaml",
        "playback_mode_default": "full_state",
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
parser.add_argument("--table-width", type=float, default=1.90)
parser.add_argument("--table-depth", type=float, default=1.55)
parser.add_argument("--tray-width", type=float, default=0.60)
parser.add_argument("--tray-depth", type=float, default=0.42)
parser.add_argument("--tray-height", type=float, default=0.1275)
parser.add_argument("--tray-wall", type=float, default=0.012)
parser.add_argument("--num-objects", type=int, default=17)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max-volume", type=float, default=0.002)
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
    tray_width=args.tray_width,
    tray_depth=args.tray_depth,
    tray_height=args.tray_height,
    tray_wall=args.tray_wall,
    num_objects=args.num_objects,
    seed=args.seed,
    max_volume=args.max_volume,
)

world = World(stage_units_in_meters=1.0)
stage = world.stage
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.Xform.Define(stage, "/World")
UsdGeom.Xform.Define(stage, "/World/Looks")
UsdGeom.Xform.Define(stage, "/World/Floor")
UsdGeom.Xform.Define(stage, "/World/Table")
UsdGeom.Xform.Define(stage, "/World/SourceTray")
UsdGeom.Xform.Define(stage, "/World/TargetTray")
UsdGeom.Xform.Define(stage, "/World/Clutter")
UsdGeom.Xform.Define(stage, "/World/Lights")
UsdGeom.Xform.Define(stage, "/World/RobotMount")
UsdGeom.Xform.Define(stage, "/World/Debug")

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
target_material = create_omnipbr_material(stage, "/World/Looks/Target", color=(0.82, 0.16, 0.12), roughness=0.25)

floor_base_prim = define_box(stage, "/World/Floor/Base", size=(7.0, 7.0, 0.02), translate=(0.0, 0.0, -0.01))
bind_material(floor_base_prim, floor_material_base)
floor_surface_prim = define_uv_plane(stage, "/World/Floor/Surface", size=(6.92, 6.92), translate=(0.0, 0.0, 0.001), uv_scale=(6.0, 6.0))
bind_material(floor_surface_prim, floor_material)

table_size = (args.table_width, args.table_depth, 0.06)
table_top_z = 0.75

leg_height = 0.72
leg_dx = table_size[0] * 0.5 - 0.08
leg_dy = table_size[1] * 0.5 - 0.08
leg_center_z = leg_height * 0.5
floor_top_z = get_floor_top_z(stage, "/World/Room/Towel_Room01_floor_bottom_218")
if floor_top_z is not None:
    leg_center_z = floor_top_z + leg_height * 0.5
    table_top_z = floor_top_z + leg_height + table_size[2] * 0.5
table_surface_z = table_top_z + table_size[2] * 0.5
table_leg_paths = []
for name, position in {
    "LegFL": (leg_dx, leg_dy, leg_center_z),
    "LegFR": (leg_dx, -leg_dy, leg_center_z),
    "LegBL": (-leg_dx, leg_dy, leg_center_z),
    "LegBR": (-leg_dx, -leg_dy, leg_center_z),
}.items():
    leg_prim = define_box(stage, f"/World/Table/{name}", size=(0.06, 0.06, leg_height), translate=position)
    bind_material(leg_prim, leg_material)
    table_leg_paths.append(str(leg_prim.GetPath()))

source_tray_center = (-0.38, 0.178)
target_tray_center = (0.38, 0.178)
task_center = (0.5 * (source_tray_center[0] + target_tray_center[0]), 0.5 * (source_tray_center[1] + target_tray_center[1]), table_surface_z)
task_coverage_xy = (abs(target_tray_center[0] - source_tray_center[0]) + args.tray_width + 0.24, args.tray_depth + 0.28)
source_tray_paths = build_tray(
    stage,
    "/World/SourceTray",
    center_xy=source_tray_center,
    surface_z=table_surface_z,
    outer_size=(args.tray_width, args.tray_depth),
    height=args.tray_height,
    wall=args.tray_wall,
    material=source_tray_material,
)
target_tray_paths = build_tray(
    stage,
    "/World/TargetTray",
    center_xy=target_tray_center,
    surface_z=table_surface_z,
    outer_size=(args.tray_width, args.tray_depth),
    height=args.tray_height,
    wall=args.tray_wall,
    material=target_tray_material,
)

for path in source_tray_paths + target_tray_paths:
    offset_prim_translate_z(stage, path, -args.tray_height)

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
table_fill.CreateWidthAttr(2.8)
table_fill.CreateHeightAttr(1.4)
table_fill.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))
UsdGeom.Xformable(table_fill.GetPrim()).AddTranslateOp().Set(Gf.Vec3f(0.3, 0.05, 2.05))
UsdGeom.Xformable(table_fill.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-90.0, 0.0, 0.0))

tabletop_paths = build_tabletop_with_cutouts(
    stage,
    "/World/Table",
    table_size=table_size,
    table_top_z=table_top_z,
    hole_specs=[
        (source_tray_center[0], source_tray_center[1], args.tray_width, args.tray_depth),
        (target_tray_center[0], target_tray_center[1], args.tray_width, args.tray_depth),
    ],
    base_material=wood_base_material,
    surface_material=wood_material,
)
apply_static_collision([str(floor_base_prim.GetPath()), *tabletop_paths, *table_leg_paths, *source_tray_paths, *target_tray_paths])

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

rng = np.random.default_rng(args.seed)
ycb_assets = [path for path in list_ycb_assets() if (compute_bbox_volume(path) or 0.0) < args.max_volume]
if not ycb_assets:
    raise RuntimeError(f"No YCB assets remain after applying max-volume filter {args.max_volume}")
selected_count = min(args.num_objects, len(ycb_assets))
selected_indices = rng.choice(len(ycb_assets), size=selected_count, replace=False)
inner_half_x = args.tray_width * 0.5 - args.tray_wall - 0.025
inner_half_y = args.tray_depth * 0.5 - args.tray_wall - 0.025
drop_base_z = table_surface_z + args.tray_height + 0.08
clutter_objects = []
object_info = []
for local_idx, asset_idx in enumerate(selected_indices):
    usd_path = ycb_assets[int(asset_idx)]
    prim_path = f"/World/Clutter/Object_{local_idx}"
    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
    auto_generate_convex_colliders(stage.GetPrimAtPath(prim_path))
    x = float(source_tray_center[0] + rng.uniform(-inner_half_x, inner_half_x))
    y = float(source_tray_center[1] + rng.uniform(-inner_half_y, inner_half_y))
    z = float(drop_base_z + 0.06 * local_idx)
    yaw = float(rng.uniform(0.0, 360.0))
    quat = np.array([np.cos(np.deg2rad(yaw) * 0.5), 0.0, 0.0, np.sin(np.deg2rad(yaw) * 0.5)], dtype=np.float32)
    rigid = world.scene.add(
        SingleRigidPrim(
            prim_path=prim_path,
            name=f"clutter_object_{local_idx}",
            position=np.array([x, y, z], dtype=np.float32),
            orientation=quat,
        )
    )
    bbox_range = compute_bbox_range(usd_path)
    bbox_size = bbox_range.GetSize() if bbox_range is not None else Gf.Vec3d(0.05, 0.05, 0.05)
    bbox_size_np = np.array([float(bbox_size[0]), float(bbox_size[1]), float(bbox_size[2])], dtype=np.float32)
    clutter_objects.append(
        {
            "rigid": rigid,
            "name": os.path.basename(usd_path),
            "bbox_size": bbox_size_np,
            "initial_position": np.array([x, y, z], dtype=np.float32),
            "initial_orientation": quat.copy(),
        }
    )
    object_info.append(
        {
            "name": os.path.basename(usd_path),
            "usd_path": usd_path,
            "initial_position": [x, y, z],
            "initial_orientation_wxyz": quat.tolist(),
            "bbox_size": bbox_size_np.tolist(),
        }
    )
scene_config["objects"]["spawned"] = copy.deepcopy(object_info)
scene_config["objects"]["bank_size_after_filter"] = len(ycb_assets)

target_marker = UsdGeom.Sphere.Define(stage, "/World/Debug/TeleopTarget")
target_marker.CreateRadiusAttr(0.018)
bind_material(target_marker.GetPrim(), target_material)
pointer_marker = UsdGeom.Sphere.Define(stage, "/World/Debug/ToolProjection")
pointer_marker.CreateRadiusAttr(0.014)
bind_material(pointer_marker.GetPrim(), target_tray_material)

set_camera_view(eye=[0.0, 2.15, 1.55], target=[0.0, 0.20, 0.82], camera_prim_path="/OmniverseKit_Persp")
if args.camera_panels:
    topdown_window = build_camera_window("Clutter Pick Top Down", topdown_provider, 640, 480, 980, 40)
    robot_window = build_camera_window("Clutter Pick Wrist", robot_provider, 640, 480, 980, 560)

world.reset()
robot.initialize()
for entry in clutter_objects:
    entry["rigid"].initialize()
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
print(f"[ClutterPick] source tray center: {source_tray_center}")
print(f"[ClutterPick] target tray center: {target_tray_center}")
print(f"[ClutterPick] ycb bank size after volume filter < {args.max_volume}: {len(ycb_assets)}")
print(f"[ClutterPick] spawned objects: {len(clutter_objects)}")
print(f"[ClutterPick] Lula active joints: {lula_solver.get_joint_names()}")
print(f"[ClutterPick] camera panels: {'enabled' if args.camera_panels else 'disabled'}")
if args.camera_panels:
    print(f"[ClutterPick] top-down camera panel: {topdown_window.title}")
    print(f"[ClutterPick] wrist camera panel: {robot_window.title}")
for entry in clutter_objects:
    print(f"  - {entry['name']}")

for prim_path in ["/World/Floor/Base", "/World/Floor/Surface", "/World/Lights/Dome", "/World/Lights/Sun"]:
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        prim.SetActive(False)

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
print(f"[ClutterPick] tool projection axis: {tool_axis_name}")

joystick = init_controller(args.controller)
last_buttons = []
gripper_open = True
recording = False
episode = None
episode_step_index = 0
save_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="episode_save")
save_futures = []
last_pending_save_count = 0
current_object_info = copy.deepcopy(object_info)
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
                current_object_info = []
                current_spawned = []
                for entry in clutter_objects:
                    obj_position, obj_orientation = entry["rigid"].get_world_pose()
                    obj_position = np.asarray(obj_position, dtype=np.float32)
                    obj_orientation = np.asarray(obj_orientation, dtype=np.float32)
                    item = {
                        "name": entry["name"],
                        "usd_path": next(info["usd_path"] for info in object_info if info["name"] == entry["name"] and np.allclose(info["bbox_size"], entry["bbox_size"].tolist())),
                        "initial_position": obj_position.tolist(),
                        "initial_orientation_wxyz": obj_orientation.tolist(),
                        "bbox_size": entry["bbox_size"].tolist(),
                    }
                    current_object_info.append(item)
                    current_spawned.append({**item, "prim_path": next(info.get("prim_path", f"/World/Clutter/Object_{idx}") for idx, info in enumerate(scene_config["objects"]["spawned"]) if info["name"] == entry["name"] and np.allclose(info["bbox_size"], entry["bbox_size"].tolist()))})
                scene_config["objects"]["spawned"] = copy.deepcopy(current_spawned)
                episode = create_episode_buffer()
                episode_step_index = 0
                recording = True
                print("[PS5] recording started")
            else:
                recording = False
                save_futures.append(
                    save_executor.submit(
                        save_episode,
                        output_dir=os.path.abspath(args.data_dir),
                        episode=episode,
                        task_name="clutter_pick_source_to_target",
                        source_tray_center=source_tray_center,
                        target_tray_center=target_tray_center,
                        object_info=current_object_info,
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
            for entry in clutter_objects:
                entry["rigid"].set_world_pose(position=entry["initial_position"], orientation=entry["initial_orientation"])
                entry["rigid"].set_linear_velocity(np.zeros(3, dtype=np.float32))
                entry["rigid"].set_angular_velocity(np.zeros(3, dtype=np.float32))
            print("[PS5] target reset -> initial pose | clutter reset")

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
        for entry in clutter_objects:
            obj_pos = np.asarray(entry["rigid"].get_world_pose()[0], dtype=np.float64)
            half = 0.5 * entry["bbox_size"].astype(np.float64)
            obj_t = ray_aabb_intersection(ee_world_position, ray_direction, obj_pos - half, obj_pos + half)
            if obj_t is not None and obj_t > 0.0 and (best_t is None or obj_t < best_t):
                best_t = obj_t
                hit_point = ee_world_position + obj_t * ray_direction

        if abs(ray_direction[2]) > 1e-6:
            table_t = (table_surface_z - ee_world_position[2]) / ray_direction[2]
            if table_t > 0.0 and (best_t is None or table_t < best_t):
                best_t = table_t
                hit_point = ee_world_position + table_t * ray_direction

        if hit_point is not None:
            set_translate(pointer_marker.GetPrim(), hit_point)
        else:
            set_translate(pointer_marker.GetPrim(), np.array([0.0, 0.0, -10.0], dtype=np.float32))

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
            object_positions = []
            object_orientations = []
            for entry in clutter_objects:
                obj_position, obj_orientation = entry["rigid"].get_world_pose()
                object_positions.append(np.asarray(obj_position, dtype=np.float32))
                object_orientations.append(np.asarray(obj_orientation, dtype=np.float32))
            episode["top_rgb"].append(top_rgba[:, :, :3].copy())
            episode["wrist_rgb"].append(wrist_rgba[:, :, :3].copy())
            episode["object_position"].append(np.stack(object_positions, axis=0))
            episode["object_orientation_wxyz"].append(np.stack(object_orientations, axis=0))
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
