from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import os
import sys

import isaacsim.core.utils.numpy.rotations as rot_utils
import numpy as np
import omni.kit.commands
import omni.ui as ui
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder, DynamicSphere
from isaacsim.core.prims import GeometryPrim, SingleArticulation, SingleRigidPrim
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats, rot_matrices_to_quats
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.semantics import add_labels
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction, ArticulationActions
from isaacsim.core.utils.viewports import create_viewport_for_camera, set_camera_view
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path
from PIL import Image
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ISAAC_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", "..", ".."))
DEFAULT_PIPER_USD = os.path.join(ISAAC_ROOT, "piper_isaac_sim", "USD", "piper_x_v1.usd")
DEFAULT_LULA_YAML = os.path.join(REPO_ROOT, "desc", "piper_x_robot_description.yaml")
DEFAULT_LULA_URDF = os.path.join(
    ISAAC_ROOT, "piper_isaac_sim", "piper_x_description", "urdf", "piper_x_description_d435.urdf"
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--robot-usd",
    type=str,
    default=DEFAULT_PIPER_USD,
    help="Path to the Piper X USD to reference into the scene.",
)
parser.add_argument("--robot-x", type=float, default=0.0, help="Robot base X position on the table.")
parser.add_argument("--robot-y", type=float, default=0.12, help="Robot base Y position on the table.")
parser.add_argument("--robot-yaw", type=float, default=90.0, help="Robot base yaw in degrees.")
parser.add_argument("--test-motion", action="store_true", help="Run a simple joint-space motion sequence after reset.")
parser.add_argument("--hold-frames", type=int, default=180, help="Frames to hold each test pose.")
parser.add_argument("--circle-motion", action="store_true", help="Run a smooth continuous wrist motion test.")
parser.add_argument("--robot-camera-view", action="store_true", help="Open a second viewport from the robot-mounted camera.")
parser.add_argument("--joint-ui", action="store_true", help="Open joint sliders for manual robot control.")
parser.add_argument("--object-source", choices=["none", "prims", "ycb"], default="none", help="Populate the side bin with dropped clutter.")
parser.add_argument("--num-objects", type=int, default=10, help="Number of clutter objects to drop into the bin.")
parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for clutter placement.")
parser.add_argument("--asset-root", type=str, default="C:/Users/Warra/Downloads/Assets/Isaac/5.1", help="Local Isaac asset root containing the top-level Isaac folder.")
parser.add_argument("--topdown-camera", action="store_true", help="Create the overhead RGB-D camera used by the clutter scene.")
parser.add_argument("--save-camera-debug", action="store_true", help="Save RGB/depth/instance-seg snapshots from the overhead camera after clutter settles.")
parser.add_argument("--lula-list-frames", action="store_true", help="Print Lula frame names and exit.")
parser.add_argument("--lula-ik-test", action="store_true", help="Run a simple Lula IK solve and drive to the target.")
parser.add_argument("--lula-yaml", type=str, default=DEFAULT_LULA_YAML, help="Path to exported Lula robot description YAML.")
parser.add_argument("--lula-urdf", type=str, default=DEFAULT_LULA_URDF, help="Path to URDF used by Lula.")
parser.add_argument("--ee-frame", type=str, default="Link6", help="End effector frame name for Lula IK.")
parser.add_argument("--ik-target-dx", type=float, default=0.0, help="Target X offset from current EE pose for Lula IK.")
parser.add_argument("--ik-target-dy", type=float, default=0.08, help="Target Y offset from current EE pose for Lula IK.")
parser.add_argument("--ik-target-dz", type=float, default=-0.04, help="Target Z offset from current EE pose for Lula IK.")
parser.add_argument(
    "--ik-position-only",
    action="store_true",
    help="Ignore EE orientation in the Lula IK test and solve only for position.",
)
parser.add_argument("--ik-ui", action="store_true", help="Open a simple Lula IK slider panel for interactive testing.")
args, _ = parser.parse_known_args()


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
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(-0.5, -0.5, 0.0),
            Gf.Vec3f(0.5, -0.5, 0.0),
            Gf.Vec3f(0.5, 0.5, 0.0),
            Gf.Vec3f(-0.5, 0.5, 0.0),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, 1.0)] * 4)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateExtentAttr([Gf.Vec3f(-0.5, -0.5, 0.0), Gf.Vec3f(0.5, 0.5, 0.0)])

    primvars_api = UsdGeom.PrimvarsAPI(mesh)
    st = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(
        Vt.Vec2fArray(
            [
                Gf.Vec2f(0.0, 0.0),
                Gf.Vec2f(uv_scale[0], 0.0),
                Gf.Vec2f(uv_scale[0], uv_scale[1]),
                Gf.Vec2f(0.0, uv_scale[1]),
            ]
        )
    )
    return mesh.GetPrim()


def create_omnipbr_material(
    stage,
    material_path,
    color=None,
    roughness=0.5,
    metallic=0.0,
    texture_path=None,
    normal_texture_path=None,
    orm_texture_path=None,
    enable_orm=False,
    diffuse_tint=None,
    albedo_brightness=None,
):
    mtl_created_list = []
    omni.kit.commands.execute(
        "CreateAndBindMdlMaterialFromLibrary",
        mdl_name="OmniPBR.mdl",
        mtl_name="OmniPBR",
        mtl_created_list=mtl_created_list,
    )

    material_prim = stage.GetPrimAtPath(mtl_created_list[0])
    if mtl_created_list[0] != material_path:
        omni.kit.commands.execute("MovePrim", path_from=mtl_created_list[0], path_to=material_path)
        material_prim = stage.GetPrimAtPath(material_path)

    if color is not None:
        omni.usd.create_material_input(
            material_prim, "diffuse_color_constant", Gf.Vec3f(*color), Sdf.ValueTypeNames.Color3f
        )
    if diffuse_tint is not None:
        omni.usd.create_material_input(material_prim, "diffuse_tint", Gf.Vec3f(*diffuse_tint), Sdf.ValueTypeNames.Color3f)
    omni.usd.create_material_input(material_prim, "reflection_roughness_constant", roughness, Sdf.ValueTypeNames.Float)
    omni.usd.create_material_input(material_prim, "metallic_constant", metallic, Sdf.ValueTypeNames.Float)
    if albedo_brightness is not None:
        omni.usd.create_material_input(material_prim, "albedo_brightness", albedo_brightness, Sdf.ValueTypeNames.Float)
    if texture_path:
        omni.usd.create_material_input(material_prim, "diffuse_texture", texture_path, Sdf.ValueTypeNames.Asset)
    if normal_texture_path:
        omni.usd.create_material_input(material_prim, "normalmap_texture", normal_texture_path, Sdf.ValueTypeNames.Asset)
    if orm_texture_path:
        omni.usd.create_material_input(material_prim, "ORM_texture", orm_texture_path, Sdf.ValueTypeNames.Asset)
    if enable_orm:
        omni.usd.create_material_input(material_prim, "enable_ORM_texture", True, Sdf.ValueTypeNames.Bool)

    return UsdShade.Material(material_prim)


def bind_material(prim, material):
    UsdShade.MaterialBindingAPI(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)


def apply_static_collision(prim_paths):
    for prim_path in prim_paths:
        GeometryPrim(prim_path).apply_collision_apis()


def resolve_asset_root(explicit_asset_root):
    if explicit_asset_root and os.path.exists(os.path.join(explicit_asset_root, "Isaac")):
        return explicit_asset_root
    try:
        detected_asset_root = get_assets_root_path()
        if detected_asset_root and os.path.exists(os.path.join(detected_asset_root, "Isaac")):
            return detected_asset_root
    except Exception:
        pass
    return None


def save_camera_debug(camera, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    rgb = np.asarray(camera.get_rgb(device="cpu"), dtype=np.uint8)
    depth = np.asarray(camera.get_depth(device="cpu"), dtype=np.float32)
    finite_depth = depth[np.isfinite(depth)]

    if finite_depth.size:
        print(
            f"[TopDownCamera] depth finite pixels={finite_depth.size} "
            f"min={finite_depth.min():.4f}m max={finite_depth.max():.4f}m mean={finite_depth.mean():.4f}m"
        )
    else:
        print("[TopDownCamera] no finite depth pixels found")

    Image.fromarray(rgb).save(os.path.join(output_dir, "realsense_rgb.png"))

    depth_to_save = depth.copy()
    depth_to_save[~np.isfinite(depth_to_save)] = 0.0
    np.save(os.path.join(output_dir, "realsense_depth.npy"), depth_to_save)

    if finite_depth.size:
        near = np.percentile(finite_depth, 2)
        far = np.percentile(finite_depth, 98)
        if far <= near:
            far = near + 1e-3
        depth_normalized = np.clip((depth_to_save - near) / (far - near), 0.0, 1.0)
        depth_normalized = (255.0 * (1.0 - depth_normalized)).astype(np.uint8)
        Image.fromarray(depth_normalized).save(os.path.join(output_dir, "realsense_depth_vis.png"))

    print(f"[TopDownCamera] saved RGB and depth to {output_dir}")


def build_segmentation_palette(num_objects):
    object_palette = []
    for idx in range(num_objects):
        hue = (37 * idx) % 255
        object_palette.append(
            np.array(
                [64 + (hue % 128), 96 + ((hue * 3) % 128), 64 + ((hue * 5) % 128)],
                dtype=np.uint8,
            )
        )
    return {
        "background": np.array([20, 20, 20], dtype=np.uint8),
        "bin": np.array([60, 120, 220], dtype=np.uint8),
        "objects": object_palette,
    }


def collect_segmentation_ids(current_frame_entry):
    info = current_frame_entry.get("info", {})
    id_to_labels = info.get("idToLabels", {})
    parsed = {}
    for key, value in id_to_labels.items():
        try:
            parsed[int(key)] = value
        except Exception:
            continue
    return parsed


def build_custom_instance_segmentation(camera, output_dir, clutter_objects, bin_prim_prefix="/World/Bin"):
    frame = camera.get_current_frame()
    seg_entry = frame.get("instance_segmentation")
    if seg_entry is None:
        print("[TopDownCamera] instance segmentation frame entry missing")
        return

    seg_data = np.asarray(seg_entry["data"], dtype=np.uint32)
    id_to_labels = collect_segmentation_ids(seg_entry)
    palette = build_segmentation_palette(len(clutter_objects))
    mask = np.zeros((seg_data.shape[0], seg_data.shape[1], 3), dtype=np.uint8)
    mask[:, :] = palette["background"]

    clutter_paths = {get_prim_at_path(obj.prim_path).GetPrimPath().pathString: idx for idx, obj in enumerate(clutter_objects)}
    bin_ids = set()
    object_ids = {}

    for seg_id, labels in id_to_labels.items():
        prim_path = None
        if isinstance(labels, dict):
            prim_path = labels.get("instancePath") or labels.get("primPath")
        elif isinstance(labels, str):
            prim_path = labels
        if prim_path is None:
            continue
        if prim_path.startswith(bin_prim_prefix):
            bin_ids.add(seg_id)
            continue
        for clutter_path, object_idx in clutter_paths.items():
            if prim_path.startswith(clutter_path):
                object_ids[seg_id] = object_idx
                break

    for seg_id in bin_ids:
        mask[seg_data == seg_id] = palette["bin"]
    for seg_id, object_idx in object_ids.items():
        mask[seg_data == seg_id] = palette["objects"][object_idx]

    out_path = os.path.join(output_dir, "realsense_instance_seg.png")
    Image.fromarray(mask).save(out_path)
    print(f"[TopDownCamera] saved instance segmentation to {out_path}")


def spawn_primitive_objects(world, rng, count, bin_center_y, table_surface_z):
    object_specs = [
        {"kind": "cuboid", "size": 1.0, "scale": np.array([0.055, 0.08, 0.035]), "mass": 0.07, "color": np.array([214, 86, 73])},
        {"kind": "cuboid", "size": 1.0, "scale": np.array([0.05, 0.05, 0.12]), "mass": 0.09, "color": np.array([78, 121, 167])},
        {"kind": "sphere", "radius": 0.03, "mass": 0.05, "color": np.array([89, 161, 79])},
        {"kind": "sphere", "radius": 0.026, "mass": 0.04, "color": np.array([242, 142, 43])},
        {"kind": "cylinder", "radius": 0.028, "height": 0.09, "mass": 0.08, "color": np.array([176, 122, 161])},
        {"kind": "cylinder", "radius": 0.022, "height": 0.12, "mass": 0.06, "color": np.array([118, 183, 178])},
    ]
    spawned = []
    drop_height = table_surface_z + 0.22
    for idx in range(count):
        spec = object_specs[idx % len(object_specs)]
        x = float(rng.uniform(-0.18, 0.18))
        y = float(bin_center_y + rng.uniform(-0.09, 0.09))
        z = float(drop_height + 0.035 * idx)
        yaw = float(rng.uniform(0.0, 360.0))
        if spec["kind"] == "cuboid":
            obj = world.scene.add(
                DynamicCuboid(
                    prim_path=f"/World/Clutter/Object_{idx}",
                    name=f"clutter_box_{idx}",
                    position=np.array([x, y, z]),
                    orientation=rot_utils.euler_angles_to_quats(np.array([0.0, 0.0, yaw]), degrees=True),
                    scale=spec["scale"],
                    size=spec["size"],
                    mass=spec["mass"],
                    color=spec["color"],
                )
            )
        elif spec["kind"] == "sphere":
            obj = world.scene.add(
                DynamicSphere(
                    prim_path=f"/World/Clutter/Object_{idx}",
                    name=f"clutter_sphere_{idx}",
                    position=np.array([x, y, z]),
                    radius=spec["radius"],
                    mass=spec["mass"],
                    color=spec["color"],
                )
            )
        else:
            obj = world.scene.add(
                DynamicCylinder(
                    prim_path=f"/World/Clutter/Object_{idx}",
                    name=f"clutter_cylinder_{idx}",
                    position=np.array([x, y, z]),
                    orientation=rot_utils.euler_angles_to_quats(np.array([90.0, 0.0, yaw]), degrees=True),
                    radius=spec["radius"],
                    height=spec["height"],
                    mass=spec["mass"],
                    color=spec["color"],
                )
            )
        add_labels(get_prim_at_path(obj.prim_path), [f"clutter_{idx}"])
        spawned.append(obj)
    return spawned


def list_ycb_assets(asset_root):
    ycb_dir = os.path.join(asset_root, "Isaac", "Props", "YCB", "Axis_Aligned_Physics")
    if not os.path.exists(ycb_dir):
        raise RuntimeError(f"YCB asset directory not found: {ycb_dir}")
    entries = sorted([os.path.join(ycb_dir, name) for name in os.listdir(ycb_dir) if name.endswith(".usd") and not name.startswith(".")])
    if not entries:
        raise RuntimeError(f"No YCB USD files found in: {ycb_dir}")
    return entries


def spawn_ycb_objects(world, rng, count, asset_root, bin_center_y, table_surface_z):
    ycb_assets = list_ycb_assets(asset_root)
    spawned = []
    drop_height = table_surface_z + 0.24
    for idx in range(count):
        usd_path = ycb_assets[idx % len(ycb_assets)]
        prim_path = f"/World/Clutter/Object_{idx}"
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        x = float(rng.uniform(-0.16, 0.16))
        y = float(bin_center_y + rng.uniform(-0.08, 0.08))
        z = float(drop_height + 0.05 * idx)
        yaw = float(rng.uniform(0.0, 360.0))
        rigid = world.scene.add(
            SingleRigidPrim(
                prim_path=prim_path,
                name=f"ycb_object_{idx}",
                position=np.array([x, y, z]),
                orientation=rot_utils.euler_angles_to_quats(np.array([0.0, 0.0, yaw]), degrees=True),
            )
        )
        add_labels(get_prim_at_path(prim_path), [f"clutter_{idx}"])
        spawned.append(rigid)
    return spawned


def spawn_objects(world, stage, rng, source, count, asset_root, bin_center_y, table_surface_z):
    UsdGeom.Xform.Define(stage, "/World/Clutter")
    if source == "prims":
        return spawn_primitive_objects(world, rng, count, bin_center_y, table_surface_z)
    if source == "ycb":
        if asset_root is None:
            raise RuntimeError("No Isaac asset root is available for YCB spawning.")
        return spawn_ycb_objects(world, rng, count, asset_root, bin_center_y, table_surface_z)
    return []


def objects_are_settled(objects, linear_threshold=0.03, angular_threshold=0.6):
    for obj in objects:
        linear = np.linalg.norm(np.asarray(obj.get_linear_velocity()))
        angular = np.linalg.norm(np.asarray(obj.get_angular_velocity()))
        if linear > linear_threshold or angular > angular_threshold:
            return False
    return True


def deactivate_embedded_environment(stage, root_prim_path):
    candidate_tokens = {
        "environment",
        "groundplane",
        "ground_plane",
        "ground",
        "floor",
        "physicsscene",
        "camerasettings",
        "viewport_l",
        "viewport_r",
    }
    deactivated = []
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if not prim_path.startswith(root_prim_path + "/"):
            continue
        token = prim.GetName().lower()
        if token in candidate_tokens:
            prim.SetActive(False)
            deactivated.append(prim_path)
    for prim_path in deactivated:
        print(f"[PiperX] deactivated embedded prim: {prim_path}")
    if not deactivated:
        print("[PiperX] no embedded environment prims matched for deactivation")


def resolve_robot_usd(robot_usd):
    candidate = os.path.abspath(robot_usd)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Piper X USD not found: {candidate}")


def resolve_required_file(path, label):
    candidate = os.path.abspath(path)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"{label} not found: {candidate}")


def get_test_poses(num_dof):
    home = np.zeros((1, num_dof), dtype=np.float32)
    approach = np.array([[0.0, 0.55, -0.75, 0.0, 0.42, 0.0, 0.01, -0.01]], dtype=np.float32)
    bin_reach = np.array([[0.2, 0.85, -1.15, 0.1, 0.78, 0.15, 0.018, -0.018]], dtype=np.float32)
    return [
        ("home", home),
        ("approach", approach),
        ("bin_reach", bin_reach),
        ("home_return", home),
    ]


def get_circle_motion_target(frame_idx, num_dof):
    target = np.zeros(num_dof, dtype=np.float32)
    t = frame_idx / 60.0
    target[1] = 0.62 + 0.18 * np.sin(t)
    target[2] = -0.95 + 0.22 * np.cos(t)
    target[4] = 0.72 + 0.16 * np.sin(t + np.pi / 2.0)
    target[5] = 0.10 * np.sin(t)
    target[6] = 0.015
    target[7] = -0.015
    return target


def create_target_marker(stage, prim_path, radius=0.018):
    sphere = UsdGeom.Sphere.Define(stage, prim_path)
    sphere.CreateRadiusAttr(radius)
    xform = UsdGeom.Xformable(sphere.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    return sphere.GetPrim()


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


def set_translate(prim, translate):
    tx, ty, tz = float(translate[0]), float(translate[1]), float(translate[2])
    xformable = UsdGeom.Xformable(prim)
    translate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if translate_ops:
        translate_ops[0].Set(Gf.Vec3d(tx, ty, tz))
    else:
        xformable.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))


def build_ik_window(initial_position):
    models = {
        "x": ui.SimpleFloatModel(float(initial_position[0])),
        "y": ui.SimpleFloatModel(float(initial_position[1])),
        "z": ui.SimpleFloatModel(float(initial_position[2])),
        "roll": ui.SimpleFloatModel(0.0),
        "pitch": ui.SimpleFloatModel(0.0),
        "yaw": ui.SimpleFloatModel(0.0),
    }
    status_model = ui.SimpleStringModel("idle")
    window = ui.Window("Piper Lula IK", width=380, height=240, visible=True)
    window.position_x = 40
    window.position_y = 80
    with window.frame:
        with ui.VStack(spacing=8, height=0):
            ui.Label("World-Space IK Target")
            for key, label, min_v, max_v in (
                ("x", "x", -0.80, 0.80),
                ("y", "y", -0.20, 0.80),
                ("z", "z", 0.05, 1.20),
            ):
                with ui.HStack(height=24):
                    ui.Label(label, width=32)
                    ui.FloatDrag(model=models[key], min=min_v, max=max_v, width=70)
                    ui.FloatSlider(model=models[key], min=min_v, max=max_v)
            ui.Label("Orientation Offset (deg)")
            for key, label in (("roll", "R"), ("pitch", "P"), ("yaw", "Y")):
                with ui.HStack(height=24):
                    ui.Label(label, width=32)
                    ui.FloatDrag(model=models[key], min=-180.0, max=180.0, width=70)
                    ui.FloatSlider(model=models[key], min=-180.0, max=180.0)
            with ui.HStack(height=24):
                ui.Label("Status", width=50)
                ui.StringField(model=status_model, read_only=True)
    return window, models, status_model


def build_joint_window(joint_names, lower_limits, upper_limits, initial_positions):
    joint_models = {}
    window = ui.Window("Piper Joint Control", width=460, height=360, visible=True)
    window.position_x = 40
    window.position_y = 360
    with window.frame:
        with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED):
            with ui.VStack(spacing=8, height=0):
                ui.Label("Manual Joint Targets")
                for idx, joint_name in enumerate(joint_names):
                    model = ui.SimpleFloatModel(float(initial_positions[idx]))
                    joint_models[joint_name] = model
                    with ui.HStack(height=24):
                        ui.Label(joint_name, width=70)
                        ui.FloatDrag(model=model, min=float(lower_limits[idx]), max=float(upper_limits[idx]), width=80)
                        ui.FloatSlider(model=model, min=float(lower_limits[idx]), max=float(upper_limits[idx]))
    return window, joint_models


robot_usd = resolve_robot_usd(args.robot_usd)
lula_enabled = args.lula_list_frames or args.lula_ik_test or args.ik_ui
lula_yaml = resolve_required_file(args.lula_yaml, "Lula robot description YAML") if lula_enabled else None
lula_urdf = resolve_required_file(args.lula_urdf, "Lula URDF") if lula_enabled else None

world = World(stage_units_in_meters=1.0)
stage = world.stage
rng = np.random.default_rng(args.seed)
assets_root = resolve_asset_root(args.asset_root)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.Xform.Define(stage, "/World")
UsdGeom.Xform.Define(stage, "/World/Looks")
UsdGeom.Xform.Define(stage, "/World/Floor")
UsdGeom.Xform.Define(stage, "/World/Table")
UsdGeom.Xform.Define(stage, "/World/Bin")
UsdGeom.Xform.Define(stage, "/World/Lights")
UsdGeom.Xform.Define(stage, "/World/RobotMount")

bamboo_texture_path = os.path.join(REPO_ROOT, "materials", "nv_bamboo_desktop.jpg")
granite_texture_path = os.path.join(REPO_ROOT, "materials", "nv_granite_tile.jpg")
aluminum_anodized_dir = os.path.join(REPO_ROOT, "materials", "Aluminum_Anodized")
aluminum_anodized_basecolor = os.path.join(aluminum_anodized_dir, "Aluminum_Anodized_BaseColor.png")
aluminum_anodized_normal = os.path.join(aluminum_anodized_dir, "Aluminum_Anodized_Normal.png")
aluminum_anodized_orm = os.path.join(aluminum_anodized_dir, "Aluminum_Anodized_ORM.png")

floor_material_base = create_omnipbr_material(stage, "/World/Looks/FloorBase", color=(0.44, 0.46, 0.48), roughness=0.72)
floor_material = create_omnipbr_material(
    stage,
    "/World/Looks/Floor",
    color=(0.85, 0.85, 0.85),
    roughness=0.62,
    texture_path=granite_texture_path if os.path.exists(granite_texture_path) else None,
)
wood_material = create_omnipbr_material(
    stage,
    "/World/Looks/Wood",
    color=(0.72, 0.66, 0.54),
    roughness=0.38,
    texture_path=bamboo_texture_path if os.path.exists(bamboo_texture_path) else None,
)
wood_base_material = create_omnipbr_material(stage, "/World/Looks/WoodBase", color=(0.49, 0.35, 0.20), roughness=0.48)
leg_material = create_omnipbr_material(stage, "/World/Looks/Legs", color=(0.19, 0.12, 0.08), roughness=0.5)
bin_material = create_omnipbr_material(
    stage,
    "/World/Looks/Bin",
    color=(0.50, 0.50, 0.50),
    roughness=0.0,
    metallic=0.0,
    texture_path=aluminum_anodized_basecolor if os.path.exists(aluminum_anodized_basecolor) else None,
    normal_texture_path=aluminum_anodized_normal if os.path.exists(aluminum_anodized_normal) else None,
    orm_texture_path=aluminum_anodized_orm if os.path.exists(aluminum_anodized_orm) else None,
    enable_orm=os.path.exists(aluminum_anodized_orm),
    diffuse_tint=(0.32, 0.32, 0.34),
    albedo_brightness=0.35,
)

floor_base_prim = define_box(stage, "/World/Floor/Base", size=(6.8, 6.8, 0.02), translate=(0.0, 0.0, -0.01))
bind_material(floor_base_prim, floor_material_base)
floor_surface_prim = define_uv_plane(
    stage,
    "/World/Floor/Surface",
    size=(6.72, 6.72),
    translate=(0.0, 0.0, 0.001),
    uv_scale=(6.0, 6.0),
)
bind_material(floor_surface_prim, floor_material)
floor_geom_paths = [str(floor_base_prim.GetPath())]

table_top_z = 0.75
table_top_thickness = 0.06
table_surface_z = table_top_z + table_top_thickness * 0.5

top_prim = define_box(stage, "/World/Table/Top", size=(1.4, 0.8, 0.06), translate=(0.0, 0.0, table_top_z))
bind_material(top_prim, wood_base_material)
table_surface_prim = define_uv_plane(
    stage,
    "/World/Table/TopSurface",
    size=(1.36, 0.76),
    translate=(0.0, 0.0, table_surface_z + 0.001),
)
bind_material(table_surface_prim, wood_material)

leg_height = 0.72
leg_center_z = leg_height * 0.5
table_leg_paths = []
for name, position in {
    "LegFL": (0.62, 0.32, leg_center_z),
    "LegFR": (0.62, -0.32, leg_center_z),
    "LegBL": (-0.62, 0.32, leg_center_z),
    "LegBR": (-0.62, -0.32, leg_center_z),
}.items():
    leg_prim = define_box(stage, f"/World/Table/{name}", size=(0.06, 0.06, leg_height), translate=position)
    bind_material(leg_prim, leg_material)
    table_leg_paths.append(str(leg_prim.GetPath()))

table_half_depth = 0.4
bin_outer_width = 0.72
bin_outer_depth = 0.48
bin_height = 0.36
bin_wall = 0.025
bin_center_z = table_surface_z - bin_height * 0.5
bin_center_y = table_half_depth + bin_outer_depth * 0.5 - bin_wall * 0.35

bin_parts = {
    "Bottom": (
        (bin_outer_width, bin_outer_depth, bin_wall),
        (0.0, bin_center_y, table_surface_z - bin_height + bin_wall * 0.5),
    ),
    "WallLeft": (
        (bin_wall, bin_outer_depth, bin_height),
        (-bin_outer_width * 0.5 + bin_wall * 0.5, bin_center_y, bin_center_z),
    ),
    "WallRight": (
        (bin_wall, bin_outer_depth, bin_height),
        (bin_outer_width * 0.5 - bin_wall * 0.5, bin_center_y, bin_center_z),
    ),
    "WallInner": (
        (bin_outer_width - 2 * bin_wall, bin_wall, bin_height),
        (0.0, bin_center_y - bin_outer_depth * 0.5 + bin_wall * 0.5, bin_center_z),
    ),
    "WallOuter": (
        (bin_outer_width - 2 * bin_wall, bin_wall, bin_height),
        (0.0, bin_center_y + bin_outer_depth * 0.5 - bin_wall * 0.5, bin_center_z),
    ),
}
bin_geom_paths = []
for name, (size, position) in bin_parts.items():
    bin_prim = define_box(stage, f"/World/Bin/{name}", size=size, translate=position)
    bind_material(bin_prim, bin_material)
    add_labels(bin_prim, ["bin"])
    bin_geom_paths.append(str(bin_prim.GetPath()))

dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Lights/Dome"))
dome_light.CreateIntensityAttr(900.0)
dome_light.CreateColorAttr(Gf.Vec3f(0.92, 0.95, 1.0))

sun_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Lights/Sun"))
sun_light.CreateIntensityAttr(3200.0)
sun_light.CreateAngleAttr(0.6)
sun_light.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
UsdGeom.Xformable(sun_light.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(315.0, 0.0, 35.0))

table_fill = UsdLux.RectLight.Define(stage, Sdf.Path("/World/Lights/TableFill"))
table_fill.CreateIntensityAttr(1800.0)
table_fill.CreateWidthAttr(1.6)
table_fill.CreateHeightAttr(1.0)
table_fill.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))
UsdGeom.Xformable(table_fill.GetPrim()).AddTranslateOp().Set(Gf.Vec3f(0.0, 0.05, 1.9))
UsdGeom.Xformable(table_fill.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-90.0, 0.0, 0.0))

apply_static_collision(floor_geom_paths + [str(top_prim.GetPath())] + table_leg_paths + bin_geom_paths)

mount_prim = UsdGeom.Xform.Define(stage, "/World/RobotMount/PiperX")
mount_xform = UsdGeom.Xformable(mount_prim.GetPrim())
mount_xform.AddTranslateOp().Set(Gf.Vec3d(args.robot_x, args.robot_y, table_surface_z))
mount_xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, args.robot_yaw))

add_reference_to_stage(usd_path=robot_usd, prim_path="/World/RobotMount/PiperX/Robot")
deactivate_embedded_environment(stage, "/World/RobotMount/PiperX/Robot")
robot = SingleArticulation(prim_path="/World/RobotMount/PiperX/Robot", name="piper_x")
robot_camera_prim = create_robot_camera(stage, "/World/RobotMount/PiperX/Robot/piper_x_camera/camera_link/DebugCamera")
robot_viewport = None
robot_sensor_camera = None

set_camera_view(
    eye=[1.7, -1.6, 1.45],
    target=[0.0, 0.15, 0.88],
    camera_prim_path="/OmniverseKit_Persp",
)
if args.robot_camera_view:
    robot_viewport = create_viewport_for_camera(
        viewport_name="Piper D435 View",
        camera_prim_path=str(robot_camera_prim.GetPath()),
        width=640,
        height=480,
        position_x=980,
        position_y=60,
    )
    print(f"[PiperX] robot camera viewport: {robot_viewport.title}")

topdown_camera = None
clutter_objects = []
settled = False
settle_counter = 0
settle_counter_target = 30
capture_done = False
frame_count = 0
max_wait_frames = 120

world.reset()
robot.initialize()
robot_sensor_camera = Camera(
    prim_path=str(robot_camera_prim.GetPath()),
    frequency=30,
    resolution=(640, 480),
)
robot_sensor_camera.initialize()
print(f"[PiperX] robot sensor camera: {robot_sensor_camera.prim_path}")
print("[PiperX] robot sensor camera mode: 640x480 @ 30 Hz")

if args.object_source != "none":
    clutter_objects = spawn_objects(world, stage, rng, args.object_source, args.num_objects, assets_root, bin_center_y, table_surface_z)
    print(f"[Scene] spawned {len(clutter_objects)} {args.object_source} clutter objects")

if args.topdown_camera or args.save_camera_debug:
    camera_width = 1280
    camera_height = 720
    d455_fx = 640.0
    d455_fy = 640.0
    d455_cx = camera_width * 0.5
    d455_cy = camera_height * 0.5
    pixel_size_um = 3.0

    topdown_camera = Camera(
        prim_path="/World/TopDownRealsense",
        position=np.array([0.0, bin_center_y, 1.65]),
        frequency=30,
        resolution=(camera_width, camera_height),
        orientation=rot_utils.euler_angles_to_quats(np.array([0.0, 90.0, 90.0]), degrees=True),
    )
    topdown_camera.initialize()
    topdown_camera.attach_annotator("distance_to_image_plane")
    topdown_camera.add_instance_segmentation_to_frame()

    horizontal_aperture = pixel_size_um * camera_width * 1e-6
    focal_length = pixel_size_um * ((d455_fx + d455_fy) * 0.5) * 1e-6
    topdown_camera.set_focal_length(focal_length)
    topdown_camera.set_focus_distance(1.65)
    topdown_camera.set_lens_aperture(0.0)
    topdown_camera.set_horizontal_aperture(horizontal_aperture)
    topdown_camera.set_vertical_aperture(horizontal_aperture / (camera_width / camera_height))
    topdown_camera.set_clipping_range(0.05, 4.0)
    topdown_camera.set_opencv_pinhole_properties(
        cx=d455_cx,
        cy=d455_cy,
        fx=d455_fx,
        fy=d455_fy,
        pinhole=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    print("[TopDownCamera] initialized")

for _ in range(15):
    world.step(render=True)

joint_names = list(robot.dof_names)
print(f"[PiperX] USD: {robot_usd}")
print(f"[PiperX] articulation prim: /World/RobotMount/PiperX/Robot")
print(f"[PiperX] dof count: {robot.num_dof}")
print(f"[PiperX] joints: {joint_names}")
print(f"[PiperX] initial joint positions: {robot.get_joint_positions()}")
print(f"[PiperX] apply_action available: {hasattr(robot, 'apply_action')}")
articulation_controller = robot.get_articulation_controller()
print(f"[PiperX] articulation controller: {type(articulation_controller).__name__}")

dof_props = robot.dof_properties
for gripper_joint in ("joint7", "joint8"):
    idx = joint_names.index(gripper_joint)
    print(
        f"[PiperX] {gripper_joint} props: "
        f"lower={dof_props['lower'][idx]:.5f} upper={dof_props['upper'][idx]:.5f} "
        f"stiffness={dof_props['stiffness'][idx]:.3f} damping={dof_props['damping'][idx]:.3f} "
        f"maxEffort={dof_props['maxEffort'][idx]:.3f} maxVelocity={dof_props['maxVelocity'][idx]:.3f}"
    )

if args.joint_ui:
    joint_ui_window, joint_ui_models = build_joint_window(
        joint_names,
        dof_props["lower"],
        dof_props["upper"],
        robot.get_joint_positions(),
    )
    print("[PiperX] joint slider UI enabled")

motion_sequence = []
current_pose_index = -1
pose_hold_counter = 0
current_target = None
motion_frame = 0
lula_action = None
lula_solver = None
art_kinematics = None
ik_ui_window = None
ik_ui_models = None
ik_ui_status = None
ik_target_base_position = None
ik_target_base_orientation = None
ik_target_marker = None
joint_ui_window = locals().get("joint_ui_window", None)
joint_ui_models = locals().get("joint_ui_models", None)

if lula_yaml is not None:
    lula_solver = LulaKinematicsSolver(robot_description_path=lula_yaml, urdf_path=lula_urdf)
    robot_base_position = np.array([args.robot_x, args.robot_y, table_surface_z], dtype=np.float64)
    robot_base_orientation = euler_angles_to_quats(np.array([0.0, 0.0, args.robot_yaw]), degrees=True)
    lula_solver.set_robot_base_pose(robot_base_position, robot_base_orientation)
    print(f"[Lula] YAML: {lula_yaml}")
    print(f"[Lula] URDF: {lula_urdf}")
    print(f"[Lula] active joints: {lula_solver.get_joint_names()}")
    print(f"[Lula] base pose position: {robot_base_position}")
    print(f"[Lula] base pose orientation (wxyz): {robot_base_orientation}")

    if args.lula_list_frames:
        print(f"[Lula] frames: {lula_solver.get_all_frame_names()}")
        simulation_app.close()
        sys.exit(0)

    art_kinematics = ArticulationKinematicsSolver(robot, lula_solver, args.ee_frame)
    ee_position, ee_rotation = art_kinematics.compute_end_effector_pose()
    ik_target_base_position = np.array(ee_position, dtype=np.float32)
    ik_target_base_orientation = rot_matrices_to_quats(np.asarray(ee_rotation))
    print(f"[Lula] end effector frame: {args.ee_frame}")
    print(f"[Lula] current ee position: {ee_position}")

    if args.lula_ik_test:
        target_position = ee_position + np.array([args.ik_target_dx, args.ik_target_dy, args.ik_target_dz], dtype=np.float32)
        target_orientation = None if args.ik_position_only else ik_target_base_orientation
        lula_action, lula_success = art_kinematics.compute_inverse_kinematics(
            target_position=target_position, target_orientation=target_orientation
        )
        print(f"[Lula] target position: {target_position}")
        print(f"[Lula] target orientation mode: {'position_only' if target_orientation is None else 'preserve_current'}")
        print(f"[Lula] IK success: {lula_success}")
        if lula_success:
            print(f"[Lula] target joint positions: {lula_action.joint_positions}")
            motion_sequence = []
        else:
            lula_action = None

    if args.ik_ui:
        ik_target_marker = create_target_marker(stage, "/World/Debug/IkTarget")
        target_material = create_omnipbr_material(stage, "/World/Looks/IkTarget", color=(0.82, 0.16, 0.12), roughness=0.25)
        bind_material(ik_target_marker, target_material)
        initial_ui_target = ik_target_base_position + np.array(
            [args.ik_target_dx, args.ik_target_dy, args.ik_target_dz], dtype=np.float32
        )
        ik_ui_window, ik_ui_models, ik_ui_status = build_ik_window(initial_ui_target)
        print("[Lula] IK slider UI enabled")

if args.test_motion:
    motion_sequence = get_test_poses(robot.num_dof)
    print("[PiperX] test motion enabled")
elif args.circle_motion:
    print("[PiperX] smooth circle-like motion enabled")
elif args.lula_ik_test:
    print("[Lula] IK motion enabled")

while simulation_app.is_running():
    if robot_viewport is not None:
        robot_viewport.viewport_api.camera_path = str(robot_camera_prim.GetPath())

    if clutter_objects and topdown_camera is not None and args.save_camera_debug and not capture_done:
        frame_count += 1
        if objects_are_settled(clutter_objects):
            settle_counter += 1
        else:
            settle_counter = 0
        settled = settle_counter >= settle_counter_target

        if frame_count % 60 == 0:
            print(
                f"[TopDownCamera] waiting for clutter to settle "
                f"(frame={frame_count}, stable_frames={settle_counter}/{settle_counter_target})"
            )

        if settled or frame_count >= max_wait_frames:
            if not settled:
                print("[TopDownCamera] settle timeout reached, capturing current frame")
            output_dir = os.path.join(ISAAC_ROOT, "camera_debug")
            save_camera_debug(topdown_camera, output_dir)
            build_custom_instance_segmentation(topdown_camera, output_dir, clutter_objects)
            capture_done = True

    if args.ik_ui and art_kinematics is not None:
        target_position = np.array(
            [
                ik_ui_models["x"].get_value_as_float(),
                ik_ui_models["y"].get_value_as_float(),
                ik_ui_models["z"].get_value_as_float(),
            ],
            dtype=np.float32,
        )
        orientation_offset = euler_angles_to_quats(
            np.array(
                [
                    ik_ui_models["roll"].get_value_as_float(),
                    ik_ui_models["pitch"].get_value_as_float(),
                    ik_ui_models["yaw"].get_value_as_float(),
                ],
                dtype=np.float64,
            ),
            degrees=True,
        )
        target_orientation_quat = Gf.Quatf(float(ik_target_base_orientation[0]), Gf.Vec3f(*ik_target_base_orientation[1:])) * Gf.Quatf(
            float(orientation_offset[0]), Gf.Vec3f(*orientation_offset[1:])
        )
        target_orientation = np.array(
            [
                float(target_orientation_quat.GetReal()),
                float(target_orientation_quat.GetImaginary()[0]),
                float(target_orientation_quat.GetImaginary()[1]),
                float(target_orientation_quat.GetImaginary()[2]),
            ],
            dtype=np.float64,
        )
        set_translate(ik_target_marker, target_position)
        lula_action, lula_success = art_kinematics.compute_inverse_kinematics(
            target_position=target_position, target_orientation=target_orientation
        )
        ik_ui_status.set_value(
            f"{'OK' if lula_success else 'FAIL'}  pos=({target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f})"
        )
        if lula_success:
            articulation_controller.apply_action(lula_action)
    elif lula_action is not None:
        articulation_controller.apply_action(lula_action)
    elif args.circle_motion:
        current_target = get_circle_motion_target(motion_frame, robot.num_dof)
        robot.apply_action(ArticulationActions(joint_positions=current_target))
        motion_frame += 1
    elif motion_sequence:
        if pose_hold_counter <= 0:
            current_pose_index += 1
            if current_pose_index < len(motion_sequence):
                pose_name, pose = motion_sequence[current_pose_index]
                current_target = pose[0].copy()
                robot.apply_action(ArticulationActions(joint_positions=current_target))
                print(f"[PiperX] commanded pose '{pose_name}': {pose.tolist()[0]}")
                pose_hold_counter = args.hold_frames
            else:
                motion_sequence = []
                current_positions = robot.get_joint_positions()
                print(f"[PiperX] motion sequence complete, current joints: {current_positions}")
        else:
            if current_target is not None:
                robot.apply_action(ArticulationActions(joint_positions=current_target))
            pose_hold_counter -= 1
    elif joint_ui_models is not None:
        joint_target = np.array([joint_ui_models[name].get_value_as_float() for name in joint_names], dtype=np.float32)
        articulation_controller.apply_action(ArticulationAction(joint_positions=joint_target))
    world.step(render=True)

simulation_app.close()
