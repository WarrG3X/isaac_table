from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import os
import sys

import numpy as np
import omni.kit.commands
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, GeometryPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
ISAAC_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", "..", ".."))
DEFAULT_PIPER_USD = os.path.join(ISAAC_ROOT, "piper_isaac_sim", "USD", "piper_x_v1.usd")

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


robot_usd = resolve_robot_usd(args.robot_usd)

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage = world.stage
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.Xform.Define(stage, "/World")
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

apply_static_collision([str(floor_base_prim.GetPath()), str(top_prim.GetPath()), *table_leg_paths, *bin_geom_paths])

mount_prim = UsdGeom.Xform.Define(stage, "/World/RobotMount/PiperX")
mount_xform = UsdGeom.Xformable(mount_prim.GetPrim())
mount_xform.AddTranslateOp().Set(Gf.Vec3d(args.robot_x, args.robot_y, table_surface_z))
mount_xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, args.robot_yaw))

add_reference_to_stage(usd_path=robot_usd, prim_path="/World/RobotMount/PiperX/Robot")
deactivate_embedded_environment(stage, "/World/RobotMount/PiperX/Robot")
robot = Articulation(prim_paths_expr="/World/RobotMount/PiperX/Robot", name="piper_x")

set_camera_view(
    eye=[1.7, -1.6, 1.45],
    target=[0.0, 0.15, 0.88],
    camera_prim_path="/OmniverseKit_Persp",
)

world.reset()

joint_names = list(robot.dof_names)
print(f"[PiperX] USD: {robot_usd}")
print(f"[PiperX] articulation prim: /World/RobotMount/PiperX/Robot")
print(f"[PiperX] dof count: {robot.num_dof}")
print(f"[PiperX] joints: {joint_names}")
print(f"[PiperX] initial joint positions: {robot.get_joint_positions()}")

motion_sequence = []
current_pose_index = -1
pose_hold_counter = 0

if args.test_motion:
    motion_sequence = get_test_poses(robot.num_dof)
    print("[PiperX] test motion enabled")

while simulation_app.is_running():
    if motion_sequence:
        if pose_hold_counter <= 0:
            current_pose_index += 1
            if current_pose_index < len(motion_sequence):
                pose_name, pose = motion_sequence[current_pose_index]
                robot.set_joint_positions(pose)
                print(f"[PiperX] commanded pose '{pose_name}': {pose.tolist()[0]}")
                pose_hold_counter = args.hold_frames
            else:
                motion_sequence = []
                current_positions = robot.get_joint_positions()
                print(f"[PiperX] motion sequence complete, current joints: {current_positions}")
        else:
            pose_hold_counter -= 1
    world.step(render=True)

simulation_app.close()
